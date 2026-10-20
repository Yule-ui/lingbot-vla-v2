"""Measure the native UR3 training loop; stop before checkpoint serialization.

Run under torchrun. Uses real dataset frames, all configured auxiliary teachers,
gradient accumulation, backward and native AdamW. It never exports a checkpoint.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MeasurementComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vla/ur3_20261016/data_check.yaml')
    parser.add_argument('--manifest', default='assets/training_data/ur3_20261016_split_seed42.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--vlm-train-last-layers', type=int, default=2)
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--active-camera-views-only', action='store_true',
                        help='Use the two real UR3 cameras without a padded third slot')
    parser.add_argument('--compile', action='store_true')
    options = parser.parse_args()
    if options.steps < 3:
        parser.error('At least 3 optimizer steps are needed to measure Adam state and subsequent iterations')
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['RANK'])
    report_path = output / f'memory_rank{rank}.json'
    report = {'status': 'starting', 'rank': rank, 'world_size': int(os.environ['WORLD_SIZE']),
              'options': vars(options), 'checkpoint_serialization_measured': False,
              'optimizer_steps': [], 'inputs': [], 'last_phase': 'imports'}

    def save():
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(report_path)

    save()
    import torch
    import torch.distributed as dist
    import tasks.vla.train_lingbotvla as trainer
    import run_ur3

    def memory():
        result = {'allocated_GiB': torch.cuda.memory_allocated() / 1024**3,
                'reserved_GiB': torch.cuda.memory_reserved() / 1024**3,
                'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 1024**3}
        cgroup = Path('/sys/fs/cgroup/memory.current')
        if cgroup.exists():
            result['container_memory_current_GiB'] = int(cgroup.read_text()) / 1024**3
        return result

    original_parse = trainer.parse_args
    original_optimizer = trainer.build_optimizer
    original_checkpointer = trainer.build_checkpointer

    @functools.wraps(original_parse)
    def parse(*args, **kwargs):
        result = original_parse(*args, **kwargs)
        result.data.image_height = options.height
        result.data.image_width = options.width
        result.data.num_workers = 2
        if options.active_camera_views_only:
            result.data.cameras = ['camera_top', 'camera_wrist_left']
        result.train.output_dir = str(output)
        result.train.save_checkpoint_path = str(output / 'checkpoints')
        result.train.model_assets_dir = str(output / 'model_assets')
        result.train.max_steps = options.steps
        result.train.save_steps = 0
        result.train.save_epochs = 0
        result.train.save_hf_weights = False
        result.train.enable_resume = False
        result.train.load_checkpoint_path = None
        result.train.use_compile = options.compile
        result.train.enable_activation_offload = options.activation_offload
        if options.activation_offload:
            result.train.activation_gpu_limit = 2.0
        result.train.enable_gradient_checkpointing = True
        result.train.enable_mixed_precision = True
        result.train.enable_fp32 = False
        result.train.micro_batch_size = 1
        result.train.gradient_accumulation_steps = 4
        result.train.global_batch_size = 4 * int(os.environ['WORLD_SIZE'])
        result.train.lr_warmup_ratio = 1 / options.steps
        result.train.align_params['visual_steps'] = 1000000
        report['status'] = 'running'
        report['configuration'] = {
            'micro_batch_size': 1, 'gradient_accumulation_steps': 4,
            'cameras': result.data.cameras,
            'cuda_allocator_configuration': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
            'global_batch_size': result.train.global_batch_size,
            'gradient_checkpointing': True, 'data_parallel_mode': result.train.data_parallel_mode,
            'full_shard': result.train.enable_full_shard, 'bf16': True,
            'activation_gpu_limit': result.train.activation_gpu_limit,
            'action_chunk': result.train.chunk_size,
            'depth_and_video_auxiliaries': result.train.align_params,
            'learning_rates': {'vlm': 1e-6 if options.vlm_train_last_layers else None,
                              'action_expert': 1e-5 if options.vlm_train_last_layers else 'native MoE groups',
                              'projection_auxiliary': result.train.lr},
            'weight_decay': result.train.weight_decay,
        }
        save()
        return result

    def build_optimizer(model, *args, **kwargs):
        optimizer = original_optimizer(model, *args, **kwargs)
        base_model = getattr(model, '_orig_mod', model)
        parameters = dict(base_model.named_parameters())
        sample_names = ['model.action_out_proj.weight', 'model.state_proj.weight',
                        'model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.gate_proj']
        if options.vlm_train_last_layers:
            for layer in sorted({36 - options.vlm_train_last_layers, 35}):
                sample_names.append(f'model.qwenvl_with_expert.qwenvl.model.language_model.layers.{layer}.self_attn.k_proj.weight')

        def local(tensor):
            return tensor.to_local() if hasattr(tensor, 'to_local') else tensor

        def sample(parameter):
            flat = local(parameter).detach().reshape(-1)
            return flat[::max(1, flat.numel() // 128)][:128].float().cpu().clone()

        report['model_loaded_memory'] = memory()
        report['trainable_parameters'] = sum(p.numel() for p in parameters.values() if p.requires_grad)
        report['gpu_name'] = torch.cuda.get_device_name()
        report['gpu_total_GiB'] = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
        report['sampled_trainable_parameters'] = sample_names
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        previous_step = time.monotonic()
        step_count = 0

        def before_forward(module, args, kwargs):
            report['last_phase'] = 'forward_or_backward'
            if len(report['inputs']) < 4:
                grid = kwargs['image_grid_thw'].detach().cpu()
                merged = grid.prod(dim=-1) // 4
                expected = (options.height // 32) * (options.width // 32)
                if not (merged == expected).all():
                    raise RuntimeError(f'Unexpected image grid: {grid.tolist()}')
                report['inputs'].append({'image_grid_thw': grid.tolist(),
                                         'images_shape': list(kwargs['images'].shape),
                                         'img_masks': kwargs['img_masks'].detach().cpu().tolist(),
                                         'depth_targets_shape': list(kwargs['depth_targets'].shape),
                                         'future_depth_targets_shape': list(kwargs['future_depth_targets'].shape),
                                         'future_video_targets_shape': list(kwargs['future_video_targets'].shape),
                                         'visual_tokens_per_slot': merged.tolist()})
                save()

        def after_forward(module, args, result):
            if not torch.isfinite(result[0].detach()).all():
                raise RuntimeError('Non-finite training loss')
            report['last_loss'] = float(result[0].detach())

        model.register_forward_pre_hook(before_forward, with_kwargs=True)
        model.register_forward_hook(after_forward)
        original_step = optimizer.step

        @functools.wraps(original_step)
        def step(self, *step_args, **step_kwargs):
            nonlocal previous_step, step_count
            report['last_phase'] = 'optimizer_step'
            gradients = {}
            before = {}
            for name in sample_names:
                parameter = parameters[name]
                before[name] = sample(parameter)
                grad = parameter.grad
                if grad is None:
                    raise RuntimeError(f'Trainable sample has no gradient: {name}')
                grad = local(grad)
                finite = bool(torch.isfinite(grad).all())
                gradients[name] = {'finite': finite, 'local_norm': float(grad.float().norm())}
                if not finite:
                    raise RuntimeError(f'Non-finite gradient: {name}')
            before_update = memory()
            result = original_step(*step_args, **step_kwargs)
            torch.cuda.synchronize()
            changed = {name: float((sample(parameters[name]) - before[name]).abs().max())
                       for name in sample_names}
            step_count += 1
            now = time.monotonic()
            report['optimizer_steps'].append({'step': step_count,
                'seconds_since_previous_update': now - previous_step,
                'loss_last_microbatch': report.get('last_loss'),
                'before_update_memory': before_update, 'after_update_memory': memory(),
                'sampled_gradients': gradients, 'sampled_max_weight_changes': changed})
            previous_step = now
            report['last_phase'] = 'optimizer_update_complete'
            save()
            return result

        optimizer.step = types.MethodType(step, optimizer)
        save()
        return optimizer

    def build_checkpointer(*args, **kwargs):
        native = original_checkpointer(*args, **kwargs)

        def stop_before_checkpoint(*save_args, **save_kwargs):
            if len(report['optimizer_steps']) != options.steps:
                raise RuntimeError('Unexpected early checkpoint during memory measurement')
            report['status'] = 'completed'
            report['last_phase'] = 'all_optimizer_steps_completed_before_checkpoint'
            report['training_peak_memory'] = memory()
            report['sampled_parameters_updated_locally'] = {
                name: any(step['sampled_max_weight_changes'][name] > 0 for step in report['optimizer_steps'])
                for name in report['sampled_trainable_parameters']}
            # Some output rows belong entirely to masked robot dimensions on a
            # rank. Require updates globally, not in every local FSDP shard.
            updated = torch.tensor(list(report['sampled_parameters_updated_locally'].values()),
                                   dtype=torch.int32, device='cuda')
            dist.all_reduce(updated, op=dist.ReduceOp.MAX)
            report['sampled_parameters_updated'] = dict(zip(
                report['sampled_trainable_parameters'], map(bool, updated.cpu().tolist())))
            if not all(report['sampled_parameters_updated'].values()):
                raise RuntimeError('A sampled trainable parameter never changed')
            save()
            raise MeasurementComplete()

        return types.SimpleNamespace(save=stop_before_checkpoint, load=native.load)

    trainer.parse_args = parse
    trainer.build_optimizer = build_optimizer
    trainer.build_checkpointer = build_checkpointer
    sys.argv = [str(ROOT / 'tools/run_ur3.py'), '--stage', 'train',
                '--manifest', options.manifest, '--vlm-train-last-layers',
                str(options.vlm_train_last_layers), options.config]
    try:
        run_ur3.main()
        raise RuntimeError('Native trainer returned without the expected measurement boundary')
    except MeasurementComplete:
        print(f'UR3 memory measurement completed on rank {rank}: {report_path}', flush=True)
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as error:
        report['status'] = 'out_of_memory' if isinstance(error, torch.OutOfMemoryError) else 'failed'
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        if torch.cuda.is_initialized():
            report['failure_memory'] = memory()
        save()
        raise


if __name__ == '__main__':
    main()
