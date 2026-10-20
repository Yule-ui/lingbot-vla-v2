"""UR3-only runtime adapters; the upstream trainer and dataset remain unchanged."""
import functools
import json
from pathlib import Path


class PreflightComplete(Exception):
    """Stop after checking distributed weights, without building the optimizer."""


def install_data_adapters(manifest_path, split):
    import torch
    from lingbotvla.data.vla_data import base_dataset
    from lingbotvla.data.vla_data.utils import FeatureTransform

    manifest = json.loads(Path(manifest_path).read_text())
    dataset_root = Path(manifest['dataset_root']).resolve()
    selected = manifest['splits'][split]
    reference = torch.tensor(manifest['state_quaternion_reference'], dtype=torch.float64)
    original_dataset_init = base_dataset.LeRobotDataset.__init__
    original_query_indices = base_dataset.LeRobotDataset._get_query_indices
    original_transform_init = FeatureTransform.__init__
    original_apply = FeatureTransform.apply

    @functools.wraps(original_dataset_init)
    def dataset_init(self, repo_id, *args, **kwargs):
        path = Path(kwargs.get('root') or repo_id).expanduser()
        is_ur3 = path.exists() and path.resolve() == dataset_root
        if is_ur3:
            requested = kwargs.get('episodes')
            if requested is not None and not set(requested).issubset(selected):
                raise ValueError(f'Requested episodes are outside the {split} split')
            kwargs['episodes'] = selected if requested is None else requested
            kwargs['root'] = dataset_root
            repo_id = dataset_root.name
        original_dataset_init(self, repo_id, *args, **kwargs)
        if is_ur3:
            loaded = set(self.hf_dataset.unique('episode_index'))
            if loaded != set(kwargs['episodes']):
                raise RuntimeError(f'Episode filter did not take effect: {loaded}')
            print(f'UR3 split={split}: {len(loaded)} episodes, {len(self)} frames', flush=True)

    @functools.wraps(original_query_indices)
    def query_indices(self, idx, ep_idx):
        if Path(self.root).resolve() == dataset_root and self.episodes is not None:
            # The v0.4.2 helper clamps against original episode indices.
            idx = int(self.hf_dataset[int(idx)]['index'].item())
        return original_query_indices(self, idx, ep_idx)

    @functools.wraps(original_transform_init)
    def transform_init(self, robot_config_path, *args, **kwargs):
        original_transform_init(self, robot_config_path, *args, **kwargs)
        self._ur3_reference = None
        if Path(robot_config_path).stem == 'ur3_robotiq_eef_relative':
            self._ur3_reference = reference
            if self.normalizer is not None:
                norm = json.loads(Path(manifest['norm_stats_file']).read_text())
                if norm.get('ur3_preprocessing', {}).get('state_quaternion_reference') != manifest['state_quaternion_reference']:
                    raise ValueError('Normalization statistics and quaternion preprocessing do not match')

    @functools.wraps(original_apply)
    def apply(self, item, *args, **kwargs):
        if getattr(self, '_ur3_reference', None) is not None:
            item = dict(item)
            state = torch.as_tensor(item['observation.state']).clone()
            quaternion = state[..., 12:16]
            lengths = quaternion.norm(dim=-1, keepdim=True)
            if not torch.isfinite(quaternion).all() or (lengths < 1e-8).any():
                raise ValueError('Invalid UR3 observation quaternion')
            quaternion = quaternion / lengths
            ref = self._ur3_reference.to(quaternion)
            sign = torch.where((quaternion * ref).sum(-1, keepdim=True) < 0, -1., 1.)
            state[..., 12:16] = quaternion * sign
            item['observation.state'] = state
        result = original_apply(self, item, *args, **kwargs)
        if (getattr(self, '_ur3_reference', None) is not None
                and self.return_item_befor_padding and self.normalizer is None):
            # Preserve small quaternion variances in RunningStats' E[x²]-E[x]².
            for key in self.states + self.actions:
                result[key] = result[key].to(torch.float64)
        return result

    base_dataset.LeRobotDataset.__init__ = dataset_init
    base_dataset.LeRobotDataset._get_query_indices = query_indices
    FeatureTransform.__init__ = transform_init
    FeatureTransform.apply = apply
    return manifest


def install_training_adapter(trainer, preflight_only=False, vlm_train_last_layers=0):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open
    from lingbotvla.distributed import torch_parallelize

    original_parse = trainer.parse_args
    original_parallelize = trainer.build_parallelize_model
    original_load = torch_parallelize.load_model_weights
    captured = {}
    if not 0 <= vlm_train_last_layers <= 36:
        raise ValueError('vlm_train_last_layers must be between 0 and 36')

    @functools.wraps(original_parse)
    def parse(*args, **kwargs):
        result = original_parse(*args, **kwargs)
        if result.train.init_device != 'meta' or not result.model.post_training:
            raise ValueError('UR3 wrapper requires meta initialization and post_training=true')
        if result.model.vlm_repo_id:
            raise ValueError('Load the complete LingBot checkpoint, not separate VLM weights')
        if vlm_train_last_layers:
            result.train.train_expert_only = False
            result.train.freeze_vision_encoder = True
            result.train.use_moe_expert_lr = False
        captured['args'] = result
        return result

    @functools.wraps(original_load)
    def strict_load(model, weights_path, *args, **kwargs):
        config = captured['args']
        kwargs.update(post_training=True, adanorm_time=config.model.adanorm_time,
                      incremental_training=False, load_vlm_only=False)
        return original_load(model, weights_path, *args, **kwargs)

    @functools.wraps(original_parallelize)
    def parallelize(model, *args, **kwargs):
        config = captured['args']
        qwen = model.model.qwenvl_with_expert.qwenvl
        if vlm_train_last_layers:
            qwen.requires_grad_(False)
            for layer in qwen.model.language_model.layers[-vlm_train_last_layers:]:
                layer.requires_grad_(True)
            model.config.ur3_vlm_train_last_layers = vlm_train_last_layers
        frozen = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        kwargs['weights_path'] = config.model.model_path
        model = original_parallelize(model, *args, **kwargs)
        for name, parameter in model.named_parameters():
            if parameter.is_meta or parameter.requires_grad != frozen[name]:
                raise RuntimeError(f'Invalid materialization/freeze state: {name}')
        for name, buffer in model.named_buffers():
            if buffer.is_meta:
                raise RuntimeError(f'Unmaterialized buffer: {name}')
        index = json.loads((Path(config.model.model_path) / 'model.safetensors.index.json').read_text())['weight_map']
        parameters = dict(model.named_parameters())
        checked = []
        names = [
            'model.state_proj.weight', 'model.action_in_proj.weight', 'model.action_out_proj.weight',
            'model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.gate_proj',
            'model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.q_proj.weight',
        ]
        for name in names:
            parameter = parameters[name]
            with safe_open(str(Path(config.model.model_path) / index[name]), framework='pt', device='cpu') as file:
                expected = file.get_tensor(name)
            if hasattr(parameter, 'to_local'):
                local = parameter.to_local().detach().cpu()
                coordinate = parameter.device_mesh.get_coordinate()
                for dimension, placement in enumerate(parameter.placements):
                    if placement.is_shard():
                        expected = torch.tensor_split(expected, parameter.device_mesh.size(dimension), dim=placement.dim)[coordinate[dimension]]
            else:
                local = parameter.detach().cpu()
            torch.testing.assert_close(local, expected.to(local.dtype), rtol=0, atol=0)
            checked.append(name)
        qwen = model.model.qwenvl_with_expert.qwenvl
        first_trainable_layer = len(qwen.model.language_model.layers) - vlm_train_last_layers
        for name, parameter in qwen.named_parameters():
            pieces = name.split('.')
            expected_trainable = (
                vlm_train_last_layers > 0
                and name.startswith('model.language_model.layers.')
                and int(pieces[3]) >= first_trainable_layer
            )
            if parameter.requires_grad != expected_trainable:
                raise RuntimeError(f'Unexpected Qwen freeze state: {name}')
        assert model.model.state_proj.weight.requires_grad
        audit = {'rank': dist.get_rank(), 'checkpoint': config.model.model_path,
                 'sampled_parameters_match': checked, 'qwen_frozen': vlm_train_last_layers == 0,
                 'vlm_train_last_layers': vlm_train_last_layers,
                 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3}
        output = Path(config.train.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f'weight_load_rank{dist.get_rank()}.json').write_text(json.dumps(audit, indent=2))
        dist.barrier()
        print('UR3 strict pretrained-weight audit passed:', audit, flush=True)
        if preflight_only:
            raise PreflightComplete()
        return model

    trainer.parse_args = parse
    trainer.build_parallelize_model = parallelize
    torch_parallelize.load_model_weights = strict_load
    if vlm_train_last_layers:
        def partial_vlm_param_groups(model, args_train):
            # Explicit LRs; do not apply the routed-expert sqrt(32/4) multiplier.
            groups = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if '.qwenvl.model.language_model.' in name:
                    lr = 1e-6
                elif '.qwen_expert.' in name:
                    lr = 1e-5
                else:
                    lr = args_train.lr
                decay = 0.0 if parameter.ndim < 2 or name.endswith('.bias') else args_train.weight_decay
                groups.setdefault((lr, decay), []).append(parameter)
            return [{'params': parameters, 'lr': lr, 'weight_decay': decay}
                    for (lr, decay), parameters in groups.items()]
        trainer.get_moe_param_groups = partial_vlm_param_groups
    return captured
