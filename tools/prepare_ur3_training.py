"""Create a reproducible episode split and UR3 configs without changing the dataset."""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='/data-teamA/datasets/ur3_eef_relative')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    table = pa.concat_tables([
        pq.read_table(path, columns=['index', 'episode_index', 'task_index', 'observation.state'])
        for path in sorted((dataset / 'data').rglob('*.parquet'))
    ]).sort_by([('index', 'ascending')]).to_pydict()
    episodes = np.asarray(table['episode_index'])
    tasks = np.asarray(table['task_index'])
    state = np.asarray(table['observation.state'], dtype=np.float64)
    by_task = {}
    for episode in sorted(np.unique(episodes).tolist()):
        task = np.unique(tasks[episodes == episode])
        if len(task) != 1:
            raise ValueError(f'Episode {episode} contains multiple tasks: {task}')
        by_task.setdefault(int(task[0]), []).append(int(episode))
    if len(by_task) != 3 or any(len(group) != 30 for group in by_task.values()):
        raise ValueError(f'Expected 3 tasks with 30 episodes each: {by_task}')
    rng = random.Random(args.seed)
    train, validation = [], []
    for task, group in sorted(by_task.items()):
        group = list(group)
        rng.shuffle(group)
        validation.extend(group[:3])
        train.extend(group[3:])
    train, validation = sorted(train), sorted(validation)
    assert len(train) == 81 and len(validation) == 9 and not set(train) & set(validation)
    selected = np.isin(episodes, train)
    reference = state[selected][0, 12:16]
    reference = reference / np.linalg.norm(reference)
    quaternions = state[:, 12:16]
    min_dot = float(np.abs(quaternions @ reference).min())
    if min_dot < 0.1:
        raise ValueError('Dataset crosses the reference hemisphere boundary; inspect quaternion preprocessing.')
    manifest_path = ROOT / f'assets/training_data/ur3_split_seed{args.seed}.json'
    norm_path = ROOT / 'assets/norm_stats/ur3_robotiq_eef_relative.json'
    manifest = {
        'version': 1, 'dataset_root': str(dataset), 'seed': args.seed,
        'splits': {'train': train, 'validation': validation},
        'frames': {'train': int(selected.sum()), 'validation': int((~selected).sum())},
        'state_quaternion_reference': reference.tolist(),
        'min_abs_reference_dot': min_dot,
        'norm_stats_file': str(norm_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f'Refusing to replace a different split: {manifest_path}')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    mapping = ROOT / 'configs/robot_configs/ur3_robotiq_eef_relative.yaml'
    if not mapping.exists():
        mapping.write_text((ROOT / 'lingbot_vla_v2_eef_relative.yaml').read_text())
    robot = yaml.safe_load(mapping.read_text())
    assert all(not next(iter(action.values()))['subtract_state'] for action in robot['actions'])

    config = yaml.safe_load((ROOT / 'configs/vla/real_robot/real_robot.yaml').read_text())
    config['model'].update(
        model_path='/data-teamA/models/lingbot-vla-v2-6b',
        tokenizer_path='/data-teamA/models/Qwen3-VL-4B-Instruct',
    )
    config['data'].update(
        data_name='ur3_robotiq_eef_relative', train_path=str(dataset),
        robot_config_root=str(ROOT / 'configs/robot_configs'),
        norm_stats_file=str(norm_path), num_workers=2,
    )
    config['train'].update(
        output_dir=f'/data-teamA/outputs/ur3/train_seed{args.seed}',
        init_device='meta', train_expert_only=True, train_state_proj=True,
        enable_full_shard=True, enable_gradient_checkpointing=True,
        enable_mixed_precision=True, enable_fp32=False,
        micro_batch_size=1, gradient_accumulation_steps=4, global_batch_size=16,
        optimizer='adamw', lr=2e-5, lr_warmup_ratio=0.03, chunk_size=50,
        max_steps=5000, save_steps=2500, use_compile=False,
        enable_resume=False, save_hf_weights=True, async_save_hf_weights=False,
    )
    config['train']['align_params']['depth'].update(
        moge_path='/data-teamA/models/MoGe-2-vitb-normal/model.pt',
        morgbd_path='/data-teamA/models/lingbot-vla-v2-6b/depth/model.pt',
    )
    config['train']['align_params']['video'].update(
        ckpt_path='/data-teamA/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth',
        config_path='/data-teamA/models/lingbot-vla-v2-6b/dino_video/config.yaml',
    )
    norm_config = {
        'data': {
            key: value for key, value in config['data'].items()
            if key not in ('norm_stats_file', 'norm_type', 'use_future_image')
        },
        # Actions are already per-step deltas: use each training frame once.
        'train': {'micro_batch_size': 128, 'chunk_size': 1, 'output_dir': '', 'max_steps': 1},
    }
    norm_config['data'].update(norm_path=str(norm_path), num_workers=2)
    smoke = copy.deepcopy(config)
    smoke['train'].update(
        output_dir=f'/data-teamA/outputs/ur3/smoke_seed{args.seed}',
        max_steps=50, save_steps=50, save_hf_weights=False,
    )
    directory = ROOT / 'configs/vla/ur3'
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in [('train', config), ('smoke', smoke), ('norm', norm_config)]:
        (directory / f'{name}.yaml').write_text(yaml.safe_dump(content, sort_keys=False))
    print(json.dumps(manifest, indent=2))
    print('Manifest:', manifest_path)
    print('Configs:', directory)


if __name__ == '__main__':
    main()
