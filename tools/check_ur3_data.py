"""Check episode isolation, action chunks, statistics and model feature round trips."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ur3_adapters import install_data_adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'validation'], default='train')
    parser.add_argument('--manifest', default=str(ROOT / 'assets/training_data/ur3_split_seed42.json'))
    parser.add_argument('--config', default=str(ROOT / 'configs/vla/ur3/smoke.yaml'))
    options = parser.parse_args()
    manifest = install_data_adapters(options.manifest, options.split)
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset, VLADataset
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    root = Path(manifest['dataset_root'])
    raw = pa.concat_tables([
        pq.read_table(path, columns=['index', 'episode_index', 'action', 'observation.state'])
        for path in sorted((root / 'data').rglob('*.parquet'))
    ]).sort_by([('index', 'ascending')]).to_pydict()
    episodes = np.asarray(raw['episode_index'])
    action = np.asarray(raw['action'], dtype=np.float64)
    state = np.asarray(raw['observation.state'], dtype=np.float64)
    selected = np.isin(episodes, manifest['splits']['train'])
    dataset = LeRobotDataset(str(root), load_image=False,
                            delta_timestamps={'action': [i / 25 for i in range(50)]})
    checked = 0
    for episode in manifest['splits'][options.split]:
        indices = np.flatnonzero(episodes == episode)
        for absolute in [int(indices[0]), int(indices[len(indices) // 2]), int(indices[-1])]:
            relative = dataset._absolute_to_relative_idx[absolute]
            sample = dataset[relative]
            expected_indices = np.minimum(np.arange(absolute, absolute + 50), indices[-1])
            np.testing.assert_array_equal(sample['action'].numpy(), action[expected_indices])
            np.testing.assert_array_equal(sample['action_is_pad'].numpy(), np.arange(absolute, absolute + 50) > indices[-1])
            assert sample['episode_index'].item() == episode
            checked += 1
    quaternion = state[:, 12:16]
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion[(quaternion @ np.asarray(manifest['state_quaternion_reference'])) < 0] *= -1
    state[:, 12:16] = quaternion
    expected = {
        'action.end.position': action[selected, :7],
        'action.effector.position': action[selected, 7:8],
        'observation.state.arm.position': state[selected, :6],
        'observation.state.end.position': np.concatenate([state[selected, 6:9], state[selected, 12:16]], axis=-1),
        'observation.state.effector.position': state[selected, 16:17],
    }
    norm = json.loads(Path(manifest['norm_stats_file']).read_text())
    assert norm['count'] == int(selected.sum()) == manifest['frames']['train']
    for key, values in expected.items():
        saved = norm['norm_stats'][key]
        np.testing.assert_allclose(saved['mean'], values.mean(0), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(saved['std'], values.std(0), rtol=1e-5, atol=1e-7)
        assert np.isfinite(saved['std']).all()
    config = yaml.safe_load(Path(options.config).read_text())
    data_config = dict(config['data'])
    for key in ['joints', 'norm_type']:
        data_config[key] = [str(value) for value in data_config[key]]
    model_config = LingbotVLAV2Config(**config['model'], **config['train'])
    processor = build_processor(config['model']['tokenizer_path'])
    vla = VLADataset(str(root), data_config['data_name'], SimpleNamespace(**data_config),
                     data_config['robot_config_root'], config=model_config, processor=processor,
                     image_size=(256, 256), chunk_size=50)
    example_episode = manifest['splits'][options.split][0]
    result = vla[dataset._absolute_to_relative_idx[int(np.flatnonzero(episodes == example_episode)[0])]]
    assert result['state'].shape == (55,)
    assert result['actions'].shape == (50, 55)
    assert result['state_joint_mask'].sum().item() == 14
    assert result['action_joint_mask'].sum().item() == 8
    recovered = vla.feature_transform.unapply(result)
    start = int(np.flatnonzero(episodes == example_episode)[0])
    np.testing.assert_allclose(recovered['action'].numpy(), action[start:start + 50], rtol=1e-5, atol=1e-7)
    print(f"Passed: {checked} episode-boundary chunk checks, {len(manifest['splits']['train'])}/{len(manifest['splits']['validation'])} isolation, all training statistics, 55D masks and 8D action round trip.")


if __name__ == '__main__':
    main()
