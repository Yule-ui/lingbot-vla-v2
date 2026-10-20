"""Prepare isolated split/statistics for the audited 20261016 UR3 dataset."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((ROOT / 'docs/ur3_20261016_data_audit.json').read_text())
    if audit['issues']:
        raise ValueError(audit['issues'])
    dataset = Path(audit['dataset'])
    split = audit['suggested_split']
    manifest_path = ROOT / 'assets/training_data/ur3_20261016_split_seed42.json'
    norm_path = ROOT / 'assets/norm_stats/ur3_robotiq_eef_relative_20261016.json'
    manifest = {
        'version': 1, 'dataset_root': str(dataset), 'seed': 42,
        'splits': {k: split[k] for k in ('train', 'validation')},
        'frames': {'train': split['train_frames'], 'validation': split['validation_frames']},
        'state_quaternion_reference': audit['state_quaternion_reference'],
        'min_abs_reference_dot': audit['min_abs_reference_dot'],
        'norm_stats_file': str(norm_path),
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f'Refusing to replace different manifest: {manifest_path}')
    raw = pa.concat_tables([pq.read_table(p, columns=['index', 'episode_index', 'observation.state', 'action'])
                            for p in sorted((dataset/'data').rglob('*.parquet'))]).sort_by([('index', 'ascending')]).to_pydict()
    train = np.isin(raw['episode_index'], split['train'])
    state = np.asarray(raw['observation.state'], dtype=np.float64)[train]
    action = np.asarray(raw['action'], dtype=np.float64)[train]
    q = state[:, 12:16]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[(q @ np.asarray(manifest['state_quaternion_reference'])) < 0] *= -1
    state[:, 12:16] = q
    values = {
        'observation.state.arm.position': state[:, :6],
        'observation.state.end.position': np.concatenate([state[:, 6:9], state[:, 12:16]], axis=1),
        'observation.state.effector.position': state[:, 16:17],
        'action.end.position': action[:, :7],
        'action.effector.position': action[:, 7:8],
    }
    statistics = {}
    for key, array in values.items():
        statistics[key] = {'mean': array.mean(0).tolist(), 'std': array.std(0).tolist(),
                           'min': array.min(0).tolist(), 'max': array.max(0).tolist()}
        for label, percentile in (('q01', .01), ('q99', .99), ('q02', .02), ('q98', .98)):
            statistics[key][label] = np.quantile(array, percentile, axis=0).tolist()
    norm = {'count': len(action), 'norm_stats': statistics,
            'ur3_preprocessing': {'state_quaternion_reference': manifest['state_quaternion_reference'],
                                  'manifest': str(manifest_path), 'train_episodes': split['train']},
            'statistics_method': 'exact NumPy float64 population mean/std and quantiles; training frames only, each once'}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    norm_path.write_text(json.dumps(norm, indent=2)+'\n')
    config = yaml.safe_load((ROOT/'configs/vla/ur3/train.yaml').read_text())
    config['data'].update(train_path=str(dataset), norm_stats_file=str(norm_path))
    config['train']['output_dir'] = '/data-teamA/outputs/ur3_20261016/data_check'
    config_dir = ROOT/'configs/vla/ur3_20261016'
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir/'data_check.yaml').write_text(
        '# Data/feature validation configuration; not the proposed partial-backbone training implementation.\n'
        +yaml.safe_dump(config, sort_keys=False))
    print('Manifest:', manifest_path)
    print('Statistics:', norm_path)
    print('Training/validation frames:', manifest['frames'])


if __name__ == '__main__':
    main()
