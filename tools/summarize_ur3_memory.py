"""Summarize completed native four-GPU memory probes without estimating peaks."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def summarize(directory):
    root = Path(directory).resolve()
    ranks = [json.loads(p.read_text()) for p in sorted(root.glob('memory_rank*.json'))]
    if len(ranks) != 4 or any(r['world_size'] != 4 or r['status'] != 'completed' for r in ranks):
        raise ValueError(f'Expected four successfully completed ranks in {root}')
    if any(len(r['optimizer_steps']) != r['options']['steps'] for r in ranks):
        raise ValueError('Incomplete optimizer-step measurements')
    nvml_peaks = {}
    totals = {}
    with (root / 'nvidia-smi.csv').open() as stream:
        rows = csv.reader(stream)
        next(rows)
        for row in rows:
            if len(row) != 5:
                continue
            gpu = int(row[1])
            nvml_peaks[gpu] = max(nvml_peaks.get(gpu, 0), int(row[2].split()[0]))
            totals[gpu] = int(row[3].split()[0])
    times = dict((int(step), float(seconds)) for step, seconds in re.findall(
        r'Step (\d+)/\d+, Epoch[^\n]*?StepTime\s+([\d.]+)s', (root / 'train.log').read_text()))
    steady = [seconds for step, seconds in sorted(times.items()) if step > 2]
    gradients_finite = all(g['finite'] for r in ranks for step in r['optimizer_steps']
                           for g in step['sampled_gradients'].values())
    parameters_updated = all(all(r['sampled_parameters_updated'].values()) for r in ranks)
    return {
        'directory': str(root), 'status': 'passed', 'world_size': 4,
        'optimizer_steps_per_rank': ranks[0]['options']['steps'],
        'options': ranks[0]['options'], 'configuration': ranks[0]['configuration'],
        'trainable_parameters': ranks[0]['trainable_parameters'],
        'input_example': ranks[0]['inputs'][0],
        'per_rank_training_peak': {r['rank']: r['training_peak_memory'] for r in ranks},
        'max_peak_allocated_GiB': max(r['training_peak_memory']['peak_allocated_GiB'] for r in ranks),
        'max_peak_reserved_GiB': max(r['training_peak_memory']['peak_reserved_GiB'] for r in ranks),
        'nvml_peak_MiB_per_gpu_sampled_every_1s': nvml_peaks,
        'nvml_min_free_MiB_per_gpu_at_sampled_peak': {gpu: totals[gpu] - peak for gpu, peak in nvml_peaks.items()},
        'native_step_seconds': times,
        'median_native_step_seconds_excluding_first_two': statistics.median(steady),
        'projected_5000_training_hours_excluding_saves': statistics.median(steady) * 5000 / 3600,
        'sampled_gradients_finite': gradients_finite,
        'sampled_parameters_updated_globally': parameters_updated,
        'checkpoint_serialization_measured': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    report = {'experiments': [summarize(root) for root in options.directories],
              'limitations': ['Short fixed-shape training probes, not a 5000-step stability test.',
                              'Checkpoint/HF export peak memory and save/resume are not measured.',
                              'NVML uses 1-second sampling and can miss subsecond driver peaks.',
                              'Camera-slot count and allocator settings vary by experiment; see each input example and configuration.']}
    Path(options.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for experiment in report['experiments']:
        print(json.dumps({key: experiment[key] for key in [
            'directory', 'status', 'trainable_parameters', 'max_peak_allocated_GiB',
            'max_peak_reserved_GiB', 'nvml_peak_MiB_per_gpu_sampled_every_1s',
            'median_native_step_seconds_excluding_first_two']}, indent=2))


if __name__ == '__main__':
    main()
