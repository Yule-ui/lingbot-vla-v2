"""Summarize optimizer update timings from an isolated training benchmark."""

import math
import statistics


def summarize_training_benchmark(step_times, global_batch_size, warmup_steps=2):
    times = [float(value) for value in step_times]
    if not times or any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError("Benchmark step times must be finite and positive")
    if global_batch_size <= 0 or warmup_steps < 0:
        raise ValueError("Invalid batch size or benchmark warmup count")
    excluded = min(warmup_steps, len(times) - 1)
    steady = times[excluded:]
    mean = statistics.mean(steady)
    return {
        "optimizer_updates": len(times),
        "excluded_initial_updates": excluded,
        "step_times_seconds": times,
        "steady_mean_seconds": mean,
        "steady_median_seconds": statistics.median(steady),
        "steady_min_seconds": min(steady),
        "steady_max_seconds": max(steady),
        "global_samples_per_second": global_batch_size / mean,
        "estimated_3000_updates_hours_excluding_saves": 3000 * mean / 3600,
        "training_checkpoints_saved": False,
    }
