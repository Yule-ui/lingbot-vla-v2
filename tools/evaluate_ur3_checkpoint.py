"""Evaluate UR3 action chunks on held-out episodes, without commanding a robot."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ur3_adapters import install_data_adapters


def summarize(predicted, target):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[1] != 8:
        raise ValueError(f"Unexpected action shapes: {predicted.shape}, {target.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite action")
    error = predicted - target
    pn = np.linalg.norm(predicted[:, 3:7], axis=1)
    tn = np.linalg.norm(target[:, 3:7], axis=1)
    if (pn < 1e-8).any() or (tn < 1e-8).any():
        raise ValueError("Invalid zero-norm quaternion")
    pq = predicted[:, 3:7] / pn[:, None]
    tq = target[:, 3:7] / tn[:, None]
    dots = np.clip(np.abs(np.sum(pq * tq, axis=1)), 0, 1)
    angles = np.rad2deg(2 * np.arccos(dots))
    return {
        "frames": len(predicted),
        "translation_rmse_mm": float(1000 * np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        "translation_mae_axes_mm": (1000 * np.mean(np.abs(error[:, :3]), axis=0)).tolist(),
        "rotation_mean_deg": float(np.mean(angles)),
        "rotation_p95_deg": float(np.percentile(angles, 95)),
        "gripper_mae": float(np.mean(np.abs(error[:, 7]))),
        "raw_action_mae_axes": np.mean(np.abs(error), axis=0).tolist(),
        "raw_action_mse_mixed_units": float(np.mean(error ** 2)),
        "predicted_quaternion_norm_min": float(pn.min()),
        "predicted_quaternion_norm_max": float(pn.max()),
    }


def image_to_numpy(image):
    image = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Unexpected image shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        if np.max(image) <= 2:
            image = image * 255
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model_path", help="Exported hf_ckpt directory")
    source.add_argument("--server_host", help="Test an existing WebSocket server")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--manifest", default=str(ROOT / "assets/training_data/ur3_split_seed42.json"))
    parser.add_argument("--traj_ids", type=int, nargs="+")
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--max_chunks_per_episode", type=int)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not 1 <= args.use_length <= 50:
        parser.error("use_length must be in [1, 50]")
    if args.max_chunks_per_episode is not None and args.max_chunks_per_episode < 1:
        parser.error("max_chunks_per_episode must be positive")

    manifest = install_data_adapters(args.manifest, "validation")
    episodes = args.traj_ids or manifest["splits"]["validation"]
    if not set(episodes).issubset(manifest["splits"]["validation"]):
        raise ValueError("Evaluation must use held-out episodes only")
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset

    if args.model_path:
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        policy = LingbotVLAv2Server(
            path_to_pi_model=args.model_path, use_bf16=True, use_fp32=False,
            use_compile=False, chunk_ret=True, use_length=args.use_length,
        )
    else:
        from deploy.websocket_client_policy import WebsocketClientPolicy
        policy = WebsocketClientPolicy(host=args.server_host, port=args.port)

    dataset_root = Path(manifest["dataset_root"])
    dataset = LeRobotDataset(
        dataset_root.name, root=dataset_root,
        delta_timestamps={"action": [t / 25 for t in range(50)]},
    )
    if dataset.meta.fps != 25:
        raise ValueError("This evaluator expects the audited 25 Hz UR3 dataset")
    per_episode = []
    all_predictions, all_targets, all_hold_actions, timings, server_timings = [], [], [], [], []
    total_chunks = 0
    for episode in episodes:
        policy.reset("ur3_robotiq_eef_relative")
        metadata = dataset.meta.episodes[episode]
        start, end = int(metadata["dataset_from_index"]), int(metadata["dataset_to_index"])
        predictions, targets, hold_actions = [], [], []
        chunks = 0
        task = None
        for global_index in range(start, end, args.use_length):
            if args.max_chunks_per_episode is not None and chunks >= args.max_chunks_per_episode:
                break
            local_index = dataset._absolute_to_relative_idx[global_index]
            item = dataset[local_index]
            task = item["task"]
            observation = {
                "observation.state": item["observation.state"].cpu().numpy().astype(np.float32),
                "observation.images.camera_global": image_to_numpy(item["observation.images.camera_global"]),
                "observation.images.camera_wrist": image_to_numpy(item["observation.images.camera_wrist"]),
                "task": task,
            }
            if args.model_path:
                # Matched initial diffusion noise for comparing checkpoints.
                torch.manual_seed(42 + global_index)
                torch.cuda.synchronize()
            before = time.perf_counter()
            result = policy.infer(observation)
            if args.model_path:
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - before)
            if "server_timing" in result:
                server_timings.append(result["server_timing"]["infer_ms"] / 1000)
            action = np.asarray(result["action"], dtype=np.float32)
            if action.shape != (args.use_length, 8):
                raise ValueError(f"Unexpected action response: {action.shape}")
            n = min(args.use_length, end - global_index)
            predictions.append(action[:n])
            targets.append(item["action"].cpu().numpy()[:n])
            hold = np.zeros((n, 8), dtype=np.float32)
            hold[:, 6] = 1
            hold[:, 7] = observation["observation.state"][-1]
            hold_actions.append(hold)
            chunks += 1
        predicted, target = np.concatenate(predictions), np.concatenate(targets)
        metrics = {"episode": episode, "task": task, "chunks": chunks, **summarize(predicted, target)}
        per_episode.append(metrics)
        all_predictions.append(predicted)
        all_targets.append(target)
        all_hold_actions.append(np.concatenate(hold_actions))
        total_chunks += chunks
        print("EPISODE", json.dumps(metrics), flush=True)

    report = {
        "model_path": args.model_path,
        "server_host": args.server_host,
        "episodes": episodes,
        "split": "validation",
        "use_bf16": True if args.model_path else "determined by server",
        "use_compile": False if args.model_path else "determined by server",
        "use_length": args.use_length,
        "chunks": total_chunks,
        "full_episodes": args.max_chunks_per_episode is None,
        "tail_padding_excluded": True,
        "rotation_metric": "geodesic angle after unit normalization, invariant to q/-q",
        "per_episode": per_episode,
        "overall": summarize(np.concatenate(all_predictions), np.concatenate(all_targets)),
        "hold_baseline": summarize(np.concatenate(all_hold_actions), np.concatenate(all_targets)),
        "hold_baseline_definition": "zero EEF translation, identity local rotation, gripper fixed at chunk-start observation",
        "policy_call_median_seconds_after_first": float(np.median(timings[1:] or timings)),
        "policy_call_p95_seconds_after_first": float(np.percentile(timings[1:] or timings, 95)),
        "first_policy_call_seconds": timings[0],
    }
    if server_timings:
        report["server_infer_median_seconds_after_first"] = float(np.median(server_timings[1:] or server_timings))
    if args.model_path:
        report["peak_gpu_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_gpu_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("REPORT", path, json.dumps(report["overall"]), flush=True)


if __name__ == "__main__":
    main()
