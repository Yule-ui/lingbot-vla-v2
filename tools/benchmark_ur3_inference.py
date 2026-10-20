"""Benchmark exported UR3 inference and loopback RPC without robot hardware."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ur3_adapters import install_data_adapters
from tools.evaluate_ur3_checkpoint import image_to_numpy


def validate_action(result):
    action = np.asarray(result["action"], dtype=np.float32)
    if action.shape != (50, 8) or not np.isfinite(action).all():
        raise ValueError(f"Invalid action chunk: {action.shape}")
    norms = np.linalg.norm(action[:, 3:7], axis=1)
    if (norms < 1e-8).any():
        raise ValueError("Zero action quaternion")
    return {
        "shape": list(action.shape), "finite": True,
        "quaternion_norm_min": float(norms.min()),
        "quaternion_norm_max": float(norms.max()),
        "translation_norm_max_mm": float(np.linalg.norm(action[:, :3], axis=1).max() * 1000),
        "gripper_min": float(action[:, 7].min()),
        "gripper_max": float(action[:, 7].max()),
    }


async def check_rpc(policy, observations):
    from deploy.websocket_policy_server import WebsocketPolicyServer, _health_check
    from deploy.websocket_client_policy import WebsocketClientPolicy
    from deploy.msgpack_numpy import Packer
    import websockets.asyncio.server

    rpc = WebsocketPolicyServer(policy, host="127.0.0.1", port=0)
    async with websockets.asyncio.server.serve(
        rpc._handler, "127.0.0.1", 0, compression=None,
        max_size=None, ping_interval=None, process_request=_health_check,
    ) as server:
        port = server.sockets[0].getsockname()[1]

        def client_calls():
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=10) as response:
                health = response.read().decode().strip()
            client = WebsocketClientPolicy(host="127.0.0.1", port=port)
            calls = []
            try:
                for observation in observations:
                    client.reset("ur3_robotiq_eef_relative")
                    before = time.perf_counter()
                    result = client.infer(observation)
                    elapsed = time.perf_counter() - before
                    calls.append({
                        "task": observation["task"], "wall_seconds": elapsed,
                        "server_seconds": result["server_timing"]["infer_ms"] / 1000,
                        "request_bytes": len(Packer().pack(observation)),
                        "response_bytes": len(Packer().pack(result)),
                        **validate_action(result),
                    })
            finally:
                client._ws.close()
            return {"host": "127.0.0.1", "healthz": health, "calls": calls,
                    "scope": "loopback only; WAN latency and robot control not tested"}

        return await asyncio.to_thread(client_calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--iterations", type=int, default=9)
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("At least three measured iterations are required")
    manifest = install_data_adapters(args.manifest, "validation")
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.data.vla_data.base_dataset import LeRobotDataset

    torch.cuda.reset_peak_memory_stats()
    before = time.perf_counter()
    policy = LingbotVLAv2Server(
        path_to_pi_model=args.model_path, use_bf16=True, use_fp32=False,
        use_compile=False, chunk_ret=True, use_length=50,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - before
    loaded_gib = torch.cuda.memory_allocated() / 2**30
    root = Path(manifest["dataset_root"])
    dataset = LeRobotDataset(root.name, root=root)
    observations, seen = [], set()
    for episode in manifest["splits"]["validation"]:
        index = int(dataset.meta.episodes[episode]["dataset_from_index"])
        item = dataset[dataset._absolute_to_relative_idx[index]]
        if item["task"] in seen:
            continue
        seen.add(item["task"])
        observations.append({
            "observation.state": item["observation.state"].cpu().numpy().astype(np.float32),
            "observation.images.camera_global": image_to_numpy(item["observation.images.camera_global"]),
            "observation.images.camera_wrist": image_to_numpy(item["observation.images.camera_wrist"]),
            "task": item["task"],
        })
    if len(observations) != 3:
        raise ValueError(f"Expected three held-out task types, got {seen}")
    policy.reset("ur3_robotiq_eef_relative")
    timings = []
    for i in range(args.iterations + 3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = policy.infer(observations[i % len(observations)])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        validate_action(result)
        timings.append(elapsed)
        print(f"BENCHMARK iteration={i} seconds={elapsed:.6f}", flush=True)
    rpc = asyncio.run(check_rpc(policy, observations))
    report = {
        "model_path": args.model_path, "manifest": args.manifest,
        "device": torch.cuda.get_device_name(),
        "device_total_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "use_bf16": True, "use_compile": False,
        "num_denoise_steps": policy.config.num_steps,
        "image_size_wh": [policy.data_config.image_width, policy.data_config.image_height],
        "cameras": policy.data_config.cameras,
        "parameter_bytes_GiB": sum(p.numel() * p.element_size() for p in policy.vla.parameters()) / 2**30,
        "load_seconds": load_seconds, "loaded_allocated_GiB": loaded_gib,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        "cold_call_seconds": timings[0], "warmup_calls_excluded": 3,
        "measured_calls": args.iterations,
        "median_seconds": float(np.median(timings[3:])),
        "p95_seconds": float(np.percentile(timings[3:], 95)),
        "max_seconds": max(timings[3:]), "all_call_seconds": timings,
        "websocket": rpc,
        "limitations": ["No RTX 4060 available for direct timing",
                        "No physical robot or remote client connected",
                        "Loopback bandwidth does not measure deployment network",
                        "Finite chunks and offline errors do not establish closed-loop success"],
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("REPORT", output, json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
