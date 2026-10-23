"""Check that every node sees the same output before writing DCP shards."""

import os
from pathlib import Path
import uuid

import torch.distributed as dist


def verify_shared_training_output(output_dir: str, configuration: dict) -> None:
    """All ranks participate; fail together if storage or run settings differ."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    root = Path(output_dir)
    payload = [None]
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            filename = f".shared_output_probe_{token}"
            with (root / filename).open("w", encoding="utf-8") as stream:
                stream.write(token)
                stream.flush()
                os.fsync(stream.fileno())
            payload[0] = {"filename": filename, "token": token}
        except OSError as exc:
            payload[0] = {"error": repr(exc)}
    dist.broadcast_object_list(payload, src=0)
    probe = payload[0]
    if "error" in probe:
        raise RuntimeError(f"Could not create shared output probe: {probe['error']}")
    error = None
    try:
        if (root / probe["filename"]).read_text() != probe["token"]:
            raise RuntimeError("Shared output probe contents differ")
    except (OSError, RuntimeError) as exc:
        error = f"rank {rank}: {exc!r}"
    local = {"error": error, "output_dir": os.path.abspath(output_dir), "configuration": configuration}
    results = [None] * world_size
    dist.all_gather_object(results, local)
    if rank == 0:
        (root / probe["filename"]).unlink(missing_ok=True)
    errors = [item["error"] for item in results if item["error"]]
    if errors:
        raise RuntimeError("All nodes must see the same shared output filesystem: " + "; ".join(errors))
    if any(item["output_dir"] != results[0]["output_dir"]
           or item["configuration"] != results[0]["configuration"] for item in results):
        raise RuntimeError("Nodes disagree on output path or training/checkpoint settings")
