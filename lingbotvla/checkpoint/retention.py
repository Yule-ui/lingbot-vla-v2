"""Commit and rotate local DCP training state without deleting HF exports."""

import json
import os
import re
import shutil
from pathlib import Path


DCP_COMPLETE = "dcp_complete.json"
_DCP_DIRS = ("model", "optimizer", "extra_state", "ema")


def mark_dcp_complete(checkpoint_path: str, step: int, world_size: int) -> None:
    """Call on rank zero after every rank has finished its synchronous save."""
    path = Path(checkpoint_path)
    for name in ("model", "optimizer"):
        if not (path / name / ".metadata").is_file():
            raise RuntimeError(f"Missing DCP metadata: {path / name}")
        if not any((path / name).glob("*.distcp")):
            raise RuntimeError(f"Missing DCP shards: {path / name}")
    for rank in range(world_size):
        if not (path / "extra_state" / f"extra_state_rank_{rank}.pt").is_file():
            raise RuntimeError(f"Missing DCP extra state for rank {rank}: {path}")
    tmp_path = path / f".{DCP_COMPLETE}.tmp"
    with tmp_path.open("w", encoding="utf-8") as stream:
        json.dump({"global_step": step, "world_size": world_size}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_path, path / DCP_COMPLETE)


def completed_dcp_candidates(checkpoint_root: str, world_size: int) -> list[str]:
    """Ignore HF-only directories and interrupted, uncommitted DCP writes."""
    root = Path(checkpoint_root)
    candidates = []
    if not root.is_dir():
        return candidates
    for path in root.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        try:
            info = json.loads((path / DCP_COMPLETE).read_text())
            step = int(match.group(1))
            if info != {"global_step": step, "world_size": world_size}:
                continue
            if not all((path / name / ".metadata").is_file() for name in ("model", "optimizer")):
                continue
            if not all((path / "extra_state" / f"extra_state_rank_{rank}.pt").is_file()
                       for rank in range(world_size)):
                continue
            candidates.append((step, str(path)))
        except (OSError, ValueError, TypeError):
            continue
    return [path for _, path in sorted(candidates, reverse=True)]


def prune_older_dcp(checkpoint_root: str, latest_step: int) -> None:
    """A newly committed DCP replaces older training states; HF files stay."""
    root = Path(checkpoint_root)
    if not (root / f"global_step_{latest_step}" / DCP_COMPLETE).is_file():
        raise RuntimeError("Refusing to prune before the new DCP is committed")
    for path in root.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if not match or not path.is_dir() or int(match.group(1)) >= latest_step:
            continue
        (path / DCP_COMPLETE).unlink(missing_ok=True)
        for name in _DCP_DIRS:
            if (path / name).exists():
                shutil.rmtree(path / name)
        if not any(path.iterdir()):
            path.rmdir()
