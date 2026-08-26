#!/usr/bin/env python3
"""Run a Python entrypoint and preserve a CUDA allocator snapshot on OOM."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _required_absolute_file(name: str) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"{name} must name an existing absolute file")
    return path.resolve()


def _required_absolute_directory(name: str) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required")
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be absolute")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _maximum_entries() -> int:
    raw = os.environ.get("PICF_CUDA_OOM_MAX_ENTRIES", "200000")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("PICF_CUDA_OOM_MAX_ENTRIES must be an integer") from error
    if value <= 0:
        raise RuntimeError("PICF_CUDA_OOM_MAX_ENTRIES must be positive")
    return value


def main() -> None:
    import torch

    target = _required_absolute_file("PICF_CUDA_OOM_TARGET")
    output_dir = _required_absolute_directory("PICF_CUDA_OOM_SNAPSHOT_DIR")
    rank = os.environ.get("RANK", "unknown")
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    output = output_dir / f"cuda-oom-rank-{rank}-local-{local_rank}.pickle"
    if output.exists():
        raise FileExistsError(f"CUDA OOM snapshot output already exists: {output}")

    original_backward = torch.Tensor.backward

    def backward_with_snapshot(self, *args, **kwargs):
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=_maximum_entries(),
        )
        try:
            return original_backward(self, *args, **kwargs)
        except torch.OutOfMemoryError:
            try:
                torch.cuda.memory._dump_snapshot(str(output))
            except Exception as snapshot_error:
                sys.stderr.write(
                    f"failed to dump CUDA OOM snapshot at {output}: "
                    f"{type(snapshot_error).__name__}: {snapshot_error}\n"
                )
                sys.stderr.flush()
            raise
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    torch.Tensor.backward = backward_with_snapshot
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        torch.Tensor.backward = original_backward


if __name__ == "__main__":
    main()
