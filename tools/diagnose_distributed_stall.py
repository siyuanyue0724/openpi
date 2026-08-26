#!/usr/bin/env python3
"""Run a Python entrypoint with bounded ProcessGroupNCCL stall diagnostics."""

from __future__ import annotations

import datetime
import os
import runpy
from pathlib import Path


def _required_absolute_file(name: str) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"{name} must name an existing absolute file")
    return path.resolve()


def _timeout_seconds() -> int:
    raw = os.environ.get("PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS", "90")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS must be an integer") from error
    if value <= 0:
        raise RuntimeError("PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS must be positive")
    return value


def main() -> None:
    target = _required_absolute_file("PICF_DISTRIBUTED_STALL_TARGET")
    timeout = datetime.timedelta(seconds=_timeout_seconds())
    diagnostic_environment = {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_DESYNC_DEBUG": "1",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        "TORCH_NCCL_ENABLE_TIMING": "1",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
        "TORCH_NCCL_TRACE_CPP_STACK": "1",
        "TORCH_FR_BUFFER_SIZE": "2000",
        "TORCH_FR_CPP_STACK": "1",
    }
    conflicts = {
        key: os.environ[key]
        for key, value in diagnostic_environment.items()
        if key in os.environ and os.environ[key] != value
    }
    if conflicts:
        raise RuntimeError(f"conflicting ProcessGroupNCCL diagnostic environment: {conflicts}")
    os.environ.update(diagnostic_environment)

    import torch.distributed as dist

    original_init_process_group = dist.init_process_group

    def init_process_group_with_timeout(*args, **kwargs):
        if "timeout" in kwargs:
            raise RuntimeError("target already declares a process-group timeout")
        return original_init_process_group(*args, timeout=timeout, **kwargs)

    dist.init_process_group = init_process_group_with_timeout
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        dist.init_process_group = original_init_process_group


if __name__ == "__main__":
    main()
