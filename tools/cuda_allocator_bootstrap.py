"""Configure the pinned PyTorch CUDA allocator before importing PyTorch."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

CUDA_ALLOCATOR_MODES = ("native", "expandable-segments", "cuda-malloc-async")
CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE = "PYTORCH_CUDA_ALLOC_CONF"
CUDA_ALLOCATOR_ALTERNATE_ENVIRONMENT_VARIABLE = "PYTORCH_ALLOC_CONF"
CUDA_EXPANDABLE_SEGMENTS_CONFIG = "expandable_segments:True"
CUDA_MALLOC_ASYNC_CONFIG = "backend:cudaMallocAsync"


def configure_cuda_allocator(mode: str) -> None:
    """Apply one explicit allocator mode in an otherwise clean process."""

    if mode not in CUDA_ALLOCATOR_MODES:
        raise ValueError("native CUDA allocator mode is unsupported")
    inherited = {
        name: os.environ[name]
        for name in (
            CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
            CUDA_ALLOCATOR_ALTERNATE_ENVIRONMENT_VARIABLE,
        )
        if name in os.environ
    }
    if inherited:
        raise RuntimeError(
            "native runner refuses inherited CUDA allocator configuration; "
            f"select it only through --cuda-allocator: {inherited}"
        )
    if mode == "expandable-segments":
        os.environ[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] = CUDA_EXPANDABLE_SEGMENTS_CONFIG
    elif mode == "cuda-malloc-async":
        os.environ[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] = CUDA_MALLOC_ASYNC_CONFIG


def bootstrap_cuda_allocator(argv: Sequence[str]) -> str:
    """Pre-parse the allocator option before any dependency can import PyTorch."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda-allocator",
        choices=CUDA_ALLOCATOR_MODES,
        default="native",
    )
    args, _unknown = parser.parse_known_args(argv)
    configure_cuda_allocator(args.cuda_allocator)
    return args.cuda_allocator
