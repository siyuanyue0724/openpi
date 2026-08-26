#!/usr/bin/env python3
"""Evaluate one adapted VidEoMT checkpoint without further optimization."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from picf_next.videomt_exact.calvin_dataset import (
    HashBoundCalvinFrameStore,
    build_calvin_videomt_split_plan,
)
from picf_next.videomt_exact.checkpoint import sha256_file
from picf_next.videomt_exact.evaluation import evaluate_calvin_anchor_windows
from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-checkpoint", required=True, type=Path)
    parser.add_argument("--adapted-checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--source-split-root", required=True, type=Path)
    parser.add_argument("--source-overlay-root", type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--golden-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-clips", type=int, default=4)
    parser.add_argument("--eval-short-edge", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def _selected_windows(
    windows: tuple[tuple[int, ...], ...],
    count: int,
) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        raise ValueError("eval-clips must be positive")
    selected = np.linspace(0, len(windows) - 1, min(count, len(windows)), dtype=np.int64)
    return tuple(windows[int(index)] for index in np.unique(selected))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.eval_short_edge <= 0:
        raise ValueError("eval-short-edge must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    started = time.perf_counter()
    store = HashBoundCalvinFrameStore(
        source_split_root=args.source_split_root,
        source_overlay_root=args.source_overlay_root,
        sidecar_root=args.sidecar_root,
    )
    source_audit = store.audit_source_rgb()
    split = build_calvin_videomt_split_plan(
        golden_manifest_path=args.golden_manifest,
        store=store,
    )
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.published_checkpoint,
            local_dinov3_bundle=args.dinov3_bundle,
            num_frames=split.clip_length,
        ),
        device="cpu",
        dtype=torch.float32,
    )
    adapted_path = args.adapted_checkpoint.expanduser().resolve()
    adapted_state = torch.load(
        adapted_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(adapted_state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in adapted_state.items()
    ):
        raise TypeError("adapted VidEoMT checkpoint must be a named tensor mapping")
    incompatible = runtime.model.load_state_dict(adapted_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("adapted VidEoMT checkpoint did not load strictly")
    runtime.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_windows = _selected_windows(split.train_windows, args.eval_clips)
    heldout_windows = _selected_windows(split.heldout_windows, args.eval_clips)
    report: dict[str, object] = {
        "schema": "picf-next.videomt-exact-calvin-anchor-evaluation.v1",
        "claim_scope": (
            "fixed task-blind temporal-component evaluation; includes oracle-Hungarian "
            "and model-ranked proposal metrics; not episode-disjoint or action evidence"
        ),
        "forward_inputs": ["rgb_static"],
        "adapted_checkpoint": {
            "path": str(adapted_path),
            "size": adapted_path.stat().st_size,
            "sha256": sha256_file(adapted_path),
            "tensor_count": len(adapted_state),
        },
        "source_rgb_audit": source_audit,
        "split": {
            "episode_disjoint": split.episode_disjoint,
            "train_windows": [list(value) for value in train_windows],
            "heldout_windows": [list(value) for value in heldout_windows],
        },
        "resolution": args.eval_short_edge,
        "dtype": str(dtype),
        "device": str(device),
        "train": evaluate_calvin_anchor_windows(
            runtime=runtime,
            store=store,
            windows=train_windows,
            short_edge=args.eval_short_edge,
            device=device,
            dtype=dtype,
            panel_path=output_dir / "train.png",
        ),
        "heldout": evaluate_calvin_anchor_windows(
            runtime=runtime,
            store=store,
            windows=heldout_windows,
            short_edge=args.eval_short_edge,
            device=device,
            dtype=dtype,
            panel_path=output_dir / "heldout.png",
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
