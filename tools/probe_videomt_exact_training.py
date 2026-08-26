#!/usr/bin/env python3
"""Probe full released VidEoMT loss and two-frame gradient reachability."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from picf_next.videomt_exact.class_agnostic import (
    VIDEOMT_MATCHER_IDENTITIES,
    VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    build_class_agnostic_criterion,
    flatten_class_agnostic_outputs,
    flatten_class_agnostic_targets,
)
from picf_next.videomt_exact.runtime import (
    ExactVidEoMTConfig,
    load_exact_videomt,
    normalize_rgb_255,
)
from picf_next.videomt_exact.training import (
    apply_released_loss_weights,
    build_released_criterion,
    build_released_online_criterion,
    flatten_video_outputs_for_released_criterion,
    flatten_video_targets_for_released_criterion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=198)
    parser.add_argument(
        "--objective",
        choices=("released-online", "released-base-ablation", "class-agnostic"),
        default="released-online",
    )
    parser.add_argument(
        "--matcher-identity",
        choices=VIDEOMT_MATCHER_IDENTITIES,
        default=VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    )
    return parser.parse_args()


def _group_gradient_receipt(module: torch.nn.Module) -> dict[str, float | int]:
    square_sum = 0.0
    maximum = 0.0
    parameter_tensors = 0
    gradient_tensors = 0
    nonzero_tensors = 0
    parameter_numel = 0
    for parameter in module.parameters():
        parameter_tensors += 1
        parameter_numel += parameter.numel()
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        gradient = parameter.grad.detach().float()
        if torch.count_nonzero(gradient):
            nonzero_tensors += 1
        square_sum += float(gradient.square().sum())
        maximum = max(maximum, float(gradient.abs().max()))
    return {
        "parameter_tensors": parameter_tensors,
        "parameter_numel": parameter_numel,
        "gradient_tensors": gradient_tensors,
        "nonzero_gradient_tensors": nonzero_tensors,
        "l2_norm": math.sqrt(square_sum),
        "max_abs": maximum,
    }


def _synthetic_clip(size: int, seed: int, num_frames: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    first = torch.randint(0, 256, (3, size, size), generator=generator, dtype=torch.uint8)
    frames = tuple(
        torch.roll(first, shifts=(3 * frame, -5 * frame), dims=(-2, -1))
        for frame in range(num_frames)
    )
    return normalize_rgb_255(torch.stack(frames))


def _synthetic_targets(size: int, num_frames: int) -> list[dict[str, torch.Tensor]]:
    masks = torch.zeros(2, num_frames, size, size)
    extent = max(8, size // 5)
    y0, x0 = size // 5, size // 6
    y1, x1 = size // 2, size // 2
    for frame in range(num_frames):
        masks[0, frame, y0 + frame : y0 + extent + frame, x0 + frame : x0 + extent + frame] = 1
        masks[1, frame, y1 - frame : y1 + extent - frame, x1 - frame : x1 + extent - frame] = 1
    return [
        {
            "labels": torch.tensor([0, 3], dtype=torch.int64),
            "ids": torch.arange(2, dtype=torch.int64).unsqueeze(1).expand(-1, num_frames),
            "masks": masks,
        }
    ]


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be positive and divisible by 16")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    load_started = time.perf_counter()
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.checkpoint,
            local_dinov3_bundle=args.dinov3_bundle,
            num_frames=args.num_frames,
        ),
        device=device,
        dtype=torch.float32,
    )
    criterion = (
        build_class_agnostic_criterion(
            matcher_identity=args.matcher_identity,
            num_frames=args.num_frames,
        )
        if args.objective == "class-agnostic"
        else (
            build_released_online_criterion(num_frames=args.num_frames)
            if args.objective == "released-online"
            else build_released_criterion()
        )
    ).to(device)
    runtime.train()
    load_seconds = time.perf_counter() - load_started

    clip = _synthetic_clip(args.image_size, args.seed, args.num_frames).to(
        device=device,
        dtype=torch.float32,
    )
    targets = [
        {name: value.to(device) for name, value in target.items()}
        for target in _synthetic_targets(args.image_size, args.num_frames)
    ]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    output = runtime(clip)
    if args.objective == "class-agnostic":
        flat_outputs = flatten_class_agnostic_outputs(output)
        flat_targets = flatten_class_agnostic_targets(targets)
    else:
        flat_outputs = flatten_video_outputs_for_released_criterion(output)
        flat_targets = flatten_video_targets_for_released_criterion(targets)
    raw_losses = criterion(flat_outputs, flat_targets)
    weighted_losses = apply_released_loss_weights(raw_losses, criterion)
    total_loss = sum(weighted_losses.values())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_loss_seconds = time.perf_counter() - forward_started

    backward_started = time.perf_counter()
    total_loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started

    backbone = runtime.model.encoder.backbone
    gradient_groups = {
        "dino_blocks_0_19": _group_gradient_receipt(backbone.blocks[:20]),
        "shared_query_patch_blocks_20_23": _group_gradient_receipt(backbone.blocks[20:24]),
        "dino_patch_embedding": _group_gradient_receipt(backbone.patch_embed),
        "dino_final_norm": _group_gradient_receipt(backbone.norm),
        "learned_query_bank": _group_gradient_receipt(runtime.model.q),
        "temporal_query_updater": _group_gradient_receipt(runtime.model.query_updater),
        "class_head": _group_gradient_receipt(runtime.model.class_head),
        "mask_head": _group_gradient_receipt(runtime.model.mask_head),
        "mask_upscaler": _group_gradient_receipt(runtime.model.upscale),
    }
    failures = [
        name
        for name, receipt in gradient_groups.items()
        if receipt["gradient_tensors"] == 0 or receipt["nonzero_gradient_tensors"] == 0
    ]
    report = {
        "schema": "picf-next.videomt-exact-training-gradient-probe.v1",
        "claim_scope": (
            "synthetic FP32 full-graph gradient reachability; not AMP parity or learning quality"
        ),
        "objective": args.objective,
        "matcher_identity": args.matcher_identity,
        "num_frames": args.num_frames,
        "device": str(device),
        "dtype": "torch.float32",
        "image_size": args.image_size,
        "model_input_shape": list(clip.shape),
        "class_logits_shape": list(output.class_logits.shape),
        "mask_logits_shape": list(output.mask_logits.shape),
        "auxiliary_output_count": len(output.auxiliary_outputs),
        "raw_loss_keys": sorted(raw_losses),
        "weighted_loss_keys": sorted(weighted_losses),
        "raw_losses": {name: float(value.detach()) for name, value in raw_losses.items()},
        "weighted_losses": {
            name: float(value.detach()) for name, value in weighted_losses.items()
        },
        "total_weighted_loss": float(total_loss.detach()),
        "timing_seconds": {
            "load": load_seconds,
            "forward_and_loss": forward_loss_seconds,
            "backward": backward_seconds,
        },
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "gradient_groups": gradient_groups,
        "failures": failures,
        "passed": not failures and bool(weighted_losses) and torch.isfinite(total_loss).item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
