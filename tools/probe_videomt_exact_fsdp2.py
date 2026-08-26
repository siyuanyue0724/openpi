#!/usr/bin/env python3
"""Run one real two-rank VidEoMT FSDP2 source-objective closure.

This probe deliberately excludes LingBot.  Its only claim is that the complete
released VidEoMT graph can be loaded, sharded at the modules its manual block
path actually calls, and differentiated through its complete five-frame source
criterion on two CUDA ranks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from picf_next.videomt_exact.fsdp2 import parallelize_exact_videomt_fsdp2
from picf_next.videomt_exact.joint_training import CompleteCalvinVidEoMTObjective
from picf_next.videomt_exact.paired_training import (
    run_complete_causal_videomt_training_transaction,
)
from picf_next.videomt_exact.runtime import (
    ExactVidEoMTConfig,
    load_exact_videomt,
    normalize_rgb_255,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--adapted-checkpoint", required=True, type=Path)
    parser.add_argument("--adapted-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=207)
    return parser.parse_args()


def _local_gradient_receipt(module: torch.nn.Module) -> dict[str, object]:
    gradient_tensors = 0
    nonzero_gradient_tensors = 0
    nonfinite_gradient_tensors = 0
    for parameter in module.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_tensors += 1
        local_gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
        local_gradient = local_gradient.detach()
        if not bool(torch.isfinite(local_gradient).all().item()):
            nonfinite_gradient_tensors += 1
        if bool(torch.count_nonzero(local_gradient).item()):
            nonzero_gradient_tensors += 1
    return {
        "gradient_tensors": gradient_tensors,
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
        "nonfinite_gradient_tensors": nonfinite_gradient_tensors,
        "passed": gradient_tensors > 0
        and nonzero_gradient_tensors > 0
        and nonfinite_gradient_tensors == 0,
    }


def _synthetic_frame(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rgb = torch.randint(0, 256, (1, 3, size, size), generator=generator, dtype=torch.uint8)
    return normalize_rgb_255(rgb)


def _synthetic_clip(size: int, seed: int) -> torch.Tensor:
    first = _synthetic_frame(size, seed)
    return torch.cat(
        tuple(
            torch.roll(first, shifts=(3 * frame, -5 * frame), dims=(-2, -1))
            for frame in range(5)
        )
    )


def _synthetic_targets(size: int) -> list[dict[str, torch.Tensor]]:
    masks = torch.zeros(2, 5, size, size)
    extent = max(8, size // 5)
    for frame in range(5):
        masks[0, frame, size // 5 + frame : size // 5 + extent + frame,
              size // 6 + frame : size // 6 + extent + frame] = 1
        masks[1, frame, size // 2 - frame : size // 2 + extent - frame,
              size // 2 - frame : size // 2 + extent - frame] = 1
    return [
        {
            "labels": torch.zeros(2, dtype=torch.long),
            "ids": torch.arange(2, dtype=torch.long).unsqueeze(1).expand(-1, 5),
            "masks": masks,
            "valid_pixels": torch.ones(5, size, size, dtype=torch.bool),
        }
    ]


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 16:
        raise ValueError("--image-size must be positive and divisible by 16")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError("exact VidEoMT FSDP2 probe requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.checkpoint.resolve(),
            local_dinov3_bundle=args.dinov3_bundle.resolve(),
            adapted_checkpoint_path=args.adapted_checkpoint.resolve(),
            adapted_checkpoint_sha256=args.adapted_checkpoint_sha256,
            num_frames=5,
        ),
        device=device,
        dtype=torch.float32,
    )
    runtime.requires_grad_(True).train()
    parallelized, receipt = parallelize_exact_videomt_fsdp2(
        runtime.model,
        parameter_dtype=torch.bfloat16,
        reduction_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cpu_offload=False,
    )
    if parallelized is not runtime.model:
        raise RuntimeError("FSDP2 replaced the authenticated VidEoMT source instance")
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    clip = _synthetic_clip(args.image_size, args.seed).to(device=device)
    targets = [
        {name: value.to(device) for name, value in target.items()}
        for target in _synthetic_targets(args.image_size)
    ]
    objective = CompleteCalvinVidEoMTObjective().to(device).train()
    reset = torch.ones(1, dtype=torch.bool, device=device)
    forward_started = time.perf_counter()
    transaction = run_complete_causal_videomt_training_transaction(
        runtime,
        objective,
        normalized_padded_rgb=clip,
        clip_targets=targets,
        previous_queries=None,
        reset=reset,
    )
    output = transaction.sequence.merged
    source = transaction.source_objective
    loss = source.total
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("VidEoMT FSDP2 probe produced a non-finite loss")
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started

    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    gradient_receipt = _local_gradient_receipt(runtime.model)
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "loss": float(loss.detach()),
        "raw_losses": {
            name: float(value.detach()) for name, value in source.raw_losses.items()
        },
        "weighted_losses": {
            name: float(value.detach()) for name, value in source.weighted_losses.items()
        },
        "class_logits_shape": list(output.class_logits.shape),
        "mask_logits_shape": list(output.mask_logits.shape),
        "gradient_receipt": gradient_receipt,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "timing_seconds": {
            "load_and_parallelize": load_seconds,
            "forward": forward_seconds,
            "backward": backward_seconds,
        },
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    ranks = [value for value in gathered if value is not None]
    passed = len(ranks) == world_size and all(
        bool(value["gradient_receipt"]["passed"]) for value in ranks
    )
    if rank == 0:
        report = {
            "schema": "picf-next.videomt-exact-fsdp2-source-objective-probe/v2",
            "claim_scope": (
                "complete released VidEoMT, exact adapted weights, two-rank FSDP2 "
                "five-frame complete source-objective forward/backward placement "
                "closure; not learning quality"
            ),
            "world_size": world_size,
            "image_size": args.image_size,
            "fsdp2_receipt": asdict(receipt),
            "ranks": ranks,
            "passed": passed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
