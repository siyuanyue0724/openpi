#!/usr/bin/env python3
"""Two-rank probe for VidEoMT's released distributed loss normalization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from picf_next.videomt_exact.class_agnostic import (
    build_class_agnostic_criterion,
    flatten_class_agnostic_outputs,
    flatten_class_agnostic_targets,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput
from picf_next.videomt_exact.training import apply_released_loss_weights

NUM_FRAMES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _rank_case(rank: int) -> tuple[ExactVidEoMTOutput, list[dict[str, torch.Tensor]]]:
    generator = torch.Generator().manual_seed(198 + rank)
    class_logits = torch.randn(
        1, NUM_FRAMES, 200, 41, generator=generator, requires_grad=True
    )
    mask_logits = torch.randn(
        1, 200, NUM_FRAMES, 8, 8, generator=generator, requires_grad=True
    )
    output = ExactVidEoMTOutput(
        class_logits=class_logits,
        mask_logits=mask_logits,
        query_embeddings=torch.randn(1, NUM_FRAMES, 200, 1024, generator=generator),
        propagated_queries=torch.randn(1, 200, 1024, generator=generator),
        auxiliary_outputs=(),
    )
    object_count = rank + 1
    masks = torch.zeros(object_count, NUM_FRAMES, 16, 16)
    for index in range(object_count):
        offset = 2 + index * 5
        masks[index, :, offset : offset + 4, offset : offset + 4] = 1
    targets = [
        {
            "labels": torch.zeros(object_count, dtype=torch.long),
            "ids": torch.arange(object_count, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, NUM_FRAMES),
            "masks": masks,
        }
    ]
    return output, targets


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("distributed VidEoMT criterion probe requires exactly two ranks")

    output, targets = _rank_case(rank)
    criterion = build_class_agnostic_criterion(num_frames=NUM_FRAMES)
    raw = criterion(
        flatten_class_agnostic_outputs(output),
        flatten_class_agnostic_targets(targets),
    )
    weighted = apply_released_loss_weights(raw, criterion)
    total = sum(weighted.values())
    total.backward()
    local = {
        "rank": rank,
        "pid": os.getpid(),
        "object_count": len(targets[0]["labels"]),
        "raw_losses": {name: float(value.detach()) for name, value in raw.items()},
        "weighted_loss": float(total.detach()),
        "class_gradient_finite": bool(
            output.class_logits.grad is not None
            and torch.isfinite(output.class_logits.grad).all()
            and output.class_logits.grad.abs().sum() > 0
        ),
        "mask_gradient_finite": bool(
            output.mask_logits.grad is not None
            and torch.isfinite(output.mask_logits.grad).all()
            and output.mask_logits.grad.abs().sum() > 0
        ),
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank == 0:
        ranks = [value for value in gathered if value is not None]
        report = {
            "schema": "picf-next.videomt-distributed-criterion-probe.v1",
            "claim_scope": (
                "two-rank released criterion all-reduce and gradient closure; "
                "not full-model FSDP"
            ),
            "backend": dist.get_backend(),
            "world_size": world_size,
            "ranks": ranks,
            "passed": len(ranks) == world_size
            and all(
                bool(value["class_gradient_finite"])
                and bool(value["mask_gradient_finite"])
                for value in ranks
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
