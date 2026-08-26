#!/usr/bin/env python3
"""Reproduce the FSDP2 contract for differentiable root side outputs.

Run with exactly two CUDA ranks. ``hidden`` is a negative control whose process
must fail during backward; CUDA may report the storage-lifetime violation
asynchronously and abort the rank. ``explicit`` is the machine-verifiable
positive control: the same tensor is in the root output pytree, and backward
must succeed with finite gradients.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint

Mode = Literal["hidden", "explicit"]


class _CheckpointedBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.linear(value))


class _SideOutputRoot(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = _CheckpointedBlock(width)
        self.side_scale = nn.Parameter(torch.ones(width))
        self.side_output: torch.Tensor | None = None

    def forward(
        self,
        value: torch.Tensor,
        *,
        expose_side_output: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = checkpoint(self.block, value, use_reentrant=False)
        side = hidden * self.side_scale.unsqueeze(0)
        self.side_output = side
        official = hidden.square().mean()
        if expose_side_output:
            return official, side
        return official


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hidden", "explicit"), required=True)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--forwards", type=int, default=4)
    return parser.parse_args()


def _require_environment() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2 or local_rank not in (0, 1):
        raise RuntimeError("probe requires torchrun with exactly two local CUDA ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("probe requires two visible CUDA devices")
    return world_size, local_rank


def _build_model(*, width: int, world_size: int, local_rank: int) -> _SideOutputRoot:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = _SideOutputRoot(width).to(device=device, dtype=torch.bfloat16).train()
    mesh = init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("dp_shard",),
    )
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    fully_shard(
        model.block,
        mesh=mesh,
        mp_policy=mixed_precision,
        reshard_after_forward=True,
    )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mixed_precision,
        reshard_after_forward=True,
    )
    return model


def _run(mode: Mode, *, width: int, forwards: int, world_size: int, local_rank: int) -> dict:
    if width <= 0 or forwards < 2:
        raise ValueError("probe width must be positive and forwards must be at least two")
    model = _build_model(width=width, world_size=world_size, local_rank=local_rank)
    device = torch.device("cuda", local_rank)
    losses: list[torch.Tensor] = []
    for index in range(forwards):
        value = torch.full(
            (1, width),
            float(index + 1) / forwards,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        output = model(value, expose_side_output=mode == "explicit")
        if mode == "explicit":
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("explicit probe root omitted its side output")
            side = output[1]
        else:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("hidden probe root changed its official output")
            side = model.side_output
            if side is None:
                raise RuntimeError("hidden probe root did not materialize its side output")
        losses.append(side.float().square().mean())
    loss = torch.stack(losses).mean()

    if mode == "hidden":
        loss.backward()
        raise RuntimeError("hidden side-output backward unexpectedly succeeded")

    loss.backward()
    gradient = model.side_scale.grad
    if gradient is None:
        raise RuntimeError("explicit side-output backward produced no root gradient")
    local_gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
    gradient_finite = bool(torch.isfinite(local_gradient).all().item())
    if not gradient_finite or not bool((local_gradient != 0).any().item()):
        raise RuntimeError("explicit side-output backward produced an invalid root gradient")

    status = torch.tensor(1, dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) != 1:
        raise RuntimeError("FSDP2 side-output result differs across ranks")
    return {
        "forwards": forwards,
        "gradient_finite": gradient_finite,
        "mode": mode,
        "passed": True,
        "schema": "picf-next.fsdp2-root-output-visibility-probe.v1",
        "torch": torch.__version__,
        "width": width,
        "world_size": world_size,
    }


def main() -> None:
    args = _parse_args()
    world_size, local_rank = _require_environment()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        report = _run(
            args.mode,
            width=args.width,
            forwards=args.forwards,
            world_size=world_size,
            local_rank=local_rank,
        )
        dist.barrier()
        if local_rank == 0:
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
