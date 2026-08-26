#!/usr/bin/env python3
"""Probe rank-local objective connectivity under FSDP2.

Every rank executes the same forward graph. The probe changes only whether an
optional objective is omitted, retained normally, or retained with an exact
zero coefficient on rank 1. It diagnoses the collective contract required when
different samples expose different supervision families.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


class _ConditionalObjectiveProbe(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_width: int,
        *,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden_width, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_width, width, bias=False),
        )
        self.required_head = nn.Linear(width, width, bias=False)
        self.optional_head = nn.Linear(width, width, bias=False)
        self.output_scale = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = (
            checkpoint(self.encoder, value, use_reentrant=False)
            if self.activation_checkpointing
            else self.encoder(value)
        )
        required = self.required_head(hidden)
        optional = self.optional_head(hidden)
        scale = self.output_scale
        return (
            (required * scale).float().square().mean(),
            (optional * scale).float().square().mean(),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--connectivity",
        choices=(
            "matched",
            "rank-detached",
            "rank-zero",
            "later-detached",
            "later-zero",
        ),
        required=True,
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--hidden-width", type=int, default=512)
    parser.add_argument("--unroll-calls", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--checkpoint-schedule",
        choices=("default", "complete-recompute"),
        default="default",
    )
    return parser.parse_args()


def _require_environment() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2 or local_rank not in (0, 1):
        raise RuntimeError("probe requires torchrun with exactly two local CUDA ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("probe requires two visible CUDA devices")
    return world_size, local_rank


def _build_model(
    *,
    width: int,
    hidden_width: int,
    world_size: int,
    device: torch.device,
    activation_checkpointing: bool,
) -> _ConditionalObjectiveProbe:
    torch.manual_seed(20260727)
    model = _ConditionalObjectiveProbe(
        width,
        hidden_width,
        activation_checkpointing=activation_checkpointing,
    ).to(device=device).train()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    options: dict[str, Any] = {
        "mesh": mesh,
        "mp_policy": policy,
        "reshard_after_forward": True,
    }
    fully_shard(model.encoder, **options)
    fully_shard(model.required_head, **options)
    fully_shard(model.optional_head, **options)
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        ),
        reshard_after_forward=True,
    )
    for name, module in (
        ("encoder", model.encoder),
        ("required_head", model.required_head),
        ("optional_head", model.optional_head),
        ("root", model),
    ):
        if not hasattr(module, "reshard") or not hasattr(module, "unshard"):
            raise RuntimeError(f"FSDP2 did not augment {name}")
    return model


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def _gradient_report(model: nn.Module) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for name, module in (
        ("encoder", model.encoder),
        ("required_head", model.required_head),
        ("optional_head", model.optional_head),
        ("root", model),
    ):
        tensors = 0
        absolute_sum = 0.0
        for parameter in module.parameters(recurse=name != "root"):
            if parameter.grad is None:
                continue
            gradient = _local_tensor(parameter.grad)
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"{name} produced a non-finite gradient")
            tensors += 1
            absolute_sum += float(gradient.float().abs().sum().item())
        report[name] = {
            "absolute_sum": absolute_sum,
            "tensor_count": tensors,
        }
    for required in ("encoder", "required_head", "optional_head"):
        if report[required]["tensor_count"] <= 0:
            raise RuntimeError(f"{required} produced no gradient tensor")
    return report


def _objective(
    *,
    required: torch.Tensor,
    optional: torch.Tensor,
    connectivity: str,
    local_rank: int,
    call_index: int,
) -> torch.Tensor:
    if connectivity == "matched":
        return required + optional
    if connectivity == "rank-detached" and local_rank == 1:
        return required
    if connectivity == "rank-zero" and local_rank == 1:
        return required + optional * 0
    if connectivity == "later-detached" and call_index > 0:
        return required
    if connectivity == "later-zero" and call_index > 0:
        return required + optional * 0
    if connectivity in {
        "rank-detached",
        "rank-zero",
        "later-detached",
        "later-zero",
    }:
        return required + optional
    raise ValueError("unknown objective connectivity")


def _run(
    *,
    connectivity: str,
    width: int,
    hidden_width: int,
    unroll_calls: int,
    steps: int,
    world_size: int,
    local_rank: int,
    activation_checkpointing: bool = False,
    checkpoint_schedule: str = "default",
) -> dict[str, Any]:
    if width <= 0 or hidden_width <= 0 or unroll_calls <= 0 or steps <= 0:
        raise ValueError("probe dimensions and steps must be positive")
    device = torch.device("cuda", local_rank)
    model = _build_model(
        width=width,
        hidden_width=hidden_width,
        world_size=world_size,
        device=device,
        activation_checkpointing=activation_checkpointing,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    step_reports: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        base = torch.linspace(
            -1,
            1,
            steps=width,
            dtype=torch.bfloat16,
            device=device,
        ).view(1, 1, width)
        checkpoint_context = (
            set_checkpoint_early_stop(False)
            if checkpoint_schedule == "complete-recompute"
            else nullcontext()
        )
        with checkpoint_context:
            call_losses = []
            for call_index in range(unroll_calls):
                required, optional = model(base + (step * unroll_calls + call_index) / 1000)
                call_losses.append(
                    _objective(
                        required=required,
                        optional=optional,
                        connectivity=connectivity,
                        local_rank=local_rank,
                        call_index=call_index,
                    )
                )
        loss = torch.stack(call_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("probe produced a non-finite loss")
        loss.backward()
        gradients = _gradient_report(model)
        optimizer.step()
        step_reports.append(
            {
                "gradients": gradients,
                "loss": float(loss.detach().item()),
            }
        )
    torch.cuda.synchronize(device)
    result = {
        "activation_checkpointing": activation_checkpointing,
        "checkpoint_schedule": checkpoint_schedule,
        "connectivity": connectivity,
        "elapsed_seconds": time.monotonic() - started,
        "local_rank": local_rank,
        "passed": True,
        "schema": "picf-next.fsdp2-rank-local-objective-probe.v1",
        "step_reports": step_reports,
        "steps": steps,
        "torch": torch.__version__,
        "unroll_calls": unroll_calls,
        "world_size": world_size,
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, result)
    if any(item is None or not item["passed"] for item in gathered):
        raise RuntimeError("probe status differs across ranks")
    result["rank_reports"] = gathered
    return result


def main() -> None:
    args = _parse_args()
    world_size, local_rank = _require_environment()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        device_id=device,
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    try:
        report = _run(
            connectivity=args.connectivity,
            width=args.width,
            hidden_width=args.hidden_width,
            unroll_calls=args.unroll_calls,
            steps=args.steps,
            world_size=world_size,
            local_rank=local_rank,
            activation_checkpointing=args.activation_checkpointing,
            checkpoint_schedule=args.checkpoint_schedule,
        )
        dist.barrier()
        if local_rank == 0:
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
