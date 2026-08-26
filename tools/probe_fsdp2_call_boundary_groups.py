#!/usr/bin/env python3
"""Exercise LingBot-style bifurcated decoder calls under nested FSDP2.

This is a two-GPU execution probe, not a training implementation. It mirrors
the relevant ownership boundary used by LingBot-VLA2: Q/K/V and output/MLP are
called in separate phases, both phases use non-reentrant checkpointing, and
the existing projection modules are sharded before their parent decoder.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop


class _DenseMLP(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden_width, bias=False)
        self.up_proj = nn.Linear(width, hidden_width, bias=False)
        self.down_proj = nn.Linear(hidden_width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value)
        return self.down_proj(gated)


class _TokenMLP(nn.Module):
    """Small stand-in for the action expert's one-call token-MoE boundary."""

    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.router = nn.Linear(width, 2, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, hidden_width, bias=False),
                    nn.SiLU(),
                    nn.Linear(hidden_width, width, bias=False),
                )
                for _ in range(2)
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weights = self.router(value).softmax(dim=-1)
        expert_outputs = torch.stack(tuple(expert(value) for expert in self.experts), dim=-2)
        return (expert_outputs * weights.unsqueeze(-1)).sum(dim=-2)


class _BifurcatedBlock(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_width: int,
        *,
        token_mlp: bool,
        runtime_dtype_source: str,
        checkpoint_owner: str,
    ) -> None:
        super().__init__()
        if runtime_dtype_source not in {"parent", "projection"}:
            raise ValueError("runtime dtype source must be parent or projection")
        if checkpoint_owner not in {"external", "layer", "none"}:
            raise ValueError("checkpoint owner must be external, layer, or none")
        self.runtime_dtype_source = runtime_dtype_source
        self.checkpoint_owner = checkpoint_owner
        self.input_layernorm = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.mlp = _TokenMLP(width, hidden_width) if token_mlp else _DenseMLP(width, hidden_width)

    def __call__(self, *args, **kwargs):
        if self.checkpoint_owner == "layer" and self.training:
            checkpointed_call = partial(super().__call__, **kwargs)
            return checkpoint(checkpointed_call, *args, use_reentrant=False)
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        hidden: torch.Tensor,
        attention: torch.Tensor | None = None,
        *,
        compute_kqv: bool = False,
        output_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if compute_kqv == output_attention:
            raise ValueError("exactly one bifurcated decoder phase must be selected")
        runtime_weight = (
            self.input_layernorm.weight
            if self.runtime_dtype_source == "parent"
            else self.q_proj.weight
        )
        hidden = hidden.to(runtime_weight.dtype)
        if attention is not None:
            attention = attention.to(runtime_weight.dtype)
        if compute_kqv:
            normalized = self.input_layernorm(hidden)
            return (
                self.q_proj(normalized),
                self.k_proj(normalized),
                self.v_proj(normalized),
            )
        if attention is None:
            raise ValueError("output phase requires attention")
        residual = hidden + self.o_proj(attention)
        return residual + self.mlp(self.post_attention_layernorm(residual))


class _BifurcatedRoot(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_width: int,
        *,
        runtime_dtype_source: str,
        checkpoint_owner: str,
    ) -> None:
        super().__init__()
        self.checkpoint_owner = checkpoint_owner
        block_checkpoint_owner = "none" if checkpoint_owner == "wrapper" else checkpoint_owner
        text = _BifurcatedBlock(
            width,
            hidden_width,
            token_mlp=False,
            runtime_dtype_source=runtime_dtype_source,
            checkpoint_owner=block_checkpoint_owner,
        )
        action = _BifurcatedBlock(
            width,
            hidden_width,
            token_mlp=True,
            runtime_dtype_source=runtime_dtype_source,
            checkpoint_owner=block_checkpoint_owner,
        )
        self.text = checkpoint_wrapper(text) if checkpoint_owner == "wrapper" else text
        self.action = checkpoint_wrapper(action) if checkpoint_owner == "wrapper" else action
        self.output_scale = nn.Parameter(torch.ones(width))

    @staticmethod
    def _qkv(
        block: _BifurcatedBlock,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = block(hidden, compute_kqv=True)
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("bifurcated Q/K/V phase returned an invalid output")
        return output

    @staticmethod
    def _output(
        block: _BifurcatedBlock,
        hidden: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        output = block(hidden, attention, output_attention=True)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("bifurcated output phase returned an invalid output")
        return output

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_owner == "external":
            text_qkv = checkpoint(self._qkv, self.text, hidden, use_reentrant=False)
            action_qkv = checkpoint(self._qkv, self.action, hidden, use_reentrant=False)
        else:
            text_qkv = self._qkv(self.text, hidden)
            action_qkv = self._qkv(self.action, hidden)
        attention = torch.stack((*text_qkv, *action_qkv), dim=0).mean(dim=0)
        if self.checkpoint_owner == "external":
            text = checkpoint(
                self._output,
                self.text,
                hidden,
                attention,
                use_reentrant=False,
            )
            action = checkpoint(
                self._output,
                self.action,
                hidden,
                attention,
                use_reentrant=False,
            )
        else:
            text = self._output(self.text, hidden, attention)
            action = self._output(self.action, hidden, attention)
        return ((text + action) * self.output_scale).float().square().mean()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--hidden-width", type=int, default=1024)
    parser.add_argument(
        "--runtime-dtype-source",
        choices=("parent", "projection"),
        default="parent",
    )
    parser.add_argument(
        "--checkpoint-owner",
        choices=("external", "layer", "wrapper"),
        default="external",
    )
    parser.add_argument(
        "--checkpoint-early-stop",
        choices=("enabled", "disabled"),
        default="enabled",
    )
    parser.add_argument(
        "--fsdp-topology",
        choices=("nested", "block"),
        default="nested",
    )
    parser.add_argument("--unroll-frames", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def _require_environment() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2 or local_rank not in (0, 1):
        raise RuntimeError("probe requires torchrun with exactly two local CUDA ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("probe requires two visible CUDA devices")
    return world_size, local_rank


def _unwrap_checkpoint_block(module: nn.Module) -> _BifurcatedBlock:
    wrapped = getattr(module, "_checkpoint_wrapped_module", module)
    if not isinstance(wrapped, _BifurcatedBlock):
        raise TypeError("probe block has an unexpected checkpoint wrapper")
    return wrapped


def _execution_units(model: _BifurcatedRoot) -> dict[str, nn.Module]:
    text = _unwrap_checkpoint_block(model.text)
    action = _unwrap_checkpoint_block(model.action)
    units: dict[str, nn.Module] = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        units[f"text.{name}"] = getattr(text, name)
        units[f"action.{name}"] = getattr(action, name)
    for name in ("gate_proj", "up_proj", "down_proj"):
        units[f"text.mlp.{name}"] = getattr(text.mlp, name)
    units["action.mlp"] = action.mlp
    return units


def _fully_shard_execution_units(
    model: _BifurcatedRoot,
    *,
    world_size: int,
    fsdp_topology: str,
) -> dict[str, int]:
    mesh = init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("dp_shard",),
    )
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    common = {
        "mesh": mesh,
        "mp_policy": policy,
        "reshard_after_forward": True,
    }
    units = _execution_units(model)

    group_bytes: dict[str, int]
    if fsdp_topology == "nested":
        group_bytes = {}
        for path, unit in units.items():
            group_bytes[path] = sum(parameter.numel() for parameter in unit.parameters()) * 2
            fully_shard(unit, **common)
    elif fsdp_topology == "block":
        group_bytes = {
            path: sum(parameter.numel() for parameter in unit.parameters()) * 2
            for path, unit in (("text", model.text), ("action", model.action))
        }
    else:
        raise ValueError("FSDP topology must be nested or block")
    fully_shard(model.text, **common)
    fully_shard(model.action, **common)
    root_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=root_policy,
        reshard_after_forward=True,
    )

    sharded_units = (
        (*units.items(), ("text", model.text), ("action", model.action))
        if fsdp_topology == "nested"
        else (("text", model.text), ("action", model.action))
    )
    for path, unit in sharded_units:
        if not hasattr(unit, "reshard") or not hasattr(unit, "unshard"):
            raise RuntimeError(f"FSDP2 did not augment {path}")
    return group_bytes


def _gradient_statistics(model: _BifurcatedRoot) -> tuple[int, float]:
    tensors = 0
    absolute_sum = 0.0
    for path, unit in _execution_units(model).items():
        unit_tensors = 0
        unit_sum = 0.0
        for parameter in unit.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
            if not torch.isfinite(local).all():
                raise RuntimeError(f"call-boundary unit {path} produced a non-finite gradient")
            unit_tensors += 1
            unit_sum += float(local.float().abs().sum().item())
        if unit_tensors == 0 or unit_sum <= 0.0:
            raise RuntimeError(f"call-boundary unit {path} produced no finite nonzero gradients")
        tensors += unit_tensors
        absolute_sum += unit_sum
    if tensors == 0 or absolute_sum <= 0.0:
        raise RuntimeError("call-boundary probe produced no finite nonzero gradients")
    return tensors, absolute_sum


def _run(
    *,
    width: int,
    hidden_width: int,
    steps: int,
    world_size: int,
    local_rank: int,
    runtime_dtype_source: str,
    checkpoint_owner: str,
    checkpoint_early_stop: str,
    fsdp_topology: str,
    unroll_frames: int,
) -> dict[str, object]:
    if width <= 0 or hidden_width <= 0 or steps <= 0 or unroll_frames <= 0:
        raise ValueError("probe dimensions and steps must be positive")
    if checkpoint_owner not in {"external", "layer", "wrapper"}:
        raise ValueError("checkpoint owner must be external, layer, or wrapper")
    if checkpoint_early_stop not in {"enabled", "disabled"}:
        raise ValueError("checkpoint early-stop must be enabled or disabled")
    if checkpoint_owner == "wrapper" and fsdp_topology != "block":
        raise ValueError("official checkpoint wrapper probe requires block FSDP topology")
    torch.manual_seed(20260727)
    device = torch.device("cuda", local_rank)
    model = (
        _BifurcatedRoot(
            width,
            hidden_width,
            runtime_dtype_source=runtime_dtype_source,
            checkpoint_owner=checkpoint_owner,
        )
        .to(device=device)
        .train()
    )
    group_bytes = _fully_shard_execution_units(
        model,
        world_size=world_size,
        fsdp_topology=fsdp_topology,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    losses: list[float] = []
    gradient_tensors: list[int] = []
    gradient_sums: list[float] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.linspace(
            -1.0,
            1.0,
            steps=width,
            dtype=torch.bfloat16,
            device=device,
        ).view(1, 1, width)
        hidden = hidden.expand(2, 3, -1) + (step / 100.0)
        checkpoint_context = (
            contextlib.nullcontext()
            if checkpoint_early_stop == "enabled"
            else set_checkpoint_early_stop(False)
        )
        with checkpoint_context:
            frame_losses = tuple(
                model(hidden + (frame_index / 1000.0)) for frame_index in range(unroll_frames)
            )
            loss = torch.stack(frame_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("call-boundary probe produced a non-finite loss")
        loss.backward()
        tensor_count, gradient_sum = _gradient_statistics(model)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        gradient_tensors.append(tensor_count)
        gradient_sums.append(gradient_sum)

    torch.cuda.synchronize(device)
    final_loss = torch.tensor(losses[-1], dtype=torch.float64, device=device)
    minimum_loss = final_loss.clone()
    maximum_loss = final_loss.clone()
    dist.all_reduce(minimum_loss, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum_loss, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum_loss, maximum_loss):
        raise RuntimeError("call-boundary probe loss differs across ranks")
    status = torch.tensor(1, dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) != 1:
        raise RuntimeError("call-boundary probe status differs across ranks")
    return {
        "checkpoint_early_stop": checkpoint_early_stop,
        "checkpoint_owner": checkpoint_owner,
        "elapsed_seconds": time.monotonic() - started,
        "fsdp_topology": fsdp_topology,
        "gradient_absolute_sums": gradient_sums,
        "gradient_tensor_counts": gradient_tensors,
        "losses": losses,
        "maximum_declared_group_bf16_bytes": max(group_bytes.values()),
        "passed": True,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "runtime_dtype_source": runtime_dtype_source,
        "schema": "picf-next.fsdp2-call-boundary-probe.v1",
        "steps": steps,
        "torch": torch.__version__,
        "unroll_frames": unroll_frames,
        "unit_count": len(group_bytes),
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
            width=args.width,
            hidden_width=args.hidden_width,
            steps=args.steps,
            world_size=world_size,
            local_rank=local_rank,
            runtime_dtype_source=args.runtime_dtype_source,
            checkpoint_owner=args.checkpoint_owner,
            checkpoint_early_stop=args.checkpoint_early_stop,
            fsdp_topology=args.fsdp_topology,
            unroll_frames=args.unroll_frames,
        )
        dist.barrier()
        if local_rank == 0:
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
