#!/usr/bin/env python3
"""Probe one selectively CPU-offloaded embedding group under FSDP2.

This is an execution and checkpoint probe, not a training implementation. It
keeps the body and root parameters GPU-sharded while optionally placing only
the shared embedding's parameter shard, gradient shard, and optimizer state on
CPU through PyTorch's public ``CPUOffloadPolicy``. Multiple calls before one
backward exercise the temporal-objective reuse boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)


class _EmbeddingProbe(nn.Module):
    def __init__(self, vocab_size: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width, bias=False),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
        )
        self.output_scale = nn.Parameter(torch.ones(width))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        hidden = self.body(hidden)
        return (hidden * self.output_scale).float().square().mean()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-placement",
        choices=("cpu", "gpu", "root"),
        required=True,
    )
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--width", type=int, default=2_560)
    parser.add_argument("--token-count", type=int, default=514)
    parser.add_argument("--unroll-calls", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def _require_environment(checkpoint_dir: Path) -> tuple[int, int, Path]:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size != 2 or local_rank not in (0, 1):
        raise RuntimeError("probe requires torchrun with exactly two local CUDA ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("probe requires two visible CUDA devices")
    if not checkpoint_dir.is_absolute():
        raise RuntimeError("--checkpoint-dir must be absolute")
    return world_size, local_rank, checkpoint_dir.resolve()


def _build_model(
    *,
    vocab_size: int,
    width: int,
    world_size: int,
    device: torch.device,
    embedding_placement: str,
) -> _EmbeddingProbe:
    torch.manual_seed(20260727)
    model = _EmbeddingProbe(vocab_size, width).to(device=device).train()
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
    if embedding_placement != "root":
        embedding_options: dict[str, Any] = {
            "mesh": mesh,
            "mp_policy": mixed_precision,
            "reshard_after_forward": True,
        }
        if embedding_placement == "cpu":
            embedding_options["offload_policy"] = CPUOffloadPolicy(pin_memory=False)
        fully_shard(model.embedding, **embedding_options)
    fully_shard(
        model.body,
        mesh=mesh,
        mp_policy=mixed_precision,
        reshard_after_forward=True,
    )
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
    sharded_modules = [("body", model.body), ("root", model)]
    if embedding_placement != "root":
        sharded_modules.insert(0, ("embedding", model.embedding))
    for name, module in sharded_modules:
        if not hasattr(module, "reshard") or not hasattr(module, "unshard"):
            raise RuntimeError(f"FSDP2 did not augment {name}")
    return model


def _token_ids(
    *,
    vocab_size: int,
    token_count: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    if vocab_size <= 0 or token_count <= 0:
        raise ValueError("vocab size and token count must be positive")
    return (
        torch.arange(token_count, dtype=torch.int64, device=device) * 7919 + step * 104729
    ).remainder(vocab_size)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def _finite_gradient_sum(model: nn.Module) -> float:
    absolute_sum = 0.0
    tensor_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = _local_tensor(parameter.grad)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("probe produced a non-finite gradient")
        absolute_sum += float(gradient.float().abs().sum().item())
        tensor_count += 1
    if tensor_count == 0 or absolute_sum <= 0.0:
        raise RuntimeError("probe produced no finite nonzero gradients")
    return absolute_sum


def _parameter_checksum(model: nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros(2, dtype=torch.float64, device=device)
    for index, parameter in enumerate(model.parameters(), start=1):
        local = _local_tensor(parameter.detach()).to(device=device, dtype=torch.float64)
        checksum[0] += local.sum()
        checksum[1] += local.square().sum() * index
    dist.all_reduce(checksum)
    return checksum


def _run_step(
    model: _EmbeddingProbe,
    optimizer: torch.optim.Optimizer,
    *,
    vocab_size: int,
    token_count: int,
    unroll_calls: int,
    step: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    call_losses = tuple(
        model(
            _token_ids(
                vocab_size=vocab_size,
                token_count=token_count,
                step=step * unroll_calls + call_index,
                device=device,
            )
        )
        for call_index in range(unroll_calls)
    )
    loss = torch.stack(call_losses).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("probe produced a non-finite loss")
    loss.backward()
    gradient_sum = _finite_gradient_sum(model)
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    return {
        "elapsed_seconds": elapsed,
        "gradient_absolute_sum": gradient_sum,
        "loss": float(loss.detach().item()),
        "parameter_checksum": _parameter_checksum(model, device).tolist(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    *,
    local_rank: int,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    if local_rank == 0:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)
    dist.barrier()
    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=str(checkpoint_dir),
    )
    dist.barrier()


def _load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=str(checkpoint_dir))
    incompatible = set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint load is incompatible: {incompatible}")


def _storage_report(model: _EmbeddingProbe) -> dict[str, str]:
    return {
        "body": _local_tensor(model.body[1].weight).device.type,
        "embedding": _local_tensor(model.embedding.weight).device.type,
        "root": _local_tensor(model.output_scale).device.type,
    }


def _build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,
    )


def _run(
    *,
    embedding_placement: str,
    vocab_size: int,
    width: int,
    token_count: int,
    unroll_calls: int,
    checkpoint_dir: Path,
    world_size: int,
    local_rank: int,
) -> dict[str, Any]:
    if vocab_size <= 0 or width <= 0 or token_count <= 0 or unroll_calls <= 0:
        raise ValueError("probe dimensions must be positive")
    device = torch.device("cuda", local_rank)
    source_model = _build_model(
        vocab_size=vocab_size,
        width=width,
        world_size=world_size,
        device=device,
        embedding_placement=embedding_placement,
    )
    source_storage = _storage_report(source_model)
    expected_embedding_device = "cpu" if embedding_placement == "cpu" else "cuda"
    if source_storage != {
        "body": "cuda",
        "embedding": expected_embedding_device,
        "root": "cuda",
    }:
        raise RuntimeError(f"unexpected selective placement: {source_storage}")
    source_optimizer = _build_optimizer(source_model)

    torch.cuda.reset_peak_memory_stats(device)
    fresh = _run_step(
        source_model,
        source_optimizer,
        vocab_size=vocab_size,
        token_count=token_count,
        unroll_calls=unroll_calls,
        step=0,
        device=device,
    )
    _save_checkpoint(
        source_model,
        source_optimizer,
        checkpoint_dir,
        local_rank=local_rank,
    )
    torch.cuda.reset_peak_memory_stats(device)
    uninterrupted = _run_step(
        source_model,
        source_optimizer,
        vocab_size=vocab_size,
        token_count=token_count,
        unroll_calls=unroll_calls,
        step=1,
        device=device,
    )

    del source_optimizer, source_model
    torch.cuda.empty_cache()

    resumed_model = _build_model(
        vocab_size=vocab_size,
        width=width,
        world_size=world_size,
        device=device,
        embedding_placement=embedding_placement,
    )
    resumed_optimizer = _build_optimizer(resumed_model)
    # PyTorch 2.8 requires optimizer-state tensor templates before DCP load.
    _run_step(
        resumed_model,
        resumed_optimizer,
        vocab_size=vocab_size,
        token_count=token_count,
        unroll_calls=unroll_calls,
        step=0,
        device=device,
    )
    _load_checkpoint(resumed_model, resumed_optimizer, checkpoint_dir)
    resumed_storage = _storage_report(resumed_model)
    if resumed_storage != source_storage:
        raise RuntimeError(
            f"checkpoint changed selective placement: {source_storage} -> {resumed_storage}"
        )
    torch.cuda.reset_peak_memory_stats(device)
    resumed = _run_step(
        resumed_model,
        resumed_optimizer,
        vocab_size=vocab_size,
        token_count=token_count,
        unroll_calls=unroll_calls,
        step=1,
        device=device,
    )

    for field in ("loss", "gradient_absolute_sum"):
        torch.testing.assert_close(
            torch.tensor(uninterrupted[field], dtype=torch.float64),
            torch.tensor(resumed[field], dtype=torch.float64),
            rtol=0,
            atol=0,
            msg=lambda message, name=field: f"{name} changed after resume: {message}",
        )
    torch.testing.assert_close(
        torch.tensor(uninterrupted["parameter_checksum"], dtype=torch.float64),
        torch.tensor(resumed["parameter_checksum"], dtype=torch.float64),
        rtol=0,
        atol=0,
        msg="parameter checksum changed after resume",
    )

    result = {
        "checkpoint_dir": str(checkpoint_dir),
        "embedding_bf16_bytes": vocab_size * width * 2,
        "embedding_placement": embedding_placement,
        "fresh": fresh,
        "passed": True,
        "resumed": resumed,
        "schema": "picf-next.fsdp2-selective-embedding-offload-probe.v1",
        "storage": resumed_storage,
        "torch": torch.__version__,
        "unroll_calls": unroll_calls,
        "uninterrupted": uninterrupted,
        "vocab_size": vocab_size,
        "width": width,
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
    world_size, local_rank, checkpoint_dir = _require_environment(args.checkpoint_dir)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    try:
        report = _run(
            embedding_placement=args.embedding_placement,
            vocab_size=args.vocab_size,
            width=args.width,
            token_count=args.token_count,
            unroll_calls=args.unroll_calls,
            checkpoint_dir=checkpoint_dir,
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
