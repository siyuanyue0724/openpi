#!/usr/bin/env python3
"""Two-rank regression for exact FSDP2 factual-gradient CPU spill."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard

from picf_next.lingbot_native.fsdp2_placement import (
    merge_fsdp2_factual_gradients_from_cpu,
    spill_fsdp2_factual_gradients_to_cpu,
)


class _BranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(16, 16, bias=False)
        self.factual_only = nn.Linear(16, 16, bias=False)
        self.omitted_only = nn.Linear(16, 16, bias=False)
        self.never_used = nn.Parameter(torch.ones(17))

    def forward(self, value: torch.Tensor, branch: str) -> torch.Tensor:
        output = self.shared(value)
        if branch in {"factual", "both"}:
            output = output + self.factual_only(value)
        if branch in {"omitted", "both"}:
            output = output + self.omitted_only(value)
        return output


def _local(value: object) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    local = to_local() if callable(to_local) else value
    if not isinstance(local, torch.Tensor):
        raise TypeError("gradient probe expected a tensor")
    return local


def _gradient_copies(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: _local(parameter.grad).detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(17)

    model = _BranchModel().to(device)
    fully_shard(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    value = torch.arange(64, device=device, dtype=torch.float32).reshape(4, 16) / 64

    model(value, "both").square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model(value, "factual").square().mean().backward()
    spill = spill_fsdp2_factual_gradients_to_cpu(model)
    factual = {shard.parameter_name: shard.local_gradient for shard in spill.shards}
    if spill.distributed_shard_count != len(spill.shards):
        raise RuntimeError("probe factual gradients were not FSDP2 DTensors")
    if spill.cuda_source_bytes <= 0:
        raise RuntimeError("probe factual spill released no CUDA gradient bytes")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("probe factual gradients remained attached after spill")

    model(value, "omitted").square().mean().backward()
    omitted = _gradient_copies(model)
    merge = merge_fsdp2_factual_gradients_from_cpu(model, spill, chunk_bytes=256)
    merged = _gradient_copies(model)
    for name in set(factual) | set(omitted):
        reference = factual.get(name, torch.zeros_like(merged[name])) + omitted.get(
            name,
            torch.zeros_like(merged[name]),
        )
        if not torch.equal(merged[name], reference):
            raise RuntimeError(f"probe gradient merge differs for {name}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print(
            json.dumps(
                {
                    "cuda_source_bytes": spill.cuda_source_bytes,
                    "merge": merge,
                    "schema": "picf-next.fsdp2-gradient-spill-probe/v2",
                    "status": "PASS",
                    "world_size": dist.get_world_size(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
