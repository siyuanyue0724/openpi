"""Exact sharded-gradient snapshots and pair moments for read-only audits."""

from __future__ import annotations

from typing import Any

_CPU_REDUCTION_CHUNK_ELEMENTS = 4 * 1024 * 1024


def local_finite_gradient(parameter: Any, *, torch_module: Any) -> Any:
    """Return one local gradient shard after strict existence/finite checks."""

    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError("native VL gradient audit found a missing trainable gradient")
    local = gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
    if not bool(torch_module.isfinite(local).all()):
        raise FloatingPointError("native VL gradient audit found a non-finite gradient")
    return local


def snapshot_local_gradients(model: Any, *, torch_module: Any) -> dict[str, Any]:
    """Copy every trainable local shard to contiguous CPU float32 storage."""

    snapshot = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError("native VL gradient audit found a frozen-parameter gradient")
            continue
        local = local_finite_gradient(parameter, torch_module=torch_module)
        snapshot[name] = (
            local.detach()
            .to(
                device="cpu",
                dtype=torch_module.float32,
            )
            .contiguous()
        )
    if not snapshot:
        raise RuntimeError("native VL gradient audit captured no trainable gradients")
    return snapshot


def cpu_pair_moments(first: Any, second: Any, *, torch_module: Any) -> tuple[float, ...]:
    """Accumulate one tensor pair in bounded CPU chunks using float64 sums."""

    if first.device.type != "cpu" or second.device.type != "cpu":
        raise RuntimeError("native VL gradient audit moments require CPU tensors")
    if (
        first.shape != second.shape
        or first.dtype != second.dtype
        or first.dtype != torch_module.float32
    ):
        raise RuntimeError("native VL gradient audit snapshots are incompatible")
    left = first.reshape(-1)
    right = second.reshape(-1)
    dot = 0.0
    left_squared = 0.0
    right_squared = 0.0
    for start in range(0, left.numel(), _CPU_REDUCTION_CHUNK_ELEMENTS):
        stop = min(start + _CPU_REDUCTION_CHUNK_ELEMENTS, left.numel())
        left_chunk = left[start:stop]
        right_chunk = right[start:stop]
        dot += float((left_chunk * right_chunk).sum(dtype=torch_module.float64).item())
        left_squared += float(left_chunk.square().sum(dtype=torch_module.float64).item())
        right_squared += float(right_chunk.square().sum(dtype=torch_module.float64).item())
    return dot, left_squared, right_squared, float(left.numel())


def distributed_pair_rows(
    model: Any,
    *,
    first_gradients: dict[str, Any],
    device: Any,
    dist: Any,
    torch_module: Any,
) -> tuple[tuple[str, ...], list[list[float]]]:
    """All-reduce exact pair moments for every declared trainable parameter."""

    named_parameters = {name: parameter for name, parameter in model.named_parameters()}
    current_trainable = {
        name for name, parameter in named_parameters.items() if parameter.requires_grad
    }
    if set(first_gradients) != current_trainable:
        raise RuntimeError("native VL gradient audit trainable scope changed between objectives")
    names = tuple(sorted(first_gradients))
    local_rows = []
    for name in names:
        parameter = named_parameters[name]
        if not parameter.requires_grad:
            raise RuntimeError(
                "native VL gradient audit trainable scope changed between objectives"
            )
        second = (
            local_finite_gradient(parameter, torch_module=torch_module)
            .detach()
            .to(device="cpu", dtype=torch_module.float32)
            .contiguous()
        )
        local_rows.append(
            cpu_pair_moments(first_gradients[name], second, torch_module=torch_module)
        )
        del second
    packed = torch_module.tensor(local_rows, dtype=torch_module.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return names, packed.detach().cpu().tolist()
