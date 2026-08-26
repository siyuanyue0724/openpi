"""Explicit FSDP2 placement contracts for LingBot-native execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

FSDP2_CPU_OFFLOAD = "cpu-offload"
FSDP2_GPU_SHARDED = "gpu-sharded"
FSDP2_SELECTIVE_EMBEDDING_OFFLOAD = "selective-embedding-offload"
FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD = (
    "selective-embedding-frozen-vision-offload"
)
FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD = (
    "selective-embedding-trainable-vision-offload"
)
FSDP2_PLACEMENTS = (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
)
FSDP2_BACKWARD_PREFETCH_DEFAULT = "default"
FSDP2_BACKWARD_PREFETCH_DISABLED = "disabled"
FSDP2_BACKWARD_PREFETCH_MODES = (
    FSDP2_BACKWARD_PREFETCH_DEFAULT,
    FSDP2_BACKWARD_PREFETCH_DISABLED,
)
FSDP2_FACTUAL_GRADIENT_GPU = "gpu"
FSDP2_FACTUAL_GRADIENT_CPU = "cpu"
FSDP2_FACTUAL_GRADIENT_STORAGE_MODES = (
    FSDP2_FACTUAL_GRADIENT_GPU,
    FSDP2_FACTUAL_GRADIENT_CPU,
)

SELECTIVE_EMBEDDING_MODULE = "model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens"
SELECTIVE_EMBEDDING_PARAMETER = f"{SELECTIVE_EMBEDDING_MODULE}.weight"
SELECTIVE_FROZEN_VISION_MODULE = "model.qwenvl_with_expert.qwenvl.model.visual"
SELECTIVE_FROZEN_VISION_MODULE_PREFIX = (
    f"{SELECTIVE_FROZEN_VISION_MODULE}."
)
FSDP2_STORAGE_FIELDS = frozenset(
    {
        "parameter_tensors",
        "local_elements",
        "master_dtype",
        "placement",
        "cpu_parameter_tensors",
        "cpu_local_elements",
        "cuda_parameter_tensors",
        "cuda_local_elements",
        "selective_cpu_parameter_names",
    }
)


@dataclass(frozen=True, slots=True)
class FSDP2GradientLayout:
    """Exact local and global identity of one parameter gradient shard."""

    parameter_id: int
    distributed: bool
    global_shape: tuple[int, ...]
    global_stride: tuple[int, ...]
    local_shape: tuple[int, ...]
    local_stride: tuple[int, ...]
    dtype: torch.dtype
    source_device: str
    source_device_type: str
    mesh_device_type: str | None
    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    mesh_ranks: tuple[int, ...]
    mesh_coordinate: tuple[int, ...]
    placements: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FSDP2CPUGradientShard:
    """One exact rank-local factual gradient copied to CPU."""

    parameter_name: str
    local_gradient: torch.Tensor
    layout: FSDP2GradientLayout


@dataclass(frozen=True, slots=True)
class FSDP2CPUGradientSpill:
    """All locally present factual gradients for one deferred branch."""

    shards: tuple[FSDP2CPUGradientShard, ...]
    total_bytes: int
    cuda_source_bytes: int
    distributed_shard_count: int
    manifest_sha256: str


def _local_gradient(value: object) -> tuple[torch.Tensor, bool]:
    to_local = getattr(value, "to_local", None)
    distributed = callable(to_local)
    local = to_local() if distributed else value
    if not isinstance(local, torch.Tensor):
        raise TypeError("FSDP2 gradient local value must be a tensor")
    if local.is_sparse:
        raise RuntimeError("FSDP2 factual gradient spill does not support sparse gradients")
    return local, distributed


def _placement_signature(value: object) -> str:
    name = type(value).__name__
    if name in {"Shard", "_StridedShard"}:
        dimension = getattr(value, "dim", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise RuntimeError("FSDP2 shard placement omitted its dimension")
        if name == "_StridedShard":
            split_factor = getattr(value, "split_factor", None)
            if isinstance(split_factor, bool) or not isinstance(split_factor, int):
                raise RuntimeError("FSDP2 strided shard omitted its split factor")
            return f"{name}(dim={dimension},split_factor={split_factor})"
        return f"{name}(dim={dimension})"
    if name == "Replicate":
        return name
    if name == "Partial":
        raise RuntimeError("FSDP2 factual gradient spill rejects Partial placements")
    raise RuntimeError(f"FSDP2 factual gradient spill received placement {name!r}")


def _dtensor_spec(value: object) -> tuple[
    str,
    tuple[int, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    mesh = getattr(value, "device_mesh", None)
    placements = getattr(value, "placements", None)
    if mesh is None or placements is None:
        raise RuntimeError("FSDP2 DTensor gradient omitted mesh or placement metadata")
    mesh_tensor = getattr(mesh, "mesh", None)
    device_type = getattr(mesh, "device_type", None)
    if not isinstance(mesh_tensor, torch.Tensor) or not isinstance(device_type, str):
        raise RuntimeError("FSDP2 DTensor gradient mesh metadata is malformed")
    names = getattr(mesh, "mesh_dim_names", None)
    mesh_dim_names = () if names is None else tuple(str(name) for name in names)
    coordinate_fn = getattr(mesh, "get_coordinate", None)
    coordinate = coordinate_fn() if callable(coordinate_fn) else None
    if coordinate is None:
        raise RuntimeError("FSDP2 DTensor gradient mesh omitted the local coordinate")
    return (
        device_type,
        tuple(int(size) for size in mesh_tensor.shape),
        mesh_dim_names,
        tuple(int(rank) for rank in mesh_tensor.reshape(-1).tolist()),
        tuple(int(index) for index in coordinate),
        tuple(_placement_signature(placement) for placement in placements),
    )


def _gradient_layout(
    *,
    parameter: nn.Parameter,
    gradient: object,
    local: torch.Tensor,
    distributed: bool,
) -> FSDP2GradientLayout:
    parameter_local, parameter_distributed = _local_gradient(parameter)
    if parameter_distributed != distributed:
        raise RuntimeError("FSDP2 parameter and gradient distribution metadata differ")
    if not local.is_contiguous():
        raise RuntimeError("FSDP2 factual gradient spill requires contiguous local shards")
    if (
        tuple(gradient.shape) != tuple(parameter.shape)
        or tuple(gradient.stride()) != tuple(parameter.stride())
        or local.shape != parameter_local.shape
        or local.stride() != parameter_local.stride()
        or local.dtype != parameter_local.dtype
        or local.device != parameter_local.device
    ):
        raise RuntimeError("FSDP2 parameter and gradient tensor layouts differ")
    if distributed:
        parameter_spec = _dtensor_spec(parameter)
        gradient_spec = _dtensor_spec(gradient)
        if parameter_spec != gradient_spec:
            raise RuntimeError("FSDP2 parameter and gradient DTensor layouts differ")
        (
            mesh_device_type,
            mesh_shape,
            mesh_dim_names,
            mesh_ranks,
            mesh_coordinate,
            placements,
        ) = parameter_spec
    else:
        mesh_device_type = None
        mesh_shape = ()
        mesh_dim_names = ()
        mesh_ranks = ()
        mesh_coordinate = ()
        placements = ()
    return FSDP2GradientLayout(
        parameter_id=id(parameter),
        distributed=distributed,
        global_shape=tuple(int(size) for size in parameter.shape),
        global_stride=tuple(int(stride) for stride in parameter.stride()),
        local_shape=tuple(int(size) for size in local.shape),
        local_stride=tuple(int(stride) for stride in local.stride()),
        dtype=local.dtype,
        source_device=str(local.device),
        source_device_type=local.device.type,
        mesh_device_type=mesh_device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        mesh_ranks=mesh_ranks,
        mesh_coordinate=mesh_coordinate,
        placements=placements,
    )


def _parameter_layout(parameter: nn.Parameter) -> FSDP2GradientLayout:
    local, distributed = _local_gradient(parameter)
    if not local.is_contiguous():
        raise RuntimeError("FSDP2 parameter manifest requires contiguous local shards")
    if distributed:
        (
            mesh_device_type,
            mesh_shape,
            mesh_dim_names,
            mesh_ranks,
            mesh_coordinate,
            placements,
        ) = _dtensor_spec(parameter)
    else:
        mesh_device_type = None
        mesh_shape = ()
        mesh_dim_names = ()
        mesh_ranks = ()
        mesh_coordinate = ()
        placements = ()
    return FSDP2GradientLayout(
        parameter_id=id(parameter),
        distributed=distributed,
        global_shape=tuple(int(size) for size in parameter.shape),
        global_stride=tuple(int(stride) for stride in parameter.stride()),
        local_shape=tuple(int(size) for size in local.shape),
        local_stride=tuple(int(stride) for stride in local.stride()),
        dtype=local.dtype,
        source_device=str(local.device),
        source_device_type=local.device.type,
        mesh_device_type=mesh_device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        mesh_ranks=mesh_ranks,
        mesh_coordinate=mesh_coordinate,
        placements=placements,
    )


def _rank_invariant_manifest_entry(
    parameter_name: str,
    layout: FSDP2GradientLayout,
) -> dict[str, object]:
    return {
        "parameter_name": parameter_name,
        "distributed": layout.distributed,
        "global_shape": layout.global_shape,
        "global_stride": layout.global_stride,
        "dtype": str(layout.dtype),
        "source_device_type": layout.source_device_type,
        "mesh_device_type": layout.mesh_device_type,
        "mesh_shape": layout.mesh_shape,
        "mesh_dim_names": layout.mesh_dim_names,
        "mesh_ranks": layout.mesh_ranks,
        "placements": layout.placements,
    }


def _gradient_manifest_sha256(entries: list[dict[str, object]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def fsdp2_present_gradient_manifest(module: nn.Module) -> dict[str, int | str]:
    """Return local evidence for the rank's currently present gradients."""

    entries: list[dict[str, object]] = []
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        local, distributed = _local_gradient(gradient)
        layout = _gradient_layout(
            parameter=parameter,
            gradient=gradient,
            local=local,
            distributed=distributed,
        )
        entries.append(_rank_invariant_manifest_entry(name, layout))
    return {
        "manifest_sha256": _gradient_manifest_sha256(entries),
        "shard_count": len(entries),
    }


def fsdp2_parameter_layout_manifest(module: nn.Module) -> dict[str, int | str]:
    """Return the rank-invariant global layout contract for all model parameters."""

    if not isinstance(module, nn.Module):
        raise TypeError("FSDP2 parameter manifest requires an nn.Module")
    entries: list[dict[str, object]] = []
    trainable_count = 0
    for name, parameter in module.named_parameters():
        layout = _parameter_layout(parameter)
        entry = _rank_invariant_manifest_entry(name, layout)
        entry["requires_grad"] = parameter.requires_grad
        entries.append(entry)
        trainable_count += int(parameter.requires_grad)
    if not entries:
        raise RuntimeError("FSDP2 parameter manifest found no parameters")
    return {
        "manifest_sha256": _gradient_manifest_sha256(entries),
        "parameter_count": len(entries),
        "trainable_parameter_count": trainable_count,
    }


def spill_fsdp2_factual_gradients_to_cpu(module: nn.Module) -> FSDP2CPUGradientSpill:
    """Copy reduced factual shards to CPU, then release their GPU storage."""

    if not isinstance(module, nn.Module):
        raise TypeError("FSDP2 factual gradient spill requires an nn.Module")
    copied: list[tuple[nn.Parameter, FSDP2CPUGradientShard, int]] = []
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            local, distributed = _local_gradient(gradient)
            layout = _gradient_layout(
                parameter=parameter,
                gradient=gradient,
                local=local,
                distributed=distributed,
            )
            cpu = local.detach().to(device="cpu", copy=True).contiguous()
            byte_count = cpu.numel() * cpu.element_size()
            copied.append(
                (
                    parameter,
                    FSDP2CPUGradientShard(
                        parameter_name=name,
                        local_gradient=cpu,
                        layout=layout,
                    ),
                    byte_count if local.device.type == "cuda" else 0,
                )
            )
        if not copied:
            raise RuntimeError("FSDP2 factual gradient spill found no local gradients")
        for parameter, _shard, _cuda_bytes in copied:
            parameter.grad = None
    shards = tuple(shard for _parameter, shard, _cuda_bytes in copied)
    manifest_entries = [
        _rank_invariant_manifest_entry(shard.parameter_name, shard.layout)
        for shard in shards
    ]
    return FSDP2CPUGradientSpill(
        shards=shards,
        total_bytes=sum(
            shard.local_gradient.numel() * shard.local_gradient.element_size()
            for shard in shards
        ),
        cuda_source_bytes=sum(cuda_bytes for _parameter, _shard, cuda_bytes in copied),
        distributed_shard_count=sum(int(shard.layout.distributed) for shard in shards),
        manifest_sha256=_gradient_manifest_sha256(manifest_entries),
    )


def _validate_gradient_pair(
    *,
    parameter: nn.Parameter,
    parameter_name: str,
    gradient: object,
    local: torch.Tensor,
    shard: FSDP2CPUGradientShard,
    distributed: bool,
) -> None:
    layout = _gradient_layout(
        parameter=parameter,
        gradient=gradient,
        local=local,
        distributed=distributed,
    )
    if layout != shard.layout:
        raise RuntimeError(f"FSDP2 gradient layout changed while restoring {parameter_name}")
    if local.shape != shard.local_gradient.shape or local.dtype != shard.local_gradient.dtype:
        raise RuntimeError(f"FSDP2 CPU gradient changed while restoring {parameter_name}")


def merge_fsdp2_factual_gradients_from_cpu(
    module: nn.Module,
    spill: FSDP2CPUGradientSpill,
    *,
    chunk_bytes: int = 1024 * 1024,
) -> dict[str, int]:
    """Add exact CPU factual shards to omitted gradients before one optimizer step."""

    if not isinstance(module, nn.Module):
        raise TypeError("FSDP2 factual gradient merge requires an nn.Module")
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("FSDP2 factual gradient merge chunk size must be positive")
    parameters = dict(module.named_parameters())
    scratch: dict[tuple[str, torch.dtype], torch.Tensor] = {}
    plans: list[
        tuple[
            nn.Parameter,
            object | None,
            torch.Tensor | None,
            FSDP2CPUGradientShard,
        ]
    ] = []
    restored = 0
    accumulated = 0
    with torch.no_grad():
        for shard in spill.shards:
            parameter = parameters.get(shard.parameter_name)
            if parameter is None:
                raise RuntimeError(
                    f"FSDP2 parameter disappeared while restoring {shard.parameter_name}"
                )
            if id(parameter) != shard.layout.parameter_id:
                raise RuntimeError(
                    f"FSDP2 parameter identity changed while restoring {shard.parameter_name}"
                )
            gradient = parameter.grad
            if gradient is None:
                if _parameter_layout(parameter) != shard.layout:
                    raise RuntimeError(
                        f"FSDP2 parameter layout changed while restoring {shard.parameter_name}"
                    )
                plans.append((parameter, None, None, shard))
                continue
            local, distributed = _local_gradient(gradient)
            _validate_gradient_pair(
                parameter=parameter,
                parameter_name=shard.parameter_name,
                gradient=gradient,
                local=local,
                shard=shard,
                distributed=distributed,
            )
            plans.append((parameter, gradient, local, shard))

        for parameter, gradient, local, shard in plans:
            if gradient is None:
                gradient = torch.empty_like(parameter)
                local, distributed = _local_gradient(gradient)
                _validate_gradient_pair(
                    parameter=parameter,
                    parameter_name=shard.parameter_name,
                    gradient=gradient,
                    local=local,
                    shard=shard,
                    distributed=distributed,
                )
                local.copy_(shard.local_gradient, non_blocking=False)
                parameter.grad = gradient
                restored += 1
                continue
            if local is None:
                raise RuntimeError("FSDP2 merge plan omitted a local gradient")
            if local.device.type == "cpu":
                local.add_(shard.local_gradient)
                accumulated += 1
                continue
            key = (str(local.device), local.dtype)
            capacity = max(1, chunk_bytes // local.element_size())
            buffer = scratch.get(key)
            if buffer is None or buffer.numel() < capacity:
                buffer = torch.empty(capacity, device=local.device, dtype=local.dtype)
                scratch[key] = buffer
            local_flat = local.view(-1)
            cpu_flat = shard.local_gradient.view(-1)
            for offset in range(0, local_flat.numel(), capacity):
                count = min(capacity, local_flat.numel() - offset)
                staging = buffer[:count]
                staging.copy_(cpu_flat[offset : offset + count], non_blocking=False)
                local_flat[offset : offset + count].add_(staging)
            accumulated += 1
    return {
        "shard_count": len(spill.shards),
        "restored_gradient_count": restored,
        "accumulated_gradient_count": accumulated,
        "total_bytes": spill.total_bytes,
        "chunk_bytes": chunk_bytes,
    }


def validate_fsdp2_placement(value: object) -> str:
    """Return one supported placement or reject an ambiguous execution contract."""

    if not isinstance(value, str) or value not in FSDP2_PLACEMENTS:
        raise ValueError("LingBot-native FSDP2 placement is unsupported")
    return value


def configure_fsdp2_backward_prefetch(module: object, *, mode: str) -> dict[str, object]:
    """Configure FSDP2 backward all-gather overlap without changing model math."""

    if mode not in FSDP2_BACKWARD_PREFETCH_MODES:
        raise ValueError("LingBot-native FSDP2 backward prefetch mode is unsupported")
    if mode == FSDP2_BACKWARD_PREFETCH_DEFAULT:
        return {"mode": mode, "configured_module_count": 0}

    modules = getattr(module, "modules", None)
    if not callable(modules):
        raise TypeError("FSDP2 backward prefetch configuration requires an nn.Module")
    configured = 0
    for child in modules():
        setter = getattr(child, "set_modules_to_backward_prefetch", None)
        if callable(setter):
            # PyTorch 2.8 treats an empty explicit list as "use the default
            # reverse post-forward schedule". Pointing each sharded module at
            # itself disables cross-module prefetch: its own group is already
            # unsharded by pre_backward, so the explicit request is a no-op.
            setter([child])
            configured += 1
    if configured <= 0:
        raise RuntimeError("FSDP2 backward prefetch configuration found no sharded modules")
    return {"mode": mode, "configured_module_count": configured}


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_fsdp2_storage_report(
    value: object,
    *,
    expected_placement: str,
) -> dict[str, Any]:
    """Validate exact rank-local parameter accounting for one placement."""

    placement = validate_fsdp2_placement(expected_placement)
    if not isinstance(value, dict) or set(value) != FSDP2_STORAGE_FIELDS:
        raise ValueError("FSDP2 parameter storage fields differ from the frozen schema")
    report = cast(dict[str, Any], value)
    parameter_tensors = _nonnegative_integer(
        report["parameter_tensors"],
        name="FSDP2 parameter tensors",
    )
    local_elements = _nonnegative_integer(
        report["local_elements"],
        name="FSDP2 local parameter elements",
    )
    cpu_tensors = _nonnegative_integer(
        report["cpu_parameter_tensors"],
        name="FSDP2 CPU parameter tensors",
    )
    cpu_elements = _nonnegative_integer(
        report["cpu_local_elements"],
        name="FSDP2 CPU local parameter elements",
    )
    cuda_tensors = _nonnegative_integer(
        report["cuda_parameter_tensors"],
        name="FSDP2 CUDA parameter tensors",
    )
    cuda_elements = _nonnegative_integer(
        report["cuda_local_elements"],
        name="FSDP2 CUDA local parameter elements",
    )
    if (
        parameter_tensors <= 0
        or local_elements <= 0
        or report["master_dtype"] != "float32"
        or report["placement"] != placement
        or cpu_tensors + cuda_tensors != parameter_tensors
        or cpu_elements + cuda_elements != local_elements
    ):
        raise ValueError("FSDP2 parameter placement accounting is inconsistent")
    selective_names = report["selective_cpu_parameter_names"]
    if not isinstance(selective_names, list) or any(
        not isinstance(name, str) or not name for name in selective_names
    ):
        raise ValueError("FSDP2 selective CPU parameter names are malformed")

    if placement == FSDP2_CPU_OFFLOAD:
        valid = (
            cpu_tensors == parameter_tensors
            and cpu_elements == local_elements
            and cuda_tensors == 0
            and cuda_elements == 0
            and selective_names == []
        )
    elif placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD:
        valid = (
            cpu_tensors == 1
            and cpu_elements > 0
            and cuda_tensors > 0
            and cuda_elements > 0
            and selective_names == [SELECTIVE_EMBEDDING_PARAMETER]
        )
    elif placement in {
        FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
        FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    }:
        valid = (
            cpu_tensors == len(selective_names)
            and cpu_tensors > 1
            and cpu_elements > 0
            and cuda_tensors > 0
            and cuda_elements > 0
            and selective_names == sorted(set(selective_names))
            and SELECTIVE_EMBEDDING_PARAMETER in selective_names
            and all(
                name == SELECTIVE_EMBEDDING_PARAMETER
                or name.startswith(SELECTIVE_FROZEN_VISION_MODULE_PREFIX)
                for name in selective_names
            )
            and any(
                name.startswith(SELECTIVE_FROZEN_VISION_MODULE_PREFIX)
                for name in selective_names
            )
        )
    else:
        valid = (
            cpu_tensors == 0
            and cpu_elements == 0
            and cuda_tensors == parameter_tensors
            and cuda_elements == local_elements
            and selective_names == []
        )
    if not valid:
        raise ValueError("FSDP2 parameter shards violate the declared placement")
    return report
