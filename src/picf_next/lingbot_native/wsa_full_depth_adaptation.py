from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

WSA_REPOSITORY = "https://github.com/zaleni/WSA"
WSA_COMMIT = "bfee742c585d5ee85722e658978111934c926ca3"
WSA_PREPROCESS_SHA256 = "8cec126303435e4049d31ee04291517a4d23b518de70f96d50678fd7afda29a8"
WSA_SOURCE_ARCHIVE_SHA256 = "8d9cadb6f6c1abff8c8fd8354226c076aa0d33d5410f984bdfb03069e0520221"
_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.(.+)$")


def official_interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """WSA `tools/preprocess_expert_backbones.py::_interpolate_last_dim`."""
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def official_resize_tensor_to_shape(
    src: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    """WSA's released sequential one-dimensional tensor resize, unchanged."""
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, "
                f"target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = official_interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            "Resize produced wrong shape for tensor. "
            f"src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def official_resize_with_alpha(
    src: torch.Tensor,
    target_shape: tuple[int, ...],
    *,
    apply_alpha_scaling: bool,
) -> torch.Tensor:
    """Apply the released WSA resize and its input-width alpha correction."""
    value = official_resize_tensor_to_shape(src, target_shape)
    if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target_shape[-1]:
        alpha = (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
        value = value.to(torch.float32) * alpha
    return value


def repeat_lingbot_kv_heads(
    tensor: torch.Tensor,
    *,
    target_heads: int,
) -> torch.Tensor:
    """Materialize LingBot GQA K/V groups without adding learned parameters."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B, S, H, D], got shape {tuple(tensor.shape)}")
    source_heads = tensor.shape[2]
    if target_heads % source_heads != 0:
        raise ValueError(
            f"target_heads ({target_heads}) must be divisible by source heads ({source_heads})"
        )
    repeats = target_heads // source_heads
    if repeats == 1:
        return tensor
    batch, sequence, _, head_dim = tensor.shape
    expanded = tensor[:, :, :, None, :].expand(
        batch,
        sequence,
        source_heads,
        repeats,
        head_dim,
    )
    return expanded.reshape(batch, sequence, target_heads, head_dim)


@dataclass(frozen=True)
class DepthLayerAssignment:
    target_layer: int
    continuous_source_position: float
    nearest_source_layer: int


def build_nearest_depth_assignments(
    *,
    source_depth: int,
    target_depth: int,
) -> tuple[DepthLayerAssignment, ...]:
    """Register the explicit 30-to-36 nearest-depth compatibility hypothesis."""
    if source_depth <= 0 or target_depth <= 0:
        raise ValueError("source_depth and target_depth must be positive")
    if source_depth == 1:
        return tuple(
            DepthLayerAssignment(
                target_layer=target_layer,
                continuous_source_position=0.0,
                nearest_source_layer=0,
            )
            for target_layer in range(target_depth)
        )
    if target_depth == 1:
        return (
            DepthLayerAssignment(
                target_layer=0,
                continuous_source_position=0.0,
                nearest_source_layer=0,
            ),
        )

    assignments = []
    for target_layer in range(target_depth):
        position = target_layer * (source_depth - 1) / (target_depth - 1)
        assignments.append(
            DepthLayerAssignment(
                target_layer=target_layer,
                continuous_source_position=position,
                nearest_source_layer=round(position),
            )
        )
    return tuple(assignments)


def percent_align_source_layers(
    source_layers: tuple[int, ...],
    *,
    source_depth: int,
    target_depth: int,
) -> tuple[int, ...]:
    """Map WSA intermediate readout depths by normalized network depth."""
    if source_depth <= 1 or target_depth <= 1:
        raise ValueError("source_depth and target_depth must both exceed one")
    if any(layer < 0 or layer >= source_depth for layer in source_layers):
        raise ValueError("source layer lies outside source depth")
    return tuple(round(layer * (target_depth - 1) / (source_depth - 1)) for layer in source_layers)


def source_key_for_target_key(
    target_key: str,
    *,
    depth_assignments: tuple[DepthLayerAssignment, ...],
) -> str:
    """Resolve one target state key without silently dropping a WSA parameter."""
    match = _BLOCK_KEY.fullmatch(target_key)
    if match is None:
        return target_key
    target_layer = int(match.group(1))
    if target_layer >= len(depth_assignments):
        raise ValueError(
            f"Target block {target_layer} lies outside {len(depth_assignments)} assignments"
        )
    assignment = depth_assignments[target_layer]
    if assignment.target_layer != target_layer:
        raise ValueError("Depth assignments are not indexed by target layer")
    return f"blocks.{assignment.nearest_source_layer}.{match.group(2)}"


@dataclass(frozen=True)
class StateAdaptationReceipt:
    source_tensor_count: int
    target_tensor_count: int
    copied_tensor_count: int
    resized_tensor_count: int
    duplicated_source_tensor_count: int
    unused_source_keys: tuple[str, ...]


def adapt_wsa_future_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_shapes: Mapping[str, tuple[int, ...]],
    *,
    source_depth: int = 30,
    target_depth: int = 36,
) -> tuple[dict[str, torch.Tensor], StateAdaptationReceipt]:
    """Build a strict full-depth state using only registered WSA transforms."""
    assignments = build_nearest_depth_assignments(
        source_depth=source_depth,
        target_depth=target_depth,
    )
    adapted: dict[str, torch.Tensor] = {}
    source_usage = {key: 0 for key in source_state}
    copied = 0
    resized = 0

    for target_key, target_shape in target_shapes.items():
        source_key = source_key_for_target_key(
            target_key,
            depth_assignments=assignments,
        )
        if source_key not in source_state:
            raise KeyError(f"Missing registered WSA source tensor for {target_key}: {source_key}")
        source = source_state[source_key]
        source_usage[source_key] += 1
        if tuple(source.shape) == tuple(target_shape):
            value = source
            copied += 1
        else:
            value = official_resize_with_alpha(
                source,
                target_shape,
                apply_alpha_scaling=True,
            )
            resized += 1
        if tuple(value.shape) != tuple(target_shape):
            raise RuntimeError(
                f"Adapted tensor {target_key} has {tuple(value.shape)}, expected {target_shape}"
            )
        # Repeated depth assignments must start numerically equal but remain
        # independent trainable layers after serialization and loading.
        adapted[target_key] = value.detach().clone().contiguous()

    unused = tuple(sorted(key for key, count in source_usage.items() if count == 0))
    if unused:
        raise ValueError(f"Unconsumed WSA tensors are forbidden: {unused[:10]}")
    return adapted, StateAdaptationReceipt(
        source_tensor_count=len(source_state),
        target_tensor_count=len(target_shapes),
        copied_tensor_count=copied,
        resized_tensor_count=resized,
        duplicated_source_tensor_count=sum(count > 1 for count in source_usage.values()),
        unused_source_keys=unused,
    )
