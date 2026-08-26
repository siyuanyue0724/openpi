"""Source-faithful native visual object memory for LingBot Qwen3-VL."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.lingbot_native.modalities import NativeObjectQuerySpatialRelation

UNIPIXEL_QWEN3_POSTERIOR_MASK_MEMORY = "unipixel_qwen3_posterior_mask_memory_v1"


def _canonical_merger_digest(linear_fc1: nn.Linear, linear_fc2: nn.Linear) -> str:
    digest = hashlib.sha256()
    for layer_name, layer in (("linear_fc1", linear_fc1), ("linear_fc2", linear_fc2)):
        for parameter_name, value in sorted(layer.state_dict().items()):
            tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            digest.update(f"{layer_name}.{parameter_name}".encode())
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeObjectMemoryOutput:
    """One same-index memory token and its mask support for every source query."""

    tokens: torch.Tensor
    support_mass: torch.Tensor
    query_valid: torch.Tensor
    capture_generation: int

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or not self.tokens.is_floating_point():
            raise ValueError("native object memory tokens must be floating [batch,query,width]")
        if self.support_mass.shape != self.tokens.shape[:2]:
            raise ValueError("native object support mass must match the token batch and query axes")
        if self.query_valid.shape != self.tokens.shape[:2] or self.query_valid.dtype != torch.bool:
            raise ValueError("native object memory validity must be boolean [batch,query]")
        if any(
            value.device != self.tokens.device for value in (self.support_mass, self.query_valid)
        ):
            raise ValueError("native object memory tensors must share one device")
        if not self.support_mass.is_floating_point():
            raise TypeError("native object support mass must be floating")
        if not torch.isfinite(self.tokens).all() or not torch.isfinite(self.support_mass).all():
            raise ValueError("native object memory contains NaN or infinity")
        if isinstance(self.capture_generation, bool) or self.capture_generation < 0:
            raise ValueError("native object memory capture generation must be non-negative")


class Qwen3NativeMergerProjection(nn.Module):
    """Exact copied Qwen3 merger MLP used by the UniPixel memory primitive."""

    def __init__(
        self,
        *,
        linear_fc1: nn.Linear,
        act_fn: nn.GELU,
        linear_fc2: nn.Linear,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.linear_fc1 = copy.deepcopy(linear_fc1).to(device=device, dtype=dtype)
        self.act_fn = copy.deepcopy(act_fn)
        self.linear_fc2 = copy.deepcopy(linear_fc2).to(device=device, dtype=dtype)

        for source, copied in (
            (linear_fc1, self.linear_fc1),
            (linear_fc2, self.linear_fc2),
        ):
            source_state = source.state_dict()
            copied_state = copied.state_dict()
            if source_state.keys() != copied_state.keys():
                raise RuntimeError("copied Qwen3 merger projection changed its parameter names")
            for name in source_state:
                expected = source_state[name].detach().to(
                    device=copied_state[name].device,
                    dtype=copied_state[name].dtype,
                )
                if not torch.equal(copied_state[name], expected):
                    raise RuntimeError("copied Qwen3 merger projection changed source values")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(value)))


class PretrainedQwen3ObjectMemory(nn.Module):
    """Capture native Qwen3 cells and encode VidEoMT mask posteriors as memory.

    The projection topology and initialization copy UniPixel's mature primitive.
    Resizing predicted mask logits and using their posterior probabilities is the
    explicit ADR-225 adaptation described in its design record.
    """

    def __init__(
        self,
        *,
        capacity: int,
        camera_count: int = 3,
        camera_slot: int = 0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("object memory capacity must be a positive integer")
        if (
            isinstance(camera_count, bool)
            or not isinstance(camera_count, int)
            or camera_count <= 0
        ):
            raise ValueError("object memory camera count must be a positive integer")
        if (
            isinstance(camera_slot, bool)
            or not isinstance(camera_slot, int)
            or not 0 <= camera_slot < camera_count
        ):
            raise ValueError("object memory camera slot must lie inside the camera ABI")
        if not isinstance(epsilon, float) or not 0.0 < epsilon < 1.0:
            raise ValueError("object memory epsilon must be a float in (0,1)")
        self.capacity = capacity
        self.camera_count = camera_count
        self.camera_slot = camera_slot
        self.epsilon = epsilon
        self.projection: Qwen3NativeMergerProjection | None = None
        self.spatial_merge_size: int | None = None
        self.source_parameter_sha256: str | None = None
        self.copied_parameter_sha256: str | None = None
        self._visual_hook_handle: Any | None = None
        self._merger_hook_handle: Any | None = None
        self._capture_generation = 0
        self._consumed_generation = 0
        self._pending_grid_thw: torch.Tensor | None = None
        self._pending_grouped_features: torch.Tensor | None = None

    @property
    def installed(self) -> bool:
        return self.projection is not None

    @property
    def host_width(self) -> int:
        projection = self.projection
        if projection is None:
            raise RuntimeError("pretrained object memory is not installed")
        return projection.linear_fc2.out_features

    def install_from_qwen3_visual(
        self,
        visual: nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Copy the native merger once and register fail-closed capture hooks."""

        if self.installed or self._visual_hook_handle is not None:
            raise RuntimeError("pretrained object memory may be installed only once")
        merger = getattr(visual, "merger", None)
        linear_fc1 = getattr(merger, "linear_fc1", None)
        act_fn = getattr(merger, "act_fn", None)
        linear_fc2 = getattr(merger, "linear_fc2", None)
        merge_size = getattr(visual, "spatial_merge_size", None)
        if (
            not isinstance(linear_fc1, nn.Linear)
            or type(act_fn) is not nn.GELU
            or not isinstance(linear_fc2, nn.Linear)
            or isinstance(merge_size, bool)
            or not isinstance(merge_size, int)
            or merge_size <= 0
            or linear_fc1.in_features != linear_fc1.out_features
            or linear_fc2.in_features != linear_fc1.out_features
        ):
            raise TypeError("LingBot visual merger is not the audited Qwen3 topology")
        projection = Qwen3NativeMergerProjection(
            linear_fc1=linear_fc1,
            act_fn=act_fn,
            linear_fc2=linear_fc2,
            device=device,
            dtype=dtype,
        )
        self.source_parameter_sha256 = _canonical_merger_digest(
            linear_fc1,
            linear_fc2,
        )
        self.copied_parameter_sha256 = _canonical_merger_digest(
            projection.linear_fc1,
            projection.linear_fc2,
        )
        if self.source_parameter_sha256 != self.copied_parameter_sha256:
            raise RuntimeError("copied native merger projection changed its source digest")
        self.projection = projection
        self.spatial_merge_size = merge_size
        self._visual_hook_handle = visual.register_forward_pre_hook(
            self._capture_visual_call,
            with_kwargs=True,
        )
        self._merger_hook_handle = linear_fc1.register_forward_pre_hook(
            self._capture_merger_input
        )

    def installation_receipt(self) -> dict[str, object]:
        projection = self.projection
        if projection is None or self.spatial_merge_size is None:
            raise RuntimeError("pretrained object memory is not installed")
        return {
            "schema": "picf-next.pretrained-qwen3-object-memory-installation.v1",
            "identity": UNIPIXEL_QWEN3_POSTERIOR_MASK_MEMORY,
            "capacity": self.capacity,
            "camera_count": self.camera_count,
            "camera_slot": self.camera_slot,
            "spatial_merge_size": self.spatial_merge_size,
            "grouped_width": projection.linear_fc1.in_features,
            "host_width": projection.linear_fc2.out_features,
            "source_parameter_sha256": self.source_parameter_sha256,
            "copied_parameter_sha256": self.copied_parameter_sha256,
            "copied_projection_trainable": all(
                parameter.requires_grad for parameter in projection.parameters()
            ),
            "pooling": "bilinear-logit-resize+sigmoid+normalized-posterior-mean",
        }

    def _capture_visual_call(
        self,
        _module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        grid_thw = kwargs.get("grid_thw")
        if grid_thw is None and len(args) >= 2:
            grid_thw = args[1]
        if not isinstance(grid_thw, torch.Tensor):
            raise RuntimeError("Qwen3 visual capture omitted grid_thw")
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3 or grid_thw.dtype != torch.long:
            raise ValueError("Qwen3 visual grid_thw must be long [images,3]")
        self._capture_generation += 1
        self._pending_grid_thw = grid_thw.detach().clone()
        self._pending_grouped_features = None

    def _capture_merger_input(
        self,
        _module: nn.Module,
        args: tuple[object, ...],
    ) -> None:
        if self._pending_grid_thw is None or self._capture_generation <= 0:
            raise RuntimeError("Qwen3 merger input arrived without a visual capture generation")
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Qwen3 merger linear_fc1 input contract changed")
        value = args[0]
        projection = self.projection
        if (
            projection is None
            or value.ndim != 2
            or value.shape[-1] != projection.linear_fc1.in_features
            or not value.is_floating_point()
        ):
            raise ValueError("Qwen3 grouped merger input differs from the copied projection")
        if self._pending_grouped_features is not None:
            raise RuntimeError("Qwen3 main merger executed twice in one capture generation")
        self._pending_grouped_features = value

    def _static_camera_features(
        self,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        grid_thw = self._pending_grid_thw
        grouped = self._pending_grouped_features
        merge_size = self.spatial_merge_size
        projection = self.projection
        if grid_thw is None or grouped is None or merge_size is None or projection is None:
            raise RuntimeError("pretrained object memory capture is incomplete")
        if grid_thw.shape[0] != batch_size * self.camera_count:
            raise ValueError("Qwen3 visual image count differs from the fixed CALVIN camera ABI")
        numerators = grid_thw.prod(dim=-1)
        denominator = merge_size**2
        if (numerators.remainder(denominator) != 0).any():
            raise ValueError("Qwen3 visual grids are not divisible by spatial merge area")
        split_sizes = torch.div(numerators, denominator, rounding_mode="floor").tolist()
        if sum(split_sizes) != grouped.shape[0]:
            raise ValueError("captured Qwen3 grouped cells do not match grid_thw")
        offsets = [0]
        for size in split_sizes:
            offsets.append(offsets[-1] + int(size))
        selected: list[torch.Tensor] = []
        selected_shape: tuple[int, int] | None = None
        for batch_index in range(batch_size):
            image_index = batch_index * self.camera_count + self.camera_slot
            temporal, height, width = (
                int(value) for value in grid_thw[image_index].tolist()
            )
            if temporal != 1 or height % merge_size or width % merge_size:
                raise ValueError("ADR-225 requires one static frame on an exact merged grid")
            shape = (height // merge_size, width // merge_size)
            if selected_shape is None:
                selected_shape = shape
            elif shape != selected_shape:
                raise ValueError("ADR-225 requires one common static-camera grid per batch")
            start, stop = offsets[image_index], offsets[image_index + 1]
            value = grouped[start:stop]
            if value.shape != (shape[0] * shape[1], projection.linear_fc1.in_features):
                raise RuntimeError("selected static-camera cells changed native raster layout")
            selected.append(value)
        if selected_shape is None:
            raise RuntimeError("static-camera capture contains no samples")
        return torch.stack(selected, dim=0), selected_shape

    def encode_mask_weights(
        self,
        *,
        grouped_features: torch.Tensor,
        mask_weights: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> NativeObjectMemoryOutput:
        """Apply the exact copied projection to normalized object-mask means."""

        projection = self.projection
        if projection is None:
            raise RuntimeError("pretrained object memory is not installed")
        if (
            grouped_features.ndim != 3
            or grouped_features.shape[-1] != projection.linear_fc1.in_features
            or not grouped_features.is_floating_point()
        ):
            raise ValueError("grouped native visual features have an invalid shape")
        batch, pixels, _width = grouped_features.shape
        if (
            mask_weights.ndim != 3
            or mask_weights.shape != (batch, self.capacity, pixels)
            or not mask_weights.is_floating_point()
            or mask_weights.device != grouped_features.device
            or not torch.isfinite(mask_weights).all()
            or (mask_weights < 0).any()
            or (mask_weights > 1).any()
        ):
            raise ValueError("object mask weights must be finite probabilities [batch,query,pixel]")
        if (
            query_valid.shape != (batch, self.capacity)
            or query_valid.dtype != torch.bool
            or query_valid.device != grouped_features.device
        ):
            raise ValueError("object memory query validity has an invalid shape or device")
        weights = mask_weights.to(dtype=torch.float32)
        features = grouped_features.to(dtype=torch.float32)
        mass = weights.sum(dim=-1)
        pooled = torch.einsum("bqp,bpd->bqd", weights, features)
        pooled = pooled / mass.clamp_min(self.epsilon).unsqueeze(-1)
        pooled = pooled * query_valid.unsqueeze(-1).to(dtype=pooled.dtype)
        parameter = projection.linear_fc1.weight
        pooled = pooled.to(device=parameter.device, dtype=parameter.dtype)
        tokens = projection(pooled)
        tokens = tokens * query_valid.unsqueeze(-1).to(dtype=tokens.dtype)
        return NativeObjectMemoryOutput(
            tokens=tokens,
            support_mass=(mass / max(pixels, 1)).to(device=tokens.device, dtype=tokens.dtype),
            query_valid=query_valid.to(device=tokens.device),
            capture_generation=self._capture_generation,
        )

    def consume(
        self,
        relation: NativeObjectQuerySpatialRelation,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> NativeObjectMemoryOutput:
        """Consume exactly one native visual capture for one spatial relation."""

        if not self.installed:
            raise RuntimeError("pretrained object memory is not installed")
        if self._capture_generation <= self._consumed_generation:
            raise RuntimeError("pretrained object memory capture is missing, stale or already used")
        if relation.batch_size != batch_size or relation.query_count != self.capacity:
            raise ValueError("object relation differs from the installed memory contract")
        if relation.geometry_kind != "image_grid" or relation.grid_shape is None:
            raise ValueError("pretrained object memory requires one image-grid relation")
        grouped, target_shape = self._static_camera_features(batch_size=batch_size)
        source_height, source_width = relation.grid_shape
        logits = relation.mask_logits.reshape(
            batch_size,
            self.capacity,
            source_height,
            source_width,
        ).to(dtype=torch.float32)
        resized_logits = F.interpolate(
            logits,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )
        source_valid = relation.pixel_valid.reshape(
            batch_size,
            1,
            source_height,
            source_width,
        ).to(dtype=torch.float32)
        resized_valid = F.interpolate(source_valid, size=target_shape, mode="nearest") > 0.5
        weights = torch.sigmoid(resized_logits) * resized_valid.to(dtype=torch.float32)
        output = self.encode_mask_weights(
            grouped_features=grouped,
            mask_weights=weights.flatten(2),
            query_valid=relation.query_valid,
        )
        if output.tokens.device != device or output.tokens.dtype != dtype:
            raise ValueError("native object memory and LingBot prefix differ in device or dtype")
        self._consumed_generation = self._capture_generation
        return output
