"""Qwen3-VL pre-context masking at the released packed-patch boundary.

LingBot's pinned Qwen3-VL processor emits block-major raw patch vectors.  The
official vision merger consumes each consecutive ``spatial_merge_size**2``
group as one visual token.  This module records and applies that exact
dependency without reading task, object, track, future, or loss metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from picf_next.lingbot_native.prediction import TokenizerDependencyMap


def qwen_patch_merger_dependency_map(
    *,
    raw_patch_count: int,
    spatial_merge_size: int,
    device: torch.device | str = "cpu",
) -> TokenizerDependencyMap:
    """Return the exact block-major dependency used by Qwen3-VL's merger."""

    integer_values = (raw_patch_count, spatial_merge_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        raise TypeError("Qwen patch counts and merge size must be integers")
    if min(integer_values) <= 0:
        raise ValueError("Qwen patch counts and merge size must be positive")
    merge_unit = spatial_merge_size**2
    if raw_patch_count % merge_unit:
        raise ValueError("raw Qwen patch count must be divisible by the spatial merge unit")
    output_count = raw_patch_count // merge_unit
    dependency = torch.zeros(
        output_count,
        raw_patch_count,
        dtype=torch.bool,
        device=device,
    )
    rows = torch.arange(output_count, device=dependency.device).repeat_interleave(merge_unit)
    columns = torch.arange(raw_patch_count, device=dependency.device)
    dependency[rows, columns] = True
    return TokenizerDependencyMap(dependency)


@dataclass(frozen=True, slots=True)
class QwenPackedPatchMask:
    """Replayable mask sampled over released Qwen visual output addresses."""

    merged_target_mask: torch.Tensor
    raw_target_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    image_valid: torch.Tensor
    query_view_indices: torch.Tensor
    query_token_indices: torch.Tensor
    query_valid: torch.Tensor
    eligible_view_indices: tuple[int, ...]
    spatial_merge_size: int
    seed: int
    probability: float

    def __post_init__(self) -> None:
        if self.merged_target_mask.ndim != 3 or self.merged_target_mask.dtype != torch.bool:
            raise ValueError("merged Qwen target mask must be boolean [batch,views,tokens]")
        if self.raw_target_mask.ndim != 3 or self.raw_target_mask.dtype != torch.bool:
            raise ValueError("raw Qwen target mask must be boolean [batch,views,patches]")
        if self.image_grid_thw.shape != (*self.image_valid.shape, 3):
            raise ValueError("Qwen image grid must have shape [batch,views,3]")
        if self.image_grid_thw.dtype != torch.long:
            raise TypeError("Qwen image grid must use torch.long")
        if self.image_valid.shape != self.merged_target_mask.shape[:2]:
            raise ValueError("Qwen image validity must match mask batch and view axes")
        if self.image_valid.dtype != torch.bool:
            raise TypeError("Qwen image validity must be boolean")
        batch, views = self.image_valid.shape
        if self.query_view_indices.ndim != 1 or self.query_view_indices.dtype != torch.long:
            raise ValueError("Qwen query view indices must be long [queries]")
        if (
            self.query_token_indices.shape != (batch, self.query_view_indices.shape[0])
            or self.query_token_indices.dtype != torch.long
        ):
            raise ValueError("Qwen query token indices must be long [batch,queries]")
        if (
            self.query_valid.shape != self.query_token_indices.shape
            or self.query_valid.dtype != torch.bool
        ):
            raise ValueError("Qwen query validity must be boolean [batch,queries]")
        if (
            not isinstance(self.eligible_view_indices, tuple)
            or tuple(sorted(set(self.eligible_view_indices))) != self.eligible_view_indices
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < views
                for value in self.eligible_view_indices
            )
        ):
            raise ValueError("eligible Qwen views must be sorted unique in-range integers")
        tensors = (
            self.raw_target_mask,
            self.image_grid_thw,
            self.image_valid,
            self.query_view_indices,
            self.query_token_indices,
            self.query_valid,
        )
        if any(value.device != self.merged_target_mask.device for value in tensors):
            raise ValueError("Qwen mask tensors must share one device")
        if (
            isinstance(self.spatial_merge_size, bool)
            or not isinstance(self.spatial_merge_size, int)
            or self.spatial_merge_size <= 0
        ):
            raise ValueError("Qwen spatial merge size must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Qwen mask seed must be a non-negative integer")
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not math.isfinite(self.probability)
            or not 0 <= self.probability <= 1
        ):
            raise ValueError("Qwen mask probability must be finite in [0,1]")
        if (self.image_grid_thw <= 0).any():
            raise ValueError("Qwen image grid dimensions must be positive")
        merge_unit = self.spatial_merge_size**2
        raw_counts = self.image_grid_thw.prod(dim=-1)
        if (raw_counts % merge_unit).any():
            raise ValueError("Qwen image grids must divide exactly by the merge unit")
        if not torch.equal(
            raw_counts,
            torch.full_like(raw_counts, self.raw_target_mask.shape[-1]),
        ):
            raise ValueError("Qwen image grids differ from the packed raw patch axis")
        if not torch.equal(
            raw_counts // merge_unit,
            torch.full_like(raw_counts, self.merged_target_mask.shape[-1]),
        ):
            raise ValueError("Qwen image grids differ from the merged target axis")
        expected_raw = self.merged_target_mask.repeat_interleave(merge_unit, dim=-1)
        if not torch.equal(self.raw_target_mask, expected_raw):
            raise ValueError("raw Qwen mask does not match the official block-major merger")
        if self.merged_target_mask[~self.image_valid].any():
            raise ValueError("an unavailable image cannot contribute a mask target")
        query_count = self.query_view_indices.shape[0]
        if query_count:
            if not self.eligible_view_indices:
                raise ValueError("Qwen mask queries require at least one eligible view")
            if ((self.query_view_indices < 0) | (self.query_view_indices >= views)).any():
                raise ValueError("Qwen query references an out-of-range view")
            if not torch.isin(
                self.query_view_indices,
                torch.tensor(self.eligible_view_indices, device=self.query_view_indices.device),
            ).all():
                raise ValueError("Qwen query references an ineligible view")
            merged_count = self.merged_target_mask.shape[-1]
            if ((self.query_token_indices < 0) | (self.query_token_indices >= merged_count)).any():
                raise ValueError("Qwen query references an out-of-range merged token")
            expected_valid = self.image_valid[:, self.query_view_indices]
            if not torch.equal(self.query_valid, expected_valid):
                raise ValueError("Qwen query validity differs from source image availability")
            expected_merged = torch.zeros_like(self.merged_target_mask)
            batch_indices = torch.arange(batch, device=self.merged_target_mask.device)[:, None]
            view_indices = self.query_view_indices[None, :].expand(batch, -1)
            expected_merged[batch_indices, view_indices, self.query_token_indices] = (
                self.query_valid
            )
            if not torch.equal(self.merged_target_mask, expected_merged):
                raise ValueError("Qwen target mask differs from its fixed-count query addresses")
        elif self.merged_target_mask.any():
            raise ValueError("Qwen target mask cannot be nonempty without query addresses")

    @property
    def digest(self) -> str:
        payload = {
            "grid": self.image_grid_thw.detach().cpu().tolist(),
            "merge_size": self.spatial_merge_size,
            "probability": float(self.probability).hex(),
            "eligible_views": self.eligible_view_indices,
            "seed": self.seed,
            "shape": list(self.merged_target_mask.shape),
            "version": 2,
        }
        mask = self.merged_target_mask.detach().cpu().contiguous().to(torch.uint8)
        query_tokens = self.query_token_indices.detach().cpu().contiguous()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(
            encoded + mask.numpy().tobytes() + query_tokens.numpy().tobytes()
        ).hexdigest()

    @property
    def query_count(self) -> int:
        return self.query_view_indices.shape[0]


@dataclass(frozen=True, slots=True)
class QwenWholeViewOmission:
    """One source-known image view removed through LingBot's native missing-view path."""

    omitted_view_index: int
    image_grid_thw: torch.Tensor
    image_valid: torch.Tensor
    seed: int

    def __post_init__(self) -> None:
        if self.image_valid.ndim != 2 or self.image_valid.dtype != torch.bool:
            raise ValueError("Qwen view availability must be boolean [batch,views]")
        if self.image_grid_thw.shape != (*self.image_valid.shape, 3):
            raise ValueError("Qwen view omission grid must be [batch,views,3]")
        if self.image_grid_thw.dtype != torch.long:
            raise TypeError("Qwen view omission grid must use torch.long")
        if self.image_grid_thw.device != self.image_valid.device:
            raise ValueError("Qwen view omission tensors must share one device")
        if (
            isinstance(self.omitted_view_index, bool)
            or not isinstance(self.omitted_view_index, int)
            or not 0 <= self.omitted_view_index < self.image_valid.shape[1]
        ):
            raise ValueError("omitted Qwen view index is outside the declared view axis")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Qwen view omission seed must be a non-negative integer")
        source_valid = self.source_valid
        if not source_valid.any():
            raise ValueError("an omitted Qwen view must be available for at least one sample")
        remaining = self.source_image_valid.any(dim=1)
        if not (remaining | ~source_valid).all():
            raise ValueError("whole-view omission cannot remove every available source view")

    @property
    def source_valid(self) -> torch.Tensor:
        return self.image_valid[:, self.omitted_view_index]

    @property
    def source_image_valid(self) -> torch.Tensor:
        value = self.image_valid.clone()
        value[:, self.omitted_view_index] = False
        return value

    @property
    def omitted_name(self) -> str:
        return f"qwen_view_{self.omitted_view_index}"

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "grid": self.image_grid_thw.detach().cpu().tolist(),
                "omitted_view_index": self.omitted_view_index,
                "seed": self.seed,
                "shape": list(self.image_valid.shape),
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        valid = self.image_valid.detach().cpu().contiguous().to(torch.uint8)
        return hashlib.sha256(payload + valid.numpy().tobytes()).hexdigest()


def sample_qwen_whole_view_omission(
    *,
    images: torch.Tensor,
    image_valid: torch.Tensor,
    image_grid_thw: torch.Tensor,
    seed: int,
    eligible_view_indices: tuple[int, ...],
) -> QwenWholeViewOmission:
    """Select one complete source view using only availability, geometry, and CPU RNG."""

    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("packed Qwen images must be floating [batch,views,patches,width]")
    if image_valid.shape != images.shape[:2] or image_valid.dtype != torch.bool:
        raise ValueError("Qwen image validity must be boolean [batch,views]")
    if image_grid_thw.shape != (*images.shape[:2], 3) or image_grid_thw.dtype != torch.long:
        raise ValueError("Qwen image grid must be long [batch,views,3]")
    if image_valid.device != images.device or image_grid_thw.device != images.device:
        raise ValueError("Qwen view omission inputs must share one device")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Qwen view omission seed must be a non-negative integer")
    views = images.shape[1]
    if (
        not isinstance(eligible_view_indices, tuple)
        or not eligible_view_indices
        or tuple(sorted(set(eligible_view_indices))) != eligible_view_indices
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < views
            for value in eligible_view_indices
        )
    ):
        raise ValueError("eligible omitted Qwen views must be sorted unique in-range indices")
    available: list[int] = []
    for view_index in eligible_view_indices:
        selected = image_valid[:, view_index]
        remaining = image_valid.clone()
        remaining[:, view_index] = False
        if selected.any() and (remaining.any(dim=1) | ~selected).all():
            available.append(view_index)
    if not available:
        raise ValueError("no eligible Qwen view can be omitted while retaining source evidence")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected_index = int(torch.randint(len(available), (), generator=generator).item())
    return QwenWholeViewOmission(
        omitted_view_index=available[selected_index],
        image_grid_thw=image_grid_thw.detach().clone(),
        image_valid=image_valid.detach().clone(),
        seed=seed,
    )


def sample_qwen_packed_patch_mask(
    *,
    images: torch.Tensor,
    image_valid: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    probability: float,
    seed: int,
    eligible_view_indices: tuple[int, ...],
) -> QwenPackedPatchMask:
    """Sample fixed-count addresses from frozen geometry and CPU RNG only."""

    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("packed Qwen images must be floating [batch,views,patches,width]")
    if image_valid.shape != images.shape[:2] or image_valid.dtype != torch.bool:
        raise ValueError("Qwen image validity must be boolean [batch,views]")
    if image_grid_thw.shape != (*images.shape[:2], 3) or image_grid_thw.dtype != torch.long:
        raise ValueError("Qwen image grid must be long [batch,views,3]")
    if image_valid.device != images.device or image_grid_thw.device != images.device:
        raise ValueError("Qwen images, validity, and grid must share one device")
    if not torch.isfinite(images).all():
        raise ValueError("packed Qwen images contain NaN or infinity")
    if (
        isinstance(spatial_merge_size, bool)
        or not isinstance(spatial_merge_size, int)
        or spatial_merge_size <= 0
    ):
        raise ValueError("Qwen spatial merge size must be a positive integer")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Qwen mask probability must be finite in [0,1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Qwen mask seed must be a non-negative integer")
    view_count = images.shape[1]
    if (
        not isinstance(eligible_view_indices, tuple)
        or tuple(sorted(set(eligible_view_indices))) != eligible_view_indices
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < view_count
            for value in eligible_view_indices
        )
    ):
        raise ValueError("eligible Qwen views must be sorted unique in-range integers")
    if probability > 0 and not eligible_view_indices:
        raise ValueError("a positive Qwen mask probability requires an eligible view")
    merge_unit = spatial_merge_size**2
    raw_counts = image_grid_thw.prod(dim=-1)
    if (raw_counts != images.shape[2]).any() or (raw_counts % merge_unit).any():
        raise ValueError("packed Qwen patch axis differs from the declared image grid")
    merged_count = images.shape[2] // merge_unit
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected_per_view = (
        min(merged_count, max(1, round(probability * merged_count))) if probability else 0
    )
    query_count = len(eligible_view_indices) * selected_per_view
    query_views = torch.tensor(
        tuple(view for view in eligible_view_indices for _ in range(selected_per_view)),
        dtype=torch.long,
        device=images.device,
    )
    query_tokens = torch.empty(
        images.shape[0],
        query_count,
        dtype=torch.long,
        device=images.device,
    )
    for batch_index in range(images.shape[0]):
        cursor = 0
        for _view in eligible_view_indices:
            selected = torch.randperm(merged_count, generator=generator)[:selected_per_view]
            query_tokens[batch_index, cursor : cursor + selected_per_view] = (
                selected.sort().values.to(images.device)
            )
            cursor += selected_per_view
    query_valid = image_valid[:, query_views] if query_count else image_valid[:, :0]
    merged = torch.zeros(
        *images.shape[:2],
        merged_count,
        dtype=torch.bool,
        device=images.device,
    )
    if query_count:
        batch_indices = torch.arange(images.shape[0], device=images.device)[:, None]
        merged[batch_indices, query_views[None, :], query_tokens] = query_valid
    raw = merged.repeat_interleave(merge_unit, dim=-1)
    return QwenPackedPatchMask(
        merged_target_mask=merged,
        raw_target_mask=raw,
        image_grid_thw=image_grid_thw.detach().clone(),
        image_valid=image_valid.detach().clone(),
        query_view_indices=query_views,
        query_token_indices=query_tokens,
        query_valid=query_valid,
        eligible_view_indices=eligible_view_indices,
        spatial_merge_size=spatial_merge_size,
        seed=seed,
        probability=float(probability),
    )


def qwen_mask_query_addresses(
    plan: QwenPackedPatchMask,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map selected merged tokens to normalized ``(x, y)`` cell centers."""

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Qwen query addresses require a floating dtype")
    if plan.query_count == 0:
        return torch.empty(
            plan.image_valid.shape[0],
            0,
            2,
            dtype=dtype,
            device=plan.image_valid.device,
        )
    selected_grids = plan.image_grid_thw[:, plan.query_view_indices]
    if (selected_grids[..., 0] != 1).any():
        raise ValueError("current-grid Qwen prediction requires single-frame views")
    rows = selected_grids[..., 1] // plan.spatial_merge_size
    columns = selected_grids[..., 2] // plan.spatial_merge_size
    if (rows * columns != plan.merged_target_mask.shape[-1]).any():
        raise ValueError("Qwen query geometry differs from the merged token axis")
    token_rows = torch.div(plan.query_token_indices, columns, rounding_mode="floor")
    token_columns = plan.query_token_indices.remainder(columns)
    x = (token_columns.to(dtype) + 0.5) * (2.0 / columns.to(dtype)) - 1.0
    y = (token_rows.to(dtype) + 0.5) * (2.0 / rows.to(dtype)) - 1.0
    return torch.stack((x, y), dim=-1)


def apply_qwen_packed_patch_mask(
    images: torch.Tensor,
    plan: QwenPackedPatchMask,
    *,
    replacement: float = 0.0,
) -> torch.Tensor:
    """Remove target content before Qwen's patch embedding and global attention."""

    if images.ndim != 4 or images.shape[:3] != plan.raw_target_mask.shape:
        raise ValueError("packed Qwen images differ from the source-mask plan")
    if images.device != plan.raw_target_mask.device or not images.is_floating_point():
        raise ValueError("packed Qwen images and source mask must share device and floating dtype")
    if not isinstance(replacement, (int, float)) or not math.isfinite(replacement):
        raise ValueError("Qwen source-mask replacement must be finite")
    return images.masked_fill(plan.raw_target_mask.unsqueeze(-1), float(replacement))


def qwen_source_masked_model_inputs(
    model_inputs: Mapping[str, Any],
    plan: QwenPackedPatchMask,
    *,
    replacement: float = 0.0,
) -> dict[str, Any]:
    """Clone the official input mapping while changing only packed image content."""

    if "images" not in model_inputs:
        raise ValueError("official LingBot inputs contain no packed images")
    result = dict(model_inputs)
    result["images"] = apply_qwen_packed_patch_mask(
        model_inputs["images"],
        plan,
        replacement=replacement,
    )
    if any(result[key] is not value for key, value in model_inputs.items() if key != "images"):
        raise RuntimeError("source masking changed an undeclared LingBot input")
    return result


def qwen_whole_view_omitted_model_inputs(
    model_inputs: Mapping[str, Any],
    plan: QwenWholeViewOmission,
) -> dict[str, Any]:
    """Use the released missing-image sentinel and validity mask for one full view."""

    if not isinstance(plan, QwenWholeViewOmission):
        raise TypeError("whole-view omission requires a QwenWholeViewOmission")
    images = model_inputs.get("images")
    image_valid = model_inputs.get("img_masks")
    image_grid_thw = model_inputs.get("image_grid_thw")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("official LingBot inputs contain no packed Qwen images")
    if (
        not isinstance(image_valid, torch.Tensor)
        or image_valid.dtype != torch.bool
        or not isinstance(image_grid_thw, torch.Tensor)
        or image_grid_thw.dtype != torch.long
    ):
        raise ValueError("official LingBot inputs contain invalid image validity or geometry")
    if images.device != image_valid.device or images.device != image_grid_thw.device:
        raise ValueError("official LingBot image inputs must share one device")
    if not torch.equal(image_valid, plan.image_valid) or not torch.equal(
        image_grid_thw, plan.image_grid_thw
    ):
        raise ValueError("whole-view omission plan differs from official LingBot inputs")
    omitted_images = images.clone()
    omitted_images[:, plan.omitted_view_index].fill_(-1.0)
    result = dict(model_inputs)
    result["images"] = omitted_images
    result["img_masks"] = plan.source_image_valid
    if any(
        result[key] is not value
        for key, value in model_inputs.items()
        if key not in {"images", "img_masks"}
    ):
        raise RuntimeError("whole-view omission changed an undeclared LingBot input")
    return result
