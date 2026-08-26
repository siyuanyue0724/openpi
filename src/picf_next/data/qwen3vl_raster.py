"""Project exclusive raster labels onto Qwen3-VL merged visual tokens.

The pinned Qwen processor packs spatial patches in
``group_h, group_w, merge_h, merge_w`` order.  Qwen's visual merger therefore
turns every consecutive ``merge_size**2`` raw patches into one row-major
spatial token.  This adapter projects labels onto that exact output address
grid; it never supplies labels or masks to the visual forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from picf_next.data.raster_targets import (
    ProjectedRasterMembership,
    _canonical_instance_ids,
    projected_membership_from_mass,
    resize_bilinear_channels,
)


@dataclass(frozen=True, slots=True)
class Qwen3VLRasterTargets:
    """Categorical labels aligned with final Qwen visual-token addresses."""

    merged: ProjectedRasterMembership
    resized_shape: tuple[int, int]
    raw_patch_grid: tuple[int, int]
    merged_grid: tuple[int, int]
    patch_size: int
    merge_size: int


def project_qwen3vl_segmentation(
    segmentation: np.ndarray,
    *,
    instance_ids: tuple[int, ...],
    image_grid_thw: np.ndarray,
    patch_size: int,
    merge_size: int,
    pixel_supervised: np.ndarray | None = None,
    minimum_supervised_fraction: float = 1.0,
) -> Qwen3VLRasterTargets:
    """Build soft object/context targets for one pinned Qwen image grid."""

    segmentation = np.asarray(segmentation)
    grid = np.asarray(image_grid_thw)
    if segmentation.ndim != 2 or not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError("segmentation must be an integer height-by-width raster")
    instance_ids = _canonical_instance_ids(instance_ids)
    if (
        grid.shape != (3,)
        or not np.issubdtype(grid.dtype, np.integer)
        or np.issubdtype(grid.dtype, np.bool_)
    ):
        raise ValueError("Qwen image grid must be an integer [time,height,width] vector")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (patch_size, merge_size)
    ):
        raise ValueError("Qwen patch and merge sizes must be positive integers")
    grid_t, grid_h, grid_w = (int(value) for value in grid)
    if grid_t != 1:
        raise ValueError("single-frame raster supervision requires Qwen grid time one")
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Qwen spatial grid must divide exactly by merge size")
    if (
        isinstance(minimum_supervised_fraction, bool | np.bool_)
        or not isinstance(minimum_supervised_fraction, Real)
        or not np.isfinite(minimum_supervised_fraction)
        or not 0.0 <= minimum_supervised_fraction <= 1.0
    ):
        raise ValueError("minimum supervised fraction must lie in [0, 1]")
    if pixel_supervised is None:
        supervised_pixels = np.ones(segmentation.shape, dtype=np.bool_)
    else:
        supervised_pixels = np.asarray(pixel_supervised)
        if supervised_pixels.dtype != np.bool_ or supervised_pixels.shape != segmentation.shape:
            raise ValueError("pixel supervision must be a bool raster aligned to segmentation")

    category_count = len(instance_ids) + 1
    category_mass = np.zeros((*segmentation.shape, category_count), dtype=np.float64)
    selected = np.zeros(segmentation.shape, dtype=np.bool_)
    for object_index, instance_id in enumerate(instance_ids):
        mask = segmentation == instance_id
        selected |= mask
        category_mass[..., object_index] = mask
    category_mass[..., -1] = ~selected
    category_mass *= supervised_pixels[..., None]
    supervised_mass = supervised_pixels[..., None].astype(np.float64)

    resized_height = grid_h * patch_size
    resized_width = grid_w * patch_size
    resized_category = resize_bilinear_channels(category_mass, resized_height, resized_width)
    resized_supervised = resize_bilinear_channels(
        supervised_mass,
        resized_height,
        resized_width,
    )[..., 0]
    merged_rows = grid_h // merge_size
    merged_columns = grid_w // merge_size
    cell = patch_size * merge_size
    token_count = merged_rows * merged_columns
    merged_category = (
        resized_category.reshape(
            merged_rows,
            cell,
            merged_columns,
            cell,
            category_count,
        )
        .mean(axis=(1, 3))
        .reshape(token_count, category_count)
    )
    merged_supervised = (
        resized_supervised.reshape(merged_rows, cell, merged_columns, cell)
        .mean(axis=(1, 3))
        .reshape(token_count)
    )
    target = projected_membership_from_mass(
        instance_ids=instance_ids,
        category_mass=merged_category,
        supervised_mass=merged_supervised,
        token_valid=np.ones(token_count, dtype=np.bool_),
        minimum_supervised_fraction=float(minimum_supervised_fraction),
    )
    return Qwen3VLRasterTargets(
        merged=target,
        resized_shape=(resized_height, resized_width),
        raw_patch_grid=(grid_h, grid_w),
        merged_grid=(merged_rows, merged_columns),
        patch_size=patch_size,
        merge_size=merge_size,
    )
