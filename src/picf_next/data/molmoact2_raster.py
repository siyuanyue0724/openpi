"""Project exclusive raster labels onto MolmoAct2 resize-mode vision supports.

The implementation follows the geometry emitted by the official
``MolmoAct2ImageProcessor`` at the pinned source revision. It does not import or
copy upstream code. The official processor metadata remains the source of truth
and is validated fail-closed at this boundary.
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
class MolmoAct2RasterTargets:
    """Targets for pre-pooling ViT patches and official pooled image tokens."""

    patch: ProjectedRasterMembership
    pooled: ProjectedRasterMembership
    resized_shape: tuple[int, int]
    patch_grid: tuple[int, int]
    pooled_grid: tuple[int, int]


def project_molmoact2_resize_segmentation(
    segmentation: np.ndarray,
    *,
    instance_ids: tuple[int, ...],
    image_token_pooling: np.ndarray,
    image_grid: np.ndarray,
    image_num_crops: int,
    resized_shape: tuple[int, int] = (378, 378),
    patch_size: int = 14,
    pixel_supervised: np.ndarray | None = None,
    minimum_supervised_fraction: float = 1.0,
    pooling_attention_mask: bool = True,
) -> MolmoAct2RasterTargets:
    """Build exact resize-grid and conservative pooled-support targets.

    The released MolmoAct2 processor uses one 378x378 resize crop, 14x14 ViT
    patches and 2x2 masked connector pooling. Every pre-pooling patch is retained
    in ``patch``. ``pooled`` follows the processor's explicit pooling indices and
    averages the spatial target over valid source patches; it does not claim to
    reproduce the connector's learned content-attention weights.
    """

    segmentation = np.asarray(segmentation)
    pooling = np.asarray(image_token_pooling)
    grid = np.asarray(image_grid)
    if segmentation.ndim != 2 or not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError("segmentation must be an integer height-by-width raster")
    instance_ids = _canonical_instance_ids(instance_ids)
    if (
        isinstance(minimum_supervised_fraction, bool | np.bool_)
        or not isinstance(minimum_supervised_fraction, Real)
        or not np.isfinite(minimum_supervised_fraction)
        or not 0.0 <= minimum_supervised_fraction <= 1.0
    ):
        raise ValueError("minimum supervised fraction must lie in [0, 1]")
    if (
        not isinstance(image_num_crops, int)
        or isinstance(image_num_crops, bool)
        or image_num_crops != 1
    ):
        raise ValueError("MolmoAct2 resize target requires exactly one processor crop")
    if not isinstance(pooling_attention_mask, bool) or not pooling_attention_mask:
        raise ValueError("released MolmoAct2 targets require masked connector pooling")
    if (
        grid.shape != (4,)
        or not np.issubdtype(grid.dtype, np.integer)
        or np.issubdtype(grid.dtype, np.bool_)
        or int(grid[0]) <= 0
        or int(grid[1]) <= 0
        or int(grid[2]) != 0
        or int(grid[3]) != 0
    ):
        raise ValueError("MolmoAct2 resize target requires a low-resolution-only image grid")
    if pooling.ndim != 2 or not np.issubdtype(pooling.dtype, np.integer):
        raise ValueError("image token pooling must be an integer token-by-support array")

    resized_height, resized_width = resized_shape
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (resized_height, resized_width, patch_size)
    ):
        raise ValueError("resize and patch dimensions must be positive")
    if resized_height % patch_size or resized_width % patch_size:
        raise ValueError("resized image must be divisible by the vision patch size")
    patch_rows = resized_height // patch_size
    patch_columns = resized_width // patch_size
    patch_count = patch_rows * patch_columns
    pooled_rows, pooled_columns = int(grid[0]), int(grid[1])
    if pooled_rows * pooled_columns != pooling.shape[0]:
        raise ValueError("image grid does not match the processor pooling token count")
    if ((pooling < -1) | (pooling >= patch_count)).any():
        raise ValueError("image pooling contains an invalid patch index")
    flat_valid_indices = pooling[pooling >= 0]
    if not np.array_equal(np.sort(flat_valid_indices), np.arange(patch_count)):
        raise ValueError("resize pooling must cover every native patch exactly once")

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

    resized_category = resize_bilinear_channels(category_mass, resized_height, resized_width)
    resized_supervised = resize_bilinear_channels(supervised_mass, resized_height, resized_width)[
        ..., 0
    ]
    patch_category = (
        resized_category.reshape(
            patch_rows,
            patch_size,
            patch_columns,
            patch_size,
            category_count,
        )
        .mean(axis=(1, 3))
        .reshape(patch_count, category_count)
    )
    patch_supervised = (
        resized_supervised.reshape(patch_rows, patch_size, patch_columns, patch_size)
        .mean(axis=(1, 3))
        .reshape(patch_count)
    )
    patch_valid = np.ones(patch_count, dtype=np.bool_)
    patch_target = projected_membership_from_mass(
        instance_ids=instance_ids,
        category_mass=patch_category,
        supervised_mass=patch_supervised,
        token_valid=patch_valid,
        minimum_supervised_fraction=minimum_supervised_fraction,
    )

    pooled_category = np.zeros((pooling.shape[0], category_count), dtype=np.float64)
    pooled_supervised = np.zeros(pooling.shape[0], dtype=np.float64)
    pooled_valid = np.asarray((pooling >= 0).any(axis=-1), dtype=np.bool_)
    for token_index, patch_indices in enumerate(pooling):
        valid_indices = patch_indices[patch_indices >= 0]
        if not len(valid_indices):
            continue
        pooled_category[token_index] = patch_category[valid_indices].mean(axis=0)
        pooled_supervised[token_index] = patch_supervised[valid_indices].mean()
    pooled_target = projected_membership_from_mass(
        instance_ids=instance_ids,
        category_mass=pooled_category,
        supervised_mass=pooled_supervised,
        token_valid=pooled_valid,
        minimum_supervised_fraction=minimum_supervised_fraction,
    )
    return MolmoAct2RasterTargets(
        patch=patch_target,
        pooled=pooled_target,
        resized_shape=resized_shape,
        patch_grid=(patch_rows, patch_columns),
        pooled_grid=(pooled_rows, pooled_columns),
    )
