"""Project exclusive raster instance labels onto an explicit token support grid."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True, slots=True)
class ProjectedRasterMembership:
    """Categorical object-plus-context membership for visible raster instances."""

    instance_ids: tuple[int, ...]
    object_probability: np.ndarray
    context_probability: np.ndarray
    observed_fraction: np.ndarray
    token_valid: np.ndarray
    supervised: np.ndarray


def _canonical_instance_ids(instance_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Validate physical raster identities without accepting bool-as-integer."""

    if not isinstance(instance_ids, tuple):
        raise ValueError("instance IDs must be a tuple of integers")
    if any(
        isinstance(instance_id, bool | np.bool_) or not isinstance(instance_id, Integral)
        for instance_id in instance_ids
    ):
        raise ValueError("instance IDs must be integers")
    canonical = tuple(int(instance_id) for instance_id in instance_ids)
    if len(set(canonical)) != len(canonical):
        raise ValueError("instance IDs must be unique")
    return canonical


def resize_bilinear_channels(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize HWC channels with half-pixel, edge-padded bilinear semantics."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("bilinear input must be height-by-width-by-channel")
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (height, width)
        )
        or min(*values.shape[:2]) <= 0
    ):
        raise ValueError("bilinear input and output dimensions must be positive")

    input_height, input_width = values.shape[:2]
    source_y = (np.arange(height, dtype=np.float64) + 0.5) * input_height / height - 0.5
    source_x = (np.arange(width, dtype=np.float64) + 0.5) * input_width / width - 0.5
    y0_raw = np.floor(source_y).astype(np.int64)
    x0_raw = np.floor(source_x).astype(np.int64)
    y1_raw = y0_raw + 1
    x1_raw = x0_raw + 1
    wy = source_y - y0_raw
    wx = source_x - x0_raw
    y0 = np.clip(y0_raw, 0, input_height - 1)
    y1 = np.clip(y1_raw, 0, input_height - 1)
    x0 = np.clip(x0_raw, 0, input_width - 1)
    x1 = np.clip(x1_raw, 0, input_width - 1)

    top = (
        values[y0[:, None], x0[None, :]] * (1.0 - wx)[None, :, None]
        + values[y0[:, None], x1[None, :]] * wx[None, :, None]
    )
    bottom = (
        values[y1[:, None], x0[None, :]] * (1.0 - wx)[None, :, None]
        + values[y1[:, None], x1[None, :]] * wx[None, :, None]
    )
    return top * (1.0 - wy)[:, None, None] + bottom * wy[:, None, None]


def projected_membership_from_mass(
    *,
    instance_ids: tuple[int, ...],
    category_mass: np.ndarray,
    supervised_mass: np.ndarray,
    token_valid: np.ndarray,
    minimum_supervised_fraction: float,
) -> ProjectedRasterMembership:
    """Normalize object-plus-context mass on sufficiently supervised tokens."""

    instance_ids = _canonical_instance_ids(instance_ids)
    category_mass = np.asarray(category_mass, dtype=np.float64)
    supervised_mass = np.asarray(supervised_mass, dtype=np.float64)
    token_valid = np.asarray(token_valid)
    if category_mass.ndim != 2 or category_mass.shape[1] != len(instance_ids) + 1:
        raise ValueError("category mass must contain every object plus context")
    token_count = category_mass.shape[0]
    if supervised_mass.shape != (token_count,) or token_valid.shape != (token_count,):
        raise ValueError("support and validity must align with category mass")
    if token_valid.dtype != np.bool_:
        raise ValueError("token validity must be boolean")
    if (
        isinstance(minimum_supervised_fraction, bool | np.bool_)
        or not isinstance(minimum_supervised_fraction, Real)
        or not np.isfinite(minimum_supervised_fraction)
        or not 0.0 <= minimum_supervised_fraction <= 1.0
    ):
        raise ValueError("minimum supervised fraction must lie in [0, 1]")
    if (
        not np.isfinite(category_mass).all()
        or not np.isfinite(supervised_mass).all()
        or (category_mass < 0).any()
        or (supervised_mass < 0).any()
        or (supervised_mass > 1.0 + 1e-6).any()
    ):
        raise ValueError("projected mass must be finite and lie in its probability range")
    if not np.allclose(
        category_mass.sum(axis=-1),
        supervised_mass,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("object-plus-context mass must equal observed mass")

    supervised = (
        token_valid & (supervised_mass > 0) & (supervised_mass >= minimum_supervised_fraction)
    )
    observed_fraction = np.where(supervised, supervised_mass, 0.0).astype(np.float32)
    probability = np.zeros_like(category_mass, dtype=np.float64)
    probability[supervised] = category_mass[supervised] / supervised_mass[supervised, None]
    probability = np.clip(probability, 0.0, 1.0)
    if supervised.any():
        total = probability[supervised].sum(axis=-1, keepdims=True)
        if (total <= 0).any():
            raise ValueError("supervised projected tokens require positive category mass")
        probability[supervised] /= total

    visible = probability[supervised, :-1].sum(axis=0) > 0.0
    visible_ids = tuple(
        instance_id
        for instance_id, is_visible in zip(instance_ids, visible, strict=True)
        if is_visible
    )
    object_probability = probability[:, :-1][:, visible].astype(np.float32)
    context_probability = probability[:, -1].astype(np.float32)
    if supervised.any():
        total = object_probability.sum(axis=-1) + context_probability
        if not np.allclose(total[supervised], 1.0, atol=1e-6, rtol=1e-6):
            raise RuntimeError("projected categorical target does not sum to one")
    return ProjectedRasterMembership(
        instance_ids=visible_ids,
        object_probability=object_probability,
        context_probability=context_probability,
        observed_fraction=observed_fraction,
        token_valid=token_valid.copy(),
        supervised=supervised,
    )


def regular_grid_pixel_boxes(*, height: int, width: int, rows: int, columns: int) -> np.ndarray:
    """Return non-overlapping ``[y0, x0, y1, x1]`` boxes in row-major order."""

    dimensions = (height, width, rows, columns)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions
    ):
        raise ValueError("image and grid dimensions must be positive")
    if rows > height or columns > width:
        raise ValueError("a regular token grid cannot exceed the raster resolution")
    y_edges = np.linspace(0, height, rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, columns + 1, dtype=np.int64)
    return np.asarray(
        [
            (y_edges[row], x_edges[column], y_edges[row + 1], x_edges[column + 1])
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.int64,
    )


def project_exclusive_segmentation(
    segmentation: np.ndarray,
    *,
    instance_ids: tuple[int, ...],
    token_boxes_yxyx: np.ndarray,
    token_valid: np.ndarray | None = None,
    pixel_supervised: np.ndarray | None = None,
    minimum_supervised_fraction: float = 1.0,
) -> ProjectedRasterMembership:
    """Aggregate one-visible-owner-per-pixel labels into token distributions.

    ``instance_ids`` contains dataset-local physical identities selected by the
    dataset adapter. Every other known pixel is context. Instances with no
    supervised visible support are omitted from the returned current-frame set;
    temporal persistence belongs to the posterior rather than this observation.
    """

    segmentation = np.asarray(segmentation)
    boxes = np.asarray(token_boxes_yxyx)
    if segmentation.ndim != 2 or not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError("segmentation must be an integer height-by-width raster")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.issubdtype(boxes.dtype, np.integer):
        raise ValueError("token boxes must be an integer token-by-four array")
    instance_ids = _canonical_instance_ids(instance_ids)
    if (
        isinstance(minimum_supervised_fraction, bool | np.bool_)
        or not isinstance(minimum_supervised_fraction, Real)
        or not np.isfinite(minimum_supervised_fraction)
        or not 0.0 <= minimum_supervised_fraction <= 1.0
    ):
        raise ValueError("minimum supervised fraction must lie in [0, 1]")

    token_count = boxes.shape[0]
    if token_valid is None:
        valid_mask = np.ones(token_count, dtype=np.bool_)
    else:
        valid_mask = np.asarray(token_valid)
        if valid_mask.dtype != np.bool_ or valid_mask.shape != (token_count,):
            raise ValueError("token validity must be a bool token vector")
    if pixel_supervised is None:
        supervision_mask = np.ones(segmentation.shape, dtype=np.bool_)
    else:
        supervision_mask = np.asarray(pixel_supervised)
        if supervision_mask.dtype != np.bool_ or supervision_mask.shape != segmentation.shape:
            raise ValueError("pixel supervision must be a bool raster aligned to segmentation")

    height, width = segmentation.shape
    object_probability = np.zeros((token_count, len(instance_ids)), dtype=np.float32)
    context_probability = np.zeros(token_count, dtype=np.float32)
    observed_fraction = np.zeros(token_count, dtype=np.float32)
    supervised = np.zeros(token_count, dtype=np.bool_)
    for token_index, (y0, x0, y1, x1) in enumerate(boxes.tolist()):
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("token boxes must be nonempty and contained in the raster")
        if not valid_mask[token_index]:
            continue
        label_region = segmentation[y0:y1, x0:x1]
        known_region = supervision_mask[y0:y1, x0:x1]
        known_count = int(known_region.sum())
        coverage = known_count / int(known_region.size)
        if known_count == 0 or coverage < minimum_supervised_fraction:
            continue
        supervised[token_index] = True
        observed_fraction[token_index] = coverage
        for object_index, instance_id in enumerate(instance_ids):
            object_probability[token_index, object_index] = float(
                ((label_region == instance_id) & known_region).sum() / known_count
            )
        context_probability[token_index] = float(
            1.0 - object_probability[token_index].sum(dtype=np.float64)
        )

    visible = object_probability.sum(axis=0) > 0.0
    visible_ids = tuple(
        instance_id
        for instance_id, is_visible in zip(instance_ids, visible, strict=True)
        if is_visible
    )
    object_probability = object_probability[:, visible]
    object_probability[~supervised] = 0.0
    context_probability[~supervised] = 0.0
    if supervised.any():
        total = object_probability.sum(axis=-1) + context_probability
        if not np.allclose(total[supervised], 1.0, atol=1e-6, rtol=1e-6):
            raise RuntimeError("projected categorical membership does not sum to one")
    return ProjectedRasterMembership(
        instance_ids=visible_ids,
        object_probability=object_probability,
        context_probability=context_probability,
        observed_fraction=observed_fraction,
        token_valid=valid_mask.copy(),
        supervised=supervised,
    )
