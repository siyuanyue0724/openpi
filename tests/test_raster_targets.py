from __future__ import annotations

import numpy as np
import pytest

from picf_next.data.raster_targets import (
    project_exclusive_segmentation,
    projected_membership_from_mass,
    regular_grid_pixel_boxes,
)


def test_regular_grid_boxes_cover_raster_once() -> None:
    boxes = regular_grid_pixel_boxes(height=4, width=6, rows=2, columns=3)
    coverage = np.zeros((4, 6), dtype=np.int64)
    for y0, x0, y1, x1 in boxes:
        coverage[y0:y1, x0:x1] += 1

    assert boxes.tolist() == [
        [0, 0, 2, 2],
        [0, 2, 2, 4],
        [0, 4, 2, 6],
        [2, 0, 4, 2],
        [2, 2, 4, 4],
        [2, 4, 4, 6],
    ]
    assert np.array_equal(coverage, np.ones_like(coverage))


def test_projection_preserves_exclusive_object_and_context_mass() -> None:
    segmentation = np.asarray(
        [
            [1, 1, 2, 0],
            [1, 0, 2, 0],
        ],
        dtype=np.int64,
    )
    boxes = regular_grid_pixel_boxes(height=2, width=4, rows=1, columns=2)
    projected = project_exclusive_segmentation(
        segmentation,
        instance_ids=(1, 2, 99),
        token_boxes_yxyx=boxes,
    )

    assert projected.instance_ids == (1, 2)
    np.testing.assert_allclose(
        projected.object_probability,
        np.asarray([[0.75, 0.0], [0.0, 0.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        projected.context_probability,
        np.asarray([0.25, 0.5], dtype=np.float32),
    )
    np.testing.assert_allclose(projected.observed_fraction, np.ones(2))
    assert projected.supervised.tolist() == [True, True]


def test_partial_or_invalid_token_support_is_not_falsely_supervised() -> None:
    segmentation = np.asarray([[1, 1], [0, 0]], dtype=np.int64)
    boxes = np.asarray([[0, 0, 2, 1], [0, 1, 2, 2]], dtype=np.int64)
    pixel_supervised = np.asarray([[True, False], [True, False]])
    projected = project_exclusive_segmentation(
        segmentation,
        instance_ids=(1,),
        token_boxes_yxyx=boxes,
        token_valid=np.asarray([True, False]),
        pixel_supervised=pixel_supervised,
        minimum_supervised_fraction=1.0,
    )

    assert projected.supervised.tolist() == [True, False]
    np.testing.assert_allclose(projected.observed_fraction, [1.0, 0.0])
    np.testing.assert_allclose(projected.object_probability[:, 0], [0.5, 0.0])
    np.testing.assert_allclose(projected.context_probability, [0.5, 0.0])


def test_zero_threshold_does_not_supervise_or_divide_by_zero_mass() -> None:
    projected = projected_membership_from_mass(
        instance_ids=(1,),
        category_mass=np.asarray([[0.0, 0.0], [0.25, 0.5]]),
        supervised_mass=np.asarray([0.0, 0.75]),
        token_valid=np.ones(2, dtype=np.bool_),
        minimum_supervised_fraction=0.0,
    )

    assert projected.supervised.tolist() == [False, True]
    np.testing.assert_allclose(projected.observed_fraction, [0.0, 0.75])
    np.testing.assert_allclose(projected.object_probability[:, 0], [0.0, 1.0 / 3.0])
    np.testing.assert_allclose(projected.context_probability, [0.0, 2.0 / 3.0])


def test_rejects_out_of_bounds_token_boxes() -> None:
    with pytest.raises(ValueError, match="contained"):
        project_exclusive_segmentation(
            np.zeros((2, 2), dtype=np.int64),
            instance_ids=(),
            token_boxes_yxyx=np.asarray([[0, 0, 3, 2]], dtype=np.int64),
        )


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_rejects_invalid_supervision_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="minimum supervised fraction"):
        project_exclusive_segmentation(
            np.zeros((2, 2), dtype=np.int64),
            instance_ids=(),
            token_boxes_yxyx=np.asarray([[0, 0, 2, 2]], dtype=np.int64),
            minimum_supervised_fraction=threshold,
        )


@pytest.mark.parametrize("instance_ids", [(True,), (1.5,), [1]])
def test_rejects_non_integer_or_non_tuple_instance_ids(instance_ids: object) -> None:
    with pytest.raises(ValueError, match="instance IDs"):
        project_exclusive_segmentation(
            np.zeros((2, 2), dtype=np.int64),
            instance_ids=instance_ids,  # type: ignore[arg-type]
            token_boxes_yxyx=np.asarray([[0, 0, 2, 2]], dtype=np.int64),
        )
