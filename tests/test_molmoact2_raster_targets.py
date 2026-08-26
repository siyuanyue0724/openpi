from __future__ import annotations

import numpy as np
import pytest

from picf_next.data.molmoact2_raster import project_molmoact2_resize_segmentation


def _pooling() -> np.ndarray:
    return np.asarray([[0, 1, 2, 3]], dtype=np.int64)


def test_patch_and_pooled_targets_preserve_every_resize_patch() -> None:
    segmentation = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ],
        dtype=np.int64,
    )
    target = project_molmoact2_resize_segmentation(
        segmentation,
        instance_ids=(1, 2),
        image_token_pooling=_pooling(),
        image_grid=np.asarray([1, 1, 0, 0]),
        image_num_crops=1,
        resized_shape=(4, 4),
        patch_size=2,
    )

    assert target.patch_grid == (2, 2)
    assert target.pooled_grid == (1, 1)
    np.testing.assert_allclose(
        target.patch.object_probability,
        np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    )
    np.testing.assert_allclose(target.patch.context_probability, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(target.pooled.object_probability, [[0.25, 0.25]])
    np.testing.assert_allclose(target.pooled.context_probability, [0.5])


def test_partial_unknown_support_is_not_converted_to_context() -> None:
    segmentation = np.zeros((4, 4), dtype=np.int64)
    pixel_supervised = np.ones((4, 4), dtype=np.bool_)
    pixel_supervised[2:, 2:] = False
    target = project_molmoact2_resize_segmentation(
        segmentation,
        instance_ids=(),
        image_token_pooling=_pooling(),
        image_grid=np.asarray([1, 1, 0, 0]),
        image_num_crops=1,
        resized_shape=(4, 4),
        patch_size=2,
        pixel_supervised=pixel_supervised,
        minimum_supervised_fraction=1.0,
    )

    assert target.patch.supervised.tolist() == [True, True, True, False]
    assert target.patch.context_probability.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert target.pooled.supervised.tolist() == [False]
    assert target.pooled.context_probability.tolist() == [0.0]


def test_resize_pooling_must_partition_all_native_patches() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        project_molmoact2_resize_segmentation(
            np.zeros((4, 4), dtype=np.int64),
            instance_ids=(),
            image_token_pooling=np.asarray([[0, 1, 2, 2]]),
            image_grid=np.asarray([1, 1, 0, 0]),
            image_num_crops=1,
            resized_shape=(4, 4),
            patch_size=2,
        )


@pytest.mark.parametrize(
    "image_grid",
    [
        np.asarray([1.0, 1.0, 0.0, 0.0]),
        np.asarray([0, 1, 0, 0]),
        np.asarray([1, 1, 1, 0]),
    ],
)
def test_resize_grid_must_be_exact_positive_integer_metadata(image_grid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="low-resolution-only image grid"):
        project_molmoact2_resize_segmentation(
            np.zeros((4, 4), dtype=np.int64),
            instance_ids=(),
            image_token_pooling=_pooling(),
            image_grid=image_grid,
            image_num_crops=1,
            resized_shape=(4, 4),
            patch_size=2,
        )


@pytest.mark.parametrize(
    "threshold",
    [True, np.bool_(False), "0.5", float("nan"), float("inf"), -0.1, 1.1],
)
def test_rejects_invalid_molmo_supervision_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="minimum supervised fraction"):
        project_molmoact2_resize_segmentation(
            np.zeros((4, 4), dtype=np.int64),
            instance_ids=(),
            image_token_pooling=_pooling(),
            image_grid=np.asarray([1, 1, 0, 0]),
            image_num_crops=1,
            resized_shape=(4, 4),
            patch_size=2,
            minimum_supervised_fraction=threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("instance_ids", [(True,), (1.5,), [1]])
def test_rejects_invalid_molmo_instance_ids(instance_ids: object) -> None:
    with pytest.raises(ValueError, match="instance IDs"):
        project_molmoact2_resize_segmentation(
            np.zeros((4, 4), dtype=np.int64),
            instance_ids=instance_ids,  # type: ignore[arg-type]
            image_token_pooling=_pooling(),
            image_grid=np.asarray([1, 1, 0, 0]),
            image_num_crops=1,
            resized_shape=(4, 4),
            patch_size=2,
        )
