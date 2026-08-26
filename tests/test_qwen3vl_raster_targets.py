from __future__ import annotations

import numpy as np
import pytest

from picf_next.data.qwen3vl_raster import project_qwen3vl_segmentation


def test_qwen_projection_follows_official_block_major_merger_addresses() -> None:
    segmentation = np.asarray(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.uint8,
    )
    target = project_qwen3vl_segmentation(
        segmentation,
        instance_ids=(1, 2, 3, 4),
        image_grid_thw=np.asarray([1, 4, 4], dtype=np.int64),
        patch_size=1,
        merge_size=2,
    )

    assert target.resized_shape == (4, 4)
    assert target.raw_patch_grid == (4, 4)
    assert target.merged_grid == (2, 2)
    assert target.merged.instance_ids == (1, 2, 3, 4)
    np.testing.assert_array_equal(target.merged.object_probability, np.eye(4, dtype=np.float32))
    np.testing.assert_array_equal(target.merged.context_probability, np.zeros(4))
    np.testing.assert_array_equal(target.merged.supervised, np.ones(4, dtype=np.bool_))


def test_qwen_projection_preserves_soft_boundary_mass_and_context() -> None:
    segmentation = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    target = project_qwen3vl_segmentation(
        segmentation,
        instance_ids=(1,),
        image_grid_thw=np.asarray([1, 2, 2], dtype=np.int64),
        patch_size=1,
        merge_size=2,
    ).merged

    np.testing.assert_allclose(target.object_probability[:, 0], [0.25])
    np.testing.assert_allclose(target.context_probability, [0.75])
    np.testing.assert_allclose(
        target.object_probability.sum(axis=-1) + target.context_probability,
        [1.0],
    )


def test_qwen_projection_fails_closed_on_partially_unknown_support() -> None:
    segmentation = np.asarray([[1, 1], [1, 1]], dtype=np.uint8)
    supervised = np.asarray([[True, True], [True, False]])
    strict = project_qwen3vl_segmentation(
        segmentation,
        instance_ids=(1,),
        image_grid_thw=np.asarray([1, 2, 2], dtype=np.int64),
        patch_size=1,
        merge_size=2,
        pixel_supervised=supervised,
    ).merged
    permissive = project_qwen3vl_segmentation(
        segmentation,
        instance_ids=(1,),
        image_grid_thw=np.asarray([1, 2, 2], dtype=np.int64),
        patch_size=1,
        merge_size=2,
        pixel_supervised=supervised,
        minimum_supervised_fraction=0.75,
    ).merged

    assert not strict.supervised.any()
    assert permissive.supervised.all()
    np.testing.assert_allclose(permissive.observed_fraction, [0.75])
    np.testing.assert_allclose(permissive.object_probability, [[1.0]])


def test_qwen_partial_token_preserves_raw_known_pixel_mass_without_unknown_leakage() -> None:
    target = project_qwen3vl_segmentation(
        np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
        instance_ids=(1,),
        image_grid_thw=np.asarray([1, 2, 2], dtype=np.int64),
        patch_size=1,
        merge_size=2,
        pixel_supervised=np.asarray([[True, True], [True, False]]),
        minimum_supervised_fraction=0.0,
    ).merged

    assert target.supervised.tolist() == [True]
    np.testing.assert_allclose(target.observed_fraction, [0.75])
    np.testing.assert_allclose(target.object_probability, [[1.0 / 3.0]])
    np.testing.assert_allclose(target.context_probability, [2.0 / 3.0])
    np.testing.assert_allclose(
        target.observed_fraction[:, None] * target.object_probability,
        [[0.25]],
    )
    np.testing.assert_allclose(
        target.observed_fraction * target.context_probability,
        [0.5],
    )


def test_qwen_projection_derives_resize_from_processor_grid() -> None:
    target = project_qwen3vl_segmentation(
        np.zeros((200, 200), dtype=np.uint8),
        instance_ids=(),
        image_grid_thw=np.asarray([1, 16, 16], dtype=np.int64),
        patch_size=16,
        merge_size=2,
    )

    assert target.resized_shape == (256, 256)
    assert target.merged_grid == (8, 8)
    assert target.merged.object_probability.shape == (64, 0)
    np.testing.assert_allclose(target.merged.context_probability, np.ones(64))


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        (np.asarray([2, 4, 4]), "time one"),
        (np.asarray([1, 3, 4]), "divide exactly"),
    ],
)
def test_qwen_projection_rejects_unsupported_grid(grid: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        project_qwen3vl_segmentation(
            np.zeros((4, 4), dtype=np.uint8),
            instance_ids=(),
            image_grid_thw=grid,
            patch_size=1,
            merge_size=2,
        )
