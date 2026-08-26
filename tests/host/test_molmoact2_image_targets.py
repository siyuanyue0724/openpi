from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
functional = pytest.importorskip("torch.nn.functional")
image_processing = pytest.importorskip("olmo.hf_model.image_processing_molmoact2")
targets = pytest.importorskip("picf_next.data.molmoact2_raster")

MolmoAct2ImageProcessor = image_processing.MolmoAct2ImageProcessor
_resize_bilinear_channels = targets._resize_bilinear_channels
project_molmoact2_resize_segmentation = targets.project_molmoact2_resize_segmentation


def test_numpy_resize_matches_official_torch_half_pixel_bilinear() -> None:
    rng = np.random.default_rng(17)
    values = rng.random((5, 7, 3), dtype=np.float32)
    expected = (
        functional.interpolate(
            torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0),
            size=(9, 11),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )

    actual = _resize_bilinear_channels(values, 9, 11)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)


def test_released_processor_metadata_yields_729_dense_and_196_pooled_targets() -> None:
    processor = MolmoAct2ImageProcessor(crop_mode="resize")
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    metadata = processor(images=[rgb], return_tensors="np")
    segmentation = np.zeros((200, 200), dtype=np.int64)
    segmentation[50:150, 60:140] = 7

    target = project_molmoact2_resize_segmentation(
        segmentation,
        instance_ids=(7,),
        image_token_pooling=metadata["image_token_pooling"],
        image_grid=metadata["image_grids"][0],
        image_num_crops=int(metadata["image_num_crops"][0]),
    )

    assert metadata["pixel_values"].shape == (1, 729, 588)
    assert target.patch_grid == (27, 27)
    assert target.patch.object_probability.shape == (729, 1)
    assert target.patch.token_valid.all()
    assert target.pooled_grid == (14, 14)
    assert target.pooled.object_probability.shape == (196, 1)
    assert target.pooled.token_valid.all()
    np.testing.assert_allclose(
        target.patch.object_probability.sum(axis=-1) + target.patch.context_probability,
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        target.pooled.object_probability.sum(axis=-1) + target.pooled.context_probability,
        1.0,
        atol=1e-6,
    )
