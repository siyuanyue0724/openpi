from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.encoders.anytouch2 import (
    ANYTOUCH2_GEOMETRY_WIDTH,
    ANYTOUCH2_TOKENS_PER_SENSOR,
    AnyTouch2DenseConfig,
    anytouch2_token_metadata,
)
from picf_next.encoders.spatiallm_sonata import (
    SPATIALLM_SONATA_FULL_GEOMETRY_WIDTH,
    SPATIALLM_SONATA_NATIVE_GEOMETRY_WIDTH,
    SpatialLMSonataConfig,
    normalize_sonata_colors,
    sonata_grid_coordinates,
)
from picf_next.encoders.vjepa21 import (
    VJEPA21_CALVIN_GEOMETRY_WIDTH,
    VJEPA21_CALVIN_VIEW_NAMES,
    Vjepa21DenseConfig,
    Vjepa21DenseEncoder,
    combine_vjepa21_calvin_views,
    vjepa21_current_grid_geometry,
    vjepa21_current_timestamp,
)


def test_vjepa21_current_grid_is_complete_ordered_and_normalized() -> None:
    geometry = vjepa21_current_grid_geometry(24)

    assert geometry.shape == (576, 2)
    np.testing.assert_allclose(geometry[0], [(0.5 / 24), (0.5 / 24)])
    np.testing.assert_allclose(geometry[-1], [(23.5 / 24), (23.5 / 24)])
    assert np.unique(geometry, axis=0).shape[0] == 576
    assert not geometry.flags.writeable


def test_vjepa21_current_timestamp_is_causal_final_tubelet_center() -> None:
    timestamps = np.arange(64, dtype=np.float32) / 30.0

    assert vjepa21_current_timestamp(timestamps) == pytest.approx(float(timestamps[-2:].mean()))
    with pytest.raises(ContractError, match="chronological"):
        vjepa21_current_timestamp(timestamps[::-1])


def test_vjepa21_batched_forward_matches_independent_forwards() -> None:
    torch = pytest.importorskip("torch")
    config = Vjepa21DenseConfig(
        image_size=4,
        frame_count=2,
        patch_size=2,
        tubelet_size=1,
        feature_mode="final",
        dtype="float32",
    )

    class _Model:
        def __call__(self, video, *, training):
            assert not training
            count = (config.frame_count // config.tubelet_size) * config.token_count
            base = torch.arange(
                count * config.token_width,
                dtype=video.dtype,
                device=video.device,
            ).reshape(1, count, config.token_width)
            return base + video.mean(dim=(1, 2, 3, 4), keepdim=False)[:, None, None]

    encoder = Vjepa21DenseEncoder(
        model=_Model(),
        torch=torch,
        device="cpu",
        config=config,
        checkpoint_path=Path("fixture.pt"),
        checkpoint_sha256="a" * 64,
        encoder_contract="fixture/v1",
    )
    clips = (
        tuple(np.full((3, 5, 3), value, dtype=np.uint8) for _ in range(2)) for value in (32, 192)
    )
    first, second = tuple(clips)
    timestamps = ((0.0, 1.0), (2.0, 3.0))

    batched = encoder.encode_clips((first, second), timestamps)
    independent = (
        encoder.encode_clip(first, timestamps[0]),
        encoder.encode_clip(second, timestamps[1]),
    )

    for actual, expected in zip(batched, independent, strict=True):
        np.testing.assert_array_equal(actual.tokens, expected.tokens)
        np.testing.assert_array_equal(actual.timestamps, expected.timestamps)


def _vjepa_view(value: float) -> DenseEvidence:
    config = Vjepa21DenseConfig()
    count = config.token_count
    return DenseEvidence(
        modality="vjepa",
        encoder_contract="vjepa2.1.test/final/v1",
        tokens=np.full((count, config.token_width), value, dtype=np.float32),
        available=True,
        timestamps=np.full(count, 1.0, dtype=np.float32),
        confidence=np.ones(count, dtype=np.float32),
        geometry=vjepa21_current_grid_geometry(config.grid_size),
        current_measurement_valid=np.ones(count, dtype=np.bool_),
    )


def test_vjepa21_calvin_views_keep_both_grids_and_explicit_camera_identity() -> None:
    combined = combine_vjepa21_calvin_views(
        {"gripper": _vjepa_view(2.0), "static": _vjepa_view(1.0)}
    )
    per_view = Vjepa21DenseConfig().token_count

    assert combined.token_count == 2 * per_view
    assert combined.geometry is not None
    assert combined.geometry.shape == (2 * per_view, VJEPA21_CALVIN_GEOMETRY_WIDTH)
    np.testing.assert_array_equal(combined.tokens[:per_view], 1.0)
    np.testing.assert_array_equal(combined.tokens[per_view:], 2.0)
    np.testing.assert_array_equal(
        combined.geometry[:per_view, 2:],
        np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (per_view, 1)),
    )
    np.testing.assert_array_equal(
        combined.geometry[per_view:, 2:],
        np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (per_view, 1)),
    )
    assert VJEPA21_CALVIN_VIEW_NAMES == ("static", "gripper")
    assert combined.group_ids is None
    assert not combined.tokens.flags.writeable
    with pytest.raises(ContractError, match="static and gripper"):
        combine_vjepa21_calvin_views({"static": _vjepa_view(1.0)})


def test_anytouch2_metadata_preserves_sensor_pose_and_patch_geometry() -> None:
    config = AnyTouch2DenseConfig()
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 4, axis=0)
    poses[:, 0, 3] = np.arange(4, dtype=np.float32)
    geometry, timestamps, current = anytouch2_token_metadata(
        sensor_id=1,
        sensor_poses_world=poses,
        frame_timestamps_s=(1.0, 2.0, 3.0, 4.0),
        config=config,
    )

    assert geometry.shape == (ANYTOUCH2_TOKENS_PER_SENSOR, ANYTOUCH2_GEOMETRY_WIDTH)
    assert timestamps.shape == current.shape == (ANYTOUCH2_TOKENS_PER_SENSOR,)
    assert current[:6].all()
    assert not current[6 : 6 + 14 * 14].any()
    assert current[-14 * 14 :].all()
    np.testing.assert_array_equal(timestamps[:6], np.full(6, 4.0, dtype=np.float32))
    np.testing.assert_array_equal(
        timestamps[6 : 6 + 14 * 14], np.full(14 * 14, 2.0, dtype=np.float32)
    )
    np.testing.assert_array_equal(timestamps[-14 * 14 :], np.full(14 * 14, 4.0, dtype=np.float32))
    np.testing.assert_allclose(
        geometry[:6, 6:],
        np.repeat(poses[-1, :3].reshape(1, 12), 6, axis=0),
    )
    np.testing.assert_allclose(geometry[6 : 6 + 14 * 14, 9], 1.0)
    np.testing.assert_allclose(geometry[6 + 14 * 14 :, 9], 3.0)
    assert geometry[0, 3] == 1.0
    assert (geometry[1:6, 4] == 1.0).all()
    assert np.unique(geometry[6:, :3], axis=0).shape[0] == 392
    assert not geometry.flags.writeable


def test_anytouch2_metadata_rejects_async_or_unknown_physical_contract() -> None:
    config = AnyTouch2DenseConfig()
    with pytest.raises(ContractError, match="released registry"):
        anytouch2_token_metadata(
            sensor_id=19,
            sensor_poses_world=np.repeat(np.eye(4)[None, ...], 4, axis=0),
            frame_timestamps_s=(0.0, 1.0, 2.0, 3.0),
            config=config,
        )
    with pytest.raises(ContractError, match="chronological"):
        anytouch2_token_metadata(
            sensor_id=1,
            sensor_poses_world=np.repeat(np.eye(4)[None, ...], 4, axis=0),
            frame_timestamps_s=(0.0, 2.0, 1.0, 3.0),
            config=config,
        )


def test_sonata_geometry_normalization_is_task_independent() -> None:
    xyz = np.asarray([[1.0, 2.0, 3.0], [1.01, 2.03, 3.02]], dtype=np.float32)
    grid = sonata_grid_coordinates(xyz, voxel_size_m=0.01)
    colors = normalize_sonata_colors(np.asarray([[0, 128, 255], [255, 0, 0]]))

    np.testing.assert_array_equal(grid, np.asarray([[0, 0, 0], [0, 2, 1]], dtype=np.int32))
    np.testing.assert_allclose(colors[0], [0.0, 128.0 / 255.0, 1.0])
    assert not grid.flags.writeable
    assert not colors.flags.writeable


def test_production_encoders_choose_native_pretrained_bottlenecks() -> None:
    vjepa = Vjepa21DenseConfig()
    sonata = SpatialLMSonataConfig()

    assert vjepa.feature_mode == "final"
    assert vjepa.token_width == 768
    assert not sonata.return_full_resolution
    assert sonata.geometry_width == SPATIALLM_SONATA_NATIVE_GEOMETRY_WIDTH
    assert SpatialLMSonataConfig(return_full_resolution=True).geometry_width == (
        SPATIALLM_SONATA_FULL_GEOMETRY_WIDTH
    )


def test_sonata_grid_rejects_invalid_points() -> None:
    with pytest.raises(ContractError, match="finite N-by-3"):
        sonata_grid_coordinates(np.asarray([[np.nan, 0.0, 0.0]]), voxel_size_m=0.01)


def test_vendored_hilbert_decoder_rejects_unsupported_bit_width_cleanly() -> None:
    import importlib.util

    torch = pytest.importorskip("torch")
    module_path = (
        Path(__file__).parents[1]
        / "src/picf_next/encoders/vendor/spatiallm_sonata/serialization/hilbert.py"
    )
    spec = importlib.util.spec_from_file_location("picf_test_vendored_hilbert", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="65 bits total"):
        module.decode(torch.zeros(1, dtype=torch.int64), num_dims=5, num_bits=13)
