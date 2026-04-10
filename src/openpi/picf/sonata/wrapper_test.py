from types import SimpleNamespace

from addict import Dict
import numpy as np
import pytest
import torch

from openpi.picf.frame_context import PointFrameContext
from openpi.picf.sonata.config import SonataPointConfig
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor
from openpi.picf.sonata.wrapper import _normalize_local_grid_coords
from openpi.picf.sonata.wrapper import _restore_full_resolution_features
from openpi.picf.sonata.wrapper import sonata_runtime_available


def test_restore_full_resolution_features_unrolls_pooling_chain() -> None:
    root = SimpleNamespace(feat=torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    coarse = SimpleNamespace(
        feat=torch.tensor([[10.0], [20.0]]),
        pooling_inverse=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        pooling_parent=root,
    )
    top = SimpleNamespace(
        feat=torch.tensor([[100.0]]),
        pooling_inverse=torch.tensor([0, 0], dtype=torch.long),
        pooling_parent=coarse,
    )

    restored = _restore_full_resolution_features(top)

    assert restored.shape == (4, 1)
    assert torch.equal(restored, torch.tensor([[100.0], [100.0], [100.0], [100.0]]))


def test_restore_full_resolution_features_stops_on_missing_pooling_keys_without_autovivify() -> None:
    root = Dict(feat=torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    coarse = Dict(
        feat=torch.tensor([[10.0], [20.0]]),
        pooling_inverse=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        pooling_parent=root,
    )
    top = Dict(
        feat=torch.tensor([[100.0]]),
        pooling_inverse=torch.tensor([0, 0], dtype=torch.long),
        pooling_parent=coarse,
    )

    restored = _restore_full_resolution_features(top)

    assert restored.shape == (4, 1)
    assert torch.equal(restored, torch.tensor([[100.0], [100.0], [100.0], [100.0]]))
    assert "pooling_parent" not in root
    assert "pooling_inverse" not in root


def test_sonata_build_sample_uses_xyzrgb_feat_and_keeps_normals_side_channel() -> None:
    context = PointFrameContext(
        grid_coord=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
        points_local=np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        normals_local=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        colors=np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]], dtype=np.float32),
        local_mask=np.array([True, True]),
        world_to_local=np.eye(4, dtype=np.float32),
        G_t=np.eye(4, dtype=np.float32),
    )
    extractor = SonataPointFeatureExtractor.__new__(SonataPointFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.model = SimpleNamespace(embedding=SimpleNamespace(in_channels=6))
    extractor.config = SonataPointConfig()

    sample = extractor._build_sample(context)  # noqa: SLF001

    assert sample["feat"].shape == (2, 6)
    assert sample["normal"].shape == (2, 3)
    assert torch.allclose(
        sample["feat"].cpu(),
        torch.tensor([[0.0, 0.1, 0.2, 0.2, 0.4, 0.6], [0.3, 0.4, 0.5, 0.1, 0.3, 0.5]], dtype=torch.float32),
    )
    assert torch.allclose(
        sample["normal"].cpu(),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
    )


def test_normalize_local_grid_coords_rebases_min_corner_to_zero() -> None:
    grid = np.array([[345, 42, 61], [438, 131, 145], [356, 90, 64]], dtype=np.int32)

    normalized = _normalize_local_grid_coords(grid)

    assert normalized.dtype == np.int32
    assert normalized.shape == grid.shape
    assert np.array_equal(normalized.min(axis=0), np.zeros((3,), dtype=np.int32))
    assert np.array_equal(normalized.max(axis=0), grid.max(axis=0) - grid.min(axis=0))


def test_sonata_build_sample_rebases_inherited_global_grid_offsets() -> None:
    context = PointFrameContext(
        grid_coord=np.array([[345, 42, 61], [438, 131, 145]], dtype=np.int32),
        points_local=np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
        normals_local=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        colors=np.array([[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]], dtype=np.float32),
        local_mask=np.array([True, True]),
        world_to_local=np.eye(4, dtype=np.float32),
        G_t=np.eye(4, dtype=np.float32),
    )
    extractor = SonataPointFeatureExtractor.__new__(SonataPointFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.model = SimpleNamespace(embedding=SimpleNamespace(in_channels=6))
    extractor.config = SonataPointConfig()

    sample = extractor._build_sample(context)  # noqa: SLF001

    assert torch.equal(
        sample["grid_coord"].cpu(),
        torch.tensor([[0, 0, 0], [93, 89, 84]], dtype=torch.int32),
    )


def test_sonata_build_sample_rejects_non_xyzrgb_in_channels() -> None:
    context = PointFrameContext(
        grid_coord=np.array([[0, 0, 0]], dtype=np.int32),
        points_local=np.array([[0.0, 0.1, 0.2]], dtype=np.float32),
        normals_local=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        colors=np.array([[0.2, 0.4, 0.6]], dtype=np.float32),
        local_mask=np.array([True]),
        world_to_local=np.eye(4, dtype=np.float32),
        G_t=np.eye(4, dtype=np.float32),
    )
    extractor = SonataPointFeatureExtractor.__new__(SonataPointFeatureExtractor)
    extractor.device = torch.device("cpu")
    extractor.model = SimpleNamespace(embedding=SimpleNamespace(in_channels=9))
    extractor.config = SonataPointConfig()

    with pytest.raises(RuntimeError, match="in_channels == 6|xyz\\+rgb|6 channels"):
        extractor._build_sample(context)  # noqa: SLF001


def test_sonata_wrapper_encodes_local_points_without_checkpoint() -> None:
    if not sonata_runtime_available():
        pytest.skip("Sonata runtime dependencies are unavailable in this environment.")
    if not torch.cuda.is_available():
        pytest.skip("Sonata wrapper is GPU-only.")
    context = PointFrameContext(
        grid_coord=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        ),
        points_local=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
            dtype=np.float32,
        ),
        normals_local=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)),
        colors=np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (4, 1)),
        local_mask=np.ones((4,), dtype=bool),
        world_to_local=np.eye(4, dtype=np.float32),
        G_t=np.eye(4, dtype=np.float32),
    )
    extractor = SonataPointFeatureExtractor(
        SonataPointConfig(
            checkpoint_path=None,
            device="cuda",
            dtype="float32",
            allow_random_init=True,
        )
    )
    features = extractor.encode_local_context(context)
    assert features.features.shape[0] == 4
    assert features.features.shape[1] == extractor.feature_dim
    feat_np = features.features.detach().cpu().numpy() if isinstance(features.features, torch.Tensor) else np.asarray(features.features)
    assert np.all(np.isfinite(feat_np))
