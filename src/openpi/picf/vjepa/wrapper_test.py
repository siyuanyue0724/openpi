import numpy as np
import pytest
import torch
from torch import nn

from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa import wrapper as vjepa_wrapper
from openpi.picf.vjepa.wrapper import Vjepa2VisualEncoder
from openpi.picf.vjepa.wrapper import _extract_encoder_state_dict
from openpi.picf.vjepa.wrapper import _resolve_checkpoint_key
from openpi.picf.vjepa.wrapper import _vjepa_uses_autocast
from openpi.picf.vjepa.wrapper import vjepa_runtime_available


def test_vjepa_wrapper_reshapes_dense_map() -> None:
    if not vjepa_runtime_available():
        pytest.skip("V-JEPA runtime dependencies are unavailable in this environment.")
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="float32",
    )
    encoder = Vjepa2VisualEncoder(config)
    clip = np.random.default_rng(0).integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)

    output = encoder.encode_clip(clip)

    assert output.tokens_thwc.shape == (2, 4, 4, encoder.encoder.embed_dim)
    assert output.current_map().shape == (4, 4, encoder.encoder.embed_dim)
    assert output.source_hw == (32, 32)


def test_vjepa_base_defaults_to_ema_encoder() -> None:
    payload = {
        "encoder": {"module.backbone.weight": torch.tensor([1.0])},
        "ema_encoder": {"module.backbone.weight": torch.tensor([2.0])},
    }
    config = VjepaVisualConfig(model_name="vjepa2_1_vit_base_384")

    checkpoint_key = _resolve_checkpoint_key(config, payload)
    state_dict = _extract_encoder_state_dict(payload, checkpoint_key)

    assert checkpoint_key == "ema_encoder"
    assert torch.equal(state_dict["weight"], torch.tensor([2.0]))


def test_vjepa_feature_map_recent_maps_preserve_time_without_mean() -> None:
    tokens = torch.arange(3 * 2 * 2 * 1, dtype=torch.float32).reshape(3, 2, 2, 1)
    fmap = vjepa_wrapper.VjepaFeatureMap(
        tokens_thwc=tokens,
        source_hw=(8, 8),
        resized_hw=(8, 8),
        checkpoint_loaded=False,
        model_name="stub",
    )

    recent = fmap.recent_maps(n=2)

    torch.testing.assert_close(recent, tokens[-2:])
    torch.testing.assert_close(fmap.current_map(use_last_two_mean=True), tokens[-2:].mean(dim=0))


def test_rotate_queries_or_keys_preserves_input_dtype() -> None:
    pytest.importorskip("timm")
    from openpi.picf.vjepa.vendor.modules import rotate_queries_or_keys

    x = torch.randn(1, 2, 6, 8, dtype=torch.bfloat16)
    pos = torch.randn(1, 1, 4, dtype=torch.float32)

    output = rotate_queries_or_keys(x, pos=pos, n_registers=1, has_cls_first=True)

    assert output.dtype == x.dtype


def test_vjepa_uses_autocast_on_cuda_mixed_precision_for_frozen_and_trainable_paths() -> None:
    assert _vjepa_uses_autocast(device=torch.device("cuda"), dtype=torch.bfloat16)
    assert _vjepa_uses_autocast(device=torch.device("cuda"), dtype=torch.float16)
    assert not _vjepa_uses_autocast(device=torch.device("cuda"), dtype=torch.float32)
    assert not _vjepa_uses_autocast(device=torch.device("cpu"), dtype=torch.bfloat16)


def test_trainable_vjepa_eval_still_uses_autocast(monkeypatch) -> None:
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="bfloat16",
        trainable=True,
    )
    clip = np.random.default_rng(0).integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    autocast_calls: list[tuple[str, torch.dtype]] = []
    encoder_calls: list[bool] = []

    class _DummyAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyEncoder(nn.Module):
        embed_dim = 8

        def forward(self, video: torch.Tensor, training: bool = False) -> torch.Tensor:
            encoder_calls.append(bool(training))
            token_count = config.temporal_tokens * (config.spatial_tokens ** 2)
            return torch.zeros((1, token_count, self.embed_dim), dtype=torch.float32, device=video.device)

    monkeypatch.setattr(vjepa_wrapper, "preprocess_video_clip", lambda clip, config: torch.zeros((1, 4, 3, 64, 64)))
    monkeypatch.setattr(vjepa_wrapper, "_vjepa_uses_autocast", lambda **_: True)
    monkeypatch.setattr(
        vjepa_wrapper.torch,
        "autocast",
        lambda device_type, dtype: autocast_calls.append((device_type, dtype)) or _DummyAutocast(),
    )

    encoder = Vjepa2VisualEncoder.__new__(Vjepa2VisualEncoder)
    nn.Module.__init__(encoder)
    encoder.config = config
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.bfloat16
    encoder.trainable = True
    encoder.encoder = _DummyEncoder()
    encoder.checkpoint_loaded = False
    encoder.eval()

    output = encoder.encode_clip(clip)

    assert output.tokens_thwc.shape == (2, 4, 4, 8)
    assert encoder_calls == [False]
    assert autocast_calls == [("cuda", torch.bfloat16)]


def test_frozen_vjepa_eval_still_uses_autocast_for_cuda_mixed_precision(monkeypatch) -> None:
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="bfloat16",
        trainable=False,
    )
    clip = np.random.default_rng(0).integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    autocast_calls: list[tuple[str, torch.dtype]] = []
    encoder_calls: list[bool] = []

    class _DummyAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyEncoder(nn.Module):
        embed_dim = 8

        def forward(self, video: torch.Tensor, training: bool = False) -> torch.Tensor:
            encoder_calls.append(bool(training))
            token_count = config.temporal_tokens * (config.spatial_tokens ** 2)
            return torch.zeros((1, token_count, self.embed_dim), dtype=torch.float32, device=video.device)

    monkeypatch.setattr(vjepa_wrapper, "preprocess_video_clip", lambda clip, config: torch.zeros((1, 4, 3, 64, 64)))
    monkeypatch.setattr(vjepa_wrapper, "_vjepa_uses_autocast", lambda **_: True)
    monkeypatch.setattr(
        vjepa_wrapper.torch,
        "autocast",
        lambda device_type, dtype: autocast_calls.append((device_type, dtype)) or _DummyAutocast(),
    )

    encoder = Vjepa2VisualEncoder.__new__(Vjepa2VisualEncoder)
    nn.Module.__init__(encoder)
    encoder.config = config
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.bfloat16
    encoder.trainable = False
    encoder.encoder = _DummyEncoder()
    encoder.checkpoint_loaded = False
    encoder.eval()

    output = encoder.encode_clip(clip)

    assert output.tokens_thwc.shape == (2, 4, 4, 8)
    assert encoder_calls == [False]
    assert autocast_calls == [("cuda", torch.bfloat16)]


def test_trainable_vjepa_eval_uses_hierarchical_contract(monkeypatch) -> None:
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="float32",
        trainable=True,
    )
    clip = np.random.default_rng(0).integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    encoder_calls: list[tuple[bool, bool]] = []

    class _DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.return_hierarchical = False
            self.embed_dim = 8

        def forward(self, video: torch.Tensor, training: bool = False) -> torch.Tensor:
            encoder_calls.append((bool(training), bool(self.return_hierarchical)))
            last_dim = 32 if (training or self.return_hierarchical) else 8
            token_count = config.temporal_tokens * (config.spatial_tokens ** 2)
            return torch.zeros((1, token_count, last_dim), dtype=torch.float32, device=video.device)

    monkeypatch.setattr(vjepa_wrapper, "preprocess_video_clip", lambda clip, config: torch.zeros((1, 4, 3, 64, 64)))

    encoder = Vjepa2VisualEncoder.__new__(Vjepa2VisualEncoder)
    nn.Module.__init__(encoder)
    encoder.config = config
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.trainable = True
    encoder.encoder = _DummyEncoder()
    encoder.checkpoint_loaded = False
    encoder.eval()

    output = encoder.encode_clip(clip)

    assert output.tokens_thwc.shape == (2, 4, 4, 32)
    assert encoder_calls == [(False, True)]
    assert encoder.encoder.return_hierarchical is False


def test_frozen_vjepa_eval_keeps_last_layer_contract(monkeypatch) -> None:
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="float32",
        trainable=False,
    )
    clip = np.random.default_rng(0).integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    encoder_calls: list[tuple[bool, bool]] = []

    class _DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.return_hierarchical = False
            self.embed_dim = 8

        def forward(self, video: torch.Tensor, training: bool = False) -> torch.Tensor:
            encoder_calls.append((bool(training), bool(self.return_hierarchical)))
            last_dim = 32 if (training or self.return_hierarchical) else 8
            token_count = config.temporal_tokens * (config.spatial_tokens ** 2)
            return torch.zeros((1, token_count, last_dim), dtype=torch.float32, device=video.device)

    monkeypatch.setattr(vjepa_wrapper, "preprocess_video_clip", lambda clip, config: torch.zeros((1, 4, 3, 64, 64)))

    encoder = Vjepa2VisualEncoder.__new__(Vjepa2VisualEncoder)
    nn.Module.__init__(encoder)
    encoder.config = config
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.trainable = False
    encoder.encoder = _DummyEncoder()
    encoder.checkpoint_loaded = False
    encoder.eval()

    output = encoder.encode_clip(clip)

    assert output.tokens_thwc.shape == (2, 4, 4, 8)
    assert encoder_calls == [(False, False)]
    assert encoder.encoder.return_hierarchical is False


def _dummy_cached_encoder(config: VjepaVisualConfig, cache_root, cache_mode: str = "read_or_encode") -> Vjepa2VisualEncoder:
    class _DummyEncoder(nn.Module):
        embed_dim = 8

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, video: torch.Tensor, training: bool = False) -> torch.Tensor:
            del training
            self.calls += 1
            token_count = config.temporal_tokens * (config.spatial_tokens ** 2)
            return torch.arange(token_count * self.embed_dim, dtype=torch.float32, device=video.device).reshape(
                1,
                token_count,
                self.embed_dim,
            )

    encoder = Vjepa2VisualEncoder.__new__(Vjepa2VisualEncoder)
    nn.Module.__init__(encoder)
    encoder.config = config
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.trainable = False
    encoder.encoder = _DummyEncoder()
    encoder.checkpoint_loaded = False
    encoder.eval()
    encoder._cache_mode = cache_mode
    encoder._cache_root = cache_root
    encoder._cache_temporal_slices = max(int(config.feature_cache_temporal_slices), 1)
    encoder._cache_storage_dtype_name = config.feature_cache_storage_dtype
    encoder._cache_contract = {
        "version": 1,
        "model_name": config.model_name,
        "arch_name": config.arch_name_override,
        "checkpoint_path": None,
        "checkpoint_key": None,
        "checkpoint_hash": None,
        "checkpoint_loaded": False,
        "img_size": config.img_size,
        "num_frames": config.num_frames,
        "patch_size": config.patch_size,
        "tubelet_size": config.tubelet_size,
        "temporal_tokens": config.temporal_tokens,
        "spatial_tokens": config.spatial_tokens,
        "feature_mode": config.feature_mode,
        "cache_temporal_slices": encoder._cache_temporal_slices,
        "cache_storage_dtype": encoder._cache_storage_dtype_name,
        "normalize_mean": [float(v) for v in config.normalize_mean],
        "normalize_std": [float(v) for v in config.normalize_std],
    }
    encoder._cache_contract_hash = vjepa_wrapper._sha256_json(encoder._cache_contract)
    return encoder


def test_vjepa_feature_cache_read_or_encode_reuses_cached_features(monkeypatch, tmp_path) -> None:
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="float32",
        trainable=False,
        feature_cache_root=str(tmp_path),
        feature_cache_mode="read_or_encode",
    )
    clip = np.random.default_rng(0).integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(vjepa_wrapper, "preprocess_video_clip", lambda clip, config: torch.zeros((1, 4, 3, 64, 64)))

    writer = _dummy_cached_encoder(config, tmp_path, cache_mode="read_or_encode")
    first = writer.encode_clip(clip)
    assert writer.encoder.calls == 1

    reader = _dummy_cached_encoder(config, tmp_path, cache_mode="read")
    second = reader.encode_clip(clip)

    assert reader.encoder.calls == 0
    expected = torch.as_tensor(first.tokens_thwc)[-config.feature_cache_temporal_slices :]
    torch.testing.assert_close(torch.as_tensor(second.tokens_thwc), expected)


def test_vjepa_feature_cache_persists_bounded_suffix_in_low_precision(monkeypatch, tmp_path) -> None:
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=8,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="float32",
        trainable=False,
        feature_cache_root=str(tmp_path),
        feature_cache_mode="read_or_encode",
        feature_cache_temporal_slices=2,
        feature_cache_storage_dtype="bfloat16",
    )
    clip = np.random.default_rng(2).integers(0, 255, size=(8, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(vjepa_wrapper, "preprocess_video_clip", lambda clip, config: torch.zeros((1, 8, 3, 64, 64)))

    writer = _dummy_cached_encoder(config, tmp_path, cache_mode="read_or_encode")
    first = writer.encode_clip(clip)
    cache_files = list(tmp_path.rglob("*.pt"))
    assert len(cache_files) == 1
    payload = torch.load(cache_files[0], map_location="cpu", weights_only=False)
    cached = payload["tokens_thwc"]
    assert tuple(cached.shape[:3]) == (2, config.spatial_tokens, config.spatial_tokens)
    assert cached.dtype == torch.bfloat16
    expected = torch.as_tensor(first.tokens_thwc)[-2:].to(dtype=torch.bfloat16)
    torch.testing.assert_close(cached, expected)


def test_vjepa_feature_cache_read_mode_fails_closed_on_missing_entry(tmp_path) -> None:
    config = VjepaVisualConfig(
        arch_name_override="vit_tiny",
        img_size=64,
        num_frames=4,
        patch_size=16,
        tubelet_size=2,
        device="cpu",
        dtype="float32",
        trainable=False,
    )
    clip = np.random.default_rng(1).integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    encoder = _dummy_cached_encoder(config, tmp_path, cache_mode="read")

    with pytest.raises(RuntimeError, match="Missing V-JEPA feature cache entry"):
        encoder.encode_clip(clip)
