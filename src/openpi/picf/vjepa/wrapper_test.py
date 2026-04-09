import numpy as np
import pytest
import torch

from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.wrapper import Vjepa2VisualEncoder
from openpi.picf.vjepa.wrapper import _extract_encoder_state_dict
from openpi.picf.vjepa.wrapper import _resolve_checkpoint_key
from openpi.picf.vjepa.wrapper import _trainable_vjepa_uses_autocast
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


def test_rotate_queries_or_keys_preserves_input_dtype() -> None:
    pytest.importorskip("timm")
    from openpi.picf.vjepa.vendor.modules import rotate_queries_or_keys

    x = torch.randn(1, 2, 6, 8, dtype=torch.bfloat16)
    pos = torch.randn(1, 1, 4, dtype=torch.float32)

    output = rotate_queries_or_keys(x, pos=pos, n_registers=1, has_cls_first=True)

    assert output.dtype == x.dtype


def test_trainable_vjepa_uses_autocast_only_on_cuda_mixed_precision() -> None:
    assert _trainable_vjepa_uses_autocast(
        trainable=True,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    assert _trainable_vjepa_uses_autocast(
        trainable=True,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    assert not _trainable_vjepa_uses_autocast(
        trainable=False,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    assert not _trainable_vjepa_uses_autocast(
        trainable=True,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    assert not _trainable_vjepa_uses_autocast(
        trainable=True,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
