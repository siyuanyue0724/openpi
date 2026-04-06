import numpy as np
import torch

from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.picf.vjepa.wrapper import Vjepa2VisualEncoder
from openpi.picf.vjepa.wrapper import _extract_encoder_state_dict
from openpi.picf.vjepa.wrapper import _resolve_checkpoint_key


def test_vjepa_wrapper_reshapes_dense_map() -> None:
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
