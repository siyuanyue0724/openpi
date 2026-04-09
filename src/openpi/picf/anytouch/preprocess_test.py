import numpy as np

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.anytouch.preprocess import preprocess_tactile_clip


def test_preprocess_tactile_clip_matches_anytouch_contract() -> None:
    clip = np.full((4, 24, 24, 3), 200, dtype=np.uint8)
    background = np.full((24, 24, 3), 100, dtype=np.uint8)

    out = preprocess_tactile_clip(clip, background, AnyTouchConfig())

    assert out.shape == (4, 3, 224, 224)
    assert np.isfinite(out.numpy()).all()
    assert float(out.mean().item()) > 0.0
