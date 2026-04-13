from pathlib import Path

import numpy as np
import torch

from openpi.picf.action_normalization import PicfActionNormalizer


def test_quantile_action_normalizer_roundtrip() -> None:
    normalizer = PicfActionNormalizer(
        mean=np.zeros((7,), dtype=np.float32),
        std=np.ones((7,), dtype=np.float32),
        q01=np.full((7,), -1.0, dtype=np.float32),
        q99=np.full((7,), 1.0, dtype=np.float32),
        mode="quantile",
    )
    action = np.asarray([0.25, -0.5, 0.75, 0.1, -0.2, 0.3, 1.0], dtype=np.float32)
    normalized = normalizer.normalize_np(action)
    restored = normalizer.unnormalize_np(normalized)
    np.testing.assert_allclose(restored, action, atol=1e-5)


def test_zscore_action_normalizer_roundtrip_tensor() -> None:
    normalizer = PicfActionNormalizer(
        mean=np.asarray([1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        std=np.asarray([2.0, 4.0, 0.5, 1.0, 2.0, 4.0, 0.5], dtype=np.float32),
        q01=None,
        q99=None,
        mode="zscore",
    )
    action = torch.tensor([3.0, 2.0, 1.0, -1.0, 2.0, -4.0, 0.5], dtype=torch.float32)
    normalized = normalizer.normalize_tensor(action)
    restored = normalizer.unnormalize_tensor(normalized)
    assert torch.allclose(restored, action, atol=1e-5)
