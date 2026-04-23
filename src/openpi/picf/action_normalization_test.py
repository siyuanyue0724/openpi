import json
from pathlib import Path

import numpy as np
import torch

from openpi.picf.action_normalization import PicfActionNormalizer
from openpi.picf.action_normalization import PicfStateNormalizer


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


def test_from_path_accepts_norm_stats_file_path(tmp_path: Path) -> None:
    stats_dir = tmp_path / "calvin"
    stats_dir.mkdir(parents=True)
    payload = {
        "norm_stats": {
            "state": {
                "mean": [0.0, 1.0],
                "std": [2.0, 4.0],
                "q01": [-1.0, -3.0],
                "q99": [1.0, 5.0],
            },
            "actions": {
                "mean": [0.0, 1.0],
                "std": [2.0, 4.0],
                "q01": [-1.0, -3.0],
                "q99": [1.0, 5.0],
            }
        }
    }
    stats_path = stats_dir / "norm_stats.json"
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    normalizer = PicfActionNormalizer.from_path(stats_path, mode="quantile")
    sample = np.asarray([0.25, 0.0], dtype=np.float32)
    normalized = normalizer.normalize_np(sample)
    restored = normalizer.unnormalize_np(normalized)
    np.testing.assert_allclose(restored, sample, atol=1e-5)


def test_state_normalizer_from_path_uses_state_entry(tmp_path: Path) -> None:
    stats_dir = tmp_path / "calvin"
    stats_dir.mkdir(parents=True)
    payload = {
        "norm_stats": {
            "state": {
                "mean": [0.0, 0.0],
                "std": [1.0, 1.0],
                "q01": [-2.0, -1.0],
                "q99": [2.0, 3.0],
            },
            "actions": {
                "mean": [100.0, 100.0],
                "std": [10.0, 10.0],
                "q01": [0.0, 0.0],
                "q99": [1.0, 1.0],
            },
        }
    }
    stats_path = stats_dir / "norm_stats.json"
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    normalizer = PicfStateNormalizer.from_path(stats_path, mode="quantile")
    sample = np.asarray([0.0, 1.0], dtype=np.float32)
    normalized = normalizer.normalize_np(sample)
    np.testing.assert_allclose(normalized, np.asarray([0.0, 0.0], dtype=np.float32), atol=1e-5)
