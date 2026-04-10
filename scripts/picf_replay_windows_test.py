from __future__ import annotations

import numpy as np

from scripts.picf_replay_windows import _generate_rng_flat_indices


def test_generate_rng_flat_indices_matches_numpy_default_rng() -> None:
    got = _generate_rng_flat_indices(
        total_windows=8,
        dataset_size=17,
        seed=3,
        rank=2,
        skip_windows=0,
    )
    rng = np.random.default_rng(3 + 17 * 2)
    expected = [int(rng.integers(0, 17)) for _ in range(8)]
    assert got == expected


def test_generate_rng_flat_indices_honors_skip_windows() -> None:
    got = _generate_rng_flat_indices(
        total_windows=4,
        dataset_size=29,
        seed=11,
        rank=1,
        skip_windows=3,
    )
    rng = np.random.default_rng(11 + 17 * 1)
    draws = [int(rng.integers(0, 29)) for _ in range(7)]
    assert got == draws[3:]
