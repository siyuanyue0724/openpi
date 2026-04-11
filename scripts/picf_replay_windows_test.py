from __future__ import annotations

import json
import random
import numpy as np
import pytest
import torch

from scripts.picf_replay_windows import _advance_rng_draws
from scripts.picf_replay_windows import _coerce_optional_bool
from scripts.picf_replay_windows import _generate_rng_flat_indices
from scripts.picf_replay_windows import _load_replay_rng_state
from scripts.picf_replay_windows import _load_replay_state
from scripts.picf_replay_windows import _prune_old_replay_checkpoints
from scripts.picf_replay_windows import _resolve_rank_seed
from scripts.picf_replay_windows import _save_replay_rng_state
from scripts.picf_replay_windows import _save_replay_state


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


def test_resolve_rank_seed_prefers_explicit_rank_seed() -> None:
    assert _resolve_rank_seed(rank_seed=3, rng_rank=0) == 3


def test_resolve_rank_seed_falls_back_to_rng_rank() -> None:
    assert _resolve_rank_seed(rank_seed=None, rng_rank=0) == 0
    assert _resolve_rank_seed(rank_seed=None, rng_rank=1) == 1


def test_resolve_rank_seed_defaults_to_one_without_rng_rank() -> None:
    assert _resolve_rank_seed(rank_seed=None, rng_rank=None) == 1


def test_coerce_optional_bool_handles_none_and_boolean_strings() -> None:
    assert _coerce_optional_bool(None) is None
    assert _coerce_optional_bool("true") is True
    assert _coerce_optional_bool("False") is False


def test_coerce_optional_bool_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        _coerce_optional_bool("maybe")


def test_advance_rng_draws_matches_manual_consumption() -> None:
    rng = np.random.default_rng(123)
    _advance_rng_draws(rng=rng, dataset_size=19, draw_count=5)
    got = int(rng.integers(0, 19))

    baseline = np.random.default_rng(123)
    for _ in range(5):
        baseline.integers(0, 19)
    expected = int(baseline.integers(0, 19))
    assert got == expected


def test_save_and_load_replay_state_round_trip(tmp_path) -> None:
    checkpoint_dir = tmp_path / "100"
    checkpoint_dir.mkdir(parents=True)
    retryable_skips = [
        {
            "flat_index": 455113,
            "raw_draw": 23,
            "replay_step": 23,
            "retryable_skip_count": 1,
            "status": "retryable_window_skipped",
        }
    ]
    _save_replay_state(
        checkpoint_dir=checkpoint_dir,
        accepted_counter=100,
        raw_draw_counter=108,
        retryable_skip_count=8,
        accepted_flat_indices=[1, 2, 3],
        retryable_skips=retryable_skips,
    )

    payload = json.loads((checkpoint_dir / "replay_state.json").read_text(encoding="utf-8"))
    assert payload["accepted_counter"] == 100
    assert payload["raw_draw_counter"] == 108

    loaded = _load_replay_state(checkpoint_dir)
    assert loaded == {
        "accepted_counter": 100,
        "raw_draw_counter": 108,
        "retryable_skip_count": 8,
        "accepted_flat_indices": [1, 2, 3],
        "retryable_skips": retryable_skips,
    }


def test_save_and_load_replay_rng_state_round_trip(tmp_path) -> None:
    checkpoint_dir = tmp_path / "100"
    checkpoint_dir.mkdir(parents=True)
    device = torch.device("cpu")

    random.seed(7)
    np.random.seed(11)
    torch.manual_seed(13)
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(()).item())

    random.seed(7)
    np.random.seed(11)
    torch.manual_seed(13)
    _save_replay_rng_state(checkpoint_dir=checkpoint_dir, device=device)

    random.seed(100)
    np.random.seed(101)
    torch.manual_seed(102)

    assert _load_replay_rng_state(checkpoint_dir, device=device) is True
    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.rand()) == pytest.approx(expected_numpy)
    assert float(torch.rand(()).item()) == pytest.approx(expected_torch)


def test_load_replay_rng_state_returns_false_when_missing(tmp_path) -> None:
    assert _load_replay_rng_state(tmp_path, device=torch.device("cpu")) is False


def test_prune_old_replay_checkpoints_keeps_latest_numeric_steps(tmp_path) -> None:
    for name in ("20", "40", "60", "80", "100", "latest.pt", "tmp_120"):
        path = tmp_path / name
        if name.isdigit() or name.startswith("tmp_"):
            path.mkdir(parents=True)
        else:
            path.write_text("stub", encoding="utf-8")
    _prune_old_replay_checkpoints(checkpoint_root=tmp_path, keep=3)
    remaining = sorted(path.name for path in tmp_path.iterdir())
    assert "20" not in remaining
    assert "40" not in remaining
    assert "60" in remaining
    assert "80" in remaining
    assert "100" in remaining
    assert "latest.pt" in remaining
    assert "tmp_120" in remaining
