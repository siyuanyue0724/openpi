from __future__ import annotations

import torch
import pytest

from picf_next.lingbot_wla_randomness import paired_wla_inference_seed, paired_wla_seed


def test_paired_wla_seed_is_stable_and_uses_both_frozen_draws() -> None:
    inputs = {
        "noise": torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
        "time": torch.tensor([0.25], dtype=torch.float32),
    }
    first = paired_wla_seed(inputs)
    second = paired_wla_seed({name: value.clone() for name, value in inputs.items()})
    assert first == second
    assert 0 <= first < 2**63

    changed_noise = dict(inputs)
    changed_noise["noise"] = inputs["noise"].clone()
    changed_noise["noise"][0, 0, 0] += 1
    changed_time = dict(inputs)
    changed_time["time"] = torch.tensor([0.5])
    assert paired_wla_seed(changed_noise) != first
    assert paired_wla_seed(changed_time) != first


def test_paired_wla_inference_seed_is_stable_domain_separated_and_noise_sensitive() -> None:
    noise = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    first = paired_wla_inference_seed(noise)
    assert first == paired_wla_inference_seed(noise.clone())
    assert 0 <= first < 2**63
    changed = noise.clone()
    changed[0, 0, 0] += 1
    assert paired_wla_inference_seed(changed) != first
    assert paired_wla_seed({"noise": noise, "time": torch.tensor([0.25])}) != first


def test_paired_wla_inference_seed_rejects_invalid_noise() -> None:
    with pytest.raises(TypeError):
        paired_wla_inference_seed(torch.ones(1, 2, 3, dtype=torch.long))
    invalid = torch.ones(1, 2, 3)
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        paired_wla_inference_seed(invalid)
