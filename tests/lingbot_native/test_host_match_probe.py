from __future__ import annotations

import pytest
import torch

from tools.probe_lingbot_host_native_match import (
    HOST_NATIVE_MATCH_OVERFIT_SCHEMA,
    posterior_prompt_effect_max_abs,
    run_host_native_match_overfit_probe,
)


def test_posterior_prompt_effect_holds_batch_slot_fixed() -> None:
    factual = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    swapped = factual.clone()

    assert posterior_prompt_effect_max_abs(factual, swapped) == 0.0

    swapped[1, 0, 0] += 0.25
    assert posterior_prompt_effect_max_abs(factual, swapped) == pytest.approx(0.25)


def test_host_native_match_fixed_batch_probe_reverses_winners_without_task_state_leak() -> None:
    report = run_host_native_match_overfit_probe(optimizer_updates=160)

    assert report.initial_loss > report.final_loss * 10
    assert report.factual_winners == (0, 1)
    assert report.swapped_prompt_winners == (1, 0)
    assert report.posterior_prompt_max_abs == 0
    assert len(report.layer_gradient_norms) == 3
    assert all(value > 0 for value in report.layer_gradient_norms)
    assert report.as_dict()["schema"] == HOST_NATIVE_MATCH_OVERFIT_SCHEMA
    assert report.as_dict()["status"] == "PASS"
