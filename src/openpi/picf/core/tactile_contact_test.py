from __future__ import annotations

import pytest
import torch

from openpi.picf.core.tactile_contact import contact_prob_with_hysteresis
from openpi.picf.core.tactile_contact import summarize_contact_context


def test_contact_prob_hysteresis_turns_on_fast_and_releases_slow() -> None:
    scores = torch.tensor([2.0], dtype=torch.float32)
    ema, prob, active = contact_prob_with_hysteresis(
        scores,
        tau_on=1.5,
        tau_off=1.0,
        temperature=0.1,
        ema_beta=0.8,
        previous_score_ema=torch.tensor([0.0], dtype=torch.float32),
        previous_active=torch.tensor([False]),
    )
    assert float(ema.item()) == pytest.approx(0.4, abs=1e-6)
    assert bool(active.item()) is True
    assert float(prob.item()) > 0.9

    ema2, prob2, active2 = contact_prob_with_hysteresis(
        torch.tensor([1.1], dtype=torch.float32),
        tau_on=1.5,
        tau_off=1.0,
        temperature=0.1,
        ema_beta=0.8,
        previous_score_ema=ema,
        previous_active=active,
    )
    assert bool(active2.item()) is True
    assert float(prob2.item()) < 0.5

    _, prob3, active3 = contact_prob_with_hysteresis(
        torch.tensor([0.2], dtype=torch.float32),
        tau_on=1.5,
        tau_off=1.0,
        temperature=0.1,
        ema_beta=0.8,
        previous_score_ema=ema2,
        previous_active=active2,
    )
    assert bool(active3.item()) is False
    assert float(prob3.item()) < 0.5


def test_summarize_contact_context_preserves_two_sensor_laterality() -> None:
    contact_prob = torch.tensor([0.2, 0.9], dtype=torch.float32)
    anchor_mask = torch.tensor([False, True])
    context = summarize_contact_context(contact_prob, anchor_mask)
    assert torch.allclose(context, torch.tensor([0.2, 0.9, 0.9, 0.55], dtype=torch.float32))


def test_summarize_contact_context_single_sensor_duplicates_probability() -> None:
    contact_prob = torch.tensor([0.4], dtype=torch.float32)
    anchor_mask = torch.tensor([False])
    context = summarize_contact_context(contact_prob, anchor_mask)
    assert torch.allclose(context, torch.tensor([0.4, 0.4, 0.4, 0.4], dtype=torch.float32))
