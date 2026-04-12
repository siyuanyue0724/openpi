from __future__ import annotations

import torch


def summarize_contact_context(contact_prob: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
    if contact_prob.numel() == 0:
        return torch.zeros((4,), device=contact_prob.device, dtype=contact_prob.dtype)
    _ = anchor_mask
    if contact_prob.numel() == 1:
        left_prob = contact_prob[0]
        right_prob = contact_prob[0]
    else:
        # Preserve per-sensor laterality for the common two-finger case used by
        # CALVIN, while keeping the token width fixed for downstream projections.
        left_prob = contact_prob[0]
        right_prob = contact_prob[1]
    return torch.stack(
        [
            left_prob,
            right_prob,
            torch.max(contact_prob),
            torch.mean(contact_prob),
        ],
        dim=0,
    )


def contact_prob_with_hysteresis(
    scores: torch.Tensor,
    *,
    tau_on: float,
    tau_off: float,
    temperature: float,
    ema_beta: float,
    previous_score_ema: torch.Tensor | None = None,
    previous_active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.as_tensor(scores)
    if scores.numel() == 0:
        empty_bool = torch.zeros((0,), device=scores.device, dtype=torch.bool)
        return scores, scores, empty_bool
    if previous_score_ema is not None and previous_score_ema.shape == scores.shape:
        ema = (float(ema_beta) * previous_score_ema.to(device=scores.device, dtype=scores.dtype)) + ((1.0 - float(ema_beta)) * scores)
    else:
        ema = scores
    # Use instantaneous evidence for fast turn-on, while preserving EMA as a slow
    # release path so brief score drops do not chatter the gate.
    evidence = torch.maximum(scores, ema)
    prev_active = (
        previous_active.to(device=scores.device, dtype=torch.bool)
        if previous_active is not None and previous_active.shape == scores.shape
        else torch.zeros_like(scores, dtype=torch.bool)
    )
    tau_on_t = torch.full_like(scores, float(tau_on))
    tau_off_t = torch.full_like(scores, float(tau_off))
    active = torch.where(prev_active, evidence > tau_off_t, evidence > tau_on_t)
    mid = 0.5 * (tau_on_t + tau_off_t)
    temp = torch.clamp(0.5 * (tau_on_t - tau_off_t), min=float(temperature))
    prob = torch.sigmoid((evidence - mid) / temp)
    return ema, prob, active
