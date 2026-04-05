from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import linear_sum_assignment

from openpi.picf.geometry import normalize_vectors


@dataclasses.dataclass(frozen=True)
class MatchResult:
    pred_idx: np.ndarray
    matched_mask: np.ndarray
    match_ratio: float
    reindex_failure_rate: float
    normal_flip_ratio: float


def match_supports(
    current_x: np.ndarray,
    current_n: np.ndarray,
    current_active: np.ndarray,
    current_eid: np.ndarray,
    prev_x_transport: np.ndarray,
    prev_n_transport: np.ndarray,
    prev_slots: np.ndarray,
    prev_eid: np.ndarray,
    *,
    tau_p: float,
    tau_n: float,
    rgb_enabled: bool,
    lambda_app_match: float,
    epsilon_app: float,
) -> MatchResult:
    pred_idx = np.full(current_active.shape, -1, dtype=np.int32)
    matched_mask = np.zeros(current_active.shape, dtype=bool)

    curr_slots = np.flatnonzero(current_active)
    prev_active_slots = np.asarray(prev_slots, dtype=np.int32)
    if curr_slots.size == 0 or prev_active_slots.size == 0:
        return MatchResult(
            pred_idx=pred_idx,
            matched_mask=matched_mask,
            match_ratio=0.0,
            reindex_failure_rate=1.0 if prev_active_slots.size > 0 else 0.0,
            normal_flip_ratio=0.0,
        )

    curr_x = np.asarray(current_x[curr_slots], dtype=np.float32)
    curr_n = normalize_vectors(current_n[curr_slots])
    prev_x = np.asarray(prev_x_transport, dtype=np.float32)
    prev_n = normalize_vectors(prev_n_transport)
    dists = np.linalg.norm(curr_x[:, None, :] - prev_x[None, :, :], axis=-1)
    dots = np.sum(curr_n[:, None, :] * prev_n[None, :, :], axis=-1)
    admissible = (dists < float(tau_p)) & (dots > float(tau_n))
    cost = np.full_like(dists, np.inf, dtype=np.float32)
    cost[admissible] = dists[admissible]
    if rgb_enabled and current_eid.shape[1] > 0 and prev_eid.shape[1] > 0:
        curr_desc = normalize_vectors(current_eid[curr_slots], eps=epsilon_app)
        prev_desc = normalize_vectors(prev_eid, eps=epsilon_app)
        sim = np.sum(curr_desc[:, None, :] * prev_desc[None, :, :], axis=-1)
        cost[admissible] += float(lambda_app_match) * (1.0 - sim[admissible]) / 2.0
    if not np.isfinite(cost).any():
        return MatchResult(
            pred_idx=pred_idx,
            matched_mask=matched_mask,
            match_ratio=0.0,
            reindex_failure_rate=1.0,
            normal_flip_ratio=0.0,
        )

    stable_tie = np.asarray(prev_active_slots, dtype=np.float32)[None, :] * 1e-9
    work = np.where(np.isfinite(cost), cost + stable_tie, 1e6)
    row_ind, col_ind = linear_sum_assignment(work)
    normal_flips = 0
    matched = 0
    for row, col in zip(row_ind.tolist(), col_ind.tolist(), strict=False):
        if not np.isfinite(cost[row, col]):
            continue
        current_slot = int(curr_slots[row])
        previous_slot = int(prev_active_slots[col])
        pred_idx[current_slot] = previous_slot
        matched_mask[current_slot] = True
        matched += 1
        if float(dots[row, col]) < 0.0:
            normal_flips += 1

    denom = max(int(prev_active_slots.size), 1)
    match_ratio = matched / denom
    return MatchResult(
        pred_idx=pred_idx,
        matched_mask=matched_mask,
        match_ratio=float(match_ratio),
        reindex_failure_rate=float(1.0 - match_ratio),
        normal_flip_ratio=float(normal_flips / max(matched, 1)),
    )
