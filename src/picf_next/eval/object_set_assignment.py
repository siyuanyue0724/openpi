"""Per-object diagnostics for a matched object-set prediction."""

from __future__ import annotations

import math
from typing import Any

from picf_next.models.discovery import ObjectDiscoveryOutput
from picf_next.models.set_loss import ObjectSetTarget, SetMatch


def object_set_assignment_diagnostics(
    output: ObjectDiscoveryOutput,
    target: ObjectSetTarget,
    match: SetMatch,
    *,
    batch_index: int,
    active_threshold: float = 0.5,
) -> dict[str, Any]:
    """Return JSON-safe matched-object and unmatched-query diagnostics.

    This function runs strictly after set matching. It is an audit surface and
    cannot affect discovery, matching, the loss, or checkpoint selection.
    """

    batch_size, query_count = output.existence_logits.shape
    if not 0 <= batch_index < batch_size:
        raise IndexError("object-set diagnostic batch index is out of range")
    if not math.isfinite(active_threshold) or not 0.0 < active_threshold < 1.0:
        raise ValueError("object-set diagnostic threshold must be inside (0, 1)")

    prediction_indices = [int(value) for value in match.prediction_indices.tolist()]
    target_indices = [int(value) for value in match.target_indices.tolist()]
    if len(prediction_indices) != len(target_indices):
        raise ValueError("object-set diagnostic match has unequal index lengths")
    if len(set(prediction_indices)) != len(prediction_indices) or len(set(target_indices)) != len(
        target_indices
    ):
        raise ValueError("object-set diagnostic match is not one-to-one")
    if any(not 0 <= value < query_count for value in prediction_indices):
        raise IndexError("object-set diagnostic prediction index is out of range")
    if any(not 0 <= value < target.num_objects for value in target_indices):
        raise IndexError("object-set diagnostic target index is out of range")
    if len(target_indices) != target.num_objects:
        raise ValueError("object-set diagnostic requires every target object to be matched")

    identities = target.temporal_identity_keys
    if identities is not None and len(identities) != target.num_objects:
        raise ValueError("object-set diagnostic identities do not align with target objects")

    existence = output.existence[batch_index].detach().float()
    localization_confidence = output.localization_confidence[batch_index].detach().float()
    measurement_probability = output.measurement_probability[batch_index].detach().float()
    training_existence = output.training_existence_score[batch_index].detach().float()
    supervised = target.supervision_valid
    ownership = output.ownership[batch_index, supervised].detach().float()
    expected = target.ownership[supervised, :-1].detach().float()
    winner = ownership.argmax(dim=-1) if ownership.shape[0] else None

    query_by_target = dict(zip(target_indices, prediction_indices, strict=True))
    objects: list[dict[str, Any]] = []
    for target_index in range(target.num_objects):
        query = query_by_target[target_index]
        prediction = ownership[:, query]
        truth = expected[:, target_index]
        intersection = (prediction * truth).sum()
        prediction_mass = prediction.sum()
        target_mass = truth.sum()
        union = prediction_mass + target_mass - intersection
        soft_dice = 2.0 * intersection / (prediction_mass + target_mass).clamp_min(1e-8)
        soft_iou = intersection / union.clamp_min(1e-8)
        dominant_fraction = (
            float((winner == query).float().mean().item()) if winner is not None else 0.0
        )
        probability = float(existence[query].item())
        objects.append(
            {
                "target_index": target_index,
                "identity_key": (
                    f"target/{target_index}" if identities is None else identities[target_index]
                ),
                "query": query,
                "active": probability > active_threshold,
                "existence": probability,
                "training_existence_score": float(training_existence[query].item()),
                "localization_confidence": float(localization_confidence[query].item()),
                "measurement_probability": float(measurement_probability[query].item()),
                "soft_dice": float(soft_dice.item()),
                "soft_iou": float(soft_iou.item()),
                "target_ownership_mass": float(target_mass.item()),
                "mean_target_ownership": (float(truth.mean().item()) if truth.numel() else 0.0),
                "mean_ownership_mass": (
                    float(prediction.mean().item()) if prediction.numel() else 0.0
                ),
                "dominant_token_fraction": dominant_fraction,
            }
        )

    matched_queries = set(prediction_indices)
    unmatched_queries = []
    for query in range(query_count):
        if query in matched_queries:
            continue
        probability = float(existence[query].item())
        unmatched_queries.append(
            {
                "query": query,
                "active": probability > active_threshold,
                "existence": probability,
                "training_existence_score": float(training_existence[query].item()),
                "localization_confidence": float(localization_confidence[query].item()),
                "measurement_probability": float(measurement_probability[query].item()),
                "mean_ownership_mass": (
                    float(ownership[:, query].mean().item()) if ownership.shape[0] else 0.0
                ),
                "dominant_token_fraction": (
                    float((winner == query).float().mean().item()) if winner is not None else 0.0
                ),
            }
        )
    unmatched_queries.sort(key=lambda row: (-float(row["existence"]), int(row["query"])))

    return {
        "supervised_token_count": int(supervised.sum().item()),
        "objects": objects,
        "unmatched_queries": unmatched_queries,
    }
