"""Task-independent metrics for a matched physical entity set."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFrameAssignment,
    PhysicalFramePredictions,
    PhysicalFrameTargets,
    eligible_physical_tracks,
)

ENTITY_SET_EVALUATION_SCHEMA = "picf-next.entity-set-evaluation-sample.v5"
ENTITY_SET_PARTITION_SUMMARY_SCHEMA = "picf-next.entity-set-evaluation-partition.v5"
ENTITY_AREA_STRATA = (
    ("lt_2_percent", 0.0, 0.02),
    ("2_to_5_percent", 0.02, 0.05),
    ("ge_5_percent", 0.05, None),
)


def _finite(value: torch.Tensor) -> float:
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise RuntimeError("entity evaluation produced a non-finite scalar")
    return result


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _area_stratum(area_fraction: float) -> str:
    if not 0 <= area_fraction <= 1:
        raise ValueError("entity evaluation area fraction lies outside [0,1]")
    for name, lower, upper in ENTITY_AREA_STRATA:
        if area_fraction >= lower and (upper is None or area_fraction < upper):
            return name
    raise RuntimeError("entity evaluation area stratum is incomplete")


def maximum_token_grid_soft_iou(target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the exact best soft-IoU representable on one token grid.

    For fixed fractional targets, soft-IoU is a linear-fractional function of
    the prediction cube. Its maximum is attained at a vertex. Sorting target
    occupancy therefore reduces the exact search to binary prefix supports.
    """

    if target.ndim != 1 or weight.shape != target.shape:
        raise ValueError("soft-IoU ceiling requires equal one-dimensional tensors")
    if not target.is_floating_point() or not weight.is_floating_point():
        raise TypeError("soft-IoU ceiling requires floating tensors")
    if target.device != weight.device:
        raise ValueError("soft-IoU ceiling tensors must share one device")
    if not torch.isfinite(target).all() or not torch.isfinite(weight).all():
        raise ValueError("soft-IoU ceiling tensors contain NaN or infinity")
    if (target < 0).any() or (target > 1).any() or (weight < 0).any():
        raise ValueError("soft-IoU ceiling target/weight lies outside its valid range")
    positive = weight > 0
    if not positive.any():
        raise ValueError("soft-IoU ceiling requires positive observed weight")
    target = target[positive].float()
    weight = weight[positive].float()
    target_mass = (target * weight).sum()
    if target_mass <= 0:
        raise ValueError("soft-IoU ceiling requires positive target mass")
    order = torch.argsort(target, descending=True, stable=True)
    ordered_target = target[order]
    ordered_weight = weight[order]
    intersection = torch.cumsum(ordered_weight * ordered_target, dim=0)
    false_positive = torch.cumsum(ordered_weight * (1 - ordered_target), dim=0)
    return (intersection / (target_mass + false_positive)).amax()


@torch.no_grad()
def evaluate_physical_entity_frame(
    predictions: PhysicalFramePredictions,
    targets: PhysicalFrameTargets,
    assignment: PhysicalFrameAssignment,
    *,
    identity_keys: Sequence[str],
) -> dict[str, object]:
    """Evaluate one class-free frame after loss-side Hungarian matching.

    Identity strings are emitted only as audit metadata. They neither choose a
    row nor enter any metric computation.
    """

    if predictions.support_logits.shape[0] != 1 or targets.masks.shape[0] != 1:
        raise ValueError("entity frame evaluation requires batch size one")
    _, tokens, rows = predictions.support_logits.shape
    if assignment.row_to_track.shape != (1, rows):
        raise ValueError("entity evaluation assignment differs from prediction rows")
    if targets.masks.shape[-1] != tokens:
        raise ValueError("entity evaluation target and prediction tokens differ")
    if not targets.exclusive_ownership:
        raise ValueError("entity ownership evaluation requires exclusive physical masks")
    if len(identity_keys) != targets.masks.shape[1]:
        raise ValueError("entity evaluation identities differ from target tracks")
    if len(set(identity_keys)) != len(identity_keys) or any(
        not isinstance(value, str) or not value for value in identity_keys
    ):
        raise ValueError("entity evaluation identities must be unique nonempty strings")

    sensor_weight = (
        predictions.sensor_valid[0].float()
        * targets.token_observed_fraction[0].float()
        * targets.token_measure[0].float()
    )
    denominator = sensor_weight.sum()
    if denominator <= 0:
        raise ValueError("entity evaluation frame has no observed sensor token")
    support_probability = predictions.support_logits[0].float().sigmoid()
    ownership_probability = predictions.ownership_log_probability[0].float().exp()
    existence_probability = predictions.existence_logits[0].float().sigmoid()
    row_to_track = assignment.row_to_track[0]
    matched_rows = (row_to_track >= 0).nonzero().flatten()
    matched_tracks = row_to_track.index_select(0, matched_rows)
    eligible_tracks = eligible_physical_tracks(targets, 0)
    if matched_tracks.numel() != torch.unique(matched_tracks).numel():
        raise RuntimeError("entity evaluation assignment repeats a physical track")
    if eligible_tracks.numel() and not torch.isin(eligible_tracks, matched_tracks).all():
        raise RuntimeError("entity evaluation assignment omitted an eligible physical track")
    track_count = targets.masks.shape[1]
    if (matched_tracks >= track_count).any():
        raise RuntimeError("entity evaluation assignment references an absent physical track")
    if matched_tracks.numel():
        assignable = (
            targets.track_valid[0, matched_tracks] & ~targets.capacity_censored[0, matched_tracks]
        )
        if not assignable.all():
            raise RuntimeError("entity evaluation assignment references an ineligible track")
    matched_has_evidence = torch.isin(matched_tracks, eligible_tracks)
    matched_is_carried = assignment.carried[0].index_select(0, matched_rows)
    if (~matched_has_evidence & ~matched_is_carried).any():
        raise RuntimeError(
            "entity evaluation assignment added an unproven current-frame physical track"
        )
    evidence_rows = matched_rows[matched_has_evidence]
    carried_unknown_rows = matched_rows[~matched_has_evidence]

    epsilon = torch.finfo(torch.float32).eps
    rows_evidence: list[dict[str, object]] = []
    target_union = torch.zeros(tokens, dtype=torch.float32, device=targets.masks.device)
    object_recall_numerator = torch.zeros((), dtype=torch.float32, device=targets.masks.device)
    object_recall_denominator = torch.zeros_like(object_recall_numerator)
    matched_support: list[torch.Tensor] = []
    for row_tensor in evidence_rows:
        row_index = int(row_tensor.item())
        track_index = int(row_to_track[row_index].item())
        valid = targets.mask_valid[0, track_index].float()
        weight = sensor_weight * valid
        target = targets.masks[0, track_index].float()
        support = support_probability[:, row_index]
        ownership = ownership_probability[:, row_index]
        target_weighted = target * weight
        target_mass = target_weighted.sum()
        if target_mass <= 0:
            # Existence-only evidence keeps an occluded physical entity in the
            # posterior assignment, but it cannot define a spatial IoU/Dice
            # denominator for this observation.
            continue
        support_mass = (support * weight).sum()
        support_intersection = (support * target_weighted).sum()
        support_union = ((support + target - support * target) * weight).sum()
        ownership_mass = (ownership * weight).sum()
        ownership_intersection = (ownership * target_weighted).sum()
        ownership_union = ((ownership + target - ownership * target) * weight).sum()
        support_iou = support_intersection / support_union.clamp_min(epsilon)
        support_iou_ceiling = maximum_token_grid_soft_iou(target, weight)
        area_fraction = _finite(target_mass / denominator)
        matched_support.append(support)
        target_union = torch.maximum(target_union, target * valid)
        object_recall_numerator += ownership_intersection
        object_recall_denominator += target_mass
        rows_evidence.append(
            {
                "row_index": row_index,
                "track_index": track_index,
                "identity_key": identity_keys[track_index],
                "area_fraction": area_fraction,
                "area_stratum": _area_stratum(area_fraction),
                "support_soft_iou": _finite(support_iou),
                "support_soft_iou_ceiling": _finite(support_iou_ceiling),
                "support_soft_iou_efficiency": _finite(
                    support_iou / support_iou_ceiling.clamp_min(epsilon)
                ),
                "support_soft_dice": _finite(
                    (2 * support_intersection) / (support_mass + target_mass).clamp_min(epsilon)
                ),
                "ownership_soft_iou": _finite(
                    ownership_intersection / ownership_union.clamp_min(epsilon)
                ),
                "ownership_target_recall": _finite(ownership_intersection / target_mass),
                "support_mass_fraction": _finite(support_mass / denominator),
                "ownership_mass_fraction": _finite(ownership_mass / denominator),
                "existence_probability": _finite(existence_probability[row_index]),
            }
        )

    context_target = (1 - target_union).clamp(0, 1)
    context_mass = (context_target * sensor_weight).sum()
    context_probability = ownership_probability[:, -1]
    context_region_probability = (
        _finite((context_probability * context_target * sensor_weight).sum() / context_mass)
        if context_mass > 0
        else None
    )
    pairwise_overlap: list[float] = []
    for first_index, first in enumerate(matched_support):
        first_mass = (first * sensor_weight).sum()
        for second in matched_support[first_index + 1 :]:
            second_mass = (second * sensor_weight).sum()
            overlap = (first * second * sensor_weight).sum() / torch.minimum(
                first_mass,
                second_mass,
            ).clamp_min(epsilon)
            pairwise_overlap.append(_finite(overlap))

    predicted_count = int((existence_probability >= 0.5).sum().item())
    target_evidence_count = int(eligible_tracks.numel())
    target_visible_count = len(rows_evidence)
    unmatched_rows = (row_to_track < 0) & ~assignment.reserved[0]
    unmatched_existence = [
        _finite(value) for value in existence_probability[unmatched_rows].unbind()
    ]
    cardinality_supervision_complete = bool(targets.inventory_exhaustive[0].item()) and not (
        bool(assignment.reserved[0].any().item()) or carried_unknown_rows.numel()
    )
    rows_evidence.sort(key=lambda item: int(item["track_index"]))
    return {
        "schema": ENTITY_SET_EVALUATION_SCHEMA,
        "target_evidence_count": target_evidence_count,
        "matched_evidence_count": int(evidence_rows.numel()),
        "matched_assignment_count": int(matched_rows.numel()),
        "carried_unknown_count": int(carried_unknown_rows.numel()),
        "reserved_unknown_count": int(assignment.reserved[0].sum().item()),
        "target_visible_count": target_visible_count,
        "matched_count": len(rows_evidence),
        "predicted_count_at_0_5": predicted_count,
        "cardinality_supervision_complete": cardinality_supervision_complete,
        "cardinality_absolute_error_at_0_5": (
            abs(predicted_count - target_evidence_count)
            if cardinality_supervision_complete
            else None
        ),
        "mean_unmatched_existence_probability": _mean(unmatched_existence),
        "context_region_probability": context_region_probability,
        "object_ownership_target_recall": (
            _finite(object_recall_numerator / object_recall_denominator)
            if object_recall_denominator > 0
            else None
        ),
        "mean_pairwise_support_overlap": _mean(pairwise_overlap),
        "rows": rows_evidence,
    }


def summarize_entity_evaluation_partition(
    samples: Sequence[Mapping[str, Any]],
    *,
    partition: str,
) -> dict[str, object]:
    """Macro-average samples and entities without task-target weighting."""

    selected = tuple(sample for sample in samples if sample.get("partition") == partition)
    if not selected:
        raise ValueError("entity evaluation partition has no samples")
    if len({sample.get("sample_key") for sample in selected}) != len(selected):
        raise ValueError("entity evaluation partition repeats a sample key")
    rows: list[Mapping[str, Any]] = []
    for sample in selected:
        if sample.get("schema") != ENTITY_SET_EVALUATION_SCHEMA:
            raise ValueError("entity evaluation sample schema changed")
        sample_rows = sample.get("rows")
        if not isinstance(sample_rows, list):
            raise ValueError("entity evaluation sample rows must be a list")
        rows.extend(sample_rows)
    if not rows:
        raise ValueError("entity evaluation partition has no matched physical entity")

    def row_mean(name: str, subset: Sequence[Mapping[str, Any]] = rows) -> float | None:
        values = [float(row[name]) for row in subset]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"entity evaluation row metric {name} is non-finite")
        return _mean(values)

    sample_metric_names = (
        "cardinality_absolute_error_at_0_5",
        "context_region_probability",
        "object_ownership_target_recall",
        "mean_pairwise_support_overlap",
    )
    sample_metrics: dict[str, float | None] = {}
    for name in sample_metric_names:
        values = [float(sample[name]) for sample in selected if sample.get(name) is not None]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"entity evaluation sample metric {name} is non-finite")
        sample_metrics[f"mean_{name}"] = _mean(values)

    strata: dict[str, object] = {}
    for name, _, _ in ENTITY_AREA_STRATA:
        subset = tuple(row for row in rows if row.get("area_stratum") == name)
        strata[name] = {
            "entity_count": len(subset),
            "mean_support_soft_iou": row_mean("support_soft_iou", subset),
            "mean_support_soft_iou_ceiling": row_mean(
                "support_soft_iou_ceiling",
                subset,
            ),
            "mean_support_soft_iou_efficiency": row_mean(
                "support_soft_iou_efficiency",
                subset,
            ),
            "mean_ownership_soft_iou": row_mean("ownership_soft_iou", subset),
            "mean_ownership_target_recall": row_mean("ownership_target_recall", subset),
        }
    return {
        "schema": ENTITY_SET_PARTITION_SUMMARY_SCHEMA,
        "partition": partition,
        "sample_count": len(selected),
        "task_count": len({sample.get("task_key") for sample in selected}),
        "entity_count": len(rows),
        "mean_support_soft_iou": row_mean("support_soft_iou"),
        "mean_support_soft_iou_ceiling": row_mean("support_soft_iou_ceiling"),
        "mean_support_soft_iou_efficiency": row_mean("support_soft_iou_efficiency"),
        "mean_support_soft_dice": row_mean("support_soft_dice"),
        "mean_ownership_soft_iou": row_mean("ownership_soft_iou"),
        "mean_ownership_target_recall": row_mean("ownership_target_recall"),
        "mean_existence_probability": row_mean("existence_probability"),
        **sample_metrics,
        "area_strata": strata,
    }
