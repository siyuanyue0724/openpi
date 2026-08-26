"""Post-hoc oracle separating posterior row drift from content loss."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from picf_next.lingbot_native.representation_evaluation import (
    REPRESENTATION_EVALUATION_SNAPSHOT_SCHEMA,
    validate_representation_evaluation_sample,
)

REPRESENTATION_ROW_PERMUTATION_AUDIT_SCHEMA = (
    "picf-next.lingbot-representation-row-permutation-audit.v1"
)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _pair_soft_iou(
    prediction: Sequence[float],
    target: Sequence[float],
    weight: Sequence[float],
) -> float:
    if not len(prediction) == len(target) == len(weight) or not prediction:
        raise ValueError("row-permutation vectors must share one nonempty token axis")
    intersection = math.fsum(
        measured * expected * token_weight
        for measured, expected, token_weight in zip(prediction, target, weight, strict=True)
    )
    union = math.fsum(
        token_weight * (measured + expected - measured * expected)
        for measured, expected, token_weight in zip(prediction, target, weight, strict=True)
    )
    if union <= 0:
        raise ValueError("row-permutation pair has no measurable union")
    return intersection / union


def audit_row_permutation_sample(sample: Mapping[str, object]) -> dict[str, object] | None:
    """Find the best visible-row permutation without changing model outputs."""

    raw_rows = sample.get("factual_ownership_rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, str | bytes) or not raw_rows:
        return None
    rows = tuple(dict(row) for row in raw_rows if isinstance(row, Mapping))
    if len(rows) != len(raw_rows):
        raise ValueError("row-permutation ownership rows are malformed")
    identities = tuple(str(row["identity_key"]) for row in rows)
    row_indices = tuple(int(row["row_index"]) for row in rows)
    if len(set(identities)) != len(rows) or len(set(row_indices)) != len(rows):
        raise ValueError("row-permutation sample reused an identity or model row")
    weights = tuple(float(value) for value in rows[0]["weight"])
    if any(tuple(float(value) for value in row["weight"]) != weights for row in rows[1:]):
        raise ValueError("row-permutation sample changed valid-token weights across rows")

    score = np.asarray(
        [
            [
                _pair_soft_iou(
                    tuple(float(value) for value in prediction_row["prediction"]),
                    tuple(float(value) for value in target_row["target"]),
                    weights,
                )
                for target_row in rows
            ]
            for prediction_row in rows
        ],
        dtype=np.float64,
    )
    prediction_positions, target_positions = linear_sum_assignment(-score)
    target_to_prediction = {
        int(target_position): int(prediction_position)
        for prediction_position, target_position in zip(
            prediction_positions,
            target_positions,
            strict=True,
        )
    }
    assignments = tuple(
        {
            "prediction_row_index": row_indices[prediction_position],
            "previous_identity_key": identities[prediction_position],
            "oracle_identity_key": identities[target_position],
            "soft_iou": float(score[prediction_position, target_position]),
        }
        for prediction_position, target_position in zip(
            prediction_positions,
            target_positions,
            strict=True,
        )
    )
    assigned_macro = math.fsum(float(row["soft_iou"]) for row in rows) / len(rows)
    oracle_macro = math.fsum(float(item["soft_iou"]) for item in assignments) / len(assignments)

    raw_target_identities = sample.get("factual_target_identity_keys")
    target_identities = (
        tuple(str(value) for value in raw_target_identities)
        if isinstance(raw_target_identities, Sequence)
        and not isinstance(raw_target_identities, str | bytes)
        else ()
    )
    identity_position = {identity: position for position, identity in enumerate(identities)}
    materialized_targets = tuple(
        identity for identity in target_identities if identity in identity_position
    )
    assigned_target_iou = _mean(
        [float(rows[identity_position[identity]]["soft_iou"]) for identity in materialized_targets]
    )
    oracle_target_iou = _mean(
        [
            float(
                score[
                    target_to_prediction[identity_position[identity]],
                    identity_position[identity],
                ]
            )
            for identity in materialized_targets
        ]
    )

    diagnostic = sample.get("factual_task_row_diagnostic")
    oracle_margin: float | None = None
    oracle_rank_one: bool | None = None
    if materialized_targets and isinstance(diagnostic, Mapping) and diagnostic.get("exact_task"):
        logits = tuple(float(value) for value in diagnostic["task_logits"])
        valid = tuple(bool(value) for value in diagnostic["row_task_valid"])
        if len(logits) != len(valid):
            raise ValueError("row-permutation task logits and validity differ")
        oracle_target_rows = tuple(
            row_indices[target_to_prediction[identity_position[identity]]]
            for identity in materialized_targets
        )
        if any(row >= len(logits) or not valid[row] for row in oracle_target_rows):
            raise ValueError("row-permutation oracle selected an invalid task row")
        negatives = tuple(
            logits[row]
            for row in range(len(logits))
            if valid[row] and row not in oracle_target_rows
        )
        if not negatives:
            raise ValueError("row-permutation task audit has no known negative row")
        oracle_margin = min(logits[row] for row in oracle_target_rows) - max(negatives)
        oracle_rank_one = oracle_margin > 0

    return {
        "partition": str(sample["partition"]),
        "task_key": str(sample["task_key"]),
        "sample_key": str(sample["sample_key"]),
        "visible_row_count": len(rows),
        "changed_identity_count": sum(
            item["previous_identity_key"] != item["oracle_identity_key"] for item in assignments
        ),
        "assigned_macro_soft_iou": assigned_macro,
        "oracle_macro_soft_iou": oracle_macro,
        "assigned_target_soft_iou": assigned_target_iou,
        "oracle_target_soft_iou": oracle_target_iou,
        "assigned_task_margin": (
            None
            if not isinstance(diagnostic, Mapping)
            else diagnostic.get("target_vs_hardest_negative_logit_margin")
        ),
        "assigned_task_rank_one": (
            None
            if not isinstance(diagnostic, Mapping)
            else diagnostic.get("all_targets_beat_known_negatives")
        ),
        "oracle_task_margin": oracle_margin,
        "oracle_task_rank_one": oracle_rank_one,
        "assignments": list(assignments),
    }


def audit_representation_row_permutation_snapshot(
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Audit one persisted reset or warm snapshot with a label-only oracle."""

    if (
        snapshot.get("schema") != REPRESENTATION_EVALUATION_SNAPSHOT_SCHEMA
        or snapshot.get("status") != "PASS"
    ):
        raise ValueError("row-permutation audit requires one passing representation snapshot")
    raw_samples = snapshot.get("samples")
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, str | bytes):
        raise ValueError("row-permutation snapshot samples are malformed")
    samples = tuple(
        validate_representation_evaluation_sample(dict(sample)) for sample in raw_samples
    )
    audited = tuple(
        result
        for sample in samples
        if (result := audit_row_permutation_sample(sample)) is not None
    )
    if not audited:
        raise ValueError("row-permutation audit found no ownership-eligible sample")

    by_partition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in audited:
        by_partition[str(sample["partition"])].append(sample)
    summaries: dict[str, object] = {}
    for partition, values in sorted(by_partition.items()):
        target_values = [
            sample for sample in values if sample["oracle_target_soft_iou"] is not None
        ]
        task_values = [sample for sample in values if sample["oracle_task_margin"] is not None]
        summaries[partition] = {
            "sample_count": len(values),
            "visible_row_count": sum(int(sample["visible_row_count"]) for sample in values),
            "changed_identity_fraction": (
                sum(int(sample["changed_identity_count"]) for sample in values)
                / sum(int(sample["visible_row_count"]) for sample in values)
            ),
            "mean_assigned_macro_soft_iou": _mean(
                [float(sample["assigned_macro_soft_iou"]) for sample in values]
            ),
            "mean_oracle_macro_soft_iou": _mean(
                [float(sample["oracle_macro_soft_iou"]) for sample in values]
            ),
            "mean_assigned_target_soft_iou": _mean(
                [float(sample["assigned_target_soft_iou"]) for sample in target_values]
            ),
            "mean_oracle_target_soft_iou": _mean(
                [float(sample["oracle_target_soft_iou"]) for sample in target_values]
            ),
            "mean_assigned_sample_task_margin": _mean(
                [float(sample["assigned_task_margin"]) for sample in task_values]
            ),
            "mean_oracle_sample_task_margin": _mean(
                [float(sample["oracle_task_margin"]) for sample in task_values]
            ),
            "assigned_sample_rank_one_fraction": _mean(
                [float(bool(sample["assigned_task_rank_one"])) for sample in task_values]
            ),
            "oracle_sample_rank_one_fraction": _mean(
                [float(bool(sample["oracle_task_rank_one"])) for sample in task_values]
            ),
        }

    value: dict[str, object] = {
        "schema": REPRESENTATION_ROW_PERMUTATION_AUDIT_SCHEMA,
        "source_snapshot_artifact_sha256": snapshot.get("artifact_sha256"),
        "checkpoint_global_step": snapshot.get("checkpoint_global_step"),
        "limitation": (
            "oracle permutes only rows already bound to visible physical identities; "
            "migration into an unmatched row is not recoverable from this snapshot"
        ),
        "partition_summaries": summaries,
        "samples": list(audited),
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return value
