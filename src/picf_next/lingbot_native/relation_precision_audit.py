"""Read-only precision audit for task-to-row relation scores."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.relations import (
    HOST_NATIVE_MATCH_INTERFACE,
    LEGACY_SHARED_COSINE_INTERFACE,
    RelationOutput,
)
from picf_next.lingbot_native.supervision import (
    assignment_binding_start_phase,
    assignment_binding_valid_at_phase,
    materialize_row_task_supervision,
)

RELATION_SCORE_PRECISION_EVIDENCE_SCHEMA = "picf-next.lingbot-relation-score-precision-evidence.v3"
RELATION_SCORE_PRECISION_SAMPLE_SCHEMA = "picf-next.lingbot-relation-score-precision-sample.v3"
RELATION_SCORE_PRECISION_AUDIT_SCHEMA = "picf-next.lingbot-relation-score-precision-audit.v3"

_STATISTIC_FIELDS = frozenset(
    {
        "all_row_unique_count",
        "matched_row_unique_count",
        "valid_row_unique_count",
        "minimum_target_logit",
        "maximum_known_negative_logit",
        "target_vs_hardest_negative_logit_margin",
        "worst_target_optimistic_rank",
        "worst_target_pessimistic_rank",
        "maximum_target_tied_alternative_count",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "task_interface",
        "production_dtype",
        "semantic_state_dtype",
        "row_embedding_dtype",
        "score_scale",
        "sequence_time_count",
        "source_time",
        "source_side",
        "source_phase",
        "binding_start_phase",
        "source_binding_valid",
        "row_to_track",
        "row_task_targets",
        "row_task_valid",
        "production_logits",
        "fp32_logits",
        "production_statistics",
        "fp32_statistics",
        "matched_pair_collision_count",
        "maximum_absolute_logit_error",
        "mean_absolute_logit_error",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "schema",
        "sample_key",
        "partition",
        "task_key",
        "factual_relation_sha256",
        "shuffled_task_relation_sha256",
        "factual",
        "shuffled_task",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "sample_count",
        "factual_mean_production_matched_unique_fraction",
        "factual_mean_fp32_matched_unique_fraction",
        "factual_restored_matched_pair_count",
        "factual_mean_production_target_margin",
        "factual_mean_fp32_target_margin",
        "factual_production_optimistic_rank_one_fraction",
        "factual_production_pessimistic_rank_one_fraction",
        "factual_fp32_optimistic_rank_one_fraction",
        "factual_fp32_pessimistic_rank_one_fraction",
        "factual_mean_maximum_absolute_logit_error",
        "shuffled_mean_production_matched_unique_fraction",
        "shuffled_mean_fp32_matched_unique_fraction",
        "shuffled_restored_matched_pair_count",
        "shuffled_mean_maximum_absolute_logit_error",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "checkpoint_global_step",
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "representation_evaluation_plan_sha256",
        "samples",
        "summary",
        "artifact_sha256",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    measured = float(value)
    if not math.isfinite(measured):
        raise ValueError(f"{name} must be finite")
    return measured


def _optional_float(value: object, *, name: str) -> float | None:
    return None if value is None else _finite_float(value, name=name)


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _score_statistics(
    *,
    logits: list[float],
    row_to_track: list[int],
    targets: list[float],
    valid: list[bool],
) -> dict[str, object]:
    matched_rows = [index for index, track in enumerate(row_to_track) if track >= 0]
    valid_rows = [index for index, is_valid in enumerate(valid) if is_valid]
    target_rows = [
        index
        for index, (target, is_valid) in enumerate(zip(targets, valid, strict=True))
        if is_valid and target > 0
    ]
    negative_rows = [
        index
        for index, (target, is_valid) in enumerate(zip(targets, valid, strict=True))
        if is_valid and target == 0
    ]
    value: dict[str, object] = {
        "all_row_unique_count": len(set(logits)),
        "matched_row_unique_count": len({logits[index] for index in matched_rows}),
        "valid_row_unique_count": len({logits[index] for index in valid_rows}),
        "minimum_target_logit": None,
        "maximum_known_negative_logit": None,
        "target_vs_hardest_negative_logit_margin": None,
        "worst_target_optimistic_rank": None,
        "worst_target_pessimistic_rank": None,
        "maximum_target_tied_alternative_count": None,
    }
    if not target_rows or not negative_rows:
        return value
    minimum_target = min(logits[index] for index in target_rows)
    maximum_negative = max(logits[index] for index in negative_rows)
    value.update(
        {
            "minimum_target_logit": minimum_target,
            "maximum_known_negative_logit": maximum_negative,
            "target_vs_hardest_negative_logit_margin": minimum_target - maximum_negative,
            "worst_target_optimistic_rank": max(
                1
                + sum(
                    int(index != target_row and is_valid and logit > logits[target_row])
                    for index, (logit, is_valid) in enumerate(zip(logits, valid, strict=True))
                )
                for target_row in target_rows
            ),
            "worst_target_pessimistic_rank": max(
                1
                + sum(
                    int(index != target_row and is_valid and logit >= logits[target_row])
                    for index, (logit, is_valid) in enumerate(zip(logits, valid, strict=True))
                )
                for target_row in target_rows
            ),
            "maximum_target_tied_alternative_count": max(
                sum(
                    int(index != target_row and is_valid and logit == logits[target_row])
                    for index, (logit, is_valid) in enumerate(zip(logits, valid, strict=True))
                )
                for target_row in target_rows
            ),
        }
    )
    return value


def _validate_statistics(value: object, *, expected: Mapping[str, object], name: str) -> None:
    if not isinstance(value, dict) or set(value) != _STATISTIC_FIELDS:
        raise ValueError(f"{name} score statistics fields differ from schema")
    integer_fields = (
        "all_row_unique_count",
        "matched_row_unique_count",
        "valid_row_unique_count",
    )
    optional_integer_fields = (
        "worst_target_optimistic_rank",
        "worst_target_pessimistic_rank",
        "maximum_target_tied_alternative_count",
    )
    optional_float_fields = (
        "minimum_target_logit",
        "maximum_known_negative_logit",
        "target_vs_hardest_negative_logit_margin",
    )
    for field in integer_fields:
        measured = value[field]
        if (
            isinstance(measured, bool)
            or not isinstance(measured, int)
            or measured < 0
            or measured != expected[field]
        ):
            raise ValueError(f"{name} {field} was not recomputed")
    for field in optional_integer_fields:
        measured = value[field]
        if (
            measured is not None and (isinstance(measured, bool) or not isinstance(measured, int))
        ) or measured != expected[field]:
            raise ValueError(f"{name} {field} was not recomputed")
    for field in optional_float_fields:
        measured = _optional_float(value[field], name=f"{name} {field}")
        expected_value = _optional_float(expected[field], name=f"expected {name} {field}")
        if (measured is None) is not (expected_value is None) or (
            measured is not None
            and expected_value is not None
            and not math.isclose(measured, expected_value, rel_tol=1e-7, abs_tol=1e-7)
        ):
            raise ValueError(f"{name} {field} was not recomputed")


def fp32_task_relation_logits(relation: RelationOutput) -> torch.Tensor:
    """Recompute task-to-row scores after the forward using FP32 arithmetic."""

    if not isinstance(relation, RelationOutput):
        raise TypeError("relation precision audit requires RelationOutput")
    if relation.task_interface == HOST_NATIVE_MATCH_INTERFACE:
        match = relation.match_embeddings
        fp32 = relation.task_relevance_logits_fp32
        if (
            relation.task_embedding is not None
            or match is None
            or fp32 is None
            or match.ndim != 3
            or fp32.shape != match.shape[:2]
            or relation.task_relevance_logits.shape != fp32.shape
        ):
            raise ValueError("host-native relation precision tensors have inconsistent shapes")
        if (
            not match.is_floating_point()
            or fp32.dtype != torch.float32
            or not torch.isfinite(match).all()
            or not torch.isfinite(fp32).all()
        ):
            raise ValueError("host-native relation precision tensors are malformed")
        return fp32.detach()
    if relation.task_interface != LEGACY_SHARED_COSINE_INTERFACE:
        raise ValueError("relation precision audit received an unknown task interface")
    task = relation.task_embedding
    rows = relation.row_embeddings
    temperature = relation.relation_temperature
    if (
        task is None
        or task.ndim != 2
        or rows.ndim != 3
        or task.shape[0] != rows.shape[0]
        or task.shape[-1] != rows.shape[-1]
        or relation.task_relevance_logits.shape != rows.shape[:2]
        or temperature.numel() != 1
    ):
        raise ValueError("relation precision tensors have inconsistent shapes")
    if any(not tensor.is_floating_point() for tensor in (task, rows, temperature)):
        raise TypeError("relation precision tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in (task, rows, temperature)):
        raise ValueError("relation precision tensors contain non-finite values")
    temperature32 = temperature.detach().float().reshape(())
    if not bool(temperature32 > 0):
        raise ValueError("relation precision temperature must be positive")
    return (
        torch.einsum(
            "bd,bkd->bk",
            task.detach().float(),
            rows.detach().float(),
        )
        / temperature32
    )


def build_relation_score_precision_evidence(
    relation: RelationOutput,
    objective: NativeCALVINObjectiveResult,
    *,
    batch_index: int,
) -> dict[str, object]:
    """Compare production and FP32 scores without changing the forward graph."""

    if not isinstance(objective, NativeCALVINObjectiveResult):
        raise TypeError("relation precision evidence requires a native CALVIN objective")
    batch, rows = relation.task_relevance_logits.shape
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or not 0 <= batch_index < batch
    ):
        raise IndexError("relation precision batch index is outside the relation")
    if objective.predictions.task_relevance_logits.shape != (batch, rows) or not torch.equal(
        objective.predictions.task_relevance_logits,
        relation.task_relevance_logits,
    ):
        raise ValueError("relation precision objective differs from the observed relation")
    if relation.task_interface == HOST_NATIVE_MATCH_INTERFACE:
        semantic_state = relation.match_embeddings
        score_scale = 1.0
        if relation.task_embedding is not None or semantic_state is None:
            raise ValueError("host-native task relation exposed the wrong semantic state")
    elif relation.task_interface == LEGACY_SHARED_COSINE_INTERFACE:
        semantic_state = relation.task_embedding
        if semantic_state is None:
            raise ValueError("legacy task relation omitted its task embedding")
        production_recomputed = torch.einsum(
            "bd,bkd->bk",
            semantic_state.detach(),
            relation.row_embeddings.detach(),
        ) / relation.relation_temperature.detach().to(semantic_state.dtype).reshape(())
        if not torch.equal(production_recomputed, relation.task_relevance_logits.detach()):
            raise ValueError("production task-to-row logits do not match their relation embeddings")
        score_scale = float(relation.relation_temperature.detach().float().reshape(()).cpu())
    else:
        raise ValueError("relation precision evidence received an unknown task interface")
    row_task = materialize_row_task_supervision(
        objective.targets,
        objective.assignment,
        batch_index=batch_index,
        dtype=torch.float32,
    )
    sequence_time_count = int(objective.targets.masks.shape[1])
    source_time = sequence_time_count - 1
    source_phase = 2 * source_time + 1
    binding_start_phase_tensor = assignment_binding_start_phase(
        objective.assignment,
        objective.targets,
    )[batch_index]
    source_binding_valid_tensor = assignment_binding_valid_at_phase(
        objective.assignment,
        objective.targets,
        source_phase=source_phase,
    )[batch_index]
    production = relation.task_relevance_logits[batch_index].detach().float().cpu().tolist()
    fp32 = fp32_task_relation_logits(relation)[batch_index].cpu().tolist()
    row_to_track = objective.assignment.row_to_track[batch_index].detach().cpu().tolist()
    targets = row_task.target.detach().float().cpu().tolist()
    valid = (row_task.valid & source_binding_valid_tensor).detach().cpu().tolist()
    binding_start_phase = binding_start_phase_tensor.detach().cpu().tolist()
    source_binding_valid = source_binding_valid_tensor.detach().cpu().tolist()
    production_values = [float(value) for value in production]
    fp32_values = [float(value) for value in fp32]
    target_values = [float(value) for value in targets]
    valid_values = [bool(value) for value in valid]
    matched_rows = [index for index, track in enumerate(row_to_track) if track >= 0]
    collisions = sum(
        int(
            production_values[left] == production_values[right]
            and fp32_values[left] != fp32_values[right]
        )
        for offset, left in enumerate(matched_rows)
        for right in matched_rows[offset + 1 :]
    )
    absolute_error = [
        abs(production_value - fp32_value)
        for production_value, fp32_value in zip(
            production_values,
            fp32_values,
            strict=True,
        )
    ]
    value: dict[str, object] = {
        "schema": RELATION_SCORE_PRECISION_EVIDENCE_SCHEMA,
        "task_interface": relation.task_interface,
        "production_dtype": str(relation.task_relevance_logits.dtype),
        "semantic_state_dtype": str(semantic_state.dtype),
        "row_embedding_dtype": str(relation.row_embeddings.dtype),
        "score_scale": score_scale,
        "sequence_time_count": sequence_time_count,
        "source_time": source_time,
        "source_side": "posterior",
        "source_phase": source_phase,
        "binding_start_phase": binding_start_phase,
        "source_binding_valid": source_binding_valid,
        "row_to_track": row_to_track,
        "row_task_targets": target_values,
        "row_task_valid": valid_values,
        "production_logits": production_values,
        "fp32_logits": fp32_values,
        "production_statistics": _score_statistics(
            logits=production_values,
            row_to_track=row_to_track,
            targets=target_values,
            valid=valid_values,
        ),
        "fp32_statistics": _score_statistics(
            logits=fp32_values,
            row_to_track=row_to_track,
            targets=target_values,
            valid=valid_values,
        ),
        "matched_pair_collision_count": collisions,
        "maximum_absolute_logit_error": max(absolute_error),
        "mean_absolute_logit_error": math.fsum(absolute_error) / len(absolute_error),
    }
    return validate_relation_score_precision_evidence(value)


def validate_relation_score_precision_evidence(value: object) -> dict[str, Any]:
    """Recompute every scalar derivable from persisted score arrays."""

    if (
        not isinstance(value, dict)
        or set(value) != _EVIDENCE_FIELDS
        or value["schema"] != RELATION_SCORE_PRECISION_EVIDENCE_SCHEMA
    ):
        raise ValueError("relation score precision evidence fields differ from schema")
    if value["task_interface"] not in (
        HOST_NATIVE_MATCH_INTERFACE,
        LEGACY_SHARED_COSINE_INTERFACE,
    ):
        raise ValueError("relation precision task interface is malformed")
    for field in ("production_dtype", "semantic_state_dtype", "row_embedding_dtype"):
        if not isinstance(value[field], str) or not value[field].startswith("torch."):
            raise ValueError(f"relation precision {field} is malformed")
    score_scale = _finite_float(value["score_scale"], name="relation precision score scale")
    if score_scale <= 0:
        raise ValueError("relation precision score scale must be positive")
    sequence_time_count = value["sequence_time_count"]
    source_time = value["source_time"]
    source_phase = value["source_phase"]
    if (
        isinstance(sequence_time_count, bool)
        or not isinstance(sequence_time_count, int)
        or sequence_time_count <= 0
        or isinstance(source_time, bool)
        or not isinstance(source_time, int)
        or source_time != sequence_time_count - 1
        or value["source_side"] != "posterior"
        or isinstance(source_phase, bool)
        or not isinstance(source_phase, int)
        or source_phase != 2 * source_time + 1
    ):
        raise ValueError("relation precision source cut is malformed")
    arrays = (
        value["row_to_track"],
        value["binding_start_phase"],
        value["source_binding_valid"],
        value["row_task_targets"],
        value["row_task_valid"],
        value["production_logits"],
        value["fp32_logits"],
    )
    if any(not isinstance(array, list) or not array for array in arrays):
        raise ValueError("relation precision arrays must be nonempty lists")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("relation precision arrays have different row counts")
    row_to_track = value["row_to_track"]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < -1 for item in row_to_track
    ):
        raise ValueError("relation precision row assignment is malformed")
    binding_start_phase = value["binding_start_phase"]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in binding_start_phase
    ):
        raise ValueError("relation precision binding phase is malformed")
    source_binding_valid = value["source_binding_valid"]
    if any(not isinstance(item, bool) for item in source_binding_valid):
        raise TypeError("relation precision source binding validity must be boolean")
    expected_source_valid = [
        track >= 0 and phase <= source_phase
        for track, phase in zip(row_to_track, binding_start_phase, strict=True)
    ]
    if source_binding_valid != expected_source_valid:
        raise ValueError("relation precision row validity differs from its source cut")
    targets = [
        _finite_float(item, name="relation precision row target")
        for item in value["row_task_targets"]
    ]
    if any(not 0 <= item <= 1 for item in targets):
        raise ValueError("relation precision row targets must lie in [0,1]")
    valid = value["row_task_valid"]
    if any(not isinstance(item, bool) for item in valid):
        raise TypeError("relation precision row validity must be boolean")
    production = [
        _finite_float(item, name="relation precision production logit")
        for item in value["production_logits"]
    ]
    fp32 = [
        _finite_float(item, name="relation precision FP32 logit") for item in value["fp32_logits"]
    ]
    production_statistics = _score_statistics(
        logits=production,
        row_to_track=row_to_track,
        targets=targets,
        valid=valid,
    )
    fp32_statistics = _score_statistics(
        logits=fp32,
        row_to_track=row_to_track,
        targets=targets,
        valid=valid,
    )
    _validate_statistics(
        value["production_statistics"],
        expected=production_statistics,
        name="production",
    )
    _validate_statistics(value["fp32_statistics"], expected=fp32_statistics, name="FP32")
    matched_rows = [index for index, track in enumerate(row_to_track) if track >= 0]
    collisions = sum(
        int(production[left] == production[right] and fp32[left] != fp32[right])
        for offset, left in enumerate(matched_rows)
        for right in matched_rows[offset + 1 :]
    )
    if value["matched_pair_collision_count"] != collisions:
        raise ValueError("relation precision collision count was not recomputed")
    absolute_error = [
        abs(production_value - fp32_value)
        for production_value, fp32_value in zip(production, fp32, strict=True)
    ]
    expected_maximum = max(absolute_error)
    expected_mean = math.fsum(absolute_error) / len(absolute_error)
    maximum = _finite_float(
        value["maximum_absolute_logit_error"],
        name="relation precision maximum logit error",
    )
    mean = _finite_float(
        value["mean_absolute_logit_error"],
        name="relation precision mean logit error",
    )
    if not math.isclose(maximum, expected_maximum, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("relation precision maximum logit error was not recomputed")
    if not math.isclose(mean, expected_mean, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("relation precision mean logit error was not recomputed")
    return value


def build_relation_score_precision_sample(
    *,
    sample_key: str,
    partition: str,
    task_key: str,
    factual_relation_sha256: str,
    shuffled_task_relation_sha256: str,
    factual: Mapping[str, object],
    shuffled_task: Mapping[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": RELATION_SCORE_PRECISION_SAMPLE_SCHEMA,
        "sample_key": sample_key,
        "partition": partition,
        "task_key": task_key,
        "factual_relation_sha256": factual_relation_sha256,
        "shuffled_task_relation_sha256": shuffled_task_relation_sha256,
        "factual": dict(factual),
        "shuffled_task": dict(shuffled_task),
    }
    return validate_relation_score_precision_sample(value)


def validate_relation_score_precision_sample(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _SAMPLE_FIELDS
        or value["schema"] != RELATION_SCORE_PRECISION_SAMPLE_SCHEMA
    ):
        raise ValueError("relation score precision sample fields differ from schema")
    for field in ("sample_key", "partition", "task_key"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"relation precision sample {field} must be nonempty")
    _sha256(value["factual_relation_sha256"], name="factual relation")
    _sha256(value["shuffled_task_relation_sha256"], name="shuffled relation")
    validate_relation_score_precision_evidence(value["factual"])
    validate_relation_score_precision_evidence(value["shuffled_task"])
    return value


def _rank_one_fraction(
    evidence: Sequence[Mapping[str, Any]],
    *,
    surface: str,
    rank: str,
) -> float | None:
    values = [
        item[f"{surface}_statistics"][f"worst_target_{rank}_rank"]
        for item in evidence
        if item[f"{surface}_statistics"][f"worst_target_{rank}_rank"] is not None
    ]
    return None if not values else sum(int(value == 1) for value in values) / len(values)


def _audit_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    factual = [sample["factual"] for sample in samples]
    shuffled = [sample["shuffled_task"] for sample in samples]

    def unique_fraction(item: Mapping[str, Any], surface: str) -> float:
        matched = sum(int(track >= 0) for track in item["row_to_track"])
        if matched <= 0:
            raise ValueError("relation precision sample has no matched row")
        return item[f"{surface}_statistics"]["matched_row_unique_count"] / matched

    def margins(items: Sequence[Mapping[str, Any]], surface: str) -> list[float]:
        return [
            float(value)
            for item in items
            if (value := item[f"{surface}_statistics"]["target_vs_hardest_negative_logit_margin"])
            is not None
        ]

    return {
        "sample_count": len(samples),
        "factual_mean_production_matched_unique_fraction": _mean(
            [unique_fraction(item, "production") for item in factual]
        ),
        "factual_mean_fp32_matched_unique_fraction": _mean(
            [unique_fraction(item, "fp32") for item in factual]
        ),
        "factual_restored_matched_pair_count": sum(
            item["matched_pair_collision_count"] for item in factual
        ),
        "factual_mean_production_target_margin": _mean(margins(factual, "production")),
        "factual_mean_fp32_target_margin": _mean(margins(factual, "fp32")),
        "factual_production_optimistic_rank_one_fraction": _rank_one_fraction(
            factual,
            surface="production",
            rank="optimistic",
        ),
        "factual_production_pessimistic_rank_one_fraction": _rank_one_fraction(
            factual,
            surface="production",
            rank="pessimistic",
        ),
        "factual_fp32_optimistic_rank_one_fraction": _rank_one_fraction(
            factual,
            surface="fp32",
            rank="optimistic",
        ),
        "factual_fp32_pessimistic_rank_one_fraction": _rank_one_fraction(
            factual,
            surface="fp32",
            rank="pessimistic",
        ),
        "factual_mean_maximum_absolute_logit_error": _mean(
            [item["maximum_absolute_logit_error"] for item in factual]
        ),
        "shuffled_mean_production_matched_unique_fraction": _mean(
            [unique_fraction(item, "production") for item in shuffled]
        ),
        "shuffled_mean_fp32_matched_unique_fraction": _mean(
            [unique_fraction(item, "fp32") for item in shuffled]
        ),
        "shuffled_restored_matched_pair_count": sum(
            item["matched_pair_collision_count"] for item in shuffled
        ),
        "shuffled_mean_maximum_absolute_logit_error": _mean(
            [item["maximum_absolute_logit_error"] for item in shuffled]
        ),
    }


def build_relation_score_precision_audit(
    *,
    checkpoint_global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    representation_split_sha256: str,
    representation_evaluation_plan_sha256: str,
    expected_sample_keys: Sequence[str],
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ordered = [dict(sample) for sample in samples]
    value: dict[str, object] = {
        "schema": RELATION_SCORE_PRECISION_AUDIT_SCHEMA,
        "status": "PASS",
        "checkpoint_global_step": checkpoint_global_step,
        "implementation_sha256": implementation_sha256,
        "model_family_sha256": model_family_sha256,
        "representation_split_sha256": representation_split_sha256,
        "representation_evaluation_plan_sha256": representation_evaluation_plan_sha256,
        "samples": ordered,
        "summary": _audit_summary(
            [validate_relation_score_precision_sample(sample) for sample in ordered]
        ),
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return validate_relation_score_precision_audit(
        value,
        expected_sample_keys=expected_sample_keys,
    )


def validate_relation_score_precision_audit(
    value: object,
    *,
    expected_sample_keys: Sequence[str],
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _AUDIT_FIELDS
        or value["schema"] != RELATION_SCORE_PRECISION_AUDIT_SCHEMA
        or value["status"] != "PASS"
    ):
        raise ValueError("relation score precision audit fields differ from schema")
    step = value["checkpoint_global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("relation precision checkpoint step must be non-negative")
    for field in (
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "representation_evaluation_plan_sha256",
    ):
        _sha256(value[field], name=f"relation precision {field}")
    raw_samples = value["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("relation precision audit requires samples")
    samples = [validate_relation_score_precision_sample(sample) for sample in raw_samples]
    sample_keys = [sample["sample_key"] for sample in samples]
    if sample_keys != list(expected_sample_keys) or len(set(sample_keys)) != len(sample_keys):
        raise ValueError("relation precision sample coverage differs from the fixed plan")
    summary = value["summary"]
    if not isinstance(summary, dict) or set(summary) != _SUMMARY_FIELDS:
        raise ValueError("relation precision summary fields differ from schema")
    expected_summary = _audit_summary(samples)
    if summary != expected_summary:
        raise ValueError("relation precision summary was not recomputed")
    artifact = _sha256(value["artifact_sha256"], name="relation precision artifact")
    payload = {field: value[field] for field in _AUDIT_FIELDS if field != "artifact_sha256"}
    if _canonical_sha256(payload) != artifact:
        raise ValueError("relation precision artifact SHA-256 changed")
    return value
