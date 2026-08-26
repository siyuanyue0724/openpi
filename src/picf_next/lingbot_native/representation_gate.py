"""Preregistered numeric decision for the bounded representation trial."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
    validate_representation_evaluation_snapshot,
)

REPRESENTATION_NUMERIC_GATE_SCHEMA = "picf-next.lingbot-representation-numeric-gate.v1"
REPRESENTATION_BASELINE_STEP = 0
REPRESENTATION_DECISION_STEP = 200
REPRESENTATION_MINIMUM_AUC_DELTA = 0.05
REPRESENTATION_MINIMUM_TASK_FRACTION = 2.0 / 3.0
REPRESENTATION_MINIMUM_TARGET_OWNERSHIP_SOFT_IOU = 0.05
REPRESENTATION_MINIMUM_OWNERSHIP_MULTIPLIER = 2.0
REPRESENTATION_MAXIMUM_ACTION_RELATIVE_REGRESSION = 0.25
REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION = 0.0

_GATE_FIELDS = frozenset(
    {
        "schema",
        "status",
        "baseline_snapshot_sha256",
        "decision_snapshot_sha256",
        "representation_evaluation_plan_sha256",
        "thresholds",
        "metrics",
        "checks",
        "visual_review_required",
        "authorizes_joint_adoption",
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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite(value: object, *, name: str) -> float | None:
    return None if value is None else _finite(value, name=name)


def _heldout(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    summaries = snapshot["partition_summaries"]
    if not isinstance(summaries, Mapping) or not isinstance(summaries.get("heldout"), Mapping):
        raise ValueError("representation gate snapshot has no held-out summary")
    return summaries["heldout"]


def _task_auc_deltas(
    baseline: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, float]:
    baseline_tasks = baseline["task_metrics"]
    decision_tasks = decision["task_metrics"]
    if not isinstance(baseline_tasks, Mapping) or not isinstance(decision_tasks, Mapping):
        raise ValueError("representation gate task metrics are malformed")
    if set(baseline_tasks) != set(decision_tasks):
        raise ValueError("representation gate task coverage changed between checkpoints")
    deltas: dict[str, float] = {}
    for task_key in sorted(baseline_tasks):
        baseline_auc = baseline_tasks[task_key]["fractional_weighted_auc"]
        decision_auc = decision_tasks[task_key]["fractional_weighted_auc"]
        if (baseline_auc is None) is not (decision_auc is None):
            raise ValueError("representation token eligibility changed between checkpoints")
        if baseline_auc is not None and decision_auc is not None:
            deltas[str(task_key)] = _finite(
                decision_auc,
                name="decision task AUC",
            ) - _finite(
                baseline_auc,
                name="baseline task AUC",
            )
    expected = int(baseline["token_eligible_task_count"])
    if len(deltas) != expected or expected != int(decision["token_eligible_task_count"]):
        raise ValueError("representation gate eligible task coverage changed")
    return deltas


def _check(
    *,
    passed: bool,
    actual: object,
    relation: str,
    threshold: object,
) -> dict[str, object]:
    return {
        "passed": bool(passed),
        "actual": actual,
        "relation": relation,
        "threshold": threshold,
    }


def _representation_numeric_gate_value(
    baseline_snapshot: Mapping[str, Any],
    decision_snapshot: Mapping[str, Any],
    *,
    plan: RepresentationEvaluationPlan,
) -> dict[str, object]:
    baseline = validate_representation_evaluation_snapshot(
        dict(baseline_snapshot),
        plan=plan,
    )
    decision = validate_representation_evaluation_snapshot(
        dict(decision_snapshot),
        plan=plan,
    )
    if (
        baseline["checkpoint_global_step"] != REPRESENTATION_BASELINE_STEP
        or decision["checkpoint_global_step"] != REPRESENTATION_DECISION_STEP
    ):
        raise ValueError("representation gate requires exactly checkpoints 0 and 200")
    binding_fields = (
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "representation_evaluation_plan_sha256",
    )
    if any(baseline[name] != decision[name] for name in binding_fields):
        raise ValueError("representation gate evidence bindings changed between checkpoints")

    baseline_heldout = _heldout(baseline)
    decision_heldout = _heldout(decision)
    task_deltas = _task_auc_deltas(baseline_heldout, decision_heldout)
    mean_auc_delta = None if not task_deltas else math.fsum(task_deltas.values()) / len(task_deltas)
    nonnegative_count = sum(delta >= 0.0 for delta in task_deltas.values())
    required_nonnegative_count = math.ceil(REPRESENTATION_MINIMUM_TASK_FRACTION * len(task_deltas))

    baseline_ownership = _optional_finite(
        baseline_heldout["mean_task_target_ownership_soft_iou"],
        name="baseline target ownership soft-IoU",
    )
    decision_ownership = _optional_finite(
        decision_heldout["mean_task_target_ownership_soft_iou"],
        name="decision target ownership soft-IoU",
    )
    required_ownership = (
        None
        if baseline_ownership is None
        else max(
            REPRESENTATION_MINIMUM_TARGET_OWNERSHIP_SOFT_IOU,
            REPRESENTATION_MINIMUM_OWNERSHIP_MULTIPLIER * baseline_ownership,
        )
    )
    baseline_action = _finite(
        baseline_heldout["mean_official_action_loss"],
        name="baseline official action loss",
    )
    decision_action = _finite(
        decision_heldout["mean_official_action_loss"],
        name="decision official action loss",
    )
    action_relative_regression = (
        0.0
        if baseline_action == 0.0 and decision_action == 0.0
        else math.inf
        if baseline_action == 0.0
        else (decision_action - baseline_action) / baseline_action
    )
    if not math.isfinite(action_relative_regression):
        action_relative_regression_for_artifact: float | None = None
    else:
        action_relative_regression_for_artifact = action_relative_regression

    rank_fraction = _optional_finite(
        decision_heldout["rank_one_task_fraction"],
        name="decision rank-one task fraction",
    )
    row_margin = _optional_finite(
        decision_heldout["mean_task_hardest_negative_logit_margin"],
        name="decision hardest-negative margin",
    )
    shuffled_task_degradation = _optional_finite(
        decision_heldout["mean_task_shuffled_task_auc_degradation"],
        name="decision shuffled-task degradation",
    )
    shuffled_target_degradation = _optional_finite(
        decision_heldout["mean_task_shuffled_target_auc_degradation"],
        name="decision shuffled-target degradation",
    )
    task_count = int(decision_heldout["task_count"])
    if task_count <= 0 or int(baseline_heldout["task_count"]) != task_count:
        raise ValueError("representation gate held-out task count changed")
    token_eligible_fraction = int(decision_heldout["token_eligible_task_count"]) / task_count
    row_eligible_fraction = int(decision_heldout["row_eligible_task_count"]) / task_count
    ownership_eligible_fraction = (
        int(decision_heldout["ownership_eligible_task_count"]) / task_count
    )
    control_eligible_count = int(decision_heldout["control_eligible_task_count"])
    control_eligible_fraction = control_eligible_count / task_count
    task_metrics = decision_heldout["task_metrics"]
    if not isinstance(task_metrics, Mapping):
        raise ValueError("representation gate held-out task metrics are malformed")
    control_pairs: list[tuple[float, float]] = []
    for task_key in sorted(task_metrics):
        task_metric = task_metrics[task_key]
        if not isinstance(task_metric, Mapping):
            raise ValueError("representation gate held-out task metric is malformed")
        shuffled_task = _optional_finite(
            task_metric["shuffled_task_auc_degradation"],
            name="task shuffled-task degradation",
        )
        shuffled_target = _optional_finite(
            task_metric["shuffled_target_auc_degradation"],
            name="task shuffled-target degradation",
        )
        if (shuffled_task is None) is not (shuffled_target is None):
            raise ValueError("representation gate control eligibility differs within a task")
        if shuffled_task is not None and shuffled_target is not None:
            control_pairs.append((shuffled_task, shuffled_target))
    if len(control_pairs) != control_eligible_count:
        raise ValueError("representation gate control task coverage was not recomputed")
    required_positive_control_count = math.ceil(
        REPRESENTATION_MINIMUM_TASK_FRACTION * control_eligible_count
    )
    positive_shuffled_task_count = sum(
        shuffled_task > REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION
        for shuffled_task, _ in control_pairs
    )
    positive_shuffled_target_count = sum(
        shuffled_target > REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION
        for _, shuffled_target in control_pairs
    )
    action_state_unchanged = (
        baseline["representation_frozen_action_state_sha256"]
        == decision["representation_frozen_action_state_sha256"]
    )
    checks = {
        "mean_task_auc_delta": _check(
            passed=(
                mean_auc_delta is not None and mean_auc_delta >= REPRESENTATION_MINIMUM_AUC_DELTA
            ),
            actual=mean_auc_delta,
            relation=">=",
            threshold=REPRESENTATION_MINIMUM_AUC_DELTA,
        ),
        "token_eligible_task_fraction": _check(
            passed=token_eligible_fraction >= REPRESENTATION_MINIMUM_TASK_FRACTION,
            actual=token_eligible_fraction,
            relation=">=",
            threshold=REPRESENTATION_MINIMUM_TASK_FRACTION,
        ),
        "nonnegative_task_auc_delta_fraction": _check(
            passed=nonnegative_count >= required_nonnegative_count,
            actual={
                "eligible_task_count": len(task_deltas),
                "nonnegative_task_count": nonnegative_count,
            },
            relation="nonnegative_task_count >= ceil(2/3 * eligible_task_count)",
            threshold=required_nonnegative_count,
        ),
        "rank_one_task_fraction": _check(
            passed=(
                rank_fraction is not None and rank_fraction >= REPRESENTATION_MINIMUM_TASK_FRACTION
            ),
            actual=rank_fraction,
            relation=">=",
            threshold=REPRESENTATION_MINIMUM_TASK_FRACTION,
        ),
        "row_eligible_task_fraction": _check(
            passed=row_eligible_fraction >= REPRESENTATION_MINIMUM_TASK_FRACTION,
            actual=row_eligible_fraction,
            relation=">=",
            threshold=REPRESENTATION_MINIMUM_TASK_FRACTION,
        ),
        "hardest_negative_logit_margin": _check(
            passed=row_margin is not None and row_margin > 0.0,
            actual=row_margin,
            relation=">",
            threshold=0.0,
        ),
        "ownership_eligible_task_fraction": _check(
            passed=ownership_eligible_fraction >= REPRESENTATION_MINIMUM_TASK_FRACTION,
            actual=ownership_eligible_fraction,
            relation=">=",
            threshold=REPRESENTATION_MINIMUM_TASK_FRACTION,
        ),
        "control_eligible_task_fraction": _check(
            passed=control_eligible_fraction >= REPRESENTATION_MINIMUM_TASK_FRACTION,
            actual=control_eligible_fraction,
            relation=">=",
            threshold=REPRESENTATION_MINIMUM_TASK_FRACTION,
        ),
        "target_ownership_soft_iou": _check(
            passed=(
                decision_ownership is not None
                and required_ownership is not None
                and decision_ownership >= required_ownership
            ),
            actual=decision_ownership,
            relation=">=",
            threshold=required_ownership,
        ),
        "shuffled_task_auc_degradation": _check(
            passed=(
                shuffled_task_degradation is not None
                and shuffled_task_degradation > REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION
            ),
            actual=shuffled_task_degradation,
            relation=">",
            threshold=REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION,
        ),
        "positive_shuffled_task_degradation_fraction": _check(
            passed=positive_shuffled_task_count >= required_positive_control_count,
            actual={
                "control_eligible_task_count": control_eligible_count,
                "positive_task_count": positive_shuffled_task_count,
            },
            relation="positive_task_count >= ceil(2/3 * control_eligible_task_count)",
            threshold=required_positive_control_count,
        ),
        "shuffled_target_auc_degradation": _check(
            passed=(
                shuffled_target_degradation is not None
                and shuffled_target_degradation > REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION
            ),
            actual=shuffled_target_degradation,
            relation=">",
            threshold=REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION,
        ),
        "positive_shuffled_target_degradation_fraction": _check(
            passed=positive_shuffled_target_count >= required_positive_control_count,
            actual={
                "control_eligible_task_count": control_eligible_count,
                "positive_task_count": positive_shuffled_target_count,
            },
            relation="positive_task_count >= ceil(2/3 * control_eligible_task_count)",
            threshold=required_positive_control_count,
        ),
        "frozen_action_state_unchanged": _check(
            passed=action_state_unchanged,
            actual=action_state_unchanged,
            relation="==",
            threshold=True,
        ),
        "official_action_relative_regression": _check(
            passed=(
                action_relative_regression <= REPRESENTATION_MAXIMUM_ACTION_RELATIVE_REGRESSION
            ),
            actual=action_relative_regression_for_artifact,
            relation="<=",
            threshold=REPRESENTATION_MAXIMUM_ACTION_RELATIVE_REGRESSION,
        ),
    }
    numeric_pass = all(bool(check["passed"]) for check in checks.values())
    value: dict[str, object] = {
        "schema": REPRESENTATION_NUMERIC_GATE_SCHEMA,
        "status": "PASS_PENDING_VISUAL_REVIEW" if numeric_pass else "FAIL",
        "baseline_snapshot_sha256": baseline["artifact_sha256"],
        "decision_snapshot_sha256": decision["artifact_sha256"],
        "representation_evaluation_plan_sha256": plan.artifact_sha256,
        "thresholds": {
            "baseline_step": REPRESENTATION_BASELINE_STEP,
            "decision_step": REPRESENTATION_DECISION_STEP,
            "minimum_auc_delta": REPRESENTATION_MINIMUM_AUC_DELTA,
            "minimum_task_fraction": REPRESENTATION_MINIMUM_TASK_FRACTION,
            "minimum_target_ownership_soft_iou": (REPRESENTATION_MINIMUM_TARGET_OWNERSHIP_SOFT_IOU),
            "minimum_ownership_multiplier": REPRESENTATION_MINIMUM_OWNERSHIP_MULTIPLIER,
            "minimum_control_auc_degradation_exclusive": (
                REPRESENTATION_MINIMUM_CONTROL_AUC_DEGRADATION
            ),
            "maximum_action_relative_regression": (
                REPRESENTATION_MAXIMUM_ACTION_RELATIVE_REGRESSION
            ),
        },
        "metrics": {
            "task_auc_deltas": task_deltas,
            "mean_task_auc_delta": mean_auc_delta,
            "nonnegative_task_auc_delta_count": nonnegative_count,
            "eligible_task_auc_delta_count": len(task_deltas),
            "control_eligible_task_count": control_eligible_count,
            "positive_shuffled_task_degradation_count": positive_shuffled_task_count,
            "positive_shuffled_target_degradation_count": positive_shuffled_target_count,
            "baseline_target_ownership_soft_iou": baseline_ownership,
            "decision_target_ownership_soft_iou": decision_ownership,
            "required_target_ownership_soft_iou": required_ownership,
            "baseline_official_action_loss": baseline_action,
            "decision_official_action_loss": decision_action,
            "official_action_relative_regression": action_relative_regression_for_artifact,
        },
        "checks": checks,
        "visual_review_required": True,
        "authorizes_joint_adoption": False,
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return value


def build_representation_numeric_gate(
    baseline_snapshot: Mapping[str, Any],
    decision_snapshot: Mapping[str, Any],
    *,
    plan: RepresentationEvaluationPlan,
) -> dict[str, object]:
    """Evaluate every non-visual ADR-108 section 29.6 criterion."""

    return validate_representation_numeric_gate(
        _representation_numeric_gate_value(
            baseline_snapshot,
            decision_snapshot,
            plan=plan,
        ),
        baseline_snapshot=baseline_snapshot,
        decision_snapshot=decision_snapshot,
        plan=plan,
    )


def validate_representation_numeric_gate(
    value: object,
    *,
    baseline_snapshot: Mapping[str, Any],
    decision_snapshot: Mapping[str, Any],
    plan: RepresentationEvaluationPlan,
) -> dict[str, Any]:
    """Recompute the complete numeric decision from immutable snapshots."""

    if not isinstance(value, dict) or set(value) != _GATE_FIELDS:
        raise ValueError("representation numeric gate fields differ from schema")
    expected = _representation_numeric_gate_value(
        baseline_snapshot,
        decision_snapshot,
        plan=plan,
    )
    if value != expected:
        raise ValueError("representation numeric gate was not recomputed")
    return value


def write_representation_numeric_gate(path: str | Path, value: Mapping[str, object]) -> None:
    """Publish one immutable numeric decision with a durable atomic rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if (
        destination.exists()
        or destination.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(f"representation numeric gate path exists: {destination}")
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
