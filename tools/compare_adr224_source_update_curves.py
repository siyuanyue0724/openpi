#!/usr/bin/env python3
"""Compare matched frozen-coordinate and joint-source ADR-224 curves."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.compare_adr224_wla_host_evidence_curves import (
    _MANIFEST_INVARIANTS,
    _RECORD_IDENTITIES,
    _canonical_sha256,
    _load_heldout_snapshots,
    _load_json,
    _load_rank_journals,
    _nested,
    _positive_loss,
    _window_summary,
    compare_heldout_snapshots,
)

SCHEMA = "picf-next.adr224-source-update-curve-comparison/v1"
FROZEN_ARM = "frozen-coordinate-control"
JOINT_ARM = "joint"
PICF_EVIDENCE_ARM = "picf_full"
ROOT_CAUSE_MINIMUM_RELATIVE_EFFECT = 0.02
FLOAT32_EQUIVALENCE_REL_TOL = 2.0**-22
FLOAT32_EQUIVALENCE_ABS_TOL = 2.0**-24


def _parse_window(value: str) -> tuple[int, int]:
    try:
        start, end = (int(item) for item in value.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("window must be START:END") from error
    if start <= 0 or end < start:
        raise argparse.ArgumentTypeError("window bounds must satisfy 0 < START <= END")
    return start, end


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run-dir", type=Path, required=True)
    parser.add_argument("--joint-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=_parse_window, action="append", default=[])
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def _normalized_contract(
    manifest: Mapping[str, Any], *, expected_update_arm: str
) -> dict[str, Any]:
    contract = manifest.get("execution_contract")
    if not isinstance(contract, dict):
        raise ValueError("run manifest has no execution contract")
    normalized = copy.deepcopy(contract)
    if normalized.get("wla_host_evidence_arm") != PICF_EVIDENCE_ARM:
        raise ValueError("source-update control requires full PICF host evidence")
    stage = normalized.get("videomt_stage_pq")
    if not isinstance(stage, dict):
        raise ValueError("execution contract has no VidEoMT source surface")
    arm = stage.pop("source_update_arm", None)
    if arm != expected_update_arm:
        raise ValueError(f"expected source-update arm {expected_update_arm!r}, found {arm!r}")
    if stage.get("source_forward_and_backward_graph") != (
        "unchanged_complete_joint_graph"
    ):
        raise ValueError("source-update arm changed the registered source graph")
    return normalized


def validate_matched_manifests(
    frozen: Mapping[str, Any], joint: Mapping[str, Any]
) -> dict[str, Any]:
    statuses = (frozen.get("status"), joint.get("status"))
    if statuses[0] != statuses[1] or statuses[0] not in {"DECLARED", "PASS"}:
        raise ValueError("both run manifests must have the same valid declaration status")
    for field in _MANIFEST_INVARIANTS:
        if frozen.get(field) != joint.get(field):
            raise ValueError(f"matched run manifests differ at {field}")
    if frozen.get("early_stop_step") != joint.get("early_stop_step"):
        raise ValueError("matched runs stop at different steps")
    frozen_contract = _normalized_contract(frozen, expected_update_arm=FROZEN_ARM)
    joint_contract = _normalized_contract(joint, expected_update_arm=JOINT_ARM)
    if frozen_contract != joint_contract:
        raise ValueError("execution contracts differ beyond the source-update arm")
    return {
        "implementation_sha256": frozen["implementation_sha256"],
        "model_family_sha256": frozen["model_family_sha256"],
        "stream_plan_sha256": frozen["stream_plan_sha256"],
        "execution_contract_without_intervention_sha256": _canonical_sha256(
            frozen_contract
        ),
    }


def validate_completed_run(
    run_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    early_stop_step = manifest.get("early_stop_step")
    declared_total_steps = manifest.get("declared_total_steps")
    if (
        isinstance(early_stop_step, bool)
        or not isinstance(early_stop_step, int)
        or early_stop_step <= 0
        or isinstance(declared_total_steps, bool)
        or not isinstance(declared_total_steps, int)
        or declared_total_steps < early_stop_step
    ):
        raise ValueError("run manifest has invalid declared completion bounds")
    summary_path = run_dir / f"run_summary_step_{early_stop_step:08d}.json"
    summary = _load_json(summary_path)
    expected_status = (
        "COMPLETE" if early_stop_step == declared_total_steps else "EARLY_STOP"
    )
    expected = {
        "schema": manifest.get("schema"),
        "status": expected_status,
        "completed_global_step": early_stop_step,
        "declared_total_steps": declared_total_steps,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"run completion summary differs at {field}")
    return {
        "status": expected_status,
        "completed_global_step": early_stop_step,
        "summary_sha256": _canonical_sha256(summary),
    }


def _validate_record_pair(
    frozen: Mapping[str, Any], joint: Mapping[str, Any]
) -> None:
    if frozen.get("wla_host_evidence_arm") != PICF_EVIDENCE_ARM or joint.get(
        "wla_host_evidence_arm"
    ) != PICF_EVIDENCE_ARM:
        raise ValueError("source-update records changed the host-evidence arm")
    for field in _RECORD_IDENTITIES:
        if frozen.get(field) != joint.get(field):
            raise ValueError(f"matched journal records differ at {field}")
    for path in (
        ("wla_action_world", "target_source_global_indices"),
        ("wla_action_world", "target_source_rgb_sha256"),
        ("wla_action_world", "world_loss_weight"),
        ("wla_action_world", "optimizer_contract"),
        ("videomt_source_objective", "global_indices"),
        ("videomt_source_objective", "query_count"),
    ):
        if _nested(frozen, *path) != _nested(joint, *path):
            raise ValueError(f"matched journal records differ at {'.'.join(path)}")
    frozen_gradient = frozen.get("gradient_metrics")
    joint_gradient = joint.get("gradient_metrics")
    if not isinstance(frozen_gradient, Mapping) or not isinstance(
        joint_gradient, Mapping
    ):
        raise ValueError("source-update records omit gradient metrics")
    if (
        frozen_gradient.get("source_update_arm") != FROZEN_ARM
        or frozen_gradient.get("source_update_applied") is not False
        or frozen_gradient.get("source_scheduler_step") != 0
    ):
        raise ValueError("frozen-coordinate record applied a source update")
    if (
        joint_gradient.get("source_update_arm") != JOINT_ARM
        or joint_gradient.get("source_update_applied") is not True
        or joint_gradient.get("source_scheduler_step") != joint.get("global_step")
    ):
        raise ValueError("joint record omitted a source update")


def _step_one_equivalence(
    frozen_records: Mapping[tuple[int, int], Mapping[str, Any]],
    joint_records: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    fields: tuple[tuple[str, ...], ...] = (
        ("official_action_loss",),
        ("official_policy_loss",),
        ("objective_total",),
        ("training_objective_total",),
        ("videomt_source_objective",),
        ("wla_action_world", "metrics"),
        ("posterior_bank_sha256",),
    )
    exact_mismatches: list[dict[str, Any]] = []
    numeric_mismatches: list[dict[str, Any]] = []
    maximum_numeric_absolute_difference = 0.0
    rank_count = 0
    for (step, rank), frozen in sorted(frozen_records.items()):
        if step != 1:
            continue
        rank_count += 1
        joint = joint_records[(step, rank)]
        for path in fields:
            frozen_value = _nested(frozen, *path)
            joint_value = _nested(joint, *path)
            if frozen_value != joint_value:
                exact_mismatches.append(
                    {
                        "rank": rank,
                        "field": ".".join(path),
                        "frozen": frozen_value,
                        "joint": joint_value,
                    }
                )
            equivalent, maximum_difference = _float32_tree_equivalent(
                frozen_value, joint_value
            )
            maximum_numeric_absolute_difference = max(
                maximum_numeric_absolute_difference, maximum_difference
            )
            if not equivalent:
                numeric_mismatches.append(
                    {
                        "rank": rank,
                        "field": ".".join(path),
                        "frozen": frozen_value,
                        "joint": joint_value,
                    }
                )
    if rank_count == 0:
        raise ValueError("matched runs contain no step-one records")
    return {
        "rank_count": rank_count,
        "compared_fields": [".".join(path) for path in fields],
        "exact": not exact_mismatches,
        "float32_numerically_equivalent": not numeric_mismatches,
        "float32_equivalence_rel_tol": FLOAT32_EQUIVALENCE_REL_TOL,
        "float32_equivalence_abs_tol": FLOAT32_EQUIVALENCE_ABS_TOL,
        "maximum_numeric_absolute_difference": maximum_numeric_absolute_difference,
        "exact_mismatches": exact_mismatches,
        "numeric_mismatches": numeric_mismatches,
    }


def _float32_tree_equivalent(left: Any, right: Any) -> tuple[bool, float]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            return False, 0.0
        comparisons = [
            _float32_tree_equivalent(left[key], right[key]) for key in left
        ]
        return (
            all(equivalent for equivalent, _ in comparisons),
            max((difference for _, difference in comparisons), default=0.0),
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False, 0.0
        comparisons = [
            _float32_tree_equivalent(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        ]
        return (
            all(equivalent for equivalent, _ in comparisons),
            max((difference for _, difference in comparisons), default=0.0),
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        left_float = float(left)
        right_float = float(right)
        difference = abs(left_float - right_float)
        equivalent = math.isclose(
            left_float,
            right_float,
            rel_tol=FLOAT32_EQUIVALENCE_REL_TOL,
            abs_tol=FLOAT32_EQUIVALENCE_ABS_TOL,
        )
        return equivalent, difference
    return left == right, 0.0


def compare_training_records(
    *,
    frozen_records: Mapping[tuple[int, int], Mapping[str, Any]],
    joint_records: Mapping[tuple[int, int], Mapping[str, Any]],
    windows: Sequence[tuple[int, int]],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if frozen_records.keys() != joint_records.keys() or not frozen_records:
        raise ValueError("matched runs have different rank-step keys")
    rows: list[tuple[int, int, float, float]] = []
    for step_rank in sorted(frozen_records):
        frozen = frozen_records[step_rank]
        joint = joint_records[step_rank]
        _validate_record_pair(frozen, joint)
        rows.append(
            (
                step_rank[0],
                step_rank[1],
                _positive_loss(frozen, "loss_action"),
                _positive_loss(joint, "loss_action"),
            )
        )
    maximum_step = max(step for step, _, _, _ in rows)
    registered_windows = tuple(windows) or ((1, maximum_step),)
    return {
        "candidate_label_for_full_fields": FROZEN_ARM,
        "reference_label_for_masked_fields": JOINT_ARM,
        "maximum_step": maximum_step,
        "step_one_forward_equivalence": _step_one_equivalence(
            frozen_records, joint_records
        ),
        "overall": _window_summary(
            rows,
            start=1,
            end=maximum_step,
            bootstrap_replicates=bootstrap_replicates,
            seed=20261126,
        ),
        "windows": [
            _window_summary(
                rows,
                start=start,
                end=end,
                bootstrap_replicates=bootstrap_replicates,
                seed=20261226 + index * 2,
            )
            for index, (start, end) in enumerate(registered_windows)
        ],
    }


def _decision(training: Mapping[str, Any], heldout: Mapping[str, Any]) -> str:
    equivalence = _nested(training, "step_one_forward_equivalence")
    if not isinstance(equivalence, Mapping):
        return "INVALID_STEP_ONE_FORWARD_MISMATCH"
    if not equivalence.get(
        "float32_numerically_equivalent", equivalence.get("exact", False)
    ):
        return "INVALID_STEP_ONE_FORWARD_MISMATCH"
    comparisons = heldout.get("learning_difference_in_differences")
    if not isinstance(comparisons, list) or not comparisons:
        return "INVALID_NO_FIXED_SET_LEARNING_COMPARISON"
    overall = comparisons[-1].get("overall")
    if not isinstance(overall, Mapping):
        return "INVALID_NO_FIXED_SET_OVERALL_COMPARISON"
    effect = float(overall.get("learning_ratio_of_ratios_delta", math.nan))
    interval = overall.get("learning_ratio_of_ratios_delta_episode_bootstrap_95")
    training_summaries = [training.get("overall"), *training.get("windows", [])]
    if not training_summaries or any(
        not isinstance(summary, Mapping) for summary in training_summaries
    ):
        return "INVALID_NO_REGISTERED_TRAINING_WINDOWS"
    training_deltas = [
        float(summary.get("paired_geometric_relative_delta", math.nan))
        for summary in training_summaries
    ]
    if (
        not math.isfinite(effect)
        or not isinstance(interval, list)
        or len(interval) != 2
        or any(not math.isfinite(float(value)) for value in interval)
        or any(not math.isfinite(value) for value in training_deltas)
    ):
        return "INVALID_NONFINITE_FIXED_SET_EFFECT"
    if (
        effect <= -ROOT_CAUSE_MINIMUM_RELATIVE_EFFECT
        and float(interval[1]) < 0.0
        and all(
            delta < ROOT_CAUSE_MINIMUM_RELATIVE_EFFECT
            for delta in training_deltas
        )
    ):
        return "SUPPORTS_COORDINATE_MOTION_AS_PRIMARY_EARLY_ROOT_CAUSE"
    return "REJECTS_COORDINATE_MOTION_AS_PRIMARY_EARLY_ROOT_CAUSE"


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    frozen_manifest = _load_json(args.frozen_run_dir / "run_manifest.json")
    joint_manifest = _load_json(args.joint_run_dir / "run_manifest.json")
    contract = validate_matched_manifests(frozen_manifest, joint_manifest)
    completion = {
        "frozen": validate_completed_run(args.frozen_run_dir, frozen_manifest),
        "joint": validate_completed_run(args.joint_run_dir, joint_manifest),
    }
    world_size = int(frozen_manifest["world_size"])
    frozen_records = _load_rank_journals(
        args.frozen_run_dir, expected_world_size=world_size
    )
    joint_records = _load_rank_journals(
        args.joint_run_dir, expected_world_size=world_size
    )
    training = compare_training_records(
        frozen_records=frozen_records,
        joint_records=joint_records,
        windows=args.window,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    heldout = compare_heldout_snapshots(
        full_snapshots=_load_heldout_snapshots(args.frozen_run_dir),
        masked_snapshots=_load_heldout_snapshots(args.joint_run_dir),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "decision": _decision(training, heldout),
        "intervention": {
            "candidate": FROZEN_ARM,
            "reference": JOINT_ARM,
            "only_changed_operation": "apply_source_optimizer_and_scheduler_step",
        },
        "root_cause_minimum_relative_effect": ROOT_CAUSE_MINIMUM_RELATIVE_EFFECT,
        "authorizes_action_lead": False,
        "action_gates_closed": 0,
        "contract": contract,
        "completion": completion,
        "training": training,
        "fixed_heldout": heldout,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
