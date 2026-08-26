#!/usr/bin/env python3
"""Validate and summarize the strict ADR-209 FLARE lambda=1/lambda=0 pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

REPORT_SCHEMA = "picf-next.adr209-flare-strict-pair-comparison/v1"
CANDIDATE_PROFILE = "adr209_native_videomt_flare_v1"
CONTROL_PROFILE = "adr209_native_videomt_query_control_t16_v1"
CANDIDATE_OBJECTIVE = "adr209_complete_source_native_query_flare_joint_action"
CONTROL_OBJECTIVE = "adr209_complete_source_native_query_t16_joint_action_control"
ALLOWED_MANIFEST_DIFFERENCES = frozenset(
    {
        "execution_contract.future_latent_objective_scale",
        "execution_contract.objective_profile",
        "execution_contract.picf_architecture_profile",
        "execution_contract_sha256",
    }
)
STREAM_FIELDS = (
    "sample_keys",
    "augmentation_seeds",
    "flow_noise_seeds",
    "flow_timestep_seeds",
    "source_digest",
    "temporal_plan_sha256",
    "frame_indices",
    "lane_ids",
    "reset",
)
ACTION_INPUT_FIELDS = (
    "sample_key",
    "partition",
    "source_digest",
    "model_inputs_sha256",
    "native_source_rgb_sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--control-run-dir", type=Path, required=True)
    parser.add_argument("--terminal-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required direct artifact is absent: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not an object: {path}")
    return value


def _validate_semantic_sha256(artifact: Mapping[str, Any], *, path: Path) -> None:
    semantic = dict(artifact)
    observed = semantic.pop("artifact_sha256", None)
    if observed is not None and observed != _canonical_sha256(semantic):
        raise ValueError(f"semantic SHA-256 differs: {path}")


def _leaf_differences(left: object, right: object, *, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_leaf_differences(left[key], right[key], path=child))
        return differences
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if len(left) != len(right):
            return [path]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.extend(
                _leaf_differences(left_item, right_item, path=f"{path}[{index}]")
            )
        return differences
    return [] if left == right else [path]


def _validate_manifests(
    candidate_path: Path,
    control_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    candidate = _load_json(candidate_path)
    control = _load_json(control_path)
    for manifest, expected_scale, expected_profile, expected_objective, label in (
        (candidate, 1.0, CANDIDATE_PROFILE, CANDIDATE_OBJECTIVE, "candidate"),
        (control, 0.0, CONTROL_PROFILE, CONTROL_OBJECTIVE, "control"),
    ):
        contract = manifest.get("execution_contract")
        if not isinstance(contract, dict):
            raise ValueError(f"{label} execution contract is absent")
        expected = {
            "future_latent_objective_scale": expected_scale,
            "picf_architecture_profile": expected_profile,
            "objective_profile": expected_objective,
        }
        for field, value in expected.items():
            if contract.get(field) != value:
                raise ValueError(f"{label} execution contract differs for {field}")
        if manifest.get("execution_contract_sha256") != _canonical_sha256(contract):
            raise ValueError(f"{label} execution-contract SHA-256 differs")
    if candidate.get("implementation_sha256") != control.get("implementation_sha256"):
        raise ValueError("candidate and control use different implementations")
    differences = _leaf_differences(candidate, control)
    if frozenset(differences) != ALLOWED_MANIFEST_DIFFERENCES:
        raise ValueError(f"unexpected manifest differences: {differences}")
    return candidate, control, differences


def _load_journals(run_dir: Path, *, terminal_step: int) -> dict[int, list[dict[str, Any]]]:
    journal_dir = run_dir / "metrics" / "rank_journal"
    paths = sorted(journal_dir.glob("rank_*.jsonl"))
    if not paths:
        raise ValueError(f"no rank journals: {journal_dir}")
    result: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        rank = int(path.stem.removeprefix("rank_"))
        rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
        if len(rows) != terminal_step:
            raise ValueError(
                f"{path} has {len(rows)} rows, expected {terminal_step}"
            )
        if [row.get("global_step") for row in rows] != list(
            range(1, terminal_step + 1)
        ):
            raise ValueError(f"{path} has a non-contiguous global-step sequence")
        result[rank] = rows
    return result


def _finite_float(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _validate_and_summarize_journals(
    candidate: dict[int, list[dict[str, Any]]],
    control: dict[int, list[dict[str, Any]]],
    *,
    terminal_step: int,
) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    if candidate.keys() != control.keys():
        raise ValueError("candidate and control rank sets differ")
    curve: list[dict[str, float | int]] = []
    for step_index in range(terminal_step):
        candidate_rows = [candidate[rank][step_index] for rank in sorted(candidate)]
        control_rows = [control[rank][step_index] for rank in sorted(control)]
        for rank, (candidate_row, control_row) in enumerate(
            zip(candidate_rows, control_rows, strict=True)
        ):
            for field in STREAM_FIELDS:
                if candidate_row.get(field) != control_row.get(field):
                    raise ValueError(
                        f"rank {rank} step {step_index + 1} stream differs for {field}"
                    )
            alignments: dict[str, dict[str, Any]] = {}
            for row, expected_scale, label in (
                (candidate_row, 1.0, "candidate"),
                (control_row, 0.0, "control"),
            ):
                alignment = row.get("future_latent_alignment")
                if not isinstance(alignment, dict):
                    raise ValueError(f"{label} future alignment is absent")
                alignments[label] = alignment
                expected = {
                    "action_layer_count": 36,
                    "capture_layer_index": 26,
                    "future_token_count": 128,
                    "objective_scale": expected_scale,
                }
                for field, value in expected.items():
                    if alignment.get(field) != value:
                        raise ValueError(
                            f"{label} step {step_index + 1} alignment differs for {field}"
                        )
                contribution = _finite_float(
                    alignment.get("objective_contribution"),
                    name=f"{label} objective contribution",
                )
                weighted = _finite_float(
                    alignment.get("weighted_loss"),
                    name=f"{label} weighted loss",
                )
                raw = _finite_float(
                    alignment.get("raw_loss"),
                    name=f"{label} raw loss",
                )
                cosine = _finite_float(
                    alignment.get("mean_cosine"),
                    name=f"{label} mean cosine",
                )
                if not math.isclose(weighted, 0.2 * raw, rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError(f"{label} FLARE internal weighting differs")
                if not math.isclose(raw, 1.0 - cosine, rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError(f"{label} FLARE cosine loss identity differs")
                expected_contribution = expected_scale * weighted
                if not math.isclose(
                    contribution,
                    expected_contribution,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        f"{label} step {step_index + 1} objective contribution differs"
                    )
                gradients = row.get("gradient_metrics")
                if not isinstance(gradients, dict):
                    raise ValueError(f"{label} gradient receipt is absent")
                if gradients.get("host_all_finite") is not True:
                    raise ValueError(f"{label} host gradients are non-finite")
                if gradients.get("source_all_finite_and_present") is not True:
                    raise ValueError(f"{label} source gradients are invalid")
            if (
                alignments["candidate"].get("target_manifest_sha256")
                != alignments["control"].get("target_manifest_sha256")
            ):
                raise ValueError(
                    f"rank {rank} step {step_index + 1} target manifest differs"
                )
            if step_index == 0:
                for field in ("official_action_loss", "official_moe_regularizer"):
                    if candidate_row.get(field) != control_row.get(field):
                        raise ValueError(f"initial training value differs for {field}")
                if (
                    candidate_row["videomt_source_objective"]["total"]
                    != control_row["videomt_source_objective"]["total"]
                ):
                    raise ValueError("initial VidEoMT source objective differs")
                for field in ("raw_loss", "weighted_loss", "mean_cosine"):
                    if alignments["candidate"].get(field) != alignments["control"].get(
                        field
                    ):
                        raise ValueError(f"initial FLARE value differs for {field}")
                objective_delta = _finite_float(
                    candidate_row.get("objective_total"),
                    name="candidate objective total",
                ) - _finite_float(
                    control_row.get("objective_total"),
                    name="control objective total",
                )
                if not math.isclose(
                    objective_delta,
                    float(alignments["candidate"]["objective_contribution"]),
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                ):
                    raise ValueError("initial objective delta is not the FLARE contribution")

        def mean(rows: list[dict[str, Any]], field: str) -> float:
            return sum(_finite_float(row[field], name=field) for row in rows) / len(rows)

        def alignment_mean(rows: list[dict[str, Any]], field: str) -> float:
            return sum(
                _finite_float(row["future_latent_alignment"][field], name=field)
                for row in rows
            ) / len(rows)

        def source_mean(rows: list[dict[str, Any]]) -> float:
            return sum(
                _finite_float(row["videomt_source_objective"]["total"], name="source total")
                for row in rows
            ) / len(rows)

        curve.append(
            {
                "global_step": step_index + 1,
                "candidate_action_loss": mean(candidate_rows, "official_action_loss"),
                "control_action_loss": mean(control_rows, "official_action_loss"),
                "candidate_flare_raw_loss": alignment_mean(candidate_rows, "raw_loss"),
                "control_flare_raw_loss": alignment_mean(control_rows, "raw_loss"),
                "candidate_flare_cosine": alignment_mean(candidate_rows, "mean_cosine"),
                "control_flare_cosine": alignment_mean(control_rows, "mean_cosine"),
                "candidate_source_loss": source_mean(candidate_rows),
                "control_source_loss": source_mean(control_rows),
                "candidate_peak_reserved_gib": max(
                    _finite_float(row["peak_cuda_reserved_bytes"], name="peak reserved")
                    / 1024**3
                    for row in candidate_rows
                ),
                "control_peak_reserved_gib": max(
                    _finite_float(row["peak_cuda_reserved_bytes"], name="peak reserved")
                    / 1024**3
                    for row in control_rows
                ),
            }
        )

    def average(field: str) -> float:
        return sum(float(row[field]) for row in curve) / len(curve)

    def window(field: str, *, first: bool) -> float:
        rows = curve[:5] if first else curve[-5:]
        return sum(float(row[field]) for row in rows) / len(rows)

    summary = {
        "candidate_action_auc": average("candidate_action_loss"),
        "control_action_auc": average("control_action_loss"),
        "action_auc_delta_candidate_minus_control": average("candidate_action_loss")
        - average("control_action_loss"),
        "candidate_action_first5": window("candidate_action_loss", first=True),
        "control_action_first5": window("control_action_loss", first=True),
        "candidate_action_last5": window("candidate_action_loss", first=False),
        "control_action_last5": window("control_action_loss", first=False),
        "action_last5_delta_candidate_minus_control": window(
            "candidate_action_loss", first=False
        )
        - window("control_action_loss", first=False),
        "candidate_flare_raw_auc": average("candidate_flare_raw_loss"),
        "control_flare_raw_auc": average("control_flare_raw_loss"),
        "candidate_flare_final": float(curve[-1]["candidate_flare_raw_loss"]),
        "control_flare_final": float(curve[-1]["control_flare_raw_loss"]),
        "candidate_source_auc": average("candidate_source_loss"),
        "control_source_auc": average("control_source_loss"),
        "candidate_peak_reserved_gib": max(
            float(row["candidate_peak_reserved_gib"]) for row in curve
        ),
        "control_peak_reserved_gib": max(
            float(row["control_peak_reserved_gib"]) for row in curve
        ),
    }
    return curve, summary


def _sample_index(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    samples = snapshot.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("action snapshot has no samples")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("action sample is not an object")
        key = (str(sample.get("partition")), str(sample.get("sample_key")))
        if key in result:
            raise ValueError(f"duplicate action sample: {key}")
        result[key] = sample
    return result


def _validate_snapshot_pair(
    candidate_path: Path,
    control_path: Path,
    *,
    step: int,
    require_equal_action: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, float | int]]]:
    candidate = _load_json(candidate_path)
    control = _load_json(control_path)
    for snapshot, path in ((candidate, candidate_path), (control, control_path)):
        _validate_semantic_sha256(snapshot, path=path)
        if snapshot.get("status") != "PASS" or snapshot.get("checkpoint_global_step") != step:
            raise ValueError(f"action snapshot did not pass at step {step}: {path}")
    if candidate.get("evaluation_input_sha256") != control.get("evaluation_input_sha256"):
        raise ValueError(f"candidate/control evaluation input differs at step {step}")
    candidate_samples = _sample_index(candidate)
    control_samples = _sample_index(control)
    if candidate_samples.keys() != control_samples.keys():
        raise ValueError(f"candidate/control action sample sets differ at step {step}")
    partition_values: dict[str, dict[str, list[float]]] = {}
    for key in sorted(candidate_samples):
        candidate_sample = candidate_samples[key]
        control_sample = control_samples[key]
        for field in ACTION_INPUT_FIELDS:
            if candidate_sample.get(field) != control_sample.get(field):
                raise ValueError(f"action input differs for {key} field {field}")
        candidate_loss = _finite_float(candidate_sample.get("action_loss"), name="action loss")
        control_loss = _finite_float(control_sample.get("action_loss"), name="action loss")
        if require_equal_action and candidate_loss != control_loss:
            raise ValueError(f"initial action loss differs for {key}")
        values = partition_values.setdefault(
            key[0],
            {"candidate": [], "control": [], "delta": []},
        )
        values["candidate"].append(candidate_loss)
        values["control"].append(control_loss)
        values["delta"].append(candidate_loss - control_loss)
    summaries: dict[str, dict[str, float | int]] = {}
    for partition, values in sorted(partition_values.items()):
        deltas = values["delta"]
        candidate_mean = _finite_float(
            candidate["partition_summaries"][partition]["mean_action_loss"],
            name="candidate partition action loss",
        )
        control_mean = _finite_float(
            control["partition_summaries"][partition]["mean_action_loss"],
            name="control partition action loss",
        )
        sample_candidate_mean = sum(values["candidate"]) / len(deltas)
        sample_control_mean = sum(values["control"]) / len(deltas)
        if not math.isclose(candidate_mean, sample_candidate_mean, abs_tol=1e-12):
            raise ValueError(f"candidate {partition} action summary differs from samples")
        if not math.isclose(control_mean, sample_control_mean, abs_tol=1e-12):
            raise ValueError(f"control {partition} action summary differs from samples")
        summaries[partition] = {
            "sample_count": len(deltas),
            "candidate_mean_action_loss": candidate_mean,
            "control_mean_action_loss": control_mean,
            "mean_paired_delta_candidate_minus_control": sum(deltas) / len(deltas),
            "median_paired_delta_candidate_minus_control": median(deltas),
            "relative_delta": (candidate_mean - control_mean) / control_mean,
            "candidate_better_count": sum(delta < 0.0 for delta in deltas),
            "equal_count": sum(delta == 0.0 for delta in deltas),
            "candidate_worse_count": sum(delta > 0.0 for delta in deltas),
        }
    return candidate, control, summaries


def _anchor_summary(path: Path, *, step: int) -> dict[str, Any]:
    snapshot = _load_json(path)
    _validate_semantic_sha256(snapshot, path=path)
    if snapshot.get("status") != "PASS" or snapshot.get("checkpoint_global_step") != step:
        raise ValueError(f"anchor snapshot did not pass at step {step}: {path}")
    result: dict[str, Any] = {
        "evaluation_input_sha256": snapshot.get("evaluation_input_sha256"),
        "partitions": {},
    }
    for partition, value in sorted(snapshot["partition_summaries"].items()):
        ranked = value["ranked_proposals"]["10"]
        result["partitions"][partition] = {
            "oracle_mean_binary_iou": _finite_float(
                value["mean_binary_iou"], name="oracle binary IoU"
            ),
            "oracle_recall_at_50": _finite_float(
                value["recall_at_50"], name="oracle recall"
            ),
            "top10_mean_binary_iou": _finite_float(
                ranked["mean_binary_iou"], name="top-10 binary IoU"
            ),
            "top10_recall_at_50": _finite_float(
                ranked["recall_at_50"], name="top-10 recall"
            ),
        }
    return result


def compare_strict_pair(
    candidate_run_dir: Path,
    control_run_dir: Path,
    *,
    terminal_step: int,
) -> dict[str, Any]:
    if terminal_step < 5:
        raise ValueError("terminal_step must be at least 5")
    candidate_manifest, control_manifest, manifest_differences = _validate_manifests(
        candidate_run_dir / "run_manifest.json",
        control_run_dir / "run_manifest.json",
    )
    candidate_journals = _load_journals(candidate_run_dir, terminal_step=terminal_step)
    control_journals = _load_journals(control_run_dir, terminal_step=terminal_step)
    curve, training_summary = _validate_and_summarize_journals(
        candidate_journals,
        control_journals,
        terminal_step=terminal_step,
    )
    action_evaluations: dict[str, Any] = {}
    for step in (0, terminal_step):
        candidate_snapshot, control_snapshot, summaries = _validate_snapshot_pair(
            candidate_run_dir
            / "action_evaluations"
            / f"step_{step:08d}"
            / "distributed.json",
            control_run_dir
            / "action_evaluations"
            / f"step_{step:08d}"
            / "distributed.json",
            step=step,
            require_equal_action=step == 0,
        )
        action_evaluations[str(step)] = {
            "evaluation_input_sha256": candidate_snapshot["evaluation_input_sha256"],
            "candidate_artifact_sha256": candidate_snapshot["artifact_sha256"],
            "control_artifact_sha256": control_snapshot["artifact_sha256"],
            "partitions": summaries,
        }
    anchor_evaluations: dict[str, Any] = {}
    for step in (0, terminal_step):
        candidate_anchor = _anchor_summary(
            candidate_run_dir
            / "heldout_native_videomt_anchor_evaluations"
            / f"step_{step:08d}"
            / "distributed.json",
            step=step,
        )
        control_anchor = _anchor_summary(
            control_run_dir
            / "heldout_native_videomt_anchor_evaluations"
            / f"step_{step:08d}"
            / "distributed.json",
            step=step,
        )
        if candidate_anchor["evaluation_input_sha256"] != control_anchor["evaluation_input_sha256"]:
            raise ValueError(f"candidate/control anchor input differs at step {step}")
        anchor_evaluations[str(step)] = {
            "candidate": candidate_anchor["partitions"],
            "control": control_anchor["partitions"],
        }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "terminal_step": terminal_step,
        "candidate_run_dir": str(candidate_run_dir.resolve()),
        "control_run_dir": str(control_run_dir.resolve()),
        "implementation_sha256": candidate_manifest["implementation_sha256"],
        "candidate_execution_contract_sha256": candidate_manifest[
            "execution_contract_sha256"
        ],
        "control_execution_contract_sha256": control_manifest[
            "execution_contract_sha256"
        ],
        "manifest_differences": manifest_differences,
        "matched_stream_fields": list(STREAM_FIELDS),
        "training_curve": curve,
        "training_summary": training_summary,
        "action_evaluations": action_evaluations,
        "anchor_evaluations": anchor_evaluations,
        "metric_direction": {
            "action_delta_candidate_minus_control": "negative_favors_candidate",
            "flare_raw_loss": "lower_is_better",
            "anchor_iou_and_recall": "higher_is_better",
        },
        "scientific_decision": "UNSET_REQUIRES_INTERPRETATION",
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    args = _parse_args()
    report = compare_strict_pair(
        args.candidate_run_dir,
        args.control_run_dir,
        terminal_step=args.terminal_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(args.output)


if __name__ == "__main__":
    main()
