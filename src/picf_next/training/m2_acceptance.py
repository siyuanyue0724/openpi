"""Validate the accepted axis-calibrated M2 boundary before temporal training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from picf_next.training.stage_checkpoints import sha256_file

M2_AXIS_PROTOCOL = {
    "checkpoint_selection": "fixed-current-frame-best-no-reselection",
    "fit_data": "train-fixed-match-residuals-only",
    "fit_objective": "axiswise-diagonal-gaussian-nll-with-declared-target-variance",
    "fitted_parameter_names": ["discovery.variance_head.bias"],
    "frozen_parameter_names": ["discovery.variance_head.weight"],
    "matching_mean_and_representation": "frozen-exactly",
    "validation_and_heldout": "evaluation-only-never-fit",
    "variance_dependency": "axis-only-task-identity-and-query-independent",
}
M2_AXIS_ACCEPTANCE_CHECKS = {
    "heldout_error_to_variance_ratio_in_bounds",
    "heldout_nll_below_reset",
    "legacy_variance_weight_zero",
    "nonvariance_state_exact",
    "softplus_roundtrip_within_tolerance",
    "train_nll_not_above_reset",
    "validation_nll_below_reset",
}
M2_AXIS_INPUT_ARTIFACTS = {
    "checkpoints/current_frame_best.pt",
    "config",
    "evaluation_report.json",
    "feature_cache/manifest.json",
    "launch_manifest.json",
    "residual_permutation_probe/report.json",
    "training_report.json",
}


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_axis_calibrated_m2(
    *,
    report_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Return a hash binding only for the exact accepted current-frame model."""

    report_file = Path(report_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    report = _read_json(report_file, "PICF M2 calibration report")
    if report.get("schema") != "picf-next.molmoact2-m2-axis-variance-calibration.v1":
        raise ValueError("PICF M2 calibration report schema changed")
    if report.get("status") != "CALIBRATED_CANDIDATE":
        raise ValueError("PICF M2 calibration candidate was not accepted")
    protocol = report.get("protocol")
    if protocol != M2_AXIS_PROTOCOL:
        raise ValueError("PICF M2 calibration protocol is not the axis-constant contract")
    decision = report.get("decision")
    if (
        not isinstance(decision, dict)
        or decision.get("status") != "PASS"
        or decision.get("later_gates_authorized") != ["M3_bounded_mechanism_smoke"]
        or decision.get("long_training_authorized") is not False
    ):
        raise ValueError("PICF M2 calibration decision does not authorize bounded M3")
    checks = decision.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != M2_AXIS_ACCEPTANCE_CHECKS
        or not all(value is True for value in checks.values())
        or decision.get("failed_checks") != []
    ):
        raise ValueError("PICF M2 calibration report contains a failed acceptance check")
    data_isolation = report.get("data_isolation")
    evaluation_rows = (
        data_isolation.get("evaluation_only_rows") if isinstance(data_isolation, dict) else None
    )
    if (
        not isinstance(data_isolation, dict)
        or data_isolation.get("fit_split") != "train"
        or not isinstance(data_isolation.get("fit_rows"), int)
        or isinstance(data_isolation.get("fit_rows"), bool)
        or data_isolation["fit_rows"] <= 0
        or not isinstance(evaluation_rows, dict)
        or set(evaluation_rows) != {"validation", "heldout"}
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in evaluation_rows.values()
        )
    ):
        raise ValueError("PICF M2 calibration data isolation is incomplete")
    state_isolation = report.get("state_isolation")
    if (
        not isinstance(state_isolation, dict)
        or state_isolation.get("nonvariance_state_exact") is not True
        or state_isolation.get("legacy_variance_weight_zero") is not True
        or any(
            not _is_sha256(state_isolation.get(name))
            for name in (
                "initial_nonvariance_state_sha256",
                "post_extraction_nonvariance_state_sha256",
                "final_nonvariance_state_sha256",
            )
        )
    ):
        raise ValueError("PICF M2 calibration did not preserve nonvariance state")
    input_hashes = report.get("input_sha256")
    if (
        not isinstance(input_hashes, dict)
        or set(input_hashes) != M2_AXIS_INPUT_ARTIFACTS
        or any(not _is_sha256(value) for value in input_hashes.values())
    ):
        raise ValueError("PICF M2 calibration input provenance is incomplete")
    output_hashes = report.get("output_sha256")
    if (
        not isinstance(output_hashes, dict)
        or set(output_hashes) != {"current_frame_axis_calibrated.pt", "metrics.json"}
        or any(not _is_sha256(value) for value in output_hashes.values())
    ):
        raise ValueError("PICF M2 calibration report omitted output hashes")
    expected_checkpoint = output_hashes["current_frame_axis_calibrated.pt"]
    metrics = report_file.parent / "metrics.json"
    if (
        report_file.name != "report.json"
        or checkpoint.name != "current_frame_axis_calibrated.pt"
        or checkpoint.parent != report_file.parent
        or not checkpoint.is_file()
        or sha256_file(checkpoint) != expected_checkpoint
        or not metrics.is_file()
        or sha256_file(metrics) != output_hashes["metrics.json"]
    ):
        raise ValueError("PICF M2 calibration checkpoint is absent or changed")
    return {
        "report_sha256": sha256_file(report_file),
        "checkpoint_sha256": expected_checkpoint,
        "metrics_sha256": output_hashes["metrics.json"],
        "feature_cache_manifest_sha256": input_hashes["feature_cache/manifest.json"],
        "protocol": protocol,
        "state_isolation": state_isolation,
    }
