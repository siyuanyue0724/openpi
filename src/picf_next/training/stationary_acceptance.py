"""Hash-bound acceptance boundary from stationary estimation to action learning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from picf_next.eval.stationary_lifecycle import (
    STATIONARY_LIFECYCLE_CALIBRATION_PASS,
    validate_stationary_lifecycle_calibration,
)
from picf_next.eval.stationary_replay import (
    STATIONARY_FIXED_REPLAY_PASS,
    validate_stationary_fixed_replay,
)
from picf_next.eval.stationary_runtime import (
    STATIONARY_RUNTIME_PROBE_PASS,
    validate_stationary_runtime_probe,
)
from picf_next.eval.stationary_visual import (
    validate_stationary_visual_artifacts,
    validate_stationary_visual_review,
)
from picf_next.training.stage_checkpoints import (
    StationaryTemporalCheckpointProvenance,
    inspect_stationary_temporal_checkpoint,
    sha256_file,
)

STATIONARY_TEMPORAL_ACCEPTANCE_SCHEMA = "picf-next.stationary-temporal-acceptance.v1"
STATIONARY_TEMPORAL_ACCEPTANCE_STATUS = "ACCEPTED_FOR_M4_ACTION_ADOPTION"
STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT = "stationary_temporal_core_accepted.pt"

_EVIDENCE_FILES = {
    "candidate_metrics.jsonl",
    "candidate_report.json",
    "fixed_checkpoint_replay.json",
    "lifecycle_calibration.json",
    "runtime_probe.json",
    "visual_artifacts.json",
    "visual_review.json",
}
_DECISION_CHECKS = {
    "candidate_metrics_detection_support_validated",
    "candidate_report_validated",
    "fixed_checkpoint_replay_passed",
    "full_stationary_checkpoint_hash_bound",
    "lifecycle_calibration_passed",
    "no_recurrent_state_serialized",
    "runtime_probe_passed",
    "visual_review_passed",
}
_DETECTION_METRICS = (
    "picf_lifecycle_detection_positive_target_mass",
    "picf_lifecycle_detection_negative_target_mass",
)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _exact_mapping(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from its frozen schema")
    return cast(dict[str, Any], value)


def validate_stationary_candidate_metrics(
    path: str | Path,
    *,
    expected_steps: int,
) -> dict[int, dict[str, float]]:
    """Require positive and negative detection support in every prefix bucket."""

    metrics_path = Path(path).expanduser()
    if metrics_path.is_symlink() or not metrics_path.is_file():
        raise ValueError("stationary candidate metrics must be one regular file")
    if (
        not isinstance(expected_steps, int)
        or isinstance(expected_steps, bool)
        or expected_steps <= 0
    ):
        raise ValueError("stationary candidate metric step count is invalid")
    try:
        lines = metrics_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("stationary candidate metrics are not ASCII JSONL") from error
    if len(lines) != expected_steps or any(not line for line in lines):
        raise ValueError("stationary candidate metrics do not cover every optimizer step")

    totals = {prefix: {"positive": 0.0, "negative": 0.0} for prefix in (0, 8, 32, 128)}
    for expected_step, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("stationary candidate metrics contain invalid JSON") from error
        payload = _exact_mapping(
            record,
            "stationary candidate metric record",
            {"optimizer_step", "metrics"},
        )
        metrics = payload["metrics"]
        if payload["optimizer_step"] != expected_step or not isinstance(metrics, dict):
            raise ValueError("stationary candidate metric steps are not contiguous")
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for name, value in metrics.items()
        ):
            raise ValueError("stationary candidate metric values are malformed")
        required = {
            "prefix_length",
            "picf_lifecycle_survival_positive_target_mass",
            "picf_lifecycle_survival_negative_target_mass",
            *_DETECTION_METRICS,
        }
        if not required.issubset(metrics):
            raise ValueError("stationary candidate metrics omit lifecycle target support")
        prefix = metrics["prefix_length"]
        if prefix not in totals or isinstance(prefix, bool) or not isinstance(prefix, int):
            raise ValueError("stationary candidate metric prefix is invalid")
        positive = float(metrics[_DETECTION_METRICS[0]])
        negative = float(metrics[_DETECTION_METRICS[1]])
        survival_positive = float(metrics["picf_lifecycle_survival_positive_target_mass"])
        survival_negative = float(metrics["picf_lifecycle_survival_negative_target_mass"])
        if (
            min(positive, negative, survival_positive, survival_negative) < 0.0
            or positive + negative <= 0.0
            or survival_positive + survival_negative <= 0.0
        ):
            raise ValueError("stationary candidate lifecycle target support is invalid")
        totals[prefix]["positive"] += positive
        totals[prefix]["negative"] += negative

    if any(values["positive"] <= 0.0 or values["negative"] <= 0.0 for values in totals.values()):
        raise ValueError(
            "stationary candidate lacks positive/negative detection support in every prefix bucket"
        )
    return totals


@dataclass(frozen=True, slots=True)
class AcceptedStationaryTemporalCore:
    """One immutable Stage-B artifact authorized only for bounded Stage C."""

    checkpoint_path: Path
    checkpoint_sha256: str
    report_sha256: str
    candidate_report_sha256: str
    candidate_metrics_sha256: str
    evidence_sha256: dict[str, str]
    provenance: StationaryTemporalCheckpointProvenance

    def contract_dict(self) -> dict[str, object]:
        return {
            "acceptance_report_sha256": self.report_sha256,
            "candidate_metrics_sha256": self.candidate_metrics_sha256,
            "candidate_report_sha256": self.candidate_report_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "evidence_sha256": dict(sorted(self.evidence_sha256.items())),
            "provenance": self.provenance.to_dict(),
            "stage_authorized": "M4_action_adoption",
        }


def validate_stationary_temporal_acceptance(
    *,
    report_path: str | Path,
    checkpoint_path: str | Path,
) -> AcceptedStationaryTemporalCore:
    """Validate the complete fixed-checkpoint evidence package for Stage C.

    This boundary deliberately does not infer acceptance from a low training
    loss. It requires a separately published decision that hash-binds the
    candidate, fixed replay, lifecycle, visual and runtime evidence.
    """

    report_input = Path(report_path).expanduser()
    checkpoint_input = Path(checkpoint_path).expanduser()
    if report_input.is_symlink() or checkpoint_input.is_symlink():
        raise ValueError("stationary acceptance artifacts cannot be symbolic links")
    report = report_input.resolve()
    checkpoint = checkpoint_input.resolve()
    if (
        report.name != "report.json"
        or checkpoint.name != STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT
        or report.parent != checkpoint.parent
        or not report.is_file()
        or not checkpoint.is_file()
    ):
        raise ValueError("stationary acceptance report/checkpoint package layout changed")

    payload = _exact_mapping(
        _read_json(report, "stationary temporal acceptance report"),
        "stationary temporal acceptance report",
        {"schema", "status", "provenance", "artifacts_sha256", "decision"},
    )
    if payload["schema"] != STATIONARY_TEMPORAL_ACCEPTANCE_SCHEMA:
        raise ValueError("stationary temporal acceptance schema changed")
    if payload["status"] != STATIONARY_TEMPORAL_ACCEPTANCE_STATUS:
        raise ValueError("stationary temporal core was not accepted for bounded Stage C")

    artifacts = _exact_mapping(
        payload["artifacts_sha256"],
        "stationary temporal acceptance artifacts",
        _EVIDENCE_FILES | {STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT},
    )
    artifact_hashes = {
        name: _sha256(value, f"acceptance artifact {name}") for name, value in artifacts.items()
    }
    for name, expected_sha256 in artifact_hashes.items():
        path = report.parent / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"stationary acceptance artifact is absent or changed: {name}")

    checkpoint_sha256 = artifact_hashes[STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT]
    checkpoint_provenance = inspect_stationary_temporal_checkpoint(
        checkpoint,
        expected_sha256=checkpoint_sha256,
    )
    provenance = StationaryTemporalCheckpointProvenance.from_dict(payload["provenance"])
    if provenance != checkpoint_provenance:
        raise ValueError("acceptance provenance differs from the checkpoint payload")
    candidate_report = _exact_mapping(
        _read_json(report.parent / "candidate_report.json", "Stage-B candidate report"),
        "Stage-B candidate report",
        {
            "schema",
            "status",
            "stage_recipe_sha256",
            "source_coverage_recipe_sha256",
            "foundation_recipe_sha256",
            "structural_recipe_sha256",
            "clip_plan_sha256",
            "optimizer_steps",
            "world_size",
            "prefix_lengths",
            "train_length",
            "required_future_horizon",
            "action_weight",
            "checkpoint_sha256",
            "metrics_sha256",
            "completed_optimizer_steps",
            "long_training_authorized",
        },
    )
    _sha256(candidate_report["structural_recipe_sha256"], "structural recipe SHA-256")
    if (
        candidate_report["schema"] != "picf-next.stationary-temporal-candidate-report.v1"
        or candidate_report["status"] != "CANDIDATE_REQUIRES_FIXED_CHECKPOINT_AUDIT"
        or candidate_report["checkpoint_sha256"] != checkpoint_sha256
        or candidate_report["metrics_sha256"] != artifact_hashes["candidate_metrics.jsonl"]
        or candidate_report["optimizer_steps"] != provenance.optimizer_steps
        or candidate_report["completed_optimizer_steps"] != provenance.optimizer_steps
        or candidate_report["long_training_authorized"] is not False
        or candidate_report["stage_recipe_sha256"] != provenance.stage_recipe_sha256
        or candidate_report["source_coverage_recipe_sha256"]
        != provenance.source_coverage_recipe_sha256
        or candidate_report["foundation_recipe_sha256"] != provenance.foundation_recipe_sha256
        or candidate_report["clip_plan_sha256"] != provenance.clip_plan_sha256
        or candidate_report["world_size"] != 2
        or candidate_report["prefix_lengths"] != [0, 8, 32, 128]
        or candidate_report["train_length"] != 2
        or candidate_report["required_future_horizon"] != 2
        or not isinstance(candidate_report["action_weight"], int | float)
        or isinstance(candidate_report["action_weight"], bool)
        or float(candidate_report["action_weight"]) != 0.0
    ):
        raise ValueError("Stage-B candidate report differs from the accepted checkpoint")
    validate_stationary_candidate_metrics(
        report.parent / "candidate_metrics.jsonl",
        expected_steps=candidate_report["completed_optimizer_steps"],
    )

    fixed_replay = validate_stationary_fixed_replay(
        _read_json(report.parent / "fixed_checkpoint_replay.json", "fixed checkpoint replay")
    )
    fixed_replay_sha256 = artifact_hashes["fixed_checkpoint_replay.json"]
    replay_bindings = fixed_replay["bindings"]
    if (
        fixed_replay["status"] != STATIONARY_FIXED_REPLAY_PASS
        or replay_bindings["candidate_checkpoint_sha256"] != checkpoint_sha256
        or replay_bindings["candidate_report_sha256"] != artifact_hashes["candidate_report.json"]
        or replay_bindings["m2_checkpoint_sha256"] != provenance.m2_checkpoint_sha256
        or replay_bindings["dataset_manifest_sha256"] != provenance.dataset_manifest_sha256
        or replay_bindings["feature_cache_manifest_sha256"]
        != provenance.feature_cache_manifest_sha256
        or replay_bindings["physical_sidecar_manifest_sha256"]
        != provenance.physical_sidecar_manifest_sha256
        or replay_bindings["foundation_recipe_sha256"] != provenance.foundation_recipe_sha256
        or replay_bindings["source_coverage_recipe_sha256"]
        != provenance.source_coverage_recipe_sha256
        or replay_bindings["stage_recipe_sha256"] != provenance.stage_recipe_sha256
        or replay_bindings["candidate_code_revision"] != provenance.code_revision
    ):
        raise ValueError("fixed checkpoint replay differs from the accepted candidate")

    lifecycle = validate_stationary_lifecycle_calibration(
        _read_json(report.parent / "lifecycle_calibration.json", "lifecycle calibration"),
        fixed_replay=fixed_replay,
        fixed_replay_sha256=fixed_replay_sha256,
    )
    if lifecycle["status"] != STATIONARY_LIFECYCLE_CALIBRATION_PASS:
        raise ValueError("stationary lifecycle calibration did not pass")

    runtime = validate_stationary_runtime_probe(
        _read_json(report.parent / "runtime_probe.json", "stationary runtime probe"),
        fixed_replay=fixed_replay,
        fixed_replay_sha256=fixed_replay_sha256,
        candidate_recurrent_state_serialized=provenance.recurrent_state_serialized,
    )
    if runtime["status"] != STATIONARY_RUNTIME_PROBE_PASS:
        raise ValueError("stationary runtime probe did not pass")

    visual_manifest = validate_stationary_visual_artifacts(
        _read_json(report.parent / "visual_artifacts.json", "stationary visual artifacts"),
        evidence_root=report.parent,
    )
    if (
        visual_manifest["candidate_checkpoint_sha256"] != checkpoint_sha256
        or visual_manifest["fixed_checkpoint_replay_sha256"] != fixed_replay_sha256
    ):
        raise ValueError("stationary visual artifacts differ from the accepted replay")
    visual_review = validate_stationary_visual_review(
        _read_json(report.parent / "visual_review.json", "stationary visual review"),
        manifest=visual_manifest,
        manifest_sha256=artifact_hashes["visual_artifacts.json"],
        evidence_root=report.parent,
    )
    if visual_review["status"] != "PASS":
        raise ValueError("stationary visual review did not pass")

    decision = _exact_mapping(
        payload["decision"],
        "stationary temporal acceptance decision",
        {
            "status",
            "checks",
            "failed_checks",
            "later_gates_authorized",
            "long_training_authorized",
        },
    )
    checks = _exact_mapping(
        decision["checks"],
        "stationary temporal acceptance checks",
        _DECISION_CHECKS,
    )
    if (
        decision["status"] != "PASS"
        or not all(value is True for value in checks.values())
        or decision["failed_checks"] != []
        or decision["later_gates_authorized"] != ["M4_action_adoption"]
        or decision["long_training_authorized"] is not False
        or provenance.recurrent_state_serialized is not False
    ):
        raise ValueError("stationary temporal acceptance decision did not pass exactly")

    evidence_sha256 = {
        name: artifact_hashes[name]
        for name in sorted(_EVIDENCE_FILES - {"candidate_metrics.jsonl", "candidate_report.json"})
    }
    return AcceptedStationaryTemporalCore(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        report_sha256=sha256_file(report),
        candidate_report_sha256=artifact_hashes["candidate_report.json"],
        candidate_metrics_sha256=artifact_hashes["candidate_metrics.jsonl"],
        evidence_sha256=evidence_sha256,
        provenance=provenance,
    )
