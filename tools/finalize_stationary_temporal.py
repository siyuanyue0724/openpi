#!/usr/bin/env python3
"""Publish the only accepted Stage-B package after every evidence gate passes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from picf_next.eval.stationary_lifecycle import (  # noqa: E402
    STATIONARY_LIFECYCLE_CALIBRATION_PASS,
    validate_stationary_lifecycle_calibration,
)
from picf_next.eval.stationary_replay import (  # noqa: E402
    STATIONARY_FIXED_REPLAY_PASS,
    validate_stationary_fixed_replay,
)
from picf_next.eval.stationary_runtime import (  # noqa: E402
    STATIONARY_RUNTIME_PROBE_PASS,
    validate_stationary_runtime_probe,
)
from picf_next.eval.stationary_visual import (  # noqa: E402
    validate_stationary_visual_artifacts,
    validate_stationary_visual_review,
)
from picf_next.training.stage_checkpoints import (  # noqa: E402
    inspect_stationary_temporal_checkpoint,
    sha256_file,
)
from picf_next.training.stationary_acceptance import (  # noqa: E402
    STATIONARY_TEMPORAL_ACCEPTANCE_SCHEMA,
    STATIONARY_TEMPORAL_ACCEPTANCE_STATUS,
    STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT,
    validate_stationary_candidate_metrics,
    validate_stationary_temporal_acceptance,
)

_ARTIFACT_NAMES = (
    STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT,
    "candidate_metrics.jsonl",
    "candidate_report.json",
    "fixed_checkpoint_replay.json",
    "lifecycle_calibration.json",
    "runtime_probe.json",
    "visual_artifacts.json",
    "visual_review.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, default=Path("/mnt"))
    return parser.parse_args()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid ASCII JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_regular(source: Path, destination: Path, name: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} must be one regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def finalize_stationary_temporal(
    *,
    candidate_run_dir: str | Path,
    replay_dir: str | Path,
    visual_review_path: str | Path,
    output_dir: str | Path,
    persistent_root: str | Path = "/mnt",
) -> dict[str, Any]:
    """Validate and atomically publish one immutable M4 input package."""

    candidate_input = Path(candidate_run_dir).expanduser()
    replay_input = Path(replay_dir).expanduser()
    review_input = Path(visual_review_path).expanduser()
    output_input = Path(output_dir).expanduser()
    persistent_input = Path(persistent_root).expanduser()
    if any(
        path.is_symlink()
        for path in (
            candidate_input,
            replay_input,
            review_input,
            output_input,
            persistent_input,
        )
    ):
        raise ValueError("stationary finalizer inputs cannot be symbolic links")
    candidate_root = candidate_input.resolve()
    replay_root = replay_input.resolve()
    review_path = review_input.resolve()
    output = output_input.resolve()
    persistence = persistent_input.resolve()
    for path, name in (
        (candidate_root, "candidate run"),
        (replay_root, "fixed replay"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{name} must be one real directory")
    if not persistence.is_dir() or persistence.is_symlink():
        raise ValueError("stationary persistence root must be one real directory")
    if persistence not in output.parents:
        raise ValueError("stationary acceptance package escaped its persistence root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("stationary acceptance parent must be one real directory")

    candidate_checkpoint = candidate_root / "stationary_temporal_core_candidate.pt"
    candidate_metrics = candidate_root / "metrics.jsonl"
    candidate_report_path = candidate_root / "report.json"
    candidate_report = _read_json(candidate_report_path, "Stage-B candidate report")
    checkpoint_sha256 = sha256_file(candidate_checkpoint)
    if candidate_report.get("checkpoint_sha256") != checkpoint_sha256 or candidate_report.get(
        "metrics_sha256"
    ) != sha256_file(candidate_metrics):
        raise ValueError("Stage-B candidate report is not bound to its files")
    provenance = inspect_stationary_temporal_checkpoint(
        candidate_checkpoint,
        expected_sha256=checkpoint_sha256,
    )
    validate_stationary_candidate_metrics(
        candidate_metrics,
        expected_steps=provenance.optimizer_steps,
    )

    fixed_path = replay_root / "fixed_checkpoint_replay.json"
    fixed_replay = validate_stationary_fixed_replay(
        _read_json(fixed_path, "fixed checkpoint replay")
    )
    fixed_sha256 = sha256_file(fixed_path)
    if (
        fixed_replay["status"] != STATIONARY_FIXED_REPLAY_PASS
        or fixed_replay["bindings"]["candidate_checkpoint_sha256"] != checkpoint_sha256
        or fixed_replay["bindings"]["candidate_report_sha256"] != sha256_file(candidate_report_path)
        or fixed_replay["bindings"]["candidate_code_revision"] != provenance.code_revision
    ):
        raise ValueError("fixed replay did not pass for this exact candidate")

    lifecycle_path = replay_root / "lifecycle_calibration.json"
    lifecycle = validate_stationary_lifecycle_calibration(
        _read_json(lifecycle_path, "stationary lifecycle calibration"),
        fixed_replay=fixed_replay,
        fixed_replay_sha256=fixed_sha256,
    )
    if lifecycle["status"] != STATIONARY_LIFECYCLE_CALIBRATION_PASS:
        raise ValueError("stationary lifecycle calibration did not pass")

    runtime_path = replay_root / "runtime_probe.json"
    runtime = validate_stationary_runtime_probe(
        _read_json(runtime_path, "stationary runtime probe"),
        fixed_replay=fixed_replay,
        fixed_replay_sha256=fixed_sha256,
        candidate_recurrent_state_serialized=provenance.recurrent_state_serialized,
    )
    if runtime["status"] != STATIONARY_RUNTIME_PROBE_PASS:
        raise ValueError("stationary runtime probe did not pass")

    visual_manifest_path = replay_root / "visual_artifacts.json"
    visual_manifest = validate_stationary_visual_artifacts(
        _read_json(visual_manifest_path, "stationary visual artifacts"),
        evidence_root=replay_root,
    )
    visual_review = validate_stationary_visual_review(
        _read_json(review_path, "stationary visual review"),
        manifest=visual_manifest,
        manifest_sha256=sha256_file(visual_manifest_path),
        evidence_root=replay_root,
    )
    if visual_review["status"] != "PASS":
        raise ValueError("stationary visual review did not pass")

    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        copies = {
            STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT: candidate_checkpoint,
            "candidate_metrics.jsonl": candidate_metrics,
            "candidate_report.json": candidate_report_path,
            "fixed_checkpoint_replay.json": fixed_path,
            "lifecycle_calibration.json": lifecycle_path,
            "runtime_probe.json": runtime_path,
            "visual_artifacts.json": visual_manifest_path,
            "visual_review.json": review_path,
        }
        for name, source in copies.items():
            _copy_regular(source, staging / name, name)
        for artifact in visual_manifest["artifacts"]:
            relative = Path(artifact["path"])
            _copy_regular(
                replay_root / relative,
                staging / relative,
                f"stationary visual {relative}",
            )

        checks = {
            "candidate_metrics_detection_support_validated": True,
            "candidate_report_validated": True,
            "fixed_checkpoint_replay_passed": True,
            "full_stationary_checkpoint_hash_bound": True,
            "lifecycle_calibration_passed": True,
            "no_recurrent_state_serialized": True,
            "runtime_probe_passed": True,
            "visual_review_passed": True,
        }
        report = {
            "schema": STATIONARY_TEMPORAL_ACCEPTANCE_SCHEMA,
            "status": STATIONARY_TEMPORAL_ACCEPTANCE_STATUS,
            "provenance": provenance.to_dict(),
            "artifacts_sha256": {name: sha256_file(staging / name) for name in _ARTIFACT_NAMES},
            "decision": {
                "status": "PASS",
                "checks": checks,
                "failed_checks": [],
                "later_gates_authorized": ["M4_action_adoption"],
                "long_training_authorized": False,
            },
        }
        _write_json(staging / "report.json", report)
        validate_stationary_temporal_acceptance(
            report_path=staging / "report.json",
            checkpoint_path=staging / STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT,
        )
        os.replace(staging, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "output_dir": str(output),
        "report_sha256": sha256_file(output / "report.json"),
        "checkpoint_sha256": sha256_file(output / STATIONARY_TEMPORAL_ACCEPTED_CHECKPOINT),
        "status": STATIONARY_TEMPORAL_ACCEPTANCE_STATUS,
        "long_training_authorized": False,
    }


def main() -> None:
    args = _parse_args()
    result = finalize_stationary_temporal(
        candidate_run_dir=args.candidate_run_dir,
        replay_dir=args.replay_dir,
        visual_review_path=args.visual_review,
        output_dir=args.output_dir,
        persistent_root=args.persistent_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
