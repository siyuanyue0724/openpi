"""Immutable step-zero replay contract for reference-derived representation trials.

The contract is scientific control-plane evidence. It does not participate in
model inputs, loss construction, recurrent state, or optimizer updates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from picf_next.lingbot_native.representation_evaluation import (
    REPRESENTATION_EVALUATION_PARTITIONS,
    REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA,
    RepresentationEvaluationPlan,
    validate_representation_evaluation_snapshot,
    validate_representation_evaluation_visual_files,
)

REPRESENTATION_EVALUATION_BASELINE_SCHEMA_V1 = (
    "picf-next.lingbot-representation-evaluation-baseline.v1"
)
REPRESENTATION_EVALUATION_BASELINE_SCHEMA = (
    "picf-next.lingbot-representation-evaluation-baseline.v2"
)
REPRESENTATION_BASELINE_SAMPLE_SCHEMA = "picf-next.lingbot-representation-baseline-sample.v1"
REPRESENTATION_BASELINE_REPLAY_REPORT_SCHEMA = (
    "picf-next.lingbot-representation-baseline-replay-report.v2"
)

_BASELINE_FIELDS_V1 = frozenset(
    {
        "schema",
        "status",
        "checkpoint_global_step",
        "source_snapshot_file_sha256",
        "source_snapshot_artifact_sha256",
        "source_implementation_sha256",
        "source_evaluation_plan_file_sha256",
        "source_evaluation_plan_artifact_sha256",
        "source_representation_split_sha256",
        "model_family_sha256",
        "representation_frozen_action_state_sha256",
        "evaluation_item_bank_sha256",
        "sample_count",
        "samples",
        "partition_stable_sha256",
        "artifact_sha256",
    }
)
_BASELINE_FIELDS = _BASELINE_FIELDS_V1 | {"source_replay_seed_sha256"}
_BASELINE_SAMPLE_FIELDS = frozenset(
    {
        "schema",
        "partition",
        "ordinal",
        "rank",
        "sample_key",
        "stable_evidence_sha256",
        "official_action_loss",
        "tensor_sha256",
        "visual_sha256",
        "visual_bytes",
    }
)
_REPLAY_REPORT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "checkpoint_global_step",
        "baseline_artifact_sha256",
        "candidate_snapshot_artifact_sha256",
        "candidate_evaluation_plan_artifact_sha256",
        "replay_seed_sha256",
        "source_model_family_sha256",
        "candidate_model_family_sha256",
        "representation_frozen_action_state_sha256",
        "evaluation_item_bank_sha256",
        "sample_count",
        "partition_stable_sha256",
        "artifact_sha256",
    }
)
_NONDETERMINISTIC_SAMPLE_FIELDS = frozenset(
    {
        "checkpoint_global_step",
        "forward_seconds",
        "peak_cuda_reserved_bytes",
    }
)
_NONDETERMINISTIC_PARTITION_FIELDS = frozenset(
    {
        "maximum_peak_cuda_reserved_bytes",
        "mean_factual_forward_seconds",
        "mean_shuffled_task_forward_seconds",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _stable_sample_sha256(sample: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            name: value
            for name, value in sample.items()
            if name not in _NONDETERMINISTIC_SAMPLE_FIELDS
        }
    )


def _partition_stable_sha256(snapshot: Mapping[str, object]) -> dict[str, str]:
    raw = snapshot["partition_summaries"]
    if not isinstance(raw, Mapping):
        raise ValueError("representation baseline snapshot has no partition summaries")
    result: dict[str, str] = {}
    for partition in REPRESENTATION_EVALUATION_PARTITIONS:
        value = raw.get(partition)
        if not isinstance(value, Mapping):
            raise ValueError(f"representation baseline omits {partition} summary")
        result[partition] = _canonical_sha256(
            {
                name: item
                for name, item in value.items()
                if name not in _NONDETERMINISTIC_PARTITION_FIELDS
            }
        )
    return result


def _evaluation_item_bank_sha256(plan: RepresentationEvaluationPlan) -> str:
    return _canonical_sha256([item.as_dict() for item in plan.items])


def _baseline_sample(sample: Mapping[str, object]) -> dict[str, object]:
    tensor_sha256 = sample.get("tensor_sha256")
    visual = sample.get("visual_artifact")
    if not isinstance(tensor_sha256, Mapping) or not isinstance(visual, Mapping):
        raise ValueError("representation baseline sample evidence is malformed")
    return {
        "schema": REPRESENTATION_BASELINE_SAMPLE_SCHEMA,
        "partition": sample["partition"],
        "ordinal": sample["ordinal"],
        "rank": sample["rank"],
        "sample_key": sample["sample_key"],
        "stable_evidence_sha256": _stable_sample_sha256(sample),
        "official_action_loss": sample["official_action_loss"],
        "tensor_sha256": dict(tensor_sha256),
        "visual_sha256": visual["sha256"],
        "visual_bytes": visual["bytes"],
    }


def _validate_baseline_sample(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _BASELINE_SAMPLE_FIELDS:
        raise ValueError("representation baseline sample fields differ from schema")
    if value["schema"] != REPRESENTATION_BASELINE_SAMPLE_SCHEMA:
        raise ValueError("representation baseline sample schema changed")
    if value["partition"] not in REPRESENTATION_EVALUATION_PARTITIONS:
        raise ValueError("representation baseline sample partition changed")
    _nonnegative_int(value["ordinal"], name="representation baseline sample ordinal")
    _nonnegative_int(value["rank"], name="representation baseline sample rank")
    if not isinstance(value["sample_key"], str) or not value["sample_key"]:
        raise ValueError("representation baseline sample key must be nonempty")
    _sha256(
        value["stable_evidence_sha256"],
        name="representation baseline stable evidence",
    )
    if (
        _finite_float(
            value["official_action_loss"],
            name="representation baseline action loss",
        )
        < 0
    ):
        raise ValueError("representation baseline action loss must be non-negative")
    tensor_sha256 = value["tensor_sha256"]
    if not isinstance(tensor_sha256, dict) or not tensor_sha256:
        raise ValueError("representation baseline tensor evidence is malformed")
    for name, digest in tensor_sha256.items():
        if not isinstance(name, str) or not name:
            raise ValueError("representation baseline tensor name is invalid")
        _sha256(digest, name=f"representation baseline tensor {name}")
    _sha256(value["visual_sha256"], name="representation baseline visual")
    _positive_int(value["visual_bytes"], name="representation baseline visual bytes")
    return value


def build_representation_evaluation_baseline(
    *,
    source_snapshot: Mapping[str, object],
    source_snapshot_file_sha256: str,
    source_evaluation_plan: RepresentationEvaluationPlan,
    source_evaluation_plan_file_sha256: str,
    source_visual_root: str | Path,
) -> dict[str, object]:
    """Compress a validated historical step-zero snapshot into a replay contract."""

    snapshot = validate_representation_evaluation_snapshot(
        dict(source_snapshot),
        plan=source_evaluation_plan,
    )
    validate_representation_evaluation_visual_files(
        snapshot,
        plan=source_evaluation_plan,
        output_root=source_visual_root,
    )
    if snapshot["checkpoint_global_step"] != 0:
        raise ValueError("representation baseline source must be checkpoint step zero")
    value: dict[str, object] = {
        "schema": REPRESENTATION_EVALUATION_BASELINE_SCHEMA,
        "status": "PASS",
        "checkpoint_global_step": 0,
        "source_snapshot_file_sha256": _sha256(
            source_snapshot_file_sha256,
            name="representation baseline source snapshot file",
        ),
        "source_snapshot_artifact_sha256": snapshot["artifact_sha256"],
        "source_implementation_sha256": snapshot["implementation_sha256"],
        "source_evaluation_plan_file_sha256": _sha256(
            source_evaluation_plan_file_sha256,
            name="representation baseline source plan file",
        ),
        "source_evaluation_plan_artifact_sha256": source_evaluation_plan.artifact_sha256,
        "source_replay_seed_sha256": source_evaluation_plan.replay_seed_sha256,
        "source_representation_split_sha256": snapshot["representation_split_sha256"],
        "model_family_sha256": snapshot["model_family_sha256"],
        "representation_frozen_action_state_sha256": (
            snapshot["representation_frozen_action_state_sha256"]
        ),
        "evaluation_item_bank_sha256": _evaluation_item_bank_sha256(source_evaluation_plan),
        "sample_count": len(snapshot["samples"]),
        "samples": [_baseline_sample(sample) for sample in snapshot["samples"]],
        "partition_stable_sha256": _partition_stable_sha256(snapshot),
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return validate_representation_evaluation_baseline(value)


def validate_representation_evaluation_baseline(value: object) -> dict[str, Any]:
    """Validate the self-contained historical replay contract."""

    if not isinstance(value, dict):
        raise ValueError("representation baseline fields differ from schema")
    schema = value.get("schema")
    fields = (
        _BASELINE_FIELDS
        if schema == REPRESENTATION_EVALUATION_BASELINE_SCHEMA
        else _BASELINE_FIELDS_V1
    )
    if set(value) != fields:
        raise ValueError("representation baseline fields differ from schema")
    if (
        schema
        not in {
            REPRESENTATION_EVALUATION_BASELINE_SCHEMA_V1,
            REPRESENTATION_EVALUATION_BASELINE_SCHEMA,
        }
        or value["status"] != "PASS"
        or value["checkpoint_global_step"] != 0
    ):
        raise ValueError("representation baseline status, schema, or step changed")
    for name in (
        "source_snapshot_file_sha256",
        "source_snapshot_artifact_sha256",
        "source_implementation_sha256",
        "source_evaluation_plan_file_sha256",
        "source_evaluation_plan_artifact_sha256",
        "source_representation_split_sha256",
        "model_family_sha256",
        "representation_frozen_action_state_sha256",
        "evaluation_item_bank_sha256",
    ):
        _sha256(value[name], name=f"representation baseline {name}")
    if schema == REPRESENTATION_EVALUATION_BASELINE_SCHEMA:
        _sha256(
            value["source_replay_seed_sha256"],
            name="representation baseline source_replay_seed_sha256",
        )
    raw_samples = value["samples"]
    sample_count = _positive_int(
        value["sample_count"],
        name="representation baseline sample count",
    )
    if not isinstance(raw_samples, list) or len(raw_samples) != sample_count:
        raise ValueError("representation baseline sample coverage changed")
    samples = tuple(_validate_baseline_sample(sample) for sample in raw_samples)
    identities = tuple(
        (sample["partition"], sample["ordinal"], sample["rank"], sample["sample_key"])
        for sample in samples
    )
    if len(set(identities)) != len(identities):
        raise ValueError("representation baseline sample identities are not unique")
    partition_hashes = value["partition_stable_sha256"]
    if not isinstance(partition_hashes, dict) or set(partition_hashes) != set(
        REPRESENTATION_EVALUATION_PARTITIONS
    ):
        raise ValueError("representation baseline partition hashes changed")
    for partition, digest in partition_hashes.items():
        _sha256(digest, name=f"representation baseline {partition} partition")
    artifact = _sha256(
        value["artifact_sha256"],
        name="representation baseline artifact",
    )
    payload = {name: value[name] for name in fields if name != "artifact_sha256"}
    if _canonical_sha256(payload) != artifact:
        raise ValueError("representation baseline artifact SHA-256 changed")
    return value


def load_representation_evaluation_baseline(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid representation evaluation baseline: {source}") from error
    return validate_representation_evaluation_baseline(value)


def write_representation_evaluation_baseline(
    path: str | Path,
    value: Mapping[str, object],
) -> None:
    baseline = validate_representation_evaluation_baseline(dict(value))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{baseline['artifact_sha256'][:12]}"
    )
    if (
        destination.exists()
        or destination.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(f"representation evaluation baseline path exists: {destination}")
    payload = json.dumps(baseline, indent=2, sort_keys=True) + "\n"
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


def validate_representation_baseline_plan(
    baseline: Mapping[str, object],
    *,
    candidate_plan: RepresentationEvaluationPlan,
) -> None:
    reference = validate_representation_evaluation_baseline(dict(baseline))
    source_plan_sha256 = reference["source_evaluation_plan_artifact_sha256"]
    if reference["schema"] == REPRESENTATION_EVALUATION_BASELINE_SCHEMA_V1:
        source_replay_seed_sha256 = source_plan_sha256
        exact_plan_replay = candidate_plan.artifact_sha256 == source_plan_sha256
        reference_derived_replay = (
            candidate_plan.schema == REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA
            and candidate_plan.evaluation_reference_plan_sha256 == source_plan_sha256
            and candidate_plan.replay_seed_sha256 == source_plan_sha256
        )
    else:
        source_replay_seed_sha256 = reference["source_replay_seed_sha256"]
        exact_plan_replay = (
            candidate_plan.artifact_sha256 == source_plan_sha256
            and candidate_plan.replay_seed_sha256 == source_replay_seed_sha256
        )
        reference_derived_replay = (
            candidate_plan.schema == REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA
            and candidate_plan.replay_seed_sha256 == source_replay_seed_sha256
        )
    if not (exact_plan_replay or reference_derived_replay):
        raise ValueError("representation candidate changed the historical replay seed")
    if _evaluation_item_bank_sha256(candidate_plan) != reference["evaluation_item_bank_sha256"]:
        raise ValueError("representation candidate changed the historical evaluation bank")


def build_representation_baseline_replay_report(
    *,
    baseline: Mapping[str, object],
    candidate_snapshot: Mapping[str, object],
    candidate_plan: RepresentationEvaluationPlan,
    candidate_visual_root: str | Path,
) -> dict[str, object]:
    """Require exact deterministic step-zero replay before any K8 optimizer update."""

    reference = validate_representation_evaluation_baseline(dict(baseline))
    validate_representation_baseline_plan(reference, candidate_plan=candidate_plan)
    snapshot = validate_representation_evaluation_snapshot(
        dict(candidate_snapshot),
        plan=candidate_plan,
    )
    validate_representation_evaluation_visual_files(
        snapshot,
        plan=candidate_plan,
        output_root=candidate_visual_root,
    )
    if snapshot["checkpoint_global_step"] != 0:
        raise ValueError("representation baseline replay must run at checkpoint step zero")
    if (
        snapshot["representation_frozen_action_state_sha256"]
        != (reference["representation_frozen_action_state_sha256"])
    ):
        raise ValueError("representation baseline replay changed frozen action state")
    # The family digest binds the execution/split/stream plan, so a reference-derived
    # K8 candidate cannot equal its K1 source. Functional equivalence is enforced below
    # over every deterministic tensor, action, visual, and partition artifact while both
    # plan-specific family digests remain explicit provenance in the replay report.
    candidate_samples = [_baseline_sample(sample) for sample in snapshot["samples"]]
    expected_samples = reference["samples"]
    if len(candidate_samples) != len(expected_samples):
        raise ValueError("representation baseline replay sample coverage changed")
    for expected, observed in zip(expected_samples, candidate_samples, strict=True):
        if expected != observed:
            identity = (
                observed.get("sample_key") if isinstance(observed, Mapping) else "<malformed>"
            )
            differing = sorted(
                name
                for name in _BASELINE_SAMPLE_FIELDS
                if not isinstance(expected, Mapping) or expected.get(name) != observed.get(name)
            )
            raise ValueError(
                "representation baseline replay changed deterministic evidence for "
                f"{identity}: {differing}"
            )
    partition_hashes = _partition_stable_sha256(snapshot)
    if partition_hashes != reference["partition_stable_sha256"]:
        raise ValueError("representation baseline replay changed stable partition metrics")
    value: dict[str, object] = {
        "schema": REPRESENTATION_BASELINE_REPLAY_REPORT_SCHEMA,
        "status": "PASS",
        "checkpoint_global_step": 0,
        "baseline_artifact_sha256": reference["artifact_sha256"],
        "candidate_snapshot_artifact_sha256": snapshot["artifact_sha256"],
        "candidate_evaluation_plan_artifact_sha256": candidate_plan.artifact_sha256,
        "replay_seed_sha256": candidate_plan.replay_seed_sha256,
        "source_model_family_sha256": reference["model_family_sha256"],
        "candidate_model_family_sha256": snapshot["model_family_sha256"],
        "representation_frozen_action_state_sha256": (
            snapshot["representation_frozen_action_state_sha256"]
        ),
        "evaluation_item_bank_sha256": _evaluation_item_bank_sha256(candidate_plan),
        "sample_count": len(candidate_samples),
        "partition_stable_sha256": partition_hashes,
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return validate_representation_baseline_replay_report(value)


def validate_representation_baseline_replay_report(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _REPLAY_REPORT_FIELDS:
        raise ValueError("representation baseline replay report fields differ from schema")
    if (
        value["schema"] != REPRESENTATION_BASELINE_REPLAY_REPORT_SCHEMA
        or value["status"] != "PASS"
        or value["checkpoint_global_step"] != 0
    ):
        raise ValueError("representation baseline replay report did not pass at step zero")
    for name in (
        "baseline_artifact_sha256",
        "candidate_snapshot_artifact_sha256",
        "candidate_evaluation_plan_artifact_sha256",
        "replay_seed_sha256",
        "source_model_family_sha256",
        "candidate_model_family_sha256",
        "representation_frozen_action_state_sha256",
        "evaluation_item_bank_sha256",
    ):
        _sha256(value[name], name=f"representation baseline replay {name}")
    _positive_int(
        value["sample_count"],
        name="representation baseline replay sample count",
    )
    partition_hashes = value["partition_stable_sha256"]
    if not isinstance(partition_hashes, dict) or set(partition_hashes) != set(
        REPRESENTATION_EVALUATION_PARTITIONS
    ):
        raise ValueError("representation baseline replay partition hashes changed")
    for partition, digest in partition_hashes.items():
        _sha256(digest, name=f"representation baseline replay {partition} partition")
    artifact = _sha256(
        value["artifact_sha256"],
        name="representation baseline replay artifact",
    )
    payload = {name: value[name] for name in _REPLAY_REPORT_FIELDS if name != "artifact_sha256"}
    if _canonical_sha256(payload) != artifact:
        raise ValueError("representation baseline replay report artifact changed")
    return value


def load_representation_baseline_replay_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid representation baseline replay report: {source}") from error
    return validate_representation_baseline_replay_report(value)


def write_representation_baseline_replay_report(
    path: str | Path,
    value: Mapping[str, object],
) -> None:
    report = validate_representation_baseline_replay_report(dict(value))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{report['artifact_sha256'][:12]}"
    )
    if (
        destination.exists()
        or destination.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(f"representation baseline replay report path exists: {destination}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
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
