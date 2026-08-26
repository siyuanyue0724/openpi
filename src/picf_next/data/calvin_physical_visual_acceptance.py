"""Hash-bound visual acceptance for the complete CALVIN physical sidecar."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
)
from picf_next.data.lingbot_calvin_projection import (
    projection_payload_sha256,
    validate_lingbot_calvin_projection_payload,
)
from picf_next.data.token_supervision_policy import (
    token_supervision_policy_sha256,
    validate_known_pixel_token_supervision_policy,
)

CALVIN_PHYSICAL_AUDIT_SCHEMA = "picf-next.calvin-physical-supervision-audit.v5"
CALVIN_PHYSICAL_VISUAL_REVIEW_SCHEMA = "picf-next.calvin-physical-visual-review.v3"
CALVIN_PHYSICAL_VISUAL_ACCEPTANCE_SCHEMA = "picf-next.calvin-physical-visual-acceptance.v4"

_SHA256_LENGTH = 64
_MAXIMUM_JSON_BYTES = 32 * 1024 * 1024
_AUDIT_FIELDS = {
    "format",
    "mode",
    "runtime_input",
    "task_used_for_owner_selection",
    "task_used_for_audit_selection",
    "selection_affects_training",
    "coverage",
    "dataset_manifest_sha256",
    "sidecar_manifest_sha256",
    "training_projection_contract_sha256",
    "training_projection_payload_sha256",
    "training_projection",
    "training_supervision_policy_sha256",
    "training_supervision_policy",
    "frame_count",
    "first_global_index",
    "last_global_index",
    "full_shard_schema_validation",
    "manifest_summary_match",
    "manifest_summary_absolute_error",
    "distributions",
    "selection_contract",
    "record_count",
    "records",
}
_AUDIT_RECORD_FIELDS = {
    "global_index",
    "selection_reasons",
    "task_annotations",
    "identity_keys",
    "visible_identity_keys",
    "panel",
    "panel_sha256",
    "cameras",
    "scanned_metrics",
}
_REVIEW_FIELDS = {
    "schema",
    "reviewer",
    "reviewed_at_utc",
    "audit_manifest_sha256",
    "sidecar_manifest_sha256",
    "rows",
    "checks",
    "status",
    "findings",
}
_REVIEW_ROW_FIELDS = {
    "global_index",
    "panel",
    "panel_sha256",
    "verdict",
    "observations",
    "context_expanded",
}
_REVIEW_CHECKS = {
    "every_panel_opened_original_resolution",
    "both_camera_views_reviewed",
    "task_annotation_matches_scene",
    "visible_owner_assignment_is_correct",
    "unknown_regions_do_not_paint_hidden_objects",
    "training_token_overlay_is_consistent",
    "partially_observed_tokens_are_visually_distinct",
    "ambiguous_cases_expanded",
}
_SELECTION_CONTRACT_FIELDS = {
    "tail_per_metric",
    "tail_directions",
    "temporal_strata",
    "one_median_occurrence_midpoint_per_task",
    "deduplicated",
}
_EXPECTED_SELECTION_CONTRACT = {
    "tail_per_metric": 4,
    "tail_directions": {
        "rgb_mae": ["high"],
        "depth_mae_m": ["high"],
        "depth_p95_m": ["high"],
        "known_pixel_fraction": ["low"],
        "raw_object_pixel_fraction": ["high"],
        "known_object_pixel_fraction": ["low", "high"],
        "known_owner_retention": ["low"],
    },
    "temporal_strata": 16,
    "one_median_occurrence_midpoint_per_task": True,
    "deduplicated": True,
}
_ACCEPTANCE_FIELDS = {
    "schema",
    "status",
    "audit_manifest_sha256",
    "review_sha256",
    "dataset_manifest_sha256",
    "sidecar_manifest_sha256",
    "training_projection_contract_sha256",
    "training_projection_payload_sha256",
    "training_projection",
    "training_supervision_policy_sha256",
    "training_supervision_policy",
    "selection_contract_sha256",
    "panel_set_sha256",
    "record_count",
    "reviewer",
    "reviewed_at_utc",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, *, name: str, fields: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != fields:
        raise ContractError(f"{name} fields differ from the frozen schema")
    return result


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be non-empty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _relative_panel(value: object) -> str:
    result = _text(value, name="CALVIN visual panel")
    path = PurePosixPath(result)
    if (
        "\\" in result
        or "\0" in result
        or path.is_absolute()
        or path.as_posix() != result
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() != ".png"
    ):
        raise ContractError("CALVIN visual panel must be one normalized relative PNG path")
    return result


def _read_json(path: Path, *, expected_sha256: str, name: str) -> dict[str, Any]:
    expected = _sha256(expected_sha256, name=f"{name} expected SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{name} must be one real file")
    if path.stat().st_size > _MAXIMUM_JSON_BYTES:
        raise ContractError(f"{name} exceeds the maximum size")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError(f"{name} differs from its expected SHA-256")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{name} is not valid JSON") from error
    return dict(_mapping(value, name=name))


def _validate_utc_timestamp(value: object) -> str:
    text = _text(value, name="CALVIN visual review timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError("CALVIN visual review timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError("CALVIN visual review timestamp must be UTC")
    return text


def _panel_set_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"picf-next.calvin-physical-visual-panel-set.v1\0")
    for record in records:
        digest.update(
            json.dumps(
                {
                    "global_index": record["global_index"],
                    "panel": record["panel"],
                    "panel_sha256": record["panel_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\0")
    digest.update(len(records).to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _canonical_sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\0")
    digest.update(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    )
    return digest.hexdigest()


def validate_calvin_physical_audit_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_sidecar_manifest_sha256: str,
) -> dict[str, Any]:
    """Reopen the full-tail manifest and every selected panel."""

    resolved = Path(path).expanduser().absolute()
    value = _exact(
        _read_json(
            resolved,
            expected_sha256=expected_sha256,
            name="CALVIN physical full-tail audit",
        ),
        name="CALVIN physical full-tail audit",
        fields=_AUDIT_FIELDS,
    )
    expected_sidecar = _sha256(
        expected_sidecar_manifest_sha256,
        name="CALVIN physical sidecar manifest SHA-256",
    )
    expected_dataset = _sha256(
        expected_dataset_manifest_sha256,
        name="CALVIN dataset manifest SHA-256",
    )
    if (
        value["format"] != CALVIN_PHYSICAL_AUDIT_SCHEMA
        or value["mode"] != "full_tail"
        or value["runtime_input"] is not False
        or value["task_used_for_owner_selection"] is not False
        or value["task_used_for_audit_selection"] is not True
        or value["selection_affects_training"] is not False
        or value["coverage"] != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
        or value["dataset_manifest_sha256"] != expected_dataset
        or value["full_shard_schema_validation"] is not True
        or value["manifest_summary_match"] is not True
        or value["sidecar_manifest_sha256"] != expected_sidecar
    ):
        raise ContractError("CALVIN physical full-tail audit did not pass its machine contract")
    projection_contract_sha256 = _sha256(
        value["training_projection_contract_sha256"],
        name="CALVIN training projection contract SHA-256",
    )
    projection = validate_lingbot_calvin_projection_payload(
        value["training_projection"],
        expected_dataset_manifest_sha256=expected_dataset,
    )
    projection_digest = _sha256(
        value["training_projection_payload_sha256"],
        name="CALVIN training projection payload SHA-256",
    )
    if projection_payload_sha256(projection) != projection_digest:
        raise ContractError("CALVIN training projection payload digest differs")
    supervision_policy = validate_known_pixel_token_supervision_policy(
        value["training_supervision_policy"]
    )
    supervision_policy_digest = _sha256(
        value["training_supervision_policy_sha256"],
        name="CALVIN training supervision policy SHA-256",
    )
    if token_supervision_policy_sha256(supervision_policy) != supervision_policy_digest:
        raise ContractError("CALVIN training supervision policy digest differs")
    frame_count = _positive_int(value["frame_count"], name="CALVIN audited frame count")
    first_global_index = _nonnegative_int(
        value["first_global_index"],
        name="CALVIN first audited global index",
    )
    last_global_index = _nonnegative_int(
        value["last_global_index"],
        name="CALVIN last audited global index",
    )
    if first_global_index != 0 or last_global_index != frame_count - 1:
        raise ContractError("CALVIN physical full-tail audit did not cover one contiguous split")
    if projection["source_frame_count"] != frame_count:
        raise ContractError("CALVIN training projection and audit frame counts differ")
    selection_contract = _exact(
        value["selection_contract"],
        name="CALVIN physical full-tail selection contract",
        fields=_SELECTION_CONTRACT_FIELDS,
    )
    if dict(selection_contract) != _EXPECTED_SELECTION_CONTRACT:
        raise ContractError("CALVIN physical full-tail selection contract changed")
    record_count = _positive_int(value["record_count"], name="CALVIN audit record count")
    records = value["records"]
    if not isinstance(records, list) or len(records) != record_count:
        raise ContractError("CALVIN full-tail audit record count differs")
    validated_records: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    seen_panels: set[str] = set()
    for index, raw in enumerate(records):
        record = _exact(
            raw,
            name=f"CALVIN audit record {index}",
            fields=_AUDIT_RECORD_FIELDS,
        )
        global_index = _nonnegative_int(
            record["global_index"],
            name=f"CALVIN audit record {index} global index",
        )
        panel = _relative_panel(record["panel"])
        panel_sha256 = _sha256(
            record["panel_sha256"],
            name=f"CALVIN audit record {index} panel SHA-256",
        )
        if global_index >= frame_count or global_index in seen_indices or panel in seen_panels:
            raise ContractError("CALVIN full-tail audit panels are duplicated or out of range")
        panel_path = resolved.parent / panel
        resolved_panel = panel_path.resolve()
        if (
            panel_path.is_symlink()
            or not panel_path.is_file()
            or not resolved_panel.is_relative_to(resolved.parent.resolve())
        ):
            raise ContractError("CALVIN full-tail audit panel is absent or symbolic")
        if sha256_file(panel_path) != panel_sha256:
            raise ContractError("CALVIN full-tail audit panel differs from its recorded digest")
        seen_indices.add(global_index)
        seen_panels.add(panel)
        validated_records.append(dict(record))
    return {
        **dict(value),
        "training_projection_contract_sha256": projection_contract_sha256,
        "training_projection_payload_sha256": projection_digest,
        "training_projection": projection,
        "training_supervision_policy_sha256": supervision_policy_digest,
        "training_supervision_policy": supervision_policy,
        "records": validated_records,
        "selection_contract_sha256": _canonical_sha256(
            selection_contract,
            domain=b"picf-next.calvin-physical-full-tail-selection.v1",
        ),
        "panel_set_sha256": _panel_set_sha256(validated_records),
    }


def build_calvin_physical_visual_acceptance(
    *,
    audit_manifest_path: str | Path,
    audit_manifest_sha256: str,
    dataset_manifest_sha256: str,
    sidecar_manifest_sha256: str,
    review_path: str | Path,
    review_sha256: str,
    require_pass: bool = True,
) -> dict[str, object]:
    """Validate one complete visual review and derive its immutable decision."""

    audit_sha = _sha256(audit_manifest_sha256, name="CALVIN audit manifest SHA-256")
    dataset_sha = _sha256(dataset_manifest_sha256, name="CALVIN dataset manifest SHA-256")
    sidecar_sha = _sha256(sidecar_manifest_sha256, name="CALVIN sidecar manifest SHA-256")
    audit = validate_calvin_physical_audit_manifest(
        audit_manifest_path,
        expected_sha256=audit_sha,
        expected_dataset_manifest_sha256=dataset_sha,
        expected_sidecar_manifest_sha256=sidecar_sha,
    )
    review_digest = _sha256(review_sha256, name="CALVIN visual review SHA-256")
    review = _exact(
        _read_json(
            Path(review_path).expanduser().absolute(),
            expected_sha256=review_digest,
            name="CALVIN physical visual review",
        ),
        name="CALVIN physical visual review",
        fields=_REVIEW_FIELDS,
    )
    reviewer = _text(review["reviewer"], name="CALVIN physical visual reviewer")
    reviewed_at = _validate_utc_timestamp(review["reviewed_at_utc"])
    if (
        review["schema"] != CALVIN_PHYSICAL_VISUAL_REVIEW_SCHEMA
        or review["audit_manifest_sha256"] != audit_sha
        or review["sidecar_manifest_sha256"] != sidecar_sha
    ):
        raise ContractError("CALVIN physical visual review identity changed")
    rows = review["rows"]
    audit_records = cast(list[Mapping[str, object]], audit["records"])
    if not isinstance(rows, list) or len(rows) != len(audit_records):
        raise ContractError("CALVIN physical visual review skipped one or more panels")
    row_pass = True
    for index, (raw, audit_record) in enumerate(zip(rows, audit_records, strict=True)):
        row = _exact(
            raw,
            name=f"CALVIN physical visual review row {index}",
            fields=_REVIEW_ROW_FIELDS,
        )
        if (
            row["global_index"] != audit_record["global_index"]
            or row["panel"] != audit_record["panel"]
            or row["panel_sha256"] != audit_record["panel_sha256"]
        ):
            raise ContractError("CALVIN physical visual review row differs from manifest order")
        if row["verdict"] not in {"PASS", "FAIL"} or not isinstance(row["context_expanded"], bool):
            raise ContractError("CALVIN physical visual review row verdict is invalid")
        observations = _text(
            row["observations"],
            name=f"CALVIN physical visual review row {index} observations",
        )
        if len(observations.strip()) < 12:
            raise ContractError("CALVIN physical visual review observations are not substantive")
        row_pass &= row["verdict"] == "PASS"
    checks = _exact(
        review["checks"],
        name="CALVIN physical visual review checks",
        fields=_REVIEW_CHECKS,
    )
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ContractError("CALVIN physical visual review checks must be boolean")
    findings = _text(review["findings"], name="CALVIN physical visual review findings")
    if len(findings.strip()) < 20:
        raise ContractError("CALVIN physical visual review findings are not substantive")
    expected_status = "PASS" if row_pass and all(checks.values()) else "FAIL"
    if review["status"] != expected_status:
        raise ContractError("CALVIN physical visual review status was not recomputed exactly")
    if require_pass and expected_status != "PASS":
        raise ContractError("CALVIN physical visual review did not pass")
    return {
        "schema": CALVIN_PHYSICAL_VISUAL_ACCEPTANCE_SCHEMA,
        "status": expected_status,
        "audit_manifest_sha256": audit_sha,
        "review_sha256": review_digest,
        "dataset_manifest_sha256": dataset_sha,
        "sidecar_manifest_sha256": sidecar_sha,
        "training_projection_contract_sha256": audit["training_projection_contract_sha256"],
        "training_projection_payload_sha256": audit["training_projection_payload_sha256"],
        "training_projection": audit["training_projection"],
        "training_supervision_policy_sha256": audit["training_supervision_policy_sha256"],
        "training_supervision_policy": audit["training_supervision_policy"],
        "selection_contract_sha256": audit["selection_contract_sha256"],
        "panel_set_sha256": audit["panel_set_sha256"],
        "record_count": len(audit_records),
        "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at,
    }


def load_calvin_physical_visual_acceptance(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_sidecar_manifest_sha256: str,
) -> dict[str, object]:
    """Load one finalized PASS decision before any teacher cache is built."""

    value = _exact(
        _read_json(
            Path(path).expanduser().absolute(),
            expected_sha256=expected_sha256,
            name="CALVIN physical visual acceptance",
        ),
        name="CALVIN physical visual acceptance",
        fields=_ACCEPTANCE_FIELDS,
    )
    expected_sidecar = _sha256(
        expected_sidecar_manifest_sha256,
        name="CALVIN physical sidecar manifest SHA-256",
    )
    expected_dataset = _sha256(
        expected_dataset_manifest_sha256,
        name="CALVIN dataset manifest SHA-256",
    )
    if (
        value["schema"] != CALVIN_PHYSICAL_VISUAL_ACCEPTANCE_SCHEMA
        or value["status"] != "PASS"
        or value["dataset_manifest_sha256"] != expected_dataset
        or value["sidecar_manifest_sha256"] != expected_sidecar
    ):
        raise ContractError("CALVIN physical visual acceptance is not a matching PASS")
    for name in (
        "audit_manifest_sha256",
        "review_sha256",
        "dataset_manifest_sha256",
        "sidecar_manifest_sha256",
        "training_projection_contract_sha256",
        "training_projection_payload_sha256",
        "training_supervision_policy_sha256",
        "selection_contract_sha256",
        "panel_set_sha256",
    ):
        _sha256(value[name], name=f"CALVIN physical visual acceptance {name}")
    projection = validate_lingbot_calvin_projection_payload(
        value["training_projection"],
        expected_dataset_manifest_sha256=expected_dataset,
    )
    if projection_payload_sha256(projection) != value["training_projection_payload_sha256"]:
        raise ContractError("CALVIN accepted training projection payload digest differs")
    supervision_policy = validate_known_pixel_token_supervision_policy(
        value["training_supervision_policy"]
    )
    if (
        token_supervision_policy_sha256(supervision_policy)
        != value["training_supervision_policy_sha256"]
    ):
        raise ContractError("CALVIN accepted training supervision policy digest differs")
    _positive_int(value["record_count"], name="CALVIN physical visual acceptance record count")
    _text(value["reviewer"], name="CALVIN physical visual acceptance reviewer")
    _validate_utc_timestamp(value["reviewed_at_utc"])
    return {
        **dict(value),
        "training_projection": projection,
        "training_supervision_policy": supervision_policy,
    }
