#!/usr/bin/env python3
"""Finalize the public native-VL retention gate after independent visual review.

The numeric comparison is recomputed from its exact control, candidate, and
training reports.  Visual decisions bind decoded panel bytes and the complete
report-side sample evidence rendered in each panel.  PASS remains fail-closed
until this repository has a mature, pinned reviewer-signature trust root.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import stat
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from PIL import Image, UnidentifiedImageError

try:
    from tools.compare_public_native_vl_retention_reports import (
        FIXED_X_SCHEMA as FIXED_X_SCHEMA,
    )
    from tools.compare_public_native_vl_retention_reports import (
        OUTPUT_SCHEMA as COMPARISON_SCHEMA,
    )
    from tools.compare_public_native_vl_retention_reports import (
        TRAINING_SCHEMA as TRAINING_SCHEMA,
    )
    from tools.compare_public_native_vl_retention_reports import (
        compare_public_native_vl_retention_reports,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from compare_public_native_vl_retention_reports import (  # type: ignore[no-redef]
        FIXED_X_SCHEMA as FIXED_X_SCHEMA,
    )
    from compare_public_native_vl_retention_reports import (
        OUTPUT_SCHEMA as COMPARISON_SCHEMA,
    )
    from compare_public_native_vl_retention_reports import (
        TRAINING_SCHEMA as TRAINING_SCHEMA,
    )
    from compare_public_native_vl_retention_reports import (
        compare_public_native_vl_retention_reports,
    )

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    native_vl_calvin_task_family,
)

REVIEW_SCHEMA = "picf-next.public-native-vl-retention-visual-review.v2"
FINALIZED_SCHEMA = "picf-next.public-native-vl-retention-finalized.v2"
PENDING_STATUS = "PASS_PENDING_CALVIN_VISUAL_REVIEW"
REVIEWER_TRUST_BLOCKER = (
    "reviewer identity is unsigned: no approved asymmetric signature dependency "
    "and pinned reviewer public-key trust root are configured"
)

_REVIEW_FIELDS = {
    "candidate_report",
    "candidate_report_sha256",
    "candidate_training_report",
    "candidate_training_report_sha256",
    "comparison_report",
    "comparison_report_sha256",
    "control_report",
    "control_report_sha256",
    "decisions",
    "reviewer",
    "schema",
    "status",
}
_CALVIN_DECISION_FIELDS = {
    "decision",
    "family",
    "finding",
    "ordinal",
    "panels",
    "partition",
    "required_visual_review_item_sha256",
    "task_key",
    "variant_index",
}
_PUBLIC_DECISION_FIELDS = {
    "decision",
    "family",
    "finding",
    "panels",
    "record_id",
    "required_visual_review_item_sha256",
    "source_row_index",
    "source_subindex",
}
_PANEL_FIELDS = {
    "image_format",
    "image_height",
    "image_width",
    "path",
    "report",
    "sample_sha256",
    "sha256",
}
_CALVIN_NORMAL_REQUIRED_FIELDS = {
    "family",
    "ordinal",
    "partition",
    "reasons",
    "task_key",
    "variant_index",
}
_CALVIN_FALLBACK_REQUIRED_FIELDS = {"ordinal", "partition", "reason"}
_PUBLIC_REQUIRED_FIELDS = {
    "family",
    "reasons",
    "record_id",
    "source_row_index",
    "source_subindex",
}
_PUBLIC_REVIEW_FIELD = "public_generation_visual_review_required"
_CALVIN_SAMPLE_BINDINGS = {
    "ordinal",
    "partition",
    "source_global_index",
    "source_rgb_sha256",
    "source_state_sha256",
}
_CALVIN_VARIANT_BINDINGS = {
    "camera_name",
    "instruction",
    "instruction_sha256",
    "target_bbox_qwen_xyxy",
    "target_identity_key",
    "task_key",
}
_PUBLIC_SAMPLE_BINDINGS = {
    "family",
    "image_height",
    "image_rgb_sha256",
    "image_width",
    "record_id",
    "record_sha256",
    "source_row_index",
    "source_subindex",
    "target_answer",
    "target_answer_sha256",
    "user_text",
    "user_text_sha256",
}
_FAMILY_GATE_FIELDS = {
    "mean_record_nll_nonregression",
    "token_weighted_mean_nll_strictly_improves",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-report", required=True, type=Path)
    parser.add_argument("--comparison-report-sha256", required=True)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--control-report-sha256", required=True)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--candidate-report-sha256", required=True)
    parser.add_argument("--candidate-training-report", required=True, type=Path)
    parser.add_argument("--candidate-training-report-sha256", required=True)
    parser.add_argument("--visual-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ContractError(f"{name} must be nonempty text")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(constant: str) -> NoReturn:
    raise ValueError(f"non-finite JSON value: {constant}")


def _load_json_bytes(payload: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except ContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError(f"{name} is not strict UTF-8 JSON") from error
    return _mapping(value, name=name)


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


def _absolute_lexical_path(path: Path, *, name: str) -> Path:
    text = os.fspath(path.expanduser())
    if not text or "\0" in text:
        raise ContractError(f"{name} path is malformed")
    return Path(os.path.abspath(text))


def _open_directory_descriptor(path: Path, *, name: str) -> int:
    if not path.is_absolute():
        raise RuntimeError("secure directory traversal requires an absolute path")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.anchor, flags)
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags | nofollow, dir_fd=descriptor)
            finally:
                os.close(descriptor)
            descriptor = child
    except OSError as error:
        raise ContractError(
            f"{name} path is missing, non-directory, or traverses a symlink"
        ) from error
    return descriptor


def _read_regular_file_descriptor(path: Path, *, name: str) -> _FileSnapshot:
    lexical = _absolute_lexical_path(path, name=name)
    parent_descriptor = _open_directory_descriptor(lexical.parent, name=name)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(lexical.name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise ContractError(
                f"{name} must be a readable regular file and must not be a symlink"
            ) from error
    finally:
        os.close(parent_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"{name} must be a regular file")
        blocks = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ContractError(f"{name} cannot be read") from error
    finally:
        os.close(descriptor)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(payload) != after.st_size:
        raise ContractError(f"{name} changed while being read")
    return _FileSnapshot(
        path=lexical,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
    )


def _load_bound_json(
    path: Path,
    *,
    expected_sha256: object,
    name: str,
) -> tuple[_FileSnapshot, Mapping[str, Any]]:
    expected = _require_sha256(expected_sha256, name=f"{name} expected digest")
    snapshot = _read_regular_file_descriptor(path, name=name)
    if not hmac.compare_digest(snapshot.sha256, expected):
        raise ContractError(f"{name} SHA-256 differs from the required digest")
    return snapshot, _load_json_bytes(snapshot.payload, name=name)


def _load_unbound_json(path: Path, *, name: str) -> tuple[_FileSnapshot, Mapping[str, Any]]:
    snapshot = _read_regular_file_descriptor(path, name=name)
    return snapshot, _load_json_bytes(snapshot.payload, name=name)


def _assert_snapshot_unchanged(snapshot: _FileSnapshot, *, name: str) -> None:
    current = _read_regular_file_descriptor(snapshot.path, name=name)
    expected_identity = (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.sha256,
    )
    current_identity = (
        current.device,
        current.inode,
        current.size,
        current.mtime_ns,
        current.sha256,
    )
    if current_identity != expected_identity:
        raise ContractError(f"{name} changed after validation")


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ContractError("visual-review evidence is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _safe_panel_snapshot(
    *,
    base: Path,
    relative: object,
    name: str,
) -> tuple[str, _FileSnapshot]:
    text = _require_text(relative, name=f"{name} path")
    if text != text.strip():
        raise ContractError(f"{name} path must not have surrounding whitespace")
    value = Path(text)
    if value.is_absolute() or value == Path(".") or ".." in value.parts:
        raise ContractError(f"{name} path must be safe and relative")
    canonical = value.as_posix()
    if text != canonical or "\\" in text:
        raise ContractError(f"{name} path must use one canonical POSIX spelling")
    snapshot = _read_regular_file_descriptor(base.joinpath(value), name=name)
    if snapshot.size <= 0:
        raise ContractError(f"{name} must not be empty")
    return canonical, snapshot


def _decoded_image_metadata(
    payload: bytes,
    *,
    relative: str,
    name: str,
) -> dict[str, int | str]:
    expected_formats = {
        ".jpeg": "JPEG",
        ".jpg": "JPEG",
        ".png": "PNG",
        ".webp": "WEBP",
    }
    expected_format = expected_formats.get(Path(relative).suffix.lower())
    if expected_format is None:
        raise ContractError(f"{name} must use a supported image extension")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                image_format = image.format
                width, height = image.size
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        UnidentifiedImageError,
    ) as error:
        raise ContractError(f"{name} is not a decodable image") from error
    if image_format != expected_format or width <= 0 or height <= 0:
        raise ContractError(f"{name} image format or dimensions are invalid")
    return {
        "image_format": image_format,
        "image_height": height,
        "image_width": width,
    }


def _sample_evidence_sha256(value: Mapping[str, Any], *, name: str) -> str:
    evidence = {key: item for key, item in value.items() if key != "visual"}
    if not evidence:
        raise ContractError(f"{name} sample evidence is empty")
    return _canonical_sha256(evidence)


def _validate_declared_visual(
    value: object,
    *,
    role: str,
    report_path: Path,
    item_index: int,
    sample_sha256: str,
    snapshots: list[_FileSnapshot],
) -> dict[str, int | str]:
    visual = _mapping(value, name=f"{role} fixed-X declared visual")
    if set(visual) != {"file", "sha256"}:
        raise ContractError(f"{role} fixed-X visual fields differ from schema")
    expected = _require_sha256(visual.get("sha256"), name=f"{role} fixed-X visual digest")
    relative, snapshot = _safe_panel_snapshot(
        base=report_path.parent,
        relative=visual.get("file"),
        name=f"required item {item_index} {role} declared visual",
    )
    if not hmac.compare_digest(snapshot.sha256, expected):
        raise ContractError(f"{role} fixed-X declared visual changed: {relative}")
    metadata = _decoded_image_metadata(
        snapshot.payload,
        relative=relative,
        name=f"required item {item_index} {role} declared visual",
    )
    snapshots.append(snapshot)
    return {
        **metadata,
        "path": relative,
        "report": role,
        "sample_sha256": _require_sha256(
            sample_sha256,
            name=f"required item {item_index} {role} sample digest",
        ),
        "sha256": snapshot.sha256,
    }


def _calvin_group_index(
    report: Mapping[str, Any],
    *,
    role: str,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"{role} fixed-X report has no CALVIN result groups")
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for value in rows:
        group = _mapping(value, name=f"{role} CALVIN result group")
        partition = _require_text(group.get("partition"), name=f"{role} CALVIN partition")
        ordinal = _require_nonnegative_integer(
            group.get("ordinal"),
            name=f"{role} CALVIN ordinal",
        )
        key = partition, ordinal
        if key in result:
            raise ContractError(f"{role} fixed-X report repeats a CALVIN result group")
        variants = group.get("variants")
        if not isinstance(variants, list) or len(variants) != 2:
            raise ContractError(f"{role} CALVIN group must contain exactly two variants")
        if any(not isinstance(variant, Mapping) for variant in variants):
            raise ContractError(f"{role} CALVIN group contains a malformed variant")
        result[key] = group
    return result


def _calvin_sample_binding(group: Mapping[str, Any], *, role: str) -> dict[str, object]:
    binding: dict[str, object] = {}
    for field in _CALVIN_SAMPLE_BINDINGS:
        value = group.get(field)
        if field in {"ordinal", "source_global_index"}:
            value = _require_nonnegative_integer(value, name=f"{role} CALVIN {field}")
        elif field in {"source_rgb_sha256", "source_state_sha256"}:
            value = _require_sha256(value, name=f"{role} CALVIN {field}")
        else:
            value = _require_text(value, name=f"{role} CALVIN {field}")
        binding[field] = value
    variants = group.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise ContractError(f"{role} CALVIN group lost its two variants")
    variant_bindings = []
    for value in variants:
        variant = _mapping(value, name=f"{role} CALVIN variant")
        normalized = {}
        for field in _CALVIN_VARIANT_BINDINGS:
            field_value = variant.get(field)
            if field == "instruction_sha256":
                field_value = _require_sha256(
                    field_value,
                    name=f"{role} CALVIN instruction digest",
                )
            elif field == "target_bbox_qwen_xyxy":
                if (
                    not isinstance(field_value, list)
                    or len(field_value) != 4
                    or any(
                        isinstance(item, bool) or not isinstance(item, int) for item in field_value
                    )
                ):
                    raise ContractError(f"{role} CALVIN target bbox is malformed")
                field_value = list(field_value)
            else:
                field_value = _require_text(
                    field_value,
                    name=f"{role} CALVIN {field}",
                )
            normalized[field] = field_value
        instruction = str(normalized["instruction"])
        if (
            hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            != normalized["instruction_sha256"]
        ):
            raise ContractError(f"{role} CALVIN instruction digest changed")
        variant_bindings.append(normalized)
    binding["variants"] = variant_bindings
    return binding


def _public_result_index(
    report: Mapping[str, Any],
    *,
    role: str,
) -> dict[tuple[str, str, int, int], Mapping[str, Any]]:
    section = _mapping(report.get("public_vl_retention"), name=f"{role} public retention")
    if section.get("enabled") is not True:
        raise ContractError(f"{role} fixed-X public retention is not enabled")
    rows = section.get("results")
    if not isinstance(rows, list) or not rows:
        raise ContractError(f"{role} fixed-X report has no public results")
    result: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    for value in rows:
        row = _mapping(value, name=f"{role} public result")
        family = _require_text(row.get("family"), name=f"{role} public family")
        record_id = _require_text(row.get("record_id"), name=f"{role} public record id")
        source_row = _require_nonnegative_integer(
            row.get("source_row_index"),
            name=f"{role} public source row",
        )
        source_subindex = _require_nonnegative_integer(
            row.get("source_subindex"),
            name=f"{role} public source subindex",
        )
        key = family, record_id, source_row, source_subindex
        if key in result:
            raise ContractError(f"{role} fixed-X report repeats a public result")
        result[key] = row
    return result


def _public_sample_binding(row: Mapping[str, Any], *, role: str) -> dict[str, object]:
    binding: dict[str, object] = {}
    for field in _PUBLIC_SAMPLE_BINDINGS:
        value = row.get(field)
        if field in {"image_height", "image_width", "source_row_index", "source_subindex"}:
            value = _require_nonnegative_integer(value, name=f"{role} public {field}")
        elif field.endswith("_sha256"):
            value = _require_sha256(value, name=f"{role} public {field}")
        else:
            value = _require_text(value, name=f"{role} public {field}")
        binding[field] = value
    for field in ("target_answer", "user_text"):
        text = str(binding[field])
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != binding[f"{field}_sha256"]:
            raise ContractError(f"{role} public {field} digest changed")
    return binding


def _validate_reasons(value: object, *, name: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(reason, str) or not reason.strip() for reason in value)
        or len(set(value)) != len(value)
    ):
        raise ContractError(f"{name} must be unique nonempty strings")


def _validate_calvin_required_item(value: object) -> dict[str, object]:
    item = _mapping(value, name="required visual-review item")
    fields = set(item)
    if fields == _CALVIN_NORMAL_REQUIRED_FIELDS:
        family: str | None = _require_text(item.get("family"), name="required item family")
        task_key: str | None = _require_text(item.get("task_key"), name="required item task key")
        variant_index: int | None = _require_nonnegative_integer(
            item.get("variant_index"),
            name="required item variant index",
        )
        _validate_reasons(item.get("reasons"), name="required item reasons")
    elif fields == _CALVIN_FALLBACK_REQUIRED_FIELDS:
        family = None
        task_key = None
        variant_index = None
        _require_text(item.get("reason"), name="required item reason")
    else:
        raise ContractError("required visual-review item fields differ from schema")
    partition = _require_text(item.get("partition"), name="required item partition")
    ordinal = _require_nonnegative_integer(item.get("ordinal"), name="required item ordinal")
    normalized = dict(item)
    return {
        "decision_fields": _CALVIN_DECISION_FIELDS,
        "decision_identity": {
            "family": family,
            "ordinal": ordinal,
            "partition": partition,
            "task_key": task_key,
            "variant_index": variant_index,
        },
        "family": family,
        "item": normalized,
        "item_sha256": _canonical_sha256(normalized),
        "kind": "calvin",
        "ordinal": ordinal,
        "partition": partition,
        "task_key": task_key,
        "variant_index": variant_index,
    }


def _validate_public_required_item(value: object) -> dict[str, object]:
    item = _mapping(value, name="public generation visual-review item")
    if set(item) != _PUBLIC_REQUIRED_FIELDS:
        raise ContractError("public generation review item fields differ from schema")
    family = _require_text(item.get("family"), name="public review family")
    if family != "referring":
        raise ContractError("only public referring records declare generation panels")
    record_id = _require_text(item.get("record_id"), name="public review record id")
    source_row = _require_nonnegative_integer(
        item.get("source_row_index"),
        name="public review source row",
    )
    source_subindex = _require_nonnegative_integer(
        item.get("source_subindex"),
        name="public review source subindex",
    )
    _validate_reasons(item.get("reasons"), name="public review reasons")
    normalized = dict(item)
    identity = {
        "family": family,
        "record_id": record_id,
        "source_row_index": source_row,
        "source_subindex": source_subindex,
    }
    return {
        "decision_fields": _PUBLIC_DECISION_FIELDS,
        "decision_identity": identity,
        "item": normalized,
        "item_sha256": _canonical_sha256(normalized),
        "kind": "public_generation",
        **identity,
    }


def _expected_calvin_panels(
    item: Mapping[str, object],
    *,
    reports: Mapping[str, Mapping[str, Any]],
    report_paths: Mapping[str, Path],
    item_index: int,
    snapshots: list[_FileSnapshot],
) -> list[dict[str, int | str]]:
    key = (
        _require_text(item.get("partition"), name="required CALVIN partition"),
        _require_nonnegative_integer(item.get("ordinal"), name="required CALVIN ordinal"),
    )
    groups = {
        role: _calvin_group_index(report, role=role).get(key) for role, report in reports.items()
    }
    if any(group is None for group in groups.values()):
        raise ContractError("required CALVIN review item is absent from a fixed-X report")
    bound = {
        role: _calvin_sample_binding(group, role=role)
        for role, group in groups.items()
        if group is not None
    }
    if bound["control"] != bound["candidate"]:
        raise ContractError("control and candidate CALVIN review samples differ")
    variant_index = item["variant_index"]
    if variant_index is not None:
        if not isinstance(variant_index, int) or variant_index not in (0, 1):
            raise ContractError("required CALVIN variant index must be 0 or 1")
        variants = bound["candidate"]["variants"]
        if not isinstance(variants, list):
            raise RuntimeError("validated CALVIN sample lost its variants")
        variant = _mapping(variants[variant_index], name="bound CALVIN review variant")
        task_key = str(variant["task_key"])
        target_identity = str(variant["target_identity_key"])
        family = native_vl_calvin_task_family(task_key, target_identity)
        if task_key != item["task_key"] or family != item["family"]:
            raise ContractError("required CALVIN review identity differs from fixed-X variant")
    panels = []
    for role in ("control", "candidate"):
        group = groups[role]
        if group is None:
            raise RuntimeError("validated CALVIN group became absent")
        panels.append(
            _validate_declared_visual(
                group.get("visual"),
                role=role,
                report_path=report_paths[role],
                item_index=item_index,
                sample_sha256=_sample_evidence_sha256(
                    group,
                    name=f"required item {item_index} {role} CALVIN",
                ),
                snapshots=snapshots,
            )
        )
    return panels


def _expected_public_panels(
    item: Mapping[str, object],
    *,
    reports: Mapping[str, Mapping[str, Any]],
    report_paths: Mapping[str, Path],
    item_index: int,
    snapshots: list[_FileSnapshot],
) -> list[dict[str, int | str]]:
    key = (
        _require_text(item.get("family"), name="required public family"),
        _require_text(item.get("record_id"), name="required public record id"),
        _require_nonnegative_integer(
            item.get("source_row_index"),
            name="required public source row",
        ),
        _require_nonnegative_integer(
            item.get("source_subindex"),
            name="required public source subindex",
        ),
    )
    rows = {
        role: _public_result_index(report, role=role).get(key) for role, report in reports.items()
    }
    if any(row is None for row in rows.values()):
        raise ContractError("required public review item is absent from a fixed-X report")
    bound = {
        role: _public_sample_binding(row, role=role)
        for role, row in rows.items()
        if row is not None
    }
    if bound["control"] != bound["candidate"]:
        raise ContractError("control and candidate public review samples differ")
    panels = []
    for role in ("control", "candidate"):
        row = rows[role]
        if row is None:
            raise RuntimeError("validated public result became absent")
        panels.append(
            _validate_declared_visual(
                row.get("visual"),
                role=role,
                report_path=report_paths[role],
                item_index=item_index,
                sample_sha256=_sample_evidence_sha256(
                    row,
                    name=f"required item {item_index} {role} public",
                ),
                snapshots=snapshots,
            )
        )
    return panels


def _required_visual_review(
    comparison: Mapping[str, Any],
    *,
    reports: Mapping[str, Mapping[str, Any]],
    report_paths: Mapping[str, Path],
) -> tuple[list[dict[str, object]], list[_FileSnapshot]]:
    calvin = _mapping(comparison.get("calvin"), name="comparison CALVIN gate")
    value = calvin.get("visual_review_required")
    if not isinstance(value, list) or not value:
        raise ContractError("pending comparison must contain required visual-review items")
    items = [_validate_calvin_required_item(item) for item in value]
    public_value = comparison.get(_PUBLIC_REVIEW_FIELD)
    if public_value is not None:
        if not isinstance(public_value, list) or not public_value:
            raise ContractError("public generation visual review must be a nonempty list")
        items.extend(_validate_public_required_item(item) for item in public_value)
    digests = [str(item["item_sha256"]) for item in items]
    if len(set(digests)) != len(digests):
        raise ContractError("required visual-review items must be unique")
    snapshots: list[_FileSnapshot] = []
    for item_index, item in enumerate(items):
        if item["kind"] == "calvin":
            panels = _expected_calvin_panels(
                item,
                reports=reports,
                report_paths=report_paths,
                item_index=item_index,
                snapshots=snapshots,
            )
        elif item["kind"] == "public_generation":
            panels = _expected_public_panels(
                item,
                reports=reports,
                report_paths=report_paths,
                item_index=item_index,
                snapshots=snapshots,
            )
        else:
            raise RuntimeError("validated visual-review item kind changed")
        item["expected_panels"] = panels
    return items, snapshots


def _validate_pending_comparison(comparison: Mapping[str, Any]) -> None:
    if comparison.get("schema") != COMPARISON_SCHEMA or comparison.get("status") != PENDING_STATUS:
        raise ContractError("only a current pending visual-review comparison can be finalized")
    calvin = _mapping(comparison.get("calvin"), name="comparison CALVIN gate")
    if calvin.get("numeric_status") != "PASS":
        raise ContractError("pending comparison CALVIN numeric gate is not PASS")
    checks = _mapping(calvin.get("checks"), name="comparison CALVIN checks")
    if not checks or any(value is not True for value in checks.values()):
        raise ContractError("pending comparison contains a failed CALVIN check")
    families = _mapping(comparison.get("families"), name="comparison public families")
    if set(families) != {"referring", "vqa"}:
        raise ContractError("pending comparison public family set changed")
    for family, value in families.items():
        gates = _mapping(value, name=f"comparison {family} gates")
        if any(gates.get(field) is not True for field in _FAMILY_GATE_FIELDS):
            raise ContractError(f"pending comparison contains a failed {family} gate")
        if gates.get("generation_nonregression") is not True:
            raise ContractError(
                f"pending comparison contains failed {family} generation nonregression"
            )
        generation_checks = _mapping(
            gates.get("generation_checks"),
            name=f"comparison {family} generation checks",
        )
        if not generation_checks or any(value is not True for value in generation_checks.values()):
            raise ContractError(f"pending comparison contains a failed {family} generation check")
        generation_metrics = _mapping(
            gates.get("generation_metrics"),
            name=f"comparison {family} generation metrics",
        )
        if not generation_metrics:
            raise ContractError(f"pending comparison {family} generation metrics are empty")


def _recompute_authentic_comparison(
    comparison: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    control_sha256: str,
    candidate: Mapping[str, Any],
    candidate_sha256: str,
    candidate_training: Mapping[str, Any],
    candidate_training_sha256: str,
) -> None:
    recomputed = compare_public_native_vl_retention_reports(
        control,
        candidate,
        candidate_training,
        control_report_sha256=control_sha256,
        candidate_report_sha256=candidate_sha256,
        candidate_training_report_sha256=candidate_training_sha256,
    )
    if dict(comparison) != recomputed:
        raise ContractError(
            "comparison report is not the exact authentic recomputation of its bound inputs"
        )


def _reviewer_authentication_blocker(*, reviewer: str) -> dict[str, str]:
    _require_text(reviewer, name="visual reviewer")
    raise ContractError(REVIEWER_TRUST_BLOCKER)


def _normalize_review_panels(
    value: object,
    *,
    expected: object,
) -> list[dict[str, int | str]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ContractError("each visual-review decision must bind exactly two panels")
    panels = []
    for panel_value in value:
        panel = _mapping(panel_value, name="visual-review panel")
        if set(panel) != _PANEL_FIELDS:
            raise ContractError("visual-review panel fields differ from schema")
        role = panel.get("report")
        if role not in {"control", "candidate"}:
            raise ContractError("visual-review panel report must be control or candidate")
        path = _require_text(panel.get("path"), name="visual-review panel path")
        if path != Path(path).as_posix() or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ContractError("visual-review panel path is not canonical and relative")
        digest = _require_sha256(panel.get("sha256"), name="visual-review panel digest")
        sample_digest = _require_sha256(
            panel.get("sample_sha256"),
            name="visual-review panel sample digest",
        )
        image_format = _require_text(
            panel.get("image_format"),
            name="visual-review panel image format",
        )
        image_height = _require_nonnegative_integer(
            panel.get("image_height"),
            name="visual-review panel image height",
        )
        image_width = _require_nonnegative_integer(
            panel.get("image_width"),
            name="visual-review panel image width",
        )
        if image_height == 0 or image_width == 0:
            raise ContractError("visual-review panel dimensions must be positive")
        panels.append(
            {
                "image_format": image_format,
                "image_height": image_height,
                "image_width": image_width,
                "path": path,
                "report": str(role),
                "sample_sha256": sample_digest,
                "sha256": digest,
            }
        )
    by_role = {panel["report"]: panel for panel in panels}
    if len(by_role) != 2 or set(by_role) != {"control", "candidate"}:
        raise ContractError("visual-review panels must cover each report exactly once")
    if not isinstance(expected, list) or len(expected) != 2:
        raise RuntimeError("validated required item lost its declared panel pair")
    expected_by_role = {}
    for value in expected:
        if not isinstance(value, dict) or value.get("report") not in {"control", "candidate"}:
            raise RuntimeError("validated declared panel pair is malformed")
        expected_by_role[str(value["report"])] = value
    if by_role != expected_by_role:
        raise ContractError("visual-review panels differ from fixed-X declared artifacts")
    return [by_role[role] for role in ("control", "candidate")]


def _validate_review(
    review: Mapping[str, Any],
    *,
    comparison_path: Path,
    comparison_sha256: str,
    control_path: Path,
    control_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
    candidate_training_path: Path,
    candidate_training_sha256: str,
    required_items: list[dict[str, object]],
) -> tuple[str, list[dict[str, object]]]:
    if set(review) != _REVIEW_FIELDS:
        raise ContractError("visual review fields differ from schema")
    expected_bindings = {
        "candidate_report": str(candidate_path),
        "candidate_report_sha256": candidate_sha256,
        "candidate_training_report": str(candidate_training_path),
        "candidate_training_report_sha256": candidate_training_sha256,
        "comparison_report": str(comparison_path),
        "comparison_report_sha256": comparison_sha256,
        "control_report": str(control_path),
        "control_report_sha256": control_sha256,
    }
    if any(review.get(field) != expected for field, expected in expected_bindings.items()):
        raise ContractError("visual review report bindings changed")
    if review.get("schema") != REVIEW_SCHEMA or review.get("status") != "PASS":
        raise ContractError("visual review must be one PASS review with the current schema")
    reviewer = _require_text(review.get("reviewer"), name="visual reviewer")
    decisions_value = review.get("decisions")
    if not isinstance(decisions_value, list) or not decisions_value:
        raise ContractError("visual review decisions must be a nonempty list")
    expected = {str(item["item_sha256"]): item for item in required_items}
    accepted: dict[str, dict[str, object]] = {}
    for value in decisions_value:
        decision = _mapping(value, name="visual-review decision")
        digest_value = decision.get("required_visual_review_item_sha256")
        digest = _require_sha256(digest_value, name="visual-review item digest")
        if digest not in expected:
            raise ContractError("visual review decided an item not required by comparison")
        item = expected[digest]
        decision_fields = item["decision_fields"]
        if not isinstance(decision_fields, set) or set(decision) != decision_fields:
            raise ContractError("visual-review decision fields differ from schema")
        if digest in accepted:
            raise ContractError("visual review contains more than one decision for an item")
        identity = item["decision_identity"]
        if not isinstance(identity, Mapping):
            raise RuntimeError("validated required item lost its decision identity")
        if any(decision.get(field) != expected_value for field, expected_value in identity.items()):
            raise ContractError("visual-review decision identity differs from required item")
        if decision.get("decision") != "accept":
            raise ContractError("visual-review finalization rejects every non-accept decision")
        finding = _require_text(decision.get("finding"), name="visual-review finding")
        panels = _normalize_review_panels(
            decision.get("panels"),
            expected=item["expected_panels"],
        )
        accepted[digest] = {
            "decision": "accept",
            "finding": finding,
            "kind": item["kind"],
            "panel_count": len(panels),
            "panels": panels,
            "panels_sha256": _canonical_sha256(panels),
            "required_visual_review_item_sha256": digest,
        }
    if set(accepted) != set(expected):
        raise ContractError("visual review must decide every required item exactly once")
    return reviewer, [accepted[str(item["item_sha256"])] for item in required_items]


def finalize_public_native_vl_retention_review(
    *,
    comparison_report_path: Path,
    comparison_report_sha256: str,
    control_report_path: Path,
    control_report_sha256: str,
    candidate_report_path: Path,
    candidate_report_sha256: str,
    candidate_training_report_path: Path,
    candidate_training_report_sha256: str,
    visual_review_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Validate and durably publish one visual-review-only PASS decision."""

    output = output_path.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    comparison_snapshot, comparison = _load_bound_json(
        comparison_report_path,
        expected_sha256=comparison_report_sha256,
        name="comparison report",
    )
    control_snapshot, control = _load_bound_json(
        control_report_path,
        expected_sha256=control_report_sha256,
        name="control report",
    )
    candidate_snapshot, candidate = _load_bound_json(
        candidate_report_path,
        expected_sha256=candidate_report_sha256,
        name="candidate report",
    )
    training_snapshot, candidate_training = _load_bound_json(
        candidate_training_report_path,
        expected_sha256=candidate_training_report_sha256,
        name="candidate training report",
    )
    review_snapshot, review = _load_unbound_json(
        visual_review_path,
        name="visual review",
    )
    top_level_snapshots = {
        "candidate report": candidate_snapshot,
        "candidate training report": training_snapshot,
        "comparison report": comparison_snapshot,
        "control report": control_snapshot,
        "visual review": review_snapshot,
    }
    evidence_paths = {snapshot.path for snapshot in top_level_snapshots.values()}
    evidence_inodes = {
        (snapshot.device, snapshot.inode) for snapshot in top_level_snapshots.values()
    }
    if (
        len(evidence_paths) != len(top_level_snapshots)
        or len(evidence_inodes) != len(top_level_snapshots)
        or _absolute_lexical_path(output, name="output") in evidence_paths
    ):
        raise ContractError(
            "comparison, reports, training, review, and output must be distinct files"
        )
    if control_snapshot.sha256 == candidate_snapshot.sha256:
        raise ContractError("control and candidate reports must be distinct evidence")
    _recompute_authentic_comparison(
        comparison,
        control=control,
        control_sha256=control_snapshot.sha256,
        candidate=candidate,
        candidate_sha256=candidate_snapshot.sha256,
        candidate_training=candidate_training,
        candidate_training_sha256=training_snapshot.sha256,
    )
    _validate_pending_comparison(comparison)
    report_paths = {
        "candidate": candidate_snapshot.path,
        "control": control_snapshot.path,
    }
    reports = {"candidate": candidate, "control": control}
    required_items, panel_snapshots = _required_visual_review(
        comparison,
        reports=reports,
        report_paths=report_paths,
    )
    reviewer, accepted = _validate_review(
        review,
        comparison_path=comparison_snapshot.path,
        comparison_sha256=comparison_snapshot.sha256,
        control_path=control_snapshot.path,
        control_sha256=control_snapshot.sha256,
        candidate_path=candidate_snapshot.path,
        candidate_sha256=candidate_snapshot.sha256,
        candidate_training_path=training_snapshot.path,
        candidate_training_sha256=training_snapshot.sha256,
        required_items=required_items,
    )
    checked: set[tuple[int, int]] = set()
    snapshots_to_recheck = list(top_level_snapshots.items())
    snapshots_to_recheck.extend(("declared visual panel", panel) for panel in panel_snapshots)
    for name, snapshot in snapshots_to_recheck:
        identity = snapshot.device, snapshot.inode
        if identity in checked:
            continue
        _assert_snapshot_unchanged(snapshot, name=name)
        checked.add(identity)
    reviewer_authentication = _reviewer_authentication_blocker(reviewer=reviewer)
    item_bindings = [
        {
            "decision": accepted[index]["decision"],
            "kind": accepted[index]["kind"],
            "panel_count": accepted[index]["panel_count"],
            "panels": accepted[index]["panels"],
            "panels_sha256": accepted[index]["panels_sha256"],
            "required_visual_review_item_sha256": item["item_sha256"],
        }
        for index, item in enumerate(required_items)
    ]
    finalized: dict[str, object] = {
        "bindings": {
            "candidate_report": {
                "path": str(candidate_snapshot.path),
                "sha256": candidate_snapshot.sha256,
            },
            "candidate_training_report": {
                "path": str(training_snapshot.path),
                "sha256": training_snapshot.sha256,
            },
            "comparison_report": {
                "path": str(comparison_snapshot.path),
                "sha256": comparison_snapshot.sha256,
            },
            "control_report": {
                "path": str(control_snapshot.path),
                "sha256": control_snapshot.sha256,
            },
            "visual_review": {
                "path": str(review_snapshot.path),
                "sha256": review_snapshot.sha256,
            },
        },
        "required_visual_review": {
            "all_decisions": "accept",
            "item_count": len(required_items),
            "items": item_bindings,
            "items_sha256": _canonical_sha256([item["item"] for item in required_items]),
            "panel_count": sum(
                _require_nonnegative_integer(
                    decision["panel_count"],
                    name="validated required-item panel count",
                )
                for decision in accepted
            ),
        },
        "reviewer": reviewer,
        "reviewer_authentication": reviewer_authentication,
        "schema": FINALIZED_SCHEMA,
        "scope": "public_native_vl_retention_visual_review_only",
        "status": "PASS",
    }
    write_text_durable_exclusive(
        output,
        json.dumps(finalized, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return finalized


def main() -> None:
    args = _parse_args()
    report = finalize_public_native_vl_retention_review(
        comparison_report_path=args.comparison_report,
        comparison_report_sha256=args.comparison_report_sha256,
        control_report_path=args.control_report,
        control_report_sha256=args.control_report_sha256,
        candidate_report_path=args.candidate_report,
        candidate_report_sha256=args.candidate_report_sha256,
        candidate_training_report_path=args.candidate_training_report,
        candidate_training_report_sha256=args.candidate_training_report_sha256,
        visual_review_path=args.visual_review,
        output_path=args.output,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
