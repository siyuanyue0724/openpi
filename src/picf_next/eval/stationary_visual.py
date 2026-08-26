"""Hash-bound visual review for stationary object-posterior replay."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

STATIONARY_VISUAL_ARTIFACTS_SCHEMA: Final = "picf-next.stationary-replay-visual-artifacts.v3"
STATIONARY_VISUAL_REVIEW_SCHEMA: Final = "picf-next.stationary-replay-visual-review.v1"

_SPLITS: Final = ("validation", "heldout")
_PREFIXES: Final = (0, 8, 32, 128)
_RANKS: Final = (0, 1)
_CAMERAS: Final = (
    "observation.images.image",
    "observation.images.wrist_image",
)
_PANELS: Final = (
    "source",
    "loss_only_target",
    "fresh_m2_discovery",
    "candidate_discovery",
    "fresh_m2_persistent_posterior",
    "candidate_persistent_posterior",
)
_REVIEW_CHECKS: Final = {
    "all_manifest_artifacts_reviewed",
    "all_camera_panels_legible",
    "candidate_object_identity_alignment_acceptable",
    "no_catastrophic_off_object_collapse",
    "occlusion_uncertainty_not_misrepresented_as_fresh_observation",
    "no_mask_or_identity_input_leak",
    "no_task_text_input_leak",
    "task_annotation_present_or_explicitly_independent",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _exact_dict(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from its frozen schema")
    return cast(dict[str, Any], value)


def _coverage() -> list[dict[str, object]]:
    return [
        {"split": split, "prefix_length": prefix, "rank": rank}
        for split in _SPLITS
        for prefix in _PREFIXES
        for rank in _RANKS
    ]


def validate_stationary_visual_artifacts(
    payload: object,
    *,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Validate every rendered file, hash and required coverage coordinate."""

    root = Path(evidence_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("stationary visual evidence root must be one real directory")
    manifest = _exact_dict(
        payload,
        "stationary visual artifact manifest",
        {
            "schema",
            "status",
            "candidate_checkpoint_sha256",
            "fixed_checkpoint_replay_sha256",
            "artifact_count",
            "required_split_prefix_rank_coverage",
            "artifacts",
            "artifacts_sha256",
            "mask_or_identity_visible_to_model",
            "task_text_visible_to_stationary_model",
        },
    )
    if (
        manifest["schema"] != STATIONARY_VISUAL_ARTIFACTS_SCHEMA
        or manifest["status"] != "PENDING_HUMAN_REVIEW"
        or manifest["mask_or_identity_visible_to_model"] is not False
        or manifest["task_text_visible_to_stationary_model"] is not False
    ):
        raise ValueError("stationary visual artifact contract changed")
    _sha256(manifest["candidate_checkpoint_sha256"], "visual candidate checkpoint")
    _sha256(manifest["fixed_checkpoint_replay_sha256"], "visual fixed replay")
    required = _coverage()
    if manifest["required_split_prefix_rank_coverage"] != required:
        raise ValueError("stationary visual required coverage changed")
    required_coordinates = {
        (item["split"], item["prefix_length"], item["rank"]) for item in required
    }
    artifacts = manifest["artifacts"]
    if (
        not isinstance(artifacts, list)
        or manifest["artifact_count"] != len(required)
        or len(artifacts) != len(required)
    ):
        raise ValueError("stationary visual artifact count changed")
    if manifest["artifacts_sha256"] != _canonical_sha256(artifacts):
        raise ValueError("stationary visual artifact list hash changed")

    observed: set[tuple[str, int, int]] = set()
    paths: set[str] = set()
    long_prefix_seen_occlusion_splits: set[str] = set()
    retained_long_prefix_seen_occlusion_splits: set[str] = set()
    for index, raw in enumerate(artifacts):
        artifact = _exact_dict(
            raw,
            f"stationary visual artifact {index}",
            {
                "bytes",
                "cameras",
                "global_index",
                "optimizer_step",
                "panels",
                "path",
                "prefix_length",
                "rank",
                "sha256",
                "split",
                "tasks",
                "lifecycle_targets",
            },
        )
        coordinate = (
            artifact["split"],
            artifact["prefix_length"],
            artifact["rank"],
        )
        if (
            not isinstance(artifact["split"], str)
            or not isinstance(artifact["prefix_length"], int)
            or isinstance(artifact["prefix_length"], bool)
            or not isinstance(artifact["rank"], int)
            or isinstance(artifact["rank"], bool)
            or coordinate not in required_coordinates
            or coordinate in observed
        ):
            raise ValueError("stationary visual coverage coordinate changed or duplicated")
        observed.add(cast(tuple[str, int, int], coordinate))
        if artifact["cameras"] != list(_CAMERAS) or artifact["panels"] != list(_PANELS):
            raise ValueError("stationary visual camera or panel contract changed")
        if (
            not isinstance(artifact["bytes"], int)
            or isinstance(artifact["bytes"], bool)
            or artifact["bytes"] <= 0
        ):
            raise ValueError("stationary visual artifact bytes is invalid")
        if (
            not isinstance(artifact["global_index"], int)
            or isinstance(artifact["global_index"], bool)
            or artifact["global_index"] < 0
        ):
            raise ValueError("stationary visual artifact global_index is invalid")
        if (
            not isinstance(artifact["optimizer_step"], int)
            or isinstance(artifact["optimizer_step"], bool)
            or artifact["optimizer_step"] < 0
        ):
            raise ValueError("stationary visual optimizer step is invalid")
        relative_text = artifact["path"]
        if not isinstance(relative_text, str) or relative_text in paths:
            raise ValueError("stationary visual path is invalid or duplicated")
        paths.add(relative_text)
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "visuals":
            raise ValueError("stationary visual path escaped its evidence directory")
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("stationary visual path contains an unsafe component")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or path.resolve().parent != (root / "visuals"):
            raise ValueError("stationary visual artifact is absent or not a regular local file")
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError("stationary visual artifact byte count changed")
        if _sha256_file(path) != _sha256(artifact["sha256"], "visual artifact hash"):
            raise ValueError("stationary visual artifact content changed")
        tasks = artifact["tasks"]
        if not isinstance(tasks, list):
            raise ValueError("stationary visual task annotation is malformed")
        for task in tasks:
            entry = _exact_dict(
                task,
                "stationary visual task annotation",
                {"segment_index", "task_key", "instruction"},
            )
            if (
                not isinstance(entry["segment_index"], int)
                or isinstance(entry["segment_index"], bool)
                or entry["segment_index"] < 0
                or not isinstance(entry["task_key"], str)
                or not entry["task_key"]
                or not isinstance(entry["instruction"], str)
                or not entry["instruction"]
            ):
                raise ValueError("stationary visual task annotation is invalid")
        if not tasks and "task_independent" not in relative_text:
            raise ValueError("task-free visual is not explicitly marked task-independent")
        lifecycle_targets = artifact["lifecycle_targets"]
        if not isinstance(lifecycle_targets, list) or not lifecycle_targets:
            raise ValueError("stationary visual requires a complete lifecycle inventory")
        identity_keys: set[str] = set()
        for raw_target in lifecycle_targets:
            lifecycle = _exact_dict(
                raw_target,
                "stationary visual lifecycle target",
                {
                    "candidate_posterior_existence",
                    "candidate_posterior_identity_retained",
                    "candidate_posterior_map_present",
                    "conditional_detection_supervised",
                    "conditional_detection_target",
                    "identity_key",
                    "currently_measurable",
                    "ever_measurable_before_final",
                    "last_measurable_global_index",
                    "seen_then_unmeasurable",
                    "terminal_unmeasurable_frames",
                },
            )
            identity_key = lifecycle["identity_key"]
            if (
                not isinstance(identity_key, str)
                or not identity_key
                or identity_key in identity_keys
                or not isinstance(lifecycle["currently_measurable"], bool)
                or not isinstance(lifecycle["conditional_detection_supervised"], bool)
                or not isinstance(lifecycle["ever_measurable_before_final"], bool)
                or not isinstance(lifecycle["seen_then_unmeasurable"], bool)
                or not isinstance(lifecycle["candidate_posterior_identity_retained"], bool)
                or not isinstance(lifecycle["candidate_posterior_map_present"], bool)
            ):
                raise ValueError("stationary visual lifecycle identity or flags are invalid")
            identity_keys.add(identity_key)
            terminal_unmeasurable = lifecycle["terminal_unmeasurable_frames"]
            if (
                not isinstance(terminal_unmeasurable, int)
                or isinstance(terminal_unmeasurable, bool)
                or terminal_unmeasurable < 0
                or terminal_unmeasurable > artifact["prefix_length"] + 2
            ):
                raise ValueError("stationary visual terminal unmeasurable length is invalid")
            last_measurable = lifecycle["last_measurable_global_index"]
            if last_measurable is not None and (
                not isinstance(last_measurable, int)
                or isinstance(last_measurable, bool)
                or last_measurable < 0
                or last_measurable > artifact["global_index"]
            ):
                raise ValueError("stationary visual last measurable frame is invalid")
            currently_measurable = lifecycle["currently_measurable"]
            ever_measurable = lifecycle["ever_measurable_before_final"]
            seen_then_unmeasurable = lifecycle["seen_then_unmeasurable"]
            if (
                seen_then_unmeasurable != (ever_measurable and not currently_measurable)
                or (currently_measurable and terminal_unmeasurable != 0)
                or (not currently_measurable and terminal_unmeasurable == 0)
                or (currently_measurable and last_measurable != artifact["global_index"])
                or (not currently_measurable and ever_measurable and last_measurable is None)
                or (
                    not currently_measurable
                    and ever_measurable
                    and last_measurable == artifact["global_index"]
                )
                or (
                    not currently_measurable and not ever_measurable and last_measurable is not None
                )
            ):
                raise ValueError("stationary visual lifecycle history is inconsistent")
            if lifecycle["conditional_detection_supervised"]:
                detection = lifecycle["conditional_detection_target"]
                if (
                    isinstance(detection, bool)
                    or not isinstance(detection, int | float)
                    or float(detection) not in {0.0, 1.0}
                    or currently_measurable != (float(detection) == 1.0)
                ):
                    raise ValueError(
                        "stationary visual detection target differs from current measurability"
                    )
            elif lifecycle["conditional_detection_target"] is not None or currently_measurable:
                raise ValueError("stationary visual unknown detection must be unmeasurable")
            retained = lifecycle["candidate_posterior_identity_retained"]
            map_present = lifecycle["candidate_posterior_map_present"]
            existence = lifecycle["candidate_posterior_existence"]
            if retained:
                if (
                    isinstance(existence, bool)
                    or not isinstance(existence, int | float)
                    or not 0.0 <= float(existence) <= 1.0
                ):
                    raise ValueError("stationary visual retained posterior existence is invalid")
            elif existence is not None or map_present:
                raise ValueError("stationary visual absent posterior cannot be MAP-present")
            if map_present and not retained:
                raise ValueError("stationary visual MAP posterior must retain its identity")
            if (
                artifact["prefix_length"] >= 32
                and seen_then_unmeasurable
                and terminal_unmeasurable >= 8
            ):
                split = cast(str, artifact["split"])
                long_prefix_seen_occlusion_splits.add(split)
                if retained:
                    retained_long_prefix_seen_occlusion_splits.add(split)
    if long_prefix_seen_occlusion_splits != set(_SPLITS):
        raise ValueError(
            "stationary visual evidence lacks long-prefix seen-then-unmeasurable coverage "
            "in every split"
        )
    if retained_long_prefix_seen_occlusion_splits != set(_SPLITS):
        raise ValueError(
            "stationary candidate failed to retain a long-prefix seen-then-unmeasurable identity"
        )
    return manifest


def validate_stationary_visual_review(
    payload: object,
    *,
    manifest: object,
    manifest_sha256: str,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Validate complete, ordered review of one immutable visual manifest."""

    artifacts = validate_stationary_visual_artifacts(manifest, evidence_root=evidence_root)
    review = _exact_dict(
        payload,
        "stationary visual review",
        {
            "schema",
            "status",
            "reviewer",
            "reviewed_at_utc",
            "bindings",
            "reviewed_artifacts",
            "checks",
            "failed_checks",
            "findings",
            "long_training_authorized",
        },
    )
    if review["schema"] != STATIONARY_VISUAL_REVIEW_SCHEMA:
        raise ValueError("stationary visual review schema changed")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError("stationary visual review requires a named reviewer")
    try:
        reviewed_at = datetime.fromisoformat(cast(str, review["reviewed_at_utc"]))
    except (TypeError, ValueError) as error:
        raise ValueError("stationary visual review timestamp is invalid") from error
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() != timezone.utc.utcoffset(reviewed_at):
        raise ValueError("stationary visual review timestamp must be UTC")
    bindings = _exact_dict(
        review["bindings"],
        "stationary visual review bindings",
        {
            "visual_artifacts_sha256",
            "candidate_checkpoint_sha256",
            "fixed_checkpoint_replay_sha256",
        },
    )
    if bindings != {
        "visual_artifacts_sha256": _sha256(manifest_sha256, "visual manifest SHA-256"),
        "candidate_checkpoint_sha256": artifacts["candidate_checkpoint_sha256"],
        "fixed_checkpoint_replay_sha256": artifacts["fixed_checkpoint_replay_sha256"],
    }:
        raise ValueError("stationary visual review is bound to different evidence")
    rows = review["reviewed_artifacts"]
    if not isinstance(rows, list) or len(rows) != len(artifacts["artifacts"]):
        raise ValueError("stationary visual review coverage changed")
    row_failures = []
    for index, (raw, artifact) in enumerate(zip(rows, artifacts["artifacts"], strict=True)):
        row = _exact_dict(
            raw,
            f"stationary visual review row {index}",
            {"path", "sha256", "status", "observations"},
        )
        if row["path"] != artifact["path"] or row["sha256"] != artifact["sha256"]:
            raise ValueError("stationary visual review row differs from manifest order")
        if row["status"] not in {"PASS", "FAIL"}:
            raise ValueError("stationary visual review row status is invalid")
        if not isinstance(row["observations"], str) or not row["observations"].strip():
            raise ValueError("stationary visual review row requires substantive observations")
        if row["status"] == "FAIL":
            row_failures.append(cast(str, row["path"]))
    checks = _exact_dict(review["checks"], "stationary visual review checks", _REVIEW_CHECKS)
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ValueError("stationary visual review checks must be boolean")
    expected_failed = sorted(name for name, passed in checks.items() if not passed)
    expected_failed.extend(f"artifact:{path}" for path in row_failures)
    expected_failed.sort()
    expected_status = "PASS" if not expected_failed else "FAIL"
    if (
        review["status"] != expected_status
        or review["failed_checks"] != expected_failed
        or review["long_training_authorized"] is not False
    ):
        raise ValueError("stationary visual review decision was not recomputed exactly")
    findings = review["findings"]
    if (
        not isinstance(findings, list)
        or not findings
        or any(not isinstance(item, str) or not item.strip() for item in findings)
    ):
        raise ValueError("stationary visual review requires substantive findings")
    return review
