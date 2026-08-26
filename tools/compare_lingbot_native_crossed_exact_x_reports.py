#!/usr/bin/env python3
"""Validate and compare the frozen ADR-128 held-out exact-X reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    native_vl_fixed_x_pair_geometry_metrics,
    native_vl_fixed_x_partition_summary,
    normalize_native_vl_answer,
)
from picf_next.lingbot_native.vl_cotraining import parse_native_vl_grounding_answer

INPUT_SCHEMA = "picf-next.lingbot-native-vl-fixed-x-g0.v9"
OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-crossed-exact-x-comparison.v1"
EXPECTED_PAIR_COUNT = 82
EXPECTED_UNIQUE_RECORD_COUNT = 115

_COMMON_TOP_LEVEL_FIELDS = (
    "checkpoint_model_file_sha256",
    "dataset_manifest_sha256",
    "eligible_item_count",
    "evaluation_plan_artifact_sha256",
    "evaluation_plan_file_sha256",
    "excluded_items",
    "item_limit_per_partition",
    "max_new_tokens",
    "native_vl_patch_sha256",
    "partition",
    "physical_sidecar_manifest_sha256",
    "picf_code_revision",
    "preload_tied_parameter_name",
    "processor_lattice",
    "public_vl_retention",
    "runtime_python_trees",
    "scene_evaluation",
    "schema",
    "seed",
    "selected_item_count",
    "source_commit",
    "teacher_prune",
    "tied_parameter_name",
)
_VARIANT_BINDING_FIELDS = (
    "camera_name",
    "grounding_request",
    "grounding_request_sha256",
    "instruction",
    "instruction_sha256",
    "source_episode_index",
    "source_global_index",
    "source_rgb_sha256",
    "source_state_sha256",
    "target_answer",
    "target_answer_sha256",
    "target_bbox_qwen_xyxy",
    "target_identity_key",
    "target_label",
    "task_key",
)
_VARIANT_GEOMETRY_FIELDS = (
    "alternate_target_center_hit",
    "alternate_target_iou",
    "diagonal_iou_advantage",
    "own_only_center_hit",
    "own_target_center_hit",
    "own_target_iou",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--candidate-report-sha256", required=True)
    parser.add_argument("--candidate-visual-root", required=True, type=Path)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--control-report-sha256", required=True)
    parser.add_argument("--control-visual-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"ADR-128 exact-X {name} must be one lowercase SHA-256")
    return value


def _load_report(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    expected = _sha256(expected_sha256, name="report digest")
    if path.is_symlink() or not path.is_file():
        raise ContractError("ADR-128 exact-X input must be a regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError("ADR-128 exact-X report digest changed")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("ADR-128 exact-X input is invalid JSON") from error
    if not isinstance(value, dict):
        raise ContractError("ADR-128 exact-X input must be a mapping")
    return cast(dict[str, Any], value)


def _validate_visual_file(
    visual: Mapping[str, Any],
    *,
    report_root: Path,
) -> None:
    relative = Path(_text(visual.get("file"), name="visual file"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError("ADR-128 exact-X visual path escapes its report root")
    try:
        root = report_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ContractError("ADR-128 exact-X visual root must be an existing directory") from error
    if not root.is_dir():
        raise ContractError("ADR-128 exact-X visual root must be an existing directory")
    unresolved = root / relative
    if any(
        (root.joinpath(*relative.parts[:index])).is_symlink()
        for index in range(1, len(relative.parts) + 1)
    ):
        raise ContractError("ADR-128 exact-X visual must be a regular file")
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ContractError("ADR-128 exact-X visual path escapes its report root") from error
    if not path.is_file():
        raise ContractError("ADR-128 exact-X visual must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != _sha256(visual.get("sha256"), name="visual digest"):
        raise ContractError("ADR-128 exact-X visual digest changed")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"ADR-128 exact-X {name} must be a mapping")
    return value


def _list(value: object, *, name: str, length: int | None = None) -> list[Any]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = "" if length is None else f" of length {length}"
        raise ContractError(f"ADR-128 exact-X {name} must be a list{suffix}")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ContractError(f"ADR-128 exact-X {name} must be nonempty text")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"ADR-128 exact-X {name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"ADR-128 exact-X {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"ADR-128 exact-X {name} must be finite")
    return result


def _bbox(value: object, *, name: str) -> tuple[int, int, int, int]:
    raw = _list(value, name=name, length=4)
    result = cast(
        tuple[int, int, int, int],
        tuple(_integer(item, name=f"{name} coordinate") for item in raw),
    )
    if any(item > 1000 for item in result) or result[0] >= result[2] or result[1] >= result[3]:
        raise ContractError(f"ADR-128 exact-X {name} is outside normalized Qwen geometry")
    return result


def _require_text_digest(row: Mapping[str, Any], field: str) -> None:
    value = _text(row.get(field), name=field)
    if row.get(f"{field}_sha256") != hashlib.sha256(value.encode("utf-8")).hexdigest():
        raise ContractError(f"ADR-128 exact-X {field} digest changed")


def _record_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(
        tuple(value) if isinstance(value := row.get(field), list) else value
        for field in _VARIANT_BINDING_FIELDS
    )


def _prediction_signature(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        row.get("generated_text"),
        row.get("generated_bbox_qwen_xyxy"),
        row.get("generated_label"),
        row.get("generated_label_present"),
        row.get("generated_label_schema_valid"),
        row.get("normalized_label_exact_match"),
    )


def _validate_variant(row: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    for field in ("grounding_request", "instruction", "target_answer"):
        _require_text_digest(row, field)
    for field in ("source_rgb_sha256", "source_state_sha256"):
        _sha256(row.get(field), name=field)
    _integer(row.get("source_episode_index"), name="source episode")
    _integer(row.get("source_global_index"), name="source global index")
    for field in ("camera_name", "target_identity_key", "target_label", "task_key"):
        _text(row.get(field), name=field)

    target_bbox = _bbox(row.get("target_bbox_qwen_xyxy"), name="target bbox")
    target = parse_native_vl_grounding_answer(_text(row.get("target_answer"), name="target answer"))
    if (
        not target.schema_valid
        or not target.label_schema_valid
        or target.bbox_qwen_xyxy != target_bbox
        or target.generated_label != row.get("target_label")
    ):
        raise ContractError("ADR-128 exact-X target serialization changed")

    generated_text = _text(row.get("generated_text"), name="generated text")
    generated = parse_native_vl_grounding_answer(generated_text)
    prediction = generated.bbox_qwen_xyxy
    expected_bbox = None if prediction is None else list(prediction)
    normalized_exact = (
        generated.label_schema_valid
        and generated.generated_label is not None
        and normalize_native_vl_answer(generated.generated_label)
        == normalize_native_vl_answer(_text(row.get("target_label"), name="target label"))
    )
    if (
        row.get("generated_bbox_qwen_xyxy") != expected_bbox
        or row.get("generated_bbox_schema_valid") is not generated.schema_valid
        or row.get("generated_label") != generated.generated_label
        or row.get("generated_label_present") is not generated.label_present
        or row.get("generated_label_schema_valid") is not generated.label_schema_valid
        or row.get("normalized_label_exact_match") is not normalized_exact
    ):
        raise ContractError("ADR-128 exact-X generated answer was not parsed faithfully")
    return prediction


def _validate_arm(
    report: Mapping[str, Any],
    *,
    arm: str,
    visual_root: Path | None,
) -> dict[str, Any]:
    if report.get("schema") != INPUT_SCHEMA:
        raise ContractError("ADR-128 exact-X report schema changed")
    exact = _mapping(report.get("crossed_exact_x_evaluation"), name=f"{arm} exact section")
    if exact.get("enabled") is not True:
        raise ContractError("ADR-128 exact-X evaluation is disabled")
    _sha256(exact.get("evaluation_plan_artifact_sha256"), name="exact plan artifact")
    _sha256(exact.get("evaluation_plan_file_sha256"), name="exact plan file")
    _finite(exact.get("elapsed_seconds"), name=f"{arm} exact elapsed seconds")

    rows = _list(exact.get("results"), name=f"{arm} exact results", length=EXPECTED_PAIR_COUNT)
    normalized_rows: list[dict[str, Any]] = []
    unique: dict[tuple[object, ...], Mapping[str, Any]] = {}
    predictions: dict[tuple[object, ...], tuple[object, ...]] = {}
    pair_keys: set[str] = set()
    pair_bindings: list[tuple[object, ...]] = []
    for raw_pair in rows:
        pair = _mapping(raw_pair, name=f"{arm} exact pair")
        pair_key = _sha256(pair.get("pair_key"), name="pair key")
        if pair_key in pair_keys:
            raise ContractError("ADR-128 exact-X pair key is duplicated")
        pair_keys.add(pair_key)
        variants = _list(pair.get("variants"), name="pair variants", length=2)
        normalized_variants: list[dict[str, Any]] = []
        pair_predictions = []
        pair_targets = []
        variant_bindings = []
        for raw_variant in variants:
            variant = _mapping(raw_variant, name=f"{arm} exact variant")
            prediction = _validate_variant(variant)
            target = _bbox(variant.get("target_bbox_qwen_xyxy"), name="target bbox")
            key = _record_key(variant)
            signature = _prediction_signature(variant)
            previous = predictions.setdefault(key, signature)
            if previous != signature:
                raise ContractError("ADR-128 exact-X repeated record generation changed")
            unique.setdefault(key, variant)
            pair_predictions.append(prediction)
            pair_targets.append(target)
            variant_bindings.append(key)

        recomputed = native_vl_fixed_x_pair_geometry_metrics(
            cast(tuple[Any, Any], tuple(pair_predictions)),
            cast(tuple[Any, Any], tuple(pair_targets)),
        )
        reported_pair = _mapping(pair.get("pair_metrics"), name="pair metrics")
        expected_pair = {key: value for key, value in recomputed.items() if key != "variants"}
        if dict(reported_pair) != expected_pair:
            raise ContractError("ADR-128 exact-X pair metrics were not recomputed")
        geometry = _list(recomputed.get("variants"), name="variant geometry", length=2)
        for raw_variant, raw_geometry in zip(variants, geometry, strict=True):
            variant = _mapping(raw_variant, name="exact variant")
            metric = _mapping(raw_geometry, name="variant geometry")
            if any(variant.get(field) != metric.get(field) for field in _VARIANT_GEOMETRY_FIELDS):
                raise ContractError("ADR-128 exact-X variant geometry was not recomputed")
            normalized_variants.append({**dict(variant), **dict(metric)})

        visual = _mapping(pair.get("visual"), name="pair visual")
        _text(visual.get("file"), name="visual file")
        _sha256(visual.get("sha256"), name="visual digest")
        if visual_root is not None:
            _validate_visual_file(visual, report_root=visual_root)
        normalized_rows.append(
            {
                "pair_key": pair_key,
                "pair_metrics": expected_pair,
                "variants": normalized_variants,
                "visual": dict(visual),
            }
        )
        pair_bindings.append((pair_key, tuple(variant_bindings), visual.get("file")))

    if (
        len(unique) != EXPECTED_UNIQUE_RECORD_COUNT
        or exact.get("unique_record_count") != EXPECTED_UNIQUE_RECORD_COUNT
    ):
        raise ContractError("ADR-128 exact-X unique record cardinality changed")
    summary = native_vl_fixed_x_partition_summary(normalized_rows)
    if dict(_mapping(exact.get("summary"), name=f"{arm} exact summary")) != summary:
        raise ContractError("ADR-128 exact-X summary was not recomputed")
    return {
        "pair_bindings": pair_bindings,
        "rows": normalized_rows,
        "summary": summary,
        "unique": unique,
    }


def _unique_metrics(rows: Mapping[tuple[object, ...], Mapping[str, Any]]) -> dict[str, object]:
    values = list(rows.values())
    return {
        "center_hit_count": sum(row.get("own_target_center_hit") is True for row in values),
        "label_exact_count": sum(row.get("normalized_label_exact_match") is True for row in values),
        "mean_own_target_iou": sum(
            _finite(row.get("own_target_iou"), name="unique own-target IoU") for row in values
        )
        / len(values),
        "record_count": len(values),
    }


def _stratified_metrics(
    candidate: Mapping[tuple[object, ...], Mapping[str, Any]],
    control: Mapping[tuple[object, ...], Mapping[str, Any]],
    *,
    fields: tuple[str, ...],
) -> dict[str, object]:
    buckets: dict[tuple[str, ...], dict[str, dict[tuple[object, ...], Mapping[str, Any]]]] = (
        defaultdict(lambda: {"candidate": {}, "control": {}})
    )
    for arm, rows in (("candidate", candidate), ("control", control)):
        for key, row in rows.items():
            bucket = tuple(_text(row.get(field), name=field) for field in fields)
            buckets[bucket][arm][key] = row
    result = {}
    for bucket in sorted(buckets):
        arm_rows = buckets[bucket]
        if set(arm_rows["candidate"]) != set(arm_rows["control"]):
            raise ContractError("ADR-128 exact-X stratum bindings differ across arms")
        name = " | ".join(bucket)
        candidate_metrics = _unique_metrics(arm_rows["candidate"])
        control_metrics = _unique_metrics(arm_rows["control"])
        result[name] = {
            "candidate": candidate_metrics,
            "candidate_minus_control": {
                "center_hit_count": candidate_metrics["center_hit_count"]
                - control_metrics["center_hit_count"],
                "label_exact_count": candidate_metrics["label_exact_count"]
                - control_metrics["label_exact_count"],
                "mean_own_target_iou": candidate_metrics["mean_own_target_iou"]
                - control_metrics["mean_own_target_iou"],
            },
            "control": control_metrics,
        }
    return result


def _hit_change_rows(
    candidate: Mapping[tuple[object, ...], Mapping[str, Any]],
    control: Mapping[tuple[object, ...], Mapping[str, Any]],
    *,
    candidate_value: bool,
) -> list[dict[str, object]]:
    rows = []
    for key in sorted(candidate):
        candidate_hit = candidate[key].get("own_target_center_hit") is True
        control_hit = control[key].get("own_target_center_hit") is True
        if candidate_hit is candidate_value and control_hit is not candidate_value:
            row = candidate[key]
            rows.append(
                {
                    "camera_name": row["camera_name"],
                    "source_global_index": row["source_global_index"],
                    "target_identity_key": row["target_identity_key"],
                    "task_key": row["task_key"],
                }
            )
    return rows


def _validate_qwen_restore(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    candidate_restore = _mapping(candidate.get("qwen_restore"), name="candidate Qwen restore")
    control_restore = _mapping(control.get("qwen_restore"), name="control Qwen restore")
    if candidate_restore.get("model_revision") != control_restore.get("model_revision"):
        raise ContractError("ADR-128 exact-X restored model revisions differ")
    for arm, restore in (("candidate", candidate_restore), ("control", control_restore)):
        load = _mapping(restore.get("load_result"), name=f"{arm} Qwen load result")
        if load.get("missing_keys") != [] or load.get("unexpected_keys") != []:
            raise ContractError("ADR-128 exact-X restored Qwen load is incomplete")
        hashes = _mapping(restore.get("model_file_sha256"), name=f"{arm} Qwen hashes")
        if not hashes or any(
            _sha256(value, name="Qwen file digest") != value for value in hashes.values()
        ):
            raise ContractError("ADR-128 exact-X restored Qwen hashes are incomplete")
    if candidate_restore.get("model_file_sha256") == control_restore.get("model_file_sha256"):
        raise ContractError("ADR-128 exact-X candidate/control Qwen weights are identical")


def compare_lingbot_native_crossed_exact_x_reports(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    candidate_visual_root: Path | None = None,
    control_visual_root: Path | None = None,
) -> dict[str, object]:
    """Recompute exact-X evidence and report strict scientific stop conditions."""

    if (candidate_visual_root is None) != (control_visual_root is None):
        raise ContractError("ADR-128 exact-X visual roots must be supplied for both arms")
    if any(candidate.get(field) != control.get(field) for field in _COMMON_TOP_LEVEL_FIELDS):
        raise ContractError("ADR-128 exact-X common evaluation binding changed")
    candidate_exact = _mapping(candidate.get("crossed_exact_x_evaluation"), name="candidate exact")
    control_exact = _mapping(control.get("crossed_exact_x_evaluation"), name="control exact")
    for field in ("evaluation_plan_artifact_sha256", "evaluation_plan_file_sha256"):
        if candidate_exact.get(field) != control_exact.get(field):
            raise ContractError("ADR-128 exact-X candidate/control plans differ")
    _validate_qwen_restore(candidate, control)
    candidate_arm = _validate_arm(
        candidate,
        arm="candidate",
        visual_root=candidate_visual_root,
    )
    control_arm = _validate_arm(
        control,
        arm="control",
        visual_root=control_visual_root,
    )
    if candidate_arm["pair_bindings"] != control_arm["pair_bindings"]:
        raise ContractError("ADR-128 exact-X candidate/control pair bindings differ")
    candidate_unique = cast(dict[tuple[object, ...], Mapping[str, Any]], candidate_arm["unique"])
    control_unique = cast(dict[tuple[object, ...], Mapping[str, Any]], control_arm["unique"])
    if set(candidate_unique) != set(control_unique):
        raise ContractError("ADR-128 exact-X candidate/control record bindings differ")

    candidate_unique_metrics = _unique_metrics(candidate_unique)
    control_unique_metrics = _unique_metrics(control_unique)
    by_target_camera = _stratified_metrics(
        candidate_unique,
        control_unique,
        fields=("target_identity_key", "camera_name"),
    )
    by_task = _stratified_metrics(candidate_unique, control_unique, fields=("task_key",))
    zero_hit_tasks = [
        key
        for key, value in by_task.items()
        if cast(Mapping[str, Any], value)["candidate"]["center_hit_count"] == 0
    ]
    candidate_summary = cast(Mapping[str, Any], candidate_arm["summary"])
    exact_swap_failure_count = EXPECTED_PAIR_COUNT - _integer(
        candidate_summary.get("bidirectional_own_only_center_hit_count"),
        name="candidate bidirectional count",
    )
    switch_key = "part/table/switch_link | static"
    switch = _mapping(by_target_camera.get(switch_key), name="switch/static stratum")
    switch_candidate = _mapping(switch.get("candidate"), name="candidate switch/static")
    switch_miss_count = _integer(
        switch_candidate.get("record_count"), name="switch count"
    ) - _integer(switch_candidate.get("center_hit_count"), name="switch hits")
    stop_rule_triggered = exact_swap_failure_count > 0 or switch_miss_count > 0

    return {
        "candidate_only_center_hit_records": _hit_change_rows(
            candidate_unique, control_unique, candidate_value=True
        ),
        "common_pair_count": EXPECTED_PAIR_COUNT,
        "common_unique_record_count": EXPECTED_UNIQUE_RECORD_COUNT,
        "control_only_center_hit_records": _hit_change_rows(
            candidate_unique, control_unique, candidate_value=False
        ),
        "evidence_integrity_status": "PASS",
        "long_run_authorized": False,
        "pair_weighted": {
            "candidate": candidate_arm["summary"],
            "candidate_minus_control": {
                "bidirectional_own_only_center_hit_count": candidate_summary[
                    "bidirectional_own_only_center_hit_count"
                ]
                - control_arm["summary"]["bidirectional_own_only_center_hit_count"],
                "mean_diagonal_iou_advantage": candidate_summary["mean_diagonal_iou_advantage"]
                - control_arm["summary"]["mean_diagonal_iou_advantage"],
                "mean_own_target_iou": candidate_summary["mean_own_target_iou"]
                - control_arm["summary"]["mean_own_target_iou"],
                "own_target_center_hit_count": candidate_summary["own_target_center_hit_count"]
                - control_arm["summary"]["own_target_center_hit_count"],
            },
            "control": control_arm["summary"],
        },
        "schema": OUTPUT_SCHEMA,
        "scientific_gate": {
            "exact_prompt_pair_failure_count": exact_swap_failure_count,
            "status": "FAIL" if stop_rule_triggered else "PASS",
            "stop_rule_triggered": stop_rule_triggered,
            "switch_static_miss_count": switch_miss_count,
            "zero_hit_task_keys": zero_hit_tasks,
        },
        "unique_record": {
            "by_target_camera": by_target_camera,
            "by_task": by_task,
            "candidate": candidate_unique_metrics,
            "candidate_minus_control": {
                "center_hit_count": candidate_unique_metrics["center_hit_count"]
                - control_unique_metrics["center_hit_count"],
                "label_exact_count": candidate_unique_metrics["label_exact_count"]
                - control_unique_metrics["label_exact_count"],
                "mean_own_target_iou": candidate_unique_metrics["mean_own_target_iou"]
                - control_unique_metrics["mean_own_target_iou"],
            },
            "control": control_unique_metrics,
        },
        "visual_file_integrity_status": (
            "PASS" if candidate_visual_root is not None else "NOT_REHASHED"
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    candidate = _load_report(
        args.candidate_report,
        expected_sha256=args.candidate_report_sha256,
    )
    control = _load_report(
        args.control_report,
        expected_sha256=args.control_report_sha256,
    )
    report = compare_lingbot_native_crossed_exact_x_reports(
        candidate,
        control,
        candidate_visual_root=args.candidate_visual_root,
        control_visual_root=args.control_visual_root,
    )
    report["input_reports"] = {
        "candidate_sha256": args.candidate_report_sha256,
        "control_sha256": args.control_report_sha256,
    }
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
