#!/usr/bin/env python3
"""Independently validate paired factual and mediator-required G3 cold reports.

The validator does not import the evaluator and never trusts its serialized
scene aggregates.  It recomputes prompt effects from the raw RMS arrays, then
rebuilds scene and cross-rank partition statistics before applying the frozen
62.5 percent cold-action gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

REPORT_SCHEMA = "picf-next.ltop-g3-evaluation-phase.v1"
OUTPUT_SCHEMA = "picf-next.ltop-g3-cold-action-evidence-validation.v1"
PARTITIONS = ("validation", "heldout")
EXPECTED_SCENES_PER_RANK_PARTITION = 4
EXPECTED_PROMPTS_PER_SCENE = 2
POSITIVE_SAMPLE_FRACTION = 0.625
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12
_RMS_FIELDS = (
    "replay_floor_rms",
    "factual_target_effect_rms",
    "factual_distractor_effect_rms",
    "blocked_target_effect_rms",
    "blocked_distractor_effect_rms",
)
_DERIVED_VECTOR_FIELDS = (
    "factual_target_minus_distractor",
    "blocked_target_minus_distractor",
    "blocked_path_difference_in_differences",
)


class ValidationInputError(ValueError):
    """Raised when a cold report cannot be interpreted safely."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factual-report", type=Path, required=True)
    parser.add_argument("--mediator-required-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationInputError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationInputError(f"{name} must be a JSON array")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationInputError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationInputError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationInputError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationInputError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationInputError(f"{name} must be finite")
    return result


def _string_vector(value: object, *, name: str) -> list[str]:
    values = _list(value, name=name)
    if not values:
        raise ValidationInputError(f"{name} cannot be empty")
    return [_string(item, name=f"{name}[{index}]") for index, item in enumerate(values)]


def _number_vector(value: object, *, name: str) -> list[float]:
    values = _list(value, name=name)
    if not values:
        raise ValidationInputError(f"{name} cannot be empty")
    return [_number(item, name=f"{name}[{index}]") for index, item in enumerate(values)]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValidationInputError("cannot compute a mean over an empty sequence")
    return math.fsum(values) / len(values)


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    )


def _check_number(
    failures: list[str],
    *,
    context: str,
    field: str,
    declared: float,
    recomputed: float,
) -> None:
    if not _close(declared, recomputed):
        failures.append(
            f"{context}: serialized {field} {declared!r} differs from recomputed {recomputed!r}"
        )


def _check_vector(
    failures: list[str],
    *,
    context: str,
    field: str,
    declared: Sequence[float],
    recomputed: Sequence[float],
) -> None:
    if len(declared) != len(recomputed) or any(
        not _close(left, right) for left, right in zip(declared, recomputed, strict=True)
    ):
        failures.append(f"{context}: serialized {field} differs from recomputed values")


def _check_count(
    failures: list[str],
    *,
    context: str,
    field: str,
    declared: int,
    recomputed: int,
    sample_count: int,
) -> None:
    if declared > sample_count:
        failures.append(f"{context}: {field} {declared} exceeds sample_count {sample_count}")
    if declared != recomputed:
        failures.append(
            f"{context}: serialized {field} {declared} differs from recomputed {recomputed}"
        )


def _aggregate(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factual_means = [float(unit["mean_factual_target_minus_distractor"]) for unit in units]
    did_means = [float(unit["mean_blocked_path_difference_in_differences"]) for unit in units]
    replay_values = [float(value) for unit in units for value in unit["replay_floor_rms"]]
    sample_count = sum(int(unit["sample_count"]) for unit in units)
    positive_factual = sum(int(unit["positive_factual_count"]) for unit in units)
    positive_did = sum(int(unit["positive_blocked_path_did_count"]) for unit in units)
    return {
        "mean_factual_target_minus_distractor": _mean(factual_means),
        "mean_blocked_path_difference_in_differences": _mean(did_means),
        "positive_factual_count": positive_factual,
        "positive_blocked_path_did_count": positive_did,
        "positive_factual_fraction": positive_factual / sample_count,
        "positive_blocked_path_did_fraction": positive_did / sample_count,
        "sample_count": sample_count,
        "replay_floor_rms": replay_values,
        "max_replay_floor_rms": max(replay_values),
    }


def _gate_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    sample_count = int(summary["sample_count"])
    minimum = math.ceil(POSITIVE_SAMPLE_FRACTION * sample_count)
    gates = {
        "bitwise_factual_replay": float(summary["max_replay_floor_rms"]) == 0.0,
        "mean_factual_target_minus_distractor_strictly_positive": (
            float(summary["mean_factual_target_minus_distractor"]) > 0.0
        ),
        "mean_blocked_path_did_strictly_positive": (
            float(summary["mean_blocked_path_difference_in_differences"]) > 0.0
        ),
        "positive_factual_count_minimum": (int(summary["positive_factual_count"]) >= minimum),
        "positive_blocked_path_did_count_minimum": (
            int(summary["positive_blocked_path_did_count"]) >= minimum
        ),
    }
    return {
        **summary,
        "minimum_positive_count": minimum,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def _validate_prompt(
    prompt_value: object,
    *,
    context: str,
    failures: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = _mapping(prompt_value, name=context)
    prompt_name = _string(prompt.get("prompt_name"), name=f"{context}.prompt_name")
    target_identity = _string(prompt.get("target_identity"), name=f"{context}.target_identity")
    distractor_identity = _string(
        prompt.get("matched_distractor_identity"),
        name=f"{context}.matched_distractor_identity",
    )
    target_row = _integer(prompt.get("target_row"), name=f"{context}.target_row", minimum=0)
    distractor_row = _integer(
        prompt.get("matched_distractor_row"),
        name=f"{context}.matched_distractor_row",
        minimum=0,
    )
    if target_identity == distractor_identity:
        failures.append(f"{context}: target and distractor identities are identical")
    if target_row == distractor_row:
        failures.append(f"{context}: target and distractor rows are identical")

    score = _mapping(prompt.get("score"), name=f"{context}.score")
    if score.get("prompt_name") != prompt_name:
        failures.append(f"{context}: prompt_name differs between prompt and score")
    sample_keys = _string_vector(score.get("sample_keys"), name=f"{context}.score.sample_keys")
    vectors = {
        field: _number_vector(score.get(field), name=f"{context}.score.{field}")
        for field in (*_RMS_FIELDS, *_DERIVED_VECTOR_FIELDS)
    }
    lengths = {field: len(values) for field, values in vectors.items()}
    lengths["sample_keys"] = len(sample_keys)
    if len(set(lengths.values())) != 1:
        raise ValidationInputError(f"{context}.score arrays have inconsistent lengths: {lengths}")
    for field in _RMS_FIELDS:
        if any(value < 0.0 for value in vectors[field]):
            failures.append(f"{context}.score.{field} contains a negative RMS value")

    factual_delta = [
        target - distractor
        for target, distractor in zip(
            vectors["factual_target_effect_rms"],
            vectors["factual_distractor_effect_rms"],
            strict=True,
        )
    ]
    blocked_delta = [
        target - distractor
        for target, distractor in zip(
            vectors["blocked_target_effect_rms"],
            vectors["blocked_distractor_effect_rms"],
            strict=True,
        )
    ]
    did = [factual - blocked for factual, blocked in zip(factual_delta, blocked_delta, strict=True)]
    recomputed_vectors = {
        "factual_target_minus_distractor": factual_delta,
        "blocked_target_minus_distractor": blocked_delta,
        "blocked_path_difference_in_differences": did,
    }
    for field, values in recomputed_vectors.items():
        _check_vector(
            failures,
            context=f"{context}.score",
            field=field,
            declared=vectors[field],
            recomputed=values,
        )

    factual_mean = _mean(factual_delta)
    did_mean = _mean(did)
    declared_factual_mean = _number(
        score.get("mean_factual_target_minus_distractor"),
        name=f"{context}.score.mean_factual_target_minus_distractor",
    )
    declared_did_mean = _number(
        score.get("mean_blocked_path_difference_in_differences"),
        name=f"{context}.score.mean_blocked_path_difference_in_differences",
    )
    _check_number(
        failures,
        context=f"{context}.score",
        field="mean_factual_target_minus_distractor",
        declared=declared_factual_mean,
        recomputed=factual_mean,
    )
    _check_number(
        failures,
        context=f"{context}.score",
        field="mean_blocked_path_difference_in_differences",
        declared=declared_did_mean,
        recomputed=did_mean,
    )

    replay_floor = vectors["replay_floor_rms"]
    positive_factual = sum(
        value > floor for value, floor in zip(factual_delta, replay_floor, strict=True)
    )
    positive_did = sum(value > floor for value, floor in zip(did, replay_floor, strict=True))
    declared_positive_factual = _integer(
        score.get("positive_factual_count"),
        name=f"{context}.score.positive_factual_count",
        minimum=0,
    )
    declared_positive_did = _integer(
        score.get("positive_blocked_path_did_count"),
        name=f"{context}.score.positive_blocked_path_did_count",
        minimum=0,
    )
    _check_count(
        failures,
        context=f"{context}.score",
        field="positive_factual_count",
        declared=declared_positive_factual,
        recomputed=positive_factual,
        sample_count=len(sample_keys),
    )
    _check_count(
        failures,
        context=f"{context}.score",
        field="positive_blocked_path_did_count",
        declared=declared_positive_did,
        recomputed=positive_did,
        sample_count=len(sample_keys),
    )

    summary = {
        "prompt_name": prompt_name,
        "sample_keys": sample_keys,
        "sample_count": len(sample_keys),
        "replay_floor_rms": replay_floor,
        "max_replay_floor_rms": max(replay_floor),
        "mean_factual_target_minus_distractor": factual_mean,
        "mean_blocked_path_difference_in_differences": did_mean,
        "positive_factual_count": positive_factual,
        "positive_blocked_path_did_count": positive_did,
        "target_identity": target_identity,
        "matched_distractor_identity": distractor_identity,
        "target_row": target_row,
        "matched_distractor_row": distractor_row,
    }
    layout = {
        "prompt_name": prompt_name,
        "score_sample_keys": sample_keys,
        "target_identity": target_identity,
        "matched_distractor_identity": distractor_identity,
        "target_row": target_row,
        "matched_distractor_row": distractor_row,
    }
    return summary, layout


def _validate_scene(
    scene_value: object,
    *,
    context: str,
    failures: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = _mapping(scene_value, name=context)
    item_id = _string(scene.get("item_id"), name=f"{context}.item_id")
    sample_key = _string(scene.get("sample_key"), name=f"{context}.sample_key")
    prompt_values = _list(scene.get("prompts"), name=f"{context}.prompts")
    if len(prompt_values) != EXPECTED_PROMPTS_PER_SCENE:
        raise ValidationInputError(
            f"{context}.prompts must contain exactly {EXPECTED_PROMPTS_PER_SCENE} crossed prompts"
        )
    prompts: list[dict[str, Any]] = []
    prompt_layouts: list[dict[str, Any]] = []
    for prompt_index, prompt_value in enumerate(prompt_values):
        prompt, layout = _validate_prompt(
            prompt_value,
            context=f"{context}.prompts[{prompt_index}]",
            failures=failures,
        )
        prompts.append(prompt)
        prompt_layouts.append(layout)

    first, second = prompts
    if first["prompt_name"] == second["prompt_name"]:
        failures.append(f"{context}: crossed prompt names are not unique")
    if first["sample_keys"] != second["sample_keys"]:
        failures.append(f"{context}: crossed prompts do not use the same samples")
    if (
        first["target_identity"] != second["matched_distractor_identity"]
        or second["target_identity"] != first["matched_distractor_identity"]
    ):
        failures.append(f"{context}: crossed prompt target/distractor identities do not swap")
    if (
        first["target_row"] != second["matched_distractor_row"]
        or second["target_row"] != first["matched_distractor_row"]
    ):
        failures.append(f"{context}: crossed prompt target/distractor rows do not swap")

    recomputed = _aggregate(prompts)
    score = _mapping(scene.get("score"), name=f"{context}.score")
    if score.get("prompt_name") != f"{item_id}/aggregate":
        failures.append(f"{context}.score: aggregate prompt_name is incorrect")
    declared_sample_keys = _string_vector(
        score.get("sample_keys"), name=f"{context}.score.sample_keys"
    )
    expected_sample_keys = [sample_key] * int(recomputed["sample_count"])
    if declared_sample_keys != expected_sample_keys:
        failures.append(
            f"{context}.score: aggregate sample_keys do not represent every prompt sample"
        )
    declared_replay = _number_vector(
        score.get("replay_floor_rms"), name=f"{context}.score.replay_floor_rms"
    )
    _check_vector(
        failures,
        context=f"{context}.score",
        field="replay_floor_rms",
        declared=declared_replay,
        recomputed=cast(list[float], recomputed["replay_floor_rms"]),
    )
    declared_factual_mean = _number(
        score.get("mean_factual_target_minus_distractor"),
        name=f"{context}.score.mean_factual_target_minus_distractor",
    )
    declared_did_mean = _number(
        score.get("mean_blocked_path_difference_in_differences"),
        name=f"{context}.score.mean_blocked_path_difference_in_differences",
    )
    _check_number(
        failures,
        context=f"{context}.score",
        field="mean_factual_target_minus_distractor",
        declared=declared_factual_mean,
        recomputed=float(recomputed["mean_factual_target_minus_distractor"]),
    )
    _check_number(
        failures,
        context=f"{context}.score",
        field="mean_blocked_path_difference_in_differences",
        declared=declared_did_mean,
        recomputed=float(recomputed["mean_blocked_path_difference_in_differences"]),
    )
    declared_positive_factual = _integer(
        score.get("positive_factual_count"),
        name=f"{context}.score.positive_factual_count",
        minimum=0,
    )
    declared_positive_did = _integer(
        score.get("positive_blocked_path_did_count"),
        name=f"{context}.score.positive_blocked_path_did_count",
        minimum=0,
    )
    _check_count(
        failures,
        context=f"{context}.score",
        field="positive_factual_count",
        declared=declared_positive_factual,
        recomputed=int(recomputed["positive_factual_count"]),
        sample_count=int(recomputed["sample_count"]),
    )
    _check_count(
        failures,
        context=f"{context}.score",
        field="positive_blocked_path_did_count",
        declared=declared_positive_did,
        recomputed=int(recomputed["positive_blocked_path_did_count"]),
        sample_count=int(recomputed["sample_count"]),
    )

    summary = {
        "item_id": item_id,
        "sample_key": sample_key,
        "prompt_count": len(prompts),
        "prompts": prompts,
        **recomputed,
    }
    layout = {
        "item_id": item_id,
        "sample_key": sample_key,
        "prompts": prompt_layouts,
    }
    return summary, layout


def _validate_threshold_contract(
    report: Mapping[str, Any], *, context: str, failures: list[str]
) -> None:
    thresholds = _mapping(report.get("thresholds"), name=f"{context}.thresholds")
    fraction = _number(
        thresholds.get("positive_sample_fraction_minimum"),
        name=f"{context}.thresholds.positive_sample_fraction_minimum",
    )
    if not _close(fraction, POSITIVE_SAMPLE_FRACTION):
        failures.append(
            f"{context}: positive_sample_fraction_minimum is {fraction}, expected "
            f"{POSITIVE_SAMPLE_FRACTION}"
        )
    for field in (
        "bitwise_factual_replay",
        "mean_factual_target_minus_distractor_strictly_positive",
        "mean_blocked_path_did_strictly_positive",
    ):
        if thresholds.get(field) is not True:
            failures.append(f"{context}: threshold {field} is not enabled")


def _validate_report(
    report_value: object,
    *,
    label: str,
    expected_information_set: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    report = _mapping(report_value, name=label)
    failures: list[str] = []
    if report.get("schema") != REPORT_SCHEMA:
        failures.append(f"{label}: report schema is not {REPORT_SCHEMA}")
    if report.get("mode") != "gate" or report.get("phase") != "evaluation":
        failures.append(f"{label}: report is not a cold gate evaluation")
    if report.get("evaluation_action_information_set") != expected_information_set:
        failures.append(
            f"{label}: evaluation_action_information_set is not {expected_information_set}"
        )
    if report.get("status") != "PASS":
        failures.append(f"{label}: declared report status is not PASS")
    declared_failures = _list(report.get("failures"), name=f"{label}.failures")
    if declared_failures:
        failures.append(f"{label}: declared report failures are not empty")
    _validate_threshold_contract(report, context=label, failures=failures)

    world_size = _integer(report.get("world_size"), name=f"{label}.world_size", minimum=1)
    rank_values = _list(report.get("rank_reports"), name=f"{label}.rank_reports")
    if len(rank_values) != world_size:
        raise ValidationInputError(
            f"{label}.rank_reports count {len(rank_values)} differs from world_size {world_size}"
        )
    ranks = [
        _mapping(value, name=f"{label}.rank_reports[{index}]")
        for index, value in enumerate(rank_values)
    ]
    rank_ids = [
        _integer(rank.get("rank"), name=f"{label}.rank_reports[{index}].rank", minimum=0)
        for index, rank in enumerate(ranks)
    ]
    if rank_ids != list(range(world_size)):
        raise ValidationInputError(
            f"{label}.rank_reports ranks must be ordered and contiguous from zero: {rank_ids}"
        )

    seen_item_ids: set[str] = set()
    seen_sample_keys: set[str] = set()
    seen_prompt_names: set[str] = set()
    rank_summaries: list[dict[str, Any]] = []
    layout_entries: list[dict[str, Any]] = []
    scenes_by_partition: dict[str, list[dict[str, Any]]] = {
        partition: [] for partition in PARTITIONS
    }
    for rank_id, rank in zip(rank_ids, ranks, strict=True):
        history = _list(rank.get("history"), name=f"{label}.rank[{rank_id}].history")
        if len(history) != 1:
            raise ValidationInputError(
                f"{label}.rank[{rank_id}].history must contain exactly one cold receipt"
            )
        receipt = _mapping(history[0], name=f"{label}.rank[{rank_id}].history[0]")
        rank_partitions: dict[str, Any] = {}
        for partition in PARTITIONS:
            partition_context = f"{label}.rank[{rank_id}].{partition}"
            partition_report = _mapping(receipt.get(partition), name=f"{partition_context}.report")
            scene_values = _list(partition_report.get("scenes"), name=f"{partition_context}.scenes")
            if len(scene_values) != EXPECTED_SCENES_PER_RANK_PARTITION:
                raise ValidationInputError(
                    f"{partition_context}.scenes must contain exactly "
                    f"{EXPECTED_SCENES_PER_RANK_PARTITION} scenes"
                )
            scenes: list[dict[str, Any]] = []
            for scene_index, scene_value in enumerate(scene_values):
                scene, scene_layout = _validate_scene(
                    scene_value,
                    context=f"{partition_context}.scenes[{scene_index}]",
                    failures=failures,
                )
                item_id = str(scene["item_id"])
                sample_key = str(scene["sample_key"])
                if item_id in seen_item_ids:
                    failures.append(f"{label}: duplicate scene item_id {item_id!r}")
                seen_item_ids.add(item_id)
                if sample_key in seen_sample_keys:
                    failures.append(f"{label}: duplicate scene sample_key {sample_key!r}")
                seen_sample_keys.add(sample_key)
                for prompt in scene["prompts"]:
                    prompt_name = str(prompt["prompt_name"])
                    if prompt_name in seen_prompt_names:
                        failures.append(f"{label}: duplicate prompt_name {prompt_name!r}")
                    seen_prompt_names.add(prompt_name)
                scenes.append(scene)
                scenes_by_partition[partition].append(scene)
                layout_entries.append(
                    {
                        "rank": rank_id,
                        "partition": partition,
                        "scene_index": scene_index,
                        **scene_layout,
                    }
                )
            recomputed_partition = _aggregate(scenes)
            declared_max_replay = _number(
                partition_report.get("max_replay_floor_rms"),
                name=f"{partition_context}.max_replay_floor_rms",
            )
            _check_number(
                failures,
                context=partition_context,
                field="max_replay_floor_rms",
                declared=declared_max_replay,
                recomputed=float(recomputed_partition["max_replay_floor_rms"]),
            )
            rank_partitions[partition] = {
                "scene_count": len(scenes),
                "prompt_count": sum(int(scene["prompt_count"]) for scene in scenes),
                "scenes": scenes,
                **recomputed_partition,
            }
        rank_summaries.append({"rank": rank_id, "partitions": rank_partitions})

    partition_summaries: dict[str, Any] = {}
    for partition in PARTITIONS:
        summary = _gate_summary(_aggregate(scenes_by_partition[partition]))
        partition_summaries[partition] = {
            "scene_count": len(scenes_by_partition[partition]),
            "prompt_count": sum(
                int(scene["prompt_count"]) for scene in scenes_by_partition[partition]
            ),
            **summary,
        }
        if summary["status"] != "PASS":
            gates = cast(Mapping[str, bool], summary["gates"])
            for gate, passed in gates.items():
                if not passed:
                    failures.append(f"{label}: {partition} recomputed gate failed: {gate}")

    layout = {
        "world_size": world_size,
        "capacity": report.get("capacity"),
        "entries": layout_entries,
    }
    layout_sha256 = _sha256_bytes(_canonical_json(layout).encode("ascii"))
    summary = {
        "declared_status": report.get("status"),
        "evaluation_action_information_set": expected_information_set,
        "world_size": world_size,
        "layout_sha256": layout_sha256,
        "partitions": partition_summaries,
        "ranks": rank_summaries,
        "recomputed_status": "PASS" if not failures else "FAIL",
    }
    return summary, layout, failures


def _lexical_absolute(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlink_components(path: Path, *, name: str) -> Path:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            component_stat = os.lstat(current)
        except FileNotFoundError as error:
            raise ValidationInputError(f"{name} does not exist: {current}") from error
        except NotADirectoryError as error:
            raise ValidationInputError(f"{name} has a non-directory parent: {current}") from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValidationInputError(f"{name} contains a symbolic-link component: {current}")
    return absolute


def _regular_file_bytes(path: Path, *, name: str) -> bytes:
    absolute = _reject_symlink_components(path, name=name)
    file_stat = os.stat(absolute, follow_symlinks=False)
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValidationInputError(f"{name} must be a regular non-symlink file: {path}")
    return absolute.read_bytes()


def _canonical_output_parent(path: Path) -> Path:
    if not path.is_absolute():
        raise ValidationInputError(f"output path must be absolute: {path}")
    parent = _reject_symlink_components(path.parent, name="output parent")
    if not stat.S_ISDIR(os.stat(parent, follow_symlinks=False).st_mode):
        raise ValidationInputError(f"output parent must be a directory: {parent}")
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != parent:
        raise ValidationInputError(
            f"output parent must be canonical: {parent} resolves to {resolved_parent}"
        )
    return parent


def _load_json(payload: bytes, *, name: str) -> object:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationInputError(f"{name} is not valid UTF-8 JSON") from error


def validate_ltop_g3_cold_action_evidence(
    *,
    factual_report_path: Path,
    mediator_required_report_path: Path,
) -> dict[str, Any]:
    """Return an independent, fail-closed validation report for both cold arms."""

    factual_resolved = factual_report_path.resolve()
    mediator_resolved = mediator_required_report_path.resolve()
    if factual_resolved == mediator_resolved:
        raise ValidationInputError("factual and mediator-required reports must be distinct files")

    inputs = {
        "factual": (factual_report_path, "factual"),
        "mediator_required": (mediator_required_report_path, "mediator-required"),
    }
    input_reports: dict[str, object] = {}
    input_summaries: dict[str, Any] = {}
    for label, (path, _information_set) in inputs.items():
        payload = _regular_file_bytes(path, name=f"{label} report")
        input_reports[label] = _load_json(payload, name=f"{label} report")
        input_summaries[label] = {
            "path": str(path.resolve()),
            "sha256": _sha256_bytes(payload),
        }

    failures: list[str] = []
    report_summaries: dict[str, Any] = {}
    layouts: dict[str, dict[str, Any]] = {}
    for label, (_path, information_set) in inputs.items():
        try:
            summary, layout, report_failures = _validate_report(
                input_reports[label],
                label=label,
                expected_information_set=information_set,
            )
        except ValidationInputError as error:
            summary = None
            layout = None
            report_failures = [f"{label}: ValidationInputError: {error}"]
        report_summaries[label] = summary
        failures.extend(report_failures)
        if layout is not None:
            layouts[label] = layout

    layout_match = (
        set(layouts) == set(inputs) and layouts["factual"] == layouts["mediator_required"]
    )
    if not layout_match:
        failures.append("factual and mediator-required scene/sample/target layouts differ")
    cross_report = {
        "layout_match": layout_match,
        "factual_layout_sha256": (
            None
            if report_summaries.get("factual") is None
            else report_summaries["factual"]["layout_sha256"]
        ),
        "mediator_required_layout_sha256": (
            None
            if report_summaries.get("mediator_required") is None
            else report_summaries["mediator_required"]["layout_sha256"]
        ),
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "inputs": input_summaries,
        "thresholds": {
            "bitwise_factual_replay": True,
            "mean_factual_target_minus_distractor_strictly_positive": True,
            "mean_blocked_path_did_strictly_positive": True,
            "positive_sample_fraction_minimum": POSITIVE_SAMPLE_FRACTION,
        },
        "reports": report_summaries,
        "cross_report": cross_report,
    }


def _failure_report(message: str) -> dict[str, Any]:
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "FAIL",
        "failures": [message],
        "inputs": None,
        "thresholds": {
            "bitwise_factual_replay": True,
            "mean_factual_target_minus_distractor_strictly_positive": True,
            "mean_blocked_path_did_strictly_positive": True,
            "positive_sample_fraction_minimum": POSITIVE_SAMPLE_FRACTION,
        },
        "reports": {},
        "cross_report": None,
    }


def _write_exclusive(path: Path, payload: str) -> None:
    parent = _canonical_output_parent(path)
    directory_fd = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    temp_path: Path | None = None
    lock = path.with_name(f".{path.name}.publish.lock")
    lock_acquired = False
    try:
        lock.mkdir(mode=0o700)
        lock_acquired = True
        os.fsync(directory_fd)
        if path.exists() or path.is_symlink():
            raise ValidationInputError(f"output already exists: {path}")

        temp_fd, temp_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        try:
            os.fchmod(temp_fd, 0o644)
            with os.fdopen(temp_fd, "w", encoding="ascii", newline="") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists() or path.is_symlink():
                raise ValidationInputError(f"output already exists: {path}")
            os.replace(temp_path, path)
            temp_path = None
            os.fsync(directory_fd)
        finally:
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()
                os.fsync(directory_fd)
    finally:
        if lock_acquired:
            with suppress(FileNotFoundError):
                lock.rmdir()
            os.fsync(directory_fd)
        os.close(directory_fd)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_paths = {args.factual_report.resolve(), args.mediator_required_report.resolve()}
    if args.output.resolve() in input_paths:
        sys.stderr.write("output must be distinct from both source reports\n")
        return 2
    try:
        result = validate_ltop_g3_cold_action_evidence(
            factual_report_path=args.factual_report,
            mediator_required_report_path=args.mediator_required_report,
        )
    except (OSError, ValidationInputError) as error:
        result = _failure_report(f"{type(error).__name__}: {error}")
    payload = _canonical_json(result) + "\n"
    try:
        _write_exclusive(args.output, payload)
    except (OSError, ValidationInputError) as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 2
    sys.stdout.write(payload)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
