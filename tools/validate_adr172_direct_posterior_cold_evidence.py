#!/usr/bin/env python3
"""Independently validate one ADR172 direct-posterior cold-action report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

REPORT_SCHEMA = "picf-next.adr172-direct-action-posterior-evaluation.v1"
OUTPUT_SCHEMA = "picf-next.adr172-direct-posterior-cold-validation.v1"
DIRECT_ACTION_SURFACE = "native-action-to-current-posterior-row-kv"
PARTITIONS = ("validation", "heldout")
EXPECTED_WORLD_SIZE = 2
EXPECTED_SCENES_PER_RANK_PARTITION = 4
POSITIVE_SCENE_FRACTION = 0.75
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12
FLOAT32_ULP_ALLOWANCE = 2.0


class ValidationInputError(ValueError):
    """Raised when a report cannot be interpreted without guessing."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
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


def _number(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationInputError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValidationInputError(f"{name} is outside its finite numeric contract")
    return result


def _sha256(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValidationInputError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _number_vector(
    value: object,
    *,
    name: str,
    nonnegative: bool = False,
) -> list[float]:
    values = _list(value, name=name)
    if not values:
        raise ValidationInputError(f"{name} cannot be empty")
    return [
        _number(item, name=f"{name}[{index}]", nonnegative=nonnegative)
        for index, item in enumerate(values)
    ]


def _integer_vector(value: object, *, name: str, minimum: int = 0) -> list[int]:
    values = _list(value, name=name)
    if not values:
        raise ValidationInputError(f"{name} cannot be empty")
    return [
        _integer(item, name=f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(values)
    ]


def _count(value: object, *, name: str, minimum: int = 0) -> int:
    """Parse a discrete count, including exact-integral legacy JSON floats."""

    if isinstance(value, bool):
        raise ValidationInputError(f"{name} must be an integer count")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    else:
        raise ValidationInputError(f"{name} must be an integer count")
    if result < minimum:
        raise ValidationInputError(f"{name} must be at least {minimum}")
    return result


def _count_vector(value: object, *, name: str, minimum: int = 0) -> list[int]:
    values = _list(value, name=name)
    if not values:
        raise ValidationInputError(f"{name} cannot be empty")
    return [
        _count(item, name=f"{name}[{index}]", minimum=minimum) for index, item in enumerate(values)
    ]


def _string_vector(value: object, *, name: str) -> list[str]:
    values = _list(value, name=name)
    if not values:
        raise ValidationInputError(f"{name} cannot be empty")
    result = [_string(item, name=f"{name}[{index}]") for index, item in enumerate(values)]
    if len(set(result)) != len(result):
        raise ValidationInputError(f"{name} must be unique")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValidationInputError("cannot compute an empty mean")
    return sum(values) / len(values)


def _float32(value: float) -> float:
    try:
        return struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError as error:
        raise ValidationInputError("evidence value exceeds the float32 score contract") from error


def _float32_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValidationInputError("cannot compute an empty float32 mean")
    total = 0.0
    for value in values:
        total = _float32(total + _float32(value))
    return _float32(total / len(values))


def _close(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right))
    if scale < 2.0**-126:
        float32_ulp = 2.0**-149
    else:
        float32_ulp = 2.0 ** (math.floor(math.log2(scale)) - 23)
    tolerance = max(
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE * scale,
        FLOAT32_ULP_ALLOWANCE * float32_ulp,
    )
    return abs(left - right) <= tolerance


def _check_number(
    failures: list[str],
    *,
    context: str,
    field: str,
    declared: float,
    recomputed: float,
) -> None:
    if not _close(declared, recomputed):
        failures.append(f"{context}: serialized {field} {declared!r} differs from {recomputed!r}")


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
        failures.append(f"{context}: serialized {field} differs from raw prompt evidence")


def _zero_safe_ratio(numerator: float, denominator: float) -> float:
    numerator = _float32(numerator)
    denominator = _float32(denominator)
    return 0.0 if denominator == 0.0 else _float32(numerator / denominator)


def _bindings(value: object, *, name: str, capacity: int) -> list[tuple[str, int]]:
    rows = _list(value, name=name)
    result: list[tuple[str, int]] = []
    for index, row in enumerate(rows):
        pair = _list(row, name=f"{name}[{index}]")
        if len(pair) != 2:
            raise ValidationInputError(f"{name}[{index}] must contain identity and row")
        result.append(
            (
                _string(pair[0], name=f"{name}[{index}][0]"),
                _integer(pair[1], name=f"{name}[{index}][1]", minimum=0),
            )
        )
    if any(row >= capacity for _, row in result):
        raise ValidationInputError(f"{name} contains a row outside capacity")
    if len({identity for identity, _ in result}) != len(result):
        raise ValidationInputError(f"{name} repeats a physical identity")
    if len({row for _, row in result}) != len(result):
        raise ValidationInputError(f"{name} repeats a posterior row")
    return result


def _validate_arm_receipts(
    value: object,
    *,
    context: str,
    capacity: int,
    failures: list[str],
) -> None:
    raw_receipts = _list(value, name=f"{context}.arm_receipts")
    receipts = [_mapping(item, name=f"{context}.arm_receipts") for item in raw_receipts]
    table: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        name = _string(receipt.get("arm_name"), name=f"{context}.arm_name")
        if name in table:
            raise ValidationInputError(f"{context}: duplicate arm receipt {name}")
        table[name] = receipt
    expected: dict[str, tuple[str, int | None]] = {
        "factual": ("factual", None),
        "factual-repeat": ("factual", None),
        "blocked": ("blocked", None),
    }
    expected.update({f"remove-row-{row}": ("row-removal", row) for row in range(capacity)})
    expected.update(
        {f"blocked-remove-row-{row}": ("blocked-row-removal", row) for row in range(capacity)}
    )
    if set(table) != set(expected):
        raise ValidationInputError(f"{context}: arm receipt set differs")
    action_hashes: dict[str, str] = {}
    visibility_hashes: dict[str, str] = {}
    active_hashes: set[str] = set()
    for name, receipt in table.items():
        expected_kind, expected_row = expected[name]
        if receipt.get("arm_kind") != expected_kind or receipt.get("row_index") != expected_row:
            failures.append(f"{context}: arm metadata differs for {name}")
        visibility_hashes[name] = _sha256(
            receipt.get("source_visibility_sha256"),
            name=f"{context}.{name}.source_visibility_sha256",
        )
        active_hashes.add(
            _sha256(
                receipt.get("active_action_mask_sha256"),
                name=f"{context}.{name}.active_action_mask_sha256",
            )
        )
        action_hashes[name] = _sha256(
            receipt.get("action_output_sha256"),
            name=f"{context}.{name}.action_output_sha256",
        )
    if len(active_hashes) != 1:
        failures.append(f"{context}: action arms changed the executable action mask")
    if action_hashes["factual"] != action_hashes["factual-repeat"]:
        failures.append(f"{context}: factual replay action hashes differ")
    if visibility_hashes["factual"] != visibility_hashes["factual-repeat"]:
        failures.append(f"{context}: factual replay visibility hashes differ")
    for row in range(capacity):
        name = f"blocked-remove-row-{row}"
        if action_hashes[name] != action_hashes["blocked"]:
            failures.append(f"{context}: blocked placebo action hash differs for row {row}")
        if visibility_hashes[name] != visibility_hashes["blocked"]:
            failures.append(f"{context}: blocked placebo visibility hash differs for row {row}")


def _validate_prompt(
    value: object,
    *,
    context: str,
    capacity: int,
    failures: list[str],
) -> dict[str, Any]:
    prompt = _mapping(value, name=context)
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
    if target_row >= capacity or distractor_row >= capacity:
        raise ValidationInputError(f"{context}: target row lies outside capacity")
    if target_identity == distractor_identity or target_row == distractor_row:
        failures.append(f"{context}: target and distractor are not distinct")
    _validate_arm_receipts(
        prompt.get("arm_receipts"),
        context=context,
        capacity=capacity,
        failures=failures,
    )
    score = _mapping(prompt.get("score"), name=f"{context}.score")
    sample_keys = _string_vector(score.get("sample_keys"), name=f"{context}.score.sample_keys")
    active_counts = _count_vector(
        score.get("active_action_counts"),
        name=f"{context}.score.active_action_counts",
        minimum=1,
    )
    fields = {
        "replay_floor_rms": True,
        "factual_all_posterior_block_effect_rms": True,
        "factual_target_effect_rms": True,
        "factual_distractor_effect_rms": True,
        "factual_target_minus_distractor": False,
        "factual_target_effect_over_all_posterior_block": False,
        "factual_distractor_effect_over_all_posterior_block": False,
        "factual_selectivity_over_all_posterior_block": False,
    }
    vectors = {
        field: _number_vector(
            score.get(field),
            name=f"{context}.score.{field}",
            nonnegative=nonnegative,
        )
        for field, nonnegative in fields.items()
    }
    lengths = {len(sample_keys), len(active_counts), *(len(value) for value in vectors.values())}
    if len(lengths) != 1:
        raise ValidationInputError(f"{context}: action score vectors have inconsistent lengths")
    if score.get("blocked_placebo_integrity_verified") is not True:
        failures.append(f"{context}: blocked placebo integrity was not verified")
    delta = [
        _float32(_float32(target) - _float32(distractor))
        for target, distractor in zip(
            vectors["factual_target_effect_rms"],
            vectors["factual_distractor_effect_rms"],
            strict=True,
        )
    ]
    target_ratio = [
        _zero_safe_ratio(target, blocked)
        for target, blocked in zip(
            vectors["factual_target_effect_rms"],
            vectors["factual_all_posterior_block_effect_rms"],
            strict=True,
        )
    ]
    distractor_ratio = [
        _zero_safe_ratio(distractor, blocked)
        for distractor, blocked in zip(
            vectors["factual_distractor_effect_rms"],
            vectors["factual_all_posterior_block_effect_rms"],
            strict=True,
        )
    ]
    selectivity_ratio = [
        _zero_safe_ratio(item, blocked)
        for item, blocked in zip(
            delta,
            vectors["factual_all_posterior_block_effect_rms"],
            strict=True,
        )
    ]
    for field, recomputed in (
        ("factual_target_minus_distractor", delta),
        ("factual_target_effect_over_all_posterior_block", target_ratio),
        ("factual_distractor_effect_over_all_posterior_block", distractor_ratio),
        ("factual_selectivity_over_all_posterior_block", selectivity_ratio),
    ):
        _check_vector(
            failures,
            context=f"{context}.score",
            field=field,
            declared=vectors[field],
            recomputed=recomputed,
        )
    declared_means = {
        "mean_factual_all_posterior_block_effect_rms": _float32_mean(
            vectors["factual_all_posterior_block_effect_rms"]
        ),
        "mean_factual_target_minus_distractor": _float32_mean(delta),
        "mean_factual_selectivity_over_all_posterior_block": _float32_mean(selectivity_ratio),
    }
    for field, recomputed in declared_means.items():
        _check_number(
            failures,
            context=f"{context}.score",
            field=field,
            declared=_number(score.get(field), name=f"{context}.score.{field}"),
            recomputed=recomputed,
        )
    return {
        "target_identity": target_identity,
        "distractor_identity": distractor_identity,
        "target_row": target_row,
        "distractor_row": distractor_row,
        "sample_keys": sample_keys,
        "active_action_counts": active_counts,
        "replay": vectors["replay_floor_rms"],
        "all_block": vectors["factual_all_posterior_block_effect_rms"],
        "delta": delta,
        "normalized_delta": selectivity_ratio,
    }


def _validate_scene(
    value: object,
    *,
    context: str,
    capacity: int,
    failures: list[str],
) -> dict[str, Any]:
    scene = _mapping(value, name=context)
    item_id = _string(scene.get("item_id"), name=f"{context}.item_id")
    sample_key = _string(scene.get("sample_key"), name=f"{context}.sample_key")
    prompts = _list(scene.get("prompts"), name=f"{context}.prompts")
    if len(prompts) != 2 or scene.get("prompt_count") != 2:
        raise ValidationInputError(f"{context}: exactly two crossed prompts are required")
    parsed = [
        _validate_prompt(
            prompt,
            context=f"{context}.prompts[{index}]",
            capacity=capacity,
            failures=failures,
        )
        for index, prompt in enumerate(prompts)
    ]
    first, second = parsed
    if (
        first["target_identity"] != second["distractor_identity"]
        or second["target_identity"] != first["distractor_identity"]
        or first["target_row"] != second["distractor_row"]
        or second["target_row"] != first["distractor_row"]
    ):
        failures.append(f"{context}: crossed prompts do not reverse one canonical row pair")
    if first["sample_keys"] != second["sample_keys"]:
        failures.append(f"{context}: crossed prompts changed the action sample axis")
    if first["active_action_counts"] != second["active_action_counts"]:
        failures.append(f"{context}: crossed prompts changed the executable action surface")
    canonical = _bindings(
        scene.get("canonical_bindings"),
        name=f"{context}.canonical_bindings",
        capacity=capacity,
    )
    independent_values = _list(
        scene.get("independent_bindings_by_prompt"),
        name=f"{context}.independent_bindings_by_prompt",
    )
    if len(independent_values) != 2:
        raise ValidationInputError(f"{context}: independent row gauges are incomplete")
    independent = [
        _bindings(
            value,
            name=f"{context}.independent_bindings_by_prompt[{index}]",
            capacity=capacity,
        )
        for index, value in enumerate(independent_values)
    ]
    shared_row_gauge = all(value == canonical for value in independent)
    if scene.get("shared_row_gauge") is not shared_row_gauge or not shared_row_gauge:
        failures.append(f"{context}: physical row gauge changed across prompts")
    scene_score = _mapping(scene.get("score"), name=f"{context}.score")
    sample_keys = _string_vector(
        scene_score.get("sample_keys"),
        name=f"{context}.score.sample_keys",
    )
    active_counts = _count_vector(
        scene_score.get("active_action_counts"),
        name=f"{context}.score.active_action_counts",
        minimum=1,
    )
    if sample_keys != first["sample_keys"] or active_counts != first["active_action_counts"]:
        failures.append(f"{context}: scene score changed the prompt action axis")
    replay = [*first["replay"], *second["replay"]]
    prompt_all_block = [
        _float32_mean(first["all_block"]),
        _float32_mean(second["all_block"]),
    ]
    crossed = [left + right for left, right in zip(first["delta"], second["delta"], strict=True)]
    normalized = [
        0.5 * (left + right)
        for left, right in zip(
            first["normalized_delta"],
            second["normalized_delta"],
            strict=True,
        )
    ]
    raw_vectors = {
        "replay_floor_rms": replay,
        "prompt_mean_factual_all_posterior_block_effect_rms": prompt_all_block,
        "crossed_prompt_target_selectivity": crossed,
        "crossed_prompt_selectivity_over_all_posterior_block": normalized,
    }
    for field, recomputed in raw_vectors.items():
        _check_vector(
            failures,
            context=f"{context}.score",
            field=field,
            declared=_number_vector(
                scene_score.get(field),
                name=f"{context}.score.{field}",
                nonnegative=field
                in {
                    "replay_floor_rms",
                    "prompt_mean_factual_all_posterior_block_effect_rms",
                },
            ),
            recomputed=recomputed,
        )
    recomputed_numbers = {
        "max_replay_floor_rms": max(replay),
        "minimum_prompt_factual_all_posterior_block_effect_rms": min(prompt_all_block),
        "mean_crossed_prompt_target_selectivity": _mean(crossed),
        "mean_crossed_prompt_selectivity_over_all_posterior_block": _mean(normalized),
    }
    for field, recomputed in recomputed_numbers.items():
        _check_number(
            failures,
            context=f"{context}.score",
            field=field,
            declared=_number(
                scene_score.get(field),
                name=f"{context}.score.{field}",
                nonnegative=field
                in {
                    "max_replay_floor_rms",
                    "minimum_prompt_factual_all_posterior_block_effect_rms",
                },
            ),
            recomputed=recomputed,
        )
    positive_count = sum(value > 0.0 for value in crossed)
    if scene_score.get("positive_crossed_prompt_target_selectivity_count") != positive_count:
        failures.append(f"{context}: positive crossed-prompt count differs")
    if scene_score.get("sample_count") != len(crossed):
        failures.append(f"{context}: scene sample count differs")
    if scene_score.get("blocked_placebo_integrity_verified") is not True:
        failures.append(f"{context}: scene blocked placebo integrity was not verified")
    return {
        "item_id": item_id,
        "sample_key": sample_key,
        "max_replay_floor_rms": max(replay),
        "mean_crossed_prompt_target_selectivity": _mean(crossed),
        "mean_crossed_prompt_selectivity_over_all_posterior_block": _mean(normalized),
        "minimum_prompt_factual_all_posterior_block_effect_rms": min(prompt_all_block),
    }


def _partition_summary(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scene_count = len(units)
    minimum = math.ceil(POSITIVE_SCENE_FRACTION * scene_count)
    positive_crossed = sum(
        float(unit["mean_crossed_prompt_target_selectivity"]) > 0.0 for unit in units
    )
    positive_normalized = sum(
        float(unit["mean_crossed_prompt_selectivity_over_all_posterior_block"]) > 0.0
        for unit in units
    )
    positive_all_block = sum(
        float(unit["minimum_prompt_factual_all_posterior_block_effect_rms"]) > 0.0 for unit in units
    )
    joint = sum(
        float(unit["mean_crossed_prompt_target_selectivity"]) > 0.0
        and float(unit["mean_crossed_prompt_selectivity_over_all_posterior_block"]) > 0.0
        and float(unit["minimum_prompt_factual_all_posterior_block_effect_rms"]) > 0.0
        for unit in units
    )
    return {
        "scene_count": scene_count,
        "positive_scene_fraction_minimum": POSITIVE_SCENE_FRACTION,
        "minimum_positive_scene_count": minimum,
        "positive_crossed_prompt_scene_count": positive_crossed,
        "positive_normalized_crossed_prompt_scene_count": positive_normalized,
        "positive_all_posterior_block_scene_count": positive_all_block,
        "joint_positive_scene_count": joint,
        "mean_crossed_prompt_target_selectivity": _mean(
            [float(unit["mean_crossed_prompt_target_selectivity"]) for unit in units]
        ),
        "mean_crossed_prompt_selectivity_over_all_posterior_block": _mean(
            [
                float(unit["mean_crossed_prompt_selectivity_over_all_posterior_block"])
                for unit in units
            ]
        ),
        "mean_minimum_prompt_factual_all_posterior_block_effect_rms": _mean(
            [float(unit["minimum_prompt_factual_all_posterior_block_effect_rms"]) for unit in units]
        ),
        "max_replay_floor_rms": max(float(unit["max_replay_floor_rms"]) for unit in units),
        "scenes": [dict(unit) for unit in units],
    }


def _check_declared_summary(
    value: object,
    *,
    context: str,
    recomputed: Mapping[str, Any],
    failures: list[str],
) -> None:
    declared = _mapping(value, name=context)
    integer_fields = (
        "scene_count",
        "minimum_positive_scene_count",
        "positive_crossed_prompt_scene_count",
        "positive_normalized_crossed_prompt_scene_count",
        "positive_all_posterior_block_scene_count",
        "joint_positive_scene_count",
    )
    for field in integer_fields:
        if declared.get(field) != recomputed[field]:
            failures.append(f"{context}: serialized {field} differs from raw evidence")
    number_fields = (
        "positive_scene_fraction_minimum",
        "mean_crossed_prompt_target_selectivity",
        "mean_crossed_prompt_selectivity_over_all_posterior_block",
        "mean_minimum_prompt_factual_all_posterior_block_effect_rms",
        "max_replay_floor_rms",
    )
    for field in number_fields:
        _check_number(
            failures,
            context=context,
            field=field,
            declared=_number(declared.get(field), name=f"{context}.{field}"),
            recomputed=float(recomputed[field]),
        )


def _validate_source_contract(value: object, *, name: str) -> None:
    contract = _mapping(value, name=name)
    for field in ("repository_commit", "repository_tree"):
        digest = _string(contract.get(field), name=f"{name}.{field}")
        if len(digest) != 40 or any(character not in "0123456789abcdef" for character in digest):
            raise ValidationInputError(f"{name}.{field} must be one Git object identity")
    files = _mapping(contract.get("critical_file_sha256"), name=f"{name}.critical_file_sha256")
    if not files:
        raise ValidationInputError(f"{name}.critical_file_sha256 cannot be empty")
    for path, digest in files.items():
        _string(path, name=f"{name}.critical_file_sha256 path")
        _sha256(digest, name=f"{name}.critical_file_sha256[{path}]")


def validate_adr172_direct_posterior_cold_evidence(report_path: Path) -> dict[str, Any]:
    raw = report_path.read_bytes()
    report = _mapping(json.loads(raw), name="report")
    failures: list[str] = []
    if report.get("schema") != REPORT_SCHEMA:
        raise ValidationInputError("report schema differs from ADR172 cold evaluation")
    if report.get("mode") != "gate" or report.get("phase") != "evaluation":
        raise ValidationInputError("ADR172 cold validation requires mode=gate phase=evaluation")
    if report.get("world_size") != EXPECTED_WORLD_SIZE:
        raise ValidationInputError("ADR172 cold report world size differs")
    if report.get("direct_action_causal_surface") != DIRECT_ACTION_SURFACE:
        failures.append("report did not use the registered direct posterior action surface")
    capacity = _integer(report.get("capacity"), name="report.capacity", minimum=2)
    thresholds = _mapping(report.get("thresholds"), name="report.thresholds")
    if thresholds.get("joint_positive_scene_fraction_minimum") != POSITIVE_SCENE_FRACTION:
        failures.append("report positive scene threshold differs from the registered 0.75 gate")
    if thresholds.get("joint_positive_scene_requires_normalized_selectivity") is not True:
        failures.append("report joint-positive scenes omitted normalized selectivity")
    if thresholds.get("bitwise_factual_replay") is not True:
        failures.append("report did not require bitwise factual replay")
    if thresholds.get("blocked_row_placebo_bitwise_equality") is not True:
        failures.append("report did not require blocked-row placebo equality")
    adoption = _mapping(report.get("causal_adoption_contract"), name="causal_adoption_contract")
    if adoption.get("exclusive_visual_path_claim") is not False:
        failures.append("report changed the incremental causal-adoption claim")
    inference = _mapping(report.get("action_inference_contract"), name="action_inference_contract")
    if inference.get("active_action_surface") != "joint_mask AND NOT action_is_pad":
        failures.append("report action effect was not bound to the executable action surface")
    _validate_source_contract(report.get("picf_source_contract"), name="picf_source_contract")
    _validate_source_contract(
        report.get("trained_picf_source_contract"),
        name="trained_picf_source_contract",
    )

    rank_reports = _list(report.get("rank_reports"), name="rank_reports")
    if len(rank_reports) != EXPECTED_WORLD_SIZE:
        raise ValidationInputError("rank report count differs from world size")
    parsed_ranks = [
        _mapping(value, name=f"rank_reports[{index}]") for index, value in enumerate(rank_reports)
    ]
    ranks = [_integer(value.get("rank"), name="rank") for value in parsed_ranks]
    if set(ranks) != set(range(EXPECTED_WORLD_SIZE)) or len(set(ranks)) != len(ranks):
        raise ValidationInputError("rank reports do not cover the distributed axis exactly once")

    summaries: dict[str, dict[str, Any]] = {}
    partition_sample_keys: dict[str, set[str]] = {}
    for partition in PARTITIONS:
        units: list[dict[str, Any]] = []
        scene_keys: set[tuple[str, str]] = set()
        sample_keys: set[str] = set()
        for rank_report in parsed_ranks:
            rank = int(rank_report["rank"])
            if rank_report.get("direct_action_causal_surface") != DIRECT_ACTION_SURFACE:
                failures.append(f"rank {rank}: direct action surface differs")
            history = _list(rank_report.get("history"), name=f"rank {rank}.history")
            if len(history) != 1:
                raise ValidationInputError(f"rank {rank}: evaluation must publish one history item")
            partition_report = _mapping(
                _mapping(history[0], name=f"rank {rank}.history[0]").get(partition),
                name=f"rank {rank}.{partition}",
            )
            scenes = _list(
                partition_report.get("scenes"),
                name=f"rank {rank}.{partition}.scenes",
            )
            if len(scenes) != EXPECTED_SCENES_PER_RANK_PARTITION:
                raise ValidationInputError(
                    f"rank {rank}: {partition} scene count differs from the registered gate"
                )
            if partition_report.get("scene_count") != len(scenes):
                failures.append(f"rank {rank}: {partition} scene count field differs")
            if partition_report.get("prompt_count") != 2 * len(scenes):
                failures.append(f"rank {rank}: {partition} prompt count field differs")
            for index, scene in enumerate(scenes):
                unit = _validate_scene(
                    scene,
                    context=f"rank {rank}.{partition}.scenes[{index}]",
                    capacity=capacity,
                    failures=failures,
                )
                key = (unit["item_id"], unit["sample_key"])
                if key in scene_keys or unit["sample_key"] in sample_keys:
                    failures.append(f"{partition}: scene/sample identity repeats across ranks")
                scene_keys.add(key)
                sample_keys.add(unit["sample_key"])
                units.append({"rank": rank, **unit})
        summary = _partition_summary(units)
        if summary["max_replay_floor_rms"] != 0.0:
            failures.append(f"{partition}: factual replay was not bitwise stable")
        if summary["mean_crossed_prompt_target_selectivity"] <= 0.0:
            failures.append(f"{partition}: mean crossed-prompt row selectivity was nonpositive")
        if summary["mean_crossed_prompt_selectivity_over_all_posterior_block"] <= 0.0:
            failures.append(f"{partition}: normalized crossed-prompt selectivity was nonpositive")
        if summary["mean_minimum_prompt_factual_all_posterior_block_effect_rms"] <= 0.0:
            failures.append(f"{partition}: all-posterior block had no executable-action effect")
        if summary["joint_positive_scene_count"] < summary["minimum_positive_scene_count"]:
            failures.append(
                f"{partition}: jointly positive causal scenes "
                f"{summary['joint_positive_scene_count']} < "
                f"{summary['minimum_positive_scene_count']}"
            )
        summaries[partition] = summary
        partition_sample_keys[partition] = sample_keys

    if partition_sample_keys["validation"] & partition_sample_keys["heldout"]:
        failures.append("validation and heldout causal samples overlap")
    declared_summaries = _mapping(report.get("cold_causal_summary"), name="cold_causal_summary")
    for partition in PARTITIONS:
        _check_declared_summary(
            declared_summaries.get(partition),
            context=f"cold_causal_summary.{partition}",
            recomputed=summaries[partition],
            failures=failures,
        )
    independent_status = "PASS" if not failures else "FAIL"
    if report.get("status") != independent_status:
        failures.append(
            f"serialized report status {report.get('status')!r} differs from independent "
            f"status {independent_status}"
        )
        independent_status = "FAIL"
    if report.get("status") == "PASS" and report.get("failures") != []:
        failures.append("serialized PASS report contains failures")
        independent_status = "FAIL"
    return {
        "schema": OUTPUT_SCHEMA,
        "status": independent_status,
        "failures": failures,
        "source_report": str(report_path.resolve()),
        "source_report_sha256": _sha256_bytes(raw),
        "validator_source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "positive_scene_fraction": POSITIVE_SCENE_FRACTION,
        "partitions": summaries,
    }


def _write_text_durable_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", closefd=False) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = validate_adr172_direct_posterior_cold_evidence(args.report)
    except BaseException as error:
        result = {
            "schema": OUTPUT_SCHEMA,
            "status": "FAIL",
            "failures": [f"{type(error).__name__}: {error}"],
            "source_report": str(args.report.resolve()),
            "source_report_sha256": (
                _sha256_bytes(args.report.read_bytes()) if args.report.is_file() else None
            ),
            "validator_source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "positive_scene_fraction": POSITIVE_SCENE_FRACTION,
            "partitions": {},
        }
    _write_text_durable_exclusive(args.output, _canonical_json(result) + "\n")
    print(_canonical_json(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
