"""Strict runtime evidence for fixed stationary-posterior replay."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, cast

from picf_next.eval.stationary_replay import (
    STATIONARY_FIXED_REPLAY_PASS,
    validate_stationary_fixed_replay,
)

STATIONARY_RUNTIME_PROBE_SCHEMA: Final = "picf-next.stationary-runtime-probe.v1"
STATIONARY_RUNTIME_PROBE_PASS: Final = "PASS"
STATIONARY_RUNTIME_PROBE_FAIL: Final = "FAIL"

_MODELS: Final = ("fresh_m2", "candidate")
_SPLITS: Final = ("validation", "heldout")
_RECORD_FIELDS: Final = {
    "model",
    "split",
    "optimizer_step",
    "rank",
    "prefix_length",
    "transition_count",
    "elapsed_seconds",
    "peak_allocated_bytes",
}


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_records(
    records: Sequence[object],
    *,
    fixed_replay: dict[str, Any],
    total_memory_bytes: int,
) -> list[dict[str, Any]]:
    expected: dict[tuple[str, str, int, int], tuple[int, int]] = {}
    for measurement in fixed_replay["measurements"]:
        coordinate = (
            measurement["model"],
            measurement["split"],
            measurement["optimizer_step"],
            measurement["rank"],
        )
        clip = measurement["clip"]
        expected[coordinate] = (
            clip["prefix_length"],
            clip["prefix_length"] + clip["train_length"],
        )
    if len(records) != len(expected):
        raise ValueError("stationary runtime measurement coverage changed")
    validated = []
    observed: set[tuple[str, str, int, int]] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict) or set(raw) != _RECORD_FIELDS:
            raise ValueError(f"stationary runtime measurement {index} fields changed")
        row = cast(dict[str, Any], raw)
        coordinate = (
            row["model"],
            row["split"],
            row["optimizer_step"],
            row["rank"],
        )
        if (
            row["model"] not in _MODELS
            or row["split"] not in _SPLITS
            or not isinstance(row["optimizer_step"], int)
            or isinstance(row["optimizer_step"], bool)
            or not isinstance(row["rank"], int)
            or isinstance(row["rank"], bool)
        ):
            raise ValueError("stationary runtime measurement coordinate is malformed")
        if coordinate not in expected or coordinate in observed:
            raise ValueError("stationary runtime measurement coordinate changed")
        observed.add(coordinate)
        expected_prefix, expected_transitions = expected[coordinate]
        if (
            not isinstance(row["prefix_length"], int)
            or isinstance(row["prefix_length"], bool)
            or row["prefix_length"] != expected_prefix
        ):
            raise ValueError("stationary runtime prefix differs from fixed replay")
        if (
            not isinstance(row["transition_count"], int)
            or isinstance(row["transition_count"], bool)
            or row["transition_count"] != expected_transitions
        ):
            raise ValueError("stationary runtime transition count differs from fixed replay")
        elapsed = _positive_float(row["elapsed_seconds"], "stationary runtime elapsed time")
        peak = _positive_integer(
            row["peak_allocated_bytes"], "stationary runtime peak allocated bytes"
        )
        if peak >= total_memory_bytes:
            raise ValueError("stationary runtime peak allocation exceeds device memory")
        validated.append(
            {
                **row,
                "elapsed_seconds": elapsed,
                "peak_allocated_bytes": peak,
            }
        )
    return validated


def build_stationary_runtime_probe(
    fixed_replay: object,
    *,
    fixed_replay_sha256: str,
    candidate_recurrent_state_serialized: bool,
    device_name: str,
    total_memory_bytes: int,
    measurements: Sequence[object],
) -> dict[str, Any]:
    """Build one replay-bound memory and throughput report."""

    replay = validate_stationary_fixed_replay(fixed_replay)
    replay_sha256 = _sha256(fixed_replay_sha256, "fixed replay SHA-256")
    if not isinstance(candidate_recurrent_state_serialized, bool):
        raise TypeError("candidate recurrent-state serialization flag must be boolean")
    if not isinstance(device_name, str) or not device_name.strip():
        raise ValueError("stationary runtime device name cannot be empty")
    memory = _positive_integer(total_memory_bytes, "stationary runtime total memory")
    rows = _validated_records(
        measurements,
        fixed_replay=replay,
        total_memory_bytes=memory,
    )
    summaries: dict[str, dict[str, int | float]] = {}
    for model_name in _MODELS:
        selected = [row for row in rows if row["model"] == model_name]
        elapsed = sum(row["elapsed_seconds"] for row in selected)
        transitions = sum(row["transition_count"] for row in selected)
        summaries[model_name] = {
            "call_count": len(selected),
            "transition_count": transitions,
            "maximum_prefix_length": max(row["prefix_length"] for row in selected),
            "elapsed_seconds_total": elapsed,
            "elapsed_seconds_per_transition": elapsed / transitions,
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in selected),
        }
    expected_calls = (
        len(_SPLITS)
        * replay["protocol"]["optimizer_steps_per_split"]
        * replay["protocol"]["world_size"]
    )
    candidate = summaries["candidate"]
    checks = {
        "fixed_replay_passed": replay["status"] == STATIONARY_FIXED_REPLAY_PASS,
        "cuda_device_is_a100_40g": "A100" in device_name and memory >= 39 * 2**30,
        "candidate_call_coverage_complete": candidate["call_count"] == expected_calls,
        "candidate_prefix_128_completed": candidate["maximum_prefix_length"] == 128,
        "candidate_elapsed_finite_positive": (
            math.isfinite(float(candidate["elapsed_seconds_total"]))
            and float(candidate["elapsed_seconds_total"]) > 0.0
        ),
        "candidate_peak_memory_below_ninety_percent": (
            int(candidate["peak_allocated_bytes"]) < int(0.9 * memory)
        ),
        "candidate_recurrent_state_not_serialized": not candidate_recurrent_state_serialized,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": STATIONARY_RUNTIME_PROBE_SCHEMA,
        "status": STATIONARY_RUNTIME_PROBE_PASS if not failed else STATIONARY_RUNTIME_PROBE_FAIL,
        "protocol": {
            "observation_inputs": "task-independent-cached-native-token-bank",
            "compute_dtype": "bfloat16",
            "device_type": "cuda",
            "memory_headroom_fraction": 0.1,
            "split_names": list(_SPLITS),
        },
        "bindings": {
            "fixed_checkpoint_replay_sha256": replay_sha256,
            "candidate_checkpoint_sha256": replay["bindings"]["candidate_checkpoint_sha256"],
            "candidate_code_revision": replay["bindings"]["candidate_code_revision"],
        },
        "hardware": {
            "device_name": device_name,
            "total_memory_bytes": memory,
        },
        "models": summaries,
        "measurements": rows,
        "checks": checks,
        "failed_checks": failed,
        "long_training_authorized": False,
    }


def validate_stationary_runtime_probe(
    payload: object,
    *,
    fixed_replay: object,
    fixed_replay_sha256: str,
    candidate_recurrent_state_serialized: bool,
) -> dict[str, Any]:
    """Recompute every runtime summary and decision from raw measurements."""

    if not isinstance(payload, dict):
        raise ValueError("stationary runtime probe must contain one JSON object")
    report = cast(dict[str, Any], payload)
    hardware = report.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("stationary runtime hardware record is malformed")
    measurements = report.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("stationary runtime measurements must be one list")
    expected = build_stationary_runtime_probe(
        fixed_replay,
        fixed_replay_sha256=fixed_replay_sha256,
        candidate_recurrent_state_serialized=candidate_recurrent_state_serialized,
        device_name=cast(str, hardware.get("device_name")),
        total_memory_bytes=cast(int, hardware.get("total_memory_bytes")),
        measurements=measurements,
    )
    if report != expected:
        raise ValueError("stationary runtime probe differs from its measurements")
    return report
