#!/usr/bin/env python3
"""Replay an M3 checkpoint and render task-labelled temporal object diagnostics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.data.calvin import CALVIN_HOST_IMAGE_KEYS  # noqa: E402
from picf_next.posterior import (  # noqa: E402
    BIRTH_EVENT,
    DEATH_EVENT,
    MATCH_EVENT,
    MISS_EVENT,
    UNUSED_EVENT,
)

_EVENT_NAMES = {
    UNUSED_EVENT: "unused",
    MATCH_EVENT: "match",
    MISS_EVENT: "miss",
    BIRTH_EVENT: "birth",
    DEATH_EVENT: "death",
}
_SAFE_NAME = re.compile(r"[^a-z0-9]+")


def _evaluation_context(torch_module: Any, *, reload_weights_midstream: bool) -> Any:
    """Disable gradients while keeping mid-replay checkpoint buffers reloadable."""

    return torch_module.no_grad() if reload_weights_midstream else torch_module.inference_mode()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m3_probe.json",
    )
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--state-prefix-checkpoint",
        type=Path,
        help=(
            "form the initial posterior with this earlier compatible checkpoint, then "
            "switch to --training-checkpoint after its completed optimizer prefix"
        ),
    )
    parser.add_argument("--vjepa2-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--render-steps",
        default="1,2,8,14,20",
        help="one-based optimizer steps to render, or 'all'",
    )
    parser.add_argument(
        "--replay-steps",
        type=int,
        help=(
            "read-only final-weight replay length; values beyond the checkpoint plan "
            "require an exactly matching frozen-plan prefix"
        ),
    )
    parser.add_argument(
        "--source-disjoint-segments",
        type=int,
        nargs="+",
        help=(
            "evaluate only these CALVIN language-segment episodes from an empty posterior; "
            "their source frames must be disjoint from the checkpoint's consumed prefix"
        ),
    )
    parser.add_argument(
        "--prompt-causal-steps",
        default="",
        help="one-based replay steps for fixed-noise prompt/row interventions",
    )
    parser.add_argument(
        "--counterfactual-task",
        help=(
            "explicit counterfactual instruction; by default the closest distinct "
            "dataset instruction is selected deterministically"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be one JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="ascii") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def _parse_render_steps(value: str, total_steps: int) -> set[int]:
    if value == "all":
        return set(range(1, total_steps + 1))
    try:
        steps = {int(item) for item in value.split(",")}
    except ValueError as error:
        raise ValueError("render steps must be comma-separated integers or 'all'") from error
    if not steps or min(steps) < 1 or max(steps) > total_steps:
        raise ValueError("render steps lie outside the checkpoint plan")
    return steps


def _parse_optional_steps(value: str, total_steps: int) -> set[int]:
    if value == "":
        return set()
    return _parse_render_steps(value, total_steps)


_PROMPT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "at",
        "from",
        "in",
        "into",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)


def _semantic_tokens(value: str) -> frozenset[str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("semantic text must be a nonempty string")
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in _PROMPT_STOP_WORDS
    )


def _select_counterfactual_task(task: str, candidates: Sequence[str]) -> str:
    """Select the closest distinct dataset instruction without task-specific rules."""

    original = _semantic_tokens(task)
    available = sorted({candidate for candidate in candidates if candidate != task})
    if not available:
        raise ValueError("prompt intervention requires a distinct counterfactual task")
    return min(
        available,
        key=lambda candidate: (
            -len(original & _semantic_tokens(candidate)),
            len(original ^ _semantic_tokens(candidate)),
            candidate,
        ),
    )


def _semantic_identity_score(task: str, identity_key: str | None) -> int:
    if identity_key is None:
        return 0
    return len(_semantic_tokens(task) & _semantic_tokens(identity_key))


def _validate_extended_plan_prefix(reference: Any, extended: Any) -> int:
    """Prove that extending a frozen plan does not alter its bound prefix."""

    if extended.total_steps < reference.total_steps:
        raise ValueError("extended replay plan is shorter than the checkpoint plan")
    for step in range(reference.total_steps):
        if extended.global_batch(step) != reference.global_batch(step):
            raise ValueError(f"extended replay plan changed checkpoint step {step + 1}")
    return reference.total_steps


def _validate_state_handoff_controls(
    primary: Mapping[str, Any],
    prefix: Mapping[str, Any],
) -> tuple[int, int]:
    """Validate that two checkpoints differ only by their frozen-plan horizon."""

    contracts: list[dict[str, Any]] = []
    progress_records: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for name, control in (("primary", primary), ("prefix", prefix)):
        contract = control.get("contract")
        progress = control.get("progress")
        plan = control.get("plan")
        if not isinstance(contract, dict) or not isinstance(progress, dict):
            raise ValueError(f"{name} state-handoff checkpoint contract/progress is malformed")
        if not isinstance(plan, dict):
            raise ValueError(f"{name} state-handoff checkpoint plan is malformed")
        contracts.append(contract)
        progress_records.append(progress)
        plans.append(plan)

    ignored_contract_fields = {"fairness_sha256", "sample_plan_sha256"}
    comparable_contracts = tuple(
        {key: value for key, value in contract.items() if key not in ignored_contract_fields}
        for contract in contracts
    )
    if comparable_contracts[0] != comparable_contracts[1]:
        raise ValueError("state-handoff checkpoints differ outside their frozen-plan horizon")

    prefix_steps = progress_records[1].get("attempted_optimizer_steps")
    if (
        not isinstance(prefix_steps, int)
        or isinstance(prefix_steps, bool)
        or prefix_steps <= 0
        or progress_records[1].get("successful_optimizer_steps") != prefix_steps
    ):
        raise ValueError("state-prefix checkpoint is not a completed successful prefix")
    prefix_plan_steps = plans[1].get("total_steps")
    if (
        not isinstance(prefix_plan_steps, int)
        or isinstance(prefix_plan_steps, bool)
        or prefix_plan_steps < prefix_steps
    ):
        raise ValueError("state-prefix checkpoint training-plan length is malformed")
    return prefix_steps, prefix_plan_steps


def _invert_observation_to_row(
    observation_to_row: Sequence[int],
    *,
    capacity: int,
) -> tuple[int | None, ...]:
    """Invert a one-to-one runtime association without hiding malformed maps."""

    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("posterior capacity must be a positive integer")
    query_by_row: list[int | None] = [None] * capacity
    for query, row in enumerate(observation_to_row):
        if row == -1:
            continue
        if not 0 <= row < capacity:
            raise ValueError("runtime observation map references an invalid posterior row")
        if query_by_row[row] is not None:
            raise ValueError("multiple runtime observations reference one posterior row")
        query_by_row[row] = query
    return tuple(query_by_row)


def _trace_address_relation_pairs(
    *,
    previous_keys: Sequence[str | None],
    current_identity_keys: Sequence[str],
    prediction_indices: Sequence[int],
    target_indices: Sequence[int],
    object_inventory_complete: bool,
    address_cosine: Sequence[Sequence[float]],
    query_existence_probability: Sequence[float],
    query_localization_confidence: Sequence[float],
    query_mask_quality: Sequence[float],
    query_mask_coherence_score: Sequence[float],
    query_object_confidence: Sequence[float],
    logit_scale: float,
    logit_bias: float,
) -> list[dict[str, Any]]:
    """Label every deployed prior-row/current-query address relation.

    This is evaluation-only instrumentation of the exact relation family used by
    the production filter.  Loss-side identities label pairs after the forward
    pass; they never alter discovery, association or action prediction.
    """

    if isinstance(logit_scale, bool) or not math.isfinite(logit_scale) or logit_scale <= 0.0:
        raise ValueError("relation logit scale must be positive and finite")
    if isinstance(logit_bias, bool) or not math.isfinite(logit_bias):
        raise ValueError("relation logit bias must be finite")
    if not isinstance(object_inventory_complete, bool):
        raise TypeError("object inventory completeness must be boolean")
    if len(prediction_indices) != len(target_indices):
        raise ValueError("relation trace received a malformed set match")
    query_count = len(query_existence_probability)
    if (
        len(query_localization_confidence) != query_count
        or len(query_mask_quality) != query_count
        or len(query_mask_coherence_score) != query_count
        or len(query_object_confidence) != query_count
    ):
        raise ValueError("query confidence diagnostics must align current queries")
    if len(address_cosine) != len(previous_keys) or any(
        len(row) != query_count for row in address_cosine
    ):
        raise ValueError("relation cosine matrix must align prior rows and current queries")
    if any(not isinstance(key, str) or not key for key in current_identity_keys):
        raise ValueError("current relation identities must be nonempty strings")
    if len(set(prediction_indices)) != len(prediction_indices) or len(set(target_indices)) != len(
        target_indices
    ):
        raise ValueError("relation trace set match must be one-to-one")

    query_identity: dict[int, str] = {}
    for query, target in zip(prediction_indices, target_indices, strict=True):
        if not 0 <= query < query_count or not 0 <= target < len(current_identity_keys):
            raise ValueError("relation trace set match index is out of bounds")
        query_identity[query] = current_identity_keys[target]

    records: list[dict[str, Any]] = []
    for prior_row, prior_key in enumerate(previous_keys):
        if prior_key is None:
            continue
        if not isinstance(prior_key, str) or not prior_key:
            raise ValueError("prior relation identities must be nonempty strings or None")
        for query, raw_cosine in enumerate(address_cosine[prior_row]):
            cosine = float(raw_cosine)
            existence = float(query_existence_probability[query])
            localization_confidence = float(query_localization_confidence[query])
            mask_quality = float(query_mask_quality[query])
            mask_coherence_score = float(query_mask_coherence_score[query])
            object_confidence = float(query_object_confidence[query])
            if not math.isfinite(cosine) or not -1.000001 <= cosine <= 1.000001:
                raise ValueError("relation cosine must be finite and lie in [-1, 1]")
            if not math.isfinite(existence) or not 0.0 <= existence <= 1.0:
                raise ValueError("query existence probability must lie in [0, 1]")
            if not math.isfinite(mask_quality) or not 0.0 <= mask_quality <= 1.0:
                raise ValueError("query mask quality must lie in [0, 1]")
            if (
                not math.isfinite(localization_confidence)
                or not 0.0 <= localization_confidence <= 1.0
            ):
                raise ValueError("query localization confidence must lie in [0, 1]")
            if not math.isfinite(mask_coherence_score) or not 0.0 <= mask_coherence_score <= 1.0:
                raise ValueError("query mask coherence score must lie in [0, 1]")
            if not math.isfinite(object_confidence) or not 0.0 <= object_confidence <= 1.0:
                raise ValueError("query object confidence must lie in [0, 1]")
            current_key = query_identity.get(query)
            if current_key is None:
                relation = "complete_inventory_null" if object_inventory_complete else "unknown"
            elif current_key == prior_key:
                relation = "same_identity"
            else:
                relation = "different_identity"
            log_likelihood_ratio = cosine * logit_scale + logit_bias
            probability = (
                1.0 / (1.0 + math.exp(-log_likelihood_ratio))
                if log_likelihood_ratio >= 0.0
                else math.exp(log_likelihood_ratio) / (1.0 + math.exp(log_likelihood_ratio))
            )
            records.append(
                {
                    "address_cosine": cosine,
                    "address_log_likelihood_ratio": log_likelihood_ratio,
                    "address_relation_probability": probability,
                    "address_relation_logit_bias": logit_bias,
                    "address_relation_logit_scale": logit_scale,
                    "current_identity_key": current_key,
                    "prior_identity_key": prior_key,
                    "prior_row": prior_row,
                    "query": query,
                    "query_existence_probability": existence,
                    "query_localization_confidence": localization_confidence,
                    "query_mask_coherence_score": mask_coherence_score,
                    "query_mask_quality": mask_quality,
                    "query_object_confidence": object_confidence,
                    "relation": relation,
                }
            )
    return records


def _binary_auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if positive.size == 0 or negative.size == 0:
        return None
    values = np.concatenate((positive, negative))
    labels = np.concatenate(
        (np.ones(positive.size, dtype=np.int8), np.zeros(negative.size, dtype=np.int8))
    )
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positive.size * (positive.size + 1) / 2.0) / (
        positive.size * negative.size
    )


def _summarize_address_relations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize discrimination and calibration without fitting on evaluation data."""

    relation_names = (
        "same_identity",
        "different_identity",
        "complete_inventory_null",
        "unknown",
    )
    grouped: dict[str, np.ndarray] = {}
    classes: dict[str, dict[str, Any]] = {}
    for relation in relation_names:
        values = np.asarray(
            [
                float(record["address_log_likelihood_ratio"])
                for record in records
                if record.get("relation") == relation
            ],
            dtype=np.float64,
        )
        grouped[relation] = values
        if values.size == 0:
            classes[relation] = {"count": 0}
            continue
        probabilities = 1.0 / (1.0 + np.exp(-np.abs(values)))
        probabilities = np.where(values >= 0.0, probabilities, 1.0 - probabilities)
        target = 1.0 if relation == "same_identity" else 0.0
        signed = values if target == 1.0 else -values
        negative_log_likelihood = np.logaddexp(0.0, -signed)
        classes[relation] = {
            "address_log_likelihood_ratio": {
                "fraction_above_zero": float(np.mean(values > 0.0)),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "minimum": float(values.min()),
                "quantiles": {
                    name: float(value)
                    for name, value in zip(
                        ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                        np.quantile(values, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
                        strict=True,
                    )
                },
            },
            "count": int(values.size),
        }
        if relation != "unknown":
            classes[relation]["brier_score"] = float(np.mean((probabilities - target) ** 2))
            classes[relation]["negative_log_likelihood"] = float(negative_log_likelihood.mean())

    positive = grouped["same_identity"]
    different = grouped["different_identity"]
    null = grouped["complete_inventory_null"]
    known_negative = np.concatenate((different, null))
    return {
        "classes": classes,
        "discrimination": {
            "same_vs_all_known_negative_auroc": _binary_auroc(positive, known_negative),
            "same_vs_complete_inventory_null_auroc": _binary_auroc(positive, null),
            "same_vs_different_identity_auroc": _binary_auroc(positive, different),
        },
        "known_pair_count": int(positive.size + known_negative.size),
        "record_count": len(records),
    }


def _trace_assignments(
    *,
    previous_keys: Sequence[str | None],
    next_keys: Sequence[str | None],
    identity_keys: Sequence[str],
    prediction_indices: Sequence[int],
    target_indices: Sequence[int],
    query_existence_probability: Sequence[float],
    query_localization_confidence: Sequence[float],
    query_mask_quality: Sequence[float],
    query_mask_coherence_score: Sequence[float],
    query_object_confidence: Sequence[float],
    query_ownership_mass: Sequence[float],
    target_ownership_mass: Sequence[float],
    observation_to_row: Sequence[int],
    event_type: Sequence[int],
    final_valid: Sequence[bool],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Explain the exact loss-track decision without changing it."""

    lengths = (len(previous_keys), len(next_keys), len(event_type), len(final_valid))
    if len(set(lengths)) != 1:
        raise ValueError("loss-track trace rows disagree")
    if len(prediction_indices) != len(target_indices):
        raise ValueError("loss-track trace match is malformed")
    if (
        len(query_existence_probability) != len(query_ownership_mass)
        or len(query_localization_confidence) != len(query_ownership_mass)
        or len(query_mask_quality) != len(query_ownership_mass)
        or len(query_mask_coherence_score) != len(query_ownership_mass)
        or len(query_object_confidence) != len(query_ownership_mass)
        or any(not 0 <= query < len(query_ownership_mass) for query in prediction_indices)
        or any(not 0 <= target < len(target_ownership_mass) for target in target_indices)
    ):
        raise ValueError("loss-track trace diagnostics are malformed")
    previous_row = {key: row for row, key in enumerate(previous_keys) if key is not None}
    retained = {row for row, event in enumerate(event_type) if event in {MATCH_EVENT, MISS_EVENT}}
    assignments: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    for query, target_index in zip(prediction_indices, target_indices, strict=True):
        key = identity_keys[target_index]
        old_row = previous_row.get(key)
        runtime_row = observation_to_row[query]
        reason: str | None = None
        if old_row is not None and old_row in retained:
            selected_row = old_row
            if runtime_row >= 0 and runtime_row != old_row:
                reason = "retained_identity_runtime_row_disagreement"
        elif not 0 <= runtime_row < len(final_valid) or not final_valid[runtime_row]:
            selected_row = None
            reason = "runtime_observation_unmapped"
        else:
            selected_row = runtime_row
            occupant = next_keys[runtime_row]
            if occupant is not None and occupant != key:
                selected_row = None
                reason = "runtime_row_already_occupied"
        record: dict[str, object] = {
            "conflict": reason is not None,
            "identity_key": key,
            "old_row": old_row,
            "query": query,
            "query_existence_probability": float(query_existence_probability[query]),
            "query_localization_confidence": float(query_localization_confidence[query]),
            "query_mask_coherence_score": float(query_mask_coherence_score[query]),
            "query_mask_quality": float(query_mask_quality[query]),
            "query_object_confidence": float(query_object_confidence[query]),
            "query_ownership_mass": float(query_ownership_mass[query]),
            "runtime_event": (
                None if runtime_row < 0 else _EVENT_NAMES.get(event_type[runtime_row], "unknown")
            ),
            "runtime_row": runtime_row,
            "selected_loss_track_row": selected_row,
            "target_index": target_index,
            "target_ownership_mass": float(target_ownership_mass[target_index]),
        }
        if reason is not None:
            record["reason"] = reason
            conflicts.append(record)
        assignments.append(record)
    return assignments, conflicts


def _color(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 212],
            [0, 128, 128],
            [220, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
        ],
        dtype=np.float32,
    )
    return palette[index % len(palette)]


def _overlay_probabilities(
    source: np.ndarray,
    probability: np.ndarray,
    color_indices: Sequence[int],
    *,
    supervised: np.ndarray | None = None,
) -> np.ndarray:
    from PIL import Image

    patch_count = probability.shape[0]
    side = math.isqrt(patch_count)
    if side * side != patch_count or probability.shape[1] != len(color_indices) + 1:
        raise ValueError("visual ownership is not one square patch grid plus context")
    labels = probability.argmax(axis=-1).reshape(side, side)
    height, width = source.shape[:2]
    resized = np.asarray(
        Image.fromarray(labels.astype(np.uint8)).resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )
    )
    result = source.astype(np.float32).copy()
    for category, color_index in enumerate(color_indices):
        selected = resized == category
        if selected.any():
            result[selected] = 0.48 * result[selected] + 0.52 * _color(color_index)
    if supervised is not None:
        known = supervised.reshape(side, side).astype(np.uint8) * 255
        known = (
            np.asarray(
                Image.fromarray(known).resize(
                    (width, height),
                    resample=Image.Resampling.NEAREST,
                )
            )
            > 0
        )
        result[~known] = 0.35 * result[~known] + 0.65 * np.asarray(
            [128, 128, 128], dtype=np.float32
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def _draw_centroids(
    image: Any,
    probability: np.ndarray,
    color_indices: Sequence[int],
) -> None:
    from PIL import ImageDraw

    side = math.isqrt(probability.shape[0])
    if side * side != probability.shape[0]:
        raise ValueError("centroid ownership is not square")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    yy, xx = np.mgrid[0:side, 0:side]
    for column, color_index in enumerate(color_indices):
        mass = probability[:, column].reshape(side, side)
        total = float(mass.sum())
        if total <= 1e-5:
            continue
        x = float((mass * (xx + 0.5)).sum() / total) / side * width
        y = float((mass * (yy + 0.5)).sum() / total) / side * height
        color = tuple(int(value) for value in _color(color_index))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2)


def _sensor_rgb(sample: Any, host_key: str) -> np.ndarray:
    source_by_host = {
        CALVIN_HOST_IMAGE_KEYS[0]: "observation.images.rgb_static",
        CALVIN_HOST_IMAGE_KEYS[1]: "observation.images.rgb_gripper",
    }
    arrays = {
        observation.key: observation.value for observation in sample.record.array_observations
    }
    value = np.asarray(arrays[source_by_host[host_key]])
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("CALVIN RGB source changed")
    return value


def _panel(title: str, array: np.ndarray, probability: np.ndarray | None, colors: Sequence[int]):
    from PIL import Image, ImageDraw

    image = Image.fromarray(array).resize((320, 320), Image.Resampling.NEAREST)
    if probability is not None:
        _draw_centroids(image, probability, colors)
    canvas = Image.new("RGB", (320, 350), "white")
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((7, 8), title, fill="black")
    return canvas


def _render_frame(
    *,
    path: Path,
    sample: Any,
    rank: int,
    one_based_step: int,
    core_output: Any,
    sequence_output: Any,
    target: Any,
    lifecycle_target: Any,
    match: Any,
    next_keys: Sequence[str | None],
    conflicts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    from PIL import Image, ImageDraw

    from picf_next.hosts.molmoact2 import MOLMO_VISION_PATCH_MODALITY

    projection = core_output.projection
    spans = tuple(span for span in projection.spans if span.modality == MOLMO_VISION_PATCH_MODALITY)
    if len(spans) != 1:
        raise ValueError("M3 temporal renderer requires one Molmo vision span")
    vision_offset = spans[0].start
    layout = sequence_output.vision_patch_layouts[0]
    if layout is None or len(layout.rows) != 1:
        raise ValueError("M3 temporal renderer requires one explicit vision layout row")

    visible_keys = tuple(target.temporal_identity_keys or ())
    inventory_keys = tuple(lifecycle_target.alive_identity_keys)
    inventory_index = {key: index for index, key in enumerate(inventory_keys)}
    visible_colors = [inventory_index[key] for key in visible_keys]
    discovery = core_output.discovery
    remapped = np.zeros(
        (discovery.ownership.shape[1], len(visible_keys) + 1),
        dtype=np.float32,
    )
    remapped[:, -1] = discovery.context_ownership[0].detach().float().cpu().numpy()
    for query, target_index in zip(
        match.prediction_indices.detach().cpu().tolist(),
        match.target_indices.detach().cpu().tolist(),
        strict=True,
    ):
        remapped[:, target_index] = discovery.ownership[0, :, query].detach().float().cpu().numpy()

    posterior = core_output.posterior
    valid_rows = posterior.belief.valid[0].detach().cpu().tolist()
    active_rows = [row for row, valid in enumerate(valid_rows) if valid]
    posterior_probability = np.zeros(
        (posterior.ownership.shape[1], len(active_rows) + 1),
        dtype=np.float32,
    )
    raw_posterior = posterior.ownership[0].detach().float().cpu().numpy()
    for column, row in enumerate(active_rows):
        posterior_probability[:, column] = raw_posterior[:, row]
    posterior_probability[:, -1] = raw_posterior[:, -1]
    posterior_colors = [
        inventory_index.get(next_keys[row], len(inventory_keys) + row) for row in active_rows
    ]
    target_probability = target.ownership.detach().float().cpu().numpy()
    target_supervised = target.supervision_valid.detach().cpu().numpy()

    rows = []
    for image_span in layout.rows[0]:
        start = vision_offset + image_span.start
        stop = vision_offset + image_span.stop
        source = _sensor_rgb(sample, image_span.image_key)
        local_target = target_probability[start:stop]
        local_matched = remapped[start:stop]
        local_posterior = posterior_probability[start:stop]
        panels = (
            _panel(f"{image_span.image_key}: source RGB", source, None, ()),
            _panel(
                f"{image_span.image_key}: loss-only target",
                _overlay_probabilities(
                    source,
                    local_target,
                    visible_colors,
                    supervised=target_supervised[start:stop],
                ),
                local_target,
                visible_colors,
            ),
            _panel(
                f"{image_span.image_key}: matched discovery",
                _overlay_probabilities(source, local_matched, visible_colors),
                local_matched,
                visible_colors,
            ),
            _panel(
                f"{image_span.image_key}: persistent posterior rows",
                _overlay_probabilities(source, local_posterior, posterior_colors),
                local_posterior,
                posterior_colors,
            ),
        )
        row = Image.new("RGB", (320 * len(panels), 350), "white")
        for index, panel in enumerate(panels):
            row.paste(panel, (320 * index, 0))
        rows.append(row)

    header = 88
    legend_height = 22 * (len(inventory_keys) + len(active_rows)) + 35
    canvas = Image.new("RGB", (1280, header + 350 * len(rows) + legend_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 8),
        (
            f"rank={rank} | step={one_based_step} | frame={sample.record.global_index} | "
            f"task={sample.host_sample.task_key} | instruction={sample.record.task}"
        ),
        fill="black",
    )
    draw.text(
        (10, 34),
        (
            f"visible objects={len(visible_keys)} | alive inventory={len(inventory_keys)} | "
            f"persistent rows={len(active_rows)} | assignment conflicts={len(conflicts)}"
        ),
        fill="black",
    )
    draw.text(
        (10, 59),
        (
            "Gray target patches are unknown and excluded from loss; "
            "masks never enter the forward path."
        ),
        fill="black",
    )
    for row_index, row in enumerate(rows):
        canvas.paste(row, (0, header + 350 * row_index))
    legend_y = header + 350 * len(rows) + 8
    for index, key in enumerate(inventory_keys):
        color = tuple(int(value) for value in _color(index))
        draw.rectangle((10, legend_y, 26, legend_y + 12), fill=color)
        visibility = "visible" if key in visible_keys else "occluded/unobserved"
        draw.text((34, legend_y - 2), f"physical {index}: {key} ({visibility})", fill="black")
        legend_y += 22
    legend_y = header + 350 * len(rows) + 8
    for column, row in enumerate(active_rows):
        color = tuple(int(value) for value in _color(posterior_colors[column]))
        event = _EVENT_NAMES.get(int(posterior.event_type[0, row]), "unknown")
        draw.rectangle((650, legend_y, 666, legend_y + 12), fill=color)
        draw.text(
            (674, legend_y - 2),
            f"posterior row={row}: key={next_keys[row]} event={event}",
            fill="black",
        )
        legend_y += 22
    if conflicts:
        draw.text((650, legend_y + 3), f"conflicts: {json.dumps(conflicts)}", fill="red")
    canvas.save(path)
    return {
        "bytes": path.stat().st_size,
        "path": path.name,
        "sha256": _sha256(path),
    }


def _checkpoint_contract(checkpoint: Path) -> tuple[dict[str, Any], Path]:
    control = _read_json(checkpoint / "picf_control.json", "checkpoint control")
    if control.get("schema") != "picf-next.checkpoint-control-manifest.v2":
        raise ValueError("unsupported training checkpoint control schema")
    state = control.get("state_files")
    if not isinstance(state, dict) or "model.safetensors" not in state:
        raise ValueError("training checkpoint has no bound model state")
    model_path = checkpoint / "model.safetensors"
    expected = state["model.safetensors"]
    if not isinstance(expected, dict) or expected.get("size_bytes") != model_path.stat().st_size:
        raise ValueError("training model size differs from checkpoint control")
    if _sha256(model_path) != expected.get("sha256"):
        raise ValueError("training model hash differs from checkpoint control")
    return control, model_path


_PROMPT_INVARIANT_VISUAL_FIELDS = (
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
)


def _maximum_absolute_tensor_delta(left: Any, right: Any) -> float:
    if left.shape != right.shape:
        return math.inf
    if left.numel() == 0:
        return 0.0
    return float((left.detach().float() - right.detach().float()).abs().max().cpu())


def _prepare_prompt_policy_batch(
    *,
    stack: Any,
    policy: Any,
    sample: Any,
    task: str,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation
    from picf_next.hosts.molmoact2_training import molmoact2_host_observation_view

    host_view = replace(molmoact2_host_observation_view(sample.record), task=task)
    raw_inputs = dict(
        stack.processor.build_observation_inputs(
            ((sample.picf_evidence_frame,),),
            (host_view,),
        )
    )
    prepared = prepare_molmoact2_lerobot_observation(policy, raw_inputs)
    target_batch = dict(stack.processor.build_action_targets((sample,)))
    collisions = sorted(set(target_batch) & set(prepared.model_inputs))
    if collisions:
        raise ValueError(f"prompt audit target/observation fields collide: {collisions}")
    policy_batch = {**target_batch, **prepared.model_inputs}
    return raw_inputs, prepared, policy_batch


def _fixed_flow_policy_probe(
    *,
    policy: Any,
    action_adapter: Any,
    policy_batch: Mapping[str, Any],
    prepared: Any,
    evidence: Any,
    flow_timesteps: Any,
    flow_noise: Any,
) -> tuple[float, Any]:
    captured: list[Any] = []

    def capture_velocity(_module: Any, _inputs: object, output: Any) -> None:
        captured.append(output.detach().float())

    final_layer = policy._action_expert().final_layer
    handle = final_layer.register_forward_hook(capture_velocity)
    try:
        context = action_adapter.prepare_picf_context(evidence)
        loss, _metrics = policy(
            dict(policy_batch),
            reduction="mean",
            action_layer_context=context,
            flow_timesteps=flow_timesteps,
            flow_noise=flow_noise,
            action_condition_input_ids=prepared.action_condition_input_ids,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("fixed-flow prompt probe did not capture exactly one velocity field")
    if loss.ndim != 0 or not loss.isfinite():
        raise ValueError("fixed-flow prompt probe produced a non-finite action loss")
    return float(loss.detach().float().cpu()), captured[0]


def _valid_velocity_rms(reference: Any, changed: Any, policy_batch: Mapping[str, Any]) -> float:
    import torch

    if reference.shape != changed.shape or reference.ndim != 3:
        raise ValueError(
            "prompt probe velocity fields must share [flow-batch, horizon, action] shape"
        )
    action = policy_batch.get("action")
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise ValueError("prompt probe requires one official action target tensor")
    batch_size, horizon, action_dim = action.shape
    if reference.shape[0] % batch_size or reference.shape[1:] != (horizon, action_dim):
        raise ValueError("prompt probe velocity field differs from the official action geometry")
    flow_count = reference.shape[0] // batch_size
    valid = torch.ones(
        (batch_size, flow_count, horizon, action_dim),
        dtype=torch.bool,
        device=reference.device,
    )
    action_dim_is_pad = policy_batch.get("action_dim_is_pad")
    if action_dim_is_pad is not None:
        if action_dim_is_pad.shape != (batch_size, action_dim):
            raise ValueError("prompt probe action-dimension padding mask is malformed")
        valid &= ~action_dim_is_pad.to(device=reference.device, dtype=torch.bool)[:, None, None, :]
    action_horizon_is_pad = policy_batch.get("action_horizon_is_pad")
    if action_horizon_is_pad is not None:
        if action_horizon_is_pad.shape != (batch_size, horizon):
            raise ValueError("prompt probe action-horizon padding mask is malformed")
        valid &= ~action_horizon_is_pad.to(device=reference.device, dtype=torch.bool)[
            :, None, :, None
        ]
    difference = (reference - changed).reshape(batch_size, flow_count, horizon, action_dim)
    selected = difference[valid]
    if selected.numel() == 0:
        raise ValueError("prompt probe has no valid action coordinates")
    return float(selected.square().mean().sqrt().cpu())


def _audit_prompt_causality(
    *,
    stack: Any,
    sequence: Any,
    sample: Any,
    planned_transition: Any,
    next_keys: Sequence[str | None],
    task_candidates: Sequence[tuple[str, str]],
    explicit_counterfactual_task: str | None,
) -> dict[str, Any]:
    import torch

    from picf_next.hosts.interventions import without_object_rows
    from picf_next.hosts.molmoact2 import MOLMO_VISION_PATCH_MODALITY
    from picf_next.hosts.molmoact2_training import materialize_molmoact2_flow_randomness

    original_task = sample.record.task
    distinct_task_candidates = tuple(
        instruction
        for task_key, instruction in task_candidates
        if task_key != sample.host_sample.task_key
    )
    counterfactual_task = (
        explicit_counterfactual_task
        if explicit_counterfactual_task is not None
        else _select_counterfactual_task(original_task, distinct_task_candidates)
    )
    if counterfactual_task == original_task:
        raise ValueError("counterfactual task must differ from the recorded task")

    policy = stack.module.joint_bridge.sequence_bridge.policy
    action_adapter = stack.module.joint_bridge.sequence_bridge.action_adapter
    original_raw, original_prepared, original_batch = _prepare_prompt_policy_batch(
        stack=stack,
        policy=policy,
        sample=sample,
        task=original_task,
    )
    swapped_raw, swapped_prepared, swapped_batch = _prepare_prompt_policy_batch(
        stack=stack,
        policy=policy,
        sample=sample,
        task=counterfactual_task,
    )
    for field in ("action", "action_dim_is_pad", "action_horizon_is_pad"):
        left = original_batch.get(field)
        right = swapped_batch.get(field)
        if left is None or right is None or not torch.equal(left, right):
            raise RuntimeError(f"prompt swap changed official target field {field}")
    visual_input_equal: dict[str, bool] = {}
    for field in _PROMPT_INVARIANT_VISUAL_FIELDS:
        left = original_raw.get(field)
        right = swapped_raw.get(field)
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"prompt swap changed the presence of {field}")
            continue
        equal = bool(torch.equal(left, right))
        visual_input_equal[field] = equal
        if not equal:
            raise RuntimeError(f"prompt swap changed physical visual input {field}")
    left_ids = original_raw["input_ids"]
    right_ids = swapped_raw["input_ids"]
    token_sequence_changed = left_ids.shape != right_ids.shape or not torch.equal(
        left_ids, right_ids
    )
    if not token_sequence_changed:
        raise RuntimeError("counterfactual task did not change the Molmo token sequence")

    original_patch_bank = original_prepared.vision_patch_bank
    swapped_patch_bank = swapped_prepared.vision_patch_bank
    if original_patch_bank is None or swapped_patch_bank is None:
        raise RuntimeError("prompt causal audit requires same-forward Molmo vision patches")
    if not torch.equal(original_patch_bank.valid, swapped_patch_bank.valid):
        raise RuntimeError("prompt swap changed native vision-patch validity")
    prompt_patch_delta = _maximum_absolute_tensor_delta(
        original_patch_bank.tokens,
        swapped_patch_bank.tokens,
    )
    if prompt_patch_delta != 0.0:
        raise RuntimeError("prompt swap changed task-independent native vision patches")

    evidence = sequence.evidences[0]
    evidence_patch_banks = tuple(
        bank for bank in evidence.dense_banks if bank.modality == MOLMO_VISION_PATCH_MODALITY
    )
    if len(evidence_patch_banks) != 1:
        raise RuntimeError("prompt audit expected exactly one deployed Molmo vision bank")
    replay_patch_delta = _maximum_absolute_tensor_delta(
        original_patch_bank.tokens,
        evidence_patch_banks[0].tokens,
    )

    action = original_batch.get("action")
    if not isinstance(action, torch.Tensor):
        raise ValueError("prompt audit lost the official action target")
    flow_timesteps, flow_noise = materialize_molmoact2_flow_randomness(
        policy,
        (planned_transition.sample,),
        action,
        transition_index=(planned_transition.transition_index,),
    )
    original_loss, original_velocity = _fixed_flow_policy_probe(
        policy=policy,
        action_adapter=action_adapter,
        policy_batch=original_batch,
        prepared=original_prepared,
        evidence=evidence,
        flow_timesteps=flow_timesteps,
        flow_noise=flow_noise,
    )
    swapped_loss, swapped_velocity = _fixed_flow_policy_probe(
        policy=policy,
        action_adapter=action_adapter,
        policy_batch=swapped_batch,
        prepared=swapped_prepared,
        evidence=evidence,
        flow_timesteps=flow_timesteps,
        flow_noise=flow_noise,
    )
    captured_original_loss = float(sequence.action_losses[0].detach().float().cpu())
    replay_loss_delta = abs(original_loss - captured_original_loss)
    if replay_loss_delta > 1e-6:
        raise RuntimeError(
            "reconstructed fixed-noise action loss differs from the captured production loss"
        )

    object_valid = evidence.object_valid
    if object_valid is None or object_valid.shape[0] != 1:
        raise ValueError("prompt row audit requires one complete posterior bank")
    if len(next_keys) != object_valid.shape[1]:
        raise ValueError("prompt row audit identities differ from posterior capacity")
    rows: list[dict[str, Any]] = []
    for row in object_valid[0].nonzero(as_tuple=False).flatten().tolist():
        selector = torch.zeros_like(object_valid)
        selector[0, row] = True
        removed = without_object_rows(evidence, selector)
        removed_original_loss, removed_original_velocity = _fixed_flow_policy_probe(
            policy=policy,
            action_adapter=action_adapter,
            policy_batch=original_batch,
            prepared=original_prepared,
            evidence=removed,
            flow_timesteps=flow_timesteps,
            flow_noise=flow_noise,
        )
        removed_swapped_loss, removed_swapped_velocity = _fixed_flow_policy_probe(
            policy=policy,
            action_adapter=action_adapter,
            policy_batch=swapped_batch,
            prepared=swapped_prepared,
            evidence=removed,
            flow_timesteps=flow_timesteps,
            flow_noise=flow_noise,
        )
        identity_key = next_keys[row]
        rows.append(
            {
                "counterfactual_loss_delta": removed_swapped_loss - swapped_loss,
                "counterfactual_semantic_identity_score": _semantic_identity_score(
                    counterfactual_task, identity_key
                ),
                "counterfactual_velocity_effect_rms": _valid_velocity_rms(
                    swapped_velocity,
                    removed_swapped_velocity,
                    swapped_batch,
                ),
                "identity_key": identity_key,
                "original_loss_delta": removed_original_loss - original_loss,
                "original_semantic_identity_score": _semantic_identity_score(
                    original_task, identity_key
                ),
                "original_velocity_effect_rms": _valid_velocity_rms(
                    original_velocity,
                    removed_original_velocity,
                    original_batch,
                ),
                "posterior_row": row,
            }
        )

    def summarize_prompt(prefix: str) -> dict[str, Any]:
        if not rows:
            return {"causal_top_row": None, "semantic_best_rows": []}
        effect_name = f"{prefix}_velocity_effect_rms"
        score_name = f"{prefix}_semantic_identity_score"
        causal_top = max(
            rows,
            key=lambda item: (float(item[effect_name]), -int(item["posterior_row"])),
        )
        maximum_score = max(int(item[score_name]) for item in rows)
        semantic_best = [
            int(item["posterior_row"])
            for item in rows
            if maximum_score > 0 and int(item[score_name]) == maximum_score
        ]
        return {
            "causal_top_identity_key": causal_top["identity_key"],
            "causal_top_is_semantic_best": (
                None if not semantic_best else int(causal_top["posterior_row"]) in semantic_best
            ),
            "causal_top_row": int(causal_top["posterior_row"]),
            "maximum_semantic_identity_score": maximum_score,
            "semantic_best_rows": semantic_best,
        }

    return {
        "captured_original_action_loss": captured_original_loss,
        "counterfactual_action_loss_on_original_target": swapped_loss,
        "counterfactual_summary": summarize_prompt("counterfactual"),
        "counterfactual_task": counterfactual_task,
        "fixed_noise_prompt_velocity_difference_rms": _valid_velocity_rms(
            original_velocity,
            swapped_velocity,
            original_batch,
        ),
        "original_action_loss_replay": original_loss,
        "original_action_loss_replay_abs_delta": replay_loss_delta,
        "original_summary": summarize_prompt("original"),
        "original_task": original_task,
        "production_evidence_patch_replay_max_abs_delta": replay_patch_delta,
        "prompt_swap_native_patch_max_abs_delta": prompt_patch_delta,
        "prompt_token_sequence_changed": token_sequence_changed,
        "rows": rows,
        "visual_input_bitwise_equal": visual_input_equal,
    }


def main() -> None:
    args = _parse_args()
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from safetensors.torch import load_model

    from picf_next.data.vjepa2_cache import Vjepa2FeatureCache
    from picf_next.hosts.vjepa2_context import CalvinVjepa2CachedContextBuilder
    from picf_next.models.temporal import empty_object_belief
    from picf_next.training.control import EpisodeSampleSequence, FrozenEpisodeStreamPlan
    from picf_next.training.molmoact2_calvin import (
        build_calvin_episode_stream_plan,
        build_molmoact2_calvin_training_stack,
        build_molmoact2_policy_config,
        load_calvin_training_assets,
    )
    from picf_next.training.recipe import load_training_recipe

    checkpoint = args.training_checkpoint.expanduser().resolve()
    control, model_path = _checkpoint_contract(checkpoint)
    state_prefix_checkpoint: Path | None = None
    state_prefix_control: dict[str, Any] | None = None
    state_prefix_model_path: Path | None = None
    state_prefix_steps = 0
    state_prefix_plan_steps = 0
    if args.state_prefix_checkpoint is not None:
        state_prefix_checkpoint = args.state_prefix_checkpoint.expanduser().resolve()
        state_prefix_control, state_prefix_model_path = _checkpoint_contract(
            state_prefix_checkpoint
        )
        state_prefix_steps, state_prefix_plan_steps = _validate_state_handoff_controls(
            control,
            state_prefix_control,
        )
    contract = control.get("contract")
    progress = control.get("progress")
    if not isinstance(contract, dict) or not isinstance(progress, dict):
        raise ValueError("training checkpoint contract/progress is malformed")
    arm = contract.get("arm_config", {}).get("causal_factorization")
    if not isinstance(arm, dict) or arm.get("id") != "D":
        raise ValueError("temporal audit currently requires the full Arm D checkpoint")
    checkpoint_steps = progress.get("attempted_optimizer_steps")
    if (
        not isinstance(checkpoint_steps, int)
        or checkpoint_steps <= 0
        or progress.get("successful_optimizer_steps") != checkpoint_steps
    ):
        raise ValueError("training checkpoint is not a completed successful prefix")
    checkpoint_plan_steps = control.get("plan", {}).get("total_steps")
    if (
        not isinstance(checkpoint_plan_steps, int)
        or isinstance(checkpoint_plan_steps, bool)
        or checkpoint_plan_steps < checkpoint_steps
    ):
        raise ValueError("checkpoint training-plan length is malformed")
    if args.source_disjoint_segments is not None and args.replay_steps is None:
        raise ValueError("source-disjoint evaluation requires an explicit replay length")
    if state_prefix_control is not None and args.source_disjoint_segments is not None:
        raise ValueError("state-handoff replay requires the checkpoint training plan")
    replay_steps = checkpoint_steps if args.replay_steps is None else args.replay_steps
    if not isinstance(replay_steps, int) or isinstance(replay_steps, bool) or replay_steps <= 0:
        raise ValueError("replay steps must be a positive integer")
    if state_prefix_control is not None and state_prefix_steps >= replay_steps:
        raise ValueError("state-handoff replay must continue beyond the state-prefix checkpoint")
    render_steps = _parse_render_steps(args.render_steps, replay_steps)
    prompt_causal_steps = _parse_optional_steps(args.prompt_causal_steps, replay_steps)
    if args.counterfactual_task is not None and not args.counterfactual_task.strip():
        raise ValueError("explicit counterfactual task cannot be empty")
    if args.counterfactual_task is not None and not prompt_causal_steps:
        raise ValueError("explicit counterfactual task requires prompt causal steps")
    ranks = tuple(args.ranks)
    world_size = int(contract["world_size"])
    if (
        not ranks
        or len(set(ranks)) != len(ranks)
        or any(not 0 <= rank < world_size for rank in ranks)
    ):
        raise ValueError("audit ranks must be unique checkpoint world ranks")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir()
    recipe = load_training_recipe(args.recipe.resolve())
    if recipe.recipe_sha256 != contract["common_config"]["recipe_sha256"]:
        raise ValueError("audit recipe differs from the training checkpoint")
    assets = load_calvin_training_assets(
        recipe,
        repository_root=_ROOT,
        split_root=args.dataset_split_root,
    )
    task_candidates = tuple(
        sorted(
            {
                (
                    assets.dataset.by_key(episode.sample_keys[0]).host_sample.task_key,
                    assets.dataset.by_key(episode.sample_keys[0]).record.task,
                )
                for episode in assets.dataset.episode_manifest
                if episode.sample_keys
            }
        )
    )
    checkpoint_plan = build_calvin_episode_stream_plan(
        recipe,
        assets.dataset,
        comparison_id=str(contract["comparison_id"]),
        seed=int(control["plan"]["seed"]),
        global_batch_size=int(contract["optimizer_global_batch_size"]),
        total_steps=checkpoint_plan_steps,
    )
    if checkpoint_plan.plan_sha256 != control["plan_sha256"]:
        raise ValueError("reconstructed audit plan differs from checkpoint")
    if state_prefix_control is not None:
        state_prefix_contract = state_prefix_control["contract"]
        state_prefix_plan = build_calvin_episode_stream_plan(
            recipe,
            assets.dataset,
            comparison_id=str(state_prefix_contract["comparison_id"]),
            seed=int(state_prefix_control["plan"]["seed"]),
            global_batch_size=int(state_prefix_contract["optimizer_global_batch_size"]),
            total_steps=state_prefix_plan_steps,
        )
        if state_prefix_plan.plan_sha256 != state_prefix_control["plan_sha256"]:
            raise ValueError("reconstructed state-prefix plan differs from checkpoint")
        _validate_extended_plan_prefix(state_prefix_plan, checkpoint_plan)
    plan = checkpoint_plan
    evaluation_mode = "checkpoint_plan"
    evaluation_segments: tuple[int, ...] = ()
    source_disjoint_from_checkpoint_prefix = False
    validated_prefix_steps = checkpoint_plan_steps
    if args.source_disjoint_segments is not None:
        evaluation_segments = tuple(args.source_disjoint_segments)
        if (
            not evaluation_segments
            or len(set(evaluation_segments)) != len(evaluation_segments)
            or any(index < 0 for index in evaluation_segments)
        ):
            raise ValueError("source-disjoint segment indices must be unique and nonnegative")
        selected = tuple(
            episode
            for episode in assets.dataset.episode_manifest
            if episode.segment_index in set(evaluation_segments)
        )
        if len(selected) != len(evaluation_segments):
            found = {episode.segment_index for episode in selected}
            raise ValueError(
                f"unknown source-disjoint segments: {sorted(set(evaluation_segments) - found)}"
            )
        global_batch_size = int(contract["optimizer_global_batch_size"])
        if len(selected) < global_batch_size:
            raise ValueError(
                "source-disjoint evaluation requires at least one episode per global lane"
            )
        plan = FrozenEpisodeStreamPlan(
            dataset_id=recipe.dataset.dataset_id,
            dataset_revision=recipe.dataset.dataset_revision,
            dataset_manifest_sha256=recipe.artifacts.dataset_tree_sha256,
            episodes=tuple(
                EpisodeSampleSequence(
                    episode_key=episode.episode_key,
                    sample_keys=episode.sample_keys,
                )
                for episode in selected
            ),
            comparison_id=(
                f"{contract['comparison_id']}/source-disjoint-identity-null-calibration-v1/"
                + "-".join(str(index) for index in sorted(evaluation_segments))
            ),
            seed=int(control["plan"]["seed"]),
            global_batch_size=global_batch_size,
            total_steps=replay_steps,
        )
        consumed_training_keys = {
            transition.sample.sample_key
            for step in range(checkpoint_steps)
            for transition in checkpoint_plan.global_batch(step).transitions
        }
        evaluation_keys = {
            transition.sample.sample_key
            for step in range(replay_steps)
            for transition in plan.global_batch(step).transitions
        }
        training_source_frames = {
            assets.dataset.by_key(key).record.global_index for key in consumed_training_keys
        }
        evaluation_source_frames = {
            assets.dataset.by_key(key).record.global_index for key in evaluation_keys
        }
        if consumed_training_keys & evaluation_keys:
            raise ValueError("source-disjoint evaluation reused a checkpoint training sample")
        overlap = training_source_frames & evaluation_source_frames
        if overlap:
            raise ValueError(
                f"source-disjoint evaluation reused checkpoint source frames: {sorted(overlap)[:8]}"
            )
        evaluation_mode = "source_disjoint_segments"
        source_disjoint_from_checkpoint_prefix = True
        validated_prefix_steps = 0
    elif replay_steps > checkpoint_plan_steps:
        recipe.assert_optimizer_steps_authorized(replay_steps)
        plan = build_calvin_episode_stream_plan(
            recipe,
            assets.dataset,
            comparison_id=str(contract["comparison_id"]),
            seed=int(control["plan"]["seed"]),
            global_batch_size=int(contract["optimizer_global_batch_size"]),
            total_steps=replay_steps,
        )
        validated_prefix_steps = _validate_extended_plan_prefix(checkpoint_plan, plan)

    vjepa_contract = contract["arm_config"].get("vjepa2_cache")
    if not isinstance(vjepa_contract, dict):
        raise ValueError("Arm D checkpoint has no V-JEPA cache binding")
    cache = Vjepa2FeatureCache.load(
        args.vjepa2_cache_root.expanduser().resolve(),
        manifest_sha256=str(vjepa_contract["manifest_sha256"]),
        dataset_tree_sha256=assets.dataset_manifest.tree_sha256,
        memory_capacity=64,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M3 temporal replay requires CUDA")
    policy_config = build_molmoact2_policy_config(
        recipe,
        checkpoint_path=args.foundation_checkpoint_dir.expanduser().resolve(),
    )
    policy = MolmoAct2Policy(policy_config).to(device).eval()
    parameter = next(policy.parameters())
    native_builder = CalvinVjepa2CachedContextBuilder(
        cache,
        device=device,
        dtype=parameter.dtype,
    )
    stack = build_molmoact2_calvin_training_stack(
        recipe,
        policy=policy,
        assets=assets,
        build_native_banks=native_builder,
        native_evidence_history_frames=native_builder.maximum_source_frames,
        action_context_token_dims=native_builder.token_dims,
        include_posterior_action_context=True,
    )
    module = stack.module.to(device).eval()
    # Molmo may refresh rotary buffers during a forward pass.  InferenceMode
    # marks those refreshed tensors as inference tensors, which cannot later
    # be overwritten by load_state_dict.  A state-handoff replay deliberately
    # reloads weights mid-stream, so it must retain ordinary no-grad buffers.
    reload_weights_midstream = state_prefix_model_path is not None

    # safetensors accepts a device string or integer, not a torch.device object.
    def strict_load_weights(path: Path) -> None:
        missing, unexpected = load_model(module, path, strict=True, device=str(device))
        if missing or unexpected:
            raise RuntimeError("strict model load unexpectedly reported key drift")

    strict_load_weights(model_path)
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    captured: dict[str, Any] = {}

    def capture_sequence(_module: Any, _inputs: object, output: Any) -> None:
        captured["sequence"] = output

    def capture_targets(_module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        captured["core_outputs"] = inputs[0]
        captured["set_targets"] = kwargs["set_targets"]
        captured["lifecycle_targets"] = kwargs["lifecycle_targets"]

    sequence_handle = module.joint_bridge.sequence_bridge.register_forward_hook(capture_sequence)
    target_handle = module.joint_bridge.objective.register_forward_pre_hook(
        capture_targets,
        with_kwargs=True,
    )
    report_rows: list[dict[str, object]] = []
    relation_pairs: list[dict[str, Any]] = []
    prompt_causal_audits: list[dict[str, Any]] = []
    visual_artifacts: list[dict[str, object]] = []
    try:
        for rank in ranks:
            if state_prefix_model_path is not None:
                strict_load_weights(state_prefix_model_path)
            belief = empty_object_belief(
                recipe.core_config.temporal,
                batch_size=1,
                capacity=recipe.core_config.posterior_capacity,
                device=device,
                dtype=parameter.dtype,
            )
            keys: tuple[tuple[str | None, ...], ...] = (
                (None,) * recipe.core_config.posterior_capacity,
            )
            previous_episode_instance_id: str | None = None
            for zero_based_step in range(replay_steps):
                if state_prefix_model_path is not None and zero_based_step == state_prefix_steps:
                    strict_load_weights(model_path)
                captured.clear()
                microbatch = plan.microbatch_for_rank(
                    zero_based_step,
                    rank=rank,
                    world_size=world_size,
                    gradient_accumulation_steps=int(contract["gradient_accumulation_steps"]),
                    accumulation_index=0,
                )
                if len(microbatch.transitions) != 1:
                    raise ValueError("temporal audit requires one transition per rank")
                transition = microbatch.transitions[0]
                sample = assets.dataset.by_key(transition.sample.sample_key)
                episode_reset = transition.episode_instance_id != previous_episode_instance_id
                if episode_reset and previous_episode_instance_id is not None:
                    belief = empty_object_belief(
                        recipe.core_config.temporal,
                        batch_size=1,
                        capacity=recipe.core_config.posterior_capacity,
                        device=device,
                        dtype=parameter.dtype,
                    )
                    keys = ((None,) * recipe.core_config.posterior_capacity,)
                with (
                    _evaluation_context(
                        torch,
                        reload_weights_midstream=reload_weights_midstream,
                    ),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    stateful = module(microbatch, belief, keys)
                sequence = captured.pop("sequence")
                core_output = captured.pop("core_outputs")[0]
                target = captured.pop("set_targets")[0][0]
                lifecycle_target = captured.pop("lifecycle_targets")[0][0]
                if captured:
                    raise RuntimeError("unexpected temporal audit capture payload")
                match = module.joint_bridge.objective.set_criterion(
                    core_output.discovery,
                    (target,),
                ).matches[0]
                next_keys = stateful.final_loss_track_keys_by_row
                if next_keys is None:
                    raise RuntimeError("M3 temporal replay lost loss-track keys")
                one_based_step = zero_based_step + 1
                if one_based_step in prompt_causal_steps:
                    with (
                        _evaluation_context(
                            torch,
                            reload_weights_midstream=reload_weights_midstream,
                        ),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        prompt_audit = _audit_prompt_causality(
                            stack=stack,
                            sequence=sequence,
                            sample=sample,
                            planned_transition=transition,
                            next_keys=next_keys[0],
                            task_candidates=task_candidates,
                            explicit_counterfactual_task=args.counterfactual_task,
                        )
                    prompt_audit.update(
                        {
                            "episode_key": sample.episode_key,
                            "frame": sample.record.global_index,
                            "rank": rank,
                            "sample_key": sample.sample_key,
                            "step": one_based_step,
                            "task_key": sample.host_sample.task_key,
                        }
                    )
                    prompt_causal_audits.append(prompt_audit)
                posterior = core_output.posterior
                query_existence = core_output.discovery.existence[0].detach().float().cpu().tolist()
                query_localization_confidence = (
                    core_output.discovery.localization_confidence[0].detach().float().cpu().tolist()
                )
                query_mask_quality = (
                    core_output.discovery.mask_quality[0].detach().float().cpu().tolist()
                )
                query_mask_coherence_score = (
                    core_output.discovery.mask_coherence_score[0].detach().float().cpu().tolist()
                )
                query_object_confidence = (
                    core_output.discovery.object_confidence[0].detach().float().cpu().tolist()
                )
                query_ownership_mass = (
                    core_output.discovery.ownership[0, :, :-1]
                    .detach()
                    .float()
                    .sum(dim=0)
                    .cpu()
                    .tolist()
                )
                target_ownership_mass = target.ownership.detach().float().sum(dim=0).cpu().tolist()
                observation_to_row = posterior.observation_to_posterior[0].detach().cpu().tolist()
                event_type = posterior.event_type[0].detach().cpu().tolist()
                final_valid = posterior.belief.valid[0].detach().cpu().tolist()
                match_probability = posterior.match_probability[0].detach().float()
                null_probability = posterior.null_probability[0].detach().float()
                prior_address = torch.nn.functional.normalize(
                    posterior.prior_prediction.belief.address_mean[0].detach().float(),
                    dim=-1,
                )
                observation_address = torch.nn.functional.normalize(
                    core_output.discovery.address_mean[0].detach().float(),
                    dim=-1,
                )
                address_cosine = prior_address @ observation_address.transpose(0, 1)
                geometry_residual_l2 = torch.cdist(
                    posterior.prior_prediction.belief.geometry_mean[0]
                    .detach()
                    .float()
                    .unsqueeze(0),
                    core_output.discovery.geometry_mean[0].detach().float().unsqueeze(0),
                ).squeeze(0)
                frame_relation_pairs = _trace_address_relation_pairs(
                    previous_keys=keys[0],
                    current_identity_keys=tuple(target.temporal_identity_keys or ()),
                    prediction_indices=match.prediction_indices.detach().cpu().tolist(),
                    target_indices=match.target_indices.detach().cpu().tolist(),
                    object_inventory_complete=target.object_inventory_complete,
                    address_cosine=address_cosine.detach().cpu().tolist(),
                    query_existence_probability=query_existence,
                    query_localization_confidence=query_localization_confidence,
                    query_mask_quality=query_mask_quality,
                    query_mask_coherence_score=query_mask_coherence_score,
                    query_object_confidence=query_object_confidence,
                    logit_scale=float(
                        posterior.address_relation_logit_scale.detach().float().item()
                    ),
                    logit_bias=float(posterior.address_relation_logit_bias.detach().float().item()),
                )
                assignments, conflicts = _trace_assignments(
                    previous_keys=keys[0],
                    next_keys=next_keys[0],
                    identity_keys=tuple(target.temporal_identity_keys or ()),
                    prediction_indices=match.prediction_indices.detach().cpu().tolist(),
                    target_indices=match.target_indices.detach().cpu().tolist(),
                    query_existence_probability=query_existence,
                    query_localization_confidence=query_localization_confidence,
                    query_mask_quality=query_mask_quality,
                    query_mask_coherence_score=query_mask_coherence_score,
                    query_object_confidence=query_object_confidence,
                    query_ownership_mass=query_ownership_mass,
                    target_ownership_mass=target_ownership_mass,
                    observation_to_row=observation_to_row,
                    event_type=event_type,
                    final_valid=final_valid,
                )
                query_by_row = _invert_observation_to_row(
                    observation_to_row,
                    capacity=recipe.core_config.posterior_capacity,
                )
                lifecycle_visibility = (
                    None
                    if lifecycle_target.visibility is None
                    else lifecycle_target.visibility.detach().float().cpu().tolist()
                )
                lifecycle_visibility_supervised = (
                    None
                    if lifecycle_target.visibility_supervised is None
                    else lifecycle_target.visibility_supervised.detach().cpu().tolist()
                )
                lifecycle_index = {
                    key: index for index, key in enumerate(lifecycle_target.alive_identity_keys)
                }
                row_traces: list[dict[str, object]] = []
                for posterior_row, (old_key, next_key, query) in enumerate(
                    zip(keys[0], next_keys[0], query_by_row, strict=True)
                ):
                    identity_key = next_key if next_key is not None else old_key
                    target_index = lifecycle_index.get(identity_key)
                    row_match_probability = match_probability[posterior_row]
                    maximum_match_probability, maximum_match_query_tensor = (
                        row_match_probability.max(dim=0)
                    )
                    maximum_match_query = int(maximum_match_query_tensor.cpu())
                    selected_address_cosine = (
                        None if query is None else float(address_cosine[posterior_row, query].cpu())
                    )
                    row_traces.append(
                        {
                            "association_null_probability": float(
                                null_probability[posterior_row].cpu()
                            ),
                            "association_real_match_mass": float(row_match_probability.sum().cpu()),
                            "diagnostic_selected_address_cosine": selected_address_cosine,
                            "diagnostic_selected_address_log_likelihood_ratio": (
                                None
                                if selected_address_cosine is None
                                else selected_address_cosine
                                / recipe.core_config.temporal.association_address_temperature
                                + recipe.core_config.temporal.association_address_logit_bias
                            ),
                            "diagnostic_selected_geometry_residual_l2": (
                                None
                                if query is None
                                else float(geometry_residual_l2[posterior_row, query].cpu())
                            ),
                            "diagnostic_selected_match_probability": (
                                None if query is None else float(row_match_probability[query].cpu())
                            ),
                            "event": _EVENT_NAMES.get(
                                int(event_type[posterior_row]),
                                "unknown",
                            ),
                            "identity_key": identity_key,
                            "innovation_l2": float(
                                posterior.innovation[0, posterior_row].detach().float().norm().cpu()
                            ),
                            "observation_existence_probability": (
                                None if query is None else float(query_existence[query])
                            ),
                            "observation_localization_confidence": (
                                None
                                if query is None
                                else float(query_localization_confidence[query])
                            ),
                            "observation_mask_coherence_score": (
                                None if query is None else float(query_mask_coherence_score[query])
                            ),
                            "observation_mask_quality": (
                                None if query is None else float(query_mask_quality[query])
                            ),
                            "observation_object_confidence": (
                                None if query is None else float(query_object_confidence[query])
                            ),
                            "observation_ownership_mass": (
                                None if query is None else float(query_ownership_mass[query])
                            ),
                            "observation_query": query,
                            "maximum_match_address_cosine": float(
                                address_cosine[posterior_row, maximum_match_query].cpu()
                            ),
                            "maximum_match_geometry_residual_l2": float(
                                geometry_residual_l2[posterior_row, maximum_match_query].cpu()
                            ),
                            "maximum_match_observation_existence_probability": float(
                                query_existence[maximum_match_query]
                            ),
                            "maximum_match_observation_localization_confidence": float(
                                query_localization_confidence[maximum_match_query]
                            ),
                            "maximum_match_observation_mask_coherence_score": float(
                                query_mask_coherence_score[maximum_match_query]
                            ),
                            "maximum_match_observation_mask_quality": float(
                                query_mask_quality[maximum_match_query]
                            ),
                            "maximum_match_observation_object_confidence": float(
                                query_object_confidence[maximum_match_query]
                            ),
                            "maximum_match_probability": float(maximum_match_probability.cpu()),
                            "maximum_match_query": maximum_match_query,
                            "posterior_existence_probability": float(
                                posterior.belief.existence[0, posterior_row].detach().float().cpu()
                            ),
                            "posterior_geometry_variance_mean": float(
                                posterior.belief.geometry_covariance_diag[0, posterior_row]
                                .detach()
                                .float()
                                .mean()
                                .cpu()
                            ),
                            "posterior_ownership_mass": float(
                                posterior.ownership[0, :, posterior_row]
                                .detach()
                                .float()
                                .sum()
                                .cpu()
                            ),
                            "posterior_row": posterior_row,
                            "posterior_visibility_probability": float(
                                posterior.belief.visibility[0, posterior_row].detach().float().cpu()
                            ),
                            "prior_existence_probability": float(
                                posterior.prior_prediction.belief.existence[0, posterior_row]
                                .detach()
                                .float()
                                .cpu()
                            ),
                            "previous_conditional_visibility_probability": float(
                                torch.sigmoid(
                                    belief.visibility_given_existence_logits[0, posterior_row]
                                    .detach()
                                    .float()
                                ).cpu()
                            ),
                            "prior_key": old_key,
                            "prior_conditional_detection_probability": float(
                                torch.sigmoid(
                                    posterior.prior_prediction.belief.visibility_given_existence_logits[
                                        0, posterior_row
                                    ]
                                    .detach()
                                    .float()
                                ).cpu()
                            ),
                            "prior_visibility_probability": float(
                                posterior.prior_prediction.belief.visibility[0, posterior_row]
                                .detach()
                                .float()
                                .cpu()
                            ),
                            "target_visibility": (
                                None
                                if lifecycle_visibility is None or target_index is None
                                else float(lifecycle_visibility[target_index])
                            ),
                            "target_visibility_supervised": (
                                False
                                if lifecycle_visibility_supervised is None or target_index is None
                                else bool(lifecycle_visibility_supervised[target_index])
                            ),
                            "valid": bool(final_valid[posterior_row]),
                        }
                    )
                reported_conflicts = int(stateful.metrics["picf_loss_track_assignment_conflicts"])
                if len(conflicts) != reported_conflicts:
                    raise RuntimeError("independent conflict trace differs from objective")
                for record in frame_relation_pairs:
                    record.update(
                        {
                            "episode_key": sample.episode_key,
                            "frame": sample.record.global_index,
                            "rank": rank,
                            "step": one_based_step,
                            "task_key": sample.host_sample.task_key,
                        }
                    )
                relation_pairs.extend(frame_relation_pairs)
                row: dict[str, object] = {
                    "assignments": assignments,
                    "conflicts": conflicts,
                    "episode_key": sample.episode_key,
                    "episode_reset": episode_reset,
                    "frame": sample.record.global_index,
                    "instruction": sample.record.task,
                    "loss_action": stateful.metrics["picf_loss_action"],
                    "loss_set": stateful.metrics["picf_loss_set"],
                    "next_keys_by_row": list(next_keys[0]),
                    "posterior_event_by_row": [
                        _EVENT_NAMES.get(int(value), "unknown")
                        for value in posterior.event_type[0].detach().cpu().tolist()
                    ],
                    "rank": rank,
                    "relation_pair_count": len(frame_relation_pairs),
                    "row_traces": row_traces,
                    "sample_key": sample.sample_key,
                    "step": one_based_step,
                    "task_key": sample.host_sample.task_key,
                    "visible_identity_keys": list(target.temporal_identity_keys or ()),
                    "within_checkpoint_optimizer_prefix": (
                        evaluation_mode == "checkpoint_plan" and one_based_step <= checkpoint_steps
                    ),
                    "weight_phase": (
                        "state_prefix"
                        if state_prefix_model_path is not None
                        and one_based_step <= state_prefix_steps
                        else "primary"
                    ),
                }
                report_rows.append(row)
                if one_based_step in render_steps:
                    safe_task = _SAFE_NAME.sub("_", sample.host_sample.task_key.lower()).strip("_")
                    filename = (
                        f"rank{rank}_step{one_based_step:04d}_frame"
                        f"{sample.record.global_index:07d}_{safe_task}.png"
                    )
                    artifact = _render_frame(
                        path=visuals_dir / filename,
                        sample=sample,
                        rank=rank,
                        one_based_step=one_based_step,
                        core_output=core_output,
                        sequence_output=sequence,
                        target=target,
                        lifecycle_target=lifecycle_target,
                        match=match,
                        next_keys=next_keys[0],
                        conflicts=conflicts,
                    )
                    artifact.update(
                        {
                            "frame": sample.record.global_index,
                            "rank": rank,
                            "step": one_based_step,
                            "task_key": sample.host_sample.task_key,
                        }
                    )
                    visual_artifacts.append(artifact)
                belief = stateful.final_belief
                keys = next_keys
                previous_episode_instance_id = transition.episode_instance_id
    finally:
        sequence_handle.remove()
        target_handle.remove()

    reason_counts: dict[str, int] = {}
    for row in report_rows:
        for conflict in row["conflicts"]:  # type: ignore[union-attr]
            reason = str(conflict["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report = {
        "audit_code_revision": _git_revision(_ROOT),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint_code_revision": contract["code_revision"],
        "checkpoint_model_sha256": control["state_files"]["model.safetensors"]["sha256"],
        "checkpoint_optimizer_steps": checkpoint_steps,
        "checkpoint_plan_sha256": checkpoint_plan.plan_sha256,
        "checkpoint_plan_steps": checkpoint_plan_steps,
        "conflict_reason_counts": reason_counts,
        "evaluation_mode": (
            "checkpoint_plan_state_handoff"
            if state_prefix_model_path is not None
            else evaluation_mode
        ),
        "evaluation_segments": list(evaluation_segments),
        "identity_null_relation_calibration": _summarize_address_relations(relation_pairs),
        "prompt_causal_audits": prompt_causal_audits,
        "prompt_causal_steps": sorted(prompt_causal_steps),
        "replay_plan_sha256": plan.plan_sha256,
        "replay_steps_per_rank": replay_steps,
        "ranks": list(ranks),
        "relation_pairs": relation_pairs,
        "rows": report_rows,
        "schema": "picf-next.molmoact2-m3-temporal-audit.v10",
        "source_disjoint_from_checkpoint_prefix": source_disjoint_from_checkpoint_prefix,
        "state_handoff": (
            None
            if state_prefix_control is None or state_prefix_checkpoint is None
            else {
                "handoff_after_step": state_prefix_steps,
                "prefix_checkpoint": str(state_prefix_checkpoint),
                "prefix_checkpoint_code_revision": state_prefix_control["contract"][
                    "code_revision"
                ],
                "prefix_checkpoint_model_sha256": state_prefix_control["state_files"][
                    "model.safetensors"
                ]["sha256"],
                "prefix_checkpoint_optimizer_steps": state_prefix_steps,
                "prefix_checkpoint_plan_sha256": state_prefix_control["plan_sha256"],
            }
        ),
        "total_conflicts": sum(reason_counts.values()),
        "validated_checkpoint_plan_prefix_steps": validated_prefix_steps,
        "visual_artifacts": visual_artifacts,
        "warning": (
            "deterministic state-handoff replay from an empty posterior; prefix steps use the "
            "earlier checkpoint and later steps use the primary checkpoint without resetting "
            "the posterior; this is a coordinate-drift intervention, not the historical online "
            "training trajectory; evaluation rows do not update model weights"
            if state_prefix_model_path is not None
            else "final-weight deterministic replay from an empty posterior; not the historical "
            "online trajectory under the pre-update weight at every training step; evaluation "
            "rows do not update model weights"
        ),
    }
    _atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "schema",
                    "total_conflicts",
                    "conflict_reason_counts",
                    "prompt_causal_audits",
                    "visual_artifacts",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
