"""Model-output producers for the LingBot-native G2-G6 evidence gates.

This module is evaluation infrastructure.  It consumes detached model outputs
and benchmark results, computes fixed per-episode metrics, and never writes a
posterior or participates in policy inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from picf_next.artifact_io import publish_prepared_file_durable_exclusive
from picf_next.lingbot_native.empirical_statistics import (
    EMPIRICAL_COMPARISON_SPECS,
    EMPIRICAL_EVALUATION_PLAN_FIELDS,
    EMPIRICAL_EVALUATION_PLAN_SCHEMA,
    EMPIRICAL_OBSERVATIONS_SCHEMA,
    validate_empirical_metric_config,
)
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
)

EMPIRICAL_PRODUCER_BUNDLE_SCHEMA = "picf-next.lingbot-native-empirical-producer.v2"
EMPIRICAL_PRODUCER_REFERENCE_SCHEMA = "picf-next.lingbot-native-empirical-producer-reference.v1"

HOTA_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
REGISTERED_STATE_AGES = (1, 8, 32, 64, 128)
REIDENTIFICATION_IOU_THRESHOLD = 0.5
MAX_EPISODE_ARRAY_ELEMENTS = 50_000_000

_BUNDLE_FIELDS = {
    "schema",
    "gate",
    "subject",
    "protocol",
    "design",
    "check_evidence",
    "episodes",
}
_EPISODE_REFERENCE_FIELDS = {"seed", "task", "episode", "path", "sha256"}
_PRODUCER_REFERENCE_FIELDS = {"schema", "path", "sha256"}

_G2_ARRAYS = {
    "c_support",
    "m_support",
    "target_masks",
    "mask_valid",
    "c_existence",
    "m_existence",
    "target_existence",
    "existence_valid",
    "c_task_relevance",
    "m_task_relevance",
    "c_dense_task_grounding",
    "m_dense_task_grounding",
    "target_task_relevance",
    "task_valid",
    "track_valid",
    "capacity_censored",
    "inventory_exhaustive",
}
_G3_ARRAYS = {
    "c_support",
    "o_support",
    "target_masks",
    "mask_valid",
    "c_existence",
    "o_existence",
    "target_existence",
    "existence_valid",
    "track_valid",
    "capacity_censored",
    "inventory_exhaustive",
    "state_age",
}
_G4_ARRAYS = {
    "same_entity_similarity",
    "hard_negative_similarity",
    "all_available_quality",
    "missing_modality_quality",
    "corrupt_modality_quality",
    "whole_static_omission_trial",
}
_G5_ARRAYS = {
    "steps",
    "action_loss_a",
    "action_loss_h",
    "action_loss_m",
    "action_loss_o",
    "action_loss_c",
    "action_loss_c_row_intervened",
}
_G6_ARRAYS = {
    "sequence_length",
    "successful_prefix_a",
    "successful_prefix_o",
    "successful_prefix_c",
    "successful_prefix_c_row_intervened",
    "recovery_o",
    "recovery_c",
    "reset_session_isolation",
}
_GATE_ARRAYS = {
    "G2": _G2_ARRAYS,
    "G3": _G3_ARRAYS,
    "G4": _G4_ARRAYS,
    "G5": _G5_ARRAYS,
    "G6": _G6_ARRAYS,
}


def _exact_dict(value: object, *, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields differ from the frozen schema")
    return cast(dict[str, Any], value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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


def _real_hashed_file(path_value: object, digest_value: object, *, name: str) -> Path:
    path = Path(path_value) if isinstance(path_value, str) else None
    digest = _sha256(digest_value, name=f"{name} SHA-256")
    if (
        path is None
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != digest
    ):
        raise ValueError(f"{name} differs from its hash-bound real file")
    return path


def _probability(value: np.ndarray, *, name: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must be floating point")
    result = value.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError(f"{name} must contain finite probabilities")
    return result


def _finite_floating(value: np.ndarray, *, name: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must be floating point")
    result = value.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: np.ndarray, *, name: str) -> np.ndarray:
    if value.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean")
    return value


def _integer(value: np.ndarray, *, name: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.integer) or value.dtype == np.bool_:
        raise ValueError(f"{name} must be integer")
    return value.astype(np.int64, copy=False)


def _load_npz(path: Path, *, gate: str) -> dict[str, np.ndarray]:
    if path.suffix != ".npz":
        raise ValueError(f"{gate} episode artifact must use non-pickle NPZ")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _GATE_ARRAYS[gate]:
                raise ValueError(f"{gate} episode arrays differ from the frozen schema")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(gate):
            raise
        raise ValueError(f"{gate} episode artifact is not a safe NPZ") from error
    total = sum(array.size for array in arrays.values())
    if total <= 0 or total > MAX_EPISODE_ARRAY_ELEMENTS:
        raise ValueError(f"{gate} episode artifact has an invalid element budget")
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError(f"{gate} episode artifact contains an object array")
    return arrays


def write_empirical_episode_artifact(
    path: Path,
    *,
    gate: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Atomically publish one detached numeric episode artifact."""

    if gate not in _GATE_ARRAYS or set(arrays) != _GATE_ARRAYS[gate]:
        raise ValueError("empirical episode arrays differ from the registered gate")
    normalized: dict[str, np.ndarray] = {}
    total = 0
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise TypeError(f"empirical episode array {name} must be one non-object ndarray")
        normalized[name] = np.ascontiguousarray(value)
        total += value.size
    if total <= 0 or total > MAX_EPISODE_ARRAY_ELEMENTS:
        raise ValueError("empirical episode arrays exceed the fixed element budget")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        with temporary.open("wb") as stream:
            # NumPy's stub treats arbitrary keyword arrays as reserved options;
            # the exact gate schema above excludes every reserved name.
            cast(Any, np.savez_compressed)(stream, **normalized)
            stream.flush()
            os.fsync(stream.fileno())
        _load_npz(temporary, gate=gate)
        publish_prepared_file_durable_exclusive(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def write_empirical_producer_bundle(
    path: Path,
    *,
    bundle: Mapping[str, object],
) -> dict[str, object]:
    """Validate and atomically publish one producer bundle.

    Evaluators should call this only after all referenced episode files and
    protocol artifacts are immutable. The validation pass executes every
    registered metric before the final bundle name becomes visible.
    """

    value = dict(bundle)
    _exact_dict(value, name="empirical producer bundle", fields=_BUNDLE_FIELDS)
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        build_empirical_observations_from_producer(
            temporary,
            expected_sha256=_sha256_bytes(payload),
        )
        publish_prepared_file_durable_exclusive(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def _as_numpy(value: torch.Tensor, *, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be one tensor")
    return value.detach().cpu().numpy()


def _native_evaluation_support(
    predictions: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    *,
    batch_index: int,
) -> torch.Tensor:
    """Use the likelihood actually trained by the target ontology."""

    if targets.exclusive_ownership:
        return predictions.ownership[batch_index, ..., :-1]
    return predictions.support_logits[batch_index].sigmoid()


def g2_episode_arrays_from_native_outputs(
    *,
    candidate: NativeSequencePredictions,
    generic_memory: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    batch_index: int,
) -> dict[str, np.ndarray]:
    """Project detached native C/M sequence outputs into the fixed G2 artifact."""

    if not isinstance(candidate, NativeSequencePredictions) or not isinstance(
        generic_memory, NativeSequencePredictions
    ):
        raise TypeError("G2 native outputs must use NativeSequencePredictions")
    if not isinstance(targets, NativeSequenceTargets):
        raise TypeError("G2 native targets must use NativeSequenceTargets")
    batch = candidate.support_logits.shape[0]
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or not 0 <= batch_index < batch
        or generic_memory.support_logits.shape != candidate.support_logits.shape
        or targets.masks.shape[:2] != candidate.support_logits.shape[:2]
        or targets.masks.shape[-1] != candidate.support_logits.shape[2]
    ):
        raise ValueError("G2 native output axes differ")
    return {
        "c_support": _as_numpy(
            _native_evaluation_support(candidate, targets, batch_index=batch_index),
            name="G2 C support",
        ),
        "m_support": _as_numpy(
            _native_evaluation_support(generic_memory, targets, batch_index=batch_index),
            name="G2 M support",
        ),
        "target_masks": _as_numpy(targets.masks[batch_index], name="G2 target masks"),
        "mask_valid": _as_numpy(targets.mask_valid[batch_index], name="G2 mask validity"),
        "c_existence": _as_numpy(
            candidate.existence_logits[batch_index].sigmoid(), name="G2 C existence"
        ),
        "m_existence": _as_numpy(
            generic_memory.existence_logits[batch_index].sigmoid(), name="G2 M existence"
        ),
        "target_existence": _as_numpy(targets.existence[batch_index], name="G2 target existence"),
        "existence_valid": _as_numpy(
            targets.existence_valid[batch_index], name="G2 existence validity"
        ),
        "c_task_relevance": _as_numpy(
            candidate.task_relevance_logits[batch_index].sigmoid(), name="G2 C task"
        ),
        "m_task_relevance": _as_numpy(
            generic_memory.task_relevance_logits[batch_index].sigmoid(), name="G2 M task"
        ),
        "c_dense_task_grounding": _as_numpy(
            candidate.dense_task_grounding_logits[batch_index].sigmoid(),
            name="G2 C dense task",
        ),
        "m_dense_task_grounding": _as_numpy(
            generic_memory.dense_task_grounding_logits[batch_index].sigmoid(),
            name="G2 M dense task",
        ),
        "target_task_relevance": _as_numpy(
            targets.task_relevance[batch_index], name="G2 target task"
        ),
        "task_valid": _as_numpy(targets.task_valid[batch_index], name="G2 task validity"),
        "track_valid": _as_numpy(targets.track_valid[batch_index], name="G2 track validity"),
        "capacity_censored": _as_numpy(
            targets.capacity_censored[batch_index], name="G2 capacity censoring"
        ),
        "inventory_exhaustive": _as_numpy(
            targets.inventory_exhaustive[batch_index], name="G2 inventory exhaustiveness"
        ),
    }


def g3_episode_arrays_from_native_outputs(
    *,
    candidate: NativeSequencePredictions,
    reset_memory: NativeSequencePredictions,
    targets: NativeSequenceTargets,
    state_ages: torch.Tensor,
    batch_index: int,
) -> dict[str, np.ndarray]:
    """Project detached native C/O sequence outputs into the fixed G3 artifact."""

    if not isinstance(candidate, NativeSequencePredictions) or not isinstance(
        reset_memory, NativeSequencePredictions
    ):
        raise TypeError("G3 native outputs must use NativeSequencePredictions")
    if not isinstance(targets, NativeSequenceTargets):
        raise TypeError("G3 native targets must use NativeSequenceTargets")
    batch, time, tokens, _rows = candidate.support_logits.shape
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or not 0 <= batch_index < batch
        or reset_memory.support_logits.shape != candidate.support_logits.shape
        or targets.masks.shape[:2] != (batch, time)
        or targets.masks.shape[-1] != tokens
        or not isinstance(state_ages, torch.Tensor)
        or state_ages.shape != (batch, time)
        or state_ages.dtype != torch.long
    ):
        raise ValueError("G3 native output axes differ")
    return {
        "c_support": _as_numpy(
            _native_evaluation_support(candidate, targets, batch_index=batch_index),
            name="G3 C support",
        ),
        "o_support": _as_numpy(
            _native_evaluation_support(reset_memory, targets, batch_index=batch_index),
            name="G3 O support",
        ),
        "target_masks": _as_numpy(targets.masks[batch_index], name="G3 target masks"),
        "mask_valid": _as_numpy(targets.mask_valid[batch_index], name="G3 mask validity"),
        "c_existence": _as_numpy(
            candidate.existence_logits[batch_index].sigmoid(), name="G3 C existence"
        ),
        "o_existence": _as_numpy(
            reset_memory.existence_logits[batch_index].sigmoid(), name="G3 O existence"
        ),
        "target_existence": _as_numpy(targets.existence[batch_index], name="G3 target existence"),
        "existence_valid": _as_numpy(
            targets.existence_valid[batch_index], name="G3 existence validity"
        ),
        "track_valid": _as_numpy(targets.track_valid[batch_index], name="G3 track validity"),
        "capacity_censored": _as_numpy(
            targets.capacity_censored[batch_index], name="G3 capacity censoring"
        ),
        "inventory_exhaustive": _as_numpy(
            targets.inventory_exhaustive[batch_index], name="G3 inventory exhaustiveness"
        ),
        "state_age": _as_numpy(state_ages[batch_index], name="G3 state age"),
    }


def g4_episode_arrays_from_trials(
    *,
    same_entity_similarity: np.ndarray,
    hard_negative_similarity: np.ndarray,
    all_available_quality: np.ndarray,
    missing_modality_quality: np.ndarray,
    corrupt_modality_quality: np.ndarray,
    whole_static_omission_trial: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the exact G4 arrays from independently registered trial outputs."""

    arrays = {
        "same_entity_similarity": same_entity_similarity,
        "hard_negative_similarity": hard_negative_similarity,
        "all_available_quality": all_available_quality,
        "missing_modality_quality": missing_modality_quality,
        "corrupt_modality_quality": corrupt_modality_quality,
        "whole_static_omission_trial": whole_static_omission_trial,
    }
    if any(not isinstance(value, np.ndarray) for value in arrays.values()):
        raise TypeError("G4 trial outputs must be NumPy arrays")
    result = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    _g4_metrics(result)
    return result


def g5_episode_arrays_from_action_curves(
    *,
    steps: np.ndarray,
    action_loss_by_arm: Mapping[str, np.ndarray],
    row_intervened_action_loss: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build exact paired G5 curves without selecting minima or checkpoints."""

    if set(action_loss_by_arm) != {"A", "H", "M", "O", "C"}:
        raise ValueError("G5 action curves require exact A/H/M/O/C arms")
    values: dict[str, np.ndarray] = {
        "steps": steps,
        **{
            f"action_loss_{arm.lower()}": action_loss_by_arm[arm]
            for arm in ("A", "H", "M", "O", "C")
        },
        "action_loss_c_row_intervened": row_intervened_action_loss,
    }
    if any(not isinstance(value, np.ndarray) for value in values.values()):
        raise TypeError("G5 action curves must be NumPy arrays")
    result = {name: np.ascontiguousarray(value) for name, value in values.items()}
    _validated_g5_curves(result)
    return result


def g6_episode_arrays_from_calvin_rollouts(
    *,
    sequence_length: int,
    successful_prefix_by_arm: Mapping[str, int],
    row_intervened_successful_prefix: int,
    recovery_o: np.ndarray,
    recovery_c: np.ndarray,
    reset_session_isolation: bool,
) -> dict[str, np.ndarray]:
    """Build exact G6 arrays from paired official CALVIN sequence outcomes."""

    integers = (
        sequence_length,
        row_intervened_successful_prefix,
        *successful_prefix_by_arm.values(),
    )
    if (
        set(successful_prefix_by_arm) != {"A", "O", "C"}
        or any(isinstance(value, bool) or not isinstance(value, int) for value in integers)
        or not isinstance(recovery_o, np.ndarray)
        or not isinstance(recovery_c, np.ndarray)
        or not isinstance(reset_session_isolation, bool)
    ):
        raise TypeError("G6 CALVIN outcomes differ from the frozen value types")
    result = {
        "sequence_length": np.asarray([sequence_length], dtype=np.int64),
        "successful_prefix_a": np.asarray([successful_prefix_by_arm["A"]], dtype=np.int64),
        "successful_prefix_o": np.asarray([successful_prefix_by_arm["O"]], dtype=np.int64),
        "successful_prefix_c": np.asarray([successful_prefix_by_arm["C"]], dtype=np.int64),
        "successful_prefix_c_row_intervened": np.asarray(
            [row_intervened_successful_prefix], dtype=np.int64
        ),
        "recovery_o": np.ascontiguousarray(recovery_o),
        "recovery_c": np.ascontiguousarray(recovery_c),
        "reset_session_isolation": np.asarray([reset_session_isolation], dtype=np.bool_),
    }
    _g6_metrics(result)
    return result


def _soft_iou(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        raise ValueError("soft IoU has no valid support")
    predicted = prediction[valid]
    expected = target[valid]
    intersection = float(np.sum(predicted * expected))
    union = float(np.sum(predicted + expected - predicted * expected))
    return 1.0 if union <= np.finfo(np.float64).eps else intersection / union


def _prediction_assignment(
    *,
    support: np.ndarray,
    existence: np.ndarray,
    task: np.ndarray | None,
    target_masks: np.ndarray,
    mask_valid: np.ndarray,
    target_existence: np.ndarray,
    existence_valid: np.ndarray,
    target_task: np.ndarray | None,
    task_valid: np.ndarray | None,
    eligible: np.ndarray,
) -> dict[int, int]:
    _time, _tokens, rows = support.shape
    if len(eligible) > rows:
        raise ValueError("eligible target count exceeds row capacity")
    if len(eligible) == 0:
        return {}
    costs = np.zeros((rows, len(eligible)), dtype=np.float64)
    for row in range(rows):
        for column, track in enumerate(eligible.tolist()):
            components: list[float] = []
            valid_mask = mask_valid[:, track]
            if np.any(valid_mask) and np.any(target_masks[:, track][valid_mask] > 0):
                components.append(
                    1.0
                    - _soft_iou(
                        support[:, :, row],
                        target_masks[:, track],
                        valid_mask,
                    )
                )
            valid_existence = existence_valid[:, track]
            if np.any(valid_existence):
                components.append(
                    float(
                        np.mean(
                            np.square(
                                existence[valid_existence, row]
                                - target_existence[valid_existence, track]
                            )
                        )
                    )
                )
            if (
                task is not None
                and target_task is not None
                and task_valid is not None
                and bool(task_valid[track])
            ):
                components.append(float((task[row] - target_task[track]) ** 2))
            if not components:
                raise ValueError("eligible target has no evaluable component")
            costs[row, column] = float(np.mean(components))
    row_indices, target_columns = linear_sum_assignment(costs)
    return {
        int(eligible[column]): int(row)
        for row, column in zip(row_indices.tolist(), target_columns.tolist(), strict=True)
    }


def _matched_mask_score(
    *,
    support: np.ndarray,
    target_masks: np.ndarray,
    mask_valid: np.ndarray,
    assignment: Mapping[int, int],
) -> tuple[float, float]:
    predicted_scores: list[float] = []
    chance_scores: list[float] = []
    for track, row in assignment.items():
        valid = mask_valid[:, track]
        target = target_masks[:, track]
        if not np.any(valid) or not np.any(target[valid] > 0):
            continue
        predicted_scores.append(_soft_iou(support[:, :, row], target, valid))
        prevalence = float(np.mean(target[valid]))
        epsilon = float(np.finfo(np.float64).eps)
        chance_scores.append(prevalence / max(2.0 - prevalence, epsilon))
    if not predicted_scores:
        raise ValueError("episode contains no visible matched target")
    return float(np.mean(predicted_scores)), float(np.mean(chance_scores))


def _existence_brier(
    *,
    existence: np.ndarray,
    target_existence: np.ndarray,
    existence_valid: np.ndarray,
    inventory_exhaustive: np.ndarray,
    assignment: Mapping[int, int],
) -> float:
    values: list[np.ndarray] = []
    matched_rows = set(assignment.values())
    for track, row in assignment.items():
        valid = existence_valid[:, track]
        if np.any(valid):
            values.append(np.square(existence[valid, row] - target_existence[valid, track]))
    for row in range(existence.shape[1]):
        if row not in matched_rows and np.any(inventory_exhaustive):
            values.append(np.square(existence[inventory_exhaustive, row]))
    if not values:
        raise ValueError("episode contains no valid existence target")
    return float(np.mean(np.concatenate(values)))


def _task_utility(
    *,
    task: np.ndarray,
    target_task: np.ndarray,
    task_valid: np.ndarray,
    assignment: Mapping[int, int],
) -> float:
    errors = [
        float((task[row] - target_task[track]) ** 2)
        for track, row in assignment.items()
        if bool(task_valid[track])
    ]
    if not errors:
        raise ValueError("episode contains no valid task target")
    return 1.0 - float(np.mean(errors))


def _dense_task_utility(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Score exact visible task support without letting background dominate."""

    scores = [
        _soft_iou(prediction[index], target[index], valid[index])
        for index in range(target.shape[0])
        if np.any(valid[index]) and np.any(target[index][valid[index]] > 0)
    ]
    if not scores:
        raise ValueError("episode contains no visible exact dense task target")
    return float(np.mean(scores))


def _row_collapse_rate(
    *,
    support: np.ndarray,
    mask_valid: np.ndarray,
    assignment: Mapping[int, int],
) -> float:
    rows = sorted(set(assignment.values()))
    if len(rows) < 2:
        return 0.0
    valid_tokens = np.any(mask_valid, axis=1).reshape(-1)
    if not np.any(valid_tokens):
        raise ValueError("collapse metric has no valid token")
    matrix = np.stack([support[:, :, row].reshape(-1)[valid_tokens] for row in rows])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 1e-12)
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = np.square(singular)
    if float(energy.sum()) <= 1e-12:
        return 1.0
    probability = energy / energy.sum()
    probability = probability[probability > 0]
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    return float(np.clip(1.0 - effective_rank / len(rows), 0.0, 1.0))


def _g2_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float | None]]:
    c_support = _probability(arrays["c_support"], name="G2 C support")
    m_support = _probability(arrays["m_support"], name="G2 M support")
    target_masks = _probability(arrays["target_masks"], name="G2 target masks")
    mask_valid = _boolean(arrays["mask_valid"], name="G2 mask validity")
    c_existence = _probability(arrays["c_existence"], name="G2 C existence")
    m_existence = _probability(arrays["m_existence"], name="G2 M existence")
    target_existence = _probability(arrays["target_existence"], name="G2 target existence")
    existence_valid = _boolean(arrays["existence_valid"], name="G2 existence validity")
    c_task = _probability(arrays["c_task_relevance"], name="G2 C task")
    m_task = _probability(arrays["m_task_relevance"], name="G2 M task")
    c_dense_task = _probability(arrays["c_dense_task_grounding"], name="G2 C dense task")
    m_dense_task = _probability(arrays["m_dense_task_grounding"], name="G2 M dense task")
    target_task = _probability(arrays["target_task_relevance"], name="G2 target task")
    task_valid = _boolean(arrays["task_valid"], name="G2 task validity")
    track_valid = _boolean(arrays["track_valid"], name="G2 track validity")
    capacity_censored = _boolean(arrays["capacity_censored"], name="G2 capacity censoring")
    inventory_exhaustive = _boolean(
        arrays["inventory_exhaustive"], name="G2 inventory exhaustiveness"
    )
    if c_support.ndim != 3 or m_support.shape != c_support.shape:
        raise ValueError("G2 support arrays must share [time,tokens,rows]")
    time, tokens, rows = c_support.shape
    if target_masks.ndim != 3:
        raise ValueError("G2 target masks must have [time,tracks,tokens]")
    tracks = target_masks.shape[1]
    expected_shapes = {
        "target_masks": (target_masks, (time, tracks, tokens)),
        "mask_valid": (mask_valid, (time, tracks, tokens)),
        "c_existence": (c_existence, (time, rows)),
        "m_existence": (m_existence, (time, rows)),
        "target_existence": (target_existence, (time, tracks)),
        "existence_valid": (existence_valid, (time, tracks)),
        "c_task": (c_task, (rows,)),
        "m_task": (m_task, (rows,)),
        "c_dense_task": (c_dense_task, (time, tokens)),
        "m_dense_task": (m_dense_task, (time, tokens)),
        "target_task": (target_task, (tracks,)),
        "task_valid": (task_valid, (tracks,)),
        "track_valid": (track_valid, (tracks,)),
        "capacity_censored": (capacity_censored, (tracks,)),
        "inventory_exhaustive": (inventory_exhaustive, (time,)),
    }
    for name, (value, shape) in expected_shapes.items():
        if value.shape != shape:
            raise ValueError(f"G2 {name} shape differs from {shape}")
    if np.any(capacity_censored & ~track_valid):
        raise ValueError("G2 only valid tracks may be capacity censored")
    eligible = np.flatnonzero(track_valid & ~capacity_censored)
    c_assignment = _prediction_assignment(
        support=c_support,
        existence=c_existence,
        task=c_task,
        target_masks=target_masks,
        mask_valid=mask_valid,
        target_existence=target_existence,
        existence_valid=existence_valid,
        target_task=target_task,
        task_valid=task_valid,
        eligible=eligible,
    )
    m_assignment = _prediction_assignment(
        support=m_support,
        existence=m_existence,
        task=m_task,
        target_masks=target_masks,
        mask_valid=mask_valid,
        target_existence=target_existence,
        existence_valid=existence_valid,
        target_task=target_task,
        task_valid=task_valid,
        eligible=eligible,
    )
    metrics: dict[str, tuple[float, float | None]] = {
        "existence_calibration_error": (
            _existence_brier(
                existence=c_existence,
                target_existence=target_existence,
                existence_valid=existence_valid,
                inventory_exhaustive=inventory_exhaustive,
                assignment=c_assignment,
            ),
            None,
        ),
    }
    has_visible_target = any(
        np.any(mask_valid[:, track]) and np.any(target_masks[:, track][mask_valid[:, track]] > 0)
        for track in c_assignment
    )
    if has_visible_target:
        c_mask, chance = _matched_mask_score(
            support=c_support,
            target_masks=target_masks,
            mask_valid=mask_valid,
            assignment=c_assignment,
        )
        m_mask, _ = _matched_mask_score(
            support=m_support,
            target_masks=target_masks,
            mask_valid=mask_valid,
            assignment=m_assignment,
        )
        metrics["object_mask_vs_chance"] = (c_mask, chance)
        metrics["object_mask_C_vs_M"] = (c_mask, m_mask)
    if any(bool(task_valid[track]) for track in c_assignment):
        metrics["task_grounding_C_vs_M"] = (
            _task_utility(
                task=c_task,
                target_task=target_task,
                task_valid=task_valid,
                assignment=c_assignment,
            ),
            _task_utility(
                task=m_task,
                target_task=target_task,
                task_valid=task_valid,
                assignment=m_assignment,
            ),
        )
    exact_task = np.array_equal(task_valid, track_valid)
    # Dense grounding has no row-capacity bottleneck: a censored persistent
    # target remains a valid prompt-to-sensor target when its mask is observed.
    relevant_tracks = np.flatnonzero(track_valid & task_valid & (target_task > 0))
    if exact_task and relevant_tracks.size:
        weighted_masks = (
            target_masks[:, relevant_tracks] * target_task[relevant_tracks][None, :, None]
        )
        dense_target = 1.0 - np.prod(1.0 - weighted_masks, axis=1)
        dense_valid = np.all(mask_valid[:, relevant_tracks], axis=1)
        if any(
            np.any(dense_valid[index]) and np.any(dense_target[index][dense_valid[index]] > 0)
            for index in range(time)
        ):
            metrics["dense_task_grounding_C_vs_M"] = (
                _dense_task_utility(
                    prediction=c_dense_task,
                    target=dense_target,
                    valid=dense_valid,
                ),
                _dense_task_utility(
                    prediction=m_dense_task,
                    target=dense_target,
                    valid=dense_valid,
                ),
            )
    if len(c_assignment) >= 2:
        metrics["row_collapse_rate"] = (
            _row_collapse_rate(
                support=c_support,
                mask_valid=mask_valid,
                assignment=c_assignment,
            ),
            None,
        )
    return metrics


def _frame_similarities(
    support: np.ndarray,
    target_masks: np.ndarray,
    mask_valid: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    time, _tokens, rows = support.shape
    tracks = target_masks.shape[1]
    similarities: list[np.ndarray] = []
    visible_tracks: list[np.ndarray] = []
    for index in range(time):
        visible = np.asarray(
            [
                track
                for track in range(tracks)
                if np.any(mask_valid[index, track])
                and np.any(target_masks[index, track][mask_valid[index, track]] > 0)
            ],
            dtype=np.int64,
        )
        matrix = np.zeros((len(visible), rows), dtype=np.float64)
        for local_track, track in enumerate(visible.tolist()):
            valid = mask_valid[index, track]
            for row in range(rows):
                matrix[local_track, row] = _soft_iou(
                    support[index, :, row], target_masks[index, track], valid
                )
        visible_tracks.append(visible)
        similarities.append(matrix)
    return similarities, visible_tracks


def _hota_association_accuracy(
    *,
    support: np.ndarray,
    target_masks: np.ndarray,
    mask_valid: np.ndarray,
) -> float:
    """Adapt TrackEval's HOTA AssA computation to persistent rows and soft IoU."""

    time, _tokens, rows = support.shape
    tracks = target_masks.shape[1]
    similarities, visible_tracks = _frame_similarities(support, target_masks, mask_valid)
    potential = np.zeros((tracks, rows), dtype=np.float64)
    gt_count = np.zeros((tracks, 1), dtype=np.float64)
    row_count = np.full((1, rows), float(time), dtype=np.float64)
    for visible, similarity in zip(visible_tracks, similarities, strict=True):
        if not len(visible):
            continue
        denominator = similarity.sum(axis=0)[None, :] + similarity.sum(axis=1)[:, None] - similarity
        normalized = np.divide(
            similarity,
            denominator,
            out=np.zeros_like(similarity),
            where=denominator > np.finfo(np.float64).eps,
        )
        potential[visible[:, None], np.arange(rows)[None, :]] += normalized
        gt_count[visible] += 1
    global_alignment = np.divide(
        potential,
        gt_count + row_count - potential,
        out=np.zeros_like(potential),
        where=(gt_count + row_count - potential) > np.finfo(np.float64).eps,
    )
    matches = [np.zeros_like(potential) for _ in HOTA_THRESHOLDS]
    true_positives = np.zeros(len(HOTA_THRESHOLDS), dtype=np.float64)
    for visible, similarity in zip(visible_tracks, similarities, strict=True):
        if not len(visible):
            continue
        score = np.multiply(
            global_alignment[visible],
            similarity,
            dtype=np.float64,
        )
        match_rows, match_columns = linear_sum_assignment(np.negative(score))
        for threshold_index, threshold in enumerate(HOTA_THRESHOLDS):
            accepted = similarity[match_rows, match_columns] >= threshold - np.finfo(np.float64).eps
            selected_rows = match_rows[accepted]
            selected_columns = match_columns[accepted]
            true_positives[threshold_index] += len(selected_rows)
            matches[threshold_index][visible[selected_rows], selected_columns] += 1
    association = []
    for threshold_index, matched in enumerate(matches):
        denominator = gt_count + row_count - matched
        pair_accuracy = np.divide(
            matched,
            denominator,
            out=np.zeros_like(matched),
            where=denominator > 0,
        )
        tp = true_positives[threshold_index]
        association.append(0.0 if tp <= 0 else float(np.sum(matched * pair_accuracy) / tp))
    return float(np.mean(association))


def _occlusion_metrics(
    *,
    support: np.ndarray,
    target_masks: np.ndarray,
    mask_valid: np.ndarray,
    target_existence: np.ndarray,
    existence_valid: np.ndarray,
    assignment: Mapping[int, int],
) -> tuple[float, float]:
    similarities, visible_tracks = _frame_similarities(support, target_masks, mask_valid)
    visible = np.zeros(target_existence.shape, dtype=np.bool_)
    for time_index, tracks in enumerate(visible_tracks):
        visible[time_index, tracks] = True
    recovery: list[float] = []
    reidentification: list[float] = []
    time = target_existence.shape[0]
    for track, assigned_row in assignment.items():
        index = 1
        while index < time - 1:
            if not visible[index - 1, track] or visible[index, track]:
                index += 1
                continue
            start = index
            while (
                index < time
                and not visible[index, track]
                and bool(existence_valid[index, track])
                and target_existence[index, track] >= 0.5
            ):
                index += 1
            if index == start or index >= time or not visible[index, track]:
                continue
            before_local = int(np.flatnonzero(visible_tracks[start - 1] == track)[0])
            after_local = int(np.flatnonzero(visible_tracks[index] == track)[0])
            before_scores = similarities[start - 1][before_local]
            after_scores = similarities[index][after_local]
            recovery.append(float(after_scores[assigned_row]))
            reidentification.append(
                float(
                    int(np.argmax(before_scores)) == int(np.argmax(after_scores))
                    and after_scores[int(np.argmax(after_scores))] >= REIDENTIFICATION_IOU_THRESHOLD
                )
            )
    if not recovery:
        raise ValueError("G3 episode contains no valid visible-occluded-visible interval")
    return float(np.mean(recovery)), float(np.mean(reidentification))


def _long_age_brier(
    *,
    existence: np.ndarray,
    target_existence: np.ndarray,
    existence_valid: np.ndarray,
    inventory_exhaustive: np.ndarray,
    state_age: np.ndarray,
    assignment: Mapping[int, int],
) -> float:
    registered = np.isin(state_age, np.asarray(REGISTERED_STATE_AGES, dtype=np.int64))
    if not np.any(registered):
        raise ValueError("G3 episode covers no registered state age")
    return _existence_brier(
        existence=existence[registered],
        target_existence=target_existence[registered],
        existence_valid=existence_valid[registered],
        inventory_exhaustive=inventory_exhaustive[registered],
        assignment=assignment,
    )


def _g3_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float | None]]:
    c_support = _probability(arrays["c_support"], name="G3 C support")
    o_support = _probability(arrays["o_support"], name="G3 O support")
    target_masks = _probability(arrays["target_masks"], name="G3 target masks")
    mask_valid = _boolean(arrays["mask_valid"], name="G3 mask validity")
    c_existence = _probability(arrays["c_existence"], name="G3 C existence")
    o_existence = _probability(arrays["o_existence"], name="G3 O existence")
    target_existence = _probability(arrays["target_existence"], name="G3 target existence")
    existence_valid = _boolean(arrays["existence_valid"], name="G3 existence validity")
    track_valid = _boolean(arrays["track_valid"], name="G3 track validity")
    capacity_censored = _boolean(arrays["capacity_censored"], name="G3 capacity censoring")
    inventory_exhaustive = _boolean(
        arrays["inventory_exhaustive"], name="G3 inventory exhaustiveness"
    )
    state_age = _integer(arrays["state_age"], name="G3 state age")
    if c_support.ndim != 3 or o_support.shape != c_support.shape:
        raise ValueError("G3 support arrays must share [time,tokens,rows]")
    time, tokens, rows = c_support.shape
    if target_masks.ndim != 3:
        raise ValueError("G3 target masks must have [time,tracks,tokens]")
    tracks = target_masks.shape[1]
    expected_shapes = {
        "target_masks": (target_masks, (time, tracks, tokens)),
        "mask_valid": (mask_valid, (time, tracks, tokens)),
        "c_existence": (c_existence, (time, rows)),
        "o_existence": (o_existence, (time, rows)),
        "target_existence": (target_existence, (time, tracks)),
        "existence_valid": (existence_valid, (time, tracks)),
        "track_valid": (track_valid, (tracks,)),
        "capacity_censored": (capacity_censored, (tracks,)),
        "inventory_exhaustive": (inventory_exhaustive, (time,)),
        "state_age": (state_age, (time,)),
    }
    for name, (value, shape) in expected_shapes.items():
        if value.shape != shape:
            raise ValueError(f"G3 {name} shape differs from {shape}")
    if np.any(state_age < 1) or np.any(capacity_censored & ~track_valid):
        raise ValueError("G3 state age or capacity censoring is invalid")
    eligible = np.flatnonzero(track_valid & ~capacity_censored)
    c_assignment = _prediction_assignment(
        support=c_support,
        existence=c_existence,
        task=None,
        target_masks=target_masks,
        mask_valid=mask_valid,
        target_existence=target_existence,
        existence_valid=existence_valid,
        target_task=None,
        task_valid=None,
        eligible=eligible,
    )
    o_assignment = _prediction_assignment(
        support=o_support,
        existence=o_existence,
        task=None,
        target_masks=target_masks,
        mask_valid=mask_valid,
        target_existence=target_existence,
        existence_valid=existence_valid,
        target_task=None,
        task_valid=None,
        eligible=eligible,
    )
    c_recovery, c_reid = _occlusion_metrics(
        support=c_support,
        target_masks=target_masks,
        mask_valid=mask_valid,
        target_existence=target_existence,
        existence_valid=existence_valid,
        assignment=c_assignment,
    )
    o_recovery, o_reid = _occlusion_metrics(
        support=o_support,
        target_masks=target_masks,
        mask_valid=mask_valid,
        target_existence=target_existence,
        existence_valid=existence_valid,
        assignment=o_assignment,
    )
    return {
        "identity_stability_C_vs_O": (
            _hota_association_accuracy(
                support=c_support,
                target_masks=target_masks,
                mask_valid=mask_valid,
            ),
            _hota_association_accuracy(
                support=o_support,
                target_masks=target_masks,
                mask_valid=mask_valid,
            ),
        ),
        "occlusion_recovery_C_vs_O": (c_recovery, o_recovery),
        "long_age_calibration_error": (
            _long_age_brier(
                existence=c_existence,
                target_existence=target_existence,
                existence_valid=existence_valid,
                inventory_exhaustive=inventory_exhaustive,
                state_age=state_age,
                assignment=c_assignment,
            ),
            None,
        ),
        "reidentification_C_vs_O": (c_reid, o_reid),
    }


def _paired_vectors(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    name: str,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_value = _finite_floating(candidate, name=f"{name} candidate")
    reference_value = _finite_floating(reference, name=f"{name} reference")
    if (
        candidate_value.ndim != 1
        or reference_value.shape != candidate_value.shape
        or candidate_value.size == 0
        or np.any(candidate_value < lower)
        or np.any(candidate_value > upper)
        or np.any(reference_value < lower)
        or np.any(reference_value > upper)
    ):
        raise ValueError(f"{name} must be one nonempty paired vector in range")
    return candidate_value, reference_value


def _g4_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float | None]]:
    same, negative = _paired_vectors(
        arrays["same_entity_similarity"],
        arrays["hard_negative_similarity"],
        name="G4 binding",
        lower=-1.0,
        upper=1.0,
    )
    missing, available = _paired_vectors(
        arrays["missing_modality_quality"],
        arrays["all_available_quality"],
        name="G4 missing modality",
        lower=0.0,
        upper=1.0,
    )
    corrupt, available_again = _paired_vectors(
        arrays["corrupt_modality_quality"],
        arrays["all_available_quality"],
        name="G4 corrupt modality",
        lower=0.0,
        upper=1.0,
    )
    omission = _boolean(arrays["whole_static_omission_trial"], name="G4 whole-static omission")
    if omission.shape != same.shape or not np.any(omission):
        raise ValueError("G4 episode must include a registered whole-static omission trial")
    if not np.array_equal(available, available_again):
        raise ValueError("G4 controls do not share the all-available reference")
    return {
        "binding_vs_within_scene_hard_negative": (
            float(np.mean(same)),
            float(np.mean(negative)),
        ),
        "missing_modality_noninferiority": (
            float(np.mean(missing)),
            float(np.mean(available)),
        ),
        "corrupt_modality_noninferiority": (
            float(np.mean(corrupt)),
            float(np.mean(available)),
        ),
    }


def _normalized_curve_auc(steps: np.ndarray, loss: np.ndarray) -> float:
    widths = np.diff(steps.astype(np.float64))
    area = float(np.sum(widths * (loss[:-1] + loss[1:]) * 0.5))
    return area / float(steps[-1] - steps[0])


def _convergence_time(steps: np.ndarray, loss: np.ndarray, threshold: float) -> float:
    reached = np.flatnonzero(loss <= threshold)
    if len(reached):
        return float(steps[int(reached[0])])
    return float(steps[-1] + (steps[-1] - steps[-2]))


def _validated_g5_curves(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    steps = _integer(arrays["steps"], name="G5 steps")
    if steps.ndim != 1 or len(steps) < 2 or np.any(np.diff(steps) <= 0) or steps[0] < 0:
        raise ValueError("G5 steps must be a strictly increasing curve axis")
    curves: dict[str, np.ndarray] = {}
    for arm in ("a", "h", "m", "o", "c", "c_row_intervened"):
        curve = _finite_floating(arrays[f"action_loss_{arm}"], name=f"G5 {arm} curve")
        if curve.shape != steps.shape or np.any(curve < 0):
            raise ValueError("G5 action curves must be complete, paired and non-negative")
        curves[arm] = curve
    return steps, curves


def _g5_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    metric_config: Mapping[str, object],
) -> dict[str, tuple[float, float | None]]:
    config = _exact_dict(
        metric_config,
        name="G5 metric config",
        fields={"action_loss_threshold"},
    )
    threshold_value = config["action_loss_threshold"]
    if (
        isinstance(threshold_value, bool)
        or not isinstance(threshold_value, (int, float))
        or not math.isfinite(threshold_value)
        or threshold_value <= 0
    ):
        raise ValueError("G5 action loss threshold must be finite and positive")
    threshold = float(threshold_value)
    steps, curves = _validated_g5_curves(arrays)
    utility = {arm: -_normalized_curve_auc(steps, curve) for arm, curve in curves.items()}
    return {
        "action_C_vs_A_noninferiority": (utility["c"], utility["a"]),
        "action_C_vs_H": (utility["c"], utility["h"]),
        "action_C_vs_M": (utility["c"], utility["m"]),
        "action_C_vs_O": (utility["c"], utility["o"]),
        "row_intervention_effect": (utility["c"], utility["c_row_intervened"]),
        "action_convergence_regression": (
            _convergence_time(steps, curves["c"], threshold),
            _convergence_time(steps, curves["a"], threshold),
        ),
    }


def _scalar_integer(array: np.ndarray, *, name: str) -> int:
    value = _integer(array, name=name)
    if value.shape != (1,):
        raise ValueError(f"{name} must contain exactly one integer")
    return int(value[0])


def _g6_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, tuple[float, float | None]]:
    length = _scalar_integer(arrays["sequence_length"], name="G6 sequence length")
    if length <= 0:
        raise ValueError("G6 sequence length must be positive")
    prefixes = {
        arm: _scalar_integer(arrays[f"successful_prefix_{arm}"], name=f"G6 {arm} prefix")
        for arm in ("a", "o", "c", "c_row_intervened")
    }
    if any(value < 0 or value > length for value in prefixes.values()):
        raise ValueError("G6 successful prefixes must lie inside the CALVIN sequence")
    recovery_o = _boolean(arrays["recovery_o"], name="G6 O recovery")
    recovery_c = _boolean(arrays["recovery_c"], name="G6 C recovery")
    reset = _boolean(arrays["reset_session_isolation"], name="G6 reset/session isolation")
    if recovery_o.ndim != 1 or recovery_o.size == 0 or recovery_c.shape != recovery_o.shape:
        raise ValueError("G6 recovery trials must be nonempty and paired")
    if reset.shape != (1,) or not bool(reset[0]):
        raise ValueError("G6 reset/session isolation did not pass")
    utility = {arm: value / length for arm, value in prefixes.items()}
    return {
        "calvin_success_C_vs_A": (utility["c"], utility["a"]),
        "calvin_success_C_vs_O": (utility["c"], utility["o"]),
        "calvin_recovery_C_vs_O": (
            float(np.mean(recovery_c)),
            float(np.mean(recovery_o)),
        ),
        "row_intervention_closed_loop_effect": (
            utility["c"],
            utility["c_row_intervened"],
        ),
    }


def _episode_metrics(
    gate: str,
    arrays: Mapping[str, np.ndarray],
    *,
    metric_config: Mapping[str, object],
) -> dict[str, tuple[float, float | None]]:
    if gate == "G2":
        metrics = _g2_metrics(arrays)
    elif gate == "G3":
        metrics = _g3_metrics(arrays)
    elif gate == "G4":
        metrics = _g4_metrics(arrays)
    elif gate == "G5":
        metrics = _g5_metrics(arrays, metric_config=metric_config)
    elif gate == "G6":
        metrics = _g6_metrics(arrays)
    else:
        raise ValueError("empirical producer gate is unsupported")
    if not metrics or not set(metrics) <= set(EMPIRICAL_COMPARISON_SPECS[gate]):
        raise RuntimeError("empirical producer emitted an invalid comparison set")
    if gate != "G2" and set(metrics) != set(EMPIRICAL_COMPARISON_SPECS[gate]):
        raise RuntimeError("empirical producer omitted a registered comparison")
    return metrics


def build_empirical_observations_from_producer(
    producer_path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute one complete observation file from raw episode artifacts."""

    path = producer_path.resolve()
    if producer_path.is_symlink() or not producer_path.is_file():
        raise ValueError("empirical producer bundle must be one real file")
    payload = producer_path.read_bytes()
    digest = _sha256_bytes(payload)
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, name="empirical producer bundle"
    ):
        raise ValueError("empirical producer bundle differs from its expected digest")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("empirical producer bundle is not valid JSON") from error
    bundle = _exact_dict(decoded, name="empirical producer bundle", fields=_BUNDLE_FIELDS)
    gate = bundle["gate"]
    if not isinstance(gate, str) or gate not in _GATE_ARRAYS:
        raise ValueError("empirical producer gate is unsupported")
    if bundle["schema"] != EMPIRICAL_PRODUCER_BUNDLE_SCHEMA:
        raise ValueError("empirical producer bundle schema changed")
    protocol = bundle["protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("empirical producer protocol is malformed")
    plan_path_value = protocol.get("evaluation_plan_path")
    plan_digest_value = protocol.get("evaluation_plan_sha256")
    plan_path = _real_hashed_file(
        plan_path_value,
        plan_digest_value,
        name="empirical producer evaluation plan",
    )
    try:
        plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("empirical producer evaluation plan is not valid JSON") from error
    plan = _exact_dict(
        plan_value,
        name="empirical evaluation plan",
        fields=EMPIRICAL_EVALUATION_PLAN_FIELDS,
    )
    if (
        plan["schema"] != EMPIRICAL_EVALUATION_PLAN_SCHEMA
        or plan["gate"] != gate
        or plan["design"] != bundle["design"]
    ):
        raise ValueError("empirical producer differs from its evaluation plan")
    metric_config = validate_empirical_metric_config(plan["metric_config"], gate=gate)
    episodes = bundle["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("empirical producer contains no episode")
    records: list[dict[str, object]] = []
    units: set[tuple[int, str, str]] = set()
    seed_layout: dict[int, dict[str, int]] = {}
    comparison_layout: dict[str, dict[int, dict[str, int]]] = {
        name: {} for name in EMPIRICAL_COMPARISON_SPECS[gate]
    }
    artifact_identities: set[tuple[int, int]] = set()
    for raw_episode in episodes:
        episode = _exact_dict(
            raw_episode,
            name=f"{gate} producer episode",
            fields=_EPISODE_REFERENCE_FIELDS,
        )
        seed = episode["seed"]
        task = episode["task"]
        episode_name = episode["episode"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or not isinstance(task, str)
            or not task.strip()
            or not isinstance(episode_name, str)
            or not episode_name.strip()
        ):
            raise ValueError(f"{gate} producer episode identity is malformed")
        unit = (seed, task, episode_name)
        if unit in units:
            raise ValueError(f"{gate} producer duplicates one episode unit")
        units.add(unit)
        seed_layout.setdefault(seed, {}).setdefault(task, 0)
        seed_layout[seed][task] += 1
        episode_path = _real_hashed_file(
            episode["path"], episode["sha256"], name=f"{gate} episode artifact"
        )
        stat = episode_path.stat()
        artifact_identity = (stat.st_dev, stat.st_ino)
        if artifact_identity in artifact_identities:
            raise ValueError(f"{gate} producer reuses one episode artifact")
        artifact_identities.add(artifact_identity)
        arrays = _load_npz(episode_path, gate=gate)
        metrics = _episode_metrics(gate, arrays, metric_config=metric_config)
        for comparison, (candidate, reference) in metrics.items():
            comparison_layout[comparison].setdefault(seed, {}).setdefault(task, 0)
            comparison_layout[comparison][seed][task] += 1
            _rule, mode, candidate_label, reference_label = EMPIRICAL_COMPARISON_SPECS[gate][
                comparison
            ]
            if mode == "value" and reference is not None:
                raise RuntimeError("absolute producer metric unexpectedly has a reference")
            if mode == "difference" and reference is None:
                raise RuntimeError("paired producer metric unexpectedly lacks a reference")
            records.append(
                {
                    "comparison": comparison,
                    "seed": seed,
                    "task": task,
                    "episode": episode_name,
                    "candidate_label": candidate_label,
                    "candidate": float(candidate),
                    "reference_label": reference_label,
                    "reference": None if reference is None else float(reference),
                }
            )
    seed_count = (
        bundle["design"].get("paired_seed_count") if isinstance(bundle["design"], dict) else None
    )
    if (
        isinstance(seed_count, bool)
        or not isinstance(seed_count, int)
        or len(seed_layout) != seed_count
    ):
        raise ValueError(f"{gate} producer does not cover every paired seed")
    layouts = [seed_layout[seed] for seed in sorted(seed_layout)]
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError(f"{gate} producer uses an unbalanced task/episode layout")
    for comparison, seeds in comparison_layout.items():
        if len(seeds) != seed_count:
            raise ValueError(
                f"{gate} producer comparison {comparison} does not cover every paired seed"
            )
        comparison_layouts = [seeds[seed] for seed in sorted(seeds)]
        if any(layout != comparison_layouts[0] for layout in comparison_layouts[1:]):
            raise ValueError(f"{gate} producer comparison {comparison} uses an unbalanced layout")
    records.sort(
        key=lambda record: (
            cast(str, record["comparison"]),
            cast(int, record["seed"]),
            cast(str, record["task"]),
            cast(str, record["episode"]),
        )
    )
    return {
        "schema": EMPIRICAL_OBSERVATIONS_SCHEMA,
        "gate": gate,
        "subject": bundle["subject"],
        "protocol": bundle["protocol"],
        "design": bundle["design"],
        "check_evidence": bundle["check_evidence"],
        "producer": {
            "schema": EMPIRICAL_PRODUCER_REFERENCE_SCHEMA,
            "path": str(path),
            "sha256": digest,
        },
        "records": records,
    }


def validate_producer_reference(value: object) -> tuple[Path, str]:
    """Validate and resolve the producer reference embedded in observations."""

    reference = _exact_dict(
        value,
        name="empirical producer reference",
        fields=_PRODUCER_REFERENCE_FIELDS,
    )
    if reference["schema"] != EMPIRICAL_PRODUCER_REFERENCE_SCHEMA:
        raise ValueError("empirical producer reference schema changed")
    path = _real_hashed_file(
        reference["path"], reference["sha256"], name="empirical producer reference"
    )
    return path, cast(str, reference["sha256"])
