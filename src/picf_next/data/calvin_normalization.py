"""Reproducible CALVIN state/action normalization artifacts for VLA hosts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CALVIN_ACTION_AXES,
    CALVIN_STATE_AXES,
    CalvinDatasetIndex,
    CalvinStatefulTransitionDataset,
)

CALVIN_NORMALIZATION_LEGACY_SCHEMA = "picf-next.calvin-molmoact2-normalization.v1"
CALVIN_NORMALIZATION_SCHEMA = "picf-next.calvin-training-normalization.v2"
CALVIN_NORMALIZATION_SCHEMAS = frozenset(
    {CALVIN_NORMALIZATION_LEGACY_SCHEMA, CALVIN_NORMALIZATION_SCHEMA}
)
LINGBOT_CALVIN_NORMALIZATION_SCHEMA = "picf-next.calvin-lingbot-normalization.v2"
_FEATURE_FIELDS = {
    "axes",
    "count",
    "max",
    "mean",
    "min",
    "normalize_mask",
    "q01",
    "q99",
    "std",
}
_LEGACY_TOP_LEVEL_FIELDS = {
    "action",
    "artifact_sha256",
    "dataset_id",
    "dataset_revision",
    "ordered_sample_keys_sha256",
    "quantile_method",
    "sample_count",
    "schema",
    "source_values_sha256",
    "state",
}
_TOP_LEVEL_FIELDS = _LEGACY_TOP_LEVEL_FIELDS | {
    "dataset_tree_sha256",
    "unique_source_frame_count",
}


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _ordered_strings_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _feature_payload(values: np.ndarray, axes: tuple[str, ...]) -> dict[str, object]:
    if values.ndim != 2 or values.shape[1] != len(axes) or values.shape[0] <= 0:
        raise ValueError("normalization values must be one nonempty sample-by-axis matrix")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("normalization values must be finite float32")
    return {
        "axes": list(axes),
        "count": int(values.shape[0]),
        "max": np.max(values, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "min": np.min(values, axis=0).astype(np.float64).tolist(),
        "normalize_mask": ["gripper" not in axis.lower() for axis in axes],
        "q01": np.quantile(values, 0.01, axis=0, method="linear").tolist(),
        "q99": np.quantile(values, 0.99, axis=0, method="linear").tolist(),
        "std": np.std(values, axis=0, dtype=np.float64, ddof=0).tolist(),
    }


def _state_action_windows(
    index: CalvinDatasetIndex,
    global_indices: np.ndarray,
    *,
    maximum_workers: int,
) -> Iterator[tuple[tuple[int, np.ndarray, np.ndarray], ...]]:
    window_size = max(1024, maximum_workers * 64)

    def load(global_index: np.int64) -> tuple[np.ndarray, np.ndarray]:
        return index.state_and_action(int(global_index))

    def positioned_window(
        start: int,
        loaded: Iterator[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
        return tuple(
            (position, state, action)
            for position, (state, action) in enumerate(loaded, start=start)
        )

    if maximum_workers == 1:
        for start in range(0, len(global_indices), window_size):
            yield positioned_window(
                start,
                map(load, global_indices[start : start + window_size]),
            )
        return

    with ThreadPoolExecutor(
        max_workers=maximum_workers,
        thread_name_prefix="calvin-normalization",
    ) as executor:
        for start in range(0, len(global_indices), window_size):
            window = global_indices[start : start + window_size]
            yield positioned_window(start, executor.map(load, window))


def build_calvin_normalization_artifact(
    index: CalvinDatasetIndex,
    *,
    maximum_workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Compute exact training-manifest statistics without decoding image arrays."""

    if not isinstance(index, CalvinDatasetIndex):
        raise TypeError("CALVIN normalization requires a CalvinDatasetIndex")
    if index.dataset_manifest is None:
        raise ContractError("CALVIN normalization requires a content-addressed dataset manifest")
    if (
        not isinstance(maximum_workers, int)
        or isinstance(maximum_workers, bool)
        or maximum_workers <= 0
    ):
        raise TypeError("maximum_workers must be a positive integer")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    sample_count = len(dataset)
    if sample_count <= 0:
        raise ValueError("CALVIN normalization requires at least one training transition")
    states = np.empty((sample_count, len(CALVIN_STATE_AXES)), dtype=np.float32)
    actions = np.empty((sample_count, len(CALVIN_ACTION_AXES)), dtype=np.float32)
    global_indices = np.fromiter(
        (locator.global_index for locator in dataset.locators),
        dtype=np.int64,
        count=sample_count,
    )
    order = np.argsort(global_indices, kind="stable")
    sorted_indices = global_indices[order]
    starts = np.flatnonzero(
        np.concatenate(
            (
                np.ones(1, dtype=np.bool_),
                sorted_indices[1:] != sorted_indices[:-1],
            )
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([sample_count], dtype=np.int64)))
    unique_indices = sorted_indices[starts]
    unique_source_frame_count = int(len(unique_indices))
    completed = 0
    for window in _state_action_windows(
        index,
        unique_indices,
        maximum_workers=maximum_workers,
    ):
        for position, state, action in window:
            rows = order[int(starts[position]) : int(ends[position])]
            states[rows] = state
            actions[rows] = action
        completed += len(window)
        if progress_callback is not None:
            progress_callback(completed, unique_source_frame_count)

    values_digest = hashlib.sha256()
    for row, sample_key in enumerate(dataset.sample_keys):
        encoded_key = sample_key.encode("utf-8")
        values_digest.update(len(encoded_key).to_bytes(8, "big"))
        values_digest.update(encoded_key)
        values_digest.update(states[row].astype("<f4", copy=False).tobytes())
        values_digest.update(actions[row].astype("<f4", copy=False).tobytes())

    payload: dict[str, object] = {
        "schema": CALVIN_NORMALIZATION_SCHEMA,
        "dataset_id": index.dataset_id,
        "dataset_revision": index.dataset_revision,
        "dataset_tree_sha256": index.dataset_manifest.tree_sha256,
        "sample_count": sample_count,
        "unique_source_frame_count": unique_source_frame_count,
        "ordered_sample_keys_sha256": _ordered_strings_sha256(dataset.sample_keys),
        "source_values_sha256": values_digest.hexdigest(),
        "quantile_method": "numpy.quantile.linear.v1",
        "state": _feature_payload(states, CALVIN_STATE_AXES),
        "action": _feature_payload(actions, CALVIN_ACTION_AXES),
    }
    payload["artifact_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    validate_calvin_normalization_artifact(payload)
    return payload


def content_identified_calvin_normalization_artifact(
    payload: dict[str, object],
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_tree_sha256: str,
) -> dict[str, object]:
    """Rebind complete statistics when the dataset bytes and sample order are unchanged."""

    validate_calvin_normalization_artifact(payload)
    if payload["schema"] != CALVIN_NORMALIZATION_SCHEMA:
        raise ContractError("CALVIN normalization identity migration requires schema v2")
    migrated = copy.deepcopy(payload)
    migrated["dataset_id"] = dataset_id
    migrated["dataset_revision"] = dataset_revision
    migrated["dataset_tree_sha256"] = dataset_tree_sha256
    migrated.pop("artifact_sha256")
    migrated["artifact_sha256"] = hashlib.sha256(_canonical_bytes(migrated)).hexdigest()
    validate_calvin_normalization_artifact(migrated)

    provenance = {
        "artifact_sha256",
        "dataset_id",
        "dataset_revision",
        "dataset_tree_sha256",
    }
    source_statistics = {key: value for key, value in payload.items() if key not in provenance}
    target_statistics = {key: value for key, value in migrated.items() if key not in provenance}
    if source_statistics != target_statistics:
        raise RuntimeError("CALVIN normalization identity migration changed statistics")
    return migrated


def _validated_feature(payload: object, *, name: str, axes: tuple[str, ...]) -> None:
    if not isinstance(payload, dict) or set(payload) != _FEATURE_FIELDS:
        raise ContractError(f"CALVIN normalization {name} fields differ from schema")
    if payload["axes"] != list(axes):
        raise ContractError(f"CALVIN normalization {name} axes differ from the adapter")
    count = payload["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ContractError(f"CALVIN normalization {name} count must be positive")
    mask = payload["normalize_mask"]
    expected_mask = ["gripper" not in axis.lower() for axis in axes]
    if mask != expected_mask:
        raise ContractError(f"CALVIN normalization {name} mask differs from gripper semantics")
    vectors: dict[str, np.ndarray] = {}
    for field in ("min", "max", "mean", "std", "q01", "q99"):
        vector = np.asarray(payload[field])
        if vector.shape != (len(axes),) or not np.issubdtype(vector.dtype, np.number):
            raise ContractError(f"CALVIN normalization {name}.{field} has the wrong shape")
        if not np.isfinite(vector).all():
            raise ContractError(f"CALVIN normalization {name}.{field} is non-finite")
        vectors[field] = vector.astype(np.float64)
    if (
        np.any(vectors["min"] > vectors["q01"])
        or np.any(vectors["q01"] > vectors["q99"])
        or np.any(vectors["q99"] > vectors["max"])
    ):
        raise ContractError(f"CALVIN normalization {name} quantiles are not ordered")
    if np.any(vectors["std"] < 0.0):
        raise ContractError(f"CALVIN normalization {name} standard deviation is negative")


def validate_calvin_normalization_artifact(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ContractError("CALVIN normalization fields differ from schema")
    schema = payload.get("schema")
    expected_fields = (
        _TOP_LEVEL_FIELDS if schema == CALVIN_NORMALIZATION_SCHEMA else _LEGACY_TOP_LEVEL_FIELDS
    )
    if set(payload) != expected_fields:
        raise ContractError("CALVIN normalization fields differ from schema")
    if schema not in CALVIN_NORMALIZATION_SCHEMAS:
        raise ContractError("unsupported CALVIN normalization schema")
    for field in ("dataset_id", "dataset_revision"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ContractError(f"CALVIN normalization {field} must be nonempty")
    sample_count = payload["sample_count"]
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise ContractError("CALVIN normalization sample_count must be positive")
    if schema == CALVIN_NORMALIZATION_SCHEMA:
        unique_source_frame_count = payload["unique_source_frame_count"]
        if (
            not isinstance(unique_source_frame_count, int)
            or isinstance(unique_source_frame_count, bool)
            or not 0 < unique_source_frame_count <= sample_count
        ):
            raise ContractError(
                "CALVIN normalization unique_source_frame_count must lie within sample_count"
            )
    for field in (
        "artifact_sha256",
        "ordered_sample_keys_sha256",
        "source_values_sha256",
        *(("dataset_tree_sha256",) if schema == CALVIN_NORMALIZATION_SCHEMA else ()),
    ):
        value = payload[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError(f"CALVIN normalization {field} must be a SHA-256 digest")
    if payload["quantile_method"] != "numpy.quantile.linear.v1":
        raise ContractError("CALVIN normalization quantile method changed")
    _validated_feature(payload["state"], name="state", axes=CALVIN_STATE_AXES)
    _validated_feature(payload["action"], name="action", axes=CALVIN_ACTION_AXES)
    if payload["state"]["count"] != sample_count or payload["action"]["count"] != sample_count:
        raise ContractError("CALVIN normalization feature counts differ from sample_count")
    canonical = dict(payload)
    recorded = canonical.pop("artifact_sha256")
    actual = hashlib.sha256(_canonical_bytes(canonical)).hexdigest()
    if actual != recorded:
        raise ContractError("CALVIN normalization artifact SHA-256 changed")


def load_calvin_normalization_artifact(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text())
    validate_calvin_normalization_artifact(payload)
    return payload


def write_calvin_normalization_artifact(
    payload: dict[str, object],
    destination: str | Path,
) -> Path:
    validate_calvin_normalization_artifact(payload)
    path = Path(destination)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    write_bytes_durable_exclusive(path, encoded)
    return path


def official_molmoact2_dataset_stats(
    payload: dict[str, object],
) -> dict[str, dict[str, np.ndarray]]:
    """Translate the audited artifact to the public LeRobot stats contract."""

    validate_calvin_normalization_artifact(payload)

    def feature(name: str) -> dict[str, np.ndarray]:
        source = payload[name]
        if not isinstance(source, dict):
            raise RuntimeError("validated CALVIN normalization feature lost its mapping contract")
        return {
            field: np.asarray(source[field], dtype=np.float32)
            for field in ("min", "max", "mean", "std", "q01", "q99")
        } | {"mask": np.asarray(source["normalize_mask"], dtype=np.bool_)}

    output = {
        "observation.state": feature("state"),
        "action": feature("action"),
    }
    if any(not np.isfinite(value).all() for stats in output.values() for value in stats.values()):
        raise ContractError("translated MolmoAct2 normalization contains non-finite values")
    return output


def official_lingbot_calvin_norm_stats(
    payload: dict[str, object],
    *,
    dataset_tree_sha256: str,
) -> dict[str, object]:
    """Translate one audited CALVIN artifact to LingBot's typed feature schema."""

    validate_calvin_normalization_artifact(payload)
    if payload["schema"] != CALVIN_NORMALIZATION_SCHEMA:
        raise ContractError("LingBot CALVIN normalization requires tree-bound source statistics")
    if (
        not isinstance(dataset_tree_sha256, str)
        or len(dataset_tree_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_tree_sha256)
    ):
        raise ContractError("LingBot CALVIN dataset tree must be a SHA-256 digest")
    if payload["dataset_tree_sha256"] != dataset_tree_sha256:
        raise ContractError("LingBot CALVIN normalization and dataset tree differ")
    state = payload["state"]
    action = payload["action"]
    if not isinstance(state, dict) or not isinstance(action, dict):
        raise RuntimeError("validated CALVIN normalization lost its feature payloads")

    def sliced(source: dict[str, object], selection: slice) -> dict[str, list[float]]:
        return {
            field: np.asarray(source[field], dtype=np.float64)[selection].tolist()
            for field in ("mean", "std", "q01", "q99")
        }

    norm_stats = {
        "observation.state.arm.position": sliced(state, slice(7, 14)),
        "observation.state.end.position": sliced(state, slice(0, 6)),
        "observation.state.effector.position": sliced(state, slice(6, 7)),
        "action.end.position": sliced(action, slice(0, 6)),
        "action.effector.position": sliced(action, slice(6, 7)),
    }
    translated: dict[str, object] = {
        "schema": LINGBOT_CALVIN_NORMALIZATION_SCHEMA,
        "count": payload["sample_count"],
        "source": {
            "schema": payload["schema"],
            "dataset_id": payload["dataset_id"],
            "dataset_revision": payload["dataset_revision"],
            "dataset_tree_sha256": dataset_tree_sha256,
            "artifact_sha256": payload["artifact_sha256"],
            "ordered_sample_keys_sha256": payload["ordered_sample_keys_sha256"],
            "unique_source_frame_count": payload["unique_source_frame_count"],
        },
        "norm_stats": norm_stats,
    }
    translated["artifact_sha256"] = hashlib.sha256(_canonical_bytes(translated)).hexdigest()
    validate_lingbot_calvin_norm_stats(translated)
    return translated


def validate_lingbot_calvin_norm_stats(payload: object) -> None:
    expected_fields = {"artifact_sha256", "count", "norm_stats", "schema", "source"}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ContractError("LingBot CALVIN normalization fields differ from schema")
    if payload["schema"] != LINGBOT_CALVIN_NORMALIZATION_SCHEMA:
        raise ContractError("unsupported LingBot CALVIN normalization schema")
    count = payload["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ContractError("LingBot CALVIN normalization count must be positive")
    source = payload["source"]
    expected_source = {
        "artifact_sha256",
        "dataset_id",
        "dataset_revision",
        "dataset_tree_sha256",
        "ordered_sample_keys_sha256",
        "schema",
        "unique_source_frame_count",
    }
    if not isinstance(source, dict) or set(source) != expected_source:
        raise ContractError("LingBot CALVIN normalization source contract changed")
    if source["schema"] != CALVIN_NORMALIZATION_SCHEMA:
        raise ContractError("LingBot CALVIN normalization source schema changed")
    for name in ("dataset_id", "dataset_revision"):
        if not isinstance(source[name], str) or not source[name]:
            raise ContractError(f"LingBot CALVIN source {name} must be nonempty")
    unique_source_frame_count = source["unique_source_frame_count"]
    if (
        not isinstance(unique_source_frame_count, int)
        or isinstance(unique_source_frame_count, bool)
        or not 0 < unique_source_frame_count <= count
    ):
        raise ContractError("LingBot CALVIN unique source count must lie within count")
    for name in ("artifact_sha256", "dataset_tree_sha256", "ordered_sample_keys_sha256"):
        value = source[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError(f"LingBot CALVIN source {name} must be a SHA-256 digest")
    expected_widths = {
        "observation.state.arm.position": 7,
        "observation.state.end.position": 6,
        "observation.state.effector.position": 1,
        "action.end.position": 6,
        "action.effector.position": 1,
    }
    norm_stats = payload["norm_stats"]
    if not isinstance(norm_stats, dict) or set(norm_stats) != set(expected_widths):
        raise ContractError("LingBot CALVIN normalized feature set changed")
    for name, width in expected_widths.items():
        feature = norm_stats[name]
        if not isinstance(feature, dict) or set(feature) != {"mean", "q01", "q99", "std"}:
            raise ContractError(f"LingBot CALVIN normalization {name} fields changed")
        vectors = {
            field: np.asarray(feature[field], dtype=np.float64)
            for field in ("mean", "std", "q01", "q99")
        }
        if any(
            vector.shape != (width,) or not np.isfinite(vector).all() for vector in vectors.values()
        ):
            raise ContractError(f"LingBot CALVIN normalization {name} is invalid")
        if np.any(vectors["std"] < 0) or np.any(vectors["q01"] > vectors["q99"]):
            raise ContractError(f"LingBot CALVIN normalization {name} is unordered")
    recorded = payload["artifact_sha256"]
    canonical = dict(payload)
    canonical.pop("artifact_sha256")
    actual = hashlib.sha256(_canonical_bytes(canonical)).hexdigest()
    if recorded != actual:
        raise ContractError("LingBot CALVIN normalization artifact SHA-256 changed")
