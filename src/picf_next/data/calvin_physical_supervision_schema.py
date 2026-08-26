"""Torch-free contract for unified CALVIN physical supervision."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
)

CALVIN_PHYSICAL_SUPERVISION_SCHEMA = "picf-next.calvin-physical-supervision-sidecar.v2"
CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA = "picf-next.calvin-physical-supervision-sidecar.v5"
CALVIN_PHYSICAL_SUPERVISION_PARTITION_SCHEMA = "picf-next.calvin-physical-supervision-partition.v2"
CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA = (
    "picf-next.calvin-physical-supervision-partition.v5"
)
CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES = "language_frames"
CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES = "all_source_frames"
CALVIN_OWNER_CONTRACT = "exclusive-visible-physical-owner.zero-is-context.v1"
CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT = (
    "exclusive-visible-physical-owner.depth-consistent-unknown.zero-is-context.v2"
)
CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION = {
    "schema": "picf-next.calvin-depth-consistent-owner-supervision.v1",
    "maximum_absolute_depth_error_m": 0.01,
    "inconsistent_pixel_semantics": "unknown",
}
CALVIN_CAMERA_SPECS = (
    {
        "camera_name": "static",
        "host_image_key": "observation.images.image",
        "source_rgb_field": "rgb_static",
        "source_depth_field": "depth_static",
        "height": 200,
        "width": 200,
    },
    {
        "camera_name": "gripper",
        "host_image_key": "observation.images.wrist_image",
        "source_rgb_field": "rgb_gripper",
        "source_depth_field": "depth_gripper",
        "height": 84,
        "width": 84,
    },
)
CALVIN_CALIBRATION_LIMITS = {
    "maximum_rgb_mae": 36.0,
    "maximum_depth_mean_absolute_error_m": 0.025,
    "maximum_depth_p95_absolute_error_m": 0.03,
}
CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS = {
    "schema": "picf-next.calvin-depth-consistent-frame-diagnostics.v1",
    "aggregate_frame_metrics": "diagnostic-only",
    "supervision_acceptance": "per-pixel-owner-supervision",
}
CALVIN_DEPTH_CONSISTENT_DIAGNOSTIC_QUANTILES = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p50", 0.50),
)


def calvin_physical_calibration_summary_fields(coverage: str) -> frozenset[str]:
    fields = {
        f"maximum_{camera}_{metric}"
        for camera in ("static", "gripper")
        for metric in ("rgb_mae", "depth_mae_m", "depth_p95_m")
    }
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        fields.update(
            f"{statistic}_{camera}_depth_consistent_fraction"
            for camera in ("static", "gripper")
            for statistic in (
                "minimum",
                *(name for name, _quantile in CALVIN_DEPTH_CONSISTENT_DIAGNOSTIC_QUANTILES),
            )
        )
    elif coverage != CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES:
        raise ContractError("CALVIN physical supervision coverage is unsupported")
    return frozenset(fields)


def validate_calvin_depth_consistent_diagnostics(
    summary: Mapping[str, float],
) -> None:
    """Require bounded, monotone diagnostics without making them acceptance gates."""

    for camera in ("static", "gripper"):
        values = (
            summary[f"minimum_{camera}_depth_consistent_fraction"],
            *(
                summary[f"{name}_{camera}_depth_consistent_fraction"]
                for name, _quantile in CALVIN_DEPTH_CONSISTENT_DIAGNOSTIC_QUANTILES
            ),
        )
        if any(value > 1.0 for value in values) or any(
            right < left for left, right in pairwise(values)
        ):
            raise ContractError(
                "CALVIN depth-consistent diagnostics must be monotone and lie in [0, 1]"
            )


def calvin_camera_name_from_host_image_key(host_image_key: str) -> str:
    """Resolve the canonical CALVIN camera without guessing from key suffixes."""

    matches = tuple(
        str(spec["camera_name"])
        for spec in CALVIN_CAMERA_SPECS
        if spec["host_image_key"] == host_image_key
    )
    if len(matches) != 1:
        raise ContractError(f"unknown or ambiguous CALVIN host image key: {host_image_key!r}")
    return matches[0]


def source_array_sha256(name: str, value: np.ndarray) -> str:
    if not isinstance(name, str) or not name:
        raise ContractError("CALVIN source array name cannot be empty")
    array = np.asarray(value)
    if array.dtype.hasobject or not array.size:
        raise ContractError("CALVIN source arrays must be nonempty and object-free")
    digest = hashlib.sha256()
    digest.update(name.encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CalvinPhysicalSupervisionShard:
    path: str
    sha256: str
    first_global_index: int
    last_global_index: int
    frame_count: int
    object_record_count: int

    @classmethod
    def from_dict(cls, payload: object) -> CalvinPhysicalSupervisionShard:
        if not isinstance(payload, dict):
            raise ContractError("CALVIN physical shard metadata must be a mapping")
        expected = {
            "path",
            "sha256",
            "first_global_index",
            "last_global_index",
            "frame_count",
            "object_record_count",
        }
        if set(payload) != expected:
            raise ContractError("CALVIN physical shard metadata fields differ from schema")
        path = payload["path"]
        digest = payload["sha256"]
        integers = tuple(
            payload[name]
            for name in (
                "first_global_index",
                "last_global_index",
                "frame_count",
                "object_record_count",
            )
        )
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in integers)
        ):
            raise ContractError("CALVIN physical shard metadata contains invalid values")
        first, last, frames, records = integers
        if first < 0 or last < first or frames <= 0 or records <= 0:
            raise ContractError("CALVIN physical shard bounds/counts are invalid")
        return cls(path, digest, first, last, frames, records)


def physical_supervision_manifest_payload(
    *,
    dataset_id: str,
    dataset_revision: str,
    split_name: str,
    scene_info_sha256: str,
    global_indices: np.ndarray,
    shards: tuple[CalvinPhysicalSupervisionShard, ...],
    calibration_summary: dict[str, float],
    coverage: str = CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
) -> dict[str, Any]:
    if global_indices.dtype != np.int64 or global_indices.ndim != 1 or not len(global_indices):
        raise ContractError("CALVIN physical manifest requires int64 global indices")
    if np.any(global_indices[1:] <= global_indices[:-1]) or not shards:
        raise ContractError("CALVIN physical manifest coverage is invalid")
    if (
        not isinstance(scene_info_sha256, str)
        or len(scene_info_sha256) != 64
        or any(character not in "0123456789abcdef" for character in scene_info_sha256)
    ):
        raise ContractError("CALVIN physical scene_info fingerprint is invalid")
    if coverage not in {
        CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
        CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    }:
        raise ContractError("CALVIN physical supervision coverage is unsupported")
    expected_summary = calvin_physical_calibration_summary_fields(coverage)
    if set(calibration_summary) != expected_summary or any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
        or value < 0.0
        for value in calibration_summary.values()
    ):
        raise ContractError("CALVIN physical calibration summary is invalid")
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        validate_calvin_depth_consistent_diagnostics(calibration_summary)
    payload = {
        "schema": (
            CALVIN_PHYSICAL_SUPERVISION_SCHEMA
            if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
            else CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA
        ),
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split_name": split_name,
        "calvin_commit": CALVIN_SOURCE_COMMIT,
        "calvin_env_commit": CALVIN_ENV_SOURCE_COMMIT,
        "state_restoration": CALVIN_STATE_RESTORATION,
        "geometry_contract": CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict(),
        "geometry_contract_sha256": CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint,
        "owner_contract": (
            CALVIN_OWNER_CONTRACT
            if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
            else CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT
        ),
        "camera_specs": [dict(value) for value in CALVIN_CAMERA_SPECS],
        "calibration_summary": dict(sorted(calibration_summary.items())),
        "runtime_input": False,
        "task_conditioned": False,
        "source_fields": [
            "depth_gripper",
            "depth_static",
            "rgb_gripper",
            "rgb_static",
            "robot_obs",
            "scene_info",
            "scene_obs",
        ],
        "scene_info_sha256": scene_info_sha256,
        "frame_count": int(sum(shard.frame_count for shard in shards)),
        "object_record_count": int(sum(shard.object_record_count for shard in shards)),
        "global_indices_sha256": hashlib.sha256(global_indices.tobytes(order="C")).hexdigest(),
        "shards": [
            {
                "path": shard.path,
                "sha256": shard.sha256,
                "first_global_index": shard.first_global_index,
                "last_global_index": shard.last_global_index,
                "frame_count": shard.frame_count,
                "object_record_count": shard.object_record_count,
            }
            for shard in shards
        ],
    }
    if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES:
        payload["calibration_limits"] = dict(CALVIN_CALIBRATION_LIMITS)
    else:
        payload["coverage"] = coverage
        payload["owner_supervision"] = dict(CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION)
        payload["frame_diagnostics"] = dict(CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS)
    return payload
