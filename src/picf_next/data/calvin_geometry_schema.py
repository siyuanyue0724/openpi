"""Torch-free schema shared by CALVIN geometry extraction and training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from picf_next.contracts import ContractError
from picf_next.geometry import PhysicalGeometryContract

CALVIN_GEOMETRY_SIDECAR_SCHEMA = "picf-next.calvin-physical-geometry-sidecar.v4"
CALVIN_STATE_RESTORATION = "direct-scene-robot-reset.zero-movable-velocity.no-step.v1"
CALVIN_SOURCE_COMMIT = "fa03f01f19c65920e18cf37398a9ce859274af76"
CALVIN_ENV_SOURCE_COMMIT = "1431a46bd36bde5903fb6345e68b5ccc30def666"
CALVIN_M2_TRAIN_VISIBLE_GEOMETRY_ROWS = 1813
CALVIN_M2_TRAIN_GEOMETRY_OFFSET = (
    0.3796050103655587,
    0.458837615707206,
    0.21802569315108686,
)
CALVIN_M2_TRAIN_GEOMETRY_SCALE = (
    0.1962029379638472,
    0.12934434519296104,
    0.09772606317702993,
)
CALVIN_OBJECT_GEOMETRY_CONTRACT = PhysicalGeometryContract(
    name="calvin.object-link-aabb-center.robot-base.m2-train-standardized.v2",
    quantity="object_link_axis_aligned_bounding_box_center",
    reference_frame="robot_base",
    axes=("x", "y", "z"),
    units=("m", "m", "m"),
    normalization_offset=CALVIN_M2_TRAIN_GEOMETRY_OFFSET,
    normalization_scale=CALVIN_M2_TRAIN_GEOMETRY_SCALE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calvin_source_state_sha256(scene_obs: np.ndarray, robot_obs: np.ndarray) -> str:
    """Fingerprint the exact privileged state used for one offline extraction."""

    digest = hashlib.sha256()
    for name, value in (("scene_obs", scene_obs), ("robot_obs", robot_obs)):
        array = np.asarray(value)
        if array.dtype.hasobject or not array.size:
            raise ContractError("CALVIN source state must be a nonempty numeric array")
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CalvinGeometryShard:
    path: str
    sha256: str
    first_global_index: int
    last_global_index: int
    frame_count: int
    object_record_count: int

    @classmethod
    def from_dict(cls, payload: object) -> CalvinGeometryShard:
        if not isinstance(payload, dict):
            raise ContractError("CALVIN geometry shard metadata must be a mapping")
        expected = {
            "path",
            "sha256",
            "first_global_index",
            "last_global_index",
            "frame_count",
            "object_record_count",
        }
        if set(payload) != expected:
            raise ContractError("CALVIN geometry shard metadata fields differ from schema")
        path = payload["path"]
        sha256 = payload["sha256"]
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
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in integers)
        ):
            raise ContractError("CALVIN geometry shard metadata contains invalid values")
        first, last, frame_count, record_count = integers
        if first < 0 or last < first or frame_count <= 0 or record_count <= 0:
            raise ContractError("CALVIN geometry shard bounds/counts are invalid")
        return cls(path, sha256, first, last, frame_count, record_count)


def geometry_manifest_payload(
    *,
    dataset_id: str,
    dataset_revision: str,
    split_name: str,
    calvin_commit: str,
    calvin_env_commit: str,
    scene_info_sha256: str,
    global_indices: np.ndarray,
    shards: tuple[CalvinGeometryShard, ...],
) -> dict[str, Any]:
    """Build the canonical manifest after independently verified extraction."""

    if global_indices.dtype != np.int64 or global_indices.ndim != 1 or not len(global_indices):
        raise ContractError("CALVIN geometry manifest requires int64 global indices")
    if np.any(global_indices[1:] <= global_indices[:-1]):
        raise ContractError("CALVIN geometry manifest indices must be strictly increasing")
    if not shards:
        raise ContractError("CALVIN geometry manifest requires shards")
    if (
        not isinstance(scene_info_sha256, str)
        or len(scene_info_sha256) != 64
        or any(character not in "0123456789abcdef" for character in scene_info_sha256)
    ):
        raise ContractError("CALVIN scene_info fingerprint is invalid")
    digest = hashlib.sha256(global_indices.tobytes(order="C")).hexdigest()
    return {
        "schema": CALVIN_GEOMETRY_SIDECAR_SCHEMA,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split_name": split_name,
        "calvin_commit": calvin_commit,
        "calvin_env_commit": calvin_env_commit,
        "state_restoration": CALVIN_STATE_RESTORATION,
        "geometry_contract": CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict(),
        "geometry_contract_sha256": CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint,
        "runtime_input": False,
        "task_conditioned": False,
        "source_fields": ["robot_obs", "scene_info", "scene_obs"],
        "scene_info_sha256": scene_info_sha256,
        "frame_count": int(sum(shard.frame_count for shard in shards)),
        "object_record_count": int(sum(shard.object_record_count for shard in shards)),
        "global_indices_sha256": digest,
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
