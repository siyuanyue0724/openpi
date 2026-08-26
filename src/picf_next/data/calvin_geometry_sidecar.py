"""Versioned loss-only CALVIN physical geometry sidecars.

The sidecar contains simulator-derived object/link AABB centres in the robot
base frame.  It is consumed only after the deploy-visible forward pass.  The
runtime record, discovery model, action expert and committed posterior never
receive simulator state or these labels.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_GEOMETRY_SIDECAR_SCHEMA,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
    CalvinGeometryShard,
    sha256_file,
)
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath
from picf_next.data.rollout_targets import PhysicalObjectGeometryFrame
from picf_next.geometry import PhysicalGeometryContract

_MAXIMUM_SIDECAR_SHARD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _LoadedShard:
    global_indices: np.ndarray
    source_state_sha256: np.ndarray
    frame_offsets: np.ndarray
    identity_keys: np.ndarray
    geometry: np.ndarray
    geometry_variance: np.ndarray
    geometry_supervised: np.ndarray
    frame_by_global_index: dict[int, int]


class CalvinPhysicalGeometrySidecar:
    """Hash-verified random access to loss-only CALVIN object geometry."""

    def __init__(
        self,
        root: Path,
        index: CalvinDatasetIndex,
        *,
        manifest_bytes: bytes | None = None,
        verify_hashes: bool = True,
        cache_shards: int = 2,
    ) -> None:
        if not isinstance(index, CalvinDatasetIndex):
            raise TypeError("CALVIN geometry sidecar requires a CalvinDatasetIndex")
        if not isinstance(verify_hashes, bool):
            raise TypeError("verify_hashes must be boolean")
        if not verify_hashes:
            raise ContractError("CALVIN geometry shard hash verification cannot be disabled")
        if not isinstance(cache_shards, int) or isinstance(cache_shards, bool) or cache_shards <= 0:
            raise ValueError("cache_shards must be positive")
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if manifest_bytes is None:
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            try:
                manifest_bytes = manifest_path.read_bytes()
            except OSError as error:
                raise ContractError("CALVIN geometry manifest cannot be read") from error
        elif not isinstance(manifest_bytes, bytes):
            raise TypeError("CALVIN geometry manifest_bytes must be immutable bytes")
        try:
            manifest = json.loads(manifest_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ContractError("CALVIN geometry manifest is not valid JSON") from error
        if not isinstance(manifest, dict):
            raise ContractError("CALVIN geometry manifest must be a mapping")
        expected_fields = {
            "schema",
            "dataset_id",
            "dataset_revision",
            "split_name",
            "calvin_commit",
            "calvin_env_commit",
            "state_restoration",
            "geometry_contract",
            "geometry_contract_sha256",
            "runtime_input",
            "task_conditioned",
            "source_fields",
            "scene_info_sha256",
            "frame_count",
            "object_record_count",
            "global_indices_sha256",
            "shards",
        }
        if set(manifest) != expected_fields:
            raise ContractError("CALVIN geometry manifest fields differ from schema v3")
        if manifest["schema"] != CALVIN_GEOMETRY_SIDECAR_SCHEMA:
            raise ContractError("unsupported CALVIN geometry sidecar schema")
        if manifest["dataset_id"] != index.dataset_id or (
            manifest["dataset_revision"] != index.dataset_revision
        ):
            raise ContractError("CALVIN geometry sidecar dataset identity differs from index")
        if manifest["runtime_input"] is not False or manifest["task_conditioned"] is not False:
            raise ContractError("CALVIN geometry sidecar must be loss-only and task-independent")
        if manifest["source_fields"] != ["robot_obs", "scene_info", "scene_obs"]:
            raise ContractError("CALVIN geometry sidecar source fields differ from schema v3")
        scene_info_path = index.split_root / "scene_info.npy"
        if not scene_info_path.is_file():
            raise FileNotFoundError(scene_info_path)
        if manifest["scene_info_sha256"] != sha256_file(scene_info_path):
            raise ContractError("CALVIN geometry sidecar scene assignment differs from dataset")
        if manifest["split_name"] != index.split_root.name:
            raise ContractError("CALVIN geometry sidecar split differs from dataset index")
        if manifest["calvin_commit"] != CALVIN_SOURCE_COMMIT or (
            manifest["calvin_env_commit"] != CALVIN_ENV_SOURCE_COMMIT
        ):
            raise ContractError("CALVIN geometry sidecar source revisions are not pinned")
        if manifest["state_restoration"] != CALVIN_STATE_RESTORATION:
            raise ContractError(
                "CALVIN geometry sidecar restoration semantics differ from schema v3"
            )
        contract = PhysicalGeometryContract.from_dict(manifest["geometry_contract"])
        if contract != CALVIN_OBJECT_GEOMETRY_CONTRACT:
            raise ContractError("CALVIN geometry sidecar uses an unexpected physical chart")
        if manifest["geometry_contract_sha256"] != contract.fingerprint:
            raise ContractError("CALVIN geometry contract fingerprint is invalid")
        counts = manifest["frame_count"], manifest["object_record_count"]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts
        ):
            raise ContractError("CALVIN geometry manifest counts must be positive integers")
        global_digest = manifest["global_indices_sha256"]
        if (
            not isinstance(global_digest, str)
            or len(global_digest) != 64
            or any(character not in "0123456789abcdef" for character in global_digest)
        ):
            raise ContractError("CALVIN geometry global-index fingerprint is invalid")
        raw_shards = manifest["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ContractError("CALVIN geometry manifest requires at least one shard")
        shards = tuple(CalvinGeometryShard.from_dict(value) for value in raw_shards)
        if tuple(sorted(shards, key=lambda shard: shard.first_global_index)) != shards:
            raise ContractError("CALVIN geometry shards must be sorted")
        for previous, current in zip(shards, shards[1:], strict=False):
            if current.first_global_index <= previous.last_global_index:
                raise ContractError("CALVIN geometry shard ranges overlap")
        if (
            sum(shard.frame_count for shard in shards) != counts[0]
            or sum(shard.object_record_count for shard in shards) != counts[1]
        ):
            raise ContractError("CALVIN geometry manifest counts disagree with shards")
        for shard in shards:
            path = self.root / shard.path
            if not path.is_file():
                raise FileNotFoundError(path)

        expected_indices = np.asarray(
            sorted(
                {
                    global_index
                    for segment in index.segments
                    for global_index in range(segment.start, segment.end + 1)
                }
            ),
            dtype=np.int64,
        )
        if expected_indices.shape != (counts[0],) or (
            hashlib.sha256(expected_indices.tobytes(order="C")).hexdigest() != global_digest
        ):
            raise ContractError("CALVIN geometry manifest does not cover every language frame")
        cursor = 0
        for metadata in shards:
            payload = read_sha256_verified_file_beneath(
                self.root,
                metadata.path,
                expected_sha256=metadata.sha256,
                maximum_bytes=_MAXIMUM_SIDECAR_SHARD_BYTES,
            )
            with np.load(BytesIO(payload), allow_pickle=False) as archive:
                if "global_indices" not in archive.files:
                    raise ContractError("CALVIN geometry shard has no global index")
                shard_indices = archive["global_indices"]
            if (
                shard_indices.dtype != np.int64
                or shard_indices.shape != (metadata.frame_count,)
                or not np.array_equal(
                    shard_indices,
                    expected_indices[cursor : cursor + metadata.frame_count],
                )
            ):
                raise ContractError("CALVIN geometry shard coverage differs from manifest")
            cursor += metadata.frame_count
        if cursor != len(expected_indices):
            raise ContractError("CALVIN geometry shard coverage is incomplete")

        self.index = index
        self.geometry_contract = contract
        self.shards = shards
        self._starts = tuple(shard.first_global_index for shard in shards)
        self._cache_capacity = cache_shards
        self._cache: OrderedDict[int, _LoadedShard] = OrderedDict()

    def _shard_index(self, global_index: int) -> int:
        position = bisect_right(self._starts, global_index) - 1
        if position < 0 or global_index > self.shards[position].last_global_index:
            raise KeyError(f"CALVIN geometry sidecar does not cover frame {global_index}")
        return position

    def _load_shard(self, shard_index: int) -> _LoadedShard:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        metadata = self.shards[shard_index]
        payload = read_sha256_verified_file_beneath(
            self.root,
            metadata.path,
            expected_sha256=metadata.sha256,
            maximum_bytes=_MAXIMUM_SIDECAR_SHARD_BYTES,
        )
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "global_indices",
                "source_state_sha256",
                "frame_offsets",
                "identity_keys",
                "geometry",
                "geometry_variance",
                "geometry_supervised",
            }:
                raise ContractError("CALVIN geometry shard arrays differ from schema v1")
            arrays = {name: archive[name].copy() for name in archive.files}
        global_indices = arrays["global_indices"]
        source_state_sha256 = arrays["source_state_sha256"]
        frame_offsets = arrays["frame_offsets"]
        identity_keys = arrays["identity_keys"]
        geometry = arrays["geometry"]
        variance = arrays["geometry_variance"]
        supervised = arrays["geometry_supervised"]
        record_count = metadata.object_record_count
        geometry_shape = (record_count, self.geometry_contract.dimension)
        if (
            global_indices.dtype != np.int64
            or global_indices.shape != (metadata.frame_count,)
            or not np.issubdtype(source_state_sha256.dtype, np.str_)
            or source_state_sha256.shape != (metadata.frame_count,)
            or frame_offsets.dtype != np.int64
            or frame_offsets.shape != (metadata.frame_count + 1,)
            or not np.issubdtype(identity_keys.dtype, np.str_)
            or identity_keys.shape != (record_count,)
            or geometry.dtype != np.float32
            or geometry.shape != geometry_shape
            or variance.dtype != np.float32
            or variance.shape != geometry_shape
            or supervised.dtype != np.bool_
            or supervised.shape != geometry_shape
        ):
            raise ContractError("CALVIN geometry shard shapes or dtypes differ from schema v1")
        if (
            np.any(global_indices[1:] <= global_indices[:-1])
            or int(global_indices[0]) != metadata.first_global_index
            or int(global_indices[-1]) != metadata.last_global_index
            or int(frame_offsets[0]) != 0
            or int(frame_offsets[-1]) != record_count
            or np.any(frame_offsets[1:] <= frame_offsets[:-1])
        ):
            raise ContractError("CALVIN geometry shard indexing is invalid")
        if (
            any(not key for key in identity_keys.tolist())
            or any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in source_state_sha256.tolist()
            )
            or not np.isfinite(geometry).all()
            or not np.isfinite(variance).all()
            or (variance < 0.0).any()
            or (geometry[~supervised] != 0.0).any()
            or (variance[~supervised] != 0.0).any()
            or not supervised.any(axis=-1).all()
        ):
            raise ContractError("CALVIN geometry shard contains invalid target values")
        for start, stop in zip(frame_offsets[:-1], frame_offsets[1:], strict=True):
            frame_keys = identity_keys[int(start) : int(stop)].tolist()
            if len(set(frame_keys)) != len(frame_keys):
                raise ContractError("CALVIN geometry keys must be unique within each frame")
        loaded = _LoadedShard(
            global_indices=global_indices,
            source_state_sha256=source_state_sha256,
            frame_offsets=frame_offsets,
            identity_keys=identity_keys,
            geometry=geometry,
            geometry_variance=variance,
            geometry_supervised=supervised,
            frame_by_global_index={
                int(value): row for row, value in enumerate(global_indices.tolist())
            },
        )
        self._cache[shard_index] = loaded
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)
        return loaded

    def __call__(self, segment_index: int, global_index: int) -> PhysicalObjectGeometryFrame:
        if (
            not isinstance(segment_index, int)
            or isinstance(segment_index, bool)
            or not 0 <= segment_index < len(self.index.segments)
        ):
            raise ContractError("unknown CALVIN language segment")
        segment = self.index.segments[segment_index]
        if global_index < segment.start or global_index > segment.end:
            raise ContractError("CALVIN geometry request lies outside its language segment")
        shard = self._load_shard(self._shard_index(global_index))
        frame_row = shard.frame_by_global_index.get(global_index)
        if frame_row is None:
            raise KeyError(f"CALVIN geometry sidecar does not cover frame {global_index}")
        start = int(shard.frame_offsets[frame_row])
        stop = int(shard.frame_offsets[frame_row + 1])
        return PhysicalObjectGeometryFrame(
            identity_keys=tuple(str(key) for key in shard.identity_keys[start:stop].tolist()),
            geometry=torch.from_numpy(shard.geometry[start:stop].copy()),
            geometry_variance=torch.from_numpy(shard.geometry_variance[start:stop].copy()),
            geometry_supervised=torch.from_numpy(shard.geometry_supervised[start:stop].copy()),
            geometry_contract=self.geometry_contract,
        )

    def source_state_sha256(self, segment_index: int, global_index: int) -> str:
        """Return provenance for an offline audit without exposing it to training."""

        if (
            not isinstance(segment_index, int)
            or isinstance(segment_index, bool)
            or not 0 <= segment_index < len(self.index.segments)
        ):
            raise ContractError("unknown CALVIN language segment")
        segment = self.index.segments[segment_index]
        if global_index < segment.start or global_index > segment.end:
            raise ContractError("CALVIN geometry request lies outside its language segment")
        shard = self._load_shard(self._shard_index(global_index))
        frame_row = shard.frame_by_global_index.get(global_index)
        if frame_row is None:
            raise KeyError(f"CALVIN geometry sidecar does not cover frame {global_index}")
        return str(shard.source_state_sha256[frame_row])

    def clear_cache(self) -> None:
        self._cache.clear()
