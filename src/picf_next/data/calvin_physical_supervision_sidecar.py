"""Hash-verified CALVIN geometry and visible-owner supervision."""

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
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
    sha256_file,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CALIBRATION_LIMITS,
    CALVIN_CAMERA_SPECS,
    CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS,
    CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT,
    CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION,
    CALVIN_OWNER_CONTRACT,
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
    CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA,
    CALVIN_PHYSICAL_SUPERVISION_SCHEMA,
    CalvinPhysicalSupervisionShard,
    calvin_physical_calibration_summary_fields,
    validate_calvin_depth_consistent_diagnostics,
)
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath
from picf_next.data.rollout_targets import PhysicalObjectGeometryFrame
from picf_next.geometry import PhysicalGeometryContract

_MAXIMUM_SIDECAR_SHARD_BYTES = 512 * 1024 * 1024
_MAXIMUM_SIDECAR_MANIFEST_BYTES = 16 * 1024 * 1024


def _expected_coverage_indices(
    index: CalvinDatasetIndex,
    coverage: str,
) -> np.ndarray:
    if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES:
        values = {
            global_index
            for segment in index.segments
            for global_index in range(segment.start, segment.end + 1)
        }
    elif coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        return np.concatenate(
            [
                np.arange(episode.start, episode.end + 1, dtype=np.int64)
                for episode in index.episodes
            ]
        )
    else:
        raise ContractError("CALVIN physical supervision coverage is unsupported")
    return np.asarray(sorted(values), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CalvinVisibleOwnerRaster:
    camera_name: str
    host_image_key: str
    owner_index: np.ndarray
    owner_supervised: np.ndarray
    source_rgb_sha256: str
    source_depth_sha256: str
    rgb_mae: float
    depth_mae_m: float
    depth_p95_m: float
    depth_consistent_fraction: float

    def __post_init__(self) -> None:
        specs = {str(value["camera_name"]): value for value in CALVIN_CAMERA_SPECS}
        spec = specs.get(self.camera_name)
        if spec is None or self.host_image_key != spec["host_image_key"]:
            raise ContractError("CALVIN visible-owner camera identity is invalid")
        expected_shape = (int(spec["height"]), int(spec["width"]))
        if self.owner_index.dtype != np.uint8 or self.owner_index.shape != expected_shape:
            raise ContractError("CALVIN visible-owner raster shape or dtype is invalid")
        if self.owner_supervised.dtype != np.bool_ or self.owner_supervised.shape != expected_shape:
            raise ContractError("CALVIN visible-owner supervision shape or dtype is invalid")
        for digest in (self.source_rgb_sha256, self.source_depth_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ContractError("CALVIN visible-owner source hash is invalid")
        metrics = (
            self.rgb_mae,
            self.depth_mae_m,
            self.depth_p95_m,
            self.depth_consistent_fraction,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in metrics):
            raise ContractError("CALVIN visible-owner calibration metric is invalid")
        if self.depth_consistent_fraction > 1.0:
            raise ContractError("CALVIN depth-consistent fraction must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class CalvinPhysicalSupervisionFrame:
    identity_keys: tuple[str, ...]
    geometry: torch.Tensor
    geometry_variance: torch.Tensor
    geometry_supervised: torch.Tensor
    geometry_contract: PhysicalGeometryContract
    cameras: tuple[CalvinVisibleOwnerRaster, ...]

    def geometry_frame(self) -> PhysicalObjectGeometryFrame:
        return PhysicalObjectGeometryFrame(
            identity_keys=self.identity_keys,
            geometry=self.geometry,
            geometry_variance=self.geometry_variance,
            geometry_supervised=self.geometry_supervised,
            geometry_contract=self.geometry_contract,
        )


@dataclass(frozen=True, slots=True)
class _LoadedPhysicalShard:
    global_indices: np.ndarray
    source_state_sha256: np.ndarray
    frame_offsets: np.ndarray
    identity_keys: np.ndarray
    geometry: np.ndarray
    geometry_variance: np.ndarray
    geometry_supervised: np.ndarray
    camera_arrays: dict[str, dict[str, np.ndarray]]
    frame_by_global_index: dict[int, int]


class CalvinPhysicalSupervisionSidecar:
    """Random-access, loss-only CALVIN physical supervision."""

    def __init__(
        self,
        root: Path,
        index: CalvinDatasetIndex,
        *,
        manifest_path: Path | None = None,
        manifest_bytes: bytes | None = None,
        expected_manifest_sha256: str | None = None,
        verify_hashes: bool = True,
        cache_shards: int = 2,
        eager_coverage_scan: bool = True,
    ) -> None:
        if not isinstance(index, CalvinDatasetIndex):
            raise TypeError("CALVIN physical sidecar requires a CalvinDatasetIndex")
        if not isinstance(verify_hashes, bool):
            raise TypeError("verify_hashes must be boolean")
        if not verify_hashes:
            raise ContractError("CALVIN physical shard hash verification cannot be disabled")
        if not isinstance(eager_coverage_scan, bool):
            raise TypeError("eager_coverage_scan must be boolean")
        if not isinstance(cache_shards, int) or isinstance(cache_shards, bool) or cache_shards <= 0:
            raise ValueError("cache_shards must be positive")
        if expected_manifest_sha256 is not None and (
            not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_manifest_sha256)
        ):
            raise ContractError("CALVIN physical manifest SHA-256 is invalid")
        self.root = Path(root).resolve()
        if manifest_path is not None and manifest_bytes is not None:
            raise TypeError("CALVIN physical manifest_path and manifest_bytes are exclusive")
        selected_manifest_path = (
            self.root / "manifest.json"
            if manifest_path is None
            else Path(manifest_path).expanduser().absolute()
        )
        if manifest_bytes is None:
            if expected_manifest_sha256 is None:
                if not selected_manifest_path.is_file():
                    raise FileNotFoundError(selected_manifest_path)
                try:
                    manifest_bytes = selected_manifest_path.read_bytes()
                except OSError as error:
                    raise ContractError("CALVIN physical manifest cannot be read") from error
            else:
                manifest_bytes = read_sha256_verified_file_beneath(
                    selected_manifest_path.parent,
                    selected_manifest_path.name,
                    expected_sha256=expected_manifest_sha256,
                    maximum_bytes=_MAXIMUM_SIDECAR_MANIFEST_BYTES,
                )
        elif not isinstance(manifest_bytes, bytes):
            raise TypeError("CALVIN physical manifest_bytes must be immutable bytes")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
            raise ContractError("CALVIN physical manifest differs from its expected SHA-256")
        try:
            manifest = json.loads(manifest_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ContractError("CALVIN physical manifest is not valid JSON") from error
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
            "owner_contract",
            "camera_specs",
            "calibration_summary",
            "runtime_input",
            "task_conditioned",
            "source_fields",
            "scene_info_sha256",
            "frame_count",
            "object_record_count",
            "global_indices_sha256",
            "shards",
        }
        if not isinstance(manifest, dict):
            raise ContractError("CALVIN physical manifest must be a mapping")
        schema = manifest.get("schema")
        if schema == CALVIN_PHYSICAL_SUPERVISION_SCHEMA:
            expected_fields.add("calibration_limits")
            coverage = CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
        elif schema == CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA:
            expected_fields.update(("coverage", "owner_supervision", "frame_diagnostics"))
            coverage = manifest.get("coverage")
            if coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
                raise ContractError("CALVIN all-source physical coverage is invalid")
        else:
            raise ContractError("unsupported CALVIN physical supervision schema")
        if set(manifest) != expected_fields:
            raise ContractError("CALVIN physical manifest fields differ from schema")
        if (
            manifest["dataset_id"] != index.dataset_id
            or manifest["dataset_revision"] != index.dataset_revision
            or manifest["split_name"] != index.split_root.name
        ):
            raise ContractError("CALVIN physical sidecar dataset identity differs from index")
        if manifest["runtime_input"] is not False or manifest["task_conditioned"] is not False:
            raise ContractError("CALVIN physical sidecar must be loss-only and task-independent")
        if (
            manifest["calvin_commit"] != CALVIN_SOURCE_COMMIT
            or manifest["calvin_env_commit"] != CALVIN_ENV_SOURCE_COMMIT
            or manifest["state_restoration"] != CALVIN_STATE_RESTORATION
        ):
            raise ContractError("CALVIN physical sidecar source/restoration contract drifted")
        expected_owner_contract = (
            CALVIN_OWNER_CONTRACT
            if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
            else CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT
        )
        if manifest["owner_contract"] != expected_owner_contract:
            raise ContractError("CALVIN visible-owner contract drifted")
        if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES and manifest.get(
            "owner_supervision"
        ) != dict(CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION):
            raise ContractError("CALVIN visible-owner supervision contract drifted")
        if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES and manifest.get(
            "frame_diagnostics"
        ) != dict(CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS):
            raise ContractError("CALVIN frame-diagnostic contract drifted")
        if manifest["camera_specs"] != [dict(value) for value in CALVIN_CAMERA_SPECS]:
            raise ContractError("CALVIN physical camera contract drifted")
        if (
            coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
            and manifest["calibration_limits"] != CALVIN_CALIBRATION_LIMITS
        ):
            raise ContractError("CALVIN physical calibration limits drifted")
        expected_source_fields = [
            "depth_gripper",
            "depth_static",
            "rgb_gripper",
            "rgb_static",
            "robot_obs",
            "scene_info",
            "scene_obs",
        ]
        if manifest["source_fields"] != expected_source_fields:
            raise ContractError("CALVIN physical source fields drifted")
        scene_info_path = index.split_root / "scene_info.npy"
        if manifest["scene_info_sha256"] != sha256_file(scene_info_path):
            raise ContractError("CALVIN physical scene assignment differs from dataset")
        contract = PhysicalGeometryContract.from_dict(manifest["geometry_contract"])
        if contract != CALVIN_OBJECT_GEOMETRY_CONTRACT:
            raise ContractError("CALVIN physical sidecar uses an unexpected geometry chart")
        if manifest["geometry_contract_sha256"] != contract.fingerprint:
            raise ContractError("CALVIN physical geometry fingerprint is invalid")
        summary = manifest["calibration_summary"]
        if not isinstance(summary, dict) or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0.0
            for value in summary.values()
        ):
            raise ContractError("CALVIN physical calibration summary is invalid")
        expected_summary_fields = calvin_physical_calibration_summary_fields(coverage)
        if set(summary) != expected_summary_fields:
            raise ContractError("CALVIN physical calibration summary fields drifted")
        if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            validate_calvin_depth_consistent_diagnostics(summary)
        if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES:
            for camera in ("static", "gripper"):
                if (
                    summary.get(f"maximum_{camera}_rgb_mae", float("inf"))
                    > CALVIN_CALIBRATION_LIMITS["maximum_rgb_mae"]
                ):
                    raise ContractError("CALVIN physical RGB calibration exceeds its limit")
                if (
                    summary.get(f"maximum_{camera}_depth_mae_m", float("inf"))
                    > CALVIN_CALIBRATION_LIMITS["maximum_depth_mean_absolute_error_m"]
                ):
                    raise ContractError("CALVIN physical mean depth calibration exceeds its limit")
                if (
                    summary.get(f"maximum_{camera}_depth_p95_m", float("inf"))
                    > CALVIN_CALIBRATION_LIMITS["maximum_depth_p95_absolute_error_m"]
                ):
                    raise ContractError("CALVIN physical p95 depth calibration exceeds its limit")
        counts = manifest["frame_count"], manifest["object_record_count"]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts
        ):
            raise ContractError("CALVIN physical manifest counts must be positive")
        raw_shards = manifest["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ContractError("CALVIN physical manifest requires shards")
        shards = tuple(CalvinPhysicalSupervisionShard.from_dict(value) for value in raw_shards)
        if tuple(sorted(shards, key=lambda item: item.first_global_index)) != shards:
            raise ContractError("CALVIN physical shards must be sorted")
        for previous, current in zip(shards, shards[1:], strict=False):
            if current.first_global_index <= previous.last_global_index:
                raise ContractError("CALVIN physical shard ranges overlap")
        if (
            sum(item.frame_count for item in shards) != counts[0]
            or sum(item.object_record_count for item in shards) != counts[1]
        ):
            raise ContractError("CALVIN physical manifest counts disagree with shards")
        for shard in shards:
            path = self.root / shard.path
            if not path.is_file():
                raise FileNotFoundError(path)

        expected_indices = _expected_coverage_indices(index, coverage)
        if (
            expected_indices.shape != (counts[0],)
            or hashlib.sha256(expected_indices.tobytes(order="C")).hexdigest()
            != manifest["global_indices_sha256"]
        ):
            raise ContractError("CALVIN physical manifest does not cover its declared frame set")
        cursor = 0
        if not eager_coverage_scan and coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            raise ContractError("lazy coverage validation requires all-source supervision")
        for metadata in shards:
            stop = cursor + metadata.frame_count
            expected_shard_indices = expected_indices[cursor:stop]
            if (
                expected_shard_indices.shape != (metadata.frame_count,)
                or int(expected_shard_indices[0]) != metadata.first_global_index
                or int(expected_shard_indices[-1]) != metadata.last_global_index
            ):
                raise ContractError("CALVIN physical shard metadata differs from coverage")
            if eager_coverage_scan:
                payload = read_sha256_verified_file_beneath(
                    self.root,
                    metadata.path,
                    expected_sha256=metadata.sha256,
                    maximum_bytes=_MAXIMUM_SIDECAR_SHARD_BYTES,
                )
                with np.load(BytesIO(payload), allow_pickle=False) as archive:
                    if "global_indices" not in archive.files:
                        raise ContractError("CALVIN physical shard has no global index")
                    shard_indices = archive["global_indices"]
                if not np.array_equal(shard_indices, expected_shard_indices):
                    raise ContractError("CALVIN physical shard coverage differs from manifest")
            cursor = stop
        if cursor != len(expected_indices):
            raise ContractError("CALVIN physical shard metadata coverage is incomplete")
        self.index = index
        self.manifest_sha256 = manifest_sha256
        self.coverage = coverage
        self.coverage_validation = (
            "eager-all-shard-content-hash/v1"
            if eager_coverage_scan
            else "manifest-bound-lazy-consumed-shard-content-hash/v1"
        )
        self.geometry_contract = contract
        self.shards = shards
        self._starts = tuple(item.first_global_index for item in shards)
        self._cache_capacity = cache_shards
        self._cache: OrderedDict[int, _LoadedPhysicalShard] = OrderedDict()

    def _camera_array_names(self, camera_name: str) -> tuple[str, ...]:
        names = (
            f"{camera_name}_source_rgb_sha256",
            f"{camera_name}_source_depth_sha256",
            f"{camera_name}_owner_index",
            f"{camera_name}_rgb_mae",
            f"{camera_name}_depth_mae_m",
            f"{camera_name}_depth_p95_m",
        )
        if self.coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            names += (
                f"{camera_name}_owner_supervised",
                f"{camera_name}_depth_consistent_fraction",
            )
        return names

    def _shard_index(self, global_index: int) -> int:
        position = bisect_right(self._starts, global_index) - 1
        if position < 0 or global_index > self.shards[position].last_global_index:
            raise KeyError(f"CALVIN physical sidecar does not cover frame {global_index}")
        return position

    def _load_shard(self, shard_index: int) -> _LoadedPhysicalShard:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        metadata = self.shards[shard_index]
        expected_arrays = {
            "global_indices",
            "source_state_sha256",
            "frame_offsets",
            "identity_keys",
            "geometry",
            "geometry_variance",
            "geometry_supervised",
            *(
                name
                for camera in ("static", "gripper")
                for name in self._camera_array_names(camera)
            ),
        }
        payload = read_sha256_verified_file_beneath(
            self.root,
            metadata.path,
            expected_sha256=metadata.sha256,
            maximum_bytes=_MAXIMUM_SIDECAR_SHARD_BYTES,
        )
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                raise ContractError("CALVIN physical shard arrays differ from schema")
            arrays = {name: archive[name].copy() for name in archive.files}
        frames = metadata.frame_count
        records = metadata.object_record_count
        global_indices = arrays["global_indices"]
        state_hashes = arrays["source_state_sha256"]
        offsets = arrays["frame_offsets"]
        keys = arrays["identity_keys"]
        geometry = arrays["geometry"]
        variance = arrays["geometry_variance"]
        supervised = arrays["geometry_supervised"]
        expected_geometry = (records, self.geometry_contract.dimension)
        if (
            global_indices.dtype != np.int64
            or global_indices.shape != (frames,)
            or not np.issubdtype(state_hashes.dtype, np.str_)
            or state_hashes.shape != (frames,)
            or offsets.dtype != np.int64
            or offsets.shape != (frames + 1,)
            or not np.issubdtype(keys.dtype, np.str_)
            or keys.shape != (records,)
            or geometry.dtype != np.float32
            or geometry.shape != expected_geometry
            or variance.dtype != np.float32
            or variance.shape != expected_geometry
            or supervised.dtype != np.bool_
            or supervised.shape != expected_geometry
        ):
            raise ContractError("CALVIN physical geometry arrays have invalid shapes or dtypes")
        if (
            int(global_indices[0]) != metadata.first_global_index
            or int(global_indices[-1]) != metadata.last_global_index
            or (
                self.coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
                and not np.array_equal(
                    global_indices,
                    np.arange(
                        metadata.first_global_index,
                        metadata.last_global_index + 1,
                        dtype=np.int64,
                    ),
                )
            )
        ):
            raise ContractError("CALVIN physical shard content differs from metadata coverage")
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in state_hashes.tolist()
        ):
            raise ContractError("CALVIN physical source-state hash is invalid")
        if (
            np.any(global_indices[1:] <= global_indices[:-1])
            or int(offsets[0]) != 0
            or int(offsets[-1]) != records
            or np.any(offsets[1:] <= offsets[:-1])
            or not np.isfinite(geometry).all()
            or not np.isfinite(variance).all()
            or (variance < 0.0).any()
            or (geometry[~supervised] != 0.0).any()
            or (variance[~supervised] != 0.0).any()
            or not supervised.any(axis=-1).all()
        ):
            raise ContractError("CALVIN physical geometry arrays contain invalid values")
        camera_arrays: dict[str, dict[str, np.ndarray]] = {}
        for spec in CALVIN_CAMERA_SPECS:
            camera = str(spec["camera_name"])
            values = {
                name.removeprefix(f"{camera}_"): arrays[name]
                for name in self._camera_array_names(camera)
            }
            owner = values["owner_index"]
            expected_owner_shape = (frames, int(spec["height"]), int(spec["width"]))
            if owner.dtype != np.uint8 or owner.shape != expected_owner_shape:
                raise ContractError("CALVIN physical owner raster shape or dtype is invalid")
            if self.coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
                owner_supervised = values["owner_supervised"]
                consistent_fraction = values["depth_consistent_fraction"]
                if (
                    owner_supervised.dtype != np.bool_
                    or owner_supervised.shape != expected_owner_shape
                    or consistent_fraction.dtype != np.float32
                    or consistent_fraction.shape != (frames,)
                    or not np.isfinite(consistent_fraction).all()
                    or (consistent_fraction < 0.0).any()
                    or (consistent_fraction > 1.0).any()
                    or not np.allclose(
                        consistent_fraction,
                        owner_supervised.mean(axis=(1, 2)),
                        atol=1e-6,
                        rtol=1e-6,
                    )
                ):
                    raise ContractError("CALVIN physical depth-consistent supervision is invalid")
            else:
                values["owner_supervised"] = np.ones(owner.shape, dtype=np.bool_)
                values["depth_consistent_fraction"] = np.ones(frames, dtype=np.float32)
            for hash_name in ("source_rgb_sha256", "source_depth_sha256"):
                hashes = values[hash_name]
                if (
                    not np.issubdtype(hashes.dtype, np.str_)
                    or hashes.shape != (frames,)
                    or any(
                        len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                        for value in hashes.tolist()
                    )
                ):
                    raise ContractError("CALVIN physical source image/depth hash is invalid")
            for metric_name in ("rgb_mae", "depth_mae_m", "depth_p95_m"):
                metric = values[metric_name]
                if (
                    metric.dtype != np.float32
                    or metric.shape != (frames,)
                    or not np.isfinite(metric).all()
                    or (metric < 0.0).any()
                ):
                    raise ContractError("CALVIN physical calibration metric is invalid")
            camera_arrays[camera] = values
        for frame_index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            frame_keys = keys[int(start) : int(stop)].tolist()
            if len(set(frame_keys)) != len(frame_keys):
                raise ContractError("CALVIN physical keys must be unique within a frame")
            maximum_owner = len(frame_keys)
            if any(
                int(camera_arrays[camera]["owner_index"][frame_index].max(initial=0))
                > maximum_owner
                for camera in camera_arrays
            ):
                raise ContractError("CALVIN owner raster references an unknown physical object")
        loaded = _LoadedPhysicalShard(
            global_indices=global_indices,
            source_state_sha256=state_hashes,
            frame_offsets=offsets,
            identity_keys=keys,
            geometry=geometry,
            geometry_variance=variance,
            geometry_supervised=supervised,
            camera_arrays=camera_arrays,
            frame_by_global_index={
                int(value): row for row, value in enumerate(global_indices.tolist())
            },
        )
        self._cache[shard_index] = loaded
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self._cache_capacity:
            self._cache.popitem(last=False)
        return loaded

    def _frame(self, global_index: int) -> CalvinPhysicalSupervisionFrame:
        if not isinstance(global_index, int) or isinstance(global_index, bool) or global_index < 0:
            raise ContractError("CALVIN physical source index must be non-negative")
        shard = self._load_shard(self._shard_index(global_index))
        row = shard.frame_by_global_index.get(global_index)
        if row is None:
            raise KeyError(f"CALVIN physical sidecar does not cover frame {global_index}")
        start = int(shard.frame_offsets[row])
        stop = int(shard.frame_offsets[row + 1])
        cameras = []
        for spec in CALVIN_CAMERA_SPECS:
            name = str(spec["camera_name"])
            arrays = shard.camera_arrays[name]
            owner = arrays["owner_index"][row].copy()
            owner.setflags(write=False)
            owner_supervised = arrays["owner_supervised"][row].copy()
            owner_supervised.setflags(write=False)
            cameras.append(
                CalvinVisibleOwnerRaster(
                    camera_name=name,
                    host_image_key=str(spec["host_image_key"]),
                    owner_index=owner,
                    owner_supervised=owner_supervised,
                    source_rgb_sha256=str(arrays["source_rgb_sha256"][row]),
                    source_depth_sha256=str(arrays["source_depth_sha256"][row]),
                    rgb_mae=float(arrays["rgb_mae"][row]),
                    depth_mae_m=float(arrays["depth_mae_m"][row]),
                    depth_p95_m=float(arrays["depth_p95_m"][row]),
                    depth_consistent_fraction=float(arrays["depth_consistent_fraction"][row]),
                )
            )
        return CalvinPhysicalSupervisionFrame(
            identity_keys=tuple(str(key) for key in shard.identity_keys[start:stop].tolist()),
            geometry=torch.from_numpy(shard.geometry[start:stop].copy()),
            geometry_variance=torch.from_numpy(shard.geometry_variance[start:stop].copy()),
            geometry_supervised=torch.from_numpy(shard.geometry_supervised[start:stop].copy()),
            geometry_contract=self.geometry_contract,
            cameras=tuple(cameras),
        )

    def source_frame(self, global_index: int) -> CalvinPhysicalSupervisionFrame:
        """Read a task-independent frame from an all-source supervision sidecar."""

        if self.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            raise ContractError("CALVIN physical sidecar is not declared for all source frames")
        try:
            return self._frame(global_index)
        except KeyError as error:
            raise ContractError(
                "CALVIN physical request lies outside its source episodes"
            ) from error

    def source_state_sha256(self, global_index: int) -> str:
        """Return the privileged source-state digest for offline audit binding."""

        if not isinstance(global_index, int) or isinstance(global_index, bool) or global_index < 0:
            raise ContractError("CALVIN physical source index must be non-negative")
        shard = self._load_shard(self._shard_index(global_index))
        row = shard.frame_by_global_index.get(global_index)
        if row is None:
            raise ContractError("CALVIN physical request lies outside its source episodes")
        return str(shard.source_state_sha256[row])

    def __call__(self, segment_index: int, global_index: int) -> CalvinPhysicalSupervisionFrame:
        if (
            not isinstance(segment_index, int)
            or isinstance(segment_index, bool)
            or not 0 <= segment_index < len(self.index.segments)
        ):
            raise ContractError("unknown CALVIN language segment")
        segment = self.index.segments[segment_index]
        if not segment.start <= global_index <= segment.end:
            raise ContractError("CALVIN physical request lies outside its language segment")
        return self._frame(global_index)

    def geometry_frame(self, segment_index: int, global_index: int) -> PhysicalObjectGeometryFrame:
        return self(segment_index, global_index).geometry_frame()

    def clear_cache(self) -> None:
        self._cache.clear()
