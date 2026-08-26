"""Content-addressed CALVIN RGB/owner clips for VidEoMT adaptation.

This loader deliberately binds only the sensor arrays used by the anchor
experiment.  It does not claim that the current full CALVIN tree is bytewise
identical to the older debug archive from which the physical sidecar was made.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CALIBRATION_LIMITS,
    CALVIN_CAMERA_SPECS,
    CALVIN_OWNER_CONTRACT,
    CALVIN_PHYSICAL_SUPERVISION_SCHEMA,
    CalvinPhysicalSupervisionShard,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath
from picf_next.geometry import PhysicalGeometryContract

_MAXIMUM_MANIFEST_BYTES = 16 * 1024 * 1024
_MAXIMUM_SHARD_BYTES = 512 * 1024 * 1024
_MAXIMUM_SOURCE_NPZ_BYTES = 64 * 1024 * 1024
_EXPECTED_SOURCE_FIELDS = [
    "depth_gripper",
    "depth_static",
    "rgb_gripper",
    "rgb_static",
    "robot_obs",
    "scene_info",
    "scene_obs",
]


@dataclass(frozen=True, slots=True)
class CalvinTemporalComponent:
    """A maximal connected component of overlapping language intervals."""

    index: int
    start: int
    end: int
    source_segment_indices: tuple[int, ...]
    split: Literal["train", "heldout"]

    def __post_init__(self) -> None:
        if self.index < 0 or self.start < 0 or self.end < self.start:
            raise ContractError("CALVIN temporal component bounds are invalid")
        if not self.source_segment_indices:
            raise ContractError("CALVIN temporal component has no source segment")


@dataclass(frozen=True, slots=True)
class CalvinVidEoMTSplitPlan:
    """Task-blind, segment-disjoint local split used only for a short gate."""

    golden_manifest_sha256: str
    sidecar_manifest_sha256: str
    components: tuple[CalvinTemporalComponent, ...]
    clip_length: int
    train_windows: tuple[tuple[int, ...], ...]
    heldout_windows: tuple[tuple[int, ...], ...]
    episode_disjoint: bool = False

    def windows(self, split: Literal["train", "heldout"]) -> tuple[tuple[int, ...], ...]:
        if split == "train":
            return self.train_windows
        if split == "heldout":
            return self.heldout_windows
        raise ValueError(f"unsupported CALVIN VidEoMT split: {split!r}")


@dataclass(frozen=True, slots=True)
class HashBoundCalvinFrame:
    global_index: int
    rgb_static: np.ndarray
    supervision: CalvinPhysicalSupervisionFrame


@dataclass(frozen=True, slots=True)
class HashBoundCalvinClip:
    global_indices: tuple[int, ...]
    rgb_static: tuple[np.ndarray, ...]
    supervision: tuple[CalvinPhysicalSupervisionFrame, ...]


@dataclass(frozen=True, slots=True)
class _PhysicalFrameRecord:
    identity_keys: tuple[str, ...]
    geometry: np.ndarray
    geometry_variance: np.ndarray
    geometry_supervised: np.ndarray
    camera_values: dict[str, dict[str, object]]


def _sha256_json_file(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file() or path.stat().st_size > _MAXIMUM_MANIFEST_BYTES:
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"invalid JSON manifest: {path}") from error
    if not isinstance(parsed, dict):
        raise ContractError(f"manifest is not a mapping: {path}")
    return parsed, hashlib.sha256(payload).hexdigest()


class HashBoundCalvinFrameStore:
    """Fail-closed sidecar reader bound to source RGB array digests."""

    def __init__(
        self,
        *,
        source_split_root: Path,
        sidecar_root: Path,
        source_overlay_root: Path | None = None,
        rgb_cache_frames: int = 128,
    ) -> None:
        if (
            isinstance(rgb_cache_frames, bool)
            or not isinstance(rgb_cache_frames, int)
            or rgb_cache_frames <= 0
        ):
            raise ValueError("rgb_cache_frames must be a positive integer")
        self.source_split_root = Path(source_split_root).expanduser().resolve()
        self.sidecar_root = Path(sidecar_root).expanduser().resolve()
        self.source_overlay_root = (
            None
            if source_overlay_root is None
            else Path(source_overlay_root).expanduser().resolve()
        )
        if not self.source_split_root.is_dir():
            raise FileNotFoundError(self.source_split_root)
        if not self.sidecar_root.is_dir():
            raise FileNotFoundError(self.sidecar_root)
        if self.source_overlay_root is not None and not self.source_overlay_root.is_dir():
            raise FileNotFoundError(self.source_overlay_root)

        manifest, self.manifest_sha256 = _sha256_json_file(self.sidecar_root / "manifest.json")
        self._validate_manifest_header(manifest)
        raw_shards = manifest["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ContractError("CALVIN sensor sidecar requires nonempty shards")
        self.shards = tuple(CalvinPhysicalSupervisionShard.from_dict(value) for value in raw_shards)
        if self.shards != tuple(sorted(self.shards, key=lambda item: item.first_global_index)):
            raise ContractError("CALVIN sensor sidecar shards are not sorted")
        for previous, current in zip(self.shards, self.shards[1:], strict=False):
            if current.first_global_index <= previous.last_global_index:
                raise ContractError("CALVIN sensor sidecar shard ranges overlap")
        if sum(value.frame_count for value in self.shards) != manifest["frame_count"]:
            raise ContractError("CALVIN sensor sidecar frame count disagrees with shards")
        if (
            sum(value.object_record_count for value in self.shards)
            != manifest["object_record_count"]
        ):
            raise ContractError("CALVIN sensor sidecar object count disagrees with shards")

        records: dict[int, _PhysicalFrameRecord] = {}
        ordered_indices: list[int] = []
        for metadata in self.shards:
            shard_records = self._load_and_validate_shard(metadata)
            for global_index, record in shard_records:
                if global_index in records:
                    raise ContractError("CALVIN sensor sidecar repeats a global index")
                records[global_index] = record
                ordered_indices.append(global_index)
        indices = np.asarray(ordered_indices, dtype=np.int64)
        if (
            indices.shape != (manifest["frame_count"],)
            or np.any(indices[1:] <= indices[:-1])
            or hashlib.sha256(indices.tobytes(order="C")).hexdigest()
            != manifest["global_indices_sha256"]
        ):
            raise ContractError("CALVIN sensor sidecar global-index receipt is invalid")
        self.global_indices = tuple(int(value) for value in indices.tolist())
        self._records = records
        self._rgb_cache_capacity = rgb_cache_frames
        self._rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._resolved_source_paths: dict[int, tuple[str, Path]] = {}

    @staticmethod
    def _validate_manifest_header(manifest: dict[str, object]) -> None:
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
            "calibration_limits",
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
            raise ContractError("CALVIN sensor sidecar manifest fields differ from v2 schema")
        if manifest["schema"] != CALVIN_PHYSICAL_SUPERVISION_SCHEMA:
            raise ContractError("CALVIN sensor sidecar is not language-frame v2")
        if manifest["split_name"] != "training":
            raise ContractError("CALVIN sensor sidecar must identify the training split")
        if manifest["runtime_input"] is not False or manifest["task_conditioned"] is not False:
            raise ContractError("CALVIN sensor supervision must remain loss-only and task-blind")
        if (
            manifest["calvin_commit"] != CALVIN_SOURCE_COMMIT
            or manifest["calvin_env_commit"] != CALVIN_ENV_SOURCE_COMMIT
            or manifest["state_restoration"] != CALVIN_STATE_RESTORATION
            or manifest["owner_contract"] != CALVIN_OWNER_CONTRACT
        ):
            raise ContractError("CALVIN sensor sidecar source contract drifted")
        if manifest["camera_specs"] != [dict(value) for value in CALVIN_CAMERA_SPECS]:
            raise ContractError("CALVIN sensor sidecar camera contract drifted")
        if manifest["calibration_limits"] != CALVIN_CALIBRATION_LIMITS:
            raise ContractError("CALVIN sensor sidecar calibration contract drifted")
        if manifest["source_fields"] != _EXPECTED_SOURCE_FIELDS:
            raise ContractError("CALVIN sensor sidecar source fields drifted")
        contract = PhysicalGeometryContract.from_dict(manifest["geometry_contract"])
        if (
            contract != CALVIN_OBJECT_GEOMETRY_CONTRACT
            or manifest["geometry_contract_sha256"] != contract.fingerprint
        ):
            raise ContractError("CALVIN sensor sidecar geometry contract drifted")
        for name in ("frame_count", "object_record_count"):
            value = manifest[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(f"CALVIN sensor sidecar {name} is invalid")
        digest = manifest["global_indices_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError("CALVIN sensor sidecar global index digest is invalid")

    @staticmethod
    def _expected_shard_arrays() -> set[str]:
        camera_fields = (
            "source_rgb_sha256",
            "source_depth_sha256",
            "owner_index",
            "rgb_mae",
            "depth_mae_m",
            "depth_p95_m",
        )
        return {
            "global_indices",
            "source_state_sha256",
            "frame_offsets",
            "identity_keys",
            "geometry",
            "geometry_variance",
            "geometry_supervised",
            *(f"{camera}_{field}" for camera in ("static", "gripper") for field in camera_fields),
        }

    def _load_and_validate_shard(
        self,
        metadata: CalvinPhysicalSupervisionShard,
    ) -> list[tuple[int, _PhysicalFrameRecord]]:
        payload = read_sha256_verified_file_beneath(
            self.sidecar_root,
            metadata.path,
            expected_sha256=metadata.sha256,
            maximum_bytes=_MAXIMUM_SHARD_BYTES,
        )
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != self._expected_shard_arrays():
                raise ContractError("CALVIN sensor shard arrays differ from v2 schema")
            arrays = {name: archive[name].copy() for name in archive.files}

        frames = metadata.frame_count
        records = metadata.object_record_count
        indices = arrays["global_indices"]
        state_hashes = arrays["source_state_sha256"]
        offsets = arrays["frame_offsets"]
        keys = arrays["identity_keys"]
        geometry = arrays["geometry"]
        variance = arrays["geometry_variance"]
        supervised = arrays["geometry_supervised"]
        if (
            indices.dtype != np.int64
            or indices.shape != (frames,)
            or int(indices[0]) != metadata.first_global_index
            or int(indices[-1]) != metadata.last_global_index
            or np.any(indices[1:] <= indices[:-1])
            or not np.issubdtype(state_hashes.dtype, np.str_)
            or state_hashes.shape != (frames,)
            or offsets.dtype != np.int64
            or offsets.shape != (frames + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != records
            or np.any(offsets[1:] <= offsets[:-1])
            or not np.issubdtype(keys.dtype, np.str_)
            or keys.shape != (records,)
            or geometry.dtype != np.float32
            or geometry.shape != (records, 3)
            or variance.dtype != np.float32
            or variance.shape != (records, 3)
            or supervised.dtype != np.bool_
            or supervised.shape != (records, 3)
            or not np.isfinite(geometry).all()
            or not np.isfinite(variance).all()
            or (variance < 0).any()
            or not supervised.any(axis=-1).all()
        ):
            raise ContractError("CALVIN sensor shard geometry arrays are invalid")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in state_hashes.tolist()
        ):
            raise ContractError("CALVIN sensor shard source-state hashes are invalid")

        cameras: dict[str, dict[str, np.ndarray]] = {}
        for spec in CALVIN_CAMERA_SPECS:
            name = str(spec["camera_name"])
            values = {
                field: arrays[f"{name}_{field}"]
                for field in (
                    "source_rgb_sha256",
                    "source_depth_sha256",
                    "owner_index",
                    "rgb_mae",
                    "depth_mae_m",
                    "depth_p95_m",
                )
            }
            expected_shape = (frames, int(spec["height"]), int(spec["width"]))
            if (
                values["owner_index"].dtype != np.uint8
                or values["owner_index"].shape != expected_shape
            ):
                raise ContractError("CALVIN sensor owner raster shape or dtype is invalid")
            for field in ("source_rgb_sha256", "source_depth_sha256"):
                hashes = values[field]
                if (
                    not np.issubdtype(hashes.dtype, np.str_)
                    or hashes.shape != (frames,)
                    or any(
                        len(value) != 64
                        or any(character not in "0123456789abcdef" for character in value)
                        for value in hashes.tolist()
                    )
                ):
                    raise ContractError("CALVIN sensor image hashes are invalid")
            for field in ("rgb_mae", "depth_mae_m", "depth_p95_m"):
                values_array = values[field]
                if (
                    values_array.dtype != np.float32
                    or values_array.shape != (frames,)
                    or not np.isfinite(values_array).all()
                    or (values_array < 0).any()
                ):
                    raise ContractError("CALVIN sensor calibration metrics are invalid")
            cameras[name] = values

        result: list[tuple[int, _PhysicalFrameRecord]] = []
        for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            start_i, stop_i = int(start), int(stop)
            frame_keys = tuple(str(value) for value in keys[start_i:stop_i].tolist())
            if not frame_keys or len(set(frame_keys)) != len(frame_keys):
                raise ContractError("CALVIN sensor frame identities are empty or duplicated")
            camera_values: dict[str, dict[str, object]] = {}
            for spec in CALVIN_CAMERA_SPECS:
                name = str(spec["camera_name"])
                values = cameras[name]
                owner = values["owner_index"][row].copy()
                if int(owner.max(initial=0)) > len(frame_keys):
                    raise ContractError("CALVIN sensor owner raster references an absent identity")
                owner.setflags(write=False)
                camera_values[name] = {
                    "owner_index": owner,
                    "source_rgb_sha256": str(values["source_rgb_sha256"][row]),
                    "source_depth_sha256": str(values["source_depth_sha256"][row]),
                    "rgb_mae": float(values["rgb_mae"][row]),
                    "depth_mae_m": float(values["depth_mae_m"][row]),
                    "depth_p95_m": float(values["depth_p95_m"][row]),
                }
            result.append(
                (
                    int(indices[row]),
                    _PhysicalFrameRecord(
                        identity_keys=frame_keys,
                        geometry=geometry[start_i:stop_i].copy(),
                        geometry_variance=variance[start_i:stop_i].copy(),
                        geometry_supervised=supervised[start_i:stop_i].copy(),
                        camera_values=camera_values,
                    ),
                )
            )
        return result

    @staticmethod
    def _decode_source_rgb(path: Path, *, root: Path) -> np.ndarray:
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ContractError("CALVIN source episode escaped its split root") from error
        if not path.is_file() or path.stat().st_size > _MAXIMUM_SOURCE_NPZ_BYTES:
            raise FileNotFoundError(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "rgb_static" not in archive.files:
                    raise ContractError("CALVIN source frame has no rgb_static array")
                rgb = archive["rgb_static"].copy()
        except (OSError, ValueError) as error:
            raise ContractError(f"CALVIN source frame cannot be decoded: {path}") from error
        if rgb.dtype != np.uint8 or rgb.shape != (200, 200, 3):
            raise ContractError("CALVIN source rgb_static shape or dtype drifted")
        return rgb

    def _source_rgb(self, global_index: int) -> np.ndarray:
        cached = self._rgb_cache.get(global_index)
        if cached is not None:
            self._rgb_cache.move_to_end(global_index)
            return cached
        expected = self._records[global_index].camera_values["static"]["source_rgb_sha256"]
        candidates = [("primary", self.source_split_root)]
        if self.source_overlay_root is not None:
            candidates.append(("overlay", self.source_overlay_root))
        failures: list[str] = []
        rgb: np.ndarray | None = None
        for source_kind, root in candidates:
            path = (root / f"episode_{global_index:07d}.npz").resolve()
            try:
                candidate = self._decode_source_rgb(path, root=root)
                if source_array_sha256("rgb_static", candidate) != expected:
                    raise ContractError(
                        f"CALVIN source RGB digest mismatch at frame {global_index}"
                    )
            except (ContractError, FileNotFoundError) as error:
                failures.append(f"{source_kind}: {error}")
                continue
            rgb = candidate
            self._resolved_source_paths[global_index] = (source_kind, path)
            break
        if rgb is None:
            detail = "; ".join(failures)
            raise ContractError(
                f"no hash-valid CALVIN source RGB at frame {global_index}: {detail}"
            )
        rgb.setflags(write=False)
        self._rgb_cache[global_index] = rgb
        self._rgb_cache.move_to_end(global_index)
        while len(self._rgb_cache) > self._rgb_cache_capacity:
            self._rgb_cache.popitem(last=False)
        return rgb

    def audit_source_rgb(self) -> dict[str, object]:
        """Decode and hash-bind every covered RGB frame before model execution."""

        for global_index in self.global_indices:
            self._source_rgb(global_index)
        overlay_files = []
        for global_index, (source_kind, path) in sorted(self._resolved_source_paths.items()):
            if source_kind != "overlay":
                continue
            payload = path.read_bytes()
            overlay_files.append(
                {
                    "global_index": global_index,
                    "path": str(path),
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        return {
            "frame_count": len(self.global_indices),
            "primary_root": str(self.source_split_root),
            "overlay_root": (
                None if self.source_overlay_root is None else str(self.source_overlay_root)
            ),
            "primary_frame_count": len(self.global_indices) - len(overlay_files),
            "overlay_frame_count": len(overlay_files),
            "overlay_files": overlay_files,
        }

    def frame(self, global_index: int) -> HashBoundCalvinFrame:
        if global_index not in self._records:
            raise KeyError(f"CALVIN sensor sidecar does not cover frame {global_index}")
        record = self._records[global_index]
        cameras: list[CalvinVisibleOwnerRaster] = []
        for spec in CALVIN_CAMERA_SPECS:
            name = str(spec["camera_name"])
            values = record.camera_values[name]
            owner = np.asarray(values["owner_index"])
            owner_supervised = np.ones(owner.shape, dtype=np.bool_)
            owner_supervised.setflags(write=False)
            cameras.append(
                CalvinVisibleOwnerRaster(
                    camera_name=name,
                    host_image_key=str(spec["host_image_key"]),
                    owner_index=owner,
                    owner_supervised=owner_supervised,
                    source_rgb_sha256=str(values["source_rgb_sha256"]),
                    source_depth_sha256=str(values["source_depth_sha256"]),
                    rgb_mae=float(values["rgb_mae"]),
                    depth_mae_m=float(values["depth_mae_m"]),
                    depth_p95_m=float(values["depth_p95_m"]),
                    depth_consistent_fraction=1.0,
                )
            )
        supervision = CalvinPhysicalSupervisionFrame(
            identity_keys=record.identity_keys,
            geometry=torch.from_numpy(record.geometry.copy()),
            geometry_variance=torch.from_numpy(record.geometry_variance.copy()),
            geometry_supervised=torch.from_numpy(record.geometry_supervised.copy()),
            geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
            cameras=tuple(cameras),
        )
        return HashBoundCalvinFrame(global_index, self._source_rgb(global_index), supervision)

    def clip(self, global_indices: Sequence[int]) -> HashBoundCalvinClip:
        values = tuple(int(value) for value in global_indices)
        if not values or any(
            right != left + 1 for left, right in zip(values, values[1:], strict=False)
        ):
            raise ContractError("CALVIN VidEoMT clip must contain consecutive source frames")
        frames = tuple(self.frame(value) for value in values)
        return HashBoundCalvinClip(
            global_indices=values,
            rgb_static=tuple(frame.rgb_static for frame in frames),
            supervision=tuple(frame.supervision for frame in frames),
        )


def build_calvin_videomt_split_plan(
    *,
    golden_manifest_path: Path,
    store: HashBoundCalvinFrameStore,
    clip_length: int = 5,
    heldout_modulus: int = 3,
    heldout_remainder: int = 2,
) -> CalvinVidEoMTSplitPlan:
    """Derive a task-blind split from immutable interval boundaries.

    Overlapping language segments are first merged, so duplicated annotations
    can never place the same source frame in both train and held-out sets.  The
    component index modulo rule is fixed before any model metric is observed.
    """

    if (
        isinstance(clip_length, bool)
        or not isinstance(clip_length, int)
        or clip_length <= 0
        or isinstance(heldout_modulus, bool)
        or not isinstance(heldout_modulus, int)
        or heldout_modulus <= 1
        or not 0 <= heldout_remainder < heldout_modulus
    ):
        raise ValueError("invalid CALVIN VidEoMT split parameters")
    manifest, golden_sha256 = _sha256_json_file(Path(golden_manifest_path))
    if manifest.get("format") != "picf-next.calvin-visible-instance-golden.v1":
        raise ContractError("CALVIN golden manifest schema drifted")
    if manifest.get("runtime_input") is not False:
        raise ContractError("CALVIN golden labels must never be runtime input")
    if manifest.get("task_used_for_instance_selection") is not False:
        raise ContractError("CALVIN golden masks must remain task-independent")
    if manifest.get("failures") != []:
        raise ContractError("CALVIN golden manifest contains unresolved failures")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("CALVIN golden manifest has no records")

    segments: dict[int, list[dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("split") != "training":
            continue
        segment_index = record.get("segment_index")
        step = record.get("step")
        phase = record.get("phase")
        if (
            not isinstance(segment_index, int)
            or isinstance(segment_index, bool)
            or not isinstance(step, int)
            or isinstance(step, bool)
            or phase not in {"start", "mid", "end"}
        ):
            raise ContractError("CALVIN golden training interval record is invalid")
        segments.setdefault(segment_index, []).append(record)
    intervals: list[tuple[int, int, int]] = []
    for segment_index, values in sorted(segments.items()):
        if {str(value["phase"]) for value in values} != {"start", "mid", "end"}:
            raise ContractError("CALVIN golden segment does not have start/mid/end receipts")
        steps = [int(value["step"]) for value in values]
        intervals.append((min(steps), max(steps), segment_index))

    merged: list[tuple[int, int, list[int]]] = []
    for start, end, segment_index in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end, [segment_index]))
        else:
            previous_start, previous_end, source_segments = merged[-1]
            merged[-1] = (
                previous_start,
                max(previous_end, end),
                [*source_segments, segment_index],
            )
    components: list[CalvinTemporalComponent] = []
    train_windows: list[tuple[int, ...]] = []
    heldout_windows: list[tuple[int, ...]] = []
    covered: list[int] = []
    for index, (start, end, source_segments) in enumerate(merged):
        split: Literal["train", "heldout"] = (
            "heldout" if index % heldout_modulus == heldout_remainder else "train"
        )
        component = CalvinTemporalComponent(
            index=index,
            start=start,
            end=end,
            source_segment_indices=tuple(sorted(source_segments)),
            split=split,
        )
        components.append(component)
        values = list(range(start, end + 1))
        covered.extend(values)
        windows = [
            tuple(range(window_start, window_start + clip_length))
            for window_start in range(start, end - clip_length + 2)
        ]
        if not windows:
            raise ContractError("CALVIN temporal component is shorter than the official clip")
        (heldout_windows if split == "heldout" else train_windows).extend(windows)
    if tuple(sorted(set(covered))) != store.global_indices or len(covered) != len(set(covered)):
        raise ContractError("CALVIN golden intervals do not exactly partition the sensor sidecar")
    if not train_windows or not heldout_windows:
        raise ContractError("CALVIN VidEoMT split must contain train and held-out windows")
    return CalvinVidEoMTSplitPlan(
        golden_manifest_sha256=golden_sha256,
        sidecar_manifest_sha256=store.manifest_sha256,
        components=tuple(components),
        clip_length=clip_length,
        train_windows=tuple(train_windows),
        heldout_windows=tuple(heldout_windows),
    )
