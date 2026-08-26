"""Verified same-renderer CALVIN factual/object-removed audit inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CALVIN_OBSERVATION_SPECS,
    CalvinPICFEvidenceFrame,
)
from picf_next.data.calvin_counterfactual_plan import (
    CALVIN_COUNTERFACTUAL_PARTITIONS,
    CalvinCounterfactualPairRequest,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CALIBRATION_LIMITS,
    CALVIN_CAMERA_SPECS,
    source_array_sha256,
)

if TYPE_CHECKING:
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionFrame,
        CalvinVisibleOwnerRaster,
    )

CALVIN_OBJECT_REMOVAL_PROBE_SCHEMA = "picf-next.calvin-object-removal-probe.v1"
CALVIN_OBJECT_REMOVAL_BANK_SCHEMA = "picf-next.calvin-object-removal-bank.v2"

_OBSERVATION_KEY_BY_SOURCE_FIELD = {
    source_field: observation_key
    for source_field, observation_key, _shape, _dtype, _units in CALVIN_OBSERVATION_SPECS
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("CALVIN removal summary is not valid ASCII JSON") from error
    if not isinstance(payload, dict):
        raise ContractError("CALVIN removal summary must contain one object")
    return payload


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} is not one lowercase SHA-256")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(value).copy()
    output.setflags(write=False)
    return output


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return None
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


@dataclass(frozen=True, slots=True)
class CalvinSameRendererRemovalCamera:
    camera_name: str
    host_image_key: str
    source_rgb_field: str
    source_observation_key: str
    target_pixel_count: int
    supervised_target_pixel_count: int
    target_bbox_xyxy: tuple[int, int, int, int] | None
    occluder_bbox_xyxy: tuple[int, int, int, int] | None
    occluder_pixel_count: int
    occluded_fraction: float
    fill_rgb: None
    source_rgb_sha256: str
    occluded_rgb_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "factual_rgb_sha256": self.source_rgb_sha256,
            "host_image_key": self.host_image_key,
            "removed_fraction": self.occluded_fraction,
            "removed_pixel_count": self.occluder_pixel_count,
            "removed_rgb_sha256": self.occluded_rgb_sha256,
            "source_observation_key": self.source_observation_key,
            "source_rgb_field": self.source_rgb_field,
            "target_bbox_xyxy": (
                None if self.target_bbox_xyxy is None else list(self.target_bbox_xyxy)
            ),
            "target_pixel_count": self.target_pixel_count,
        }


@dataclass(frozen=True, slots=True)
class CalvinSameRendererRemoval:
    factual_evidence_frame: CalvinPICFEvidenceFrame
    evidence_frame: CalvinPICFEvidenceFrame
    target_identity_keys: tuple[str, ...]
    cameras: tuple[CalvinSameRendererRemovalCamera, ...]
    artifact_contract: dict[str, Any]
    factual_physical_frame: CalvinPhysicalSupervisionFrame | None = None
    removed_physical_frame: CalvinPhysicalSupervisionFrame | None = None

    def contract_dict(self) -> dict[str, Any]:
        return dict(self.artifact_contract)


@dataclass(frozen=True, slots=True)
class _RemovalRecord:
    global_index: int
    target_identity_key: str
    target_owner_index: int
    archive_path: Path
    archive_sha256: str
    pair: dict[str, Any]
    plan_request: CalvinCounterfactualPairRequest | None


class CalvinSameRendererRemovalStore:
    """Load immutable pair artifacts and reconstruct target-free evidence frames."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_id: str,
        dataset_revision: str,
    ) -> None:
        self.root = Path(root).resolve()
        summary_path = self.root / "summary.json"
        summary = _read_json(summary_path)
        schema = summary.get("schema")
        if (
            schema not in {CALVIN_OBJECT_REMOVAL_PROBE_SCHEMA, CALVIN_OBJECT_REMOVAL_BANK_SCHEMA}
            or summary.get("dataset_id") != dataset_id
            or summary.get("dataset_revision") != dataset_revision
        ):
            raise ContractError("CALVIN removal store dataset contract differs")
        pair_plan_sha256 = None
        source_sidecar_manifest_sha256 = None
        if schema == CALVIN_OBJECT_REMOVAL_BANK_SCHEMA:
            if set(summary) != {
                "dataset_id",
                "dataset_revision",
                "pair_plan",
                "pair_plan_sha256",
                "probe_count",
                "probes",
                "schema",
                "source_sidecar_manifest_sha256",
            }:
                raise ContractError("CALVIN planned removal summary fields changed")
            if not isinstance(summary["pair_plan"], str) or not summary["pair_plan"]:
                raise ContractError("CALVIN planned removal summary has no plan path")
            pair_plan_sha256 = _require_sha256(
                summary["pair_plan_sha256"],
                name="CALVIN counterfactual pair plan hash",
            )
            source_sidecar_manifest_sha256 = _require_sha256(
                summary["source_sidecar_manifest_sha256"],
                name="CALVIN counterfactual source sidecar hash",
            )
        probes = summary.get("probes")
        if not isinstance(probes, list) or not probes or summary.get("probe_count") != len(probes):
            raise ContractError("CALVIN removal store probe inventory is malformed")
        records: dict[tuple[int, str], _RemovalRecord] = {}
        for probe in probes:
            if not isinstance(probe, dict) or not isinstance(probe.get("pair"), dict):
                raise ContractError("CALVIN removal probe record is malformed")
            pair = probe["pair"]
            global_index = pair.get("source_global_index")
            target_identity_key = pair.get("target_identity_key")
            target_owner_index = pair.get("target_owner_index")
            archive_name = probe.get("array_archive")
            plan_request = None
            if schema == CALVIN_OBJECT_REMOVAL_BANK_SCHEMA:
                if set(probe) != {
                    "array_archive",
                    "array_archive_sha256",
                    "calibration",
                    "contact_sheet",
                    "contact_sheet_sha256",
                    "pair",
                    "plan_request",
                    "tasks",
                }:
                    raise ContractError("CALVIN planned removal record fields changed")
                plan_request = CalvinCounterfactualPairRequest.from_dict(probe.get("plan_request"))
                if (
                    plan_request.global_index != global_index
                    or plan_request.target_identity_key != target_identity_key
                ):
                    raise ContractError("CALVIN planned removal request differs from pair")
            if (
                not isinstance(global_index, int)
                or isinstance(global_index, bool)
                or global_index < 0
                or not isinstance(target_identity_key, str)
                or not target_identity_key
                or not isinstance(target_owner_index, int)
                or isinstance(target_owner_index, bool)
                or target_owner_index <= 0
                or not isinstance(archive_name, str)
                or Path(archive_name).name != archive_name
            ):
                raise ContractError("CALVIN removal probe identity record is malformed")
            archive_path = self.root / archive_name
            if not archive_path.is_file() or archive_path.is_symlink():
                raise FileNotFoundError(archive_path)
            archive_sha256 = _require_sha256(
                probe.get("array_archive_sha256"),
                name="CALVIN removal archive hash",
            )
            if _sha256(archive_path) != archive_sha256:
                raise ContractError("CALVIN removal archive differs from its summary hash")
            key = global_index, target_identity_key
            if key in records:
                raise ContractError("CALVIN removal store contains a duplicate probe")
            records[key] = _RemovalRecord(
                global_index=global_index,
                target_identity_key=target_identity_key,
                target_owner_index=target_owner_index,
                archive_path=archive_path,
                archive_sha256=archive_sha256,
                pair=pair,
                plan_request=plan_request,
            )
        if schema == CALVIN_OBJECT_REMOVAL_BANK_SCHEMA:
            planned = tuple(
                record.plan_request
                for record in records.values()
                if record.plan_request is not None
            )
            if len(planned) != len(records) or {request.partition for request in planned} != set(
                CALVIN_COUNTERFACTUAL_PARTITIONS
            ):
                raise ContractError("CALVIN planned removal bank partition inventory is incomplete")
            if len({request.global_index for request in planned}) != len(planned):
                raise ContractError("CALVIN planned removal bank reuses a source frame")
        self._records = records
        self.schema = str(schema)
        self.pair_plan_sha256 = pair_plan_sha256
        self.source_sidecar_manifest_sha256 = source_sidecar_manifest_sha256
        self.summary_path = summary_path
        self.summary_sha256 = _sha256(summary_path)

    @property
    def keys(self) -> tuple[tuple[int, str], ...]:
        return tuple(sorted(self._records))

    def keys_for_partition(self, partition: str) -> tuple[tuple[int, str], ...]:
        if partition not in CALVIN_COUNTERFACTUAL_PARTITIONS:
            raise ValueError(f"unknown CALVIN removal partition: {partition}")
        if self.schema != CALVIN_OBJECT_REMOVAL_BANK_SCHEMA:
            raise ContractError("manual CALVIN removal probes have no data partition")
        return tuple(
            sorted(
                key
                for key, record in self._records.items()
                if record.plan_request is not None and record.plan_request.partition == partition
            )
        )

    def _load_arrays(self, record: _RemovalRecord) -> dict[str, np.ndarray]:
        expected = {
            f"{camera}_{branch}_{field}"
            for camera in ("static", "gripper")
            for branch, fields in (
                ("archived", ("rgb",)),
                ("factual", ("depth_m", "owner", "rgb")),
                ("removed", ("depth_m", "owner", "rgb")),
            )
            for field in fields
        }
        with np.load(record.archive_path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ContractError("CALVIN removal archive field inventory changed")
            return {name: _readonly(np.asarray(archive[name])) for name in sorted(expected)}

    def __call__(
        self,
        source: CalvinPICFEvidenceFrame,
        *,
        global_index: int,
        target_identity_keys: tuple[str, ...],
        physical_frame: CalvinPhysicalSupervisionFrame | None = None,
    ) -> CalvinSameRendererRemoval | None:
        if len(target_identity_keys) != 1:
            return None
        record = self._records.get((global_index, target_identity_keys[0]))
        if record is None:
            return None
        arrays = self._load_arrays(record)
        pair_cameras = record.pair.get("cameras")
        if not isinstance(pair_cameras, list) or len(pair_cameras) != 2:
            raise ContractError("CALVIN removal pair camera contract is malformed")
        pair_camera_by_name = {
            camera.get("camera_name"): camera for camera in pair_cameras if isinstance(camera, dict)
        }
        if set(pair_camera_by_name) != {"static", "gripper"}:
            raise ContractError("CALVIN removal pair camera names changed")
        source_by_key = {item.key: item for item in source.sensor_observations}
        if len(source_by_key) != len(source.sensor_observations):
            raise ContractError("CALVIN source evidence contains duplicate sensors")
        replacements: dict[str, dict[str, np.ndarray]] = {"factual": {}, "removed": {}}
        physical_cameras: dict[str, list[CalvinVisibleOwnerRaster]] | None = None
        if physical_frame is not None:
            from picf_next.data.calvin_physical_supervision_sidecar import (
                CalvinPhysicalSupervisionFrame,
                CalvinVisibleOwnerRaster,
            )

            if not isinstance(physical_frame, CalvinPhysicalSupervisionFrame):
                raise TypeError("CALVIN removal measurement target requires a physical frame")
            physical_cameras = {"factual": [], "removed": []}
        cameras = []
        for spec in CALVIN_CAMERA_SPECS:
            camera_name = str(spec["camera_name"])
            rgb_field = str(spec["source_rgb_field"])
            depth_field = str(spec["source_depth_field"])
            rgb_key = _OBSERVATION_KEY_BY_SOURCE_FIELD[rgb_field]
            depth_key = _OBSERVATION_KEY_BY_SOURCE_FIELD[depth_field]
            source_rgb = source_by_key.get(rgb_key)
            source_depth = source_by_key.get(depth_key)
            if source_rgb is None or source_depth is None:
                raise ContractError("CALVIN removal source evidence lacks RGB-D cameras")
            height, width = int(spec["height"]), int(spec["width"])
            expected_rgb_shape = (height, width, 3)
            expected_depth_shape = (height, width)
            values = {
                name: arrays[f"{camera_name}_{name}"]
                for name in (
                    "archived_rgb",
                    "factual_depth_m",
                    "factual_owner",
                    "factual_rgb",
                    "removed_depth_m",
                    "removed_owner",
                    "removed_rgb",
                )
            }
            if (
                values["archived_rgb"].shape != expected_rgb_shape
                or values["archived_rgb"].dtype != np.uint8
                or not np.array_equal(values["archived_rgb"], source_rgb.value)
            ):
                raise ContractError("CALVIN removal archive differs from source RGB evidence")
            for branch in ("factual", "removed"):
                rgb = values[f"{branch}_rgb"]
                depth = values[f"{branch}_depth_m"]
                owner = values[f"{branch}_owner"]
                if (
                    rgb.shape != expected_rgb_shape
                    or rgb.dtype != np.uint8
                    or depth.shape != expected_depth_shape
                    or depth.dtype != np.float32
                    or not np.isfinite(depth).all()
                    or (depth <= 0.0).any()
                    or owner.shape != expected_depth_shape
                    or owner.dtype != np.uint8
                ):
                    raise ContractError("CALVIN removal RGB-D-owner arrays are malformed")
                replacements[branch][rgb_key] = rgb
                replacements[branch][depth_key] = depth
                if physical_cameras is not None:
                    rgb_error = np.abs(rgb.astype(np.float32) - source_rgb.value.astype(np.float32))
                    depth_error = np.abs(
                        depth.astype(np.float32) - source_depth.value.astype(np.float32)
                    )
                    supervised = np.ones(owner.shape, dtype=np.bool_)
                    supervised.setflags(write=False)
                    physical_cameras[branch].append(
                        CalvinVisibleOwnerRaster(
                            camera_name=camera_name,
                            host_image_key=str(spec["host_image_key"]),
                            owner_index=owner,
                            owner_supervised=supervised,
                            source_rgb_sha256=source_array_sha256(rgb_field, rgb),
                            source_depth_sha256=source_array_sha256(depth_field, depth),
                            rgb_mae=float(rgb_error.mean()),
                            depth_mae_m=float(depth_error.mean()),
                            depth_p95_m=float(np.quantile(depth_error, 0.95)),
                            depth_consistent_fraction=1.0,
                        )
                    )
            camera_contract = pair_camera_by_name[camera_name]
            factual_rgb_hash = _require_sha256(
                camera_contract.get("factual_rgb_sha256"),
                name="CALVIN factual RGB hash",
            )
            removed_rgb_hash = _require_sha256(
                camera_contract.get("removed_rgb_sha256"),
                name="CALVIN removed RGB hash",
            )
            if (
                source_array_sha256(f"{camera_name}_factual_rgb", values["factual_rgb"])
                != factual_rgb_hash
                or source_array_sha256(f"{camera_name}_removed_rgb", values["removed_rgb"])
                != removed_rgb_hash
            ):
                raise ContractError("CALVIN removal RGB hashes differ from pair contract")
            target = values["factual_owner"] == record.target_owner_index
            removed_target = values["removed_owner"] == record.target_owner_index
            changed = (
                np.any(values["factual_rgb"] != values["removed_rgb"], axis=-1)
                | (values["factual_depth_m"] != values["removed_depth_m"])
                | (values["factual_owner"] != values["removed_owner"])
            )
            if (
                np.any(removed_target)
                or np.any(changed & ~target)
                or int(target.sum()) != camera_contract.get("target_pixel_count")
                or int(changed.sum()) != camera_contract.get("changed_pixel_count")
            ):
                raise ContractError("CALVIN removal changed outside exact target support")
            rgb_delta = np.abs(
                values["factual_rgb"].astype(np.float32) - source_rgb.value.astype(np.float32)
            )
            depth_delta = np.abs(
                values["factual_depth_m"].astype(np.float32) - source_depth.value.astype(np.float32)
            )
            if (
                float(rgb_delta.mean()) > CALVIN_CALIBRATION_LIMITS["maximum_rgb_mae"]
                or float(depth_delta.mean())
                > CALVIN_CALIBRATION_LIMITS["maximum_depth_mean_absolute_error_m"]
                or float(np.quantile(depth_delta, 0.95))
                > CALVIN_CALIBRATION_LIMITS["maximum_depth_p95_absolute_error_m"]
            ):
                raise ContractError("CALVIN same-renderer factual branch is out of domain")
            target_bbox = _bbox(target)
            cameras.append(
                CalvinSameRendererRemovalCamera(
                    camera_name=camera_name,
                    host_image_key=str(spec["host_image_key"]),
                    source_rgb_field=rgb_field,
                    source_observation_key=rgb_key,
                    target_pixel_count=int(target.sum()),
                    supervised_target_pixel_count=int(target.sum()),
                    target_bbox_xyxy=target_bbox,
                    occluder_bbox_xyxy=target_bbox,
                    occluder_pixel_count=int(changed.sum()),
                    occluded_fraction=float(changed.mean()),
                    fill_rgb=None,
                    source_rgb_sha256=factual_rgb_hash,
                    occluded_rgb_sha256=removed_rgb_hash,
                )
            )

        def evidence(branch: str) -> CalvinPICFEvidenceFrame:
            observations = []
            for observation in source.sensor_observations:
                value = replacements[branch].get(observation.key, observation.value)
                observations.append(replace(observation, value=value))
            return replace(source, sensor_observations=tuple(observations))

        factual_physical = None
        removed_physical = None
        if physical_frame is not None:
            if physical_cameras is None:
                raise RuntimeError("CALVIN removal physical camera construction was skipped")
            pair_identity_keys = record.pair.get("identity_keys")
            if (
                pair_identity_keys != list(physical_frame.identity_keys)
                or record.target_identity_key not in physical_frame.identity_keys
            ):
                raise ContractError(
                    "CALVIN removal physical inventory differs from the verified pair"
                )

            def physical(branch: str) -> CalvinPhysicalSupervisionFrame:
                return CalvinPhysicalSupervisionFrame(
                    identity_keys=physical_frame.identity_keys,
                    geometry=physical_frame.geometry.detach().clone(),
                    geometry_variance=physical_frame.geometry_variance.detach().clone(),
                    geometry_supervised=physical_frame.geometry_supervised.detach().clone(),
                    geometry_contract=physical_frame.geometry_contract,
                    cameras=tuple(physical_cameras[branch]),
                )

            factual_physical = physical("factual")
            removed_physical = physical("removed")
            target_owner = record.target_owner_index
            if not any(
                np.any(camera.owner_index == target_owner) for camera in factual_physical.cameras
            ) or any(
                np.any(camera.owner_index == target_owner) for camera in removed_physical.cameras
            ):
                raise ContractError(
                    "CALVIN removal measurement targets do not remove exactly the target"
                )

        return CalvinSameRendererRemoval(
            factual_evidence_frame=evidence("factual"),
            evidence_frame=evidence("removed"),
            target_identity_keys=target_identity_keys,
            cameras=tuple(cameras),
            artifact_contract={
                "array_archive": record.archive_path.name,
                "array_archive_sha256": record.archive_sha256,
                "method": "same-restored-state.exact-link-alpha-removal.v1",
                "model_input_contains_identity_or_owner": False,
                "pair": record.pair,
                "source_summary_sha256": self.summary_sha256,
                "target_identity_keys": list(target_identity_keys),
            },
            factual_physical_frame=factual_physical,
            removed_physical_frame=removed_physical,
        )
