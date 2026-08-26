"""Offline CALVIN simulator extraction for task-independent object geometry."""

from __future__ import annotations

import io
import subprocess
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    calvin_source_state_sha256,
)
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.dataset_manifest import DatasetFileManifest, read_verified_dataset_file


@dataclass(frozen=True, slots=True)
class CalvinObjectLink:
    identity_key: str
    body_id: int
    link_index: int


@dataclass(frozen=True, slots=True)
class CalvinSceneRange:
    scene: str
    first_global_index: int
    last_global_index: int

    def contains(self, global_index: int) -> bool:
        return self.first_global_index <= global_index <= self.last_global_index


@dataclass(frozen=True, slots=True)
class CalvinRenderedCameraOwnership:
    """One loss-only camera render with exclusive physical-instance owners."""

    camera_name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    owner_index: np.ndarray

    def __post_init__(self) -> None:
        expected_shapes = {
            "static": (200, 200),
            "gripper": (84, 84),
        }
        expected = expected_shapes.get(self.camera_name)
        if expected is None:
            raise ContractError("CALVIN ownership render uses an unknown camera")
        if self.rgb.shape != (*expected, 3) or self.rgb.dtype != np.uint8:
            raise ContractError("CALVIN ownership RGB shape or dtype is invalid")
        if (
            self.depth_m.shape != expected
            or not np.issubdtype(self.depth_m.dtype, np.floating)
            or not np.isfinite(self.depth_m).all()
            or (self.depth_m <= 0.0).any()
        ):
            raise ContractError("CALVIN ownership depth is invalid")
        if self.owner_index.shape != expected or self.owner_index.dtype != np.uint8:
            raise ContractError("CALVIN ownership raster shape or dtype is invalid")


@dataclass(frozen=True, slots=True)
class CalvinObjectRemovalCameraPair:
    """One exact-renderer factual/removed diagnostic pair."""

    camera_name: str
    factual: CalvinRenderedCameraOwnership
    removed: CalvinRenderedCameraOwnership
    target_pixel_count: int
    changed_pixel_count: int
    factual_rgb_sha256: str
    removed_rgb_sha256: str
    factual_depth_sha256: str
    removed_depth_sha256: str
    factual_owner_sha256: str
    removed_owner_sha256: str

    def __post_init__(self) -> None:
        if (
            self.factual.camera_name != self.camera_name
            or self.removed.camera_name != self.camera_name
        ):
            raise ContractError("CALVIN removal pair camera names disagree")
        pixel_count = self.factual.owner_index.size
        if (
            isinstance(self.target_pixel_count, bool)
            or isinstance(self.changed_pixel_count, bool)
            or not 0 <= self.target_pixel_count <= pixel_count
            or not 0 <= self.changed_pixel_count <= pixel_count
        ):
            raise ContractError("CALVIN removal pair pixel counts are invalid")
        for digest in (
            self.factual_rgb_sha256,
            self.removed_rgb_sha256,
            self.factual_depth_sha256,
            self.removed_depth_sha256,
            self.factual_owner_sha256,
            self.removed_owner_sha256,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ContractError("CALVIN removal pair RGB hash is invalid")

    def contract_dict(self) -> dict[str, object]:
        return {
            "camera_name": self.camera_name,
            "changed_pixel_count": self.changed_pixel_count,
            "factual_depth_sha256": self.factual_depth_sha256,
            "factual_owner_sha256": self.factual_owner_sha256,
            "factual_rgb_sha256": self.factual_rgb_sha256,
            "removed_depth_sha256": self.removed_depth_sha256,
            "removed_owner_sha256": self.removed_owner_sha256,
            "removed_rgb_sha256": self.removed_rgb_sha256,
            "target_pixel_count": self.target_pixel_count,
        }


@dataclass(frozen=True, slots=True)
class CalvinObjectRemovalPair:
    """Task-independent same-state object-removal diagnostic contract."""

    source_global_index: int
    source_state_sha256: str
    target_identity_key: str
    target_owner_index: int
    identity_keys: tuple[str, ...]
    cameras: tuple[CalvinObjectRemovalCameraPair, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_global_index, bool)
            or not isinstance(self.source_global_index, int)
            or self.source_global_index < 0
        ):
            raise ContractError("CALVIN removal pair source index is invalid")
        if len(self.source_state_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_state_sha256
        ):
            raise ContractError("CALVIN removal pair source-state hash is invalid")
        if (
            not self.identity_keys
            or len(set(self.identity_keys)) != len(self.identity_keys)
            or self.target_identity_key not in self.identity_keys
            or self.target_owner_index != self.identity_keys.index(self.target_identity_key) + 1
        ):
            raise ContractError("CALVIN removal pair physical identity contract is invalid")
        if tuple(camera.camera_name for camera in self.cameras) != ("static", "gripper"):
            raise ContractError("CALVIN removal pair requires both ordered cameras")
        if not any(camera.target_pixel_count > 0 for camera in self.cameras):
            raise ContractError("CALVIN removal pair target is invisible in both cameras")

    def contract_dict(self) -> dict[str, object]:
        return {
            "cameras": [camera.contract_dict() for camera in self.cameras],
            "identity_keys": list(self.identity_keys),
            "method": "same-restored-state.exact-link-alpha-removal.v1",
            "model_input_contains_identity_or_owner": False,
            "source_global_index": self.source_global_index,
            "source_state_sha256": self.source_state_sha256,
            "target_identity_key": self.target_identity_key,
            "target_owner_index": self.target_owner_index,
        }


def load_calvin_scene_ranges(
    split_root: Path,
    *,
    dataset_manifest: DatasetFileManifest,
) -> tuple[CalvinSceneRange, ...]:
    """Load the official frame-to-scene map used by ABC/ABCD datasets."""

    root = Path(split_root).resolve()
    path = root / "scene_info.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(dataset_manifest, DatasetFileManifest):
        raise TypeError("CALVIN scene ranges require a DatasetFileManifest")
    source = io.BytesIO(
        read_verified_dataset_file(
            dataset_manifest,
            root,
            "scene_info.npy",
            maximum_bytes=16 * 1024 * 1024,
        )
    )
    raw = np.load(source, allow_pickle=True)
    if raw.shape != ():
        raise ContractError("CALVIN scene_info must be one scalar mapping")
    payload = raw.item()
    if not isinstance(payload, dict) or not payload:
        raise ContractError("CALVIN scene_info must contain a nonempty mapping")
    ranges = []
    for scene, interval in payload.items():
        values = np.asarray(interval)
        if (
            not isinstance(scene, str)
            or scene not in {f"calvin_scene_{name}" for name in "ABCD"}
            or values.shape != (2,)
            or values.dtype == np.bool_
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ContractError("CALVIN scene_info entry is malformed")
        first, last = int(values[0]), int(values[1])
        if first < 0 or last < first:
            raise ContractError("CALVIN scene_info range is invalid")
        ranges.append(CalvinSceneRange(scene, first, last))
    ordered = tuple(sorted(ranges, key=lambda item: item.first_global_index))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.first_global_index <= previous.last_global_index:
            raise ContractError("CALVIN scene_info ranges overlap")
    return ordered


def scene_for_global_index(
    ranges: tuple[CalvinSceneRange, ...],
    global_index: int,
) -> str:
    if (
        isinstance(global_index, bool | np.bool_)
        or not isinstance(global_index, Integral)
        or global_index < 0
    ):
        raise ContractError("CALVIN scene lookup index must be non-negative")
    matches = tuple(item.scene for item in ranges if item.contains(int(global_index)))
    if len(matches) != 1:
        raise ContractError(f"CALVIN frame {global_index} has no unique scene assignment")
    return matches[0]


def _git_checkout_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = f": {error.stderr.strip()}"
        raise ContractError(f"cannot verify CALVIN checkout {checkout}{detail}") from error
    return completed.stdout.strip()


def validate_calvin_environment_checkout(
    calvin_env_root: Path,
    *,
    expected_calvin_commit: str = CALVIN_SOURCE_COMMIT,
    expected_calvin_env_commit: str = CALVIN_ENV_SOURCE_COMMIT,
) -> tuple[str, str]:
    """Require the exact clean CALVIN parent and environment submodule."""

    root = Path(calvin_env_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    parent = root.parent.resolve()
    identities = (
        ("CALVIN environment", root, expected_calvin_env_commit),
        ("CALVIN", parent, expected_calvin_commit),
    )
    actual_commits: dict[str, str] = {}
    for label, checkout, expected_commit in identities:
        top_level = Path(_git_checkout_output(checkout, "rev-parse", "--show-toplevel")).resolve()
        if top_level != checkout:
            raise ContractError(
                f"{label} checkout root differs from the required path: {top_level} != {checkout}"
            )
        actual_commit = _git_checkout_output(checkout, "rev-parse", "HEAD")
        if actual_commit != expected_commit:
            raise ContractError(
                f"{label} checkout commit differs: {actual_commit} != {expected_commit}"
            )
        dirty = _git_checkout_output(
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if dirty:
            raise ContractError(f"{label} checkout is dirty: {dirty}")
        actual_commits[label] = actual_commit
    return actual_commits["CALVIN"], actual_commits["CALVIN environment"]


def build_calvin_geometry_environment(
    calvin_env_root: Path,
    *,
    scene: str,
    include_cameras: bool = False,
):
    """Instantiate the pinned headless CALVIN scene lazily.

    The dependency is intentionally absent from the runtime package import
    path.  Only the offline sidecar builder calls this function.
    """

    root = Path(calvin_env_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if scene not in {f"calvin_scene_{name}" for name in "ABCD"}:
        raise ContractError("unsupported CALVIN geometry scene")
    if not isinstance(include_cameras, bool):
        raise TypeError("include_cameras must be boolean")
    validate_calvin_environment_checkout(root)
    if not (root / "conf" / "scene" / f"{scene}.yaml").is_file():
        raise FileNotFoundError(root / "conf" / "scene" / f"{scene}.yaml")
    sys.path.insert(0, str(root))
    import hydra  # type: ignore[import-untyped]
    from hydra import compose, initialize_config_dir  # type: ignore[import-untyped]

    with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
        config = compose(
            config_name="config_data_collection",
            overrides=[
                f"cameras={'static_and_gripper' if include_cameras else 'no_cameras'}",
                f"scene={scene}",
                "robot=panda_longer_finger",
                "use_vr=false",
                "env.use_egl=false",
                "env.use_scene_info=true",
            ],
        )
    return hydra.utils.instantiate(
        config.env,
        show_gui=False,
        use_vr=False,
        use_scene_info=True,
    )


def close_calvin_geometry_environment(environment: Any) -> None:
    try:
        environment.close()
    finally:
        environment.ownsPhysicsClient = False
        environment.cid = -1


def scene_object_links(scene_info: object) -> tuple[CalvinObjectLink, ...]:
    """Return the complete task-independent movable/link inventory."""

    if not isinstance(scene_info, dict):
        raise ContractError("CALVIN scene_info must be a mapping")
    movable = scene_info.get("movable_objects")
    fixed = scene_info.get("fixed_objects")
    if not isinstance(movable, dict) or not isinstance(fixed, dict):
        raise ContractError("CALVIN scene_info has no physical object inventory")
    objects: list[CalvinObjectLink] = []
    for name, raw_info in sorted(movable.items()):
        if not isinstance(name, str) or not name or not isinstance(raw_info, dict):
            raise ContractError("CALVIN movable object metadata is malformed")
        uid = raw_info.get("uid")
        if not isinstance(uid, int) or isinstance(uid, bool) or uid < 0:
            raise ContractError("CALVIN movable object uid is invalid")
        objects.append(CalvinObjectLink(f"movable/{name}", uid, -1))
    for body_name, raw_info in sorted(fixed.items()):
        if not isinstance(body_name, str) or not body_name or not isinstance(raw_info, dict):
            raise ContractError("CALVIN fixed object metadata is malformed")
        uid = raw_info.get("uid")
        links = raw_info.get("links", {})
        if (
            not isinstance(uid, int)
            or isinstance(uid, bool)
            or uid < 0
            or not isinstance(links, dict)
        ):
            raise ContractError("CALVIN fixed object metadata is malformed")
        for link_name, link_index in sorted(links.items()):
            if (
                not isinstance(link_name, str)
                or not link_name
                or not isinstance(link_index, int)
                or isinstance(link_index, bool)
            ):
                raise ContractError("CALVIN fixed link metadata is malformed")
            if link_index >= 0:
                objects.append(
                    CalvinObjectLink(
                        f"part/{body_name}/{link_name}",
                        uid,
                        link_index,
                    )
                )
    keys = tuple(item.identity_key for item in objects)
    if not keys or len(set(keys)) != len(keys):
        raise ContractError("CALVIN physical object inventory is empty or duplicated")
    return tuple(objects)


def calvin_segmentation_identity_map(scene_info: object) -> dict[int, str]:
    """Map PyBullet object/link segmentation IDs onto physical identity keys."""

    mapping: dict[int, str] = {}
    for item in scene_object_links(scene_info):
        encoded_id = item.body_id + ((item.link_index + 1) << 24)
        if encoded_id in mapping:
            raise ContractError("CALVIN physical inventory aliases one segmentation ID")
        mapping[encoded_id] = item.identity_key
    return mapping


def _calvin_camera_matrices(environment: Any, camera: Any) -> tuple[Any, Any]:
    """Reproduce the pinned CALVIN camera pose calculation without rendering."""

    if getattr(camera, "name", None) == "static":
        if not hasattr(camera, "viewMatrix") or not hasattr(camera, "projectionMatrix"):
            raise ContractError("CALVIN static camera has no calibrated matrices")
        return camera.viewMatrix, camera.projectionMatrix
    if getattr(camera, "name", None) != "gripper":
        raise ContractError("CALVIN physical supervision requires static and gripper cameras")
    required = ("robot_uid", "gripper_cam_link", "fov", "aspect", "nearval", "farval")
    if any(not hasattr(camera, name) for name in required):
        raise ContractError("CALVIN gripper camera calibration is incomplete")
    link_state = environment.p.getLinkState(
        bodyUniqueId=int(camera.robot_uid),
        linkIndex=int(camera.gripper_cam_link),
        physicsClientId=environment.cid,
    )
    if not isinstance(link_state, tuple | list) or len(link_state) < 2:
        raise ContractError("CALVIN gripper camera link state is malformed")
    camera_position = np.asarray(link_state[0], dtype=np.float64)
    camera_orientation = link_state[1]
    rotation = np.asarray(
        environment.p.getMatrixFromQuaternion(camera_orientation),
        dtype=np.float64,
    ).reshape(3, 3)
    if camera_position.shape != (3,) or not np.isfinite(rotation).all():
        raise ContractError("CALVIN gripper camera transform is malformed")
    view = environment.p.computeViewMatrix(
        camera_position.tolist(),
        (camera_position + rotation[:, 1]).tolist(),
        (-rotation[:, 2]).tolist(),
    )
    projection = environment.p.computeProjectionMatrixFOV(
        fov=float(camera.fov),
        aspect=float(camera.aspect),
        nearVal=float(camera.nearval),
        farVal=float(camera.farval),
    )
    return view, projection


def render_calvin_camera_ownership(
    environment: Any,
    *,
    identity_keys: tuple[str, ...],
) -> tuple[CalvinRenderedCameraOwnership, ...]:
    """Render both official cameras into one exclusive physical-owner chart.

    Owner zero is known context. Owners ``1..K`` index ``identity_keys``. The
    chart is task-independent and comes from the same restored simulator state
    as physical geometry; it is never a runtime observation.
    """

    if not identity_keys or len(set(identity_keys)) != len(identity_keys):
        raise ContractError("CALVIN ownership requires a unique physical inventory")
    if len(identity_keys) >= np.iinfo(np.uint8).max:
        raise ContractError("CALVIN physical inventory exceeds uint8 owner capacity")
    cameras: tuple[Any, ...] = tuple(getattr(environment, "cameras", ()))
    if tuple(getattr(camera, "name", None) for camera in cameras) != (
        "static",
        "gripper",
    ):
        raise ContractError("CALVIN camera ordering differs from the pinned contract")
    info = environment.get_info()
    if not isinstance(info, dict) or "scene_info" not in info:
        raise ContractError("CALVIN simulator did not expose scene_info")
    segmentation_to_key = calvin_segmentation_identity_map(info["scene_info"])
    key_to_owner = {key: index + 1 for index, key in enumerate(identity_keys)}
    if set(segmentation_to_key.values()) != set(identity_keys):
        raise ContractError("CALVIN segmentation and geometry inventories differ")

    output = []
    for camera in cameras:
        view, projection = _calvin_camera_matrices(environment, camera)
        raw = environment.p.getCameraImage(
            width=int(camera.width),
            height=int(camera.height),
            viewMatrix=view,
            projectionMatrix=projection,
            flags=environment.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=environment.cid,
        )
        if not isinstance(raw, tuple | list) or len(raw) < 5:
            raise ContractError("CALVIN camera render is malformed")
        processed: Any = camera.process_rgbd(raw, camera.nearval, camera.farval)
        if not isinstance(processed, tuple | list) or len(processed) != 2:
            raise ContractError("CALVIN camera RGB-D processing result is malformed")
        rgb, depth = processed
        segmentation = np.asarray(raw[4]).reshape(int(camera.height), int(camera.width))
        owner = np.zeros(segmentation.shape, dtype=np.uint8)
        for encoded_id, key in segmentation_to_key.items():
            owner[segmentation == encoded_id] = key_to_owner[key]
        output.append(
            CalvinRenderedCameraOwnership(
                camera_name=str(camera.name),
                rgb=np.asarray(rgb, dtype=np.uint8),
                depth_m=np.asarray(depth, dtype=np.float32),
                owner_index=owner,
            )
        )
    return tuple(output)


def build_calvin_object_removal_pair(
    environment: Any,
    *,
    scene_obs: np.ndarray,
    robot_obs: np.ndarray,
    source_global_index: int,
    target_identity_key: str,
) -> CalvinObjectRemovalPair:
    """Render a fail-closed same-domain factual/removed object pair.

    This function is for offline measurement diagnostics. It does not produce
    a lifecycle visibility label: alpha removal is not a transition-predictable
    physical occlusion.
    """

    if (
        isinstance(source_global_index, bool | np.bool_)
        or not isinstance(source_global_index, Integral)
        or source_global_index < 0
    ):
        raise ContractError("CALVIN removal pair source index must be non-negative")
    if not isinstance(target_identity_key, str) or not target_identity_key:
        raise ContractError("CALVIN removal pair target identity cannot be empty")
    restore_calvin_archived_state(
        environment,
        scene_obs=scene_obs,
        robot_obs=robot_obs,
    )
    info = environment.get_info()
    if not isinstance(info, dict) or "scene_info" not in info:
        raise ContractError("CALVIN simulator did not expose scene_info")
    objects = scene_object_links(info["scene_info"])
    identity_keys = tuple(item.identity_key for item in objects)
    matches = tuple(item for item in objects if item.identity_key == target_identity_key)
    if len(matches) != 1:
        raise ContractError("CALVIN removal target has no unique physical link")
    target = matches[0]
    target_owner_index = identity_keys.index(target_identity_key) + 1

    raw_shapes = environment.p.getVisualShapeData(
        target.body_id,
        physicsClientId=environment.cid,
    )
    if not isinstance(raw_shapes, tuple | list):
        raise ContractError("CALVIN removal target visual-shape data is malformed")
    target_shapes = tuple(
        shape
        for shape in raw_shapes
        if isinstance(shape, tuple | list) and len(shape) >= 8 and shape[1] == target.link_index
    )
    # PyBullet's public visual-shape tuple does not expose a stable per-link
    # shape index. Refuse ambiguity instead of hiding or restoring the wrong
    # sub-shape.
    if len(target_shapes) != 1:
        raise ContractError("CALVIN removal target must have exactly one visual shape")
    original_rgba = np.asarray(target_shapes[0][7], dtype=np.float64)
    if (
        original_rgba.shape != (4,)
        or not np.isfinite(original_rgba).all()
        or (original_rgba < 0.0).any()
        or (original_rgba > 1.0).any()
    ):
        raise ContractError("CALVIN removal target RGBA is malformed")

    factual = render_calvin_camera_ownership(environment, identity_keys=identity_keys)
    hidden_rgba = (*original_rgba[:3].tolist(), 0.0)
    try:
        environment.p.changeVisualShape(
            target.body_id,
            target.link_index,
            rgbaColor=hidden_rgba,
            physicsClientId=environment.cid,
        )
        removed = render_calvin_camera_ownership(environment, identity_keys=identity_keys)
    finally:
        environment.p.changeVisualShape(
            target.body_id,
            target.link_index,
            rgbaColor=tuple(original_rgba.tolist()),
            physicsClientId=environment.cid,
        )
    restored = render_calvin_camera_ownership(environment, identity_keys=identity_keys)

    cameras = []
    for factual_camera, removed_camera, restored_camera in zip(
        factual,
        removed,
        restored,
        strict=True,
    ):
        if not (
            factual_camera.camera_name == removed_camera.camera_name == restored_camera.camera_name
        ):
            raise ContractError("CALVIN removal pair camera ordering changed")
        if (
            not np.array_equal(factual_camera.rgb, restored_camera.rgb)
            or not np.array_equal(factual_camera.depth_m, restored_camera.depth_m)
            or not np.array_equal(factual_camera.owner_index, restored_camera.owner_index)
        ):
            raise ContractError("CALVIN removal target restoration is not render-exact")
        target_support = factual_camera.owner_index == target_owner_index
        if np.any(removed_camera.owner_index == target_owner_index):
            raise ContractError("CALVIN removed render still contains target ownership")
        changed = (
            np.any(factual_camera.rgb != removed_camera.rgb, axis=-1)
            | (factual_camera.depth_m != removed_camera.depth_m)
            | (factual_camera.owner_index != removed_camera.owner_index)
        )
        if np.any(changed & ~target_support):
            raise ContractError("CALVIN removal changed pixels outside factual target support")
        if np.any(target_support) and not np.any(changed & target_support):
            raise ContractError("CALVIN removal did not change its visible target support")
        camera_name = factual_camera.camera_name
        cameras.append(
            CalvinObjectRemovalCameraPair(
                camera_name=camera_name,
                factual=factual_camera,
                removed=removed_camera,
                target_pixel_count=int(target_support.sum()),
                changed_pixel_count=int(changed.sum()),
                factual_rgb_sha256=source_array_sha256(
                    f"{camera_name}_factual_rgb",
                    factual_camera.rgb,
                ),
                removed_rgb_sha256=source_array_sha256(
                    f"{camera_name}_removed_rgb",
                    removed_camera.rgb,
                ),
                factual_depth_sha256=source_array_sha256(
                    f"{camera_name}_factual_depth",
                    factual_camera.depth_m,
                ),
                removed_depth_sha256=source_array_sha256(
                    f"{camera_name}_removed_depth",
                    removed_camera.depth_m,
                ),
                factual_owner_sha256=source_array_sha256(
                    f"{camera_name}_factual_owner",
                    factual_camera.owner_index,
                ),
                removed_owner_sha256=source_array_sha256(
                    f"{camera_name}_removed_owner",
                    removed_camera.owner_index,
                ),
            )
        )
    return CalvinObjectRemovalPair(
        source_global_index=int(source_global_index),
        source_state_sha256=calvin_source_state_sha256(scene_obs, robot_obs),
        target_identity_key=target_identity_key,
        target_owner_index=target_owner_index,
        identity_keys=identity_keys,
        cameras=tuple(cameras),
    )


def restore_calvin_archived_state(
    environment: Any,
    *,
    scene_obs: np.ndarray,
    robot_obs: np.ndarray,
) -> None:
    """Restore one archived frame without advancing simulator time.

    ``PlayTableSimEnv.reset`` performs one physics step after resetting poses,
    while CALVIN's movable-object reset does not clear base velocity. Reusing an
    environment can therefore make extracted geometry depend on the previously
    processed frame. Offline labels instead restore scene/robot state directly,
    zero every movable velocity and refresh collision geometry without a step.
    """

    scene_obs = np.asarray(scene_obs)
    robot_obs = np.asarray(robot_obs)
    if (
        scene_obs.ndim != 1
        or robot_obs.shape != (15,)
        or not np.issubdtype(scene_obs.dtype, np.floating)
        or not np.issubdtype(robot_obs.dtype, np.floating)
        or not np.isfinite(scene_obs).all()
        or not np.isfinite(robot_obs).all()
    ):
        raise ContractError("CALVIN simulator state arrays are malformed")
    environment.scene.reset(scene_obs)
    environment.robot.reset(robot_obs)
    movable_objects = tuple(environment.scene.movable_objects)
    if not movable_objects:
        raise ContractError("CALVIN scene has no movable physical inventory")
    for item in movable_objects:
        environment.p.resetBaseVelocity(
            int(item.uid),
            linearVelocity=(0.0, 0.0, 0.0),
            angularVelocity=(0.0, 0.0, 0.0),
            physicsClientId=environment.cid,
        )
    environment.p.performCollisionDetection(physicsClientId=environment.cid)


def extract_robot_base_aabb_centres(
    environment: Any,
    *,
    scene_obs: np.ndarray,
    robot_obs: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Reset one archived state and return normalized physical object centres."""

    restore_calvin_archived_state(
        environment,
        scene_obs=scene_obs,
        robot_obs=robot_obs,
    )
    info = environment.get_info()
    if not isinstance(info, dict) or "scene_info" not in info:
        raise ContractError("CALVIN simulator did not expose scene_info")
    objects = scene_object_links(info["scene_info"])
    robot_uid = environment.robot.robot_uid
    base_position, base_orientation = environment.p.getBasePositionAndOrientation(
        int(robot_uid),
        physicsClientId=environment.cid,
    )
    inverse_position, inverse_orientation = environment.p.invertTransform(
        base_position,
        base_orientation,
    )
    rows = []
    for item in objects:
        bounds = environment.p.getAABB(
            item.body_id,
            item.link_index,
            physicsClientId=environment.cid,
        )
        if bounds is None or len(bounds) != 2:
            raise ContractError(f"CALVIN object has no AABB: {item.identity_key}")
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(upper < lower)
        ):
            raise ContractError(f"CALVIN object AABB is malformed: {item.identity_key}")
        world_center = tuple(((lower + upper) * 0.5).tolist())
        local_center, _ = environment.p.multiplyTransforms(
            inverse_position,
            inverse_orientation,
            world_center,
            (0.0, 0.0, 0.0, 1.0),
        )
        rows.append(CALVIN_OBJECT_GEOMETRY_CONTRACT.normalize_values(tuple(local_center)))
    geometry = np.asarray(rows, dtype=np.float32)
    if geometry.shape != (len(objects), CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension) or (
        not np.isfinite(geometry).all()
    ):
        raise ContractError("CALVIN normalized object geometry is malformed")
    return tuple(item.identity_key for item in objects), geometry
