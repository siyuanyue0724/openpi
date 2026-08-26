"""Versioned, lossless robot transition records for heterogeneous VLA data.

The record is the boundary between an immutable dataset row and model-specific
preprocessing.  It carries task text for the vanilla VLA host, but contains no
mask, object ID, role label, scorer output, or future observation.  PICF object
discovery therefore cannot obtain a runtime shortcut through this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError

Float32Vector = NDArray[np.float32]
BoolVector = NDArray[np.bool_]

MOLMOACT2_LIBERO_DATASET_ID = "allenai/MolmoAct2-LIBERO-Dataset"
MOLMOACT2_LIBERO_REVISION = "fe3ead447f44c0ea950396360b304cc2fb6be8f8"
MOLMOACT2_LIBERO_CONTRACT = "molmoact2-libero-transition/v1"
MOLMOACT2_LIBERO_CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
MOLMOACT2_LIBERO_STATE_AXES = (
    "eef.x",
    "eef.y",
    "eef.z",
    "eef.axis_angle.x",
    "eef.axis_angle.y",
    "eef.axis_angle.z",
    "gripper.left_qpos",
    "gripper.right_qpos",
)
MOLMOACT2_LIBERO_ACTION_AXES = (
    "delta_eef.x",
    "delta_eef.y",
    "delta_eef.z",
    "delta_eef.axis_angle.x",
    "delta_eef.axis_angle.y",
    "delta_eef.axis_angle.z",
    "gripper.command",
)


def _readonly_vector(value: Any, *, width: int, name: str) -> Float32Vector:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,):
        raise ContractError(f"{name} must have shape ({width},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains NaN or infinity")
    array = array.copy()
    array.setflags(write=False)
    return array


def _readonly_validity(width: int) -> BoolVector:
    value = np.ones(width, dtype=np.bool_)
    value.setflags(write=False)
    return value


def _require_index(value: Any, *, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Integral):
        raise ContractError(f"{name} must be an integer")
    output = int(value)
    if output < 0:
        raise ContractError(f"{name} must be non-negative")
    return output


def _require_real(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise ContractError(f"{name} must be a real number")
    output = float(value)
    if not np.isfinite(output) or (output <= 0.0 if positive else output < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ContractError(f"{name} must be finite and {qualifier}")
    return output


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One losslessly retained embedded RGB payload at a source timestamp."""

    key: str
    encoded_bytes: bytes
    source_path: str
    timestamp_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ContractError("camera key must be non-empty")
        if not isinstance(self.encoded_bytes, bytes) or not self.encoded_bytes:
            raise ContractError(f"{self.key} has no embedded image bytes")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ContractError(f"{self.key} has no source path")
        _require_real(self.timestamp_s, name=f"{self.key} timestamp")


@dataclass(frozen=True, slots=True)
class ArrayObservation:
    """One immutable source array that is deploy-visible at this transition.

    Array observations cover datasets such as CALVIN whose RGB, depth and
    tactile payloads are stored directly inside a frame archive. Privileged
    simulator state and training targets must never be represented here.
    """

    key: str
    value: NDArray
    source_path: str
    timestamp_s: float
    units: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.key, self.source_path, self.units)
        ):
            raise ContractError("array observation metadata must be explicit")
        if not isinstance(self.value, np.ndarray) or self.value.dtype.hasobject:
            raise ContractError(f"{self.key} must be a non-object NumPy array")
        if not self.value.size:
            raise ContractError(f"{self.key} cannot be empty")
        if np.issubdtype(self.value.dtype, np.number) and not np.isfinite(self.value).all():
            raise ContractError(f"{self.key} contains NaN or infinity")
        if self.value.flags.writeable:
            raise ContractError(f"{self.key} must be immutable")
        _require_real(self.timestamp_s, name=f"{self.key} timestamp")


@dataclass(frozen=True, slots=True)
class RobotTransitionRecord:
    """A source-faithful transition row with explicit physical semantics.

    `action` is the demonstrator command executed between this row and the next
    row when `transition_valid` is true.  The final frame remains a valid action
    training target but is not used as a posterior transition.  Normalization,
    action chunking, image decoding, and object targets are intentionally not
    part of this record.
    """

    contract: str
    dataset_id: str
    dataset_revision: str
    embodiment: str
    control_mode: str
    control_frame: str
    state_axes: tuple[str, ...]
    state_units: tuple[str, ...]
    action_axes: tuple[str, ...]
    action_units: tuple[str, ...]
    episode_index: int
    frame_index: int
    global_index: int
    task_index: int
    task: str
    timestamp_s: float
    delta_t_s: float
    transition_valid: bool
    cameras: tuple[CameraFrame, ...]
    state: Float32Vector
    state_valid: BoolVector
    action: Float32Vector
    action_valid: BoolVector
    array_observations: tuple[ArrayObservation, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.contract, self.dataset_id, self.dataset_revision)
        ):
            raise ContractError("dataset contract, identity and revision must be explicit")
        if any(
            not isinstance(value, str) or not value
            for value in (self.embodiment, self.control_mode, self.control_frame)
        ):
            raise ContractError("physical control metadata must be explicit")
        if not all(
            isinstance(value, np.ndarray)
            for value in (self.state, self.state_valid, self.action, self.action_valid)
        ):
            raise ContractError("state, action and validity fields must be NumPy arrays")
        if self.state.ndim != 1 or self.state_valid.shape != self.state.shape:
            raise ContractError("state and state validity must be aligned vectors")
        if self.action.ndim != 1 or self.action_valid.shape != self.action.shape:
            raise ContractError("action and action validity must be aligned vectors")
        if self.state.dtype != np.float32 or self.action.dtype != np.float32:
            raise ContractError("model-boundary state and action must be float32")
        if self.state_valid.dtype != np.bool_ or self.action_valid.dtype != np.bool_:
            raise ContractError("state and action validity must be boolean")
        if not np.isfinite(self.state).all() or not np.isfinite(self.action).all():
            raise ContractError("state and action must be finite")
        if any(
            value.flags.writeable
            for value in (self.state, self.state_valid, self.action, self.action_valid)
        ):
            raise ContractError("state, action and validity fields must be immutable")
        if len(self.state_axes) != self.state.size or len(self.state_units) != self.state.size:
            raise ContractError("state semantic metadata must cover every dimension")
        if len(self.action_axes) != self.action.size or len(self.action_units) != self.action.size:
            raise ContractError("action semantic metadata must cover every dimension")
        for name, value in (
            ("episode_index", self.episode_index),
            ("frame_index", self.frame_index),
            ("global_index", self.global_index),
            ("task_index", self.task_index),
        ):
            _require_index(value, name=name)
        if not isinstance(self.task, str) or not self.task:
            raise ContractError("task identity and text must be retained for the VLA host")
        _require_real(self.timestamp_s, name="timestamp_s")
        _require_real(self.delta_t_s, name="delta_t_s", positive=True)
        if not isinstance(self.transition_valid, bool):
            raise ContractError("transition_valid must be boolean")
        camera_keys = tuple(camera.key for camera in self.cameras)
        if len(set(camera_keys)) != len(camera_keys):
            raise ContractError("camera keys must be unique")
        if any(abs(camera.timestamp_s - self.timestamp_s) > 1e-7 for camera in self.cameras):
            raise ContractError("camera and robot timestamps must be synchronous")
        array_keys = tuple(observation.key for observation in self.array_observations)
        if len(set(array_keys)) != len(array_keys):
            raise ContractError("array observation keys must be unique")
        if any(
            abs(observation.timestamp_s - self.timestamp_s) > 1e-7
            for observation in self.array_observations
        ):
            raise ContractError("array observations and robot timestamps must be synchronous")


def validate_molmoact2_libero_metadata(info: Mapping[str, Any]) -> None:
    """Reject metadata drift without trusting the release's mislabeled axes.

    The public `info.json` names the eight state values as XYZ, quaternion and
    one gripper value.  The pinned executable `LiberoProcessorStep` instead
    emits XYZ, axis-angle and two gripper qpos values.  This adapter fixes the
    executable contract explicitly and treats info metadata as dimensions only.
    """

    expected = {
        "fps": 10,
        "total_episodes": 1693,
        "total_frames": 273465,
        "total_tasks": 40,
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise ContractError(f"LIBERO metadata {key} differs from pinned release")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ContractError("LIBERO feature metadata is absent")
    state = features.get("observation.state")
    action = features.get("action")
    if not isinstance(state, Mapping) or state.get("shape") != [8]:
        raise ContractError("LIBERO state width differs from executable contract")
    if not isinstance(action, Mapping) or action.get("shape") != [7]:
        raise ContractError("LIBERO action width differs from executable contract")


def decode_molmoact2_libero_row(
    row: Mapping[str, Any],
    *,
    task: str,
    episode_length: int,
) -> RobotTransitionRecord:
    """Decode one immutable public LIBERO row without adding learned targets."""

    required = {
        *MOLMOACT2_LIBERO_CAMERA_KEYS,
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ContractError(f"LIBERO row is missing fields: {missing}")
    frame_index = _require_index(row["frame_index"], name="frame_index")
    if isinstance(episode_length, bool | np.bool_) or not isinstance(episode_length, Integral):
        raise ContractError("episode_length must be a positive integer")
    episode_length = int(episode_length)
    if episode_length <= 0 or frame_index >= episode_length:
        raise ContractError("frame index is outside the declared episode")
    timestamp = _require_real(row["timestamp"], name="timestamp")
    expected_timestamp = frame_index / 10.0
    if not np.isclose(timestamp, expected_timestamp, atol=2e-6, rtol=0.0):
        raise ContractError("row timestamp does not match the 10 Hz frame index")

    cameras: list[CameraFrame] = []
    for key in MOLMOACT2_LIBERO_CAMERA_KEYS:
        payload = row[key]
        if not isinstance(payload, Mapping):
            raise ContractError(f"{key} must be an embedded image structure")
        encoded_bytes = payload.get("bytes")
        source_path = payload.get("path")
        if not isinstance(encoded_bytes, bytes) or not encoded_bytes:
            raise ContractError(f"{key} has no embedded image bytes")
        if not isinstance(source_path, str) or not source_path:
            raise ContractError(f"{key} has no source path")
        cameras.append(
            CameraFrame(
                key=key,
                encoded_bytes=encoded_bytes,
                source_path=source_path,
                timestamp_s=timestamp,
            )
        )

    state = _readonly_vector(row["observation.state"], width=8, name="observation.state")
    action = _readonly_vector(row["action"], width=7, name="action")

    return RobotTransitionRecord(
        contract=MOLMOACT2_LIBERO_CONTRACT,
        dataset_id=MOLMOACT2_LIBERO_DATASET_ID,
        dataset_revision=MOLMOACT2_LIBERO_REVISION,
        embodiment="franka-libero-single-arm/v1",
        control_mode="delta end-effector pose",
        control_frame="LIBERO world Cartesian frame; axis-angle rotation",
        state_axes=MOLMOACT2_LIBERO_STATE_AXES,
        state_units=("m", "m", "m", "rad", "rad", "rad", "rad", "rad"),
        action_axes=MOLMOACT2_LIBERO_ACTION_AXES,
        action_units=("LIBERO normalized controller delta",) * 6 + ("binary",),
        episode_index=_require_index(row["episode_index"], name="episode_index"),
        frame_index=frame_index,
        global_index=_require_index(row["index"], name="index"),
        task_index=_require_index(row["task_index"], name="task_index"),
        task=task,
        timestamp_s=timestamp,
        delta_t_s=0.1,
        transition_valid=frame_index + 1 < episode_length,
        cameras=tuple(cameras),
        state=state,
        state_valid=_readonly_validity(8),
        action=action,
        action_valid=_readonly_validity(7),
    )
