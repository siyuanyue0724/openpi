"""Source-faithful CALVIN temporal ABI for the released V-JEPA2-AC donor.

The DROID recipe conditions on differences between consecutive *observed*
Cartesian poses.  These values are realized motion, not CALVIN's normalized
policy commands.  Keeping that distinction in the type and field names avoids
silently overstating what the donor gate proves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinEpisode

VJEPA2_AC_CALVIN_CONTRACT = "calvin-vjepa2-ac-realized-motion-clip/v1"
VJEPA2_AC_FRAME_COUNT = 8
VJEPA2_AC_TARGET_FPS = 4
VJEPA2_AC_IMAGE_SIZE = 256
VJEPA2_AC_PATCH_SIZE = 16
VJEPA2_AC_TUBELET_SIZE = 2
VJEPA2_AC_CAMERA_KEYS = ("rgb_static", "rgb_gripper")
VJEPA2_AC_STATE_AXES = (
    "tcp.position.x",
    "tcp.position.y",
    "tcp.position.z",
    "tcp.euler.x",
    "tcp.euler.y",
    "tcp.euler.z",
    "gripper.opening_width",
)
VJEPA2_AC_REALIZED_MOTION_AXES = (
    "realized_delta_tcp.position.x",
    "realized_delta_tcp.position.y",
    "realized_delta_tcp.position.z",
    "realized_delta_tcp.rotation_xyz.x",
    "realized_delta_tcp.rotation_xyz.y",
    "realized_delta_tcp.rotation_xyz.z",
    "realized_delta_gripper.opening_width",
)


def _readonly(value: NDArray, *, dtype: np.dtype) -> NDArray:
    array = np.ascontiguousarray(np.asarray(value), dtype=dtype)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)
    return frozen


def vjepa2_ac_calvin_stride(*, control_hz: int, target_fps: int = VJEPA2_AC_TARGET_FPS) -> int:
    """Match the donor's ``ceil(video_fps / requested_fps)`` sampling rule."""

    for value, name in ((control_hz, "control_hz"), (target_fps, "target_fps")):
        if isinstance(value, bool | np.bool_) or not isinstance(value, Integral) or value <= 0:
            raise ContractError(f"V-JEPA2-AC {name} must be a positive integer")
    return int(math.ceil(int(control_hz) / int(target_fps)))


def vjepa2_ac_realized_pose_differences(states: NDArray) -> NDArray[np.float32]:
    """Port ``DROIDVideoDataset.poses_to_diffs`` without changing its geometry."""

    poses = np.asarray(states)
    if (
        poses.ndim != 2
        or poses.shape[0] < 2
        or poses.shape[1] != 7
        or not np.issubdtype(poses.dtype, np.floating)
        or not np.isfinite(poses).all()
    ):
        raise ContractError("V-JEPA2-AC states must be finite floating [time, 7] poses")

    xyz = poses[:, :3]
    rotations = Rotation.from_euler("xyz", poses[:, 3:6], degrees=False).as_matrix()
    xyz_diff = xyz[1:] - xyz[:-1]
    relative_rotations = rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
    angle_diff = Rotation.from_matrix(relative_rotations).as_euler("xyz", degrees=False)
    gripper_diff = poses[1:, -1:] - poses[:-1, -1:]
    realized = np.concatenate((xyz_diff, angle_diff, gripper_diff), axis=1)
    return _readonly(realized, dtype=np.dtype(np.float32))


def calvin_vjepa2_ac_frame_indices(
    episode: CalvinEpisode,
    *,
    end_global_index: int,
    control_hz: int,
) -> tuple[int, ...]:
    """Return one 8-frame, approximately 4 Hz donor clip ending at a source frame."""

    if not isinstance(episode, CalvinEpisode):
        raise TypeError("V-JEPA2-AC sampling requires a CalvinEpisode")
    if (
        isinstance(end_global_index, bool | np.bool_)
        or not isinstance(end_global_index, Integral)
    ):
        raise ContractError("V-JEPA2-AC clip end must be an integer source index")
    stride = vjepa2_ac_calvin_stride(control_hz=control_hz)
    first = int(end_global_index) - (VJEPA2_AC_FRAME_COUNT - 1) * stride
    indices = tuple(first + offset * stride for offset in range(VJEPA2_AC_FRAME_COUNT))
    if first < episode.start or indices[-1] > episode.end:
        raise ContractError("V-JEPA2-AC clip crosses a raw CALVIN episode boundary")
    return indices


@dataclass(frozen=True, slots=True)
class CalvinVjepa2AcClip:
    """One immutable donor clip with an explicit realized-motion conditioning ABI."""

    episode_index: int
    camera_key: str
    frame_indices: tuple[int, ...]
    frame_timestamps_s: NDArray[np.float32]
    images: NDArray[np.uint8]
    states: NDArray[np.float32]
    realized_motion: NDArray[np.float32]
    source_control_hz: int
    source_stride: int
    contract: str = VJEPA2_AC_CALVIN_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != VJEPA2_AC_CALVIN_CONTRACT:
            raise ContractError("V-JEPA2-AC CALVIN clip contract changed")
        if (
            isinstance(self.episode_index, bool | np.bool_)
            or not isinstance(self.episode_index, Integral)
            or self.episode_index < 0
        ):
            raise ContractError("V-JEPA2-AC episode identity is invalid")
        if self.camera_key not in VJEPA2_AC_CAMERA_KEYS:
            raise ContractError("V-JEPA2-AC CALVIN camera is unsupported")
        if len(self.frame_indices) != VJEPA2_AC_FRAME_COUNT:
            raise ContractError("V-JEPA2-AC clip must contain exactly eight frames")
        expected_indices = tuple(
            self.frame_indices[0] + offset * self.source_stride
            for offset in range(VJEPA2_AC_FRAME_COUNT)
        )
        if self.frame_indices != expected_indices:
            raise ContractError("V-JEPA2-AC source indices are not uniformly sampled")
        if self.source_stride != vjepa2_ac_calvin_stride(control_hz=self.source_control_hz):
            raise ContractError("V-JEPA2-AC source stride differs from the donor sampling rule")
        if (
            self.frame_timestamps_s.shape != (VJEPA2_AC_FRAME_COUNT,)
            or self.frame_timestamps_s.dtype != np.float32
            or self.frame_timestamps_s.flags.writeable
            or not np.isfinite(self.frame_timestamps_s).all()
            or np.any(np.diff(self.frame_timestamps_s) <= 0.0)
        ):
            raise ContractError("V-JEPA2-AC frame timestamps are invalid")
        if (
            self.images.ndim != 4
            or self.images.shape[0] != VJEPA2_AC_FRAME_COUNT
            or self.images.shape[-1] != 3
            or self.images.dtype != np.uint8
            or self.images.flags.writeable
        ):
            raise ContractError("V-JEPA2-AC images must be immutable uint8 [8, H, W, 3]")
        if (
            self.states.shape != (VJEPA2_AC_FRAME_COUNT, 7)
            or self.states.dtype != np.float32
            or self.states.flags.writeable
            or not np.isfinite(self.states).all()
        ):
            raise ContractError("V-JEPA2-AC states must be immutable finite float32 [8, 7]")
        if (
            self.realized_motion.shape != (VJEPA2_AC_FRAME_COUNT - 1, 7)
            or self.realized_motion.dtype != np.float32
            or self.realized_motion.flags.writeable
            or not np.isfinite(self.realized_motion).all()
        ):
            raise ContractError(
                "V-JEPA2-AC realized motion must be immutable finite float32 [7, 7]"
            )
        expected_motion = vjepa2_ac_realized_pose_differences(self.states)
        if not np.array_equal(self.realized_motion, expected_motion):
            raise ContractError("V-JEPA2-AC realized motion disagrees with consecutive states")


def load_calvin_vjepa2_ac_clip(
    index: CalvinDatasetIndex,
    *,
    end_global_index: int,
    camera_key: str = "rgb_static",
) -> CalvinVjepa2AcClip:
    """Load a source-episode clip without language, masks or policy action labels."""

    if not isinstance(index, CalvinDatasetIndex):
        raise TypeError("V-JEPA2-AC loading requires a CalvinDatasetIndex")
    if camera_key not in VJEPA2_AC_CAMERA_KEYS:
        raise ContractError("V-JEPA2-AC CALVIN camera is unsupported")
    episode = index.source_episode(end_global_index)
    frame_indices = calvin_vjepa2_ac_frame_indices(
        episode,
        end_global_index=end_global_index,
        control_hz=index.control_hz,
    )
    payloads = tuple(
        index.validated_source_frame_arrays(
            source_index,
            fields=(camera_key, "robot_obs"),
        )
        for source_index in frame_indices
    )
    images = _readonly(
        np.stack([payload[camera_key] for payload in payloads], axis=0),
        dtype=np.dtype(np.uint8),
    )
    states = _readonly(
        np.stack([payload["robot_obs"][:7] for payload in payloads], axis=0),
        dtype=np.dtype(np.float32),
    )
    stride = vjepa2_ac_calvin_stride(control_hz=index.control_hz)
    timestamps = _readonly(
        np.asarray(
            [
                (source_index - episode.start) / float(index.control_hz)
                for source_index in frame_indices
            ],
            dtype=np.float32,
        ),
        dtype=np.dtype(np.float32),
    )
    return CalvinVjepa2AcClip(
        episode_index=episode.index,
        camera_key=camera_key,
        frame_indices=frame_indices,
        frame_timestamps_s=timestamps,
        images=images,
        states=states,
        realized_motion=vjepa2_ac_realized_pose_differences(states),
        source_control_hz=index.control_hz,
        source_stride=stride,
    )
