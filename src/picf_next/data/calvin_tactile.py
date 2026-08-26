"""Causal CALVIN tactile measurements and official fingertip geometry.

This module decides only whether a released tactile measurement rises above a
calibrated sensor noise floor.  It never predicts contact ownership, object
identity, task relevance, or posterior lifecycle.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinPICFEvidenceFrame

CALVIN_TACTILE_SOURCE_COMMIT = "797142c588c21e76717268b7b430958dbd13bf48"
CALVIN_TACTILE_SOURCE_FILES_SHA256 = {
    "calvin_env/camera/tactile_sensor.py": (
        "6be09ff93ec39d0f6ed4168b00f687c867eaf7a55fab83b18b55b548b47d8b5b"
    ),
    "conf/digit_sensor/config_digit.yml": (
        "6c54686da166d0562a90cfce621dde756f5ee00ec8169f6eab2b08add3ac73d8"
    ),
    "data/franka_panda/panda_digit.urdf": (
        "6c5bfdb97e6dd709467c06c7993958f69537e1fa4cb2525c3633d08a7f2df353"
    ),
}
CALVIN_TACTILE_POSE_SOURCE_FILES_SHA256 = {
    "calvin_env/robot/robot.py": (
        "b214a6d655f64abb5857e9d164b95fdcc956df3e5349e3b2735e67b4d4d2bf01"
    ),
    "data/franka_panda/panda_digit.urdf": (
        "6c5bfdb97e6dd709467c06c7993958f69537e1fa4cb2525c3633d08a7f2df353"
    ),
}
CALVIN_TACTILE_STREAM_NAMES = ("left_digit", "right_digit")
CALVIN_TACTILE_HARDWARE_TYPE = "digit"
CALVIN_TACTILE_FRAME_COUNT = 4
CALVIN_TACTILE_POSE_RECONSTRUCTION = "symmetric-observed-joint-sum-unclipped/v1"

# Source: mees/calvin_env@797142c, data/franka_panda/panda_digit.urdf.
# The TCP is 0.1 m above panda_hand.  Each DIGIT link is 0.03 m above its
# parent finger and 0.018 m outward; the finger joint origin is z=0.0584 m and
# y=+/-0.002 m.  Thus the link origin relative to TCP has z=-0.0116 m and
# y=+/- (0.020 + opening_width / 2).
_DIGIT_TCP_Z_M = -0.0116
_DIGIT_FIXED_HALF_SEPARATION_M = 0.020
_DIGIT_LINK_YAW_RAD = 1.57


def _readonly(value: NDArray[np.generic]) -> NDArray[np.generic]:
    contiguous = np.ascontiguousarray(value)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    return output


def _rpy_zyx_to_matrix(rpy: NDArray[np.generic]) -> NDArray[np.float32]:
    roll, pitch, yaw = (float(value) for value in np.asarray(rpy).reshape(3))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=np.float32)
    ry = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=np.float32)
    rz = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)
    return rz @ ry @ rx


def _transform(
    rotation: NDArray[np.generic],
    translation: NDArray[np.generic],
) -> NDArray[np.float32]:
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    output[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return output


def calvin_digit_sensor_poses_world(
    robot_state: NDArray[np.generic],
) -> Mapping[str, NDArray[np.float32]]:
    """Recover both DIGIT link poses from deploy-visible TCP state and the official URDF."""

    state = np.asarray(robot_state)
    if state.shape != (15,) or not np.issubdtype(state.dtype, np.floating):
        raise ContractError("CALVIN tactile geometry requires the floating shape-(15,) robot state")
    if not np.isfinite(state).all():
        raise ContractError("CALVIN tactile geometry received non-finite robot state")
    # CALVIN records the sum of both observed PyBullet finger-joint positions.
    # Contact forces can move those observations beyond the URDF command limits;
    # clipping would therefore corrupt the pose exactly when touch is informative.
    opening_width_m = float(state[6])
    half_separation_m = _DIGIT_FIXED_HALF_SEPARATION_M + 0.5 * opening_width_m
    world_from_tcp = _transform(_rpy_zyx_to_matrix(state[3:6]), state[:3])
    poses: dict[str, NDArray[np.float32]] = {}
    for name, side in zip(CALVIN_TACTILE_STREAM_NAMES, (1.0, -1.0), strict=True):
        tcp_from_sensor = _transform(
            _rpy_zyx_to_matrix(
                np.asarray((0.0, 0.0, -side * _DIGIT_LINK_YAW_RAD), dtype=np.float32)
            ),
            np.asarray((0.0, side * half_separation_m, _DIGIT_TCP_Z_M), dtype=np.float32),
        )
        poses[name] = _readonly((world_from_tcp @ tcp_from_sensor).astype(np.float32))
    return poses


def calvin_tactile_source_sha256(
    *,
    stream_name: str,
    rgb: NDArray[np.generic],
    deformation_m: NDArray[np.generic],
    timestamp_s: float,
) -> str:
    """Bind RGB, measured deformation, stream identity and deploy-visible time."""

    if stream_name not in CALVIN_TACTILE_STREAM_NAMES:
        raise ContractError(f"unknown CALVIN tactile stream {stream_name!r}")
    image = np.asarray(rgb)
    deformation = np.asarray(deformation_m)
    if image.shape != (160, 120, 3) or image.dtype != np.uint8:
        raise ContractError("CALVIN tactile RGB must be 160-by-120-by-3 uint8")
    if deformation.shape != (160, 120) or deformation.dtype != np.float32:
        raise ContractError("CALVIN tactile deformation must be 160-by-120 float32")
    if not np.isfinite(deformation).all() or not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ContractError("CALVIN tactile source contains invalid deformation or time")
    digest = hashlib.sha256()
    digest.update(b"picf-next.calvin-tactile-source/v1\0")
    digest.update(stream_name.encode("ascii"))
    digest.update(b"\0")
    digest.update(float(timestamp_s).hex().encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(image).tobytes(order="C"))
    digest.update(np.ascontiguousarray(deformation).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CalvinTactileSourceFrame:
    stream_name: str
    hardware_type: str
    rgb: NDArray[np.uint8]
    deformation_m: NDArray[np.float32]
    timestamp_s: float
    source_sha256: str

    def __post_init__(self) -> None:
        if self.hardware_type != CALVIN_TACTILE_HARDWARE_TYPE:
            raise ContractError("both released CALVIN tactile streams are DIGIT sensors")
        expected = calvin_tactile_source_sha256(
            stream_name=self.stream_name,
            rgb=self.rgb,
            deformation_m=self.deformation_m,
            timestamp_s=self.timestamp_s,
        )
        if self.source_sha256 != expected:
            raise ContractError("CALVIN tactile source hash differs from its raw measurement")
        if self.rgb.flags.writeable or self.deformation_m.flags.writeable:
            raise ContractError("CALVIN tactile source arrays must be immutable")

    @property
    def absolute_deformation_max_m(self) -> float:
        return float(np.abs(self.deformation_m).max())


@dataclass(frozen=True, slots=True)
class CalvinTactileEncoderClip:
    """Fixed four-frame AnyTouch input with explicit non-evidence left padding."""

    stream_name: str
    hardware_type: str
    frames: tuple[NDArray[np.uint8], ...]
    frame_timestamps_s: tuple[float, ...]
    source_frame_sha256: tuple[str, ...]
    source_valid: tuple[bool, ...]
    current_absolute_deformation_max_m: float
    validity_threshold_m: float

    def __post_init__(self) -> None:
        count = CALVIN_TACTILE_FRAME_COUNT
        if not (
            len(self.frames)
            == len(self.frame_timestamps_s)
            == len(self.source_frame_sha256)
            == len(self.source_valid)
            == count
        ):
            raise ContractError("CALVIN AnyTouch clips must have four aligned frames")
        if self.stream_name not in CALVIN_TACTILE_STREAM_NAMES:
            raise ContractError("CALVIN AnyTouch clip has an unknown stream")
        if self.hardware_type != CALVIN_TACTILE_HARDWARE_TYPE:
            raise ContractError("CALVIN AnyTouch clip has the wrong hardware type")
        valid = np.asarray(self.source_valid, dtype=np.bool_)
        if not valid.any():
            raise ContractError("CALVIN AnyTouch clip must contain one real source frame")
        first_valid = int(np.flatnonzero(valid)[0])
        if valid[:first_valid].any() or not valid[first_valid:].all():
            raise ContractError("CALVIN AnyTouch padding must be one left prefix")
        timestamps = np.asarray(self.frame_timestamps_s, dtype=np.float64)
        if first_valid and not np.all(timestamps[:first_valid] == timestamps[first_valid]):
            raise ContractError("CALVIN AnyTouch padding must repeat the first real timestamp")
        if len(set(self.source_frame_sha256[first_valid:])) != count - first_valid:
            raise ContractError("real CALVIN AnyTouch sources must remain unique")
        if first_valid and any(
            value != self.source_frame_sha256[first_valid]
            for value in self.source_frame_sha256[:first_valid]
        ):
            raise ContractError("CALVIN AnyTouch padding must repeat the first real source hash")
        first_image = self.frames[first_valid]
        for index, image in enumerate(self.frames):
            if image.shape != (160, 120, 3) or image.dtype != np.uint8 or image.flags.writeable:
                raise ContractError("CALVIN AnyTouch frame shape, dtype or mutability changed")
            if index < first_valid and image is not first_image:
                raise ContractError("CALVIN AnyTouch padding must reference the first real image")
        if (
            not math.isfinite(self.validity_threshold_m)
            or self.validity_threshold_m <= 0.0
            or not math.isfinite(self.current_absolute_deformation_max_m)
            or self.current_absolute_deformation_max_m <= self.validity_threshold_m
        ):
            raise ContractError("CALVIN AnyTouch clip lacks a calibrated current measurement")

    @property
    def padding_count(self) -> int:
        return self.source_valid.index(True)

    def as_array(self) -> NDArray[np.uint8]:
        return np.stack(self.frames, axis=0)


def calvin_tactile_source_frames(
    frame: CalvinPICFEvidenceFrame,
) -> tuple[CalvinTactileSourceFrame, CalvinTactileSourceFrame]:
    """Split the two released DIGIT streams without assigning object ownership."""

    if not isinstance(frame, CalvinPICFEvidenceFrame):
        raise TypeError("CALVIN tactile decoding requires one evidence frame")
    values = {item.key: item.value for item in frame.sensor_observations}
    try:
        rgb = np.asarray(values["observation.tactile.rgb"])
        deformation = np.asarray(values["observation.tactile.depth"])
    except KeyError as exc:
        raise ContractError("CALVIN evidence frame omitted tactile RGB or deformation") from exc
    if rgb.shape != (160, 120, 6) or rgb.dtype != np.uint8:
        raise ContractError("CALVIN combined tactile RGB contract changed")
    if deformation.shape != (160, 120, 2) or deformation.dtype != np.float32:
        raise ContractError("CALVIN combined tactile deformation contract changed")
    output: list[CalvinTactileSourceFrame] = []
    for index, stream_name in enumerate(CALVIN_TACTILE_STREAM_NAMES):
        image = _readonly(rgb[..., 3 * index : 3 * (index + 1)])
        depth = _readonly(deformation[..., index].astype(np.float32, copy=False))
        output.append(
            CalvinTactileSourceFrame(
                stream_name=stream_name,
                hardware_type=CALVIN_TACTILE_HARDWARE_TYPE,
                rgb=image,
                deformation_m=depth,
                timestamp_s=frame.timestamp_s,
                source_sha256=calvin_tactile_source_sha256(
                    stream_name=stream_name,
                    rgb=image,
                    deformation_m=depth,
                    timestamp_s=frame.timestamp_s,
                ),
            )
        )
    return output[0], output[1]


def build_calvin_tactile_encoder_clips(
    source_prefix: Sequence[CalvinPICFEvidenceFrame],
    *,
    validity_thresholds_m: Mapping[str, float],
) -> tuple[CalvinTactileEncoderClip, ...]:
    """Build clips only for independently valid current sensor measurements."""

    if not source_prefix:
        raise ContractError("CALVIN tactile clip builder requires a causal source prefix")
    if set(validity_thresholds_m) != set(CALVIN_TACTILE_STREAM_NAMES):
        raise ContractError("CALVIN tactile thresholds must cover both DIGIT streams exactly")
    thresholds = {name: float(validity_thresholds_m[name]) for name in CALVIN_TACTILE_STREAM_NAMES}
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds.values()):
        raise ContractError("CALVIN tactile thresholds must be finite and positive")
    timestamps = np.asarray([frame.timestamp_s for frame in source_prefix], dtype=np.float64)
    if not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0.0).any():
        raise ContractError("CALVIN tactile source prefix must be strictly chronological")
    streams: dict[str, list[CalvinTactileSourceFrame]] = {
        name: [] for name in CALVIN_TACTILE_STREAM_NAMES
    }
    for frame in source_prefix:
        for tactile in calvin_tactile_source_frames(frame):
            streams[tactile.stream_name].append(tactile)

    clips: list[CalvinTactileEncoderClip] = []
    for name in CALVIN_TACTILE_STREAM_NAMES:
        history = streams[name]
        current_max = history[-1].absolute_deformation_max_m
        threshold = thresholds[name]
        if current_max <= threshold:
            continue
        selected = history[-CALVIN_TACTILE_FRAME_COUNT:]
        padding_count = CALVIN_TACTILE_FRAME_COUNT - len(selected)
        first = selected[0]
        padded = (first,) * padding_count + tuple(selected)
        clips.append(
            CalvinTactileEncoderClip(
                stream_name=name,
                hardware_type=CALVIN_TACTILE_HARDWARE_TYPE,
                frames=tuple(item.rgb for item in padded),
                frame_timestamps_s=tuple(item.timestamp_s for item in padded),
                source_frame_sha256=tuple(item.source_sha256 for item in padded),
                source_valid=(False,) * padding_count + (True,) * len(selected),
                current_absolute_deformation_max_m=current_max,
                validity_threshold_m=threshold,
            )
        )
    return tuple(clips)
