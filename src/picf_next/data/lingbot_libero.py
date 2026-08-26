"""Lossless LIBERO view for LingBot-VLA 2.0's typed 55D host boundary.

The mapping follows the feature order in the pinned official
``configs/vla/real_robot/real_robot.yaml`` rather than the prose ordering in the
model card. Missing embodiment fields are invalid, not observed zeros. The raw
source record remains the authority; this view is an additional host input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.robot_record import (
    MOLMOACT2_LIBERO_CAMERA_KEYS,
    MOLMOACT2_LIBERO_CONTRACT,
    RobotTransitionRecord,
)

Float32Vector = NDArray[np.float32]
BoolVector = NDArray[np.bool_]

LINGBOT_VLA2_WIDTH = 55
LINGBOT_VLA2_FEATURE_SLICES = {
    "arm.position": slice(0, 14),
    "end.position": slice(14, 28),
    "effector.position": slice(28, 30),
    "waist.position": slice(30, 34),
    "head.position": slice(34, 36),
    "base.position": slice(36, 39),
    "hand.position": slice(39, 51),
    "reserved": slice(51, 55),
}
LINGBOT_LIBERO_CAMERA_KEYS = (
    "observation.images.camera_top",
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
)


def _readonly(value: NDArray, *, dtype: np.dtype) -> NDArray:
    output = np.asarray(value, dtype=dtype).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class LingBotLIBEROView:
    """One source-faithful record mapped into the official LingBot feature order."""

    state: Float32Vector
    state_valid: BoolVector
    action: Float32Vector
    action_valid: BoolVector
    camera_payloads: tuple[bytes | None, ...]
    camera_valid: BoolVector
    task: str
    source_global_index: int

    def __post_init__(self) -> None:
        for name, value in (("state", self.state), ("action", self.action)):
            if value.shape != (LINGBOT_VLA2_WIDTH,) or value.dtype != np.float32:
                raise ContractError(f"LingBot {name} must be float32[55]")
            if not np.isfinite(value).all():
                raise ContractError(f"LingBot {name} contains NaN or infinity")
        for name, value in (
            ("state_valid", self.state_valid),
            ("action_valid", self.action_valid),
        ):
            if value.shape != (LINGBOT_VLA2_WIDTH,) or value.dtype != np.bool_:
                raise ContractError(f"LingBot {name} must be bool[55]")
        if len(self.camera_payloads) != 3 or self.camera_valid.shape != (3,):
            raise ContractError("LingBot LIBERO view requires three camera slots")
        if self.camera_valid.dtype != np.bool_:
            raise ContractError("LingBot camera validity must be boolean")
        for payload, valid in zip(self.camera_payloads, self.camera_valid, strict=True):
            if bool(valid) != (payload is not None):
                raise ContractError("camera payload and validity disagree")
        if not self.task or self.source_global_index < 0:
            raise ContractError("LingBot view must retain task and source index")


def map_libero_record_to_lingbot(record: RobotTransitionRecord) -> LingBotLIBEROView:
    """Map a public LIBERO transition without altering source action semantics.

    The single-arm Cartesian pose occupies the first six dimensions of
    ``end.position``. The two opposing Panda finger qpos values are represented
    by one physical aperture ``left - right`` in the first effector dimension;
    both original values remain losslessly available in ``record.state``.
    LIBERO's normalized delta Cartesian action is not converted to an absolute
    pose or quaternion.
    """

    if record.contract != MOLMOACT2_LIBERO_CONTRACT:
        raise ContractError("LingBot LIBERO mapping received an unsupported record")
    if tuple(camera.key for camera in record.cameras) != MOLMOACT2_LIBERO_CAMERA_KEYS:
        raise ContractError("LIBERO camera order differs from the audited contract")

    state = np.zeros(LINGBOT_VLA2_WIDTH, dtype=np.float32)
    state_valid = np.zeros(LINGBOT_VLA2_WIDTH, dtype=np.bool_)
    action = np.zeros(LINGBOT_VLA2_WIDTH, dtype=np.float32)
    action_valid = np.zeros(LINGBOT_VLA2_WIDTH, dtype=np.bool_)

    end_slice = LINGBOT_VLA2_FEATURE_SLICES["end.position"]
    effector_slice = LINGBOT_VLA2_FEATURE_SLICES["effector.position"]
    state[end_slice.start : end_slice.start + 6] = record.state[:6]
    state_valid[end_slice.start : end_slice.start + 6] = record.state_valid[:6]
    state[effector_slice.start] = record.state[6] - record.state[7]
    state_valid[effector_slice.start] = bool(record.state_valid[6:8].all())

    action[end_slice.start : end_slice.start + 6] = record.action[:6]
    action_valid[end_slice.start : end_slice.start + 6] = record.action_valid[:6]
    action[effector_slice.start] = record.action[6]
    action_valid[effector_slice.start] = bool(record.action_valid[6])

    payloads: tuple[bytes | None, ...] = (
        record.cameras[0].encoded_bytes,
        record.cameras[1].encoded_bytes,
        None,
    )
    camera_valid = np.array([True, True, False], dtype=np.bool_)
    return LingBotLIBEROView(
        state=_readonly(state, dtype=np.dtype(np.float32)),
        state_valid=_readonly(state_valid, dtype=np.dtype(np.bool_)),
        action=_readonly(action, dtype=np.dtype(np.float32)),
        action_valid=_readonly(action_valid, dtype=np.dtype(np.bool_)),
        camera_payloads=payloads,
        camera_valid=_readonly(camera_valid, dtype=np.dtype(np.bool_)),
        task=record.task,
        source_global_index=record.global_index,
    )
