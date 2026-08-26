"""Typed CALVIN view for LingBot-VLA2's native 55D policy boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CALVIN_CONTRACT,
    CALVIN_HOST_IMAGE_KEYS,
    CalvinPhysicalSample,
    CalvinStatefulTransitionSample,
)
from picf_next.data.lingbot_libero import (
    LINGBOT_VLA2_FEATURE_SLICES,
    LINGBOT_VLA2_WIDTH,
)

Float32Vector = NDArray[np.float32]
Float32Matrix = NDArray[np.float32]
BoolVector = NDArray[np.bool_]


def _readonly(value: NDArray, *, dtype: np.dtype) -> NDArray:
    output = np.asarray(value, dtype=dtype).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class LingBotCALVINTransition:
    """One stateful CALVIN sample mapped without changing action semantics."""

    state: Float32Vector
    state_valid: BoolVector
    actions: Float32Matrix
    action_valid: BoolVector
    action_is_pad: BoolVector
    previous_executed_action: Float32Vector
    previous_action_valid: bool
    camera_top: NDArray[np.uint8]
    camera_wrist_left: NDArray[np.uint8]
    task: str
    sample_key: str
    episode_key: str
    transition_index: int
    elapsed_time_s: float

    def __post_init__(self) -> None:
        if self.state.shape != (LINGBOT_VLA2_WIDTH,) or self.state.dtype != np.float32:
            raise ContractError("LingBot CALVIN state must be float32[55]")
        if self.state_valid.shape != (LINGBOT_VLA2_WIDTH,) or self.state_valid.dtype != np.bool_:
            raise ContractError("LingBot CALVIN state validity must be bool[55]")
        if (
            self.actions.ndim != 2
            or self.actions.shape[1] != LINGBOT_VLA2_WIDTH
            or self.actions.dtype != np.float32
        ):
            raise ContractError("LingBot CALVIN actions must be float32[horizon,55]")
        if self.action_valid.shape != (LINGBOT_VLA2_WIDTH,) or self.action_valid.dtype != np.bool_:
            raise ContractError("LingBot CALVIN action validity must be bool[55]")
        if self.action_is_pad.shape != self.actions.shape[:1] or (
            self.action_is_pad.dtype != np.bool_
        ):
            raise ContractError("LingBot CALVIN padding must align with the action horizon")
        if (
            self.previous_executed_action.shape != (LINGBOT_VLA2_WIDTH,)
            or self.previous_executed_action.dtype != np.float32
        ):
            raise ContractError("previous LingBot action must be float32[55]")
        numeric = (self.state, self.actions, self.previous_executed_action)
        if any(not np.isfinite(value).all() for value in numeric):
            raise ContractError("LingBot CALVIN numeric fields must be finite")
        arrays = (
            self.state,
            self.state_valid,
            self.actions,
            self.action_valid,
            self.action_is_pad,
            self.previous_executed_action,
            self.camera_top,
            self.camera_wrist_left,
        )
        if any(value.flags.writeable for value in arrays):
            raise ContractError("LingBot CALVIN arrays must be immutable")
        for image in (self.camera_top, self.camera_wrist_left):
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ContractError("LingBot CALVIN cameras must be HWC uint8 RGB")
        if not self.task or not self.sample_key or not self.episode_key:
            raise ContractError("LingBot CALVIN source identity is incomplete")
        if self.transition_index < 0:
            raise ContractError("LingBot CALVIN transition index must be non-negative")
        if not np.isfinite(self.elapsed_time_s) or self.elapsed_time_s <= 0:
            raise ContractError("LingBot CALVIN elapsed time must be finite and positive")
        if self.previous_action_valid != (self.transition_index > 0):
            raise ContractError("previous action validity must match the segment boundary")
        if not self.previous_action_valid and np.any(self.previous_executed_action != 0):
            raise ContractError("segment reset must carry a zero previous action")

    def feature_transform_item(self) -> dict[str, Any]:
        """Return the exact raw item consumed by the pinned FeatureTransform."""

        import torch

        def image(value: NDArray[np.uint8]) -> torch.Tensor:
            return torch.from_numpy(value.copy()).permute(2, 0, 1).to(torch.float32)

        return {
            "observation.state.lingbot": torch.from_numpy(self.state.copy()),
            "action.lingbot": torch.from_numpy(self.actions.copy()),
            "action.lingbot_is_pad": torch.from_numpy(self.action_is_pad.copy()),
            "observation.images.camera_top": image(self.camera_top),
            "observation.images.camera_wrist_left": image(self.camera_wrist_left),
            "task": self.task,
        }


def map_calvin_action_to_lingbot(action: NDArray[np.float32]) -> NDArray[np.float32]:
    """Embed exact CALVIN controls in LingBot's released 55D action chart."""

    if action.shape[-1] != 7:
        raise ContractError("CALVIN action width must be seven")
    mapped = np.zeros((*action.shape[:-1], LINGBOT_VLA2_WIDTH), dtype=np.float32)
    end = LINGBOT_VLA2_FEATURE_SLICES["end.position"]
    effector = LINGBOT_VLA2_FEATURE_SLICES["effector.position"]
    mapped[..., end.start : end.start + 6] = action[..., :6]
    mapped[..., effector.start] = action[..., 6]
    return mapped


def map_calvin_transition_to_lingbot(
    sample: CalvinStatefulTransitionSample | CalvinPhysicalSample,
) -> LingBotCALVINTransition:
    """Map current targets and previous executed action through separate paths."""

    if not isinstance(sample, CalvinStatefulTransitionSample | CalvinPhysicalSample):
        raise TypeError("LingBot CALVIN mapping requires a typed transition sample")
    if sample.record.contract != CALVIN_CONTRACT:
        raise ContractError("LingBot CALVIN mapping received an unsupported record")
    state = np.zeros(LINGBOT_VLA2_WIDTH, dtype=np.float32)
    state_valid = np.zeros(LINGBOT_VLA2_WIDTH, dtype=np.bool_)
    arm = LINGBOT_VLA2_FEATURE_SLICES["arm.position"]
    end = LINGBOT_VLA2_FEATURE_SLICES["end.position"]
    effector = LINGBOT_VLA2_FEATURE_SLICES["effector.position"]
    state[arm.start : arm.start + 7] = sample.record.state[7:14]
    state_valid[arm.start : arm.start + 7] = sample.record.state_valid[7:14]
    state[end.start : end.start + 6] = sample.record.state[:6]
    state_valid[end.start : end.start + 6] = sample.record.state_valid[:6]
    state[effector.start] = sample.record.state[6]
    state_valid[effector.start] = sample.record.state_valid[6]

    actions = map_calvin_action_to_lingbot(sample.host_sample.action)
    action_valid = np.zeros(LINGBOT_VLA2_WIDTH, dtype=np.bool_)
    action_valid[end.start : end.start + 6] = sample.record.action_valid[:6]
    action_valid[effector.start] = sample.record.action_valid[6]
    if isinstance(sample, CalvinPhysicalSample):
        previous_raw = (
            np.zeros(7, dtype=np.float32)
            if sample.reset
            else sample.incoming_control_span.raw_actions[-1]
        )
    else:
        previous_raw = sample.previous_executed_action
    previous = map_calvin_action_to_lingbot(previous_raw)
    observation = sample.host_sample.observation
    camera_top = observation[CALVIN_HOST_IMAGE_KEYS[0]]
    camera_wrist_left = observation[CALVIN_HOST_IMAGE_KEYS[1]]
    if not isinstance(camera_top, np.ndarray) or not isinstance(camera_wrist_left, np.ndarray):
        raise ContractError("LingBot CALVIN camera observations must be arrays")
    return LingBotCALVINTransition(
        state=_readonly(state, dtype=np.dtype(np.float32)),
        state_valid=_readonly(state_valid, dtype=np.dtype(np.bool_)),
        actions=_readonly(actions, dtype=np.dtype(np.float32)),
        action_valid=_readonly(action_valid, dtype=np.dtype(np.bool_)),
        action_is_pad=_readonly(sample.host_sample.action_is_pad, dtype=np.dtype(np.bool_)),
        previous_executed_action=_readonly(previous, dtype=np.dtype(np.float32)),
        previous_action_valid=sample.transition_index > 0,
        camera_top=_readonly(camera_top, dtype=np.dtype(np.uint8)),
        camera_wrist_left=_readonly(camera_wrist_left, dtype=np.dtype(np.uint8)),
        task=sample.record.task,
        sample_key=sample.sample_key,
        episode_key=sample.episode_key,
        transition_index=sample.transition_index,
        elapsed_time_s=sample.record.delta_t_s,
    )
