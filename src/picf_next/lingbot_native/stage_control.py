"""Fail-closed launch semantics for representation and joint-adoption stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

NATIVE_STAGE_CONTROL_SCHEMA = "picf-next.lingbot-native-stage-control.v1"
NATIVE_REPRESENTATION_STAGE = "representation"
NATIVE_JOINT_ADOPTION_STAGE = "joint_adoption"
NATIVE_STAGED_TRAINING_STAGES = (
    NATIVE_REPRESENTATION_STAGE,
    NATIVE_JOINT_ADOPTION_STAGE,
)
NATIVE_RELEASED_INITIALIZATION = "released_initialization"
NATIVE_EXACT_RESUME = "exact_resume"
NATIVE_REPRESENTATION_ADOPTION = "representation_model_adoption"
NATIVE_STAGE_LAUNCH_MODES = (
    NATIVE_RELEASED_INITIALIZATION,
    NATIVE_EXACT_RESUME,
    NATIVE_REPRESENTATION_ADOPTION,
)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class NativeStagedLaunchPlan:
    """One immutable stage transition and its exact state-loading boundary."""

    training_stage: str
    launch_mode: str
    input_checkpoint_stage: str | None
    input_checkpoint_step: int | None
    input_stage_step: int
    saved_stage_step: int
    total_planned_stage_steps: int
    stream_plan_sha256: str
    representation_split_sha256: str
    representation_acceptance_sha256: str | None
    load_model: bool
    load_optimizer: bool
    load_lane_state: bool
    load_rng_state: bool
    reset_stream_cursor: bool
    action_enabled: bool
    schema: str = NATIVE_STAGE_CONTROL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NATIVE_STAGE_CONTROL_SCHEMA:
            raise ValueError("native staged-launch schema changed")
        if self.training_stage not in NATIVE_STAGED_TRAINING_STAGES:
            raise ValueError("native staged-launch training stage is unsupported")
        if self.launch_mode not in NATIVE_STAGE_LAUNCH_MODES:
            raise ValueError("native staged-launch mode is unsupported")
        _require_nonnegative_int(self.input_stage_step, "native input stage step")
        _require_nonnegative_int(self.saved_stage_step, "native saved stage step")
        _require_nonnegative_int(
            self.total_planned_stage_steps,
            "native total planned stage steps",
        )
        if (
            self.saved_stage_step <= self.input_stage_step
            or self.saved_stage_step > self.total_planned_stage_steps
        ):
            raise ValueError("native staged-launch step interval is invalid")
        _require_sha256(self.stream_plan_sha256, "native stream plan sha256")
        _require_sha256(
            self.representation_split_sha256,
            "native representation split sha256",
        )
        flags = (
            self.load_model,
            self.load_optimizer,
            self.load_lane_state,
            self.load_rng_state,
            self.reset_stream_cursor,
            self.action_enabled,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise TypeError("native staged-launch state flags must be boolean")

        if self.launch_mode == NATIVE_RELEASED_INITIALIZATION:
            if (
                self.training_stage != NATIVE_REPRESENTATION_STAGE
                or self.input_checkpoint_stage is not None
                or self.input_checkpoint_step is not None
                or self.input_stage_step != 0
                or self.representation_acceptance_sha256 is not None
                or self.load_model
                or self.load_optimizer
                or self.load_lane_state
                or self.load_rng_state
                or not self.reset_stream_cursor
                or self.action_enabled
            ):
                raise ValueError("native released representation initialization is inconsistent")
            return

        if self.input_checkpoint_stage not in NATIVE_STAGED_TRAINING_STAGES:
            raise ValueError("native staged launch omitted its input checkpoint stage")
        if self.input_checkpoint_step is None:
            raise ValueError("native staged launch omitted its input checkpoint step")
        _require_nonnegative_int(
            self.input_checkpoint_step,
            "native input checkpoint step",
        )
        if not self.load_model:
            raise ValueError("native staged checkpoint launch must load model state")

        if self.launch_mode == NATIVE_EXACT_RESUME:
            if (
                self.input_checkpoint_stage != self.training_stage
                or self.input_checkpoint_step != self.input_stage_step
                or self.input_stage_step <= 0
                or self.representation_acceptance_sha256 is not None
                or not self.load_optimizer
                or not self.load_lane_state
                or not self.load_rng_state
                or self.reset_stream_cursor
                or self.action_enabled != (self.training_stage == NATIVE_JOINT_ADOPTION_STAGE)
            ):
                raise ValueError("native exact stage resume is inconsistent")
            return

        if (
            self.training_stage != NATIVE_JOINT_ADOPTION_STAGE
            or self.input_checkpoint_stage != NATIVE_REPRESENTATION_STAGE
            or self.input_checkpoint_step <= 0
            or self.input_stage_step != 0
            or self.representation_acceptance_sha256 is None
            or self.load_optimizer
            or self.load_lane_state
            or self.load_rng_state
            or not self.reset_stream_cursor
            or not self.action_enabled
        ):
            raise ValueError("native representation-to-joint adoption is inconsistent")
        _require_sha256(
            self.representation_acceptance_sha256,
            "native representation acceptance sha256",
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "training_stage": self.training_stage,
            "launch_mode": self.launch_mode,
            "input_checkpoint_stage": self.input_checkpoint_stage,
            "input_checkpoint_step": self.input_checkpoint_step,
            "input_stage_step": self.input_stage_step,
            "saved_stage_step": self.saved_stage_step,
            "total_planned_stage_steps": self.total_planned_stage_steps,
            "stream_plan_sha256": self.stream_plan_sha256,
            "representation_split_sha256": self.representation_split_sha256,
            "representation_acceptance_sha256": (self.representation_acceptance_sha256),
            "load_model": self.load_model,
            "load_optimizer": self.load_optimizer,
            "load_lane_state": self.load_lane_state,
            "load_rng_state": self.load_rng_state,
            "reset_stream_cursor": self.reset_stream_cursor,
            "action_enabled": self.action_enabled,
        }

    @property
    def digest(self) -> str:
        return _canonical_digest(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "digest": self.digest}


def plan_native_staged_launch(
    *,
    training_stage: str,
    launch_mode: str,
    invocation_steps: int,
    total_planned_stage_steps: int,
    stream_plan_sha256: str,
    representation_split_sha256: str,
    input_checkpoint_stage: str | None = None,
    input_checkpoint_step: int | None = None,
    representation_acceptance_sha256: str | None = None,
) -> NativeStagedLaunchPlan:
    """Derive exact load/reset semantics without inspecting mutable checkpoint state."""

    if training_stage not in NATIVE_STAGED_TRAINING_STAGES:
        raise ValueError("native staged training stage is unsupported")
    if launch_mode not in NATIVE_STAGE_LAUNCH_MODES:
        raise ValueError("native staged launch mode is unsupported")
    if not isinstance(invocation_steps, int) or isinstance(invocation_steps, bool):
        raise TypeError("native staged invocation steps must be an integer")
    if invocation_steps <= 0:
        raise ValueError("native staged invocation steps must be positive")
    total_planned_stage_steps = _require_nonnegative_int(
        total_planned_stage_steps,
        "native total planned stage steps",
    )
    if total_planned_stage_steps <= 0:
        raise ValueError("native total planned stage steps must be positive")
    _require_sha256(stream_plan_sha256, "native stream plan sha256")
    _require_sha256(
        representation_split_sha256,
        "native representation split sha256",
    )

    if launch_mode == NATIVE_RELEASED_INITIALIZATION:
        input_stage_step = 0
        flags = (False, False, False, False, True, False)
    elif launch_mode == NATIVE_EXACT_RESUME:
        if input_checkpoint_step is None:
            raise ValueError("native exact resume requires an input checkpoint step")
        input_stage_step = input_checkpoint_step
        flags = (
            True,
            True,
            True,
            True,
            False,
            training_stage == NATIVE_JOINT_ADOPTION_STAGE,
        )
    else:
        input_stage_step = 0
        flags = (True, False, False, False, True, True)
    saved_stage_step = input_stage_step + invocation_steps
    return NativeStagedLaunchPlan(
        training_stage=training_stage,
        launch_mode=launch_mode,
        input_checkpoint_stage=input_checkpoint_stage,
        input_checkpoint_step=input_checkpoint_step,
        input_stage_step=input_stage_step,
        saved_stage_step=saved_stage_step,
        total_planned_stage_steps=total_planned_stage_steps,
        stream_plan_sha256=stream_plan_sha256,
        representation_split_sha256=representation_split_sha256,
        representation_acceptance_sha256=representation_acceptance_sha256,
        load_model=flags[0],
        load_optimizer=flags[1],
        load_lane_state=flags[2],
        load_rng_state=flags[3],
        reset_stream_cursor=flags[4],
        action_enabled=flags[5],
    )
