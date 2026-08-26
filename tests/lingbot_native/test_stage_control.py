from __future__ import annotations

from dataclasses import replace

import pytest

from picf_next.lingbot_native.stage_control import (
    NATIVE_EXACT_RESUME,
    NATIVE_JOINT_ADOPTION_STAGE,
    NATIVE_RELEASED_INITIALIZATION,
    NATIVE_REPRESENTATION_ADOPTION,
    NATIVE_REPRESENTATION_STAGE,
    plan_native_staged_launch,
)

_STREAM_SHA256 = "1" * 64
_SPLIT_SHA256 = "2" * 64
_ACCEPTANCE_SHA256 = "3" * 64


def test_representation_released_initialization_resets_all_mutable_state() -> None:
    plan = plan_native_staged_launch(
        training_stage=NATIVE_REPRESENTATION_STAGE,
        launch_mode=NATIVE_RELEASED_INITIALIZATION,
        invocation_steps=1,
        total_planned_stage_steps=200,
        stream_plan_sha256=_STREAM_SHA256,
        representation_split_sha256=_SPLIT_SHA256,
    )
    assert (plan.input_stage_step, plan.saved_stage_step) == (0, 1)
    assert not plan.load_model
    assert not plan.load_optimizer
    assert not plan.load_lane_state
    assert not plan.load_rng_state
    assert plan.reset_stream_cursor
    assert not plan.action_enabled
    assert plan.as_dict()["digest"] == plan.digest


def test_representation_exact_resume_inherits_every_mutable_boundary() -> None:
    plan = plan_native_staged_launch(
        training_stage=NATIVE_REPRESENTATION_STAGE,
        launch_mode=NATIVE_EXACT_RESUME,
        input_checkpoint_stage=NATIVE_REPRESENTATION_STAGE,
        input_checkpoint_step=20,
        invocation_steps=30,
        total_planned_stage_steps=200,
        stream_plan_sha256=_STREAM_SHA256,
        representation_split_sha256=_SPLIT_SHA256,
    )
    assert (plan.input_stage_step, plan.saved_stage_step) == (20, 50)
    assert plan.load_model
    assert plan.load_optimizer
    assert plan.load_lane_state
    assert plan.load_rng_state
    assert not plan.reset_stream_cursor
    assert not plan.action_enabled


def test_joint_adoption_loads_only_representation_model_and_restarts_stream() -> None:
    plan = plan_native_staged_launch(
        training_stage=NATIVE_JOINT_ADOPTION_STAGE,
        launch_mode=NATIVE_REPRESENTATION_ADOPTION,
        input_checkpoint_stage=NATIVE_REPRESENTATION_STAGE,
        input_checkpoint_step=200,
        representation_acceptance_sha256=_ACCEPTANCE_SHA256,
        invocation_steps=1,
        total_planned_stage_steps=120,
        stream_plan_sha256=_STREAM_SHA256,
        representation_split_sha256=_SPLIT_SHA256,
    )
    assert (plan.input_stage_step, plan.saved_stage_step) == (0, 1)
    assert plan.load_model
    assert not plan.load_optimizer
    assert not plan.load_lane_state
    assert not plan.load_rng_state
    assert plan.reset_stream_cursor
    assert plan.action_enabled


def test_joint_adoption_exact_resume_keeps_joint_state() -> None:
    plan = plan_native_staged_launch(
        training_stage=NATIVE_JOINT_ADOPTION_STAGE,
        launch_mode=NATIVE_EXACT_RESUME,
        input_checkpoint_stage=NATIVE_JOINT_ADOPTION_STAGE,
        input_checkpoint_step=20,
        invocation_steps=100,
        total_planned_stage_steps=120,
        stream_plan_sha256=_STREAM_SHA256,
        representation_split_sha256=_SPLIT_SHA256,
    )
    assert plan.saved_stage_step == 120
    assert plan.load_model
    assert plan.load_optimizer
    assert plan.load_lane_state
    assert plan.load_rng_state
    assert not plan.reset_stream_cursor
    assert plan.action_enabled


@pytest.mark.parametrize(
    ("training_stage", "launch_mode"),
    (
        (NATIVE_JOINT_ADOPTION_STAGE, NATIVE_RELEASED_INITIALIZATION),
        (NATIVE_REPRESENTATION_STAGE, NATIVE_REPRESENTATION_ADOPTION),
    ),
)
def test_stage_control_rejects_incoherent_stage_launch_pairs(
    training_stage: str,
    launch_mode: str,
) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        plan_native_staged_launch(
            training_stage=training_stage,
            launch_mode=launch_mode,
            input_checkpoint_stage=NATIVE_REPRESENTATION_STAGE,
            input_checkpoint_step=200,
            representation_acceptance_sha256=_ACCEPTANCE_SHA256,
            invocation_steps=1,
            total_planned_stage_steps=200,
            stream_plan_sha256=_STREAM_SHA256,
            representation_split_sha256=_SPLIT_SHA256,
        )


def test_stage_control_rejects_cross_stage_resume_and_tampered_flags() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        plan_native_staged_launch(
            training_stage=NATIVE_JOINT_ADOPTION_STAGE,
            launch_mode=NATIVE_EXACT_RESUME,
            input_checkpoint_stage=NATIVE_REPRESENTATION_STAGE,
            input_checkpoint_step=20,
            invocation_steps=1,
            total_planned_stage_steps=120,
            stream_plan_sha256=_STREAM_SHA256,
            representation_split_sha256=_SPLIT_SHA256,
        )

    valid = plan_native_staged_launch(
        training_stage=NATIVE_JOINT_ADOPTION_STAGE,
        launch_mode=NATIVE_REPRESENTATION_ADOPTION,
        input_checkpoint_stage=NATIVE_REPRESENTATION_STAGE,
        input_checkpoint_step=200,
        representation_acceptance_sha256=_ACCEPTANCE_SHA256,
        invocation_steps=1,
        total_planned_stage_steps=120,
        stream_plan_sha256=_STREAM_SHA256,
        representation_split_sha256=_SPLIT_SHA256,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        replace(valid, load_optimizer=True)


def test_stage_control_rejects_out_of_bounds_interval() -> None:
    with pytest.raises(ValueError, match="interval is invalid"):
        plan_native_staged_launch(
            training_stage=NATIVE_REPRESENTATION_STAGE,
            launch_mode=NATIVE_EXACT_RESUME,
            input_checkpoint_stage=NATIVE_REPRESENTATION_STAGE,
            input_checkpoint_step=190,
            invocation_steps=11,
            total_planned_stage_steps=200,
            stream_plan_sha256=_STREAM_SHA256,
            representation_split_sha256=_SPLIT_SHA256,
        )
