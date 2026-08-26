from __future__ import annotations

import pytest

from picf_next.lingbot_native.task_action_supervision import (
    TASK_ACTION_SUPERVISION_SCHEMA,
    TaskActionSupervisionScope,
    require_factual_action_supervision,
    task_action_supervision_receipt,
)

ACTION_SHA256 = "a" * 64


def test_source_task_and_instruction_enable_official_action_loss() -> None:
    receipt = task_action_supervision_receipt(
        sample_key="episode/frame",
        source_task_key="move_slider_right",
        source_instruction="move the sliding door to the right",
        candidate_task_key="move_slider_right",
        candidate_instruction="move the sliding door to the right",
        source_action_targets_sha256=ACTION_SHA256,
        candidate_action_targets_sha256=ACTION_SHA256,
    )

    assert receipt.scope is TaskActionSupervisionScope.FACTUAL_ACTION
    assert receipt.official_action_loss_enabled is True
    assert receipt.to_dict()["schema"] == TASK_ACTION_SUPERVISION_SCHEMA
    require_factual_action_supervision(receipt)


@pytest.mark.parametrize(
    ("candidate_task_key", "candidate_instruction"),
    [
        ("lift_blue_block_table", "pick up the blue block from the table"),
        ("move_slider_right", "slide it right"),
    ],
)
def test_changed_language_is_representation_only(
    candidate_task_key: str,
    candidate_instruction: str,
) -> None:
    receipt = task_action_supervision_receipt(
        sample_key="episode/frame",
        source_task_key="move_slider_right",
        source_instruction="move the sliding door to the right",
        candidate_task_key=candidate_task_key,
        candidate_instruction=candidate_instruction,
        source_action_targets_sha256=ACTION_SHA256,
        candidate_action_targets_sha256=ACTION_SHA256,
    )

    assert receipt.scope is TaskActionSupervisionScope.REPRESENTATION_ONLY
    assert receipt.official_action_loss_enabled is False
    with pytest.raises(ValueError, match="immutable source task and instruction"):
        require_factual_action_supervision(receipt)


def test_same_observation_candidate_cannot_change_action_targets() -> None:
    with pytest.raises(ValueError, match="changed immutable action targets"):
        task_action_supervision_receipt(
            sample_key="episode/frame",
            source_task_key="move_slider_right",
            source_instruction="move the sliding door to the right",
            candidate_task_key="lift_blue_block_table",
            candidate_instruction="pick up the blue block from the table",
            source_action_targets_sha256=ACTION_SHA256,
            candidate_action_targets_sha256="b" * 64,
        )
