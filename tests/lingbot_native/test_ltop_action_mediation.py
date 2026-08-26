from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch

from picf_next.lingbot_native.host import ObjectReadActionIntervention
from picf_next.lingbot_native.ltop_action_mediation import (
    LTOPActionArmKind,
    LTOPActionReceipt,
    OfflineLTOPActionTargets,
    build_label_blind_ltop_action_arms,
    direct_posterior_action_row_visibility,
    score_offline_ltop_action_mediation,
    seal_ltop_action_receipt,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_ltop_action_arms_cover_factual_every_row_and_blocked_controls() -> None:
    arms = build_label_blind_ltop_action_arms(batch_size=2, capacity=4, device="cpu")

    assert len(arms) == 11
    assert [arm.name for arm in arms] == [
        "factual",
        "factual-repeat",
        "remove-row-0",
        "remove-row-1",
        "remove-row-2",
        "remove-row-3",
        "blocked",
        "blocked-remove-row-0",
        "blocked-remove-row-1",
        "blocked-remove-row-2",
        "blocked-remove-row-3",
    ]
    assert arms[0].object_read_action_intervention is ObjectReadActionIntervention.FACTUAL
    assert arms[6].object_read_action_intervention is ObjectReadActionIntervention.BLOCKED
    assert not arms[3].object_read_source_row_visible[:, 1].any()
    assert not arms[9].object_read_source_row_visible[:, 2].any()


def test_forward_receipt_schema_contains_no_target_metadata() -> None:
    field_names = {field.name for field in dataclasses.fields(LTOPActionReceipt)}

    assert not {"target", "target_row", "matched_distractor", "sidecar"} & field_names
    assert {"active_action_mask", "active_action_mask_sha256"} <= field_names


def test_direct_posterior_arms_block_the_entire_row_surface() -> None:
    arms = build_label_blind_ltop_action_arms(batch_size=2, capacity=4, device="cpu")
    table = {arm.name: arm for arm in arms}

    assert direct_posterior_action_row_visibility(table["factual"]).all()
    assert not direct_posterior_action_row_visibility(table["remove-row-2"])[:, 2].any()
    assert not direct_posterior_action_row_visibility(table["blocked"]).any()
    assert not direct_posterior_action_row_visibility(table["blocked-remove-row-2"]).any()


def test_ltop_action_receipt_detaches_output_and_binds_digests() -> None:
    arm = build_label_blind_ltop_action_arms(
        batch_size=2,
        capacity=4,
        device="cpu",
    )[0]
    output = torch.randn(2, 3, 7, requires_grad=True)
    joint_mask = torch.ones_like(output, dtype=torch.bool)
    joint_mask[:, :, -1] = False
    action_is_pad = torch.tensor(
        [[False, True, False], [False, False, False]],
        dtype=torch.bool,
    )

    receipt = seal_ltop_action_receipt(
        prompt_name="prompt-a",
        sample_keys=("sample-a", "sample-b"),
        arm=arm,
        deploy_inputs_sha256=_sha("inputs"),
        inference_randomness_sha256=_sha("noise"),
        action_output=output,
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
    )

    assert not receipt.action_output.requires_grad
    assert receipt.action_output.data_ptr() != output.data_ptr()
    assert torch.equal(
        receipt.active_action_mask,
        joint_mask & ~action_is_pad[..., None],
    )
    assert receipt.active_action_mask.data_ptr() != joint_mask.data_ptr()
    assert receipt.arm_kind is LTOPActionArmKind.FACTUAL
    assert len(receipt.active_action_mask_sha256) == 64
    assert len(receipt.action_output_sha256) == 64


def test_ltop_action_receipt_binds_the_executed_direct_visibility() -> None:
    arm = build_label_blind_ltop_action_arms(
        batch_size=1,
        capacity=2,
        device="cpu",
    )[-1]
    direct_visibility = direct_posterior_action_row_visibility(arm)

    receipt = seal_ltop_action_receipt(
        prompt_name="prompt-a",
        sample_keys=("sample-a",),
        arm=arm,
        deploy_inputs_sha256=_sha("inputs"),
        inference_randomness_sha256=_sha("noise"),
        action_output=torch.randn(1, 2, 3),
        joint_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
        executed_source_row_visible=direct_visibility,
    )

    wrong_visibility_receipt = seal_ltop_action_receipt(
        prompt_name="prompt-a",
        sample_keys=("sample-a",),
        arm=arm,
        deploy_inputs_sha256=_sha("inputs"),
        inference_randomness_sha256=_sha("noise"),
        action_output=torch.randn(1, 2, 3),
        joint_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
        executed_source_row_visible=arm.object_read_source_row_visible,
    )
    assert not direct_visibility.any()
    assert receipt.source_visibility_sha256 != wrong_visibility_receipt.source_visibility_sha256


def test_ltop_action_receipt_hashes_bfloat16_output() -> None:
    arm = build_label_blind_ltop_action_arms(
        batch_size=1,
        capacity=2,
        device="cpu",
    )[0]

    receipt = seal_ltop_action_receipt(
        prompt_name="prompt-a",
        sample_keys=("sample-a",),
        arm=arm,
        deploy_inputs_sha256=_sha("inputs"),
        inference_randomness_sha256=_sha("noise"),
        action_output=torch.randn(1, 2, 3, dtype=torch.bfloat16),
        joint_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        action_is_pad=torch.zeros(1, 2, dtype=torch.bool),
    )

    assert receipt.action_output.dtype is torch.bfloat16
    assert len(receipt.action_output_sha256) == 64


def test_ltop_action_receipt_rejects_sample_without_active_action() -> None:
    arm = build_label_blind_ltop_action_arms(
        batch_size=2,
        capacity=2,
        device="cpu",
    )[0]
    joint_mask = torch.ones(2, 2, 3, dtype=torch.bool)
    joint_mask[1] = False

    with pytest.raises(ValueError, match="every LTOP action receipt sample"):
        seal_ltop_action_receipt(
            prompt_name="prompt-a",
            sample_keys=("sample-a", "sample-b"),
            arm=arm,
            deploy_inputs_sha256=_sha("inputs"),
            inference_randomness_sha256=_sha("noise"),
            action_output=torch.zeros(2, 2, 3),
            joint_mask=joint_mask,
            action_is_pad=torch.zeros(2, 2, dtype=torch.bool),
        )


def _seal_receipts(
    arm_outputs: dict[str, torch.Tensor],
    *,
    joint_mask: torch.Tensor,
    action_is_pad: torch.Tensor,
    capacity: int = 4,
    visibility_overrides: dict[str, torch.Tensor] | None = None,
    joint_mask_overrides: dict[str, torch.Tensor] | None = None,
) -> tuple[LTOPActionReceipt, ...]:
    arms = build_label_blind_ltop_action_arms(
        batch_size=joint_mask.shape[0],
        capacity=capacity,
        device="cpu",
    )
    visibility_overrides = {} if visibility_overrides is None else visibility_overrides
    joint_mask_overrides = {} if joint_mask_overrides is None else joint_mask_overrides
    return tuple(
        seal_ltop_action_receipt(
            prompt_name="prompt-a",
            sample_keys=tuple(f"sample-{index}" for index in range(joint_mask.shape[0])),
            arm=arm,
            deploy_inputs_sha256=_sha("inputs"),
            inference_randomness_sha256=_sha("noise"),
            action_output=arm_outputs[arm.name],
            joint_mask=joint_mask_overrides.get(arm.name, joint_mask),
            action_is_pad=action_is_pad,
            executed_source_row_visible=visibility_overrides.get(
                arm.name,
                direct_posterior_action_row_visibility(arm),
            ),
        )
        for arm in arms
    )


def _default_arm_outputs(*, capacity: int = 4) -> dict[str, torch.Tensor]:
    arms = build_label_blind_ltop_action_arms(
        batch_size=2,
        capacity=capacity,
        device="cpu",
    )
    factual = torch.zeros(2, 1, 2)
    outputs = {}
    for arm in arms:
        value = factual.clone()
        if arm.kind is LTOPActionArmKind.ROW_REMOVAL:
            assert arm.row_index is not None
            value[:, :, 0] = float(arm.row_index + 1)
        elif arm.kind in {
            LTOPActionArmKind.BLOCKED,
            LTOPActionArmKind.BLOCKED_ROW_REMOVAL,
        }:
            value[:, :, 1] = 0.25
        outputs[arm.name] = value
    return outputs


def _receipts(*, capacity: int = 4) -> tuple[LTOPActionReceipt, ...]:
    return _seal_receipts(
        _default_arm_outputs(capacity=capacity),
        joint_mask=torch.ones(2, 1, 2, dtype=torch.bool),
        action_is_pad=torch.zeros(2, 1, dtype=torch.bool),
        capacity=capacity,
    )


def test_offline_ltop_action_score_reports_factual_selectivity_and_all_block_effect() -> None:
    score = score_offline_ltop_action_mediation(
        _receipts(),
        targets=OfflineLTOPActionTargets(
            prompt_name="prompt-a",
            sample_keys=("sample-0", "sample-1"),
            target_rows=torch.tensor([3, 2]),
            matched_distractor_rows=torch.tensor([0, 0]),
        ),
        capacity=4,
    )

    assert torch.equal(score.replay_floor_rms, torch.zeros(2))
    assert (score.factual_target_minus_distractor > 0).all()
    assert (score.factual_all_posterior_block_effect_rms > 0).all()
    assert score.blocked_placebo_integrity_verified
    assert torch.equal(score.active_action_counts, torch.full((2,), 2))
    assert (
        score.factual_selectivity_over_all_posterior_block
        == score.factual_target_minus_distractor / score.factual_all_posterior_block_effect_rms
    ).all()
    score_fields = {field.name for field in dataclasses.fields(score)}
    assert (
        not {
            "blocked_target_effect_rms",
            "blocked_target_minus_distractor",
            "blocked_path_difference_in_differences",
            "positive_blocked_path_did_count",
        }
        & score_fields
    )


def test_offline_ltop_action_score_rejects_non_fixed_randomness() -> None:
    receipts = list(_receipts())
    receipt = receipts[-1]
    receipts[-1] = dataclasses.replace(receipt, inference_randomness_sha256=_sha("other"))

    with pytest.raises(ValueError, match="fixed randomness"):
        score_offline_ltop_action_mediation(
            receipts,
            targets=OfflineLTOPActionTargets(
                prompt_name="prompt-a",
                sample_keys=("sample-0", "sample-1"),
                target_rows=torch.tensor([3, 2]),
                matched_distractor_rows=torch.tensor([0, 0]),
            ),
            capacity=4,
        )


def test_offline_ltop_action_score_rejects_arm_specific_active_mask() -> None:
    joint_mask = torch.ones(2, 1, 2, dtype=torch.bool)
    different_joint_mask = joint_mask.clone()
    different_joint_mask[:, :, 1] = False
    receipts = _seal_receipts(
        _default_arm_outputs(),
        joint_mask=joint_mask,
        action_is_pad=torch.zeros(2, 1, dtype=torch.bool),
        joint_mask_overrides={"remove-row-1": different_joint_mask},
    )

    with pytest.raises(ValueError, match="active action masks differ"):
        score_offline_ltop_action_mediation(
            receipts,
            targets=OfflineLTOPActionTargets(
                prompt_name="prompt-a",
                sample_keys=("sample-0", "sample-1"),
                target_rows=torch.tensor([3, 2]),
                matched_distractor_rows=torch.tensor([0, 0]),
            ),
            capacity=4,
        )


@pytest.mark.parametrize("corruption", ["output", "visibility"])
def test_offline_ltop_action_score_rejects_inconsistent_blocked_placebo(
    corruption: str,
) -> None:
    outputs = _default_arm_outputs()
    visibility_overrides = None
    if corruption == "output":
        outputs["blocked-remove-row-2"] = outputs["blocked-remove-row-2"].clone()
        outputs["blocked-remove-row-2"][:, :, 0] = 1.0
    else:
        visibility_overrides = {
            "blocked-remove-row-2": torch.ones(2, 4, dtype=torch.bool),
        }
    receipts = _seal_receipts(
        outputs,
        joint_mask=torch.ones(2, 1, 2, dtype=torch.bool),
        action_is_pad=torch.zeros(2, 1, dtype=torch.bool),
        visibility_overrides=visibility_overrides,
    )

    with pytest.raises(ValueError, match="blocked.*(?:output|visibility)|source visibility"):
        score_offline_ltop_action_mediation(
            receipts,
            targets=OfflineLTOPActionTargets(
                prompt_name="prompt-a",
                sample_keys=("sample-0", "sample-1"),
                target_rows=torch.tensor([3, 2]),
                matched_distractor_rows=torch.tensor([0, 0]),
            ),
            capacity=4,
        )


def _calvin_active_surface() -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    executable_dimensions = (*range(14, 20), 28)
    joint_mask = torch.zeros(1, 2, 55, dtype=torch.bool)
    joint_mask[:, :, executable_dimensions] = True
    action_is_pad = torch.tensor([[False, True]], dtype=torch.bool)
    return joint_mask, action_is_pad, executable_dimensions


def test_inactive_chart_dimensions_and_padded_timesteps_have_zero_effect() -> None:
    capacity = 4
    arms = build_label_blind_ltop_action_arms(
        batch_size=1,
        capacity=capacity,
        device="cpu",
    )
    joint_mask, action_is_pad, executable_dimensions = _calvin_active_surface()
    inactive_dimensions = tuple(
        dimension for dimension in range(55) if dimension not in executable_dimensions
    )
    assert len(inactive_dimensions) == 48
    outputs = {}
    for arm in arms:
        output = torch.zeros(1, 2, 55)
        if arm.kind is LTOPActionArmKind.ROW_REMOVAL:
            assert arm.row_index is not None
            output[:, :, inactive_dimensions] = float(arm.row_index + 1)
            output[:, 1, executable_dimensions] = float(arm.row_index + 1)
        outputs[arm.name] = output
    score = score_offline_ltop_action_mediation(
        _seal_receipts(
            outputs,
            joint_mask=joint_mask,
            action_is_pad=action_is_pad,
            capacity=capacity,
        ),
        targets=OfflineLTOPActionTargets(
            prompt_name="prompt-a",
            sample_keys=("sample-0",),
            target_rows=torch.tensor([3]),
            matched_distractor_rows=torch.tensor([0]),
        ),
        capacity=capacity,
    )

    assert torch.equal(score.active_action_counts, torch.tensor([7]))
    assert torch.equal(score.replay_floor_rms, torch.zeros(1))
    assert torch.equal(score.factual_all_posterior_block_effect_rms, torch.zeros(1))
    assert torch.equal(score.factual_target_effect_rms, torch.zeros(1))
    assert torch.equal(score.factual_distractor_effect_rms, torch.zeros(1))
    assert torch.equal(score.factual_target_minus_distractor, torch.zeros(1))
    assert torch.equal(score.factual_selectivity_over_all_posterior_block, torch.zeros(1))


def test_active_target_change_and_all_posterior_block_effect_are_detected() -> None:
    capacity = 4
    arms = build_label_blind_ltop_action_arms(
        batch_size=1,
        capacity=capacity,
        device="cpu",
    )
    joint_mask, action_is_pad, executable_dimensions = _calvin_active_surface()
    outputs = {}
    for arm in arms:
        output = torch.zeros(1, 2, 55)
        if arm.kind is LTOPActionArmKind.ROW_REMOVAL:
            assert arm.row_index is not None
            active_change = 1.0 if arm.row_index == 3 else 0.25 if arm.row_index == 0 else 0.0
            output[:, 0, executable_dimensions] = active_change
        elif arm.kind in {
            LTOPActionArmKind.BLOCKED,
            LTOPActionArmKind.BLOCKED_ROW_REMOVAL,
        }:
            output[:, 0, executable_dimensions] = 2.0
        outputs[arm.name] = output
    score = score_offline_ltop_action_mediation(
        _seal_receipts(
            outputs,
            joint_mask=joint_mask,
            action_is_pad=action_is_pad,
            capacity=capacity,
        ),
        targets=OfflineLTOPActionTargets(
            prompt_name="prompt-a",
            sample_keys=("sample-0",),
            target_rows=torch.tensor([3]),
            matched_distractor_rows=torch.tensor([0]),
        ),
        capacity=capacity,
    )

    assert score.factual_target_effect_rms.item() == pytest.approx(1.0)
    assert score.factual_distractor_effect_rms.item() == pytest.approx(0.25)
    assert score.factual_target_minus_distractor.item() == pytest.approx(0.75)
    assert score.factual_all_posterior_block_effect_rms.item() == pytest.approx(2.0)
    assert score.factual_target_effect_over_all_posterior_block.item() == pytest.approx(0.5)
    assert score.factual_distractor_effect_over_all_posterior_block.item() == pytest.approx(0.125)
    assert score.factual_selectivity_over_all_posterior_block.item() == pytest.approx(0.375)


def test_offline_targets_reject_target_equal_to_distractor() -> None:
    with pytest.raises(ValueError, match="must differ"):
        OfflineLTOPActionTargets(
            prompt_name="prompt-a",
            sample_keys=("sample-a", "sample-b"),
            target_rows=torch.tensor([1, 2]),
            matched_distractor_rows=torch.tensor([1, 0]),
        )
