from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from picf_next.lingbot_native.empirical_producers import (
    _g2_metrics,
    _g5_metrics,
    _hota_association_accuracy,
    _load_npz,
    _row_collapse_rate,
    g2_episode_arrays_from_native_outputs,
    g3_episode_arrays_from_native_outputs,
    g4_episode_arrays_from_trials,
    g5_episode_arrays_from_action_curves,
    g6_episode_arrays_from_calvin_rollouts,
    write_empirical_episode_artifact,
)
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
)


def _native_sequence_fixture() -> tuple[
    NativeSequencePredictions,
    NativeSequencePredictions,
    NativeSequenceTargets,
]:
    batch, time, tokens, rows = 1, 3, 4, 2
    support = torch.zeros(batch, time, tokens, rows)
    ownership = torch.softmax(torch.zeros(batch, time, tokens, rows + 1), dim=-1)
    candidate = NativeSequencePredictions(
        support_logits=support,
        ownership=ownership,
        existence_logits=torch.zeros(batch, time, rows),
        task_relevance_logits=torch.zeros(batch, rows),
        dense_task_grounding_logits=torch.zeros(batch, time, tokens),
    )
    control = NativeSequencePredictions(
        support_logits=torch.full_like(support, -1),
        ownership=ownership.clone(),
        existence_logits=torch.full((batch, time, rows), -1.0),
        task_relevance_logits=torch.full((batch, rows), -1.0),
        dense_task_grounding_logits=torch.full((batch, time, tokens), -1.0),
    )
    masks = torch.zeros(batch, time, rows, tokens)
    masks[:, :, 0, :2] = 1
    masks[:, :, 1, 2:] = 1
    targets = NativeSequenceTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=torch.ones(batch, time, rows),
        existence_valid=torch.ones(batch, time, rows, dtype=torch.bool),
        task_relevance=torch.ones(batch, rows),
        task_valid=torch.ones(batch, rows, dtype=torch.bool),
        track_valid=torch.ones(batch, rows, dtype=torch.bool),
        capacity_censored=torch.zeros(batch, rows, dtype=torch.bool),
        token_observed_fraction=torch.ones(batch, time, tokens),
        inventory_exhaustive=torch.ones(batch, time, dtype=torch.bool),
    )
    return candidate, control, targets


def test_native_sequence_projection_is_detached_and_axis_exact() -> None:
    candidate, control, targets = _native_sequence_fixture()

    g2 = g2_episode_arrays_from_native_outputs(
        candidate=candidate,
        generic_memory=control,
        targets=targets,
        batch_index=0,
    )
    g3 = g3_episode_arrays_from_native_outputs(
        candidate=candidate,
        reset_memory=control,
        targets=targets,
        state_ages=torch.asarray([[1, 8, 32]], dtype=torch.long),
        batch_index=0,
    )

    assert g2["c_support"].shape == (3, 4, 2)
    assert np.allclose(g2["c_support"], 0.5)
    assert g3["state_age"].tolist() == [1, 8, 32]
    assert all(isinstance(value, np.ndarray) for value in (*g2.values(), *g3.values()))


def test_g2_no_object_episode_contributes_only_existence_calibration() -> None:
    time, tokens, rows = 2, 4, 2
    arrays = {
        "c_support": np.zeros((time, tokens, rows), dtype=np.float32),
        "m_support": np.full((time, tokens, rows), 0.5, dtype=np.float32),
        "target_masks": np.zeros((time, 0, tokens), dtype=np.float32),
        "mask_valid": np.zeros((time, 0, tokens), dtype=np.bool_),
        "c_existence": np.zeros((time, rows), dtype=np.float32),
        "m_existence": np.full((time, rows), 0.5, dtype=np.float32),
        "target_existence": np.zeros((time, 0), dtype=np.float32),
        "existence_valid": np.zeros((time, 0), dtype=np.bool_),
        "c_task_relevance": np.zeros(rows, dtype=np.float32),
        "m_task_relevance": np.zeros(rows, dtype=np.float32),
        "c_dense_task_grounding": np.zeros((time, tokens), dtype=np.float32),
        "m_dense_task_grounding": np.zeros((time, tokens), dtype=np.float32),
        "target_task_relevance": np.zeros(0, dtype=np.float32),
        "task_valid": np.zeros(0, dtype=np.bool_),
        "track_valid": np.zeros(0, dtype=np.bool_),
        "capacity_censored": np.zeros(0, dtype=np.bool_),
        "inventory_exhaustive": np.ones(time, dtype=np.bool_),
    }

    metrics = _g2_metrics(arrays)

    assert metrics == {"existence_calibration_error": (0.0, None)}


def test_exclusive_native_metrics_use_categorical_ownership_and_dense_task() -> None:
    candidate, control, targets = _native_sequence_fixture()
    target_masks = targets.masks
    target_ownership = target_masks.permute(0, 1, 3, 2)
    context = 1 - target_ownership.sum(dim=-1, keepdim=True)
    candidate_ownership = torch.cat((target_ownership, context), dim=-1)
    control_ownership = torch.full_like(candidate_ownership, 1 / candidate_ownership.shape[-1])
    candidate = NativeSequencePredictions(
        support_logits=torch.full_like(candidate.support_logits, -20),
        ownership=candidate_ownership,
        existence_logits=candidate.existence_logits,
        task_relevance_logits=torch.full_like(candidate.task_relevance_logits, 8),
        dense_task_grounding_logits=torch.full_like(candidate.dense_task_grounding_logits, 8),
    )
    control = NativeSequencePredictions(
        support_logits=torch.full_like(control.support_logits, 20),
        ownership=control_ownership,
        existence_logits=control.existence_logits,
        task_relevance_logits=torch.full_like(control.task_relevance_logits, -8),
        dense_task_grounding_logits=torch.full_like(control.dense_task_grounding_logits, -8),
    )
    exclusive_targets = NativeSequenceTargets(
        masks=targets.masks,
        mask_valid=targets.mask_valid,
        existence=targets.existence,
        existence_valid=targets.existence_valid,
        task_relevance=targets.task_relevance,
        task_valid=targets.task_valid,
        track_valid=targets.track_valid,
        capacity_censored=targets.capacity_censored,
        token_observed_fraction=targets.token_observed_fraction,
        inventory_exhaustive=targets.inventory_exhaustive,
        exclusive_ownership=True,
    )

    arrays = g2_episode_arrays_from_native_outputs(
        candidate=candidate,
        generic_memory=control,
        targets=exclusive_targets,
        batch_index=0,
    )
    metrics = _g2_metrics(arrays)

    np.testing.assert_allclose(arrays["c_support"], target_ownership[0].numpy())
    assert metrics["object_mask_C_vs_M"][0] > metrics["object_mask_C_vs_M"][1]
    assert metrics["dense_task_grounding_C_vs_M"][0] > metrics["dense_task_grounding_C_vs_M"][1]


def test_dense_task_metric_retains_capacity_censored_visible_targets() -> None:
    candidate, control, targets = _native_sequence_fixture()
    candidate_dense = torch.full_like(candidate.dense_task_grounding_logits, -8)
    candidate_dense[:, :, 2:] = 8
    candidate = NativeSequencePredictions(
        support_logits=candidate.support_logits,
        ownership=candidate.ownership,
        existence_logits=candidate.existence_logits,
        task_relevance_logits=candidate.task_relevance_logits,
        dense_task_grounding_logits=candidate_dense,
    )
    targets = NativeSequenceTargets(
        masks=targets.masks,
        mask_valid=targets.mask_valid,
        existence=targets.existence,
        existence_valid=targets.existence_valid,
        task_relevance=torch.tensor([[0.0, 1.0]]),
        task_valid=targets.task_valid,
        track_valid=targets.track_valid,
        capacity_censored=torch.tensor([[False, True]]),
        token_observed_fraction=targets.token_observed_fraction,
        inventory_exhaustive=targets.inventory_exhaustive,
    )

    arrays = g2_episode_arrays_from_native_outputs(
        candidate=candidate,
        generic_memory=control,
        targets=targets,
        batch_index=0,
    )
    metrics = _g2_metrics(arrays)

    assert metrics["dense_task_grounding_C_vs_M"][0] > metrics["dense_task_grounding_C_vs_M"][1]


def test_hota_association_penalizes_a_mid_sequence_row_swap() -> None:
    time, tokens, rows = 4, 4, 2
    masks = np.zeros((time, rows, tokens), dtype=np.float32)
    masks[:, 0, :2] = 1
    masks[:, 1, 2:] = 1
    stable = np.transpose(masks, (0, 2, 1)).copy()
    switched = stable.copy()
    switched[2:] = switched[2:, :, ::-1]
    valid = np.ones_like(masks, dtype=np.bool_)

    stable_score = _hota_association_accuracy(
        support=stable,
        target_masks=masks,
        mask_valid=valid,
    )
    switched_score = _hota_association_accuracy(
        support=switched,
        target_masks=masks,
        mask_valid=valid,
    )

    assert stable_score == pytest.approx(1.0)
    assert switched_score < stable_score


def test_spectral_collapse_does_not_penalize_complementary_object_rows() -> None:
    support = np.asarray([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    valid = np.ones((1, 2, 4), dtype=np.bool_)

    distinct = _row_collapse_rate(
        support=support,
        mask_valid=valid,
        assignment={0: 0, 1: 1},
    )
    duplicated = _row_collapse_rate(
        support=np.repeat(support[:, :, :1], 2, axis=2),
        mask_valid=valid,
        assignment={0: 0, 1: 1},
    )

    assert distinct == pytest.approx(0.0)
    assert duplicated == pytest.approx(0.5)


def test_episode_writer_and_g5_metric_reject_malformed_inputs(tmp_path: Path) -> None:
    arrays = g4_episode_arrays_from_trials(
        same_entity_similarity=np.ones(2, dtype=np.float32),
        hard_negative_similarity=np.zeros(2, dtype=np.float32),
        all_available_quality=np.ones(2, dtype=np.float32),
        missing_modality_quality=np.ones(2, dtype=np.float32),
        corrupt_modality_quality=np.ones(2, dtype=np.float32),
        whole_static_omission_trial=np.asarray([True, False]),
    )
    artifact = tmp_path / "g4.npz"
    write_empirical_episode_artifact(artifact, gate="G4", arrays=arrays)
    assert set(_load_npz(artifact, gate="G4")) == set(arrays)
    with pytest.raises(FileExistsError):
        write_empirical_episode_artifact(artifact, gate="G4", arrays=arrays)

    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, **arrays, unexpected=np.ones(1))
    with pytest.raises(ValueError, match="arrays differ from the frozen schema"):
        _load_npz(malformed, gate="G4")

    steps = np.asarray([0, 10, 20], dtype=np.int64)
    curves = {
        "steps": steps,
        "action_loss_a": np.ones(3, dtype=np.float32),
        "action_loss_h": np.ones(3, dtype=np.float32),
        "action_loss_m": np.ones(3, dtype=np.float32),
        "action_loss_o": np.ones(3, dtype=np.float32),
        "action_loss_c": np.ones(3, dtype=np.float32),
        "action_loss_c_row_intervened": np.ones(3, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="must be finite and positive"):
        _g5_metrics(curves, metric_config={"action_loss_threshold": 0.0})


def test_action_curve_and_calvin_builders_freeze_arm_sets() -> None:
    steps = np.asarray([0, 10, 20], dtype=np.int64)
    curves = {
        arm: np.full(3, 0.1 + index * 0.1, dtype=np.float32)
        for index, arm in enumerate(("A", "H", "M", "O", "C"))
    }
    g5 = g5_episode_arrays_from_action_curves(
        steps=steps,
        action_loss_by_arm=curves,
        row_intervened_action_loss=np.ones(3, dtype=np.float32),
    )
    assert set(g5) == {
        "steps",
        "action_loss_a",
        "action_loss_h",
        "action_loss_m",
        "action_loss_o",
        "action_loss_c",
        "action_loss_c_row_intervened",
    }
    with pytest.raises(ValueError, match="exact A/H/M/O/C"):
        g5_episode_arrays_from_action_curves(
            steps=steps,
            action_loss_by_arm={"A": curves["A"]},
            row_intervened_action_loss=np.ones(3, dtype=np.float32),
        )

    g6 = g6_episode_arrays_from_calvin_rollouts(
        sequence_length=5,
        successful_prefix_by_arm={"A": 1, "O": 2, "C": 4},
        row_intervened_successful_prefix=0,
        recovery_o=np.asarray([False, True]),
        recovery_c=np.asarray([True, True]),
        reset_session_isolation=True,
    )
    assert g6["successful_prefix_c"].tolist() == [4]
    with pytest.raises(TypeError, match="frozen value types"):
        g6_episode_arrays_from_calvin_rollouts(
            sequence_length=5,
            successful_prefix_by_arm={"A": 1, "C": 4},
            row_intervened_successful_prefix=0,
            recovery_o=np.asarray([False, True]),
            recovery_c=np.asarray([True, True]),
            reset_session_isolation=True,
        )
