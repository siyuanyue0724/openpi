# ruff: noqa: E402  # Optional torch gate must precede tool import.
from __future__ import annotations

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)

import torch

from tools.audit_molmoact2_m4_action_interventions import (
    _aggregate,
    _association_target_row_diagnostics,
    _AuditIdentityAttribution,
    _clone_object_belief,
    _controlled_target_row_summary,
    _extend_plan_for_read_only_audit,
    _intervention_indices,
    _measurement_matches_age_selection,
    _measurement_summary,
    _old_row_trajectory_records,
    _PreparedActionStep,
    _render_action_visual,
    _render_old_row_trajectories,
    _rows_at_least_one_reference_step_old,
    _runtime_visual_snapshot,
    _selection_outcome,
    _source_frame_flow_seed_contract,
    _target_address_permutation,
    _task_protocol_inventory_for_sample,
    _tensor_sha256,
    _validate_controlled_observation_pair,
    _validate_extended_plan_prefix,
)


def test_task_protocol_inventory_uses_segment_scoped_sidecar_read() -> None:
    inventory = (
        "movable/block_blue",
        "movable/block_pink",
        "movable/block_red",
        "part/table/button_link",
        "part/table/drawer_link",
        "part/table/led_link",
        "part/table/light_link",
        "part/table/plank_link",
        "part/table/slide_link",
        "part/table/switch_link",
    )

    class SegmentSidecar:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def __call__(self, segment_index: int, global_index: int) -> SimpleNamespace:
            self.calls.append((segment_index, global_index))
            return SimpleNamespace(identity_keys=inventory)

        def source_frame(self, _global_index: int) -> None:
            raise AssertionError("language-frame sidecars must not use source_frame")

    sidecar = SegmentSidecar()
    sample = SimpleNamespace(record=SimpleNamespace(task_index=7, global_index=101))

    result = _task_protocol_inventory_for_sample(sidecar, sample)

    assert set(result) == set(inventory)
    assert sidecar.calls == [(7, 101)]


def test_selection_outcome_distinguishes_complete_partial_and_empty_searches() -> None:
    assert (
        _selection_outcome(
            ({"selected_samples": 2}, {"selected_samples": 2}),
            samples_per_rank=2,
        )
        == "REQUESTED_SAMPLE_COUNT_SATISFIED"
    )
    assert (
        _selection_outcome(
            ({"selected_samples": 1}, {"selected_samples": 0}),
            samples_per_rank=2,
        )
        == "PARTIAL_ELIGIBLE_SAMPLE_COVERAGE"
    )
    assert (
        _selection_outcome(
            ({"selected_samples": 0}, {"selected_samples": 0}),
            samples_per_rank=2,
        )
        == "NO_ELIGIBLE_SAMPLE_IN_SEARCH_WINDOW"
    )
    with pytest.raises(ValueError, match="requires rank search results"):
        _selection_outcome((), samples_per_rank=1)


def test_intervention_indices_swap_two_strongest_valid_rows() -> None:
    valid = torch.tensor([[False, True, True, False, True]])
    log_prior = torch.tensor([[0.0, -0.5, -0.1, 0.0, -0.2]])

    permutation, removed, pair = _intervention_indices(valid, log_prior)

    assert pair == (2, 4)
    assert permutation.tolist() == [0, 1, 4, 3, 2]
    assert removed.tolist() == [[False, False, True, False, False]]


def test_intervention_indices_reject_one_valid_row() -> None:
    with pytest.raises(ValueError, match="at least two valid objects"):
        _intervention_indices(
            torch.tensor([[False, True]]),
            torch.tensor([[0.0, -0.1]]),
        )


def test_target_address_permutation_swaps_targets_with_strong_controls() -> None:
    valid = torch.tensor([[True, True, False, True, True]])
    log_prior = torch.tensor([[-0.3, -0.1, 0.0, -0.2, -0.4]])
    targets = torch.tensor([[True, False, False, False, True]])

    permutation, controls = _target_address_permutation(valid, log_prior, targets)

    assert controls == (1, 3)
    assert permutation.tolist() == [1, 0, 2, 4, 3]

    with pytest.raises(ValueError, match="remain valid"):
        _target_address_permutation(
            valid,
            log_prior,
            torch.tensor([[False, False, True, False, False]]),
        )
    with pytest.raises(ValueError, match="lacks distinct"):
        _target_address_permutation(
            torch.tensor([[True, False]]),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[True, False]]),
        )


def test_counterfactual_belief_clone_has_no_shared_tensor_storage() -> None:
    from picf_next.models.temporal import ObjectBeliefBatch

    belief = ObjectBeliefBatch(
        address_mean=torch.ones(1, 2, 3),
        content_mean=torch.ones(1, 2, 4),
        geometry_mean=torch.ones(1, 2, 2),
        geometry_covariance_diag=torch.ones(1, 2, 2),
        existence_logits=torch.ones(1, 2),
        visibility_given_existence_logits=torch.ones(1, 2),
        measurement_age_s=torch.ones(1, 2),
        valid=torch.ones(1, 2, dtype=torch.bool),
        age=torch.ones(1, 2, dtype=torch.long),
    )

    cloned = _clone_object_belief(belief)

    for name in (
        "address_mean",
        "content_mean",
        "geometry_mean",
        "geometry_covariance_diag",
        "existence_logits",
        "visibility_given_existence_logits",
        "measurement_age_s",
        "valid",
        "age",
    ):
        source = getattr(belief, name)
        changed = getattr(cloned, name)
        assert torch.equal(source, changed)
        assert source.data_ptr() != changed.data_ptr()


def test_controlled_observation_pair_changes_only_visual_embedding_tokens() -> None:
    planned = object()
    sample = SimpleNamespace(sample_key="sample-0")
    shared = {
        "action": torch.ones(1, 2, 3),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    clean_embeddings = torch.zeros(1, 3, 4)
    occluded_embeddings = clean_embeddings.clone()
    occluded_embeddings[:, (0, 2)] = 1
    clean_batch = {**shared, "inputs_embeds": clean_embeddings}
    occluded_batch = {**shared, "inputs_embeds": occluded_embeddings}
    common = {
        "optimizer_plan_step": 1,
        "planned_transition": planned,
        "sample": sample,
        "evidence": None,
        "final_belief": None,
        "flow_timesteps": torch.tensor([[0.5]]),
        "flow_noise": torch.zeros(1, 1, 1, 1),
        "action_condition_input_ids": torch.tensor([[3, 1, 3]]),
        "vision_patch_layout": ("same",),
    }

    result = _validate_controlled_observation_pair(
        _PreparedActionStep(policy_batch=clean_batch, **common),
        _PreparedActionStep(policy_batch=occluded_batch, **common),
        image_patch_token_id=3,
    )

    assert result["changed_policy_fields"] == ["inputs_embeds"]
    assert result["changed_visual_token_count"] == 2
    assert result["changed_nonvisual_token_count"] == 0
    assert result["nonvisual_policy_fields_exact"] is True
    bad = dict(occluded_batch)
    bad_embeddings = occluded_embeddings.clone()
    bad_embeddings[:, 1] = 1
    bad["inputs_embeds"] = bad_embeddings
    with pytest.raises(ValueError, match="outside image-patch tokens"):
        _validate_controlled_observation_pair(
            _PreparedActionStep(policy_batch=clean_batch, **common),
            _PreparedActionStep(policy_batch=bad, **common),
            image_patch_token_id=3,
        )


def test_controlled_target_uses_map_miss_not_soft_age_as_binary_gate() -> None:
    from picf_next.models.temporal import ObjectBeliefBatch
    from picf_next.posterior import MATCH_EVENT, MISS_EVENT

    def belief(measurement_age_s: torch.Tensor) -> ObjectBeliefBatch:
        return ObjectBeliefBatch(
            address_mean=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            content_mean=torch.zeros(1, 2, 2),
            geometry_mean=torch.zeros(1, 2, 1),
            geometry_covariance_diag=torch.ones(1, 2, 1),
            existence_logits=torch.ones(1, 2),
            visibility_given_existence_logits=torch.ones(1, 2),
            measurement_age_s=measurement_age_s,
            valid=torch.ones(1, 2, dtype=torch.bool),
            age=torch.ones(1, 2, dtype=torch.long),
        )

    clean_belief = belief(torch.zeros(1, 2))
    occluded_belief = belief(torch.tensor([[0.03, 0.0]]))
    clean_evidence = SimpleNamespace(
        object_valid=clean_belief.valid,
        dense_banks=(),
        dense_ownership=None,
    )
    occluded_evidence = SimpleNamespace(
        object_valid=torch.tensor([[False, True]]),
        dense_banks=(),
        dense_ownership=None,
    )
    common = {
        "optimizer_plan_step": 1,
        "planned_transition": object(),
        "sample": SimpleNamespace(sample_key="sample-0"),
        "policy_batch": {},
        "flow_timesteps": None,
        "flow_noise": None,
        "action_condition_input_ids": None,
        "vision_patch_layout": None,
    }
    clean = _PreparedActionStep(
        evidence=clean_evidence,
        final_belief=clean_belief,
        core_output=None,
        **common,
    )
    occluded = _PreparedActionStep(
        evidence=occluded_evidence,
        final_belief=occluded_belief,
        core_output=SimpleNamespace(
            posterior=SimpleNamespace(
                event_type=torch.tensor([[MISS_EVENT, MATCH_EVENT]]),
                null_probability=torch.tensor([[0.8, 0.2]]),
                match_probability=torch.tensor([[[0.1, 0.1], [0.7, 0.1]]]),
            )
        ),
        **common,
    )

    result = _controlled_target_row_summary(
        clean,
        occluded,
        (0,),
        reference_delta_t_s=1.0 / 30.0,
        clean_identity_attribution=_AuditIdentityAttribution(
            next_keys_by_row=(("object/0", None),),
            currently_measurable_identity_keys=("object/0",),
            current_set_match_by_identity={
                "object/0": {
                    "discovery_query": 1,
                    "matched_soft_iou": 0.75,
                }
            },
            track_conflicts=0,
            task_selection=None,
        ),
        target_identity_keys=("object/0",),
    )

    assert result["all_target_rows_map_missed"] is True
    assert result["all_target_rows_remain_in_posterior"] is True
    assert result["all_target_rows_action_exposed"] is False
    assert result["all_target_rows_expected_age_at_least_one_reference_step"] is False
    assert result["rows"][0]["clean_posterior_retained"] is True
    assert result["rows"][0]["occluded_posterior_retained"] is True
    assert result["rows"][0]["clean_action_exposed"] is True
    assert result["rows"][0]["occluded_action_exposed"] is False
    assert result["rows"][0]["runtime_event"] == "miss"
    assert result["rows"][0]["null_probability"] == pytest.approx(0.8)
    assert result["rows"][0]["total_match_probability"] == pytest.approx(0.2)
    assert result["rows"][0]["target_identity_key"] == "object/0"
    assert result["rows"][0]["clean_loss_side_set_match"]["matched_soft_iou"] == 0.75


def test_association_diagnostics_decompose_the_runtime_edge_equation() -> None:
    from picf_next.models.temporal import ObjectBeliefBatch

    predicted = ObjectBeliefBatch(
        address_mean=torch.tensor([[[1.0, 0.0]]]),
        content_mean=torch.zeros(1, 1, 2),
        geometry_mean=torch.zeros(1, 1, 1),
        geometry_covariance_diag=torch.full((1, 1, 1), 0.25),
        existence_logits=torch.tensor([[4.0]]),
        visibility_given_existence_logits=torch.tensor([[3.0]]),
        measurement_age_s=torch.zeros(1, 1),
        valid=torch.ones(1, 1, dtype=torch.bool),
        age=torch.ones(1, 1, dtype=torch.long),
    )
    ownership = torch.tensor(
        [
            [
                [0.7, 0.1, 0.2],
                [0.6, 0.1, 0.3],
                [0.1, 0.7, 0.2],
                [0.1, 0.6, 0.3],
            ]
        ]
    )
    discovery = SimpleNamespace(
        address_mean=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        geometry_mean=torch.tensor([[[0.0], [2.0]]]),
        geometry_variance=torch.full((1, 2, 1), 0.25),
        existence=torch.tensor([[0.9, 0.8]]),
        localization_confidence=torch.tensor([[0.8, 0.7]]),
        measurement_probability=torch.tensor([[0.72, 0.56]]),
        mask_quality=torch.tensor([[0.75, 0.65]]),
        ownership=ownership,
        ownership_logits=ownership.log(),
        query_features=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        token_valid=torch.ones(1, 4, dtype=torch.bool),
    )
    posterior = SimpleNamespace(
        prior_prediction=SimpleNamespace(belief=predicted),
        address_relation_logit_scale=torch.tensor(10.0),
        address_relation_logit_bias=torch.tensor(-2.7),
        match_probability=torch.tensor([[[0.85, 0.01]]]),
        null_probability=torch.tensor([[0.14]]),
        birth_probability=torch.tensor([[0.05, 0.04]]),
        observation_to_posterior=torch.tensor([[0, -1]]),
    )
    step = _PreparedActionStep(
        optimizer_plan_step=0,
        planned_transition=None,
        sample=None,
        evidence=None,
        final_belief=None,
        policy_batch={},
        flow_timesteps=None,
        flow_noise=None,
        action_condition_input_ids=None,
        vision_patch_layout=SimpleNamespace(
            rows=(
                (
                    SimpleNamespace(
                        image_key="observation.images.image",
                        start=0,
                        stop=4,
                        patches_per_crop=4,
                    ),
                ),
            )
        ),
        core_output=SimpleNamespace(
            discovery=discovery,
            posterior=posterior,
            projection=SimpleNamespace(
                spans=(SimpleNamespace(modality="molmo_vision_patch", start=0, stop=4),)
            ),
        ),
        temporal_config=SimpleNamespace(
            minimum_variance=1e-6,
            empty_bank_birth_to_clutter_prior_odds=1.0,
            recurrent_birth_to_clutter_prior_odds=0.1,
        ),
    )

    result = _association_target_row_diagnostics(step, posterior_row=0)

    assert result is not None
    assert result["map_query"] == 0
    assert result["top_queries"][0]["address_cosine"] == pytest.approx(1.0)
    assert result["top_queries"][0]["address_log_likelihood_ratio"] == pytest.approx(7.3)
    assert result["top_queries"][0]["query_is_mutual_map_for_target_row"] is True
    assert result["top_queries"][0]["ownership"]["total_ownership_mass"] == pytest.approx(1.5)
    assert result["top_queries"][0]["ownership"]["camera_ownership"]["observation.images.image"][
        "ownership_mass"
    ] == pytest.approx(1.5)
    assert result["top_queries"][0]["log_edge_odds"] > result["top_queries"][1]["log_edge_odds"]
    capacity = result["finite_capacity_projection"]
    expected_no_detection = (
        torch.sigmoid(torch.tensor(4.0))
        * (1.0 - torch.sigmoid(torch.tensor(3.0)))
        / (1.0 - torch.sigmoid(torch.tensor(4.0)) * torch.sigmoid(torch.tensor(3.0)))
    )
    assert capacity["existence_given_no_detection"] == pytest.approx(float(expected_no_detection))
    assert capacity["posterior_existence_before_capacity"] == pytest.approx(
        0.86 + 0.14 * float(expected_no_detection)
    )
    assert capacity["selected_as_existing"] is True
    assert capacity["higher_score_candidate_count"] == 0
    assert capacity["target_candidate_rank"] == 1
    assert capacity["finite_candidate_count"] == 2
    assert capacity["selected_candidates"][0] == {
        "candidate_index": 0,
        "kind": "existing",
        "local_index": 0,
        "score": pytest.approx(capacity["posterior_existence_before_capacity"]),
    }
    assert capacity["topk_boundary_score"] == pytest.approx(
        capacity["posterior_existence_before_capacity"]
    )


def test_read_only_plan_extension_preserves_every_checkpoint_batch() -> None:
    from picf_next.training.control import EpisodeSampleSequence, FrozenEpisodeStreamPlan

    reference = FrozenEpisodeStreamPlan(
        dataset_id="dataset",
        dataset_revision="revision",
        dataset_manifest_sha256="a" * 64,
        episodes=(
            EpisodeSampleSequence("episode-0", ("sample-0", "sample-1", "sample-2")),
            EpisodeSampleSequence("episode-1", ("sample-3", "sample-4")),
        ),
        comparison_id="comparison",
        seed=17,
        global_batch_size=2,
        total_steps=3,
    )

    extended = _extend_plan_for_read_only_audit(reference, total_steps=8)

    assert _validate_extended_plan_prefix(reference, extended) == 3
    assert extended.total_steps == 8
    assert extended.global_batch(3).optimizer_step == 3


def test_measurement_summary_uses_only_final_deploy_visible_belief() -> None:
    from picf_next.models.temporal import ObjectBeliefBatch

    action_valid = torch.tensor([[True, True, False]])
    belief_valid = torch.tensor([[True, True, True]])
    belief = ObjectBeliefBatch(
        address_mean=torch.zeros(1, 3, 2),
        content_mean=torch.zeros(1, 3, 2),
        geometry_mean=torch.zeros(1, 3, 1),
        geometry_covariance_diag=torch.zeros(1, 3, 1),
        existence_logits=torch.tensor([[2.0, 1.0, 0.0]]),
        visibility_given_existence_logits=torch.tensor([[1.0, -1.0, 0.0]]),
        measurement_age_s=torch.tensor([[0.0, 2.0 / 30.0, 0.0]]),
        valid=belief_valid,
        age=torch.tensor([[4, 7, 1]]),
    )
    step = _PreparedActionStep(
        optimizer_plan_step=0,
        planned_transition=None,
        sample=None,
        evidence=SimpleNamespace(object_valid=action_valid),
        final_belief=belief,
        policy_batch={},
        flow_timesteps=None,
        flow_noise=None,
        action_condition_input_ids=None,
        vision_patch_layout=None,
    )

    result = _measurement_summary(step, reference_delta_t_s=1.0 / 30.0)

    assert result["has_row_with_expected_age_at_least_one_reference_step"] is True
    assert result["positive_expected_age_rows"] == 1
    assert result["valid_rows"] == 2
    assert result["maximum_measurement_age_s"] == pytest.approx(2.0 / 30.0)
    assert result["maximum_measurement_age_reference_steps"] == pytest.approx(2.0)

    subframe = replace(
        belief,
        measurement_age_s=torch.tensor([[1.0e-6, 0.0, 0.0]]),
    )
    subframe_result = _measurement_summary(
        replace(step, final_belief=subframe),
        reference_delta_t_s=1.0 / 30.0,
    )
    assert subframe_result["positive_expected_age_rows"] == 1
    assert subframe_result["has_row_with_expected_age_at_least_one_reference_step"] is False

    with pytest.raises(ValueError, match="reference delta time"):
        _measurement_summary(step, reference_delta_t_s=0.0)


def test_old_row_intervention_uses_deploy_visible_expected_age() -> None:
    from picf_next.models.temporal import ObjectBeliefBatch

    belief = ObjectBeliefBatch(
        address_mean=torch.zeros(1, 4, 2),
        content_mean=torch.zeros(1, 4, 2),
        geometry_mean=torch.zeros(1, 4, 1),
        geometry_covariance_diag=torch.ones(1, 4, 1),
        existence_logits=torch.zeros(1, 4),
        visibility_given_existence_logits=torch.zeros(1, 4),
        measurement_age_s=torch.tensor([[0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0]]),
        valid=torch.tensor([[True, True, True, False]]),
        age=torch.tensor([[0, 1, 2, 3]]),
    )
    step = _PreparedActionStep(
        optimizer_plan_step=0,
        planned_transition=None,
        sample=None,
        evidence=SimpleNamespace(object_valid=torch.tensor([[True, True, True, False]])),
        final_belief=belief,
        policy_batch={},
        flow_timesteps=None,
        flow_noise=None,
        action_condition_input_ids=None,
        vision_patch_layout=None,
    )

    selected = _rows_at_least_one_reference_step_old(
        step,
        reference_delta_t_s=1.0 / 30.0,
    )

    assert selected.tolist() == [[False, True, True, False]]
    with pytest.raises(ValueError, match="finite positive reference delta time"):
        _rows_at_least_one_reference_step_old(step, reference_delta_t_s=0.0)


def test_source_frame_flow_seeds_match_overlapping_language_segments() -> None:
    first = _source_frame_flow_seed_contract(global_source_step=358773, repeat_index=0)
    repeated = _source_frame_flow_seed_contract(global_source_step=358773, repeat_index=0)
    next_repeat = _source_frame_flow_seed_contract(global_source_step=358773, repeat_index=1)
    next_frame = _source_frame_flow_seed_contract(global_source_step=358774, repeat_index=0)

    assert first == repeated
    assert first[2] == 358773
    assert first[:2] != next_repeat[:2]
    assert first[:2] != next_frame[:2]
    with pytest.raises(ValueError, match="global source step"):
        _source_frame_flow_seed_contract(global_source_step=-1, repeat_index=0)
    with pytest.raises(ValueError, match="flow repeat index"):
        _source_frame_flow_seed_contract(global_source_step=0, repeat_index=-1)


def test_flow_randomness_hash_is_dtype_and_value_stable() -> None:
    first = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    repeated = first.clone()
    changed = torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)

    assert _tensor_sha256(first) == _tensor_sha256(repeated)
    assert _tensor_sha256(first) != _tensor_sha256(changed)


def test_aggregate_keeps_measurement_separate_from_m4_acceptance() -> None:
    names = (
        "baseline",
        "without_posterior",
        "joint_row_permutation",
        "wrong_address",
        "remove_max_prior_row",
        "remove_rows_at_least_one_reference_step",
        "stale_previous_frame",
    )

    def row(scale: float, *, rank: int, at_least_one_step_old: bool) -> dict[str, object]:
        conditions = {
            name: {
                "action_loss": 0.2 + (scale if name != "baseline" else 0.0),
                "loss_delta_from_baseline": scale if name != "baseline" else 0.0,
                "velocity_rms_from_baseline": scale if name != "baseline" else 0.0,
            }
            for name in names
        }
        conditions["joint_row_permutation"] = {
            "action_loss": 0.2,
            "loss_delta_from_baseline": 0.0,
            "velocity_rms_from_baseline": 0.0,
        }
        return {
            "interventions": {
                "conditions": conditions,
                "integrity": {
                    "baseline_replay_exact": True,
                    "joint_permutation_action_loss_abs_delta": 0.0,
                    "joint_permutation_velocity_rms": 0.0,
                },
            },
            "measurement": {
                "has_row_with_expected_age_at_least_one_reference_step": (at_least_one_step_old)
            },
            "rank": rank,
        }

    result = _aggregate(
        (
            row(0.01, rank=0, at_least_one_step_old=False),
            row(-0.01, rank=0, at_least_one_step_old=True),
            row(0.03, rank=1, at_least_one_step_old=True),
        )
    )

    assert result["all_baseline_replays_exact"] is True
    assert result["all_joint_permutations_exact"] is True
    assert result["conditions"]["wrong_address"]["loss_delta_from_baseline"] == pytest.approx(0.01)
    assert result["conditions"]["wrong_address"]["positive_loss_delta_samples"] == 2
    assert result["conditions"]["wrong_address"]["positive_loss_delta_ranks"] == 1
    assert result["maximum_causal_velocity_rms"] == pytest.approx(0.01)
    assert result["rank_count"] == 2
    assert result["sample_count"] == 3
    assert result["strata"]["maximum_expected_age_below_one_reference_step"]["sample_count"] == 1
    assert result["strata"]["maximum_expected_age_at_least_one_reference_step"]["sample_count"] == 2
    assert result["m4_acceptance"] == "NOT_DECIDED_BY_READ_ONLY_DIAGNOSTIC"


def test_measurement_age_selection_is_deploy_visible_and_explicit() -> None:
    recent = {"has_row_with_expected_age_at_least_one_reference_step": False}
    persistent = {"has_row_with_expected_age_at_least_one_reference_step": True}

    assert _measurement_matches_age_selection(recent, "any") is True
    assert _measurement_matches_age_selection(persistent, "any") is True
    assert _measurement_matches_age_selection(recent, "sub-reference-step") is True
    assert _measurement_matches_age_selection(persistent, "sub-reference-step") is False
    assert _measurement_matches_age_selection(recent, "at-least-one-reference-step") is False
    assert _measurement_matches_age_selection(persistent, "at-least-one-reference-step") is True

    with pytest.raises(ValueError, match="unsupported measurement-age selection"):
        _measurement_matches_age_selection(recent, "task-specific")
    with pytest.raises(ValueError, match="boolean threshold"):
        _measurement_matches_age_selection({}, "any")


def test_action_visual_uses_runtime_ownership_without_targets(tmp_path) -> None:
    from picf_next.hosts.molmoact2_layout import MOLMO_VISION_PATCH_MODALITY
    from picf_next.models.temporal import ObjectBeliefBatch

    belief = ObjectBeliefBatch(
        address_mean=torch.zeros(1, 3, 2),
        content_mean=torch.zeros(1, 3, 2),
        geometry_mean=torch.zeros(1, 3, 1),
        geometry_covariance_diag=torch.ones(1, 3, 1),
        existence_logits=torch.tensor([[2.0, 1.0, 0.0]]),
        visibility_given_existence_logits=torch.tensor([[1.0, -1.0, 0.0]]),
        measurement_age_s=torch.tensor([[0.0, 2.0 / 30.0, 0.0]]),
        valid=torch.tensor([[True, True, False]]),
        age=torch.tensor([[4, 7, 0]]),
    )
    ownership = torch.zeros(1, 8, 4)
    ownership[:, :4, 0] = 0.8
    ownership[:, :4, -1] = 0.2
    ownership[:, 4:, 1] = 0.7
    ownership[:, 4:, -1] = 0.3
    image_keys = ("observation.images.image", "observation.images.wrist_image")
    spans = tuple(
        SimpleNamespace(image_key=key, start=4 * index, stop=4 * (index + 1), patches_per_crop=4)
        for index, key in enumerate(image_keys)
    )
    sample = SimpleNamespace(
        host_sample=SimpleNamespace(
            observation={
                image_keys[0]: np.zeros((8, 8, 3), dtype=np.uint8),
                image_keys[1]: np.zeros((8, 8, 3), dtype=np.uint8),
            },
            task_key="push_pink_block_right",
        ),
        record=SimpleNamespace(global_index=359754, task="sweep the pink block to the right"),
    )
    step = _PreparedActionStep(
        optimizer_plan_step=201,
        planned_transition=SimpleNamespace(
            episode_instance_id="episode-0",
            transition_index=5,
        ),
        sample=sample,
        evidence=SimpleNamespace(
            dense_banks=(SimpleNamespace(modality=MOLMO_VISION_PATCH_MODALITY),),
            dense_ownership=(ownership,),
            object_valid=belief.valid,
            object_log_prior=torch.tensor([[-0.1, -0.2, 0.0]]),
        ),
        final_belief=belief,
        policy_batch={},
        flow_timesteps=None,
        flow_noise=None,
        action_condition_input_ids=None,
        vision_patch_layout=SimpleNamespace(rows=(spans,)),
    )
    interventions = {
        "conditions": {
            "without_posterior": {"loss_delta_from_baseline": -0.001},
            "wrong_address": {"loss_delta_from_baseline": 0.002},
            "remove_rows_at_least_one_reference_step": {"loss_delta_from_baseline": 0.003},
        }
    }
    output = tmp_path / "action.png"

    artifact = _render_action_visual(
        path=output,
        step=step,
        rank=0,
        rank_sample_index=0,
        measurement={
            "maximum_measurement_age_reference_steps": 2.0,
            "reference_delta_t_s": 1.0 / 30.0,
        },
        interventions=interventions,
    )

    assert artifact["path"] == output.name
    assert artifact["task_key"] == "push_pink_block_right"
    assert artifact["bytes"] == output.stat().st_size
    assert len(artifact["sha256"]) == 64

    current = _runtime_visual_snapshot(step)
    previous = replace(
        current,
        optimizer_plan_step=200,
        global_source_step=359753,
        transition_index=4,
        measurement_age_s=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        visibility=np.asarray([0.8, 0.6, 0.0], dtype=np.float32),
    )
    records = _old_row_trajectory_records(
        (previous, current),
        1,
        reference_delta_t_s=1.0 / 30.0,
    )

    assert [record["global_source_step"] for record in records] == [359753, 359754]
    assert records[0]["measurement_age_reference_steps"] == pytest.approx(0.0)
    assert records[1]["measurement_age_reference_steps"] == pytest.approx(2.0)
    assert records[1]["address_cosine_to_current"] is None
    assert records[1]["camera_ownership"][image_keys[1]]["mass"] == pytest.approx(2.8)

    temporal = _render_old_row_trajectories(
        path_stem=tmp_path / "history",
        history=(previous, current),
        rows=(1,),
        rank=0,
        rank_sample_index=0,
        reference_delta_t_s=1.0 / 30.0,
        loss_delta_from_baseline=0.003,
    )

    assert len(temporal) == 1
    assert temporal[0]["kind"] == "runtime_old_posterior_row_trajectory"
    assert temporal[0]["global_source_steps"] == [359753, 359754]
    assert (tmp_path / temporal[0]["path"]).is_file()
