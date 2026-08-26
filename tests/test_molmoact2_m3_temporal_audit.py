import subprocess
import sys
from pathlib import Path

import pytest

from picf_next.posterior import BIRTH_EVENT, MATCH_EVENT, MISS_EVENT, UNUSED_EVENT
from picf_next.training.control import EpisodeSampleSequence, FrozenEpisodeStreamPlan
from tools.audit_molmoact2_m3_temporal import (
    _evaluation_context,
    _invert_observation_to_row,
    _select_counterfactual_task,
    _semantic_identity_score,
    _summarize_address_relations,
    _trace_address_relation_pairs,
    _trace_assignments,
    _validate_extended_plan_prefix,
    _validate_state_handoff_controls,
)


def test_state_handoff_evaluation_context_keeps_reloadable_tensors() -> None:
    class FakeTorch:
        @staticmethod
        def no_grad() -> str:
            return "reloadable-no-grad"

        @staticmethod
        def inference_mode() -> str:
            return "fixed-weight-inference"

    assert _evaluation_context(FakeTorch, reload_weights_midstream=True) == "reloadable-no-grad"
    assert (
        _evaluation_context(FakeTorch, reload_weights_midstream=False) == "fixed-weight-inference"
    )


def test_prompt_audit_uses_generic_semantic_overlap_without_task_rules() -> None:
    assert (
        _semantic_identity_score(
            "in the slider grasp the blue block",
            "movable/block_blue",
        )
        == 2
    )
    assert (
        _semantic_identity_score(
            "in the slider grasp the blue block",
            "movable/block_pink",
        )
        == 1
    )
    assert _semantic_identity_score("slide down the switch", "part/table/switch_link") == 1
    assert _semantic_identity_score("slide down the switch", None) == 0


def test_prompt_audit_selects_closest_distinct_counterfactual_deterministically() -> None:
    assert (
        _select_counterfactual_task(
            "turn off the light bulb",
            (
                "sweep the pink block to the right",
                "turn on the light bulb",
                "lift the blue block",
            ),
        )
        == "turn on the light bulb"
    )
    with pytest.raises(ValueError, match="distinct counterfactual"):
        _select_counterfactual_task("same task", ("same task",))


def test_relation_trace_covers_deployed_same_different_and_null_candidates() -> None:
    records = _trace_address_relation_pairs(
        previous_keys=("track:a", "track:b", None),
        current_identity_keys=("track:a", "track:c"),
        prediction_indices=(0, 2),
        target_indices=(0, 1),
        object_inventory_complete=True,
        address_cosine=((0.9, 0.1, 0.2), (0.3, -0.2, 0.4), (0.0, 0.0, 0.0)),
        query_existence_probability=(0.9, 0.1, 0.8),
        query_localization_confidence=(0.7, 0.3, 0.6),
        query_mask_quality=(0.8, 0.2, 0.5),
        query_mask_coherence_score=(0.72, 0.02, 0.4),
        query_object_confidence=(0.63, 0.03, 0.48),
        logit_scale=10.0,
        logit_bias=-2.71,
    )

    assert len(records) == 6
    assert [record["relation"] for record in records[:3]] == [
        "same_identity",
        "complete_inventory_null",
        "different_identity",
    ]
    assert [record["relation"] for record in records[3:]] == [
        "different_identity",
        "complete_inventory_null",
        "different_identity",
    ]
    assert records[0]["address_log_likelihood_ratio"] == pytest.approx(6.29)
    assert records[1]["address_log_likelihood_ratio"] == pytest.approx(-1.71)
    assert records[0]["address_relation_logit_scale"] == 10.0
    assert records[0]["address_relation_logit_bias"] == -2.71
    assert records[0]["query_localization_confidence"] == 0.7
    assert records[0]["query_mask_quality"] == 0.8
    assert records[0]["query_mask_coherence_score"] == 0.72
    assert records[0]["query_object_confidence"] == 0.63


def test_relation_trace_preserves_unmatched_queries_as_unknown_for_partial_inventory() -> None:
    records = _trace_address_relation_pairs(
        previous_keys=("track:a",),
        current_identity_keys=("track:a",),
        prediction_indices=(0,),
        target_indices=(0,),
        object_inventory_complete=False,
        address_cosine=((0.8, 0.6),),
        query_existence_probability=(0.9, 0.2),
        query_localization_confidence=(0.7, 0.4),
        query_mask_quality=(0.8, 0.1),
        query_mask_coherence_score=(0.72, 0.02),
        query_object_confidence=(0.63, 0.08),
        logit_scale=5.0,
        logit_bias=-2.0,
    )

    assert [record["relation"] for record in records] == ["same_identity", "unknown"]


def test_relation_summary_reports_separate_null_discrimination_without_fitting() -> None:
    records = (
        [
            {"relation": "same_identity", "address_log_likelihood_ratio": value}
            for value in (3.0, 4.0)
        ]
        + [
            {"relation": "different_identity", "address_log_likelihood_ratio": value}
            for value in (-2.0, -1.0)
        ]
        + [
            {"relation": "complete_inventory_null", "address_log_likelihood_ratio": value}
            for value in (-4.0, -3.0)
        ]
    )

    summary = _summarize_address_relations(records)

    assert summary["record_count"] == 6
    assert summary["known_pair_count"] == 6
    assert summary["classes"]["same_identity"]["count"] == 2
    assert summary["classes"]["unknown"]["count"] == 0
    assert summary["discrimination"]["same_vs_all_known_negative_auroc"] == 1.0
    assert summary["discrimination"]["same_vs_complete_inventory_null_auroc"] == 1.0


def test_relation_summary_never_converts_unknown_pairs_to_negatives() -> None:
    summary = _summarize_address_relations(
        ({"relation": "unknown", "address_log_likelihood_ratio": 8.0},)
    )

    assert summary["known_pair_count"] == 0
    assert "negative_log_likelihood" not in summary["classes"]["unknown"]
    assert "brier_score" not in summary["classes"]["unknown"]


def test_audit_imports_current_checkout_without_editable_install() -> None:
    script = Path(__file__).resolve().parents[1] / "tools/audit_molmoact2_m3_temporal.py"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy,sys;runpy.run_path(sys.argv[1],run_name='picf_m3_temporal_audit')",
            str(script),
        ],
        check=True,
    )


def test_extended_replay_plan_preserves_every_checkpoint_transition() -> None:
    episodes = tuple(
        EpisodeSampleSequence(
            episode_key=f"episode-{episode}",
            sample_keys=tuple(f"episode-{episode}/sample-{step}" for step in range(6)),
        )
        for episode in range(4)
    )
    common = {
        "dataset_id": "dataset",
        "dataset_revision": "revision",
        "dataset_manifest_sha256": "a" * 64,
        "episodes": episodes,
        "comparison_id": "temporal-audit",
        "seed": 17,
        "global_batch_size": 2,
    }
    checkpoint = FrozenEpisodeStreamPlan(**common, total_steps=4)
    extended = FrozenEpisodeStreamPlan(**common, total_steps=8)

    assert _validate_extended_plan_prefix(checkpoint, extended) == 4

    class ChangedPlan:
        total_steps = 8

        @staticmethod
        def global_batch(step: int) -> object:
            if step == 2:
                return object()
            return extended.global_batch(step)

    with pytest.raises(ValueError, match="changed checkpoint step 3"):
        _validate_extended_plan_prefix(checkpoint, ChangedPlan())


def test_state_handoff_accepts_only_plan_horizon_differences() -> None:
    common_contract = {
        "schema": "contract",
        "code_revision": "revision",
        "comparison_id": "comparison",
        "common_config": {"recipe_sha256": "a" * 64},
    }

    def control(*, steps: int, fairness: str, plan_hash: str) -> dict[str, object]:
        return {
            "contract": {
                **common_contract,
                "fairness_sha256": fairness,
                "sample_plan_sha256": plan_hash,
            },
            "plan": {"total_steps": steps},
            "progress": {
                "attempted_optimizer_steps": steps,
                "successful_optimizer_steps": steps,
            },
        }

    primary = control(steps=200, fairness="primary", plan_hash="b" * 64)
    prefix = control(steps=20, fairness="prefix", plan_hash="c" * 64)
    assert _validate_state_handoff_controls(primary, prefix) == (20, 20)

    incompatible = control(steps=20, fairness="prefix", plan_hash="c" * 64)
    incompatible["contract"]["code_revision"] = "other"  # type: ignore[index]
    with pytest.raises(ValueError, match="outside their frozen-plan horizon"):
        _validate_state_handoff_controls(primary, incompatible)


def test_runtime_observation_map_inversion_is_strict() -> None:
    assert _invert_observation_to_row((2, -1, 0), capacity=4) == (2, None, 0, None)

    with pytest.raises(ValueError, match="multiple runtime observations"):
        _invert_observation_to_row((1, 1), capacity=2)
    with pytest.raises(ValueError, match="invalid posterior row"):
        _invert_observation_to_row((2,), capacity=2)


def test_trace_assignments_distinguishes_every_conflict_class() -> None:
    assignments, conflicts = _trace_assignments(
        previous_keys=("track:a", "track:b", None),
        next_keys=("track:a", "track:b", None),
        identity_keys=("track:a",),
        prediction_indices=(0,),
        target_indices=(0,),
        query_existence_probability=(0.8,),
        query_localization_confidence=(0.5,),
        query_mask_quality=(0.5,),
        query_mask_coherence_score=(0.4,),
        query_object_confidence=(0.4,),
        query_ownership_mass=(2.0,),
        target_ownership_mass=(3.0,),
        observation_to_row=(1,),
        event_type=(MATCH_EVENT, MATCH_EVENT, UNUSED_EVENT),
        final_valid=(True, True, False),
    )
    assert assignments[0]["selected_loss_track_row"] == 0
    assert conflicts[0]["reason"] == "retained_identity_runtime_row_disagreement"

    _assignments, conflicts = _trace_assignments(
        previous_keys=(None,),
        next_keys=(None,),
        identity_keys=("track:new",),
        prediction_indices=(0,),
        target_indices=(0,),
        query_existence_probability=(0.2,),
        query_localization_confidence=(0.5,),
        query_mask_quality=(0.5,),
        query_mask_coherence_score=(0.1,),
        query_object_confidence=(0.1,),
        query_ownership_mass=(1.0,),
        target_ownership_mass=(3.0,),
        observation_to_row=(-1,),
        event_type=(UNUSED_EVENT,),
        final_valid=(False,),
    )
    assert conflicts[0]["reason"] == "runtime_observation_unmapped"

    _assignments, conflicts = _trace_assignments(
        previous_keys=("track:other", None),
        next_keys=("track:other", None),
        identity_keys=("track:new",),
        prediction_indices=(0,),
        target_indices=(0,),
        query_existence_probability=(0.7,),
        query_localization_confidence=(0.5,),
        query_mask_quality=(0.5,),
        query_mask_coherence_score=(0.35,),
        query_object_confidence=(0.35,),
        query_ownership_mass=(2.0,),
        target_ownership_mass=(3.0,),
        observation_to_row=(0,),
        event_type=(MISS_EVENT, UNUSED_EVENT),
        final_valid=(True, False),
    )
    assert conflicts[0]["reason"] == "runtime_row_already_occupied"


def test_trace_assignments_accepts_a_clean_birth() -> None:
    assignments, conflicts = _trace_assignments(
        previous_keys=(None, None),
        next_keys=("track:new", None),
        identity_keys=("track:new",),
        prediction_indices=(0,),
        target_indices=(0,),
        query_existence_probability=(0.9,),
        query_localization_confidence=(0.5,),
        query_mask_quality=(0.5,),
        query_mask_coherence_score=(0.45,),
        query_object_confidence=(0.45,),
        query_ownership_mass=(2.0,),
        target_ownership_mass=(3.0,),
        observation_to_row=(0,),
        event_type=(BIRTH_EVENT, UNUSED_EVENT),
        final_valid=(True, False),
    )

    assert conflicts == []
    assert assignments == [
        {
            "conflict": False,
            "identity_key": "track:new",
            "old_row": None,
            "query": 0,
            "query_existence_probability": 0.9,
            "query_localization_confidence": 0.5,
            "query_mask_coherence_score": 0.45,
            "query_mask_quality": 0.5,
            "query_object_confidence": 0.45,
            "query_ownership_mass": 2.0,
            "runtime_event": "birth",
            "runtime_row": 0,
            "selected_loss_track_row": 0,
            "target_index": 0,
            "target_ownership_mass": 3.0,
        }
    ]
