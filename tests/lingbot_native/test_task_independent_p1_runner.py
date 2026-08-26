from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_lingbot_vla2_task_independent_p1 import (
    _MAXIMUM_P1_CURVE_STEPS,
    _MAXIMUM_P1_STEPS,
    _MAXIMUM_STAGED_P2_STEPS,
    _SUPPORTED_P1_VISUAL_LATTICES,
    _adr175_rank_step_receipt,
    _audit_optimizer_family_state,
    _calvin_causal_replay_dependency_closure,
    _distributed_phase_error,
    _entity_evaluation_replay_seed,
    _evaluation_steps,
    _evaluation_visual_sample_keys,
    _implementation_provenance,
    _load_p2_causal_replay_closure,
    _optimizer_state_family_counts,
    _p2_optimizer_state_families,
    _publish_p2_update_stage,
    _require_carried_bindings_preserved,
    _select_p2_causal_records,
    _select_staged_p2_records_from_plan,
    _summarize_p2_causal_evidence,
    _synchronize_p2_update_ranks,
    _validate_current_frame_objective_weights,
    _validate_entity_loss_weights,
    _validate_visual_lattice_inputs,
)


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "tools/run_lingbot_vla2_task_independent_p1.py").read_text()


def test_p1_runner_is_bounded_and_publishes_no_checkpoint() -> None:
    source = _source()

    assert _MAXIMUM_P1_STEPS == 20
    assert _MAXIMUM_P1_CURVE_STEPS == 200
    assert _MAXIMUM_STAGED_P2_STEPS == 200
    assert "build_checkpointer" not in source
    assert "checkpointer.save" not in source
    assert '"checkpoint_published": False' in source


def test_p1_curve_steps_are_explicit_unique_and_sorted() -> None:
    assert _evaluation_steps(None) == ()
    assert _evaluation_steps("0,20,100,200") == (0, 20, 100, 200)


def test_evaluation_visuals_prefer_distinct_tasks_over_heldout_replicates() -> None:
    items = tuple(
        SimpleNamespace(partition=partition, task_key=task, sample_key=sample)
        for partition, task, sample in (
            ("validation", "a", "v-a-0"),
            ("validation", "b", "v-b-0"),
            ("heldout", "a", "h-a-0"),
            ("heldout", "a", "h-a-1"),
            ("heldout", "b", "h-b-0"),
            ("heldout", "b", "h-b-1"),
        )
    )

    assert _evaluation_visual_sample_keys(
        items,
        partitions=("validation", "heldout"),
        per_partition=2,
    ) == ("v-a-0", "v-b-0", "h-a-0", "h-b-0")


def test_adr175_receipt_publishes_the_raw_single_sample_prompt_receipt() -> None:
    import torch

    raw_prompt_receipt = "a" * 64
    planned = SimpleNamespace(
        physical_prompt_selection_receipts=(raw_prompt_receipt,),
    )
    collated = SimpleNamespace(
        model_inputs={
            "actions": torch.zeros(1, 1, 1),
            "action_is_pad": torch.zeros(1, 1, dtype=torch.bool),
            "noise": torch.zeros(1, 1, 1),
            "time": torch.zeros(1),
        },
        routing=SimpleNamespace(sample_keys=("sample",)),
        source_digest="b" * 64,
    )

    receipt = _adr175_rank_step_receipt(planned=planned, collated=collated)

    assert receipt["prompt_sha256"] == raw_prompt_receipt


def test_p1_accepts_a_zero_ownership_ablation_without_accepting_an_empty_objective() -> None:
    weights = SimpleNamespace(
        mask_focal_weight=1.0,
        mask_dice_weight=1.0,
        existence_weight=1.0,
        ownership_weight=0.0,
    )
    _validate_entity_loss_weights(weights)

    weights.mask_focal_weight = 0.0
    weights.mask_dice_weight = 0.0
    weights.existence_weight = 0.0
    with pytest.raises(ValueError, match="at least one active"):
        _validate_entity_loss_weights(weights)

    weights.ownership_weight = -1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        _validate_entity_loss_weights(weights)


def test_current_frame_joint_action_weights_are_explicit_and_nonnegative() -> None:
    weights = SimpleNamespace(
        current_frame_action_weight=0.0,
        current_frame_entity_weight=1.0,
    )
    assert _validate_current_frame_objective_weights(weights) is False

    weights.current_frame_action_weight = 1.0
    weights.current_frame_entity_weight = 0.08
    assert _validate_current_frame_objective_weights(weights) is True

    weights.current_frame_entity_weight = 0.0
    with pytest.raises(ValueError, match="positive entity weight"):
        _validate_current_frame_objective_weights(weights)

    weights.current_frame_entity_weight = 0.08
    weights.current_frame_action_weight = float("nan")
    with pytest.raises(ValueError, match="finite and non-negative"):
        _validate_current_frame_objective_weights(weights)


def test_p1_visual_lattice_is_explicit_and_exact_for_two_calvin_views() -> None:
    import torch

    assert _SUPPORTED_P1_VISUAL_LATTICES == (8, 12)
    model_inputs = {
        "image_grid_thw": torch.tensor(
            [[[1, 24, 24], [1, 24, 24], [1, 24, 24]]],
            dtype=torch.long,
        ),
        "img_masks": torch.tensor([[True, True, False]]),
    }

    assert _validate_visual_lattice_inputs(
        model_inputs,
        visual_lattice=12,
        merge_size=2,
    ) == {
        "valid_views_per_sample": 2,
        "merged_tokens_per_view": 144,
        "merged_visual_tokens_per_sample": 288,
    }
    with pytest.raises(RuntimeError, match="declared visual lattice"):
        _validate_visual_lattice_inputs(
            model_inputs,
            visual_lattice=8,
            merge_size=2,
        )


def test_staged_p2_binding_audit_requires_exact_carried_rows() -> None:
    prior = ((("object/a", 1), ("object/b", 3)),)
    current = ((("object/a", 1), ("object/b", 3), ("object/c", 2)),)
    assert _require_carried_bindings_preserved(prior, current, capacity=4) == 2

    with pytest.raises(RuntimeError, match="changed a carried physical row"):
        _require_carried_bindings_preserved(
            prior,
            ((("object/a", 2), ("object/b", 3)),),
            capacity=4,
        )


def test_p1_evaluation_replay_seed_depends_only_on_plan_and_sample() -> None:
    plan = "a" * 64
    assert _entity_evaluation_replay_seed(plan, "sample-a") == (
        _entity_evaluation_replay_seed(plan, "sample-a")
    )
    assert _entity_evaluation_replay_seed(plan, "sample-a") != (
        _entity_evaluation_replay_seed(plan, "sample-b")
    )


def test_p1_evaluation_exchanges_rank_local_failures_before_next_forward() -> None:
    class Dist:
        @staticmethod
        def all_gather_object(gathered: list[object], local: object) -> None:
            gathered[:] = [local, None]

    _distributed_phase_error(
        error=None,
        phase="prepare",
        rank=0,
        dist_module=Dist,
    )

    try:
        _distributed_phase_error(
            error=ValueError("bad fixed sample"),
            phase="evidence",
            rank=0,
            dist_module=Dist,
        )
    except RuntimeError as error:
        assert (
            str(error)
            == "distributed P1 phase failed: rank 0 evidence ValueError: bad fixed sample"
        )
    else:
        raise AssertionError("rank-local evaluation failure was not propagated")


def test_p1_runner_uses_one_shared_host_for_registered_action_ablation() -> None:
    source = _source()

    assert "TASK_INDEPENDENT_ENTITY_POSTERIOR" in source
    assert "run_task_independent_calvin_current_frame_objective" in source
    assert "NativeTrainingLaneCoordinator" not in source
    assert '"posterior_input_mode": "current_frame_discovery_only"' in source
    assert '"previous_state_input_absent": True' in source
    assert "run_native_policy_training_forward" not in source
    assert "build_lingbot_representation_optimizer" in source
    assert "build_lingbot_official_optimizer" in source
    assert "build_moe_load_balance_hook" in source
    assert "task_identity_resolver" not in source
    assert "semantic_scorer" not in source
    assert "action_weight=args.current_frame_action_weight" in source
    assert '"current_frame_joint_action"' in source
    assert "current-frame joint action is exclusive to P1" in source
    assert '("vlm_host", ".qwenvl.")' in source
    assert "*_ACTION_ONLY_GRADIENT_METRICS" in source
    assert "configure_native_representation_parameter_scope(policy)" in source
    assert '"action_suffix_executed": False' in source
    assert "render_task_independent_entity_visuals" in source
    assert "optimizer_step in {0, args.steps - 1}" in source
    assert '"visual_artifacts": visual_artifacts' in source
    assert '"event": "task_independent_p1_step"' in source
    assert '"event": "task_independent_p1_evaluation_progress"' in source
    assert "_distributed_phase_error(" in source
    assert "flush=True" in source


def test_p2_gate_carries_the_loss_side_prefix_gauge() -> None:
    source = _source()

    causal_filter = source.index("p2_selected_records = _select_p2_causal_records(")
    model_load = source.index("load_model_weights(")
    prefix_assignment = source.index("prefix_row_bindings = physical_frame_row_bindings(")
    sequence_call = source.index("run_task_independent_calvin_sequence_objective(")
    carried_binding = source.index(
        "prior_row_bindings_by_batch=prefix_row_bindings,",
        sequence_call,
    )

    assert causal_filter < model_load < prefix_assignment
    assert prefix_assignment < sequence_call < carried_binding
    assert '"prefix_row_bindings": [' in source


def test_staged_optimizer_audits_enter_collectives_only_after_local_error_exchange() -> None:
    source = _source()

    p1_local_check = source.index(
        "p1_boundary_optimizer_state = _optimizer_state_family_counts(policy, optimizer)"
    )
    p1_local_exchange = source.index('phase="staged-p1-boundary-local"')
    p1_collective_audit = source.index(
        "p1_boundary_host_optimizer_audit = _audit_optimizer_family_state("
    )
    assert p1_local_check < p1_local_exchange < p1_collective_audit

    p2_local_check = source.index("family_state_entries = _p2_optimizer_state_families(")
    p2_local_exchange = source.index(
        'phase=f"staged-p2-step-{p2_optimizer_step}-post-update-local"'
    )
    p2_collective_audit = source.index(
        '"vlm_host": (',
        p2_local_exchange,
    )
    assert p2_local_check < p2_local_exchange < p2_collective_audit

    p1_log_capture = source.index("step_log_error: BaseException | None = None")
    p1_log_exchange = source.index('phase=f"p1-step-{optimizer_step}-log"')
    next_p1_collective = source.index("run_entity_evaluation(completed_step)")
    assert p1_log_capture < p1_log_exchange < next_p1_collective

    p2_log_capture = source.index("p2_log_error: BaseException | None = None")
    p2_log_exchange = source.index('phase=f"staged-p2-step-{p2_optimizer_step}-log"')
    next_p2_collective = source.index("dist.all_gather_object(\n                gathered_p2")
    assert p2_log_capture < p2_log_exchange < next_p2_collective


def test_p2_update_gate_audits_optimizer_families_without_action_state() -> None:
    import torch

    names = {
        "model.picf_native_graph.relation_readout.projection.weight": torch.nn.Parameter(
            torch.ones(2)
        ),
        (
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.18.input_layernorm.weight"
        ): torch.nn.Parameter(torch.ones(2)),
        "model.picf_native_graph.predictive_readouts.dino_video.weight": torch.nn.Parameter(
            torch.ones(2)
        ),
        (
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm.weight"
        ): torch.nn.Parameter(torch.ones(2)),
        "model.action_out_proj.weight": torch.nn.Parameter(torch.ones(2)),
    }

    class Model:
        @staticmethod
        def named_parameters():
            return tuple(names.items())

    optimizer = torch.optim.AdamW(tuple(names.values()), lr=1e-3)
    for name, parameter in names.items():
        if ".qwen_expert." not in name and ".action_out_proj." not in name:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    families = _p2_optimizer_state_families(Model(), optimizer)

    assert families == {
        "native_graph": 2,
        "vlm_host": 1,
        "predictive_readout": 1,
        "action_expert": 0,
        "action_only": 0,
    }
    p2_update_branch = _source().split("if args.p2_optimizer_update:", 1)[1]
    assert "optimizer.step()" in p2_update_branch
    assert "_validate_optimizer_state(" in p2_update_branch
    assert "_p2_optimizer_state_families(" in p2_update_branch
    assert "_capture_p2_update_probes(" not in p2_update_branch
    assert "P2_UPDATE_GATE_REPORT_SCHEMA" in p2_update_branch


def test_p2_update_stage_is_rank_local_atomic_and_strict(tmp_path: Path) -> None:
    _publish_p2_update_stage(
        run_dir=tmp_path,
        rank=1,
        stage="optimizer_step_started",
    )

    assert (tmp_path / "p2_update_rank_1.stage").read_text(encoding="ascii") == (
        "optimizer_step_started\n"
    )
    assert not (tmp_path / "p2_update_rank_1.stage.tmp").exists()
    with pytest.raises(ValueError, match="lowercase ASCII"):
        _publish_p2_update_stage(run_dir=tmp_path, rank=1, stage="optimizer-step")


def test_p2_update_rank_synchronization_orders_cuda_before_barrier() -> None:
    calls: list[tuple[str, object]] = []

    class CUDA:
        @staticmethod
        def synchronize(device: object) -> None:
            calls.append(("cuda", device))

    class Torch:
        cuda = CUDA()

    class Dist:
        @staticmethod
        def barrier(*, device_ids: list[int]) -> None:
            calls.append(("barrier", device_ids))

    device = SimpleNamespace(type="cuda", index=1)
    _synchronize_p2_update_ranks(
        device=device,
        dist_module=Dist(),
        torch_module=Torch(),
    )
    assert calls == [("cuda", device), ("barrier", [1])]


def test_p2_causal_selection_skips_episode_boundaries_before_rank_selection() -> None:
    import torch

    records = tuple(
        SimpleNamespace(
            source_global_index=source,
            horizon=1,
            importance=torch.tensor([1.0]),
        )
        for source in (0, 3, 4, 5, 6)
    )
    segments = (SimpleNamespace(start=0, index=0), SimpleNamespace(start=4, index=1))
    episodes = (
        SimpleNamespace(sample_keys=("a0", "a1", "a2", "a3")),
        SimpleNamespace(sample_keys=("b0", "b1", "b2", "b3")),
    )

    selected = _select_p2_causal_records(
        records=records,
        segments=segments,
        episodes=episodes,
        horizon=1,
        count=2,
    )

    assert tuple(item[0].source_global_index for item in selected) == (5, 6)


def test_p2_causal_selection_can_require_two_observed_prefix_frames() -> None:
    import torch

    records = tuple(
        SimpleNamespace(
            source_global_index=source,
            horizon=1,
            importance=torch.tensor([1.0]),
        )
        for source in (1, 2, 3)
    )
    selected = _select_p2_causal_records(
        records=records,
        segments=(SimpleNamespace(start=0, index=0),),
        episodes=(SimpleNamespace(sample_keys=("a0", "a1", "a2", "a3", "a4")),),
        horizon=1,
        count=2,
        prefix_frames=2,
    )

    assert tuple(item[0].source_global_index for item in selected) == (2, 3)


def test_calvin_causal_replay_dependency_closure_includes_control_reads() -> None:
    replay, required = _calvin_causal_replay_dependency_closure(
        source_global_index=10,
        segment_start=0,
        segment_end=30,
        horizon=1,
        prefix_frames=2,
        action_horizon=4,
    )

    assert replay == (8, 9, 10, 11)
    assert required == tuple(range(7, 15))


def test_p2_causal_replay_closure_binds_all_immutable_inputs(tmp_path: Path) -> None:
    dataset_manifest = tmp_path / "dataset-manifest.json"
    representation_split = tmp_path / "representation-split.json"
    cache_root = tmp_path / "predictive-cache"
    cache_root.mkdir()
    cache_manifest = cache_root / "manifest.json"
    dataset_manifest.write_text("dataset\n", encoding="ascii")
    representation_split.write_text("split\n", encoding="ascii")
    cache_manifest.write_text("cache\n", encoding="ascii")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    selections = [
        {
            "replay_global_indices": [source - 2, source - 1, source, source + 1],
            "required_global_indices": [source],
            "segment_index": ordinal,
            "source_episode_index": 1000 + ordinal,
            "source_global_index": source,
            "target_global_index": source + 1,
            "transition_index": 3,
        }
        for ordinal, source in enumerate(range(100, 114))
    ]
    payload = {
        "schema": "picf-next.calvin-causal-replay-file-closure/v2",
        "selection_seed": 20260805 + 2_000_003,
        "selection_domain": "all-nontraining",
        "training_source_episode_indices_sha256": "0" * 64,
        "count": 14,
        "horizon": 1,
        "prefix_frames": 2,
        "action_horizon": 16,
        "selections": selections,
        "required_paths": [f"episode_{source:07d}.npz" for source in range(100, 114)],
        "missing_paths": [],
        "available_roots": [str(tmp_path)],
        "dataset_manifest_sha256": digest(dataset_manifest),
        "representation_split_file_sha256": digest(representation_split),
        "predictive_cache_manifest_sha256": digest(cache_manifest),
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    closure = tmp_path / "closure.json"
    closure.write_text(json.dumps(payload), encoding="ascii")
    args = SimpleNamespace(
        p2_causal_replay_closure=closure,
        p2_causal_replay_closure_sha256=digest(closure),
        dataset_manifest=dataset_manifest,
        representation_split=representation_split,
        p2_causal_probe_cache_root=cache_root,
        p2_causal_probe_steps=7,
        p2_horizon=1,
        seed=20260805,
    )

    assert _load_p2_causal_replay_closure(args) == payload


def test_p2_causal_summary_preregisters_sample_size_and_all_interventions() -> None:
    names = (
        "absent_source",
        "batch_shift_control",
        "batch_shift_source",
        "matched_noise_source",
        "row_shift_source",
        "wrong_time_source",
        "zero_control",
        "zero_current_observation",
        "zero_source",
    )

    def rank_report(rank: int, *, failed_name: str | None = None) -> dict[str, object]:
        return {
            "rank": rank,
            "causal_steps": [
                {
                    "source_global_index": 1000 + step * 2 + rank,
                    "source_episode_index": 2000 + step * 2 + rank,
                    "partition": "causal_audit",
                    "exact_correction_then_prior_route": True,
                    "diagnostics": {
                        "interventions": [
                            {
                                "name": name,
                                "loss_margin_over_factual": (
                                    -0.1 if name == failed_name else 0.1
                                ),
                                "normalized_prediction_l1": 0.2,
                            }
                            for name in names
                        ]
                    },
                }
                for step in range(7)
            ],
        }

    passed = _summarize_p2_causal_evidence(
        [rank_report(0), rank_report(1)],
        expected_global_steps=7,
    )
    assert passed["sample_count"] == 14
    assert passed["status"] == "PASS"
    assert passed["neutral_controls"]["matched_noise_source"]["role"] == (
        "paired_null_control_not_an_acceptance_arm"
    )

    failed = _summarize_p2_causal_evidence(
        [
            rank_report(0, failed_name="wrong_time_source"),
            rank_report(1, failed_name="wrong_time_source"),
        ],
        expected_global_steps=7,
    )
    assert failed["status"] == "FAIL"


def test_p2_causal_runner_journals_each_rank_before_final_aggregation() -> None:
    source = _source()

    journal = source.index('journal_root = args.run_dir / "p2_causal_rank_journal"')
    aggregate = source.index("dist.all_gather_object(\n                gathered_p2")
    assert journal < aggregate
    assert 'phase=f"p2-causal-step-{causal_step}-journal"' in source


def test_p2_causal_selection_canonicalizes_overlapping_language_segments() -> None:
    import torch

    record = SimpleNamespace(
        source_global_index=3,
        horizon=1,
        importance=torch.tensor([1.0]),
    )
    selected = _select_p2_causal_records(
        records=(record,),
        segments=(
            SimpleNamespace(start=0, index=7),
            SimpleNamespace(start=2, index=3),
        ),
        episodes=(
            SimpleNamespace(sample_keys=tuple(f"a{value}" for value in range(6))),
            SimpleNamespace(sample_keys=tuple(f"b{value}" for value in range(5))),
        ),
        horizon=1,
        count=1,
    )

    assert selected[0][1].index == 3
    assert selected[0][3] == 1


def test_p2_causal_selection_can_be_restricted_to_training_episodes() -> None:
    import torch

    records = tuple(
        SimpleNamespace(
            source_global_index=source,
            horizon=1,
            importance=torch.tensor([1.0]),
        )
        for source in (1, 5)
    )
    selected = _select_p2_causal_records(
        records=records,
        segments=(
            SimpleNamespace(start=0, index=0, episode_index=10),
            SimpleNamespace(start=4, index=1, episode_index=20),
        ),
        episodes=(
            SimpleNamespace(sample_keys=("a0", "a1", "a2", "a3")),
            SimpleNamespace(sample_keys=("b0", "b1", "b2", "b3")),
        ),
        horizon=1,
        count=1,
        allowed_episode_indices=frozenset({20}),
    )

    assert selected[0][0].source_global_index == 5
    assert selected[0][1].episode_index == 20


def test_p2_causal_selection_can_require_distinct_source_episodes() -> None:
    import torch

    records = tuple(
        SimpleNamespace(
            source_global_index=source,
            horizon=1,
            importance=torch.tensor([1.0]),
        )
        for source in (1, 2, 5)
    )
    selected = _select_p2_causal_records(
        records=records,
        segments=(
            SimpleNamespace(start=0, index=0, episode_index=10),
            SimpleNamespace(start=4, index=1, episode_index=20),
        ),
        episodes=(
            SimpleNamespace(sample_keys=("a0", "a1", "a2", "a3")),
            SimpleNamespace(sample_keys=("b0", "b1", "b2", "b3")),
        ),
        horizon=1,
        count=2,
        distinct_source_episodes=True,
    )

    assert tuple(item[0].source_global_index for item in selected) == (1, 5)
    assert tuple(item[1].episode_index for item in selected) == (10, 20)


def test_staged_p2_selection_is_seeded_and_does_not_read_labels() -> None:
    records = tuple(
        SimpleNamespace(source_global_index=source, target_global_index=source + 1, horizon=1)
        for source in range(1, 8)
    )
    segments = (SimpleNamespace(start=0, index=0, episode_index=10),)
    episodes = (SimpleNamespace(sample_keys=tuple(f"a{value}" for value in range(10))),)

    first = _select_p2_causal_records(
        records=records,
        segments=segments,
        episodes=episodes,
        horizon=1,
        count=4,
        allowed_episode_indices=frozenset({10}),
        require_positive_importance=False,
        selection_seed=17,
    )
    replay = _select_p2_causal_records(
        records=reversed(records),
        segments=segments,
        episodes=episodes,
        horizon=1,
        count=4,
        allowed_episode_indices=frozenset({10}),
        require_positive_importance=False,
        selection_seed=17,
    )

    assert tuple(item[0].source_global_index for item in first) == tuple(
        item[0].source_global_index for item in replay
    )


def test_staged_p2_uses_the_frozen_stream_suffix_without_labels() -> None:
    episodes = (
        SimpleNamespace(
            episode_key="episode-a",
            segment_index=0,
            sample_keys=("a0", "a1", "a2", "a3"),
        ),
        SimpleNamespace(
            episode_key="episode-b",
            segment_index=1,
            sample_keys=("b0", "b1", "b2", "b3"),
        ),
    )

    class Plan:
        total_steps = 3

        @staticmethod
        def global_batch(step: int) -> SimpleNamespace:
            index = step
            return SimpleNamespace(
                transitions=tuple(
                    SimpleNamespace(
                        episode_key=episode.episode_key,
                        transition_index=index,
                        sample=SimpleNamespace(sample_key=episode.sample_keys[index]),
                    )
                    for episode in episodes
                )
            )

    source_by_key = {
        key: offset + index
        for offset, episode in zip((100, 200), episodes, strict=True)
        for index, key in enumerate(episode.sample_keys)
    }

    class Dataset:
        @staticmethod
        def source_global_index_by_key(key: str) -> int:
            return source_by_key[key]

    class Cache:
        @staticmethod
        def has_supported_target(*, source_global_index: int, horizon: int) -> bool:
            raise AssertionError("staged schedule must not inspect target importance")

        @staticmethod
        def record_for(*, source_global_index: int, horizon: int) -> SimpleNamespace:
            return SimpleNamespace(
                source_global_index=source_global_index,
                target_global_index=source_global_index + horizon,
                horizon=horizon,
                importance=(0.0,),
            )

    selected = _select_staged_p2_records_from_plan(
        plan=Plan(),
        dataset=Dataset(),
        predictive_cache=Cache(),
        segments=(
            SimpleNamespace(index=0, episode_index=10),
            SimpleNamespace(index=1, episode_index=20),
        ),
        episodes=episodes,
        start_optimizer_step=0,
        steps=1,
        world_size=2,
        horizon=1,
        allowed_episode_indices=frozenset({10, 20}),
    )

    assert tuple(item.record.source_global_index for item in selected) == (101, 201)
    assert tuple(item.plan_optimizer_step for item in selected) == (1, 1)


def test_staged_p2_never_combines_asymmetric_frozen_plan_steps() -> None:
    episodes = (
        SimpleNamespace(
            episode_key="episode-a",
            segment_index=0,
            sample_keys=("a0", "a1", "a2", "a3", "a4"),
        ),
        SimpleNamespace(
            episode_key="episode-b",
            segment_index=1,
            sample_keys=("b0", "b1", "b2", "b3", "b4"),
        ),
    )

    class Plan:
        total_steps = 3

        @staticmethod
        def global_batch(step: int) -> SimpleNamespace:
            indices = ((1, 0), (1, 1), (2, 2))[step]
            return SimpleNamespace(
                transitions=tuple(
                    SimpleNamespace(
                        episode_key=episode.episode_key,
                        transition_index=index,
                        sample=SimpleNamespace(sample_key=episode.sample_keys[index]),
                    )
                    for episode, index in zip(episodes, indices, strict=True)
                )
            )

    source_by_key = {
        key: offset + index
        for offset, episode in zip((100, 200), episodes, strict=True)
        for index, key in enumerate(episode.sample_keys)
    }

    class Dataset:
        @staticmethod
        def source_global_index_by_key(key: str) -> int:
            return source_by_key[key]

    class Cache:
        @staticmethod
        def record_for(*, source_global_index: int, horizon: int) -> SimpleNamespace:
            return SimpleNamespace(
                source_global_index=source_global_index,
                target_global_index=source_global_index + horizon,
                horizon=horizon,
                importance=(1.0,),
            )

    selected = _select_staged_p2_records_from_plan(
        plan=Plan(),
        dataset=Dataset(),
        predictive_cache=Cache(),
        segments=(
            SimpleNamespace(index=0, episode_index=10),
            SimpleNamespace(index=1, episode_index=20),
        ),
        episodes=episodes,
        start_optimizer_step=0,
        steps=1,
        world_size=2,
        horizon=1,
        allowed_episode_indices=frozenset({10, 20}),
    )

    assert tuple(item.record.source_global_index for item in selected) == (101, 201)
    assert tuple(item.plan_optimizer_step for item in selected) == (1, 1)


def test_staged_p2_reuses_the_p1_model_and_optimizer_without_action() -> None:
    source = _source()

    p1_update = source.index("for optimizer_step in range(args.steps):")
    staged_update = source.index("for p2_optimizer_step in range(args.staged_p2_steps):")
    assert p1_update < staged_update
    assert source.count("build_lingbot_representation_optimizer(") == 1
    assert source.count("load_model_weights(") == 1
    assert '"same_process_model_optimizer": True' in source
    assert (
        'STAGED_P2_REPORT_SCHEMA = "picf-next.lingbot-vla2-task-independent-staged-p2.v5"' in source
    )
    assert '"schema": "picf-next.task-independent-staged-p2-schedule.v3"' in source
    assert '"action_suffix_executed": False' in source
    assert '"task_scorer_present": False' in source
    assert "allowed_episode_indices=p2_training_sources" in source
    assert "_select_staged_p2_records_from_plan(" in source
    assert 'parser.add_argument("--p2-stream-plan", type=Path)' in source
    assert 'parser.add_argument("--p2-representation-split", type=Path)' in source
    assert 'parser.add_argument("--p2-causal-probe-cache-root", type=Path)' in source
    assert 'parser.add_argument("--p2-causal-replay-closure", type=Path)' in source
    assert "run_native_future_counterfactual_forwards(" in source
    assert "predictive_future_counterfactual_diagnostics(" in source
    assert '"schema": "picf-next.task-independent-p2-causal-schedule.v2"' in source
    assert "distinct_source_episodes=True" in source
    assert 'else "causal_audit"' in source
    assert '"task_prompt_entered_shared_host": True' in source
    assert '"loss_side_label_or_mask_entered_probe_forward": False' in source
    assert "P2 training sources escape the frozen P1 training domain" in source
    assert "P2 training overlaps P1 validation or held-out sources" in source
    assert "predictive_cache.contract.stream_plan_sha256 != p2_plan.plan_sha256" in source
    assert '"p1_stream_plan_sha256": plan.plan_sha256' in source
    assert '"p2_stream_plan_sha256": p2_plan.plan_sha256' in source
    assert "frozen-stream-causal-global-batch-subsequence-without-label-or-" in source
    assert "verify_representation_trial_split_training_evidence(" in source
    assert "predictive_valid_global_steps != args.staged_p2_steps" in source
    assert '"staged P2 consumed a non-training source episode"' in source
    assert "staged_objective_config = TaskIndependentEntityObjectiveConfig(" in source
    assert "predictive_weight=1.0" in source
    assert "num_steps=args.steps + args.staged_p2_steps" in source
    assert source.count("capacity_seeds=prefix_planned.augmentation_seeds") == 6
    assert "capacity_seeds=source_planned.augmentation_seeds" not in source
    assert '"predictive_readout"] != 0' in source
    assert '"action_only"] != 0' in source


def test_optimizer_family_counts_preserve_lazy_predictive_and_action_state() -> None:
    import torch

    names = {
        "model.picf_native_graph.relation.weight": torch.nn.Parameter(torch.ones(2)),
        "model.qwenvl.model.layer.weight": torch.nn.Parameter(torch.ones(2)),
        "model.picf_native_graph.predictive_readouts.dino_video.weight": (
            torch.nn.Parameter(torch.ones(2))
        ),
        "model.qwenvl_with_expert.qwen_expert.layer.weight": torch.nn.Parameter(torch.ones(2)),
        "model.action_out_proj.weight": torch.nn.Parameter(torch.ones(2)),
    }

    class Model:
        @staticmethod
        def named_parameters():
            return tuple(names.items())

    optimizer = torch.optim.AdamW(tuple(names.values()), lr=1e-3)
    names["model.picf_native_graph.relation.weight"].grad = torch.ones(2)
    names["model.qwenvl.model.layer.weight"].grad = torch.ones(2)
    optimizer.step()
    assert _optimizer_state_family_counts(Model(), optimizer) == {
        "native_graph": 1,
        "vlm_host": 1,
        "predictive_readout": 0,
        "action_expert": 0,
        "action_only": 0,
    }
    host_audit = _audit_optimizer_family_state(
        Model(),
        optimizer,
        torch,
        family="vlm host",
        fragment=".qwenvl.",
        expected_adamw_step=1,
    )
    assert host_audit["entries"] == 1
    assert host_audit["adamw_step_minimum"] == 1.0
    assert int(host_audit["local_nonzero_moment_elements"]) > 0

    optimizer.zero_grad(set_to_none=True)
    names["model.picf_native_graph.predictive_readouts.dino_video.weight"].grad = torch.ones(2)
    optimizer.step()
    assert _optimizer_state_family_counts(Model(), optimizer) == {
        "native_graph": 2,
        "vlm_host": 1,
        "predictive_readout": 1,
        "action_expert": 0,
        "action_only": 0,
    }
    predictive_audit = _audit_optimizer_family_state(
        Model(),
        optimizer,
        torch,
        family="predictive readout",
        fragment="predictive_readouts.dino_video",
        expected_adamw_step=1,
    )
    assert predictive_audit["entries"] == 1
    assert predictive_audit["adamw_step_maximum"] == 1.0


def test_optimizer_family_audit_accepts_an_empty_local_fsdp_shard_with_global_evidence() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.empty(0, 2))

    class Model:
        @staticmethod
        def named_parameters():
            return (("model.predictive_readouts.dino_video.weight", parameter),)

    optimizer = torch.optim.AdamW((parameter,), lr=1e-3)
    parameter.grad = torch.empty_like(parameter)
    optimizer.step()

    class TwoRankDist:
        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def all_gather_object(values: list[object], value: object) -> None:
            values[0] = value
            values[1] = {
                "error": None,
                "entries": 1,
                "adamw_entries": 1,
                "muon_entries": 0,
                "moment_elements": 4,
                "nonzero_moment_elements": 2,
                "adamw_steps": [1.0],
            }

    audit = _audit_optimizer_family_state(
        Model(),
        optimizer,
        torch,
        family="predictive readout",
        fragment="predictive_readouts.dino_video",
        expected_adamw_step=1,
        dist_module=TwoRankDist(),
    )

    assert audit["local_moment_elements"] == 0
    assert audit["local_nonzero_moment_elements"] == 0
    assert audit["global_moment_elements"] == 4
    assert audit["global_nonzero_moment_elements"] == 2


def test_p1_runner_uses_real_sidecar_schema_and_rank_zero_artifact_validation() -> None:
    source = _source()

    assert "picf_next.data.calvin_physical_supervision_schema" in source
    assert "picf_next.data.calvin_physical_supervision import" not in source
    assert source.count("validate_checkpoint(args.checkpoint_dir)") == 1
    assert source.count("validate_processor(args.processor_dir)") == 1
    assert source.index("if rank == 0:") < source.index("validate_checkpoint(args.checkpoint_dir)")
    assert "dist.broadcast_object_list(artifact_contract, src=0)" in source
    assert 'args.fsdp2_placement == "cpu_offload"' not in source


def test_p1_report_binds_runner_and_entity_implementation() -> None:
    root = Path(__file__).resolve().parents[2]
    files, digest = _implementation_provenance(root)

    assert len(digest) == 64
    assert "tools/run_lingbot_vla2_task_independent_p1.py" in files
    assert "src/picf_next/lingbot_native/entity_set_objective.py" in files
    assert "src/picf_next/lingbot_native/calvin_entity_training.py" in files
    assert "src/picf_next/lingbot_native/graph.py" in files
    assert "src/picf_next/lingbot_native/lattice_feasibility.py" in files
    assert "src/picf_next/lingbot_native/physical_sequence.py" in files
    assert "src/picf_next/lingbot_native/predictive_cache.py" in files
    assert "src/picf_next/lingbot_native/predictive_objective.py" in files
    assert "src/picf_next/lingbot_native/temporal.py" in files
    assert "src/picf_next/lingbot_native/visual_audit.py" in files
    assert "tools/lingbot_vla2_runtime_helpers.py" in files
    assert "tools/run_lingbot_vla2_native_g0.py" in files
    assert all(len(value) == 64 for value in files.values())
