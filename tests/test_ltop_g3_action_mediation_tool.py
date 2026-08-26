from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.run_lingbot_vla2_ltop_g3_action_mediation import (
    _action_targets_sha256,
    _require_canonical_bindings_applied,
    _resolve_physical_target_row,
    _validate_g3_training_checkpoint_manifest,
    build_g3_mediator_counterbalanced_schedule,
    build_g3_source_action_counterbalanced_schedule,
)


def test_g3_action_target_digest_uses_the_transformed_forward_abi() -> None:
    batch = SimpleNamespace(
        model_inputs={
            "actions": torch.tensor([[[1.0, 2.0]]]),
            "action_is_pad": torch.tensor([[False]]),
        }
    )

    assert len(_action_targets_sha256(batch)) == 64
    with pytest.raises(KeyError, match="actions"):
        _action_targets_sha256(
            SimpleNamespace(
                model_inputs={
                    "action.lingbot": torch.tensor([[[1.0, 2.0]]]),
                    "action.lingbot_is_pad": torch.tensor([[False]]),
                }
            )
        )


def test_g3_source_target_row_disables_only_unobservable_address_supervision() -> None:
    identities = ("visible", "hidden")
    bindings = (("visible", 3),)

    assert _resolve_physical_target_row(
        target_identity="visible",
        identity_keys=identities,
        eligible_track_indices=(0,),
        bindings=bindings,
        allow_unobservable=True,
    ) == (3, "bound-current-frame-target")
    assert _resolve_physical_target_row(
        target_identity="hidden",
        identity_keys=identities,
        eligible_track_indices=(0,),
        bindings=bindings,
        allow_unobservable=True,
    ) == (None, "unobservable-current-frame-target")
    assert _resolve_physical_target_row(
        target_identity=None,
        identity_keys=identities,
        eligible_track_indices=(0,),
        bindings=bindings,
        allow_unobservable=True,
    ) == (None, "no-singleton-source-target")


def test_g3_source_target_row_rejects_missing_or_unbound_visible_targets() -> None:
    with pytest.raises(RuntimeError, match="absent from inventory"):
        _resolve_physical_target_row(
            target_identity="missing",
            identity_keys=("visible",),
            eligible_track_indices=(0,),
            bindings=(("visible", 2),),
            allow_unobservable=True,
        )
    with pytest.raises(RuntimeError, match="eligible target identity"):
        _resolve_physical_target_row(
            target_identity="visible",
            identity_keys=("visible",),
            eligible_track_indices=(0,),
            bindings=(),
            allow_unobservable=True,
        )
    with pytest.raises(RuntimeError, match="target identity is unbound"):
        _resolve_physical_target_row(
            target_identity="hidden",
            identity_keys=("visible", "hidden"),
            eligible_track_indices=(0,),
            bindings=(("visible", 2),),
            allow_unobservable=False,
        )


def _source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "tools/run_lingbot_vla2_ltop_g3_action_mediation.py").read_text(encoding="utf-8")


def _training_checkpoint_manifest() -> dict[str, object]:
    return {
        "schema": "picf-next.ltop-g3-training-checkpoint.v5",
        "status": "PASS",
        "global_step": 256,
        "optimizer_saved": False,
        "format": "lingbot-fsdp2-dcp-model-only",
        "world_size": 2,
        "model_tree_schema": "picf-next.ltop-g3-model-dcp-tree.v1",
        "model_tree_sha256": "a" * 64,
        "action_supervision_schema": "picf-next.task-action-supervision.v1",
        "picf_source_contract": {
            "schema": "picf-next.g3-picf-source-contract.v1",
            "repository_commit": "1" * 40,
            "repository_tree": "2" * 40,
            "worktree_clean": True,
            "critical_file_sha256": {
                "tools/run_lingbot_vla2_ltop_g3_action_mediation.py": "3" * 64,
                "src/picf_next/lingbot_native/task_address_learning.py": "4" * 64,
                "src/picf_next/lingbot_native/task_action_supervision.py": "5" * 64,
            },
        },
        "task_address_supervision_depth": {
            "schema": "picf-next.action-consumable-task-address-depth.v1",
            "producer_layer_index": 34,
            "consumer_layer_index": 35,
            "layer_count": 36,
            "final_layer_excluded": True,
            "reason": "address-output-must-precede-a-later-action-attention-layer",
        },
        "action_information_set_schedule_sha256": "b" * 64,
        "training_final_model_local_state_sha256_by_rank": ["c" * 64, "d" * 64],
    }


def test_g3_cold_runner_validates_the_complete_v5_checkpoint_manifest() -> None:
    manifest = _training_checkpoint_manifest()

    digests, model_tree_sha256 = _validate_g3_training_checkpoint_manifest(
        manifest,
        expected_layer_count=36,
        expected_picf_source_contract=manifest["picf_source_contract"],
    )

    assert digests == ["c" * 64, "d" * 64]
    assert model_tree_sha256 == "a" * 64
    for field, value in (
        ("world_size", 4),
        ("format", "another-format"),
        ("global_step", 255),
        ("model_tree_sha256", "A" * 64),
        ("action_supervision_schema", "legacy"),
        ("action_information_set_schedule_sha256", None),
        ("training_final_model_local_state_sha256_by_rank", ["c" * 64]),
    ):
        changed = dict(manifest)
        changed[field] = value
        with pytest.raises(ValueError):
            _validate_g3_training_checkpoint_manifest(changed)

    wrong_depth = dict(manifest)
    wrong_depth["task_address_supervision_depth"] = {
        **manifest["task_address_supervision_depth"],
        "producer_layer_index": 35,
        "consumer_layer_index": 36,
        "layer_count": 37,
    }
    with pytest.raises(ValueError, match="loaded host graph"):
        _validate_g3_training_checkpoint_manifest(wrong_depth, expected_layer_count=36)

    wrong_source = {
        **manifest["picf_source_contract"],
        "repository_commit": "6" * 40,
    }
    with pytest.raises(ValueError, match="loaded runner"):
        _validate_g3_training_checkpoint_manifest(
            manifest,
            expected_picf_source_contract=wrong_source,
        )


def test_g3_uses_strict_stage_runtime_and_released_training_surfaces() -> None:
    source = _source()

    assert "prepare_lingbot_vla2_ltop_stage_transfer" in source
    assert "open_lingbot_vla2_ltop_stage_runtime" in source
    assert "build_lingbot_official_optimizer" in source
    assert "run_native_policy_training_forward" in source
    assert "policy.sample_actions(" in source
    assert "LINGBOT_RELEASED_ACTION_SAMPLING_STEPS" in source


def test_g3_uses_production_row_and_edge_interventions_without_a_new_model() -> None:
    source = _source()

    assert "build_label_blind_ltop_action_arms" in source
    assert "object_read_source_row_visible=arm.object_read_source_row_visible" in source
    assert "object_read_action_intervention=arm.object_read_action_intervention" in source
    assert "score_offline_ltop_action_mediation" in source
    assert "class G3" not in source
    assert "torch.nn.Linear" not in source
    assert "nn.Linear" not in source


def test_g3_receipt_uses_the_executed_layout_and_only_the_expanded_prefix() -> None:
    source = _source()

    assert "context.task_address_attention_layout" in source
    assert 'active_capture["layer_count"] < graph.config.num_layers' in source
    assert "action_consumable_task_address" in source
    assert 'active_capture["action_consumable_row_mass"] = receipt.row_mass' in source
    assert "row_mass_by_layer" not in source
    assert 'captured["final_row_mass"]' not in source
    assert "action_consumable_task_address_depth_contract" in source
    assert "action_consumable_task_address_depth_contract(" in source
    assert "graph.config.num_layers" in source
    assert "graph._instruction_span" not in source
    assert 'active_capture["prior_slice"]' not in source


def test_g3_seals_every_action_before_opening_offline_rows() -> None:
    source = _source()
    evaluation = source[source.index("def evaluate_scene(") : source.index("evaluation_scenes =")]

    seal = evaluation.index("seal_ltop_action_receipt(")
    target = evaluation.index("target_identity = scene[")
    score = evaluation.index("score_offline_ltop_action_mediation(")
    assert seal < target < score
    assert "distractor_identity = scene[" in evaluation
    assert '"target_identities"' in evaluation
    assert "[1 - prompt_index]" in evaluation


def test_g3_cold_action_can_diagnose_each_registered_information_set() -> None:
    source = _source()
    launcher = (
        Path(__file__).resolve().parents[1] / "adr165/run_ltop_g3_mediator_cold_action_2gpu.sh"
    ).read_text(encoding="utf-8")

    assert '"--evaluation-action-information-set"' in source
    assert "TaskAddressActionInformationSet(" in source
    assert '"evaluation_action_information_set"' in source
    assert "PICF_G3_EVALUATION_ACTION_INFORMATION_SET" in launcher
    assert '--evaluation-action-information-set "$evaluation_action_information_set"' in launcher


def test_g3_registered_gate_is_bounded_and_fail_closed() -> None:
    source = _source()

    assert "G3_DEFAULT_STEPS = 128" in source
    assert "G3_DEFAULT_EVAL_EVERY = 32" in source
    assert 'args.mode == "gate"' in source
    assert "action loss did not improve by at least five percent" in source
    assert "blocked-path difference-in-differences was nonpositive" in source
    assert "peak allocated memory reached the A100 safety bound" in source


def test_g3_mediator_trial_counterbalances_every_scene_prompt_cell() -> None:
    schedule = build_g3_mediator_counterbalanced_schedule(
        scene_prompt_keys=tuple((f"scene-{scene}", ("prompt-a", "prompt-b")) for scene in range(8)),
        steps=256,
    )

    assert schedule["single_forward_per_optimizer_step"] is True
    assert schedule["arm_counts"] == {"factual": 128, "mediator-required": 128}
    assert all(
        counts == {"factual": 8, "mediator-required": 8}
        for counts in schedule["cell_arm_counts"].values()
    )
    entries = schedule["entries"]
    assert all(entries[index]["arm"] != entries[index + 16]["arm"] for index in range(240))
    assert entries[1]["arm"] == entries[2]["arm"]
    assert len(schedule["sha256"]) == 64


def test_g3_source_action_trial_counterbalances_every_real_scene() -> None:
    schedule = build_g3_source_action_counterbalanced_schedule(
        scene_source_keys=tuple(
            (f"scene-{scene}", f"source-task-{scene}") for scene in range(8)
        ),
        steps=256,
    )

    assert schedule["single_forward_per_optimizer_step"] is True
    assert schedule["action_labels"] == "immutable-source-trajectory-only"
    assert schedule["crossed_prompts_used_for_action_loss"] is False
    assert schedule["arm_counts"] == {"factual": 128, "mediator-required": 128}
    assert all(
        counts == {"factual": 16, "mediator-required": 16}
        for counts in schedule["scene_arm_counts"].values()
    )
    entries = schedule["entries"]
    assert all(entries[index]["arm"] != entries[index + 8]["arm"] for index in range(248))
    assert len(schedule["sha256"]) == 64


def test_g3_smoke_reduces_only_redundant_evaluation_schedule() -> None:
    source = _source()

    assert 'prompt_batches = scene["batches"][:1] if args.mode == "smoke"' in source
    assert '{"validation": scenes["validation"][:1]}' in source
    assert 'if args.phase == "combined" and args.mode != "smoke":\n            record(0)' in source
    assert 'partitions = ("validation",) if mode == "smoke"' in source
    assert "build_label_blind_ltop_action_arms" in source
    assert '"picf-next.ltop-g3-smoke-stage-trace.v1"' in source
    for stage in (
        "forward-begin",
        "forward-done",
        "backward-begin",
        "backward-done",
        "gradient-metrics-done",
        "gradient-clip-done",
        "optimizer-done",
    ):
        assert f'stage="{stage}"' in source


def test_g3_training_phase_excludes_in_process_diagnostics_and_publishes_model() -> None:
    source = _source()
    training = source[
        source.index("for step in range(1, training_steps + 1):") : source.index(
            'if args.phase == "evaluation":\n            record(args.steps)'
        )
    ]

    assert 'G3_PHASES = ("combined", "training", "evaluation", "retention")' in source
    assert 'if args.phase == "combined" and (' in source
    assert 'if args.phase == "training":' in source
    assert '{"model": policy}' in source
    assert '"optimizer_saved": False' in source
    assert "G3_TRAINING_SCHEMA" in source
    assert "run_native_policy_observation_diagnostic_forward" not in source
    assert 'batch = batch_to_device(scene["source_batch"])' in training
    assert 'require_factual_action_supervision(supervision)' in training
    assert 'target_identity=scene["source_target_identity"]' in training
    assert "allow_unobservable_target=True" in training
    assert '"task_address_supervision_reason"' in training
    assert 'scene["batches"][prompt_index]' not in training
    assert 'canonical_gauge_by_scene' not in training
    assert "with torch.no_grad()" not in training
    assert '"official_action_loss": "immutable-source-task-action-pairs-only"' in source
    assert '"crossed_prompt_action_loss": False' in source


def test_g3_evaluation_phase_cold_restores_and_does_not_optimize() -> None:
    source = _source()

    assert 'elif args.phase == "evaluation":' in source
    assert 'phase="ltop-g3-staged-evaluation-cold-load"' in source
    assert "policy.requires_grad_(False)" in source
    assert "optimizer = None" in source
    assert '"trained_model_local_state_sha256"' in source
    assert 'training_steps = 0 if args.phase in {"evaluation", "retention"}' in source
    assert 'if args.phase == "evaluation":\n            record(args.steps)' in source
    assert "G3_EVALUATION_SCHEMA" in source


def test_g3_retention_phase_reuses_full_g2b_representation_gate_without_actions() -> None:
    source = _source()

    assert 'G3_PHASES = ("combined", "training", "evaluation", "retention")' in source
    assert 'args.phase in {"evaluation", "retention"}' in source
    assert 'training_steps = 0 if args.phase in {"evaluation", "retention"}' in source
    assert (
        'if args.phase in {"combined", "training"}:\n            if args.journal_dir is None:'
        in source
    )
    assert 'if args.phase != "evaluation":\n            if args.journal_dir is None:' not in source
    mode_start = source.index(
        'if args.phase == "retention":',
        source.index("policy.requires_grad_(False)"),
    )
    mode_stop = source.index("torch.cuda.synchronize(device)", mode_start)
    mode_contract = source[mode_start:mode_stop]
    assert "policy.train()" in mode_contract
    assert "graph.train()" in mode_contract
    assert "policy.eval()" in mode_contract
    assert "graph.eval()" in mode_contract
    assert "evaluate_retention_partition" in source
    retention_scene = source[
        source.index("def evaluate_retention_scene") : source.index(
            "def evaluate_retention_partition"
        )
    ]
    assert "result, row_mass = training_forward(" in retention_scene
    assert "TaskAddressActionInformationSet.FACTUAL" in retention_scene
    assert "* batch.routing.batch_size" in retention_scene
    assert "context=result.context" in source
    assert "_scene_metrics(" in source
    assert "_physical_relation_prompt_drift(" in source
    assert "_scene_level_robustness(" in source
    assert '"scientific_action_evidence": False' in source


def test_g3_retention_launcher_is_fixed_to_the_cold_g3_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "adr164/run_ltop_g3_retention_2gpu.sh").read_text(encoding="utf-8")

    assert "--phase retention" in launcher
    assert "--mode gate" in launcher
    assert "adr163-g3-training-540b490-v1" in launcher
    assert "--nproc_per_node=2" in launcher
    assert "source and output must live under /mnt" in launcher


def test_g3_canonical_gauge_checks_the_assignment_used_by_the_loss() -> None:
    canonical = (("blue_block", 2), ("red_block", 3))

    _require_canonical_bindings_applied(
        physical={
            "bindings": canonical,
            "matched_bindings": (("blue_block", 3), ("red_block", 2)),
        },
        canonical_bindings=canonical,
    )

    with pytest.raises(RuntimeError, match="failed to apply the canonical physical row gauge"):
        _require_canonical_bindings_applied(
            physical={
                "bindings": (("blue_block", 3), ("red_block", 2)),
                "matched_bindings": canonical,
            },
            canonical_bindings=canonical,
        )
