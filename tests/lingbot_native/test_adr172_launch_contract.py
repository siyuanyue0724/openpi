from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tools.run_lingbot_vla2_ltop_adr172_direct_posterior import (
    ADR172_ACTION_HEAD_SCOPE,
    ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
    ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES,
    ADR172_GUIDEDVLA_UPSTREAM_CONTRACT,
    G3_DIRECT_ACTION_CAUSAL_SURFACE,
    G3_DIRECT_ROUTE,
    G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE,
    G3_PHYSICAL_SET_LOSS_COMPONENTS,
    _cold_causal_partition_evaluation,
    _direct_posterior_head_indices,
    _direct_posterior_registered_layer_indices,
    _evaluation_failures,
    _g2_physical_retention_reference,
    _physical_retention_summary,
    _retention_failures,
    _score_to_json,
    _training_failures,
    _validate_g3_training_checkpoint_manifest,
    build_g3_direct_source_schedule,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/run_lingbot_vla2_ltop_adr172_direct_posterior.py"
SMOKE = ROOT / "adr172/run_direct_posterior_smoke_2gpu.sh"
TRIAL = ROOT / "adr172/run_direct_posterior_trial_2gpu_256.sh"
COLD_ACTION = ROOT / "adr172/run_direct_posterior_cold_action_2gpu.sh"
RETENTION = ROOT / "adr172/run_direct_posterior_retention_2gpu.sh"
GUIDEDVLA_SMOKE = ROOT / "adr172/run_guidedvla_fixed_object_heads_smoke_2gpu.sh"
GUIDEDVLA_TRIAL = ROOT / "adr172/run_guidedvla_fixed_object_heads_trial_2gpu_256.sh"
GUIDEDVLA_COLD_ACTION = ROOT / "adr172/run_guidedvla_fixed_object_heads_cold_action_2gpu.sh"
GUIDEDVLA_RETENTION = ROOT / "adr172/run_guidedvla_fixed_object_heads_retention_2gpu.sh"
MATCHED_LBOT = ROOT / "adr172/run_matched_lbot_2gpu_256.sh"


def test_adr172_matched_lbot_launcher_locks_the_exact_two_gpu_stream() -> None:
    source = MATCHED_LBOT.read_text(encoding="utf-8")

    assert "--nproc_per_node=2" in source
    assert "--adr172-exact-stream" in source
    assert "--runtime-hotfix" in source
    assert "--stage-checkpoint" in source
    assert "--g2-report" in source
    assert "--execution-contract" in source
    assert "--offline-labels" in source
    assert "configs/vla/robotwin/robotwin.yaml" in source
    assert "--evaluation-steps 0,32,64,96,128,160,192,224,256" in source
    assert "--steps 256" in source
    assert "--seed 20260813" in source
    assert "--learning-rate 1e-4" in source
    assert "--max-grad-norm 1.0" in source
    assert "--fsdp2-placement selective-embedding-offload" in source
    assert "--cuda-allocator expandable-segments" in source
    assert "CUDA_VISIBLE_DEVICES=0,1" in source
    assert "status --porcelain=v1 --untracked-files=all" in source


@dataclass(frozen=True)
class _SerializedScore:
    active_action_counts: torch.Tensor
    effect: torch.Tensor


def test_adr172_score_json_preserves_discrete_action_counts() -> None:
    payload = _score_to_json(
        _SerializedScore(
            active_action_counts=torch.tensor([7], dtype=torch.int64),
            effect=torch.tensor([0.25], dtype=torch.float32),
        )
    )

    assert payload["active_action_counts"] == [7]
    assert isinstance(payload["active_action_counts"][0], int)
    assert payload["effect"] == [0.25]


def test_adr172_guidedvla_profile_registers_only_fixed_native_object_heads() -> None:
    assert (
        _direct_posterior_head_indices(
            ADR172_ACTION_HEAD_SCOPE,
            head_count=32,
        )
        is None
    )
    assert _direct_posterior_head_indices(
        ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
        head_count=32,
    ) == (0, 1)

    with pytest.raises(ValueError, match="omits one registered GuidedVLA object head"):
        _direct_posterior_head_indices(
            ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
            head_count=1,
        )


def _training_rank_report(
    *,
    target_valid: bool,
    reason: str,
    head_scope: str = ADR172_ACTION_HEAD_SCOPE,
) -> dict[str, object]:
    target_mass = 0.125 if target_valid else 0.0
    grounding_loss = 2.0 if target_valid else 0.0
    return {
        "rank": 0,
        "all_gradients_finite": True,
        "cuda_memory_bytes": {"peak_allocated": 1},
        "gradient_metrics_history": [
            {
                "native_graph_norm": 1.0,
                "shared_host_norm": 1.0,
                "shared_q_projection_norm": 1.0,
                "shared_k_projection_norm": 1.0,
                "action_output_norm": 1.0,
            }
        ],
        "action_losses": [0.5],
        "action_supervision_history": [
            {
                "schema": "picf-next.task-action-supervision.v1",
                "scope": "factual-action",
                "official_action_loss_enabled": True,
                "source_task_key": "task",
                "candidate_task_key": "task",
                "source_instruction_sha256": "a" * 64,
                "candidate_instruction_sha256": "a" * 64,
            }
        ],
        "direct_grounding_losses": [grounding_loss],
        "direct_grounding_history": [
            {
                "target_valid": target_valid,
                "target_row": 3 if target_valid else None,
                "head_scope": head_scope,
                "head_indices": (
                    None
                    if head_scope == ADR172_ACTION_HEAD_SCOPE
                    else list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES)
                ),
                "registered_layer_indices": [32, 35],
                "layers": [
                    {
                        "layer_index": layer_index,
                        "target_mass_mean": target_mass,
                        "total_posterior_mass_mean": 0.5,
                    }
                    for layer_index in (32, 35)
                ],
            }
        ],
        "task_address_supervision_history": [{"enabled": target_valid, "reason": reason}],
        "arm_journal": {"record_count": 1},
    }


def _training_checkpoint_manifest(
    *,
    head_scope: str = ADR172_ACTION_HEAD_SCOPE,
) -> dict[str, object]:
    manifest = {
        "schema": "picf-next.adr172-direct-posterior-training-checkpoint.v1",
        "status": "PASS",
        "global_step": 256,
        "optimizer_saved": False,
        "format": "lingbot-fsdp2-dcp-model-only",
        "world_size": 2,
        "model_tree_schema": "picf-next.ltop-g3-model-dcp-tree.v1",
        "model_tree_sha256": "a" * 64,
        "action_supervision_schema": "picf-next.task-action-supervision.v1",
        "direct_action_causal_surface": G3_DIRECT_ACTION_CAUSAL_SURFACE,
        "direct_route": G3_DIRECT_ROUTE,
        "picf_source_contract": {
            "schema": "picf-next.g3-picf-source-contract.v1",
            "repository_commit": "1" * 40,
            "repository_tree": "2" * 40,
            "worktree_clean": True,
            "critical_file_sha256": {
                "tools/run_lingbot_vla2_ltop_adr172_direct_posterior.py": "3" * 64,
                "src/picf_next/lingbot_native/action_posterior_receipt.py": "4" * 64,
                "src/picf_next/lingbot_native/action_posterior_collector.py": "5" * 64,
                "src/picf_next/lingbot_native/action_posterior_learning.py": "6" * 64,
                "src/picf_next/lingbot_native/graph.py": "7" * 64,
                "src/picf_next/lingbot_native/host.py": "8" * 64,
                "src/picf_next/lingbot_native/ltop_action_mediation.py": "9" * 64,
                "src/picf_next/lingbot_native/task_address_target.py": "a" * 64,
                "src/picf_next/lingbot_native/task_action_supervision.py": "b" * 64,
            },
        },
        "direct_posterior_registered_layer_indices": [32, 35],
        "direct_posterior_head_scope": head_scope,
        "direct_route_schedule_sha256": "b" * 64,
        "training_final_model_local_state_sha256_by_rank": ["c" * 64, "d" * 64],
        "source_stage_checkpoint": "/mnt/picf-next/checkpoints/g2b",
        "g2_report_sha256": "e" * 64,
        "runtime_source_contract": {
            "native_patch_sha256": "f" * 64,
            "runtime_hotfix_sha256": None,
            "runtime_patched_source_sha256": {
                "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py": "1" * 64,
            },
        },
    }
    if head_scope == ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE:
        manifest.update(
            {
                "direct_posterior_head_indices": list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES),
                "direct_grounding_weight": 0.001,
                "direct_grounding_upstream_contract": ADR172_GUIDEDVLA_UPSTREAM_CONTRACT,
            }
        )
    return manifest


def _evaluation_rank_report(
    *,
    rank: int,
    positive_scene_count: int,
    normalized_positive_scene_count: int | None = None,
) -> dict[str, object]:
    if not 0 <= positive_scene_count <= 4:
        raise ValueError("test positive count exceeds the rank scene axis")
    if normalized_positive_scene_count is None:
        normalized_positive_scene_count = positive_scene_count
    if not 0 <= normalized_positive_scene_count <= 4:
        raise ValueError("test normalized-positive count exceeds the rank scene axis")

    def partition_report(partition: str) -> dict[str, object]:
        scenes = []
        for index in range(4):
            positive = index < positive_scene_count
            normalized_positive = index < normalized_positive_scene_count
            crossed = 0.1 if positive else -0.01
            normalized_crossed = 0.5 if normalized_positive else -0.05
            item_id = f"{partition}-{rank}-{index}"
            sample_key = f"sample-{partition}-{rank}-{index}"
            scenes.append(
                {
                    "item_id": item_id,
                    "sample_key": sample_key,
                    "prompt_count": 2,
                    "shared_row_gauge": True,
                    "prompts": [
                        {
                            "target_identity": "object-a",
                            "matched_distractor_identity": "object-b",
                            "target_row": 0,
                            "matched_distractor_row": 1,
                        },
                        {
                            "target_identity": "object-b",
                            "matched_distractor_identity": "object-a",
                            "target_row": 1,
                            "matched_distractor_row": 0,
                        },
                    ],
                    "score": {
                        "sample_keys": [sample_key],
                        "active_action_counts": [7],
                        "blocked_placebo_integrity_verified": True,
                        "replay_floor_rms": [0.0, 0.0],
                        "max_replay_floor_rms": 0.0,
                        "prompt_mean_factual_all_posterior_block_effect_rms": [0.2, 0.2],
                        "minimum_prompt_factual_all_posterior_block_effect_rms": 0.2,
                        "crossed_prompt_target_selectivity": [crossed],
                        "crossed_prompt_selectivity_over_all_posterior_block": [normalized_crossed],
                        "mean_crossed_prompt_target_selectivity": crossed,
                        "mean_crossed_prompt_selectivity_over_all_posterior_block": (
                            normalized_crossed
                        ),
                        "positive_crossed_prompt_target_selectivity_count": int(positive),
                        "sample_count": 1,
                    },
                }
            )
        return {
            "scene_count": len(scenes),
            "prompt_count": 2 * len(scenes),
            "max_replay_floor_rms": 0.0,
            "scenes": scenes,
        }

    return {
        "rank": rank,
        "direct_action_causal_surface": G3_DIRECT_ACTION_CAUSAL_SURFACE,
        "cuda_memory_bytes": {"peak_allocated": 1},
        "history": [
            {
                "validation": partition_report("validation"),
                "heldout": partition_report("heldout"),
            }
        ],
    }


def _retention_rank_report(
    *,
    rank: int,
    physical_loss: float | None = None,
    scene_losses: tuple[float, float, float, float] | None = None,
    reference_components: bool = False,
) -> dict[str, object]:
    if scene_losses is None:
        if physical_loss is None:
            raise ValueError("test retention report requires physical loss evidence")
        scene_losses = (physical_loss,) * 4
    if physical_loss is None:
        physical_loss = sum(scene_losses) / len(scene_losses)

    def partition_report(partition: str) -> tuple[dict[str, object], dict[str, object]]:
        current_scenes = []
        reference_scenes = []
        for index, scene_loss in enumerate(scene_losses):
            item_id = f"{partition}-{rank}-{index}"
            sample_key = f"sample-{partition}-{rank}-{index}"
            current_scenes.append(
                {
                    "item_id": item_id,
                    "sample_key": sample_key,
                    "mean_physical_set_loss": scene_loss,
                    "physical_set_loss_components": {
                        name: 0.1 for name in G3_PHYSICAL_SET_LOSS_COMPONENTS
                    },
                }
            )
            reference_scenes.append(
                {
                    "item_id": item_id,
                    "sample_key": sample_key,
                    "mean_physical_set_loss": 1.0,
                    "physical_set_loss_components": (
                        {name: 0.1 for name in G3_PHYSICAL_SET_LOSS_COMPONENTS}
                        if reference_components
                        else None
                    ),
                }
            )
        current = {
            "scene_count": 4,
            "prompt_count": 8,
            "shared_row_gauge": True,
            "physical_prompt_drift_max_abs": 0.0,
            "mean_physical_set_loss": physical_loss,
            "direct_action_diagnostic": {
                "minimum_adoption_mass": 0.1,
                "metric_self_checks": {"matched_row_permutation_max_abs_error": 0.0},
            },
            "scenes": current_scenes,
        }
        reference = {
            "mean_physical_set_loss": 1.0,
            "scenes": reference_scenes,
            "component_gate": {
                "available_in_g2_reference": reference_components,
                "components": (
                    list(G3_PHYSICAL_SET_LOSS_COMPONENTS) if reference_components else []
                ),
                "gap": (
                    None
                    if reference_components
                    else ("accepted G2b report does not publish per-scene physical loss components")
                ),
            },
        }
        return current, reference

    validation, validation_reference = partition_report("validation")
    heldout, heldout_reference = partition_report("heldout")
    return {
        "rank": rank,
        "cuda_memory_bytes": {"peak_allocated": 1},
        "g2_physical_retention_reference": {
            "validation": validation_reference,
            "heldout": heldout_reference,
        },
        "history": [{"validation": validation, "heldout": heldout}],
    }


def test_adr172_training_accepts_visible_and_explicitly_masked_grounding() -> None:
    visible = _training_rank_report(
        target_valid=True,
        reason="bound-current-frame-target",
    )
    masked = _training_rank_report(
        target_valid=False,
        reason="unobservable-current-frame-target",
    )
    visible["rank"] = 0
    masked["rank"] = 1

    assert _training_failures([visible, masked], mode="smoke") == []


def test_adr172_training_accepts_the_registered_guidedvla_head_scope() -> None:
    visible = _training_rank_report(
        target_valid=True,
        reason="bound-current-frame-target",
        head_scope=ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
    )
    masked = _training_rank_report(
        target_valid=False,
        reason="unobservable-current-frame-target",
        head_scope=ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
    )
    visible["rank"] = 0
    masked["rank"] = 1

    assert (
        _training_failures(
            [visible, masked],
            mode="smoke",
            head_scope=ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
        )
        == []
    )


def test_adr172_training_rejects_unexplained_zero_target_receipt() -> None:
    visible = _training_rank_report(
        target_valid=True,
        reason="bound-current-frame-target",
    )
    invalid = _training_rank_report(target_valid=False, reason="bound-current-frame-target")
    visible["rank"] = 0
    invalid["rank"] = 1

    assert _training_failures([visible, invalid], mode="smoke") == [
        "rank 1: direct posterior receipt is invalid or inconsistently masked"
    ]


def test_adr172_checkpoint_validates_the_direct_posterior_contract() -> None:
    manifest = _training_checkpoint_manifest()

    digests, model_tree_sha256 = _validate_g3_training_checkpoint_manifest(
        manifest,
        expected_layer_count=36,
        expected_picf_source_contract=manifest["picf_source_contract"],
        expected_source_stage_checkpoint=manifest["source_stage_checkpoint"],
        expected_g2_report_sha256=manifest["g2_report_sha256"],
        expected_runtime_source_contract=manifest["runtime_source_contract"],
    )

    assert digests == ["c" * 64, "d" * 64]
    assert model_tree_sha256 == "a" * 64
    assert _direct_posterior_registered_layer_indices(36) == (32, 35)


def test_adr172_checkpoint_validates_the_guidedvla_source_bound_profile() -> None:
    manifest = _training_checkpoint_manifest(head_scope=ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE)

    digests, model_tree_sha256 = _validate_g3_training_checkpoint_manifest(
        manifest,
        expected_head_scope=ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
    )

    assert digests == ["c" * 64, "d" * 64]
    assert model_tree_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("direct_posterior_head_indices", [0, 2], "head indices differ"),
        ("direct_grounding_weight", 1.0, "weight differs"),
        ("direct_grounding_upstream_contract", None, "upstream contract differs"),
    ],
)
def test_adr172_checkpoint_rejects_guidedvla_profile_drift(
    field: str,
    replacement: object,
    message: str,
) -> None:
    manifest = _training_checkpoint_manifest(head_scope=ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE)
    manifest[field] = replacement

    with pytest.raises(ValueError, match=message):
        _validate_g3_training_checkpoint_manifest(
            manifest,
            expected_head_scope=ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "source_stage_checkpoint",
            "/mnt/picf-next/checkpoints/another-g2b",
            "source stage checkpoint differs",
        ),
        ("g2_report_sha256", "0" * 64, "G2 report differs"),
    ],
)
def test_adr172_checkpoint_rejects_stage_or_g2_lineage_mismatch(
    field: str,
    replacement: object,
    message: str,
) -> None:
    manifest = _training_checkpoint_manifest()
    mismatched = {**manifest, field: replacement}

    with pytest.raises(ValueError, match=message):
        _validate_g3_training_checkpoint_manifest(
            mismatched,
            expected_source_stage_checkpoint=manifest["source_stage_checkpoint"],
            expected_g2_report_sha256=manifest["g2_report_sha256"],
        )


def test_adr172_checkpoint_rejects_runtime_source_lineage_mismatch() -> None:
    manifest = _training_checkpoint_manifest()
    runtime_source_contract = manifest["runtime_source_contract"]
    assert isinstance(runtime_source_contract, dict)
    mismatched = {
        **manifest,
        "runtime_source_contract": {
            **runtime_source_contract,
            "runtime_hotfix_sha256": "0" * 64,
        },
    }

    with pytest.raises(ValueError, match="runtime source contract differs"):
        _validate_g3_training_checkpoint_manifest(
            mismatched,
            expected_runtime_source_contract=runtime_source_contract,
        )


def test_adr172_cold_evaluator_separates_training_and_evaluator_source_identity() -> None:
    manifest = _training_checkpoint_manifest()
    evaluator_source = {
        **manifest["picf_source_contract"],
        "repository_commit": "9" * 40,
    }

    _validate_g3_training_checkpoint_manifest(manifest)
    with pytest.raises(ValueError, match="PICF source identity differs"):
        _validate_g3_training_checkpoint_manifest(
            manifest,
            expected_picf_source_contract=evaluator_source,
        )

    source = RUNNER.read_text(encoding="utf-8")
    assert "expected_picf_source_contract=picf_source_contract" not in source
    assert '"trained_picf_source_contract"' in source


@pytest.mark.parametrize(
    "field",
    ["source_stage_checkpoint", "g2_report_sha256", "runtime_source_contract"],
)
def test_adr172_checkpoint_lineage_fields_are_required(field: str) -> None:
    manifest = _training_checkpoint_manifest()
    del manifest[field]

    with pytest.raises(ValueError):
        _validate_g3_training_checkpoint_manifest(manifest)


def test_adr172_checkpoint_rejects_wrong_or_legacy_route_contracts() -> None:
    manifest = _training_checkpoint_manifest()
    wrong_layers = {**manifest, "direct_posterior_registered_layer_indices": [31, 35]}
    with pytest.raises(ValueError, match="loaded host graph"):
        _validate_g3_training_checkpoint_manifest(wrong_layers, expected_layer_count=36)

    wrong_heads = {**manifest, "direct_posterior_head_scope": "one-head"}
    with pytest.raises(ValueError, match="head scope"):
        _validate_g3_training_checkpoint_manifest(wrong_heads)

    legacy = {
        **manifest,
        "task_address_supervision_depth": {"layer_count": 36},
    }
    with pytest.raises(ValueError, match="rejected two-hop"):
        _validate_g3_training_checkpoint_manifest(legacy)

    legacy_information_set = {
        **manifest,
        "action_information_set_schedule_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="rejected action-information"):
        _validate_g3_training_checkpoint_manifest(legacy_information_set)


def test_adr172_training_path_uses_native_direct_posterior_receipts() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "RegisteredActionPosteriorReceiptCollector" in source
    assert "action_posterior_target_mass_loss" in source
    assert "posterior_adoption_route=torch.ones" in source
    assert "action_attention_callback=collector" in source
    assert "args.direct_grounding_weight * grounding_loss" in source
    assert '"shared_q_projection", "q_proj"' in source
    assert '"shared_k_projection", "k_proj"' in source
    assert source.index("prepublication_outcome") < source.index("checkpointer.save(")
    assert 'if args.phase == "training" and not training_prepublication_failures:' in source
    assert "posterior_action_row_visible=posterior_row_visible" in source
    assert "direct_posterior_action_row_visibility(arm)" in source
    assert "aggregate_action_posterior_distribution(" in source
    assert "object_read_action_intervention=" not in source
    assert "object_read_source_row_visible=" not in source
    assert "TaskAddressActionInformationSet" not in source
    assert "evaluation-action-information-set" not in source
    assert "scene_contract_items = local_items" in source
    assert 'if args.mode == "direct-trial"\n            else local_items' not in source


def test_adr172_source_schedule_has_one_structural_direct_route() -> None:
    schedule = build_g3_direct_source_schedule(
        scene_source_keys=(("scene-a", "task-a"), ("scene-b", "task-b")),
        steps=4,
    )

    assert schedule["route"] == G3_DIRECT_ROUTE
    assert schedule["scene_counts"] == {"scene-a": 2, "scene-b": 2}
    assert {entry["route"] for entry in schedule["entries"]} == {G3_DIRECT_ROUTE}
    assert all("arm" not in entry for entry in schedule["entries"])


def test_adr172_formal_evaluation_requires_three_quarters_positive() -> None:
    assert (
        _evaluation_failures(
            [
                _evaluation_rank_report(rank=0, positive_scene_count=3),
                _evaluation_rank_report(rank=1, positive_scene_count=3),
            ],
            mode="gate",
        )
        == []
    )

    failures = _evaluation_failures(
        [
            _evaluation_rank_report(rank=0, positive_scene_count=3),
            _evaluation_rank_report(rank=1, positive_scene_count=2),
        ],
        mode="gate",
    )
    assert "validation: jointly positive causal scenes 5 < 6" in failures
    assert "heldout: jointly positive causal scenes 5 < 6" in failures


def test_adr172_formal_evaluation_requires_scene_level_normalized_selectivity() -> None:
    failures = _evaluation_failures(
        [
            _evaluation_rank_report(
                rank=0,
                positive_scene_count=3,
                normalized_positive_scene_count=3,
            ),
            _evaluation_rank_report(
                rank=1,
                positive_scene_count=3,
                normalized_positive_scene_count=2,
            ),
        ],
        mode="gate",
    )

    assert "validation: jointly positive causal scenes 5 < 6" in failures
    assert "heldout: jointly positive causal scenes 5 < 6" in failures


def test_adr172_cold_gate_rejects_cross_partition_overlap() -> None:
    reports = [
        _evaluation_rank_report(rank=0, positive_scene_count=3),
        _evaluation_rank_report(rank=1, positive_scene_count=3),
    ]
    reports[0]["history"][0]["heldout"]["scenes"][0]["sample_key"] = reports[0]["history"][0][
        "validation"
    ]["scenes"][0]["sample_key"]

    assert "G3 validation and heldout causal samples overlap" in _evaluation_failures(
        reports,
        mode="gate",
    )


def test_adr172_cold_gate_rejects_noncanonical_crossed_rows() -> None:
    reports = [
        _evaluation_rank_report(rank=0, positive_scene_count=3),
        _evaluation_rank_report(rank=1, positive_scene_count=3),
    ]
    reports[0]["history"][0]["validation"]["scenes"][0]["prompts"][1]["matched_distractor_row"] = 2

    failures = _evaluation_failures(reports, mode="gate")

    assert any(
        "crossed prompts do not reverse one canonical row pair" in value for value in failures
    )


def test_adr172_cold_summary_recomputes_serialized_scene_effects() -> None:
    reports = [
        _evaluation_rank_report(rank=0, positive_scene_count=3),
        _evaluation_rank_report(rank=1, positive_scene_count=3),
    ]
    reports[0]["history"][0]["validation"]["scenes"][0]["score"][
        "mean_crossed_prompt_target_selectivity"
    ] = 9.0

    summary = _cold_causal_partition_evaluation(
        reports,
        partition="validation",
        expected_scenes_per_rank=4,
        apply_scientific_gate=True,
    )

    assert any(
        "serialized mean_crossed_prompt_target_selectivity differs from raw evidence" in value
        for value in summary["failures"]
    )


def test_adr172_retention_uses_physical_g2b_endpoint_not_object_read_margins() -> None:
    assert (
        _retention_failures(
            [
                _retention_rank_report(rank=0, physical_loss=0.9),
                _retention_rank_report(rank=1, physical_loss=0.8),
            ]
        )
        == []
    )

    failures = _retention_failures(
        [
            _retention_rank_report(rank=0, physical_loss=1.1),
            _retention_rank_report(rank=1, physical_loss=0.8),
        ]
    )
    assert any(
        failure.startswith("rank 0: validation mean physical set loss regressed")
        for failure in failures
    )
    assert any(
        failure.startswith("rank 0: heldout mean physical set loss regressed")
        for failure in failures
    )


def test_adr172_physical_summary_handles_scene_shape_that_broke_old_robustness() -> None:
    reports = [
        _retention_rank_report(rank=0, physical_loss=0.9),
        _retention_rank_report(rank=1, physical_loss=0.8),
    ]

    summary = _physical_retention_summary(reports, partition="validation")

    assert summary["status"] == "PASS"
    assert summary["absolute_tolerance"] == G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE
    assert summary["all_paired_scenes_within_tolerance"] is True
    assert summary["component_gate"]["status"] == "NOT_ENFORCED_G2_REFERENCE_GAP"
    assert summary["action_diagnostic_gating"] is False
    assert all(
        "mean_margin" not in scene
        for report in reports
        for scene in report["history"][0]["validation"]["scenes"]
    )
    source = RUNNER.read_text(encoding="utf-8")
    assert "_scene_level_robustness" not in source
    assert 'report["physical_retention_summary"]' in source
    assert '"physical_set_loss_absolute_tolerance"' in source


def test_adr172_action_diagnostic_does_not_change_physical_retention_status() -> None:
    reports = [
        _retention_rank_report(rank=0, physical_loss=0.9),
        _retention_rank_report(rank=1, physical_loss=0.8),
    ]
    for report in reports:
        for partition in ("validation", "heldout"):
            report["history"][0][partition]["direct_action_diagnostic"] = {
                "minimum_adoption_mass": 0.0,
                "metric_self_checks": {"matched_row_permutation_max_abs_error": float("inf")},
            }

    assert _retention_failures(reports) == []


def test_adr172_paired_scene_gate_catches_aggregate_masking() -> None:
    reports = [
        _retention_rank_report(rank=0, scene_losses=(1.2, 0.6, 0.6, 0.6)),
        _retention_rank_report(rank=1, physical_loss=0.8),
    ]

    summary = _physical_retention_summary(reports, partition="validation")
    failures = _retention_failures(reports)

    rank_zero_partition = next(pair for pair in summary["partition_pairs"] if pair["rank"] == 0)
    assert rank_zero_partition["passed"] is True
    assert summary["all_paired_scenes_within_tolerance"] is False
    assert any(
        failure.startswith("rank 0: validation scene validation-0-0 physical set loss regressed")
        for failure in failures
    )


def test_adr172_paired_component_gate_is_enforced_when_g2_publishes_components() -> None:
    reports = [
        _retention_rank_report(rank=0, physical_loss=0.9, reference_components=True),
        _retention_rank_report(rank=1, physical_loss=0.8, reference_components=True),
    ]
    reports[0]["history"][0]["validation"]["scenes"][0]["physical_set_loss_components"][
        "mask_focal"
    ] = 0.2

    summary = _physical_retention_summary(reports, partition="validation")

    assert summary["component_gate"]["status"] == "ENFORCED"
    assert any(
        "scene validation-0-0 physical component mask_focal regressed" in failure
        for failure in summary["failures"]
    )


def test_adr172_extracts_exact_g2b_physical_retention_reference() -> None:
    def partition_report(partition: str, value: float, rank: int) -> dict[str, object]:
        return {
            "mean_physical_set_loss": value,
            "scenes": [
                {
                    "item_id": f"{partition}-{rank}-{index}",
                    "sample_key": f"sample-{partition}-{rank}-{index}",
                    "mean_physical_set_loss": value,
                }
                for index in range(4)
            ],
        }

    report = {
        "status": "PASS",
        "rank_reports": [
            {
                "rank": rank,
                "history": [
                    {
                        "validation": partition_report("validation", 0.5 + rank, rank),
                        "heldout": partition_report("heldout", 0.75 + rank, rank),
                    }
                ],
            }
            for rank in range(2)
        ],
    }

    reference = _g2_physical_retention_reference(report)

    assert reference[0]["validation"]["mean_physical_set_loss"] == 0.5
    assert reference[1]["heldout"]["mean_physical_set_loss"] == 1.75
    assert reference[0]["validation"]["component_gate"] == {
        "available_in_g2_reference": False,
        "components": [],
        "gap": "accepted G2b report does not publish per-scene physical loss components",
    }
    assert all(
        scene["physical_set_loss_components"] is None
        for scene in reference[0]["validation"]["scenes"]
    )


def test_adr172_g2_reference_fails_closed_without_per_scene_physical_metrics() -> None:
    report = {
        "status": "PASS",
        "rank_reports": [
            {
                "rank": rank,
                "history": [
                    {
                        "validation": {"mean_physical_set_loss": 0.5},
                        "heldout": {"mean_physical_set_loss": 0.75},
                    }
                ],
            }
            for rank in range(2)
        ],
    }

    with pytest.raises(ValueError, match="paired scene axis"):
        _g2_physical_retention_reference(report)


def test_adr172_runner_imports_and_exposes_the_bounded_gpu_contract() -> None:
    completed = subprocess.run(
        (sys.executable, str(RUNNER), "--help"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--direct-grounding-weight" in completed.stdout
    assert "--direct-posterior-head-scope" in completed.stdout
    assert "--phase {combined,training,evaluation,retention}" in completed.stdout
    assert "--steps STEPS" in completed.stdout


def test_adr172_launchers_are_persistent_bounded_and_training_only() -> None:
    smoke = SMOKE.read_text(encoding="utf-8")
    trial = TRIAL.read_text(encoding="utf-8")

    for launcher in (smoke, trial):
        assert "run_lingbot_vla2_ltop_adr172_direct_posterior.py" in launcher
        assert '[[ "$run_root" == /mnt/*' in launcher
        assert "requires an exact clean PICF checkout" in launcher
        assert "--phase training" in launcher
        assert '--direct-grounding-weight "$direct_grounding_weight"' in launcher
        assert '--direct-posterior-head-scope "$direct_posterior_head_scope"' in launcher
        assert "guidedvla-fixed-object-heads-0-1) direct_grounding_weight=0.001" in launcher
        assert "--nproc_per_node=2" in launcher
        assert "timeout --signal=TERM --kill-after=60s" in launcher
        assert "--g2-report" in launcher
        assert "--offline-labels" in launcher
        assert "--physical-sidecar-manifest-sha256" in launcher

    assert "--mode smoke" in smoke
    assert "--steps 8" in smoke
    assert "--progress-every 1" in smoke
    assert "--mode direct-trial" in trial
    assert "--steps 256" in trial
    assert "--progress-every 8" in trial

    for wrapper in (GUIDEDVLA_SMOKE, GUIDEDVLA_TRIAL):
        source = wrapper.read_text(encoding="utf-8")
        assert (
            "PICF_ADR172_DIRECT_POSTERIOR_HEAD_SCOPE=guidedvla-fixed-object-heads-0-1"
        ) in source
        assert "exec" in source


def test_adr172_formal_evaluation_launchers_fix_the_full_scope() -> None:
    cold_action = COLD_ACTION.read_text(encoding="utf-8")
    retention = RETENTION.read_text(encoding="utf-8")

    for launcher in (cold_action, retention):
        assert "run_lingbot_vla2_ltop_adr172_direct_posterior.py" in launcher
        assert '[[ "$trained_checkpoint" == /mnt/*' in launcher
        assert '[[ "$run_root" == /mnt/*' in launcher
        assert "requires an exact clean PICF checkout" in launcher
        assert "--trained-checkpoint" in launcher
        assert "--nproc_per_node=2" in launcher
        assert "timeout --signal=TERM --kill-after=60s" in launcher
        assert "--mode gate" in launcher
        assert "--steps 128" in launcher
        assert "--eval-every 32" in launcher
        assert '--direct-grounding-weight "$direct_grounding_weight"' in launcher
        assert '--direct-posterior-head-scope "$direct_posterior_head_scope"' in launcher
        assert "model/.metadata" in launcher
        assert 'compgen -G "$trained_checkpoint/model/*.distcp"' in launcher

    assert "--phase evaluation" in cold_action
    assert "validate_adr172_direct_posterior_cold_evidence.py" in cold_action
    assert "adr172_direct_posterior_cold_validation.json" in cold_action
    assert "runner_status" in cold_action
    assert "validator_status" in cold_action

    for wrapper in (GUIDEDVLA_COLD_ACTION, GUIDEDVLA_RETENTION):
        source = wrapper.read_text(encoding="utf-8")
        assert (
            "PICF_ADR172_DIRECT_POSTERIOR_HEAD_SCOPE=guidedvla-fixed-object-heads-0-1"
        ) in source
        assert "exec" in source
    assert "evaluation-action-information-set" not in cold_action
    assert "factual|mediator-required" not in cold_action
    assert "--evaluation-scenes-per-partition 4" in cold_action
    assert "--phase retention" in retention
