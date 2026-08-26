from __future__ import annotations

import copy

import pytest

from picf_next.contracts import ContractError
from tools.compare_lingbot_native_crossed_bounded_reports import (
    EXPECTED_TRAINABLE_GRADIENT_ELEMENTS,
    INPUT_SCHEMA,
    compare_lingbot_native_crossed_bounded_reports,
)


def _target_row(*, arm: str, step: int, rank: int) -> dict[str, object]:
    cell = "P" if step % 2 == 0 else "X"
    shared = cell == "P"
    return {
        "assistant_text_sha256": f"{step + rank + 1:064x}",
        "camera_name": "static" if step % 4 < 2 else "gripper",
        "crossed_bbox_qwen_xyxy": [step, rank, 500 + step, 500 + rank],
        "crossed_group_index": step if shared else step + (0 if arm == "candidate" else 100),
        "crossed_instruction_sha256": (
            f"{step + rank + 2:064x}"
            if shared
            else f"{step + rank + (3 if arm == 'candidate' else 30):064x}"
        ),
        "crossed_variant_index": rank,
        "elapsed_seconds": 3.0 + rank,
        "factor_name": "target",
        "global_index": step if shared else step + (0 if arm == "candidate" else 100),
        "image_grid_thw": [[1, 16, 16]],
        "instruction": f"prompt-{step}-{rank}" if shared else f"{arm}-prompt-{step}-{rank}",
        "loss": 1.0 + step / 16 + rank / 32,
        "loss_weight": 1.0,
        "rank": rank,
        "record_type": "target",
        "source_rgb_sha256": f"{step + rank + (4 if shared else 40):064x}",
        "supervised_token_count": 24 + rank,
        "target_identity_key": f"object-{step // 2}",
        "task_key": f"task-{step // 2}",
        "user_text_sha256": f"{step + rank + 5:064x}",
        "visual_lattice": 8,
    }


def _public_row(*, step: int, rank: int) -> dict[str, object]:
    return {
        "assistant_text_sha256": f"{step + rank + 10:064x}",
        "elapsed_seconds": 2.0 + rank,
        "family": "referring" if rank == 0 else "vqa",
        "grid_budget": {"merged_visual_tokens": 54},
        "image_grid_thw": [[1, 12, 18]],
        "image_height": 480,
        "image_rgb_sha256": f"{step + rank + 11:064x}",
        "image_width": 640,
        "loss": 2.0 + step / 8 + rank / 16,
        "loss_weight": 0.1,
        "rank": rank,
        "record_id": f"record-{step}-{rank}",
        "record_sha256": f"{step + rank + 12:064x}",
        "source_row_index": step,
        "source_subindex": rank,
        "supervised_token_count": 20 + rank,
        "target_answer_sha256": f"{step + rank + 10:064x}",
        "user_text": f"public-{step}-{rank}",
        "user_text_sha256": f"{step + rank + 13:064x}",
    }


def _step(*, arm: str, index: int) -> dict[str, object]:
    return {
        "crossed_arm": arm,
        "crossed_cell": "P" if index % 2 == 0 else "X",
        "crossed_plan_optimizer_step": index,
        "elapsed_seconds": 8.0,
        "gradient_metrics": {
            "all_finite": True,
            "clip_coefficient": 0.01,
            "frozen_gradient_elements": 0,
            "global_norm_before_clip": 100.0,
            "trainable_gradient_elements": EXPECTED_TRAINABLE_GRADIENT_ELEMENTS,
        },
        "learning_rate": 1e-6,
        "microbatches": [
            {
                "factors": [
                    {
                        "factor_name": "target",
                        "loss_weight": 1.0,
                        "ranks": [_target_row(arm=arm, step=index, rank=rank) for rank in range(2)],
                    }
                ],
                "visual_lattice": 8,
            }
        ],
        "observation_mode": "official_native_once",
        "optimizer_step": index,
        "public_vl_retention": {"ranks": [_public_row(step=index, rank=rank) for rank in range(2)]},
    }


def _report(*, arm: str) -> dict[str, object]:
    return {
        "calvin_factor_contract": {
            "adr127_smoke": False,
            "adr128_smoke": True,
            "crossed_arm": arm,
            "mode": "target_only",
        },
        "candidate_model_file_sha256": {
            "model.safetensors": ("a" if arm == "candidate" else "b") * 64
        },
        "counterfactual_gradient_audit": {"enabled": False},
        "crossed_cpu_materialization": {"unique_record_count": 178},
        "cuda_allocator": "expandable-segments",
        "dataset_manifest_sha256": "c" * 64,
        "fsdp2_placement": "gpu-sharded",
        "hyperparameters": {"max_steps": 2},
        "initial_qwen": {"revision": "d" * 40},
        "native_vl_patch_sha256": "e" * 64,
        "observation_mode": "official_native_once",
        "optimizer": "torch.optim.AdamW",
        "optimizer_state_parameter_count": 404,
        "optimizer_tied_parameter_name": "embed_tokens.weight",
        "physical_sidecar_manifest_sha256": "f" * 64,
        "picf_code_revision": "1" * 40,
        "processor_lattices": {"8": {}},
        "processor_snapshot_size": {"longest_edge": 504},
        "public_vl_retention": {"enabled": True},
        "runtime_python_trees": {"picf": "2" * 64},
        "schema": INPUT_SCHEMA,
        "seed": 20260802,
        "source_commit": "3" * 40,
        "status": "PASS",
        "step_reports": [_step(arm=arm, index=index) for index in range(2)],
        "teacher_prune": {"removed": True},
        "trainable_scope": {"trainable_numel": EXPECTED_TRAINABLE_GRADIENT_ELEMENTS},
        "training_plan": {
            "arm": arm,
            "bounded_training_authorized": True,
            "long_training_authorized": False,
        },
        "world_size": 2,
    }


def test_comparison_accepts_matched_smokes_and_ignores_timing() -> None:
    candidate = _report(arm="candidate")
    control = _report(arm="control")
    control["step_reports"][0]["elapsed_seconds"] = 99.0
    control["step_reports"][0]["microbatches"][0]["factors"][0]["ranks"][0]["elapsed_seconds"] = (
        88.0
    )

    result = compare_lingbot_native_crossed_bounded_reports(candidate, control)

    assert result["status"] == "PASS"
    assert result["step_zero_exact_losses"] is True
    assert result["p_exact_binding_rows"] == 2
    assert result["x_matched_semantic_rows"] == 2
    assert len(result["loss_curves"]) == 2
    assert result["loss_summary"]["target"]["all"]["candidate_minus_control_mean"] == 0.0


def test_comparison_rejects_step_zero_loss_or_gradient_drift() -> None:
    candidate = _report(arm="candidate")
    control = _report(arm="control")
    control["step_reports"][0]["microbatches"][0]["factors"][0]["ranks"][0]["loss"] += 0.125
    with pytest.raises(ContractError, match="step-zero parity"):
        compare_lingbot_native_crossed_bounded_reports(candidate, control)

    control = _report(arm="control")
    control["step_reports"][0]["gradient_metrics"]["global_norm_before_clip"] += 1e-5
    with pytest.raises(ContractError, match="step-zero parity"):
        compare_lingbot_native_crossed_bounded_reports(candidate, control)


def test_comparison_rejects_changed_p_binding_or_missing_gradient() -> None:
    candidate = _report(arm="candidate")
    control = _report(arm="control")
    changed = copy.deepcopy(control)
    changed["step_reports"][0]["microbatches"][0]["factors"][0]["ranks"][0]["source_rgb_sha256"] = (
        "9" * 64
    )
    with pytest.raises(ContractError, match="P rank binding changed"):
        compare_lingbot_native_crossed_bounded_reports(candidate, changed)

    changed = copy.deepcopy(control)
    changed["step_reports"][1]["gradient_metrics"]["trainable_gradient_elements"] -= 1
    with pytest.raises(ContractError, match="gradient coverage failed"):
        compare_lingbot_native_crossed_bounded_reports(candidate, changed)
