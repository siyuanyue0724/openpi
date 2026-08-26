from __future__ import annotations

import copy

import pytest

from picf_next.contracts import ContractError
from tools.compare_native_vl_grounding_replay_reports import (
    BASELINE_SCHEMA,
    REPLAY_SCHEMA,
    compare_native_vl_grounding_replay_reports,
)


def _step(index: int) -> dict[str, object]:
    ranks = [
        {
            "camera_name": "static",
            "elapsed_seconds": 2.0 + rank,
            "global_index": 800_000 + index,
            "image_grid_thw": [[1, 16, 16]],
            "instruction": f"instruction-{index}-{rank}",
            "loss": 2.0 + index / 8 + rank / 16,
            "loss_weight": 1.0,
            "rank": rank,
            "supervised_token_count": 26,
            "target_identity_key": f"object-{rank}",
            "task_key": f"task-{rank}",
            "visual_lattice": 8,
        }
        for rank in range(2)
    ]
    return {
        "curriculum_group_index": 100 + index,
        "curriculum_optimizer_step": index,
        "elapsed_seconds": 5.0,
        "gradient_metrics": {
            "all_finite": True,
            "clip_coefficient": 0.01 + index / 1000,
            "frozen_gradient_elements": 0,
            "global_norm_before_clip": 100.0 + index,
            "trainable_gradient_elements": 1000,
        },
        "learning_rate": 1e-6,
        "microbatches": [{"ranks": ranks, "visual_lattice": 8}],
        "observation_mode": "official_native_once",
        "optimizer_step": index,
    }


def _report(*, replay: bool) -> dict[str, object]:
    steps = [_step(index) for index in range(2 if replay else 64)]
    if replay:
        for step in steps:
            step["public_vl_retention"] = None
    return {
        "cuda_allocator": "expandable-segments",
        "dataset_manifest_sha256": "a" * 64,
        "fsdp2_placement": "gpu-sharded",
        "hyperparameters": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "learning_rate": 1e-6,
            "max_grad_norm": 1.0,
            "max_steps": 2 if replay else 64,
            "schedule": "linear-warmup-cosine-decay",
            "schedule_total_steps": 432,
            "warmup_steps": 0,
            "weight_decay": 0.0,
        },
        "initial_qwen": {"model_file_sha256": {"model.safetensors": "b" * 64}},
        "native_vl_patch_sha256": "c" * 64,
        "observation_mode": "official_native_once",
        "optimizer": "torch.optim.AdamW",
        "optimizer_state_parameter_count": 10,
        "optimizer_tied_parameter_name": "embed_tokens.weight",
        "picf_code_revision": ("e" if replay else "d") * 40,
        "processor_lattices": {"8": {"height": 256, "width": 256}},
        "processor_snapshot_size": {"height": 504, "width": 504},
        "public_vl_retention": {"enabled": False} if replay else None,
        "schema": REPLAY_SCHEMA if replay else BASELINE_SCHEMA,
        "source_commit": "f" * 40,
        "status": "PASS",
        "step_reports": steps,
        "teacher_prune": {"removed": []},
        "trainable_scope": {"trainable_numel": 1000},
        "training_plan": {"artifact_sha256": "1" * 64},
        "world_size": 2,
    }


def test_replay_comparison_accepts_exact_losses_and_tiny_gradient_roundoff() -> None:
    baseline = _report(replay=False)
    replay = _report(replay=True)
    replay["step_reports"][0]["gradient_metrics"]["global_norm_before_clip"] += 5e-5

    result = compare_native_vl_grounding_replay_reports(baseline, replay)

    assert result["status"] == "PASS"
    assert result["exact_bf16_losses"] is True
    assert result["gradient_metrics_within_tolerance"] is True
    assert result["compared_rank_losses"] == 4


def test_replay_comparison_rejects_one_changed_bf16_loss() -> None:
    baseline = _report(replay=False)
    replay = _report(replay=True)
    replay["step_reports"][1]["microbatches"][0]["ranks"][1]["loss"] += 1 / 64

    result = compare_native_vl_grounding_replay_reports(baseline, replay)

    assert result["status"] == "FAIL"
    assert result["exact_bf16_losses"] is False


def test_replay_comparison_rejects_retention_or_changed_record_identity() -> None:
    baseline = _report(replay=False)
    replay = _report(replay=True)
    replay["public_vl_retention"] = {"enabled": True}
    with pytest.raises(ContractError, match="enabled public retention"):
        compare_native_vl_grounding_replay_reports(baseline, replay)

    replay = _report(replay=True)
    changed = copy.deepcopy(replay)
    changed["step_reports"][0]["microbatches"][0]["ranks"][0]["task_key"] = "changed"
    with pytest.raises(ContractError, match="rank binding changed"):
        compare_native_vl_grounding_replay_reports(baseline, changed)
