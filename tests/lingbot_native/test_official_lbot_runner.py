from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from picf_next.lingbot_native.trainable_scope import (
    TRAINABLE_SCOPE_FROZEN_VISION_HOST,
    TRAINABLE_SCOPE_FULL_HOST,
    lingbot_trainable_scope_receipt,
)
from tools.run_lingbot_vla2_official_lbot import (
    _ADR172_ALLOWED_DCP_EXTRA_PREFIXES,
    _ADR172_EXACT_EVALUATION_STEPS,
    _ADR172_EXACT_SEED,
    _ADR172_EXACT_STEPS,
    _MAXIMUM_LBOT_CURVE_STEPS,
    _MAXIMUM_LBOT_STEPS,
    _adr172_exact_input_hashes,
    _adr172_exact_input_receipt,
    _audit_adr172_shared_checkpoint_metadata,
    _distributed_phase_error,
    _evaluation_replay_seed,
    _evaluation_steps,
    _implementation_provenance,
    _picf_graph_installed,
    _require_adr172_exact_input_receipt,
    _runtime_lbot_world_size,
    _summarize_action_partition,
    LINGBOT_ATTENTION_IMPLEMENTATIONS,
)


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "tools/run_lingbot_vla2_official_lbot.py").read_text()


def test_lbot_runner_is_bounded_and_publishes_no_checkpoint() -> None:
    source = _source()

    assert _MAXIMUM_LBOT_STEPS == 20
    assert _MAXIMUM_LBOT_CURVE_STEPS == 2_000
    assert "checkpointer.save" not in source
    assert '"checkpoint_published": False' in source
    assert "allow_partial_load=True" in source
    assert '"load_scope": "shared-lingbot-tensors-only"' in source


def test_lbot_runner_accepts_only_registered_two_or_four_rank_topologies() -> None:
    assert _runtime_lbot_world_size({}) == 2
    assert _runtime_lbot_world_size({"WORLD_SIZE": "2"}) == 2
    assert _runtime_lbot_world_size({"WORLD_SIZE": "4"}) == 4
    for value in ("0", "3", "04", "eight"):
        with pytest.raises(RuntimeError):
            _runtime_lbot_world_size({"WORLD_SIZE": value})

    assert "combinations(sample_sets, 2)" in _source()


def test_lbot_runner_has_a_strict_official_policy_boundary() -> None:
    source = _source()

    assert "run_official_policy_training_forward" in source
    assert "run_official_policy_diagnostic_forward" in source
    assert "build_lingbot_official_optimizer" in source
    assert "CalvinPhysicalSupervisionSidecar" not in source
    assert "run_task_independent_calvin" not in source
    assert "install_lingbot_native_graph" not in source
    assert "LingBotNativeGraph" not in source
    assert '"picf_graph_installed": False' in source
    assert '"physical_sidecar_read": False' in source
    assert '"posterior_present": False' in source
    assert '"official_output_arity": 11' in source


def test_lbot_runner_exposes_the_released_joint_attention_backends() -> None:
    source = _source()

    assert LINGBOT_ATTENTION_IMPLEMENTATIONS == ("eager", "flex_cached")
    assert '"--attention-implementation"' in source
    assert 'merged["attention_implementation"] = args.attention_implementation' in source
    assert '"attention_implementation": args.attention_implementation' in source


def test_lbot_runner_can_restore_the_released_whole_model_compile_order() -> None:
    source = _source()

    fsdp = source.index("policy = build_parallelize_model(")
    compile_model = source.index("policy = torch.compile(policy)")
    optimizer = source.index("optimizer = build_lingbot_official_optimizer(")
    assert fsdp < compile_model < optimizer
    assert 'merged["use_compile"] = args.lingbot_compile_mode == "upstream-default"' in source
    assert '"ordering": "fsdp2_then_whole_model_compile_then_optimizer"' in source


def test_lbot_matched_curve_replays_the_approved_muon_hotfix() -> None:
    source = _source()

    for value in (
        "verify_muon_collective_hotfix(",
        "validate_prepared_native_source_with_muon_collective_hotfix(",
        'elif args.runtime_hotfix is not None:',
        'report["runtime_hotfix_sha256"]',
    ):
        assert value in source
    assert "immutable native-plus-Muon replay" in source


def _toy_lingbot_scope_policy() -> torch.nn.Module:
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.qwenvl_with_expert = torch.nn.Module()
    policy.model.qwenvl_with_expert.qwenvl = torch.nn.Module()
    policy.model.qwenvl_with_expert.qwenvl.model = torch.nn.Module()
    policy.model.qwenvl_with_expert.qwenvl.model.visual = torch.nn.Linear(3, 4)
    policy.model.qwenvl_with_expert.qwen_expert = torch.nn.Linear(4, 5)
    return policy


def test_lbot_trainable_scope_can_match_the_frozen_vision_candidate() -> None:
    policy = _toy_lingbot_scope_policy()
    full = lingbot_trainable_scope_receipt(policy, scope=TRAINABLE_SCOPE_FULL_HOST)
    assert full["trainable_visual_numel"] == full["visual_numel"] == 16

    for parameter in policy.model.qwenvl_with_expert.qwenvl.model.visual.parameters():
        parameter.requires_grad_(False)
    frozen = lingbot_trainable_scope_receipt(
        policy,
        scope=TRAINABLE_SCOPE_FROZEN_VISION_HOST,
    )
    assert frozen["forward_model_complete"] is True
    assert frozen["visual_forward_enabled"] is True
    assert frozen["trainable_visual_numel"] == 0
    assert frozen["trainable_numel"] < frozen["total_numel"]

    source = _source()
    assert '"--trainable-scope"' in source
    assert 'report["trainable_scope"] = trainable_scope_receipt' in source
    assert "ADR176_FROZEN_VISION_LBOT_REPORT_SCHEMA" in source


def test_adr176_frozen_vision_launcher_is_a_matched_1500_step_control() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "adr176/run_lbot_2gpu_prefix1500_frozen_vision.sh").read_text()

    for value in (
        "--physical-event-stream",
        "--maximum-control-tokens 64",
        "--evaluation-steps 0,20,100,200,500,1000,1500",
        "--steps 1500",
        "--seed 20260721",
        "--learning-rate 1e-4",
        "--max-grad-norm 1.0",
        "--trainable-scope frozen-vision-host",
        "--fsdp2-placement selective-embedding-offload",
        "--cuda-allocator expandable-segments",
    ):
        assert value in launcher
    assert "dense-evidence" not in launcher
    assert "task_independent_full" not in launcher


def test_lbot_can_share_adr149_physical_events_without_installing_picf() -> None:
    source = _source()

    assert '"--physical-event-stream"' in source
    assert "CalvinPhysicalTransitionDataset" in source
    assert "build_native_calvin_physical_episode_domain" in source
    assert "build_native_calvin_physical_stream_plan" in source
    assert 'PHYSICAL_LBOT_COMPARISON_ID = "lingbot-vla2-native-picf-full"' in source
    assert "args.maximum_control_tokens if args.physical_event_stream else None" in source
    assert '"physical_event_stream": args.physical_event_stream' in source
    assert "install_lingbot_native_graph" not in source


def test_lbot_can_replay_a_future_filtered_physical_domain() -> None:
    source = _source()

    assert '"--minimum-future-source-frames"' in source
    assert source.count(
        "minimum_future_source_frames=args.minimum_future_source_frames"
    ) == 2
    assert (
        '"minimum_future_source_frames": args.minimum_future_source_frames'
        in source
    )
    assert (
        'if args.minimum_future_source_frames and not args.physical_event_stream:'
        in source
    )


def test_lbot_curve_reconstructs_the_schema_bound_stream_domain() -> None:
    source = _source()
    plan_block = source[
        source.index("plan = load_frozen_episode_stream_plan(") : source.index(
            "evaluation_plan = EntityEvaluationPlan.load"
        )
    ]

    assert (
        plan_block.count("representation_split.stream_domain_excluded_source_episode_indices") == 2
    )
    assert "representation_split.evaluation_source_episode_indices" not in plan_block


def test_lbot_graph_audit_reads_the_actual_patched_host_mount() -> None:
    class Host:
        picf_native_graph = None

    class Model:
        qwenvl_with_expert = Host()

    class Policy:
        model = Model()

    policy = Policy()
    assert not _picf_graph_installed(policy)
    policy.model.qwenvl_with_expert.picf_native_graph = object()
    assert _picf_graph_installed(policy)
    with pytest.raises(RuntimeError, match="graph mount contract"):
        _picf_graph_installed(object())


def test_lbot_fixed_evaluation_restores_released_moe_runtime_state() -> None:
    source = _source()

    assert "snapshot_official_runtime_buffers" in source
    assert "restore_official_runtime_buffers" in source
    assert '"avg_topk_sigmoid_score"' in source
    assert '"tokens_per_expert"' in source
    assert "finally:" in source
    assert "restore_official_runtime_buffers(runtime_snapshot)" in source


def test_lbot_curve_steps_and_replay_seed_are_deterministic() -> None:
    assert _evaluation_steps(None) == ()
    assert _evaluation_steps("0,20,100,200") == (0, 20, 100, 200)
    with pytest.raises(ValueError, match="unique sorted"):
        _evaluation_steps("20,0,20")

    plan = "a" * 64
    assert _evaluation_replay_seed(plan, "sample-a") == _evaluation_replay_seed(plan, "sample-a")
    assert _evaluation_replay_seed(plan, "sample-a") != _evaluation_replay_seed(plan, "sample-b")


def test_lbot_action_partition_summary_is_loss_only() -> None:
    summary = _summarize_action_partition(
        [
            {
                "partition": "heldout",
                "action_loss": 0.4,
                "total_loss": 0.5,
                "moe_regularizer": 0.1,
                "forward_seconds": 2.0,
            },
            {
                "partition": "heldout",
                "action_loss": 0.2,
                "total_loss": 0.3,
                "moe_regularizer": 0.1,
                "forward_seconds": 4.0,
            },
        ],
        partition="heldout",
    )

    assert summary == {
        "sample_count": 2,
        "mean_action_loss": pytest.approx(0.3),
        "mean_total_loss": pytest.approx(0.4),
        "mean_moe_regularizer": pytest.approx(0.1),
        "mean_forward_seconds": pytest.approx(3.0),
    }


def test_lbot_evaluation_exchanges_rank_local_failures() -> None:
    class Dist:
        @staticmethod
        def all_gather_object(gathered: list[object], local: object) -> None:
            gathered[:] = [local, None]

    _distributed_phase_error(error=None, phase="prepare", rank=0, dist_module=Dist)
    with pytest.raises(RuntimeError, match="distributed LBOT action evaluation failed"):
        _distributed_phase_error(
            error=ValueError("bad fixed sample"),
            phase="forward",
            rank=0,
            dist_module=Dist,
        )


def test_lbot_report_binds_only_data_and_official_forward_implementation() -> None:
    root = Path(__file__).resolve().parents[2]
    files, digest = _implementation_provenance(root)

    assert len(digest) == 64
    assert "tools/run_lingbot_vla2_official_lbot.py" in files
    assert "src/picf_next/lingbot_native/training.py" in files
    assert "src/picf_next/lingbot_native/calvin.py" in files
    assert "src/picf_next/lingbot_native/entity_set_objective.py" not in files
    assert all(len(value) == 64 for value in files.values())


class _TensorLike:
    def __init__(self, *, dtype: str = "torch.float32", shape: tuple[int, ...] = (2, 3)):
        self.dtype = dtype
        self.shape = shape

    def size(self) -> tuple[int, ...]:
        return self.shape


class _DCPMetadataLike:
    def __init__(self, *, dtype: str = "torch.float32", shape: tuple[int, ...] = (2, 3)):
        self.properties = SimpleNamespace(dtype=dtype)
        self.size = shape


def test_adr172_shared_checkpoint_audit_accepts_only_picf_extras() -> None:
    extra_name = f"{_ADR172_ALLOWED_DCP_EXTRA_PREFIXES[0]}role_embeddings"
    report = _audit_adr172_shared_checkpoint_metadata(
        checkpoint_metadata={
            "state.model.model.weight": _DCPMetadataLike(),
            extra_name: _DCPMetadataLike(shape=(8, 16)),
        },
        shared_state={"model.weight": _TensorLike()},
    )

    assert report["status"] == "PASS"
    assert report["shared_tensor_count"] == 1
    assert report["missing_shared_tensors"] == []
    assert report["extra_tensor_count"] == 1
    assert report["rejected_extra_tensors"] == []


def test_adr172_shared_checkpoint_audit_fails_closed() -> None:
    extra_name = f"{_ADR172_ALLOWED_DCP_EXTRA_PREFIXES[0]}role_embeddings"
    with pytest.raises(RuntimeError, match="omits shared LingBot tensors"):
        _audit_adr172_shared_checkpoint_metadata(
            checkpoint_metadata={extra_name: _DCPMetadataLike()},
            shared_state={"model.weight": _TensorLike()},
        )
    with pytest.raises(RuntimeError, match="non-PICF extra tensors"):
        _audit_adr172_shared_checkpoint_metadata(
            checkpoint_metadata={
                "state.model.model.weight": _DCPMetadataLike(),
                "state.model.unregistered.weight": _DCPMetadataLike(),
            },
            shared_state={"model.weight": _TensorLike()},
        )
    with pytest.raises(RuntimeError, match="schemas differ"):
        _audit_adr172_shared_checkpoint_metadata(
            checkpoint_metadata={
                "state.model.model.weight": _DCPMetadataLike(shape=(3, 2)),
                extra_name: _DCPMetadataLike(),
            },
            shared_state={"model.weight": _TensorLike()},
        )


def test_adr172_exact_input_receipt_hashes_every_comparison_axis() -> None:
    model_inputs = {
        "actions": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "action_is_pad": torch.tensor([[False, True]]),
        "noise": torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16),
        "time": torch.tensor([0.75], dtype=torch.bfloat16),
    }
    hashes = _adr172_exact_input_hashes(model_inputs)
    receipt = _adr172_exact_input_receipt(
        sample_key="sample-a",
        replay_seed=17,
        source_digest="a" * 64,
        model_inputs=model_inputs,
        model_inputs_sha256="b" * 64,
    )

    assert _ADR172_EXACT_STEPS == 256
    assert _ADR172_EXACT_SEED == 20260813
    assert _ADR172_EXACT_EVALUATION_STEPS == (0, 32, 64, 96, 128, 160, 192, 224, 256)
    assert receipt["action_targets_sha256"] == hashes["action_targets_sha256"]
    assert len(receipt["sample_action_noise_timestep_sha256"]) == 64

    changed = dict(model_inputs)
    changed["noise"] = torch.tensor([[0.25, -0.25]], dtype=torch.bfloat16)
    assert _adr172_exact_input_hashes(changed)["noise_sha256"] != hashes["noise_sha256"]


def test_adr172_exact_input_receipt_comparison_fails_closed() -> None:
    expected = {"sample_action_noise_timestep_sha256": "a" * 64}
    assert (
        _require_adr172_exact_input_receipt(
            expected=expected,
            actual=expected,
            phase="evaluation",
        )
        == expected
    )

    with pytest.raises(RuntimeError, match="training input differs"):
        _require_adr172_exact_input_receipt(
            expected=expected,
            actual={"sample_action_noise_timestep_sha256": "b" * 64},
            phase="training",
        )
