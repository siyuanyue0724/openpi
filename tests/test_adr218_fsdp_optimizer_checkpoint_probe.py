from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools/probe_adr218_fsdp_optimizer_checkpoint.py"


def test_adr218_fsdp_probe_reuses_production_primitives_and_real_data() -> None:
    source = PROBE.read_text(encoding="utf-8")
    required = (
        "build_parallelize_model(",
        "enable_full_shard=True",
        "enable_gradient_checkpointing=True",
        "enable_fsdp_offload=False",
        "enable_shared_embedding_offload=True",
        "selective_cpu_module_classes=(",
        'FSDP_OFFLOAD_MODE = "shared-embedding-plus-future3d-classes"',
        "selective_cpu_modules != SELECTIVE_CPU_MODULES",
        "all_teacher_layers = tuple(layer.cpu() for layer in teacher(teacher_images))",
        '"da3_teacher_cache_residency": "cpu-between-updates"',
        "register_wsa_lingbot_fsdp_forward_methods(policy)",
        "configure_wsa_lingbot_optimizer_contract(optimizer_contract)",
        "build_lingbot_official_optimizer(",
        "install_wsa_lingbot_optimizer(policy, optimizer)",
        "audit_wsa_lingbot_optimizer(policy, optimizer)",
        "CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)",
        "collate_native_calvin_training_batch(",
        "DA3BackboneTeacher(",
        "checkpointer.save(",
        "checkpointer.load(",
        '"cold_resume_bit_exact": True',
        'CUBLAS_WORKSPACE_CONFIG = ":4096:8"',
        'PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"',
        'os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG',
        'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF',
        "torch_module.use_deterministic_algorithms(True)",
        "torch_module.backends.cudnn.deterministic = True",
        'torch_module.set_float32_matmul_precision("highest")',
        '"determinism": determinism_receipt',
        "gc.collect()",
        "torch_module.cuda.empty_cache()",
        'phase="checkpoint-allocator-cache-released"',
    )
    for fragment in required:
        assert fragment in source


def test_adr218_fsdp_probe_does_not_authorize_a_simplified_or_long_run() -> None:
    source = PROBE.read_text(encoding="utf-8")
    forbidden = (
        "robotwin.json",
        "robot_obs[:14]",
        "torch.optim.AdamW(",
        "max_steps=30000",
        "cached_inference_authorized=True",
    )
    for fragment in forbidden:
        assert fragment not in source
    assert source.count('"scheduler_status": "open-before-long-training"') == 2


def test_adr218_causal_gate_uses_the_full_fixed_weight_graph() -> None:
    source = PROBE.read_text(encoding="utf-8")
    required = (
        'choices=("fresh", "resume", "composition", "causality")',
        "run_native_policy_diagnostic_forward(",
        "wsa_measurement_callback=capture_joint",
        "dist.all_gather(gathered_rows, previous_state.layer_rows.contiguous())",
        "current_modalities.omit((stream.name,))",
        'candidate_inputs["actions"] = perturbed_actions',
        "WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION",
        '"future_reaches_action": future_reaches_action',
        '"posterior_reaches_action": posterior_reaches_action',
        '"all_modalities_reach_action": all_modalities_reach_action',
        '"scientific_advantage_claimed": False',
    )
    for fragment in required:
        assert fragment in source
