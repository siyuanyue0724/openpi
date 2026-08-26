from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.run_lingbot_vla2_task_independent_full import (
    ADR209_CONTROL_ARCHITECTURE_PROFILE,
    ADR209_FLARE_ARCHITECTURE_PROFILE,
    LINGBOT_COMPILE_UPSTREAM_DEFAULT,
    TRAINABLE_SCOPE_FULL_HOST,
    VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
    _disabled_auxiliary_digest,
    _required_future_source_frames,
    _validate_future_latent_cache_args,
    _validate_picf_architecture_profile,
)


def _adr209_profile_args(profile: str) -> argparse.Namespace:
    return argparse.Namespace(
        picf_architecture_profile=profile,
        posterior_architecture="two_pass_v3",
        dense_evidence_mode="calvin_full_v1",
        dense_token_bridge="exact_tokens_v1",
        videomt_stage_pq_mode="trainable-adapted-native-query-causal-c5",
        videomt_idle_placement=VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
        trainable_scope=TRAINABLE_SCOPE_FULL_HOST,
        capacity=200,
        task_query_count=0,
        relation_supervision_layers=(),
        learning_rate=1e-4,
        picf_learning_rate_multiplier=1.0,
        modality_bridge_learning_rate_multiplier=1.0,
        entity_weight=0.0,
        predictive_weight=0.0,
        local_bptt_probability=0.0,
        overshoot_probability=0.0,
        source_mask_probability=0.0,
        lingbot_compile_mode=LINGBOT_COMPILE_UPSTREAM_DEFAULT,
        future_latent_objective_scale=(
            1.0 if profile == ADR209_FLARE_ARCHITECTURE_PROFILE else 0.0
        ),
        minimum_future_source_frames=16,
        future_latent_cache_root=None,
        future_latent_cache_manifest_sha256=None,
        future_latent_cache_build_report=None,
        future_latent_cache_build_report_sha256=None,
    )


def test_adr209_runner_wires_the_complete_target_into_both_objective_paths() -> None:
    source = Path("tools/run_lingbot_vla2_task_independent_full.py").read_text("utf-8")
    training = Path("src/picf_next/lingbot_native/training.py").read_text("utf-8")
    overlay = Path(
        "references/patches/lingbot_vla2_flare_generic_target.patch"
    ).read_text("utf-8")

    assert "FLARE_REQUIRED_FUTURE_SOURCE_FRAMES = 16" in source
    assert "minimum_future_source_frames=_required_future_source_frames(args)" in source
    assert source.count("future_latent_target_for_batch(primary_batch)") == 2
    assert "future_latent_target=future_latent_target_for_batch(batch)" in source
    assert '"future_latent_alignment": future_alignment_report' in source
    assert "install_lingbot_future_latent_alignment(" in source
    assert "future_context.finalized_result(" in training
    assert "future_context.finalize(" not in training
    assert "picf_future_latent_context.finalize(" in overlay
    assert "require_grad=torch.is_grad_enabled()" in overlay


def test_adr209_contract_and_cache_builders_forbid_short_future_substitutes() -> None:
    contract = Path("adr209/prepare_contracts_4gpu.sh").read_text("utf-8")
    cache = Path("adr209/build_flare_cache_4gpu_contract.sh").read_text("utf-8")
    builder = Path("tools/build_flare_future_target_cache.py").read_text("utf-8")

    assert contract.count("--minimum-future-source-frames 16") == 2
    assert "--global-batch-size \"$WORLD_SIZE\"" in contract
    assert "--total-steps 30000" in contract
    assert "--training-prefix-steps \"$TRAINING_PREFIX_STEPS\"" in cache
    assert "FrozenSiglip2FutureEncoder.from_pretrained(" in builder
    assert "config.assert_adr209_complete()" in builder
    assert "count=config.target_offset_source_frames" in builder


def test_adr209_launcher_keeps_every_full_modal_arm_and_restores_upstream_compile() -> None:
    launcher = Path("adr209/run_flare_native_videomt_4gpu.sh").read_text("utf-8")
    publisher = Path("adr209/build_dense_cache_4gpu_contract.sh").read_text("utf-8")

    assert "export PICF_WORLD_SIZE=4" in launcher
    assert "export PICF_TRAINABLE_SCOPE=full-host" in launcher
    assert "export PICF_LINGBOT_COMPILE_MODE=upstream-default" in launcher
    assert "PICF_CUDA_ALLOCATOR:-expandable-segments" in launcher
    assert (
        "PICF_FSDP2_PLACEMENT:-selective-embedding-trainable-vision-offload"
        in launcher
    )
    assert "trainable-adapted-native-query-causal-c5" in launcher
    assert "PICF_FUTURE_LATENT_CACHE_ROOT" in launcher
    assert "PICF_ADR209_ENABLE_FLARE" in launcher
    assert "PICF_ARCHITECTURE_PROFILE=adr209_native_videomt_flare_v1" in launcher
    assert (
        "PICF_ARCHITECTURE_PROFILE=adr209_native_videomt_query_control_t16_v1"
        in launcher
    )
    assert "export PICF_MINIMUM_FUTURE_SOURCE_FRAMES=16" in launcher
    assert "export PICF_FUTURE_LATENT_OBJECTIVE_SCALE=1" in launcher
    assert "export PICF_FUTURE_LATENT_OBJECTIVE_SCALE=0" in launcher
    assert launcher.count("export PICF_FUTURE_LATENT_CACHE_ROOT=") == 1
    assert "PICF_USE_DENSE_SUPPLEMENT=0" in launcher
    assert "case \"$MODALITY\" in anytouch|sonata|vjepa" in publisher
    assert "republish_calvin_frozen_evidence_cache.py" in publisher


def test_adr209_full_host_gate_can_use_native_cpu_offload_without_model_reduction() -> None:
    base = Path("adr178/run_direct_action_posterior_full_modal.sh").read_text("utf-8")
    launcher = Path("adr209/run_flare_native_videomt_4gpu.sh").read_text("utf-8")

    assert (
        "cpu-offload|selective-embedding-offload|"
        "selective-embedding-frozen-vision-offload|"
        "selective-embedding-trainable-vision-offload"
    ) in base
    assert (
        "PICF_FSDP2_PLACEMENT:-selective-embedding-trainable-vision-offload"
        in launcher
    )
    assert "export PICF_TRAINABLE_SCOPE=full-host" in launcher
    assert "export PICF_CAPACITY=200" in launcher


def test_adr209_trainable_vision_offload_is_an_exact_execution_only_overlay() -> None:
    patch = Path(
        "references/patches/lingbot_vla2_selective_trainable_vision_offload.patch"
    ).read_text("utf-8")
    runner = Path("tools/run_lingbot_vla2_task_independent_full.py").read_text("utf-8")

    assert "enable_trainable_vision_offload = kwargs.pop(" in patch
    assert "enable_vision_offload = (" in patch
    assert 'if kind == "vision" and enable_vision_offload:' in patch
    assert "fully_shard(visual_root, **kind_fsdp_kwargs)" in patch
    assert "parameter.requires_grad =" not in patch
    assert "enable_trainable_vision_offload=(" in runner
    assert "verify_selective_trainable_vision_offload(" in runner


def test_adr209_muon_mixed_device_closure_preserves_optimizer_math() -> None:
    patch = Path(
        "references/patches/lingbot_vla2_muon_mixed_device_megabatch.patch"
    ).read_text("utf-8")

    assert "local_parameter = p.to_local() if isinstance(p, DTensor) else p" in patch
    assert "key = (global_shape, str(p.dtype), local_parameter.device.type)" in patch
    assert 'if stacked_local.device.type == "cpu":' in patch
    assert 'device=torch.device("cuda", torch.cuda.current_device())' in patch
    assert "ortho_local.to(device=p_local.device, dtype=p_local.dtype)" in patch
    assert "_MEGABATCH_MAX_GROUP_SIZE" not in patch
    assert "adjusted_lr =" not in patch
    assert "ns_steps =" not in patch
    assert "requires_grad" not in patch


def test_adr209_profiles_freeze_upstream_compile_and_the_t16_domain() -> None:
    for profile in (
        ADR209_CONTROL_ARCHITECTURE_PROFILE,
        ADR209_FLARE_ARCHITECTURE_PROFILE,
    ):
        args = _adr209_profile_args(profile)
        _validate_picf_architecture_profile(args)
        assert _required_future_source_frames(args) == 16
        args.lingbot_compile_mode = "disabled"
        with pytest.raises(ValueError, match="changed frozen fields"):
            _validate_picf_architecture_profile(args)


def test_adr209_candidate_and_lambda_zero_control_require_identical_assets() -> None:
    flare = _adr209_profile_args(ADR209_FLARE_ARCHITECTURE_PROFILE)
    with pytest.raises(ValueError, match="require the complete future-latent cache ABI"):
        _validate_future_latent_cache_args(flare)

    for name, value in {
        "future_latent_cache_root": Path("cache"),
        "future_latent_cache_manifest_sha256": "a" * 64,
        "future_latent_cache_build_report": Path("report.json"),
        "future_latent_cache_build_report_sha256": "b" * 64,
    }.items():
        setattr(flare, name, value)
    _validate_future_latent_cache_args(flare)

    control = _adr209_profile_args(ADR209_CONTROL_ARCHITECTURE_PROFILE)
    with pytest.raises(ValueError, match="require the complete future-latent cache ABI"):
        _validate_future_latent_cache_args(control)
    for name in (
        "future_latent_cache_root",
        "future_latent_cache_manifest_sha256",
        "future_latent_cache_build_report",
        "future_latent_cache_build_report_sha256",
    ):
        setattr(control, name, getattr(flare, name))
    _validate_future_latent_cache_args(control)


def test_absent_flare_cache_has_a_stable_provenance_digest() -> None:
    digest = _disabled_auxiliary_digest("future_latent_cache")

    assert len(digest) == 64
    assert digest == _disabled_auxiliary_digest("future_latent_cache")
