from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import math
import os
import types
from pathlib import Path
import sys
import tempfile

import pytest
import numpy as np
import torch

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import PicfTactilePacket
from openpi.picf.contracts import PicfPointCloudFrame
from openpi.picf.contracts import TactileSensorFrame
from openpi.picf.core.pipeline import LazyCrossAttentionRead
from openpi.picf.fsdp_utils import call_fsdp_method
from openpi.picf.fsdp_utils import call_module_forward_or_method
from openpi.picf.test_utils import build_mini_calvin_dataset


_SCRIPT_PATH = Path(__file__).with_name("picf_core_train.py")
_GEMMA_PYTORCH_PATH = Path(__file__).resolve().parents[1] / "src" / "openpi" / "models_pytorch" / "gemma_pytorch.py"
_SPEC = importlib.util.spec_from_file_location("picf_core_train_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        num_train_steps=30000,
        log_interval=100,
        save_interval=5000,
        diagnostic_interval=500,
        diagnostic_visual_upscale=64,
        accum_steps=1,
        max_empty_window_retries=32,
        unroll_steps=2,
        burnin_steps=0,
        burnin_mode="full",
        action_horizon=16,
        stride=4,
        max_points=1024,
        crop_radius_m=0.10,
        point_focus_sigma_m=0.03,
        visual_grid=8,
        visual_real_grid=64,
        visual_num_frames=64,
        visual_img_size=384,
        visual_patch_size=16,
        visual_tubelet_size=2,
        tactile_num_frames=4,
        tactile_stride=2,
        hidden_dim=512,
        posterior_hidden_dim=512,
        latent_dim=112,
        innovation_dim=512,
        control_dim=512,
        semantic_dim=2048,
        semantic_cross_dim=2048,
        future_hidden_dim=512,
        persistent_anchors=8,
        observation_anchors=16,
        effector_persistent_anchors=2,
        effector_observation_anchors=2,
        fusion_layers=4,
        posterior_layers=2,
        predictive_layers=2,
        control_layers=2,
        control_query_tokens=1,
        predictive_query_tokens=1,
        task_local_queries=8,
        task_effector_queries=2,
        task_global_queries=1,
        task_instruction_queries=2,
        task_self_layers=1,
        conditioned_control_queries=4,
        pi_prefix_queries=4,
        conditioned_future_queries=2,
        predictive_semantic_reads=2,
        control_semantic_reads=2,
        predictive_semantic_dropout_prob=0.1,
        semantic_prefix_dropout_prob=0.0,
        task_visual_reread_topk=32,
        task_tactile_reread_groups=2,
        task_point_reread_topk=32,
        vl_anchor_router_enabled=False,
        vl_grounding_view="static",
        vl_anchor_modes=6,
        vl_anchor_nms_radius_m=0.04,
        vl_anchor_local_sigma_m=0.04,
        vl_min_visible_mass=1e-4,
        vl_heatmap_temperature=1.0,
        vl_obs_anchor_gate_init=-4.0,
        vl_task_point_gate_init=-4.0,
        vl_posterior_bind_gate_init=-6.0,
        vl_prior_bias_clip=4.0,
        global_scene_point_cap=1024,
        require_pi0_action_generator=True,
        attention_heads=8,
        future_vote_heads=4,
        warmup_steps=None,
        lr=2e-4,
        min_lr=2e-5,
        weight_decay=1e-4,
        optimizer_sharding="none",
        optimizer_checkpoint_mode="auto",
        grad_clip_norm=1.0,
        grad_clip_mode="percentile",
        grad_clip_percentile=75.0,
        grad_clip_window=100,
        action_normalization="quantile",
        action_norm_stats_path=None,
        action_output_clip=None,
        prompt_state_normalization="inherit",
        prompt_state_norm_stats_path=None,
        lambda_action_pos=2.0,
        lambda_action_rot=2.0,
        lambda_action_gripper=2.0,
        lambda_visual_latent=0.2,
        lambda_visual_real=0.1,
        lambda_tactile_real=0.3,
        lambda_point_real=0.3,
        lambda_semantic_future_aux=0.25,
        lambda_anchor_pv=0.1,
        lambda_pv_weak=0.02,
        lambda_focus_pv=0.0,
        lambda_pt=1.0,
        tau_pv=0.07,
        tau_pt=0.07,
        tau_route_p=0.1,
        tau_route_v=0.1,
        pt_bag_radius_m=0.045,
        pt_bag_sigma_m=0.015,
        pt_bag_kmin=32,
        pt_back_slack_m=0.008,
        p_align_on=0.55,
        p_align_off=0.35,
        device="cuda",
        point_backbone="sonata",
        point_backbone_trainable=False,
        point_backbone_lr_scale=0.25,
        visual_trainable=False,
        visual_lr_scale=0.25,
        visual_activation_checkpointing=True,
        tactile_trainable=False,
        tactile_lr_scale=0.25,
        semantic_mode="zero",
        semantic_source="auto",
        semantic_model_name="google/paligemma2-3b-pt-224",
        semantic_checkpoint_path=None,
        semantic_checkpoint_config_path=None,
        semantic_revision=None,
        semantic_paligemma_variant="gemma_2b",
        semantic_action_expert_variant="gemma_300m",
        semantic_dtype="bfloat16",
        semantic_trainable=False,
        semantic_lr_scale=0.25,
        semantic_gradient_checkpointing=True,
        window_activation_checkpointing=False,
        semantic_use_gripper=True,
        semantic_max_length=256,
        semantic_tokenwise_chunk_size=0,
        semantic_projection_chunk_size=None,
        semantic_mlp_chunk_size=None,
        visual_mode="stub",
        visual_finetune_mode="auto",
        tactile_mode="stub",
        use_tactile=False,
        visual_model_name="vjepa2_1_vit_base_384",
        visual_checkpoint_path=None,
        visual_checkpoint_key=None,
        visual_dtype="bfloat16",
        visual_feature_mode="auto",
        tactile_checkpoint_path=None,
        tactile_dtype="float32",
        tactile_sensor_names="digit,gelsight_mini",
        tactile_sensor_offsets_m="0.01,0,0;-0.01,0,0",
        tactile_calibration_path=None,
        tactile_backgrounds_path=None,
        tactile_contact_stats_path=None,
        tactile_contact_tau_on=None,
        tactile_contact_tau_off=None,
        tactile_contact_temperature=None,
        tactile_contact_ema_beta=0.8,
        tactile_anchor_prob_on=0.8,
        use_scene_obs=False,
        use_foundation_backbones=False,
        perception_finetune_mode="auto",
        picf_augmentation_mode="off",
        picf_photometric_strength="conservative",
    )


def test_retryable_first_step_error_detection() -> None:
    exc = RuntimeError("PICF core requires non-empty local xyzrgb support on the first control step.")
    assert _MODULE._is_retryable_first_step_error(exc) is True
    later = RuntimeError("PICF core requires non-empty local xyzrgb support on window step 1.")
    assert _MODULE._is_retryable_first_step_error(later) is True
    other = RuntimeError("some other training failure")
    assert _MODULE._is_retryable_first_step_error(other) is False


def test_normalize_train_args_sets_default_warmup_fraction() -> None:
    args = _base_args()
    _MODULE._normalize_train_args(args)
    assert args.warmup_steps == 600


def test_normalize_train_args_caps_visual_real_diagnostic_upscale_for_64_grid() -> None:
    args = _base_args()
    args.visual_real_grid = 64
    args.diagnostic_visual_upscale = 64
    _MODULE._normalize_train_args(args)

    assert args.visual_real_grid == 64
    assert args.diagnostic_visual_upscale == 4


def test_normalize_train_args_tracks_burnin_effective_window_length() -> None:
    args = _base_args()
    args.unroll_steps = 1
    args.burnin_steps = 8
    args.burnin_mode = "state-only"
    _MODULE._normalize_train_args(args)

    assert args.unroll_steps == 1
    assert args.burnin_steps == 8
    assert args.burnin_mode == "state_only"
    assert args.effective_unroll_steps == 9


def test_validate_train_args_rejects_burnin_for_ablated_mode() -> None:
    args = _base_args()
    args.picf_mode = "ablated"
    args.burnin_steps = 8
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="burnin_steps > 0 requires picf_mode=enabled"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_unknown_burnin_mode() -> None:
    args = _base_args()
    args.burnin_mode = "surprise"
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="burnin_mode must be one of"):
        _MODULE._validate_train_args(args)


def test_normalize_train_args_inherits_prompt_state_normalization_from_action_contract() -> None:
    args = _base_args()
    _MODULE._normalize_train_args(args)

    assert args.action_normalization == "quantile"
    assert args.prompt_state_normalization == "quantile"
    assert args.prompt_state_norm_stats_path == args.action_norm_stats_path


def test_parse_tactile_sensor_names_accepts_stringified_tuple() -> None:
    assert _MODULE._parse_tactile_sensor_names("('digit', 'gelsight_mini')") == ("digit", "gelsight_mini")


def test_parse_tactile_sensor_offsets_accepts_stringified_tuple() -> None:
    assert _MODULE._parse_tactile_sensor_offsets("((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0))") == (
        (0.01, 0.0, 0.0),
        (-0.01, 0.0, 0.0),
    )


def test_normalize_train_args_splits_semantic_chunk_knobs_from_legacy_default() -> None:
    args = _base_args()
    args.training_strategy = "fsdp_full_shard"
    args.semantic_mode = "paligemma"
    args.semantic_trainable = True
    args.semantic_tokenwise_chunk_size = 64
    args.semantic_projection_chunk_size = None
    args.semantic_mlp_chunk_size = None

    _MODULE._normalize_train_args(args)

    assert args.semantic_tokenwise_chunk_size == 64
    assert args.semantic_projection_chunk_size == 64
    assert args.semantic_mlp_chunk_size == 64


def test_normalize_train_args_uses_balanced_default_semantic_chunk_split_for_fsdp_profile() -> None:
    args = _base_args()
    args.training_strategy = "fsdp_full_shard"
    args.semantic_mode = "paligemma"
    args.semantic_trainable = True
    args.semantic_tokenwise_chunk_size = 0
    args.semantic_projection_chunk_size = None
    args.semantic_mlp_chunk_size = None

    _MODULE._normalize_train_args(args)

    assert args.semantic_tokenwise_chunk_size == 64
    assert args.semantic_projection_chunk_size == 128
    assert args.semantic_mlp_chunk_size == 64


def test_normalize_train_args_preserves_explicit_semantic_chunk_split() -> None:
    args = _base_args()
    args.training_strategy = "fsdp_full_shard"
    args.semantic_mode = "paligemma"
    args.semantic_trainable = True
    args.semantic_tokenwise_chunk_size = 64
    args.semantic_projection_chunk_size = 128
    args.semantic_mlp_chunk_size = 48

    _MODULE._normalize_train_args(args)

    assert args.semantic_tokenwise_chunk_size == 64
    assert args.semantic_projection_chunk_size == 128
    assert args.semantic_mlp_chunk_size == 48


def test_normalize_train_args_ablated_disables_picf_backbone_paths() -> None:
    args = _base_args()
    args.picf_mode = "ablated"
    args.use_foundation_backbones = True
    args.point_backbone = "sonata"
    args.visual_mode = "encoder"
    args.tactile_mode = "encoder"
    args.use_tactile = True

    _MODULE._normalize_train_args(args)

    assert args.point_backbone == "rgb"
    assert args.point_backbone_trainable is False
    assert args.visual_mode == "stub"
    assert args.visual_trainable is False
    assert args.tactile_mode == "stub"
    assert args.tactile_trainable is False
    assert args.use_tactile is False
    assert args.use_scene_obs is False
    assert args.require_pi0_action_generator is True


def test_validate_train_args_ablated_requires_paligemma() -> None:
    args = _base_args()
    args.picf_mode = "ablated"
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="semantic_mode=paligemma"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_requires_prompt_state_norm_stats_when_enabled() -> None:
    args = _base_args()
    args.prompt_state_normalization = "quantile"
    args.prompt_state_norm_stats_path = "/tmp/does-not-exist.json"
    _MODULE._normalize_train_args(args)

    with pytest.raises(FileNotFoundError, match="Prompt-state normalization requires a valid norm_stats.json"):
        _MODULE._validate_train_args(args)


def test_foundation_profile_does_not_force_window_activation_checkpointing() -> None:
    args = _base_args()
    args.use_foundation_backbones = True
    _MODULE._apply_foundation_profile(args)
    assert args.window_activation_checkpointing is False
    assert args.diagnostic_interval == 0


def test_window_output_tensor_tuple_roundtrip() -> None:
    outputs = {
        key: torch.tensor(float(index + 1), dtype=torch.float32)
        for index, key in enumerate(_MODULE._WINDOW_OUTPUT_TENSOR_KEYS)
    }
    packed = _MODULE._window_outputs_to_tensor_tuple(outputs)
    restored = _MODULE._window_outputs_from_tensor_tuple(packed)
    assert tuple(restored) == _MODULE._WINDOW_OUTPUT_TENSOR_KEYS
    for key in _MODULE._WINDOW_OUTPUT_TENSOR_KEYS:
        torch.testing.assert_close(restored[key], outputs[key])


def test_checkpoint_dummy_input_is_not_a_parameter_view() -> None:
    module = torch.nn.Linear(4, 3, bias=False)
    dummy = _MODULE._checkpoint_dummy_input(module)
    assert dummy.shape == ()
    assert dummy.requires_grad is True
    assert dummy._base is None
    param = next(module.parameters())
    assert dummy.device == param.device


def test_normalize_train_args_sets_percentile_clip_defaults() -> None:
    args = _base_args()
    args.grad_clip_mode = None
    args.grad_clip_percentile = None
    args.grad_clip_window = None

    _MODULE._normalize_train_args(args)

    assert args.grad_clip_mode == "percentile"
    assert args.grad_clip_percentile == pytest.approx(75.0)
    assert args.grad_clip_window == 100


def test_final_tactile_defaults_align_with_current_training_spec() -> None:
    args = _base_args()
    assert args.hidden_dim == 512
    assert args.posterior_hidden_dim == 512
    assert args.innovation_dim == 512
    assert args.control_dim == 512
    assert args.future_hidden_dim == 512
    assert args.semantic_dim == 2048
    assert args.semantic_cross_dim == 2048
    assert args.stride == 4
    assert args.max_points == 1024
    assert args.crop_radius_m == pytest.approx(0.10)
    assert args.pt_bag_radius_m == pytest.approx(0.045)
    assert args.pt_bag_sigma_m == pytest.approx(0.015)
    assert args.pt_bag_kmin == 32
    assert _MODULE._SPEC_DEFAULTS.crop_radius_m == pytest.approx(0.10)


def test_normalize_train_args_disables_semantic_gradient_checkpointing_for_accumulation() -> None:
    args = _base_args()
    args.semantic_mode = "paligemma"
    args.semantic_gradient_checkpointing = True
    args.accum_steps = 2

    _MODULE._normalize_train_args(args)

    assert args.semantic_gradient_checkpointing is False
    assert args.semantic_gradient_checkpointing_disabled_for_accum is True


def test_normalize_train_args_keeps_semantic_gradient_checkpointing_for_fsdp() -> None:
    args = _base_args()
    args.training_strategy = "fsdp_full_shard"
    args.semantic_mode = "paligemma"
    args.semantic_gradient_checkpointing = True

    _MODULE._normalize_train_args(args)

    assert args.semantic_gradient_checkpointing is True
    assert getattr(args, "semantic_gradient_checkpointing_disabled_for_fsdp", False) is False


def test_gemma_training_does_not_force_gradient_checkpointing() -> None:
    source = _GEMMA_PYTORCH_PATH.read_text(encoding="utf-8")

    assert "Forcing gradient checkpointing to be enabled for Gemma expert model" not in source
    assert "self.gemma_expert.model.gradient_checkpointing = True" not in source


def test_setup_distributed_defaults_torch_distributed_debug_to_info_for_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.delenv("TORCH_DISTRIBUTED_DEBUG", raising=False)
    monkeypatch.delenv("OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL", raising=False)
    monkeypatch.setattr(_MODULE.dist, "is_initialized", lambda: False)
    init_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        _MODULE.dist,
        "init_process_group",
        lambda backend, init_method=None: init_calls.append((backend, init_method)),
    )

    use_ddp, rank, world_size, device, runtime_env = _MODULE._setup_distributed("cpu")

    assert use_ddp is True
    assert rank == 0
    assert world_size == 4
    assert device.type == "cpu"
    assert os.environ["TORCH_DISTRIBUTED_DEBUG"] == "INFO"
    assert runtime_env.torch_distributed_debug == "INFO"
    assert runtime_env.torch_distributed_debug_source == "defaulted_for_ddp"
    assert init_calls == [("gloo", "env://")]


def test_setup_distributed_rejects_detail_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    monkeypatch.delenv("OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL", raising=False)
    monkeypatch.setattr(_MODULE.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(_MODULE.dist, "init_process_group", lambda backend, init_method=None: None)

    with pytest.raises(RuntimeError, match="TORCH_DISTRIBUTED_DEBUG=DETAIL is not allowed by default"):
        _MODULE._setup_distributed("cpu")


def test_setup_distributed_preserves_detail_with_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    monkeypatch.setenv("OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL", "1")
    monkeypatch.setattr(_MODULE.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(_MODULE.dist, "init_process_group", lambda backend, init_method=None: None)

    _use_ddp, _rank, _world_size, _device, runtime_env = _MODULE._setup_distributed("cpu")

    assert os.environ["TORCH_DISTRIBUTED_DEBUG"] == "DETAIL"
    assert runtime_env.torch_distributed_debug == "DETAIL"
    assert runtime_env.allow_torch_distributed_debug_detail is True
    assert runtime_env.torch_distributed_debug_source == "inherited"


def test_configure_cuda_allocator_env_defaults_expandable_segments_for_fsdp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    alloc_conf, source = _MODULE._configure_cuda_allocator_env(
        requested_device="cuda",
        training_strategy="fsdp_full_shard",
        world_size=4,
    )

    assert alloc_conf == "expandable_segments:True"
    assert source == "defaulted_for_fsdp"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_configure_cuda_allocator_env_rejects_non_expandable_override_for_fsdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

    with pytest.raises(RuntimeError, match="expects PYTORCH_CUDA_ALLOC_CONF to include expandable_segments:True"):
        _MODULE._configure_cuda_allocator_env(
            requested_device="cuda",
            training_strategy="fsdp_full_shard",
            world_size=4,
        )


def test_setup_distributed_requires_local_rank_for_ddp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(RuntimeError, match="LOCAL_RANK must be set"):
        _MODULE._setup_distributed("cpu")


def test_validate_train_args_rejects_invalid_grad_clip_percentile_mode() -> None:
    args = _base_args()
    args.grad_clip_mode = "weird"
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="grad_clip_mode"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_invalid_grad_clip_percentile_range() -> None:
    args = _base_args()
    args.grad_clip_mode = "percentile"
    args.grad_clip_percentile = 125.0
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="grad_clip_percentile"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_invalid_grad_clip_window() -> None:
    args = _base_args()
    args.grad_clip_window = 0
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="grad_clip_window"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_negative_semantic_projection_chunk_size() -> None:
    args = _base_args()
    args.semantic_projection_chunk_size = -1
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="semantic_projection_chunk_size"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_negative_semantic_mlp_chunk_size() -> None:
    args = _base_args()
    args.semantic_mlp_chunk_size = -1
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="semantic_mlp_chunk_size"):
        _MODULE._validate_train_args(args)


def test_grad_clip_controller_percentile_mode_has_no_threshold_until_history_full() -> None:
    args = _base_args()
    args.grad_clip_mode = "percentile"
    args.grad_clip_percentile = 75.0
    args.grad_clip_window = 4
    _MODULE._normalize_train_args(args)

    controller = _MODULE._GradClipController.from_args(args)

    assert controller.threshold() is None
    controller.observe(1.0)
    controller.observe(2.0)
    controller.observe(3.0)
    assert controller.threshold() is None
    controller.observe(4.0)
    assert controller.threshold() == pytest.approx(float(np.percentile([1.0, 2.0, 3.0, 4.0], 75.0)))


def test_grad_clip_controller_percentile_mode_slides_window() -> None:
    args = _base_args()
    args.grad_clip_mode = "percentile"
    args.grad_clip_percentile = 75.0
    args.grad_clip_window = 4
    _MODULE._normalize_train_args(args)

    controller = _MODULE._GradClipController.from_args(args)
    for value in (1.0, 2.0, 3.0, 4.0, 10.0):
        controller.observe(value)

    assert controller.history_size() == 4
    assert list(controller.history) == [2.0, 3.0, 4.0, 10.0]
    assert controller.threshold() == pytest.approx(float(np.percentile([2.0, 3.0, 4.0, 10.0], 75.0)))


def test_grad_clip_controller_state_roundtrip_requires_matching_config() -> None:
    args = _base_args()
    args.grad_clip_mode = "percentile"
    args.grad_clip_percentile = 75.0
    args.grad_clip_window = 4
    _MODULE._normalize_train_args(args)

    controller = _MODULE._GradClipController.from_args(args)
    for value in (1.0, 2.0, 3.0, 4.0):
        controller.observe(value)

    restored = _MODULE._GradClipController.from_args(args)
    assert restored.load_state_dict(controller.state_dict()) is True
    assert list(restored.history) == [1.0, 2.0, 3.0, 4.0]

    mismatched_args = _base_args()
    mismatched_args.grad_clip_mode = "percentile"
    mismatched_args.grad_clip_percentile = 90.0
    mismatched_args.grad_clip_window = 4
    _MODULE._normalize_train_args(mismatched_args)
    mismatched = _MODULE._GradClipController.from_args(mismatched_args)
    assert mismatched.load_state_dict(controller.state_dict()) is False


def test_grad_clip_controller_percentile_restore_ignores_fixed_norm() -> None:
    args = _base_args()
    args.grad_clip_mode = "percentile"
    args.grad_clip_percentile = 75.0
    args.grad_clip_window = 4
    args.grad_clip_norm = 1.0
    _MODULE._normalize_train_args(args)

    controller = _MODULE._GradClipController.from_args(args)
    for value in (1.0, 2.0, 3.0, 4.0):
        controller.observe(value)

    restored_args = _base_args()
    restored_args.grad_clip_mode = "percentile"
    restored_args.grad_clip_percentile = 75.0
    restored_args.grad_clip_window = 4
    restored_args.grad_clip_norm = 7.5
    _MODULE._normalize_train_args(restored_args)

    restored = _MODULE._GradClipController.from_args(restored_args)
    assert restored.load_state_dict(controller.state_dict()) is True
    assert list(restored.history) == [1.0, 2.0, 3.0, 4.0]


def test_foundation_profile_enables_semantic_and_trainable_backbones() -> None:
    args = _base_args()
    args.use_foundation_backbones = True
    _MODULE._apply_foundation_profile(args)
    assert args.semantic_mode == "paligemma"
    assert args.semantic_source == "auto"
    assert args.visual_trainable is True
    assert args.tactile_trainable is True
    assert args.point_backbone_trainable is True
    assert args.semantic_trainable is True


def test_perception_finetune_mode_frozen_overrides_foundation_backbone_trainability() -> None:
    args = _base_args()
    args.use_foundation_backbones = True
    args.perception_finetune_mode = "frozen"
    _MODULE._apply_foundation_profile(args)
    _MODULE._normalize_train_args(args)

    assert args.semantic_mode == "paligemma"
    assert args.semantic_trainable is True
    assert args.point_backbone == "sonata"
    assert args.point_backbone_trainable is False
    assert args.visual_mode == "encoder"
    assert args.visual_finetune_mode == "frozen"
    assert args.visual_trainable is False
    assert args.visual_feature_mode == "auto"
    assert args.tactile_mode == "encoder"
    assert args.tactile_trainable is False
    assert args.use_tactile is True


def test_perception_finetune_mode_full_forces_foundation_backbone_trainability() -> None:
    args = _base_args()
    args.use_foundation_backbones = True
    args.perception_finetune_mode = "full"
    args.visual_finetune_mode = "frozen"
    _MODULE._apply_foundation_profile(args)
    _MODULE._normalize_train_args(args)

    assert args.point_backbone_trainable is True
    assert args.visual_finetune_mode == "full"
    assert args.visual_trainable is True
    assert args.tactile_trainable is True


def test_normalize_train_args_resolves_visual_finetune_mode() -> None:
    args = _base_args()
    args.visual_finetune_mode = "frozen"
    args.visual_trainable = True

    _MODULE._normalize_train_args(args)

    assert args.visual_finetune_mode == "frozen"
    assert args.visual_trainable is False

    args = _base_args()
    args.visual_finetune_mode = "full"
    args.visual_trainable = False

    _MODULE._normalize_train_args(args)

    assert args.visual_finetune_mode == "full"
    assert args.visual_trainable is True


def test_picf_photometric_augmentation_preserves_geometry_related_fields() -> None:
    rng = np.random.default_rng(7)
    image = np.full((4, 5, 3), 128, dtype=np.uint8)
    augmented = _MODULE._apply_picf_photometric_augmentation(
        image,
        rng=rng,
        strength="conservative",
    )

    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype
    assert not np.array_equal(augmented, image)


def test_validate_train_args_rejects_unimplemented_multimodal_geometry_augmentation(tmp_path: Path) -> None:
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}", encoding="utf-8")
    args = _base_args()
    args.picf_augmentation_mode = "multimodal_geometry"
    args.action_norm_stats_path = str(stats)
    args.prompt_state_norm_stats_path = str(stats)
    _MODULE._normalize_train_args(args)

    with pytest.raises(NotImplementedError, match="multimodal_geometry"):
        _MODULE._validate_train_args(args)


def test_fsdp_root_ignored_modules_collects_fully_frozen_backbones() -> None:
    class _DummyTrainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.semantic_encoder = torch.nn.Linear(4, 4)
            self.core = types.SimpleNamespace(
                point_feature_extractor=torch.nn.Linear(4, 4),
                visual_encoder=torch.nn.Linear(4, 4),
                tactile_encoder=torch.nn.Linear(4, 4),
            )

    trainer = _DummyTrainer()
    for param in trainer.core.visual_encoder.parameters():
        param.requires_grad_(False)

    ignored = _MODULE._fsdp_root_ignored_modules(trainer)

    assert trainer.core.visual_encoder in ignored
    assert trainer.semantic_encoder not in ignored
    assert trainer.core.point_feature_extractor not in ignored
    assert trainer.core.tactile_encoder not in ignored


def test_fsdp_root_ignored_modules_collects_fully_frozen_core() -> None:
    class _Core(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.point_feature_extractor = torch.nn.Linear(4, 4)
            self.visual_encoder = torch.nn.Linear(4, 4)
            self.tactile_encoder = torch.nn.Linear(4, 4)

    class _DummyTrainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.semantic_encoder = torch.nn.Linear(4, 4)
            self.core = _Core()

    trainer = _DummyTrainer()
    _MODULE._freeze_initialized_module_parameters(trainer.core)

    ignored = _MODULE._fsdp_root_ignored_modules(trainer)

    assert trainer.core in ignored
    assert trainer.semantic_encoder not in ignored
    assert trainer.core.point_feature_extractor not in ignored
    assert trainer.core.visual_encoder not in ignored
    assert trainer.core.tactile_encoder not in ignored


def test_load_tactile_backgrounds_npz_roundtrip(tmp_path: Path) -> None:
    bg_path = tmp_path / "tactile_backgrounds.npz"
    np.savez(
        bg_path,
        digit=np.full((8, 8, 3), 11, dtype=np.uint8),
        gelsight_mini=np.full((8, 8, 3), 22, dtype=np.uint8),
    )

    payload = _MODULE._load_tactile_backgrounds_npz(str(bg_path))

    assert payload is not None
    assert tuple(sorted(payload)) == ("digit", "gelsight_mini")
    assert int(payload["digit"][0, 0, 0]) == 11
    assert int(payload["gelsight_mini"][0, 0, 0]) == 22


def test_load_tactile_contact_stats_json_roundtrip(tmp_path: Path) -> None:
    stats_path = tmp_path / "tactile_contact_stats.json"
    stats_path.write_text(
        '{"tau_on": 0.23, "tau_off": 0.22, "temperature": 0.005, "score_mode": "rgb_latent"}',
        encoding="utf-8",
    )

    payload = _MODULE._load_tactile_contact_stats_json(str(stats_path))

    assert payload is not None
    assert payload["tau_on"] == pytest.approx(0.23)
    assert payload["tau_off"] == pytest.approx(0.22)
    assert payload["temperature"] == pytest.approx(0.005)


def test_load_tactile_calibration_json_roundtrip(tmp_path: Path) -> None:
    calibration_path = tmp_path / "tactile_fingertip_calibration.json"
    calibration_path.write_text(
        '{"u_open_local": [0, 0, 1], "o_local": [0.0, 0.02, -0.03], "recommended_pt_bag_radius_m": 0.035, "recommended_pt_bag_sigma_m": 0.0117}',
        encoding="utf-8",
    )

    payload = _MODULE._load_tactile_calibration_json(str(calibration_path))

    assert payload is not None
    assert payload["recommended_pt_bag_radius_m"] == pytest.approx(0.035)
    assert payload["recommended_pt_bag_sigma_m"] == pytest.approx(0.0117)


def test_calvin_transition_source_emits_dynamic_tactile_packet_and_extra_fields(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)
    backgrounds = {
        "digit": np.full((32, 32, 3), 7, dtype=np.uint8),
        "gelsight_mini": np.full((32, 32, 3), 9, dtype=np.uint8),
    }
    source = _MODULE._CalvinTransitionSource(
        str(root),
        split="training",
        backend="dir",
        unroll_steps=2,
        action_horizon=1,
        use_tactile=True,
        tactile_backgrounds_by_sensor=backgrounds,
        use_scene_obs=True,
    )

    window = source.window(0)
    frame = window.frames[0]

    assert frame.depth_gripper is not None
    assert frame.scene_obs is not None
    assert frame.tactile is not None
    assert frame.tactile.background_rgb_by_sensor is not None
    np.testing.assert_array_equal(frame.tactile.background_rgb_by_sensor["digit"], backgrounds["digit"])
    width = float(frame.robot_obs[6])
    left_x = float(frame.tactile.sensors[0].T_sens_to_wrist[0, 3])
    right_x = float(frame.tactile.sensors[1].T_sens_to_wrist[0, 3])
    assert left_x == pytest.approx(0.5 * width)
    assert right_x == pytest.approx(-0.5 * width)


def test_calvin_transition_source_normalizes_actions_when_requested(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)

    class _Normalizer:
        def normalize_np(self, x):
            return np.asarray(x, dtype=np.float32) * 0.5

    source = _MODULE._CalvinTransitionSource(
        str(root),
        split="training",
        backend="dir",
        unroll_steps=1,
        action_horizon=1,
        action_normalizer=_Normalizer(),
    )

    window = source.window(0)
    frame = window.frames[0]
    raw = source.reader.read_npz(frame.step_id, keys=["rel_actions"])["rel_actions"].astype(np.float32)
    np.testing.assert_allclose(frame.action, raw * 0.5)
    source.close()


def test_calvin_transition_source_emits_action_chunk_when_action_horizon_requested(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)

    source = _MODULE._CalvinTransitionSource(
        str(root),
        split="training",
        backend="dir",
        unroll_steps=1,
        action_horizon=3,
    )

    window = source.window(0)
    frame = window.frames[0]

    assert frame.action_chunk is not None
    assert frame.action_chunk.ndim == 2
    assert frame.action_chunk.shape[0] == 3
    source.close()


def test_calvin_transition_source_samples_segments_uniformly_but_picks_start_within_segment(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)

    source = _MODULE._CalvinTransitionSource(
        str(root),
        split="training",
        backend="dir",
        unroll_steps=1,
        action_horizon=1,
    )

    assert len(source) == 2
    assert len(source.window_index) == 6
    assert source.sample_window_metadata(0) == (0, 0)
    assert source.sample_window_metadata(1) == (1, 4)

    rng = np.random.default_rng(0)
    seg0_starts = {source.sample_window_metadata(0, rng=rng)[1] for _ in range(24)}
    seg1_starts = {source.sample_window_metadata(1, rng=rng)[1] for _ in range(24)}

    assert seg0_starts <= {0, 1, 2}
    assert seg1_starts <= {4, 5, 6}
    assert len(seg0_starts) > 1
    assert len(seg1_starts) > 1
    source.close()


def test_validate_train_args_rejects_incompatible_attention_shape() -> None:
    args = _base_args()
    args.hidden_dim = 250
    _MODULE._normalize_train_args(args)
    with pytest.raises(ValueError, match="hidden_dim must be divisible by attention_heads"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_incompatible_semantic_attention_shape() -> None:
    args = _base_args()
    args.semantic_dim = 2050
    _MODULE._normalize_train_args(args)
    with pytest.raises(ValueError, match="semantic_dim must be divisible by attention_heads"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_incompatible_semantic_cross_shape() -> None:
    args = _base_args()
    args.semantic_cross_dim = 510
    _MODULE._normalize_train_args(args)
    with pytest.raises(ValueError, match="semantic_cross_dim must be divisible by attention_heads"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_invalid_predictive_semantic_dropout_prob() -> None:
    args = _base_args()
    args.predictive_semantic_dropout_prob = 1.0
    _MODULE._normalize_train_args(args)
    with pytest.raises(ValueError, match="predictive_semantic_dropout_prob must be in \\[0, 1\\)"):
        _MODULE._validate_train_args(args)


def test_validate_backbone_args_loads_tactile_contact_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-4frames.pth"
    ckpt.write_bytes(b"stub")
    backgrounds = tmp_path / "tactile_backgrounds.npz"
    np.savez(backgrounds, digit=np.zeros((4, 4, 3), dtype=np.uint8), gelsight_mini=np.zeros((4, 4, 3), dtype=np.uint8))
    calibration = tmp_path / "tactile_fingertip_calibration.json"
    calibration.write_text('{"u_open_local": [1, 0, 0], "o_local": [0, 0, 0]}', encoding="utf-8")
    stats = tmp_path / "tactile_contact_stats.json"
    stats.write_text('{"tau_on": 0.23, "tau_off": 0.22, "temperature": 0.005}', encoding="utf-8")

    monkeypatch.setattr(_MODULE, "_default_anytouch_checkpoint", lambda: str(ckpt))
    monkeypatch.setattr(_MODULE, "_default_tactile_backgrounds_path", lambda: str(backgrounds))
    monkeypatch.setattr(_MODULE, "_default_tactile_calibration_path", lambda: str(calibration))
    monkeypatch.setattr(_MODULE, "_default_tactile_contact_stats_path", lambda: str(stats))

    args = _base_args()
    args.tactile_mode = "encoder"
    args.point_backbone = "rgb"
    args.device = "cuda"
    _MODULE._validate_backbone_args(args)

    assert args.tactile_contact_stats_path == str(stats)
    assert args.tactile_contact_tau_on == pytest.approx(0.23)
    assert args.tactile_contact_tau_off == pytest.approx(0.22)
    assert args.tactile_contact_temperature == pytest.approx(0.005)


def test_validate_backbone_args_applies_recommended_pt_bag_geometry_when_defaults_are_implicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "checkpoint-4frames.pth"
    ckpt.write_bytes(b"stub")
    backgrounds = tmp_path / "tactile_backgrounds.npz"
    np.savez(backgrounds, digit=np.zeros((4, 4, 3), dtype=np.uint8), gelsight_mini=np.zeros((4, 4, 3), dtype=np.uint8))
    calibration = tmp_path / "tactile_fingertip_calibration.json"
    calibration.write_text(
        '{"u_open_local": [0, 0, 1], "o_local": [0.0, 0.02, -0.03], "recommended_pt_bag_radius_m": 0.03545, "recommended_pt_bag_sigma_m": 0.01182}',
        encoding="utf-8",
    )
    stats = tmp_path / "tactile_contact_stats.json"
    stats.write_text('{"tau_on": 0.23, "tau_off": 0.22, "temperature": 0.005}', encoding="utf-8")

    monkeypatch.setattr(_MODULE, "_default_anytouch_checkpoint", lambda: str(ckpt))
    monkeypatch.setattr(_MODULE, "_default_tactile_backgrounds_path", lambda: str(backgrounds))
    monkeypatch.setattr(_MODULE, "_default_tactile_calibration_path", lambda: str(calibration))
    monkeypatch.setattr(_MODULE, "_default_tactile_contact_stats_path", lambda: str(stats))

    args = _base_args()
    args.tactile_mode = "encoder"
    args.point_backbone = "rgb"
    args.device = "cuda"
    args.pt_bag_radius_m = None
    args.pt_bag_sigma_m = None
    _MODULE._normalize_train_args(args)
    _MODULE._validate_backbone_args(args)

    assert args.pt_bag_radius_m == pytest.approx(0.03545)
    assert args.pt_bag_sigma_m == pytest.approx(0.01182)
    assert args.pt_bag_radius_m_source == "calibration"
    assert args.pt_bag_sigma_m_source == "calibration"


def test_validate_train_args_rejects_cpu_sonata() -> None:
    args = _base_args()
    args.device = "cpu"
    _MODULE._normalize_train_args(args)
    with pytest.raises(RuntimeError, match="point_backbone=sonata currently requires CUDA"):
        _MODULE._validate_train_args(args)


def test_build_optimizer_preserves_foundation_lr_scales() -> None:
    core = types.SimpleNamespace(
        point_feature_extractor=torch.nn.Linear(3, 4, bias=False),
        visual_encoder=torch.nn.Linear(5, 6, bias=False),
        tactile_encoder=torch.nn.Linear(7, 8, bias=False),
    )
    trainer = torch.nn.Module()
    trainer.core = core
    trainer.semantic_encoder = torch.nn.Linear(9, 10, bias=False)
    trainer.head = torch.nn.Linear(11, 12, bias=False)
    args = _base_args()
    optimizer, group_info = _MODULE._build_optimizer(trainer, args=args)
    del optimizer
    name_to_scale = {item["name"]: item["lr_scale"] for item in group_info}
    assert name_to_scale["point_backbone"] == pytest.approx(0.25)
    assert name_to_scale["visual_backbone"] == pytest.approx(0.25)
    assert name_to_scale["tactile_backbone"] == pytest.approx(0.25)
    assert name_to_scale["semantic_backbone"] == pytest.approx(0.25)
    assert name_to_scale["picf_core"] == pytest.approx(1.0)
    assert {item["optimizer_sharding"] for item in group_info} == {"none"}


def test_build_optimizer_rejects_zero1_without_initialized_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = torch.nn.Module()
    trainer.core = types.SimpleNamespace(
        point_feature_extractor=torch.nn.Linear(3, 4, bias=False),
        visual_encoder=None,
        tactile_encoder=None,
    )
    trainer.semantic_encoder = None
    args = _base_args()
    args.optimizer_sharding = "zero1"

    monkeypatch.setattr(_MODULE.dist, "is_available", lambda: True)
    monkeypatch.setattr(_MODULE.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="optimizer_sharding=zero1 requires an initialized multi-rank"):
        _MODULE._build_optimizer(trainer, args=args)


def test_validate_train_args_rejects_fsdp_with_optimizer_sharding() -> None:
    args = _base_args()
    _MODULE._normalize_train_args(args)
    args.training_strategy = "fsdp_full_shard"
    args.optimizer_sharding = "zero1"

    with pytest.raises(ValueError, match="training_strategy=fsdp_full_shard is incompatible with optimizer_sharding"):
        _MODULE._validate_train_args(args)


def test_split_optimizer_groups_by_dense_type_preserves_group_metadata() -> None:
    fp32 = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    bf16 = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    groups = [
        {
            "name": "mixed",
            "params": [fp32, bf16],
            "lr": 0.01,
            "lr_scale": 0.25,
        }
    ]

    partitions = _MODULE._split_optimizer_groups_by_dense_type(groups)

    assert [dense_type for dense_type, _groups in partitions] == [str(fp32.type()), str(bf16.type())]
    assert [part_groups[0]["params"] for _dense_type, part_groups in partitions] == [[fp32], [bf16]]
    assert [part_groups[0]["name"] for _dense_type, part_groups in partitions] == ["mixed", "mixed"]
    assert [part_groups[0]["lr_scale"] for _dense_type, part_groups in partitions] == [0.25, 0.25]
    assert [part_groups[0]["dense_type"] for _dense_type, part_groups in partitions] == [
        str(fp32.type()),
        str(bf16.type()),
    ]


def test_split_optimizer_groups_by_dense_type_skips_uninitialized_params() -> None:
    fp32 = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    lazy = torch.nn.LazyLinear(4, bias=False).weight
    groups = [
        {
            "name": "mixed",
            "params": [lazy, fp32],
            "lr": 0.01,
            "lr_scale": 0.25,
        }
    ]

    partitions = _MODULE._split_optimizer_groups_by_dense_type(groups)

    assert [dense_type for dense_type, _groups in partitions] == [str(fp32.type())]
    assert partitions[0][1][0]["params"] == [fp32]


def test_infer_tactile_dense_dim_prefers_anytouch_native_hidden_size() -> None:
    vision_cfg = types.SimpleNamespace(hidden_size=768)
    tactile_model = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=vision_cfg))
    tactile_encoder = types.SimpleNamespace(model=tactile_model)
    core = types.SimpleNamespace(tactile_encoder=tactile_encoder)

    assert _MODULE._infer_tactile_dense_dim(core) == 768


def test_infer_tactile_dense_dim_falls_back_to_default() -> None:
    core = types.SimpleNamespace(tactile_encoder=None)

    assert _MODULE._infer_tactile_dense_dim(core) == 768


def test_materialize_model_parameters_initializes_task_tactile_reread() -> None:
    class _FakeCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.config = types.SimpleNamespace(
                hidden_dim=8,
                tactile_group_proposals=2,
                task_local_queries=8,
                task_global_queries=1,
                task_instruction_queries=2,
            )
            self.tactile_token_proj = torch.nn.Linear(4, 4)
            self.tactile_error_encoder = torch.nn.Linear(4, 4)
            self.visual_error_encoder = torch.nn.Linear(4, 4)
            self.visual_real_error_encoder = torch.nn.Linear(4, 4)
            self.point_error_encoder = torch.nn.Linear(4, 4)
            self.innovation_proj = torch.nn.Linear(4, 4)
            self.tactile_route_reread = LazyCrossAttentionRead(self.config.hidden_dim, inner_dim=self.config.hidden_dim)
            self.task_tactile_reread = LazyCrossAttentionRead(self.config.hidden_dim, inner_dim=self.config.hidden_dim)
            self.tactile_encoder = types.SimpleNamespace(
                model=types.SimpleNamespace(
                    config=types.SimpleNamespace(vision_config=types.SimpleNamespace(hidden_size=768))
                )
            )

    class _FakeTrainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = _FakeCore()

        def forward(self, _window):
            return torch.tensor(0.0)

    class _FakeSource:
        def __len__(self) -> int:
            return 1

        def window(self, _index: int):
            return object()

    trainer = _FakeTrainer()

    assert isinstance(trainer.core.task_tactile_reread.key_proj.weight, torch.nn.parameter.UninitializedParameter)

    _MODULE._materialize_model_parameters(trainer, source=_FakeSource(), rank=0)

    assert not isinstance(
        trainer.core.task_tactile_reread.key_proj.weight,
        torch.nn.parameter.UninitializedParameter,
    )
    assert tuple(trainer.core.task_tactile_reread.key_proj.weight.shape) == (8, 768)


@contextlib.contextmanager
def _single_rank_process_group() -> None:
    fd, init_path = tempfile.mkstemp()
    os.close(fd)
    os.unlink(init_path)
    _MODULE.dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=0, world_size=1)
    try:
        yield
    finally:
        _MODULE.dist.destroy_process_group()


def test_fsdp_wrap_and_checkpoint_roundtrip_on_cpu(tmp_path: Path) -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _TinyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.point_feature_extractor = torch.nn.Linear(4, 4)
            self.visual_encoder = torch.nn.Linear(4, 4)
            self.tactile_encoder = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 2)

    class _TinyTrainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = _TinyCore()
            self.semantic_encoder = torch.nn.Linear(4, 4)
            self.policy = types.SimpleNamespace(semantic_encoder=self.semantic_encoder)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            hidden = self.core.point_feature_extractor(inputs)
            hidden = self.core.visual_encoder(hidden)
            hidden = self.core.tactile_encoder(hidden)
            hidden = self.semantic_encoder(hidden)
            return self.core.head(hidden).sum()

    args = _base_args()
    args.training_strategy = "fsdp_full_shard"
    args.optimizer_sharding = "none"

    with _single_rank_process_group():
        trainer = _TinyTrainer()
        wrapped = _MODULE._wrap_model_for_training_strategy(trainer, args=args, device=torch.device("cpu"))
        assert _MODULE._is_fsdp_model(wrapped)
        unwrapped = _MODULE._unwrap_training_model(wrapped)
        assert _MODULE._is_fsdp_model(unwrapped.semantic_encoder)
        assert _MODULE._is_fsdp_model(unwrapped.core.point_feature_extractor)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        loss = wrapped(torch.randn(2, 4))
        loss.backward()
        optimizer.step()

        output_dir = tmp_path / "fsdp_ckpt"
        output_dir.mkdir(parents=True, exist_ok=True)
        _MODULE._save_checkpoint(
            output_dir=output_dir,
            model=wrapped,
            optimizer=optimizer,
            step=13,
            args=args,
            rank=0,
            device=torch.device("cpu"),
        )

        reloaded = _TinyTrainer()
        wrapped_reloaded = _MODULE._wrap_model_for_training_strategy(
            reloaded,
            args=args,
            device=torch.device("cpu"),
        )
        reloaded_optimizer = torch.optim.AdamW(wrapped_reloaded.parameters(), lr=1e-3)
        step = _MODULE._load_checkpoint(
            path=output_dir / "13",
            model=wrapped_reloaded,
            optimizer=reloaded_optimizer,
            device=torch.device("cpu"),
        )

        assert step == 13


def test_fsdp_wrap_reattaches_core_transformer_stack_children_on_cpu() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _TinyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.point_feature_extractor = torch.nn.Linear(4, 4)
            self.visual_encoder = torch.nn.Linear(4, 4)
            self.tactile_encoder = torch.nn.Linear(4, 4)
            self.token_fusion = torch.nn.Linear(4, 4)
            self.obs_self = torch.nn.Linear(4, 4)
            self.posterior_self = torch.nn.Linear(4, 4)
            self.task_self = torch.nn.Linear(4, 4)
            self.predictive_world = torch.nn.Linear(4, 4)
            self.predictive_semantic_world = torch.nn.Linear(4, 4)
            self.control_world = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 2)

    class _TinyTrainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = _TinyCore()
            self.semantic_encoder = torch.nn.Linear(4, 4)
            self.policy = types.SimpleNamespace(semantic_encoder=self.semantic_encoder)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            hidden = self.core.point_feature_extractor(inputs)
            hidden = self.core.token_fusion(hidden)
            hidden = self.core.obs_self(hidden)
            hidden = self.core.posterior_self(hidden)
            hidden = self.core.task_self(hidden)
            hidden = self.core.predictive_world(hidden)
            hidden = self.core.predictive_semantic_world(hidden)
            hidden = self.core.control_world(hidden)
            hidden = self.core.visual_encoder(hidden)
            hidden = self.core.tactile_encoder(hidden)
            hidden = self.semantic_encoder(hidden)
            return self.core.head(hidden).sum()

    args = _base_args()
    args.training_strategy = "fsdp_full_shard"
    args.optimizer_sharding = "none"

    with _single_rank_process_group():
        trainer = _TinyTrainer()
        wrapped = _MODULE._wrap_model_for_training_strategy(trainer, args=args, device=torch.device("cpu"))
        unwrapped = _MODULE._unwrap_training_model(wrapped)
        for attr_name in (
            "token_fusion",
            "obs_self",
            "posterior_self",
            "task_self",
            "predictive_world",
            "predictive_semantic_world",
            "control_world",
        ):
            assert _MODULE._is_fsdp_model(getattr(unwrapped.core, attr_name)), attr_name


def test_fsdp_wrap_kwargs_prefers_backward_post() -> None:
    if _MODULE.FullyShardedDataParallel is None or _MODULE.BackwardPrefetch is None:
        pytest.skip("FSDP BackwardPrefetch is not available in this torch build.")

    kwargs = _MODULE._fsdp_wrap_kwargs(device=torch.device("cpu"))
    assert kwargs["use_orig_params"] is False
    assert kwargs["limit_all_gathers"] is True
    assert kwargs["backward_prefetch"] == _MODULE.BackwardPrefetch.BACKWARD_POST


def test_fsdp_wrap_recursively_splits_mixed_dtype_subtrees_on_cpu() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _MixedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch = torch.nn.Linear(4, 4).float()
            self.block = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.norm = torch.nn.LayerNorm(4).float()
            self.head = torch.nn.Linear(4, 4).float()
            self.position_ids = torch.nn.Parameter(torch.arange(4, dtype=torch.int64), requires_grad=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            x = self.patch(inputs.float())
            y = self.block(x.to(torch.bfloat16)).to(torch.float32)
            x = self.norm(x + y)
            return self.head(x)

    with _single_rank_process_group():
        module = _MixedModule()
        wrapped = _MODULE._fsdp_wrap_uniform_dtype_subtrees(module, device=torch.device("cpu"))
        assert not _MODULE._is_fsdp_model(wrapped)
        nested_wrappers = [m for m in wrapped.modules() if _MODULE._is_fsdp_model(m)]
        assert nested_wrappers, "generic mixed-dtype helper should recursively split into nested FSDP wrappers"
        assert "position_ids" not in dict(wrapped.named_parameters())
        assert "position_ids" in dict(wrapped.named_buffers())
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        loss = wrapped(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()


def test_fsdp_wrap_root_with_ignored_non_dominant_dtypes_keeps_single_boundary_on_cpu() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _MixedRootModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch = torch.nn.Linear(4, 4).float()
            self.block = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.norm = torch.nn.LayerNorm(4).float()
            self.head = torch.nn.Linear(4, 4).to(torch.bfloat16)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            x = self.patch(inputs.float())
            x = self.block(x.to(torch.bfloat16)).to(torch.float32) + x
            x = self.norm(x)
            return self.head(x.to(torch.bfloat16)).to(torch.float32)

    with _single_rank_process_group():
        module = _MixedRootModule()
        wrapped = _MODULE._fsdp_wrap_root_with_ignored_non_dominant_dtypes(module, device=torch.device("cpu"))

        assert _MODULE._is_fsdp_model(wrapped)
        assert not any(
            _MODULE._is_fsdp_model(child) for child in wrapped.module.modules() if child is not wrapped.module
        ), "semantic-style root wrap should not recursively shard internal mixed-dtype subtrees"

        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        loss = wrapped(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()


def test_prepare_semantic_runtime_leaf_fsdp_wraps_declared_hot_leaves_on_cpu() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _SemanticStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.uniform_leaf = torch.nn.Linear(4, 4)
            self.mixed_root = torch.nn.Sequential(
                torch.nn.Linear(4, 4).float(),
                torch.nn.Linear(4, 4).to(torch.bfloat16),
            )

        def fsdp_runtime_leaf_module_specs(self):
            return [
                (self, "uniform_leaf", "uniform_recursive"),
                (self, "mixed_root", "mixed_root"),
            ]

    with _single_rank_process_group():
        module = _SemanticStub()
        prepared = _MODULE._prepare_semantic_runtime_leaf_fsdp(module, device=torch.device("cpu"))

        assert prepared is module
        assert _MODULE._is_fsdp_model(module.uniform_leaf)
        assert _MODULE._is_fsdp_model(module.mixed_root)


def test_fsdp_wrap_keeps_uniform_subtrees_at_single_boundary_on_cpu() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _UniformNestedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.inner(inputs)

    with _single_rank_process_group():
        module = _UniformNestedModule()
        wrapped = _MODULE._fsdp_wrap_uniform_dtype_subtrees(module, device=torch.device("cpu"))
        assert _MODULE._is_fsdp_model(wrapped)
        assert not any(_MODULE._is_fsdp_model(child) for child in wrapped.module.children())


def test_fsdp_wrap_recursively_splits_large_uniform_subtrees_when_over_byte_budget() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _UniformNestedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.inner(inputs)

    with _single_rank_process_group():
        module = _UniformNestedModule()
        original_budget = _MODULE._FSDP_UNIFORM_WRAP_MAX_PARAM_BYTES
        _MODULE._FSDP_UNIFORM_WRAP_MAX_PARAM_BYTES = 1
        try:
            wrapped = _MODULE._fsdp_wrap_uniform_dtype_subtrees(module, device=torch.device("cpu"))
        finally:
            _MODULE._FSDP_UNIFORM_WRAP_MAX_PARAM_BYTES = original_budget

        assert not _MODULE._is_fsdp_model(wrapped)
        nested_wrappers = [m for m in wrapped.modules() if _MODULE._is_fsdp_model(m)]
        assert nested_wrappers, "large uniform subtrees should recurse into smaller FSDP boundaries"


def test_fsdp_sharded_child_modules_include_core_transformer_stacks() -> None:
    trainer = types.SimpleNamespace()
    trainer.semantic_encoder = torch.nn.Linear(4, 4)
    trainer.core = types.SimpleNamespace(
        point_feature_extractor=torch.nn.Linear(4, 4),
        visual_encoder=torch.nn.Linear(4, 4),
        tactile_encoder=torch.nn.Linear(4, 4),
        token_fusion=torch.nn.Linear(4, 4),
        obs_self=torch.nn.Linear(4, 4),
        posterior_self=torch.nn.Linear(4, 4),
        task_self=torch.nn.Linear(4, 4),
        predictive_world=torch.nn.Linear(4, 4),
        predictive_semantic_world=torch.nn.Linear(4, 4),
        control_world=torch.nn.Linear(4, 4),
    )

    children = _MODULE._fsdp_sharded_child_modules(trainer)
    assert children == [
        trainer.semantic_encoder,
        trainer.core.point_feature_extractor,
        trainer.core.visual_encoder,
        trainer.core.tactile_encoder,
        trainer.core.token_fusion,
        trainer.core.obs_self,
        trainer.core.posterior_self,
        trainer.core.task_self,
        trainer.core.predictive_world,
        trainer.core.predictive_semantic_world,
        trainer.core.control_world,
    ]


def test_call_fsdp_method_supports_custom_module_methods_on_wrapped_modules() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _CustomMethodModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.inner(inputs)

    with _single_rank_process_group():
        wrapped = _MODULE.FullyShardedDataParallel(
            _CustomMethodModule(),
            sharding_strategy=_MODULE.ShardingStrategy.FULL_SHARD,
            use_orig_params=False,
            device_id=torch.device("cpu"),
            limit_all_gathers=True,
        )
        result = call_fsdp_method(wrapped, "encode", torch.randn(2, 4))
        assert tuple(result.shape) == (2, 4)


def test_call_fsdp_method_supports_plain_modules_with_nested_fsdp_children() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _InnerCustomMethodModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.inner(inputs)

    class _OuterPlainModule(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner_module = inner

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.inner_module.encode(inputs)

    with _single_rank_process_group():
        wrapped_inner = _MODULE.FullyShardedDataParallel(
            _InnerCustomMethodModule(),
            sharding_strategy=_MODULE.ShardingStrategy.FULL_SHARD,
            use_orig_params=False,
            device_id=torch.device("cpu"),
            limit_all_gathers=True,
        )
        outer = _OuterPlainModule(wrapped_inner)
        result = call_fsdp_method(outer, "encode", torch.randn(2, 4))
        assert tuple(result.shape) == (2, 4)


def test_call_module_forward_or_method_prefers_forward_for_wrapped_modules() -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _ForwardDispatchModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = torch.nn.Linear(4, 4)

        def encode(self, inputs: torch.Tensor) -> torch.Tensor:
            raise AssertionError("explicit custom method should not be used through the FSDP path")

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.inner(inputs)

    with _single_rank_process_group():
        wrapped = _MODULE.FullyShardedDataParallel(
            _ForwardDispatchModule(),
            sharding_strategy=_MODULE.ShardingStrategy.FULL_SHARD,
            use_orig_params=False,
            device_id=torch.device("cpu"),
            limit_all_gathers=True,
        )
        result = call_module_forward_or_method(wrapped, "encode", torch.randn(2, 4))
        assert tuple(result.shape) == (2, 4)


def test_call_module_forward_or_method_strips_dispatch_opcode_for_method_fallback() -> None:
    class _DispatchlessSemanticStub:
        def encode_observation(self, value: torch.Tensor) -> torch.Tensor:
            return value + 1

    result = call_module_forward_or_method(
        _DispatchlessSemanticStub(),
        "encode_observation",
        "encode_observation",
        torch.tensor([1.0]),
    )
    assert torch.equal(result, torch.tensor([2.0]))


def test_optimizer_collection_exposes_unified_optimizer_interface() -> None:
    param_a = torch.nn.Parameter(torch.tensor([1.0]))
    param_b = torch.nn.Parameter(torch.tensor([2.0]))
    opt_a = torch.optim.SGD([{"params": [param_a], "lr": 0.1, "lr_scale": 1.0}], lr=0.1)
    opt_b = torch.optim.SGD([{"params": [param_b], "lr": 0.01, "lr_scale": 0.1}], lr=0.01)
    collection = _MODULE._OptimizerCollection([opt_a, opt_b])

    assert collection.param_groups == opt_a.param_groups + opt_b.param_groups
    _MODULE._set_optimizer_lr(collection, 0.2)
    assert opt_a.param_groups[0]["lr"] == pytest.approx(0.2)
    assert opt_b.param_groups[0]["lr"] == pytest.approx(0.02)

    state = collection.state_dict()
    restored = _MODULE._OptimizerCollection(
        [
            torch.optim.SGD([{"params": [torch.nn.Parameter(torch.tensor([3.0]))], "lr": 1.0, "lr_scale": 1.0}], lr=1.0),
            torch.optim.SGD([{"params": [torch.nn.Parameter(torch.tensor([4.0]))], "lr": 1.0, "lr_scale": 0.1}], lr=1.0),
        ]
    )
    restored.load_state_dict(state)
    assert len(restored.param_groups) == 2
    assert restored.state_dict()["format"] == "picf_optimizer_collection_v1"


def test_consolidate_optimizer_state_for_checkpoint_is_collective_on_zero_style_optimizer() -> None:
    class _FakeShardedOptimizer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int | None]] = []

        def consolidate_state_dict(self, *, to: int) -> None:
            self.calls.append(("consolidate", int(to)))

        def state_dict(self) -> dict[str, object]:
            self.calls.append(("state_dict", None))
            return {}

    rank0 = _FakeShardedOptimizer()
    _MODULE._consolidate_optimizer_state_for_checkpoint(rank0, rank=0)
    assert rank0.calls == [("consolidate", 0), ("state_dict", None)]

    rank1 = _FakeShardedOptimizer()
    _MODULE._consolidate_optimizer_state_for_checkpoint(rank1, rank=1)
    assert rank1.calls == [("consolidate", 0)]


def test_build_optimizer_reports_zero_num_params_for_uninitialized_lazy_modules() -> None:
    core = types.SimpleNamespace(
        point_feature_extractor=torch.nn.LazyLinear(4, bias=False),
        visual_encoder=None,
        tactile_encoder=None,
    )
    trainer = torch.nn.Module()
    trainer.core = core
    trainer.semantic_encoder = None
    args = _base_args()
    args.point_backbone = "rgb"
    optimizer, group_info = _MODULE._build_optimizer(trainer, args=args)
    del optimizer
    point_group = next(item for item in group_info if item["name"] == "point_backbone")
    assert point_group["num_params"] == 0


def test_build_loss_config_reflects_cli_values() -> None:
    args = _base_args()
    args.lambda_action_pos = 3.0
    args.lambda_action_rot = 2.5
    args.lambda_action_gripper = 1.5
    args.lambda_pt = 0.25
    args.pt_bag_radius_m = 0.05
    args.p_align_on = 0.6
    args.p_align_off = 0.3

    cfg = _MODULE._build_loss_config(args)

    assert cfg.lambda_action_pos == pytest.approx(3.0)
    assert cfg.lambda_action_rot == pytest.approx(2.5)
    assert cfg.lambda_action_gripper == pytest.approx(1.5)
    assert cfg.lambda_pt == pytest.approx(0.25)
    assert cfg.pt_bag_radius_m == pytest.approx(0.05)
    assert cfg.p_align_on == pytest.approx(0.6)
    assert cfg.p_align_off == pytest.approx(0.3)


def test_build_model_propagates_final_tactile_runtime_defaults(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)
    args = _base_args()
    args.calvin_root = str(root)
    args.device = "cpu"
    args.point_backbone = "rgb"
    args.crop_radius_m = 0.10
    args.point_focus_sigma_m = 0.03
    args.tactile_contact_tau_on = 0.23
    args.tactile_contact_tau_off = 0.22
    args.tactile_contact_temperature = 0.005
    args.tactile_anchor_prob_on = 0.8

    core, semantic_encoder, use_visual_override = _MODULE._build_model(args, device=torch.device("cpu"))

    assert semantic_encoder is None
    assert use_visual_override is True
    assert core.config.crop_radius_m == pytest.approx(0.10)
    assert core.config.point_focus_sigma_m == pytest.approx(0.03)
    assert core.config.tactile_contact_tau_on == pytest.approx(0.23)
    assert core.config.tactile_contact_tau_off == pytest.approx(0.22)
    assert core.config.tactile_contact_temperature == pytest.approx(0.005)
    assert core.config.tactile_anchor_prob_on == pytest.approx(0.8)


def test_build_model_propagates_v22_conditioned_control_knobs(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)
    args = _base_args()
    _MODULE._normalize_train_args(args)
    args.calvin_root = str(root)
    args.device = "cpu"
    args.point_backbone = "rgb"
    args.task_local_queries = 6
    args.task_global_queries = 2
    args.task_instruction_queries = 3
    args.task_self_layers = 2
    args.conditioned_control_queries = 5
    args.pi_prefix_queries = 3
    args.conditioned_future_queries = 4
    args.task_visual_reread_topk = 11
    args.task_tactile_reread_groups = 1
    args.task_point_reread_topk = 9
    args.require_pi0_action_generator = False

    core, semantic_encoder, use_visual_override = _MODULE._build_model(args, device=torch.device("cpu"))

    assert semantic_encoder is None
    assert use_visual_override is True
    assert core.config.task_local_queries == 6
    assert core.config.task_global_queries == 2
    assert core.config.task_instruction_queries == 3
    assert core.config.task_self_layers == 2
    assert core.config.conditioned_control_queries == 5
    assert core.config.pi_prefix_queries == 3
    assert core.config.conditioned_future_queries == 4
    assert core.config.task_visual_reread_topk == 11
    assert core.config.task_tactile_reread_groups == 1
    assert core.config.task_point_reread_topk == 9
    assert core.config.require_pi0_action_generator is False


def test_build_model_propagates_vl_anchor_router_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)
    monkeypatch.setattr(_MODULE, "PaliGemmaSemanticEncoder", lambda config: torch.nn.Identity())
    args = _base_args()
    args.calvin_root = str(root)
    args.device = "cpu"
    args.point_backbone = "rgb"
    args.semantic_mode = "paligemma"
    args.semantic_source = "hf"
    args.semantic_model_name = "stub-paligemma"
    args.semantic_checkpoint_path = None
    args.vl_anchor_router_enabled = True
    args.vl_grounding_view = "gripper"
    args.vl_anchor_modes = 5
    args.vl_anchor_nms_radius_m = 0.031
    args.vl_anchor_local_sigma_m = 0.047
    args.vl_min_visible_mass = 0.002
    args.vl_heatmap_temperature = 0.75
    args.vl_obs_anchor_gate_init = -3.0
    args.vl_task_point_gate_init = -2.5
    args.vl_posterior_bind_gate_init = -5.5
    args.vl_prior_bias_clip = 2.25
    _MODULE._normalize_train_args(args)

    core, semantic_encoder, use_visual_override = _MODULE._build_model(args, device=torch.device("cpu"))

    assert semantic_encoder is not None
    assert use_visual_override is True
    assert core.config.vl_anchor_router_enabled is True
    assert core.config.vl_grounding_view == "gripper"
    assert core.config.vl_anchor_modes == 5
    assert core.config.vl_anchor_nms_radius_m == pytest.approx(0.031)
    assert core.config.vl_anchor_local_sigma_m == pytest.approx(0.047)
    assert core.config.vl_min_visible_mass == pytest.approx(0.002)
    assert core.config.vl_heatmap_temperature == pytest.approx(0.75)
    assert core.config.vl_obs_anchor_gate_init == pytest.approx(-3.0)
    assert core.config.vl_task_point_gate_init == pytest.approx(-2.5)
    assert core.config.vl_posterior_bind_gate_init == pytest.approx(-5.5)
    assert core.config.vl_prior_bias_clip == pytest.approx(2.25)


def test_validate_train_args_rejects_vl_router_without_paligemma() -> None:
    args = _base_args()
    args.semantic_mode = "zero"
    args.vl_anchor_router_enabled = True
    _MODULE._normalize_train_args(args)

    with pytest.raises(ValueError, match="vl_anchor_router_enabled requires semantic_mode=paligemma"):
        _MODULE._validate_train_args(args)


def test_build_model_ablated_safely_freezes_lazy_core_parameters(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "calvin", make_zip=False)
    args = _base_args()
    args.picf_mode = "ablated"
    _MODULE._normalize_train_args(args)
    args.calvin_root = str(root)
    args.device = "cpu"
    args.point_backbone = "rgb"

    core, semantic_encoder, use_visual_override = _MODULE._build_model(args, device=torch.device("cpu"))

    assert semantic_encoder is None
    assert isinstance(use_visual_override, bool)
    lazy_params = [param for param in core.parameters() if isinstance(param, torch.nn.parameter.UninitializedParameter)]
    assert lazy_params
    assert all(not bool(getattr(param, "requires_grad", False)) for param in lazy_params)


def test_freeze_initialized_module_parameters_handles_uninitialized_lazy_params() -> None:
    class _LazyContainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lazy = torch.nn.LazyLinear(4, bias=False)
            self.ready = torch.nn.Linear(3, 2, bias=False)

    model = _LazyContainer()

    _MODULE._freeze_initialized_module_parameters(model)

    assert model.ready.weight.requires_grad is False
    assert model.lazy.weight.requires_grad is False


def test_collect_nonfinite_gradient_diagnostics_reports_group_and_parameter_name() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=False), torch.nn.Linear(4, 2, bias=False))
    optimizer = torch.optim.AdamW(
        [
            {"name": "first", "params": list(model[0].parameters())},
            {"name": "second", "params": list(model[1].parameters())},
        ],
        lr=1e-3,
    )
    model[1].weight.grad = torch.full_like(model[1].weight, float("nan"))
    diag = _MODULE._collect_nonfinite_gradient_diagnostics(model, optimizer=optimizer, max_items=4)
    assert diag["nonfinite_grad_count"] == 1
    assert diag["samples"][0]["name"] == "1.weight"
    assert diag["samples"][0]["group"] == "second"
    assert diag["samples"][0]["grad_has_nan"] is True


def test_collect_nonfinite_parameter_diagnostics_reports_group_and_parameter_name() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=False), torch.nn.Linear(4, 2, bias=False))
    optimizer = torch.optim.AdamW(
        [
            {"name": "first", "params": list(model[0].parameters())},
            {"name": "second", "params": list(model[1].parameters())},
        ],
        lr=1e-3,
    )
    with torch.no_grad():
        model[0].weight.fill_(float("inf"))
    diag = _MODULE._collect_nonfinite_parameter_diagnostics(model, optimizer=optimizer, max_items=4)
    assert diag["nonfinite_param_count"] == 1
    assert diag["samples"][0]["name"] == "0.weight"
    assert diag["samples"][0]["group"] == "first"
    assert diag["samples"][0]["param_has_inf"] is True


def test_collect_nonfinite_diagnostics_ignore_uninitialized_lazy_parameters() -> None:
    class _LazyContainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lazy = torch.nn.LazyLinear(4, bias=False)
            self.ready = torch.nn.Linear(3, 2, bias=False)

    model = _LazyContainer()
    optimizer = torch.optim.AdamW(
        [
            {"name": "lazy", "params": list(model.lazy.parameters())},
            {"name": "ready", "params": list(model.ready.parameters())},
        ],
        lr=1e-3,
    )
    with torch.no_grad():
        model.ready.weight.fill_(float("inf"))
    model.ready.weight.grad = torch.full_like(model.ready.weight, float("nan"))

    grad_diag = _MODULE._collect_nonfinite_gradient_diagnostics(model, optimizer=optimizer, max_items=4)
    param_diag = _MODULE._collect_nonfinite_parameter_diagnostics(model, optimizer=optimizer, max_items=4)

    assert grad_diag["nonfinite_grad_count"] == 1
    assert grad_diag["samples"][0]["name"] == "ready.weight"
    assert grad_diag["samples"][0]["group"] == "ready"
    assert param_diag["nonfinite_param_count"] == 1
    assert param_diag["samples"][0]["name"] == "ready.weight"
    assert param_diag["samples"][0]["group"] == "ready"


def test_postclip_grad_norm_for_logging_avoids_second_grad_scan() -> None:
    assert _MODULE._postclip_grad_norm_for_logging(
        preclip_grad_norm=3.0,
        grad_clip_threshold=None,
        grad_clip_applied=False,
    ) == pytest.approx(3.0)
    assert _MODULE._postclip_grad_norm_for_logging(
        preclip_grad_norm=3.0,
        grad_clip_threshold=1.25,
        grad_clip_applied=True,
    ) == pytest.approx(1.25)
    assert _MODULE._postclip_grad_norm_for_logging(
        preclip_grad_norm=0.5,
        grad_clip_threshold=1.25,
        grad_clip_applied=True,
    ) == pytest.approx(0.5)


def test_metric_accumulator_reports_tactile_contact_observability() -> None:
    accum = _MODULE._MetricAccumulator(
        tactile_contact_prob_mean=0.6,
        tactile_active_rate=0.25,
        num_windows=2,
    )

    averages = accum.averages()

    assert averages["tactile_contact_prob_mean"] == pytest.approx(0.3)
    assert averages["tactile_active_rate"] == pytest.approx(0.125)


def test_metric_accumulator_update_from_outputs_tracks_semantic_future_aux() -> None:
    accum = _MODULE._MetricAccumulator()
    outputs = {
        "loss_total": torch.tensor(1.0),
        "loss_action": torch.tensor(0.2),
        "loss_action_active7": torch.tensor(0.07),
        "loss_action_pos": torch.tensor(0.05),
        "loss_action_rot": torch.tensor(0.06),
        "loss_action_gripper": torch.tensor(0.09),
        "loss_visual_latent": torch.tensor(0.01),
        "loss_visual_real": torch.tensor(0.02),
        "loss_tactile_real": torch.tensor(0.03),
        "loss_tactile_map": torch.tensor(0.011),
        "loss_tactile_aux": torch.tensor(0.012),
        "loss_point_real": torch.tensor(0.04),
        "loss_semantic_future_aux": torch.tensor(0.17),
        "loss_semantic_group_raw": torch.tensor(0.0425),
        "loss_semantic_group_capped": torch.tensor(0.02),
        "loss_physical_aux": torch.tensor(0.051),
        "loss_physical_aux_capped": torch.tensor(0.03),
        "loss_alignment": torch.tensor(0.05),
        "loss_alignment_raw": torch.tensor(0.055),
        "loss_total_minus_action": torch.tensor(0.8),
        "loss_anchor_pv": torch.tensor(0.06),
        "loss_pv_weak": torch.tensor(0.07),
        "loss_focus_pv": torch.tensor(0.08),
        "loss_pt": torch.tensor(0.09),
        "physical_aux_budget_scale": torch.tensor(0.8),
        "semantic_aux_budget_scale": torch.tensor(0.7),
        "alignment_budget_scale": torch.tensor(0.6),
        "projective_candidate_density": torch.tensor(0.11),
        "tactile_contact_prob_mean": torch.tensor(0.12),
        "tactile_active_rate": torch.tensor(0.13),
    }

    accum.update_from_outputs(outputs)
    averages = accum.averages()

    assert averages["loss_semantic_future_aux"] == pytest.approx(0.17)
    assert averages["loss_action"] == pytest.approx(0.2)
    assert averages["loss_action_active7"] == pytest.approx(0.07)
    assert averages["loss_tactile_map"] == pytest.approx(0.011)
    assert averages["loss_tactile_aux"] == pytest.approx(0.012)
    assert averages["loss_semantic_group_capped"] == pytest.approx(0.02)
    assert averages["loss_physical_aux_capped"] == pytest.approx(0.03)
    assert averages["loss_total_minus_action"] == pytest.approx(0.8)
    assert averages["physical_aux_budget_scale"] == pytest.approx(0.8)
    assert averages["tactile_contact_prob_mean"] == pytest.approx(0.12)


def test_picf_window_trainer_passes_semantic_override_to_core() -> None:
    class _DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.calls: list[torch.Tensor | None] = []
            self.config = types.SimpleNamespace(visual_real_grid=4)

        def step(self, _current, *, previous=None, visual_map_override=None, semantic_override=None, action_future=None):
            del previous, visual_map_override, action_future
            self.calls.append(semantic_override)
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    physical_prediction_cache=types.SimpleNamespace(visual_real=torch.linspace(0.0, 1.0, 48)),
                    prediction_cache=types.SimpleNamespace(visual_real=torch.linspace(1.0, 0.0, 48)),
                )
            )
            return types.SimpleNamespace(
                state=state,
                debug={
                    "projective_candidate_density": 0.0,
                    "tactile_contact_prob_mean": 0.25,
                    "tactile_active_rate": 0.5,
                },
            )

    dummy_losses = types.SimpleNamespace(
        total=torch.tensor(1.0),
        action=torch.tensor(0.1),
        action_active7=torch.tensor(0.04),
        action_pos=torch.tensor(0.03),
        action_rot=torch.tensor(0.04),
        action_gripper=torch.tensor(0.03),
        visual_latent=torch.tensor(0.1),
        visual_real=torch.tensor(0.1),
        tactile_real=torch.tensor(0.1),
        tactile_map=torch.tensor(0.05),
        tactile_aux=torch.tensor(0.05),
        point_real=torch.tensor(0.1),
        semantic_future_aux=torch.tensor(0.1),
        semantic_group_raw=torch.tensor(0.025),
        semantic_group_capped=torch.tensor(0.02),
        physical_aux=torch.tensor(0.15),
        physical_aux_capped=torch.tensor(0.03),
        alignment=torch.tensor(0.1),
        alignment_raw=torch.tensor(0.11),
        total_minus_action=torch.tensor(0.9),
        anchor_pv=torch.tensor(0.1),
        pv_weak=torch.tensor(0.1),
        focus_pv=torch.tensor(0.1),
        pt=torch.tensor(0.1),
        physical_aux_budget_scale=torch.tensor(1.0),
        semantic_aux_budget_scale=torch.tensor(1.0),
        alignment_budget_scale=torch.tensor(1.0),
    )

    class _SemanticStub(torch.nn.Module):
        def encode_observation(self, observation):
            return torch.full((1, 4), float(observation.step_id))

    trainer = _MODULE._PicfWindowTrainer(
        _DummyCore(),
        semantic_encoder=_SemanticStub(),
        visual_grid=8,
        use_visual_override=False,
    )
    frame0 = PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.zeros((8, 8), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=1,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        action=np.zeros((7,), dtype=np.float32),
    )
    frame1 = dataclasses.replace(frame0, step_id=2, reset_scaffold=False)
    window = _MODULE._TransitionWindow(segment_id=0, start_step_id=0, prompt="test", frames=(frame0, frame1))
    original_loss = _MODULE.compute_transition_loss
    try:
        _MODULE.compute_transition_loss = lambda *args, **kwargs: dummy_losses
        result = trainer(window, capture_visual_diagnostics=True)
    finally:
        _MODULE.compute_transition_loss = original_loss
    assert trainer.core.calls
    torch.testing.assert_close(trainer.core.calls[0], torch.full((1, 4), 1.0))
    assert len(result["diagnostic_physical_visual_real_seq"]) == 1
    assert len(result["diagnostic_semantic_visual_real_seq"]) == 1
    assert result["tactile_contact_prob_mean"].item() == pytest.approx(0.25)
    assert result["tactile_active_rate"].item() == pytest.approx(0.5)


def test_picf_window_trainer_reuses_middle_frame_targets_with_detached_override() -> None:
    class _DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.config = types.SimpleNamespace(visual_real_grid=4)

    def _dummy_output() -> types.SimpleNamespace:
        state = types.SimpleNamespace(
            predictive=types.SimpleNamespace(
                physical_prediction_cache=types.SimpleNamespace(visual_real=torch.linspace(0.0, 1.0, 48)),
                prediction_cache=types.SimpleNamespace(visual_real=torch.linspace(1.0, 0.0, 48)),
            )
        )
        return types.SimpleNamespace(
            state=state,
            debug={
                "projective_candidate_density": 0.125,
                "tactile_contact_prob_mean": 0.25,
                "tactile_active_rate": 0.5,
            },
        )

    class _DummyPolicy:
        def forward_train_transition(self, *, current, previous=None, visual_map_override=None, action_chunk_target=None):
            del previous, visual_map_override, action_chunk_target
            observed = types.SimpleNamespace(
                current_targets={
                    "visual_latent": torch.tensor([float(current.step_id)], requires_grad=True),
                    "visual_real": torch.tensor([float(current.step_id) + 1.0], requires_grad=True),
                    "tactile_real": torch.tensor([float(current.step_id) + 2.0], requires_grad=True),
                    "point_real": torch.tensor([float(current.step_id) + 3.0], requires_grad=True),
                },
                availability=torch.ones((4,), dtype=torch.float32, requires_grad=True),
            )
            return types.SimpleNamespace(
                output=_dummy_output(),
                observed=observed,
                flow_override=None,
                next_state=f"state-{current.step_id}",
            )

    dummy_losses = types.SimpleNamespace(
        total=torch.tensor(1.0),
        action=torch.tensor(0.1),
        action_active7=torch.tensor(0.04),
        action_pos=torch.tensor(0.03),
        action_rot=torch.tensor(0.04),
        action_gripper=torch.tensor(0.03),
        visual_latent=torch.tensor(0.1),
        visual_real=torch.tensor(0.1),
        tactile_real=torch.tensor(0.1),
        tactile_map=torch.tensor(0.05),
        tactile_aux=torch.tensor(0.05),
        point_real=torch.tensor(0.1),
        semantic_future_aux=torch.tensor(0.1),
        semantic_group_raw=torch.tensor(0.025),
        semantic_group_capped=torch.tensor(0.02),
        physical_aux=torch.tensor(0.15),
        physical_aux_capped=torch.tensor(0.03),
        alignment=torch.tensor(0.1),
        alignment_raw=torch.tensor(0.11),
        total_minus_action=torch.tensor(0.9),
        anchor_pv=torch.tensor(0.1),
        pv_weak=torch.tensor(0.1),
        focus_pv=torch.tensor(0.1),
        pt=torch.tensor(0.1),
        physical_aux_budget_scale=torch.tensor(1.0),
        semantic_aux_budget_scale=torch.tensor(1.0),
        alignment_budget_scale=torch.tensor(1.0),
    )

    trainer = _MODULE._PicfWindowTrainer(
        _DummyCore(),
        semantic_encoder=None,
        visual_grid=8,
        use_visual_override=False,
    )
    trainer.policy = _DummyPolicy()

    frame0 = PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.zeros((8, 8), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=1,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        action=np.zeros((7,), dtype=np.float32),
    )
    frame1 = dataclasses.replace(frame0, step_id=2, reset_scaffold=False)
    frame2 = dataclasses.replace(frame0, step_id=3, reset_scaffold=False)
    window = _MODULE._TransitionWindow(segment_id=0, start_step_id=0, prompt="test", frames=(frame0, frame1, frame2))

    captured: list[tuple[int, bool, float | None, bool | None]] = []
    original_loss = _MODULE.compute_transition_loss
    try:
        def _fake_compute_transition_loss(core, output_t, next_observation, **kwargs):
            del core, output_t
            override = kwargs.get("future_targets_override")
            captured.append(
                (
                    int(next_observation.step_id),
                    override is not None,
                    None if override is None or override.visual_latent is None else float(override.visual_latent.item()),
                    None if override is None or override.visual_latent is None else bool(override.visual_latent.requires_grad),
                )
            )
            return dummy_losses

        _MODULE.compute_transition_loss = _fake_compute_transition_loss
        trainer(window)
    finally:
        _MODULE.compute_transition_loss = original_loss

    assert captured == [
        (2, True, 2.0, False),
        (3, False, None, None),
    ]


def test_picf_window_trainer_state_only_burnin_skips_policy_flow_until_suffix() -> None:
    class _DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.config = types.SimpleNamespace(visual_real_grid=4)

    def _dummy_output() -> types.SimpleNamespace:
        return types.SimpleNamespace(
            state=types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    physical_prediction_cache=types.SimpleNamespace(visual_real=None),
                    prediction_cache=types.SimpleNamespace(visual_real=None),
                )
            ),
            debug={
                "projective_candidate_density": 0.125,
                "tactile_contact_prob_mean": 0.25,
                "tactile_active_rate": 0.5,
            },
        )

    class _DummyPolicy:
        picf_enabled = True

        def __init__(self) -> None:
            self.burnin_calls: list[tuple[int, object]] = []
            self.train_calls: list[tuple[int, object]] = []

        def burnin_recurrent_transition(self, *, current, previous=None, visual_map_override=None):
            del visual_map_override
            self.burnin_calls.append((int(current.step_id), previous))
            return f"burnin-state-{current.step_id}"

        def forward_train_transition(self, *, current, previous=None, visual_map_override=None, action_chunk_target=None):
            del visual_map_override, action_chunk_target
            self.train_calls.append((int(current.step_id), previous))
            return types.SimpleNamespace(
                output=_dummy_output(),
                observed=None,
                flow_override=None,
                next_state=f"train-state-{current.step_id}",
            )

    dummy_losses = types.SimpleNamespace(
        total=torch.tensor(1.0),
        action=torch.tensor(0.1),
        action_active7=torch.tensor(0.04),
        action_pos=torch.tensor(0.03),
        action_rot=torch.tensor(0.04),
        action_gripper=torch.tensor(0.03),
        visual_latent=torch.tensor(0.1),
        visual_real=torch.tensor(0.1),
        tactile_real=torch.tensor(0.1),
        tactile_map=torch.tensor(0.05),
        tactile_aux=torch.tensor(0.05),
        point_real=torch.tensor(0.1),
        semantic_future_aux=torch.tensor(0.1),
        semantic_group_raw=torch.tensor(0.025),
        semantic_group_capped=torch.tensor(0.02),
        physical_aux=torch.tensor(0.15),
        physical_aux_capped=torch.tensor(0.03),
        alignment=torch.tensor(0.1),
        alignment_raw=torch.tensor(0.11),
        total_minus_action=torch.tensor(0.9),
        anchor_pv=torch.tensor(0.1),
        pv_weak=torch.tensor(0.1),
        focus_pv=torch.tensor(0.1),
        pt=torch.tensor(0.1),
        physical_aux_budget_scale=torch.tensor(1.0),
        semantic_aux_budget_scale=torch.tensor(1.0),
        alignment_budget_scale=torch.tensor(1.0),
    )

    trainer = _MODULE._PicfWindowTrainer(
        _DummyCore(),
        semantic_encoder=None,
        visual_grid=8,
        use_visual_override=False,
        burnin_steps=1,
        burnin_mode="state_only",
    )
    policy = _DummyPolicy()
    trainer.policy = policy

    frame0 = PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.zeros((8, 8), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=1,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        action=np.zeros((7,), dtype=np.float32),
    )
    frame1 = dataclasses.replace(frame0, step_id=2, reset_scaffold=False)
    frame2 = dataclasses.replace(frame0, step_id=3, reset_scaffold=False)
    window = _MODULE._TransitionWindow(segment_id=0, start_step_id=0, prompt="test", frames=(frame0, frame1, frame2))

    original_loss = _MODULE.compute_transition_loss
    try:
        _MODULE.compute_transition_loss = lambda *args, **kwargs: dummy_losses
        result = trainer(window)
    finally:
        _MODULE.compute_transition_loss = original_loss

    assert policy.burnin_calls == [(1, None)]
    assert policy.train_calls == [(2, "burnin-state-1")]
    assert result["loss_total"].item() == pytest.approx(1.0)
    assert result["loss_action"].item() == pytest.approx(0.1)


def test_picf_window_trainer_ablated_uses_action_only_metrics() -> None:
    class _DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.config = types.SimpleNamespace(visual_real_grid=4)

    class _DummyPolicy:
        picf_enabled = False

        def forward_train_transition(self, *, current, previous=None, visual_map_override=None, action_chunk_target=None):
            del previous, visual_map_override
            target = torch.as_tensor(action_chunk_target, dtype=torch.float32)
            return types.SimpleNamespace(
                output=None,
                observed=None,
                flow_override={
                    "total": torch.tensor(0.25),
                    "action_pos": torch.tensor(0.10),
                    "action_rot": torch.tensor(0.10),
                    "action_gripper": torch.tensor(0.05),
                    "predicted_action": target[0] if target.ndim > 1 else target,
                    "predicted_chunk": target,
                },
                next_state=None,
            )

    trainer = _MODULE._PicfWindowTrainer(
        _DummyCore(),
        semantic_encoder=None,
        visual_grid=8,
        use_visual_override=False,
        picf_mode="ablated",
    )
    trainer.policy = _DummyPolicy()

    frame0 = PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.zeros((8, 8), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=1,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        action=np.zeros((7,), dtype=np.float32),
        action_chunk=np.zeros((2, 7), dtype=np.float32),
    )
    frame1 = dataclasses.replace(frame0, step_id=2, reset_scaffold=False)
    frame2 = dataclasses.replace(frame0, step_id=3, reset_scaffold=False)
    window = _MODULE._TransitionWindow(segment_id=0, start_step_id=0, prompt="test", frames=(frame0, frame1, frame2))

    result = trainer(window, capture_visual_diagnostics=True)

    assert result["loss_total"].item() == pytest.approx(0.25)
    assert result["loss_action"].item() == pytest.approx(0.25)
    assert result["loss_total_minus_action"].item() == pytest.approx(0.0)
    assert result["loss_semantic_future_aux"].item() == pytest.approx(0.0)
    assert result["loss_alignment"].item() == pytest.approx(0.0)
    assert result["projective_candidate_density"].item() == pytest.approx(0.0)
    assert result["diagnostic_physical_visual_real_seq"] == []
    assert result["diagnostic_semantic_visual_real_seq"] == []


def test_decode_visual_real_prediction_upsamples_grid() -> None:
    flat = torch.linspace(0.0, 1.0, 48)
    image = _MODULE._decode_visual_real_prediction(flat, grid=4, upscale=8)
    assert image is not None
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8


def test_save_visual_diagnostics_writes_png_gif_and_metadata(tmp_path: Path) -> None:
    frame0 = PicfObservation(
        rgb_static=np.full((16, 16, 3), 32, dtype=np.uint8),
        depth_static=np.zeros((16, 16), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="stack blocks",
        step_id=10,
        segment_id=3,
        timestamp_s=0.0,
        reset_scaffold=True,
        action=np.zeros((7,), dtype=np.float32),
    )
    frame1 = dataclasses.replace(frame0, step_id=11, reset_scaffold=False, rgb_static=np.full((16, 16, 3), 96, dtype=np.uint8))
    frame2 = dataclasses.replace(frame0, step_id=12, reset_scaffold=False, rgb_static=np.full((16, 16, 3), 160, dtype=np.uint8))
    window = _MODULE._TransitionWindow(segment_id=3, start_step_id=10, prompt="stack blocks", frames=(frame0, frame1, frame2))

    physical = [torch.linspace(0.0, 1.0, 48), torch.linspace(1.0, 0.0, 48)]
    semantic = [torch.linspace(0.25, 0.75, 48), torch.linspace(0.75, 0.25, 48)]

    _MODULE._save_visual_diagnostics(
        output_dir=tmp_path,
        step=500,
        window=window,
        physical_visual_real_seq=physical,
        semantic_visual_real_seq=semantic,
        visual_real_grid=4,
        visual_real_upscale=8,
    )

    diag_dir = tmp_path / "diagnostics" / "000500"
    assert (diag_dir / "gt_window_static.gif").is_file()
    assert (diag_dir / "pred_physical_window_static.gif").is_file()
    assert (diag_dir / "pred_semantic_window_static.gif").is_file()
    assert (diag_dir / "compare_grid.png").is_file()
    assert (diag_dir / "gt_static_t0.png").is_file()
    assert (diag_dir / "gt_static_t1.png").is_file()
    assert (diag_dir / "pred_physical_t1.png").is_file()
    assert (diag_dir / "pred_semantic_t2.png").is_file()
    metadata = (diag_dir / "metadata.json").read_text(encoding="utf-8")
    assert "coarse 4x4 RGB reconstructions" in metadata
    assert "stack blocks" in metadata


def test_first_step_window_precheck_rejects_empty_local_support() -> None:
    class _DummyLocalFrame:
        def make_transform(self, _robot_obs):
            return np.eye(4, dtype=np.float32)

    class _DummyCore:
        def __init__(self) -> None:
            self.local_frame = _DummyLocalFrame()
            self.config = types.SimpleNamespace(crop_radius_m=0.08)

        def pointcloud_builder(self, _payload):
            raise AssertionError("pointcloud_builder should not run when point_set is pre-populated")

        def _build_runtime_meta(self, _obs, _previous):
            return types.SimpleNamespace(point_contract_ok=True)

        def _point_subset(self, _obs):
            return types.SimpleNamespace(points_local=np.zeros((0, 3), dtype=np.float32))

    trainer = types.SimpleNamespace(core=_DummyCore())
    obs = PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.zeros((8, 8), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=0,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        point_set=PicfPointCloudFrame(
            grid_coord=np.zeros((0, 3), dtype=np.int32),
            xyz_world=np.zeros((0, 3), dtype=np.float32),
            rgb=np.zeros((0, 3), dtype=np.float32),
            normal_world=np.zeros((0, 3), dtype=np.float32),
            valid_point_mask=np.zeros((0,), dtype=bool),
            frame_valid=True,
        ),
        G_t=np.eye(4, dtype=np.float32),
    )
    window = _MODULE._TransitionWindow(segment_id=0, start_step_id=0, prompt="test", frames=(obs, obs))
    with pytest.raises(RuntimeError, match="non-empty local xyzrgb support"):
        _MODULE._ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)


def test_later_step_window_precheck_rejects_empty_local_support() -> None:
    class _DummyLocalFrame:
        def make_transform(self, _robot_obs):
            return np.eye(4, dtype=np.float32)

    class _DummyCore:
        def __init__(self) -> None:
            self.local_frame = _DummyLocalFrame()
            self.config = types.SimpleNamespace(crop_radius_m=0.08)

        def pointcloud_builder(self, _payload):
            raise AssertionError("pointcloud_builder should not run when point_set is pre-populated")

        def _build_runtime_meta(self, _obs, _previous):
            return types.SimpleNamespace(point_contract_ok=True)

        def _point_subset(self, obs):
            count = int(obs.step_id)
            return types.SimpleNamespace(points_local=np.zeros((count, 3), dtype=np.float32))

    trainer = types.SimpleNamespace(core=_DummyCore())
    obs0 = PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.zeros((8, 8), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=1,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        point_set=PicfPointCloudFrame(
            grid_coord=np.zeros((1, 3), dtype=np.int32),
            xyz_world=np.zeros((1, 3), dtype=np.float32),
            rgb=np.zeros((1, 3), dtype=np.float32),
            normal_world=np.zeros((1, 3), dtype=np.float32),
            valid_point_mask=np.ones((1,), dtype=bool),
            frame_valid=True,
        ),
        G_t=np.eye(4, dtype=np.float32),
    )
    obs1 = dataclasses.replace(obs0, step_id=0, reset_scaffold=False)
    window = _MODULE._TransitionWindow(segment_id=0, start_step_id=0, prompt="test", frames=(obs0, obs1))
    with pytest.raises(RuntimeError, match="window step 1"):
        _MODULE._ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)


def test_first_step_window_precheck_uses_full_pointcloud_payload() -> None:
    class _DummyLocalFrame:
        def make_transform(self, _robot_obs):
            return np.eye(4, dtype=np.float32)

    class _DummyCore:
        def __init__(self) -> None:
            self.local_frame = _DummyLocalFrame()
            self.config = types.SimpleNamespace(crop_radius_m=0.10)
            self.payloads: list[dict[str, np.ndarray | float | None]] = []

        def pointcloud_builder(self, payload):
            self.payloads.append(payload)
            return PicfPointCloudFrame(
                grid_coord=np.zeros((1, 3), dtype=np.int32),
                xyz_world=np.zeros((1, 3), dtype=np.float32),
                rgb=np.zeros((1, 3), dtype=np.float32),
                normal_world=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (1, 1)),
                valid_point_mask=np.ones((1,), dtype=bool),
                frame_valid=True,
            )

        def _build_runtime_meta(self, _obs, _previous):
            return types.SimpleNamespace(point_contract_ok=True)

        def _point_subset(self, _obs):
            return types.SimpleNamespace(points_local=np.zeros((1, 3), dtype=np.float32))

    tactile = PicfTactilePacket(
        sensors=(
            TactileSensorFrame(
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                sensor_name="left",
                T_sens_to_wrist=np.asarray(
                    [[1.0, 0.0, 0.0, 0.10], [0.0, 1.0, 0.0, 0.00], [0.0, 0.0, 1.0, 0.00], [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                timestamp_s=0.0,
            ),
            TactileSensorFrame(
                rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                sensor_name="right",
                T_sens_to_wrist=np.asarray(
                    [[1.0, 0.0, 0.0, -0.10], [0.0, 1.0, 0.0, 0.00], [0.0, 0.0, 1.0, 0.00], [0.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                timestamp_s=0.0,
            ),
        )
    )
    base = PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.ones((8, 8), dtype=np.float32),
        rgb_gripper=np.full((6, 6, 3), 127, dtype=np.uint8),
        depth_gripper=np.full((6, 6), 0.25, dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=0,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        tactile=tactile,
        G_t=np.eye(4, dtype=np.float32),
    )
    window = _MODULE._TransitionWindow(
        segment_id=0,
        start_step_id=0,
        prompt="test",
        frames=(base, dataclasses.replace(base, step_id=1, reset_scaffold=False)),
    )
    trainer = types.SimpleNamespace(core=_DummyCore())

    point_counts = _MODULE._ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)

    assert point_counts == (1, 1)
    assert len(trainer.core.payloads) == 2
    for payload in trainer.core.payloads:
        assert payload["rgb_gripper"] is not None
        assert payload["depth_gripper"] is not None
        assert payload["robot_obs"] is not None
        np.testing.assert_allclose(
            np.asarray(payload["focus_centers_world"], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]], dtype=np.float32),
        )
        assert float(payload["focus_radius_m"]) == pytest.approx(0.10)


def test_checkpoint_roundtrip_preserves_trainable_semantic_state(tmp_path: Path) -> None:
    class _DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)

    trainer = torch.nn.Module()
    trainer.core = _DummyCore()
    trainer.semantic_encoder = torch.nn.Linear(4, 2, bias=False)

    with torch.no_grad():
        trainer.core.proj.weight.fill_(0.5)
        trainer.semantic_encoder.weight.fill_(1.5)

    optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-3)
    args = argparse.Namespace(dummy=True)
    clip_args = _base_args()
    clip_args.grad_clip_mode = "percentile"
    clip_args.grad_clip_percentile = 75.0
    clip_args.grad_clip_window = 4
    _MODULE._normalize_train_args(clip_args)
    grad_clip_controller = _MODULE._GradClipController.from_args(clip_args)
    for value in (1.0, 2.0, 3.0, 4.0):
        grad_clip_controller.observe(value)
    output_dir = tmp_path / "picf_ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)

    _MODULE._save_checkpoint(
        output_dir=output_dir,
        model=trainer,
        optimizer=optimizer,
        step=7,
        args=args,
        grad_clip_controller=grad_clip_controller,
    )

    reloaded = torch.nn.Module()
    reloaded.core = _DummyCore()
    reloaded.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        reloaded.core.proj.weight.zero_()
        reloaded.semantic_encoder.weight.zero_()
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    reloaded_controller = _MODULE._GradClipController.from_args(clip_args)

    step = _MODULE._load_checkpoint(
        path=output_dir / "7",
        model=reloaded,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
        grad_clip_controller=reloaded_controller,
    )

    assert step == 7
    torch.testing.assert_close(reloaded.core.proj.weight, trainer.core.proj.weight)
    torch.testing.assert_close(reloaded.semantic_encoder.weight, trainer.semantic_encoder.weight)
    assert list(reloaded_controller.history) == [1.0, 2.0, 3.0, 4.0]


def test_model_only_checkpoint_roundtrip_reinitializes_optimizer(tmp_path: Path) -> None:
    class _DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)

    trainer = torch.nn.Module()
    trainer.core = _DummyCore()
    trainer.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        trainer.core.proj.weight.fill_(0.75)
        trainer.semantic_encoder.weight.fill_(1.25)

    optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-3)
    args = argparse.Namespace(optimizer_checkpoint_mode="model_only")
    output_dir = tmp_path / "picf_model_only_ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)

    _MODULE._save_checkpoint(
        output_dir=output_dir,
        model=trainer,
        optimizer=optimizer,
        step=11,
        args=args,
        save_optimizer_state=False,
    )

    ckpt_dir = output_dir / "11"
    assert (ckpt_dir / "model.pt").exists()
    assert not (ckpt_dir / "optimizer.pt").exists()
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location="cpu", weights_only=False)
    assert metadata["optimizer_state_saved"] is False

    reloaded = torch.nn.Module()
    reloaded.core = _DummyCore()
    reloaded.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    step = _MODULE._load_checkpoint(
        path=ckpt_dir,
        model=reloaded,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )

    assert step == 11
    torch.testing.assert_close(reloaded.core.proj.weight, trainer.core.proj.weight)
    torch.testing.assert_close(reloaded.semantic_encoder.weight, trainer.semantic_encoder.weight)


def test_checkpoint_loader_accepts_legacy_core_only_state(tmp_path: Path) -> None:
    class _DummyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)

    trainer = torch.nn.Module()
    trainer.core = _DummyCore()
    trainer.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-3)

    legacy_dir = tmp_path / "legacy" / "9"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        trainer.core.proj.weight.fill_(2.0)
    torch.save(trainer.core.state_dict(), legacy_dir / "model.pt")
    torch.save(optimizer.state_dict(), legacy_dir / "optimizer.pt")
    torch.save({"step": 9, "checkpoint_format": "legacy_core_only"}, legacy_dir / "metadata.pt")

    reloaded = torch.nn.Module()
    reloaded.core = _DummyCore()
    reloaded.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    step = _MODULE._load_checkpoint(
        path=legacy_dir,
        model=reloaded,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )

    assert step == 9
    torch.testing.assert_close(reloaded.core.proj.weight, trainer.core.proj.weight)


def test_checkpoint_save_and_load_supports_ablated_semantic_only_lazy_core(tmp_path: Path) -> None:
    class _LazyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lazy = torch.nn.LazyLinear(4, bias=False)

    trainer = torch.nn.Module()
    trainer.core = _LazyCore()
    _MODULE._freeze_initialized_module_parameters(trainer.core)
    trainer.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        trainer.semantic_encoder.weight.fill_(1.5)
    optimizer = torch.optim.AdamW(trainer.semantic_encoder.parameters(), lr=1e-3)

    args = argparse.Namespace(picf_mode="ablated", optimizer_checkpoint_mode="auto")
    ckpt_root = tmp_path / "ablated_semantic_only"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    _MODULE._save_checkpoint(
        output_dir=ckpt_root,
        model=trainer,
        optimizer=optimizer,
        step=7,
        args=args,
        save_optimizer_state=False,
    )

    ckpt_dir = ckpt_root / "7"
    model_state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    assert model_state["checkpoint_model_format"] == "picf_ablated_semantic_only_v1"
    assert list(model_state["semantic_encoder"]) == ["weight"]

    reloaded = torch.nn.Module()
    reloaded.core = _LazyCore()
    _MODULE._freeze_initialized_module_parameters(reloaded.core)
    reloaded.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        reloaded.semantic_encoder.weight.zero_()
    reloaded_optimizer = torch.optim.AdamW(reloaded.semantic_encoder.parameters(), lr=1e-3)

    step = _MODULE._load_checkpoint(
        path=ckpt_dir,
        model=reloaded,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )

    assert step == 7
    torch.testing.assert_close(reloaded.semantic_encoder.weight, trainer.semantic_encoder.weight)


def test_should_save_optimizer_state_defaults_to_model_only_for_ablated_auto() -> None:
    args = _base_args()
    args.picf_mode = "ablated"
    args.optimizer_checkpoint_mode = "auto"

    assert _MODULE._should_save_optimizer_state(args=args) is False


def test_should_save_optimizer_state_defaults_to_full_state_for_enabled_auto() -> None:
    args = _base_args()
    args.picf_mode = "enabled"
    args.optimizer_checkpoint_mode = "auto"
    args.optimizer_sharding = "none"

    assert _MODULE._should_save_optimizer_state(args=args) is True


def test_enabled_auto_checkpoint_saves_full_model_and_optimizer_state(tmp_path: Path) -> None:
    class _Core(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)

    trainer = torch.nn.Module()
    trainer.core = _Core()
    trainer.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-3)

    args = _base_args()
    args.picf_mode = "enabled"
    args.optimizer_checkpoint_mode = "auto"
    args.optimizer_sharding = "none"

    ckpt_root = tmp_path / "enabled_auto_full_state"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    _MODULE._save_checkpoint(
        output_dir=ckpt_root,
        model=trainer,
        optimizer=optimizer,
        step=11,
        args=args,
        save_optimizer_state=_MODULE._should_save_optimizer_state(args=args),
    )

    ckpt_dir = ckpt_root / "11"
    model_state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location="cpu", weights_only=False)

    assert (ckpt_dir / "optimizer.pt").exists()
    assert "checkpoint_model_format" not in model_state
    assert "core.proj.weight" in model_state
    assert "semantic_encoder.weight" in model_state
    assert metadata["optimizer_state_saved"] is True


def test_fsdp_checkpoint_roundtrip_supports_ablated_semantic_only_lazy_core(tmp_path: Path) -> None:
    if _MODULE.FullyShardedDataParallel is None:
        pytest.skip("FSDP is not available in this torch build.")

    class _LazyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.point_feature_extractor = torch.nn.LazyLinear(4, bias=False)
            self.visual_encoder = torch.nn.LazyLinear(4, bias=False)
            self.tactile_encoder = torch.nn.LazyLinear(4, bias=False)

    class _AblatedTrainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = _LazyCore()
            _MODULE._freeze_initialized_module_parameters(self.core)
            self.semantic_encoder = torch.nn.Linear(4, 4, bias=False)
            self.policy = types.SimpleNamespace(semantic_encoder=self.semantic_encoder)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.semantic_encoder(inputs).sum()

    args = _base_args()
    args.training_strategy = "fsdp_full_shard"
    args.optimizer_sharding = "none"
    args.picf_mode = "ablated"

    with _single_rank_process_group():
        trainer = _AblatedTrainer()
        wrapped = _MODULE._wrap_model_for_training_strategy(trainer, args=args, device=torch.device("cpu"))
        optimizer, _ = _MODULE._build_optimizer(_MODULE._unwrap_training_model(wrapped), args=args)
        loss = wrapped(torch.randn(2, 4))
        loss.backward()
        optimizer.step()

        output_dir = tmp_path / "fsdp_ablated_ckpt"
        output_dir.mkdir(parents=True, exist_ok=True)
        _MODULE._save_checkpoint(
            output_dir=output_dir,
            model=wrapped,
            optimizer=optimizer,
            step=5,
            args=args,
            rank=0,
            device=torch.device("cpu"),
        )

        model_payload = torch.load(output_dir / "5" / "model.pt", map_location="cpu", weights_only=False)
        assert model_payload["checkpoint_model_format"] == "picf_ablated_semantic_only_v1"
        assert list(model_payload["semantic_encoder"]) == ["weight"]

        reloaded = _AblatedTrainer()
        wrapped_reloaded = _MODULE._wrap_model_for_training_strategy(
            reloaded,
            args=args,
            device=torch.device("cpu"),
        )
        reloaded_optimizer, _ = _MODULE._build_optimizer(_MODULE._unwrap_training_model(wrapped_reloaded), args=args)
        step = _MODULE._load_checkpoint(
            path=output_dir / "5",
            model=wrapped_reloaded,
            optimizer=reloaded_optimizer,
            device=torch.device("cpu"),
        )

        assert step == 5
        forward_value = wrapped_reloaded(torch.randn(2, 4))
        assert torch.isfinite(forward_value).item()


def test_ablated_auto_checkpoint_skips_optimizer_state(tmp_path: Path) -> None:
    class _LazyCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lazy = torch.nn.LazyLinear(4, bias=False)

    trainer = torch.nn.Module()
    trainer.core = _LazyCore()
    _MODULE._freeze_initialized_module_parameters(trainer.core)
    trainer.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    optimizer = torch.optim.AdamW(trainer.semantic_encoder.parameters(), lr=1e-3)

    args = argparse.Namespace(picf_mode="ablated", optimizer_checkpoint_mode="auto")
    ckpt_root = tmp_path / "ablated_auto_model_only"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    _MODULE._save_checkpoint(
        output_dir=ckpt_root,
        model=trainer,
        optimizer=optimizer,
        step=3,
        args=args,
        save_optimizer_state=_MODULE._should_save_optimizer_state(args=args),
    )

    ckpt_dir = ckpt_root / "3"
    assert not (ckpt_dir / "optimizer.pt").exists()
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location="cpu", weights_only=False)
    assert metadata["optimizer_state_saved"] is False


def test_load_state_dict_picf_compat_skips_shape_mismatches_and_keeps_matching_weights() -> None:
    class _OldCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)

    class _NewCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 3, bias=False)

    old = torch.nn.Module()
    old.core = _OldCore()
    old.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        old.core.proj.weight.fill_(1.25)
        old.semantic_encoder.weight.fill_(2.5)

    new = torch.nn.Module()
    new.core = _NewCore()
    new.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        new.core.proj.weight.zero_()
        new.semantic_encoder.weight.zero_()

    missing, unexpected, shape_mismatches = _MODULE._load_state_dict_picf_compat(new, old.state_dict())

    assert "core.proj.weight" in missing
    assert unexpected == []
    assert any(item.startswith("core.proj.weight: checkpoint_shape=(2, 3) model_shape=(3, 3)") for item in shape_mismatches)
    torch.testing.assert_close(new.semantic_encoder.weight, old.semantic_encoder.weight)
    torch.testing.assert_close(new.core.proj.weight, torch.zeros_like(new.core.proj.weight))


def test_checkpoint_loader_accepts_shape_mismatched_trainer_state_with_optimizer_reset(tmp_path: Path) -> None:
    class _OldCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 2, bias=False)

    class _NewCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 3, bias=False)

    trainer = torch.nn.Module()
    trainer.core = _OldCore()
    trainer.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        trainer.core.proj.weight.fill_(0.75)
        trainer.semantic_encoder.weight.fill_(1.75)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-3)

    ckpt_dir = tmp_path / "shape_mismatch" / "11"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save({"step": 11, "checkpoint_format": "picf_trainer_v2"}, ckpt_dir / "metadata.pt")

    reloaded = torch.nn.Module()
    reloaded.core = _NewCore()
    reloaded.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        reloaded.core.proj.weight.zero_()
        reloaded.semantic_encoder.weight.zero_()
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)

    step = _MODULE._load_checkpoint(
        path=ckpt_dir,
        model=reloaded,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )

    assert step == 11
    torch.testing.assert_close(reloaded.semantic_encoder.weight, trainer.semantic_encoder.weight)
    torch.testing.assert_close(reloaded.core.proj.weight, torch.zeros_like(reloaded.core.proj.weight))


def test_load_checkpoint_sequential_across_ranks_serializes_resume_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_load_checkpoint(*, path, model, optimizer, device, grad_clip_controller):
        del path, model, optimizer, device, grad_clip_controller
        calls.append(("load", 0))
        return 123

    def fake_barrier(*, use_ddp, device):
        del device
        calls.append(("barrier", int(use_ddp)))

    def fake_broadcast(tensor, src):
        calls.append(("broadcast", int(src)))

    monkeypatch.setattr(_MODULE, "_load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(_MODULE, "_distributed_barrier", fake_barrier)
    monkeypatch.setattr(_MODULE.dist, "broadcast", fake_broadcast)

    step = _MODULE._load_checkpoint_sequential_across_ranks(
        path=Path("/tmp/fake"),
        model=object(),
        optimizer=object(),
        device=torch.device("cpu"),
        rank=0,
        world_size=2,
    )

    assert step == 123
    assert calls == [
        ("load", 0),
        ("barrier", 1),
        ("barrier", 1),
        ("broadcast", 0),
    ]


def test_load_checkpoint_sequential_across_ranks_only_loads_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[int] = []
    barriers: list[int] = []

    def fake_load_checkpoint(*, path, model, optimizer, device, grad_clip_controller):
        del path, model, optimizer, device, grad_clip_controller
        loaded.append(1)
        return 456

    def fake_barrier(*, use_ddp, device):
        del device
        barriers.append(int(use_ddp))

    def fake_broadcast(tensor, src):
        del src

    monkeypatch.setattr(_MODULE, "_load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(_MODULE, "_distributed_barrier", fake_barrier)
    monkeypatch.setattr(_MODULE.dist, "broadcast", fake_broadcast)

    step = _MODULE._load_checkpoint_sequential_across_ranks(
        path=Path("/tmp/fake"),
        model=object(),
        optimizer=object(),
        device=torch.device("cpu"),
        rank=1,
        world_size=2,
    )

    assert step == 456
    assert loaded == [1]
    assert barriers == [1, 1]


def test_build_model_sequential_across_ranks_now_builds_in_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_build_model(args, *, device):
        del args, device
        calls.append(("build", 0))
        return object(), object(), True

    def fake_barrier(*, use_ddp, device):
        del device
        calls.append(("barrier", int(use_ddp)))

    monkeypatch.setattr(_MODULE, "_build_model", fake_build_model)
    monkeypatch.setattr(_MODULE, "_distributed_barrier", fake_barrier)

    core, semantic_encoder, use_visual_override = _MODULE._build_model_sequential_across_ranks(
        _base_args(),
        device=torch.device("cpu"),
        rank=0,
        world_size=2,
    )

    assert core is not None
    assert semantic_encoder is not None
    assert use_visual_override is True
    assert calls == [("build", 0)]


def test_build_model_sequential_across_ranks_has_no_rank_barrier_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[int] = []
    barriers: list[int] = []

    def fake_build_model(args, *, device):
        del args, device
        built.append(1)
        return "core", None, False

    def fake_barrier(*, use_ddp, device):
        del device
        barriers.append(int(use_ddp))

    monkeypatch.setattr(_MODULE, "_build_model", fake_build_model)
    monkeypatch.setattr(_MODULE, "_distributed_barrier", fake_barrier)

    result = _MODULE._build_model_sequential_across_ranks(
        _base_args(),
        device=torch.device("cpu"),
        rank=1,
        world_size=2,
    )

    assert result == ("core", None, False)
    assert built == [1]
    assert barriers == []


def test_fsdp_grad_norm_handles_mixed_grad_dtypes_without_clip_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyFSDP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bf16 = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.bfloat16))
            self.fp32 = torch.nn.Parameter(torch.tensor([4.0], dtype=torch.float32))

        def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
            raise AssertionError("FSDP clip_grad_norm_ should not be used for mixed-dtype grad norm probing")

    model = _DummyFSDP()
    model.bf16.grad = torch.tensor([3.0], dtype=torch.bfloat16)
    model.fp32.grad = torch.tensor([4.0], dtype=torch.float32)
    monkeypatch.setattr(_MODULE, "_is_fsdp_model", lambda candidate: candidate is model)

    grad_norm = _MODULE._grad_norm_for_training_model(model)

    assert math.isclose(grad_norm, 5.0, rel_tol=0.0, abs_tol=1e-6)


def test_fsdp_grad_clipping_handles_mixed_grad_dtypes_without_clip_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyFSDP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bf16 = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.bfloat16))
            self.fp32 = torch.nn.Parameter(torch.tensor([4.0], dtype=torch.float32))

        def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
            raise AssertionError("FSDP clip_grad_norm_ should not be used for mixed-dtype grad clipping")

    model = _DummyFSDP()
    model.bf16.grad = torch.tensor([3.0], dtype=torch.bfloat16)
    model.fp32.grad = torch.tensor([4.0], dtype=torch.float32)
    monkeypatch.setattr(_MODULE, "_is_fsdp_model", lambda candidate: candidate is model)

    total_norm = _MODULE._clip_grad_norm_for_training_model(model, max_norm=2.5)

    assert math.isclose(total_norm, 5.0, rel_tol=0.0, abs_tol=1e-6)
    torch.testing.assert_close(model.fp32.grad, torch.tensor([2.0], dtype=torch.float32), atol=1e-5, rtol=0)
    torch.testing.assert_close(model.bf16.grad.float(), torch.tensor([1.5], dtype=torch.float32), atol=5e-3, rtol=0)
