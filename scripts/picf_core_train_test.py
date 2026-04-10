from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import math
import types
from pathlib import Path
import sys

import pytest
import numpy as np
import torch

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import PicfPointCloudFrame


_SCRIPT_PATH = Path(__file__).with_name("picf_core_train.py")
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
        stride=8,
        max_points=512,
        visual_grid=8,
        visual_num_frames=64,
        visual_img_size=384,
        visual_patch_size=16,
        visual_tubelet_size=2,
        tactile_num_frames=4,
        tactile_stride=2,
        hidden_dim=256,
        posterior_hidden_dim=256,
        latent_dim=112,
        innovation_dim=256,
        control_dim=256,
        semantic_dim=2048,
        semantic_cross_dim=512,
        future_hidden_dim=256,
        persistent_anchors=16,
        observation_anchors=24,
        fusion_layers=4,
        posterior_layers=2,
        predictive_layers=2,
        control_layers=2,
        predictive_semantic_reads=2,
        control_semantic_reads=2,
        predictive_semantic_dropout_prob=0.1,
        attention_heads=8,
        future_vote_heads=4,
        warmup_steps=None,
        lr=2e-4,
        min_lr=2e-5,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
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
        semantic_use_gripper=True,
        semantic_max_length=256,
        visual_mode="stub",
        tactile_mode="stub",
        use_tactile=False,
        visual_model_name="vjepa2_1_vit_base_384",
        visual_checkpoint_path=None,
        visual_checkpoint_key=None,
        visual_dtype="bfloat16",
        tactile_checkpoint_path=None,
        tactile_dtype="float32",
        tactile_sensor_names="digit,gelsight_mini",
        tactile_sensor_offsets_m="0.01,0,0;-0.01,0,0",
        use_foundation_backbones=False,
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


def test_normalize_train_args_disables_semantic_gradient_checkpointing_for_accumulation() -> None:
    args = _base_args()
    args.semantic_mode = "paligemma"
    args.semantic_gradient_checkpointing = True
    args.accum_steps = 2

    _MODULE._normalize_train_args(args)

    assert args.semantic_gradient_checkpointing is False
    assert args.semantic_gradient_checkpointing_disabled_for_accum is True


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


def test_validate_train_args_rejects_incompatible_attention_shape() -> None:
    args = _base_args()
    args.hidden_dim = 250
    _MODULE._normalize_train_args(args)
    with pytest.raises(ValueError, match="hidden_dim must be divisible by attention_heads"):
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
                debug={"projective_candidate_density": 0.0},
            )

    dummy_losses = types.SimpleNamespace(
        total=torch.tensor(1.0),
        action=torch.tensor(0.1),
        visual_latent=torch.tensor(0.1),
        visual_real=torch.tensor(0.1),
        tactile_real=torch.tensor(0.1),
        point_real=torch.tensor(0.1),
        semantic_future_aux=torch.tensor(0.1),
        alignment=torch.tensor(0.1),
        anchor_pv=torch.tensor(0.1),
        pv_weak=torch.tensor(0.1),
        focus_pv=torch.tensor(0.1),
        pt=torch.tensor(0.1),
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
    output_dir = tmp_path / "picf_ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)

    _MODULE._save_checkpoint(
        output_dir=output_dir,
        model=trainer,
        optimizer=optimizer,
        step=7,
        args=args,
    )

    reloaded = torch.nn.Module()
    reloaded.core = _DummyCore()
    reloaded.semantic_encoder = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        reloaded.core.proj.weight.zero_()
        reloaded.semantic_encoder.weight.zero_()
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)

    step = _MODULE._load_checkpoint(
        path=output_dir / "7",
        model=reloaded,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
    )

    assert step == 7
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
