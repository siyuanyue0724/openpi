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
from openpi.picf.test_utils import build_mini_calvin_dataset


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
        stride=4,
        max_points=1024,
        crop_radius_m=0.10,
        point_focus_sigma_m=0.03,
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
    source.close()


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
        action_pos=torch.tensor(0.03),
        action_rot=torch.tensor(0.04),
        action_gripper=torch.tensor(0.03),
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
