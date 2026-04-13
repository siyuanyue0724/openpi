import argparse
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))

import serve_picf_policy as sut


def test_as_sensor_names_arg_from_sequence() -> None:
    assert sut._as_sensor_names_arg(["digit", "gelsight_mini"]) == "digit,gelsight_mini"


def test_as_sensor_offsets_arg_from_sequence() -> None:
    assert sut._as_sensor_offsets_arg(((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0))) == "0.01,0.0,0.0;-0.01,0.0,0.0"


def test_resolve_checkpoint_dir_from_step_dir(tmp_path) -> None:
    output_dir = tmp_path / "exp"
    step_dir = output_dir / "5000"
    step_dir.mkdir(parents=True)
    (step_dir / "model.pt").write_bytes(b"m")
    (step_dir / "metadata.pt").write_bytes(b"d")
    root, step = sut._resolve_checkpoint_dir(step_dir)
    assert root == output_dir
    assert step == step_dir


def test_load_runtime_args_coerces_sequence_fields(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "5000"
    ckpt.mkdir(parents=True)
    import torch

    torch.save(
        {
            "args": {
                "warmup_steps": 10,
                "num_train_steps": 100,
                "log_interval": 10,
                "save_interval": 50,
                "diagnostic_interval": 0,
                "diagnostic_visual_upscale": 64,
                "accum_steps": 1,
                "max_empty_window_retries": 1,
                "unroll_steps": 1,
                "stride": 4,
                "max_points": 128,
                "visual_grid": 4,
                "visual_num_frames": 4,
                "visual_img_size": 224,
                "visual_patch_size": 16,
                "visual_tubelet_size": 2,
                "tactile_num_frames": 4,
                "tactile_stride": 1,
                "pt_bag_kmin": 32,
                "hidden_dim": 256,
                "posterior_hidden_dim": 256,
                "latent_dim": 112,
                "innovation_dim": 256,
                "control_dim": 256,
                "semantic_dim": 2048,
                "semantic_cross_dim": 512,
                "future_hidden_dim": 256,
                "persistent_anchors": 16,
                "observation_anchors": 24,
                "fusion_layers": 4,
                "posterior_layers": 2,
                "predictive_layers": 2,
                "control_layers": 2,
                "predictive_semantic_reads": 2,
                "control_semantic_reads": 2,
                "attention_heads": 8,
                "future_vote_heads": 4,
                "lr": 2e-4,
                "min_lr": 2e-5,
                "weight_decay": 0.0,
                "grad_clip_norm": 1.0,
                "crop_radius_m": 0.1,
                "point_focus_sigma_m": 0.03,
                "point_backbone_lr_scale": 0.25,
                "visual_lr_scale": 0.25,
                "tactile_lr_scale": 0.25,
                "semantic_lr_scale": 0.25,
                "lambda_action_pos": 2.0,
                "lambda_action_rot": 2.0,
                "lambda_action_gripper": 2.0,
                "lambda_visual_latent": 0.2,
                "lambda_visual_real": 0.1,
                "lambda_tactile_real": 0.3,
                "lambda_point_real": 0.3,
                "lambda_semantic_future_aux": 0.25,
                "lambda_anchor_pv": 0.1,
                "lambda_pv_weak": 0.02,
                "lambda_focus_pv": 0.0,
                "lambda_pt": 1.0,
                "pt_bag_radius_m": 0.04,
                "pt_bag_sigma_m": 0.013,
                "pt_back_slack_m": 0.008,
                "p_align_off": 0.35,
                "p_align_on": 0.55,
                "tactile_anchor_prob_on": 0.8,
                "predictive_semantic_dropout_prob": 0.1,
                "device": "cpu",
                "point_backbone": "rgb",
                "visual_mode": "stub",
                "tactile_mode": "stub",
                "semantic_mode": "zero",
                "tactile_sensor_names": ("digit", "gelsight_mini"),
                "tactile_sensor_offsets_m": ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0)),
            }
        },
        ckpt / "metadata.pt",
    )
    monkeypatch.setattr(sut._trainer, "_validate_backbone_args", lambda args: None)
    args = sut._load_runtime_args(ckpt)
    assert isinstance(args, argparse.Namespace)
    assert args.tactile_sensor_names == "digit,gelsight_mini"
    assert args.tactile_sensor_offsets_m == "0.01,0.0,0.0;-0.01,0.0,0.0"


def test_load_model_state_only_uses_compat_loader_on_shape_mismatch(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "5000"
    ckpt.mkdir(parents=True)
    import torch

    torch.save({"step": 123}, ckpt / "metadata.pt")
    torch.save({"bad": torch.tensor([1.0])}, ckpt / "model.pt")

    class _DummyCore:
        def load_state_dict(self, *_args, **_kwargs):
            raise RuntimeError("strict core load failed")

    class _DummyModule:
        def __init__(self):
            self.core = _DummyCore()

        def load_state_dict(self, *_args, **_kwargs):
            raise RuntimeError("strict module load failed")

    compat_calls: list[str] = []

    def _compat(target, state):
        compat_calls.append(type(target).__name__)

    monkeypatch.setattr(sut._trainer, "_load_state_dict_picf_compat", _compat)
    step = sut._load_model_state_only(checkpoint_dir=ckpt, model=_DummyModule(), device=torch.device("cpu"))
    assert step == 123
    assert compat_calls == ["_DummyModule"]
