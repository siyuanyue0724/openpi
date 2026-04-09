from __future__ import annotations

import argparse
import importlib.util
import types
from pathlib import Path
import sys

import pytest
import numpy as np

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
        semantic_dim=256,
        future_hidden_dim=256,
        persistent_anchors=16,
        observation_anchors=24,
        fusion_layers=4,
        posterior_layers=2,
        predictive_layers=2,
        control_layers=2,
        attention_heads=8,
        future_vote_heads=4,
        warmup_steps=None,
        lr=2e-4,
        min_lr=2e-5,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        device="cuda",
        point_backbone="sonata",
    )


def test_retryable_first_step_error_detection() -> None:
    exc = RuntimeError("PICF core requires non-empty local xyzrgb support on the first control step.")
    assert _MODULE._is_retryable_first_step_error(exc) is True
    other = RuntimeError("some other training failure")
    assert _MODULE._is_retryable_first_step_error(other) is False


def test_normalize_train_args_sets_default_warmup_fraction() -> None:
    args = _base_args()
    _MODULE._normalize_train_args(args)
    assert args.warmup_steps == 600


def test_validate_train_args_rejects_incompatible_attention_shape() -> None:
    args = _base_args()
    args.hidden_dim = 250
    _MODULE._normalize_train_args(args)
    with pytest.raises(ValueError, match="hidden_dim must be divisible by attention_heads"):
        _MODULE._validate_train_args(args)


def test_validate_train_args_rejects_cpu_sonata() -> None:
    args = _base_args()
    args.device = "cpu"
    _MODULE._normalize_train_args(args)
    with pytest.raises(RuntimeError, match="point_backbone=sonata currently requires CUDA"):
        _MODULE._validate_train_args(args)


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
