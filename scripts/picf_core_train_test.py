from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


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
