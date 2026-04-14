from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_SCRIPT_PATH = Path(__file__).with_name("picf_resume_train.py")
_SCRIPT_DIR = _SCRIPT_PATH.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_SPEC = importlib.util.spec_from_file_location("picf_resume_train_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_TRAIN_TEST_PATH = Path(__file__).with_name("picf_core_train_test.py")
_TRAIN_TEST_SPEC = importlib.util.spec_from_file_location("picf_core_train_test_script", _TRAIN_TEST_PATH)
assert _TRAIN_TEST_SPEC is not None and _TRAIN_TEST_SPEC.loader is not None
_TRAIN_TEST = importlib.util.module_from_spec(_TRAIN_TEST_SPEC)
sys.modules[_TRAIN_TEST_SPEC.name] = _TRAIN_TEST
_TRAIN_TEST_SPEC.loader.exec_module(_TRAIN_TEST)


def _write_args_json(path: Path) -> Path:
    payload = vars(_TRAIN_TEST._base_args()).copy()
    args_json = path / "args.json"
    args_json.write_text(json.dumps(payload), encoding="utf-8")
    return args_json


def test_load_runtime_args_fills_grad_clip_defaults(tmp_path: Path) -> None:
    args_json = _write_args_json(tmp_path)

    runtime_args = _MODULE._load_runtime_args(args_json)

    assert runtime_args.grad_clip_mode == "percentile"
    assert runtime_args.grad_clip_percentile == pytest.approx(75.0)
    assert runtime_args.grad_clip_window == 100

def test_main_applies_resume_cli_overrides_before_train(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args_json = _write_args_json(tmp_path)
    captured: dict[str, argparse.Namespace] = {}

    monkeypatch.setattr(_MODULE._trainer, "_validate_train_args", lambda args: None)
    monkeypatch.setattr(_MODULE._trainer, "_validate_backbone_args", lambda args: None)

    def _capture_train(args: argparse.Namespace) -> None:
        captured["args"] = args

    monkeypatch.setattr(_MODULE._trainer, "train", _capture_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "picf_resume_train.py",
            "--args-json",
            str(args_json),
            "--resume-checkpoint",
            "/tmp/checkpoint/10000",
            "--exp-name",
            "semantic_prefix_primary_cli_test",
            "--device",
            "cpu",
            "--grad-clip-mode",
            "percentile",
            "--grad-clip-percentile",
            "80",
            "--grad-clip-window",
            "32",
        ],
    )

    _MODULE.main()

    args = captured["args"]
    assert args.resume_checkpoint == "/tmp/checkpoint/10000"
    assert args.exp_name == "semantic_prefix_primary_cli_test"
    assert args.device == "cpu"
    assert args.grad_clip_mode == "percentile"
    assert args.grad_clip_percentile == pytest.approx(80.0)
    assert args.grad_clip_window == 32
