from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import picf_core_train as _trainer


def _as_sensor_names_arg(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("(", "[")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
            if isinstance(parsed, (list, tuple)):
                return ",".join(str(item) for item in parsed)
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    raise TypeError(f"Unsupported tactile_sensor_names payload: {type(value).__name__}")


def _as_sensor_offsets_arg(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("(", "[")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
            value = parsed
        else:
            return value
    if isinstance(value, (list, tuple)):
        blocks: list[str] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise TypeError(f"Unsupported tactile_sensor_offsets_m item: {item!r}")
            blocks.append(",".join(str(float(component)) for component in item))
        return ";".join(blocks)
    raise TypeError(f"Unsupported tactile_sensor_offsets_m payload: {type(value).__name__}")


def _load_runtime_args(path: Path) -> argparse.Namespace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = argparse.Namespace(**payload)
    if hasattr(args, "tactile_sensor_names"):
        args.tactile_sensor_names = _as_sensor_names_arg(args.tactile_sensor_names)
    if hasattr(args, "tactile_sensor_offsets_m"):
        args.tactile_sensor_offsets_m = _as_sensor_offsets_arg(args.tactile_sensor_offsets_m)
    if bool(getattr(args, "use_foundation_backbones", False)):
        _trainer._apply_foundation_profile(args)
    _trainer._normalize_train_args(args)
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--diagnostic-interval", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--grad-clip-mode", choices=["fixed", "percentile"], default=None)
    parser.add_argument("--grad-clip-percentile", type=float, default=None)
    parser.add_argument("--grad-clip-window", type=int, default=None)
    args = parser.parse_args()

    runtime_args = _load_runtime_args(args.args_json)
    runtime_args.resume_checkpoint = args.resume_checkpoint
    runtime_args.exp_name = args.exp_name
    if args.device is not None:
        runtime_args.device = args.device
    if args.num_train_steps is not None:
        runtime_args.num_train_steps = int(args.num_train_steps)
    if args.save_interval is not None:
        runtime_args.save_interval = int(args.save_interval)
    if args.log_interval is not None:
        runtime_args.log_interval = int(args.log_interval)
    if args.diagnostic_interval is not None:
        runtime_args.diagnostic_interval = int(args.diagnostic_interval)
    if args.grad_clip_norm is not None:
        runtime_args.grad_clip_norm = float(args.grad_clip_norm)
    if args.grad_clip_mode is not None:
        runtime_args.grad_clip_mode = str(args.grad_clip_mode)
    if args.grad_clip_percentile is not None:
        runtime_args.grad_clip_percentile = float(args.grad_clip_percentile)
    if args.grad_clip_window is not None:
        runtime_args.grad_clip_window = int(args.grad_clip_window)

    _trainer._normalize_train_args(runtime_args)
    _trainer._validate_train_args(runtime_args)
    _trainer._validate_backbone_args(runtime_args)
    _trainer.train(runtime_args)


if __name__ == "__main__":
    main()
