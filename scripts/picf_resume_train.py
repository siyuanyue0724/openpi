from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import picf_core_train as _trainer


def _load_runtime_args(path: Path) -> argparse.Namespace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = argparse.Namespace(**payload)
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

    _trainer._normalize_train_args(runtime_args)
    _trainer._validate_train_args(runtime_args)
    _trainer._validate_backbone_args(runtime_args)
    _trainer.train(runtime_args)


if __name__ == "__main__":
    main()
