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


def _as_sensor_names_arg(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    raise TypeError(f"Unsupported tactile_sensor_names payload: {type(value).__name__}")


def _as_sensor_offsets_arg(value: object) -> str:
    if isinstance(value, str):
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
    parser.add_argument("--task-anchor-sidecar-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--legacy-semantic-prefix-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-anchor-queries", type=int, default=None)
    parser.add_argument("--task-global-queries", type=int, default=None)
    parser.add_argument("--task-query-layers", type=int, default=None)
    parser.add_argument("--task-query-rounds", type=int, default=None)
    parser.add_argument("--task-anchor-dropout-prob", type=float, default=None)
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
    if args.task_anchor_sidecar_enabled is not None:
        runtime_args.task_anchor_sidecar_enabled = bool(args.task_anchor_sidecar_enabled)
    if args.legacy_semantic_prefix_enabled is not None:
        runtime_args.legacy_semantic_prefix_enabled = bool(args.legacy_semantic_prefix_enabled)
    if args.task_anchor_queries is not None:
        runtime_args.task_anchor_queries = int(args.task_anchor_queries)
    if args.task_global_queries is not None:
        runtime_args.task_global_queries = int(args.task_global_queries)
    if args.task_query_layers is not None:
        runtime_args.task_query_layers = int(args.task_query_layers)
    if args.task_query_rounds is not None:
        runtime_args.task_query_rounds = int(args.task_query_rounds)
    if args.task_anchor_dropout_prob is not None:
        runtime_args.task_anchor_dropout_prob = float(args.task_anchor_dropout_prob)

    _trainer._normalize_train_args(runtime_args)
    _trainer._validate_train_args(runtime_args)
    _trainer._validate_backbone_args(runtime_args)
    _trainer.train(runtime_args)


if __name__ == "__main__":
    main()
