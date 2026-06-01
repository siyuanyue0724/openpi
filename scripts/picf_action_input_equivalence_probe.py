from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any
import types

import torch

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.picf_action_bridge_capacity_probe import _set_override
from scripts.picf_core_train import _build_loss_config
from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _ensure_window_has_valid_first_step_xyzrgb_support
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _load_tactile_backgrounds_npz
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _resolve_action_normalizer
from scripts.picf_core_train import _seed_everything
from scripts.picf_fixed_window_action_probe import _load_window_records
from scripts.picf_fixed_window_action_probe import _numeric_snapshot
from scripts.picf_replay_windows import _coerce_loaded_args
from scripts.picf_replay_windows import _resolve_rank_seed
from scripts.sonata_window_probe import _override_build_sample


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


def _tensor_stats(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"present": False, "type": type(value).__name__}
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total = int(detached.numel())
    stats: dict[str, Any] = {
        "present": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": total,
        "finite_count": finite_count,
        "nonfinite_count": total - finite_count,
    }
    if total == 0:
        return stats
    x = detached.to(dtype=torch.float32)
    finite_x = x[finite] if finite.any() else x.reshape(0)
    if finite_x.numel() > 0:
        stats.update(
            {
                "mean": float(finite_x.mean().item()),
                "std": float(finite_x.std(unbiased=False).item()) if finite_x.numel() > 1 else 0.0,
                "rms": float(torch.sqrt(torch.mean(finite_x.square())).item()),
                "min": float(finite_x.min().item()),
                "max": float(finite_x.max().item()),
                "abs_sum": float(finite_x.abs().sum().item()),
                "signed_sum": float(finite_x.sum().item()),
            }
        )
    return stats


def _install_flow_capture(semantic_encoder: Any, calls: list[dict[str, Any]]) -> Any:
    original = getattr(semantic_encoder, "compute_action_flow_loss")

    def _wrapped(_self: Any, *args: Any, **kwargs: Any) -> Any:
        call: dict[str, Any] = {
            "call_index": len(calls),
            "num_positional_args": len(args),
            "semantic_override": _tensor_stats(args[0] if args else None),
            "action_chunk_target": _tensor_stats(kwargs.get("action_chunk_target")),
            "extra_prefix_tokens": _tensor_stats(kwargs.get("extra_prefix_tokens")),
            "extra_action_context_tokens": _tensor_stats(kwargs.get("extra_action_context_tokens")),
            "kwarg_keys": sorted(str(key) for key in kwargs.keys()),
        }
        result = original(*args, **kwargs)
        if isinstance(result, dict):
            for key in (
                "total",
                "action_pos",
                "action_rot",
                "action_gripper",
                "predicted_action",
                "predicted_chunk",
            ):
                if key in result:
                    call[f"result_{key}"] = _tensor_stats(result[key])
        calls.append(call)
        return result

    setattr(semantic_encoder, "compute_action_flow_loss", types.MethodType(_wrapped, semantic_encoder))
    return original


def _restore_flow_capture(semantic_encoder: Any, original: Any) -> None:
    setattr(semantic_encoder, "compute_action_flow_loss", original)


def _make_source(train_args: Any) -> _CalvinTransitionSource:
    segment_indices = None
    if getattr(train_args, "calvin_segment_indices", None):
        segment_indices = [
            int(part)
            for part in str(getattr(train_args, "calvin_segment_indices")).split(",")
            if part.strip()
        ]
    return _CalvinTransitionSource(
        train_args.calvin_root,
        split=train_args.split,
        backend=train_args.backend,
        unroll_steps=train_args.effective_unroll_steps,
        action_horizon=train_args.action_horizon,
        use_tactile=bool(train_args.use_tactile),
        tactile_sensor_names=train_args.tactile_sensor_names,
        tactile_sensor_offsets_m=train_args.tactile_sensor_offsets_m,
        tactile_calibration=train_args.tactile_calibration_path,
        tactile_backgrounds_by_sensor=_load_tactile_backgrounds_npz(train_args.tactile_backgrounds_path),
        use_scene_obs=bool(train_args.use_scene_obs),
        load_tracklet_fields=bool(getattr(train_args, "tracklet_memory_enabled", False)),
        load_proposal_fields=bool(getattr(train_args, "proposal_memory_enabled", False)),
        mvtrack_sidecar_root=getattr(train_args, "mvtrack_sidecar_root", None),
        mvtrack_sidecar_proposal_nearest_max_gap=int(
            getattr(train_args, "mvtrack_sidecar_proposal_nearest_max_gap", 0)
        ),
        action_normalizer=_resolve_action_normalizer(train_args),
        augmentation_mode=train_args.picf_augmentation_mode,
        photometric_strength=train_args.picf_photometric_strength,
        segment_indices=segment_indices,
    )


def _prepare_args(base_payload: dict[str, Any], *, device: str, burnin_steps: int) -> Any:
    train_args = _coerce_loaded_args(base_payload, device_override=device)
    _set_override(train_args, "training_strategy", "ddp")
    _set_override(train_args, "optimizer_sharding", "none")
    _set_override(train_args, "picf_trainable_scope", "policy_only")
    _set_override(train_args, "semantic_trainable", True)
    _set_override(train_args, "semantic_trainable_scope", "action_head_and_adapter")
    _set_override(train_args, "picf_action_condition_enabled", False)
    _set_override(train_args, "burnin_steps", int(burnin_steps))
    _set_override(train_args, "lr", 0.0)
    _set_override(train_args, "min_lr", 0.0)
    return train_args


def _run_mode(
    *,
    mode: str,
    base_payload: dict[str, Any],
    checkpoint: Path,
    window_record: dict[str, Any],
    device: torch.device,
    rank_seed: int,
    point_grid_mode: str,
    deterministic_seed: int,
    burnin_steps: int,
) -> dict[str, Any]:
    _seed_everything(int(_coerce_loaded_args(base_payload, device_override=str(device)).seed), int(rank_seed))
    train_args = _prepare_args(base_payload, device=str(device), burnin_steps=burnin_steps)
    if mode == "enabled_native":
        _set_override(train_args, "picf_mode", "enabled")
    elif mode == "ablated":
        # Build full architecture first, then disable PICF after strict load.
        pass
    else:
        raise ValueError(f"Unsupported mode={mode!r}")

    override_context = _override_build_sample(str(point_grid_mode)) if point_grid_mode != "default" else _NullContext()
    with override_context:
        source = _make_source(train_args)
        window = source.window_from_metadata(
            segment_id=int(window_record["segment"]),
            start_step_id=int(window_record["start_step"]),
        )
        core, semantic_encoder, use_visual_override = _build_model(train_args, device=device)
        core = core.to(device)
        trainer = _PicfWindowTrainer(
            core,
            semantic_encoder=semantic_encoder,
            visual_grid=train_args.visual_grid,
            use_visual_override=use_visual_override,
            loss_config=_build_loss_config(train_args),
            picf_mode=str(getattr(train_args, "picf_mode", "enabled")),
            burnin_steps=int(getattr(train_args, "burnin_steps", 0)),
            burnin_mode=str(getattr(train_args, "burnin_mode", "full")),
        ).to(device)
        _materialize_model_parameters(trainer, source=source, rank=int(rank_seed))
        dummy_optimizer = torch.optim.AdamW(
            [p for p in trainer.parameters() if getattr(p, "requires_grad", False)],
            lr=0.0,
        )
        loaded_step = _load_checkpoint(path=checkpoint, model=trainer, optimizer=dummy_optimizer, device=device)
        if mode == "ablated":
            trainer.picf_mode = "ablated"
            trainer.policy.picf_enabled = False
        point_counts = _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
        calls: list[dict[str, Any]] = []
        original = _install_flow_capture(trainer.semantic_encoder, calls)
        try:
            with torch.no_grad():
                with torch.random.fork_rng(devices=[int(device.index or 0)] if device.type == "cuda" else [], enabled=True):
                    torch.manual_seed(int(deterministic_seed))
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(int(deterministic_seed))
                    outputs = trainer(window, capture_visual_diagnostics=False)
        finally:
            _restore_flow_capture(trainer.semantic_encoder, original)
        snapshot = _numeric_snapshot(outputs)
        result = {
            "mode": mode,
            "loaded_step": int(loaded_step),
            "burnin_steps": int(burnin_steps),
            "point_counts": point_counts,
            "metrics": snapshot,
            "flow_call_count": len(calls),
            "flow_calls": calls,
        }
        del trainer, core, semantic_encoder, source, window, dummy_optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dump action-flow inputs for ablated vs PICF-enabled/native-action paths "
            "on the same exact window. This is a dataflow equivalence probe, not training."
        )
    )
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window-jsonl", required=True)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank-seed", type=int, default=None)
    parser.add_argument("--point-grid-mode", choices=("default", "original", "rebased"), default="default")
    parser.add_argument("--deterministic-seed", type=int, default=20260601)
    parser.add_argument("--burnin-steps", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.args_json).read_text(encoding="utf-8"))
    records = _load_window_records(Path(args.window_jsonl))
    if int(args.window_index) < 0 or int(args.window_index) >= len(records):
        raise IndexError(f"window-index out of range: {args.window_index} for {len(records)} records")
    record = records[int(args.window_index)]
    device = torch.device(args.device)
    rank_seed = _resolve_rank_seed(rank_seed=args.rank_seed, rng_rank=None)

    results = []
    for mode in ("ablated", "enabled_native"):
        results.append(
            _run_mode(
                mode=mode,
                base_payload=payload,
                checkpoint=Path(args.checkpoint),
                window_record=record,
                device=device,
                rank_seed=int(rank_seed),
                point_grid_mode=str(args.point_grid_mode),
                deterministic_seed=int(args.deterministic_seed),
                burnin_steps=int(args.burnin_steps),
            )
        )

    summary: dict[str, Any] = {
        "window_index": int(args.window_index),
        "record": record,
        "results": results,
    }
    if len(results) == 2 and results[0]["flow_calls"] and results[1]["flow_calls"]:
        a = results[0]["flow_calls"][0]
        b = results[1]["flow_calls"][0]
        summary["first_call_deltas"] = {
            "semantic_override_rms_delta": (
                b["semantic_override"].get("rms", math.nan) - a["semantic_override"].get("rms", math.nan)
            ),
            "action_chunk_target_rms_delta": (
                b["action_chunk_target"].get("rms", math.nan) - a["action_chunk_target"].get("rms", math.nan)
            ),
            "extra_prefix_present_pair": [
                bool(a["extra_prefix_tokens"].get("present")),
                bool(b["extra_prefix_tokens"].get("present")),
            ],
            "extra_action_context_present_pair": [
                bool(a["extra_action_context_tokens"].get("present")),
                bool(b["extra_action_context_tokens"].get("present")),
            ],
        }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"stage": "done", "output_json": args.output_json, "modes": [r["mode"] for r in results]}))


if __name__ == "__main__":
    main()
