from __future__ import annotations

import argparse
import contextlib
import json
from collections import deque
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any
from typing import Callable

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _build_optimizer
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _checkpoint_dir_for_step
from scripts.picf_core_train import _collect_nonfinite_gradient_diagnostics
from scripts.picf_core_train import _collect_nonfinite_parameter_diagnostics
from scripts.picf_core_train import _dump_debug_index_trace
from scripts.picf_core_train import _ensure_window_has_valid_first_step_xyzrgb_support
from scripts.picf_core_train import _grad_norm
from scripts.picf_core_train import _is_retryable_first_step_error
from scripts.picf_core_train import _lr_for_step
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _normalize_train_args
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _save_checkpoint
from scripts.picf_core_train import _install_debug_tensor_index_guards
from scripts.picf_core_train import _seed_everything
from scripts.picf_core_train import _set_optimizer_lr
from scripts.picf_core_train import _validate_train_args
from scripts.sonata_window_probe import _override_build_sample


def _coerce_loaded_args(payload: dict[str, Any], *, device_override: str | None) -> argparse.Namespace:
    args = argparse.Namespace(**payload)
    if not hasattr(args, "sonata_disable_flash"):
        args.sonata_disable_flash = False
    if device_override is not None:
        args.device = str(device_override)
    tactile_names = getattr(args, "tactile_sensor_names", ("digit", "gelsight_mini"))
    if isinstance(tactile_names, str):
        tactile_names = tuple(part.strip() for part in tactile_names.split(",") if part.strip())
    else:
        tactile_names = tuple(str(name) for name in tactile_names)
    tactile_offsets = getattr(args, "tactile_sensor_offsets_m", ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0)))
    if isinstance(tactile_offsets, str):
        blocks = [block.strip() for block in tactile_offsets.split(";") if block.strip()]
        tactile_offsets = tuple(tuple(float(value.strip()) for value in block.split(",") if value.strip()) for block in blocks)
    else:
        tactile_offsets = tuple(tuple(float(value) for value in offset) for offset in tactile_offsets)
    args.tactile_sensor_names = tactile_names
    args.tactile_sensor_offsets_m = tactile_offsets
    _normalize_train_args(args)
    _validate_train_args(args)
    return args


def _parse_flat_indices(raw: str) -> list[int]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one flat index.")
    return [int(value) for value in values]


def _generate_rng_flat_indices(
    *,
    total_windows: int,
    dataset_size: int,
    seed: int,
    rank: int,
    skip_windows: int,
) -> list[int]:
    if total_windows < 1:
        raise ValueError(f"Expected total_windows >= 1, got {total_windows}.")
    if dataset_size < 1:
        raise ValueError(f"Expected dataset_size >= 1, got {dataset_size}.")
    if skip_windows < 0:
        raise ValueError(f"Expected skip_windows >= 0, got {skip_windows}.")
    rng = np.random.default_rng(int(seed) + 17 * int(rank))
    flat_indices: list[int] = []
    total_required = int(skip_windows) + int(total_windows)
    for index in range(total_required):
        flat_index = int(rng.integers(0, dataset_size))
        if index >= skip_windows:
            flat_indices.append(flat_index)
    return flat_indices


def _resolve_rank_seed(*, rank_seed: int | None, rng_rank: int | None) -> int:
    if rank_seed is not None:
        return int(rank_seed)
    if rng_rank is not None:
        return int(rng_rank)
    return 1


def _coerce_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected an optional boolean-like value, got {value!r}.")


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _advance_rng_draws(*, rng: np.random.Generator, dataset_size: int, draw_count: int) -> None:
    if int(draw_count) <= 0:
        return
    rng.integers(0, int(dataset_size), size=int(draw_count))


def _save_replay_state(
    *,
    checkpoint_dir: Path,
    accepted_counter: int,
    raw_draw_counter: int,
    retryable_skip_count: int,
    accepted_flat_indices: list[int],
    retryable_skips: list[dict[str, Any]],
) -> None:
    payload = {
        "accepted_counter": int(accepted_counter),
        "raw_draw_counter": int(raw_draw_counter),
        "retryable_skip_count": int(retryable_skip_count),
        "accepted_flat_indices": [int(value) for value in accepted_flat_indices],
        "retryable_skips": retryable_skips,
    }
    (checkpoint_dir / "replay_state.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _load_replay_state(checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / "replay_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Replay checkpoint is missing {state_path}.")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["accepted_counter"] = int(payload.get("accepted_counter", 0))
    payload["raw_draw_counter"] = int(payload.get("raw_draw_counter", 0))
    payload["retryable_skip_count"] = int(payload.get("retryable_skip_count", 0))
    payload["accepted_flat_indices"] = [int(value) for value in payload.get("accepted_flat_indices", [])]
    payload["retryable_skips"] = list(payload.get("retryable_skips", []))
    return payload


def _capture_replay_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device=device)
    return state


def _restore_replay_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device=device)


def _save_replay_rng_state(*, checkpoint_dir: Path, device: torch.device) -> None:
    torch.save(_capture_replay_rng_state(device), checkpoint_dir / "replay_rng.pt")


def _load_replay_rng_state(checkpoint_dir: Path, *, device: torch.device) -> bool:
    state_path = checkpoint_dir / "replay_rng.pt"
    if not state_path.exists():
        return False
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    _restore_replay_rng_state(payload, device)
    return True


def _prune_old_replay_checkpoints(*, checkpoint_root: Path, keep: int) -> None:
    if keep <= 0 or not checkpoint_root.exists():
        return
    step_dirs = sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and path.name.isdigit() and not path.name.startswith("tmp_")
        ),
        key=lambda path: int(path.name),
    )
    stale = step_dirs[:-int(keep)]
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)


def _ordered_loss_component_keys(outputs: dict[str, torch.Tensor]) -> list[str]:
    ordered_keys = [
        "loss_action",
        "loss_visual_latent",
        "loss_visual_real",
        "loss_tactile_real",
        "loss_point_real",
        "loss_semantic_future_aux",
        "loss_alignment",
    ]
    return [key for key in ordered_keys if key in outputs and outputs[key] is not None]


def _scalar_loss_snapshot(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = [
        "loss_total",
        "loss_action",
        "loss_action_pos",
        "loss_action_rot",
        "loss_action_gripper",
        "loss_visual_latent",
        "loss_visual_real",
        "loss_tactile_real",
        "loss_point_real",
        "loss_semantic_future_aux",
        "loss_alignment",
        "loss_anchor_pv",
        "loss_pv_weak",
        "loss_focus_pv",
        "loss_pt",
        "projective_candidate_density",
    ]
    snapshot: dict[str, float] = {}
    for key in keys:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            snapshot[key] = float(value.detach().item())
    return snapshot


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device=device)
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device=device)


def _capture_buffer_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.named_buffers()
    }


def _restore_buffer_state(model: torch.nn.Module, buffer_state: dict[str, torch.Tensor]) -> None:
    for name, buffer in model.named_buffers():
        saved = buffer_state.get(name)
        if saved is None:
            continue
        buffer.copy_(saved.to(device=buffer.device, dtype=buffer.dtype))


def _backward_loss_components(
    *,
    outputs: dict[str, torch.Tensor],
    device: torch.device,
    split_backward: bool,
    replay_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    forward_fn: Callable[[], dict[str, torch.Tensor]] | None = None,
    pre_forward_rng_state: dict[str, Any] | None = None,
    pre_forward_buffer_state: dict[str, torch.Tensor] | None = None,
) -> None:
    if not split_backward:
        _synchronize_if_cuda(device)
        outputs["loss_total"].backward()
        _synchronize_if_cuda(device)
        return

    if forward_fn is None:
        raise ValueError("forward_fn is required when split_backward=True.")

    active = _ordered_loss_component_keys(outputs)
    if not active:
        raise RuntimeError("No backward components found in replay outputs.")
    if pre_forward_rng_state is None or pre_forward_buffer_state is None:
        raise ValueError("pre_forward_rng_state and pre_forward_buffer_state are required when split_backward=True.")
    rng_state = pre_forward_rng_state
    buffer_state = pre_forward_buffer_state
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for key in active:
        _restore_rng_state(rng_state, device)
        _restore_buffer_state(model, buffer_state)
        optimizer.zero_grad(set_to_none=True)
        component_outputs = forward_fn()
        if key not in component_outputs or component_outputs[key] is None:
            continue
        print(
            json.dumps(
                {
                    "stage": "component_backward_start",
                    "replay_step": int(replay_step),
                    "component": str(key),
                    "diagnostic_mode": "recompute_per_component",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _synchronize_if_cuda(device)
        component_outputs[key].backward()
        _synchronize_if_cuda(device)
        grad_issue = _collect_nonfinite_gradient_diagnostics(model, max_items=24)
        if int(grad_issue["nonfinite_grad_count"]) > 0:
            print(
                json.dumps(
                    {
                        "stage": "component_backward_nonfinite_grad",
                        "replay_step": int(replay_step),
                        "component": str(key),
                        "grad_issue": grad_issue,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise RuntimeError(
                f"Non-finite gradients detected during split backward for component {key!r} at replay_step={replay_step}."
            )
        optimizer.zero_grad(set_to_none=True)
        del component_outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "stage": "component_backward_ok",
                    "replay_step": int(replay_step),
                    "component": str(key),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _restore_rng_state(rng_state, device)
    _restore_buffer_state(model, buffer_state)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_outputs = forward_fn()
    outputs.clear()
    outputs.update(final_outputs)
    _synchronize_if_cuda(device)
    outputs["loss_total"].backward()
    _synchronize_if_cuda(device)


def _diagnose_nonfinite_loss_components(
    *,
    replay_step: int,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    forward_fn: Callable[[], dict[str, torch.Tensor]],
    pre_forward_rng_state: dict[str, Any],
    pre_forward_buffer_state: dict[str, torch.Tensor],
) -> None:
    _restore_rng_state(pre_forward_rng_state, device)
    _restore_buffer_state(model, pre_forward_buffer_state)
    baseline_outputs = forward_fn()
    print(
        json.dumps(
            {
                "stage": "nonfinite_component_diagnosis_start",
                "replay_step": int(replay_step),
                "losses": _scalar_loss_snapshot(baseline_outputs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _backward_loss_components(
        outputs=baseline_outputs,
        device=device,
        split_backward=True,
        replay_step=int(replay_step),
        model=model,
        optimizer=optimizer,
        forward_fn=forward_fn,
        pre_forward_rng_state=pre_forward_rng_state,
        pre_forward_buffer_state=pre_forward_buffer_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay specific PICF CALVIN transition windows through the current training graph.")
    parser.add_argument("--args-json", required=True, help="Path to args.json from a PICF training run.")
    parser.add_argument("--flat-indices", default=None, help="Comma-separated list of flat indices to replay in order.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the flat-index sequence.")
    parser.add_argument("--device", default=None, help="Override device from args.json, e.g. cuda or cpu.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint directory or latest.pt to load before replay.")
    parser.add_argument(
        "--rank-seed",
        type=int,
        default=None,
        help="Rank used for model/dropout RNG. Defaults to --rng-rank when provided, else 1.",
    )
    parser.add_argument("--optimizer-step", action="store_true", help="Apply optimizer.step() after each replayed window.")
    parser.add_argument(
        "--rng-num-windows",
        type=int,
        default=None,
        help="Generate the exact training flat-index sequence from args.seed/rank instead of passing --flat-indices.",
    )
    parser.add_argument(
        "--rng-rank",
        type=int,
        default=None,
        help="Rank to reproduce for --rng-num-windows. Defaults to --rank-seed.",
    )
    parser.add_argument(
        "--rng-skip-windows",
        type=int,
        default=0,
        help="Skip this many training RNG draws before replaying --rng-num-windows windows.",
    )
    parser.add_argument(
        "--dump-generated-sequence",
        default=None,
        help="Optional path to save generated flat indices as JSON when using --rng-num-windows.",
    )
    parser.add_argument(
        "--point-grid-mode",
        choices=("default", "original", "rebased"),
        default="default",
        help="Override Sonata local-grid preprocessing during replay for debugging.",
    )
    parser.add_argument(
        "--override-sonata-disable-flash",
        default=None,
        help="Override args_json sonata_disable_flash with true/false for controlled replay.",
    )
    parser.add_argument(
        "--split-backward-from-step",
        type=int,
        default=None,
        help="From this replay step onward, backward individual loss components with synchronization.",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="Stop replay after this many steps even if more flat indices remain.",
    )
    parser.add_argument(
        "--save-checkpoint-every",
        type=int,
        default=0,
        help="Save replay model/optimizer checkpoint every N replay steps. Disabled when <= 0.",
    )
    parser.add_argument(
        "--save-checkpoint-dir",
        default=None,
        help="Directory to write replay checkpoints when --save-checkpoint-every > 0.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="Maximum number of replay checkpoint step directories to retain when checkpoint saving is enabled.",
    )
    parser.add_argument(
        "--diagnose-nonfinite-by-component",
        action="store_true",
        help="When a full backward yields non-finite grads, recompute the same step component-by-component to isolate the first bad branch.",
    )
    args = parser.parse_args()

    args_json_path = Path(args.args_json)
    payload = json.loads(args_json_path.read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=args.device)
    override_sonata_disable_flash = _coerce_optional_bool(args.override_sonata_disable_flash)
    if override_sonata_disable_flash is not None:
        train_args.sonata_disable_flash = bool(override_sonata_disable_flash)
    device = torch.device(str(train_args.device))
    if int(args.repeat) < 1:
        raise ValueError(f"--repeat must be >= 1, got {args.repeat}.")
    if int(args.save_checkpoint_every) > 0 and args.save_checkpoint_dir is None:
        raise ValueError("--save-checkpoint-dir is required when --save-checkpoint-every > 0.")
    if int(args.max_checkpoints) < 1:
        raise ValueError(f"--max-checkpoints must be >= 1, got {args.max_checkpoints}.")
    save_checkpoint_dir = Path(args.save_checkpoint_dir) if args.save_checkpoint_dir is not None else None
    if save_checkpoint_dir is not None:
        save_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.stop_after_step is not None and int(args.stop_after_step) < 1:
        raise ValueError(f"--stop-after-step must be >= 1, got {args.stop_after_step}.")
    if args.split_backward_from_step is not None and int(args.split_backward_from_step) < 1:
        raise ValueError(
            f"--split-backward-from-step must be >= 1, got {args.split_backward_from_step}."
        )

    effective_rank_seed = _resolve_rank_seed(rank_seed=args.rank_seed, rng_rank=args.rng_rank)
    _seed_everything(int(train_args.seed), int(effective_rank_seed))
    debug_autograd_anomaly = os.environ.get("OPENPI_DEBUG_AUTOGRAD_ANOMALY", "").strip() not in {"", "0", "false", "False"}
    debug_tensor_index_guards = os.environ.get("OPENPI_DEBUG_TENSOR_INDEX_GUARDS", "").strip() not in {"", "0", "false", "False"}
    if debug_autograd_anomaly:
        torch.autograd.set_detect_anomaly(True)

    override_context = contextlib.nullcontext() if args.point_grid_mode == "default" else _override_build_sample(str(args.point_grid_mode))
    with override_context:
        start_time = time.time()
        source = _CalvinTransitionSource(
            train_args.calvin_root,
            split=train_args.split,
            backend=train_args.backend,
            unroll_steps=train_args.unroll_steps,
            use_tactile=bool(train_args.use_tactile),
            tactile_sensor_names=train_args.tactile_sensor_names,
            tactile_sensor_offsets_m=train_args.tactile_sensor_offsets_m,
        )
        try:
            print(json.dumps({"stage": "source_ready", "elapsed_s": round(time.time() - start_time, 3)}), flush=True)
            core, semantic_encoder, use_visual_override = _build_model(train_args, device=device)
            core = core.to(device)
            print(json.dumps({"stage": "model_built", "elapsed_s": round(time.time() - start_time, 3)}), flush=True)
            trainer = _PicfWindowTrainer(
                core,
                semantic_encoder=semantic_encoder,
                visual_grid=train_args.visual_grid,
                use_visual_override=use_visual_override,
            ).to(device)
            _materialize_model_parameters(trainer, source=source, rank=int(effective_rank_seed))
            print(json.dumps({"stage": "params_materialized", "elapsed_s": round(time.time() - start_time, 3)}), flush=True)
            optimizer, _ = _build_optimizer(trainer, args=train_args)
            print(json.dumps({"stage": "optimizer_built", "elapsed_s": round(time.time() - start_time, 3)}), flush=True)
            loaded_step = 0
            replay_rng_loaded = False
            replay_resume_state: dict[str, Any] | None = None
            if args.checkpoint:
                checkpoint_path = Path(args.checkpoint)
                loaded_step = _load_checkpoint(path=checkpoint_path, model=trainer, optimizer=optimizer, device=device)
                if args.rng_num_windows is not None:
                    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else Path(
                        torch.load(checkpoint_path, map_location="cpu", weights_only=False)["checkpoint_dir"]
                    )
                    replay_resume_state = _load_replay_state(checkpoint_dir)
                    if int(replay_resume_state["accepted_counter"]) != int(loaded_step):
                        raise RuntimeError(
                            f"Replay checkpoint accepted_counter={replay_resume_state['accepted_counter']} does not match loaded_step={loaded_step}."
                        )
                    replay_rng_loaded = _load_replay_rng_state(checkpoint_dir, device=device)
            trainer.train()
            if debug_tensor_index_guards:
                _install_debug_tensor_index_guards()
            print(json.dumps({"stage": "entering_replay_loop", "elapsed_s": round(time.time() - start_time, 3)}), flush=True)

            if args.rng_num_windows is not None:
                rng_rank = int(args.rng_rank if args.rng_rank is not None else effective_rank_seed)
                accepted_target = int(args.rng_num_windows) * int(args.repeat)
                rng = np.random.default_rng(int(train_args.seed) + 17 * int(rng_rank))
                raw_draw_counter = int(args.rng_skip_windows)
                accepted_counter = 0
                retryable_skip_count = 0
                accepted_flat_indices: list[int] = []
                retryable_skips: list[dict[str, Any]] = []
                if replay_resume_state is not None:
                    raw_draw_counter = int(replay_resume_state["raw_draw_counter"])
                    accepted_counter = int(replay_resume_state["accepted_counter"])
                    retryable_skip_count = int(replay_resume_state["retryable_skip_count"])
                    accepted_flat_indices = list(replay_resume_state["accepted_flat_indices"])
                    retryable_skips = list(replay_resume_state["retryable_skips"])
                _advance_rng_draws(
                    rng=rng,
                    dataset_size=len(source),
                    draw_count=raw_draw_counter,
                )
                print(
                    json.dumps(
                        {
                            "generated_sequence": True,
                            "generation_mode": "accepted_windows_with_retryable_resampling",
                            "dataset_size": int(len(source)),
                            "seed": int(train_args.seed),
                            "rng_rank": int(rng_rank),
                            "rank_seed": int(effective_rank_seed),
                            "autograd_anomaly": bool(debug_autograd_anomaly),
                            "tensor_index_guards": bool(debug_tensor_index_guards),
                            "rng_skip_windows": int(args.rng_skip_windows),
                            "rng_num_windows": int(args.rng_num_windows),
                            "accepted_target": int(accepted_target),
                            "resume_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                            "resumed_accepted_counter": int(accepted_counter),
                            "resumed_raw_draw_counter": int(raw_draw_counter),
                            "resume_rng_state_loaded": bool(replay_rng_loaded),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                recent: deque[dict[str, Any]] = deque(maxlen=8)
                while accepted_counter < accepted_target:
                    raw_draw_counter += 1
                    flat_index = int(rng.integers(0, len(source)))
                    window = source.window(flat_index)
                    record = {
                        "replay_step": int(accepted_counter + 1),
                        "repeat": int(accepted_counter // max(int(args.rng_num_windows), 1)),
                        "flat_index": int(flat_index),
                        "segment": int(window.segment_id),
                        "start_step": int(window.start_step_id),
                        "prompt": str(window.prompt),
                        "point_grid_mode": str(args.point_grid_mode),
                        "raw_draw": int(raw_draw_counter),
                    }
                    lr = _lr_for_step(
                        accepted_counter,
                        base_lr=float(train_args.lr),
                        warmup_steps=int(train_args.warmup_steps),
                        min_lr=float(train_args.min_lr),
                        total_steps=int(train_args.num_train_steps),
                    )
                    _set_optimizer_lr(optimizer, float(lr))
                    record["lr"] = float(lr)
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        point_counts = _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
                    except RuntimeError as exc:
                        if not _is_retryable_first_step_error(exc):
                            recent.append({**record, "status": "hard_failure"})
                            print("PICF replay failure:", flush=True)
                            print(json.dumps(record, sort_keys=True), flush=True)
                            print(f"exception={type(exc).__name__}: {exc}", flush=True)
                            print(f"recent_history={list(recent)}", flush=True)
                            if _dump_debug_index_trace():
                                print(f"recent_tensor_index_trace={_dump_debug_index_trace()}", flush=True)
                            raise
                        retryable_skip_count += 1
                        skip_record = {
                            **record,
                            "status": "retryable_window_skipped",
                            "retryable_skip_count": int(retryable_skip_count),
                        }
                        recent.append(skip_record)
                        if len(retryable_skips) < 128:
                            retryable_skips.append(skip_record)
                        print(json.dumps(skip_record, sort_keys=True), flush=True)
                        continue
                    accepted_counter += 1
                    record["replay_step"] = int(accepted_counter)
                    record["repeat"] = int((accepted_counter - 1) // max(int(args.rng_num_windows), 1))
                    record["point_counts"] = tuple(int(count) for count in point_counts)
                    recent.append({**record, "status": "accepted"})
                    accepted_flat_indices.append(int(flat_index))
                    try:
                        forward_fn = lambda: trainer(window, capture_visual_diagnostics=False)
                        split_backward = (
                            args.split_backward_from_step is not None
                            and int(accepted_counter) >= int(args.split_backward_from_step)
                        )
                        pre_forward_rng_state = None
                        pre_forward_buffer_state = None
                        if split_backward or bool(args.diagnose_nonfinite_by_component):
                            pre_forward_rng_state = _capture_rng_state(device)
                            pre_forward_buffer_state = _capture_buffer_state(trainer)
                        outputs = forward_fn()
                        _backward_loss_components(
                            outputs=outputs,
                            device=device,
                            split_backward=bool(split_backward),
                            replay_step=int(accepted_counter),
                            model=trainer,
                            optimizer=optimizer,
                            forward_fn=forward_fn,
                            pre_forward_rng_state=pre_forward_rng_state,
                            pre_forward_buffer_state=pre_forward_buffer_state,
                        )
                        grad_issue = _collect_nonfinite_gradient_diagnostics(trainer, optimizer=optimizer, max_items=24)
                        if int(grad_issue["nonfinite_grad_count"]) > 0:
                            if bool(args.diagnose_nonfinite_by_component):
                                _diagnose_nonfinite_loss_components(
                                    replay_step=int(accepted_counter),
                                    device=device,
                                    model=trainer,
                                    optimizer=optimizer,
                                    forward_fn=forward_fn,
                                    pre_forward_rng_state=pre_forward_rng_state or _capture_rng_state(device),
                                    pre_forward_buffer_state=pre_forward_buffer_state or _capture_buffer_state(trainer),
                                )
                            raise RuntimeError(f"Non-finite gradients detected after backward: {grad_issue}")
                        preclip_grad_norm = _grad_norm(trainer.parameters())
                        if float(getattr(train_args, "grad_clip_norm", 0.0)) > 0.0:
                            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=float(train_args.grad_clip_norm))
                        grad_norm = _grad_norm(trainer.parameters())
                        if args.optimizer_step:
                            optimizer.step()
                            param_issue = _collect_nonfinite_parameter_diagnostics(trainer, optimizer=optimizer, max_items=24)
                            if int(param_issue["nonfinite_param_count"]) > 0:
                                raise RuntimeError(f"Non-finite parameters detected after optimizer step: {param_issue}")
                        if (
                            save_checkpoint_dir is not None
                            and int(args.save_checkpoint_every) > 0
                            and (int(accepted_counter) % int(args.save_checkpoint_every) == 0)
                        ):
                            checkpoint_dir = _checkpoint_dir_for_step(save_checkpoint_dir, int(accepted_counter))
                            _save_checkpoint(
                                output_dir=save_checkpoint_dir,
                                model=trainer,
                                optimizer=optimizer,
                                step=int(accepted_counter),
                                args=train_args,
                            )
                            _save_replay_state(
                                checkpoint_dir=checkpoint_dir,
                                accepted_counter=int(accepted_counter),
                                raw_draw_counter=int(raw_draw_counter),
                                retryable_skip_count=int(retryable_skip_count),
                                accepted_flat_indices=accepted_flat_indices,
                                retryable_skips=retryable_skips,
                            )
                            _save_replay_rng_state(checkpoint_dir=checkpoint_dir, device=device)
                            _prune_old_replay_checkpoints(
                                checkpoint_root=save_checkpoint_dir,
                                keep=int(args.max_checkpoints),
                            )
                        print(
                            json.dumps(
                                {
                                    **record,
                                    "grad_norm": float(grad_norm),
                                    "preclip_grad_norm": float(preclip_grad_norm),
                                    **_scalar_loss_snapshot(outputs),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if args.stop_after_step is not None and int(accepted_counter) >= int(args.stop_after_step):
                            if args.dump_generated_sequence is not None:
                                dump_path = Path(args.dump_generated_sequence)
                                dump_path.parent.mkdir(parents=True, exist_ok=True)
                                dump_path.write_text(
                                    json.dumps(
                                        {
                                            "accepted_flat_indices": accepted_flat_indices,
                                            "retryable_skips": retryable_skips,
                                            "raw_draws_consumed": int(raw_draw_counter),
                                        }
                                    ),
                                    encoding="utf-8",
                                )
                            print(
                                json.dumps(
                                    {
                                        "stage": "stop_after_reached",
                                        "replay_step": int(accepted_counter),
                                        "raw_draws_consumed": int(raw_draw_counter),
                                        "retryable_skip_count": int(retryable_skip_count),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            return
                    except Exception as exc:
                        print("PICF replay failure:", flush=True)
                        print(json.dumps(record, sort_keys=True), flush=True)
                        print(f"exception={type(exc).__name__}: {exc}", flush=True)
                        print(f"recent_history={list(recent)}", flush=True)
                        if _dump_debug_index_trace():
                            print(f"recent_tensor_index_trace={_dump_debug_index_trace()}", flush=True)
                        raise
                if args.dump_generated_sequence is not None:
                    dump_path = Path(args.dump_generated_sequence)
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_path.write_text(
                        json.dumps(
                            {
                                "accepted_flat_indices": accepted_flat_indices,
                                "retryable_skips": retryable_skips,
                                "raw_draws_consumed": int(raw_draw_counter),
                            }
                        ),
                        encoding="utf-8",
                    )
                print(
                    json.dumps(
                        {
                            "stage": "rng_sequence_complete",
                            "accepted_windows": int(accepted_counter),
                            "raw_draws_consumed": int(raw_draw_counter),
                            "retryable_skip_count": int(retryable_skip_count),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return
            elif args.flat_indices is not None:
                flat_indices = _parse_flat_indices(args.flat_indices)
            else:
                raise ValueError("Pass either --flat-indices or --rng-num-windows.")

            recent: deque[dict[str, Any]] = deque(maxlen=8)
            step_counter = 0
            for repeat_index in range(int(args.repeat)):
                for flat_index in flat_indices:
                    step_counter += 1
                    window = source.window(flat_index)
                    record = {
                        "replay_step": int(step_counter),
                        "repeat": int(repeat_index),
                        "flat_index": int(flat_index),
                        "segment": int(window.segment_id),
                        "start_step": int(window.start_step_id),
                        "prompt": str(window.prompt),
                        "point_grid_mode": str(args.point_grid_mode),
                    }
                    lr = _lr_for_step(
                        step_counter - 1,
                        base_lr=float(train_args.lr),
                        warmup_steps=int(train_args.warmup_steps),
                        min_lr=float(train_args.min_lr),
                        total_steps=int(train_args.num_train_steps),
                    )
                    _set_optimizer_lr(optimizer, float(lr))
                    record["lr"] = float(lr)
                    recent.append(record)
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
                        forward_fn = lambda: trainer(window, capture_visual_diagnostics=False)
                        split_backward = (
                            args.split_backward_from_step is not None
                            and int(step_counter) >= int(args.split_backward_from_step)
                        )
                        pre_forward_rng_state = None
                        pre_forward_buffer_state = None
                        if split_backward or bool(args.diagnose_nonfinite_by_component):
                            pre_forward_rng_state = _capture_rng_state(device)
                            pre_forward_buffer_state = _capture_buffer_state(trainer)
                        outputs = forward_fn()
                        _backward_loss_components(
                            outputs=outputs,
                            device=device,
                            split_backward=bool(split_backward),
                            replay_step=int(step_counter),
                            model=trainer,
                            optimizer=optimizer,
                            forward_fn=forward_fn,
                            pre_forward_rng_state=pre_forward_rng_state,
                            pre_forward_buffer_state=pre_forward_buffer_state,
                        )
                        grad_issue = _collect_nonfinite_gradient_diagnostics(trainer, optimizer=optimizer, max_items=24)
                        if int(grad_issue["nonfinite_grad_count"]) > 0:
                            if bool(args.diagnose_nonfinite_by_component):
                                _diagnose_nonfinite_loss_components(
                                    replay_step=int(step_counter),
                                    device=device,
                                    model=trainer,
                                    optimizer=optimizer,
                                    forward_fn=forward_fn,
                                    pre_forward_rng_state=pre_forward_rng_state or _capture_rng_state(device),
                                    pre_forward_buffer_state=pre_forward_buffer_state or _capture_buffer_state(trainer),
                                )
                            raise RuntimeError(f"Non-finite gradients detected after backward: {grad_issue}")
                        preclip_grad_norm = _grad_norm(trainer.parameters())
                        if float(getattr(train_args, "grad_clip_norm", 0.0)) > 0.0:
                            torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=float(train_args.grad_clip_norm))
                        grad_norm = _grad_norm(trainer.parameters())
                        if args.optimizer_step:
                            optimizer.step()
                            param_issue = _collect_nonfinite_parameter_diagnostics(trainer, optimizer=optimizer, max_items=24)
                            if int(param_issue["nonfinite_param_count"]) > 0:
                                raise RuntimeError(f"Non-finite parameters detected after optimizer step: {param_issue}")
                        if (
                            save_checkpoint_dir is not None
                            and int(args.save_checkpoint_every) > 0
                            and (int(step_counter) % int(args.save_checkpoint_every) == 0)
                        ):
                            checkpoint_dir = _checkpoint_dir_for_step(save_checkpoint_dir, int(step_counter))
                            _save_checkpoint(
                                output_dir=save_checkpoint_dir,
                                model=trainer,
                                optimizer=optimizer,
                                step=int(step_counter),
                                args=train_args,
                            )
                            _save_replay_rng_state(checkpoint_dir=checkpoint_dir, device=device)
                            _prune_old_replay_checkpoints(
                                checkpoint_root=save_checkpoint_dir,
                                keep=int(args.max_checkpoints),
                            )
                        print(
                            json.dumps(
                                {
                                    **record,
                                    "grad_norm": float(grad_norm),
                                    "preclip_grad_norm": float(preclip_grad_norm),
                                    **_scalar_loss_snapshot(outputs),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if args.stop_after_step is not None and int(step_counter) >= int(args.stop_after_step):
                            print(
                                json.dumps(
                                    {
                                        "stage": "stop_after_reached",
                                        "replay_step": int(step_counter),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            return
                    except Exception as exc:
                        print("PICF replay failure:", flush=True)
                        print(json.dumps(record, sort_keys=True), flush=True)
                        print(f"exception={type(exc).__name__}: {exc}", flush=True)
                        print(f"recent_history={list(recent)}", flush=True)
                        if _dump_debug_index_trace():
                            print(f"recent_tensor_index_trace={_dump_debug_index_trace()}", flush=True)
                        raise
        finally:
            source.close()


if __name__ == "__main__":
    main()
