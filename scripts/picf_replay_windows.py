from __future__ import annotations

import argparse
import contextlib
import json
from collections import deque
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _build_optimizer
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _dump_debug_index_trace
from scripts.picf_core_train import _ensure_window_has_valid_first_step_xyzrgb_support
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _normalize_train_args
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _save_checkpoint
from scripts.picf_core_train import _install_debug_tensor_index_guards
from scripts.picf_core_train import _seed_everything
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


def _backward_loss_components(
    *,
    outputs: dict[str, torch.Tensor],
    device: torch.device,
    split_backward: bool,
    replay_step: int,
) -> None:
    if not split_backward:
        _synchronize_if_cuda(device)
        outputs["loss_total"].backward()
        _synchronize_if_cuda(device)
        return

    ordered_keys = [
        "loss_action",
        "loss_visual_latent",
        "loss_visual_real",
        "loss_tactile_real",
        "loss_point_real",
        "loss_semantic_future_aux",
        "loss_alignment",
    ]
    active = [key for key in ordered_keys if key in outputs and outputs[key] is not None]
    if not active:
        raise RuntimeError("No backward components found in replay outputs.")
    for index, key in enumerate(active):
        retain_graph = index < (len(active) - 1)
        print(
            json.dumps(
                {
                    "stage": "component_backward_start",
                    "replay_step": int(replay_step),
                    "component": str(key),
                    "retain_graph": bool(retain_graph),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _synchronize_if_cuda(device)
        outputs[key].backward(retain_graph=retain_graph)
        _synchronize_if_cuda(device)
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
            if args.checkpoint:
                _load_checkpoint(path=Path(args.checkpoint), model=trainer, optimizer=optimizer, device=device)
            trainer.train()
            if debug_tensor_index_guards:
                _install_debug_tensor_index_guards()
            print(json.dumps({"stage": "entering_replay_loop", "elapsed_s": round(time.time() - start_time, 3)}), flush=True)

            if args.rng_num_windows is not None:
                rng_rank = int(args.rng_rank if args.rng_rank is not None else effective_rank_seed)
                flat_indices = _generate_rng_flat_indices(
                    total_windows=int(args.rng_num_windows),
                    dataset_size=len(source),
                    seed=int(train_args.seed),
                    rank=rng_rank,
                    skip_windows=int(args.rng_skip_windows),
                )
                if args.dump_generated_sequence is not None:
                    dump_path = Path(args.dump_generated_sequence)
                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                    dump_path.write_text(json.dumps(flat_indices), encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "generated_sequence": True,
                            "dataset_size": int(len(source)),
                            "seed": int(train_args.seed),
                            "rng_rank": int(rng_rank),
                            "rank_seed": int(effective_rank_seed),
                            "autograd_anomaly": bool(debug_autograd_anomaly),
                            "tensor_index_guards": bool(debug_tensor_index_guards),
                            "rng_skip_windows": int(args.rng_skip_windows),
                            "rng_num_windows": int(args.rng_num_windows),
                            "first_indices": flat_indices[: min(16, len(flat_indices))],
                            "last_indices": flat_indices[-min(16, len(flat_indices)) :],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
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
                    recent.append(record)
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
                        outputs = trainer(window, capture_visual_diagnostics=False)
                        split_backward = (
                            args.split_backward_from_step is not None
                            and int(step_counter) >= int(args.split_backward_from_step)
                        )
                        _backward_loss_components(
                            outputs=outputs,
                            device=device,
                            split_backward=bool(split_backward),
                            replay_step=int(step_counter),
                        )
                        if args.optimizer_step:
                            optimizer.step()
                        if (
                            save_checkpoint_dir is not None
                            and int(args.save_checkpoint_every) > 0
                            and (int(step_counter) % int(args.save_checkpoint_every) == 0)
                        ):
                            _save_checkpoint(
                                output_dir=save_checkpoint_dir,
                                model=trainer,
                                optimizer=optimizer,
                                step=int(step_counter),
                                args=train_args,
                            )
                        print(
                            json.dumps(
                                {
                                    **record,
                                    "loss_total": float(outputs["loss_total"].detach().item()),
                                    "loss_alignment": float(outputs["loss_alignment"].detach().item()),
                                    "loss_pt": float(outputs["loss_pt"].detach().item()),
                                    "projective_candidate_density": float(outputs["projective_candidate_density"].detach().item()),
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
