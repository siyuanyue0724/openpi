from __future__ import annotations

import argparse
import contextlib
import json
from collections import deque
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.picf_core_train import _build_model
from scripts.picf_core_train import _build_optimizer
from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_core_train import _ensure_window_has_valid_first_step_xyzrgb_support
from scripts.picf_core_train import _load_checkpoint
from scripts.picf_core_train import _materialize_model_parameters
from scripts.picf_core_train import _normalize_train_args
from scripts.picf_core_train import _PicfWindowTrainer
from scripts.picf_core_train import _seed_everything
from scripts.picf_core_train import _validate_train_args
from scripts.sonata_window_probe import _override_build_sample


def _coerce_loaded_args(payload: dict[str, Any], *, device_override: str | None) -> argparse.Namespace:
    args = argparse.Namespace(**payload)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay specific PICF CALVIN transition windows through the current training graph.")
    parser.add_argument("--args-json", required=True, help="Path to args.json from a PICF training run.")
    parser.add_argument("--flat-indices", default=None, help="Comma-separated list of flat indices to replay in order.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the flat-index sequence.")
    parser.add_argument("--device", default=None, help="Override device from args.json, e.g. cuda or cpu.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint directory or latest.pt to load before replay.")
    parser.add_argument("--rank-seed", type=int, default=1, help="Rank offset for seed reproduction. rank=1 matches prior failing worker.")
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
    args = parser.parse_args()

    args_json_path = Path(args.args_json)
    payload = json.loads(args_json_path.read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=args.device)
    device = torch.device(str(train_args.device))
    if int(args.repeat) < 1:
        raise ValueError(f"--repeat must be >= 1, got {args.repeat}.")

    _seed_everything(int(train_args.seed), int(args.rank_seed))

    override_context = contextlib.nullcontext() if args.point_grid_mode == "default" else _override_build_sample(str(args.point_grid_mode))
    with override_context:
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
            core, semantic_encoder, use_visual_override = _build_model(train_args, device=device)
            core = core.to(device)
            trainer = _PicfWindowTrainer(
                core,
                semantic_encoder=semantic_encoder,
                visual_grid=train_args.visual_grid,
                use_visual_override=use_visual_override,
            ).to(device)
            _materialize_model_parameters(trainer, source=source, rank=int(args.rank_seed))
            optimizer, _ = _build_optimizer(trainer, args=train_args)
            if args.checkpoint:
                _load_checkpoint(path=Path(args.checkpoint), model=trainer, optimizer=optimizer, device=device)
            trainer.train()

            if args.rng_num_windows is not None:
                rng_rank = int(args.rng_rank if args.rng_rank is not None else args.rank_seed)
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
                        if device.type == "cuda":
                            torch.cuda.synchronize(device=device)
                        outputs["loss_total"].backward()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device=device)
                        if args.optimizer_step:
                            optimizer.step()
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
                    except Exception as exc:
                        print("PICF replay failure:", flush=True)
                        print(json.dumps(record, sort_keys=True), flush=True)
                        print(f"exception={type(exc).__name__}: {exc}", flush=True)
                        print(f"recent_history={list(recent)}", flush=True)
                        raise
        finally:
            source.close()


if __name__ == "__main__":
    main()
