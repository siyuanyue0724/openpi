from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import torch

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay specific PICF CALVIN transition windows through the current training graph.")
    parser.add_argument("--args-json", required=True, help="Path to args.json from a PICF training run.")
    parser.add_argument("--flat-indices", required=True, help="Comma-separated list of flat indices to replay in order.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the flat-index sequence.")
    parser.add_argument("--device", default=None, help="Override device from args.json, e.g. cuda or cpu.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint directory or latest.pt to load before replay.")
    parser.add_argument("--rank-seed", type=int, default=1, help="Rank offset for seed reproduction. rank=1 matches prior failing worker.")
    parser.add_argument("--optimizer-step", action="store_true", help="Apply optimizer.step() after each replayed window.")
    args = parser.parse_args()

    args_json_path = Path(args.args_json)
    payload = json.loads(args_json_path.read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=args.device)
    device = torch.device(str(train_args.device))
    flat_indices = _parse_flat_indices(args.flat_indices)
    if int(args.repeat) < 1:
        raise ValueError(f"--repeat must be >= 1, got {args.repeat}.")

    _seed_everything(int(train_args.seed), int(args.rank_seed))

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
