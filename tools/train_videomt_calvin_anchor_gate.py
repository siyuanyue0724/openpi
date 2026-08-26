#!/usr/bin/env python3
"""Run a falsifiable real-CALVIN gate for the exact VidEoMT transplant."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from picf_next.videomt_exact.calvin_dataset import (
    HashBoundCalvinFrameStore,
    build_calvin_videomt_split_plan,
)
from picf_next.videomt_exact.calvin_targets import (
    VIDEOMT_YTVIS19_TRAIN_MAX_SIZE,
    VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES,
    prepare_calvin_videomt_training_clip,
)
from picf_next.videomt_exact.class_agnostic import (
    VIDEOMT_MATCHER_IDENTITIES,
    VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    build_class_agnostic_criterion,
    flatten_class_agnostic_outputs,
    flatten_class_agnostic_targets,
)
from picf_next.videomt_exact.evaluation import (
    evaluate_calvin_anchor_windows,
)
from picf_next.videomt_exact.optimizer import (
    VIDEOMT_ADAPTATION_BUDGET_STEPS,
    VIDEOMT_RELEASED_TOTAL_STEPS,
    build_exact_videomt_optimizer,
    build_exact_videomt_scheduler,
    optimizer_group_learning_rates,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt
from picf_next.videomt_exact.training import apply_released_loss_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--source-split-root", required=True, type=Path)
    parser.add_argument("--source-overlay-root", type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--golden-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--budget-steps", type=int, default=VIDEOMT_ADAPTATION_BUDGET_STEPS)
    parser.add_argument("--eval-steps", default="0,50,100,250")
    parser.add_argument("--eval-clips", type=int, default=4)
    parser.add_argument(
        "--train-short-edges",
        default=",".join(str(value) for value in VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES),
    )
    parser.add_argument(
        "--train-max-size",
        type=int,
        default=VIDEOMT_YTVIS19_TRAIN_MAX_SIZE,
    )
    parser.add_argument("--eval-short-edge", type=int, default=480)
    parser.add_argument("--seed", type=int, default=198)
    parser.add_argument(
        "--matcher-identity",
        choices=VIDEOMT_MATCHER_IDENTITIES,
        default=VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    )
    parser.add_argument("--save-final-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        choices=("released-amp-fp16", "fp32-debug"),
        default="released-amp-fp16",
    )
    return parser.parse_args()


def _parse_positive_ints(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} contains an invalid value")
    return values


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selected_windows(
    windows: tuple[tuple[int, ...], ...],
    count: int,
) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        raise ValueError("eval-clips must be positive")
    selected = np.linspace(0, len(windows) - 1, min(count, len(windows)), dtype=np.int64)
    return tuple(windows[int(index)] for index in np.unique(selected))


@torch.no_grad()
def _evaluate(
    *,
    runtime: torch.nn.Module,
    store: HashBoundCalvinFrameStore,
    windows: tuple[tuple[int, ...], ...],
    short_edge: int,
    device: torch.device,
    dtype: torch.dtype,
    released_amp: bool,
    panel_path: Path,
) -> dict[str, object]:
    autocast = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if released_amp
        else contextlib.nullcontext()
    )
    with autocast:
        return evaluate_calvin_anchor_windows(
            runtime=runtime,
            store=store,
            windows=windows,
            short_edge=short_edge,
            device=device,
            dtype=dtype,
            panel_path=panel_path,
        )


def _gradient_receipt(model: torch.nn.Module) -> dict[str, float | int]:
    tensors = 0
    nonzero = 0
    square_sum = 0.0
    maximum = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        tensors += 1
        gradient = parameter.grad.detach().float()
        if torch.count_nonzero(gradient):
            nonzero += 1
        square_sum += float(gradient.square().sum())
        maximum = max(maximum, float(gradient.abs().max()))
    return {
        "gradient_tensors": tensors,
        "nonzero_gradient_tensors": nonzero,
        "l2_norm": square_sum**0.5,
        "max_abs": maximum,
    }


def main() -> None:
    args = parse_args()
    train_short_edges = _parse_positive_ints(args.train_short_edges, name="train-short-edges")
    if any(value <= 0 for value in train_short_edges):
        raise ValueError("train-short-edges must contain only positive values")
    eval_steps = set(_parse_positive_ints(args.eval_steps, name="eval-steps"))
    if args.steps <= 0 or args.budget_steps < args.steps or max(eval_steps) > args.steps:
        raise ValueError("steps, budget-steps, and eval-steps are inconsistent")
    if 0 not in eval_steps:
        raise ValueError("eval-steps must include the immutable step-0 baseline")
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    if args.precision == "released-amp-fp16" and device.type != "cuda":
        raise ValueError("released VidEoMT AMP requires a CUDA device")
    released_amp = args.precision == "released-amp-fp16"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    started = time.perf_counter()
    store = HashBoundCalvinFrameStore(
        source_split_root=args.source_split_root,
        sidecar_root=args.sidecar_root,
        source_overlay_root=args.source_overlay_root,
    )
    source_rgb_audit = store.audit_source_rgb()
    split_plan = build_calvin_videomt_split_plan(
        golden_manifest_path=args.golden_manifest,
        store=store,
    )
    train_eval_windows = _selected_windows(split_plan.train_windows, args.eval_clips)
    heldout_eval_windows = _selected_windows(split_plan.heldout_windows, args.eval_clips)

    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.checkpoint,
            local_dinov3_bundle=args.dinov3_bundle,
            num_frames=split_plan.clip_length,
        ),
        device=device,
        dtype=torch.float32,
    )
    criterion = build_class_agnostic_criterion(
        matcher_identity=args.matcher_identity,
        num_frames=split_plan.clip_length,
    ).to(device)
    optimizer, optimizer_receipt = build_exact_videomt_optimizer(runtime.model)
    scheduler = build_exact_videomt_scheduler(
        optimizer,
        optimizer_receipt,
        total_steps=args.budget_steps,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"
    full_recipe_deviations = [
        "CALVIN replaces the released COCO/YTVIS dataset mixture",
        "one clip replaces released effective batch eight",
    ]
    if args.budget_steps != VIDEOMT_RELEASED_TOTAL_STEPS:
        full_recipe_deviations.append("CALVIN budget replaces the released 160k horizon")
    selected_online_component_contract_exact = bool(
        train_short_edges == VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES
        and args.train_max_size == VIDEOMT_YTVIS19_TRAIN_MAX_SIZE
        and args.matcher_identity == VIDEOMT_ONLINE_CONSISTENT_MATCHER
        and released_amp
    )
    report: dict[str, object] = {
        "schema": "picf-next.videomt-exact-calvin-anchor-gate.v2",
        "claim_scope": (
            "real hash-bound CALVIN five-frame single-clip adaptation with segment-disjoint "
            "but not episode-disjoint heldout; not full selected-recipe, PICF, or action acceptance"
        ),
        "source_contract": {
            "sidecar_manifest_sha256": store.manifest_sha256,
            "golden_manifest_sha256": split_plan.golden_manifest_sha256,
            "source_frame_count": len(store.global_indices),
            "train_window_count": len(split_plan.train_windows),
            "heldout_window_count": len(split_plan.heldout_windows),
            "episode_disjoint": split_plan.episode_disjoint,
            "source_rgb_audit": source_rgb_audit,
            "components": [asdict(value) for value in split_plan.components],
        },
        "recipe": {
            "steps": args.steps,
            "budget_steps": args.budget_steps,
            "released_total_steps": VIDEOMT_RELEASED_TOTAL_STEPS,
            "schedule_identity": (
                "released-160k"
                if args.budget_steps == VIDEOMT_RELEASED_TOTAL_STEPS
                else "calvin-explicit-adaptation"
            ),
            "released_effective_batch_clips": 8,
            "effective_batch_clips": 1,
            "eval_steps": sorted(eval_steps),
            "train_short_edges": list(train_short_edges),
            "train_max_size": args.train_max_size,
            "eval_short_edge": args.eval_short_edge,
            "official_ytvis19_spatial_augmentation": (
                train_short_edges == VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES
                and args.train_max_size == VIDEOMT_YTVIS19_TRAIN_MAX_SIZE
            ),
            "clip_length": split_plan.clip_length,
            "matcher_identity": args.matcher_identity,
            "selected_online_matcher_exact": (
                args.matcher_identity == VIDEOMT_ONLINE_CONSISTENT_MATCHER
            ),
            "released_amp_semantics": released_amp,
            "adamw_zero_lr_moment_state_preserved": True,
            "selected_online_component_contract_exact": (
                selected_online_component_contract_exact
            ),
            "full_selected_training_recipe_exact": False,
            "full_recipe_deviations": full_recipe_deviations,
            "seed": args.seed,
            "precision": args.precision,
            "parameter_dtype": "torch.float32",
            "autocast_dtype": "torch.float16" if released_amp else None,
            "device": str(device),
        },
        "optimizer": asdict(optimizer_receipt),
        "evaluations": {},
        "training": [],
        "failures": [],
        "passed_engineering_gate": False,
        "approved_for_picf_integration": False,
    }

    def run_evaluation(step: int) -> None:
        evaluations = report["evaluations"]
        assert isinstance(evaluations, dict)
        evaluations[str(step)] = {
            "train": _evaluate(
                runtime=runtime,
                store=store,
                windows=train_eval_windows,
                short_edge=args.eval_short_edge,
                device=device,
                dtype=torch.float32,
                released_amp=released_amp,
                panel_path=output_dir / "visuals" / f"train_step{step:04d}.png",
            ),
            "heldout": _evaluate(
                runtime=runtime,
                store=store,
                windows=heldout_eval_windows,
                short_edge=args.eval_short_edge,
                device=device,
                dtype=torch.float32,
                released_amp=released_amp,
                panel_path=output_dir / "visuals" / f"heldout_step{step:04d}.png",
            ),
        }
        _atomic_json(output_dir / "report.json", report)

    run_evaluation(0)
    generator = random.Random(args.seed)
    runtime.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=released_amp)
    with log_path.open("w", encoding="utf-8") as log_file:
        for optimizer_step in range(args.steps):
            step_learning_rates = optimizer_group_learning_rates(optimizer)
            window = split_plan.train_windows[generator.randrange(len(split_plan.train_windows))]
            source = store.clip(window)
            clip = prepare_calvin_videomt_training_clip(
                source.rgb_static,
                source.supervision,
                short_edges=train_short_edges,
                max_size=args.train_max_size,
            )
            targets = [
                {name: value.to(device) for name, value in clip.target.items()}
            ]
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            autocast = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if released_amp
                else contextlib.nullcontext()
            )
            with autocast:
                output = runtime(clip.frames.model_input.to(device=device, dtype=torch.float32))
                flat_outputs = flatten_class_agnostic_outputs(output)
                flat_targets = flatten_class_agnostic_targets(targets)
                raw_losses = criterion(flat_outputs, flat_targets)
                weighted_losses = apply_released_loss_weights(raw_losses, criterion)
                total_loss = sum(weighted_losses.values())
            if not weighted_losses or not torch.isfinite(total_loss):
                raise RuntimeError(f"non-finite or empty VidEoMT loss at step {optimizer_step}")
            grad_scaler.scale(total_loss).backward()
            grad_scaler.unscale_(optimizer)
            gradients = _gradient_receipt(runtime.model)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            completed_step = optimizer_step + 1
            step_record = {
                "step": completed_step,
                "global_indices": list(window),
                "padded_size": list(clip.frames.padded_size),
                "backbone_trainable": all(
                    parameter.requires_grad
                    for parameter in runtime.model.encoder.backbone.parameters()
                ),
                "backbone_learning_rate_zero": all(
                    value == 0.0
                    for value in step_learning_rates[
                        : optimizer_receipt.backbone_parameter_group_count
                    ]
                ),
                "loss": float(total_loss.detach()),
                "weighted_losses": {
                    name: float(value.detach()) for name, value in weighted_losses.items()
                },
                "gradient_receipt": gradients,
                "learning_rate_min": min(step_learning_rates),
                "learning_rate_max": max(step_learning_rates),
                "seconds": time.perf_counter() - step_started,
            }
            training = report["training"]
            assert isinstance(training, list)
            training.append(step_record)
            log_file.write(json.dumps(step_record) + "\n")
            log_file.flush()
            if completed_step in eval_steps:
                run_evaluation(completed_step)
                runtime.train()

    evaluations = report["evaluations"]
    assert isinstance(evaluations, dict)
    initial = evaluations["0"]["heldout"]
    final = evaluations[str(args.steps)]["heldout"]
    heldout_improvement = float(final["mean_soft_iou"]) - float(initial["mean_soft_iou"])
    report["heldout_mean_soft_iou_improvement"] = heldout_improvement
    report["passed_engineering_gate"] = bool(
        heldout_improvement > 0.02
        and float(final["mean_binary_iou"]) > 0.15
        and float(final["recall_at_50"]) > 0.0
    )
    report["approved_for_picf_integration"] = bool(
        report["passed_engineering_gate"]
        and report["recipe"]["selected_online_component_contract_exact"]
        and report["recipe"]["official_ytvis19_spatial_augmentation"]
        and report["recipe"]["selected_online_matcher_exact"]
        and split_plan.episode_disjoint
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_cuda_allocated_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    if args.save_final_checkpoint:
        checkpoint_path = output_dir / "model_final.pt"
        torch.save(runtime.model.state_dict(), checkpoint_path)
        report["final_checkpoint"] = str(checkpoint_path)
    _atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
