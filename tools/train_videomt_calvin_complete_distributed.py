#!/usr/bin/env python3
"""Train the complete VidEoMT donor on all-source CALVIN supervision."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.videomt_exact.calvin_full_dataset import (
    CalvinVidEoMTEpisodeSplitPlan,
    CalvinVidEoMTSplit,
    build_calvin_videomt_episode_split_plan,
    materialize_calvin_videomt_clip,
    stateless_calvin_videomt_window,
)
from picf_next.videomt_exact.calvin_targets import (
    VIDEOMT_YTVIS19_TRAIN_MAX_SIZE,
    VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES,
    PreparedCalvinVidEoMTClip,
    prepare_calvin_videomt_training_clip,
)
from picf_next.videomt_exact.checkpoint import sha256_file
from picf_next.videomt_exact.class_agnostic import (
    VIDEOMT_ONLINE_CONSISTENT_MATCHER,
    build_class_agnostic_criterion,
    flatten_class_agnostic_outputs,
    flatten_class_agnostic_targets,
)
from picf_next.videomt_exact.distributed_training import (
    VidEoMTEffectiveBatchReceipt,
    make_effective_batch_receipt,
    scale_videomt_microstep_losses,
)
from picf_next.videomt_exact.evaluation import evaluate_calvin_anchor_windows
from picf_next.videomt_exact.optimizer import (
    VIDEOMT_ADAPTATION_BUDGET_STEPS,
    build_exact_videomt_optimizer,
    build_exact_videomt_scheduler,
    optimizer_group_learning_rates,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt
from picf_next.videomt_exact.training import apply_released_loss_weights

REPORT_SCHEMA = "picf-next.videomt-complete-distributed-calvin/v1"
CHECKPOINT_SCHEMA = "picf-next.videomt-complete-distributed-checkpoint/v1"
EXPECTED_WORLD_SIZE = 2
DEFAULT_ACCUMULATION_STEPS = 4
EXPECTED_COMPLETE_MODEL_STATE_NUMEL = 315_986_989
EXPECTED_COMPLETE_PARAMETER_NUMEL = 315_986_985
IMPLEMENTATION_ROOTS = (
    "src/picf_next/videomt_exact",
    "src/picf_next/_vendor/videomt",
)


class _EvaluationStore:
    def __init__(
        self,
        index: CalvinDatasetIndex,
        sidecar: CalvinPhysicalSupervisionSidecar,
    ) -> None:
        self.index = index
        self.sidecar = sidecar

    def clip(self, global_indices: tuple[int, ...]) -> Any:
        return materialize_calvin_videomt_clip(self.index, self.sidecar, global_indices)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--dataset-split", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--physical-sidecar-root", required=True, type=Path)
    parser.add_argument("--physical-sidecar-manifest", required=True, type=Path)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--budget-steps", type=int, default=VIDEOMT_ADAPTATION_BUDGET_STEPS)
    parser.add_argument("--eval-steps", default="0,50,100,250")
    parser.add_argument("--eval-clips", type=int, default=8)
    parser.add_argument("--eval-short-edge", type=int, default=480)
    parser.add_argument("--accumulation-steps", type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument("--heldout-modulus", type=int, default=5)
    parser.add_argument("--heldout-remainder", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def _parse_steps(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    except ValueError as error:
        raise ValueError("eval steps must be comma-separated integers") from error
    if not values or values[0] < 0:
        raise ValueError("eval steps must be non-negative")
    return values


def _sha256(path: Path) -> str:
    return sha256_file(path.expanduser().resolve())


def _implementation_receipt(tool_path: Path) -> dict[str, object]:
    repository = tool_path.resolve().parents[1]
    paths = [tool_path.resolve()]
    for relative_root in IMPLEMENTATION_ROOTS:
        paths.extend(sorted((repository / relative_root).rglob("*.py")))
    files = []
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(repository).as_posix()
        file_sha256 = _sha256(path)
        files.append({"path": relative, "sha256": file_sha256, "bytes": path.stat().st_size})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "identity": "picf-complete-videomt-python-sources/v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _microstep_seed(base_seed: int, global_clip_visit: int, *, phase: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{global_clip_visit}:{phase}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _selected_windows(
    plan: CalvinVidEoMTEpisodeSplitPlan,
    split: CalvinVidEoMTSplit,
    count: int,
) -> tuple[tuple[int, ...], ...]:
    available = plan.window_count(split)
    if count <= 0:
        raise ValueError("eval-clips must be positive")
    positions = np.linspace(0, available - 1, min(count, available), dtype=np.int64)
    return tuple(
        plan.window_at(split, int(position))
        for position in np.unique(positions)
    )


def _distributed_error(local_error: BaseException | None) -> tuple[str, ...]:
    payload = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    gathered: list[str | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return tuple(value for value in gathered if value is not None)


def _gradient_inventory(model: torch.nn.Module) -> dict[str, object]:
    missing: list[str] = []
    zero: list[str] = []
    nonfinite: list[str] = []
    square_sum = 0.0
    maximum = 0.0
    tensors = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        tensors += 1
        value = gradient.detach().float()
        if not torch.isfinite(value).all():
            nonfinite.append(name)
            continue
        if not torch.count_nonzero(value):
            zero.append(name)
        square_sum += float(value.square().sum())
        maximum = max(maximum, float(value.abs().max()))
    return {
        "gradient_tensors": tensors,
        "missing_gradient_names": missing,
        "zero_gradient_names": zero,
        "nonfinite_gradient_names": nonfinite,
        "l2_norm": square_sum**0.5,
        "max_abs": maximum,
        "all_trainable_tensors_reached": not missing and not zero and not nonfinite,
    }


def _reduce_float_mapping(values: Mapping[str, float], device: torch.device) -> dict[str, float]:
    names = tuple(sorted(values))
    tensor = torch.tensor([values[name] for name in names], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return {name: float(value) for name, value in zip(names, tensor.tolist(), strict=True)}


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng_state(value: Mapping[str, object]) -> None:
    random.setstate(cast(tuple[Any, ...], value["python"]))
    np.random.set_state(cast(tuple[Any, ...], value["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, value["torch_cpu"]))
    torch.cuda.set_rng_state(cast(torch.Tensor, value["torch_cuda"]))


def _evaluate(
    *,
    runtime: torch.nn.Module,
    store: _EvaluationStore,
    windows: tuple[tuple[int, ...], ...],
    short_edge: int,
    device: torch.device,
    panel_path: Path,
) -> dict[str, object]:
    runtime.eval()
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            return evaluate_calvin_anchor_windows(
                runtime=runtime,  # type: ignore[arg-type]
                store=store,  # type: ignore[arg-type]
                windows=windows,
                short_edge=short_edge,
                device=device,
                dtype=torch.float32,
                panel_path=panel_path,
            )
    finally:
        runtime.reset_state()


def _checkpoint_payload(
    *,
    runtime: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    global_step: int,
    plan: CalvinVidEoMTEpisodeSplitPlan,
    implementation_sha256: str,
    rank_rng_states: tuple[dict[str, object], ...],
) -> dict[str, object]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "global_step": global_step,
        "split_plan_sha256": plan.fingerprint,
        "implementation_sha256": implementation_sha256,
        "model": runtime.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rank_rng_states": rank_rng_states,
    }


def _save_checkpoint(
    *,
    output_dir: Path,
    runtime: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    global_step: int,
    plan: CalvinVidEoMTEpisodeSplitPlan,
    implementation_sha256: str,
    rank_rng_states: tuple[dict[str, object], ...],
) -> dict[str, object]:
    checkpoint_path = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    with temporary.open("wb") as stream:
        torch.save(
            _checkpoint_payload(
                runtime=runtime,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                plan=plan,
                implementation_sha256=implementation_sha256,
                rank_rng_states=rank_rng_states,
            ),
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, checkpoint_path)
    directory = os.open(checkpoint_path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "global_step": global_step,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
    }


def _load_checkpoint(
    path: Path,
    *,
    runtime: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    plan: CalvinVidEoMTEpisodeSplitPlan,
    implementation_sha256: str,
    rank: int,
    world_size: int,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("complete VidEoMT checkpoint schema differs")
    if payload.get("split_plan_sha256") != plan.fingerprint:
        raise ValueError("complete VidEoMT checkpoint uses another split plan")
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError("complete VidEoMT checkpoint uses another implementation")
    runtime.model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    rank_rng_states = payload.get("rank_rng_states")
    if not isinstance(rank_rng_states, tuple) or len(rank_rng_states) != world_size:
        raise ValueError("complete VidEoMT checkpoint rank RNG inventory is invalid")
    rank_rng_state = rank_rng_states[rank]
    if not isinstance(rank_rng_state, Mapping):
        raise ValueError("complete VidEoMT checkpoint rank RNG state is invalid")
    _restore_rng_state(rank_rng_state)
    global_step = payload.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("complete VidEoMT checkpoint step is invalid")
    return global_step


def main() -> None:
    args = _parse_args()
    eval_steps = _parse_steps(args.eval_steps)
    if (
        args.steps <= 0
        or args.steps > args.budget_steps
        or eval_steps[0] != 0
        or eval_steps[-1] > args.steps
        or args.accumulation_steps <= 0
        or args.checkpoint_every <= 0
        or args.eval_short_edge <= 0
    ):
        raise ValueError("complete VidEoMT step or cadence arguments are invalid")
    output_dir = args.output_dir.expanduser().resolve()
    if not str(output_dir).startswith("/mnt/"):
        raise ValueError("complete VidEoMT output must be persistent beneath /mnt")
    if output_dir.exists() and args.resume_checkpoint is None:
        raise FileExistsError(output_dir)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"complete VidEoMT gate requires {EXPECTED_WORLD_SIZE} ranks")
    if args.accumulation_steps * world_size != 8:
        raise RuntimeError("complete VidEoMT gate requires effective batch eight")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    _seed_all(args.seed + rank)

    implementation = _implementation_receipt(Path(__file__))
    implementation_sha256 = str(implementation["sha256"])
    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    dataset_binding = validate_dataset_runtime_binding(
        dataset_manifest,
        args.dataset_split.resolve(),
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        split_name=args.dataset_split.name,
    )
    index = CalvinDatasetIndex.load(
        args.dataset_split.resolve(),
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(
        args.physical_sidecar_root.resolve(),
        index,
        manifest_path=args.physical_sidecar_manifest.resolve(),
        expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
        eager_coverage_scan=False,
    )
    if sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise RuntimeError("complete VidEoMT adaptation requires all-source supervision")
    plan = build_calvin_videomt_episode_split_plan(
        index,
        clip_length=5,
        heldout_modulus=args.heldout_modulus,
        heldout_remainder=args.heldout_remainder,
    )
    plan_digests: list[str | None] = [None for _ in range(world_size)]
    dist.all_gather_object(plan_digests, plan.fingerprint)
    if len(set(plan_digests)) != 1:
        raise RuntimeError("CALVIN episode split differs across ranks")

    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.checkpoint.resolve(),
            local_dinov3_bundle=args.dinov3_bundle.resolve(),
            num_frames=5,
        ),
        device=device,
        dtype=torch.float32,
    )
    model_state_numel = sum(value.numel() for value in runtime.model.state_dict().values())
    parameter_numel = sum(parameter.numel() for parameter in runtime.model.parameters())
    trainable_parameter_numel = sum(
        parameter.numel() for parameter in runtime.model.parameters() if parameter.requires_grad
    )
    if (
        model_state_numel != EXPECTED_COMPLETE_MODEL_STATE_NUMEL
        or parameter_numel != EXPECTED_COMPLETE_PARAMETER_NUMEL
        or trainable_parameter_numel != EXPECTED_COMPLETE_PARAMETER_NUMEL
    ):
        raise RuntimeError(
            "complete VidEoMT parameter inventory drifted: "
            f"state={model_state_numel}, parameters={parameter_numel}, "
            f"trainable={trainable_parameter_numel}"
        )
    runtime.train()
    criterion = build_class_agnostic_criterion(
        matcher_identity=VIDEOMT_ONLINE_CONSISTENT_MATCHER,
        num_frames=5,
    ).to(device)
    optimizer, optimizer_receipt = build_exact_videomt_optimizer(runtime.model)
    scheduler = build_exact_videomt_scheduler(
        optimizer,
        optimizer_receipt,
        total_steps=args.budget_steps,
    )
    # Match Detectron2 AMPTrainer: overflow is handled by GradScaler's dynamic
    # backoff instead of being promoted to a fatal training error.
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    global_step = 0
    if args.resume_checkpoint is not None:
        global_step = _load_checkpoint(
            args.resume_checkpoint.resolve(),
            runtime=runtime,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            plan=plan,
            implementation_sha256=implementation_sha256,
            rank=rank,
            world_size=world_size,
        )
    if global_step >= args.steps:
        raise ValueError("resume checkpoint is already at or beyond the requested stop")
    ddp = DistributedDataParallel(
        runtime,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    evaluation_store = _EvaluationStore(index, sidecar)
    train_eval_windows = _selected_windows(plan, "train", args.eval_clips)
    heldout_eval_windows = _selected_windows(plan, "heldout", args.eval_clips)
    report_path = output_dir / "report.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "RUNNING",
        "claim_scope": (
            "complete 315,986,985-parameter VidEoMT donor adaptation; no reduced decoder, "
            "no task-conditioned forward input, and no PICF/action claim"
        ),
        "implementation_sha256": implementation_sha256,
        "implementation": implementation,
        "assets": {
            "released_checkpoint_sha256": _sha256(args.checkpoint),
            "dinov3_bundle": str(args.dinov3_bundle.resolve()),
            "dataset_manifest_sha256": _sha256(args.dataset_manifest),
            "physical_sidecar_manifest_sha256": args.physical_sidecar_manifest_sha256,
            "physical_sidecar_coverage_validation": sidecar.coverage_validation,
        },
        "dataset": {
            **dataset_binding,
            "split_identity": plan.identity,
            "split_plan_sha256": plan.fingerprint,
            "train_episode_count": len(plan.episode_indices("train")),
            "heldout_episode_count": len(plan.episode_indices("heldout")),
            "train_episode_indices": list(plan.episode_indices("train")),
            "heldout_episode_indices": list(plan.episode_indices("heldout")),
            "train_window_count": plan.window_count("train"),
            "heldout_window_count": plan.window_count("heldout"),
            "episode_disjoint": True,
            "task_or_prompt_forward_input": False,
            "owner_target_forward_input": False,
        },
        "recipe": {
            "steps": args.steps,
            "budget_steps": args.budget_steps,
            "world_size": world_size,
            "accumulation_steps": args.accumulation_steps,
            "effective_batch_clips": world_size * args.accumulation_steps,
            "clip_length": 5,
            "train_short_edges": list(VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES),
            "train_max_size": VIDEOMT_YTVIS19_TRAIN_MAX_SIZE,
            "eval_short_edge": args.eval_short_edge,
            "matcher": VIDEOMT_ONLINE_CONSISTENT_MATCHER,
            "precision": "fp32-parameters+released-fp16-amp",
            "model_state_numel": model_state_numel,
            "parameter_numel": parameter_numel,
            "trainable_parameter_numel": trainable_parameter_numel,
            "optimizer": asdict(optimizer_receipt),
            "eval_steps": list(eval_steps),
            "checkpoint_every": args.checkpoint_every,
            "save_final_checkpoint": args.save_final_checkpoint,
        },
        "evaluations": {},
        "training": [],
        "checkpoints": [],
    }

    def run_evaluation(step: int) -> None:
        dist.barrier()
        if rank == 0:
            started = time.perf_counter()
            report["evaluations"][str(step)] = {
                "train": _evaluate(
                    runtime=runtime,
                    store=evaluation_store,
                    windows=train_eval_windows,
                    short_edge=args.eval_short_edge,
                    device=device,
                    panel_path=output_dir / "visuals" / f"train_step_{step:06d}.png",
                ),
                "heldout": _evaluate(
                    runtime=runtime,
                    store=evaluation_store,
                    windows=heldout_eval_windows,
                    short_edge=args.eval_short_edge,
                    device=device,
                    panel_path=output_dir / "visuals" / f"heldout_step_{step:06d}.png",
                ),
                "seconds": time.perf_counter() - started,
            }
            _atomic_json(report_path, report)
        dist.barrier()
        runtime.train()

    if global_step == 0:
        run_evaluation(0)
    log_path = output_dir / f"train_rank_{rank}.jsonl"
    log_mode = "a" if global_step else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        while global_step < args.steps:
            started = time.perf_counter()
            prepared_clips: list[PreparedCalvinVidEoMTClip] = []
            windows: list[tuple[int, ...]] = []
            local_target_counts: list[int] = []
            preparation_error: BaseException | None = None
            try:
                for microstep in range(args.accumulation_steps):
                    global_clip_visit = (
                        global_step * (world_size * args.accumulation_steps)
                        + microstep * world_size
                        + rank
                    )
                    window = stateless_calvin_videomt_window(
                        plan,
                        split="train",
                        visit_index=global_clip_visit,
                        seed=args.seed,
                    )
                    _seed_all(
                        _microstep_seed(args.seed, global_clip_visit, phase="augmentation")
                    )
                    source = materialize_calvin_videomt_clip(index, sidecar, window)
                    prepared = prepare_calvin_videomt_training_clip(
                        source.rgb_static,
                        source.supervision,
                    )
                    prepared_clips.append(prepared)
                    windows.append(window)
                    local_target_counts.append(int(prepared.target["labels"].numel()))
            except BaseException as error:
                preparation_error = error
            preparation_failures = _distributed_error(preparation_error)
            if preparation_failures:
                raise RuntimeError(
                    f"distributed CALVIN clip preparation failed: {preparation_failures}"
                ) from preparation_error

            target_counts = torch.tensor(
                local_target_counts,
                device=device,
                dtype=torch.long,
            )
            dist.all_reduce(target_counts, op=dist.ReduceOp.SUM)
            batch_receipt = make_effective_batch_receipt(
                target_counts.tolist(),
                world_size=world_size,
            )
            optimizer.zero_grad(set_to_none=True)
            local_scaled_losses: dict[str, float] = {}
            for microstep, (window, prepared) in enumerate(
                zip(windows, prepared_clips, strict=True)
            ):
                global_clip_visit = (
                    global_step * (world_size * args.accumulation_steps)
                    + microstep * world_size
                    + rank
                )
                _seed_all(_microstep_seed(args.seed, global_clip_visit, phase="model"))
                frames = prepared.frames.model_input.to(device=device, dtype=torch.float32)
                targets = [
                    {name: value.to(device=device) for name, value in prepared.target.items()}
                ]
                synchronization = (
                    contextlib.nullcontext()
                    if microstep == args.accumulation_steps - 1
                    else ddp.no_sync()
                )
                with synchronization, torch.cuda.amp.autocast(dtype=torch.float16):
                    output = ddp(frames)
                    flat_outputs = flatten_class_agnostic_outputs(output)
                    flat_targets = flatten_class_agnostic_targets(targets)
                    raw_losses = criterion(flat_outputs, flat_targets)
                    weighted_losses = apply_released_loss_weights(raw_losses, criterion)
                    total_loss, scaled_losses = scale_videomt_microstep_losses(
                        weighted_losses,
                        batch_receipt,
                        microstep=microstep,
                    )
                scaler.scale(total_loss).backward()
                for name, value in scaled_losses.items():
                    local_scaled_losses[name] = local_scaled_losses.get(name, 0.0) + float(
                        value.detach()
                    )
                del frames, targets, output, flat_outputs, flat_targets, raw_losses
                del weighted_losses, total_loss, scaled_losses
                runtime.reset_state()

            amp_scale_before = float(scaler.get_scale())
            scaler.unscale_(optimizer)
            gradient_inventory = _gradient_inventory(runtime.model)
            scaler.step(optimizer)
            scaler.update()
            amp_scale_after = float(scaler.get_scale())
            optimizer_update_skipped = amp_scale_after < amp_scale_before
            if gradient_inventory["nonfinite_gradient_names"] and not optimizer_update_skipped:
                raise RuntimeError(
                    "complete VidEoMT produced non-finite gradients that GradScaler did not reject"
                )
            scheduler.step()
            global_step += 1
            torch.cuda.synchronize(device)

            global_losses = _reduce_float_mapping(local_scaled_losses, device)
            gathered_windows: list[object] = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered_windows,
                {"rank": rank, "windows": [list(value) for value in windows]},
            )
            gathered_gradients: list[object] = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered_gradients,
                {"rank": rank, "inventory": gradient_inventory},
            )
            step_record = {
                "step": global_step,
                "seconds": time.perf_counter() - started,
                "effective_batch": asdict(batch_receipt),
                "scaled_effective_losses": global_losses,
                "total_scaled_effective_loss": sum(global_losses.values()),
                "amp_scale_before": amp_scale_before,
                "amp_scale_after": amp_scale_after,
                "optimizer_update_skipped": optimizer_update_skipped,
                "learning_rate_min": min(optimizer_group_learning_rates(optimizer)),
                "learning_rate_max": max(optimizer_group_learning_rates(optimizer)),
                "rank_windows": gathered_windows,
                "rank_gradient_inventories": gathered_gradients,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            log_file.write(json.dumps(step_record, sort_keys=True) + "\n")
            log_file.flush()
            if rank == 0:
                report["training"].append(step_record)
                _atomic_json(report_path, report)
                print(json.dumps({"event": "optimizer_completed", **step_record}), flush=True)

            checkpoint_due = global_step % args.checkpoint_every == 0 or (
                args.save_final_checkpoint and global_step == args.steps
            )
            if checkpoint_due:
                local_rng_state = _capture_rng_state()
                gathered_rng_states: list[object] = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_rng_states, local_rng_state)
                dist.barrier()
                if rank == 0:
                    checkpoint_receipt = _save_checkpoint(
                        output_dir=output_dir,
                        runtime=runtime,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        global_step=global_step,
                        plan=plan,
                        implementation_sha256=implementation_sha256,
                        rank_rng_states=tuple(
                            cast(dict[str, object], value) for value in gathered_rng_states
                        ),
                    )
                    report["checkpoints"].append(checkpoint_receipt)
                    _atomic_json(report_path, report)
                dist.barrier()
            if global_step in eval_steps:
                run_evaluation(global_step)

    if rank == 0:
        report["status"] = "COMPLETE"
        report["completed_step"] = global_step
        report["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        report["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        _atomic_json(report_path, report)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
