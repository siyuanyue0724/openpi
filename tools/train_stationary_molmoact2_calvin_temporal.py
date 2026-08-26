#!/usr/bin/env python3
"""Train the bounded stationary PICF temporal core on 2xA100-40G."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from picf_next.training.accelerate_runner import (  # noqa: E402
    distributed_main_process_call,
)
from picf_next.training.m2_acceptance import validate_axis_calibrated_m2  # noqa: E402
from picf_next.training.run_lease import ExclusiveRunLease  # noqa: E402
from picf_next.training.stage_checkpoints import (  # noqa: E402
    StationaryTemporalCheckpointProvenance,
    parameter_scope_sha256,
    save_stationary_temporal_checkpoint,
    sha256_file,
)
from picf_next.training.stationary_accelerate_checkpoint import (  # noqa: E402
    StationaryAccelerateCheckpointIdentity,
    load_stationary_accelerate_checkpoint,
    save_stationary_accelerate_checkpoint,
)
from picf_next.training.stationary_calvin_stage import (  # noqa: E402
    build_stationary_temporal_trainer,
    load_stationary_calvin_stage_assets,
    load_stationary_calvin_stage_definition,
)

_MODES = {"definition", "preflight", "smoke", "train"}
_SMOKE_STEPS = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(_MODES), required=True)
    parser.add_argument(
        "--stage-recipe",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m3_stationary_temporal.json",
    )
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--feature-cache-root", type=Path)
    parser.add_argument("--physical-sidecar-root", type=Path)
    parser.add_argument("--m2-report", type=Path)
    parser.add_argument("--m2-checkpoint", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("ascii")
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(path)
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: object) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("ascii")
    with path.open("ab", buffering=0) as stream:
        stream.write(encoded)
        os.fsync(stream.fileno())


def _git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("repository HEAD is not one full Git revision")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("stationary cloud training requires a clean committed worktree")
    return revision


def _require_runtime_arguments(args: argparse.Namespace) -> None:
    required = {
        "split_root": args.split_root,
        "feature_cache_root": args.feature_cache_root,
        "physical_sidecar_root": args.physical_sidecar_root,
        "m2_report": args.m2_report,
        "m2_checkpoint": args.m2_checkpoint,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f"{args.mode} mode is missing required arguments: {missing}")
    if args.mode in {"smoke", "train"} and args.run_root is None:
        raise ValueError(f"{args.mode} mode requires --run-root")
    if not isinstance(args.checkpoint_every, int) or not 0 < args.checkpoint_every <= 200:
        raise ValueError("checkpoint-every must lie in [1, 200]")
    if args.resume is not None and args.mode != "train":
        raise ValueError("only formal Stage-B training may resume")


def _validate_hardware(accelerator: Any) -> None:
    if int(accelerator.num_processes) != 2:
        raise RuntimeError("stationary Stage B requires exactly two distributed processes")
    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("stationary Stage B requires CUDA")
    name = torch.cuda.get_device_name(accelerator.device)
    memory_gib = torch.cuda.get_device_properties(accelerator.device).total_memory / 2**30
    if "A100" not in name or memory_gib < 39.0:
        raise RuntimeError(f"expected A100-40G, observed {name!r} with {memory_gib:.2f} GiB")


def _reduce_mean(accelerator: Any, value: float) -> float:
    tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float64)
    reduced = accelerator.reduce(tensor, reduction="sum") / float(accelerator.num_processes)
    return float(reduced.item())


def _reduce_sum(accelerator: Any, value: int) -> int:
    tensor = torch.tensor(value, device=accelerator.device, dtype=torch.int64)
    return int(accelerator.reduce(tensor, reduction="sum").item())


def _reduce_diagnostic_totals(
    accelerator: Any,
    diagnostics: dict[str, int | float],
) -> dict[str, float]:
    """Reduce count-like objective diagnostics over the global two-rank batch."""

    output = {}
    for name, value in diagnostics.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("stationary objective diagnostic is malformed")
        tensor = torch.tensor(float(value), device=accelerator.device, dtype=torch.float64)
        output[f"picf_{name}"] = float(accelerator.reduce(tensor, reduction="sum").item())
    return output


def _reduce_max(accelerator: Any, value: float) -> float:
    tensor = torch.tensor([value], device=accelerator.device, dtype=torch.float64)
    gathered = accelerator.gather(tensor)
    return float(gathered.max().item())


def _validate_gradients(module: torch.nn.Module) -> None:
    missing = []
    nonfinite = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
    if missing or nonfinite:
        raise RuntimeError(
            "stationary temporal gradient coverage failed; "
            f"missing={missing[:8]}, nonfinite={nonfinite[:8]}"
        )


def _accelerator_runtime_kwargs() -> dict[str, object]:
    """Keep one scheduler tick equal to one global optimizer update."""

    return {
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 1,
        "step_scheduler_with_optimizer": False,
    }


def _validate_scheduler_epoch(scheduler: Any, *, completed_steps: int) -> int:
    state = scheduler.state_dict()
    epoch = state.get("last_epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise RuntimeError("stationary scheduler omitted its integer global-step epoch")
    if epoch != completed_steps:
        raise RuntimeError(
            "stationary scheduler drifted from global optimizer progress: "
            f"scheduler={epoch}, optimizer={completed_steps}"
        )
    return epoch


def _definition_report(definition: Any) -> dict[str, object]:
    return {
        "schema": "picf-next.stationary-temporal-definition-report.v1",
        "stage_recipe_sha256": definition.stage.recipe_sha256,
        "source_coverage_recipe_sha256": definition.source_coverage.recipe_sha256,
        "foundation_recipe_sha256": definition.historical_foundation.recipe_sha256,
        "structural_recipe_sha256": definition.structural_foundation.recipe_sha256,
        "clip_plan_sha256": definition.clip_plan.plan_sha256,
        "optimizer_steps": definition.clip_plan.optimizer_steps,
        "world_size": definition.clip_plan.world_size,
        "prefix_lengths": list(definition.clip_plan.prefix_lengths),
        "train_length": definition.clip_plan.train_length,
        "required_future_horizon": definition.clip_plan.required_future_horizon,
        "action_weight": definition.structural_foundation.objective_config.action_weight,
        "long_training_authorized": False,
    }


def _prepare_runtime(args: argparse.Namespace, definition: Any) -> tuple[Any, dict[str, Any]]:
    _require_runtime_arguments(args)
    assert args.m2_report is not None
    assert args.m2_checkpoint is not None
    binding = validate_axis_calibrated_m2(
        report_path=args.m2_report,
        checkpoint_path=args.m2_checkpoint,
    )
    assets = load_stationary_calvin_stage_assets(
        definition,
        repository_root=_ROOT,
        split_root=args.split_root,
        feature_cache_root=args.feature_cache_root,
        feature_cache_manifest_sha256=binding["feature_cache_manifest_sha256"],
        physical_sidecar_root=args.physical_sidecar_root,
    )
    return assets, binding


def _run_preflight(args: argparse.Namespace, definition: Any, assets: Any, binding: Any) -> None:
    assert args.m2_checkpoint is not None
    trainer = build_stationary_temporal_trainer(
        definition,
        m2_checkpoint_path=args.m2_checkpoint,
        m2_checkpoint_sha256=binding["checkpoint_sha256"],
        device="cpu",
    )
    trainable, frozen = parameter_scope_sha256(trainer.core, trainer.objective)
    report = {
        **_definition_report(definition),
        "schema": "picf-next.stationary-temporal-preflight.v1",
        "status": "PASS",
        "dataset_manifest_sha256": (
            definition.historical_foundation.artifacts.dataset_file_manifest_sha256
        ),
        "feature_cache_manifest_sha256": assets.feature_cache.manifest_sha256,
        "physical_sidecar_manifest_sha256": (
            definition.source_coverage.physical_sidecar_manifest_sha256
        ),
        "m2_checkpoint_sha256": binding["checkpoint_sha256"],
        "trainable_parameter_scope_sha256": trainable,
        "frozen_parameter_scope_sha256": frozen,
    }
    print(_canonical_json(report))


def _initialize_run_root(
    accelerator: Any,
    run_root: Path,
    *,
    mode: str,
    definition: Any,
    assets: Any,
    binding: dict[str, Any],
    code_revision: str,
    resume: bool,
) -> Path:
    resolved = run_root.expanduser().resolve()
    if not str(resolved).startswith("/mnt/"):
        raise ValueError("stationary cloud run_root must be beneath /mnt")
    manifest = {
        **_definition_report(definition),
        "schema": "picf-next.stationary-temporal-run.v1",
        "mode": mode,
        "code_revision": code_revision,
        "dataset_manifest_sha256": (
            definition.historical_foundation.artifacts.dataset_file_manifest_sha256
        ),
        "feature_cache_manifest_sha256": assets.feature_cache.manifest_sha256,
        "physical_sidecar_manifest_sha256": (
            definition.source_coverage.physical_sidecar_manifest_sha256
        ),
        "m2_binding": binding,
        "recurrent_state_checkpointed": False,
    }

    def initialize() -> None:
        if resume:
            if not resolved.is_dir() or resolved.is_symlink():
                raise FileNotFoundError("stationary resume run root is absent or unsafe")
            try:
                observed_plan = json.loads((resolved / "clip_plan.json").read_text())
                observed_manifest = json.loads((resolved / "run_manifest.json").read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("stationary resume metadata is invalid") from exc
            if observed_plan != definition.clip_plan.to_dict() or observed_manifest != manifest:
                raise ValueError("stationary resume metadata differs from the active run")
            if (resolved / "report.json").exists():
                raise ValueError("stationary run already has a terminal report")
            return
        if resolved.exists() or resolved.is_symlink():
            raise FileExistsError(resolved)
        resolved.mkdir(parents=True)
        _write_json_atomic(resolved / "clip_plan.json", definition.clip_plan.to_dict())
        _write_json_atomic(resolved / "run_manifest.json", manifest)

    distributed_main_process_call(
        accelerator,
        label="stationary run metadata initialization",
        action=initialize,
    )
    return resolved


def _checkpoint_identity(
    definition: Any,
    assets: Any,
    binding: dict[str, Any],
    *,
    code_revision: str,
) -> StationaryAccelerateCheckpointIdentity:
    return StationaryAccelerateCheckpointIdentity(
        stage_recipe_sha256=definition.stage.recipe_sha256,
        source_coverage_recipe_sha256=definition.source_coverage.recipe_sha256,
        foundation_recipe_sha256=definition.historical_foundation.recipe_sha256,
        m2_checkpoint_sha256=binding["checkpoint_sha256"],
        feature_cache_manifest_sha256=assets.feature_cache.manifest_sha256,
        dataset_manifest_sha256=(
            definition.historical_foundation.artifacts.dataset_file_manifest_sha256
        ),
        physical_sidecar_manifest_sha256=(
            definition.source_coverage.physical_sidecar_manifest_sha256
        ),
        clip_plan_sha256=definition.clip_plan.plan_sha256,
        code_revision=code_revision,
        world_size=definition.stage.distributed.world_size,
        total_steps=definition.stage.optimizer.optimizer_steps,
    )


def _reconcile_metrics(path: Path, *, completed_steps: int) -> None:
    if not path.is_file():
        if completed_steps == 0:
            return
        raise FileNotFoundError("stationary resume has no metrics history")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"stationary metrics line {line_number} is invalid") from exc
        if not isinstance(payload, dict) or set(payload) != {"optimizer_step", "metrics"}:
            raise ValueError("stationary metrics history schema changed")
        if (
            not isinstance(payload["optimizer_step"], int)
            or isinstance(payload["optimizer_step"], bool)
            or payload["optimizer_step"] <= 0
            or not isinstance(payload["metrics"], dict)
        ):
            raise ValueError("stationary metrics record is malformed")
        records.append(payload)
    retained = [record for record in records if record["optimizer_step"] <= completed_steps]
    if [record["optimizer_step"] for record in retained] != list(range(1, completed_steps + 1)):
        raise ValueError("stationary metrics history does not cover checkpoint progress exactly")
    if len(retained) == len(records):
        return
    temporary = path.with_name(f".{path.name}.reconcile-{os.getpid()}")
    with temporary.open("xb") as stream:
        for record in retained:
            stream.write((_canonical_json(record) + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run_distributed(
    args: argparse.Namespace,
    definition: Any,
    assets: Any,
    binding: dict[str, Any],
) -> None:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    accelerator = Accelerator(
        **_accelerator_runtime_kwargs(),
        kwargs_handlers=[kwargs],
    )
    _validate_hardware(accelerator)
    torch.manual_seed(definition.stage.clip.seed)
    torch.cuda.manual_seed_all(definition.stage.clip.seed)
    code_revision = _git_revision(_ROOT)
    assert args.m2_checkpoint is not None
    assert args.run_root is not None
    run_root = _initialize_run_root(
        accelerator,
        args.run_root,
        mode=args.mode,
        definition=definition,
        assets=assets,
        binding=binding,
        code_revision=code_revision,
        resume=args.resume is not None,
    )
    run_lease: ExclusiveRunLease | None = None

    def acquire_run_lease() -> dict[str, Any]:
        nonlocal run_lease
        run_lease = ExclusiveRunLease.acquire(run_root)
        return run_lease.owner

    distributed_main_process_call(
        accelerator,
        label="stationary single-writer lease acquisition",
        action=acquire_run_lease,
    )
    trainer = build_stationary_temporal_trainer(
        definition,
        m2_checkpoint_path=args.m2_checkpoint,
        m2_checkpoint_sha256=binding["checkpoint_sha256"],
        device=accelerator.device,
    )
    optimizer, scheduler = definition.stage.build_optimizer_and_scheduler(trainer)
    model, optimizer, scheduler = accelerator.prepare(trainer, optimizer, scheduler)
    metrics_path = run_root / "metrics.jsonl"
    total_steps = (
        _SMOKE_STEPS if args.mode == "smoke" else definition.stage.optimizer.optimizer_steps
    )
    unwrapped = accelerator.unwrap_model(model)
    identity = _checkpoint_identity(
        definition,
        assets,
        binding,
        code_revision=code_revision,
    )
    start_step = 0
    if args.resume is not None:
        resume = args.resume.expanduser().resolve()
        if run_root not in resume.parents:
            raise ValueError("stationary resume checkpoint must be beneath its run root")
        start_step = load_stationary_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=resume,
            identity=identity,
        )
        distributed_main_process_call(
            accelerator,
            label="stationary metrics reconciliation",
            action=lambda: _reconcile_metrics(metrics_path, completed_steps=start_step),
        )
    for optimizer_step in range(start_step, total_steps):
        clip = definition.clip_plan.clip(optimizer_step, int(accelerator.process_index))
        batch = assets.batch_builder.build(
            (clip,),
            device=accelerator.device,
            dtype=torch.bfloat16,
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        accelerator.wait_for_everyone()
        started = time.perf_counter()
        with accelerator.autocast():
            output = model(
                batch.observations,
                prefix_length=batch.prefix_length,
                supervision_builder=batch.build_supervision,
                geometry_builder=batch.build_geometry_rollout,
            )
        loss = output.objective.loss
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError("stationary temporal objective became non-finite")
        accelerator.backward(loss)
        _validate_gradients(unwrapped)
        grad_norm = accelerator.clip_grad_norm_(
            model.parameters(),
            definition.stage.optimizer.gradient_clip_norm,
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError("stationary temporal gradient norm became non-finite")
        optimizer.step()
        if bool(getattr(accelerator, "optimizer_step_was_skipped", False)):
            raise RuntimeError("stationary Stage B does not permit skipped optimizer steps")
        scheduler.step()
        completed_steps = optimizer_step + 1
        scheduler_epoch = _validate_scheduler_epoch(
            scheduler,
            completed_steps=completed_steps,
        )
        accelerator.wait_for_everyone()
        elapsed = time.perf_counter() - started
        metrics = {
            name: _reduce_mean(accelerator, float(value.detach().float().item()))
            for name, value in output.objective.losses.items()
        }
        metrics.update(_reduce_diagnostic_totals(accelerator, output.objective.diagnostics))
        metrics.update(
            {
                "loss": _reduce_mean(accelerator, float(loss.detach().float().item())),
                "grad_norm": _reduce_mean(accelerator, float(grad_norm.detach().float().item())),
                "elapsed_seconds": _reduce_max(accelerator, elapsed),
                "peak_allocated_bytes": _reduce_max(
                    accelerator,
                    float(torch.cuda.max_memory_allocated(accelerator.device)),
                ),
                "prefix_assignment_conflicts": _reduce_sum(
                    accelerator,
                    output.prefix_assignment_conflicts,
                ),
                "prefix_length": clip.prefix_length,
                "train_length": clip.train_length,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "scheduler_epoch": scheduler_epoch,
            }
        )
        if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("stationary temporal metrics became non-finite")
        if accelerator.is_main_process:
            _append_jsonl(
                metrics_path,
                {
                    "optimizer_step": optimizer_step + 1,
                    "metrics": metrics,
                },
            )
        if args.mode == "train" and (
            completed_steps % args.checkpoint_every == 0 or completed_steps == total_steps
        ):
            save_stationary_accelerate_checkpoint(
                accelerator=accelerator,
                checkpoint_dir=(run_root / "checkpoints" / f"step-{completed_steps:08d}"),
                identity=identity,
                completed_steps=completed_steps,
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.mode == "train":
            trainable, frozen = parameter_scope_sha256(unwrapped.core, unwrapped.objective)
            provenance = StationaryTemporalCheckpointProvenance(
                stage_recipe_sha256=definition.stage.recipe_sha256,
                source_coverage_recipe_sha256=definition.source_coverage.recipe_sha256,
                foundation_recipe_sha256=definition.historical_foundation.recipe_sha256,
                m2_checkpoint_sha256=binding["checkpoint_sha256"],
                feature_cache_manifest_sha256=assets.feature_cache.manifest_sha256,
                dataset_manifest_sha256=(
                    definition.historical_foundation.artifacts.dataset_file_manifest_sha256
                ),
                physical_sidecar_manifest_sha256=(
                    definition.source_coverage.physical_sidecar_manifest_sha256
                ),
                clip_plan_sha256=definition.clip_plan.plan_sha256,
                trainable_parameter_scope_sha256=trainable,
                frozen_parameter_scope_sha256=frozen,
                code_revision=code_revision,
                optimizer_steps=total_steps,
                state_parameter_version=total_steps,
            )
            checkpoint = run_root / "stationary_temporal_core_candidate.pt"
            checkpoint_sha = save_stationary_temporal_checkpoint(
                checkpoint,
                core=unwrapped.core,
                objective=unwrapped.objective,
                provenance=provenance,
            )
            report = {
                **_definition_report(definition),
                "schema": "picf-next.stationary-temporal-candidate-report.v1",
                "status": "CANDIDATE_REQUIRES_FIXED_CHECKPOINT_AUDIT",
                "checkpoint_sha256": checkpoint_sha,
                "metrics_sha256": sha256_file(metrics_path),
                "completed_optimizer_steps": total_steps,
                "long_training_authorized": False,
            }
        else:
            report = {
                **_definition_report(definition),
                "schema": "picf-next.stationary-temporal-smoke-report.v1",
                "status": "SMOKE_COMPLETE_NOT_AN_ACCEPTED_CHECKPOINT",
                "metrics_sha256": sha256_file(metrics_path),
                "completed_optimizer_steps": total_steps,
                "long_training_authorized": False,
            }
        _write_json_atomic(run_root / "report.json", report)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        if run_lease is None:  # pragma: no cover - guarded by distributed acquisition
            raise RuntimeError("stationary main process lost its run lease")
        run_lease.close()


def main() -> None:
    args = _parse_args()
    definition = load_stationary_calvin_stage_definition(
        args.stage_recipe,
        repository_root=_ROOT,
    )
    if args.mode == "definition":
        print(_canonical_json(_definition_report(definition)))
        return
    assets, binding = _prepare_runtime(args, definition)
    if args.mode == "preflight":
        _run_preflight(args, definition, assets, binding)
        return
    _run_distributed(args, definition, assets, binding)


if __name__ == "__main__":
    main()
