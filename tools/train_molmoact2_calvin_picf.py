#!/usr/bin/env python3
"""Run the authorized stateful MolmoAct2/CALVIN PICF probe on 2xA100-40G."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
for _path in (_ROOT, _SOURCE_ROOT):
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

from picf_next.data.vjepa2_cache import Vjepa2FeatureCache  # noqa: E402
from picf_next.training.accelerate_runner import (  # noqa: E402
    distributed_main_process_call,
    load_accelerate_checkpoint,
    register_progress_for_checkpointing,
    save_accelerate_checkpoint,
)
from picf_next.training.control import (  # noqa: E402
    ExperimentRunContract,
    RunProgress,
)
from picf_next.training.recipe import load_training_recipe  # noqa: E402
from picf_next.training.stateful_runner import StatefulEpisodeTrainingRunner  # noqa: E402
from picf_next.training.stationary_acceptance import (  # noqa: E402
    AcceptedStationaryTemporalCore,
    validate_stationary_temporal_acceptance,
)
from picf_next.training.stream_state import PosteriorStreamStateGroup  # noqa: E402
from tools.bootstrap_molmoact2 import validate_checkpoint  # noqa: E402
from tools.preflight_cloud import validate_config  # noqa: E402
from tools.run_molmoact2_m0_cloud import validate_m0_report  # noqa: E402

_ACTION_ARMS = {
    "A": {
        "name": "current_without_posterior_action_context",
        "include_causal_video": False,
        "include_posterior_action_context": False,
    },
    "B": {
        "name": "current_plus_causal_video_without_posterior_action_context",
        "include_causal_video": True,
        "include_posterior_action_context": False,
    },
    "C": {
        "name": "current_plus_posterior_action_context",
        "include_causal_video": False,
        "include_posterior_action_context": True,
    },
    "D": {
        "name": "current_plus_causal_video_plus_posterior_action_context",
        "include_causal_video": True,
        "include_posterior_action_context": True,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud-config",
        type=Path,
        default=_ROOT / "configs/cloud/2xa100_40g_m4_action_adoption.json",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m4_action_adoption.json",
    )
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--stationary-acceptance-report",
        type=Path,
        help="hash-bound Stage-B acceptance report; required for execution",
    )
    parser.add_argument(
        "--stationary-checkpoint",
        type=Path,
        help="accepted full stationary temporal core; required for execution",
    )
    parser.add_argument(
        "--m0-report",
        type=Path,
        help="accepted M0 report; optional only for non-loading --verify-only assembly",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--arm", choices=tuple(_ACTION_ARMS), required=True)
    parser.add_argument("--vjepa2-cache-root", type=Path)
    parser.add_argument("--vjepa2-cache-manifest-sha256")
    parser.add_argument("--vjepa2-cache-memory-capacity", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise RuntimeError("repository HEAD is not one full git revision")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("cloud training requires a clean committed worktree")
    return revision


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".incomplete-{os.getpid()}")
    stale = tuple(path.parent.glob(f"{path.name}.incomplete-*"))
    if path.exists() or stale:
        raise FileExistsError(path)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_fresh_run_metadata(
    run_root: Path,
    *,
    static_report: dict[str, object],
    sample_plan: dict[str, object],
) -> None:
    """Publish the immutable run metadata as one durable directory transaction."""

    parent = run_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{run_root.name}.incomplete-{os.getpid()}"
    stale = tuple(parent.glob(f".{run_root.name}.incomplete-*"))
    if run_root.exists() or stale:
        raise FileExistsError(run_root)
    staging.mkdir()
    _fsync_directory(parent)
    _atomic_json(staging / "static_preflight.json", static_report)
    _atomic_json(staging / "sample_plan.json", sample_plan)
    _fsync_directory(staging)
    os.replace(staging, run_root)
    _fsync_directory(parent)


def _validate_persistent_run_root(run_root: Path) -> None:
    if Path("/mnt") not in run_root.parents:
        raise RuntimeError("cloud run_root must be a strict descendant of persistent /mnt storage")


def _action_arm_spec(arm: object) -> dict[str, object]:
    if not isinstance(arm, str) or arm not in _ACTION_ARMS:
        raise ValueError("action arm must be exactly one of A/B/C/D")
    return {"id": arm, **_ACTION_ARMS[arm]}


def _load_vjepa2_cache_for_arm(
    *,
    arm_spec: Mapping[str, object],
    cache_root: Path | None,
    cache_manifest_sha256: str | None,
    cache_memory_capacity: int,
    dataset_tree_sha256: str,
    require_persistent_root: bool,
) -> tuple[Vjepa2FeatureCache | None, dict[str, object] | None]:
    use_video = arm_spec.get("include_causal_video") is True
    supplied = (cache_root is not None, cache_manifest_sha256 is not None)
    if supplied[0] != supplied[1]:
        raise ValueError("V-JEPA2 cache root and manifest SHA must be supplied together")
    if use_video and not all(supplied):
        raise ValueError("M3 video arms B/D require a hash-bound V-JEPA2 cache")
    if not use_video and any(supplied):
        raise ValueError("M3 non-video arms A/C forbid V-JEPA2 cache arguments")
    if not isinstance(cache_memory_capacity, int) or isinstance(cache_memory_capacity, bool):
        raise ValueError("V-JEPA2 cache memory capacity must be a positive integer")
    if cache_memory_capacity <= 0:
        raise ValueError("V-JEPA2 cache memory capacity must be a positive integer")
    if not use_video:
        return None, None
    if cache_root is None or cache_manifest_sha256 is None:  # pragma: no cover
        raise RuntimeError("validated video arm lost its cache arguments")
    resolved_root = cache_root.expanduser().resolve()
    if require_persistent_root and Path("/mnt") not in resolved_root.parents:
        raise RuntimeError("cloud V-JEPA2 cache must be a strict descendant of /mnt")
    cache = Vjepa2FeatureCache.load(
        resolved_root,
        manifest_sha256=cache_manifest_sha256,
        dataset_tree_sha256=dataset_tree_sha256,
        memory_capacity=cache_memory_capacity,
    )
    binding = {
        "dataset_tree_sha256": cache.dataset_tree_sha256,
        "encoder_contract": cache.encoder_contract,
        "entries": len(cache.entries),
        "hidden_size": cache.hidden_size,
        "manifest_sha256": cache_manifest_sha256,
        "maximum_frames": cache.maximum_frames,
        "root": str(resolved_root),
    }
    return cache, binding


def _validate_m0(
    *,
    report_path: Path,
    cloud_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    report = _read_json(report_path, "M0 report")
    validate_m0_report(report, config=cloud_config, root=_ROOT)
    shards = report.get("checkpoint_weight_shard_sha256")
    if not isinstance(shards, dict):
        raise ValueError("M0 report omitted checkpoint shard hashes")
    return report, _canonical_sha256(shards)


def _validate_m0_for_mode(
    *,
    report_path: Path | None,
    cloud_config: dict[str, Any],
    verify_only: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Keep M0 mandatory for execution while permitting local static assembly."""

    if report_path is None:
        if verify_only:
            return None, None
        raise ValueError("training requires --m0-report from the accepted full-weight M0 gate")
    return _validate_m0(report_path=report_path.resolve(), cloud_config=cloud_config)


def _validate_training_checkpoint(
    *,
    checkpoint_dir: Path,
    m0_report: dict[str, Any],
    checkpoint_id: str,
    checkpoint_revision: str,
) -> None:
    checkpoint_assets = validate_checkpoint(checkpoint_dir, validate_weight_shards=True)
    if checkpoint_assets["checkpoint_id"] != checkpoint_id:
        raise ValueError("training checkpoint id differs from the frozen recipe")
    if checkpoint_assets["checkpoint_revision"] != checkpoint_revision:
        raise ValueError("training checkpoint revision differs from the frozen recipe")
    if checkpoint_assets["weight_shard_sha256"] != m0_report["checkpoint_weight_shard_sha256"]:
        raise ValueError("training checkpoint weight shards differ from the accepted M0 report")


def _validate_stationary_core_for_mode(
    *,
    report_path: Path | None,
    checkpoint_path: Path | None,
    verify_only: bool,
) -> AcceptedStationaryTemporalCore | None:
    """Bind action execution to one accepted full stationary estimator."""

    if report_path is None or checkpoint_path is None:
        if verify_only and report_path is None and checkpoint_path is None:
            return None
        raise ValueError(
            "training requires both --stationary-acceptance-report and --stationary-checkpoint"
        )
    return validate_stationary_temporal_acceptance(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )


def _validate_hardware(accelerator: Any) -> None:
    if int(accelerator.num_processes) != 2:
        raise RuntimeError("the frozen M4 deployment requires exactly two processes")
    if accelerator.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen M4 deployment requires CUDA")
    name = torch.cuda.get_device_name(accelerator.device)
    memory_gib = torch.cuda.get_device_properties(accelerator.device).total_memory / 2**30
    if "A100" not in name or memory_gib < 39.0:
        raise RuntimeError(f"expected A100-40G, observed {name!r} with {memory_gib:.2f} GiB")


def _mean_metrics(accelerator: Any, metrics: tuple[dict[str, float], ...]) -> dict[str, float]:
    if not metrics:
        raise ValueError("at least one microbatch metric mapping is required")
    expected_names = tuple(sorted(metrics[0]))
    local: dict[str, list[float]] = {}
    for microbatch_index, microbatch in enumerate(metrics):
        if tuple(sorted(microbatch)) != expected_names:
            raise RuntimeError(
                f"metric key schema differs across local microbatches at index {microbatch_index}"
            )
        for name, value in microbatch.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FloatingPointError(f"metric {name!r} is non-finite")
            local.setdefault(name, []).append(numeric)
    metric_names = tuple(sorted(local))
    schema_digest = hashlib.sha256("\0".join(metric_names).encode("utf-8")).digest()
    schema_fingerprint = int.from_bytes(schema_digest[:8], "big") & ((1 << 63) - 1)
    local_schema = torch.tensor(
        [len(metric_names), schema_fingerprint],
        device=accelerator.device,
        dtype=torch.int64,
    )
    gathered_schema = accelerator.gather(local_schema).reshape(-1, 2)
    if not torch.equal(gathered_schema, gathered_schema[0].expand_as(gathered_schema)):
        raise RuntimeError(
            "metric key schema differs across ranks; refusing order-dependent collectives"
        )
    output: dict[str, float] = {}
    for name in metric_names:
        values = local[name]
        tensor = torch.tensor(
            sum(values) / len(values),
            device=accelerator.device,
            dtype=torch.float64,
        )
        output[name] = float(accelerator.reduce(tensor, reduction="mean").item())
    return output


def _optimizer_step_observability(result: Any, optimizer: Any) -> dict[str, float]:
    losses = tuple(float(microstep.loss) for microstep in result.microsteps)
    grad_norms = tuple(
        float(microstep.grad_norm)
        for microstep in result.microsteps
        if microstep.grad_norm is not None
    )
    learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
    values = (*losses, *grad_norms, *learning_rates)
    if (
        not losses
        or not grad_norms
        or not learning_rates
        or any(not math.isfinite(value) for value in values)
    ):
        raise FloatingPointError(
            "optimizer-step observability contains missing or non-finite values"
        )
    return {
        "system_optimizer_loss": sum(losses) / len(losses),
        "system_synchronized_grad_norm": grad_norms[-1],
        "system_learning_rate_min": min(learning_rates),
        "system_learning_rate_max": max(learning_rates),
    }


def _distributed_system_telemetry(accelerator: Any, *, elapsed_seconds: float) -> dict[str, float]:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise ValueError("training-step elapsed time must be finite and positive")
    local = torch.tensor(
        [
            elapsed_seconds,
            float(torch.cuda.memory_allocated(accelerator.device)),
            float(torch.cuda.memory_reserved(accelerator.device)),
            float(torch.cuda.max_memory_allocated(accelerator.device)),
            float(torch.cuda.max_memory_reserved(accelerator.device)),
        ],
        device=accelerator.device,
        dtype=torch.float64,
    )
    gathered = accelerator.gather(local).reshape(-1, local.numel())
    names = (
        "train_step_wall_seconds",
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    )
    output: dict[str, float] = {}
    for index, name in enumerate(names):
        output[f"system_{name}_rank_mean"] = float(gathered[:, index].mean().item())
        output[f"system_{name}_rank_max"] = float(gathered[:, index].max().item())
    return output


def _synchronize_step_timing(accelerator: Any) -> None:
    """Put CUDA wall-time markers on completed work rather than queued kernels."""

    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)


def _validate_scheduler_epoch(scheduler: Any, *, successful_optimizer_steps: int) -> int:
    epoch = scheduler.state_dict().get("last_epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise RuntimeError("action scheduler omitted its integer global-step epoch")
    if epoch != successful_optimizer_steps:
        raise RuntimeError(
            "action scheduler drifted from global optimizer progress: "
            f"scheduler={epoch}, optimizer={successful_optimizer_steps}"
        )
    return epoch


def _append_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("a", encoding="ascii") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if created:
        _fsync_directory(path.parent)


def _reconcile_metrics_for_resume(
    path: Path,
    *,
    attempted_optimizer_steps: int,
    successful_optimizer_steps: int,
) -> int:
    """Validate JSONL history and discard records newer than the loaded checkpoint."""

    counters = (attempted_optimizer_steps, successful_optimizer_steps)
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counters
        )
        or successful_optimizer_steps > attempted_optimizer_steps
    ):
        raise ValueError("resume progress counters are invalid")
    if not path.is_file():
        if attempted_optimizer_steps == 0:
            return 0
        raise FileNotFoundError(path)

    records: list[dict[str, object]] = []
    previous_successful = 0
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not line:
            raise ValueError(f"metrics line {line_number} is empty")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"metrics line {line_number} is not valid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {
            "attempted_optimizer_steps",
            "metrics",
            "optimizer_step_skipped",
            "successful_optimizer_steps",
        }:
            raise ValueError(f"metrics line {line_number} fields are malformed")
        attempted = payload["attempted_optimizer_steps"]
        successful = payload["successful_optimizer_steps"]
        skipped = payload["optimizer_step_skipped"]
        metrics = payload["metrics"]
        if attempted != line_number or not isinstance(skipped, bool):
            raise ValueError("metrics optimizer attempts are not a contiguous one-based sequence")
        expected_successful = previous_successful + int(not skipped)
        if successful != expected_successful:
            raise ValueError("metrics successful-step progression is inconsistent with skip state")
        if not isinstance(metrics, dict) or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for name, value in metrics.items()
        ):
            raise ValueError(f"metrics line {line_number} values are malformed")
        previous_successful = expected_successful
        records.append(payload)

    if len(records) < attempted_optimizer_steps:
        raise ValueError("metrics history ends before the loaded checkpoint")
    if attempted_optimizer_steps > 0 and (
        records[attempted_optimizer_steps - 1]["successful_optimizer_steps"]
        != successful_optimizer_steps
    ):
        raise ValueError("metrics history disagrees with loaded checkpoint progress")
    removed = len(records) - attempted_optimizer_steps
    if removed <= 0:
        return 0

    temporary = path.with_name(f".{path.name}.reconcile-{os.getpid()}")
    if tuple(path.parent.glob(f".{path.name}.reconcile-*")):
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="ascii") as stream:
        for payload in records[:attempted_optimizer_steps]:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return removed


def _validate_existing_json(path: Path, expected: object, name: str) -> None:
    if _read_json(path, name) != expected:
        raise ValueError(f"existing {name} differs from the active run")


def _build_picf_run_contract(
    *,
    recipe: Any,
    plan: Any,
    assets: Any,
    args: argparse.Namespace,
    code_revision: str,
    checkpoint_manifest_sha256: str,
    stationary_core_binding: Mapping[str, Any],
    action_arm: Mapping[str, object],
    vjepa2_binding: Mapping[str, object] | None,
    world_size: int,
) -> ExperimentRunContract:
    """Bind the run to the verified file tree and reject plan drift immediately."""

    semantic_arm = (
        "picf" if action_arm.get("include_posterior_action_context") is True else "full_evidence"
    )
    contract = ExperimentRunContract.build(
        arm=semantic_arm,
        comparison_id=args.comparison_id,
        code_revision=code_revision,
        host_name=recipe.host.name,
        host_source_revision=recipe.host.source_commit,
        training_source_revision=recipe.host.trainer_commit,
        foundation_checkpoint_id=recipe.host.checkpoint_id,
        foundation_checkpoint_revision=recipe.host.checkpoint_revision,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        dataset_id=recipe.dataset.dataset_id,
        dataset_revision=recipe.dataset.dataset_revision,
        dataset_manifest_sha256=assets.dataset_manifest.tree_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=args.global_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        precision=recipe.policy.model_dtype,
        action_convention="calvin.normalized-relative-ee-pose.v1",
        detached_context_frames=recipe.detached_context_frames,
        gradient_transitions=recipe.gradient_transitions,
        trainable_scope="host_action_expert_plus_arm_parameters.v1",
        common_config={
            "action_horizon": recipe.dataset.action_horizon,
            "policy": recipe.to_dict()["policy"],
            "recipe_sha256": recipe.recipe_sha256,
        },
        arm_config={
            "causal_factorization": dict(action_arm),
            "core": recipe.to_dict()["core"],
            "objective": recipe.to_dict()["objective"],
            "stationary_temporal_initialization": dict(stationary_core_binding),
            "vjepa2_cache": None if vjepa2_binding is None else dict(vjepa2_binding),
        },
    )
    contract.validate_plan(plan)
    return contract


def main() -> None:
    args = _parse_args()
    cloud_config = _read_json(args.cloud_config.resolve(), "cloud config")
    validate_config(cloud_config, root=_ROOT)
    recipe = load_training_recipe(args.recipe.resolve())
    if recipe.recipe_sha256 != cloud_config["training_recipe"]["sha256"]:
        raise ValueError("CLI recipe differs from the cloud profile")
    recipe.assert_optimizer_steps_authorized(args.total_steps)
    if args.checkpoint_every <= 0 or args.checkpoint_every > args.total_steps:
        raise ValueError("checkpoint_every must lie within the authorized run")
    if args.seed < 0 or args.global_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("seed and batch topology are invalid")
    m0_report, checkpoint_manifest_sha256 = _validate_m0_for_mode(
        report_path=args.m0_report,
        cloud_config=cloud_config,
        verify_only=args.verify_only,
    )
    accepted_temporal_core = _validate_stationary_core_for_mode(
        report_path=args.stationary_acceptance_report,
        checkpoint_path=args.stationary_checkpoint,
        verify_only=args.verify_only,
    )
    stationary_core_binding = (
        None if accepted_temporal_core is None else accepted_temporal_core.contract_dict()
    )
    from picf_next.training.molmoact2_calvin import (
        build_calvin_episode_stream_plan,
        build_molmoact2_calvin_training_stack,
        build_molmoact2_optimizer_and_scheduler,
        build_molmoact2_policy_config,
        load_calvin_training_assets,
    )

    assets = load_calvin_training_assets(
        recipe,
        repository_root=_ROOT,
        split_root=args.dataset_split_root,
    )
    action_arm = _action_arm_spec(args.arm)
    vjepa2_cache, vjepa2_binding = _load_vjepa2_cache_for_arm(
        arm_spec=action_arm,
        cache_root=args.vjepa2_cache_root,
        cache_manifest_sha256=args.vjepa2_cache_manifest_sha256,
        cache_memory_capacity=args.vjepa2_cache_memory_capacity,
        dataset_tree_sha256=assets.dataset_manifest.tree_sha256,
        require_persistent_root=not args.verify_only,
    )
    policy_config = build_molmoact2_policy_config(
        recipe,
        checkpoint_path=args.checkpoint_dir,
    )
    plan = build_calvin_episode_stream_plan(
        recipe,
        assets.dataset,
        comparison_id=args.comparison_id,
        seed=args.seed,
        global_batch_size=args.global_batch_size,
        total_steps=args.total_steps,
    )
    static_report = {
        "causal_factorization": action_arm,
        "artifacts": recipe.validate_repository_artifacts(_ROOT),
        "dataset_samples": len(assets.dataset),
        "episode_count": len(assets.dataset.episode_manifest),
        "m0_report_validated": m0_report is not None,
        "stationary_temporal_core_validated": accepted_temporal_core is not None,
        "stationary_temporal_initialization": stationary_core_binding,
        "plan_sha256": plan.plan_sha256,
        "recipe_sha256": recipe.recipe_sha256,
        "schema": "picf-next.molmoact2-calvin-m4-static.v1",
        "vjepa2_cache": vjepa2_binding,
    }
    if args.verify_only:
        print(json.dumps(static_report, sort_keys=True, separators=(",", ":")))
        return

    if m0_report is None or checkpoint_manifest_sha256 is None:  # pragma: no cover
        raise RuntimeError("training reached execution without an accepted M0 report")
    if accepted_temporal_core is None or stationary_core_binding is None:  # pragma: no cover
        raise RuntimeError("action training reached execution without an accepted Stage-B core")

    from accelerate import Accelerator
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    _validate_hardware(accelerator)
    if args.global_batch_size % (int(accelerator.num_processes) * args.gradient_accumulation_steps):
        raise ValueError("global batch is not divisible by rank and accumulation topology")

    distributed_main_process_call(
        accelerator,
        label="training checkpoint validation",
        action=partial(
            _validate_training_checkpoint,
            checkpoint_dir=args.checkpoint_dir.expanduser().resolve(),
            m0_report=m0_report,
            checkpoint_id=recipe.host.checkpoint_id,
            checkpoint_revision=recipe.host.checkpoint_revision,
        ),
    )
    code_revision = distributed_main_process_call(
        accelerator,
        label="source revision validation",
        action=partial(_git_revision, _ROOT),
    )

    run_root = args.run_root.expanduser().resolve()
    _validate_persistent_run_root(run_root)

    def initialize_run_metadata() -> None:
        if args.resume is None:
            _publish_fresh_run_metadata(
                run_root,
                static_report=static_report,
                sample_plan={"metadata": plan.metadata, "plan_sha256": plan.plan_sha256},
            )
        else:
            _validate_existing_json(
                run_root / "static_preflight.json",
                static_report,
                "static preflight",
            )
            plan_payload = _read_json(run_root / "sample_plan.json", "sample plan")
            if plan_payload.get("plan_sha256") != plan.plan_sha256:
                raise ValueError("existing sample plan differs from the active run")

    distributed_main_process_call(
        accelerator,
        label="run metadata initialization",
        action=initialize_run_metadata,
    )

    policy = MolmoAct2Policy(policy_config).to(accelerator.device).train()
    native_bank_builder = None
    native_history_frames = 1
    action_context_token_dims = None
    if vjepa2_cache is not None:
        from picf_next.hosts.vjepa2_context import CalvinVjepa2CachedContextBuilder

        policy_parameter = next(policy.parameters())
        native_bank_builder = CalvinVjepa2CachedContextBuilder(
            vjepa2_cache,
            device=accelerator.device,
            dtype=policy_parameter.dtype,
        )
        native_history_frames = native_bank_builder.maximum_source_frames
        action_context_token_dims = native_bank_builder.token_dims
    stack = build_molmoact2_calvin_training_stack(
        recipe,
        policy=policy,
        assets=assets,
        accepted_temporal_core=accepted_temporal_core,
        build_native_banks=native_bank_builder,
        native_evidence_history_frames=native_history_frames,
        action_context_token_dims=action_context_token_dims,
        include_posterior_action_context=bool(action_arm["include_posterior_action_context"]),
    )
    optimizer, scheduler = build_molmoact2_optimizer_and_scheduler(recipe, stack)
    model, optimizer, scheduler = accelerator.prepare(stack.module, optimizer, scheduler)

    contract = _build_picf_run_contract(
        recipe=recipe,
        plan=plan,
        assets=assets,
        args=args,
        code_revision=code_revision,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        stationary_core_binding=stationary_core_binding,
        action_arm=action_arm,
        vjepa2_binding=vjepa2_binding,
        world_size=int(accelerator.num_processes),
    )
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=args.global_batch_size,
    )
    register_progress_for_checkpointing(accelerator, progress)
    unwrapped = accelerator.unwrap_model(model)
    parameter = next(unwrapped.parameters())
    stream_state = PosteriorStreamStateGroup.for_rank_partition(
        recipe.core_config.temporal,
        plan,
        rank=int(accelerator.process_index),
        world_size=int(accelerator.num_processes),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        capacity=recipe.core_config.posterior_capacity,
        device=accelerator.device,
        dtype=parameter.dtype,
        max_parameter_lag=0,
    )
    metrics_path = run_root / "metrics.jsonl"
    if args.resume is not None:
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=args.resume.resolve(),
            contract=contract,
            plan=plan,
            progress=progress,
            rank_state=stream_state,
        )
        distributed_main_process_call(
            accelerator,
            label="metrics resume reconciliation",
            action=lambda: _reconcile_metrics_for_resume(
                metrics_path,
                attempted_optimizer_steps=progress.attempted_optimizer_steps,
                successful_optimizer_steps=progress.successful_optimizer_steps,
            ),
        )
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=unwrapped.joint_bridge.sequence_bridge.core,
        optimizer=optimizer,
        plan=plan,
        progress=progress,
        stream_state=stream_state,
        lr_scheduler=scheduler,
        max_grad_norm=recipe.optimizer.gradient_clip_norm,
    )

    while progress.next_plan_step < args.total_steps:
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        _synchronize_step_timing(accelerator)
        step_started = time.perf_counter()
        result = runner.run_optimizer_step(model)
        scheduler_epoch = _validate_scheduler_epoch(
            scheduler,
            successful_optimizer_steps=progress.successful_optimizer_steps,
        )
        _synchronize_step_timing(accelerator)
        elapsed_seconds = time.perf_counter() - step_started
        step_observability = _optimizer_step_observability(result, optimizer)
        local_metrics = []
        for value in result.metrics:
            combined = dict(value)
            combined.update(step_observability)
            local_metrics.append(combined)
        reduced = _mean_metrics(accelerator, tuple(local_metrics))
        reduced.update(
            _distributed_system_telemetry(
                accelerator,
                elapsed_seconds=elapsed_seconds,
            )
        )
        reduced["system_scheduler_epoch"] = float(scheduler_epoch)
        metrics_payload = {
            "attempted_optimizer_steps": progress.attempted_optimizer_steps,
            "metrics": reduced,
            "optimizer_step_skipped": result.optimizer_step_was_skipped,
            "successful_optimizer_steps": progress.successful_optimizer_steps,
        }
        distributed_main_process_call(
            accelerator,
            label="metrics append",
            action=partial(_append_metrics, metrics_path, metrics_payload),
        )
        should_checkpoint = (
            progress.attempted_optimizer_steps % args.checkpoint_every == 0
            or progress.attempted_optimizer_steps == args.total_steps
        )
        if should_checkpoint:
            save_accelerate_checkpoint(
                accelerator=accelerator,
                checkpoint_dir=(
                    run_root / "checkpoints" / f"step-{progress.attempted_optimizer_steps:08d}"
                ),
                contract=contract,
                plan=plan,
                progress=progress,
                rank_state=stream_state,
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
