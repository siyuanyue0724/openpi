#!/usr/bin/env python3
"""Probe task-bucket gradient compatibility on the live PICF training graph.

This is a diagnostic, not a training script.  It loads the same args/checkpoint
as a training run, samples real CALVIN windows per task bucket, runs the normal
PICF forward/loss path, and estimates pairwise gradient cosine on deterministic
sampled parameter subspaces.  The goal is to decide whether dynamic mixing or
adapter-level gradient surgery is justified before adding either to production.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable
import types

import numpy as np
import torch

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# Some local dev environments import tqdm_loggable before training starts.
# tqdm_loggable probes IPython and can trip a wandb-vendored pygments bug before
# our CLI even parses arguments.  The training path does not require IPython, so
# the diagnostic CLI disables that optional notebook probe by default.
if os.environ.get("PICF_PROBE_ALLOW_IPYTHON", "0") != "1" and "IPython" not in sys.modules:
    _ipython_stub = types.ModuleType("IPython")
    _ipython_stub.get_ipython = lambda: None  # type: ignore[attr-defined]
    sys.modules["IPython"] = _ipython_stub

from scripts.picf_action_bridge_capacity_probe import (  # noqa: E402
    _DeterministicFlowRng,
    _PicfRuntimeStateGuard,
    _reset_picf_runtime_buffers,
)
from scripts.picf_core_train import (  # noqa: E402
    _apply_picf_trainable_scope,
    _build_loss_config,
    _build_model,
    _build_optimizer,
    _CalvinTransitionSource,
    _ensure_window_has_valid_first_step_xyzrgb_support,
    _is_retryable_first_step_error,
    _load_checkpoint,
    _load_tactile_backgrounds_npz,
    _materialize_model_parameters,
    _PicfWindowTrainer,
    _resolve_action_normalizer,
    _seed_everything,
)
from scripts.picf_fixed_window_action_probe import _numeric_snapshot  # noqa: E402
from scripts.picf_replay_windows import _coerce_loaded_args, _resolve_rank_seed  # noqa: E402
from scripts.sonata_window_probe import _override_build_sample  # noqa: E402


def _set_override(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _canonical_name(name: str) -> str:
    parts = str(name).split(".")
    while parts and parts[0] in {"module", "_fsdp_wrapped_module"}:
        parts = parts[1:]
    return ".".join(parts)


def _param_group_name(name: str) -> str:
    canonical = _canonical_name(name)
    if canonical.startswith("core."):
        return "picf_core"
    if canonical.startswith("semantic_encoder."):
        return "semantic"
    return "policy_head"


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _trainable_group_numel(model: torch.nn.Module, groups: set[str]) -> dict[str, int]:
    counts = {group: 0 for group in groups}
    for name, param in model.named_parameters():
        group = _param_group_name(name)
        if group not in groups or not bool(getattr(param, "requires_grad", False)):
            continue
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            continue
        counts[group] += int(param.numel())
    return counts


def _sample_positions(
    *,
    total_numel: int,
    max_elements: int,
    seed: int,
) -> torch.Tensor:
    total = int(total_numel)
    if total <= 0:
        return torch.empty((0,), dtype=torch.long)
    count = min(int(max_elements), total)
    rng = np.random.default_rng(int(seed))
    if count >= total:
        return torch.arange(total, dtype=torch.long)
    positions = np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))
    return torch.from_numpy(positions)


def _sampled_grad_vector(
    model: torch.nn.Module,
    *,
    group: str,
    positions: torch.Tensor,
) -> torch.Tensor:
    if positions.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    out = torch.zeros((int(positions.numel()),), dtype=torch.float32)
    cursor = 0
    write_cursor = 0
    pos_np = positions.cpu().numpy()
    for name, param in model.named_parameters():
        if _param_group_name(name) != group or not bool(getattr(param, "requires_grad", False)):
            continue
        if isinstance(param, torch.nn.parameter.UninitializedParameter):
            continue
        numel = int(param.numel())
        start = cursor
        end = cursor + numel
        left = int(np.searchsorted(pos_np, start, side="left"))
        right = int(np.searchsorted(pos_np, end, side="left"))
        if right > left:
            local_indices = torch.from_numpy(pos_np[left:right] - start).to(device=param.device, dtype=torch.long)
            grad = getattr(param, "grad", None)
            if grad is not None:
                values = grad.detach().reshape(-1).index_select(0, local_indices).to(device="cpu", dtype=torch.float32)
                out[write_cursor : write_cursor + int(values.numel())] = values
            write_cursor += int(right - left)
        cursor = end
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.numel() == 0 or b.numel() == 0:
        return None
    denom = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
    if denom <= 0.0 or not math.isfinite(denom):
        return None
    return float(torch.dot(a, b).item() / denom)


def _sample_bucket_window(
    source: _CalvinTransitionSource,
    *,
    bucket: str,
    seed: int,
    attempt_base: int,
    trainer: _PicfWindowTrainer,
) -> Any:
    candidates = list(source.bucket_to_slot_indices.get(str(bucket), []))
    if not candidates:
        raise RuntimeError(f"No candidate segments for bucket {bucket!r}.")
    for attempt in range(64):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(attempt_base), int(attempt)]))
        slot = int(candidates[int(rng.integers(0, len(candidates)))])
        window = source.window(slot, rng=rng)
        try:
            _ensure_window_has_valid_first_step_xyzrgb_support(trainer, window)
            return window
        except RuntimeError as exc:
            if _is_retryable_first_step_error(exc):
                continue
            raise
    raise RuntimeError(f"Could not sample a valid first-step-support window for bucket {bucket!r}.")


def _matrix_rows(vectors: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = sorted(vectors)
    matrix: dict[str, dict[str, float | None]] = {}
    negatives = 0
    finite = 0
    min_cos: float | None = None
    for a in names:
        row: dict[str, float | None] = {}
        for b in names:
            value = _cosine(vectors[a], vectors[b])
            row[b] = value
            if a != b and value is not None:
                finite += 1
                if value < 0.0:
                    negatives += 1
                min_cos = value if min_cos is None else min(min_cos, value)
        matrix[a] = row
    return {
        "matrix": matrix,
        "finite_pairs": int(finite),
        "negative_pairs": int(negatives),
        "negative_fraction": None if finite == 0 else float(negatives / finite),
        "min_cosine": min_cos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PICF task-bucket sampled gradient-cosine diagnostic.")
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--rank-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--buckets", default="")
    parser.add_argument("--windows-per-bucket", type=int, default=1)
    parser.add_argument("--loss-keys", default="loss_action")
    parser.add_argument("--groups", default="policy_head,picf_core")
    parser.add_argument("--max-elements-per-group", type=int, default=1_000_000)
    parser.add_argument("--deterministic-flow-seed", type=int, default=314159)
    parser.add_argument("--picf-trainable-scope", choices=("all", "policy_only", "anchor_only"), default="all")
    parser.add_argument(
        "--semantic-trainable-scope",
        choices=("action_adapter_only", "action_head_only", "action_head_and_adapter", "backbone_only", "all"),
        default="action_head_and_adapter",
    )
    parser.add_argument("--semantic-lr-scale", type=float, default=1.0)
    parser.add_argument("--policy-head-lr-scale", type=float, default=1.0)
    parser.add_argument("--picf-core-lr-scale", type=float, default=1.0)
    parser.add_argument("--action-context-integration", choices=("prefix_fusion", "suffix_cross_attention", "append"), default=None)
    parser.add_argument("--action-context-tokens", type=int, default=None)
    parser.add_argument("--enable-picf-action-condition", action="store_true")
    parser.add_argument("--disable-picf-action-condition", action="store_true")
    parser.add_argument("--point-grid-mode", choices=("default", "original", "rebased"), default="default")
    args = parser.parse_args()

    started = time.time()
    payload = json.loads(Path(args.args_json).read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=args.device)
    if args.split is not None:
        train_args.split = str(args.split)
    _set_override(train_args, "training_strategy", "ddp")
    _set_override(train_args, "optimizer_sharding", "none")
    _set_override(train_args, "picf_trainable_scope", str(args.picf_trainable_scope))
    _set_override(train_args, "semantic_trainable", True)
    _set_override(train_args, "semantic_trainable_scope", str(args.semantic_trainable_scope))
    _set_override(train_args, "semantic_lr_scale", float(args.semantic_lr_scale))
    _set_override(train_args, "policy_head_lr_scale", float(args.policy_head_lr_scale))
    _set_override(train_args, "picf_core_lr_scale", float(args.picf_core_lr_scale))
    if args.action_context_integration is not None:
        _set_override(train_args, "action_context_integration", str(args.action_context_integration))
    if args.action_context_tokens is not None:
        _set_override(train_args, "action_context_tokens", int(args.action_context_tokens))
    if bool(args.enable_picf_action_condition) and bool(args.disable_picf_action_condition):
        raise ValueError("Cannot pass both --enable-picf-action-condition and --disable-picf-action-condition.")
    if bool(args.enable_picf_action_condition):
        _set_override(train_args, "picf_action_condition_enabled", True)
    if bool(args.disable_picf_action_condition):
        _set_override(train_args, "picf_action_condition_enabled", False)

    device = torch.device(str(train_args.device))
    rank_seed = _resolve_rank_seed(rank_seed=args.rank_seed, rng_rank=None)
    _seed_everything(int(train_args.seed), int(rank_seed))
    action_normalizer = _resolve_action_normalizer(train_args)
    calvin_segment_indices = None
    if getattr(train_args, "calvin_segment_indices", None):
        calvin_segment_indices = [
            int(part)
            for part in str(getattr(train_args, "calvin_segment_indices")).split(",")
            if part.strip()
        ]

    override_context = (
        _override_build_sample(str(args.point_grid_mode))
        if str(args.point_grid_mode) != "default"
        else None
    )
    context_manager = override_context if override_context is not None else torch.no_grad()
    # torch.no_grad is not entered here; it is only used as a no-op context when
    # no point-grid override is required.
    if override_context is None:
        class _Null:
            def __enter__(self) -> None:
                return None
            def __exit__(self, *exc: object) -> bool:
                return False
        context_manager = _Null()

    with context_manager:
        source = _CalvinTransitionSource(
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
            action_normalizer=action_normalizer,
            augmentation_mode=train_args.picf_augmentation_mode,
            photometric_strength=train_args.picf_photometric_strength,
            segment_indices=calvin_segment_indices,
            bucket_sampling_mode=getattr(train_args, "calvin_bucket_sampling_mode", "round_robin"),
            bucket_temperature_alpha=float(getattr(train_args, "calvin_bucket_temperature_alpha", 0.0)),
            bucket_weight_spec=str(getattr(train_args, "calvin_bucket_weight_spec", "")),
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
        trainable_scope = _apply_picf_trainable_scope(trainer, args=train_args, logger=None)
        optimizer, optimizer_groups = _build_optimizer(trainer, args=train_args)
        loaded_step = _load_checkpoint(
            path=Path(args.checkpoint),
            model=trainer,
            optimizer=optimizer,
            device=device,
            optimizer_checkpoint_mode="model_only",
        )
        trainer.train()

        requested_buckets = _parse_csv(args.buckets) or list(source.bucket_names)
        buckets = [bucket for bucket in requested_buckets if bucket in source.bucket_to_slot_indices]
        if len(buckets) < 2:
            raise RuntimeError(f"Need at least two valid buckets for cosine, got {buckets!r}.")
        loss_keys = _parse_csv(args.loss_keys)
        if not loss_keys:
            raise ValueError("--loss-keys must contain at least one scalar loss key.")
        groups = set(_parse_csv(args.groups))
        if not groups:
            raise ValueError("--groups must contain at least one parameter group.")
        group_numel = _trainable_group_numel(trainer, groups)
        sampled_positions = {
            group: _sample_positions(
                total_numel=int(group_numel.get(group, 0)),
                max_elements=int(args.max_elements_per_group),
                seed=int(args.seed) + 1009 * index,
            )
            for index, group in enumerate(sorted(groups))
        }

        result: dict[str, Any] = {
            "stage": "bucket_gradient_cosine_probe",
            "checkpoint": str(args.checkpoint),
            "loaded_step": int(loaded_step),
            "args_json": str(args.args_json),
            "buckets": buckets,
            "loss_keys": loss_keys,
            "groups": sorted(groups),
            "group_numel": group_numel,
            "sampled_elements": {group: int(pos.numel()) for group, pos in sampled_positions.items()},
            "trainable_scope": trainable_scope,
            "optimizer_groups": optimizer_groups,
            "records": {},
            "elapsed_s": None,
        }

        for loss_key in loss_keys:
            loss_records: dict[str, Any] = {}
            vectors_by_group: dict[str, dict[str, torch.Tensor]] = {group: {} for group in sorted(groups)}
            for bucket_index, bucket in enumerate(buckets):
                bucket_snapshots: list[dict[str, float]] = []
                bucket_vectors: dict[str, list[torch.Tensor]] = {group: [] for group in sorted(groups)}
                for window_index in range(int(args.windows_per_bucket)):
                    window = _sample_bucket_window(
                        source,
                        bucket=bucket,
                        seed=int(args.seed),
                        attempt_base=10000 * bucket_index + window_index,
                        trainer=trainer,
                    )
                    trainer.zero_grad(set_to_none=True)
                    with _PicfRuntimeStateGuard(trainer):
                        _reset_picf_runtime_buffers(trainer)
                        with _DeterministicFlowRng(
                            trainer.semantic_encoder,
                            seed=int(args.deterministic_flow_seed) + 1000 * bucket_index + window_index,
                            device=device,
                        ):
                            outputs = trainer(window, capture_visual_diagnostics=False)
                        loss = outputs.get(loss_key)
                        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                            raise RuntimeError(f"Missing scalar loss key {loss_key!r} for bucket {bucket!r}.")
                        if not bool(loss.requires_grad):
                            raise RuntimeError(f"Loss key {loss_key!r} does not require grad.")
                        loss.backward()
                        bucket_snapshots.append(_numeric_snapshot(outputs))
                        for group in sorted(groups):
                            bucket_vectors[group].append(
                                _sampled_grad_vector(trainer, group=group, positions=sampled_positions[group])
                            )
                    trainer.zero_grad(set_to_none=True)
                merged_vectors: dict[str, torch.Tensor] = {}
                for group, group_vectors in bucket_vectors.items():
                    if not group_vectors:
                        merged_vectors[group] = torch.empty((0,), dtype=torch.float32)
                    else:
                        merged_vectors[group] = torch.stack(group_vectors, dim=0).mean(dim=0)
                        vectors_by_group[group][bucket] = merged_vectors[group]
                loss_records[bucket] = {
                    "windows": int(args.windows_per_bucket),
                    "prompt": str(window.prompt),
                    "segment": int(window.segment_id),
                    "start_step": int(window.start_step_id),
                    "losses": {
                        key: float(sum(s.get(key, float("nan")) for s in bucket_snapshots) / len(bucket_snapshots))
                        for key in sorted(set().union(*(snapshot.keys() for snapshot in bucket_snapshots)))
                        if all(isinstance(snapshot.get(key), (int, float)) for snapshot in bucket_snapshots)
                    },
                    "grad_norm": {
                        group: float(torch.linalg.vector_norm(merged_vectors[group]).item())
                        for group in sorted(groups)
                    },
                }
            group_cosines = {
                group: _matrix_rows(vectors_by_group[group])
                for group in sorted(groups)
            }
            result["records"][loss_key] = {
                "buckets": loss_records,
                "cosine_by_group": group_cosines,
            }

        result["elapsed_s"] = round(time.time() - started, 3)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
