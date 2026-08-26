#!/usr/bin/env python3
"""Audit M2 loss gradients and step cost on the immutable real-feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder  # noqa: E402
from picf_next.models.discovery import ObjectExistenceCalibration  # noqa: E402
from picf_next.models.set_loss import ObjectSetCriterion  # noqa: E402
from picf_next.training.molmoact2_calvin import load_calvin_training_assets  # noqa: E402
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _build_targets,
    _keys_for_split,
    _load_cache,
    _native_bank,
    _stack_batch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_representation.json",
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "heldout"), default="train")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--timing-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--existence-weight", type=float)
    parser.add_argument("--ownership-ce-weight", type=float)
    parser.add_argument("--ownership-dice-weight", type=float)
    parser.add_argument("--unmatched-query-weight", type=float)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_group(name: str) -> str:
    if name.startswith("projection.") or name.startswith("projector."):
        return "projection"
    if "geometry_log_variance_head" in name or "variance_head" in name:
        return "geometry_variance_head"
    if "geometry_mean_head" in name or "geometry_head" in name:
        return "geometry_mean_head"
    if "existence_head" in name:
        return "existence_head"
    if "ownership_query" in name or "context_head" in name:
        return "ownership_heads"
    if "address_head" in name:
        return "address_head"
    if "content_head" in name:
        return "content_head"
    if name.startswith("discovery."):
        return "discovery_shared"
    return "other"


def _gradient_statistics(
    *,
    named_parameters: Sequence[tuple[str, Any]],
    gradients: Sequence[Any | None],
) -> dict[str, Any]:
    import torch

    squared_norm = torch.zeros((), device=named_parameters[0][1].device, dtype=torch.float64)
    maximum_absolute = torch.zeros_like(squared_norm)
    nonfinite = 0
    group_squared: dict[str, Any] = {}
    per_parameter: list[tuple[str, float]] = []
    for (name, _parameter), gradient in zip(named_parameters, gradients, strict=True):
        if gradient is None:
            continue
        detached = gradient.detach().double()
        finite = torch.isfinite(detached)
        nonfinite += int((~finite).sum().item())
        if not finite.all():
            detached = torch.where(finite, detached, torch.zeros_like(detached))
        square = detached.square().sum()
        squared_norm = squared_norm + square
        maximum_absolute = torch.maximum(maximum_absolute, detached.abs().max())
        group = _parameter_group(name)
        group_squared[group] = group_squared.get(group, torch.zeros_like(square)) + square
        per_parameter.append((name, math.sqrt(float(square.item()))))
    per_parameter.sort(key=lambda item: item[1], reverse=True)
    return {
        "l2_norm": math.sqrt(float(squared_norm.item())),
        "maximum_absolute_gradient": float(maximum_absolute.item()),
        "nonfinite_gradient_elements": nonfinite,
        "group_l2_norm": {
            name: math.sqrt(float(value.item())) for name, value in sorted(group_squared.items())
        },
        "top_parameter_l2_norm": [
            {"name": name, "l2_norm": value} for name, value in per_parameter[:12]
        ],
    }


def _gradient_dot(left: Sequence[Any | None], right: Sequence[Any | None]) -> float:
    total: Any | None = None
    for first, second in zip(left, right, strict=True):
        if first is None or second is None:
            continue
        term = (first.detach().double() * second.detach().double()).sum()
        total = term if total is None else total + term
    return 0.0 if total is None else float(total.item())


def _loss_components(result: Any, criterion: ObjectSetCriterion) -> dict[str, Any]:
    config = criterion.config
    losses = result.losses
    return {
        "weighted_existence": config.existence_weight * losses["loss_existence"],
        "weighted_localization_confidence": (
            config.localization_confidence_weight * losses["loss_localization_confidence"]
        ),
        "weighted_ownership_ce": config.ownership_ce_weight * losses["loss_ownership_ce"],
        "weighted_ownership_dice": (config.ownership_dice_weight * losses["loss_ownership_dice"]),
        "weighted_address": config.address_cosine_weight * losses["loss_address_cosine"],
        "weighted_content": config.content_cosine_weight * losses["loss_content_cosine"],
        "weighted_geometry": config.geometry_weight * losses["loss_geometry"],
        "diagnostic_geometry_mean": losses["loss_geometry_mean"],
        "diagnostic_geometry_calibration": losses["loss_geometry_calibration"],
        "total": result.total,
    }


def _audit_gradients(
    *,
    model: Any,
    criterion: ObjectSetCriterion,
    tokens: Any,
    valid: Any,
    targets: Sequence[Any],
    gradient_clip_norm: float,
) -> dict[str, Any]:
    import torch

    named_parameters = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    parameters = tuple(parameter for _name, parameter in named_parameters)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(_native_bank(tokens, valid))
    result = criterion(output.discovery, targets)
    components = _loss_components(result, criterion)
    total_gradients = torch.autograd.grad(
        components["total"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total_statistics = _gradient_statistics(
        named_parameters=named_parameters,
        gradients=total_gradients,
    )
    total_norm = total_statistics["l2_norm"]
    clip_multiplier = min(1.0, gradient_clip_norm / max(total_norm, 1e-12))
    component_reports: dict[str, Any] = {}
    for index, (name, loss) in enumerate(components.items()):
        if name == "total":
            continue
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index + 1 < len(components),
            allow_unused=True,
        )
        statistics = _gradient_statistics(
            named_parameters=named_parameters,
            gradients=gradients,
        )
        component_norm = statistics["l2_norm"]
        denominator = max(component_norm * total_norm, 1e-12)
        statistics.update(
            {
                "loss": float(loss.detach().float().item()),
                "cosine_with_total_gradient": _gradient_dot(gradients, total_gradients)
                / denominator,
                "l2_norm_after_total_global_clip": component_norm * clip_multiplier,
            }
        )
        component_reports[name] = statistics
        del gradients
    model.zero_grad(set_to_none=True)
    return {
        "losses": {
            name: float(value.detach().float().item()) for name, value in result.losses.items()
        },
        "total_gradient": total_statistics,
        "global_clip_norm": gradient_clip_norm,
        "global_clip_multiplier": clip_multiplier,
        "components": component_reports,
    }


def _target_statistics(targets: Sequence[Any]) -> dict[str, Any]:
    inventory = [target.num_objects for target in targets]
    supervised_rows = [target.supervision_valid for target in targets]
    object_mass = []
    context_mass = []
    for target in targets:
        valid = target.supervision_valid
        ownership = target.ownership[valid].detach().float()
        object_mass.append(float(ownership[:, :-1].sum().item()))
        context_mass.append(float(ownership[:, -1].sum().item()))
    return {
        "inventory_object_count": inventory,
        "supervised_token_rows": [int(value.sum().item()) for value in supervised_rows],
        "object_ownership_mass": object_mass,
        "context_ownership_mass": context_mass,
    }


def _time_step(
    *,
    model: Any,
    criterion: ObjectSetCriterion,
    target_builder: CalvinVisibleObjectTargetBuilder,
    records: Sequence[Mapping[str, Any]],
    tokens: Any,
    valid: Any,
    layout_payload: Sequence[Mapping[str, Any]],
    token_count: int,
    gradient_clip_norm: float,
    warmup_steps: int,
    timing_steps: int,
) -> dict[str, Any]:
    import torch

    if warmup_steps < 0 or timing_steps <= 0:
        raise ValueError("timing steps must be positive and warmup steps nonnegative")
    rows = []
    torch.cuda.reset_peak_memory_stats(tokens.device)
    for iteration in range(warmup_steps + timing_steps):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(tokens.device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(_native_bank(tokens, valid))
        torch.cuda.synchronize(tokens.device)
        after_forward = time.perf_counter()
        targets = _build_targets(
            target_builder=target_builder,
            records=records,
            token_valid=output.projection.token_valid,
            target_dtype=output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=token_count,
        )
        after_targets = time.perf_counter()
        result = criterion(output.discovery, targets)
        torch.cuda.synchronize(tokens.device)
        after_criterion = time.perf_counter()
        result.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        torch.cuda.synchronize(tokens.device)
        finished = time.perf_counter()
        if iteration >= warmup_steps:
            rows.append(
                {
                    "forward_seconds": after_forward - started,
                    "target_seconds": after_targets - after_forward,
                    "criterion_seconds": after_criterion - after_targets,
                    "backward_and_clip_seconds": finished - after_criterion,
                    "total_seconds": finished - started,
                    "gradient_norm": float(gradient_norm.detach().float().item()),
                }
            )
    model.zero_grad(set_to_none=True)
    means = {
        name: sum(row[name] for row in rows) / len(rows)
        for name in (
            "forward_seconds",
            "target_seconds",
            "criterion_seconds",
            "backward_and_clip_seconds",
            "total_seconds",
            "gradient_norm",
        )
    }
    return {
        "warmup_steps": warmup_steps,
        "timing_steps": timing_steps,
        "mean": means,
        "rows": rows,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(tokens.device)),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    import torch

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M2 gradient audit requires CUDA")
    device = torch.device(args.device)
    recipe = load_molmoact2_m2_recipe(args.config.resolve())
    foundation = recipe.load_foundation(_ROOT)
    weight_overrides = {
        "existence_weight": args.existence_weight,
        "ownership_ce_weight": args.ownership_ce_weight,
        "ownership_dice_weight": args.ownership_dice_weight,
    }
    applied_weight_overrides = {
        name: value for name, value in weight_overrides.items() if value is not None
    }
    if applied_weight_overrides:
        foundation = replace(
            foundation,
            set_loss_config=replace(
                foundation.set_loss_config,
                **applied_weight_overrides,
            ),
        )
    if args.unmatched_query_weight is not None:
        foundation = replace(
            foundation,
            core_config=replace(
                foundation.core_config,
                discovery=replace(
                    foundation.core_config.discovery,
                    existence_calibration=ObjectExistenceCalibration(args.unmatched_query_weight),
                ),
            ),
        )
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=args.dataset_split_root.expanduser().resolve(),
    )
    cache_manifest, cache = _load_cache(args.feature_cache.expanduser().resolve(), recipe)
    keys = _keys_for_split(cache, args.split)
    batch_size = args.batch_size or recipe.optimization.batch_size
    if batch_size <= 0 or batch_size > len(keys):
        raise ValueError("batch size is outside the selected split")
    selected = keys[:batch_size]
    tokens, valid, records = _stack_batch(cache, selected, device=device)

    torch.manual_seed(recipe.optimization.seed)
    model = foundation.core_config.build_current_frame().to(device)
    checkpoint_identity = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model"] if set(payload) == {"model"} else payload
        model.load_state_dict(state, strict=True)
        checkpoint_identity = {"path": str(checkpoint), "sha256": _sha256(checkpoint)}
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(device)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        initial_output = model(_native_bank(tokens, valid))
    targets = _build_targets(
        target_builder=target_builder,
        records=records,
        token_valid=initial_output.projection.token_valid,
        target_dtype=initial_output.discovery.ownership.dtype,
        layout_payload=cache_manifest["processor_layout"],
        token_count=recipe.cache.token_count,
    )
    report = {
        "schema": "picf-next.molmoact2-m2-gradient-audit.v1",
        "config": str(args.config.resolve()),
        "recipe_sha256": recipe.recipe_sha256,
        "development_set_loss_weight_overrides": applied_weight_overrides,
        "development_unmatched_query_weight_override": args.unmatched_query_weight,
        "feature_cache": str(args.feature_cache.expanduser().resolve()),
        "feature_cache_manifest_sha256": _sha256(
            args.feature_cache.expanduser().resolve() / "manifest.json"
        ),
        "checkpoint": checkpoint_identity,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "split": args.split,
        "sample_keys": selected,
        "target_statistics": _target_statistics(targets),
        "gradient_audit": _audit_gradients(
            model=model,
            criterion=criterion,
            tokens=tokens,
            valid=valid,
            targets=targets,
            gradient_clip_norm=recipe.optimization.gradient_clip_norm,
        ),
        "timing": _time_step(
            model=model,
            criterion=criterion,
            target_builder=target_builder,
            records=records,
            tokens=tokens,
            valid=valid,
            layout_payload=cache_manifest["processor_layout"],
            token_count=recipe.cache.token_count,
            gradient_clip_norm=recipe.optimization.gradient_clip_norm,
            warmup_steps=args.warmup_steps,
            timing_steps=args.timing_steps,
        ),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
