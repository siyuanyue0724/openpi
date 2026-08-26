#!/usr/bin/env python3
"""Test whether final M2 residual uncertainty is learnable with a frozen mean."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

_SCHEMA = "picf-next.molmoact2-m2-frozen-mean-variance-probe.v1"
_CONFIG_SCHEMA = "picf-next.molmoact2-m2-frozen-mean-variance-probe-config.v1"
_OUTPUT_NAME = "frozen_mean_variance_probe"
_VARIANCE_PREFIX = "discovery.variance_head."
_REPRESENTATION_METRICS = (
    "count_mae",
    "exact_count_accuracy",
    "ownership_accuracy",
    "token_ownership_accuracy",
    "context_accuracy",
    "mean_object_dice",
    "geometry_mae_model_chart",
    "geometry_mae_physical",
    "fragmentation_excess_per_object",
    "maximum_active_query_pair_dice",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(_ROOT / "configs/training/molmoact2_calvin_m2_frozen_mean_variance_probe.json"),
    )
    return parser.parse_args()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_probe_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="ascii") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != _CONFIG_SCHEMA:
        raise ValueError("frozen-mean variance probe config schema changed")
    if payload.get("gate") != "M2_frozen_mean_variance_diagnostic":
        raise ValueError("frozen-mean variance probe gate changed")
    optimization = payload.get("optimization")
    protocol = payload.get("protocol")
    acceptance = payload.get("acceptance")
    if not isinstance(optimization, dict):
        raise ValueError("probe config requires an optimization object")
    if not isinstance(protocol, dict):
        raise ValueError("probe config requires a protocol object")
    if not isinstance(acceptance, dict):
        raise ValueError("probe config requires an acceptance object")
    _positive_int(optimization.get("batch_size"), "batch_size")
    _positive_int(optimization.get("passes"), "passes")
    _positive_int(optimization.get("validation_interval"), "validation_interval")
    _positive_int(optimization.get("warmup_steps"), "warmup_steps")
    seed = optimization.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name in ("learning_rate", "gradient_clip_norm"):
        if _finite_float(optimization.get(name), name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if _finite_float(optimization.get("weight_decay"), "weight_decay") < 0.0:
        raise ValueError("weight_decay must be nonnegative")
    final_multiplier = _finite_float(
        optimization.get("final_learning_rate_multiplier"),
        "final_learning_rate_multiplier",
    )
    if not 0.0 < final_multiplier <= 1.0:
        raise ValueError("final_learning_rate_multiplier must lie in (0, 1]")
    expected_protocol = {
        "checkpoint_selection": "fixed-final-step-no-reselection",
        "control": "same-inputs-deterministic-deranged-targets",
        "initialization": "reset-both-variance-heads-to-declared-initial-variance",
        "mean_and_representation": "frozen-exactly",
        "training_data_exposure": "one-complete-deterministic-pass",
        "updated_parameter_prefixes": [_VARIANCE_PREFIX],
    }
    if protocol != expected_protocol:
        raise ValueError("frozen-mean variance protocol changed")
    maximum_drift = _finite_float(
        acceptance.get("maximum_representation_metric_absolute_drift"),
        "maximum_representation_metric_absolute_drift",
    )
    if maximum_drift < 0.0:
        raise ValueError("maximum representation metric drift must be nonnegative")
    minimum_margin = _finite_float(
        acceptance.get("minimum_aligned_control_heldout_spearman_margin"),
        "minimum_aligned_control_heldout_spearman_margin",
    )
    if not -2.0 <= minimum_margin <= 2.0:
        raise ValueError("aligned-control Spearman margin must lie in [-2, 2]")
    minimum_spearman = _finite_float(
        acceptance.get("minimum_uncertainty_error_spearman"),
        "minimum_uncertainty_error_spearman",
    )
    if not -1.0 <= minimum_spearman <= 1.0:
        raise ValueError("minimum Spearman must lie in [-1, 1]")
    return payload


def _complete_pass_plan(
    keys: Sequence[str],
    *,
    batch_size: int,
    seed: int,
    passes: int,
) -> list[list[str]]:
    """Return deterministic complete passes without replacement within a pass."""

    if not keys:
        raise ValueError("complete-pass plan requires at least one key")
    if len(set(keys)) != len(keys):
        raise ValueError("complete-pass plan keys must be unique")
    if batch_size <= 0 or passes <= 0:
        raise ValueError("batch_size and passes must be positive")
    plan: list[list[str]] = []
    for epoch in range(passes):
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(f"{seed}:{epoch}:{key}".encode()).digest(),
        )
        plan.extend(
            ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)
        )
    return plan


def _learning_rate_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    final_multiplier: float,
) -> float:
    if not 1 <= step <= total_steps:
        raise ValueError("step must lie in [1, total_steps]")
    if warmup_steps <= 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must lie in [1, total_steps)")
    if not 0.0 < final_multiplier <= 1.0:
        raise ValueError("final_multiplier must lie in (0, 1]")
    if step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return final_multiplier + (1.0 - final_multiplier) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _state_without_variance(model: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith(_VARIANCE_PREFIX)
    }


def _variance_state(model: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(_VARIANCE_PREFIX)
    }


def _reset_variance_head(model: Any, foundation: Any) -> None:
    fresh = foundation.core_config.build_current_frame()
    model.discovery.variance_head.load_state_dict(
        fresh.discovery.variance_head.state_dict(),
        strict=True,
    )


def _representation_metric_drift(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    return {
        name: abs(
            _finite_float(reference.get(name), f"reference.{name}")
            - _finite_float(candidate.get(name), f"candidate.{name}")
        )
        for name in _REPRESENTATION_METRICS
    }


def _spearman(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get("uncertainty_error_spearman")
    if value is None:
        return None
    return _finite_float(value, name)


def _decision(
    *,
    baseline_reset: Mapping[str, Mapping[str, Any]],
    aligned: Mapping[str, Mapping[str, Any]],
    control_heldout: Mapping[str, Any],
    frozen_state_exact: bool,
    representation_metric_maximum_drift: float,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    minimum = _finite_float(
        acceptance["minimum_uncertainty_error_spearman"],
        "minimum_uncertainty_error_spearman",
    )
    margin_minimum = _finite_float(
        acceptance["minimum_aligned_control_heldout_spearman_margin"],
        "minimum_aligned_control_heldout_spearman_margin",
    )
    maximum_drift = _finite_float(
        acceptance["maximum_representation_metric_absolute_drift"],
        "maximum_representation_metric_absolute_drift",
    )
    ranks = {
        split: _spearman(aligned[split], f"aligned.{split}.spearman")
        for split in ("train", "validation", "heldout")
    }
    control_rank = _spearman(control_heldout, "control.heldout.spearman")
    aligned_rank = ranks["heldout"]
    control_effective_rank = 0.0 if control_rank is None else control_rank
    margin = None if aligned_rank is None else aligned_rank - control_effective_rank
    reset_loss = _finite_float(
        baseline_reset["heldout"]["losses"]["loss_geometry_calibration"],
        "baseline_reset.heldout.loss_geometry_calibration",
    )
    aligned_loss = _finite_float(
        aligned["heldout"]["losses"]["loss_geometry_calibration"],
        "aligned.heldout.loss_geometry_calibration",
    )
    checks = {
        "frozen_non_variance_state_exact": frozen_state_exact,
        "representation_metrics_preserved": (representation_metric_maximum_drift <= maximum_drift),
        "aligned_train_uncertainty_ranks_errors": (
            ranks["train"] is not None and ranks["train"] >= minimum
        ),
        "aligned_validation_uncertainty_ranks_errors": (
            ranks["validation"] is not None and ranks["validation"] >= minimum
        ),
        "aligned_heldout_uncertainty_ranks_errors": (
            aligned_rank is not None and aligned_rank >= minimum
        ),
        "aligned_beats_deranged_target_control": (margin is not None and margin >= margin_minimum),
        "aligned_heldout_calibration_loss_improves_from_reset": (aligned_loss < reset_loss),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "SUPPORTS_TWO_TIMESCALE" if not failed else "DOES_NOT_SUPPORT_TWO_TIMESCALE",
        "checks": checks,
        "failed_checks": failed,
        "aligned_uncertainty_error_spearman": ranks,
        "control_heldout_uncertainty_error_spearman": control_rank,
        "control_heldout_effective_rank": control_effective_rank,
        "control_undefined_rank_semantics": (
            "zero-ranking-capability" if control_rank is None else "not-applicable"
        ),
        "aligned_control_heldout_spearman_margin": margin,
        "heldout_calibration_loss": {
            "reset": reset_loss,
            "aligned": aligned_loss,
            "relative_improvement": (reset_loss - aligned_loss) / max(abs(reset_loss), 1e-12),
        },
        "later_gates_authorized": [],
        "production_training_changes_authorized": [],
    }


def main() -> None:
    args = _parse_args()

    import torch

    from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder
    from picf_next.models.set_loss import ObjectSetCriterion
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets
    from picf_next.training.molmoact2_m2_source_coverage import (
        load_molmoact2_m2_source_coverage_recipe,
    )
    from tools import audit_molmoact2_m2_uncertainty as uncertainty
    from tools import run_molmoact2_m2_cloud as m2
    from tools import run_molmoact2_m2_source_coverage_cloud as source

    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = _load_probe_config(config_path)
    if not m2._is_under_mnt(run_dir):
        raise RuntimeError("frozen-mean variance probes must bind a persistent /mnt run")
    decision = source.validate_source_coverage_machine_decision(run_dir)
    if decision.get("status") != "FAIL" or decision.get("failed_checks") != [
        "uncertainty_ranks_errors"
    ]:
        raise RuntimeError("probe requires an otherwise-passing source-coverage M2 run")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("paired frozen-mean variance probe requires two CUDA devices")

    source_config = (_ROOT / str(config["source_coverage_config_path"])).expanduser().resolve()
    if m2._sha256(source_config) != config["source_coverage_config_sha256"]:
        raise ValueError("source-coverage config differs from the preregistered hash")
    launch = uncertainty._load_json(run_dir / "launch_manifest.json")
    training = uncertainty._load_json(run_dir / "training_report.json")
    source_evaluation = uncertainty._load_json(run_dir / "evaluation_report.json")
    if m2._sha256(source_config) != launch.get("config_file_sha256"):
        raise ValueError("probe source config differs from the M2 launch config")

    source_recipe = load_molmoact2_m2_source_coverage_recipe(source_config)
    recipe = source_recipe.load_base_m2(_ROOT)
    foundation = recipe.load_foundation(_ROOT)
    optimization = config["optimization"]
    exact_matches = {
        "batch_size": recipe.optimization.batch_size,
        "learning_rate": recipe.optimization.learning_rate,
        "gradient_clip_norm": recipe.optimization.gradient_clip_norm,
        "weight_decay": recipe.optimization.weight_decay,
        "seed": recipe.optimization.seed,
        "warmup_steps": recipe.optimization.warmup_steps,
    }
    for name, expected in exact_matches.items():
        if optimization[name] != expected:
            raise ValueError(f"probe {name} must equal the source M2 value")

    audit_revision = m2._clean_git_revision()
    source_equivalence = uncertainty._source_equivalence(
        run_source_root=uncertainty._source_root_from_launch(launch),
        current_root=_ROOT,
        sha256=m2._sha256,
    )
    sidecar_artifact_root = Path(str(launch["sidecar_artifact_root"])).expanduser().resolve()
    sidecar_materialization = m2.materialize_persistent_sidecars(sidecar_artifact_root)
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=Path(str(launch["dataset_split_root"])).expanduser().resolve(),
    )
    assets, source_sidecar = source._load_source_sidecar(
        artifact_root=sidecar_artifact_root,
        recipe=source_recipe,
        assets=assets,
    )
    cache_manifest, cache = m2._load_cache(run_dir / "feature_cache", recipe)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    checkpoint_hashes = training.get("checkpoints")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("M2 training report has no checkpoint hashes")

    train_keys = m2._keys_for_split(cache, "train")
    validation_keys = m2._keys_for_split(cache, "validation")
    heldout_keys = m2._keys_for_split(cache, "heldout")
    plan = _complete_pass_plan(
        train_keys,
        batch_size=int(optimization["batch_size"]),
        seed=int(optimization["seed"]),
        passes=int(optimization["passes"]),
    )
    total_steps = len(plan)
    if total_steps <= int(optimization["warmup_steps"]):
        raise ValueError("complete-pass plan is too short for the fixed warmup")
    expected_examples = len(train_keys) * int(optimization["passes"])
    if sum(len(batch) for batch in plan) != expected_examples:
        raise RuntimeError("complete-pass plan omitted training samples")
    shuffle = m2._derangement(train_keys, seed=int(optimization["seed"]))
    original_plan = uncertainty._load_json(run_dir / "batch_plan.json")
    if original_plan.get("label_shuffle") != shuffle:
        raise ValueError("probe derangement differs from the immutable M2 mapping")

    checkpoint = run_dir / "checkpoints/current_frame_best.pt"
    cpu = torch.device("cpu")
    aligned = uncertainty._load_model(
        foundation=foundation,
        checkpoint=checkpoint,
        expected_sha256=str(checkpoint_hashes["current_frame_best.pt"]),
        device=cpu,
        sha256=m2._sha256,
    )
    _reset_variance_head(aligned, foundation)
    control = copy.deepcopy(aligned)
    initial_non_variance = m2._state_dict_sha256(_state_without_variance(aligned))
    initial_variance = m2._state_dict_sha256(_variance_state(aligned))

    for model in (aligned, control):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.discovery.variance_head.parameters():
            parameter.requires_grad_(True)
        model.eval()
    trainable_names = sorted(
        name for name, parameter in aligned.named_parameters() if parameter.requires_grad
    )
    if trainable_names != [
        "discovery.variance_head.bias",
        "discovery.variance_head.weight",
    ]:
        raise RuntimeError("probe trainable parameter set changed")

    aligned_device = torch.device("cuda:0")
    control_device = torch.device("cuda:1")
    aligned.to(aligned_device)
    control.to(control_device)
    aligned_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(aligned_device)
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(control_device)
    layout_payload = cache_manifest["processor_layout"]
    aligned_parameters = tuple(aligned.discovery.variance_head.parameters())
    control_parameters = tuple(control.discovery.variance_head.parameters())
    aligned_optimizer = torch.optim.AdamW(
        aligned_parameters,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    control_optimizer = torch.optim.AdamW(
        control_parameters,
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )

    baseline_reset = {
        split: m2._evaluate(
            model=aligned,
            cache=cache,
            keys=keys,
            target_builder=target_builder,
            criterion=aligned_criterion,
            layout_payload=layout_payload,
            recipe=recipe,
            device=aligned_device,
        )
        for split, keys in (
            ("train", train_keys),
            ("validation", validation_keys),
            ("heldout", heldout_keys),
        )
    }

    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(aligned_device)
    torch.cuda.reset_peak_memory_stats(control_device)
    torch.cuda.synchronize(aligned_device)
    torch.cuda.synchronize(control_device)
    started = time.perf_counter()
    for step, batch_keys in enumerate(plan, start=1):
        aligned_optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        aligned_tokens, aligned_valid, aligned_records = m2._stack_batch(
            cache,
            batch_keys,
            device=aligned_device,
        )
        control_tokens, control_valid, _ = m2._stack_batch(
            cache,
            batch_keys,
            device=control_device,
        )
        shuffled_records = [dict(cache[shuffle[key]][2]) for key in batch_keys]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            aligned_output = aligned(m2._native_bank(aligned_tokens, aligned_valid))
            control_output = control(m2._native_bank(control_tokens, control_valid))
        aligned_targets = m2._build_targets(
            target_builder=target_builder,
            records=aligned_records,
            token_valid=aligned_output.projection.token_valid,
            target_dtype=aligned_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        control_targets = m2._build_targets(
            target_builder=target_builder,
            records=shuffled_records,
            token_valid=control_output.projection.token_valid,
            target_dtype=control_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        aligned_result = aligned_criterion(aligned_output.discovery, aligned_targets)
        control_result = control_criterion(control_output.discovery, control_targets)
        aligned_loss = aligned_result.losses["loss_geometry_calibration"]
        control_loss = control_result.losses["loss_geometry_calibration"]
        aligned_loss.backward()
        control_loss.backward()
        aligned_grad = torch.nn.utils.clip_grad_norm_(
            aligned_parameters,
            float(optimization["gradient_clip_norm"]),
        )
        control_grad = torch.nn.utils.clip_grad_norm_(
            control_parameters,
            float(optimization["gradient_clip_norm"]),
        )
        if not torch.isfinite(aligned_grad) or not torch.isfinite(control_grad):
            raise FloatingPointError("frozen-mean variance gradient became non-finite")
        multiplier = _learning_rate_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=int(optimization["warmup_steps"]),
            final_multiplier=float(optimization["final_learning_rate_multiplier"]),
        )
        learning_rate = float(optimization["learning_rate"]) * multiplier
        for optimizer in (aligned_optimizer, control_optimizer):
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.step()
        row: dict[str, Any] = {
            "step": step,
            "learning_rate": learning_rate,
            "aligned_loss_geometry_calibration": float(aligned_loss.detach().float().item()),
            "control_loss_geometry_calibration": float(control_loss.detach().float().item()),
            "aligned_gradient_norm": float(aligned_grad.detach().float().item()),
            "control_gradient_norm": float(control_grad.detach().float().item()),
        }
        if step % int(optimization["validation_interval"]) == 0 or step == total_steps:
            validation = m2._evaluate(
                model=aligned,
                cache=cache,
                keys=validation_keys,
                target_builder=target_builder,
                criterion=aligned_criterion,
                layout_payload=layout_payload,
                recipe=recipe,
                device=aligned_device,
            )
            row["validation_uncertainty_error_spearman"] = validation["uncertainty_error_spearman"]
            row["validation_loss_geometry_calibration"] = validation["losses"][
                "loss_geometry_calibration"
            ]
            print(
                json.dumps(
                    {
                        "event": "frozen_mean_variance_validation",
                        "step": step,
                        "total_steps": total_steps,
                        "uncertainty_error_spearman": validation["uncertainty_error_spearman"],
                        "loss_geometry_calibration": validation["losses"][
                            "loss_geometry_calibration"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        rows.append(row)
        del (
            aligned_tokens,
            aligned_valid,
            control_tokens,
            control_valid,
            aligned_output,
            control_output,
            aligned_result,
            control_result,
        )

    torch.cuda.synchronize(aligned_device)
    torch.cuda.synchronize(control_device)
    elapsed = time.perf_counter() - started
    peak_memory = {
        "cuda:0": int(torch.cuda.max_memory_allocated(aligned_device)),
        "cuda:1": int(torch.cuda.max_memory_allocated(control_device)),
    }
    final_aligned = {
        split: m2._evaluate(
            model=aligned,
            cache=cache,
            keys=keys,
            target_builder=target_builder,
            criterion=aligned_criterion,
            layout_payload=layout_payload,
            recipe=recipe,
            device=aligned_device,
        )
        for split, keys in (
            ("train", train_keys),
            ("validation", validation_keys),
            ("heldout", heldout_keys),
        )
    }
    final_control_heldout = m2._evaluate(
        model=control,
        cache=cache,
        keys=heldout_keys,
        target_builder=target_builder,
        criterion=control_criterion,
        layout_payload=layout_payload,
        recipe=recipe,
        device=control_device,
    )

    final_aligned_non_variance = m2._state_dict_sha256(_state_without_variance(aligned))
    final_control_non_variance = m2._state_dict_sha256(_state_without_variance(control))
    frozen_state_exact = (
        final_aligned_non_variance == final_control_non_variance == initial_non_variance
    )
    reference_heldout = source_evaluation["actual"]["heldout"]
    metric_drift = _representation_metric_drift(
        reference_heldout,
        final_aligned["heldout"],
    )
    maximum_metric_drift = max(metric_drift.values(), default=0.0)
    probe_decision = _decision(
        baseline_reset=baseline_reset,
        aligned=final_aligned,
        control_heldout=final_control_heldout,
        frozen_state_exact=frozen_state_exact,
        representation_metric_maximum_drift=maximum_metric_drift,
        acceptance=config["acceptance"],
    )

    output_dir = run_dir / _OUTPUT_NAME
    temporary = run_dir / f".{_OUTPUT_NAME}.tmp-{os.getpid()}"
    if output_dir.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite frozen-mean variance probe")
    temporary.mkdir()
    try:
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        m2._write_json_atomic(
            temporary / "probe_batch_plan.json",
            {
                "schema": "picf-next.molmoact2-m2-frozen-mean-variance-batch-plan.v1",
                "algorithm": "sha256-complete-pass-sort.v1",
                "seed": optimization["seed"],
                "passes": optimization["passes"],
                "batches": plan,
                "batches_sha256": m2._canonical_sha256(plan),
                "label_shuffle": shuffle,
                "label_shuffle_sha256": m2._canonical_sha256(shuffle),
            },
        )
        aligned_head_path = checkpoint_dir / "aligned_variance_head_final.pt"
        control_head_path = checkpoint_dir / "deranged_variance_head_final.pt"
        m2._write_torch_atomic(
            aligned_head_path,
            {"variance_head": m2._state_dict_cpu(aligned.discovery.variance_head)},
        )
        m2._write_torch_atomic(
            control_head_path,
            {"variance_head": m2._state_dict_cpu(control.discovery.variance_head)},
        )
        m2._write_json_atomic(
            temporary / "metrics.json",
            {
                "schema": "picf-next.molmoact2-m2-frozen-mean-variance-metrics.v1",
                "baseline_reset": baseline_reset,
                "aligned": final_aligned,
                "deranged_control_heldout": final_control_heldout,
                "optimization_curve": rows,
            },
        )
        report = {
            "schema": _SCHEMA,
            "status": "DIAGNOSTIC_ONLY",
            "run_dir": str(run_dir),
            "run_code_revision": launch["code_revision"],
            "audit_code_revision": audit_revision,
            "source_equivalence": source_equivalence,
            "sidecar_materialization": sidecar_materialization,
            "source_sidecar": source_sidecar,
            "protocol": config["protocol"],
            "optimization": {
                **optimization,
                "steps": total_steps,
                "training_examples": expected_examples,
                "unique_training_examples_per_pass": len(train_keys),
                "elapsed_s": elapsed,
                "seconds_per_joint_aligned_and_control_step": elapsed / total_steps,
                "cuda_peak_allocated_bytes": peak_memory,
                "trainable_parameter_names": trainable_names,
            },
            "state_isolation": {
                "initial_non_variance_state_sha256": initial_non_variance,
                "final_aligned_non_variance_state_sha256": final_aligned_non_variance,
                "final_control_non_variance_state_sha256": final_control_non_variance,
                "initial_variance_state_sha256": initial_variance,
                "final_aligned_variance_state_sha256": m2._state_dict_sha256(
                    _variance_state(aligned)
                ),
                "final_control_variance_state_sha256": m2._state_dict_sha256(
                    _variance_state(control)
                ),
                "frozen_non_variance_state_exact": frozen_state_exact,
            },
            "representation_metric_absolute_drift": metric_drift,
            "representation_metric_maximum_absolute_drift": maximum_metric_drift,
            "decision": probe_decision,
            "input_sha256": {
                "probe_config": m2._sha256(config_path),
                "source_coverage_config": m2._sha256(source_config),
                "machine_decision.json": m2._sha256(run_dir / "machine_decision.json"),
                "evaluation_report.json": m2._sha256(run_dir / "evaluation_report.json"),
                "training_report.json": m2._sha256(run_dir / "training_report.json"),
                "batch_plan.json": m2._sha256(run_dir / "batch_plan.json"),
                "feature_cache/manifest.json": m2._sha256(run_dir / "feature_cache/manifest.json"),
                "checkpoints/current_frame_best.pt": checkpoint_hashes["current_frame_best.pt"],
            },
            "output_sha256": {
                "metrics.json": m2._sha256(temporary / "metrics.json"),
                "probe_batch_plan.json": m2._sha256(temporary / "probe_batch_plan.json"),
                "checkpoints/aligned_variance_head_final.pt": m2._sha256(aligned_head_path),
                "checkpoints/deranged_variance_head_final.pt": m2._sha256(control_head_path),
            },
            "later_gates_authorized": [],
            "production_training_changes_authorized": [],
        }
        m2._write_json_atomic(temporary / "report.json", report)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
