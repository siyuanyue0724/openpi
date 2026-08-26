#!/usr/bin/env python3
"""Test frame-conditional M2 variance with fixed residuals and a strict control."""

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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

_SCHEMA = "picf-next.molmoact2-m2-residual-permutation-probe.v1"
_CONFIG_SCHEMA = "picf-next.molmoact2-m2-residual-permutation-probe-config.v1"
_OUTPUT_NAME = "residual_permutation_probe"
_SOURCE_PROBE_NAME = "frozen_mean_variance_probe"
_VARIANCE_PREFIX = "discovery.variance_head."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(_ROOT / "configs/training/molmoact2_calvin_m2_residual_permutation_probe.json"),
    )
    return parser.parse_args()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="ascii") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != _CONFIG_SCHEMA:
        raise ValueError("residual-permutation probe config schema changed")
    if payload.get("gate") != (
        "M2_frozen_mean_identity_stratified_residual_permutation_diagnostic"
    ):
        raise ValueError("residual-permutation probe gate changed")
    expected_protocol = {
        "checkpoint_selection": "fixed-final-step-no-reselection",
        "control": "identity-stratified-fixed-match-residual-derangement",
        "initialization": "reset-both-variance-heads-to-declared-initial-variance",
        "mean_matching_and_representation": "frozen-exactly",
        "residual_target": "final-mean-squared-residual-after-fixed-hungarian-match",
        "training_data_exposure": "one-complete-deterministic-pass",
        "updated_parameter_prefixes": [_VARIANCE_PREFIX],
    }
    if payload.get("protocol") != expected_protocol:
        raise ValueError("residual-permutation protocol changed")
    optimization = payload.get("optimization")
    acceptance = payload.get("acceptance")
    if not isinstance(optimization, dict) or not isinstance(acceptance, dict):
        raise ValueError("probe config requires optimization and acceptance objects")
    for name in ("batch_size", "passes", "validation_interval", "warmup_steps"):
        value = optimization.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
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
        raise ValueError("final learning-rate multiplier must lie in (0, 1]")
    bounded = (
        "maximum_absolute_control_target_within_identity_rank_correlation",
        "minimum_aligned_control_within_identity_axis_rank_margin",
        "minimum_global_uncertainty_error_spearman",
        "minimum_within_identity_axis_centered_rank_correlation",
    )
    for name in bounded:
        value = _finite_float(acceptance.get(name), name)
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [-1, 1]")
    for name in (
        "require_aligned_nll_below_control_on_validation_and_heldout",
        "require_aligned_nll_below_reset_on_heldout",
    ):
        if not isinstance(acceptance.get(name), bool) or not acceptance[name]:
            raise ValueError(f"{name} must remain true")
    report_hash = payload.get("frozen_mean_probe_report_sha256")
    if not isinstance(report_hash, str) or len(report_hash) != 64:
        raise ValueError("frozen-mean probe report hash is invalid")
    return payload


def _object_key(sample_key: str, identity_key: str) -> str:
    if "\x1f" in sample_key or "\x1f" in identity_key:
        raise ValueError("sample and identity keys cannot contain the object-key separator")
    return f"{sample_key}\x1f{identity_key}"


def _identity_stratified_derangement(
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, str]:
    """Derange complete residual vectors within each physical identity."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for example in examples:
        key = str(example["object_key"])
        identity = str(example["identity_key"])
        if key in seen:
            raise ValueError("residual examples contain a duplicate object key")
        seen.add(key)
        grouped[identity].append(example)
    if not grouped:
        raise ValueError("residual derangement requires at least one identity")

    mapping: dict[str, str] = {}
    for identity, rows in sorted(grouped.items()):
        if len(rows) < 2:
            raise ValueError(f"identity has fewer than two residual observations: {identity}")
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{identity}:{row['sample_key']}".encode()
            ).digest(),
        )
        rotated = ordered[1:] + ordered[:1]
        for source, target in zip(ordered, rotated, strict=True):
            source_key = str(source["object_key"])
            target_key = str(target["object_key"])
            if source_key == target_key or source["sample_key"] == target["sample_key"]:
                raise RuntimeError("identity-stratified residual mapping contains a fixed point")
            if source["identity_key"] != target["identity_key"]:
                raise RuntimeError("identity-stratified residual mapping changed identity")
            mapping[source_key] = target_key
    if set(mapping) != seen or set(mapping.values()) != seen:
        raise RuntimeError("identity-stratified residual mapping is not a permutation")
    return mapping


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal length")
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    result = spearmanr(left, right)
    value = float(result.statistic if hasattr(result, "statistic") else result[0])
    return value if math.isfinite(value) else None


def _within_group_centered_rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
    groups: Sequence[str],
) -> float | None:
    if len(left) != len(right) or len(left) != len(groups):
        raise ValueError("centered-rank inputs must have equal length")
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped_indices[group].append(index)
    centered_left: list[float] = []
    centered_right: list[float] = []
    for group in sorted(grouped_indices):
        indices = grouped_indices[group]
        group_left = [left[index] for index in indices]
        group_right = [right[index] for index in indices]
        if len(group_left) < 2 or len(set(group_left)) < 2 or len(set(group_right)) < 2:
            continue
        left_rank = rankdata(group_left, method="average")
        right_rank = rankdata(group_right, method="average")
        centered_left.extend((left_rank - left_rank.mean()).astype(float).tolist())
        centered_right.extend((right_rank - right_rank.mean()).astype(float).tolist())
    if len(centered_left) < 2 or not np.std(centered_left) or not np.std(centered_right):
        return None
    value = float(np.corrcoef(centered_left, centered_right)[0, 1])
    return value if math.isfinite(value) else None


def _control_target_alignment(
    examples: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    by_key = {str(example["object_key"]): example for example in examples}
    aligned: list[float] = []
    deranged: list[float] = []
    identities: list[str] = []
    for example in examples:
        target = by_key[mapping[str(example["object_key"])]]
        supervised = np.asarray(example["supervised"]).astype(bool)
        target_supervised = np.asarray(target["supervised"]).astype(bool)
        if not np.array_equal(supervised, target_supervised):
            raise ValueError("residual control requires identical coordinate supervision masks")
        aligned.append(float(np.asarray(example["squared_residual"])[supervised].mean()))
        deranged.append(float(np.asarray(target["squared_residual"])[supervised].mean()))
        identities.append(str(example["identity_key"]))
    return {
        "observation_count": len(aligned),
        "identity_count": len(set(identities)),
        "global_rank_correlation": _safe_spearman(aligned, deranged),
        "within_identity_centered_rank_correlation": (
            _within_group_centered_rank_correlation(aligned, deranged, identities)
        ),
        "fixed_point_count": sum(key == value for key, value in mapping.items()),
    }


def _calibration_loss(
    head: Any,
    examples: Sequence[Mapping[str, Any]],
    target_examples: Sequence[Mapping[str, Any]],
    *,
    minimum_variance: float,
    device: Any,
) -> Any:
    import torch
    from torch.nn import functional as F

    if not examples or len(examples) != len(target_examples):
        raise ValueError("calibration loss requires paired nonempty examples")
    features = torch.stack([row["query_feature"] for row in examples]).to(device)
    squared = torch.stack([row["squared_residual"] for row in target_examples]).to(device)
    measurement = torch.stack([row["measurement_variance"] for row in target_examples]).to(device)
    supervised = torch.stack([row["supervised"] for row in target_examples]).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw = head(features)
    variance = (F.softplus(raw.float()) + minimum_variance).to(features.dtype)
    combined = variance.float() + measurement.float()
    calibration = 0.5 * (squared.float() / combined + combined.log())
    counts = supervised.sum(dim=-1)
    if not bool((counts > 0).all()):
        raise ValueError("every residual example must supervise at least one coordinate")
    return ((calibration * supervised).sum(dim=-1) / counts).mean()


def _source_batch_calibration(
    head: Any,
    examples: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    batch_size: int,
    minimum_variance: float,
    device: Any,
) -> float:
    """Reproduce the source evaluator's equal weighting of batch means."""

    import torch

    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        by_sample[str(example["sample_key"])].append(example)
    if set(by_sample) != set(keys):
        raise ValueError("source calibration reproduction requires objects in every source frame")
    batch_losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            selected = [row for key in keys[start : start + batch_size] for row in by_sample[key]]
            loss = _calibration_loss(
                head,
                selected,
                selected,
                minimum_variance=minimum_variance,
                device=device,
            )
            batch_losses.append(float(loss.item()))
    if not batch_losses:
        raise ValueError("source calibration reproduction requires at least one batch")
    return sum(batch_losses) / len(batch_losses)


def _extract_compact_metrics(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    compact = {}
    for split, metrics in summary["splits"].items():
        object_metrics = metrics["gate_object_normalized"]
        identity_axis = metrics["decomposition"]["identity_axis"]
        compact[split] = {
            "global_uncertainty_error_spearman": object_metrics["variance_error_spearman"],
            "within_identity_axis_centered_rank_correlation": identity_axis[
                "within_group_centered_rank_correlation"
            ],
            "gaussian_nll_without_constant": object_metrics["gaussian_nll_without_constant"],
            "aggregate_error_to_variance_ratio": object_metrics[
                "aggregate_error_to_variance_ratio"
            ],
            "row_count": object_metrics["row_count"],
        }
    return compact


def _decision(
    *,
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    control_target_alignment: Mapping[str, Any],
    frozen_non_variance_state_exact: bool,
    reset_metric_reproduction_exact: bool,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    global_minimum = _finite_float(
        acceptance["minimum_global_uncertainty_error_spearman"],
        "minimum_global_uncertainty_error_spearman",
    )
    within_minimum = _finite_float(
        acceptance["minimum_within_identity_axis_centered_rank_correlation"],
        "minimum_within_identity_axis_centered_rank_correlation",
    )
    margin_minimum = _finite_float(
        acceptance["minimum_aligned_control_within_identity_axis_rank_margin"],
        "minimum_aligned_control_within_identity_axis_rank_margin",
    )
    control_maximum = _finite_float(
        acceptance["maximum_absolute_control_target_within_identity_rank_correlation"],
        "maximum_absolute_control_target_within_identity_rank_correlation",
    )

    checks = {
        "frozen_non_variance_state_exact": frozen_non_variance_state_exact,
        "reset_metric_reproduction_exact": reset_metric_reproduction_exact,
    }
    control_alignment = control_target_alignment["within_identity_centered_rank_correlation"]
    checks["control_targets_decorrelated_within_identity"] = (
        control_alignment is not None and abs(float(control_alignment)) <= control_maximum
    )
    for split in ("train", "validation", "heldout"):
        aligned_global = metrics["aligned"][split]["global_uncertainty_error_spearman"]
        checks[f"aligned_{split}_global_rank_nonnegative"] = (
            aligned_global is not None and float(aligned_global) >= global_minimum
        )
    for split in ("validation", "heldout"):
        aligned_within = metrics["aligned"][split]["within_identity_axis_centered_rank_correlation"]
        control_within = metrics["control"][split]["within_identity_axis_centered_rank_correlation"]
        margin = (
            None
            if aligned_within is None or control_within is None
            else float(aligned_within) - float(control_within)
        )
        checks[f"aligned_{split}_within_identity_axis_rank"] = (
            aligned_within is not None and float(aligned_within) >= within_minimum
        )
        checks[f"aligned_beats_control_{split}_within_identity_axis_rank"] = (
            margin is not None and margin >= margin_minimum
        )
        checks[f"aligned_{split}_nll_below_control"] = float(
            metrics["aligned"][split]["gaussian_nll_without_constant"]
        ) < float(metrics["control"][split]["gaussian_nll_without_constant"])
    checks["aligned_heldout_nll_below_reset"] = float(
        metrics["aligned"]["heldout"]["gaussian_nll_without_constant"]
    ) < float(metrics["reset"]["heldout"]["gaussian_nll_without_constant"])
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": (
            "SUPPORTS_CONDITIONAL_RESIDUAL_CALIBRATION"
            if not failed
            else "DOES_NOT_SUPPORT_CONDITIONAL_RESIDUAL_CALIBRATION"
        ),
        "checks": checks,
        "failed_checks": failed,
        "compact_metrics": metrics,
        "control_target_alignment": dict(control_target_alignment),
        "later_gates_authorized": [],
        "production_training_changes_authorized": [],
    }


def _collect_residual_examples(
    *,
    model: Any,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
    target_builder: Any,
    criterion: Any,
    layout_payload: Sequence[Mapping[str, Any]],
    recipe: Any,
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    from tools import run_molmoact2_m2_cloud as m2

    examples: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(keys), recipe.optimization.batch_size):
        batch_keys = keys[start : start + recipe.optimization.batch_size]
        tokens, valid, records = m2._stack_batch(cache, batch_keys, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(m2._native_bank(tokens, valid))
        targets = m2._build_targets(
            target_builder=target_builder,
            records=records,
            token_valid=output.projection.token_valid,
            target_dtype=output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        result = criterion(output.discovery, targets)
        for batch_index, (target, match, sample_key, record) in enumerate(
            zip(targets, result.matches, batch_keys, records, strict=True)
        ):
            identities = target.temporal_identity_keys
            if identities is None or len(identities) != target.num_objects:
                raise RuntimeError("residual probe requires explicit physical identities")
            if target.geometry is None or target.geometry_supervised is None:
                raise RuntimeError("residual probe requires selective geometry targets")
            target_variance = target.geometry_variance
            if target_variance is None:
                target_variance = torch.zeros_like(target.geometry)
            for query_index, target_index in zip(
                match.prediction_indices.tolist(),
                match.target_indices.tolist(),
                strict=True,
            ):
                supervised = target.geometry_supervised[target_index]
                if not bool(supervised.any()):
                    continue
                identity = str(identities[target_index])
                residual = (
                    output.discovery.geometry_mean[batch_index, query_index].float()
                    - target.geometry[target_index].float()
                )
                examples.append(
                    {
                        "object_key": _object_key(sample_key, identity),
                        "sample_key": sample_key,
                        "split": str(record["split"]),
                        "global_index": int(record["global_index"]),
                        "identity_key": identity,
                        "query_index": int(query_index),
                        "query_feature": output.discovery.query_features[batch_index, query_index]
                        .detach()
                        .to(device="cpu", dtype=torch.bfloat16)
                        .clone(),
                        "squared_residual": residual.square()
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .clone(),
                        "measurement_variance": target_variance[target_index]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .clone(),
                        "supervised": supervised.detach().to(device="cpu").clone(),
                    }
                )
        del tokens, valid, output, result
    object_keys = [str(example["object_key"]) for example in examples]
    if len(object_keys) != len(set(object_keys)):
        raise RuntimeError("fixed matching emitted duplicate sample/identity observations")
    return examples


def _evaluate_head(
    head: Any,
    examples: Sequence[Mapping[str, Any]],
    *,
    minimum_variance: float,
    geometry_contract: Any,
    device: Any,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from tools.audit_molmoact2_m2_uncertainty import summarize_uncertainty

    object_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    head.eval()
    for start in range(0, len(examples), 4096):
        selected_examples = examples[start : start + 4096]
        features = torch.stack([row["query_feature"] for row in selected_examples]).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            raw = head(features)
        predicted = (F.softplus(raw.float()) + minimum_variance).to(features.dtype).float().cpu()
        for example, predicted_variance in zip(selected_examples, predicted, strict=True):
            supervised = example["supervised"].numpy().astype(bool)
            squared = example["squared_residual"].numpy()
            measurement = example["measurement_variance"].numpy()
            combined = predicted_variance.numpy() + measurement
            common = {
                "sample_key": str(example["sample_key"]),
                "split": str(example["split"]),
                "global_index": int(example["global_index"]),
                "identity_key": str(example["identity_key"]),
                "query_index": int(example["query_index"]),
            }
            object_rows.append(
                {
                    **common,
                    "predicted_variance_normalized_mean": float(combined[supervised].mean()),
                    "squared_error_normalized_mean": float(squared[supervised].mean()),
                }
            )
            for axis_index in np.flatnonzero(supervised).tolist():
                scale = float(geometry_contract.normalization_scale[axis_index])
                coordinate_rows.append(
                    {
                        **common,
                        "axis": str(geometry_contract.axes[axis_index]),
                        "predicted_variance_normalized": float(combined[axis_index]),
                        "squared_error_normalized": float(squared[axis_index]),
                        "predicted_variance_physical": float(combined[axis_index] * scale * scale),
                        "squared_error_physical": float(squared[axis_index] * scale * scale),
                    }
                )
    return summarize_uncertainty(object_rows, coordinate_rows)


def _state_dict_cpu(module: Any) -> dict[str, Any]:
    return {
        name: value.detach().to(device="cpu").clone() for name, value in module.state_dict().items()
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
    from tools import run_molmoact2_m2_frozen_mean_variance_probe as frozen
    from tools import run_molmoact2_m2_source_coverage_cloud as source

    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    if not m2._is_under_mnt(run_dir):
        raise RuntimeError("residual-permutation probes must bind a persistent /mnt run")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("paired residual-permutation probe requires two CUDA devices")
    source_probe_dir = run_dir / _SOURCE_PROBE_NAME
    source_probe_report_path = source_probe_dir / "report.json"
    if m2._sha256(source_probe_report_path) != config["frozen_mean_probe_report_sha256"]:
        raise ValueError("frozen-mean source probe report is absent or changed")
    source_probe = uncertainty._load_json(source_probe_report_path)
    if source_probe.get("decision", {}).get("status") != "DOES_NOT_SUPPORT_TWO_TIMESCALE":
        raise RuntimeError("residual probe requires the failed frozen-mean source probe")
    if source_probe["decision"].get("failed_checks") != ["aligned_beats_deranged_target_control"]:
        raise RuntimeError("frozen-mean source probe failed for an unexpected reason")

    source_config = (_ROOT / str(config["source_coverage_config_path"])).resolve()
    if m2._sha256(source_config) != config["source_coverage_config_sha256"]:
        raise ValueError("source-coverage config differs from the preregistered hash")
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
            raise ValueError(f"residual probe {name} must equal source M2")

    launch = uncertainty._load_json(run_dir / "launch_manifest.json")
    training = uncertainty._load_json(run_dir / "training_report.json")
    source_equivalence = uncertainty._source_equivalence(
        run_source_root=uncertainty._source_root_from_launch(launch),
        current_root=_ROOT,
        sha256=m2._sha256,
    )
    audit_revision = m2._clean_git_revision()
    sidecar_artifact_root = Path(str(launch["sidecar_artifact_root"])).resolve()
    sidecar_materialization = m2.materialize_persistent_sidecars(sidecar_artifact_root)
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=Path(str(launch["dataset_split_root"])).resolve(),
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

    cpu = torch.device("cpu")
    model = uncertainty._load_model(
        foundation=foundation,
        checkpoint=run_dir / "checkpoints/current_frame_best.pt",
        expected_sha256=str(checkpoint_hashes["current_frame_best.pt"]),
        device=cpu,
        sha256=m2._sha256,
    )
    frozen._reset_variance_head(model, foundation)
    initial_non_variance = m2._state_dict_sha256(frozen._state_without_variance(model))
    reset_head = copy.deepcopy(model.discovery.variance_head).cpu()
    extraction_device = torch.device("cuda:0")
    model.to(extraction_device).eval()
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(extraction_device)
    keys_by_split = {
        split: m2._keys_for_split(cache, split) for split in ("train", "validation", "heldout")
    }
    examples_by_split = {
        split: _collect_residual_examples(
            model=model,
            cache=cache,
            keys=keys,
            target_builder=target_builder,
            criterion=criterion,
            layout_payload=cache_manifest["processor_layout"],
            recipe=recipe,
            device=extraction_device,
        )
        for split, keys in keys_by_split.items()
    }
    final_non_variance = m2._state_dict_sha256(frozen._state_without_variance(model))
    frozen_non_variance_state_exact = initial_non_variance == final_non_variance
    del criterion
    model.to(cpu)
    torch.cuda.empty_cache()

    train_examples = examples_by_split["train"]
    train_by_key = {str(example["object_key"]): example for example in train_examples}
    residual_mapping = _identity_stratified_derangement(
        train_examples,
        seed=int(optimization["seed"]),
    )
    control_target_alignment = _control_target_alignment(train_examples, residual_mapping)
    maximum_control_alignment = float(
        config["acceptance"]["maximum_absolute_control_target_within_identity_rank_correlation"]
    )
    observed_control_alignment = control_target_alignment[
        "within_identity_centered_rank_correlation"
    ]
    if observed_control_alignment is None or (
        abs(float(observed_control_alignment)) > maximum_control_alignment
    ):
        raise RuntimeError("identity-stratified control residuals remain correlated")

    plan = frozen._complete_pass_plan(
        keys_by_split["train"],
        batch_size=int(optimization["batch_size"]),
        seed=int(optimization["seed"]),
        passes=int(optimization["passes"]),
    )
    total_steps = len(plan)
    if total_steps <= int(optimization["warmup_steps"]):
        raise ValueError("residual probe plan is too short for warmup")
    examples_per_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in train_examples:
        examples_per_sample[str(example["sample_key"])].append(example)

    aligned_device = torch.device("cuda:0")
    control_device = torch.device("cuda:1")
    aligned_head = copy.deepcopy(reset_head).to(aligned_device)
    control_head = copy.deepcopy(reset_head).to(control_device)
    aligned_optimizer = torch.optim.AdamW(
        aligned_head.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    control_optimizer = torch.optim.AdamW(
        control_head.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(aligned_device)
    torch.cuda.reset_peak_memory_stats(control_device)
    torch.cuda.synchronize(aligned_device)
    torch.cuda.synchronize(control_device)
    started = time.perf_counter()
    for step, batch_keys in enumerate(plan, start=1):
        aligned_optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        batch_examples = [example for key in batch_keys for example in examples_per_sample[key]]
        control_targets = [
            train_by_key[residual_mapping[str(example["object_key"])]] for example in batch_examples
        ]
        aligned_loss = _calibration_loss(
            aligned_head,
            batch_examples,
            batch_examples,
            minimum_variance=foundation.core_config.discovery.minimum_variance,
            device=aligned_device,
        )
        control_loss = _calibration_loss(
            control_head,
            batch_examples,
            control_targets,
            minimum_variance=foundation.core_config.discovery.minimum_variance,
            device=control_device,
        )
        aligned_loss.backward()
        control_loss.backward()
        aligned_grad = torch.nn.utils.clip_grad_norm_(
            aligned_head.parameters(), float(optimization["gradient_clip_norm"])
        )
        control_grad = torch.nn.utils.clip_grad_norm_(
            control_head.parameters(), float(optimization["gradient_clip_norm"])
        )
        if not torch.isfinite(aligned_grad) or not torch.isfinite(control_grad):
            raise FloatingPointError("residual-permutation gradient became non-finite")
        multiplier = frozen._learning_rate_multiplier(
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
            "aligned_loss_geometry_calibration": float(aligned_loss.detach().item()),
            "control_loss_geometry_calibration": float(control_loss.detach().item()),
            "aligned_gradient_norm": float(aligned_grad.detach().item()),
            "control_gradient_norm": float(control_grad.detach().item()),
        }
        if step % int(optimization["validation_interval"]) == 0 or step == total_steps:
            validation_summary = _evaluate_head(
                aligned_head,
                examples_by_split["validation"],
                minimum_variance=foundation.core_config.discovery.minimum_variance,
                geometry_contract=foundation.geometry_contract,
                device=aligned_device,
            )
            validation_metrics = _extract_compact_metrics(validation_summary)["validation"]
            row["validation_global_uncertainty_error_spearman"] = validation_metrics[
                "global_uncertainty_error_spearman"
            ]
            row["validation_within_identity_axis_centered_rank_correlation"] = validation_metrics[
                "within_identity_axis_centered_rank_correlation"
            ]
            print(
                json.dumps(
                    {
                        "event": "residual_permutation_validation",
                        "step": step,
                        "total_steps": total_steps,
                        **validation_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        rows.append(row)
    torch.cuda.synchronize(aligned_device)
    torch.cuda.synchronize(control_device)
    elapsed = time.perf_counter() - started

    summaries = {
        "reset": _extract_compact_metrics(
            _evaluate_head(
                reset_head.to(aligned_device),
                [example for split in examples_by_split.values() for example in split],
                minimum_variance=foundation.core_config.discovery.minimum_variance,
                geometry_contract=foundation.geometry_contract,
                device=aligned_device,
            )
        ),
        "aligned": _extract_compact_metrics(
            _evaluate_head(
                aligned_head,
                [example for split in examples_by_split.values() for example in split],
                minimum_variance=foundation.core_config.discovery.minimum_variance,
                geometry_contract=foundation.geometry_contract,
                device=aligned_device,
            )
        ),
        "control": _extract_compact_metrics(
            _evaluate_head(
                control_head,
                [example for split in examples_by_split.values() for example in split],
                minimum_variance=foundation.core_config.discovery.minimum_variance,
                geometry_contract=foundation.geometry_contract,
                device=control_device,
            )
        ),
    }
    source_reset = source_probe["decision"]
    source_probe_metrics = uncertainty._load_json(source_probe_dir / "metrics.json")
    source_semantics_reset_calibration = {
        split: _source_batch_calibration(
            reset_head.to(aligned_device),
            examples_by_split[split],
            keys_by_split[split],
            batch_size=recipe.optimization.batch_size,
            minimum_variance=foundation.core_config.discovery.minimum_variance,
            device=aligned_device,
        )
        for split in ("train", "validation", "heldout")
    }
    reset_reproduction_error = max(
        abs(
            float(source_semantics_reset_calibration[split])
            - float(
                source_probe_metrics["baseline_reset"][split]["losses"]["loss_geometry_calibration"]
            )
        )
        for split in ("train", "validation", "heldout")
    )
    reset_metric_reproduction_exact = reset_reproduction_error <= 1e-6
    decision = _decision(
        metrics=summaries,
        control_target_alignment=control_target_alignment,
        frozen_non_variance_state_exact=frozen_non_variance_state_exact,
        reset_metric_reproduction_exact=reset_metric_reproduction_exact,
        acceptance=config["acceptance"],
    )

    output_dir = run_dir / _OUTPUT_NAME
    temporary = run_dir / f".{_OUTPUT_NAME}.tmp-{os.getpid()}"
    if output_dir.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite residual-permutation probe")
    temporary.mkdir()
    try:
        checkpoint_dir = temporary / "checkpoints"
        checkpoint_dir.mkdir()
        aligned_path = checkpoint_dir / "aligned_variance_head_final.pt"
        control_path = checkpoint_dir / "identity_deranged_variance_head_final.pt"
        m2._write_torch_atomic(aligned_path, {"variance_head": _state_dict_cpu(aligned_head)})
        m2._write_torch_atomic(control_path, {"variance_head": _state_dict_cpu(control_head)})
        mapping_payload = [
            {"source_object_key": key, "target_object_key": residual_mapping[key]}
            for key in sorted(residual_mapping)
        ]
        m2._write_json_atomic(
            temporary / "residual_derangement.json",
            {
                "schema": "picf-next.identity-stratified-residual-derangement.v1",
                "seed": int(optimization["seed"]),
                "mapping": mapping_payload,
                "mapping_sha256": m2._canonical_sha256(mapping_payload),
                "alignment": control_target_alignment,
            },
        )
        m2._write_json_atomic(
            temporary / "metrics.json",
            {
                "schema": "picf-next.molmoact2-m2-residual-permutation-metrics.v1",
                "summaries": summaries,
                "source_semantics_reset_calibration": source_semantics_reset_calibration,
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
                "training_frames": len(keys_by_split["train"]),
                "training_object_observations": len(train_examples),
                "elapsed_s": elapsed,
                "seconds_per_joint_aligned_and_control_step": elapsed / total_steps,
                "cuda_peak_allocated_bytes": {
                    "cuda:0": int(torch.cuda.max_memory_allocated(aligned_device)),
                    "cuda:1": int(torch.cuda.max_memory_allocated(control_device)),
                },
                "trainable_parameter_names": [
                    "discovery.variance_head.bias",
                    "discovery.variance_head.weight",
                ],
            },
            "state_isolation": {
                "initial_non_variance_state_sha256": initial_non_variance,
                "final_non_variance_state_sha256": final_non_variance,
                "frozen_non_variance_state_exact": frozen_non_variance_state_exact,
                "reset_metric_maximum_absolute_reproduction_error": (reset_reproduction_error),
                "reset_metric_reproduction_exact_at_1e_6": (reset_metric_reproduction_exact),
                "source_semantics_reset_calibration": source_semantics_reset_calibration,
            },
            "decision": decision,
            "source_failed_probe_decision": source_reset,
            "input_sha256": {
                "probe_config": m2._sha256(config_path),
                "source_coverage_config": m2._sha256(source_config),
                "frozen_mean_variance_probe/report.json": m2._sha256(source_probe_report_path),
                "frozen_mean_variance_probe/metrics.json": m2._sha256(
                    source_probe_dir / "metrics.json"
                ),
                "feature_cache/manifest.json": m2._sha256(run_dir / "feature_cache/manifest.json"),
                "checkpoints/current_frame_best.pt": checkpoint_hashes["current_frame_best.pt"],
            },
            "output_sha256": {
                "metrics.json": m2._sha256(temporary / "metrics.json"),
                "residual_derangement.json": m2._sha256(temporary / "residual_derangement.json"),
                "checkpoints/aligned_variance_head_final.pt": m2._sha256(aligned_path),
                "checkpoints/identity_deranged_variance_head_final.pt": m2._sha256(control_path),
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
