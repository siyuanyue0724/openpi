#!/usr/bin/env python3
"""Calibrate the accepted axis-constant M2 observation covariance."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

_SCHEMA = "picf-next.molmoact2-m2-axis-variance-calibration.v1"
_CONFIG_SCHEMA = "picf-next.molmoact2-m2-axis-variance-calibration-config.v1"
_OUTPUT_NAME = "axis_constant_observation_covariance"
_NEGATIVE_RESIDUAL_DECISION = "DOES_NOT_SUPPORT_CONDITIONAL_RESIDUAL_CALIBRATION"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_axis_variance_calibration.json",
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
        raise ValueError("axis-variance calibration config schema changed")
    if payload.get("gate") != "M2_axis_constant_observation_covariance_calibration":
        raise ValueError("axis-variance calibration gate changed")
    expected_protocol = {
        "checkpoint_selection": "fixed-current-frame-best-no-reselection",
        "fit_data": "train-fixed-match-residuals-only",
        "fit_objective": ("axiswise-diagonal-gaussian-nll-with-declared-target-variance"),
        "fitted_parameter_names": ["discovery.variance_head.bias"],
        "frozen_parameter_names": ["discovery.variance_head.weight"],
        "matching_mean_and_representation": "frozen-exactly",
        "validation_and_heldout": "evaluation-only-never-fit",
        "variance_dependency": "axis-only-task-identity-and-query-independent",
    }
    if payload.get("protocol") != expected_protocol:
        raise ValueError("axis-variance calibration protocol changed")
    artifacts = payload.get("source_artifact_sha256")
    expected_artifacts = {
        "checkpoints/current_frame_best.pt",
        "evaluation_report.json",
        "feature_cache/manifest.json",
        "launch_manifest.json",
        "training_report.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("axis-variance source artifact set changed")
    hash_fields = [
        *artifacts.values(),
        payload.get("residual_permutation_report_sha256"),
        payload.get("source_coverage_config_sha256"),
    ]
    if any(not isinstance(value, str) or len(value) != 64 for value in hash_fields):
        raise ValueError("axis-variance calibration requires exact SHA-256 bindings")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("axis-variance calibration requires acceptance thresholds")
    minimum_ratio = _finite_float(
        acceptance.get("minimum_heldout_error_to_variance_ratio"),
        "minimum_heldout_error_to_variance_ratio",
    )
    maximum_ratio = _finite_float(
        acceptance.get("maximum_heldout_error_to_variance_ratio"),
        "maximum_heldout_error_to_variance_ratio",
    )
    if not 0.0 < minimum_ratio < maximum_ratio:
        raise ValueError("heldout error-to-variance bounds are invalid")
    roundtrip = _finite_float(
        acceptance.get("maximum_softplus_roundtrip_absolute_error"),
        "maximum_softplus_roundtrip_absolute_error",
    )
    if roundtrip < 0.0:
        raise ValueError("softplus roundtrip tolerance must be nonnegative")
    for name in (
        "require_fitted_nll_below_reset_on_validation",
        "require_fitted_nll_below_reset_on_heldout",
    ):
        if acceptance.get(name) is not True:
            raise ValueError(f"{name} must remain true")
    return payload


def _examples_to_arrays(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("axis-variance calibration requires residual examples")
    squared = np.stack([np.asarray(row["squared_residual"], dtype=np.float64) for row in examples])
    target_variance = np.stack(
        [np.asarray(row["measurement_variance"], dtype=np.float64) for row in examples]
    )
    supervised = np.stack([np.asarray(row["supervised"], dtype=np.bool_) for row in examples])
    if (
        squared.ndim != 2
        or target_variance.shape != squared.shape
        or supervised.shape != squared.shape
        or not np.isfinite(squared).all()
        or not np.isfinite(target_variance).all()
        or (squared < 0.0).any()
        or (target_variance < 0.0).any()
    ):
        raise ValueError("residual examples contain invalid calibration arrays")
    return np.sqrt(squared), target_variance, supervised


def _calibration_metrics(
    examples: Sequence[Mapping[str, Any]],
    observation_variance: Sequence[float],
) -> dict[str, Any]:
    from picf_next.models.observation_calibration import gaussian_axis_nll_without_constant

    residual, target_variance, supervised = _examples_to_arrays(examples)
    observation = np.asarray(observation_variance, dtype=np.float64)
    nll, axis_nll = gaussian_axis_nll_without_constant(
        residual,
        target_variance,
        observation,
        supervised=supervised,
    )
    combined = target_variance + observation[None, :]
    squared = np.square(residual)
    standardized = squared[supervised] / combined[supervised]
    axis_ratio = []
    for axis in range(residual.shape[1]):
        selected = supervised[:, axis]
        axis_ratio.append(float(squared[selected, axis].sum() / combined[selected, axis].sum()))
    return {
        "row_count": int(residual.shape[0]),
        "coordinate_count": int(supervised.sum()),
        "gaussian_nll_without_constant": nll,
        "axis_gaussian_nll_without_constant": list(axis_nll),
        "aggregate_error_to_variance_ratio": float(
            squared[supervised].sum() / combined[supervised].sum()
        ),
        "axis_error_to_variance_ratio": axis_ratio,
        "standardized_squared_error_coverage": {
            "within_1_sigma": float(np.mean(standardized <= 1.0)),
            "within_1_96_sigma": float(np.mean(standardized <= 1.96**2)),
            "within_2_576_sigma": float(np.mean(standardized <= 2.576**2)),
        },
    }


def _decision(
    *,
    reset_metrics: Mapping[str, Mapping[str, Any]],
    fitted_metrics: Mapping[str, Mapping[str, Any]],
    nonvariance_state_exact: bool,
    variance_weight_zero: bool,
    softplus_roundtrip_error: float,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    minimum_ratio = float(acceptance["minimum_heldout_error_to_variance_ratio"])
    maximum_ratio = float(acceptance["maximum_heldout_error_to_variance_ratio"])
    heldout_ratio = float(fitted_metrics["heldout"]["aggregate_error_to_variance_ratio"])
    checks = {
        "nonvariance_state_exact": nonvariance_state_exact,
        "legacy_variance_weight_zero": variance_weight_zero,
        "softplus_roundtrip_within_tolerance": softplus_roundtrip_error
        <= float(acceptance["maximum_softplus_roundtrip_absolute_error"]),
        "train_nll_not_above_reset": float(fitted_metrics["train"]["gaussian_nll_without_constant"])
        <= float(reset_metrics["train"]["gaussian_nll_without_constant"]) + 1e-12,
        "validation_nll_below_reset": float(
            fitted_metrics["validation"]["gaussian_nll_without_constant"]
        )
        < float(reset_metrics["validation"]["gaussian_nll_without_constant"]),
        "heldout_nll_below_reset": float(fitted_metrics["heldout"]["gaussian_nll_without_constant"])
        < float(reset_metrics["heldout"]["gaussian_nll_without_constant"]),
        "heldout_error_to_variance_ratio_in_bounds": minimum_ratio
        <= heldout_ratio
        <= maximum_ratio,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "later_gates_authorized": ["M3_bounded_mechanism_smoke"] if not failed else [],
        "long_training_authorized": False,
    }


def main() -> None:
    import torch

    from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder
    from picf_next.models.observation_calibration import (
        fit_axis_constant_observation_variance,
    )
    from picf_next.models.set_loss import ObjectSetCriterion
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets
    from picf_next.training.molmoact2_m2_source_coverage import (
        load_molmoact2_m2_source_coverage_recipe,
    )
    from tools import audit_molmoact2_m2_uncertainty as uncertainty
    from tools import run_molmoact2_m2_cloud as m2
    from tools import run_molmoact2_m2_frozen_mean_variance_probe as frozen
    from tools import run_molmoact2_m2_residual_permutation_probe as residual_probe
    from tools import run_molmoact2_m2_source_coverage_cloud as source

    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    if not m2._is_under_mnt(run_dir):
        raise RuntimeError("axis-variance calibration must bind a persistent /mnt run")
    if not torch.cuda.is_available():
        raise RuntimeError("axis-variance residual extraction requires CUDA")
    for relative, expected in config["source_artifact_sha256"].items():
        path = run_dir / relative
        if not path.is_file() or m2._sha256(path) != expected:
            raise ValueError(f"axis-variance source artifact changed: {relative}")
    residual_report_path = run_dir / "residual_permutation_probe/report.json"
    if m2._sha256(residual_report_path) != config["residual_permutation_report_sha256"]:
        raise ValueError("negative residual-permutation report is absent or changed")
    residual_report = uncertainty._load_json(residual_report_path)
    if residual_report.get("decision", {}).get("status") != _NEGATIVE_RESIDUAL_DECISION:
        raise RuntimeError("conditional-variance rejection is not established")

    source_config = (_ROOT / str(config["source_coverage_config_path"])).resolve()
    if m2._sha256(source_config) != config["source_coverage_config_sha256"]:
        raise ValueError("source-coverage config differs from the calibration binding")
    source_recipe = load_molmoact2_m2_source_coverage_recipe(source_config)
    recipe = source_recipe.load_base_m2(_ROOT)
    foundation = recipe.load_foundation(_ROOT)
    launch = uncertainty._load_json(run_dir / "launch_manifest.json")
    training = uncertainty._load_json(run_dir / "training_report.json")
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
    initial_nonvariance = m2._state_dict_sha256(frozen._state_without_variance(model))
    extraction_device = torch.device("cuda:0")
    model.to(extraction_device).eval()
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(extraction_device)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    keys_by_split = {
        split: m2._keys_for_split(cache, split) for split in ("train", "validation", "heldout")
    }
    examples_by_split = {
        split: residual_probe._collect_residual_examples(
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
    del criterion
    model.to(cpu)
    torch.cuda.empty_cache()
    post_extraction_nonvariance = m2._state_dict_sha256(frozen._state_without_variance(model))

    train_residual, train_target_variance, train_supervised = _examples_to_arrays(
        examples_by_split["train"]
    )
    calibration = fit_axis_constant_observation_variance(
        train_residual,
        train_target_variance,
        train_supervised=train_supervised,
        minimum_variance=foundation.core_config.discovery.minimum_variance,
    )
    reset_variance = [
        foundation.core_config.discovery.initial_variance
    ] * foundation.geometry_contract.dimension
    reset_metrics = {
        split: _calibration_metrics(examples, reset_variance)
        for split, examples in examples_by_split.items()
    }
    fitted_metrics = {
        split: _calibration_metrics(examples, calibration.observation_variance)
        for split, examples in examples_by_split.items()
    }

    with torch.no_grad():
        model.discovery.variance_head.weight.zero_()
        model.discovery.variance_head.bias.copy_(
            torch.as_tensor(
                calibration.raw_softplus_bias,
                dtype=model.discovery.variance_head.bias.dtype,
            )
        )
    actual_variance = (
        torch.nn.functional.softplus(model.discovery.variance_head.bias.float())
        + foundation.core_config.discovery.minimum_variance
    )
    expected_variance = torch.as_tensor(calibration.observation_variance, dtype=torch.float32)
    roundtrip_error = float((actual_variance - expected_variance).abs().max().item())
    final_nonvariance = m2._state_dict_sha256(frozen._state_without_variance(model))
    nonvariance_state_exact = (
        initial_nonvariance == post_extraction_nonvariance == final_nonvariance
    )
    variance_weight_zero = not bool(torch.count_nonzero(model.discovery.variance_head.weight))
    decision = _decision(
        reset_metrics=reset_metrics,
        fitted_metrics=fitted_metrics,
        nonvariance_state_exact=nonvariance_state_exact,
        variance_weight_zero=variance_weight_zero,
        softplus_roundtrip_error=roundtrip_error,
        acceptance=config["acceptance"],
    )

    output_dir = run_dir / _OUTPUT_NAME
    temporary = run_dir / f".{_OUTPUT_NAME}.tmp-{os.getpid()}"
    if output_dir.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite axis-variance calibration")
    temporary.mkdir()
    try:
        checkpoint_path = temporary / "current_frame_axis_calibrated.pt"
        m2._write_torch_atomic(
            checkpoint_path,
            {"model": m2._state_dict_cpu(model)},
        )
        metrics_path = temporary / "metrics.json"
        m2._write_json_atomic(
            metrics_path,
            {
                "schema": "picf-next.molmoact2-m2-axis-variance-metrics.v1",
                "calibration": {
                    "observation_variance": list(calibration.observation_variance),
                    "raw_softplus_bias": list(calibration.raw_softplus_bias),
                    "supervised_count": list(calibration.supervised_count),
                    "axis_nll_without_constant": list(calibration.axis_nll_without_constant),
                    "train_nll_without_constant": calibration.train_nll_without_constant,
                    "minimum_variance": calibration.minimum_variance,
                    "fit_method": list(calibration.fit_method),
                },
                "reset": reset_metrics,
                "fitted": fitted_metrics,
            },
        )
        report = {
            "schema": _SCHEMA,
            "status": "CALIBRATED_CANDIDATE" if decision["status"] == "PASS" else "REJECTED",
            "run_dir": str(run_dir),
            "audit_code_revision": m2._clean_git_revision(),
            "protocol": config["protocol"],
            "acceptance": config["acceptance"],
            "decision": decision,
            "data_isolation": {
                "fit_split": "train",
                "fit_rows": len(examples_by_split["train"]),
                "evaluation_only_rows": {
                    split: len(examples_by_split[split]) for split in ("validation", "heldout")
                },
            },
            "state_isolation": {
                "initial_nonvariance_state_sha256": initial_nonvariance,
                "post_extraction_nonvariance_state_sha256": post_extraction_nonvariance,
                "final_nonvariance_state_sha256": final_nonvariance,
                "nonvariance_state_exact": nonvariance_state_exact,
                "legacy_variance_weight_zero": variance_weight_zero,
                "softplus_roundtrip_maximum_absolute_error": roundtrip_error,
            },
            "sidecar_materialization": sidecar_materialization,
            "source_sidecar": source_sidecar,
            "input_sha256": {
                "config": m2._sha256(config_path),
                **config["source_artifact_sha256"],
                "residual_permutation_probe/report.json": m2._sha256(residual_report_path),
            },
            "output_sha256": {
                "metrics.json": m2._sha256(metrics_path),
                "current_frame_axis_calibrated.pt": m2._sha256(checkpoint_path),
            },
        }
        m2._write_json_atomic(temporary / "report.json", report)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
