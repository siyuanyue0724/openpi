#!/usr/bin/env python3
"""Diagnose a failed M2 geometry-uncertainty gate without changing the model."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
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

_SCHEMA = "picf-next.molmoact2-m2-uncertainty-diagnostic.v1"
_OUTPUT_NAME = "uncertainty_diagnostic"
_MODEL_SOURCE_PATHS = (
    "src/picf_next/models/core.py",
    "src/picf_next/models/discovery.py",
    "src/picf_next/models/set_loss.py",
    "tools/run_molmoact2_m2_cloud.py",
)
_UPSTREAM_REFERENCES = (
    {
        "name": "Faithful Heteroscedastic Regression",
        "repository": "https://github.com/astirn/faithful-heteroscedasticity",
        "revision": "5531f82a4f1199f48710d8af9aa487ab18f2eb98",
        "license": "MIT",
        "inspected_source": "models.py",
        "runtime_code_copied": False,
    },
    {
        "name": "beta-NLL",
        "repository": "https://github.com/martius-lab/beta-nll",
        "revision": "669c9f251eb41464c1ec8b43751c7c742dd75cf9",
        "license": "MIT",
        "inspected_source": "src/models/utils.py",
        "runtime_code_copied": False,
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_source_coverage.json",
    )
    return parser.parse_args()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal length")
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Pearson inputs must have equal length")
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    value = float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])
    return value if math.isfinite(value) else None


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    variance_field: str,
    error_field: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("uncertainty summary requires at least one row")
    variance = [_finite_float(row[variance_field], variance_field) for row in rows]
    error = [_finite_float(row[error_field], error_field) for row in rows]
    if any(value <= 0.0 for value in variance) or any(value < 0.0 for value in error):
        raise ValueError("uncertainty rows require positive variance and nonnegative error")
    standardized = [squared / predicted for squared, predicted in zip(error, variance, strict=True)]
    return {
        "row_count": len(rows),
        "variance_error_spearman": _safe_spearman(variance, error),
        "variance_error_pearson": _pearson(variance, error),
        "mean_predicted_variance": float(np.mean(variance)),
        "mean_squared_error": float(np.mean(error)),
        "mean_error_to_variance_ratio": float(np.mean(standardized)),
        "aggregate_error_to_variance_ratio": float(np.mean(error) / np.mean(variance)),
        "gaussian_nll_without_constant": float(
            np.mean(
                [
                    0.5 * (squared / predicted + math.log(predicted))
                    for squared, predicted in zip(error, variance, strict=True)
                ]
            )
        ),
        "standardized_squared_error_coverage": {
            "within_1_sigma": float(np.mean(np.asarray(standardized) <= 1.0)),
            "within_1_96_sigma": float(np.mean(np.asarray(standardized) <= 1.96**2)),
            "within_2_576_sigma": float(np.mean(np.asarray(standardized) <= 2.576**2)),
        },
        "minimum_predicted_variance": min(variance),
        "maximum_predicted_variance": max(variance),
    }


def _reliability_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    variance_field: str,
    error_field: str,
    bin_count: int = 10,
) -> dict[str, Any]:
    if not rows or bin_count <= 0:
        raise ValueError("reliability bins require rows and a positive bin count")
    ordered = sorted(
        rows,
        key=lambda row: (
            _finite_float(row[variance_field], variance_field),
            str(row.get("sample_key", "")),
            str(row.get("identity_key", "")),
            str(row.get("axis", "")),
        ),
    )
    bins = []
    split_indices = np.array_split(
        np.arange(len(ordered)),
        min(bin_count, len(ordered)),
    )
    for index, indices in enumerate(split_indices):
        selected = [ordered[int(row)] for row in indices]
        mean_variance = float(
            np.mean([_finite_float(row[variance_field], variance_field) for row in selected])
        )
        mean_error = float(
            np.mean([_finite_float(row[error_field], error_field) for row in selected])
        )
        bins.append(
            {
                "bin": index,
                "row_count": len(selected),
                "mean_predicted_variance": mean_variance,
                "mean_squared_error": mean_error,
                "error_to_variance_ratio": mean_error / mean_variance,
            }
        )
    return {
        "bins": bins,
        "binned_variance_error_spearman": _safe_spearman(
            [row["mean_predicted_variance"] for row in bins],
            [row["mean_squared_error"] for row in bins],
        ),
        "all_rows_accounted_for": sum(row["row_count"] for row in bins) == len(rows),
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    group_fields: tuple[str, ...],
) -> dict[tuple[object, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    return dict(grouped)


def _group_decomposition(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...],
    variance_field: str,
    error_field: str,
) -> dict[str, Any]:
    grouped = _group_rows(rows, group_fields)
    group_summaries = []
    centered_variance_ranks: list[float] = []
    centered_error_ranks: list[float] = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        variance = [_finite_float(row[variance_field], variance_field) for row in selected]
        error = [_finite_float(row[error_field], error_field) for row in selected]
        coefficient = _safe_spearman(variance, error)
        group_summaries.append(
            {
                "group": list(key),
                "row_count": len(selected),
                "mean_predicted_variance": float(np.mean(variance)),
                "mean_squared_error": float(np.mean(error)),
                "variance_error_spearman": coefficient,
            }
        )
        if coefficient is not None:
            variance_rank = rankdata(variance, method="average")
            error_rank = rankdata(error, method="average")
            centered_variance_ranks.extend(
                (variance_rank - variance_rank.mean()).astype(float).tolist()
            )
            centered_error_ranks.extend((error_rank - error_rank.mean()).astype(float).tolist())
    valid_coefficients = [
        row["variance_error_spearman"]
        for row in group_summaries
        if row["variance_error_spearman"] is not None
    ]
    between = _safe_spearman(
        [row["mean_predicted_variance"] for row in group_summaries],
        [row["mean_squared_error"] for row in group_summaries],
    )
    return {
        "group_fields": list(group_fields),
        "group_count": len(group_summaries),
        "groups_with_defined_spearman": len(valid_coefficients),
        "between_group_mean_spearman": between,
        "within_group_centered_rank_correlation": _pearson(
            centered_variance_ranks,
            centered_error_ranks,
        ),
        "median_group_spearman": (
            float(np.median(valid_coefficients)) if valid_coefficients else None
        ),
        "negative_group_spearman_fraction": (
            sum(value < 0.0 for value in valid_coefficients) / len(valid_coefficients)
            if valid_coefficients
            else None
        ),
        "groups": group_summaries,
    }


def summarize_uncertainty(
    object_rows: Sequence[Mapping[str, Any]],
    coordinate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return gate-level and coordinate-level calibration decompositions."""

    if not object_rows or not coordinate_rows:
        raise ValueError("uncertainty audit requires object and coordinate rows")
    split_names = sorted({str(row["split"]) for row in object_rows})
    split_summary = {}
    for split in split_names:
        objects = [row for row in object_rows if row["split"] == split]
        coordinates = [row for row in coordinate_rows if row["split"] == split]
        split_summary[split] = {
            "gate_object_normalized": {
                **_metric_summary(
                    objects,
                    variance_field="predicted_variance_normalized_mean",
                    error_field="squared_error_normalized_mean",
                ),
                "reliability": _reliability_bins(
                    objects,
                    variance_field="predicted_variance_normalized_mean",
                    error_field="squared_error_normalized_mean",
                ),
            },
            "coordinate_normalized": _metric_summary(
                coordinates,
                variance_field="predicted_variance_normalized",
                error_field="squared_error_normalized",
            ),
            "coordinate_physical": _metric_summary(
                coordinates,
                variance_field="predicted_variance_physical",
                error_field="squared_error_physical",
            ),
            "decomposition": {
                "axis": _group_decomposition(
                    coordinates,
                    group_fields=("axis",),
                    variance_field="predicted_variance_normalized",
                    error_field="squared_error_normalized",
                ),
                "identity_axis": _group_decomposition(
                    coordinates,
                    group_fields=("identity_key", "axis"),
                    variance_field="predicted_variance_normalized",
                    error_field="squared_error_normalized",
                ),
                "query_axis": _group_decomposition(
                    coordinates,
                    group_fields=("query_index", "axis"),
                    variance_field="predicted_variance_normalized",
                    error_field="squared_error_normalized",
                ),
            },
        }
    return {"splits": split_summary}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source_root_from_launch(launch: Mapping[str, Any]) -> Path:
    config = Path(str(launch["config"])).expanduser().resolve()
    if config.parent.name != "training" or config.parent.parent.name != "configs":
        raise ValueError("M2 launch config does not identify a repository root")
    return config.parents[2]


def _source_equivalence(
    *,
    run_source_root: Path,
    current_root: Path,
    sha256: Any,
) -> dict[str, Any]:
    rows = {}
    for relative in _MODEL_SOURCE_PATHS:
        run_path = run_source_root / relative
        current_path = current_root / relative
        if not run_path.is_file() or not current_path.is_file():
            raise FileNotFoundError(f"model source is absent: {relative}")
        run_hash = sha256(run_path)
        current_hash = sha256(current_path)
        rows[relative] = {
            "run_source_sha256": run_hash,
            "audit_source_sha256": current_hash,
            "exact": run_hash == current_hash,
        }
    if not all(row["exact"] for row in rows.values()):
        raise ValueError("audit model source differs from the source that produced M2")
    return rows


def _collect_rows(
    *,
    model: Any,
    model_arm: str,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
    target_builder: Any,
    criterion: Any,
    layout_payload: Sequence[Mapping[str, Any]],
    recipe: Any,
    device: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    from tools import run_molmoact2_m2_cloud as m2

    model.eval()
    contract = target_builder.sidecar.geometry_contract
    object_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
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
        prediction = output.discovery
        for batch_index, (target, match, key) in enumerate(
            zip(targets, result.matches, batch_keys, strict=True)
        ):
            identities = target.temporal_identity_keys
            if identities is None or len(identities) != target.num_objects:
                raise RuntimeError("uncertainty audit requires explicit physical identities")
            if target.geometry is None or target.geometry_supervised is None:
                raise RuntimeError("uncertainty audit requires selective geometry targets")
            target_variance = target.geometry_variance
            if target_variance is None:
                target_variance = torch.zeros_like(target.geometry)
            record = cache[key][2]
            group_kind, group_index = m2._record_group_identity(record)
            token_rows = target.supervision_valid
            for row, (query_index, target_index) in enumerate(
                zip(
                    match.prediction_indices.tolist(),
                    match.target_indices.tolist(),
                    strict=True,
                )
            ):
                selected = target.geometry_supervised[target_index]
                if not bool(selected.any()):
                    continue
                predicted = prediction.geometry_mean[batch_index, query_index].float()
                expected = target.geometry[target_index].float()
                predicted_variance = prediction.geometry_variance[batch_index, query_index].float()
                measurement_variance = target_variance[target_index].float()
                residual = predicted - expected
                selected_indices = torch.nonzero(selected, as_tuple=False).flatten().tolist()

                predicted_mask = prediction.ownership[batch_index, token_rows, query_index].float()
                expected_mask = target.ownership[token_rows, target_index].float()
                dice = float(
                    (
                        (2.0 * (predicted_mask * expected_mask).sum() + 1.0)
                        / (predicted_mask.sum() + expected_mask.sum() + 1.0)
                    ).item()
                )
                common = {
                    "model_arm": model_arm,
                    "sample_key": key,
                    "split": str(record["split"]),
                    "global_index": int(record["global_index"]),
                    "task_key": str(record["task_key"]),
                    "group_kind": group_kind,
                    "group_index": group_index,
                    "identity_key": identities[target_index],
                    "query_index": query_index,
                    "target_index": target_index,
                    "match_row": row,
                    "existence_probability": float(
                        prediction.existence[batch_index, query_index].float().item()
                    ),
                    "object_dice": dice,
                }
                selected_residual = residual[selected]
                selected_predicted_variance = predicted_variance[selected]
                selected_measurement_variance = measurement_variance[selected]
                scales = torch.as_tensor(
                    contract.normalization_scale,
                    dtype=torch.float32,
                    device=residual.device,
                )[selected]
                object_rows.append(
                    {
                        **common,
                        "supervised_coordinate_count": len(selected_indices),
                        "squared_error_normalized_mean": float(
                            selected_residual.square().mean().item()
                        ),
                        "predicted_variance_normalized_mean": float(
                            selected_predicted_variance.mean().item()
                        ),
                        "measurement_variance_normalized_mean": float(
                            selected_measurement_variance.mean().item()
                        ),
                        "squared_error_physical_mean": float(
                            (selected_residual * scales).square().mean().item()
                        ),
                        "predicted_variance_physical_mean": float(
                            (selected_predicted_variance * scales.square()).mean().item()
                        ),
                    }
                )
                for axis_index in selected_indices:
                    scale = float(contract.normalization_scale[axis_index])
                    coordinate_rows.append(
                        {
                            **common,
                            "axis_index": axis_index,
                            "axis": contract.axes[axis_index],
                            "unit": contract.units[axis_index],
                            "normalization_scale": scale,
                            "predicted_mean_normalized": float(predicted[axis_index].item()),
                            "target_normalized": float(expected[axis_index].item()),
                            "residual_normalized": float(residual[axis_index].item()),
                            "squared_error_normalized": float(residual[axis_index].square().item()),
                            "predicted_variance_normalized": float(
                                predicted_variance[axis_index].item()
                            ),
                            "measurement_variance_normalized": float(
                                measurement_variance[axis_index].item()
                            ),
                            "squared_error_physical": float(
                                (residual[axis_index] * scale).square().item()
                            ),
                            "predicted_variance_physical": float(
                                predicted_variance[axis_index].item() * scale * scale
                            ),
                        }
                    )
        del tokens, valid, output
    return object_rows, coordinate_rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="ascii") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")


def _load_model(
    *,
    foundation: Any,
    checkpoint: Path,
    expected_sha256: str,
    device: Any,
    sha256: Any,
) -> Any:
    import torch

    if not checkpoint.is_file() or sha256(checkpoint) != expected_sha256:
        raise ValueError(f"M2 checkpoint is absent or changed: {checkpoint.name}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"model"} or not isinstance(payload["model"], dict):
        raise ValueError("M2 checkpoint payload changed")
    model = foundation.core_config.build_current_frame()
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def main() -> None:
    args = _parse_args()

    import torch

    from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder
    from picf_next.models.set_loss import ObjectSetCriterion
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets
    from picf_next.training.molmoact2_m2_source_coverage import (
        load_molmoact2_m2_source_coverage_recipe,
    )
    from tools import run_molmoact2_m2_cloud as m2
    from tools import run_molmoact2_m2_source_coverage_cloud as source

    run_dir = args.run_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not m2._is_under_mnt(run_dir):
        raise RuntimeError("M2 uncertainty diagnostics must bind a persistent /mnt run")
    decision = source.validate_source_coverage_machine_decision(run_dir)
    if decision.get("status") != "FAIL" or decision.get("failed_checks") != [
        "uncertainty_ranks_errors"
    ]:
        raise RuntimeError(
            "uncertainty diagnostic requires an otherwise-passing source-coverage M2 run"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("M2 uncertainty diagnostic requires CUDA")
    audit_revision = m2._clean_git_revision()
    launch = _load_json(run_dir / "launch_manifest.json")
    training = _load_json(run_dir / "training_report.json")
    evaluation = _load_json(run_dir / "evaluation_report.json")
    if m2._sha256(config) != launch.get("config_file_sha256"):
        raise ValueError("audit config differs from the M2 launch config")
    source_recipe = load_molmoact2_m2_source_coverage_recipe(config)
    base_recipe = source_recipe.load_base_m2(_ROOT)
    foundation = base_recipe.load_foundation(_ROOT)
    source_equivalence = _source_equivalence(
        run_source_root=_source_root_from_launch(launch),
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
    cache_manifest, cache = m2._load_cache(run_dir / "feature_cache", base_recipe)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    layout_payload = cache_manifest["processor_layout"]
    checkpoint_hashes = training.get("checkpoints")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("M2 training report has no checkpoint hashes")

    devices = [torch.device("cuda:0")]
    if torch.cuda.device_count() >= 2:
        devices.append(torch.device("cuda:1"))
    actual = _load_model(
        foundation=foundation,
        checkpoint=run_dir / "checkpoints/current_frame_best.pt",
        expected_sha256=str(checkpoint_hashes["current_frame_best.pt"]),
        device=devices[0],
        sha256=m2._sha256,
    )
    control = _load_model(
        foundation=foundation,
        checkpoint=run_dir / "checkpoints/label_shuffle_paired_best_step.pt",
        expected_sha256=str(checkpoint_hashes["label_shuffle_paired_best_step.pt"]),
        device=devices[-1],
        sha256=m2._sha256,
    )
    actual_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(devices[0])
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(devices[-1])

    object_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "heldout"):
        keys = m2._keys_for_split(cache, split)
        objects, coordinates = _collect_rows(
            model=actual,
            model_arm="actual",
            cache=cache,
            keys=keys,
            target_builder=target_builder,
            criterion=actual_criterion,
            layout_payload=layout_payload,
            recipe=base_recipe,
            device=devices[0],
        )
        object_rows.extend(objects)
        coordinate_rows.extend(coordinates)
    control_objects, control_coordinates = _collect_rows(
        model=control,
        model_arm="label_shuffle",
        cache=cache,
        keys=m2._keys_for_split(cache, "heldout"),
        target_builder=target_builder,
        criterion=control_criterion,
        layout_payload=layout_payload,
        recipe=base_recipe,
        device=devices[-1],
    )

    actual_object_rows = [row for row in object_rows if row["model_arm"] == "actual"]
    actual_coordinate_rows = [row for row in coordinate_rows if row["model_arm"] == "actual"]
    actual_summary = summarize_uncertainty(actual_object_rows, actual_coordinate_rows)
    control_summary = summarize_uncertainty(control_objects, control_coordinates)
    reproduction = {}
    for split in ("train", "validation", "heldout"):
        expected = evaluation["actual"][split]["uncertainty_error_spearman"]
        observed = actual_summary["splits"][split]["gate_object_normalized"][
            "variance_error_spearman"
        ]
        exact = (
            expected is None
            and observed is None
            or expected is not None
            and observed is not None
            and math.isclose(float(expected), float(observed), rel_tol=0.0, abs_tol=1e-6)
        )
        reproduction[split] = {
            "reported": expected,
            "reproduced": observed,
            "within_absolute_tolerance_1e_6": exact,
        }
    if not all(row["within_absolute_tolerance_1e_6"] for row in reproduction.values()):
        raise RuntimeError("uncertainty diagnostic did not reproduce the original M2 metric")

    def sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, int, int]:
        return (
            ("train", "validation", "heldout").index(str(row["split"])),
            int(row["global_index"]),
            str(row["identity_key"]),
            int(row["query_index"]),
            int(row.get("axis_index", -1)),
        )

    object_rows = sorted((*object_rows, *control_objects), key=sort_key)
    coordinate_rows = sorted((*coordinate_rows, *control_coordinates), key=sort_key)
    output_dir = run_dir / _OUTPUT_NAME
    temporary = run_dir / f".{_OUTPUT_NAME}.tmp-{os.getpid()}"
    if output_dir.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite M2 uncertainty diagnostic")
    temporary.mkdir()
    try:
        _write_jsonl(temporary / "object_rows.jsonl", object_rows)
        _write_jsonl(temporary / "coordinate_rows.jsonl", coordinate_rows)
        report = {
            "schema": _SCHEMA,
            "gate": source_recipe.gate,
            "status": "DIAGNOSTIC_ONLY",
            "later_gates_authorized": [],
            "training_changes_authorized": [],
            "run_dir": str(run_dir),
            "run_code_revision": launch["code_revision"],
            "audit_code_revision": audit_revision,
            "source_equivalence": source_equivalence,
            "sidecar_materialization": sidecar_materialization,
            "source_sidecar": source_sidecar,
            "input_sha256": {
                "machine_decision.json": m2._sha256(run_dir / "machine_decision.json"),
                "evaluation_report.json": m2._sha256(run_dir / "evaluation_report.json"),
                "training_report.json": m2._sha256(run_dir / "training_report.json"),
                "feature_cache/manifest.json": m2._sha256(run_dir / "feature_cache/manifest.json"),
                "checkpoints/current_frame_best.pt": checkpoint_hashes["current_frame_best.pt"],
                "checkpoints/label_shuffle_paired_best_step.pt": checkpoint_hashes[
                    "label_shuffle_paired_best_step.pt"
                ],
            },
            "metric_reproduction": reproduction,
            "actual": actual_summary,
            "label_shuffle_heldout": control_summary["splits"]["heldout"],
            "upstream_references": list(_UPSTREAM_REFERENCES),
            "raw_rows": {
                "object_rows.jsonl": {
                    "row_count": len(object_rows),
                    "sha256": m2._sha256(temporary / "object_rows.jsonl"),
                },
                "coordinate_rows.jsonl": {
                    "row_count": len(coordinate_rows),
                    "sha256": m2._sha256(temporary / "coordinate_rows.jsonl"),
                },
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
