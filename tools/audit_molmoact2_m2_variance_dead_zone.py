#!/usr/bin/env python3
"""Prove or reject saturation in the trained M2 observation-variance head."""

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

_SCHEMA = "picf-next.molmoact2-m2-variance-dead-zone-probe.v1"
_OUTPUT_NAME = "variance_dead_zone_probe"


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


def _subset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "exact_zero_gradient_count": 0,
            "exact_zero_gradient_fraction": None,
            "mean_absolute_gradient": None,
            "maximum_absolute_gradient": None,
            "mean_counterfactual_softplus_absolute_local_gradient": None,
        }
    gradient = np.asarray(
        [_finite_float(row["raw_gradient"], "raw_gradient") for row in rows],
        dtype=np.float64,
    )
    counterfactual = np.asarray(
        [
            abs(
                _finite_float(
                    row["counterfactual_softplus_local_gradient"],
                    "counterfactual_softplus_local_gradient",
                )
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    absolute = np.abs(gradient)
    exact_zero = absolute == 0.0
    return {
        "row_count": len(rows),
        "exact_zero_gradient_count": int(exact_zero.sum()),
        "exact_zero_gradient_fraction": float(exact_zero.mean()),
        "mean_absolute_gradient": float(absolute.mean()),
        "maximum_absolute_gradient": float(absolute.max()),
        "mean_counterfactual_softplus_absolute_local_gradient": float(counterfactual.mean()),
    }


def summarize_dead_zone(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_variance: float,
) -> dict[str, Any]:
    """Summarize elementwise gradients through the trained variance transform."""

    if not rows:
        raise ValueError("dead-zone probe requires at least one supervised coordinate")
    if not math.isfinite(minimum_variance) or minimum_variance <= 0.0:
        raise ValueError("minimum_variance must be finite and positive")
    lower_raw = math.log(minimum_variance)
    saturated = [row for row in rows if _finite_float(row["variance_raw"], "variance_raw") > 0.0]
    floor_saturated = [
        row for row in rows if _finite_float(row["variance_raw"], "variance_raw") < lower_raw
    ]
    interior = [
        row for row in rows if lower_raw < _finite_float(row["variance_raw"], "variance_raw") < 0.0
    ]
    all_summary = _subset_summary(rows)
    saturated_summary = _subset_summary(saturated)
    floor_summary = _subset_summary(floor_saturated)
    interior_summary = _subset_summary(interior)
    saturated_has_counterfactual_signal = any(
        abs(
            _finite_float(
                row["counterfactual_softplus_local_gradient"],
                "counterfactual_softplus_local_gradient",
            )
        )
        > 0.0
        for row in saturated
    )
    dead_zone_established = (
        bool(saturated)
        and saturated_summary["exact_zero_gradient_fraction"] == 1.0
        and saturated_has_counterfactual_signal
        and bool(interior)
        and interior_summary["maximum_absolute_gradient"] is not None
        and interior_summary["maximum_absolute_gradient"] > 0.0
    )
    return {
        "supervised_coordinate_count": len(rows),
        "current_transform": {
            "equation": "exp(clamp(raw, log(minimum_variance), 0))",
            "minimum_variance": minimum_variance,
            "maximum_variance": 1.0,
            "upper_dead_zone_condition": "raw > 0",
            "lower_dead_zone_condition": "raw < log(minimum_variance)",
        },
        "all": all_summary,
        "upper_saturated": saturated_summary,
        "lower_saturated": floor_summary,
        "interior": interior_summary,
        "upper_saturated_counterfactual_has_nonzero_signal": (saturated_has_counterfactual_signal),
        "upper_dead_zone_established": dead_zone_established,
    }


def _counterfactual_softplus_local_gradient(
    raw: float,
    squared_residual: float,
    target_variance: float,
    minimum_variance: float,
) -> float:
    softplus = raw + math.log1p(math.exp(-raw)) if raw >= 0.0 else math.log1p(math.exp(raw))
    predicted_variance = softplus + minimum_variance
    combined_variance = predicted_variance + target_variance
    if raw >= 0.0:
        sigmoid = 1.0 / (1.0 + math.exp(-raw))
    else:
        exp_raw = math.exp(raw)
        sigmoid = exp_raw / (1.0 + exp_raw)
    return (
        0.5
        * (-squared_residual / (combined_variance * combined_variance) + 1.0 / combined_variance)
        * sigmoid
    )


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
    config = args.config.expanduser().resolve()
    if not m2._is_under_mnt(run_dir):
        raise RuntimeError("variance dead-zone probes must bind a persistent /mnt run")
    decision = source.validate_source_coverage_machine_decision(run_dir)
    if decision.get("status") != "FAIL" or decision.get("failed_checks") != [
        "uncertainty_ranks_errors"
    ]:
        raise RuntimeError("probe requires an otherwise-passing source-coverage M2 run")
    if not torch.cuda.is_available():
        raise RuntimeError("variance dead-zone probe requires CUDA")

    audit_revision = m2._clean_git_revision()
    launch = uncertainty._load_json(run_dir / "launch_manifest.json")
    training = uncertainty._load_json(run_dir / "training_report.json")
    if m2._sha256(config) != launch.get("config_file_sha256"):
        raise ValueError("probe config differs from the M2 launch config")
    source_recipe = load_molmoact2_m2_source_coverage_recipe(config)
    recipe = source_recipe.load_base_m2(_ROOT)
    foundation = recipe.load_foundation(_ROOT)
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

    device = torch.device("cuda:0")
    model = uncertainty._load_model(
        foundation=foundation,
        checkpoint=run_dir / "checkpoints/current_frame_best.pt",
        expected_sha256=str(checkpoint_hashes["current_frame_best.pt"]),
        device=device,
        sha256=m2._sha256,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.discovery.variance_head.parameters():
        parameter.requires_grad_(True)
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(device)
    minimum_variance = model.discovery.config.minimum_variance
    layout_payload = cache_manifest["processor_layout"]

    rows: list[dict[str, Any]] = []
    captured_raw: list[torch.Tensor] = []

    def capture_raw(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        captured_raw.append(output)

    handle = model.discovery.variance_head.register_forward_hook(capture_raw)
    try:
        keys = m2._keys_for_split(cache, "heldout")
        for start in range(0, len(keys), recipe.optimization.batch_size):
            batch_keys = keys[start : start + recipe.optimization.batch_size]
            tokens, valid, records = m2._stack_batch(cache, batch_keys, device=device)
            captured_raw.clear()
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
            result.losses["loss_geometry_calibration"].backward()
            if not captured_raw or captured_raw[-1].grad is None:
                raise RuntimeError("final variance-head activation did not retain a gradient")
            raw = captured_raw[-1].detach().float()
            raw_gradient = captured_raw[-1].grad.detach().float()
            prediction = output.discovery
            for batch_index, (target, match, key) in enumerate(
                zip(targets, result.matches, batch_keys, strict=True)
            ):
                identities = target.temporal_identity_keys
                if (
                    identities is None
                    or target.geometry is None
                    or target.geometry_supervised is None
                ):
                    raise RuntimeError("probe requires identified selective geometry targets")
                target_variance = target.geometry_variance
                if target_variance is None:
                    target_variance = torch.zeros_like(target.geometry)
                record = cache[key][2]
                for query_index, target_index in zip(
                    match.prediction_indices.tolist(),
                    match.target_indices.tolist(),
                    strict=True,
                ):
                    supervised = target.geometry_supervised[target_index]
                    for axis_index in torch.nonzero(supervised, as_tuple=False).flatten().tolist():
                        raw_value = float(raw[batch_index, query_index, axis_index].item())
                        gradient_value = float(
                            raw_gradient[batch_index, query_index, axis_index].item()
                        )
                        residual = (
                            prediction.geometry_mean[batch_index, query_index, axis_index].float()
                            - target.geometry[target_index, axis_index].float()
                        )
                        squared_residual = float(residual.square().item())
                        measurement_variance = float(
                            target_variance[target_index, axis_index].float().item()
                        )
                        rows.append(
                            {
                                "sample_key": key,
                                "global_index": int(record["global_index"]),
                                "task_key": str(record["task_key"]),
                                "identity_key": identities[target_index],
                                "query_index": query_index,
                                "target_index": target_index,
                                "axis_index": axis_index,
                                "axis": target.geometry_contract.axes[axis_index],
                                "variance_raw": raw_value,
                                "predicted_variance": float(
                                    prediction.geometry_variance[
                                        batch_index, query_index, axis_index
                                    ]
                                    .float()
                                    .item()
                                ),
                                "target_measurement_variance": measurement_variance,
                                "squared_residual": squared_residual,
                                "raw_gradient": gradient_value,
                                "counterfactual_softplus_local_gradient": (
                                    _counterfactual_softplus_local_gradient(
                                        raw_value,
                                        squared_residual,
                                        measurement_variance,
                                        minimum_variance,
                                    )
                                ),
                            }
                        )
            del output, result, tokens, valid
    finally:
        handle.remove()

    rows.sort(
        key=lambda row: (
            int(row["global_index"]),
            str(row["identity_key"]),
            int(row["query_index"]),
            int(row["axis_index"]),
        )
    )
    summary = summarize_dead_zone(rows, minimum_variance=minimum_variance)
    output_dir = run_dir / _OUTPUT_NAME
    temporary = run_dir / f".{_OUTPUT_NAME}.tmp-{os.getpid()}"
    if output_dir.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite variance dead-zone probe")
    temporary.mkdir()
    try:
        _write_jsonl(temporary / "rows.jsonl", rows)
        report = {
            "schema": _SCHEMA,
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
                "training_report.json": m2._sha256(run_dir / "training_report.json"),
                "feature_cache/manifest.json": m2._sha256(run_dir / "feature_cache/manifest.json"),
                "checkpoints/current_frame_best.pt": checkpoint_hashes["current_frame_best.pt"],
            },
            "summary": summary,
            "raw_rows": {
                "path": "rows.jsonl",
                "row_count": len(rows),
                "sha256": m2._sha256(temporary / "rows.jsonl"),
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
