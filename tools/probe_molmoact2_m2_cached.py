#!/usr/bin/env python3
"""Run a non-authorizing M2 development probe from a verified feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.models.discovery import ObjectExistenceCalibration  # noqa: E402
from picf_next.training.molmoact2_calvin import load_calvin_training_assets  # noqa: E402
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _load_cache,
    _render_visuals,
    _train_models,
    _write_json_atomic,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--existence-weight", type=float)
    parser.add_argument("--ownership-ce-weight", type=float)
    parser.add_argument("--ownership-dice-weight", type=float)
    parser.add_argument("--geometry-weight", type=float)
    parser.add_argument("--unmatched-query-weight", type=float)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_identity() -> dict[str, object]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        text=True,
    ).strip()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"],
        cwd=_ROOT,
    )
    paths = (
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "src/picf_next/hosts/molmoact2_training.py",
        "tools/run_molmoact2_m2_cloud.py",
    )
    return {
        "base_revision": revision,
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "audited_file_sha256": {
            relative: _sha256_bytes((_ROOT / relative).read_bytes()) for relative in paths
        },
        "clean_authorizing_run": False,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    recipe = load_molmoact2_m2_recipe(args.config.resolve())
    optimization = replace(
        recipe.optimization,
        steps=args.steps,
        validation_interval=args.validation_interval,
        warmup_steps=min(recipe.optimization.warmup_steps, args.steps - 1),
    )
    recipe = replace(recipe, optimization=optimization)
    foundation = recipe.load_foundation(_ROOT)
    weight_overrides = {
        "existence_weight": args.existence_weight,
        "ownership_ce_weight": args.ownership_ce_weight,
        "ownership_dice_weight": args.ownership_dice_weight,
        "geometry_weight": args.geometry_weight,
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
        calibration = ObjectExistenceCalibration(args.unmatched_query_weight)
        foundation = replace(
            foundation,
            core_config=replace(
                foundation.core_config,
                discovery=replace(
                    foundation.core_config.discovery,
                    existence_calibration=calibration,
                ),
            ),
        )
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=args.dataset_split_root.expanduser().resolve(),
    )
    feature_cache = args.feature_cache.expanduser().resolve()
    cache_manifest, cache = _load_cache(feature_cache, recipe)
    _write_json_atomic(
        output_dir / "probe_manifest.json",
        {
            "schema": "picf-next.molmoact2-m2-cached-probe.v1",
            "authorizes_later_gates": False,
            "source": _source_identity(),
            "config": str(args.config.resolve()),
            "recipe": recipe.to_dict(),
            "recipe_sha256": recipe.recipe_sha256,
            "development_set_loss_weight_overrides": applied_weight_overrides,
            "development_unmatched_query_weight_override": args.unmatched_query_weight,
            "feature_cache": str(feature_cache),
            "feature_cache_manifest_sha256": _sha256_bytes(
                (feature_cache / "manifest.json").read_bytes()
            ),
            "dataset_split_root": str(args.dataset_split_root.expanduser().resolve()),
        },
    )
    training, evaluation, model = _train_models(
        run_dir=output_dir,
        recipe=recipe,
        foundation=foundation,
        assets=assets,
        cache_manifest=cache_manifest,
        cache=cache,
    )
    _write_json_atomic(output_dir / "training_report.json", training)
    _write_json_atomic(output_dir / "evaluation_report.json", evaluation)
    visuals = _render_visuals(
        run_dir=output_dir,
        model=model,
        assets=assets,
        cache=cache,
        cache_manifest=cache_manifest,
        foundation=foundation,
        recipe=recipe,
    )
    _write_json_atomic(output_dir / "visual_artifacts.json", visuals)
    heldout = evaluation["actual"]["heldout"]
    summary = {
        "best_validation_step": training["best_validation_step"],
        "seconds_per_joint_actual_and_control_step": training[
            "seconds_per_joint_actual_and_control_step"
        ],
        "heldout_balanced_ownership_accuracy": heldout["balanced_ownership_accuracy"],
        "heldout_token_ownership_accuracy": heldout["token_ownership_accuracy"],
        "heldout_mean_object_dice": heldout["mean_object_dice"],
        "heldout_exact_count_accuracy": heldout["exact_count_accuracy"],
        "heldout_count_mae": heldout["count_mae"],
        "heldout_geometry_mae_model_chart": heldout["geometry_mae_model_chart"],
        "heldout_geometry_mae_physical": heldout["geometry_mae_physical"],
        "heldout_geometry_mae_physical_unit": heldout["geometry_mae_physical_unit"],
        "heldout_mean_active_queries": heldout["mean_active_queries"],
        "label_shuffle_mean_object_dice": evaluation["label_shuffle"]["mean_object_dice"],
        "random_mean_object_dice": evaluation["random_initialization"]["mean_object_dice"],
        "all_context_balanced_ownership_accuracy": evaluation["all_context"][
            "balanced_ownership_accuracy"
        ],
    }
    _write_json_atomic(output_dir / "probe_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
