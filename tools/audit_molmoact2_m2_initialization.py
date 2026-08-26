#!/usr/bin/env python3
"""Run a non-authorizing M2 geometry-initialization factorial audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.models.discovery import (  # noqa: E402
    TaskIndependentObjectDiscovery,
    _inverse_softplus,
)
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _load_cache,
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
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument(
        "--geometry-mean-initialization",
        choices=("chart_origin", "linear_default"),
        required=True,
    )
    parser.add_argument("--initial-variance", type=float, required=True)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_identity() -> dict[str, Any]:
    paths = (
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "tools/audit_molmoact2_m2_initialization.py",
        "tools/run_molmoact2_m2_cloud.py",
    )
    return {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
        ).strip(),
        "tracked_diff_sha256": _sha256_bytes(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=_ROOT)
        ),
        "audited_file_sha256": {
            relative: _sha256_bytes((_ROOT / relative).read_bytes()) for relative in paths
        },
        "clean_authorizing_run": False,
    }


def _linear_default_geometry_reset(self: TaskIndependentObjectDiscovery) -> None:
    """Reproduce the old reset while retaining nn.Linear's geometry draw."""

    import torch
    from torch import nn

    nn.init.normal_(self.query_embeddings, std=self.config.hidden_dim**-0.5)
    nn.init.zeros_(self.existence_head.weight)
    nn.init.constant_(
        self.existence_head.bias,
        self.config.existence_calibration.training_logit_at_half_posterior,
    )
    nn.init.zeros_(self.context_head.bias)
    nn.init.zeros_(self.variance_head.weight)
    nn.init.constant_(
        self.variance_head.bias,
        _inverse_softplus(self.config.initial_variance - self.config.minimum_variance),
    )
    self.variance_head.weight.requires_grad_(False)
    if not torch.count_nonzero(self.geometry_head.weight):
        raise RuntimeError("linear-default geometry initialization was unexpectedly zero")


@contextmanager
def _geometry_mean_initialization(mode: str):
    if mode == "chart_origin":
        yield
        return
    if mode != "linear_default":
        raise ValueError(f"unsupported geometry mean initialization: {mode}")
    original = TaskIndependentObjectDiscovery.reset_parameters
    TaskIndependentObjectDiscovery.reset_parameters = _linear_default_geometry_reset
    try:
        yield
    finally:
        TaskIndependentObjectDiscovery.reset_parameters = original


def main() -> None:
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets

    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.validation_interval <= 0:
        raise ValueError("validation interval must be positive")
    if not math.isfinite(args.initial_variance) or args.initial_variance <= 0.0:
        raise ValueError("initial variance must be finite and positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    recipe = load_molmoact2_m2_recipe(args.config.resolve())
    recipe = replace(
        recipe,
        optimization=replace(
            recipe.optimization,
            steps=args.steps,
            validation_interval=args.validation_interval,
            warmup_steps=min(recipe.optimization.warmup_steps, args.steps - 1),
        ),
    )
    foundation = recipe.load_foundation(_ROOT)
    discovery = replace(
        foundation.core_config.discovery,
        initial_variance=args.initial_variance,
    )
    foundation = replace(
        foundation,
        core_config=replace(foundation.core_config, discovery=discovery),
    )
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=args.dataset_split_root.expanduser().resolve(),
    )
    feature_cache = args.feature_cache.expanduser().resolve()
    cache_manifest, cache = _load_cache(feature_cache, recipe)
    _write_json_atomic(
        output_dir / "audit_manifest.json",
        {
            "schema": "picf-next.molmoact2-m2-initialization-audit.v1",
            "authorizes_later_gates": False,
            "source": _source_identity(),
            "config": str(args.config.resolve()),
            "recipe": recipe.to_dict(),
            "recipe_sha256": recipe.recipe_sha256,
            "geometry_mean_initialization": args.geometry_mean_initialization,
            "initial_variance": args.initial_variance,
            "feature_cache": str(feature_cache),
            "feature_cache_manifest_sha256": _sha256_bytes(
                (feature_cache / "manifest.json").read_bytes()
            ),
            "dataset_split_root": str(args.dataset_split_root.expanduser().resolve()),
        },
    )

    with _geometry_mean_initialization(args.geometry_mean_initialization):
        training, evaluation, _model = _train_models(
            run_dir=output_dir,
            recipe=recipe,
            foundation=foundation,
            assets=assets,
            cache_manifest=cache_manifest,
            cache=cache,
        )
    _write_json_atomic(output_dir / "training_report.json", training)
    _write_json_atomic(output_dir / "evaluation_report.json", evaluation)
    heldout = evaluation["actual"]["heldout"]
    summary = {
        "geometry_mean_initialization": args.geometry_mean_initialization,
        "initial_variance": args.initial_variance,
        "best_validation_step": training["best_validation_step"],
        "seconds_per_joint_actual_and_control_step": training[
            "seconds_per_joint_actual_and_control_step"
        ],
        "validation": evaluation["actual"]["validation"],
        "heldout": {
            key: heldout[key]
            for key in (
                "balanced_ownership_accuracy",
                "token_ownership_accuracy",
                "mean_object_dice",
                "count_mae",
                "exact_count_accuracy",
                "geometry_mae_model_chart",
                "geometry_mae_physical",
                "geometry_mae_physical_unit",
                "mean_active_queries",
                "fragmentation_excess_per_object",
                "maximum_active_query_pair_dice",
            )
        },
        "label_shuffle_mean_object_dice": evaluation["label_shuffle"]["mean_object_dice"],
        "random_mean_object_dice": evaluation["random_initialization"]["mean_object_dice"],
    }
    _write_json_atomic(output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
