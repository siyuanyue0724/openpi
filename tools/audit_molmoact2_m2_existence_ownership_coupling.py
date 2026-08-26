#!/usr/bin/env python3
"""Audit calibrated existence as a prior in competitive token ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
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

import torch  # noqa: E402

from picf_next.models.discovery import (  # noqa: E402
    ObjectExistenceCalibration,
    TaskIndependentObjectDiscovery,
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
        "--upstream-root",
        type=Path,
        default=_ROOT / "references/source_checkouts/icml2026-rethinking-ocl",
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_identity(upstream_root: Path) -> dict[str, Any]:
    upstream_root = upstream_root.expanduser().resolve()
    upstream_decoder = upstream_root / "slotcontrast/modules/decoders_mocsp.py"
    if not upstream_decoder.is_file():
        raise FileNotFoundError(upstream_decoder)
    paths = (
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "tools/audit_molmoact2_m2_existence_ownership_coupling.py",
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
        "upstream": {
            "repository": subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                cwd=upstream_root,
                text=True,
            ).strip(),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=upstream_root,
                text=True,
            ).strip(),
            "decoder_file": str(upstream_decoder),
            "decoder_sha256": _sha256_bytes(upstream_decoder.read_bytes()),
            "adapted_principle": (
                "apply object existence before normalization across competing slots"
            ),
            "copied_code": False,
        },
        "clean_authorizing_run": False,
    }


def _couple_existence_and_ownership(
    ownership_logits: torch.Tensor,
    existence_logits: torch.Tensor,
    calibration: ObjectExistenceCalibration,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return categorical ownership with calibrated existence as a log prior.

    For object category ``q`` and token ``i`` this computes

    ``P(z_i=q|x) ∝ exp(s_iq) * odds(P(e_q=1|x))``.

    Context keeps its original logit.  Therefore a neutral physical posterior
    of 0.5 is an exact no-op, while unsupported objects must compete with less
    prior mass.  The operation has no parameters and is permutation equivariant
    over object queries.
    """

    if ownership_logits.ndim != 3:
        raise ValueError("ownership logits must be batch-by-token-by-category")
    if existence_logits.ndim != 2:
        raise ValueError("existence logits must be batch-by-query")
    if ownership_logits.shape[0] != existence_logits.shape[0]:
        raise ValueError("ownership and existence batch sizes differ")
    if ownership_logits.shape[2] != existence_logits.shape[1] + 1:
        raise ValueError("ownership categories must equal queries plus context")
    if not isinstance(calibration, ObjectExistenceCalibration):
        raise TypeError("calibration must use ObjectExistenceCalibration")
    if not torch.is_floating_point(ownership_logits) or not torch.is_floating_point(
        existence_logits
    ):
        raise TypeError("ownership and existence logits must be floating tensors")

    compute_dtype = (
        torch.float32
        if ownership_logits.dtype in {torch.float16, torch.bfloat16}
        else ownership_logits.dtype
    )
    object_logits = ownership_logits[..., :-1].to(compute_dtype)
    context_logits = ownership_logits[..., -1:].to(compute_dtype)
    posterior_log_odds = calibration.posterior_logit(existence_logits).to(compute_dtype)
    coupled_logits = torch.cat(
        (object_logits + posterior_log_odds.unsqueeze(1), context_logits),
        dim=-1,
    )
    ownership = torch.softmax(coupled_logits, dim=-1)
    return coupled_logits.to(ownership_logits.dtype), ownership.to(ownership_logits.dtype)


@contextmanager
def _existence_ownership_coupling():
    original = TaskIndependentObjectDiscovery._predict

    def coupled_predict(
        self: TaskIndependentObjectDiscovery,
        queries: torch.Tensor,
        memory: torch.Tensor,
        token_valid: torch.Tensor,
        token_group_id: torch.Tensor,
    ):
        output = original(self, queries, memory, token_valid, token_group_id)
        logits, ownership = _couple_existence_and_ownership(
            output.ownership_logits,
            output.existence_logits,
            output.existence_calibration,
        )
        return replace(output, ownership_logits=logits, ownership=ownership)

    TaskIndependentObjectDiscovery._predict = coupled_predict
    try:
        yield
    finally:
        TaskIndependentObjectDiscovery._predict = original


def main() -> None:
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets

    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.validation_interval <= 0:
        raise ValueError("validation interval must be positive")
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
            "schema": "picf-next.molmoact2-m2-existence-ownership-coupling-audit.v1",
            "authorizes_later_gates": False,
            "source": _source_identity(args.upstream_root),
            "config": str(args.config.resolve()),
            "recipe": recipe.to_dict(),
            "recipe_sha256": recipe.recipe_sha256,
            "mathematical_change": (
                "object_ownership_logit += calibrated_physical_existence_log_odds"
            ),
            "learned_parameters_added": 0,
            "feature_cache": str(feature_cache),
            "feature_cache_manifest_sha256": _sha256_bytes(
                (feature_cache / "manifest.json").read_bytes()
            ),
            "dataset_split_root": str(args.dataset_split_root.expanduser().resolve()),
        },
    )

    with _existence_ownership_coupling():
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
