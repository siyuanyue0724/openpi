#!/usr/bin/env python3
"""Test Slot-Attention-style competitive query reads against baseline M2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from picf_next.data.calvin import (  # noqa: E402
    CalvinDatasetIndex,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (  # noqa: E402
    load_dataset_file_manifest,
    validate_dataset_files,
)
from picf_next.hosts.molmoact2_training import (  # noqa: E402
    CalvinVisibleObjectTargetBuilder,
)
from picf_next.models.discovery import (  # noqa: E402
    ObjectDiscoveryOutput,
    TaskIndependentObjectDiscovery,
)
from picf_next.models.set_loss import ObjectSetCriterion  # noqa: E402
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from tools.audit_molmoact2_m2_count_support import (  # noqa: E402
    _source_hashes,
    _train_paired,
)
from tools.audit_molmoact2_m2_external_validation import (  # noqa: E402
    _group_by_target_count,
    _unique_source_keys,
)
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _batch_plan,
    _evaluate,
    _keys_for_split,
    _load_cache,
    _render_visuals,
    _sha256,
    _write_json_atomic,
)

_EXTERNAL_SPLIT = "external_validation"
_GOOGLE_SLOT_REPOSITORY = "references/source_checkouts/google-slot-attention"
_GOOGLE_SLOT_FILE = "slot_attention/model.py"
_METASLOT_REPOSITORY = "references/source_checkouts/metaslot-neurips2025"
_METASLOT_FILE = "object_centric_bench/model/metaslot.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_representation.json",
    )
    parser.add_argument("--training-feature-cache", required=True, type=Path)
    parser.add_argument("--external-feature-cache", required=True, type=Path)
    parser.add_argument("--training-dataset-root", required=True, type=Path)
    parser.add_argument("--external-dataset-root", required=True, type=Path)
    parser.add_argument("--external-dataset-manifest", required=True, type=Path)
    parser.add_argument("--external-physical-sidecar-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=20)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _upstream_identity(repository: str, relative_file: str) -> dict[str, Any]:
    root = (_ROOT / repository).resolve()
    source = root / relative_file
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "repository": subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            text=True,
        ).strip(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "source_file": relative_file,
        "source_file_sha256": _sha256_bytes(source.read_bytes()),
    }


def _source_identity() -> dict[str, Any]:
    paths = (
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "tools/audit_molmoact2_m2_competitive_read.py",
        "tools/audit_molmoact2_m2_count_support.py",
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
            "google_slot_attention": _upstream_identity(
                _GOOGLE_SLOT_REPOSITORY,
                _GOOGLE_SLOT_FILE,
            ),
            "metaslot_neurips2025": _upstream_identity(
                _METASLOT_REPOSITORY,
                _METASLOT_FILE,
            ),
        },
        "copied_equation": (
            "softmax competition over slots followed by per-slot normalization "
            "over valid input tokens"
        ),
        "copied_code": False,
        "clean_authorizing_run": False,
    }


def _normalized_competitive_ownership(
    ownership: torch.Tensor,
    token_valid: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize competitive object ownership over valid tokens per query."""

    if ownership.ndim != 3:
        raise ValueError("ownership must be batch-by-token-by-category")
    if token_valid.dtype != torch.bool or token_valid.shape != ownership.shape[:2]:
        raise ValueError("token_valid must be bool batch-by-token")
    if ownership.shape[-1] < 2:
        raise ValueError("ownership must contain object queries plus context")
    if ownership.device != token_valid.device:
        raise ValueError("ownership and token_valid must share a device")
    if not torch.is_floating_point(ownership):
        raise TypeError("ownership must use a floating dtype")
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    weights = ownership[..., :-1].float()
    valid = token_valid.unsqueeze(-1)
    weights = torch.where(valid, weights + epsilon, torch.zeros_like(weights))
    denominator = weights.sum(dim=1, keepdim=True)
    return torch.where(
        denominator > 0.0,
        weights / denominator.clamp_min(epsilon),
        torch.zeros_like(weights),
    )


def _competitive_cross_read(
    layer: Any,
    memory: torch.Tensor,
    memory_valid: torch.Tensor,
    ownership: torch.Tensor,
) -> torch.Tensor:
    """Reuse the baseline MHA value/output maps with competitive slot weights."""

    attention = layer.cross_attention
    width = int(attention.embed_dim)
    if attention.in_proj_weight is None or attention.in_proj_weight.shape != (3 * width, width):
        raise ValueError("competitive read requires packed equal-width MHA projections")
    value_weight = attention.in_proj_weight[2 * width :]
    value_bias = None
    if attention.in_proj_bias is not None:
        value_bias = attention.in_proj_bias[2 * width :]
    values = F.linear(memory, value_weight, value_bias)
    weights = _normalized_competitive_ownership(ownership, memory_valid)
    update = torch.einsum(
        "bnk,bnh->bkh",
        weights.to(values.dtype),
        values,
    )
    return attention.out_proj(update)


def _competitive_layer_forward(
    layer: Any,
    queries: torch.Tensor,
    memory: torch.Tensor,
    memory_valid: torch.Tensor,
    ownership: torch.Tensor,
) -> torch.Tensor:
    """Apply one source-backed competitive read and unchanged set interaction."""

    if memory.shape[1] > 0:
        active = memory_valid.any(dim=1)
        update = _competitive_cross_read(layer, memory, memory_valid, ownership)
        queries = queries + layer.dropout(update * active[:, None, None])

    normalized = layer.self_norm(queries)
    update, _ = layer.self_attention(
        normalized,
        normalized,
        normalized,
        need_weights=False,
    )
    queries = queries + layer.dropout(update)
    return queries + layer.dropout(layer.ffn(layer.ffn_norm(queries)))


def _competitive_discovery_forward(
    self: TaskIndependentObjectDiscovery,
    binding_features: torch.Tensor,
    token_valid: torch.Tensor,
    token_group_id: torch.Tensor | None = None,
) -> ObjectDiscoveryOutput:
    """Run discovery with ownership competition inside every query update."""

    if token_group_id is None:
        token_group_id = torch.full_like(token_valid, -1, dtype=torch.long)
    self._validate(binding_features, token_valid, token_group_id)
    batch_size = binding_features.shape[0]
    memory = self.input_projection(self.input_norm(binding_features))
    memory = memory * token_valid.unsqueeze(-1)
    queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    predictions = [self._predict(queries, memory, token_valid, token_group_id)]
    for layer in self.layers:
        queries = _competitive_layer_forward(
            layer,
            queries,
            memory,
            token_valid,
            predictions[-1].ownership,
        )
        predictions.append(self._predict(queries, memory, token_valid, token_group_id))
    return replace(
        predictions[-1],
        auxiliary_outputs=tuple(predictions[:-1]),
    )


def _enable_competitive_read(current_frame_model: Any) -> None:
    discovery = current_frame_model.discovery
    if not isinstance(discovery, TaskIndependentObjectDiscovery):
        raise TypeError("competitive-read treatment requires PICF object discovery")
    if "forward" in discovery.__dict__:
        raise RuntimeError("discovery instance already overrides forward")
    before = tuple(
        (name, tuple(parameter.shape)) for name, parameter in discovery.named_parameters()
    )
    discovery.forward = MethodType(_competitive_discovery_forward, discovery)
    after = tuple(
        (name, tuple(parameter.shape)) for name, parameter in discovery.named_parameters()
    )
    if after != before:
        raise RuntimeError("competitive-read treatment changed trainable parameters")


def _metrics_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "sample_count",
            "mean_object_dice",
            "ownership_accuracy",
            "count_mae",
            "exact_count_accuracy",
            "geometry_mae_physical",
            "geometry_mae_physical_unit",
            "mean_active_queries",
            "maximum_active_query_pair_dice",
        )
    }


def _check_results(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, bool]:
    control_duplicate = float(control["maximum_active_query_pair_dice"])
    duplicate_limit = 0.8 * control_duplicate
    return {
        "external_count_mae_improves_at_least_25_percent": (
            float(treatment["count_mae"]) <= 0.75 * float(control["count_mae"])
        ),
        "external_exact_count_improves_at_least_0_10": (
            float(treatment["exact_count_accuracy"])
            >= float(control["exact_count_accuracy"]) + 0.10
        ),
        "external_dice_noninferior_within_0_03": (
            float(treatment["mean_object_dice"]) >= float(control["mean_object_dice"]) - 0.03
        ),
        "external_ownership_noninferior_within_0_03": (
            float(treatment["ownership_accuracy"]) >= float(control["ownership_accuracy"]) - 0.03
        ),
        "external_geometry_noninferior_within_10_percent": (
            float(treatment["geometry_mae_physical"])
            <= 1.10 * float(control["geometry_mae_physical"])
        ),
        "external_duplicate_pair_dice_reduces_at_least_20_percent": (
            float(treatment["maximum_active_query_pair_dice"]) <= duplicate_limit
        ),
    }


def main() -> None:
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets

    args = _parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("competitive-read paired audit requires two CUDA devices")
    if args.steps <= 0 or args.validation_interval <= 0:
        raise ValueError("steps and validation interval must be positive")
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
    training_root = args.training_dataset_root.expanduser().resolve()
    training_assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=training_root,
    )
    training_cache_root = args.training_feature_cache.expanduser().resolve()
    training_manifest, training_cache = _load_cache(training_cache_root, recipe)
    train_keys = _keys_for_split(training_cache, "train")
    validation_keys = _keys_for_split(training_cache, "validation")
    if len(train_keys) != 192 or len(validation_keys) != 64:
        raise RuntimeError("competitive-read source split size changed")
    plan = _batch_plan(train_keys, recipe)
    _write_json_atomic(
        output_dir / "batch_plan.json",
        {
            "schema": "picf-next.molmoact2-m2-competitive-read-plan.v1",
            "steps": recipe.optimization.steps,
            "batch_size": recipe.optimization.batch_size,
            "treatment_equals_control_at_every_slot": True,
            "plan": plan,
        },
    )

    external_root = args.external_dataset_root.expanduser().resolve()
    external_manifest_path = args.external_dataset_manifest.expanduser().resolve()
    external_dataset_manifest = load_dataset_file_manifest(external_manifest_path)
    validate_dataset_files(
        external_dataset_manifest,
        external_root,
        dataset_id=foundation.dataset.dataset_id,
        dataset_revision=foundation.dataset.dataset_revision,
        split_name=external_root.name,
        verify_hashes=True,
    )
    external_index = CalvinDatasetIndex.load(
        external_root,
        dataset_id=foundation.dataset.dataset_id,
        dataset_revision=foundation.dataset.dataset_revision,
        dataset_manifest=external_dataset_manifest,
    )
    external_dataset = CalvinStatefulTransitionDataset(
        external_index,
        action_horizon=foundation.dataset.action_horizon,
    )
    external_physical = CalvinPhysicalSupervisionSidecar(
        args.external_physical_sidecar_root.expanduser().resolve(),
        external_index,
        verify_hashes=True,
    )
    external_cache_root = args.external_feature_cache.expanduser().resolve()
    external_manifest, external_cache = _load_cache(external_cache_root, recipe)
    if external_manifest["processor_layout_sha256"] != training_manifest["processor_layout_sha256"]:
        raise RuntimeError("training and external dense-patch layouts differ")
    external_keys = _unique_source_keys(external_cache)
    if set(external_cache[key][2]["split"] for key in external_keys) != {_EXTERNAL_SPLIT}:
        raise RuntimeError("external competitive-read keys have an unexpected split")
    learned_hashes = _source_hashes(
        training_cache,
        train_keys + validation_keys,
    )
    external_hashes = _source_hashes(external_cache, external_keys)
    if learned_hashes & external_hashes:
        raise RuntimeError("competitive-read learned and external source frames overlap")

    _write_json_atomic(
        output_dir / "audit_manifest.json",
        {
            "schema": "picf-next.molmoact2-m2-competitive-read-audit.v1",
            "authorizes_later_gates": False,
            "source": _source_identity(),
            "recipe": recipe.to_dict(),
            "recipe_sha256": recipe.recipe_sha256,
            "mathematical_change": (
                "query cross-read uses supervised query-plus-context ownership "
                "competition followed by per-query valid-token normalization"
            ),
            "learned_parameters_added": 0,
            "training_feature_cache": str(training_cache_root),
            "training_feature_cache_manifest_sha256": _sha256(
                training_cache_root / "manifest.json"
            ),
            "external_feature_cache": str(external_cache_root),
            "external_feature_cache_manifest_sha256": _sha256(
                external_cache_root / "manifest.json"
            ),
            "training_sample_count": len(train_keys),
            "selection_validation_sample_count": len(validation_keys),
            "external_unique_source_count": len(external_keys),
            "learned_external_source_hash_intersection": 0,
        },
    )

    training_report, treatment, control = _train_paired(
        output_dir=output_dir,
        recipe=recipe,
        foundation=foundation,
        training_assets=training_assets,
        cache_manifest=training_manifest,
        cache=training_cache,
        treatment_plan=plan,
        control_plan=plan,
        validation_keys=validation_keys,
        treatment_setup=_enable_competitive_read,
        progress_event="competitive_read_validation",
        report_schema="picf-next.molmoact2-m2-competitive-read-training.v1",
        checkpoint_filenames=("competitive_read_treatment.pt", "baseline_control.pt"),
    )
    _write_json_atomic(output_dir / "training_report.json", training_report)

    treatment_device = torch.device("cuda:0")
    control_device = torch.device("cuda:1")
    treatment_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(treatment_device)
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(control_device)
    external_target_builder = CalvinVisibleObjectTargetBuilder(external_physical)
    treatment_external = _evaluate(
        model=treatment,
        cache=external_cache,
        keys=external_keys,
        target_builder=external_target_builder,
        criterion=treatment_criterion,
        layout_payload=external_manifest["processor_layout"],
        recipe=recipe,
        device=treatment_device,
        include_per_sample=True,
    )
    control_external = _evaluate(
        model=control,
        cache=external_cache,
        keys=external_keys,
        target_builder=external_target_builder,
        criterion=control_criterion,
        layout_payload=external_manifest["processor_layout"],
        recipe=recipe,
        device=control_device,
        include_per_sample=True,
    )
    checks = _check_results(treatment_external, control_external)

    external_assets = SimpleNamespace(
        dataset=external_dataset,
        physical_sidecar=external_physical,
    )
    treatment_dir = output_dir / "competitive_read_treatment"
    treatment_dir.mkdir()
    treatment_visuals = _render_visuals(
        run_dir=treatment_dir,
        model=treatment,
        assets=external_assets,
        cache=external_cache,
        cache_manifest=external_manifest,
        foundation=foundation,
        recipe=recipe,
        visual_splits=(_EXTERNAL_SPLIT,),
        expected_segments={segment.index for segment in external_index.segments},
    )
    _write_json_atomic(treatment_dir / "visual_artifacts.json", treatment_visuals)
    treatment.cpu()
    del treatment_criterion
    gc.collect()
    torch.cuda.empty_cache()
    control.to(treatment_device)
    control_dir = output_dir / "baseline_control"
    control_dir.mkdir()
    control_visuals = _render_visuals(
        run_dir=control_dir,
        model=control,
        assets=external_assets,
        cache=external_cache,
        cache_manifest=external_manifest,
        foundation=foundation,
        recipe=recipe,
        visual_splits=(_EXTERNAL_SPLIT,),
        expected_segments={segment.index for segment in external_index.segments},
    )
    _write_json_atomic(control_dir / "visual_artifacts.json", control_visuals)

    report = {
        "schema": "picf-next.molmoact2-m2-competitive-read-result.v1",
        "authorizes_later_gates": False,
        "structural_hypothesis_checks": checks,
        "structural_hypothesis_supported": all(checks.values()),
        "competitive_read_treatment_external": treatment_external,
        "baseline_control_external": control_external,
        "competitive_read_external_by_target_count": _group_by_target_count(
            treatment_external["per_sample"]
        ),
        "baseline_external_by_target_count": _group_by_target_count(control_external["per_sample"]),
        "treatment_visuals_sha256": _sha256(treatment_dir / "visual_artifacts.json"),
        "control_visuals_sha256": _sha256(control_dir / "visual_artifacts.json"),
    }
    _write_json_atomic(output_dir / "competitive_read_report.json", report)
    print(
        json.dumps(
            {
                "structural_hypothesis_checks": checks,
                "structural_hypothesis_supported": all(checks.values()),
                "competitive_read_treatment_external": _metrics_subset(treatment_external),
                "baseline_control_external": _metrics_subset(control_external),
                "competitive_read_external_by_target_count": report[
                    "competitive_read_external_by_target_count"
                ],
                "baseline_external_by_target_count": report["baseline_external_by_target_count"],
                "seconds_per_paired_step": training_report["seconds_per_paired_step"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
