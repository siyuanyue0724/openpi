#!/usr/bin/env python3
"""Pair weighted-mean and constant-scaled competitive object reads.

This audit tests the cardinality-generalization treatment published in
Krimmel et al., TMLR 2024, against PICF's current weighted-mean read.  It does
not authorize later gates.  The optional source-recurrent arm tests the
normalization inside the complete Slot Attention update rather than
transplanting its aggregation equation into an unrelated residual decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
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
from torch import nn  # noqa: E402

from picf_next.data.calvin import CalvinDatasetIndex  # noqa: E402
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
from picf_next.models.discovery import TaskIndependentObjectDiscovery  # noqa: E402
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
_UPSTREAM_REPOSITORY = "references/source_checkouts/slot-attention-normalization"
_UPSTREAM_FILE = "sa_generalization/slot_attention/slot_attention.py"
_GOOGLE_UPSTREAM_REPOSITORY = "references/source_checkouts/google-slot-attention"
_GOOGLE_UPSTREAM_FILE = "slot_attention/model.py"
_RESIDUAL_SET_DECODER = "residual_set_decoder"
_SOURCE_RECURRENT = "source_recurrent"
_DISTINCT_LEARNED = "distinct_learned"
_SOURCE_GAUSSIAN = "source_gaussian_train_fixed_eval"
_ALL_STAGES = "all_including_no_evidence"
_POST_EVIDENCE_ONLY = "post_evidence_only"


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
    parser.add_argument(
        "--update-dynamics",
        choices=(_RESIDUAL_SET_DECODER, _SOURCE_RECURRENT),
        default=_RESIDUAL_SET_DECODER,
        help=(
            "Use the production residual decoder or the source-backed "
            "GRU/slot-wise-MLP update in both paired arms."
        ),
    )
    parser.add_argument(
        "--query-initialization",
        choices=(_DISTINCT_LEARNED, _SOURCE_GAUSSIAN),
        default=_DISTINCT_LEARNED,
        help=(
            "Keep learned indexed queries or use the upstream shared Gaussian "
            "slot prior with fresh train noise and one fixed evaluation draw."
        ),
    )
    parser.add_argument(
        "--supervision-stages",
        choices=(_ALL_STAGES, _POST_EVIDENCE_ONLY),
        default=_POST_EVIDENCE_ONLY,
        help=(
            "Supervise the legacy no-evidence prediction or only predictions "
            "formed after at least one competitive evidence update."
        ),
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _upstream_identity(repository: str, source_file: str) -> dict[str, Any]:
    upstream_root = (_ROOT / repository).resolve()
    upstream_path = upstream_root / source_file
    if not upstream_path.is_file():
        raise FileNotFoundError(upstream_path)
    return {
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
        "source_file": source_file,
        "source_file_sha256": _sha256_bytes(upstream_path.read_bytes()),
    }


def _source_identity(
    *,
    update_dynamics: str,
    query_initialization: str,
    supervision_stages: str,
) -> dict[str, Any]:
    upstream_root = (_ROOT / _UPSTREAM_REPOSITORY).resolve()
    upstream_file = upstream_root / _UPSTREAM_FILE
    if not upstream_file.is_file():
        raise FileNotFoundError(upstream_file)
    paths = (
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "tools/audit_molmoact2_m2_read_normalization.py",
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
            **_upstream_identity(_UPSTREAM_REPOSITORY, _UPSTREAM_FILE),
            "copied_equation": "constant update normalization: attention / input_count",
            "copied_code": False,
            "direct_runtime_reuse": (
                "GRUCellTF, value projection, pre-MLP normalization, and ReLU MLP"
                if update_dynamics == _SOURCE_RECURRENT
                else None
            ),
        },
        "update_dynamics": update_dynamics,
        "query_initialization": query_initialization,
        "supervision_stages": supervision_stages,
        "slot_attention_upstream": (
            {
                **_upstream_identity(
                    _GOOGLE_UPSTREAM_REPOSITORY,
                    _GOOGLE_UPSTREAM_FILE,
                ),
                "copied_structure": (
                    "shared GRUCell update followed by normalized slot-wise MLP; "
                    "no query self-attention"
                ),
                "copied_code": False,
                "direct_runtime_reuse": False,
            }
            if update_dynamics == _SOURCE_RECURRENT
            else None
        ),
        "clean_authorizing_run": False,
    }


class _SourceRecurrentSlotLayer(nn.Module):
    """Clean-room Slot Attention update adapted to PICF ownership tensors."""

    def __init__(self, source_layer: nn.Module, upstream_slot_attention: nn.Module) -> None:
        super().__init__()
        self.cross_read = source_layer.cross_read
        self.cross_read.value_projection = upstream_slot_attention.to_v
        # Slot Attention feeds the value aggregation directly to its GRU.
        self.cross_read.output_projection = nn.Identity()
        self.gru = upstream_slot_attention.gru
        self.ffn_norm = upstream_slot_attention.norm_pre_ff
        self.ffn = upstream_slot_attention.mlp
        self.dropout = source_layer.dropout

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        ownership: torch.Tensor,
    ) -> torch.Tensor:
        if memory.shape[1] == 0:
            return queries
        active = memory_valid.any(dim=1)
        update = self.cross_read(memory, memory_valid, ownership)
        recurrent = self.gru(
            update.reshape(-1, update.shape[-1]),
            queries.reshape(-1, queries.shape[-1]),
        ).reshape_as(queries)
        recurrent = recurrent + self.dropout(self.ffn(self.ffn_norm(recurrent)))
        return torch.where(active[:, None, None], recurrent, queries)


def _build_upstream_slot_attention(discovery: TaskIndependentObjectDiscovery) -> nn.Module:
    """Instantiate the audited MIT implementation for direct component reuse."""

    source_path = (_ROOT / _UPSTREAM_REPOSITORY / _UPSTREAM_FILE).resolve()
    spec = importlib.util.spec_from_file_location(
        "_picf_audit_slot_attention_normalization",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream Slot Attention source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SlotAttention(
        input_dim=discovery.config.hidden_dim,
        slot_dim=discovery.config.hidden_dim,
        common_dim=discovery.config.hidden_dim,
        num_slots=discovery.config.num_queries,
        iters=discovery.config.num_layers,
        hidden_dim=discovery.config.hidden_dim,
        update_normalization="mean",
        tf_gru=True,
    )


def _source_recurrent_discovery_forward(
    self: TaskIndependentObjectDiscovery,
    binding_features: torch.Tensor,
    token_valid: torch.Tensor,
    token_group_id: torch.Tensor | None = None,
) -> Any:
    if token_group_id is None:
        token_group_id = torch.full_like(token_valid, -1, dtype=torch.long)
    self._validate(binding_features, token_valid, token_group_id)
    batch_size = binding_features.shape[0]
    memory = self.input_projection(self.input_norm(binding_features))
    memory = memory * token_valid.unsqueeze(-1)
    if self.query_embeddings is not None:
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        if self.training:
            noise = torch.randn(
                batch_size,
                self.config.num_queries,
                self.config.hidden_dim,
                device=memory.device,
                dtype=self.slot_mu.dtype,
            )
        else:
            noise = self.slot_eval_noise.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.slot_mu + self.slot_logsigma.exp() * noise

    initial_prediction = self._predict(queries, memory, token_valid, token_group_id)
    prediction = initial_prediction
    post_evidence_predictions = []
    for layer in self.layers:
        queries = layer(
            queries,
            memory,
            token_valid,
            prediction.ownership,
        )
        prediction = self._predict(queries, memory, token_valid, token_group_id)
        post_evidence_predictions.append(prediction)

    if self._source_supervision_stages == _ALL_STAGES:
        auxiliary_outputs = (initial_prediction, *post_evidence_predictions[:-1])
    elif self._source_supervision_stages == _POST_EVIDENCE_ONLY:
        auxiliary_outputs = tuple(post_evidence_predictions[:-1])
    else:
        raise RuntimeError(
            f"unsupported source supervision stages: {self._source_supervision_stages}"
        )
    return replace(
        post_evidence_predictions[-1],
        auxiliary_outputs=auxiliary_outputs,
    )


def _enable_source_recurrent_update(
    current_frame_model: Any,
    *,
    query_initialization: str = _DISTINCT_LEARNED,
    supervision_stages: str = _POST_EVIDENCE_ONLY,
) -> None:
    """Install the complete source-backed recurrent slot update in both arms."""

    discovery = current_frame_model.discovery
    if not isinstance(discovery, TaskIndependentObjectDiscovery):
        raise TypeError("source-recurrent treatment requires PICF object discovery")
    if not discovery.layers:
        raise ValueError("source-recurrent treatment requires at least one refinement iteration")
    if supervision_stages not in {_ALL_STAGES, _POST_EVIDENCE_ONLY}:
        raise ValueError(f"unsupported source supervision stages: {supervision_stages}")
    iteration_count = len(discovery.layers)
    upstream_slot_attention = _build_upstream_slot_attention(discovery)
    shared_layer = _SourceRecurrentSlotLayer(
        discovery.layers[0],
        upstream_slot_attention,
    )
    # Official Slot Attention reuses one q/k/v, GRU and MLP update at every
    # refinement iteration. ModuleList retains the host forward contract while
    # all entries intentionally reference the same recurrent operator.
    discovery.layers = nn.ModuleList([shared_layer] * iteration_count)
    if query_initialization == _SOURCE_GAUSSIAN:
        discovery.query_embeddings = None
        discovery.slot_mu = upstream_slot_attention.slots_mu
        discovery.slot_logsigma = upstream_slot_attention.slots_logsigma
        discovery.register_buffer(
            "slot_eval_noise",
            torch.randn(
                discovery.config.num_queries,
                discovery.config.hidden_dim,
                dtype=discovery.slot_mu.dtype,
            ),
        )
    elif query_initialization != _DISTINCT_LEARNED:
        raise ValueError(f"unsupported query initialization: {query_initialization}")
    discovery._source_supervision_stages = supervision_stages
    discovery.forward = MethodType(_source_recurrent_discovery_forward, discovery)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _constant_scaled_competitive_ownership(
    ownership: torch.Tensor,
    token_valid: torch.Tensor,
) -> torch.Tensor:
    """Preserve competitive assignment mass using the upstream constant rule."""

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

    valid = token_valid.unsqueeze(-1)
    weights = torch.where(
        valid,
        ownership[..., :-1].float(),
        torch.zeros_like(ownership[..., :-1], dtype=torch.float32),
    )
    input_count = token_valid.sum(dim=1, keepdim=True).unsqueeze(-1)
    return torch.where(
        input_count > 0,
        weights / input_count.clamp_min(1),
        torch.zeros_like(weights),
    )


def _constant_scaled_cross_read(
    self: Any,
    memory: torch.Tensor,
    memory_valid: torch.Tensor,
    ownership: torch.Tensor,
) -> torch.Tensor:
    weights = _constant_scaled_competitive_ownership(ownership, memory_valid)
    values = self.value_projection(memory)
    update = torch.einsum(
        "bnk,bnh->bkh",
        weights.to(values.dtype),
        values,
    )
    return self.output_projection(update)


def _enable_constant_scaled_read(current_frame_model: Any) -> None:
    discovery = current_frame_model.discovery
    if not isinstance(discovery, TaskIndependentObjectDiscovery):
        raise TypeError("constant-scaled treatment requires PICF object discovery")
    before = tuple(
        (name, tuple(parameter.shape)) for name, parameter in discovery.named_parameters()
    )
    seen: set[int] = set()
    for layer in discovery.layers:
        cross_read = layer.cross_read
        if id(cross_read) in seen:
            continue
        seen.add(id(cross_read))
        if "forward" in cross_read.__dict__:
            raise RuntimeError("competitive cross-read instance already overrides forward")
        cross_read.forward = MethodType(_constant_scaled_cross_read, cross_read)
    after = tuple(
        (name, tuple(parameter.shape)) for name, parameter in discovery.named_parameters()
    )
    if after != before:
        raise RuntimeError("constant-scaled treatment changed trainable parameters")


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


def _noninferior(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    exact_margin: float,
) -> dict[str, bool]:
    return {
        "dice_noninferior_within_0_03": (
            float(treatment["mean_object_dice"]) >= float(control["mean_object_dice"]) - 0.03
        ),
        "ownership_noninferior_within_0_03": (
            float(treatment["ownership_accuracy"]) >= float(control["ownership_accuracy"]) - 0.03
        ),
        "geometry_noninferior_within_10_percent": (
            float(treatment["geometry_mae_physical"])
            <= 1.10 * float(control["geometry_mae_physical"])
        ),
        "exact_count_noninferior": (
            float(treatment["exact_count_accuracy"])
            >= float(control["exact_count_accuracy"]) - exact_margin
        ),
        "duplicate_pair_dice_noninferior_within_0_05": (
            float(treatment["maximum_active_query_pair_dice"])
            <= float(control["maximum_active_query_pair_dice"]) + 0.05
        ),
    }


def _cardinality_response(
    grouped_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_counts = (7, 8, 9, 10)
    if tuple(sorted(int(value) for value in grouped_metrics)) != expected_counts:
        raise RuntimeError("cardinality audit requires target-count groups 7, 8, 9, and 10")
    predictions = [
        float(grouped_metrics[str(target)]["predicted_count_mean"]) for target in expected_counts
    ]
    target_mean = sum(expected_counts) / len(expected_counts)
    prediction_mean = sum(predictions) / len(predictions)
    numerator = sum(
        (target - target_mean) * (prediction - prediction_mean)
        for target, prediction in zip(expected_counts, predictions, strict=True)
    )
    denominator = sum((target - target_mean) ** 2 for target in expected_counts)
    return {
        "target_counts": list(expected_counts),
        "predicted_count_means": predictions,
        "least_squares_slope": numerator / denominator,
        "predicted_mean_range": max(predictions) - min(predictions),
        "maximum_absolute_group_bias": max(
            abs(prediction - target)
            for target, prediction in zip(expected_counts, predictions, strict=True)
        ),
        "minimum_group_exact_count_accuracy": min(
            float(grouped_metrics[str(target)]["exact_count_accuracy"])
            for target in expected_counts
        ),
    }


def main() -> None:
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets

    args = _parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("paired read-normalization audit requires two CUDA devices")
    if args.steps <= 0 or args.validation_interval <= 0:
        raise ValueError("steps and validation interval must be positive")
    if args.query_initialization == _SOURCE_GAUSSIAN and args.update_dynamics != _SOURCE_RECURRENT:
        raise ValueError("source Gaussian queries require source-recurrent update dynamics")
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
    architecture_probe = foundation.core_config.build_current_frame()
    production_parameter_count = _parameter_count(architecture_probe)
    if args.update_dynamics == _SOURCE_RECURRENT:
        _enable_source_recurrent_update(
            architecture_probe,
            query_initialization=args.query_initialization,
            supervision_stages=args.supervision_stages,
        )
    audited_parameter_count = _parameter_count(architecture_probe)
    del architecture_probe
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
    heldout_keys = _keys_for_split(training_cache, "heldout")
    if (len(train_keys), len(validation_keys), len(heldout_keys)) != (192, 64, 110):
        raise RuntimeError("read-normalization source split sizes changed")
    plan = _batch_plan(train_keys, recipe)
    _write_json_atomic(
        output_dir / "batch_plan.json",
        {
            "schema": "picf-next.molmoact2-m2-read-normalization-plan.v1",
            "steps": recipe.optimization.steps,
            "batch_size": recipe.optimization.batch_size,
            "treatment_equals_control_at_every_slot": True,
            "plan": plan,
        },
    )

    external_root = args.external_dataset_root.expanduser().resolve()
    external_dataset_manifest = load_dataset_file_manifest(
        args.external_dataset_manifest.expanduser().resolve()
    )
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
        raise RuntimeError("external read-normalization keys have an unexpected split")
    learned_hashes = _source_hashes(
        training_cache,
        train_keys + validation_keys + heldout_keys,
    )
    external_hashes = _source_hashes(external_cache, external_keys)
    if learned_hashes & external_hashes:
        raise RuntimeError("read-normalization formal and external source frames overlap")

    _write_json_atomic(
        output_dir / "audit_manifest.json",
        {
            "schema": "picf-next.molmoact2-m2-read-normalization-audit.v1",
            "authorizes_later_gates": False,
            "source": _source_identity(
                update_dynamics=args.update_dynamics,
                query_initialization=args.query_initialization,
                supervision_stages=args.supervision_stages,
            ),
            "recipe": recipe.to_dict(),
            "recipe_sha256": recipe.recipe_sha256,
            "mathematical_change": {
                "paired_difference": (
                    "replace per-query weighted mean by source-backed constant-scaled "
                    "weighted sum so query updates retain competitive assignment mass"
                ),
                "common_update_dynamics": (
                    "production residual query self-attention decoder"
                    if args.update_dynamics == _RESIDUAL_SET_DECODER
                    else (
                        "source-backed shared GRU update followed by normalized "
                        "slot-wise MLP, with no query self-attention"
                    )
                ),
                "common_query_initialization": args.query_initialization,
                "common_supervision_stages": args.supervision_stages,
            },
            "parameter_counts": {
                "production_current_frame": production_parameter_count,
                "audited_current_frame_per_arm": audited_parameter_count,
                "difference": audited_parameter_count - production_parameter_count,
                "paired_arms_equal": True,
            },
            "training_feature_cache": str(training_cache_root),
            "training_feature_cache_manifest_sha256": _sha256(
                training_cache_root / "manifest.json"
            ),
            "external_feature_cache": str(external_cache_root),
            "external_feature_cache_manifest_sha256": _sha256(
                external_cache_root / "manifest.json"
            ),
            "formal_sample_counts": {
                "train": len(train_keys),
                "validation": len(validation_keys),
                "heldout": len(heldout_keys),
            },
            "external_unique_source_count": len(external_keys),
            "formal_external_source_hash_intersection": 0,
            "preregistered_checks": {
                "formal_exact_count_gate": (recipe.acceptance.minimum_heldout_exact_count_accuracy),
                "formal_exact_improvement": 0.10,
                "formal_count_mae_improvement_fraction": 0.25,
                "representation_noninferiority_margin": 0.03,
                "external_exact_noninferiority_margin": 0.05,
                "production_reference": {
                    "formal_minimum_dice": 0.602,
                    "formal_minimum_ownership": 0.805,
                    "formal_maximum_geometry_mae_m": 0.047,
                    "external_minimum_exact_count": 0.65,
                    "external_maximum_count_mae": 0.55,
                    "external_minimum_dice": 0.564,
                    "external_minimum_ownership": 0.741,
                    "external_maximum_geometry_mae_m": 0.058,
                    "external_maximum_duplicate_pair_dice": 0.567,
                },
                "variable_cardinality": {
                    "minimum_response_slope": 0.5,
                    "minimum_predicted_mean_range": 1.5,
                    "maximum_absolute_group_bias": 0.75,
                    "minimum_per_group_exact_count_accuracy": 0.15,
                    "target_count_groups": [7, 8, 9, 10],
                },
            },
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
        common_setup=(
            partial(
                _enable_source_recurrent_update,
                query_initialization=args.query_initialization,
                supervision_stages=args.supervision_stages,
            )
            if args.update_dynamics == _SOURCE_RECURRENT
            else None
        ),
        treatment_setup=_enable_constant_scaled_read,
        progress_event=(
            f"read_normalization_{args.update_dynamics}_"
            f"{args.query_initialization}_{args.supervision_stages}_validation"
        ),
        report_schema=(
            f"picf-next.molmoact2-m2-read-normalization-"
            f"{args.update_dynamics}-{args.query_initialization}-"
            f"{args.supervision_stages}-training.v2"
        ),
        checkpoint_filenames=(
            (
                f"constant_scaled_{args.update_dynamics}_"
                f"{args.query_initialization}_{args.supervision_stages}_treatment.pt"
            ),
            (
                f"weighted_mean_{args.update_dynamics}_"
                f"{args.query_initialization}_{args.supervision_stages}_control.pt"
            ),
        ),
    )
    _write_json_atomic(output_dir / "training_report.json", training_report)

    treatment_device = torch.device("cuda:0")
    control_device = torch.device("cuda:1")
    treatment_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(treatment_device)
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(control_device)
    training_target_builder = CalvinVisibleObjectTargetBuilder(training_assets.physical_sidecar)
    external_target_builder = CalvinVisibleObjectTargetBuilder(external_physical)
    treatment_heldout = _evaluate(
        model=treatment,
        cache=training_cache,
        keys=heldout_keys,
        target_builder=training_target_builder,
        criterion=treatment_criterion,
        layout_payload=training_manifest["processor_layout"],
        recipe=recipe,
        device=treatment_device,
        include_per_sample=True,
    )
    control_heldout = _evaluate(
        model=control,
        cache=training_cache,
        keys=heldout_keys,
        target_builder=training_target_builder,
        criterion=control_criterion,
        layout_payload=training_manifest["processor_layout"],
        recipe=recipe,
        device=control_device,
        include_per_sample=True,
    )
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

    treatment_formal_groups = _group_by_target_count(treatment_heldout["per_sample"])
    control_formal_groups = _group_by_target_count(control_heldout["per_sample"])
    treatment_external_groups = _group_by_target_count(treatment_external["per_sample"])
    control_external_groups = _group_by_target_count(control_external["per_sample"])
    treatment_external_cardinality = _cardinality_response(treatment_external_groups)
    control_external_cardinality = _cardinality_response(control_external_groups)

    checks = {
        "formal_exact_count_reaches_gate": (
            float(treatment_heldout["exact_count_accuracy"])
            >= recipe.acceptance.minimum_heldout_exact_count_accuracy
        ),
        "formal_exact_count_improves_at_least_0_10": (
            float(treatment_heldout["exact_count_accuracy"])
            >= float(control_heldout["exact_count_accuracy"]) + 0.10
        ),
        "formal_count_mae_improves_at_least_25_percent": (
            float(treatment_heldout["count_mae"]) <= 0.75 * float(control_heldout["count_mae"])
        ),
        "formal_dice_noninferior_to_production": (
            float(treatment_heldout["mean_object_dice"]) >= 0.602
        ),
        "formal_ownership_noninferior_to_production": (
            float(treatment_heldout["ownership_accuracy"]) >= 0.805
        ),
        "formal_geometry_noninferior_to_production": (
            float(treatment_heldout["geometry_mae_physical"]) <= 0.047
        ),
        "external_exact_count_noninferior_to_production": (
            float(treatment_external["exact_count_accuracy"]) >= 0.65
        ),
        "external_count_mae_noninferior_to_production": (
            float(treatment_external["count_mae"]) <= 0.55
        ),
        "external_dice_noninferior_to_production": (
            float(treatment_external["mean_object_dice"]) >= 0.564
        ),
        "external_ownership_noninferior_to_production": (
            float(treatment_external["ownership_accuracy"]) >= 0.741
        ),
        "external_geometry_noninferior_to_production": (
            float(treatment_external["geometry_mae_physical"]) <= 0.058
        ),
        "external_duplicate_pair_dice_noninferior_to_production": (
            float(treatment_external["maximum_active_query_pair_dice"]) <= 0.567
        ),
        "external_count_response_slope": (
            float(treatment_external_cardinality["least_squares_slope"]) >= 0.5
        ),
        "external_count_response_range": (
            float(treatment_external_cardinality["predicted_mean_range"]) >= 1.5
        ),
        "external_maximum_group_count_bias": (
            float(treatment_external_cardinality["maximum_absolute_group_bias"]) <= 0.75
        ),
        "external_minimum_per_group_exact_count": (
            float(treatment_external_cardinality["minimum_group_exact_count_accuracy"]) >= 0.15
        ),
    }
    checks.update(
        {
            f"formal_{name}": passed
            for name, passed in _noninferior(
                treatment_heldout,
                control_heldout,
                exact_margin=0.0,
            ).items()
        }
    )
    checks.update(
        {
            f"external_{name}": passed
            for name, passed in _noninferior(
                treatment_external,
                control_external,
                exact_margin=0.05,
            ).items()
        }
    )
    checks["external_count_mae_noninferior_within_10_percent"] = float(
        treatment_external["count_mae"]
    ) <= 1.10 * float(control_external["count_mae"])

    treatment_visual_dir = output_dir / (
        f"constant_scaled_{args.update_dynamics}_{args.query_initialization}_"
        f"{args.supervision_stages}_treatment"
    )
    treatment_visual_dir.mkdir()
    treatment_visuals = _render_visuals(
        run_dir=treatment_visual_dir,
        model=treatment,
        assets=SimpleNamespace(
            dataset=training_assets.dataset,
            physical_sidecar=training_assets.physical_sidecar,
        ),
        cache=training_cache,
        cache_manifest=training_manifest,
        foundation=foundation,
        recipe=recipe,
        visual_splits=("heldout",),
        expected_segments=set(recipe.splits.heldout_segments),
    )
    _write_json_atomic(
        treatment_visual_dir / "visual_artifacts.json",
        treatment_visuals,
    )

    report = {
        "schema": "picf-next.molmoact2-m2-read-normalization-result.v2",
        "authorizes_later_gates": False,
        "update_dynamics": args.update_dynamics,
        "query_initialization": args.query_initialization,
        "supervision_stages": args.supervision_stages,
        "structural_hypothesis_checks": checks,
        "structural_hypothesis_supported": all(checks.values()),
        "formal_heldout": {
            "constant_scaled_treatment": treatment_heldout,
            "weighted_mean_control": control_heldout,
        },
        "external_unique_source": {
            "constant_scaled_treatment": treatment_external,
            "weighted_mean_control": control_external,
        },
        "formal_treatment_by_target_count": treatment_formal_groups,
        "formal_control_by_target_count": control_formal_groups,
        "external_treatment_by_target_count": treatment_external_groups,
        "external_control_by_target_count": control_external_groups,
        "external_cardinality_response": {
            "constant_scaled_treatment": treatment_external_cardinality,
            "weighted_mean_control": control_external_cardinality,
        },
        "treatment_visuals_sha256": _sha256(treatment_visual_dir / "visual_artifacts.json"),
    }
    _write_json_atomic(output_dir / "read_normalization_report.json", report)
    print(
        json.dumps(
            {
                "structural_hypothesis_checks": checks,
                "structural_hypothesis_supported": all(checks.values()),
                "formal_heldout": {
                    "constant_scaled_treatment": _metrics_subset(treatment_heldout),
                    "weighted_mean_control": _metrics_subset(control_heldout),
                },
                "external_unique_source": {
                    "constant_scaled_treatment": _metrics_subset(treatment_external),
                    "weighted_mean_control": _metrics_subset(control_external),
                },
                "seconds_per_paired_step": training_report["seconds_per_paired_step"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
