#!/usr/bin/env python3
"""Evaluate an unchanged M2 checkpoint on source-disjoint CALVIN validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.data.calvin import (  # noqa: E402
    CalvinDatasetIndex,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.calvin_normalization import (  # noqa: E402
    load_calvin_normalization_artifact,
    official_molmoact2_dataset_stats,
)
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (  # noqa: E402
    load_dataset_file_manifest,
    validate_dataset_files,
)
from picf_next.eval.m2_protocol import (  # noqa: E402
    group_by_target_count as _group_by_target_count,
)
from picf_next.eval.m2_protocol import (  # noqa: E402
    language_samples as _language_samples,
)
from picf_next.eval.m2_protocol import (  # noqa: E402
    task_intervention_check as _task_intervention_check,
)
from picf_next.eval.m2_protocol import (  # noqa: E402
    unique_source_keys as _unique_source_keys,
)
from picf_next.hosts.molmoact2_training import (  # noqa: E402
    CalvinVisibleObjectTargetBuilder,
    calvin_visible_object_target_request,
    molmoact2_host_observation_view,
)
from picf_next.models.set_loss import ObjectSetCriterion  # noqa: E402
from picf_next.training.molmoact2_calvin import (  # noqa: E402
    build_molmoact2_policy_config,
)
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _all_context_baseline,
    _canonical_sha256,
    _emit_progress,
    _evaluate,
    _layout_row_payload,
    _load_cache,
    _move_inputs,
    _regular_cpu_copy,
    _render_visuals,
    _sha256,
    _write_json_atomic,
    _write_torch_atomic,
)

_EXTERNAL_SPLIT = "external_validation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_representation.json",
    )
    parser.add_argument("--dataset-split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--physical-sidecar-root", required=True, type=Path)
    parser.add_argument("--training-normalization", required=True, type=Path)
    parser.add_argument("--foundation-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--m2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _source_identity() -> dict[str, Any]:
    paths = (
        "src/picf_next/data/calvin_normalization.py",
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "tools/audit_molmoact2_m2_external_validation.py",
        "tools/run_molmoact2_m2_cloud.py",
    )
    return {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=_ROOT)
        ).hexdigest(),
        "audited_file_sha256": {
            relative: hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
            for relative in paths
        },
    }


def _extract_cache(
    *,
    output_dir: Path,
    recipe: Any,
    foundation: Any,
    assets: Any,
    samples: list[Any],
    checkpoint_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation
    from picf_next.hosts.molmoact2_calvin_processor import CalvinMolmoAct2ProcessorBridge

    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir()
    policy_config = build_molmoact2_policy_config(
        foundation,
        checkpoint_path=checkpoint_dir,
    )
    device = torch.device("cuda:0")
    policy = MolmoAct2Policy(policy_config).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    processor = CalvinMolmoAct2ProcessorBridge.from_official_config(
        policy.config,
        dataset_stats=official_molmoact2_dataset_stats(assets.normalization_payload),
    )

    records: list[dict[str, Any]] = []
    canonical_layout: dict[str, Any] | None = None
    pending_tokens = []
    pending_valid = []
    pending_records = []
    shards: list[dict[str, Any]] = []
    shard_index = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def flush() -> None:
        nonlocal shard_index
        if not pending_tokens:
            return
        path = cache_dir / f"features-{shard_index:05d}.pt"
        tokens = torch.cat(pending_tokens, dim=0).contiguous()
        valid = torch.cat(pending_valid, dim=0).contiguous()
        _write_torch_atomic(path, {"tokens": tokens, "valid": valid})
        for row_index, record in enumerate(pending_records):
            record["shard"] = path.name
            record["row"] = row_index
            records.append(record)
        shards.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "rows": len(pending_records),
                "bytes": path.stat().st_size,
            }
        )
        _emit_progress(
            "external_feature_cache_shard",
            shard=path.name,
            shard_rows=len(pending_records),
            completed_rows=len(records),
            total_rows=len(samples),
        )
        pending_tokens.clear()
        pending_valid.clear()
        pending_records.clear()
        shard_index += 1

    for start in range(0, len(samples), recipe.cache.extraction_batch_size):
        batch = samples[start : start + recipe.cache.extraction_batch_size]
        evidence = tuple((sample.picf_evidence_frame,) for sample in batch)
        views = tuple(molmoact2_host_observation_view(sample.record) for sample in batch)
        observation_inputs = _move_inputs(
            processor.build_observation_inputs(evidence, views),
            device,
        )
        with torch.inference_mode():
            prepared = prepare_molmoact2_lerobot_observation(policy, observation_inputs)
        bank = prepared.vision_patch_bank
        layout = prepared.vision_patch_layout
        if bank is None or layout is None:
            raise RuntimeError("external validation produced no dense Molmo patch bank")
        if (
            bank.modality != recipe.cache.modality
            or bank.tokens.shape[1:] != (recipe.cache.token_count, recipe.cache.token_dim)
            or bank.valid.shape != bank.tokens.shape[:2]
        ):
            raise RuntimeError("external validation Molmo feature contract changed")
        for row in layout.rows:
            payload = _layout_row_payload(row)
            if canonical_layout is None:
                canonical_layout = payload
            elif payload != canonical_layout:
                raise RuntimeError("external validation patch layout changed across samples")
        cpu_tokens = _regular_cpu_copy(bank.tokens, dtype=torch.bfloat16)
        cpu_valid = _regular_cpu_copy(bank.valid)
        for batch_index, sample in enumerate(batch):
            request = calvin_visible_object_target_request(sample)
            pending_tokens.append(cpu_tokens[batch_index : batch_index + 1])
            pending_valid.append(cpu_valid[batch_index : batch_index + 1])
            pending_records.append(
                {
                    "sample_key": sample.sample_key,
                    "split": _EXTERNAL_SPLIT,
                    "segment_index": sample.record.task_index,
                    "transition_index": sample.transition_index,
                    "global_index": sample.record.global_index,
                    "task_key": sample.host_sample.task_key,
                    "instruction": sample.record.task,
                    "source_sensor_sha256": [list(item) for item in request.source_sensor_sha256],
                }
            )
            if len(pending_records) == recipe.cache.shard_rows:
                flush()
        del prepared, bank, observation_inputs
    flush()
    if canonical_layout is None or len(records) != len(samples):
        raise RuntimeError("external validation feature extraction is incomplete")

    manifest = {
        "schema": "picf-next.molmoact2-m2-feature-cache.v1",
        "gate": "M2_external_validation_audit",
        "checkpoint_id": foundation.host.checkpoint_id,
        "checkpoint_revision": foundation.host.checkpoint_revision,
        "foundation_recipe_sha256": foundation.recipe_sha256,
        "modality": recipe.cache.modality,
        "dtype": recipe.cache.dtype,
        "token_shape": [recipe.cache.token_count, recipe.cache.token_dim],
        "processor_layout": canonical_layout,
        "processor_layout_sha256": _canonical_sha256(canonical_layout),
        "records": records,
        "records_sha256": _canonical_sha256(records),
        "shards": shards,
        "sample_count": len(records),
        "unique_source_frame_count": len({record["global_index"] for record in records}),
        "model_input_fields": ["tokens", "valid"],
        "loss_target_fields_in_feature_shards": [],
        "normalization_role": "training-fit statistics reused read-only for validation",
        "provenance": provenance,
        "elapsed_s": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _write_json_atomic(cache_dir / "manifest.json", manifest)
    _emit_progress(
        "external_feature_cache_complete",
        sample_count=len(records),
        unique_source_frame_count=manifest["unique_source_frame_count"],
        elapsed_s=manifest["elapsed_s"],
    )
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def main() -> None:
    import torch

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("external M2 validation requires CUDA")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    recipe = load_molmoact2_m2_recipe(args.config.resolve())
    foundation = recipe.load_foundation(_ROOT)
    split_root = args.dataset_split_root.expanduser().resolve()
    if split_root.name == foundation.dataset.split_name:
        raise ValueError("external validation must not reuse the training split")
    dataset_manifest_path = args.dataset_manifest.expanduser().resolve()
    dataset_manifest = load_dataset_file_manifest(dataset_manifest_path)
    validate_dataset_files(
        dataset_manifest,
        split_root,
        dataset_id=foundation.dataset.dataset_id,
        dataset_revision=foundation.dataset.dataset_revision,
        split_name=split_root.name,
        verify_hashes=True,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=foundation.dataset.dataset_id,
        dataset_revision=foundation.dataset.dataset_revision,
        dataset_manifest=dataset_manifest,
    )
    dataset = CalvinStatefulTransitionDataset(
        index,
        action_horizon=foundation.dataset.action_horizon,
    )
    physical_root = args.physical_sidecar_root.expanduser().resolve()
    physical = CalvinPhysicalSupervisionSidecar(physical_root, index, verify_hashes=True)
    normalization_path = args.training_normalization.expanduser().resolve()
    if _sha256(normalization_path) != foundation.artifacts.normalization_file_sha256:
        raise ValueError("external validation normalization is not the pinned training artifact")
    normalization = load_calvin_normalization_artifact(normalization_path)
    if (
        normalization["dataset_id"] != foundation.dataset.dataset_id
        or normalization["dataset_revision"] != foundation.dataset.dataset_revision
    ):
        raise ValueError("training normalization dataset identity changed")
    assets = SimpleNamespace(
        index=index,
        dataset=dataset,
        normalization_payload=normalization,
        physical_sidecar=physical,
    )
    samples = _language_samples(dataset)
    provenance = {
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "physical_sidecar_manifest": str(physical_root / "manifest.json"),
        "physical_sidecar_manifest_sha256": _sha256(physical_root / "manifest.json"),
        "training_normalization": str(normalization_path),
        "training_normalization_sha256": _sha256(normalization_path),
        "validation_statistics_fitted": False,
    }
    cache_manifest = _extract_cache(
        output_dir=output_dir,
        recipe=recipe,
        foundation=foundation,
        assets=assets,
        samples=samples,
        checkpoint_dir=args.foundation_checkpoint_dir.expanduser().resolve(),
        provenance=provenance,
    )
    cache_manifest, cache = _load_cache(output_dir / "feature_cache", recipe)
    task_intervention = _task_intervention_check(cache)
    _write_json_atomic(output_dir / "task_intervention.json", task_intervention)

    all_keys = sorted(
        cache,
        key=lambda key: (
            int(cache[key][2]["segment_index"]),
            int(cache[key][2]["transition_index"]),
        ),
    )
    unique_keys = _unique_source_keys(cache)
    device = torch.device("cuda:0")
    torch.manual_seed(recipe.optimization.seed)
    model = foundation.core_config.build_current_frame().to(device)
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(device)
    target_builder = CalvinVisibleObjectTargetBuilder(physical)
    random_unique = _evaluate(
        model=model,
        cache=cache,
        keys=unique_keys,
        target_builder=target_builder,
        criterion=criterion,
        layout_payload=cache_manifest["processor_layout"],
        recipe=recipe,
        device=device,
    )
    all_context = _all_context_baseline(
        cache=cache,
        keys=unique_keys,
        target_builder=target_builder,
        layout_payload=cache_manifest["processor_layout"],
        recipe=recipe,
        device=device,
    )

    checkpoint = args.m2_checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"] if set(payload) == {"model"} else payload
    model.load_state_dict(state, strict=True)
    actual_unique = _evaluate(
        model=model,
        cache=cache,
        keys=unique_keys,
        target_builder=target_builder,
        criterion=criterion,
        layout_payload=cache_manifest["processor_layout"],
        recipe=recipe,
        device=device,
        include_per_sample=True,
    )
    actual_language_weighted = _evaluate(
        model=model,
        cache=cache,
        keys=all_keys,
        target_builder=target_builder,
        criterion=criterion,
        layout_payload=cache_manifest["processor_layout"],
        recipe=recipe,
        device=device,
    )
    count_strata = _group_by_target_count(actual_unique["per_sample"])
    visual_manifest = _render_visuals(
        run_dir=output_dir,
        model=model,
        assets=assets,
        cache=cache,
        cache_manifest=cache_manifest,
        foundation=foundation,
        recipe=recipe,
        visual_splits=(_EXTERNAL_SPLIT,),
        expected_segments={segment.index for segment in index.segments},
    )
    _write_json_atomic(output_dir / "visual_artifacts.json", visual_manifest)

    report = {
        "schema": "picf-next.molmoact2-m2-external-validation-audit.v1",
        "authorizes_later_gates": False,
        "source": _source_identity(),
        "config": str(args.config.resolve()),
        "recipe_sha256": recipe.recipe_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "dataset_split": split_root.name,
        "language_sample_count": len(all_keys),
        "unique_source_frame_count": len(unique_keys),
        "task_intervention": task_intervention,
        "random_initialization_unique_source": random_unique,
        "all_context_unique_source": all_context,
        "actual_unique_source": actual_unique,
        "actual_language_weighted": actual_language_weighted,
        "actual_unique_source_by_target_count": count_strata,
        "visual_artifacts_sha256": _sha256(output_dir / "visual_artifacts.json"),
    }
    _write_json_atomic(output_dir / "external_validation_report.json", report)
    print(
        json.dumps(
            {
                "unique_source": {
                    name: actual_unique[name]
                    for name in (
                        "sample_count",
                        "mean_object_dice",
                        "balanced_ownership_accuracy",
                        "count_mae",
                        "exact_count_accuracy",
                        "geometry_mae_physical",
                    )
                },
                "by_target_count": count_strata,
                "task_intervention": task_intervention,
                "visual_count": len(visual_manifest["artifacts"]),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
