#!/usr/bin/env python3
"""Test real low-count data support against an equal-size high-count control."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

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
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (  # noqa: E402
    load_dataset_file_manifest,
    validate_dataset_files,
)
from picf_next.eval.m2_protocol import (  # noqa: E402
    low_count_metrics as _low_count_metrics,
)
from picf_next.eval.m2_protocol import (  # noqa: E402
    paired_count_support_plan as _paired_batch_plan,
)
from picf_next.hosts.molmoact2_training import (  # noqa: E402
    CalvinVisibleObjectTargetBuilder,
)
from picf_next.models.set_loss import ObjectSetCriterion  # noqa: E402
from picf_next.training.molmoact2_m2 import load_molmoact2_m2_recipe  # noqa: E402
from tools.audit_molmoact2_m2_external_validation import (  # noqa: E402
    _group_by_target_count,
    _unique_source_keys,
)
from tools.run_molmoact2_m2_cloud import (  # noqa: E402
    _build_targets,
    _emit_progress,
    _evaluate,
    _keys_for_split,
    _learning_rate_multiplier,
    _load_cache,
    _native_bank,
    _render_visuals,
    _sha256,
    _stack_batch,
    _state_dict_cpu,
    _state_dict_sha256,
    _validation_selection_key,
    _write_json_atomic,
    _write_torch_atomic,
)

_LOW_COUNT_SEGMENT = 1
_EXTERNAL_SPLIT = "external_validation"


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
    parser.add_argument("--baseline-external-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=20)
    return parser.parse_args()


def _source_identity() -> dict[str, Any]:
    paths = (
        "src/picf_next/models/discovery.py",
        "src/picf_next/models/set_loss.py",
        "tools/audit_molmoact2_m2_count_support.py",
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


def _keys_for_segment(
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    segment_index: int,
) -> list[str]:
    return sorted(
        (
            key
            for key, (_tokens, _valid, record) in cache.items()
            if int(record["segment_index"]) == segment_index
        ),
        key=lambda key: int(cache[key][2]["transition_index"]),
    )


def _visible_count(
    sidecar: CalvinPhysicalSupervisionSidecar,
    record: Mapping[str, Any],
) -> int:
    frame = sidecar(int(record["segment_index"]), int(record["global_index"]))
    owners = {
        int(owner)
        for camera in frame.cameras
        for owner in np.unique(camera.owner_index).tolist()
        if int(owner) > 0
    }
    if any(owner > len(frame.identity_keys) for owner in owners):
        raise RuntimeError("physical supervision references an unknown owner")
    return len(owners)


def _count_histogram(
    *,
    sidecar: CalvinPhysicalSupervisionSidecar,
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
) -> dict[str, int]:
    return {
        str(count): frequency
        for count, frequency in sorted(
            Counter(_visible_count(sidecar, cache[key][2]) for key in keys).items()
        )
    }


def _source_hashes(
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    keys: Sequence[str],
) -> set[tuple[tuple[str, str], ...]]:
    return {
        tuple((str(name), str(digest)) for name, digest in cache[key][2]["source_sensor_sha256"])
        for key in keys
    }


def _train_paired(
    *,
    output_dir: Path,
    recipe: Any,
    foundation: Any,
    training_assets: Any,
    cache_manifest: Mapping[str, Any],
    cache: Mapping[str, tuple[Any, Any, Mapping[str, Any]]],
    treatment_plan: Sequence[Sequence[str]],
    control_plan: Sequence[Sequence[str]],
    validation_keys: Sequence[str],
    common_setup: Callable[[Any], None] | None = None,
    treatment_setup: Callable[[Any], None] | None = None,
    control_setup: Callable[[Any], None] | None = None,
    progress_event: str = "count_support_validation",
    report_schema: str = "picf-next.molmoact2-m2-count-support-training.v1",
    checkpoint_filenames: tuple[str, str] = (
        "low_count_treatment.pt",
        "equal_high_count_control.pt",
    ),
) -> tuple[dict[str, Any], Any, Any]:
    import torch

    if len(treatment_plan) != recipe.optimization.steps or len(control_plan) != len(treatment_plan):
        raise ValueError("paired audit plans differ from the recipe")
    if not progress_event or not report_schema:
        raise ValueError("paired audit progress event and report schema must be nonempty")
    if (
        len(checkpoint_filenames) != 2
        or any(Path(name).name != name or not name for name in checkpoint_filenames)
        or checkpoint_filenames[0] == checkpoint_filenames[1]
    ):
        raise ValueError("paired checkpoint filenames must be distinct basenames")
    torch.manual_seed(recipe.optimization.seed)
    torch.cuda.manual_seed_all(recipe.optimization.seed)
    treatment = foundation.core_config.build_current_frame()
    if common_setup is not None:
        common_setup(treatment)
    control = copy.deepcopy(treatment)
    if treatment_setup is not None:
        treatment_setup(treatment)
    if control_setup is not None:
        control_setup(control)
    initial_state = _state_dict_cpu(treatment)
    treatment_device = torch.device("cuda:0")
    control_device = torch.device("cuda:1")
    treatment.to(treatment_device)
    control.to(control_device)
    target_builder = CalvinVisibleObjectTargetBuilder(training_assets.physical_sidecar)
    treatment_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(treatment_device)
    control_criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(control_device)
    treatment_optimizer = torch.optim.AdamW(
        treatment.parameters(),
        lr=recipe.optimization.learning_rate,
        weight_decay=recipe.optimization.weight_decay,
    )
    control_optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=recipe.optimization.learning_rate,
        weight_decay=recipe.optimization.weight_decay,
    )
    layout_payload = cache_manifest["processor_layout"]
    best_keys: dict[str, tuple[float, ...] | None] = {"treatment": None, "control": None}
    best_steps = {"treatment": 0, "control": 0}
    best_states: dict[str, dict[str, Any] | None] = {"treatment": None, "control": None}
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(treatment_device)
    torch.cuda.reset_peak_memory_stats(control_device)
    torch.cuda.synchronize(treatment_device)
    torch.cuda.synchronize(control_device)
    started = time.perf_counter()

    for step, (treatment_keys, control_keys) in enumerate(
        zip(treatment_plan, control_plan, strict=True),
        start=1,
    ):
        treatment.train()
        control.train()
        treatment_optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        treatment_tokens, treatment_valid, treatment_records = _stack_batch(
            cache,
            treatment_keys,
            device=treatment_device,
        )
        control_tokens, control_valid, control_records = _stack_batch(
            cache,
            control_keys,
            device=control_device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            treatment_output = treatment(_native_bank(treatment_tokens, treatment_valid))
            control_output = control(_native_bank(control_tokens, control_valid))
        treatment_targets = _build_targets(
            target_builder=target_builder,
            records=treatment_records,
            token_valid=treatment_output.projection.token_valid,
            target_dtype=treatment_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        control_targets = _build_targets(
            target_builder=target_builder,
            records=control_records,
            token_valid=control_output.projection.token_valid,
            target_dtype=control_output.discovery.ownership.dtype,
            layout_payload=layout_payload,
            token_count=recipe.cache.token_count,
        )
        treatment_result = treatment_criterion(treatment_output.discovery, treatment_targets)
        control_result = control_criterion(control_output.discovery, control_targets)
        treatment_result.total.backward()
        control_result.total.backward()
        treatment_grad = torch.nn.utils.clip_grad_norm_(
            treatment.parameters(),
            recipe.optimization.gradient_clip_norm,
        )
        control_grad = torch.nn.utils.clip_grad_norm_(
            control.parameters(),
            recipe.optimization.gradient_clip_norm,
        )
        if not torch.isfinite(treatment_grad) or not torch.isfinite(control_grad):
            raise FloatingPointError("count-support gradient became non-finite")
        multiplier = _learning_rate_multiplier(step, recipe)
        for optimizer in (treatment_optimizer, control_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = recipe.optimization.learning_rate * multiplier
            optimizer.step()
        row: dict[str, Any] = {
            "step": step,
            "learning_rate": recipe.optimization.learning_rate * multiplier,
            "treatment_loss": float(treatment_result.total.detach().float().item()),
            "control_loss": float(control_result.total.detach().float().item()),
            "treatment_gradient_norm": float(treatment_grad.detach().float().item()),
            "control_gradient_norm": float(control_grad.detach().float().item()),
        }

        if step % recipe.optimization.validation_interval == 0:
            validation = {}
            for name, model, criterion, device in (
                ("treatment", treatment, treatment_criterion, treatment_device),
                ("control", control, control_criterion, control_device),
            ):
                metrics = _evaluate(
                    model=model,
                    cache=cache,
                    keys=validation_keys,
                    target_builder=target_builder,
                    criterion=criterion,
                    layout_payload=layout_payload,
                    recipe=recipe,
                    device=device,
                )
                validation[name] = metrics
                selection_key = _validation_selection_key(metrics)
                row[f"{name}_validation_selection_key"] = list(selection_key)
                if best_keys[name] is None or selection_key > best_keys[name]:
                    best_keys[name] = selection_key
                    best_steps[name] = step
                    best_states[name] = _state_dict_cpu(model)
            _emit_progress(
                progress_event,
                step=step,
                treatment_dice=validation["treatment"]["mean_object_dice"],
                treatment_count_mae=validation["treatment"]["count_mae"],
                control_dice=validation["control"]["mean_object_dice"],
                control_count_mae=validation["control"]["count_mae"],
                treatment_best_step=best_steps["treatment"],
                control_best_step=best_steps["control"],
            )
        rows.append(row)

    torch.cuda.synchronize(treatment_device)
    torch.cuda.synchronize(control_device)
    elapsed = time.perf_counter() - started
    if any(best_states[name] is None for name in best_states):
        raise RuntimeError("count-support training selected no validation checkpoints")
    treatment_state = best_states["treatment"]
    control_state = best_states["control"]
    assert treatment_state is not None and control_state is not None
    treatment.load_state_dict(treatment_state, strict=True)
    control.load_state_dict(control_state, strict=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    treatment_path = checkpoint_dir / checkpoint_filenames[0]
    control_path = checkpoint_dir / checkpoint_filenames[1]
    _write_torch_atomic(treatment_path, {"model": treatment_state})
    _write_torch_atomic(control_path, {"model": control_state})
    report = {
        "schema": report_schema,
        "steps": recipe.optimization.steps,
        "batch_size": recipe.optimization.batch_size,
        "initial_state_sha256": _state_dict_sha256(initial_state),
        "treatment_best_step": best_steps["treatment"],
        "control_best_step": best_steps["control"],
        "treatment_best_selection_key": list(best_keys["treatment"] or ()),
        "control_best_selection_key": list(best_keys["control"] or ()),
        "elapsed_s": elapsed,
        "seconds_per_paired_step": elapsed / recipe.optimization.steps,
        "peak_allocated_bytes": {
            "cuda:0": int(torch.cuda.max_memory_allocated(treatment_device)),
            "cuda:1": int(torch.cuda.max_memory_allocated(control_device)),
        },
        "checkpoints": {
            treatment_path.name: _sha256(treatment_path),
            control_path.name: _sha256(control_path),
        },
        "metrics": rows,
    }
    return report, treatment, control


def main() -> None:
    import torch

    from picf_next.training.molmoact2_calvin import load_calvin_training_assets

    args = _parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("paired count-support audit requires two CUDA devices")
    if args.steps <= 0 or args.validation_interval <= 0:
        raise ValueError("count-support steps and validation interval must be positive")
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
    base_keys = _keys_for_split(training_cache, "train")
    validation_keys = _keys_for_split(training_cache, "validation")
    support_segment_keys = _keys_for_segment(training_cache, _LOW_COUNT_SEGMENT)
    support_counts = {
        key: _visible_count(training_assets.physical_sidecar, training_cache[key][2])
        for key in support_segment_keys
    }
    low_keys = [key for key in support_segment_keys if support_counts[key] <= 8]
    high_candidates = [key for key in support_segment_keys if support_counts[key] == 9]
    high_keys = high_candidates[: len(low_keys)]
    if len(base_keys) != 192 or len(low_keys) != 16 or len(high_keys) != 16:
        raise RuntimeError("count-support source sample counts changed")
    histograms = {
        "base": _count_histogram(
            sidecar=training_assets.physical_sidecar,
            cache=training_cache,
            keys=base_keys,
        ),
        "low_count_treatment": _count_histogram(
            sidecar=training_assets.physical_sidecar,
            cache=training_cache,
            keys=low_keys,
        ),
        "equal_high_count_control": _count_histogram(
            sidecar=training_assets.physical_sidecar,
            cache=training_cache,
            keys=high_keys,
        ),
    }
    expected_histograms = {
        "base": {"9": 107, "10": 85},
        "low_count_treatment": {"7": 4, "8": 12},
        "equal_high_count_control": {"9": 16},
    }
    if histograms != expected_histograms:
        raise RuntimeError(f"count-support histograms changed: {histograms}")

    treatment_plan, control_plan, plan_report = _paired_batch_plan(
        base_keys=base_keys,
        treatment_supplement=low_keys,
        control_supplement=high_keys,
        seed=recipe.optimization.seed,
        steps=recipe.optimization.steps,
        batch_size=recipe.optimization.batch_size,
    )
    _write_json_atomic(output_dir / "batch_plan.json", plan_report)

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
        raise RuntimeError("external count-support keys have an unexpected split")
    train_hashes = _source_hashes(
        training_cache,
        base_keys + low_keys + high_keys + validation_keys,
    )
    external_hashes = _source_hashes(external_cache, external_keys)
    if train_hashes & external_hashes:
        raise RuntimeError("count-support training and external source frames overlap")

    baseline_report_path = args.baseline_external_report.expanduser().resolve()
    baseline_report = json.loads(baseline_report_path.read_text())
    baseline_actual = baseline_report["actual_unique_source"]
    if int(baseline_actual["sample_count"]) != len(external_keys):
        raise RuntimeError("baseline and count-support external sample sets differ")
    _write_json_atomic(
        output_dir / "audit_manifest.json",
        {
            "schema": "picf-next.molmoact2-m2-count-support-audit.v1",
            "authorizes_later_gates": False,
            "source": _source_identity(),
            "recipe": recipe.to_dict(),
            "recipe_sha256": recipe.recipe_sha256,
            "training_feature_cache": str(training_cache_root),
            "training_feature_cache_manifest_sha256": _sha256(
                training_cache_root / "manifest.json"
            ),
            "external_feature_cache": str(external_cache_root),
            "external_feature_cache_manifest_sha256": _sha256(
                external_cache_root / "manifest.json"
            ),
            "baseline_external_report": str(baseline_report_path),
            "baseline_external_report_sha256": _sha256(baseline_report_path),
            "training_sample_histograms": histograms,
            "external_unique_source_count": len(external_keys),
            "training_external_source_hash_intersection": 0,
        },
    )

    training_report, treatment, control = _train_paired(
        output_dir=output_dir,
        recipe=recipe,
        foundation=foundation,
        training_assets=training_assets,
        cache_manifest=training_manifest,
        cache=training_cache,
        treatment_plan=treatment_plan,
        control_plan=control_plan,
        validation_keys=validation_keys,
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
    treatment_low = _low_count_metrics(treatment_external["per_sample"])
    control_low = _low_count_metrics(control_external["per_sample"])
    checks = {
        "low_count_mae_improves_at_least_25_percent": (
            float(treatment_low["count_mae"]) <= 0.75 * float(control_low["count_mae"])
        ),
        "low_count_exact_improves_at_least_0_10": (
            float(treatment_low["exact_count_accuracy"])
            >= float(control_low["exact_count_accuracy"]) + 0.10
        ),
        "external_dice_noninferior_within_0_03": (
            treatment_external["mean_object_dice"] >= control_external["mean_object_dice"] - 0.03
        ),
        "external_ownership_noninferior_within_0_03": (
            treatment_external["ownership_accuracy"]
            >= control_external["ownership_accuracy"] - 0.03
        ),
        "external_geometry_noninferior_within_10_percent": (
            treatment_external["geometry_mae_physical"]
            <= 1.10 * control_external["geometry_mae_physical"]
        ),
    }

    external_assets = SimpleNamespace(
        dataset=external_dataset,
        physical_sidecar=external_physical,
    )
    treatment_dir = output_dir / "low_count_treatment"
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
    control_dir = output_dir / "equal_high_count_control"
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
        "schema": "picf-next.molmoact2-m2-count-support-result.v1",
        "authorizes_later_gates": False,
        "support_hypothesis_checks": checks,
        "support_hypothesis_supported": all(checks.values()),
        "baseline_external": {
            key: baseline_actual[key]
            for key in (
                "sample_count",
                "mean_object_dice",
                "ownership_accuracy",
                "count_mae",
                "exact_count_accuracy",
                "geometry_mae_physical",
            )
        },
        "low_count_treatment_external": treatment_external,
        "equal_high_count_control_external": control_external,
        "low_count_treatment_external_by_target_count": _group_by_target_count(
            treatment_external["per_sample"]
        ),
        "equal_high_count_control_external_by_target_count": _group_by_target_count(
            control_external["per_sample"]
        ),
        "low_count_treatment_7_8": treatment_low,
        "equal_high_count_control_7_8": control_low,
        "treatment_visuals_sha256": _sha256(treatment_dir / "visual_artifacts.json"),
        "control_visuals_sha256": _sha256(control_dir / "visual_artifacts.json"),
    }
    _write_json_atomic(output_dir / "count_support_report.json", report)
    print(
        json.dumps(
            {
                "support_hypothesis_checks": checks,
                "support_hypothesis_supported": all(checks.values()),
                "baseline_external": report["baseline_external"],
                "low_count_treatment_7_8": treatment_low,
                "equal_high_count_control_7_8": control_low,
                "low_count_treatment_external_by_target_count": report[
                    "low_count_treatment_external_by_target_count"
                ],
                "equal_high_count_control_external_by_target_count": report[
                    "equal_high_count_control_external_by_target_count"
                ],
                "seconds_per_paired_step": training_report["seconds_per_paired_step"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
