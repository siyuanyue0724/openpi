#!/usr/bin/env python3
"""Evaluate one selected all-source M2 checkpoint on every external CALVIN frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

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
)
from picf_next.data.calvin_physical_supervision_schema import (  # noqa: E402
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
)
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (  # noqa: E402
    load_dataset_file_manifest,
    validate_dataset_files,
)
from picf_next.eval.m2_protocol import group_by_target_count  # noqa: E402
from picf_next.hosts.molmoact2_training import (  # noqa: E402
    CalvinVisibleObjectTargetBuilder,
)
from picf_next.models.set_loss import ObjectSetCriterion  # noqa: E402
from picf_next.training.molmoact2_m2_source_coverage import (  # noqa: E402
    M2_SOURCE_COVERAGE_GATE,
    load_molmoact2_m2_source_coverage_recipe,
)
from tools import run_molmoact2_m2_cloud as m2  # noqa: E402
from tools import run_molmoact2_m2_source_coverage_cloud as source_runner  # noqa: E402

_EXTERNAL_SPLIT = "external_validation"
_EXTERNAL_SCHEMA = "picf-next.molmoact2-m2-source-coverage-external.v1"
_EXTERNAL_REPORTS = (
    "launch_manifest.json",
    "feature_cache/manifest.json",
    "task_intervention.json",
    "evaluation_report.json",
    "visual_artifacts.json",
)


def validate_external_machine_decision(
    run_dir: Path,
    *,
    training_run: Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    training_run = Path(training_run).expanduser().resolve()
    if run_dir != training_run / "external_validation":
        raise ValueError("external M2 run is not uniquely bound to its training run")
    decision_path = run_dir / "machine_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("external M2 machine decision is absent")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema") != "picf-next.molmoact2-m2-source-coverage-external-decision.v1"
        or decision.get("gate") != M2_SOURCE_COVERAGE_GATE
        or decision.get("status") not in {"PASS_PENDING_VISUAL_REVIEW", "FAIL"}
        or decision.get("later_gates_authorized") != []
    ):
        raise ValueError("external M2 machine decision identity or status changed")
    hashes = decision.get("required_report_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_EXTERNAL_REPORTS):
        raise ValueError("external M2 machine report set changed")
    for relative, digest in hashes.items():
        path = run_dir / relative
        if not path.is_file() or m2._sha256(path) != digest:
            raise ValueError(f"external M2 machine report changed: {relative}")
    launch = json.loads((run_dir / "launch_manifest.json").read_text())
    if Path(
        str(launch.get("training_run", ""))
    ).expanduser().resolve() != training_run or launch.get(
        "training_visual_decision_sha256"
    ) != m2._sha256(training_run / "training_visual_decision.json"):
        raise ValueError("external M2 launch is not bound to accepted training visuals")
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_source_coverage.json",
    )
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument(
        "--sidecar-artifact-root",
        type=Path,
        default=Path("/mnt/picf-next/artifacts/calvin_loss_sidecars"),
    )
    parser.add_argument("--training-normalization", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint-dir", type=Path, required=True)
    return parser.parse_args()


def _external_rows(start: int, stop: int, *, block_frames: int = 100) -> list[tuple[int, str, int]]:
    if stop <= start or block_frames <= 0:
        raise ValueError("external source range and block size must be positive")
    return [
        (global_index, _EXTERNAL_SPLIT, offset // block_frames)
        for offset, global_index in enumerate(range(start, stop))
    ]


def _external_acceptance(
    *,
    recipe: Any,
    actual: dict[str, Any],
    random: dict[str, Any],
    all_context: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    acceptance = recipe.acceptance
    checks = {
        "external_exact_count_accuracy": actual["exact_count_accuracy"]
        >= acceptance.minimum_heldout_exact_count_accuracy,
        "external_mean_object_dice": actual["mean_object_dice"]
        >= acceptance.minimum_mean_object_dice,
        "external_ownership_accuracy": actual["ownership_accuracy"]
        >= acceptance.minimum_ownership_accuracy,
        "external_dice_beats_random": (
            actual["mean_object_dice"] - random["mean_object_dice"]
            >= acceptance.minimum_random_dice_margin
        ),
        "external_ownership_beats_all_context": (
            actual["ownership_accuracy"] - all_context["ownership_accuracy"]
            >= acceptance.minimum_ownership_accuracy_improvement_vs_all_context
        ),
        "external_count_mae_beats_random": m2._safe_ratio_improvement(
            random["count_mae"],
            actual["count_mae"],
        )
        >= acceptance.minimum_count_mae_improvement_fraction_vs_random,
        "external_geometry_mae_beats_random": m2._safe_ratio_improvement(
            random["geometry_mae_physical"],
            actual["geometry_mae_physical"],
        )
        >= acceptance.minimum_geometry_mae_improvement_fraction_vs_random,
        "external_task_intervention_exact": intervention["all_dense_features_exact"] is True
        and intervention["maximum_absolute_error"]
        <= acceptance.maximum_task_intervention_feature_error,
        "external_no_nonfinite_metrics": actual["nonfinite_metric_count"] == 0,
    }
    return {
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "status": "PASS_PENDING_VISUAL_REVIEW" if all(checks.values()) else "FAIL",
    }


def _load_assets(
    *,
    source_recipe: Any,
    foundation: Any,
    split_root: Path,
    sidecar_artifact_root: Path,
    normalization_path: Path,
) -> tuple[Any, dict[str, Any]]:
    external = source_recipe.external_validation
    manifest_path = (_ROOT / external.dataset_manifest_path).resolve()
    if m2._sha256(manifest_path) != external.dataset_manifest_sha256:
        raise ValueError("external CALVIN dataset manifest changed")
    manifest = load_dataset_file_manifest(manifest_path)
    validate_dataset_files(
        manifest,
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
        dataset_manifest=manifest,
    )
    episodes = index.episodes
    if len(episodes) != 1 or (episodes[0].start, episodes[0].end + 1) != external.source_episode:
        raise ValueError("external CALVIN split differs from the preregistered source episode")
    physical_root = sidecar_artifact_root / external.physical_sidecar_name
    manifest_bytes = (physical_root / "manifest.json").read_bytes()
    if m2._sha256(physical_root / "manifest.json") != external.physical_sidecar_manifest_sha256:
        raise ValueError("external all-source physical sidecar manifest changed")
    physical_manifest = json.loads(manifest_bytes)
    shards = physical_manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("external all-source physical sidecar has no shards")
    physical = CalvinPhysicalSupervisionSidecar(
        physical_root,
        index,
        manifest_bytes=manifest_bytes,
        verify_hashes=True,
        cache_shards=len(shards),
    )
    if physical.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise ValueError("external physical sidecar is not all-source")
    if m2._sha256(normalization_path) != foundation.artifacts.normalization_file_sha256:
        raise ValueError("external evaluation did not reuse pinned training normalization")
    normalization = load_calvin_normalization_artifact(normalization_path)
    if (
        normalization["dataset_id"] != foundation.dataset.dataset_id
        or normalization["dataset_revision"] != foundation.dataset.dataset_revision
    ):
        raise ValueError("external training-normalization dataset identity changed")
    assets = SimpleNamespace(
        index=index,
        dataset=CalvinStatefulTransitionDataset(
            index,
            action_horizon=foundation.dataset.action_horizon,
        ),
        normalization_payload=normalization,
        physical_sidecar=physical,
    )
    provenance = {
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": external.dataset_manifest_sha256,
        "dataset_tree_sha256": manifest.tree_sha256,
        "physical_sidecar_manifest": str(physical_root / "manifest.json"),
        "physical_sidecar_manifest_sha256": external.physical_sidecar_manifest_sha256,
        "training_normalization": str(normalization_path),
        "training_normalization_sha256": foundation.artifacts.normalization_file_sha256,
        "validation_statistics_fitted": False,
    }
    return assets, provenance


def main() -> None:
    from tools.train_molmoact2_calvin_picf import _validate_training_checkpoint

    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("all-source external M2 evaluation requires CUDA")
    code_revision = m2._clean_git_revision()
    training_run = args.training_run.expanduser().resolve()
    output_dir = training_run / "external_validation"
    if not m2._is_under_mnt(training_run):
        raise RuntimeError("external evaluation and its selected run must persist under /mnt")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    machine = source_runner.validate_source_coverage_machine_decision(training_run)
    if machine["status"] != "PASS_PENDING_VISUAL_REVIEW":
        raise ValueError("failed source-coverage machine checks cannot enter external evaluation")
    training_visual = source_runner.validate_source_coverage_training_visual_decision(training_run)
    if training_visual["status"] != "PASS":
        raise ValueError("failed training visuals cannot authorize external evaluation")

    source_recipe = load_molmoact2_m2_source_coverage_recipe(args.config.resolve())
    base_recipe = source_recipe.load_base_m2(_ROOT)
    foundation = base_recipe.load_foundation(_ROOT)
    source_recipe.load_external_target_probe(_ROOT)
    launch = json.loads((training_run / "launch_manifest.json").read_text())
    if launch.get("source_coverage_recipe_sha256") != source_recipe.recipe_sha256:
        raise ValueError("selected source-coverage run used a different recipe")
    checkpoint_dir = args.foundation_checkpoint_dir.expanduser().resolve()
    if Path(str(launch.get("checkpoint_dir", ""))).resolve() != checkpoint_dir:
        raise ValueError("external foundation checkpoint differs from the selected training run")
    prior_m1 = m2.validate_prior_m1(Path(launch["prior_m1"]["run_dir"]))
    _validate_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        m0_report=prior_m1.pop("m0_raw_report"),
        checkpoint_id=foundation.host.checkpoint_id,
        checkpoint_revision=foundation.host.checkpoint_revision,
    )

    split_root = args.dataset_split_root.expanduser().resolve()
    sidecar_root = args.sidecar_artifact_root.expanduser().resolve()
    if not m2._is_under_mnt(sidecar_root):
        raise RuntimeError("external physical sidecars must persist under /mnt")
    normalization_path = args.training_normalization.expanduser().resolve()
    assets, provenance = _load_assets(
        source_recipe=source_recipe,
        foundation=foundation,
        split_root=split_root,
        sidecar_artifact_root=sidecar_root,
        normalization_path=normalization_path,
    )
    rows = _external_rows(*source_recipe.external_validation.source_episode)
    output_dir.mkdir(parents=True)
    m2._write_json_atomic(
        output_dir / "launch_manifest.json",
        {
            "schema": _EXTERNAL_SCHEMA,
            "gate": M2_SOURCE_COVERAGE_GATE,
            "code_revision": code_revision,
            "training_run": str(training_run),
            "training_machine_decision_sha256": m2._sha256(training_run / "machine_decision.json"),
            "training_visual_decision_sha256": m2._sha256(
                training_run / "training_visual_decision.json"
            ),
            "selected_checkpoint": str(training_run / "checkpoints/current_frame_best.pt"),
            "source_coverage_recipe_sha256": source_recipe.recipe_sha256,
            "frame_count": len(rows),
            "provenance": provenance,
            "threshold_reselection_authorized": False,
            "checkpoint_reselection_authorized": False,
        },
    )
    cache_manifest, intervention = source_runner._extract_source_feature_cache(
        run_dir=output_dir,
        source_recipe=source_recipe,
        base_recipe=base_recipe,
        foundation=foundation,
        assets=assets,
        checkpoint_dir=checkpoint_dir,
        rows=rows,
        gate=M2_SOURCE_COVERAGE_GATE,
    )
    m2._write_json_atomic(output_dir / "task_intervention.json", intervention)
    cache_manifest, cache = m2._load_cache(output_dir / "feature_cache", base_recipe)
    keys = m2._keys_for_split(cache, _EXTERNAL_SPLIT)
    if len(keys) != source_recipe.external_validation.frame_count:
        raise RuntimeError("external feature cache does not cover every source frame")

    device = torch.device("cuda:0")
    torch.manual_seed(base_recipe.optimization.seed)
    model = foundation.core_config.build_current_frame().to(device)
    criterion = ObjectSetCriterion(config=foundation.set_loss_config).to(device)
    target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    random = m2._evaluate(
        model=model,
        cache=cache,
        keys=keys,
        target_builder=target_builder,
        criterion=criterion,
        layout_payload=cache_manifest["processor_layout"],
        recipe=base_recipe,
        device=device,
    )
    all_context = m2._all_context_baseline(
        cache=cache,
        keys=keys,
        target_builder=target_builder,
        layout_payload=cache_manifest["processor_layout"],
        recipe=base_recipe,
        device=device,
    )
    checkpoint = training_run / "checkpoints/current_frame_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"] if set(payload) == {"model"} else payload
    model.load_state_dict(state, strict=True)
    actual = m2._evaluate(
        model=model,
        cache=cache,
        keys=keys,
        target_builder=target_builder,
        criterion=criterion,
        layout_payload=cache_manifest["processor_layout"],
        recipe=base_recipe,
        device=device,
        include_per_sample=True,
    )
    evaluation = {
        "schema": _EXTERNAL_SCHEMA,
        "split": _EXTERNAL_SPLIT,
        "checkpoint_sha256": m2._sha256(checkpoint),
        "random_initialization": random,
        "all_context": all_context,
        "actual": actual,
        "actual_by_target_count": group_by_target_count(actual["per_sample"]),
    }
    m2._write_json_atomic(output_dir / "evaluation_report.json", evaluation)
    expected_blocks = {block for _global_index, _split, block in rows}
    visuals = m2._render_visuals(
        run_dir=output_dir,
        model=model,
        assets=assets,
        cache=cache,
        cache_manifest=cache_manifest,
        foundation=foundation,
        recipe=base_recipe,
        visual_splits=(_EXTERNAL_SPLIT,),
        expected_segments=expected_blocks,
        gate=M2_SOURCE_COVERAGE_GATE,
    )
    m2._write_json_atomic(output_dir / "visual_artifacts.json", visuals)
    acceptance = _external_acceptance(
        recipe=base_recipe,
        actual=actual,
        random=random,
        all_context=all_context,
        intervention=intervention,
    )
    hashes = {relative: m2._sha256(output_dir / relative) for relative in _EXTERNAL_REPORTS}
    decision = {
        "schema": "picf-next.molmoact2-m2-source-coverage-external-decision.v1",
        "gate": M2_SOURCE_COVERAGE_GATE,
        "status": acceptance["status"],
        "checks": acceptance["checks"],
        "failed_checks": acceptance["failed_checks"],
        "required_report_sha256": hashes,
        "later_gates_authorized": [],
    }
    m2._write_json_atomic(output_dir / "machine_decision.json", decision)
    m2._emit_progress(
        "external_machine_decision",
        output_dir=str(output_dir),
        status=decision["status"],
        failed_checks=decision["failed_checks"],
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
