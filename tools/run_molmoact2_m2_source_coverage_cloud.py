#!/usr/bin/env python3
"""Run the pre-registered all-source root-cause audit for MolmoAct2 M2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
if str(_MOLMO_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_MOLMO_EXPERIMENTS))

from picf_next.data.calvin_physical_supervision_schema import (  # noqa: E402
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
)
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.hosts.molmoact2_training import (  # noqa: E402
    molmoact2_host_observation_view,
)
from picf_next.training.molmoact2_m2_source_coverage import (  # noqa: E402
    M2_SOURCE_COVERAGE_GATE,
    MolmoAct2M2SourceCoverageRecipe,
    load_molmoact2_m2_source_coverage_recipe,
    m2_source_coverage_report,
)
from tools import run_molmoact2_m2_cloud as m2  # noqa: E402

_MACHINE_REPORTS = m2._M2_MACHINE_REPORTS


def validate_source_coverage_machine_decision(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    decision_path = run_dir / "machine_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("M2 source-coverage machine decision is absent")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema") != "picf-next.molmoact2-m2-source-coverage-decision.v1"
        or decision.get("gate") != M2_SOURCE_COVERAGE_GATE
        or decision.get("base_gate") != m2.M2_GATE
        or decision.get("status") not in {"PASS_PENDING_VISUAL_REVIEW", "FAIL"}
        or decision.get("external_validation_required_before_m2_acceptance") is not True
    ):
        raise ValueError("M2 source-coverage machine decision identity or status changed")
    hashes = decision.get("required_report_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_MACHINE_REPORTS):
        raise ValueError("M2 source-coverage machine report set changed")
    for relative, expected in hashes.items():
        path = run_dir / relative
        if not path.is_file() or m2._sha256(path) != expected:
            raise ValueError(f"M2 source-coverage machine report hash changed: {relative}")
    return decision


def validate_source_coverage_training_visual_decision(
    run_dir: Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    decision_path = run_dir / "training_visual_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("M2 source-coverage training visual decision is absent")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema")
        != "picf-next.molmoact2-m2-source-coverage-training-visual-decision.v1"
        or decision.get("gate") != M2_SOURCE_COVERAGE_GATE
        or decision.get("status") not in {"PASS", "FAIL"}
        or not isinstance(decision.get("external_validation_authorized"), bool)
        or decision.get("external_validation_authorized") != (decision.get("status") == "PASS")
    ):
        raise ValueError("M2 source-coverage training visual decision changed")
    hashes = decision.get("required_report_sha256")
    expected = {
        "machine_decision.json",
        "training_visual_review.json",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise ValueError("M2 source-coverage training visual report set changed")
    for relative, digest in hashes.items():
        path = run_dir / relative
        if not path.is_file() or m2._sha256(path) != digest:
            raise ValueError(f"M2 source-coverage training visual report changed: {relative}")
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m2_source_coverage.json",
    )
    parser.add_argument("--m1-run", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument(
        "--sidecar-artifact-root",
        type=Path,
        default=Path("/mnt/picf-next/artifacts/calvin_loss_sidecars"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/picf-next/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_id(value: str | None) -> str:
    resolved = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not m2._RUN_ID.fullmatch(resolved):
        raise ValueError(f"invalid M2 source-coverage run id: {resolved!r}")
    return resolved


def _validate_source_split(
    assets: Any,
    recipe: MolmoAct2M2SourceCoverageRecipe,
) -> dict[str, Any]:
    episodes = assets.index.episodes
    expected = recipe.split.source_episode
    if len(episodes) != 1 or (episodes[0].start, episodes[0].end + 1) != expected:
        raise ValueError("M2 source-coverage recipe differs from the exact CALVIN source episode")
    rows = [
        {
            "split": split,
            "source_block_index": index,
            "start": start,
            "end_exclusive": stop,
            "frame_count": stop - start,
        }
        for index, (split, start, stop) in enumerate(recipe.split.learned_ranges)
    ]
    return {
        "schema": "picf-next.molmoact2-m2-source-split.v1",
        "strategy": recipe.split.strategy,
        "source_episode": list(recipe.split.source_episode),
        "rows": rows,
        "guard_ranges": [list(value) for value in recipe.split.guard_ranges],
        "minimum_guard_frames": recipe.split.minimum_guard_frames,
        "transition_counts": {
            name: sum(row["frame_count"] for row in rows if row["split"] == name)
            for name in ("train", "validation", "heldout")
        },
        "learned_source_ranges_disjoint": True,
        "guard_frames_enter_feature_cache": False,
    }


def _source_sensor_hashes(
    frame: CalvinPhysicalSupervisionFrame,
) -> tuple[tuple[str, str], ...]:
    hashes = []
    for camera in frame.cameras:
        hashes.extend(
            (
                (f"rgb_{camera.camera_name}", camera.source_rgb_sha256),
                (f"depth_{camera.camera_name}", camera.source_depth_sha256),
            )
        )
    return tuple(sorted(hashes))


def _source_rows(
    recipe: MolmoAct2M2SourceCoverageRecipe,
) -> list[tuple[int, str, int]]:
    rows = [
        (global_index, split, block_index)
        for block_index, (split, start, stop) in enumerate(recipe.split.learned_ranges)
        for global_index in range(start, stop)
    ]
    if len({global_index for global_index, _split, _block in rows}) != len(rows):
        raise RuntimeError("M2 source-coverage rows are not unique")
    return rows


def _neutral_task_intervention_probe(
    *,
    policy: Any,
    processor: Any,
    assets: Any,
    device: torch.device,
) -> dict[str, Any]:
    from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation

    sample_positions = sorted({0, len(assets.dataset) // 2, len(assets.dataset) - 1})
    pairs = []
    maximum_error = 0.0
    exact = True
    for position in sample_positions:
        sample = assets.dataset[position]
        global_index = int(sample.record.global_index)
        source_observation = assets.index.molmoact2_source_observation(global_index)
        neutral_inputs = m2._move_inputs(
            processor.build_source_observation_inputs((source_observation,)),
            device,
        )
        language_inputs = m2._move_inputs(
            processor.build_observation_inputs(
                ((sample.picf_evidence_frame,),),
                (molmoact2_host_observation_view(sample.record),),
            ),
            device,
        )
        with torch.inference_mode():
            neutral = prepare_molmoact2_lerobot_observation(policy, neutral_inputs)
            language = prepare_molmoact2_lerobot_observation(policy, language_inputs)
        if neutral.vision_patch_bank is None or language.vision_patch_bank is None:
            raise RuntimeError("M2 neutral-task intervention omitted dense vision patches")
        difference = (
            neutral.vision_patch_bank.tokens.float() - language.vision_patch_bank.tokens.float()
        ).abs()
        error = float(difference.max().item())
        row_exact = bool(
            torch.equal(
                neutral.vision_patch_bank.tokens,
                language.vision_patch_bank.tokens,
            )
        )
        maximum_error = max(maximum_error, error)
        exact = exact and row_exact
        pairs.append(
            {
                "global_index": global_index,
                "language_instruction": sample.record.task,
                "neutral_task_field_supplied": False,
                "language_task_field_supplied": True,
                "dense_features_exact": row_exact,
                "maximum_absolute_error": error,
            }
        )
    return {
        "schema": "picf-next.molmoact2-m2-neutral-task-intervention.v1",
        "pair_count": len(pairs),
        "pairs": pairs,
        "all_dense_features_exact": exact,
        "maximum_absolute_error": maximum_error,
        "task_text_enters_trainable_m2_graph": False,
    }


def _extract_source_feature_cache(
    *,
    run_dir: Path,
    source_recipe: MolmoAct2M2SourceCoverageRecipe,
    base_recipe: Any,
    foundation: Any,
    assets: Any,
    checkpoint_dir: Path,
    rows: list[tuple[int, str, int]] | None = None,
    gate: str = M2_SOURCE_COVERAGE_GATE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation
    from picf_next.hosts.molmoact2_calvin_processor import CalvinMolmoAct2ProcessorBridge
    from picf_next.training.molmoact2_calvin import build_molmoact2_policy_config

    cache_dir = run_dir / "feature_cache"
    cache_dir.mkdir()
    policy_config = build_molmoact2_policy_config(
        foundation,
        checkpoint_path=checkpoint_dir,
    )
    device = torch.device("cuda:0")
    policy = MolmoAct2Policy(policy_config).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    stats = __import__(
        "picf_next.data.calvin_normalization",
        fromlist=["official_molmoact2_dataset_stats"],
    ).official_molmoact2_dataset_stats(assets.normalization_payload)
    processor = CalvinMolmoAct2ProcessorBridge.from_official_config(
        policy.config,
        dataset_stats=stats,
    )

    rows = _source_rows(source_recipe) if rows is None else rows
    if (
        not rows
        or any(
            not isinstance(global_index, int)
            or isinstance(global_index, bool)
            or global_index < 0
            or not isinstance(split, str)
            or not split
            or not isinstance(block, int)
            or isinstance(block, bool)
            or block < 0
            for global_index, split, block in rows
        )
        or len({global_index for global_index, _split, _block in rows}) != len(rows)
    ):
        raise ValueError("M2 task-free feature rows must be nonempty and source-unique")
    records: list[dict[str, Any]] = []
    canonical_layout: list[dict[str, Any]] | None = None
    pending_tokens = []
    pending_valid = []
    pending_records = []
    shard_index = 0
    shards: list[dict[str, Any]] = []
    extraction_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def flush() -> None:
        nonlocal shard_index
        if not pending_tokens:
            return
        path = cache_dir / f"features-{shard_index:05d}.pt"
        tokens = torch.cat(pending_tokens, dim=0).contiguous()
        valid = torch.cat(pending_valid, dim=0).contiguous()
        m2._write_torch_atomic(path, {"tokens": tokens, "valid": valid})
        for row_index, record in enumerate(pending_records):
            record["shard"] = path.name
            record["row"] = row_index
            records.append(record)
        shards.append(
            {
                "path": path.name,
                "sha256": m2._sha256(path),
                "rows": len(pending_records),
                "bytes": path.stat().st_size,
            }
        )
        m2._emit_progress(
            "feature_cache_shard",
            shard=path.name,
            shard_rows=len(pending_records),
            completed_rows=len(records),
            total_rows=len(rows),
        )
        pending_tokens.clear()
        pending_valid.clear()
        pending_records.clear()
        shard_index += 1

    batch_size = base_recipe.cache.extraction_batch_size
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        observations = tuple(
            assets.index.molmoact2_source_observation(global_index)
            for global_index, _split, _block in batch_rows
        )
        observation_inputs = m2._move_inputs(
            processor.build_source_observation_inputs(observations),
            device,
        )
        with torch.inference_mode():
            prepared = prepare_molmoact2_lerobot_observation(policy, observation_inputs)
        bank = prepared.vision_patch_bank
        layout = prepared.vision_patch_layout
        if bank is None or layout is None:
            raise RuntimeError("M2 source observation omitted dense Molmo patches or layout")
        if (
            bank.modality != base_recipe.cache.modality
            or bank.tokens.shape[1:] != (base_recipe.cache.token_count, base_recipe.cache.token_dim)
            or bank.valid.shape != bank.tokens.shape[:2]
        ):
            raise RuntimeError("M2 source native Molmo feature contract changed")
        layout_rows = [m2._layout_row_payload(row) for row in layout.rows]
        for layout_row in layout_rows:
            if canonical_layout is None:
                canonical_layout = layout_row
            elif layout_row != canonical_layout:
                raise RuntimeError("M2 source processor patch layout changed across rows")
        cpu_tokens = m2._regular_cpu_copy(bank.tokens, dtype=torch.bfloat16)
        cpu_valid = m2._regular_cpu_copy(bank.valid)
        for batch_index, (global_index, split, block_index) in enumerate(batch_rows):
            physical = assets.physical_sidecar.source_frame(global_index)
            pending_tokens.append(cpu_tokens[batch_index : batch_index + 1])
            pending_valid.append(cpu_valid[batch_index : batch_index + 1])
            pending_records.append(
                {
                    "sample_key": f"source-frame-{global_index:07d}",
                    "split": split,
                    "source_block_index": block_index,
                    "global_index": global_index,
                    "task_key": "task-independent-source-frame",
                    "instruction": "task field absent",
                    "target_request_contract": "source_frame",
                    "source_sensor_sha256": [
                        list(item) for item in _source_sensor_hashes(physical)
                    ],
                }
            )
            if len(pending_records) == base_recipe.cache.shard_rows:
                flush()
        del prepared, bank, observation_inputs
    flush()
    if canonical_layout is None:
        raise RuntimeError("M2 source coverage selected no features")
    if len(records) != len(rows) or len({row["sample_key"] for row in records}) != len(records):
        raise RuntimeError("M2 source feature cache is not one-to-one")

    intervention = _neutral_task_intervention_probe(
        policy=policy,
        processor=processor,
        assets=assets,
        device=device,
    )
    manifest = {
        "schema": "picf-next.molmoact2-m2-feature-cache.v1",
        "gate": gate,
        "checkpoint_id": foundation.host.checkpoint_id,
        "checkpoint_revision": foundation.host.checkpoint_revision,
        "foundation_recipe_sha256": foundation.recipe_sha256,
        "source_coverage_recipe_sha256": source_recipe.recipe_sha256,
        "modality": base_recipe.cache.modality,
        "dtype": base_recipe.cache.dtype,
        "token_shape": [base_recipe.cache.token_count, base_recipe.cache.token_dim],
        "processor_layout": canonical_layout,
        "processor_layout_sha256": m2._canonical_sha256(canonical_layout),
        "records": records,
        "records_sha256": m2._canonical_sha256(records),
        "shards": shards,
        "sample_count": len(records),
        "model_input_fields": ["tokens", "valid"],
        "loss_target_fields_in_feature_shards": [],
        "task_field_supplied": False,
        "elapsed_s": time.perf_counter() - extraction_started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    m2._write_json_atomic(cache_dir / "manifest.json", manifest)
    m2._emit_progress(
        "feature_cache_complete",
        sample_count=len(records),
        shard_count=len(shards),
        elapsed_s=manifest["elapsed_s"],
        neutral_task_intervention_maximum_absolute_error=intervention["maximum_absolute_error"],
    )
    del policy
    torch.cuda.empty_cache()
    return manifest, intervention


def _load_source_sidecar(
    *,
    artifact_root: Path,
    recipe: MolmoAct2M2SourceCoverageRecipe,
    assets: Any,
) -> tuple[Any, dict[str, Any]]:
    root = artifact_root / recipe.physical_sidecar_name
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if m2._sha256(manifest_path) != recipe.physical_sidecar_manifest_sha256:
        raise ValueError("M2 source-coverage physical sidecar manifest changed")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    shards = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(shards, list) or not shards:
        raise ValueError("M2 source-coverage physical sidecar has no declared shards")
    sidecar = CalvinPhysicalSupervisionSidecar(
        root,
        assets.index,
        manifest_bytes=manifest_bytes,
        verify_hashes=True,
        cache_shards=len(shards),
    )
    if sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise ValueError("M2 source-coverage sidecar is not all-source")
    return replace(assets, physical_sidecar=sidecar), {
        "root": str(root),
        "manifest_sha256": recipe.physical_sidecar_manifest_sha256,
        "coverage": sidecar.coverage,
        "frame_count": recipe.source_frame_count,
        "cache_shards": len(shards),
    }


def main() -> None:
    from picf_next.training.molmoact2_calvin import load_calvin_training_assets
    from tools.train_molmoact2_calvin_picf import _validate_training_checkpoint

    args = _parse_args()
    source_recipe = load_molmoact2_m2_source_coverage_recipe(args.config)
    base_recipe = source_recipe.load_base_m2(_ROOT)
    foundation = base_recipe.load_foundation(_ROOT)
    prior_m1 = m2.validate_prior_m1(args.m1_run)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    dataset_split_root = args.dataset_split_root.expanduser().resolve()
    sidecar_artifact_root = args.sidecar_artifact_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if not m2._is_under_mnt(run_root) or not m2._is_under_mnt(sidecar_artifact_root):
        raise RuntimeError("M2 source-coverage runs and sidecars must persist under /mnt")
    run_dir = run_root / "molmoact2" / M2_SOURCE_COVERAGE_GATE / _run_id(args.run_id)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite M2 source-coverage run: {run_dir}")

    code_revision = m2._clean_git_revision()
    static_report = m2_source_coverage_report(source_recipe, repository_root=_ROOT)
    if args.dry_run:
        print(json.dumps(static_report, indent=2, sort_keys=True))
        return

    resources = m2._validate_devices()
    _validate_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        m0_report=prior_m1.pop("m0_raw_report"),
        checkpoint_id=foundation.host.checkpoint_id,
        checkpoint_revision=foundation.host.checkpoint_revision,
    )
    sidecar_materialization = m2.materialize_persistent_sidecars(sidecar_artifact_root)
    assets = load_calvin_training_assets(
        foundation,
        repository_root=_ROOT,
        split_root=dataset_split_root,
    )
    assets, source_sidecar = _load_source_sidecar(
        artifact_root=sidecar_artifact_root,
        recipe=source_recipe,
        assets=assets,
    )
    split_report = _validate_source_split(assets, source_recipe)
    target_probe = source_recipe.load_target_probe(_ROOT)
    m2._emit_progress(
        "preflight_complete",
        code_revision=code_revision,
        source_frames=source_recipe.source_frame_count,
        split_transition_counts=split_report["transition_counts"],
        restored_sidecar_shards=len(sidecar_materialization["restored"]),
    )

    run_dir.mkdir(parents=True)
    launch = {
        "schema": "picf-next.molmoact2-m2-source-coverage-launch.v1",
        "gate": M2_SOURCE_COVERAGE_GATE,
        "base_gate": m2.M2_GATE,
        "run_dir": str(run_dir),
        "code_revision": code_revision,
        "config": str(args.config.resolve()),
        "config_file_sha256": m2._sha256(args.config.resolve()),
        "source_coverage_recipe_sha256": source_recipe.recipe_sha256,
        "base_m2_recipe_sha256": base_recipe.recipe_sha256,
        "foundation_recipe_sha256": foundation.recipe_sha256,
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_split_root": str(dataset_split_root),
        "sidecar_artifact_root": str(sidecar_artifact_root),
        "sidecar_materialization": sidecar_materialization,
        "source_sidecar": source_sidecar,
        "target_probe": {
            "path": source_recipe.target_probe_path,
            "sha256": source_recipe.target_probe_sha256,
            "frame_count": target_probe["frame_count"],
        },
        "prior_m1": prior_m1,
        "candidate_under_test": static_report["candidate_under_test"],
        "historical_comparison_scope": static_report["historical_comparison_scope"],
        "single_variable_source_coverage_attribution_authorized": False,
        "worktree_clean": True,
    }
    m2._write_json_atomic(run_dir / "launch_manifest.json", launch)
    m2._write_json_atomic(
        run_dir / "environment.json",
        {
            "schema": "picf-next.molmoact2-m2-source-coverage-environment.v1",
            "resources": resources,
            "python": sys.version,
            "torch": torch.__version__,
        },
    )
    m2._write_json_atomic(run_dir / "split_manifest.json", split_report)

    cache_manifest, intervention = _extract_source_feature_cache(
        run_dir=run_dir,
        source_recipe=source_recipe,
        base_recipe=base_recipe,
        foundation=foundation,
        assets=assets,
        checkpoint_dir=checkpoint_dir,
    )
    m2._write_json_atomic(run_dir / "task_intervention.json", intervention)
    cache_manifest, cache = m2._load_cache(run_dir / "feature_cache", base_recipe)
    training, evaluation, actual_model = m2._train_models(
        run_dir=run_dir,
        recipe=base_recipe,
        foundation=foundation,
        assets=assets,
        cache_manifest=cache_manifest,
        cache=cache,
    )
    m2._write_json_atomic(run_dir / "training_report.json", training)
    m2._write_json_atomic(run_dir / "evaluation_report.json", evaluation)
    visuals = m2._render_visuals(
        run_dir=run_dir,
        model=actual_model,
        assets=assets,
        cache=cache,
        cache_manifest=cache_manifest,
        foundation=foundation,
        recipe=base_recipe,
        expected_segments={
            block_index for block_index, _row in enumerate(source_recipe.split.learned_ranges)
        },
        gate=M2_SOURCE_COVERAGE_GATE,
    )
    m2._write_json_atomic(run_dir / "visual_artifacts.json", visuals)
    acceptance = m2._evaluate_acceptance(
        recipe=base_recipe,
        evaluation=evaluation,
        task_intervention=intervention,
        training=training,
    )
    report_hashes = {relative: m2._sha256(run_dir / relative) for relative in _MACHINE_REPORTS}
    decision = {
        "schema": "picf-next.molmoact2-m2-source-coverage-decision.v1",
        "gate": M2_SOURCE_COVERAGE_GATE,
        "base_gate": m2.M2_GATE,
        "status": acceptance["status"],
        "checks": acceptance["checks"],
        "failed_checks": acceptance["failed_checks"],
        "required_report_sha256": report_hashes,
        "later_gates_authorized": [],
        "external_validation_required_before_m2_acceptance": True,
    }
    m2._write_json_atomic(run_dir / "machine_decision.json", decision)
    m2._emit_progress(
        "machine_decision",
        run_dir=str(run_dir),
        status=decision["status"],
        failed_checks=decision["failed_checks"],
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
