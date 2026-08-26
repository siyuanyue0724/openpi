"""Hash-closed CALVIN assembly for stationary Stage-B temporal calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import DatasetFileManifest
from picf_next.data.molmoact2_source_cache import MolmoAct2SourceFeatureCache
from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder
from picf_next.training.molmoact2_calvin_temporal import (
    CalvinStationaryTemporalBatchBuilder,
)
from picf_next.training.molmoact2_m2_source_coverage import (
    MolmoAct2M2SourceCoverageRecipe,
)
from picf_next.training.recipe import PICFTrainingRecipe
from picf_next.training.stage_checkpoints import (
    load_picf_current_frame_checkpoint,
    sha256_file,
)
from picf_next.training.stationary_stage import (
    StationaryTemporalStageRecipe,
    load_stationary_temporal_stage_recipe,
)
from picf_next.training.stationary_temporal import StationaryTemporalCoreTrainer
from picf_next.training.temporal_clips import (
    DistributedStationaryTemporalClipPlan,
    build_distributed_stationary_temporal_clip_plan,
)


@dataclass(frozen=True, slots=True)
class StationaryCalvinStageDefinition:
    stage: StationaryTemporalStageRecipe
    source_coverage: MolmoAct2M2SourceCoverageRecipe
    historical_foundation: PICFTrainingRecipe
    structural_foundation: PICFTrainingRecipe
    clip_plan: DistributedStationaryTemporalClipPlan

    @property
    def maximum_horizon(self) -> int:
        return self.structural_foundation.geometry_overshooting.horizons[-1]


@dataclass(frozen=True, slots=True)
class StationaryCalvinStageAssets:
    index: CalvinDatasetIndex
    dataset_manifest: DatasetFileManifest
    feature_cache: MolmoAct2SourceFeatureCache
    physical_sidecar: CalvinPhysicalSupervisionSidecar
    batch_builder: CalvinStationaryTemporalBatchBuilder


def load_stationary_calvin_stage_definition(
    stage_recipe_path: str | Path,
    *,
    repository_root: str | Path,
) -> StationaryCalvinStageDefinition:
    root = Path(repository_root).resolve()
    stage = load_stationary_temporal_stage_recipe(stage_recipe_path)
    source = stage.load_source_coverage(root)
    historical = stage.load_foundation(root)
    structural = stage.structural_foundation(root)
    horizons = structural.geometry_overshooting.horizons
    if not horizons or horizons != tuple(sorted(set(horizons))) or horizons[0] != 1:
        raise ValueError("stationary CALVIN geometry horizons are not canonical")
    plan = build_distributed_stationary_temporal_clip_plan(
        source_ranges=source.split.train_ranges,
        prefix_lengths=stage.clip.prefix_lengths,
        train_length=stage.clip.train_length,
        required_future_horizon=horizons[-1],
        optimizer_steps=stage.optimizer.optimizer_steps,
        world_size=stage.distributed.world_size,
        seed=stage.clip.seed,
    )
    return StationaryCalvinStageDefinition(
        stage=stage,
        source_coverage=source,
        historical_foundation=historical,
        structural_foundation=structural,
        clip_plan=plan,
    )


def _load_dataset_manifest(
    definition: StationaryCalvinStageDefinition,
    repository_root: Path,
) -> DatasetFileManifest:
    foundation = definition.historical_foundation
    relative = foundation.artifacts.dataset_file_manifest_path
    path = (repository_root / relative).resolve()
    if repository_root not in path.parents or not path.is_file():
        raise FileNotFoundError("stationary CALVIN dataset manifest is absent or escaped")
    payload = path.read_bytes()
    if sha256_file(path) != foundation.artifacts.dataset_file_manifest_sha256:
        raise ValueError("stationary CALVIN dataset manifest changed")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("stationary CALVIN dataset manifest is invalid JSON") from exc
    return DatasetFileManifest.from_dict(raw)


def load_stationary_calvin_stage_assets(
    definition: StationaryCalvinStageDefinition,
    *,
    repository_root: str | Path,
    split_root: str | Path,
    feature_cache_root: str | Path,
    feature_cache_manifest_sha256: str,
    physical_sidecar_root: str | Path,
    cache_shards: int = 2,
) -> StationaryCalvinStageAssets:
    if not isinstance(definition, StationaryCalvinStageDefinition):
        raise TypeError("stationary CALVIN assets require a parsed stage definition")
    root = Path(repository_root).resolve()
    foundation = definition.historical_foundation
    manifest = _load_dataset_manifest(definition, root)
    index = CalvinDatasetIndex.load(
        Path(split_root).resolve(),
        dataset_id=foundation.dataset.dataset_id,
        dataset_revision=foundation.dataset.dataset_revision,
        verify_files=True,
        dataset_manifest=manifest,
    )
    sidecar_root = Path(physical_sidecar_root).resolve()
    sidecar_manifest = sidecar_root / "manifest.json"
    if (
        not sidecar_manifest.is_file()
        or sha256_file(sidecar_manifest)
        != definition.source_coverage.physical_sidecar_manifest_sha256
    ):
        raise ValueError("stationary CALVIN physical sidecar manifest changed")
    sidecar = CalvinPhysicalSupervisionSidecar(
        sidecar_root,
        index,
        manifest_bytes=sidecar_manifest.read_bytes(),
        verify_hashes=True,
        cache_shards=cache_shards,
    )
    base_m2 = definition.source_coverage.load_base_m2(root)
    cache = MolmoAct2SourceFeatureCache.load(
        feature_cache_root,
        manifest_sha256=feature_cache_manifest_sha256,
        expected_modality=base_m2.cache.modality,
        expected_token_count=base_m2.cache.token_count,
        expected_token_dim=base_m2.cache.token_dim,
        expected_checkpoint_id=foundation.host.checkpoint_id,
        expected_checkpoint_revision=foundation.host.checkpoint_revision,
        memory_capacity=cache_shards,
    )
    expected_records = {
        global_index: split
        for split, start, stop in definition.source_coverage.split.learned_ranges
        for global_index in range(start, stop)
    }
    observed_records = {
        global_index: record.split for global_index, record in cache.records.items()
    }
    if observed_records != expected_records:
        raise ValueError("stationary Molmo cache coverage differs from preregistered source splits")
    visible_builder = CalvinVisibleObjectTargetBuilder(sidecar)
    batch_builder = CalvinStationaryTemporalBatchBuilder(
        index,
        cache,
        visible_target_builder=visible_builder.source_frames,
        geometry_contract=foundation.geometry_contract,
        geometry_provider=lambda global_index: sidecar.source_frame(global_index).geometry_frame(),
        maximum_horizon=definition.maximum_horizon,
        supervised_horizons=foundation.geometry_overshooting.horizons,
    )
    return StationaryCalvinStageAssets(
        index=index,
        dataset_manifest=manifest,
        feature_cache=cache,
        physical_sidecar=sidecar,
        batch_builder=batch_builder,
    )


def build_stationary_temporal_trainer(
    definition: StationaryCalvinStageDefinition,
    *,
    m2_checkpoint_path: str | Path,
    m2_checkpoint_sha256: str,
    device: torch.device | str,
) -> StationaryTemporalCoreTrainer:
    """Initialize full Stage B from accepted current-frame weights and a fresh posterior."""

    structural = definition.structural_foundation
    core = structural.build_core()
    load_picf_current_frame_checkpoint(
        core,
        m2_checkpoint_path,
        expected_sha256=m2_checkpoint_sha256,
    )
    objective = structural.build_objective()
    trainer = StationaryTemporalCoreTrainer(
        core,
        objective,
        capacity=structural.core_config.posterior_capacity,
    )
    trainer.to(device=device, dtype=torch.float32)
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in trainer.parameters()
    ):
        raise RuntimeError("stationary temporal trainable parameter storage escaped FP32")
    return trainer
