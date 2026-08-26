from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
    CalvinVisibleOwnerRaster,
)
from picf_next.lingbot_native.current_grid_cache import (
    CurrentGridCacheContract,
    CurrentGridCacheRecord,
    LingBotCurrentGridTargetCache,
    current_grid_coverage_digest,
    current_grid_source_keys_digest,
    write_current_grid_target_cache,
)
from picf_next.lingbot_native.predictive_cache import (
    LINGBOT_PREDICTIVE_TARGET_SPACE,
    LingBotPredictiveTargetCache,
    PredictiveCacheContract,
    PredictiveObjectCacheRecord,
    native_predictive_coverage_digest,
    native_predictive_pair_keys_digest,
    native_predictive_query_schema_digest,
    pool_dino_object_summaries,
    write_predictive_target_cache,
)
from tools.audit_lingbot_predictive_temporal_targets import (
    _feature_rgb,
    _matching_task_annotations,
    _render_visual_panel,
    _select_visual_records,
    audit_predictive_temporal_content,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _owner(source_hash: str) -> CalvinVisibleOwnerRaster:
    owner = np.ones((200, 200), dtype=np.uint8)
    owner[:, 100:] = 2
    return CalvinVisibleOwnerRaster(
        camera_name="static",
        host_image_key="observation.images.image",
        owner_index=owner,
        owner_supervised=np.ones((200, 200), dtype=np.bool_),
        source_rgb_sha256=source_hash,
        source_depth_sha256=_sha(f"depth-{source_hash}"),
        rgb_mae=0.0,
        depth_mae_m=0.0,
        depth_p95_m=0.0,
        depth_consistent_fraction=1.0,
    )


class _Sidecar(CalvinPhysicalSupervisionSidecar):
    def __init__(self, *, manifest_sha256: str, source_hashes: dict[int, str]) -> None:
        self.manifest_sha256 = manifest_sha256
        self.source_hashes = source_hashes

    def source_frame(self, global_index: int) -> CalvinPhysicalSupervisionFrame:
        static = _owner(self.source_hashes[global_index])
        return CalvinPhysicalSupervisionFrame(
            identity_keys=("object/a", "object/b"),
            geometry=torch.zeros(2, 3),
            geometry_variance=torch.zeros(2, 3),
            geometry_supervised=torch.ones(2, 3, dtype=torch.bool),
            geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
            cameras=(static,),
        )


def test_temporal_target_audit_uses_complete_hash_bound_cache_intersection(
    tmp_path: Path,
) -> None:
    sources = (10, 20)
    pairs = ((10, 1), (20, 1))
    dataset = _sha("dataset")
    sidecar_digest = _sha("sidecar")
    stream = _sha("stream")
    temporal = _sha("temporal")
    teacher_config = _sha("teacher-config")
    teacher_checkpoint = _sha("teacher-checkpoint")
    source_hashes = {source: _sha(f"rgb-{source}") for source in sources}
    source_keys = current_grid_source_keys_digest(sources)
    current_contract = CurrentGridCacheContract(
        dataset_id="calvin",
        dataset_revision="fixture",
        split_name="training",
        dataset_tree_sha256=dataset,
        physical_sidecar_manifest_sha256=sidecar_digest,
        lingbot_source_commit="2838c",
        lingbot_checkpoint_revision="released",
        teacher_config_sha256=teacher_config,
        teacher_checkpoint_sha256=teacher_checkpoint,
        stream_plan_sha256=stream,
        temporal_estimator_sha256=temporal,
        source_keys_sha256=source_keys,
        coverage_sha256=current_grid_coverage_digest(
            dataset_tree_sha256=dataset,
            stream_plan_sha256=stream,
            temporal_estimator_sha256=temporal,
            source_keys_sha256=source_keys,
            expected_record_count=len(sources),
        ),
        expected_record_count=len(sources),
    )
    pair_keys = native_predictive_pair_keys_digest(pairs)
    predictive_contract = PredictiveCacheContract(
        dataset_id="calvin",
        dataset_revision="fixture",
        split_name="training",
        dataset_tree_sha256=dataset,
        physical_sidecar_manifest_sha256=sidecar_digest,
        lingbot_source_commit="2838c",
        lingbot_checkpoint_revision="released",
        teacher_config_sha256=teacher_config,
        teacher_checkpoint_sha256=teacher_checkpoint,
        query_schema_sha256=native_predictive_query_schema_digest(
            target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
            route_id=0,
            horizons=(1,),
        ),
        horizons=(1,),
        stream_plan_sha256=stream,
        temporal_estimator_sha256=temporal,
        pair_keys_sha256=pair_keys,
        coverage_sha256=native_predictive_coverage_digest(
            dataset_tree_sha256=dataset,
            stream_plan_sha256=stream,
            temporal_estimator_sha256=temporal,
            pair_keys_sha256=pair_keys,
            expected_record_count=len(pairs),
            horizons=(1,),
        ),
        expected_record_count=len(pairs),
    )
    patch_grid = []
    future_records = []
    for source in sources:
        channel = torch.linspace(0, 2 * torch.pi, 1024)[None, :]
        columns = torch.arange(256).remainder(16)[:, None]
        identity_pattern = torch.where(
            columns < 8,
            torch.sin(channel * 2.0),
            torch.cos(channel * 3.0),
        )
        source_pattern = ((source - 15) / 5.0) * 0.2 * torch.sin(channel * 5.0)
        patches = identity_pattern + source_pattern
        patch_grid.append(
            CurrentGridCacheRecord(source, source_hashes[source], patches.half().numpy())
        )
        camera = _owner(source_hashes[source])
        summaries, importance = pool_dino_object_summaries(
            patches,
            owner_index=camera.owner_index,
            owner_supervised=camera.owner_supervised,
            identity_keys=("object/a", "object/b"),
            minimum_visible_fraction=0.0,
        )
        summaries = summaries.copy()
        summaries[0, ::2] += np.float16(0.5 + source / 100)
        summaries[1, 1::2] -= np.float16(0.25 + source / 200)
        future_records.append(
            PredictiveObjectCacheRecord(
                source_global_index=source,
                target_global_index=source + 1,
                horizon=1,
                source_rgb_sha256=source_hashes[source],
                target_rgb_sha256=_sha(f"rgb-{source + 1}"),
                identity_keys=("object/a", "object/b"),
                features=summaries,
                importance=importance,
            )
        )

    current_root = tmp_path / "current"
    current_manifest = write_current_grid_target_cache(
        current_root,
        contract=current_contract,
        records=tuple(patch_grid),
        shard_rows=1,
    )
    predictive_root = tmp_path / "future"
    predictive_manifest = write_predictive_target_cache(
        predictive_root,
        contract=predictive_contract,
        records=tuple(future_records),
        shard_rows=1,
    )
    current_cache = LingBotCurrentGridTargetCache.load(
        current_root,
        manifest_sha256=current_manifest,
        dataset_tree_sha256=dataset,
        physical_sidecar_manifest_sha256=sidecar_digest,
        encoder_digest=current_contract.encoder_digest,
        coverage_sha256=current_contract.coverage_sha256,
    )
    predictive_cache = LingBotPredictiveTargetCache.load(
        predictive_root,
        manifest_sha256=predictive_manifest,
        dataset_tree_sha256=dataset,
        physical_sidecar_manifest_sha256=sidecar_digest,
        encoder_digest=predictive_contract.encoder_digest,
        query_schema_sha256=predictive_contract.query_schema_sha256,
        coverage_sha256=predictive_contract.coverage_sha256,
    )

    report = audit_predictive_temporal_content(
        predictive_cache,
        current_cache,
        _Sidecar(manifest_sha256=sidecar_digest, source_hashes=source_hashes),
        maximum_samples=4,
    )

    assert report["scanned_current_record_count"] == 2
    assert report["matched_future_record_count"] == 2
    assert report["supported_aligned_pair_count"] == 4
    assert report["horizon_supported_pair_counts"] == {"1": 4}
    assert report["current_correction_supported_object_target_count"] == 4
    assert report["current_correction_identity_count"] == 2
    assert (
        report["interpretation"]["current_correction_pretraining_readiness"] == "PASS"  # type: ignore[index]
    )
    assert report["interpretation"]["pretraining_readiness"] == "PASS"  # type: ignore[index]
    assert report["interpretation"]["scientific_acceptance"] is False  # type: ignore[index]


def test_teacher_cache_visual_selection_covers_observed_horizon_tails() -> None:
    records = tuple(
        PredictiveObjectCacheRecord(
            source_global_index=source,
            target_global_index=source + horizon,
            horizon=horizon,
            source_rgb_sha256=_sha(f"source-{source}"),
            target_rgb_sha256=_sha(f"target-{source + horizon}"),
            identity_keys=("object/a", "object/b"),
            features=np.ones((2, 1024), dtype=np.float16),
            importance=np.asarray((importance, importance / 2), dtype=np.float32),
        )
        for horizon in (1, 2)
        for source, importance in ((10, 0.8), (20, 0.1), (30, 0.5), (40, 0.7))
    )

    selected = _select_visual_records(records, declared_horizons=(1, 2, 64))
    selected_by_horizon = {
        horizon: {
            record.source_global_index: reasons
            for record, reasons in selected
            if record.horizon == horizon
        }
        for horizon in (1, 2, 64)
    }

    assert selected_by_horizon[1] == {
        10: ("first",),
        20: ("minimum_total_visible_importance", "temporal_median"),
        40: ("last",),
    }
    assert selected_by_horizon[2] == selected_by_horizon[1]
    assert selected_by_horizon[64] == {}
    with pytest.raises(ContractError, match="source ordered"):
        _select_visual_records(
            tuple(reversed(records[:4])),
            declared_horizons=(1,),
        )


def test_teacher_cache_visual_panel_is_legible_and_rejects_bad_feature_grid() -> None:
    source_hash = _sha("source-rgb")
    target_hash = _sha("target-rgb")
    owner = _owner(source_hash)
    frame = CalvinPhysicalSupervisionFrame(
        identity_keys=("object/a", "object/b"),
        geometry=torch.zeros(2, 3),
        geometry_variance=torch.zeros(2, 3),
        geometry_supervised=torch.ones(2, 3, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(owner,),
    )
    features = np.linspace(-1, 1, 256 * 1024, dtype=np.float32).reshape(256, 1024)
    current = CurrentGridCacheRecord(
        source_global_index=10,
        source_rgb_sha256=source_hash,
        features=features.astype(np.float16),
    )
    future = PredictiveObjectCacheRecord(
        source_global_index=10,
        target_global_index=12,
        horizon=2,
        source_rgb_sha256=source_hash,
        target_rgb_sha256=target_hash,
        identity_keys=frame.identity_keys,
        features=np.ones((2, 1024), dtype=np.float16),
        importance=np.asarray((0.4, 0.2), dtype=np.float32),
    )
    source_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    target_rgb = np.full((200, 200, 3), 96, dtype=np.uint8)

    panel = _render_visual_panel(
        source_rgb=source_rgb,
        source_frame=frame,
        target_rgb=target_rgb,
        target_frame=frame,
        current_record=current,
        future_record=future,
        task_keys=("move_blue_block",),
        instructions=("move the blue block",),
        selection_reasons=("first",),
    )

    assert panel.width == 960
    assert panel.height > 640
    assert _feature_rgb(current.features).shape == (16, 16, 3)
    with pytest.raises(ContractError, match="square patch grid"):
        _feature_rgb(np.ones((255, 1024), dtype=np.float16))


def test_teacher_cache_visual_task_annotations_preserve_same_task_paraphrases() -> None:
    index = cast(
        Any,
        SimpleNamespace(
            segments=(
                SimpleNamespace(
                    start=10,
                    end=20,
                    task_key="turn_on_lightbulb",
                    instruction="turn on the yellow lamp",
                ),
                SimpleNamespace(
                    start=12,
                    end=22,
                    task_key="turn_on_lightbulb",
                    instruction="toggle the switch",
                ),
            )
        ),
    )

    task_keys, instructions, segments = _matching_task_annotations(
        index,
        source_global_index=15,
        target_global_index=18,
    )

    assert task_keys == ("turn_on_lightbulb",)
    assert instructions == ("toggle the switch", "turn on the yellow lamp")
    assert segments == (
        (10, 20, "turn on the yellow lamp"),
        (12, 22, "toggle the switch"),
    )

    index.segments += (
        SimpleNamespace(
            start=14,
            end=19,
            task_key="move_blue_block",
            instruction="move the blue block",
        ),
    )
    task_keys, instructions, segments = _matching_task_annotations(
        index,
        source_global_index=15,
        target_global_index=18,
    )
    assert task_keys == ("move_blue_block", "turn_on_lightbulb")
    assert instructions == (
        "move the blue block",
        "toggle the switch",
        "turn on the yellow lamp",
    )
    assert segments == (
        (10, 20, "turn on the yellow lamp"),
        (12, 22, "toggle the switch"),
        (14, 19, "move the blue block"),
    )
