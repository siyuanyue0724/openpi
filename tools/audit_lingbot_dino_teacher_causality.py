#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Verify released LingBot DINO current causality and predictive-cache replay."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections.abc import Callable, Sequence
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="teacher causality audit",
)

import numpy as np
import torch

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.current_grid_cache import (
    CurrentGridCacheRecord,
    LingBotCurrentGridTargetCache,
)
from picf_next.lingbot_native.predictive_cache import (
    LingBotPredictiveTargetCache,
    PredictiveObjectCacheRecord,
    pool_dino_object_summaries,
    predictive_effective_fps,
)
from picf_next.lingbot_native.predictive_diagnostics import (
    TEACHER_CAUSALITY_AUDIT_SCHEMA,
    predictive_temporal_diagnostics,
    predictive_temporal_pretraining_readiness,
)

try:
    from tools.bootstrap_lingbot_vla2 import validate_checkpoint
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        verify_native_patch,
    )
    from tools.build_lingbot_calvin_predictive_cache import (
        OfficialLingBotDinoVideoExtractor,
        _rgb_batch,
        _resolve_training_config,
        _VerifiedFrameCache,
        _VerifiedStaticFrame,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import validate_checkpoint  # type: ignore[no-redef]
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        verify_native_patch,
    )
    from build_lingbot_calvin_predictive_cache import (  # type: ignore[no-redef]
        OfficialLingBotDinoVideoExtractor,
        _rgb_batch,
        _resolve_training_config,
        _VerifiedFrameCache,
        _VerifiedStaticFrame,
    )

_SAMPLE_SCHEMA = b"picf-next.lingbot-dino-teacher-causality-sample/v1\0"


def _priority(record: PredictiveObjectCacheRecord) -> int:
    payload = (
        _SAMPLE_SCHEMA
        + record.source_global_index.to_bytes(8, byteorder="big", signed=False)
        + record.horizon.to_bytes(8, byteorder="big", signed=False)
        + record.source_rgb_sha256.encode("ascii")
        + record.target_rgb_sha256.encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big", signed=False)


def select_predictive_records(
    cache: LingBotPredictiveTargetCache,
    *,
    maximum_records: int,
) -> tuple[tuple[PredictiveObjectCacheRecord, ...], int]:
    """Scan complete cache coverage and retain the lowest content-derived priorities."""

    if not isinstance(cache, LingBotPredictiveTargetCache):
        raise TypeError("teacher causality selection requires a predictive cache")
    if (
        isinstance(maximum_records, bool)
        or not isinstance(maximum_records, int)
        or maximum_records < 2
    ):
        raise ValueError("teacher causality audit requires at least two records")
    retained: list[tuple[int, int, PredictiveObjectCacheRecord]] = []
    best_by_horizon: dict[int, tuple[int, int, PredictiveObjectCacheRecord]] = {}
    scanned = 0
    for serial, record in enumerate(cache.iter_records()):
        scanned += 1
        priority = _priority(record)
        item = (-priority, serial, record)
        incumbent = best_by_horizon.get(record.horizon)
        if incumbent is None or priority < -incumbent[0]:
            best_by_horizon[record.horizon] = item
        if len(retained) < maximum_records:
            heapq.heappush(retained, item)
        elif priority < -retained[0][0]:
            heapq.heapreplace(retained, item)
    configured_horizons = set(cache.contract.horizons)
    observed_horizons = set(best_by_horizon)
    if (
        scanned != cache.contract.expected_record_count
        or len(retained) < 2
        or not observed_horizons
        or not observed_horizons.issubset(configured_horizons)
    ):
        raise RuntimeError("teacher causality audit did not scan complete usable coverage")
    if len(observed_horizons) > maximum_records:
        raise ValueError("teacher causality maximum records cannot cover every observed horizon")
    selected = {
        (item[2].source_global_index, item[2].horizon): item for item in best_by_horizon.values()
    }
    for item in sorted(retained, key=lambda value: (-value[0], value[1])):
        if len(selected) >= maximum_records:
            break
        selected.setdefault((item[2].source_global_index, item[2].horizon), item)
    ordered = tuple(
        item[2] for item in sorted(selected.values(), key=lambda value: (-value[0], value[1]))
    )
    return ordered, scanned


def audit_selected_teacher_pairs(
    records: Sequence[PredictiveObjectCacheRecord],
    *,
    frame_for: Callable[[int], _VerifiedStaticFrame],
    current_record_for: Callable[[int], CurrentGridCacheRecord | None],
    extractor: OfficialLingBotDinoVideoExtractor,
    configured_horizons: Sequence[int],
    input_size: int,
    minimum_visible_fraction: float,
    source_fps: float,
    batch_size: int,
) -> dict[str, object]:
    """Recompute one bounded selection without trusting either target cache."""

    if len(records) < 2 or any(
        not isinstance(value, PredictiveObjectCacheRecord) for value in records
    ):
        raise ValueError("teacher causality audit requires typed predictive records")
    if not callable(frame_for):
        raise TypeError("teacher causality audit requires a verified frame loader")
    if not callable(current_record_for):
        raise TypeError("teacher causality audit requires a current-cache record loader")
    if not isinstance(extractor, OfficialLingBotDinoVideoExtractor):
        raise TypeError("teacher causality audit requires the official extractor")
    configured_horizons = tuple(configured_horizons)
    if (
        not configured_horizons
        or configured_horizons != tuple(sorted(set(configured_horizons)))
        or any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
            for horizon in configured_horizons
        )
        or not {record.horizon for record in records}.issubset(configured_horizons)
    ):
        raise ValueError("teacher causality configured horizons are malformed")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("teacher causality batch size must be positive")

    current_patch_elements = 0
    current_cache_patch_elements = 0
    future_feature_elements = 0
    future_importance_elements = 0
    current_patch_mismatch_count = 0
    current_cache_patch_mismatch_count = 0
    future_feature_mismatch_count = 0
    future_importance_mismatch_count = 0
    maximum_current_patch_absolute_error = 0.0
    maximum_current_cache_patch_absolute_error = 0.0
    maximum_future_feature_absolute_error = 0.0
    maximum_future_importance_absolute_error = 0.0
    selection_digest = hashlib.sha256()
    temporal_current: list[np.ndarray] = []
    temporal_future: list[np.ndarray] = []
    temporal_identities: list[str] = []
    temporal_horizons: list[int] = []

    for start in range(0, len(records), batch_size):
        batch = tuple(records[start : start + batch_size])
        sources = tuple(frame_for(record.source_global_index) for record in batch)
        targets = tuple(frame_for(record.target_global_index) for record in batch)
        for record, source, target in zip(batch, sources, targets, strict=True):
            if (
                source.global_index != record.source_global_index
                or target.global_index != record.target_global_index
                or source.rgb_sha256 != record.source_rgb_sha256
                or target.rgb_sha256 != record.target_rgb_sha256
            ):
                raise ContractError("teacher causality sample differs from cache RGB provenance")
            selection_digest.update(_priority(record).to_bytes(32, byteorder="big"))

        current_rgb = _rgb_batch(sources)
        future_rgb = _rgb_batch(targets)
        effective_fps = predictive_effective_fps(
            torch.tensor(tuple(record.horizon for record in batch), dtype=torch.long),
            source_fps=source_fps,
        )
        future_patch, same_call_current_patch = extractor.paired(
            current_rgb,
            future_rgb,
            effective_fps=effective_fps,
        )
        current_only_patch = extractor.current(current_rgb, effective_fps=effective_fps)
        current_cache_replay_patch = extractor.current(current_rgb)
        expected_shape = (len(batch), 256, 1024)
        tensors = (
            future_patch,
            same_call_current_patch,
            current_only_patch,
            current_cache_replay_patch,
        )
        if any(
            value.shape != expected_shape
            or not value.is_floating_point()
            or value.requires_grad
            or value.grad_fn is not None
            or not torch.isfinite(value).all()
            for value in tensors
        ):
            raise ContractError("released teacher returned malformed causal-audit patches")

        current_error = (same_call_current_patch.float() - current_only_patch.float()).abs()
        current_patch_elements += current_error.numel()
        current_patch_mismatch_count += int(torch.count_nonzero(current_error))
        maximum_current_patch_absolute_error = max(
            maximum_current_patch_absolute_error,
            float(current_error.max()),
        )

        replay_values = (
            current_cache_replay_patch.detach().cpu().float().numpy().astype(np.float16, copy=False)
        )
        for index, (record, source) in enumerate(zip(batch, sources, strict=True)):
            current_record = current_record_for(record.source_global_index)
            if current_record is None:
                raise ContractError("current cache omits a sampled predictive source frame")
            if (
                current_record.source_global_index != source.global_index
                or current_record.source_rgb_sha256 != source.rgb_sha256
            ):
                raise ContractError("current cache differs from sampled source RGB provenance")
            if current_record.features.shape != expected_shape[1:]:
                raise ContractError("current cache differs from released teacher geometry")
            cache_error = np.abs(
                replay_values[index].astype(np.float32) - current_record.features.astype(np.float32)
            )
            current_cache_patch_elements += cache_error.size
            current_cache_patch_mismatch_count += int(np.count_nonzero(cache_error))
            maximum_current_cache_patch_absolute_error = max(
                maximum_current_cache_patch_absolute_error,
                float(cache_error.max(initial=0.0)),
            )

        for index, (record, target) in enumerate(zip(batch, targets, strict=True)):
            source = sources[index]
            current_features, current_importance = pool_dino_object_summaries(
                same_call_current_patch[index],
                owner_index=source.camera.owner_index,
                owner_supervised=source.camera.owner_supervised,
                identity_keys=source.physical.identity_keys,
                minimum_visible_fraction=minimum_visible_fraction,
                input_size=input_size,
            )
            recomputed_features, recomputed_importance = pool_dino_object_summaries(
                future_patch[index],
                owner_index=target.camera.owner_index,
                owner_supervised=target.camera.owner_supervised,
                identity_keys=target.physical.identity_keys,
                minimum_visible_fraction=minimum_visible_fraction,
                input_size=input_size,
            )
            if record.identity_keys != target.physical.identity_keys:
                raise ContractError("predictive cache identities differ from target sidecar")
            feature_error = np.abs(
                recomputed_features.astype(np.float32) - record.features.astype(np.float32)
            )
            importance_error = np.abs(recomputed_importance - record.importance)
            future_feature_elements += feature_error.size
            future_importance_elements += importance_error.size
            future_feature_mismatch_count += int(np.count_nonzero(feature_error))
            future_importance_mismatch_count += int(np.count_nonzero(importance_error))
            maximum_future_feature_absolute_error = max(
                maximum_future_feature_absolute_error,
                float(feature_error.max(initial=0.0)),
            )
            maximum_future_importance_absolute_error = max(
                maximum_future_importance_absolute_error,
                float(importance_error.max(initial=0.0)),
            )
            current_positions = {
                identity: position
                for position, identity in enumerate(source.physical.identity_keys)
            }
            for future_position, identity in enumerate(record.identity_keys):
                current_position = current_positions.get(identity)
                if (
                    current_position is None
                    or float(current_importance[current_position]) <= 0
                    or float(recomputed_importance[future_position]) <= 0
                ):
                    continue
                temporal_current.append(
                    current_features[current_position].astype(np.float32, copy=True)
                )
                temporal_future.append(
                    recomputed_features[future_position].astype(np.float32, copy=True)
                )
                temporal_identities.append(identity)
                temporal_horizons.append(record.horizon)

    if len(temporal_current) < 2:
        raise RuntimeError("teacher causality audit found insufficient aligned temporal support")
    temporal_diagnostics = predictive_temporal_diagnostics(
        torch.from_numpy(np.stack(temporal_current)),
        torch.from_numpy(np.stack(temporal_future)),
        identity_keys=temporal_identities,
        horizons=temporal_horizons,
    )
    temporal_ready, temporal_failures = predictive_temporal_pretraining_readiness(
        temporal_diagnostics
    )

    exact = (
        current_patch_mismatch_count == 0
        and current_cache_patch_mismatch_count == 0
        and future_feature_mismatch_count == 0
        and future_importance_mismatch_count == 0
        and temporal_ready
    )
    return {
        "current_cache_patch_elements": current_cache_patch_elements,
        "current_cache_patch_mismatch_count": current_cache_patch_mismatch_count,
        "current_patch_elements": current_patch_elements,
        "current_patch_mismatch_count": current_patch_mismatch_count,
        "future_feature_elements": future_feature_elements,
        "future_feature_mismatch_count": future_feature_mismatch_count,
        "future_importance_elements": future_importance_elements,
        "future_importance_mismatch_count": future_importance_mismatch_count,
        "maximum_current_cache_patch_absolute_error": (maximum_current_cache_patch_absolute_error),
        "maximum_current_patch_absolute_error": maximum_current_patch_absolute_error,
        "maximum_future_feature_absolute_error": maximum_future_feature_absolute_error,
        "maximum_future_importance_absolute_error": maximum_future_importance_absolute_error,
        "sample_selection_sha256": selection_digest.hexdigest(),
        "sampled_horizon_record_counts": {
            str(horizon): sum(record.horizon == horizon for record in records)
            for horizon in configured_horizons
        },
        "sampled_record_count": len(records),
        "same_call_supported_pair_count": len(temporal_current),
        "same_call_temporal_diagnostics": temporal_diagnostics.as_dict(),
        "same_call_temporal_pretraining_readiness": "PASS" if temporal_ready else "FAIL",
        "same_call_temporal_pretraining_readiness_failures": list(temporal_failures),
        "status": "PASS" if exact else "FAIL",
    }


def _write_json_durable(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    write_bytes_durable_exclusive(path, payload)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, default=root / CHECKOUT_RELATIVE_PATH)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--dataset-split", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--physical-sidecar-root", required=True, type=Path)
    parser.add_argument("--physical-sidecar-manifest", type=Path)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--current-cache-root", required=True, type=Path)
    parser.add_argument("--current-cache-manifest-sha256", required=True)
    parser.add_argument("--current-coverage-sha256", required=True)
    parser.add_argument("--current-encoder-digest", required=True)
    parser.add_argument("--predictive-cache-root", required=True, type=Path)
    parser.add_argument("--predictive-cache-manifest-sha256", required=True)
    parser.add_argument("--predictive-query-schema-sha256", required=True)
    parser.add_argument("--predictive-coverage-sha256", required=True)
    parser.add_argument("--predictive-encoder-digest", required=True)
    parser.add_argument("--maximum-records", default=16, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--memory-capacity", default=1, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.maximum_records < 2 or args.batch_size <= 0 or args.memory_capacity <= 0:
        raise ValueError("teacher causality audit counts are outside their valid range")
    training_config = _resolve_training_config(args.source_checkout, args.training_config)
    root = Path(__file__).resolve().parents[1]
    patch = verify_native_patch(root=root, checkout=args.source_checkout, check_apply=True)
    validate_checkpoint(args.checkpoint_dir)
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.resolve().name,
    )
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(
        args.physical_sidecar_root,
        index,
        manifest_path=args.physical_sidecar_manifest,
        expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
    )
    current_cache = LingBotCurrentGridTargetCache.load(
        args.current_cache_root,
        manifest_sha256=args.current_cache_manifest_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
        physical_sidecar_manifest_sha256=sidecar.manifest_sha256,
        encoder_digest=args.current_encoder_digest,
        coverage_sha256=args.current_coverage_sha256,
        memory_capacity=args.memory_capacity,
    )
    cache = LingBotPredictiveTargetCache.load(
        args.predictive_cache_root,
        manifest_sha256=args.predictive_cache_manifest_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
        physical_sidecar_manifest_sha256=sidecar.manifest_sha256,
        encoder_digest=args.predictive_encoder_digest,
        query_schema_sha256=args.predictive_query_schema_sha256,
        coverage_sha256=args.predictive_coverage_sha256,
        memory_capacity=args.memory_capacity,
    )
    if (
        cache.contract.lingbot_source_commit != LINGBOT_NATIVE_SOURCE_COMMIT
        or current_cache.contract.lingbot_source_commit != LINGBOT_NATIVE_SOURCE_COMMIT
    ):
        raise ContractError("teacher causality cache uses another LingBot source commit")
    if (
        current_cache.contract.lingbot_checkpoint_revision
        != cache.contract.lingbot_checkpoint_revision
        or current_cache.contract.teacher_config_sha256 != cache.contract.teacher_config_sha256
        or current_cache.contract.teacher_checkpoint_sha256
        != cache.contract.teacher_checkpoint_sha256
        or current_cache.contract.input_size != cache.contract.input_size
        or current_cache.contract.patch_tokens != cache.contract.patch_tokens
        or current_cache.contract.hidden_size != cache.contract.hidden_size
    ):
        raise ContractError("current and predictive caches use different released teachers")
    records, scanned = select_predictive_records(cache, maximum_records=args.maximum_records)
    frame_cache = _VerifiedFrameCache(index, sidecar, capacity=max(2, 2 * args.batch_size))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("released teacher causality audit requires one CUDA device")
    extractor = OfficialLingBotDinoVideoExtractor(
        source_checkout=args.source_checkout,
        checkpoint_dir=args.checkpoint_dir,
        training_config=training_config,
        device=device,
    )
    diagnostics = audit_selected_teacher_pairs(
        records,
        frame_for=frame_cache.get,
        current_record_for=lambda source: current_cache.record_for(source_global_index=source),
        extractor=extractor,
        configured_horizons=cache.contract.horizons,
        input_size=cache.contract.input_size,
        minimum_visible_fraction=cache.contract.minimum_visible_fraction,
        source_fps=cache.contract.source_fps,
        batch_size=args.batch_size,
    )
    report = {
        "dataset_tree_sha256": manifest.tree_sha256,
        "diagnostics": diagnostics,
        "current_cache_manifest_sha256": current_cache.manifest_sha256,
        "current_encoder_digest": current_cache.contract.encoder_digest,
        "patch_sha256": patch["patch_sha256"],
        "physical_sidecar_manifest_sha256": sidecar.manifest_sha256,
        "predictive_cache_manifest_sha256": cache.manifest_sha256,
        "predictive_encoder_digest": cache.contract.encoder_digest,
        "scanned_record_count": scanned,
        "schema": TEACHER_CAUSALITY_AUDIT_SCHEMA,
    }
    _write_json_durable(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    if diagnostics["status"] != "PASS":
        raise RuntimeError("released teacher causality or cache replay failed")


if __name__ == "__main__":
    main()
