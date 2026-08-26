#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build the exact plan-bound current-frame DINO patch bank for ADR-74."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Protocol

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="current-grid cache builder",
)

import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_physical_visual_acceptance import (
    load_calvin_physical_visual_acceptance,
)
from picf_next.data.calvin_official_source import (
    validate_calvin_content_identity_migration,
    validate_calvin_official_source_receipt,
)
from picf_next.data.dataset_manifest import (
    file_sha256,
    load_dataset_file_manifest,
    read_sha256_verified_file_beneath,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.calvin import (
    build_native_calvin_physical_stream_plan,
    build_native_calvin_training_stream_plan,
)
from picf_next.lingbot_native.current_grid_cache import (
    CurrentGridCacheContract,
    CurrentGridCacheRecord,
    LingBotCurrentGridTargetCache,
    rebind_current_grid_target_cache,
    write_current_grid_target_cache,
)
from picf_next.lingbot_native.predictive_plan import (
    NativeCurrentGridCoveragePlan,
    build_native_current_grid_coverage_plan,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    RepresentationTrialSplit,
)
from picf_next.lingbot_native.stream_plan import (
    add_reset_mixture_arguments,
    reset_mixture_values,
    validate_stream_optimizer_lag,
)
from picf_next.lingbot_native.temporal import TemporalEstimatorConfig

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_REVISION,
        validate_checkpoint,
    )
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        verify_native_patch,
    )
    from tools.build_lingbot_calvin_predictive_cache import (
        OfficialLingBotDinoVideoExtractor,
        _rgb_batch,
        _resolve_training_config,
        _sha256,
        _VerifiedFrameCache,
        _VerifiedStaticFrame,
        _video_config,
        _write_build_report,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_CHECKPOINT_REVISION,
        validate_checkpoint,
    )
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        verify_native_patch,
    )
    from build_lingbot_calvin_predictive_cache import (  # type: ignore[no-redef]
        OfficialLingBotDinoVideoExtractor,
        _rgb_batch,
        _resolve_training_config,
        _sha256,
        _VerifiedFrameCache,
        _VerifiedStaticFrame,
        _video_config,
        _write_build_report,
    )


class CurrentPatchExtractor(Protocol):
    def current(self, current_rgb: torch.Tensor) -> torch.Tensor: ...


def _extract_current_batch(
    frames: Sequence[_VerifiedStaticFrame],
    *,
    extractor: CurrentPatchExtractor,
    contract: CurrentGridCacheContract,
) -> tuple[CurrentGridCacheRecord, ...]:
    if not frames:
        raise ValueError("current-grid extraction batch cannot be empty")
    patches = extractor.current(_rgb_batch(frames))
    expected = (len(frames), contract.patch_tokens, contract.hidden_size)
    if (
        not isinstance(patches, torch.Tensor)
        or patches.shape != expected
        or not patches.is_floating_point()
        or patches.requires_grad
        or patches.grad_fn is not None
        or not torch.isfinite(patches).all()
    ):
        raise ContractError(
            f"official DINO current teacher output differs from {expected} detached finite tokens"
        )
    values = patches.detach().cpu().float().numpy().astype("float16", copy=False)
    return tuple(
        CurrentGridCacheRecord(
            source_global_index=frame.global_index,
            source_rgb_sha256=frame.rgb_sha256,
            features=values[index],
        )
        for index, frame in enumerate(frames)
    )


def iter_calvin_current_grid_records(
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    *,
    extractor: CurrentPatchExtractor,
    contract: CurrentGridCacheContract,
    coverage: NativeCurrentGridCoveragePlan,
    batch_size: int,
    frame_cache_capacity: int,
    donor_cache: LingBotCurrentGridTargetCache | None = None,
    donor_content_identity_verified: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
) -> Iterator[CurrentGridCacheRecord]:
    if not isinstance(index, CalvinDatasetIndex) or not isinstance(
        sidecar, CalvinPhysicalSupervisionSidecar
    ):
        raise TypeError("current-grid extraction requires verified CALVIN data")
    if not isinstance(contract, CurrentGridCacheContract) or not isinstance(
        coverage, NativeCurrentGridCoveragePlan
    ):
        raise TypeError("current-grid extraction requires typed contract and coverage")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("current-grid extraction batch size must be positive")
    if sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise ContractError("current-grid extraction requires all-source physical supervision")
    if (
        index.dataset_manifest is None
        or index.dataset_manifest.tree_sha256 != contract.dataset_tree_sha256
        or sidecar.manifest_sha256 != contract.physical_sidecar_manifest_sha256
        or coverage.coverage_sha256 != contract.coverage_sha256
        or coverage.source_keys_sha256 != contract.source_keys_sha256
    ):
        raise ContractError("current-grid extraction provenance differs from contract")
    if donor_cache is not None:
        if not isinstance(donor_cache, LingBotCurrentGridTargetCache):
            raise TypeError("current-grid donor must be one verified source bank")
        donor_contract = donor_cache.contract
        _validate_donor_cache_semantics(
            donor_contract,
            contract,
            content_identity_verified=donor_content_identity_verified,
        )
    cache = _VerifiedFrameCache(index, sidecar, capacity=frame_cache_capacity)
    pending: list[_VerifiedStaticFrame] = []
    ordered: deque[CurrentGridCacheRecord | _VerifiedStaticFrame] = deque()
    resolved: dict[int, CurrentGridCacheRecord] = {}
    completed = 0
    started = time.perf_counter()

    def drain_ready() -> Iterator[CurrentGridCacheRecord]:
        nonlocal completed
        while ordered:
            item = ordered[0]
            if isinstance(item, CurrentGridCacheRecord):
                record = item
            else:
                record = resolved.get(item.global_index)
                if record is None:
                    break
                del resolved[item.global_index]
            ordered.popleft()
            completed += 1
            yield record
            if progress is not None:
                progress(
                    completed,
                    contract.expected_record_count,
                    time.perf_counter() - started,
                )

    for source_index in coverage.source_global_indices:
        donated = (
            None
            if donor_cache is None
            else donor_cache.record_for(source_global_index=source_index)
        )
        if donated is not None:
            frame = cache.get(source_index)
            if donated.source_rgb_sha256 != frame.rgb_sha256:
                raise ContractError("current-grid donor RGB identity differs from source frame")
            ordered.append(donated)
        else:
            frame = cache.get(source_index)
            ordered.append(frame)
            pending.append(frame)
            if len(pending) == batch_size:
                for record in _extract_current_batch(
                    pending,
                    extractor=extractor,
                    contract=contract,
                ):
                    if record.source_global_index in resolved:
                        raise RuntimeError("current-grid extraction resolved one source twice")
                    resolved[record.source_global_index] = record
                pending.clear()
        yield from drain_ready()
    if pending:
        for record in _extract_current_batch(pending, extractor=extractor, contract=contract):
            if record.source_global_index in resolved:
                raise RuntimeError("current-grid extraction resolved one source twice")
            resolved[record.source_global_index] = record
        pending.clear()
    yield from drain_ready()
    if ordered or resolved or completed != contract.expected_record_count:
        raise RuntimeError("current-grid extraction did not traverse complete coverage")
    if progress is not None:
        progress(completed, contract.expected_record_count, time.perf_counter() - started)


def _validate_donor_cache_semantics(
    donor: CurrentGridCacheContract,
    target: CurrentGridCacheContract,
    *,
    content_identity_verified: bool,
) -> None:
    """Keep cross-identity reuse explicit while preserving frozen-teacher equality."""

    if not isinstance(content_identity_verified, bool):
        raise TypeError("current-grid donor content identity flag must be boolean")
    identity_matches = (
        donor.dataset_tree_sha256 == target.dataset_tree_sha256
        and donor.physical_sidecar_manifest_sha256 == target.physical_sidecar_manifest_sha256
    )
    if not identity_matches and not content_identity_verified:
        raise ContractError("current-grid donor source identity differs from contract")
    if (
        donor.encoder_digest != target.encoder_digest
        or donor.input_size != target.input_size
        or donor.patch_tokens != target.patch_tokens
        or donor.hidden_size != target.hidden_size
        or donor.camera_name != target.camera_name
    ):
        raise ContractError("current-grid donor frozen-teacher semantics differ from contract")


def _build_contract(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    coverage: NativeCurrentGridCoveragePlan,
    checkpoint_dir: Path,
) -> CurrentGridCacheContract:
    if index.dataset_manifest is None:
        raise ContractError("current-grid cache requires a content-addressed dataset")
    return CurrentGridCacheContract(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        split_name=index.split_root.name,
        dataset_tree_sha256=index.dataset_manifest.tree_sha256,
        physical_sidecar_manifest_sha256=sidecar.manifest_sha256,
        lingbot_source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
        lingbot_checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
        teacher_config_sha256=_sha256(checkpoint_dir / "dino_video/config.yaml"),
        teacher_checkpoint_sha256=_sha256(checkpoint_dir / "dino_video/teacher_step_10000.pth"),
        stream_plan_sha256=coverage.stream_plan_sha256,
        temporal_estimator_sha256=coverage.temporal_estimator_sha256,
        source_keys_sha256=coverage.source_keys_sha256,
        coverage_sha256=coverage.coverage_sha256,
        expected_record_count=len(coverage.source_global_indices),
    )


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, default=root / CHECKOUT_RELATIVE_PATH)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest", type=Path)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--physical-visual-acceptance", type=Path, required=True)
    parser.add_argument("--physical-visual-acceptance-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--plan-seed", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--local-bptt-probability", type=float, required=True)
    parser.add_argument("--overshoot-probability", type=float, required=True)
    parser.add_argument("--source-mask-probability", type=float, required=True)
    parser.add_argument("--maximum-optimizer-lag", type=int, required=True)
    parser.add_argument("--lane-interleave-factor", type=int, default=1)
    parser.add_argument("--physical-event-stream", action="store_true")
    add_reset_mixture_arguments(parser)
    parser.add_argument("--representation-split", type=Path)
    parser.add_argument("--representation-split-sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-rows", type=int, default=512)
    parser.add_argument("--frame-cache-capacity", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--donor-cache-root", type=Path)
    parser.add_argument("--donor-cache-manifest-sha256")
    parser.add_argument("--rebind-exact-donor", action="store_true")
    parser.add_argument("--donor-content-manifest", type=Path)
    parser.add_argument("--donor-official-source-receipt", type=Path)
    parser.add_argument("--donor-official-source-receipt-sha256")
    return parser.parse_args()


def _temporal_config(args: argparse.Namespace) -> TemporalEstimatorConfig:
    return TemporalEstimatorConfig(
        local_bptt_probability=args.local_bptt_probability,
        overshoot_probability=args.overshoot_probability,
        source_mask_probability=args.source_mask_probability,
        maximum_optimizer_lag=args.maximum_optimizer_lag,
    )


def _validate_builder_args(
    args: argparse.Namespace,
    *,
    reset_mixture: tuple[int, int] | None = None,
) -> None:
    """Validate resource sizes without coupling bank coverage to an objective branch."""

    for name in (
        "batch_size",
        "shard_rows",
        "frame_cache_capacity",
        "progress_every",
        "global_batch_size",
        "total_steps",
        "lane_interleave_factor",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    validate_stream_optimizer_lag(
        reset_mixture=reset_mixture,
        lane_interleave_factor=args.lane_interleave_factor,
        maximum_optimizer_lag=args.maximum_optimizer_lag,
    )
    if (args.representation_split is None) != (args.representation_split_sha256 is None):
        raise ValueError("representation split path and SHA-256 must be provided together")
    if (args.donor_cache_root is None) != (args.donor_cache_manifest_sha256 is None):
        raise ValueError("donor cache root and manifest SHA-256 must be provided together")
    identity_inputs = (
        args.donor_content_manifest,
        args.donor_official_source_receipt,
        args.donor_official_source_receipt_sha256,
    )
    supplied_identity_inputs = tuple(value is not None for value in identity_inputs)
    if any(supplied_identity_inputs) and not all(supplied_identity_inputs):
        raise ValueError("content-identical donor identity inputs must be provided together")
    if all(supplied_identity_inputs) and args.donor_cache_root is None:
        raise ValueError("content-identical donor identity requires one donor cache")
    if args.rebind_exact_donor and (
        args.donor_cache_root is None or not all(supplied_identity_inputs)
    ):
        raise ValueError(
            "exact donor rebind requires donor cache, content manifest, and source receipt"
        )
    if args.physical_event_stream and reset_mixture is not None:
        raise ValueError("the ADR149 physical stream does not use a synthetic reset mixture")


def _load_content_identical_donor(
    args: argparse.Namespace,
    *,
    target_manifest: object,
    target_contract: CurrentGridCacheContract,
) -> tuple[LingBotCurrentGridTargetCache, dict[str, object]]:
    """Authenticate one content-identical source bank without weakening runtime loads."""

    source_manifest_path = args.donor_content_manifest.resolve()
    target_manifest_path = args.dataset_manifest.resolve()
    source_manifest_sha256 = file_sha256(source_manifest_path)
    target_manifest_sha256 = file_sha256(target_manifest_path)
    source_manifest = load_dataset_file_manifest(source_manifest_path)
    if file_sha256(source_manifest_path) != source_manifest_sha256:
        raise ContractError("current-grid donor content manifest changed while loading")
    validate_calvin_content_identity_migration(source_manifest, target_manifest)

    receipt_path = args.donor_official_source_receipt.resolve()
    receipt_bytes = read_sha256_verified_file_beneath(
        receipt_path.parent,
        receipt_path.name,
        expected_sha256=args.donor_official_source_receipt_sha256,
        maximum_bytes=4 * 1024 * 1024,
    )
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("current-grid donor source receipt is not valid JSON") from error
    if not isinstance(receipt, dict):
        raise ContractError("current-grid donor source receipt must be a mapping")
    validate_calvin_official_source_receipt(
        receipt,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
    )

    donor_root = args.donor_cache_root.resolve()
    donor_manifest_bytes = read_sha256_verified_file_beneath(
        donor_root,
        "manifest.json",
        expected_sha256=args.donor_cache_manifest_sha256,
        maximum_bytes=32 * 1024 * 1024,
    )
    try:
        donor_manifest = json.loads(donor_manifest_bytes)
        donor_contract = CurrentGridCacheContract.from_mapping(donor_manifest["contract"])
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("current-grid donor manifest is malformed") from error
    if (
        donor_contract.dataset_id != source_manifest.dataset_id
        or donor_contract.dataset_revision != source_manifest.dataset_revision
        or donor_contract.split_name != source_manifest.split_name
        or donor_contract.dataset_tree_sha256 != source_manifest.tree_sha256
    ):
        raise ContractError("current-grid donor cache differs from its content manifest")
    if donor_contract.encoder_digest != target_contract.encoder_digest:
        raise ContractError("current-grid donor and target use different frozen teachers")
    donor_cache = LingBotCurrentGridTargetCache.load(
        donor_root,
        manifest_sha256=args.donor_cache_manifest_sha256,
        dataset_tree_sha256=donor_contract.dataset_tree_sha256,
        physical_sidecar_manifest_sha256=(donor_contract.physical_sidecar_manifest_sha256),
        encoder_digest=donor_contract.encoder_digest,
        coverage_sha256=donor_contract.coverage_sha256,
    )
    return donor_cache, {
        "donor_cache_manifest_sha256": args.donor_cache_manifest_sha256,
        "donor_content_manifest_sha256": source_manifest_sha256,
        "official_source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "target_dataset_manifest_sha256": target_manifest_sha256,
        "stable_inputs": (
            (source_manifest_path, source_manifest_sha256),
            (target_manifest_path, target_manifest_sha256),
            (receipt_path, hashlib.sha256(receipt_bytes).hexdigest()),
        ),
    }


def main() -> None:
    args = _parse_args()
    reset_mixture = reset_mixture_values(args)
    _validate_builder_args(args, reset_mixture=reset_mixture)
    temporal = _temporal_config(args)
    training_config = _resolve_training_config(args.source_checkout, args.training_config)
    root = Path(__file__).resolve().parents[1]
    patch = verify_native_patch(root=root, checkout=args.source_checkout, check_apply=True)
    actual_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.source_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise ContractError("current-grid teacher source differs from pinned LingBot commit")
    validate_checkpoint(args.checkpoint_dir)
    _video_config(training_config)
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
    load_calvin_physical_visual_acceptance(
        args.physical_visual_acceptance,
        expected_sha256=args.physical_visual_acceptance_sha256,
        expected_dataset_manifest_sha256=_sha256(args.dataset_manifest),
        expected_sidecar_manifest_sha256=sidecar.manifest_sha256,
    )
    dataset = (
        CalvinPhysicalTransitionDataset(index, action_horizon=1)
        if args.physical_event_stream
        else CalvinStatefulTransitionDataset(index, action_horizon=1)
    )
    representation_split: RepresentationTrialSplit | None = None
    if args.representation_split is not None:
        if _sha256(args.representation_split) != args.representation_split_sha256:
            raise ValueError("representation split file SHA-256 differs")
        representation_split = RepresentationTrialSplit.load(args.representation_split)
    excluded_source_episode_indices = (
        representation_split.evaluation_source_episode_indices
        if representation_split is not None
        and representation_split.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
        else ()
    )
    stream = (
        build_native_calvin_physical_stream_plan(
            dataset,
            comparison_id=args.comparison_id,
            seed=args.plan_seed,
            global_batch_size=args.global_batch_size,
            total_steps=args.total_steps,
            lane_interleave_factor=args.lane_interleave_factor,
            excluded_source_episode_indices=excluded_source_episode_indices,
        )
        if args.physical_event_stream
        else build_native_calvin_training_stream_plan(
            dataset,
            comparison_id=args.comparison_id,
            seed=args.plan_seed,
            global_batch_size=args.global_batch_size,
            total_steps=args.total_steps,
            lane_interleave_factor=args.lane_interleave_factor,
            excluded_source_episode_indices=excluded_source_episode_indices,
            reset_numerator=(None if reset_mixture is None else reset_mixture[0]),
            reset_denominator=(None if reset_mixture is None else reset_mixture[1]),
        )
    )
    if (
        representation_split is not None
        and representation_split.stream_plan_sha256 != stream.plan_sha256
    ):
        raise ValueError("representation split differs from current-grid cache stream")
    coverage = build_native_current_grid_coverage_plan(
        stream,
        temporal,
        source_global_index_for_sample=dataset.source_global_index_by_key,
        required_future_offsets=((1,) if args.physical_event_stream else ()),
    )
    contract = _build_contract(
        index=index,
        sidecar=sidecar,
        coverage=coverage,
        checkpoint_dir=args.checkpoint_dir,
    )
    donor_cache = None
    donor_identity_provenance: dict[str, object] | None = None
    if args.donor_cache_root is not None:
        if args.donor_content_manifest is not None:
            donor_cache, donor_identity_provenance = _load_content_identical_donor(
                args,
                target_manifest=manifest,
                target_contract=contract,
            )
        else:
            donor_cache = LingBotCurrentGridTargetCache.load_reusable_source_bank(
                args.donor_cache_root,
                manifest_sha256=args.donor_cache_manifest_sha256,
                dataset_tree_sha256=contract.dataset_tree_sha256,
                physical_sidecar_manifest_sha256=(contract.physical_sidecar_manifest_sha256),
                encoder_digest=contract.encoder_digest,
            )
    reused_record_count = (
        0
        if donor_cache is None
        else sum(
            source_index in donor_cache.locator for source_index in coverage.source_global_indices
        )
    )
    last_reported = -1

    def progress(completed: int, total: int, elapsed: float) -> None:
        nonlocal last_reported
        if completed != total and completed - last_reported < args.progress_every:
            return
        last_reported = completed
        rate = completed / elapsed if elapsed > 0 else 0.0
        print(
            json.dumps(
                {
                    "completed": completed,
                    "elapsed_s": elapsed,
                    "records_per_second": rate,
                    "remaining_s": (total - completed) / rate if rate > 0 else None,
                    "total": total,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.rebind_exact_donor:
        if donor_cache is None or donor_identity_provenance is None:
            raise RuntimeError("exact current-grid rebind lost its authenticated donor")
        verified_frames = _VerifiedFrameCache(
            index,
            sidecar,
            capacity=args.frame_cache_capacity,
        )
        manifest_sha256 = rebind_current_grid_target_cache(
            args.output_root,
            source_cache=donor_cache,
            contract=contract,
            source_rgb_sha256_for=lambda source_index: verified_frames.get(source_index).rgb_sha256,
        )
    else:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("released DINO current-grid extraction requires one CUDA device")
        extractor = OfficialLingBotDinoVideoExtractor(
            source_checkout=args.source_checkout,
            checkpoint_dir=args.checkpoint_dir,
            training_config=training_config,
            device=device,
        )
        records = iter_calvin_current_grid_records(
            index,
            sidecar,
            extractor=extractor,
            contract=contract,
            coverage=coverage,
            batch_size=args.batch_size,
            frame_cache_capacity=args.frame_cache_capacity,
            donor_cache=donor_cache,
            donor_content_identity_verified=donor_identity_provenance is not None,
            progress=progress,
        )
        manifest_sha256 = write_current_grid_target_cache(
            args.output_root,
            contract=contract,
            records=records,
            shard_rows=args.shard_rows,
        )
    report = {
        "cache_manifest_sha256": manifest_sha256,
        "coverage_sha256": contract.coverage_sha256,
        "expected_record_count": contract.expected_record_count,
        "output_root": str(args.output_root.resolve()),
        "patch_sha256": patch["patch_sha256"],
        "physical_visual_acceptance_sha256": args.physical_visual_acceptance_sha256,
        "source_keys_sha256": contract.source_keys_sha256,
        "stream_plan_sha256": contract.stream_plan_sha256,
        "teacher_encoder_digest": contract.encoder_digest,
        "temporal_estimator_sha256": contract.temporal_estimator_sha256,
    }
    if donor_identity_provenance is not None:
        report["content_identical_donor"] = {
            key: value for key, value in donor_identity_provenance.items() if key != "stable_inputs"
        }
        report["content_identical_donor"]["reused_record_count"] = reused_record_count
    print(
        json.dumps(
            {
                "cache_source_reuse": {
                    "donor_cache_manifest_sha256": args.donor_cache_manifest_sha256,
                    "extracted_record_count": contract.expected_record_count - reused_record_count,
                    "physical_event_stream": args.physical_event_stream,
                    "reused_record_count": reused_record_count,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    resolved_output = args.output_root.resolve()
    report_path = resolved_output.parent / f"{resolved_output.name}.build_report.json"
    if donor_identity_provenance is not None:
        stable_inputs = donor_identity_provenance["stable_inputs"]
        if not isinstance(stable_inputs, tuple):
            raise RuntimeError("current-grid donor stable-input ledger is malformed")
        for path, expected_sha256 in stable_inputs:
            if file_sha256(path) != expected_sha256:
                raise ContractError("current-grid donor identity input changed during publication")
    if args.rebind_exact_donor:
        if donor_identity_provenance is None or donor_cache is None:
            raise RuntimeError("current-grid rebind lost donor provenance")
        rebind_receipt = {
            "schema": "picf-next.current-grid-content-identity-rebind/v1",
            **{
                key: value
                for key, value in donor_identity_provenance.items()
                if key != "stable_inputs"
            },
            "target_cache_manifest_sha256": manifest_sha256,
            "target_coverage_sha256": contract.coverage_sha256,
            "target_source_keys_sha256": contract.source_keys_sha256,
            "verified_source_rgb_count": contract.expected_record_count,
            "all_source_rgb_sha256_matches": True,
            "frozen_teacher_unchanged": True,
            "shard_publication": ("external-content-addressed-store-after-full-authentication"),
            "target_shard_store_root": str(donor_cache.root.resolve()),
            "training_authorized": False,
        }
        _write_build_report(
            resolved_output.parent / f"{resolved_output.name}.rebind_receipt.json",
            rebind_receipt,
        )
    _write_build_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
