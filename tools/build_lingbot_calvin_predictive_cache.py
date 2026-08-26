#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build exact plan-bound loss-only CALVIN DINO-video targets for ADR-74."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="predictive cache builder",
)

import numpy as np
import torch

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
    CalvinVisibleOwnerRaster,
)
from picf_next.data.calvin_physical_visual_acceptance import (
    load_calvin_physical_visual_acceptance,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.calvin import build_native_calvin_training_stream_plan
from picf_next.lingbot_native.predictive_cache import (
    LINGBOT_PREDICTIVE_TARGET_SPACE,
    PredictiveCacheContract,
    PredictiveObjectCacheRecord,
    native_predictive_query_schema_digest,
    pool_dino_object_summaries,
    predictive_effective_fps,
    write_predictive_target_cache,
)
from picf_next.lingbot_native.predictive_plan import (
    NativePredictiveCoveragePlan,
    build_native_predictive_coverage_plan,
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
    from tools.lingbot_vla2_runtime_helpers import load_lingbot_training_config
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
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        load_lingbot_training_config,
    )


class PredictivePatchExtractor(Protocol):
    """The only teacher surface consumed by the cache builder."""

    def __call__(
        self,
        current_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
        *,
        effective_fps: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class _VerifiedStaticFrame:
    global_index: int
    rgb: np.ndarray
    rgb_sha256: str
    physical: CalvinPhysicalSupervisionFrame
    camera: CalvinVisibleOwnerRaster


class _VerifiedFrameCache:
    """Small offline LRU; contents never cross the loss-side boundary."""

    def __init__(
        self,
        index: CalvinDatasetIndex,
        sidecar: CalvinPhysicalSupervisionSidecar,
        *,
        capacity: int,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("predictive frame-cache capacity must be positive")
        self.index = index
        self.sidecar = sidecar
        self.capacity = capacity
        self._frames: OrderedDict[int, _VerifiedStaticFrame] = OrderedDict()

    def get(self, global_index: int) -> _VerifiedStaticFrame:
        cached = self._frames.get(global_index)
        if cached is not None:
            self._frames.move_to_end(global_index)
            return cached
        arrays = self.index.validated_source_frame_arrays(
            global_index,
            fields=("rgb_static",),
        )
        rgb = np.asarray(arrays["rgb_static"])
        physical = self.sidecar.source_frame(global_index)
        cameras = tuple(camera for camera in physical.cameras if camera.camera_name == "static")
        if len(cameras) != 1:
            raise ContractError("CALVIN predictive targets require exactly one static camera")
        camera = cameras[0]
        digest = source_array_sha256("rgb_static", rgb)
        if digest != camera.source_rgb_sha256:
            raise ContractError("CALVIN predictive RGB differs from the physical sidecar")
        frozen_rgb = np.ascontiguousarray(rgb).copy()
        frozen_rgb.setflags(write=False)
        frame = _VerifiedStaticFrame(
            global_index=global_index,
            rgb=frozen_rgb,
            rgb_sha256=digest,
            physical=physical,
            camera=camera,
        )
        self._frames[global_index] = frame
        self._frames.move_to_end(global_index)
        while len(self._frames) > self.capacity:
            self._frames.popitem(last=False)
        return frame


def _rgb_batch(frames: Sequence[_VerifiedStaticFrame]) -> torch.Tensor:
    if not frames:
        raise ValueError("DINO extraction requires a non-empty frame batch")
    array = np.stack(tuple(frame.rgb for frame in frames))
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ContractError("CALVIN static RGB batch differs from uint8 BHWC")
    return torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(1).contiguous()


def _extract_predictive_batch(
    pairs: Sequence[tuple[_VerifiedStaticFrame, _VerifiedStaticFrame, int]],
    *,
    extractor: PredictivePatchExtractor,
    contract: PredictiveCacheContract,
) -> tuple[PredictiveObjectCacheRecord, ...]:
    if not pairs:
        raise ValueError("predictive extraction batch cannot be empty")
    current = _rgb_batch(tuple(value[0] for value in pairs))
    future = _rgb_batch(tuple(value[1] for value in pairs))
    effective_fps = predictive_effective_fps(
        torch.tensor(tuple(value[2] for value in pairs), dtype=torch.long),
        source_fps=contract.source_fps,
    )
    patch_features = extractor(current, future, effective_fps=effective_fps)
    expected_shape = (len(pairs), contract.patch_tokens, contract.hidden_size)
    if (
        not isinstance(patch_features, torch.Tensor)
        or patch_features.shape != expected_shape
        or not patch_features.is_floating_point()
        or patch_features.requires_grad
        or patch_features.grad_fn is not None
        or not torch.isfinite(patch_features).all()
    ):
        raise ContractError(
            "official DINO-video teacher output differs from "
            f"{expected_shape} detached finite tokens"
        )
    records = []
    for row, (source, target, horizon) in enumerate(pairs):
        summaries, importance = pool_dino_object_summaries(
            patch_features[row],
            owner_index=target.camera.owner_index,
            owner_supervised=target.camera.owner_supervised,
            identity_keys=target.physical.identity_keys,
            minimum_visible_fraction=contract.minimum_visible_fraction,
            input_size=contract.input_size,
        )
        records.append(
            PredictiveObjectCacheRecord(
                source_global_index=source.global_index,
                target_global_index=target.global_index,
                horizon=horizon,
                source_rgb_sha256=source.rgb_sha256,
                target_rgb_sha256=target.rgb_sha256,
                identity_keys=target.physical.identity_keys,
                features=summaries,
                importance=importance,
            )
        )
    return tuple(records)


def iter_calvin_predictive_records(
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    *,
    extractor: PredictivePatchExtractor,
    contract: PredictiveCacheContract,
    coverage: NativePredictiveCoveragePlan,
    batch_size: int,
    frame_cache_capacity: int,
    progress: Callable[[int, int, float], None] | None = None,
) -> Iterator[PredictiveObjectCacheRecord]:
    """Extract the canonical complete pair stream in bounded host memory."""

    if not isinstance(index, CalvinDatasetIndex) or not isinstance(
        sidecar, CalvinPhysicalSupervisionSidecar
    ):
        raise TypeError("predictive extraction requires verified CALVIN index and sidecar")
    if not isinstance(contract, PredictiveCacheContract):
        raise TypeError("predictive extraction requires a typed cache contract")
    if not isinstance(coverage, NativePredictiveCoveragePlan):
        raise TypeError("predictive extraction requires a typed coverage plan")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("predictive extraction batch size must be positive")
    expected_identity = (
        index.dataset_id,
        index.dataset_revision,
        index.split_root.name,
        index.dataset_manifest.tree_sha256 if index.dataset_manifest is not None else None,
    )
    if expected_identity != (
        contract.dataset_id,
        contract.dataset_revision,
        contract.split_name,
        contract.dataset_tree_sha256,
    ):
        raise ContractError("predictive cache contract differs from the CALVIN index")
    if sidecar.manifest_sha256 != contract.physical_sidecar_manifest_sha256:
        raise ContractError("predictive cache contract differs from the physical sidecar")
    if sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise ContractError("predictive extraction requires all-source physical supervision")
    if (
        coverage.dataset_tree_sha256 != contract.dataset_tree_sha256
        or coverage.stream_plan_sha256 != contract.stream_plan_sha256
        or coverage.temporal_estimator_sha256 != contract.temporal_estimator_sha256
        or coverage.pair_keys_sha256 != contract.pair_keys_sha256
        or coverage.coverage_sha256 != contract.coverage_sha256
        or len(coverage.pairs) != contract.expected_record_count
    ):
        raise ContractError("predictive coverage plan differs from the cache contract")

    cache = _VerifiedFrameCache(index, sidecar, capacity=frame_cache_capacity)
    pending: list[tuple[_VerifiedStaticFrame, _VerifiedStaticFrame, int]] = []
    completed = 0
    started = time.perf_counter()
    for source_index, horizon in coverage.pairs:
        pending.append(
            (
                cache.get(source_index),
                cache.get(source_index + horizon),
                horizon,
            )
        )
        if len(pending) < batch_size:
            continue
        for record in _extract_predictive_batch(
            pending,
            extractor=extractor,
            contract=contract,
        ):
            completed += 1
            yield record
        pending.clear()
        if progress is not None:
            progress(completed, contract.expected_record_count, time.perf_counter() - started)
    if pending:
        for record in _extract_predictive_batch(
            pending,
            extractor=extractor,
            contract=contract,
        ):
            completed += 1
            yield record
    if completed != contract.expected_record_count:
        raise RuntimeError("predictive extraction did not traverse complete coverage")
    if progress is not None:
        progress(completed, contract.expected_record_count, time.perf_counter() - started)


def _video_config(training_config: Path) -> dict[str, object]:
    raw = load_lingbot_training_config(training_config)
    try:
        value = raw["train"]["align_params"]["video"]
    except (KeyError, TypeError) as error:
        raise ContractError("LingBot training config omits train.align_params.video") from error
    if not isinstance(value, Mapping):
        raise ContractError("LingBot video alignment config must be a mapping")
    config = dict(value)
    required = {
        "attention_mode": "flex_block_causal",
        "input_size": 256,
        "num_future_frames": 1,
        "use_warmup_frame": True,
        "effective_fps": 1.0,
        "n_blocks": 1,
        "cls_pool": "last",
        "num_backbone_tokens": 256,
        "dim_out": 1024,
        "use_patch_loss": True,
        "use_current_patch_loss": True,
        "use_cls_loss": False,
    }
    drift = {
        name: (config.get(name), expected)
        for name, expected in required.items()
        if config.get(name) != expected
    }
    if drift:
        raise ContractError(f"released LingBot DINO-video contract drifted: {drift}")
    return config


def _resolve_training_config(
    source_checkout: Path,
    training_config: Path | None,
) -> Path:
    """Resolve the released config from the selected source checkout by default."""

    if training_config is not None:
        return training_config
    return source_checkout / "configs/vla/robotwin/robotwin.yaml"


class OfficialLingBotDinoVideoExtractor:
    """Thin adapter over the released teacher and released preprocessing helper."""

    def __init__(
        self,
        *,
        source_checkout: Path,
        checkpoint_dir: Path,
        training_config: Path,
        device: torch.device,
    ) -> None:
        video = _video_config(training_config)
        video["ckpt_path"] = str(checkpoint_dir / "dino_video/teacher_step_10000.pth")
        video["config_path"] = str(checkpoint_dir / "dino_video/config.yaml")
        video["device"] = str(device)
        checkout_text = str(source_checkout.resolve())
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
        module = importlib.import_module("lingbotvla.models.vla.vision_models.module_utils")
        build_video_model = getattr(module, "build_video_model", None)
        get_video_target = getattr(module, "get_video_target", None)
        if not callable(build_video_model) or not callable(get_video_target):
            raise ContractError("released LingBot DINO-video helpers are unavailable")

        self.device = device
        self.config = {"video": video}
        default_effective_fps = video["effective_fps"]
        if isinstance(default_effective_fps, bool) or not isinstance(
            default_effective_fps, (int, float)
        ):
            raise ContractError("released LingBot effective FPS is not a real scalar")
        self.default_effective_fps = float(default_effective_fps)
        self.teacher = build_video_model(self.config)
        self._get_video_target = get_video_target

    def _effective_fps(
        self,
        value: torch.Tensor | float | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | float:
        fps = self.default_effective_fps if value is None else value
        if isinstance(fps, torch.Tensor):
            if fps.shape != (batch_size,) or not fps.is_floating_point():
                raise ValueError("teacher effective FPS must be floating [batch]")
            if not torch.isfinite(fps).all() or (fps <= 0).any():
                raise ValueError("teacher effective FPS must be finite and positive")
            return fps.to(device=self.device, dtype=torch.float32, non_blocking=True)
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise TypeError("teacher effective FPS must be a real scalar or tensor")
        measured = float(fps)
        if not np.isfinite(measured) or measured <= 0:
            raise ValueError("teacher effective FPS must be finite and positive")
        return measured

    @torch.no_grad()
    def paired(
        self,
        current_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
        *,
        effective_fps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return future and causal-current patches from one teacher invocation."""

        current = current_rgb.to(device=self.device, non_blocking=True)
        future = future_rgb.to(device=self.device, non_blocking=True)
        fps = self._effective_fps(effective_fps, batch_size=current.shape[0])
        output = self._get_video_target(
            self.teacher,
            current,
            future,
            self.config,
            effective_fps=fps,
        )
        if not isinstance(output, dict):
            raise ContractError("official DINO-video teacher returned no paired patch mapping")
        future_patches = output.get("patch")
        current_patches = output.get("current_patch")
        if not isinstance(future_patches, torch.Tensor) or not isinstance(
            current_patches, torch.Tensor
        ):
            raise ContractError("official DINO-video teacher returned incomplete paired patches")
        return future_patches.detach(), current_patches.detach()

    @torch.no_grad()
    def __call__(
        self,
        current_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
        *,
        effective_fps: torch.Tensor,
    ) -> torch.Tensor:
        future_patches, _current_patches = self.paired(
            current_rgb,
            future_rgb,
            effective_fps=effective_fps,
        )
        return future_patches

    @torch.no_grad()
    def current(
        self,
        current_rgb: torch.Tensor,
        *,
        effective_fps: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return the released teacher's causal current-frame patch tokens."""

        current = current_rgb.to(device=self.device, non_blocking=True)
        fps = self._effective_fps(effective_fps, batch_size=current.shape[0])
        output = self._get_video_target(
            self.teacher,
            current,
            current,
            self.config,
            effective_fps=fps,
        )
        if not isinstance(output, dict):
            raise ContractError("official DINO-video teacher returned no current-patch mapping")
        patches = output.get("current_patch")
        if not isinstance(patches, torch.Tensor):
            raise ContractError("official DINO-video teacher returned no current patch tensor")
        return patches.detach()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_build_report(path: Path, report: Mapping[str, object]) -> None:
    """Atomically publish and durably bind a cache report to its parent directory."""

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    write_text_durable_exclusive(path, payload, encoding="ascii")


def _build_contract(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    coverage: NativePredictiveCoveragePlan,
    checkpoint_dir: Path,
    minimum_visible_fraction: float,
) -> PredictiveCacheContract:
    if index.dataset_manifest is None:
        raise ContractError("predictive cache requires a content-addressed dataset")
    if coverage.dataset_tree_sha256 != index.dataset_manifest.tree_sha256:
        raise ContractError("predictive coverage belongs to another dataset")
    horizons = coverage.horizons
    query = native_predictive_query_schema_digest(
        target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
        route_id=0,
        horizons=horizons,
    )
    return PredictiveCacheContract(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        split_name=index.split_root.name,
        dataset_tree_sha256=index.dataset_manifest.tree_sha256,
        physical_sidecar_manifest_sha256=sidecar.manifest_sha256,
        lingbot_source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
        lingbot_checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
        teacher_config_sha256=_sha256(checkpoint_dir / "dino_video/config.yaml"),
        teacher_checkpoint_sha256=_sha256(checkpoint_dir / "dino_video/teacher_step_10000.pth"),
        query_schema_sha256=query,
        horizons=horizons,
        stream_plan_sha256=coverage.stream_plan_sha256,
        temporal_estimator_sha256=coverage.temporal_estimator_sha256,
        pair_keys_sha256=coverage.pair_keys_sha256,
        coverage_sha256=coverage.coverage_sha256,
        expected_record_count=len(coverage.pairs),
        source_fps=float(index.control_hz),
        minimum_visible_fraction=minimum_visible_fraction,
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
    add_reset_mixture_arguments(parser)
    parser.add_argument("--representation-split", type=Path)
    parser.add_argument("--representation-split-sha256")
    parser.add_argument(
        "--required-future-horizons",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Sorted future horizons required for every planned source with a "
            "same-segment target; bounded terminal sources are omitted."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-rows", type=int, default=2048)
    parser.add_argument("--frame-cache-capacity", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=1024)
    parser.add_argument("--minimum-visible-fraction", type=float, default=0.0)
    return parser.parse_args()


def _temporal_config_from_args(args: argparse.Namespace) -> TemporalEstimatorConfig:
    return TemporalEstimatorConfig(
        local_bptt_probability=args.local_bptt_probability,
        overshoot_probability=args.overshoot_probability,
        source_mask_probability=args.source_mask_probability,
        maximum_optimizer_lag=args.maximum_optimizer_lag,
    )


def main() -> None:
    args = _parse_args()
    reset_mixture = reset_mixture_values(args)
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
    if (
        isinstance(args.plan_seed, bool)
        or not isinstance(args.plan_seed, int)
        or args.plan_seed < 0
    ):
        raise ValueError("--plan-seed must be a non-negative integer")
    if not isinstance(args.comparison_id, str) or not args.comparison_id.strip():
        raise ValueError("--comparison-id must be non-empty")
    temporal_config = _temporal_config_from_args(args)
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
        raise ContractError("predictive teacher source differs from the pinned LingBot commit")
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
    stateful_dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    representation_split: RepresentationTrialSplit | None = None
    if args.representation_split is not None:
        if _sha256(args.representation_split) != args.representation_split_sha256:
            raise ValueError("representation split file SHA-256 differs")
        representation_split = RepresentationTrialSplit.load(args.representation_split)
    stream_plan = build_native_calvin_training_stream_plan(
        stateful_dataset,
        comparison_id=args.comparison_id,
        seed=args.plan_seed,
        global_batch_size=args.global_batch_size,
        total_steps=args.total_steps,
        lane_interleave_factor=args.lane_interleave_factor,
        excluded_source_episode_indices=(
            representation_split.evaluation_source_episode_indices
            if representation_split is not None
            and representation_split.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
            else ()
        ),
        reset_numerator=(None if reset_mixture is None else reset_mixture[0]),
        reset_denominator=(None if reset_mixture is None else reset_mixture[1]),
    )
    if (
        representation_split is not None
        and representation_split.stream_plan_sha256 != stream_plan.plan_sha256
    ):
        raise ValueError("representation split differs from predictive cache stream")
    coverage = build_native_predictive_coverage_plan(
        stream_plan,
        temporal_config,
        source_global_index_for_sample=stateful_dataset.source_global_index_by_key,
        required_horizons=tuple(args.required_future_horizons),
    )
    contract = _build_contract(
        index=index,
        sidecar=sidecar,
        coverage=coverage,
        checkpoint_dir=args.checkpoint_dir,
        minimum_visible_fraction=args.minimum_visible_fraction,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("released DINO-video cache extraction requires one CUDA device")
    extractor = OfficialLingBotDinoVideoExtractor(
        source_checkout=args.source_checkout,
        checkpoint_dir=args.checkpoint_dir,
        training_config=training_config,
        device=device,
    )
    last_reported = -1

    def progress(completed: int, total: int, elapsed: float) -> None:
        nonlocal last_reported
        if completed != total and completed - last_reported < args.progress_every:
            return
        last_reported = completed
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (total - completed) / rate if rate > 0 else None
        print(
            json.dumps(
                {
                    "completed": completed,
                    "elapsed_s": elapsed,
                    "records_per_second": rate,
                    "remaining_s": remaining,
                    "total": total,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    records = iter_calvin_predictive_records(
        index,
        sidecar,
        extractor=extractor,
        contract=contract,
        coverage=coverage,
        batch_size=args.batch_size,
        frame_cache_capacity=args.frame_cache_capacity,
        progress=progress,
    )
    manifest_sha256 = write_predictive_target_cache(
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
        "pair_keys_sha256": contract.pair_keys_sha256,
        "patch_sha256": patch["patch_sha256"],
        "physical_visual_acceptance_sha256": args.physical_visual_acceptance_sha256,
        "stream_plan_sha256": contract.stream_plan_sha256,
        "teacher_encoder_digest": contract.encoder_digest,
        "temporal_estimator_sha256": contract.temporal_estimator_sha256,
    }
    resolved_output = args.output_root.resolve()
    report_path = resolved_output.parent / f"{resolved_output.name}.build_report.json"
    _write_build_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
