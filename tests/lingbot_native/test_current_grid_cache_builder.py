from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.lingbot_native.current_grid_cache import (
    CurrentGridCacheContract,
    current_grid_coverage_digest,
    current_grid_source_keys_digest,
)
from tools.build_lingbot_calvin_current_grid_cache import (
    _extract_current_batch,
    _temporal_config,
    _validate_builder_args,
    _validate_donor_cache_semantics,
)
from tools.build_lingbot_calvin_predictive_cache import _VerifiedStaticFrame


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _contract() -> CurrentGridCacheContract:
    dataset = _sha("dataset")
    stream = _sha("stream")
    temporal = _sha("temporal")
    sources = current_grid_source_keys_digest((10,))
    return CurrentGridCacheContract(
        dataset_id="calvin",
        dataset_revision="fixture",
        split_name="training",
        dataset_tree_sha256=dataset,
        physical_sidecar_manifest_sha256=_sha("sidecar"),
        lingbot_source_commit="2838c",
        lingbot_checkpoint_revision="released",
        teacher_config_sha256=_sha("config"),
        teacher_checkpoint_sha256=_sha("checkpoint"),
        stream_plan_sha256=stream,
        temporal_estimator_sha256=temporal,
        source_keys_sha256=sources,
        coverage_sha256=current_grid_coverage_digest(
            dataset_tree_sha256=dataset,
            stream_plan_sha256=stream,
            temporal_estimator_sha256=temporal,
            source_keys_sha256=sources,
            expected_record_count=1,
        ),
        expected_record_count=1,
    )


def test_current_grid_builder_uses_official_current_patch_surface_only() -> None:
    frame = _VerifiedStaticFrame(
        global_index=10,
        rgb=np.full((200, 200, 3), 17, dtype=np.uint8),
        rgb_sha256=_sha("rgb-10"),
        physical=SimpleNamespace(),
        camera=SimpleNamespace(),
    )

    class _Extractor:
        def __init__(self) -> None:
            self.observed: torch.Tensor | None = None

        def current(self, current_rgb: torch.Tensor) -> torch.Tensor:
            self.observed = current_rgb.clone()
            channels = torch.arange(1024, dtype=torch.float32)[None, :]
            patches = torch.arange(256, dtype=torch.float32)[:, None]
            return (channels + patches).unsqueeze(0)

    extractor = _Extractor()
    (record,) = _extract_current_batch((frame,), extractor=extractor, contract=_contract())

    assert extractor.observed is not None
    assert extractor.observed.shape == (1, 1, 3, 200, 200)
    assert extractor.observed.unique().item() == 17
    assert record.source_global_index == 10
    assert record.source_rgb_sha256 == _sha("rgb-10")
    assert record.features.shape == (256, 1024)
    assert record.features.dtype == np.float16


def test_current_patch_bank_builder_accepts_correction_only_temporal_config() -> None:
    args = SimpleNamespace(
        batch_size=1,
        shard_rows=1,
        frame_cache_capacity=1,
        progress_every=1,
        global_batch_size=2,
        total_steps=4,
        local_bptt_probability=0.5,
        overshoot_probability=0.5,
        source_mask_probability=0.0,
        maximum_optimizer_lag=8,
        lane_interleave_factor=1,
        physical_event_stream=False,
        representation_split=None,
        representation_split_sha256=None,
        donor_cache_root=None,
        donor_cache_manifest_sha256=None,
        rebind_exact_donor=False,
        donor_content_manifest=None,
        donor_official_source_receipt=None,
        donor_official_source_receipt_sha256=None,
    )

    _validate_builder_args(args)
    assert _temporal_config(args).source_mask_probability == 0.0

    args.rebind_exact_donor = True
    with pytest.raises(ValueError, match="exact donor rebind requires"):
        _validate_builder_args(args)

    args.rebind_exact_donor = False
    args.donor_content_manifest = "manifest.json"
    with pytest.raises(ValueError, match="must be provided together"):
        _validate_builder_args(args)

    args.donor_cache_root = "cache"
    args.donor_cache_manifest_sha256 = _sha("cache-manifest")
    args.donor_official_source_receipt = "receipt.json"
    args.donor_official_source_receipt_sha256 = _sha("receipt")
    _validate_builder_args(args)


def test_current_grid_cross_identity_donor_requires_explicit_verified_migration() -> None:
    target = _contract()
    donor_tree = _sha("donor-dataset")
    donor = replace(
        target,
        dataset_id="legacy-calvin",
        dataset_revision="legacy-revision",
        dataset_tree_sha256=donor_tree,
        physical_sidecar_manifest_sha256=_sha("donor-sidecar"),
        coverage_sha256=current_grid_coverage_digest(
            dataset_tree_sha256=donor_tree,
            stream_plan_sha256=target.stream_plan_sha256,
            temporal_estimator_sha256=target.temporal_estimator_sha256,
            source_keys_sha256=target.source_keys_sha256,
            expected_record_count=target.expected_record_count,
        ),
    )

    with pytest.raises(ContractError, match="source identity"):
        _validate_donor_cache_semantics(
            donor,
            target,
            content_identity_verified=False,
        )
    _validate_donor_cache_semantics(donor, target, content_identity_verified=True)

    changed_teacher = replace(
        donor,
        teacher_checkpoint_sha256=_sha("different-teacher"),
    )
    with pytest.raises(ContractError, match="frozen-teacher"):
        _validate_donor_cache_semantics(
            changed_teacher,
            target,
            content_identity_verified=True,
        )
