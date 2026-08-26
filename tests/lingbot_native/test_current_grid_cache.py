from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import picf_next.lingbot_native.current_grid_cache as current_grid_cache_module
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
    current_correction_summary_query_schema_digest,
    current_grid_coverage_digest,
    current_grid_query_schema_digest,
    current_grid_source_keys_digest,
    omitted_static_summary_query_schema_digest,
    rebind_current_grid_target_cache,
    write_current_grid_target_cache,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.source_mask import QwenWholeViewOmission


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _contract(source_indices: tuple[int, ...] = (10,)) -> CurrentGridCacheContract:
    dataset = _sha("dataset")
    stream = _sha("stream")
    temporal = _sha("temporal")
    sources = current_grid_source_keys_digest(source_indices)
    return CurrentGridCacheContract(
        dataset_id="calvin",
        dataset_revision="fixture",
        split_name="training",
        dataset_tree_sha256=dataset,
        physical_sidecar_manifest_sha256=_sha("sidecar"),
        lingbot_source_commit="2838c",
        lingbot_checkpoint_revision="released",
        teacher_config_sha256=_sha("teacher-config"),
        teacher_checkpoint_sha256=_sha("teacher-checkpoint"),
        stream_plan_sha256=stream,
        temporal_estimator_sha256=temporal,
        source_keys_sha256=sources,
        coverage_sha256=current_grid_coverage_digest(
            dataset_tree_sha256=dataset,
            stream_plan_sha256=stream,
            temporal_estimator_sha256=temporal,
            source_keys_sha256=sources,
            expected_record_count=len(source_indices),
        ),
        expected_record_count=len(source_indices),
    )


def _features(offset: int = 0) -> np.ndarray:
    channels = np.arange(1024, dtype=np.float32)[None, :]
    patches = np.arange(256, dtype=np.float32)[:, None]
    return (channels + patches + offset).astype(np.float16)


def test_current_grid_contract_parser_rejects_wrong_dynamic_field_types() -> None:
    payload = _contract().to_dict()
    payload["expected_record_count"] = "1"
    with pytest.raises(ContractError, match="positive integer"):
        CurrentGridCacheContract.from_mapping(payload)


class _Sidecar(CalvinPhysicalSupervisionSidecar):
    def __init__(self, *, manifest_sha256: str, source_hash: str) -> None:
        self.manifest_sha256 = manifest_sha256
        self.source_hash = source_hash

    def source_frame(self, global_index: int) -> CalvinPhysicalSupervisionFrame:
        assert global_index == 10
        owner = np.ones((200, 200), dtype=np.uint8)
        owner[:, 100:] = 2
        supervised = np.ones((200, 200), dtype=np.bool_)
        static = CalvinVisibleOwnerRaster(
            camera_name="static",
            host_image_key="observation.images.image",
            owner_index=owner,
            owner_supervised=supervised,
            source_rgb_sha256=self.source_hash,
            source_depth_sha256=_sha("depth-static"),
            rgb_mae=0.0,
            depth_mae_m=0.0,
            depth_p95_m=0.0,
            depth_consistent_fraction=1.0,
        )
        gripper = CalvinVisibleOwnerRaster(
            camera_name="gripper",
            host_image_key="observation.images.wrist_image",
            owner_index=np.pad(
                np.ones((84, 42), dtype=np.uint8),
                ((0, 0), (0, 42)),
            ),
            owner_supervised=np.ones((84, 84), dtype=np.bool_),
            source_rgb_sha256=_sha("rgb-gripper"),
            source_depth_sha256=_sha("depth-gripper"),
            rgb_mae=0.0,
            depth_mae_m=0.0,
            depth_p95_m=0.0,
            depth_consistent_fraction=1.0,
        )
        return CalvinPhysicalSupervisionFrame(
            identity_keys=("object/a", "object/b"),
            geometry=torch.zeros(2, 3),
            geometry_variance=torch.zeros(2, 3),
            geometry_supervised=torch.ones(2, 3, dtype=torch.bool),
            geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
            cameras=(static, gripper),
        )


def _load(
    root: Path,
    manifest_sha256: str,
    contract: CurrentGridCacheContract,
    *,
    shard_root: Path | None = None,
):
    return LingBotCurrentGridTargetCache.load(
        root,
        shard_root=shard_root,
        manifest_sha256=manifest_sha256,
        dataset_tree_sha256=contract.dataset_tree_sha256,
        physical_sidecar_manifest_sha256=contract.physical_sidecar_manifest_sha256,
        encoder_digest=contract.encoder_digest,
        coverage_sha256=contract.coverage_sha256,
    )


def test_current_grid_cache_roundtrip_and_exact_loss_side_owner_selection(
    tmp_path: Path,
) -> None:
    contract = _contract()
    source_hash = _sha("rgb-static")
    root = tmp_path / "current"
    manifest = write_current_grid_target_cache(
        root,
        contract=contract,
        records=(CurrentGridCacheRecord(10, source_hash, _features()),),
        shard_rows=1,
    )
    cache = _load(root, manifest, contract)
    sidecar = _Sidecar(
        manifest_sha256=contract.physical_sidecar_manifest_sha256,
        source_hash=source_hash,
    )
    assert cache.has_supported_current_summary(
        source_global_index=10,
        physical_sidecar=sidecar,
        minimum_visible_fraction=0.49,
    )
    assert (
        cache.supported_current_summary_count(
            source_global_index=10,
            physical_sidecar=sidecar,
            minimum_visible_fraction=0.49,
        )
        == 2
    )
    with pytest.raises(ContractError, match="non-negative integer"):
        cache.supported_current_summary_count(
            source_global_index=True,
            physical_sidecar=sidecar,
            minimum_visible_fraction=0.49,
        )
    assert not cache.has_supported_current_summary(
        source_global_index=10,
        physical_sidecar=sidecar,
        minimum_visible_fraction=0.5,
    )
    (record,) = tuple(cache.iter_records())
    assert record.source_global_index == 10
    original = record.features.copy()
    record.features[:] = 99
    np.testing.assert_array_equal(next(cache.iter_records()).features, original)
    direct = cache.record_for(source_global_index=10)
    assert direct is not None
    direct.features[:] = 77
    reread = cache.record_for(source_global_index=10)
    assert reread is not None
    np.testing.assert_array_equal(reread.features, original)
    assert cache.record_for(source_global_index=11) is None
    selected = torch.tensor([[0, 255]], dtype=torch.long)
    grid = torch.tensor([[16, 16]], dtype=torch.long)
    addresses = torch.tensor(
        [[[-0.9375, -0.9375], [0.9375, 0.9375]]],
        dtype=torch.float32,
    )
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.CURRENT_RANDOM_GRID,
        route_ids=torch.zeros(1, 2, dtype=torch.long),
        horizons=torch.zeros(1, 2, dtype=torch.long),
        addresses=addresses,
        valid=torch.ones(1, 2, dtype=torch.bool),
    )
    target = cache.target_for(
        source_global_indices=(10,),
        source_rgb_sha256=(source_hash,),
        track_identity_keys=(("object/a", "object/b"),),
        selected_token_indices=selected,
        merged_grid_hw=grid,
        request=request,
        physical_sidecar=sidecar,
        device="cpu",
    )

    assert target.features.shape == (1, 2, 2, 1024)
    assert target.valid[0, 0, 0]
    assert not target.valid[0, 0, 1]
    assert not target.valid[0, 1, 0]
    assert target.valid[0, 1, 1]
    assert target.supports_object_binding_claim is False
    assert target.query_schema_digest == current_grid_query_schema_digest(route_id=0)
    assert not target.features.requires_grad
    assert target.features[0, 0, 0].std() > 0


def test_current_grid_cache_source_bank_reuse_allows_only_plan_drift(tmp_path: Path) -> None:
    contract = _contract()
    root = tmp_path / "donor"
    manifest = write_current_grid_target_cache(
        root,
        contract=contract,
        records=(CurrentGridCacheRecord(10, _sha("rgb-static"), _features()),),
        shard_rows=1,
    )

    donor = LingBotCurrentGridTargetCache.load_reusable_source_bank(
        root,
        manifest_sha256=manifest,
        dataset_tree_sha256=contract.dataset_tree_sha256,
        physical_sidecar_manifest_sha256=contract.physical_sidecar_manifest_sha256,
        encoder_digest=contract.encoder_digest,
    )
    reused = donor.record_for(source_global_index=10)
    assert reused is not None
    assert reused.source_rgb_sha256 == _sha("rgb-static")

    with pytest.raises(ContractError, match="source bank provenance differs"):
        LingBotCurrentGridTargetCache.load_reusable_source_bank(
            root,
            manifest_sha256=manifest,
            dataset_tree_sha256=contract.dataset_tree_sha256,
            physical_sidecar_manifest_sha256=contract.physical_sidecar_manifest_sha256,
            encoder_digest=_sha("another-encoder"),
        )


def test_current_grid_cache_rebinds_only_exact_rgb_and_encoder_content(
    tmp_path: Path,
) -> None:
    source_contract = _contract((10, 20))
    source_root = tmp_path / "source"
    source_manifest = write_current_grid_target_cache(
        source_root,
        contract=source_contract,
        records=(
            CurrentGridCacheRecord(10, _sha("rgb-10"), _features()),
            CurrentGridCacheRecord(20, _sha("rgb-20"), _features(1)),
        ),
        shard_rows=1,
    )
    source = _load(source_root, source_manifest, source_contract)
    target_dataset = _sha("official-dataset")
    target_stream = _sha("official-stream")
    target_sources = current_grid_source_keys_digest((10, 20))
    target_contract = CurrentGridCacheContract(
        **{
            **source_contract.to_dict(),
            "dataset_id": "official-calvin",
            "dataset_revision": "content-revision",
            "dataset_tree_sha256": target_dataset,
            "physical_sidecar_manifest_sha256": _sha("official-sidecar"),
            "stream_plan_sha256": target_stream,
            "source_keys_sha256": target_sources,
            "coverage_sha256": current_grid_coverage_digest(
                dataset_tree_sha256=target_dataset,
                stream_plan_sha256=target_stream,
                temporal_estimator_sha256=source_contract.temporal_estimator_sha256,
                source_keys_sha256=target_sources,
                expected_record_count=2,
            ),
        }
    )
    target_root = tmp_path / "target"
    target_manifest = rebind_current_grid_target_cache(
        target_root,
        source_cache=source,
        contract=target_contract,
        source_rgb_sha256_for=lambda index: _sha(f"rgb-{index}"),
    )
    target = _load(
        target_root,
        target_manifest,
        target_contract,
        shard_root=source_root,
    )
    assert target.contract.dataset_tree_sha256 == target_dataset
    assert tuple(record.source_rgb_sha256 for record in target.iter_records()) == (
        _sha("rgb-10"),
        _sha("rgb-20"),
    )
    assert target.manifest_root == target_root
    assert target.root == source_root
    assert tuple(path.name for path in target_root.iterdir()) == ("manifest.json",)

    with pytest.raises(ContractError, match="RGB identity differs"):
        rebind_current_grid_target_cache(
            tmp_path / "wrong-rgb",
            source_cache=source,
            contract=target_contract,
            source_rgb_sha256_for=lambda _index: _sha("wrong"),
        )
    assert not (tmp_path / "wrong-rgb").exists()

    changed_encoder = CurrentGridCacheContract(
        **{
            **target_contract.to_dict(),
            "teacher_checkpoint_sha256": _sha("another-teacher"),
        }
    )
    with pytest.raises(ContractError, match="encoder semantics"):
        rebind_current_grid_target_cache(
            tmp_path / "wrong-encoder",
            source_cache=source,
            contract=changed_encoder,
            source_rgb_sha256_for=lambda index: _sha(f"rgb-{index}"),
        )


def test_current_grid_cache_rejects_mutation_and_incomplete_coverage(tmp_path: Path) -> None:
    contract = _contract((10, 20))
    with pytest.raises(Exception, match="record count"):
        write_current_grid_target_cache(
            tmp_path / "incomplete",
            contract=contract,
            records=(CurrentGridCacheRecord(10, _sha("rgb-10"), _features()),),
        )

    complete = tmp_path / "complete"
    manifest = write_current_grid_target_cache(
        complete,
        contract=contract,
        records=(
            CurrentGridCacheRecord(10, _sha("rgb-10"), _features()),
            CurrentGridCacheRecord(20, _sha("rgb-20"), _features(1)),
        ),
        shard_rows=1,
    )
    (complete / "shard-000000.npz").write_bytes(b"mutated")
    cache = _load(complete, manifest, contract)
    with pytest.raises(Exception, match="content hash mismatch"):
        cache._record(10)


def test_current_grid_cache_binds_every_lazy_shard_index_to_manifest(tmp_path: Path) -> None:
    contract = _contract((10, 20, 30))
    root = tmp_path / "current"
    write_current_grid_target_cache(
        root,
        contract=contract,
        records=tuple(
            CurrentGridCacheRecord(index, _sha(f"rgb-{index}"), _features(index))
            for index in (10, 20, 30)
        ),
        shard_rows=3,
    )
    shard_path = root / "shard-000000.npz"
    with np.load(shard_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["source_global_indices"][1] = 21
    np.savez(shard_path, **arrays)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("ascii") + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    cache = _load(root, hashlib.sha256(manifest_bytes).hexdigest(), contract)
    with pytest.raises(ContractError, match="indices differ from manifest coverage"):
        tuple(cache.iter_records())


def test_current_grid_cache_refuses_existing_symlink_destination(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "cache"
    root.symlink_to(external, target_is_directory=True)

    with pytest.raises(FileExistsError):
        write_current_grid_target_cache(
            root,
            contract=_contract(),
            records=(CurrentGridCacheRecord(10, _sha("rgb-static"), _features()),),
        )
    assert not tuple(external.iterdir())


def test_current_grid_cache_rejects_symlink_and_oversized_shard_reads(tmp_path: Path) -> None:
    contract = _contract()
    source_hash = _sha("rgb-static")

    symlink_root = tmp_path / "symlink-cache"
    manifest = write_current_grid_target_cache(
        symlink_root,
        contract=contract,
        records=(CurrentGridCacheRecord(10, source_hash, _features()),),
        shard_rows=1,
    )
    cache = _load(symlink_root, manifest, contract)
    shard = symlink_root / "shard-000000.npz"
    external = tmp_path / "external-shard.npz"
    external.write_bytes(shard.read_bytes())
    shard.unlink()
    shard.symlink_to(external)
    with pytest.raises(Exception, match="must not use symlinks"):
        cache._record(10)

    oversized_root = tmp_path / "oversized-cache"
    manifest = write_current_grid_target_cache(
        oversized_root,
        contract=contract,
        records=(CurrentGridCacheRecord(10, source_hash, _features()),),
        shard_rows=1,
    )
    cache = _load(oversized_root, manifest, contract)
    oversized_shard = oversized_root / "shard-000000.npz"
    oversized_shard.write_bytes(b"")
    with oversized_shard.open("r+b") as stream:
        stream.truncate(32 * 1024 * 1024)
    with pytest.raises(Exception, match="verified-read byte limit"):
        cache._record(10)


@pytest.mark.parametrize(
    ("image_valid", "error"),
    (
        (torch.tensor([[True, True, False]]), None),
        (torch.tensor([[True, True]]), "official three-slot camera ABI"),
        (torch.tensor([[True, True, True]]), "image availability"),
    ),
)
def test_current_grid_cache_validates_official_omitted_static_camera_slots(
    tmp_path: Path,
    image_valid: torch.Tensor,
    error: str | None,
) -> None:
    contract = _contract()
    static_hash = _sha("rgb-static")
    gripper_hash = _sha("rgb-gripper")
    root = tmp_path / "omitted-static"
    manifest = write_current_grid_target_cache(
        root,
        contract=contract,
        records=(CurrentGridCacheRecord(10, static_hash, _features()),),
        shard_rows=1,
    )
    cache = _load(root, manifest, contract)
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=torch.tensor([[[1, 32, 32] for _ in range(image_valid.shape[1])]]),
        image_valid=image_valid,
        seed=3,
    )
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.OMITTED_MODALITY,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    def target_for_omission():
        return cache.omitted_static_summary_target_for(
            source_global_indices=(10,),
            source_static_rgb_sha256=(static_hash,),
            source_gripper_rgb_sha256=(gripper_hash,),
            track_identity_keys=(("object/a", "object/b"),),
            request=request,
            omission=omission,
            physical_sidecar=_Sidecar(
                manifest_sha256=contract.physical_sidecar_manifest_sha256,
                source_hash=static_hash,
            ),
            device="cpu",
        )

    if error is not None:
        with pytest.raises(ValueError, match=error):
            target_for_omission()
        return
    target = target_for_omission()

    assert target.features.shape == (1, 2, 1, 1024)
    assert target.valid[0, 0, 0]
    assert not target.valid[0, 1, 0]
    # Static support is one half and the available wrist-view source support is
    # one half, so the absolute cross-view supervision mass is one quarter.
    torch.testing.assert_close(target.importance[0, 0, 0], torch.tensor(0.25))
    assert target.supports_object_binding_claim
    assert target.query_schema_digest == omitted_static_summary_query_schema_digest(route_id=0)
    assert not target.features.requires_grad
    assert target.features[0, 0, 0].std() > 0


def test_current_grid_cache_unpublishes_after_post_rename_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    source_hash = _sha("rgb-static")
    replaced = False
    original_replace = current_grid_cache_module.os.replace
    original_fsync = current_grid_cache_module.os.fsync

    def track_replace(source: Path, destination: Path) -> None:
        nonlocal replaced
        original_replace(source, destination)
        replaced = True

    def fail_parent_fsync(descriptor: int) -> None:
        if replaced:
            raise OSError("injected post-rename fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(current_grid_cache_module.os, "replace", track_replace)
    monkeypatch.setattr(current_grid_cache_module.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="post-rename"):
        write_current_grid_target_cache(
            root,
            contract=_contract(),
            records=(CurrentGridCacheRecord(10, source_hash, _features()),),
            shard_rows=1,
        )

    assert not root.exists()
    assert not tuple(tmp_path.glob(".cache.*.incomplete"))


def test_current_grid_cache_detects_rename_completion_before_publication_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    source_hash = _sha("rgb-static")
    original_replace = current_grid_cache_module.os.replace

    def replace_then_raise(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        raise OSError("injected exception after rename syscall")

    monkeypatch.setattr(current_grid_cache_module.os, "replace", replace_then_raise)
    with pytest.raises(OSError, match="after rename syscall"):
        write_current_grid_target_cache(
            root,
            contract=_contract(),
            records=(CurrentGridCacheRecord(10, source_hash, _features()),),
            shard_rows=1,
        )

    assert not root.exists()
    assert not tuple(tmp_path.glob(".cache.*.incomplete"))


def test_current_grid_cache_builds_prior_to_current_object_summary_with_absolute_support(
    tmp_path: Path,
) -> None:
    contract = _contract()
    static_hash = _sha("rgb-static")
    root = tmp_path / "current-correction"
    manifest = write_current_grid_target_cache(
        root,
        contract=contract,
        records=(CurrentGridCacheRecord(10, static_hash, _features()),),
        shard_rows=1,
    )
    cache = _load(root, manifest, contract)
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.zeros(1, 1, 2),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    target = cache.current_correction_summary_target_for(
        source_global_indices=(10,),
        source_static_rgb_sha256=(static_hash,),
        track_identity_keys=(("object/a", "object/b"),),
        request=request,
        physical_sidecar=_Sidecar(
            manifest_sha256=contract.physical_sidecar_manifest_sha256,
            source_hash=static_hash,
        ),
        minimum_visible_fraction=0.0,
        device="cpu",
    )

    assert target.features.shape == (1, 2, 1, 1024)
    assert target.valid.all()
    torch.testing.assert_close(target.importance, torch.full_like(target.importance, 0.5))
    assert target.supports_object_binding_claim
    assert target.query_schema_digest == current_correction_summary_query_schema_digest(
        route_id=0,
        address_width=2,
    )
    assert not target.features.requires_grad

    rejected = cache.current_correction_summary_target_for(
        source_global_indices=(10,),
        source_static_rgb_sha256=(static_hash,),
        track_identity_keys=(("object/a", "object/b"),),
        request=request,
        physical_sidecar=_Sidecar(
            manifest_sha256=contract.physical_sidecar_manifest_sha256,
            source_hash=static_hash,
        ),
        minimum_visible_fraction=0.5,
        device="cpu",
    )
    assert not rejected.valid.any()
    assert not rejected.importance.any()
    assert rejected.target_data_digest != target.target_data_digest


@pytest.mark.parametrize(
    ("source", "evidence"),
    (
        (PredictionSource.PRIOR, PredictionEvidence.CURRENT_PRIOR),
        (PredictionSource.POSTERIOR, PredictionEvidence.CURRENT_POSTERIOR),
    ),
)
def test_current_summary_cache_preserves_explicit_filter_phase(
    tmp_path: Path,
    source: PredictionSource,
    evidence: PredictionEvidence,
) -> None:
    contract = _contract()
    static_hash = _sha("rgb-static")
    root = tmp_path / evidence.value
    manifest = write_current_grid_target_cache(
        root,
        contract=contract,
        records=(CurrentGridCacheRecord(10, static_hash, _features()),),
        shard_rows=1,
    )
    cache = _load(root, manifest, contract)
    request = NativePredictionRequest(
        source=source,
        evidence=evidence,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.zeros(1, 1, 2),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    target = cache.current_correction_summary_target_for(
        source_global_indices=(10,),
        source_static_rgb_sha256=(static_hash,),
        track_identity_keys=(("object/a", "object/b"),),
        request=request,
        physical_sidecar=_Sidecar(
            manifest_sha256=contract.physical_sidecar_manifest_sha256,
            source_hash=static_hash,
        ),
        minimum_visible_fraction=0.0,
        device="cpu",
    )

    assert target.source is source
    assert target.evidence is evidence
    assert target.query_schema_digest == current_correction_summary_query_schema_digest(
        route_id=0,
        address_width=2,
        source=source,
        evidence=evidence,
    )
