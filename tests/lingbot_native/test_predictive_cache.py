from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

import picf_next.lingbot_native.predictive_cache as predictive_cache_module
from picf_next.contracts import ContractError
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
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
    predictive_effective_fps,
    write_predictive_target_cache,
)
from tools.audit_lingbot_predictive_targets import audit_predictive_target_cache


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _contract(*, horizons: tuple[int, ...] = (1, 2)) -> PredictiveCacheContract:
    if horizons in ((1, 2), (1, 2, 64)):
        pairs = (
            (10, 1),
            (10, 2),
            (11, 1),
            (20, 1),
            (20, 2),
            (21, 1),
        )
    else:
        pairs = ((10, 1),)
    dataset_tree_sha256 = _sha("dataset")
    stream_plan_sha256 = _sha("stream-plan")
    temporal_estimator_sha256 = _sha("temporal-estimator")
    pair_keys_sha256 = native_predictive_pair_keys_digest(pairs)
    return PredictiveCacheContract(
        dataset_id="calvin",
        dataset_revision="abc123",
        split_name="training",
        dataset_tree_sha256=dataset_tree_sha256,
        physical_sidecar_manifest_sha256=_sha("sidecar"),
        lingbot_source_commit="2838c",
        lingbot_checkpoint_revision="released",
        teacher_config_sha256=_sha("teacher-config"),
        teacher_checkpoint_sha256=_sha("teacher-checkpoint"),
        query_schema_sha256=native_predictive_query_schema_digest(
            target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
            route_id=0,
            horizons=horizons,
        ),
        horizons=horizons,
        stream_plan_sha256=stream_plan_sha256,
        temporal_estimator_sha256=temporal_estimator_sha256,
        pair_keys_sha256=pair_keys_sha256,
        coverage_sha256=native_predictive_coverage_digest(
            dataset_tree_sha256=dataset_tree_sha256,
            stream_plan_sha256=stream_plan_sha256,
            temporal_estimator_sha256=temporal_estimator_sha256,
            pair_keys_sha256=pair_keys_sha256,
            expected_record_count=len(pairs),
            horizons=horizons,
        ),
        expected_record_count=len(pairs),
    )


def _record(
    source: int,
    horizon: int,
    *,
    identities: tuple[str, ...] = ("object/b", "object/a"),
    supported: bool = True,
) -> PredictiveObjectCacheRecord:
    features = np.stack(
        tuple(np.full(1024, index + source, dtype=np.float16) for index in range(len(identities)))
    )
    importance = np.asarray([0.25, 0.0] if supported else [0.0, 0.0], dtype=np.float32)
    features[importance == 0] = 0
    return PredictiveObjectCacheRecord(
        source_global_index=source,
        target_global_index=source + horizon,
        horizon=horizon,
        source_rgb_sha256=_sha(f"source-{source}"),
        target_rgb_sha256=_sha(f"target-{source + horizon}"),
        identity_keys=identities,
        features=features,
        importance=importance,
    )


def test_predictive_effective_fps_matches_released_lingbot_gap_rule() -> None:
    horizons = torch.tensor([1, 2, 4, 8, 16, 32, 64], dtype=torch.long)
    result = predictive_effective_fps(horizons, source_fps=30.0)
    torch.testing.assert_close(result, 30.0 / horizons.float())
    assert result.device == horizons.device

    with pytest.raises(ValueError, match="positive"):
        predictive_effective_fps(torch.tensor([0]), source_fps=30.0)
    with pytest.raises(ContractError, match="source FPS must be positive"):
        predictive_effective_fps(torch.tensor([1]), source_fps=0.0)


def test_predictive_encoder_identity_binds_dynamic_fps_semantics() -> None:
    contract = _contract()
    different_source_rate = replace(contract, source_fps=15.0)

    assert contract.encoder_payload["source_fps"] == 30.0
    assert different_source_rate.encoder_payload["source_fps"] == 15.0
    assert contract.encoder_digest != different_source_rate.encoder_digest

    with pytest.raises(ContractError, match="effective-FPS semantics changed"):
        replace(contract, effective_fps_semantics="constant-one-fps/v1")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_fps", "30.0", "finite real"),
        ("effective_fps_semantics", 1, "non-empty text"),
        ("minimum_visible_fraction", True, "finite real"),
        ("use_warmup_frame", 1, "boolean"),
        ("horizons", [1, "2"], "positive integer"),
    ),
)
def test_predictive_contract_parser_rejects_wrong_dynamic_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = asdict(_contract())
    payload["horizons"] = list(payload["horizons"])
    payload[field] = value
    with pytest.raises(ContractError, match=message):
        LingBotPredictiveTargetCache._parse_contract(payload)


def _write(root: Path) -> tuple[PredictiveCacheContract, str]:
    contract = _contract()
    digest = write_predictive_target_cache(
        root,
        contract=contract,
        records=(
            _record(10, 1),
            _record(10, 2),
            _record(11, 1),
            _record(20, 1),
            _record(20, 2),
            _record(21, 1),
        ),
        shard_rows=2,
    )
    return contract, digest


def _load(
    root: Path,
    contract: PredictiveCacheContract,
    digest: str,
) -> LingBotPredictiveTargetCache:
    return LingBotPredictiveTargetCache.load(
        root,
        manifest_sha256=digest,
        dataset_tree_sha256=contract.dataset_tree_sha256,
        physical_sidecar_manifest_sha256=contract.physical_sidecar_manifest_sha256,
        encoder_digest=contract.encoder_digest,
        query_schema_sha256=contract.query_schema_sha256,
        coverage_sha256=contract.coverage_sha256,
        memory_capacity=1,
    )


def _request(
    horizons: torch.Tensor,
    *,
    source: PredictionSource = PredictionSource.PRIOR,
    evidence: PredictionEvidence = PredictionEvidence.FUTURE,
    valid: torch.Tensor | None = None,
) -> NativePredictionRequest:
    batch, queries = horizons.shape
    return NativePredictionRequest(
        source=source,
        evidence=evidence,
        route_ids=torch.zeros(batch, queries, dtype=torch.long),
        horizons=horizons,
        addresses=torch.empty(batch, queries, 0),
        valid=torch.ones(batch, queries, dtype=torch.bool) if valid is None else valid,
    )


def test_predictive_cache_record_iteration_is_complete_and_defensive(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    contract, digest = _write(root)
    cache = _load(root, contract, digest)

    records = tuple(cache.iter_records())

    assert len(records) == contract.expected_record_count
    assert [(value.source_global_index, value.horizon) for value in records] == [
        (10, 1),
        (10, 2),
        (11, 1),
        (20, 1),
        (20, 2),
        (21, 1),
    ]
    records[0].features[:] = 99
    assert not np.equal(next(cache.iter_records()).features, 99).any()
    selected = cache.record_for(source_global_index=10, horizon=1)
    assert selected is not None
    selected.features[:] = 77
    selected_again = cache.record_for(source_global_index=10, horizon=1)
    assert selected_again is not None and not np.equal(selected_again.features, 77).any()
    assert cache.record_for(source_global_index=999, horizon=1) is None
    assert cache.has_supported_target(source_global_index=10, horizon=1)
    assert not cache.has_supported_target(source_global_index=999, horizon=1)


def test_predictive_cache_reports_rows_without_visible_target_mass(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    contract = _contract(horizons=(1,))
    digest = write_predictive_target_cache(
        root,
        contract=contract,
        records=(_record(10, 1, supported=False),),
        shard_rows=1,
    )
    cache = _load(root, contract, digest)

    assert not cache.has_supported_target(source_global_index=10, horizon=1)


def test_predictive_target_audit_scans_complete_cache_and_detects_fixture_collapse(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    contract, digest = _write(root)
    cache = _load(root, contract, digest)

    first = audit_predictive_target_cache(cache, maximum_samples=4)
    second = audit_predictive_target_cache(cache, maximum_samples=4)

    assert first == second
    assert first["scanned_record_count"] == contract.expected_record_count
    assert first["supported_object_target_count"] == contract.expected_record_count
    assert first["sampled_target_count"] == 4
    assert first["visible_support_diagnostics"] == {
        "supported_count": contract.expected_record_count,
        "sampled_count": 4,
        "minimum_visible_image_fraction": 0.25,
        "mean_visible_image_fraction": 0.25,
        "maximum_visible_image_fraction": 0.25,
        "sampled_p05_visible_image_fraction": 0.25,
        "sampled_median_visible_image_fraction": 0.25,
        "sampled_p95_visible_image_fraction": 0.25,
    }
    assert first["interpretation"] == {
        "numerical_status": "obvious_target_collapse",
        "pretraining_readiness": "FAIL",
        "pretraining_readiness_failures": [
            "obvious_numerical_collapse",
            "insufficient_identity_or_target_group_support",
            "cross_frame_identity_retrieval_unavailable",
        ],
        "retrieval_is_computable": False,
        "scientific_acceptance": False,
        "scientific_acceptance_reason": (
            "target statistics cannot establish source-conditioned learnability, "
            "shared-host gradient reach, object semantics or action benefit"
        ),
    }


def test_predictive_target_audit_records_unreachable_declared_horizon_as_zero(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    contract = _contract(horizons=(1, 2, 64))
    digest = write_predictive_target_cache(
        root,
        contract=contract,
        records=(
            _record(10, 1),
            _record(10, 2),
            _record(11, 1),
            _record(20, 1),
            _record(20, 2),
            _record(21, 1),
        ),
        shard_rows=2,
    )
    cache = _load(root, contract, digest)

    report = audit_predictive_target_cache(cache, maximum_samples=4)

    assert report["horizon_record_counts"] == {"1": 4, "2": 2, "64": 0}


def test_predictive_cache_roundtrip_reorders_physical_identities_and_censors_occlusion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    contract, digest = _write(root)
    cache = _load(root, contract, digest)
    request = _request(torch.tensor([[1, 2], [2, 1]]))

    target = cache.target_for(
        source_global_indices=(10, 20),
        source_rgb_sha256=(_sha("source-10"), _sha("source-20")),
        track_identity_keys=(
            ("object/a", "object/b", "object/c"),
            ("object/b", "object/a"),
        ),
        request=request,
        device="cpu",
    )

    assert target.features.shape == (2, 3, 2, 1024)
    assert target.track_identity_keys[0] == ("object/a", "object/b", "object/c")
    assert not target.valid[0, 0].any()
    assert target.valid[0, 1].all()
    assert not target.valid[0, 2].any()
    torch.testing.assert_close(target.features[0, 1, 0], torch.full((1024,), 10.0))
    assert not target.features[0, 0].any()
    assert not target.importance[0, 0].any()
    assert tuple(cache._loaded) == (1,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_tree_sha256", _sha("wrong-data"), "another dataset tree"),
        ("physical_sidecar_manifest_sha256", _sha("wrong-sidecar"), "another physical"),
        ("encoder_digest", _sha("wrong-encoder"), "another teacher"),
        ("query_schema_sha256", _sha("wrong-query"), "query schema differs"),
        ("coverage_sha256", _sha("wrong-coverage"), "coverage differs"),
    ),
)
def test_predictive_cache_rejects_provenance_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    root = tmp_path / "cache"
    contract, digest = _write(root)
    arguments = {
        "manifest_sha256": digest,
        "dataset_tree_sha256": contract.dataset_tree_sha256,
        "physical_sidecar_manifest_sha256": contract.physical_sidecar_manifest_sha256,
        "encoder_digest": contract.encoder_digest,
        "query_schema_sha256": contract.query_schema_sha256,
        "coverage_sha256": contract.coverage_sha256,
    }
    arguments[field] = value
    with pytest.raises(ContractError, match=message):
        LingBotPredictiveTargetCache.load(root, **arguments)


def test_predictive_cache_rejects_pre_dynamic_fps_schema(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    contract, _digest = _write(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["schema"] = "picf-next.lingbot-predictive-object-cache/v2"
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii")
    manifest_path.write_bytes(payload)

    with pytest.raises(ContractError, match="schema changed"):
        LingBotPredictiveTargetCache.load(
            root,
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
            dataset_tree_sha256=contract.dataset_tree_sha256,
            physical_sidecar_manifest_sha256=contract.physical_sidecar_manifest_sha256,
            encoder_digest=contract.encoder_digest,
            query_schema_sha256=contract.query_schema_sha256,
            coverage_sha256=contract.coverage_sha256,
            memory_capacity=1,
        )


def test_predictive_cache_rejects_shard_mutation_and_source_frame_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    contract, digest = _write(root)
    manifest = json.loads((root / "manifest.json").read_text())
    shard_path = root / manifest["shards"][0]["path"]
    shard_path.write_bytes(shard_path.read_bytes() + b"corruption")
    with pytest.raises(ContractError, match="content hash mismatch"):
        _load(root, contract, digest)

    clean_root = tmp_path / "clean-cache"
    contract, digest = _write(clean_root)
    cache = _load(clean_root, contract, digest)
    with pytest.raises(ContractError, match="source RGB differs"):
        cache.target_for(
            source_global_indices=(10,),
            source_rgb_sha256=(_sha("wrong-source"),),
            track_identity_keys=(("object/a", "object/b"),),
            request=_request(torch.tensor([[1]])),
            device="cpu",
        )


def test_predictive_cache_serves_posterior_and_prior_future_query_contracts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    contract, digest = _write(root)
    cache = _load(root, contract, digest)
    common = {
        "source_global_indices": (10,),
        "source_rgb_sha256": (_sha("source-10"),),
        "track_identity_keys": (("object/a", "object/b"),),
        "device": "cpu",
    }
    posterior = cache.target_for(
        **common,
        request=_request(
            torch.tensor([[1]]),
            source=PredictionSource.POSTERIOR,
        ),
    )
    prior = cache.target_for(
        **common,
        request=_request(torch.tensor([[1]]), source=PredictionSource.PRIOR),
    )
    assert posterior.source is PredictionSource.POSTERIOR
    assert prior.source is PredictionSource.PRIOR
    torch.testing.assert_close(posterior.features, prior.features)
    with pytest.raises(ValueError, match="horizon differs"):
        cache.target_for(
            **common,
            request=_request(torch.tensor([[3]])),
        )


def test_dino_pooling_uses_visible_owner_area_and_omits_hidden_objects() -> None:
    patches = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, 16.0]],
    )
    owner_index = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    summaries, importance = pool_dino_object_summaries(
        patches,
        owner_index=owner_index,
        owner_supervised=np.ones((4, 4), dtype=np.bool_),
        identity_keys=("object/a", "object/b"),
        minimum_visible_fraction=0.0,
    )

    assert summaries.shape == (2, 2)
    np.testing.assert_allclose(importance, np.asarray([0.25, 0.0], dtype=np.float32))
    assert np.any(summaries[0] != 0)
    assert not np.any(summaries[1])


def test_dino_pooling_accepts_immutable_sidecar_rasters_without_warning() -> None:
    patches = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    owner_index = np.ones((4, 4), dtype=np.uint8)
    owner_supervised = np.ones((4, 4), dtype=np.bool_)
    owner_index.setflags(write=False)
    owner_supervised.setflags(write=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        summaries, importance = pool_dino_object_summaries(
            patches,
            owner_index=owner_index,
            owner_supervised=owner_supervised,
            identity_keys=("object/a",),
            minimum_visible_fraction=0.0,
        )

    assert not recorded
    assert summaries.shape == (1, 2)
    np.testing.assert_allclose(importance, np.ones((1,), dtype=np.float32))


def test_predictive_cache_publication_is_atomic_on_writer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected writer failure")

    monkeypatch.setattr(np, "savez", fail_save)
    with pytest.raises(OSError, match="injected"):
        write_predictive_target_cache(
            root,
            contract=_contract(horizons=(1,)),
            records=(_record(10, 1),),
            shard_rows=1,
        )

    assert not root.exists()
    assert not tuple(tmp_path.glob(".cache.*.tmp"))


def test_predictive_cache_unpublishes_after_post_rename_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    replaced = False
    original_replace = predictive_cache_module.os.replace
    original_fsync = predictive_cache_module.os.fsync

    def track_replace(source: Path, destination: Path) -> None:
        nonlocal replaced
        original_replace(source, destination)
        replaced = True

    def fail_parent_fsync(descriptor: int) -> None:
        if replaced:
            raise OSError("injected post-rename fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(predictive_cache_module.os, "replace", track_replace)
    monkeypatch.setattr(predictive_cache_module.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="post-rename"):
        write_predictive_target_cache(
            root,
            contract=_contract(horizons=(1,)),
            records=(_record(10, 1),),
            shard_rows=1,
        )

    assert not root.exists()
    assert not tuple(tmp_path.glob(".cache.*.tmp"))


def test_predictive_cache_detects_rename_completion_before_publication_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cache"
    original_replace = predictive_cache_module.os.replace

    def replace_then_raise(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        raise OSError("injected exception after rename syscall")

    monkeypatch.setattr(predictive_cache_module.os, "replace", replace_then_raise)
    with pytest.raises(OSError, match="after rename syscall"):
        write_predictive_target_cache(
            root,
            contract=_contract(horizons=(1,)),
            records=(_record(10, 1),),
            shard_rows=1,
        )

    assert not root.exists()
    assert not tuple(tmp_path.glob(".cache.*.tmp"))


def test_predictive_cache_refuses_existing_symlink_destination(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "cache"
    root.symlink_to(external, target_is_directory=True)

    with pytest.raises(FileExistsError):
        write_predictive_target_cache(
            root,
            contract=_contract(horizons=(1,)),
            records=(_record(10, 1),),
            shard_rows=1,
        )
    assert not tuple(external.iterdir())


@pytest.mark.parametrize(
    "records",
    (
        (_record(10, 1),),
        (_record(10, 2), _record(10, 1)),
        (_record(10, 1), _record(10, 2), _record(11, 1), _record(11, 1)),
    ),
)
def test_predictive_cache_requires_exact_complete_ordered_coverage(
    tmp_path: Path,
    records: tuple[PredictiveObjectCacheRecord, ...],
) -> None:
    with pytest.raises(ContractError, match="predictive cache"):
        write_predictive_target_cache(
            tmp_path / "cache",
            contract=_contract(),
            records=records,
        )
