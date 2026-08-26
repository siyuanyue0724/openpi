from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data.dense_evidence_cache import (
    DenseEvidenceCacheContract,
    DenseEvidenceCacheRecord,
    FrozenDenseEvidenceCache,
    FrozenDenseEvidenceCacheBank,
    FrozenDenseEvidenceCacheView,
    compose_dense_evidence_cache_banks,
    dense_evidence_cache_resume_state,
    merge_dense_evidence_cache_partitions,
    publish_dense_evidence_cache,
    publish_dense_evidence_cache_resumable,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _evidence(*, available: bool, value: float, grouped: bool = True) -> DenseEvidence:
    count = 2 if available else 0
    return DenseEvidence(
        modality="anytouch",
        encoder_contract="anytouch2@revision/frozen-cls/v1",
        tokens=np.full((count, 4), value, dtype=np.float32),
        available=available,
        timestamps=np.full(count, 0.25, dtype=np.float32),
        confidence=np.ones(count, dtype=np.float32),
        geometry=np.full((count, 2), value / 10.0, dtype=np.float32),
        group_ids=np.zeros(count, dtype=np.int64) if grouped else None,
        current_measurement_valid=np.ones(count, dtype=np.bool_),
    )


def _contract(*, grouped: bool = True) -> DenseEvidenceCacheContract:
    return DenseEvidenceCacheContract(
        dataset_id="calvin",
        dataset_revision="revision",
        dataset_tree_sha256=_digest("tree"),
        coverage_plan_sha256=_digest("coverage"),
        modality="anytouch",
        encoder_contract="anytouch2@revision/frozen-cls/v1",
        token_width=4,
        geometry_width=2,
        maximum_tokens=2,
        has_group_ids=grouped,
    )


def _records() -> tuple[DenseEvidenceCacheRecord, ...]:
    return tuple(
        DenseEvidenceCacheRecord(
            source_global_index=index,
            sample_key=f"sample-{index}",
            source_input_sha256=_digest(f"input-{index}"),
            evidence=_evidence(available=index != 11, value=float(index)),
        )
        for index in (10, 11, 12)
    )


def test_dense_evidence_cache_round_trip_is_lazy_and_lossless(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    manifest_sha = publish_dense_evidence_cache(
        root,
        contract=_contract(),
        records=_records(),
        shard_rows=2,
    )
    cache = FrozenDenseEvidenceCache.load(
        root,
        manifest_sha256=manifest_sha,
        dataset_tree_sha256=_digest("tree"),
        memory_capacity=1,
    )

    assert len(cache.shards) == 2
    assert not cache._loaded
    active = cache.evidence_for(
        source_global_index=10,
        sample_key="sample-10",
        source_input_sha256=_digest("input-10"),
    )
    assert active.available
    assert active.tokens.dtype == np.float16
    np.testing.assert_array_equal(active.tokens, np.full((2, 4), 10.0, dtype=np.float16))
    np.testing.assert_array_equal(active.group_ids, np.zeros(2, dtype=np.int64))
    assert not active.tokens.flags.writeable

    missing = cache.evidence_for(source_global_index=11, sample_key="sample-11")
    assert not missing.available
    assert missing.token_count == 0

    cache.evidence_for(source_global_index=12, sample_key="sample-12")
    assert tuple(cache._loaded) == (1,)


def test_dense_evidence_cache_view_composes_exact_authenticated_union(tmp_path: Path) -> None:
    rows = _records()
    primary_root = tmp_path / "primary"
    supplement_root = tmp_path / "supplement"
    primary_sha = publish_dense_evidence_cache(
        primary_root,
        contract=_contract(),
        records=rows[:2],
        shard_rows=1,
    )
    supplement_sha = publish_dense_evidence_cache(
        supplement_root,
        contract=replace(_contract(), coverage_plan_sha256=_digest("supplement")),
        records=rows[1:],
        shard_rows=1,
    )
    primary = FrozenDenseEvidenceCacheBank.load(
        (primary_root,),
        manifest_sha256s=(primary_sha,),
        dataset_tree_sha256=_digest("tree"),
    )
    supplement = FrozenDenseEvidenceCacheBank.load(
        (supplement_root,),
        manifest_sha256s=(supplement_sha,),
        dataset_tree_sha256=_digest("tree"),
    )

    target_coverage = _digest("target")
    view = compose_dense_evidence_cache_banks(
        (primary, supplement),
        record_identities=tuple((row.source_global_index, row.sample_key) for row in rows),
        coverage_plan_sha256=target_coverage,
    )

    assert view.coverage_plan_sha256 == target_coverage
    assert view.record_count == 3
    assert isinstance(view.caches[0], FrozenDenseEvidenceCacheView)
    assert view.caches[0].source_record_counts == (2, 1)
    for row in rows:
        actual = view.evidence_for(
            source_global_index=row.source_global_index,
            sample_key=row.sample_key,
        )[0]
        np.testing.assert_array_equal(actual.tokens, row.evidence.tokens.astype(np.float16))


def test_dense_evidence_cache_view_rejects_missing_or_conflicting_records(
    tmp_path: Path,
) -> None:
    rows = _records()
    primary_root = tmp_path / "primary"
    primary_sha = publish_dense_evidence_cache(
        primary_root,
        contract=_contract(),
        records=rows[:2],
    )
    primary = FrozenDenseEvidenceCacheBank.load(
        (primary_root,),
        manifest_sha256s=(primary_sha,),
        dataset_tree_sha256=_digest("tree"),
    )
    identities = tuple((row.source_global_index, row.sample_key) for row in rows)
    with pytest.raises(ContractError, match="cover every target"):
        compose_dense_evidence_cache_banks(
            (primary,),
            record_identities=identities,
            coverage_plan_sha256=_digest("target"),
        )

    conflicting = DenseEvidenceCacheRecord(
        source_global_index=rows[0].source_global_index,
        sample_key=rows[0].sample_key,
        source_input_sha256=_digest("conflicting-input"),
        evidence=rows[0].evidence,
    )
    conflict_root = tmp_path / "conflict"
    conflict_sha = publish_dense_evidence_cache(
        conflict_root,
        contract=replace(_contract(), coverage_plan_sha256=_digest("conflict")),
        records=(conflicting,),
    )
    conflict = FrozenDenseEvidenceCacheBank.load(
        (conflict_root,),
        manifest_sha256s=(conflict_sha,),
        dataset_tree_sha256=_digest("tree"),
    )
    with pytest.raises(ContractError, match="overlapping metadata"):
        compose_dense_evidence_cache_banks(
            (primary, conflict),
            record_identities=((rows[0].source_global_index, rows[0].sample_key),),
            coverage_plan_sha256=_digest("target"),
        )


def test_dense_evidence_cache_reader_bounds_high_dimensional_geometry(tmp_path: Path) -> None:
    """Every shard accepted by the publisher must fit the verified reader bound."""

    token_count = 1024
    geometry_width = 512
    contract = DenseEvidenceCacheContract(
        dataset_id="calvin",
        dataset_revision="revision",
        dataset_tree_sha256=_digest("tree"),
        coverage_plan_sha256=_digest("coverage"),
        modality="anytouch",
        encoder_contract="anytouch2@revision/high-dimensional-geometry/v1",
        token_width=4,
        geometry_width=geometry_width,
        maximum_tokens=token_count,
        has_group_ids=True,
    )
    random = np.random.default_rng(7)
    evidence = DenseEvidence(
        modality=contract.modality,
        encoder_contract=contract.encoder_contract,
        tokens=random.standard_normal((token_count, contract.token_width)).astype(np.float32),
        available=True,
        timestamps=np.linspace(0.0, 1.0, token_count, dtype=np.float64),
        confidence=np.ones(token_count, dtype=np.float32),
        geometry=random.standard_normal((token_count, geometry_width)).astype(np.float32),
        group_ids=np.arange(token_count, dtype=np.int64),
        current_measurement_valid=np.arange(token_count) == token_count - 1,
    )
    record = DenseEvidenceCacheRecord(
        source_global_index=10,
        sample_key="sample-with-high-dimensional-geometry",
        source_input_sha256=_digest("high-dimensional-input"),
        evidence=evidence,
    )
    root = tmp_path / "high-dimensional"
    manifest_sha = publish_dense_evidence_cache(root, contract=contract, records=(record,))

    cache = FrozenDenseEvidenceCache.load(
        root,
        manifest_sha256=manifest_sha,
        dataset_tree_sha256=_digest("tree"),
    )
    loaded = cache.evidence_for(source_global_index=10, sample_key=record.sample_key)
    assert loaded.geometry is not None
    assert loaded.geometry.shape == (token_count, geometry_width)


def test_dense_evidence_cache_publisher_flushes_a_one_pass_stream_by_shard(
    tmp_path: Path,
) -> None:
    root = tmp_path / "streamed"
    staging = root.with_name(f".{root.name}.staging-{os.getpid()}")
    rows = _records()

    def one_pass_records():
        yield rows[0]
        yield rows[1]
        assert (staging / "shard-000000.npz").is_file()
        yield rows[2]

    manifest_sha = publish_dense_evidence_cache(
        root,
        contract=_contract(),
        records=one_pass_records(),
        shard_rows=2,
    )

    cache = FrozenDenseEvidenceCache.load(
        root,
        manifest_sha256=manifest_sha,
        dataset_tree_sha256=_digest("tree"),
    )
    assert tuple(shard.row_count for shard in cache.shards) == (2, 1)


def test_dense_evidence_cache_partition_merge_is_exact_and_order_independent(
    tmp_path: Path,
) -> None:
    rows = _records()
    first_root = tmp_path / "part-0"
    second_root = tmp_path / "part-1"
    first_sha = publish_dense_evidence_cache(
        first_root,
        contract=_contract(),
        records=rows[:2],
        shard_rows=1,
    )
    second_sha = publish_dense_evidence_cache(
        second_root,
        contract=_contract(),
        records=rows[2:],
        shard_rows=1,
    )

    merged_root = tmp_path / "merged"
    merged_sha = merge_dense_evidence_cache_partitions(
        merged_root,
        partition_roots=(second_root, first_root),
        manifest_sha256s=(second_sha, first_sha),
        dataset_tree_sha256=_digest("tree"),
        coverage_plan_sha256=_digest("coverage"),
        expected_records=tuple((row.source_global_index, row.sample_key) for row in rows),
        shard_rows=2,
    )
    merged = FrozenDenseEvidenceCache.load(
        merged_root,
        manifest_sha256=merged_sha,
        dataset_tree_sha256=_digest("tree"),
    )

    assert tuple((row.source_global_index, row.sample_key) for row in merged.records) == tuple(
        (row.source_global_index, row.sample_key) for row in rows
    )
    for row in rows:
        actual = merged.evidence_for(
            source_global_index=row.source_global_index,
            sample_key=row.sample_key,
            source_input_sha256=row.source_input_sha256,
        )
        np.testing.assert_array_equal(actual.tokens, row.evidence.tokens.astype(np.float16))


def test_dense_evidence_cache_partition_merge_can_hardlink_verified_shards(
    tmp_path: Path,
) -> None:
    rows = _records()
    first_root = tmp_path / "part-0"
    second_root = tmp_path / "part-1"
    first_sha = publish_dense_evidence_cache(
        first_root,
        contract=_contract(),
        records=rows[:2],
        shard_rows=1,
    )
    second_sha = publish_dense_evidence_cache(
        second_root,
        contract=_contract(),
        records=rows[2:],
        shard_rows=1,
    )

    merged_root = tmp_path / "merged-linked"
    merged_sha = merge_dense_evidence_cache_partitions(
        merged_root,
        partition_roots=(second_root, first_root),
        manifest_sha256s=(second_sha, first_sha),
        dataset_tree_sha256=_digest("tree"),
        coverage_plan_sha256=_digest("coverage"),
        expected_records=tuple((row.source_global_index, row.sample_key) for row in rows),
        link_shards=True,
    )
    merged = FrozenDenseEvidenceCache.load(
        merged_root,
        manifest_sha256=merged_sha,
        dataset_tree_sha256=_digest("tree"),
    )

    assert (first_root / "shard-000000.npz").stat().st_ino == (
        merged_root / "shard-000000.npz"
    ).stat().st_ino
    assert (second_root / "shard-000000.npz").stat().st_ino == (
        merged_root / "shard-000002.npz"
    ).stat().st_ino
    for row in rows:
        actual = merged.evidence_for(
            source_global_index=row.source_global_index,
            sample_key=row.sample_key,
            source_input_sha256=row.source_input_sha256,
        )
        np.testing.assert_array_equal(actual.tokens, row.evidence.tokens.astype(np.float16))


def test_dense_evidence_cache_partition_index_is_exact_without_copying_shards(
    tmp_path: Path,
) -> None:
    rows = _records()
    namespace = tmp_path / "namespace"
    first_root = namespace / "parts" / "part-0"
    second_root = namespace / "parts" / "part-1"
    first_sha = publish_dense_evidence_cache(
        first_root,
        contract=_contract(),
        records=rows[:2],
        shard_rows=1,
    )
    second_sha = publish_dense_evidence_cache(
        second_root,
        contract=_contract(),
        records=rows[2:],
        shard_rows=1,
    )

    indexed_root = namespace / "indexed"
    indexed_sha = merge_dense_evidence_cache_partitions(
        indexed_root,
        partition_roots=(second_root, first_root),
        manifest_sha256s=(second_sha, first_sha),
        dataset_tree_sha256=_digest("tree"),
        coverage_plan_sha256=_digest("coverage"),
        expected_records=tuple((row.source_global_index, row.sample_key) for row in rows),
        reference_partitions=True,
    )
    assert tuple(path.name for path in indexed_root.iterdir()) == ("manifest.json",)

    bank = FrozenDenseEvidenceCacheBank.load(
        (indexed_root,),
        manifest_sha256s=(indexed_sha,),
        dataset_tree_sha256=_digest("tree"),
    )
    indexed = bank.caches[0]
    assert tuple((row.source_global_index, row.sample_key) for row in indexed.records) == tuple(
        (row.source_global_index, row.sample_key) for row in rows
    )
    for row in rows:
        actual = indexed.evidence_for(
            source_global_index=row.source_global_index,
            sample_key=row.sample_key,
            source_input_sha256=row.source_input_sha256,
        )
        np.testing.assert_array_equal(actual.tokens, row.evidence.tokens.astype(np.float16))

    manifest_path = first_root / "manifest.json"
    manifest_path.write_text(manifest_path.read_text(encoding="ascii") + "\n", encoding="ascii")
    with pytest.raises(ContractError, match="hash mismatch"):
        FrozenDenseEvidenceCacheBank.load(
            (indexed_root,),
            manifest_sha256s=(indexed_sha,),
            dataset_tree_sha256=_digest("tree"),
        )


def test_dense_evidence_cache_partition_index_rejects_roots_outside_namespace(
    tmp_path: Path,
) -> None:
    rows = _records()
    partition_root = tmp_path / "outside" / "part"
    manifest_sha = publish_dense_evidence_cache(
        partition_root,
        contract=_contract(),
        records=rows,
    )
    with pytest.raises(ContractError, match="output namespace"):
        merge_dense_evidence_cache_partitions(
            tmp_path / "namespace" / "indexed",
            partition_roots=(partition_root,),
            manifest_sha256s=(manifest_sha,),
            dataset_tree_sha256=_digest("tree"),
            coverage_plan_sha256=_digest("coverage"),
            expected_records=tuple((row.source_global_index, row.sample_key) for row in rows),
            reference_partitions=True,
        )


def test_dense_evidence_cache_partition_merge_rejects_missing_coverage(tmp_path: Path) -> None:
    rows = _records()
    root = tmp_path / "part"
    manifest_sha = publish_dense_evidence_cache(
        root,
        contract=_contract(),
        records=rows[:2],
    )

    with pytest.raises(ContractError, match="exactly cover"):
        merge_dense_evidence_cache_partitions(
            tmp_path / "merged",
            partition_roots=(root,),
            manifest_sha256s=(manifest_sha,),
            dataset_tree_sha256=_digest("tree"),
            coverage_plan_sha256=_digest("coverage"),
            expected_records=tuple((row.source_global_index, row.sample_key) for row in rows),
        )


def test_dense_evidence_cache_rejects_runtime_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    manifest_sha = publish_dense_evidence_cache(root, contract=_contract(), records=_records())
    cache = FrozenDenseEvidenceCache.load(
        root,
        manifest_sha256=manifest_sha,
        dataset_tree_sha256=_digest("tree"),
    )
    with pytest.raises(ContractError, match="sample key"):
        cache.evidence_for(source_global_index=10, sample_key="sample-11")
    with pytest.raises(ContractError, match="runtime source input"):
        cache.evidence_for(
            source_global_index=10,
            sample_key="sample-10",
            source_input_sha256=_digest("wrong"),
        )


def test_dense_evidence_cache_rejects_manifest_and_shard_tampering(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    manifest_sha = publish_dense_evidence_cache(root, contract=_contract(), records=_records())
    with pytest.raises(ContractError, match="belongs to another dataset tree"):
        FrozenDenseEvidenceCache.load(
            root,
            manifest_sha256=manifest_sha,
            dataset_tree_sha256=_digest("another-tree"),
        )

    shard = root / "shard-000000.npz"
    shard.write_bytes(shard.read_bytes() + b"tamper")
    cache = FrozenDenseEvidenceCache.load(
        root,
        manifest_sha256=manifest_sha,
        dataset_tree_sha256=_digest("tree"),
    )
    with pytest.raises(ContractError, match="hash"):
        cache.evidence_for(source_global_index=10, sample_key="sample-10")


def test_dense_evidence_cache_rejects_group_contract_drift(tmp_path: Path) -> None:
    record = DenseEvidenceCacheRecord(
        source_global_index=1,
        sample_key="sample-1",
        source_input_sha256=_digest("input"),
        evidence=_evidence(available=True, value=1.0, grouped=False),
    )
    with pytest.raises(ContractError, match="grouping"):
        publish_dense_evidence_cache(
            tmp_path / "cache",
            contract=_contract(grouped=True),
            records=(record,),
        )


def test_dense_evidence_cache_record_table_is_hashed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    publish_dense_evidence_cache(root, contract=_contract(), records=_records())
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    payload["records"][0]["sample_key"] = "mutated"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="ascii")
    changed_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ContractError, match="record table hash"):
        FrozenDenseEvidenceCache.load(
            root,
            manifest_sha256=changed_manifest_sha,
            dataset_tree_sha256=_digest("tree"),
        )


def test_dense_evidence_cache_bank_requires_exact_source_alignment(tmp_path: Path) -> None:
    touch_root = tmp_path / "touch"
    touch_sha = publish_dense_evidence_cache(
        touch_root,
        contract=_contract(),
        records=_records(),
        shard_rows=2,
    )
    geometry_contract = DenseEvidenceCacheContract(
        dataset_id="calvin",
        dataset_revision="revision",
        dataset_tree_sha256=_digest("tree"),
        coverage_plan_sha256=_digest("coverage"),
        modality="geometry",
        encoder_contract="sonata@revision/final-hierarchy/v1",
        token_width=4,
        geometry_width=2,
        maximum_tokens=2,
    )
    geometry_records = tuple(
        DenseEvidenceCacheRecord(
            source_global_index=record.source_global_index,
            sample_key=record.sample_key,
            source_input_sha256=record.source_input_sha256,
            evidence=DenseEvidence(
                modality="geometry",
                encoder_contract=geometry_contract.encoder_contract,
                tokens=record.evidence.tokens,
                available=record.evidence.available,
                timestamps=record.evidence.timestamps,
                confidence=record.evidence.confidence,
                geometry=record.evidence.geometry,
                current_measurement_valid=record.evidence.current_measurement_valid,
            ),
        )
        for record in _records()
    )
    geometry_root = tmp_path / "geometry"
    geometry_sha = publish_dense_evidence_cache(
        geometry_root,
        contract=geometry_contract,
        records=geometry_records,
        shard_rows=2,
    )
    bank = FrozenDenseEvidenceCacheBank.load(
        (touch_root, geometry_root),
        manifest_sha256s=(touch_sha, geometry_sha),
        dataset_tree_sha256=_digest("tree"),
        memory_capacity=1,
    )

    assert bank.modalities == ("anytouch", "geometry")
    assert bank.record_count == 3
    evidence = bank.evidence_for(source_global_index=10, sample_key="sample-10")
    assert tuple(item.modality for item in evidence) == bank.modalities

    misaligned_root = tmp_path / "misaligned"
    misaligned = tuple(
        DenseEvidenceCacheRecord(
            source_global_index=record.source_global_index,
            sample_key=("different" if index == 0 else record.sample_key),
            source_input_sha256=record.source_input_sha256,
            evidence=geometry_records[index].evidence,
        )
        for index, record in enumerate(_records())
    )
    misaligned_sha = publish_dense_evidence_cache(
        misaligned_root,
        contract=geometry_contract,
        records=misaligned,
    )
    with pytest.raises(ContractError, match="identical source records"):
        FrozenDenseEvidenceCacheBank.load(
            (touch_root, misaligned_root),
            manifest_sha256s=(touch_sha, misaligned_sha),
            dataset_tree_sha256=_digest("tree"),
        )


def test_resumable_dense_evidence_cache_skips_authenticated_shards_after_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resumable"
    rows = _records()
    factory_offsets: list[int] = []

    def interrupted_factory(offset: int):
        factory_offsets.append(offset)
        yield rows[0]
        yield rows[1]
        raise RuntimeError("simulated preemption")

    with pytest.raises(RuntimeError, match="simulated preemption"):
        publish_dense_evidence_cache_resumable(
            root,
            contract=_contract(),
            expected_record_count=len(rows),
            record_factory=interrupted_factory,
            shard_rows=2,
        )

    state = dense_evidence_cache_resume_state(
        root,
        contract=_contract(),
        expected_record_count=len(rows),
        shard_rows=2,
    )
    assert state.completed_record_count == 2
    assert state.last_source_global_index == 11

    def resumed_factory(offset: int):
        factory_offsets.append(offset)
        yield from rows[offset:]

    manifest_sha = publish_dense_evidence_cache_resumable(
        root,
        contract=_contract(),
        expected_record_count=len(rows),
        record_factory=resumed_factory,
        shard_rows=2,
    )
    assert factory_offsets == [0, 2]
    cache = FrozenDenseEvidenceCache.load(
        root,
        manifest_sha256=manifest_sha,
        dataset_tree_sha256=_digest("tree"),
    )
    assert tuple(shard.row_count for shard in cache.shards) == (2, 1)
    assert not root.with_name(f".{root.name}.partial").exists()


def test_resumable_dense_evidence_cache_rejects_contract_drift(tmp_path: Path) -> None:
    root = tmp_path / "resumable"
    rows = _records()

    def interrupted_factory(_offset: int):
        yield rows[0]
        yield rows[1]
        raise RuntimeError("simulated preemption")

    with pytest.raises(RuntimeError):
        publish_dense_evidence_cache_resumable(
            root,
            contract=_contract(),
            expected_record_count=len(rows),
            record_factory=interrupted_factory,
            shard_rows=2,
        )
    changed = DenseEvidenceCacheContract(
        dataset_id="calvin",
        dataset_revision="revision",
        dataset_tree_sha256=_digest("tree"),
        coverage_plan_sha256=_digest("coverage"),
        modality="anytouch",
        encoder_contract="anytouch2@another-revision/frozen-cls/v1",
        token_width=4,
        geometry_width=2,
        maximum_tokens=2,
        has_group_ids=True,
    )
    with pytest.raises(ContractError, match="contract changed"):
        dense_evidence_cache_resume_state(
            root,
            contract=changed,
            expected_record_count=len(rows),
            shard_rows=2,
        )


def test_resumable_dense_evidence_cache_revalidates_committed_shards(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resumable"
    rows = _records()

    def interrupted_factory(_offset: int):
        yield rows[0]
        yield rows[1]
        raise RuntimeError("simulated preemption")

    with pytest.raises(RuntimeError):
        publish_dense_evidence_cache_resumable(
            root,
            contract=_contract(),
            expected_record_count=len(rows),
            record_factory=interrupted_factory,
            shard_rows=2,
        )
    staging = root.with_name(f".{root.name}.partial")
    shard = staging / "shard-000000.npz"
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ContractError, match="hash"):
        dense_evidence_cache_resume_state(
            root,
            contract=_contract(),
            expected_record_count=len(rows),
            shard_rows=2,
        )
