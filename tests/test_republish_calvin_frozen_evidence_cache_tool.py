from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data import calvin_official_source as official_source
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    build_dataset_file_manifest,
    content_identified_dataset_manifest,
    file_sha256,
)
from picf_next.data.dense_evidence_cache import (
    DenseEvidenceCacheContract,
    DenseEvidenceCacheRecord,
    FrozenDenseEvidenceCache,
    merge_dense_evidence_cache_partitions,
    publish_dense_evidence_cache,
)
from picf_next.data.dense_evidence_coverage import (
    DenseEvidenceCoveragePlan,
    DenseEvidenceCoverageRecord,
)
from tests.test_calvin_data import _split_manifest, _write_split
from tools import republish_calvin_frozen_evidence_cache as tool


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _write_manifest(path: Path, manifest: DatasetFileManifest) -> None:
    path.write_bytes(_json_bytes(manifest.to_dict()))


def _relabel_manifest(
    split: Path,
    source: DatasetFileManifest,
    *,
    dataset_id: str,
    dataset_revision: str,
) -> DatasetFileManifest:
    return build_dataset_file_manifest(
        split,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        split_name=source.split_name,
        relative_paths=tuple(record.path for record in source.files),
    )


def _official_source_claims(selected_file_count: int) -> dict[str, object]:
    return {
        "official_archive": {
            "url": official_source.CALVIN_OFFICIAL_ARCHIVE_URL,
            "transport": "http",
            "content_length": official_source.CALVIN_OFFICIAL_ARCHIVE_CONTENT_LENGTH,
            "last_modified": official_source.CALVIN_OFFICIAL_ARCHIVE_LAST_MODIFIED,
            "etag": official_source.CALVIN_OFFICIAL_ARCHIVE_ETAG,
            "tail_size_bytes": official_source.CALVIN_OFFICIAL_ARCHIVE_TAIL_SIZE_BYTES,
            "tail_sha256": official_source.CALVIN_OFFICIAL_ARCHIVE_TAIL_SHA256,
            "central_directory_offset": (official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_OFFSET),
            "central_directory_size": official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SIZE,
            "central_directory_sha256": (official_source.CALVIN_OFFICIAL_CENTRAL_DIRECTORY_SHA256),
            "entry_count": official_source.CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT,
            "zip64": True,
            "publisher_authenticity": official_source.CALVIN_OFFICIAL_PUBLISHER_AUTHENTICITY,
        },
        "official_training_inventory": {
            "archive_prefix": official_source.CALVIN_OFFICIAL_TRAINING_PREFIX,
            "archive_entry_count": official_source.CALVIN_OFFICIAL_ARCHIVE_ENTRY_COUNT,
            "file_count": selected_file_count
            + len(official_source.CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES),
            "excluded_non_runtime_files": list(
                official_source.CALVIN_OFFICIAL_NON_RUNTIME_TRAINING_FILES
            ),
        },
    }


def _coverage(
    manifest: DatasetFileManifest,
    records: tuple[DenseEvidenceCoverageRecord, ...],
    *,
    name: str,
) -> DenseEvidenceCoveragePlan:
    training_count = sum(record.partition == "training" for record in records)
    evaluation_count = sum(record.partition == "evaluation" for record in records)
    return DenseEvidenceCoveragePlan(
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_tree_sha256=manifest.tree_sha256,
        stream_plan_sha256=_digest(f"{name}-stream"),
        representation_split_sha256=_digest(f"{name}-split"),
        evaluation_plan_sha256=_digest(f"{name}-evaluation"),
        training_visit_count=training_count,
        training_visits_sha256=_digest(f"{name}-visits"),
        evaluation_item_count=evaluation_count,
        records=records,
    )


def _evidence(*, value: float, encoder_contract: str = "vjepa.test/frozen/v1") -> DenseEvidence:
    return DenseEvidence(
        modality="vjepa",
        encoder_contract=encoder_contract,
        tokens=np.full((1, 2), value, dtype=np.float32),
        available=True,
        timestamps=np.asarray([0.25], dtype=np.float64),
        confidence=np.ones(1, dtype=np.float32),
        current_measurement_valid=np.ones(1, dtype=np.bool_),
    )


def _contract(
    manifest: DatasetFileManifest,
    coverage: DenseEvidenceCoveragePlan,
    *,
    encoder_contract: str = "vjepa.test/frozen/v1",
) -> DenseEvidenceCacheContract:
    return DenseEvidenceCacheContract(
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_tree_sha256=manifest.tree_sha256,
        coverage_plan_sha256=coverage.artifact_sha256,
        modality="vjepa",
        encoder_contract=encoder_contract,
        token_width=2,
        geometry_width=0,
        maximum_tokens=1,
        token_dtype="float16",
    )


class _Builder:
    def __init__(
        self,
        contract: DenseEvidenceCacheContract,
        source_index_by_key: dict[str, int],
    ) -> None:
        self.cache_contract = contract
        self.source_index_by_key = source_index_by_key
        self.calls: list[tuple[str, ...]] = []

    def records_for_sample_keys(
        self,
        sample_keys: tuple[str, ...],
    ) -> tuple[DenseEvidenceCacheRecord, ...]:
        self.calls.append(sample_keys)
        return tuple(
            DenseEvidenceCacheRecord(
                source_global_index=self.source_index_by_key[sample_key],
                sample_key=sample_key,
                source_input_sha256=_digest(f"encoded-{sample_key}"),
                evidence=_evidence(value=99.0),
            )
            for sample_key in sample_keys
        )


def _fixture(tmp_path: Path) -> SimpleNamespace:
    split = tmp_path / "training"
    _write_split(split)
    receipt_source_manifest = _split_manifest(split)
    donor_manifest = _relabel_manifest(
        split,
        receipt_source_manifest,
        dataset_id="calvin-donor",
        dataset_revision="donor-revision",
    )
    target_manifest = content_identified_dataset_manifest(
        receipt_source_manifest,
        dataset_id=official_source.CALVIN_OFFICIAL_DATASET_ID,
    )

    source_manifest_path = tmp_path / "receipt-source-manifest.json"
    donor_manifest_path = tmp_path / "donor-manifest.json"
    target_manifest_path = tmp_path / official_source.CALVIN_OFFICIAL_MANIFEST_NAME
    _write_manifest(source_manifest_path, receipt_source_manifest)
    _write_manifest(donor_manifest_path, donor_manifest)
    _write_manifest(target_manifest_path, target_manifest)

    receipt = {
        **_official_source_claims(len(target_manifest.files)),
        "schema": official_source.CALVIN_OFFICIAL_SOURCE_RECEIPT_SCHEMA,
        "source_manifest": {
            "file_sha256": file_sha256(source_manifest_path),
            "tree_sha256": receipt_source_manifest.tree_sha256,
            "declared_dataset_id": receipt_source_manifest.dataset_id,
            "declared_dataset_revision": receipt_source_manifest.dataset_revision,
        },
        "migrated_manifest": {
            "file_name": official_source.CALVIN_OFFICIAL_MANIFEST_NAME,
            "file_sha256": file_sha256(target_manifest_path),
            "tree_sha256": target_manifest.tree_sha256,
        },
        "verified_content": {
            "dataset_id": target_manifest.dataset_id,
            "dataset_revision": target_manifest.dataset_revision,
            "content_sha256": target_manifest.content_sha256,
            "split_name": target_manifest.split_name,
            "file_count": len(target_manifest.files),
            "total_size_bytes": target_manifest.total_size_bytes,
            "verification_mode": (official_source.CALVIN_OFFICIAL_SOURCE_VERIFICATION_MODE),
            "all_manifest_sha256_matches": True,
            "all_official_crc32_matches": True,
            "official_inventory_exact_after_declared_exclusions": True,
        },
        "training_authorized": False,
    }
    source_receipt_path = tmp_path / "official-source-receipt.json"
    source_receipt_path.write_bytes(_json_bytes(receipt))

    index = CalvinDatasetIndex.load(
        split,
        dataset_id=target_manifest.dataset_id,
        dataset_revision=target_manifest.dataset_revision,
        dataset_manifest=target_manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)
    sample_keys = dataset.sample_keys[:3]
    identities = tuple(
        (dataset.source_global_index_by_key(sample_key), sample_key) for sample_key in sample_keys
    )
    target_records = (
        DenseEvidenceCoverageRecord(*identities[0], "training"),
        DenseEvidenceCoverageRecord(*identities[1], "training"),
        DenseEvidenceCoverageRecord(*identities[2], "evaluation"),
    )
    donor_records = (target_records[0], target_records[2])
    target_coverage = _coverage(target_manifest, target_records, name="target")
    donor_coverage = _coverage(donor_manifest, donor_records, name="donor")
    target_coverage_path = tmp_path / "target-coverage.json"
    donor_coverage_path = tmp_path / "donor-coverage.json"
    target_coverage.write(target_coverage_path)
    donor_coverage.write(donor_coverage_path)

    donor_contract = _contract(donor_manifest, donor_coverage)
    frozen_records = tuple(
        DenseEvidenceCacheRecord(
            source_global_index=record.source_global_index,
            sample_key=record.sample_key,
            source_input_sha256=_digest(f"donor-{record.sample_key}"),
            evidence=_evidence(value=float(position + 1)),
        )
        for position, record in enumerate(donor_records)
    )
    donor_cache_root = tmp_path / "donor-cache"
    donor_cache_manifest_sha256 = publish_dense_evidence_cache(
        donor_cache_root,
        contract=donor_contract,
        records=frozen_records,
        shard_rows=1,
    )

    builder = _Builder(
        _contract(target_manifest, target_coverage),
        dict((sample_key, source_index) for source_index, sample_key in identities),
    )
    asset_manifest = tmp_path / "assets.json"
    asset_manifest.write_bytes(b"{}\n")
    args = argparse.Namespace(
        action_horizon=1,
        asset_manifest=asset_manifest,
        camera_calibration=None,
        coverage_plan=target_coverage_path,
        coverage_plan_sha256=file_sha256(target_coverage_path),
        dataset_root=tmp_path,
        device=None,
        donor_cache_manifest_sha256=donor_cache_manifest_sha256,
        donor_cache_root=donor_cache_root,
        donor_coverage_plan=donor_coverage_path,
        donor_coverage_plan_sha256=file_sha256(donor_coverage_path),
        donor_dataset_manifest=donor_manifest_path,
        encoder_batch_size=8,
        modality="vjepa",
        output_root=tmp_path / "republished-cache",
        point_budget=4096,
        point_pixel_stride=2,
        record_start=0,
        record_stop=None,
        receipt_output=tmp_path / "republished-cache.receipt.json",
        shard_rows=1,
        source_dataset_manifest=source_manifest_path,
        source_receipt=source_receipt_path,
        source_receipt_sha256=file_sha256(source_receipt_path),
        split="training",
        tactile_calibration_archive=None,
        tactile_calibration_receipt=None,
        tactile_calibration_receipt_sha256=None,
        target_dataset_manifest=target_manifest_path,
        token_dtype="float16",
        verify_all_dataset_files=False,
    )
    return SimpleNamespace(
        args=args,
        builder=builder,
        donor_contract=donor_contract,
        donor_coverage=donor_coverage,
        donor_manifest=donor_manifest,
        donor_records=frozen_records,
        identities=identities,
        target_manifest=target_manifest,
    )


def test_republisher_reuses_exact_rows_and_encodes_only_target_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)

    receipt = tool._run(fixture.args)

    assert receipt["reused_record_count"] == 2
    assert receipt["encoded_record_count"] == 1
    assert fixture.builder.calls == [(fixture.identities[1][1],)]
    assert (
        receipt["donor"]["dataset_manifest"]["file_sha256"]
        != receipt["official_source"]["original_dataset_manifest"]["file_sha256"]
    )
    assert json.loads(fixture.args.receipt_output.read_text(encoding="ascii")) == receipt

    cache = FrozenDenseEvidenceCache.load(
        fixture.args.output_root,
        manifest_sha256=receipt["cache"]["manifest_sha256"],
        dataset_tree_sha256=fixture.target_manifest.tree_sha256,
    )
    values = tuple(
        float(
            cache.evidence_for(source_global_index=source_index, sample_key=sample_key).tokens[0, 0]
        )
        for source_index, sample_key in fixture.identities
    )
    assert values == (1.0, 99.0, 2.0)
    assert receipt["reuse_policy"] == {
        "directory_discovery": False,
        "match_fields": ["source_global_index", "sample_key"],
        "semantic_heads": False,
        "symlink_reuse": False,
    }
    assert receipt["recovered_complete_cache"] is False
    assert receipt["publication"] == {
        "full_coverage_record_count": 3,
        "record_count": 3,
        "record_start": 0,
        "record_stop": 3,
        "records_sha256": receipt["cache"]["records_sha256"],
    }


def test_republisher_can_publish_only_identities_missing_from_donor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.args.missing_from_donor_only = True
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)

    receipt = tool._run(fixture.args)

    assert receipt["reused_record_count"] == 0
    assert receipt["encoded_record_count"] == 1
    assert fixture.builder.calls == [(fixture.identities[1][1],)]
    assert receipt["publication"]["missing_from_donor_only"] is True
    assert receipt["publication"]["full_coverage_record_count"] == 3
    assert receipt["publication"]["record_count"] == 1
    cache = FrozenDenseEvidenceCache.load(
        fixture.args.output_root,
        manifest_sha256=receipt["cache"]["manifest_sha256"],
        dataset_tree_sha256=fixture.target_manifest.tree_sha256,
    )
    assert tuple(
        (record.source_global_index, record.sample_key) for record in cache.records
    ) == (fixture.identities[1],)
    assert float(
        cache.evidence_for(
            source_global_index=fixture.identities[1][0],
            sample_key=fixture.identities[1][1],
        ).tokens[0, 0]
    ) == 99.0


def test_republisher_accepts_authenticated_partition_index_donor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    namespace = tmp_path / "partitioned-donor"
    partition_roots: list[Path] = []
    partition_hashes: list[str] = []
    for index, record in enumerate(fixture.donor_records):
        root = namespace / f"partition-{index:03d}"
        partition_roots.append(root)
        partition_hashes.append(
            publish_dense_evidence_cache(
                root,
                contract=fixture.donor_contract,
                records=(record,),
                shard_rows=1,
            )
        )
    index_root = namespace / "index"
    index_hash = merge_dense_evidence_cache_partitions(
        index_root,
        partition_roots=partition_roots,
        manifest_sha256s=partition_hashes,
        dataset_tree_sha256=fixture.donor_manifest.tree_sha256,
        coverage_plan_sha256=fixture.donor_coverage.artifact_sha256,
        expected_records=tuple(
            (record.source_global_index, record.sample_key)
            for record in fixture.donor_records
        ),
        reference_partitions=True,
    )
    fixture.args.donor_cache_root = index_root
    fixture.args.donor_cache_manifest_sha256 = index_hash
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)

    receipt = tool._run(fixture.args)

    assert receipt["reused_record_count"] == 2
    assert receipt["encoded_record_count"] == 1
    assert fixture.builder.calls == [(fixture.identities[1][1],)]
    assert receipt["donor"]["cache"]["root"] == str(index_root.resolve())
    assert json.loads((index_root / "manifest.json").read_text(encoding="ascii"))["schema"] == (
        "picf-next.frozen-dense-evidence-cache-partition-index/v1"
    )


def test_republisher_publishes_one_authenticated_contiguous_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.args.record_start = 1
    fixture.args.record_stop = 3
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)

    receipt = tool._run(fixture.args)

    assert receipt["record_count"] == 2
    assert receipt["reused_record_count"] == 1
    assert receipt["encoded_record_count"] == 1
    assert receipt["publication"]["record_start"] == 1
    assert receipt["publication"]["record_stop"] == 3
    cache = FrozenDenseEvidenceCache.load(
        fixture.args.output_root,
        manifest_sha256=receipt["cache"]["manifest_sha256"],
        dataset_tree_sha256=fixture.target_manifest.tree_sha256,
    )
    assert (
        tuple((record.source_global_index, record.sample_key) for record in cache.records)
        == fixture.identities[1:3]
    )
    assert fixture.builder.calls == [(fixture.identities[1][1],)]


@pytest.mark.parametrize(
    ("start", "stop"),
    ((-1, 1), (0, 0), (2, 1), (0, 4), (True, 2), (0, False)),
)
def test_republisher_rejects_invalid_partition_bounds(start: object, stop: object) -> None:
    args = argparse.Namespace(record_start=start, record_stop=stop)
    with pytest.raises(ContractError, match="publication bounds"):
        tool._publication_bounds(args, 3)


def test_republisher_recovers_receipt_after_complete_cache_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)
    first = tool._run(fixture.args)
    fixture.args.receipt_output.unlink()
    fixture.builder.calls.clear()

    recovered = tool._run(fixture.args)

    assert recovered["recovered_complete_cache"] is True
    assert recovered["cache"]["manifest_sha256"] == first["cache"]["manifest_sha256"]
    assert fixture.builder.calls == []
    assert json.loads(fixture.args.receipt_output.read_text(encoding="ascii")) == recovered


def test_republisher_rejects_recovering_complete_cache_as_another_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)
    tool._run(fixture.args)
    fixture.args.receipt_output.unlink()
    fixture.args.record_start = 1

    with pytest.raises(ContractError, match="coverage slice"):
        tool._run(fixture.args)


def test_republisher_rejects_resuming_partial_cache_as_another_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    class FailingBuilder(_Builder):
        def records_for_sample_keys(
            self,
            sample_keys: tuple[str, ...],
        ) -> tuple[DenseEvidenceCacheRecord, ...]:
            raise RuntimeError("simulated encoder interruption")

    failing = FailingBuilder(
        fixture.builder.cache_contract,
        fixture.builder.source_index_by_key,
    )
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: failing)
    with pytest.raises(RuntimeError, match="encoder interruption"):
        tool._run(fixture.args)
    fixture.args.record_stop = 2
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)

    with pytest.raises(ContractError, match="partial publication contract changed"):
        tool._run(fixture.args)


def test_republisher_rejects_target_builder_technical_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.builder.cache_contract = replace(
        fixture.builder.cache_contract,
        encoder_contract="vjepa.changed/frozen/v1",
    )
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)

    with pytest.raises(ContractError, match="technical encoder contract"):
        tool._run(fixture.args)

    assert not fixture.args.output_root.exists()
    assert fixture.builder.calls == []


def test_donor_coverage_must_match_cache_identity_and_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    changed_records = (
        fixture.donor_coverage.records[0],
        DenseEvidenceCoverageRecord(*fixture.identities[1], "evaluation"),
    )
    changed_plan = _coverage(fixture.donor_manifest, changed_records, name="changed")
    changed_contract = replace(
        fixture.donor_contract,
        coverage_plan_sha256=changed_plan.artifact_sha256,
    )
    changed_root = tmp_path / "changed-cache"
    changed_sha256 = publish_dense_evidence_cache(
        changed_root,
        contract=changed_contract,
        records=fixture.donor_records,
        shard_rows=1,
    )
    changed_cache = FrozenDenseEvidenceCache.load(
        changed_root,
        manifest_sha256=changed_sha256,
        dataset_tree_sha256=fixture.donor_manifest.tree_sha256,
    )

    with pytest.raises(ContractError, match="identity/order"):
        tool._validate_cache_and_plan(
            changed_cache,
            changed_plan,
            fixture.donor_manifest,
        )


def test_republisher_authenticates_donor_shards_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)
    shard = fixture.args.donor_cache_root / "shard-000001.npz"
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(ContractError, match="content hash mismatch"):
        tool._run(fixture.args)

    assert not fixture.args.output_root.exists()
    assert not fixture.args.receipt_output.exists()


def test_republisher_reauthenticates_donor_prefix_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    class FailingBuilder(_Builder):
        def records_for_sample_keys(
            self,
            sample_keys: tuple[str, ...],
        ) -> tuple[DenseEvidenceCacheRecord, ...]:
            raise RuntimeError("simulated encoder interruption")

    failing = FailingBuilder(
        fixture.builder.cache_contract,
        fixture.builder.source_index_by_key,
    )
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: failing)
    with pytest.raises(RuntimeError, match="encoder interruption"):
        tool._run(fixture.args)

    partial = fixture.args.output_root.with_name(f".{fixture.args.output_root.name}.partial")
    assert partial.is_dir()
    shard = fixture.args.donor_cache_root / "shard-000000.npz"
    shard.write_bytes(shard.read_bytes() + b"tampered-after-interruption")
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)

    with pytest.raises(ContractError, match="content hash mismatch"):
        tool._run(fixture.args)

    assert fixture.builder.calls == []
    assert not fixture.args.output_root.exists()
    assert not fixture.args.receipt_output.exists()


def test_republisher_rejects_symlinked_donor_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(tool, "_builder", lambda *args, **kwargs: fixture.builder)
    linked_root = tmp_path / "linked-donor"
    linked_root.symlink_to(fixture.args.donor_cache_root, target_is_directory=True)
    fixture.args.donor_cache_root = linked_root

    with pytest.raises(ContractError, match="symlink"):
        tool._run(fixture.args)

    assert not fixture.args.output_root.exists()
