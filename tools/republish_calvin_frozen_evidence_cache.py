#!/usr/bin/env python3
"""Republish a CALVIN frozen-evidence cache from one strict authenticated donor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.calvin_official_source import (
    validate_calvin_content_identity_migration,
    validate_calvin_official_source_receipt,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    file_sha256,
    load_dataset_file_manifest,
)
from picf_next.data.dense_evidence_cache import (
    DenseEvidenceCacheContract,
    DenseEvidenceCacheRecord,
    FrozenDenseEvidenceCache,
    _load_dense_evidence_cache,
    publish_dense_evidence_cache_resumable,
)
from picf_next.data.dense_evidence_coverage import (
    DenseEvidenceCoveragePlan,
    DenseEvidenceCoverageRecord,
)
from tools.build_calvin_frozen_evidence_cache import _builder

CALVIN_FROZEN_EVIDENCE_REPUBLICATION_SCHEMA = (
    "picf-next.calvin-frozen-evidence-donor-republication/v1"
)

_TECHNICAL_CONTRACT_FIELDS = (
    "encoder_contract",
    "geometry_width",
    "has_group_ids",
    "maximum_tokens",
    "modality",
    "token_dtype",
    "token_width",
)


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _canonical_compact(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _publication_bounds(args: argparse.Namespace, record_count: int) -> tuple[int, int]:
    """Return one explicit contiguous slice of the authenticated full coverage."""

    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
        raise ContractError("full coverage record count must be a positive integer")
    start = getattr(args, "record_start", 0)
    raw_stop = getattr(args, "record_stop", None)
    stop = record_count if raw_stop is None else raw_stop
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= record_count
    ):
        raise ContractError(
            "publication bounds must satisfy 0 <= record_start < record_stop <= coverage"
        )
    return start, stop


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise ContractError(f"{label} must not contain symlink components")


def _regular_file(path: str | Path, *, label: str) -> Path:
    result = _absolute(path)
    _reject_symlink_components(result, label=label)
    if not result.is_file():
        raise ContractError(f"{label} must be one existing regular file")
    return result


def _directory(path: str | Path, *, label: str) -> Path:
    result = _absolute(path)
    _reject_symlink_components(result, label=label)
    if not result.is_dir():
        raise ContractError(f"{label} must be one existing directory")
    return result


def _output_path(path: str | Path, *, label: str) -> Path:
    result = _absolute(path)
    _reject_symlink_components(result, label=label)
    return result


def _stable_file_sha256(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    source = _regular_file(path, label=label)
    observed = file_sha256(source)
    if expected_sha256 is not None and observed != _sha256(expected_sha256, label):
        raise ContractError(f"{label} SHA-256 differs from the pinned input")
    return source, observed


def _load_stable_manifest(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, DatasetFileManifest, str]:
    source, digest = _stable_file_sha256(path, label=label)
    manifest = load_dataset_file_manifest(source)
    if file_sha256(source) != digest:
        raise ContractError(f"{label} changed while loading")
    return source, manifest, digest


def _load_stable_mapping(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, object], str]:
    source, digest = _stable_file_sha256(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )
    try:
        payload = json.loads(source.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid ASCII JSON") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a mapping")
    if file_sha256(source) != digest:
        raise ContractError(f"{label} changed while loading")
    return source, payload, digest


def _load_stable_coverage_plan(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str,
) -> tuple[Path, DenseEvidenceCoveragePlan, str]:
    source, digest = _stable_file_sha256(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )
    plan = DenseEvidenceCoveragePlan.load(source)
    if file_sha256(source) != digest:
        raise ContractError(f"{label} changed while loading")
    return source, plan, digest


def _plan_identity(
    plan: DenseEvidenceCoveragePlan,
) -> tuple[str, str, str]:
    return plan.dataset_id, plan.dataset_revision, plan.dataset_tree_sha256


def _manifest_identity(manifest: DatasetFileManifest) -> tuple[str, str, str]:
    return manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256


def _validate_plan_manifest_identity(
    plan: DenseEvidenceCoveragePlan,
    manifest: DatasetFileManifest,
    *,
    label: str,
) -> None:
    if _plan_identity(plan) != _manifest_identity(manifest):
        raise ContractError(f"{label} coverage plan belongs to another dataset manifest")


def _technical_contract(contract: DenseEvidenceCacheContract) -> dict[str, object]:
    if not isinstance(contract, DenseEvidenceCacheContract):
        raise TypeError("technical cache comparison requires a typed contract")
    payload = contract.payload()
    return {field: payload[field] for field in _TECHNICAL_CONTRACT_FIELDS}


def _validate_cache_and_plan(
    cache: FrozenDenseEvidenceCache,
    plan: DenseEvidenceCoveragePlan,
    manifest: DatasetFileManifest,
) -> None:
    if (
        cache.contract.dataset_id,
        cache.contract.dataset_revision,
        cache.contract.dataset_tree_sha256,
    ) != _manifest_identity(manifest):
        raise ContractError("donor cache belongs to another donor dataset manifest")
    if cache.contract.coverage_plan_sha256 != plan.artifact_sha256:
        raise ContractError("donor cache coverage artifact differs from its pinned plan")
    cache_identities = tuple(
        (record.source_global_index, record.sample_key) for record in cache.records
    )
    if cache_identities != plan.record_identities:
        raise ContractError("donor coverage plan identity/order differs from donor cache")


def _validate_plan_records(
    plan: DenseEvidenceCoveragePlan,
    dataset: CalvinPhysicalTransitionDataset,
    *,
    label: str,
) -> None:
    for record in plan.records:
        try:
            observed = dataset.source_global_index_by_key(record.sample_key)
        except KeyError as error:
            raise ContractError(
                f"{label} coverage contains an unknown CALVIN sample key"
            ) from error
        if observed != record.source_global_index:
            raise ContractError(f"{label} coverage sample identity differs from CALVIN")


def _authenticated_donor_record(
    cache: FrozenDenseEvidenceCache,
    location: Any,
) -> DenseEvidenceCacheRecord:
    evidence = cache.evidence_for(
        source_global_index=location.source_global_index,
        sample_key=location.sample_key,
        source_input_sha256=location.source_input_sha256,
    )
    return DenseEvidenceCacheRecord(
        source_global_index=location.source_global_index,
        sample_key=location.sample_key,
        source_input_sha256=location.source_input_sha256,
        evidence=evidence,
    )


def _encoded_records(
    builder: Any,
    expected: Sequence[DenseEvidenceCoverageRecord],
) -> tuple[DenseEvidenceCacheRecord, ...]:
    sample_keys = tuple(record.sample_key for record in expected)
    batch_method = getattr(builder, "records_for_sample_keys", None)
    if callable(batch_method):
        encoded = tuple(batch_method(sample_keys))
    else:
        encoded = tuple(builder.record(sample_key) for sample_key in sample_keys)
    if any(not isinstance(record, DenseEvidenceCacheRecord) for record in encoded):
        raise TypeError("CALVIN modality builder returned an untyped cache record")
    expected_identities = tuple(
        (record.source_global_index, record.sample_key) for record in expected
    )
    observed_identities = tuple(
        (record.source_global_index, record.sample_key) for record in encoded
    )
    if observed_identities != expected_identities:
        raise ContractError("CALVIN modality builder changed requested coverage identity/order")
    return encoded


def _republication_record_factory(
    *,
    target_records: tuple[DenseEvidenceCoverageRecord, ...],
    donor_cache: FrozenDenseEvidenceCache,
    builder: Any,
    encoder_batch_size: int,
) -> Callable[[int], Iterable[DenseEvidenceCacheRecord]]:
    batch_size = _positive_integer(encoder_batch_size, "encoder batch size")

    def records_from(completed: int) -> Iterable[DenseEvidenceCacheRecord]:
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 0 <= completed <= len(target_records)
        ):
            raise ContractError("republisher resume offset is outside target coverage")
        donor_iterator = iter(donor_cache.records)
        donor = next(donor_iterator, None)
        missing: list[DenseEvidenceCoverageRecord] = []

        for target in target_records[completed:]:
            while donor is not None and donor.source_global_index < target.source_global_index:
                _authenticated_donor_record(donor_cache, donor)
                donor = next(donor_iterator, None)

            donated = None
            if donor is not None and donor.source_global_index == target.source_global_index:
                authenticated = _authenticated_donor_record(donor_cache, donor)
                donor = next(donor_iterator, None)
                if authenticated.sample_key == target.sample_key:
                    donated = authenticated

            if donated is not None:
                if missing:
                    yield from _encoded_records(builder, missing)
                    missing.clear()
                yield donated
                continue

            missing.append(target)
            if len(missing) == batch_size:
                yield from _encoded_records(builder, missing)
                missing.clear()

        if missing:
            yield from _encoded_records(builder, missing)
        while donor is not None:
            _authenticated_donor_record(donor_cache, donor)
            donor = next(donor_iterator, None)

    return records_from


def _assert_stable_inputs(inputs: Sequence[tuple[Path, str, str]]) -> None:
    for path, expected_sha256, label in inputs:
        if file_sha256(path) != expected_sha256:
            raise ContractError(f"{label} changed during cache republication")


def _write_receipt_exclusive(path: Path, payload: bytes) -> str:
    _reject_symlink_components(path.parent, label="receipt output parent")
    write_bytes_durable_exclusive(path, payload)
    return _sha256_bytes(payload)


def _builder_input_hashes(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, str]], list[tuple[Path, str, str]]]:
    requested: list[tuple[str, object, str | None]] = [
        ("asset_manifest", args.asset_manifest, None),
    ]
    for name, value, expected in (
        ("camera_calibration", args.camera_calibration, None),
        ("tactile_calibration_archive", args.tactile_calibration_archive, None),
        (
            "tactile_calibration_receipt",
            args.tactile_calibration_receipt,
            args.tactile_calibration_receipt_sha256,
        ),
    ):
        if value is not None:
            requested.append((name, value, expected))

    hashes: dict[str, dict[str, str]] = {}
    stable: list[tuple[Path, str, str]] = []
    for name, value, expected in requested:
        label = name.replace("_", " ")
        path, digest = _stable_file_sha256(
            value,
            label=label,
            expected_sha256=expected,
        )
        hashes[name] = {"file_sha256": digest, "path": str(path)}
        stable.append((path, digest, label))
    return hashes, stable


def _cache_hash_summary(
    cache: FrozenDenseEvidenceCache,
    manifest_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "contract": cache.contract.payload(),
        "record_count": len(cache.records),
        "records_sha256": manifest_payload["records_sha256"],
        "shard_sha256s": [shard.sha256 for shard in cache.shards],
    }


def _coverage_hash_summary(
    plan: DenseEvidenceCoveragePlan,
    *,
    path: Path,
    file_sha256: str,
) -> dict[str, str]:
    return {
        "artifact_sha256": plan.artifact_sha256,
        "evaluation_plan_sha256": plan.evaluation_plan_sha256,
        "file_sha256": file_sha256,
        "path": str(path),
        "records_sha256": plan.records_sha256,
        "representation_split_sha256": plan.representation_split_sha256,
        "stream_plan_sha256": plan.stream_plan_sha256,
        "training_visits_sha256": plan.training_visits_sha256,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    _positive_integer(args.shard_rows, "shard rows")
    _positive_integer(args.encoder_batch_size, "encoder batch size")
    _positive_integer(args.action_horizon, "action horizon")
    _positive_integer(args.point_pixel_stride, "point pixel stride")
    _positive_integer(args.point_budget, "point budget")
    missing_from_donor_only = bool(getattr(args, "missing_from_donor_only", False))

    output_root = _output_path(args.output_root, label="cache output root")
    receipt_output = _output_path(
        args.receipt_output
        if args.receipt_output is not None
        else output_root.with_name(f"{output_root.name}.receipt.json"),
        label="receipt output",
    )
    if receipt_output.exists() or receipt_output.is_symlink():
        raise FileExistsError(receipt_output)

    target_path, target_manifest, target_manifest_sha256 = _load_stable_manifest(
        args.target_dataset_manifest,
        label="official target dataset manifest",
    )
    donor_path, donor_manifest, donor_manifest_sha256 = _load_stable_manifest(
        args.donor_dataset_manifest,
        label="donor dataset manifest",
    )
    source_path, source_manifest, source_manifest_sha256 = _load_stable_manifest(
        args.source_dataset_manifest,
        label="receipt source dataset manifest",
    )
    validate_calvin_content_identity_migration(donor_manifest, target_manifest)
    validate_calvin_content_identity_migration(source_manifest, target_manifest)

    source_receipt_path, source_receipt, source_receipt_sha256 = _load_stable_mapping(
        args.source_receipt,
        label="official source receipt",
        expected_sha256=args.source_receipt_sha256,
    )
    validate_calvin_official_source_receipt(
        source_receipt,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
    )

    coverage_path, coverage, coverage_file_sha256 = _load_stable_coverage_plan(
        args.coverage_plan,
        label="official target coverage plan",
        expected_sha256=args.coverage_plan_sha256,
    )
    donor_coverage_path, donor_coverage, donor_coverage_file_sha256 = _load_stable_coverage_plan(
        args.donor_coverage_plan,
        label="donor coverage plan",
        expected_sha256=args.donor_coverage_plan_sha256,
    )
    _validate_plan_manifest_identity(coverage, target_manifest, label="official target")
    _validate_plan_manifest_identity(donor_coverage, donor_manifest, label="donor")
    record_start, record_stop = _publication_bounds(args, len(coverage.records))
    target_records = coverage.records[record_start:record_stop]

    split_root = _directory(
        _absolute(args.dataset_root) / args.split,
        label="official CALVIN split root",
    )
    if target_manifest.split_name != args.split:
        raise ContractError("official target manifest split differs from the requested split")
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=target_manifest.dataset_id,
        dataset_revision=target_manifest.dataset_revision,
        verify_files=args.verify_all_dataset_files,
        dataset_manifest=target_manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=args.action_horizon)
    _validate_plan_records(coverage, dataset, label="official target")
    _validate_plan_records(donor_coverage, dataset, label="donor")

    donor_root = _directory(args.donor_cache_root, label="donor cache root")
    donor_cache_manifest_path, donor_cache_manifest, donor_cache_manifest_sha256 = (
        _load_stable_mapping(
            donor_root / "manifest.json",
            label="donor cache manifest",
            expected_sha256=args.donor_cache_manifest_sha256,
        )
    )
    donor_cache = _load_dense_evidence_cache(
        donor_root,
        manifest_sha256=donor_cache_manifest_sha256,
        dataset_tree_sha256=donor_manifest.tree_sha256,
        memory_capacity=2,
    )
    _validate_cache_and_plan(donor_cache, donor_coverage, donor_manifest)

    if missing_from_donor_only:
        if args.record_start != 0 or args.record_stop is not None:
            raise ContractError(
                "missing-from-donor publication cannot also select positional bounds"
            )
        donor_identities = set(donor_coverage.record_identities)
        target_records = tuple(
            record
            for record in coverage.records
            if (record.source_global_index, record.sample_key) not in donor_identities
        )
        if not target_records:
            raise ContractError("target coverage contains no records absent from donor")
        record_start = 0
        record_stop = len(coverage.records)

    builder_inputs, builder_stable_inputs = _builder_input_hashes(args)
    builder = _builder(
        args,
        dataset,
        coverage_plan_sha256=coverage.artifact_sha256,
    )
    target_contract = builder.cache_contract
    if not isinstance(target_contract, DenseEvidenceCacheContract):
        raise TypeError("CALVIN modality builder returned an untyped cache contract")
    if (
        target_contract.dataset_id,
        target_contract.dataset_revision,
        target_contract.dataset_tree_sha256,
        target_contract.coverage_plan_sha256,
    ) != (*_manifest_identity(target_manifest), coverage.artifact_sha256):
        raise ContractError("target builder cache provenance differs from official coverage")
    if _technical_contract(target_contract) != _technical_contract(donor_cache.contract):
        raise ContractError("target builder technical encoder contract differs from donor")

    donor_identities = set(donor_coverage.record_identities)
    reused_record_count = sum(
        (record.source_global_index, record.sample_key) in donor_identities
        for record in target_records
    )
    encoded_record_count = len(target_records) - reused_record_count
    if missing_from_donor_only:
        def record_factory(completed: int) -> Iterable[DenseEvidenceCacheRecord]:
            if (
                isinstance(completed, bool)
                or not isinstance(completed, int)
                or not 0 <= completed <= len(target_records)
            ):
                raise ContractError("republisher resume offset is outside target coverage")
            pending = target_records[completed:]
            for start in range(0, len(pending), args.encoder_batch_size):
                yield from _encoded_records(
                    builder,
                    pending[start : start + args.encoder_batch_size],
                )
    else:
        record_factory = _republication_record_factory(
            target_records=target_records,
            donor_cache=donor_cache,
            builder=builder,
            encoder_batch_size=args.encoder_batch_size,
        )

    started = time.perf_counter()
    recovered_complete_cache = output_root.is_dir() and not output_root.is_symlink()
    if recovered_complete_cache:
        published_manifest_path, published_manifest, observed_manifest_sha256 = (
            _load_stable_mapping(
                output_root / "manifest.json",
                label="published cache manifest",
            )
        )
    else:
        manifest_sha256 = publish_dense_evidence_cache_resumable(
            output_root,
            contract=target_contract,
            expected_record_count=len(target_records),
            record_factory=record_factory,
            shard_rows=args.shard_rows,
        )
        published_manifest_path, published_manifest, observed_manifest_sha256 = (
            _load_stable_mapping(
                output_root / "manifest.json",
                label="published cache manifest",
                expected_sha256=manifest_sha256,
            )
        )
    elapsed = time.perf_counter() - started

    published_cache = FrozenDenseEvidenceCache.load(
        output_root,
        manifest_sha256=observed_manifest_sha256,
        dataset_tree_sha256=target_manifest.tree_sha256,
    )
    if published_cache.contract != target_contract or tuple(
        (record.source_global_index, record.sample_key) for record in published_cache.records
    ) != tuple((record.source_global_index, record.sample_key) for record in target_records):
        raise ContractError("published cache differs from its official target coverage slice")

    stable_inputs = [
        (target_path, target_manifest_sha256, "official target dataset manifest"),
        (donor_path, donor_manifest_sha256, "donor dataset manifest"),
        (source_path, source_manifest_sha256, "receipt source dataset manifest"),
        (source_receipt_path, source_receipt_sha256, "official source receipt"),
        (coverage_path, coverage_file_sha256, "official target coverage plan"),
        (donor_coverage_path, donor_coverage_file_sha256, "donor coverage plan"),
        (donor_cache_manifest_path, donor_cache_manifest_sha256, "donor cache manifest"),
        *builder_stable_inputs,
    ]
    _assert_stable_inputs(stable_inputs)
    if file_sha256(published_manifest_path) != observed_manifest_sha256:
        raise ContractError("published cache manifest changed before receipt publication")

    technical_contract = _technical_contract(target_contract)
    receipt: dict[str, object] = {
        "builder_inputs": builder_inputs,
        "cache": {
            **_cache_hash_summary(published_cache, published_manifest),
            "manifest_sha256": observed_manifest_sha256,
            "output_root": str(output_root),
        },
        "donor": {
            "cache": {
                **_cache_hash_summary(donor_cache, donor_cache_manifest),
                "manifest_sha256": donor_cache_manifest_sha256,
                "root": str(donor_root),
            },
            "coverage_plan": _coverage_hash_summary(
                donor_coverage,
                path=donor_coverage_path,
                file_sha256=donor_coverage_file_sha256,
            ),
            "dataset_manifest": {
                "content_sha256": donor_manifest.content_sha256,
                "file_sha256": donor_manifest_sha256,
                "path": str(donor_path),
                "tree_sha256": donor_manifest.tree_sha256,
            },
        },
        "elapsed_seconds": elapsed,
        "encoded_record_count": encoded_record_count,
        "official_source": {
            "original_dataset_manifest": {
                "content_sha256": source_manifest.content_sha256,
                "file_sha256": source_manifest_sha256,
                "path": str(source_path),
                "tree_sha256": source_manifest.tree_sha256,
            },
            "receipt_file_sha256": source_receipt_sha256,
            "receipt_path": str(source_receipt_path),
        },
        "official_target": {
            "coverage_plan": _coverage_hash_summary(
                coverage,
                path=coverage_path,
                file_sha256=coverage_file_sha256,
            ),
            "dataset_manifest": {
                "content_sha256": target_manifest.content_sha256,
                "file_sha256": target_manifest_sha256,
                "path": str(target_path),
                "tree_sha256": target_manifest.tree_sha256,
            },
        },
        "publication": {
            "full_coverage_record_count": len(coverage.records),
            "record_count": len(target_records),
            "record_start": record_start,
            "record_stop": record_stop,
            "records_sha256": published_manifest["records_sha256"],
        },
        "record_count": len(target_records),
        "recovered_complete_cache": recovered_complete_cache,
        "records_per_second": len(target_records) / max(elapsed, 1e-12),
        "reuse_policy": {
            "directory_discovery": False,
            "match_fields": ["source_global_index", "sample_key"],
            "semantic_heads": False,
            "symlink_reuse": False,
        },
        "reused_record_count": reused_record_count,
        "schema": CALVIN_FROZEN_EVIDENCE_REPUBLICATION_SCHEMA,
        "technical_encoder_contract": technical_contract,
        "technical_encoder_contract_sha256": _sha256_bytes(_canonical_compact(technical_contract)),
        "training_authorized": False,
    }
    if missing_from_donor_only:
        receipt["publication"]["missing_from_donor_only"] = True
    receipt_payload = _canonical_json(receipt)
    _write_receipt_exclusive(receipt_output, receipt_payload)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", default="training", choices=("training", "validation"))
    parser.add_argument(
        "--target-dataset-manifest",
        "--dataset-manifest",
        dest="target_dataset_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--donor-dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("vjepa", "sonata", "anytouch"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument(
        "--coverage-plan",
        "--target-coverage-plan",
        dest="coverage_plan",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--coverage-plan-sha256",
        "--target-coverage-plan-sha256",
        dest="coverage_plan_sha256",
        required=True,
    )
    parser.add_argument("--donor-coverage-plan", required=True, type=Path)
    parser.add_argument("--donor-coverage-plan-sha256", required=True)
    parser.add_argument("--donor-cache-root", required=True, type=Path)
    parser.add_argument("--donor-cache-manifest-sha256", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--token-dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--shard-rows", default=64, type=int)
    parser.add_argument("--encoder-batch-size", default=1, type=int)
    parser.add_argument("--record-start", default=0, type=int)
    parser.add_argument("--record-stop", type=int)
    parser.add_argument(
        "--missing-from-donor-only",
        action="store_true",
        help="Publish exactly target identities absent from the authenticated donor plan.",
    )
    parser.add_argument("--action-horizon", default=1, type=int)
    parser.add_argument("--verify-all-dataset-files", action="store_true")
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--point-pixel-stride", default=2, type=int)
    parser.add_argument("--point-budget", default=4096, type=int)
    parser.add_argument("--tactile-calibration-archive", type=Path)
    parser.add_argument("--tactile-calibration-receipt", type=Path)
    parser.add_argument("--tactile-calibration-receipt-sha256")
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt_output = _output_path(
        args.receipt_output
        if args.receipt_output is not None
        else args.output_root.with_name(f"{args.output_root.name}.receipt.json"),
        label="receipt output",
    )
    receipt = _run(args)
    print(
        json.dumps(
            {
                **receipt,
                "receipt_output": str(receipt_output),
                "receipt_sha256": file_sha256(receipt_output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
