#!/usr/bin/env python3
"""Rebind a verified CALVIN tactile calibration to the official content identity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from picf_next.artifact_io import (
    publish_prepared_directory_durable_exclusive,
    write_bytes_durable_exclusive,
)
from picf_next.contracts import ContractError
from picf_next.data.calvin_official_source import (
    validate_calvin_content_identity_migration,
    validate_calvin_official_source_receipt,
)
from picf_next.data.calvin_tactile import CALVIN_TACTILE_STREAM_NAMES
from picf_next.data.calvin_tactile_calibration import (
    canonical_calibration_receipt_sha256,
    load_calvin_tactile_backgrounds,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    file_sha256,
    load_dataset_file_manifest,
    read_sha256_verified_file_beneath,
)

MIGRATION_RECEIPT_SCHEMA = "picf-next.calvin-tactile-calibration-identity-migration.v1"
OUTPUT_ARCHIVE_NAME = "tactile_backgrounds.npz"
OUTPUT_CALIBRATION_RECEIPT_NAME = "tactile_backgrounds.receipt.json"
OUTPUT_RECEIPT_NAME = "migration-receipt.json"
_MAXIMUM_ARCHIVE_BYTES = 16 * 1024 * 1024
_CALIBRATION_SEMANTIC_FIELDS = frozenset(
    {"archive", "calibration", "official_calvin_source", "sampling", "schema"}
)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _exact_mapping(value: object, *, fields: set[str], label: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise ContractError(f"{label} fields differ from the frozen schema")
    return value


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "ascii"
        )
        + b"\n"
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_stable_mapping(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, object], str]:
    expected = _require_sha256(expected_sha256, f"expected {label}")
    digest = file_sha256(path)
    if digest != expected:
        raise ContractError(f"{label} digest differs from the pinned input")
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid ASCII JSON") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a mapping")
    if file_sha256(path) != digest:
        raise ContractError(f"{label} changed while loading")
    return payload, digest


def _load_stable_manifest(
    path: Path,
    *,
    label: str,
) -> tuple[DatasetFileManifest, str]:
    digest = file_sha256(path)
    manifest = load_dataset_file_manifest(path)
    if file_sha256(path) != digest:
        raise ContractError(f"CALVIN {label} changed while loading")
    return manifest, digest


def _read_stable_archive(path: Path, *, expected_sha256: str) -> tuple[bytes, str]:
    expected = _require_sha256(expected_sha256, "expected source tactile archive")
    digest = file_sha256(path)
    if digest != expected:
        raise ContractError("source tactile archive digest differs from the pinned input")
    payload = read_sha256_verified_file_beneath(
        path.parent,
        path.name,
        expected_sha256=digest,
        maximum_bytes=_MAXIMUM_ARCHIVE_BYTES,
    )
    if file_sha256(path) != digest:
        raise ContractError("source tactile archive changed while loading")
    return payload, digest


def _validate_source_calibration_receipt(
    receipt: Mapping[str, object],
    *,
    source_manifest: DatasetFileManifest,
    source_manifest_sha256: str,
    source_archive_path: Path,
    source_archive_sha256: str,
) -> None:
    dataset = _exact_mapping(
        receipt.get("dataset"),
        fields={
            "dataset_id",
            "dataset_revision",
            "file_count",
            "manifest_sha256",
            "split_name",
            "tree_sha256",
        },
        label="source tactile calibration dataset",
    )
    expected_dataset = {
        "dataset_id": source_manifest.dataset_id,
        "dataset_revision": source_manifest.dataset_revision,
        "file_count": len(source_manifest.files),
        "manifest_sha256": source_manifest_sha256,
        "split_name": source_manifest.split_name,
        "tree_sha256": source_manifest.tree_sha256,
    }
    if dict(dataset) != expected_dataset:
        raise ContractError("source tactile calibration identity differs from its manifest")

    archive = _exact_mapping(
        receipt.get("archive"),
        fields={"path", "sha256"},
        label="source tactile calibration archive",
    )
    if (
        archive.get("path") != str(source_archive_path)
        or archive.get("sha256") != source_archive_sha256
    ):
        raise ContractError("source tactile calibration archive binding differs")

    sampling = _exact_mapping(
        receipt.get("sampling"),
        fields={
            "sample_count",
            "sampled_steps_sha256",
            "tactile_audit_sha256",
            "visual_review_manifest_sha256",
        },
        label="source tactile calibration sampling",
    )
    sample_count = _require_positive_int(
        sampling.get("sample_count"),
        "source tactile calibration sample count",
    )
    for field in (
        "sampled_steps_sha256",
        "tactile_audit_sha256",
        "visual_review_manifest_sha256",
    ):
        _require_sha256(sampling.get(field), f"source tactile calibration {field}")

    calibration = _exact_mapping(
        receipt.get("calibration"),
        fields={"algorithm", "background_noise_ceiling_m", "streams"},
        label="source tactile calibration",
    )
    streams = _exact_mapping(
        calibration.get("streams"),
        fields=set(CALVIN_TACTILE_STREAM_NAMES),
        label="source tactile calibration streams",
    )
    for name in CALVIN_TACTILE_STREAM_NAMES:
        stream = _exact_mapping(
            streams.get(name),
            fields={
                "background_sha256",
                "candidate_count",
                "candidate_steps_sha256",
                "selected_count",
                "selected_source_sha256",
                "selected_steps",
                "validity_threshold_m",
            },
            label=f"source {name} tactile calibration",
        )
        candidate_count = _require_positive_int(
            stream.get("candidate_count"),
            f"source {name} tactile candidate count",
        )
        selected_count = _require_positive_int(
            stream.get("selected_count"),
            f"source {name} tactile selected count",
        )
        if selected_count > candidate_count or candidate_count > sample_count:
            raise ContractError(f"source {name} tactile calibration counts are inconsistent")
        for field in (
            "background_sha256",
            "candidate_steps_sha256",
            "selected_source_sha256",
        ):
            _require_sha256(stream.get(field), f"source {name} tactile {field}")


def _calibration_semantics(receipt: Mapping[str, object]) -> dict[str, object]:
    semantics = {key: copy.deepcopy(receipt[key]) for key in _CALIBRATION_SEMANTIC_FIELDS}
    archive = semantics.get("archive")
    if not isinstance(archive, dict) or set(archive) != {"path", "sha256"}:
        raise ContractError("tactile calibration archive fields differ from the frozen schema")
    del archive["path"]
    return semantics


def _target_calibration_receipt(
    source_receipt: Mapping[str, object],
    *,
    target_manifest: DatasetFileManifest,
    target_manifest_sha256: str,
    target_archive_path: Path,
) -> dict[str, object]:
    migrated = copy.deepcopy(dict(source_receipt))
    migrated["dataset"] = {
        "dataset_id": target_manifest.dataset_id,
        "dataset_revision": target_manifest.dataset_revision,
        "file_count": len(target_manifest.files),
        "manifest_sha256": target_manifest_sha256,
        "split_name": target_manifest.split_name,
        "tree_sha256": target_manifest.tree_sha256,
    }
    archive = migrated.get("archive")
    if not isinstance(archive, dict):
        raise ContractError("source tactile calibration archive must be a mapping")
    archive["path"] = str(target_archive_path)
    migrated.pop("receipt_payload_sha256", None)
    migrated["receipt_payload_sha256"] = canonical_calibration_receipt_sha256(migrated)
    if _calibration_semantics(source_receipt) != _calibration_semantics(migrated):
        raise ContractError("tactile calibration migration changed semantic fields")
    return migrated


def _publish(
    *,
    output_dir: Path,
    archive_bytes: bytes,
    archive_sha256: str,
    calibration_receipt: dict[str, object],
    migration_receipt: dict[str, object],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    partial.mkdir(exist_ok=False)
    try:
        if hashlib.sha256(archive_bytes).hexdigest() != archive_sha256:
            raise ContractError("retained tactile archive bytes changed before publication")
        write_bytes_durable_exclusive(partial / OUTPUT_ARCHIVE_NAME, archive_bytes)
        calibration_receipt_bytes = _json_bytes(calibration_receipt)
        write_bytes_durable_exclusive(
            partial / OUTPUT_CALIBRATION_RECEIPT_NAME,
            calibration_receipt_bytes,
        )
        migration_receipt["target_tactile_calibration"] = {
            "archive_file_name": OUTPUT_ARCHIVE_NAME,
            "archive_sha256": archive_sha256,
            "receipt_file_name": OUTPUT_CALIBRATION_RECEIPT_NAME,
            "receipt_file_sha256": hashlib.sha256(calibration_receipt_bytes).hexdigest(),
            "receipt_payload_sha256": calibration_receipt["receipt_payload_sha256"],
        }
        write_bytes_durable_exclusive(
            partial / OUTPUT_RECEIPT_NAME,
            _json_bytes(migration_receipt),
        )
        publish_prepared_directory_durable_exclusive(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _run(args: argparse.Namespace) -> dict[str, object]:
    source_manifest_path = args.source_dataset_manifest.resolve()
    official_source_manifest_path = args.official_source_dataset_manifest.resolve()
    target_manifest_path = args.target_dataset_manifest.resolve()
    source_manifest, source_manifest_sha256 = _load_stable_manifest(
        source_manifest_path,
        label="source dataset manifest",
    )
    official_source_manifest, official_source_manifest_sha256 = _load_stable_manifest(
        official_source_manifest_path,
        label="official-source dataset manifest",
    )
    target_manifest, target_manifest_sha256 = _load_stable_manifest(
        target_manifest_path,
        label="target dataset manifest",
    )
    validate_calvin_content_identity_migration(source_manifest, target_manifest)
    validate_calvin_content_identity_migration(official_source_manifest, target_manifest)

    official_receipt_path = args.official_source_receipt.resolve()
    official_receipt, official_receipt_sha256 = _read_stable_mapping(
        official_receipt_path,
        expected_sha256=args.expected_official_source_receipt_sha256,
        label="official source receipt",
    )
    validate_calvin_official_source_receipt(
        official_receipt,
        source_manifest=official_source_manifest,
        source_manifest_sha256=official_source_manifest_sha256,
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
    )

    source_archive_path = args.source_tactile_archive.resolve()
    source_archive_bytes, source_archive_sha256 = _read_stable_archive(
        source_archive_path,
        expected_sha256=args.expected_source_tactile_archive_sha256,
    )
    source_tactile_receipt_path = args.source_tactile_receipt.resolve()
    source_tactile_receipt, source_tactile_receipt_sha256 = _read_stable_mapping(
        source_tactile_receipt_path,
        expected_sha256=args.expected_source_tactile_receipt_sha256,
        label="source tactile calibration receipt",
    )
    loaded = load_calvin_tactile_backgrounds(
        source_archive_path,
        source_tactile_receipt_path,
        receipt_sha256=source_tactile_receipt_sha256,
        dataset_tree_sha256=source_manifest.tree_sha256,
    )
    if loaded.archive_sha256 != source_archive_sha256:
        raise ContractError("source tactile archive digest differs from its receipt")
    _validate_source_calibration_receipt(
        source_tactile_receipt,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        source_archive_path=source_archive_path,
        source_archive_sha256=source_archive_sha256,
    )

    source_semantics = _calibration_semantics(source_tactile_receipt)
    source_semantics_sha256 = _canonical_sha256(source_semantics)
    output_dir = _absolute_path(args.output_dir)
    target_calibration_receipt = _target_calibration_receipt(
        source_tactile_receipt,
        target_manifest=target_manifest,
        target_manifest_sha256=target_manifest_sha256,
        target_archive_path=output_dir / OUTPUT_ARCHIVE_NAME,
    )
    if _canonical_sha256(_calibration_semantics(target_calibration_receipt)) != (
        source_semantics_sha256
    ):
        raise ContractError("tactile calibration migration changed semantic hashes")

    stable_inputs = (
        (source_manifest_path, source_manifest_sha256, "source dataset manifest"),
        (
            official_source_manifest_path,
            official_source_manifest_sha256,
            "official-source dataset manifest",
        ),
        (target_manifest_path, target_manifest_sha256, "target dataset manifest"),
        (official_receipt_path, official_receipt_sha256, "official source receipt"),
        (source_archive_path, source_archive_sha256, "source tactile archive"),
        (
            source_tactile_receipt_path,
            source_tactile_receipt_sha256,
            "source tactile calibration receipt",
        ),
    )
    for path, expected_sha256, label in stable_inputs:
        if file_sha256(path) != expected_sha256:
            raise ContractError(f"CALVIN {label} changed during tactile calibration migration")

    migration_receipt: dict[str, object] = {
        "schema": MIGRATION_RECEIPT_SCHEMA,
        "source_dataset_manifest_sha256": source_manifest_sha256,
        "official_source_dataset_manifest_sha256": official_source_manifest_sha256,
        "target_dataset_manifest_sha256": target_manifest_sha256,
        "official_source_receipt_sha256": official_receipt_sha256,
        "source_tactile_archive_sha256": source_archive_sha256,
        "source_tactile_receipt_sha256": source_tactile_receipt_sha256,
        "source_tactile_receipt_payload_sha256": loaded.receipt_payload_sha256,
        "calibration_semantics_sha256": source_semantics_sha256,
        "calibration_semantics_unchanged": True,
        "archive_bytes_unchanged": True,
        "migration_semantics": "identity-only;calibration-not-recomputed",
        "target_dataset_id": target_manifest.dataset_id,
        "target_dataset_revision": target_manifest.dataset_revision,
        "target_dataset_tree_sha256": target_manifest.tree_sha256,
        "training_authorized": False,
        "training_authorization_reason": (
            "tactile calibration identity migration does not authorize model training"
        ),
    }
    _publish(
        output_dir=output_dir,
        archive_bytes=source_archive_bytes,
        archive_sha256=source_archive_sha256,
        calibration_receipt=target_calibration_receipt,
        migration_receipt=migration_receipt,
    )
    return migration_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-manifest", type=Path, required=True)
    parser.add_argument("--official-source-dataset-manifest", type=Path, required=True)
    parser.add_argument("--target-dataset-manifest", type=Path, required=True)
    parser.add_argument("--official-source-receipt", type=Path, required=True)
    parser.add_argument("--expected-official-source-receipt-sha256", required=True)
    parser.add_argument("--source-tactile-archive", type=Path, required=True)
    parser.add_argument("--expected-source-tactile-archive-sha256", required=True)
    parser.add_argument("--source-tactile-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-tactile-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    receipt = _run(parser.parse_args())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
