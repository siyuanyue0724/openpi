#!/usr/bin/env python3
"""Rebind complete CALVIN/LingBot normalization to a verified content identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from picf_next.artifact_io import (
    publish_prepared_directory_durable_exclusive,
    write_bytes_durable_exclusive,
)
from picf_next.contracts import ContractError
from picf_next.data.calvin_normalization import (
    content_identified_calvin_normalization_artifact,
    load_calvin_normalization_artifact,
    official_lingbot_calvin_norm_stats,
    validate_lingbot_calvin_norm_stats,
)
from picf_next.data.calvin_official_source import (
    validate_calvin_content_identity_migration,
    validate_calvin_official_source_receipt,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    file_sha256,
    load_dataset_file_manifest,
)

MIGRATION_RECEIPT_SCHEMA = "picf-next.calvin-normalization-identity-migration.v1"
OUTPUT_CALVIN_NAME = "calvin-training-normalization.json"
OUTPUT_LINGBOT_NAME = "calvin-lingbot-norm-stats.json"
OUTPUT_RECEIPT_NAME = "migration-receipt.json"
_PROVENANCE_FIELDS = frozenset(
    {"artifact_sha256", "dataset_id", "dataset_revision", "dataset_tree_sha256"}
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "ascii"
        )
        + b"\n"
    )


def _read_stable_mapping(path: Path, *, label: str) -> tuple[dict[str, object], str]:
    digest = file_sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid ASCII JSON") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a mapping")
    if file_sha256(path) != digest:
        raise ContractError(f"{label} changed while loading")
    return payload, digest


def _load_stable_manifest(path: Path) -> tuple[DatasetFileManifest, str]:
    digest = file_sha256(path)
    manifest = load_dataset_file_manifest(path)
    if file_sha256(path) != digest:
        raise ContractError("CALVIN dataset manifest changed while loading")
    return manifest, digest


def _statistics_sha256(payload: dict[str, object]) -> str:
    statistics = {key: value for key, value in payload.items() if key not in _PROVENANCE_FIELDS}
    return hashlib.sha256(_canonical_bytes(statistics)).hexdigest()


def _publish(
    *,
    output_dir: Path,
    calvin: dict[str, object],
    lingbot: dict[str, object],
    receipt: dict[str, object],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    partial.mkdir(exist_ok=False)
    try:
        calvin_bytes = _json_bytes(calvin)
        lingbot_bytes = _json_bytes(lingbot)
        write_bytes_durable_exclusive(partial / OUTPUT_CALVIN_NAME, calvin_bytes)
        write_bytes_durable_exclusive(partial / OUTPUT_LINGBOT_NAME, lingbot_bytes)
        receipt["target_calvin_normalization"] = {
            "artifact_sha256": calvin["artifact_sha256"],
            "file_name": OUTPUT_CALVIN_NAME,
            "file_sha256": hashlib.sha256(calvin_bytes).hexdigest(),
        }
        receipt["target_lingbot_normalization"] = {
            "artifact_sha256": lingbot["artifact_sha256"],
            "file_name": OUTPUT_LINGBOT_NAME,
            "file_sha256": hashlib.sha256(lingbot_bytes).hexdigest(),
        }
        write_bytes_durable_exclusive(partial / OUTPUT_RECEIPT_NAME, _json_bytes(receipt))
        publish_prepared_directory_durable_exclusive(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _run(args: argparse.Namespace) -> dict[str, object]:
    source_dataset_path = args.source_dataset_manifest.resolve()
    target_dataset_path = args.target_dataset_manifest.resolve()
    source_dataset, source_dataset_sha256 = _load_stable_manifest(source_dataset_path)
    target_dataset, target_dataset_sha256 = _load_stable_manifest(target_dataset_path)
    validate_calvin_content_identity_migration(source_dataset, target_dataset)

    source_receipt_path = args.source_receipt.resolve()
    source_receipt, source_receipt_sha256 = _read_stable_mapping(
        source_receipt_path,
        label="CALVIN source receipt",
    )
    validate_calvin_official_source_receipt(
        source_receipt,
        source_manifest=source_dataset,
        source_manifest_sha256=source_dataset_sha256,
        target_manifest=target_dataset,
        target_manifest_sha256=target_dataset_sha256,
    )

    source_normalization_path = args.source_normalization.resolve()
    source_normalization_sha256 = file_sha256(source_normalization_path)
    if source_normalization_sha256 != args.expected_source_normalization_sha256:
        raise ContractError("CALVIN source normalization digest differs from the pinned input")
    source_normalization = load_calvin_normalization_artifact(source_normalization_path)
    if file_sha256(source_normalization_path) != source_normalization_sha256:
        raise ContractError("CALVIN source normalization changed while loading")
    if (
        source_normalization["dataset_id"] != source_dataset.dataset_id
        or source_normalization["dataset_revision"] != source_dataset.dataset_revision
        or source_normalization["dataset_tree_sha256"] != source_dataset.tree_sha256
    ):
        raise ContractError("CALVIN source normalization identity differs from its manifest")

    migrated = content_identified_calvin_normalization_artifact(
        source_normalization,
        dataset_id=target_dataset.dataset_id,
        dataset_revision=target_dataset.dataset_revision,
        dataset_tree_sha256=target_dataset.tree_sha256,
    )
    source_statistics_sha256 = _statistics_sha256(source_normalization)
    target_statistics_sha256 = _statistics_sha256(migrated)
    if source_statistics_sha256 != target_statistics_sha256:
        raise ContractError("CALVIN normalization migration changed statistics")
    lingbot = official_lingbot_calvin_norm_stats(
        migrated,
        dataset_tree_sha256=target_dataset.tree_sha256,
    )
    validate_lingbot_calvin_norm_stats(lingbot)

    receipt: dict[str, object] = {
        "schema": MIGRATION_RECEIPT_SCHEMA,
        "source_receipt_sha256": source_receipt_sha256,
        "source_dataset_manifest_sha256": source_dataset_sha256,
        "target_dataset_manifest_sha256": target_dataset_sha256,
        "source_normalization_file_sha256": source_normalization_sha256,
        "source_normalization_artifact_sha256": source_normalization["artifact_sha256"],
        "sample_count": source_normalization["sample_count"],
        "unique_source_frame_count": source_normalization["unique_source_frame_count"],
        "ordered_sample_keys_sha256": source_normalization["ordered_sample_keys_sha256"],
        "source_values_sha256": source_normalization["source_values_sha256"],
        "statistics_sha256": source_statistics_sha256,
        "statistics_unchanged": True,
        "migration_semantics": "identity-only;complete-statistics-not-recomputed",
        "target_dataset_id": target_dataset.dataset_id,
        "target_dataset_revision": target_dataset.dataset_revision,
        "target_dataset_tree_sha256": target_dataset.tree_sha256,
        "training_authorized": False,
        "training_authorization_reason": (
            "normalization migration does not replace semantic or action acceptance gates"
        ),
    }
    stable_inputs = (
        (source_dataset_path, source_dataset_sha256, "source dataset manifest"),
        (target_dataset_path, target_dataset_sha256, "target dataset manifest"),
        (source_receipt_path, source_receipt_sha256, "official source receipt"),
        (source_normalization_path, source_normalization_sha256, "source normalization"),
    )
    for path, expected_sha256, label in stable_inputs:
        if file_sha256(path) != expected_sha256:
            raise ContractError(f"CALVIN {label} changed during normalization migration")
    _publish(
        output_dir=args.output_dir.resolve(),
        calvin=migrated,
        lingbot=lingbot,
        receipt=receipt,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-manifest", type=Path, required=True)
    parser.add_argument("--target-dataset-manifest", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-normalization", type=Path, required=True)
    parser.add_argument("--expected-source-normalization-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    receipt = _run(parser.parse_args())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
