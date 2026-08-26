#!/usr/bin/env python3
"""Publish a verified CALVIN sidecar manifest under a new content identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
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
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA,
    CalvinPhysicalSupervisionShard,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    file_sha256,
    load_dataset_file_manifest,
    read_sha256_verified_file_beneath,
)

MIGRATION_RECEIPT_SCHEMA = "picf-next.calvin-sidecar-identity-migration.v1"
OUTPUT_MANIFEST_NAME = "physical-sidecar-manifest.json"
OUTPUT_RECEIPT_NAME = "migration-receipt.json"
_MAXIMUM_SHARD_BYTES = 512 * 1024 * 1024


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "ascii"
        )
        + b"\n"
    )


def _read_stable_json(path: Path, *, label: str) -> tuple[dict[str, object], str]:
    digest_before = file_sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} is not valid ASCII JSON") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a mapping")
    if file_sha256(path) != digest_before:
        raise ContractError(f"{label} changed while loading")
    return payload, digest_before


def _sidecar_shards(
    payload: Mapping[str, object],
    source_manifest: DatasetFileManifest,
) -> tuple[CalvinPhysicalSupervisionShard, ...]:
    if payload.get("schema") != CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA:
        raise ContractError("CALVIN sidecar migration requires all-source schema v5")
    if (
        payload.get("coverage") != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
        or payload.get("runtime_input") is not False
        or payload.get("task_conditioned") is not False
    ):
        raise ContractError("CALVIN sidecar migration requires loss-only all-source semantics")
    if (
        payload.get("dataset_id") != source_manifest.dataset_id
        or payload.get("dataset_revision") != source_manifest.dataset_revision
        or payload.get("split_name") != source_manifest.split_name
    ):
        raise ContractError("CALVIN source sidecar identity differs from its dataset manifest")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ContractError("CALVIN source sidecar has no shards")
    shards = tuple(CalvinPhysicalSupervisionShard.from_dict(item) for item in raw_shards)
    if tuple(sorted(shards, key=lambda item: item.first_global_index)) != shards:
        raise ContractError("CALVIN source sidecar shards are not sorted")
    for previous, current in zip(shards, shards[1:], strict=False):
        if current.first_global_index <= previous.last_global_index:
            raise ContractError("CALVIN source sidecar shard ranges overlap")
    frame_count = payload.get("frame_count")
    object_count = payload.get("object_record_count")
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or not isinstance(object_count, int)
        or isinstance(object_count, bool)
        or sum(shard.frame_count for shard in shards) != frame_count
        or sum(shard.object_record_count for shard in shards) != object_count
    ):
        raise ContractError("CALVIN source sidecar counts differ from its shards")
    return shards


def _verify_shards(
    *,
    root: Path,
    shards: tuple[CalvinPhysicalSupervisionShard, ...],
    workers: int,
    progress_every: int,
) -> int:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise TypeError("workers must be a positive integer")
    if (
        not isinstance(progress_every, int)
        or isinstance(progress_every, bool)
        or progress_every <= 0
    ):
        raise TypeError("progress_every must be a positive integer")
    started = time.monotonic()
    verified_bytes = 0
    completed = 0
    next_report = progress_every

    def verify(shard: CalvinPhysicalSupervisionShard) -> int:
        return len(
            read_sha256_verified_file_beneath(
                root,
                shard.path,
                expected_sha256=shard.sha256,
                maximum_bytes=_MAXIMUM_SHARD_BYTES,
            )
        )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sidecar-migration") as pool:
        for size_bytes in pool.map(verify, shards):
            completed += 1
            verified_bytes += size_bytes
            if completed >= next_report or completed == len(shards):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "completed_shards": completed,
                            "elapsed_seconds": round(elapsed, 3),
                            "event": "calvin_sidecar_identity_progress",
                            "mib_per_second": round(verified_bytes / elapsed / 2**20, 3),
                            "total_shards": len(shards),
                            "verified_bytes": verified_bytes,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                while next_report <= completed:
                    next_report += progress_every
    return verified_bytes


def _publish(
    *,
    output_dir: Path,
    sidecar_manifest: dict[str, object],
    receipt: dict[str, object],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    partial.mkdir(exist_ok=False)
    try:
        manifest_bytes = _json_bytes(sidecar_manifest)
        write_bytes_durable_exclusive(partial / OUTPUT_MANIFEST_NAME, manifest_bytes)
        receipt["target_sidecar_manifest"] = {
            "file_name": OUTPUT_MANIFEST_NAME,
            "file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        write_bytes_durable_exclusive(partial / OUTPUT_RECEIPT_NAME, _json_bytes(receipt))
        publish_prepared_directory_durable_exclusive(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _run(args: argparse.Namespace) -> dict[str, object]:
    source_dataset_path = args.source_dataset_manifest.resolve()
    target_dataset_path = args.target_dataset_manifest.resolve()
    source_dataset_sha256 = file_sha256(source_dataset_path)
    target_dataset_sha256 = file_sha256(target_dataset_path)
    source_dataset = load_dataset_file_manifest(source_dataset_path)
    target_dataset = load_dataset_file_manifest(target_dataset_path)
    if (
        file_sha256(source_dataset_path) != source_dataset_sha256
        or file_sha256(target_dataset_path) != target_dataset_sha256
    ):
        raise ContractError("CALVIN dataset manifest changed while loading")
    validate_calvin_content_identity_migration(source_dataset, target_dataset)

    source_receipt_path = args.source_receipt.resolve()
    source_receipt, source_receipt_sha256 = _read_stable_json(
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

    sidecar_root = args.source_sidecar_root.resolve()
    source_sidecar_path = sidecar_root / "manifest.json"
    source_sidecar, source_sidecar_sha256 = _read_stable_json(
        source_sidecar_path,
        label="CALVIN source sidecar manifest",
    )
    if source_sidecar_sha256 != args.expected_source_sidecar_manifest_sha256:
        raise ContractError("CALVIN source sidecar manifest digest differs from the pinned input")
    shards = _sidecar_shards(source_sidecar, source_dataset)
    verified_shard_bytes = _verify_shards(
        root=sidecar_root,
        shards=shards,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    if file_sha256(source_sidecar_path) != source_sidecar_sha256:
        raise ContractError("CALVIN source sidecar manifest changed during shard verification")
    stable_inputs = (
        (source_dataset_path, source_dataset_sha256, "source dataset manifest"),
        (target_dataset_path, target_dataset_sha256, "target dataset manifest"),
        (source_receipt_path, source_receipt_sha256, "official source receipt"),
    )
    for path, expected_sha256, label in stable_inputs:
        if file_sha256(path) != expected_sha256:
            raise ContractError(f"CALVIN {label} changed during sidecar migration")

    migrated_sidecar = dict(source_sidecar)
    migrated_sidecar["dataset_id"] = target_dataset.dataset_id
    migrated_sidecar["dataset_revision"] = target_dataset.dataset_revision
    receipt: dict[str, object] = {
        "schema": MIGRATION_RECEIPT_SCHEMA,
        "source_receipt_sha256": source_receipt_sha256,
        "source_dataset_manifest_sha256": source_dataset_sha256,
        "target_dataset_manifest_sha256": target_dataset_sha256,
        "source_sidecar_manifest_sha256": source_sidecar_sha256,
        "source_sidecar_shard_count": len(shards),
        "verified_sidecar_shard_bytes": verified_shard_bytes,
        "all_sidecar_shard_sha256_matches": True,
        "migration_semantics": "identity-only;immutable-shards-not-copied",
        "target_dataset_id": target_dataset.dataset_id,
        "target_dataset_revision": target_dataset.dataset_revision,
        "training_authorized": False,
        "training_authorization_reason": (
            "identity migration does not replace semantic or action acceptance gates"
        ),
    }
    _publish(
        output_dir=args.output_dir.resolve(),
        sidecar_manifest=migrated_sidecar,
        receipt=receipt,
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-manifest", type=Path, required=True)
    parser.add_argument("--target-dataset-manifest", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-sidecar-root", type=Path, required=True)
    parser.add_argument("--expected-source-sidecar-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=512)
    receipt = _run(parser.parse_args())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
