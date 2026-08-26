#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Materialize the pinned public native-Qwen retention gate from train data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="public native VL retention builder",
)

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from PIL import Image

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.public_native_vl import (
    PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY,
    PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_BYTES,
    PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_PIXELS,
    PUBLIC_NATIVE_VL_REFERRING_DATASET_ID,
    PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION,
    PUBLIC_NATIVE_VL_REFERRING_REVISION,
    PUBLIC_NATIVE_VL_REFERRING_SOURCE_FILE,
    PUBLIC_NATIVE_VL_REFERRING_SOURCE_SHA256,
    PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY,
    PUBLIC_NATIVE_VL_VQA_DATASET_ID,
    PUBLIC_NATIVE_VL_VQA_REVISION,
    PUBLIC_NATIVE_VL_VQA_SOURCE_FILE,
    PUBLIC_NATIVE_VL_VQA_SOURCE_SHA256,
    PublicNativeVLFamily,
    PublicNativeVLManifestRecord,
    PublicNativeVLPartition,
    PublicNativeVLQualityExclusion,
    PublicNativeVLRetentionManifest,
    PublicNativeVLSource,
    native_vl_rgb_sha256,
)

REFCOCO_DATASET_ID = PUBLIC_NATIVE_VL_REFERRING_DATASET_ID
REFCOCO_REVISION = PUBLIC_NATIVE_VL_REFERRING_REVISION
REFCOCO_SOURCE_FILE = PUBLIC_NATIVE_VL_REFERRING_SOURCE_FILE
REFCOCO_SOURCE_SHA256 = PUBLIC_NATIVE_VL_REFERRING_SOURCE_SHA256

VQAV2_DATASET_ID = PUBLIC_NATIVE_VL_VQA_DATASET_ID
VQAV2_REVISION = PUBLIC_NATIVE_VL_VQA_REVISION
VQAV2_SOURCE_FILE = PUBLIC_NATIVE_VL_VQA_SOURCE_FILE
VQAV2_SOURCE_SHA256 = PUBLIC_NATIVE_VL_VQA_SOURCE_SHA256

OUTPUT_MANIFEST = "public_native_vl_retention_manifest.json"
OUTPUT_REPORT = "materialization_report.json"
_BBOX_PATTERN = re.compile(r"<bbox>\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]</bbox>")
_IMAGE_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


@dataclass(frozen=True, slots=True)
class _Candidate:
    family: PublicNativeVLFamily
    source_row_index: int
    source_subindex: int
    priority_sha256: str
    user_text: str
    assistant_text: str
    encoded_image_sha256: str
    image_rgb_sha256: str
    image_format: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _ImageMetadata:
    encoded_image_sha256: str
    image_rgb_sha256: str
    image_format: str
    width: int
    height: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refcoco-parquet", required=True, type=Path)
    parser.add_argument("--vqav2-parquet", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--train-count", type=int, default=PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY
    )
    parser.add_argument(
        "--heldout-count", type=int, default=PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"public native VL source cannot be read: {path}") from error
    return digest.hexdigest()


def _require_exact_source(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"public native VL source is not a regular file: {path}")
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ContractError(
            f"public native VL source digest changed for {path}: {actual} != {expected_sha256}"
        )


def _rows(path: Path, columns: tuple[str, ...]) -> Iterator[tuple[int, dict[str, object]]]:
    try:
        parquet = pq.ParquetFile(path)
        if set(columns) - set(parquet.schema_arrow.names):
            raise ContractError(f"public native VL parquet schema changed: {path}")
        offset = 0
        for batch in parquet.iter_batches(batch_size=64, columns=list(columns)):
            for local_index, raw in enumerate(batch.to_pylist()):
                if not isinstance(raw, dict):
                    raise ContractError("public native VL parquet row is not a mapping")
                yield offset + local_index, cast(dict[str, object], raw)
            offset += len(batch)
        if offset != parquet.metadata.num_rows:
            raise ContractError("public native VL parquet row count changed while scanning")
    except ContractError:
        raise
    except Exception as error:
        raise ContractError(f"public native VL parquet cannot be scanned: {path}") from error


def _image_bytes(value: object) -> bytes:
    if not isinstance(value, dict) or set(value) != {"bytes", "path"}:
        raise ContractError("public native VL embedded image schema changed")
    payload = value.get("bytes")
    if not isinstance(payload, bytes) or not payload:
        raise ContractError("public native VL source image has no embedded bytes")
    if len(payload) > PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_BYTES:
        raise ContractError("public native VL source image exceeds the byte limit")
    return payload


def _decode_image(payload: bytes) -> tuple[NDArray[np.uint8], str]:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            image_format = source.format
            source_width, source_height = source.size
            if (
                source_width <= 0
                or source_height <= 0
                or source_width * source_height > PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_PIXELS
            ):
                raise ContractError("public native VL source image dimensions are invalid")
            source.load()
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    except ContractError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ContractError("public native VL source image cannot be decoded") from error
    if image_format not in _IMAGE_SUFFIXES:
        raise ContractError("public native VL source image encoding is unsupported")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != (source_height, source_width):
        raise ContractError("public native VL source image dimensions are invalid")
    return np.ascontiguousarray(image), image_format


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ContractError(f"public native VL {field} is malformed")
    return value.strip()


def _priority(
    *,
    family: PublicNativeVLFamily,
    row_index: int,
    subindex: int,
    image_rgb_sha256: str,
    user_text: str,
    assistant_text: str,
) -> str:
    value = {
        "assistant_text": assistant_text,
        "family": family,
        "image_rgb_sha256": image_rgb_sha256,
        "source_row_index": row_index,
        "source_subindex": subindex,
        "user_text": user_text,
    }
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(b"picf-next.public-native-vl-priority.v1\0" + payload).hexdigest()


def _candidate(
    *,
    family: PublicNativeVLFamily,
    row_index: int,
    subindex: int,
    image_metadata: _ImageMetadata,
    user_text: str,
    assistant_text: str,
) -> _Candidate:
    return _Candidate(
        family=family,
        source_row_index=row_index,
        source_subindex=subindex,
        priority_sha256=_priority(
            family=family,
            row_index=row_index,
            subindex=subindex,
            image_rgb_sha256=image_metadata.image_rgb_sha256,
            user_text=user_text,
            assistant_text=assistant_text,
        ),
        user_text=user_text,
        assistant_text=assistant_text,
        encoded_image_sha256=image_metadata.encoded_image_sha256,
        image_rgb_sha256=image_metadata.image_rgb_sha256,
        image_format=image_metadata.image_format,
        width=image_metadata.width,
        height=image_metadata.height,
    )


def _image_metadata(payload: bytes) -> _ImageMetadata:
    image, image_format = _decode_image(payload)
    return _ImageMetadata(
        encoded_image_sha256=hashlib.sha256(payload).hexdigest(),
        image_rgb_sha256=native_vl_rgb_sha256(image),
        image_format=image_format,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )


def _one_per_image(candidates: Iterator[_Candidate]) -> tuple[_Candidate, ...]:
    by_image: dict[str, _Candidate] = {}
    for candidate in candidates:
        previous = by_image.get(candidate.image_rgb_sha256)
        if previous is None or candidate.priority_sha256 < previous.priority_sha256:
            by_image[candidate.image_rgb_sha256] = candidate
    return tuple(sorted(by_image.values(), key=lambda item: item.priority_sha256))


def _refcoco_candidates(path: Path) -> tuple[_Candidate, ...]:
    def candidates() -> Iterator[_Candidate]:
        for row_index, row in _rows(path, ("answer", "image", "question")):
            if row_index == PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[0]:
                continue
            question = _text(row["question"], field="RefCOCO question")
            if not question.startswith("[detect]"):
                raise ContractError("public native VL RefCOCO question prefix changed")
            expression = question.removeprefix("[detect]").strip()
            if not expression:
                raise ContractError("public native VL RefCOCO expression is empty")
            raw_answer = _text(row["answer"], field="RefCOCO answer")
            match = _BBOX_PATTERN.fullmatch(raw_answer)
            if match is None:
                raise ContractError("public native VL RefCOCO bbox answer changed")
            bbox = tuple(int(value) for value in match.groups())
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                raise ContractError("public native VL RefCOCO bbox is invalid")
            user_text = (
                f"Locate the region described by this referring expression: {expression}\n"
                'Return its bounding box as a JSON list with key "bbox_2d".'
            )
            assistant_text = json.dumps(
                [{"bbox_2d": list(bbox)}],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            yield _candidate(
                family="referring",
                row_index=row_index,
                subindex=0,
                image_metadata=_image_metadata(_image_bytes(row["image"])),
                user_text=user_text,
                assistant_text=assistant_text,
            )

    return _one_per_image(candidates())


def _vqav2_candidates(path: Path) -> tuple[_Candidate, ...]:
    def candidates() -> Iterator[_Candidate]:
        for row_index, row in _rows(path, ("images", "texts")):
            images = row["images"]
            conversations = row["texts"]
            if not isinstance(images, list) or len(images) != 1:
                raise ContractError("public native VL VQAv2 requires one embedded image per row")
            if not isinstance(conversations, list) or not conversations:
                raise ContractError("public native VL VQAv2 row has no conversations")
            image_payload = _image_bytes(images[0])
            image_metadata = _image_metadata(image_payload)
            row_candidates = []
            for subindex, conversation in enumerate(conversations):
                if not isinstance(conversation, dict) or set(conversation) != {
                    "assistant",
                    "source",
                    "user",
                }:
                    raise ContractError("public native VL VQAv2 conversation schema changed")
                row_candidates.append(
                    _candidate(
                        family="vqa",
                        row_index=row_index,
                        subindex=subindex,
                        image_metadata=image_metadata,
                        user_text=_text(conversation["user"], field="VQAv2 user text"),
                        assistant_text=_text(
                            conversation["assistant"], field="VQAv2 assistant text"
                        ),
                    )
                )
            yield min(row_candidates, key=lambda item: item.priority_sha256)

    return _one_per_image(candidates())


def _selected(
    candidates: tuple[_Candidate, ...],
    *,
    train_count: int,
    heldout_count: int,
) -> tuple[tuple[PublicNativeVLPartition, _Candidate], ...]:
    required = train_count + heldout_count
    if len(candidates) < required:
        raise ContractError(
            f"public native VL source has {len(candidates)} unique images, needs {required}"
        )
    selected: list[tuple[PublicNativeVLPartition, _Candidate]] = []
    selected.extend(("train", candidate) for candidate in candidates[:train_count])
    selected.extend(
        ("heldout", candidate)
        for candidate in candidates[train_count : train_count + heldout_count]
    )
    return tuple(selected)


def _selected_source_bytes(
    path: Path,
    *,
    family: PublicNativeVLFamily,
    selected: tuple[tuple[PublicNativeVLPartition, _Candidate], ...],
) -> dict[tuple[int, str], bytes]:
    by_row = {candidate.source_row_index: candidate for _, candidate in selected}
    columns = ("image",) if family == "referring" else ("images",)
    found: dict[tuple[int, str], bytes] = {}
    for row_index, row in _rows(path, columns):
        candidate = by_row.get(row_index)
        if candidate is None:
            continue
        if family == "referring":
            payload = _image_bytes(row["image"])
        else:
            images = row["images"]
            if not isinstance(images, list) or len(images) != 1:
                raise ContractError("public native VL selected VQAv2 image schema changed")
            payload = _image_bytes(images[0])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != candidate.encoded_image_sha256:
            raise ContractError("public native VL source image changed between scans")
        found[(row_index, digest)] = payload
    if len(found) != len(selected):
        raise ContractError("public native VL failed to recover every selected source image")
    return found


def _records(
    *,
    output_root: Path,
    source_path: Path,
    family: PublicNativeVLFamily,
    selected: tuple[tuple[PublicNativeVLPartition, _Candidate], ...],
) -> tuple[PublicNativeVLManifestRecord, ...]:
    payloads = _selected_source_bytes(source_path, family=family, selected=selected)
    partition_ordinals = {"train": 0, "heldout": 0}
    records = []
    for partition, candidate in selected:
        ordinal = partition_ordinals[partition]
        partition_ordinals[partition] += 1
        record_id = f"{family}-{partition}-{ordinal:04d}"
        suffix = _IMAGE_SUFFIXES[candidate.image_format]
        relative = f"images/{record_id}{suffix}"
        payload = payloads[(candidate.source_row_index, candidate.encoded_image_sha256)]
        write_bytes_durable_exclusive(output_root / relative, payload)
        records.append(
            PublicNativeVLManifestRecord.create(
                record_id=record_id,
                family=family,
                partition=partition,
                source_key=family,
                source_row_index=candidate.source_row_index,
                source_subindex=candidate.source_subindex,
                priority_sha256=candidate.priority_sha256,
                user_text=candidate.user_text,
                assistant_text=candidate.assistant_text,
                image_file=relative,
                image_file_sha256=candidate.encoded_image_sha256,
                image_rgb_sha256=candidate.image_rgb_sha256,
                image_size_bytes=len(payload),
                width=candidate.width,
                height=candidate.height,
            )
        )
    return tuple(records)


def materialize_public_native_vl_retention(
    *,
    refcoco_parquet: Path,
    vqav2_parquet: Path,
    output_root: Path,
    train_count: int,
    heldout_count: int,
) -> dict[str, object]:
    if train_count <= 0 or heldout_count <= 0:
        raise ContractError("public native VL train and heldout counts must be positive")
    partial = output_root.with_name(f"{output_root.name}.partial")
    if any(path.exists() or path.is_symlink() for path in (output_root, partial)):
        raise ContractError("public native VL output or partial root must not already exist")
    _require_exact_source(refcoco_parquet, REFCOCO_SOURCE_SHA256)
    _require_exact_source(vqav2_parquet, VQAV2_SOURCE_SHA256)

    referring = _selected(
        _refcoco_candidates(refcoco_parquet),
        train_count=train_count,
        heldout_count=heldout_count,
    )
    vqa = _selected(
        _vqav2_candidates(vqav2_parquet),
        train_count=train_count,
        heldout_count=heldout_count,
    )
    partial.mkdir(parents=True)
    records = tuple(
        sorted(
            (
                *_records(
                    output_root=partial,
                    source_path=refcoco_parquet,
                    family="referring",
                    selected=referring,
                ),
                *_records(
                    output_root=partial,
                    source_path=vqav2_parquet,
                    family="vqa",
                    selected=vqa,
                ),
            ),
            key=lambda record: record.record_id,
        )
    )
    manifest = PublicNativeVLRetentionManifest.create(
        sources={
            "referring": PublicNativeVLSource(
                dataset_id=REFCOCO_DATASET_ID,
                dataset_revision=REFCOCO_REVISION,
                split="train",
                source_file=REFCOCO_SOURCE_FILE,
                source_file_sha256=REFCOCO_SOURCE_SHA256,
            ),
            "vqa": PublicNativeVLSource(
                dataset_id=VQAV2_DATASET_ID,
                dataset_revision=VQAV2_REVISION,
                split="train",
                source_file=VQAV2_SOURCE_FILE,
                source_file_sha256=VQAV2_SOURCE_SHA256,
            ),
        },
        records=records,
        quality_exclusions=(
            PublicNativeVLQualityExclusion(
                family="referring",
                source_row_index=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[0],
                source_subindex=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[1],
                reason=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[2],
            ),
        ),
    )
    manifest_payload = (
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    write_bytes_durable_exclusive(partial / OUTPUT_MANIFEST, manifest_payload)
    report = {
        "artifact_sha256": manifest.artifact_sha256,
        "family_partition_counts": manifest.family_partition_counts,
        "manifest_file": OUTPUT_MANIFEST,
        "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "output_root": str(output_root.resolve()),
        "quality_exclusions": [item.to_dict() for item in manifest.quality_exclusions],
        "source_file_sha256": {
            "referring": REFCOCO_SOURCE_SHA256,
            "vqa": VQAV2_SOURCE_SHA256,
        },
        "status": "PASS",
    }
    report_payload = (
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    write_bytes_durable_exclusive(partial / OUTPUT_REPORT, report_payload)
    os.replace(partial, output_root)
    return report


def main() -> None:
    args = _parse_args()
    report = materialize_public_native_vl_retention(
        refcoco_parquet=args.refcoco_parquet,
        vqav2_parquet=args.vqav2_parquet,
        output_root=args.output_root,
        train_count=args.train_count,
        heldout_count=args.heldout_count,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
