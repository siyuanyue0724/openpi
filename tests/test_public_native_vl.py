from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from picf_next.contracts import ContractError
from picf_next.data.public_native_vl import (
    PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY,
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
    NativeVLInstructionRecord,
    PublicNativeVLFamily,
    PublicNativeVLManifestRecord,
    PublicNativeVLPartition,
    PublicNativeVLQualityExclusion,
    PublicNativeVLRetentionManifest,
    PublicNativeVLSource,
    load_frozen_public_native_vl_retention_gate,
    native_vl_rgb_sha256,
    validate_frozen_public_native_vl_retention_gate,
)


def _source(dataset: str) -> PublicNativeVLSource:
    return PublicNativeVLSource(
        dataset_id=dataset,
        dataset_revision="revision",
        split="train",
        source_file=f"source/{dataset}.parquet",
        source_file_sha256=hashlib.sha256(dataset.encode("ascii")).hexdigest(),
    )


def _record(
    root: Path,
    *,
    family: str,
    partition: str,
    index: int,
    color: tuple[int, int, int],
) -> PublicNativeVLManifestRecord:
    image = np.full((4, 5, 3), color, dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    payload = stream.getvalue()
    relative = f"images/{family}-{partition}-{index}.png"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    record_id = f"{family}-{partition}-{index:04d}"
    return PublicNativeVLManifestRecord.create(
        record_id=record_id,
        family=cast(PublicNativeVLFamily, family),
        partition=cast(PublicNativeVLPartition, partition),
        source_key=family,
        source_row_index=index + (0 if partition == "train" else 100),
        source_subindex=0,
        priority_sha256=hashlib.sha256(record_id.encode("ascii")).hexdigest(),
        user_text="Is the object blue?" if family == "vqa" else "Locate the blue object.",
        assistant_text=("blue" if family == "vqa" else '[{"bbox_2d":[100,200,300,400]}]'),
        image_file=relative,
        image_file_sha256=hashlib.sha256(payload).hexdigest(),
        image_rgb_sha256=native_vl_rgb_sha256(image),
        image_size_bytes=len(payload),
        width=image.shape[1],
        height=image.shape[0],
    )


def _manifest(root: Path) -> PublicNativeVLRetentionManifest:
    records = (
        _record(root, family="referring", partition="heldout", index=0, color=(1, 2, 3)),
        _record(root, family="referring", partition="train", index=0, color=(4, 5, 6)),
        _record(root, family="vqa", partition="heldout", index=0, color=(7, 8, 9)),
        _record(root, family="vqa", partition="train", index=0, color=(10, 11, 12)),
    )
    return PublicNativeVLRetentionManifest.create(
        sources={"referring": _source("refcoco"), "vqa": _source("vqav2")},
        records=records,
    )


def test_public_native_vl_manifest_round_trip_and_verified_materialization(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    loaded = PublicNativeVLRetentionManifest.load(path)
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.family_partition_counts == {
        "referring/heldout": 1,
        "referring/train": 1,
        "vqa/heldout": 1,
        "vqa/train": 1,
    }
    assert loaded.training_record_for_rank(optimizer_step=0, rank=0).family == "referring"
    assert loaded.training_record_for_rank(optimizer_step=0, rank=1).family == "vqa"
    runtime = loaded.materialize_record(
        loaded.training_record_for_rank(optimizer_step=0, rank=1),
        artifact_root=tmp_path,
    )
    assert isinstance(runtime, NativeVLInstructionRecord)
    assert runtime.assistant_text == "blue"
    assert not runtime.image.flags.writeable
    assert runtime.qwen_user_messages()[0]["content"][1]["text"] == "Is the object blue?"


def test_public_native_vl_fails_closed_on_image_corruption(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = manifest.training_record_for_rank(optimizer_step=0, rank=0)
    (tmp_path / record.image_file).write_bytes(b"changed")
    with pytest.raises(ContractError, match="content hash mismatch"):
        manifest.materialize_record(record, artifact_root=tmp_path)


def test_public_native_vl_rejects_overlap_duplicates_and_mutation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    records = list(manifest.records)
    referring_train = next(
        record for record in records if record.family == "referring" and record.partition == "train"
    )
    referring_heldout_index = next(
        index
        for index, record in enumerate(records)
        if record.family == "referring" and record.partition == "heldout"
    )
    heldout = records[referring_heldout_index]
    records[referring_heldout_index] = PublicNativeVLManifestRecord.create(
        **{
            **heldout._payload(),
            "image_rgb_sha256": referring_train.image_rgb_sha256,
        }
    )
    with pytest.raises(ContractError, match="overlap"):
        PublicNativeVLRetentionManifest.create(
            sources=manifest.sources,
            records=tuple(records),
        )
    with pytest.raises(TypeError):
        cast(dict[str, PublicNativeVLSource], manifest.sources)["vqa"] = _source("replacement")


def test_public_native_vl_rejects_cross_family_train_heldout_leakage(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    records = list(manifest.records)
    referring_train = next(
        record for record in records if record.family == "referring" and record.partition == "train"
    )
    vqa_heldout_index = next(
        index
        for index, record in enumerate(records)
        if record.family == "vqa" and record.partition == "heldout"
    )
    heldout = records[vqa_heldout_index]
    records[vqa_heldout_index] = PublicNativeVLManifestRecord.create(
        **{
            **heldout._payload(),
            "image_rgb_sha256": referring_train.image_rgb_sha256,
        }
    )
    with pytest.raises(ContractError, match="across task families"):
        PublicNativeVLRetentionManifest.create(
            sources=manifest.sources,
            records=tuple(records),
        )


def test_public_native_vl_rejects_changed_manifest_and_unsafe_paths(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    value = manifest.to_dict()
    value["artifact_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="artifact SHA-256 changed"):
        PublicNativeVLRetentionManifest.from_dict(value)

    record = manifest.records[0]
    with pytest.raises(ContractError, match="normalized and relative"):
        replace(record, image_file="../escape.png")
    with pytest.raises(ContractError, match="exceeds"):
        manifest.training_record_for_rank(optimizer_step=1, rank=0)
    with pytest.raises(ContractError, match="rank zero or one"):
        manifest.training_record_for_rank(optimizer_step=0, rank=2)


def test_public_native_vl_rejects_non_train_source() -> None:
    with pytest.raises(ContractError, match="only a train split"):
        PublicNativeVLSource(
            dataset_id="dataset",
            dataset_revision="revision",
            split="validation",
            source_file="source/data.parquet",
            source_file_sha256="0" * 64,
        )


def _frozen_manifest(root: Path) -> PublicNativeVLRetentionManifest:
    sources = {
        "referring": PublicNativeVLSource(
            dataset_id=PUBLIC_NATIVE_VL_REFERRING_DATASET_ID,
            dataset_revision=PUBLIC_NATIVE_VL_REFERRING_REVISION,
            split="train",
            source_file=PUBLIC_NATIVE_VL_REFERRING_SOURCE_FILE,
            source_file_sha256=PUBLIC_NATIVE_VL_REFERRING_SOURCE_SHA256,
        ),
        "vqa": PublicNativeVLSource(
            dataset_id=PUBLIC_NATIVE_VL_VQA_DATASET_ID,
            dataset_revision=PUBLIC_NATIVE_VL_VQA_REVISION,
            split="train",
            source_file=PUBLIC_NATIVE_VL_VQA_SOURCE_FILE,
            source_file_sha256=PUBLIC_NATIVE_VL_VQA_SOURCE_SHA256,
        ),
    }
    records = []
    color_index = 0
    for family in ("referring", "vqa"):
        for partition, count, row_offset in (
            ("heldout", PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY, 10_000),
            ("train", PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY, 0),
        ):
            for index in range(count):
                identity = f"{family}-{partition}-{index:04d}"
                image = np.array(
                    [[[(color_index >> 16) & 255, (color_index >> 8) & 255, color_index & 255]]],
                    dtype=np.uint8,
                )
                color_index += 1
                stream = io.BytesIO()
                Image.fromarray(image).save(stream, format="PNG")
                image_payload = stream.getvalue()
                image_file = f"images/{identity}.png"
                image_path = root / image_file
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_payload)
                records.append(
                    PublicNativeVLManifestRecord.create(
                        record_id=identity,
                        family=cast(PublicNativeVLFamily, family),
                        partition=cast(PublicNativeVLPartition, partition),
                        source_key=family,
                        source_row_index=row_offset + index,
                        source_subindex=0,
                        priority_sha256=f"{index:064x}",
                        user_text=f"Question {identity}",
                        assistant_text=f"Answer {identity}",
                        image_file=image_file,
                        image_file_sha256=hashlib.sha256(image_payload).hexdigest(),
                        image_rgb_sha256=native_vl_rgb_sha256(image),
                        image_size_bytes=len(image_payload),
                        width=1,
                        height=1,
                    )
                )
    exclusion = PublicNativeVLQualityExclusion(
        family="referring",
        source_row_index=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[0],
        source_subindex=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[1],
        reason=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[2],
    )
    return PublicNativeVLRetentionManifest.create(
        sources=sources,
        records=tuple(sorted(records, key=lambda record: record.record_id)),
        quality_exclusions=(exclusion,),
    )


def test_frozen_public_native_vl_gate_accepts_only_the_preregistered_contract(
    tmp_path: Path,
) -> None:
    manifest = _frozen_manifest(tmp_path)
    assert validate_frozen_public_native_vl_retention_gate(manifest, max_steps=64) is manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_payload = (
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    manifest_path.write_bytes(manifest_payload)
    loaded = load_frozen_public_native_vl_retention_gate(
        manifest_path=manifest_path,
        manifest_file_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        artifact_root=tmp_path,
        max_steps=64,
    )
    assert loaded.artifact_sha256 == manifest.artifact_sha256
    with pytest.raises(ContractError, match="manifest file changed"):
        load_frozen_public_native_vl_retention_gate(
            manifest_path=manifest_path,
            manifest_file_sha256="0" * 64,
            artifact_root=tmp_path,
            max_steps=1,
        )

    changed_sources = dict(manifest.sources)
    changed_sources["vqa"] = replace(
        changed_sources["vqa"],
        dataset_revision="changed",
    )
    changed_source_manifest = PublicNativeVLRetentionManifest.create(
        sources=changed_sources,
        records=manifest.records,
        quality_exclusions=manifest.quality_exclusions,
    )
    with pytest.raises(ContractError, match="sources changed"):
        validate_frozen_public_native_vl_retention_gate(
            changed_source_manifest,
            max_steps=1,
        )

    changed_exclusion_manifest = PublicNativeVLRetentionManifest.create(
        sources=manifest.sources,
        records=manifest.records,
    )
    with pytest.raises(ContractError, match="quality exclusions changed"):
        validate_frozen_public_native_vl_retention_gate(
            changed_exclusion_manifest,
            max_steps=1,
        )
