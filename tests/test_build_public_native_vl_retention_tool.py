from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from picf_next.contracts import ContractError
from picf_next.data.public_native_vl import PublicNativeVLRetentionManifest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
builder = pytest.importorskip("tools.build_public_native_vl_retention")


def _png(color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.full((8, 10, 3), color, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    refcoco = tmp_path / "refcoco.parquet"
    ref_rows = []
    for index in range(4):
        ref_rows.append(
            {
                "image_path": f"image-{index}.png",
                "answer": f"<bbox>[{index + 1}, 20, 300, 400]</bbox>",
                "question": f"[detect]object {index}",
                "image": {"bytes": _png((index, index + 1, index + 2)), "path": None},
            }
        )
    pq.write_table(pa.Table.from_pylist(ref_rows), refcoco)

    vqav2 = tmp_path / "vqav2.parquet"
    vqa_rows = []
    for index in range(4):
        vqa_rows.append(
            {
                "images": [{"bytes": _png((index + 10, index + 11, index + 12)), "path": None}],
                "texts": [
                    {
                        "user": f"Question {index}A?",
                        "assistant": f"Answer {index}A.",
                        "source": "VQAv2",
                    },
                    {
                        "user": f"Question {index}B?",
                        "assistant": f"Answer {index}B.",
                        "source": "VQAv2",
                    },
                ],
            }
        )
    pq.write_table(pa.Table.from_pylist(vqa_rows), vqav2)
    return refcoco, vqav2


def _bind_source_digests(monkeypatch: pytest.MonkeyPatch, refcoco: Path, vqav2: Path) -> None:
    monkeypatch.setattr(builder, "REFCOCO_SOURCE_SHA256", builder._file_sha256(refcoco))
    monkeypatch.setattr(builder, "VQAV2_SOURCE_SHA256", builder._file_sha256(vqav2))


def test_public_native_vl_builder_is_deterministic_and_materializes_every_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refcoco, vqav2 = _write_sources(tmp_path)
    _bind_source_digests(monkeypatch, refcoco, vqav2)
    first = tmp_path / "first"
    second = tmp_path / "second"

    report = builder.materialize_public_native_vl_retention(
        refcoco_parquet=refcoco,
        vqav2_parquet=vqav2,
        output_root=first,
        train_count=2,
        heldout_count=1,
    )
    builder.materialize_public_native_vl_retention(
        refcoco_parquet=refcoco,
        vqav2_parquet=vqav2,
        output_root=second,
        train_count=2,
        heldout_count=1,
    )

    assert report["status"] == "PASS"
    assert report["family_partition_counts"] == {
        "referring/heldout": 1,
        "referring/train": 2,
        "vqa/heldout": 1,
        "vqa/train": 2,
    }
    assert (first / builder.OUTPUT_MANIFEST).read_bytes() == (
        second / builder.OUTPUT_MANIFEST
    ).read_bytes()
    manifest = PublicNativeVLRetentionManifest.load(first / builder.OUTPUT_MANIFEST)
    assert len(manifest.records) == 6
    for record in manifest.records:
        runtime = manifest.materialize_record(record, artifact_root=first)
        assert runtime.record_id == record.record_id


def test_public_native_vl_builder_fails_closed_on_digest_schema_and_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refcoco, vqav2 = _write_sources(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(builder, "REFCOCO_SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(builder, "VQAV2_SOURCE_SHA256", hashlib.sha256(b"wrong").hexdigest())
    with pytest.raises(ContractError, match="source digest changed"):
        builder.materialize_public_native_vl_retention(
            refcoco_parquet=refcoco,
            vqav2_parquet=vqav2,
            output_root=output,
            train_count=2,
            heldout_count=1,
        )

    _bind_source_digests(monkeypatch, refcoco, vqav2)
    output.mkdir()
    with pytest.raises(ContractError, match="must not already exist"):
        builder.materialize_public_native_vl_retention(
            refcoco_parquet=refcoco,
            vqav2_parquet=vqav2,
            output_root=output,
            train_count=2,
            heldout_count=1,
        )


def test_public_native_vl_builder_rejects_changed_refcoco_bbox_schema(tmp_path: Path) -> None:
    refcoco = tmp_path / "invalid.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "answer": "[1,2,3,4]",
                    "question": "[detect]object",
                    "image": {"bytes": _png((1, 2, 3)), "path": None},
                }
            ]
        ),
        refcoco,
    )
    with pytest.raises(ContractError, match="bbox answer changed"):
        builder._refcoco_candidates(refcoco)
