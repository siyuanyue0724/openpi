#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Render labelled contact sheets for every frozen public retention record."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import textwrap
from pathlib import Path
from typing import cast

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="public native VL retention renderer",
)

from PIL import Image, ImageDraw, ImageFont

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.public_native_vl import (
    PUBLIC_NATIVE_VL_FAMILIES,
    PUBLIC_NATIVE_VL_PARTITIONS,
    PublicNativeVLFamily,
    PublicNativeVLPartition,
    PublicNativeVLRetentionManifest,
    validate_frozen_public_native_vl_retention_gate,
)

_COLUMNS = 4
_ROWS = 4
_TILE_WIDTH = 320
_TILE_HEIGHT = 300
_IMAGE_WIDTH = 300
_IMAGE_HEIGHT = 210
_HEADER_HEIGHT = 34


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(text: str) -> tuple[int, int, int, int]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ContractError("public retention audit bbox answer is not JSON") from error
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], dict)
        or set(value[0]) != {"bbox_2d"}
        or not isinstance(value[0]["bbox_2d"], list)
        or len(value[0]["bbox_2d"]) != 4
        or any(not isinstance(item, int) for item in value[0]["bbox_2d"])
    ):
        raise ContractError("public retention audit bbox answer schema changed")
    x0, y0, x1, y1 = value[0]["bbox_2d"]
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ContractError("public retention audit bbox answer is invalid")
    return x0, y0, x1, y1


def _fit_image(image: Image.Image) -> tuple[Image.Image, float, int, int]:
    scale = min(_IMAGE_WIDTH / image.width, _IMAGE_HEIGHT / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    left = (_IMAGE_WIDTH - width) // 2
    top = (_IMAGE_HEIGHT - height) // 2
    return resized, scale, left, top


def _draw_record(
    sheet: Image.Image,
    *,
    manifest: PublicNativeVLRetentionManifest,
    artifact_root: Path,
    record_index: int,
    x: int,
    y: int,
) -> None:
    record = manifest.records[record_index]
    runtime = manifest.materialize_record(record, artifact_root=artifact_root)
    source = Image.fromarray(runtime.image)
    resized, scale, left, top = _fit_image(source)
    image_x = x + 10 + left
    image_y = y + 8 + top
    sheet.paste(resized, (image_x, image_y))
    draw = ImageDraw.Draw(sheet)
    if record.family == "referring":
        x0, y0, x1, y1 = _bbox(record.assistant_text)
        source_box = (
            x0 / 1000 * source.width,
            y0 / 1000 * source.height,
            x1 / 1000 * source.width,
            y1 / 1000 * source.height,
        )
        draw.rectangle(
            (
                image_x + source_box[0] * scale,
                image_y + source_box[1] * scale,
                image_x + source_box[2] * scale,
                image_y + source_box[3] * scale,
            ),
            outline=(235, 35, 35),
            width=3,
        )
    text_y = y + _IMAGE_HEIGHT + 14
    label = f"{record.record_id} row={record.source_row_index}:{record.source_subindex}"
    draw.text((x + 8, text_y), label, fill=(15, 15, 15), font=ImageFont.load_default())
    prompt = record.user_text.replace("\n", " ")
    answer = record.assistant_text.replace("\n", " ")
    lines = textwrap.wrap(f"Q: {prompt}", width=48)[:2]
    lines.extend(textwrap.wrap(f"A: {answer}", width=48)[:1])
    for line_index, line in enumerate(lines):
        draw.text(
            (x + 8, text_y + 16 + line_index * 14),
            line,
            fill=(30, 30, 30),
            font=ImageFont.load_default(),
        )
    draw.rectangle(
        (x, y, x + _TILE_WIDTH - 1, y + _TILE_HEIGHT - 1),
        outline=(170, 170, 170),
        width=1,
    )


def render_public_native_vl_retention_audit(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    if _sha256(manifest_path) != manifest_sha256:
        raise ContractError("public retention audit manifest file changed")
    partial = output_dir.with_name(f"{output_dir.name}.partial")
    if any(path.exists() or path.is_symlink() for path in (output_dir, partial)):
        raise ContractError("public retention audit output or partial directory must not exist")
    manifest = validate_frozen_public_native_vl_retention_gate(
        PublicNativeVLRetentionManifest.load(manifest_path),
        max_steps=1,
    )
    partial.mkdir(parents=True)
    panels = []
    for raw_family in PUBLIC_NATIVE_VL_FAMILIES:
        family = cast(PublicNativeVLFamily, raw_family)
        for raw_partition in PUBLIC_NATIVE_VL_PARTITIONS:
            partition = cast(PublicNativeVLPartition, raw_partition)
            records = manifest.records_for(family, partition)
            indices = tuple(manifest.records.index(record) for record in records)
            page_count = math.ceil(len(indices) / (_COLUMNS * _ROWS))
            for page in range(page_count):
                title = f"{family}/{partition} page {page + 1}/{page_count}"
                sheet = Image.new(
                    "RGB",
                    (_COLUMNS * _TILE_WIDTH, _HEADER_HEIGHT + _ROWS * _TILE_HEIGHT),
                    (248, 248, 248),
                )
                draw = ImageDraw.Draw(sheet)
                draw.text((10, 10), title, fill=(0, 0, 0), font=ImageFont.load_default())
                selected = indices[page * _COLUMNS * _ROWS : (page + 1) * _COLUMNS * _ROWS]
                for local_index, record_index in enumerate(selected):
                    _draw_record(
                        sheet,
                        manifest=manifest,
                        artifact_root=artifact_root,
                        record_index=record_index,
                        x=(local_index % _COLUMNS) * _TILE_WIDTH,
                        y=_HEADER_HEIGHT + (local_index // _COLUMNS) * _TILE_HEIGHT,
                    )
                stream = io.BytesIO()
                sheet.save(stream, format="PNG", optimize=True)
                payload = stream.getvalue()
                filename = f"{family}-{partition}-page-{page + 1:02d}.png"
                write_bytes_durable_exclusive(partial / filename, payload)
                panels.append(
                    {
                        "family": family,
                        "file": filename,
                        "page": page + 1,
                        "record_count": len(selected),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
    report = {
        "manifest_artifact_sha256": manifest.artifact_sha256,
        "manifest_file_sha256": manifest_sha256,
        "panels": panels,
        "quality_exclusions": [item.to_dict() for item in manifest.quality_exclusions],
        "record_count": len(manifest.records),
        "status": "PASS",
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_durable_exclusive(partial / "report.json", payload)
    os.replace(partial, output_dir)
    return report


def main() -> None:
    args = _parse_args()
    report = render_public_native_vl_retention_audit(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
