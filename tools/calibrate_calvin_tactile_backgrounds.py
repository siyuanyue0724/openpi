#!/usr/bin/env python3
"""Publish manifest-bound CALVIN DIGIT optical backgrounds and a receipt."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_tactile import (
    CALVIN_TACTILE_SOURCE_COMMIT,
    CALVIN_TACTILE_SOURCE_FILES_SHA256,
    CALVIN_TACTILE_STREAM_NAMES,
)
from picf_next.data.calvin_tactile_calibration import (
    CALVIN_TACTILE_CALIBRATION_SCHEMA,
    CalvinTactileBackgroundCalibration,
    CalvinTactileCalibrationSample,
    build_calvin_tactile_background_calibration,
    canonical_calibration_receipt_sha256,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    file_sha256,
    load_dataset_file_manifest,
    read_verified_dataset_file,
)
from tools.audit_calvin_tactile_validity import (
    AUDIT_SCHEMA,
    deterministic_sample_steps,
)

CALIBRATION_ARCHIVE_SCHEMA = "picf-next.calvin-digit-background-archive/v1"
MAXIMUM_FRAME_BYTES = 32 * 1024 * 1024


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid calibration input JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"calibration input JSON must be one mapping: {path}")
    return payload


def _audit_steps(
    split_root: Path,
    manifest: DatasetFileManifest,
    audit: dict[str, object],
) -> tuple[int, ...]:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ContractError("tactile audit schema differs from the calibration contract")
    sample_count = audit.get("requested_uniform_sample_count")
    include_steps = audit.get("explicit_include_steps")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
        or not isinstance(include_steps, list)
        or any(isinstance(step, bool) or not isinstance(step, int) for step in include_steps)
    ):
        raise ContractError("tactile audit sampling fields are invalid")
    ranges_record = manifest.record_for("ep_start_end_ids.npy")
    ranges_payload = read_verified_dataset_file(
        manifest,
        split_root,
        ranges_record.path,
        maximum_bytes=max(ranges_record.size_bytes, 1),
    )
    ranges = np.load(io.BytesIO(ranges_payload), allow_pickle=False)
    steps = deterministic_sample_steps(
        ranges,
        sample_count=sample_count,
        include_steps=include_steps,
    )
    digest = hashlib.sha256(np.asarray(steps, dtype="<i8").tobytes()).hexdigest()
    if (
        audit.get("sampled_steps_count") != len(steps)
        or audit.get("sampled_steps_sha256") != digest
    ):
        raise ContractError("tactile audit sample identity cannot be reproduced")
    return steps


def _load_sample(
    *,
    split_root: Path,
    manifest: DatasetFileManifest,
    step: int,
) -> CalvinTactileCalibrationSample:
    relative = f"episode_{step:07d}.npz"
    record = manifest.record_for(relative)
    if record.size_bytes > MAXIMUM_FRAME_BYTES:
        raise ContractError(f"CALVIN frame exceeds calibration memory budget: {relative}")
    payload = read_verified_dataset_file(
        manifest,
        split_root,
        relative,
        maximum_bytes=MAXIMUM_FRAME_BYTES,
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if "rgb_tactile" not in archive.files or "depth_tactile" not in archive.files:
                raise ContractError(f"CALVIN frame omits tactile arrays: {relative}")
            rgb = np.asarray(archive["rgb_tactile"]).copy()
            deformation = np.asarray(archive["depth_tactile"]).astype(np.float32, copy=True)
    except (OSError, ValueError) as error:
        raise ContractError(f"unsafe or corrupt CALVIN frame: {relative}") from error
    rgb.setflags(write=False)
    deformation.setflags(write=False)
    return CalvinTactileCalibrationSample(
        source_global_index=step,
        source_file_sha256=record.sha256,
        rgb=rgb,
        deformation_m=deformation,
    )


def _write_archive(
    path: Path,
    *,
    calibration: CalvinTactileBackgroundCalibration,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(CALIBRATION_ARCHIVE_SCHEMA),
                left_digit=calibration.backgrounds_by_stream["left_digit"],
                right_digit=calibration.backgrounds_by_stream["right_digit"],
                left_digit_selected_steps=np.asarray(
                    calibration.selected_steps_by_stream["left_digit"], dtype=np.int64
                ),
                right_digit_selected_steps=np.asarray(
                    calibration.selected_steps_by_stream["right_digit"], dtype=np.int64
                ),
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--tactile-audit", required=True, type=Path)
    parser.add_argument("--visual-review-manifest", required=True, type=Path)
    parser.add_argument("--background-noise-ceiling-m", type=float, default=1e-6)
    parser.add_argument("--validity-threshold-m", type=float, default=1e-4)
    parser.add_argument("--minimum-candidates", type=int, default=16)
    parser.add_argument("--maximum-selected", type=int, default=256)
    parser.add_argument("--output-archive", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    args = parser.parse_args()

    split_root = args.split_root.resolve()
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    if manifest.split_name != split_root.name:
        raise ContractError("CALVIN calibration split differs from its manifest")
    audit = _load_json(args.tactile_audit)
    steps = _audit_steps(split_root, manifest, audit)
    calibration = build_calvin_tactile_background_calibration(
        (_load_sample(split_root=split_root, manifest=manifest, step=step) for step in steps),
        background_noise_ceiling_m=args.background_noise_ceiling_m,
        validity_thresholds_m={
            name: args.validity_threshold_m for name in CALVIN_TACTILE_STREAM_NAMES
        },
        minimum_candidates_per_stream=args.minimum_candidates,
        maximum_selected_per_stream=args.maximum_selected,
    )
    archive_sha256 = _write_archive(args.output_archive, calibration=calibration)
    receipt = {
        "schema": CALVIN_TACTILE_CALIBRATION_SCHEMA,
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "file_count": len(manifest.files),
            "manifest_sha256": file_sha256(args.dataset_manifest),
            "split_name": manifest.split_name,
            "tree_sha256": manifest.tree_sha256,
        },
        "sampling": {
            "sample_count": len(steps),
            "sampled_steps_sha256": hashlib.sha256(
                np.asarray(steps, dtype="<i8").tobytes()
            ).hexdigest(),
            "tactile_audit_sha256": file_sha256(args.tactile_audit),
            "visual_review_manifest_sha256": file_sha256(args.visual_review_manifest),
        },
        "official_calvin_source": {
            "commit": CALVIN_TACTILE_SOURCE_COMMIT,
            "files_sha256": CALVIN_TACTILE_SOURCE_FILES_SHA256,
        },
        "calibration": calibration.receipt_payload(),
        "archive": {
            "path": str(args.output_archive.resolve()),
            "sha256": archive_sha256,
        },
    }
    receipt["receipt_payload_sha256"] = canonical_calibration_receipt_sha256(receipt)
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    args.output_receipt.write_text(encoded, encoding="ascii")
    print(encoded, end="")


if __name__ == "__main__":
    main()
