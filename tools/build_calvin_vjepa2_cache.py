#!/usr/bin/env python3
"""Build a resumable content-addressed V-JEPA2 cache from target-free CALVIN RGB."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.causal_video import build_calvin_causal_video_clip
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    read_sha256_verified_file_beneath,
)
from picf_next.data.vjepa2_cache import (
    VJEPA2_CACHE_AUGMENTATION,
    VJEPA2_CACHE_SCHEMA,
    VJEPA2_CONTEXT_SENSORS,
)
from picf_next.encoders.vjepa2 import (
    VJEPA2_MODEL_ID,
    VJEPA2_MODEL_REVISION,
    Vjepa2DenseEncoder,
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clip_digest(encoder_contract: str, sensor_key: str, source_hashes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"picf-next.vjepa2-cache-key/v1\0")
    digest.update(encoder_contract.encode("utf-8"))
    digest.update(b"\0")
    digest.update(sensor_key.encode("utf-8"))
    for source_hash in source_hashes:
        digest.update(b"\0")
        digest.update(source_hash.encode("ascii"))
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _token_artifact(
    root: Path,
    *,
    sample_key: str,
    modality: str,
    clip_digest: str,
    tokens: np.ndarray | None,
    expected_shape: tuple[int, int],
    trusted_sha256: str | None,
) -> tuple[str, str, int]:
    sample_digest = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()
    relative = f"entries/{sample_digest}/{modality}-{clip_digest}.tokens.npy"
    path = root / relative
    if path.is_symlink():
        raise RuntimeError(f"refusing a symlinked cache token artifact: {path}")
    if trusted_sha256 is not None:
        payload = read_sha256_verified_file_beneath(
            root,
            relative,
            expected_sha256=trusted_sha256,
            maximum_bytes=expected_shape[0] * expected_shape[1] * 4 + 4096,
        )
        try:
            existing = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"existing cache artifact is not a safe NPY array: {path}") from exc
        if (
            not isinstance(existing, np.ndarray)
            or existing.shape != expected_shape
            or existing.dtype != np.float32
            or not np.isfinite(existing).all()
        ):
            raise RuntimeError(f"existing cache artifact differs from its expected tensor: {path}")
        return relative, trusted_sha256, len(payload)
    if tokens is None:
        raise RuntimeError("an untrusted cache artifact requires freshly encoded tokens")
    array = np.asarray(tokens)
    if array.shape != expected_shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise RuntimeError("fresh V-JEPA2 tokens differ from the expected cache tensor")
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    _write_atomic(path, payload)
    return relative, _sha256(payload), len(payload)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sha_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _trusted_partial_artifacts(
    root: Path,
    *,
    expected_manifest: Mapping[str, object],
    sample_keys: Sequence[str],
) -> dict[tuple[str, str, str, tuple[str, ...], int, str], str]:
    """Authenticate the checkpointed prefix; ignore later uncommitted artifacts."""

    partial_path = root / "manifest.partial.json"
    if not partial_path.exists():
        return {}
    if partial_path.is_symlink() or partial_path.stat().st_size > 32 * 1024 * 1024:
        raise RuntimeError("partial V-JEPA2 manifest is unsafe or unexpectedly large")
    try:
        payload = _mapping(json.loads(partial_path.read_bytes()), "partial V-JEPA2 manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("partial V-JEPA2 manifest is not valid JSON") from exc
    if set(payload) != set(expected_manifest):
        raise RuntimeError("partial V-JEPA2 manifest fields changed")
    for field, expected in expected_manifest.items():
        if field != "entries" and payload[field] != expected:
            raise RuntimeError(f"partial V-JEPA2 manifest changed {field!r}")

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > len(sample_keys):
        raise RuntimeError("partial V-JEPA2 entries are not a bounded checkpoint prefix")
    encoder = _mapping(expected_manifest["encoder"], "expected V-JEPA2 encoder")
    image_size = int(encoder["image_size"])
    patch_size = int(encoder["patch_size"])
    tubelet_size = int(encoder["tubelet_size"])
    maximum_frames = int(encoder["maximum_frames"])
    patches_per_frame = (image_size // patch_size) ** 2

    trusted: dict[tuple[str, str, str, tuple[str, ...], int, str], str] = {}
    for entry_index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"partial entry[{entry_index}]")
        if set(entry) != {"sample_key", "sensors"}:
            raise RuntimeError("partial V-JEPA2 entry fields changed")
        sample_key = entry["sample_key"]
        if sample_key != sample_keys[entry_index]:
            raise RuntimeError("partial V-JEPA2 entries are not the expected sorted prefix")
        sensors = entry["sensors"]
        if not isinstance(sensors, list) or len(sensors) != len(VJEPA2_CONTEXT_SENSORS):
            raise RuntimeError("partial V-JEPA2 entry does not contain both cameras")
        for sensor_index, (raw_sensor, expected_sensor) in enumerate(
            zip(sensors, VJEPA2_CONTEXT_SENSORS, strict=True)
        ):
            sensor = _mapping(raw_sensor, f"partial sensor[{entry_index},{sensor_index}]")
            if set(sensor) != {
                "artifact_path",
                "artifact_sha256",
                "modality",
                "sensor_key",
                "source_frame_sha256",
                "token_count",
            }:
                raise RuntimeError("partial V-JEPA2 sensor fields changed")
            sensor_key, modality = expected_sensor
            if (sensor["sensor_key"], sensor["modality"]) != expected_sensor:
                raise RuntimeError("partial V-JEPA2 sensor order or identity changed")
            raw_hashes = sensor["source_frame_sha256"]
            if not isinstance(raw_hashes, list):
                raise RuntimeError("partial V-JEPA2 source hashes must be a list")
            source_hashes = tuple(
                _sha_text(value, "partial source frame sha256") for value in raw_hashes
            )
            if (
                len(set(source_hashes)) != len(source_hashes)
                or len(source_hashes) > maximum_frames
                or (source_hashes and len(source_hashes) % tubelet_size)
            ):
                raise RuntimeError("partial V-JEPA2 clip violates the causal tubelet contract")
            token_count = sensor["token_count"]
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count != len(source_hashes) // tubelet_size * patches_per_frame
            ):
                raise RuntimeError("partial V-JEPA2 token count changed")
            if token_count == 0:
                if sensor["artifact_path"] is not None or sensor["artifact_sha256"] is not None:
                    raise RuntimeError("an unavailable partial clip cannot carry an artifact")
                continue
            relative = sensor["artifact_path"]
            if not isinstance(relative, str):
                raise RuntimeError("partial V-JEPA2 artifact path must be text")
            artifact_sha = _sha_text(sensor["artifact_sha256"], "partial artifact sha256")
            key = (sample_key, sensor_key, modality, source_hashes, token_count, relative)
            if key in trusted:
                raise RuntimeError("partial V-JEPA2 manifest repeats an artifact")
            trusted[key] = artifact_sha
    return trusted


def _manifest(
    *,
    dataset_tree_sha256: str,
    encoder: Vjepa2DenseEncoder,
    maximum_frames: int,
    expected_entries: int,
    entries: list[dict[str, object]],
    complete: bool,
) -> dict[str, object]:
    return {
        "augmentation_contract": VJEPA2_CACHE_AUGMENTATION,
        "complete": complete,
        "dataset_tree_sha256": dataset_tree_sha256,
        "encoder": {
            "checkpoint_revision": encoder.checkpoint_revision,
            "encoder_contract": encoder.encoder_contract,
            "hidden_size": encoder.hidden_size,
            "image_size": encoder.image_size,
            "maximum_frames": maximum_frames,
            "model_id": encoder.model_id,
            "patch_size": encoder.patch_size,
            "tubelet_size": encoder.tubelet_size,
        },
        "entries": entries,
        "expected_entries": expected_entries,
        "schema": VJEPA2_CACHE_SCHEMA,
        "sensors": [
            {"sensor_key": sensor_key, "modality": modality}
            for sensor_key, modality in VJEPA2_CONTEXT_SENSORS
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split", default="training", choices=("training", "validation"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--maximum-frames", default=4, type=int)
    parser.add_argument("--checkpoint-interval", default=25, type=int)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.maximum_frames <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("frame and checkpoint counts must be positive")

    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        (args.dataset_root / args.split).resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    encoder = Vjepa2DenseEncoder.from_pretrained(
        VJEPA2_MODEL_ID,
        checkpoint_revision=VJEPA2_MODEL_REVISION,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    if args.maximum_frames % encoder.tubelet_size:
        raise ValueError("maximum_frames must contain complete V-JEPA2 tubelets")
    requested_output_root = args.output_root.expanduser()
    if requested_output_root.is_symlink():
        raise RuntimeError("V-JEPA2 cache output root cannot be a symlink")
    output_root = requested_output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    entries: list[dict[str, object]] = []
    frame_histogram: Counter[int] = Counter()
    artifact_bytes = 0
    encoded_artifacts = 0
    reused_artifacts = 0
    sample_keys = tuple(sorted(dataset.sample_keys))
    expected_partial = _manifest(
        dataset_tree_sha256=manifest.tree_sha256,
        encoder=encoder,
        maximum_frames=args.maximum_frames,
        expected_entries=len(sample_keys),
        entries=[],
        complete=False,
    )
    trusted_artifacts = _trusted_partial_artifacts(
        output_root,
        expected_manifest=expected_partial,
        sample_keys=sample_keys,
    )
    for sample_index, sample_key in enumerate(sample_keys):
        prefix = dataset.evidence_prefix_by_key(
            sample_key,
            maximum_source_frames=args.maximum_frames,
        )
        sensor_entries = []
        for sensor_key, modality in VJEPA2_CONTEXT_SENSORS:
            clip = build_calvin_causal_video_clip(
                prefix,
                sensor_key=sensor_key,
                maximum_frames=args.maximum_frames,
                tubelet_size=encoder.tubelet_size,
            )
            if clip is None:
                frame_histogram[0] += 1
                sensor_entries.append(
                    {
                        "artifact_path": None,
                        "artifact_sha256": None,
                        "modality": modality,
                        "sensor_key": sensor_key,
                        "source_frame_sha256": [],
                        "token_count": 0,
                    }
                )
                continue
            frame_count = len(clip.images)
            frame_histogram[frame_count] += 1
            patches_per_frame = (encoder.image_size // encoder.patch_size) ** 2
            token_count = frame_count // encoder.tubelet_size * patches_per_frame
            digest = _clip_digest(
                encoder.encoder_contract,
                sensor_key,
                clip.source_frame_sha256,
            )
            sample_digest = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()
            relative = f"entries/{sample_digest}/{modality}-{digest}.tokens.npy"
            resume_key = (
                sample_key,
                sensor_key,
                modality,
                clip.source_frame_sha256,
                token_count,
                relative,
            )
            trusted_sha = trusted_artifacts.get(resume_key)
            tokens = None
            if trusted_sha is None:
                encoded_artifacts += 1
                evidence = encoder.encode_clip(
                    clip.images,
                    clip.frame_timestamps_s,
                    require_pretrained_frame_count=False,
                )
                if (
                    evidence.current_measurement_valid is None
                    or evidence.current_measurement_valid.any()
                ):
                    raise RuntimeError("V-JEPA2 cache encoder emitted a posterior measurement")
                tokens = evidence.tokens
            else:
                reused_artifacts += 1
            artifact_path, artifact_sha, size_bytes = _token_artifact(
                output_root,
                sample_key=sample_key,
                modality=modality,
                clip_digest=digest,
                tokens=tokens,
                expected_shape=(token_count, encoder.hidden_size),
                trusted_sha256=trusted_sha,
            )
            artifact_bytes += size_bytes
            sensor_entries.append(
                {
                    "artifact_path": artifact_path,
                    "artifact_sha256": artifact_sha,
                    "modality": modality,
                    "sensor_key": sensor_key,
                    "source_frame_sha256": list(clip.source_frame_sha256),
                    "token_count": token_count,
                }
            )
        entries.append({"sample_key": sample_key, "sensors": sensor_entries})
        if (sample_index + 1) % args.checkpoint_interval == 0:
            partial = _manifest(
                dataset_tree_sha256=manifest.tree_sha256,
                encoder=encoder,
                maximum_frames=args.maximum_frames,
                expected_entries=len(sample_keys),
                entries=entries,
                complete=False,
            )
            _write_atomic(output_root / "manifest.partial.json", _canonical_json(partial))
            print(f"cached {sample_index + 1}/{len(sample_keys)} samples", flush=True)

    final_manifest = _manifest(
        dataset_tree_sha256=manifest.tree_sha256,
        encoder=encoder,
        maximum_frames=args.maximum_frames,
        expected_entries=len(sample_keys),
        entries=entries,
        complete=True,
    )
    final_partial = dict(final_manifest)
    final_partial["complete"] = False
    _write_atomic(output_root / "manifest.partial.json", _canonical_json(final_partial))
    manifest_bytes = _canonical_json(final_manifest)
    _write_atomic(output_root / "manifest.json", manifest_bytes)
    report = {
        "artifact_bytes": artifact_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "encoded_artifacts": encoded_artifacts,
        "frame_count_histogram_across_sensors": {
            str(frame_count): count for frame_count, count in sorted(frame_histogram.items())
        },
        "manifest_sha256": _sha256(manifest_bytes),
        "output_root": str(output_root),
        "reused_artifacts": reused_artifacts,
        "samples": len(sample_keys),
        "schema": VJEPA2_CACHE_SCHEMA,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
