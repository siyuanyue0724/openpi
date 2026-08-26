#!/usr/bin/env python3
"""Run official V-JEPA2 on one real duplicate-free CALVIN causal clip."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.causal_video import build_calvin_causal_video_clip
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.data.vjepa2_cache import VJEPA2_CONTEXT_SENSORS
from picf_next.encoders.vjepa2 import (
    VJEPA2_MODEL_ID,
    VJEPA2_MODEL_REVISION,
    Vjepa2DenseEncoder,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-id", default=VJEPA2_MODEL_ID)
    parser.add_argument("--model-revision", default=VJEPA2_MODEL_REVISION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--maximum-frames", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    encoder = Vjepa2DenseEncoder.from_pretrained(
        args.model_id,
        checkpoint_revision=args.model_revision,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    split_root = (args.dataset_root / args.split).resolve()
    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    selected_key = next(
        sample_key
        for sample_key in dataset.sample_keys
        if len(
            dataset.evidence_prefix_by_key(
                sample_key,
                maximum_source_frames=args.maximum_frames,
            )
        )
        == args.maximum_frames
    )
    prefix = dataset.evidence_prefix_by_key(
        selected_key,
        maximum_source_frames=args.maximum_frames,
    )

    started = time.perf_counter()
    sensor_reports = {}
    for sensor_key, modality in VJEPA2_CONTEXT_SENSORS:
        clip = build_calvin_causal_video_clip(
            prefix,
            sensor_key=sensor_key,
            maximum_frames=args.maximum_frames,
            tubelet_size=encoder.tubelet_size,
        )
        if clip is None or len(clip.images) != args.maximum_frames:
            raise RuntimeError("selected CALVIN sample did not form the requested causal clip")
        evidence = encoder.encode_clip(
            clip.images,
            clip.frame_timestamps_s,
            require_pretrained_frame_count=False,
        )
        sensor_reports[modality] = {
            "all_tokens_finite": bool(np.isfinite(evidence.tokens).all()),
            "current_measurement_tokens": int(evidence.effective_current_measurement_valid.sum()),
            "geometry_shape": list(evidence.geometry.shape),
            "source_frame_sha256": list(clip.source_frame_sha256),
            "timestamp_shape": list(evidence.timestamps.shape),
            "token_dtype": str(evidence.tokens.dtype),
            "token_shape": list(evidence.tokens.shape),
        }
    elapsed = time.perf_counter() - started
    report = {
        "model_id": encoder.model_id,
        "checkpoint_revision": encoder.checkpoint_revision,
        "encoder_contract": encoder.encoder_contract,
        "dataset_split": args.split,
        "sample_key": selected_key,
        "input_frames": args.maximum_frames,
        "pretrained_frames_per_clip": encoder.frames_per_clip,
        "sensors": sensor_reports,
        "elapsed_seconds": elapsed,
        "pooling_or_selection": False,
        "padding_or_repeated_frames": False,
        "posterior_measurement": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
