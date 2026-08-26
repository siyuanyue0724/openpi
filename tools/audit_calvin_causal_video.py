#!/usr/bin/env python3
"""Audit target-free CALVIN causal clips and render a task-labelled sheet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from picf_next.data.calvin import (
    CALVIN_DEBUG_DATASET_ID,
    CALVIN_DEBUG_REVISION,
    CalvinDatasetIndex,
    CalvinPICFEvidenceFrame,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.causal_video import CausalVideoClip, build_calvin_causal_video_clip
from picf_next.data.dataset_manifest import load_dataset_file_manifest

SENSOR_KEYS = (
    "observation.images.rgb_static",
    "observation.images.rgb_gripper",
)


def _clips_for_prefix(
    prefix: Sequence[CalvinPICFEvidenceFrame],
    *,
    maximum_frames: int,
    tubelet_size: int,
) -> tuple[CausalVideoClip | None, ...]:
    return tuple(
        build_calvin_causal_video_clip(
            prefix,
            sensor_key=sensor_key,
            maximum_frames=maximum_frames,
            tubelet_size=tubelet_size,
        )
        for sensor_key in SENSOR_KEYS
    )


def audit_stateful_dataset(
    dataset: CalvinStatefulTransitionDataset,
    *,
    maximum_frames: int,
    tubelet_size: int,
) -> tuple[dict, tuple[tuple[str, str, tuple[CausalVideoClip, ...]], ...]]:
    """Audit every transition and retain one full causal clip per task segment."""

    histograms = {sensor_key: Counter() for sensor_key in SENSOR_KEYS}
    tubelet_ready = Counter({sensor_key: 0 for sensor_key in SENSOR_KEYS})
    full_window = Counter({sensor_key: 0 for sensor_key in SENSOR_KEYS})
    selected: list[tuple[str, str, tuple[CausalVideoClip, ...]]] = []

    for episode in dataset.episode_manifest:
        candidates: list[tuple[str, str, tuple[CausalVideoClip, ...]]] = []
        segment = dataset.index.segments[episode.segment_index]
        for sample_key in episode.sample_keys:
            prefix = dataset.evidence_prefix_by_key(
                sample_key,
                maximum_source_frames=maximum_frames,
            )
            clips = _clips_for_prefix(
                prefix,
                maximum_frames=maximum_frames,
                tubelet_size=tubelet_size,
            )
            lengths = []
            for sensor_key, clip in zip(SENSOR_KEYS, clips, strict=True):
                length = 0 if clip is None else len(clip.images)
                histograms[sensor_key][length] += 1
                tubelet_ready[sensor_key] += int(clip is not None)
                full_window[sensor_key] += int(length == maximum_frames)
                lengths.append(length)
            if len(set(lengths)) != 1:
                raise RuntimeError("CALVIN camera streams produced different temporal lengths")
            if lengths[0] == maximum_frames:
                candidates.append(
                    (
                        segment.task_key,
                        f"{segment.instruction} | {sample_key}",
                        tuple(clip for clip in clips if clip is not None),
                    )
                )
        if candidates:
            phase_indices = tuple(dict.fromkeys((0, len(candidates) // 2, len(candidates) - 1)))
            phase_names = ("early", "middle", "late")
            for phase_name, candidate_index in zip(phase_names, phase_indices, strict=False):
                task_key, label, clips = candidates[candidate_index]
                selected.append((task_key, f"[{phase_name}] {label}", clips))

    transitions = len(dataset)
    sensor_reports = {}
    for sensor_key in SENSOR_KEYS:
        sensor_reports[sensor_key] = {
            "frame_count_histogram": {
                str(count): frequency for count, frequency in sorted(histograms[sensor_key].items())
            },
            "complete_tubelet_clips": tubelet_ready[sensor_key],
            "complete_tubelet_fraction": tubelet_ready[sensor_key] / transitions,
            "full_window_clips": full_window[sensor_key],
            "full_window_fraction": full_window[sensor_key] / transitions,
        }
    return (
        {
            "format": "picf-next.calvin-causal-video-audit/v1",
            "transitions": transitions,
            "language_segments": len(dataset.episode_manifest),
            "maximum_frames": maximum_frames,
            "tubelet_size": tubelet_size,
            "padding_or_repeated_frames": False,
            "runtime_target_fields": [],
            "sensors": sensor_reports,
            "visual_samples": len(selected),
        },
        tuple(selected),
    )


def render_contact_sheet(
    samples: Sequence[tuple[str, str, tuple[CausalVideoClip, ...]]],
    output: Path,
) -> None:
    """Render all retained clip frames without cropping their visible content."""

    if not samples:
        raise ValueError("contact sheet requires at least one full causal clip")
    thumb = 112
    label_width = 410
    header_height = 24
    gap = 6
    frame_count = len(samples[0][2][0].images)
    row_height = thumb + header_height + gap * 2
    width = label_width + 2 * frame_count * (thumb + gap) + gap
    height = len(samples) * row_height + gap
    sheet = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    for row, (task_key, label, clips) in enumerate(samples):
        if len(clips) != len(SENSOR_KEYS) or any(len(clip.images) != frame_count for clip in clips):
            raise ValueError("contact-sheet samples must have one aligned clip per camera")
        y = gap + row * row_height
        draw.text((gap, y), task_key, fill=(10, 10, 10))
        draw.text((gap, y + 16), label[:68], fill=(45, 45, 45))
        for camera_index, clip in enumerate(clips):
            for frame_index, array in enumerate(clip.images):
                column = camera_index * frame_count + frame_index
                x = label_width + column * (thumb + gap)
                image = Image.fromarray(array)
                contained = ImageOps.contain(
                    image, (thumb, thumb), method=Image.Resampling.BILINEAR
                )
                tile = Image.new("RGB", (thumb, thumb), color=(230, 230, 230))
                tile.paste(
                    contained,
                    ((thumb - contained.width) // 2, (thumb - contained.height) // 2),
                )
                sheet.paste(tile, (x, y + header_height))
                current = frame_index == frame_count - 1
                color = (190, 35, 35) if current else (70, 70, 70)
                draw.rectangle(
                    (x, y + header_height, x + thumb - 1, y + header_height + thumb - 1),
                    outline=color,
                    width=2 if current else 1,
                )
                camera = "static" if camera_index == 0 else "gripper"
                draw.text(
                    (x + 2, y + 2),
                    f"{camera} t-{frame_count - 1 - frame_index}",
                    fill=color,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split", default="training", choices=("training", "validation"))
    parser.add_argument("--dataset-id", default=CALVIN_DEBUG_DATASET_ID)
    parser.add_argument("--dataset-revision", default=CALVIN_DEBUG_REVISION)
    parser.add_argument("--maximum-frames", default=4, type=int)
    parser.add_argument("--tubelet-size", default=2, type=int)
    parser.add_argument("--action-horizon", default=30, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contact-sheet", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        (args.dataset_root / args.split).resolve(),
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=args.action_horizon)
    report, samples = audit_stateful_dataset(
        dataset,
        maximum_frames=args.maximum_frames,
        tubelet_size=args.tubelet_size,
    )
    render_contact_sheet(samples, args.contact_sheet.resolve())
    report["dataset_tree_sha256"] = manifest.tree_sha256
    report["contact_sheet"] = str(args.contact_sheet.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
