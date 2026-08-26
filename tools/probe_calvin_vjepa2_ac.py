#!/usr/bin/env python3
"""Run the frozen exact V-JEPA2-AC causal donor gate on real CALVIN clips."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_vjepa2_ac import (
    VJEPA2_AC_FRAME_COUNT,
    calvin_vjepa2_ac_frame_indices,
    load_calvin_vjepa2_ac_clip,
    vjepa2_ac_calvin_stride,
)
from picf_next.data.dataset_manifest import DatasetFileManifest, load_dataset_file_manifest
from picf_next.encoders.vjepa2_ac import Vjepa2AcDonor

MINIMUM_CAUSAL_GATE_CLIPS = 16
CAUSAL_GATE_FAMILY_ALPHA = 0.05
CAUSAL_GATE_COMPARISON_COUNT = 6


def _one_sided_sign_test_pvalue(margins: np.ndarray) -> float:
    nonzero = np.asarray(margins, dtype=np.float64)
    nonzero = nonzero[nonzero != 0.0]
    if nonzero.size == 0:
        return 1.0
    wins = int(np.count_nonzero(nonzero > 0.0))
    numerator = sum(math.comb(nonzero.size, count) for count in range(wins, nonzero.size + 1))
    return float(numerator / (2**nonzero.size))


def summarize_control_reports(
    reports: Sequence[Mapping[str, Mapping[str, float]]],
) -> dict[str, object]:
    if not reports:
        raise ValueError("V-JEPA2-AC summary requires at least one clip report")
    control_names = ("actual", "zero", "reversed", "shuffled")
    metric_names = ("teacher_forced_l1", "autoregressive_l1")
    values: dict[str, dict[str, list[float]]] = {
        control: {metric: [] for metric in metric_names} for control in control_names
    }
    for clip_index, report in enumerate(reports):
        if set(report) != set(control_names):
            raise ValueError(f"clip {clip_index} has changed V-JEPA2-AC controls")
        for control in control_names:
            if set(report[control]) != set(metric_names):
                raise ValueError(f"clip {clip_index} has changed V-JEPA2-AC metrics")
            for metric in metric_names:
                value = float(report[control][metric])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError("V-JEPA2-AC report contains an invalid loss")
                values[control][metric].append(value)

    means = {
        control: {
            metric: float(np.mean(values[control][metric])) for metric in metric_names
        }
        for control in control_names
    }
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    all_pass = len(reports) >= MINIMUM_CAUSAL_GATE_CLIPS
    per_comparison_alpha = CAUSAL_GATE_FAMILY_ALPHA / CAUSAL_GATE_COMPARISON_COUNT
    for control in control_names[1:]:
        comparisons[control] = {}
        for metric in metric_names:
            actual = np.asarray(values["actual"][metric], dtype=np.float64)
            counterfactual = np.asarray(values[control][metric], dtype=np.float64)
            margins = counterfactual - actual
            mean_margin = float(np.mean(margins))
            win_fraction = float(np.mean(margins > 0.0))
            pvalue = _one_sided_sign_test_pvalue(margins)
            passed = mean_margin > 0.0 and pvalue <= per_comparison_alpha
            comparisons[control][metric] = {
                "counterfactual_minus_actual_mean": mean_margin,
                "actual_win_fraction": win_fraction,
                "one_sided_sign_test_pvalue": pvalue,
                "bonferroni_alpha": per_comparison_alpha,
                "pass": passed,
            }
            all_pass = all_pass and passed
    return {
        "clip_count": len(reports),
        "means": means,
        "comparisons": comparisons,
        "causal_signal_pass": all_pass,
        "minimum_clip_count": MINIMUM_CAUSAL_GATE_CLIPS,
        "acceptance_rule": (
            "at least 16 distinct episodes; actual mean error is below every "
            "zero/reversed/shuffled control; and every paired one-sided sign test passes "
            "Bonferroni alpha 0.05/6, for both teacher-forced and AR losses"
        ),
    }


def _select_clip_ends(
    index: CalvinDatasetIndex,
    *,
    clip_count: int,
    seed: int,
) -> tuple[int, ...]:
    if clip_count <= 0:
        raise ValueError("clip_count must be positive")
    stride = vjepa2_ac_calvin_stride(control_hz=index.control_hz)
    minimum_span = (VJEPA2_AC_FRAME_COUNT - 1) * stride
    eligible = tuple(episode for episode in index.episodes if episode.length > minimum_span)
    if len(eligible) < clip_count:
        raise RuntimeError("CALVIN contains too few distinct source episodes for the donor gate")
    generator = np.random.default_rng(seed)
    selected_positions = generator.permutation(len(eligible))[:clip_count]
    ends: list[int] = []
    for position in selected_positions:
        episode = eligible[int(position)]
        minimum_end = episode.start + minimum_span
        end = int(generator.integers(minimum_end, episode.end + 1))
        calvin_vjepa2_ac_frame_indices(
            episode,
            end_global_index=end,
            control_hz=index.control_hz,
        )
        ends.append(end)
    return tuple(ends)


def _load_gate_index(
    dataset_split: Path,
    manifest: DatasetFileManifest,
) -> CalvinDatasetIndex:
    # Selected arrays are verified against the immutable manifest when clips are read.
    return CalvinDatasetIndex.load(
        dataset_split.resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--camera", choices=("rgb_static", "rgb_gripper"), default="rgb_static")
    parser.add_argument("--clip-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = _load_gate_index(args.dataset_split, manifest)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("V-JEPA2-AC shard coordinates are invalid")
    global_clip_ends = _select_clip_ends(
        index,
        clip_count=args.clip_count,
        seed=args.seed,
    )
    global_positions = tuple(range(len(global_clip_ends)))
    shard_positions = global_positions[args.shard_index :: args.shard_count]
    clip_ends = tuple(global_clip_ends[position] for position in shard_positions)
    if not clip_ends:
        raise ValueError("V-JEPA2-AC shard contains no clips")
    clips = tuple(
        load_calvin_vjepa2_ac_clip(
            index,
            end_global_index=end,
            camera_key=args.camera,
        )
        for end in clip_ends
    )
    donor = Vjepa2AcDonor.from_checkpoint(
        args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        device=args.device,
    )

    started = time.perf_counter()
    clip_reports: list[dict[str, object]] = []
    numerical_reports: list[dict[str, dict[str, float]]] = []
    for global_position, clip in zip(shard_positions, clips, strict=True):
        clip_control_seed = args.seed + global_position
        controls = donor.evaluate_controls(clip, seed=clip_control_seed)
        numerical_reports.append(controls)
        clip_reports.append(
            {
                "control_seed": clip_control_seed,
                "episode_index": clip.episode_index,
                "frame_indices": list(clip.frame_indices),
                "frame_timestamps_s": clip.frame_timestamps_s.tolist(),
                "global_clip_position": global_position,
                "controls": controls,
            }
        )
    elapsed = time.perf_counter() - started
    report = {
        "schema": "picf-next.vjepa2-ac-calvin-causal-gate/v1",
        "checkpoint": {
            "bytes": donor.checkpoint_path.stat().st_size,
            "path": str(donor.checkpoint_path),
            "sha256": donor.checkpoint_sha256,
        },
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "manifest_tree_sha256": manifest.tree_sha256,
            "split": str(args.dataset_split.resolve()),
        },
        "camera": args.camera,
        "selection": {
            "control_seed": args.seed,
            "global_clip_count": len(global_clip_ends),
            "global_clip_ends": list(global_clip_ends),
            "shard_global_positions": list(shard_positions),
            "shard_clip_ends": list(clip_ends),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
        },
        "conditioning_semantics": "realized_pose_difference_not_policy_command",
        "future_targets_are_loss_only": True,
        "model_parameters_frozen": True,
        "elapsed_seconds": elapsed,
        "seconds_per_clip": elapsed / len(clips),
        "clips": clip_reports,
        "aggregate": summarize_control_reports(numerical_reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
