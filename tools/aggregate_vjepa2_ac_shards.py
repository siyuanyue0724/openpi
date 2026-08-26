#!/usr/bin/env python3
"""Aggregate an exact, globally partitioned four-GPU V-JEPA2-AC gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.probe_calvin_vjepa2_ac import summarize_control_reports


def aggregate_shard_reports(reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not reports:
        raise ValueError("V-JEPA2-AC shard aggregation requires reports")
    selections = [report["selection"] for report in reports]
    if not all(isinstance(selection, Mapping) for selection in selections):
        raise ValueError("V-JEPA2-AC report omitted shard selection metadata")

    first = reports[0]
    first_selection = selections[0]
    assert isinstance(first_selection, Mapping)
    shard_count = int(first_selection["shard_count"])
    global_ends = tuple(int(value) for value in first_selection["global_clip_ends"])
    global_count = int(first_selection["global_clip_count"])
    if shard_count <= 1 or len(reports) != shard_count:
        raise ValueError("V-JEPA2-AC aggregate requires exactly one report per shard")
    if len(global_ends) != global_count or len(set(global_ends)) != global_count:
        raise ValueError("V-JEPA2-AC global clip selection is not unique")

    invariant_fields = (
        "camera",
        "conditioning_semantics",
        "future_targets_are_loss_only",
        "model_parameters_frozen",
    )
    combined: dict[int, Mapping[str, object]] = {}
    shard_summaries: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    for report, selection in zip(reports, selections, strict=True):
        assert isinstance(selection, Mapping)
        shard_index = int(selection["shard_index"])
        if shard_index in seen_indices or not 0 <= shard_index < shard_count:
            raise ValueError("V-JEPA2-AC shard index is duplicated or invalid")
        seen_indices.add(shard_index)
        if int(selection["shard_count"]) != shard_count:
            raise ValueError("V-JEPA2-AC shard counts disagree")
        if tuple(int(value) for value in selection["global_clip_ends"]) != global_ends:
            raise ValueError("V-JEPA2-AC shards used different global clip selections")
        if int(selection["control_seed"]) != int(first_selection["control_seed"]):
            raise ValueError("V-JEPA2-AC shards used different control seeds")
        expected_ends = global_ends[shard_index::shard_count]
        expected_positions = tuple(range(shard_index, global_count, shard_count))
        shard_positions = tuple(int(value) for value in selection["shard_global_positions"])
        if shard_positions != expected_positions:
            raise ValueError("V-JEPA2-AC shard positions do not match the frozen partition")
        shard_ends = tuple(int(value) for value in selection["shard_clip_ends"])
        if shard_ends != expected_ends:
            raise ValueError("V-JEPA2-AC shard does not match the frozen partition")
        for field in invariant_fields:
            if report[field] != first[field]:
                raise ValueError(f"V-JEPA2-AC shard changed invariant field {field}")
        if report["checkpoint"] != first["checkpoint"] or report["dataset"] != first["dataset"]:
            raise ValueError("V-JEPA2-AC shard identity differs")

        clips = report["clips"]
        if not isinstance(clips, Sequence) or len(clips) != len(expected_ends):
            raise ValueError("V-JEPA2-AC shard clip count differs from its partition")
        actual_ends: list[int] = []
        for expected_position, clip in zip(expected_positions, clips, strict=True):
            if not isinstance(clip, Mapping):
                raise ValueError("V-JEPA2-AC clip report is malformed")
            if int(clip["global_clip_position"]) != expected_position:
                raise ValueError("V-JEPA2-AC clip position differs from its shard partition")
            expected_control_seed = int(first_selection["control_seed"]) + expected_position
            if int(clip["control_seed"]) != expected_control_seed:
                raise ValueError("V-JEPA2-AC clip control seed differs from the global protocol")
            frame_indices = clip["frame_indices"]
            if not isinstance(frame_indices, Sequence) or not frame_indices:
                raise ValueError("V-JEPA2-AC clip omitted frame indices")
            end = int(frame_indices[-1])
            actual_ends.append(end)
            if end in combined:
                raise ValueError("V-JEPA2-AC shards duplicated a clip")
            combined[end] = clip
        if tuple(actual_ends) != expected_ends:
            raise ValueError("V-JEPA2-AC report order differs from its shard selection")
        shard_summaries.append(
            {
                "elapsed_seconds": float(report["elapsed_seconds"]),
                "shard_clip_count": len(clips),
                "shard_index": shard_index,
            }
        )

    if seen_indices != set(range(shard_count)) or set(combined) != set(global_ends):
        raise ValueError("V-JEPA2-AC shard coverage is incomplete")
    ordered_clips = [combined[end] for end in global_ends]
    episode_indices = [int(clip["episode_index"]) for clip in ordered_clips]
    if len(set(episode_indices)) != global_count:
        raise ValueError("V-JEPA2-AC global gate reused a raw episode")
    numerical = [clip["controls"] for clip in ordered_clips]
    if not all(isinstance(control, Mapping) for control in numerical):
        raise ValueError("V-JEPA2-AC clip omitted control metrics")
    aggregate = summarize_control_reports(numerical)
    passed = aggregate["causal_signal_pass"] is True
    return {
        "schema": "picf-next.vjepa2-ac-calvin-four-gpu-gate/v1",
        "checkpoint": first["checkpoint"],
        "dataset": first["dataset"],
        "camera": first["camera"],
        "control_seed": int(first_selection["control_seed"]),
        "global_clip_count": global_count,
        "global_episode_count": len(set(episode_indices)),
        "shard_count": shard_count,
        "shards": sorted(shard_summaries, key=lambda value: int(value["shard_index"])),
        "wall_clock_upper_bound_seconds": max(
            float(summary["elapsed_seconds"]) for summary in shard_summaries
        ),
        "aggregate": aggregate,
        "authorizes_arm_c": passed,
        "authorizes_policy_training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_paths = tuple(path.resolve() for path in args.report)
    reports = tuple(json.loads(path.read_text()) for path in report_paths)
    aggregate = aggregate_shard_reports(reports)
    aggregate["source_reports"] = [str(path) for path in report_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    raise SystemExit(0 if aggregate["authorizes_arm_c"] else 3)


if __name__ == "__main__":
    main()
