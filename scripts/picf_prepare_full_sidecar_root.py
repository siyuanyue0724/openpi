#!/usr/bin/env python3
"""Prepare and audit a CALVIN MVTrack sidecar root for long PICF runs.

The script is intentionally metadata-only: it does not run a model and does not
rewrite frame payloads.  It scans one sidecar split directory, maps covered
episode ids back to CALVIN language segments, writes `calvin_segment_indices.txt`,
and records a compact manifest that the long-run launcher can trust.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


_EPISODE_RE = re.compile(r"episode_(\d{7})\.npz$")

_PROPOSAL_REQUIRED = (
    "proposal_centers_xy",
    "proposal_boxes_xyxy",
    "proposal_objectness",
    "proposal_view_ids",
    "proposal_source_ids",
)
_PROPOSAL_MASK = (
    "proposal_mask_xy",
    "proposal_mask_weights",
    "proposal_mask_offsets",
)
_TRACKLET_REQUIRED = (
    "tracklet_xy",
    "tracklet_velocity",
    "tracklet_visibility",
    "tracklet_confidence",
    "tracklet_ids",
    "tracklet_view_ids",
    "tracklet_age",
)


def _load_intervals(calvin_root: Path, split: str) -> list[tuple[int, int]]:
    ann_path = calvin_root / split / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    return [tuple(int(v) for v in interval) for interval in ann["info"]["indx"]]


def _episode_id(path: Path) -> int | None:
    match = _EPISODE_RE.match(path.name)
    return None if match is None else int(match.group(1))


def _segment_coverage(step_ids: list[int], intervals: list[tuple[int, int]]) -> dict[int, int]:
    coverage: dict[int, int] = {}
    if not step_ids:
        return coverage
    # CALVIN language intervals are not guaranteed to be sorted by episode id.
    # Use binary searches per interval instead of a single monotonic cursor, or
    # most covered segments are silently skipped when `info["indx"]` is shuffled.
    ordered_steps = np.asarray(sorted(step_ids), dtype=np.int64)
    for segment_id, (start, end) in enumerate(intervals):
        left = int(np.searchsorted(ordered_steps, int(start), side="left"))
        right = int(np.searchsorted(ordered_steps, int(end), side="right"))
        count = right - left
        if count:
            coverage[int(segment_id)] = int(count)
    return coverage


def _sample_paths(paths: list[Path], sample_limit: int) -> list[Path]:
    """Return a deterministic root-wide sample instead of only the first files."""

    if sample_limit <= 0 or not paths:
        return []
    if len(paths) <= sample_limit:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, num=int(sample_limit), dtype=np.int64)
    # `linspace(..., dtype=int)` can repeat indices for very small ranges.  Keep
    # ordering deterministic and unique so sampled_files is truthful.
    return [paths[int(i)] for i in np.unique(indices)]


def _proposal_count(path: Path) -> int:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "proposal_centers_xy" in data.files:
                return int(np.asarray(data["proposal_centers_xy"]).reshape(-1, 2).shape[0])
            if "proposal_boxes_xyxy" in data.files:
                return int(np.asarray(data["proposal_boxes_xyxy"]).reshape(-1, 4).shape[0])
    except Exception:
        return 0
    return 0


def _has_mask(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            if not all(key in data.files for key in _PROPOSAL_MASK):
                return False
            if "proposal_mask_xy" not in data.files:
                return False
            return int(np.asarray(data["proposal_mask_xy"]).reshape(-1, 2).shape[0]) > 0
    except Exception:
        return False


def _reachability_fraction(
    sample_paths: list[Path],
    path_by_step: dict[int, Path],
    *,
    max_gap: int,
    predicate,
) -> float:
    if not sample_paths:
        return 0.0
    ok = 0
    for path in sample_paths:
        step_id = _episode_id(path)
        if step_id is None:
            continue
        if predicate(path):
            ok += 1
            continue
        for gap in range(1, max(int(max_gap), 0) + 1):
            left = path_by_step.get(int(step_id) - gap)
            right = path_by_step.get(int(step_id) + gap)
            if (left is not None and predicate(left)) or (right is not None and predicate(right)):
                ok += 1
                break
    return float(ok) / float(len(sample_paths))


def _sample_key_stats(paths: list[Path], sample_limit: int) -> dict[str, object]:
    sample_paths = _sample_paths(paths, max(int(sample_limit), 0))
    key_counts: dict[str, int] = {}
    bad_files: list[str] = []
    first_shapes: dict[str, list[int]] = {}
    for path in sample_paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                for key in keys:
                    key_counts[key] = key_counts.get(key, 0) + 1
                    if key not in first_shapes:
                        first_shapes[key] = list(data[key].shape)
        except Exception:
            bad_files.append(str(path))
    def present_fraction(keys: tuple[str, ...]) -> float:
        if not sample_paths:
            return 0.0
        ok = 0
        for path in sample_paths:
            try:
                with np.load(path, allow_pickle=False) as data:
                    if all(key in data.files for key in keys):
                        ok += 1
            except Exception:
                pass
        return float(ok) / float(len(sample_paths))
    return {
        "sampled_files": len(sample_paths),
        "sample_strategy": "deterministic_evenly_spaced",
        "bad_files": bad_files[:20],
        "keys": sorted(key_counts),
        "first_shapes": first_shapes,
        "proposal_required_present_fraction": present_fraction(_PROPOSAL_REQUIRED),
        "proposal_mask_present_fraction": present_fraction(_PROPOSAL_MASK),
        "tracklet_required_present_fraction": present_fraction(_TRACKLET_REQUIRED),
    }


def _sample_sparse_proposal_stats(paths: list[Path], sample_limit: int, proposal_nearest_max_gap: int) -> dict[str, float]:
    sample_paths = _sample_paths(paths, max(int(sample_limit), 0))
    if not sample_paths:
        return {
            "proposal_nonempty_fraction": 0.0,
            "proposal_reachable_fraction": 0.0,
            "proposal_mask_nonempty_fraction": 0.0,
            "proposal_mask_reachable_fraction": 0.0,
        }
    path_by_step = {
        int(step_id): path
        for path in paths
        for step_id in [_episode_id(path)]
        if step_id is not None
    }
    proposal_nonempty = sum(1 for path in sample_paths if _proposal_count(path) > 0) / float(len(sample_paths))
    mask_nonempty = sum(1 for path in sample_paths if _has_mask(path)) / float(len(sample_paths))
    return {
        "proposal_nonempty_fraction": float(proposal_nonempty),
        "proposal_reachable_fraction": _reachability_fraction(
            sample_paths,
            path_by_step,
            max_gap=int(proposal_nearest_max_gap),
            predicate=lambda path: _proposal_count(path) > 0,
        ),
        "proposal_mask_nonempty_fraction": float(mask_nonempty),
        "proposal_mask_reachable_fraction": _reachability_fraction(
            sample_paths,
            path_by_step,
            max_gap=int(proposal_nearest_max_gap),
            predicate=_has_mask,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--min-frames-per-segment", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=256)
    parser.add_argument("--require-proposal", action="store_true")
    parser.add_argument("--require-mask", action="store_true")
    parser.add_argument("--require-tracklet", action="store_true")
    parser.add_argument("--proposal-nearest-max-gap", type=int, default=0)
    parser.add_argument("--min-proposal-nonempty-fraction", type=float, default=0.0)
    parser.add_argument("--min-proposal-reachable-fraction", type=float, default=0.0)
    parser.add_argument("--min-mask-reachable-fraction", type=float, default=0.0)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    calvin_root = Path(args.calvin_root)
    sidecar_root = Path(args.sidecar_root)
    split_dir = sidecar_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing sidecar split directory: {split_dir}")

    paths = sorted(path for path in split_dir.iterdir() if _episode_id(path) is not None)
    step_ids = [int(_episode_id(path)) for path in paths if _episode_id(path) is not None]
    intervals = _load_intervals(calvin_root, args.split)
    coverage = _segment_coverage(step_ids, intervals)
    selected_segments = [
        segment_id for segment_id, count in sorted(coverage.items()) if count >= int(args.min_frames_per_segment)
    ]
    stats = _sample_key_stats(paths, int(args.sample_limit))
    sparse_stats = _sample_sparse_proposal_stats(
        paths,
        int(args.sample_limit),
        int(args.proposal_nearest_max_gap),
    )
    ok = True
    failures: list[str] = []
    if not selected_segments:
        ok = False
        failures.append("no covered CALVIN language segments")
    if stats["bad_files"]:
        ok = False
        failures.append("bad sidecar files in sampled audit set")
    if bool(args.require_proposal) and float(stats["proposal_required_present_fraction"]) < 0.999:
        ok = False
        failures.append("proposal required keys missing in sampled files")
    if bool(args.require_mask) and float(stats["proposal_mask_present_fraction"]) < 0.999:
        ok = False
        failures.append("proposal mask keys missing in sampled files")
    if bool(args.require_tracklet) and float(stats["tracklet_required_present_fraction"]) < 0.999:
        ok = False
        failures.append("tracklet required keys missing in sampled files")
    if float(sparse_stats["proposal_nonempty_fraction"]) < float(args.min_proposal_nonempty_fraction):
        ok = False
        failures.append("proposal non-empty sampled fraction below threshold")
    if float(sparse_stats["proposal_reachable_fraction"]) < float(args.min_proposal_reachable_fraction):
        ok = False
        failures.append("proposal nearest-reachable sampled fraction below threshold")
    if float(sparse_stats["proposal_mask_reachable_fraction"]) < float(args.min_mask_reachable_fraction):
        ok = False
        failures.append("proposal mask nearest-reachable sampled fraction below threshold")

    manifest = {
        "sidecar_root": str(sidecar_root),
        "split": args.split,
        "npz_files": int(len(paths)),
        "covered_segments": int(len(selected_segments)),
        "min_frames_per_segment": int(args.min_frames_per_segment),
        "coverage_min": int(min((coverage[s] for s in selected_segments), default=0)),
        "coverage_max": int(max((coverage[s] for s in selected_segments), default=0)),
        "coverage_mean": float(np.mean([coverage[s] for s in selected_segments])) if selected_segments else 0.0,
        "proposal_required_present_fraction": stats["proposal_required_present_fraction"],
        "proposal_mask_present_fraction": stats["proposal_mask_present_fraction"],
        "proposal_nonempty_fraction": sparse_stats["proposal_nonempty_fraction"],
        "proposal_nearest_max_gap": int(args.proposal_nearest_max_gap),
        "proposal_reachable_fraction": sparse_stats["proposal_reachable_fraction"],
        "proposal_mask_nonempty_fraction": sparse_stats["proposal_mask_nonempty_fraction"],
        "proposal_mask_reachable_fraction": sparse_stats["proposal_mask_reachable_fraction"],
        "tracklet_required_present_fraction": stats["tracklet_required_present_fraction"],
        "sampled_files": stats["sampled_files"],
        "sample_strategy": stats["sample_strategy"],
        "sample_keys": stats["keys"],
        "first_shapes": stats["first_shapes"],
        "ok": bool(ok),
        "failures": failures,
    }
    if bool(args.write_manifest):
        sidecar_root.mkdir(parents=True, exist_ok=True)
        (sidecar_root / "calvin_segment_indices.txt").write_text(
            ",".join(str(segment_id) for segment_id in selected_segments) + ("\n" if selected_segments else ""),
            encoding="utf-8",
        )
        (sidecar_root / "long_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
