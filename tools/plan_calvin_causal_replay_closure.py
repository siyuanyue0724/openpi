#!/usr/bin/env python3
"""Plan the exact CALVIN file closure for a causal replay schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
from tools.run_lingbot_vla2_task_independent_p1 import (
    P2_CAUSAL_REPLAY_CLOSURE_SCHEMA,
    _calvin_causal_replay_dependency_closure,
    _select_p2_causal_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--predictive-cache", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--prefix-frames", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument(
        "--selection-domain",
        choices=("registered-evaluation", "all-nontraining"),
        default="registered-evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--available-root", type=Path, action="append", default=[])
    return parser.parse_args()


def _cache_records(cache_root: Path) -> tuple[SimpleNamespace, ...]:
    records: list[SimpleNamespace] = []
    for shard in sorted(cache_root.glob("shard-*.npz")):
        with np.load(shard, allow_pickle=False) as archive:
            starts = archive["frame_offsets"][:-1]
            stops = archive["frame_offsets"][1:]
            for source, target, horizon, start, stop in zip(
                archive["source_global_indices"],
                archive["target_global_indices"],
                archive["horizons"],
                starts,
                stops,
                strict=True,
            ):
                records.append(
                    SimpleNamespace(
                        source_global_index=int(source),
                        target_global_index=int(target),
                        horizon=int(horizon),
                        importance=archive["importance"][int(start) : int(stop)],
                    )
                )
    if not records:
        raise RuntimeError("predictive cache contains no records")
    return tuple(records)


def main() -> None:
    args = _parse_args()
    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        args.dataset_split.resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=args.action_horizon)
    split = RepresentationTrialSplit.load(args.representation_split)
    if args.selection_domain == "registered-evaluation":
        selected_segment_indices = {
            item.segment_index for item in (*split.validation_segments, *split.heldout_segments)
        }
        allowed_episode_indices = frozenset(split.evaluation_source_episode_indices)
    else:
        selected_segment_indices = {int(segment.index) for segment in index.segments}
        allowed_episode_indices = frozenset(
            int(segment.episode_index)
            for segment in index.segments
            if int(segment.episode_index) not in split.training_source_episode_indices
        )
    selected_pairs = tuple(
        (segment, episode)
        for segment, episode in zip(index.segments, dataset.episode_manifest, strict=True)
        if int(segment.index) in selected_segment_indices
        and int(segment.episode_index) in allowed_episode_indices
    )
    if not selected_pairs:
        raise RuntimeError("selection domain contains no source-disjoint CALVIN segments")

    selected = _select_p2_causal_records(
        records=_cache_records(args.predictive_cache),
        segments=tuple(item[0] for item in selected_pairs),
        episodes=tuple(item[1] for item in selected_pairs),
        horizon=args.horizon,
        count=args.count,
        prefix_frames=args.prefix_frames,
        allowed_episode_indices=allowed_episode_indices,
        require_positive_importance=True,
        selection_seed=args.selection_seed,
        distinct_source_episodes=True,
    )

    required_indices: set[int] = set()
    selections: list[dict[str, object]] = []
    for record, segment, _episode, transition_index in selected:
        replay_indices, closure = _calvin_causal_replay_dependency_closure(
            source_global_index=int(record.source_global_index),
            segment_start=int(segment.start),
            segment_end=int(segment.end),
            horizon=args.horizon,
            prefix_frames=args.prefix_frames,
            action_horizon=args.action_horizon,
        )
        required_indices.update(closure)
        selections.append(
            {
                "source_global_index": int(record.source_global_index),
                "target_global_index": int(record.target_global_index),
                "segment_index": int(segment.index),
                "source_episode_index": int(segment.episode_index),
                "transition_index": int(transition_index),
                "replay_global_indices": list(replay_indices),
                "required_global_indices": list(closure),
            }
        )

    required_paths = [f"episode_{index_value:07d}.npz" for index_value in sorted(required_indices)]
    available_roots = (args.dataset_split, *args.available_root)
    missing_paths = [
        relative
        for relative in required_paths
        if not any((root / relative).is_file() for root in available_roots)
    ]
    payload: dict[str, object] = {
        "schema": P2_CAUSAL_REPLAY_CLOSURE_SCHEMA,
        "selection_seed": args.selection_seed,
        "selection_domain": args.selection_domain,
        "training_source_episode_indices_sha256": hashlib.sha256(
            json.dumps(sorted(split.training_source_episode_indices)).encode("ascii")
        ).hexdigest(),
        "count": args.count,
        "horizon": args.horizon,
        "prefix_frames": args.prefix_frames,
        "action_horizon": args.action_horizon,
        "selections": selections,
        "required_paths": required_paths,
        "missing_paths": missing_paths,
        "available_roots": [str(root.resolve()) for root in available_roots],
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "representation_split_file_sha256": _sha256(args.representation_split),
        "predictive_cache_manifest_sha256": _sha256(args.predictive_cache / "manifest.json"),
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
