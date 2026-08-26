#!/usr/bin/env python3
"""Audit duration-dependent CALVIN visibility through the production target builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from picf_next.eval.lifecycle import (  # noqa: E402
    audit_partitioned_visibility_target_sequences,
    audit_visibility_target_sequences,
)
from picf_next.hosts.molmoact2_training import (  # noqa: E402
    CalvinStatefulLossTargetLayout,
)
from picf_next.models.evidence import ModalityTokenSpan  # noqa: E402
from picf_next.training.stationary_calvin_stage import (  # noqa: E402
    load_stationary_calvin_stage_assets,
    load_stationary_calvin_stage_definition,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-recipe",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m3_stationary_temporal.json",
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--feature-cache-manifest-sha256", required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-examples", type=int, default=32)
    return parser.parse_args()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _write_atomic(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload)
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _task_annotations(index: Any, start: int, stop: int) -> list[dict[str, object]]:
    return [
        {
            "segment_index": segment.index,
            "task_key": segment.task_key,
            "instruction": segment.instruction,
        }
        for segment in index.segments
        if segment.start <= stop and segment.end >= start
    ]


def _hidden_run_examples(
    indexed_sequences: Sequence[Sequence[tuple[int, Mapping[str, int | None]]]],
    *,
    index: Any,
    split_by_global_index: Mapping[int, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reacquired: list[dict[str, object]] = []
    right_censored: list[dict[str, object]] = []
    for sequence in indexed_sequences:
        identities = sorted({identity for _step, frame in sequence for identity in frame})
        for identity in identities:
            run_start: int | None = None
            prior_visible: int | None = None
            for offset, (global_index, frame) in enumerate(sequence):
                label = frame.get(identity)
                if label == 0:
                    if run_start is None:
                        run_start = offset
                    continue
                if run_start is not None:
                    start_global_index = sequence[run_start][0]
                    stop_global_index = sequence[offset - 1][0]
                    base = {
                        "identity": identity,
                        "split": split_by_global_index[start_global_index],
                        "hidden_start_global_index": start_global_index,
                        "hidden_stop_global_index": stop_global_index,
                        "hidden_length": offset - run_start,
                        "prior_visible_global_index": prior_visible,
                        "task_annotations": _task_annotations(
                            index,
                            start_global_index,
                            stop_global_index,
                        ),
                    }
                    if label == 1:
                        reacquired.append(
                            {
                                **base,
                                "reappeared_global_index": global_index,
                            }
                        )
                    run_start = None
                if label == 1:
                    prior_visible = global_index
                elif label is None:
                    prior_visible = None
            if run_start is not None:
                start_global_index = sequence[run_start][0]
                stop_global_index = sequence[-1][0]
                right_censored.append(
                    {
                        "identity": identity,
                        "split": split_by_global_index[start_global_index],
                        "hidden_start_global_index": start_global_index,
                        "hidden_stop_global_index": stop_global_index,
                        "hidden_length": len(sequence) - run_start,
                        "prior_visible_global_index": prior_visible,
                        "task_annotations": _task_annotations(
                            index,
                            start_global_index,
                            stop_global_index,
                        ),
                    }
                )

    def sort_key(row: Mapping[str, object]) -> tuple[int, int, str]:
        return (
            -int(row["hidden_length"]),
            int(row["hidden_start_global_index"]),
            str(row["identity"]),
        )

    return sorted(reacquired, key=sort_key), sorted(right_censored, key=sort_key)


def main() -> None:
    args = _parse_args()
    if args.maximum_examples <= 0:
        raise ValueError("maximum examples must be positive")
    definition = load_stationary_calvin_stage_definition(
        args.stage_recipe,
        repository_root=_ROOT,
    )
    assets = load_stationary_calvin_stage_assets(
        definition,
        repository_root=_ROOT,
        split_root=args.split_root,
        feature_cache_root=args.feature_cache_root,
        feature_cache_manifest_sha256=args.feature_cache_manifest_sha256,
        physical_sidecar_root=args.physical_sidecar_root,
        cache_shards=1,
    )
    cache = assets.feature_cache
    token_valid = torch.ones((1, cache.token_count), dtype=torch.bool)
    layout = CalvinStatefulLossTargetLayout(
        token_valid=token_valid,
        spans=(ModalityTokenSpan(cache.modality, 0, cache.token_count),),
        target_dtype=torch.float32,
        rollout_input_dtype=torch.float32,
        vision_patch_layout=cache.vision_layout(1),
    )

    visibility_by_global_index: dict[int, dict[str, int | None]] = {}
    split_by_global_index = {
        global_index: record.split for global_index, record in cache.records.items()
    }
    for global_index in sorted(cache.records):
        targets = assets.batch_builder.visible_target_builder(
            (cache.target_request(global_index),),
            layout,
        )
        if targets.lifecycle_targets is None or targets.lifecycle_targets[0] is None:
            raise RuntimeError("production target builder omitted lifecycle supervision")
        lifecycle = targets.lifecycle_targets[0]
        if lifecycle.visibility is None or lifecycle.visibility_supervised is None:
            raise RuntimeError("production target builder omitted visibility supervision")
        visibility_by_global_index[global_index] = {
            identity: (
                int(lifecycle.visibility[row].item())
                if bool(lifecycle.visibility_supervised[row].item())
                else None
            )
            for row, identity in enumerate(lifecycle.alive_identity_keys)
        }

    indexed_sequences: list[list[tuple[int, Mapping[str, int | None]]]] = []
    current: list[tuple[int, Mapping[str, int | None]]] = []
    previous_global_index: int | None = None
    previous_episode: int | None = None
    for global_index, visibility in visibility_by_global_index.items():
        episode = assets.index.source_episode(global_index).index
        if previous_global_index is not None and (
            global_index != previous_global_index + 1 or episode != previous_episode
        ):
            indexed_sequences.append(current)
            current = []
        current.append((global_index, visibility))
        previous_global_index = global_index
        previous_episode = episode
    if current:
        indexed_sequences.append(current)
    if not indexed_sequences:
        raise RuntimeError("duration audit found no contiguous source sequences")

    target_sequences = tuple(
        tuple(frame for _global_index, frame in sequence) for sequence in indexed_sequences
    )
    census = audit_visibility_target_sequences(target_sequences)
    target_sequences_by_split: dict[
        str,
        list[tuple[Mapping[str, int | None], ...]],
    ] = {}
    for sequence in indexed_sequences:
        sequence_splits = {split_by_global_index[global_index] for global_index, _frame in sequence}
        if len(sequence_splits) != 1:
            raise RuntimeError("one contiguous duration sequence crossed a frozen data split")
        split = next(iter(sequence_splits))
        target_sequences_by_split.setdefault(split, []).append(
            tuple(frame for _global_index, frame in sequence)
        )
    census_by_split = audit_partitioned_visibility_target_sequences(target_sequences_by_split)
    reacquired, right_censored = _hidden_run_examples(
        indexed_sequences,
        index=assets.index,
        split_by_global_index=split_by_global_index,
    )
    seen_then_reacquired = [
        row for row in reacquired if row["prior_visible_global_index"] is not None
    ]
    unseen_then_visible = [row for row in reacquired if row["prior_visible_global_index"] is None]
    seen_then_right_censored = [
        row for row in right_censored if row["prior_visible_global_index"] is not None
    ]
    visibility_digest = hashlib.sha256(
        _canonical_json(
            [
                [global_index, visibility_by_global_index[global_index]]
                for global_index in visibility_by_global_index
            ]
        )
    ).hexdigest()
    report = {
        "schema": "picf-next.calvin-visibility-duration-audit.v3",
        "status": "PASS",
        "stage_recipe_sha256": definition.stage.recipe_sha256,
        "source_coverage_recipe_sha256": definition.source_coverage.recipe_sha256,
        "feature_cache_manifest_sha256": cache.manifest_sha256,
        "physical_sidecar_manifest_sha256": (
            definition.source_coverage.physical_sidecar_manifest_sha256
        ),
        "frame_count": len(visibility_by_global_index),
        "sequence_count": len(indexed_sequences),
        "visibility_targets_sha256": visibility_digest,
        "visibility_target_census": census,
        "visibility_target_census_by_split": census_by_split,
        "reacquired_hidden_run_count": len(reacquired),
        "right_censored_hidden_run_count": len(right_censored),
        "seen_then_reacquired_hidden_run_count": len(seen_then_reacquired),
        "unseen_then_visible_hidden_run_count": len(unseen_then_visible),
        "seen_then_right_censored_hidden_run_count": len(seen_then_right_censored),
        "longest_reacquired_hidden_runs": reacquired[: args.maximum_examples],
        "longest_right_censored_hidden_runs": right_censored[: args.maximum_examples],
        "longest_seen_then_reacquired_hidden_runs": seen_then_reacquired[: args.maximum_examples],
        "longest_seen_then_right_censored_hidden_runs": seen_then_right_censored[
            : args.maximum_examples
        ],
        "runtime_target_leakage": False,
    }
    _write_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
