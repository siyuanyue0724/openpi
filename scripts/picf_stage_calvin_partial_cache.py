from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.picf_core_train import _CalvinTransitionSource


def _parse_ranks(raw: str | None, *, world_size: int) -> list[int]:
    if raw is None or not str(raw).strip():
        return list(range(int(world_size)))
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _required_step_ids(
    *,
    source: _CalvinTransitionSource,
    seed: int,
    ranks: list[int],
    steps_per_rank: int,
) -> tuple[set[int], dict[int, dict[str, int]]]:
    def _resolve_window_metadata(flat_index: int, *, rng: np.random.Generator | None = None) -> tuple[int, int]:
        if hasattr(source, "sample_window_metadata"):
            return source.sample_window_metadata(int(flat_index), rng=rng)
        segment_id, start_step_id = source.window_index[int(flat_index)]
        return int(segment_id), int(start_step_id)

    step_ids: set[int] = set()
    summary: dict[int, dict[str, int]] = {}
    for rank in ranks:
        rng = np.random.default_rng(int(seed) + 17 * int(rank))
        sampled_flat_indices = [int(rng.integers(0, len(source))) for _ in range(int(steps_per_rank))]
        # picf_core_train does a pre-DDP lazy-module warmup with source.window(rank),
        # so the partial cache must include that deterministic initialization path too.
        warmup_flat_index = int(rank) % max(len(source), 1)
        rank_steps: set[int] = set()
        rank_segments: set[int] = set()
        for flat_index in sampled_flat_indices:
            segment_id, start_step_id = _resolve_window_metadata(flat_index, rng=rng)
            rank_segments.add(int(segment_id))
            for offset in range(source.unroll_steps + 1):
                step_id = int(start_step_id + offset)
                rank_steps.add(step_id)
                step_ids.add(step_id)
        warmup_segment_id, warmup_start_step_id = _resolve_window_metadata(warmup_flat_index)
        rank_segments.add(int(warmup_segment_id))
        for offset in range(source.unroll_steps + 1):
            step_id = int(warmup_start_step_id + offset)
            rank_steps.add(step_id)
            step_ids.add(step_id)
        summary[int(rank)] = {
            "num_windows": int(steps_per_rank),
            "unique_flat_indices": int(len(set(sampled_flat_indices + [warmup_flat_index]))),
            "unique_segments": int(len(rank_segments)),
            "unique_step_ids": int(len(rank_steps)),
            "warmup_flat_index": int(warmup_flat_index),
        }
    return step_ids, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage only the CALVIN frames needed for the first N PICF training steps into a local partial mirror."
    )
    parser.add_argument("--source-root", required=True, help="Source CALVIN task directory, e.g. /mnt/calvin_data/task_ABC_D")
    parser.add_argument("--dest-root", required=True, help="Destination task directory, e.g. /tmp/task_ABC_D_partial")
    parser.add_argument("--split", default="training")
    parser.add_argument("--backend", choices=("dir",), default="dir")
    parser.add_argument("--unroll-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--ranks", default=None, help="Comma-separated ranks to stage. Defaults to 0..world_size-1.")
    parser.add_argument("--steps-per-rank", type=int, default=3000)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()
    ranks = _parse_ranks(args.ranks, world_size=args.world_size)

    source = _CalvinTransitionSource(
        str(source_root),
        split=args.split,
        backend=args.backend,
        unroll_steps=args.unroll_steps,
        use_tactile=True,
    )
    step_ids, rank_summary = _required_step_ids(
        source=source,
        seed=args.seed,
        ranks=ranks,
        steps_per_rank=args.steps_per_rank,
    )

    rel_ann = Path(args.split) / "lang_annotations" / "auto_lang_ann.npy"
    ann_src = source_root / rel_ann
    ann_dst = dest_root / rel_ann
    ann_dst.parent.mkdir(parents=True, exist_ok=True)
    if not ann_dst.exists():
        shutil.copy2(ann_src, ann_dst)

    total_bytes = int(ann_src.stat().st_size)
    calib_src = source_root / "calib"
    calib_bytes = 0
    if calib_src.is_dir():
        for src_path in sorted(calib_src.rglob("*")):
            rel_path = src_path.relative_to(source_root)
            dst_path = dest_root / rel_path
            if src_path.is_dir():
                dst_path.mkdir(parents=True, exist_ok=True)
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            calib_bytes += int(src_path.stat().st_size)
    total_bytes += int(calib_bytes)
    for step_id in sorted(step_ids):
        rel_npz = Path(args.split) / f"episode_{step_id:07d}.npz"
        src_path = source_root / rel_npz
        dst_path = dest_root / rel_npz
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        total_bytes += int(src_path.stat().st_size)

    manifest = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "split": str(args.split),
        "backend": str(args.backend),
        "seed": int(args.seed),
        "world_size": int(args.world_size),
        "ranks": [int(rank) for rank in ranks],
        "steps_per_rank": int(args.steps_per_rank),
        "unroll_steps": int(args.unroll_steps),
        "unique_step_ids": int(len(step_ids)),
        "total_bytes": int(total_bytes),
        "total_gib": float(total_bytes / (1024**3)),
        "copied_calib": bool(calib_src.is_dir()),
        "calib_bytes": int(calib_bytes),
        "rank_summary": rank_summary,
    }
    manifest_path = dest_root / "partial_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
