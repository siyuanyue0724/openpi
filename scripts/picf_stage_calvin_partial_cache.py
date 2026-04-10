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
    step_ids: set[int] = set()
    summary: dict[int, dict[str, int]] = {}
    for rank in ranks:
        rng = np.random.default_rng(int(seed) + 17 * int(rank))
        flat_indices = [int(rng.integers(0, len(source))) for _ in range(int(steps_per_rank))]
        rank_steps: set[int] = set()
        rank_segments: set[int] = set()
        for flat_index in flat_indices:
            segment_id, start_step_id = source.window_index[flat_index]
            rank_segments.add(int(segment_id))
            for offset in range(source.unroll_steps + 1):
                step_id = int(start_step_id + offset)
                rank_steps.add(step_id)
                step_ids.add(step_id)
        summary[int(rank)] = {
            "num_windows": int(len(flat_indices)),
            "unique_flat_indices": int(len(set(flat_indices))),
            "unique_segments": int(len(rank_segments)),
            "unique_step_ids": int(len(rank_steps)),
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
    shutil.copy2(ann_src, ann_dst)

    total_bytes = int(ann_src.stat().st_size)
    for step_id in sorted(step_ids):
        rel_npz = Path(args.split) / f"episode_{step_id:07d}.npz"
        src_path = source_root / rel_npz
        dst_path = dest_root / rel_npz
        dst_path.parent.mkdir(parents=True, exist_ok=True)
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
        "rank_summary": rank_summary,
    }
    manifest_path = dest_root / "partial_cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
