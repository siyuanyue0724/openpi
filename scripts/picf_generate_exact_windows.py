from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.picf_core_train import _CalvinTransitionSource
from scripts.picf_replay_windows import _coerce_loaded_args


_BUCKET_PATTERNS = (
    ("block", re.compile(r"block", re.I)),
    ("drawer", re.compile(r"drawer", re.I)),
    ("slider", re.compile(r"slid|slide", re.I)),
    ("button", re.compile(r"button", re.I)),
    ("switch", re.compile(r"switch", re.I)),
    ("light", re.compile(r"light|lamp|bulb|led", re.I)),
    ("push", re.compile(r"push", re.I)),
    ("grasp", re.compile(r"grasp|pick|lift", re.I)),
    ("turn", re.compile(r"turn|rotate", re.I)),
    ("remove", re.compile(r"remove", re.I)),
)


def _bucket(prompt: str) -> str:
    hits = [name for name, pattern in _BUCKET_PATTERNS if pattern.search(str(prompt))]
    return "+".join(hits) if hits else "other"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_source(args_json: Path, *, device_override: str | None, split_override: str | None) -> tuple[argparse.Namespace, _CalvinTransitionSource]:
    payload = json.loads(args_json.read_text(encoding="utf-8"))
    train_args = _coerce_loaded_args(payload, device_override=device_override)
    if split_override is not None:
        train_args.split = str(split_override)
    calvin_segment_indices = None
    if getattr(train_args, "calvin_segment_indices", None):
        calvin_segment_indices = [
            int(part)
            for part in str(getattr(train_args, "calvin_segment_indices")).split(",")
            if part.strip()
        ]
    source = _CalvinTransitionSource(
        train_args.calvin_root,
        split=train_args.split,
        backend=train_args.backend,
        unroll_steps=train_args.effective_unroll_steps,
        action_horizon=train_args.action_horizon,
        use_tactile=bool(train_args.use_tactile),
        use_scene_obs=bool(train_args.use_scene_obs),
        action_normalizer=None,
        augmentation_mode="off",
        segment_indices=calvin_segment_indices,
    )
    return train_args, source


def _old_resume_records(
    *,
    source: _CalvinTransitionSource,
    seed: int,
    ranks: list[int],
    accum_steps: int,
    resume_step: int,
    num_steps: int,
    skip_steps: int,
) -> list[dict[str, Any]]:
    if int(num_steps) < 1:
        raise ValueError("--num-steps must be >= 1.")
    if int(skip_steps) < 0:
        raise ValueError("--skip-steps must be >= 0.")
    if int(accum_steps) < 1:
        raise ValueError("--accum-steps must be >= 1.")
    rngs = {int(rank): np.random.default_rng(int(seed) + 17 * int(rank)) for rank in ranks}
    records: list[dict[str, Any]] = []
    total_steps = int(skip_steps) + int(num_steps)
    for local_zero in range(total_steps):
        for rank in ranks:
            rng = rngs[int(rank)]
            for micro in range(int(accum_steps)):
                flat_index = int(rng.integers(0, len(source)))
                segment_id, start_step = source.sample_window_metadata(flat_index, rng=rng)
                if local_zero < int(skip_steps):
                    continue
                prompt = str(source.segments[int(segment_id)].lang)
                records.append(
                    {
                        "source": "old_resume_rng_reset",
                        "resume_step": int(resume_step),
                        "local_step": int(local_zero - int(skip_steps) + 1),
                        "source_step": int(resume_step + local_zero + 1),
                        "rank": int(rank),
                        "micro_step": int(micro + 1),
                        "flat_index": int(flat_index),
                        "segment": int(segment_id),
                        "start_step": int(start_step),
                        "prompt": prompt,
                        "bucket": _bucket(prompt),
                    }
                )
    return records


def _stratified_records(
    *,
    source: _CalvinTransitionSource,
    seed: int,
    bucket_names: list[str],
    per_bucket: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    by_bucket: dict[str, list[int]] = {name: [] for name in bucket_names}
    for slot_index, slot in enumerate(source.segment_sampling_slots):
        prompt = str(source.segments[int(slot.segment_id)].lang)
        b = _bucket(prompt)
        for name in bucket_names:
            if name == b or (name == "manipulator" and any(part in b for part in ("button", "switch", "light"))):
                by_bucket[name].append(int(slot_index))
    records: list[dict[str, Any]] = []
    for name in bucket_names:
        candidates = by_bucket.get(name, [])
        if not candidates:
            raise RuntimeError(f"No CALVIN segment slots found for bucket {name!r}.")
        replace = len(candidates) < int(per_bucket)
        chosen = rng.choice(np.asarray(candidates, dtype=np.int64), size=int(per_bucket), replace=replace)
        for idx, flat_index in enumerate(chosen.tolist()):
            segment_id, start_step = source.sample_window_metadata(int(flat_index), rng=rng)
            prompt = str(source.segments[int(segment_id)].lang)
            records.append(
                {
                    "source": "stratified_bucket",
                    "bucket_request": str(name),
                    "bucket": _bucket(prompt),
                    "local_step": int(idx + 1),
                    "flat_index": int(flat_index),
                    "segment": int(segment_id),
                    "start_step": int(start_step),
                    "prompt": prompt,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact CALVIN window records for fixed-window action probes.")
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default=None)
    parser.add_argument("--mode", choices=("old-resume", "stratified"), required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ranks", default="0,1")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--skip-steps", type=int, default=0)
    parser.add_argument("--buckets", default="block+grasp,block+push,block+slider,drawer,slider,manipulator")
    parser.add_argument("--per-bucket", type=int, default=32)
    args = parser.parse_args()

    train_args, source = _load_source(Path(args.args_json), device_override=args.device, split_override=args.split)
    try:
        seed = int(train_args.seed if args.seed is None else args.seed)
        if args.mode == "old-resume":
            ranks = [int(part.strip()) for part in str(args.ranks).split(",") if part.strip()]
            records = _old_resume_records(
                source=source,
                seed=seed,
                ranks=ranks,
                accum_steps=int(args.accum_steps),
                resume_step=int(args.resume_step),
                num_steps=int(args.num_steps),
                skip_steps=int(args.skip_steps),
            )
        else:
            buckets = [part.strip() for part in str(args.buckets).split(",") if part.strip()]
            records = _stratified_records(
                source=source,
                seed=seed,
                bucket_names=buckets,
                per_bucket=int(args.per_bucket),
            )
        _write_jsonl(Path(args.output_jsonl), records)
        summary = {
            "stage": "exact_windows_generated",
            "mode": str(args.mode),
            "output_jsonl": str(args.output_jsonl),
            "records": int(len(records)),
            "dataset_size": int(len(source)),
            "seed": int(seed),
            "split": str(train_args.split),
        }
        print(json.dumps(summary, sort_keys=True))
    finally:
        source.close()


if __name__ == "__main__":
    main()
