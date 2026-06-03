#!/usr/bin/env python3
"""Audit CALVIN action-window targets and MVTrack sidecar coverage.

This diagnostic intentionally avoids importing `picf_core_train`: it must audit
dataflow, not model construction.  It reuses the lightweight CALVIN bucket
sampler, samples the same logical task windows as training, and reports raw and
normalized action target scale plus optional proposal/tracklet sidecar coverage.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

if __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from openpi.picf.action_normalization import PicfActionNormalizer  # noqa: E402
from openpi.picf.action_normalization import default_calvin_action_norm_stats_path  # noqa: E402
from scripts.picf_calvin_bucket_sampler_audit import _AuditSource  # noqa: E402


_SIDECAR_KEYS = (
    "tracklet_xy",
    "tracklet_velocity",
    "tracklet_visibility",
    "tracklet_confidence",
    "tracklet_ids",
    "tracklet_view_ids",
    "tracklet_age",
    "proposal_centers_xy",
    "proposal_boxes_xyxy",
    "proposal_objectness",
    "proposal_view_ids",
    "proposal_source_ids",
    "proposal_age",
    "proposal_mask_xy",
    "proposal_mask_weights",
    "proposal_mask_offsets",
)


def _payload_get(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    value = payload.get(key, default)
    return default if value is None else value


def _bool_payload(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = _payload_get(payload, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int_payload(payload: dict[str, Any], key: str, default: int) -> int:
    return int(_payload_get(payload, key, default))


def _float_payload(payload: dict[str, Any], key: str, default: float) -> float:
    return float(_payload_get(payload, key, default))


def _parse_segment_indices(value: str | None) -> list[int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return [int(part) for part in text.split(",") if part.strip()]


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = _finite(values)
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _as_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def _normalize_action_np(normalizer: PicfActionNormalizer | None, action: np.ndarray | None) -> np.ndarray | None:
    if action is None:
        return None
    arr = np.asarray(action, dtype=np.float32)
    return arr if normalizer is None else normalizer.normalize_np(arr)


def _load_action_chunk(
    reader: Any,
    *,
    step_id: int,
    segment_end: int,
    action_horizon: int,
    current_action: np.ndarray | None = None,
    action_key: str = "rel_actions",
) -> np.ndarray | None:
    if int(action_horizon) <= 1:
        return None
    current = (
        np.asarray(reader.read_npz(step_id, keys=[action_key])[action_key], dtype=np.float32)
        if current_action is None
        else np.asarray(current_action, dtype=np.float32)
    )
    actions = [current]
    last = current
    for future_step in range(int(step_id) + 1, int(step_id) + int(action_horizon)):
        if future_step < int(segment_end):
            last = np.asarray(reader.read_npz(future_step, keys=[action_key])[action_key], dtype=np.float32)
        actions.append(last)
    return np.stack(actions, axis=0)


def _append_action_stats(prefix: str, action: np.ndarray | None, stats: dict[str, list[float]]) -> None:
    arr = _as_float_array(action)
    if arr is None:
        stats[f"{prefix}_present"].append(0.0)
        return
    stats[f"{prefix}_present"].append(1.0)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(1, -1)
    stats[f"{prefix}_dim"].append(float(flat.shape[-1]))
    stats[f"{prefix}_l2"].extend(np.linalg.norm(flat, axis=-1).astype(np.float64).tolist())
    stats[f"{prefix}_absmax"].extend(np.max(np.abs(flat), axis=-1).astype(np.float64).tolist())
    stats[f"{prefix}_mean_abs"].extend(np.mean(np.abs(flat), axis=-1).astype(np.float64).tolist())
    if flat.shape[-1] >= 3:
        stats[f"{prefix}_pos_l2"].extend(np.linalg.norm(flat[:, :3], axis=-1).astype(np.float64).tolist())
    if flat.shape[-1] >= 6:
        stats[f"{prefix}_rot_l2"].extend(np.linalg.norm(flat[:, 3:6], axis=-1).astype(np.float64).tolist())
    if flat.shape[-1] >= 7:
        stats[f"{prefix}_gripper_abs"].extend(np.abs(flat[:, 6]).astype(np.float64).tolist())
    stats[f"{prefix}_nonfinite_fraction"].append(float(1.0 - np.isfinite(flat).mean()))
    stats[f"{prefix}_outside_unit_fraction"].append(float((np.abs(flat) > 1.0).mean()))
    stats[f"{prefix}_outside_two_fraction"].append(float((np.abs(flat) > 2.0).mean()))
    if flat.shape[0] > 1:
        diffs = np.diff(flat, axis=0)
        stats[f"{prefix}_temporal_delta_l2"].extend(np.linalg.norm(diffs, axis=-1).astype(np.float64).tolist())


def _proposal_count(frame: dict[str, np.ndarray]) -> int:
    if "proposal_centers_xy" in frame:
        return int(np.asarray(frame["proposal_centers_xy"]).reshape(-1, 2).shape[0])
    if "proposal_boxes_xyxy" in frame:
        return int(np.asarray(frame["proposal_boxes_xyxy"]).reshape(-1, 4).shape[0])
    return 0


def _load_sidecar_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in _SIDECAR_KEYS if key in data.files}


def _read_sidecar_fields(
    sidecar_root: str | Path | None,
    *,
    split: str,
    step_id: int,
    nearest_max_gap: int,
) -> dict[str, np.ndarray]:
    if not sidecar_root:
        return {}
    root = Path(sidecar_root)
    for path in (
        root / split / f"episode_{int(step_id):07d}.npz",
        root / f"episode_{int(step_id):07d}.npz",
    ):
        if path.exists():
            frame = _load_sidecar_npz(path)
            if _proposal_count(frame) > 0 or int(nearest_max_gap) <= 0:
                return frame
            break
    for gap in range(1, max(int(nearest_max_gap), 0) + 1):
        for signed_gap in (-gap, gap):
            for path in (
                root / split / f"episode_{int(step_id + signed_gap):07d}.npz",
                root / f"episode_{int(step_id + signed_gap):07d}.npz",
            ):
                if not path.exists():
                    continue
                frame = _load_sidecar_npz(path)
                count = _proposal_count(frame)
                if count <= 0:
                    continue
                frame["proposal_age"] = np.full((count,), float(abs(int(signed_gap))), dtype=np.float32)
                return frame
    return {}


def _append_sidecar_stats(sidecar: dict[str, np.ndarray], stats: dict[str, list[float]]) -> None:
    proposal_centers = _as_float_array(sidecar.get("proposal_centers_xy"))
    proposal_boxes = _as_float_array(sidecar.get("proposal_boxes_xyxy"))
    proposal_objectness = _as_float_array(sidecar.get("proposal_objectness"))
    proposal_age = _as_float_array(sidecar.get("proposal_age"))
    proposal_mask_xy = _as_float_array(sidecar.get("proposal_mask_xy"))
    proposal_mask_weights = _as_float_array(sidecar.get("proposal_mask_weights"))
    tracklet_xy = _as_float_array(sidecar.get("tracklet_xy"))
    tracklet_conf = _as_float_array(sidecar.get("tracklet_confidence"))

    proposal_count = 0
    if proposal_centers is not None:
        proposal_count = int(proposal_centers.reshape(-1, 2).shape[0])
    elif proposal_boxes is not None:
        proposal_count = int(proposal_boxes.reshape(-1, 4).shape[0])
    stats["proposal_count"].append(float(proposal_count))
    stats["proposal_mask_point_count"].append(float(0 if proposal_mask_xy is None else proposal_mask_xy.reshape(-1, 2).shape[0]))
    stats["tracklet_count"].append(float(0 if tracklet_xy is None else tracklet_xy.reshape(-1, 2).shape[0]))
    if proposal_objectness is not None:
        stats["proposal_objectness"].extend(proposal_objectness.reshape(-1).astype(np.float64).tolist())
    if proposal_age is not None:
        stats["proposal_age"].extend(proposal_age.reshape(-1).astype(np.float64).tolist())
    if proposal_mask_weights is not None:
        weights = proposal_mask_weights.reshape(-1)
        stats["proposal_mask_weight"].extend(weights.astype(np.float64).tolist())
        stats["proposal_mask_weight_nonzero_fraction"].append(float((weights > 0.0).mean()) if weights.size else 0.0)
    if tracklet_conf is not None:
        stats["tracklet_confidence"].extend(tracklet_conf.reshape(-1).astype(np.float64).tolist())


def _action_normalizer_from_payload(payload: dict[str, Any]) -> PicfActionNormalizer | None:
    mode = str(_payload_get(payload, "action_normalization", "quantile")).lower()
    if mode == "none":
        return None
    stats_path = _payload_get(payload, "action_norm_stats_path", None)
    if not stats_path:
        stats_path = str(default_calvin_action_norm_stats_path())
    return PicfActionNormalizer.from_path(stats_path, mode=mode)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--rank-count", type=int, default=None)
    parser.add_argument("--calvin-bucket-sampling-mode", default=None)
    parser.add_argument("--calvin-bucket-temperature-alpha", type=float, default=None)
    parser.add_argument("--calvin-bucket-weight-spec", default=None)
    parser.add_argument("--calvin-bucket-sample-without-replacement", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.args_json).read_text(encoding="utf-8"))
    split = str(args.split or _payload_get(payload, "split", "training"))
    seed = int(args.seed if args.seed is not None else _payload_get(payload, "seed", 0))
    accum_steps = int(args.accum_steps if args.accum_steps is not None else _payload_get(payload, "accum_steps", 1))
    action_horizon = _int_payload(payload, "action_horizon", 16)
    unroll_steps = _int_payload(payload, "effective_unroll_steps", _int_payload(payload, "unroll_steps", 2))
    bucket_mode = str(args.calvin_bucket_sampling_mode or _payload_get(payload, "calvin_bucket_sampling_mode", "round_robin"))
    bucket_alpha = float(
        args.calvin_bucket_temperature_alpha
        if args.calvin_bucket_temperature_alpha is not None
        else _payload_get(payload, "calvin_bucket_temperature_alpha", 0.0)
    )
    bucket_spec = str(
        args.calvin_bucket_weight_spec
        if args.calvin_bucket_weight_spec is not None
        else _payload_get(payload, "calvin_bucket_weight_spec", "")
    )
    bucket_wor = (
        bool(args.calvin_bucket_sample_without_replacement)
        if args.calvin_bucket_sample_without_replacement is not None
        else _bool_payload(payload, "calvin_bucket_sample_without_replacement", True)
    )
    source = _AuditSource(
        root=str(_payload_get(payload, "calvin_root")),
        split=split,
        backend=str(_payload_get(payload, "backend", "dir")),
        unroll_steps=unroll_steps,
        action_horizon=action_horizon,
        segment_indices=_parse_segment_indices(_payload_get(payload, "calvin_segment_indices", "")),
        bucket_sampling_mode=bucket_mode,
        bucket_temperature_alpha=bucket_alpha,
        bucket_weight_spec=bucket_spec,
        bucket_sample_without_replacement=bucket_wor,
    )
    normalizer = _action_normalizer_from_payload(payload)
    sidecar_root = _payload_get(payload, "mvtrack_sidecar_root", None)
    sidecar_gap = _int_payload(payload, "mvtrack_sidecar_proposal_nearest_max_gap", 0)
    action_key = str(_payload_get(payload, "action_key", "rel_actions"))
    try:
        world_size = int(args.world_size)
        rank_count = world_size if args.rank_count is None else min(int(args.rank_count), world_size)
        stats_by_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        examples: list[dict[str, Any]] = []
        distinct_counts: list[int] = []
        sample_count = 0
        for step_offset in range(int(args.steps)):
            step = int(args.start_step) + int(step_offset)
            if int(args.progress_every) > 0 and step_offset % int(args.progress_every) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "picf_action_window_target_audit_progress",
                            "step_offset": int(step_offset),
                            "steps": int(args.steps),
                            "sample_count": int(sample_count),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            step_buckets: list[str] = []
            for rank in range(rank_count):
                for micro_step in range(accum_steps):
                    slot_index, bucket, rng = source.balanced_bucket_slot_index(
                        seed=seed,
                        rank=int(rank),
                        world_size=world_size,
                        step=step,
                        micro_step=int(micro_step),
                        accum_steps=accum_steps,
                    )
                    slot = source.segment_sampling_slots[int(slot_index)]
                    segment_id = int(slot["segment_id"])
                    segment = source.segments[segment_id]
                    start_step_id = int(rng.integers(slot["first_valid_start_step_id"], slot["valid_start_exclusive"]))
                    current = np.asarray(source.reader.read_npz(start_step_id, keys=[action_key])[action_key], dtype=np.float32)
                    chunk = _load_action_chunk(
                        source.reader,
                        step_id=start_step_id,
                        segment_end=int(segment.end),
                        action_horizon=action_horizon,
                        current_action=current,
                        action_key=action_key,
                    )
                    stats = stats_by_bucket[str(bucket)]
                    _append_action_stats("raw_action", current, stats)
                    _append_action_stats("raw_action_chunk", chunk, stats)
                    _append_action_stats("norm_action", _normalize_action_np(normalizer, current), stats)
                    _append_action_stats("norm_action_chunk", _normalize_action_np(normalizer, chunk), stats)
                    sidecar = _read_sidecar_fields(
                        sidecar_root,
                        split=split,
                        step_id=start_step_id,
                        nearest_max_gap=sidecar_gap,
                    )
                    _append_sidecar_stats(sidecar, stats)
                    stats["segment_id"].append(float(segment_id))
                    stats["start_step_id"].append(float(start_step_id))
                    stats["window_duration"].append(float(unroll_steps + action_horizon))
                    step_buckets.append(str(bucket))
                    sample_count += 1
                    if len(examples) < 32:
                        examples.append(
                            {
                                "step": int(step),
                                "rank": int(rank),
                                "micro_step": int(micro_step),
                                "bucket": str(bucket),
                                "segment_id": int(segment_id),
                                "start_step_id": int(start_step_id),
                                "prompt": str(segment.lang),
                            }
                        )
            distinct_counts.append(int(len(set(step_buckets))))

        overall: dict[str, list[float]] = defaultdict(list)
        for stats in stats_by_bucket.values():
            for key, values in stats.items():
                overall[key].extend(values)
        output = {
            "stage": "picf_action_window_target_audit",
            "args_json": str(args.args_json),
            "split": split,
            "seed": seed,
            "steps": int(args.steps),
            "start_step": int(args.start_step),
            "world_size": world_size,
            "rank_count": rank_count,
            "accum_steps": accum_steps,
            "sample_count": int(sample_count),
            "action_horizon": int(action_horizon),
            "unroll_steps": int(unroll_steps),
            "action_normalization": str(_payload_get(payload, "action_normalization", "quantile")),
            "action_norm_stats_path": str(_payload_get(payload, "action_norm_stats_path", default_calvin_action_norm_stats_path())),
            "sidecar_root": None if sidecar_root is None else str(sidecar_root),
            "sidecar_nearest_max_gap": int(sidecar_gap),
            "calvin_bucket_sampling_mode": bucket_mode,
            "calvin_bucket_temperature_alpha": float(bucket_alpha),
            "calvin_bucket_weight_spec": bucket_spec,
            "calvin_bucket_sample_without_replacement": bool(bucket_wor),
            "bucket_names": list(source.bucket_names),
            "bucket_segment_counts": dict(source.bucket_segment_counts),
            "bucket_target_weights": dict(source.bucket_target_weights),
            "logical_step_distinct_bucket_count": _summary(distinct_counts),
            "overall": {key: _summary(values) for key, values in sorted(overall.items())},
            "buckets": {
                str(bucket): {key: _summary(values) for key, values in sorted(stats.items())}
                for bucket, stats in sorted(stats_by_bucket.items())
            },
            "examples": examples,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(output, sort_keys=True), flush=True)
    finally:
        source.close()


if __name__ == "__main__":
    main()
