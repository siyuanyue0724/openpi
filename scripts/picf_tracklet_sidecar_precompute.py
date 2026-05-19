#!/usr/bin/env python3
"""Offline proposal-seeded tracklet sidecar generator for PICF/MVTrack.

The generator keeps tracking outside the online training graph.  It reads
CALVIN RGB frames, uses existing proposal sidecars as high-quality seed
points when available, augments them with a small set of generic visual seeds,
tracks short windows with KLT optical flow, and writes the optional
`tracklet_*` arrays already consumed by PICF/MVTrack.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile
import zlib

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

_VIEW_IDS = {"static": 0, "gripper": 1}
_TRACKLET_KEYS = (
    "tracklet_xy",
    "tracklet_velocity",
    "tracklet_visibility",
    "tracklet_confidence",
    "tracklet_ids",
    "tracklet_view_ids",
    "tracklet_age",
)
_PROPOSAL_KEYS = (
    "proposal_centers_xy",
    "proposal_boxes_xyxy",
    "proposal_objectness",
    "proposal_view_ids",
    "proposal_source_ids",
)


def _cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/environment dependent.
        raise RuntimeError(
            "OpenCV is required for tracklet precompute. Install cv2/opencv-python "
            "in the offline preprocessing environment."
        ) from exc
    return cv2


def _calvin_dataset_cls():
    from openpi.training.calvin_dataset import CalvinLangSegmentDataset

    return CalvinLangSegmentDataset


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {arr.shape}.")
    if arr.dtype == np.uint8:
        return arr
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if float(arr.max(initial=0.0)) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _gray(image: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    return cv2.cvtColor(_as_uint8_rgb(image), cv2.COLOR_RGB2GRAY)


def _sidecar_path(root: str | Path | None, split: str, step_id: int) -> Path | None:
    if root is None:
        return None
    root = Path(root)
    candidates = (
        root / split / f"episode_{int(step_id):07d}.npz",
        root / f"episode_{int(step_id):07d}.npz",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _load_sidecar_payload(root: str | Path | None, split: str, step_id: int) -> dict[str, np.ndarray]:
    path = _sidecar_path(root, split, step_id)
    if path is None:
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (EOFError, ValueError, OSError, zipfile.BadZipFile, zlib.error):
        # Large sidecar jobs can be preempted while a compressed npz is being
        # written. Treat incomplete files as missing evidence so resume can
        # rewrite them instead of killing the whole shard.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return {}


def _atomic_savez_compressed(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz", delete=False) as tmp:
        tmp_name = tmp.name
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _proposal_seed_points(
    proposal_payload: dict[str, np.ndarray],
    *,
    view_id: int,
    max_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(proposal_payload.get("proposal_centers_xy", np.zeros((0, 2))), dtype=np.float32).reshape(-1, 2)
    objectness = np.asarray(proposal_payload.get("proposal_objectness", np.zeros((0,))), dtype=np.float32).reshape(-1)
    proposal_view_ids = np.asarray(proposal_payload.get("proposal_view_ids", np.zeros((0,), dtype=np.int64))).reshape(-1)
    if centers.shape[0] == 0 or objectness.shape[0] == 0 or proposal_view_ids.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    count = min(centers.shape[0], objectness.shape[0], proposal_view_ids.shape[0])
    keep = np.where(proposal_view_ids[:count].astype(np.int64) == int(view_id))[0]
    if keep.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    order = keep[np.argsort(-objectness[keep])]
    order = order[: max(int(max_count), 0)]
    return np.clip(centers[order], 0.0, 1.0).astype(np.float32), np.clip(objectness[order], 0.0, 1.0).astype(np.float32)


def _grid_seed_points(max_count: int) -> tuple[np.ndarray, np.ndarray]:
    max_count = max(int(max_count), 0)
    if max_count <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    cols = int(np.ceil(np.sqrt(max_count)))
    rows = int(np.ceil(max_count / max(cols, 1)))
    xs = np.linspace(0.16, 0.84, cols, dtype=np.float32)
    ys = np.linspace(0.16, 0.84, rows, dtype=np.float32)
    pts = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)[:max_count]
    # Grid seeds are deliberately lower confidence than SAM seeds.  They provide
    # coverage, not objectness truth.
    conf = np.full((pts.shape[0],), 0.45, dtype=np.float32)
    return pts, conf


@dataclass
class _TrackRows:
    xy: list[np.ndarray]
    velocity: list[np.ndarray]
    visibility: list[float]
    confidence: list[float]
    track_ids: list[int]
    view_ids: list[int]
    age: list[float]

    @classmethod
    def empty(cls) -> "_TrackRows":
        return cls([], [], [], [], [], [], [])

    def add(
        self,
        *,
        xy: np.ndarray,
        velocity: np.ndarray,
        visibility: float,
        confidence: float,
        track_id: int,
        view_id: int,
        age: float,
    ) -> None:
        self.xy.append(np.asarray(xy, dtype=np.float32))
        self.velocity.append(np.asarray(velocity, dtype=np.float32))
        self.visibility.append(float(visibility))
        self.confidence.append(float(confidence))
        self.track_ids.append(int(track_id))
        self.view_ids.append(int(view_id))
        self.age.append(float(age))

    def to_arrays(self, max_tracklets: int) -> dict[str, np.ndarray]:
        if not self.xy:
            return {
                "tracklet_xy": np.zeros((0, 2), dtype=np.float32),
                "tracklet_velocity": np.zeros((0, 2), dtype=np.float32),
                "tracklet_visibility": np.zeros((0,), dtype=np.float32),
                "tracklet_confidence": np.zeros((0,), dtype=np.float32),
                "tracklet_ids": np.zeros((0,), dtype=np.int64),
                "tracklet_view_ids": np.zeros((0,), dtype=np.int64),
                "tracklet_age": np.zeros((0,), dtype=np.float32),
            }
        xy = np.stack(self.xy).astype(np.float32)
        velocity = np.stack(self.velocity).astype(np.float32)
        visibility = np.asarray(self.visibility, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        track_ids = np.asarray(self.track_ids, dtype=np.int64)
        view_ids = np.asarray(self.view_ids, dtype=np.int64)
        age = np.asarray(self.age, dtype=np.float32)
        order = np.argsort(-(visibility * confidence))
        if int(max_tracklets) > 0:
            order = order[: int(max_tracklets)]
        return {
            "tracklet_xy": np.clip(xy[order], 0.0, 1.0).astype(np.float32),
            "tracklet_velocity": np.clip(velocity[order], -1.0, 1.0).astype(np.float32),
            "tracklet_visibility": np.clip(visibility[order], 0.0, 1.0).astype(np.float32),
            "tracklet_confidence": np.clip(confidence[order], 0.0, 1.0).astype(np.float32),
            "tracklet_ids": track_ids[order].astype(np.int64),
            "tracklet_view_ids": view_ids[order].astype(np.int64),
            "tracklet_age": np.clip(age[order], 0.0, 1.0).astype(np.float32),
        }


def _track_view_window(
    *,
    frames: dict[int, np.ndarray],
    key_step: int,
    step_ids: list[int],
    view_id: int,
    seeds_xy_norm: np.ndarray,
    seed_conf: np.ndarray,
    source_code: int,
    confidence_decay: float,
    max_error_px: float,
    rows_by_step: dict[int, _TrackRows],
) -> int:
    if seeds_xy_norm.shape[0] == 0 or key_step not in frames:
        return 0
    first_rgb = _as_uint8_rgb(frames[key_step])
    h, w = first_rgb.shape[:2]
    prev_gray = _gray(first_rgb)
    pts = np.stack(
        [
            seeds_xy_norm[:, 0] * float(max(w - 1, 1)),
            seeds_xy_norm[:, 1] * float(max(h - 1, 1)),
        ],
        axis=-1,
    ).astype(np.float32)
    alive = np.ones((pts.shape[0],), dtype=bool)
    prev_norm = np.clip(seeds_xy_norm.astype(np.float32), 0.0, 1.0)
    track_ids = np.asarray(
        [int(key_step) * 100000 + int(view_id) * 10000 + int(source_code) * 1000 + idx for idx in range(pts.shape[0])],
        dtype=np.int64,
    )
    emitted = 0
    for age_idx, step_id in enumerate(step_ids):
        if step_id not in frames:
            continue
        if age_idx == 0:
            cur_norm = prev_norm.copy()
            error = np.zeros((pts.shape[0],), dtype=np.float32)
            status = alive.copy()
            cur_gray = prev_gray
        else:
            cur_rgb = _as_uint8_rgb(frames[step_id])
            cur_gray = _gray(cur_rgb)
            cv2 = _cv2()
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                cur_gray,
                pts.reshape(-1, 1, 2),
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_pts is None or st is None:
                break
            next_pts = next_pts.reshape(-1, 2).astype(np.float32)
            err = np.zeros((pts.shape[0],), dtype=np.float32) if err is None else err.reshape(-1).astype(np.float32)
            status = (st.reshape(-1) > 0) & alive
            status &= next_pts[:, 0] >= 0.0
            status &= next_pts[:, 0] <= float(max(w - 1, 1))
            status &= next_pts[:, 1] >= 0.0
            status &= next_pts[:, 1] <= float(max(h - 1, 1))
            status &= err <= float(max_error_px)
            pts[status] = next_pts[status]
            alive &= status
            cur_norm = np.stack(
                [pts[:, 0] / float(max(w - 1, 1)), pts[:, 1] / float(max(h - 1, 1))],
                axis=-1,
            ).astype(np.float32)
            error = err
        if not alive.any():
            break
        age_norm = float(age_idx) / float(max(len(step_ids) - 1, 1))
        error_conf = np.exp(-np.clip(error, 0.0, float(max_error_px)) / max(float(max_error_px), 1.0))
        for idx in np.where(alive)[0]:
            conf = float(seed_conf[idx]) * float(error_conf[idx]) * float(confidence_decay ** age_idx)
            if conf <= 0.0:
                continue
            velocity = cur_norm[idx] - prev_norm[idx]
            rows_by_step[int(step_id)].add(
                xy=cur_norm[idx],
                velocity=velocity,
                visibility=1.0,
                confidence=conf,
                track_id=int(track_ids[idx]),
                view_id=int(view_id),
                age=age_norm,
            )
            emitted += 1
        prev_gray = cur_gray
        prev_norm = cur_norm.copy()
    return emitted


def _iter_segment_ids(total_segments: int, segment_indices: list[int] | None, shard_index: int, num_shards: int):
    selected = list(range(int(total_segments))) if segment_indices is None else [int(value) for value in segment_indices]
    for order, segment_id in enumerate(selected):
        if int(num_shards) > 1 and (order % int(num_shards)) != int(shard_index):
            continue
        yield int(segment_id)


def _merge_and_save(
    *,
    output_root: Path,
    proposal_root: Path | None,
    split: str,
    step_id: int,
    rows: _TrackRows,
    max_tracklets: int,
    skip_existing_tracklets: bool,
) -> tuple[int, bool]:
    output_split = output_root / split
    output_split.mkdir(parents=True, exist_ok=True)
    out_path = output_split / f"episode_{int(step_id):07d}.npz"
    existing = _load_sidecar_payload(out_path.parent.parent, split, step_id) if out_path.exists() else {}
    if skip_existing_tracklets and all(key in existing for key in _TRACKLET_KEYS):
        return int(existing.get("tracklet_confidence", np.zeros((0,), dtype=np.float32)).shape[0]), True
    payload = dict(existing)
    proposal_payload = _load_sidecar_payload(proposal_root, split, step_id)
    for key in _PROPOSAL_KEYS:
        if key in proposal_payload and key not in payload:
            payload[key] = proposal_payload[key]
    track_payload = rows.to_arrays(max_tracklets)
    payload.update(track_payload)
    if payload:
        _atomic_savez_compressed(out_path, **payload)
    return int(track_payload["tracklet_confidence"].shape[0]), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--backend", choices=("zip", "dir"), default="zip")
    parser.add_argument("--split", default="training")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--proposal-root", default=None, help="Optional proposal sidecar root used for object seeds and merge.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--views", choices=("static", "gripper", "both"), default="both")
    parser.add_argument("--segment-indices", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--keyframe-stride", type=int, default=16)
    parser.add_argument("--keyframe-offset", type=int, default=0)
    parser.add_argument("--window-forward", type=int, default=15)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--seeds-per-view", type=int, default=32)
    parser.add_argument("--proposal-seed-fraction", type=float, default=0.5)
    parser.add_argument("--sam-seed-fraction", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--require-proposal-keyframe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip keyframes without proposal sidecars instead of falling back to grid-only tracks.",
    )
    parser.add_argument("--max-tracklets-per-frame", type=int, default=96)
    parser.add_argument("--klt-max-error-px", type=float, default=18.0)
    parser.add_argument("--confidence-decay", type=float, default=0.985)
    parser.add_argument("--skip-existing-tracklets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if int(args.keyframe_stride) < 1:
        raise ValueError("--keyframe-stride must be >= 1.")
    if int(args.keyframe_offset) < 0 or int(args.keyframe_offset) >= int(args.keyframe_stride):
        raise ValueError("--keyframe-offset must satisfy 0 <= keyframe_offset < keyframe_stride.")
    if int(args.window_forward) < 0:
        raise ValueError("--window-forward must be >= 0.")
    proposal_seed_fraction_arg = args.proposal_seed_fraction if args.sam_seed_fraction is None else args.sam_seed_fraction
    proposal_seed_fraction = float(np.clip(float(proposal_seed_fraction_arg), 0.0, 1.0))
    proposal_root = None if args.proposal_root is None else Path(args.proposal_root)
    output_root = Path(args.output_root)
    segment_indices = None
    if args.segment_indices:
        segment_indices = [int(part) for part in str(args.segment_indices).split(",") if part.strip()]

    CalvinLangSegmentDataset = _calvin_dataset_cls()
    dataset = CalvinLangSegmentDataset(
        root=args.calvin_root,
        split=args.split,
        action_horizon=int(args.action_horizon),
        backend=args.backend,
        use_wrist_rgb=args.views in {"gripper", "both"},
        sample_within_segment=False,
    )
    manifest: list[dict[str, Any]] = []
    segment_count = 0
    total_saved_frames = 0
    total_tracklets = 0
    try:
        for segment_id in _iter_segment_ids(len(dataset.segments), segment_indices, int(args.shard_index), int(args.num_shards)):
            if args.max_segments is not None and segment_count >= int(args.max_segments):
                break
            segment = dataset.segments[int(segment_id)]
            rows_by_step: dict[int, _TrackRows] = defaultdict(_TrackRows.empty)
            step_start = int(segment.start)
            step_end = int(segment.end)
            keys = ["rgb_static"]
            if args.views in {"gripper", "both"}:
                keys.append("rgb_gripper")
            segment_frames: dict[int, dict[str, np.ndarray]] = {}
            # Keep the memory footprint bounded to one episode/segment.
            for step_id in range(step_start, step_end):
                segment_frames[int(step_id)] = dict(dataset.reader.read_npz(int(step_id), keys=keys))

            keyframes = [
                step_id
                for local_idx, step_id in enumerate(range(step_start, step_end))
                if ((local_idx - int(args.keyframe_offset)) % int(args.keyframe_stride)) == 0
            ]
            emitted_tracks = 0
            skipped_keyframes_without_proposal = 0
            for key_step in keyframes:
                proposal_payload = _load_sidecar_payload(proposal_root, args.split, int(key_step))
                has_keyframe_proposals = (
                    "proposal_centers_xy" in proposal_payload
                    and np.asarray(proposal_payload.get("proposal_centers_xy")).reshape(-1, 2).shape[0] > 0
                )
                if bool(args.require_proposal_keyframe) and not has_keyframe_proposals:
                    skipped_keyframes_without_proposal += 1
                    continue
                window_steps = list(range(int(key_step), min(int(key_step) + int(args.window_forward) + 1, step_end)))
                view_names = []
                if args.views in {"static", "both"}:
                    view_names.append("static")
                if args.views in {"gripper", "both"}:
                    view_names.append("gripper")
                for view_name in view_names:
                    if view_name == "gripper" and segment_frames.get(int(key_step), {}).get("rgb_gripper") is None:
                        continue
                    view_id = _VIEW_IDS[view_name]
                    proposal_count = int(round(int(args.seeds_per_view) * proposal_seed_fraction))
                    grid_count = max(int(args.seeds_per_view) - proposal_count, 0)
                    proposal_xy, proposal_conf = _proposal_seed_points(proposal_payload, view_id=view_id, max_count=proposal_count)
                    if proposal_xy.shape[0] < proposal_count:
                        grid_count += proposal_count - proposal_xy.shape[0]
                    grid_xy, grid_conf = _grid_seed_points(grid_count)
                    seeds_xy = (
                        np.concatenate([proposal_xy, grid_xy], axis=0)
                        if grid_xy.size or proposal_xy.size
                        else np.zeros((0, 2), dtype=np.float32)
                    )
                    seed_conf = (
                        np.concatenate([proposal_conf, grid_conf], axis=0)
                        if grid_conf.size or proposal_conf.size
                        else np.zeros((0,), dtype=np.float32)
                    )
                    if seeds_xy.shape[0] == 0:
                        continue
                    frame_images = {
                        step_id: segment_frames[step_id]["rgb_static" if view_name == "static" else "rgb_gripper"]
                        for step_id in window_steps
                        if ("rgb_static" if view_name == "static" else "rgb_gripper") in segment_frames.get(step_id, {})
                        and segment_frames[step_id].get("rgb_static" if view_name == "static" else "rgb_gripper") is not None
                    }
                    source_code = 1 if proposal_xy.shape[0] > 0 else 2
                    emitted_tracks += _track_view_window(
                        frames=frame_images,
                        key_step=int(key_step),
                        step_ids=window_steps,
                        view_id=view_id,
                        seeds_xy_norm=seeds_xy,
                        seed_conf=seed_conf,
                        source_code=source_code,
                        confidence_decay=float(args.confidence_decay),
                        max_error_px=float(args.klt_max_error_px),
                        rows_by_step=rows_by_step,
                    )

            saved_frames = 0
            saved_tracklets = 0
            skipped_existing = 0
            for step_id, rows in rows_by_step.items():
                count, skipped = _merge_and_save(
                    output_root=output_root,
                    proposal_root=proposal_root,
                    split=args.split,
                    step_id=int(step_id),
                    rows=rows,
                    max_tracklets=int(args.max_tracklets_per_frame),
                    skip_existing_tracklets=bool(args.skip_existing_tracklets),
                )
                skipped_existing += int(bool(skipped))
                if not skipped and count > 0:
                    saved_frames += 1
                    saved_tracklets += count
            segment_count += 1
            total_saved_frames += saved_frames
            total_tracklets += saved_tracklets
            item = {
                "segment_id": int(segment_id),
                "keyframes": len(keyframes),
                "saved_frames": int(saved_frames),
                "tracklet_count": int(saved_tracklets),
                "emitted_track_observations": int(emitted_tracks),
                "skipped_existing": int(skipped_existing),
                "skipped_keyframes_without_proposal": int(skipped_keyframes_without_proposal),
                "shard_index": int(args.shard_index),
                "num_shards": int(args.num_shards),
            }
            manifest.append(item)
            if int(args.log_every) > 0 and segment_count % int(args.log_every) == 0:
                print(
                    json.dumps(
                        {
                            "progress_segments": int(segment_count),
                            "saved_frames": int(total_saved_frames),
                            "tracklet_count": int(total_tracklets),
                            "last_segment_id": int(segment_id),
                            "shard_index": int(args.shard_index),
                            "num_shards": int(args.num_shards),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        dataset.reader.close()

    manifest_name = f"tracklet_manifest_{args.split}.json"
    if int(args.num_shards) > 1:
        manifest_name = f"tracklet_manifest_{args.split}_shard{int(args.shard_index):03d}-of-{int(args.num_shards):03d}.json"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "segments": int(segment_count),
                "saved_frames": int(total_saved_frames),
                "tracklet_count": int(total_tracklets),
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "proposal_root": None if proposal_root is None else str(proposal_root),
                "keyframe_stride": int(args.keyframe_stride),
                "window_forward": int(args.window_forward),
                "proposal_seed_fraction": float(proposal_seed_fraction),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
