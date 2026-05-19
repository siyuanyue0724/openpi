#!/usr/bin/env python3
"""LEGACY/ARCHIVED offline SAM-to-PICF proposal sidecar generator.

Archived on 2026-05-18 after blind SAM proposals were rejected as too noisy for
the maintained PICF-AQR-OWM training path.  Keep this file only for historical
reproduction of SAM ablations.  Do not use it for current training launches.

The maintained proposal sidecar path is now contact/task guided and lives in:

    scripts/picf_contact_motion_sidecar_precompute.py

This legacy script intentionally keeps SAM out of the online PICF training graph.
It reads CALVIN frames, runs a frozen mask proposal generator, and writes
per-frame sidecar npz files containing optional `proposal_*` arrays.  Those
arrays remain a generic sidecar contract, but blind SAM is no longer an active
generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

_VIEW_IDS = {"static": 0, "gripper": 1}
_SOURCE_SAM_2D = 5


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


def _load_sam_generator(args: argparse.Namespace) -> Any:
    if args.segment_anything_repo:
        repo = Path(args.segment_anything_repo).expanduser().resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
    try:
        from segment_anything import SamAutomaticMaskGenerator
        from segment_anything import sam_model_registry
    except Exception as exc:  # pragma: no cover - dependency-dependent.
        raise RuntimeError(
            "segment_anything is not importable. Install it or pass "
            "--segment-anything-repo /path/to/segment-anything for offline precompute."
        ) from exc
    if not args.sam_checkpoint:
        raise ValueError("--sam-checkpoint is required unless --dry-run is set.")
    import torch

    device = args.device
    model = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    model.to(device=device)
    model.eval()
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )


def _proposal_arrays_from_masks(
    masks: list[dict[str, Any]],
    *,
    image_hw: tuple[int, int],
    view_id: int,
    max_proposals: int,
) -> dict[str, np.ndarray]:
    h, w = image_hw
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    for mask in masks:
        bbox_xywh = np.asarray(mask.get("bbox", ()), dtype=np.float32).reshape(-1)
        if bbox_xywh.shape[0] != 4:
            continue
        x, y, bw, bh = bbox_xywh.tolist()
        if bw <= 1.0 or bh <= 1.0:
            continue
        xyxy = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        norm = np.array([w, h, w, h], dtype=np.float32)
        xyxy_norm = np.clip(xyxy / np.maximum(norm, 1.0), 0.0, 1.0)
        center = 0.5 * (xyxy_norm[:2] + xyxy_norm[2:])
        iou = float(mask.get("predicted_iou", 0.0))
        stability = float(mask.get("stability_score", 0.0))
        # Geometric mean requires both the mask decoder and threshold stability
        # to agree; a high value from only one source is not enough.
        objectness = float(np.sqrt(max(iou, 0.0) * max(stability, 0.0)))
        rows.append((objectness, center.astype(np.float32), xyxy_norm.astype(np.float32)))
    rows.sort(key=lambda row: row[0], reverse=True)
    if max_proposals > 0:
        rows = rows[:max_proposals]
    if not rows:
        return {
            "proposal_centers_xy": np.zeros((0, 2), dtype=np.float32),
            "proposal_boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
            "proposal_objectness": np.zeros((0,), dtype=np.float32),
            "proposal_view_ids": np.zeros((0,), dtype=np.int64),
            "proposal_source_ids": np.zeros((0,), dtype=np.int64),
        }
    objectness = np.asarray([row[0] for row in rows], dtype=np.float32)
    centers = np.stack([row[1] for row in rows]).astype(np.float32)
    boxes = np.stack([row[2] for row in rows]).astype(np.float32)
    return {
        "proposal_centers_xy": centers,
        "proposal_boxes_xyxy": boxes,
        "proposal_objectness": np.clip(objectness, 0.0, 1.0).astype(np.float32),
        "proposal_view_ids": np.full((len(rows),), int(view_id), dtype=np.int64),
        "proposal_source_ids": np.full((len(rows),), _SOURCE_SAM_2D, dtype=np.int64),
    }


def _concat_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = (
        "proposal_centers_xy",
        "proposal_boxes_xyxy",
        "proposal_objectness",
        "proposal_view_ids",
        "proposal_source_ids",
    )
    out: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [payload[key] for payload in payloads if key in payload and payload[key].size > 0]
        if arrays:
            out[key] = np.concatenate(arrays, axis=0)
        elif key in {"proposal_centers_xy"}:
            out[key] = np.zeros((0, 2), dtype=np.float32)
        elif key in {"proposal_boxes_xyxy"}:
            out[key] = np.zeros((0, 4), dtype=np.float32)
        elif key in {"proposal_objectness"}:
            out[key] = np.zeros((0,), dtype=np.float32)
        else:
            out[key] = np.zeros((0,), dtype=np.int64)
    return out


def _save_preview(
    *,
    preview_root: Path | None,
    split: str,
    step_id: int,
    view_name: str,
    image: np.ndarray,
    payload: dict[str, np.ndarray],
) -> str | None:
    if preview_root is None:
        return None
    preview_dir = preview_root / split
    preview_dir.mkdir(parents=True, exist_ok=True)
    rgb = _as_uint8_rgb(image)
    pil = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(pil)
    boxes = np.asarray(payload.get("proposal_boxes_xyxy", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
    centers = np.asarray(payload.get("proposal_centers_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    objectness = np.asarray(payload.get("proposal_objectness", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    h, w = rgb.shape[:2]
    for idx in range(min(int(boxes.shape[0]), 32)):
        score = float(objectness[idx]) if idx < objectness.shape[0] else 0.0
        x0, y0, x1, y1 = np.clip(boxes[idx], 0.0, 1.0).tolist()
        box_px = (
            int(round(x0 * max(w - 1, 1))),
            int(round(y0 * max(h - 1, 1))),
            int(round(x1 * max(w - 1, 1))),
            int(round(y1 * max(h - 1, 1))),
        )
        color = (40, 220, 90) if score >= 0.85 else (80, 180, 120)
        draw.rectangle(box_px, outline=color, width=1)
        if idx < centers.shape[0]:
            cx, cy = np.clip(centers[idx], 0.0, 1.0).tolist()
            cx_px = int(round(cx * max(w - 1, 1)))
            cy_px = int(round(cy * max(h - 1, 1)))
            draw.ellipse((cx_px - 2, cy_px - 2, cx_px + 2, cy_px + 2), outline=color, width=1)
        draw.text((box_px[0] + 1, max(box_px[1] - 10, 0)), f"{idx}:{score:.2f}", fill=color)
    header = f"SAM proposals | step {int(step_id)} | {view_name} | n={int(boxes.shape[0])}"
    draw.rectangle((0, 0, min(w, max(360, 8 * len(header))), 18), fill=(0, 0, 0))
    draw.text((4, 3), header, fill=(255, 255, 255))
    out = preview_dir / f"episode_{int(step_id):07d}__{view_name}.png"
    pil.save(out)
    return str(out)


def _iter_step_ids(
    dataset: Any,
    segment_indices: list[int] | None,
    max_frames: int | None,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
    frame_stride: int = 1,
    frame_offset: int = 0,
):
    count = 0
    seen = 0
    selected = segment_indices if segment_indices is not None else list(range(len(dataset.segments)))
    for segment_id in selected:
        if int(num_shards) > 1 and (seen % int(num_shards)) != int(shard_index):
            seen += 1
            continue
        seen += 1
        segment = dataset.segments[int(segment_id)]
        for local_idx, step_id in enumerate(range(int(segment.start), int(segment.end))):
            if int(frame_stride) > 1 and ((local_idx - int(frame_offset)) % int(frame_stride)) != 0:
                continue
            yield int(step_id)
            count += 1
            if max_frames is not None and count >= int(max_frames):
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--backend", choices=("zip", "dir"), default="zip")
    parser.add_argument("--split", default="training")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--views", choices=("static", "gripper", "both"), default="both")
    parser.add_argument("--segment-indices", default=None, help="Comma-separated segment indices, default all.")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based segment shard index. Segment-level sharding keeps each episode contiguous.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Number of segment shards for parallel precompute.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Generate only every Nth frame within each segment. Use for bounded offline coverage.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Frame-stride phase within each segment. Must satisfy 0 <= offset < stride.",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Emit progress every N processed frames.")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip frames whose sidecar npz already exists. Enabled by default for restartable large runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--segment-anything-repo", default="/tmp/picf_sam_code/segment-anything")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-proposals-per-view", type=int, default=32)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.90)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--min-mask-region-area", type=int, default=32)
    parser.add_argument(
        "--preview-root",
        default=None,
        help="Optional directory for static/gripper proposal preview PNGs. This is diagnostic-only.",
    )
    args = parser.parse_args()

    segment_indices = None
    if args.segment_indices:
        segment_indices = [int(part) for part in str(args.segment_indices).split(",") if part.strip()]
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if int(args.frame_stride) < 1:
        raise ValueError("--frame-stride must be >= 1.")
    if int(args.frame_offset) < 0 or int(args.frame_offset) >= int(args.frame_stride):
        raise ValueError("--frame-offset must satisfy 0 <= frame_offset < frame_stride.")

    CalvinLangSegmentDataset = _calvin_dataset_cls()
    dataset = CalvinLangSegmentDataset(
        root=args.calvin_root,
        split=args.split,
        action_horizon=int(args.action_horizon),
        backend=args.backend,
        use_wrist_rgb=args.views in {"gripper", "both"},
        sample_within_segment=False,
    )
    output_split = Path(args.output_root) / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    preview_root = None if args.preview_root is None else Path(args.preview_root)
    generator = None if args.dry_run else _load_sam_generator(args)
    manifest: list[dict[str, Any]] = []
    try:
        for step_id in _iter_step_ids(
            dataset,
            segment_indices,
            args.max_frames,
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
            frame_stride=int(args.frame_stride),
            frame_offset=int(args.frame_offset),
        ):
            path = output_split / f"episode_{int(step_id):07d}.npz"
            if bool(args.skip_existing) and path.exists() and not args.dry_run:
                with np.load(path) as existing:
                    proposal_count = int(existing.get("proposal_objectness", np.zeros((0,), dtype=np.float32)).shape[0])
                manifest.append(
                    {
                        "step_id": int(step_id),
                        "output": str(path),
                        "proposal_count": proposal_count,
                        "views": [],
                        "previews": [],
                        "dry_run": bool(args.dry_run),
                        "skipped_existing": True,
                    }
                )
                continue
            keys = ["rgb_static"]
            if args.views in {"gripper", "both"}:
                keys.append("rgb_gripper")
            frame = dataset.reader.read_npz(step_id, keys=keys)
            views: list[tuple[str, np.ndarray]] = []
            if args.views in {"static", "both"} and "rgb_static" in frame:
                views.append(("static", frame["rgb_static"]))
            if args.views in {"gripper", "both"} and frame.get("rgb_gripper") is not None:
                views.append(("gripper", frame["rgb_gripper"]))
            payloads: list[dict[str, np.ndarray]] = []
            previews: list[str] = []
            for view_name, image in views:
                rgb = _as_uint8_rgb(image)
                if args.dry_run:
                    masks: list[dict[str, Any]] = []
                else:
                    masks = generator.generate(rgb)
                view_payload = _proposal_arrays_from_masks(
                    masks,
                    image_hw=(int(rgb.shape[0]), int(rgb.shape[1])),
                    view_id=_VIEW_IDS[view_name],
                    max_proposals=int(args.max_proposals_per_view),
                )
                payloads.append(view_payload)
                preview = _save_preview(
                    preview_root=preview_root,
                    split=args.split,
                    step_id=int(step_id),
                    view_name=view_name,
                    image=rgb,
                    payload=view_payload,
                )
                if preview is not None:
                    previews.append(preview)
            payload = _concat_payloads(payloads)
            if not args.dry_run:
                np.savez_compressed(path, **payload)
            manifest.append(
                {
                    "step_id": int(step_id),
                    "output": str(path),
                    "proposal_count": int(payload["proposal_objectness"].shape[0]),
                    "views": [name for name, _ in views],
                    "previews": previews,
                    "dry_run": bool(args.dry_run),
                    "skipped_existing": False,
                }
            )
            if int(args.log_every) > 0 and len(manifest) % int(args.log_every) == 0:
                processed_total = sum(int(item["proposal_count"]) for item in manifest)
                skipped_total = sum(1 for item in manifest if bool(item.get("skipped_existing", False)))
                print(
                    json.dumps(
                        {
                            "progress_frames": len(manifest),
                            "proposal_count": processed_total,
                            "skipped_existing": skipped_total,
                            "last_step_id": int(step_id),
                            "shard_index": int(args.shard_index),
                            "num_shards": int(args.num_shards),
                            "frame_stride": int(args.frame_stride),
                            "frame_offset": int(args.frame_offset),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        dataset.reader.close()

    manifest_name = f"manifest_{args.split}.json"
    if int(args.num_shards) > 1:
        manifest_name = f"manifest_{args.split}_shard{int(args.shard_index):03d}-of-{int(args.num_shards):03d}.json"
    manifest_path = Path(args.output_root) / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(int(item["proposal_count"]) for item in manifest)
    print(
        json.dumps(
            {
                "frames": len(manifest),
                "proposal_count": total,
                "output_root": str(Path(args.output_root)),
                "manifest": str(manifest_path),
                "dry_run": bool(args.dry_run),
                "shard_index": int(args.shard_index),
                "num_shards": int(args.num_shards),
                "skip_existing": bool(args.skip_existing),
                "frame_stride": int(args.frame_stride),
                "frame_offset": int(args.frame_offset),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
