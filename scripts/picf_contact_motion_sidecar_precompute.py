#!/usr/bin/env python3
"""Precompute contact/task proposal sidecars for PICF anchor diagnostics.

This script writes the existing MVTrack `proposal_*` sidecar contract without
changing the CALVIN dataset.  The default mode is causal at the frame level: it
uses the current RGB-D frame, current end-effector pose, and language-derived
object color/contact heuristics.  It does not use future frames as online input.

The generated proposals are weak scaffolds for short anchor diagnostics, not
ground-truth masks.  They should remain optional typed evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from PIL import ImageDraw

from picf_object3d_slot_video import _foreground_score
from picf_object3d_slot_video import _frame_to_points
from picf_object3d_slot_video import _load_cameras


def _load_annotations(calvin_root: Path) -> tuple[list[str], list[tuple[int, int]]]:
    ann_path = calvin_root / "training" / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    texts = [str(text) for text in ann["language"]["ann"]]
    intervals = [tuple(int(x) for x in interval) for interval in ann["info"]["indx"]]
    return texts, intervals


def _atomic_savez_compressed(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _select_mask_samples(
    pixels: np.ndarray,
    scores: np.ndarray,
    *,
    max_samples: int,
) -> np.ndarray:
    """Deterministically keep high-score but spatially spread mask samples."""

    pixels = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    count = min(int(pixels.shape[0]), int(scores.shape[0]))
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    pixels = pixels[:count]
    scores = np.nan_to_num(scores[:count], nan=0.0, posinf=0.0, neginf=0.0)
    take = min(max(int(max_samples), 1), count)
    if count <= take:
        return np.arange(count, dtype=np.int64)
    score_norm = scores / max(float(scores.max(initial=0.0)), 1e-6)
    chosen = [int(np.argmax(score_norm))]
    min_dist = np.linalg.norm(pixels - pixels[chosen[0] : chosen[0] + 1], axis=1)
    while len(chosen) < take:
        score = min_dist * (0.25 + score_norm)
        score[np.asarray(chosen, dtype=np.int64)] = -1.0
        nxt = int(np.argmax(score))
        if score[nxt] < 0.0:
            break
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(pixels - pixels[nxt : nxt + 1], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def _proposal_from_static_scores(
    *,
    pixels_static: np.ndarray,
    scores_static: np.ndarray,
    image_hw: tuple[int, int],
    top_fraction: float,
    min_top_points: int,
    min_score: float,
    box_pad_px: float,
    max_proposals: int,
    component_radius_px: float,
    component_min_points: int,
    box_percentile_low: float,
    box_percentile_high: float,
    mask_samples_per_proposal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None:
    scores = np.asarray(scores_static, dtype=np.float32).reshape(-1)
    pixels = np.asarray(pixels_static, dtype=np.float32).reshape(-1, 2)
    count = min(int(scores.shape[0]), int(pixels.shape[0]))
    if count <= 0:
        return None
    scores = np.nan_to_num(scores[:count], nan=0.0, posinf=0.0, neginf=0.0)
    pixels = pixels[:count]
    max_score = float(scores.max(initial=0.0))
    if max_score < float(min_score):
        return None
    top_count = max(int(min_top_points), int(np.ceil(float(top_fraction) * float(count))))
    top_count = min(max(top_count, 1), count)
    order = np.argsort(-scores)[:top_count]
    top_scores = np.clip(scores[order], 0.0, 1.0)
    keep = top_scores >= max(float(min_score) * 0.50, 1e-6)
    if not np.any(keep):
        return None
    selected = order[keep]
    selected_scores = top_scores[keep]
    selected_pixels = pixels[selected]
    height, width = [int(v) for v in image_hw]
    denom = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)

    radius = max(float(component_radius_px), 1.0)
    min_component = max(int(component_min_points), 1)
    unused = np.ones((selected_pixels.shape[0],), dtype=bool)
    components: list[np.ndarray] = []
    order_local = np.argsort(-selected_scores)
    for seed in order_local.tolist():
        if not bool(unused[seed]):
            continue
        queue = [int(seed)]
        unused[seed] = False
        comp: list[int] = []
        while queue:
            idx = queue.pop()
            comp.append(idx)
            remaining = np.nonzero(unused)[0]
            if remaining.size == 0:
                continue
            dist = np.linalg.norm(selected_pixels[remaining] - selected_pixels[idx][None, :], axis=1)
            near = remaining[dist <= radius]
            if near.size:
                unused[near] = False
                queue.extend(int(x) for x in near.tolist())
        if len(comp) >= min_component:
            components.append(np.asarray(comp, dtype=np.int64))

    if not components:
        components = [np.arange(selected_pixels.shape[0], dtype=np.int64)]

    proposal_rows: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = []
    total_selected_mass = float(np.sum(selected_scores) + 1e-6)
    lo_pct = min(max(float(box_percentile_low), 0.0), 49.0)
    hi_pct = max(min(float(box_percentile_high), 100.0), 51.0)
    if hi_pct <= lo_pct:
        lo_pct, hi_pct = 8.0, 92.0
    for comp in components:
        comp_pixels = selected_pixels[comp]
        comp_scores = selected_scores[comp]
        weights = comp_scores + 1e-4
        center_px = np.sum(comp_pixels * weights[:, None], axis=0) / float(np.sum(weights))
        lo = np.percentile(comp_pixels, lo_pct, axis=0) - float(box_pad_px)
        hi = np.percentile(comp_pixels, hi_pct, axis=0) + float(box_pad_px)
        center_xy = np.clip(center_px / denom, 0.0, 1.0).astype(np.float32)
        box_xyxy = np.concatenate([lo / denom, hi / denom], axis=0)
        box_xyxy = np.clip(box_xyxy, 0.0, 1.0).astype(np.float32)
        wh = np.maximum(box_xyxy[2:] - box_xyxy[:2], 1e-4)
        area = float(wh[0] * wh[1])
        compactness = float(np.clip(0.08 / max(area, 1e-4), 0.20, 1.0))
        mass_fraction = float(np.clip(np.sum(comp_scores) / total_selected_mass, 0.0, 1.0))
        objectness = float(
            np.clip(
                np.sqrt(float(comp_scores.max()) * float(comp_scores.mean()))
                * compactness
                * (0.5 + 0.5 * mass_fraction),
                0.0,
                1.0,
            )
        )
        sample_idx = _select_mask_samples(comp_pixels, comp_scores, max_samples=int(mask_samples_per_proposal))
        mask_xy = np.clip(comp_pixels[sample_idx] / denom, 0.0, 1.0).astype(np.float32)
        mask_weights = np.clip(comp_scores[sample_idx], 0.0, 1.0).astype(np.float32)
        if mask_weights.size and float(mask_weights.max(initial=0.0)) > 1e-6:
            mask_weights = mask_weights / float(mask_weights.max())
        proposal_rows.append(
            (
                objectness,
                center_xy,
                box_xyxy,
                mask_xy,
                mask_weights,
                {
                    "component_points": float(comp.shape[0]),
                    "mask_samples": float(mask_xy.shape[0]),
                    "component_score_mean": float(comp_scores.mean()),
                    "component_score_max": float(comp_scores.max()),
                    "component_mass_fraction": mass_fraction,
                    "box_area": area,
                    "objectness": objectness,
                },
            )
        )

    proposal_rows.sort(key=lambda item: item[0], reverse=True)
    proposal_rows = proposal_rows[: max(int(max_proposals), 1)]
    centers = np.stack([row[1] for row in proposal_rows], axis=0).astype(np.float32)
    boxes = np.stack([row[2] for row in proposal_rows], axis=0).astype(np.float32)
    objectness_values = np.asarray([row[0] for row in proposal_rows], dtype=np.float32)
    mask_parts = [row[3] for row in proposal_rows]
    weight_parts = [row[4] for row in proposal_rows]
    offsets = [0]
    for part in mask_parts:
        offsets.append(offsets[-1] + int(part.shape[0]))
    mask_xy = np.concatenate(mask_parts, axis=0).astype(np.float32) if mask_parts else np.zeros((0, 2), dtype=np.float32)
    mask_weights = (
        np.concatenate(weight_parts, axis=0).astype(np.float32) if weight_parts else np.zeros((0,), dtype=np.float32)
    )
    mask_offsets = np.asarray(offsets, dtype=np.int64)
    stats = {
        "max_score": max_score,
        "mean_selected_score": float(selected_scores.mean()),
        "selected_points": float(selected.shape[0]),
        "component_count": float(len(components)),
        "proposal_count": float(len(proposal_rows)),
        "mask_samples": float(mask_xy.shape[0]),
        "box_area": float(max(row[5]["box_area"] for row in proposal_rows)),
        "objectness": float(objectness_values.max(initial=0.0)),
    }
    return centers, boxes, objectness_values, mask_xy, mask_weights, mask_offsets, stats


def _save_preview(
    output_dir: Path,
    *,
    rgb_static: np.ndarray,
    proposal_center: np.ndarray,
    proposal_box: np.ndarray,
    proposal_mask_xy: np.ndarray | None = None,
    step_id: int,
    segment_id: int,
    text: str,
    stats: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(rgb_static, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    boxes = proposal_box.reshape(-1, 4).astype(float)
    centers = proposal_center.reshape(-1, 2).astype(float)
    mask_xy = None if proposal_mask_xy is None else proposal_mask_xy.reshape(-1, 2).astype(float)
    for rank, box in enumerate(boxes):
        color = (255, 80, 30) if rank == 0 else (70, 220, 100)
        xyxy = [box[0] * (w - 1), box[1] * (h - 1), box[2] * (w - 1), box[3] * (h - 1)]
        draw.rectangle(xyxy, outline=color, width=3 if rank == 0 else 2)
    for rank, (cx, cy) in enumerate(centers):
        color = (30, 220, 255) if rank == 0 else (250, 220, 60)
        x = cx * (w - 1)
        y = cy * (h - 1)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3 if rank == 0 else 2)
    if mask_xy is not None and mask_xy.size:
        max_points = min(int(mask_xy.shape[0]), 512)
        if mask_xy.shape[0] > max_points:
            idx = np.linspace(0, mask_xy.shape[0] - 1, max_points).astype(np.int64)
        else:
            idx = np.arange(mask_xy.shape[0], dtype=np.int64)
        for i in idx:
            x = float(mask_xy[i, 0]) * (w - 1)
            y = float(mask_xy[i, 1]) * (h - 1)
            draw.ellipse((x - 1.3, y - 1.3, x + 1.3, y + 1.3), fill=(255, 220, 30))
    draw.rectangle((0, 0, w, 38), fill=(255, 255, 255))
    draw.text((6, 4), f"segment={segment_id} step={step_id} obj={stats['objectness']:.3f}", fill=(0, 0, 0))
    draw.text((6, 20), text[:80], fill=(0, 0, 0))
    image.save(output_dir / f"segment_{segment_id:05d}_step_{step_id:07d}.png")


def _iter_selected_segments(
    *,
    texts: list[str],
    intervals: list[tuple[int, int]],
    prompts: list[str] | None,
    max_segments: int | None,
) -> list[tuple[int, str, tuple[int, int]]]:
    selected = []
    for segment_id, (text, interval) in enumerate(zip(texts, intervals)):
        if prompts:
            text_l = text.lower()
            if not any(prompt.lower() in text_l for prompt in prompts):
                continue
        selected.append((int(segment_id), str(text), tuple(int(x) for x in interval)))
        if max_segments is not None and len(selected) >= int(max_segments):
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--target-frames", type=int, default=1000)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--max-frames-per-segment", type=int, default=96)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--static-stride", type=int, default=4)
    parser.add_argument("--gripper-stride", type=int, default=2)
    parser.add_argument("--top-fraction", type=float, default=0.020)
    parser.add_argument("--min-top-points", type=int, default=24)
    parser.add_argument("--min-score", type=float, default=0.015)
    parser.add_argument("--box-pad-px", type=float, default=4.0)
    parser.add_argument("--max-proposals-per-frame", type=int, default=3)
    parser.add_argument("--component-radius-px", type=float, default=10.0)
    parser.add_argument("--component-min-points", type=int, default=6)
    parser.add_argument("--box-percentile-low", type=float, default=12.0)
    parser.add_argument("--box-percentile-high", type=float, default=88.0)
    parser.add_argument("--mask-samples-per-proposal", type=int, default=96)
    parser.add_argument("--source-id", type=int, default=8)
    parser.add_argument(
        "--use-segment-ee-path",
        action="store_true",
        help=(
            "Diagnostic only: use the whole language segment end-effector path to score contact. "
            "Default is current-frame causal scoring, which avoids future-input leakage."
        ),
    )
    parser.add_argument("--preview-count", type=int, default=32)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.split != "training":
        raise ValueError("This diagnostic currently expects CALVIN training split files.")
    calvin_root = Path(args.calvin_root)
    output_root = Path(args.output_root)
    output_split = output_root / args.split
    preview_dir = output_root / "previews"
    output_split.mkdir(parents=True, exist_ok=True)
    cameras = _load_cameras(calvin_root)
    texts, intervals = _load_annotations(calvin_root)
    selected_segments = _iter_selected_segments(
        texts=texts,
        intervals=intervals,
        prompts=args.prompts,
        max_segments=args.max_segments,
    )
    if not selected_segments:
        raise RuntimeError("No CALVIN language segments selected.")

    manifest_items = []
    written = 0
    skipped = 0
    attempted = 0
    preview_written = 0
    used_segment_ids: list[int] = []
    for segment_id, text, (start, end) in selected_segments:
        if written >= int(args.target_frames):
            break
        frame_ids = list(range(int(start), min(int(end) + 1, int(start) + int(args.max_frames_per_segment))))
        if not frame_ids:
            continue
        if args.use_segment_ee_path:
            ee_positions = []
            for step_id in frame_ids:
                path = calvin_root / args.split / f"episode_{step_id:07d}.npz"
                if not path.exists():
                    continue
                try:
                    item = _frame_to_points(calvin_root, path, cameras, args.static_stride, args.gripper_stride)
                except Exception:
                    continue
                ee_positions.append(item[7])
            ee_path = np.stack(ee_positions, axis=0).astype(np.float32) if ee_positions else None
        else:
            ee_path = None
        segment_written = 0
        segment_attempted = 0
        for step_id in frame_ids:
            if written >= int(args.target_frames):
                break
            sidecar_path = output_split / f"episode_{step_id:07d}.npz"
            if sidecar_path.exists() and bool(args.skip_existing):
                skipped += 1
                continue
            frame_path = calvin_root / args.split / f"episode_{step_id:07d}.npz"
            if not frame_path.exists():
                continue
            segment_attempted += 1
            attempted += 1
            try:
                rgb_static, _rgb_gripper, xyz, rgb, view_ids, pixels_static, static_count, ee_pos = _frame_to_points(
                    calvin_root,
                    frame_path,
                    cameras,
                    args.static_stride,
                    args.gripper_stride,
                )
            except Exception as exc:
                manifest_items.append(
                    {
                        "segment_id": int(segment_id),
                        "step_id": int(step_id),
                        "status": "frame_error",
                        "error": str(exc),
                    }
                )
                continue
            foreground = _foreground_score(xyz, rgb, view_ids, ee_pos, text, ee_path=ee_path)
            proposal = _proposal_from_static_scores(
                pixels_static=pixels_static,
                scores_static=foreground[:static_count],
                image_hw=tuple(int(v) for v in rgb_static.shape[:2]),
                top_fraction=float(args.top_fraction),
                min_top_points=int(args.min_top_points),
                min_score=float(args.min_score),
                box_pad_px=float(args.box_pad_px),
                max_proposals=int(args.max_proposals_per_frame),
                component_radius_px=float(args.component_radius_px),
                component_min_points=int(args.component_min_points),
                box_percentile_low=float(args.box_percentile_low),
                box_percentile_high=float(args.box_percentile_high),
                mask_samples_per_proposal=int(args.mask_samples_per_proposal),
            )
            if proposal is None:
                manifest_items.append(
                    {
                        "segment_id": int(segment_id),
                        "step_id": int(step_id),
                        "status": "no_proposal",
                    }
                )
                continue
            centers, boxes, objectness, mask_xy, mask_weights, mask_offsets, stats = proposal
            payload = {
                "proposal_centers_xy": centers.astype(np.float32),
                "proposal_boxes_xyxy": boxes.astype(np.float32),
                "proposal_objectness": np.asarray(objectness, dtype=np.float32).reshape(-1)[: int(centers.shape[0])],
                "proposal_view_ids": np.zeros((int(centers.shape[0]),), dtype=np.int64),
                "proposal_source_ids": np.full((int(centers.shape[0]),), int(args.source_id), dtype=np.int64),
                "proposal_mask_xy": mask_xy.astype(np.float32),
                "proposal_mask_weights": mask_weights.astype(np.float32),
                "proposal_mask_offsets": mask_offsets.astype(np.int64),
            }
            _atomic_savez_compressed(sidecar_path, **payload)
            if preview_written < int(args.preview_count):
                _save_preview(
                    preview_dir,
                    rgb_static=rgb_static,
                    proposal_center=centers,
                    proposal_box=boxes,
                    proposal_mask_xy=mask_xy,
                    step_id=step_id,
                    segment_id=segment_id,
                    text=text,
                    stats=stats,
                )
                preview_written += 1
            written += 1
            segment_written += 1
            manifest_items.append(
                {
                    "segment_id": int(segment_id),
                    "step_id": int(step_id),
                    "status": "written",
                    "objectness": objectness,
                    **stats,
                }
            )
        if segment_written > 0:
            used_segment_ids.append(int(segment_id))
        print(
            json.dumps(
                {
                    "segment_id": int(segment_id),
                    "prompt": text,
                    "attempted": int(segment_attempted),
                    "written_total": int(written),
                    "target_frames": int(args.target_frames),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "calvin_segment_indices.txt").write_text(
        ",".join(str(value) for value in sorted(set(used_segment_ids))),
        encoding="utf-8",
    )
    manifest = {
        "output_root": str(output_root),
        "split": str(args.split),
        "target_frames": int(args.target_frames),
        "written_frames": int(written),
        "skipped_existing": int(skipped),
        "attempted_frames": int(attempted),
        "used_segment_ids": sorted(set(used_segment_ids)),
        "source_id": int(args.source_id),
        "causal_current_frame_default": not bool(args.use_segment_ee_path),
        "preview_dir": str(preview_dir),
        "items": manifest_items,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "items"}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
