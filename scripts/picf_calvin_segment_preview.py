#!/usr/bin/env python3
"""Generate CALVIN segment/frame previews before running slot diagnostics.

This script is intentionally diagnostic-only.  CALVIN language intervals mark
where a natural-language subtask occurs, but the middle frame is not a task
completion/contact oracle.  The preview therefore saves start/q25/mid/q75/end
frames plus two data-driven candidates:

* best_contact: maximum task/contact/color foreground score.
* best_scene_delta: maximum scene_obs delta from the interval start, if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

from picf_object3d_slot_video import _display_foreground_weight
from picf_object3d_slot_video import _foreground_score
from picf_object3d_slot_video import _frame_to_points
from picf_object3d_slot_video import _load_cameras
from picf_object3d_slot_video import _render_static_object_focus
from picf_object3d_slot_video import _render_topdown_object_focus
from picf_object3d_slot_video import _write_video_or_gif


_DEFAULT_PROMPTS = (
    "red block",
    "blue block",
    "pink block",
    "drawer",
    "button",
    "switch",
)


def _load_annotations(calvin_root: Path) -> tuple[list[str], list[tuple[int, int]]]:
    ann_path = calvin_root / "training" / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    texts = [str(text) for text in ann["language"]["ann"]]
    intervals = [tuple(int(x) for x in interval) for interval in ann["info"]["indx"]]
    return texts, intervals


def _find_segment(
    texts: list[str],
    intervals: list[tuple[int, int]],
    prompt_contains: str,
    occurrence: int,
) -> tuple[str, tuple[int, int]]:
    needle = prompt_contains.lower()
    matches = [(text, interval) for text, interval in zip(texts, intervals) if needle in text.lower()]
    if not matches:
        raise RuntimeError(f"No CALVIN language interval contains {prompt_contains!r}")
    return matches[min(max(occurrence, 0), len(matches) - 1)]


def _scene_obs(frame_path: Path) -> np.ndarray | None:
    data = np.load(frame_path, allow_pickle=True)
    if "scene_obs" not in data:
        return None
    return np.asarray(data["scene_obs"], dtype=np.float32).reshape(-1)


def _frame_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _unique_indices(indices: list[int], count: int) -> list[int]:
    seen = set()
    out = []
    for idx in indices:
        idx = int(np.clip(idx, 0, max(count - 1, 0)))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _motion_scores_from_items(
    frame_items: list[tuple[Path, tuple]],
    *,
    arm_radius: float = 0.11,
    grid_size: float = 0.025,
) -> tuple[list[float], list[float], list[float]]:
    """Approximate non-arm static-scene point motion inside a language segment.

    CALVIN training annotations delimit language segments, not exact contact
    windows.  This score detects when non-arm scene points change relative to
    the previous frame.  It intentionally ignores points close to the
    end-effector trajectory so arm motion does not dominate the slice boundary.
    """

    if not frame_items:
        return [], [], []
    ee_path = np.stack([item[7] for _, item in frame_items], axis=0).astype(np.float32)
    prev_keys: set[tuple[int, int, int]] | None = None
    object_motion = []
    arm_proxy = []
    settled = []
    for _, item in frame_items:
        _, _, xyz, _rgb, view_ids, _pixels_static, static_count, ee_pos = item
        static_xyz = xyz[:static_count]
        if static_xyz.size == 0:
            object_motion.append(0.0)
            arm_proxy.append(0.0)
            settled.append(1.0)
            continue
        distance_to_ee_path = np.linalg.norm(static_xyz[:, None, :] - ee_path[None, :, :], axis=-1).min(axis=1)
        distance_to_current_ee = np.linalg.norm(static_xyz - ee_pos[None, :], axis=1)
        arm_mask = distance_to_current_ee < arm_radius
        non_arm_mask = distance_to_ee_path >= arm_radius
        arm_proxy.append(float(np.mean(arm_mask)))
        non_arm_xyz = static_xyz[non_arm_mask]
        if non_arm_xyz.shape[0] == 0:
            object_motion.append(0.0)
            settled.append(1.0)
            prev_keys = set()
            continue
        quantized = np.floor(non_arm_xyz / grid_size).astype(np.int32)
        keys = {tuple(row.tolist()) for row in quantized}
        if prev_keys is None:
            motion = 0.0
        else:
            union = max(len(keys | prev_keys), 1)
            motion = 1.0 - (len(keys & prev_keys) / union)
        object_motion.append(float(motion))
        settled.append(float(1.0 / (1.0 + 20.0 * motion)))
        prev_keys = keys
    return object_motion, arm_proxy, settled


def _select_motion_window(
    contact_scores: list[float],
    scene_deltas: list[float],
    object_motion: list[float],
    *,
    min_len: int = 6,
    settle_patience: int = 3,
) -> tuple[int, int, int]:
    count = len(contact_scores)
    if count == 0:
        return 0, 0, 0
    contact = np.asarray(contact_scores, dtype=np.float32)
    scene = np.asarray(scene_deltas, dtype=np.float32)
    motion = np.asarray(object_motion, dtype=np.float32)
    # Contact starts the candidate, scene/motion completion ends it.
    contact_norm = contact / max(float(contact.max()), 1e-6)
    motion_norm = motion / max(float(motion.max()), 1e-6)
    scene_delta_step = np.concatenate([[0.0], np.maximum(scene[1:] - scene[:-1], 0.0)])
    scene_norm = scene_delta_step / max(float(scene_delta_step.max()), 1e-6)
    event = 0.55 * contact_norm + 0.30 * motion_norm + 0.15 * scene_norm
    best = int(np.argmax(event))
    active_threshold = max(0.18, 0.35 * float(event.max()))
    start = best
    while start > 0 and event[start - 1] >= active_threshold:
        start -= 1
    # Include a short pre-contact context so the preview shows approach.
    start = max(0, start - 2)
    end = max(best + min_len, start + min_len)
    calm = 0
    for idx in range(best + 1, count):
        if event[idx] < active_threshold and motion_norm[idx] < 0.25:
            calm += 1
            if calm >= settle_patience:
                end = idx
                break
        else:
            calm = 0
        end = idx
    end = min(max(end, start), count - 1)
    return start, best, end


def _make_preview_panel(
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    pixels_static: np.ndarray,
    static_count: int,
    xyz: np.ndarray,
    foreground: np.ndarray,
    text: str,
    frame_number: int,
    label: str,
    contact_score: float,
    scene_delta: float,
) -> np.ndarray:
    display = _display_foreground_weight(foreground)
    labels = np.ones((xyz.shape[0],), dtype=np.int64)
    left = _render_static_object_focus(
        rgb_static,
        pixels_static,
        labels[:static_count],
        display[:static_count],
        point_limit=7000,
    ).resize((300, 300))
    top = _render_topdown_object_focus(xyz, labels, display, width=300, height=300)
    grip = Image.fromarray(rgb_gripper).resize((150, 150))
    raw = Image.fromarray(rgb_static).resize((180, 180))
    panel = Image.new("RGB", (820, 360), (245, 245, 245))
    panel.paste(left, (0, 42))
    panel.paste(top, (315, 42))
    panel.paste(raw, (630, 42))
    panel.paste(grip, (660, 232))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"{label} | frame {frame_number}", fill=(0, 0, 0))
    draw.text((8, 24), f"contact_score={contact_score:.4f} scene_delta={scene_delta:.4f}", fill=(0, 0, 0))
    draw.text((320, 20), "topdown task/contact salience", fill=(0, 0, 0))
    draw.text((635, 20), "raw static / gripper", fill=(0, 0, 0))
    draw.text((8, 338), text[:112], fill=(0, 0, 0))
    return np.asarray(panel, dtype=np.uint8)


def _make_process_panel(
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    pixels_static: np.ndarray,
    static_count: int,
    xyz: np.ndarray,
    foreground: np.ndarray,
    text: str,
    frame_number: int,
    relative_index: int,
    contact_score: float,
    scene_delta: float,
    object_motion: float,
    window_label: str,
) -> np.ndarray:
    display = _display_foreground_weight(foreground)
    labels = np.ones((xyz.shape[0],), dtype=np.int64)
    raw = Image.fromarray(rgb_static).resize((300, 300))
    salience = _render_static_object_focus(
        rgb_static,
        pixels_static,
        labels[:static_count],
        display[:static_count],
        point_limit=7000,
    ).resize((300, 300))
    top = _render_topdown_object_focus(xyz, labels, display, width=220, height=220)
    grip = Image.fromarray(rgb_gripper).resize((150, 150))
    panel = Image.new("RGB", (860, 360), (245, 245, 245))
    panel.paste(raw, (0, 42))
    panel.paste(salience, (310, 42))
    panel.paste(top, (620, 42))
    panel.paste(grip, (690, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"raw static | rel={relative_index} frame={frame_number} {window_label}", fill=(0, 0, 0))
    draw.text((315, 8), "task/contact salience overlay", fill=(0, 0, 0))
    draw.text((625, 8), "3D topdown salience", fill=(0, 0, 0))
    draw.text(
        (8, 24),
        f"contact={contact_score:.4f} scene_delta={scene_delta:.4f} nonarm_motion={object_motion:.4f}",
        fill=(0, 0, 0),
    )
    draw.text((8, 338), text[:118], fill=(0, 0, 0))
    return np.asarray(panel, dtype=np.uint8)


def _write_task_preview(
    calvin_root: Path,
    cameras: dict,
    output_root: Path,
    texts: list[str],
    intervals: list[tuple[int, int]],
    prompt: str,
    occurrence: int,
    max_frames: int,
    static_stride: int,
    gripper_stride: int,
) -> dict:
    text, (start, end) = _find_segment(texts, intervals, prompt, occurrence)
    frame_ids = list(range(start, min(end + 1, start + max_frames)))
    frame_paths = [calvin_root / "training" / f"episode_{idx:07d}.npz" for idx in frame_ids]
    frame_paths = [path for path in frame_paths if path.exists()]
    if not frame_paths:
        raise RuntimeError(f"No frame paths found for {prompt!r} interval {(start, end)}")

    frame_items = [
        (path, _frame_to_points(calvin_root, path, cameras, static_stride, gripper_stride))
        for path in frame_paths
    ]
    ee_path = np.stack([item[7] for _, item in frame_items], axis=0).astype(np.float32)
    start_scene = _scene_obs(frame_paths[0])
    contact_scores = []
    scene_deltas = []
    for path, item in frame_items:
        _, _, xyz, rgb, view_ids, _, _, ee_pos = item
        foreground = _foreground_score(xyz, rgb, view_ids, ee_pos, text, ee_path=ee_path)
        display = _display_foreground_weight(foreground)
        top_count = max(1, display.size // 200)
        contact_scores.append(float(np.mean(np.sort(display)[-top_count:])))
        current_scene = _scene_obs(path)
        if start_scene is None or current_scene is None or current_scene.shape != start_scene.shape:
            scene_deltas.append(0.0)
        else:
            scene_deltas.append(float(np.linalg.norm(current_scene - start_scene)))

    object_motion, arm_proxy, settled_scores = _motion_scores_from_items(frame_items)

    count = len(frame_items)
    best_contact = int(np.argmax(np.asarray(contact_scores, dtype=np.float32)))
    best_scene_delta = int(np.argmax(np.asarray(scene_deltas, dtype=np.float32)))
    motion_start, motion_peak, motion_end = _select_motion_window(contact_scores, scene_deltas, object_motion)
    chosen = _unique_indices(
        [
            0,
            round(0.25 * (count - 1)),
            round(0.50 * (count - 1)),
            round(0.75 * (count - 1)),
            count - 1,
            best_contact,
            best_scene_delta,
            motion_start,
            motion_peak,
            motion_end,
        ],
        count,
    )

    safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in prompt.lower()).strip("_")
    output_dir = output_root / safe_prompt
    output_dir.mkdir(parents=True, exist_ok=True)
    strip_panels = []
    process_panels = []
    selected = []
    label_by_index = {
        0: "start",
        round(0.25 * (count - 1)): "q25",
        round(0.50 * (count - 1)): "mid",
        round(0.75 * (count - 1)): "q75",
        count - 1: "end",
        best_contact: "best_contact",
        best_scene_delta: "best_scene_delta",
        motion_start: "motion_start",
        motion_peak: "motion_peak",
        motion_end: "motion_end",
    }
    for idx, (path, item) in enumerate(frame_items):
        rgb_static, rgb_gripper, xyz, rgb, view_ids, pixels_static, static_count, ee_pos = item
        foreground = _foreground_score(xyz, rgb, view_ids, ee_pos, text, ee_path=ee_path)
        process_panels.append(
            _make_process_panel(
                rgb_static,
                rgb_gripper,
                pixels_static,
                static_count,
                xyz,
                foreground,
                text,
                _frame_number(path),
                idx,
                contact_scores[idx],
                scene_deltas[idx],
                object_motion[idx],
                "[motion-window]" if motion_start <= idx <= motion_end else "",
            )
        )

    process_video, process_error = _write_video_or_gif(output_dir / "segment_process.mp4", process_panels, fps=8)
    if process_error:
        (output_dir / "segment_process_video_error.txt").write_text(process_error, encoding="utf-8")

    for idx in chosen:
        path, item = frame_items[idx]
        rgb_static, rgb_gripper, xyz, rgb, view_ids, pixels_static, static_count, ee_pos = item
        foreground = _foreground_score(xyz, rgb, view_ids, ee_pos, text, ee_path=ee_path)
        label = label_by_index.get(idx, f"frame_{idx}")
        panel = _make_preview_panel(
            rgb_static,
            rgb_gripper,
            pixels_static,
            static_count,
            xyz,
            foreground,
            text,
            _frame_number(path),
            label,
            contact_scores[idx],
            scene_deltas[idx],
        )
        Image.fromarray(panel).save(output_dir / f"{idx:03d}_{label}_frame_{_frame_number(path):07d}.png")
        strip_panels.append(Image.fromarray(panel).resize((410, 180)))
        selected.append(
            {
                "relative_index": idx,
                "label": label,
                "frame": _frame_number(path),
                "contact_score": contact_scores[idx],
                "scene_delta": scene_deltas[idx],
                "path": str(path),
            }
        )

    strip = Image.new("RGB", (410 * len(strip_panels), 180), (245, 245, 245))
    for idx, panel in enumerate(strip_panels):
        strip.paste(panel, (410 * idx, 0))
    strip.save(output_dir / "segment_strip.png")

    report = {
        "prompt_contains": prompt,
        "matched_prompt": text,
        "interval": [start, end],
        "max_frames_used": count,
            "best_contact_relative_index": best_contact,
            "best_scene_delta_relative_index": best_scene_delta,
        "motion_window": {
            "start_relative_index": motion_start,
            "peak_relative_index": motion_peak,
            "end_relative_index": motion_end,
            "start_frame": _frame_number(frame_paths[motion_start]),
            "peak_frame": _frame_number(frame_paths[motion_peak]),
            "end_frame": _frame_number(frame_paths[motion_end]),
        },
        "segment_process_video": str(process_video),
        "segment_process_video_error": process_error,
        "selected": selected,
        "contact_scores": contact_scores,
        "scene_deltas": scene_deltas,
        "non_arm_object_motion_scores": object_motion,
        "arm_proxy_scores": arm_proxy,
        "settled_scores": settled_scores,
        "note": (
            "Language interval is not a CALVIN completion oracle.  Use these "
            "candidate frames to verify contact/completion before slot diagnosis."
        ),
    }
    with open(output_dir / "segment_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", nargs="*", default=list(_DEFAULT_PROMPTS))
    parser.add_argument("--occurrence", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--static-stride", type=int, default=4)
    parser.add_argument("--gripper-stride", type=int, default=2)
    args = parser.parse_args()

    calvin_root = Path(args.calvin_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cameras = _load_cameras(calvin_root)
    texts, intervals = _load_annotations(calvin_root)
    reports = []
    for prompt in args.prompts:
        reports.append(
            _write_task_preview(
                calvin_root,
                cameras,
                output_root,
                texts,
                intervals,
                prompt,
                args.occurrence,
                args.max_frames,
                args.static_stride,
                args.gripper_stride,
            )
        )
    manifest = {"output_root": str(output_root), "reports": reports}
    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2)[:4000])


if __name__ == "__main__":
    main()
