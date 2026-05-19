#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch
import torch.nn.functional as F

from openpi.picf.object3d_slot_lifter import Object3DSlotLifter
from openpi.picf.object3d_slot_lifter import make_object3d_point_features


_PALETTE = np.asarray(
    [
        [80, 80, 80],
        [230, 80, 60],
        [60, 140, 240],
        [70, 190, 110],
        [245, 170, 55],
        [170, 90, 220],
        [40, 210, 210],
        [250, 110, 180],
        [180, 170, 60],
        [120, 120, 255],
        [255, 120, 80],
        [100, 220, 120],
    ],
    dtype=np.uint8,
)


def _load_cameras(calvin_root: Path) -> dict:
    with open(calvin_root / "calib" / "cameras.json", encoding="utf-8") as handle:
        return json.load(handle)


def _rpy_zyx_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in np.asarray(rpy, dtype=np.float32).reshape(3)]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(-1, 3).astype(np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return (np.asarray(transform, dtype=np.float32) @ np.concatenate([points, ones], axis=1).T).T[:, :3]


def _unproject(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    W_T_C: np.ndarray,
    *,
    stride: int,
    view_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_np = np.asarray(depth, dtype=np.float32)
    rgb_np = np.asarray(rgb, dtype=np.uint8)
    height, width = depth_np.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    valid = np.isfinite(depth_np) & (depth_np > 0.01) & (depth_np < 10.0)
    stride_mask = np.zeros_like(valid, dtype=bool)
    stride_mask[::stride, ::stride] = True
    valid &= stride_mask
    z = depth_np[valid]
    u = xx[valid]
    v = yy[valid]
    x = (u - float(K[0, 2])) / float(K[0, 0]) * z
    y = (v - float(K[1, 2])) / float(K[1, 1]) * z
    xyz_cam = np.stack([x, y, z], axis=-1).astype(np.float32)
    xyz = _transform_points(xyz_cam, W_T_C).astype(np.float32)
    colors = rgb_np[valid].astype(np.float32) / 255.0
    view_ids = np.full((xyz.shape[0],), int(view_id), dtype=np.int64)
    pixels = np.stack([u, v], axis=-1).astype(np.float32)
    return xyz, colors, view_ids, pixels


def _frame_to_points(calvin_root: Path, frame_path: Path, cameras: dict, static_stride: int, gripper_stride: int):
    data = np.load(frame_path, allow_pickle=True)
    rgb_static = np.asarray(data["rgb_static"], dtype=np.uint8)
    depth_static = np.asarray(data["depth_static"], dtype=np.float32)
    rgb_gripper = np.asarray(data["rgb_gripper"], dtype=np.uint8)
    depth_gripper = np.asarray(data["depth_gripper"], dtype=np.float32)
    robot_obs = np.asarray(data["robot_obs"], dtype=np.float32).reshape(-1)
    static = cameras["static"]
    gripper = cameras["gripper"]
    W_T_E = _make_transform(_rpy_zyx_to_matrix(robot_obs[3:6]), robot_obs[0:3])
    W_T_C_gripper = W_T_E @ np.asarray(gripper["E_T_C"], dtype=np.float32)
    ps = _unproject(
        rgb_static,
        depth_static,
        np.asarray(static["K"], dtype=np.float32),
        np.asarray(static["W_T_C"], dtype=np.float32),
        stride=static_stride,
        view_id=0,
    )
    pg = _unproject(
        rgb_gripper,
        depth_gripper,
        np.asarray(gripper["K"], dtype=np.float32),
        W_T_C_gripper.astype(np.float32),
        stride=gripper_stride,
        view_id=1,
    )
    xyz = np.concatenate([ps[0], pg[0]], axis=0)
    rgb = np.concatenate([ps[1], pg[1]], axis=0)
    view_ids = np.concatenate([ps[2], pg[2]], axis=0)
    pixels_static = ps[3]
    return rgb_static, rgb_gripper, xyz, rgb, view_ids, pixels_static, ps[0].shape[0], W_T_E[:3, 3].astype(np.float32)


def _normalize_scene(xyz_list: list[np.ndarray]) -> tuple[np.ndarray, float]:
    all_xyz = np.concatenate(xyz_list, axis=0)
    center = all_xyz.mean(axis=0).astype(np.float32)
    scale = float(np.percentile(np.linalg.norm(all_xyz - center[None, :], axis=1), 95) + 1e-6)
    return center, scale


def _render_static_overlay(
    rgb_static: np.ndarray,
    pixels: np.ndarray,
    labels: np.ndarray,
    *,
    point_limit: int = 5000,
    weights: np.ndarray | None = None,
) -> Image.Image:
    image = Image.fromarray(rgb_static.copy()).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if pixels.shape[0] > point_limit:
        idx = np.linspace(0, pixels.shape[0] - 1, point_limit).astype(np.int64)
    else:
        idx = np.arange(pixels.shape[0], dtype=np.int64)
    for i in idx:
        x, y = pixels[i]
        alpha = 185
        if weights is not None:
            alpha = int(25 + 225 * float(np.clip(weights[i], 0.0, 1.0)))
        color = tuple(int(c) for c in _PALETTE[int(labels[i]) % len(_PALETTE)]) + (alpha,)
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=color)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _render_static_object_focus(
    rgb_static: np.ndarray,
    pixels: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    point_limit: int = 5000,
) -> Image.Image:
    rgb = np.asarray(rgb_static, dtype=np.float32)
    gray = np.repeat(rgb.mean(axis=2, keepdims=True), 3, axis=2)
    faded = np.clip(0.30 * rgb + 0.70 * gray + 35.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(faded).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if pixels.shape[0] > point_limit:
        idx = np.linspace(0, pixels.shape[0] - 1, point_limit).astype(np.int64)
    else:
        idx = np.arange(pixels.shape[0], dtype=np.int64)
    for i in idx:
        w = float(np.clip(weights[i], 0.0, 1.0))
        if w <= 0.02:
            continue
        x, y = pixels[i]
        color = tuple(int(c) for c in _PALETTE[int(labels[i]) % len(_PALETTE)])
        alpha = int(60 + 195 * w)
        radius = 1.2 + 2.4 * w
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color + (alpha,))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _render_topdown(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    width: int = 300,
    height: int = 300,
    weights: np.ndarray | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    xy = xyz[:, [0, 1]]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pts = (xy - lo[None, :]) / span[None, :]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    if pts.shape[0] > 7000:
        idx = np.linspace(0, pts.shape[0] - 1, 7000).astype(np.int64)
    else:
        idx = np.arange(pts.shape[0], dtype=np.int64)
    for i in idx:
        x = int(pts[i, 0] * (width - 1))
        y = int((1.0 - pts[i, 1]) * (height - 1))
        color_np = _PALETTE[int(labels[i]) % len(_PALETTE)].astype(np.float32)
        if weights is not None:
            w = float(np.clip(weights[i], 0.0, 1.0))
            color_np = (1.0 - w) * np.asarray([55.0, 55.0, 55.0]) + w * color_np
        color = tuple(int(c) for c in np.clip(color_np, 0, 255))
        radius = 1 if weights is None or weights[i] < 0.6 else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def _render_topdown_object_focus(
    xyz: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    width: int = 300,
    height: int = 300,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    xy = xyz[:, [0, 1]]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pts = (xy - lo[None, :]) / span[None, :]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    if pts.shape[0] > 7000:
        idx = np.linspace(0, pts.shape[0] - 1, 7000).astype(np.int64)
    else:
        idx = np.arange(pts.shape[0], dtype=np.int64)
    for i in idx:
        x = int(pts[i, 0] * (width - 1))
        y = int((1.0 - pts[i, 1]) * (height - 1))
        w = float(np.clip(weights[i], 0.0, 1.0))
        if w <= 0.02:
            color = (48, 48, 48)
            draw.point((x, y), fill=color)
            continue
        color_np = _PALETTE[int(labels[i]) % len(_PALETTE)].astype(np.float32)
        color_np = (1.0 - w) * np.asarray([75.0, 75.0, 75.0]) + w * color_np
        color = tuple(int(c) for c in np.clip(color_np, 0, 255))
        radius = 1 if w < 0.55 else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return canvas


def _robust01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _target_color_score(rgb: np.ndarray, text: str) -> tuple[np.ndarray, bool]:
    rgb = np.asarray(rgb, dtype=np.float32)
    text_l = text.lower()
    eps = 1e-6
    chroma = rgb / (rgb.sum(axis=1, keepdims=True) + eps)
    saturation = rgb.max(axis=1) - rgb.min(axis=1)
    sat_gate = 1.0 / (1.0 + np.exp(-12.0 * (saturation - 0.20)))
    # In CALVIN language, color words can describe either the manipulated
    # object ("red block") or an effect state ("green light").  Object binding
    # must use color only for colored object references; actuator tasks such as
    # button/switch/lamp should be grounded by contact/geometry, not by the
    # visual state that changes after the action.
    colored_object = any(
        phrase in text_l
        for phrase in (
            "red block",
            "blue block",
            "pink block",
            "purple block",
            "green block",
            "yellow block",
            "red cube",
            "blue cube",
            "pink cube",
            "purple cube",
            "green cube",
            "yellow cube",
            "red object",
            "blue object",
            "pink object",
            "purple object",
            "green object",
            "yellow object",
        )
    )
    actuator_or_effect_task = any(
        word in text_l
        for word in ("button", "switch", "drawer", "handle", "led", "lamp", "light")
    )
    if actuator_or_effect_task and not colored_object:
        return np.zeros((rgb.shape[0],), dtype=np.float32), False
    if "red" in text_l:
        score = (
            1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 0] - 0.55)))
            * (1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 0] - rgb[:, 1] - 0.25))))
            * (1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 0] - rgb[:, 2] - 0.25))))
            * sat_gate
        )
        has_color = True
    elif "blue" in text_l:
        score = (
            1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 2] - 0.45)))
            * (1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 2] - rgb[:, 0] - 0.20))))
            * (1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 2] - rgb[:, 1] - 0.20))))
            * sat_gate
        )
        has_color = True
    elif "pink" in text_l or "magenta" in text_l or "purple" in text_l:
        pink = np.minimum(rgb[:, 0], rgb[:, 2]) - rgb[:, 1]
        score = (1.0 / (1.0 + np.exp(-14.0 * (pink - 0.12)))) * sat_gate
        has_color = True
    elif "green" in text_l:
        score = (
            1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 1] - 0.40)))
            * (1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 1] - rgb[:, 0] - 0.16))))
            * (1.0 / (1.0 + np.exp(-18.0 * (rgb[:, 1] - rgb[:, 2] - 0.16))))
            * sat_gate
        )
        has_color = True
    elif "yellow" in text_l:
        margin = np.minimum(chroma[:, 0], chroma[:, 1]) - chroma[:, 2]
        score = (1.0 / (1.0 + np.exp(-14.0 * (margin - 0.16)))) * sat_gate
        has_color = True
    else:
        score = sat_gate
        has_color = False
    return np.clip(score, 0.0, 1.0).astype(np.float32), has_color


def _foreground_score(
    xyz: np.ndarray,
    rgb: np.ndarray,
    view_ids: np.ndarray,
    ee_pos: np.ndarray,
    text: str,
    ee_path: np.ndarray | None = None,
) -> np.ndarray:
    color_score, has_color = _target_color_score(rgb, text)
    if ee_path is not None and ee_path.size:
        path = np.asarray(ee_path, dtype=np.float32).reshape(-1, 3)
        distance = np.linalg.norm(xyz[:, None, :] - path[None, :, :], axis=-1).min(axis=1)
    else:
        distance = np.linalg.norm(xyz - ee_pos[None, :], axis=1)
    saturation = _robust01(rgb.max(axis=1) - rgb.min(axis=1))
    wrist_score = (view_ids.astype(np.int64) == 1).astype(np.float32)
    if has_color:
        contact_score = np.exp(-0.5 * np.square(distance / 0.18)).astype(np.float32)
        score = 0.70 * color_score + 0.18 * contact_score + 0.08 * wrist_score + 0.04 * saturation
    else:
        contact_sigma = 0.050 if any(word in text.lower() for word in ("button", "switch")) else 0.070
        if any(word in text.lower() for word in ("drawer", "handle")):
            contact_sigma = 0.075
        contact_score = np.exp(-0.5 * np.square(distance / contact_sigma)).astype(np.float32)
        # Non-color actuator tasks must stay causally grounded at the contact
        # locus.  Saturation is intentionally excluded here because it promotes
        # the lamp/LED effect region instead of the button/switch actuator.
        contact_score = np.square(contact_score)
        score = 0.98 * contact_score + 0.02 * wrist_score
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _display_foreground_weight(score: np.ndarray, threshold: float = 0.38) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    return np.clip((score - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0).astype(np.float32)


def _make_panel(
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    pixels_static: np.ndarray,
    static_labels: np.ndarray,
    xyz: np.ndarray,
    labels: np.ndarray,
    text: str,
    step: int,
    title: str = "RGB static + slot colors",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    static_weights = None if weights is None else weights[: static_labels.shape[0]]
    left = _render_static_overlay(rgb_static, pixels_static, static_labels, weights=static_weights).resize((300, 300))
    grip = Image.fromarray(rgb_gripper).resize((150, 150))
    top = _render_topdown(xyz, labels, width=300, height=300, weights=weights)
    panel = Image.new("RGB", (650, 360), (245, 245, 245))
    panel.paste(left, (0, 30))
    panel.paste(top, (315, 30))
    panel.paste(grip, (500, 205))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), f"{title} | frame {step}", fill=(0, 0, 0))
    draw.text((320, 6), "3D topdown slot colors", fill=(0, 0, 0))
    draw.text((8, 338), text[:88], fill=(0, 0, 0))
    return np.asarray(panel, dtype=np.uint8)


def _make_object_focus_panel(
    rgb_static: np.ndarray,
    rgb_gripper: np.ndarray,
    pixels_static: np.ndarray,
    static_labels: np.ndarray,
    xyz: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    text: str,
    step: int,
) -> np.ndarray:
    static_weights = weights[: static_labels.shape[0]]
    left = _render_static_object_focus(rgb_static, pixels_static, static_labels, static_weights).resize((300, 300))
    grip = Image.fromarray(rgb_gripper).resize((150, 150))
    top = _render_topdown_object_focus(xyz, labels, weights, width=300, height=300)
    panel = Image.new("RGB", (650, 360), (245, 245, 245))
    panel.paste(left, (0, 30))
    panel.paste(top, (315, 30))
    panel.paste(grip, (500, 205))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), f"background faded; object slots strong | frame {step}", fill=(0, 0, 0))
    draw.text((320, 6), "3D foreground instance-map view", fill=(0, 0, 0))
    draw.text((8, 338), text[:88], fill=(0, 0, 0))
    return np.asarray(panel, dtype=np.uint8)


def _select_segment(calvin_root: Path, prompt_contains: str, max_frames: int) -> tuple[list[Path], str]:
    ann_path = calvin_root / "training" / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    texts = ann["language"]["ann"]
    intervals = ann["info"]["indx"]
    needle = prompt_contains.lower()
    selected_text = None
    selected_interval = None
    for text, interval in zip(texts, intervals):
        if needle in str(text).lower():
            selected_text = str(text)
            selected_interval = tuple(int(x) for x in interval)
            break
    if selected_interval is None:
        selected_text = str(texts[0])
        selected_interval = tuple(int(x) for x in intervals[0])
    start, end = selected_interval
    frame_ids = list(range(start, min(end + 1, start + max_frames)))
    paths = [calvin_root / "training" / f"episode_{idx:07d}.npz" for idx in frame_ids]
    paths = [path for path in paths if path.exists()]
    if not paths:
        paths = sorted((calvin_root / "training").glob("episode_*.npz"))[:max_frames]
        selected_text = "fallback sorted frames"
    return paths, selected_text


def _write_video_or_gif(output_path: Path, frames: list[np.ndarray], fps: int) -> tuple[Path, str | None]:
    try:
        import imageio.v2 as imageio

        imageio.mimsave(output_path, frames, fps=fps)
        return output_path, None
    except Exception as exc:  # pragma: no cover - depends on remote ffmpeg/imageio plugins.
        gif_path = output_path.with_suffix(".gif")
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(1, int(1000 / max(fps, 1))),
            loop=0,
        )
        return gif_path, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-contains", default="red block")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--static-stride", type=int, default=4)
    parser.add_argument("--gripper-stride", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=96)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    calvin_root = Path(args.calvin_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = _load_cameras(calvin_root)
    paths, text = _select_segment(calvin_root, args.prompt_contains, args.frames)
    if not paths:
        raise RuntimeError("No CALVIN frames found")

    frames = []
    xyz_list = []
    for path in paths:
        item = _frame_to_points(calvin_root, path, cameras, args.static_stride, args.gripper_stride)
        frames.append((path, item))
        xyz_list.append(item[2])
    center, scale = _normalize_scene(xyz_list)
    ee_path = np.stack([item[7] for _, item in frames], axis=0).astype(np.float32)

    batch_features = []
    batch_xyz_norm = []
    batch_xyz_world = []
    batch_valid = []
    batch_foreground = []
    max_points = max(item[2].shape[0] for _, item in frames)
    for _, item in frames:
        _, _, xyz, rgb, view_ids, _, _, ee_pos = item
        xyz_norm = (xyz - center[None, :]) / scale
        foreground = _foreground_score(xyz, rgb, view_ids, ee_pos, text, ee_path=ee_path)
        pad = max_points - xyz.shape[0]
        xyz_pad = np.pad(xyz, ((0, pad), (0, 0)), constant_values=0.0)
        xyz_norm_pad = np.pad(xyz_norm, ((0, pad), (0, 0)), constant_values=0.0)
        rgb_pad = np.pad(rgb, ((0, pad), (0, 0)), constant_values=0.0)
        view_pad = np.pad(view_ids, (0, pad), constant_values=0)
        fg_pad = np.pad(foreground, (0, pad), constant_values=0.0)
        valid = np.zeros((max_points,), dtype=bool)
        valid[: xyz.shape[0]] = True
        xyz_norm_t = torch.from_numpy(xyz_norm_pad).float()
        rgb_t = torch.from_numpy(rgb_pad).float()
        view_t = torch.from_numpy(view_pad).long()
        feat = make_object3d_point_features(xyz_norm_t[None], rgb_t[None], view_t[None], num_views=2).squeeze(0)
        batch_features.append(feat)
        batch_xyz_norm.append(xyz_norm_t)
        batch_xyz_world.append(torch.from_numpy(xyz_pad).float())
        batch_valid.append(torch.from_numpy(valid))
        batch_foreground.append(torch.from_numpy(fg_pad).float())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    features = torch.stack(batch_features, dim=0).to(device=device)
    xyz_norm = torch.stack(batch_xyz_norm, dim=0).to(device=device)
    valid = torch.stack(batch_valid, dim=0).to(device=device)
    foreground = torch.stack(batch_foreground, dim=0).to(device=device)

    model = Object3DSlotLifter(
        input_dim=features.shape[-1],
        slot_dim=args.slot_dim,
        num_slots=args.num_slots,
        num_iterations=args.iters,
    ).to(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    losses = []
    for step in range(args.train_steps):
        out = model(features, xyz_norm, valid)
        recon_per_point = (out.reconstruction - features).square().mean(dim=-1)
        fg_weight = (0.20 + 1.80 * foreground).detach()
        recon_loss = (recon_per_point * fg_weight * valid.float()).sum() / (fg_weight * valid.float()).sum().clamp_min(1.0)
        weights = out.point_slot_weights[..., 1:]
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_loss = (entropy * foreground * valid.float()).sum() / (foreground * valid.float()).sum().clamp_min(1.0)
        fg_valid = (foreground * valid.float())
        mass = (weights * fg_valid[..., None]).sum(dim=1) / fg_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        balance_loss = ((mass - (1.0 / args.num_slots)) ** 2).mean()
        bg_mean = out.background_weight[valid].mean()
        fg_bg_loss = (out.background_weight * foreground * valid.float()).sum() / (foreground * valid.float()).sum().clamp_min(1.0)
        compactness = (out.covariance_diag.sum(dim=-1) * mass.detach()).sum(dim=-1).mean()
        loss = recon_loss + 0.01 * entropy_loss + 0.10 * balance_loss + 0.005 * bg_mean + 0.10 * fg_bg_loss + 0.02 * compactness
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 20 == 0 or step == args.train_steps - 1:
            losses.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "recon_loss": float(recon_loss.detach()),
                    "entropy_loss": float(entropy_loss.detach()),
                    "balance_loss": float(balance_loss.detach()),
                    "background_mean": float(bg_mean.detach()),
                    "foreground_background_loss": float(fg_bg_loss.detach()),
                    "compactness_loss": float(compactness.detach()),
                }
            )

    with torch.no_grad():
        out = model(features, xyz_norm, valid)
        labels_empty_aware = out.point_slot_weights.argmax(dim=-1).cpu().numpy()
        labels_object_only = (out.encoder_attention.argmax(dim=1) + 1).cpu().numpy()
        encoder_mass = (
            out.encoder_attention * valid[:, None, :].to(dtype=out.encoder_attention.dtype)
        ).sum(dim=-1) / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        objectness = out.objectness.cpu().numpy()
        centers = out.centers.cpu().numpy() * scale + center[None, None, :]

    video_frames_empty = []
    video_frames_object = []
    video_frames_focused = []
    video_frames_object_focus = []
    foreground_frame_scores = []
    panel_cache: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for frame_idx, (path, item) in enumerate(frames):
        rgb_static, rgb_gripper, xyz, rgb, view_ids, pixels_static, static_count, ee_pos = item
        foreground_np = _foreground_score(xyz, rgb, view_ids, ee_pos, text, ee_path=ee_path)
        display_foreground = _display_foreground_weight(foreground_np)
        foreground_frame_scores.append(float(np.mean(np.sort(display_foreground)[-max(1, display_foreground.size // 200) :])))
        empty_labels = labels_empty_aware[frame_idx, : xyz.shape[0]]
        object_labels = labels_object_only[frame_idx, : xyz.shape[0]]
        empty_panel = _make_panel(
            rgb_static,
            rgb_gripper,
            pixels_static,
            empty_labels[:static_count],
            xyz,
            empty_labels,
            text,
            int(path.stem.split("_")[1]),
            title="empty-aware point decoder",
        )
        object_panel = _make_panel(
            rgb_static,
            rgb_gripper,
            pixels_static,
            object_labels[:static_count],
            xyz,
            object_labels,
            text,
            int(path.stem.split("_")[1]),
            title="object-only SlotAttention",
        )
        focused_panel = _make_panel(
            rgb_static,
            rgb_gripper,
            pixels_static,
            object_labels[:static_count],
            xyz,
            object_labels,
            text,
            int(path.stem.split("_")[1]),
            title="foreground-enhanced object slots",
            weights=display_foreground,
        )
        object_focus_panel = _make_object_focus_panel(
            rgb_static,
            rgb_gripper,
            pixels_static,
            object_labels[:static_count],
            xyz,
            object_labels,
            display_foreground,
            text,
            int(path.stem.split("_")[1]),
        )
        video_frames_empty.append(empty_panel)
        video_frames_object.append(object_panel)
        video_frames_focused.append(focused_panel)
        video_frames_object_focus.append(object_focus_panel)
        panel_cache.append((frame_idx, empty_panel, object_panel, focused_panel, object_focus_panel))
        if frame_idx in {0, len(frames) // 2, len(frames) - 1}:
            Image.fromarray(empty_panel).save(output_dir / f"frame_{frame_idx:03d}_empty_aware_panel.png")
            Image.fromarray(object_panel).save(output_dir / f"frame_{frame_idx:03d}_object_only_panel.png")
            Image.fromarray(focused_panel).save(output_dir / f"frame_{frame_idx:03d}_foreground_enhanced_panel.png")
            Image.fromarray(object_focus_panel).save(output_dir / f"frame_{frame_idx:03d}_background_faded_objects_strong_panel.png")

    best_frame_idx = int(np.argmax(np.asarray(foreground_frame_scores, dtype=np.float32)))
    for frame_idx, empty_panel, object_panel, focused_panel, object_focus_panel in panel_cache:
        if frame_idx == best_frame_idx:
            Image.fromarray(empty_panel).save(output_dir / "frame_best_empty_aware_panel.png")
            Image.fromarray(object_panel).save(output_dir / "frame_best_object_only_panel.png")
            Image.fromarray(focused_panel).save(output_dir / "frame_best_foreground_enhanced_panel.png")
            Image.fromarray(object_focus_panel).save(output_dir / "frame_best_background_faded_objects_strong_panel.png")
            break

    empty_video_path, empty_error = _write_video_or_gif(
        output_dir / "object3d_slot_empty_aware.mp4", video_frames_empty, args.fps
    )
    object_video_path, object_error = _write_video_or_gif(
        output_dir / "object3d_slot_object_only.mp4", video_frames_object, args.fps
    )
    focused_video_path, focused_error = _write_video_or_gif(
        output_dir / "object3d_slot_foreground_enhanced.mp4", video_frames_focused, args.fps
    )
    object_focus_video_path, object_focus_error = _write_video_or_gif(
        output_dir / "object3d_slot_background_faded_objects_strong.mp4", video_frames_object_focus, args.fps
    )
    if empty_error or object_error or focused_error or object_focus_error:
        (output_dir / "video_write_error.txt").write_text(
            json.dumps(
                {
                    "empty": empty_error,
                    "object": object_error,
                    "foreground": focused_error,
                    "background_faded_objects_strong": object_focus_error,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    report = {
        "calvin_root": str(calvin_root),
        "prompt": text,
        "frames": [str(path) for path, _ in frames],
        "num_frames": len(frames),
        "num_slots": args.num_slots,
        "slot_dim": args.slot_dim,
        "train_steps": args.train_steps,
        "device": str(device),
        "empty_aware_video_path": str(empty_video_path),
        "object_only_video_path": str(object_video_path),
        "foreground_enhanced_video_path": str(focused_video_path),
        "background_faded_objects_strong_video_path": str(object_focus_video_path),
        "losses": losses,
        "objectness_mean": objectness.mean(axis=0).tolist(),
        "encoder_slot_mass_mean": encoder_mass.mean(dim=0).cpu().tolist(),
        "background_mean_final": float(out.background_weight[valid].mean().detach().cpu()),
        "foreground_mean": float(foreground[valid].mean().detach().cpu()),
        "foreground_display_threshold": 0.38,
        "ee_path_length": int(ee_path.shape[0]),
        "best_foreground_frame_index": best_frame_idx,
        "best_foreground_frame_path": str(frames[best_frame_idx][0]),
        "foreground_frame_scores": foreground_frame_scores,
        "centers_world_mean": centers.mean(axis=0).tolist(),
        "warning": "diagnostic video; foreground-enhanced view uses task/contact/color salience only for visualization/diagnosis, not production object labels",
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
