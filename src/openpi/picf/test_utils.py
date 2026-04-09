from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np


def _make_rgb(height: int, width: int, *, shift: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    red = (xx * 7 + shift) % 255
    green = (yy * 9 + 2 * shift) % 255
    blue = ((xx + yy) * 5 + 3 * shift) % 255
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def _make_depth(height: int, width: int, *, bias: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    depth = 0.6 + 0.0015 * xx + 0.001 * yy + bias
    return depth.astype(np.float32)


def _make_tactile_rgb(height: int, width: int, *, shift: int) -> np.ndarray:
    left = _make_rgb(height, width, shift=shift)
    right = _make_rgb(height, width, shift=shift + 17)
    return np.concatenate([left, right], axis=-1)


def _make_tactile_depth(height: int, width: int, *, bias: float) -> np.ndarray:
    left = _make_depth(height, width, bias=bias)
    right = _make_depth(height, width, bias=bias + 0.002)
    return np.stack([left, right], axis=-1)


def _make_robot_obs(step_id: int, *, still: bool) -> np.ndarray:
    robot_obs = np.zeros((15,), dtype=np.float32)
    if still:
        robot_obs[0:3] = np.array([0.0, 0.0, 0.7], dtype=np.float32)
        robot_obs[3:6] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        robot_obs[0:3] = np.array([0.004 * step_id, 0.0, 0.7], dtype=np.float32)
        robot_obs[3:6] = np.array([0.0, 0.0, 0.02 * step_id], dtype=np.float32)
    robot_obs[6] = 1.0
    return robot_obs


def build_mini_calvin_dataset(base_dir: Path, *, make_zip: bool = False) -> str:
    task_root = base_dir / "task_ABCD_D"
    calib_dir = task_root / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    cameras = {
        "cameras": {
            "static": {
                "K": [[80.0, 0.0, 16.0], [0.0, 80.0, 16.0], [0.0, 0.0, 1.0]],
                "W_T_C": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            },
            "gripper": {
                "K": [[54.0, 0.0, 42.0], [0.0, 54.0, 42.0], [0.0, 0.0, 1.0]],
                "W_T_C": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            },
        }
    }
    (calib_dir / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")

    segments = [(0, 4, "hold pose"), (4, 8, "slide right")]
    ann_payload = {"ann": [s[2] for s in segments], "indx": np.asarray([[s[0], s[1]] for s in segments], dtype=np.int32)}

    for split in ("training", "validation"):
        split_dir = task_root / split
        (split_dir / "lang_annotations").mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "lang_annotations" / "auto_lang_ann.npy", ann_payload, allow_pickle=True)
        for step_id in range(8):
            still = step_id < 4
            rgb_static = _make_rgb(32, 32, shift=step_id * (3 if split == "training" else 5))
            rgb_gripper = _make_rgb(16, 16, shift=step_id * (2 if split == "training" else 4))
            depth_static = _make_depth(32, 32, bias=0.002 * step_id)
            rgb_tactile = _make_tactile_rgb(16, 16, shift=step_id * (11 if split == "training" else 13))
            depth_tactile = _make_tactile_depth(16, 16, bias=0.001 * step_id)
            robot_obs = _make_robot_obs(step_id, still=still)
            rel_actions = np.full((7,), 0.01 * step_id, dtype=np.float32)
            np.savez(
                split_dir / f"episode_{step_id:07d}.npz",
                rgb_static=rgb_static,
                rgb_gripper=rgb_gripper,
                depth_static=depth_static,
                rgb_tactile=rgb_tactile,
                depth_tactile=depth_tactile,
                robot_obs=robot_obs,
                rel_actions=rel_actions,
            )

    if not make_zip:
        return str(task_root)

    zip_path = base_dir / "task_ABCD_D.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in task_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(base_dir).as_posix())
    return str(zip_path)
