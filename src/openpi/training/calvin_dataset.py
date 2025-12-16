# src/openpi/training/calvin_dataset.py
'''
zip/dir 两种 backend

从 auto_lang_ann.npy 按段取样

observation 只取段内某个 t（图像/深度/状态），actions 取 t..t+H-1

'''


from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CalvinSegment:
    start: int
    end: int   # exclusive
    lang: str


class _CalvinReader:
    """Read npz files either from a big zip or from an extracted directory."""
    def __init__(self, root: str, split: str, backend: str = "zip"):
        self.root = root  # e.g. "/path/task_ABCD_D.zip" OR "/path/task_ABCD_D"
        self.split = split  # "training" or "validation"
        self.backend = backend  # "zip" or "dir"

        self._zf: Optional[zipfile.ZipFile] = None

        if backend not in ("zip", "dir"):
            raise ValueError(f"backend must be 'zip' or 'dir', got {backend}")

        if backend == "zip":
            if not root.endswith(".zip"):
                raise ValueError(f"zip backend requires .zip path, got {root}")
            if not os.path.exists(root):
                raise FileNotFoundError(root)
        else:
            if not os.path.isdir(root):
                raise FileNotFoundError(root)

    def _zip(self) -> zipfile.ZipFile:
        if self._zf is None:
            # Keep one handle per worker process.
            self._zf = zipfile.ZipFile(self.root, "r")
        return self._zf

    def _npz_path(self, step_id: int) -> str:
        # In zip, files look like: task_ABCD_D/training/episode_0538190.npz
        # In dir, same relative path.
        return f"task_ABCD_D/{self.split}/episode_{step_id:07d}.npz"

    def read_npz(self, step_id: int) -> Dict[str, np.ndarray]:
        rel = self._npz_path(step_id)
        if self.backend == "zip":
            raw = self._zip().read(rel)
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        else:
            abs_path = os.path.join(self.root, rel)
            with np.load(abs_path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}

    def read_npy(self, rel_path: str) -> Any:
        # rel_path e.g. "task_ABCD_D/training/lang_annotations/auto_lang_ann.npy"
        if self.backend == "zip":
            raw = self._zip().read(rel_path)
            return np.load(io.BytesIO(raw), allow_pickle=True)
        else:
            abs_path = os.path.join(self.root, rel_path)
            return np.load(abs_path, allow_pickle=True)

    def read_json(self, rel_path: str) -> Any:
        if self.backend == "zip":
            raw = self._zip().read(rel_path)
            return json.loads(raw.decode("utf-8"))
        else:
            abs_path = os.path.join(self.root, rel_path)
            with open(abs_path, "r", encoding="utf-8") as f:
                return json.load(f)


class CalvinLangSegmentDataset(Dataset):
    """
    Each item corresponds to one language segment (start,end,lang).
    We pick a timestep t inside [start, end-action_horizon], and return:
      - observation from t (rgb/depth/state)
      - action chunk from t..t+H-1 (rel_actions by default)
      - prompt = lang
    """
    def __init__(
        self,
        root: str,
        split: str,
        action_horizon: int,
        backend: str = "zip",
        action_key: str = "rel_actions",
        use_wrist_rgb: bool = True,
        rng_seed: int = 0,
        sample_within_segment: bool = True,
    ):
        self.reader = _CalvinReader(root=root, split=split, backend=backend)
        self.split = split
        self.action_horizon = int(action_horizon)
        self.action_key = action_key
        self.use_wrist_rgb = use_wrist_rgb
        self.sample_within_segment = sample_within_segment

        self.rng = np.random.default_rng(rng_seed)

        # Load language segments
        ann_path = f"task_ABCD_D/{split}/lang_annotations/auto_lang_ann.npy"
        ann = self.reader.read_npy(ann_path)

        segs: List[CalvinSegment] = []
        for entry in ann:
            # entry is a dict-like object
            lang = entry.get("language", None)
            if lang is None:
                # fallback: info['ann']
                info = entry.get("info", {})
                lang = info.get("ann", "")
            info = entry.get("info", {})
            for s, e in info.get("indx", []):
                segs.append(CalvinSegment(int(s), int(e), str(lang)))

        if len(segs) == 0:
            raise RuntimeError(f"No segments found in {ann_path}")

        self.segments = segs

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seg = self.segments[idx]
        start, end = seg.start, seg.end
        if end - start < self.action_horizon:
            # Fail-fast (or you can clamp)
            raise ValueError(f"segment too short: [{start},{end}) for H={self.action_horizon}")

        if self.sample_within_segment and (end - start > self.action_horizon):
            t = int(self.rng.integers(start, end - self.action_horizon + 1))
        else:
            t = start

        d0 = self.reader.read_npz(t)

        rgb_static = d0["rgb_static"]  # (200,200,3) uint8
        depth_static = d0["depth_static"]  # (200,200) float32/float64
        robot_obs = d0["robot_obs"].astype(np.float32)  # (15,)

        out: Dict[str, Any] = {
            "prompt": seg.lang,
            # Use calvin-native keys; later RepackTransform can map them.
            "rgb_static": rgb_static,
            "depth_static": depth_static.astype(np.float32),
            "robot_obs": robot_obs,
        }

        if self.use_wrist_rgb:
            out["rgb_gripper"] = d0["rgb_gripper"]  # (84,84,3) uint8

        # Stack action horizon
        acts = []
        for j in range(t, t + self.action_horizon):
            dj = self.reader.read_npz(j)
            acts.append(dj[self.action_key].astype(np.float32))
        out["actions"] = np.stack(acts, axis=0)  # (H,7)

        return out
