# src/openpi/training/calvin_dataset.py
'''
zip/dir 两种 backend

从 auto_lang_ann.npy 按段取样

observation 只取段内某个 t（图像/深度/状态），actions 取 t..t+H-1

'''


from __future__ import annotations

import io
import json
from collections.abc import Mapping
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

    def close(self) -> None:
        """Close any open zip handle (important for DataLoader spawn/pickling)."""
        if self._zf is not None:
            try:
                self._zf.close()
            except Exception:
                pass
        self._zf = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        # Drop non-picklable zipfile handle; each worker will re-open lazily.
        d = dict(self.__dict__)
        d["_zf"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._zf = None

    def _npz_path(self, step_id: int) -> str:
        # In zip, files look like: task_ABCD_D/training/episode_0538190.npz
        # In dir, same relative path.
        return f"task_ABCD_D/{self.split}/episode_{step_id:07d}.npz"

    def read_npz(self, step_id: int, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        rel = self._npz_path(step_id)
        if self.backend == "zip":
            raw = self._zip().read(rel)
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                if keys is None:
                    return {k: z[k] for k in z.files}
                out: Dict[str, np.ndarray] = {}
                for k in keys:
                    if k not in z.files:
                        raise KeyError(f"Missing key '{k}' in {rel}. Available keys: {list(z.files)}")
                    out[k] = z[k]
                return out
        else:
            abs_path = os.path.join(self.root, rel)
            with np.load(abs_path, allow_pickle=False) as z:
                if keys is None:
                    return {k: z[k] for k in z.files}
                out: Dict[str, np.ndarray] = {}
                for k in keys:
                    if k not in z.files:
                        raise KeyError(f"Missing key '{k}' in {abs_path}. Available keys: {list(z.files)}")
                    out[k] = z[k]
                return out

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
    
        # --- Robustly unwrap annotations across CALVIN releases ---
        # np.load(..., allow_pickle=True) may return:
        #   - ndarray(shape=(), dtype=object) wrapping the real python object (list/dict)
        #   - ndarray(shape=(N,), dtype=object) of dict-like entries
        #   - a dict wrapper
        if isinstance(ann, np.ndarray):
            if ann.ndim == 0:
                ann = ann.item()
            elif ann.dtype == object:
                ann = ann.tolist()

        if isinstance(ann, dict):
            # Preserve "Format A" dict that directly provides {"ann": ..., "indx": ...}.
            # If we unwrap "ann" here we would drop "indx" and break segment extraction.
            # This is O(1) and runs once in __init__ (no per-sample overhead).
            if ("ann" in ann) and ("indx" in ann) and not (("language" in ann) or ("info" in ann)):
                pass
            # If it's already a single "entry dict", treat it as one-element list.
            elif ("language" in ann) or ("info" in ann):
                ann = [ann]
            else:
                # Common wrapper keys (different preprocessors use different names).
                for k in ("annotations", "ann", "data", "entries", "arr_0"):
                    if k in ann:
                        ann = ann[k]
                        break
                else:
                    # Fallback: assume dict values are entries.
                    ann = list(ann.values())

        # Normalize any remaining ndarray wrapper
        if isinstance(ann, np.ndarray):
            if ann.ndim == 0:
                ann = ann.item()
            if ann.dtype == object:
                ann = ann.tolist()

        # Allow "Format A" top-level dict: {"ann": ..., "indx": ...}
        if not isinstance(ann, (list, tuple)) and not (
            isinstance(ann, dict) and ("ann" in ann) and ("indx" in ann)
        ):
            raise TypeError(
                f"Unexpected annotations container loaded from {ann_path}: "
                f"type={type(ann)}"
            )

        def _as_entry_dict(x: Any) -> Dict[str, Any]:
            """Convert one annotation entry to a plain dict."""
            # dict / Mapping
            if isinstance(x, Mapping):
                return dict(x)
            # 0-d numpy scalar wrapping a python object (often a dict)
            if isinstance(x, np.ndarray) and x.ndim == 0:
                try:
                    v = x.item()
                    if isinstance(v, Mapping):
                        return dict(v)
                except Exception:
                    pass
            # numpy structured scalar (np.void) with named fields
            if isinstance(x, np.void) and getattr(x.dtype, "names", None):
                return {k: x[k] for k in x.dtype.names}
            raise TypeError(f"Unexpected annotation entry type: {type(x)}")

        def _unwrap0(x: Any) -> Any:
            # np.load(..., allow_pickle=True) sometimes returns a 0-d object array.
            if isinstance(x, np.ndarray) and x.ndim == 0:
                return x.item()
            return x

        def _to_list(x: Any) -> List[Any] | None:
            x = _unwrap0(x)
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            return None

        def _clean_text(x: Any) -> str:
            x = _unwrap0(x)
            if x is None:
                return ""
            if isinstance(x, bytes):
                return x.decode("utf-8", "ignore").strip()
            if isinstance(x, str):
                return x.strip()
            # Common CALVIN formats: dict with "ann", or list/array of strings.
            if isinstance(x, dict) and "ann" in x:
                return _clean_text(x.get("ann"))
            if isinstance(x, (list, tuple, np.ndarray)):
                xs = _to_list(x) or []
                if len(xs) > 0:
                    return _clean_text(xs[0])
                return ""
            # Last resort: stringify (avoid doing this on big dicts by handling above).
            return str(x).strip()

        ann = _unwrap0(ann)
        segs: List[CalvinSegment] = []

        def _add_from_ann_and_indx(anns: Any, indx: Any) -> None:
            indx_list = _to_list(indx) or []
            if len(indx_list) == 0:
                return
            ann_list = _to_list(anns)
            # If we have one instruction per segment: zip them.
            if ann_list is not None and len(ann_list) == len(indx_list):
                for (s, e), txt in zip(indx_list, ann_list):
                    segs.append(CalvinSegment(int(s), int(e), _clean_text(txt)))
                return
            # Otherwise: broadcast a single instruction across all segments.
            txt = _clean_text(anns)
            for s, e in indx_list:
                segs.append(CalvinSegment(int(s), int(e), txt))

        # Format A: a single dict with "ann" + "indx"
        if isinstance(ann, dict) and ("indx" in ann) and ("ann" in ann) and not ("info" in ann):
            _add_from_ann_and_indx(ann.get("ann"), ann.get("indx"))
        else:
            entries = ann
            if isinstance(entries, np.ndarray):
                entries = entries.tolist()
            if isinstance(entries, dict):
                entries = [entries]
            for entry in (entries or []):
                entry = _unwrap0(entry)
                if not isinstance(entry, dict):
                    continue

                # Some files wrap everything under entry["info"]
                info = entry.get("info") if isinstance(entry.get("info"), dict) else entry
                indx = info.get("indx") if isinstance(info, dict) else None

                # Determine language annotations (per-segment or broadcast)
                anns = entry.get("language", None)
                if anns is None:
                    anns = info.get("ann", None) if isinstance(info, dict) else None
                if anns is None:
                    anns = entry.get("ann", None)

                # Sometimes "language" itself is a dict like {"ann": [...], "indx": [...]}
                if isinstance(anns, dict) and "ann" in anns:
                    if indx is None and "indx" in anns:
                        indx = anns.get("indx")
                    anns = anns.get("ann")

                _add_from_ann_and_indx(anns, indx)

        # Important: close zip handle opened during annotation loading so DataLoader(num_workers>0, spawn)
        # can pickle the dataset; each worker re-opens lazily.
        self.reader.close()

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

        # Read observation from step t.
        # Performance: only load keys we actually use for the observation + the first action.
        keys0 = ["rgb_static", "depth_static", "robot_obs", self.action_key]
        if self.use_wrist_rgb:
            keys0.append("rgb_gripper")
        d0 = self.reader.read_npz(t, keys=keys0)

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
        # Stack action horizon. Avoid re-loading step t (we already have it in d0).
        # Performance: for j>t, load only the action key (avoid decompressing rgb/depth repeatedly).
        acts = [d0[self.action_key].astype(np.float32)]
        for j in range(t + 1, t + self.action_horizon):
            dj = self.reader.read_npz(j, keys=[self.action_key])
            acts.append(dj[self.action_key].astype(np.float32))
        out["actions"] = np.stack(acts, axis=0)  # (H,7)

        return out
