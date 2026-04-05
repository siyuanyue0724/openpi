from __future__ import annotations

import io
import json
import os
from typing import Any
import zipfile

import numpy as np


def infer_task_prefix_from_zip(zf: zipfile.ZipFile) -> str | None:
    names = zf.namelist()
    suffixes = (
        "/training/lang_annotations/auto_lang_ann.npy",
        "/validation/lang_annotations/auto_lang_ann.npy",
        "/calib/cameras.json",
    )
    cands: set[str] = set()
    for suffix in suffixes:
        for name in names:
            if name.endswith(suffix):
                cands.add(name[: -len(suffix)].rstrip("/"))
    cands = {candidate for candidate in cands if candidate}
    if len(cands) == 1:
        return next(iter(cands))
    tops = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(tops) == 1:
        return next(iter(tops))
    return None


def load_json(path: str) -> dict[str, Any]:
    if os.path.isdir(path):
        candidate = os.path.join(path, "calib", "cameras.json")
        if os.path.exists(candidate):
            path = candidate

    if "::" in path:
        zip_path, inner_path = path.split("::", 1)
        zip_path = zip_path.strip()
        inner_path = inner_path.lstrip("/").strip()
        if not os.path.exists(zip_path):
            raise FileNotFoundError(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf, zf.open(inner_path, "r") as handle:
            return json.load(io.TextIOWrapper(handle, encoding="utf-8"))

    if path.endswith(".zip") and os.path.exists(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            candidates: list[str] = []
            default_inner = f"{stem}/calib/cameras.json"
            if default_inner in names:
                candidates.append(default_inner)
            inferred = infer_task_prefix_from_zip(zf)
            if inferred is not None:
                inferred_inner = f"{inferred}/calib/cameras.json"
                if inferred_inner in names and inferred_inner not in candidates:
                    candidates.append(inferred_inner)
            for name in names:
                if (name.endswith("/calib/cameras.json") or name == "calib/cameras.json") and name not in candidates:
                    candidates.append(name)
            if candidates:
                with zf.open(candidates[0], "r") as handle:
                    return json.load(io.TextIOWrapper(handle, encoding="utf-8"))
        sibling = path[:-4]
        alt = os.path.join(sibling, "calib", "cameras.json")
        if os.path.exists(alt):
            with open(alt, encoding="utf-8") as handle:
                return json.load(handle)
        raise FileNotFoundError(f"cameras.json not found for zip '{path}'")

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def as_4x4(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.shape == (4, 4):
        return value
    if value.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        return np.concatenate([value, bottom], axis=0)
    raise ValueError(f"Expected matrix shape (4,4) or (3,4), got {value.shape}")


def as_3x3(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.shape == (3, 3):
        return value
    if value.size == 9:
        return value.reshape(3, 3)
    raise ValueError(f"Expected matrix shape (3,3) or flat 9, got {value.shape}")
