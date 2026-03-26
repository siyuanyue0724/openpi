#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALVIN cameras.json rebuilder / regression runner (single-file, safe output)

What this script does
---------------------
1) Reads Hydra config directly from task_ABCD_D.zip (no full unzip).
2) Recomputes static camera intrinsics/extrinsics from official Hydra.
3) Recomputes URDF raw hand-eye candidates for multiple link pairs.
4) Writes everything under ./test/calib by default, never overwriting the
   original ./dataset/task_ABCD_D/calib.
5) If an original cameras.json is found (or passed via --reference-cameras),
   it uses it as a regression target to rebuild a *parity* cameras.json:
      - static is recomputed from Hydra
      - gripper E_ref is taken from the historical file
      - Delta is recomputed as inv(E_raw) @ E_ref using the fresh URDF raw pose
      - strong equivalence / depth checks are re-run
6) If no historical cameras.json is available, it emits a best-effort URDF-only
   cameras.json by selecting the best raw candidate via depth consistency.
7) Optionally exports gripper_poses-*.parquet to ./test/calib.

Design notes
------------
- This script is intentionally lightweight: it borrows the *idea* of generating
  camera parameters from CALVIN state + simulator metadata, but does not depend
  on RoboUniView's rendering/visualization stack.
- The exact historical dataset-calibration step that created E_ref/Delta does
  not survive in the uploaded scripts. The surviving local scripts consume
  cameras.json (verify/export) and load Delta from it, rather than generating it.
  Therefore, when an original cameras.json exists, this script uses it as the
  regression target and rebuilds an equivalent file under ./test/calib.

Typical usage
-------------
python calvin_rebuild_cameras_test.py \
  --zip dataset/task_ABCD_D.zip \
  --repo-root . \
  --out-root ./test \
  --reference-cameras dataset/task_ABCD_D/calib/cameras.json \
  --verify-episodes 40 \
  --candidate-episodes 12 \
  --rpy-order zyx \
  --t-mode auto \
  --bilinear

Optional smoke export of parquet:
python calvin_rebuild_cameras_test.py \
  --zip dataset/task_ABCD_D.zip \
  --repo-root . \
  --out-root ./test \
  --reference-cameras dataset/task_ABCD_D/calib/cameras.json \
  --export-parquet --export-episodes-max 2000
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ------------------------- small linalg -------------------------

def Rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def Rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def T_from_R_t(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def R_from_rpy(rpy: Sequence[float], order: str = "zyx") -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    if order == "zyx":
        return Rz(y) @ Ry(p) @ Rx(r)
    if order == "xyz":
        return Rx(r) @ Ry(p) @ Rz(y)
    raise ValueError(f"unsupported rpy order: {order}")


def W_T_from_posrpy(pos: Sequence[float], rpy: Sequence[float], order: str = "zyx") -> np.ndarray:
    return T_from_R_t(R_from_rpy(rpy, order=order), pos)


def fro_err(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(A, float) - np.asarray(B, float), ord="fro"))


def quat_xyzw_from_cfg(x) -> List[float]:
    if isinstance(x, (list, tuple)) and len(x) == 4:
        return [float(v) for v in x]
    if isinstance(x, (list, tuple)) and len(x) == 3:
        r, p, y = [float(v) for v in x]
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        w = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return [qx, qy, qz, w]
    return [0.0, 0.0, 0.0, 1.0]


# ------------------------- data classes -------------------------

@dataclass
class StaticCam:
    K: np.ndarray
    W_T_C: np.ndarray
    C_T_W_gl: np.ndarray
    C_T_W_cv: np.ndarray
    H: int
    W: int
    fov: float
    near: float
    far: float
    hydra_path: str


@dataclass
class GripperCamIntr:
    K: np.ndarray
    H: int
    W: int
    fov: float
    near: float
    far: float


@dataclass
class Candidate:
    ref_link_id: int
    cam_link_id: int
    ref_label: str
    E_raw: np.ndarray
    file_name: str
    depth_val: Optional[dict] = None
    depth_train: Optional[dict] = None


# ------------------------- filesystem helpers -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)



def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def tolist_f32(a: np.ndarray) -> List:
    return np.asarray(a, dtype=float).tolist()


# ------------------------- Hydra / camera reconstruction -------------------------

def load_hydra_from_zip(zip_path: Path, preferred_split: str = "validation") -> Tuple[dict, str]:
    cand = [
        f"task_ABCD_D/{preferred_split}/.hydra/merged_config.yaml",
        "task_ABCD_D/validation/.hydra/merged_config.yaml",
        "task_ABCD_D/training/.hydra/merged_config.yaml",
    ]
    with zipfile.ZipFile(zip_path, "r") as zf:
        for p in cand:
            try:
                with zf.open(p, "r") as f:
                    return yaml.safe_load(f), p
            except KeyError:
                continue
    raise FileNotFoundError("Could not find merged_config.yaml inside zip")



def intrinsic_from_fov(width: int, height: int, fov_deg: float, aspect: float = 1.0) -> np.ndarray:
    # README baseline: fx ≈ H / (2 * tan(fov/2)). For square CALVIN cameras, fx == fy.
    fy = float(height) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    fx = fy * float(aspect)
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)



def view_from_lookat(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=float)
    center = np.asarray(center, dtype=float)
    up = np.asarray(up, dtype=float)
    f = center - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=float)
    M[:3, :3] = np.vstack([r, u, -f])
    M[:3, 3] = -(M[:3, :3] @ eye)
    return M



def build_static_from_hydra(cfg: dict, hydra_path: str) -> StaticCam:
    cams = cfg.get("cameras") or (cfg.get("env") or {}).get("cameras")
    if cams is None or "static" not in cams:
        raise KeyError("Hydra cameras.static not found")
    s = cams["static"]
    W = int(s["width"])
    H = int(s["height"])
    fov = float(s["fov"])
    aspect = float(s.get("aspect", 1.0))
    near = float(s.get("nearval", s.get("near", 0.01)))
    far = float(s.get("farval", s.get("far", 10.0)))
    eye = np.array(s["look_from"], float)
    center = np.array(s["look_at"], float)
    up = np.array(s.get("up_vector", [0.0, 0.0, 1.0]), float)

    K = intrinsic_from_fov(W, H, fov, aspect)
    C_T_W_gl = view_from_lookat(eye, center, up)
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    C_T_W_cv = F @ C_T_W_gl
    W_T_C_cv = inv_T(C_T_W_cv)
    return StaticCam(
        K=K,
        W_T_C=W_T_C_cv,
        C_T_W_gl=C_T_W_gl,
        C_T_W_cv=C_T_W_cv,
        H=H,
        W=W,
        fov=fov,
        near=near,
        far=far,
        hydra_path=hydra_path,
    )



def build_gripper_intr_from_hydra(cfg: dict) -> GripperCamIntr:
    cams = cfg.get("cameras") or (cfg.get("env") or {}).get("cameras")
    if cams is None or "gripper" not in cams:
        raise KeyError("Hydra cameras.gripper not found")
    g = cams["gripper"]
    W = int(g["width"])
    H = int(g["height"])
    fov = float(g["fov"])
    aspect = float(g.get("aspect", 1.0))
    near = float(g.get("nearval", g.get("near", 0.01)))
    far = float(g.get("farval", g.get("far", 2.0)))
    K = intrinsic_from_fov(W, H, fov, aspect)
    return GripperCamIntr(K=K, H=H, W=W, fov=fov, near=near, far=far)


# ------------------------- URDF / pybullet helpers -------------------------

def require_pybullet():
    try:
        import pybullet as p  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pybullet is required for URDF-based reconstruction. Install it in the current env."
        ) from e
    return p



def resolve_urdf_path(cfg: dict, repo_root: Path, override: Optional[Path]) -> Path:
    if override is not None:
        p = override.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--urdf not found: {p}")
        return p
    robot = cfg.get("robot") or {}
    fname = robot.get("filename")
    if not fname:
        raise KeyError("robot.filename not found in hydra")
    cand = [
        repo_root / "calvin_env" / "data" / str(fname),
        repo_root / str(fname),
        Path(str(fname)).expanduser(),
    ]
    for p in cand:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"Unable to resolve URDF from robot.filename={fname!r}")



def build_pb_body(urdf_path: Path, base_pos: Sequence[float], base_quat_xyzw: Sequence[float]):
    p = require_pybullet()
    cid = p.connect(p.DIRECT)
    body = p.loadURDF(
        str(urdf_path),
        basePosition=[float(v) for v in base_pos],
        baseOrientation=[float(v) for v in base_quat_xyzw],
        useFixedBase=True,
    )
    return p, cid, body



def pb_disconnect(p, cid: int) -> None:
    try:
        p.disconnect(cid)
    except Exception:
        pass



def pb_link_state_T(p, body: int, link_id: int) -> np.ndarray:
    pos, orn = p.getLinkState(body, link_id, computeForwardKinematics=True)[4:6]
    R = np.array(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
    return T_from_R_t(R, pos)



def pb_reset_arm_q(p, body: int, q: Sequence[float], arm_joint_ids: Sequence[int]) -> None:
    for jid, qv in zip(arm_joint_ids, q):
        p.resetJointState(body, int(jid), float(qv))



def get_link_table(p, body: int) -> List[dict]:
    out = []
    nj = p.getNumJoints(body)
    for j in range(nj):
        info = p.getJointInfo(body, j)
        joint_name = info[1].decode("utf-8", errors="ignore") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
        link_name = info[12].decode("utf-8", errors="ignore") if isinstance(info[12], (bytes, bytearray)) else str(info[12])
        out.append({"joint_index": j, "joint_name": joint_name, "link_name": link_name})
    return out



def compute_E_raw_from_urdf(p, body: int, ref_link_id: int, cam_link_id: int) -> np.ndarray:
    # zero pose (all joints zero)
    nj = p.getNumJoints(body)
    for j in range(nj):
        p.resetJointState(body, j, 0.0)
    W_T_ref = pb_link_state_T(p, body, ref_link_id)
    W_T_cam = pb_link_state_T(p, body, cam_link_id)
    return inv_T(W_T_ref) @ W_T_cam


# ------------------------- zip / robot_obs helpers -------------------------

_EP_RE = re.compile(r"^task_ABCD_D/(training|validation)(?:/.*)?/episode_(\d+)\.npz$")


def list_episode_paths(zf: zipfile.ZipFile, split: Optional[str] = None) -> List[str]:
    names = [n for n in zf.namelist() if _EP_RE.match(n)]
    if split is not None:
        names = [n for n in names if f"task_ABCD_D/{split}/" in n]
    names.sort(key=lambda x: int(_EP_RE.match(x).group(2)))
    return names



def norm_robot_obs(arr) -> np.ndarray:
    A = arr
    try:
        if isinstance(A, np.ndarray) and A.dtype != object:
            if A.ndim == 2:
                return A.astype(np.float32)
            if A.ndim == 1:
                return A.astype(np.float32).reshape(1, -1)
            return np.zeros((0, 0), np.float32)
        if isinstance(A, np.ndarray) and A.dtype == object:
            rows = []
            for x in A:
                x = np.asarray(x)
                x = np.ravel(x).astype(np.float32)
                if x.size == 0:
                    return np.zeros((0, 0), np.float32)
                rows.append(x)
            D = max(r.size for r in rows)
            out = np.zeros((len(rows), D), np.float32)
            for i, r in enumerate(rows):
                out[i, : len(r)] = r
            return out
        B = np.asarray(A)
        if B.ndim == 2:
            return B.astype(np.float32)
        if B.ndim == 1:
            return B.astype(np.float32).reshape(1, -1)
        return np.zeros((0, 0), np.float32)
    except Exception:
        return np.zeros((0, 0), np.float32)



def plausible_pos(P: np.ndarray) -> bool:
    return bool(P.ndim == 2 and P.shape[1] == 3 and np.isfinite(P).all() and np.abs(P).max() <= 5 and np.std(P, 0).mean() > 1e-5)



def plausible_rpy(R: np.ndarray) -> bool:
    return bool(R.ndim == 2 and R.shape[1] == 3 and np.isfinite(R).all() and np.abs(R).max() <= 3.2 and np.std(R, 0).mean() > 1e-5)



def autodetect_posrpy_index(zf: zipfile.ZipFile, names: Sequence[str], probe: int = 50) -> int:
    votes: Dict[int, int] = {}
    scanned = 0
    for n in names[:probe]:
        try:
            d = np.load(io.BytesIO(zf.read(n)), allow_pickle=True)
        except Exception:
            continue
        if "robot_obs" not in d.files:
            continue
        A = norm_robot_obs(d["robot_obs"])
        if A.size == 0 or A.shape[1] < 6:
            continue
        scanned += 1
        if plausible_pos(A[:, 0:3]) and plausible_rpy(A[:, 3:6]):
            votes[0] = votes.get(0, 0) + 1
            continue
        for i in range(0, A.shape[1] - 5):
            if plausible_pos(A[:, i : i + 3]) and plausible_rpy(A[:, i + 3 : i + 6]):
                votes[i] = votes.get(i, 0) + 1
                break
    if not votes:
        return 0
    idx, _ = max(votes.items(), key=lambda kv: kv[1])
    return int(idx)



def guess_q_start(A: np.ndarray) -> Optional[int]:
    if A.ndim != 2 or A.shape[1] < 14:
        return None
    if A.shape[1] >= 14:
        q = A[:, 7:14]
        if np.isfinite(q).all() and np.abs(q).max() <= 4.0:
            return 7
    for i in range(0, max(0, A.shape[1] - 6)):
        q = A[:, i : i + 7]
        if q.shape[1] == 7 and np.isfinite(q).all() and np.abs(q).max() <= 4.0:
            return i
    return None


# ------------------------- depth verification helpers -------------------------

def depth_to_meters(d: np.ndarray, near: float, far: float) -> np.ndarray:
    d = np.asarray(d, np.float32)
    if d.size == 0:
        return d
    mx = float(np.nanmax(d))
    mn = float(np.nanmin(d))
    if np.isfinite(mn) and mn >= 0.0 and mx <= 1.5:
        return (far * near) / (far - (far - near) * d)
    return d



def pick_frame(arr, t: int, H: int, W: int) -> Optional[np.ndarray]:
    a = np.asarray(arr)
    if a.ndim == 2:
        if a.shape == (H, W):
            return a
        if a.shape == (W, H):
            return a.T
        return None
    if a.ndim == 3:
        cands = []
        if a.shape == (H, W, a.shape[2]):
            cands.append(a[:, :, min(t, a.shape[2] - 1)])
        if a.shape == (H, a.shape[1], W):
            cands.append(a[:, min(t, a.shape[1] - 1), :])
        if a.shape == (a.shape[0], H, W):
            cands.append(a[min(t, a.shape[0] - 1)])
        for c in cands:
            if c.shape == (H, W):
                return c
    return None



def edge_mask(ds: np.ndarray, thr: float = 0.03) -> np.ndarray:
    gy = np.abs(np.diff(ds, axis=0, prepend=ds[:1, :]))
    gx = np.abs(np.diff(ds, axis=1, prepend=ds[:, :1]))
    return (gx + gy) < float(thr)



def bilinear_sample(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    u = np.clip(u, 0, W - 1 - 1e-6)
    v = np.clip(v, 0, H - 1 - 1e-6)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    du = u - u0
    dv = v - v0
    d00 = depth[v0, u0]
    d10 = depth[v0, u1]
    d01 = depth[v1, u0]
    d11 = depth[v1, u1]
    return d00 * (1 - du) * (1 - dv) + d10 * du * (1 - dv) + d01 * (1 - du) * dv + d11 * du * dv



def build_depth_metric_dict(errs: np.ndarray, frames: int, pixels: int, choose_sem: dict, choose_t: dict) -> dict:
    if errs.size == 0:
        return {
            "ok": False,
            "frames": int(frames),
            "pixels": int(pixels),
            "message": "no valid depth correspondences",
        }
    return {
        "ok": True,
        "frames": int(frames),
        "pixels": int(pixels),
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "p90": float(np.percentile(errs, 90)),
        "lt_5cm": float((errs < 0.05).mean()),
        "choose_sem": dict(choose_sem),
        "choose_t": dict(choose_t),
    }



def verify_depth_for_E(
    zip_path: Path,
    cfg: dict,
    static: StaticCam,
    gripper: GripperCamIntr,
    E_T_C: np.ndarray,
    ref_link_id: int,
    urdf_path: Optional[Path],
    split: str,
    episodes: int,
    q_start: Optional[int],
    posrpy_start: int,
    arm_joint_ids: Sequence[int],
    rpy_order: str,
    t_mode: str,
    max_pix: int,
    edge_thr: float,
    bilinear: bool,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    K_s = static.K
    C_T_W_s = inv_T(static.W_T_C)
    K_g = gripper.K
    Kinv_g = np.linalg.inv(K_g)
    Hs, Ws = static.H, static.W
    Hg, Wg = gripper.H, gripper.W
    ns, fs = static.near, static.far
    ng, fg = gripper.near, gripper.far

    scene = cfg.get("scene") or {}
    base_pos = [float(v) for v in scene.get("robot_base_position", [0, 0, 0])]
    base_quat_xyzw = quat_xyzw_from_cfg(scene.get("robot_base_orientation", [0, 0, 0, 1]))

    pb_ctx = None
    if urdf_path is not None and q_start is not None and len(arm_joint_ids) >= 7:
        pb_ctx = build_pb_body(urdf_path, base_pos, base_quat_xyzw)

    errs: List[float] = []
    frames = 0
    choose_sem = {"zz": 0, "zr": 0, "rz": 0, "rr": 0}
    choose_t = {"start": 0, "mid": 0, "end": 0}

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = list_episode_paths(zf, split=split)
        random.shuffle(members)
        members = members[: int(episodes)]
        try:
            for ref in members:
                d = np.load(io.BytesIO(zf.read(ref)), allow_pickle=True)
                if not ({"robot_obs", "depth_gripper", "depth_static"} <= set(d.files)):
                    continue
                A = norm_robot_obs(d["robot_obs"])
                if A.size == 0:
                    continue
                T = A.shape[0]

                dg0 = pick_frame(d["depth_gripper"], 0, Hg, Wg)
                ds0 = pick_frame(d["depth_static"], 0, Hs, Ws)
                if dg0 is None or ds0 is None:
                    continue
                dg = depth_to_meters(dg0, ng, fg)
                ds = depth_to_meters(ds0, ns, fs)
                mask_edge = edge_mask(ds, edge_thr) if edge_thr > 0 else np.ones_like(ds, dtype=bool)

                if t_mode == "auto":
                    cand_t = [("start", 0), ("mid", max(0, T // 2)), ("end", max(0, T - 1))]
                elif t_mode == "start":
                    cand_t = [("start", 0)]
                elif t_mode == "mid":
                    cand_t = [("mid", max(0, T // 2))]
                else:
                    cand_t = [("end", max(0, T - 1))]

                best = None
                best_sem = None
                best_tag = None
                for tag, t in cand_t:
                    # Build W_T_ref either from joints+URDF or fallback pos+rpy.
                    if pb_ctx is not None and A.shape[1] >= q_start + 7:
                        p, cid, body = pb_ctx
                        q = A[t, q_start : q_start + 7]
                        pb_reset_arm_q(p, body, q, arm_joint_ids[:7])
                        W_T_ref = pb_link_state_T(p, body, ref_link_id)
                    else:
                        if A.shape[1] < posrpy_start + 6:
                            continue
                        pos = A[t, posrpy_start : posrpy_start + 3]
                        rpy = A[t, posrpy_start + 3 : posrpy_start + 6]
                        W_T_ref = W_T_from_posrpy(pos, rpy, order=rpy_order)
                    W_T_Cg = W_T_ref @ E_T_C

                    valid = np.argwhere(np.isfinite(dg) & (dg > ng + 1e-6) & (dg < fg - 1e-6))
                    if valid.size == 0:
                        continue
                    take = min(int(max_pix), valid.shape[0])
                    sel = valid[np.random.choice(valid.shape[0], take, replace=False)]
                    u_idx = sel[:, 1]
                    v_idx = sel[:, 0]
                    dd = dg[v_idx, u_idx].astype(float)
                    u = u_idx.astype(float)
                    v = v_idx.astype(float)

                    homog = np.stack([u, v, np.ones_like(u)], axis=1).T
                    ray = (Kinv_g @ homog).T
                    ray_n = ray / np.linalg.norm(ray, axis=1, keepdims=True)
                    Xcg_Z = ray * dd.reshape(-1, 1)
                    Xcg_R = ray_n * dd.reshape(-1, 1)

                    def project_and_err(Xcg: np.ndarray, static_mode: str) -> Optional[np.ndarray]:
                        Xw = (W_T_Cg @ np.hstack([Xcg, np.ones((len(Xcg), 1))]).T).T[:, :3]
                        Xc = (C_T_W_s @ np.hstack([Xw, np.ones((len(Xw), 1))]).T).T[:, :3]
                        z = Xc[:, 2]
                        rho = np.linalg.norm(Xc, axis=1)
                        uvh = (K_s @ Xc.T).T
                        uu = uvh[:, 0] / uvh[:, 2]
                        vv = uvh[:, 1] / uvh[:, 2]
                        m = (
                            (uu >= 0)
                            & (uu < Ws)
                            & (vv >= 0)
                            & (vv < Hs)
                            & np.isfinite(z)
                            & (z > ns + 1e-6)
                            & (z < fs - 1e-6)
                        )
                        if not np.any(m):
                            return None
                        uu = uu[m]
                        vv = vv[m]
                        if bilinear:
                            ds_sample = bilinear_sample(ds, uu, vv)
                            ui_nn = np.clip(np.rint(uu).astype(int), 0, Ws - 1)
                            vi_nn = np.clip(np.rint(vv).astype(int), 0, Hs - 1)
                            me = mask_edge[vi_nn, ui_nn]
                        else:
                            ui = np.clip(np.rint(uu).astype(int), 0, Ws - 1)
                            vi = np.clip(np.rint(vv).astype(int), 0, Hs - 1)
                            ds_sample = ds[vi, ui]
                            me = mask_edge[vi, ui]
                        if not np.any(me):
                            return None
                        pred = z[m][me] if static_mode == "Z" else rho[m][me]
                        obs = ds_sample[me]
                        good = np.isfinite(obs)
                        if not np.any(good):
                            return None
                        return np.abs(obs[good] - pred[good])

                    combos = [("zz", Xcg_Z, "Z"), ("zr", Xcg_Z, "R"), ("rz", Xcg_R, "Z"), ("rr", Xcg_R, "R")]
                    best_local = None
                    best_sem_local = None
                    for key, Xcg_try, smode in combos:
                        e = project_and_err(Xcg_try, smode)
                        if e is not None and (best_local is None or np.median(e) < np.median(best_local)):
                            best_local = e
                            best_sem_local = key
                    if best_local is not None and (best is None or np.median(best_local) < np.median(best)):
                        best = best_local
                        best_sem = best_sem_local
                        best_tag = tag

                if best is not None:
                    errs.extend(best.tolist())
                    frames += 1
                    choose_sem[str(best_sem)] += 1
                    choose_t[str(best_tag)] += 1
        finally:
            if pb_ctx is not None:
                p, cid, _ = pb_ctx
                pb_disconnect(p, cid)

    return build_depth_metric_dict(np.asarray(errs, float), frames, len(errs), choose_sem, choose_t)


# ------------------------- equivalence checks -------------------------

def urdf_q_invariance_and_delta_equiv(
    urdf_path: Path,
    base_pos: Sequence[float],
    base_quat_xyzw: Sequence[float],
    ee_link_id: int,
    cam_link_id: int,
    Delta: np.ndarray,
    E_ref: np.ndarray,
    arm_joint_ids: Sequence[int],
    trials: int = 40,
    seed: int = 0,
) -> dict:
    p, cid, body = build_pb_body(urdf_path, base_pos, base_quat_xyzw)
    random.seed(seed)
    try:
        for jid in arm_joint_ids[:7]:
            p.resetJointState(body, int(jid), 0.0)
        E0 = inv_T(pb_link_state_T(p, body, ee_link_id)) @ pb_link_state_T(p, body, cam_link_id)
        mx_ang = 0.0
        mx_terr = 0.0
        mx_fro_raw = 0.0
        mx_fro_ref = 0.0
        for _ in range(int(trials)):
            q = [random.uniform(-0.4, 0.4) for _ in range(min(7, len(arm_joint_ids)))]
            pb_reset_arm_q(p, body, q, arm_joint_ids[:7])
            E = inv_T(pb_link_state_T(p, body, ee_link_id)) @ pb_link_state_T(p, body, cam_link_id)

            R, t = E[:3, :3], E[:3, 3]
            R0, t0 = E0[:3, :3], E0[:3, 3]
            Rd = R0.T @ R
            c = max(-1.0, min(1.0, (np.trace(Rd) - 1) / 2))
            ang_deg = math.degrees(math.acos(c))
            terr = float(np.linalg.norm(t - t0))
            fro_raw = float(np.linalg.norm(E - E0))
            fro_ref = float(np.linalg.norm(E @ Delta - E_ref))
            mx_ang = max(mx_ang, ang_deg)
            mx_terr = max(mx_terr, terr)
            mx_fro_raw = max(mx_fro_raw, fro_raw)
            mx_fro_ref = max(mx_fro_ref, fro_ref)
        return {
            "max_rot_deg": mx_ang,
            "max_trans_m": mx_terr,
            "max_fro_raw": mx_fro_raw,
            "max_fro_ref": mx_fro_ref,
            "pass_tol_1e-6": bool(mx_fro_raw < 1e-6 and mx_fro_ref < 1e-6),
        }
    finally:
        pb_disconnect(p, cid)


# ------------------------- cameras.json builders -------------------------

ROWFLIP_4X4 = np.diag([1.0, -1.0, -1.0, 1.0])



def legacy_conventions_dict(reference_json: Optional[dict] = None) -> dict:
    base = {
        "intrinsics": "f=H/(2*tan(fov/2)), cx=W/2, cy=H/2",
        "static_extrinsics": "OpenGL C_T_W from viewMatrix; OpenCV C_T_W = rowflip(OpenGL); W_T_C=inv(OpenCV C_T_W)",
        "gripper_handeye": "E_T_C from URDF (OpenGL); OpenCV version = rowflip(OpenGL)",
        "rowflip": "row-wise multiply by diag([1,-1,-1,1])",
    }
    if isinstance(reference_json, dict) and isinstance(reference_json.get("conventions"), dict):
        merged = dict(base)
        merged.update(reference_json["conventions"])
        return merged
    return base



def _matrix_from_json(d: dict, key: str) -> Optional[np.ndarray]:
    try:
        if isinstance(d, dict) and key in d:
            return np.array(d[key], dtype=float)
    except Exception:
        return None
    return None



def legacy_ref_frame_string(ref_link_id: int, cam_link_id: int, tcp_link_id: int, ee_link_id: int) -> str:
    label = ref_label(ref_link_id, tcp_link_id=tcp_link_id, ee_link_id=ee_link_id)
    if label == "EE":
        return f"EE({ref_link_id})->cam({cam_link_id})"
    if label == "TCP":
        return f"TCP({ref_link_id})->cam({cam_link_id})"
    return f"link({ref_link_id})->cam({cam_link_id})"



def static_to_json_dict(static: StaticCam) -> dict:
    return {
        "W": int(static.W),
        "H": int(static.H),
        "fov": float(static.fov),
        "near": float(static.near),
        "far": float(static.far),
        "K": tolist_f32(static.K),
        "extrinsic_opengl_4x4": tolist_f32(static.C_T_W_gl),
        "extrinsic_opencv_4x4": tolist_f32(static.C_T_W_cv),
        "W_T_C": tolist_f32(static.W_T_C),
    }



def gripper_intr_to_json_dict(
    gripper: GripperCamIntr,
    E_T_C: np.ndarray,
    *,
    ref_link_id: int,
    cam_link_id: int,
    tcp_link_id: int,
    ee_link_id: int,
    urdf_path: Path,
    reference_json: Optional[dict] = None,
    preserve_reference_legacy: bool = False,
) -> dict:
    gref = reference_json.get("gripper", {}) if isinstance(reference_json, dict) else {}
    E_cv = np.asarray(E_T_C, dtype=float)
    E_ogl_guess = ROWFLIP_4X4 @ E_cv
    E_ogl = _matrix_from_json(gref, "E_T_C_opengl_4x4") if preserve_reference_legacy else None
    E_cv_aux = _matrix_from_json(gref, "E_T_C_opencv_4x4") if preserve_reference_legacy else None
    if E_ogl is None:
        E_ogl = E_ogl_guess
    if E_cv_aux is None:
        E_cv_aux = E_cv
    return {
        "W": int(gripper.W),
        "H": int(gripper.H),
        "fov": float(gripper.fov),
        "near": float(gripper.near),
        "far": float(gripper.far),
        "K": tolist_f32(gripper.K),
        "E_T_C_opengl_4x4": tolist_f32(E_ogl),
        "E_T_C_opencv_4x4": tolist_f32(E_cv_aux),
        "E_T_C": tolist_f32(E_cv),
        "end_effector_link_id": int(gref.get("end_effector_link_id", ref_link_id)),
        "gripper_cam_link": int(gref.get("gripper_cam_link", cam_link_id)),
        "urdf_path": str(gref.get("urdf_path", str(urdf_path))),
        "ref_frame": str(gref.get("ref_frame", legacy_ref_frame_string(ref_link_id, cam_link_id, tcp_link_id, ee_link_id))),
    }



def make_cameras_json(
    static: StaticCam,
    gripper: GripperCamIntr,
    E_T_C: np.ndarray,
    meta: dict,
    *,
    ref_link_id: int,
    cam_link_id: int,
    tcp_link_id: int,
    ee_link_id: int,
    urdf_path: Path,
    reference_json: Optional[dict] = None,
    preserve_reference_legacy: bool = False,
) -> dict:
    return {
        "hydra_path": static.hydra_path,
        "static": static_to_json_dict(static),
        "gripper": gripper_intr_to_json_dict(
            gripper,
            E_T_C,
            ref_link_id=ref_link_id,
            cam_link_id=cam_link_id,
            tcp_link_id=tcp_link_id,
            ee_link_id=ee_link_id,
            urdf_path=urdf_path,
            reference_json=reference_json,
            preserve_reference_legacy=preserve_reference_legacy,
        ),
        "conventions": legacy_conventions_dict(reference_json),
        "meta": meta,
    }



def ref_label(ref_link_id: int, tcp_link_id: int, ee_link_id: int) -> str:
    if ref_link_id == tcp_link_id:
        return "TCP"
    if ref_link_id == ee_link_id:
        return "EE"
    return f"link{ref_link_id}"



def build_candidate_meta(
    *,
    urdf_path: Path,
    ref_link_id: int,
    cam_link_id: int,
    rpy_order: str,
    variant_type: str,
    selection_note: str,
    tcp_link_id: Optional[int] = None,
) -> dict:
    return {
        "axis": "opencv",
        "coord_convention": "opencv",
        "depth_unit": "meter",
        "rpy_order": rpy_order,
        "urdf": str(urdf_path),
        "ee_link_id": int(ref_link_id),
        "urdf_cam_link_id": int(cam_link_id),
        "gripper_cam_link_id": int(cam_link_id),
        "tcp_link_id": int(ref_link_id if tcp_link_id is None else tcp_link_id),
        "base_used": False,
        "urdf_to_dataset_delta_4x4": tolist_f32(np.eye(4)),
        "variant_type": variant_type,
        "selection_note": selection_note,
    }



def auto_reference_path(zip_path: Path) -> Optional[Path]:
    cand = zip_path.with_suffix("") / "calib" / "cameras.json"
    return cand if cand.is_file() else None


# ------------------------- parquet export -------------------------

def export_parquet_from_zip(
    zip_path: Path,
    out_dir: Path,
    E_T_C: np.ndarray,
    rpy_order: str,
    t_mode: str,
    episodes_max: Optional[int],
    shard_size: int,
) -> dict:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required for --export-parquet") from e

    def write_shard(shard_idx: int, rows: list) -> int:
        if not rows:
            return shard_idx
        ep = [r[0] for r in rows]
        spl = [r[1] for r in rows]
        eid = [r[2] for r in rows]
        tix = [r[3] for r in rows]
        ee = [r[4] for r in rows]
        cg = [r[5] for r in rows]
        tab = pa.Table.from_arrays(
            [
                pa.array(ep, pa.string()),
                pa.array(spl, pa.string()),
                pa.array(eid, pa.int32()),
                pa.array(tix, pa.int32()),
                pa.array(ee, type=pa.list_(pa.float32())),
                pa.array(cg, type=pa.list_(pa.float32())),
            ],
            names=["episode_path", "split", "episode_id", "t_index", "W_T_EE_3x4", "W_T_Cgripper_3x4"],
        )
        outp = out_dir / f"gripper_poses-{shard_idx:06d}.parquet"
        pq.write_table(tab, outp, compression="ZSTD")
        rows.clear()
        return shard_idx + 1

    rows = []
    shard_idx = 0
    total_ok = 0
    total_bad = 0

    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = list_episode_paths(zf, split=None)
        if episodes_max is not None:
            names = names[: int(episodes_max)]
        posrpy_idx = autodetect_posrpy_index(zf, names, probe=50)
        for n in names:
            m = _EP_RE.match(n)
            if not m:
                total_bad += 1
                continue
            split, eid = m.group(1), int(m.group(2))
            try:
                d = np.load(io.BytesIO(zf.read(n)), allow_pickle=True)
            except Exception:
                total_bad += 1
                continue
            if "robot_obs" not in d.files:
                total_bad += 1
                continue
            A = norm_robot_obs(d["robot_obs"])
            if A.size == 0 or A.shape[1] < posrpy_idx + 6:
                total_bad += 1
                continue
            T = A.shape[0]
            t = 0 if t_mode == "start" else (T - 1 if t_mode == "end" else T // 2)
            pos = A[t, posrpy_idx : posrpy_idx + 3]
            rpy = A[t, posrpy_idx + 3 : posrpy_idx + 6]
            W_T_EE = W_T_from_posrpy(pos, rpy, order=rpy_order).astype(np.float32)
            W_T_Cg = (W_T_EE @ np.asarray(E_T_C, dtype=np.float32)).astype(np.float32)
            rows.append(
                (
                    n,
                    split,
                    eid,
                    int(t),
                    W_T_EE[:3, :].reshape(-1).tolist(),
                    W_T_Cg[:3, :].reshape(-1).tolist(),
                )
            )
            total_ok += 1
            if len(rows) >= shard_size:
                shard_idx = write_shard(shard_idx, rows)
        shard_idx = write_shard(shard_idx, rows)

    return {
        "episodes_processed": int(total_ok),
        "bad_or_skipped": int(total_bad),
        "shards": int(shard_idx),
        "out_dir": str(out_dir),
    }


# ------------------------- main orchestration -------------------------

def parse_int_list(x: Optional[str]) -> Optional[List[int]]:
    if x is None or str(x).strip() == "":
        return None
    out = []
    for part in str(x).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out



def load_json_if_exists(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild CALVIN cameras.json into ./test/calib without touching the original.")
    ap.add_argument("--zip", required=True, type=Path, help="Path to dataset/task_ABCD_D.zip")
    ap.add_argument("--repo-root", type=Path, default=Path("."), help="CALVIN repo root containing calvin_env/")
    ap.add_argument("--out-root", type=Path, default=Path("./test"), help="Safe output root; default ./test")
    ap.add_argument("--reference-cameras", type=Path, default=None, help="Historical cameras.json for exact regression if available")
    ap.add_argument("--no-reference-auto", action="store_true", help="Do not auto-pick zip-without-suffix/calib/cameras.json when --reference-cameras is omitted")
    ap.add_argument("--urdf", type=Path, default=None, help="Override URDF path; otherwise resolved from hydra + repo-root")
    ap.add_argument("--hydra-split", default="validation", choices=["validation", "training"], help="Preferred split for reading Hydra")
    ap.add_argument("--candidate-split", default="validation", choices=["validation", "training"], help="Split used to rank URDF raw candidates")
    ap.add_argument("--candidate-episodes", type=int, default=12, help="Episodes per raw candidate for ranking")
    ap.add_argument("--verify-episodes", type=int, default=40, help="Episodes for final regression verify (val/train)")
    ap.add_argument("--ref-link-cands", type=str, default=None, help="Comma-separated reference link ids; default uses Hydra tcp_link_id,end_effector_link_id")
    ap.add_argument("--cam-link-cands", type=str, default=None, help="Comma-separated camera link ids; default uses Hydra gripper_cam_link and +2")
    ap.add_argument("--q-start", type=int, default=7, help="robot_obs arm-joint start index; set -1 to auto/disable")
    ap.add_argument("--rpy-order", default="zyx", choices=["xyz", "zyx"], help="Fallback robot_obs Euler order")
    ap.add_argument("--t-mode", default="auto", choices=["auto", "start", "mid", "end"], help="Frame selection for depth verification")
    ap.add_argument("--edge-thr", type=float, default=0.03, help="Static-depth edge rejection threshold")
    ap.add_argument("--max-pix", type=int, default=800, help="Max gripper pixels sampled per frame")
    ap.add_argument("--bilinear", action="store_true", help="Use bilinear static-depth sampling")
    ap.add_argument("--export-parquet", action="store_true", help="Also export gripper_poses-*.parquet into out-root/calib")
    ap.add_argument("--export-episodes-max", type=int, default=None, help="Optional cap for parquet export smoke run")
    ap.add_argument("--shard-size", type=int, default=200_000, help="Parquet shard size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    zip_path = args.zip.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    calib_dir = out_root / "calib"
    variants_dir = calib_dir / "variants"
    ensure_dir(variants_dir)

    cfg, hydra_path = load_hydra_from_zip(zip_path, preferred_split=args.hydra_split)
    static = build_static_from_hydra(cfg, hydra_path)
    gripper_intr = build_gripper_intr_from_hydra(cfg)

    robot = cfg.get("robot") or {}
    scene = cfg.get("scene") or {}
    tcp_link_id = int(robot.get("tcp_link_id", 15))
    ee_link_id = int(robot.get("end_effector_link_id", 7))
    gripper_cam_link = int(robot.get("gripper_cam_link", 12))
    arm_joint_ids = [int(v) for v in robot.get("arm_joint_ids", list(range(7)))]
    base_pos = [float(v) for v in scene.get("robot_base_position", [0, 0, 0])]
    base_quat_xyzw = quat_xyzw_from_cfg(scene.get("robot_base_orientation", [0, 0, 0, 1]))

    urdf_path = resolve_urdf_path(cfg, repo_root, args.urdf)

    reference_path = args.reference_cameras
    if reference_path is None and not args.no_reference_auto:
        reference_path = auto_reference_path(zip_path)
    reference_path = reference_path.expanduser().resolve() if reference_path is not None else None
    reference_json = load_json_if_exists(reference_path)

    # Build a pybullet body once for link discovery and URDF raw candidates.
    p, cid, body = build_pb_body(urdf_path, base_pos, base_quat_xyzw)
    try:
        link_table = get_link_table(p, body)
        save_json({"links": link_table}, calib_dir / "link_table.json")

        ref_cands = parse_int_list(args.ref_link_cands)
        if not ref_cands:
            ref_cands = []
            for x in [tcp_link_id, ee_link_id]:
                if x not in ref_cands:
                    ref_cands.append(int(x))
        cam_cands = parse_int_list(args.cam_link_cands)
        if not cam_cands:
            cam_cands = []
            for x in [gripper_cam_link, gripper_cam_link + 2]:
                if x not in cam_cands:
                    cam_cands.append(int(x))

        link_ids_available = {row["joint_index"] for row in link_table}
        candidates: List[Candidate] = []
        for ref_id in ref_cands:
            if ref_id not in link_ids_available:
                continue
            label = ref_label(ref_id, tcp_link_id=tcp_link_id, ee_link_id=ee_link_id)
            for cam_id in cam_cands:
                if cam_id not in link_ids_available:
                    continue
                E_raw = compute_E_raw_from_urdf(p, body, ref_id, cam_id)
                file_name = f"cameras_{label}2cam{cam_id}.json"
                meta = build_candidate_meta(
                    urdf_path=urdf_path,
                    ref_link_id=ref_id,
                    cam_link_id=cam_id,
                    rpy_order=args.rpy_order,
                    variant_type="urdf_raw_candidate",
                    selection_note="Raw URDF hand-eye candidate reconstructed from pybullet link states.",
                    tcp_link_id=tcp_link_id,
                )
                cand_json = make_cameras_json(
                    static,
                    gripper_intr,
                    E_raw,
                    meta,
                    ref_link_id=ref_id,
                    cam_link_id=cam_id,
                    tcp_link_id=tcp_link_id,
                    ee_link_id=ee_link_id,
                    urdf_path=urdf_path,
                    reference_json=None,
                    preserve_reference_legacy=False,
                )
                save_json(cand_json, variants_dir / file_name)
                candidates.append(
                    Candidate(
                        ref_link_id=ref_id,
                        cam_link_id=cam_id,
                        ref_label=label,
                        E_raw=E_raw,
                        file_name=file_name,
                    )
                )
    finally:
        pb_disconnect(p, cid)

    if not candidates:
        raise RuntimeError("No valid URDF candidate link pairs were found. Check --ref-link-cands / --cam-link-cands and link_table.json.")

    # Use first episodes to detect pose/joint layout.
    with zipfile.ZipFile(zip_path, "r") as zf:
        names_for_detect = list_episode_paths(zf, split=args.candidate_split)
        posrpy_idx = autodetect_posrpy_index(zf, names_for_detect, probe=50)
        q_start = None if int(args.q_start) < 0 else int(args.q_start)
        if q_start is None and names_for_detect:
            try:
                d0 = np.load(io.BytesIO(zf.read(names_for_detect[0])), allow_pickle=True)
                if "robot_obs" in d0.files:
                    q_start = guess_q_start(norm_robot_obs(d0["robot_obs"]))
            except Exception:
                q_start = None

    report: dict = {
        "zip": str(zip_path),
        "repo_root": str(repo_root),
        "hydra_path": hydra_path,
        "urdf": str(urdf_path),
        "reference_cameras": str(reference_path) if reference_path is not None else None,
        "posrpy_index": posrpy_idx,
        "q_start": q_start,
        "candidates": [],
    }

    # Rank raw URDF candidates on candidate_split.
    for cand in candidates:
        cand.depth_val = verify_depth_for_E(
            zip_path=zip_path,
            cfg=cfg,
            static=static,
            gripper=gripper_intr,
            E_T_C=cand.E_raw,
            ref_link_id=cand.ref_link_id,
            urdf_path=urdf_path,
            split=args.candidate_split,
            episodes=args.candidate_episodes,
            q_start=q_start,
            posrpy_start=posrpy_idx,
            arm_joint_ids=arm_joint_ids,
            rpy_order=args.rpy_order,
            t_mode=args.t_mode,
            max_pix=args.max_pix,
            edge_thr=args.edge_thr,
            bilinear=args.bilinear,
            seed=args.seed,
        )
        report["candidates"].append(
            {
                "ref_link_id": cand.ref_link_id,
                "cam_link_id": cand.cam_link_id,
                "ref_label": cand.ref_label,
                "file_name": cand.file_name,
                "depth_candidate_split": cand.depth_val,
            }
        )

    def cand_sort_key(c: Candidate):
        m = c.depth_val or {}
        if not m.get("ok"):
            return (1, float("inf"), float("inf"), 0)
        return (0, float(m["median"]), float(m["p90"]), -int(m["pixels"]))

    best_raw = sorted(candidates, key=cand_sort_key)[0]

    # Build final cameras.json.
    if reference_json is not None:
        ref_meta = reference_json.get("meta", {})
        chosen_ref = int(ref_meta.get("ee_link_id", best_raw.ref_link_id))
        chosen_cam = int(ref_meta.get("urdf_cam_link_id", ref_meta.get("gripper_cam_link_id", best_raw.cam_link_id)))
        chosen_match = None
        for cand in candidates:
            if cand.ref_link_id == chosen_ref and cand.cam_link_id == chosen_cam:
                chosen_match = cand
                break
        if chosen_match is None:
            # Build it directly if the reference selected a pair outside the default candidate pool.
            p, cid, body = build_pb_body(urdf_path, base_pos, base_quat_xyzw)
            try:
                E_raw_chosen = compute_E_raw_from_urdf(p, body, chosen_ref, chosen_cam)
            finally:
                pb_disconnect(p, cid)
            chosen_match = Candidate(
                ref_link_id=chosen_ref,
                cam_link_id=chosen_cam,
                ref_label=ref_label(chosen_ref, tcp_link_id=tcp_link_id, ee_link_id=ee_link_id),
                E_raw=E_raw_chosen,
                file_name=f"cameras_{ref_label(chosen_ref, tcp_link_id, ee_link_id)}2cam{chosen_cam}.json",
            )
        E_ref = np.array(reference_json["gripper"]["E_T_C"], dtype=float)
        Delta = inv_T(chosen_match.E_raw) @ E_ref
        final_meta = dict(ref_meta)
        final_meta.update(
            {
                "coord_convention": final_meta.get("coord_convention", "opencv"),
                "depth_unit": final_meta.get("depth_unit", "meter"),
                "rpy_order": final_meta.get("rpy_order", args.rpy_order),
                "urdf": str(urdf_path),
                "ee_link_id": int(chosen_ref),
                "urdf_cam_link_id": int(chosen_cam),
                "gripper_cam_link_id": int(chosen_cam),
                "etC_source": final_meta.get("etC_source", "reference_regression"),
                "urdf_to_dataset_delta_4x4": tolist_f32(Delta),
                "rebuilt_under": str(calib_dir),
                "rebuild_mode": "reference_regression",
                "hydra_path": hydra_path,
            }
        )
        final_json = make_cameras_json(
            static,
            gripper_intr,
            E_ref,
            final_meta,
            ref_link_id=chosen_ref,
            cam_link_id=chosen_cam,
            tcp_link_id=tcp_link_id,
            ee_link_id=ee_link_id,
            urdf_path=urdf_path,
            reference_json=reference_json,
            preserve_reference_legacy=True,
        )
        chosen_for_final = chosen_match
        report["selection_mode"] = "reference_regression"
    else:
        chosen_for_final = best_raw
        E_ref = chosen_for_final.E_raw.copy()
        Delta = np.eye(4, dtype=float)
        final_meta = {
            "axis": "opencv",
            "coord_convention": "opencv",
            "depth_unit": "meter",
            "rpy_order": args.rpy_order,
            "urdf": str(urdf_path),
            "ee_link_id": int(chosen_for_final.ref_link_id),
            "urdf_cam_link_id": int(chosen_for_final.cam_link_id),
            "gripper_cam_link_id": int(chosen_for_final.cam_link_id),
            "tcp_link_id": int(tcp_link_id),
            "base_used": False,
            "etC_source": "best_effort_urdf_only",
            "urdf_to_dataset_delta_4x4": tolist_f32(Delta),
            "rebuilt_under": str(calib_dir),
            "rebuild_mode": "best_effort_without_reference",
            "selection_note": "No historical cameras.json was found; E_ref is set to the best URDF raw candidate.",
            "hydra_path": hydra_path,
        }
        final_json = make_cameras_json(
            static,
            gripper_intr,
            E_ref,
            final_meta,
            ref_link_id=chosen_for_final.ref_link_id,
            cam_link_id=chosen_for_final.cam_link_id,
            tcp_link_id=tcp_link_id,
            ee_link_id=ee_link_id,
            urdf_path=urdf_path,
            reference_json=None,
            preserve_reference_legacy=False,
        )
        report["selection_mode"] = "best_effort_without_reference"

    # Write chosen URDF-equivalent variants.
    urdf_equiv_json = make_cameras_json(
        static,
        gripper_intr,
        chosen_for_final.E_raw,
        build_candidate_meta(
            urdf_path=urdf_path,
            ref_link_id=chosen_for_final.ref_link_id,
            cam_link_id=chosen_for_final.cam_link_id,
            rpy_order=args.rpy_order,
            variant_type="urdf_equiv",
            selection_note="Chosen URDF raw hand-eye for deploy-side equivalence checks.",
            tcp_link_id=tcp_link_id,
        ),
        ref_link_id=chosen_for_final.ref_link_id,
        cam_link_id=chosen_for_final.cam_link_id,
        tcp_link_id=tcp_link_id,
        ee_link_id=ee_link_id,
        urdf_path=urdf_path,
        reference_json=None,
        preserve_reference_legacy=False,
    )
    save_json(urdf_equiv_json, variants_dir / f"cameras_urdf_cam{chosen_for_final.cam_link_id}_equiv.json")
    save_json(urdf_equiv_json, variants_dir / "cameras_urdf_equiv.json")

    final_path = calib_dir / "cameras.json"
    save_json(final_json, final_path)
    save_json(final_json, calib_dir / "cameras.backup.json")

    # Regression compares against the historical file when available.
    if reference_json is not None:
        report["reference_compare"] = {
            "static_W_T_C_fro": fro_err(static.W_T_C, np.array(reference_json["static"]["W_T_C"], float)),
            "static_K_fro": fro_err(static.K, np.array(reference_json["static"]["K"], float)),
            "gripper_K_fro": fro_err(gripper_intr.K, np.array(reference_json["gripper"]["K"], float)),
            "E_ref_fro": fro_err(E_ref, np.array(reference_json["gripper"]["E_T_C"], float)),
            "Delta_fro": fro_err(Delta, np.array(reference_json.get("meta", {}).get("urdf_to_dataset_delta_4x4", np.eye(4)), float)),
        }

    # Final depth checks.
    chosen_ref = int(final_json["meta"]["ee_link_id"])
    final_val = verify_depth_for_E(
        zip_path=zip_path,
        cfg=cfg,
        static=static,
        gripper=gripper_intr,
        E_T_C=E_ref,
        ref_link_id=chosen_ref,
        urdf_path=urdf_path,
        split="validation",
        episodes=args.verify_episodes,
        q_start=q_start,
        posrpy_start=posrpy_idx,
        arm_joint_ids=arm_joint_ids,
        rpy_order=args.rpy_order,
        t_mode=args.t_mode,
        max_pix=args.max_pix,
        edge_thr=args.edge_thr,
        bilinear=args.bilinear,
        seed=args.seed,
    )
    final_train = verify_depth_for_E(
        zip_path=zip_path,
        cfg=cfg,
        static=static,
        gripper=gripper_intr,
        E_T_C=E_ref,
        ref_link_id=chosen_ref,
        urdf_path=urdf_path,
        split="training",
        episodes=args.verify_episodes,
        q_start=q_start,
        posrpy_start=posrpy_idx,
        arm_joint_ids=arm_joint_ids,
        rpy_order=args.rpy_order,
        t_mode=args.t_mode,
        max_pix=args.max_pix,
        edge_thr=args.edge_thr,
        bilinear=args.bilinear,
        seed=args.seed,
    )
    urdf_equiv_val = verify_depth_for_E(
        zip_path=zip_path,
        cfg=cfg,
        static=static,
        gripper=gripper_intr,
        E_T_C=chosen_for_final.E_raw,
        ref_link_id=chosen_for_final.ref_link_id,
        urdf_path=urdf_path,
        split="validation",
        episodes=max(8, min(args.verify_episodes, 20)),
        q_start=q_start,
        posrpy_start=posrpy_idx,
        arm_joint_ids=arm_joint_ids,
        rpy_order=args.rpy_order,
        t_mode=args.t_mode,
        max_pix=args.max_pix,
        edge_thr=args.edge_thr,
        bilinear=args.bilinear,
        seed=args.seed,
    )

    equiv = urdf_q_invariance_and_delta_equiv(
        urdf_path=urdf_path,
        base_pos=base_pos,
        base_quat_xyzw=base_quat_xyzw,
        ee_link_id=chosen_for_final.ref_link_id,
        cam_link_id=chosen_for_final.cam_link_id,
        Delta=Delta,
        E_ref=E_ref,
        arm_joint_ids=arm_joint_ids,
        trials=40,
        seed=args.seed,
    )

    report["chosen_final"] = {
        "ref_link_id": chosen_for_final.ref_link_id,
        "cam_link_id": chosen_for_final.cam_link_id,
        "ref_label": chosen_for_final.ref_label,
        "candidate_file": chosen_for_final.file_name,
    }
    report["final_depth_validation"] = final_val
    report["final_depth_training"] = final_train
    report["urdf_equiv_depth_validation"] = urdf_equiv_val
    report["train_deploy_equivalence"] = equiv

    if args.export_parquet:
        export_report = export_parquet_from_zip(
            zip_path=zip_path,
            out_dir=calib_dir,
            E_T_C=E_ref,
            rpy_order=args.rpy_order,
            t_mode="mid" if args.t_mode == "auto" else args.t_mode,
            episodes_max=args.export_episodes_max,
            shard_size=args.shard_size,
        )
        report["parquet_export"] = export_report

    save_json(report, calib_dir / "rebuild_report.json")

    # Console summary.
    print("[OK] wrote:")
    print(f"  final      : {final_path}")
    print(f"  report     : {calib_dir / 'rebuild_report.json'}")
    print(f"  variants   : {variants_dir}")
    print(f"  link table : {calib_dir / 'link_table.json'}")
    print()
    print("[SELECTED]")
    print(f"  ref_link_id={chosen_for_final.ref_link_id}  cam_link_id={chosen_for_final.cam_link_id}  mode={report['selection_mode']}")
    if final_val.get("ok"):
        print(
            "[VAL  ] mean={mean:.4f}  median={median:.4f}  p90={p90:.4f}  <5cm={lt:.2%}".format(
                mean=final_val["mean"], median=final_val["median"], p90=final_val["p90"], lt=final_val["lt_5cm"]
            )
        )
    else:
        print(f"[VAL  ] {final_val.get('message')}")
    if final_train.get("ok"):
        print(
            "[TRAIN] mean={mean:.4f}  median={median:.4f}  p90={p90:.4f}  <5cm={lt:.2%}".format(
                mean=final_train["mean"], median=final_train["median"], p90=final_train["p90"], lt=final_train["lt_5cm"]
            )
        )
    else:
        print(f"[TRAIN] {final_train.get('message')}")
    print(
        "[EQV  ] raw_fro={raw:.3e}  ref_fro={ref:.3e}".format(
            raw=equiv["max_fro_raw"], ref=equiv["max_fro_ref"]
        )
    )
    if reference_json is not None:
        cmp = report["reference_compare"]
        print(
            "[DIFF ] static_W_T_C={sw:.3e}  static_K={sk:.3e}  gripper_K={gk:.3e}  E_ref={er:.3e}  Delta={de:.3e}".format(
                sw=cmp["static_W_T_C_fro"],
                sk=cmp["static_K_fro"],
                gk=cmp["gripper_K_fro"],
                er=cmp["E_ref_fro"],
                de=cmp["Delta_fro"],
            )
        )


if __name__ == "__main__":
    main()
