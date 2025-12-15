#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALVIN 风格双相机（静态 200x200 / 腕部 84x84）渲染基准 + K/E 打印 + 几何验证 + 结果落盘(JSON)

运行示例：
  CPU：
    python scripts/bench/calvin_render_bench.py --frames 2000 --egl 0 \
      --estimate_frames 3000000 --save_dir outputs/bench

  GPU（严格要求 EGL 成功，否则退出）：
    python scripts/bench/calvin_render_bench.py --frames 2000 --egl 1 --egl-strict \
      --estimate_frames 3000000 --save_dir outputs/bench
"""

import os
import time
import math
import json
import argparse
import itertools
from datetime import datetime

import numpy as np
try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    raise SystemExit("请先安装依赖： pip install 'pybullet>=3.2.7' 'numpy>=1.24,<2.0'") from e


# ---------- 基础几何 ----------

def k_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """由垂直 FOV 计算针孔内参 K（主点在中心，skew=0）"""
    f = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def E_from_view_matrix(view_m) -> np.ndarray:
    """PyBullet viewMatrix(OpenGL, 行优先) -> 3x4 外参 [R|t]（世界→相机）"""
    M = np.array(view_m, dtype=np.float64).reshape(4, 4).T  # 转为列向量右乘约定
    R, t = M[:3, :3], M[:3, 3:4]
    return np.hstack([R, t])  # (3,4)

def project_points(K: np.ndarray, E: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """世界坐标 Xw(N,3) → 像素 u,v"""
    Xw = np.asarray(Xw, dtype=np.float64)
    if Xw.ndim == 1:
        Xw = Xw[None, :]
    Xc = (E[:, :3] @ Xw.T + E[:, 3:4]).T  # (N,3)
    z = Xc[:, 2:3]
    z[z == 0.0] = 1e-9  # 避免除零
    u = (K[0, 0] * Xc[:, 0] / z[:, 0]) + K[0, 2]
    v = (K[1, 1] * Xc[:, 1] / z[:, 0]) + K[1, 2]
    return np.stack([u, v], axis=1)

def seg_to_mask(seg: np.ndarray, obj_id: int) -> np.ndarray:
    """兼容两种编码：直接等于 obj_id，或低 24 位为 obj_id。"""
    seg = np.asarray(seg, dtype=np.int64)
    m = (seg == obj_id)
    if not m.any():
        m = ((seg & 0xFFFFFF) == obj_id)
    return m

def quat_to_R(q):
    m = p.getMatrixFromQuaternion(q)  # 9 floats, row-major
    return np.array(m, dtype=np.float64).reshape(3, 3)

def cube_corners_world(center, quat, half=0.05) -> np.ndarray:
    R = quat_to_R(quat)
    offs = np.array(list(itertools.product([-half, half], repeat=3)), dtype=np.float64)  # (8,3)
    return (R @ offs.T).T + np.asarray(center, dtype=np.float64)

def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 1:
        return np.zeros((0, 2), dtype=np.float64)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x,y
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower = []
    for pnt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pnt) <= 0:
            lower.pop()
        lower.append(tuple(pnt))
    upper = []
    for pnt in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pnt) <= 0:
            upper.pop()
        upper.append(tuple(pnt))
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)

def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    if poly.ndim != 2 or len(poly) == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    if len(poly) < 3:
        return poly.mean(axis=0)
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross
    A *= 0.5
    if abs(A) < 1e-6:
        return poly.mean(axis=0)
    return np.array([Cx / (6 * A), Cy / (6 * A)], dtype=np.float64)

def mask_hull_centroid(mask: np.ndarray) -> np.ndarray:
    """把分割 mask 的像素坐标做凸包，再求凸包质心（更接近“投影凸包质心”）。"""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    hull = convex_hull_2d(pts)
    if hull.shape[0] == 0:
        return None
    return polygon_centroid(hull)


# ---------- 主流程 ----------

def run_bench(frames=2000, egl=True, egl_strict=False, estimate_frames=None, save_dir=None,
              static_fov=60.0, wrist_fov=60.0, near=0.01, far=2.5,
              validate_every=10, wrist_pass_px=5.0, static_pass_px=2.0,
              res_static=(200, 200), res_wrist=(84, 84), no_validate=False):

    # 连接物理引擎（离屏）
    p.connect(p.DIRECT)

    # --- EGL 加载（鲁棒顺序：pkgutil 路径 -> getPluginFilename -> 裸名字） ---
    renderer_flag = p.ER_TINY_RENDERER
    renderer_name = "CPU/Tiny"
    plugin_id = -1
    plugin_path = None
    if egl:
        import pkgutil
        try_paths = []
        loader = pkgutil.get_loader('eglRenderer')
        if loader is not None:
            try_paths.append(('so_with_init', loader.get_filename()))
        if hasattr(p, "getPluginFilename"):
            try:
                try_paths.append(('plain', p.getPluginFilename("eglRendererPlugin")))
            except Exception:
                pass
        try_paths.append(('plain', "eglRendererPlugin"))

        last_err = None
        for kind, path in try_paths:
            try:
                if kind == 'so_with_init':
                    plugin_id = p.loadPlugin(path, "_eglRendererPlugin")
                else:
                    plugin_id = p.loadPlugin(path)
                plugin_path = path
                if plugin_id >= 0:
                    renderer_flag = p.ER_BULLET_HARDWARE_OPENGL
                    renderer_name = f"EGL/GPU ({path})"
                    print(f"[INFO] EGL/GPU 渲染已启用: {path} (plugin_id={plugin_id})")
                    break
            except Exception as e:
                last_err = e

        if plugin_id < 0:
            msg = f"[WARN] EGL 加载失败。尝试路径={try_paths}，最后错误={last_err}"
            if egl_strict:
                raise SystemExit(msg)
            print(msg)

    # 基础场景
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    box_vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05],
                                 rgbaColor=[0.8, 0.2, 0.2, 1])
    box_cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
    box_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=box_cs,
                               baseVisualShapeIndex=box_vs,
                               basePosition=[0.6, 0.0, 0.05])

    Ws, Hs = res_static
    Ww, Hw = res_wrist
    K_static = k_from_fov(Ws, Hs, static_fov)
    K_wrist = k_from_fov(Ww, Hw, wrist_fov)

    print("\n[INFO] 静态相机 K=\n", K_static)
    print("[INFO] 腕部相机 K=\n", K_wrist)

    proj_static = p.computeProjectionMatrixFOV(fov=static_fov, aspect=Ws / float(Hs),
                                               nearVal=near, farVal=far)
    proj_wrist = p.computeProjectionMatrixFOV(fov=wrist_fov, aspect=Ww / float(Hw),
                                              nearVal=near, farVal=far)

    cam_up = [0.0, 0.0, 1.0]
    cam_static_eye = [0.9, 0.0, 0.7]
    cam_static_tgt = [0.6, 0.0, 0.05]
    view_static = p.computeViewMatrix(cam_static_eye, cam_static_tgt, cam_up)
    E_static = E_from_view_matrix(view_static)

    print("\n[INFO] 静态相机示例 E(世界->相机)=\n",
          np.array_str(E_static, precision=4, suppress_small=True))

    def wrist_view(fidx: int):
        radius = 0.18
        theta = 0.01 * fidx
        eye = [0.6 + radius * math.cos(theta),
               0.0 + radius * math.sin(theta),
               0.10]
        tgt = p.getBasePositionAndOrientation(box_id)[0]
        return p.computeViewMatrix(eye, tgt, cam_up)

    # 预热
    for _ in range(30):
        p.getCameraImage(Ww, Hw, wrist_view(_), proj_wrist, renderer=renderer_flag)
        p.getCameraImage(Ws, Hs, view_static,  proj_static, renderer=renderer_flag)

    # 主循环
    t0 = time.perf_counter()
    frames_done = 0
    errs_static, errs_wrist = [], []

    for f in range(frames):
        # 让立方体轻微运动
        x = 0.6 + 0.02 * math.sin(0.02 * f)
        p.resetBasePositionAndOrientation(box_id, [x, 0.0, 0.05], [0, 0, 0, 1])

        vm_w = wrist_view(f)

        # 腕部
        w, h, _, _, seg_w = p.getCameraImage(
            Ww, Hw, vm_w, proj_wrist, renderer=renderer_flag,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )
        # 静态
        w2, h2, _, _, seg_s = p.getCameraImage(
            Ws, Hs, view_static, proj_static, renderer=renderer_flag,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )

        frames_done += 1

        # 验证：每隔 validate_every 帧
        if not no_validate and validate_every > 0 and (f % validate_every) == 0:
            pos, quat = p.getBasePositionAndOrientation(box_id)
            corners_w = cube_corners_world(pos, quat, half=0.05)
            E_wrist = E_from_view_matrix(vm_w)

            # 角点投影 → 二维凸包质心（预测）
            uv_w_all = project_points(K_wrist, E_wrist,  corners_w)
            uv_s_all = project_points(K_static, E_static, corners_w)
            hull_w_pred = convex_hull_2d(uv_w_all)
            hull_s_pred = convex_hull_2d(uv_s_all)
            poly_c_w = polygon_centroid(hull_w_pred)
            poly_c_s = polygon_centroid(hull_s_pred)

            # 分割掩码的“凸包质心”（更接近投影凸包）
            seg_w_np = np.asarray(seg_w).reshape(h,  w)
            seg_s_np = np.asarray(seg_s).reshape(h2, w2)
            mw = seg_to_mask(seg_w_np, box_id)
            ms = seg_to_mask(seg_s_np, box_id)

            c_w = mask_hull_centroid(mw)
            c_s = mask_hull_centroid(ms)

            if c_w is not None and np.all(np.isfinite(poly_c_w)):
                errs_wrist.append(np.linalg.norm(poly_c_w - c_w))
            if c_s is not None and np.all(np.isfinite(poly_c_s)):
                errs_static.append(np.linalg.norm(poly_c_s - c_s))

    dt = time.perf_counter() - t0
    fps = frames_done / dt
    ips = (frames_done * 2) / dt  # 每帧两路相机

    print("\n========== 结果 ==========")
    print(f"[RESULT] 帧数(每帧含静态+腕部) : {frames_done}")
    print(f"[RESULT] 总耗时 : {dt:.2f} s")
    print(f"[RESULT] 帧率 : {fps:.2f} FPS")
    print(f"[RESULT] 图像吞吐 : {ips:.2f} images/s")

    E_w0 = E_from_view_matrix(wrist_view(0))
    print("\n[INFO] 腕部相机示例 E(世界->相机)=\n",
          np.array_str(E_w0, precision=4, suppress_small=True))

    # --- 验证输出 ---
    def summarize(name: str, arr, thr_px: float):
        if len(arr) == 0:
            print(f"[VALID] {name}: 无mask/数据，跳过。")
            return None, None, False
        arr = np.asarray(arr, dtype=np.float64)
        rms = float(np.sqrt((arr ** 2).mean()))
        mx = float(arr.max())
        ok = (rms < thr_px)
        print(f"[VALID] {name}: N={len(arr)}  RMS={rms:.2f}px  Max={mx:.2f}px  -> {'PASS' if ok else 'FAIL'}(thr={thr_px}px)")
        return rms, mx, ok

    print("\n------ 重投影验证（分割凸包质心 vs 投影凸包质心）------")
    rms_s, mx_s, ok_s = summarize("静态相机", errs_static, static_pass_px)
    rms_w, mx_w, ok_w = summarize("腕部相机", errs_wrist, wrist_pass_px)
    overall_ok = (ok_s is True) and (ok_w is True)

    if overall_ok:
        print("\n[PASS] 几何一致性通过：K/E 与渲染一致。")
    else:
        print("\n[WARN] 几何一致性偏差：请检查 FOV/K/坐标系/seg 编码，或调大 validate_every 观察。")

    # --- 估时 ---
    est_hours = None
    if estimate_frames:
        est_hours = float(estimate_frames) / fps / 3600.0
        print(f"\n[ESTIMATE] 估算 {int(estimate_frames):,} 帧耗时 ≈ {est_hours:.2f} 小时")

    # --- 落盘(JSON) ---
    out_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"calvin_render_bench_{stamp}.json")
        result = {
            "meta": {
                "timestamp_utc": stamp,
                "renderer": renderer_name,
                "pybullet_api_version": int(p.getAPIVersion()) if hasattr(p, "getAPIVersion") else None,
                "numpy_version": np.__version__,
            },
            "config": {
                "frames": int(frames),
                "validate_every": int(validate_every),
                "no_validate": bool(no_validate),
                "static_fov_deg": float(static_fov),
                "wrist_fov_deg": float(wrist_fov),
                "near": float(near),
                "far": float(far),
                "res_static": [int(Ws), int(Hs)],
                "res_wrist": [int(Ww), int(Hw)],
                "pass_thresholds_px": {"static": float(static_pass_px), "wrist": float(wrist_pass_px)},
                "estimate_frames": int(estimate_frames) if estimate_frames else None,
                "egl_requested": bool(egl),
                "egl_strict": bool(egl_strict),
                "egl_plugin_path": plugin_path,
                "egl_plugin_id": int(plugin_id),
            },
            "intrinsics": {
                "K_static": K_static.tolist(),
                "K_wrist": K_wrist.tolist(),
            },
            "extrinsics_examples": {
                "E_static": E_static.tolist(),
                "E_wrist_f0": E_w0.tolist(),
            },
            "perf": {
                "seconds": float(dt),
                "fps_frames": float(fps),
                "images_per_sec": float(ips),
            },
            "validation": {
                "static_rms_px": float(rms_s) if rms_s is not None else None,
                "static_max_px": float(mx_s) if mx_s is not None else None,
                "wrist_rms_px": float(rms_w) if rms_w is not None else None,
                "wrist_max_px": float(mx_w) if mx_w is not None else None,
                "pass": bool(overall_ok),
            },
            "estimate_hours": float(est_hours) if est_hours is not None else None,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVED] 基准结果已写入：{out_path}")

    # 清理
    try:
        if plugin_id >= 0:
            p.unloadPlugin(plugin_id)
    except Exception:
        pass
    p.disconnect()

    return out_path


def main():
    ap = argparse.ArgumentParser(description="CALVIN 风格双相机渲染基准 + K/E 打印 + 几何验证")
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--egl", type=int, default=0, help="1=尝试 EGL/GPU；0=CPU TinyRenderer")
    ap.add_argument("--egl-strict", action="store_true", help="开启后，EGL 加载失败将直接退出")
    ap.add_argument("--estimate_frames", type=float, default=None, help="可选：填你的全量帧数估算总时长")
    ap.add_argument("--save_dir", type=str, default=None, help="结果 JSON 保存目录，如 outputs/bench")
    ap.add_argument("--static_fov", type=float, default=60.0)
    ap.add_argument("--wrist_fov", type=float, default=60.0)
    ap.add_argument("--near", type=float, default=0.01)
    ap.add_argument("--far", type=float, default=2.5)
    ap.add_argument("--validate_every", type=int, default=10)
    ap.add_argument("--wrist_pass_px", type=float, default=5.0, help="腕部相机 RMS 阈值(px)")
    ap.add_argument("--static_pass_px", type=float, default=2.0, help="静态相机 RMS 阈值(px)")
    ap.add_argument("--no_validate", action="store_true", help="仅测速，不做几何验证")
    args = ap.parse_args()

    run_bench(frames=args.frames,
              egl=bool(args.egl),
              egl_strict=bool(args.egl_strict),
              estimate_frames=args.estimate_frames,
              save_dir=args.save_dir,
              static_fov=args.static_fov,
              wrist_fov=args.wrist_fov,
              near=args.near,
              far=args.far,
              validate_every=0 if args.no_validate else args.validate_every,
              wrist_pass_px=args.wrist_pass_px,
              static_pass_px=args.static_pass_px,
              no_validate=args.no_validate)

if __name__ == "__main__":
    main()
