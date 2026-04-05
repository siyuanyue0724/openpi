from __future__ import annotations

import numpy as np


def coverage_weights(points: np.ndarray, centers: np.ndarray, *, sigma: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if centers.shape[0] == 0:
        return np.ones((points.shape[0],), dtype=np.float32)
    dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    coverage = np.exp(-(dists**2) / max(2.0 * sigma * sigma, 1e-8)).max(axis=1)
    return 1.0 - coverage.astype(np.float32)


def weighted_fps(points: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if count <= 0 or points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if points.shape[0] <= count:
        return np.arange(points.shape[0], dtype=np.int64)
    weights = np.maximum(weights, 0.0)
    chosen: list[int] = []
    start = int(np.argmax(weights))
    chosen.append(start)
    min_dist = np.linalg.norm(points - points[start : start + 1], axis=1)
    while len(chosen) < count:
        score = min_dist * (0.5 + weights)
        score[np.asarray(chosen, dtype=np.int64)] = -1.0
        next_idx = int(np.argmax(score))
        if score[next_idx] < 0:
            break
        chosen.append(next_idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[next_idx : next_idx + 1], axis=1))
    return np.asarray(chosen, dtype=np.int64)
