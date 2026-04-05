from __future__ import annotations

import math

import numpy as np


def normalize_vectors(vectors: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def rpy_zyx_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in np.asarray(rpy, dtype=np.float32).reshape(3)]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1, 3)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points, ones], axis=1)
    return (np.asarray(transform, dtype=np.float32) @ homo.T).T[:, :3]


def transform_normals(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    if normals.size == 0:
        return normals.reshape(-1, 3)
    rotation = np.asarray(transform, dtype=np.float32)[:3, :3]
    return normalize_vectors((rotation @ normals.T).T)


def relative_transform(from_transform: np.ndarray, to_transform: np.ndarray) -> np.ndarray:
    return invert_transform(to_transform) @ np.asarray(from_transform, dtype=np.float32)
