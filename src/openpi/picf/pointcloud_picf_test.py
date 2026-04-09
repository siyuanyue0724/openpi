from pathlib import Path

import numpy as np

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.test_utils import build_mini_calvin_dataset


def test_picf_pointcloud_builds_normals(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=2, max_points=128)
    rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(32, dtype=np.float32), np.arange(32, dtype=np.float32), indexing="ij")
    depth = 0.6 + 0.001 * xx + 0.0015 * yy
    point_set = builder({"rgb_static": rgb, "depth_static": depth.astype(np.float32)})

    assert point_set.frame_valid
    assert point_set.xyz_world.shape[0] <= 128
    assert point_set.grid_coord.dtype == np.int32
    assert np.all(point_set.grid_coord >= 0)
    norms = np.linalg.norm(point_set.normal_world, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_picf_pointcloud_handles_invalid_depth(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=2, max_points=64)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    depth = np.zeros((16, 16), dtype=np.float32)
    point_set = builder({"rgb_static": rgb, "depth_static": depth})

    assert not point_set.frame_valid
    assert point_set.xyz_world.shape == (0, 3)


def test_picf_pointcloud_focus_selection_prioritizes_local_points(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=False)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=1, max_points=40, min_peripheral_points=8)
    xyz = np.concatenate(
        [
            np.stack([np.linspace(-0.02, 0.02, 100, dtype=np.float32), np.zeros((100,), dtype=np.float32), np.ones((100,), dtype=np.float32)], axis=-1),
            np.stack([np.linspace(0.2, 0.6, 100, dtype=np.float32), np.zeros((100,), dtype=np.float32), np.ones((100,), dtype=np.float32)], axis=-1),
        ],
        axis=0,
    )
    focus_mask = np.linalg.norm(xyz[:, :2], axis=1) <= 0.08
    focus_weights = 1.0 + builder.focus_boost * np.exp(-(np.linalg.norm(xyz[:, :2], axis=1) ** 2) / (2.0 * 0.08 * 0.08))

    baseline = builder._select_indices(xyz)  # noqa: SLF001
    chosen = builder._select_indices(xyz, focus_mask=focus_mask, focus_weights=focus_weights)  # noqa: SLF001

    assert chosen.shape == (40,)
    assert int(focus_mask[chosen].sum()) > int(focus_mask[baseline].sum())
    assert int((~focus_mask[chosen]).sum()) > 0
