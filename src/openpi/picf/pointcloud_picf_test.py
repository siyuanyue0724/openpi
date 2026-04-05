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
