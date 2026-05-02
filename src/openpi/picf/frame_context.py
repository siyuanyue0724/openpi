from __future__ import annotations

import dataclasses

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.geometry import invert_transform
from openpi.picf.geometry import transform_normals
from openpi.picf.geometry import transform_points
from openpi.picf.scaffold.local_frame import EndEffectorLocalFrame


@dataclasses.dataclass
class PointFrameContext:
    grid_coord: np.ndarray
    points_local: np.ndarray
    normals_local: np.ndarray
    colors: np.ndarray
    local_mask: np.ndarray
    world_to_local: np.ndarray
    G_t: np.ndarray
    # 0 = effector/contact-local point, 1 = global scene/object point.
    # Older callers may leave this unset; core code treats unset as all-local.
    pool_ids: np.ndarray | None = None


def build_point_frame_context(
    observation: PicfObservation,
    *,
    crop_radius_m: float,
    local_frame: EndEffectorLocalFrame | None = None,
) -> PointFrameContext:
    if observation.point_set is None:
        raise ValueError("PointFrameContext requires observation.point_set to be populated.")
    frame_builder = local_frame or EndEffectorLocalFrame()
    if observation.G_t is None:
        observation.G_t = frame_builder.make_transform(observation.robot_obs)
    world_to_local = invert_transform(observation.G_t)
    points_local = transform_points(observation.point_set.xyz_world, world_to_local)
    normals_local = transform_normals(observation.point_set.normal_world, world_to_local)
    dists = np.linalg.norm(points_local, axis=1)
    keep = dists <= float(crop_radius_m)
    return PointFrameContext(
        grid_coord=observation.point_set.grid_coord[keep],
        points_local=points_local[keep],
        normals_local=normals_local[keep],
        colors=observation.point_set.rgb[keep],
        local_mask=keep,
        world_to_local=world_to_local,
        G_t=np.asarray(observation.G_t, dtype=np.float32),
        pool_ids=np.zeros((int(keep.sum()),), dtype=np.int64),
    )
