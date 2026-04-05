from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class PicfPointCloudFrame:
    """Variable-length PICF point set for scaffold replay."""

    grid_coord: np.ndarray
    xyz_world: np.ndarray
    rgb: np.ndarray
    normal_world: np.ndarray
    valid_point_mask: np.ndarray
    frame_valid: bool

    def __post_init__(self) -> None:
        self.grid_coord = np.asarray(self.grid_coord, dtype=np.int32)
        self.xyz_world = np.asarray(self.xyz_world, dtype=np.float32)
        self.rgb = np.asarray(self.rgb, dtype=np.float32)
        self.normal_world = np.asarray(self.normal_world, dtype=np.float32)
        self.valid_point_mask = np.asarray(self.valid_point_mask, dtype=bool)
        self.frame_valid = bool(self.frame_valid)
        n = self.xyz_world.shape[0]
        expected_shape = (n, 3)
        for name, value in (
            ("grid_coord", self.grid_coord),
            ("xyz_world", self.xyz_world),
            ("rgb", self.rgb),
            ("normal_world", self.normal_world),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
        if self.valid_point_mask.shape != (n,):
            raise ValueError(f"valid_point_mask must have shape {(n,)}, got {self.valid_point_mask.shape}")

    @property
    def num_points(self) -> int:
        return int(self.xyz_world.shape[0])


@dataclasses.dataclass
class RuntimeMeta:
    t_v_last: float = float("-inf")
    t_p_last: float = float("-inf")
    t_t_last: float = float("-inf")
    t_rgb_last: float = float("-inf")
    b_rgb_avail: bool = False
    rgb_proj_residual: float = float("inf")
    n_vis_upd: int = 0
    v_rgb_p: bool = False
    v_pc_scaf: bool = False
    stale_scaffold_steps: int = 0


@dataclasses.dataclass
class PicfObservation:
    rgb_static: np.ndarray
    depth_static: np.ndarray
    robot_obs: np.ndarray
    prompt: str
    step_id: int
    segment_id: int
    timestamp_s: float
    reset_scaffold: bool
    rgb_gripper: np.ndarray | None = None
    point_set: PicfPointCloudFrame | None = None
    runtime_meta: RuntimeMeta | None = None
    G_t: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.rgb_static = np.asarray(self.rgb_static)
        self.depth_static = np.asarray(self.depth_static, dtype=np.float32)
        self.robot_obs = np.asarray(self.robot_obs, dtype=np.float32).reshape(-1)
        if self.rgb_gripper is not None:
            self.rgb_gripper = np.asarray(self.rgb_gripper)
        if self.G_t is not None:
            self.G_t = np.asarray(self.G_t, dtype=np.float32)


@dataclasses.dataclass
class ScaffoldDebugMetrics:
    num_points_local: int
    num_active: int
    num_birth: int
    match_ratio: float
    mean_radius: float
    normal_fallback_ratio: float
    empty_support_ratio: float
    hold_triggered: bool
    hold_reason: str | None
    reindex_failure_rate: float
    normal_flip_ratio: float
    fresh_scaffold: bool


@dataclasses.dataclass
class SupportScaffoldState:
    pi_geom: np.ndarray
    x: np.ndarray
    n: np.ndarray
    r: np.ndarray
    omega: np.ndarray
    active_mask: np.ndarray
    pred_idx: np.ndarray
    matched_mask: np.ndarray
    birth_mask: np.ndarray
    e_id: np.ndarray
    s_qry: np.ndarray
    G_t: np.ndarray
    step_id: int
    segment_id: int
    runtime_meta: RuntimeMeta
    debug: ScaffoldDebugMetrics

    def __post_init__(self) -> None:
        self.pi_geom = np.asarray(self.pi_geom, dtype=np.float32)
        self.x = np.asarray(self.x, dtype=np.float32)
        self.n = np.asarray(self.n, dtype=np.float32)
        self.r = np.asarray(self.r, dtype=np.float32)
        self.omega = np.asarray(self.omega, dtype=np.float32)
        self.active_mask = np.asarray(self.active_mask, dtype=bool)
        self.pred_idx = np.asarray(self.pred_idx, dtype=np.int32)
        self.matched_mask = np.asarray(self.matched_mask, dtype=bool)
        self.birth_mask = np.asarray(self.birth_mask, dtype=bool)
        self.e_id = np.asarray(self.e_id, dtype=np.float32)
        self.s_qry = np.asarray(self.s_qry, dtype=np.float32)
        self.G_t = np.asarray(self.G_t, dtype=np.float32)
