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
    visual_available: bool = False
    tactile_available: bool = False
    point_contract_ok: bool = True
    sync_valid: bool = True


@dataclasses.dataclass
class TactileSensorFrame:
    rgb: np.ndarray
    sensor_name: str
    T_sens_to_wrist: np.ndarray
    timestamp_s: float
    depth: np.ndarray | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        self.rgb = np.asarray(self.rgb)
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.float32)
        self.T_sens_to_wrist = np.asarray(self.T_sens_to_wrist, dtype=np.float32)
        self.timestamp_s = float(self.timestamp_s)
        self.valid = bool(self.valid)
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError(f"tactile rgb must have shape [H,W,3], got {self.rgb.shape}")
        if self.T_sens_to_wrist.shape != (4, 4):
            raise ValueError(f"T_sens_to_wrist must have shape (4,4), got {self.T_sens_to_wrist.shape}")
        if self.depth is not None and self.depth.ndim not in (2, 3):
            raise ValueError(f"tactile depth must have shape [H,W] or [H,W,1], got {self.depth.shape}")


@dataclasses.dataclass
class PicfTactilePacket:
    sensors: tuple[TactileSensorFrame, ...] | list[TactileSensorFrame]
    background_rgb_by_sensor: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.sensors = tuple(self.sensors)
        if self.background_rgb_by_sensor is None:
            self.background_rgb_by_sensor = {}
        normalized: dict[str, np.ndarray] = {}
        for key, value in self.background_rgb_by_sensor.items():
            bg = np.asarray(value)
            if bg.ndim != 3 or bg.shape[-1] != 3:
                raise ValueError(f"background tactile rgb must have shape [H,W,3], got {bg.shape}")
            normalized[str(key)] = bg
        self.background_rgb_by_sensor = normalized

    def background_for(self, sensor_name: str) -> np.ndarray | None:
        return self.background_rgb_by_sensor.get(sensor_name)


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
    depth_gripper: np.ndarray | None = None
    scene_obs: np.ndarray | None = None
    point_set: PicfPointCloudFrame | None = None
    runtime_meta: RuntimeMeta | None = None
    G_t: np.ndarray | None = None
    proprio: np.ndarray | None = None
    action: np.ndarray | None = None
    action_chunk: np.ndarray | None = None
    contact_pose: np.ndarray | None = None
    force_vec: np.ndarray | None = None
    indent_depth_m: float | None = None
    tactile_pressure: float | None = None
    tactile: PicfTactilePacket | None = None
    tracklet_xy: np.ndarray | None = None
    tracklet_velocity: np.ndarray | None = None
    tracklet_visibility: np.ndarray | None = None
    tracklet_confidence: np.ndarray | None = None
    tracklet_ids: np.ndarray | None = None
    tracklet_view_ids: np.ndarray | None = None
    tracklet_age: np.ndarray | None = None
    proposal_centers_xy: np.ndarray | None = None
    proposal_boxes_xyxy: np.ndarray | None = None
    proposal_objectness: np.ndarray | None = None
    proposal_view_ids: np.ndarray | None = None
    proposal_source_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.rgb_static = np.asarray(self.rgb_static)
        self.depth_static = np.asarray(self.depth_static, dtype=np.float32)
        self.robot_obs = np.asarray(self.robot_obs, dtype=np.float32).reshape(-1)
        if self.rgb_gripper is not None:
            self.rgb_gripper = np.asarray(self.rgb_gripper)
        if self.depth_gripper is not None:
            self.depth_gripper = np.asarray(self.depth_gripper, dtype=np.float32)
        if self.scene_obs is not None:
            self.scene_obs = np.asarray(self.scene_obs, dtype=np.float32).reshape(-1)
        if self.G_t is not None:
            self.G_t = np.asarray(self.G_t, dtype=np.float32)
        if self.proprio is not None:
            self.proprio = np.asarray(self.proprio, dtype=np.float32).reshape(-1)
        if self.tracklet_xy is not None:
            self.tracklet_xy = np.asarray(self.tracklet_xy, dtype=np.float32)
        if self.tracklet_velocity is not None:
            self.tracklet_velocity = np.asarray(self.tracklet_velocity, dtype=np.float32)
        if self.tracklet_visibility is not None:
            self.tracklet_visibility = np.asarray(self.tracklet_visibility, dtype=np.float32).reshape(-1)
        if self.tracklet_confidence is not None:
            self.tracklet_confidence = np.asarray(self.tracklet_confidence, dtype=np.float32).reshape(-1)
        if self.tracklet_ids is not None:
            self.tracklet_ids = np.asarray(self.tracklet_ids, dtype=np.int64).reshape(-1)
        if self.tracklet_view_ids is not None:
            self.tracklet_view_ids = np.asarray(self.tracklet_view_ids, dtype=np.int64).reshape(-1)
        if self.tracklet_age is not None:
            self.tracklet_age = np.asarray(self.tracklet_age, dtype=np.float32).reshape(-1)
        if self.proposal_centers_xy is not None:
            self.proposal_centers_xy = np.asarray(self.proposal_centers_xy, dtype=np.float32).reshape(-1, 2)
        if self.proposal_boxes_xyxy is not None:
            self.proposal_boxes_xyxy = np.asarray(self.proposal_boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if self.proposal_objectness is not None:
            self.proposal_objectness = np.asarray(self.proposal_objectness, dtype=np.float32).reshape(-1)
        if self.proposal_view_ids is not None:
            self.proposal_view_ids = np.asarray(self.proposal_view_ids, dtype=np.int64).reshape(-1)
        if self.proposal_source_ids is not None:
            self.proposal_source_ids = np.asarray(self.proposal_source_ids, dtype=np.int64).reshape(-1)
        if self.action is not None:
            self.action = np.asarray(self.action, dtype=np.float32).reshape(-1)
        if self.action_chunk is not None:
            action_chunk = np.asarray(self.action_chunk, dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[None, :]
            elif action_chunk.ndim != 2:
                raise ValueError(f"action_chunk must have shape (H, A) or (A,), got {action_chunk.shape}")
            self.action_chunk = action_chunk
        if self.contact_pose is not None:
            self.contact_pose = np.asarray(self.contact_pose, dtype=np.float32)
            if self.contact_pose.shape != (4, 4):
                raise ValueError(f"contact_pose must have shape (4,4), got {self.contact_pose.shape}")
        if self.force_vec is not None:
            self.force_vec = np.asarray(self.force_vec, dtype=np.float32).reshape(-1)
            if self.force_vec.shape not in ((3,), (6,)):
                raise ValueError(f"force_vec must have shape (3,) or (6,), got {self.force_vec.shape}")
        if self.indent_depth_m is not None:
            self.indent_depth_m = float(self.indent_depth_m)
        if self.tactile_pressure is not None:
            self.tactile_pressure = float(self.tactile_pressure)


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
