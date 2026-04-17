from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
from pathlib import Path

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import PicfTactilePacket
from openpi.picf.contracts import TactileSensorFrame
from openpi.picf.geometry import make_transform
from openpi.picf.geometry import normalize_vectors
from openpi.training.calvin_dataset import CalvinLangSegmentDataset

_LEGACY_TACTILE_SENSOR_OFFSETS_M = ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0))


def _orthonormal_sensor_frame(normal_local: np.ndarray, up_local: np.ndarray) -> np.ndarray:
    x_axis = normalize_vectors(np.asarray(normal_local, dtype=np.float32).reshape(1, 3))[0]
    up_ref = normalize_vectors(np.asarray(up_local, dtype=np.float32).reshape(1, 3))[0]
    if abs(float(np.dot(x_axis, up_ref))) > 0.95:
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(x_axis, up_ref))) > 0.95:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y_axis = normalize_vectors(np.cross(up_ref, x_axis)[None, :])[0]
    z_axis = normalize_vectors(np.cross(x_axis, y_axis)[None, :])[0]
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def _resolve_tactile_calibration(
    calibration: dict[str, object] | str | Path | None,
) -> dict[str, np.ndarray | float]:
    if calibration is None:
        payload: dict[str, object] = {}
    elif isinstance(calibration, (str, Path)):
        payload = json.loads(Path(calibration).read_text(encoding="utf-8"))
    else:
        payload = dict(calibration)
    return {
        "u_open_local": np.asarray(payload.get("u_open_local", (1.0, 0.0, 0.0)), dtype=np.float32).reshape(3),
        "o_local": np.asarray(payload.get("o_local", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(3),
        "up_local": np.asarray(payload.get("up_local", (0.0, 0.0, 1.0)), dtype=np.float32).reshape(3),
        "sensor_centers_local": (
            None
            if payload.get("sensor_centers_local") is None
            else np.asarray(payload["sensor_centers_local"], dtype=np.float32).reshape(2, 3)
        ),
        "left_normal_sign": float(payload.get("left_normal_sign", 1.0)),
        "right_normal_sign": float(payload.get("right_normal_sign", -1.0)),
    }


def _calvin_tactile_packet(
    frame: dict[str, np.ndarray],
    *,
    timestamp_s: float,
    sensor_names: Sequence[str],
    robot_obs: np.ndarray,
    calibration: dict[str, np.ndarray | float],
    background_rgb_by_sensor: dict[str, np.ndarray] | None,
) -> PicfTactilePacket | None:
    rgb_tactile = frame.get("rgb_tactile")
    if rgb_tactile is None:
        raise ValueError("rgb_tactile is required when constructing a CALVIN tactile packet.")
    rgb = np.asarray(rgb_tactile)
    if rgb.ndim != 3 or rgb.shape[-1] % 3 != 0:
        raise ValueError(f"rgb_tactile must have shape [H,W,3*K], got {rgb.shape}")
    num_sensors = rgb.shape[-1] // 3
    if num_sensors != len(sensor_names):
        raise ValueError(
            "Configured tactile sensor count does not match CALVIN tactile channels: "
            f"channels imply {num_sensors} sensors, names={len(sensor_names)}."
        )
    depth_tactile = frame.get("depth_tactile")
    depth = np.asarray(depth_tactile, dtype=np.float32) if depth_tactile is not None else None
    if depth is not None and (depth.ndim != 3 or depth.shape[-1] != num_sensors):
        raise ValueError(
            f"depth_tactile must have shape [H,W,{num_sensors}] when rgb_tactile has {num_sensors} sensors, got {depth.shape}"
        )
    sensors: list[TactileSensorFrame] = []
    robot_obs = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
    if num_sensors != 2:
        raise ValueError(f"CALVIN tactile packet currently expects 2 sensors, got {num_sensors}.")
    gripper_width = float(max(robot_obs[6], 0.0)) if robot_obs.shape[0] > 6 else 0.0
    u_open_local = np.asarray(calibration["u_open_local"], dtype=np.float32).reshape(3)
    o_local = np.asarray(calibration["o_local"], dtype=np.float32).reshape(3)
    up_local = np.asarray(calibration["up_local"], dtype=np.float32).reshape(3)
    if calibration["sensor_centers_local"] is None:
        left_center = o_local + (0.5 * gripper_width * u_open_local)
        right_center = o_local - (0.5 * gripper_width * u_open_local)
        centers = (left_center, right_center)
    else:
        centers = tuple(np.asarray(calibration["sensor_centers_local"], dtype=np.float32))
    normal_signs = (
        float(calibration["left_normal_sign"]),
        float(calibration["right_normal_sign"]),
    )
    for idx, sensor_name in enumerate(sensor_names):
        normal_local = normal_signs[idx] * u_open_local
        pose = make_transform(_orthonormal_sensor_frame(normal_local, up_local), centers[idx])
        sensors.append(
            TactileSensorFrame(
                rgb=rgb[..., 3 * idx : 3 * (idx + 1)],
                depth=None if depth is None else depth[..., idx : idx + 1],
                sensor_name=sensor_name,
                T_sens_to_wrist=pose,
                timestamp_s=timestamp_s,
            )
        )
    return PicfTactilePacket(sensors=tuple(sensors), background_rgb_by_sensor=background_rgb_by_sensor)


def _load_action_chunk(
    reader,
    *,
    step_id: int,
    segment_end: int,
    action_horizon: int,
    current_action: np.ndarray | None = None,
    action_key: str = "rel_actions",
) -> np.ndarray | None:
    if int(action_horizon) <= 1:
        return None
    if current_action is None:
        current = reader.read_npz(step_id, keys=[action_key])[action_key]
    else:
        current = np.asarray(current_action, dtype=np.float32)
    actions = [np.asarray(current, dtype=np.float32)]
    last = actions[0]
    for future_step in range(step_id + 1, step_id + int(action_horizon)):
        if future_step < int(segment_end):
            last = np.asarray(reader.read_npz(future_step, keys=[action_key])[action_key], dtype=np.float32)
        actions.append(last)
    return np.stack(actions, axis=0)


class CalvinSequentialReplay:
    """Replay CALVIN segments sequentially for scaffold state continuity."""

    def __init__(
        self,
        root: str,
        *,
        split: str = "training",
        backend: str = "zip",
        action_horizon: int = 1,
        use_wrist_rgb: bool = True,
        use_tactile: bool = False,
        frame_dt_s: float = 1.0 / 30.0,
        segment_indices: Sequence[int] | None = None,
        tactile_sensor_names: Sequence[str] = ("digit", "gelsight_mini"),
        tactile_sensor_offsets_m: Sequence[tuple[float, float, float]] | None = None,
        tactile_calibration: dict[str, object] | str | Path | None = None,
        tactile_backgrounds_by_sensor: dict[str, np.ndarray] | None = None,
        use_scene_obs: bool = False,
    ):
        if int(action_horizon) < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
        self._dataset = CalvinLangSegmentDataset(
            root=root,
            split=split,
            action_horizon=int(action_horizon),
            backend=backend,
            use_wrist_rgb=use_wrist_rgb,
            sample_within_segment=False,
        )
        self._reader = self._dataset.reader
        self._segments = self._dataset.segments
        self._use_wrist_rgb = bool(use_wrist_rgb)
        self._use_tactile = bool(use_tactile)
        self._frame_dt_s = float(frame_dt_s)
        self._action_horizon = int(action_horizon)
        self._segment_indices = list(segment_indices) if segment_indices is not None else list(range(len(self._segments)))
        self._tactile_sensor_names = tuple(tactile_sensor_names)
        calibration_payload = tactile_calibration
        explicit_legacy_offsets = (
            tactile_sensor_offsets_m is not None
            and tuple(tuple(float(value) for value in offset) for offset in tactile_sensor_offsets_m) != _LEGACY_TACTILE_SENSOR_OFFSETS_M
        )
        if explicit_legacy_offsets and tactile_calibration is None:
            calibration_payload = {"sensor_centers_local": tactile_sensor_offsets_m}
        self._tactile_calibration = _resolve_tactile_calibration(calibration_payload)
        self._tactile_backgrounds_by_sensor = None if tactile_backgrounds_by_sensor is None else {
            str(name): np.asarray(image) for name, image in tactile_backgrounds_by_sensor.items()
        }
        self._use_scene_obs = bool(use_scene_obs)

    def __iter__(self) -> Iterator[PicfObservation]:
        for segment_id in self._segment_indices:
            segment = self._segments[segment_id]
            for offset, step_id in enumerate(range(segment.start, segment.end)):
                keys = ["rgb_static", "depth_static", "depth_gripper", "robot_obs", "rel_actions"]
                if self._use_wrist_rgb:
                    keys.append("rgb_gripper")
                if self._use_tactile:
                    keys.extend(["rgb_tactile", "depth_tactile"])
                if self._use_scene_obs:
                    keys.append("scene_obs")
                frame = self._reader.read_npz(step_id, keys=keys)
                timestamp_s = float(step_id) * self._frame_dt_s
                action_chunk = _load_action_chunk(
                    self._reader,
                    step_id=step_id,
                    segment_end=segment.end,
                    action_horizon=self._action_horizon,
                    current_action=frame.get("rel_actions"),
                )
                yield PicfObservation(
                    rgb_static=frame["rgb_static"],
                    depth_static=frame["depth_static"],
                    depth_gripper=frame.get("depth_gripper"),
                    robot_obs=frame["robot_obs"],
                    prompt=segment.lang,
                    step_id=step_id,
                    segment_id=segment_id,
                    timestamp_s=timestamp_s,
                    reset_scaffold=(offset == 0),
                    rgb_gripper=frame.get("rgb_gripper"),
                    scene_obs=frame.get("scene_obs"),
                    proprio=frame["robot_obs"],
                    action=frame.get("rel_actions"),
                    action_chunk=action_chunk,
                    tactile=(
                        _calvin_tactile_packet(
                            frame,
                            timestamp_s=timestamp_s,
                            sensor_names=self._tactile_sensor_names,
                            robot_obs=frame["robot_obs"],
                            calibration=self._tactile_calibration,
                            background_rgb_by_sensor=self._tactile_backgrounds_by_sensor,
                        )
                        if self._use_tactile
                        else None
                    ),
                )

    def close(self) -> None:
        self._reader.close()

    def __len__(self) -> int:
        return sum(self._segments[i].end - self._segments[i].start for i in self._segment_indices)
