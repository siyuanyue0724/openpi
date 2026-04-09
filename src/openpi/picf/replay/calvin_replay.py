from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.contracts import PicfTactilePacket
from openpi.picf.contracts import TactileSensorFrame
from openpi.training.calvin_dataset import CalvinLangSegmentDataset


def _calvin_tactile_packet(
    frame: dict[str, np.ndarray],
    *,
    timestamp_s: float,
    sensor_names: Sequence[str],
    sensor_offsets_m: Sequence[tuple[float, float, float]],
) -> PicfTactilePacket | None:
    rgb_tactile = frame.get("rgb_tactile")
    if rgb_tactile is None:
        raise ValueError("rgb_tactile is required when constructing a CALVIN tactile packet.")
    rgb = np.asarray(rgb_tactile)
    if rgb.ndim != 3 or rgb.shape[-1] % 3 != 0:
        raise ValueError(f"rgb_tactile must have shape [H,W,3*K], got {rgb.shape}")
    num_sensors = rgb.shape[-1] // 3
    if num_sensors != len(sensor_names) or num_sensors != len(sensor_offsets_m):
        raise ValueError(
            "Configured tactile sensor count does not match CALVIN tactile channels: "
            f"channels imply {num_sensors} sensors, names={len(sensor_names)}, offsets={len(sensor_offsets_m)}."
        )
    depth_tactile = frame.get("depth_tactile")
    depth = np.asarray(depth_tactile, dtype=np.float32) if depth_tactile is not None else None
    if depth is not None and (depth.ndim != 3 or depth.shape[-1] != num_sensors):
        raise ValueError(
            f"depth_tactile must have shape [H,W,{num_sensors}] when rgb_tactile has {num_sensors} sensors, got {depth.shape}"
        )
    sensors: list[TactileSensorFrame] = []
    for idx, (sensor_name, offset) in enumerate(zip(sensor_names, sensor_offsets_m, strict=True)):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = np.asarray(offset, dtype=np.float32)
        sensors.append(
            TactileSensorFrame(
                rgb=rgb[..., 3 * idx : 3 * (idx + 1)],
                depth=None if depth is None else depth[..., idx : idx + 1],
                sensor_name=sensor_name,
                T_sens_to_wrist=pose,
                timestamp_s=timestamp_s,
            )
        )
    return PicfTactilePacket(sensors=tuple(sensors))


class CalvinSequentialReplay:
    """Replay CALVIN segments sequentially for scaffold state continuity."""

    def __init__(
        self,
        root: str,
        *,
        split: str = "training",
        backend: str = "zip",
        use_wrist_rgb: bool = True,
        use_tactile: bool = False,
        frame_dt_s: float = 1.0 / 30.0,
        segment_indices: Sequence[int] | None = None,
        tactile_sensor_names: Sequence[str] = ("digit", "gelsight_mini"),
        tactile_sensor_offsets_m: Sequence[tuple[float, float, float]] = ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0)),
    ):
        self._dataset = CalvinLangSegmentDataset(
            root=root,
            split=split,
            action_horizon=1,
            backend=backend,
            use_wrist_rgb=use_wrist_rgb,
            sample_within_segment=False,
        )
        self._reader = self._dataset.reader
        self._segments = self._dataset.segments
        self._use_wrist_rgb = bool(use_wrist_rgb)
        self._use_tactile = bool(use_tactile)
        self._frame_dt_s = float(frame_dt_s)
        self._segment_indices = list(segment_indices) if segment_indices is not None else list(range(len(self._segments)))
        self._tactile_sensor_names = tuple(tactile_sensor_names)
        self._tactile_sensor_offsets_m = tuple(tuple(offset) for offset in tactile_sensor_offsets_m)

    def __iter__(self) -> Iterator[PicfObservation]:
        for segment_id in self._segment_indices:
            segment = self._segments[segment_id]
            for offset, step_id in enumerate(range(segment.start, segment.end)):
                keys = ["rgb_static", "depth_static", "robot_obs", "rel_actions"]
                if self._use_wrist_rgb:
                    keys.append("rgb_gripper")
                if self._use_tactile:
                    keys.extend(["rgb_tactile", "depth_tactile"])
                frame = self._reader.read_npz(step_id, keys=keys)
                timestamp_s = float(step_id) * self._frame_dt_s
                yield PicfObservation(
                    rgb_static=frame["rgb_static"],
                    depth_static=frame["depth_static"],
                    robot_obs=frame["robot_obs"],
                    prompt=segment.lang,
                    step_id=step_id,
                    segment_id=segment_id,
                    timestamp_s=timestamp_s,
                    reset_scaffold=(offset == 0),
                    rgb_gripper=frame.get("rgb_gripper"),
                    proprio=frame["robot_obs"],
                    action=frame.get("rel_actions"),
                    tactile=(
                        _calvin_tactile_packet(
                            frame,
                            timestamp_s=timestamp_s,
                            sensor_names=self._tactile_sensor_names,
                            sensor_offsets_m=self._tactile_sensor_offsets_m,
                        )
                        if self._use_tactile
                        else None
                    ),
                )

    def close(self) -> None:
        self._reader.close()

    def __len__(self) -> int:
        return sum(self._segments[i].end - self._segments[i].start for i in self._segment_indices)
