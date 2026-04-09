from __future__ import annotations

from collections import deque
import dataclasses

import numpy as np

from openpi.picf.contracts import PicfTactilePacket


@dataclasses.dataclass
class _SensorHistory:
    frames: deque[np.ndarray]
    T_sens_to_wrist: np.ndarray | None = None
    background_rgb: np.ndarray | None = None


class MultiSensorTactileClipBuffer:
    def __init__(self, *, num_frames: int = 4, frame_stride: int = 2):
        if num_frames <= 0 or frame_stride <= 0:
            raise ValueError("num_frames and frame_stride must be positive.")
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self._segment_id: int | None = None
        self._histories: dict[str, _SensorHistory] = {}

    def reset(self, *, segment_id: int | None = None) -> None:
        self._segment_id = segment_id
        self._histories.clear()

    def push(self, packet: PicfTactilePacket, *, segment_id: int, reset: bool) -> None:
        if reset or (self._segment_id is not None and self._segment_id != segment_id):
            self.reset(segment_id=segment_id)
        elif self._segment_id is None:
            self._segment_id = segment_id
        for sensor in packet.sensors:
            if not sensor.valid:
                continue
            history = self._histories.setdefault(
                sensor.sensor_name,
                _SensorHistory(frames=deque(maxlen=self.num_frames * self.frame_stride)),
            )
            history.frames.append(np.asarray(sensor.rgb).copy())
            history.T_sens_to_wrist = np.asarray(sensor.T_sens_to_wrist, dtype=np.float32)
            background = packet.background_for(sensor.sensor_name)
            if background is not None:
                history.background_rgb = np.asarray(background).copy()

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._histories))

    def has_frames(self, sensor_name: str) -> bool:
        history = self._histories.get(sensor_name)
        return history is not None and len(history.frames) > 0

    def background_for(self, sensor_name: str) -> np.ndarray | None:
        history = self._histories.get(sensor_name)
        return None if history is None else history.background_rgb

    def latest_pose(self, sensor_name: str) -> np.ndarray | None:
        history = self._histories.get(sensor_name)
        return None if history is None else history.T_sens_to_wrist

    def get_clip(self, sensor_name: str) -> np.ndarray:
        history = self._histories.get(sensor_name)
        if history is None or len(history.frames) == 0:
            raise KeyError(f"No tactile history available for sensor '{sensor_name}'.")
        frames = list(history.frames)
        selected: list[np.ndarray] = []
        cursor = len(frames) - 1
        for _ in range(self.num_frames):
            selected.append(frames[max(cursor, 0)])
            cursor -= self.frame_stride
        selected.reverse()
        return np.stack(selected, axis=0)
