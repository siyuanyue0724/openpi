from __future__ import annotations

from collections import deque

import numpy as np


class VisualClipBuffer:
    """Segment-scoped visual history with deterministic left padding."""

    def __init__(self, *, num_frames: int):
        self.num_frames = int(num_frames)
        self._frames: deque[np.ndarray] = deque(maxlen=self.num_frames)
        self._segment_id: int | None = None

    def reset(self, *, segment_id: int | None = None) -> None:
        self._frames.clear()
        self._segment_id = segment_id

    def push(self, frame: np.ndarray, *, segment_id: int, reset: bool = False) -> None:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected frame shape [H,W,3], got {frame.shape}")
        if reset or self._segment_id is None or int(segment_id) != int(self._segment_id):
            self.reset(segment_id=int(segment_id))
        self._frames.append(frame.copy())

    def get_clip(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("VisualClipBuffer is empty.")
        frames = list(self._frames)
        if len(frames) < self.num_frames:
            pad = [frames[0].copy() for _ in range(self.num_frames - len(frames))]
            frames = pad + frames
        return np.stack(frames[-self.num_frames :], axis=0)
