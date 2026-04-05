from __future__ import annotations

from collections.abc import Iterator, Sequence

from openpi.picf.contracts import PicfObservation
from openpi.training.calvin_dataset import CalvinLangSegmentDataset


class CalvinSequentialReplay:
    """Replay CALVIN segments sequentially for scaffold state continuity."""

    def __init__(
        self,
        root: str,
        *,
        split: str = "training",
        backend: str = "zip",
        use_wrist_rgb: bool = True,
        frame_dt_s: float = 1.0 / 30.0,
        segment_indices: Sequence[int] | None = None,
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
        self._frame_dt_s = float(frame_dt_s)
        self._segment_indices = list(segment_indices) if segment_indices is not None else list(range(len(self._segments)))

    def __iter__(self) -> Iterator[PicfObservation]:
        for segment_id in self._segment_indices:
            segment = self._segments[segment_id]
            for offset, step_id in enumerate(range(segment.start, segment.end)):
                keys = ["rgb_static", "depth_static", "robot_obs"]
                if self._use_wrist_rgb:
                    keys.append("rgb_gripper")
                frame = self._reader.read_npz(step_id, keys=keys)
                yield PicfObservation(
                    rgb_static=frame["rgb_static"],
                    depth_static=frame["depth_static"],
                    robot_obs=frame["robot_obs"],
                    prompt=segment.lang,
                    step_id=step_id,
                    segment_id=segment_id,
                    timestamp_s=float(step_id) * self._frame_dt_s,
                    reset_scaffold=(offset == 0),
                    rgb_gripper=frame.get("rgb_gripper"),
                )

    def close(self) -> None:
        self._reader.close()

    def __len__(self) -> int:
        return sum(self._segments[i].end - self._segments[i].start for i in self._segment_indices)
