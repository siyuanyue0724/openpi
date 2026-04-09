import numpy as np

from openpi.picf.anytouch.history import MultiSensorTactileClipBuffer
from openpi.picf.contracts import PicfTactilePacket
from openpi.picf.contracts import TactileSensorFrame


def _packet(step: int) -> PicfTactilePacket:
    image = np.full((8, 8, 3), step, dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    return PicfTactilePacket(
        sensors=(TactileSensorFrame(rgb=image, sensor_name="digit", T_sens_to_wrist=pose, timestamp_s=float(step)),),
        background_rgb_by_sensor={"digit": np.zeros((8, 8, 3), dtype=np.uint8)},
    )


def test_tactile_history_uses_stride_sampling() -> None:
    history = MultiSensorTactileClipBuffer(num_frames=4, frame_stride=2)
    for step in range(7):
        history.push(_packet(step), segment_id=0, reset=(step == 0))

    clip = history.get_clip("digit")

    assert clip.shape == (4, 8, 8, 3)
    assert [int(frame[0, 0, 0]) for frame in clip] == [0, 2, 4, 6]
    assert history.background_for("digit") is not None
    assert history.latest_pose("digit") is not None
