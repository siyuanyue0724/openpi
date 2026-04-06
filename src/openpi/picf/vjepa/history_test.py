import numpy as np

from openpi.picf.vjepa.history import VisualClipBuffer


def test_visual_clip_buffer_left_pads_and_resets() -> None:
    buffer = VisualClipBuffer(num_frames=4)
    frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_b = np.full((8, 8, 3), 7, dtype=np.uint8)

    buffer.push(frame_a, segment_id=0, reset=True)
    clip = buffer.get_clip()
    assert clip.shape == (4, 8, 8, 3)
    assert np.all(clip[0] == frame_a)
    assert np.all(clip[-1] == frame_a)

    buffer.push(frame_b, segment_id=0)
    clip = buffer.get_clip()
    assert np.all(clip[-1] == frame_b)
    assert np.all(clip[-2] == frame_a)

    buffer.push(frame_b, segment_id=1)
    clip = buffer.get_clip()
    assert np.all(clip[0] == frame_b)
    assert np.all(clip[-1] == frame_b)
