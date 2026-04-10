import numpy as np

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.anytouch.wrapper import AnyTouch2TactileEncoder


def test_anytouch_wrapper_emits_probe_tokens_and_pooled_features() -> None:
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            device="cpu",
            dtype="float32",
            allow_random_init=True,
        )
    )
    clip = np.full((4, 32, 32, 3), 120, dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)

    bundle = encoder.encode_sensor_clips(
        clips_by_sensor={"digit": clip},
        backgrounds_by_sensor={"digit": np.zeros((32, 32, 3), dtype=np.uint8)},
        poses_by_sensor={"digit": pose},
    )

    assert bundle is not None
    assert bundle.global_feature.shape == (3072,)
    assert "digit" in bundle.sensors
    assert bundle.sensors["digit"].tokens.shape == (398, 768)
    assert bundle.sensors["digit"].pseudo_contact_score == 0.0


def test_anytouch_wrapper_emits_pseudo_contact_from_temporal_change() -> None:
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            device="cpu",
            dtype="float32",
            allow_random_init=True,
        )
    )
    clip = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    clip[-1] = 255
    pose = np.eye(4, dtype=np.float32)

    bundle = encoder.encode_sensor_clips(
        clips_by_sensor={"digit": clip},
        backgrounds_by_sensor={},
        poses_by_sensor={"digit": pose},
    )

    assert bundle is not None
    assert bundle.sensors["digit"].pseudo_contact_score > 0.0
