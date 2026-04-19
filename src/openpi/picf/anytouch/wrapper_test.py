import numpy as np
import pytest
import torch

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.anytouch.wrapper import AnyTouch2TactileEncoder


def test_anytouch_wrapper_emits_probe_tokens_and_pooled_features() -> None:
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            device="cpu",
            dtype="float32",
            allow_random_init=True,
            trainable=True,
        )
    )
    clip = np.full((4, 32, 32, 3), 120, dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)

    background = np.full((32, 32, 3), 120, dtype=np.uint8)

    bundle = encoder.encode_sensor_clips(
        clips_by_sensor={"digit": clip},
        backgrounds_by_sensor={"digit": background},
        poses_by_sensor={"digit": pose},
    )

    assert bundle is not None
    assert bundle.global_feature.shape == (3072,)
    assert "digit" in bundle.sensors
    assert bundle.sensors["digit"].tokens.shape == (398, 768)
    assert bundle.sensors["digit"].pseudo_contact_score == pytest.approx(0.0, abs=1e-6)
    assert bundle.sensors["digit"].rgb_residual_score == 0.0
    assert bundle.sensors["digit"].contact_score < 1e-6


def test_anytouch_wrapper_emits_pseudo_contact_from_temporal_change() -> None:
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            device="cpu",
            dtype="float32",
            allow_random_init=True,
            require_background=False,
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
    assert bundle.sensors["digit"].rgb_residual_score > 0.0


def test_anytouch_wrapper_uses_calibrated_contact_stats_for_contact_score() -> None:
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            device="cpu",
            dtype="float32",
            allow_random_init=True,
            contact_stats_payload={
                "score_mode": "rgb_latent",
                "per_sensor_rgb_stats": {
                    "digit": {"median": 0.0, "iqr": 1.349},
                },
                "per_sensor_latent_stats": {
                    "digit": {"median": 0.0, "iqr": 1.349},
                },
            },
        )
    )
    clip = np.full((4, 32, 32, 3), 128, dtype=np.uint8)
    clip[-1] = 255
    pose = np.eye(4, dtype=np.float32)
    background = np.zeros((32, 32, 3), dtype=np.uint8)

    bundle = encoder.encode_sensor_clips(
        clips_by_sensor={"digit": clip},
        backgrounds_by_sensor={"digit": background},
        poses_by_sensor={"digit": pose},
    )

    assert bundle is not None
    sensor = bundle.sensors["digit"]
    expected = 0.5 * (sensor.rgb_residual_score + sensor.latent_residual_score)
    assert sensor.contact_score == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_anytouch_wrapper_uses_train_checkpoint_when_trainable(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            device="cpu",
            dtype="float32",
            allow_random_init=True,
            trainable=True,
        )
    )
    encoder.train()
    called = {"count": 0}

    def _checkpoint(func, *args, **kwargs):
        called["count"] += 1
        return func(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _checkpoint)
    clip = np.full((4, 32, 32, 3), 120, dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    background = np.full((32, 32, 3), 120, dtype=np.uint8)

    bundle = encoder.encode_sensor_clips(
        clips_by_sensor={"digit": clip},
        backgrounds_by_sensor={"digit": background},
        poses_by_sensor={"digit": pose},
    )

    assert bundle is not None
    assert called["count"] >= 2
