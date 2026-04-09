import pytest

from openpi.picf.anytouch.sensor_registry import NUM_POSE_SENSOR_CLASSES
from openpi.picf.anytouch.sensor_registry import resolve_sensor_id
from openpi.picf.anytouch.sensor_registry import sensor_pose_class


def test_universal_sensor_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="reserved for smoke tests only"):
        resolve_sensor_id("universal", allow_universal=False)


def test_universal_sensor_keeps_distinct_runtime_and_pose_identity() -> None:
    assert resolve_sensor_id("universal", allow_universal=True) == 19
    assert sensor_pose_class("universal") == 6
    assert NUM_POSE_SENSOR_CLASSES == 7
    assert sensor_pose_class("dm") != sensor_pose_class("universal")
