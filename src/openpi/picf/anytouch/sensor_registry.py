from __future__ import annotations

SENSOR_NAME_TO_ID = {
    "gelsight": 0,
    "digit": 1,
    "gelslim": 2,
    "gelsight_mini": 3,
    "duragel": 4,
    "dm": 5,
    "universal": -1,
}

POSE_SENSOR_CLASS = {
    "gelsight": 0,
    "digit": 1,
    "gelslim": 2,
    "gelsight_mini": 3,
    "duragel": 4,
    "dm": 5,
    "universal": 6,
}

NUM_POSE_SENSOR_CLASSES = len(POSE_SENSOR_CLASS)


def resolve_sensor_id(sensor_name: str, *, allow_universal: bool) -> int:
    key = str(sensor_name)
    if key not in SENSOR_NAME_TO_ID:
        raise KeyError(f"Unknown AnyTouch sensor '{sensor_name}'.")
    sensor_id = int(SENSOR_NAME_TO_ID[key])
    if sensor_id < 0 and not allow_universal:
        raise ValueError(
            "The AnyTouch 'universal' sensor token is reserved for smoke tests only. "
            "Use an explicit hardware sensor id for force-aware deployment."
        )
    if sensor_id < 0:
        sensor_id = 19
    return sensor_id


def sensor_pose_class(sensor_name: str) -> int:
    key = str(sensor_name)
    if key not in POSE_SENSOR_CLASS:
        raise KeyError(f"Unknown AnyTouch sensor '{sensor_name}'.")
    return int(POSE_SENSOR_CLASS[key])
