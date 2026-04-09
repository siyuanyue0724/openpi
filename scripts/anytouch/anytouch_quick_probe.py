from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import tyro

from openpi.picf.anytouch import AnyTouch2TactileEncoder
from openpi.picf.anytouch import AnyTouchConfig


def main(
    example_dir: str = "TEMP_REPO/AnyTouch2-main/example_data",
    checkpoint_path: str | None = None,
    sensor_name: str = "digit",
    *,
    allow_random_init: bool = False,
) -> None:
    data_dir = Path(example_dir)
    clip = np.stack(
        [
            np.asarray(Image.open(data_dir / f"{index}.png").convert("RGB"))
            for index in (0, 2, 4, 6)
        ],
        axis=0,
    )
    background = np.asarray(Image.open(data_dir / "bg.png").convert("RGB"))
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            checkpoint_path=checkpoint_path,
            allow_random_init=allow_random_init,
        )
    )
    bundle = encoder.encode_sensor_clips(
        clips_by_sensor={sensor_name: clip},
        backgrounds_by_sensor={sensor_name: background},
        poses_by_sensor={sensor_name: np.eye(4, dtype=np.float32)},
    )
    if bundle is None:
        raise RuntimeError("AnyTouch quick probe produced no features.")
    sensor = bundle.sensors[sensor_name]
    print("sensor_name:", sensor_name)
    print("checkpoint_loaded:", bundle.checkpoint_loaded)
    print("tokens_shape:", tuple(sensor.tokens.shape))
    print("pooled_shape:", tuple(sensor.pooled_feature.shape))
    print("global_shape:", tuple(bundle.global_feature.shape))


if __name__ == "__main__":
    tyro.cli(main)
