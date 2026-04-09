from __future__ import annotations

import argparse
import json

from openpi.picf.anytouch import AnyTouch2TactileEncoder
from openpi.picf.anytouch import AnyTouchConfig
from openpi.picf.anytouch import MultiSensorTactileClipBuffer
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test CALVIN tactile replay through the PICF AnyTouch2 wrapper.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--backend", choices=("dir", "zip"), default="zip")
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--allow-random-init", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    encoder = AnyTouch2TactileEncoder(
        AnyTouchConfig(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            dtype=args.dtype,
            allow_random_init=bool(args.allow_random_init),
        )
    )
    replay = CalvinSequentialReplay(
        args.calvin_root,
        backend=args.backend,
        segment_indices=list(range(args.segments)),
        use_tactile=True,
    )
    buffer = MultiSensorTactileClipBuffer(num_frames=4, frame_stride=2)
    frames_seen = 0
    tactile_frames = 0
    encoded = None
    for frame in replay:
        frames_seen += 1
        if frame.reset_scaffold:
            buffer.reset(segment_id=int(frame.segment_id))
        if frame.tactile is None:
            raise RuntimeError("Replay emitted no tactile packet while use_tactile=True.")
        tactile_frames += 1
        buffer.push(frame.tactile, segment_id=int(frame.segment_id), reset=bool(frame.reset_scaffold))
        sensor_names = [name for name in buffer.sensor_names if buffer.has_frames(name)]
        if not sensor_names:
            raise RuntimeError("Tactile replay buffer produced no valid sensor history.")
        encoded = encoder.encode_sensor_clips(
            clips_by_sensor={name: buffer.get_clip(name) for name in sensor_names},
            backgrounds_by_sensor={name: buffer.background_for(name) for name in sensor_names},
            poses_by_sensor={name: buffer.latest_pose(name) for name in sensor_names},
        )
        if encoded is not None:
            break
    if encoded is None:
        raise RuntimeError("No tactile clip could be encoded from replay.")
    first_sensor = next(iter(encoded.sensors.values()))
    print(
        json.dumps(
            {
                "frames_seen": frames_seen,
                "tactile_frames": tactile_frames,
                "checkpoint_loaded": encoded.checkpoint_loaded,
                "sensor_names": list(encoded.sensors),
                "first_tokens_shape": list(first_sensor.tokens.shape),
                "pooled_dim": encoded.pooled_dim,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
