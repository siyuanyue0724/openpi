from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpi.models.tokenizer import PaligemmaTokenizer
from scripts.picf_core_train import _CalvinTransitionSource


def _shape(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(np.asarray(value).shape)


def _finite_stats(name: str, value) -> dict[str, object]:
    if value is None:
        return {f"{name}_shape": None}
    arr = np.asarray(value)
    finite = np.isfinite(arr)
    return {
        f"{name}_shape": tuple(arr.shape),
        f"{name}_finite": bool(finite.all()),
        f"{name}_min": float(arr.min()) if arr.size > 0 else None,
        f"{name}_max": float(arr.max()) if arr.size > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect specific CALVIN transition windows used by PICF training.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--backend", default="dir", choices=["dir", "zip"])
    parser.add_argument("--unroll-steps", type=int, default=2)
    parser.add_argument("--flat-indices", required=True, help="Comma-separated flat window indices.")
    parser.add_argument("--use-wrist-rgb", action="store_true", default=True)
    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--max-token-len", type=int, default=48)
    args = parser.parse_args()

    source = _CalvinTransitionSource(
        args.calvin_root,
        split=args.split,
        backend=args.backend,
        unroll_steps=args.unroll_steps,
        use_wrist_rgb=bool(args.use_wrist_rgb),
        use_tactile=bool(args.use_tactile),
    )
    tokenizer = PaligemmaTokenizer(max_len=int(args.max_token_len))
    try:
        for raw in [part.strip() for part in str(args.flat_indices).split(",") if part.strip()]:
            flat_index = int(raw)
            window = source.window(flat_index)
            token_ids, token_mask = tokenizer.tokenize(str(window.prompt), state=None)
            token_ids = np.asarray(token_ids)
            token_mask = np.asarray(token_mask, dtype=bool)
            valid_ids = token_ids[token_mask]
            frames = []
            for frame in window.frames:
                tactile_shapes = None
                if frame.tactile is not None:
                    tactile_shapes = [
                        {
                            "sensor_name": sensor.sensor_name,
                            "rgb_shape": tuple(sensor.rgb.shape),
                            "depth_shape": None if sensor.depth is None else tuple(np.asarray(sensor.depth).shape),
                        }
                        for sensor in frame.tactile.sensors
                    ]
                frames.append(
                    {
                        "step_id": int(frame.step_id),
                        "rgb_static_shape": _shape(frame.rgb_static),
                        "rgb_gripper_shape": _shape(frame.rgb_gripper),
                        **_finite_stats("depth_static", frame.depth_static),
                        "robot_obs_shape": _shape(frame.robot_obs),
                        "action_shape": _shape(frame.action),
                        "tactile": tactile_shapes,
                    }
                )
            payload = {
                "flat_index": flat_index,
                "segment": int(window.segment_id),
                "start_step": int(window.start_step_id),
                "prompt": str(window.prompt),
                "prompt_char_len": len(str(window.prompt)),
                "prompt_token_count": int(token_mask.sum()),
                "prompt_token_min": int(valid_ids.min()) if valid_ids.size > 0 else None,
                "prompt_token_max": int(valid_ids.max()) if valid_ids.size > 0 else None,
                "window_num_frames": len(window.frames),
                "frames": frames,
            }
            print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)
    finally:
        source.close()


if __name__ == "__main__":
    main()
