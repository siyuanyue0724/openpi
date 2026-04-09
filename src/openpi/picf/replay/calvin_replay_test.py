from pathlib import Path

import numpy as np
import pytest

from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.test_utils import build_mini_calvin_dataset


def test_calvin_replay_supports_dir_and_zip(tmp_path: Path) -> None:
    dir_root = build_mini_calvin_dataset(tmp_path / "dir_case", make_zip=False)
    zip_root = build_mini_calvin_dataset(tmp_path / "zip_case", make_zip=True)

    dir_frames = list(CalvinSequentialReplay(dir_root, backend="dir", segment_indices=[0]))
    zip_frames = list(CalvinSequentialReplay(zip_root, backend="zip", segment_indices=[0]))

    assert len(dir_frames) == len(zip_frames) == 4
    assert dir_frames[0].reset_scaffold
    assert zip_frames[0].reset_scaffold
    assert dir_frames[0].prompt == zip_frames[0].prompt == "hold pose"
    assert dir_frames[1].step_id == dir_frames[0].step_id + 1
    assert zip_frames[1].step_id == zip_frames[0].step_id + 1


def test_calvin_replay_can_emit_explicit_action_proprio_and_tactile_packet(tmp_path: Path) -> None:
    root = build_mini_calvin_dataset(tmp_path / "tactile_case", make_zip=False)
    frames = list(CalvinSequentialReplay(root, backend="dir", segment_indices=[0], use_tactile=True))

    assert len(frames) == 4
    first = frames[0]
    assert first.proprio is not None
    assert first.action is not None
    assert first.tactile is not None
    assert len(first.tactile.sensors) == 2
    assert first.tactile.sensors[0].rgb.shape[-1] == 3
    assert first.tactile.sensors[1].rgb.shape[-1] == 3
    assert first.tactile.sensors[0].depth is not None
    assert first.tactile.sensors[0].depth.shape[-1] == 1


def test_calvin_replay_use_tactile_fails_fast_when_dataset_lacks_tactile_fields(tmp_path: Path) -> None:
    root = Path(build_mini_calvin_dataset(tmp_path / "strict_case", make_zip=False))
    episode_path = root / "training" / "episode_0000000.npz"
    with np.load(episode_path) as data:
        payload = {key: data[key] for key in data.files if key not in {"rgb_tactile", "depth_tactile"}}
    np.savez(episode_path, **payload)

    replay = CalvinSequentialReplay(str(root), backend="dir", segment_indices=[0], use_tactile=True)
    with pytest.raises(KeyError, match="rgb_tactile|depth_tactile"):
        list(replay)
