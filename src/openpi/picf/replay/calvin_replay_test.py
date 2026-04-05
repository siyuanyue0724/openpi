from pathlib import Path

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
