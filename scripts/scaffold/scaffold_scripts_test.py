from pathlib import Path

from openpi.picf.test_utils import build_mini_calvin_dataset

from .scaffold_replay_smoke import run_smoke
from .scaffold_stability_eval import run_stability_eval


def test_scaffold_scripts_run_on_mini_dataset(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=True)
    smoke = run_smoke(calvin_root=calvin_root, split="training", backend="zip", num_segments=2, max_points=128)
    stability = run_stability_eval(
        calvin_root=calvin_root,
        split="training",
        backend="zip",
        num_segments=2,
        max_points=128,
    )

    assert smoke["frames"] == 8
    assert smoke["mean_num_active"] > 0
    assert "baseline" in stability
    assert stability["baseline"]["frames"] == 8
