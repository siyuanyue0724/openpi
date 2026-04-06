from pathlib import Path

from openpi.picf.test_utils import build_mini_calvin_dataset

from .posterior_visual_acceptance_check import run_visual_acceptance_check
from .posterior_visual_full_check import run_visual_full_check
from .posterior_visual_invariant_audit import run_visual_invariant_audit
from .posterior_visual_replay_smoke import run_visual_smoke
from .posterior_visual_stage1_spec_audit import run_visual_stage1_spec_audit


def test_visual_posterior_scripts_run_on_mini_dataset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=True)
    kwargs = {
        "calvin_root": calvin_root,
        "split": "training",
        "backend": "zip",
        "num_segments": 2,
        "max_points": 256,
        "arch_name_override": "vit_tiny",
        "img_size": 64,
        "num_frames": 4,
        "device": "cpu",
        "dtype": "float32",
    }
    smoke_visual_only = run_visual_smoke(mode="visual_only", **kwargs)
    smoke_point_visual = run_visual_smoke(mode="point_visual", **kwargs)
    invariants = run_visual_invariant_audit(mode="point_visual", **kwargs)
    acceptance = run_visual_acceptance_check(mode="point_visual", **kwargs)

    assert smoke_visual_only["frames"] == 8
    assert smoke_visual_only["mean_visual_gate_ratio"] > 0.0
    assert smoke_point_visual["mean_point_precision_gain_count"] > 0.0
    assert smoke_point_visual["mean_visual_precision_gain_count"] > 0.0
    assert invariants["all_pass"]
    assert acceptance["all_pass"]

    spec = run_visual_stage1_spec_audit(repo_root)
    full = run_visual_full_check(repo_root=str(repo_root), **kwargs)
    assert spec["all_pass"]
    assert full["all_pass"]
