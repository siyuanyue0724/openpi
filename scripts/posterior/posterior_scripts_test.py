from pathlib import Path

from openpi.picf.test_utils import build_mini_calvin_dataset

from .posterior_acceptance_check import run_acceptance_check
from .posterior_full_check import run_full_check
from .posterior_invariant_audit import run_invariant_audit
from .posterior_replay_smoke import run_smoke
from .posterior_spec_audit import run_spec_audit


def test_posterior_scripts_run_on_mini_dataset(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=True)
    smoke = run_smoke(calvin_root=calvin_root, split="training", backend="zip", num_segments=2, max_points=256)
    invariants = run_invariant_audit(
        calvin_root=calvin_root,
        split="training",
        backend="zip",
        num_segments=2,
        max_points=256,
    )
    acceptance = run_acceptance_check(
        calvin_root=calvin_root,
        split="training",
        backend="zip",
        num_segments=2,
        max_points=256,
    )

    assert smoke["frames"] == 8
    assert invariants["all_pass"]
    assert acceptance["all_pass"]
    spec = run_spec_audit(repo_root)
    full = run_full_check(
        repo_root=str(repo_root),
        calvin_root=calvin_root,
        split="training",
        backend="zip",
        num_segments=2,
        max_points=256,
    )
    assert spec["all_pass"]
    assert full["all_pass"]
