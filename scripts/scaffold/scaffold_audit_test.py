from pathlib import Path

from openpi.picf.test_utils import build_mini_calvin_dataset

from .scaffold_acceptance_check import run_acceptance_check
from .scaffold_invariant_audit import run_invariant_audit


def test_scaffold_audits_pass_on_mini_dataset(tmp_path: Path) -> None:
    calvin_root = build_mini_calvin_dataset(tmp_path, make_zip=True)
    invariants = run_invariant_audit(
        calvin_root=calvin_root,
        split="training",
        backend="zip",
        num_segments=2,
        max_points=128,
    )
    acceptance = run_acceptance_check(
        calvin_root=calvin_root,
        split="training",
        backend="zip",
        num_segments=2,
        max_points=128,
    )

    assert invariants["all_pass"]
    assert acceptance["all_pass"]
