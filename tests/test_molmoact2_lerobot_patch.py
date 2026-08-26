from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_molmoact2_lerobot_patch import verify_patch

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_molmoact2_lerobot_patch_is_exact_and_reversible() -> None:
    if not (ROOT / "references/source_checkouts/molmoact2-lerobot/.git").exists():
        pytest.skip("optional pinned MolmoAct2 LeRobot checkout is absent")
    result = verify_patch(
        root=ROOT,
        check_apply=True,
        required_state="applied",
    )
    assert result["apply_checked"] is True
    assert result["patch_state"] == "applied"


def test_molmoact2_patch_verifier_rejects_incomplete_patch(tmp_path: Path) -> None:
    patch = tmp_path / "references" / "patches" / "molmoact2_lerobot_action_layer_adapter.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("class MolmoAct2Policy\n")
    with pytest.raises(ValueError, match="omits required"):
        verify_patch(root=tmp_path, check_apply=False)
