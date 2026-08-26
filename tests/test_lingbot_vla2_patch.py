from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_lingbot_vla2_patch import verify_patch

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_lingbot_patch_applies_and_covers_training_and_inference() -> None:
    if not (ROOT / "references/source_checkouts/lingbot-vla-v2/.git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    result = verify_patch(root=ROOT, check_apply=True, required_state="either")
    assert result["apply_checked"] is True
    assert result["patch_state"] in {"baseline", "applied"}


def test_lingbot_patch_verifier_rejects_an_incomplete_patch(tmp_path: Path) -> None:
    (tmp_path / "references" / "patches").mkdir(parents=True)
    patch = tmp_path / "references" / "patches" / "lingbot_vla2_action_layer_adapter.patch"
    patch.write_text("class QwenvlWithExpertV2Model\n")
    with pytest.raises(ValueError, match="omits required"):
        verify_patch(root=tmp_path, check_apply=False)
