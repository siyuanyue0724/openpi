from __future__ import annotations

from pathlib import Path

import pytest

from tools.bootstrap_lingbot_vla2_native import CHECKOUT_RELATIVE_PATH
from tools.bootstrap_lingbot_vla2_native_vl import (
    NATIVE_VL_PATCH_RELATIVE_PATH,
    NATIVE_VL_PATCH_SHA256,
    NATIVE_VL_PATCHED_MODEL_SHA256,
    _validate_native_vl_patch,
    verify_native_vl_patch,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / CHECKOUT_RELATIVE_PATH


@pytest.fixture
def local_source() -> Path:
    if not (SOURCE / ".git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    return SOURCE


def test_native_vl_overlay_replays_from_immutable_lingbot_commit(local_source: Path) -> None:
    report = verify_native_vl_patch(root=ROOT, checkout=local_source, check_apply=True)
    assert report["apply_checked"] is True
    assert report["native_vl_patch_sha256"] == NATIVE_VL_PATCH_SHA256
    assert report["patched_model_sha256"] == NATIVE_VL_PATCHED_MODEL_SHA256


def test_native_vl_overlay_rejects_tampering(tmp_path: Path) -> None:
    source = ROOT / NATIVE_VL_PATCH_RELATIVE_PATH
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest differs"):
        _validate_native_vl_patch(changed)
