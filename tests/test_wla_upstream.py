from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from picf_next.wla_upstream import (
    WLA_COMMIT,
    WLA_CRITICAL_FILES,
    verify_wla_source,
)


def _source_root() -> Path:
    return Path(os.environ.get("PICF_WLA_SOURCE_ROOT", "/tmp/wla-upstream-20260826-v1"))


def _copy_critical_tree(source: Path, target: Path) -> None:
    for relative in WLA_CRITICAL_FILES:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)


def test_pinned_wla_source_verifies_when_checkout_is_available() -> None:
    root = _source_root()
    if not root.exists():
        pytest.skip("pinned WLA audit checkout is not available")
    receipt = verify_wla_source(root)
    assert receipt.commit == WLA_COMMIT
    assert dict(receipt.files) == WLA_CRITICAL_FILES


def test_pinned_wla_source_rejects_one_byte_change(tmp_path: Path) -> None:
    root = _source_root()
    if not root.exists():
        pytest.skip("pinned WLA audit checkout is not available")
    _copy_critical_tree(root, tmp_path)
    changed = tmp_path / "models/action_model/action_model.py"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_wla_source(tmp_path)


def test_pinned_wla_source_rejects_path_escape(tmp_path: Path) -> None:
    root = _source_root()
    if not root.exists():
        pytest.skip("pinned WLA audit checkout is not available")
    _copy_critical_tree(root, tmp_path)
    outside = tmp_path.parent / "outside-action-model.py"
    shutil.copy2(root / "models/action_model/action_model.py", outside)
    target = tmp_path / "models/action_model/action_model.py"
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        verify_wla_source(tmp_path)
