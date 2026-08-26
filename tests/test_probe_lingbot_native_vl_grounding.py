from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from tools.probe_lingbot_native_vl_grounding import (
    _validate_optional_qwen_restore,
    _validate_qwen_restore_load_result,
)


def test_optional_qwen_restore_requires_a_complete_pinned_pair(tmp_path: Path) -> None:
    _validate_optional_qwen_restore(None, None)
    _validate_optional_qwen_restore(tmp_path, "a" * 40)
    with pytest.raises(ContractError, match="provided together"):
        _validate_optional_qwen_restore(tmp_path, None)
    with pytest.raises(ContractError, match="lowercase Git commit"):
        _validate_optional_qwen_restore(tmp_path, "main")


def test_qwen_restore_accepts_only_the_tied_head_as_missing() -> None:
    result_type = namedtuple("LoadResult", ("missing_keys", "unexpected_keys"))
    assert _validate_qwen_restore_load_result(result_type(["lm_head.weight"], [])) == {
        "missing_keys": ["lm_head.weight"],
        "unexpected_keys": [],
    }
    with pytest.raises(ContractError, match="unexpected missing tensors"):
        _validate_qwen_restore_load_result(result_type(["model.visual.patch_embed.weight"], []))
    with pytest.raises(ContractError, match="unexpected tensors"):
        _validate_qwen_restore_load_result(result_type([], ["private_head.weight"]))
    with pytest.raises(ContractError, match="malformed"):
        _validate_qwen_restore_load_result(result_type([1], []))
