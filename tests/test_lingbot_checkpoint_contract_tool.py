from __future__ import annotations

from tools.audit_lingbot_checkpoint_contract import compare_state_keys


def test_checkpoint_contract_key_comparison_is_symmetric_and_fail_closed() -> None:
    exact = compare_state_keys({"a", "b"}, {"b", "a"})
    assert exact["exact_key_match"] is True
    assert exact["missing_checkpoint_keys"] == []
    assert exact["unexpected_checkpoint_keys"] == []

    mismatch = compare_state_keys({"a", "missing"}, {"a", "unexpected"})
    assert mismatch["exact_key_match"] is False
    assert mismatch["missing_checkpoint_keys"] == ["missing"]
    assert mismatch["unexpected_checkpoint_keys"] == ["unexpected"]
