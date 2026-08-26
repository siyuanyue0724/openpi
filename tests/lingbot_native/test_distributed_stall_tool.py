from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.diagnose_distributed_stall import _required_absolute_file, _timeout_seconds

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/diagnose_distributed_stall.py"


def test_distributed_stall_tool_has_bounded_fail_closed_contract() -> None:
    source = TOOL.read_text(encoding="utf-8")
    ast.parse(source)
    for fragment in (
        '"TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"',
        '"TORCH_NCCL_DESYNC_DEBUG": "1"',
        '"TORCH_NCCL_DUMP_ON_TIMEOUT": "1"',
        '"TORCH_NCCL_ENABLE_TIMING": "1"',
        '"TORCH_NCCL_TRACE_BUFFER_SIZE": "2000"',
        '"TORCH_NCCL_TRACE_CPP_STACK": "1"',
        '"TORCH_FR_BUFFER_SIZE": "2000"',
        '"TORCH_FR_CPP_STACK": "1"',
        "if conflicts:",
        'raise RuntimeError("target already declares a process-group timeout")',
        "dist.init_process_group = init_process_group_with_timeout",
        'runpy.run_path(str(target), run_name="__main__")',
        "dist.init_process_group = original_init_process_group",
    ):
        assert fragment in source


def test_distributed_stall_target_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    monkeypatch.setenv("TARGET", str(target))
    assert _required_absolute_file("TARGET") == target.resolve()

    monkeypatch.setenv("TARGET", "relative.py")
    with pytest.raises(RuntimeError, match="existing absolute file"):
        _required_absolute_file("TARGET")


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_distributed_stall_timeout_is_positive(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS", value)
    with pytest.raises(RuntimeError, match="must be"):
        _timeout_seconds()


def test_distributed_stall_timeout_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS", raising=False)
    assert _timeout_seconds() == 90
