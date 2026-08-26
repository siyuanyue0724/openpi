from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.diagnose_cuda_oom_snapshot import (
    _maximum_entries,
    _required_absolute_directory,
    _required_absolute_file,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/diagnose_cuda_oom_snapshot.py"


def test_cuda_oom_snapshot_tool_has_bounded_diagnostic_contract() -> None:
    source = TOOL.read_text(encoding="utf-8")
    ast.parse(source)
    for fragment in (
        'runpy.run_path(str(target), run_name="__main__")',
        "original_backward = torch.Tensor.backward",
        "def backward_with_snapshot(self, *args, **kwargs):",
        "except torch.OutOfMemoryError:",
        "torch.cuda.memory._record_memory_history(",
        "torch.cuda.memory._dump_snapshot(str(output))",
        "torch.cuda.memory._record_memory_history(enabled=None)",
        "torch.Tensor.backward = original_backward",
    ):
        assert fragment in source

    history = source.index("torch.cuda.memory._record_memory_history(")
    run_target = source.index('runpy.run_path(str(target), run_name="__main__")')
    assert history < run_target


def test_cuda_oom_snapshot_paths_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    output = tmp_path / "snapshots"
    monkeypatch.setenv("TARGET", str(target))
    monkeypatch.setenv("OUTPUT", str(output))
    assert _required_absolute_file("TARGET") == target.resolve()
    assert _required_absolute_directory("OUTPUT") == output.resolve()
    assert output.is_dir()

    monkeypatch.setenv("TARGET", "relative.py")
    with pytest.raises(RuntimeError, match="existing absolute file"):
        _required_absolute_file("TARGET")
    monkeypatch.setenv("OUTPUT", "relative")
    with pytest.raises(RuntimeError, match="must be absolute"):
        _required_absolute_directory("OUTPUT")


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_cuda_oom_snapshot_history_bound_is_positive(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PICF_CUDA_OOM_MAX_ENTRIES", value)
    with pytest.raises(RuntimeError, match="must be"):
        _maximum_entries()


def test_cuda_oom_snapshot_history_bound_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PICF_CUDA_OOM_MAX_ENTRIES", raising=False)
    assert _maximum_entries() == 200000
