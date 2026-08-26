from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.bootstrap_lingbot_vla2_unified import prepare_unified_source
from tools.verify_lingbot_vla2_unified_patch import (
    DATA_PATCH_RELATIVE_PATH,
    GRAPH_PATCH_RELATIVE_PATH,
    PATCHED_SOURCES,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "references/source_checkouts/lingbot-vla-v2"
DATA_PATCH = ROOT / DATA_PATCH_RELATIVE_PATH
GRAPH_PATCH = ROOT / GRAPH_PATCH_RELATIVE_PATH


@pytest.fixture
def local_source() -> Path:
    if not (SOURCE / ".git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    return SOURCE


def test_prepare_unified_source_is_exact_and_idempotent(
    tmp_path: Path,
    local_source: Path,
) -> None:
    checkout = tmp_path / "lingbot"
    first = prepare_unified_source(
        checkout=checkout,
        data_patch=DATA_PATCH,
        graph_patch=GRAPH_PATCH,
        source_url=str(local_source),
    )
    second = prepare_unified_source(
        checkout=checkout,
        data_patch=DATA_PATCH,
        graph_patch=GRAPH_PATCH,
        source_url=str(local_source),
    )
    assert first["patch_states"] == second["patch_states"] == ["applied", "applied"]
    assert first["patched_source_sha256"] == second["patched_source_sha256"]
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    assert {line[3:] for line in status.splitlines()} == {str(path) for path in PATCHED_SOURCES}
    model = (checkout / PATCHED_SOURCES[-1]).read_text()
    assert "set_unified_belief_graph" in model
    assert "action_layer_adapter" not in model

    with (checkout / PATCHED_SOURCES[-1]).open("a") as stream:
        stream.write("\n# undeclared local mutation\n")
    with pytest.raises(RuntimeError, match="source digests"):
        prepare_unified_source(
            checkout=checkout,
            data_patch=DATA_PATCH,
            graph_patch=GRAPH_PATCH,
        )


def test_prepare_unified_source_rejects_partial_or_unrelated_state(
    tmp_path: Path,
    local_source: Path,
) -> None:
    checkout = tmp_path / "lingbot"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(local_source), str(checkout)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "69729b4ef24c63ec25e750915491635f4753be1d"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "apply", str(DATA_PATCH)], cwd=checkout, check=True)
    with pytest.raises(ValueError, match="partially applied"):
        prepare_unified_source(
            checkout=checkout,
            data_patch=DATA_PATCH,
            graph_patch=GRAPH_PATCH,
        )

    subprocess.run(["git", "apply", str(GRAPH_PATCH)], cwd=checkout, check=True)
    (checkout / "README.md").write_text("unrelated\n")
    with pytest.raises(ValueError, match="unrelated changes"):
        prepare_unified_source(
            checkout=checkout,
            data_patch=DATA_PATCH,
            graph_patch=GRAPH_PATCH,
        )
