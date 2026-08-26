from __future__ import annotations

from pathlib import Path

import pytest

from tools.verify_lingbot_vla2_unified_patch import (
    DATA_PATCH_RELATIVE_PATH,
    GRAPH_PATCH_RELATIVE_PATH,
    verify_unified_patches,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unified_patches_apply_to_immutable_pinned_source() -> None:
    if not (ROOT / "references/source_checkouts/lingbot-vla-v2/.git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    result = verify_unified_patches(root=ROOT)
    assert result["apply_checked"] is True
    assert result["attention_output_contract"] == "[batch,tokens,query_heads*head_dim]"
    assert result["patched_sources"] == [
        "lingbotvla/data/vla_data/base_dataset.py",
        "lingbotvla/data/vla_data/utils.py",
        "lingbotvla/distributed/torch_parallelize.py",
        "lingbotvla/checkpoint/checkpointer.py",
        "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py",
    ]
    assert set(result["patched_source_sha256"]) == set(result["patched_sources"])
    assert all(len(digest) == 64 for digest in result["patched_source_sha256"].values())


def test_unified_patch_verifier_accepts_a_persistent_prepared_checkout(
    tmp_path: Path,
) -> None:
    checkout = ROOT / "references/source_checkouts/lingbot-vla-v2-unified"
    if not (checkout / ".git").exists():
        pytest.skip("optional prepared LingBot source checkout is absent")
    for relative in (DATA_PATCH_RELATIVE_PATH, GRAPH_PATCH_RELATIVE_PATH):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    assert not (tmp_path / "references/source_checkouts/lingbot-vla-v2").exists()
    result = verify_unified_patches(root=tmp_path, checkout=checkout)
    assert result["apply_checked"] is True
    assert len(result["patched_source_sha256"]) == 5


def test_unified_patch_verifier_rejects_incomplete_graph(tmp_path: Path) -> None:
    data = tmp_path / DATA_PATCH_RELATIVE_PATH
    graph = tmp_path / GRAPH_PATCH_RELATIVE_PATH
    data.parent.mkdir(parents=True)
    data.write_text((ROOT / DATA_PATCH_RELATIVE_PATH).read_text())
    graph.write_text("self.unified_belief_graph = None\n")
    with pytest.raises(ValueError, match="omits required"):
        verify_unified_patches(root=tmp_path, check_apply=False)


def test_unified_patch_verifier_rejects_legacy_sidecar(tmp_path: Path) -> None:
    data = tmp_path / DATA_PATCH_RELATIVE_PATH
    graph = tmp_path / GRAPH_PATCH_RELATIVE_PATH
    data.parent.mkdir(parents=True)
    data.write_text((ROOT / DATA_PATCH_RELATIVE_PATH).read_text())
    graph.write_text(
        (ROOT / GRAPH_PATCH_RELATIVE_PATH).read_text() + "\nset_action_layer_adapter\n"
    )
    with pytest.raises(ValueError, match="forbidden legacy"):
        verify_unified_patches(root=tmp_path, check_apply=False)
