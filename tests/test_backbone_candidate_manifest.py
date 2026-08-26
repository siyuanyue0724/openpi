from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.audit_backbone_candidates import audit_candidates
from tools.bootstrap_lingbot_vla2_native import (
    CHECKOUT_RELATIVE_PATH,
    LINGBOT_NATIVE_SOURCE_COMMIT,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "references" / "backbone_candidates.json"


def _write_manifest(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "backbones.json"
    path.write_text(json.dumps(data))
    return path


def _require_checkouts(*relative_paths: str) -> None:
    missing = [path for path in relative_paths if not (ROOT / path / ".git").exists()]
    if missing:
        pytest.skip(f"optional pinned source checkouts are absent: {missing}")


def test_pinned_backbone_decision_is_valid_without_optional_checkouts() -> None:
    result = audit_candidates(MANIFEST, root=ROOT, check_checkouts=False)
    assert result == {
        "candidates": 12,
        "causal_host": "LingBot-VLA2",
        "deployment_host": "LingBot-VLA2",
        "primary_unified_host": "LingBot-VLA2",
        "architecture_reference": "Qwen-RobotManip",
    }


def test_all_local_candidate_checkouts_match_the_pinned_manifest_when_present() -> None:
    candidates = json.loads(MANIFEST.read_text())["candidates"]
    _require_checkouts(*(candidate["checkout"] for candidate in candidates))

    result = audit_candidates(MANIFEST, root=ROOT, check_checkouts=True)

    assert result["candidates"] == 12


def test_lingbot_is_the_single_primary_causal_and_deployment_host() -> None:
    data = json.loads(MANIFEST.read_text())
    lingbot = next(item for item in data["candidates"] if item["name"] == "LingBot-VLA2")

    assert data["decision"]["primary_unified_host"] == lingbot["name"]
    assert data["decision"]["deployment_host"] == lingbot["name"]
    assert data["decision"]["causal_host"] == lingbot["name"]
    assert lingbot["weights"] == "public"
    assert lingbot["code_scope"] == "post_training"
    assert "primary_unified_host" in lingbot["roles"]
    assert "causal_host" in lingbot["roles"]


def test_lingbot_manifests_follow_the_frozen_native_production_source() -> None:
    candidates = json.loads(MANIFEST.read_text())["candidates"]
    upstream = json.loads((ROOT / "references" / "upstream_sources.json").read_text())["sources"]
    candidate = next(item for item in candidates if item["name"] == "LingBot-VLA2")
    source = next(item for item in upstream if item["name"] == "LingBot-VLA2")

    expected_checkout = str(CHECKOUT_RELATIVE_PATH)
    assert candidate["commit"] == source["commit"] == LINGBOT_NATIVE_SOURCE_COMMIT
    assert candidate["checkout"] == source["checkout"] == expected_checkout
    assert source["status"] == "substantial-adaptation"
    assert "references/patches/lingbot_vla2_picf_native.patch" in source["local_paths"]


def test_cosmos_edge_is_not_an_active_backbone_candidate() -> None:
    data = json.loads(MANIFEST.read_text())
    names = {item["name"] for item in data["candidates"]}

    assert "NVIDIA-Cosmos-Framework" not in names


def test_qwen_robotmanip_is_a_no_weight_architecture_reference() -> None:
    data = json.loads(MANIFEST.read_text())
    qwen = next(item for item in data["candidates"] if item["name"] == "Qwen-RobotManip")

    assert data["decision"]["architecture_reference"] == qwen["name"]
    assert qwen["weights"] == "absent"
    assert qwen["code_scope"] == "announcement_only"
    assert qwen["runtime_deployment"] is False


def test_falcon_is_spatial_reference_not_a_deep_action_host() -> None:
    candidates = json.loads(MANIFEST.read_text())["candidates"]
    falcon = next(item for item in candidates if item["name"] == "FALCON")

    assert falcon["weights"] == "public"
    assert falcon["code_scope"] == "full_training"
    assert falcon["public_control_dataset"] is True
    assert falcon["deep_action_context"] is False
    assert "spatial_prior_reference" in falcon["roles"]


def test_candidates_shared_with_upstream_manifest_have_identical_source_identity() -> None:
    candidates = json.loads(MANIFEST.read_text())["candidates"]
    upstream = json.loads((ROOT / "references" / "upstream_sources.json").read_text())["sources"]
    upstream_by_name = {item["name"]: item for item in upstream}

    for candidate in candidates:
        source = upstream_by_name.get(candidate["name"])
        if source is None:
            continue
        assert source["commit"] == candidate["commit"]
        assert source["checkout"] == candidate["checkout"]
        assert (
            source["url"].removesuffix(".git").lower()
            == candidate["official_repo"].removesuffix(".git").lower()
        )


def test_announcement_without_weights_cannot_be_selected_as_causal_host(tmp_path: Path) -> None:
    data = json.loads(MANIFEST.read_text())
    data["decision"]["causal_host"] = "Qwen-VLA"
    with pytest.raises(ValueError, match="public weights and complete post-training code"):
        audit_candidates(_write_manifest(tmp_path, data), root=ROOT, check_checkouts=False)


def test_private_foundation_summary_does_not_disqualify_deployment_host(
    tmp_path: Path,
) -> None:
    data = json.loads(MANIFEST.read_text())
    edited = deepcopy(data)
    lingbot = next(item for item in edited["candidates"] if item["name"] == "LingBot-VLA2")
    lingbot["foundation_data"] = "private_summary"
    result = audit_candidates(_write_manifest(tmp_path, edited), root=ROOT, check_checkouts=False)
    assert result["causal_host"] == "LingBot-VLA2"
    assert result["deployment_host"] == "LingBot-VLA2"


def test_causal_host_requires_a_reproducible_control_dataset(tmp_path: Path) -> None:
    data = json.loads(MANIFEST.read_text())
    causal = next(item for item in data["candidates"] if item["name"] == "LingBot-VLA2")
    causal["public_control_dataset"] = False
    with pytest.raises(ValueError, match="reproducible controlled-data route"):
        audit_candidates(_write_manifest(tmp_path, data), root=ROOT, check_checkouts=False)


def test_deployment_host_must_have_cross_embodiment_schema(tmp_path: Path) -> None:
    data = json.loads(MANIFEST.read_text())
    deployment = next(item for item in data["candidates"] if item["name"] == "LingBot-VLA2")
    deployment["cross_embodiment_action_schema"] = False
    with pytest.raises(ValueError, match="deployment property"):
        audit_candidates(_write_manifest(tmp_path, data), root=ROOT, check_checkouts=False)


def test_pinned_evidence_path_must_exist(tmp_path: Path) -> None:
    _require_checkouts("references/source_checkouts/wla")
    data = json.loads(MANIFEST.read_text())
    data["candidates"][0]["evidence_symbols"] = ["missing.py::Missing.forward"]
    with pytest.raises(ValueError, match="missing evidence file"):
        audit_candidates(_write_manifest(tmp_path, data), root=ROOT, check_checkouts=True)


def test_molmoact2_training_submodule_must_match_parent_gitlink(tmp_path: Path) -> None:
    _require_checkouts(
        "references/source_checkouts/molmoact2",
        "references/source_checkouts/molmoact2-lerobot",
    )
    data = json.loads(MANIFEST.read_text())
    molmo = next(item for item in data["candidates"] if item["name"] == "MolmoAct2")
    molmo["source_submodule"]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="parent pins submodule"):
        audit_candidates(_write_manifest(tmp_path, data), root=ROOT, check_checkouts=True)
