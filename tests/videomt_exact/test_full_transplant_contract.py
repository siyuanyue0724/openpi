from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from picf_next.videomt_exact.checkpoint import (
    PUBLISHED_BACKBONE_TENSORS,
    PUBLISHED_CHECKPOINT_BYTES,
    PUBLISHED_CHECKPOINT_SHA256,
    PUBLISHED_MODEL_NUMEL,
    PUBLISHED_MODEL_TENSORS,
)
from tools.audit_full_transplant_contract import (
    _validate_videomt_manifest,
    audit_full_transplant_contract,
)
from tools.bootstrap_lingbot_vla2 import (
    CHECKPOINT_ASSET_CONTRACT,
    LINGBOT_CHECKPOINT_ID,
    LINGBOT_CHECKPOINT_REVISION,
    PROCESSOR_ASSET_CONTRACT,
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
)
from tools.bootstrap_lingbot_vla2_native import (
    FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MUON_COLLECTIVE_HOTFIX_SHA256,
    PATCH_SHA256,
    PATCHED_ACTION_DECODER_SHA256,
    PATCHED_CHECKPOINTER_SHA256,
    PATCHED_MODEL_SHA256,
    PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
    PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256,
    PATCHED_QWEN25_TEXT_DECODER_SHA256,
    PATCHED_TEXT_DECODER_SHA256,
    SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "references/full_transplant_sources.json"
STAGE_PQ_LAUNCHER = ROOT / "adr199/run_full_transplant_stage_pq_2gpu.sh"


def _asset_map(rows: list[dict[str, object]]) -> dict[str, tuple[int, str]]:
    return {str(row["path"]): (int(row["bytes"]), str(row["sha256"])) for row in rows}


def test_repository_full_transplant_contract_is_fail_closed_and_valid() -> None:
    report = audit_full_transplant_contract(MANIFEST_PATH, root=ROOT)
    assert report["status"] == "passed"
    assert report["strict_runtime"] is False
    assert report["external_checks_complete"] is False
    assert report["repository"]["lingbot"]["source_file_count"] == 10
    assert report["repository"]["lingbot"]["patched_source_file_count"] == 7
    assert report["repository"]["videomt"]["byte-identical"] == 9
    assert report["repository"]["videomt"]["single-import-rewrite"] == 1
    assert report["repository"]["videomt"]["normative-reference"] == 6


def test_contract_matches_existing_lingbot_and_videomt_asset_ledgers() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    lingbot = manifest["donors"]["lingbot_vla2"]
    assert lingbot["source_commit"] == LINGBOT_NATIVE_SOURCE_COMMIT
    assert lingbot["checkpoint"]["repository"] == LINGBOT_CHECKPOINT_ID
    assert lingbot["checkpoint"]["revision"] == LINGBOT_CHECKPOINT_REVISION
    assert lingbot["processor"]["repository"] == QWEN_PROCESSOR_ID
    assert lingbot["processor"]["revision"] == QWEN_PROCESSOR_REVISION
    assert _asset_map(lingbot["checkpoint"]["assets"]) == CHECKPOINT_ASSET_CONTRACT
    assert _asset_map(lingbot["processor"]["assets"]) == PROCESSOR_ASSET_CONTRACT
    assert [item["sha256"] for item in lingbot["ordered_overlays"]] == [
        PATCH_SHA256,
        MUON_COLLECTIVE_HOTFIX_SHA256,
        SELECTIVE_FROZEN_VISION_OFFLOAD_SHA256,
        FROZEN_VISUAL_ROOT_OFFLOAD_SHA256,
    ]
    assert lingbot["patched_source_files"] == {
        "lingbotvla/checkpoint/checkpointer.py": PATCHED_CHECKPOINTER_SHA256,
        "lingbotvla/distributed/torch_parallelize.py": (
            PATCHED_PARALLEL_WITH_FROZEN_VISION_OFFLOAD_SHA256
        ),
        "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py": (PATCHED_MODEL_SHA256),
        "lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py": (PATCHED_ACTION_DECODER_SHA256),
        "lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py": (PATCHED_TEXT_DECODER_SHA256),
        "lingbotvla/models/vla/lingbot_vla/qwenvl_in_vla.py": (PATCHED_QWEN25_TEXT_DECODER_SHA256),
        "lingbotvla/optim/muon.py": PATCHED_MUON_WITH_COLLECTIVE_HOTFIX_SHA256,
    }

    videomt = manifest["donors"]["videomt"]["checkpoint"]
    assert videomt["bytes"] == PUBLISHED_CHECKPOINT_BYTES
    assert videomt["sha256"] == PUBLISHED_CHECKPOINT_SHA256
    assert videomt["model_tensors"] == PUBLISHED_MODEL_TENSORS
    assert videomt["backbone_tensors"] == PUBLISHED_BACKBONE_TENSORS
    assert videomt["model_numel"] == PUBLISHED_MODEL_NUMEL


def test_strict_runtime_rejects_missing_external_assets() -> None:
    with pytest.raises(ValueError, match="strict runtime audit requires"):
        audit_full_transplant_contract(MANIFEST_PATH, root=ROOT, strict_runtime=True)


def test_stage_pq_launcher_requires_the_complete_strict_runtime_audit() -> None:
    text = STAGE_PQ_LAUNCHER.read_text(encoding="utf-8")
    assert "tools/audit_full_transplant_contract.py" in text
    for argument in (
        "--lingbot-repository",
        "--prepared-lingbot-checkout",
        "--lingbot-checkpoint-dir",
        "--processor-dir",
        "--videomt-checkpoint",
        "--dinov3-bundle",
        "--strict-runtime",
        "--json-out",
    ):
        assert argument in text
    assert "${RUN_DIR}.full-transplant-audit.json" in text
    assert text.index("--strict-runtime") < text.index("run_implicit_multimodal_anchor_2gpu.sh")


def test_videomt_audit_rejects_unapproved_vendor_change(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    donor = manifest["donors"]["videomt"]
    snapshot = tmp_path / "snapshot"
    vendor = tmp_path / "vendor"
    shutil.copytree(ROOT / donor["snapshot_root"], snapshot)
    shutil.copytree(ROOT / donor["vendor_root"], vendor)
    donor["snapshot_root"] = "snapshot"
    donor["vendor_root"] = "vendor"
    target = vendor / "modeling/backbone/videomt.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# unauthorized\n")

    with pytest.raises(ValueError, match="not byte-identical"):
        _validate_videomt_manifest(donor, root=tmp_path)


def test_videomt_audit_rejects_second_criterion_adapter(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    donor = manifest["donors"]["videomt"]
    snapshot = tmp_path / "snapshot"
    vendor = tmp_path / "vendor"
    shutil.copytree(ROOT / donor["snapshot_root"], snapshot)
    shutil.copytree(ROOT / donor["vendor_root"], vendor)
    donor["snapshot_root"] = "snapshot"
    donor["vendor_root"] = "vendor"
    target = vendor / "criterion_videomt.py"
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace("class VideoSetCriterion", "class SimplifiedCriterion"))

    with pytest.raises(ValueError, match="differs beyond the approved import rewrite"):
        _validate_videomt_manifest(donor, root=tmp_path)
