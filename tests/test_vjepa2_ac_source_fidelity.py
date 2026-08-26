from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARCHIVE = (
    PROJECT_ROOT
    / "references/source_archives/"
    "vjepa2-204698b45b3712590f06245fbfba32d3be539812.tar.gz"
)
VENDORED_SOURCE = PROJECT_ROOT / "src/picf_next/_vendor/vjepa2_ac"
SOURCE_ARCHIVE_SHA256 = "53630196f62950038acf8c9debdc64d7e242b71c68f507b5ad716c795a617d53"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def test_complete_vjepa2_source_archive_has_frozen_identity() -> None:
    assert _sha256_path(SOURCE_ARCHIVE) == SOURCE_ARCHIVE_SHA256


def _restore_upstream_import_root(value: str) -> str:
    return value.replace("picf_next._vendor.vjepa2_ac.", "src.")


def test_arm_b_file_classification_binds_every_transplant_and_adapter() -> None:
    classification = json.loads(
        (PROJECT_ROOT / "adr208/arm_b_file_classification.json").read_text()
    )
    assert classification["counts"] == {
        "import_root_only": 33,
        "picf_adapter_or_falsification_harness": 9,
    }
    for record in classification["transplanted_files"]:
        assert record["classification"] == "import_root_only"
        assert _sha256_path(PROJECT_ROOT / record["deployed_path"]) == record[
            "deployed_sha256"
        ]
        assert record["semantic_restore_sha256"] == record["upstream_sha256"]
    for record in classification["picf_files"]:
        assert record["classification"] == "picf_adapter_or_falsification_harness"
        assert _sha256_path(PROJECT_ROOT / record["path"]) == record["sha256"]


def test_complete_vjepa2_source_tree_is_import_only_transplanted() -> None:
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        source_members = {
            member.name: member
            for member in archive.getmembers()
            if member.isfile() and member.name.startswith("src/") and member.name.endswith(".py")
        }
        vendored_paths = tuple(sorted(VENDORED_SOURCE.rglob("*.py")))
        assert len(source_members) == len(vendored_paths) == 32
        assert {path.relative_to(VENDORED_SOURCE).as_posix() for path in vendored_paths} == {
            name.removeprefix("src/") for name in source_members
        }
        for source_name, member in source_members.items():
            source_stream = archive.extractfile(member)
            assert source_stream is not None
            source_text = source_stream.read().decode("utf-8")
            vendored_text = (VENDORED_SOURCE / source_name.removeprefix("src/")).read_text()
            assert _restore_upstream_import_root(vendored_text) == source_text


def test_official_vjepa2_ac_transform_is_import_only_transplanted() -> None:
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        source_stream = archive.extractfile("app/vjepa_droid/transforms.py")
        assert source_stream is not None
        source_text = source_stream.read().decode("utf-8")
    vendored = (
        PROJECT_ROOT / "src/picf_next/_vendor/vjepa2_ac_app/transforms.py"
    ).read_text()
    restored = vendored.replace(
        "import picf_next._vendor.vjepa2_ac.datasets.utils.video.transforms as video_transforms",
        "import src.datasets.utils.video.transforms as video_transforms",
    ).replace(
        "from picf_next._vendor.vjepa2_ac.datasets.utils.video.randerase",
        "from src.datasets.utils.video.randerase",
    )
    assert restored == source_text


def test_adr208_manifest_binds_the_complete_source_freeze_and_staged_approval() -> None:
    manifest_path = PROJECT_ROOT / "adr208/vjepa2_ac_exact_source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["upstream"]["commit"] == "204698b45b3712590f06245fbfba32d3be539812"
    assert manifest["upstream"]["checkpoint_http_content_length"] == 11_760_743_310
    assert manifest["upstream"]["checkpoint_sha256"] == (
        "0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b"
    )
    assert manifest["upstream"]["checkpoint_state"] == "downloaded_and_locally_verified"
    assert manifest["source_freeze"]["sha256"] == SOURCE_ARCHIVE_SHA256
    assert manifest["implementation_gate"] == {
        "approved": True,
        "approval_received_at_cst": "2026-08-23",
        "may_download_checkpoint": True,
        "may_implement_exact_arm_b": True,
        "may_implement_labeled_arm_c_after_arm_b_gates": True,
        "may_start_long_training": False,
        "required_next_decision": "pass_arm_b_causal_gates_then_pass_arm_c_strict_100_step_gates",
    }
