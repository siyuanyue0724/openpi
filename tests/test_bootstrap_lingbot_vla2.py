from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

import tools.bootstrap_lingbot_vla2 as bootstrap
from tools.bootstrap_lingbot_vla2 import (
    CHECKPOINT_ASSET_CONTRACT,
    LINGBOT_CHECKPOINT_ID,
    LINGBOT_CHECKPOINT_REVISION,
    LINGBOT_PATCHED_SOURCES,
    PROCESSOR_ASSET_CONTRACT,
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
    REQUIRED_CHECKPOINT_FILES,
    REQUIRED_PROCESSOR_FILES,
    checkpoint_download_command,
    prepare_source,
    processor_download_command,
    validate_checkpoint,
    validate_processor,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "references/source_checkouts/lingbot-vla-v2"
PATCH = ROOT / "references/patches/lingbot_vla2_action_layer_adapter.patch"


@pytest.fixture
def local_source() -> Path:
    if not (SOURCE / ".git").exists():
        pytest.skip("optional pinned LingBot source checkout is absent")
    return SOURCE


def test_prepare_source_applies_exact_patch_and_is_idempotent(
    tmp_path: Path, local_source: Path
) -> None:
    checkout = tmp_path / "lingbot"
    first = prepare_source(
        checkout=checkout,
        patch_path=PATCH,
        source_url=str(local_source),
    )
    second = prepare_source(
        checkout=checkout,
        patch_path=PATCH,
        source_url=str(local_source),
    )

    assert first["patch_state"] == second["patch_state"] == "applied"
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    assert {line[3:] for line in status.splitlines()} == {
        str(relative_path) for relative_path in LINGBOT_PATCHED_SOURCES
    }


def test_prepare_source_rejects_unrelated_changes(tmp_path: Path, local_source: Path) -> None:
    checkout = tmp_path / "lingbot"
    prepare_source(checkout=checkout, patch_path=PATCH, source_url=str(local_source))
    (checkout / "README.md").write_text("unrelated\n")
    with pytest.raises(ValueError, match="unrelated changes"):
        prepare_source(checkout=checkout, patch_path=PATCH, source_url=str(local_source))


def test_checkpoint_download_command_is_revision_pinned(tmp_path: Path) -> None:
    command = checkpoint_download_command(hf_command="hf", checkpoint_dir=tmp_path)
    assert command[:3] == ["hf", "download", LINGBOT_CHECKPOINT_ID]
    assert command[command.index("--revision") + 1] == LINGBOT_CHECKPOINT_REVISION
    assert "token" not in " ".join(command).lower()


def test_checkpoint_validation_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        validate_checkpoint(tmp_path)


def test_checkpoint_contract_covers_every_required_asset() -> None:
    assert set(CHECKPOINT_ASSET_CONTRACT) == set(REQUIRED_CHECKPOINT_FILES)


def test_checkpoint_validation_rejects_tampered_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"exact"
    contract = {"config.json": (len(payload), hashlib.sha256(payload).hexdigest())}
    monkeypatch.setattr(bootstrap, "REQUIRED_CHECKPOINT_FILES", ("config.json",))
    monkeypatch.setattr(bootstrap, "CHECKPOINT_ASSET_CONTRACT", contract)
    (tmp_path / "config.json").write_bytes(payload)
    report = validate_checkpoint(tmp_path)
    assert report["checkpoint_assets"] == [
        {
            "path": "config.json",
            "bytes": len(payload),
            "sha256": contract["config.json"][1],
        }
    ]
    (tmp_path / "config.json").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="digest differs"):
        validate_checkpoint(tmp_path)


def test_processor_download_is_revision_pinned_and_excludes_base_weights(
    tmp_path: Path,
) -> None:
    command = processor_download_command(hf_command="hf", processor_dir=tmp_path)
    assert command[:3] == ["hf", "download", QWEN_PROCESSOR_ID]
    assert command[command.index("--revision") + 1] == QWEN_PROCESSOR_REVISION
    assert command[command.index("--exclude") + 1] == "*.safetensors"
    assert "token" not in " ".join(command).lower()


def test_processor_validation_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="processor/config snapshot is incomplete"):
        validate_processor(tmp_path)


def test_processor_contract_covers_every_required_asset() -> None:
    assert set(PROCESSOR_ASSET_CONTRACT) == set(REQUIRED_PROCESSOR_FILES)


def test_processor_validation_rejects_wrong_asset_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"exact"
    contract = {"config.json": (len(payload), hashlib.sha256(payload).hexdigest())}
    monkeypatch.setattr(bootstrap, "REQUIRED_PROCESSOR_FILES", ("config.json",))
    monkeypatch.setattr(bootstrap, "PROCESSOR_ASSET_CONTRACT", contract)
    (tmp_path / "config.json").write_bytes(payload + b"!")
    with pytest.raises(ValueError, match="size differs"):
        validate_processor(tmp_path)
