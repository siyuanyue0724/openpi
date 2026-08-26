from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.bootstrap_molmoact2 import (
    EXPECTED_INDEX,
    METADATA_SHA256,
    MOLMO_CHECKPOINT_ID,
    MOLMO_CHECKPOINT_REVISION,
    MOLMO_LEROBOT_COMMIT,
    MOLMO_LEROBOT_MODEL_SOURCE,
    MOLMO_SOURCE_COMMIT,
    REQUIRED_CHECKPOINT_FILES,
    WEIGHT_SHARD_SHA256,
    checkpoint_download_command,
    picf_install_command,
    prepare_sources,
    runtime_sync_command,
    validate_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "references/source_checkouts/molmoact2"
TRAINER = ROOT / "references/source_checkouts/molmoact2-lerobot"


@pytest.fixture
def local_sources() -> tuple[Path, Path]:
    if not (PARENT / ".git").exists() or not (TRAINER / ".git").exists():
        pytest.skip("optional pinned MolmoAct2 source checkouts are absent")
    return PARENT, TRAINER


def test_prepare_sources_reproduces_parent_gitlink(
    tmp_path: Path, local_sources: tuple[Path, Path]
) -> None:
    parent, trainer = local_sources
    source_checkout = tmp_path / "molmoact2"
    trainer_checkout = tmp_path / "lerobot"
    first = prepare_sources(
        source_checkout=source_checkout,
        lerobot_checkout=trainer_checkout,
        source_url=str(parent),
        lerobot_url=str(trainer),
    )
    second = prepare_sources(
        source_checkout=source_checkout,
        lerobot_checkout=trainer_checkout,
        source_url=str(parent),
        lerobot_url=str(trainer),
    )

    assert first == second
    assert first["source_commit"] == MOLMO_SOURCE_COMMIT
    assert first["lerobot_commit"] == MOLMO_LEROBOT_COMMIT
    assert first["lerobot_patch_state"] == "applied"
    assert (
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=source_checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    trainer_status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=trainer_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    assert {line[3:] for line in trainer_status.splitlines()} == {str(MOLMO_LEROBOT_MODEL_SOURCE)}


def test_checkpoint_download_and_runtime_commands_are_frozen(tmp_path: Path) -> None:
    command = checkpoint_download_command(hf_command="hf", checkpoint_dir=tmp_path)
    assert command[:3] == ["hf", "download", MOLMO_CHECKPOINT_ID]
    assert command[command.index("--revision") + 1] == MOLMO_CHECKPOINT_REVISION
    assert runtime_sync_command() == [
        "uv",
        "sync",
        "--frozen",
        "--extra",
        "molmoact2",
        "--extra",
        "training",
    ]
    assert "token" not in " ".join(command).lower()
    install = picf_install_command(
        repo_root=tmp_path / "repo",
        lerobot_checkout=tmp_path / "lerobot",
    )
    assert "--no-deps" in install
    assert "scipy==1.17.1" in install
    assert "--editable" in install
    assert install[-1] == str((tmp_path / "repo").absolute())


def test_checkpoint_validation_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        validate_checkpoint(tmp_path)


def test_every_non_weight_checkpoint_asset_is_content_pinned() -> None:
    non_weight_files = {
        name for name in REQUIRED_CHECKPOINT_FILES if not name.endswith(".safetensors")
    }
    assert set(METADATA_SHA256) == non_weight_files


def test_checkpoint_validation_rejects_symlinked_assets(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"fixture")
    for name in REQUIRED_CHECKPOINT_FILES:
        (tmp_path / name).symlink_to(target)

    with pytest.raises(ValueError, match="regular non-symlink files"):
        validate_checkpoint(tmp_path)


def test_checkpoint_index_and_metadata_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import bootstrap_molmoact2 as bootstrap

    for name in bootstrap.REQUIRED_CHECKPOINT_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"fixture")
    index = {
        "metadata": {
            "total_parameters": EXPECTED_INDEX["total_parameters"],
            "total_size": EXPECTED_INDEX["total_size"],
        },
        "weight_map": {
            f"tensor.{index}": f"model-{(index % 5) + 1:05d}-of-00005.safetensors"
            for index in range(EXPECTED_INDEX["tensor_keys"])
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    monkeypatch.setattr(
        bootstrap,
        "_sha256",
        lambda path: METADATA_SHA256.get(
            path.name,
            WEIGHT_SHARD_SHA256.get(path.name, "fixture"),
        ),
    )
    report = validate_checkpoint(tmp_path)
    assert report["total_parameters"] == 5_485_309_424
    assert report["shards"] == 5
    assert report["weight_shard_sha256"] == WEIGHT_SHARD_SHA256


def test_checkpoint_validation_rejects_changed_remote_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import bootstrap_molmoact2 as bootstrap

    for name in bootstrap.REQUIRED_CHECKPOINT_FILES:
        (tmp_path / name).write_bytes(b"fixture")

    def changed_hash(path: Path) -> str:
        if path.name == "modeling_molmoact2.py":
            return "0" * 64
        return METADATA_SHA256.get(
            path.name,
            WEIGHT_SHARD_SHA256.get(path.name, "fixture"),
        )

    monkeypatch.setattr(bootstrap, "_sha256", changed_hash)
    with pytest.raises(ValueError, match="metadata hashes changed"):
        validate_checkpoint(tmp_path)


def test_checkpoint_validation_rejects_a_changed_weight_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import bootstrap_molmoact2 as bootstrap

    for name in bootstrap.REQUIRED_CHECKPOINT_FILES:
        (tmp_path / name).write_bytes(b"fixture")
    index = {
        "metadata": {
            "total_parameters": EXPECTED_INDEX["total_parameters"],
            "total_size": EXPECTED_INDEX["total_size"],
        },
        "weight_map": {
            f"tensor.{index}": f"model-{(index % 5) + 1:05d}-of-00005.safetensors"
            for index in range(EXPECTED_INDEX["tensor_keys"])
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    def changed_hash(path: Path) -> str:
        if path.name == "model-00003-of-00005.safetensors":
            return "0" * 64
        return METADATA_SHA256.get(
            path.name,
            WEIGHT_SHARD_SHA256.get(path.name, "fixture"),
        )

    monkeypatch.setattr(bootstrap, "_sha256", changed_hash)
    with pytest.raises(ValueError, match="weight hashes changed"):
        validate_checkpoint(tmp_path)
