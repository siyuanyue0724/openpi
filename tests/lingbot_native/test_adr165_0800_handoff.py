from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from stat import S_IXUSR

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "adr165/prepare_0800_persistent_handoff.sh"
HANDOFF_SCHEMA = "picf-next.adr165-0800-persistent-handoff.v1"


def _run(*arguments: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *(str(value) for value in arguments)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PICF_PYTHON_BIN": sys.executable},
    )


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def _make_clean_repo(root: Path) -> tuple[Path, str]:
    repo = root / "source"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "ADR165 Test")
    _git(repo, "config", "user.email", "adr165@example.invalid")
    (repo / "tracked.txt").write_text("persistent source\n", encoding="ascii")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    head = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "branch", "retained-branch")
    _git(repo, "tag", "retained-tag")
    return repo.resolve(), head


@contextmanager
def _persistent_test_root() -> Iterator[Path]:
    roots = [Path("/mnt")]
    if Path("/mnt").is_dir():
        roots.extend(path for path in sorted(Path("/mnt").iterdir()) if path.is_dir())
    for root in roots:
        if not os.access(root, os.W_OK):
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="adr165-0800-test-", dir=root) as value:
                yield Path(value).resolve()
                return
        except OSError:
            continue
    pytest.skip("functional ADR165 handoff tests require a writable directory below /mnt")


def _make_evidence(root: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = root / "checkpoint-model-only"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata\n")
    (model / "__0_0.distcp").write_bytes(b"checkpoint shard must remain in place\n")
    checkpoint_manifest = checkpoint / "ltop_g3_training_checkpoint.json"
    _write_json(
        checkpoint_manifest,
        {
            "schema": "picf-next.ltop-g3-training-checkpoint.v2",
            "status": "PASS",
            "format": "lingbot-fsdp2-dcp-model-only",
        },
    )

    trial = root / "trial"
    trial_report = trial / "ltop_g3_mediator_trial_training_report.json"
    _write_json(
        trial_report,
        {
            "schema": "picf-next.ltop-g3-training-phase.v1",
            "status": "PASS",
            "phase": "training",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "format": "lingbot-fsdp2-dcp-model-only",
                "optimizer_saved": False,
                "manifest_sha256": _sha256(checkpoint_manifest),
                "model_tree_schema": "picf-next.ltop-g3-model-dcp-tree.v1",
                "model_tree_sha256": "a" * 64,
            },
        },
    )
    action = root / "action.json"
    retention = root / "retention.json"
    _write_json(
        action,
        {"schema": "picf-next.ltop-g3-evaluation-phase.v1", "status": "PASS"},
    )
    _write_json(
        retention,
        {"schema": "picf-next.ltop-g3-representation-retention.v1", "status": "PASS"},
    )
    return trial.resolve(), action.resolve(), retention.resolve(), checkpoint.resolve()


def _verify_checksums(handoff: Path) -> None:
    lines = (handoff / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert lines
    for line in lines:
        digest, relative = line.split("  ", maxsplit=1)
        assert _sha256(handoff / relative) == digest


def test_handoff_script_has_fail_closed_atomic_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert 'git(repo, "status", "--porcelain=v1", "--untracked-files=all")' in source
    assert 'git(repo, "bundle", "create", str(bundle_path), "HEAD", "--all")' in source
    assert 'git(repo, "bundle", "verify", str(bundle_path))' in source
    assert "RENAME_NOREPLACE = 1" in source
    assert "renameat2" in source
    assert "os.path.lexists(target)" in source
    assert "not target.is_relative_to(PERSISTENT_ROOT)" in source
    assert '"copied": False' in source
    assert "checkpoint already persists under /mnt" in source
    assert "SHA256SUMS" in source
    assert "evidence_manifest.json" in source
    assert "source_status.txt" in source
    assert "repository.bundle" in source
    assert "shutil" not in source
    assert "rm -" not in source


def test_handoff_publishes_offline_bundle_and_evidence_without_checkpoint_copy(
    tmp_path: Path,
) -> None:
    repo, head = _make_clean_repo(tmp_path)
    with _persistent_test_root() as persistent:
        trial, action, retention, checkpoint = _make_evidence(persistent)
        handoff = persistent / "handoff"
        result = _run(repo, handoff, trial, action, retention)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(handoff)
        assert handoff.is_dir() and not handoff.is_symlink()
        assert {path.name for path in handoff.iterdir()} == {
            "HEAD",
            "README.md",
            "SHA256SUMS",
            "evidence_manifest.json",
            "refs.txt",
            "repository.bundle",
            "source_status.txt",
        }
        assert (handoff / "HEAD").read_text(encoding="ascii").strip() == head
        assert "clean=true" in (handoff / "source_status.txt").read_text(encoding="utf-8")
        _verify_checksums(handoff)

        bundle_heads = _git(repo, "bundle", "list-heads", str(handoff / "repository.bundle"))
        assert f"{head} HEAD" in bundle_heads
        assert "refs/heads/main" in bundle_heads
        assert "refs/heads/retained-branch" in bundle_heads
        assert "refs/tags/retained-tag" in bundle_heads
        _git(repo, "bundle", "verify", str(handoff / "repository.bundle"))
        verifier = tmp_path / "bundle-verifier.git"
        verifier.mkdir()
        _git(verifier, "init", "--bare")
        _git(verifier, "bundle", "verify", str(handoff / "repository.bundle"))
        recovered = tmp_path / "recovered"
        subprocess.run(
            ["git", "clone", str(handoff / "repository.bundle"), str(recovered)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert _git(recovered, "rev-parse", "HEAD").strip() == head

        manifest = json.loads((handoff / "evidence_manifest.json").read_text(encoding="ascii"))
        assert manifest["schema"] == HANDOFF_SCHEMA
        assert manifest["status"] == "PASS"
        assert manifest["repository"]["head_sha"] == head
        assert manifest["repository"]["remote_access_required"] is False
        assert manifest["checkpoint_copy_policy"]["copied"] is False
        trial_checkpoint = manifest["evidence"]["trial"]["checkpoint"]
        assert trial_checkpoint["path"] == str(checkpoint)
        assert trial_checkpoint["copied"] is False
        assert trial_checkpoint["manifest"]["sha256"] == _sha256(
            checkpoint / "ltop_g3_training_checkpoint.json"
        )
        assert not list(handoff.rglob("*.distcp"))
        assert (checkpoint / "model" / "__0_0.distcp").is_file()


def test_handoff_allows_all_evidence_inputs_to_be_omitted(tmp_path: Path) -> None:
    repo, head = _make_clean_repo(tmp_path)
    with _persistent_test_root() as persistent:
        handoff = persistent / "handoff-no-evidence"
        result = _run(repo, handoff)

        assert result.returncode == 0, result.stderr
        manifest = json.loads((handoff / "evidence_manifest.json").read_text(encoding="ascii"))
        assert manifest["repository"]["head_sha"] == head
        assert manifest["evidence"] == {"trial": None, "action": None, "retention": None}


def test_handoff_rejects_dirty_repository_before_staging(tmp_path: Path) -> None:
    repo, _ = _make_clean_repo(tmp_path)
    (repo / "untracked.txt").write_text("dirty\n", encoding="ascii")
    with _persistent_test_root() as persistent:
        handoff = persistent / "dirty-handoff"
        result = _run(repo, handoff)

        assert result.returncode != 0
        assert "clean repo root is dirty" in result.stderr
        assert not os.path.lexists(handoff)
        assert not list(persistent.glob(f".{handoff.name}.staging-*"))


def test_handoff_rejects_existing_symlink_and_non_mnt_destinations(tmp_path: Path) -> None:
    repo, _ = _make_clean_repo(tmp_path)
    with _persistent_test_root() as persistent:
        existing = persistent / "existing"
        existing.mkdir()
        existing_result = _run(repo, existing)
        assert existing_result.returncode != 0
        assert "already exists" in existing_result.stderr

        symlink = persistent / "symlink"
        symlink.symlink_to(existing, target_is_directory=True)
        symlink_result = _run(repo, symlink)
        assert symlink_result.returncode != 0
        assert "already exists" in symlink_result.stderr

    non_mnt = tmp_path / "non-mnt-handoff"
    non_mnt_result = _run(repo, non_mnt)
    assert non_mnt_result.returncode != 0
    assert "below /mnt" in non_mnt_result.stderr
    assert not non_mnt.exists()


def test_handoff_rejects_symlink_evidence_without_publication(tmp_path: Path) -> None:
    repo, _ = _make_clean_repo(tmp_path)
    with _persistent_test_root() as persistent:
        real_action = persistent / "real-action.json"
        _write_json(real_action, {"schema": "example", "status": "PASS"})
        linked_action = persistent / "linked-action.json"
        linked_action.symlink_to(real_action)
        handoff = persistent / "symlink-evidence-handoff"
        result = _run(repo, handoff, "-", linked_action, "-")

        assert result.returncode != 0
        assert "must not be a symbolic link" in result.stderr
        assert not os.path.lexists(handoff)


def test_handoff_launcher_is_executable() -> None:
    assert SCRIPT.stat().st_mode & S_IXUSR
