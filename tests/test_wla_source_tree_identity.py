from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from picf_next.wla_source_tree_identity import (
    WLA_PINNED_FILE_COUNT,
    WLA_PINNED_TOTAL_BYTES,
    WLA_PINNED_TREE_SHA256,
    WLA_UPSTREAM_COMMIT,
    WLASourceIdentityError,
    WLASourceTreeReceipt,
    build_wla_source_archive_receipt,
    build_wla_source_tree_receipt,
    load_wla_source_tree_receipt,
    verify_pinned_wla_source_archive,
    verify_pinned_wla_source_tree,
    verify_wla_source_archive_receipt,
    verify_wla_source_tree_receipt,
    wla_source_tree_receipt_bytes,
    write_pinned_wla_source_archive_receipt,
    write_wla_source_tree_receipt,
)

_PINNED_TREE = Path(
    os.environ.get("PICF_WLA_PINNED_TREE", "/tmp/wla-upstream-20260826-v1")
)
_TEST_COMMIT = "a" * 40


def _make_source_tree(root: Path) -> Path:
    root.mkdir()
    (root / "README.md").write_text("upstream\n", encoding="ascii")
    (root / "models").mkdir()
    (root / "models" / "z.py").write_text("Z = 1\n", encoding="ascii")
    (root / "models" / "a.py").write_text("A = 2\n", encoding="ascii")
    (root / "configs").mkdir()
    (root / "configs" / "train.yaml").write_text("steps: 10\n", encoding="ascii")
    return root


def _write_tar(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname="World_Language_Action-155ac94e")


def _write_zip(source: Path, destination: Path) -> None:
    prefix = Path("World_Language_Action-155ac94e")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, (prefix / path.relative_to(source)).as_posix())


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    return _make_source_tree(tmp_path / "source")


def _require_pinned_tree() -> Path:
    if not (_PINNED_TREE / ".git").exists() or not (_PINNED_TREE / "models").is_dir():
        pytest.skip(f"pinned WLA checkout is unavailable: {_PINNED_TREE}")
    return _PINNED_TREE


def _copy_pinned_tree(destination: Path) -> Path:
    source = _require_pinned_tree()
    return Path(shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git")))


def test_directory_receipt_is_deterministic_complete_and_sorted(source_tree: Path) -> None:
    first = build_wla_source_tree_receipt(
        source_tree, upstream_commit=_TEST_COMMIT, require_git_head=False
    )
    second = build_wla_source_tree_receipt(
        source_tree, upstream_commit=_TEST_COMMIT, require_git_head=False
    )

    assert first == second
    assert [item.path for item in first.files] == [
        "README.md",
        "configs/train.yaml",
        "models/a.py",
        "models/z.py",
    ]
    assert first.file_count == 4
    assert first.total_bytes == sum(
        path.stat().st_size for path in source_tree.rglob("*") if path.is_file()
    )


def test_tar_and_zip_receipts_equal_extracted_tree(source_tree: Path, tmp_path: Path) -> None:
    expected = build_wla_source_tree_receipt(
        source_tree, upstream_commit=_TEST_COMMIT, require_git_head=False
    )
    tar_path = tmp_path / "source.tar.gz"
    zip_path = tmp_path / "source.zip"
    _write_tar(source_tree, tar_path)
    _write_zip(source_tree, zip_path)

    assert build_wla_source_archive_receipt(tar_path, upstream_commit=_TEST_COMMIT) == expected
    assert build_wla_source_archive_receipt(zip_path, upstream_commit=_TEST_COMMIT) == expected
    assert verify_wla_source_archive_receipt(tar_path, expected) == expected
    assert verify_wla_source_archive_receipt(zip_path, expected) == expected


def test_tree_receipt_detects_add_change_and_delete(source_tree: Path) -> None:
    receipt = build_wla_source_tree_receipt(
        source_tree, upstream_commit=_TEST_COMMIT, require_git_head=False
    )

    (source_tree / "models" / "new.py").write_text("NEW = 1\n", encoding="ascii")
    with pytest.raises(WLASourceIdentityError, match="does not match"):
        verify_wla_source_tree_receipt(source_tree, receipt, require_git_head=False)
    (source_tree / "models" / "new.py").unlink()

    (source_tree / "models" / "a.py").write_text("A = 3\n", encoding="ascii")
    with pytest.raises(WLASourceIdentityError, match="does not match"):
        verify_wla_source_tree_receipt(source_tree, receipt, require_git_head=False)
    (source_tree / "models" / "a.py").write_text("A = 2\n", encoding="ascii")

    (source_tree / "README.md").unlink()
    with pytest.raises(WLASourceIdentityError, match="does not match"):
        verify_wla_source_tree_receipt(source_tree, receipt, require_git_head=False)


def test_directory_symlink_is_rejected(source_tree: Path) -> None:
    (source_tree / "models" / "alias.py").symlink_to(source_tree / "models" / "a.py")
    with pytest.raises(WLASourceIdentityError, match="symlink is forbidden"):
        build_wla_source_tree_receipt(
            source_tree, upstream_commit=_TEST_COMMIT, require_git_head=False
        )


def test_tar_symlink_is_rejected(source_tree: Path, tmp_path: Path) -> None:
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(source_tree, arcname="source")
        link = tarfile.TarInfo("source/alias.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "models/a.py"
        archive.addfile(link)

    with pytest.raises(WLASourceIdentityError, match="non-regular tar member"):
        build_wla_source_archive_receipt(archive_path, upstream_commit=_TEST_COMMIT)


def test_real_git_head_must_match_claimed_commit(source_tree: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is unavailable")
    subprocess.run(["git", "init", "-q", source_tree], check=True)
    subprocess.run(
        ["git", "-C", source_tree, "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", source_tree, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", source_tree, "add", "-A"], check=True)
    subprocess.run(["git", "-C", source_tree, "commit", "-qm", "fixture"], check=True)

    with pytest.raises(WLASourceIdentityError, match="Git HEAD mismatch"):
        build_wla_source_tree_receipt(source_tree, upstream_commit=_TEST_COMMIT)


def test_receipt_is_self_validating_and_written_without_replacement(
    source_tree: Path, tmp_path: Path
) -> None:
    receipt = build_wla_source_tree_receipt(
        source_tree, upstream_commit=_TEST_COMMIT, require_git_head=False
    )
    destination = tmp_path / "receipt.json"

    assert write_wla_source_tree_receipt(receipt, destination) == destination
    assert destination.read_bytes() == wla_source_tree_receipt_bytes(receipt)
    assert load_wla_source_tree_receipt(destination) == receipt
    assert stat_mode(destination) == 0o444
    with pytest.raises(FileExistsError):
        write_wla_source_tree_receipt(receipt, destination)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_receipt_tampering_is_rejected(source_tree: Path) -> None:
    receipt = build_wla_source_tree_receipt(
        source_tree, upstream_commit=_TEST_COMMIT, require_git_head=False
    )
    tampered = json.loads(wla_source_tree_receipt_bytes(receipt))
    tampered["files"][0]["sha256"] = "0" * 64

    with pytest.raises(WLASourceIdentityError, match="tree_sha256"):
        WLASourceTreeReceipt.from_mapping(tampered)


def test_pinned_checkout_matches_full_72_file_identity() -> None:
    source = _require_pinned_tree()
    receipt = verify_pinned_wla_source_tree(source)

    assert receipt.upstream_commit == WLA_UPSTREAM_COMMIT
    assert receipt.file_count == WLA_PINNED_FILE_COUNT == 72
    assert receipt.total_bytes == WLA_PINNED_TOTAL_BYTES == 617_184
    assert receipt.tree_sha256 == WLA_PINNED_TREE_SHA256
    assert "models/action_model/action_encoder.py" in {item.path for item in receipt.files}
    assert "models/transformer_encoder.py" in {item.path for item in receipt.files}


def test_pinned_archive_without_git_has_same_receipt(tmp_path: Path) -> None:
    source = _require_pinned_tree()
    archive_path = tmp_path / "wla-155ac94e.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or ".git" in path.relative_to(source).parts:
                continue
            archive.add(
                path,
                arcname=Path("World_Language_Action-155ac94e") / path.relative_to(source),
                recursive=False,
            )

    checkout_receipt = verify_pinned_wla_source_tree(source)
    archive_receipt = verify_pinned_wla_source_archive(archive_path)
    assert archive_receipt == checkout_receipt

    receipt_path = tmp_path / "wla-source-receipt.json"
    assert (
        write_pinned_wla_source_archive_receipt(archive_path, receipt_path)
        == checkout_receipt
    )
    assert load_wla_source_tree_receipt(receipt_path) == checkout_receipt


@pytest.mark.parametrize("mutation", ["unprotected-change", "addition", "deletion"])
def test_pinned_full_tree_rejects_files_outside_old_eight_file_allowlist(
    tmp_path: Path, mutation: str
) -> None:
    source = _copy_pinned_tree(tmp_path / "pinned")
    verify_pinned_wla_source_tree(source)

    if mutation == "unprotected-change":
        target = source / "models" / "action_model" / "action_encoder.py"
        target.write_bytes(target.read_bytes() + b"\n# drift\n")
    elif mutation == "addition":
        (source / "models" / "unregistered.py").write_text("DRIFT = True\n", encoding="ascii")
    else:
        (source / "models" / "transformer_encoder.py").unlink()

    with pytest.raises(WLASourceIdentityError, match="source identity mismatch"):
        verify_pinned_wla_source_tree(source)
