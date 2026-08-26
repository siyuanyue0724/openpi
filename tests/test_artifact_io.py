from __future__ import annotations

from pathlib import Path

import pytest

import picf_next.artifact_io as artifact_io
from picf_next.artifact_io import (
    directory_tree_sha256,
    publish_prepared_directory_durable_exclusive,
    publish_prepared_file_durable_exclusive,
    write_text_durable_exclusive,
)


def test_directory_tree_digest_binds_paths_bytes_and_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    model = root / "model"
    model.mkdir(parents=True)
    shard = model / "rank.distcp"
    shard.write_bytes(b"first")
    first = directory_tree_sha256(root, schema="checkpoint.v1")
    assert first == directory_tree_sha256(root, schema="checkpoint.v1")
    assert first != directory_tree_sha256(root, schema="checkpoint.v2")

    shard.write_bytes(b"second")
    assert first != directory_tree_sha256(root, schema="checkpoint.v1")
    shard.rename(model / "renamed.distcp")
    renamed = directory_tree_sha256(root, schema="checkpoint.v1")
    assert renamed != first

    (model / "indirect").symlink_to(model / "renamed.distcp")
    with pytest.raises(ValueError, match="symbolic link"):
        directory_tree_sha256(root, schema="checkpoint.v1")


def test_durable_text_publication_is_exclusive_and_cleans_staging(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"
    write_text_durable_exclusive(output, '{"status":"PASS"}\n')
    assert output.read_text() == '{"status":"PASS"}\n'
    assert not tuple(output.parent.glob(".*.tmp"))
    assert not tuple(output.parent.glob("*.publish-lock"))

    with pytest.raises(FileExistsError):
        write_text_durable_exclusive(output, '{"status":"REPLACED"}\n')
    assert output.read_text() == '{"status":"PASS"}\n'


def test_durable_publication_rejects_symlink_and_stale_lock(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.write_text("original\n")
    link = tmp_path / "report"
    link.symlink_to(external)
    with pytest.raises(FileExistsError):
        write_text_durable_exclusive(link, "replacement\n")
    assert external.read_text() == "original\n"

    destination = tmp_path / "locked"
    lock = tmp_path / ".locked.publish-lock"
    lock.mkdir()
    with pytest.raises(FileExistsError, match="publish-lock"):
        write_text_durable_exclusive(destination, "blocked\n")
    assert not destination.exists()


def test_prepared_publication_requires_same_directory(tmp_path: Path) -> None:
    staging = tmp_path / "stage" / "value.tmp"
    staging.parent.mkdir()
    staging.write_bytes(b"value")
    with pytest.raises(ValueError, match="same-directory"):
        publish_prepared_file_durable_exclusive(staging, tmp_path / "value")


def test_replace_failure_removes_staging_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(artifact_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        write_text_durable_exclusive(output, "payload\n")
    assert not output.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob("*.publish-lock"))


def test_prepared_directory_publication_is_durable_and_exclusive(tmp_path: Path) -> None:
    staging = tmp_path / ".evidence.partial"
    staging.mkdir()
    (staging / "receipt.json").write_text('{"status":"PASS"}\n')
    output = tmp_path / "evidence"

    publish_prepared_directory_durable_exclusive(staging, output)

    assert not staging.exists()
    assert (output / "receipt.json").read_text() == '{"status":"PASS"}\n'
    assert not (tmp_path / ".evidence.publish-lock").exists()
    replacement = tmp_path / ".replacement.partial"
    replacement.mkdir()
    with pytest.raises(FileExistsError):
        publish_prepared_directory_durable_exclusive(replacement, output)
    assert replacement.is_dir()


def test_directory_replace_failure_removes_staging_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".evidence.partial"
    staging.mkdir()
    (staging / "receipt.json").write_text("payload\n")
    output = tmp_path / "evidence"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected directory replace failure")

    monkeypatch.setattr(artifact_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected directory"):
        publish_prepared_directory_durable_exclusive(staging, output)
    assert not staging.exists()
    assert not output.exists()
    assert not (tmp_path / ".evidence.publish-lock").exists()


def test_directory_post_replace_fsync_failure_rolls_back_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".evidence.partial"
    staging.mkdir()
    (staging / "receipt.json").write_text("payload\n")
    output = tmp_path / "evidence"
    original = artifact_io._fsync_directory  # noqa: SLF001
    calls = 0

    def fail_third_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected post-replace fsync failure")
        original(path)

    monkeypatch.setattr(artifact_io, "_fsync_directory", fail_third_fsync)
    with pytest.raises(OSError, match="post-replace"):
        publish_prepared_directory_durable_exclusive(staging, output)
    assert calls == 5
    assert not staging.exists()
    assert not output.exists()
    assert not (tmp_path / ".evidence.publish-lock").exists()
