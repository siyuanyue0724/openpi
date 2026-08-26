#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 CLEAN_REPO_ROOT ABSENT_HANDOFF_ROOT [/ABS/TRIAL_EVIDENCE|-] [/ABS/ACTION_EVIDENCE|-] [/ABS/RETENTION_EVIDENCE|-]" >&2
  exit 2
fi

repo_root=$1
handoff_root=$2
trial_evidence=${3-}
action_evidence=${4-}
retention_evidence=${5-}
python_bin=${PICF_PYTHON_BIN:-python3}

exec "$python_bin" - \
  "$repo_root" \
  "$handoff_root" \
  "$trial_evidence" \
  "$action_evidence" \
  "$retention_evidence" <<'PY'
from __future__ import annotations

import ctypes
import datetime as dt
import errno
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

HANDOFF_SCHEMA = "picf-next.adr165-0800-persistent-handoff.v1"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
PERSISTENT_ROOT = Path("/mnt")


def fail(message: str) -> None:
    raise SystemExit(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def direct_absolute_path(value: str, *, name: str, must_exist: bool) -> Path:
    if not value:
        fail(f"{name} is empty")
    path = Path(value)
    if not path.is_absolute():
        fail(f"{name} must be absolute")
    if ".." in path.parts:
        fail(f"{name} must not contain parent traversal")
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != path:
        fail(f"{name} must be one normalized direct path")
    if must_exist:
        if not os.path.lexists(path):
            fail(f"{name} is absent: {path}")
        if path.is_symlink():
            fail(f"{name} must not be a symbolic link: {path}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            fail(f"{name} cannot be resolved: {error}")
        if resolved != path:
            fail(f"{name} contains a symbolic-link path component: {path}")
    return path


def path_status(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": True,
        "kind": "regular_file" if path.is_file() else "directory",
        "is_symlink": False,
        "under_mnt": path.is_relative_to(PERSISTENT_ROOT),
    }


def require_regular_file(path: Path, *, name: str) -> Path:
    direct_absolute_path(str(path), name=name, must_exist=True)
    if not path.is_file():
        fail(f"{name} must be a regular file: {path}")
    return path


def require_directory(path: Path, *, name: str) -> Path:
    direct_absolute_path(str(path), name=name, must_exist=True)
    if not path.is_dir():
        fail(f"{name} must be a directory: {path}")
    return path


def read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    require_regular_file(path, name=name)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"{name} is not one readable JSON artifact: {error}")
    if not isinstance(value, dict):
        fail(f"{name} must contain one JSON object")
    return value


def artifact_receipt(path: Path, *, role: str) -> dict[str, object]:
    payload = read_json_object(path, name=f"{role} artifact")
    receipt = path_status(path)
    receipt.update(
        {
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "schema": payload.get("schema"),
            "status": payload.get("status"),
        }
    )
    return receipt


def git(repo: Path, *arguments: str) -> str:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def source_snapshot(repo: Path) -> dict[str, object]:
    top_level = Path(git(repo, "rev-parse", "--show-toplevel").strip()).resolve(strict=True)
    if top_level != repo:
        fail("clean repo root is not the exact Git worktree root")
    head_sha256 = git(repo, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if not head_sha256:
        fail("clean repo root has no committed HEAD")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        fail("clean repo root is dirty; commit or remove every tracked/untracked change first")
    symbolic = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"},
    )
    if symbolic.returncode not in (0, 1):
        fail(f"git symbolic-ref failed: {symbolic.stderr.strip()}")
    head_ref = symbolic.stdout.strip() if symbolic.returncode == 0 else None
    refs = git(
        repo,
        "for-each-ref",
        "--sort=refname",
        "--format=%(objectname) %(objecttype) %(refname)",
    )
    return {
        "head_sha": head_sha256,
        "head_ref": head_ref,
        "refs_text": refs,
        "status_porcelain": status,
    }


def collect_key_artifacts(label: str, evidence: Path) -> list[dict[str, object]]:
    if evidence.is_file():
        return [artifact_receipt(evidence, role=f"{label}_evidence")]
    patterns = {
        "trial": (
            "ltop_g3_mediator_trial_training_report.json",
            "ltop_g3_training_checkpoint.json",
        ),
        "action": (
            "ltop_g3_mediator_cold_action_*_report.json",
            "ltop_g3_evaluation_report.json",
        ),
        "retention": (
            "ltop_g3_mediator_representation_retention_report.json",
            "ltop_g3_representation_retention_report.json",
        ),
    }[label]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(evidence.glob(pattern))
    if label == "trial":
        nested_manifest = evidence / "checkpoint-model-only" / "ltop_g3_training_checkpoint.json"
        if os.path.lexists(nested_manifest):
            paths.add(nested_manifest)
    if not paths:
        fail(f"{label} evidence directory contains no registered manifest/report")
    return [
        artifact_receipt(path, role=f"{label}_{path.name}") for path in sorted(paths)
    ]


def training_report_path(evidence: Path) -> Path | None:
    if evidence.is_file():
        try:
            payload = read_json_object(evidence, name="trial evidence")
        except SystemExit:
            raise
        if payload.get("phase") == "training" or "checkpoint" in payload:
            return evidence
        return None
    candidate = evidence / "ltop_g3_mediator_trial_training_report.json"
    return candidate if candidate.is_file() and not candidate.is_symlink() else None


def checkpoint_receipt_from_trial(evidence: Path) -> dict[str, object] | None:
    report_path = training_report_path(evidence)
    if report_path is None:
        return None
    report = read_json_object(report_path, name="trial training report")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        fail("trial training report omits its checkpoint receipt")
    checkpoint_value = checkpoint.get("path")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        fail("trial training report checkpoint path is invalid")
    checkpoint_path = direct_absolute_path(
        checkpoint_value,
        name="trial checkpoint",
        must_exist=True,
    )
    require_directory(checkpoint_path, name="trial checkpoint")
    if not checkpoint_path.is_relative_to(PERSISTENT_ROOT):
        fail("trial checkpoint must already persist under /mnt")
    manifest_path = require_regular_file(
        checkpoint_path / "ltop_g3_training_checkpoint.json",
        name="trial checkpoint manifest",
    )
    model_path = require_directory(
        checkpoint_path / "model",
        name="trial checkpoint model directory",
    )
    manifest = read_json_object(manifest_path, name="trial checkpoint manifest")
    manifest_sha256 = sha256_file(manifest_path)
    claimed_manifest_sha256 = checkpoint.get("manifest_sha256")
    if claimed_manifest_sha256 is not None and claimed_manifest_sha256 != manifest_sha256:
        fail("trial checkpoint manifest SHA-256 differs from the training receipt")
    return {
        **path_status(checkpoint_path),
        "copied": False,
        "copy_policy": "checkpoint already persists under /mnt; handoff records identity only",
        "format": checkpoint.get("format"),
        "optimizer_saved": checkpoint.get("optimizer_saved"),
        "model_directory": path_status(model_path),
        "manifest": {
            **path_status(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_sha256,
            "schema": manifest.get("schema"),
            "status": manifest.get("status"),
        },
        "receipt_manifest_sha256": claimed_manifest_sha256,
        "receipt_model_tree_schema": checkpoint.get("model_tree_schema"),
        "receipt_model_tree_sha256": checkpoint.get("model_tree_sha256"),
    }


def evidence_receipt(label: str, value: str) -> dict[str, object] | None:
    if value in ("", "-"):
        return None
    evidence = direct_absolute_path(value, name=f"{label} evidence", must_exist=True)
    if not evidence.is_file() and not evidence.is_dir():
        fail(f"{label} evidence must be one regular file or directory")
    receipt: dict[str, object] = {
        **path_status(evidence),
        "key_artifacts": collect_key_artifacts(label, evidence),
    }
    if label == "trial":
        receipt["checkpoint"] = checkpoint_receipt_from_trial(evidence)
    return receipt


def create_staging_directory(target: Path) -> Path:
    for _attempt in range(64):
        candidate = target.parent / f".{target.name}.staging-{secrets.token_hex(12)}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    fail("cannot allocate a unique handoff staging directory")


def atomic_rename_noreplace(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            AT_FDCWD,
            os.fsencode(source),
            AT_FDCWD,
            os.fsencode(target),
            RENAME_NOREPLACE,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            fail("handoff root appeared during publication; refusing overwrite")
        if error_number not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
            fail(f"atomic handoff publication failed: {os.strerror(error_number)}")

    # Some persistent FUSE/DrvFS mounts reject renameat2 flags. Serialize every
    # cooperating publisher with a durable, never-deleted lock directory, then
    # use the filesystem's same-directory atomic rename after one final check.
    publication_lock = target.parent / f".{target.name}.publish-lock"
    try:
        publication_lock.mkdir(mode=0o700)
    except FileExistsError:
        fail(f"handoff publication lock already exists: {publication_lock}")
    write_text_exclusive(
        publication_lock / "receipt.txt",
        f"source={source}\ntarget={target}\nmode=atomic-rename-under-exclusive-lock\n",
    )
    fsync_directory(publication_lock)
    fsync_directory(target.parent)
    if os.path.lexists(target):
        fail("handoff root appeared during locked publication; refusing overwrite")
    try:
        os.rename(source, target)
    except OSError as error:
        fail(f"locked atomic handoff publication failed: {error}")


def main() -> None:
    repo = direct_absolute_path(sys.argv[1], name="clean repo root", must_exist=True)
    require_directory(repo, name="clean repo root")
    target = direct_absolute_path(sys.argv[2], name="handoff root", must_exist=False)
    if target == PERSISTENT_ROOT or not target.is_relative_to(PERSISTENT_ROOT):
        fail("handoff root must be one absent direct path below /mnt")
    if os.path.lexists(target):
        fail("handoff root already exists; refusing overwrite")
    parent = direct_absolute_path(
        str(target.parent),
        name="handoff parent",
        must_exist=True,
    )
    require_directory(parent, name="handoff parent")
    if not parent.is_relative_to(PERSISTENT_ROOT):
        fail("handoff parent must be a direct directory under /mnt")

    before = source_snapshot(repo)
    evidence = {
        "trial": evidence_receipt("trial", sys.argv[3]),
        "action": evidence_receipt("action", sys.argv[4]),
        "retention": evidence_receipt("retention", sys.argv[5]),
    }
    staging = create_staging_directory(target)

    bundle_path = staging / "repository.bundle"
    git(repo, "bundle", "create", str(bundle_path), "HEAD", "--all")
    require_regular_file(bundle_path, name="repository bundle")
    fsync_file(bundle_path)
    git(repo, "bundle", "verify", str(bundle_path))
    bundle_heads_text = git(repo, "bundle", "list-heads", str(bundle_path))
    bundle_heads = {
        (parts[0], parts[1])
        for line in bundle_heads_text.splitlines()
        if len(parts := line.split(maxsplit=1)) == 2
    }
    if (before["head_sha"], "HEAD") not in bundle_heads:
        fail("repository bundle omits the current HEAD")
    for line in str(before["refs_text"]).splitlines():
        object_name, _object_type, refname = line.split(maxsplit=2)
        if (object_name, refname) not in bundle_heads:
            fail(f"repository bundle omits ref {refname}")

    write_text_exclusive(staging / "HEAD", f"{before['head_sha']}\n")
    source_status = "\n".join(
        (
            f"repository_root={repo}",
            f"head_sha={before['head_sha']}",
            f"head_ref={before['head_ref'] or '(detached)'}",
            "clean=true",
            "status_porcelain_bytes=0",
            "bundle_scope=current HEAD plus all refs",
            "bundle_remote_access=none",
            "",
        )
    )
    write_text_exclusive(staging / "source_status.txt", source_status)
    write_text_exclusive(staging / "refs.txt", str(before["refs_text"]))

    manifest = {
        "schema": HANDOFF_SCHEMA,
        "status": "PASS",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository": {
            "source_path": str(repo),
            "head_sha": before["head_sha"],
            "head_ref": before["head_ref"],
            "clean": True,
            "bundle": "repository.bundle",
            "bundle_scope": "current HEAD plus all refs",
            "remote_access_required": False,
        },
        "checkpoint_copy_policy": {
            "copied": False,
            "reason": "large checkpoints are identity-bound in place because they already persist under /mnt",
        },
        "evidence": evidence,
    }
    write_text_exclusive(staging / "evidence_manifest.json", canonical_json(manifest) + "\n")

    readme = f"""# ADR165 08:00 Persistent Handoff

This directory was atomically published from a clean Git worktree at `{before['head_sha']}`.
`repository.bundle` contains that HEAD and every ref visible in the source repository; recovery does not require a remote.

## Verify

```bash
sha256sum -c SHA256SUMS
mkdir /tmp/adr165-handoff-verify.git
git -C /tmp/adr165-handoff-verify.git init --bare
git -C /tmp/adr165-handoff-verify.git bundle verify "$PWD/repository.bundle"
```

## Recover source

```bash
git clone repository.bundle recovered-source
git -C recovered-source checkout {before['head_sha']}
```

`evidence_manifest.json` records the absolute trial/action/retention evidence paths and SHA-256 identities of key reports/manifests. Large model checkpoints are deliberately not copied: the checkpoint receipt records its direct `/mnt` path, model-directory state, and manifest identity because the checkpoint already persists under `/mnt`.

Publication never overwrites a destination and never deletes source, evidence, checkpoint, staging, or handoff content. If publication failed before the final rename, an explicitly named sibling `.staging-*` directory may remain for inspection. Filesystems without `RENAME_NOREPLACE` retain a sibling `.publish-lock` receipt after successful or failed publication.
"""
    write_text_exclusive(staging / "README.md", readme)

    checksum_paths = sorted(
        path for path in staging.rglob("*") if path.is_file() and not path.is_symlink()
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}"
        for path in checksum_paths
    ]
    write_text_exclusive(staging / "SHA256SUMS", "\n".join(checksum_lines) + "\n")

    after = source_snapshot(repo)
    if after != before:
        fail(f"source changed while preparing handoff; unpublished staging remains at {staging}")
    if os.path.lexists(target):
        fail(f"handoff root appeared before publication; unpublished staging remains at {staging}")
    for path in sorted(staging.rglob("*")):
        if path.is_symlink():
            fail(f"handoff staging contains a symbolic link: {path}")
        if path.is_file():
            fsync_file(path)
    fsync_directory(staging)
    fsync_directory(parent)
    atomic_rename_noreplace(staging, target)
    fsync_directory(parent)
    print(target)


main()
PY
