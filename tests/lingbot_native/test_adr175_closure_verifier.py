from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools/build_adr175_implementation_closure.py"
TOOL = ROOT / "tools/verify_adr175_implementation_closure.py"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


def _manifest(root: Path, path: Path) -> tuple[str, str]:
    content = (root / "module.py").read_bytes()
    semantic = {
        "files": [
            {
                "bytes": len(content),
                "path": "module.py",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
        "schema": "picf-next.adr175-implementation-closure.v1",
        "source_root": str(root),
    }
    artifact_sha256 = _canonical_sha256(semantic)
    payload = {"artifact_sha256": artifact_sha256, **semantic}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return artifact_sha256, hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    root: Path,
    manifest: Path,
    artifact: str,
    manifest_file: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(TOOL),
            "--root",
            str(root),
            "--manifest",
            str(manifest),
            "--manifest-file-sha256",
            manifest_file,
            "--expected-artifact-sha256",
            artifact,
        ),
        check=False,
        capture_output=True,
        text=True,
    )


def test_adr175_closure_verifier_hashes_every_declared_file(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n", encoding="ascii")
    manifest = tmp_path / "closure.json"
    artifact, manifest_file = _manifest(root, manifest)

    accepted = _run(root, manifest, artifact, manifest_file)

    assert accepted.returncode == 0, accepted.stderr
    receipt = json.loads(accepted.stdout)
    assert receipt["file_count"] == 1
    assert receipt["closure_artifact_sha256"] == artifact

    (root / "module.py").write_text("value = 2\n", encoding="ascii")
    rejected = _run(root, manifest, artifact, manifest_file)

    assert rejected.returncode != 0
    assert "file SHA-256 differs" in rejected.stderr


def test_adr175_closure_builder_and_verifier_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n", encoding="ascii")
    output = root / "ADR175_IMPLEMENTATION_CLOSURE.json"
    built = subprocess.run(
        (sys.executable, str(BUILDER), "--root", str(root), "--output", str(output)),
        check=False,
        capture_output=True,
        text=True,
    )

    assert built.returncode == 0, built.stderr
    identity = json.loads(built.stdout)
    verified = _run(
        root,
        output,
        identity["artifact_sha256"],
        identity["file_sha256"],
    )

    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["file_count"] == 1


def test_adr175_closure_verifier_rejects_undeclared_source_files(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n", encoding="ascii")
    manifest = tmp_path / "closure.json"
    artifact, manifest_file = _manifest(root, manifest)
    (root / "unfrozen.py").write_text("value = 2\n", encoding="ascii")

    rejected = _run(root, manifest, artifact, manifest_file)

    assert rejected.returncode != 0
    assert "undeclared=['unfrozen.py']" in rejected.stderr
