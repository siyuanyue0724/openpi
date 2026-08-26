from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_adr175_launcher_is_explicit_unique_and_verifies_the_source_tree() -> None:
    source = (ROOT / "configs/cloud/adr175_matched_arm.sh").read_text(encoding="ascii")

    assert 'if [ "$#" -ne 2 ]' in source
    assert "date +%Y%m%dT%H%M%S%N%z" in source
    assert 'mkdir "$run_dir"' in source
    assert "verify_adr175_implementation_closure.py" in source
    assert '--root "$WORKTREE"' in source
    assert '--expected-artifact-sha256 "$IMPLEMENTATION_CLOSURE_ARTIFACT_SHA256"' in source
    assert '--output "$run_dir/implementation-closure-verification.json"' in source
    assert "PICF_ADR175_WORKTREE:?" in source
    assert "PICF_ADR175_CONTRACT_ROOT:?" in source
    assert "PICF_ADR175_IMPLEMENTATION_CLOSURE_FILE_SHA256:?" in source
    assert "PICF_ADR175_IMPLEMENTATION_CLOSURE_ARTIFACT_SHA256:?" in source
    assert "PICF_ADR175_CONTRACT_FILE_SHA256:?" in source
    assert "PICF_ADR175_STREAM_PLAN_FILE_SHA256:?" in source
    assert "PICF_ADR175_REPRESENTATION_SPLIT_FILE_SHA256:?" in source
    assert "PICF_ADR175_ENTITY_EVALUATION_PLAN_FILE_SHA256:?" in source
    assert "eaf6aab1a662ac4ac2acf747bd7fd69d00470a007d9835e2876264c75c91f9cc" not in source
