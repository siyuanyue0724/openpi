# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from picf_next.training.stationary_accelerate_checkpoint import (
    StationaryAccelerateCheckpointIdentity,
    _write_control,
    validate_stationary_accelerate_checkpoint,
)


def _identity() -> StationaryAccelerateCheckpointIdentity:
    return StationaryAccelerateCheckpointIdentity(
        stage_recipe_sha256="1" * 64,
        source_coverage_recipe_sha256="2" * 64,
        foundation_recipe_sha256="3" * 64,
        m2_checkpoint_sha256="4" * 64,
        feature_cache_manifest_sha256="5" * 64,
        dataset_manifest_sha256="6" * 64,
        physical_sidecar_manifest_sha256="7" * 64,
        clip_plan_sha256="8" * 64,
        code_revision="9" * 40,
        world_size=2,
        total_steps=200,
    )


def test_stationary_accelerate_control_binds_every_state_file(tmp_path: Path) -> None:
    identity = _identity()
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "optimizer.bin").write_bytes(b"optimizer")
    _write_control(tmp_path, identity=identity, completed_steps=20)

    assert validate_stationary_accelerate_checkpoint(tmp_path, identity=identity) == 20
    (tmp_path / "optimizer.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing, added, or corrupt"):
        validate_stationary_accelerate_checkpoint(tmp_path, identity=identity)


def test_stationary_accelerate_control_rejects_posterior_rank_state(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "picf_rank_state_00000.pt").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="posterior rank state"):
        _write_control(tmp_path, identity=_identity(), completed_steps=20)


def test_stationary_accelerate_identity_forbids_recurrent_state() -> None:
    payload = _identity().to_dict()
    payload["recurrent_state_serialized"] = True
    with pytest.raises(ValueError, match="cannot serialize recurrent state"):
        StationaryAccelerateCheckpointIdentity(**payload)
