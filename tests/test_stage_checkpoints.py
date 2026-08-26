# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from picf_next.training.stage_checkpoints import (
    StationaryTemporalCheckpointProvenance,
    load_stationary_temporal_checkpoint,
    parameter_scope_sha256,
    save_stationary_temporal_checkpoint,
    sha256_file,
)
from picf_next.training.stationary_stage import load_stationary_temporal_stage_recipe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/molmoact2_calvin_m3_stationary_temporal.json"


def _modules():
    stage = load_stationary_temporal_stage_recipe(CONFIG)
    foundation = stage.structural_foundation(ROOT)
    return stage, foundation, foundation.build_core(), foundation.build_objective()


def _provenance(core, objective) -> StationaryTemporalCheckpointProvenance:
    stage, foundation, _unused_core, _unused_objective = _modules()
    trainable, frozen = parameter_scope_sha256(core, objective)
    return StationaryTemporalCheckpointProvenance(
        stage_recipe_sha256=stage.recipe_sha256,
        source_coverage_recipe_sha256=stage.source_coverage_recipe_sha256,
        foundation_recipe_sha256=foundation.recipe_sha256,
        m2_checkpoint_sha256="1" * 64,
        feature_cache_manifest_sha256="2" * 64,
        dataset_manifest_sha256="3" * 64,
        physical_sidecar_manifest_sha256="4" * 64,
        clip_plan_sha256="5" * 64,
        trainable_parameter_scope_sha256=trainable,
        frozen_parameter_scope_sha256=frozen,
        code_revision="6" * 40,
        optimizer_steps=200,
        state_parameter_version=200,
    )


def test_stationary_temporal_checkpoint_round_trip_is_full_and_stateless(
    tmp_path: Path,
) -> None:
    _stage, _foundation, source_core, source_objective = _modules()
    provenance = _provenance(source_core, source_objective)
    path = tmp_path / "stationary-temporal-core.pt"
    digest = save_stationary_temporal_checkpoint(
        path,
        core=source_core,
        objective=source_objective,
        provenance=provenance,
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"schema", "provenance", "core", "objective"}
    assert raw["provenance"]["recurrent_state_serialized"] is False
    assert any(name.startswith("posterior_filter.") for name in raw["core"])

    _stage, _foundation, target_core, target_objective = _modules()
    with torch.no_grad():
        next(target_core.parameters()).add_(10.0)
    loaded = load_stationary_temporal_checkpoint(
        target_core,
        target_objective,
        path,
        expected_sha256=digest,
        expected_provenance=provenance,
    )
    assert loaded == provenance
    for name, value in source_core.state_dict().items():
        torch.testing.assert_close(target_core.state_dict()[name], value)
    for name, value in source_objective.state_dict().items():
        torch.testing.assert_close(target_objective.state_dict()[name], value)


def test_stationary_temporal_checkpoint_rejects_partial_m2_state(tmp_path: Path) -> None:
    _stage, _foundation, core, objective = _modules()
    provenance = _provenance(core, objective)
    path = tmp_path / "partial.pt"
    torch.save(
        {
            "schema": "picf-next.stationary-temporal-core.v1",
            "provenance": provenance.to_dict(),
            "core": {
                name: value
                for name, value in core.state_dict().items()
                if name.startswith(("projector.", "discovery."))
            },
            "objective": objective.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="posterior filter"):
        load_stationary_temporal_checkpoint(
            core,
            objective,
            path,
            expected_sha256=sha256_file(path),
        )


def test_stationary_temporal_checkpoint_rejects_legacy_sensor_state(tmp_path: Path) -> None:
    _stage, _foundation, core, objective = _modules()
    provenance = _provenance(core, objective)
    state = dict(core.state_dict())
    state["posterior_filter.transition.visibility_persistence_head.weight"] = torch.zeros(1)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema": "picf-next.stationary-temporal-core.v1",
            "provenance": provenance.to_dict(),
            "core": state,
            "objective": objective.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="legacy non-identifiable sensor"):
        load_stationary_temporal_checkpoint(
            core,
            objective,
            path,
            expected_sha256=sha256_file(path),
        )


def test_stationary_temporal_provenance_forbids_recurrent_state() -> None:
    _stage, _foundation, core, objective = _modules()
    payload = _provenance(core, objective).to_dict()
    payload["recurrent_state_serialized"] = True
    with pytest.raises(ValueError, match="cannot serialize recurrent state"):
        StationaryTemporalCheckpointProvenance.from_dict(payload)
