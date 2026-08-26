from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from picf_next.contracts import ContractError

pytest.importorskip("torch")
calvin_geometry_module = pytest.importorskip("picf_next.data.calvin_geometry_schema")
recipe_module = pytest.importorskip("picf_next.training.recipe")

CALVIN_OBJECT_GEOMETRY_CONTRACT = calvin_geometry_module.CALVIN_OBJECT_GEOMETRY_CONTRACT
RECIPE_SCHEMA = recipe_module.RECIPE_SCHEMA
load_training_recipe = recipe_module.load_training_recipe
training_recipe_from_dict = recipe_module.training_recipe_from_dict
write_preflight_report = recipe_module.write_preflight_report


RECIPE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "training" / "molmoact2_calvin_m3_probe.json"
)


def _payload() -> dict[str, object]:
    return json.loads(RECIPE_PATH.read_text())


def test_m3_recipe_roundtrips_builds_and_is_not_long_train_authorized(tmp_path: Path) -> None:
    recipe = load_training_recipe(RECIPE_PATH)
    reconstructed = training_recipe_from_dict(recipe.to_dict())

    assert recipe.recipe_sha256 == reconstructed.recipe_sha256
    assert recipe.geometry_contract == CALVIN_OBJECT_GEOMETRY_CONTRACT
    assert recipe.authorization.stage == "M3_structural_probe"
    assert recipe.authorization.long_training_authorized is False
    assert recipe.geometry_overshooting.horizons == (1, 2)
    assert recipe.dataset.state_dim == 15
    assert recipe.dataset.action_dim == 7
    assert recipe.dataset.action_horizon == 10
    assert recipe.policy.num_flow_timesteps == 8
    assert recipe.core_config.object_value_dim == 784
    assert recipe.core_config.dense_token_dims == {"molmo_vision_patch": 2304}
    assert recipe.set_loss_config.existence_weight == 2.0
    assert recipe.set_loss_config.ownership_ce_weight == 5.0
    assert recipe.set_loss_config.ownership_dice_weight == 5.0
    assert recipe.set_loss_config.localization_confidence_weight == 1.0
    assert recipe.set_loss_config.geometry_weight == 1.0

    core = recipe.build_core()
    objective = recipe.build_objective()
    # Directly supervised p11/p01 branches make the two-state observation
    # kernel identifiable without exposing previous detectability to either head.
    # ADR-60 adds exactly one shared semi-Markov duration coefficient.
    assert sum(parameter.numel() for parameter in core.parameters()) == 29_056_341
    assert sum(parameter.numel() for parameter in objective.parameters()) == 2
    assert objective.geometry_overshooting_criterion.config.weight == 0.5
    assert objective.binding_criterion.config == objective.temporal_binding_criterion.config

    recipe.assert_optimizer_steps_authorized(200)
    with pytest.raises(PermissionError, match="at most 200"):
        recipe.assert_optimizer_steps_authorized(201)

    report_path = tmp_path / "preflight.json"
    write_preflight_report(recipe, report_path, root=RECIPE_PATH.parents[2])
    report = json.loads(report_path.read_text())
    assert report["schema"] == RECIPE_SCHEMA
    assert report["recipe_sha256"] == recipe.recipe_sha256
    assert report["long_training_authorized"] is False
    assert report["artifacts"]["normalization_samples"] == 494


def test_recipe_builds_the_pinned_host_training_contract() -> None:
    pytest.importorskip("picf_next.hosts.molmoact2_training")
    host_training = load_training_recipe(RECIPE_PATH).build_host_training_config()

    assert host_training.sequence_length == 1
    assert host_training.require_explicit_flow_randomness is True


def test_recipe_horizons_drive_the_concrete_loss_target_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_module = pytest.importorskip("picf_next.hosts.molmoact2_training")
    captured: dict[str, object] = {}

    def visible(sidecar):
        captured["sidecar"] = sidecar
        return "visible"

    def geometry(index, **kwargs):
        captured["index"] = index
        captured.update(kwargs)
        return "geometry"

    def compose(*builders):
        captured["builders"] = builders
        return "composed"

    monkeypatch.setattr(training_module, "CalvinVisibleObjectTargetBuilder", visible)
    monkeypatch.setattr(training_module, "CalvinGeometryOvershootingTargetBuilder", geometry)
    monkeypatch.setattr(training_module, "compose_calvin_loss_target_builders", compose)
    recipe = load_training_recipe(RECIPE_PATH)

    result = recipe.build_calvin_loss_target_builder("index", "sidecar", "geometry-provider")

    assert result == "composed"
    assert captured["builders"] == ("visible", "geometry")
    assert captured["maximum_horizon"] == 2
    assert captured["supervised_horizons"] == (1, 2)
    assert captured["geometry_contract"] == CALVIN_OBJECT_GEOMETRY_CONTRACT
    assert captured["geometry_provider"] == "geometry-provider"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__("silent_typo", 1),
            "fields differ from schema",
        ),
        (
            lambda payload: payload["core"]["temporal"].__setitem__("hidden_typo", 1),
            "core.temporal fields differ from schema",
        ),
        (
            lambda payload: payload["geometry_contract"].__setitem__(
                "normalization_scale", [0.5, 1.0, 1.0]
            ),
            "v3 physical sidecar",
        ),
        (
            lambda payload: payload["objective"]["geometry_overshooting"].__setitem__(
                "horizons", [1, 3]
            ),
            "overshooting horizons",
        ),
        (
            lambda payload: payload["streaming"].__setitem__("gradient_transitions", 2),
            r"audited 0\+1 exposure",
        ),
        (
            lambda payload: payload["objective"]["geometry_overshooting"].__setitem__(
                "fraction_denominator", 2
            ),
            "exact unit fraction",
        ),
        (
            lambda payload: payload["dataset"].__setitem__("action_horizon", 11),
            "action horizon",
        ),
        (
            lambda payload: payload["policy"].__setitem__("num_flow_timesteps", 7),
            "flow schedule",
        ),
        (
            lambda payload: payload["core"]["temporal"].__setitem__(
                "association_address_temperature", 0.2
            ),
            "runtime address temperature must equal temporal binding temperature",
        ),
        (
            lambda payload: payload["core"]["temporal"].__setitem__(
                "association_address_logit_bias", -1.0
            ),
            "runtime address bias must equal temporal binding logit bias",
        ),
        (
            lambda payload: payload["objective"]["binding"].__setitem__(
                "objective", "multi_positive_infonce"
            ),
            "runtime identity association requires calibrated sigmoid address binding",
        ),
    ],
)
def test_recipe_rejects_schema_and_semantic_drift(mutate, message: str) -> None:
    payload = copy.deepcopy(_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        training_recipe_from_dict(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["policy"].__setitem__("flow_matching_cutoff", True),
            "policy.flow_matching_cutoff must be a finite nonnegative number",
        ),
        (
            lambda payload: payload["policy"].__setitem__("optimizer_betas", [True, 0.95]),
            r"policy.optimizer_betas\[0\] must be one finite number",
        ),
        (
            lambda payload: payload["dataset"].__setitem__("state_axes", "not-a-list"),
            "dataset.state_axes must be a sequence of nonempty strings",
        ),
    ],
)
def test_recipe_rejects_json_scalar_type_confusion(mutate, message: str) -> None:
    payload = copy.deepcopy(_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        training_recipe_from_dict(payload)


def test_recipe_long_training_authorization_is_atomic() -> None:
    payload = _payload()
    payload["authorization"]["long_training_authorized"] = True

    with pytest.raises(ValueError, match="only an M6_long_train"):
        training_recipe_from_dict(payload)


def test_recipe_hash_covers_every_objective_weight() -> None:
    baseline = load_training_recipe(RECIPE_PATH)
    payload = _payload()
    payload["objective"]["weights"]["binding"] = 0.051
    changed = training_recipe_from_dict(payload)

    assert changed.recipe_sha256 != baseline.recipe_sha256


def test_recipe_repository_artifacts_are_content_addressed() -> None:
    recipe = load_training_recipe(RECIPE_PATH)
    report = recipe.validate_repository_artifacts(RECIPE_PATH.parents[2])
    assert report["physical_frames"] == 378
    assert report["geometry_object_records"] == 3780

    payload = _payload()
    payload["artifacts"]["normalization_file_sha256"] = "0" * 64
    changed = training_recipe_from_dict(payload)
    with pytest.raises(ContractError, match="artifact SHA-256 changed"):
        changed.validate_repository_artifacts(RECIPE_PATH.parents[2])
