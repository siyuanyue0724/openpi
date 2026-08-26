import math
import subprocess
import sys
from pathlib import Path

import pytest

from tools.audit_molmoact2_m3_gradients import (
    _audit_plan_steps,
    _copy_family_gradients,
    _cosine_with_reference,
    _geometry_rollout_diagnostics,
    _gradient_statistics,
    _loss_only_oracle_plan,
    _loss_only_oracle_posterior,
    _parameter_family,
    _replace_oracle_geometry_with_current_discovery,
)


def test_gradient_audit_requires_explicit_plan_extension_at_plan_end() -> None:
    with pytest.raises(ValueError, match="no next frozen transition"):
        _audit_plan_steps(
            checkpoint_steps=20,
            checkpoint_plan_steps=20,
            extended_plan_steps=None,
        )

    assert (
        _audit_plan_steps(
            checkpoint_steps=20,
            checkpoint_plan_steps=20,
            extended_plan_steps=21,
        )
        == 21
    )


@pytest.mark.parametrize("extended", (0, 19, 20, True))
def test_gradient_audit_rejects_non_extension(extended: int) -> None:
    with pytest.raises(ValueError, match="longer than the checkpoint plan"):
        _audit_plan_steps(
            checkpoint_steps=20,
            checkpoint_plan_steps=20,
            extended_plan_steps=extended,
        )


def test_gradient_audit_uses_remaining_checkpoint_plan_without_extension() -> None:
    assert (
        _audit_plan_steps(
            checkpoint_steps=20,
            checkpoint_plan_steps=200,
            extended_plan_steps=None,
        )
        == 200
    )


def test_gradient_audit_imports_current_checkout_without_editable_install() -> None:
    script = Path(__file__).resolve().parents[1] / "tools/audit_molmoact2_m3_gradients.py"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy,sys;runpy.run_path(sys.argv[1],run_name='picf_m3_gradient_audit')",
            str(script),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("name", "family"),
    (
        ("joint_bridge.sequence_bridge.policy.model.weight", "host_policy"),
        (
            "joint_bridge.sequence_bridge.policy.action_layer_adapter.object_k_proj.weight",
            "action_adapter",
        ),
        ("joint_bridge.sequence_bridge.action_adapter.gates", "action_adapter"),
        ("joint_bridge.sequence_bridge.core.discovery.weight", "picf_core"),
        (
            "joint_bridge.objective.binding_criterion.relation.logit_bias",
            "multimodal_relation_calibration",
        ),
        (
            "joint_bridge.sequence_bridge.core.posterior_filter."
            "address_relation.logit_scale_parameter",
            "temporal_relation_calibration",
        ),
        ("unexpected.weight", "other"),
    ),
)
def test_gradient_parameter_family_is_exhaustive(name: str, family: str) -> None:
    assert _parameter_family(name) == family


def test_gradient_statistics_and_reference_cosine_use_actual_trainable_paths() -> None:
    torch = pytest.importorskip("torch")
    host = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    core = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    host.grad = torch.tensor([3.0, 4.0])
    core.grad = torch.tensor([0.0, 12.0])
    named = (
        ("joint_bridge.sequence_bridge.policy.host", host),
        ("joint_bridge.sequence_bridge.core.core", core),
    )

    statistics = _gradient_statistics(named, clip_norm=6.5)

    assert statistics["global_l2_norm"] == pytest.approx(13.0)
    assert statistics["clip_multiplier"] == pytest.approx(0.5)
    assert statistics["group_l2_norm"] == {
        "host_policy": pytest.approx(5.0),
        "picf_core": pytest.approx(12.0),
    }
    reference = _copy_family_gradients(named, "picf_core")
    core.grad = torch.tensor([12.0, 0.0])
    assert _cosine_with_reference(named, reference, "picf_core") == pytest.approx(0.0)
    core.grad = torch.tensor([0.0, -6.0])
    assert _cosine_with_reference(named, reference, "picf_core") == pytest.approx(-1.0)
    assert math.isfinite(statistics["maximum_absolute_gradient"])


def test_geometry_rollout_diagnostics_reconstructs_the_exact_criterion() -> None:
    torch = pytest.importorskip("torch")
    from picf_next.models.dynamics_loss import (
        ObjectGeometryOvershootingCriterion,
        ObjectGeometryRolloutTarget,
    )
    from picf_next.models.temporal import (
        ActionConditionedObjectTransition,
        ObjectBeliefBatch,
        TemporalFilterConfig,
    )
    from tests.geometry_contract import synthetic_geometry_contract

    geometry_contract = synthetic_geometry_contract(2)
    config = TemporalFilterConfig(
        address_dim=2,
        content_dim=2,
        geometry_dim=2,
        geometry_contract=geometry_contract,
        action_dim=2,
        reference_delta_t_s=0.1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    transition = ActionConditionedObjectTransition(config)
    valid = torch.tensor([[True, False]])
    start = ObjectBeliefBatch(
        address_mean=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        content_mean=torch.tensor([[[0.2, 0.3], [0.0, 0.0]]]),
        geometry_mean=torch.tensor([[[0.4, -0.2], [0.0, 0.0]]]),
        geometry_covariance_diag=torch.tensor([[[0.02, 0.03], [0.0, 0.0]]]),
        existence_logits=torch.tensor([[2.0, 0.0]]),
        visibility_given_existence_logits=torch.tensor([[1.0, 0.0]]),
        measurement_age_s=torch.tensor([[0.3, 0.0]]),
        valid=valid,
        age=torch.tensor([[3, 0]]),
    )
    target = ObjectGeometryRolloutTarget(
        executed_actions=torch.tensor([[[0.1, -0.2], [0.2, 0.1]]]),
        delta_t_s=torch.tensor([[0.1, 0.1]]),
        step_valid=torch.tensor([[True, True]]),
        identity_keys=((("object",), ("object",)),),
        geometry=torch.tensor([[[[0.45, -0.15]], [[0.5, -0.1]]]]),
        geometry_variance=torch.tensor([[[[0.01, 0.01]], [[0.01, 0.01]]]]),
        geometry_supervised=torch.ones(1, 2, 1, 2, dtype=torch.bool),
        geometry_contract=geometry_contract,
    )
    row_keys = (("object", None),)

    expected = ObjectGeometryOvershootingCriterion()(transition, start, row_keys, target).loss
    report = _geometry_rollout_diagnostics(transition, start, row_keys, target)

    assert report["criterion_loss_reconstructed"] == pytest.approx(
        float(expected.detach().item()), rel=1e-6, abs=1e-6
    )
    assert [item["horizon"] for item in report["horizons"]] == [1, 2]
    assert report["horizons"][0]["objects"][0]["identity_key"] == "object"
    assert report["start_rows"][0]["row"] == 0


def test_loss_only_oracle_plan_retains_identity_and_allocates_only_true_births() -> None:
    torch = pytest.importorskip("torch")
    from picf_next.models.set_loss import ObjectSetTarget, SetMatch
    from picf_next.posterior import BIRTH_EVENT, MATCH_EVENT, MISS_EVENT, UNUSED_EVENT

    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(2, dtype=torch.bool),
        token_supervised=torch.ones(2, dtype=torch.bool),
        temporal_identity_keys=("existing", "new"),
    )
    match = SetMatch(
        prediction_indices=torch.tensor([2, 0]),
        target_indices=torch.tensor([0, 1]),
    )

    plan, keys, diagnostics = _loss_only_oracle_plan(
        torch.tensor([[True, True, False, False]]),
        (("existing", "occluded", None, None),),
        (target,),
        (match,),
        observation_count=3,
    )

    assert plan.observation_to_posterior.tolist() == [[2, -1, 0]]
    assert plan.matched_observation_for_row.tolist() == [[2, -1, -1, -1]]
    assert plan.birth_observation_for_row.tolist() == [[-1, -1, 0, -1]]
    assert plan.event_type.tolist() == [[MATCH_EVENT, MISS_EVENT, BIRTH_EVENT, UNUSED_EVENT]]
    assert keys == (("existing", "occluded", "new", None),)
    assert diagnostics == {"births": 1, "matches": 1, "unallocated_births": 0}


def test_historical_oracle_posterior_is_self_contained_outside_production_filter() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from picf_next.models.set_loss import ObjectSetTarget, SetMatch
    from picf_next.models.temporal import ObjectBeliefBatch, TemporalFilterConfig
    from tests.geometry_contract import synthetic_geometry_contract

    config = TemporalFilterConfig(
        address_dim=2,
        content_dim=2,
        geometry_dim=2,
        geometry_contract=synthetic_geometry_contract(2),
        action_dim=2,
        reference_delta_t_s=0.1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    valid = torch.tensor([[True, False, False]])
    prior = ObjectBeliefBatch(
        address_mean=torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        content_mean=torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        geometry_mean=torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        geometry_covariance_diag=torch.tensor([[[0.2, 0.2], [0.0, 0.0], [0.0, 0.0]]]),
        existence_logits=torch.tensor([[3.0, 0.0, 0.0]]),
        visibility_given_existence_logits=torch.tensor([[2.0, 0.0, 0.0]]),
        measurement_age_s=torch.tensor([[0.4, 0.0, 0.0]]),
        valid=valid,
        age=torch.tensor([[4, 0, 0]]),
    )
    discovery = SimpleNamespace(
        address_mean=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        content_mean=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        geometry_mean=torch.tensor([[[2.0, -2.0], [5.0, 6.0]]]),
        geometry_variance=torch.full((1, 2, 2), 0.2),
        existence=torch.tensor([[0.9, 0.95]]),
        existence_logits=torch.zeros(1, 2),
    )
    target = ObjectSetTarget(
        ownership=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        token_valid=torch.ones(2, dtype=torch.bool),
        token_supervised=torch.ones(2, dtype=torch.bool),
        temporal_identity_keys=("existing", "new"),
    )
    match = SetMatch(
        prediction_indices=torch.tensor([0, 1]),
        target_indices=torch.tensor([0, 1]),
    )
    core_output = SimpleNamespace(
        posterior=SimpleNamespace(prior_prediction=SimpleNamespace(belief=prior)),
        discovery=discovery,
    )

    belief, keys, diagnostics, _plan = _loss_only_oracle_posterior(
        SimpleNamespace(config=config),
        core_output,
        (target,),
        (match,),
        (("existing", None, None),),
    )

    assert keys == (("existing", "new", None),)
    assert diagnostics == {"births": 1, "matches": 1, "unallocated_births": 0}
    assert belief.valid.tolist() == [[True, True, False]]
    torch.testing.assert_close(belief.geometry_mean[0, 0], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(
        belief.geometry_covariance_diag[0, 0],
        torch.tensor([0.1, 0.1]),
    )
    torch.testing.assert_close(belief.geometry_mean[0, 1], torch.tensor([5.0, 6.0]))


def test_discovery_mean_counterfactual_replaces_only_oracle_mapped_rows() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from picf_next.models.temporal import ObjectBeliefBatch

    valid = torch.tensor([[True, True, False]])
    belief = ObjectBeliefBatch(
        address_mean=torch.zeros(1, 3, 2),
        content_mean=torch.zeros(1, 3, 2),
        geometry_mean=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]),
        geometry_covariance_diag=torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]]),
        existence_logits=torch.tensor([[2.0, 1.0, 0.0]]),
        visibility_given_existence_logits=torch.tensor([[1.0, 1.0, 0.0]]),
        measurement_age_s=torch.tensor([[0.5, 0.2, 0.0]]),
        valid=valid,
        age=torch.tensor([[5, 2, 0]]),
    )
    discovery = SimpleNamespace(geometry_mean=torch.tensor([[[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]]]))
    plan = SimpleNamespace(observation_to_posterior=torch.tensor([[1, -1, 0]]))

    replaced, coordinates = _replace_oracle_geometry_with_current_discovery(belief, discovery, plan)

    torch.testing.assert_close(
        replaced.geometry_mean,
        torch.tensor([[[5.0, 4.0], [9.0, 8.0], [0.0, 0.0]]]),
    )
    torch.testing.assert_close(replaced.geometry_covariance_diag, belief.geometry_covariance_diag)
    assert coordinates == 4
