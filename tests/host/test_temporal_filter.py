from __future__ import annotations

from dataclasses import replace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
temporal = pytest.importorskip("picf_next.models.temporal")

ActionConditionedObjectTransition = temporal.ActionConditionedObjectTransition
ObjectBeliefBatch = temporal.ObjectBeliefBatch
TemporalFilterConfig = temporal.TemporalFilterConfig
empty_object_belief = temporal.empty_object_belief
GEOMETRY = synthetic_geometry_contract(2)


def _config() -> TemporalFilterConfig:
    return TemporalFilterConfig(
        address_dim=2,
        content_dim=2,
        geometry_dim=2,
        geometry_contract=GEOMETRY,
        action_dim=3,
        reference_delta_t_s=0.1,
        hidden_dim=12,
        num_layers=2,
        num_heads=3,
        dropout=0.0,
    )


def _prior() -> ObjectBeliefBatch:
    torch.manual_seed(503)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])

    def rows(width: int):
        return torch.randn(2, 4, width) * valid.unsqueeze(-1)

    address = torch.nn.functional.normalize(torch.randn(2, 4, 2), dim=-1)
    address = address * valid.unsqueeze(-1)
    covariance = torch.full((2, 4, 2), 0.2) * valid.unsqueeze(-1)
    logits = torch.full((2, 4), 2.0) * valid
    visibility_logits = torch.full((2, 4), 1.0) * valid
    measurement_age_s = torch.tensor([[0.3, 0.7, 0.2, 0.0], [0.5, 0.1, 0.0, 0.0]]) * valid
    age = torch.tensor([[3, 7, 2, 0], [5, 1, 0, 0]], dtype=torch.long)
    return ObjectBeliefBatch(
        address_mean=address,
        content_mean=rows(2),
        geometry_mean=rows(2),
        geometry_covariance_diag=covariance,
        existence_logits=logits,
        visibility_given_existence_logits=visibility_logits,
        measurement_age_s=measurement_age_s,
        valid=valid,
        age=age,
    )


def test_empty_posterior_factory_is_valid_on_requested_dtype_and_device() -> None:
    config = _config()
    belief = empty_object_belief(
        config,
        batch_size=3,
        capacity=5,
        device="cpu",
        dtype=torch.float64,
    )

    assert belief.address_mean.shape == (3, 5, config.address_dim)
    assert belief.address_mean.dtype == torch.float64
    assert not belief.valid.any()
    assert torch.equal(
        belief.geometry_covariance_diag, torch.zeros_like(belief.geometry_covariance_diag)
    )
    prediction = (
        ActionConditionedObjectTransition(config)
        .to(dtype=torch.float64)
        .eval()(
            belief,
            torch.zeros(3, config.action_dim, dtype=torch.float64),
            torch.full((3,), config.reference_delta_t_s, dtype=torch.float64),
        )
    )
    assert torch.equal(prediction.belief.state_mean, torch.zeros_like(belief.state_mean))

    with pytest.raises(ValueError, match="positive"):
        empty_object_belief(config, batch_size=0, capacity=5)
    with pytest.raises(ValueError, match="floating"):
        empty_object_belief(config, batch_size=1, capacity=1, dtype=torch.int64)


def _permute_belief(belief: ObjectBeliefBatch, permutation: torch.Tensor) -> ObjectBeliefBatch:
    return ObjectBeliefBatch(
        address_mean=belief.address_mean[:, permutation],
        content_mean=belief.content_mean[:, permutation],
        geometry_mean=belief.geometry_mean[:, permutation],
        geometry_covariance_diag=belief.geometry_covariance_diag[:, permutation],
        existence_logits=belief.existence_logits[:, permutation],
        visibility_given_existence_logits=belief.visibility_given_existence_logits[:, permutation],
        measurement_age_s=belief.measurement_age_s[:, permutation],
        valid=belief.valid[:, permutation],
        age=belief.age[:, permutation],
    )


def test_prediction_is_near_identity_with_stable_address_and_growing_uncertainty() -> None:
    torch.manual_seed(509)
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    output = transition(prior, torch.zeros(2, 3), torch.full((2,), 0.1))

    torch.testing.assert_close(output.belief.address_mean, prior.address_mean, atol=0.0, rtol=0.0)
    assert (
        output.belief.geometry_covariance_diag[prior.valid]
        > prior.geometry_covariance_diag[prior.valid]
    ).all()
    assert (output.survival_probability[prior.valid] > 0.9).all()
    assert (output.conditional_detection_probability[prior.valid] > 0.8).all()
    assert torch.isfinite(output.survival_logits).all()
    assert output.dynamic_delta[prior.valid].abs().max() < 0.1
    assert torch.equal(output.belief.state_mean[~prior.valid], torch.zeros(3, 6))
    assert torch.equal(output.belief.geometry_covariance_diag[~prior.valid], torch.zeros(3, 2))
    assert torch.equal(output.belief.age[prior.valid], prior.age[prior.valid] + 1)
    torch.testing.assert_close(
        output.belief.measurement_age_s[prior.valid],
        prior.measurement_age_s[prior.valid] + 0.1,
    )


def test_transition_is_object_permutation_equivariant() -> None:
    torch.manual_seed(521)
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    action = torch.randn(2, 3)
    permutation = torch.tensor([2, 0, 3, 1])
    delta_t_s = torch.full((2,), 0.1)
    expected = transition(prior, action, delta_t_s)
    actual = transition(_permute_belief(prior, permutation), action, delta_t_s)

    torch.testing.assert_close(actual.belief.state_mean, expected.belief.state_mean[:, permutation])
    torch.testing.assert_close(
        actual.belief.geometry_covariance_diag,
        expected.belief.geometry_covariance_diag[:, permutation],
    )
    torch.testing.assert_close(
        actual.survival_probability, expected.survival_probability[:, permutation]
    )
    torch.testing.assert_close(actual.survival_logits, expected.survival_logits[:, permutation])
    torch.testing.assert_close(
        actual.conditional_detection_logits,
        expected.conditional_detection_logits[:, permutation],
    )
    torch.testing.assert_close(
        actual.detectability_if_detected_logits,
        expected.detectability_if_detected_logits[:, permutation],
    )
    torch.testing.assert_close(
        actual.detectability_if_missed_logits,
        expected.detectability_if_missed_logits[:, permutation],
    )
    torch.testing.assert_close(
        actual.belief.visibility_given_existence_logits,
        expected.belief.visibility_given_existence_logits[:, permutation],
    )
    torch.testing.assert_close(
        actual.belief.measurement_age_s,
        expected.belief.measurement_age_s[:, permutation],
    )


def test_detection_probability_is_identifiable_markov_mixture() -> None:
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    previous_visibility = torch.tensor(
        [[0.999, 0.001, 0.4, 0.0], [0.8, 0.2, 0.0, 0.0]],
        dtype=prior.address_mean.dtype,
    )
    prior = ObjectBeliefBatch(
        address_mean=prior.address_mean,
        content_mean=prior.content_mean,
        geometry_mean=prior.geometry_mean,
        geometry_covariance_diag=prior.geometry_covariance_diag,
        existence_logits=prior.existence_logits,
        visibility_given_existence_logits=torch.where(
            prior.valid,
            torch.logit(previous_visibility.clamp(1e-6, 1.0 - 1e-6)),
            torch.zeros_like(previous_visibility),
        ),
        measurement_age_s=prior.measurement_age_s,
        valid=prior.valid,
        age=prior.age,
    )
    detected_branch = 0.8
    missed_branch = 0.2
    with torch.no_grad():
        transition.detectability_if_detected_head.weight.zero_()
        transition.detectability_if_detected_head.bias.fill_(
            torch.logit(torch.tensor(detected_branch))
        )
        transition.detectability_if_missed_head.weight.zero_()
        transition.detectability_if_missed_head.bias.fill_(torch.logit(torch.tensor(missed_branch)))

    output = transition(prior, torch.zeros(2, 3), torch.full((2,), 0.1))

    actual = output.conditional_detection_probability
    expected = previous_visibility * detected_branch + (1.0 - previous_visibility) * missed_branch
    torch.testing.assert_close(
        actual[prior.valid],
        expected[prior.valid],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(actual[~prior.valid], torch.zeros_like(actual[~prior.valid]))
    assert not hasattr(transition, "visibility_persistence_head")
    assert not hasattr(transition, "visibility_reappearance_head")


def test_previous_detectability_only_selects_kernel_mixture_not_branch_context() -> None:
    torch.manual_seed(522)
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    changed = ObjectBeliefBatch(
        address_mean=prior.address_mean,
        content_mean=prior.content_mean,
        geometry_mean=prior.geometry_mean,
        geometry_covariance_diag=prior.geometry_covariance_diag,
        existence_logits=prior.existence_logits,
        visibility_given_existence_logits=torch.where(
            prior.valid,
            -prior.visibility_given_existence_logits,
            prior.visibility_given_existence_logits,
        ),
        measurement_age_s=prior.measurement_age_s,
        valid=prior.valid,
        age=prior.age,
    )
    action = torch.randn(2, 3)
    delta_t_s = torch.full((2,), 0.1)

    expected = transition(prior, action, delta_t_s)
    actual = transition(changed, action, delta_t_s)

    torch.testing.assert_close(
        actual.detectability_if_detected_logits,
        expected.detectability_if_detected_logits,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual.detectability_if_missed_logits,
        expected.detectability_if_missed_logits,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(actual.dynamic_delta, expected.dynamic_delta, atol=0.0, rtol=0.0)
    assert not torch.equal(
        actual.conditional_detection_probability,
        expected.conditional_detection_probability,
    )


def test_previous_executed_action_conditions_prediction() -> None:
    torch.manual_seed(523)
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    delta_t_s = torch.full((2,), 0.1)
    zero = transition(prior, torch.zeros(2, 3), delta_t_s)
    moved = transition(prior, torch.ones(2, 3), delta_t_s)

    assert not torch.equal(zero.dynamic_delta, moved.dynamic_delta)
    assert not torch.equal(zero.process_variance, moved.process_variance)


def test_integer_step_age_is_diagnostic_not_a_transition_feature() -> None:
    torch.manual_seed(525)
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    older = ObjectBeliefBatch(
        address_mean=prior.address_mean,
        content_mean=prior.content_mean,
        geometry_mean=prior.geometry_mean,
        geometry_covariance_diag=prior.geometry_covariance_diag,
        existence_logits=prior.existence_logits,
        visibility_given_existence_logits=prior.visibility_given_existence_logits,
        measurement_age_s=prior.measurement_age_s,
        valid=prior.valid,
        age=prior.age + 100 * prior.valid,
    )
    action = torch.randn(2, 3)
    delta_t_s = torch.full((2,), 0.1)

    expected = transition(prior, action, delta_t_s)
    actual = transition(older, action, delta_t_s)

    torch.testing.assert_close(actual.dynamic_delta, expected.dynamic_delta, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        actual.process_variance,
        expected.process_variance,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual.survival_probability,
        expected.survival_probability,
        atol=0.0,
        rtol=0.0,
    )


def test_measurement_age_modulates_only_the_missed_detectability_hazard() -> None:
    torch.manual_seed(526)
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    older = replace(
        prior,
        measurement_age_s=prior.measurement_age_s + 10.0 * prior.valid,
    )
    with torch.no_grad():
        transition.missed_duration_logit_slope.fill_(-1.0)
    action = torch.randn(2, 3)
    delta_t_s = torch.full((2,), 0.1)

    expected = transition(prior, action, delta_t_s)
    actual = transition(older, action, delta_t_s)

    torch.testing.assert_close(
        actual.detectability_if_detected_logits,
        expected.detectability_if_detected_logits,
        atol=0.0,
        rtol=0.0,
    )
    assert (
        actual.detectability_if_missed_logits[prior.valid]
        < expected.detectability_if_missed_logits[prior.valid]
    ).all()
    torch.testing.assert_close(actual.dynamic_delta, expected.dynamic_delta, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        actual.process_variance, expected.process_variance, atol=0.0, rtol=0.0
    )


def test_elapsed_time_conditions_prediction_without_changing_reference_period() -> None:
    torch.manual_seed(527)
    transition = ActionConditionedObjectTransition(_config()).eval()
    prior = _prior()
    action = torch.randn(2, 3)

    reference = transition(prior, action, torch.full((2,), 0.1))
    with torch.no_grad():
        weights = torch.arange(
            transition.time_projection.weight.numel(),
            dtype=transition.time_projection.weight.dtype,
        ).reshape_as(transition.time_projection.weight)
        transition.time_projection.weight.copy_(weights / weights.numel())
    unchanged_reference = transition(prior, action, torch.full((2,), 0.1))
    longer = transition(prior, action, torch.full((2,), 0.2))

    torch.testing.assert_close(
        unchanged_reference.belief.state_mean,
        reference.belief.state_mean,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        unchanged_reference.process_variance,
        reference.process_variance,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(longer.dynamic_delta, unchanged_reference.dynamic_delta)
    assert not torch.equal(longer.process_variance, unchanged_reference.process_variance)


@pytest.mark.parametrize(
    ("delta_t_s", "message"),
    [
        (torch.full((2, 1), 0.1), "one floating colocated value"),
        (torch.ones(2, dtype=torch.long), "one floating colocated value"),
        (torch.tensor([0.1, 0.0]), "finite and positive"),
        (torch.tensor([0.1, -0.1]), "finite and positive"),
        (torch.tensor([0.1, float("nan")]), "finite and positive"),
        (torch.tensor([0.1, float("inf")]), "finite and positive"),
    ],
)
def test_transition_rejects_invalid_elapsed_time(
    delta_t_s: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ActionConditionedObjectTransition(_config()).eval()(
            _prior(),
            torch.zeros(2, 3),
            delta_t_s,
        )


@pytest.mark.parametrize(
    "action",
    [
        torch.zeros(2, 3, dtype=torch.float64),
        torch.zeros(2, 3, dtype=torch.long),
    ],
)
def test_transition_rejects_executed_action_dtype_mismatch(action: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="floating dtype and device"):
        ActionConditionedObjectTransition(_config()).eval()(
            _prior(),
            action,
            torch.full((2,), 0.1),
        )


def test_transition_receives_gradient_from_dynamic_and_uncertainty_outputs() -> None:
    torch.manual_seed(563)
    transition = ActionConditionedObjectTransition(_config()).train()
    prior = _prior()
    action = torch.randn(2, 3, requires_grad=True)
    prediction = transition(prior, action, torch.full((2,), 0.1))
    loss = (
        prediction.belief.state_mean.square().mean()
        + prediction.belief.geometry_covariance_diag.mean()
    )
    loss.backward()

    assert action.grad is not None and action.grad.abs().sum() > 0.0
    assert transition.action_projection.weight.grad is not None
    assert transition.layers[0].attention.in_proj_weight.grad is not None
    assert transition.dynamic_head.weight.grad is not None


def test_empty_belief_batch_remains_finite_and_zero() -> None:
    config = _config()
    valid = torch.zeros(2, 3, dtype=torch.bool)
    prior = ObjectBeliefBatch(
        address_mean=torch.zeros(2, 3, 2),
        content_mean=torch.zeros(2, 3, 2),
        geometry_mean=torch.zeros(2, 3, 2),
        geometry_covariance_diag=torch.zeros(2, 3, 2),
        existence_logits=torch.zeros(2, 3),
        visibility_given_existence_logits=torch.zeros(2, 3),
        measurement_age_s=torch.zeros(2, 3),
        valid=valid,
        age=torch.zeros(2, 3, dtype=torch.long),
    )
    output = ActionConditionedObjectTransition(config).eval()(
        prior,
        torch.zeros(2, 3),
        torch.full((2,), config.reference_delta_t_s),
    )

    assert torch.isfinite(output.belief.state_mean).all()
    assert torch.equal(output.belief.state_mean, torch.zeros_like(output.belief.state_mean))
    assert torch.equal(
        output.belief.geometry_covariance_diag,
        torch.zeros_like(output.belief.geometry_covariance_diag),
    )
