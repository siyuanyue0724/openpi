from __future__ import annotations

import inspect
import math
from dataclasses import replace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
discovery_module = pytest.importorskip("picf_next.models.discovery")
filter_module = pytest.importorskip("picf_next.models.filter")
temporal = pytest.importorskip("picf_next.models.temporal")
posterior = pytest.importorskip("picf_next.posterior")

ObjectDiscoveryOutput = discovery_module.ObjectDiscoveryOutput
ObjectExistenceCalibration = discovery_module.ObjectExistenceCalibration
PersistentObjectFilter = filter_module.PersistentObjectFilter
build_object_action_bank = filter_module.build_object_action_bank
marginal_association = filter_module._marginal_association
assemble_marginal_lifecycle = filter_module._assemble_marginal_lifecycle
MarginalAssociation = pytest.importorskip("picf_next.models.marginal").MarginalAssociation
ObjectBeliefBatch = temporal.ObjectBeliefBatch
TemporalFilterConfig = temporal.TemporalFilterConfig
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


def _delta_t() -> torch.Tensor:
    return torch.full((1,), 0.1)


def _belief(
    *,
    valid: torch.Tensor | None = None,
    existence_logits: torch.Tensor | None = None,
    visibility_logits: torch.Tensor | None = None,
    measurement_age_s: torch.Tensor | None = None,
) -> ObjectBeliefBatch:
    if valid is None:
        valid = torch.tensor([[True, True, False]])
    state = torch.tensor(
        [
            [
                [1.0, 0.0, 0.2, 0.3, 0.1, 0.2],
                [0.0, 1.0, 1.2, 1.3, 1.1, 1.2],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    state = state * valid.unsqueeze(-1)
    if existence_logits is None:
        existence_logits = torch.tensor([[6.0, 6.0, 0.0]]) * valid
    if visibility_logits is None:
        visibility_logits = torch.tensor([[6.0, 6.0, 0.0]]) * valid
    if measurement_age_s is None:
        measurement_age_s = torch.zeros(1, 3)
    return ObjectBeliefBatch(
        address_mean=state[..., :2],
        content_mean=state[..., 2:4],
        geometry_mean=state[..., 4:],
        geometry_covariance_diag=torch.full((1, 3, 2), 0.05) * valid.unsqueeze(-1),
        existence_logits=existence_logits,
        visibility_given_existence_logits=visibility_logits,
        measurement_age_s=measurement_age_s * valid,
        valid=valid,
        age=torch.tensor([[4, 7, 0]], dtype=torch.long) * valid,
    )


def _discovery(
    *,
    high: tuple[int, ...] = (0, 1),
    state: torch.Tensor | None = None,
    ownership_logits: torch.Tensor | None = None,
    existence_logits: torch.Tensor | None = None,
) -> ObjectDiscoveryOutput:
    if state is None:
        state = torch.tensor(
            [
                [
                    [5.02, 4.98, 1.21, 1.29, 1.08, 1.22],
                    [0.01, -0.01, 0.19, 0.31, 0.11, 0.19],
                    [9.0] * 6,
                ]
            ]
        )
    if existence_logits is None:
        existence_logits = torch.full((1, 3), -6.0)
        existence_logits[:, list(high)] = 6.0
    if ownership_logits is None:
        ownership_logits = torch.tensor(
            [
                [
                    [-8.0, 8.0, -8.0, -8.0],
                    [8.0, -8.0, -8.0, -8.0],
                    [-8.0, -8.0, 8.0, -8.0],
                    [-torch.inf, -torch.inf, -torch.inf, 0.0],
                ]
            ]
        )
    token_valid = torch.tensor([[True, True, True, False]])
    raw_address = state[..., :2]
    normalized_address = torch.nn.functional.normalize(raw_address, dim=-1)
    fallback = raw_address.new_tensor([1.0, 0.0]).expand_as(raw_address)
    normalized_address = torch.where(
        torch.linalg.vector_norm(raw_address, dim=-1, keepdim=True) > 0.0,
        normalized_address,
        fallback,
    )
    return ObjectDiscoveryOutput(
        query_features=torch.zeros(1, 3, 8),
        address_mean=normalized_address,
        content_mean=state[..., 2:4],
        geometry_mean=state[..., 4:],
        geometry_variance=torch.full((1, 3, 2), 0.05),
        geometry_contract=GEOMETRY,
        existence_logits=existence_logits,
        localization_confidence_logits=torch.full_like(existence_logits, 12.0),
        ownership_logits=ownership_logits,
        ownership=torch.softmax(ownership_logits, dim=-1),
        token_valid=token_valid,
        token_group_id=torch.tensor([[-1, -1, 9, -1]], dtype=torch.long),
        evidence_available=torch.tensor([True]),
        existence_calibration=ObjectExistenceCalibration(),
    )


def _permute_belief(prior: ObjectBeliefBatch, permutation: torch.Tensor) -> ObjectBeliefBatch:
    return ObjectBeliefBatch(
        address_mean=prior.address_mean[:, permutation],
        content_mean=prior.content_mean[:, permutation],
        geometry_mean=prior.geometry_mean[:, permutation],
        geometry_covariance_diag=prior.geometry_covariance_diag[:, permutation],
        existence_logits=prior.existence_logits[:, permutation],
        visibility_given_existence_logits=prior.visibility_given_existence_logits[:, permutation],
        measurement_age_s=prior.measurement_age_s[:, permutation],
        valid=prior.valid[:, permutation],
        age=prior.age[:, permutation],
    )


def _permute_discovery_queries(
    output: ObjectDiscoveryOutput,
    permutation: torch.Tensor,
) -> ObjectDiscoveryOutput:
    context = output.ownership.shape[-1] - 1
    ownership_permutation = torch.cat(
        (permutation, torch.tensor([context], device=permutation.device))
    )
    return replace(
        output,
        query_features=output.query_features[:, permutation],
        address_mean=output.address_mean[:, permutation],
        content_mean=output.content_mean[:, permutation],
        geometry_mean=output.geometry_mean[:, permutation],
        geometry_variance=output.geometry_variance[:, permutation],
        existence_logits=output.existence_logits[:, permutation],
        localization_confidence_logits=output.localization_confidence_logits[:, permutation],
        ownership_logits=output.ownership_logits[..., ownership_permutation],
        ownership=output.ownership[..., ownership_permutation],
    )


def _cast_belief(prior: ObjectBeliefBatch, dtype: torch.dtype) -> ObjectBeliefBatch:
    return ObjectBeliefBatch(
        address_mean=prior.address_mean.to(dtype),
        content_mean=prior.content_mean.to(dtype),
        geometry_mean=prior.geometry_mean.to(dtype),
        geometry_covariance_diag=prior.geometry_covariance_diag.to(dtype),
        existence_logits=prior.existence_logits.to(dtype),
        visibility_given_existence_logits=prior.visibility_given_existence_logits.to(dtype),
        measurement_age_s=prior.measurement_age_s.to(dtype),
        valid=prior.valid,
        age=prior.age,
    )


def _cast_discovery(
    output: ObjectDiscoveryOutput,
    dtype: torch.dtype,
) -> ObjectDiscoveryOutput:
    return replace(
        output,
        query_features=output.query_features.to(dtype),
        address_mean=output.address_mean.to(dtype),
        content_mean=output.content_mean.to(dtype),
        geometry_mean=output.geometry_mean.to(dtype),
        geometry_variance=output.geometry_variance.to(dtype),
        existence_logits=output.existence_logits.to(dtype),
        localization_confidence_logits=output.localization_confidence_logits.to(dtype),
        ownership_logits=output.ownership_logits.to(dtype),
        ownership=output.ownership.to(dtype),
    )


def test_bfloat16_marginals_use_float32_probability_arithmetic() -> None:
    config = _config()
    bf16_prior = _cast_belief(_belief(), torch.bfloat16)
    bf16_discovery = _cast_discovery(_discovery(), torch.bfloat16)
    expected = marginal_association(
        _cast_belief(bf16_prior, torch.float32),
        _cast_discovery(bf16_discovery, torch.float32),
        config,
    )
    actual = marginal_association(bf16_prior, bf16_discovery, config)

    for actual_value, expected_value in (
        (actual[0].match_probability, expected[0].match_probability),
        (actual[0].null_probability, expected[0].null_probability),
        (
            actual[0].unexplained_observation_probability,
            expected[0].unexplained_observation_probability,
        ),
        (actual[1], expected[1]),
        (actual[2], expected[2]),
    ):
        assert actual_value.dtype == torch.float32
        torch.testing.assert_close(actual_value, expected_value, atol=0.0, rtol=0.0)


def test_runtime_address_relation_reuses_calibrated_binding_density_ratio() -> None:
    config = _config()
    valid = torch.tensor([[True, True]])
    prior = ObjectBeliefBatch(
        address_mean=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        content_mean=torch.zeros(1, 2, config.content_dim),
        geometry_mean=torch.zeros(1, 2, config.geometry_dim),
        geometry_covariance_diag=torch.full((1, 2, config.geometry_dim), 0.05),
        existence_logits=torch.full((1, 2), 6.0),
        visibility_given_existence_logits=torch.zeros(1, 2),
        measurement_age_s=torch.zeros(1, 2),
        valid=valid,
        age=torch.zeros(1, 2, dtype=torch.long),
    )
    ownership_logits = torch.tensor([[[12.0, -12.0]]])
    discovery = ObjectDiscoveryOutput(
        query_features=torch.zeros(1, 1, 8),
        address_mean=torch.tensor([[[1.0, 0.0]]]),
        content_mean=torch.zeros(1, 1, config.content_dim),
        geometry_mean=torch.zeros(1, 1, config.geometry_dim),
        geometry_variance=torch.full((1, 1, config.geometry_dim), 0.05),
        geometry_contract=GEOMETRY,
        existence_logits=torch.full((1, 1), 6.0),
        localization_confidence_logits=torch.full((1, 1), 12.0),
        ownership_logits=ownership_logits,
        ownership=torch.softmax(ownership_logits, dim=-1),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        token_group_id=torch.full((1, 1), -1, dtype=torch.long),
        evidence_available=torch.ones(1, dtype=torch.bool),
        existence_calibration=ObjectExistenceCalibration(),
    )

    association, _observation_probability, _birth_odds = marginal_association(
        prior,
        discovery,
        config,
    )

    exact_identity = association.match_probability[0, 0, 0]
    orthogonal_identity = association.match_probability[0, 1, 0]
    # The two rows have identical lifecycle and geometry. The calibrated
    # address relation must therefore make the exact identity overwhelmingly
    # preferable while the variational one-to-one posterior retains finite
    # uncertainty rather than promising equality to the raw edge-odds ratio.
    assert exact_identity > 0.99
    assert orthogonal_identity < 0.01
    assert exact_identity > association.null_probability[0, 0]
    assert orthogonal_identity < association.null_probability[0, 1]


@pytest.mark.parametrize("seed", range(20))
def test_batched_marginal_lifecycle_preserves_probability_contracts(seed: int) -> None:
    generator = torch.Generator().manual_seed(1100 + seed)
    config = _config()
    batch_size, capacity, observations, tokens = 3, 4, 5, 7
    valid = torch.rand(batch_size, capacity, generator=generator) > 0.35
    state = torch.randn(batch_size, capacity, config.state_dim, generator=generator)
    state = state * valid.unsqueeze(-1)
    address = torch.nn.functional.normalize(state[..., : config.address_dim], dim=-1)
    address = address * valid.unsqueeze(-1)
    predicted = ObjectBeliefBatch(
        address_mean=address,
        content_mean=state[
            ...,
            config.address_dim : config.address_dim + config.content_dim,
        ],
        geometry_mean=state[..., config.address_dim + config.content_dim :],
        geometry_covariance_diag=(
            0.05
            + torch.rand(
                batch_size,
                capacity,
                config.geometry_dim,
                generator=generator,
            )
        )
        * valid.unsqueeze(-1),
        existence_logits=torch.randn(batch_size, capacity, generator=generator) * valid,
        visibility_given_existence_logits=torch.randn(
            batch_size,
            capacity,
            generator=generator,
        )
        * valid,
        measurement_age_s=torch.rand(batch_size, capacity, generator=generator) * valid,
        valid=valid,
        age=torch.randint(0, 20, (batch_size, capacity), generator=generator) * valid,
    )
    observation_state = torch.randn(
        batch_size,
        observations,
        config.state_dim,
        generator=generator,
    )
    ownership_logits = torch.randn(
        batch_size,
        tokens,
        observations + 1,
        generator=generator,
    )
    token_valid = torch.ones(batch_size, tokens, dtype=torch.bool)
    discovery = ObjectDiscoveryOutput(
        query_features=torch.randn(
            batch_size,
            observations,
            8,
            generator=generator,
        ),
        address_mean=torch.nn.functional.normalize(
            observation_state[..., : config.address_dim],
            dim=-1,
        ),
        content_mean=observation_state[
            ...,
            config.address_dim : config.address_dim + config.content_dim,
        ],
        geometry_mean=observation_state[..., config.address_dim + config.content_dim :],
        geometry_variance=0.05
        + torch.rand(
            batch_size,
            observations,
            config.geometry_dim,
            generator=generator,
        ),
        geometry_contract=GEOMETRY,
        existence_logits=torch.randn(batch_size, observations, generator=generator),
        localization_confidence_logits=torch.full((batch_size, observations), 12.0),
        ownership_logits=ownership_logits,
        ownership=torch.softmax(ownership_logits, dim=-1),
        token_valid=token_valid,
        token_group_id=torch.full((batch_size, tokens), -1, dtype=torch.long),
        evidence_available=torch.ones(batch_size, dtype=torch.bool),
        existence_calibration=ObjectExistenceCalibration(),
    )

    model = PersistentObjectFilter(config).eval()
    actual = model(
        predicted,
        discovery,
        torch.zeros(batch_size, config.action_dim),
        torch.full((batch_size,), config.reference_delta_t_s),
    )

    torch.testing.assert_close(
        actual.null_probability + actual.match_probability.sum(dim=-1),
        torch.ones_like(actual.null_probability),
        atol=2e-6,
        rtol=2e-6,
    )
    assert (actual.match_probability.sum(dim=1) <= 1.0 + 2e-6).all()
    torch.testing.assert_close(
        actual.ownership.float().sum(dim=-1),
        torch.ones_like(actual.ownership[..., 0], dtype=torch.float32),
        atol=2e-6,
        rtol=2e-6,
    )
    occupied = (
        (actual.event_type == posterior.MATCH_EVENT)
        | (actual.event_type == posterior.MISS_EVENT)
        | (actual.event_type == posterior.BIRTH_EVENT)
    )
    assert torch.equal(occupied, actual.belief.valid)
    assert torch.isfinite(actual.belief.state_mean).all()
    assert torch.isfinite(actual.belief.existence).all()
    for mapping in actual.observation_to_posterior.tolist():
        mapped_rows = [row for row in mapping if row >= 0]
        assert len(mapped_rows) == len(set(mapped_rows))


def test_filter_matches_transient_queries_to_persistent_rows_and_remaps_ownership() -> None:
    torch.manual_seed(601)
    model = PersistentObjectFilter(_config()).eval()
    output = model(_belief(), _discovery(), torch.zeros(1, 3), _delta_t())

    assert output.observation_to_posterior.tolist() == [[1, 0, 2]]
    assert output.event_type[0, :2].tolist() == [posterior.MATCH_EVENT, posterior.MATCH_EVENT]
    assert output.event_type[0, 2] == posterior.BIRTH_EVENT
    assert not output.map_present[0, 2]
    assert output.ownership[0, 0, 0] > 0.9
    assert output.ownership[0, 1, 1] > 0.9
    assert output.ownership[0, 0, -1] > 0.0
    assert output.ownership[0, 1, -1] > 0.0
    assert output.ownership[0, 2, -1] > 0.99
    torch.testing.assert_close(output.ownership.sum(dim=-1), torch.ones(1, 4))


def test_low_existence_query_mass_returns_to_context_without_token_loss() -> None:
    torch.manual_seed(607)
    model = PersistentObjectFilter(_config()).eval()
    output = model(_belief(), _discovery(high=(0, 1)), torch.zeros(1, 3), _delta_t())

    assert output.observation_to_posterior[0, 2] == 2
    assert not output.map_present[0, 2]
    assert not build_object_action_bank(output).valid[0, 2]
    assert output.ownership[0, 2, -1] > 0.99
    assert torch.equal(output.ownership[0, 3, :-1], torch.zeros(3))
    assert output.ownership[0, 3, -1] == 1.0


def test_low_localization_confidence_suppresses_a_coherent_wrong_measurement() -> None:
    prior = _belief()
    confident = _discovery()
    unreliable = replace(
        confident,
        localization_confidence_logits=torch.full_like(
            confident.localization_confidence_logits,
            -12.0,
        ),
    )

    confident_association, confident_probability, _ = marginal_association(
        prior,
        confident,
        _config(),
    )
    unreliable_association, unreliable_probability, _ = marginal_association(
        prior,
        unreliable,
        _config(),
    )

    torch.testing.assert_close(
        unreliable_probability,
        unreliable.existence.float() * unreliable.localization_confidence,
    )
    assert (unreliable_probability < confident_probability * 1e-4).all()
    assert (
        unreliable_association.match_probability.sum()
        < confident_association.match_probability.sum()
    )


def test_joint_map_can_match_low_confidence_query_to_strong_persistent_object() -> None:
    """A detection threshold must not run before temporal association."""

    valid = torch.tensor([[True, False, False]])
    prior = _belief(
        valid=valid,
        existence_logits=torch.tensor([[6.0, 0.0, 0.0]]),
        visibility_logits=torch.tensor([[6.0, 0.0, 0.0]]),
    )
    state = torch.tensor(
        [
            [
                [0.0, 0.0, 0.2, 0.3, 0.1, 0.2],
                [50.0] * 6,
                [100.0] * 6,
            ]
        ]
    )
    physical_existence = 0.49
    calibration = ObjectExistenceCalibration()
    weighted_logit = math.log(physical_existence / (1.0 - physical_existence)) - math.log(
        calibration.unmatched_query_weight
    )
    discovery = _discovery(
        state=state,
        existence_logits=torch.tensor([[weighted_logit, -6.0, -6.0]]),
    )

    output = PersistentObjectFilter(_config()).eval()(
        prior,
        discovery,
        torch.zeros(1, 3),
        _delta_t(),
    )

    assert discovery.existence[0, 0] < 0.5
    assert output.observation_to_posterior[0, 0] == 0
    assert output.event_type[0, 0] == posterior.MATCH_EVENT
    assert output.ownership[0, 1, 0] > output.ownership[0, 1, -1]
    assert output.birth_probability[0, 0] < 0.5


def test_unobserved_high_existence_object_survives_with_miss_event() -> None:
    valid = torch.tensor([[True, False, False]])
    prior = _belief(
        valid=valid,
        existence_logits=torch.tensor([[6.0, 0.0, 0.0]]),
        visibility_logits=torch.tensor([[-6.0, 0.0, 0.0]]),
    )
    output = PersistentObjectFilter(_config()).eval()(
        prior, _discovery(high=()), torch.zeros(1, 3), _delta_t()
    )

    assert output.belief.valid[0, 0]
    assert output.event_type[0, 0] == posterior.MISS_EVENT
    expected_visibility = output.match_probability[0, 0].sum()
    torch.testing.assert_close(output.belief.visibility[0, 0], expected_visibility)
    assert output.belief.visibility[0, 0] > 0.0
    assert output.belief.visibility[0, 0] < output.null_probability[0, 0]
    predicted = output.prior_prediction.belief
    missed_alive = predicted.existence[0, 0] - predicted.visibility[0, 0]
    existence_given_no_detection = missed_alive / (missed_alive + 1.0 - predicted.existence[0, 0])
    expected_existence = (
        output.match_probability[0, 0].sum()
        + output.null_probability[0, 0] * existence_given_no_detection
    )
    torch.testing.assert_close(output.belief.existence[0, 0], expected_existence)

    action_bank = build_object_action_bank(output)
    assert action_bank.valid[0, 0]
    torch.testing.assert_close(action_bank.address[0, 0], output.belief.address_mean[0, 0])
    assert torch.isfinite(action_bank.value[0, 0]).all()
    torch.testing.assert_close(
        action_bank.log_prior[0, 0],
        output.belief.existence[0, 0].log(),
    )


def test_reappearance_after_a_miss_recovers_the_same_persistent_row() -> None:
    """The production transition must not make a miss an identity sink."""

    valid = torch.tensor([[True, False, False]])
    prior = _belief(
        valid=valid,
        existence_logits=torch.tensor([[6.0, 0.0, 0.0]]),
        visibility_logits=torch.tensor([[-6.0, 0.0, 0.0]]),
    )
    model = PersistentObjectFilter(_config()).eval()
    missed = model(prior, _discovery(high=()), torch.zeros(1, 3), _delta_t())
    assert missed.event_type[0, 0] == posterior.MISS_EVENT
    torch.testing.assert_close(
        missed.belief.visibility[0, 0],
        missed.match_probability[0, 0].sum(),
    )
    assert missed.belief.visibility[0, 0] > 0.0
    assert missed.belief.visibility[0, 0] < missed.null_probability[0, 0]

    state = torch.full((1, 3, 6), 100.0)
    state[0, 0] = missed.belief.state_mean[0, 0]
    reappeared = model(
        missed.belief,
        _discovery(high=(0,), state=state),
        torch.zeros(1, 3),
        _delta_t(),
    )

    assert reappeared.event_type[0, 0] == posterior.MATCH_EVENT
    assert reappeared.observation_to_posterior[0, 0] == 0
    assert not reappeared.map_present[0, 1:].any()
    assert not build_object_action_bank(reappeared).valid[0, 1:].any()
    assert reappeared.belief.age[0, 0] > missed.belief.age[0, 0]


def test_birth_uses_an_explicit_prior_distinct_from_observation_objectness() -> None:
    config = _config()
    prior = _belief(valid=torch.zeros(1, 3, dtype=torch.bool))
    discovery = _discovery(existence_logits=torch.zeros(1, 3), high=())

    output = PersistentObjectFilter(config).eval()(prior, discovery, torch.zeros(1, 3), _delta_t())

    assert output.belief.valid.all()
    assert not output.map_present.any()
    assert not build_object_action_bank(output).valid.any()


@pytest.mark.parametrize("offset, expects_birth", [(-0.01, False), (0.01, True)])
def test_empty_prior_birth_clutter_boundary_matches_probability_model(
    offset: float,
    expects_birth: bool,
) -> None:
    """Lock the exact birth-vs-clutter calibration implied by the MAP costs."""

    config = _config()
    threshold = _discovery().existence_calibration.training_probability_at_half_posterior
    objectness = threshold + offset
    candidate_logit = math.log(objectness / (1.0 - objectness))
    discovery = _discovery(
        high=(),
        existence_logits=torch.tensor([[candidate_logit, -12.0, -12.0]]),
    )
    prior = _belief(valid=torch.zeros(1, 3, dtype=torch.bool))

    output = PersistentObjectFilter(config).eval()(
        prior,
        discovery,
        torch.zeros(1, 3),
        _delta_t(),
    )

    assert output.belief.valid[0, 0]
    assert bool(output.map_present[0, 0]) is expects_birth
    assert bool(build_object_action_bank(output).valid[0, 0]) is expects_birth
    assert not output.map_present[0, 1:].any()


def test_unobserved_low_existence_component_is_retained_but_not_extracted() -> None:
    valid = torch.tensor([[True, False, False]])
    prior = _belief(
        valid=valid,
        existence_logits=torch.tensor([[-6.0, 0.0, 0.0]]),
        visibility_logits=torch.tensor([[-6.0, 0.0, 0.0]]),
    )
    output = PersistentObjectFilter(_config()).eval()(
        prior, _discovery(high=()), torch.zeros(1, 3), _delta_t()
    )

    assert output.belief.valid[0, 0]
    assert output.event_type[0, 0] == posterior.MISS_EVENT
    assert not output.map_present[0, 0]
    assert not build_object_action_bank(output).valid[0, 0]
    torch.testing.assert_close(output.belief.address_mean[0, 0], prior.address_mean[0, 0])


def test_tentative_component_reassociates_without_a_second_birth() -> None:
    """A below-MAP Bernoulli remains available for later identity recovery."""

    valid = torch.tensor([[True, False, False]])
    prior = _belief(
        valid=valid,
        existence_logits=torch.tensor([[-0.4, 0.0, 0.0]]),
        visibility_logits=torch.tensor([[-6.0, 0.0, 0.0]]),
    )
    model = PersistentObjectFilter(_config()).eval()
    missed = model(prior, _discovery(high=()), torch.zeros(1, 3), _delta_t())

    assert missed.belief.valid[0, 0]
    assert not missed.map_present[0, 0]
    assert missed.event_type[0, 0] == posterior.MISS_EVENT

    state = torch.full((1, 3, 6), 100.0)
    state[0, 0] = missed.belief.state_mean[0, 0]
    reappeared = model(
        missed.belief,
        _discovery(high=(0,), state=state),
        torch.zeros(1, 3),
        _delta_t(),
    )

    assert reappeared.event_type[0, 0] == posterior.MATCH_EVENT
    assert reappeared.observation_to_posterior[0, 0] == 0
    assert not reappeared.born[0, 0]
    assert reappeared.map_present[0, 0]
    assert build_object_action_bank(reappeared).valid[0, 0]


def test_recurrent_sub_map_birth_is_stored_as_a_tentative_component() -> None:
    """Recurrent birth odds affect probability, not whether a hypothesis exists."""

    valid = torch.tensor([[True, False, False]])
    prior = _belief(
        valid=valid,
        existence_logits=torch.tensor([[6.0, 0.0, 0.0]]),
        visibility_logits=torch.tensor([[-6.0, 0.0, 0.0]]),
    )
    state = torch.full((1, 3, 6), 100.0)
    physical_existence = 0.8
    calibration = ObjectExistenceCalibration()
    weighted_logit = math.log(physical_existence / (1.0 - physical_existence)) - math.log(
        calibration.unmatched_query_weight
    )
    discovery = _discovery(
        high=(),
        state=state,
        existence_logits=torch.tensor([[weighted_logit, -12.0, -12.0]]),
    )

    output = PersistentObjectFilter(_config()).eval()(
        prior,
        discovery,
        torch.zeros(1, 3),
        _delta_t(),
    )

    assert output.birth_probability[0, 0] < 0.5
    assert output.born[0, 1]
    assert output.belief.valid[0, 1]
    assert not output.map_present[0, 1]
    assert output.observation_to_posterior[0, 0] == 1
    assert not build_object_action_bank(output).valid[0, 1]


def test_birth_competes_with_aggregate_existing_origin_probability() -> None:
    """Mutually exclusive weak matches must jointly suppress a duplicate birth."""

    config = _config()
    predicted = _belief(
        valid=torch.tensor([[True, True, False]]),
        existence_logits=torch.tensor([[6.0, 6.0, 0.0]]),
        visibility_logits=torch.tensor([[0.0, 0.0, 0.0]]),
    )
    discovery = _discovery(high=(0,))
    match_probability = torch.zeros(1, 3, 3)
    match_probability[0, 0, 0] = 0.3
    match_probability[0, 1, 0] = 0.3
    association = MarginalAssociation(
        match_probability=match_probability,
        null_probability=torch.tensor([[0.7, 0.7, 1.0]]),
        unexplained_observation_probability=torch.tensor([[0.4, 1.0, 1.0]]),
        convergence_residual=torch.zeros(1),
    )

    lifecycle = assemble_marginal_lifecycle(
        predicted,
        discovery,
        association,
        observation_probability=torch.tensor([[1.0, 0.0, 0.0]]),
        birth_odds=torch.ones(1, 3),
        config=config,
    )

    torch.testing.assert_close(lifecycle.birth_probability[0, 0], torch.tensor(0.4))
    assert lifecycle.birth_probability[0, 0] > match_probability[0, :, 0].max()
    assert lifecycle.birth_probability[0, 0] < match_probability[0, :, 0].sum()
    assert not lifecycle.born.any()
    assert lifecycle.belief.valid[0, :2].all()
    assert not lifecycle.belief.valid[0, 2]


def test_marginal_correction_moment_matches_measurement_age() -> None:
    config = _config()
    predicted = _belief(
        valid=torch.tensor([[True, True, False]]),
        existence_logits=torch.tensor([[6.0, 6.0, 0.0]]),
        visibility_logits=torch.zeros(1, 3),
        measurement_age_s=torch.tensor([[2.0, 4.0, 0.0]]),
    )
    discovery = _discovery(high=(0,))
    match_probability = torch.zeros(1, 3, 3)
    match_probability[0, 0, 0] = 0.4
    association = MarginalAssociation(
        match_probability=match_probability,
        null_probability=torch.tensor([[0.6, 1.0, 1.0]]),
        unexplained_observation_probability=torch.tensor([[0.6, 1.0, 1.0]]),
        convergence_residual=torch.zeros(1),
    )

    lifecycle = assemble_marginal_lifecycle(
        predicted,
        discovery,
        association,
        observation_probability=torch.tensor([[1.0, 0.0, 0.0]]),
        birth_odds=torch.zeros(1, 3),
        config=config,
    )

    existence = predicted.existence[0, 0]
    detection = torch.sigmoid(predicted.visibility_given_existence_logits[0, 0])
    existence_given_miss = existence * (1.0 - detection) / (1.0 - existence * detection)
    missed_alive_mass = 0.6 * existence_given_miss
    expected_null_weight = missed_alive_mass / (0.4 + missed_alive_mass)
    torch.testing.assert_close(
        lifecycle.belief.measurement_age_s[0, 0],
        expected_null_weight * predicted.measurement_age_s[0, 0],
    )
    torch.testing.assert_close(
        lifecycle.belief.measurement_age_s[0, 1],
        predicted.measurement_age_s[0, 1],
    )


def test_visible_candidate_births_into_free_row() -> None:
    prior = _belief(valid=torch.zeros(1, 3, dtype=torch.bool))
    output = PersistentObjectFilter(_config()).eval()(
        prior, _discovery(high=(0,)), torch.zeros(1, 3), _delta_t()
    )

    assert output.event_type[0, 0] == posterior.BIRTH_EVENT
    assert output.belief.valid[0, 0]
    assert output.belief.age[0, 0] == 0
    assert output.observation_to_posterior[0, 0] == 0
    torch.testing.assert_close(output.belief.address_mean[0, 0], _discovery().address_mean[0, 0])
    torch.testing.assert_close(output.belief.existence[0, 0], _discovery().existence[0, 0])


def test_capacity_map_replaces_weaker_missed_object_with_stronger_birth() -> None:
    valid = torch.ones(1, 3, dtype=torch.bool)
    prior = _belief(
        valid=valid,
        existence_logits=torch.tensor([[6.0, 6.0, -2.0]]),
        visibility_logits=torch.full((1, 3), -6.0),
    )
    far_state = torch.full((1, 3, 6), 100.0)
    output = PersistentObjectFilter(_config()).eval()(
        prior,
        _discovery(high=(0,), state=far_state),
        torch.zeros(1, 3),
        _delta_t(),
    )

    assert output.belief.valid.sum() == 3
    assert output.event_type[0, :2].tolist() == [posterior.MISS_EVENT, posterior.MISS_EVENT]
    assert output.event_type[0, 2] == posterior.BIRTH_EVENT
    assert output.observation_to_posterior[0, 0] == 2
    torch.testing.assert_close(
        output.belief.address_mean[0, 2],
        _discovery(state=far_state).address_mean[0, 0],
    )

    action_bank = build_object_action_bank(output)
    posterior_dynamic = torch.cat(
        (output.belief.content_mean[0, 2], output.belief.geometry_mean[0, 2])
    )
    torch.testing.assert_close(action_bank.value[0, 2, :4], posterior_dynamic)
    assert not torch.equal(
        output.prior_prediction.belief.state_mean[0, 2, 2:],
        posterior_dynamic,
    )


def test_existing_object_row_permutation_is_equivariant() -> None:
    torch.manual_seed(613)
    model = PersistentObjectFilter(_config()).eval()
    prior = _belief()
    discovery = _discovery()
    permutation = torch.tensor([1, 0, 2])
    expected = model(prior, discovery, torch.zeros(1, 3), _delta_t())
    actual = model(
        _permute_belief(prior, permutation),
        discovery,
        torch.zeros(1, 3),
        _delta_t(),
    )

    torch.testing.assert_close(actual.belief.state_mean, expected.belief.state_mean[:, permutation])
    torch.testing.assert_close(actual.innovation, expected.innovation[:, permutation])
    torch.testing.assert_close(actual.ownership[..., :-1], expected.ownership[..., permutation])
    torch.testing.assert_close(actual.ownership[..., -1], expected.ownership[..., -1])


def test_discovery_query_permutation_is_equivariant_end_to_end() -> None:
    torch.manual_seed(615)
    model = PersistentObjectFilter(_config()).eval()
    prior = _belief()
    discovery = _discovery()
    permutation = torch.tensor([2, 0, 1])
    expected = model(prior, discovery, torch.zeros(1, 3), _delta_t())
    actual = model(
        prior,
        _permute_discovery_queries(discovery, permutation),
        torch.zeros(1, 3),
        _delta_t(),
    )

    torch.testing.assert_close(actual.belief.state_mean, expected.belief.state_mean)
    torch.testing.assert_close(
        actual.belief.geometry_covariance_diag,
        expected.belief.geometry_covariance_diag,
    )
    torch.testing.assert_close(actual.belief.existence, expected.belief.existence)
    torch.testing.assert_close(actual.ownership, expected.ownership)
    torch.testing.assert_close(
        actual.match_probability,
        expected.match_probability[:, :, permutation],
    )
    torch.testing.assert_close(actual.birth_probability, expected.birth_probability[:, permutation])
    torch.testing.assert_close(actual.null_probability, expected.null_probability)
    assert torch.equal(
        actual.observation_to_posterior,
        expected.observation_to_posterior[:, permutation],
    )


def test_diagnostic_event_projection_cannot_change_state_ownership_or_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(616)
    model = PersistentObjectFilter(_config()).eval()
    prior = _belief()
    discovery = _discovery()
    expected = model(prior, discovery, torch.zeros(1, 3), _delta_t())
    expected_bank = build_object_action_bank(expected)

    def arbitrary_diagnostics(*args, **kwargs):
        del args, kwargs
        return (
            torch.full_like(expected.observation_to_posterior, -1),
            torch.full_like(expected.event_type, posterior.BIRTH_EVENT),
        )

    monkeypatch.setattr(
        filter_module,
        "_diagnostic_lifecycle_projection",
        arbitrary_diagnostics,
    )
    actual = model(prior, discovery, torch.zeros(1, 3), _delta_t())
    actual_bank = build_object_action_bank(actual)

    assert not torch.equal(actual.event_type, expected.event_type)
    assert not torch.equal(actual.observation_to_posterior, expected.observation_to_posterior)
    assert torch.equal(actual.born, expected.born)
    torch.testing.assert_close(actual.belief.state_mean, expected.belief.state_mean)
    torch.testing.assert_close(
        actual.belief.geometry_covariance_diag,
        expected.belief.geometry_covariance_diag,
    )
    torch.testing.assert_close(actual.ownership, expected.ownership)
    torch.testing.assert_close(actual_bank.address, expected_bank.address)
    torch.testing.assert_close(actual_bank.value, expected_bank.value)
    torch.testing.assert_close(actual_bank.log_prior, expected_bank.log_prior)
    assert torch.equal(actual_bank.valid, expected_bank.valid)


def test_selected_filter_paths_retain_gradient() -> None:
    torch.manual_seed(617)
    model = PersistentObjectFilter(_config()).train()
    action = torch.randn(1, 3, requires_grad=True)
    output = model(_belief(), _discovery(), action, _delta_t())
    loss = output.belief.state_mean.square().mean() + output.ownership.square().mean()
    loss.backward()

    assert action.grad is not None and action.grad.abs().sum() > 0.0
    assert model.transition.action_projection.weight.grad is not None
    assert output.match_probability.grad_fn is not None
    assert not hasattr(model, "corrector")


def test_integer_step_age_does_not_leak_into_action_evidence() -> None:
    torch.manual_seed(619)
    model = PersistentObjectFilter(_config()).eval()
    prior = _belief()
    older_prior = replace(prior, age=prior.age + 100 * prior.valid)

    expected = build_object_action_bank(model(prior, _discovery(), torch.zeros(1, 3), _delta_t()))
    actual = build_object_action_bank(
        model(older_prior, _discovery(), torch.zeros(1, 3), _delta_t())
    )

    torch.testing.assert_close(actual.address, expected.address, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual.value, expected.value, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual.log_prior, expected.log_prior, atol=0.0, rtol=0.0)
    assert torch.equal(actual.valid, expected.valid)


def test_runtime_filter_has_no_task_or_training_target_argument() -> None:
    parameters = tuple(inspect.signature(PersistentObjectFilter.forward).parameters)
    assert parameters == (
        "self",
        "prior",
        "discovery",
        "previous_executed_action",
        "delta_t_s",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "geometry_variance",
            torch.zeros(1, 3, 2),
            "variance is below",
        ),
        (
            "evidence_available",
            torch.tensor([False]),
            "availability must equal",
        ),
        (
            "localization_confidence_logits",
            torch.full((1, 3), float("nan")),
            "NaN or infinity",
        ),
    ],
)
def test_filter_rejects_malformed_discovery_contract(
    field: str, value: torch.Tensor, message: str
) -> None:
    discovery = replace(_discovery(), **{field: value})
    with pytest.raises(ValueError, match=message):
        PersistentObjectFilter(_config()).eval()(
            _belief(),
            discovery,
            torch.zeros(1, 3),
            _delta_t(),
        )

    ownership = _discovery().ownership.clone()
    ownership[:, 0] *= 2.0
    with pytest.raises(ValueError, match="per-token simplex"):
        PersistentObjectFilter(_config()).eval()(
            _belief(),
            replace(_discovery(), ownership=ownership),
            torch.zeros(1, 3),
            _delta_t(),
        )
