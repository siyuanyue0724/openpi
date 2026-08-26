from __future__ import annotations

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
discovery_module = pytest.importorskip("picf_next.models.discovery")
filter_module = pytest.importorskip("picf_next.models.filter")
temporal_module = pytest.importorskip("picf_next.models.temporal")

ObjectDiscoveryOutput = discovery_module.ObjectDiscoveryOutput
ObjectExistenceCalibration = discovery_module.ObjectExistenceCalibration
PersistentObjectFilter = filter_module.PersistentObjectFilter
ObjectBeliefBatch = temporal_module.ObjectBeliefBatch
TemporalFilterConfig = temporal_module.TemporalFilterConfig
GEOMETRY = synthetic_geometry_contract(5)


def _config() -> TemporalFilterConfig:
    return TemporalFilterConfig(
        address_dim=3,
        content_dim=4,
        geometry_dim=5,
        geometry_contract=GEOMETRY,
        action_dim=7,
        reference_delta_t_s=0.1,
        hidden_dim=24,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )


def _empty_belief(batch_size: int, capacity: int, config: TemporalFilterConfig):
    valid = torch.zeros(batch_size, capacity, dtype=torch.bool)
    return ObjectBeliefBatch(
        address_mean=torch.zeros(batch_size, capacity, config.address_dim),
        content_mean=torch.zeros(batch_size, capacity, config.content_dim),
        geometry_mean=torch.zeros(batch_size, capacity, config.geometry_dim),
        geometry_covariance_diag=torch.zeros(batch_size, capacity, config.geometry_dim),
        existence_logits=torch.zeros(batch_size, capacity),
        visibility_given_existence_logits=torch.zeros(batch_size, capacity),
        measurement_age_s=torch.zeros(batch_size, capacity),
        valid=valid,
        age=torch.zeros(batch_size, capacity, dtype=torch.long),
    )


def _discovery(
    state: torch.Tensor,
    *,
    token_count: int,
    generator: torch.Generator,
) -> ObjectDiscoveryOutput:
    batch_size, query_count, _state_dim = state.shape
    geometry_mean = state[..., 7:]
    logits = torch.randn(
        batch_size,
        token_count,
        query_count + 1,
        generator=generator,
    )
    ownership = torch.softmax(logits, dim=-1)
    valid = torch.ones(batch_size, token_count, dtype=torch.bool)
    return ObjectDiscoveryOutput(
        query_features=torch.randn(
            batch_size,
            query_count,
            16,
            generator=generator,
        ),
        address_mean=torch.nn.functional.normalize(state[..., :3], dim=-1),
        content_mean=state[..., 3:7],
        geometry_mean=geometry_mean,
        geometry_variance=torch.rand(
            batch_size,
            query_count,
            geometry_mean.shape[-1],
            generator=generator,
        )
        * 0.02
        + 1e-4,
        geometry_contract=GEOMETRY,
        existence_logits=torch.full((batch_size, query_count), 8.0),
        localization_confidence_logits=torch.full((batch_size, query_count), 12.0),
        ownership_logits=logits,
        ownership=ownership,
        token_valid=valid,
        token_group_id=torch.full(
            (batch_size, token_count),
            -1,
            dtype=torch.long,
        ),
        evidence_available=torch.ones(batch_size, dtype=torch.bool),
        existence_calibration=ObjectExistenceCalibration(),
    )


def test_randomized_persistent_filter_remains_finite_and_normalized() -> None:
    """Exercise repeated predict/associate/correct cycles beyond one-step tests."""

    generator = torch.Generator().manual_seed(20260715)
    config = _config()
    batch_size = 2
    capacity = 5
    model = PersistentObjectFilter(config).eval()
    belief = _empty_belief(batch_size, capacity, config)
    latent_state = torch.randn(
        batch_size,
        capacity,
        config.state_dim,
        generator=generator,
    )

    with torch.no_grad():
        for _ in range(128):
            latent_state = latent_state + 0.005 * torch.randn(
                latent_state.shape,
                generator=generator,
            )
            latent_state[..., : config.address_dim] = torch.nn.functional.normalize(
                latent_state[..., : config.address_dim],
                dim=-1,
            )
            permutations = torch.stack(
                [torch.randperm(capacity, generator=generator) for _ in range(batch_size)]
            )
            observations = torch.stack(
                [latent_state[row, permutations[row]] for row in range(batch_size)]
            )
            output = model(
                belief,
                _discovery(observations, token_count=17, generator=generator),
                torch.randn(batch_size, config.action_dim, generator=generator),
                torch.full((batch_size,), config.reference_delta_t_s),
            )
            belief = output.belief

            assert torch.isfinite(belief.state_mean).all()
            assert torch.isfinite(belief.geometry_covariance_diag).all()
            assert torch.isfinite(output.innovation).all()
            assert belief.valid.all()
            assert (belief.geometry_covariance_diag[belief.valid] >= config.minimum_variance).all()
            assert ((belief.existence >= 0.0) & (belief.existence <= 1.0)).all()
            assert ((belief.visibility >= 0.0) & (belief.visibility <= belief.existence)).all()
            torch.testing.assert_close(
                output.ownership.sum(dim=-1),
                torch.ones(batch_size, 17),
                atol=1e-6,
                rtol=1e-6,
            )
