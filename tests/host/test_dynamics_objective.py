from __future__ import annotations

from dataclasses import replace

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
rollout_target_module = pytest.importorskip("picf_next.data.rollout_targets")
binding_module = pytest.importorskip("picf_next.models.binding_loss")
core_module = pytest.importorskip("picf_next.models.core")
discovery_module = pytest.importorskip("picf_next.models.discovery")
dynamics_module = pytest.importorskip("picf_next.models.dynamics_loss")
evidence_module = pytest.importorskip("picf_next.models.evidence")
objective_module = pytest.importorskip("picf_next.models.objective")
temporal_module = pytest.importorskip("picf_next.models.temporal")
set_loss_module = pytest.importorskip("picf_next.models.set_loss")
posterior_module = pytest.importorskip("picf_next.posterior")

PICFCoreConfig = core_module.PICFCoreConfig
ObjectDiscoveryConfig = discovery_module.ObjectDiscoveryConfig
AlignedObjectDynamicsTarget = dynamics_module.AlignedObjectDynamicsTarget
AlignedObjectLifecycleTarget = dynamics_module.AlignedObjectLifecycleTarget
ObjectLifecycleInventoryTarget = dynamics_module.ObjectLifecycleInventoryTarget
ObjectDynamicsCriterion = dynamics_module.ObjectDynamicsCriterion
ObjectDynamicsLossConfig = dynamics_module.ObjectDynamicsLossConfig
ObjectGeometryOvershootingConfig = dynamics_module.ObjectGeometryOvershootingConfig
ObjectGeometryOvershootingCriterion = dynamics_module.ObjectGeometryOvershootingCriterion
ObjectGeometryRolloutTarget = dynamics_module.ObjectGeometryRolloutTarget
align_object_lifecycle_inventory = dynamics_module.align_object_lifecycle_inventory
object_detectability_transition_loss = dynamics_module.object_detectability_transition_loss
balanced_conditional_detectability_loss = dynamics_module.balanced_conditional_detectability_loss
ModalityProjectionSpec = evidence_module.ModalityProjectionSpec
NativeTokenBank = evidence_module.NativeTokenBank
PICFObjective = objective_module.PICFObjective
PICFObjectiveConfig = objective_module.PICFObjectiveConfig
TemporalFilterConfig = temporal_module.TemporalFilterConfig
ObjectSetTarget = set_loss_module.ObjectSetTarget
BindingLossConfig = binding_module.BindingLossConfig
BindingLossOutput = binding_module.BindingLossOutput
DEATH_EVENT = posterior_module.DEATH_EVENT
BIRTH_EVENT = posterior_module.BIRTH_EVENT
MATCH_EVENT = posterior_module.MATCH_EVENT
MISS_EVENT = posterior_module.MISS_EVENT
UNUSED_EVENT = posterior_module.UNUSED_EVENT


GEOMETRY = synthetic_geometry_contract(2)


def test_historical_dynamics_api_reexports_the_data_owned_rollout_contract() -> None:
    assert ObjectGeometryRolloutTarget is rollout_target_module.ObjectGeometryRolloutTarget


class _ScheduledBindingCriterion(torch.nn.Module):
    def __init__(self, schedule: tuple[float | None, ...]) -> None:
        super().__init__()
        self.config = BindingLossConfig()
        self.schedule = schedule
        self.calls = 0

    def forward(self, projection, _targets) -> BindingLossOutput:
        scheduled = self.schedule[self.calls]
        self.calls += 1
        zero = projection.binding_features.sum() * 0.0
        return BindingLossOutput(
            loss=zero if scheduled is None else zero + scheduled,
            object_modality_views=0 if scheduled is None else 2,
            positive_pairs=0 if scheduled is None else 2,
            negative_pairs=0 if scheduled is None else 2,
        )


def _core_config() -> PICFCoreConfig:
    return PICFCoreConfig(
        modality_specs=(ModalityProjectionSpec("vision", token_dim=6),),
        binding_dim=8,
        discovery=ObjectDiscoveryConfig(
            input_dim=8,
            hidden_dim=12,
            num_queries=3,
            num_layers=1,
            num_heads=3,
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            initial_variance=0.1,
        ),
        temporal=TemporalFilterConfig(
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            action_dim=3,
            reference_delta_t_s=0.1,
            hidden_dim=12,
            num_layers=1,
            num_heads=3,
        ),
        # Keep spare posterior rows so lifecycle validation exercises the
        # distinction between a live track and an unused capacity slot.
        posterior_capacity=5,
    )


def _two_step_outputs():
    torch.manual_seed(811)
    config = _core_config()
    core = config.build().train()
    with torch.no_grad():
        core.discovery.existence_head.weight.zero_()
        core.discovery.existence_head.bias.fill_(6.0)
    valid = torch.ones(2, 5, dtype=torch.bool)
    first = NativeTokenBank("vision", torch.randn(2, 5, 6), valid)
    second = NativeTokenBank("vision", first.tokens + 0.01 * torch.randn(2, 5, 6), valid)
    empty = config.empty_belief(batch_size=2)
    delta_t_s = torch.full((2,), 0.1)
    output0 = core((first,), empty, torch.zeros(2, 3), delta_t_s)
    output1 = core(
        (second,),
        output0.posterior.belief,
        torch.randn(2, 3) * 0.01,
        delta_t_s,
    )
    return core, output0, output1


def _geometry_rollout_fixture(
    output,
    *,
    horizon: int = 2,
    target_requires_grad: bool = False,
    unaligned: bool = False,
):
    belief = output.posterior.belief
    batch_size, capacity = belief.valid.shape
    valid_cpu = belief.valid.detach().cpu()
    selected_rows = [
        int(valid_cpu[batch_index].nonzero(as_tuple=False)[0]) for batch_index in range(batch_size)
    ]
    row_keys = tuple(
        tuple(
            f"track:{batch_index}:{row}" if bool(valid_cpu[batch_index, row]) else None
            for row in range(capacity)
        )
        for batch_index in range(batch_size)
    )
    future_keys = tuple(
        tuple(
            (
                f"track:unaligned:{batch_index}"
                if unaligned
                else row_keys[batch_index][selected_rows[batch_index]],
            )
            for _step in range(horizon)
        )
        for batch_index in range(batch_size)
    )
    geometry = torch.stack(
        [
            belief.geometry_mean[batch_index, row].detach()
            for batch_index, row in enumerate(selected_rows)
        ]
    )
    geometry = geometry[:, None, None, :].repeat(1, horizon, 1, 1)
    geometry = (
        geometry
        + torch.arange(
            1,
            horizon + 1,
            device=geometry.device,
            dtype=geometry.dtype,
        )[None, :, None, None]
        * 0.2
    )
    if target_requires_grad:
        geometry.requires_grad_(True)
    target = ObjectGeometryRolloutTarget(
        executed_actions=torch.zeros(
            batch_size,
            horizon,
            3,
            device=geometry.device,
            dtype=geometry.dtype,
        ),
        delta_t_s=torch.full(
            (batch_size, horizon),
            0.1,
            device=geometry.device,
            dtype=geometry.dtype,
        ),
        step_valid=torch.ones(
            batch_size,
            horizon,
            device=geometry.device,
            dtype=torch.bool,
        ),
        identity_keys=future_keys,
        geometry=geometry,
        geometry_variance=torch.full_like(geometry, 0.01),
        geometry_supervised=torch.ones_like(geometry, dtype=torch.bool),
        geometry_contract=GEOMETRY,
    )
    return row_keys, target


def test_geometry_overshooting_trains_only_production_dynamics_from_detached_state() -> None:
    core, _output0, output1 = _two_step_outputs()
    transition = core.posterior_filter.transition
    belief = output1.posterior.belief
    belief.geometry_mean.retain_grad()
    belief.geometry_covariance_diag.retain_grad()
    row_keys, target = _geometry_rollout_fixture(output1, horizon=2)
    criterion = ObjectGeometryOvershootingCriterion(ObjectGeometryOvershootingConfig(weight=0.5))

    result = criterion(transition, belief, row_keys, target)

    assert result.active_horizons == 2
    assert result.maximum_horizon == 2
    assert result.matched_predictions == 4
    assert result.unaligned_target_objects == 0
    assert torch.isfinite(result.loss)
    result.loss.backward()

    assert belief.geometry_mean.grad is None
    assert belief.geometry_covariance_diag.grad is None
    assert transition.dynamic_head.weight.grad is not None
    assert torch.count_nonzero(transition.dynamic_head.weight.grad) > 0
    assert (
        torch.count_nonzero(transition.dynamic_head.weight.grad[: transition.config.content_dim])
        > 0
    )
    assert transition.process_variance_head.weight.grad is not None
    assert torch.count_nonzero(transition.process_variance_head.weight.grad) > 0
    for head in (
        transition.survival_head,
        transition.detectability_if_detected_head,
        transition.detectability_if_missed_head,
    ):
        assert head.weight.grad is None or torch.count_nonzero(head.weight.grad) == 0


def test_geometry_overshooting_content_credit_begins_only_after_one_rollout_step() -> None:
    one_step_core, _output0, one_step_output = _two_step_outputs()
    one_step_transition = one_step_core.posterior_filter.transition
    one_step_keys, one_step_target = _geometry_rollout_fixture(one_step_output, horizon=1)
    ObjectGeometryOvershootingCriterion()(
        one_step_transition,
        one_step_output.posterior.belief,
        one_step_keys,
        one_step_target,
    ).loss.backward()

    one_step_gradient = one_step_transition.dynamic_head.weight.grad
    assert one_step_gradient is not None
    assert torch.count_nonzero(one_step_gradient[: one_step_transition.config.content_dim]) == 0
    assert torch.count_nonzero(one_step_gradient[one_step_transition.config.content_dim :]) > 0

    two_step_core, _output0, two_step_output = _two_step_outputs()
    two_step_transition = two_step_core.posterior_filter.transition
    two_step_keys, two_step_target = _geometry_rollout_fixture(two_step_output, horizon=2)
    ObjectGeometryOvershootingCriterion()(
        two_step_transition,
        two_step_output.posterior.belief,
        two_step_keys,
        two_step_target,
    ).loss.backward()

    two_step_gradient = two_step_transition.dynamic_head.weight.grad
    assert two_step_gradient is not None
    assert torch.count_nonzero(two_step_gradient[: two_step_transition.config.content_dim]) > 0


def test_geometry_overshooting_is_sensitive_to_action_order_and_physical_time() -> None:
    core, _output0, output1 = _two_step_outputs()
    transition = core.posterior_filter.transition
    row_keys, target = _geometry_rollout_fixture(output1, horizon=2)
    actions = target.executed_actions.clone()
    actions[:, 0, 0] = 1.25
    actions[:, 1, 1] = -0.75
    ordered = replace(target, executed_actions=actions)
    reversed_actions = replace(target, executed_actions=actions.flip(dims=(1,)))
    criterion = ObjectGeometryOvershootingCriterion()

    ordered_loss = criterion(
        transition,
        output1.posterior.belief,
        row_keys,
        ordered,
    ).loss
    reversed_loss = criterion(
        transition,
        output1.posterior.belief,
        row_keys,
        reversed_actions,
    ).loss
    assert not torch.allclose(ordered_loss, reversed_loss, atol=1e-8, rtol=1e-6)

    delta_t = target.delta_t_s.clone()
    delta_t[:, 0] = 0.05
    delta_t[:, 1] = 0.2
    timed_loss = criterion(
        transition,
        output1.posterior.belief,
        row_keys,
        replace(target, delta_t_s=delta_t),
    ).loss
    reversed_time_loss = criterion(
        transition,
        output1.posterior.belief,
        row_keys,
        replace(target, delta_t_s=delta_t.flip(dims=(1,))),
    ).loss
    assert not torch.allclose(timed_loss, reversed_time_loss, atol=1e-8, rtol=1e-6)


def test_geometry_overshooting_uses_next_state_target_at_every_horizon() -> None:
    """Regress the two-step target shift reported for released V-JEPA2-AC.

    For actions ``u[t]`` and ``u[t+1]``, the criterion must compare production
    transition predictions to ``g[t+1]`` and ``g[t+2]`` respectively. Reusing
    the first target at the second horizon must increase the exact Gaussian
    objective.
    """

    core, _output0, output1 = _two_step_outputs()
    transition = core.posterior_filter.transition
    start = output1.posterior.belief
    batch_size, capacity = start.valid.shape
    selected_rows = [
        int(start.valid[batch_index].detach().nonzero(as_tuple=False)[0])
        for batch_index in range(batch_size)
    ]
    row_keys = tuple(
        tuple(
            f"chronology:{batch_index}:{row}" if bool(start.valid[batch_index, row]) else None
            for row in range(capacity)
        )
        for batch_index in range(batch_size)
    )
    actions = torch.tensor(
        [[[1.5, -0.25, 0.75], [-1.0, 1.25, -0.5]]],
        dtype=start.geometry_mean.dtype,
        device=start.geometry_mean.device,
    ).repeat(batch_size, 1, 1)
    delta_t = torch.tensor(
        [[0.05, 0.2]],
        dtype=start.geometry_mean.dtype,
        device=start.geometry_mean.device,
    ).repeat(batch_size, 1)

    belief = dynamics_module._detached_rollout_belief(start)
    future_geometry = []
    for horizon_index in range(2):
        prediction = transition(
            belief,
            actions[:, horizon_index],
            delta_t[:, horizon_index],
        )
        belief = dynamics_module._detach_rollout_lifecycle(prediction.belief)
        future_geometry.append(
            torch.stack(
                [
                    belief.geometry_mean[batch_index, row]
                    for batch_index, row in enumerate(selected_rows)
                ]
            ).detach()
        )
    geometry = torch.stack(future_geometry, dim=1).unsqueeze(2)
    identity_keys = tuple(
        tuple((row_keys[batch_index][selected_rows[batch_index]],) for _ in range(2))
        for batch_index in range(batch_size)
    )
    target = ObjectGeometryRolloutTarget(
        executed_actions=actions,
        delta_t_s=delta_t,
        step_valid=torch.ones(batch_size, 2, dtype=torch.bool, device=actions.device),
        identity_keys=identity_keys,
        geometry=geometry,
        geometry_variance=torch.zeros_like(geometry),
        geometry_supervised=torch.ones_like(geometry, dtype=torch.bool),
        geometry_contract=GEOMETRY,
    )
    criterion = ObjectGeometryOvershootingCriterion()

    aligned = criterion(transition, start, row_keys, target).loss
    stale_second_target = geometry.clone()
    stale_second_target[:, 1] = stale_second_target[:, 0]
    shifted = criterion(
        transition,
        start,
        row_keys,
        replace(target, geometry=stale_second_target),
    ).loss

    assert shifted > aligned


def test_geometry_overshooting_rejects_same_width_but_different_physical_contract() -> None:
    core, _output0, output1 = _two_step_outputs()
    row_keys, target = _geometry_rollout_fixture(output1)
    incompatible = synthetic_geometry_contract(2, name="picf.other-position.v1")

    with pytest.raises(ValueError, match="contracts differ"):
        ObjectGeometryOvershootingCriterion()(
            core.posterior_filter.transition,
            output1.posterior.belief,
            row_keys,
            replace(target, geometry_contract=incompatible),
        )


def test_geometry_overshooting_rejects_differentiable_or_unaligned_future_targets() -> None:
    core, _output0, output1 = _two_step_outputs()
    transition = core.posterior_filter.transition
    criterion = ObjectGeometryOvershootingCriterion()
    row_keys, differentiable = _geometry_rollout_fixture(
        output1,
        target_requires_grad=True,
    )

    with pytest.raises(ValueError, match="finite, detached and colocated"):
        criterion(transition, output1.posterior.belief, row_keys, differentiable)

    row_keys, unaligned = _geometry_rollout_fixture(output1, unaligned=True)
    with pytest.raises(ValueError, match="no rollout geometry target aligns"):
        criterion(transition, output1.posterior.belief, row_keys, unaligned)


def test_geometry_overshooting_supports_prefix_padded_episode_boundaries() -> None:
    core, _output0, output1 = _two_step_outputs()
    row_keys, target = _geometry_rollout_fixture(output1, horizon=2)
    step_valid = target.step_valid.clone()
    step_valid[1, 1] = False
    delta_t = target.delta_t_s.clone()
    delta_t[1, 1] = 0.0
    geometry = target.geometry.clone()
    geometry[1, 1] = 0.0
    variance = target.geometry_variance.clone()
    variance[1, 1] = 0.0
    supervised = target.geometry_supervised.clone()
    supervised[1, 1] = False
    keys = tuple(
        sample if batch_index == 0 else (sample[0], (None,))
        for batch_index, sample in enumerate(target.identity_keys)
    )
    padded = replace(
        target,
        delta_t_s=delta_t,
        step_valid=step_valid,
        identity_keys=keys,
        geometry=geometry,
        geometry_variance=variance,
        geometry_supervised=supervised,
    )

    result = ObjectGeometryOvershootingCriterion()(
        core.posterior_filter.transition,
        output1.posterior.belief,
        row_keys,
        padded,
    )

    assert result.active_horizons == 2
    assert result.matched_predictions == 3

    non_prefix = step_valid.clone()
    non_prefix[0] = torch.tensor([False, True])
    with pytest.raises(ValueError, match="contiguous prefix"):
        ObjectGeometryOvershootingCriterion()(
            core.posterior_filter.transition,
            output1.posterior.belief,
            row_keys,
            replace(padded, step_valid=non_prefix),
        )


def test_unified_objective_counts_geometry_overshooting_once_inside_dynamics() -> None:
    core, _output0, output1 = _two_step_outputs()
    row_keys, rollout_target = _geometry_rollout_fixture(output1, horizon=2)
    set_target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        object_inventory_complete=True,
    )
    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=0.25,
            binding_weight=0.0,
        ),
        geometry_overshooting_criterion=ObjectGeometryOvershootingCriterion(
            ObjectGeometryOvershootingConfig(weight=0.5)
        ),
    )

    result = objective(
        [output1],
        action_loss=None,
        set_targets=((set_target, set_target),),
        initial_loss_track_keys_by_row=row_keys,
        geometry_rollout_target=rollout_target,
        transition=core.posterior_filter.transition,
    )

    expected_dynamics = (
        result.losses["loss_dynamics_one_step"]
        + 0.5 * result.losses["loss_dynamics_geometry_overshooting"]
    )
    torch.testing.assert_close(result.losses["loss_dynamics"], expected_dynamics)
    torch.testing.assert_close(result.loss, 0.25 * expected_dynamics)
    assert result.diagnostics["geometry_overshooting_active_horizons"] == 2
    assert result.diagnostics["geometry_overshooting_matched_predictions"] == 4
    assert result.diagnostics["geometry_overshooting_unaligned_target_objects"] == 0


def test_unified_objective_rejects_inactive_or_incomplete_geometry_overshooting() -> None:
    core, _output0, output1 = _two_step_outputs()
    row_keys, rollout_target = _geometry_rollout_fixture(output1)
    inactive = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=1.0,
            binding_weight=0.0,
        )
    )
    with pytest.raises(ValueError, match="while overshooting is inactive"):
        inactive(
            [output1],
            action_loss=None,
            set_targets=None,
            geometry_rollout_target=rollout_target,
            transition=core.posterior_filter.transition,
        )

    active = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=1.0,
            binding_weight=0.0,
        ),
        geometry_overshooting_criterion=ObjectGeometryOvershootingCriterion(
            ObjectGeometryOvershootingConfig(weight=1.0)
        ),
    )
    with pytest.raises(ValueError, match="future rollout target"):
        active(
            [output1],
            action_loss=None,
            set_targets=None,
            initial_loss_track_keys_by_row=row_keys,
            transition=core.posterior_filter.transition,
        )


def test_dynamics_loss_uses_stop_gradient_current_observations() -> None:
    core, _output0, output1 = _two_step_outputs()
    output1.discovery.address_mean.retain_grad()
    output1.discovery.content_mean.retain_grad()
    output1.posterior.match_probability.retain_grad()
    result = ObjectDynamicsCriterion()(output1)

    assert result.matched_predictions > 0
    assert result.lifecycle_predictions == 0
    assert torch.isfinite(result.total)
    result.total.backward()

    assert output1.discovery.address_mean.grad is None
    assert output1.discovery.content_mean.grad is None
    assert output1.posterior.match_probability.grad is None
    transition = core.posterior_filter.transition
    assert transition.dynamic_head.weight.grad is not None
    assert transition.process_variance_head.weight.grad is not None
    for head in (
        transition.survival_head,
        transition.detectability_if_detected_head,
        transition.detectability_if_missed_head,
    ):
        assert head.weight.grad is None or torch.count_nonzero(head.weight.grad) == 0


def test_marginal_em_dynamics_losses_match_expected_complete_data_factorisation() -> None:
    _core, _output0, output1 = _two_step_outputs()
    result = ObjectDynamicsCriterion()(output1)
    prediction = output1.posterior.prior_prediction.belief
    weight = output1.posterior.match_probability.detach().float()
    weight = weight * prediction.valid.unsqueeze(-1)
    predicted_content = prediction.content_mean.float().unsqueeze(2)
    observed_content = output1.discovery.content_mean.detach().float().unsqueeze(1)
    content_terms = 1.0 - torch.nn.functional.cosine_similarity(
        predicted_content.expand(-1, -1, observed_content.shape[2], -1),
        observed_content.expand(-1, predicted_content.shape[1], -1, -1),
        dim=-1,
    )
    predicted_geometry = prediction.geometry_mean.float().unsqueeze(2)
    observed_geometry = output1.discovery.geometry_mean.detach().float().unsqueeze(1)
    total_variance = prediction.geometry_covariance_diag.float().unsqueeze(
        2
    ) + output1.discovery.geometry_variance.detach().float().unsqueeze(1)
    geometry_terms = torch.nn.functional.gaussian_nll_loss(
        predicted_geometry.expand(-1, -1, observed_geometry.shape[2], -1),
        observed_geometry.expand(-1, predicted_geometry.shape[1], -1, -1),
        total_variance,
        full=False,
        reduction="none",
    ).mean(dim=-1)
    normalizer = weight.sum().clamp_min(1e-6)
    expected_content = (weight * content_terms).sum() / normalizer
    expected_geometry = (weight * geometry_terms).sum() / normalizer
    torch.testing.assert_close(result.losses["loss_dynamics_content_cosine"], expected_content)
    torch.testing.assert_close(result.losses["loss_dynamics_geometry_nll"], expected_geometry)
    assert result.matched_predictions == int((weight.sum(dim=-1) > 1e-6).sum())


def test_dynamics_nll_is_invariant_to_detached_discovery_address() -> None:
    """Runtime marginals may weight a row, but address itself is not a target."""

    _core, _output0, output1 = _two_step_outputs()
    criterion = ObjectDynamicsCriterion()
    baseline = criterion(output1).total
    shifted_discovery = replace(
        output1.discovery,
        address_mean=output1.discovery.address_mean.detach() + 1000.0,
    )
    shifted = criterion(replace(output1, discovery=shifted_discovery)).total

    torch.testing.assert_close(shifted, baseline, atol=0.0, rtol=0.0)


def test_physical_dynamics_alignment_is_independent_of_runtime_map_rows() -> None:
    _core, _output0, output1 = _two_step_outputs()
    row_to_observation = torch.full_like(output1.posterior.event_type, -1)
    for batch_index, sample_mapping in enumerate(
        output1.posterior.observation_to_posterior.tolist()
    ):
        for observation, row in enumerate(sample_mapping):
            if row >= 0 and int(output1.posterior.event_type[batch_index, row]) == 1:
                row_to_observation[batch_index, row] = observation
    target = AlignedObjectDynamicsTarget(row_to_observation)
    criterion = ObjectDynamicsCriterion()
    expected = criterion(output1, dynamics_target=target)

    runtime_mapping = output1.posterior.observation_to_posterior.clone()
    runtime_mapping[:, 0], runtime_mapping[:, 1] = (
        runtime_mapping[:, 1].clone(),
        runtime_mapping[:, 0].clone(),
    )
    swapped_runtime = replace(
        output1,
        posterior=replace(
            output1.posterior,
            observation_to_posterior=runtime_mapping,
        ),
    )
    actual = criterion(swapped_runtime, dynamics_target=target)

    assert actual.independently_aligned_predictions == expected.matched_predictions
    assert actual.independently_aligned_predictions > 0
    for name, expected_loss in expected.losses.items():
        torch.testing.assert_close(actual.losses[name], expected_loss, atol=0.0, rtol=0.0)


def test_marginal_em_dynamics_is_independent_of_diagnostic_lifecycle_projection() -> None:
    _core, _output0, output1 = _two_step_outputs()
    expected = ObjectDynamicsCriterion()(output1)
    mapping = output1.posterior.observation_to_posterior.clone()
    mapping[:, 1] = mapping[:, 0]
    malformed = replace(
        output1,
        posterior=replace(
            output1.posterior,
            observation_to_posterior=mapping,
            event_type=torch.full_like(output1.posterior.event_type, DEATH_EVENT),
        ),
    )

    actual = ObjectDynamicsCriterion()(malformed)

    for name, expected_loss in expected.losses.items():
        torch.testing.assert_close(actual.losses[name], expected_loss, atol=0.0, rtol=0.0)


def test_dynamics_does_not_evaluate_zero_marginal_observation_values() -> None:
    _core, _output0, output1 = _two_step_outputs()
    match_probability = output1.posterior.match_probability.clone()
    match_probability[:, :, 0] = 0.0
    posterior = replace(output1.posterior, match_probability=match_probability)
    expected = ObjectDynamicsCriterion()(replace(output1, posterior=posterior))
    content = output1.discovery.content_mean.clone()
    geometry = output1.discovery.geometry_mean.clone()
    content[0, 0] = 1e30
    geometry[0, 0] = -1e30
    modified = replace(
        output1,
        discovery=replace(output1.discovery, content_mean=content, geometry_mean=geometry),
        posterior=posterior,
    )

    result = ObjectDynamicsCriterion()(modified)

    assert torch.isfinite(result.total)
    for name, expected_loss in expected.losses.items():
        torch.testing.assert_close(result.losses[name], expected_loss, atol=0.0, rtol=0.0)


def test_lifecycle_loss_requires_independent_targets_and_trains_lifecycle_heads() -> None:
    core, _output0, output1 = _two_step_outputs()
    criterion = ObjectDynamicsCriterion(
        ObjectDynamicsLossConfig(
            content_cosine_weight=1.0,
            geometry_nll_weight=1.0,
            survival_weight=1.0,
            visibility_weight=1.0,
        )
    )
    with pytest.raises(ValueError, match="independent loss-side lifecycle targets"):
        criterion(output1)

    valid = output1.posterior.prior_prediction.belief.valid
    previous_visibility = (torch.arange(valid.shape[1], device=valid.device) % 2 == 0).unsqueeze(
        0
    ).expand_as(valid).to(torch.float32) * valid
    target = AlignedObjectLifecycleTarget(
        survival=valid.to(torch.float32),
        survival_supervised=valid,
        visibility=valid.to(torch.float32),
        visibility_supervised=valid,
        previous_visibility=previous_visibility,
        previous_visibility_supervised=valid,
    )
    result = criterion(output1, target)
    assert result.lifecycle_predictions == 2 * int(valid.sum().item())
    assert result.survival_positive_target_mass.item() == int(valid.sum().item())
    assert result.survival_negative_target_mass.item() == 0.0
    assert result.visibility_positive_target_mass.item() == int(valid.sum().item())
    assert result.visibility_negative_target_mass.item() == 0.0
    result.total.backward()

    transition = core.posterior_filter.transition
    assert transition.survival_head.weight.grad is not None
    assert transition.detectability_if_detected_head.weight.grad is not None
    assert transition.detectability_if_missed_head.weight.grad is not None


def test_detectability_kernel_trains_selected_branches_not_the_runtime_mixture() -> None:
    core, _output0, output1 = _two_step_outputs()
    transition = core.posterior_filter.transition
    prior = output1.posterior.prior_prediction.belief
    prior = replace(
        prior,
        address_mean=prior.address_mean.detach(),
        content_mean=prior.content_mean.detach(),
        geometry_mean=prior.geometry_mean.detach(),
        geometry_covariance_diag=prior.geometry_covariance_diag.detach(),
        existence_logits=prior.existence_logits.detach(),
        visibility_given_existence_logits=prior.visibility_given_existence_logits.detach(),
        measurement_age_s=(prior.measurement_age_s.detach() + 0.5) * prior.valid,
    )
    detected_probability = 0.8
    missed_probability = 0.2
    with torch.no_grad():
        transition.detectability_if_detected_head.weight.zero_()
        transition.detectability_if_detected_head.bias.fill_(
            torch.logit(torch.tensor(detected_probability))
        )
        transition.detectability_if_missed_head.weight.zero_()
        transition.detectability_if_missed_head.bias.fill_(
            torch.logit(torch.tensor(missed_probability))
        )
    prediction = transition(
        prior,
        torch.zeros(2, transition.config.action_dim),
        torch.full((2,), transition.config.reference_delta_t_s),
    )
    valid = prediction.belief.valid
    row_index = torch.arange(valid.numel(), device=valid.device).reshape_as(valid)
    current = ((row_index % 3) == 0).to(torch.float32) * valid
    previous = ((row_index % 2) == 0).to(torch.float32) * valid
    target = AlignedObjectLifecycleTarget(
        survival=torch.zeros_like(current),
        survival_supervised=torch.zeros_like(valid),
        visibility=current,
        visibility_supervised=valid,
        previous_visibility=previous,
        previous_visibility_supervised=valid,
    )

    result = object_detectability_transition_loss(
        prediction,
        target,
        probability_epsilon=1e-6,
    )
    detected_terms = torch.nn.functional.binary_cross_entropy(
        torch.full_like(current, detected_probability), current, reduction="none"
    )
    missed_terms = torch.nn.functional.binary_cross_entropy(
        torch.full_like(current, missed_probability), current, reduction="none"
    )
    detected_weight = previous * valid
    missed_weight = (1.0 - previous) * valid
    expected = 0.5 * (
        (detected_terms * detected_weight).sum() / detected_weight.sum()
        + (missed_terms * missed_weight).sum() / missed_weight.sum()
    )

    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(
        result.loss_sum,
        result.detected_loss_sum + result.missed_loss_sum,
    )
    result.loss.backward()
    assert torch.count_nonzero(transition.detectability_if_detected_head.bias.grad) > 0
    assert torch.count_nonzero(transition.detectability_if_missed_head.bias.grad) > 0
    assert transition.missed_duration_logit_slope.grad is not None
    assert transition.missed_duration_logit_slope.grad.abs() > 0.0


def test_detectability_branch_aggregation_does_not_starve_rare_missed_states() -> None:
    detected_mass = torch.tensor(100.0)
    missed_mass = torch.tensor(1.0)
    loss = balanced_conditional_detectability_loss(
        detected_loss_sum=torch.tensor(100.0),
        detected_mass=detected_mass,
        missed_loss_sum=torch.tensor(10.0),
        missed_mass=missed_mass,
    )

    torch.testing.assert_close(loss, torch.tensor(5.5))
    assert loss > torch.tensor(110.0 / 101.0)


def test_lifecycle_target_rejects_supervision_on_unused_rows() -> None:
    _core, _output0, output1 = _two_step_outputs()
    criterion = ObjectDynamicsCriterion(
        ObjectDynamicsLossConfig(
            content_cosine_weight=1.0,
            geometry_nll_weight=1.0,
            survival_weight=1.0,
            visibility_weight=0.0,
        )
    )
    valid = output1.posterior.prior_prediction.belief.valid
    invalid_supervision = ~valid
    target = AlignedObjectLifecycleTarget(
        survival=invalid_supervision.to(torch.float32),
        survival_supervised=invalid_supervision,
        visibility=torch.zeros_like(valid, dtype=torch.float32),
        visibility_supervised=torch.zeros_like(valid),
        previous_visibility=torch.zeros_like(valid, dtype=torch.float32),
        previous_visibility_supervised=torch.zeros_like(valid),
    )
    with pytest.raises(ValueError, match="unused posterior row"):
        criterion(output1, target)


def test_lifecycle_target_rejects_a_differentiable_label() -> None:
    _core, _output0, output1 = _two_step_outputs()
    criterion = ObjectDynamicsCriterion(
        ObjectDynamicsLossConfig(
            content_cosine_weight=1.0,
            geometry_nll_weight=1.0,
            survival_weight=1.0,
            visibility_weight=0.0,
        )
    )
    valid = output1.posterior.prior_prediction.belief.valid
    target = AlignedObjectLifecycleTarget(
        survival=valid.to(torch.float32).requires_grad_(True),
        survival_supervised=valid,
        visibility=torch.zeros_like(valid, dtype=torch.float32),
        visibility_supervised=torch.zeros_like(valid),
        previous_visibility=torch.zeros_like(valid, dtype=torch.float32),
        previous_visibility_supervised=torch.zeros_like(valid),
    )

    with pytest.raises(ValueError, match="loss-only lifecycle survival"):
        criterion(output1, target)


def test_physical_inventory_aligns_occluded_and_dead_objects_without_query_labels() -> None:
    valid = torch.tensor([[True, True, True, False]])
    keys = (("track:visible", "track:dead", "track:occluded", None),)
    inventory = ObjectLifecycleInventoryTarget(
        alive_identity_keys=("track:visible", "track:occluded"),
        inventory_complete=True,
        visibility=torch.tensor([1.0, 0.0]),
        visibility_supervised=torch.tensor([True, True]),
    )

    aligned = align_object_lifecycle_inventory(
        (inventory,),
        keys,
        valid,
        dtype=torch.float32,
        previous_targets=(inventory,),
    )

    torch.testing.assert_close(aligned.survival, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    assert aligned.survival_supervised.tolist() == [[True, True, True, False]]
    torch.testing.assert_close(aligned.visibility, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert aligned.visibility_supervised.tolist() == [[True, False, True, False]]
    torch.testing.assert_close(aligned.previous_visibility, aligned.visibility)
    assert torch.equal(aligned.previous_visibility_supervised, aligned.visibility_supervised)


def test_partial_physical_inventory_never_turns_absence_into_a_death_label() -> None:
    valid = torch.tensor([[True, True]])
    inventory = ObjectLifecycleInventoryTarget(
        alive_identity_keys=("track:known",),
        inventory_complete=False,
    )

    aligned = align_object_lifecycle_inventory(
        (inventory,),
        (("track:known", "track:unknown"),),
        valid,
        dtype=torch.float32,
    )

    torch.testing.assert_close(aligned.survival, torch.tensor([[1.0, 0.0]]))
    assert aligned.survival_supervised.tolist() == [[True, False]]
    assert not aligned.visibility_supervised.any()


def test_inventory_visibility_is_an_atomic_selective_target() -> None:
    valid = torch.tensor([[True]])
    malformed = ObjectLifecycleInventoryTarget(
        alive_identity_keys=("track:a",),
        visibility=torch.tensor([1.0]),
    )

    with pytest.raises(ValueError, match="are atomic"):
        align_object_lifecycle_inventory(
            (malformed,),
            (("track:a",),),
            valid,
            dtype=torch.float32,
        )


def test_unified_objective_aligns_lifecycle_inventory_from_checkpointed_tracks() -> None:
    core, _output0, output1 = _two_step_outputs()
    observed = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        object_inventory_complete=True,
        temporal_identity_keys=("track:a",),
    )
    inventory = ObjectLifecycleInventoryTarget(
        alive_identity_keys=("track:a",),
        inventory_complete=True,
        visibility=torch.tensor([1.0]),
        visibility_supervised=torch.tensor([True]),
    )
    initial_tracks = (
        ("track:a", "track:dead", None, None, None),
        ("track:a", "track:dead", None, None, None),
    )
    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=1.0,
            binding_weight=0.0,
        ),
        dynamics_criterion=ObjectDynamicsCriterion(
            ObjectDynamicsLossConfig(
                content_cosine_weight=1.0,
                geometry_nll_weight=1.0,
                survival_weight=1.0,
                visibility_weight=1.0,
            )
        ),
    )

    result = objective(
        [output1],
        action_loss=None,
        set_targets=((observed, observed),),
        lifecycle_targets=((inventory, inventory),),
        initial_lifecycle_targets=(inventory, inventory),
        initial_loss_track_keys_by_row=initial_tracks,
    )
    result.loss.backward()

    assert result.diagnostics["lifecycle_predictions"] == 6
    assert result.diagnostics["lifecycle_survival_positive_target_mass"] == 2.0
    assert result.diagnostics["lifecycle_survival_negative_target_mass"] == 2.0
    assert result.diagnostics["lifecycle_detection_positive_target_mass"] == 2.0
    assert result.diagnostics["lifecycle_detection_negative_target_mass"] == 0.0
    assert result.losses["loss_dynamics_survival"] > 0.0
    assert result.losses["loss_dynamics_visibility"] > 0.0
    transition = core.posterior_filter.transition
    assert transition.survival_head.weight.grad is not None
    assert transition.detectability_if_detected_head.weight.grad is not None


@pytest.mark.parametrize(
    (
        "survival_weight",
        "visibility_weight",
        "active_loss",
        "inactive_loss",
        "expected_predictions",
    ),
    (
        (
            1.0,
            0.0,
            "loss_dynamics_survival",
            "loss_dynamics_visibility",
            4,
        ),
        (
            0.0,
            1.0,
            "loss_dynamics_visibility",
            "loss_dynamics_survival",
            2,
        ),
    ),
)
def test_unified_objective_aligns_only_the_enabled_lifecycle_family(
    survival_weight: float,
    visibility_weight: float,
    active_loss: str,
    inactive_loss: str,
    expected_predictions: int,
) -> None:
    core, _output0, output1 = _two_step_outputs()
    observed = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        object_inventory_complete=True,
        temporal_identity_keys=("track:a",),
    )
    inventory = ObjectLifecycleInventoryTarget(
        alive_identity_keys=("track:a",),
        inventory_complete=True,
        visibility=torch.tensor([1.0]),
        visibility_supervised=torch.tensor([True]),
    )
    initial_tracks = (
        ("track:a", "track:dead", None, None, None),
        ("track:a", "track:dead", None, None, None),
    )
    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=1.0,
            binding_weight=0.0,
        ),
        dynamics_criterion=ObjectDynamicsCriterion(
            ObjectDynamicsLossConfig(
                content_cosine_weight=1.0,
                geometry_nll_weight=1.0,
                survival_weight=survival_weight,
                visibility_weight=visibility_weight,
            )
        ),
    )

    result = objective(
        [output1],
        action_loss=None,
        set_targets=((observed, observed),),
        lifecycle_targets=((inventory, inventory),),
        initial_lifecycle_targets=((inventory, inventory) if visibility_weight > 0.0 else None),
        initial_loss_track_keys_by_row=initial_tracks,
    )
    result.loss.backward()

    assert result.diagnostics["lifecycle_predictions"] == expected_predictions
    assert result.losses[active_loss] > 0.0
    assert result.losses[inactive_loss] == 0.0
    transition = core.posterior_filter.transition
    detection_heads = (
        transition.detectability_if_detected_head,
        transition.detectability_if_missed_head,
    )
    if survival_weight > 0.0:
        assert transition.survival_head.weight.grad is not None
        assert all(
            head.weight.grad is None or torch.count_nonzero(head.weight.grad) == 0
            for head in detection_heads
        )
    else:
        assert (
            transition.survival_head.weight.grad is None
            or torch.count_nonzero(transition.survival_head.weight.grad) == 0
        )
        assert all(head.weight.grad is not None for head in detection_heads)
        assert any(torch.count_nonzero(head.weight.grad) > 0 for head in detection_heads)


def test_unified_objective_supports_explicit_representation_and_joint_phases() -> None:
    _core, _output0, output1 = _two_step_outputs()
    representation = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=1.0,
            binding_weight=0.0,
        )
    )([output1], action_loss=None, set_targets=None)
    assert torch.equal(representation.loss, representation.losses["loss_dynamics"])

    action_loss = output1.action_bank.value.square().mean()
    joint = PICFObjective(
        PICFObjectiveConfig(
            action_weight=1.0,
            set_weight=0.0,
            dynamics_weight=0.5,
            binding_weight=0.0,
        )
    )([output1], action_loss=action_loss, set_targets=None)
    torch.testing.assert_close(
        joint.loss,
        action_loss + 0.5 * joint.losses["loss_dynamics"],
    )


def test_unified_objective_rejects_implicit_or_unused_inputs() -> None:
    _core, _output0, output1 = _two_step_outputs()
    action_only = PICFObjective(
        PICFObjectiveConfig(
            action_weight=1.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=0.0,
        )
    )
    with pytest.raises(ValueError, match="requires one finite scalar"):
        action_only([output1], action_loss=None, set_targets=None)
    with pytest.raises(ValueError, match="requires one finite scalar"):
        action_only([output1], action_loss=torch.ones((), dtype=torch.long), set_targets=None)
    with pytest.raises(ValueError, match="targets were supplied"):
        action_only(
            [output1],
            action_loss=torch.ones(()),
            set_targets=[[]],
        )
    with pytest.raises(ValueError, match="lifecycle targets were supplied"):
        action_only(
            [output1],
            action_loss=torch.ones(()),
            set_targets=None,
            lifecycle_targets=[None],
        )


def test_inactive_multimodal_graph_does_not_dilute_temporal_binding() -> None:
    _core, output0, output1 = _two_step_outputs()
    ownership = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    target = ObjectSetTarget(
        ownership=ownership,
        token_valid=torch.ones(5, dtype=torch.bool),
        temporal_identity_keys=("track:a", "track:b"),
    )
    targets = ((target, target), (target, target))
    result = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=1.0,
            require_temporal_positive_pairs=True,
        )
    )(
        [output0, output1],
        action_loss=None,
        set_targets=targets,
    )

    assert result.losses["loss_binding_multimodal"] == 0.0
    assert result.losses["loss_binding_temporal_address"] > 0.0
    torch.testing.assert_close(
        result.losses["loss_binding"],
        result.losses["loss_binding_temporal_address"],
    )
    assert result.diagnostics["active_binding_families"] == 1
    assert result.diagnostics["multimodal_positive_pairs"] == 0
    assert result.diagnostics["temporal_positive_pairs"] > 0


def test_temporal_credit_allows_episode_start_without_a_cross_time_relation() -> None:
    _core, output0, _output1 = _two_step_outputs()
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        temporal_identity_keys=("track:a", "track:b"),
    )
    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=1.0,
            require_temporal_positive_pairs=True,
        )
    )

    result = objective(
        [output0],
        action_loss=None,
        set_targets=((target, target),),
        initial_loss_track_keys_by_row=(
            (None, None, None, None, None),
            (None, None, None, None, None),
        ),
    )

    assert result.diagnostics["temporal_eligible_samples"] == 0
    assert result.diagnostics["temporal_positive_pairs"] == 0


def test_checkpointed_loss_tracks_enable_temporal_credit_in_one_transition() -> None:
    _core, _output0, output1 = _two_step_outputs()
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        object_inventory_complete=True,
        temporal_identity_keys=("track:a", "track:b"),
    )
    initial_tracks = (
        ("track:a", "track:b", None, None, None),
        ("track:a", "track:b", None, None, None),
    )
    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=1.0,
            require_temporal_positive_pairs=True,
        )
    )

    result = objective(
        [output1],
        action_loss=None,
        set_targets=((target, target),),
        initial_loss_track_keys_by_row=initial_tracks,
    )

    assert result.diagnostics["transitions"] == 1
    assert result.diagnostics["temporal_positive_pairs"] > 0
    assert result.losses["loss_binding_temporal_address"] > 0.0
    assert result.diagnostics["loss_track_rows"] == 4


def test_loss_track_advance_reports_runtime_row_swaps_without_endorsing_them() -> None:
    _core, _output0, output1 = _two_step_outputs()
    mapping = output1.posterior.observation_to_posterior.clone()
    mapping[:, 0] = 1
    mapping[:, 1] = 0
    mapping[:, 2] = 2
    event_type = output1.posterior.event_type.clone()
    event_type[:, :3] = MATCH_EVENT
    swapped = replace(
        output1,
        posterior=replace(
            output1.posterior,
            observation_to_posterior=mapping,
            event_type=event_type,
        ),
    )
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        object_inventory_complete=True,
        temporal_identity_keys=("track:a", "track:b"),
    )
    match = set_loss_module.SetMatch(
        prediction_indices=torch.tensor([0, 1]),
        target_indices=torch.tensor([0, 1]),
    )
    initial = (
        ("track:a", "track:b", None, None, None),
        ("track:a", "track:b", None, None, None),
    )

    advanced, conflicts, dynamics_alignment = objective_module._advance_loss_track_keys(
        (swapped,),
        ((target, target),),
        ((match, match),),
        initial,
    )

    assert advanced == initial
    assert conflicts == 4
    assert dynamics_alignment[0] is not None
    assert dynamics_alignment[0].observation_index_by_row[:, :2].tolist() == [[0, 1], [0, 1]]


def test_loss_track_birth_recycling_never_inherits_the_dead_rows_identity() -> None:
    _core, _output0, output1 = _two_step_outputs()
    final_valid = output1.posterior.belief.valid
    event_type = torch.where(
        final_valid,
        torch.full_like(output1.posterior.event_type, MISS_EVENT),
        torch.full_like(output1.posterior.event_type, UNUSED_EVENT),
    )
    event_type[:, 0] = BIRTH_EVENT
    mapping = torch.full_like(output1.posterior.observation_to_posterior, -1)
    mapping[:, 0] = 0
    recycled = replace(
        output1,
        posterior=replace(
            output1.posterior,
            event_type=event_type,
            observation_to_posterior=mapping,
        ),
    )
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        temporal_identity_keys=("track:new",),
    )
    set_match = set_loss_module.SetMatch(
        prediction_indices=torch.tensor([0]),
        target_indices=torch.tensor([0]),
    )
    initial = (
        ("track:dead-a", "track:retained-a", None, None, None),
        ("track:dead-b", "track:retained-b", None, None, None),
    )

    advanced, conflicts, dynamics_alignment = objective_module._advance_loss_track_keys(
        (recycled,),
        ((target, target),),
        ((set_match, set_match),),
        initial,
    )

    assert advanced == (
        ("track:new", "track:retained-a", None, None, None),
        ("track:new", "track:retained-b", None, None, None),
    )
    assert all("dead" not in key for rows in advanced for key in rows if key is not None)
    assert conflicts == 0
    assert dynamics_alignment[0] is not None
    assert (dynamics_alignment[0].observation_index_by_row == -1).all()


def test_loss_track_advance_rejects_identity_on_an_unused_prior_row() -> None:
    _core, _output0, output1 = _two_step_outputs()
    malformed = (
        (None, None, None, None, "track:invalid"),
        (None, None, None, None, None),
    )

    with pytest.raises(ValueError, match="cannot name unused posterior rows"):
        objective_module._advance_loss_track_keys((output1,), (), (), malformed)


def test_normal_one_transition_update_does_not_run_temporal_binding() -> None:
    _core, output0, _output1 = _two_step_outputs()
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        temporal_identity_keys=("track:a", "track:b"),
    )

    class ForbiddenTemporalCriterion(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("one-transition exposure must not execute temporal binding")

    class ForbiddenSetCriterion(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError(
                "one-transition binding-only updates must not execute set matching"
            )

    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=1.0,
        ),
        set_criterion=ForbiddenSetCriterion(),
        temporal_binding_criterion=ForbiddenTemporalCriterion(),
    )
    result = objective(
        [output0],
        action_loss=None,
        set_targets=((target, target),),
    )

    assert result.losses["loss_binding_temporal_address"] == 0.0
    assert result.diagnostics["temporal_positive_pairs"] == 0
    assert result.diagnostics["set_matches"] == 0


def test_single_modality_objective_keeps_relation_calibration_ddp_live() -> None:
    _core, output0, _output1 = _two_step_outputs()
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
    )
    objective = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=1.0,
        )
    )

    result = objective(
        [output0],
        action_loss=None,
        set_targets=((target, target),),
    )
    result.loss.backward()

    relation = objective.binding_criterion.relation
    assert relation is not None
    assert result.diagnostics["multimodal_positive_pairs"] == 0
    assert result.losses["loss_binding"] == 0.0
    for parameter in relation.parameters():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter))


def test_missing_modality_frame_does_not_dilute_an_active_binding_frame() -> None:
    _core, output0, output1 = _two_step_outputs()
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
    )
    binding = _ScheduledBindingCriterion((2.0, None))
    result = PICFObjective(
        PICFObjectiveConfig(
            action_weight=0.0,
            set_weight=0.0,
            dynamics_weight=0.0,
            binding_weight=1.0,
        ),
        binding_criterion=binding,
    )(
        [output0, output1],
        action_loss=None,
        set_targets=((target, target), (target, target)),
    )

    torch.testing.assert_close(result.losses["loss_binding_multimodal"], torch.tensor(2.0))
    torch.testing.assert_close(result.losses["loss_binding"], torch.tensor(2.0))
    assert result.diagnostics["multimodal_active_transitions"] == 1
    assert result.diagnostics["target_samples"] == 4


def test_objective_reports_exact_weighted_components_and_supervision_density() -> None:
    _core, _output0, output1 = _two_step_outputs()
    target = ObjectSetTarget(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        token_valid=torch.ones(5, dtype=torch.bool),
        object_inventory_complete=True,
    )
    result = PICFObjective(
        PICFObjectiveConfig(
            action_weight=2.0,
            set_weight=0.5,
            dynamics_weight=0.25,
            binding_weight=0.0,
        )
    )(
        [output1],
        action_loss=torch.tensor(3.0),
        set_targets=((target, target),),
    )

    torch.testing.assert_close(result.losses["loss_weighted_action"], torch.tensor(6.0))
    torch.testing.assert_close(
        result.loss,
        result.losses["loss_weighted_action"]
        + result.losses["loss_weighted_set"]
        + result.losses["loss_weighted_dynamics"]
        + result.losses["loss_weighted_binding"],
    )
    assert result.diagnostics["transitions"] == 1
    assert result.diagnostics["target_samples"] == 2
    assert result.diagnostics["target_objects"] == 4
    assert result.diagnostics["target_supervised_tokens"] == 10
    assert result.diagnostics["complete_inventory_samples"] == 2
    assert result.diagnostics["set_matches"] == 4
    assert result.diagnostics["dynamics_matched_predictions"] > 0
