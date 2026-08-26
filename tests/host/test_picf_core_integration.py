from __future__ import annotations

import inspect
import io

import pytest

from tests.geometry_contract import synthetic_geometry_contract

torch = pytest.importorskip("torch")
configuration = pytest.importorskip("olmo.hf_model.configuration_molmoact2")
modeling = pytest.importorskip("olmo.hf_model.modeling_molmoact2")
core_module = pytest.importorskip("picf_next.models.core")
discovery_module = pytest.importorskip("picf_next.models.discovery")
evidence_module = pytest.importorskip("picf_next.models.evidence")
filter_module = pytest.importorskip("picf_next.models.filter")
host_module = pytest.importorskip("picf_next.hosts.molmoact2")
posterior_module = pytest.importorskip("picf_next.posterior")
temporal = pytest.importorskip("picf_next.models.temporal")

MolmoAct2ActionExpertConfig = configuration.MolmoAct2ActionExpertConfig
ActionExpert = modeling.ActionExpert
PICFCore = core_module.PICFCore
PICFCoreConfig = core_module.PICFCoreConfig
ObjectDiscoveryConfig = discovery_module.ObjectDiscoveryConfig
TaskIndependentObjectDiscovery = discovery_module.TaskIndependentObjectDiscovery
ModalityProjectionSpec = evidence_module.ModalityProjectionSpec
MultimodalBindingProjector = evidence_module.MultimodalBindingProjector
NativeTokenBank = evidence_module.NativeTokenBank
PersistentObjectFilter = filter_module.PersistentObjectFilter
MolmoAct2PICFActionExpert = host_module.MolmoAct2PICFActionExpert
PICFActionEvidence = host_module.PICFActionEvidence
BIRTH_EVENT = posterior_module.BIRTH_EVENT
ObjectBeliefBatch = temporal.ObjectBeliefBatch
TemporalFilterConfig = temporal.TemporalFilterConfig
GEOMETRY = synthetic_geometry_contract(2)


def _core() -> PICFCore:
    projector = MultimodalBindingProjector(
        (
            ModalityProjectionSpec("vjepa", token_dim=6, geometry_dim=2),
            ModalityProjectionSpec("anytouch", token_dim=5, require_single_active_group=True),
        ),
        binding_dim=8,
    )
    discovery = TaskIndependentObjectDiscovery(
        ObjectDiscoveryConfig(
            input_dim=8,
            hidden_dim=12,
            num_queries=3,
            num_layers=2,
            num_heads=3,
            address_dim=2,
            content_dim=2,
            geometry_dim=2,
            geometry_contract=GEOMETRY,
            initial_variance=0.1,
        )
    )
    with torch.no_grad():
        discovery.existence_head.weight.zero_()
        discovery.existence_head.bias.fill_(6.0)
        discovery.localization_confidence_head.weight.zero_()
        discovery.localization_confidence_head.bias.fill_(6.0)
    temporal_config = TemporalFilterConfig(
        address_dim=2,
        content_dim=2,
        geometry_dim=2,
        geometry_contract=GEOMETRY,
        action_dim=3,
        reference_delta_t_s=0.1,
        hidden_dim=12,
        num_layers=2,
        num_heads=3,
    )
    return PICFCore(projector, discovery, PersistentObjectFilter(temporal_config))


def _core_config() -> PICFCoreConfig:
    return PICFCoreConfig(
        modality_specs=(
            ModalityProjectionSpec("vjepa", token_dim=6, geometry_dim=2),
            ModalityProjectionSpec(
                "anytouch",
                token_dim=5,
                require_single_active_group=True,
            ),
        ),
        binding_dim=8,
        discovery=ObjectDiscoveryConfig(
            input_dim=8,
            hidden_dim=12,
            num_queries=3,
            num_layers=2,
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
            num_layers=2,
            num_heads=3,
        ),
        posterior_capacity=3,
    )


def _banks() -> tuple[NativeTokenBank, ...]:
    torch.manual_seed(653)
    vision_valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    touch_valid = torch.tensor([[True, True, True], [False, False, False]])
    return (
        NativeTokenBank(
            "vjepa",
            torch.randn(2, 4, 6) * vision_valid.unsqueeze(-1),
            vision_valid,
            torch.randn(2, 4, 2) * vision_valid.unsqueeze(-1),
        ),
        NativeTokenBank(
            "anytouch",
            torch.randn(2, 3, 5) * touch_valid.unsqueeze(-1),
            touch_valid,
            group_id=torch.tensor([[4, 4, 4], [-1, -1, -1]], dtype=torch.long),
        ),
    )


def _empty_prior() -> ObjectBeliefBatch:
    valid = torch.zeros(2, 3, dtype=torch.bool)
    return ObjectBeliefBatch(
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


def _expert() -> ActionExpert:
    return ActionExpert(
        MolmoAct2ActionExpertConfig(
            max_action_horizon=3,
            max_action_dim=4,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            ffn_multiple_of=8,
            timestep_embed_dim=8,
            dropout=0.0,
            attn_dropout=0.0,
        ),
        llm_dim=16,
        llm_kv_dim=8,
        llm_num_layers=2,
    )


def test_core_config_builds_all_shared_dimensions_from_one_contract() -> None:
    config = _core_config()
    core = config.build()
    belief = config.empty_belief(batch_size=2, dtype=torch.float32)

    assert isinstance(core, PICFCore)
    assert config.dense_token_dims == {"vjepa": 6, "anytouch": 5}
    assert config.object_address_dim == 2
    assert config.object_value_dim == 18
    assert belief.valid.shape == (2, 3)
    assert not belief.valid.any()

    with pytest.raises(ValueError, match="binding width"):
        PICFCoreConfig(
            modality_specs=config.modality_specs,
            binding_dim=7,
            discovery=config.discovery,
            temporal=config.temporal,
            posterior_capacity=3,
        )


def test_full_core_retains_native_tokens_and_builds_persistent_action_bank() -> None:
    torch.manual_seed(659)
    core = _core().eval()
    banks = _banks()
    output = core(banks, _empty_prior(), torch.zeros(2, 3), torch.full((2,), 0.1))

    assert output.projection.total_tokens == 7
    assert all(
        actual is expected
        for actual, expected in zip(output.projection.native_banks, banks, strict=True)
    )
    assert output.action_bank.address.shape == (2, 3, 2)
    assert output.action_bank.value.shape == (2, 3, 18)
    assert output.action_bank.valid.all()
    assert tuple(item.shape[1] for item in output.dense_ownership) == (4, 3)
    torch.testing.assert_close(output.posterior.ownership.sum(dim=-1), torch.ones(2, 7))
    torch.testing.assert_close(
        output.posterior.ownership[0, 4:7],
        output.posterior.ownership[0, 4].expand(3, -1),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(output.posterior.ownership[1, 4:, :-1], torch.zeros(3, 3))


def test_float32_core_uses_bfloat16_autocast_at_typed_runtime_boundaries() -> None:
    core = _core().float().eval()
    bf16_banks = tuple(
        NativeTokenBank(
            modality=bank.modality,
            tokens=bank.tokens.to(torch.bfloat16),
            valid=bank.valid,
            geometry=(None if bank.geometry is None else bank.geometry.to(torch.bfloat16)),
            group_id=bank.group_id,
        )
        for bank in _banks()
    )
    empty = _empty_prior()
    prior = ObjectBeliefBatch(
        address_mean=empty.address_mean.to(torch.bfloat16),
        content_mean=empty.content_mean.to(torch.bfloat16),
        geometry_mean=empty.geometry_mean.to(torch.bfloat16),
        geometry_covariance_diag=empty.geometry_covariance_diag.to(torch.bfloat16),
        existence_logits=empty.existence_logits.to(torch.bfloat16),
        visibility_given_existence_logits=(
            empty.visibility_given_existence_logits.to(torch.bfloat16)
        ),
        measurement_age_s=empty.measurement_age_s.to(torch.bfloat16),
        valid=empty.valid,
        age=empty.age,
    )
    action = torch.zeros(2, 3, dtype=torch.bfloat16)
    delta_t = torch.full((2,), 0.1, dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = core(bf16_banks, prior, action, delta_t)
        second = core(bf16_banks, first.posterior.belief, action, delta_t)

    assert all(parameter.dtype == torch.float32 for parameter in core.parameters())
    assert first.posterior.belief.valid.any()
    assert second.posterior.prior_prediction.survival_probability.dtype == torch.float32
    assert second.posterior.belief.address_mean.dtype == torch.bfloat16
    assert second.action_bank.address.dtype == torch.bfloat16
    assert second.action_bank.value.dtype == torch.bfloat16
    for value in (
        second.posterior.belief.state_mean,
        second.posterior.belief.geometry_covariance_diag,
        second.action_bank.value,
    ):
        assert torch.isfinite(value).all()


def test_all_missing_modalities_leave_posterior_empty_without_fake_tokens() -> None:
    core = _core().eval()
    banks = (
        NativeTokenBank(
            "vjepa",
            torch.zeros(2, 0, 6),
            torch.zeros(2, 0, dtype=torch.bool),
            torch.zeros(2, 0, 2),
        ),
        NativeTokenBank(
            "anytouch",
            torch.zeros(2, 0, 5),
            torch.zeros(2, 0, dtype=torch.bool),
            group_id=torch.full((2, 0), -1, dtype=torch.long),
        ),
    )

    output = core(banks, _empty_prior(), torch.zeros(2, 3), torch.full((2,), 0.1))

    assert output.projection.total_tokens == 0
    assert not output.discovery.evidence_available.any()
    assert not output.action_bank.valid.any()
    assert output.posterior.ownership.shape == (2, 0, 4)
    assert all(item.shape[1] == 0 for item in output.dense_ownership)


def test_duplicated_historical_context_cannot_recorrect_the_current_posterior() -> None:
    torch.manual_seed(657)
    core = _core().eval()
    current_tokens = torch.tensor(
        [[[0.2, -0.1, 0.4, 0.3, -0.2, 0.1], [-0.3, 0.5, 0.1, -0.2, 0.4, 0.2]]]
    ).expand(2, -1, -1)
    current_geometry = torch.tensor([[[0.2, 0.3], [0.6, 0.1]]]).expand(2, -1, -1)

    def banks(history_count: int) -> tuple[NativeTokenBank, ...]:
        history_tokens = (
            torch.arange(
                2 * history_count * 6,
                dtype=torch.float32,
            ).reshape(2, history_count, 6)
            / 17.0
        )
        history_geometry = (
            torch.arange(
                2 * history_count * 2,
                dtype=torch.float32,
            ).reshape(2, history_count, 2)
            / 13.0
        )
        tokens = torch.cat((history_tokens, current_tokens), dim=1)
        geometry = torch.cat((history_geometry, current_geometry), dim=1)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool)
        current = torch.zeros_like(valid)
        current[:, history_count:] = True
        timestamps = torch.cat(
            (
                torch.linspace(0.0, 0.1, history_count).expand(2, -1),
                torch.full((2, 2), 0.2),
            ),
            dim=1,
        )
        return (
            NativeTokenBank(
                "vjepa",
                tokens,
                valid,
                geometry,
                timestamps=timestamps,
                current_measurement_valid=current,
            ),
            NativeTokenBank(
                "anytouch",
                torch.zeros(2, 0, 5),
                torch.zeros(2, 0, dtype=torch.bool),
                group_id=torch.full((2, 0), -1, dtype=torch.long),
            ),
        )

    expected = core(
        banks(2),
        _empty_prior(),
        torch.zeros(2, 3),
        torch.full((2,), 0.1),
    )
    duplicated = core(
        banks(5),
        _empty_prior(),
        torch.zeros(2, 3),
        torch.full((2,), 0.1),
    )

    pairs = (
        (duplicated.discovery.address_mean, expected.discovery.address_mean),
        (duplicated.discovery.content_mean, expected.discovery.content_mean),
        (duplicated.discovery.geometry_mean, expected.discovery.geometry_mean),
        (duplicated.discovery.geometry_variance, expected.discovery.geometry_variance),
        (duplicated.discovery.existence_logits, expected.discovery.existence_logits),
        (duplicated.posterior.belief.state_mean, expected.posterior.belief.state_mean),
        (
            duplicated.posterior.belief.geometry_covariance_diag,
            expected.posterior.belief.geometry_covariance_diag,
        ),
        (
            duplicated.posterior.belief.existence_logits,
            expected.posterior.belief.existence_logits,
        ),
        (
            duplicated.posterior.belief.visibility_given_existence_logits,
            expected.posterior.belief.visibility_given_existence_logits,
        ),
        (duplicated.posterior.belief.valid, expected.posterior.belief.valid),
        (duplicated.posterior.belief.age, expected.posterior.belief.age),
        (duplicated.posterior.event_type, expected.posterior.event_type),
        (duplicated.action_bank.address, expected.action_bank.address),
        (duplicated.action_bank.value, expected.action_bank.value),
        (duplicated.action_bank.valid, expected.action_bank.valid),
    )
    for actual, reference in pairs:
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)

    assert duplicated.projection.token_valid.sum(dim=1).tolist() == [7, 7]
    assert duplicated.projection.current_measurement_valid.sum(dim=1).tolist() == [2, 2]
    assert torch.equal(
        duplicated.posterior.ownership[:, :5, -1],
        torch.ones(2, 5),
    )
    assert duplicated.projection.native_banks[0].tokens.shape[1] == 7


def test_core_output_is_accepted_by_deep_molmoact2_context_without_compression() -> None:
    torch.manual_seed(661)
    output = _core().eval()(
        _banks(),
        _empty_prior(),
        torch.zeros(2, 3),
        torch.full((2,), 0.1),
    )
    adapter = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vjepa": 6, "anytouch": 5},
        object_address_dim=2,
        object_value_dim=18,
    ).eval()
    evidence = PICFActionEvidence(
        dense_banks=output.projection.native_banks,
        object_address=output.action_bank.address,
        object_value=output.action_bank.value,
        object_valid=output.action_bank.valid,
        object_log_prior=output.action_bank.log_prior,
        dense_ownership=output.dense_ownership,
    )
    context = adapter.prepare_picf_context(evidence)

    assert context.dense_kv_contexts is not None
    assert context.object_kv_contexts is not None
    assert context.dense_kv_contexts[0][0].shape[1] == 7
    assert context.object_kv_contexts[0][0].shape[1] == 3
    assert context.dense_mask.shape[-1] == 7
    assert context.object_mask.shape[-1] == 3


def test_core_has_no_task_label_or_mask_forward_argument_and_retains_gradients() -> None:
    assert tuple(inspect.signature(PICFCore.forward).parameters) == (
        "self",
        "native_banks",
        "prior",
        "previous_executed_action",
        "delta_t_s",
    )
    torch.manual_seed(673)
    core = _core().train()
    output = core(_banks(), _empty_prior(), torch.zeros(2, 3), torch.full((2,), 0.1))
    loss = output.action_bank.value.square().mean() + output.posterior.ownership.square().mean()
    loss.backward()

    assert core.projector.content_projection["vjepa"].weight.grad is not None
    assert core.projector.content_projection["anytouch"].weight.grad is not None
    assert core.discovery.ownership_query.weight.grad is not None
    assert core.discovery.content_head.weight.grad is not None


def test_action_loss_reaches_discovery_through_persistent_dense_ownership() -> None:
    torch.manual_seed(675)
    core = _core().train()
    output = core(_banks(), _empty_prior(), torch.zeros(2, 3), torch.full((2,), 0.1))
    expert = _expert()
    with torch.no_grad():
        for block in expert.blocks:
            torch.nn.init.xavier_uniform_(block.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.linear.weight)
    adapter = MolmoAct2PICFActionExpert(
        expert,
        dense_token_dims={"vjepa": 6, "anytouch": 5},
        object_address_dim=2,
        object_value_dim=18,
    ).train()
    for branch in adapter.dense_branches:
        branch.gate.data.fill_(0.2)
    evidence = PICFActionEvidence(
        dense_banks=output.projection.native_banks,
        object_address=output.action_bank.address.detach(),
        object_value=output.action_bank.value.detach(),
        object_valid=output.action_bank.valid,
        object_log_prior=output.action_bank.log_prior.detach(),
        dense_ownership=output.dense_ownership,
    )
    actions = torch.randn(2, 3, 4)
    timesteps = torch.tensor([0.2, 0.8])
    native_kv = [(torch.randn(2, 5, 8), torch.randn(2, 5, 8)) for _ in range(2)]

    predicted_action = adapter(
        actions,
        timesteps,
        encoder_kv_states=native_kv,
        encoder_attention_mask=torch.ones(2, 5, dtype=torch.bool),
        evidence=evidence,
    )
    predicted_action.square().mean().backward()

    gradient = core.discovery.ownership_query.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0.0
    assert adapter.dense_owner_v_proj.weight.grad is not None


def test_direct_adapter_move_without_owning_host_fails_fast() -> None:
    adapter = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vjepa": 6, "anytouch": 5},
        object_address_dim=2,
        object_value_dim=18,
    ).to(dtype=torch.float64)
    native_kv = [
        (torch.randn(2, 5, 8, dtype=torch.float64), torch.randn(2, 5, 8, dtype=torch.float64))
        for _ in range(2)
    ]
    with pytest.raises(ValueError, match="must use float32 storage"):
        adapter(
            torch.randn(2, 3, 4, dtype=torch.float64),
            torch.tensor([0.2, 0.8], dtype=torch.float64),
            encoder_kv_states=native_kv,
            encoder_attention_mask=torch.ones(2, 5, dtype=torch.bool),
        )


def test_object_key_is_address_only_while_dynamic_changes_remain_in_value() -> None:
    torch.manual_seed(677)
    output = _core().eval()(
        _banks(),
        _empty_prior(),
        torch.zeros(2, 3),
        torch.full((2,), 0.1),
    )
    bank = output.action_bank
    dynamic_width = 4
    predicted_dynamic = torch.cat(
        (
            output.posterior.prior_prediction.belief.content_mean,
            output.posterior.prior_prediction.belief.geometry_mean,
        ),
        dim=-1,
    )
    posterior_dynamic = torch.cat(
        (
            output.posterior.belief.content_mean,
            output.posterior.belief.geometry_mean,
        ),
        dim=-1,
    )
    expected_dynamic = torch.where(
        output.posterior.born.unsqueeze(-1),
        posterior_dynamic,
        predicted_dynamic,
    )
    predicted_lifecycle = torch.stack(
        (
            output.posterior.prior_prediction.belief.existence,
            output.posterior.prior_prediction.belief.visibility,
        ),
        dim=-1,
    )
    posterior_lifecycle = torch.stack(
        (
            output.posterior.belief.existence,
            output.posterior.belief.visibility,
        ),
        dim=-1,
    )
    expected_lifecycle = torch.where(
        output.posterior.born.unsqueeze(-1),
        posterior_lifecycle,
        predicted_lifecycle,
    )

    torch.testing.assert_close(bank.address, output.posterior.belief.address_mean)
    torch.testing.assert_close(bank.value[..., :dynamic_width], expected_dynamic)
    torch.testing.assert_close(bank.value[..., 14:16], expected_lifecycle)
    torch.testing.assert_close(bank.value[..., 16:18], posterior_lifecycle)
    assert bank.address.shape[-1] < bank.value.shape[-1]


def test_checkpoint_roundtrip_preserves_explicit_multistep_posterior() -> None:
    torch.manual_seed(683)
    source = _core().eval()
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)

    restored = _core().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    source_prior = _empty_prior()
    restored_prior = _empty_prior()
    actions = (
        torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1]]),
        torch.tensor([[0.0, 0.4, -0.1], [0.2, -0.1, -0.4]]),
        torch.tensor([[-0.2, 0.1, 0.2], [0.1, 0.3, -0.2]]),
    )
    for action in actions:
        delta_t_s = torch.full((2,), 0.1)
        expected = source(_banks(), source_prior, action, delta_t_s)
        actual = restored(_banks(), restored_prior, action, delta_t_s)
        torch.testing.assert_close(
            actual.posterior.belief.state_mean, expected.posterior.belief.state_mean
        )
        torch.testing.assert_close(
            actual.posterior.belief.geometry_covariance_diag,
            expected.posterior.belief.geometry_covariance_diag,
        )
        torch.testing.assert_close(actual.posterior.innovation, expected.posterior.innovation)
        torch.testing.assert_close(actual.posterior.ownership, expected.posterior.ownership)
        source_prior = expected.posterior.belief
        restored_prior = actual.posterior.belief


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is unavailable")
def test_cuda_multimodal_posterior_action_backward_and_roundtrip() -> None:
    device = torch.device("cuda")
    torch.manual_seed(691)
    torch.cuda.manual_seed_all(691)
    torch.cuda.reset_peak_memory_stats(device)

    def move_bank(bank: NativeTokenBank) -> NativeTokenBank:
        return NativeTokenBank(
            modality=bank.modality,
            tokens=bank.tokens.to(device),
            valid=bank.valid.to(device),
            geometry=None if bank.geometry is None else bank.geometry.to(device),
            group_id=None if bank.group_id is None else bank.group_id.to(device),
        )

    def move_belief(belief: ObjectBeliefBatch) -> ObjectBeliefBatch:
        return ObjectBeliefBatch(
            address_mean=belief.address_mean.to(device),
            content_mean=belief.content_mean.to(device),
            geometry_mean=belief.geometry_mean.to(device),
            geometry_covariance_diag=belief.geometry_covariance_diag.to(device),
            existence_logits=belief.existence_logits.to(device),
            visibility_given_existence_logits=(belief.visibility_given_existence_logits.to(device)),
            measurement_age_s=belief.measurement_age_s.to(device),
            valid=belief.valid.to(device),
            age=belief.age.to(device),
        )

    core = _core().to(device).train()
    adapter = (
        MolmoAct2PICFActionExpert(
            _expert().to(device),
            dense_token_dims={"vjepa": 6, "anytouch": 5},
            object_address_dim=2,
            object_value_dim=18,
        )
        .to(device)
        .train()
    )
    for branch in adapter.dense_branches:
        branch.gate.data.fill_(0.2)
    for branch in adapter.object_branches:
        branch.gate.data.fill_(0.2)

    banks = tuple(move_bank(bank) for bank in _banks())
    delta_t_s = torch.full((2,), 0.1, device=device)
    first = core(
        banks,
        move_belief(_empty_prior()),
        torch.zeros(2, 3, device=device),
        delta_t_s,
    )
    second = core(
        banks,
        first.posterior.belief,
        torch.full((2, 3), 0.1, device=device),
        delta_t_s,
    )
    evidence = PICFActionEvidence(
        dense_banks=second.projection.native_banks,
        object_address=second.action_bank.address,
        object_value=second.action_bank.value,
        object_valid=second.action_bank.valid,
        object_log_prior=second.action_bank.log_prior,
        dense_ownership=second.dense_ownership,
    )
    native_kv = [
        (torch.randn(2, 5, 8, device=device), torch.randn(2, 5, 8, device=device)) for _ in range(2)
    ]
    predicted_action = adapter(
        torch.randn(2, 3, 4, device=device),
        torch.tensor([0.2, 0.8], device=device),
        encoder_kv_states=native_kv,
        encoder_attention_mask=torch.ones(2, 5, dtype=torch.bool, device=device),
        evidence=evidence,
    )
    loss = predicted_action.square().mean() + second.posterior.innovation.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    assert torch.isfinite(predicted_action).all()
    assert core.discovery.ownership_query.weight.grad is not None
    assert adapter.dense_owner_v_proj.weight.grad is not None
    assert torch.cuda.max_memory_allocated(device) < 1 * 2**30

    checkpoint = io.BytesIO()
    torch.save({"core": core.state_dict(), "adapter": adapter.state_dict()}, checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True, map_location=device)
    restored_core = _core().to(device).eval()
    restored_adapter = (
        MolmoAct2PICFActionExpert(
            _expert().to(device),
            dense_token_dims={"vjepa": 6, "anytouch": 5},
            object_address_dim=2,
            object_value_dim=18,
        )
        .to(device)
        .eval()
    )
    restored_core.load_state_dict(state["core"])
    restored_adapter.load_state_dict(state["adapter"])

    expected = core.eval()(
        banks,
        move_belief(_empty_prior()),
        torch.zeros(2, 3, device=device),
        delta_t_s,
    )
    actual = restored_core(
        banks,
        move_belief(_empty_prior()),
        torch.zeros(2, 3, device=device),
        delta_t_s,
    )
    torch.testing.assert_close(
        actual.posterior.belief.state_mean, expected.posterior.belief.state_mean
    )
