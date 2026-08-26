from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import (
    NATIVE_VIDEOMT_PRETRAINED_OBJECT_MEMORY_POSTERIOR,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    install_lingbot_native_graph,
)
from picf_next.lingbot_native.modalities import (
    CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
    NativeObjectQuerySpatialRelation,
    NativeObjectQuerySpatialSpec,
)
from picf_next.lingbot_native.pretrained_object_memory import PretrainedQwen3ObjectMemory
from picf_next.lingbot_native.state import NativeLayerwisePriorTrace
from picf_next.lingbot_native.training import audit_native_optimizer_coverage


class _FakeQwen3Merger(nn.Module):
    def __init__(self, *, grouped_width: int = 8, host_width: int = 8) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(grouped_width, grouped_width)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(grouped_width, host_width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(value)))


class _FakeQwen3Visual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_merge_size = 2
        self.merger = _FakeQwen3Merger()

    def forward(self, value: torch.Tensor, *, grid_thw: torch.Tensor) -> torch.Tensor:
        del grid_thw
        return self.merger(value)


def _spatial_spec() -> NativeObjectQuerySpatialSpec:
    return NativeObjectQuerySpatialSpec(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
    )


def _relation(
    mask_logits: torch.Tensor,
    *,
    query_count: int,
) -> NativeObjectQuerySpatialRelation:
    batch = mask_logits.shape[0]
    query_valid = torch.ones(batch, query_count, dtype=torch.bool)
    canonical = torch.arange(query_count).expand(batch, -1)
    class_logits = torch.zeros(batch, query_count, 3, dtype=mask_logits.dtype)
    return NativeObjectQuerySpatialRelation(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout="videomt.calvin.static.2x2.v1",
        object_logits=torch.zeros(batch, query_count, dtype=mask_logits.dtype),
        mask_logits=mask_logits,
        query_valid=query_valid,
        pixel_valid=torch.ones(batch, 4, dtype=torch.bool),
        canonical_query_ids=canonical,
        grid_shape=(2, 2),
        class_logits=class_logits,
    )


def _installed_memory(*, capacity: int) -> tuple[PretrainedQwen3ObjectMemory, _FakeQwen3Visual]:
    torch.manual_seed(225)
    visual = _FakeQwen3Visual()
    memory = PretrainedQwen3ObjectMemory(capacity=capacity)
    memory.install_from_qwen3_visual(
        visual,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return memory, visual


def test_install_copies_complete_native_merger_mlp_exactly() -> None:
    memory, visual = _installed_memory(capacity=2)
    projection = memory.projection
    assert projection is not None
    assert type(projection.act_fn) is nn.GELU
    torch.testing.assert_close(
        projection.linear_fc1.weight,
        visual.merger.linear_fc1.weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        projection.linear_fc2.bias,
        visual.merger.linear_fc2.bias,
        rtol=0,
        atol=0,
    )
    receipt = memory.installation_receipt()
    assert receipt["source_parameter_sha256"] == receipt["copied_parameter_sha256"]
    assert receipt["camera_slot"] == 0


def test_one_hot_and_binary_pooling_match_unipixel_mean_then_native_mlp() -> None:
    memory, _visual = _installed_memory(capacity=2)
    grouped = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    weights = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
    output = memory.encode_mask_weights(
        grouped_features=grouped,
        mask_weights=weights,
        query_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    projection = memory.projection
    assert projection is not None
    expected = projection(
        torch.stack((grouped[0, 0], grouped[0, 1:3].mean(dim=0))).unsqueeze(0)
    )
    torch.testing.assert_close(output.tokens, expected)


def test_soft_mask_limit_converges_to_binary_pooling() -> None:
    memory, _visual = _installed_memory(capacity=1)
    grouped = torch.randn(1, 4, 8)
    binary = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    logits = torch.where(
        binary.bool(),
        torch.full_like(binary, 30.0),
        torch.full_like(binary, -30.0),
    )
    soft = memory.encode_mask_weights(
        grouped_features=grouped,
        mask_weights=torch.sigmoid(logits),
        query_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    hard = memory.encode_mask_weights(
        grouped_features=grouped,
        mask_weights=binary,
        query_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(soft.tokens, hard.tokens, rtol=1e-5, atol=1e-6)


def test_capture_selects_static_camera_and_is_consumed_once() -> None:
    memory, visual = _installed_memory(capacity=2)
    static = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    wrist_left = torch.full((4, 8), 1000.0)
    wrist_right = torch.full((4, 8), 2000.0)
    grouped = torch.cat((static, wrist_left, wrist_right), dim=0)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]], dtype=torch.long)
    visual(grouped, grid_thw=grid)
    logits = torch.tensor([[[30.0, -30.0, -30.0, -30.0], [-30.0, 30.0, 30.0, -30.0]]])
    output = memory.consume(
        _relation(logits, query_count=2),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    projection = memory.projection
    assert projection is not None
    expected = projection(
        torch.stack((static[0], static[1:3].mean(dim=0))).unsqueeze(0)
    )
    torch.testing.assert_close(output.tokens, expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(RuntimeError, match="missing, stale or already used"):
        memory.consume(
            _relation(logits, query_count=2),
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_capture_preserves_batch_major_three_camera_order() -> None:
    memory, visual = _installed_memory(capacity=1)
    static_zero = torch.full((4, 8), 1.0)
    static_one = torch.full((4, 8), 2.0)
    grouped = torch.cat(
        (
            static_zero,
            torch.full((4, 8), 101.0),
            torch.full((4, 8), 201.0),
            static_one,
            torch.full((4, 8), 102.0),
            torch.full((4, 8), 202.0),
        ),
        dim=0,
    )
    grid = torch.tensor([[1, 4, 4]] * 6, dtype=torch.long)
    visual(grouped, grid_thw=grid)
    logits = torch.tensor(
        [
            [[30.0, -30.0, -30.0, -30.0]],
            [[30.0, -30.0, -30.0, -30.0]],
        ]
    )
    output = memory.consume(
        _relation(logits, query_count=1),
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    projection = memory.projection
    assert projection is not None
    expected = projection(torch.stack((static_zero[0], static_one[0])).unsqueeze(1))
    torch.testing.assert_close(output.tokens, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("grid", "grouped_rows", "message"),
    (
        (
            torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long),
            8,
            "fixed CALVIN camera ABI",
        ),
        (
            torch.tensor([[2, 4, 4], [1, 4, 4], [1, 4, 4]], dtype=torch.long),
            16,
            "one static frame",
        ),
    ),
)
def test_capture_rejects_wrong_camera_count_or_temporal_grid(
    grid: torch.Tensor,
    grouped_rows: int,
    message: str,
) -> None:
    memory, visual = _installed_memory(capacity=1)
    visual(torch.randn(grouped_rows, 8), grid_thw=grid)
    logits = torch.tensor([[[8.0, -8.0, -8.0, -8.0]]])

    with pytest.raises(ValueError, match=message):
        memory.consume(
            _relation(logits, query_count=1),
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_bfloat16_capture_and_projection_preserve_the_host_abi() -> None:
    torch.manual_seed(225)
    visual = _FakeQwen3Visual().to(dtype=torch.bfloat16)
    memory = PretrainedQwen3ObjectMemory(capacity=1)
    memory.install_from_qwen3_visual(
        visual,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    grouped = torch.randn(12, 8, dtype=torch.bfloat16, requires_grad=True)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]], dtype=torch.long)
    visual(grouped, grid_thw=grid)
    logits = torch.tensor(
        [[[8.0, -8.0, -8.0, -8.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = memory.consume(
        _relation(logits, query_count=1),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert output.tokens.dtype == torch.bfloat16
    assert output.tokens.shape == (1, 1, 8)
    assert torch.isfinite(output.tokens).all()
    output.tokens.float().square().mean().backward()
    assert grouped.grad is not None and bool((grouped.grad.abs() > 0).any())
    assert logits.grad is not None and bool((logits.grad.abs() > 0).any())


def test_overlapping_masks_remain_independent_and_gradients_reach_both_sources() -> None:
    memory, visual = _installed_memory(capacity=2)
    grouped = torch.randn(12, 8, requires_grad=True)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]], dtype=torch.long)
    visual(grouped, grid_thw=grid)
    logits = torch.tensor(
        [[[8.0, 8.0, -8.0, -8.0], [-8.0, 8.0, 8.0, -8.0]]],
        requires_grad=True,
    )
    output = memory.consume(
        _relation(logits, query_count=2),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert not torch.equal(output.tokens[:, 0], output.tokens[:, 1])
    output.tokens.square().mean().backward()
    assert grouped.grad is not None and bool((grouped.grad[:4].abs() > 0).any())
    assert logits.grad is not None and bool((logits.grad.abs() > 0).any())
    projection = memory.projection
    assert projection is not None
    assert projection.linear_fc1.weight.grad is not None


def _controls() -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.25, -0.5]]]),
        field_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.full((1, 1), 0.1),
        reset=torch.zeros(1, 1, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )


def test_adr225_profile_uses_pretrained_memory_not_random_query_projection() -> None:
    query_count = 200
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=query_count,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(NativeModalitySpec("videomt_queries", 4, query_count),),
            object_query_spatial_specs=(_spatial_spec(),),
            architecture_identity=NATIVE_VIDEOMT_PRETRAINED_OBJECT_MEMORY_POSTERIOR,
        )
    )
    assert "videomt_queries" not in graph.modality_projections
    bridge = graph.pretrained_object_memory
    assert bridge is not None
    visual = _FakeQwen3Visual()
    bridge.install_from_qwen3_visual(
        visual,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    grouped = torch.randn(12, 8)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]], dtype=torch.long)
    visual(grouped, grid_thw=grid)

    mask_logits = torch.full((1, query_count, 4), -8.0)
    mask_logits[:, :, 0] = 8.0
    relation = _relation(mask_logits, query_count=query_count)
    canonical = torch.arange(query_count).unsqueeze(0)
    query_stream = NativeModalityStream(
        "videomt_queries",
        torch.randn(1, query_count, 4),
        torch.ones(1, query_count, dtype=torch.bool),
        canonical_token_ids=canonical,
    )
    modalities = NativeModalityBatch((query_stream,), (relation,))
    context = LingBotNativeContext(
        controls=_controls(),
        modalities=modalities,
        prior_trace=NativeLayerwisePriorTrace(torch.zeros(1, 3, query_count, 8)),
        native_roles=torch.tensor(
            [[int(NativeRole.SENSOR), int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]]
        ),
        native_valid=torch.ones(1, 3, dtype=torch.bool),
        instruction_last_index=torch.tensor([2]),
    )
    prefix = torch.randn(1, 3, 8)
    action = torch.randn(1, 2, 4)
    inputs, _mask, _positions, _visual, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=torch.ones(1, 5, 5, dtype=torch.bool),
        position_ids=torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone(),
        visual_pos_masks=torch.tensor([[True, True, False]]),
        context=context,
    )
    torch.testing.assert_close(inputs[0][:, : prefix.shape[1]], prefix, rtol=0, atol=0)
    assert context.object_memory_capture_generation == 1
    assert context.object_memory_support_mass is not None
    assert context.object_memory_query_valid is not None
    projection = bridge.projection
    assert projection is not None
    expected_memory = projection(grouped[0].unsqueeze(0).expand(query_count, -1)).unsqueeze(0)
    expected = expected_memory + graph.modality_embeddings[0] + graph.role_embeddings[1]
    torch.testing.assert_close(
        inputs[0][:, runtime.posterior_slice],
        expected,
        rtol=2e-3,
        atol=2e-3,
    )


def test_object_relation_cannot_drift_from_same_index_source_validity() -> None:
    relation = _relation(
        torch.zeros(1, 2, 4),
        query_count=2,
    )
    stream = NativeModalityStream(
        "videomt_queries",
        torch.randn(1, 2, 4),
        torch.tensor([[True, False]]),
        canonical_token_ids=torch.tensor([[0, -1]]),
    )

    with pytest.raises(ValueError, match="axes differ from canonical source queries"):
        NativeModalityBatch((stream,), (relation,))


def test_graph_installation_binds_the_exact_qwen3_visual_merger() -> None:
    query_count = 200
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=query_count,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(NativeModalitySpec("videomt_queries", 4, query_count),),
            object_query_spatial_specs=(_spatial_spec(),),
            architecture_identity=NATIVE_VIDEOMT_PRETRAINED_OBJECT_MEMORY_POSTERIOR,
        )
    )
    visual = _FakeQwen3Visual()
    language_model = SimpleNamespace(
        layers=tuple(SimpleNamespace(hidden_size=8) for _ in range(3)),
        config=SimpleNamespace(initializer_range=0.02),
    )
    action_model = SimpleNamespace(layers=tuple(object() for _ in range(3)))
    host = SimpleNamespace(
        qwenvl=SimpleNamespace(
            model=SimpleNamespace(language_model=language_model),
            visual=visual,
        ),
        qwen_expert=SimpleNamespace(model=action_model),
    )

    def set_graph(value: LingBotNativeGraph) -> None:
        host.picf_native_graph = value

    host.set_picf_native_graph = set_graph
    policy = SimpleNamespace(
        model=SimpleNamespace(
            qwenvl_with_expert=host,
            config=SimpleNamespace(max_action_dim=2),
        )
    )

    install_lingbot_native_graph(policy, graph)

    assert host.picf_native_graph is graph
    bridge = graph.pretrained_object_memory
    assert bridge is not None and bridge.installed
    receipt = bridge.installation_receipt()
    assert receipt["source_parameter_sha256"] == receipt["copied_parameter_sha256"]
    assert receipt["copied_projection_trainable"] is True


def test_copied_merger_projection_is_owned_once_by_the_policy_optimizer() -> None:
    query_count = 200
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=query_count,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            maximum_control_tokens=2,
            modality_specs=(NativeModalitySpec("videomt_queries", 4, query_count),),
            object_query_spatial_specs=(_spatial_spec(),),
            architecture_identity=NATIVE_VIDEOMT_PRETRAINED_OBJECT_MEMORY_POSTERIOR,
        )
    )
    bridge = graph.pretrained_object_memory
    assert bridge is not None
    bridge.install_from_qwen3_visual(
        _FakeQwen3Visual(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    policy = nn.Module()
    policy.picf_native_graph = graph
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    manifest = audit_native_optimizer_coverage(
        modules={"policy": policy},
        optimizer=optimizer,
    )

    copied_names = tuple(
        name
        for name in manifest.canonical_names
        if ".pretrained_object_memory.projection." in name
    )
    assert len(copied_names) == 4
    assert any(name.endswith("linear_fc1.weight") for name in copied_names)
    assert any(name.endswith("linear_fc2.weight") for name in copied_names)


def test_copied_merger_projection_round_trips_in_policy_state() -> None:
    config = LingBotNativeGraphConfig(
        capacity=200,
        host_width=8,
        executed_action_dim=2,
        num_layers=3,
        maximum_control_tokens=2,
        modality_specs=(NativeModalitySpec("videomt_queries", 4, 200),),
        object_query_spatial_specs=(_spatial_spec(),),
        architecture_identity=NATIVE_VIDEOMT_PRETRAINED_OBJECT_MEMORY_POSTERIOR,
    )
    source = LingBotNativeGraph(config)
    source_bridge = source.pretrained_object_memory
    assert source_bridge is not None
    source_bridge.install_from_qwen3_visual(
        _FakeQwen3Visual(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.no_grad():
        projection = source_bridge.projection
        assert projection is not None
        projection.linear_fc1.weight.add_(0.125)
    state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    copied_state_names = tuple(
        name for name in state if name.startswith("pretrained_object_memory.projection.")
    )
    assert len(copied_state_names) == 4

    restored = LingBotNativeGraph(config)
    restored_bridge = restored.pretrained_object_memory
    assert restored_bridge is not None
    restored_bridge.install_from_qwen3_visual(
        _FakeQwen3Visual(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    restored.load_state_dict(state, strict=True)

    for name in copied_state_names:
        torch.testing.assert_close(restored.state_dict()[name], state[name], rtol=0, atol=0)
