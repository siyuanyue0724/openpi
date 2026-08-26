from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from picf_next.lingbot_wla_shared import (
    LINGBOT_HOST_LAYER_COUNT,
    WLA_ADDED_TOKEN_COUNT,
    WLA_ACTION_LAYER_COUNT,
    WLA_METAQUERY_COUNT,
    WLA_META_TOKEN_COUNT,
    LingBotWLASharedInterface,
    append_wla_metaquery_surface,
    wla_calvin_action_mask,
)
from picf_next.wla_upstream import WLASourceReceipt


def test_append_wla_metaquery_surface_preserves_base_and_adds_causal_suffix() -> None:
    base_hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    base_valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    base_mask = base_valid[:, :, None] & base_valid[:, None, :]
    positions = torch.arange(4).reshape(1, 1, 4).expand(3, 2, -1).clone()
    visual = torch.tensor([[True, False, False, False], [True, True, False, False]])
    meta = torch.randn(WLA_META_TOKEN_COUNT, 8)

    hidden, mask, position_ids, visual_mask, layout = append_wla_metaquery_surface(
        base_hidden=base_hidden,
        base_attention_mask=base_mask,
        base_position_ids=positions,
        visual_pos_masks=visual,
        meta_token_embeddings=meta,
    )

    assert hidden.shape == (2, 4 + WLA_META_TOKEN_COUNT, 8)
    assert torch.equal(hidden[:, :4], base_hidden)
    assert torch.equal(mask[:, :4, :4], base_mask)
    assert not mask[:, :4, 4:].any()
    assert torch.equal(mask[:, 4:, :4], base_valid[:, None].expand(-1, WLA_META_TOKEN_COUNT, -1))
    assert torch.equal(
        mask[0, 4:, 4:],
        torch.ones(WLA_META_TOKEN_COUNT, WLA_META_TOKEN_COUNT, dtype=torch.bool).tril(),
    )
    assert torch.equal(position_ids[:, 0, 4:], torch.arange(3, 3 + WLA_META_TOKEN_COUNT).expand(3, -1))
    assert torch.equal(position_ids[:, 1, 4:], torch.arange(4, 4 + WLA_META_TOKEN_COUNT).expand(3, -1))
    assert visual_mask is not None
    assert torch.equal(visual_mask[:, :4], visual)
    assert not visual_mask[:, 4:].any()
    assert layout.query_slice.stop - layout.query_slice.start == WLA_METAQUERY_COUNT


def test_calvin_action_mask_excludes_time_padding_and_invalid_joints() -> None:
    actions = torch.zeros(2, 3, 4)
    joint_mask = torch.tensor(
        [
            [[True, True, False, True], [True, False, True, True], [True] * 4],
            [[True] * 4, [False, True, True, False], [True] * 4],
        ]
    )
    time_padding = torch.tensor([[False, True, False], [False, False, True]])
    observed = wla_calvin_action_mask(
        {
            "actions": actions,
            "joint_mask": joint_mask,
            "action_is_pad": time_padding,
        }
    )
    expected = joint_mask & ~time_padding.unsqueeze(-1)
    assert torch.equal(observed, expected)
    assert not observed[0, 1].any()
    assert not observed[1, 2].any()


class _FakeActionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reference = nn.Parameter(torch.zeros(()))
        self.action_horizon = 3
        self.action_dim = 4
        self.model = SimpleNamespace(
            transformer_blocks=nn.ModuleList([nn.Identity() for _ in range(WLA_ACTION_LAYER_COUNT)]),
            config=SimpleNamespace(cross_attention_dim=2048),
        )

    def predict_action(self, vl_embs_list, state):
        assert len(vl_embs_list) == WLA_ACTION_LAYER_COUNT
        assert state.shape[1:] == (1, 4)
        return torch.zeros(
            state.shape[0],
            self.action_horizon,
            self.action_dim,
            dtype=self.reference.dtype,
            device=self.reference.device,
        )


class _FakeHostLayer(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        attention_output: torch.Tensor | None = None,
        start: int | None = None,
        end: int | None = None,
        *,
        compute_kqv: bool = False,
        qk_input_bias: torch.Tensor | None = None,
        output_atten: bool = False,
    ):
        if compute_kqv:
            value = hidden if qk_input_bias is None else hidden + qk_input_bias
            value = value.reshape(value.shape[0], value.shape[1], 32, 64)
            return value, value, value
        if not output_atten or attention_output is None or start is None or end is None:
            raise AssertionError("fake LingBot layer received an unexpected call")
        return hidden + attention_output[:, start:end].reshape_as(hidden) * 0.01


class _FakeHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeHostLayer() for _ in range(LINGBOT_HOST_LAYER_COUNT)])
        self.norm = nn.LayerNorm(2048)
        self.embed_tokens = nn.Embedding(32, 2048)


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(language_model=_FakeHost())
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))

    @staticmethod
    def _init_added_embeddings_weights_with_mean(
        old_embeddings,
        new_embeddings,
        old_embedding_dim,
        old_num_tokens,
        added_num_tokens,
    ):
        assert old_embedding_dim == old_embeddings.embedding_dim
        assert old_num_tokens == old_embeddings.num_embeddings
        assert added_num_tokens == new_embeddings.num_embeddings
        mean = old_embeddings.weight.float().mean(dim=0)
        new_embeddings.weight.copy_(mean.to(new_embeddings.weight.dtype))


class _FakeJoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwenvl = _FakeQwen()
        self.config = SimpleNamespace(attention_implementation="eager")
        self.picf_native_graph = None

    @staticmethod
    def apply_mrope(query, key, position_ids):
        del position_ids
        return query, key

    @staticmethod
    def attention_interface(query, key, value, mask):
        assert mask.shape == (query.shape[0], query.shape[1], key.shape[1])
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / (query.shape[-1] ** 0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", weights, value)

    @staticmethod
    def _apply_deepstack(hidden, layer_index, visual_pos_masks, deepstack_visual_embeds):
        del layer_index, visual_pos_masks, deepstack_visual_embeds
        return hidden


class _FakePriorGraph:
    def __init__(self) -> None:
        self.rows: list[torch.Tensor] = []
        self.finalized = False

    def prepare_joint_inputs(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        visual_pos_masks,
        context,
    ):
        del context
        prefix = inputs_embeds[0]
        assert prefix is not None and prefix.shape[1] == 0 and inputs_embeds[1] is None
        batch, _, width = prefix.shape
        inserted = prefix.new_zeros(batch, 3, width)
        mask = torch.ones(batch, 3, 3, dtype=torch.bool, device=prefix.device)
        positions = torch.zeros(3, batch, 3, dtype=position_ids.dtype, device=prefix.device)
        visual = torch.zeros(batch, 3, dtype=torch.bool, device=prefix.device)
        return [inserted, None], mask, positions, visual, self

    @staticmethod
    def layerwise_qk_address_bias(*, prefix_hidden, runtime):
        del prefix_hidden, runtime
        return None

    @staticmethod
    def layerwise_memory_inputs(*, layer_index, runtime):
        del layer_index, runtime
        return None

    def record_layerwise_posterior(self, *, prefix_hidden, runtime, layer_index):
        assert runtime is self and layer_index == len(self.rows)
        self.rows.append(prefix_hidden.clone())

    @staticmethod
    def requires_intermediate_relation(*, layer_index, runtime):
        del layer_index, runtime
        return False

    def finalize_joint_outputs(self, *, outputs_embeds, runtime):
        assert runtime is self and outputs_embeds[1] is None
        self.finalized = True
        return outputs_embeds


def test_meta_token_initialization_preserves_host_initializer_across_fp32_to_bf16() -> None:
    interface = LingBotWLASharedInterface(
        action_head=_FakeActionHead(),
        source=WLASourceReceipt(root=__file__, commit="test", files=()),
        dtype=torch.bfloat16,
    )
    joint = _FakeJoint().to(torch.float32)
    expected = joint.qwenvl.model.language_model.embed_tokens.weight.mean(dim=0)

    interface.initialize_meta_tokens_from_lingbot(
        joint,
        newline_token_id=1,
        im_end_token_id=2,
    )

    assert interface.added_meta_token_embeddings.dtype == torch.bfloat16
    torch.testing.assert_close(
        interface.added_meta_token_embeddings.float(),
        expected.to(torch.bfloat16).float().expand_as(
            interface.added_meta_token_embeddings.float()
        ),
    )


def test_host_records_exact_wla_last_28_layer_query_surface() -> None:
    interface = LingBotWLASharedInterface(
        action_head=_FakeActionHead(),
        source=WLASourceReceipt(root=__file__, commit="test", files=()),
        dtype=torch.float32,
    )
    with torch.no_grad():
        interface.added_meta_token_embeddings.normal_()
        interface.newline_token_id.fill_(1)
        interface.im_end_token_id.fill_(2)
        interface.meta_tokens_initialized.fill_(True)
    joint = _FakeJoint()
    prefix = torch.randn(1, 5, 2048)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, -1)

    with torch.no_grad():
        output = interface.encode_host(
            joint,
            prefix_embeds=prefix,
            attention_mask=mask,
            position_ids=positions,
            visual_pos_masks=torch.zeros(1, 5, dtype=torch.bool),
            deepstack_visual_embeds=None,
        )

    assert len(output.layerwise_query_states) == WLA_ACTION_LAYER_COUNT
    assert all(
        value.shape == (1, WLA_METAQUERY_COUNT, 2048)
        for value in output.layerwise_query_states
    )
    assert output.normalized_host.shape == prefix.shape
    assert output.layout.base_count == prefix.shape[1]


def test_prior_rollout_uses_the_same_complete_host_without_an_action_expert() -> None:
    interface = LingBotWLASharedInterface(
        action_head=_FakeActionHead(),
        source=WLASourceReceipt(root=__file__, commit="test", files=()),
        dtype=torch.float32,
    )
    with torch.no_grad():
        interface.added_meta_token_embeddings.normal_()
        interface.newline_token_id.fill_(1)
        interface.im_end_token_id.fill_(2)
        interface.meta_tokens_initialized.fill_(True)
    joint = _FakeJoint()
    graph = _FakePriorGraph()
    joint.picf_native_graph = graph
    empty = torch.empty(1, 0, 2048)

    outputs, cache, router_logits = interface.run_prior_rollout(
        joint,
        attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
        position_ids=torch.empty(3, 1, 0, dtype=torch.long),
        inputs_embeds=[empty, None],
        visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
        picf_native_context=object(),
    )

    assert cache is None and router_logits == []
    assert outputs[0] is not None and outputs[0].shape == (1, 3, 2048)
    assert outputs[1] is None
    assert len(graph.rows) == LINGBOT_HOST_LAYER_COUNT
    assert graph.finalized


def test_prior_rows_are_invariant_to_wla_suffix_values() -> None:
    torch.manual_seed(7)
    joint = _FakeJoint()
    interfaces = []
    for seed in (11, 29):
        interface = LingBotWLASharedInterface(
            action_head=_FakeActionHead(),
            source=WLASourceReceipt(root=__file__, commit="test", files=()),
            dtype=torch.float32,
        )
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            interface.added_meta_token_embeddings.copy_(
                torch.randn(
                    interface.added_meta_token_embeddings.shape,
                    generator=generator,
                )
            )
            interface.newline_token_id.fill_(1)
            interface.im_end_token_id.fill_(2)
            interface.meta_tokens_initialized.fill_(True)
        interfaces.append(interface)

    traces = []
    outputs = []
    for interface in interfaces:
        graph = _FakePriorGraph()
        joint.picf_native_graph = graph
        result, _, _ = interface.run_prior_rollout(
            joint,
            attention_mask=torch.empty(1, 0, 0, dtype=torch.bool),
            position_ids=torch.empty(3, 1, 0, dtype=torch.long),
            inputs_embeds=[torch.empty(1, 0, 2048), None],
            visual_pos_masks=torch.empty(1, 0, dtype=torch.bool),
            picf_native_context=object(),
        )
        traces.append(torch.stack(graph.rows, dim=1))
        outputs.append(result[0])

    assert outputs[0] is not None and outputs[1] is not None
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(traces[0], traces[1], rtol=0.0, atol=0.0)


def test_inference_delegates_to_upstream_action_head_predict_action() -> None:
    interface = LingBotWLASharedInterface(
        action_head=_FakeActionHead(),
        source=WLASourceReceipt(root=__file__, commit="test", files=()),
        dtype=torch.float32,
    )
    with torch.no_grad():
        interface.added_meta_token_embeddings.normal_()
        interface.newline_token_id.fill_(1)
        interface.im_end_token_id.fill_(2)
        interface.meta_tokens_initialized.fill_(True)
    joint = _FakeJoint()
    prefix = torch.randn(1, 5, 2048)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, -1)

    output = interface.predict_action(
        joint,
        prefix_embeds=prefix,
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=torch.zeros(1, 5, dtype=torch.bool),
        deepstack_visual_embeds=None,
        state=torch.zeros(1, 4),
    )

    assert output.actions.shape == (1, 3, 4)
    assert torch.equal(output.actions, torch.zeros_like(output.actions))
    assert len(output.host.layerwise_query_states) == WLA_ACTION_LAYER_COUNT
