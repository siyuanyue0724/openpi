from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import picf_next.lingbot_native.frozen_posterior_lingbot_adapter as adapter_module
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.frozen_posterior_diagnostic import (
    FrozenPosteriorActionRequest,
    FrozenPosteriorVisibility,
    LanguagePromptBatch,
)
from picf_next.lingbot_native.frozen_posterior_lingbot_adapter import (
    LingBotFrozenPosteriorActionAdapter,
    frozen_posterior_action_information_contract,
    run_native_frozen_posterior_action_forward,
)
from picf_next.lingbot_native.host import (
    UNIFIED_LAYERWISE_PREDICT_CORRECT,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.state import NativeLayerwisePosteriorState


class _FakeLanguageLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.hidden_size = width

    def forward(
        self,
        hidden: torch.Tensor,
        attention_output: torch.Tensor | None = None,
        start: int | None = None,
        end: int | None = None,
        *,
        compute_kqv: bool = False,
        output_atten: bool = False,
        qk_input_bias: torch.Tensor | None = None,
        **_: object,
    ):
        if compute_kqv:
            qk_hidden = hidden if qk_input_bias is None else hidden + qk_input_bias
            return (
                qk_hidden.unsqueeze(2),
                (0.75 * qk_hidden).unsqueeze(2),
                hidden.unsqueeze(2),
            )
        if output_atten:
            assert attention_output is not None and start is not None and end is not None
            return hidden + attention_output[:, start:end, 0]
        raise AssertionError("unexpected fake LingBot layer call")


class _FakeJointHost(nn.Module):
    def __init__(self, graph: LingBotNativeGraph, *, vocabulary: int = 64) -> None:
        super().__init__()
        width = graph.config.host_width
        layers = graph.config.num_layers
        self.picf_native_graph = graph
        self.config = SimpleNamespace(attention_implementation="eager")
        language_model = nn.Module()
        language_model.layers = nn.ModuleList(_FakeLanguageLayer(width) for _ in range(layers))
        language_model.embed_tokens = nn.Embedding(vocabulary, width)
        language_model.norm = nn.Identity()
        self.qwenvl = nn.Module()
        self.qwenvl.model = nn.Module()
        self.qwenvl.model.language_model = language_model
        self.qwenvl.config = SimpleNamespace(text_config=SimpleNamespace(num_attention_heads=1))
        self.qwen_expert = nn.Module()
        self.qwen_expert.model = nn.Module()
        self.qwen_expert.model.layers = nn.ModuleList(nn.Identity() for _ in range(layers))
        self.prefix_attention_masks: list[torch.Tensor] = []
        self.action_cache_lengths: list[int] = []

    def embed_language_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.qwenvl.model.language_model.embed_tokens(token_ids)

    @staticmethod
    def build_prefix_position_ids(
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        image_grid_thw: None,
        video_grid_thw: None,
    ) -> torch.Tensor:
        del image_grid_thw, video_grid_thw
        positions = torch.cumsum(attention_mask.long(), dim=1) - 1
        positions = positions.masked_fill(~attention_mask.bool(), 0)
        assert positions.shape == token_ids.shape
        return positions.unsqueeze(0).expand(3, -1, -1)

    @staticmethod
    def apply_mrope(
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids
        return query, key

    def attention_interface(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        self.prefix_attention_masks.append(mask.detach().clone())
        return self._attention(query, key, value, mask)

    @staticmethod
    def _attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / math.sqrt(query.shape[-1])
        scores = scores.masked_fill(~mask[:, None], -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", weights, value)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: dict[int, dict[str, torch.Tensor]],
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
        fill_kv_cache: bool,
        ada_cond: torch.Tensor | None,
        picf_native_context: None,
    ):
        del position_ids, use_cache, fill_kv_cache, ada_cond
        assert picf_native_context is None
        assert inputs_embeds[0] is None and inputs_embeds[1] is not None
        hidden = inputs_embeds[1]
        for layer_index in range(len(self.qwenvl.model.language_model.layers)):
            cached_key = past_key_values[layer_index]["key_states"]
            cached_value = past_key_values[layer_index]["value_states"]
            self.action_cache_lengths.append(cached_key.shape[1])
            current = hidden.unsqueeze(2)
            key = torch.cat((cached_key, current), dim=1)
            value = torch.cat((cached_value, current), dim=1)
            attention = self._attention(current, key, value, attention_mask)
            hidden = hidden + attention[:, :, 0]
        return [None, hidden], past_key_values, []


class _FakeFlow(nn.Module):
    def __init__(self, joint: _FakeJointHost, *, state_dim: int, action_dim: int) -> None:
        super().__init__()
        width = joint.picf_native_graph.config.host_width
        self.qwenvl_with_expert = joint
        self.config = SimpleNamespace(
            use_cache=True,
            num_steps=2,
            n_action_steps=2,
            max_action_dim=action_dim,
            vlm_causal=False,
            adanorm_time=False,
            action_fp32=False,
        )
        self.state_projection = nn.Linear(state_dim, width, bias=False)
        self.action_projection = nn.Linear(action_dim, width, bias=False)
        self.action_out_proj = nn.Linear(width, action_dim, bias=False)
        self.embed_prefix_calls = 0

    def embed_prefix(self, *_: object, **__: object) -> None:
        self.embed_prefix_calls += 1
        raise AssertionError("isolated diagnostic must not embed RGB/current scene")

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_token = self.state_projection(state).unsqueeze(1)
        action_tokens = self.action_projection(noisy_action)
        action_tokens = action_tokens + timestep[:, None, None].to(action_tokens.dtype)
        suffix = torch.cat((state_token, action_tokens), dim=1)
        valid = torch.ones(suffix.shape[:2], dtype=torch.bool, device=suffix.device)
        block_start = torch.zeros_like(valid)
        block_start[:, :2] = True
        return timestep[:, None], suffix, valid, block_start

    @staticmethod
    def _build_full_position_ids(
        prefix_position_ids: torch.Tensor,
        prefix_valid: torch.Tensor,
        suffix_valid: torch.Tensor,
    ) -> torch.Tensor:
        prefix_position = prefix_position_ids.masked_fill(~prefix_valid.unsqueeze(0), 0)
        offset = prefix_position.amax(dim=(0, 2)) + 1
        suffix = offset[:, None] + torch.cumsum(suffix_valid.long(), dim=1) - 1
        suffix = suffix.masked_fill(~suffix_valid, 1)
        suffix = suffix.unsqueeze(0).expand(3, -1, -1)
        return torch.cat((prefix_position_ids, suffix), dim=-1)


class _FakePolicy(nn.Module):
    def __init__(self, flow: _FakeFlow) -> None:
        super().__init__()
        self.model = flow
        self.sample_action_calls = 0

    def sample_actions(self, *_: object, **__: object) -> None:
        self.sample_action_calls += 1
        raise AssertionError("isolated diagnostic must not call production sample_actions")

    def picf_native_frozen_posterior_action_forward(
        self,
        *,
        request: FrozenPosteriorActionRequest,
    ) -> torch.Tensor:
        return LingBotFrozenPosteriorActionAdapter(self)(request)


def _policy(*, layers: int = 3, capacity: int = 3, width: int = 8) -> _FakePolicy:
    torch.manual_seed(11)
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=capacity,
            host_width=width,
            executed_action_dim=3,
            num_layers=layers,
            maximum_control_tokens=2,
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
        )
    )
    policy = _FakePolicy(_FakeFlow(_FakeJointHost(graph), state_dim=5, action_dim=3))
    policy.eval()
    return policy


def _controls(*, batch: int = 2) -> ExecutedControlBatch:
    values = (
        torch.tensor(
            [[[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        .expand(batch, -1, -1)
        .clone()
    )
    token_valid = torch.tensor([[True, False]]).expand(batch, -1).clone()
    field_valid = token_valid.unsqueeze(-1).expand_as(values).clone()
    values = values.masked_fill(~field_valid, 0)
    return ExecutedControlBatch(
        values=values,
        field_valid=field_valid,
        token_valid=token_valid,
        delta_time=torch.tensor([[0.1, 0.0]]).expand(batch, -1).clone(),
        reset=torch.zeros(batch, 2, dtype=torch.bool),
        acknowledged=token_valid.clone(),
    )


def _request(
    visibility: FrozenPosteriorVisibility,
    *,
    row_visible: torch.Tensor | None = None,
    posterior_offset: float = 0.0,
) -> FrozenPosteriorActionRequest:
    batch, layers, capacity, width = 2, 3, 3, 8
    posterior = torch.arange(
        batch * layers * capacity * width,
        dtype=torch.float32,
    ).reshape(batch, layers, capacity, width)
    posterior = posterior / 100.0 + posterior_offset
    if row_visible is None:
        row_visible = torch.ones(batch, capacity, dtype=torch.bool)
    return FrozenPosteriorActionRequest(
        language=LanguagePromptBatch(
            token_ids=torch.tensor([[3, 4, 5], [6, 7, 8]]),
            token_valid=torch.ones(batch, 3, dtype=torch.bool),
        ),
        controls=_controls(batch=batch),
        proprioception=torch.arange(batch * 5, dtype=torch.float32).reshape(batch, 5) / 10,
        posterior=NativeLayerwisePosteriorState(posterior),
        posterior_row_visible=row_visible,
        inference_noise=torch.linspace(-1, 1, batch * 2 * 3).reshape(batch, 2, 3),
        visibility=visibility,
    )


@pytest.mark.parametrize(
    ("visibility", "language_reads", "direct_cache"),
    (
        (FrozenPosteriorVisibility.DIRECT_ONLY, False, True),
        (FrozenPosteriorVisibility.LANGUAGE_MEDIATED, True, False),
        (FrozenPosteriorVisibility.BOTH, True, True),
    ),
)
def test_real_adapter_enforces_each_visibility_contract(
    visibility: FrozenPosteriorVisibility,
    language_reads: bool,
    direct_cache: bool,
) -> None:
    policy = _policy()
    joint = policy.model.qwenvl_with_expert
    adapter = LingBotFrozenPosteriorActionAdapter(policy)
    request = _request(visibility)
    action = adapter(request)

    assert action.shape == request.inference_noise.shape
    assert torch.isfinite(action).all()
    assert policy.sample_action_calls == 0
    assert policy.model.embed_prefix_calls == 0
    contract = frozen_posterior_action_information_contract(visibility)
    assert contract.language_reads_posterior is language_reads
    assert contract.action_cache_contains_posterior is direct_cache

    language_count = request.language.token_ids.shape[1]
    control_count = request.controls.token_count
    posterior_count = request.posterior.capacity
    expected_prefix_key_count = language_count + control_count
    if language_reads:
        expected_prefix_key_count += posterior_count
    assert all(
        mask.shape
        == (
            request.posterior.batch_size,
            language_count + control_count,
            expected_prefix_key_count,
        )
        for mask in joint.prefix_attention_masks
    )
    expected_cache_count = language_count + control_count
    if direct_cache:
        expected_cache_count += posterior_count
    assert set(joint.action_cache_lengths) == {expected_cache_count}

    if language_reads:
        first_mask = joint.prefix_attention_masks[0]
        assert first_mask[:, :language_count, :posterior_count].all()
        assert not first_mask[:, language_count:, :posterior_count].any()
    else:
        first_mask = joint.prefix_attention_masks[0]
    prefix_offset = posterior_count if language_reads else 0
    assert not first_mask[:, :language_count, prefix_offset + language_count :].any()
    assert not first_mask[:, language_count:, prefix_offset : prefix_offset + language_count].any()


def test_real_adapter_uses_call_local_cache_when_training_config_disables_cache() -> None:
    policy = _policy()
    policy.model.config.use_cache = False
    request = _request(FrozenPosteriorVisibility.BOTH)

    action = LingBotFrozenPosteriorActionAdapter(policy)(request)

    assert action.shape == request.inference_noise.shape
    assert torch.isfinite(action).all()
    assert policy.model.qwenvl_with_expert.action_cache_lengths


def test_real_adapter_is_deterministic_and_posterior_sensitive() -> None:
    policy = _policy()
    adapter = LingBotFrozenPosteriorActionAdapter(policy)
    factual = _request(FrozenPosteriorVisibility.BOTH)
    first = adapter(factual)
    second = adapter(factual)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    changed = adapter(
        _request(
            FrozenPosteriorVisibility.BOTH,
            posterior_offset=3.0,
        )
    )
    assert not torch.equal(first, changed)
    hidden = adapter(
        _request(
            FrozenPosteriorVisibility.BOTH,
            row_visible=torch.zeros(2, 3, dtype=torch.bool),
        )
    )
    assert not torch.equal(first, hidden)


@pytest.mark.parametrize("visibility", tuple(FrozenPosteriorVisibility))
def test_each_visibility_retains_a_posterior_to_action_path(
    visibility: FrozenPosteriorVisibility,
) -> None:
    policy = _policy()
    adapter = LingBotFrozenPosteriorActionAdapter(policy)
    factual = adapter(_request(visibility))
    changed = adapter(_request(visibility, posterior_offset=2.0))
    assert not torch.equal(factual, changed)


def test_real_adapter_prompt_switch_uses_only_language_tokens() -> None:
    policy = _policy()
    adapter = LingBotFrozenPosteriorActionAdapter(policy)
    request = _request(FrozenPosteriorVisibility.LANGUAGE_MEDIATED)
    first = adapter(request)
    switched = FrozenPosteriorActionRequest(
        language=LanguagePromptBatch(
            token_ids=request.language.token_ids + 10,
            token_valid=request.language.token_valid.clone(),
        ),
        controls=request.controls,
        proprioception=request.proprioception,
        posterior=request.posterior,
        posterior_row_visible=request.posterior_row_visible,
        inference_noise=request.inference_noise,
        visibility=request.visibility,
    )
    second = adapter(switched)
    assert not torch.equal(first, second)
    assert policy.sample_action_calls == 0
    assert policy.model.embed_prefix_calls == 0


def test_real_adapter_does_not_mutate_model_or_request() -> None:
    policy = _policy()
    adapter = LingBotFrozenPosteriorActionAdapter(policy)
    request = _request(FrozenPosteriorVisibility.DIRECT_ONLY)
    model_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    posterior_before = request.posterior.layer_rows.clone()
    noise_before = request.inference_noise.clone()
    adapter(request)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, model_before[name], rtol=0, atol=0)
    torch.testing.assert_close(request.posterior.layer_rows, posterior_before, rtol=0, atol=0)
    torch.testing.assert_close(request.inference_noise, noise_before, rtol=0, atol=0)


def test_real_adapter_fails_closed_for_training_and_shape_or_dtype_drift() -> None:
    policy = _policy()
    policy.train()
    with pytest.raises(RuntimeError, match="requires eval mode"):
        LingBotFrozenPosteriorActionAdapter(policy)

    policy.eval()
    adapter = LingBotFrozenPosteriorActionAdapter(policy)
    request = _request(FrozenPosteriorVisibility.DIRECT_ONLY)
    bad_noise = FrozenPosteriorActionRequest(
        language=request.language,
        controls=request.controls,
        proprioception=request.proprioception,
        posterior=request.posterior,
        posterior_row_visible=request.posterior_row_visible,
        inference_noise=torch.zeros(2, 1, 3),
        visibility=request.visibility,
    )
    with pytest.raises(ValueError, match="released action surface"):
        adapter(bad_noise)

    bad_dtype = FrozenPosteriorActionRequest(
        language=request.language,
        controls=request.controls,
        proprioception=request.proprioception.double(),
        posterior=request.posterior,
        posterior_row_visible=request.posterior_row_visible,
        inference_noise=request.inference_noise.double(),
        visibility=request.visibility,
    )
    with pytest.raises(TypeError, match="LingBot host"):
        adapter(bad_dtype)


def test_real_adapter_rejects_sharded_or_unmaterialized_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    addresses = policy.model.qwenvl_with_expert.picf_native_graph.object_addresses
    monkeypatch.setattr(adapter_module, "_is_dtensor", lambda value: value is addresses)
    with pytest.raises(RuntimeError, match="sharded DTensor object addresses"):
        LingBotFrozenPosteriorActionAdapter(policy)

    monkeypatch.setattr(adapter_module, "_is_dtensor", lambda value: False)
    weight = policy.model.state_projection.weight
    policy.model.state_projection.weight = nn.Parameter(
        torch.empty(weight.shape, dtype=weight.dtype, device="meta")
    )
    with pytest.raises(RuntimeError, match="materialized LingBot weights"):
        LingBotFrozenPosteriorActionAdapter(policy)


def test_information_contract_is_explicitly_fail_closed() -> None:
    for visibility in FrozenPosteriorVisibility:
        contract = frozen_posterior_action_information_contract(visibility)
        assert "frozen-posterior" in contract.allowed_action_sources
        assert set(contract.forbidden_action_sources) == {
            "current-rgb",
            "current-dense-modality",
            "history",
            "prior",
            "external-trace",
            "host-aux",
            "match",
            "ground-truth-action",
            "target-row",
            "sidecar-label",
        }


def test_public_root_wrapper_uses_only_the_registered_action_method() -> None:
    policy = _policy()
    request = _request(FrozenPosteriorVisibility.BOTH)

    expected = LingBotFrozenPosteriorActionAdapter(policy)(request)
    actual = run_native_frozen_posterior_action_forward(policy, request=request)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert policy.sample_action_calls == 0
    assert policy.model.embed_prefix_calls == 0


def test_public_root_wrapper_fails_closed_for_training_or_missing_method() -> None:
    policy = _policy()
    request = _request(FrozenPosteriorVisibility.DIRECT_ONLY)
    policy.train()
    with pytest.raises(ValueError, match="eval mode"):
        run_native_frozen_posterior_action_forward(policy, request=request)

    missing = nn.Linear(2, 2).eval()
    with pytest.raises(TypeError, match="registered frozen-posterior root method"):
        run_native_frozen_posterior_action_forward(missing, request=request)
