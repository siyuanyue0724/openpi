from __future__ import annotations

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.addresses import address_codebook_sha256
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.host import (
    LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    TASK_INDEPENDENT_ENTITY_POSTERIOR,
    UNIFIED_LAYERWISE_PREDICT_CORRECT,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
)
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.runtime import LingBotNativePolicyRuntime
from picf_next.lingbot_native.session import (
    NativeObservationBatch,
    NativeSessionConfig,
    NativeSessionManager,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    NativeLayerwisePosteriorState,
)


def _controls(*, reset: bool) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.tensor([[[0.25, -0.5]]]) if not reset else torch.zeros(1, 1, 2),
        field_valid=torch.full((1, 1, 2), not reset, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.tensor([[0.1 if not reset else 0.0]]),
        reset=torch.full((1, 1), reset, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )


def _value_controls(*values: float) -> ExecutedControlBatch:
    tensor = torch.tensor([[[value, -value] for value in values]], dtype=torch.float32)
    return ExecutedControlBatch(
        values=tensor,
        field_valid=torch.ones_like(tensor, dtype=torch.bool),
        token_valid=torch.ones(1, len(values), dtype=torch.bool),
        delta_time=torch.full((1, len(values)), 0.1),
        reset=torch.zeros(1, len(values), dtype=torch.bool),
        acknowledged=torch.ones(1, len(values), dtype=torch.bool),
    )


def _observation(
    *,
    sequence: int,
    reset: bool,
    modalities: NativeModalityBatch | None = None,
    controls: ExecutedControlBatch | None = None,
    prior_control_chunks: tuple[ExecutedControlBatch, ...] = (),
    reset_epoch: int = 1,
) -> NativeObservationBatch:
    resolved_controls = _controls(reset=reset) if controls is None else controls
    return NativeObservationBatch(
        environment_keys=("calvin-0",),
        reset_epochs=(reset_epoch,),
        observation_sequences=(sequence,),
        observation_times=torch.tensor([float(sequence + 1)]),
        reset=(reset,),
        controls=resolved_controls,
        modalities=modalities,
        prior_control_chunks=prior_control_chunks,
    )


def _model_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    return {
        "images": torch.randn(1, 2, 8, generator=generator),
        "img_masks": torch.ones(1, 2, dtype=torch.bool),
        "lang_tokens": torch.ones(1, 1, dtype=torch.long),
        "lang_masks": torch.ones(1, 1, dtype=torch.bool),
        "state": torch.randn(1, 4, generator=generator),
        "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
    }


class _FakeNativePolicy(nn.Module):
    def __init__(self, graph: LingBotNativeGraph) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.qwenvl_with_expert = nn.Module()
        self.model.qwenvl_with_expert.picf_native_graph = graph
        self.contexts = []
        self.prior_contexts = []
        self.call_order: list[str] = []
        self.fail = False
        self.fail_prior = False

    def sample_actions(
        self,
        *,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        image_grid_thw: torch.Tensor,
        noise: torch.Tensor,
        picf_native_context,
    ) -> torch.Tensor:
        del img_masks, lang_tokens, state, image_grid_thw
        self.call_order.append("action")
        self.contexts.append(picf_native_context)
        prefix = torch.cat((images, torch.zeros(images.shape[0], 1, 8)), dim=1)
        picf_native_context.bind_native_prefix(
            native_valid=torch.ones(prefix.shape[:2], dtype=torch.bool),
            visual_sensor_mask=torch.tensor([[True, True, False]]),
            language_start=2,
            language_valid=lang_masks,
        )
        action = torch.nn.functional.pad(noise, (0, 6))
        total = prefix.shape[1] + action.shape[1]
        prepared, attention, _, _, runtime = (
            self.model.qwenvl_with_expert.picf_native_graph.prepare_joint_inputs(
                inputs_embeds=[prefix, action],
                attention_mask=torch.ones(1, total, total, dtype=torch.bool),
                position_ids=torch.zeros(3, 1, total, dtype=torch.long),
                visual_pos_masks=torch.tensor([[True, True, False]]),
                context=picf_native_context,
            )
        )
        hidden = torch.cat((prepared[0], prepared[1]), dim=1)
        weights = attention.float() / attention.sum(dim=-1, keepdim=True).clamp_min(1)
        graph = self.model.qwenvl_with_expert.picf_native_graph
        for layer_index in range(3):
            memory_update = torch.zeros_like(hidden)
            if graph.unified_predict_correct:
                memory_inputs = graph.layerwise_memory_inputs(
                    layer_index=layer_index,
                    runtime=runtime,
                )
                assert memory_inputs is not None
                memory_hidden, _memory_address, visibility = memory_inputs
                memory_update = visibility.to(hidden.dtype) @ memory_hidden
            hidden = torch.nn.functional.layer_norm(
                hidden + weights @ hidden + memory_update,
                (8,),
            )
            if graph.layerwise_recurrence:
                graph.record_layerwise_posterior(
                    prefix_hidden=hidden,
                    runtime=runtime,
                    layer_index=layer_index,
                )
        prefix_count = prepared[0].shape[1]
        graph.finalize_joint_outputs(
            outputs_embeds=[hidden[:, :prefix_count], hidden[:, prefix_count:]],
            runtime=runtime,
        )
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        return noise + picf_native_context.posterior_state.rows.mean(dim=(1, 2))[:, None, None]

    def picf_native_prior_forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: list[torch.Tensor | None],
        visual_pos_masks: torch.Tensor,
        picf_native_context,
    ) -> tuple[()]:
        self.call_order.append("prior")
        self.prior_contexts.append(picf_native_context)
        if self.fail_prior:
            raise RuntimeError("synthetic prior failure")
        graph = self.model.qwenvl_with_expert.picf_native_graph
        prepared, attention, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            context=picf_native_context,
        )
        hidden = prepared[0]
        weights = attention.float() / attention.sum(dim=-1, keepdim=True).clamp_min(1)
        for layer_index in range(3):
            memory_update = torch.zeros_like(hidden)
            memory_inputs = graph.layerwise_memory_inputs(
                layer_index=layer_index,
                runtime=runtime,
            )
            if memory_inputs is not None:
                memory_hidden, _memory_address, visibility = memory_inputs
                memory_update = visibility.to(hidden.dtype) @ memory_hidden
            hidden = torch.nn.functional.layer_norm(
                hidden + weights @ hidden + memory_update,
                (8,),
            )
            graph.record_layerwise_posterior(
                prefix_hidden=hidden,
                runtime=runtime,
                layer_index=layer_index,
            )
        graph.finalize_joint_outputs(
            outputs_embeds=[hidden, None],
            runtime=runtime,
        )
        return ()


def _runtime(
    *,
    modality_specs: tuple[NativeModalitySpec, ...] = (),
    architecture_identity: str | None = None,
) -> tuple[LingBotNativePolicyRuntime, _FakeNativePolicy]:
    graph_options = {}
    if architecture_identity is not None:
        graph_options["architecture_identity"] = architecture_identity
    if architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ:
        graph_options["task_query_count"] = 2
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            modality_specs=modality_specs,
            **graph_options,
        )
    )
    policy = _FakeNativePolicy(graph).eval()
    sessions = NativeSessionManager(
        NativeSessionConfig(
            model_digest="test-model",
            capacity=2,
            host_width=8,
            num_layers=3 if graph.layerwise_recurrence else None,
            addressed_architecture_identity=(
                graph.config.architecture_identity
                if graph.task_query_object_value_read
                else None
            ),
            address_codebook_sha256=(
                address_codebook_sha256(graph.episode_address_codebook)
                if graph.task_query_object_value_read
                else None
            ),
        )
    )
    return LingBotNativePolicyRuntime(policy=policy, graph=graph, sessions=sessions), policy


def test_runtime_atomically_carries_only_final_posterior_between_observations() -> None:
    runtime, policy = _runtime()
    noise = torch.randn(1, 2, 2)
    first = runtime.sample_actions(
        _observation(sequence=0, reset=True),
        model_inputs=_model_inputs(),
        noise=noise,
    )
    assert len(runtime.sessions) == 1
    assert policy.contexts[0].previous_state is None
    second = runtime.sample_actions(
        _observation(sequence=1, reset=False),
        model_inputs=_model_inputs(),
        noise=noise,
    )
    assert policy.contexts[1].previous_state is not None
    torch.testing.assert_close(
        policy.contexts[1].previous_state.rows,
        first.posterior_state.rows,
    )
    assert second.actions.shape == noise.shape
    assert second.relations.ownership.shape == (1, 3, 3)


def test_task_independent_runtime_returns_the_physical_abi_across_real_age_steps() -> None:
    runtime, _policy = _runtime(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )
    noise = torch.randn(1, 2, 2)

    first = runtime.sample_actions(
        _observation(sequence=0, reset=True),
        model_inputs=_model_inputs(),
        noise=noise,
    )
    second = runtime.sample_actions(
        _observation(sequence=1, reset=False),
        model_inputs=_model_inputs(),
        noise=noise,
    )

    assert isinstance(first.relations, PhysicalRelationOutput)
    assert isinstance(second.relations, PhysicalRelationOutput)
    assert not hasattr(second.relations, "task_relevance")
    assert second.relations.ownership.shape == (1, 3, 3)


def test_runtime_aborts_failed_inference_and_accepts_an_exact_retry() -> None:
    runtime, policy = _runtime()
    observation = _observation(sequence=0, reset=True)
    policy.fail = True
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        runtime.sample_actions(
            observation,
            model_inputs=_model_inputs(),
            noise=torch.randn(1, 2, 2),
        )
    assert len(runtime.sessions) == 0
    policy.fail = False
    runtime.sample_actions(
        observation,
        model_inputs=_model_inputs(),
        noise=torch.randn(1, 2, 2),
    )
    assert len(runtime.sessions) == 1


def test_unified_runtime_executes_exact_prior_chain_before_factual_action() -> None:
    runtime, policy = _runtime(architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT)
    noise = torch.randn(1, 2, 2)
    runtime.sample_actions(
        _observation(sequence=0, reset=True),
        model_inputs=_model_inputs(),
        noise=noise,
    )

    assert policy.call_order == ["prior", "action"]
    assert policy.prior_contexts[0].previous_memory is None
    assert not policy.prior_contexts[0].previous_memory_valid.any()
    first_correction = policy.contexts[0]
    assert first_correction.previous_state is None
    assert first_correction.previous_memory is None
    assert first_correction.prior_trace is policy.prior_contexts[0].prior_trace
    assert isinstance(first_correction.posterior_memory, NativeLayerwisePosteriorState)

    chunk_a = _value_controls(0.1, 0.2)
    chunk_b = _value_controls(0.3, 0.4, 0.5)
    correction_controls = _value_controls(0.6)
    second = runtime.sample_actions(
        _observation(
            sequence=1,
            reset=False,
            controls=correction_controls,
            prior_control_chunks=(chunk_a, chunk_b, correction_controls),
        ),
        model_inputs=_model_inputs(),
        noise=noise,
    )

    assert policy.call_order == ["prior", "action", "prior", "prior", "prior", "action"]
    resumed = policy.prior_contexts[1]
    assert isinstance(resumed.previous_memory, NativeLayerwisePosteriorState)
    assert resumed.previous_memory_valid.all()
    torch.testing.assert_close(
        resumed.previous_memory.layer_rows,
        first_correction.posterior_memory.layer_rows,
    )
    assert policy.prior_contexts[2].source_prior_trace is policy.prior_contexts[1].prior_trace
    assert policy.prior_contexts[3].source_prior_trace is policy.prior_contexts[2].prior_trace
    second_correction = policy.contexts[1]
    assert second_correction.previous_state is None
    assert second_correction.previous_memory is None
    assert second_correction.prior_trace is policy.prior_contexts[3].prior_trace
    assert second.actions.shape == noise.shape
    assert isinstance(second_correction.posterior_memory, NativeLayerwisePosteriorState)


def test_task_addressed_runtime_persists_one_episode_gauge_across_prior_and_session() -> None:
    runtime, policy = _runtime(
        architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    )
    noise = torch.randn(1, 2, 2)
    runtime.sample_actions(
        _observation(sequence=0, reset=True),
        model_inputs=_model_inputs(),
        noise=noise,
    )
    first_prior = policy.prior_contexts[0].prior_trace
    first_posterior = policy.contexts[0].posterior_memory
    assert isinstance(first_posterior, AddressedLayerwisePosteriorState)
    assert first_prior is not None
    assert first_posterior.address_receipt == first_prior.address_receipt

    runtime.sample_actions(
        _observation(sequence=1, reset=False),
        model_inputs=_model_inputs(),
        noise=noise,
    )
    resumed = policy.prior_contexts[1].previous_memory
    second_prior = policy.prior_contexts[1].prior_trace
    second_posterior = policy.contexts[1].posterior_memory
    assert isinstance(resumed, AddressedLayerwisePosteriorState)
    assert isinstance(second_posterior, AddressedLayerwisePosteriorState)
    assert second_prior is not None
    assert resumed.address_receipt == first_posterior.address_receipt
    assert second_prior.address_receipt == first_posterior.address_receipt
    assert second_posterior.address_receipt == first_posterior.address_receipt

    snapshot = runtime.sessions.serialize()
    restored = NativeSessionManager.deserialize(runtime.sessions.config, snapshot)
    assert restored.serialize() == snapshot


@pytest.mark.parametrize("failure_stage", ("prior", "action"))
def test_unified_runtime_aborts_both_passes_and_accepts_exact_retry(
    failure_stage: str,
) -> None:
    runtime, policy = _runtime(architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT)
    observation = _observation(sequence=0, reset=True)
    policy.fail_prior = failure_stage == "prior"
    policy.fail = failure_stage == "action"
    message = (
        "synthetic prior failure" if failure_stage == "prior" else "synthetic inference failure"
    )
    with pytest.raises(RuntimeError, match=message):
        runtime.sample_actions(
            observation,
            model_inputs=_model_inputs(),
            noise=torch.randn(1, 2, 2),
        )
    assert len(runtime.sessions) == 0

    policy.fail_prior = False
    policy.fail = False
    runtime.sample_actions(
        observation,
        model_inputs=_model_inputs(),
        noise=torch.randn(1, 2, 2),
    )
    assert len(runtime.sessions) == 1


def test_observation_reset_covers_the_complete_prior_control_chain() -> None:
    reset_chunk = _controls(reset=True)
    correction_controls = _value_controls(0.25)
    observation = _observation(
        sequence=0,
        reset=True,
        controls=correction_controls,
        prior_control_chunks=(reset_chunk, correction_controls),
    )
    assert observation.effective_prior_control_chunks == (reset_chunk, correction_controls)

    with pytest.raises(ValueError, match="final prior-control chunk"):
        _observation(
            sequence=0,
            reset=True,
            controls=correction_controls,
            prior_control_chunks=(reset_chunk, _value_controls(0.25)),
        )


def test_runtime_rejects_target_or_identity_fields_before_opening_a_transaction() -> None:
    runtime, _ = _runtime()
    inputs = _model_inputs()
    inputs["track_id"] = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="schema mismatch.*track_id"):
        runtime.sample_actions(
            _observation(sequence=0, reset=True),
            model_inputs=inputs,
            noise=torch.randn(1, 2, 2),
        )
    assert len(runtime.sessions) == 0


def test_runtime_passes_optional_dense_modalities_through_the_same_host_transaction() -> None:
    specs = (NativeModalitySpec("touch", 3, 2),)
    runtime, policy = _runtime(modality_specs=specs)
    modalities = NativeModalityBatch(
        (
            NativeModalityStream(
                "touch",
                torch.randn(1, 2, 3),
                torch.tensor([[True, False]]),
            ),
        )
    )
    result = runtime.sample_actions(
        _observation(sequence=0, reset=True, modalities=modalities),
        model_inputs=_model_inputs(),
        noise=torch.randn(1, 2, 2),
    )
    assert policy.contexts[0].modalities is modalities
    assert result.relations.sensor_valid.shape == (1, 5)
    assert result.relations.sensor_valid.sum().item() == 3

    missing_runtime, _ = _runtime(modality_specs=specs)
    with pytest.raises(ValueError, match="require one typed runtime batch"):
        missing_runtime.sample_actions(
            _observation(sequence=0, reset=True),
            model_inputs=_model_inputs(),
            noise=torch.randn(1, 2, 2),
        )
    assert len(missing_runtime.sessions) == 0
