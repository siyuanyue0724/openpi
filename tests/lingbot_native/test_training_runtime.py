from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.addresses import EpisodeAddressState, address_codebook_sha256
from picf_next.lingbot_native.calvin import NativeCALVINRouting
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.host import (
    LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
    LEGACY_TASK_MATCH_ARCHITECTURE,
    LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    TASK_INDEPENDENT_ENTITY_POSTERIOR,
    UNIFIED_LAYERWISE_PREDICT_CORRECT,
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    LingBotPriorRolloutContext,
)
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalitySpec,
    NativeModalityStream,
    sample_native_modality_omission,
)
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.source_mask import (
    QwenWholeViewOmission,
    sample_qwen_packed_patch_mask,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    AddressedLayerwisePriorTrace,
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativePosteriorState,
    persistent_state_tensor,
)
from picf_next.lingbot_native.temporal import (
    NativeLaneConfig,
    NativeLaneError,
    NativeTrainingLaneBank,
)
from picf_next.lingbot_native.training import (
    NativeLocalBPTTStep,
    NativeTrainingLaneCoordinator,
    NativeV3FilterPredictionSpec,
    NativeV3TwoPassStep,
    _raise_first_failed_tensor_check,
    _run_native_observation_training_forward,
    _targetless_official_loss_contract,
    native_cold_state_for_episode_keys,
    native_persistent_output,
    reconstruct_native_state_no_grad,
    run_native_local_bptt,
    run_native_omitted_image_view_training_forward,
    run_native_omitted_modality_training_forward,
    run_native_policy_observation_diagnostic_forward,
    run_native_policy_relation_training_forward,
    run_native_policy_representation_training_forward,
    run_native_policy_training_forward,
    run_native_relation_local_bptt,
    run_native_representation_window,
    run_native_source_masked_training_forward,
    run_native_state_reconstruction_step,
    run_native_v3_attached_egress,
    run_native_v3_omitted_static_view_policy_training_forward,
    run_native_v3_two_pass_policy_training_forward,
    run_native_v3_two_pass_sequence,
    run_official_policy_diagnostic_forward,
    run_official_policy_training_forward,
)


def _controls(batch_size: int, *, reset: bool) -> ExecutedControlBatch:
    valid = not reset
    return ExecutedControlBatch(
        values=torch.full((batch_size, 1, 2), 0.25 if valid else 0.0),
        field_valid=torch.full((batch_size, 1, 2), valid, dtype=torch.bool),
        token_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        delta_time=torch.full((batch_size, 1), 0.1 if valid else 0.0),
        reset=torch.full((batch_size, 1), reset, dtype=torch.bool),
        acknowledged=torch.ones(batch_size, 1, dtype=torch.bool),
    )


def _model_inputs(batch_size: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(12 + batch_size)
    return {
        "action_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
        "actions": torch.randn(batch_size, 2, 2, generator=generator),
        "image_grid_thw": torch.tensor([[[1, 2, 2], [1, 2, 2]]]).expand(batch_size, -1, -1),
        "images": torch.randn(batch_size, 2, 4, 8, generator=generator),
        "img_masks": torch.ones(batch_size, 2, dtype=torch.bool),
        "joint_mask": torch.ones(batch_size, 2, 2, dtype=torch.bool),
        "lang_masks": torch.ones(batch_size, 1, dtype=torch.bool),
        "lang_tokens": torch.ones(batch_size, 1, dtype=torch.long),
        "noise": torch.randn(batch_size, 2, 2, generator=generator),
        "state": torch.randn(batch_size, 4, generator=generator),
        "time": torch.full((batch_size,), 0.5),
    }


def _routing(
    lane_id: int,
    *,
    optimizer_step: int,
    frame_index: int,
    episode_key: str,
) -> NativeCALVINRouting:
    return NativeCALVINRouting(
        lane_ids=(lane_id,),
        episode_keys=(episode_key,),
        frame_indices=(frame_index,),
        reset=(frame_index == 0,),
        sample_keys=(f"{episode_key}/{frame_index}",),
        optimizer_step=optimizer_step,
    )


class _FakeVanillaTrainingPolicy(nn.Module):
    def __init__(self, *, append_native_output: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))
        self.append_native_output = append_native_output
        self.calls: list[tuple[frozenset[str], bool, bool]] = []

    def forward(
        self,
        *,
        compute_alignment_losses: bool,
        **model_inputs: torch.Tensor,
    ) -> tuple[object, ...]:
        self.calls.append(
            (
                frozenset(model_inputs),
                compute_alignment_losses,
                torch.is_grad_enabled(),
            )
        )
        actions = model_inputs["actions"]
        action_loss = (self.scale * actions).square().mean()
        zero = action_loss * 0
        outputs: tuple[object, ...] = (
            action_loss,
            action_loss,
            zero,
            zero,
            zero,
            zero,
            {
                "batch_mean_losses": actions.square().mean(dim=(1, 2)).detach(),
                "router_z_loss": zero.detach(),
            },
            None,
            None,
            None,
            None,
        )
        return (*outputs, ()) if self.append_native_output else outputs


class _FakeOfficialTrainingPolicy(nn.Module):
    def __init__(self, graph: LingBotNativeGraph) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.qwenvl_with_expert = nn.Module()
        self.model.qwenvl_with_expert.picf_native_graph = graph
        self.last_modalities: NativeModalityBatch | None = None
        self.forward_grad_enabled: list[bool] = []
        self.observation_forward_grad_enabled: list[bool] = []
        self.prior_forward_grad_enabled: list[bool] = []
        self.call_order: list[str] = []
        self.forward_contexts: list[LingBotNativeContext] = []
        self.action_attention_callbacks: list[Callable[..., object] | None] = []
        self.observation_contexts: list[LingBotNativeContext] = []
        self.prior_contexts: list[LingBotPriorRolloutContext] = []
        self.action_randomness: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        *,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        joint_mask: torch.Tensor,
        action_is_pad: torch.Tensor,
        image_grid_thw: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor,
        picf_native_context: LingBotNativeContext,
        compute_alignment_losses: bool,
        picf_action_attention_callback: Callable[..., object] | None = None,
    ) -> tuple[object, ...]:
        assert compute_alignment_losses is False
        self.call_order.append("policy")
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        self.forward_contexts.append(picf_native_context)
        self.action_attention_callbacks.append(picf_action_attention_callback)
        self.action_randomness.append((actions, noise, time))
        self.last_modalities = picf_native_context.modalities
        del lang_tokens, state, actions, joint_mask, action_is_pad, image_grid_thw, time
        batch = images.shape[0]
        prefix = torch.cat((images.mean(dim=2), torch.zeros(batch, 1, 8)), dim=1)
        native_valid = torch.cat((img_masks, lang_masks), dim=1)
        visual_sensor_mask = torch.cat(
            (img_masks, torch.zeros_like(lang_masks)),
            dim=1,
        )
        picf_native_context.bind_native_prefix(
            native_valid=native_valid,
            visual_sensor_mask=visual_sensor_mask,
            language_start=images.shape[1],
            language_valid=lang_masks,
        )
        action_hidden = torch.nn.functional.pad(noise, (0, 6))
        graph = self.model.qwenvl_with_expert.picf_native_graph
        total_tokens = prefix.shape[1] + action_hidden.shape[1]
        attention_mask = torch.ones(batch, total_tokens, total_tokens, dtype=torch.bool)
        attention_mask[:, : prefix.shape[1], prefix.shape[1] :] = False
        prepared, attention, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=[prefix, action_hidden],
            attention_mask=attention_mask,
            position_ids=torch.zeros(3, batch, total_tokens, dtype=torch.long),
            visual_pos_masks=visual_sensor_mask,
            context=picf_native_context,
        )
        prefix_count = prepared[0].shape[1]
        hidden = torch.cat((prepared[0], prepared[1]), dim=1)
        weights = attention.float() / attention.sum(dim=-1, keepdim=True).clamp_min(1)
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
                    prefix_hidden=hidden[:, :prefix_count],
                    runtime=runtime,
                    layer_index=layer_index,
                )
            if graph.requires_intermediate_relation(
                layer_index=layer_index,
                runtime=runtime,
            ):
                graph.record_intermediate_relation(
                    normalized_prefix=torch.nn.functional.layer_norm(
                        hidden[:, :prefix_count],
                        (8,),
                    ),
                    runtime=runtime,
                    layer_index=layer_index,
                )
        graph.finalize_joint_outputs(
            outputs_embeds=[hidden[:, :prefix_count], hidden[:, prefix_count:]],
            runtime=runtime,
        )
        action_loss = hidden[:, prefix_count:].square().mean()
        router_loss = (
            0.1
            * persistent_state_tensor(native_persistent_output(picf_native_context)).square().mean()
        )
        total_loss = action_loss + router_loss
        zero = total_loss * 0
        official_outputs = (
            total_loss,
            action_loss,
            zero,
            zero,
            zero,
            zero,
            {
                "batch_mean_losses": action_loss.detach().expand(batch),
                "router_z_loss": router_loss.detach(),
            },
            None,
            None,
            None,
            None,
        )
        return (*official_outputs, picf_native_context.root_output_tensors())

    def picf_native_observation_forward(
        self,
        *,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: torch.Tensor,
        picf_native_context: LingBotNativeContext,
    ) -> tuple[torch.Tensor, ...]:
        self.call_order.append("observation")
        self.observation_forward_grad_enabled.append(torch.is_grad_enabled())
        self.observation_contexts.append(picf_native_context)
        self.last_modalities = picf_native_context.modalities
        del lang_tokens, image_grid_thw
        batch = images.shape[0]
        prefix = torch.cat((images.mean(dim=2), torch.zeros(batch, 1, 8)), dim=1)
        native_valid = torch.cat((img_masks, lang_masks), dim=1)
        visual_sensor_mask = torch.cat(
            (img_masks, torch.zeros_like(lang_masks)),
            dim=1,
        )
        picf_native_context.bind_native_prefix(
            native_valid=native_valid,
            visual_sensor_mask=visual_sensor_mask,
            language_start=images.shape[1],
            language_valid=lang_masks,
        )
        graph = self.model.qwenvl_with_expert.picf_native_graph
        prepared, attention, _, _, runtime = graph.prepare_joint_inputs(
            inputs_embeds=[prefix, None],
            attention_mask=torch.ones(batch, prefix.shape[1], prefix.shape[1], dtype=torch.bool),
            position_ids=torch.zeros(3, batch, prefix.shape[1], dtype=torch.long),
            visual_pos_masks=visual_sensor_mask,
            context=picf_native_context,
        )
        hidden = prepared[0]
        weights = attention.float() / attention.sum(dim=-1, keepdim=True).clamp_min(1)
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
            if graph.requires_intermediate_relation(
                layer_index=layer_index,
                runtime=runtime,
            ):
                graph.record_intermediate_relation(
                    normalized_prefix=torch.nn.functional.layer_norm(hidden, (8,)),
                    runtime=runtime,
                    layer_index=layer_index,
                )
        graph.finalize_joint_outputs(
            outputs_embeds=[hidden, None],
            runtime=runtime,
        )
        return picf_native_context.root_output_tensors()

    def picf_native_prior_forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: list[torch.Tensor | None],
        visual_pos_masks: torch.Tensor,
        picf_native_context: LingBotPriorRolloutContext,
    ) -> tuple[()]:
        self.call_order.append("prior")
        self.prior_forward_grad_enabled.append(torch.is_grad_enabled())
        self.prior_contexts.append(picf_native_context)
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
            if graph.layerwise_recurrence:
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


class _NonzeroAlignmentPolicy(_FakeOfficialTrainingPolicy):
    def forward(self, **kwargs: object) -> tuple[object, ...]:
        outputs = list(super().forward(**kwargs))
        action_loss = outputs[1]
        assert isinstance(action_loss, torch.Tensor)
        outputs[2] = action_loss * 0.01
        outputs[0] = outputs[0] + outputs[2]
        return tuple(outputs)


class _MissingNativeRootOutputsPolicy(_FakeOfficialTrainingPolicy):
    def forward(self, **kwargs: object) -> tuple[object, ...]:
        return super().forward(**kwargs)[:11]


class _WrongNativeRootOutputsPolicy(_FakeOfficialTrainingPolicy):
    def forward(self, **kwargs: object) -> tuple[object, ...]:
        outputs = list(super().forward(**kwargs))
        native_outputs = outputs[-1]
        assert isinstance(native_outputs, tuple)
        outputs[-1] = tuple(value.clone() for value in native_outputs)
        return tuple(outputs)


class _DeclaredRootOutputCastPolicy(_FakeOfficialTrainingPolicy):
    class _MixedPrecision:
        output_dtype = torch.bfloat16

    class _FSDPState:
        _mp_policy = None

    def __init__(self, graph: LingBotNativeGraph) -> None:
        super().__init__(graph)
        self._fsdp_state = self._FSDPState()
        self._fsdp_state._mp_policy = self._MixedPrecision()

    def _get_fsdp_state(self):
        return self._fsdp_state

    def forward(self, **kwargs: object) -> tuple[object, ...]:
        outputs = list(super().forward(**kwargs))
        native_outputs = outputs[-1]
        assert isinstance(native_outputs, tuple)
        outputs[-1] = tuple(
            value if value.dtype == torch.bfloat16 else value.to(torch.bfloat16)
            for value in native_outputs
        )
        return tuple(outputs)


class _DetachedRootOutputCastPolicy(_DeclaredRootOutputCastPolicy):
    def forward(self, **kwargs: object) -> tuple[object, ...]:
        outputs = list(super().forward(**kwargs))
        native_outputs = outputs[-1]
        assert isinstance(native_outputs, tuple)
        outputs[-1] = tuple(value.detach() for value in native_outputs)
        return tuple(outputs)


class _StatefulStochasticTrainingPolicy(_FakeOfficialTrainingPolicy):
    def __init__(self, graph: LingBotNativeGraph) -> None:
        super().__init__(graph)
        self.register_buffer("tokens_per_expert", torch.zeros(2))
        self.register_buffer("routing_bias", torch.zeros(2))

    def forward(self, **kwargs: object) -> tuple[object, ...]:
        images = kwargs["images"]
        if not isinstance(images, torch.Tensor):
            raise TypeError("test images must be tensors")
        dropped = torch.nn.functional.dropout(images, p=0.25, training=True)
        kwargs["images"] = dropped
        with torch.no_grad():
            selected = (dropped.flatten(1).mean(dim=1) > 0).to(torch.long)
            self.tokens_per_expert.add_(
                torch.nn.functional.one_hot(selected, num_classes=2).sum(dim=0)
            )
        return super().forward(**kwargs)

    def update_routing_bias(self) -> None:
        deviation = (self.tokens_per_expert - self.tokens_per_expert.mean()).sign()
        self.routing_bias.add_(-0.001 * deviation)


class _AddressedReconstructionPolicy(_FakeOfficialTrainingPolicy):
    """Expose an addressed layerwise output through the shared test host."""

    def __init__(
        self,
        graph: LingBotNativeGraph,
        *,
        episode_address_state: EpisodeAddressState,
    ) -> None:
        super().__init__(graph)
        self.episode_address_state = episode_address_state

    def forward(self, **kwargs: object) -> tuple[object, ...]:
        outputs = super().forward(**kwargs)
        context = kwargs["picf_native_context"]
        if not isinstance(context, LingBotNativeContext):
            raise TypeError("test policy requires LingBotNativeContext")
        posterior = context.posterior_memory
        if not isinstance(posterior, NativeLayerwisePosteriorState):
            raise TypeError("test policy requires a layerwise posterior")
        context.posterior_memory = AddressedLayerwisePosteriorState(
            layer_rows=posterior.layer_rows,
            episode_address_state=self.episode_address_state,
            architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        )
        return outputs


def _components(
    *,
    architecture_identity: str = LEGACY_TASK_MATCH_ARCHITECTURE,
    relation_supervision_layers: tuple[int, ...] = (),
) -> tuple[
    _FakeOfficialTrainingPolicy,
    NativeTrainingLaneCoordinator,
]:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            architecture_identity=architecture_identity,
            relation_supervision_layers=relation_supervision_layers,
        )
    )
    policy = _FakeOfficialTrainingPolicy(graph).train()
    bank = NativeTrainingLaneBank(
        NativeLaneConfig(
            model_digest="training-runtime-test",
            schema_digest="native-rows-v1",
            capacity=2,
            host_width=8,
            maximum_optimizer_lag=8,
        )
    )
    return policy, NativeTrainingLaneCoordinator(bank)


def _v3_predictive_policy() -> _FakeOfficialTrainingPolicy:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
            predictive_target_widths=(("dino_video", 4),),
        )
    )
    return _FakeOfficialTrainingPolicy(graph).train()


def _v3_filter_spec(
    *,
    valid: bool,
    posterior_valid: bool | None = None,
) -> NativeV3FilterPredictionSpec:
    resolved_posterior_valid = valid if posterior_valid is None else posterior_valid
    common = {
        "route_ids": torch.zeros(1, 1, dtype=torch.long),
        "horizons": torch.zeros(1, 1, dtype=torch.long),
        "addresses": torch.empty(1, 1, 0),
    }
    return NativeV3FilterPredictionSpec(
        prior_request=NativePredictionRequest(
            source=PredictionSource.PRIOR,
            evidence=PredictionEvidence.CURRENT_PRIOR,
            valid=torch.full((1, 1), valid, dtype=torch.bool),
            **common,
        ),
        posterior_request=NativePredictionRequest(
            source=PredictionSource.POSTERIOR,
            evidence=PredictionEvidence.CURRENT_POSTERIOR,
            valid=torch.full((1, 1), resolved_posterior_valid, dtype=torch.bool),
            **common,
        ),
        target_name="dino_video",
    )


def test_official_baseline_forward_bypasses_picf_and_keeps_action_gradients() -> None:
    policy = _FakeVanillaTrainingPolicy().train()
    model_inputs = _model_inputs(2)

    result = run_official_policy_training_forward(
        policy,
        model_inputs=model_inputs,
    )

    assert len(result.official_outputs) == 11
    assert result.official_total_loss is result.official_action_loss
    assert float(result.official_moe_regularizer.detach()) == 0.0
    assert policy.calls == [(frozenset(model_inputs), False, True)]
    result.official_total_loss.backward()
    assert policy.scale.grad is not None
    assert torch.isfinite(policy.scale.grad)
    assert bool(policy.scale.grad.ne(0))


def test_addressed_lane_preparation_preserves_reset_and_continuation_gauges() -> None:
    config = NativeLaneConfig(
        model_digest="addressed-training-runtime-test",
        schema_digest="addressed-native-rows-v1",
        capacity=2,
        host_width=8,
        maximum_optimizer_lag=8,
        num_layers=3,
        addressed_architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        episode_address_codebook_sha256="a" * 64,
    )
    coordinator = NativeTrainingLaneCoordinator(NativeTrainingLaneBank(config))

    bootstrap = coordinator.begin(optimizer_step=0, source_weight_version=1)
    initial = bootstrap.prepare(
        _routing(1, optimizer_step=0, frame_index=0, episode_key="episode-b")
    )
    assert isinstance(initial.previous_state, AddressedLayerwisePosteriorState)
    committed = AddressedLayerwisePosteriorState(
        layer_rows=torch.ones_like(initial.previous_state.layer_rows),
        episode_address_state=initial.previous_state.episode_address_state,
        architecture_identity=initial.previous_state.architecture_identity,
    )
    bootstrap.stage(initial, committed, row_bindings_by_batch=((),))
    assert bootstrap.finish(lambda: 1)

    mixed_routing = NativeCALVINRouting(
        lane_ids=(0, 1),
        episode_keys=("episode-a", "episode-b"),
        frame_indices=(0, 1),
        reset=(True, False),
        sample_keys=("episode-a/0", "episode-b/1"),
        optimizer_step=1,
    )
    attempt = coordinator.begin(optimizer_step=1, source_weight_version=1)
    prepared = attempt.prepare(mixed_routing)
    assert isinstance(prepared.previous_state, AddressedLayerwisePosteriorState)
    assert isinstance(prepared.wrong_time_state, AddressedLayerwisePosteriorState)
    assert prepared.previous_state_valid.tolist() == [False, True]
    assert prepared.wrong_time_state_valid.tolist() == [False, False]
    assert torch.count_nonzero(prepared.previous_state.layer_rows[0]) == 0
    assert torch.equal(
        prepared.previous_state.layer_rows[1],
        committed.layer_rows[0],
    )
    assert prepared.previous_state.episode_address_state.index_select(
        torch.tensor([1])
    ).same_assignment(committed.episode_address_state)
    assert prepared.wrong_time_state.episode_address_state.same_assignment(
        prepared.previous_state.episode_address_state
    )

    cold = native_cold_state_for_episode_keys(
        config,
        episode_keys=mixed_routing.episode_keys,
    )
    assert isinstance(cold, AddressedLayerwisePosteriorState)
    assert torch.count_nonzero(cold.layer_rows) == 0
    assert cold.episode_address_state.same_assignment(
        prepared.previous_state.episode_address_state
    )
    attempt.discard(prepared)
    attempt.abort()


def test_task_addressed_two_pass_forward_preserves_one_routing_receipt() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            task_query_count=2,
            architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        )
    )
    assert graph.episode_address_codebook is not None
    policy = _FakeOfficialTrainingPolicy(graph).train()
    config = NativeLaneConfig(
        model_digest="task-addressed-two-pass-test",
        schema_digest="task-addressed-two-pass-stream",
        capacity=2,
        host_width=8,
        maximum_optimizer_lag=8,
        num_layers=3,
        addressed_architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        episode_address_codebook_sha256=address_codebook_sha256(graph.episode_address_codebook),
    )
    coordinator = NativeTrainingLaneCoordinator(NativeTrainingLaneBank(config))
    attempt = coordinator.begin(optimizer_step=0, source_weight_version=1)
    prepared = attempt.prepare(
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a")
    )
    assert isinstance(prepared.previous_state, AddressedLayerwisePosteriorState)

    result = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_memory=prepared.previous_state,
        previous_memory_valid=prepared.previous_state_valid,
    )

    assert isinstance(result.prior_trace, AddressedLayerwisePriorTrace)
    posterior = native_persistent_output(result.policy_forward.context)
    assert isinstance(posterior, AddressedLayerwisePosteriorState)
    assert result.prior_trace.episode_address_state.same_assignment(
        prepared.previous_state.episode_address_state
    )
    assert posterior.episode_address_state.same_assignment(
        prepared.previous_state.episode_address_state
    )
    assert policy.call_order == ["prior", "policy"]
    result.policy_forward.official_total_loss.backward()
    assert torch.isfinite(result.policy_forward.official_total_loss)
    attempt.stage(prepared, posterior, row_bindings_by_batch=((),))
    assert attempt.finish(lambda: 1)


def test_official_baseline_diagnostic_uses_the_same_targetless_root_without_gradients() -> None:
    policy = _FakeVanillaTrainingPolicy().train()
    model_inputs = _model_inputs(1)

    result = run_official_policy_diagnostic_forward(
        policy,
        model_inputs=model_inputs,
    )

    assert not result.official_total_loss.requires_grad
    assert not result.official_action_loss.requires_grad
    assert policy.calls == [(frozenset(model_inputs), False, False)]


def test_official_baseline_rejects_a_picf_augmented_output() -> None:
    policy = _FakeVanillaTrainingPolicy(append_native_output=True).train()

    with pytest.raises(RuntimeError, match="exactly 11 fields"):
        run_official_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
        )


def test_task_independent_graph_crosses_official_action_training_boundary() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )

    result = run_native_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
    )

    relation = result.context.relation_output
    assert isinstance(relation, PhysicalRelationOutput)
    assert result.official_total_loss.requires_grad
    assert relation.ownership.requires_grad
    assert not hasattr(relation, "task_relevance")


def test_native_training_forward_passes_registered_action_attention_callback() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
    )

    def callback(**_surface: object) -> None:
        return None

    run_native_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
        action_attention_callback=callback,
    )

    assert policy.action_attention_callbacks == [callback]


def test_task_independent_representation_forward_requires_only_physical_fields() -> None:
    policy, _coordinator = _components(
        architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
        relation_supervision_layers=(0,),
    )

    context = run_native_policy_representation_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(
            controls=_controls(1, reset=True),
            supervise_intermediate_relations=True,
        ),
    )

    relation = context.relation_output
    assert isinstance(relation, PhysicalRelationOutput)
    assert tuple(context.intermediate_relation_outputs) == (0,)
    assert all(
        isinstance(value, PhysicalRelationOutput)
        for value in context.intermediate_relation_outputs.values()
    )
    (relation.ownership.mean() + relation.existence.mean()).backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in graph.parameters()
    )


def _forward_and_stage(
    policy: _FakeOfficialTrainingPolicy,
    attempt,
    routing: NativeCALVINRouting,
) -> torch.Tensor:
    prepared = attempt.prepare(routing)
    controls = _controls(1, reset=routing.reset[0])
    context = LingBotNativeContext(
        controls=controls,
        previous_state=prepared.previous_state,
        previous_state_valid=prepared.previous_state_valid,
    )
    result = run_native_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=context,
    )
    attempt.stage(
        prepared,
        result.context.posterior_state,
        row_bindings_by_batch=prepared.previous_row_bindings,
    )
    return result.official_action_loss


def test_official_forward_rejects_a_root_float_that_escaped_explicit_typing() -> None:
    policy, _coordinator = _components()
    inputs = _model_inputs(1)
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    context = LingBotNativeContext(controls=_controls(1, reset=True))

    with pytest.raises(TypeError, match="floating input 'images'.*compute dtype"):
        run_native_policy_training_forward(
            policy,
            model_inputs=inputs,
            context=context,
        )


def test_official_forward_rejects_prediction_addresses_from_the_fp32_master_dtype() -> None:
    policy, _coordinator = _components()
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    context = LingBotNativeContext(
        controls=_controls(1, reset=True),
        prediction_request=request,
    )

    with pytest.raises(TypeError, match="prediction addresses.*compute tensor contract"):
        run_native_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=context,
        )


def test_official_forward_and_accumulated_lanes_publish_only_after_optimizer_success() -> None:
    policy, coordinator = _components()
    first = _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a")
    second = _routing(1, optimizer_step=0, frame_index=0, episode_key="episode-b")

    overflow = coordinator.begin(optimizer_step=0, source_weight_version=3)
    _forward_and_stage(policy, overflow, first)
    _forward_and_stage(policy, overflow, second)
    assert not overflow.finish(lambda: None)
    assert len(coordinator.bank) == 0

    attempt = coordinator.begin(optimizer_step=0, source_weight_version=3)
    losses = (
        _forward_and_stage(policy, attempt, first),
        _forward_and_stage(policy, attempt, second),
    )
    sum(losses).backward()
    assert attempt.finish(lambda: 1)
    assert len(coordinator.bank) == 2

    continuation = coordinator.begin(optimizer_step=1, source_weight_version=3)
    prepared = continuation.prepare(
        _routing(0, optimizer_step=1, frame_index=1, episode_key="episode-a")
    )
    assert prepared.previous_state_valid.tolist() == [True]
    assert prepared.wrong_time_state_valid.tolist() == [False]
    assert prepared.next_state_ages == (1,)
    assert prepared.optimizer_lags == (1,)
    assert prepared.previous_state.rows.abs().sum() > 0
    continuation.discard(prepared)
    continuation.abort()
    assert len(coordinator.bank) == 2


def test_stateless_optimizer_step_updates_weights_without_reading_or_writing_bank() -> None:
    policy, coordinator = _components()
    causal = coordinator.begin(optimizer_step=0, source_weight_version=3)
    causal_loss = _forward_and_stage(
        policy,
        causal,
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a"),
    )
    causal_loss.backward()
    assert causal.finish(lambda: 1)
    before = coordinator.bank.serialize()

    reset = coordinator.begin(optimizer_step=1, source_weight_version=4)
    prepared = reset.prepare(
        _routing(0, optimizer_step=1, frame_index=0, episode_key="episode-reset")
    )
    assert not prepared.previous_state_valid.any()
    assert not prepared.wrong_time_state_valid.any()
    reset.discard(prepared)
    assert reset.finish_stateless(lambda: 2)
    assert coordinator.bank.serialize() == before

    overflow = coordinator.begin(optimizer_step=2, source_weight_version=5)
    prepared = overflow.prepare(
        _routing(0, optimizer_step=2, frame_index=0, episode_key="episode-reset-2")
    )
    overflow.discard(prepared)
    assert not overflow.finish_stateless(lambda: None)
    assert coordinator.bank.serialize() == before


def test_stateless_optimizer_step_rejects_staged_state_and_poisoned_optimizer() -> None:
    policy, coordinator = _components()
    staged = coordinator.begin(optimizer_step=0, source_weight_version=1)
    _forward_and_stage(
        policy,
        staged,
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a"),
    )
    with pytest.raises(NativeLaneError, match="cannot publish recurrent lanes"):
        staged.finish_stateless(lambda: 1)
    staged.abort()

    failed = coordinator.begin(optimizer_step=0, source_weight_version=1)
    prepared = failed.prepare(
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-reset")
    )
    failed.discard(prepared)

    def fail_after_unknown_optimizer_state() -> int:
        raise RuntimeError("synthetic stateless optimizer failure")

    with pytest.raises(RuntimeError, match="synthetic stateless optimizer failure"):
        failed.finish_stateless(fail_after_unknown_optimizer_state)
    assert coordinator.poisoned
    assert len(coordinator.bank) == 0


def test_prepared_lane_exposes_checkpointed_wrong_time_only_after_two_commits() -> None:
    policy, coordinator = _components()
    first = coordinator.begin(optimizer_step=0, source_weight_version=3)
    first_loss = _forward_and_stage(
        policy,
        first,
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a"),
    )
    first_loss.backward()
    assert first.finish(lambda: 1)

    second = coordinator.begin(optimizer_step=1, source_weight_version=3)
    second_prepared = second.prepare(
        _routing(0, optimizer_step=1, frame_index=1, episode_key="episode-a")
    )
    assert not second_prepared.wrong_time_state_valid.any()
    first_state = second_prepared.previous_state.rows.detach().clone()
    second_result = run_native_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(
            controls=_controls(1, reset=False),
            previous_state=second_prepared.previous_state,
            previous_state_valid=second_prepared.previous_state_valid,
        ),
    )
    second.stage(
        second_prepared,
        second_result.context.posterior_state,
        row_bindings_by_batch=second_prepared.previous_row_bindings,
    )
    second_result.official_total_loss.backward()
    assert second.finish(lambda: 2)

    third = coordinator.begin(optimizer_step=2, source_weight_version=3)
    third_prepared = third.prepare(
        _routing(0, optimizer_step=2, frame_index=2, episode_key="episode-a")
    )
    assert third_prepared.wrong_time_state_valid.tolist() == [True]
    torch.testing.assert_close(third_prepared.wrong_time_state.rows, first_state)
    third.discard(third_prepared)
    third.abort()


def test_official_forward_preserves_targetless_moe_regularization() -> None:
    policy, _coordinator = _components()
    result = run_native_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(
            controls=_controls(1, reset=True),
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
        ),
    )

    assert result.official_moe_regularizer.requires_grad
    assert result.official_moe_regularizer.detach().gt(0)
    torch.testing.assert_close(
        result.official_total_loss,
        result.official_action_loss + result.official_moe_regularizer,
    )
    assert all(result.official_outputs[index].detach().item() == 0.0 for index in (2, 3, 4))


def test_relation_forward_keeps_ownership_attached_with_frozen_policy_loss() -> None:
    policy, _coordinator = _components()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    selected = {
        id(graph.relation_readout.projection.weight),
        id(graph.relation_readout.no_object),
        id(graph.relation_readout.temperature_parameter),
    }
    for parameter in policy.parameters():
        parameter.requires_grad_(id(parameter) in selected)

    result = run_native_policy_relation_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(
            controls=_controls(1, reset=True),
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
        ),
    )

    assert not result.official_total_loss.requires_grad
    assert not result.official_action_loss.requires_grad
    relation = result.context.relation_output
    assert relation is not None and relation.ownership.requires_grad
    relation.ownership.square().mean().backward()
    gradient = graph.relation_readout.projection.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0

    with pytest.raises(RuntimeError, match="total loss is detached"):
        run_native_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=LingBotNativeContext(
                controls=_controls(1, reset=True),
                previous_state=None,
                previous_state_valid=torch.zeros(1, dtype=torch.bool),
            ),
        )


def test_relation_forward_rejects_detached_ownership() -> None:
    policy, _coordinator = _components()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(RuntimeError, match="relation output 'ownership' is detached"):
        run_native_policy_relation_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=LingBotNativeContext(
                controls=_controls(1, reset=True),
                previous_state=None,
                previous_state_valid=torch.zeros(1, dtype=torch.bool),
            ),
        )


def test_targetless_contract_accepts_bfloat16_reassociation_within_one_ulp() -> None:
    action = torch.tensor(0.1015625, dtype=torch.bfloat16, requires_grad=True)
    sequence = torch.tensor(0.0012054443359375, dtype=torch.bfloat16, requires_grad=True)
    router = torch.tensor(2.9802322387695312e-05, dtype=torch.bfloat16, requires_grad=True)
    zero = torch.zeros((), dtype=torch.bfloat16)
    total = action + (sequence + router)
    outputs = (total, action, zero, zero, zero, sequence, {"router_z_loss": router})

    regularizer, checks = _targetless_official_loss_contract(
        outputs,
        total_loss=total,
        action_loss=action,
    )
    _raise_first_failed_tensor_check(checks)
    regularizer.backward()

    torch.testing.assert_close(action.grad, torch.zeros_like(action))
    torch.testing.assert_close(sequence.grad, torch.ones_like(sequence))
    torch.testing.assert_close(router.grad, torch.ones_like(router))


def test_targetless_contract_rejects_unexplained_loss_beyond_one_ulp() -> None:
    action = torch.tensor(0.1015625, dtype=torch.bfloat16)
    sequence = torch.tensor(0.0012054443359375, dtype=torch.bfloat16)
    router = torch.tensor(2.9802322387695312e-05, dtype=torch.bfloat16)
    zero = torch.zeros((), dtype=torch.bfloat16)
    expected_total = action + (sequence + router)
    infinity = torch.full_like(expected_total, math.inf)
    unexplained_total = torch.nextafter(torch.nextafter(expected_total, infinity), infinity)
    outputs = (
        unexplained_total,
        action,
        zero,
        zero,
        zero,
        sequence,
        {"router_z_loss": router},
    )

    _regularizer, checks = _targetless_official_loss_contract(
        outputs,
        total_loss=unexplained_total,
        action_loss=action,
    )
    with pytest.raises(RuntimeError, match="more than one output ULP"):
        _raise_first_failed_tensor_check(checks)


def test_observation_root_is_prefix_equivalent_and_target_invariant() -> None:
    policy, _coordinator = _components()
    inputs = _model_inputs(1)
    full_context = LingBotNativeContext(controls=_controls(1, reset=True))
    full = run_native_policy_training_forward(
        policy,
        model_inputs=inputs,
        context=full_context,
    )
    observation_context = _run_native_observation_training_forward(
        policy,
        model_inputs=inputs,
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
    )
    for actual, expected in zip(
        observation_context.root_output_tensors(),
        full.context.root_output_tensors(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    target_changed = dict(inputs)
    for name in ("actions", "noise", "time", "joint_mask", "action_is_pad"):
        target_changed[name] = torch.zeros_like(inputs[name])
    target_invariant_context = _run_native_observation_training_forward(
        policy,
        model_inputs=target_changed,
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
    )
    for actual, expected in zip(
        target_invariant_context.root_output_tensors(),
        observation_context.root_output_tensors(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    assert policy.forward_grad_enabled == [True]
    assert policy.observation_forward_grad_enabled == [True, True]


def test_observation_diagnostic_uses_exact_root_without_gradients() -> None:
    policy, _coordinator = _components()
    context = run_native_policy_observation_diagnostic_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
    )

    assert policy.observation_forward_grad_enabled == [False]
    state = context.posterior_state
    relation = context.relation_output
    assert state is not None
    assert relation is not None
    assert not state.rows.requires_grad
    assert not relation.ownership.requires_grad


def test_representation_forward_uses_attached_observation_root_without_action() -> None:
    policy, _coordinator = _components()
    context = run_native_policy_representation_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
    )

    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled == [True]
    relation = context.relation_output
    state = context.posterior_state
    assert relation is not None
    assert state is not None and state.rows.requires_grad
    for value in (
        relation.ownership,
        relation.task_relevance,
        relation.dense_task_grounding,
        relation.existence,
    ):
        assert value.requires_grad
    (
        relation.ownership.mean()
        + relation.task_relevance.mean()
        + relation.dense_task_grounding.mean()
        + relation.existence.mean()
    ).backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in graph.parameters()
    )


def test_observation_root_matches_full_prefix_gradients() -> None:
    policy, _coordinator = _components()
    inputs = _model_inputs(1)

    def native_loss(context: LingBotNativeContext) -> torch.Tensor:
        state = context.posterior_state
        relation = context.relation_output
        assert state is not None
        assert relation is not None
        return (
            state.rows.square().mean()
            + relation.support_logits.square().mean()
            + relation.ownership.square().mean()
            + relation.task_relevance_logits.square().mean()
            + relation.existence_logits.square().mean()
        )

    full = run_native_policy_training_forward(
        policy,
        model_inputs=inputs,
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
    )
    native_loss(full.context).backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in graph.named_parameters()
        if parameter.grad is not None
    }
    assert full_gradients

    policy.zero_grad(set_to_none=True)
    observation = _run_native_observation_training_forward(
        policy,
        model_inputs=inputs,
        context=LingBotNativeContext(controls=_controls(1, reset=True)),
    )
    native_loss(observation).backward()
    observation_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in graph.named_parameters()
        if parameter.grad is not None
    }

    assert observation_gradients.keys() == full_gradients.keys()
    for name, expected in full_gradients.items():
        # Removing the masked suffix changes matmul reduction extents, not the function.
        torch.testing.assert_close(
            observation_gradients[name],
            expected,
            rtol=1e-4,
            atol=1e-5,
        )


def test_official_forward_rejects_any_targetless_alignment_loss() -> None:
    base, _coordinator = _components()
    graph = base.model.qwenvl_with_expert.picf_native_graph
    policy = _NonzeroAlignmentPolicy(graph).train()
    with pytest.raises(RuntimeError, match="depth loss must be zero"):
        run_native_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=LingBotNativeContext(
                controls=_controls(1, reset=True),
                previous_state=None,
                previous_state_valid=torch.zeros(1, dtype=torch.bool),
            ),
        )


@pytest.mark.parametrize(
    ("policy_type", "message"),
    (
        (
            _MissingNativeRootOutputsPolicy,
            "11 official fields and one native root-output tuple",
        ),
        (
            _WrongNativeRootOutputsPolicy,
            "native root outputs differ from the finalized context",
        ),
    ),
)
def test_official_forward_requires_exact_native_root_output_identity(
    policy_type: type[_FakeOfficialTrainingPolicy],
    message: str,
) -> None:
    base, _coordinator = _components()
    graph = base.model.qwenvl_with_expert.picf_native_graph
    policy = policy_type(graph).train()
    with pytest.raises(RuntimeError, match=message):
        run_native_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=LingBotNativeContext(
                controls=_controls(1, reset=True),
                previous_state=None,
                previous_state_valid=torch.zeros(1, dtype=torch.bool),
            ),
        )


def test_official_forward_accepts_the_declared_fsdp_root_output_cast() -> None:
    base, _coordinator = _components()
    graph = base.model.qwenvl_with_expert.picf_native_graph
    policy = _DeclaredRootOutputCastPolicy(graph).train()

    result = run_native_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        context=LingBotNativeContext(
            controls=_controls(1, reset=True),
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
        ),
    )

    assert result.context.posterior_state is not None


def test_official_forward_rejects_a_detached_fsdp_root_output_cast() -> None:
    base, _coordinator = _components()
    graph = base.model.qwenvl_with_expert.picf_native_graph
    policy = _DetachedRootOutputCastPolicy(graph).train()

    with pytest.raises(RuntimeError, match="native root outputs differ"):
        run_native_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=LingBotNativeContext(
                controls=_controls(1, reset=True),
                previous_state=None,
                previous_state_valid=torch.zeros(1, dtype=torch.bool),
            ),
        )


def test_lane_attempt_rejects_repeated_accumulation_lane_and_poisoned_optimizer_state() -> None:
    policy, coordinator = _components()
    routing = _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a")
    attempt = coordinator.begin(optimizer_step=0, source_weight_version=1)
    _forward_and_stage(policy, attempt, routing)
    with pytest.raises(NativeLaneError, match="each recurrent lane only once"):
        attempt.prepare(routing)

    def fail_after_unknown_optimizer_state() -> int:
        raise RuntimeError("synthetic optimizer failure")

    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        attempt.finish(fail_after_unknown_optimizer_state)
    assert coordinator.poisoned
    assert len(coordinator.bank) == 0
    with pytest.raises(NativeLaneError, match="poisoned"):
        coordinator.begin(optimizer_step=0, source_weight_version=1)


def test_lane_attempt_poisoned_when_state_publication_fails_after_optimizer_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, coordinator = _components()
    attempt = coordinator.begin(optimizer_step=0, source_weight_version=1)
    _forward_and_stage(
        policy,
        attempt,
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a"),
    )

    def fail_publication(*_args, **_kwargs) -> None:
        raise RuntimeError("synthetic posterior publication failure")

    monkeypatch.setattr(coordinator.bank, "commit_batch_after_optimizer", fail_publication)
    with pytest.raises(RuntimeError, match="synthetic posterior publication failure"):
        attempt.finish(lambda: 1)

    assert coordinator.poisoned
    assert len(coordinator.bank) == 0
    with pytest.raises(NativeLaneError, match="poisoned"):
        coordinator.begin(optimizer_step=1, source_weight_version=1)


def test_training_forward_rejects_eval_mode_and_non_official_output_shape() -> None:
    policy, coordinator = _components()
    attempt = coordinator.begin(optimizer_step=0, source_weight_version=1)
    prepared = attempt.prepare(
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a")
    )
    context = LingBotNativeContext(
        controls=_controls(1, reset=True),
        previous_state=prepared.previous_state,
        previous_state_valid=prepared.previous_state_valid,
    )
    policy.eval()
    with pytest.raises(ValueError, match="train mode"):
        run_native_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=context,
        )
    attempt.discard(prepared)
    attempt.abort()


def test_source_masked_branch_exposes_prediction_only_and_cannot_publish_rows() -> None:
    policy, coordinator = _components()
    attempt = coordinator.begin(optimizer_step=0, source_weight_version=1)
    prepared = attempt.prepare(
        _routing(0, optimizer_step=0, frame_index=0, episode_key="episode-a")
    )
    inputs = _model_inputs(1)
    plan = sample_qwen_packed_patch_mask(
        images=inputs["images"],
        image_valid=inputs["img_masks"],
        image_grid_thw=inputs["image_grid_thw"],
        spatial_merge_size=2,
        probability=1.0,
        seed=7,
        eligible_view_indices=(0,),
    )
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.CURRENT_RANDOM_GRID,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    source_prediction = run_native_source_masked_training_forward(
        policy,
        model_inputs=inputs,
        controls=_controls(1, reset=True),
        previous_state=prepared.previous_state,
        previous_state_valid=prepared.previous_state_valid,
        prediction_request=request,
        source_mask=plan,
    )
    assert source_prediction.prediction_hidden.shape == (1, 2, 1, 8)
    assert source_prediction.source_mask_digest == plan.digest
    assert not hasattr(source_prediction, "posterior_state")
    assert not hasattr(source_prediction, "context")
    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled == [True]
    assert len(coordinator.bank) == 0
    attempt.discard(prepared)
    attempt.abort()


def test_omitted_modality_branch_uses_shared_host_without_exposing_rows() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            predictive_target_widths=(("touch_latent", 4),),
            modality_specs=(NativeModalitySpec("touch", 3, 2),),
        )
    )
    policy = _FakeOfficialTrainingPolicy(graph).train()
    modalities = NativeModalityBatch(
        (
            NativeModalityStream(
                name="touch",
                tokens=torch.randn(1, 2, 3),
                valid=torch.ones(1, 2, dtype=torch.bool),
            ),
        )
    )
    omission = sample_native_modality_omission(modalities, seed=31)
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.OMITTED_MODALITY,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )

    prediction = run_native_omitted_modality_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prediction_request=request,
        modalities=modalities,
        omission=omission,
    )

    assert policy.last_modalities is not None
    assert policy.last_modalities.streams[0].token_count == 0
    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled == [True]
    assert prediction.omitted_name == "touch"
    assert prediction.omission_digest == omission.digest
    assert prediction.prediction_outputs["touch_latent"].shape == (1, 2, 1, 4)
    assert not hasattr(prediction, "posterior_state")
    assert not hasattr(prediction, "context")
    prediction.prediction_outputs["touch_latent"].square().mean().backward()
    omitted_gradient = graph.modality_projections["touch"].weight.grad
    assert omitted_gradient is None or torch.count_nonzero(omitted_gradient) == 0
    assert graph.object_queries.grad is not None


def test_omitted_modality_branch_rejects_query_availability_mismatch() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            modality_specs=(NativeModalitySpec("touch", 3, 2),),
        )
    )
    policy = _FakeOfficialTrainingPolicy(graph).train()
    modalities = NativeModalityBatch(
        (
            NativeModalityStream(
                name="touch",
                tokens=torch.randn(1, 2, 3),
                valid=torch.ones(1, 2, dtype=torch.bool),
            ),
        )
    )
    omission = sample_native_modality_omission(modalities, seed=31)
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.OMITTED_MODALITY,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.zeros(1, 1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="query validity"):
        run_native_omitted_modality_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            controls=_controls(1, reset=True),
            previous_state=None,
            previous_state_valid=torch.zeros(1, dtype=torch.bool),
            prediction_request=request,
            modalities=modalities,
            omission=omission,
        )


def test_omitted_image_view_branch_uses_official_missing_view_without_publishing_rows() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            predictive_target_widths=(("dino_video", 4),),
        )
    )

    class _CapturePolicy(_FakeOfficialTrainingPolicy):
        observed_images: torch.Tensor | None = None
        observed_img_masks: torch.Tensor | None = None

        def picf_native_observation_forward(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
            self.observed_images = kwargs["images"]  # type: ignore[assignment]
            self.observed_img_masks = kwargs["img_masks"]  # type: ignore[assignment]
            return super().picf_native_observation_forward(**kwargs)

    policy = _CapturePolicy(graph).train()
    inputs = _model_inputs(1)
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=inputs["image_grid_thw"],
        image_valid=inputs["img_masks"],
        seed=43,
    )
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.OMITTED_MODALITY,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    prediction = run_native_omitted_image_view_training_forward(
        policy,
        model_inputs=inputs,
        controls=_controls(1, reset=True),
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
        prediction_request=request,
        omission=omission,
    )

    assert policy.observed_images is not None
    assert policy.observed_img_masks is not None
    assert (policy.observed_images[:, 0] == -1).all()
    assert torch.equal(policy.observed_images[:, 1], inputs["images"][:, 1])
    assert torch.equal(policy.observed_img_masks, torch.tensor([[False, True]]))
    assert prediction.omitted_name == "qwen_view_0"
    assert prediction.omission_digest == omission.digest
    assert prediction.prediction_outputs["dino_video"].shape == (1, 2, 1, 4)
    assert not hasattr(prediction, "posterior_state")
    assert not hasattr(prediction, "context")
    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled == [True]
    prediction.prediction_outputs["dino_video"].square().mean().backward()
    assert graph.object_queries.grad is not None


def test_local_bptt_carries_live_rows_but_exposes_only_primary_state() -> None:
    policy, coordinator = _components()
    result = run_native_local_bptt(
        policy,
        steps=(
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=True),
            ),
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=False),
            ),
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=False),
            ),
        ),
        previous_state=None,
        previous_state_valid=None,
    )
    assert len(result.auxiliary) == 2
    assert result.primary.context.posterior_state is not None
    assert all(not hasattr(item, "posterior_state") for item in result.auxiliary)
    assert all(not hasattr(item, "official_action_loss") for item in result.auxiliary)
    assert policy.forward_grad_enabled == [True]
    assert policy.observation_forward_grad_enabled == [True, True]
    loss = result.primary.official_action_loss + sum(
        item.relation_output.support_logits.square().mean() for item in result.auxiliary
    )
    loss.backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert graph.object_queries.grad is not None
    assert graph.object_queries.grad.abs().sum() > 0
    assert len(coordinator.bank) == 0


def test_representation_window_uses_only_observation_roots_and_live_state() -> None:
    policy, coordinator = _components()
    result = run_native_representation_window(
        policy,
        steps=tuple(
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=index == 0),
            )
            for index in range(3)
        ),
        previous_state=None,
        previous_state_valid=None,
    )

    assert len(result.contexts) == 3
    assert policy.forward_grad_enabled == []
    assert policy.observation_forward_grad_enabled == [True, True, True]
    assert all(
        context.posterior_state is not None and context.posterior_state.rows.requires_grad
        for context in result.contexts
    )
    result.contexts[-1].posterior_state.rows.square().mean().backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert graph.object_queries.grad is not None
    assert bool(torch.count_nonzero(graph.object_queries.grad))
    assert len(coordinator.bank) == 0

    with pytest.raises(ValueError, match="share one batch size"):
        run_native_representation_window(
            policy,
            steps=(
                NativeLocalBPTTStep(
                    model_inputs=_model_inputs(1),
                    controls=_controls(1, reset=True),
                ),
                NativeLocalBPTTStep(
                    model_inputs=_model_inputs(2),
                    controls=_controls(2, reset=False),
                ),
            ),
            previous_state=None,
            previous_state_valid=None,
        )


def test_relation_local_bptt_accepts_frozen_state_but_requires_live_ownership() -> None:
    policy, coordinator = _components()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    selected = {
        id(graph.relation_readout.projection.weight),
        id(graph.relation_readout.no_object),
        id(graph.relation_readout.temperature_parameter),
    }
    for parameter in policy.parameters():
        parameter.requires_grad_(id(parameter) in selected)

    result = run_native_relation_local_bptt(
        policy,
        steps=(
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=True),
            ),
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=False),
            ),
        ),
        previous_state=None,
        previous_state_valid=None,
    )

    state = result.primary.context.posterior_state
    assert state is not None and not state.rows.requires_grad
    relations = (
        result.primary.context.relation_output,
        *(item.relation_output for item in result.auxiliary),
    )
    assert all(relation is not None and relation.ownership.requires_grad for relation in relations)
    relation_loss = sum(
        relation.ownership.square().mean() for relation in relations if relation is not None
    )
    relation_loss.backward()
    gradient = graph.relation_readout.projection.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    assert not result.primary.official_total_loss.requires_grad
    assert len(coordinator.bank) == 0


def test_state_reconstruction_replays_shared_graph_without_gradients() -> None:
    policy, _coordinator = _components()
    result = reconstruct_native_state_no_grad(
        policy,
        steps=(
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=True),
            ),
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=False),
            ),
        ),
    )
    assert result.rows.shape == (1, 2, 8)
    assert not result.rows.requires_grad
    assert torch.isfinite(result.rows).all()

    single = run_native_state_reconstruction_step(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
    )
    assert single.rows.shape == result.rows.shape
    assert not single.rows.requires_grad


def _assert_exact_reconstruction_clone(
    actual: NativePosteriorState | NativeLayerwisePosteriorState,
    emitted: NativePosteriorState | NativeLayerwisePosteriorState,
) -> None:
    assert type(actual) is type(emitted)
    actual_tensor = persistent_state_tensor(actual)
    emitted_tensor = persistent_state_tensor(emitted)
    torch.testing.assert_close(actual_tensor, emitted_tensor, rtol=0, atol=0)
    assert actual_tensor.untyped_storage().data_ptr() != emitted_tensor.untyped_storage().data_ptr()
    assert not actual_tensor.requires_grad
    if isinstance(emitted, AddressedLayerwisePosteriorState):
        assert isinstance(actual, AddressedLayerwisePosteriorState)
        assert actual.architecture_identity == emitted.architecture_identity
        assert actual.address_receipt == emitted.address_receipt
        assert actual.episode_address_state.same_assignment(emitted.episode_address_state)
        assert (
            actual.episode_address_state.permutation.untyped_storage().data_ptr()
            != emitted.episode_address_state.permutation.untyped_storage().data_ptr()
        )


@pytest.mark.parametrize(
    ("architecture_identity", "expected_type"),
    (
        (LEGACY_TASK_MATCH_ARCHITECTURE, NativePosteriorState),
        (LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR, NativeLayerwisePosteriorState),
    ),
)
def test_state_reconstruction_preserves_unaddressed_state_schema(
    architecture_identity: str,
    expected_type: type[NativePosteriorState] | type[NativeLayerwisePosteriorState],
) -> None:
    policy, _coordinator = _components(architecture_identity=architecture_identity)

    single = run_native_state_reconstruction_step(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
    )
    emitted_single = native_persistent_output(policy.forward_contexts[-1])
    assert type(single) is expected_type
    _assert_exact_reconstruction_clone(single, emitted_single)

    replay = reconstruct_native_state_no_grad(
        policy,
        steps=(
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=True),
            ),
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=False),
            ),
        ),
    )
    emitted_replay = native_persistent_output(policy.forward_contexts[-1])
    assert type(replay) is expected_type
    _assert_exact_reconstruction_clone(replay, emitted_replay)


def test_state_reconstruction_preserves_addressed_state_and_receipt() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            architecture_identity=LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
        )
    )
    address_state = EpisodeAddressState(
        permutation=torch.tensor([[1, 0]], dtype=torch.long),
        codebook_sha256="1" * 64,
    )
    policy = _AddressedReconstructionPolicy(
        graph,
        episode_address_state=address_state,
    ).train()

    single = run_native_state_reconstruction_step(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_state=None,
        previous_state_valid=torch.zeros(1, dtype=torch.bool),
    )
    emitted_single = native_persistent_output(policy.forward_contexts[-1])
    _assert_exact_reconstruction_clone(single, emitted_single)

    policy.forward_contexts.clear()
    replay = reconstruct_native_state_no_grad(
        policy,
        steps=(
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=True),
            ),
            NativeLocalBPTTStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=False),
            ),
        ),
    )
    emitted_replay = native_persistent_output(policy.forward_contexts[-1])
    _assert_exact_reconstruction_clone(replay, emitted_replay)
    assert len(policy.forward_contexts) == 2
    continued = policy.forward_contexts[1].previous_memory
    assert isinstance(continued, AddressedLayerwisePosteriorState)
    assert continued.address_receipt == address_state.receipt


def test_v3_two_pass_uses_one_policy_and_never_forwards_raw_previous_memory() -> None:
    policy = _v3_predictive_policy()
    history = NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8))
    result = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=False),
        previous_memory=history,
        previous_memory_valid=torch.ones(1, dtype=torch.bool),
        filter_prediction=_v3_filter_spec(valid=True),
    )

    assert policy.call_order == ["prior", "policy"]
    assert policy.prior_contexts[0].previous_memory is history
    correction = policy.forward_contexts[0]
    assert correction is result.committable_context
    assert correction.prior_trace is result.prior_trace
    assert correction.previous_state is None
    assert correction.previous_memory is None
    assert correction.previous_state_valid.tolist() == [False]
    assert correction.previous_memory_valid.tolist() == [False]
    assert result.filter_predictions is not None
    assert result.filter_predictions.spec.prior_request.evidence is PredictionEvidence.CURRENT_PRIOR


def test_v3_two_pass_forwards_factual_route_and_action_attention_callback() -> None:
    policy = _v3_predictive_policy()
    route = torch.ones(1, dtype=torch.bool)

    def callback(**_surface: object) -> None:
        return None

    result = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_memory=None,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
        posterior_adoption_route=route,
        action_attention_callback=callback,
    )

    assert result.policy_forward.context.posterior_adoption_route is route
    assert policy.action_attention_callbacks[-1] is callback


def test_v3_sequence_limits_factual_action_collection_to_primary_frame() -> None:
    policy = _v3_predictive_policy()

    def callback(**_surface: object) -> None:
        return None

    result = run_native_v3_two_pass_sequence(
        policy,
        steps=tuple(
            NativeV3TwoPassStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=index == 0),
            )
            for index in range(2)
        ),
        previous_memory=None,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
        action_attention_callback=callback,
    )

    assert len(result.auxiliary) == 1
    assert policy.action_attention_callbacks == [callback]


def test_v3_two_pass_chains_transient_priors_for_long_control_receipts() -> None:
    policy = _v3_predictive_policy()
    history = NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8, requires_grad=True))
    first_controls = _controls(1, reset=False)
    final_controls = _controls(1, reset=False)

    result = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        controls=final_controls,
        previous_memory=history,
        previous_memory_valid=torch.ones(1, dtype=torch.bool),
        filter_prediction=_v3_filter_spec(valid=True),
        prior_control_chunks=(first_controls, final_controls),
    )

    assert policy.call_order == ["prior", "prior", "policy"]
    first_context, final_context = policy.prior_contexts
    assert first_context.previous_memory is history
    assert first_context.source_prior_trace is None
    assert final_context.previous_memory is None
    assert final_context.source_prior_trace is first_context.prior_trace
    assert final_context.source_prior_trace_valid.tolist() == [True]
    assert result.prior_trace is final_context.prior_trace
    assert not hasattr(result.prior_trace, "serialize")
    assert result.prior_trace.layer_rows.requires_grad
    assert (
        result.filter_predictions.spec.posterior_request.evidence
        is PredictionEvidence.CURRENT_POSTERIOR
    )

    with pytest.raises(ValueError, match="cannot read a previous posterior directly"):
        run_native_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            context=LingBotNativeContext(
                controls=_controls(1, reset=False),
                previous_memory=history,
                previous_memory_valid=torch.ones(1, dtype=torch.bool),
            ),
        )


def test_v3_prior_burnin_preserves_values_and_attaches_only_the_trailing_host_step() -> None:
    policy = _v3_predictive_policy()
    history = NativeLayerwisePosteriorState(torch.randn(1, 3, 2, 8, requires_grad=True))
    first_controls = _controls(1, reset=False)
    final_controls = _controls(1, reset=False)
    model_inputs = _model_inputs(1)
    spec = _v3_filter_spec(valid=True)

    torch.manual_seed(733)
    reference = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=model_inputs,
        controls=final_controls,
        previous_memory=history,
        previous_memory_valid=torch.ones(1, dtype=torch.bool),
        filter_prediction=spec,
        prior_control_chunks=(first_controls, final_controls),
    )
    torch.manual_seed(733)
    burnin = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=model_inputs,
        controls=final_controls,
        previous_memory=history,
        previous_memory_valid=torch.ones(1, dtype=torch.bool),
        filter_prediction=spec,
        prior_control_chunks=(first_controls, final_controls),
        prior_gradient_suffix_steps=1,
    )

    torch.testing.assert_close(
        burnin.prior_trace.layer_rows,
        reference.prior_trace.layer_rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        burnin.policy_forward.official_total_loss,
        reference.policy_forward.official_total_loss,
        rtol=0,
        atol=0,
    )
    burnin_prefix, burnin_suffix = policy.prior_contexts[-2:]
    assert burnin_prefix.prior_trace is not None
    assert burnin_suffix.prior_trace is burnin.prior_trace
    assert not burnin_prefix.prior_trace.layer_rows.requires_grad
    assert burnin_suffix.prior_trace.layer_rows.requires_grad

    burnin.policy_forward.official_total_loss.backward()
    assert history.layer_rows.grad is None


def test_v3_static_host_padding_is_value_identity_with_zero_dummy_gradient() -> None:
    policy = _v3_predictive_policy()
    reset = _controls(1, reset=True)
    model_inputs = _model_inputs(1)
    spec = _v3_filter_spec(valid=False, posterior_valid=True)

    torch.manual_seed(431)
    unpadded = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=model_inputs,
        controls=reset,
        previous_memory=None,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
        filter_prediction=spec,
    )
    torch.manual_seed(431)
    padded = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=model_inputs,
        controls=reset,
        previous_memory=None,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
        filter_prediction=spec,
        prior_host_steps=3,
    )

    assert policy.call_order == ["prior", "policy", "prior", "prior", "prior", "policy"]
    torch.testing.assert_close(
        padded.prior_trace.layer_rows,
        unpadded.prior_trace.layer_rows,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        padded.policy_forward.official_action_loss,
        unpadded.policy_forward.official_action_loss,
        rtol=0,
        atol=0,
    )
    assert padded.filter_predictions is not None
    assert unpadded.filter_predictions is not None
    torch.testing.assert_close(
        padded.filter_predictions.prior,
        unpadded.filter_predictions.prior,
        rtol=0,
        atol=0,
    )
    dummy_traces = tuple(context.prior_trace for context in policy.prior_contexts[1:3])
    assert all(trace is not None for trace in dummy_traces)
    for trace in dummy_traces:
        assert trace is not None
        trace.layer_rows.retain_grad()
    padded.policy_forward.official_total_loss.backward()
    for trace in dummy_traces:
        assert trace is not None and trace.layer_rows.grad is not None
        torch.testing.assert_close(
            trace.layer_rows.grad,
            torch.zeros_like(trace.layer_rows.grad),
            rtol=0,
            atol=0,
        )

    with pytest.raises(ValueError, match="shorter than the control chain"):
        run_native_v3_two_pass_policy_training_forward(
            policy,
            model_inputs=model_inputs,
            controls=reset,
            previous_memory=None,
            previous_memory_valid=torch.zeros(1, dtype=torch.bool),
            prior_control_chunks=(reset, reset),
            prior_host_steps=1,
        )


def test_v3_filter_spec_rejects_legacy_current_correction() -> None:
    posterior = _v3_filter_spec(valid=True).posterior_request
    with pytest.raises(ValueError, match="Pass A requires PRIOR/CURRENT_PRIOR"):
        NativeV3FilterPredictionSpec(
            prior_request=NativePredictionRequest(
                source=PredictionSource.PRIOR,
                evidence=PredictionEvidence.CURRENT_CORRECTION,
                route_ids=posterior.route_ids,
                horizons=posterior.horizons,
                addresses=posterior.addresses,
                valid=posterior.valid,
            ),
            posterior_request=posterior,
            target_name="dino_video",
        )


def test_v3_reset_requires_explicitly_invalid_layerwise_memory() -> None:
    policy = _v3_predictive_policy()
    placeholder = NativeLayerwisePosteriorState(torch.full((1, 3, 2, 8), 17.0, requires_grad=True))
    with pytest.raises(ValueError, match="reset requires explicitly invalid"):
        run_native_v3_two_pass_policy_training_forward(
            policy,
            model_inputs=_model_inputs(1),
            controls=_controls(1, reset=True),
            previous_memory=placeholder,
            previous_memory_valid=torch.ones(1, dtype=torch.bool),
        )
    assert policy.call_order == []

    result = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_memory=placeholder,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
        filter_prediction=_v3_filter_spec(valid=False, posterior_valid=True),
    )
    assert policy.call_order == ["prior", "policy"]
    assert policy.prior_contexts[0].previous_memory_valid.tolist() == [False]
    assert isinstance(result.prior_trace, NativeLayerwisePriorTrace)
    assert result.filter_predictions is not None
    assert result.filter_predictions.spec.prior_request.valid.tolist() == [[False]]
    assert result.filter_predictions.spec.posterior_request.valid.tolist() == [[True]]


def test_v3_probe_sequence_orders_every_prior_before_its_correction() -> None:
    policy = _v3_predictive_policy()
    result = run_native_v3_two_pass_sequence(
        policy,
        steps=tuple(
            NativeV3TwoPassStep(
                model_inputs=_model_inputs(1),
                controls=_controls(1, reset=index == 0),
                filter_prediction=_v3_filter_spec(valid=index > 0),
            )
            for index in range(3)
        ),
        previous_memory=None,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
    )

    assert policy.call_order == [
        "prior",
        "policy",
        "prior",
        "observation",
        "prior",
        "observation",
    ]
    assert len(result.prior_traces) == len(result.filter_predictions) == 3
    assert len(result.auxiliary) == 2
    assert result.committable_context is result.primary.context
    assert all(not hasattr(value, "context") for value in result.auxiliary)
    assert policy.prior_contexts[1].previous_memory is result.primary.context.posterior_memory
    assert all(
        context.previous_state is None and context.previous_memory is None
        for context in (*policy.forward_contexts, *policy.observation_contexts)
    )


def test_v3_attached_egress_backpropagates_without_next_full_image_forward() -> None:
    policy = _v3_predictive_policy()
    factual = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=_model_inputs(1),
        controls=_controls(1, reset=True),
        previous_memory=None,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
    )
    memory = factual.committable_context.posterior_memory
    assert isinstance(memory, NativeLayerwisePosteriorState)
    memory.layer_rows.retain_grad()

    egress = run_native_v3_attached_egress(
        policy,
        posterior_memory=memory,
        posterior_memory_valid=torch.ones(1, dtype=torch.bool),
        controls=_controls(1, reset=False),
        prediction_request=_v3_filter_spec(valid=True).prior_request,
        target_name="dino_video",
    )

    assert policy.call_order == ["prior", "policy", "prior"]
    assert policy.forward_grad_enabled == [True]
    assert policy.observation_forward_grad_enabled == []
    assert not hasattr(egress, "context")
    assert not hasattr(egress, "posterior_state")
    egress.prediction.square().mean().backward()
    assert memory.layer_rows.grad is not None
    assert memory.layer_rows.grad.abs().sum() > 0


def test_v3_omitted_static_view_runs_official_action_and_is_uncommittable() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            architecture_identity=UNIFIED_LAYERWISE_PREDICT_CORRECT,
            predictive_target_widths=(("dino_video", 4),),
        )
    )

    class _CapturePolicy(_FakeOfficialTrainingPolicy):
        seen_images: list[torch.Tensor]
        seen_image_valid: list[torch.Tensor]

        def __init__(self, native_graph: LingBotNativeGraph) -> None:
            super().__init__(native_graph)
            self.seen_images = []
            self.seen_image_valid = []

        def forward(self, **kwargs: object) -> tuple[object, ...]:
            self.seen_images.append(kwargs["images"])  # type: ignore[arg-type]
            self.seen_image_valid.append(kwargs["img_masks"])  # type: ignore[arg-type]
            return super().forward(**kwargs)

    policy = _CapturePolicy(graph).train()
    inputs = _model_inputs(1)
    factual = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=inputs,
        controls=_controls(1, reset=True),
        previous_memory=None,
        previous_memory_valid=torch.zeros(1, dtype=torch.bool),
    )
    omission = QwenWholeViewOmission(
        omitted_view_index=0,
        image_grid_thw=inputs["image_grid_thw"],
        image_valid=inputs["img_masks"],
        seed=51,
    )
    request = NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.OMITTED_MODALITY,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=omission.source_valid[:, None],
    )
    omitted = run_native_v3_omitted_static_view_policy_training_forward(
        policy,
        model_inputs=inputs,
        controls=_controls(1, reset=True),
        prior_trace=factual.prior_trace,
        prediction_request=request,
        omission=omission,
        posterior_adoption_route=torch.ones(1, dtype=torch.bool),
    )

    assert policy.call_order == ["prior", "policy", "policy"]
    assert policy.forward_contexts[1].prior_trace is factual.prior_trace
    assert policy.forward_contexts[1].previous_memory is None
    assert torch.equal(
        policy.forward_contexts[1].posterior_adoption_route,
        torch.ones(1, dtype=torch.bool),
    )
    assert all(
        factual_value is omitted_value
        for factual_value, omitted_value in zip(
            policy.action_randomness[0],
            policy.action_randomness[1],
            strict=True,
        )
    )
    assert torch.equal(policy.seen_image_valid[1], torch.tensor([[False, True]]))
    assert (policy.seen_images[1][:, 0] == -1).all()
    assert omitted.official_action_loss.requires_grad
    assert not hasattr(omitted, "context")
    assert not hasattr(omitted, "posterior_memory")
    assert not hasattr(omitted, "committable_context")
