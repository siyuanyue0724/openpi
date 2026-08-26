"""Complete native-query VidEoMT/LingBot training transaction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from picf_next.lingbot_native.calvin import (
    CollatedNativeCALVINBatch,
    with_native_modalities,
)
from picf_next.lingbot_native.future_latent_alignment import (
    FutureLatentTargetBatch,
    future_latent_objective_contribution,
)
from picf_next.lingbot_native.host import (
    NATIVE_VIDEOMT_QUERY_POSTERIOR,
    LingBotNativeContext,
    LingBotNativeGraph,
    native_context_from_prior_trace,
)
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalityStream,
    NativeObjectQuerySpatialSpec,
)
from picf_next.lingbot_native.physical_relations import NativeObjectQueryPosteriorOutput
from picf_next.lingbot_native.state import (
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativeVidEoMTPairedPosteriorState,
)
from picf_next.lingbot_native.training import (
    NativePolicyForwardResult,
    NativeV3TwoPassForwardResult,
    native_persistent_output,
    run_native_policy_diagnostic_forward,
    run_native_policy_observation_diagnostic_forward,
    run_native_v3_prior_chain,
    run_native_v3_two_pass_policy_training_forward,
)
from picf_next.lingbot_native.wsa_da3_loss import WSADA3TeacherTargets
from picf_next.lingbot_native.wsa_lingbot_training_runtime import (
    WSALingBotAttentionIntervention,
)
from picf_next.lingbot_wla_calvin import WLACalvinTargetBatch
from picf_next.videomt_exact.joint_training import (
    CompleteCalvinVidEoMTObjective,
    complete_picf_joint_total,
)
from picf_next.videomt_exact.observations import VidEoMTQueryObservation
from picf_next.videomt_exact.paired_training import (
    CompleteCausalVidEoMTTrainingTransaction,
    run_complete_causal_videomt_training_transaction,
)
from picf_next.videomt_exact.runtime import (
    ExactVidEoMTCausalSequenceOutput,
    ExactVidEoMTOutput,
    ExactVidEoMTRuntime,
)

WLA_HOST_EVIDENCE_ARMS = ("picf_full", "wla_lbot_masked")


def _validate_wla_host_evidence_arm(value: str) -> str:
    if value not in WLA_HOST_EVIDENCE_ARMS:
        raise ValueError(f"unknown WLA host-evidence arm: {value}")
    return value


def mask_wla_lbot_host_evidence(
    batch: CollatedNativeCALVINBatch,
) -> CollatedNativeCALVINBatch:
    """Keep the full graph while removing PICF evidence before the host."""

    if not isinstance(batch, CollatedNativeCALVINBatch) or batch.modalities is None:
        raise TypeError("WLA-LBOT evidence masking requires a collated modality batch")
    source_name = "videomt_queries"
    source_count = sum(stream.name == source_name for stream in batch.modalities.streams)
    if source_count != 1:
        raise ValueError("WLA-LBOT evidence masking requires one VidEoMT query stream")

    masked_streams: list[NativeModalityStream] = []
    for stream in batch.modalities.streams:
        source = stream.name == source_name
        valid = stream.valid if source else torch.zeros_like(stream.valid)
        canonical_ids = stream.canonical_token_ids
        if canonical_ids is not None and not source:
            canonical_ids = torch.full_like(canonical_ids, -1)
        masked_streams.append(
            NativeModalityStream(
                name=stream.name,
                tokens=torch.zeros_like(stream.tokens),
                valid=valid,
                metadata=(
                    None if stream.metadata is None else torch.zeros_like(stream.metadata)
                ),
                canonical_token_ids=canonical_ids,
            )
        )
    masked = NativeModalityBatch(
        streams=tuple(masked_streams),
        object_query_spatial_relations=batch.modalities.object_query_spatial_relations,
    )
    return replace(batch, modalities=masked)


@dataclass(frozen=True, slots=True)
class CompleteNativeVidEoMTLingBotStep:
    """One source-complete, host-current and atomically committable step."""

    source: CompleteCausalVidEoMTTrainingTransaction
    host_batch: CollatedNativeCALVINBatch
    prior_trace: NativeLayerwisePriorTrace
    policy: NativePolicyForwardResult
    next_state: NativeVidEoMTPairedPosteriorState
    total: torch.Tensor

    def __post_init__(self) -> None:
        if self.total.ndim != 0 or not self.total.is_floating_point():
            raise ValueError("joint VidEoMT/LingBot total must be one floating scalar")
        if not torch.isfinite(self.total):
            raise ValueError("joint VidEoMT/LingBot total is not finite")
        if self.total.device != self.policy.official_total_loss.device:
            raise ValueError("joint total and official LingBot loss must share one device")
        if not isinstance(self.prior_trace, NativeLayerwisePriorTrace):
            raise TypeError("joint VidEoMT/LingBot prior trace uses an invalid schema")
        if self.next_state.architecture_identity != NATIVE_VIDEOMT_QUERY_POSTERIOR:
            raise ValueError("joint state uses the wrong native-query architecture identity")
        if self.next_state.source_queries is not self.source.current_propagated_queries:
            raise ValueError("joint state did not commit the post-current source boundary")
        host_state = native_persistent_output(self.policy.context)
        if not isinstance(host_state, NativeLayerwisePosteriorState):
            raise TypeError("native-query LingBot output must be a layerwise posterior")
        if self.next_state.layer_rows is not host_state.layer_rows:
            raise ValueError("joint state did not commit the current LingBot posterior")


@dataclass(frozen=True, slots=True)
class ColdNativeVidEoMTLingBotEvaluation:
    """Target-free current RGB/action forward for matched held-out evaluation."""

    source_output: ExactVidEoMTOutput
    host_batch: CollatedNativeCALVINBatch
    prior_trace: NativeLayerwisePriorTrace
    policy: NativePolicyForwardResult

    def __post_init__(self) -> None:
        if self.source_output.class_logits.shape[:2] != (1, 1):
            raise ValueError("cold native-query evaluation must use one current source frame")
        relation = self.policy.context.relation_output
        if not isinstance(relation, NativeObjectQueryPosteriorOutput):
            raise TypeError("cold native-query evaluation lost the source-query relation")
        if relation.posterior_rows.shape[1] != self.source_output.class_logits.shape[2]:
            raise ValueError("cold native-query evaluation changed source query capacity")
        if self.policy.official_total_loss.requires_grad:
            raise ValueError("cold native-query evaluation retained an action graph")


@dataclass(frozen=True, slots=True)
class NativeVidEoMTHostDiagnostic:
    """One target-free host replay with an already materialized source query set."""

    host_batch: CollatedNativeCALVINBatch
    prior_trace: NativeLayerwisePriorTrace
    policy: NativePolicyForwardResult

    def __post_init__(self) -> None:
        relation = self.policy.context.relation_output
        if not isinstance(relation, NativeObjectQueryPosteriorOutput):
            raise TypeError("native-query host diagnostic lost the source-query relation")
        if self.policy.official_total_loss.requires_grad:
            raise ValueError("native-query host diagnostic retained an action graph")


@dataclass(frozen=True, slots=True)
class NativeVidEoMTHostObservationDiagnostic:
    """One exact observation correction used only to reconstruct causal state."""

    host_batch: CollatedNativeCALVINBatch
    prior_trace: NativeLayerwisePriorTrace
    context: LingBotNativeContext

    def __post_init__(self) -> None:
        relation = self.context.relation_output
        if not isinstance(relation, NativeObjectQueryPosteriorOutput):
            raise TypeError("native-query observation replay lost the source-query relation")
        posterior = native_persistent_output(self.context)
        if not isinstance(posterior, NativeLayerwisePosteriorState):
            raise TypeError("native-query observation replay lost layerwise posterior state")
        if posterior.layer_rows.requires_grad:
            raise ValueError("native-query observation replay retained a graph")


@dataclass(frozen=True, slots=True)
class CausalWarmNativeVidEoMTLingBotEvaluation:
    """Past-only source/host replay followed by one official current action."""

    source_sequence: ExactVidEoMTCausalSequenceOutput
    history: tuple[NativeVidEoMTHostObservationDiagnostic, ...]
    current: NativeVidEoMTHostDiagnostic
    next_state: NativeVidEoMTPairedPosteriorState

    def __post_init__(self) -> None:
        if not self.history:
            raise ValueError("causal-warm evaluation requires at least one real past frame")
        if len(self.source_sequence.per_frame) != len(self.history) + 1:
            raise ValueError("causal-warm source and host replay lengths differ")
        host_state = native_persistent_output(self.current.policy.context)
        if not isinstance(host_state, NativeLayerwisePosteriorState):
            raise TypeError("causal-warm current action omitted layerwise posterior state")
        if self.next_state.layer_rows is not host_state.layer_rows:
            raise ValueError("causal-warm evaluation committed another host boundary")
        if self.next_state.source_queries is not (
            self.source_sequence.propagated_queries_by_frame[-1]
        ):
            raise ValueError("causal-warm evaluation committed another source boundary")


def _native_videomt_host_prior(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    host_batch: CollatedNativeCALVINBatch,
    previous_state: NativeLayerwisePosteriorState | None,
    prior_host_steps: int,
) -> NativeLayerwisePriorTrace:
    if host_batch.routing.batch_size != 1:
        raise ValueError("native-query host diagnostic requires one sample per rank")
    if not isinstance(graph, LingBotNativeGraph) or not graph.native_videomt_query_posterior:
        raise ValueError("native-query host diagnostic requires the ADR-207 host graph")
    relation_spec = graph.config.object_query_spatial_specs
    if len(relation_spec) != 1:
        raise ValueError("ADR-207 host diagnostic requires one native source-query relation")
    if host_batch.modalities is None:
        raise ValueError("native-query host diagnostic requires typed modalities")
    source_streams = tuple(
        stream
        for stream in host_batch.modalities.streams
        if stream.name == relation_spec[0].query_modality
    )
    if len(source_streams) != 1:
        raise ValueError("native-query host diagnostic requires one source-query stream")
    previous_valid = torch.full(
        (1,),
        previous_state is not None,
        dtype=torch.bool,
        device=host_batch.controls.values.device,
    )
    prior_trace, _prediction = run_native_v3_prior_chain(
        policy,
        graph=graph,
        previous_memory=previous_state,
        previous_memory_valid=previous_valid,
        control_chunks=host_batch.effective_prior_control_chunks,
        filter_prediction=None,
        require_attached_memory=False,
        require_grad=False,
        host_step_count=prior_host_steps,
    )
    return prior_trace


@torch.no_grad()
def run_native_videomt_host_diagnostic(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    host_batch: CollatedNativeCALVINBatch,
    prior_host_steps: int,
    previous_state: NativeLayerwisePosteriorState | None = None,
    posterior_adoption_route: torch.Tensor | None = None,
    wsa_attention_intervention: WSALingBotAttentionIntervention | None = None,
) -> NativeVidEoMTHostDiagnostic:
    """Replay the shared LingBot host while holding source queries and inputs fixed."""

    prior_trace = _native_videomt_host_prior(
        policy,
        graph=graph,
        host_batch=host_batch,
        previous_state=previous_state,
        prior_host_steps=prior_host_steps,
    )
    forward_kwargs: dict[str, Any] = {}
    if wsa_attention_intervention is not None:
        forward_kwargs["wsa_attention_intervention"] = wsa_attention_intervention
    result = run_native_policy_diagnostic_forward(
        policy,
        model_inputs=host_batch.model_inputs,
        context=native_context_from_prior_trace(
            controls=host_batch.controls,
            prior_trace=prior_trace,
            modalities=host_batch.modalities,
            posterior_adoption_route=posterior_adoption_route,
        ),
        **forward_kwargs,
    )
    return NativeVidEoMTHostDiagnostic(
        host_batch=host_batch,
        prior_trace=prior_trace,
        policy=result,
    )


@torch.no_grad()
def run_native_videomt_host_observation_diagnostic(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    host_batch: CollatedNativeCALVINBatch,
    prior_host_steps: int,
    previous_state: NativeLayerwisePosteriorState | None,
) -> NativeVidEoMTHostObservationDiagnostic:
    """Advance prior and correction without executing the action suffix."""

    prior_trace = _native_videomt_host_prior(
        policy,
        graph=graph,
        host_batch=host_batch,
        previous_state=previous_state,
        prior_host_steps=prior_host_steps,
    )
    context = run_native_policy_observation_diagnostic_forward(
        policy,
        model_inputs=host_batch.model_inputs,
        context=native_context_from_prior_trace(
            controls=host_batch.controls,
            prior_trace=prior_trace,
            modalities=host_batch.modalities,
        ),
    )
    return NativeVidEoMTHostObservationDiagnostic(
        host_batch=host_batch,
        prior_trace=prior_trace,
        context=context,
    )


@torch.no_grad()
def run_causal_warm_native_videomt_lingbot_evaluation(
    policy: nn.Module,
    source_runtime: ExactVidEoMTRuntime,
    *,
    graph: LingBotNativeGraph,
    history_batches: Sequence[CollatedNativeCALVINBatch],
    current_batch: CollatedNativeCALVINBatch,
    normalized_rgb_sequence: torch.Tensor,
    relation_spec: NativeObjectQuerySpatialSpec,
    host_dtype: torch.dtype,
    prior_host_steps: Sequence[int],
    posterior_adoption_route: torch.Tensor | None = None,
    wla_host_evidence_arm: str = "picf_full",
) -> CausalWarmNativeVidEoMTLingBotEvaluation:
    """Reconstruct both recurrent states from strictly past observations."""

    history_values = tuple(history_batches)
    batches = (*history_values, current_batch)
    host_steps = tuple(prior_host_steps)
    if not history_values:
        raise ValueError("causal-warm evaluation requires a non-empty history")
    if len(host_steps) != len(batches) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in host_steps
    ):
        raise ValueError("causal-warm host schedules must align with every frame")
    if normalized_rgb_sequence.ndim != 4 or normalized_rgb_sequence.shape[1] != 3:
        raise ValueError("causal-warm RGB must have shape [time,3,H,W]")
    if normalized_rgb_sequence.shape[0] != len(batches):
        raise ValueError("causal-warm RGB and host frame counts differ")
    if host_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("causal-warm host dtype is unsupported")
    if any(batch.routing.batch_size != 1 for batch in batches):
        raise ValueError("causal-warm evaluation requires one sample per frame")
    evidence_arm = _validate_wla_host_evidence_arm(wla_host_evidence_arm)
    episode_keys = tuple(batch.routing.episode_keys[0] for batch in batches)
    frame_indices = tuple(batch.routing.frame_indices[0] for batch in batches)
    if len(set(episode_keys)) != 1:
        raise ValueError("causal-warm history crosses an episode reset")
    if any(right != left + 1 for left, right in zip(frame_indices, frame_indices[1:])):
        raise ValueError("causal-warm history is not consecutive")
    for name in ("lang_tokens", "lang_masks"):
        if any(
            not torch.equal(batches[0].model_inputs[name], batch.model_inputs[name])
            for batch in batches[1:]
        ):
            raise ValueError("causal-warm history changed its natural instruction")

    reset = torch.ones(1, dtype=torch.bool, device=normalized_rgb_sequence.device)
    try:
        source_runtime.bind_mixed_propagated_queries(None, reset=reset)
        source_sequence = source_runtime.forward_causal_sequence(
            normalized_rgb_sequence,
            resume=True,
        )
        history_results: list[NativeVidEoMTHostObservationDiagnostic] = []
        previous_host: NativeLayerwisePosteriorState | None = None
        for batch, source_output, steps in zip(
            history_values,
            source_sequence.per_frame[:-1],
            host_steps[:-1],
            strict=True,
        ):
            source_modalities = VidEoMTQueryObservation.from_exact_output(
                source_output
            ).as_native_pqm_batch(
                relation_spec=relation_spec,
                dtype=host_dtype,
            )
            host_batch = with_native_modalities(batch, source_modalities)
            host_previous = previous_host
            if evidence_arm == "wla_lbot_masked":
                host_batch = mask_wla_lbot_host_evidence(host_batch)
                if host_previous is not None:
                    host_previous = NativeLayerwisePosteriorState(
                        torch.zeros_like(host_previous.layer_rows)
                    )
            observation = run_native_videomt_host_observation_diagnostic(
                policy,
                graph=graph,
                host_batch=host_batch,
                prior_host_steps=steps,
                previous_state=host_previous,
            )
            posterior = native_persistent_output(observation.context)
            if not isinstance(posterior, NativeLayerwisePosteriorState):
                raise TypeError("causal-warm history omitted layerwise posterior state")
            previous_host = posterior.detached()
            history_results.append(observation)
        current_source = source_sequence.per_frame[-1]
        current_modalities = VidEoMTQueryObservation.from_exact_output(
            current_source
        ).as_native_pqm_batch(
            relation_spec=relation_spec,
            dtype=host_dtype,
        )
        current_batch = with_native_modalities(current_batch, current_modalities)
        current_previous = previous_host
        if evidence_arm == "wla_lbot_masked":
            current_batch = mask_wla_lbot_host_evidence(current_batch)
            if current_previous is not None:
                current_previous = NativeLayerwisePosteriorState(
                    torch.zeros_like(current_previous.layer_rows)
                )
        current = run_native_videomt_host_diagnostic(
            policy,
            graph=graph,
            host_batch=current_batch,
            prior_host_steps=host_steps[-1],
            previous_state=current_previous,
            posterior_adoption_route=posterior_adoption_route,
        )
        current_host = native_persistent_output(current.policy.context)
        if not isinstance(current_host, NativeLayerwisePosteriorState):
            raise TypeError("causal-warm current action omitted layerwise posterior state")
        next_state = NativeVidEoMTPairedPosteriorState(
            layer_rows=current_host.layer_rows,
            source_queries=source_sequence.propagated_queries_by_frame[-1],
            architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
        )
        return CausalWarmNativeVidEoMTLingBotEvaluation(
            source_sequence=source_sequence,
            history=tuple(history_results),
            current=current,
            next_state=next_state,
        )
    finally:
        source_runtime.reset_state()


@torch.no_grad()
def run_cold_native_videomt_lingbot_evaluation(
    policy: nn.Module,
    source_runtime: ExactVidEoMTRuntime,
    *,
    graph: LingBotNativeGraph,
    batch: CollatedNativeCALVINBatch,
    normalized_current_rgb: torch.Tensor,
    relation_spec: NativeObjectQuerySpatialSpec,
    host_dtype: torch.dtype,
    prior_host_steps: int,
    posterior_adoption_route: torch.Tensor | None = None,
    wsa_attention_intervention: WSALingBotAttentionIntervention | None = None,
    wla_host_evidence_arm: str = "picf_full",
) -> ColdNativeVidEoMTLingBotEvaluation:
    """Run a cold matched action forward before any target is materialized."""

    if batch.routing.batch_size != 1:
        raise ValueError("cold native-query evaluation requires one sample per rank")
    if not isinstance(graph, LingBotNativeGraph) or not graph.native_videomt_query_posterior:
        raise ValueError("cold native-query evaluation requires the ADR-207 host graph")
    if normalized_current_rgb.ndim != 4 or normalized_current_rgb.shape[:2] != (1, 3):
        raise ValueError("cold native-query evaluation RGB must be [1,3,H,W]")
    if host_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("cold native-query evaluation host dtype is unsupported")
    evidence_arm = _validate_wla_host_evidence_arm(wla_host_evidence_arm)
    source_reset = torch.ones(
        1,
        dtype=torch.bool,
        device=normalized_current_rgb.device,
    )
    try:
        source_runtime.bind_mixed_propagated_queries(None, reset=source_reset)
        source_sequence = source_runtime.forward_causal_sequence(
            normalized_current_rgb,
            resume=True,
        )
        source_output = source_sequence.per_frame[0]
    except BaseException:
        source_runtime.reset_state()
        raise
    source_runtime.reset_state()
    source_modalities = VidEoMTQueryObservation.from_exact_output(
        source_output
    ).as_native_pqm_batch(
        relation_spec=relation_spec,
        dtype=host_dtype,
    )
    host_batch = with_native_modalities(batch, source_modalities)
    if evidence_arm == "wla_lbot_masked":
        host_batch = mask_wla_lbot_host_evidence(host_batch)
    host = run_native_videomt_host_diagnostic(
        policy,
        graph=graph,
        host_batch=host_batch,
        prior_host_steps=prior_host_steps,
        posterior_adoption_route=posterior_adoption_route,
        wsa_attention_intervention=wsa_attention_intervention,
    )
    return ColdNativeVidEoMTLingBotEvaluation(
        source_output=source_output,
        host_batch=host_batch,
        prior_trace=host.prior_trace,
        policy=host.policy,
    )


def run_complete_native_videomt_lingbot_step(
    policy: nn.Module,
    source_runtime: ExactVidEoMTRuntime,
    source_objective: CompleteCalvinVidEoMTObjective,
    *,
    batch: CollatedNativeCALVINBatch,
    normalized_padded_rgb: torch.Tensor,
    clip_targets: Sequence[Mapping[str, torch.Tensor]],
    relation_spec: NativeObjectQuerySpatialSpec,
    previous_state: NativeVidEoMTPairedPosteriorState | None,
    previous_state_valid: torch.Tensor | None,
    host_dtype: torch.dtype,
    prior_host_steps: int | None = None,
    prior_gradient_suffix_steps: int | None = None,
    posterior_adoption_route: torch.Tensor | None = None,
    action_attention_callback: Callable[..., Any] | None = None,
    wsa_da3_teacher_targets: WSADA3TeacherTargets | None = None,
    wla_world_target: WLACalvinTargetBatch | None = None,
    future_latent_target: FutureLatentTargetBatch | None = None,
    future_latent_objective_scale: float = 1.0,
    wla_host_evidence_arm: str = "picf_full",
) -> CompleteNativeVidEoMTLingBotStep:
    """Train both complete models while preserving the causal decision boundary.

    Source frames one through four supervise VidEoMT but never enter the host
    batch. The state committed to the lane contains the source state immediately
    after frame zero and the LingBot posterior produced from that same frame.
    """

    if not isinstance(batch, CollatedNativeCALVINBatch):
        raise TypeError("native-query joint training requires a collated CALVIN batch")
    if not isinstance(source_objective, CompleteCalvinVidEoMTObjective):
        raise TypeError("native-query joint training requires the complete source objective")
    if host_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native-query host dtype must be a supported floating dtype")
    if previous_state is not None and (
        previous_state.architecture_identity != NATIVE_VIDEOMT_QUERY_POSTERIOR
    ):
        raise ValueError("previous paired state belongs to another architecture")
    evidence_arm = _validate_wla_host_evidence_arm(wla_host_evidence_arm)

    reset = torch.tensor(
        batch.routing.reset,
        dtype=torch.bool,
        device=batch.controls.values.device,
    )
    if previous_state_valid is None:
        previous_state_valid = (
            torch.zeros_like(reset) if previous_state is None else ~reset
        )
    if (
        previous_state_valid.shape != reset.shape
        or previous_state_valid.dtype != torch.bool
        or previous_state_valid.device != reset.device
    ):
        raise ValueError("paired posterior validity must be boolean [batch] on the host device")
    if (reset & previous_state_valid).any():
        raise ValueError("a reset sample cannot read the paired posterior")
    if ((~reset) & (~previous_state_valid)).any():
        raise ValueError("a continuing sample requires a valid paired posterior")
    source = run_complete_causal_videomt_training_transaction(
        source_runtime,
        source_objective,
        normalized_padded_rgb=normalized_padded_rgb,
        clip_targets=clip_targets,
        previous_queries=(None if previous_state is None else previous_state.source_queries),
        reset=reset.to(device=normalized_padded_rgb.device),
    )
    source_modalities = VidEoMTQueryObservation.from_exact_output(
        source.current_output
    ).as_native_pqm_batch(
        relation_spec=relation_spec,
        dtype=host_dtype,
    )
    host_batch = with_native_modalities(batch, source_modalities)
    previous_host = None if previous_state is None else previous_state.host_state
    if evidence_arm == "wla_lbot_masked":
        host_batch = mask_wla_lbot_host_evidence(host_batch)
        if previous_host is not None:
            previous_host = NativeLayerwisePosteriorState(
                torch.zeros_like(previous_host.layer_rows)
            )
    host = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=host_batch.model_inputs,
        controls=host_batch.controls,
        previous_memory=previous_host,
        previous_memory_valid=previous_state_valid,
        modalities=host_batch.modalities,
        prior_control_chunks=host_batch.prior_control_chunks,
        prior_host_steps=prior_host_steps,
        prior_gradient_suffix_steps=prior_gradient_suffix_steps,
        posterior_adoption_route=posterior_adoption_route,
        action_attention_callback=action_attention_callback,
        wsa_da3_teacher_targets=wsa_da3_teacher_targets,
        wla_world_target=wla_world_target,
        future_latent_target=future_latent_target,
    )
    if not isinstance(host, NativeV3TwoPassForwardResult):
        raise TypeError("native-query host returned an incompatible two-pass result")
    policy_result = host.policy_forward
    host_state = native_persistent_output(policy_result.context)
    if not isinstance(host_state, NativeLayerwisePosteriorState):
        raise TypeError("native-query host did not produce layerwise posterior memory")
    if host_state.capacity != source.current_propagated_queries.shape[1]:
        raise ValueError("source query and host posterior capacities differ")
    next_state = NativeVidEoMTPairedPosteriorState(
        layer_rows=host_state.layer_rows,
        source_queries=source.current_propagated_queries,
        architecture_identity=NATIVE_VIDEOMT_QUERY_POSTERIOR,
    )
    host_total = policy_result.official_total_loss
    if policy_result.future_latent_alignment is not None:
        host_total = host_total + future_latent_objective_contribution(
            policy_result.future_latent_alignment,
            scale=future_latent_objective_scale,
        )
    total = complete_picf_joint_total(
        host_total=host_total,
        source=source.source_objective,
    )
    return CompleteNativeVidEoMTLingBotStep(
        source=source,
        host_batch=host_batch,
        prior_trace=host.prior_trace,
        policy=policy_result,
        next_state=next_state,
        total=total,
    )
