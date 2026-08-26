"""Atomic deployment runtime around the official LingBot action interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from picf_next.lingbot_native.addresses import address_codebook_sha256
from picf_next.lingbot_native.host import (
    LingBotNativeGraph,
    LingBotNativePriorStepper,
    native_context_from_persistent_state,
    native_context_from_prior_trace,
)
from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.session import NativeObservationBatch, NativeSessionManager
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    AddressedLayerwisePriorTrace,
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativePosteriorState,
)

_RUNTIME_MODEL_FIELDS = frozenset(
    {
        "image_grid_thw",
        "images",
        "img_masks",
        "lang_masks",
        "lang_tokens",
        "state",
    }
)


@dataclass(frozen=True, slots=True)
class NativePolicyStep:
    actions: torch.Tensor
    prior_state: NativePosteriorState
    posterior_state: NativePosteriorState
    relations: RelationOutput | PhysicalRelationOutput


def audit_native_runtime_inputs(
    model_inputs: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    """Accept only the released LingBot deployment tensor boundary."""

    if set(model_inputs) != _RUNTIME_MODEL_FIELDS:
        missing = sorted(_RUNTIME_MODEL_FIELDS - set(model_inputs))
        unexpected = sorted(set(model_inputs) - _RUNTIME_MODEL_FIELDS)
        raise ValueError(
            f"native runtime input schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, value in model_inputs.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise TypeError(f"native runtime field {name} must be a non-scalar tensor")
        if value.device != device:
            raise ValueError(f"native runtime field {name} is on the wrong device")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"native runtime field {name} contains NaN or infinity")
        if name == "image_grid_thw" and batch_size == 1 and value.ndim == 2:
            continue
        if value.shape[0] != batch_size:
            raise ValueError(f"native runtime field {name} has the wrong batch axis")


class LingBotNativePolicyRuntime:
    """Own one fail-closed state transaction around each accepted observation."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        graph: LingBotNativeGraph,
        sessions: NativeSessionManager,
    ) -> None:
        if not isinstance(policy, nn.Module) or not isinstance(graph, LingBotNativeGraph):
            raise TypeError("native runtime requires a policy module and native graph")
        sample_actions = getattr(policy, "sample_actions", None)
        if not callable(sample_actions):
            raise TypeError("native runtime policy must expose sample_actions")
        if policy.training:
            raise ValueError("native deployment policy must be in evaluation mode")
        host = getattr(getattr(policy, "model", None), "qwenvl_with_expert", None)
        if getattr(host, "picf_native_graph", None) is not graph:
            raise ValueError("native runtime requires the exact graph installed in LingBot")
        config = sessions.config
        reference = graph._parameter_reference
        if (
            config.capacity != graph.config.capacity
            or config.host_width != graph.config.host_width
            or config.num_layers
            != (graph.config.num_layers if graph.layerwise_recurrence else None)
            or torch.device(config.device) != reference.device
            or config.dtype != reference.dtype
        ):
            raise ValueError("native session and installed graph contracts differ")
        if graph.task_query_object_value_read:
            codebook = graph.episode_address_codebook
            if not isinstance(codebook, torch.Tensor):
                raise RuntimeError("task-addressed graph omitted its fixed address codebook")
            if (
                config.addressed_architecture_identity != graph.config.architecture_identity
                or config.address_codebook_sha256 != address_codebook_sha256(codebook)
            ):
                raise ValueError("task-addressed runtime and session routing contracts differ")
        elif config.addressed:
            raise ValueError("historical runtime cannot use an addressed session contract")
        self.policy = policy
        self.graph = graph
        self.sessions = sessions
        self._sample_actions: Callable[..., Any] = sample_actions
        self._prior_stepper = (
            LingBotNativePriorStepper(policy, graph) if graph.unified_predict_correct else None
        )

    def sample_actions(
        self,
        observation: NativeObservationBatch,
        *,
        model_inputs: Mapping[str, Any],
        noise: torch.Tensor,
    ) -> NativePolicyStep:
        batch_size = len(observation.environment_keys)
        device = self.graph._parameter_reference.device
        audit_native_runtime_inputs(model_inputs, batch_size=batch_size, device=device)
        if (
            not isinstance(noise, torch.Tensor)
            or noise.ndim != 3
            or noise.shape[0] != batch_size
            or not noise.is_floating_point()
            or noise.device != device
            or not torch.isfinite(noise).all()
        ):
            raise ValueError("native runtime noise must be finite [batch,horizon,width]")
        transaction = self.sessions.prepare(observation)
        try:
            with torch.inference_mode():
                if self.graph.unified_predict_correct:
                    expected_state_type = (
                        AddressedLayerwisePosteriorState
                        if self.graph.task_query_object_value_read
                        else NativeLayerwisePosteriorState
                    )
                    if transaction.previous_state is not None and not isinstance(
                        transaction.previous_state,
                        expected_state_type,
                    ):
                        raise RuntimeError("unified runtime received non-layerwise session state")
                    if self._prior_stepper is None:
                        raise RuntimeError("unified runtime lost its shared-host prior stepper")
                    source: NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None = (
                        transaction.previous_state
                    )
                    source_valid = transaction.previous_state_valid
                    for controls in observation.effective_prior_control_chunks:
                        advanced = self._prior_stepper(
                            source,
                            controls,
                            previous_memory_valid=source_valid,
                            episode_ids=(
                                transaction.episode_ids
                                if self.graph.task_query_object_value_read
                                else None
                            ),
                        )
                        expected_trace_type = (
                            AddressedLayerwisePriorTrace
                            if self.graph.task_query_object_value_read
                            else NativeLayerwisePriorTrace
                        )
                        if not isinstance(advanced, expected_trace_type):
                            raise RuntimeError(
                                "unified runtime prior pass returned the wrong schema"
                            )
                        source = advanced
                        source_valid = torch.ones(
                            batch_size,
                            dtype=torch.bool,
                            device=controls.values.device,
                        )
                    if not isinstance(source, expected_trace_type):
                        raise RuntimeError("unified runtime did not produce a prior trace")
                    context = native_context_from_prior_trace(
                        controls=observation.controls,
                        prior_trace=source,
                        modalities=observation.modalities,
                    )
                else:
                    context = native_context_from_persistent_state(
                        controls=observation.controls,
                        persistent_state=transaction.previous_state,
                        persistent_state_valid=transaction.previous_state_valid,
                        modalities=observation.modalities,
                    )
                actions = self._sample_actions(
                    **dict(model_inputs),
                    noise=noise,
                    picf_native_context=context,
                )
            if (
                not isinstance(actions, torch.Tensor)
                or actions.ndim != 3
                or actions.shape[0] != batch_size
                or not actions.is_floating_point()
                or not torch.isfinite(actions).all()
            ):
                raise RuntimeError("LingBot returned invalid native actions")
            if (
                context.prior_state is None
                or context.posterior_state is None
                or context.relation_output is None
            ):
                raise RuntimeError("LingBot did not finalize every native runtime output")
            persistent = (
                context.posterior_memory
                if self.graph.layerwise_recurrence
                else context.posterior_state
            )
            if persistent is None:
                raise RuntimeError("LingBot omitted the architecture-owned recurrent state")
            self.sessions.commit(transaction, persistent)
        except BaseException:
            self.sessions.abort(transaction)
            raise
        return NativePolicyStep(
            actions=actions,
            prior_state=context.prior_state,
            posterior_state=context.posterior_state,
            relations=context.relation_output,
        )
