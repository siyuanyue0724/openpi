"""Transactional temporal orchestration for the LingBot unified PICF graph.

This module owns no learned state. It turns ordered dataset metadata into the
single :class:`LingBotUnifiedContext` consumed by the native per-layer graph,
then publishes exactly one detached posterior per worker lane only after a
successful optimizer step. Failed forwards, failed optimizer steps and partially
invalid batches cannot mutate persistent session state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from picf_next.hosts.lingbot_unified import (
    LingBotUnifiedBeliefGraph,
    LingBotUnifiedContext,
)
from picf_next.unified.objective import (
    DeclaredUnifiedObjective,
    ObjectiveTerm,
    UnifiedObjective,
    combine_declared_objective,
    normalized_scalar_term,
)
from picf_next.unified.predictive import (
    PredictionQueryRequest,
    PredictiveTarget,
    PredictiveTargetProvenance,
    predictive_source_batch_digest,
    row_conditioned_predictive_term,
)
from picf_next.unified.state import (
    UnifiedBeliefState,
    deterministic_birth_noise,
    empty_belief_state,
    stack_belief_states,
    unbind_belief_state,
)
from picf_next.unified.temporal import (
    EpisodeLaneBank,
    LaneStateError,
    SparseBPTTPlan,
    StateStamp,
)

_SESSION_MAGIC = b"PICFUS01"
_SESSION_VERSION = 2
_SESSION_HEADER = struct.Struct("<8sBII")


def _optional_official_scalar(
    name: str,
    value: Any,
    *,
    reference: torch.Tensor,
) -> ObjectiveTerm | None:
    if isinstance(value, bool):
        raise TypeError(f"official {name} loss cannot be boolean")
    if isinstance(value, (int, float)):
        if value != 0:
            raise TypeError(f"nonzero official {name} loss must be a tensor")
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"official {name} loss has an unsupported type")
    if value.device != reference.device:
        raise ValueError("official LingBot loss tensors must share one device")
    return normalized_scalar_term(name, value)


def combine_lingbot_policy_objective(
    model_outputs: Sequence[Any],
    *,
    set_supervision: tuple[ObjectiveTerm, ...] = (),
    cross_modal_prediction: tuple[ObjectiveTerm, ...] = (),
    future_prediction: tuple[ObjectiveTerm, ...] = (),
    overshooting: tuple[ObjectiveTerm, ...] = (),
) -> UnifiedObjective:
    """Compose PICF losses around LingBot's exact released training tuple.

    The released policy returns total, action, current-depth, future-depth,
    future-video and sequence losses in positions 0-5. The differentiable
    remainder preserves host router/MoE regularization without relying on the
    detached logging dictionary in position 6.
    """

    if len(model_outputs) != 11:
        raise ValueError("LingBot VLA2 training must return the pinned 11-item tuple")
    total, action, current_depth, future_depth, future_video, sequence = model_outputs[:6]
    if not isinstance(total, torch.Tensor) or not isinstance(action, torch.Tensor):
        raise TypeError("official LingBot total and action losses must be tensors")
    if total.ndim != 0 or action.ndim != 0:
        raise ValueError("official LingBot total and action losses must be scalar")
    if total.device != action.device:
        raise ValueError("official LingBot total and action losses must share one device")

    current_depth_term = _optional_official_scalar(
        "host/current_depth",
        current_depth,
        reference=action,
    )
    future_depth_term = _optional_official_scalar(
        "future/host_depth",
        future_depth,
        reference=action,
    )
    future_video_term = _optional_official_scalar(
        "future/host_video",
        future_video,
        reference=action,
    )
    sequence_term = _optional_official_scalar(
        "host/sequence",
        sequence,
        reference=action,
    )
    differentiable_components = [action]
    for term in (current_depth_term, future_depth_term, future_video_term, sequence_term):
        if term is not None:
            differentiable_components.append(term.values[0])
    host_remainder = total - sum(differentiable_components[1:], differentiable_components[0])
    declaration = DeclaredUnifiedObjective(
        action=normalized_scalar_term("action", action),
        host_regularization=tuple(
            term for term in (current_depth_term, sequence_term) if term is not None
        )
        + (normalized_scalar_term("host/remainder", host_remainder),),
        set_supervision=set_supervision,
        cross_modal_prediction=cross_modal_prediction,
        future_prediction=tuple(
            term for term in (future_depth_term, future_video_term) if term is not None
        )
        + future_prediction,
        overshooting=overshooting,
    )
    return combine_declared_objective(declaration)


def lingbot_row_prediction_term(
    context: LingBotUnifiedContext,
    target: PredictiveTarget,
    provenance: PredictiveTargetProvenance,
    *,
    weight: float,
) -> ObjectiveTerm:
    """Bind the final shared-host row query to its immutable loss-side target."""

    if context.prediction_request is None:
        raise ValueError("row prediction target has no source-side query request")
    if context.row_prediction_hidden is None:
        raise RuntimeError("LingBot host did not publish final row prediction hidden states")
    provenance.validate_target(target)
    return row_conditioned_predictive_term(
        context.row_prediction_hidden,
        target,
        context.prediction_request,
        weight=weight,
    )


@dataclass(frozen=True, slots=True)
class LingBotUnifiedSessionConfig:
    """Static state and replay contract shared by every ordered lane."""

    model_family_digest: str
    capacity: int
    max_optimizer_lag: int = 1
    birth_hazard: float = 0.01
    birth_noise_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_family_digest, str) or not self.model_family_digest:
            raise ValueError("model_family_digest must be a non-empty string")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise TypeError("belief capacity must be an integer")
        if self.capacity <= 0:
            raise ValueError("belief capacity must be positive")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.max_optimizer_lag, self.birth_noise_seed)
        ):
            raise TypeError("staleness and birth-noise seed must be integers")
        if self.max_optimizer_lag < 0 or self.birth_noise_seed < 0:
            raise ValueError("staleness and birth-noise seed must be non-negative")
        if isinstance(self.birth_hazard, bool) or not isinstance(self.birth_hazard, (int, float)):
            raise TypeError("birth_hazard must be real-valued")
        if not math.isfinite(self.birth_hazard) or not 0 < self.birth_hazard < 1:
            raise ValueError("birth_hazard must be finite and lie strictly between zero and one")

    @property
    def contract_digest(self) -> str:
        payload = {
            "birth_hazard": float(self.birth_hazard).hex(),
            "birth_noise_seed": self.birth_noise_seed,
            "capacity": self.capacity,
            "max_optimizer_lag": self.max_optimizer_lag,
            "model_family_digest": self.model_family_digest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LingBotUnifiedStepBatch:
    """Causal metadata for one ordered training or deployment batch."""

    lane_ids: tuple[int, ...]
    episode_keys: tuple[str, ...]
    frame_indices: tuple[int, ...]
    reset: tuple[bool, ...]
    optimizer_step: int
    elapsed_time: torch.Tensor
    previous_executed_action: torch.Tensor
    previous_action_valid: torch.Tensor
    modality_geometry_valid: torch.Tensor | None = None
    native_roles: torch.Tensor | None = None
    native_valid: torch.Tensor | None = None
    native_footprint: torch.Tensor | None = None
    native_modality_ids: torch.Tensor | None = None
    native_group_ids: torch.Tensor | None = None
    prediction_request: PredictionQueryRequest | None = None

    def __post_init__(self) -> None:
        size = len(self.lane_ids)
        if size == 0:
            raise ValueError("a unified step batch cannot be empty")
        if len(set(self.lane_ids)) != size:
            raise ValueError("lane IDs must be unique within a batch")
        if any(
            isinstance(lane_id, bool) or not isinstance(lane_id, int) or lane_id < 0
            for lane_id in self.lane_ids
        ):
            raise ValueError("lane IDs must be non-negative")
        if not (len(self.episode_keys) == len(self.frame_indices) == len(self.reset) == size):
            raise ValueError("lane metadata lengths must match")
        if any(not isinstance(key, str) or not key for key in self.episode_keys):
            raise ValueError("episode keys must be non-empty strings")
        if any(
            isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
            for frame in self.frame_indices
        ):
            raise ValueError("frame indices must be non-negative integers")
        if any(not isinstance(value, bool) for value in self.reset):
            raise TypeError("reset flags must be boolean")
        if (
            isinstance(self.optimizer_step, bool)
            or not isinstance(self.optimizer_step, int)
            or self.optimizer_step < 0
        ):
            raise ValueError("optimizer_step must be non-negative")
        if self.elapsed_time.shape != (size,):
            raise ValueError("elapsed_time must contain one value per lane")
        if not self.elapsed_time.is_floating_point() or not torch.isfinite(self.elapsed_time).all():
            raise ValueError("elapsed_time must be finite floating point")
        if (self.elapsed_time < 0).any():
            raise ValueError("elapsed_time must be non-negative")
        if self.previous_executed_action.ndim != 2 or (
            self.previous_executed_action.shape[0] != size
        ):
            raise ValueError("previous_executed_action must have shape [batch, action_dim]")
        if (
            not self.previous_executed_action.is_floating_point()
            or not torch.isfinite(self.previous_executed_action).all()
        ):
            raise ValueError("previous_executed_action must be finite floating point")
        if (
            self.previous_action_valid.shape != (size,)
            or self.previous_action_valid.dtype != torch.bool
        ):
            raise ValueError("previous_action_valid must be boolean with one value per lane")
        if self.elapsed_time.device != self.previous_executed_action.device or (
            self.previous_action_valid.device != self.previous_executed_action.device
        ):
            raise ValueError("causal step tensors must share one device")
        native_metadata = (
            self.native_roles,
            self.native_valid,
            self.native_footprint,
            self.native_modality_ids,
        )
        if any(value is None for value in native_metadata) and any(
            value is not None for value in native_metadata
        ):
            raise ValueError("native token metadata must be supplied completely or omitted")
        if all(value is not None for value in native_metadata):
            assert self.native_roles is not None
            assert self.native_valid is not None
            assert self.native_footprint is not None
            assert self.native_modality_ids is not None
            if self.native_roles.ndim != 2 or self.native_roles.shape[0] != size:
                raise ValueError("native_roles must have shape [batch, prefix_tokens]")
            if self.native_roles.dtype != torch.long:
                raise TypeError("native_roles must use torch.long")
            native_shape = self.native_roles.shape
            if self.native_valid.shape != native_shape or self.native_valid.dtype != torch.bool:
                raise ValueError("native_valid must be boolean and match native_roles")
            if (
                self.native_footprint.shape != native_shape
                or not self.native_footprint.is_floating_point()
                or not torch.isfinite(self.native_footprint).all()
                or (self.native_footprint < 0).any()
            ):
                raise ValueError("native_footprint must be finite, non-negative, and match roles")
            if (
                self.native_modality_ids.shape != native_shape
                or self.native_modality_ids.dtype != torch.long
            ):
                raise ValueError("native_modality_ids must be long and match native_roles")
            if any(
                value.device != self.previous_executed_action.device
                for value in (
                    self.native_roles,
                    self.native_valid,
                    self.native_footprint,
                    self.native_modality_ids,
                )
            ):
                raise ValueError("native token metadata must share the session device")
            if self.native_group_ids is not None and (
                self.native_group_ids.shape != self.native_roles.shape
                or self.native_group_ids.dtype != torch.long
                or self.native_group_ids.device != self.previous_executed_action.device
                or (self.native_group_ids < -1).any()
            ):
                raise ValueError(
                    "native_group_ids must be long, use -1 for independent tokens, "
                    "and match native_roles"
                )
        elif self.native_group_ids is not None:
            raise ValueError("native_group_ids require complete native token metadata")
        if self.prediction_request is not None and not isinstance(
            self.prediction_request,
            PredictionQueryRequest,
        ):
            raise TypeError("prediction_request must be a PredictionQueryRequest")


@dataclass(slots=True)
class PreparedLingBotUnifiedStep:
    """A context transaction that is committed only after a successful forward."""

    context: LingBotUnifiedContext
    batch: LingBotUnifiedStepBatch
    schema_digest: str
    model_family_digest: str
    committed: bool = False


@dataclass(frozen=True, slots=True)
class LingBotUnifiedForwardResult:
    """One host forward plus only the loss families valid for that step."""

    model_outputs: tuple[Any, ...]
    set_supervision: tuple[ObjectiveTerm, ...] = ()
    cross_modal_prediction: tuple[ObjectiveTerm, ...] = ()
    future_prediction: tuple[ObjectiveTerm, ...] = ()
    overshooting: tuple[ObjectiveTerm, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedLingBotUnifiedSequence:
    """A short recurrent transaction awaiting one optimizer outcome."""

    prepared_steps: tuple[PreparedLingBotUnifiedStep, ...]
    objectives: tuple[UnifiedObjective, ...]
    burn_in_steps: int

    def __post_init__(self) -> None:
        if not self.prepared_steps:
            raise ValueError("a prepared sequence cannot be empty")
        if isinstance(self.burn_in_steps, bool) or not isinstance(self.burn_in_steps, int):
            raise TypeError("burn-in steps must be an integer")
        if not 0 <= self.burn_in_steps < len(self.prepared_steps):
            raise ValueError("burn-in must leave at least one differentiable step")
        if len(self.objectives) != len(self.prepared_steps) - self.burn_in_steps:
            raise ValueError("every differentiable step requires exactly one objective")

    @property
    def loss(self) -> torch.Tensor:
        return torch.stack([objective.total for objective in self.objectives]).mean()


@dataclass(frozen=True, slots=True)
class LingBotUnifiedOptimizerAttemptResult:
    """Auditable outcome of one complete gradient-accumulation attempt."""

    optimizer_step: int
    source_digest: str
    normalized_loss: float
    published: bool


class LingBotUnifiedLaneSession:
    """Build and atomically advance the one-posterior-per-lane runtime state."""

    def __init__(
        self,
        graph: LingBotUnifiedBeliefGraph,
        config: LingBotUnifiedSessionConfig,
        *,
        lane_bank: EpisodeLaneBank | None = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.lane_bank = EpisodeLaneBank() if lane_bank is None else lane_bank
        codec = graph.config.codec
        self._schema_prefix = {
            "schema": "picf-next.unified-belief-state.v1",
            "content_dim": codec.content_dim,
            "geometry_dim": codec.geometry_dim,
            "geometry_schema": graph.config.geometry_schema.canonical_dict(),
            "uncertainty_dim": codec.uncertainty_dim,
            "modalities": graph.config.modality_names,
            "graph_contract_digest": graph.config.contract_digest,
        }
        self._capacity = config.capacity
        self._last_published_optimizer_step: int | None = None
        self._poisoned = False

    @property
    def schema_digest(self) -> str:
        return self._schema_digest(self._capacity)

    @property
    def last_published_optimizer_step(self) -> int | None:
        return self._last_published_optimizer_step

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _ensure_usable(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "unified session is poisoned; restore the last coordinated checkpoint"
            )

    def _schema_digest(self, capacity: int) -> str:
        payload = {**self._schema_prefix, "capacity": capacity}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def prepare(self, batch: LingBotUnifiedStepBatch) -> PreparedLingBotUnifiedStep:
        self._ensure_usable()
        geometry_valid, schema_digest = self._validate_batch_contract(batch)
        device = batch.previous_executed_action.device
        graph_config = self.graph.config

        states = []
        for lane_id, episode_key, frame_index, reset in zip(
            batch.lane_ids,
            batch.episode_keys,
            batch.frame_indices,
            batch.reset,
            strict=True,
        ):
            previous = None
            if not reset:
                previous = self.lane_bank.read_for_next_frame(
                    lane_id,
                    episode_key=episode_key,
                    frame_index=frame_index,
                    schema_digest=schema_digest,
                    model_family_digest=self.config.model_family_digest,
                    optimizer_step=batch.optimizer_step,
                    max_optimizer_lag=self.config.max_optimizer_lag,
                )
                if previous is None:
                    raise LaneStateError("non-reset lane has no cached posterior")
            if previous is None:
                previous = empty_belief_state(
                    batch_size=1,
                    capacity=self._capacity,
                    content_dim=graph_config.codec.content_dim,
                    geometry_dim=graph_config.codec.geometry_dim,
                    uncertainty_dim=graph_config.codec.uncertainty_dim,
                    birth_hazard=self.config.birth_hazard,
                    device=device,
                )
            states.append(previous)
        return self._prepare_with_previous(
            batch,
            previous=stack_belief_states(states, device=device),
            geometry_valid=geometry_valid,
            schema_digest=schema_digest,
        )

    def prepare_continuation(
        self,
        previous: PreparedLingBotUnifiedStep,
        batch: LingBotUnifiedStepBatch,
        *,
        detach_previous: bool,
    ) -> PreparedLingBotUnifiedStep:
        """Advance a verified short unroll without consulting detached lane state."""

        self._ensure_usable()
        if not isinstance(detach_previous, bool):
            raise TypeError("detach_previous must be boolean")
        if previous.committed:
            raise RuntimeError("a committed step cannot seed a differentiable continuation")
        if previous.schema_digest != self.schema_digest or (
            previous.model_family_digest != self.config.model_family_digest
        ):
            raise RuntimeError("previous step no longer matches the session contract")
        if batch.lane_ids != previous.batch.lane_ids:
            raise LaneStateError("a short unroll must preserve lane order")
        if batch.episode_keys != previous.batch.episode_keys:
            raise LaneStateError("a short unroll cannot cross episode identity")
        if any(batch.reset):
            raise LaneStateError("only the first step of a short unroll may reset a lane")
        expected_frames = tuple(frame + 1 for frame in previous.batch.frame_indices)
        if batch.frame_indices != expected_frames:
            raise LaneStateError("short-unroll frames must advance by exactly one")
        if batch.optimizer_step != previous.batch.optimizer_step:
            raise LaneStateError("one short unroll cannot cross an optimizer boundary")
        posterior = previous.context.posterior
        if posterior is None:
            raise RuntimeError("previous LingBot forward did not publish a posterior")
        if posterior.batch_size != len(batch.lane_ids):
            raise RuntimeError("previous posterior batch differs from the continuation batch")
        geometry_valid, schema_digest = self._validate_batch_contract(batch)
        prior = posterior.detached() if detach_previous else posterior
        return self._prepare_with_previous(
            batch,
            previous=prior,
            geometry_valid=geometry_valid,
            schema_digest=schema_digest,
        )

    def _validate_batch_contract(
        self,
        batch: LingBotUnifiedStepBatch,
    ) -> tuple[torch.Tensor, str]:
        device = batch.previous_executed_action.device
        graph_config = self.graph.config
        if batch.previous_executed_action.shape[-1] != graph_config.executed_action_dim:
            raise ValueError("previous action width differs from the LingBot host contract")
        geometry_valid = batch.modality_geometry_valid
        if geometry_valid is None:
            geometry_valid = torch.zeros(
                len(batch.lane_ids),
                graph_config.modality_count,
                self._capacity,
                graph_config.codec.geometry_dim,
                dtype=torch.bool,
                device=device,
            )
        expected_prefix = (len(batch.lane_ids), graph_config.modality_count)
        if (
            geometry_valid.ndim != 4
            or geometry_valid.shape[:2] != expected_prefix
            or geometry_valid.shape[-1] != graph_config.codec.geometry_dim
        ):
            raise ValueError("modality_geometry_valid has an incompatible schema")
        if geometry_valid.dtype != torch.bool or geometry_valid.device != device:
            raise ValueError("modality_geometry_valid must be boolean on the session device")
        if geometry_valid.shape[2] != self._capacity:
            raise ValueError("batch belief capacity differs from the immutable session schema")
        if batch.prediction_request is not None:
            expected_source = predictive_source_batch_digest(
                batch.episode_keys,
                batch.frame_indices,
            )
            if batch.prediction_request.source_batch_digest != expected_source:
                raise ValueError("prediction request is bound to a different source batch")
            if batch.prediction_request.source_batch_size != len(batch.lane_ids):
                raise ValueError("prediction request source batch size differs from lanes")
        return geometry_valid, self.schema_digest

    def _prepare_with_previous(
        self,
        batch: LingBotUnifiedStepBatch,
        *,
        previous: UnifiedBeliefState,
        geometry_valid: torch.Tensor,
        schema_digest: str,
    ) -> PreparedLingBotUnifiedStep:
        device = batch.previous_executed_action.device
        graph_config = self.graph.config
        expected_state = (
            len(batch.lane_ids),
            self._capacity,
            graph_config.codec.content_dim,
            graph_config.codec.geometry_dim,
            graph_config.codec.uncertainty_dim,
        )
        actual_state = (
            previous.batch_size,
            previous.capacity,
            previous.content_dim,
            previous.geometry_dim,
            previous.uncertainty_dim,
        )
        if actual_state != expected_state:
            raise ValueError("previous posterior differs from the immutable session schema")
        if previous.content.device != device:
            raise ValueError("previous posterior and step batch must share one device")
        context = LingBotUnifiedContext(
            previous_posterior=previous,
            modality_geometry_valid=geometry_valid,
            elapsed_time=batch.elapsed_time,
            previous_executed_action=batch.previous_executed_action,
            previous_action_valid=batch.previous_action_valid,
            birth_proposal_noise=deterministic_birth_noise(
                episode_keys=batch.episode_keys,
                frame_indices=batch.frame_indices,
                capacity=self._capacity,
                content_dim=graph_config.codec.content_dim,
                base_seed=self.config.birth_noise_seed,
                device=device,
            ),
            native_roles=batch.native_roles,
            native_valid=batch.native_valid,
            native_footprint=batch.native_footprint,
            native_modality_ids=batch.native_modality_ids,
            native_group_ids=batch.native_group_ids,
            prediction_request=batch.prediction_request,
        )
        return PreparedLingBotUnifiedStep(
            context=context,
            batch=batch,
            schema_digest=schema_digest,
            model_family_digest=self.config.model_family_digest,
        )

    def commit(
        self,
        prepared: PreparedLingBotUnifiedStep,
        *,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.commit_many((prepared,), checkpoint_path=checkpoint_path)

    def commit_many(
        self,
        prepared_steps: tuple[PreparedLingBotUnifiedStep, ...],
        *,
        checkpoint_path: Path | None = None,
    ) -> None:
        """Atomically publish one-step accumulation shards."""

        self.commit_sequences(
            tuple((prepared,) for prepared in prepared_steps),
            checkpoint_path=checkpoint_path,
        )

    def commit_sequences(
        self,
        prepared_sequences: tuple[tuple[PreparedLingBotUnifiedStep, ...], ...],
        *,
        checkpoint_path: Path | None = None,
    ) -> None:
        """Atomically publish complete short unrolls from one optimizer attempt."""

        self._ensure_usable()
        if not prepared_sequences or any(not sequence for sequence in prepared_sequences):
            raise ValueError("at least one nonempty prepared sequence is required")
        prepared_steps = tuple(prepared for sequence in prepared_sequences for prepared in sequence)
        if any(prepared.committed for prepared in prepared_steps):
            raise RuntimeError("a prepared unified step can only be committed once")
        optimizer_steps = {prepared.batch.optimizer_step for prepared in prepared_steps}
        if len(optimizer_steps) != 1:
            raise RuntimeError("one atomic commit cannot mix optimizer steps")
        optimizer_step = next(iter(optimizer_steps))
        if self._last_published_optimizer_step is not None and (
            optimizer_step <= self._last_published_optimizer_step
        ):
            raise RuntimeError("optimizer publication steps must advance monotonically")
        final_lane_ids = tuple(
            lane_id for sequence in prepared_sequences for lane_id in sequence[-1].batch.lane_ids
        )
        if len(set(final_lane_ids)) != len(final_lane_ids):
            raise RuntimeError("independent accumulation sequences cannot write one lane twice")

        staged_bank = EpisodeLaneBank.from_snapshot(self.lane_bank.snapshot())
        for sequence in prepared_sequences:
            for sequence_index, prepared in enumerate(sequence):
                if prepared.schema_digest != self.schema_digest or (
                    prepared.model_family_digest != self.config.model_family_digest
                ):
                    raise RuntimeError("prepared step no longer matches the session contract")
                if sequence_index:
                    prior = sequence[sequence_index - 1]
                    if prepared.batch.lane_ids != prior.batch.lane_ids or (
                        prepared.batch.episode_keys != prior.batch.episode_keys
                    ):
                        raise RuntimeError("prepared sequence changed lane or episode identity")
                    if prepared.batch.frame_indices != tuple(
                        frame + 1 for frame in prior.batch.frame_indices
                    ):
                        raise RuntimeError("prepared sequence is not frame-contiguous")
                    if any(prepared.batch.reset):
                        raise RuntimeError("a prepared sequence reset after its first step")
                posterior = prepared.context.posterior
                if posterior is None:
                    raise RuntimeError("LingBot forward did not publish a posterior")
                if posterior.batch_size != len(prepared.batch.lane_ids):
                    raise RuntimeError("posterior batch differs from the prepared lane batch")
                states = unbind_belief_state(posterior)
                records = tuple(
                    (
                        lane_id,
                        state,
                        StateStamp(
                            episode_key=episode_key,
                            frame_index=frame_index,
                            schema_digest=prepared.schema_digest,
                            model_family_digest=prepared.model_family_digest,
                            optimizer_step=prepared.batch.optimizer_step,
                        ),
                        reset,
                    )
                    for lane_id, state, episode_key, frame_index, reset in zip(
                        prepared.batch.lane_ids,
                        states,
                        prepared.batch.episode_keys,
                        prepared.batch.frame_indices,
                        prepared.batch.reset,
                        strict=True,
                    )
                )
                staged_bank.write_batch(records)
        if checkpoint_path is not None:
            self._write_atomic_snapshot(
                checkpoint_path,
                self._snapshot(
                    staged_bank,
                    published_optimizer_step=optimizer_step,
                ),
            )
        self.lane_bank = staged_bank
        self._last_published_optimizer_step = optimizer_step
        for prepared in prepared_steps:
            prepared.committed = True

    def publish_after_optimizer_step(
        self,
        prepared_steps: tuple[PreparedLingBotUnifiedStep, ...],
        *,
        optimizer_step_succeeded: bool,
        checkpoint_path: Path | None = None,
    ) -> bool:
        """Publish posterior state only after the optimizer reports success."""

        self._ensure_usable()
        if not isinstance(optimizer_step_succeeded, bool):
            raise TypeError("optimizer_step_succeeded must be boolean")
        if not optimizer_step_succeeded:
            return False
        try:
            self.commit_many(prepared_steps, checkpoint_path=checkpoint_path)
        except BaseException:
            self._poisoned = True
            raise
        return True

    def publish_sequences_after_optimizer_step(
        self,
        prepared_sequences: tuple[PreparedLingBotUnifiedSequence, ...],
        *,
        optimizer_step_succeeded: bool,
        checkpoint_path: Path | None = None,
    ) -> bool:
        """Publish short recurrent windows only after one successful optimizer step."""

        self._ensure_usable()
        if not isinstance(optimizer_step_succeeded, bool):
            raise TypeError("optimizer_step_succeeded must be boolean")
        if not optimizer_step_succeeded:
            return False
        try:
            self.commit_sequences(
                tuple(sequence.prepared_steps for sequence in prepared_sequences),
                checkpoint_path=checkpoint_path,
            )
        except BaseException:
            self._poisoned = True
            raise
        return True

    def complete_sequences_optimizer_transaction(
        self,
        prepared_sequences: tuple[PreparedLingBotUnifiedSequence, ...],
        *,
        optimizer_attempt: Callable[[], bool],
        checkpoint_path: Path | None = None,
    ) -> bool:
        """Execute the non-rollback boundary and publish or fail-stop as one API.

        ``optimizer_attempt`` owns the optimizer, scaler and scheduler update and
        returns ``False`` only when no parameter update occurred (for example an
        AMP overflow skip). Once the callback raises or reports an invalid
        result, in-process recovery is unsafe because optimizer internals may
        already have changed; the session is therefore poisoned.
        """

        self._ensure_usable()
        if not callable(optimizer_attempt):
            raise TypeError("optimizer_attempt must be callable")
        try:
            succeeded = optimizer_attempt()
            if not isinstance(succeeded, bool):
                raise TypeError("optimizer_attempt must return boolean success")
        except BaseException:
            self._poisoned = True
            raise
        return self.publish_sequences_after_optimizer_step(
            prepared_sequences,
            optimizer_step_succeeded=succeeded,
            checkpoint_path=checkpoint_path,
        )

    def snapshot(self) -> bytes:
        """Serialize one session with graph, model-family and lane-bank binding."""

        self._ensure_usable()
        if self._last_published_optimizer_step is None:
            raise RuntimeError("cannot snapshot a session before its first optimizer publication")
        return self._snapshot(
            self.lane_bank,
            published_optimizer_step=self._last_published_optimizer_step,
        )

    def _snapshot(
        self,
        lane_bank: EpisodeLaneBank,
        *,
        published_optimizer_step: int,
    ) -> bytes:
        bank_payload = lane_bank.snapshot()
        metadata = json.dumps(
            {
                "capacity": self._capacity,
                "lane_bank_sha256": hashlib.sha256(bank_payload).hexdigest(),
                "model_family_digest": self.config.model_family_digest,
                "published_optimizer_step": published_optimizer_step,
                "schema_digest": self.schema_digest,
                "session_config_digest": self.config.contract_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return bytes(
            _SESSION_HEADER.pack(
                _SESSION_MAGIC,
                _SESSION_VERSION,
                len(metadata),
                len(bank_payload),
            )
            + metadata
            + bank_payload
        )

    @staticmethod
    def _write_atomic_snapshot(path: Path, payload: bytes) -> None:
        path = path.expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def from_snapshot(
        cls,
        graph: LingBotUnifiedBeliefGraph,
        config: LingBotUnifiedSessionConfig,
        payload: bytes,
        *,
        expected_optimizer_step: int,
    ) -> LingBotUnifiedLaneSession:
        if len(payload) < _SESSION_HEADER.size:
            raise ValueError("unified session snapshot is truncated")
        magic, version, metadata_size, bank_size = _SESSION_HEADER.unpack_from(payload)
        if magic != _SESSION_MAGIC or version != _SESSION_VERSION:
            raise ValueError("unified session snapshot schema is unsupported")
        metadata_end = _SESSION_HEADER.size + metadata_size
        bank_end = metadata_end + bank_size
        if bank_end != len(payload):
            raise ValueError("unified session snapshot lengths are inconsistent")
        metadata = json.loads(payload[_SESSION_HEADER.size : metadata_end].decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("unified session snapshot metadata must be a mapping")
        required = {
            "capacity",
            "lane_bank_sha256",
            "model_family_digest",
            "published_optimizer_step",
            "schema_digest",
            "session_config_digest",
        }
        if set(metadata) != required:
            raise ValueError("unified session snapshot metadata is incomplete")
        bank_payload = payload[metadata_end:bank_end]
        if hashlib.sha256(bank_payload).hexdigest() != metadata["lane_bank_sha256"]:
            raise ValueError("unified session lane-bank digest differs")
        if metadata["model_family_digest"] != config.model_family_digest:
            raise ValueError("unified session model-family digest differs")
        capacity = metadata["capacity"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("unified session capacity is invalid")
        if capacity != config.capacity:
            raise ValueError("unified session capacity differs from the configured graph")
        if metadata["session_config_digest"] != config.contract_digest:
            raise ValueError("unified session process contract differs")
        published_optimizer_step = metadata["published_optimizer_step"]
        if (
            isinstance(published_optimizer_step, bool)
            or not isinstance(published_optimizer_step, int)
            or published_optimizer_step < 0
        ):
            raise ValueError("unified session optimizer step is invalid")
        if (
            isinstance(expected_optimizer_step, bool)
            or not isinstance(expected_optimizer_step, int)
            or expected_optimizer_step < 0
        ):
            raise ValueError("expected optimizer step is invalid")
        if published_optimizer_step != expected_optimizer_step:
            raise ValueError("unified session optimizer checkpoint differs")
        session = cls(
            graph,
            config,
            lane_bank=EpisodeLaneBank.from_snapshot(bank_payload),
        )
        session._last_published_optimizer_step = published_optimizer_step
        if session.schema_digest != metadata["schema_digest"]:
            raise ValueError("unified session graph schema differs")
        return session

    @classmethod
    def load_snapshot(
        cls,
        graph: LingBotUnifiedBeliefGraph,
        config: LingBotUnifiedSessionConfig,
        path: Path,
        *,
        expected_optimizer_step: int,
    ) -> LingBotUnifiedLaneSession:
        return cls.from_snapshot(
            graph,
            config,
            path.expanduser().read_bytes(),
            expected_optimizer_step=expected_optimizer_step,
        )


def run_lingbot_unified_sequence(
    session: LingBotUnifiedLaneSession,
    batches: Sequence[LingBotUnifiedStepBatch],
    plan: SparseBPTTPlan,
    forward_step: Callable[
        [int, LingBotUnifiedContext, bool],
        LingBotUnifiedForwardResult,
    ],
) -> PreparedLingBotUnifiedSequence:
    """Execute one burn-in plus short-BPTT transaction without publishing state.

    The callback invokes the official host forward exactly once per listed
    observation. Burn-in runs under ``no_grad`` and each transition into the
    differentiable window receives an explicitly detached boundary state.
    Publication remains a separate optimizer-success transaction on the
    session, so any exception leaves persistent lanes unchanged.
    """

    if not isinstance(session, LingBotUnifiedLaneSession):
        raise TypeError("session must be a LingBotUnifiedLaneSession")
    if not isinstance(plan, SparseBPTTPlan):
        raise TypeError("plan must be a SparseBPTTPlan")
    if len(batches) != plan.loaded_steps:
        raise ValueError("sequence length must equal burn-in plus differentiable steps")
    if not callable(forward_step):
        raise TypeError("forward_step must be callable")

    prepared_steps: list[PreparedLingBotUnifiedStep] = []
    objectives: list[UnifiedObjective] = []
    for index, batch in enumerate(batches):
        if index == 0:
            prepared = session.prepare(batch)
        else:
            prepared = session.prepare_continuation(
                prepared_steps[-1],
                batch,
                detach_previous=index <= plan.burn_in_steps,
            )
        differentiable = index >= plan.burn_in_steps
        if differentiable:
            result = forward_step(index, prepared.context, True)
        else:
            with torch.no_grad():
                result = forward_step(index, prepared.context, False)
        if not isinstance(result, LingBotUnifiedForwardResult):
            raise TypeError("forward_step must return LingBotUnifiedForwardResult")
        posterior = prepared.context.posterior
        if posterior is None:
            raise RuntimeError("LingBot sequence forward did not publish a posterior")
        posterior.validate()
        if differentiable:
            objectives.append(
                combine_lingbot_policy_objective(
                    result.model_outputs,
                    set_supervision=result.set_supervision,
                    cross_modal_prediction=result.cross_modal_prediction,
                    future_prediction=result.future_prediction,
                    overshooting=result.overshooting,
                )
            )
        prepared_steps.append(prepared)
    return PreparedLingBotUnifiedSequence(
        prepared_steps=tuple(prepared_steps),
        objectives=tuple(objectives),
        burn_in_steps=plan.burn_in_steps,
    )


def run_lingbot_unified_step(
    session: LingBotUnifiedLaneSession,
    batch: LingBotUnifiedStepBatch,
    forward_step: Callable[[LingBotUnifiedContext], LingBotUnifiedForwardResult],
) -> PreparedLingBotUnifiedSequence:
    """Run the normal one-observation training path without publishing state."""

    if not isinstance(session, LingBotUnifiedLaneSession):
        raise TypeError("session must be a LingBotUnifiedLaneSession")
    if not callable(forward_step):
        raise TypeError("forward_step must be callable")
    prepared = session.prepare(batch)
    result = forward_step(prepared.context)
    if not isinstance(result, LingBotUnifiedForwardResult):
        raise TypeError("forward_step must return LingBotUnifiedForwardResult")
    posterior = prepared.context.posterior
    if posterior is None:
        raise RuntimeError("LingBot step forward did not publish a posterior")
    posterior.validate()
    objective = combine_lingbot_policy_objective(
        result.model_outputs,
        set_supervision=result.set_supervision,
        cross_modal_prediction=result.cross_modal_prediction,
        future_prediction=result.future_prediction,
        overshooting=result.overshooting,
    )
    return PreparedLingBotUnifiedSequence(
        prepared_steps=(prepared,),
        objectives=(objective,),
        burn_in_steps=0,
    )


def lingbot_optimizer_source_digest(
    microbatches: Sequence[tuple[LingBotUnifiedStepBatch, Mapping[str, Any], str]],
) -> str:
    """Content-address one ordered gradient-accumulation source transaction."""

    source_payload = [
        {
            "episode_keys": batch.episode_keys,
            "frame_indices": batch.frame_indices,
            "lane_ids": batch.lane_ids,
            "reset": batch.reset,
            "source_digest": source_digest,
        }
        for batch, _, source_digest in microbatches
    ]
    if any(
        not isinstance(item["source_digest"], str)
        or len(item["source_digest"]) != 64
        or any(character not in "0123456789abcdef" for character in item["source_digest"])
        for item in source_payload
    ):
        raise ValueError("every microbatch requires a lowercase source SHA-256 digest")
    return hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_lingbot_unified_optimizer_attempt(
    session: LingBotUnifiedLaneSession,
    microbatches: Sequence[tuple[LingBotUnifiedStepBatch, Mapping[str, Any], str]],
    *,
    forward_step: Callable[
        [Mapping[str, Any], LingBotUnifiedContext],
        LingBotUnifiedForwardResult,
    ],
    backward_step: Callable[[torch.Tensor], None],
    optimizer_attempt: Callable[[], bool],
    clear_gradients_after_skip: Callable[[], None],
    checkpoint_path: Path | None = None,
) -> LingBotUnifiedOptimizerAttemptResult:
    """Execute one source-addressed official-host optimizer transaction.

    A skipped update leaves the lane bank untouched and clears gradients inside
    the fail-stop callback boundary. The caller must retry the same optimizer
    step; a deterministic stream plan then reproduces ``source_digest``.
    """

    if not microbatches:
        raise ValueError("an optimizer attempt requires at least one microbatch")
    if not all(callable(fn) for fn in (forward_step, backward_step, optimizer_attempt)):
        raise TypeError("forward, backward and optimizer callbacks must be callable")
    if not callable(clear_gradients_after_skip):
        raise TypeError("clear_gradients_after_skip must be callable")
    optimizer_steps = {batch.optimizer_step for batch, _, _ in microbatches}
    if len(optimizer_steps) != 1:
        raise ValueError("one optimizer attempt cannot mix optimizer-step indices")
    optimizer_step = next(iter(optimizer_steps))
    final_lane_ids = tuple(lane for batch, _, _ in microbatches for lane in batch.lane_ids)
    if len(set(final_lane_ids)) != len(final_lane_ids):
        raise ValueError("gradient-accumulation microbatches cannot repeat a lane")
    source_digest = lingbot_optimizer_source_digest(microbatches)

    prepared_sequences: list[PreparedLingBotUnifiedSequence] = []
    detached_losses: list[torch.Tensor] = []
    scale = 1.0 / len(microbatches)
    try:
        for batch, model_inputs, _ in microbatches:
            if not isinstance(model_inputs, Mapping):
                raise TypeError("official LingBot model inputs must be a mapping")
            sequence = run_lingbot_unified_step(
                session,
                batch,
                lambda context, inputs=model_inputs: forward_step(inputs, context),
            )
            if not torch.isfinite(sequence.loss):
                raise FloatingPointError("unified LingBot objective is non-finite")
            backward_step(sequence.loss * scale)
            detached_losses.append(sequence.loss.detach().float())
            prepared_sequences.append(sequence)
    except BaseException:
        clear_gradients_after_skip()
        raise

    def optimizer_boundary() -> bool:
        succeeded = optimizer_attempt()
        if not isinstance(succeeded, bool):
            raise TypeError("optimizer_attempt must return boolean success")
        if not succeeded:
            clear_gradients_after_skip()
        return succeeded

    published = session.complete_sequences_optimizer_transaction(
        tuple(prepared_sequences),
        optimizer_attempt=optimizer_boundary,
        checkpoint_path=checkpoint_path,
    )
    normalized_loss = float(torch.stack(detached_losses).mean().item())
    return LingBotUnifiedOptimizerAttemptResult(
        optimizer_step=optimizer_step,
        source_digest=source_digest,
        normalized_loss=normalized_loss,
        published=published,
    )
