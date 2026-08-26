"""Fail-stop one-transition episodic training over the PICF posterior stream.

This module is deliberately a thin orchestration layer. It does not own a
model, objective, sampler policy or second memory. It joins the frozen global
episode plan, rank-local detached posterior state and the official Accelerate
microstep contract at the optimizer-step transaction boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from picf_next.models.temporal import ObjectBeliefBatch
from picf_next.training.accelerate_runner import (
    AcceleratedMicrostepOutput,
    accelerated_microstep,
    distributed_rank_local_call,
)
from picf_next.training.control import (
    FrozenEpisodeStreamPlan,
    PlannedStreamMicrobatch,
    RunProgress,
)
from picf_next.training.stream_state import PosteriorStreamStateGroup

_BELIEF_FIELDS = (
    "address_mean",
    "content_mean",
    "geometry_mean",
    "geometry_covariance_diag",
    "existence_logits",
    "visibility_given_existence_logits",
    "measurement_age_s",
    "valid",
    "age",
)


@dataclass(frozen=True, slots=True)
class StatefulForwardOutput:
    """Minimal host-neutral output required to advance one posterior lane."""

    loss: torch.Tensor
    final_belief: ObjectBeliefBatch
    metrics: Mapping[str, float] | None = None
    final_loss_track_keys_by_row: tuple[tuple[str | None, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class StatefulOptimizerStepOutput:
    """Detached observability for one committed global optimizer attempt."""

    plan_step: int
    parameter_version_before: int
    parameter_version_after: int
    microsteps: tuple[AcceleratedMicrostepOutput, ...]
    metrics: tuple[Mapping[str, float], ...]

    @property
    def optimizer_step_was_skipped(self) -> bool:
        return self.microsteps[-1].optimizer_step_was_skipped


StatefulForward = Callable[
    [
        PlannedStreamMicrobatch,
        ObjectBeliefBatch,
        tuple[tuple[str | None, ...], ...],
    ],
    StatefulForwardOutput,
]


def _detach_belief(belief: ObjectBeliefBatch) -> ObjectBeliefBatch:
    return ObjectBeliefBatch(**{field: getattr(belief, field).detach() for field in _BELIEF_FIELDS})


def _validated_metrics(metrics: Mapping[str, float] | None) -> Mapping[str, float]:
    if metrics is None:
        return {}
    if not isinstance(metrics, Mapping):
        raise TypeError("stateful forward metrics must be a mapping")
    validated: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("stateful forward metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("stateful forward metric values must be real scalars")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("stateful forward metric values must be finite")
        validated[name] = numeric
    return validated


class StatefulEpisodeTrainingRunner:
    """Execute one accumulation cycle as a coordinated fail-stop transaction.

    A failure poisons this in-memory runner. The caller must restart from the
    latest completed checkpoint because the final microstep may already have
    changed optimizer/scaler state before a later scheduler or commit failure.
    This is deliberately not described as rollback atomicity: distributed ranks
    agree on success or failure, but an already-mutated optimizer cannot be
    restored in place without loading the last completed checkpoint.
    """

    def __init__(
        self,
        *,
        accelerator: Any,
        model: Any,
        state_producer: torch.nn.Module,
        optimizer: Any,
        plan: FrozenEpisodeStreamPlan,
        progress: RunProgress,
        stream_state: PosteriorStreamStateGroup,
        lr_scheduler: Any | None = None,
        max_grad_norm: float = 0.0,
    ) -> None:
        if not isinstance(plan, FrozenEpisodeStreamPlan):
            raise TypeError("plan must be a FrozenEpisodeStreamPlan")
        if not isinstance(progress, RunProgress):
            raise TypeError("progress must be RunProgress")
        if not isinstance(stream_state, PosteriorStreamStateGroup):
            raise TypeError("stream_state must be PosteriorStreamStateGroup")
        if not isinstance(state_producer, torch.nn.Module):
            raise TypeError("state_producer must be a torch module")
        trainable_state_parameters = tuple(
            name for name, parameter in state_producer.named_parameters() if parameter.requires_grad
        )
        if trainable_state_parameters:
            raise ValueError(
                "stateful posterior carry requires a frozen state producer; trainable="
                f"{trainable_state_parameters[:8]}"
            )
        if progress.sample_plan_sha256 != plan.plan_sha256:
            raise ValueError("progress and frozen episode stream plan hashes differ")
        if progress.optimizer_global_batch_size != plan.global_batch_size:
            raise ValueError("progress and frozen episode stream global batch sizes differ")
        progress.validate_capacity(plan)

        accumulation_steps = getattr(accelerator, "gradient_accumulation_steps", None)
        if (
            not isinstance(accumulation_steps, int)
            or isinstance(accumulation_steps, bool)
            or accumulation_steps <= 0
        ):
            raise ValueError("accelerator must expose positive gradient_accumulation_steps")
        expected_names = tuple(f"accumulation-{index:05d}" for index in range(accumulation_steps))
        if stream_state.stream_names != expected_names:
            raise ValueError("stream-state shards differ from accelerator accumulation topology")
        if stream_state.has_pending_chunks:
            raise RuntimeError("stateful runner cannot start with pending posterior chunks")
        nonstationary_streams = tuple(
            name for name in stream_state.stream_names if stream_state[name].max_parameter_lag != 0
        )
        if nonstationary_streams:
            raise ValueError(
                "frozen-state-producer streaming requires max_parameter_lag=0; "
                f"streams={nonstationary_streams}"
            )
        if not bool(getattr(accelerator, "sync_gradients", False)):
            raise RuntimeError(
                "stateful runner must start at an optimizer synchronization boundary"
            )

        self.accelerator = accelerator
        self.model = model
        self.state_producer = state_producer
        self.optimizer = optimizer
        self.plan = plan
        self.progress = progress
        self.stream_state = stream_state
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = accumulation_steps
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def _fail_transaction(self) -> None:
        self.stream_state.abort_pending_chunks()
        for parameter in self.model.parameters():
            parameter.grad = None
        self._failed = True

    def run_optimizer_step(
        self,
        forward_step: StatefulForward,
    ) -> StatefulOptimizerStepOutput:
        """Advance every local lane exactly once and commit only after success."""

        if self._failed:
            raise RuntimeError("stateful runner is poisoned; restore a completed checkpoint")
        if not callable(forward_step):
            raise TypeError("forward_step must be callable")
        if self.stream_state.has_pending_chunks:
            raise RuntimeError("posterior chunks are pending before an optimizer attempt")
        self.progress.validate_capacity(self.plan)
        plan_step = self.progress.next_plan_step
        if plan_step >= self.plan.total_steps:
            raise StopIteration("frozen episode stream plan is exhausted")

        optimizer_parameter_version = self.progress.successful_optimizer_steps
        state_parameter_version = 0
        staged_beliefs: dict[str, ObjectBeliefBatch] = {}
        staged_loss_track_keys: dict[str, tuple[tuple[str | None, ...], ...]] = {}
        microstep_outputs: list[AcceleratedMicrostepOutput] = []
        metric_outputs: list[Mapping[str, float]] = []
        try:

            def prepare_all_stream_shards() -> tuple[
                tuple[
                    str,
                    PlannedStreamMicrobatch,
                    Any,
                    ObjectBeliefBatch,
                    tuple[tuple[str | None, ...], ...],
                ],
                ...,
            ]:
                prepared = []
                for accumulation_index in range(self.gradient_accumulation_steps):
                    stream_name = f"accumulation-{accumulation_index:05d}"
                    microbatch = self.plan.microbatch_for_rank(
                        plan_step,
                        rank=int(self.accelerator.process_index),
                        world_size=int(self.accelerator.num_processes),
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        accumulation_index=accumulation_index,
                    )
                    stream = self.stream_state[stream_name]
                    initial_belief = stream.prepare_planned_transitions(
                        microbatch.transitions,
                        current_parameter_version=state_parameter_version,
                    )
                    prepared.append(
                        (
                            stream_name,
                            microbatch,
                            stream,
                            initial_belief,
                            stream.pending_loss_track_keys_by_row,
                        )
                    )
                return tuple(prepared)

            prepared_shards = distributed_rank_local_call(
                self.accelerator,
                label="posterior preparation for optimizer step",
                action=prepare_all_stream_shards,
            )
            for accumulation_index, prepared in enumerate(prepared_shards):
                stream_name, microbatch, stream, initial_belief, initial_loss_tracks = prepared
                captured: StatefulForwardOutput | None = None
                captured_metrics: Mapping[str, float] = {}

                def forward_loss(
                    planned: PlannedStreamMicrobatch = microbatch,
                    initial: ObjectBeliefBatch = initial_belief,
                    loss_tracks: tuple[tuple[str | None, ...], ...] = initial_loss_tracks,
                    active_stream=stream,
                ) -> torch.Tensor:
                    nonlocal captured, captured_metrics
                    output = forward_step(planned, initial, loss_tracks)
                    if not isinstance(output, StatefulForwardOutput):
                        raise TypeError("forward_step must return StatefulForwardOutput")
                    if not isinstance(output.loss, torch.Tensor) or output.loss.ndim != 0:
                        raise TypeError("stateful forward loss must be one scalar tensor")
                    state_is_valid = active_stream.candidate_value_validity(output.final_belief)
                    invalid_guard = torch.where(
                        state_is_valid,
                        output.loss.new_zeros(()),
                        output.loss.new_full((), float("nan")),
                    )
                    captured_metrics = _validated_metrics(output.metrics)
                    captured = output
                    return output.loss + invalid_guard

                microstep = accelerated_microstep(
                    accelerator=self.accelerator,
                    model=self.model,
                    optimizer=self.optimizer,
                    forward_loss=forward_loss,
                    lr_scheduler=self.lr_scheduler,
                    max_grad_norm=self.max_grad_norm,
                )
                expected_boundary = accumulation_index + 1 == self.gradient_accumulation_steps
                if microstep.synchronization_boundary != expected_boundary:
                    raise RuntimeError("Accelerate accumulation boundary differs from frozen plan")
                if captured is None:
                    raise RuntimeError("successful microstep did not expose a posterior output")
                staged_beliefs[stream_name] = _detach_belief(captured.final_belief)
                staged_loss_track_keys[stream_name] = (
                    initial_loss_tracks
                    if captured.final_loss_track_keys_by_row is None
                    else captured.final_loss_track_keys_by_row
                )
                metric_outputs.append(dict(captured_metrics))
                microstep_outputs.append(microstep)
                captured = None

            final_microstep = microstep_outputs[-1]

            def commit_posterior_and_progress() -> None:
                self.stream_state.commit_prepared_chunks(
                    staged_beliefs,
                    transition_count=1,
                    state_parameter_version=state_parameter_version,
                    final_loss_track_keys_by_row=staged_loss_track_keys,
                )
                self.progress.advance_optimizer_step(
                    optimizer_step_was_skipped=final_microstep.optimizer_step_was_skipped
                )

            distributed_rank_local_call(
                self.accelerator,
                label="posterior and progress optimizer-step commit",
                action=commit_posterior_and_progress,
            )
        except Exception:
            self._fail_transaction()
            raise

        return StatefulOptimizerStepOutput(
            plan_step=plan_step,
            parameter_version_before=optimizer_parameter_version,
            parameter_version_after=self.progress.successful_optimizer_steps,
            microsteps=tuple(microstep_outputs),
            metrics=tuple(metric_outputs),
        )
