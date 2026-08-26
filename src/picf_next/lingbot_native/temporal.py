"""Efficient long-time training contracts for the native posterior.

The dominant estimator advances one detached real-age lane by one frame.  A
source-independent sampler adds sparse local BPTT and log-spaced row-only
overshooting.  This module owns bookkeeping, not physical state semantics or a
second transition model.  Full-prefix replay is an offline fixed-weight audit,
never a state-changing training branch.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.host import LingBotNativePriorStepper
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.row_binding import (
    RowBindings,
    normalize_row_bindings,
    row_binding_map,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    NativeLayerwisePosteriorState,
    NativePersistentState,
    NativePosteriorState,
    NativeVidEoMTPairedPosteriorState,
    clone_persistent_state,
    persistent_state_tensor,
)
from picf_next.training.control import derive_subseed

FROZEN_OVERSHOOT_HORIZONS = (1, 2, 4, 8, 16, 32, 64)
FROZEN_AUXILIARY_SAMPLING = "stratified_exclusive"
TEMPORAL_ESTIMATOR_SCHEMA = "picf-next.lingbot-native-temporal-estimator/v3"
TEMPORAL_BATCH_STREAM = "picf-next.lingbot-native-temporal-batch/v3"


class NativeLaneError(RuntimeError):
    """A cached training lane cannot causally serve the requested frame."""


def _probability(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite in [0,1]")


@dataclass(frozen=True, slots=True)
class TemporalEstimatorConfig:
    """Manifest-frozen probabilities; none may depend on current labels/loss."""

    local_bptt_probability: float
    overshoot_probability: float
    source_mask_probability: float
    maximum_optimizer_lag: int
    local_minimum_steps: int = 2
    local_maximum_steps: int = 4
    overshoot_horizons: tuple[int, ...] = FROZEN_OVERSHOOT_HORIZONS

    def __post_init__(self) -> None:
        for name in (
            "local_bptt_probability",
            "overshoot_probability",
            "source_mask_probability",
        ):
            _probability(getattr(self, name), name)
        if (
            math.fsum(
                (
                    self.local_bptt_probability,
                    self.overshoot_probability,
                    self.source_mask_probability,
                )
            )
            > 1.0
        ):
            raise ValueError("exclusive temporal auxiliary probabilities must sum to at most one")
        integers = (self.maximum_optimizer_lag, self.local_minimum_steps, self.local_maximum_steps)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("temporal estimator integer controls must be integers")
        if self.maximum_optimizer_lag < 0:
            raise ValueError("temporal optimizer lag must be non-negative")
        if (self.local_minimum_steps, self.local_maximum_steps) != (2, 4):
            raise ValueError("the frozen local BPTT range is exactly 2..4")
        if self.overshoot_horizons != FROZEN_OVERSHOOT_HORIZONS:
            raise ValueError("the frozen overshoot support is exactly 1,2,4,8,16,32,64")

    @property
    def metadata(self) -> dict[str, object]:
        """Canonical, exact configuration identity for plans and checkpoints."""

        return {
            "auxiliary_sampling": FROZEN_AUXILIARY_SAMPLING,
            "local_bptt_probability": float(self.local_bptt_probability).hex(),
            "local_maximum_steps": self.local_maximum_steps,
            "local_minimum_steps": self.local_minimum_steps,
            "maximum_optimizer_lag": self.maximum_optimizer_lag,
            "overshoot_horizons": self.overshoot_horizons,
            "overshoot_probability": float(self.overshoot_probability).hex(),
            "schema": TEMPORAL_ESTIMATOR_SCHEMA,
            "source_mask_probability": float(self.source_mask_probability).hex(),
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()


def native_temporal_batch_seed(
    *,
    parent_seed: int,
    comparison_id: str,
    optimizer_step: int,
    sample_keys: tuple[str, ...],
) -> int:
    """Derive one topology-neutral branch seed for a complete global batch."""

    if not isinstance(comparison_id, str) or not comparison_id:
        raise ValueError("temporal comparison ID must be non-empty")
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise ValueError("temporal optimizer step must be non-negative")
    if (
        not isinstance(sample_keys, tuple)
        or not sample_keys
        or any(not isinstance(value, str) or not value for value in sample_keys)
    ):
        raise ValueError("temporal global batch requires non-empty sample keys")
    identity = hashlib.sha256(
        json.dumps(sample_keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return derive_subseed(
        parent_seed,
        TEMPORAL_BATCH_STREAM,
        comparison_id,
        str(optimizer_step),
        identity,
    )


@dataclass(frozen=True, slots=True)
class TemporalBatchPlan:
    """One collective-safe auxiliary graph schedule for a global batch."""

    seed: int
    state_ages: tuple[int, ...]
    local_bptt_steps: int | None
    overshoot_horizon: int | None
    source_masked_branch: bool

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("temporal batch seed must be a non-negative integer")
        if not self.state_ages or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.state_ages
        ):
            raise ValueError("temporal batch state ages must be non-negative integers")
        if self.local_bptt_steps is not None and not 2 <= self.local_bptt_steps <= 4:
            raise ValueError("temporal batch local BPTT must contain 2..4 steps")
        if (
            self.overshoot_horizon is not None
            and self.overshoot_horizon not in FROZEN_OVERSHOOT_HORIZONS
        ):
            raise ValueError("temporal batch overshoot is outside frozen support")
        active_auxiliaries = sum(
            (
                self.local_bptt_steps is not None,
                self.overshoot_horizon is not None,
                self.source_masked_branch,
            )
        )
        if active_auxiliaries > 1:
            raise ValueError("a temporal batch may activate at most one sparse auxiliary")

    @property
    def digest(self) -> str:
        payload = {
            "local_bptt_steps": self.local_bptt_steps,
            "overshoot_horizon": self.overshoot_horizon,
            "schema": TEMPORAL_BATCH_STREAM,
            "seed": self.seed,
            "source_masked_branch": self.source_masked_branch,
            "state_ages": self.state_ages,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def sample_temporal_batch_plan(
    config: TemporalEstimatorConfig,
    *,
    seed: int,
    state_ages: tuple[int, ...],
    available_future_steps: tuple[int, ...],
    optimizer_lags: tuple[int, ...],
) -> TemporalBatchPlan:
    """Sample a rank-synchronous graph shape from source-known metadata only."""

    if not isinstance(config, TemporalEstimatorConfig):
        raise TypeError("temporal batch planning requires a frozen estimator config")
    batch = len(state_ages)
    if batch == 0 or not (len(available_future_steps) == len(optimizer_lags) == batch):
        raise ValueError("temporal batch metadata must share one non-empty batch axis")
    values = (*state_ages, *available_future_steps, *optimizer_lags)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("temporal batch metadata must contain non-negative integers")
    if any(value > config.maximum_optimizer_lag for value in optimizer_lags):
        raise NativeLaneError("a temporal batch lane exceeds the hard optimizer staleness limit")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("temporal batch seed must be a non-negative integer")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    auxiliary_draw = float(torch.rand((), generator=generator).item())
    local_boundary = config.local_bptt_probability
    overshoot_boundary = local_boundary + config.overshoot_probability
    source_boundary = overshoot_boundary + config.source_mask_probability
    selected_local = auxiliary_draw < local_boundary
    selected_overshoot = local_boundary <= auxiliary_draw < overshoot_boundary
    selected_source = overshoot_boundary <= auxiliary_draw < source_boundary
    common_future = min(available_future_steps)
    local_steps = None
    if selected_local and common_future >= 1:
        maximum = min(config.local_maximum_steps, common_future + 1)
        if maximum >= config.local_minimum_steps:
            local_steps = int(
                torch.randint(
                    config.local_minimum_steps,
                    maximum + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
    eligible_horizons = tuple(
        horizon for horizon in config.overshoot_horizons if horizon <= common_future
    )
    overshoot = None
    if selected_overshoot and eligible_horizons:
        index = int(torch.randint(len(eligible_horizons), (1,), generator=generator).item())
        overshoot = eligible_horizons[index]
    return TemporalBatchPlan(
        seed=seed,
        state_ages=state_ages,
        local_bptt_steps=local_steps,
        overshoot_horizon=overshoot,
        source_masked_branch=selected_source,
    )


@dataclass(frozen=True, slots=True)
class TemporalWorkload:
    """Measured forward counts per successful optimizer step."""

    local_extra_full_steps: float
    prior_row_steps: float
    source_masked_steps: float

    def __post_init__(self) -> None:
        values = (
            self.local_extra_full_steps,
            self.prior_row_steps,
            self.source_masked_steps,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("temporal workload rates must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TemporalCostProfile:
    """Measured per-forward timings; no branch probability is inferred here."""

    full_step_seconds: float
    row_step_seconds: float
    source_masked_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.full_step_seconds,
            self.row_step_seconds,
            self.source_masked_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("temporal cost components must be finite and non-negative")
        if self.full_step_seconds == 0:
            raise ValueError("full-step timing must be positive")

    def estimated_seconds(self, workload: TemporalWorkload) -> float:
        """Estimate the registered steady-state and sparse auxiliary work."""

        if not isinstance(workload, TemporalWorkload):
            raise TypeError("temporal timing requires a measured TemporalWorkload")
        return (
            self.full_step_seconds * (1 + workload.local_extra_full_steps)
            + self.row_step_seconds * workload.prior_row_steps
            + self.source_masked_seconds * workload.source_masked_steps
        )


@dataclass(frozen=True, slots=True)
class NativePriorPredictiveRollout:
    """One recursively rolled state and its same-graph predictive query output."""

    horizon: int
    state: NativePosteriorState
    target_name: str
    prediction: torch.Tensor
    request: NativePredictionRequest

    def __post_init__(self) -> None:
        if self.horizon not in FROZEN_OVERSHOOT_HORIZONS:
            raise ValueError("predictive rollout horizon is outside the frozen support")
        if self.request.source is not PredictionSource.PRIOR or (
            self.request.evidence is not PredictionEvidence.FUTURE
        ):
            raise ValueError("predictive rollout requires a prior-to-future request")
        if not (self.request.horizons == self.horizon).all():
            raise ValueError("predictive rollout request differs from its rollout horizon")
        if not isinstance(self.target_name, str) or not self.target_name:
            raise ValueError("predictive rollout target name must be non-empty")
        expected_prefix = (
            self.state.batch_size,
            self.state.capacity,
            self.request.query_count,
        )
        if self.prediction.ndim != 4 or self.prediction.shape[:3] != expected_prefix:
            raise ValueError("predictive rollout output differs from its row/query axes")
        if not self.prediction.is_floating_point() or not torch.isfinite(self.prediction).all():
            raise ValueError("predictive rollout output must be finite floating point")


def rollout_native_prior_prediction(
    stepper: LingBotNativePriorStepper,
    initial_state: NativePosteriorState,
    controls: tuple[ExecutedControlBatch, ...],
    *,
    request: NativePredictionRequest,
    target_name: str,
) -> NativePriorPredictiveRollout:
    """Roll the shared prior to one horizon and query that row in the same graph.

    Activation memory is owned by the host's transformer-layer checkpointing.
    Wrapping the FSDP2 root call here would create a nested recomputation
    boundary around a registered FSDP forward method.
    """

    if not isinstance(stepper, LingBotNativePriorStepper):
        raise TypeError("predictive rollout requires the parameter-free native stepper")
    if not isinstance(request, NativePredictionRequest):
        raise TypeError("predictive rollout requires a native prediction request")
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("predictive rollout target name must be non-empty")
    if request.source is not PredictionSource.PRIOR or (
        request.evidence is not PredictionEvidence.FUTURE
    ):
        raise ValueError("predictive rollout requires a prior-to-future request")
    horizons = request.horizons.detach().cpu().reshape(-1).tolist()
    if not horizons or len(set(horizons)) != 1:
        raise ValueError("one predictive rollout request must use one shared horizon")
    horizon = int(horizons[0])
    if horizon not in FROZEN_OVERSHOOT_HORIZONS:
        raise ValueError("predictive rollout horizon is outside the frozen support")
    if len(controls) < horizon:
        raise ValueError("predictive rollout lacks executed controls for its horizon")
    for control in controls[:horizon]:
        if control.batch_size != initial_state.batch_size:
            raise ValueError("predictive rollout controls and state batches differ")
        control.validate_bound(stepper.graph.config.maximum_control_tokens)

    rows = initial_state.rows
    prediction: torch.Tensor | None = None
    for step_index, control in enumerate(controls[:horizon], start=1):
        if step_index == horizon:
            state, prediction = stepper.step_with_prediction(
                NativePosteriorState(rows),
                control,
                request,
                target_name=target_name,
            )
            rows = state.rows
        else:
            rows = stepper(NativePosteriorState(rows), control).rows
    if prediction is None:
        raise RuntimeError("predictive row rollout produced no projected output")
    return NativePriorPredictiveRollout(
        horizon=horizon,
        state=NativePosteriorState(rows),
        target_name=target_name,
        prediction=prediction,
        request=request,
    )


@dataclass(frozen=True, slots=True)
class NativeLaneConfig:
    model_digest: str
    schema_digest: str
    capacity: int
    host_width: int
    maximum_optimizer_lag: int
    num_layers: int | None = None
    addressed_architecture_identity: str | None = None
    episode_address_codebook_sha256: str | None = None
    paired_source_width: int | None = None
    paired_architecture_identity: str | None = None
    paired_source_dtype: torch.dtype = torch.float32
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if not self.model_digest or not self.schema_digest:
            raise ValueError("lane model and schema digests must be non-empty")
        integers = (self.capacity, self.host_width, self.maximum_optimizer_lag)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("lane dimensions and optimizer lag must be integers")
        if min(self.capacity, self.host_width) <= 0 or self.maximum_optimizer_lag < 0:
            raise ValueError("lane dimensions must be positive and lag non-negative")
        if self.num_layers is not None and (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers <= 0
        ):
            raise ValueError("layerwise lane depth must be a positive integer")
        address_fields_present = (
            self.addressed_architecture_identity is not None,
            self.episode_address_codebook_sha256 is not None,
        )
        if any(address_fields_present) and not all(address_fields_present):
            raise ValueError(
                "addressed lane architecture identity and codebook receipt "
                "must be declared together"
            )
        if all(address_fields_present):
            if self.num_layers is None:
                raise ValueError("addressed lanes require layerwise posterior state")
            if (
                not isinstance(self.addressed_architecture_identity, str)
                or not self.addressed_architecture_identity
            ):
                raise ValueError("addressed lane architecture identity must be non-empty")
            if (
                not isinstance(self.episode_address_codebook_sha256, str)
                or len(self.episode_address_codebook_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.episode_address_codebook_sha256
                )
            ):
                raise ValueError("addressed lane codebook receipt must be lowercase SHA-256")
        paired_fields_present = (
            self.paired_source_width is not None,
            self.paired_architecture_identity is not None,
        )
        if any(paired_fields_present) and not all(paired_fields_present):
            raise ValueError(
                "paired lane source width and architecture identity must be declared together"
            )
        if all(paired_fields_present):
            if self.addressed:
                raise ValueError("paired source lanes cannot also use an episode permutation")
            if self.num_layers is None:
                raise ValueError("paired source lanes require layerwise host posterior state")
            if (
                isinstance(self.paired_source_width, bool)
                or not isinstance(self.paired_source_width, int)
                or self.paired_source_width <= 0
            ):
                raise ValueError("paired source width must be a positive integer")
            if (
                not isinstance(self.paired_architecture_identity, str)
                or not self.paired_architecture_identity
            ):
                raise ValueError("paired source architecture identity must be non-empty")
        if self.paired_source_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("paired source lane dtype must be float16, bfloat16 or float32")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("lane state dtype must be float16, bfloat16 or float32")

    @property
    def addressed(self) -> bool:
        return self.addressed_architecture_identity is not None

    @property
    def paired(self) -> bool:
        return self.paired_architecture_identity is not None

    @property
    def contract_digest(self) -> str:
        payload: dict[str, object] = {
            "capacity": self.capacity,
            "dtype": str(self.dtype),
            "host_width": self.host_width,
            "maximum_optimizer_lag": self.maximum_optimizer_lag,
            "model_digest": self.model_digest,
            "schema_digest": self.schema_digest,
            "version": 2,
        }
        if self.num_layers is not None:
            payload["num_layers"] = self.num_layers
            payload["version"] = 3
        if self.addressed:
            payload["addressed_architecture_identity"] = self.addressed_architecture_identity
            payload["episode_address_codebook_sha256"] = (
                self.episode_address_codebook_sha256
            )
            payload["version"] = 4
        if self.paired:
            payload["paired_architecture_identity"] = self.paired_architecture_identity
            payload["paired_source_dtype"] = str(self.paired_source_dtype)
            payload["paired_source_width"] = self.paired_source_width
            payload["version"] = 5
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeLaneStamp:
    episode_key: str
    frame_index: int
    state_age: int
    producer_optimizer_step: int
    source_weight_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.episode_key, str) or not self.episode_key:
            raise ValueError("lane episode key must be non-empty")
        values = (
            self.frame_index,
            self.state_age,
            self.producer_optimizer_step,
            self.source_weight_version,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("lane stamp counters must be non-negative integers")


@dataclass(frozen=True, slots=True)
class NativeLaneRead:
    state: NativePersistentState
    stamp: NativeLaneStamp
    optimizer_lag: int
    row_bindings: RowBindings = ()


@dataclass(frozen=True, slots=True)
class NativeLaneTransaction:
    token: int
    lane_id: int
    state: NativePersistentState
    stamp: NativeLaneStamp
    reset: bool
    row_bindings: RowBindings = ()


@dataclass(frozen=True, slots=True)
class _NativeLaneRecord:
    state: NativePersistentState
    stamp: NativeLaneStamp
    row_bindings: RowBindings


class NativeTrainingLaneBank:
    """Atomic detached state lanes that advance only after optimizer success."""

    def __init__(self, config: NativeLaneConfig) -> None:
        self.config = config
        self._records: dict[int, _NativeLaneRecord] = {}
        # One committed predecessor is retained only for causal diagnostics.
        # Its posterior and labels never participate in a model forward.
        self._history: dict[int, _NativeLaneRecord] = {}
        self._pending: dict[int, NativeLaneTransaction] = {}
        self._pending_lanes: set[int] = set()
        self._next_token = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _lane_id(lane_id: int) -> None:
        if isinstance(lane_id, bool) or not isinstance(lane_id, int) or lane_id < 0:
            raise ValueError("lane ID must be a non-negative integer")

    def _validate_state(self, state: NativePersistentState) -> None:
        expected_type = (
            NativePosteriorState
            if self.config.num_layers is None
            else NativeLayerwisePosteriorState
        )
        tensor = persistent_state_tensor(state)
        addressed_state = isinstance(state, AddressedLayerwisePosteriorState)
        if (
            not isinstance(state, expected_type)
            or state.batch_size != 1
            or state.capacity != self.config.capacity
            or state.host_width != self.config.host_width
            or tensor.device != torch.device(self.config.device)
            or tensor.dtype != self.config.dtype
            or (
                isinstance(state, NativeLayerwisePosteriorState)
                and state.num_layers != self.config.num_layers
            )
        ):
            raise ValueError("lane posterior differs from the frozen lane contract")
        if self.config.addressed:
            if not isinstance(state, AddressedLayerwisePosteriorState):
                raise ValueError("addressed lane requires addressed posterior state")
            if (
                state.architecture_identity
                != self.config.addressed_architecture_identity
                or state.episode_address_state.codebook_sha256
                != self.config.episode_address_codebook_sha256
            ):
                raise ValueError("addressed lane posterior differs from its routing contract")
        elif addressed_state:
            raise ValueError("historical lane cannot accept addressed posterior state")
        paired_state = isinstance(state, NativeVidEoMTPairedPosteriorState)
        if self.config.paired:
            if not isinstance(state, NativeVidEoMTPairedPosteriorState):
                raise ValueError("paired source lane requires an atomic paired posterior state")
            if (
                state.source_width != self.config.paired_source_width
                or state.source_queries.dtype != self.config.paired_source_dtype
                or state.architecture_identity != self.config.paired_architecture_identity
            ):
                raise ValueError("paired posterior differs from its source lane contract")
        elif paired_state:
            raise ValueError("historical lane cannot accept a paired source posterior")

    def read(
        self,
        lane_id: int,
        *,
        episode_key: str,
        next_frame_index: int,
        optimizer_step: int,
        source_weight_version: int,
    ) -> NativeLaneRead | None:
        self._lane_id(lane_id)
        counters = (next_frame_index, optimizer_step, source_weight_version)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters
        ):
            raise ValueError("lane read counters must be non-negative integers")
        with self._lock:
            record = self._records.get(lane_id)
            if record is None:
                return None
            stamp = record.stamp
            if stamp.episode_key != episode_key:
                raise NativeLaneError("cached lane belongs to another episode")
            if next_frame_index != stamp.frame_index + 1:
                raise NativeLaneError("cached lane is not contiguous with the requested frame")
            if stamp.source_weight_version != source_weight_version:
                raise NativeLaneError("lane source-mixture version changed without a reset")
            lag = optimizer_step - stamp.producer_optimizer_step
            if lag < 0:
                raise NativeLaneError("cached lane was produced by a future optimizer step")
            if lag > self.config.maximum_optimizer_lag:
                raise NativeLaneError("cached lane exceeds the hard optimizer staleness limit")
            return NativeLaneRead(
                state=clone_persistent_state(record.state),
                stamp=stamp,
                optimizer_lag=lag,
                row_bindings=record.row_bindings,
            )

    def read_predecessor(
        self,
        lane_id: int,
        *,
        episode_key: str,
        next_frame_index: int,
        source_weight_version: int,
    ) -> NativePersistentState | None:
        """Read exact ``H_(t-2)`` for a diagnostic at input frame ``t``.

        The predecessor is a checkpointed observation of real state lineage,
        not a reconstructed or learned lifecycle signal. Absence is expected
        during reset and the first continuation.
        """

        self._lane_id(lane_id)
        counters = (next_frame_index, source_weight_version)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters
        ):
            raise ValueError("lane predecessor counters must be non-negative integers")
        with self._lock:
            current = self._records.get(lane_id)
            predecessor = self._history.get(lane_id)
            if current is None:
                return None
            if (
                current.stamp.episode_key != episode_key
                or current.stamp.frame_index != next_frame_index - 1
                or current.stamp.source_weight_version != source_weight_version
            ):
                raise NativeLaneError("current lane is incompatible with predecessor lookup")
            if predecessor is None:
                return None
            stamp = predecessor.stamp
            if (
                stamp.episode_key != episode_key
                or stamp.frame_index != next_frame_index - 2
                or stamp.source_weight_version != source_weight_version
            ):
                raise NativeLaneError("checkpointed lane predecessor is not exactly one age older")
            return clone_persistent_state(predecessor.state)

    def stage(
        self,
        lane_id: int,
        state: NativePersistentState,
        stamp: NativeLaneStamp,
        *,
        reset: bool,
        row_bindings: RowBindings = (),
    ) -> NativeLaneTransaction:
        self._lane_id(lane_id)
        if not isinstance(reset, bool):
            raise TypeError("lane reset flag must be boolean")
        self._validate_state(state)
        bindings = normalize_row_bindings(
            row_bindings,
            capacity=self.config.capacity,
        )
        with self._lock:
            if lane_id in self._pending_lanes:
                raise NativeLaneError("the lane already has a pending optimizer transaction")
            previous = self._records.get(lane_id)
            if previous is None and not reset:
                raise NativeLaneError("a new lane must begin at an explicit reset")
            if reset:
                if stamp.state_age != 0 or stamp.frame_index != 0:
                    raise NativeLaneError("a reset lane must start at frame and age zero")
                if previous is not None and stamp.episode_key == previous.stamp.episode_key:
                    raise NativeLaneError("an explicit reset must begin a new episode identity")
            else:
                if previous is None:
                    raise NativeLaneError("a continuing lane has no previous record")
                old = previous.stamp
                if stamp.episode_key != old.episode_key:
                    raise NativeLaneError("episode changed without an explicit lane reset")
                if stamp.frame_index != old.frame_index + 1:
                    raise NativeLaneError("lane frames must advance by exactly one")
                if stamp.state_age != old.state_age + 1:
                    raise NativeLaneError("lane state age must advance by exactly one")
                if stamp.source_weight_version != old.source_weight_version:
                    raise NativeLaneError("source-mixture version changed without a reset")
                if stamp.producer_optimizer_step < old.producer_optimizer_step:
                    raise NativeLaneError("lane producer optimizer step moved backwards")
                old_bindings = row_binding_map(
                    previous.row_bindings,
                    capacity=self.config.capacity,
                )
                new_bindings = row_binding_map(
                    bindings,
                    capacity=self.config.capacity,
                )
                if any(new_bindings.get(identity) != row for identity, row in old_bindings.items()):
                    raise NativeLaneError(
                        "a continuing lane removed or rebound a supervised identity"
                    )
            token = self._next_token
            self._next_token += 1
            transaction = NativeLaneTransaction(
                token=token,
                lane_id=lane_id,
                state=clone_persistent_state(state),
                stamp=stamp,
                reset=reset,
                row_bindings=bindings,
            )
            self._pending[token] = transaction
            self._pending_lanes.add(lane_id)
            return transaction

    def commit_after_optimizer(
        self,
        transaction: NativeLaneTransaction,
        *,
        successful_optimizer_step: int,
    ) -> None:
        self.commit_batch_after_optimizer(
            (transaction,),
            successful_optimizer_step=successful_optimizer_step,
        )

    def commit_batch_after_optimizer(
        self,
        transactions: tuple[NativeLaneTransaction, ...],
        *,
        successful_optimizer_step: int,
    ) -> None:
        """Publish every lane from one optimizer attempt or publish none."""

        if not transactions:
            raise NativeLaneError("an optimizer publication batch cannot be empty")
        if (
            isinstance(successful_optimizer_step, bool)
            or not isinstance(successful_optimizer_step, int)
            or successful_optimizer_step < 1
        ):
            raise NativeLaneError("successful optimizer step must be a positive integer")
        if len({item.token for item in transactions}) != len(transactions) or len(
            {item.lane_id for item in transactions}
        ) != len(transactions):
            raise NativeLaneError("an optimizer publication batch contains duplicates")
        if any(
            successful_optimizer_step != item.stamp.producer_optimizer_step + 1
            for item in transactions
        ):
            raise NativeLaneError(
                "lane commit requires the immediately successful optimizer transaction"
            )
        with self._lock:
            if any(self._pending.get(item.token) is not item for item in transactions):
                raise NativeLaneError("lane transaction is unknown, aborted or already committed")
            if any(item.lane_id not in self._pending_lanes for item in transactions):
                raise NativeLaneError("lane pending indexes are internally inconsistent")
            staged_records = dict(self._records)
            staged_history = dict(self._history)
            staged_pending = dict(self._pending)
            staged_pending_lanes = set(self._pending_lanes)
            for item in transactions:
                previous = staged_records.get(item.lane_id)
                if item.reset:
                    staged_history.pop(item.lane_id, None)
                else:
                    if previous is None:
                        raise NativeLaneError("a continuing lane lost its committed predecessor")
                    staged_history[item.lane_id] = previous
                staged_records[item.lane_id] = _NativeLaneRecord(
                    item.state,
                    item.stamp,
                    item.row_bindings,
                )
                del staged_pending[item.token]
                staged_pending_lanes.remove(item.lane_id)
            self._records = staged_records
            self._history = staged_history
            self._pending = staged_pending
            self._pending_lanes = staged_pending_lanes

    def abort(self, transaction: NativeLaneTransaction) -> None:
        self.abort_batch((transaction,))

    def abort_batch(self, transactions: tuple[NativeLaneTransaction, ...]) -> None:
        """Abort a complete optimizer attempt without mutating published lanes."""

        if not transactions:
            raise NativeLaneError("an optimizer abort batch cannot be empty")
        if len({item.token for item in transactions}) != len(transactions):
            raise NativeLaneError("an optimizer abort batch contains duplicate transactions")
        with self._lock:
            if any(self._pending.get(item.token) is not item for item in transactions):
                raise NativeLaneError("lane transaction is unknown, aborted or already committed")
            if any(item.lane_id not in self._pending_lanes for item in transactions):
                raise NativeLaneError("lane pending indexes are internally inconsistent")
            staged_pending = dict(self._pending)
            staged_pending_lanes = set(self._pending_lanes)
            for item in transactions:
                del staged_pending[item.token]
                staged_pending_lanes.remove(item.lane_id)
            self._pending = staged_pending
            self._pending_lanes = staged_pending_lanes

    @staticmethod
    def _serialized_records(records: Mapping[int, _NativeLaneRecord]) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for lane_id in sorted(records):
            record = records[lane_id]
            serialized.append(
                {
                    "lane_id": lane_id,
                    "row_bindings": [list(item) for item in record.row_bindings],
                    "stamp": {
                        "episode_key": record.stamp.episode_key,
                        "frame_index": record.stamp.frame_index,
                        "producer_optimizer_step": record.stamp.producer_optimizer_step,
                        "source_weight_version": record.stamp.source_weight_version,
                        "state_age": record.stamp.state_age,
                    },
                    "state": base64.b64encode(record.state.serialize()).decode("ascii"),
                }
            )
        return serialized

    def serialize(self) -> bytes:
        with self._lock:
            if self._pending:
                raise NativeLaneError("cannot snapshot while optimizer transactions are pending")
            payload = {
                "contract_digest": self.config.contract_digest,
                "history": self._serialized_records(self._history),
                "records": self._serialized_records(self._records),
                "version": (
                    7
                    if self.config.paired
                    else 6
                    if self.config.addressed
                    else 4
                    if self.config.num_layers is None
                    else 5
                ),
            }
            return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def deserialize(cls, config: NativeLaneConfig, encoded: bytes) -> NativeTrainingLaneBank:
        if not isinstance(encoded, bytes):
            raise TypeError("lane snapshot must be bytes")
        try:
            payload = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError("lane snapshot is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("lane snapshot has an incompatible top-level schema")
        version = payload.get("version")
        expected_fields = (
            {"contract_digest", "records", "version"}
            if version == 1
            else {"contract_digest", "history", "records", "version"}
        )
        if set(payload) != expected_fields:
            raise ValueError("lane snapshot has an incompatible top-level schema")
        if (
            version not in (1, 2, 3, 4, 5, 6, 7)
            or (config.paired and version != 7)
            or (not config.paired and version == 7)
            or (config.addressed and version != 6)
            or (
                not config.addressed
                and not config.paired
                and config.num_layers is None
                and version in (5, 6)
            )
            or (
                not config.addressed
                and not config.paired
                and config.num_layers is not None
                and version != 5
            )
            or payload["contract_digest"] != config.contract_digest
        ):
            raise ValueError("lane snapshot differs from the frozen lane contract")
        bank = cls(config)

        def restore_records(value: object, *, name: str) -> dict[int, _NativeLaneRecord]:
            if not isinstance(value, list):
                raise ValueError(f"lane snapshot {name} must be a list")
            restored: dict[int, _NativeLaneRecord] = {}
            for item in value:
                expected_item_fields = (
                    {"lane_id", "stamp", "state", "row_bindings"}
                    if version in (4, 5, 6, 7)
                    else {"lane_id", "stamp", "state"}
                )
                if not isinstance(item, dict) or set(item) != expected_item_fields:
                    raise ValueError("lane snapshot record has an incompatible schema")
                lane_id = item["lane_id"]
                bank._lane_id(lane_id)
                if lane_id in restored:
                    raise ValueError(f"lane snapshot {name} contains a duplicate lane")
                legacy_stamp_fields = {
                    "episode_key",
                    "frame_index",
                    "last_reconstruction_frame",
                    "producer_optimizer_step",
                    "source_weight_version",
                    "state_age",
                }
                current_stamp_fields = legacy_stamp_fields - {"last_reconstruction_frame"}
                expected_stamp_fields = (
                    legacy_stamp_fields if version in (1, 2) else current_stamp_fields
                )
                if (
                    not isinstance(item["stamp"], dict)
                    or set(item["stamp"]) != expected_stamp_fields
                ):
                    raise ValueError("lane snapshot stamp has an incompatible schema")
                if not isinstance(item["state"], str):
                    raise ValueError("lane snapshot state must be base64 text")
                try:
                    state_bytes = base64.b64decode(item["state"], validate=True)
                except (binascii.Error, ValueError) as error:
                    raise ValueError("lane snapshot state is not valid base64") from error
                if config.paired:
                    state = NativeVidEoMTPairedPosteriorState.deserialize(
                        state_bytes,
                        device=config.device,
                    )
                elif config.addressed:
                    state = AddressedLayerwisePosteriorState.deserialize(
                        state_bytes,
                        device=config.device,
                    )
                elif config.num_layers is None:
                    state = NativePosteriorState.deserialize(state_bytes, device=config.device)
                else:
                    state = NativeLayerwisePosteriorState.deserialize(
                        state_bytes,
                        device=config.device,
                    )
                bank._validate_state(state)
                stamp_payload = dict(item["stamp"])
                stamp_payload.pop("last_reconstruction_frame", None)
                restored[lane_id] = _NativeLaneRecord(
                    state=state,
                    stamp=NativeLaneStamp(**stamp_payload),
                    row_bindings=(
                        ()
                        if version not in (4, 5, 6, 7)
                        else normalize_row_bindings(
                            tuple(tuple(pair) for pair in item["row_bindings"]),
                            capacity=config.capacity,
                        )
                    ),
                )
            return restored

        bank._records = restore_records(payload["records"], name="records")
        bank._history = {} if version == 1 else restore_records(payload["history"], name="history")
        for lane_id, predecessor in bank._history.items():
            current = bank._records.get(lane_id)
            if current is None:
                raise ValueError("lane snapshot history has no current record")
            current_stamp = current.stamp
            predecessor_stamp = predecessor.stamp
            if (
                predecessor_stamp.episode_key != current_stamp.episode_key
                or predecessor_stamp.frame_index + 1 != current_stamp.frame_index
                or predecessor_stamp.state_age + 1 != current_stamp.state_age
                or predecessor_stamp.source_weight_version != current_stamp.source_weight_version
                or predecessor_stamp.producer_optimizer_step > current_stamp.producer_optimizer_step
            ):
                raise ValueError("lane snapshot history is not an exact committed predecessor")
        return bank

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.serialize()).hexdigest()
