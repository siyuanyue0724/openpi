"""Checkpointable rank-local state for episodic posterior streaming.

This is training orchestration, not a second model memory. It preserves the
detached :class:`ObjectBeliefBatch` that the deploy-time filter itself emits and
enforces episode order and bounded parameter-version staleness across stateful
training chunks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from picf_next.models.temporal import (
    ObjectBeliefBatch,
    TemporalFilterConfig,
    empty_object_belief,
)
from picf_next.training.control import FrozenEpisodeStreamPlan, PlannedStreamTransition

_STREAM_STATE_SCHEMA = "picf-next.posterior-stream-state.v5"
_STREAM_STATE_GROUP_SCHEMA = "picf-next.posterior-stream-state-group.v3"
_STATE_FACTORIZATION = (
    "address-unit-sphere_content-dirac_geometry-diag-gaussian_lifecycle-semi-markov-bernoulli.v2"
)
_FLOAT_BELIEF_FIELDS = (
    "address_mean",
    "content_mean",
    "geometry_mean",
    "geometry_covariance_diag",
    "existence_logits",
    "visibility_given_existence_logits",
    "measurement_age_s",
)
_BELIEF_FIELDS = (
    *_FLOAT_BELIEF_FIELDS,
    "valid",
    "age",
)
_STATE_FIELDS = frozenset(
    {
        "belief",
        "capacity",
        "dimensions",
        "episode_keys",
        "factorization",
        "lane_ids",
        "loss_track_keys_by_row",
        "max_parameter_lag",
        "next_transition_indices",
        "schema",
        "state_parameter_versions",
    }
)


def _unit_norm_tolerance(dtype: torch.dtype) -> float:
    """Return a validation tolerance that is representable by ``dtype``."""

    return max(1e-5, torch.finfo(dtype).eps)


def _detach_clone_belief(belief: ObjectBeliefBatch) -> ObjectBeliefBatch:
    return ObjectBeliefBatch(
        **{field: getattr(belief, field).detach().clone() for field in _BELIEF_FIELDS}
    )


class PosteriorStreamState:
    """Carry one detached PICF posterior per deterministic local stream lane."""

    def __init__(
        self,
        config: TemporalFilterConfig,
        *,
        lane_ids: Sequence[str],
        capacity: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_parameter_lag: int = 1,
    ) -> None:
        frozen_lane_ids = tuple(lane_ids)
        if not frozen_lane_ids or any(
            not isinstance(lane_id, str) or not lane_id for lane_id in frozen_lane_ids
        ):
            raise ValueError("lane_ids must be non-empty strings")
        if len(set(frozen_lane_ids)) != len(frozen_lane_ids):
            raise ValueError("lane_ids must be unique")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if (
            not isinstance(max_parameter_lag, int)
            or isinstance(max_parameter_lag, bool)
            or max_parameter_lag < 0
        ):
            raise ValueError("max_parameter_lag must be a non-negative integer")
        self.config = config
        self.lane_ids = frozen_lane_ids
        self.capacity = capacity
        self.max_parameter_lag = max_parameter_lag
        self.belief = empty_object_belief(
            config,
            batch_size=len(frozen_lane_ids),
            capacity=capacity,
            device=device,
            dtype=dtype,
        )
        self.episode_keys: tuple[str | None, ...] = (None,) * len(frozen_lane_ids)
        self.next_transition_indices: tuple[int, ...] = (0,) * len(frozen_lane_ids)
        self.state_parameter_versions: tuple[int, ...] = (-1,) * len(frozen_lane_ids)
        self.loss_track_keys_by_row: tuple[tuple[str | None, ...], ...] = tuple(
            (None,) * capacity for _lane in frozen_lane_ids
        )
        self._pending: (
            tuple[
                tuple[str, ...],
                tuple[int, ...],
                int,
                tuple[tuple[str | None, ...], ...],
            ]
            | None
        ) = None

    @property
    def batch_size(self) -> int:
        return len(self.lane_ids)

    @property
    def has_pending_chunk(self) -> bool:
        return self._pending is not None

    @property
    def pending_loss_track_keys_by_row(self) -> tuple[tuple[str | None, ...], ...]:
        """Return reset-aware loss-only keys for the currently prepared chunk."""

        if self._pending is None:
            raise RuntimeError("prepare_chunk must run before reading pending loss tracks")
        return self._pending[3]

    def _validate_chunk_coordinates(
        self,
        episode_keys: Sequence[str],
        start_transition_indices: Sequence[int],
        current_parameter_version: int,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        keys = tuple(episode_keys)
        starts = tuple(start_transition_indices)
        if len(keys) != self.batch_size or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("episode_keys must match stream batch size and be non-empty")
        if len(starts) != self.batch_size or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in starts
        ):
            raise ValueError("start_transition_indices must be non-negative integers")
        if (
            not isinstance(current_parameter_version, int)
            or isinstance(current_parameter_version, bool)
            or current_parameter_version < 0
        ):
            raise ValueError("current_parameter_version must be a non-negative integer")
        return keys, starts

    def prepare_chunk(
        self,
        *,
        episode_keys: Sequence[str],
        start_transition_indices: Sequence[int],
        current_parameter_version: int,
    ) -> ObjectBeliefBatch:
        """Validate ordered continuity and return the detached chunk initial state."""

        if self._pending is not None:
            raise RuntimeError("a stream chunk is already pending commit or abort")
        keys, starts = self._validate_chunk_coordinates(
            episode_keys,
            start_transition_indices,
            current_parameter_version,
        )
        reset_lanes: list[int] = []
        for lane, (key, start) in enumerate(zip(keys, starts, strict=True)):
            stored_key = self.episode_keys[lane]
            if key != stored_key:
                if start != 0:
                    raise ValueError("a new episode must begin at transition zero")
                reset_lanes.append(lane)
                continue
            if start != self.next_transition_indices[lane]:
                raise ValueError("stream chunk is discontinuous with the stored episode cursor")
            state_version = self.state_parameter_versions[lane]
            lag = current_parameter_version - state_version
            if lag < 0:
                raise ValueError("posterior state was produced by a future parameter version")
            if lag > self.max_parameter_lag:
                raise ValueError("posterior state exceeds the configured parameter-version lag")

        initial = _detach_clone_belief(self.belief)
        if reset_lanes:
            lane_index = torch.as_tensor(
                reset_lanes,
                device=initial.valid.device,
                dtype=torch.long,
            )
            for field in _BELIEF_FIELDS:
                getattr(initial, field)[lane_index] = 0
        prepared_tracks = list(self.loss_track_keys_by_row)
        for lane in reset_lanes:
            prepared_tracks[lane] = (None,) * self.capacity
        self._pending = (keys, starts, current_parameter_version, tuple(prepared_tracks))
        return initial

    def prepare_planned_transitions(
        self,
        transitions: Sequence[PlannedStreamTransition],
        *,
        current_parameter_version: int,
    ) -> ObjectBeliefBatch:
        """Prepare one single-transition stream microbatch without coordinate glue."""

        planned = tuple(transitions)
        if len(planned) != self.batch_size or any(
            not isinstance(transition, PlannedStreamTransition) for transition in planned
        ):
            raise ValueError("planned transitions must match the stream batch size")
        if tuple(transition.lane_id for transition in planned) != self.lane_ids:
            raise ValueError("planned transition lanes differ from the stream state lanes")
        return self.prepare_chunk(
            episode_keys=tuple(transition.episode_instance_id for transition in planned),
            start_transition_indices=tuple(transition.transition_index for transition in planned),
            current_parameter_version=current_parameter_version,
        )

    def abort_chunk(self) -> None:
        if self._pending is None:
            raise RuntimeError("no stream chunk is pending")
        self._pending = None

    def commit_chunk(
        self,
        final_belief: ObjectBeliefBatch,
        *,
        transition_count: int,
        state_parameter_version: int,
        final_loss_track_keys_by_row: Sequence[Sequence[str | None]] | None = None,
    ) -> None:
        """Commit the state produced by one successful contiguous training chunk."""

        payload = self._validated_commit_payload(
            final_belief,
            transition_count=transition_count,
            state_parameter_version=state_parameter_version,
            final_loss_track_keys_by_row=final_loss_track_keys_by_row,
        )
        self._apply_commit_payload(payload)

    def _validated_commit_payload(
        self,
        final_belief: ObjectBeliefBatch,
        *,
        transition_count: int,
        state_parameter_version: int,
        final_loss_track_keys_by_row: Sequence[Sequence[str | None]] | None = None,
    ) -> tuple[
        ObjectBeliefBatch,
        tuple[str, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[str | None, ...], ...],
    ]:
        """Validate and materialize a commit without mutating the live cursor."""

        if self._pending is None:
            raise RuntimeError("prepare_chunk must run before commit_chunk")
        if (
            not isinstance(transition_count, int)
            or isinstance(transition_count, bool)
            or transition_count <= 0
        ):
            raise ValueError("transition_count must be a positive integer")
        keys, starts, prepared_version, prepared_tracks = self._pending
        if (
            not isinstance(state_parameter_version, int)
            or isinstance(state_parameter_version, bool)
            or state_parameter_version < 0
        ):
            raise ValueError("state_parameter_version must be a non-negative integer")
        if state_parameter_version != prepared_version:
            raise ValueError("state_parameter_version must equal the prepared model version")
        self._validate_belief(final_belief, validate_values=False)
        final_tracks = self._validate_loss_track_keys(
            prepared_tracks
            if final_loss_track_keys_by_row is None
            else final_loss_track_keys_by_row,
            final_belief.valid,
        )
        return (
            _detach_clone_belief(final_belief),
            keys,
            tuple(index + transition_count for index in starts),
            (state_parameter_version,) * self.batch_size,
            final_tracks,
        )

    def _apply_commit_payload(
        self,
        payload: tuple[
            ObjectBeliefBatch,
            tuple[str, ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[str | None, ...], ...],
        ],
    ) -> None:
        (
            self.belief,
            self.episode_keys,
            self.next_transition_indices,
            self.state_parameter_versions,
            self.loss_track_keys_by_row,
        ) = payload
        self._pending = None

    def _validate_loss_track_keys(
        self,
        keys_by_row: Sequence[Sequence[str | None]],
        belief_valid: torch.Tensor,
    ) -> tuple[tuple[str | None, ...], ...]:
        lanes = tuple(tuple(keys) for keys in keys_by_row)
        if len(lanes) != self.batch_size or any(len(keys) != self.capacity for keys in lanes):
            raise ValueError("loss track keys must be lane-by-posterior-row")
        for lane, keys in enumerate(lanes):
            present = [key for key in keys if key is not None]
            if any(not isinstance(key, str) or not key for key in present):
                raise ValueError("loss track keys must be nonempty strings or None")
            if len(set(present)) != len(present):
                raise ValueError("loss track keys must be unique within each lane")
            valid_rows = belief_valid[lane].detach().cpu().tolist()
            if any(
                key is not None and not valid for key, valid in zip(keys, valid_rows, strict=True)
            ):
                raise ValueError("loss track keys cannot name unused posterior rows")
        return lanes

    def candidate_value_validity(self, belief: ObjectBeliefBatch) -> torch.Tensor:
        """Return a device scalar for hot-path finite/domain validation.

        Shape, dtype and device mismatches fail immediately. Value checks remain
        tensor operations so the caller can fold this scalar into its existing
        distributed finite-loss agreement without adding a host synchronization.
        """

        self._validate_belief(belief, validate_values=False)
        valid = belief.valid
        valid_feature = valid.unsqueeze(-1)
        address_norm = torch.linalg.vector_norm(belief.address_mean.float(), dim=-1)
        address_tolerance = _unit_norm_tolerance(belief.address_mean.dtype)
        checks = [
            *(torch.isfinite(getattr(belief, field)).all() for field in _FLOAT_BELIEF_FIELDS),
            torch.where(
                valid_feature,
                belief.geometry_covariance_diag >= self.config.minimum_variance,
                belief.geometry_covariance_diag == 0,
            ).all(),
            torch.where(
                valid,
                torch.isclose(
                    address_norm,
                    torch.ones_like(address_norm),
                    atol=address_tolerance,
                    rtol=address_tolerance,
                ),
                torch.ones_like(valid),
            ).all(),
            (belief.age >= 0).all(),
            (belief.measurement_age_s >= 0.0).all(),
            torch.where(valid, torch.zeros_like(belief.age), belief.age).eq(0).all(),
            torch.where(
                valid,
                torch.zeros_like(belief.measurement_age_s),
                belief.measurement_age_s,
            )
            .eq(0.0)
            .all(),
        ]
        checks.extend(
            torch.where(valid_feature, torch.zeros_like(tensor), tensor).eq(0).all()
            for tensor in (
                belief.address_mean,
                belief.content_mean,
                belief.geometry_mean,
            )
        )
        checks.extend(
            torch.where(valid, torch.zeros_like(tensor), tensor).eq(0).all()
            for tensor in (
                belief.existence_logits,
                belief.visibility_given_existence_logits,
            )
        )
        return torch.stack(checks).all()

    def _validate_belief(
        self,
        belief: ObjectBeliefBatch,
        *,
        validate_values: bool,
    ) -> None:
        expected_shapes = {
            "address_mean": (self.batch_size, self.capacity, self.config.address_dim),
            "content_mean": (self.batch_size, self.capacity, self.config.content_dim),
            "geometry_mean": (self.batch_size, self.capacity, self.config.geometry_dim),
            "geometry_covariance_diag": (
                self.batch_size,
                self.capacity,
                self.config.geometry_dim,
            ),
            "existence_logits": (self.batch_size, self.capacity),
            "visibility_given_existence_logits": (self.batch_size, self.capacity),
            "measurement_age_s": (self.batch_size, self.capacity),
            "valid": (self.batch_size, self.capacity),
            "age": (self.batch_size, self.capacity),
        }
        reference = self.belief.address_mean
        for field, shape in expected_shapes.items():
            tensor = getattr(belief, field)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"stream belief field {field} has an incompatible shape")
            if tensor.device != reference.device:
                raise ValueError("stream belief fields must remain on the configured device")
        for field in _FLOAT_BELIEF_FIELDS:
            tensor = getattr(belief, field)
            if tensor.dtype != reference.dtype:
                raise ValueError("stream floating belief fields must share configured dtype")
        if belief.valid.dtype != torch.bool or belief.age.dtype != torch.long:
            raise ValueError("stream valid/age fields require bool/int64 dtypes")
        if validate_values:
            if any(
                not torch.isfinite(getattr(belief, field)).all() for field in _FLOAT_BELIEF_FIELDS
            ):
                raise ValueError("stream floating belief fields must be finite")
            valid = belief.valid
            if (belief.geometry_covariance_diag[valid] < self.config.minimum_variance).any() or (
                belief.age < 0
            ).any():
                raise ValueError("valid stream covariance and age violate their domains")
            if (belief.measurement_age_s < 0.0).any():
                raise ValueError("stream measurement age must be nonnegative")
            if (belief.geometry_covariance_diag[~valid] != 0).any():
                raise ValueError("invalid stream covariance rows must be exactly zero")
            if valid.any():
                address_norm = torch.linalg.vector_norm(belief.address_mean[valid].float(), dim=-1)
                address_tolerance = _unit_norm_tolerance(belief.address_mean.dtype)
                if not torch.allclose(
                    address_norm,
                    torch.ones_like(address_norm),
                    atol=address_tolerance,
                    rtol=address_tolerance,
                ):
                    raise ValueError("valid stream addresses must have unit norm")
            for field in (
                "address_mean",
                "content_mean",
                "geometry_mean",
                "existence_logits",
                "visibility_given_existence_logits",
            ):
                if (getattr(belief, field)[~valid] != 0).any():
                    raise ValueError("invalid stream belief rows must be exactly zero")
            if (belief.age[~valid] != 0).any():
                raise ValueError("invalid stream belief age must be exactly zero")
            if (belief.measurement_age_s[~valid] != 0.0).any():
                raise ValueError("invalid stream measurement age must be exactly zero")

    def state_dict(self) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("cannot checkpoint a stream state with an uncommitted chunk")
        # Full value scans synchronize accelerators, so they belong at the
        # infrequent checkpoint boundary rather than every transition commit.
        self._validate_belief(self.belief, validate_values=True)
        belief = {
            field: getattr(self.belief, field).detach().cpu().clone() for field in _BELIEF_FIELDS
        }
        return {
            "belief": belief,
            "capacity": self.capacity,
            "dimensions": {
                "address": self.config.address_dim,
                "content": self.config.content_dim,
                "geometry": self.config.geometry_dim,
            },
            "episode_keys": list(self.episode_keys),
            "factorization": _STATE_FACTORIZATION,
            "lane_ids": list(self.lane_ids),
            "loss_track_keys_by_row": [list(keys) for keys in self.loss_track_keys_by_row],
            "max_parameter_lag": self.max_parameter_lag,
            "next_transition_indices": list(self.next_transition_indices),
            "schema": _STREAM_STATE_SCHEMA,
            "state_parameter_versions": list(self.state_parameter_versions),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._pending is not None:
            raise RuntimeError("cannot load while a stream chunk is pending")
        if not isinstance(state, Mapping):
            raise ValueError("posterior stream state fields are malformed")
        if state.get("schema") != _STREAM_STATE_SCHEMA:
            raise ValueError("unsupported posterior stream state schema")
        if set(state) != _STATE_FIELDS:
            raise ValueError("posterior stream state fields are malformed")
        if state.get("factorization") != _STATE_FACTORIZATION:
            raise ValueError("posterior stream state factorization differs from the active model")
        expected_dimensions = {
            "address": self.config.address_dim,
            "content": self.config.content_dim,
            "geometry": self.config.geometry_dim,
        }
        if state.get("dimensions") != expected_dimensions or state.get("capacity") != self.capacity:
            raise ValueError("posterior stream state dimensions differ from the active model")
        if tuple(state.get("lane_ids", ())) != self.lane_ids:
            raise ValueError("posterior stream lane identities differ from the active plan")
        if state.get("max_parameter_lag") != self.max_parameter_lag:
            raise ValueError("posterior stream parameter-lag contract differs")

        keys = tuple(state.get("episode_keys", ()))
        starts = tuple(state.get("next_transition_indices", ()))
        versions = tuple(state.get("state_parameter_versions", ()))
        raw_tracks = state.get("loss_track_keys_by_row", ())
        if len(keys) != self.batch_size or any(
            key is not None and (not isinstance(key, str) or not key) for key in keys
        ):
            raise ValueError("checkpoint episode keys are malformed")
        if len(starts) != self.batch_size or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in starts
        ):
            raise ValueError("checkpoint stream cursors are malformed")
        if len(versions) != self.batch_size or any(
            not isinstance(version, int) or isinstance(version, bool) or version < -1
            for version in versions
        ):
            raise ValueError("checkpoint stream parameter versions are malformed")
        if any(
            (key is None and (start != 0 or version != -1)) or (key is not None and version < 0)
            for key, start, version in zip(keys, starts, versions, strict=True)
        ):
            raise ValueError("checkpoint stream metadata is internally inconsistent")

        raw_belief = state.get("belief")
        if not isinstance(raw_belief, Mapping) or set(raw_belief) != set(_BELIEF_FIELDS):
            raise ValueError("checkpoint posterior belief payload is malformed")
        reference = self.belief.address_mean
        loaded_fields: dict[str, torch.Tensor] = {}
        for field in _BELIEF_FIELDS:
            tensor = raw_belief[field]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("checkpoint posterior belief fields must be tensors")
            expected_dtype = (
                torch.bool
                if field == "valid"
                else torch.long
                if field == "age"
                else reference.dtype
            )
            if tensor.dtype != expected_dtype:
                raise ValueError("checkpoint posterior belief dtype differs from the active model")
            loaded_fields[field] = tensor.to(device=reference.device).detach().clone()
        loaded = ObjectBeliefBatch(**loaded_fields)
        self._validate_belief(loaded, validate_values=True)
        tracks = self._validate_loss_track_keys(raw_tracks, loaded.valid)
        self.belief = loaded
        self.episode_keys = keys
        self.next_transition_indices = starts
        self.state_parameter_versions = versions
        self.loss_track_keys_by_row = tracks

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate a checkpoint payload without mutating the live stream state."""

        if self._pending is not None:
            raise RuntimeError("cannot validate while a stream chunk is pending")
        probe = PosteriorStreamState(
            self.config,
            lane_ids=self.lane_ids,
            capacity=self.capacity,
            device=self.belief.address_mean.device,
            dtype=self.belief.address_mean.dtype,
            max_parameter_lag=self.max_parameter_lag,
        )
        probe.load_state_dict(state)


class PosteriorStreamStateGroup:
    """One checkpointable collection of stable accumulation shards on a rank."""

    def __init__(self, streams: Mapping[str, PosteriorStreamState]) -> None:
        if not isinstance(streams, Mapping) or not streams:
            raise ValueError("streams must be a non-empty mapping")
        if any(not isinstance(name, str) or not name for name in streams):
            raise ValueError("stream names must be non-empty strings")
        if any(not isinstance(stream, PosteriorStreamState) for stream in streams.values()):
            raise ValueError("stream values must be PosteriorStreamState instances")
        self._streams = dict(sorted(streams.items()))

    @classmethod
    def for_rank_partition(
        cls,
        config: TemporalFilterConfig,
        plan: FrozenEpisodeStreamPlan,
        *,
        rank: int,
        world_size: int,
        gradient_accumulation_steps: int,
        capacity: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_parameter_lag: int = 1,
    ) -> PosteriorStreamStateGroup:
        if not isinstance(plan, FrozenEpisodeStreamPlan):
            raise TypeError("plan must be a FrozenEpisodeStreamPlan")
        if (
            not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        streams = {}
        for accumulation_index in range(gradient_accumulation_steps):
            microbatch = plan.microbatch_for_rank(
                0,
                rank=rank,
                world_size=world_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                accumulation_index=accumulation_index,
            )
            name = f"accumulation-{accumulation_index:05d}"
            streams[name] = PosteriorStreamState(
                config,
                lane_ids=tuple(transition.lane_id for transition in microbatch.transitions),
                capacity=capacity,
                device=device,
                dtype=dtype,
                max_parameter_lag=max_parameter_lag,
            )
        return cls(streams)

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(self._streams)

    def __getitem__(self, name: str) -> PosteriorStreamState:
        return self._streams[name]

    @property
    def has_pending_chunks(self) -> bool:
        return any(stream.has_pending_chunk for stream in self._streams.values())

    def abort_pending_chunks(self) -> None:
        """Abort every prepared shard without changing committed stream state."""

        for stream in self._streams.values():
            if stream.has_pending_chunk:
                stream.abort_chunk()

    def commit_prepared_chunks(
        self,
        final_beliefs: Mapping[str, ObjectBeliefBatch],
        *,
        transition_count: int,
        state_parameter_version: int,
        final_loss_track_keys_by_row: Mapping[str, Sequence[Sequence[str | None]]] | None = None,
    ) -> None:
        """Atomically commit all accumulation shards for one optimizer attempt.

        Every payload is validated and detached before any live cursor mutates.
        This prevents an error in a later accumulation shard from advancing only
        a prefix of rank-local episode lanes.
        """

        if not isinstance(final_beliefs, Mapping) or set(final_beliefs) != set(self._streams):
            raise ValueError("final beliefs must cover every stream exactly once")
        if final_loss_track_keys_by_row is not None and (
            not isinstance(final_loss_track_keys_by_row, Mapping)
            or set(final_loss_track_keys_by_row) != set(self._streams)
        ):
            raise ValueError("final loss track keys must cover every stream exactly once")
        payloads = {
            name: stream._validated_commit_payload(
                final_beliefs[name],
                transition_count=transition_count,
                state_parameter_version=state_parameter_version,
                final_loss_track_keys_by_row=(
                    None
                    if final_loss_track_keys_by_row is None
                    else final_loss_track_keys_by_row[name]
                ),
            )
            for name, stream in self._streams.items()
        }
        for name, stream in self._streams.items():
            stream._apply_commit_payload(payloads[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": _STREAM_STATE_GROUP_SCHEMA,
            "streams": {name: stream.state_dict() for name, stream in self._streams.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("posterior stream state group fields are malformed")
        if state.get("schema") != _STREAM_STATE_GROUP_SCHEMA:
            raise ValueError("unsupported posterior stream state group schema")
        if set(state) != {"schema", "streams"}:
            raise ValueError("posterior stream state group fields are malformed")
        payload = state.get("streams")
        if not isinstance(payload, Mapping) or set(payload) != set(self._streams):
            raise ValueError("posterior stream state group membership differs")

        previous = {name: stream.state_dict() for name, stream in self._streams.items()}
        try:
            for name, stream in self._streams.items():
                stream.load_state_dict(payload[name])
        except Exception:
            for name, stream in self._streams.items():
                stream.load_state_dict(previous[name])
            raise

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate every rank-local stream payload without mutating live cursors."""

        if not isinstance(state, Mapping):
            raise ValueError("posterior stream state group fields are malformed")
        if state.get("schema") != _STREAM_STATE_GROUP_SCHEMA:
            raise ValueError("unsupported posterior stream state group schema")
        if set(state) != {"schema", "streams"}:
            raise ValueError("posterior stream state group fields are malformed")
        payload = state.get("streams")
        if not isinstance(payload, Mapping) or set(payload) != set(self._streams):
            raise ValueError("posterior stream state group membership differs")
        for name, stream in self._streams.items():
            stream.validate_state_dict(payload[name])
