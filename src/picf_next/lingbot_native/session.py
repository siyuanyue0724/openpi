"""Atomic per-environment state transactions for training and deployment."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import threading
from dataclasses import dataclass

import torch

from picf_next.lingbot_native.addresses import (
    EpisodeAddressState,
    deterministic_episode_permutation,
)
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.modalities import NativeModalityBatch
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    NativeLayerwisePosteriorState,
    NativePersistentState,
    NativePosteriorState,
    clone_persistent_state,
    persistent_state_tensor,
    stack_persistent_states,
    unbind_persistent_state,
)


@dataclass(frozen=True, slots=True)
class NativeSessionConfig:
    model_digest: str
    capacity: int
    host_width: int
    num_layers: int | None = None
    dtype: torch.dtype = torch.float32
    device: torch.device | str = "cpu"
    addressed_architecture_identity: str | None = None
    address_codebook_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_digest, str) or not self.model_digest:
            raise ValueError("model_digest must be a non-empty string")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.capacity, self.host_width)
        ):
            raise ValueError("session capacity and host width must be positive integers")
        if self.num_layers is not None and (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers <= 0
        ):
            raise ValueError("layerwise session depth must be a positive integer")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("session state dtype must be float16, bfloat16 or float32")
        addressed = (
            self.addressed_architecture_identity is not None,
            self.address_codebook_sha256 is not None,
        )
        if addressed[0] != addressed[1]:
            raise ValueError("addressed session identity and codebook receipt are inseparable")
        if addressed[0]:
            if self.num_layers is None:
                raise ValueError("addressed sessions require layerwise posterior state")
            if (
                not isinstance(self.addressed_architecture_identity, str)
                or not self.addressed_architecture_identity
            ):
                raise ValueError("addressed session architecture identity must be non-empty")
            receipt = self.address_codebook_sha256
            if (
                not isinstance(receipt, str)
                or len(receipt) != 64
                or any(character not in "0123456789abcdef" for character in receipt)
            ):
                raise ValueError("addressed session requires a lowercase SHA-256 receipt")

    @property
    def addressed(self) -> bool:
        return self.addressed_architecture_identity is not None

    @property
    def contract_digest(self) -> str:
        payload: dict[str, object] = {
            "capacity": self.capacity,
            "dtype": str(self.dtype),
            "host_width": self.host_width,
            "model_digest": self.model_digest,
        }
        if self.num_layers is not None:
            payload["num_layers"] = self.num_layers
            payload["state_schema"] = 3 if self.addressed else 2
        if self.addressed:
            payload["address_codebook_sha256"] = self.address_codebook_sha256
            payload["addressed_architecture_identity"] = self.addressed_architecture_identity
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeObservationBatch:
    environment_keys: tuple[str, ...]
    reset_epochs: tuple[int, ...]
    observation_sequences: tuple[int, ...]
    observation_times: torch.Tensor
    reset: tuple[bool, ...]
    controls: ExecutedControlBatch
    modalities: NativeModalityBatch | None = None
    prior_control_chunks: tuple[ExecutedControlBatch, ...] = ()

    def __post_init__(self) -> None:
        batch = len(self.environment_keys)
        if batch == 0 or len(set(self.environment_keys)) != batch:
            raise ValueError("environment keys must be non-empty and unique within a batch")
        if any(not isinstance(key, str) or not key for key in self.environment_keys):
            raise ValueError("environment keys must be non-empty strings")
        if not (
            len(self.reset_epochs) == len(self.observation_sequences) == len(self.reset) == batch
        ):
            raise ValueError("observation metadata lengths must match")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*self.reset_epochs, *self.observation_sequences)
        ):
            raise ValueError("reset epochs and observation sequences must be non-negative integers")
        if any(not isinstance(value, bool) for value in self.reset):
            raise TypeError("observation reset flags must be boolean")
        if (
            self.observation_times.shape != (batch,)
            or not self.observation_times.is_floating_point()
            or not torch.isfinite(self.observation_times).all()
        ):
            raise ValueError("observation_times must be finite with one value per environment")
        if self.controls.batch_size != batch:
            raise ValueError("observation and executed-control batches differ")
        if not isinstance(self.prior_control_chunks, tuple):
            raise TypeError("prior-control chunks must use an immutable tuple")
        if self.prior_control_chunks:
            if self.prior_control_chunks[-1] is not self.controls:
                raise ValueError("the final prior-control chunk must be the correction control")
            for chunk in self.prior_control_chunks:
                if not isinstance(chunk, ExecutedControlBatch):
                    raise TypeError("prior controls must use ExecutedControlBatch")
                if (
                    chunk.batch_size != batch
                    or chunk.action_dim != self.controls.action_dim
                    or chunk.values.device != self.controls.values.device
                    or chunk.values.dtype != self.controls.values.dtype
                    or chunk.delta_time.dtype != self.controls.delta_time.dtype
                ):
                    raise ValueError(
                        "prior-control chunks differ from the observation control contract"
                    )
        if self.modalities is not None:
            if not isinstance(self.modalities, NativeModalityBatch):
                raise TypeError("observation modalities must use the native typed contract")
            if self.modalities.batch_size != batch:
                raise ValueError("observation and modality batches differ")
            if self.modalities.device != self.controls.values.device:
                raise ValueError("observation modalities and controls must share one device")
        for index, reset in enumerate(self.reset):
            reset_event = any(
                bool(chunk.reset[index, chunk.token_valid[index]].any())
                for chunk in self.effective_prior_control_chunks
            )
            if reset_event != reset:
                raise ValueError(
                    "atomic observation reset must match its control-ledger reset event"
                )

    @property
    def effective_prior_control_chunks(self) -> tuple[ExecutedControlBatch, ...]:
        """Return the exact ordered control interval consumed before correction."""

        return self.prior_control_chunks or (self.controls,)


@dataclass(frozen=True, slots=True)
class PreparedNativeTransaction:
    token: int
    observation: NativeObservationBatch
    previous_state: NativePersistentState | None
    previous_state_valid: torch.Tensor
    episode_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    state: NativePersistentState
    reset_epoch: int
    observation_sequence: int
    observation_time: float


class NativeSessionManager:
    """Isolate recurrent rows and reject stale/cross-epoch state updates."""

    def __init__(self, config: NativeSessionConfig) -> None:
        self.config = config
        self._records: dict[str, _SessionRecord] = {}
        self._pending: dict[int, NativeObservationBatch] = {}
        self._pending_environments: set[str] = set()
        self._next_token = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._records)

    def _episode_ids(self, observation: NativeObservationBatch) -> torch.Tensor:
        values: list[int] = []
        for key, reset_epoch in zip(
            observation.environment_keys,
            observation.reset_epochs,
            strict=True,
        ):
            payload = json.dumps(
                {
                    "contract_digest": self.config.contract_digest,
                    "environment_key": key,
                    "reset_epoch": reset_epoch,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            values.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 1)
        return torch.tensor(values, dtype=torch.long, device=self.config.device)

    def _zero_state(
        self,
        *,
        episode_id: torch.Tensor,
    ) -> NativePersistentState:
        zero = torch.zeros(
            (
                (1, self.config.capacity, self.config.host_width)
                if self.config.num_layers is None
                else (
                    1,
                    self.config.num_layers,
                    self.config.capacity,
                    self.config.host_width,
                )
            ),
            device=self.config.device,
            dtype=self.config.dtype,
        )
        if self.config.num_layers is None:
            return NativePosteriorState(zero)
        if not self.config.addressed:
            return NativeLayerwisePosteriorState(zero)
        receipt = self.config.address_codebook_sha256
        identity = self.config.addressed_architecture_identity
        if not isinstance(receipt, str) or not isinstance(identity, str):
            raise RuntimeError("addressed session lost its frozen routing contract")
        return AddressedLayerwisePosteriorState(
            layer_rows=zero,
            episode_address_state=EpisodeAddressState(
                permutation=deterministic_episode_permutation(
                    episode_id,
                    self.config.capacity,
                ),
                codebook_sha256=receipt,
            ),
            architecture_identity=identity,
        )

    def prepare(self, observation: NativeObservationBatch) -> PreparedNativeTransaction:
        with self._lock:
            overlap = self._pending_environments.intersection(observation.environment_keys)
            if overlap:
                raise RuntimeError("an environment already has a pending state transaction")
            lanes: list[NativePersistentState] = []
            valid: list[bool] = []
            episode_ids = self._episode_ids(observation)
            for index, key in enumerate(observation.environment_keys):
                record = self._records.get(key)
                reset = observation.reset[index]
                epoch = observation.reset_epochs[index]
                sequence = observation.observation_sequences[index]
                timestamp = float(observation.observation_times[index].item())
                if record is None:
                    if not reset:
                        raise ValueError("a new environment must begin with an atomic reset")
                    lane_valid = False
                elif reset:
                    if epoch <= record.reset_epoch:
                        raise ValueError("reset epoch must strictly increase")
                    lane_valid = False
                else:
                    if epoch != record.reset_epoch:
                        raise ValueError("non-reset observations cannot cross reset epochs")
                    if sequence <= record.observation_sequence:
                        raise ValueError("duplicate or out-of-order observation sequence")
                    if timestamp <= record.observation_time:
                        raise ValueError("duplicate or out-of-order observation timestamp")
                    lane_valid = True
                if lane_valid:
                    if record is None:
                        raise RuntimeError("a valid session lane has no committed record")
                    lanes.append(record.state)
                else:
                    lanes.append(self._zero_state(episode_id=episode_ids[index : index + 1]))
                valid.append(lane_valid)
            previous = stack_persistent_states(tuple(lanes)) if any(valid) else None
            token = self._next_token
            self._next_token += 1
            self._pending[token] = observation
            self._pending_environments.update(observation.environment_keys)
            return PreparedNativeTransaction(
                token=token,
                observation=observation,
                previous_state=previous,
                previous_state_valid=torch.tensor(
                    valid,
                    dtype=torch.bool,
                    device=self.config.device,
                ),
                episode_ids=episode_ids,
            )

    def commit(
        self,
        transaction: PreparedNativeTransaction,
        posterior_state: NativePersistentState,
    ) -> None:
        with self._lock:
            observation = self._pending.get(transaction.token)
            if observation is None or observation is not transaction.observation:
                raise RuntimeError("transaction is unknown, aborted or already committed")
            if (
                (
                    self.config.num_layers is None
                    and not isinstance(posterior_state, NativePosteriorState)
                )
                or (
                    self.config.num_layers is not None
                    and not isinstance(posterior_state, NativeLayerwisePosteriorState)
                )
                or posterior_state.batch_size != len(observation.environment_keys)
                or posterior_state.capacity != self.config.capacity
                or posterior_state.host_width != self.config.host_width
                or persistent_state_tensor(posterior_state).device
                != torch.device(self.config.device)
                or persistent_state_tensor(posterior_state).dtype != self.config.dtype
                or (
                    isinstance(posterior_state, NativeLayerwisePosteriorState)
                    and posterior_state.num_layers != self.config.num_layers
                )
                or (
                    self.config.addressed
                    and not isinstance(posterior_state, AddressedLayerwisePosteriorState)
                )
                or (
                    not self.config.addressed
                    and isinstance(posterior_state, AddressedLayerwisePosteriorState)
                )
            ):
                raise ValueError("committed posterior differs from the session state contract")
            if isinstance(posterior_state, AddressedLayerwisePosteriorState) and (
                    posterior_state.architecture_identity
                    != self.config.addressed_architecture_identity
                    or posterior_state.episode_address_state.codebook_sha256
                    != self.config.address_codebook_sha256
            ):
                raise ValueError("committed posterior uses another address routing contract")
            states = unbind_persistent_state(clone_persistent_state(posterior_state))
            staged_records = dict(self._records)
            for index, key in enumerate(observation.environment_keys):
                staged_records[key] = _SessionRecord(
                    state=states[index],
                    reset_epoch=observation.reset_epochs[index],
                    observation_sequence=observation.observation_sequences[index],
                    observation_time=float(observation.observation_times[index].item()),
                )
            self._records = staged_records
            del self._pending[transaction.token]
            self._pending_environments.difference_update(observation.environment_keys)

    def abort(self, transaction: PreparedNativeTransaction) -> None:
        with self._lock:
            observation = self._pending.pop(transaction.token, None)
            if observation is None:
                raise RuntimeError("transaction is unknown, aborted or already committed")
            self._pending_environments.difference_update(observation.environment_keys)

    def serialize(self) -> bytes:
        with self._lock:
            if self._pending:
                raise RuntimeError("cannot snapshot while state transactions are pending")
            records = []
            for key in sorted(self._records):
                record = self._records[key]
                records.append(
                    {
                        "environment_key": key,
                        "observation_sequence": record.observation_sequence,
                        "observation_time": float(record.observation_time).hex(),
                        "reset_epoch": record.reset_epoch,
                        "state": base64.b64encode(record.state.serialize()).decode("ascii"),
                    }
                )
            payload = {
                "contract_digest": self.config.contract_digest,
                "records": records,
                "version": (
                    1
                    if self.config.num_layers is None
                    else (3 if self.config.addressed else 2)
                ),
            }
            return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def deserialize(cls, config: NativeSessionConfig, encoded: bytes) -> NativeSessionManager:
        if not isinstance(encoded, bytes):
            raise TypeError("session snapshot must be bytes")
        try:
            payload = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError("session snapshot is not valid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {
            "contract_digest",
            "records",
            "version",
        }:
            raise ValueError("session snapshot has an incompatible top-level schema")
        expected_version = 1 if config.num_layers is None else (3 if config.addressed else 2)
        if (
            payload["version"] != expected_version
            or payload["contract_digest"] != config.contract_digest
        ):
            raise ValueError("session snapshot differs from the frozen runtime contract")
        if not isinstance(payload["records"], list):
            raise ValueError("session snapshot records must be a list")
        manager = cls(config)
        for item in payload["records"]:
            if not isinstance(item, dict) or set(item) != {
                "environment_key",
                "observation_sequence",
                "observation_time",
                "reset_epoch",
                "state",
            }:
                raise ValueError("session snapshot record has an incompatible schema")
            key = item["environment_key"]
            if not isinstance(key, str) or not key:
                raise ValueError("session snapshot environment key must be non-empty")
            if key in manager._records:
                raise ValueError("session snapshot contains a duplicate environment")
            counters = (item["reset_epoch"], item["observation_sequence"])
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counters
            ):
                raise ValueError("session snapshot counters must be non-negative integers")
            if not isinstance(item["state"], str):
                raise ValueError("session snapshot state must be base64 text")
            try:
                state_bytes = base64.b64decode(item["state"], validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("session snapshot state is not valid base64") from error
            if config.num_layers is None:
                state: NativePersistentState = NativePosteriorState.deserialize(
                    state_bytes,
                    device=config.device,
                )
            elif config.addressed:
                state = AddressedLayerwisePosteriorState.deserialize(
                    state_bytes,
                    device=config.device,
                )
            else:
                state = NativeLayerwisePosteriorState.deserialize(
                    state_bytes,
                    device=config.device,
                )
            state_tensor = persistent_state_tensor(state)
            if (
                state.batch_size != 1
                or state.capacity != config.capacity
                or state.host_width != config.host_width
                or state_tensor.dtype != config.dtype
                or (
                    isinstance(state, NativeLayerwisePosteriorState)
                    and state.num_layers != config.num_layers
                )
            ):
                raise ValueError("snapshot posterior differs from the session contract")
            if isinstance(state, AddressedLayerwisePosteriorState) and (
                    state.architecture_identity != config.addressed_architecture_identity
                    or state.episode_address_state.codebook_sha256
                    != config.address_codebook_sha256
            ):
                raise ValueError("snapshot posterior uses another address routing contract")
            if not isinstance(item["observation_time"], str):
                raise ValueError("session snapshot observation time must be hexadecimal text")
            observation_time = float.fromhex(item["observation_time"])
            if not math.isfinite(observation_time):
                raise ValueError("snapshot contains a non-finite observation time")
            manager._records[key] = _SessionRecord(
                state=state,
                reset_epoch=item["reset_epoch"],
                observation_sequence=item["observation_sequence"],
                observation_time=observation_time,
            )
        return manager
