"""The sole learned state serialized by strict LingBot-native PICF."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import cast

import torch

from picf_next.lingbot_native.addresses import EpisodeAddressState

_MAGIC = b"PICFNR01"
_VERSION = 1
_HEADER = struct.Struct("<8sBIIIIB")
_DIGEST_BYTES = 32
_DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}

_LAYERWISE_MAGIC = b"PICFNLR2"
_LAYERWISE_VERSION = 2
_LAYERWISE_HEADER = struct.Struct("<8sBIIIIIB")

_ADDRESSED_LAYERWISE_MAGIC = b"PICFALR3"
_ADDRESSED_LAYERWISE_VERSION = 1
_ADDRESSED_LAYERWISE_HEADER = struct.Struct("<8sBIII")

_VIDEOMT_PAIRED_MAGIC = b"PICFVQH1"
_VIDEOMT_PAIRED_VERSION = 1
_VIDEOMT_PAIRED_HEADER = struct.Struct("<8sBIII")


@dataclass(frozen=True, slots=True)
class NativePosteriorState:
    """Exactly ``K x d_host`` recurrent rows, with no lifecycle side state."""

    rows: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.rows, torch.Tensor):
            raise TypeError("posterior rows must be a torch tensor")
        if self.rows.ndim != 3 or min(self.rows.shape) <= 0:
            raise ValueError("posterior rows must have shape [batch, capacity, host_width]")
        if self.rows.dtype not in _DTYPE_TO_CODE:
            raise TypeError("posterior rows must use float16, bfloat16 or float32")
        if not torch.isfinite(self.rows).all():
            raise ValueError("posterior rows contain NaN or infinity")

    @property
    def batch_size(self) -> int:
        return self.rows.shape[0]

    @property
    def capacity(self) -> int:
        return self.rows.shape[1]

    @property
    def host_width(self) -> int:
        return self.rows.shape[2]

    def detached(self) -> NativePosteriorState:
        return NativePosteriorState(self.rows.detach())

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> NativePosteriorState:
        return NativePosteriorState(
            self.rows.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
            )
        )

    def index_select(self, indices: torch.Tensor) -> NativePosteriorState:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("batch indices must be a rank-one long tensor")
        return NativePosteriorState(self.rows.index_select(0, indices))

    def permute_rows(self, permutation: torch.Tensor) -> NativePosteriorState:
        if (
            permutation.ndim != 1
            or permutation.dtype != torch.long
            or permutation.shape[0] != self.capacity
        ):
            raise ValueError("row permutation must contain one long index per row")
        expected = torch.arange(self.capacity, device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("row permutation must contain every row exactly once")
        return NativePosteriorState(self.rows.index_select(1, permutation.to(self.rows.device)))

    def serialize(self) -> bytes:
        """Return deterministic, checksummed bytes without Python pickle."""

        cpu_rows = self.rows.detach().contiguous().cpu()
        payload = cpu_rows.view(torch.uint8).numpy().tobytes()
        header = _HEADER.pack(
            _MAGIC,
            _VERSION,
            self.batch_size,
            self.capacity,
            self.host_width,
            len(payload),
            _DTYPE_TO_CODE[cpu_rows.dtype],
        )
        return header + hashlib.sha256(payload).digest() + payload

    @classmethod
    def deserialize(
        cls, encoded: bytes, *, device: torch.device | str = "cpu"
    ) -> NativePosteriorState:
        if not isinstance(encoded, bytes):
            raise TypeError("serialized posterior must be bytes")
        minimum = _HEADER.size + _DIGEST_BYTES
        if len(encoded) < minimum:
            raise ValueError("serialized posterior is truncated")
        magic, version, batch, capacity, width, payload_size, dtype_code = _HEADER.unpack_from(
            encoded
        )
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("serialized posterior has an incompatible schema")
        dtype = _CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError("serialized posterior uses an unknown dtype")
        digest = encoded[_HEADER.size : minimum]
        payload = encoded[minimum:]
        if len(payload) != payload_size:
            raise ValueError("serialized posterior payload length is invalid")
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError("serialized posterior checksum does not match")
        element_size = torch.empty((), dtype=dtype).element_size()
        if payload_size != batch * capacity * width * element_size:
            raise ValueError("serialized posterior shape and payload disagree")
        mutable = bytearray(payload)
        rows = torch.frombuffer(mutable, dtype=dtype).clone().reshape(batch, capacity, width)
        return cls(rows.to(device=device))


def stack_native_states(states: tuple[NativePosteriorState, ...]) -> NativePosteriorState:
    if not states:
        raise ValueError("at least one posterior state is required")
    if any(state.batch_size != 1 for state in states):
        raise ValueError("only singleton lane states may be stacked")
    reference = states[0].rows
    if any(
        state.capacity != states[0].capacity
        or state.host_width != states[0].host_width
        or state.rows.device != reference.device
        or state.rows.dtype != reference.dtype
        for state in states
    ):
        raise ValueError("posterior lane states must share shape, device and dtype")
    return NativePosteriorState(torch.cat([state.rows for state in states], dim=0))


def unbind_native_state(state: NativePosteriorState) -> tuple[NativePosteriorState, ...]:
    return tuple(NativePosteriorState(rows.unsqueeze(0)) for rows in state.rows.unbind(0))


@dataclass(frozen=True, slots=True)
class NativeLayerwisePosteriorState:
    """The sole persistent object state: one hidden row set per LingBot layer."""

    layer_rows: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.layer_rows, torch.Tensor):
            raise TypeError("layerwise posterior rows must be a torch tensor")
        if self.layer_rows.ndim != 4 or min(self.layer_rows.shape) <= 0:
            raise ValueError(
                "layerwise posterior rows must have shape [batch, layers, capacity, host_width]"
            )
        if self.layer_rows.dtype not in _DTYPE_TO_CODE:
            raise TypeError("layerwise posterior rows must use float16, bfloat16 or float32")
        if not torch.isfinite(self.layer_rows).all():
            raise ValueError("layerwise posterior rows contain NaN or infinity")

    @property
    def batch_size(self) -> int:
        return self.layer_rows.shape[0]

    @property
    def num_layers(self) -> int:
        return self.layer_rows.shape[1]

    @property
    def capacity(self) -> int:
        return self.layer_rows.shape[2]

    @property
    def host_width(self) -> int:
        return self.layer_rows.shape[3]

    def layer(self, layer_index: int) -> torch.Tensor:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.num_layers
        ):
            raise IndexError("layerwise posterior index is outside the stored host depth")
        return self.layer_rows[:, layer_index]

    def detached(self) -> NativeLayerwisePosteriorState:
        return NativeLayerwisePosteriorState(self.layer_rows.detach())

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> NativeLayerwisePosteriorState:
        return NativeLayerwisePosteriorState(
            self.layer_rows.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
            )
        )

    def index_select(self, indices: torch.Tensor) -> NativeLayerwisePosteriorState:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("batch indices must be a rank-one long tensor")
        return NativeLayerwisePosteriorState(self.layer_rows.index_select(0, indices))

    def permute_rows(self, permutation: torch.Tensor) -> NativeLayerwisePosteriorState:
        if (
            permutation.ndim != 1
            or permutation.dtype != torch.long
            or permutation.shape[0] != self.capacity
        ):
            raise ValueError("row permutation must contain one long index per row")
        expected = torch.arange(self.capacity, device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("row permutation must contain every row exactly once")
        return NativeLayerwisePosteriorState(
            self.layer_rows.index_select(2, permutation.to(self.layer_rows.device))
        )

    def serialize(self) -> bytes:
        """Return deterministic schema-v2 bytes; rejected v1 rows cannot be loaded."""

        cpu_rows = self.layer_rows.detach().contiguous().cpu()
        payload = cpu_rows.view(torch.uint8).numpy().tobytes()
        header = _LAYERWISE_HEADER.pack(
            _LAYERWISE_MAGIC,
            _LAYERWISE_VERSION,
            self.batch_size,
            self.num_layers,
            self.capacity,
            self.host_width,
            len(payload),
            _DTYPE_TO_CODE[cpu_rows.dtype],
        )
        return header + hashlib.sha256(payload).digest() + payload

    @classmethod
    def deserialize(
        cls, encoded: bytes, *, device: torch.device | str = "cpu"
    ) -> NativeLayerwisePosteriorState:
        if not isinstance(encoded, bytes):
            raise TypeError("serialized layerwise posterior must be bytes")
        minimum = _LAYERWISE_HEADER.size + _DIGEST_BYTES
        if len(encoded) < minimum:
            raise ValueError("serialized layerwise posterior is truncated")
        (
            magic,
            version,
            batch,
            layers,
            capacity,
            width,
            payload_size,
            dtype_code,
        ) = _LAYERWISE_HEADER.unpack_from(encoded)
        if magic != _LAYERWISE_MAGIC or version != _LAYERWISE_VERSION:
            raise ValueError("serialized layerwise posterior has an incompatible schema")
        dtype = _CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError("serialized layerwise posterior uses an unknown dtype")
        digest = encoded[_LAYERWISE_HEADER.size : minimum]
        payload = encoded[minimum:]
        if len(payload) != payload_size:
            raise ValueError("serialized layerwise posterior payload length is invalid")
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError("serialized layerwise posterior checksum does not match")
        element_size = torch.empty((), dtype=dtype).element_size()
        if payload_size != batch * layers * capacity * width * element_size:
            raise ValueError("serialized layerwise posterior shape and payload disagree")
        mutable = bytearray(payload)
        rows = (
            torch.frombuffer(mutable, dtype=dtype).clone().reshape(batch, layers, capacity, width)
        )
        return cls(rows.to(device=device))


@dataclass(frozen=True, slots=True)
class AddressedLayerwisePosteriorState(NativeLayerwisePosteriorState):
    """Persistent posterior rows bound to their episode-local routing gauge."""

    episode_address_state: EpisodeAddressState
    architecture_identity: str

    def __post_init__(self) -> None:
        NativeLayerwisePosteriorState.__post_init__(self)
        if not isinstance(self.episode_address_state, EpisodeAddressState):
            raise TypeError("addressed posterior requires EpisodeAddressState")
        if self.episode_address_state.batch_size != self.batch_size:
            raise ValueError("addressed posterior and address state batches differ")
        if self.episode_address_state.capacity != self.capacity:
            raise ValueError("addressed posterior and address state capacities differ")
        if self.episode_address_state.device != self.layer_rows.device:
            raise ValueError("addressed posterior and address state must share one device")
        if not isinstance(self.architecture_identity, str) or not self.architecture_identity:
            raise ValueError("addressed posterior requires an architecture identity")

    @property
    def address_receipt(self) -> str:
        return self.episode_address_state.receipt

    def detached(self) -> AddressedLayerwisePosteriorState:
        return AddressedLayerwisePosteriorState(
            layer_rows=self.layer_rows.detach(),
            episode_address_state=self.episode_address_state,
            architecture_identity=self.architecture_identity,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> AddressedLayerwisePosteriorState:
        rows = self.layer_rows.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        return AddressedLayerwisePosteriorState(
            layer_rows=rows,
            episode_address_state=self.episode_address_state.to(
                rows.device,
                non_blocking=non_blocking,
            ),
            architecture_identity=self.architecture_identity,
        )

    def index_select(self, indices: torch.Tensor) -> AddressedLayerwisePosteriorState:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("addressed posterior batch indices must be rank-one long")
        return AddressedLayerwisePosteriorState(
            layer_rows=self.layer_rows.index_select(
                0,
                indices.to(self.layer_rows.device),
            ),
            episode_address_state=self.episode_address_state.index_select(indices),
            architecture_identity=self.architecture_identity,
        )

    def permute_rows(self, permutation: torch.Tensor) -> AddressedLayerwisePosteriorState:
        if (
            permutation.ndim != 1
            or permutation.dtype != torch.long
            or permutation.shape[0] != self.capacity
        ):
            raise ValueError("row permutation must contain one long index per row")
        expected = torch.arange(self.capacity, device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("row permutation must contain every row exactly once")
        return AddressedLayerwisePosteriorState(
            layer_rows=self.layer_rows.index_select(
                2,
                permutation.to(self.layer_rows.device),
            ),
            episode_address_state=self.episode_address_state.permute_rows(permutation),
            architecture_identity=self.architecture_identity,
        )

    def serialize(self) -> bytes:
        """Return deterministic bytes containing rows and their routing receipt."""

        metadata = json.dumps(
            {
                "architecture_identity": self.architecture_identity,
                "codebook_sha256": self.episode_address_state.codebook_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        permutation = (
            self.episode_address_state.permutation.detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
            .tobytes(order="C")
        )
        rows = NativeLayerwisePosteriorState(self.layer_rows).serialize()
        payload = metadata + permutation + rows
        header = _ADDRESSED_LAYERWISE_HEADER.pack(
            _ADDRESSED_LAYERWISE_MAGIC,
            _ADDRESSED_LAYERWISE_VERSION,
            len(metadata),
            len(permutation),
            len(rows),
        )
        return header + hashlib.sha256(payload).digest() + payload

    @classmethod
    def deserialize(
        cls,
        encoded: bytes,
        *,
        device: torch.device | str = "cpu",
    ) -> AddressedLayerwisePosteriorState:
        if not isinstance(encoded, bytes):
            raise TypeError("serialized addressed posterior must be bytes")
        minimum = _ADDRESSED_LAYERWISE_HEADER.size + _DIGEST_BYTES
        if len(encoded) < minimum:
            raise ValueError("serialized addressed posterior is truncated")
        magic, version, metadata_size, permutation_size, rows_size = (
            _ADDRESSED_LAYERWISE_HEADER.unpack_from(encoded)
        )
        if magic != _ADDRESSED_LAYERWISE_MAGIC or version != _ADDRESSED_LAYERWISE_VERSION:
            raise ValueError("serialized addressed posterior has an incompatible schema")
        payload = encoded[minimum:]
        if len(payload) != metadata_size + permutation_size + rows_size:
            raise ValueError("serialized addressed posterior payload length is invalid")
        digest = encoded[_ADDRESSED_LAYERWISE_HEADER.size : minimum]
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError("serialized addressed posterior checksum does not match")
        metadata_bytes = payload[:metadata_size]
        permutation_bytes = payload[metadata_size : metadata_size + permutation_size]
        rows_bytes = payload[metadata_size + permutation_size :]
        try:
            metadata = json.loads(metadata_bytes)
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise ValueError("serialized addressed posterior metadata is invalid") from error
        if not isinstance(metadata, dict) or set(metadata) != {
            "architecture_identity",
            "codebook_sha256",
        }:
            raise ValueError("serialized addressed posterior metadata schema is invalid")
        rows = NativeLayerwisePosteriorState.deserialize(rows_bytes, device=device)
        expected_permutation_size = rows.batch_size * rows.capacity * torch.int64.itemsize
        if permutation_size != expected_permutation_size:
            raise ValueError("serialized addressed posterior permutation size is invalid")
        permutation = torch.frombuffer(
            bytearray(permutation_bytes),
            dtype=torch.int64,
        ).reshape(rows.batch_size, rows.capacity)
        address_state = EpisodeAddressState(
            permutation=permutation.to(device=device),
            codebook_sha256=metadata["codebook_sha256"],
        )
        return cls(
            layer_rows=rows.layer_rows,
            episode_address_state=address_state,
            architecture_identity=metadata["architecture_identity"],
        )


@dataclass(frozen=True, slots=True)
class NativeVidEoMTPairedPosteriorState(NativeLayerwisePosteriorState):
    """Atomic recurrent state for one native VidEoMT/LingBot query gauge.

    ``source_queries[:, i]`` and ``layer_rows[:, :, i]`` are the two views of
    the same native source-query address. They are serialized and permuted as
    one object so an optimizer failure or checkpoint cannot advance only one
    side of the posterior.
    """

    source_queries: torch.Tensor
    architecture_identity: str

    def __post_init__(self) -> None:
        NativeLayerwisePosteriorState.__post_init__(self)
        if not isinstance(self.source_queries, torch.Tensor):
            raise TypeError("paired VidEoMT source queries must be a tensor")
        if (
            self.source_queries.ndim != 3
            or self.source_queries.shape[:2] != (self.batch_size, self.capacity)
            or self.source_queries.shape[-1] <= 0
        ):
            raise ValueError("paired VidEoMT source queries must be [batch, capacity, width]")
        if self.source_queries.dtype not in _DTYPE_TO_CODE:
            raise TypeError("paired VidEoMT source queries use an unsupported dtype")
        if self.source_queries.device != self.layer_rows.device:
            raise ValueError("paired source and host posterior must share one device")
        if not torch.isfinite(self.source_queries).all():
            raise ValueError("paired VidEoMT source queries contain NaN or infinity")
        if not isinstance(self.architecture_identity, str) or not self.architecture_identity:
            raise ValueError("paired VidEoMT posterior requires an architecture identity")

    @property
    def source_width(self) -> int:
        return self.source_queries.shape[-1]

    @property
    def host_state(self) -> NativeLayerwisePosteriorState:
        return NativeLayerwisePosteriorState(self.layer_rows)

    def detached(self) -> NativeVidEoMTPairedPosteriorState:
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=self.layer_rows.detach(),
            source_queries=self.source_queries.detach(),
            architecture_identity=self.architecture_identity,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> NativeVidEoMTPairedPosteriorState:
        rows = self.layer_rows.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        # Host mixed precision must not silently reduce the source recurrence.
        source = self.source_queries.to(
            device=rows.device,
            dtype=self.source_queries.dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=rows,
            source_queries=source,
            architecture_identity=self.architecture_identity,
        )

    def index_select(self, indices: torch.Tensor) -> NativeVidEoMTPairedPosteriorState:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("paired posterior batch indices must be rank-one long")
        indices = indices.to(self.layer_rows.device)
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=self.layer_rows.index_select(0, indices),
            source_queries=self.source_queries.index_select(0, indices),
            architecture_identity=self.architecture_identity,
        )

    def permute_rows(self, permutation: torch.Tensor) -> NativeVidEoMTPairedPosteriorState:
        if (
            permutation.ndim != 1
            or permutation.dtype != torch.long
            or permutation.shape[0] != self.capacity
        ):
            raise ValueError("row permutation must contain one long index per row")
        expected = torch.arange(self.capacity, device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("row permutation must contain every row exactly once")
        permutation = permutation.to(self.layer_rows.device)
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=self.layer_rows.index_select(2, permutation),
            source_queries=self.source_queries.index_select(1, permutation),
            architecture_identity=self.architecture_identity,
        )

    def serialize(self) -> bytes:
        metadata = json.dumps(
            {"architecture_identity": self.architecture_identity},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        host = NativeLayerwisePosteriorState(self.layer_rows).serialize()
        source = NativePosteriorState(self.source_queries).serialize()
        payload = metadata + host + source
        header = _VIDEOMT_PAIRED_HEADER.pack(
            _VIDEOMT_PAIRED_MAGIC,
            _VIDEOMT_PAIRED_VERSION,
            len(metadata),
            len(host),
            len(source),
        )
        return header + hashlib.sha256(payload).digest() + payload

    @classmethod
    def deserialize(
        cls,
        encoded: bytes,
        *,
        device: torch.device | str = "cpu",
    ) -> NativeVidEoMTPairedPosteriorState:
        if not isinstance(encoded, bytes):
            raise TypeError("serialized paired posterior must be bytes")
        minimum = _VIDEOMT_PAIRED_HEADER.size + _DIGEST_BYTES
        if len(encoded) < minimum:
            raise ValueError("serialized paired posterior is truncated")
        magic, version, metadata_size, host_size, source_size = (
            _VIDEOMT_PAIRED_HEADER.unpack_from(encoded)
        )
        if magic != _VIDEOMT_PAIRED_MAGIC or version != _VIDEOMT_PAIRED_VERSION:
            raise ValueError("serialized paired posterior has an incompatible schema")
        payload = encoded[minimum:]
        if len(payload) != metadata_size + host_size + source_size:
            raise ValueError("serialized paired posterior payload length is invalid")
        digest = encoded[_VIDEOMT_PAIRED_HEADER.size : minimum]
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError("serialized paired posterior checksum does not match")
        metadata_bytes = payload[:metadata_size]
        host_bytes = payload[metadata_size : metadata_size + host_size]
        source_bytes = payload[metadata_size + host_size :]
        try:
            metadata = json.loads(metadata_bytes)
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise ValueError("serialized paired posterior metadata is invalid") from error
        if not isinstance(metadata, dict) or set(metadata) != {"architecture_identity"}:
            raise ValueError("serialized paired posterior metadata schema is invalid")
        host = NativeLayerwisePosteriorState.deserialize(host_bytes, device=device)
        source = NativePosteriorState.deserialize(source_bytes, device=device)
        return cls(
            layer_rows=host.layer_rows,
            source_queries=source.rows,
            architecture_identity=metadata["architecture_identity"],
        )


@dataclass(frozen=True, slots=True)
class NativeLayerwisePriorTrace:
    """Transient per-layer prior rows produced by the shared LingBot host.

    This type is deliberately absent from ``NativePersistentState`` and has no
    serialization method. A prior trace may remain attached inside one model
    graph, but it cannot be mistaken for the posterior state committed to a
    deployment lane.
    """

    layer_rows: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.layer_rows, torch.Tensor):
            raise TypeError("layerwise prior rows must be a torch tensor")
        if self.layer_rows.ndim != 4 or min(self.layer_rows.shape) <= 0:
            raise ValueError(
                "layerwise prior rows must have shape [batch, layers, capacity, host_width]"
            )
        if self.layer_rows.dtype not in _DTYPE_TO_CODE:
            raise TypeError("layerwise prior rows must use float16, bfloat16 or float32")
        if not torch.isfinite(self.layer_rows).all():
            raise ValueError("layerwise prior rows contain NaN or infinity")

    @property
    def batch_size(self) -> int:
        return self.layer_rows.shape[0]

    @property
    def num_layers(self) -> int:
        return self.layer_rows.shape[1]

    @property
    def capacity(self) -> int:
        return self.layer_rows.shape[2]

    @property
    def host_width(self) -> int:
        return self.layer_rows.shape[3]

    def layer(self, layer_index: int) -> torch.Tensor:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.num_layers
        ):
            raise IndexError("layerwise prior index is outside the produced host depth")
        return self.layer_rows[:, layer_index]

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> NativeLayerwisePriorTrace:
        return NativeLayerwisePriorTrace(
            self.layer_rows.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
            )
        )

    def permute_rows(self, permutation: torch.Tensor) -> NativeLayerwisePriorTrace:
        if (
            permutation.ndim != 1
            or permutation.dtype != torch.long
            or permutation.shape[0] != self.capacity
        ):
            raise ValueError("row permutation must contain one long index per row")
        expected = torch.arange(self.capacity, device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("row permutation must contain every row exactly once")
        return NativeLayerwisePriorTrace(
            self.layer_rows.index_select(2, permutation.to(self.layer_rows.device))
        )


@dataclass(frozen=True, slots=True)
class AddressedLayerwisePriorTrace(NativeLayerwisePriorTrace):
    """A transient prior trace bound to one episode-local address gauge.

    The historical ``NativeLayerwisePriorTrace`` schema remains unchanged. This
    subtype preserves its public surface and ``isinstance`` behavior while
    making the LTOP prior-to-correction receipt impossible to omit silently.
    """

    episode_address_state: EpisodeAddressState
    architecture_identity: str

    def __post_init__(self) -> None:
        NativeLayerwisePriorTrace.__post_init__(self)
        if not isinstance(self.episode_address_state, EpisodeAddressState):
            raise TypeError("addressed prior trace requires EpisodeAddressState")
        if self.episode_address_state.batch_size != self.batch_size:
            raise ValueError("addressed prior trace and address state batches differ")
        if self.episode_address_state.capacity != self.capacity:
            raise ValueError("addressed prior trace and address state capacities differ")
        if self.episode_address_state.device != self.layer_rows.device:
            raise ValueError("addressed prior trace and address state must share one device")
        if not isinstance(self.architecture_identity, str) or not self.architecture_identity:
            raise ValueError("addressed prior trace requires an architecture identity")

    @property
    def address_receipt(self) -> str:
        return self.episode_address_state.receipt

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> AddressedLayerwisePriorTrace:
        rows = self.layer_rows.to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
        return AddressedLayerwisePriorTrace(
            layer_rows=rows,
            episode_address_state=self.episode_address_state.to(
                rows.device,
                non_blocking=non_blocking,
            ),
            architecture_identity=self.architecture_identity,
        )

    def index_select(self, indices: torch.Tensor) -> AddressedLayerwisePriorTrace:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("addressed trace batch indices must be rank-one long")
        return AddressedLayerwisePriorTrace(
            layer_rows=self.layer_rows.index_select(
                0,
                indices.to(self.layer_rows.device),
            ),
            episode_address_state=self.episode_address_state.index_select(indices),
            architecture_identity=self.architecture_identity,
        )

    def permute_rows(self, permutation: torch.Tensor) -> AddressedLayerwisePriorTrace:
        if (
            permutation.ndim != 1
            or permutation.dtype != torch.long
            or permutation.shape[0] != self.capacity
        ):
            raise ValueError("row permutation must contain one long index per row")
        expected = torch.arange(self.capacity, device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("row permutation must contain every row exactly once")
        return AddressedLayerwisePriorTrace(
            layer_rows=self.layer_rows.index_select(
                2,
                permutation.to(self.layer_rows.device),
            ),
            episode_address_state=self.episode_address_state.permute_rows(permutation),
            architecture_identity=self.architecture_identity,
        )


def stack_layerwise_states(
    states: tuple[NativeLayerwisePosteriorState, ...],
) -> NativeLayerwisePosteriorState:
    if not states:
        raise ValueError("at least one layerwise posterior state is required")
    if any(state.batch_size != 1 for state in states):
        raise ValueError("only singleton layerwise lane states may be stacked")
    reference = states[0].layer_rows
    if any(
        state.num_layers != states[0].num_layers
        or state.capacity != states[0].capacity
        or state.host_width != states[0].host_width
        or state.layer_rows.device != reference.device
        or state.layer_rows.dtype != reference.dtype
        for state in states
    ):
        raise ValueError("layerwise lane states must share shape, device and dtype")
    paired = tuple(isinstance(state, NativeVidEoMTPairedPosteriorState) for state in states)
    if any(paired):
        if not all(paired):
            raise TypeError("paired and unpaired posterior lanes cannot be stacked")
        typed_paired = cast(tuple[NativeVidEoMTPairedPosteriorState, ...], states)
        identity = typed_paired[0].architecture_identity
        source_reference = typed_paired[0].source_queries
        if any(
            state.architecture_identity != identity
            or state.source_width != typed_paired[0].source_width
            or state.source_queries.device != source_reference.device
            or state.source_queries.dtype != source_reference.dtype
            for state in typed_paired
        ):
            raise ValueError("paired posterior lanes use different source contracts")
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=torch.cat([state.layer_rows for state in typed_paired], dim=0),
            source_queries=torch.cat(
                [state.source_queries for state in typed_paired],
                dim=0,
            ),
            architecture_identity=identity,
        )
    addressed = tuple(isinstance(state, AddressedLayerwisePosteriorState) for state in states)
    if any(addressed):
        if not all(addressed):
            raise TypeError("addressed and unaddressed posterior lanes cannot be stacked")
        typed = cast(tuple[AddressedLayerwisePosteriorState, ...], states)
        identity = typed[0].architecture_identity
        codebook_sha256 = typed[0].episode_address_state.codebook_sha256
        if any(
            state.architecture_identity != identity
            or state.episode_address_state.codebook_sha256 != codebook_sha256
            for state in typed
        ):
            raise ValueError("addressed posterior lanes use different routing contracts")
        return AddressedLayerwisePosteriorState(
            layer_rows=torch.cat([state.layer_rows for state in typed], dim=0),
            episode_address_state=EpisodeAddressState(
                permutation=torch.cat(
                    [state.episode_address_state.permutation for state in typed],
                    dim=0,
                ),
                codebook_sha256=codebook_sha256,
            ),
            architecture_identity=identity,
        )
    return NativeLayerwisePosteriorState(torch.cat([state.layer_rows for state in states], dim=0))


def unbind_layerwise_state(
    state: NativeLayerwisePosteriorState,
) -> tuple[NativeLayerwisePosteriorState, ...]:
    if isinstance(state, NativeVidEoMTPairedPosteriorState):
        return tuple(
            NativeVidEoMTPairedPosteriorState(
                layer_rows=rows.unsqueeze(0),
                source_queries=state.source_queries[index : index + 1],
                architecture_identity=state.architecture_identity,
            )
            for index, rows in enumerate(state.layer_rows.unbind(0))
        )
    if isinstance(state, AddressedLayerwisePosteriorState):
        return tuple(
            AddressedLayerwisePosteriorState(
                layer_rows=rows.unsqueeze(0),
                episode_address_state=state.episode_address_state.index_select(
                    torch.tensor(
                        [index],
                        dtype=torch.long,
                        device=state.layer_rows.device,
                    )
                ),
                architecture_identity=state.architecture_identity,
            )
            for index, rows in enumerate(state.layer_rows.unbind(0))
        )
    return tuple(
        NativeLayerwisePosteriorState(rows.unsqueeze(0)) for rows in state.layer_rows.unbind(0)
    )


NativePersistentState = (
    NativePosteriorState
    | NativeLayerwisePosteriorState
    | AddressedLayerwisePosteriorState
    | NativeVidEoMTPairedPosteriorState
)


def persistent_state_tensor(state: NativePersistentState) -> torch.Tensor:
    if isinstance(state, NativePosteriorState):
        return state.rows
    if isinstance(state, NativeLayerwisePosteriorState):
        return state.layer_rows
    raise TypeError("persistent state uses an unknown schema")


def persistent_state_with_tensor(
    state: NativePersistentState,
    tensor: torch.Tensor,
    *,
    episode_address_state: EpisodeAddressState | None = None,
) -> NativePersistentState:
    """Replace state values without silently discarding an addressed routing gauge."""

    if isinstance(state, NativeVidEoMTPairedPosteriorState):
        if episode_address_state is not None:
            raise ValueError("a paired source posterior cannot receive episode addresses")
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=tensor,
            source_queries=state.source_queries,
            architecture_identity=state.architecture_identity,
        )
    if isinstance(state, AddressedLayerwisePosteriorState):
        return AddressedLayerwisePosteriorState(
            layer_rows=tensor,
            episode_address_state=(
                state.episode_address_state
                if episode_address_state is None
                else episode_address_state
            ),
            architecture_identity=state.architecture_identity,
        )
    if episode_address_state is not None:
        raise ValueError("an unaddressed posterior cannot receive an episode address state")
    if isinstance(state, NativePosteriorState):
        return NativePosteriorState(tensor)
    if isinstance(state, NativeLayerwisePosteriorState):
        return NativeLayerwisePosteriorState(tensor)
    raise TypeError("persistent state uses an unknown schema")


def layerwise_prior_trace_with_tensor(
    trace: NativeLayerwisePriorTrace,
    tensor: torch.Tensor,
    *,
    episode_address_state: EpisodeAddressState | None = None,
) -> NativeLayerwisePriorTrace:
    """Replace prior values while preserving or explicitly replacing their address receipt."""

    if isinstance(trace, AddressedLayerwisePriorTrace):
        return AddressedLayerwisePriorTrace(
            layer_rows=tensor,
            episode_address_state=(
                trace.episode_address_state
                if episode_address_state is None
                else episode_address_state
            ),
            architecture_identity=trace.architecture_identity,
        )
    if episode_address_state is not None:
        raise ValueError("an unaddressed prior cannot receive an episode address state")
    if isinstance(trace, NativeLayerwisePriorTrace):
        return NativeLayerwisePriorTrace(tensor)
    raise TypeError("prior trace uses an unknown schema")


def clone_persistent_state(
    state: NativePersistentState,
    *,
    detach: bool = True,
) -> NativePersistentState:
    tensor = persistent_state_tensor(state)
    if detach:
        tensor = tensor.detach()
    tensor = tensor.clone()
    if isinstance(state, NativeVidEoMTPairedPosteriorState):
        source = state.source_queries.detach() if detach else state.source_queries
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=tensor,
            source_queries=source.clone(),
            architecture_identity=state.architecture_identity,
        )
    if isinstance(state, AddressedLayerwisePosteriorState):
        return AddressedLayerwisePosteriorState(
            layer_rows=tensor,
            episode_address_state=EpisodeAddressState(
                permutation=state.episode_address_state.permutation.clone(),
                codebook_sha256=state.episode_address_state.codebook_sha256,
            ),
            architecture_identity=state.architecture_identity,
        )
    if isinstance(state, NativePosteriorState):
        return NativePosteriorState(tensor)
    return NativeLayerwisePosteriorState(tensor)


def stack_persistent_states(
    states: tuple[NativePersistentState, ...],
) -> NativePersistentState:
    if not states:
        raise ValueError("at least one persistent state is required")
    if all(isinstance(state, NativePosteriorState) for state in states):
        return stack_native_states(cast(tuple[NativePosteriorState, ...], states))
    if all(isinstance(state, NativeLayerwisePosteriorState) for state in states):
        return stack_layerwise_states(cast(tuple[NativeLayerwisePosteriorState, ...], states))
    raise TypeError("persistent states from different schemas cannot be stacked")


def unbind_persistent_state(
    state: NativePersistentState,
) -> tuple[NativePersistentState, ...]:
    if isinstance(state, NativePosteriorState):
        return unbind_native_state(state)
    if isinstance(state, NativeLayerwisePosteriorState):
        return unbind_layerwise_state(state)
    raise TypeError("persistent state uses an unknown schema")
