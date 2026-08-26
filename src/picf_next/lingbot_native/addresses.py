"""Episode-local, non-semantic Q/K addresses for persistent object rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _power_of_two_at_least(value: int) -> int:
    order = 1
    while order < value:
        order *= 2
    return order


def _walsh_hadamard(order: int, *, device: torch.device | str | None) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float64, device=device)
    while matrix.shape[0] < order:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix


def fixed_orthogonal_address_codebook(
    capacity: int,
    host_width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a deterministic unit-norm codebook with orthogonal row addresses.

    The codebook is a buffer, never a trainable semantic embedding. A Walsh
    basis is tiled across the host width so that pairwise orthogonality is exact
    up to floating-point roundoff while evidence Values remain content-only.
    """

    dimensions = (capacity, host_width)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
        raise TypeError("address dimensions must be integers")
    if capacity <= 0 or host_width <= 0:
        raise ValueError("address dimensions must be positive")
    if not dtype.is_floating_point:
        raise TypeError("address dtype must be floating point")
    order = _power_of_two_at_least(capacity)
    if order > host_width:
        raise ValueError("host width is too small for the requested orthogonal capacity")
    repeat = host_width // order
    basis = _walsh_hadamard(order, device=device)[:capacity]
    codebook = basis.repeat(1, repeat)
    if codebook.shape[1] < host_width:
        codebook = F.pad(codebook, (0, host_width - codebook.shape[1]))
    return F.normalize(codebook, dim=-1).to(dtype=dtype)


def address_codebook_sha256(codebook: torch.Tensor) -> str:
    """Return a stable receipt for one immutable address codebook."""

    if codebook.ndim != 2 or not codebook.is_floating_point():
        raise ValueError("address codebook must be a rank-two floating tensor")
    canonical = codebook.detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = canonical.numpy().tobytes(order="C")
    metadata = f"{tuple(canonical.shape)}:{canonical.dtype}".encode()
    return hashlib.sha256(metadata + payload).hexdigest()


def deterministic_episode_permutation(
    episode_ids: torch.Tensor,
    capacity: int,
    *,
    salt: int = 0x50494346,
) -> torch.Tensor:
    """Map integer episode IDs to reproducible, uniformly sampled row gauges."""

    if episode_ids.ndim != 1 or episode_ids.dtype != torch.long:
        raise ValueError("episode_ids must be long [batch]")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if isinstance(salt, bool) or not isinstance(salt, int):
        raise TypeError("address salt must be an integer")
    permutations: list[torch.Tensor] = []
    for episode_id in episode_ids.detach().to(device="cpu").tolist():
        seed = (int(episode_id) ^ salt) & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutations.append(torch.randperm(capacity, generator=generator))
    return torch.stack(permutations, dim=0).to(device=episode_ids.device)


def validate_episode_permutation(permutation: torch.Tensor, capacity: int) -> None:
    """Reject duplicate, missing or out-of-range address assignments."""

    if permutation.ndim != 2 or permutation.shape[1] != capacity or permutation.dtype != torch.long:
        raise ValueError("address permutation must be long [batch,capacity]")
    expected = torch.arange(capacity, device=permutation.device).expand_as(permutation)
    if not torch.equal(permutation.sort(dim=1).values, expected):
        raise ValueError("every episode address assignment must be a permutation")


def episode_address_codes(
    codebook: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Materialize per-episode row addresses without changing semantic content."""

    if codebook.ndim != 2 or not codebook.is_floating_point():
        raise ValueError("address codebook must be a rank-two floating tensor")
    capacity = codebook.shape[0]
    validate_episode_permutation(permutation, capacity)
    if codebook.device != permutation.device:
        raise ValueError("address codebook and permutation must share one device")
    return codebook[permutation]


@dataclass(frozen=True, slots=True)
class EpisodeAddressState:
    """Serializable routing gauge carried with one chronological episode lane."""

    permutation: torch.Tensor
    codebook_sha256: str

    def __post_init__(self) -> None:
        if self.permutation.ndim != 2 or self.permutation.dtype != torch.long:
            raise ValueError("address state permutation must be long [batch,capacity]")
        validate_episode_permutation(self.permutation, self.permutation.shape[1])
        if (
            not isinstance(self.codebook_sha256, str)
            or len(self.codebook_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.codebook_sha256)
        ):
            raise ValueError("address state requires a lowercase SHA-256 receipt")

    @property
    def batch_size(self) -> int:
        return self.permutation.shape[0]

    @property
    def capacity(self) -> int:
        return self.permutation.shape[1]

    @property
    def device(self) -> torch.device:
        return self.permutation.device

    @property
    def receipt(self) -> str:
        canonical = self.permutation.detach().to(device="cpu").contiguous()
        payload = canonical.numpy().tobytes(order="C")
        metadata = (f"{self.codebook_sha256}:{tuple(canonical.shape)}:{canonical.dtype}").encode()
        return hashlib.sha256(metadata + payload).hexdigest()

    @classmethod
    def from_episode_ids(
        cls,
        *,
        codebook: torch.Tensor,
        episode_ids: torch.Tensor,
    ) -> EpisodeAddressState:
        if codebook.device != episode_ids.device:
            raise ValueError("address codebook and episode IDs must share one device")
        return cls(
            permutation=deterministic_episode_permutation(
                episode_ids,
                codebook.shape[0],
            ),
            codebook_sha256=address_codebook_sha256(codebook),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> EpisodeAddressState:
        return EpisodeAddressState(
            permutation=self.permutation.to(
                device=device,
                non_blocking=non_blocking,
            ),
            codebook_sha256=self.codebook_sha256,
        )

    def index_select(self, indices: torch.Tensor) -> EpisodeAddressState:
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise TypeError("address batch indices must be rank-one long")
        return EpisodeAddressState(
            permutation=self.permutation.index_select(
                0,
                indices.to(self.permutation.device),
            ),
            codebook_sha256=self.codebook_sha256,
        )

    def permute_rows(self, row_permutation: torch.Tensor) -> EpisodeAddressState:
        if (
            row_permutation.ndim != 1
            or row_permutation.dtype != torch.long
            or row_permutation.shape[0] != self.capacity
        ):
            raise ValueError("address row permutation must contain one long index per row")
        expected = torch.arange(self.capacity, device=row_permutation.device)
        if not torch.equal(row_permutation.sort().values, expected):
            raise ValueError("address row permutation must contain every row exactly once")
        return EpisodeAddressState(
            permutation=self.permutation.index_select(
                1,
                row_permutation.to(self.permutation.device),
            ),
            codebook_sha256=self.codebook_sha256,
        )

    def same_assignment(self, other: EpisodeAddressState) -> bool:
        if not isinstance(other, EpisodeAddressState):
            return False
        return self.codebook_sha256 == other.codebook_sha256 and torch.equal(
            self.permutation,
            other.permutation.to(self.permutation.device),
        )

    def materialize(self, codebook: torch.Tensor) -> torch.Tensor:
        if address_codebook_sha256(codebook) != self.codebook_sha256:
            raise ValueError("address state was created for another immutable codebook")
        return episode_address_codes(codebook, self.permutation)
