"""The single typed write/read codec for paired PICF belief distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from picf_next.unified.state import UnifiedBeliefState


def _project_symmetric_psd_straight_through(
    matrix: torch.Tensor,
    *,
    floor: float,
) -> torch.Tensor:
    """Project onto the PSD cone without differentiating eigenvectors.

    The Euclidean projection is the identity in the positive-definite interior.
    At repeated or clipped eigenvalues, differentiating ``eigh`` eigenvectors is
    numerically undefined and can emit NaNs.  The straight-through Jacobian picks
    the identity generalized derivative at that boundary while retaining the
    exact constrained projection in the forward pass.
    """

    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    with torch.no_grad():
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric.float())
        projected = eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(floor))
        projected = projected @ eigenvectors.transpose(-1, -2)
    return symmetric + (projected.to(symmetric) - symmetric).detach()


@dataclass(frozen=True, slots=True)
class BeliefCodecConfig:
    content_dim: int
    geometry_dim: int
    uncertainty_dim: int
    host_width: int
    information_floor: float = 1e-5

    def __post_init__(self) -> None:
        dimensions = (
            self.content_dim,
            self.geometry_dim,
            self.uncertainty_dim,
            self.host_width,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("codec dimensions must be integers")
        if min(dimensions) <= 0:
            raise ValueError("codec dimensions must be positive")
        if self.host_width < self.canonical_width:
            raise ValueError(
                f"host_width={self.host_width} cannot hold canonical width={self.canonical_width}"
            )
        if isinstance(self.information_floor, bool) or not isinstance(
            self.information_floor, (int, float)
        ):
            raise TypeError("information_floor must be real-valued")
        if not math.isfinite(self.information_floor) or self.information_floor <= 0:
            raise ValueError("information_floor must be finite and positive")

    @property
    def canonical_width(self) -> int:
        return UnifiedBeliefState.canonical_width(
            content_dim=self.content_dim,
            geometry_dim=self.geometry_dim,
            uncertainty_dim=self.uncertainty_dim,
        )

    @property
    def prediction_width(self) -> int:
        triangle = self.geometry_dim * (self.geometry_dim + 1) // 2
        return self.content_dim + 2 + self.geometry_dim + triangle + self.uncertainty_dim + 2


@dataclass(frozen=True, slots=True)
class PairedBeliefTokens:
    tokens: torch.Tensor
    prior_canonical: torch.Tensor
    posterior_canonical: torch.Tensor
    capacity: int

    @property
    def canonical(self) -> torch.Tensor:
        """Return the two marginal sets for audit and serialization checks."""

        return torch.cat((self.prior_canonical, self.posterior_canonical), dim=1)

    @property
    def paired_canonical(self) -> torch.Tensor:
        """Return exact row-wise ``[prior_k, posterior_k]`` sufficient statistics."""

        return torch.cat((self.prior_canonical, self.posterior_canonical), dim=-1)

    @property
    def prior_tokens(self) -> torch.Tensor:
        return self.tokens[:, : self.capacity]

    @property
    def pair_tokens(self) -> torch.Tensor:
        return self.tokens[:, self.capacity :]


class UnifiedBeliefCodec(nn.Module):
    """One row-wise typed codec, with no attention stack or recurrent state."""

    def __init__(self, config: BeliefCodecConfig) -> None:
        super().__init__()
        self.config = config
        tail_width = config.host_width - config.canonical_width
        self.tail_projection = (
            nn.Linear(config.canonical_width, tail_width, bias=False) if tail_width else None
        )
        self.prediction_projection = nn.Linear(config.host_width, config.prediction_width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.tail_projection is not None:
            nn.init.zeros_(self.tail_projection.weight)
        nn.init.zeros_(self.prediction_projection.weight)
        nn.init.zeros_(self.prediction_projection.bias)
        # The canonical sufficient statistics occupy an exact host subspace.
        # Initialize the write decoder as identity on those coordinates while
        # skipping geometry-valid bits, which are explicit typed inputs.
        canonical_chunks = self._canonical_chunks()
        prediction_chunks = self._prediction_chunks()
        with torch.no_grad():
            for name in prediction_chunks:
                output = prediction_chunks[name]
                source = canonical_chunks[name]
                width = output.stop - output.start
                self.prediction_projection.weight[output, source] = torch.eye(
                    width, dtype=self.prediction_projection.weight.dtype
                )

    def encode(self, state: UnifiedBeliefState) -> torch.Tensor:
        self._validate_dimensions(state)
        reference = self.prediction_projection.weight
        if state.content.device != reference.device:
            raise ValueError("belief state and codec parameters must share one device")
        canonical = state.canonical().to(dtype=reference.dtype)
        if self.tail_projection is None:
            return canonical
        return torch.cat((canonical, self.tail_projection(canonical)), dim=-1)

    def paired_action_tokens(
        self,
        prior: UnifiedBeliefState,
        posterior: UnifiedBeliefState,
    ) -> PairedBeliefTokens:
        """Encode ``K`` prior rows followed by ``K`` exact prior/posterior pairs.

        Keeping a pair in one host token makes the row correspondence directly
        observable to the action stream.  Two independent marginal sets cannot
        preserve that correspondence under an independent posterior
        permutation.
        """

        self._validate_pair(prior, posterior)
        pair_width = 2 * self.config.canonical_width
        if self.config.host_width < pair_width:
            raise ValueError(
                f"host_width={self.config.host_width} cannot hold exact action pair "
                f"width={pair_width}"
            )
        reference = self.prediction_projection.weight
        prior_canonical = prior.canonical().to(dtype=reference.dtype)
        posterior_canonical = posterior.canonical().to(dtype=reference.dtype)
        paired_canonical = torch.cat((prior_canonical, posterior_canonical), dim=-1)
        pair_tail = paired_canonical.new_zeros(
            (*paired_canonical.shape[:-1], self.config.host_width - pair_width)
        )
        pair_tokens = torch.cat((paired_canonical, pair_tail), dim=-1)
        tokens = torch.cat((self.encode(prior), pair_tokens), dim=1)
        return PairedBeliefTokens(
            tokens=tokens,
            prior_canonical=prior_canonical,
            posterior_canonical=posterior_canonical,
            capacity=prior.capacity,
        )

    def decode_prediction(
        self,
        hidden: torch.Tensor,
        *,
        geometry_valid: torch.Tensor,
    ) -> UnifiedBeliefState:
        """Decode a constrained posterior predicted at the state-write boundary."""

        if hidden.ndim != 3 or hidden.shape[-1] != self.config.host_width:
            raise ValueError("hidden must have shape [batch, capacity, host_width]")
        if geometry_valid.shape != (*hidden.shape[:2], self.config.geometry_dim):
            raise ValueError("geometry_valid has the wrong shape")
        if geometry_valid.dtype != torch.bool:
            raise TypeError("geometry_valid must be boolean")
        raw = self.prediction_projection(hidden)
        cursor = 0

        def take(width: int) -> torch.Tensor:
            nonlocal cursor
            value = raw[..., cursor : cursor + width]
            cursor += width
            return value

        content = take(self.config.content_dim).float()
        lifecycle_odds = take(2)
        lifecycle_logits = torch.cat(
            (lifecycle_odds, torch.zeros_like(lifecycle_odds[..., :1])), dim=-1
        )
        lifecycle_log_probs = torch.log_softmax(lifecycle_logits.float(), dim=-1)
        geometry_mean = take(self.config.geometry_dim).float() * geometry_valid
        triangle = self.config.geometry_dim * (self.config.geometry_dim + 1) // 2
        raw_information = take(triangle).float()
        rows, cols = torch.triu_indices(
            self.config.geometry_dim,
            self.config.geometry_dim,
            device=raw.device,
        )
        information = raw_information.new_zeros(
            (*raw.shape[:2], self.config.geometry_dim, self.config.geometry_dim)
        )
        information[..., rows, cols] = raw_information
        information[..., cols, rows] = raw_information
        information = _project_symmetric_psd_straight_through(
            information,
            floor=self.config.information_floor,
        )
        valid_pair = geometry_valid.unsqueeze(-1) & geometry_valid.unsqueeze(-2)
        information = information.masked_fill(~valid_pair, 0)
        log_variance = take(self.config.uncertainty_dim).float().clamp(-20, 20)
        expected_age = take(1).float().squeeze(-1).clamp_min(0)
        evidence_age = take(1).float().squeeze(-1).clamp_min(0)
        if cursor != self.config.prediction_width:
            raise AssertionError("prediction parser did not consume its declared width")
        return UnifiedBeliefState(
            content=content,
            lifecycle_log_probs=lifecycle_log_probs,
            geometry_mean=geometry_mean,
            geometry_information=information,
            geometry_valid=geometry_valid,
            content_log_variance=log_variance,
            expected_age=expected_age,
            evidence_age=evidence_age,
        )

    def _validate_dimensions(self, state: UnifiedBeliefState) -> None:
        actual = (state.content_dim, state.geometry_dim, state.uncertainty_dim)
        expected = (
            self.config.content_dim,
            self.config.geometry_dim,
            self.config.uncertainty_dim,
        )
        if actual != expected:
            raise ValueError(f"belief dimensions {actual} do not match codec {expected}")

    def _validate_pair(
        self,
        prior: UnifiedBeliefState,
        posterior: UnifiedBeliefState,
    ) -> None:
        self._validate_dimensions(prior)
        self._validate_dimensions(posterior)
        if prior.batch_size != posterior.batch_size or prior.capacity != posterior.capacity:
            raise ValueError("prior and posterior batch/capacity must match")

    def _canonical_chunks(self) -> dict[str, slice]:
        cursor = 0

        def chunk(width: int) -> slice:
            nonlocal cursor
            value = slice(cursor, cursor + width)
            cursor += width
            return value

        triangle = self.config.geometry_dim * (self.config.geometry_dim + 1) // 2
        chunks = {
            "content": chunk(self.config.content_dim),
            "lifecycle": chunk(2),
            "geometry_mean": chunk(self.config.geometry_dim),
            "geometry_information": chunk(triangle),
            "geometry_valid": chunk(self.config.geometry_dim),
            "content_log_variance": chunk(self.config.uncertainty_dim),
            "expected_age": chunk(1),
            "evidence_age": chunk(1),
        }
        if cursor != self.config.canonical_width:
            raise AssertionError("canonical chunk map has the wrong width")
        return chunks

    def _prediction_chunks(self) -> dict[str, slice]:
        cursor = 0

        def chunk(width: int) -> slice:
            nonlocal cursor
            value = slice(cursor, cursor + width)
            cursor += width
            return value

        triangle = self.config.geometry_dim * (self.config.geometry_dim + 1) // 2
        chunks = {
            "content": chunk(self.config.content_dim),
            "lifecycle": chunk(2),
            "geometry_mean": chunk(self.config.geometry_dim),
            "geometry_information": chunk(triangle),
            "content_log_variance": chunk(self.config.uncertainty_dim),
            "expected_age": chunk(1),
            "evidence_age": chunk(1),
        }
        if cursor != self.config.prediction_width:
            raise AssertionError("prediction chunk map has the wrong width")
        return chunks
