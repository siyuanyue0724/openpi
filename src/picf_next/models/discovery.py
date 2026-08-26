"""Task-independent set discovery over complete multimodal binding features.

The repeated prediction, query self-attention and set-loss structure follows
the set-decoder semantics of
`facebookresearch/Mask2Former@9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`.
The competitive object read follows the inverted-attention equation in
`google-research/google-research@95e3a1da2d27cb9c8289f6fd3076cfed608c3c94`
`slot_attention/model.py`: tokens first compete across object queries plus
context, then each object column is normalized across valid tokens. This is a
clean-room PyTorch implementation: no upstream code is copied.

Transient discovery-query indices are not persistent identities. The module
emits uncertain current-frame observations for the separate probabilistic
association and posterior update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.geometry import PhysicalGeometryContract


def _inverse_softplus(value: float) -> float:
    """Return the numerically stable inverse of softplus for a positive scalar."""

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse softplus requires a finite positive value")
    return value + math.log(-math.expm1(-value))


@dataclass(frozen=True, slots=True)
class ObjectExistenceCalibration:
    """Invert the class weighting used to train discovery existence logits.

    The set criterion assigns weight ``w_-`` to complete-inventory unmatched
    queries.  At the population optimum its sigmoid output ``q`` is therefore
    a cost-sensitive score, not the physical posterior ``p``.  Their odds obey

    ``p / (1 - p) = w_- * q / (1 - q)``.

    Keeping this value with the discovery output gives training, association
    and lifecycle update one probability contract.  It is not a learned gate
    or an extra birth prior.
    """

    unmatched_query_weight: float = 0.1

    def __post_init__(self) -> None:
        value = self.unmatched_query_weight
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise ValueError("unmatched_query_weight must be finite and positive")

    def posterior_logit(self, training_logit: torch.Tensor) -> torch.Tensor:
        """Return the calibrated physical-object posterior logit."""

        return training_logit + math.log(self.unmatched_query_weight)

    def posterior_probability(self, training_logit: torch.Tensor) -> torch.Tensor:
        if training_logit.dtype in {torch.float16, torch.bfloat16}:
            training_logit = training_logit.float()
        calibrated = self.posterior_logit(training_logit)
        return torch.sigmoid(calibrated)

    @property
    def training_probability_at_half_posterior(self) -> float:
        """Raw sigmoid boundary corresponding to physical probability 0.5."""

        return 1.0 / (1.0 + self.unmatched_query_weight)

    @property
    def training_logit_at_half_posterior(self) -> float:
        """Raw logit whose calibrated physical-object posterior is neutral."""

        return -math.log(self.unmatched_query_weight)


@dataclass(frozen=True, slots=True)
class ObjectDiscoveryConfig:
    input_dim: int
    hidden_dim: int
    num_queries: int
    num_layers: int
    num_heads: int
    address_dim: int
    content_dim: int
    geometry_dim: int
    geometry_contract: PhysicalGeometryContract
    initial_variance: float
    ffn_multiplier: int = 4
    dropout: float = 0.0
    minimum_variance: float = 1e-4
    existence_calibration: ObjectExistenceCalibration = field(
        default_factory=ObjectExistenceCalibration
    )

    def __post_init__(self) -> None:
        positive = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_queries": self.num_queries,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "address_dim": self.address_dim,
            "content_dim": self.content_dim,
            "geometry_dim": self.geometry_dim,
            "ffn_multiplier": self.ffn_multiplier,
        }
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.geometry_contract, PhysicalGeometryContract):
            raise TypeError("discovery geometry requires a physical geometry contract")
        if self.geometry_contract.dimension != self.geometry_dim:
            raise ValueError("discovery geometry width differs from its physical contract")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if isinstance(self.dropout, bool) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            isinstance(self.minimum_variance, bool)
            or not math.isfinite(self.minimum_variance)
            or self.minimum_variance <= 0.0
        ):
            raise ValueError("minimum_variance must be finite and positive")
        if (
            isinstance(self.initial_variance, bool)
            or not math.isfinite(self.initial_variance)
            or self.initial_variance <= self.minimum_variance
        ):
            raise ValueError("initial_variance must exceed minimum_variance")
        if not isinstance(self.existence_calibration, ObjectExistenceCalibration):
            raise TypeError("existence_calibration must use ObjectExistenceCalibration")

    @property
    def dynamic_dim(self) -> int:
        return self.content_dim + self.geometry_dim


@dataclass(frozen=True, slots=True)
class ObjectDiscoveryOutput:
    query_features: torch.Tensor
    address_mean: torch.Tensor
    content_mean: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_variance: torch.Tensor
    geometry_contract: PhysicalGeometryContract
    existence_logits: torch.Tensor
    localization_confidence_logits: torch.Tensor
    ownership_logits: torch.Tensor
    ownership: torch.Tensor
    token_valid: torch.Tensor
    token_group_id: torch.Tensor
    evidence_available: torch.Tensor
    existence_calibration: ObjectExistenceCalibration
    auxiliary_outputs: tuple[ObjectDiscoveryOutput, ...] = ()

    @property
    def observation_mean(self) -> torch.Tensor:
        return torch.cat(
            (self.address_mean, self.content_mean, self.geometry_mean),
            dim=-1,
        )

    @property
    def existence(self) -> torch.Tensor:
        """Calibrated probability that a query is a physical object."""

        return self.existence_calibration.posterior_probability(self.existence_logits)

    @property
    def training_existence_score(self) -> torch.Tensor:
        """Cost-sensitive sigmoid score optimized by the weighted set BCE."""

        return torch.sigmoid(self.existence_logits)

    @property
    def localization_confidence(self) -> torch.Tensor:
        """Conditional expected fidelity of the query's spatial support.

        Object existence and localization quality are distinct random
        variables.  A coherent but misplaced ownership mask can have high
        internal mask confidence while still being an unsafe measurement for
        the persistent filter.  This value is trained with a proper soft-label
        Bernoulli score whose detached target is matched soft IoU, so its Bayes
        optimum is conditional expected overlap.  It never participates in
        target matching.
        """

        logits = self.localization_confidence_logits
        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()
        return torch.sigmoid(logits)

    @property
    def measurement_probability(self) -> torch.Tensor:
        """Return quality-weighted physical-observation probability mass.

        If an independent uniform variable accepts the observation whenever it
        is below the realized support IoU, this product is exactly the joint
        Bernoulli probability of a physical and support-valid observation.
        """

        return self.existence.float() * self.localization_confidence

    @property
    def mask_quality(self) -> torch.Tensor:
        """Return Mask2Former-style conditional mask confidence per query.

        Ownership is one categorical simplex over queries plus context.  The
        object-vs-context conditional probability is therefore exactly
        ``sigmoid(object_logit - context_logit)``.  Average that probability on
        the query's positive-odds support; a query with no supported token has
        zero mask quality.
        """

        relative_logits = self.ownership_logits[..., :-1] - self.ownership_logits[..., -1:]
        supported = (relative_logits > 0.0) & self.token_valid.unsqueeze(-1)
        conditional = torch.sigmoid(relative_logits.float())
        numerator = (conditional * supported).sum(dim=1)
        denominator = supported.sum(dim=1)
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1),
            torch.zeros_like(numerator),
        )

    @property
    def object_confidence(self) -> torch.Tensor:
        """Alias for the runtime-valid object measurement probability."""

        return self.measurement_probability

    @property
    def mask_coherence_score(self) -> torch.Tensor:
        """Diagnostic-only existence times self-reported mask coherence.

        Unlike :attr:`measurement_probability`, this score has no correctness
        supervision and must not drive filtering.  It is retained to compare
        against historical Mask2Former-style confidence diagnostics.
        """

        return self.existence.float() * self.mask_quality

    @property
    def context_ownership(self) -> torch.Tensor:
        return self.ownership[..., -1]


def _normalized_competitive_ownership(
    ownership: torch.Tensor,
    token_valid: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize competitive object ownership over valid tokens per query."""

    if ownership.ndim != 3:
        raise ValueError("ownership must be batch-by-token-by-category")
    if token_valid.dtype != torch.bool or token_valid.shape != ownership.shape[:2]:
        raise ValueError("token_valid must be bool batch-by-token")
    if ownership.shape[-1] < 2:
        raise ValueError("ownership must contain object queries plus context")
    if ownership.device != token_valid.device:
        raise ValueError("ownership and token_valid must share a device")
    if not torch.is_floating_point(ownership):
        raise TypeError("ownership must use a floating dtype")
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    weights = ownership[..., :-1].float()
    weights = torch.where(
        token_valid.unsqueeze(-1),
        weights + epsilon,
        torch.zeros_like(weights),
    )
    denominator = weights.sum(dim=1, keepdim=True)
    return torch.where(
        denominator > 0.0,
        weights / denominator.clamp_min(epsilon),
        torch.zeros_like(weights),
    )


class _CompetitiveCrossRead(nn.Module):
    """Read one exclusive token partition into a permutation-equivariant set."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        ownership: torch.Tensor,
    ) -> torch.Tensor:
        weights = _normalized_competitive_ownership(ownership, memory_valid)
        values = self.value_projection(memory)
        update = torch.einsum(
            "bnk,bnh->bkh",
            weights.to(values.dtype),
            values,
        )
        return self.output_projection(update)


class _SetDecoderLayer(nn.Module):
    """Competitive evidence read, set interaction and feed-forward update."""

    def __init__(self, config: ObjectDiscoveryConfig) -> None:
        super().__init__()
        kwargs = {
            "embed_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "dropout": config.dropout,
            "batch_first": True,
        }
        self.cross_read = _CompetitiveCrossRead(config.hidden_dim)
        self.self_norm = nn.LayerNorm(config.hidden_dim)
        self.self_attention = nn.MultiheadAttention(**kwargs)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        ffn_dim = config.hidden_dim * config.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.hidden_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        ownership: torch.Tensor,
    ) -> torch.Tensor:
        if memory.shape[1] > 0:
            active = memory_valid.any(dim=1)
            update = self.cross_read(
                memory,
                memory_valid,
                ownership,
            )
            queries = queries + self.dropout(update * active[:, None, None])

        normalized = self.self_norm(queries)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class TaskIndependentObjectDiscovery(nn.Module):
    """Amortized current-frame object observation model.

    `binding_features` contains one comparison-space vector for every retained
    native evidence token. This module neither receives task language nor
    deletes/mutates tokens. Invalid padding participates only in the context
    category and a sample with no evidence cannot emit a birth candidate.
    """

    def __init__(
        self,
        config: ObjectDiscoveryConfig,
        *,
        validate_tensor_values: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")
        self.config = config
        self.validate_tensor_values = validate_tensor_values
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.query_embeddings = nn.Parameter(torch.empty(config.num_queries, config.hidden_dim))
        self.layers = nn.ModuleList([_SetDecoderLayer(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.address_head = nn.Linear(config.hidden_dim, config.address_dim)
        self.content_head = nn.Linear(config.hidden_dim, config.content_dim)
        self.geometry_head = nn.Linear(config.hidden_dim, config.geometry_dim)
        # Address and content are deterministic descriptors. Diagonal Gaussian
        # uncertainty is defined only for calibrated physical geometry. Keep a
        # Linear-shaped parameter container for checkpoint compatibility, but
        # production observation covariance is an axis-wise constant: the M2
        # conditional-residual control rejected query-conditioned variance.
        self.variance_head = nn.Linear(config.hidden_dim, config.geometry_dim)
        self.existence_head = nn.Linear(config.hidden_dim, 1)
        self.localization_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.ownership_token = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ownership_query = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.context_head = nn.Linear(config.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_embeddings, std=self.config.hidden_dim**-0.5)
        # The unmatched-query weight makes the raw BCE score cost-sensitive.
        # Start at the exact neutral physical posterior, and keep a random
        # classifier from reshaping the shared representation on update one.
        nn.init.zeros_(self.existence_head.weight)
        nn.init.constant_(
            self.existence_head.bias,
            self.config.existence_calibration.training_logit_at_half_posterior,
        )
        # A conditional probability of one half is the neutral reliability
        # prior.  Zero initialization also prevents an untrained quality head
        # from reshaping the shared query representation on the first update.
        nn.init.zeros_(self.localization_confidence_head.weight)
        nn.init.zeros_(self.localization_confidence_head.bias)
        nn.init.zeros_(self.context_head.bias)
        # Start physical regression at the declared chart origin. A random
        # coordinate must not reshape the shared representation on update one.
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        nn.init.zeros_(self.variance_head.weight)
        nn.init.constant_(
            self.variance_head.bias,
            _inverse_softplus(self.config.initial_variance - self.config.minimum_variance),
        )
        self.variance_head.weight.requires_grad_(False)

    def _validate(
        self,
        binding_features: torch.Tensor,
        token_valid: torch.Tensor,
        token_group_id: torch.Tensor,
    ) -> None:
        if binding_features.ndim != 3:
            raise ValueError("binding_features must be batch-by-token-by-feature")
        if binding_features.shape[-1] != self.config.input_dim:
            raise ValueError("binding feature width differs from discovery config")
        if token_valid.dtype != torch.bool or token_valid.shape != binding_features.shape[:2]:
            raise ValueError("token_valid must be a bool batch-by-token tensor")
        if token_valid.device != binding_features.device:
            raise ValueError("token_valid and binding_features must share a device")
        if not torch.is_floating_point(binding_features):
            raise ValueError("binding_features must use a floating dtype")
        if self.validate_tensor_values:
            if not torch.isfinite(binding_features).all():
                raise ValueError("binding_features contains NaN or infinity")
            if (binding_features[~token_valid] != 0.0).any():
                raise ValueError("invalid binding-feature padding must be exactly zero")
        if token_group_id.shape != token_valid.shape or token_group_id.dtype != torch.long:
            raise ValueError("token_group_id must be a long batch-by-token tensor")
        if token_group_id.device != binding_features.device:
            raise ValueError("token_group_id and binding_features must share a device")
        if self.validate_tensor_values:
            if (token_group_id[~token_valid] != -1).any():
                raise ValueError("invalid tokens must use group ID -1")
            if (token_group_id < -1).any():
                raise ValueError("group IDs must be -1 or nonnegative")

    @staticmethod
    def _tie_group_logits(
        logits: torch.Tensor,
        token_group_id: torch.Tensor,
    ) -> torch.Tensor:
        token_count = logits.shape[1]
        if token_count == 0:
            return logits

        # Stable sorting makes equal arbitrary group IDs contiguous. Segment
        # reduction then ties each group without data-dependent Python loops or
        # tensor-to-host scalar reads. Ungrouped (-1) tokens are restored
        # individually and therefore never tie to one another.
        sorted_groups, order = torch.sort(token_group_id, dim=1, stable=True)
        feature_count = logits.shape[-1]
        sorted_logits = logits.gather(
            1,
            order.unsqueeze(-1).expand(-1, -1, feature_count),
        )
        starts = torch.ones_like(sorted_groups, dtype=torch.bool)
        starts[:, 1:] = sorted_groups[:, 1:] != sorted_groups[:, :-1]
        segment = starts.long().cumsum(dim=1) - 1
        grouped = sorted_groups >= 0
        sums = logits.new_zeros(logits.shape[0], token_count, feature_count)
        sums.scatter_add_(
            1,
            segment.unsqueeze(-1).expand(-1, -1, feature_count),
            sorted_logits * grouped.unsqueeze(-1),
        )
        counts = logits.new_zeros(logits.shape[0], token_count, 1)
        counts.scatter_add_(1, segment.unsqueeze(-1), grouped.unsqueeze(-1).to(logits.dtype))
        means = sums.gather(
            1,
            segment.unsqueeze(-1).expand(-1, -1, feature_count),
        ) / counts.gather(1, segment.unsqueeze(-1)).clamp_min(1.0)
        tied_sorted = torch.where(grouped.unsqueeze(-1), means, sorted_logits)
        tied = torch.empty_like(logits)
        return tied.scatter(
            1,
            order.unsqueeze(-1).expand(-1, -1, feature_count),
            tied_sorted,
        )

    def _predict(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        token_valid: torch.Tensor,
        token_group_id: torch.Tensor,
    ) -> ObjectDiscoveryOutput:
        # CUDA autocast intentionally evaluates LayerNorm in float32. Restore the
        # declared evidence compute dtype at this boundary so every observation
        # field can be compared with and written into one typed posterior state.
        # The following linear heads were already autocast to this dtype, so this
        # makes the state contract explicit without changing their effective math.
        queries = self.output_norm(queries).to(memory.dtype)
        address_raw = self.address_head(queries)
        address = F.normalize(address_raw.float(), dim=-1).to(address_raw.dtype)
        content = self.content_head(queries)
        geometry = self.geometry_head(queries)

        # Observation variance is a train-only calibrated constant for each
        # physical axis. It is deliberately independent of query identity and
        # current features: the preregistered residual-permutation control found
        # no conditional reliability signal. The unused frozen weight remains
        # solely so older checkpoints retain an exact state-dict schema.
        variance_raw = (
            self.variance_head.bias.view(1, 1, -1)
            .expand(
                queries.shape[0],
                queries.shape[1],
                -1,
            )
            .float()
        )
        variance = (F.softplus(variance_raw) + self.config.minimum_variance).to(queries.dtype)

        existence_logits = self.existence_head(queries).squeeze(-1)
        localization_confidence_logits = self.localization_confidence_head(queries).squeeze(-1)
        evidence_available = token_valid.any(dim=1)
        existence_logits = existence_logits.masked_fill(
            ~evidence_available.unsqueeze(1),
            torch.finfo(existence_logits.dtype).min,
        )

        token_keys = self.ownership_token(memory)
        query_keys = self.ownership_query(queries)
        object_logits = torch.einsum("bnh,bkh->bnk", token_keys, query_keys)
        object_logits = object_logits / math.sqrt(self.config.hidden_dim)
        context_logits = self.context_head(memory)
        ownership_logits = torch.cat((object_logits, context_logits), dim=-1)
        ownership_logits = self._tie_group_logits(ownership_logits, token_group_id)
        invalid = ~token_valid
        ownership_logits = ownership_logits.masked_fill(
            invalid.unsqueeze(-1),
            torch.finfo(ownership_logits.dtype).min,
        )
        ownership_logits[..., -1] = torch.where(
            invalid,
            torch.zeros_like(ownership_logits[..., -1]),
            ownership_logits[..., -1],
        )
        ownership = torch.softmax(ownership_logits.float(), dim=-1).to(ownership_logits.dtype)

        return ObjectDiscoveryOutput(
            query_features=queries,
            address_mean=address,
            content_mean=content,
            geometry_mean=geometry,
            geometry_variance=variance,
            geometry_contract=self.config.geometry_contract,
            existence_logits=existence_logits,
            localization_confidence_logits=localization_confidence_logits,
            ownership_logits=ownership_logits,
            ownership=ownership,
            token_valid=token_valid,
            token_group_id=token_group_id,
            evidence_available=evidence_available,
            existence_calibration=self.config.existence_calibration,
        )

    def forward(
        self,
        binding_features: torch.Tensor,
        token_valid: torch.Tensor,
        token_group_id: torch.Tensor | None = None,
    ) -> ObjectDiscoveryOutput:
        if token_group_id is None:
            token_group_id = torch.full_like(token_valid, -1, dtype=torch.long)
        self._validate(binding_features, token_valid, token_group_id)
        batch_size = binding_features.shape[0]
        memory = self.input_projection(self.input_norm(binding_features))
        memory = memory * token_valid.unsqueeze(-1)
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        predictions = [self._predict(queries, memory, token_valid, token_group_id)]
        for layer in self.layers:
            queries = layer(
                queries,
                memory,
                token_valid,
                predictions[-1].ownership,
            )
            predictions.append(self._predict(queries, memory, token_valid, token_group_id))
        return replace(
            predictions[-1],
            auxiliary_outputs=tuple(predictions[:-1]),
        )
