"""Learned action-conditioned prediction for one probabilistic object filter.

This module implements the neural terms inside one probabilistic object filter;
it is not a second recurrent state. The action-conditioned prediction pattern
is adapted from the one-step V-JEPA 2-AC transition at commit
``204698b45b3712590f06245fbfba32d3be539812`` (MIT). Multi-step target
chronology is deliberately not inherited from that revision: the corrected
prefix/action/suffix alignment is cross-checked against JEPA-WMs commit
``13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`` (CC-BY-NC-4.0) and is implemented
only in the separate loss-side geometry overshooting criterion. Explicit control-period
conditioning is principle-only from Time-Aware World
Model commit ``ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c``; that repository has no
license file at the pinned revision, so no source code or custom integration
formula is copied.

Query or row indices never enter positional embeddings, preserving set
permutation equivariance. Marginal association and correlated-evidence
correction live together in :mod:`picf_next.models.marginal` and ``filter``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.geometry import PhysicalGeometryContract


@dataclass(frozen=True, slots=True)
class TemporalFilterConfig:
    address_dim: int
    content_dim: int
    geometry_dim: int
    geometry_contract: PhysicalGeometryContract
    action_dim: int
    reference_delta_t_s: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    ffn_multiplier: int = 4
    dropout: float = 0.0
    minimum_variance: float = 1e-6
    initial_process_variance: float = 1e-4
    initial_survival_probability: float = 0.995
    initial_detection_probability: float = 0.85
    # The initial random-finite-set prior and the transition birth process are
    # distinct distributions.  An empty bank must be able to initialize the
    # visible scene, while an occupied bank should prefer explaining evidence
    # with persistent rows over spawning duplicate identities.
    empty_bank_birth_to_clutter_prior_odds: float = 1.0
    recurrent_birth_to_clutter_prior_odds: float = 0.1
    # The persistent-address log likelihood ratio is the same spherical
    # relation used by temporal address binding. The sigmoid training loss
    # applies its sampling-prior correction internally; runtime consumes this
    # prior-free LLR directly. The recipe rejects any parameter mismatch.
    association_address_temperature: float = 0.1
    association_address_logit_bias: float = -2.71

    def __post_init__(self) -> None:
        dimensions = (
            self.address_dim,
            self.content_dim,
            self.geometry_dim,
            self.action_dim,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.ffn_multiplier,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in dimensions
        ):
            raise ValueError("all temporal filter dimensions must be positive")
        if not isinstance(self.geometry_contract, PhysicalGeometryContract):
            raise TypeError("temporal geometry requires a physical geometry contract")
        if self.geometry_contract.dimension != self.geometry_dim:
            raise ValueError("temporal geometry width differs from its physical contract")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (
            isinstance(self.reference_delta_t_s, bool)
            or not math.isfinite(self.reference_delta_t_s)
            or self.reference_delta_t_s <= 0.0
        ):
            raise ValueError("reference_delta_t_s must be finite and positive")
        if isinstance(self.dropout, bool) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            isinstance(self.minimum_variance, bool)
            or not math.isfinite(self.minimum_variance)
            or self.minimum_variance <= 0.0
        ):
            raise ValueError("minimum_variance must be positive")
        if (
            isinstance(self.initial_process_variance, bool)
            or not math.isfinite(self.initial_process_variance)
            or self.initial_process_variance <= self.minimum_variance
        ):
            raise ValueError("initial_process_variance must exceed minimum_variance")
        for name, value in (
            ("initial_survival_probability", self.initial_survival_probability),
            ("initial_detection_probability", self.initial_detection_probability),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        for name, value in (
            (
                "empty_bank_birth_to_clutter_prior_odds",
                self.empty_bank_birth_to_clutter_prior_odds,
            ),
            (
                "recurrent_birth_to_clutter_prior_odds",
                self.recurrent_birth_to_clutter_prior_odds,
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.association_address_temperature, bool)
            or not math.isfinite(self.association_address_temperature)
            or self.association_address_temperature <= 0.0
        ):
            raise ValueError("association_address_temperature must be finite and positive")
        if isinstance(self.association_address_logit_bias, bool) or not math.isfinite(
            self.association_address_logit_bias
        ):
            raise ValueError("association_address_logit_bias must be finite")

    @property
    def state_dim(self) -> int:
        return self.address_dim + self.content_dim + self.geometry_dim

    @property
    def dynamic_dim(self) -> int:
        return self.content_dim + self.geometry_dim


@dataclass(frozen=True, slots=True)
class ObjectBeliefBatch:
    address_mean: torch.Tensor
    content_mean: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_covariance_diag: torch.Tensor
    existence_logits: torch.Tensor
    visibility_given_existence_logits: torch.Tensor
    measurement_age_s: torch.Tensor
    valid: torch.Tensor
    age: torch.Tensor

    @property
    def state_mean(self) -> torch.Tensor:
        return torch.cat((self.address_mean, self.content_mean, self.geometry_mean), dim=-1)

    @property
    def dynamic_mean(self) -> torch.Tensor:
        return torch.cat((self.content_mean, self.geometry_mean), dim=-1)

    @property
    def existence(self) -> torch.Tensor:
        logits = self.existence_logits
        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()
        return torch.sigmoid(logits) * self.valid

    @property
    def visibility(self) -> torch.Tensor:
        logits = self.visibility_given_existence_logits
        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()
        conditional = torch.sigmoid(logits)
        return self.existence * conditional


def empty_object_belief(
    config: TemporalFilterConfig,
    *,
    batch_size: int,
    capacity: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ObjectBeliefBatch:
    """Construct the unique empty posterior state for a new sequence.

    Empty rows are exactly zero, including covariance. The configured
    covariance floor applies only after a row is occupied; this distinction is
    required by :func:`_validate_belief` and avoids fabricating uncertainty for
    nonexistent objects.
    """

    dimensions = (batch_size, capacity)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions
    ):
        raise ValueError("batch_size and capacity must be positive")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not dtype.is_floating_point:
        raise ValueError("empty posterior dtype must be floating point")
    factory = {"device": device, "dtype": dtype}
    return ObjectBeliefBatch(
        address_mean=torch.zeros(batch_size, capacity, config.address_dim, **factory),
        content_mean=torch.zeros(batch_size, capacity, config.content_dim, **factory),
        geometry_mean=torch.zeros(batch_size, capacity, config.geometry_dim, **factory),
        geometry_covariance_diag=torch.zeros(
            batch_size,
            capacity,
            config.geometry_dim,
            **factory,
        ),
        existence_logits=torch.zeros(batch_size, capacity, **factory),
        visibility_given_existence_logits=torch.zeros(batch_size, capacity, **factory),
        measurement_age_s=torch.zeros(batch_size, capacity, **factory),
        valid=torch.zeros(batch_size, capacity, dtype=torch.bool, device=device),
        age=torch.zeros(batch_size, capacity, dtype=torch.long, device=device),
    )


@dataclass(frozen=True, slots=True)
class ObjectPredictionOutput:
    belief: ObjectBeliefBatch
    dynamic_delta: torch.Tensor
    process_variance: torch.Tensor
    survival_logits: torch.Tensor
    detectability_if_detected_logits: torch.Tensor
    detectability_if_missed_logits: torch.Tensor
    conditional_detection_logits: torch.Tensor

    @property
    def survival_probability(self) -> torch.Tensor:
        logits = self.survival_logits
        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()
        return torch.sigmoid(logits) * self.belief.valid

    @property
    def conditional_detection_probability(self) -> torch.Tensor:
        """Return the fresh P(detected_t | exists_t, transition context)."""

        logits = self.conditional_detection_logits
        if logits.dtype in {torch.float16, torch.bfloat16}:
            logits = logits.float()
        return torch.sigmoid(logits) * self.belief.valid


class _ResidualSetTransitionLayer(nn.Module):
    def __init__(self, config: TemporalFilterConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
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
        state: torch.Tensor,
        valid: torch.Tensor,
        existence: torch.Tensor,
    ) -> torch.Tensor:
        # The transition integrates interactions under the current
        # multi-Bernoulli posterior. A component contributes as a key in
        # proportion to its existence probability; tentative hypotheses can
        # retain identity without influencing certain objects as full peers.
        # Empty samples receive one zero sentinel key and their update is gated
        # out, avoiding a per-layer CUDA synchronization.
        if state.shape[1] > 0:
            active = valid.any(dim=1)
            minimum = torch.finfo(state.dtype).min
            probability_floor = torch.finfo(torch.float32).eps
            log_prior = torch.where(
                valid,
                existence.float().clamp_min(probability_floor).log().to(state.dtype),
                torch.full_like(existence, minimum, dtype=state.dtype),
            )
            log_prior[:, 0] = torch.where(
                active,
                log_prior[:, 0],
                torch.zeros_like(log_prior[:, 0]),
            )
            batch_size, capacity = valid.shape
            attention_mask = (
                log_prior[:, None, None, :]
                .expand(batch_size, self.num_heads, capacity, capacity)
                .reshape(batch_size * self.num_heads, capacity, capacity)
            )
            normalized = self.attention_norm(state)
            update, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=attention_mask,
                need_weights=False,
            )
            state = state + self.dropout(update * active[:, None, None])
        state = state + self.dropout(self.ffn(self.ffn_norm(state)))
        return state * valid.unsqueeze(-1)


class ActionConditionedObjectTransition(nn.Module):
    """Permutation-equivariant residual prediction over a persistent object set."""

    def __init__(
        self,
        config: TemporalFilterConfig,
        *,
        validate_tensor_values: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")
        self.config = config
        self.validate_tensor_values = validate_tensor_values
        # Physical dynamics receive physical state, uncertainty and existence.
        # Previous detectability is deliberately excluded: it is the discrete
        # state mixed by the observation Markov kernel below, not a second input
        # to both unrestricted branches. Integer age remains telemetry only.
        input_dim = config.state_dim + config.geometry_dim + 1
        self.state_projection = nn.Linear(input_dim, config.hidden_dim)
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.time_projection = nn.Linear(2, config.hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            [_ResidualSetTransitionLayer(config) for _ in range(config.num_layers)]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.dynamic_head = nn.Linear(config.hidden_dim, config.dynamic_dim)
        self.process_variance_head = nn.Linear(config.hidden_dim, config.geometry_dim)
        self.survival_head = nn.Linear(config.hidden_dim, 1)
        self.detectability_if_detected_head = nn.Linear(config.hidden_dim, 1)
        self.detectability_if_missed_head = nn.Linear(config.hidden_dim, 1)
        self.missed_duration_logit_slope = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.dynamic_head.weight, std=1e-3)
        nn.init.zeros_(self.dynamic_head.bias)
        nn.init.normal_(self.process_variance_head.weight, std=1e-3)
        process_raw = _inverse_softplus(
            self.config.initial_process_variance - self.config.minimum_variance
        )
        nn.init.constant_(self.process_variance_head.bias, process_raw)
        nn.init.normal_(self.survival_head.weight, std=1e-3)
        nn.init.constant_(
            self.survival_head.bias,
            _probability_logit(self.config.initial_survival_probability),
        )
        for head in (
            self.detectability_if_detected_head,
            self.detectability_if_missed_head,
        ):
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.constant_(
                head.bias,
                _probability_logit(self.config.initial_detection_probability),
            )
        nn.init.zeros_(self.missed_duration_logit_slope)

    def forward(
        self,
        prior: ObjectBeliefBatch,
        previous_executed_action: torch.Tensor,
        delta_t_s: torch.Tensor,
    ) -> ObjectPredictionOutput:
        _validate_belief(
            prior,
            self.config,
            validate_values=self.validate_tensor_values,
        )
        batch_size, capacity = prior.valid.shape
        if not isinstance(previous_executed_action, torch.Tensor) or (
            previous_executed_action.shape != (batch_size, self.config.action_dim)
        ):
            raise ValueError("previous_executed_action must be batch-by-action-dimension")
        if (
            previous_executed_action.device != prior.address_mean.device
            or previous_executed_action.dtype != prior.address_mean.dtype
            or not torch.is_floating_point(previous_executed_action)
        ):
            raise ValueError(
                "previous_executed_action must match the belief's floating dtype and device"
            )
        if self.validate_tensor_values and not torch.isfinite(previous_executed_action).all():
            raise ValueError("previous_executed_action contains NaN or infinity")
        if (
            not isinstance(delta_t_s, torch.Tensor)
            or delta_t_s.shape != (batch_size,)
            or delta_t_s.device != prior.address_mean.device
            or delta_t_s.dtype != prior.address_mean.dtype
            or not torch.is_floating_point(delta_t_s)
        ):
            raise ValueError("delta_t_s must be one floating colocated value per sample")
        if self.validate_tensor_values and (
            not torch.isfinite(delta_t_s).all() or (delta_t_s <= 0.0).any()
        ):
            raise ValueError("delta_t_s must be finite and positive")

        valid = prior.valid
        state_features = torch.cat(
            (
                prior.state_mean,
                torch.log(
                    prior.geometry_covariance_diag.float().clamp_min(self.config.minimum_variance)
                ).to(prior.address_mean.dtype),
                prior.existence.to(prior.address_mean.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        hidden = self.state_projection(state_features)
        hidden = hidden + self.action_projection(previous_executed_action).unsqueeze(1)
        relative_time = delta_t_s / self.config.reference_delta_t_s
        time_features = torch.stack((relative_time - 1.0, relative_time.log()), dim=-1)
        hidden = hidden + self.time_projection(time_features).unsqueeze(1)
        hidden = hidden * valid.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, valid, prior.existence)
        hidden = self.output_norm(hidden) * valid.unsqueeze(-1)

        dynamic_delta = self.dynamic_head(hidden) * valid.unsqueeze(-1)
        process_variance_raw = self.process_variance_head(hidden)
        process_variance = (
            F.softplus(process_variance_raw.float()) + self.config.minimum_variance
        ) * valid.unsqueeze(-1)
        survival_logits = self.survival_head(hidden).squeeze(-1)
        survival = torch.sigmoid(survival_logits.float()) * valid
        detectability_if_detected_logits = self.detectability_if_detected_head(hidden).squeeze(-1)
        detectability_if_missed_logits = self.detectability_if_missed_head(hidden).squeeze(-1)
        missed_duration = torch.log1p(
            prior.measurement_age_s.float() / self.config.reference_delta_t_s
        )
        detectability_if_missed_logits = detectability_if_missed_logits.float() + (
            self.missed_duration_logit_slope.float() * missed_duration
        )

        predicted_dynamic = prior.dynamic_mean + dynamic_delta
        covariance = (prior.geometry_covariance_diag.float() + process_variance) * valid.unsqueeze(
            -1
        )
        existence = prior.existence * survival
        # One conditional-detectability HMM is part of this same posterior. The
        # two branches are trained directly from adjacent loss-only labels; the
        # previous posterior probability appears only in this external
        # Chapman-Kolmogorov mixture. It cannot enter either branch context.
        previous_detectability = torch.sigmoid(prior.visibility_given_existence_logits.float())
        detectability_if_detected = torch.sigmoid(detectability_if_detected_logits.float())
        detectability_if_missed = torch.sigmoid(detectability_if_missed_logits.float())
        conditional_detection = (
            previous_detectability * detectability_if_detected
            + (1.0 - previous_detectability) * detectability_if_missed
        )
        conditional_detection_logits = _safe_logit(conditional_detection)
        visibility_logits = conditional_detection_logits.to(
            prior.visibility_given_existence_logits.dtype
        )
        existence_logits = _safe_logit(existence).to(prior.existence_logits.dtype)
        process_variance = process_variance.to(prior.geometry_covariance_diag.dtype)
        covariance = covariance.to(prior.geometry_covariance_diag.dtype)
        existence_logits = torch.where(valid, existence_logits, torch.zeros_like(existence_logits))
        visibility_logits = torch.where(
            valid, visibility_logits, torch.zeros_like(visibility_logits)
        )
        survival_logits = torch.where(
            valid,
            survival_logits,
            torch.zeros_like(survival_logits),
        )
        detectability_if_detected_logits = torch.where(
            valid,
            detectability_if_detected_logits,
            torch.zeros_like(detectability_if_detected_logits),
        )
        detectability_if_missed_logits = torch.where(
            valid,
            detectability_if_missed_logits,
            torch.zeros_like(detectability_if_missed_logits),
        )
        conditional_detection_logits = torch.where(
            valid,
            conditional_detection_logits,
            torch.zeros_like(conditional_detection_logits),
        )
        age = torch.where(valid, prior.age + 1, torch.zeros_like(prior.age))
        measurement_age_s = torch.where(
            valid,
            prior.measurement_age_s + delta_t_s.unsqueeze(1),
            torch.zeros_like(prior.measurement_age_s),
        )

        content_end = self.config.content_dim
        belief = ObjectBeliefBatch(
            address_mean=prior.address_mean,
            content_mean=predicted_dynamic[..., :content_end] * valid.unsqueeze(-1),
            geometry_mean=predicted_dynamic[..., content_end:] * valid.unsqueeze(-1),
            geometry_covariance_diag=covariance,
            existence_logits=existence_logits,
            visibility_given_existence_logits=visibility_logits,
            measurement_age_s=measurement_age_s,
            valid=valid,
            age=age,
        )
        return ObjectPredictionOutput(
            belief=belief,
            dynamic_delta=dynamic_delta,
            process_variance=process_variance,
            survival_logits=survival_logits,
            detectability_if_detected_logits=detectability_if_detected_logits,
            detectability_if_missed_logits=detectability_if_missed_logits,
            conditional_detection_logits=conditional_detection_logits,
        )


def _validate_belief(
    belief: ObjectBeliefBatch,
    config: TemporalFilterConfig,
    *,
    validate_values: bool = True,
) -> None:
    if belief.address_mean.ndim != 3:
        raise ValueError("belief tensors must be batch-by-object-by-feature")
    batch_size, capacity = belief.address_mean.shape[:2]
    expected = {
        "address_mean": (batch_size, capacity, config.address_dim),
        "content_mean": (batch_size, capacity, config.content_dim),
        "geometry_mean": (batch_size, capacity, config.geometry_dim),
        "geometry_covariance_diag": (batch_size, capacity, config.geometry_dim),
        "existence_logits": (batch_size, capacity),
        "visibility_given_existence_logits": (batch_size, capacity),
        "measurement_age_s": (batch_size, capacity),
        "valid": (batch_size, capacity),
        "age": (batch_size, capacity),
    }
    for name, shape in expected.items():
        value = getattr(belief, name)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if belief.valid.dtype != torch.bool or belief.age.dtype != torch.long:
        raise ValueError("valid must be bool and age must be long")
    floating = (
        belief.address_mean,
        belief.content_mean,
        belief.geometry_mean,
        belief.geometry_covariance_diag,
        belief.existence_logits,
        belief.visibility_given_existence_logits,
        belief.measurement_age_s,
    )
    device = belief.address_mean.device
    dtype = belief.address_mean.dtype
    if any(
        value.device != device or value.dtype != dtype or not torch.is_floating_point(value)
        for value in floating
    ):
        raise ValueError("belief floating tensors must share one floating dtype and device")
    if belief.valid.device != device or belief.age.device != device:
        raise ValueError("belief metadata must be colocated")
    if validate_values:
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("belief contains NaN or infinity")
        if (
            belief.geometry_covariance_diag < config.minimum_variance * belief.valid.unsqueeze(-1)
        ).any():
            raise ValueError("valid belief covariance is below the configured minimum")
        if belief.valid.any():
            address_norm = torch.linalg.vector_norm(
                belief.address_mean[belief.valid].float(), dim=-1
            )
            tolerance = max(1e-5, torch.finfo(belief.address_mean.dtype).eps)
            if not torch.allclose(
                address_norm,
                torch.ones_like(address_norm),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError("valid belief addresses must have unit norm")
        if (belief.age < 0).any():
            raise ValueError("belief age must be nonnegative")
        if (belief.measurement_age_s < 0.0).any():
            raise ValueError("belief measurement age must be nonnegative")
        for value in floating:
            if (value[~belief.valid] != 0.0).any():
                raise ValueError("invalid belief rows must be exactly zero")
        if (belief.age[~belief.valid] != 0).any():
            raise ValueError("invalid belief age must be exactly zero")


def _safe_logit(probability: torch.Tensor) -> torch.Tensor:
    control_probability = probability.float()
    return torch.logit(control_probability.clamp(min=1e-6, max=1.0 - 1e-6))


def _probability_logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))
