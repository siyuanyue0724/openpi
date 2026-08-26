"""The three-family ADR-74 objective, with no hidden rescue terms."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from picf_next.objective import (
    ObjectiveSupport,
    ObjectiveTerm,
    UnifiedObjective,
    combine_objective,
    normalized_scalar_term,
)


@dataclass(frozen=True, slots=True)
class NativeObjectiveConfig:
    """Manifest-frozen family weights for the strict candidate graph."""

    predictive_weight: float
    structural_weight: float
    action_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("action_weight", self.action_weight),
            ("predictive_weight", self.predictive_weight),
            ("structural_weight", self.structural_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NativePredictiveNormalizationEntry:
    """One route denominator frozen before sequential branch execution."""

    name: str
    denominator: float
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("predictive normalization entry name must be non-empty")
        if not math.isfinite(self.denominator) or self.denominator < 0:
            raise ValueError("predictive normalization denominator must be finite and non-negative")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("predictive normalization weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NativePredictiveNormalizationLedger:
    """Immutable union support for mathematically exact sequential backward."""

    entries: tuple[NativePredictiveNormalizationEntry, ...]
    active_weight: float

    def __post_init__(self) -> None:
        names = tuple(entry.name for entry in self.entries)
        if len(set(names)) != len(names):
            raise ValueError("predictive normalization ledger names must be unique")
        expected = sum(entry.weight for entry in self.entries if entry.denominator > 0)
        if not math.isfinite(self.active_weight) or self.active_weight < 0:
            raise ValueError("predictive normalization active weight must be finite")
        if not math.isclose(self.active_weight, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("predictive normalization active weight differs from its entries")

    def entry_by_name(self) -> dict[str, NativePredictiveNormalizationEntry]:
        return {entry.name: entry for entry in self.entries}


@dataclass(frozen=True, slots=True)
class NativeSequentialBranchObjective:
    """One additive branch of a union-normalized native objective."""

    total: torch.Tensor
    family_terms: dict[str, torch.Tensor]


def build_native_predictive_normalization_ledger(
    supports: Sequence[ObjectiveSupport],
) -> NativePredictiveNormalizationLedger:
    """Freeze denominators after assignment but before the deferred branch forward."""

    grouped: dict[str, list[ObjectiveSupport]] = defaultdict(list)
    for support in supports:
        if not isinstance(support, ObjectiveSupport):
            raise TypeError("predictive normalization requires ObjectiveSupport values")
        grouped[support.name].append(support)
    entries: list[NativePredictiveNormalizationEntry] = []
    for name, values in grouped.items():
        reference = values[0]
        if any(value.weight != reference.weight for value in values[1:]):
            raise ValueError(f"predictive support {name!r} has inconsistent weights")
        entries.append(
            NativePredictiveNormalizationEntry(
                name=name,
                denominator=sum(value.denominator() for value in values),
                weight=reference.weight,
            )
        )
    ordered = tuple(sorted(entries, key=lambda entry: entry.name))
    return NativePredictiveNormalizationLedger(
        entries=ordered,
        active_weight=sum(entry.weight for entry in ordered if entry.denominator > 0),
    )


def _objective_term_numerator(term: ObjectiveTerm) -> torch.Tensor:
    if term.sample_weight is None:
        return term.values.masked_fill(~term.valid, 0).sum()
    effective_weight = term.sample_weight.masked_fill(~term.valid, 0)
    return (term.values * effective_weight).sum()


def native_predictive_sequential_branch_mean(
    terms: Sequence[ObjectiveTerm],
    *,
    ledger: NativePredictiveNormalizationLedger,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return one branch's additive share of the union-normalized family mean."""

    if not isinstance(ledger, NativePredictiveNormalizationLedger):
        raise TypeError("sequential predictive reduction requires its frozen ledger")
    zero = reference.sum() * 0
    entries = ledger.entry_by_name()
    grouped: dict[str, list[ObjectiveTerm]] = defaultdict(list)
    for term in terms:
        if not isinstance(term, ObjectiveTerm):
            raise TypeError("sequential predictive reduction requires ObjectiveTerm values")
        try:
            entry = entries[term.name]
        except KeyError as error:
            raise ValueError(
                f"predictive branch term {term.name!r} is absent from its ledger"
            ) from error
        if term.weight != entry.weight:
            raise ValueError(f"predictive branch term {term.name!r} changed its ledger weight")
        grouped[term.name].append(term)
    numerator = zero
    for name, branch_terms in grouped.items():
        entry = entries[name]
        branch_denominator = sum(term.support().denominator() for term in branch_terms)
        if branch_denominator > entry.denominator + 1e-12:
            raise ValueError(f"predictive branch term {name!r} exceeds its ledger support")
        route_numerator = sum(
            (_objective_term_numerator(term) for term in branch_terms),
            zero,
        )
        # Keep every declared route connected to autograd even when this
        # rank has zero valid mass. FSDP ranks may observe different entities,
        # but they must traverse the same sharded parameter graph and issue
        # collectives in the same order. A zero coefficient yields exactly
        # zero gradient while preserving that graph participation.
        coefficient = (
            0.0
            if entry.denominator == 0
            else entry.weight / entry.denominator
        )
        numerator = numerator + route_numerator * coefficient
    if ledger.active_weight == 0:
        return numerator
    return numerator / ledger.active_weight


def combine_native_sequential_branch(
    *,
    official_policy_loss: torch.Tensor,
    action_scale: float,
    predictive_terms: Sequence[ObjectiveTerm],
    structural_terms: Sequence[ObjectiveTerm],
    predictive_ledger: NativePredictiveNormalizationLedger,
    config: NativeObjectiveConfig,
) -> NativeSequentialBranchObjective:
    """Compose one exact additive branch of a two-forward native objective."""

    if (
        not isinstance(official_policy_loss, torch.Tensor)
        or official_policy_loss.ndim != 0
        or not official_policy_loss.is_floating_point()
        or not torch.isfinite(official_policy_loss)
    ):
        raise ValueError("a sequential branch requires one finite policy loss scalar")
    if (
        isinstance(action_scale, bool)
        or not isinstance(action_scale, (int, float))
        or not math.isfinite(action_scale)
        or action_scale < 0
    ):
        raise ValueError("sequential action scale must be finite and non-negative")
    reference = official_policy_loss
    predictive = native_predictive_sequential_branch_mean(
        predictive_terms,
        ledger=predictive_ledger,
        reference=reference,
    )
    structural_terms_tuple = tuple(structural_terms)
    if structural_terms_tuple:
        components = combine_objective(structural_terms_tuple)
        structural = _active_weighted_family_mean(
            structural_terms_tuple,
            normalized_terms=components.normalized_terms,
            reference=reference,
        )
    else:
        structural = reference.sum() * 0
    family_terms = {
        "action": config.action_weight * action_scale * official_policy_loss,
        "predictive": config.predictive_weight * predictive,
        "structural": config.structural_weight * structural,
    }
    total = sum(family_terms.values(), reference.sum() * 0)
    return NativeSequentialBranchObjective(total=total, family_terms=family_terms)


def merge_repeated_objective_terms(
    terms: Sequence[ObjectiveTerm],
) -> tuple[ObjectiveTerm, ...]:
    """Merge branch-local terms into the same route semantics as one joint graph."""

    grouped: dict[str, list[ObjectiveTerm]] = defaultdict(list)
    for term in terms:
        grouped[term.name].append(term)
    merged: list[ObjectiveTerm] = []
    for name, values in grouped.items():
        reference = values[0]
        if any(value.weight != reference.weight for value in values[1:]):
            raise ValueError(f"repeated objective term {name!r} has inconsistent weights")
        if any(
            value.values.device != reference.values.device
            or value.values.dtype != reference.values.dtype
            for value in values[1:]
        ):
            raise ValueError(f"repeated objective term {name!r} has incompatible tensors")
        if any(
            (value.sample_weight is None) != (reference.sample_weight is None)
            for value in values
        ):
            raise ValueError(f"repeated objective term {name!r} changed weighting semantics")
        merged.append(
            ObjectiveTerm(
                name=name,
                values=torch.cat(tuple(value.values.reshape(-1) for value in values)),
                valid=torch.cat(tuple(value.valid.reshape(-1) for value in values)),
                weight=reference.weight,
                sample_weight=(
                    None
                    if reference.sample_weight is None
                    else torch.cat(
                        tuple(
                            value.sample_weight.reshape(-1)
                            for value in values
                            if value.sample_weight is not None
                        )
                    )
                ),
            )
        )
    return tuple(merged)


def _active_weighted_family_mean(
    terms: tuple[ObjectiveTerm, ...],
    *,
    normalized_terms: dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Average one objective family without making its scale route-count dependent."""

    zero = reference.sum() * 0
    if not terms:
        return zero
    numerator = sum(
        (normalized_terms[term.name] * term.weight for term in terms),
        zero,
    )

    def _has_effective_mass(term: ObjectiveTerm) -> torch.Tensor:
        if term.sample_weight is None:
            return term.valid.any()
        return term.sample_weight.masked_select(term.valid).sum() > 0

    active_weight = torch.stack(
        tuple(
            _has_effective_mass(term).to(device=reference.device, dtype=reference.dtype)
            * term.weight
            for term in terms
        )
    ).sum()
    return numerator / active_weight.clamp_min(torch.finfo(reference.dtype).tiny)


def combine_native_objective(
    *,
    official_policy_loss: torch.Tensor | None,
    predictive_terms: tuple[ObjectiveTerm, ...],
    structural_terms: tuple[ObjectiveTerm, ...],
    config: NativeObjectiveConfig,
) -> UnifiedObjective:
    """Combine only action, shared predictive state and structural set losses.

    ``official_policy_loss`` is LingBot's targetless base-policy loss: released
    flow-action loss plus its mandatory MoE router regularizers.  Optional
    depth/video teacher losses are disabled and verified outside this function.
    A modality with no valid target supplies either no term or a zero-valid term;
    it never contributes a placeholder denominator.
    """

    if not isinstance(config, NativeObjectiveConfig):
        raise TypeError("the native objective requires a frozen NativeObjectiveConfig")
    if config.action_weight > 0:
        if (
            not isinstance(official_policy_loss, torch.Tensor)
            or official_policy_loss.ndim != 0
            or not official_policy_loss.is_floating_point()
            or not torch.isfinite(official_policy_loss)
        ):
            raise ValueError("an active action family requires one finite policy loss scalar")
    elif official_policy_loss is not None:
        raise ValueError("an inactive action family requires an absent policy loss")
    predictive_prefixes = (
        "xmod/",
        "correction/",
        "filter_prior/",
        "filter_posterior/",
        "rollout/",
    )
    if any(not term.name.startswith(predictive_prefixes) for term in predictive_terms):
        raise ValueError(
            "native predictive terms must use xmod/, correction/, filter_prior/, "
            "filter_posterior/, or rollout/ names"
        )
    if any(not term.name.startswith("set/") for term in structural_terms):
        raise ValueError("native structural terms must use set/ names")
    action_term = (
        None
        if official_policy_loss is None
        else normalized_scalar_term("action", official_policy_loss)
    )
    terms = (
        *((action_term,) if action_term is not None else ()),
        *predictive_terms,
        *structural_terms,
    )
    if not terms:
        raise ValueError("native objective requires at least one active loss term")
    components = combine_objective(terms)
    reference = official_policy_loss if official_policy_loss is not None else terms[0].values
    predictive = _active_weighted_family_mean(
        predictive_terms,
        normalized_terms=components.normalized_terms,
        reference=reference,
    )
    structural = _active_weighted_family_mean(
        structural_terms,
        normalized_terms=components.normalized_terms,
        reference=reference,
    )
    action = (
        reference.sum() * 0
        if action_term is None
        else components.normalized_terms[action_term.name]
    )
    total = (
        config.action_weight * action
        + config.predictive_weight * predictive
        + config.structural_weight * structural
    )
    return UnifiedObjective(
        total=total,
        normalized_terms=components.normalized_terms,
        valid_counts=components.valid_counts,
        family_terms={
            "action": config.action_weight * action,
            "predictive": config.predictive_weight * predictive,
            "structural": config.structural_weight * structural,
        },
    )
