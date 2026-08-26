"""Exact, architecture-aware summaries for paired Qwen gradient audits."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from picf_next.contracts import ContractError

_LANGUAGE_PREFIX = "model.qwenvl_with_expert.qwenvl.model.language_model."
_VISUAL_MERGER_PREFIX = "model.qwenvl_with_expert.qwenvl.model.visual.merger."
_LANGUAGE_LAYER = re.compile(r"^layers\.(\d+)\.")


@dataclass(frozen=True, slots=True)
class GradientPairMoments:
    """Global moments for one parameter under two isolated objectives."""

    dot: float
    lattice8_squared_norm: float
    lattice14_squared_norm: float
    elements: int

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.dot,
                    self.lattice8_squared_norm,
                    self.lattice14_squared_norm,
                )
            )
            or self.lattice8_squared_norm < 0.0
            or self.lattice14_squared_norm < 0.0
            or isinstance(self.elements, bool)
            or not isinstance(self.elements, int)
            or self.elements <= 0
        ):
            raise ContractError("paired gradient moments are invalid")


@dataclass(frozen=True, slots=True)
class WeightedGradientPairMoments:
    """Global moments for two named objectives with an explicit loss mixture."""

    dot: float
    first_squared_norm: float
    second_squared_norm: float
    elements: int

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.dot,
                    self.first_squared_norm,
                    self.second_squared_norm,
                )
            )
            or self.first_squared_norm < 0.0
            or self.second_squared_norm < 0.0
            or isinstance(self.elements, bool)
            or not isinstance(self.elements, int)
            or self.elements <= 0
        ):
            raise ContractError("weighted gradient pair moments are invalid")


@dataclass(frozen=True, slots=True)
class WeightedGradientTripleMoments:
    """Global Gram-matrix moments for three named objective gradients."""

    first_squared_norm: float
    second_squared_norm: float
    third_squared_norm: float
    first_second_dot: float
    first_third_dot: float
    second_third_dot: float
    elements: int

    def __post_init__(self) -> None:
        values = (
            self.first_squared_norm,
            self.second_squared_norm,
            self.third_squared_norm,
            self.first_second_dot,
            self.first_third_dot,
            self.second_third_dot,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or any(value < 0.0 for value in values[:3])
            or isinstance(self.elements, bool)
            or not isinstance(self.elements, int)
            or self.elements <= 0
        ):
            raise ContractError("weighted gradient triple moments are invalid")
        squared = values[:3]
        dots = values[3:]
        for left, right, dot in zip((0, 0, 1), (1, 2, 2), dots, strict=True):
            product = squared[left] * squared[right]
            tolerance = 1e-8 * max(product, dot * dot, 1e-300)
            if dot * dot > product + tolerance:
                raise ContractError("weighted gradient triple violates Cauchy-Schwarz")
        first, second, third = squared
        first_second, first_third, second_third = dots
        determinant_terms = (
            first * second * third,
            2.0 * first_second * first_third * second_third,
            -first * second_third**2,
            -second * first_third**2,
            -third * first_second**2,
        )
        determinant = math.fsum(determinant_terms)
        tolerance = 1e-8 * max(*(abs(value) for value in determinant_terms), 1e-300)
        if determinant < -tolerance:
            raise ContractError("weighted gradient triple Gram matrix is not positive semidefinite")


def qwen_gradient_group(parameter_name: str) -> str:
    """Map every trainable native-Qwen parameter to one declared depth group."""

    if not isinstance(parameter_name, str) or not parameter_name:
        raise ContractError("Qwen gradient parameter name must be nonempty text")
    if parameter_name.startswith(_VISUAL_MERGER_PREFIX):
        return "visual_merger"
    if not parameter_name.startswith(_LANGUAGE_PREFIX):
        raise ContractError(f"undeclared native-Qwen gradient parameter: {parameter_name}")
    suffix = parameter_name.removeprefix(_LANGUAGE_PREFIX)
    match = _LANGUAGE_LAYER.match(suffix)
    if match is not None:
        return f"language_layer_{int(match.group(1)):02d}"
    if suffix == "embed_tokens.weight":
        return "language_embedding_tied_lm_head"
    if suffix.startswith("norm."):
        return "language_final_norm"
    raise ContractError(f"undeclared native-Qwen language parameter: {parameter_name}")


def _alignment_summary(
    moments: tuple[GradientPairMoments, ...],
) -> dict[str, float | int | bool | None]:
    dot = sum(moment.dot for moment in moments)
    lattice8_squared = sum(moment.lattice8_squared_norm for moment in moments)
    lattice14_squared = sum(moment.lattice14_squared_norm for moment in moments)
    lattice8_norm = math.sqrt(lattice8_squared)
    lattice14_norm = math.sqrt(lattice14_squared)
    denominator = lattice8_norm * lattice14_norm
    cosine = None if denominator == 0.0 else max(-1.0, min(1.0, dot / denominator))
    absolute_dot_mass = sum(abs(moment.dot) for moment in moments)
    negative_dot_mass = sum(max(-moment.dot, 0.0) for moment in moments)
    mean_descent_on_lattice8 = 0.5 * (lattice8_squared + dot)
    mean_descent_on_lattice14 = 0.5 * (lattice14_squared + dot)
    return {
        "cosine": cosine,
        "dot_product": dot,
        "element_count": sum(moment.elements for moment in moments),
        "lattice8_norm": lattice8_norm,
        "lattice14_norm": lattice14_norm,
        "lattice14_to_lattice8_norm_ratio": (
            None if lattice8_norm == 0.0 else lattice14_norm / lattice8_norm
        ),
        "mean_gradient_descends_lattice8": mean_descent_on_lattice8 > 0.0,
        "mean_gradient_descends_lattice14": mean_descent_on_lattice14 > 0.0,
        "mean_gradient_lattice8_directional_inner_product": mean_descent_on_lattice8,
        "mean_gradient_lattice14_directional_inner_product": mean_descent_on_lattice14,
        "parameter_tensor_count": len(moments),
        "parameter_tensor_negative_dot_count": sum(moment.dot < 0.0 for moment in moments),
        "parameter_tensor_negative_dot_mass_fraction": (
            0.0 if absolute_dot_mass == 0.0 else negative_dot_mass / absolute_dot_mass
        ),
    }


def summarize_qwen_gradient_alignment(
    parameter_moments: Mapping[str, GradientPairMoments],
) -> dict[str, object]:
    """Summarize exact global parameter moments without changing gradients."""

    if not parameter_moments:
        raise ContractError("Qwen gradient alignment has no parameter moments")
    grouped: dict[str, list[GradientPairMoments]] = {}
    ordered_moments = []
    for name in sorted(parameter_moments):
        moment = parameter_moments[name]
        if not isinstance(moment, GradientPairMoments):
            raise ContractError("Qwen gradient alignment contains an invalid moment")
        grouped.setdefault(qwen_gradient_group(name), []).append(moment)
        ordered_moments.append(moment)
    if "visual_merger" not in grouped or "language_embedding_tied_lm_head" not in grouped:
        raise ContractError("Qwen gradient alignment omits a required architecture group")
    return {
        "global": _alignment_summary(tuple(ordered_moments)),
        "groups": {name: _alignment_summary(tuple(grouped[name])) for name in sorted(grouped)},
        "parameter_count": len(parameter_moments),
    }


def _require_objective_name(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ContractError(f"weighted gradient {field} must be nonempty text")
    return value


def _require_positive_weight(value: float, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"weighted gradient {field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ContractError(f"weighted gradient {field} must be finite and positive")
    return numeric


def _weighted_alignment_summary(
    moments: tuple[WeightedGradientPairMoments, ...],
    *,
    first_weight: float,
    second_weight: float,
) -> dict[str, float | int | bool | None]:
    dot = sum(moment.dot for moment in moments)
    first_squared = sum(moment.first_squared_norm for moment in moments)
    second_squared = sum(moment.second_squared_norm for moment in moments)
    first_norm = math.sqrt(first_squared)
    second_norm = math.sqrt(second_squared)
    denominator = first_norm * second_norm
    cosine = None if denominator == 0.0 else max(-1.0, min(1.0, dot / denominator))
    absolute_dot_mass = sum(abs(moment.dot) for moment in moments)
    negative_dot_mass = sum(max(-moment.dot, 0.0) for moment in moments)
    first_directional_inner_product = first_weight * first_squared + second_weight * dot
    second_directional_inner_product = first_weight * dot + second_weight * second_squared
    first_term = first_weight * first_weight * first_squared
    cross_term = 2.0 * first_weight * second_weight * dot
    second_term = second_weight * second_weight * second_squared
    mixed_squared_norm = first_term + cross_term + second_term
    cancellation_tolerance = 1e-12 * (abs(first_term) + abs(cross_term) + abs(second_term))
    if mixed_squared_norm < 0.0 and mixed_squared_norm >= -cancellation_tolerance:
        mixed_squared_norm = 0.0
    if mixed_squared_norm < 0.0:
        raise ContractError("weighted mixed gradient norm is mathematically invalid")
    return {
        "cosine": cosine,
        "dot_product": dot,
        "element_count": sum(moment.elements for moment in moments),
        "first_gradient_norm": first_norm,
        "second_gradient_norm": second_norm,
        "second_to_first_norm_ratio": (None if first_norm == 0.0 else second_norm / first_norm),
        "mixed_gradient_norm": math.sqrt(mixed_squared_norm),
        "mixed_gradient_descends_first_objective": first_directional_inner_product > 0.0,
        "mixed_gradient_descends_second_objective": second_directional_inner_product > 0.0,
        "mixed_gradient_first_directional_inner_product": first_directional_inner_product,
        "mixed_gradient_second_directional_inner_product": second_directional_inner_product,
        "parameter_tensor_count": len(moments),
        "parameter_tensor_negative_dot_count": sum(moment.dot < 0.0 for moment in moments),
        "parameter_tensor_negative_dot_mass_fraction": (
            0.0 if absolute_dot_mass == 0.0 else negative_dot_mass / absolute_dot_mass
        ),
    }


def summarize_weighted_qwen_gradient_alignment(
    parameter_moments: Mapping[str, WeightedGradientPairMoments],
    *,
    first_objective: str,
    second_objective: str,
    first_weight: float,
    second_weight: float,
) -> dict[str, object]:
    """Summarize the exact direction of one weighted two-objective Qwen update."""

    if not parameter_moments:
        raise ContractError("weighted Qwen gradient alignment has no parameter moments")
    first_name = _require_objective_name(first_objective, field="first objective")
    second_name = _require_objective_name(second_objective, field="second objective")
    if first_name == second_name:
        raise ContractError("weighted gradient objectives must be distinct")
    first_factor = _require_positive_weight(first_weight, field="first weight")
    second_factor = _require_positive_weight(second_weight, field="second weight")
    grouped: dict[str, list[WeightedGradientPairMoments]] = {}
    ordered_moments = []
    for name in sorted(parameter_moments):
        moment = parameter_moments[name]
        if not isinstance(moment, WeightedGradientPairMoments):
            raise ContractError("weighted Qwen gradient alignment contains an invalid moment")
        grouped.setdefault(qwen_gradient_group(name), []).append(moment)
        ordered_moments.append(moment)
    if "visual_merger" not in grouped or "language_embedding_tied_lm_head" not in grouped:
        raise ContractError("weighted Qwen gradient alignment omits a required architecture group")
    return {
        "first_objective": first_name,
        "first_weight": first_factor,
        "global": _weighted_alignment_summary(
            tuple(ordered_moments),
            first_weight=first_factor,
            second_weight=second_factor,
        ),
        "groups": {
            name: _weighted_alignment_summary(
                tuple(grouped[name]),
                first_weight=first_factor,
                second_weight=second_factor,
            )
            for name in sorted(grouped)
        },
        "parameter_count": len(parameter_moments),
        "second_objective": second_name,
        "second_weight": second_factor,
    }


def _weighted_triple_summary(
    moments: tuple[WeightedGradientTripleMoments, ...],
    *,
    objective_names: tuple[str, str, str],
    weights: tuple[float, float, float],
) -> dict[str, object]:
    squared = tuple(
        math.fsum(getattr(moment, field) for moment in moments)
        for field in (
            "first_squared_norm",
            "second_squared_norm",
            "third_squared_norm",
        )
    )
    dots = tuple(
        math.fsum(getattr(moment, field) for moment in moments)
        for field in (
            "first_second_dot",
            "first_third_dot",
            "second_third_dot",
        )
    )
    norms = tuple(math.sqrt(value) for value in squared)
    pair_indices = ((0, 1), (0, 2), (1, 2))
    pair_names = tuple(
        f"{objective_names[left]}__{objective_names[right]}" for left, right in pair_indices
    )
    pairwise_cosines = {}
    for name, dot, (left, right) in zip(pair_names, dots, pair_indices, strict=True):
        denominator = norms[left] * norms[right]
        pairwise_cosines[name] = (
            None if denominator == 0.0 else max(-1.0, min(1.0, dot / denominator))
        )

    first_weight, second_weight, third_weight = weights
    first_second_dot, first_third_dot, second_third_dot = dots
    directional = (
        first_weight * squared[0]
        + second_weight * first_second_dot
        + third_weight * first_third_dot,
        first_weight * first_second_dot
        + second_weight * squared[1]
        + third_weight * second_third_dot,
        first_weight * first_third_dot
        + second_weight * second_third_dot
        + third_weight * squared[2],
    )
    mixed_terms = (
        first_weight**2 * squared[0],
        second_weight**2 * squared[1],
        third_weight**2 * squared[2],
        2.0 * first_weight * second_weight * first_second_dot,
        2.0 * first_weight * third_weight * first_third_dot,
        2.0 * second_weight * third_weight * second_third_dot,
    )
    mixed_squared_norm = math.fsum(mixed_terms)
    cancellation_tolerance = 1e-12 * math.fsum(abs(value) for value in mixed_terms)
    if mixed_squared_norm < 0.0 and mixed_squared_norm >= -cancellation_tolerance:
        mixed_squared_norm = 0.0
    if mixed_squared_norm < 0.0:
        raise ContractError("weighted triple mixed gradient norm is mathematically invalid")

    negative_counts = {}
    negative_mass_fractions = {}
    for name, field in zip(
        pair_names,
        ("first_second_dot", "first_third_dot", "second_third_dot"),
        strict=True,
    ):
        values = tuple(getattr(moment, field) for moment in moments)
        absolute_mass = math.fsum(abs(value) for value in values)
        negative_mass = math.fsum(max(-value, 0.0) for value in values)
        negative_counts[name] = sum(value < 0.0 for value in values)
        negative_mass_fractions[name] = (
            0.0 if absolute_mass == 0.0 else negative_mass / absolute_mass
        )
    return {
        "element_count": sum(moment.elements for moment in moments),
        "gradient_norms": dict(zip(objective_names, norms, strict=True)),
        "gradient_squared_norms": dict(zip(objective_names, squared, strict=True)),
        "mixed_gradient_descends": {
            name: value > 0.0 for name, value in zip(objective_names, directional, strict=True)
        },
        "mixed_gradient_directional_inner_products": dict(
            zip(objective_names, directional, strict=True)
        ),
        "mixed_gradient_norm": math.sqrt(mixed_squared_norm),
        "pairwise_cosines": pairwise_cosines,
        "pairwise_dot_products": dict(zip(pair_names, dots, strict=True)),
        "parameter_tensor_count": len(moments),
        "parameter_tensor_pair_negative_dot_counts": negative_counts,
        "parameter_tensor_pair_negative_dot_mass_fractions": negative_mass_fractions,
    }


def summarize_weighted_qwen_gradient_triple(
    parameter_moments: Mapping[str, WeightedGradientTripleMoments],
    *,
    objective_names: tuple[str, str, str],
    weights: tuple[float, float, float],
) -> dict[str, object]:
    """Summarize an exact three-objective Qwen update from its Gram matrix."""

    if not parameter_moments:
        raise ContractError("weighted Qwen gradient triple has no parameter moments")
    if (
        not isinstance(objective_names, tuple)
        or len(objective_names) != 3
        or len(set(objective_names)) != 3
    ):
        raise ContractError("weighted gradient triple requires three distinct objectives")
    names = cast(
        tuple[str, str, str],
        tuple(
            _require_objective_name(value, field=f"objective {index}")
            for index, value in enumerate(objective_names)
        ),
    )
    if not isinstance(weights, tuple) or len(weights) != 3:
        raise ContractError("weighted gradient triple requires three weights")
    factors = cast(
        tuple[float, float, float],
        tuple(
            _require_positive_weight(value, field=f"weight {index}")
            for index, value in enumerate(weights)
        ),
    )
    grouped: dict[str, list[WeightedGradientTripleMoments]] = {}
    ordered_moments = []
    for parameter_name in sorted(parameter_moments):
        moment = parameter_moments[parameter_name]
        if not isinstance(moment, WeightedGradientTripleMoments):
            raise ContractError("weighted Qwen gradient triple contains an invalid moment")
        grouped.setdefault(qwen_gradient_group(parameter_name), []).append(moment)
        ordered_moments.append(moment)
    if "visual_merger" not in grouped or "language_embedding_tied_lm_head" not in grouped:
        raise ContractError("weighted Qwen gradient triple omits a required architecture group")
    return {
        "global": _weighted_triple_summary(
            tuple(ordered_moments),
            objective_names=names,
            weights=factors,
        ),
        "groups": {
            group_name: _weighted_triple_summary(
                tuple(grouped[group_name]),
                objective_names=names,
                weights=factors,
            )
            for group_name in sorted(grouped)
        },
        "objective_weights": dict(zip(names, factors, strict=True)),
        "parameter_count": len(parameter_moments),
    }
