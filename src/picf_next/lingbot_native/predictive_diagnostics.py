"""Diagnostics for the information content of native predictive targets.

These measurements never contribute a training loss.  They test whether a
target bank has enough variation and cross-frame identity structure to make a
predictive experiment meaningful before expensive model training starts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

PREDICTIVE_TARGET_AUDIT_SCHEMA = "picf-next.lingbot-predictive-target-audit/v2"
PREDICTIVE_TEMPORAL_AUDIT_SCHEMA = "picf-next.lingbot-predictive-temporal-audit/v3"
TEACHER_CAUSALITY_AUDIT_SCHEMA = "picf-next.lingbot-dino-teacher-causality-audit/v4"
PREDICTIVE_TEMPORAL_FEATURE_PAIRING = (
    "separate-offline-invocations/same-pinned-frame-causal-teacher/v1"
)


@dataclass(frozen=True, slots=True)
class PredictiveLatentDiagnostics:
    valid_count: int
    feature_width: int
    identity_count: int
    target_group_count: int
    mean_dimension_variance: float
    effective_rank: float
    effective_rank_ceiling: int
    effective_rank_fraction: float
    zero_template_l1: float
    median_template_l1: float
    retrieval_query_count: int
    identity_top1_accuracy: float | None
    identity_chance_accuracy: float | None
    same_identity_cosine: float | None
    different_identity_cosine: float | None
    same_different_cosine_margin: float | None
    obvious_numerical_collapse: bool

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


PREDICTIVE_LATENT_DIAGNOSTIC_FIELDS = frozenset(
    field.name for field in PredictiveLatentDiagnostics.__dataclass_fields__.values()
)


@dataclass(frozen=True, slots=True)
class PredictiveVisibleSupportDiagnostics:
    """Observed image-area support without pretending it measures amodal visibility."""

    supported_count: int
    sampled_count: int
    minimum_visible_image_fraction: float
    mean_visible_image_fraction: float
    maximum_visible_image_fraction: float
    sampled_p05_visible_image_fraction: float
    sampled_median_visible_image_fraction: float
    sampled_p95_visible_image_fraction: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


PREDICTIVE_VISIBLE_SUPPORT_DIAGNOSTIC_FIELDS = frozenset(
    field.name for field in PredictiveVisibleSupportDiagnostics.__dataclass_fields__.values()
)


@dataclass(frozen=True, slots=True)
class PredictiveTemporalDiagnostics:
    pair_count: int
    feature_width: int
    identity_count: int
    horizon_count: int
    mean_current_future_l1: float
    sampled_median_current_future_l1: float
    sampled_p90_current_future_l1: float
    mean_current_future_cosine: float
    zero_template_l1: float
    median_template_l1: float
    current_copy_advantage_over_median: float
    current_copy_to_zero_ratio: float
    numerically_unchanged_fraction: float
    obvious_no_temporal_content: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


PREDICTIVE_TEMPORAL_DIAGNOSTIC_FIELDS = frozenset(
    field.name for field in PredictiveTemporalDiagnostics.__dataclass_fields__.values()
)


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    measured = float(value)
    if not math.isfinite(measured):
        raise ValueError(f"{name} must be finite")
    return measured


def _optional_finite(value: object, *, name: str) -> float | None:
    return None if value is None else _finite(value, name=name)


def predictive_visible_support_diagnostics_from_mapping(
    value: object,
) -> PredictiveVisibleSupportDiagnostics:
    """Parse visible-image support diagnostics and recompute their invariants."""

    if not isinstance(value, Mapping) or set(value) != PREDICTIVE_VISIBLE_SUPPORT_DIAGNOSTIC_FIELDS:
        raise ValueError("predictive visible-support diagnostic fields differ from schema")
    result = PredictiveVisibleSupportDiagnostics(
        supported_count=_integer(value["supported_count"], name="supported count", minimum=2),
        sampled_count=_integer(value["sampled_count"], name="sampled count", minimum=2),
        minimum_visible_image_fraction=_finite(
            value["minimum_visible_image_fraction"], name="minimum visible image fraction"
        ),
        mean_visible_image_fraction=_finite(
            value["mean_visible_image_fraction"], name="mean visible image fraction"
        ),
        maximum_visible_image_fraction=_finite(
            value["maximum_visible_image_fraction"], name="maximum visible image fraction"
        ),
        sampled_p05_visible_image_fraction=_finite(
            value["sampled_p05_visible_image_fraction"], name="sampled p05 visible image fraction"
        ),
        sampled_median_visible_image_fraction=_finite(
            value["sampled_median_visible_image_fraction"],
            name="sampled median visible image fraction",
        ),
        sampled_p95_visible_image_fraction=_finite(
            value["sampled_p95_visible_image_fraction"], name="sampled p95 visible image fraction"
        ),
    )
    ordered = (
        result.minimum_visible_image_fraction,
        result.sampled_p05_visible_image_fraction,
        result.sampled_median_visible_image_fraction,
        result.sampled_p95_visible_image_fraction,
        result.maximum_visible_image_fraction,
    )
    if (
        result.sampled_count > result.supported_count
        or any(not 0 < item <= 1 for item in ordered)
        or tuple(sorted(ordered)) != ordered
        or not result.minimum_visible_image_fraction
        <= result.mean_visible_image_fraction
        <= result.maximum_visible_image_fraction
    ):
        raise ValueError("predictive visible-support diagnostic values are inconsistent")
    return result


def predictive_latent_diagnostics_from_mapping(
    value: object,
) -> PredictiveLatentDiagnostics:
    """Parse an immutable diagnostic report without trusting JSON scalar types."""

    if not isinstance(value, Mapping) or set(value) != PREDICTIVE_LATENT_DIAGNOSTIC_FIELDS:
        raise ValueError("predictive latent diagnostic fields differ from schema")
    collapse = value["obvious_numerical_collapse"]
    if not isinstance(collapse, bool):
        raise ValueError("predictive collapse flag must be boolean")
    result = PredictiveLatentDiagnostics(
        valid_count=_integer(value["valid_count"], name="valid count", minimum=2),
        feature_width=_integer(value["feature_width"], name="feature width", minimum=2),
        identity_count=_integer(value["identity_count"], name="identity count", minimum=1),
        target_group_count=_integer(
            value["target_group_count"], name="target group count", minimum=1
        ),
        mean_dimension_variance=_finite(
            value["mean_dimension_variance"], name="mean dimension variance"
        ),
        effective_rank=_finite(value["effective_rank"], name="effective rank"),
        effective_rank_ceiling=_integer(
            value["effective_rank_ceiling"], name="effective-rank ceiling", minimum=1
        ),
        effective_rank_fraction=_finite(
            value["effective_rank_fraction"], name="effective-rank fraction"
        ),
        zero_template_l1=_finite(value["zero_template_l1"], name="zero-template loss"),
        median_template_l1=_finite(value["median_template_l1"], name="median-template loss"),
        retrieval_query_count=_integer(
            value["retrieval_query_count"], name="retrieval query count", minimum=0
        ),
        identity_top1_accuracy=_optional_finite(
            value["identity_top1_accuracy"], name="identity top-1 accuracy"
        ),
        identity_chance_accuracy=_optional_finite(
            value["identity_chance_accuracy"], name="identity chance accuracy"
        ),
        same_identity_cosine=_optional_finite(
            value["same_identity_cosine"], name="same-identity cosine"
        ),
        different_identity_cosine=_optional_finite(
            value["different_identity_cosine"], name="different-identity cosine"
        ),
        same_different_cosine_margin=_optional_finite(
            value["same_different_cosine_margin"], name="identity cosine margin"
        ),
        obvious_numerical_collapse=collapse,
    )
    if (
        result.identity_count > result.valid_count
        or result.target_group_count > result.valid_count
        or result.retrieval_query_count > result.valid_count
        or result.effective_rank < 0
        or result.effective_rank > result.effective_rank_ceiling + 1e-4
        or result.effective_rank_fraction < 0
        or result.effective_rank_fraction > 1 + 1e-4
        or result.mean_dimension_variance < 0
        or result.zero_template_l1 < 0
        or result.median_template_l1 < 0
        or not math.isclose(
            result.effective_rank_fraction,
            result.effective_rank / result.effective_rank_ceiling,
            rel_tol=1e-5,
            abs_tol=1e-7,
        )
        or result.obvious_numerical_collapse
        is not (
            result.mean_dimension_variance <= 100 * torch.finfo(torch.float32).eps
            or result.effective_rank <= 1.01
        )
    ):
        raise ValueError("predictive latent diagnostic values are inconsistent")
    accuracies = (result.identity_top1_accuracy, result.identity_chance_accuracy)
    cosines = (result.same_identity_cosine, result.different_identity_cosine)
    if any(value is not None and not 0 <= value <= 1 for value in accuracies):
        raise ValueError("predictive retrieval accuracy lies outside [0,1]")
    if any(value is not None and not -1.0001 <= value <= 1.0001 for value in cosines):
        raise ValueError("predictive cosine lies outside [-1,1]")
    if result.retrieval_query_count == 0:
        if any(value is not None for value in accuracies):
            raise ValueError("empty predictive retrieval has non-empty accuracies")
        expected_margin = (
            None
            if result.same_identity_cosine is None or result.different_identity_cosine is None
            else result.same_identity_cosine - result.different_identity_cosine
        )
        if (expected_margin is None and result.same_different_cosine_margin is not None) or (
            expected_margin is not None
            and (
                result.same_different_cosine_margin is None
                or not math.isclose(
                    result.same_different_cosine_margin,
                    expected_margin,
                    rel_tol=1e-5,
                    abs_tol=1e-7,
                )
            )
        ):
            raise ValueError("predictive cosine margin is inconsistent")
    else:
        if any(value is None for value in (*accuracies, *cosines)):
            raise ValueError("predictive retrieval omitted required metrics")
        if (
            result.same_different_cosine_margin is None
            or result.same_identity_cosine is None
            or result.different_identity_cosine is None
            or not math.isclose(
                result.same_different_cosine_margin,
                result.same_identity_cosine - result.different_identity_cosine,
                rel_tol=1e-5,
                abs_tol=1e-7,
            )
        ):
            raise ValueError("predictive cosine margin is inconsistent")
    return result


def predictive_target_pretraining_readiness(
    diagnostics: PredictiveLatentDiagnostics,
) -> tuple[bool, tuple[str, ...]]:
    """Apply only directional, scale-free target checks before model loading."""

    if not isinstance(diagnostics, PredictiveLatentDiagnostics):
        raise TypeError("predictive readiness requires typed diagnostics")
    failures: list[str] = []
    if diagnostics.obvious_numerical_collapse:
        failures.append("obvious_numerical_collapse")
    if diagnostics.identity_count < 2 or diagnostics.target_group_count < 2:
        failures.append("insufficient_identity_or_target_group_support")
    if diagnostics.retrieval_query_count <= 0:
        failures.append("cross_frame_identity_retrieval_unavailable")
    else:
        top1 = diagnostics.identity_top1_accuracy
        chance = diagnostics.identity_chance_accuracy
        margin = diagnostics.same_different_cosine_margin
        if top1 is None or chance is None or top1 <= chance:
            failures.append("identity_top1_not_above_chance")
        if margin is None or margin <= 0:
            failures.append("same_identity_cosine_not_above_negative")
    return not failures, tuple(failures)


def predictive_temporal_diagnostics_from_mapping(
    value: object,
) -> PredictiveTemporalDiagnostics:
    """Parse temporal-content diagnostics and recompute all dependent fields."""

    if not isinstance(value, Mapping) or set(value) != PREDICTIVE_TEMPORAL_DIAGNOSTIC_FIELDS:
        raise ValueError("predictive temporal diagnostic fields differ from schema")
    no_content = value["obvious_no_temporal_content"]
    if not isinstance(no_content, bool):
        raise ValueError("predictive temporal-content flag must be boolean")
    result = PredictiveTemporalDiagnostics(
        pair_count=_integer(value["pair_count"], name="temporal pair count", minimum=2),
        feature_width=_integer(value["feature_width"], name="temporal feature width", minimum=2),
        identity_count=_integer(value["identity_count"], name="temporal identity count", minimum=1),
        horizon_count=_integer(value["horizon_count"], name="temporal horizon count", minimum=1),
        mean_current_future_l1=_finite(
            value["mean_current_future_l1"], name="mean current-future L1"
        ),
        sampled_median_current_future_l1=_finite(
            value["sampled_median_current_future_l1"],
            name="sampled median current-future L1",
        ),
        sampled_p90_current_future_l1=_finite(
            value["sampled_p90_current_future_l1"],
            name="sampled p90 current-future L1",
        ),
        mean_current_future_cosine=_finite(
            value["mean_current_future_cosine"], name="mean current-future cosine"
        ),
        zero_template_l1=_finite(value["zero_template_l1"], name="temporal zero loss"),
        median_template_l1=_finite(value["median_template_l1"], name="temporal median loss"),
        current_copy_advantage_over_median=_finite(
            value["current_copy_advantage_over_median"],
            name="current-copy advantage",
        ),
        current_copy_to_zero_ratio=_finite(
            value["current_copy_to_zero_ratio"], name="current-copy zero ratio"
        ),
        numerically_unchanged_fraction=_finite(
            value["numerically_unchanged_fraction"],
            name="numerically unchanged fraction",
        ),
        obvious_no_temporal_content=no_content,
    )
    nonnegative = (
        result.mean_current_future_l1,
        result.sampled_median_current_future_l1,
        result.sampled_p90_current_future_l1,
        result.zero_template_l1,
        result.median_template_l1,
        result.current_copy_to_zero_ratio,
    )
    threshold = 100 * torch.finfo(torch.float32).eps
    if (
        result.identity_count > result.pair_count
        or result.horizon_count > result.pair_count
        or any(number < 0 for number in nonnegative)
        or result.sampled_p90_current_future_l1 < result.sampled_median_current_future_l1
        or not -1.0001 <= result.mean_current_future_cosine <= 1.0001
        or not 0 <= result.numerically_unchanged_fraction <= 1
        or not math.isclose(
            result.current_copy_advantage_over_median,
            result.median_template_l1 - result.mean_current_future_l1,
            rel_tol=1e-5,
            abs_tol=1e-7,
        )
        or not math.isclose(
            result.current_copy_to_zero_ratio,
            result.mean_current_future_l1
            / max(result.zero_template_l1, torch.finfo(torch.float32).tiny),
            rel_tol=1e-5,
            abs_tol=1e-7,
        )
        or result.obvious_no_temporal_content is not (result.mean_current_future_l1 <= threshold)
    ):
        raise ValueError("predictive temporal diagnostic values are inconsistent")
    return result


def predictive_temporal_pretraining_readiness(
    diagnostics: PredictiveTemporalDiagnostics,
) -> tuple[bool, tuple[str, ...]]:
    """Reject only exact/no-content temporal targets before model training."""

    if not isinstance(diagnostics, PredictiveTemporalDiagnostics):
        raise TypeError("predictive temporal readiness requires typed diagnostics")
    failures: list[str] = []
    if diagnostics.obvious_no_temporal_content:
        failures.append("no_measurable_current_to_future_target_change")
    if diagnostics.identity_count < 2:
        failures.append("insufficient_temporal_identity_support")
    return not failures, tuple(failures)


@torch.no_grad()
def predictive_visible_support_diagnostics(
    sampled_importance: torch.Tensor,
    *,
    supported_count: int,
    total_importance: float,
    minimum_importance: float,
    maximum_importance: float,
) -> PredictiveVisibleSupportDiagnostics:
    """Summarize target support while retaining exact full-scan moments."""

    count = _integer(supported_count, name="supported count", minimum=2)
    if (
        sampled_importance.ndim != 1
        or sampled_importance.numel() < 2
        or sampled_importance.numel() > count
        or not sampled_importance.is_floating_point()
        or sampled_importance.device.type != "cpu"
        or not torch.isfinite(sampled_importance).all()
        or (sampled_importance <= 0).any()
        or (sampled_importance > 1).any()
    ):
        raise ValueError("predictive support sample must be positive finite CPU [rows]")
    total = _finite(total_importance, name="total visible image fraction")
    minimum = _finite(minimum_importance, name="minimum visible image fraction")
    maximum = _finite(maximum_importance, name="maximum visible image fraction")
    mean = total / count
    if not 0 < minimum <= mean <= maximum <= 1:
        raise ValueError("predictive full-scan support moments are inconsistent")
    sample = sampled_importance.detach().float()
    quantiles = torch.quantile(sample, torch.tensor([0.05, 0.5, 0.95]))
    result = PredictiveVisibleSupportDiagnostics(
        supported_count=count,
        sampled_count=sample.numel(),
        minimum_visible_image_fraction=minimum,
        mean_visible_image_fraction=mean,
        maximum_visible_image_fraction=maximum,
        sampled_p05_visible_image_fraction=float(quantiles[0]),
        sampled_median_visible_image_fraction=float(quantiles[1]),
        sampled_p95_visible_image_fraction=float(quantiles[2]),
    )
    predictive_visible_support_diagnostics_from_mapping(result.as_dict())
    return result


def _encoded_labels(values: Sequence[str], *, name: str) -> torch.Tensor:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    index: dict[str, int] = {}
    encoded: list[int] = []
    for value in values:
        encoded.append(index.setdefault(value, len(index)))
    return torch.tensor(encoded, dtype=torch.long)


@torch.no_grad()
def predictive_latent_diagnostics(
    features: torch.Tensor,
    *,
    identity_keys: Sequence[str],
    target_group_keys: Sequence[str],
    pair_chunk_size: int = 256,
) -> PredictiveLatentDiagnostics:
    """Measure target collapse and cross-frame identity retrieval.

    Target groups identify independently encoded target observations, normally
    by target-frame content digest.  Same-group pairs are excluded so repeated
    cache routes to one target frame cannot inflate identity retrieval.
    """

    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 2:
        raise ValueError("predictive diagnostics require at least two [rows,width] features")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("predictive diagnostic features must be finite floating point")
    if features.device.type != "cpu":
        raise ValueError("predictive target diagnostics must run on bounded CPU samples")
    if (
        isinstance(pair_chunk_size, bool)
        or not isinstance(pair_chunk_size, int)
        or pair_chunk_size <= 0
    ):
        raise ValueError("pair chunk size must be a positive integer")
    count, width = features.shape
    if len(identity_keys) != count or len(target_group_keys) != count:
        raise ValueError("predictive labels must match the feature row count")

    identities = _encoded_labels(identity_keys, name="identity keys")
    groups = _encoded_labels(target_group_keys, name="target group keys")
    normalized = F.layer_norm(features.detach().float(), (width,))
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    mean_variance = centered.square().mean(dim=0).mean()
    singular_values = torch.linalg.svdvals(centered)
    spectrum = singular_values.square()
    spectrum_mass = spectrum.sum()
    rank_ceiling = min(count - 1, width)
    if float(spectrum_mass) <= torch.finfo(spectrum.dtype).tiny:
        effective_rank = normalized.new_zeros(())
    else:
        probability = spectrum / spectrum_mass
        positive = probability > 0
        entropy = -(probability[positive] * probability[positive].log()).sum()
        effective_rank = entropy.exp()

    zero_template_l1 = normalized.abs().mean()
    median_template = normalized.median(dim=0).values
    normalized_median = F.layer_norm(median_template.unsqueeze(0), (width,)).squeeze(0)
    median_template_l1 = (normalized - normalized_median).abs().mean()

    unit = F.normalize(normalized, dim=-1)
    same_sum = normalized.new_zeros(())
    different_sum = normalized.new_zeros(())
    same_count = 0
    different_count = 0
    retrieval_correct = 0
    retrieval_queries = 0
    chance_sum = 0.0
    for start in range(0, count, pair_chunk_size):
        stop = min(start + pair_chunk_size, count)
        similarities = unit[start:stop] @ unit.T
        cross_group = groups[start:stop, None] != groups[None, :]
        same_identity = identities[start:stop, None] == identities[None, :]
        positive = cross_group & same_identity
        negative = cross_group & ~same_identity
        same_sum += similarities[positive].sum()
        different_sum += similarities[negative].sum()
        same_count += int(positive.sum())
        different_count += int(negative.sum())

        positive_per_query = positive.sum(dim=1)
        candidate_per_query = cross_group.sum(dim=1)
        eligible = (positive_per_query > 0) & (candidate_per_query > positive_per_query)
        if eligible.any():
            eligible_similarity = similarities[eligible].masked_fill(
                ~cross_group[eligible],
                -torch.inf,
            )
            nearest = eligible_similarity.argmax(dim=1)
            query_identity = identities[start:stop][eligible]
            retrieval_correct += int((identities[nearest] == query_identity).sum())
            retrieval_queries += int(eligible.sum())
            chance_sum += float(
                (positive_per_query[eligible] / candidate_per_query[eligible]).sum()
            )

    same_cosine = None if same_count == 0 else float(same_sum / same_count)
    different_cosine = None if different_count == 0 else float(different_sum / different_count)
    cosine_margin = (
        None if same_cosine is None or different_cosine is None else same_cosine - different_cosine
    )
    top1 = None if retrieval_queries == 0 else retrieval_correct / retrieval_queries
    chance = None if retrieval_queries == 0 else chance_sum / retrieval_queries
    variance_value = float(mean_variance)
    rank_value = float(effective_rank)
    rank_fraction = rank_value / rank_ceiling
    obvious_collapse = (
        variance_value <= 100 * torch.finfo(torch.float32).eps
        or rank_value <= 1.01
        or not math.isfinite(rank_fraction)
    )
    return PredictiveLatentDiagnostics(
        valid_count=count,
        feature_width=width,
        identity_count=len(set(identity_keys)),
        target_group_count=len(set(target_group_keys)),
        mean_dimension_variance=variance_value,
        effective_rank=rank_value,
        effective_rank_ceiling=rank_ceiling,
        effective_rank_fraction=rank_fraction,
        zero_template_l1=float(zero_template_l1),
        median_template_l1=float(median_template_l1),
        retrieval_query_count=retrieval_queries,
        identity_top1_accuracy=top1,
        identity_chance_accuracy=chance,
        same_identity_cosine=same_cosine,
        different_identity_cosine=different_cosine,
        same_different_cosine_margin=cosine_margin,
        obvious_numerical_collapse=obvious_collapse,
    )


@torch.no_grad()
def predictive_temporal_diagnostics(
    current_features: torch.Tensor,
    future_features: torch.Tensor,
    *,
    identity_keys: Sequence[str],
    horizons: Sequence[int],
) -> PredictiveTemporalDiagnostics:
    """Measure whether frozen future targets add content beyond a current copy."""

    if (
        current_features.ndim != 2
        or future_features.shape != current_features.shape
        or current_features.shape[0] < 2
        or current_features.shape[1] < 2
    ):
        raise ValueError("temporal diagnostics require paired [rows,width] features")
    if (
        not current_features.is_floating_point()
        or not future_features.is_floating_point()
        or not torch.isfinite(current_features).all()
        or not torch.isfinite(future_features).all()
    ):
        raise ValueError("temporal diagnostic features must be finite floating point")
    if current_features.device.type != "cpu" or future_features.device.type != "cpu":
        raise ValueError("predictive temporal diagnostics must run on bounded CPU samples")
    count, width = current_features.shape
    if len(identity_keys) != count or len(horizons) != count:
        raise ValueError("temporal diagnostic labels must match the paired feature count")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in horizons
    ):
        raise ValueError("temporal diagnostic horizons must be positive integers")
    _encoded_labels(identity_keys, name="temporal identity keys")

    current = F.layer_norm(current_features.detach().float(), (width,))
    future = F.layer_norm(future_features.detach().float(), (width,))
    distances = (current - future).abs().mean(dim=-1)
    mean_distance = distances.mean()
    cosine = F.cosine_similarity(current, future, dim=-1).mean()
    zero = future.abs().mean()
    median = future.median(dim=0).values
    normalized_median = F.layer_norm(median.unsqueeze(0), (width,)).squeeze(0)
    median_loss = (future - normalized_median).abs().mean()
    threshold = 100 * torch.finfo(torch.float32).eps
    mean_value = float(mean_distance)
    zero_value = float(zero)
    median_value = float(median_loss)
    return PredictiveTemporalDiagnostics(
        pair_count=count,
        feature_width=width,
        identity_count=len(set(identity_keys)),
        horizon_count=len(set(horizons)),
        mean_current_future_l1=mean_value,
        sampled_median_current_future_l1=float(torch.quantile(distances, 0.5)),
        sampled_p90_current_future_l1=float(torch.quantile(distances, 0.9)),
        mean_current_future_cosine=float(cosine),
        zero_template_l1=zero_value,
        median_template_l1=median_value,
        current_copy_advantage_over_median=median_value - mean_value,
        current_copy_to_zero_ratio=mean_value / max(zero_value, torch.finfo(torch.float32).tiny),
        numerically_unchanged_fraction=float((distances <= threshold).float().mean()),
        obvious_no_temporal_content=mean_value <= threshold,
    )
