"""Causal evidence v2 for the P2 future-prediction probe.

The v1 gate in ``tools/run_lingbot_vla2_task_independent_p1.py`` decided each arm with

    pass = mean_margin > 1e-6 and mean_distance > 1e-6 and positive_fraction >= threshold

on the *training* loss returned by :func:`native_predictive_term`. Three properties of
that construction make its verdict uninterpretable, and this module fixes each one.

1.  **The readout is importance-scaled.** ``native_predictive_term`` folds the target's
    importance into the per-element values and then lets ``ObjectiveTerm.normalized``
    divide by the *count* of valid entries, so the reported scalar is

        Sigma_i w_i d_i / N          rather than    Sigma_i w_i d_i / Sigma_i w_i

    where ``w`` is track support (object footprint, clamped to one) and ``d`` is the
    mean absolute deviation between layer-normalised prediction and target. Because ``w``
    is small and varies per sample, the scalar mixes "how wrong the prediction is" with
    "how large the object is", and an absolute epsilon of 1e-6 on its differences is a
    threshold on object area as much as on prediction quality. ``normalised_margin``
    below divides the margin by the sample's own evidence mass, which recovers the
    importance-weighted mean and puts every arm on one scale in [0, ~1.13].

2.  **There is no null model.** With n=14 the v1 fraction thresholds of 0.5/0.6 admit a
    ~21% per-arm false-positive rate under exchangeable signs, no multiplicity control
    is applied across the eight arms, and two arms landing on the same 7/14 split were
    given opposite verdicts because their means happened to fall on opposite sides of
    zero. This module reports an exact paired sign test, a paired percentile bootstrap
    interval, and Holm-Bonferroni adjustment across arms.

3.  **"Corrupting an input did not change the loss" was read as "the model ignores the
    input".** It is equally consistent with "the substitution did not change the input".
    At h=1 on 30 Hz CALVIN the posterior at t-1 and at t are nearly the same tensor, so
    ``wrong_time_source`` is close to a null intervention by construction. Every
    substitution arm therefore carries a *manipulation strength*, and samples whose
    manipulation falls below threshold are excluded from the arm rather than counted as
    evidence of invariance.

A fourth addition has no v1 counterpart: an explicit ``NEUTRAL`` negative-control arm.
Its margin distribution is the empirical noise floor. The equivalence band that separates
NO_EFFECT from INCONCLUSIVE is deliberately *not* derived from it -- a band taken from the
control's confidence interval would shrink like 1/sqrt(n) alongside the interval being
tested against it, so no sample size would ever support a negative result. The band is a
preregistered fraction of the factual loss on the importance-weighted scale, and the
control arm instead certifies that the run's noise floor is small enough to resolve it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

CAUSAL_EVIDENCE_SCHEMA_V2 = "picf-next.lingbot-vla2-p2-future-causal-evidence.v2"

_DEFAULT_ALPHA = 0.05
_DEFAULT_BOOTSTRAP_ITERATIONS = 20000
_DEFAULT_MINIMUM_SAMPLES = 12
_DEFAULT_EQUIVALENCE_FRACTION = 0.01


class ArmExpectation(str, Enum):
    """What the preregistration predicts for an arm."""

    HARMS = "harms"
    """Corrupting this input must make the prediction worse."""

    NEUTRAL = "neutral"
    """Negative control: this substitution carries no information, so the margin
    distribution it produces is the measurement's noise floor."""


class ArmVerdictLabel(str, Enum):
    EFFECT = "EFFECT"
    """The margin is significantly non-zero in the predicted direction."""

    NO_EFFECT = "NO_EFFECT"
    """The interval is inside the equivalence band: an effect this small is ruled out."""

    INCONCLUSIVE = "INCONCLUSIVE"
    """Neither significant nor bounded. The arm is underpowered, not negative."""

    NULL_INTERVENTION = "NULL_INTERVENTION"
    """Too few samples survived the manipulation check, so the arm tested nothing."""

    WRONG_DIRECTION = "WRONG_DIRECTION"
    """Significant, but with the sign opposite to the preregistered prediction."""


def _finite(value: float, *, name: str, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be real-valued")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if non_negative and number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True, slots=True)
class InterventionObservation:
    """One paired factual/intervened measurement for one arm on one sample.

    ``evidence_mass`` is the mean target importance over the valid entries the loss was
    reduced over. Dividing a raw margin by it recovers the importance-weighted mean that
    ``ObjectiveTerm.normalized`` would have produced had the importance been supplied as
    ``sample_weight`` instead of folded into the values.
    """

    arm: str
    sample_key: str
    horizon: int
    factual_loss: float
    intervened_loss: float
    evidence_mass: float
    valid_target_count: int
    prediction_displacement: float
    source_manipulation: float | None
    target_changed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or not self.arm:
            raise ValueError("arm name must be non-empty")
        if not isinstance(self.sample_key, str) or not self.sample_key:
            raise ValueError("sample key must be non-empty")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("horizon must be an integer")
        if self.horizon < 1:
            raise ValueError("horizon must be at least one")
        if isinstance(self.valid_target_count, bool) or not isinstance(
            self.valid_target_count, int
        ):
            raise TypeError("valid target count must be an integer")
        if self.valid_target_count < 1:
            raise ValueError("valid target count must be positive")
        if not isinstance(self.target_changed, bool):
            raise TypeError("target_changed must be boolean")
        _finite(self.factual_loss, name="factual loss", non_negative=True)
        _finite(self.intervened_loss, name="intervened loss", non_negative=True)
        _finite(self.prediction_displacement, name="prediction displacement", non_negative=True)
        if self.source_manipulation is not None:
            _finite(self.source_manipulation, name="source manipulation", non_negative=True)
        mass = _finite(self.evidence_mass, name="evidence mass", non_negative=True)
        if mass <= 0:
            raise ValueError("evidence mass must be positive; a zero-mass sample carries no loss")

    @property
    def raw_margin(self) -> float:
        """The v1 quantity: a difference of importance-scaled, count-normalised losses."""

        return self.intervened_loss - self.factual_loss

    @property
    def normalised_margin(self) -> float:
        """The raw margin divided by this sample's evidence mass.

        This is the difference of importance-*weighted* mean deviations, so it is
        comparable across samples whose objects occupy very different areas, and it lives
        on the same scale as the layer-normalised prediction space (chance is ~1.13).
        """

        return self.raw_margin / self.evidence_mass

    @property
    def relative_margin(self) -> float:
        """The margin as a fraction of the sample's own factual loss."""

        return self.raw_margin / self.factual_loss if self.factual_loss > 0 else 0.0

    @property
    def factual_normalised_loss(self) -> float:
        """The factual loss on the importance-weighted scale.

        This is the number that says whether the predictor is any good: it is a mean
        absolute deviation between layer-normalised vectors, so ~1.13 is chance and 0 is
        exact. The importance-scaled figure the v1 report published (~0.02) cannot be read
        that way, because it is that quantity multiplied by object area.
        """

        return self.factual_loss / self.evidence_mass


def exact_sign_test(positives: int, total: int, *, one_sided: bool = True) -> float:
    """Exact binomial tail probability under sign exchangeability.

    The v1 gate compared ``positives / total`` against 0.5 or 0.6 with no reference to
    this distribution, which is why a 0.6 threshold at n=14 admitted a false positive
    about one time in five.
    """

    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("sign test requires a positive sample count")
    if isinstance(positives, bool) or not isinstance(positives, int):
        raise TypeError("sign test requires an integer positive count")
    if not 0 <= positives <= total:
        raise ValueError("positive count must lie within the sample count")
    upper = math.fsum(
        math.comb(total, index) for index in range(positives, total + 1)
    ) / 2.0**total
    if one_sided:
        return min(1.0, upper)
    lower = math.fsum(
        math.comb(total, index) for index in range(0, positives + 1)
    ) / 2.0**total
    return min(1.0, 2.0 * min(upper, lower))


def percentile_bootstrap(
    values: Sequence[float],
    *,
    alpha: float = _DEFAULT_ALPHA,
    iterations: int = _DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of paired differences."""

    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("bootstrap requires at least one value")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if len(sample) == 1:
        return sample[0], sample[0]
    rng = random.Random(seed)
    size = len(sample)
    means = []
    for _ in range(iterations):
        means.append(math.fsum(sample[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    low = means[max(0, int(math.floor((alpha / 2) * iterations)))]
    high = means[min(iterations - 1, int(math.ceil((1 - alpha / 2) * iterations)) - 1)]
    return low, high


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment, monotone in the input order."""

    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[name] = running
    return adjusted


def minimum_detectable_margin(
    values: Sequence[float], *, alpha: float = _DEFAULT_ALPHA, power: float = 0.8
) -> float:
    """Smallest true mean margin this sample size could detect at the stated power.

    Reported so that an INCONCLUSIVE arm carries the size of the effect it failed to
    rule out, instead of being silently read as a negative result.
    """

    sample = [float(value) for value in values]
    size = len(sample)
    if size < 2:
        return float("inf")
    mean = math.fsum(sample) / size
    variance = math.fsum((value - mean) ** 2 for value in sample) / (size - 1)
    deviation = math.sqrt(variance)
    # Normal approximation; z(0.95) = 1.6449, z(0.80) = 0.8416.
    z_alpha = 1.6448536269514722 if alpha == _DEFAULT_ALPHA else _normal_quantile(1 - alpha)
    z_power = _normal_quantile(power)
    return (z_alpha + z_power) * deviation / math.sqrt(size)


def _normal_quantile(probability: float) -> float:
    """Acklam's rational approximation to the standard normal quantile."""

    if not 0 < probability < 1:
        raise ValueError("quantile probability must lie strictly between zero and one")
    a = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    )
    low, high = 0.02425, 1 - 0.02425
    if probability < low:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if probability > high:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


@dataclass(frozen=True, slots=True)
class ArmVerdict:
    arm: str
    expectation: ArmExpectation
    label: ArmVerdictLabel
    submitted_samples: int
    scored_samples: int
    positive_samples: int
    sign_p_value: float
    holm_p_value: float
    mean_normalised_margin: float
    bootstrap_low: float
    bootstrap_high: float
    mean_relative_margin: float
    mean_prediction_displacement: float
    mean_source_manipulation: float | None
    minimum_detectable_margin: float
    equivalence_band: float

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "expectation": self.expectation.value,
            "verdict": self.label.value,
            "submitted_samples": self.submitted_samples,
            "scored_samples": self.scored_samples,
            "positive_samples": self.positive_samples,
            "sign_p_value": self.sign_p_value,
            "holm_p_value": self.holm_p_value,
            "mean_normalised_margin": self.mean_normalised_margin,
            "bootstrap_low": self.bootstrap_low,
            "bootstrap_high": self.bootstrap_high,
            "mean_relative_margin": self.mean_relative_margin,
            "mean_prediction_displacement": self.mean_prediction_displacement,
            "mean_source_manipulation": self.mean_source_manipulation,
            "minimum_detectable_margin": self.minimum_detectable_margin,
            "equivalence_band": self.equivalence_band,
        }


@dataclass(frozen=True, slots=True)
class CausalEvidenceReport:
    schema: str
    status: str
    alpha: float
    minimum_samples: int
    minimum_manipulation: float
    equivalence_band: float
    equivalence_band_fraction: float
    reference_scale: float
    noise_floor_arm: str | None
    noise_floor_within_band: bool
    horizons: tuple[int, ...]
    arms: tuple[ArmVerdict, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "alpha": self.alpha,
            "minimum_samples": self.minimum_samples,
            "minimum_manipulation": self.minimum_manipulation,
            "equivalence_band": self.equivalence_band,
            "equivalence_band_fraction": self.equivalence_band_fraction,
            "reference_scale": self.reference_scale,
            "noise_floor_arm": self.noise_floor_arm,
            "noise_floor_within_band": self.noise_floor_within_band,
            "horizons": list(self.horizons),
            "arms": [arm.as_dict() for arm in self.arms],
        }


def score_causal_evidence(
    observations: Iterable[InterventionObservation],
    *,
    expectations: Mapping[str, ArmExpectation],
    alpha: float = _DEFAULT_ALPHA,
    minimum_samples: int = _DEFAULT_MINIMUM_SAMPLES,
    minimum_manipulation: float = 0.0,
    equivalence_band_fraction: float = _DEFAULT_EQUIVALENCE_FRACTION,
    equivalence_band: float | None = None,
    require_target_change: bool = True,
    require_noise_floor: bool = True,
    bootstrap_iterations: int = _DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = 0,
) -> CausalEvidenceReport:
    """Score paired interventions with a null model, a manipulation check and multiplicity control.

    ``equivalence_band`` bounds what counts as "no effect". It must be a *fixed* effect
    size, so by default it is ``equivalence_band_fraction`` of the mean factual loss on
    the importance-weighted scale. Deriving it instead from the control arm's confidence
    interval would be self-defeating: that interval and the tested arm's interval both
    shrink like 1/sqrt(n), so no sample size would ever let an arm be declared negative.

    The ``NEUTRAL`` arm therefore serves a different purpose. It measures the noise floor,
    and if its own interval does not fit inside the band, the run cannot resolve effects
    at the size that was preregistered as meaningful and every verdict is INCONCLUSIVE.
    """

    grouped: dict[str, list[InterventionObservation]] = {}
    for observation in observations:
        if not isinstance(observation, InterventionObservation):
            raise TypeError("causal evidence requires typed intervention observations")
        grouped.setdefault(observation.arm, []).append(observation)
    if not grouped:
        raise ValueError("causal evidence requires at least one observation")
    missing = sorted(set(grouped) - set(expectations))
    if missing:
        raise ValueError(f"arms without a preregistered expectation: {missing}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int):
        raise TypeError("minimum sample count must be an integer")
    if minimum_samples < 1:
        raise ValueError("minimum sample count must be positive")
    _finite(minimum_manipulation, name="minimum manipulation", non_negative=True)

    horizons = tuple(sorted({item.horizon for items in grouped.values() for item in items}))

    def scored(items: Sequence[InterventionObservation]) -> list[InterventionObservation]:
        if minimum_manipulation > 0 and any(
            item.source_manipulation is None for item in items
        ):
            raise ValueError(
                "a manipulation threshold was requested but some observations never "
                "measured how much their substitution changed the source"
            )
        kept = [
            item
            for item in items
            if item.source_manipulation is None
            or item.source_manipulation >= minimum_manipulation
        ]
        if require_target_change:
            kept = [item for item in kept if item.target_changed]
        return kept

    # The band is a fixed fraction of the factual loss on the importance-weighted scale,
    # so it does not shrink as samples accumulate and NO_EFFECT stays reachable.
    every = [item for items in grouped.values() for item in items]
    reference_scale = math.fsum(item.factual_normalised_loss for item in every) / len(every)
    if equivalence_band is None:
        if not 0 < equivalence_band_fraction < 1:
            raise ValueError("equivalence band fraction must lie strictly between zero and one")
        band = equivalence_band_fraction * reference_scale
    else:
        band = _finite(equivalence_band, name="equivalence band", non_negative=True)
        equivalence_band_fraction = band / reference_scale if reference_scale > 0 else 0.0
    if band <= 0:
        raise ValueError("equivalence band must be positive")

    # The neutral arm is the noise floor. If it cannot itself be resolved inside the band,
    # nothing in this run can be, and saying so is the only honest verdict.
    neutral_arms = [name for name, kind in expectations.items() if kind is ArmExpectation.NEUTRAL]
    noise_floor_arm: str | None = None
    noise_floor_within_band = False
    for name in sorted(neutral_arms):
        items = scored(grouped.get(name, []))
        if len(items) >= 2:
            margins = [item.normalised_margin for item in items]
            low, high = percentile_bootstrap(
                margins, alpha=alpha, iterations=bootstrap_iterations, seed=seed
            )
            noise_floor_arm = name
            noise_floor_within_band = -band <= low and high <= band
            break
    if noise_floor_arm is None and require_noise_floor:
        raise ValueError(
            "causal evidence requires a NEUTRAL control arm with at least two scored "
            "samples to establish the measurement's noise floor"
        )

    raw_p: dict[str, float] = {}
    prepared: dict[str, tuple[list[InterventionObservation], list[float], int]] = {}
    for name, items in grouped.items():
        kept = scored(items)
        margins = [item.normalised_margin for item in kept]
        positives = sum(1 for value in margins if value > 0)
        prepared[name] = (kept, margins, positives)
        if len(kept) < minimum_samples:
            raw_p[name] = 1.0
            continue
        one_sided = expectations[name] is ArmExpectation.HARMS
        raw_p[name] = exact_sign_test(positives, len(kept), one_sided=one_sided)
    adjusted = holm_adjust(raw_p)

    verdicts: list[ArmVerdict] = []
    for name in sorted(grouped):
        items = grouped[name]
        kept, margins, positives = prepared[name]
        expectation = expectations[name]
        if len(kept) < minimum_samples:
            label = ArmVerdictLabel.NULL_INTERVENTION
            low = high = mean = 0.0
            mde = float("inf")
        else:
            mean = math.fsum(margins) / len(margins)
            low, high = percentile_bootstrap(
                margins, alpha=alpha, iterations=bootstrap_iterations, seed=seed
            )
            mde = minimum_detectable_margin(margins, alpha=alpha)
            significant = adjusted[name] <= alpha and low > 0
            opposite = high < 0 and exact_sign_test(
                len(margins) - positives, len(margins), one_sided=True
            ) * len(grouped) <= alpha
            # A negative result needs a certified noise floor. Without one, an interval
            # that happens to sit inside the band proves nothing about the mechanism --
            # it may only mean this run cannot resolve anything at all.
            bounded = noise_floor_within_band and -band <= low and high <= band
            if expectation is ArmExpectation.HARMS:
                if significant:
                    label = ArmVerdictLabel.EFFECT
                elif opposite:
                    label = ArmVerdictLabel.WRONG_DIRECTION
                elif bounded:
                    label = ArmVerdictLabel.NO_EFFECT
                else:
                    label = ArmVerdictLabel.INCONCLUSIVE
            else:
                label = ArmVerdictLabel.NO_EFFECT if bounded else (
                    ArmVerdictLabel.EFFECT if significant or opposite
                    else ArmVerdictLabel.INCONCLUSIVE
                )
        verdicts.append(
            ArmVerdict(
                arm=name,
                expectation=expectation,
                label=label,
                submitted_samples=len(items),
                scored_samples=len(kept),
                positive_samples=positives,
                sign_p_value=raw_p[name],
                holm_p_value=adjusted[name],
                mean_normalised_margin=mean,
                bootstrap_low=low,
                bootstrap_high=high,
                mean_relative_margin=(
                    math.fsum(item.relative_margin for item in kept) / len(kept) if kept else 0.0
                ),
                mean_prediction_displacement=(
                    math.fsum(item.prediction_displacement for item in kept) / len(kept)
                    if kept
                    else 0.0
                ),
                mean_source_manipulation=_mean_manipulation(kept),
                minimum_detectable_margin=mde,
                equivalence_band=band,
            )
        )

    status = _overall_status(verdicts)
    if not noise_floor_within_band and status != "FAIL":
        status = "INCONCLUSIVE"
    return CausalEvidenceReport(
        schema=CAUSAL_EVIDENCE_SCHEMA_V2,
        status=status,
        alpha=alpha,
        minimum_samples=minimum_samples,
        minimum_manipulation=minimum_manipulation,
        equivalence_band=band,
        equivalence_band_fraction=equivalence_band_fraction,
        reference_scale=reference_scale,
        noise_floor_arm=noise_floor_arm,
        noise_floor_within_band=noise_floor_within_band,
        horizons=horizons,
        arms=tuple(verdicts),
    )


def _mean_manipulation(items: Sequence[InterventionObservation]) -> float | None:
    """Average measured manipulation strength, or None when it was never measured."""

    measured = [
        item.source_manipulation for item in items if item.source_manipulation is not None
    ]
    return math.fsum(measured) / len(measured) if measured else None


def _overall_status(verdicts: Sequence[ArmVerdict]) -> str:
    """Collapse arm verdicts into one status.

    v1 emitted PASS/FAIL only, so an arm that was merely underpowered was indistinguishable
    from an arm that was measured and found dead. The three-way outcome here keeps that
    distinction, because it is the difference between "the architecture is wrong" and
    "the experiment could not tell".
    """

    testable = [
        verdict
        for verdict in verdicts
        if verdict.expectation is ArmExpectation.HARMS
    ]
    if not testable:
        return "INCONCLUSIVE"
    if any(verdict.label is ArmVerdictLabel.NULL_INTERVENTION for verdict in testable):
        return "INCONCLUSIVE"
    if any(verdict.label is ArmVerdictLabel.INCONCLUSIVE for verdict in testable):
        return "INCONCLUSIVE"
    if any(verdict.label is ArmVerdictLabel.WRONG_DIRECTION for verdict in testable):
        return "FAIL"
    if all(verdict.label is ArmVerdictLabel.EFFECT for verdict in testable):
        return "PASS"
    return "FAIL"


# ---------------------------------------------------------------------------
# Torch-side measurements. Everything above is pure Python so the statistics can
# be tested and re-run on shipped reports without a GPU or a model.
# ---------------------------------------------------------------------------

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from picf_next.lingbot_native.prediction import NativePredictionRequest  # noqa: E402
from picf_next.lingbot_native.predictive_objective import (  # noqa: E402
    NativePredictiveTarget,
    PredictiveRowAssignment,
)


def predictive_evidence_mass(
    *,
    request: NativePredictionRequest,
    target: NativePredictiveTarget,
    assignment: PredictiveRowAssignment,
    row_binding_valid: torch.Tensor,
) -> float:
    """Mean target importance over exactly the entries the predictive loss reduces over.

    ``native_predictive_term`` multiplies its per-entry deviations by this importance and
    then divides by the *count* of valid entries, so its scalar is ``Sigma w d / N``.
    Dividing that scalar by the value returned here yields ``Sigma w d / Sigma w``: the
    importance-weighted mean deviation, which is what ``ObjectiveTerm.normalized`` would
    have produced had the importance been passed as ``sample_weight``. The training loss
    is deliberately left alone; this is a measurement-side correction only.
    """

    if not isinstance(row_binding_valid, torch.Tensor) or row_binding_valid.dtype != torch.bool:
        raise TypeError("row-binding validity must be a boolean tensor")
    row_to_track = assignment.row_to_track
    if row_binding_valid.shape != row_to_track.shape:
        raise ValueError("row-binding validity must match the row assignment")
    batch = row_to_track.shape[0]
    matched = (row_to_track >= 0) & row_binding_valid
    gather_track = row_to_track.clamp_min(0)
    batch_index = torch.arange(batch, device=row_to_track.device).unsqueeze(1)
    valid = target.valid[batch_index, gather_track] & matched.unsqueeze(-1)
    valid = valid & request.valid.unsqueeze(1)
    count = valid.sum()
    if int(count.item()) <= 0:
        raise ValueError("evidence mass requires at least one valid predictive target")
    importance = target.importance[batch_index, gather_track].to(torch.float32)
    mass = importance.masked_fill(~valid, 0).sum() / count.to(torch.float32)
    return float(mass.item())


def state_manipulation_strength(
    factual_rows: torch.Tensor,
    substituted_rows: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> float:
    """How much a source substitution actually changed the posterior it replaced.

    Without this number, "corrupting the source did not move the loss" cannot be told
    apart from "the corruption did not move the source". At h=1 on 30 Hz CALVIN the two
    are easy to confuse: the posterior one frame earlier is nearly the tensor it replaces,
    so ``wrong_time_source`` is close to a null intervention by construction.

    Scaled the same way as the probe's prediction displacement — mean absolute deviation
    between layer-normalised rows — so manipulation strength and response are comparable.
    """

    if factual_rows.shape != substituted_rows.shape:
        raise ValueError("manipulation strength requires matching row shapes")
    if factual_rows.ndim < 2:
        raise ValueError("posterior rows must have a trailing width axis")
    width = factual_rows.shape[-1]
    left = F.layer_norm(factual_rows.detach().float(), (width,))
    right = F.layer_norm(substituted_rows.detach().float(), (width,))
    difference = (left - right).abs().mean(dim=-1)
    if valid is None:
        return float(difference.mean().item())
    if valid.dtype != torch.bool or valid.shape != difference.shape:
        raise ValueError("manipulation validity must be boolean and match the row axes")
    count = valid.sum()
    if int(count.item()) <= 0:
        return 0.0
    return float((difference.masked_fill(~valid, 0).sum() / count).item())


def matched_noise_rows(
    rows: torch.Tensor,
    *,
    target_distance: float,
    generator: torch.Generator | None = None,
    tolerance: float = 0.02,
    maximum_iterations: int = 40,
) -> torch.Tensor:
    """A Gaussian-perturbed copy of ``rows`` displaced by a requested amount.

    This is the negative control the v1 arm set lacked. The arms differ enormously in how
    hard they push -- in v24, removing the posterior moved the prediction by 0.103 in
    layer-normalised L1 while substituting the previous frame's posterior moved it by
    0.007 -- so a small margin on a gentle arm says nothing on its own. Perturbing the
    posterior by *the same amount* with noise that carries no temporal or episode
    information isolates the response to displacement magnitude, and only a real arm's
    excess over this control is evidence about information flow.

    The scale is found by bisection because the map from noise scale to layer-normalised
    distance has no closed form once the normalisation is applied.
    """

    _finite(target_distance, name="target manipulation distance", non_negative=True)
    if target_distance == 0:
        return rows.detach().clone()
    if not 0 < tolerance < 1:
        raise ValueError("matched-noise tolerance must lie strictly between zero and one")
    reference = rows.detach()
    noise = torch.randn(
        reference.shape, generator=generator, device=reference.device, dtype=torch.float32
    )

    def distance_at(scale: float) -> float:
        return state_manipulation_strength(
            reference, (reference.float() + scale * noise).to(reference.dtype)
        )

    low, high = 0.0, 1.0
    for _ in range(maximum_iterations):
        if distance_at(high) >= target_distance:
            break
        low, high = high, high * 2.0
    else:
        raise RuntimeError("matched-noise search could not reach the requested distance")
    for _ in range(maximum_iterations):
        middle = 0.5 * (low + high)
        achieved = distance_at(middle)
        if abs(achieved - target_distance) <= tolerance * target_distance:
            return (reference.float() + middle * noise).to(reference.dtype)
        if achieved < target_distance:
            low = middle
        else:
            high = middle
    return (reference.float() + 0.5 * (low + high) * noise).to(reference.dtype)
