from __future__ import annotations

import math
import random

import pytest
import torch

from picf_next.lingbot_native.causal_evidence import (
    CAUSAL_EVIDENCE_SCHEMA_V2,
    ArmExpectation,
    ArmVerdictLabel,
    InterventionObservation,
    exact_sign_test,
    holm_adjust,
    minimum_detectable_margin,
    percentile_bootstrap,
    predictive_evidence_mass,
    score_causal_evidence,
    state_manipulation_strength,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_objective import (
    TargetEncoderMode,
    make_native_predictive_target,
    native_predictive_term,
)
from picf_next.lingbot_native.supervision import SequenceAssignment

DIGEST = "a" * 64

# The v1 gate, reproduced verbatim from
# tools/run_lingbot_vla2_task_independent_p1.py::_summarize_p2_causal_evidence.
_V1_EPSILON = 1e-6


def _v1_arm_pass(margins, distances, threshold):
    mean_margin = math.fsum(margins) / len(margins)
    mean_distance = math.fsum(distances) / len(distances)
    positive_fraction = sum(value > _V1_EPSILON for value in margins) / len(margins)
    return mean_margin > _V1_EPSILON and mean_distance > _V1_EPSILON and (
        positive_fraction >= threshold
    )


def _request(
    *,
    source: PredictionSource = PredictionSource.PRIOR,
    evidence: PredictionEvidence = PredictionEvidence.CURRENT_CORRECTION,
) -> NativePredictionRequest:
    return NativePredictionRequest(
        source=source,
        evidence=evidence,
        route_ids=torch.tensor([[0]]),
        horizons=torch.tensor([[0]]),
        addresses=torch.empty(1, 1, 0),
        valid=torch.tensor([[True]]),
    )


def _target(features, request, *, importance=None):
    return make_native_predictive_target(
        modality="vision",
        features=features,
        valid=torch.ones(features.shape[:-1], dtype=torch.bool),
        importance=importance,
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=request.source,
        evidence=request.evidence,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=DIGEST,
        target_data_digest=DIGEST,
        encoder_digest=DIGEST,
        query_schema_digest=DIGEST,
        validity_semantics="independent-track-mask",
        track_identity_keys=(("object/0",),),
    )


def _observation(arm, index, margin, *, mass=0.02, manipulation=1.0, factual=0.02, horizon=1):
    return InterventionObservation(
        arm=arm,
        sample_key=f"sample-{index:04d}",
        horizon=horizon,
        factual_loss=factual,
        intervened_loss=factual + margin,
        evidence_mass=mass,
        valid_target_count=8,
        prediction_displacement=abs(margin) * 100 + 1e-3,
        source_manipulation=manipulation,
    )


# ---------------------------------------------------------------------------
# The defect the v1 gate shipped with.
# ---------------------------------------------------------------------------


def test_v1_gate_gives_opposite_verdicts_to_two_arms_drawn_from_the_same_noise() -> None:
    """v24 put zero_control and batch_shift_control both at 7/14 and passed only one.

    Both arms were pure noise; the verdict was set by which side of zero the mean of
    fourteen sign-symmetric values happened to land on. This reproduces that with two
    samples drawn from the same symmetric distribution.
    """

    rng = random.Random(20260806)
    left = [rng.gauss(0.0, 5e-6) for _ in range(14)]
    right = [-value for value in left]  # same distribution, mirrored
    distances = [1e-2] * 14

    left_pass = _v1_arm_pass(left, distances, 0.5)
    right_pass = _v1_arm_pass(right, distances, 0.5)
    assert left_pass != right_pass, "mirrored noise must expose the v1 sign dependence"


def test_v1_positive_fraction_threshold_admits_frequent_false_positives() -> None:
    """A 0.6 fraction threshold at n=14 passes noise about one time in five."""

    false_positive_rate = sum(
        math.comb(14, k) for k in range(9, 15)
    ) / 2.0**14
    assert 0.15 < false_positive_rate < 0.25
    # The v2 sign test reports that same tail as a p-value instead of a pass.
    assert math.isclose(exact_sign_test(9, 14), false_positive_rate, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------


def test_exact_sign_test_matches_closed_form_tails() -> None:
    assert math.isclose(exact_sign_test(14, 14), 1 / 2**14)
    assert math.isclose(exact_sign_test(13, 14), 15 / 2**14)
    assert math.isclose(exact_sign_test(0, 14), 1.0)
    two_sided = exact_sign_test(14, 14, one_sided=False)
    assert math.isclose(two_sided, 2 / 2**14)


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.4, "d": 0.9})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= adjusted["d"]
    assert math.isclose(adjusted["a"], 0.004)
    assert all(value <= 1.0 for value in adjusted.values())


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = [0.1, 0.2, 0.15, 0.18, 0.12, 0.22, 0.14, 0.19]
    low, high = percentile_bootstrap(values, iterations=4000, seed=3)
    mean = math.fsum(values) / len(values)
    assert low < mean < high


def test_minimum_detectable_margin_shrinks_with_sample_size() -> None:
    rng = random.Random(11)
    small = [rng.gauss(0.0, 1.0) for _ in range(14)]
    large = [rng.gauss(0.0, 1.0) for _ in range(140)]
    assert minimum_detectable_margin(large) < minimum_detectable_margin(small)


# ---------------------------------------------------------------------------
# The v2 gate.
# ---------------------------------------------------------------------------


def _expectations(*harms, neutral="null_reencode_source"):
    mapping = {name: ArmExpectation.HARMS for name in harms}
    mapping[neutral] = ArmExpectation.NEUTRAL
    return mapping


def _noise_report(count, *, seed, scale=1e-3):
    rng = random.Random(seed)
    observations = []
    for index in range(count):
        observations.append(_observation("wrong_time_source", index, rng.gauss(0.0, scale)))
        observations.append(_observation("null_reencode_source", index, rng.gauss(0.0, scale)))
    return score_causal_evidence(
        observations,
        expectations=_expectations("wrong_time_source"),
        minimum_samples=12,
        bootstrap_iterations=4000,
    )


def test_v2_calls_noise_inconclusive_at_the_v24_sample_size() -> None:
    """The distinction v1 could not make: underpowered is not the same as dead.

    v24 ran fourteen samples and reported FAIL. Fourteen samples of pure noise cannot
    support any conclusion, and v2 says so instead of converting the absence of evidence
    into evidence of absence.
    """

    report = _noise_report(14, seed=7)
    arm = next(item for item in report.arms if item.arm == "wrong_time_source")
    assert arm.label is ArmVerdictLabel.INCONCLUSIVE
    assert report.status == "INCONCLUSIVE"
    assert report.schema == CAUSAL_EVIDENCE_SCHEMA_V2
    assert report.noise_floor_arm == "null_reencode_source"
    # The noise floor itself is not resolvable at this size, which is the run-level reason
    # nothing here can be concluded.
    assert not report.noise_floor_within_band
    # The band is a fixed effect size, not something that shrank to meet the data.
    assert arm.equivalence_band == pytest.approx(0.01 * report.reference_scale)
    # And the arm carries the size of the effect it failed to rule out.
    assert arm.minimum_detectable_margin > arm.equivalence_band


def test_v2_can_declare_no_effect_once_the_sample_size_supports_it() -> None:
    """The same noise, measured enough times, becomes a bounded negative result.

    This is the property that a control-derived band cannot have: if the band were the
    control arm's confidence interval, it would shrink at the same rate as the interval
    being tested against it and NO_EFFECT would be unreachable at every sample size.
    """

    small = _noise_report(14, seed=7)
    large = _noise_report(400, seed=7)
    assert large.equivalence_band == pytest.approx(small.equivalence_band, rel=1e-6)
    arm = next(item for item in large.arms if item.arm == "wrong_time_source")
    assert arm.label is ArmVerdictLabel.NO_EFFECT
    assert large.noise_floor_within_band
    assert large.status == "PASS" or arm.label is ArmVerdictLabel.NO_EFFECT
    small_arm = next(item for item in small.arms if item.arm == "wrong_time_source")
    assert arm.minimum_detectable_margin < small_arm.minimum_detectable_margin


def test_v2_recovers_a_planted_effect_and_reports_it_on_the_normalised_scale() -> None:
    rng = random.Random(23)
    observations = []
    for index in range(14):
        observations.append(
            _observation("absent_source", index, 2.0e-4 + rng.gauss(0.0, 2e-5))
        )
        observations.append(
            _observation("null_reencode_source", index, rng.gauss(0.0, 2e-6))
        )
    report = score_causal_evidence(
        observations,
        expectations=_expectations("absent_source"),
        minimum_samples=12,
        bootstrap_iterations=4000,
    )
    arm = next(item for item in report.arms if item.arm == "absent_source")
    assert arm.label is ArmVerdictLabel.EFFECT
    assert arm.holm_p_value <= 0.05
    assert arm.bootstrap_low > 0
    # raw margin 2e-4 at evidence mass 0.02 is a 1e-2 effect on the prediction scale
    assert 0.008 < arm.mean_normalised_margin < 0.012


def test_v2_flags_a_wrong_direction_effect_instead_of_calling_it_a_pass() -> None:
    rng = random.Random(31)
    observations = []
    for index in range(14):
        observations.append(
            _observation("zero_control", index, -2.0e-4 + rng.gauss(0.0, 1e-5))
        )
        observations.append(
            _observation("null_reencode_source", index, rng.gauss(0.0, 1e-6))
        )
    report = score_causal_evidence(
        observations,
        expectations=_expectations("zero_control"),
        minimum_samples=12,
        bootstrap_iterations=4000,
    )
    arm = next(item for item in report.arms if item.arm == "zero_control")
    assert arm.label is ArmVerdictLabel.WRONG_DIRECTION
    assert report.status == "FAIL"


def test_manipulation_check_rejects_an_arm_whose_substitution_changed_nothing() -> None:
    """A substitution that leaves the source almost unchanged tested nothing.

    v1 counted these samples as evidence that the model ignores the posterior.
    """

    rng = random.Random(5)
    observations = []
    for index in range(14):
        observations.append(
            _observation(
                "wrong_time_source",
                index,
                rng.gauss(0.0, 5e-6),
                manipulation=1e-4,  # the t-1 posterior is nearly the tensor it replaced
            )
        )
        observations.append(
            _observation("null_reencode_source", index, rng.gauss(0.0, 5e-6))
        )
    report = score_causal_evidence(
        observations,
        expectations=_expectations("wrong_time_source"),
        minimum_samples=12,
        minimum_manipulation=1e-2,
        bootstrap_iterations=4000,
    )
    arm = next(item for item in report.arms if item.arm == "wrong_time_source")
    assert arm.label is ArmVerdictLabel.NULL_INTERVENTION
    assert arm.scored_samples == 0
    assert arm.submitted_samples == 14
    assert report.status == "INCONCLUSIVE"


def test_v2_requires_an_expectation_for_every_arm() -> None:
    with pytest.raises(ValueError, match="without a preregistered expectation"):
        score_causal_evidence(
            [_observation("mystery_arm", 0, 1e-5)],
            expectations={"null_reencode_source": ArmExpectation.NEUTRAL},
        )


def test_v2_refuses_to_score_without_a_noise_floor_control() -> None:
    with pytest.raises(ValueError, match="requires a NEUTRAL control arm"):
        score_causal_evidence(
            [_observation("absent_source", index, 1e-5) for index in range(14)],
            expectations={"absent_source": ArmExpectation.HARMS},
        )


# ---------------------------------------------------------------------------
# The measurement-side correction, checked against the real training loss.
# ---------------------------------------------------------------------------


def test_evidence_mass_recovers_the_importance_weighted_mean() -> None:
    """``raw_loss / evidence_mass`` is invariant to object area; ``raw_loss`` is not.

    The shipped contract (test_predictive_loss_preserves_absolute_visible_evidence_mass)
    makes the training loss scale linearly with importance, on purpose. That makes it
    unusable as a causal readout, because a margin computed from it also scales with
    object area. Dividing by the evidence mass undoes exactly that factor and leaves the
    training loss untouched.
    """

    request = _request()
    prediction = torch.tensor([[[[2.0, -1.0, 0.5, 3.0]]]])
    features = torch.tensor([[[[-2.0, 4.0, 1.0, 0.0]]]])
    assignment = SequenceAssignment(torch.tensor([[0]]))
    row_valid = torch.ones(1, 1, dtype=torch.bool)

    raw_losses = {}
    masses = {}
    for weight in (1.0, 0.25, 0.02):
        target = _target(features, request, importance=torch.full((1, 1, 1), weight))
        term = native_predictive_term(
            prediction=prediction,
            request=request,
            target=target,
            assignment=assignment,
            row_binding_valid=row_valid,
            weight=1.0,
        )
        raw_losses[weight] = float(term.normalized().item())
        masses[weight] = predictive_evidence_mass(
            request=request,
            target=target,
            assignment=assignment,
            row_binding_valid=row_valid,
        )
        assert math.isclose(masses[weight], weight, rel_tol=1e-6)

    # The raw loss tracks object area, exactly as the shipped contract requires.
    assert math.isclose(raw_losses[0.25], raw_losses[1.0] * 0.25, rel_tol=1e-5)
    assert math.isclose(raw_losses[0.02], raw_losses[1.0] * 0.02, rel_tol=1e-5)

    # The normalised loss does not.
    normalised = [raw_losses[w] / masses[w] for w in (1.0, 0.25, 0.02)]
    for value in normalised[1:]:
        assert math.isclose(value, normalised[0], rel_tol=1e-5)


def test_evidence_mass_matches_the_entries_the_loss_reduces_over() -> None:
    """Unmatched rows must not contribute mass, or the normalisation would be wrong."""

    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
        route_ids=torch.tensor([[0]]),
        horizons=torch.tensor([[0]]),
        addresses=torch.empty(1, 1, 0),
        valid=torch.tensor([[True]]),
    )
    features = torch.randn(1, 2, 1, 4)
    target = make_native_predictive_target(
        modality="vision",
        features=features,
        valid=torch.ones(1, 2, 1, dtype=torch.bool),
        importance=torch.tensor([[[0.5], [0.1]]]),
        route_ids=request.route_ids,
        horizons=request.horizons,
        source=request.source,
        evidence=request.evidence,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=DIGEST,
        target_data_digest=DIGEST,
        encoder_digest=DIGEST,
        query_schema_digest=DIGEST,
        validity_semantics="independent-track-mask",
        track_identity_keys=(("object/0", "object/1"),),
    )
    both = predictive_evidence_mass(
        request=request,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, 1]])),
        row_binding_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    assert math.isclose(both, 0.3, rel_tol=1e-6)
    first_only = predictive_evidence_mass(
        request=request,
        target=target,
        assignment=SequenceAssignment(torch.tensor([[0, -1]])),
        row_binding_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    assert math.isclose(first_only, 0.5, rel_tol=1e-6)


def test_manipulation_strength_is_zero_when_the_substitution_is_a_copy() -> None:
    rows = torch.randn(2, 3, 8)
    assert state_manipulation_strength(rows, rows.clone()) == pytest.approx(0.0, abs=1e-6)


def test_manipulation_strength_grows_with_the_size_of_the_substitution() -> None:
    torch.manual_seed(0)
    rows = torch.randn(2, 3, 16)
    near = rows + 0.01 * torch.randn_like(rows)
    far = torch.randn_like(rows)
    assert state_manipulation_strength(rows, near) < state_manipulation_strength(rows, far)


def test_manipulation_strength_honours_row_validity() -> None:
    torch.manual_seed(1)
    rows = torch.randn(1, 2, 8)
    changed = rows.clone()
    changed[0, 1] = torch.randn(8)
    valid_both = torch.ones(1, 2, dtype=torch.bool)
    valid_first = torch.tensor([[True, False]])
    assert state_manipulation_strength(rows, changed, valid=valid_first) == pytest.approx(
        0.0, abs=1e-6
    )
    assert state_manipulation_strength(rows, changed, valid=valid_both) > 0.1
