"""Re-score a shipped v1 P2 causal report under the v2 gate.

The v1 records do not carry the target importance, so the raw margins cannot be put on
the importance-weighted scale after the fact. What they do carry is each sample's own
factual loss, and the ratio

    margin / factual_loss = (weighted-mean deviation change) / (weighted-mean deviation)

is scale-free: the importance factor cancels between numerator and denominator. This
script scores that ratio, which is the strongest statement the shipped artifact supports.

v1 also shipped no negative control, so no arm can be given a NO_EFFECT verdict here --
only EFFECT, WRONG_DIRECTION or INCONCLUSIVE. That is the correct treatment: without a
noise floor there is no way to tell "the model ignores this input" from "this run could
not resolve anything".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picf_next.lingbot_native.causal_evidence import (
    ArmExpectation,
    ArmVerdictLabel,
    InterventionObservation,
    score_causal_evidence,
)

# Every v1 arm corrupts an input the prediction is supposed to depend on, so every one
# of them predicts a positive margin. v1 shipped no NEUTRAL arm at all.
V1_EXPECTATIONS = {
    "absent_source": ArmExpectation.HARMS,
    "batch_shift_control": ArmExpectation.HARMS,
    "batch_shift_source": ArmExpectation.HARMS,
    "row_shift_source": ArmExpectation.HARMS,
    "wrong_time_source": ArmExpectation.HARMS,
    "zero_control": ArmExpectation.HARMS,
    "zero_current_observation": ArmExpectation.HARMS,
    "zero_source": ArmExpectation.HARMS,
}


def load_observations(report: dict) -> list[InterventionObservation]:
    records = report["causal_evidence"]["records"]
    observations: list[InterventionObservation] = []
    for record in records:
        diagnostics = record["diagnostics"]
        factual = float(diagnostics["factual_loss"])
        horizon = int(record["target_global_index"]) - int(record["source_global_index"])
        key = (
            f"rank{record['rank']}-step{record['causal_global_step']}"
            f"-ep{record['source_episode_index']}"
        )
        for item in diagnostics["interventions"]:
            observations.append(
                InterventionObservation(
                    arm=str(item["name"]),
                    sample_key=key,
                    horizon=max(1, horizon),
                    factual_loss=factual,
                    intervened_loss=float(item["loss"]),
                    # Setting the mass to the factual loss makes normalised_margin the
                    # scale-free ratio described in the module docstring and puts the
                    # factual loss itself at 1.0 on that scale.
                    evidence_mass=factual,
                    valid_target_count=int(diagnostics["valid_target_count"]),
                    prediction_displacement=float(item["normalized_prediction_l1"]),
                    # v1 never measured how much a substitution changed its source, so
                    # no sample can be excluded for having tested nothing.
                    source_manipulation=None,
                )
            )
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--band-fraction", type=float, default=0.01)
    arguments = parser.parse_args()

    report = json.loads(arguments.report.read_text())
    observations = load_observations(report)
    scored = score_causal_evidence(
        observations,
        expectations=V1_EXPECTATIONS,
        minimum_samples=12,
        equivalence_band_fraction=arguments.band_fraction,
        require_noise_floor=False,
    )

    shipped = report["causal_evidence"]["interventions"]
    print("=" * 112)
    print(f"v2 re-score of {arguments.report.name}")
    print(f"shipped status: {report['causal_evidence']['status']}   "
          f"v2 status: {scored.status}")
    print(f"equivalence band: {scored.equivalence_band:.4f} of a factual loss of "
          f"{scored.reference_scale:.1f} (scale-free)   "
          f"noise floor certified: {scored.noise_floor_within_band}")
    print("=" * 112)
    header = (
        f"{'arm':<26}{'v1':>6}{'v2':>18}{'k/n':>8}{'sign p':>10}{'holm p':>10}"
        f"{'effect':>11}{'95% CI':>24}"
    )
    print(header)
    print("-" * len(header))
    for arm in scored.arms:
        v1 = "PASS" if shipped[arm.arm]["pass"] else "FAIL"
        print(
            f"{arm.arm:<26}{v1:>6}{arm.label.value:>18}"
            f"{f'{arm.positive_samples}/{arm.scored_samples}':>8}"
            f"{arm.sign_p_value:>10.2}{arm.holm_p_value:>10.2}"
            f"{arm.mean_normalised_margin:>10.3%} "
            f"{f'[{arm.bootstrap_low:+.2%},{arm.bootstrap_high:+.2%}]':>23}"
        )
    print()

    flipped = [
        arm.arm
        for arm in scored.arms
        if (shipped[arm.arm]["pass"]) != (arm.label is ArmVerdictLabel.EFFECT)
    ]
    print(f"arms whose verdict changes under v2: {flipped}")
    print()
    print("Reading:")
    print("  EFFECT        the margin is significantly positive after multiplicity control")
    print("  INCONCLUSIVE  not significant, and with no negative control the run cannot")
    print("                bound the effect either -- absence of evidence, nothing more")

    if arguments.output is not None:
        payload = scored.as_dict()
        payload["source_report"] = str(arguments.report)
        payload["shipped_status"] = report["causal_evidence"]["status"]
        arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
