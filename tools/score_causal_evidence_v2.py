"""Score an instrumented P2 causal report under the v2 gate.

Unlike the legacy re-scoring path, records produced by the patched runner carry the two
measurements the v1 records lacked:

  * ``evidence_mass`` -- the mean target importance over the entries the loss reduced
    over, which converts the importance-scaled training loss into the importance-weighted
    mean the statistics need;
  * ``source_manipulation`` -- how far each substitution actually moved the posterior, so
    an arm that changed nothing is reported as a null intervention rather than as evidence
    that the model ignores its input.

``matched_noise_source`` is the negative control: the posterior displaced by exactly as
much as the wrong-time substitution displaces it, using noise that carries no temporal or
episode information. It is the noise floor every other arm is judged against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picf_next.lingbot_native.causal_evidence import (
    ArmExpectation,
    InterventionObservation,
    score_causal_evidence,
)

EXPECTATIONS = {
    "absent_source": ArmExpectation.HARMS,
    "batch_shift_control": ArmExpectation.HARMS,
    "batch_shift_source": ArmExpectation.HARMS,
    "row_shift_source": ArmExpectation.HARMS,
    "wrong_time_source": ArmExpectation.HARMS,
    "zero_control": ArmExpectation.HARMS,
    "zero_current_observation": ArmExpectation.HARMS,
    "zero_source": ArmExpectation.HARMS,
    "matched_noise_source": ArmExpectation.NEUTRAL,
}


def load(report: dict) -> list[InterventionObservation]:
    records = report["causal_evidence"]["records"]
    observations: list[InterventionObservation] = []
    for record in records:
        diagnostics = record["diagnostics"]
        mass = record.get("evidence_mass")
        if mass is None:
            raise SystemExit(
                "this report predates the evidence-mass instrumentation; use "
                "tools/rescore_v1_causal_report.py for it instead"
            )
        manipulation = record.get("source_manipulation") or {}
        horizon = int(record.get("horizon") or 1)
        key = (
            f"rank{record['rank']}-step{record['causal_global_step']}"
            f"-ep{record['source_episode_index']}"
        )
        for item in diagnostics["interventions"]:
            name = str(item["name"])
            observations.append(
                InterventionObservation(
                    arm=name,
                    sample_key=key,
                    horizon=horizon,
                    factual_loss=float(diagnostics["factual_loss"]),
                    intervened_loss=float(item["loss"]),
                    evidence_mass=float(mass),
                    valid_target_count=int(diagnostics["valid_target_count"]),
                    prediction_displacement=float(item["normalized_prediction_l1"]),
                    source_manipulation=(
                        float(manipulation[name]) if name in manipulation else None
                    ),
                )
            )
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--band-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-manipulation", type=float, default=0.0)
    arguments = parser.parse_args()

    report = json.loads(arguments.report.read_text())
    observations = load(report)
    scored = score_causal_evidence(
        observations,
        expectations=EXPECTATIONS,
        minimum_samples=12,
        equivalence_band_fraction=arguments.band_fraction,
        minimum_manipulation=arguments.minimum_manipulation,
    )

    print("=" * 118)
    print(f"v2 causal evidence  |  {arguments.report.name}")
    print(f"horizons {list(scored.horizons)}   shipped v1 status "
          f"{report['causal_evidence'].get('status')}   v2 status {scored.status}")
    print(
        f"factual loss on the importance-weighted scale: {scored.reference_scale:.4f} "
        f"(0 = exact, ~1.13 = chance)"
    )
    print(
        f"equivalence band {scored.equivalence_band:.5f}   noise floor arm "
        f"{scored.noise_floor_arm}   floor certified {scored.noise_floor_within_band}"
    )
    print("=" * 118)
    header = (
        f"{'arm':<26}{'expect':>9}{'verdict':>19}{'k/n':>9}{'holm p':>10}"
        f"{'effect':>11}{'95% CI':>25}{'manip':>9}"
    )
    print(header)
    print("-" * len(header))
    for arm in scored.arms:
        manipulation = (
            f"{arm.mean_source_manipulation:.4f}"
            if arm.mean_source_manipulation is not None
            else "n/a"
        )
        print(
            f"{arm.arm:<26}{arm.expectation.value:>9}{arm.label.value:>19}"
            f"{f'{arm.positive_samples}/{arm.scored_samples}':>9}{arm.holm_p_value:>10.2}"
            f"{arm.mean_normalised_margin:>11.4f}"
            f"{f'[{arm.bootstrap_low:+.4f},{arm.bootstrap_high:+.4f}]':>25}{manipulation:>9}"
        )
    print()
    control = next(
        (a for a in scored.arms if a.expectation is ArmExpectation.NEUTRAL), None
    )
    if control is not None:
        print("Excess over the magnitude-matched noise control:")
        for arm in scored.arms:
            if arm.expectation is ArmExpectation.NEUTRAL:
                continue
            excess = arm.mean_normalised_margin - control.mean_normalised_margin
            print(f"  {arm.arm:<26}{excess:+.5f}")
    if arguments.output is not None:
        payload = scored.as_dict()
        payload["source_report"] = str(arguments.report)
        arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
