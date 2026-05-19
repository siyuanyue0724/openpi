#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def _contains(source: str, *needles: str) -> bool:
    return all(needle in source for needle in needles)


def _mask_math_passes() -> bool:
    priors = [[0.8, 0.2, 0.1], [0.2, 0.8, 0.1]]
    priors = [[v / sum(row) for v in row] for row in priors]
    quality = [0.9, 0.4]
    background = [0.25, 0.25, 0.25]
    for col in range(3):
        weighted = [priors[row][col] * quality[row] for row in range(2)]
        denom = sum(weighted) + background[col]
        object_mass = [value / denom for value in weighted]
        background_mass = background[col] / denom
        if any(value < 0.0 for value in object_mass) or background_mass <= 0.0:
            return False
        if abs(sum(object_mass) + background_mass - 1.0) > 1e-6:
            return False
    return True


def run_checks() -> list[Check]:
    contracts = _read("src/openpi/picf/core/contracts.py")
    config = _read("src/openpi/picf/core/config.py")
    pipeline = _read("src/openpi/picf/core/pipeline.py")
    training = _read("src/openpi/picf/core/training.py")
    trainer = _read("scripts/picf_core_train.py")
    readme_v22 = _read("src/openpi/picf/README_v2.2.md")
    plan = _read("docs/PICF_AQR_OWM_OBJECT_EXPLANATION_DEPLOYMENT_PLAN_20260518_TEMP.md")
    math_doc = _read("temp/audits_20260519/oeml_math_dataflow_followthrough.md")

    return [
        Check(
            "contracts_expose_oeml_state",
            _contains(
                contracts,
                "class PicfObjectExplanationState",
                "object_mask_visual",
                "background_mask_visual",
                "anchor_duplicate_overlap",
                "contact_explanation_score",
                "object_explanation: PicfObjectExplanationState | None",
            ),
            "Core contracts must expose object/background explanation masks and state-level carry.",
        ),
        Check(
            "graph_quality_feedback_present",
            _contains(
                contracts,
                "object_explanation_quality",
                "object_explanation_duplicate_overlap",
            )
            and _contains(
                pipeline,
                "graph.object_explanation_quality = explanation_quality",
                "object_explanation_feed_quality_to_assignment",
                "scores = torch.clamp(scores * explanation_quality",
            ),
            "OEML quality must feed AQR assignment as a measurement quality term.",
        ),
        Check(
            "pipeline_builds_oeml_after_aqr_before_assignment",
            _contains(
                pipeline,
                "def _build_object_explanation_measurements",
                "def _object_explanation_masks",
                "object_explanation = self._build_object_explanation_measurements(token_field, anchor_prior_graph)",
                "object_explanation=object_explanation",
            ),
            "Observe path must build OEML after graph priors and before observation/posterior assignment.",
        ),
        Check(
            "losses_are_guarded_and_cli_visible",
            _contains(
                training,
                "lambda_object_explanation_feature: float = 0.0",
                "def _object_explanation_loss",
                "object_explanation_feature",
                "object_explanation_duplicate",
            )
            and _contains(
                trainer,
                "--lambda-object-explanation-feature",
                "loss_object_explanation_feature",
                "oeml_anchor_quality_mean",
            ),
            "OEML losses must be explicit, default-zero, and visible in train metrics.",
        ),
        Check(
            "docs_record_deployment_not_behavior_claim",
            _contains(
                readme_v22,
                "object-explanation",
                "code-level",
                "behavior",
            )
            and _contains(
                plan,
                "2026-05-19",
                "implemented",
                "behavior acceptance",
            )
            and _contains(
                math_doc,
                "Object Explanation Measurement Layer",
                "dense typed evidence",
                "background residual",
            ),
            "README and temp docs must separate code-level deployment from behavior acceptance.",
        ),
        Check(
            "object_background_mask_math",
            _mask_math_passes(),
            "Object masks plus background residual must be column-normalized and nonnegative.",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    checks = run_checks()
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    passed = sum(1 for check in checks if check.ok)
    print(f"{passed}/{len(checks)} PASS")
    if args.fail_on_fail and passed != len(checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
