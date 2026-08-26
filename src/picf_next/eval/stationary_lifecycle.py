"""Deterministic lifecycle-calibration evidence derived from fixed replay."""

from __future__ import annotations

from typing import Any, Final, cast

from picf_next.eval.stationary_replay import (
    STATIONARY_FIXED_REPLAY_PASS,
    validate_stationary_fixed_replay,
)

STATIONARY_LIFECYCLE_CALIBRATION_SCHEMA: Final = "picf-next.stationary-lifecycle-calibration.v1"
STATIONARY_LIFECYCLE_CALIBRATION_PASS: Final = "PASS"
STATIONARY_LIFECYCLE_CALIBRATION_FAIL: Final = "FAIL"

_SPLITS: Final = ("validation", "heldout")
_MODELS: Final = ("fresh_m2", "candidate")


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def build_stationary_lifecycle_calibration(
    fixed_replay: object,
    *,
    fixed_replay_sha256: str,
) -> dict[str, Any]:
    """Project preregistered Bernoulli scores from one validated replay.

    The report introduces no new threshold or data pass. Survival and direct
    conditional-detection scores are sliced from the same clips already used
    by the fixed-checkpoint decision, preventing post-hoc sample selection.
    """

    replay = validate_stationary_fixed_replay(fixed_replay)
    replay_sha256 = _sha256(fixed_replay_sha256, "fixed replay SHA-256")
    split_reports: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for split_name in _SPLITS:
        replay_split = replay["splits"][split_name]
        models = {
            model_name: {
                "survival_log_loss": replay_split["models"][model_name]["loss_dynamics_survival"],
                "visibility_log_loss": replay_split["models"][model_name][
                    "loss_dynamics_visibility"
                ],
            }
            for model_name in _MODELS
        }
        comparisons = {
            "candidate_survival_log_loss_not_worse": replay_split["comparisons"][
                "candidate_loss_dynamics_survival_not_worse"
            ],
            "candidate_visibility_log_loss_not_worse": replay_split["comparisons"][
                "candidate_loss_dynamics_visibility_not_worse"
            ],
        }
        split_reports[split_name] = {
            "clip_count": replay_split["clip_count"],
            "models": models,
            "comparisons": comparisons,
        }
        checks.update({f"{split_name}_{name}": passed for name, passed in comparisons.items()})
    failed = sorted(name for name, passed in checks.items() if not passed)
    status = (
        STATIONARY_LIFECYCLE_CALIBRATION_PASS
        if replay["status"] == STATIONARY_FIXED_REPLAY_PASS and not failed
        else STATIONARY_LIFECYCLE_CALIBRATION_FAIL
    )
    return {
        "schema": STATIONARY_LIFECYCLE_CALIBRATION_SCHEMA,
        "status": status,
        "protocol": {
            "score": "independently-aligned-bernoulli-negative-log-likelihood.v1",
            "target_use": "post-forward-loss-and-evaluation-only",
            "control": "fresh-m2-prior-on-identical-frozen-clips",
            "split_names": list(_SPLITS),
        },
        "bindings": {
            "fixed_checkpoint_replay_sha256": replay_sha256,
            "candidate_checkpoint_sha256": replay["bindings"]["candidate_checkpoint_sha256"],
            "candidate_report_sha256": replay["bindings"]["candidate_report_sha256"],
        },
        "splits": split_reports,
        "checks": checks,
        "failed_checks": failed,
        "long_training_authorized": False,
    }


def validate_stationary_lifecycle_calibration(
    payload: object,
    *,
    fixed_replay: object,
    fixed_replay_sha256: str,
) -> dict[str, Any]:
    """Require the exact deterministic lifecycle projection of fixed replay."""

    if not isinstance(payload, dict):
        raise ValueError("stationary lifecycle calibration must contain one JSON object")
    report = cast(dict[str, Any], payload)
    expected = build_stationary_lifecycle_calibration(
        fixed_replay,
        fixed_replay_sha256=fixed_replay_sha256,
    )
    if report != expected:
        raise ValueError("stationary lifecycle calibration differs from fixed replay")
    return report
