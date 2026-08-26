"""Owner-frozen semantics for the LingBot-native predictive objective.

This module contains no model component.  It makes the semantic claim and the
visible-evidence weighting used by a learned pilot explicit, hash-bound and
machine-checkable before training authorization can be issued.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREDICTIVE_OBJECTIVE_DECISION_SCHEMA = "picf-next.lingbot-predictive-objective-decision.v1"

PREDICTIVE_OBJECTIVE_PRIOR_CURRENT = "prior-current-predictive-correction/v1"
PREDICTIVE_OBJECTIVE_ACTION_CONDITIONED_FUTURE = "posterior-future-action-conditioned/v1"
PREDICTIVE_OBJECTIVE_POSTERIOR_FUTURE_IDENTITY = (
    "posterior-future-action-marginalized-identity-plus-prior-overshoot/v1"
)

PREDICTIVE_OBJECTIVE_CLAIMS = {
    PREDICTIVE_OBJECTIVE_PRIOR_CURRENT: "executed-control-predictive-correction/v1",
    PREDICTIVE_OBJECTIVE_ACTION_CONDITIONED_FUTURE: "controlled-future-dynamics/v1",
    PREDICTIVE_OBJECTIVE_POSTERIOR_FUTURE_IDENTITY: ("action-marginalized-identity-coreference/v1"),
}

PREDICTIVE_VISIBLE_SUPPORT_RELATIVE = (
    "relative-visible-image-fraction-renormalized-to-valid-count/v1"
)
PREDICTIVE_VISIBLE_SUPPORT_ABSOLUTE = "absolute-visible-image-fraction/v1"
PREDICTIVE_VISIBLE_SUPPORT_WEIGHTINGS = frozenset(
    {
        PREDICTIVE_VISIBLE_SUPPORT_RELATIVE,
        PREDICTIVE_VISIBLE_SUPPORT_ABSOLUTE,
    }
)

# These identify the code that is presently implemented.  Changing either
# constant is not sufficient to change behavior: objective tests and the
# execution-contract digest must change in the same reviewed commit.
IMPLEMENTED_PREDICTIVE_OBJECTIVE = PREDICTIVE_OBJECTIVE_PRIOR_CURRENT
IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING = PREDICTIVE_VISIBLE_SUPPORT_ABSOLUTE


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _minimum_visible_fraction(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("minimum visible fraction must use an exact hexadecimal float")
    try:
        measured = float.fromhex(value)
    except ValueError as error:
        raise ValueError("minimum visible fraction must use an exact hexadecimal float") from error
    if not math.isfinite(measured) or not 0 <= measured < 1:
        raise ValueError("minimum visible fraction must lie in [0,1)")
    if measured.hex() != value:
        raise ValueError("minimum visible fraction is not in canonical hexadecimal form")
    return measured


@dataclass(frozen=True, slots=True)
class PredictiveObjectiveDecision:
    reviewer: str
    temporal_objective: str
    claim_scope: str
    visible_support_weighting: str
    minimum_visible_fraction: float
    decision_record_path: Path
    decision_record_sha256: str


def validate_predictive_objective_decision(
    value: object,
    *,
    expected_temporal_objective: str | None = None,
    expected_visible_support_weighting: str | None = None,
    expected_minimum_visible_fraction: float | None = None,
) -> PredictiveObjectiveDecision:
    """Validate one owner decision and re-hash the reviewed decision record."""

    required = {
        "schema",
        "status",
        "reviewer",
        "temporal_objective",
        "claim_scope",
        "visible_support",
        "decision_record",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("predictive objective decision fields differ from schema")
    if value["schema"] != PREDICTIVE_OBJECTIVE_DECISION_SCHEMA or value["status"] != "PASS":
        raise ValueError("predictive objective decision has not passed")
    reviewer = value["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("predictive objective decision requires an explicit reviewer")
    temporal_objective = value["temporal_objective"]
    if temporal_objective not in PREDICTIVE_OBJECTIVE_CLAIMS:
        raise ValueError("predictive temporal objective is outside the reviewed alternatives")
    claim_scope = value["claim_scope"]
    if claim_scope != PREDICTIVE_OBJECTIVE_CLAIMS[temporal_objective]:
        raise ValueError("predictive claim scope differs from the selected objective")

    visible_support = value["visible_support"]
    if not isinstance(visible_support, Mapping) or set(visible_support) != {
        "weighting",
        "minimum_visible_fraction_hex",
    }:
        raise ValueError("predictive visible-support decision is malformed")
    weighting = visible_support["weighting"]
    if weighting not in PREDICTIVE_VISIBLE_SUPPORT_WEIGHTINGS:
        raise ValueError("predictive visible-support weighting is outside reviewed alternatives")
    minimum = _minimum_visible_fraction(visible_support["minimum_visible_fraction_hex"])

    decision_record = value["decision_record"]
    if not isinstance(decision_record, Mapping) or set(decision_record) != {"path", "sha256"}:
        raise ValueError("predictive decision-record reference is malformed")
    path_value = decision_record["path"]
    path = Path(path_value) if isinstance(path_value, str) else None
    digest = _sha256(decision_record["sha256"], name="predictive decision record sha256")
    if (
        path is None
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ):
        raise ValueError("predictive decision record differs from its reviewed content")

    if (
        expected_temporal_objective is not None
        and temporal_objective != expected_temporal_objective
    ):
        raise ValueError("predictive owner decision targets another temporal objective")
    if (
        expected_visible_support_weighting is not None
        and weighting != expected_visible_support_weighting
    ):
        raise ValueError("predictive owner decision targets another support weighting")
    if expected_minimum_visible_fraction is not None:
        if (
            isinstance(expected_minimum_visible_fraction, bool)
            or not isinstance(expected_minimum_visible_fraction, (int, float))
            or not math.isfinite(expected_minimum_visible_fraction)
            or not 0 <= expected_minimum_visible_fraction < 1
        ):
            raise ValueError("expected minimum visible fraction must lie in [0,1)")
        if minimum.hex() != float(expected_minimum_visible_fraction).hex():
            raise ValueError("predictive owner decision targets another support threshold")

    return PredictiveObjectiveDecision(
        reviewer=reviewer.strip(),
        temporal_objective=temporal_objective,
        claim_scope=claim_scope,
        visible_support_weighting=weighting,
        minimum_visible_fraction=minimum,
        decision_record_path=path,
        decision_record_sha256=digest,
    )


def load_predictive_objective_decision(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_temporal_objective: str | None = None,
    expected_visible_support_weighting: str | None = None,
    expected_minimum_visible_fraction: float | None = None,
) -> PredictiveObjectiveDecision:
    """Load one immutable owner decision from a real JSON file."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("predictive objective decision must be one real JSON file")
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != _sha256(
        expected_sha256,
        name="predictive objective decision sha256",
    ):
        raise ValueError("predictive objective decision differs from its expected digest")
    try:
        value: Any = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("predictive objective decision is not valid ASCII JSON") from error
    return validate_predictive_objective_decision(
        value,
        expected_temporal_objective=expected_temporal_objective,
        expected_visible_support_weighting=expected_visible_support_weighting,
        expected_minimum_visible_fraction=expected_minimum_visible_fraction,
    )
