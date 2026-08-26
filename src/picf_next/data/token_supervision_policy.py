"""Frozen known-pixel projection and categorical diagnostic measure."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from picf_next.contracts import ContractError

KNOWN_PIXEL_TOKEN_SUPERVISION_SCHEMA = "picf-next.known-pixel-marginalized-token-supervision.v1"
_FIELDS = {
    "schema",
    "runtime_input",
    "owner_space",
    "target_measure",
    "unknown_pixel_semantics",
    "token_loss_weight",
    "reduction",
    "assignment_cost",
    "minimum_observed_fraction_hex",
}
_FIXED = {
    "schema": KNOWN_PIXEL_TOKEN_SUPERVISION_SCHEMA,
    "runtime_input": False,
    "owner_space": "exclusive-physical-object-plus-context",
    "target_measure": "known-owner-mass-conditioned-within-token",
    "unknown_pixel_semantics": "zero-loss-mass-never-context",
    "token_loss_weight": "observed-pixel-fraction",
    "reduction": "sum-weighted-loss-over-sum-observed-mass",
    "assignment_cost": "observed-mass-weighted-bce-plus-dice",
}


def _minimum_observed_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError("minimum observed token fraction must be real-valued")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractError("minimum observed token fraction must lie in [0,1]")
    return result


def build_known_pixel_token_supervision_policy(
    *,
    minimum_observed_fraction: float = 0.0,
) -> dict[str, object]:
    """Return the exact token measure used by audit, assignment and diagnostics."""

    minimum = _minimum_observed_fraction(minimum_observed_fraction)
    return {
        **_FIXED,
        "minimum_observed_fraction_hex": minimum.hex(),
    }


def validate_known_pixel_token_supervision_policy(
    value: object,
) -> dict[str, Any]:
    """Validate one exact policy payload and return a detached plain mapping."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError("known-pixel token supervision policy must be a mapping")
    if set(value) != _FIELDS:
        raise ContractError("known-pixel token supervision policy fields changed")
    if any(value.get(name) != expected for name, expected in _FIXED.items()):
        raise ContractError("known-pixel token supervision policy semantics changed")
    encoded = value["minimum_observed_fraction_hex"]
    if not isinstance(encoded, str):
        raise ContractError("minimum observed token fraction must use hexadecimal text")
    try:
        minimum = float.fromhex(encoded)
    except ValueError as error:
        raise ContractError("minimum observed token fraction is invalid") from error
    minimum = _minimum_observed_fraction(minimum)
    if minimum.hex() != encoded:
        raise ContractError("minimum observed token fraction is not canonical")
    return {**_FIXED, "minimum_observed_fraction_hex": encoded}


def token_supervision_policy_sha256(value: object) -> str:
    """Hash a validated policy under a dedicated domain separator."""

    policy = validate_known_pixel_token_supervision_policy(value)
    digest = hashlib.sha256()
    digest.update(b"picf-next.known-pixel-token-supervision-policy.v1\0")
    digest.update(
        json.dumps(
            policy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    return digest.hexdigest()
