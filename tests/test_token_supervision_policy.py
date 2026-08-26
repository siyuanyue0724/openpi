from __future__ import annotations

import pytest

from picf_next.contracts import ContractError
from picf_next.data.token_supervision_policy import (
    KNOWN_PIXEL_TOKEN_SUPERVISION_SCHEMA,
    build_known_pixel_token_supervision_policy,
    token_supervision_policy_sha256,
    validate_known_pixel_token_supervision_policy,
)


def test_known_pixel_policy_is_canonical_hash_stable_and_loss_only() -> None:
    policy = build_known_pixel_token_supervision_policy()

    assert policy["schema"] == KNOWN_PIXEL_TOKEN_SUPERVISION_SCHEMA
    assert policy["runtime_input"] is False
    assert policy["unknown_pixel_semantics"] == "zero-loss-mass-never-context"
    assert policy["minimum_observed_fraction_hex"] == "0x0.0p+0"
    assert validate_known_pixel_token_supervision_policy(policy) == policy
    assert token_supervision_policy_sha256(policy) == token_supervision_policy_sha256(
        dict(reversed(policy.items()))
    )


@pytest.mark.parametrize("minimum", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_known_pixel_policy_rejects_invalid_minimum(minimum: object) -> None:
    with pytest.raises(ContractError, match="minimum observed"):
        build_known_pixel_token_supervision_policy(
            minimum_observed_fraction=minimum,  # type: ignore[arg-type]
        )


def test_known_pixel_policy_rejects_semantic_or_noncanonical_drift() -> None:
    policy = build_known_pixel_token_supervision_policy()
    policy["unknown_pixel_semantics"] = "unknown-becomes-context"
    with pytest.raises(ContractError, match="semantics changed"):
        validate_known_pixel_token_supervision_policy(policy)

    policy = build_known_pixel_token_supervision_policy()
    policy["minimum_observed_fraction_hex"] = "0x0p+0"
    with pytest.raises(ContractError, match="not canonical"):
        validate_known_pixel_token_supervision_policy(policy)
