from __future__ import annotations

import numpy as np
import pytest

from tools.audit_calvin_measurement_contract import legacy_one_token_fixed_point


def test_legacy_fixed_point_reproduces_cascading_object_deletion() -> None:
    ownership = np.asarray(
        [
            [0.6, 0.4, 0.0],
            [0.0, 0.7, 0.3],
            [0.0, 0.4, 0.6],
        ],
        dtype=np.float64,
    )
    supervised = np.ones(3, dtype=np.bool_)

    keep = legacy_one_token_fixed_point(ownership, supervised)

    assert keep.tolist() == [False, True]


def test_legacy_fixed_point_retains_independent_one_token_objects() -> None:
    ownership = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    keep = legacy_one_token_fixed_point(ownership, np.ones(2, dtype=np.bool_))

    assert keep.tolist() == [True, True]


@pytest.mark.parametrize("minimum", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_legacy_fixed_point_rejects_invalid_mass(minimum: float | bool) -> None:
    with pytest.raises(ValueError, match="minimum mass"):
        legacy_one_token_fixed_point(
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            np.ones(1, dtype=np.bool_),
            minimum_mass=minimum,
        )
