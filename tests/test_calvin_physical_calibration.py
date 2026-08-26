from __future__ import annotations

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_calibration import (
    calvin_depth_consistent_fraction,
    calvin_depth_consistent_supervision,
)


def test_depth_consistency_marks_disagreement_unknown_at_schema_tolerance() -> None:
    source = np.asarray([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    rendered = np.asarray([[0.5, 0.509], [1.011, 0.99]], dtype=np.float32)

    supervised = calvin_depth_consistent_supervision(source, rendered)

    assert supervised.tolist() == [[True, True], [False, True]]
    assert not supervised.flags.writeable
    assert calvin_depth_consistent_fraction(supervised) == 0.75


def test_gross_depth_misalignment_cannot_publish_owner_supervision() -> None:
    source = np.ones((3, 4), dtype=np.float32)
    rendered = source + np.float32(0.03)

    supervised = calvin_depth_consistent_supervision(source, rendered)

    assert not supervised.any()
    assert calvin_depth_consistent_fraction(supervised) == 0.0


@pytest.mark.parametrize(
    ("source", "rendered"),
    [
        (np.ones((2, 2), dtype=np.int64), np.ones((2, 2), dtype=np.float32)),
        (np.ones((2, 2), dtype=np.float32), np.ones((2, 3), dtype=np.float32)),
        (np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)),
        (
            np.asarray([[1.0, np.nan]], dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        ),
    ],
)
def test_depth_consistency_rejects_invalid_sensor_arrays(
    source: np.ndarray,
    rendered: np.ndarray,
) -> None:
    with pytest.raises(ContractError, match="inputs are invalid"):
        calvin_depth_consistent_supervision(source, rendered)


def test_depth_consistent_fraction_requires_bool_raster() -> None:
    with pytest.raises(ContractError, match="mask is invalid"):
        calvin_depth_consistent_fraction(np.ones((2, 2), dtype=np.float32))
