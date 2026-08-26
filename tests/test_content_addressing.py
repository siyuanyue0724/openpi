from __future__ import annotations

import numpy as np
import pytest

from picf_next.content_addressing import (
    canonical_mapping_sha256,
    canonical_payload_sha256,
    combine_named_sha256,
    ndarray_sha256,
)
from picf_next.contracts import ContractError


def test_canonical_payload_hash_is_order_invariant_but_domain_separated() -> None:
    first = canonical_mapping_sha256("domain-a", {"b": 2, "a": 1})
    second = canonical_mapping_sha256("domain-a", {"a": 1, "b": 2})

    assert first == second
    assert first != canonical_mapping_sha256("domain-b", {"a": 1, "b": 2})


def test_named_hash_combination_binds_names_and_order() -> None:
    left = canonical_payload_sha256("leaf", {"value": 1})
    right = canonical_payload_sha256("leaf", {"value": 2})

    combined = combine_named_sha256("root", (("left", left), ("right", right)))

    assert combined != combine_named_sha256("root", (("right", right), ("left", left)))
    with pytest.raises(ContractError, match="unique"):
        combine_named_sha256("root", (("left", left), ("left", right)))


def test_array_hash_binds_name_dtype_shape_and_exact_bytes() -> None:
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    reference = ndarray_sha256("rgb", value)

    assert reference == ndarray_sha256("rgb", np.asfortranarray(value))
    assert reference != ndarray_sha256("depth", value)
    assert reference != ndarray_sha256("rgb", value.astype(np.float64))
    assert reference != ndarray_sha256("rgb", value.reshape(3, 2))
    changed = value.copy()
    changed[0, 0] = 1.0
    assert reference != ndarray_sha256("rgb", changed)


def test_array_hash_rejects_nonfinite_and_object_values() -> None:
    with pytest.raises(ContractError, match="finite"):
        ndarray_sha256("bad", np.asarray([np.nan], dtype=np.float32))
    with pytest.raises(ContractError, match="object-free"):
        ndarray_sha256("bad", np.asarray([object()], dtype=object))
