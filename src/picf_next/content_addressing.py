"""Canonical content identities shared by frozen encoder and cache contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError


def canonical_payload_sha256(domain: str, payload: object) -> str:
    """Hash one JSON-compatible payload under an explicit semantic domain."""

    if not isinstance(domain, str) or not domain:
        raise ContractError("content-address domain must be nonempty text")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(encoded)
    return digest.hexdigest()


def combine_named_sha256(domain: str, components: Sequence[tuple[str, str]]) -> str:
    """Bind an ordered set of named SHA-256 values without string ambiguity."""

    if not isinstance(components, Sequence) or not components:
        raise ContractError("content-address components must be one nonempty sequence")
    normalized: list[dict[str, str]] = []
    names: set[str] = set()
    for name, value in components:
        if not isinstance(name, str) or not name or name in names:
            raise ContractError("content-address component names must be unique nonempty text")
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError("content-address component must be one lowercase SHA-256 digest")
        names.add(name)
        normalized.append({"name": name, "sha256": value})
    return canonical_payload_sha256(domain, normalized)


def canonical_mapping_sha256(domain: str, payload: Mapping[str, object]) -> str:
    """Typed convenience boundary for configuration mappings."""

    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise ContractError("content-address configuration must be a string-keyed mapping")
    return canonical_payload_sha256(domain, dict(payload))


def ndarray_sha256(name: str, value: NDArray[np.generic]) -> str:
    """Bind an array's semantic name, dtype, shape and exact contiguous bytes."""

    if not isinstance(name, str) or not name:
        raise ContractError("content-address array name must be nonempty text")
    array = np.asarray(value)
    if array.dtype.hasobject or not array.size:
        raise ContractError("content-address arrays must be nonempty and object-free")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ContractError("content-address arrays must contain only finite numbers")
    digest = hashlib.sha256()
    digest.update(b"picf-next.ndarray/v1\0")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()
