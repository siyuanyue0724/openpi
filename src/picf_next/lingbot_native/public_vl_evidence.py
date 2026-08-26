"""Canonical processor and schedule evidence for public native-VL records."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence

import torch
from torch import Tensor

from picf_next.contracts import ContractError

PUBLIC_VL_SCHEDULE_FIELDS = (
    "family",
    "record_id",
    "record_sha256",
    "image_rgb_sha256",
    "image_height",
    "image_width",
    "source_row_index",
    "source_subindex",
    "supervised_token_count",
    "target_answer_sha256",
    "user_text_sha256",
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ContractError("public native VL evidence is not canonical JSON") from error


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"public native VL {name} must be one lowercase SHA-256")
    return value


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"public native VL {name} must be a nonnegative integer")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    result = _require_nonnegative_integer(value, name=name)
    if result == 0:
        raise ContractError(f"public native VL {name} must be positive")
    return result


def text_sha256(value: str) -> str:
    """Hash one exact UTF-8 instruction or target string."""

    if not isinstance(value, str) or not value or "\0" in value:
        raise ContractError("public native VL evidence text must be nonempty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_CANONICAL_TENSOR_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
}


def cpu_tensor_evidence(tensor: Tensor) -> dict[str, object]:
    """Return a canonical, bit-exact digest for one dense CPU tensor.

    The payload is a canonical JSON header followed by the contiguous little-endian
    storage bytes. This binds floating-point bit patterns, including signed zero and
    NaN payloads, without the lossy decimal conversion used by ``Tensor.tolist``.
    """

    if not isinstance(tensor, Tensor):
        raise TypeError("public native VL tensor evidence requires a tensor")
    if tensor.device.type != "cpu":
        raise ContractError("public native VL tensor evidence must remain on CPU")
    if sys.byteorder != "little":
        raise ContractError("public native VL tensor evidence requires little-endian CPU storage")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ContractError("public native VL tensor evidence requires dense unquantized storage")
    if tensor.dtype not in _CANONICAL_TENSOR_DTYPES:
        raise ContractError(f"public native VL tensor dtype is unsupported: {tensor.dtype}")
    try:
        value = tensor.detach().resolve_conj().resolve_neg().contiguous()
        header = {
            "byte_order": "little",
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "shape": list(value.shape),
        }
        storage = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest = hashlib.sha256()
        digest.update(_canonical_bytes(header))
        digest.update(b"\0")
        digest.update(storage)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ContractError("public native VL tensor cannot be canonicalized") from error
    return {**header, "sha256": digest.hexdigest()}


def cpu_tensor_sha256(tensor: Tensor) -> str:
    """Hash one tensor using the canonical bit-exact CPU evidence protocol."""

    return str(cpu_tensor_evidence(tensor)["sha256"])


def tensor_evidence_aggregate_sha256(
    rows: Sequence[Sequence[object]],
) -> str:
    """Hash a nonempty ordered tensor-evidence schedule."""

    normalized: list[list[object]] = []
    for row in rows:
        if isinstance(row, str | bytes) or not row:
            raise ContractError("public native VL tensor aggregate row is invalid")
        normalized.append(list(row))
    if not normalized:
        raise ContractError("public native VL tensor aggregate is empty")
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def public_vl_schedule_row(
    *,
    family: object,
    record_id: object,
    record_sha256: object,
    image_rgb_sha256: object,
    image_height: object,
    image_width: object,
    source_row_index: object,
    source_subindex: object,
    supervised_token_count: object,
    target_answer_sha256: object,
    user_text_sha256: object,
) -> list[object]:
    """Return one row in the comparator's frozen public schedule field order."""

    if family not in ("referring", "vqa"):
        raise ContractError("public native VL schedule family is unsupported")
    if not isinstance(record_id, str) or not record_id or "\0" in record_id:
        raise ContractError("public native VL schedule record ID is invalid")
    values: dict[str, object] = {
        "family": family,
        "record_id": record_id,
        "record_sha256": _require_sha256(record_sha256, name="record digest"),
        "image_rgb_sha256": _require_sha256(image_rgb_sha256, name="image RGB digest"),
        "image_height": _require_positive_integer(image_height, name="image height"),
        "image_width": _require_positive_integer(image_width, name="image width"),
        "source_row_index": _require_nonnegative_integer(
            source_row_index,
            name="source row index",
        ),
        "source_subindex": _require_nonnegative_integer(
            source_subindex,
            name="source subindex",
        ),
        "supervised_token_count": _require_positive_integer(
            supervised_token_count,
            name="supervised token count",
        ),
        "target_answer_sha256": _require_sha256(
            target_answer_sha256,
            name="target-answer digest",
        ),
        "user_text_sha256": _require_sha256(user_text_sha256, name="user-text digest"),
    }
    return [values[field] for field in PUBLIC_VL_SCHEDULE_FIELDS]


def public_vl_schedule_sha256(rows: Sequence[Sequence[object]]) -> str:
    """Hash a complete ordered schedule using the comparator's JSON convention."""

    normalized: list[list[object]] = []
    for row in rows:
        if isinstance(row, str | bytes) or len(row) != len(PUBLIC_VL_SCHEDULE_FIELDS):
            raise ContractError("public native VL schedule row width changed")
        normalized.append(list(row))
    if not normalized:
        raise ContractError("public native VL schedule is empty")
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def require_cpu_tensor(tensor: object, *, name: str) -> Tensor:
    """Validate one processor output before any evidence is derived from it."""

    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ContractError(f"public native VL processor {name} must remain on CPU")
    return tensor
