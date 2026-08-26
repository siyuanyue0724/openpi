"""Manifest-only semantic audit for frozen CALVIN dense evidence caches.

This module validates the production CALVIN cache ABI and record-table
semantics.  It deliberately never calls ``evidence_for`` and therefore never
loads a token shard.  The returned report is canonically content-addressed so
that downstream launch receipts can pin the exact aggregate that was audited.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from picf_next.content_addressing import canonical_payload_sha256
from picf_next.contracts import ContractError
from picf_next.data.dense_evidence_cache import (
    DenseEvidenceCacheContract,
    FrozenDenseEvidenceCache,
    FrozenDenseEvidenceCacheBank,
)
from picf_next.encoders.anytouch2 import (
    ANYTOUCH2_GEOMETRY_WIDTH,
    ANYTOUCH2_TOKEN_WIDTH,
    ANYTOUCH2_TOKENS_PER_SENSOR,
)
from picf_next.encoders.spatiallm_sonata import (
    SPATIALLM_SONATA_NATIVE_GEOMETRY_WIDTH,
    SPATIALLM_SONATA_TOKEN_WIDTH,
    SpatialLMSonataConfig,
)
from picf_next.encoders.vjepa21 import (
    VJEPA21_CALVIN_GEOMETRY_WIDTH,
    VJEPA21_CALVIN_VIEW_NAMES,
    Vjepa21DenseConfig,
)

CALVIN_DENSE_EVIDENCE_AUDIT_SCHEMA = "picf-next.calvin-dense-evidence-semantic-audit/v1"
CALVIN_DENSE_EVIDENCE_MODALITIES = ("anytouch", "sonata", "vjepa")

_ARTIFACT_DOMAIN = "picf-next.calvin-dense-evidence-semantic-audit-artifact/v1"
_CONTRACT_DOMAIN = "picf-next.calvin-dense-evidence-semantic-audit-contracts/v1"
_RECORD_DOMAIN = "picf-next.calvin-dense-evidence-semantic-audit-records/v1"
_SHARD_DOMAIN = "picf-next.calvin-dense-evidence-semantic-audit-shards/v1"


@dataclass(frozen=True, slots=True)
class _ContractConstraint:
    token_width: int
    geometry_width: int
    maximum_tokens: int
    has_group_ids: bool
    token_dtype: str = "float16"

    def as_dict(self) -> dict[str, object]:
        return {
            "geometry_width": self.geometry_width,
            "has_group_ids": self.has_group_ids,
            "maximum_tokens": self.maximum_tokens,
            "token_dtype": self.token_dtype,
            "token_width": self.token_width,
        }


_SONATA_CONFIG = SpatialLMSonataConfig()
_VJEPA_CONFIG = Vjepa21DenseConfig()
_CONTRACT_CONSTRAINTS = {
    "anytouch": _ContractConstraint(
        token_width=ANYTOUCH2_TOKEN_WIDTH,
        geometry_width=ANYTOUCH2_GEOMETRY_WIDTH,
        maximum_tokens=2 * ANYTOUCH2_TOKENS_PER_SENSOR,
        has_group_ids=True,
    ),
    "sonata": _ContractConstraint(
        token_width=SPATIALLM_SONATA_TOKEN_WIDTH,
        geometry_width=SPATIALLM_SONATA_NATIVE_GEOMETRY_WIDTH,
        maximum_tokens=_SONATA_CONFIG.maximum_points,
        has_group_ids=False,
    ),
    "vjepa": _ContractConstraint(
        token_width=_VJEPA_CONFIG.token_width,
        geometry_width=VJEPA21_CALVIN_GEOMETRY_WIDTH,
        maximum_tokens=len(VJEPA21_CALVIN_VIEW_NAMES) * _VJEPA_CONFIG.token_count,
        has_group_ids=False,
    ),
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _validate_contract(contract: DenseEvidenceCacheContract) -> None:
    try:
        expected = _CONTRACT_CONSTRAINTS[contract.modality].as_dict()
    except KeyError as error:
        raise ContractError(
            f"unsupported CALVIN dense evidence modality {contract.modality!r}"
        ) from error
    observed = {
        "geometry_width": contract.geometry_width,
        "has_group_ids": contract.has_group_ids,
        "maximum_tokens": contract.maximum_tokens,
        "token_dtype": contract.token_dtype,
        "token_width": contract.token_width,
    }
    mismatches = [
        f"{name}={observed[name]!r} (expected {expected[name]!r})"
        for name in expected
        if observed[name] != expected[name]
    ]
    if mismatches:
        raise ContractError(
            f"CALVIN {contract.modality} dense cache contract differs from production ABI: "
            + ", ".join(mismatches)
        )


def _validate_record_semantics(
    *,
    modality: str,
    position: int,
    available: bool,
    token_count: int,
    maximum_tokens: int,
) -> None:
    label = f"CALVIN {modality} record[{position}]"
    if modality == "vjepa":
        if not available or token_count != maximum_tokens:
            raise ContractError(f"{label} must be available with exactly {maximum_tokens} tokens")
        return
    if modality == "sonata":
        if not available or not 1 <= token_count <= maximum_tokens:
            raise ContractError(f"{label} must be available with 1..{maximum_tokens} tokens")
        return
    if modality == "anytouch":
        if maximum_tokens % 2:
            raise ContractError("CALVIN AnyTouch maximum token count must split into two sensors")
        allowed = (0, maximum_tokens // 2, maximum_tokens)
        if token_count not in allowed:
            raise ContractError(f"{label} token count must be one of {allowed}, got {token_count}")
        if available != (token_count > 0):
            raise ContractError(f"{label} availability must equal token_count > 0")
        return
    raise ContractError(f"unsupported CALVIN dense evidence modality {modality!r}")


def _shard_metadata(cache: FrozenDenseEvidenceCache) -> list[dict[str, object]]:
    """Reconcile shard manifest aggregates without opening a shard file."""

    payload: list[dict[str, object]] = []
    offset = 0
    for shard_index, shard in enumerate(cache.shards):
        stop = offset + shard.row_count
        records = cache.records[offset:stop]
        expected_locations = tuple((shard_index, row) for row in range(shard.row_count))
        if (
            len(records) != shard.row_count
            or tuple((record.shard_index, record.row) for record in records) != expected_locations
        ):
            raise ContractError(
                f"CALVIN {cache.contract.modality} shard[{shard_index}] row metadata changed"
            )
        if sum(record.token_count for record in records) != shard.token_count:
            raise ContractError(
                f"CALVIN {cache.contract.modality} shard[{shard_index}] token total changed"
            )
        if (
            records[0].source_global_index != shard.first_source_global_index
            or records[-1].source_global_index != shard.last_source_global_index
        ):
            raise ContractError(
                f"CALVIN {cache.contract.modality} shard[{shard_index}] source bounds changed"
            )
        payload.append(
            {
                "first_source_global_index": shard.first_source_global_index,
                "last_source_global_index": shard.last_source_global_index,
                "path": shard.path,
                "row_count": shard.row_count,
                "sha256": shard.sha256,
                "token_count": shard.token_count,
            }
        )
        offset = stop
    if offset != len(cache.records):
        raise ContractError(f"CALVIN {cache.contract.modality} shard rows omit cache records")
    return payload


def _histogram_payload(histogram: Counter[int]) -> list[dict[str, int]]:
    return [
        {"record_count": record_count, "token_count": token_count}
        for token_count, record_count in sorted(histogram.items())
    ]


def audit_calvin_dense_evidence_cache_bank(
    bank: FrozenDenseEvidenceCacheBank,
) -> dict[str, object]:
    """Fail closed over all CALVIN full-modal cache manifest records.

    The function reads only the contracts, record tables and shard metadata
    already loaded from each manifest.  It does not inspect or materialize any
    dense token array.
    """

    if not isinstance(bank, FrozenDenseEvidenceCacheBank):
        raise TypeError("CALVIN dense evidence audit requires a frozen cache bank")
    if bank.modalities != CALVIN_DENSE_EVIDENCE_MODALITIES:
        raise ContractError(
            "CALVIN dense evidence audit requires exact modalities "
            f"{CALVIN_DENSE_EVIDENCE_MODALITIES}, got {bank.modalities}"
        )
    if bank.record_count <= 0:
        raise ContractError("CALVIN dense evidence audit requires manifest records")

    caches = {cache.contract.modality: cache for cache in bank.caches}
    for contract in bank.contracts:
        _validate_contract(contract)

    contracts = {
        modality: caches[modality].contract.payload()
        for modality in CALVIN_DENSE_EVIDENCE_MODALITIES
    }
    contract_sha256 = canonical_payload_sha256(_CONTRACT_DOMAIN, contracts)
    shard_metadata = {
        modality: _shard_metadata(caches[modality]) for modality in CALVIN_DENSE_EVIDENCE_MODALITIES
    }
    shard_metadata_sha256 = canonical_payload_sha256(_SHARD_DOMAIN, shard_metadata)

    histograms = {modality: Counter() for modality in CALVIN_DENSE_EVIDENCE_MODALITIES}
    available_counts = {modality: 0 for modality in CALVIN_DENSE_EVIDENCE_MODALITIES}
    records_digest = hashlib.sha256()
    records_digest.update(_RECORD_DOMAIN.encode("ascii"))
    records_digest.update(b"\0")

    for position in range(bank.record_count):
        locations = {
            modality: caches[modality].records[position]
            for modality in CALVIN_DENSE_EVIDENCE_MODALITIES
        }
        identities = {
            (record.source_global_index, record.sample_key) for record in locations.values()
        }
        if len(identities) != 1:
            raise ContractError(f"CALVIN dense cache records lost alignment at position {position}")
        source_global_index, sample_key = next(iter(identities))
        ordered_modalities: list[dict[str, object]] = []
        for modality in CALVIN_DENSE_EVIDENCE_MODALITIES:
            cache = caches[modality]
            record = locations[modality]
            _validate_record_semantics(
                modality=modality,
                position=position,
                available=record.available,
                token_count=record.token_count,
                maximum_tokens=cache.contract.maximum_tokens,
            )
            histograms[modality][record.token_count] += 1
            available_counts[modality] += int(record.available)
            ordered_modalities.append(
                {
                    "available": record.available,
                    "modality": modality,
                    "row": record.row,
                    "shard_index": record.shard_index,
                    "source_input_sha256": record.source_input_sha256,
                    "token_count": record.token_count,
                }
            )
        metadata = _canonical_bytes(
            {
                "modalities": ordered_modalities,
                "position": position,
                "sample_key": sample_key,
                "source_global_index": source_global_index,
            }
        )
        records_digest.update(len(metadata).to_bytes(8, byteorder="big", signed=False))
        records_digest.update(metadata)
    records_digest.update(bank.record_count.to_bytes(8, byteorder="big", signed=False))

    anytouch_available = available_counts["anytouch"]
    if not 0 < anytouch_available < bank.record_count:
        raise ContractError(
            "CALVIN AnyTouch full coverage must contain both available and unavailable records"
        )

    token_count_histogram = {
        modality: _histogram_payload(histograms[modality])
        for modality in CALVIN_DENSE_EVIDENCE_MODALITIES
    }
    availability_histogram = {
        modality: {
            "available": available_counts[modality],
            "unavailable": bank.record_count - available_counts[modality],
        }
        for modality in CALVIN_DENSE_EVIDENCE_MODALITIES
    }
    first_contract = bank.contracts[0]
    payload: dict[str, object] = {
        "availability_histogram": availability_histogram,
        "contract_sha256": contract_sha256,
        "contracts": contracts,
        "coverage_plan_sha256": first_contract.coverage_plan_sha256,
        "dataset_id": first_contract.dataset_id,
        "dataset_revision": first_contract.dataset_revision,
        "dataset_tree_sha256": first_contract.dataset_tree_sha256,
        "modalities": list(CALVIN_DENSE_EVIDENCE_MODALITIES),
        "ordered_record_metadata_sha256": records_digest.hexdigest(),
        "record_count": bank.record_count,
        "schema": CALVIN_DENSE_EVIDENCE_AUDIT_SCHEMA,
        "shard_metadata_sha256": shard_metadata_sha256,
        "status": "PASS",
        "token_count_histogram": token_count_histogram,
    }
    return {
        **payload,
        "artifact_sha256": canonical_payload_sha256(_ARTIFACT_DOMAIN, payload),
    }


def validate_calvin_dense_evidence_audit(
    report: object,
    bank: FrozenDenseEvidenceCacheBank,
) -> dict[str, object]:
    """Recompute and exactly match a published semantic audit report."""

    if not isinstance(report, Mapping) or any(not isinstance(key, str) for key in report):
        raise ContractError("CALVIN dense evidence semantic audit must be one JSON object")
    expected = audit_calvin_dense_evidence_cache_bank(bank)
    try:
        observed_bytes = _canonical_bytes(report)
        expected_bytes = _canonical_bytes(expected)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ContractError("CALVIN dense evidence semantic audit is not canonical JSON") from error
    if observed_bytes != expected_bytes:
        raise ContractError("CALVIN dense evidence semantic audit differs from exact replay")
    return json.loads(expected_bytes.decode("ascii"))
