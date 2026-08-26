from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from picf_next.content_addressing import canonical_payload_sha256
from picf_next.contracts import ContractError
from picf_next.data.calvin_dense_evidence_audit import (
    CALVIN_DENSE_EVIDENCE_AUDIT_SCHEMA,
    audit_calvin_dense_evidence_cache_bank,
    validate_calvin_dense_evidence_audit,
)
from picf_next.data.dense_evidence_cache import (
    DENSE_EVIDENCE_CACHE_SCHEMA,
    FrozenDenseEvidenceCacheBank,
)

_TREE_SHA256 = hashlib.sha256(b"calvin-tree").hexdigest()
_COVERAGE_SHA256 = hashlib.sha256(b"calvin-coverage").hexdigest()
_ARTIFACT_DOMAIN = "picf-next.calvin-dense-evidence-semantic-audit-artifact/v1"

_DEFAULT_COUNTS = {
    "anytouch": (0, 398, 796),
    "sonata": (127, 512, 4096),
    "vjepa": (1152, 1152, 1152),
}
_DEFAULT_AVAILABILITY = {
    "anytouch": (False, True, True),
    "sonata": (True, True, True),
    "vjepa": (True, True, True),
}
_CONTRACTS = {
    "anytouch": {
        "geometry_width": 18,
        "has_group_ids": True,
        "maximum_tokens": 796,
        "token_width": 768,
    },
    "sonata": {
        "geometry_width": 3,
        "has_group_ids": False,
        "maximum_tokens": 4096,
        "token_width": 512,
    },
    "vjepa": {
        "geometry_width": 4,
        "has_group_ids": False,
        "maximum_tokens": 1152,
        "token_width": 768,
    },
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_manifest(
    root: Path,
    *,
    modality: str,
    token_counts: Sequence[int],
    availability: Sequence[bool],
    contract_overrides: Mapping[str, object] | None = None,
    shard_token_delta: int = 0,
    source_hash_salt: str = "",
) -> str:
    assert len(token_counts) == len(availability) == 3
    contract = {
        "dataset_id": "calvin",
        "dataset_revision": "calvin-revision",
        "dataset_tree_sha256": _TREE_SHA256,
        "coverage_plan_sha256": _COVERAGE_SHA256,
        "encoder_contract": f"{modality}@pinned-revision/frozen/v1",
        "modality": modality,
        "token_dtype": "float16",
        **_CONTRACTS[modality],
    }
    if contract_overrides is not None:
        contract.update(contract_overrides)
    records = [
        {
            "available": available,
            "row": position,
            "sample_key": f"sample-{position + 10}",
            "shard_index": 0,
            "source_global_index": position + 10,
            "source_input_sha256": _digest(f"{source_hash_salt}{modality}-input-{position + 10}"),
            "token_count": token_count,
        }
        for position, (token_count, available) in enumerate(
            zip(token_counts, availability, strict=True)
        )
    ]
    manifest = {
        "complete": True,
        "contract": contract,
        "records": records,
        "records_sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
        "schema": DENSE_EVIDENCE_CACHE_SCHEMA,
        "shards": [
            {
                "first_source_global_index": 10,
                "last_source_global_index": 12,
                "path": "deliberately-absent-shard.npz",
                "row_count": 3,
                "sha256": _digest(f"{modality}-absent-shard"),
                "token_count": sum(token_counts) + shard_token_delta,
            }
        ],
    }
    encoded = _canonical_bytes(manifest)
    root.mkdir(parents=True)
    (root / "manifest.json").write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _load_bank(
    tmp_path: Path,
    *,
    token_counts: Mapping[str, Sequence[int]] | None = None,
    availability: Mapping[str, Sequence[bool]] | None = None,
    contract_overrides: Mapping[str, Mapping[str, object]] | None = None,
    shard_token_delta: Mapping[str, int] | None = None,
    source_hash_salt: str = "",
) -> FrozenDenseEvidenceCacheBank:
    counts = {**_DEFAULT_COUNTS, **({} if token_counts is None else token_counts)}
    active = {
        **_DEFAULT_AVAILABILITY,
        **({} if availability is None else availability),
    }
    overrides = {} if contract_overrides is None else contract_overrides
    deltas = {} if shard_token_delta is None else shard_token_delta
    roots: list[Path] = []
    manifest_sha256s: list[str] = []
    for modality in ("vjepa", "anytouch", "sonata"):
        root = tmp_path / modality
        roots.append(root)
        manifest_sha256s.append(
            _write_manifest(
                root,
                modality=modality,
                token_counts=counts[modality],
                availability=active[modality],
                contract_overrides=overrides.get(modality),
                shard_token_delta=deltas.get(modality, 0),
                source_hash_salt=source_hash_salt,
            )
        )
    return FrozenDenseEvidenceCacheBank.load(
        roots,
        manifest_sha256s=manifest_sha256s,
        dataset_tree_sha256=_TREE_SHA256,
    )


def test_calvin_dense_evidence_audit_accepts_exact_full_modal_semantics_without_tokens(
    tmp_path: Path,
) -> None:
    bank = _load_bank(tmp_path)
    assert all(not (cache.root / cache.shards[0].path).exists() for cache in bank.caches)
    assert all(not cache._loaded for cache in bank.caches)

    report = audit_calvin_dense_evidence_cache_bank(bank)

    assert report["schema"] == CALVIN_DENSE_EVIDENCE_AUDIT_SCHEMA
    assert report["status"] == "PASS"
    assert report["record_count"] == 3
    assert report["modalities"] == ["anytouch", "sonata", "vjepa"]
    assert report["availability_histogram"] == {
        "anytouch": {"available": 2, "unavailable": 1},
        "sonata": {"available": 3, "unavailable": 0},
        "vjepa": {"available": 3, "unavailable": 0},
    }
    assert report["token_count_histogram"] == {
        "anytouch": [
            {"record_count": 1, "token_count": 0},
            {"record_count": 1, "token_count": 398},
            {"record_count": 1, "token_count": 796},
        ],
        "sonata": [
            {"record_count": 1, "token_count": 127},
            {"record_count": 1, "token_count": 512},
            {"record_count": 1, "token_count": 4096},
        ],
        "vjepa": [{"record_count": 3, "token_count": 1152}],
    }
    contracts = report["contracts"]
    assert isinstance(contracts, dict)
    assert contracts["anytouch"]["has_group_ids"] is True
    assert contracts["sonata"]["geometry_width"] == 3
    assert contracts["vjepa"]["maximum_tokens"] == 1152
    assert all(not cache._loaded for cache in bank.caches)
    assert report == audit_calvin_dense_evidence_cache_bank(bank)
    assert validate_calvin_dense_evidence_audit(report, bank) == report

    payload = {key: value for key, value in report.items() if key != "artifact_sha256"}
    assert report["artifact_sha256"] == canonical_payload_sha256(_ARTIFACT_DOMAIN, payload)
    for name in (
        "artifact_sha256",
        "contract_sha256",
        "ordered_record_metadata_sha256",
        "shard_metadata_sha256",
    ):
        assert len(report[name]) == 64

    tampered = copy.deepcopy(report)
    artifact_sha256 = tampered.pop("artifact_sha256")
    tampered["record_count"] = 4
    assert artifact_sha256 != canonical_payload_sha256(_ARTIFACT_DOMAIN, tampered)
    with pytest.raises(ContractError, match="differs from exact replay"):
        validate_calvin_dense_evidence_audit(tampered, bank)


def test_calvin_dense_evidence_audit_rejects_all_empty_records(tmp_path: Path) -> None:
    bank = _load_bank(
        tmp_path,
        token_counts={modality: (0, 0, 0) for modality in _DEFAULT_COUNTS},
        availability={modality: (False, False, False) for modality in _DEFAULT_COUNTS},
    )

    with pytest.raises(ContractError, match="Sonata|sonata|V-JEPA|vjepa"):
        audit_calvin_dense_evidence_cache_bank(bank)


def test_calvin_dense_evidence_audit_rejects_available_zero_anytouch(tmp_path: Path) -> None:
    bank = _load_bank(
        tmp_path,
        availability={"anytouch": (True, True, True)},
    )

    with pytest.raises(ContractError, match="anytouch.*availability"):
        audit_calvin_dense_evidence_cache_bank(bank)


@pytest.mark.parametrize(
    ("modality", "token_counts", "match"),
    (
        ("anytouch", (0, 399, 796), "anytouch.*token count"),
        ("sonata", (127, 0, 4096), "sonata.*1..4096"),
        ("vjepa", (1152, 1151, 1152), "vjepa.*exactly 1152"),
    ),
)
def test_calvin_dense_evidence_audit_rejects_abnormal_token_counts(
    tmp_path: Path,
    modality: str,
    token_counts: tuple[int, int, int],
    match: str,
) -> None:
    bank = _load_bank(tmp_path, token_counts={modality: token_counts})

    with pytest.raises(ContractError, match=match):
        audit_calvin_dense_evidence_cache_bank(bank)


def test_calvin_dense_evidence_audit_requires_both_anytouch_coverage_states(
    tmp_path: Path,
) -> None:
    bank = _load_bank(
        tmp_path,
        token_counts={"anytouch": (398, 398, 796)},
        availability={"anytouch": (True, True, True)},
    )

    with pytest.raises(ContractError, match="both available and unavailable"):
        audit_calvin_dense_evidence_cache_bank(bank)


@pytest.mark.parametrize(
    ("modality", "overrides", "match"),
    (
        ("anytouch", {"has_group_ids": False}, "anytouch.*has_group_ids"),
        ("sonata", {"geometry_width": 4}, "sonata.*geometry_width"),
        ("vjepa", {"token_width": 3072}, "vjepa.*token_width"),
    ),
)
def test_calvin_dense_evidence_audit_rejects_contract_abi_drift(
    tmp_path: Path,
    modality: str,
    overrides: dict[str, object],
    match: str,
) -> None:
    bank = _load_bank(tmp_path, contract_overrides={modality: overrides})

    with pytest.raises(ContractError, match=match):
        audit_calvin_dense_evidence_cache_bank(bank)


def test_calvin_dense_evidence_audit_rejects_shard_aggregate_drift(tmp_path: Path) -> None:
    bank = _load_bank(tmp_path, shard_token_delta={"sonata": 1})

    with pytest.raises(ContractError, match="sonata.*token total"):
        audit_calvin_dense_evidence_cache_bank(bank)


def test_calvin_dense_evidence_audit_digest_binds_ordered_source_metadata(
    tmp_path: Path,
) -> None:
    first = audit_calvin_dense_evidence_cache_bank(_load_bank(tmp_path / "first"))
    second = audit_calvin_dense_evidence_cache_bank(
        _load_bank(tmp_path / "second", source_hash_salt="changed-")
    )

    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["ordered_record_metadata_sha256"] != second["ordered_record_metadata_sha256"]
    assert first["artifact_sha256"] != second["artifact_sha256"]
