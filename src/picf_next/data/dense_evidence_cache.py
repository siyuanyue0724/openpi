"""Immutable sharded cache for frozen optional-modality encoder outputs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath

DENSE_EVIDENCE_CACHE_SCHEMA = "picf-next.frozen-dense-evidence-cache/v2"
DENSE_EVIDENCE_CACHE_PARTIAL_SCHEMA = "picf-next.frozen-dense-evidence-cache-partial/v2"
DENSE_EVIDENCE_CACHE_PARTITION_INDEX_SCHEMA = (
    "picf-next.frozen-dense-evidence-cache-partition-index/v1"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be nonempty text")
    return value


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ContractError(f"{name} must be {qualifier}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name)
    if set(result) != fields:
        raise ContractError(f"{name} fields differ from the frozen schema")
    return result


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or "\0" in text
        or path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractError(f"{name} must be a normalized relative POSIX path")
    return text


@dataclass(frozen=True, slots=True)
class DenseEvidenceCacheContract:
    dataset_id: str
    dataset_revision: str
    dataset_tree_sha256: str
    coverage_plan_sha256: str
    modality: str
    encoder_contract: str
    token_width: int
    geometry_width: int
    maximum_tokens: int
    has_group_ids: bool = False
    token_dtype: str = "float16"

    def __post_init__(self) -> None:
        for value, name in (
            (self.dataset_id, "dataset id"),
            (self.dataset_revision, "dataset revision"),
            (self.modality, "modality"),
            (self.encoder_contract, "encoder contract"),
        ):
            _text(value, name)
        _sha256_text(self.dataset_tree_sha256, "dataset tree sha256")
        _sha256_text(self.coverage_plan_sha256, "coverage plan sha256")
        _integer(self.token_width, "token width", positive=True)
        _integer(self.geometry_width, "geometry width")
        _integer(self.maximum_tokens, "maximum tokens", positive=True)
        if not isinstance(self.has_group_ids, bool):
            raise ContractError("has_group_ids must be boolean")
        if self.token_dtype not in {"float16", "float32"}:
            raise ContractError("dense evidence cache tokens must be float16 or float32")

    @property
    def numpy_token_dtype(self) -> np.dtype:
        return np.dtype(self.token_dtype)

    def payload(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "dataset_tree_sha256": self.dataset_tree_sha256,
            "coverage_plan_sha256": self.coverage_plan_sha256,
            "encoder_contract": self.encoder_contract,
            "geometry_width": self.geometry_width,
            "has_group_ids": self.has_group_ids,
            "maximum_tokens": self.maximum_tokens,
            "modality": self.modality,
            "token_dtype": self.token_dtype,
            "token_width": self.token_width,
        }


@dataclass(frozen=True, slots=True)
class DenseEvidenceCacheRecord:
    source_global_index: int
    sample_key: str
    source_input_sha256: str
    evidence: DenseEvidence

    def __post_init__(self) -> None:
        _integer(self.source_global_index, "source global index")
        _text(self.sample_key, "sample key")
        _sha256_text(self.source_input_sha256, "source input sha256")
        if not isinstance(self.evidence, DenseEvidence):
            raise TypeError("dense evidence cache record requires DenseEvidence")


@dataclass(frozen=True, slots=True)
class _RecordLocation:
    source_global_index: int
    sample_key: str
    source_input_sha256: str
    shard_index: int
    row: int
    token_count: int
    available: bool


@dataclass(frozen=True, slots=True)
class _Shard:
    path: str
    sha256: str
    row_count: int
    token_count: int
    first_source_global_index: int
    last_source_global_index: int


@dataclass(frozen=True, slots=True)
class _LoadedShard:
    source_global_indices: NDArray[np.int64]
    sample_keys: NDArray[np.str_]
    source_input_sha256: NDArray[np.str_]
    available: NDArray[np.bool_]
    offsets: NDArray[np.int64]
    tokens: NDArray[np.floating]
    geometry: NDArray[np.float32]
    timestamps: NDArray[np.float64]
    confidence: NDArray[np.float32]
    current_measurement_valid: NDArray[np.bool_]
    group_ids: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DenseEvidenceCacheResumeState:
    """Authenticated prefix retained by an interrupted cache publication."""

    completed_record_count: int
    last_source_global_index: int | None

    def __post_init__(self) -> None:
        _integer(self.completed_record_count, "completed record count")
        if self.last_source_global_index is not None:
            _integer(self.last_source_global_index, "last source global index")
        if (self.completed_record_count == 0) != (self.last_source_global_index is None):
            raise ContractError("dense evidence resume state has inconsistent source bounds")


class FrozenDenseEvidenceCache:
    """Lazy cache reader bound to one dataset tree and encoder contract."""

    def __init__(
        self,
        *,
        root: Path,
        contract: DenseEvidenceCacheContract,
        records: tuple[_RecordLocation, ...],
        shards: tuple[_Shard, ...],
        memory_capacity: int,
    ) -> None:
        self.root = root
        self.contract = contract
        self.records = records
        self.shards = shards
        self.memory_capacity = memory_capacity
        self._locations = {record.source_global_index: record for record in records}
        self._maximum_sample_key_characters = max(len(record.sample_key) for record in records)
        self._loaded: OrderedDict[int, _LoadedShard] = OrderedDict()

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        manifest_sha256: str,
        dataset_tree_sha256: str,
        memory_capacity: int = 2,
    ) -> FrozenDenseEvidenceCache:
        expected_manifest = _sha256_text(manifest_sha256, "cache manifest sha256")
        expected_tree = _sha256_text(dataset_tree_sha256, "dataset tree sha256")
        memory_capacity = _integer(memory_capacity, "cache memory capacity", positive=True)
        resolved = Path(root).resolve()
        raw_manifest = read_sha256_verified_file_beneath(
            resolved,
            "manifest.json",
            expected_sha256=expected_manifest,
            maximum_bytes=128 * 1024 * 1024,
        )
        try:
            decoded = json.loads(raw_manifest)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("dense evidence cache manifest is not valid JSON") from error
        payload = _exact(
            decoded,
            "dense evidence cache manifest",
            {"complete", "contract", "records", "records_sha256", "schema", "shards"},
        )
        if payload["schema"] != DENSE_EVIDENCE_CACHE_SCHEMA or payload["complete"] is not True:
            raise ContractError("dense evidence cache is incomplete or uses another schema")
        contract_payload = _exact(
            payload["contract"],
            "dense evidence cache contract",
            {
                "dataset_id",
                "dataset_revision",
                "dataset_tree_sha256",
                "coverage_plan_sha256",
                "encoder_contract",
                "geometry_width",
                "has_group_ids",
                "maximum_tokens",
                "modality",
                "token_dtype",
                "token_width",
            },
        )
        if not isinstance(contract_payload["has_group_ids"], bool):
            raise ContractError("has_group_ids must be boolean")
        contract = DenseEvidenceCacheContract(
            dataset_id=_text(contract_payload["dataset_id"], "dataset id"),
            dataset_revision=_text(contract_payload["dataset_revision"], "dataset revision"),
            dataset_tree_sha256=_sha256_text(
                contract_payload["dataset_tree_sha256"], "dataset tree sha256"
            ),
            coverage_plan_sha256=_sha256_text(
                contract_payload["coverage_plan_sha256"], "coverage plan sha256"
            ),
            modality=_text(contract_payload["modality"], "modality"),
            encoder_contract=_text(contract_payload["encoder_contract"], "encoder contract"),
            token_width=_integer(contract_payload["token_width"], "token width", positive=True),
            geometry_width=_integer(contract_payload["geometry_width"], "geometry width"),
            maximum_tokens=_integer(
                contract_payload["maximum_tokens"], "maximum tokens", positive=True
            ),
            has_group_ids=contract_payload["has_group_ids"],
            token_dtype=_text(contract_payload["token_dtype"], "token dtype"),
        )
        if contract.dataset_tree_sha256 != expected_tree:
            raise ContractError("dense evidence cache belongs to another dataset tree")
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or not raw_records:
            raise ContractError("dense evidence cache requires records")
        if _sha256_bytes(_canonical_bytes(raw_records)) != _sha256_text(
            payload["records_sha256"], "records sha256"
        ):
            raise ContractError("dense evidence cache record table hash changed")
        records = tuple(
            cls._parse_record(value, index=index) for index, value in enumerate(raw_records)
        )
        indices = tuple(record.source_global_index for record in records)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ContractError("dense evidence source indices must be sorted and unique")
        if len({record.sample_key for record in records}) != len(records):
            raise ContractError("dense evidence sample keys must be unique")

        raw_shards = payload["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ContractError("dense evidence cache requires shards")
        shards = tuple(
            cls._parse_shard(value, index=index) for index, value in enumerate(raw_shards)
        )
        for record in records:
            if (
                record.shard_index >= len(shards)
                or record.row >= shards[record.shard_index].row_count
            ):
                raise ContractError("dense evidence record points outside its shard")
            if record.token_count > contract.maximum_tokens:
                raise ContractError("dense evidence record exceeds its declared token budget")
            if not record.available and record.token_count:
                raise ContractError("unavailable dense evidence record has tokens")
        if sum(shard.row_count for shard in shards) != len(records):
            raise ContractError("dense evidence shard rows differ from the record table")
        expected_locations = tuple(
            (shard_index, row)
            for shard_index, shard in enumerate(shards)
            for row in range(shard.row_count)
        )
        if tuple((record.shard_index, record.row) for record in records) != expected_locations:
            raise ContractError("dense evidence records do not cover shard rows exactly once")
        return cls(
            root=resolved,
            contract=contract,
            records=records,
            shards=shards,
            memory_capacity=memory_capacity,
        )

    @staticmethod
    def _parse_record(raw: object, *, index: int) -> _RecordLocation:
        item = _exact(
            raw,
            f"dense evidence record[{index}]",
            {
                "available",
                "row",
                "sample_key",
                "shard_index",
                "source_global_index",
                "source_input_sha256",
                "token_count",
            },
        )
        if not isinstance(item["available"], bool):
            raise ContractError("dense evidence record availability must be boolean")
        return _RecordLocation(
            source_global_index=_integer(item["source_global_index"], "source global index"),
            sample_key=_text(item["sample_key"], "sample key"),
            source_input_sha256=_sha256_text(item["source_input_sha256"], "source input sha256"),
            shard_index=_integer(item["shard_index"], "shard index"),
            row=_integer(item["row"], "shard row"),
            token_count=_integer(item["token_count"], "token count"),
            available=item["available"],
        )

    @staticmethod
    def _parse_shard(raw: object, *, index: int) -> _Shard:
        item = _exact(
            raw,
            f"dense evidence shard[{index}]",
            {
                "first_source_global_index",
                "last_source_global_index",
                "path",
                "row_count",
                "sha256",
                "token_count",
            },
        )
        return _Shard(
            path=_relative_path(item["path"], "shard path"),
            sha256=_sha256_text(item["sha256"], "shard sha256"),
            row_count=_integer(item["row_count"], "shard rows", positive=True),
            token_count=_integer(item["token_count"], "shard tokens"),
            first_source_global_index=_integer(
                item["first_source_global_index"], "first source global index"
            ),
            last_source_global_index=_integer(
                item["last_source_global_index"], "last source global index"
            ),
        )

    def _load_shard(self, shard_index: int) -> _LoadedShard:
        cached = self._loaded.get(shard_index)
        if cached is not None:
            self._loaded.move_to_end(shard_index)
            return cached
        metadata = self.shards[shard_index]
        # ``np.savez`` stores every array without compression. Bound the complete
        # schema payload, including geometry and validity metadata, rather than
        # approximating all non-token arrays with one fixed per-token allowance.
        token_bytes = metadata.token_count * (
            self.contract.token_width * self.contract.numpy_token_dtype.itemsize
            + self.contract.geometry_width * np.dtype(np.float32).itemsize
            + np.dtype(np.float64).itemsize  # timestamps
            + np.dtype(np.float32).itemsize  # confidence
            + np.dtype(np.bool_).itemsize  # current_measurement_valid
            + np.dtype(np.int64).itemsize  # group_ids, also stored for ungrouped caches
        )
        row_bytes = metadata.row_count * (
            np.dtype(np.int64).itemsize  # source_global_indices
            + self._maximum_sample_key_characters * np.dtype("<U1").itemsize
            + 64 * np.dtype("<U1").itemsize  # source_input_sha256
            + np.dtype(np.bool_).itemsize  # available
            + np.dtype(np.int64).itemsize  # offsets
        )
        # Covers the final offset plus NPY headers, ZIP members and central directory.
        maximum_bytes = token_bytes + row_bytes + 4 * 1024 * 1024
        payload = read_sha256_verified_file_beneath(
            self.root,
            metadata.path,
            expected_sha256=metadata.sha256,
            maximum_bytes=maximum_bytes,
        )
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                expected = {
                    "available",
                    "confidence",
                    "current_measurement_valid",
                    "geometry",
                    "group_ids",
                    "offsets",
                    "sample_keys",
                    "source_global_indices",
                    "source_input_sha256",
                    "timestamps",
                    "tokens",
                }
                if set(archive.files) != expected:
                    raise ContractError("dense evidence shard arrays differ from schema")
                arrays = {name: archive[name].copy() for name in expected}
        except (OSError, ValueError) as error:
            raise ContractError("dense evidence shard is not a safe NPZ archive") from error
        rows = metadata.row_count
        tokens = metadata.token_count
        if (
            arrays["source_global_indices"].dtype != np.int64
            or arrays["source_global_indices"].shape != (rows,)
            or arrays["sample_keys"].dtype.kind != "U"
            or arrays["sample_keys"].shape != (rows,)
            or arrays["source_input_sha256"].dtype.kind != "U"
            or arrays["source_input_sha256"].shape != (rows,)
            or arrays["available"].dtype != np.bool_
            or arrays["available"].shape != (rows,)
            or arrays["offsets"].dtype != np.int64
            or arrays["offsets"].shape != (rows + 1,)
            or arrays["tokens"].dtype != self.contract.numpy_token_dtype
            or arrays["tokens"].shape != (tokens, self.contract.token_width)
            or arrays["geometry"].dtype != np.float32
            or arrays["geometry"].shape != (tokens, self.contract.geometry_width)
            or arrays["timestamps"].dtype != np.float64
            or arrays["timestamps"].shape != (tokens,)
            or arrays["confidence"].dtype != np.float32
            or arrays["confidence"].shape != (tokens,)
            or arrays["current_measurement_valid"].dtype != np.bool_
            or arrays["current_measurement_valid"].shape != (tokens,)
            or arrays["group_ids"].dtype != np.int64
            or arrays["group_ids"].shape != (tokens,)
        ):
            raise ContractError("dense evidence shard shapes or dtypes changed")
        offsets = arrays["offsets"]
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != tokens
            or (np.diff(offsets) < 0).any()
            or not np.isfinite(arrays["tokens"]).all()
            or not np.isfinite(arrays["geometry"]).all()
            or not np.isfinite(arrays["timestamps"]).all()
            or not np.isfinite(arrays["confidence"]).all()
            or ((arrays["confidence"] < 0.0) | (arrays["confidence"] > 1.0)).any()
        ):
            raise ContractError("dense evidence shard values violate their contract")
        if tuple(int(value) for value in arrays["source_global_indices"][[0, -1]]) != (
            metadata.first_source_global_index,
            metadata.last_source_global_index,
        ):
            raise ContractError("dense evidence shard source bounds changed")
        loaded = _LoadedShard(**arrays)
        self._loaded[shard_index] = loaded
        self._loaded.move_to_end(shard_index)
        while len(self._loaded) > self.memory_capacity:
            self._loaded.popitem(last=False)
        return loaded

    def evidence_for(
        self,
        *,
        source_global_index: int,
        sample_key: str,
        source_input_sha256: str | None = None,
    ) -> DenseEvidence:
        source_global_index = _integer(source_global_index, "runtime source global index")
        _text(sample_key, "runtime sample key")
        try:
            location = self._locations[source_global_index]
        except KeyError as error:
            raise KeyError(f"unknown dense evidence source index {source_global_index}") from error
        if location.sample_key != sample_key:
            raise ContractError("dense evidence sample key differs from its source index")
        if source_input_sha256 is not None and (
            _sha256_text(source_input_sha256, "runtime source input sha256")
            != location.source_input_sha256
        ):
            raise ContractError("runtime source input differs from the frozen encoder input")
        shard = self._load_shard(location.shard_index)
        row = location.row
        if (
            int(shard.source_global_indices[row]) != source_global_index
            or str(shard.sample_keys[row]) != sample_key
            or str(shard.source_input_sha256[row]) != location.source_input_sha256
            or bool(shard.available[row]) != location.available
        ):
            raise ContractError("dense evidence locator differs from its shard row")
        start = int(shard.offsets[row])
        stop = int(shard.offsets[row + 1])
        if stop - start != location.token_count:
            raise ContractError("dense evidence token count differs from its record table")
        values: list[NDArray] = [
            shard.tokens[start:stop],
            shard.geometry[start:stop],
            shard.timestamps[start:stop],
            shard.confidence[start:stop],
            shard.current_measurement_valid[start:stop],
            shard.group_ids[start:stop],
        ]
        for value in values:
            value.setflags(write=False)
        group_ids = values[5] if self.contract.has_group_ids else None
        if not self.contract.has_group_ids and (values[5] != -1).any():
            raise ContractError("ungrouped dense evidence shard contains group identifiers")
        return DenseEvidence(
            modality=self.contract.modality,
            encoder_contract=self.contract.encoder_contract,
            tokens=values[0],
            available=location.available,
            timestamps=values[2],
            confidence=values[3],
            geometry=values[1] if self.contract.geometry_width else None,
            group_ids=group_ids,
            current_measurement_valid=values[4],
        )


class FrozenDenseEvidenceCacheView(FrozenDenseEvidenceCache):
    """Exact record view over authenticated immutable cache sources.

    A distributed-topology change can require a different ordered subset of a
    much larger frozen cache. Rewriting the selected token payload is both
    unnecessary and expensive. This view authenticates every source cache,
    requires an exact target identity sequence, and delegates each lookup to
    the first source containing that identity. Conflicting overlapping
    metadata fail closed.
    """

    def __init__(
        self,
        *,
        sources: tuple[FrozenDenseEvidenceCache, ...],
        record_identities: tuple[tuple[int, str], ...],
        coverage_plan_sha256: str,
    ) -> None:
        if not sources or any(
            not isinstance(source, FrozenDenseEvidenceCache) for source in sources
        ):
            raise TypeError("dense evidence cache view requires authenticated cache sources")
        target_coverage = _sha256_text(coverage_plan_sha256, "view coverage plan sha256")
        first_contract = sources[0].contract
        technical_contract = {
            key: value
            for key, value in first_contract.payload().items()
            if key != "coverage_plan_sha256"
        }
        for source in sources[1:]:
            observed = {
                key: value
                for key, value in source.contract.payload().items()
                if key != "coverage_plan_sha256"
            }
            if observed != technical_contract:
                raise ContractError("dense evidence view sources use different encoder contracts")

        if not record_identities:
            raise ContractError("dense evidence cache view requires target records")
        indices = tuple(identity[0] for identity in record_identities)
        sample_keys = tuple(identity[1] for identity in record_identities)
        if (
            any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or any(not isinstance(key, str) or not key for key in sample_keys)
            or indices != tuple(sorted(indices))
            or len(set(indices)) != len(indices)
            or len(set(sample_keys)) != len(sample_keys)
        ):
            raise ContractError("dense evidence view target identities must be sorted and unique")

        source_locations = tuple(source._locations for source in sources)
        selected: list[_RecordLocation] = []
        routes: dict[int, FrozenDenseEvidenceCache] = {}
        source_record_counts = [0 for _ in sources]
        for source_global_index, sample_key in record_identities:
            candidates: list[tuple[int, FrozenDenseEvidenceCache, _RecordLocation]] = []
            for source_index, (source, locations) in enumerate(
                zip(sources, source_locations, strict=True)
            ):
                location = locations.get(source_global_index)
                if location is None:
                    continue
                if location.sample_key != sample_key:
                    raise ContractError(
                        "dense evidence view source index maps to a conflicting sample key"
                    )
                candidates.append((source_index, source, location))
            if not candidates:
                raise ContractError("dense evidence view sources do not cover every target record")
            reference = candidates[0][2]
            reference_metadata = (
                reference.source_input_sha256,
                reference.token_count,
                reference.available,
            )
            if any(
                (
                    candidate.source_input_sha256,
                    candidate.token_count,
                    candidate.available,
                )
                != reference_metadata
                for _, _, candidate in candidates[1:]
            ):
                raise ContractError("dense evidence view sources disagree on overlapping metadata")
            selected_source_index, selected_source, selected_location = candidates[0]
            selected.append(selected_location)
            routes[source_global_index] = selected_source
            source_record_counts[selected_source_index] += 1

        self.root = Path("/")
        self.contract = replace(first_contract, coverage_plan_sha256=target_coverage)
        self.records = tuple(selected)
        self.shards = ()
        self.memory_capacity = sum(source.memory_capacity for source in sources)
        self._locations = {record.source_global_index: record for record in self.records}
        self._maximum_sample_key_characters = max(len(record.sample_key) for record in self.records)
        self._loaded = OrderedDict()
        self._routes = routes
        self.source_record_counts = tuple(source_record_counts)
        self.source_coverage_plan_sha256s = tuple(
            source.contract.coverage_plan_sha256 for source in sources
        )

    def evidence_for(
        self,
        *,
        source_global_index: int,
        sample_key: str,
        source_input_sha256: str | None = None,
    ) -> DenseEvidence:
        source_global_index = _integer(source_global_index, "runtime source global index")
        try:
            source = self._routes[source_global_index]
        except KeyError as error:
            raise KeyError(
                f"unknown dense evidence view source index {source_global_index}"
            ) from error
        return source.evidence_for(
            source_global_index=source_global_index,
            sample_key=sample_key,
            source_input_sha256=source_input_sha256,
        )


def _aggregate_partition_metadata(
    caches: Sequence[FrozenDenseEvidenceCache],
) -> tuple[tuple[_RecordLocation, ...], tuple[_Shard, ...]]:
    records: list[_RecordLocation] = []
    shards: list[_Shard] = []
    for partition_index, cache in enumerate(caches):
        shard_offset = len(shards)
        shards.extend(
            _Shard(
                path=f"partition-{partition_index:03d}/{shard.path}",
                sha256=shard.sha256,
                row_count=shard.row_count,
                token_count=shard.token_count,
                first_source_global_index=shard.first_source_global_index,
                last_source_global_index=shard.last_source_global_index,
            )
            for shard in cache.shards
        )
        records.extend(
            _RecordLocation(
                source_global_index=record.source_global_index,
                sample_key=record.sample_key,
                source_input_sha256=record.source_input_sha256,
                shard_index=shard_offset + record.shard_index,
                row=record.row,
                token_count=record.token_count,
                available=record.available,
            )
            for record in cache.records
        )
    return tuple(records), tuple(shards)


def _record_location_payload(record: _RecordLocation) -> dict[str, object]:
    return {
        "available": record.available,
        "row": record.row,
        "sample_key": record.sample_key,
        "shard_index": record.shard_index,
        "source_global_index": record.source_global_index,
        "source_input_sha256": record.source_input_sha256,
        "token_count": record.token_count,
    }


class FrozenDenseEvidencePartitionedCache(FrozenDenseEvidenceCache):
    """One logical cache backed by independently authenticated partitions."""

    def __init__(
        self,
        *,
        root: Path,
        partitions: tuple[FrozenDenseEvidenceCache, ...],
        records: tuple[_RecordLocation, ...],
        shards: tuple[_Shard, ...],
        memory_capacity: int,
    ) -> None:
        if not partitions:
            raise ContractError("partitioned dense evidence cache requires partitions")
        super().__init__(
            root=root,
            contract=partitions[0].contract,
            records=records,
            shards=shards,
            memory_capacity=memory_capacity,
        )
        self.partitions = partitions
        self._partition_by_source_index = {
            record.source_global_index: partition
            for partition in partitions
            for record in partition.records
        }
        if len(self._partition_by_source_index) != len(records):
            raise ContractError("partitioned dense evidence cache has duplicate source indices")

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        manifest_sha256: str,
        dataset_tree_sha256: str,
        memory_capacity: int = 2,
    ) -> FrozenDenseEvidencePartitionedCache:
        expected_manifest = _sha256_text(manifest_sha256, "cache manifest sha256")
        expected_tree = _sha256_text(dataset_tree_sha256, "dataset tree sha256")
        memory_capacity = _integer(memory_capacity, "cache memory capacity", positive=True)
        resolved = Path(root).resolve()
        raw_manifest = read_sha256_verified_file_beneath(
            resolved,
            "manifest.json",
            expected_sha256=expected_manifest,
            maximum_bytes=16 * 1024 * 1024,
        )
        try:
            decoded = json.loads(raw_manifest)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("partitioned dense evidence manifest is not valid JSON") from error
        payload = _exact(
            decoded,
            "partitioned dense evidence manifest",
            {"complete", "contract", "partitions", "records_sha256", "schema"},
        )
        if (
            payload["schema"] != DENSE_EVIDENCE_CACHE_PARTITION_INDEX_SCHEMA
            or payload["complete"] is not True
        ):
            raise ContractError("partitioned dense evidence cache is incomplete")
        raw_partitions = payload["partitions"]
        if not isinstance(raw_partitions, list) or not raw_partitions:
            raise ContractError("partitioned dense evidence cache requires partition entries")
        partition_roots: list[Path] = []
        partition_manifest_sha256s: list[str] = []
        declared_entries: list[Mapping[str, object]] = []
        namespace = resolved.parent.resolve()
        for index, value in enumerate(raw_partitions):
            entry = _exact(
                value,
                f"dense evidence partition[{index}]",
                {
                    "first_source_global_index",
                    "last_source_global_index",
                    "manifest_sha256",
                    "record_count",
                    "root",
                },
            )
            relative_root = _relative_path(entry["root"], "dense evidence partition root")
            partition_root = namespace.joinpath(*PurePosixPath(relative_root).parts)
            if partition_root.is_symlink() or not partition_root.is_dir():
                raise ContractError("dense evidence partition root is absent or indirect")
            partition_roots.append(partition_root)
            partition_manifest_sha256s.append(
                _sha256_text(entry["manifest_sha256"], "partition manifest sha256")
            )
            declared_entries.append(entry)
        if len(set(partition_roots)) != len(partition_roots):
            raise ContractError("partitioned dense evidence cache repeats a partition root")
        partitions = tuple(
            FrozenDenseEvidenceCache.load(
                partition_root,
                manifest_sha256=partition_manifest_sha256,
                dataset_tree_sha256=expected_tree,
                memory_capacity=memory_capacity,
            )
            for partition_root, partition_manifest_sha256 in zip(
                partition_roots,
                partition_manifest_sha256s,
                strict=True,
            )
        )
        contract = partitions[0].contract
        if any(partition.contract != contract for partition in partitions[1:]):
            raise ContractError("dense evidence partitions use different contracts")
        if payload["contract"] != contract.payload():
            raise ContractError("partition index contract differs from its partitions")
        ordered = tuple(
            sorted(partitions, key=lambda partition: partition.records[0].source_global_index)
        )
        if ordered != partitions:
            raise ContractError("dense evidence partition index is not source ordered")
        for index, (entry, partition) in enumerate(zip(declared_entries, partitions, strict=True)):
            expected_entry = {
                "first_source_global_index": partition.records[0].source_global_index,
                "last_source_global_index": partition.records[-1].source_global_index,
                "manifest_sha256": partition_manifest_sha256s[index],
                "record_count": len(partition.records),
                "root": str(partition.root.relative_to(namespace)),
            }
            if dict(entry) != expected_entry:
                raise ContractError("dense evidence partition entry differs from its manifest")
        records, shards = _aggregate_partition_metadata(partitions)
        indices = tuple(record.source_global_index for record in records)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ContractError(
                "partitioned dense evidence source indices are not unique and sorted"
            )
        records_sha256 = _sha256_bytes(
            _canonical_bytes([_record_location_payload(record) for record in records])
        )
        if records_sha256 != _sha256_text(payload["records_sha256"], "records sha256"):
            raise ContractError("partitioned dense evidence record table hash changed")
        return cls(
            root=resolved,
            partitions=partitions,
            records=records,
            shards=shards,
            memory_capacity=memory_capacity,
        )

    def evidence_for(
        self,
        *,
        source_global_index: int,
        sample_key: str,
        source_input_sha256: str | None = None,
    ) -> DenseEvidence:
        source_global_index = _integer(source_global_index, "runtime source global index")
        try:
            partition = self._partition_by_source_index[source_global_index]
        except KeyError as error:
            raise KeyError(f"unknown dense evidence source index {source_global_index}") from error
        return partition.evidence_for(
            source_global_index=source_global_index,
            sample_key=sample_key,
            source_input_sha256=source_input_sha256,
        )


def _load_dense_evidence_cache(
    root: str | Path,
    *,
    manifest_sha256: str,
    dataset_tree_sha256: str,
    memory_capacity: int,
) -> FrozenDenseEvidenceCache:
    resolved = Path(root).resolve()
    raw_manifest = read_sha256_verified_file_beneath(
        resolved,
        "manifest.json",
        expected_sha256=_sha256_text(manifest_sha256, "cache manifest sha256"),
        maximum_bytes=128 * 1024 * 1024,
    )
    try:
        schema = _mapping(json.loads(raw_manifest), "dense evidence manifest").get("schema")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("dense evidence cache manifest is not valid JSON") from error
    loader = (
        FrozenDenseEvidencePartitionedCache
        if schema == DENSE_EVIDENCE_CACHE_PARTITION_INDEX_SCHEMA
        else FrozenDenseEvidenceCache
    )
    return loader.load(
        resolved,
        manifest_sha256=manifest_sha256,
        dataset_tree_sha256=dataset_tree_sha256,
        memory_capacity=memory_capacity,
    )


class FrozenDenseEvidenceCacheBank:
    """One source-aligned set of immutable modality caches.

    Every cache must cover the same dataset records, including explicit
    zero-token rows for unavailable modalities.  This keeps missing evidence
    distinct from a missing cache artifact and makes multi-modal collation a
    pure lookup rather than a semantic fallback.
    """

    def __init__(self, caches: tuple[FrozenDenseEvidenceCache, ...]) -> None:
        if not isinstance(caches, tuple) or not caches:
            raise ContractError("dense evidence cache bank requires at least one cache")
        if any(not isinstance(cache, FrozenDenseEvidenceCache) for cache in caches):
            raise TypeError("dense evidence cache bank contains an untyped cache")
        names = tuple(cache.contract.modality for cache in caches)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ContractError("dense evidence caches must be sorted by unique modality")
        identity = {
            (
                cache.contract.dataset_id,
                cache.contract.dataset_revision,
                cache.contract.dataset_tree_sha256,
                cache.contract.coverage_plan_sha256,
            )
            for cache in caches
        }
        if len(identity) != 1:
            raise ContractError(
                "dense evidence caches belong to different dataset or coverage revisions"
            )
        first_records = tuple(
            (record.source_global_index, record.sample_key) for record in caches[0].records
        )
        if any(
            tuple((record.source_global_index, record.sample_key) for record in cache.records)
            != first_records
            for cache in caches[1:]
        ):
            raise ContractError("dense evidence caches do not cover identical source records")
        self.caches = caches

    @classmethod
    def load(
        cls,
        roots: Sequence[str | Path],
        *,
        manifest_sha256s: Sequence[str],
        dataset_tree_sha256: str,
        memory_capacity: int = 2,
    ) -> FrozenDenseEvidenceCacheBank:
        if not roots or len(roots) != len(manifest_sha256s):
            raise ContractError("cache roots and manifest hashes must be equal nonempty lists")
        loaded = tuple(
            _load_dense_evidence_cache(
                root,
                manifest_sha256=manifest_sha256,
                dataset_tree_sha256=dataset_tree_sha256,
                memory_capacity=memory_capacity,
            )
            for root, manifest_sha256 in zip(roots, manifest_sha256s, strict=True)
        )
        return cls(tuple(sorted(loaded, key=lambda cache: cache.contract.modality)))

    @property
    def contracts(self) -> tuple[DenseEvidenceCacheContract, ...]:
        return tuple(cache.contract for cache in self.caches)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(contract.modality for contract in self.contracts)

    @property
    def record_count(self) -> int:
        return len(self.caches[0].records)

    @property
    def coverage_plan_sha256(self) -> str:
        return self.caches[0].contract.coverage_plan_sha256

    def evidence_for(
        self,
        *,
        source_global_index: int,
        sample_key: str,
        source_input_sha256_by_modality: Mapping[str, str] | None = None,
    ) -> tuple[DenseEvidence, ...]:
        if source_input_sha256_by_modality is not None and set(
            source_input_sha256_by_modality
        ) != set(self.modalities):
            raise ContractError("runtime source hashes differ from cache-bank modalities")
        return tuple(
            cache.evidence_for(
                source_global_index=source_global_index,
                sample_key=sample_key,
                source_input_sha256=(
                    None
                    if source_input_sha256_by_modality is None
                    else source_input_sha256_by_modality[cache.contract.modality]
                ),
            )
            for cache in self.caches
        )


def compose_dense_evidence_cache_banks(
    banks: Sequence[FrozenDenseEvidenceCacheBank],
    *,
    record_identities: Sequence[tuple[int, str]],
    coverage_plan_sha256: str,
) -> FrozenDenseEvidenceCacheBank:
    """Create one exact zero-copy target view from authenticated cache banks."""

    if not banks or any(not isinstance(bank, FrozenDenseEvidenceCacheBank) for bank in banks):
        raise TypeError("dense evidence composition requires authenticated cache banks")
    modalities = banks[0].modalities
    if any(bank.modalities != modalities for bank in banks[1:]):
        raise ContractError("dense evidence composition sources use different modalities")
    identities = tuple(record_identities)
    views = tuple(
        FrozenDenseEvidenceCacheView(
            sources=tuple(bank.caches[modality_index] for bank in banks),
            record_identities=identities,
            coverage_plan_sha256=coverage_plan_sha256,
        )
        for modality_index in range(len(modalities))
    )
    return FrozenDenseEvidenceCacheBank(views)


def _validate_record(
    record: DenseEvidenceCacheRecord,
    contract: DenseEvidenceCacheContract,
) -> None:
    evidence = record.evidence
    if (
        evidence.modality != contract.modality
        or evidence.encoder_contract != contract.encoder_contract
    ):
        raise ContractError("dense evidence record differs from its cache contract")
    if evidence.tokens.shape[1] != contract.token_width:
        raise ContractError("dense evidence token width differs from its cache contract")
    if evidence.token_count > contract.maximum_tokens:
        raise ContractError("dense evidence record exceeds its cache token budget")
    if contract.geometry_width:
        if evidence.geometry is None or evidence.geometry.shape[1] != contract.geometry_width:
            raise ContractError("dense evidence geometry differs from its cache contract")
    elif evidence.geometry is not None:
        raise ContractError("dense evidence supplied undeclared geometry")
    if contract.has_group_ids != (evidence.group_ids is not None):
        raise ContractError("dense evidence grouping differs from its cache contract")


def _write_shard(
    path: Path,
    records: Sequence[DenseEvidenceCacheRecord],
    contract: DenseEvidenceCacheContract,
) -> tuple[_Shard, list[dict[str, object]]]:
    offsets = [0]
    for record in records:
        offsets.append(offsets[-1] + record.evidence.token_count)
    total = offsets[-1]
    token_dtype = contract.numpy_token_dtype

    def concatenate(name: str, *, dtype: np.dtype, width: int | None = None) -> NDArray:
        values = [np.asarray(getattr(record.evidence, name), dtype=dtype) for record in records]
        if total:
            return np.concatenate(values, axis=0)
        shape = (0,) if width is None else (0, width)
        return np.empty(shape, dtype=dtype)

    tokens = concatenate("tokens", dtype=token_dtype, width=contract.token_width)
    geometry = (
        concatenate("geometry", dtype=np.dtype(np.float32), width=contract.geometry_width)
        if contract.geometry_width
        else np.empty((total, 0), dtype=np.float32)
    )
    timestamps = concatenate("timestamps", dtype=np.dtype(np.float64))
    confidence = concatenate("confidence", dtype=np.dtype(np.float32))
    current = (
        np.concatenate(
            [np.asarray(record.evidence.effective_current_measurement_valid) for record in records],
            axis=0,
        )
        if total
        else np.empty(0, dtype=np.bool_)
    )
    group_ids = (
        concatenate("group_ids", dtype=np.dtype(np.int64))
        if contract.has_group_ids
        else np.full(total, -1, dtype=np.int64)
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                source_global_indices=np.asarray(
                    [record.source_global_index for record in records], dtype=np.int64
                ),
                sample_keys=np.asarray([record.sample_key for record in records], dtype=np.str_),
                source_input_sha256=np.asarray(
                    [record.source_input_sha256 for record in records], dtype="<U64"
                ),
                available=np.asarray(
                    [record.evidence.available for record in records], dtype=np.bool_
                ),
                offsets=np.asarray(offsets, dtype=np.int64),
                tokens=tokens,
                geometry=geometry,
                timestamps=timestamps,
                confidence=confidence,
                current_measurement_valid=current,
                group_ids=group_ids,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    payload = path.read_bytes()
    shard = _Shard(
        path=path.name,
        sha256=_sha256_bytes(payload),
        row_count=len(records),
        token_count=total,
        first_source_global_index=records[0].source_global_index,
        last_source_global_index=records[-1].source_global_index,
    )
    table = [
        {
            "available": record.evidence.available,
            "row": row,
            "sample_key": record.sample_key,
            "shard_index": -1,
            "source_global_index": record.source_global_index,
            "source_input_sha256": record.source_input_sha256,
            "token_count": record.evidence.token_count,
        }
        for row, record in enumerate(records)
    ]
    return shard, table


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def publish_dense_evidence_cache(
    root: str | Path,
    *,
    contract: DenseEvidenceCacheContract,
    records: Iterable[DenseEvidenceCacheRecord],
    shard_rows: int = 2048,
) -> str:
    """Publish a complete cache atomically and return its manifest SHA-256."""

    if not isinstance(contract, DenseEvidenceCacheContract):
        raise TypeError("dense evidence cache publisher requires a typed contract")
    shard_rows = _integer(shard_rows, "shard rows", positive=True)
    destination = Path(root).resolve()
    if destination.exists():
        raise FileExistsError(f"dense evidence cache destination already exists: {destination}")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shards: list[_Shard] = []
        table: list[dict[str, object]] = []
        shard_buffer: list[DenseEvidenceCacheRecord] = []
        previous_source_index = -1
        sample_keys: set[str] = set()

        def flush_shard() -> None:
            if not shard_buffer:
                return
            shard_index = len(shards)
            shard, shard_table = _write_shard(
                staging / f"shard-{shard_index:06d}.npz",
                shard_buffer,
                contract,
            )
            for item in shard_table:
                item["shard_index"] = shard_index
            shards.append(shard)
            table.extend(shard_table)
            shard_buffer.clear()

        for record in records:
            if not isinstance(record, DenseEvidenceCacheRecord):
                raise TypeError("dense evidence cache publisher received an untyped record")
            if record.source_global_index <= previous_source_index:
                raise ContractError("dense evidence cache records must be sorted and source-unique")
            if record.sample_key in sample_keys:
                raise ContractError("dense evidence cache sample keys must be unique")
            _validate_record(record, contract)
            previous_source_index = record.source_global_index
            sample_keys.add(record.sample_key)
            shard_buffer.append(record)
            if len(shard_buffer) == shard_rows:
                flush_shard()
        flush_shard()
        if not table:
            raise ContractError("cannot publish an empty dense evidence cache")
        manifest = {
            "complete": True,
            "contract": contract.payload(),
            "records": table,
            "records_sha256": _sha256_bytes(_canonical_bytes(table)),
            "schema": DENSE_EVIDENCE_CACHE_SCHEMA,
            "shards": [
                {
                    "first_source_global_index": shard.first_source_global_index,
                    "last_source_global_index": shard.last_source_global_index,
                    "path": shard.path,
                    "row_count": shard.row_count,
                    "sha256": shard.sha256,
                    "token_count": shard.token_count,
                }
                for shard in shards
            ],
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        (staging / "manifest.json").write_text(encoded, encoding="ascii")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _sha256_bytes((destination / "manifest.json").read_bytes())


@dataclass(frozen=True, slots=True)
class _ResumePrefix:
    records: tuple[_RecordLocation, ...]
    shards: tuple[_Shard, ...]
    table: tuple[dict[str, object], ...]

    @property
    def state(self) -> DenseEvidenceCacheResumeState:
        return DenseEvidenceCacheResumeState(
            completed_record_count=len(self.records),
            last_source_global_index=(
                None if not self.records else self.records[-1].source_global_index
            ),
        )


def _partial_staging_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.partial")


def _partial_manifest(
    *,
    contract: DenseEvidenceCacheContract,
    expected_record_count: int,
    shard_rows: int,
    table: Sequence[Mapping[str, object]],
    shards: Sequence[_Shard],
) -> dict[str, object]:
    records = [dict(value) for value in table]
    return {
        "complete": False,
        "contract": contract.payload(),
        "expected_record_count": expected_record_count,
        "records": records,
        "records_sha256": _sha256_bytes(_canonical_bytes(records)),
        "schema": DENSE_EVIDENCE_CACHE_PARTIAL_SCHEMA,
        "shard_rows": shard_rows,
        "shards": [
            {
                "first_source_global_index": shard.first_source_global_index,
                "last_source_global_index": shard.last_source_global_index,
                "path": shard.path,
                "row_count": shard.row_count,
                "sha256": shard.sha256,
                "token_count": shard.token_count,
            }
            for shard in shards
        ],
    }


def _load_resume_prefix(
    staging: Path,
    *,
    contract: DenseEvidenceCacheContract,
    expected_record_count: int,
    shard_rows: int,
) -> _ResumePrefix:
    partial_path = staging / "manifest.partial.json"
    if not partial_path.exists():
        return _ResumePrefix(records=(), shards=(), table=())
    if partial_path.is_symlink() or partial_path.stat().st_size > 128 * 1024 * 1024:
        raise ContractError("dense evidence partial manifest is unsafe or unexpectedly large")
    try:
        decoded = json.loads(partial_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("dense evidence partial manifest is not valid JSON") from error
    payload = _exact(
        decoded,
        "dense evidence partial manifest",
        {
            "complete",
            "contract",
            "expected_record_count",
            "records",
            "records_sha256",
            "schema",
            "shard_rows",
            "shards",
        },
    )
    if (
        payload["schema"] != DENSE_EVIDENCE_CACHE_PARTIAL_SCHEMA
        or payload["complete"] is not False
        or payload["contract"] != contract.payload()
        or _integer(
            payload["expected_record_count"], "partial expected record count", positive=True
        )
        != expected_record_count
        or _integer(payload["shard_rows"], "partial shard rows", positive=True) != shard_rows
    ):
        raise ContractError("dense evidence partial publication contract changed")
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise ContractError("dense evidence partial records must be a list")
    if len(raw_records) > expected_record_count:
        raise ContractError("dense evidence partial cache exceeds its expected record count")
    if _sha256_bytes(_canonical_bytes(raw_records)) != _sha256_text(
        payload["records_sha256"], "partial records sha256"
    ):
        raise ContractError("dense evidence partial record table hash changed")
    records = tuple(
        FrozenDenseEvidenceCache._parse_record(value, index=index)
        for index, value in enumerate(raw_records)
    )
    raw_shards = payload["shards"]
    if not isinstance(raw_shards, list):
        raise ContractError("dense evidence partial shards must be a list")
    shards = tuple(
        FrozenDenseEvidenceCache._parse_shard(value, index=index)
        for index, value in enumerate(raw_shards)
    )
    indices = tuple(record.source_global_index for record in records)
    if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
        raise ContractError("dense evidence partial source indices are not sorted and unique")
    if len({record.sample_key for record in records}) != len(records):
        raise ContractError("dense evidence partial sample keys are not unique")
    if sum(shard.row_count for shard in shards) != len(records):
        raise ContractError("dense evidence partial shard rows differ from its record table")
    expected_locations = tuple(
        (shard_index, row)
        for shard_index, shard in enumerate(shards)
        for row in range(shard.row_count)
    )
    if tuple((record.shard_index, record.row) for record in records) != expected_locations:
        raise ContractError("dense evidence partial records do not cover shard rows exactly")
    if any(shard.row_count > shard_rows for shard in shards):
        raise ContractError("dense evidence partial shard exceeds its configured row count")
    for record in records:
        if record.token_count > contract.maximum_tokens:
            raise ContractError("dense evidence partial record exceeds its token budget")
        if not record.available and record.token_count:
            raise ContractError("unavailable dense evidence partial record has tokens")
    if records:
        cache = FrozenDenseEvidenceCache(
            root=staging,
            contract=contract,
            records=records,
            shards=shards,
            memory_capacity=1,
        )
        for shard_index in range(len(shards)):
            cache._load_shard(shard_index)
    table = tuple(dict(_mapping(value, "partial record")) for value in raw_records)
    return _ResumePrefix(records=records, shards=shards, table=table)


def _remove_uncommitted_staging_files(staging: Path, prefix: _ResumePrefix) -> None:
    allowed = {"manifest.partial.json", *(shard.path for shard in prefix.shards)}
    for path in staging.iterdir():
        if path.name in allowed:
            continue
        if path.is_symlink():
            raise ContractError("dense evidence partial staging contains a symlink")
        if path.is_file() and (path.name.startswith(".shard-") or path.name.startswith("shard-")):
            path.unlink()
            continue
        raise ContractError(f"dense evidence partial staging contains an unknown path: {path.name}")


def dense_evidence_cache_resume_state(
    root: str | Path,
    *,
    contract: DenseEvidenceCacheContract,
    expected_record_count: int,
    shard_rows: int = 2048,
) -> DenseEvidenceCacheResumeState:
    """Authenticate and report the durable prefix of an interrupted publication."""

    if not isinstance(contract, DenseEvidenceCacheContract):
        raise TypeError("dense evidence resume inspection requires a typed contract")
    expected_record_count = _integer(expected_record_count, "expected record count", positive=True)
    shard_rows = _integer(shard_rows, "shard rows", positive=True)
    destination = Path(root).resolve()
    if destination.exists():
        raise FileExistsError(f"dense evidence cache destination already exists: {destination}")
    staging = _partial_staging_path(destination)
    if not staging.exists():
        return DenseEvidenceCacheResumeState(0, None)
    if staging.is_symlink() or not staging.is_dir():
        raise ContractError("dense evidence partial staging root is unsafe")
    prefix = _load_resume_prefix(
        staging,
        contract=contract,
        expected_record_count=expected_record_count,
        shard_rows=shard_rows,
    )
    _remove_uncommitted_staging_files(staging, prefix)
    return prefix.state


def publish_dense_evidence_cache_resumable(
    root: str | Path,
    *,
    contract: DenseEvidenceCacheContract,
    expected_record_count: int,
    record_factory: Callable[[int], Iterable[DenseEvidenceCacheRecord]],
    shard_rows: int = 2048,
) -> str:
    """Publish atomically while retaining only authenticated complete shards on failure.

    ``record_factory`` receives the number of already committed records. It must
    begin at exactly that canonical row, allowing expensive frozen encoders to
    skip verified work after a preemption without trusting loose artifacts.
    """

    if not callable(record_factory):
        raise TypeError("resumable dense evidence publication requires a record factory")
    expected_record_count = _integer(expected_record_count, "expected record count", positive=True)
    shard_rows = _integer(shard_rows, "shard rows", positive=True)
    destination = Path(root).resolve()
    if destination.exists():
        raise FileExistsError(f"dense evidence cache destination already exists: {destination}")
    staging = _partial_staging_path(destination)
    if staging.exists() and (staging.is_symlink() or not staging.is_dir()):
        raise ContractError("dense evidence partial staging root is unsafe")
    staging.mkdir(parents=True, exist_ok=True)
    prefix = _load_resume_prefix(
        staging,
        contract=contract,
        expected_record_count=expected_record_count,
        shard_rows=shard_rows,
    )
    _remove_uncommitted_staging_files(staging, prefix)
    shards = list(prefix.shards)
    table = list(prefix.table)
    previous_source_index = prefix.state.last_source_global_index
    sample_keys = {record.sample_key for record in prefix.records}
    shard_buffer: list[DenseEvidenceCacheRecord] = []

    def checkpoint_shard() -> None:
        if not shard_buffer:
            return
        shard_index = len(shards)
        shard, shard_table = _write_shard(
            staging / f"shard-{shard_index:06d}.npz",
            shard_buffer,
            contract,
        )
        for item in shard_table:
            item["shard_index"] = shard_index
        shards.append(shard)
        table.extend(shard_table)
        shard_buffer.clear()
        _write_json_atomic(
            staging / "manifest.partial.json",
            _partial_manifest(
                contract=contract,
                expected_record_count=expected_record_count,
                shard_rows=shard_rows,
                table=table,
                shards=shards,
            ),
        )

    for record in record_factory(len(table)):
        if not isinstance(record, DenseEvidenceCacheRecord):
            raise TypeError("dense evidence cache publisher received an untyped record")
        if (
            previous_source_index is not None
            and record.source_global_index <= previous_source_index
        ):
            raise ContractError(
                "resumed dense evidence records do not continue after the authenticated prefix"
            )
        if record.sample_key in sample_keys:
            raise ContractError("resumed dense evidence cache repeats a sample key")
        _validate_record(record, contract)
        previous_source_index = record.source_global_index
        sample_keys.add(record.sample_key)
        shard_buffer.append(record)
        if len(table) + len(shard_buffer) > expected_record_count:
            raise ContractError("dense evidence record factory exceeded its declared coverage")
        if len(shard_buffer) == shard_rows:
            checkpoint_shard()
    checkpoint_shard()
    if len(table) != expected_record_count:
        raise ContractError(
            "dense evidence record factory ended before its declared complete coverage"
        )

    manifest = {
        "complete": True,
        "contract": contract.payload(),
        "records": table,
        "records_sha256": _sha256_bytes(_canonical_bytes(table)),
        "schema": DENSE_EVIDENCE_CACHE_SCHEMA,
        "shards": [
            {
                "first_source_global_index": shard.first_source_global_index,
                "last_source_global_index": shard.last_source_global_index,
                "path": shard.path,
                "row_count": shard.row_count,
                "sha256": shard.sha256,
                "token_count": shard.token_count,
            }
            for shard in shards
        ],
    }
    _write_json_atomic(staging / "manifest.json", manifest)
    (staging / "manifest.partial.json").unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return _sha256_bytes((destination / "manifest.json").read_bytes())


def merge_dense_evidence_cache_partitions(
    root: str | Path,
    *,
    partition_roots: Sequence[str | Path],
    manifest_sha256s: Sequence[str],
    dataset_tree_sha256: str,
    coverage_plan_sha256: str,
    expected_records: Sequence[tuple[int, str]],
    shard_rows: int = 2048,
    link_shards: bool = False,
    reference_partitions: bool = False,
) -> str:
    """Validate contiguous cache partitions and republish one canonical cache."""

    if not partition_roots or len(partition_roots) != len(manifest_sha256s):
        raise ContractError(
            "cache partition roots and manifest hashes must be equal nonempty lists"
        )
    expected_tree = _sha256_text(dataset_tree_sha256, "dataset tree sha256")
    expected_coverage = _sha256_text(coverage_plan_sha256, "coverage plan sha256")
    shard_rows = _integer(shard_rows, "shard rows", positive=True)
    if not isinstance(link_shards, bool):
        raise TypeError("cache partition shard-link mode must be boolean")
    if not isinstance(reference_partitions, bool):
        raise TypeError("cache partition reference mode must be boolean")
    if link_shards and reference_partitions:
        raise ContractError("cache partitions cannot be linked and referenced together")
    resolved_partition_roots = tuple(Path(value).resolve() for value in partition_roots)
    if len(set(resolved_partition_roots)) != len(resolved_partition_roots):
        raise ContractError("cache partition roots must be unique")
    normalized_expected = tuple(
        (
            _integer(source_global_index, "expected source global index"),
            _text(sample_key, "expected sample key"),
        )
        for source_global_index, sample_key in expected_records
    )
    if not normalized_expected:
        raise ContractError("cache merge requires nonempty expected coverage")
    if normalized_expected != tuple(sorted(normalized_expected)) or len(
        {source_global_index for source_global_index, _ in normalized_expected}
    ) != len(normalized_expected):
        raise ContractError("expected cache coverage must be sorted and source-unique")

    loaded = tuple(
        FrozenDenseEvidenceCache.load(
            partition_root,
            manifest_sha256=manifest_sha256,
            dataset_tree_sha256=expected_tree,
            memory_capacity=1,
        )
        for partition_root, manifest_sha256 in zip(partition_roots, manifest_sha256s, strict=True)
    )
    contract = loaded[0].contract
    if contract.coverage_plan_sha256 != expected_coverage or any(
        cache.contract != contract for cache in loaded[1:]
    ):
        raise ContractError("cache partitions use different contracts or coverage identities")
    ordered = tuple(sorted(loaded, key=lambda cache: cache.records[0].source_global_index))
    observed = tuple(
        (record.source_global_index, record.sample_key)
        for cache in ordered
        for record in cache.records
    )
    if observed != normalized_expected:
        raise ContractError("cache partitions do not exactly cover the canonical record sequence")

    if reference_partitions:
        destination = Path(root).resolve()
        if destination.exists():
            raise FileExistsError(f"dense evidence cache destination already exists: {destination}")
        staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
        if staging.exists():
            raise FileExistsError(f"dense evidence cache staging already exists: {staging}")
        namespace = destination.parent.resolve()
        manifest_by_root = {
            Path(partition_root).resolve(): _sha256_text(manifest_sha256, "manifest sha256")
            for partition_root, manifest_sha256 in zip(
                partition_roots,
                manifest_sha256s,
                strict=True,
            )
        }
        entries: list[dict[str, object]] = []
        for cache in ordered:
            try:
                relative_root = cache.root.relative_to(namespace).as_posix()
            except ValueError as error:
                raise ContractError(
                    "referenced cache partitions must remain inside the output namespace"
                ) from error
            _relative_path(relative_root, "referenced cache partition root")
            for shard_index in range(len(cache.shards)):
                cache._load_shard(shard_index)
            entries.append(
                {
                    "first_source_global_index": cache.records[0].source_global_index,
                    "last_source_global_index": cache.records[-1].source_global_index,
                    "manifest_sha256": manifest_by_root[cache.root],
                    "record_count": len(cache.records),
                    "root": relative_root,
                }
            )
        indexed_records, _ = _aggregate_partition_metadata(ordered)
        manifest = {
            "complete": True,
            "contract": contract.payload(),
            "partitions": entries,
            "records_sha256": _sha256_bytes(
                _canonical_bytes([_record_location_payload(record) for record in indexed_records])
            ),
            "schema": DENSE_EVIDENCE_CACHE_PARTITION_INDEX_SCHEMA,
        }
        staging.mkdir(parents=True)
        try:
            _write_json_atomic(staging / "manifest.json", manifest)
            os.replace(staging, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return _sha256_bytes((destination / "manifest.json").read_bytes())

    if link_shards:
        destination = Path(root).resolve()
        if destination.exists():
            raise FileExistsError(f"dense evidence cache destination already exists: {destination}")
        staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
        if staging.exists():
            raise FileExistsError(f"dense evidence cache staging already exists: {staging}")
        staging.mkdir(parents=True)
        try:
            table: list[dict[str, object]] = []
            shards: list[_Shard] = []
            for cache in ordered:
                shard_offset = len(shards)
                for local_index, shard in enumerate(cache.shards):
                    cache._load_shard(local_index)
                    source = cache.root / shard.path
                    target_name = f"shard-{len(shards):06d}.npz"
                    target = staging / target_name
                    if source.is_symlink() or not source.is_file():
                        raise ContractError("cache partition shard is absent or indirect")
                    if source.stat().st_dev != staging.stat().st_dev:
                        raise ContractError(
                            "linked cache merge requires partitions on the destination filesystem"
                        )
                    os.link(source, target, follow_symlinks=False)
                    shards.append(
                        _Shard(
                            path=target_name,
                            sha256=shard.sha256,
                            row_count=shard.row_count,
                            token_count=shard.token_count,
                            first_source_global_index=shard.first_source_global_index,
                            last_source_global_index=shard.last_source_global_index,
                        )
                    )
                table.extend(
                    {
                        "available": record.available,
                        "row": record.row,
                        "sample_key": record.sample_key,
                        "shard_index": shard_offset + record.shard_index,
                        "source_global_index": record.source_global_index,
                        "source_input_sha256": record.source_input_sha256,
                        "token_count": record.token_count,
                    }
                    for record in cache.records
                )
            manifest = {
                "complete": True,
                "contract": contract.payload(),
                "records": table,
                "records_sha256": _sha256_bytes(_canonical_bytes(table)),
                "schema": DENSE_EVIDENCE_CACHE_SCHEMA,
                "shards": [
                    {
                        "first_source_global_index": shard.first_source_global_index,
                        "last_source_global_index": shard.last_source_global_index,
                        "path": shard.path,
                        "row_count": shard.row_count,
                        "sha256": shard.sha256,
                        "token_count": shard.token_count,
                    }
                    for shard in shards
                ],
            }
            _write_json_atomic(staging / "manifest.json", manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return _sha256_bytes((destination / "manifest.json").read_bytes())

    def records() -> Iterable[DenseEvidenceCacheRecord]:
        for cache in ordered:
            for location in cache.records:
                yield DenseEvidenceCacheRecord(
                    source_global_index=location.source_global_index,
                    sample_key=location.sample_key,
                    source_input_sha256=location.source_input_sha256,
                    evidence=cache.evidence_for(
                        source_global_index=location.source_global_index,
                        sample_key=location.sample_key,
                        source_input_sha256=location.source_input_sha256,
                    ),
                )

    return publish_dense_evidence_cache(
        root,
        contract=contract,
        records=records(),
        shard_rows=shard_rows,
    )
