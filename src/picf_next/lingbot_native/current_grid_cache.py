"""Immutable current-frame DINO patch bank for LingBot training.

The cache is a query-independent loss-side artifact. It stores detached patch
features indexed only by content-addressed source frames. Each objective
materializer binds its own query schema. Object ownership is resolved after the
weight-shared forward and never affects host inputs or recurrent state.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath
from picf_next.data.lingbot_calvin_projection import LINGBOT_CALVIN_CAMERA_SLOTS
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_cache import (
    LINGBOT_PREDICTIVE_MODALITY,
    LINGBOT_PREDICTIVE_TARGET_SPACE,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveTarget,
    TargetEncoderMode,
    make_native_predictive_target,
    make_object_summary_target,
)
from picf_next.lingbot_native.source_mask import QwenWholeViewOmission

LINGBOT_CURRENT_GRID_CACHE_SCHEMA = "picf-next.lingbot-current-grid-cache/v2"
LINGBOT_CURRENT_GRID_ADDRESS_SCHEMA = "qwen-merged-static-cell-center-xy-minus1-plus1/v1"
LINGBOT_CURRENT_GRID_VALIDITY = (
    "positive-task-independent-visible-owner-area-at-selected-current-static-grid-address/v1"
)
LINGBOT_CURRENT_CORRECTION_SUMMARY_ADDRESS_SCHEMA = (
    "one-nonspatial-prior-to-current-object-summary/v1"
)
LINGBOT_CURRENT_CORRECTION_SUMMARY_VALIDITY = (
    "valid-prior-and-positive-current-static-visible-owner-image-fraction/v1"
)
LINGBOT_OMITTED_STATIC_SUMMARY_ADDRESS_SCHEMA = "one-nonspatial-static-object-summary/v1"
LINGBOT_OMITTED_STATIC_SUMMARY_VALIDITY = (
    "positive-static-owner-support-and-positive-visible-owner-support-in-remaining-gripper-view/v1"
)
_SOURCE_KEYS_SCHEMA = b"picf-next.lingbot-current-grid-source-keys/v1\0"
_MANIFEST_MAXIMUM_BYTES = 32 * 1024 * 1024
_SHARD_OVERHEAD_BYTES = 16 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pooled_owner_patch_support(
    *,
    owner_index: NDArray[np.integer],
    owner_supervised: NDArray[np.bool_],
    owner_ids: Sequence[int],
    input_size: int,
    patch_tokens: int,
) -> torch.Tensor:
    """Apply the correction estimator's exact CPU mask resize and pooling."""

    grid = int(round(patch_tokens**0.5))
    if grid * grid != patch_tokens:
        raise ContractError("current-correction patch count must form a square grid")
    if owner_index.shape != owner_supervised.shape or owner_index.ndim != 2:
        raise ContractError("current-correction owner and supervision rasters differ")
    if any(
        isinstance(owner_id, bool) or not isinstance(owner_id, int) or owner_id <= 0
        for owner_id in owner_ids
    ):
        raise ValueError("current-correction owner IDs must be positive integers")
    support = torch.zeros(patch_tokens, len(owner_ids), dtype=torch.float32)
    owners = torch.from_numpy(owner_index.copy())
    supervised = torch.from_numpy(owner_supervised.copy())
    for track_index, owner_id in enumerate(owner_ids):
        visible = ((owners == owner_id) & supervised).float()[None, None]
        resized = F.interpolate(
            visible,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        support[:, track_index] = (
            F.adaptive_avg_pool2d(resized, (grid, grid)).reshape(-1).clamp(0, 1)
        )
    return support


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != fields:
        raise ContractError(f"{name} fields differ from the frozen schema")
    return payload


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be non-empty text")
    return value


def _sha256(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return text


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


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


def current_grid_query_schema_digest(*, route_id: int) -> str:
    if isinstance(route_id, bool) or not isinstance(route_id, int) or route_id < 0:
        raise ValueError("current-grid route ID must be a non-negative integer")
    return _sha256_bytes(
        _canonical_json(
            {
                "address_schema": LINGBOT_CURRENT_GRID_ADDRESS_SCHEMA,
                "evidence": PredictionEvidence.CURRENT_RANDOM_GRID.value,
                "horizon": 0,
                "route_id": route_id,
                "source": PredictionSource.POSTERIOR.value,
                "target_space": LINGBOT_PREDICTIVE_TARGET_SPACE,
            }
        )
    )


def omitted_static_summary_query_schema_digest(*, route_id: int) -> str:
    """Digest the single-query cross-view object-binding contract."""

    if isinstance(route_id, bool) or not isinstance(route_id, int) or route_id < 0:
        raise ValueError("omitted-static route ID must be a non-negative integer")
    return _sha256_bytes(
        _canonical_json(
            {
                "address_schema": LINGBOT_OMITTED_STATIC_SUMMARY_ADDRESS_SCHEMA,
                "evidence": PredictionEvidence.OMITTED_MODALITY.value,
                "horizon": 0,
                "route_id": route_id,
                "source": PredictionSource.POSTERIOR.value,
                "target_space": LINGBOT_PREDICTIVE_TARGET_SPACE,
            }
        )
    )


def current_correction_summary_query_schema_digest(
    *,
    route_id: int,
    address_width: int,
    source: PredictionSource = PredictionSource.PRIOR,
    evidence: PredictionEvidence = PredictionEvidence.CURRENT_CORRECTION,
) -> str:
    """Digest one current object-summary query in an explicit latent phase."""

    if isinstance(route_id, bool) or not isinstance(route_id, int) or route_id < 0:
        raise ValueError("current-correction route ID must be a non-negative integer")
    if isinstance(address_width, bool) or not isinstance(address_width, int) or address_width < 0:
        raise ValueError("current-correction address width must be non-negative")
    valid_pair = (source, evidence) in {
        (PredictionSource.PRIOR, PredictionEvidence.CURRENT_CORRECTION),
        (PredictionSource.PRIOR, PredictionEvidence.CURRENT_PRIOR),
        (PredictionSource.POSTERIOR, PredictionEvidence.CURRENT_POSTERIOR),
    }
    if not valid_pair:
        raise ValueError("current-summary source and evidence phase are inconsistent")
    return _sha256_bytes(
        _canonical_json(
            {
                "address_schema": LINGBOT_CURRENT_CORRECTION_SUMMARY_ADDRESS_SCHEMA,
                "address_width": address_width,
                "evidence": evidence.value,
                "horizon": 0,
                "route_id": route_id,
                "source": source.value,
                "target_space": LINGBOT_PREDICTIVE_TARGET_SPACE,
            }
        )
    )


def current_grid_source_keys_digest(source_global_indices: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_KEYS_SCHEMA)
    previous: int | None = None
    count = 0
    for value in source_global_indices:
        index = _nonnegative_int(value, "current-grid source index")
        if previous is not None and index <= previous:
            raise ContractError("current-grid source indices must be sorted and unique")
        digest.update(index.to_bytes(8, byteorder="big", signed=False))
        previous = index
        count += 1
    if count == 0:
        raise ContractError("current-grid source coverage cannot be empty")
    digest.update(count.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def current_grid_coverage_digest(
    *,
    dataset_tree_sha256: str,
    stream_plan_sha256: str,
    temporal_estimator_sha256: str,
    source_keys_sha256: str,
    expected_record_count: int,
) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "dataset_tree_sha256": _sha256(
                    dataset_tree_sha256, "current-grid dataset tree sha256"
                ),
                "expected_record_count": _positive_int(
                    expected_record_count, "current-grid expected record count"
                ),
                "source_keys_sha256": _sha256(
                    source_keys_sha256, "current-grid source keys sha256"
                ),
                "stream_plan_sha256": _sha256(
                    stream_plan_sha256, "current-grid stream plan sha256"
                ),
                "temporal_estimator_sha256": _sha256(
                    temporal_estimator_sha256, "current-grid temporal estimator sha256"
                ),
            }
        )
    )


@dataclass(frozen=True, slots=True)
class CurrentGridCacheContract:
    dataset_id: str
    dataset_revision: str
    split_name: str
    dataset_tree_sha256: str
    physical_sidecar_manifest_sha256: str
    lingbot_source_commit: str
    lingbot_checkpoint_revision: str
    teacher_config_sha256: str
    teacher_checkpoint_sha256: str
    stream_plan_sha256: str
    temporal_estimator_sha256: str
    source_keys_sha256: str
    coverage_sha256: str
    expected_record_count: int
    hidden_size: int = 1024
    input_size: int = 256
    patch_tokens: int = 256
    route_id: int = 0
    camera_name: str = "static"

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "dataset_revision",
            "split_name",
            "lingbot_source_commit",
            "lingbot_checkpoint_revision",
            "camera_name",
        ):
            _text(getattr(self, name), name)
        for name in (
            "dataset_tree_sha256",
            "physical_sidecar_manifest_sha256",
            "teacher_config_sha256",
            "teacher_checkpoint_sha256",
            "stream_plan_sha256",
            "temporal_estimator_sha256",
            "source_keys_sha256",
            "coverage_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("hidden_size", "input_size", "patch_tokens", "expected_record_count"):
            _positive_int(getattr(self, name), name)
        _nonnegative_int(self.route_id, "current-grid route ID")
        if self.camera_name != "static":
            raise ContractError("current-grid targets require the frozen static camera")
        if self.input_size != 256 or self.patch_tokens != 256 or self.hidden_size != 1024:
            raise ContractError("released current-grid DINO geometry changed")
        expected_coverage = current_grid_coverage_digest(
            dataset_tree_sha256=self.dataset_tree_sha256,
            stream_plan_sha256=self.stream_plan_sha256,
            temporal_estimator_sha256=self.temporal_estimator_sha256,
            source_keys_sha256=self.source_keys_sha256,
            expected_record_count=self.expected_record_count,
        )
        if expected_coverage != self.coverage_sha256:
            raise ContractError("current-grid coverage digest differs from its semantics")

    @property
    def encoder_digest(self) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "checkpoint_revision": self.lingbot_checkpoint_revision,
                    "checkpoint_sha256": self.teacher_checkpoint_sha256,
                    "config_sha256": self.teacher_config_sha256,
                    "hidden_size": self.hidden_size,
                    "input_size": self.input_size,
                    "lingbot_source_commit": self.lingbot_source_commit,
                    "patch_tokens": self.patch_tokens,
                    "target": "released-dino-video-current-patch",
                }
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: object) -> CurrentGridCacheContract:
        fields = set(cls.__dataclass_fields__)
        payload = _exact(value, "current-grid contract", fields)
        return cls(
            dataset_id=_text(payload["dataset_id"], "dataset_id"),
            dataset_revision=_text(payload["dataset_revision"], "dataset_revision"),
            split_name=_text(payload["split_name"], "split_name"),
            dataset_tree_sha256=_sha256(payload["dataset_tree_sha256"], "dataset_tree_sha256"),
            physical_sidecar_manifest_sha256=_sha256(
                payload["physical_sidecar_manifest_sha256"],
                "physical_sidecar_manifest_sha256",
            ),
            lingbot_source_commit=_text(payload["lingbot_source_commit"], "lingbot_source_commit"),
            lingbot_checkpoint_revision=_text(
                payload["lingbot_checkpoint_revision"], "lingbot_checkpoint_revision"
            ),
            teacher_config_sha256=_sha256(
                payload["teacher_config_sha256"], "teacher_config_sha256"
            ),
            teacher_checkpoint_sha256=_sha256(
                payload["teacher_checkpoint_sha256"], "teacher_checkpoint_sha256"
            ),
            stream_plan_sha256=_sha256(payload["stream_plan_sha256"], "stream_plan_sha256"),
            temporal_estimator_sha256=_sha256(
                payload["temporal_estimator_sha256"], "temporal_estimator_sha256"
            ),
            source_keys_sha256=_sha256(payload["source_keys_sha256"], "source_keys_sha256"),
            coverage_sha256=_sha256(payload["coverage_sha256"], "coverage_sha256"),
            expected_record_count=_positive_int(
                payload["expected_record_count"], "expected_record_count"
            ),
            hidden_size=_positive_int(payload["hidden_size"], "hidden_size"),
            input_size=_positive_int(payload["input_size"], "input_size"),
            patch_tokens=_positive_int(payload["patch_tokens"], "patch_tokens"),
            route_id=_nonnegative_int(payload["route_id"], "route_id"),
            camera_name=_text(payload["camera_name"], "camera_name"),
        )


@dataclass(frozen=True, slots=True)
class CurrentGridCacheRecord:
    source_global_index: int
    source_rgb_sha256: str
    features: NDArray[np.float16]

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_global_index, "current-grid source index")
        _sha256(self.source_rgb_sha256, "current-grid source RGB sha256")
        if (
            not isinstance(self.features, np.ndarray)
            or self.features.dtype != np.float16
            or self.features.ndim != 2
            or not np.isfinite(self.features).all()
        ):
            raise ContractError("current-grid features must be finite float16 [patches,width]")


@dataclass(frozen=True, slots=True)
class _CurrentGridShard:
    path: str
    sha256: str
    row_count: int
    first_source_global_index: int
    last_source_global_index: int


@dataclass(frozen=True, slots=True)
class _LoadedCurrentGridShard:
    source_global_indices: NDArray[np.int64]
    source_rgb_sha256: NDArray[np.str_]
    features: NDArray[np.float16]


def _write_shard(path: Path, records: Sequence[CurrentGridCacheRecord]) -> _CurrentGridShard:
    indices = np.asarray([record.source_global_index for record in records], dtype=np.int64)
    hashes = np.asarray([record.source_rgb_sha256 for record in records], dtype="<U64")
    features = np.stack([record.features for record in records]).astype(np.float16, copy=False)
    np.savez(path, source_global_indices=indices, source_rgb_sha256=hashes, features=features)
    return _CurrentGridShard(
        path=path.name,
        sha256=_sha256_file(path),
        row_count=len(records),
        first_source_global_index=int(indices[0]),
        last_source_global_index=int(indices[-1]),
    )


def write_current_grid_target_cache(
    root: str | Path,
    *,
    contract: CurrentGridCacheContract,
    records: Iterable[CurrentGridCacheRecord],
    shard_rows: int = 2048,
) -> str:
    """Atomically publish an exact ordered current-target cache."""

    if not isinstance(contract, CurrentGridCacheContract):
        raise TypeError("current-grid writer requires a typed contract")
    _positive_int(shard_rows, "current-grid shard rows")
    destination = Path(os.path.abspath(os.fspath(root)))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.incomplete")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir()
    shards: list[_CurrentGridShard] = []
    sources: list[int] = []
    buffer: list[CurrentGridCacheRecord] = []
    previous: int | None = None
    published = False
    try:
        for record in records:
            if not isinstance(record, CurrentGridCacheRecord):
                raise TypeError("current-grid writer received an invalid record")
            if record.features.shape != (contract.patch_tokens, contract.hidden_size):
                raise ContractError("current-grid record differs from the teacher geometry")
            if previous is not None and record.source_global_index <= previous:
                raise ContractError("current-grid records must be source-sorted and unique")
            previous = record.source_global_index
            sources.append(record.source_global_index)
            buffer.append(record)
            if len(buffer) == shard_rows:
                path = staging / f"shard-{len(shards):06d}.npz"
                shards.append(_write_shard(path, buffer))
                buffer.clear()
        if buffer:
            path = staging / f"shard-{len(shards):06d}.npz"
            shards.append(_write_shard(path, buffer))
        if len(sources) != contract.expected_record_count:
            raise ContractError("current-grid record count differs from frozen coverage")
        if current_grid_source_keys_digest(sources) != contract.source_keys_sha256:
            raise ContractError("current-grid records differ from frozen source coverage")
        manifest = {
            "schema": LINGBOT_CURRENT_GRID_CACHE_SCHEMA,
            "contract": contract.to_dict(),
            "source_global_indices": sources,
            "shards": [
                {
                    "path": shard.path,
                    "sha256": shard.sha256,
                    "row_count": shard.row_count,
                    "first_source_global_index": shard.first_source_global_index,
                    "last_source_global_index": shard.last_source_global_index,
                }
                for shard in shards
            ],
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("ascii") + b"\n"
        if len(manifest_bytes) > _MANIFEST_MAXIMUM_BYTES:
            raise ContractError("current-grid manifest exceeds the bounded size")
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        for shard in shards:
            with (staging / shard.path).open("rb") as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, destination)
        published = True
        parent_descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return _sha256_bytes(manifest_bytes)
    except BaseException:
        rename_completed = not staging.exists() and destination.is_dir()
        if published or rename_completed:
            shutil.rmtree(destination, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def rebind_current_grid_target_cache(
    root: str | Path,
    *,
    source_cache: LingBotCurrentGridTargetCache,
    contract: CurrentGridCacheContract,
    source_rgb_sha256_for: Callable[[int], str],
) -> str:
    """Publish a new plan identity over an exactly equivalent frozen source bank.

    DINO current-grid rows are a deterministic function of one RGB frame and
    one frozen encoder. Rebinding is therefore valid only when the complete
    target source set is unchanged, the encoder is unchanged, and every target
    RGB digest matches the authenticated donor row. Runtime loading remains
    bound to the newly published target contract. The target directory contains
    only that manifest; authenticated shard bytes remain in the caller-supplied
    content-addressed source bank. This avoids a second large copy on persistent
    filesystems without hard-link support.
    """

    if not isinstance(source_cache, LingBotCurrentGridTargetCache):
        raise TypeError("current-grid rebind requires one authenticated source cache")
    if not isinstance(contract, CurrentGridCacheContract):
        raise TypeError("current-grid rebind requires one typed target contract")
    if not callable(source_rgb_sha256_for):
        raise TypeError("current-grid rebind requires one RGB identity resolver")
    source_contract = source_cache.contract
    if (
        source_contract.encoder_digest != contract.encoder_digest
        or source_contract.input_size != contract.input_size
        or source_contract.patch_tokens != contract.patch_tokens
        or source_contract.hidden_size != contract.hidden_size
        or source_contract.camera_name != contract.camera_name
    ):
        raise ContractError("current-grid rebind changed frozen encoder semantics")
    sources = source_cache.source_global_indices
    if (
        len(sources) != contract.expected_record_count
        or current_grid_source_keys_digest(sources) != contract.source_keys_sha256
    ):
        raise ContractError("current-grid rebind changed exact source coverage")

    # Authenticate every donor shard and prove equality of every encoder input
    # before publishing any target artifact.
    for source_global_index in sources:
        observed_rgb_sha256, _features = source_cache._record(source_global_index)
        expected_rgb_sha256 = _sha256(
            source_rgb_sha256_for(source_global_index),
            "target current-grid RGB sha256",
        )
        if observed_rgb_sha256 != expected_rgb_sha256:
            raise ContractError("current-grid rebind source RGB identity differs")

    destination = Path(os.path.abspath(os.fspath(root)))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.rebind-incomplete")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir()
    published = False
    try:
        for shard in source_cache.shards:
            source_path = source_cache.root / shard.path
            if source_path.is_symlink() or not source_path.is_file():
                raise ContractError("current-grid rebind donor shard is absent or indirect")
        manifest = {
            "schema": LINGBOT_CURRENT_GRID_CACHE_SCHEMA,
            "contract": contract.to_dict(),
            "source_global_indices": list(sources),
            "shards": [
                {
                    "path": shard.path,
                    "sha256": shard.sha256,
                    "row_count": shard.row_count,
                    "first_source_global_index": shard.first_source_global_index,
                    "last_source_global_index": shard.last_source_global_index,
                }
                for shard in source_cache.shards
            ],
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("ascii") + b"\n"
        if len(manifest_bytes) > _MANIFEST_MAXIMUM_BYTES:
            raise ContractError("current-grid rebind manifest exceeds the bounded size")
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, destination)
        published = True
        parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return _sha256_bytes(manifest_bytes)
    except BaseException:
        rename_completed = not staging.exists() and destination.is_dir()
        if published or rename_completed:
            shutil.rmtree(destination, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


class LingBotCurrentGridTargetCache:
    """Bounded reader and exact current-grid target materializer."""

    def __init__(
        self,
        *,
        root: Path,
        manifest_root: Path,
        manifest_sha256: str,
        contract: CurrentGridCacheContract,
        shards: tuple[_CurrentGridShard, ...],
        source_global_indices: tuple[int, ...],
        locator: dict[int, tuple[int, int]],
        memory_capacity: int,
    ) -> None:
        self.root = root
        self.manifest_root = manifest_root
        self.manifest_sha256 = manifest_sha256
        self.contract = contract
        self.shards = shards
        self.source_global_indices = source_global_indices
        self.locator = locator
        self.memory_capacity = memory_capacity
        self._loaded: OrderedDict[int, _LoadedCurrentGridShard] = OrderedDict()

    @classmethod
    def load_reusable_source_bank(
        cls,
        root: str | Path,
        *,
        manifest_sha256: str,
        dataset_tree_sha256: str,
        physical_sidecar_manifest_sha256: str,
        encoder_digest: str,
        memory_capacity: int = 2,
    ) -> LingBotCurrentGridTargetCache:
        """Load a verified cache as a query-independent source-feature donor.

        A donor may belong to another frozen stream plan, but it must bind the
        exact dataset, physical sidecar, and teacher encoder.  The returned
        cache still validates its own complete coverage before any record can
        be read; callers must publish a new exact plan-bound cache rather than
        using this relaxed identity at training time.
        """

        cache_root = Path(root).resolve()
        expected_manifest = _sha256(manifest_sha256, "current-grid manifest sha256")
        manifest_bytes = read_sha256_verified_file_beneath(
            cache_root,
            "manifest.json",
            expected_sha256=expected_manifest,
            maximum_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
        try:
            raw = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise ContractError("current-grid manifest is not valid JSON") from error
        payload = _exact(
            raw,
            "current-grid manifest",
            {"schema", "contract", "source_global_indices", "shards"},
        )
        if payload["schema"] != LINGBOT_CURRENT_GRID_CACHE_SCHEMA:
            raise ContractError("current-grid manifest schema differs")
        contract = CurrentGridCacheContract.from_mapping(payload["contract"])
        expected_identity = (
            _sha256(dataset_tree_sha256, "current-grid dataset sha256"),
            _sha256(physical_sidecar_manifest_sha256, "current-grid sidecar sha256"),
            _sha256(encoder_digest, "current-grid encoder digest"),
        )
        observed_identity = (
            contract.dataset_tree_sha256,
            contract.physical_sidecar_manifest_sha256,
            contract.encoder_digest,
        )
        if observed_identity != expected_identity:
            raise ContractError("reusable current-grid source bank provenance differs")
        return cls.load(
            cache_root,
            manifest_sha256=expected_manifest,
            dataset_tree_sha256=contract.dataset_tree_sha256,
            physical_sidecar_manifest_sha256=contract.physical_sidecar_manifest_sha256,
            encoder_digest=contract.encoder_digest,
            coverage_sha256=contract.coverage_sha256,
            memory_capacity=memory_capacity,
        )

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        shard_root: str | Path | None = None,
        manifest_sha256: str,
        dataset_tree_sha256: str,
        physical_sidecar_manifest_sha256: str,
        encoder_digest: str,
        coverage_sha256: str,
        memory_capacity: int = 2,
    ) -> LingBotCurrentGridTargetCache:
        _positive_int(memory_capacity, "current-grid memory capacity")
        cache_root = Path(root).resolve()
        cache_shard_root = cache_root if shard_root is None else Path(shard_root).resolve()
        if cache_shard_root.is_symlink() or not cache_shard_root.is_dir():
            raise ContractError("current-grid shard root must be one direct directory")
        expected_manifest = _sha256(manifest_sha256, "current-grid manifest sha256")
        manifest_bytes = read_sha256_verified_file_beneath(
            cache_root,
            "manifest.json",
            expected_sha256=expected_manifest,
            maximum_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
        actual_manifest = _sha256_bytes(manifest_bytes)
        try:
            raw = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise ContractError("current-grid manifest is not valid JSON") from error
        payload = _exact(
            raw,
            "current-grid manifest",
            {"schema", "contract", "source_global_indices", "shards"},
        )
        if payload["schema"] != LINGBOT_CURRENT_GRID_CACHE_SCHEMA:
            raise ContractError("current-grid manifest schema differs")
        contract = CurrentGridCacheContract.from_mapping(payload["contract"])
        expected = (
            _sha256(dataset_tree_sha256, "current-grid dataset sha256"),
            _sha256(physical_sidecar_manifest_sha256, "current-grid sidecar sha256"),
            _sha256(encoder_digest, "current-grid encoder digest"),
            _sha256(coverage_sha256, "current-grid coverage sha256"),
        )
        observed = (
            contract.dataset_tree_sha256,
            contract.physical_sidecar_manifest_sha256,
            contract.encoder_digest,
            contract.coverage_sha256,
        )
        if observed != expected:
            raise ContractError("current-grid cache provenance differs from runtime")
        source_raw = payload["source_global_indices"]
        if not isinstance(source_raw, list):
            raise ContractError("current-grid source indices must be a list")
        sources = tuple(
            _nonnegative_int(value, "current-grid source index") for value in source_raw
        )
        if (
            len(sources) != contract.expected_record_count
            or current_grid_source_keys_digest(sources) != contract.source_keys_sha256
        ):
            raise ContractError("current-grid manifest source coverage differs")
        shard_raw = payload["shards"]
        if not isinstance(shard_raw, list) or not shard_raw:
            raise ContractError("current-grid manifest requires shards")
        shards: list[_CurrentGridShard] = []
        locator: dict[int, tuple[int, int]] = {}
        cursor = 0
        for shard_index, value in enumerate(shard_raw):
            item = _exact(
                value,
                "current-grid shard",
                {
                    "path",
                    "sha256",
                    "row_count",
                    "first_source_global_index",
                    "last_source_global_index",
                },
            )
            shard = _CurrentGridShard(
                path=_relative_path(item["path"], "current-grid shard path"),
                sha256=_sha256(item["sha256"], "current-grid shard sha256"),
                row_count=_positive_int(item["row_count"], "current-grid shard rows"),
                first_source_global_index=_nonnegative_int(
                    item["first_source_global_index"], "current-grid first source"
                ),
                last_source_global_index=_nonnegative_int(
                    item["last_source_global_index"], "current-grid last source"
                ),
            )
            shard_sources = sources[cursor : cursor + shard.row_count]
            if (
                len(shard_sources) != shard.row_count
                or shard_sources[0] != shard.first_source_global_index
                or shard_sources[-1] != shard.last_source_global_index
            ):
                raise ContractError("current-grid shard ranges differ from source coverage")
            for row, source in enumerate(shard_sources):
                locator[source] = (shard_index, row)
            cursor += shard.row_count
            shards.append(shard)
        if cursor != len(sources) or len(locator) != len(sources):
            raise ContractError("current-grid shard coverage is incomplete")
        return cls(
            root=cache_shard_root,
            manifest_root=cache_root,
            manifest_sha256=actual_manifest,
            contract=contract,
            shards=tuple(shards),
            source_global_indices=sources,
            locator=locator,
            memory_capacity=memory_capacity,
        )

    def _load_shard(self, shard_index: int) -> _LoadedCurrentGridShard:
        cached = self._loaded.get(shard_index)
        if cached is not None:
            self._loaded.move_to_end(shard_index)
            return cached
        metadata = self.shards[shard_index]
        maximum_bytes = (
            metadata.row_count
            * self.contract.patch_tokens
            * self.contract.hidden_size
            * np.dtype(np.float16).itemsize
            + _SHARD_OVERHEAD_BYTES
        )
        payload = read_sha256_verified_file_beneath(
            self.root,
            metadata.path,
            expected_sha256=metadata.sha256,
            maximum_bytes=maximum_bytes,
        )
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if set(archive.files) != {
                    "source_global_indices",
                    "source_rgb_sha256",
                    "features",
                }:
                    raise ContractError("current-grid shard arrays differ from schema")
                indices = archive["source_global_indices"].copy()
                hashes = archive["source_rgb_sha256"].copy()
                features = archive["features"].copy()
        except (OSError, ValueError) as error:
            raise ContractError("current-grid cache shard is not a safe NPZ archive") from error
        if (
            indices.dtype != np.int64
            or indices.shape != (metadata.row_count,)
            or hashes.shape != (metadata.row_count,)
            or hashes.dtype.kind != "U"
            or features.dtype != np.float16
            or features.shape
            != (metadata.row_count, self.contract.patch_tokens, self.contract.hidden_size)
            or not np.isfinite(features).all()
            or int(indices[0]) != metadata.first_source_global_index
            or int(indices[-1]) != metadata.last_source_global_index
        ):
            raise ContractError("current-grid shard tensors differ from contract")
        source_offset = sum(shard.row_count for shard in self.shards[:shard_index])
        expected_sources = np.asarray(
            self.source_global_indices[source_offset : source_offset + metadata.row_count],
            dtype=np.int64,
        )
        if not np.array_equal(indices, expected_sources):
            raise ContractError("current-grid shard indices differ from manifest coverage")
        for digest in hashes.tolist():
            _sha256(str(digest), "current-grid shard RGB sha256")
        loaded = _LoadedCurrentGridShard(indices, hashes, features)
        self._loaded[shard_index] = loaded
        self._loaded.move_to_end(shard_index)
        while len(self._loaded) > self.memory_capacity:
            self._loaded.popitem(last=False)
        return loaded

    def _record(self, source_global_index: int) -> tuple[str, NDArray[np.float16]]:
        location = self.locator.get(source_global_index)
        if location is None:
            raise ContractError(f"current-grid cache omits source frame {source_global_index}")
        shard_index, row = location
        shard = self._load_shard(shard_index)
        if int(shard.source_global_indices[row]) != source_global_index:
            raise ContractError("current-grid locator differs from shard contents")
        return str(shard.source_rgb_sha256[row]), shard.features[row]

    def iter_records(self) -> Iterable[CurrentGridCacheRecord]:
        """Yield validated defensive copies in canonical source order."""

        for shard_index, metadata in enumerate(self.shards):
            loaded = self._load_shard(shard_index)
            for row in range(metadata.row_count):
                yield CurrentGridCacheRecord(
                    source_global_index=int(loaded.source_global_indices[row]),
                    source_rgb_sha256=str(loaded.source_rgb_sha256[row]),
                    features=loaded.features[row].copy(),
                )

    def record_for(self, *, source_global_index: int) -> CurrentGridCacheRecord | None:
        """Return one validated defensive current record, or ``None`` outside coverage."""

        source = _nonnegative_int(source_global_index, "current-grid source global index")
        location = self.locator.get(source)
        if location is None:
            return None
        cached_hash, features = self._record(source)
        return CurrentGridCacheRecord(
            source_global_index=source,
            source_rgb_sha256=cached_hash,
            features=features.copy(),
        )

    def has_supported_current_summary(
        self,
        *,
        source_global_index: int,
        physical_sidecar: CalvinPhysicalSupervisionSidecar,
        minimum_visible_fraction: float,
    ) -> bool:
        """Apply the exact target estimator and report any valid current object."""

        return (
            self.supported_current_summary_count(
                source_global_index=source_global_index,
                physical_sidecar=physical_sidecar,
                minimum_visible_fraction=minimum_visible_fraction,
            )
            > 0
        )

    def supported_current_summary_count(
        self,
        *,
        source_global_index: int,
        physical_sidecar: CalvinPhysicalSupervisionSidecar,
        minimum_visible_fraction: float,
    ) -> int:
        """Count valid object summaries under the exact target support estimator."""

        if not isinstance(physical_sidecar, CalvinPhysicalSupervisionSidecar):
            raise TypeError("current-summary coverage requires a verified physical sidecar")
        if (
            isinstance(minimum_visible_fraction, bool)
            or not isinstance(minimum_visible_fraction, (int, float))
            or not np.isfinite(minimum_visible_fraction)
            or not 0 <= minimum_visible_fraction < 1
        ):
            raise ValueError("minimum visible fraction must lie in [0,1)")
        if physical_sidecar.manifest_sha256 != self.contract.physical_sidecar_manifest_sha256:
            raise ContractError("current-summary coverage sidecar provenance differs")
        source = _nonnegative_int(
            source_global_index,
            "current-summary source global index",
        )
        cached_hash, _features = self._record(source)
        frame = physical_sidecar.source_frame(source)
        cameras = tuple(camera for camera in frame.cameras if camera.camera_name == "static")
        if len(cameras) != 1 or cameras[0].source_rgb_sha256 != cached_hash:
            raise ContractError("current-summary coverage raster differs from cached RGB")
        camera = cameras[0]
        support = _pooled_owner_patch_support(
            owner_index=camera.owner_index,
            owner_supervised=camera.owner_supervised,
            owner_ids=tuple(range(1, len(frame.identity_keys) + 1)),
            input_size=self.contract.input_size,
            patch_tokens=self.contract.patch_tokens,
        )
        return int((support.mean(dim=0) > minimum_visible_fraction).sum().item())

    def target_for(
        self,
        *,
        source_global_indices: tuple[int, ...],
        source_rgb_sha256: tuple[str, ...],
        track_identity_keys: tuple[tuple[str, ...], ...],
        selected_token_indices: torch.Tensor,
        merged_grid_hw: torch.Tensor,
        request: NativePredictionRequest,
        physical_sidecar: CalvinPhysicalSupervisionSidecar,
        device: torch.device | str,
    ) -> NativePredictiveTarget:
        """Resolve selected exact patch targets after the masked forward."""

        if request.source is not PredictionSource.POSTERIOR or request.evidence is not (
            PredictionEvidence.CURRENT_RANDOM_GRID
        ):
            raise ValueError("current-grid cache requires a posterior random-grid request")
        batch = len(source_global_indices)
        if not (batch == len(source_rgb_sha256) == len(track_identity_keys) == request.batch_size):
            raise ValueError("current-grid source, identity and request batches differ")
        if (
            selected_token_indices.shape != (batch, request.query_count)
            or selected_token_indices.dtype != torch.long
            or merged_grid_hw.shape != (batch, 2)
            or merged_grid_hw.dtype != torch.long
        ):
            raise ValueError("current-grid selected addresses have invalid shapes")
        if selected_token_indices.device != request.route_ids.device or (
            merged_grid_hw.device != request.route_ids.device
        ):
            raise ValueError("current-grid addresses and request must share one device")
        if request.address_width != 2 or (request.horizons != 0).any():
            raise ValueError("current-grid requests require 2D zero-horizon addresses")
        if (request.route_ids != self.contract.route_id).any():
            raise ValueError("current-grid request route differs from cache")
        grid = int(round(self.contract.patch_tokens**0.5))
        expected_grid = torch.full_like(merged_grid_hw, grid)
        if not torch.equal(merged_grid_hw, expected_grid):
            raise ValueError("Qwen and DINO current grids are not exactly identical")
        if (selected_token_indices < 0).any() or (
            selected_token_indices >= self.contract.patch_tokens
        ).any():
            raise ValueError("current-grid selected token lies outside the DINO grid")
        token_rows = torch.div(selected_token_indices, grid, rounding_mode="floor")
        token_columns = selected_token_indices.remainder(grid)
        expected_addresses = torch.stack(
            (
                (token_columns.to(request.addresses.dtype) + 0.5) * (2.0 / grid) - 1.0,
                (token_rows.to(request.addresses.dtype) + 0.5) * (2.0 / grid) - 1.0,
            ),
            dim=-1,
        )
        tolerance = torch.finfo(request.addresses.dtype).eps * 4
        if not torch.allclose(request.addresses, expected_addresses, atol=tolerance, rtol=0):
            raise ValueError("current-grid request addresses differ from selected tokens")
        if not isinstance(physical_sidecar, CalvinPhysicalSupervisionSidecar):
            raise TypeError("current-grid targets require a verified physical sidecar")
        if physical_sidecar.manifest_sha256 != self.contract.physical_sidecar_manifest_sha256:
            raise ContractError("current-grid physical sidecar provenance differs")

        target_device = torch.device(device)
        maximum_tracks = max(len(keys) for keys in track_identity_keys)
        features = torch.zeros(
            batch,
            maximum_tracks,
            request.query_count,
            self.contract.hidden_size,
            dtype=torch.float32,
            device=target_device,
        )
        importance = torch.zeros(
            batch,
            maximum_tracks,
            request.query_count,
            dtype=torch.float32,
            device=target_device,
        )
        target_digest = hashlib.sha256()
        target_digest.update(b"picf-next.current-grid-target-data/v1\0")
        for batch_index, (source_index, source_hash, requested_keys) in enumerate(
            zip(source_global_indices, source_rgb_sha256, track_identity_keys, strict=True)
        ):
            cached_hash, cached_features = self._record(source_index)
            if cached_hash != source_hash:
                raise ContractError("current-grid source RGB differs from cached target")
            frame = physical_sidecar.source_frame(source_index)
            cameras = tuple(camera for camera in frame.cameras if camera.camera_name == "static")
            if len(cameras) != 1 or cameras[0].source_rgb_sha256 != source_hash:
                raise ContractError("current-grid static owner raster differs from source RGB")
            identity_to_owner = {key: index + 1 for index, key in enumerate(frame.identity_keys)}
            if any(key not in identity_to_owner for key in requested_keys):
                raise ContractError("current-grid requested track is absent from physical frame")
            patch = torch.from_numpy(cached_features.copy()).to(
                device=target_device,
                dtype=torch.float32,
            )
            patch = F.layer_norm(patch, (self.contract.hidden_size,))
            selected = selected_token_indices[batch_index].to(target_device)
            selected_features = patch[selected]
            camera = cameras[0]
            owners = torch.from_numpy(camera.owner_index.copy()).to(target_device)
            supervised = torch.from_numpy(camera.owner_supervised.copy()).to(target_device)
            for track_index, key in enumerate(requested_keys):
                visible = ((owners == identity_to_owner[key]) & supervised).float()[None, None]
                resized = F.interpolate(
                    visible,
                    size=(self.contract.input_size, self.contract.input_size),
                    mode="bilinear",
                    align_corners=False,
                )
                support = F.adaptive_avg_pool2d(resized, (grid, grid)).reshape(-1)[selected]
                support = support * request.valid[batch_index].to(target_device)
                valid = support > 0
                features[batch_index, track_index, valid] = selected_features[valid]
                importance[batch_index, track_index] = support.clamp(0, 1)
            target_digest.update(source_index.to_bytes(8, "big", signed=False))
            target_digest.update(bytes.fromhex(source_hash))
            target_digest.update(cached_features.tobytes())
            target_digest.update(camera.owner_index.tobytes())
            target_digest.update(camera.owner_supervised.tobytes())
        source_batch_digest = _sha256_bytes(
            _canonical_json(
                {
                    "source_global_indices": source_global_indices,
                    "source_rgb_sha256": source_rgb_sha256,
                }
            )
        )
        return make_native_predictive_target(
            modality=LINGBOT_PREDICTIVE_MODALITY,
            features=features,
            valid=importance > 0,
            importance=importance,
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=source_batch_digest,
            target_data_digest=target_digest.hexdigest(),
            encoder_digest=self.contract.encoder_digest,
            query_schema_digest=current_grid_query_schema_digest(route_id=self.contract.route_id),
            validity_semantics=LINGBOT_CURRENT_GRID_VALIDITY,
            track_identity_keys=track_identity_keys,
        )

    def current_correction_summary_target_for(
        self,
        *,
        source_global_indices: tuple[int, ...],
        source_static_rgb_sha256: tuple[str, ...],
        track_identity_keys: tuple[tuple[str, ...], ...],
        request: NativePredictionRequest,
        physical_sidecar: CalvinPhysicalSupervisionSidecar,
        minimum_visible_fraction: float,
        device: torch.device | str,
    ) -> NativePredictiveTarget:
        """Pool current evidence after the host forward, never as a model input."""

        valid_phase = (request.source, request.evidence) in {
            (PredictionSource.PRIOR, PredictionEvidence.CURRENT_CORRECTION),
            (PredictionSource.PRIOR, PredictionEvidence.CURRENT_PRIOR),
            (PredictionSource.POSTERIOR, PredictionEvidence.CURRENT_POSTERIOR),
        }
        if not valid_phase:
            raise ValueError("current-summary cache requires an explicit prior/posterior phase")
        batch = len(source_global_indices)
        if not (
            batch == len(source_static_rgb_sha256) == len(track_identity_keys) == request.batch_size
        ):
            raise ValueError("current-correction source, identity and request batches differ")
        if (
            request.query_count != 1
            or (request.horizons != 0).any()
            or (request.route_ids != self.contract.route_id).any()
            or (request.addresses != 0).any()
        ):
            raise ValueError("current-correction requests require one zero-address cache route")
        if not isinstance(physical_sidecar, CalvinPhysicalSupervisionSidecar):
            raise TypeError("current-correction targets require a verified physical sidecar")
        if physical_sidecar.manifest_sha256 != self.contract.physical_sidecar_manifest_sha256:
            raise ContractError("current-correction physical sidecar provenance differs")

        target_device = torch.device(device)
        maximum_tracks = max(len(keys) for keys in track_identity_keys)
        token_features = torch.zeros(
            batch,
            self.contract.patch_tokens,
            self.contract.hidden_size,
            dtype=torch.float32,
            device=target_device,
        )
        track_support = torch.zeros(
            batch,
            self.contract.patch_tokens,
            maximum_tracks,
            dtype=torch.float32,
            device=target_device,
        )
        target_digest = hashlib.sha256()
        target_digest.update(b"picf-next.current-correction-summary-target-data/v1\0")
        if (
            isinstance(minimum_visible_fraction, bool)
            or not isinstance(minimum_visible_fraction, (int, float))
            or not np.isfinite(minimum_visible_fraction)
            or not 0 <= minimum_visible_fraction < 1
        ):
            raise ValueError("minimum visible fraction must lie in [0,1)")
        target_digest.update(float(minimum_visible_fraction).hex().encode("ascii"))
        for batch_index, (source_index, static_hash, requested_keys) in enumerate(
            zip(
                source_global_indices,
                source_static_rgb_sha256,
                track_identity_keys,
                strict=True,
            )
        ):
            cached_hash, cached_features = self._record(source_index)
            if cached_hash != static_hash:
                raise ContractError("current-correction RGB differs from the cached target")
            frame = physical_sidecar.source_frame(source_index)
            cameras = tuple(camera for camera in frame.cameras if camera.camera_name == "static")
            if len(cameras) != 1 or cameras[0].source_rgb_sha256 != static_hash:
                raise ContractError("current-correction owner raster differs from source RGB")
            identity_to_owner = {key: index + 1 for index, key in enumerate(frame.identity_keys)}
            if any(key not in identity_to_owner for key in requested_keys):
                raise ContractError("current-correction requested track is absent from frame")
            token_features[batch_index] = torch.from_numpy(cached_features.copy()).to(
                device=target_device,
                dtype=torch.float32,
            )
            static = cameras[0]
            request_weight = request.valid[batch_index, 0].to(torch.float32)
            pooled_support = _pooled_owner_patch_support(
                owner_index=static.owner_index,
                owner_supervised=static.owner_supervised,
                owner_ids=tuple(identity_to_owner[key] for key in requested_keys),
                input_size=self.contract.input_size,
                patch_tokens=self.contract.patch_tokens,
            )
            track_support[batch_index, :, : len(requested_keys)] = (
                pooled_support.to(target_device) * request_weight
            )
            target_digest.update(source_index.to_bytes(8, "big", signed=False))
            target_digest.update(bytes.fromhex(static_hash))
            target_digest.update(cached_features.tobytes())
            target_digest.update(static.owner_index.tobytes())
            target_digest.update(static.owner_supervised.tobytes())

        source_batch_digest = _sha256_bytes(
            _canonical_json(
                {
                    "source_global_indices": source_global_indices,
                    "source_static_rgb_sha256": source_static_rgb_sha256,
                }
            )
        )
        token_valid = request.valid.expand(-1, self.contract.patch_tokens)
        return make_object_summary_target(
            modality=LINGBOT_PREDICTIVE_MODALITY,
            token_features=token_features,
            track_support=track_support,
            token_valid=token_valid,
            token_footprint=torch.full_like(
                token_valid,
                1.0 / self.contract.patch_tokens,
                dtype=torch.float32,
            ),
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=source_batch_digest,
            target_data_digest=target_digest.hexdigest(),
            encoder_digest=self.contract.encoder_digest,
            query_schema_digest=current_correction_summary_query_schema_digest(
                route_id=self.contract.route_id,
                address_width=request.address_width,
                source=request.source,
                evidence=request.evidence,
            ),
            validity_semantics=LINGBOT_CURRENT_CORRECTION_SUMMARY_VALIDITY,
            track_identity_keys=track_identity_keys,
            minimum_support=minimum_visible_fraction,
        )

    def omitted_static_summary_target_for(
        self,
        *,
        source_global_indices: tuple[int, ...],
        source_static_rgb_sha256: tuple[str, ...],
        source_gripper_rgb_sha256: tuple[str, ...],
        track_identity_keys: tuple[tuple[str, ...], ...],
        request: NativePredictionRequest,
        omission: QwenWholeViewOmission,
        physical_sidecar: CalvinPhysicalSupervisionSidecar,
        device: torch.device | str,
    ) -> NativePredictiveTarget:
        """Pool omitted static targets only for objects visible in the source view."""

        if (
            request.source is not PredictionSource.POSTERIOR
            or request.evidence is not PredictionEvidence.OMITTED_MODALITY
        ):
            raise ValueError("omitted-static cache requires posterior omitted-modality evidence")
        if not isinstance(omission, QwenWholeViewOmission):
            raise TypeError("omitted-static target requires a QwenWholeViewOmission")
        batch = len(source_global_indices)
        if not (
            batch
            == len(source_static_rgb_sha256)
            == len(source_gripper_rgb_sha256)
            == len(track_identity_keys)
            == request.batch_size
        ):
            raise ValueError("omitted-static source, identity and request batches differ")
        if (
            request.query_count != 1
            or (request.horizons != 0).any()
            or (request.route_ids != self.contract.route_id).any()
            or (request.addresses != 0).any()
        ):
            raise ValueError("omitted-static requests require one zero-address cache route")
        if omission.image_valid.device != request.valid.device:
            raise ValueError("omitted-static request and omission plan must share one device")
        static_view_indices = tuple(
            index
            for index, slot in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS)
            if slot.physical_camera_name == "static"
        )
        gripper_view_indices = tuple(
            index
            for index, slot in enumerate(LINGBOT_CALVIN_CAMERA_SLOTS)
            if slot.physical_camera_name == "gripper"
        )
        if static_view_indices != (0,) or gripper_view_indices != (1,):
            raise RuntimeError("the frozen LingBot CALVIN camera-slot ABI changed")
        expected_image_valid = torch.tensor(
            [slot.valid for slot in LINGBOT_CALVIN_CAMERA_SLOTS],
            dtype=torch.bool,
            device=omission.image_valid.device,
        ).expand(batch, -1)
        if (
            omission.omitted_view_index != static_view_indices[0]
            or omission.image_valid.shape != expected_image_valid.shape
        ):
            raise ValueError(
                "CALVIN omitted-static targets require static slot 0 of the official "
                "three-slot camera ABI"
            )
        if not torch.equal(omission.image_valid, expected_image_valid):
            raise ValueError(
                "CALVIN omitted-static image availability differs from the official camera ABI"
            )
        if not torch.equal(request.valid, omission.source_valid[:, None]):
            raise ValueError("omitted-static request validity differs from the omission plan")
        if not omission.source_image_valid[request.valid[:, 0], gripper_view_indices[0]].all():
            raise ValueError("omitted-static samples require an available gripper source view")
        if not isinstance(physical_sidecar, CalvinPhysicalSupervisionSidecar):
            raise TypeError("omitted-static targets require a verified physical sidecar")
        if physical_sidecar.manifest_sha256 != self.contract.physical_sidecar_manifest_sha256:
            raise ContractError("omitted-static physical sidecar provenance differs")

        target_device = torch.device(device)
        maximum_tracks = max(len(keys) for keys in track_identity_keys)
        token_features = torch.zeros(
            batch,
            self.contract.patch_tokens,
            self.contract.hidden_size,
            dtype=torch.float32,
            device=target_device,
        )
        track_support = torch.zeros(
            batch,
            self.contract.patch_tokens,
            maximum_tracks,
            dtype=torch.float32,
            device=target_device,
        )
        target_digest = hashlib.sha256()
        target_digest.update(b"picf-next.omitted-static-summary-target-data/v1\0")
        grid = int(round(self.contract.patch_tokens**0.5))
        for batch_index, (source_index, static_hash, gripper_hash, requested_keys) in enumerate(
            zip(
                source_global_indices,
                source_static_rgb_sha256,
                source_gripper_rgb_sha256,
                track_identity_keys,
                strict=True,
            )
        ):
            cached_hash, cached_features = self._record(source_index)
            if cached_hash != static_hash:
                raise ContractError("omitted-static RGB differs from the cached target")
            frame = physical_sidecar.source_frame(source_index)
            cameras = {camera.camera_name: camera for camera in frame.cameras}
            if len(frame.cameras) != 2 or set(cameras) != {"static", "gripper"}:
                raise ContractError("omitted-static target requires exact static/gripper cameras")
            static = cameras["static"]
            gripper = cameras["gripper"]
            if static.source_rgb_sha256 != static_hash or gripper.source_rgb_sha256 != gripper_hash:
                raise ContractError("omitted-static owner rasters differ from source RGB")
            identity_to_owner = {key: index + 1 for index, key in enumerate(frame.identity_keys)}
            if any(key not in identity_to_owner for key in requested_keys):
                raise ContractError("omitted-static requested track is absent from physical frame")
            token_features[batch_index] = torch.from_numpy(cached_features.copy()).to(
                device=target_device,
                dtype=torch.float32,
            )
            static_owner = torch.from_numpy(static.owner_index.copy()).to(target_device)
            static_supervised = torch.from_numpy(static.owner_supervised.copy()).to(target_device)
            gripper_owner = torch.from_numpy(gripper.owner_index.copy()).to(target_device)
            gripper_supervised = torch.from_numpy(gripper.owner_supervised.copy()).to(target_device)
            request_weight = request.valid[batch_index, 0].to(torch.float32)
            for track_index, key in enumerate(requested_keys):
                owner_index = identity_to_owner[key]
                visible_gripper = ((gripper_owner == owner_index) & gripper_supervised).to(
                    torch.float32
                )
                # Preserve absolute source evidence. A one-pixel wrist-camera
                # sliver must not authorize a full omitted-view object target.
                source_weight = visible_gripper.mean() * request_weight
                visible_static = ((static_owner == owner_index) & static_supervised).float()[
                    None, None
                ]
                resized = F.interpolate(
                    visible_static,
                    size=(self.contract.input_size, self.contract.input_size),
                    mode="bilinear",
                    align_corners=False,
                )
                track_support[batch_index, :, track_index] = (
                    F.adaptive_avg_pool2d(resized, (grid, grid)).reshape(-1).clamp(0, 1)
                    * source_weight
                )
            target_digest.update(source_index.to_bytes(8, "big", signed=False))
            target_digest.update(bytes.fromhex(static_hash))
            target_digest.update(bytes.fromhex(gripper_hash))
            target_digest.update(cached_features.tobytes())
            for camera in (static, gripper):
                target_digest.update(camera.owner_index.tobytes())
                target_digest.update(camera.owner_supervised.tobytes())

        source_batch_digest = _sha256_bytes(
            _canonical_json(
                {
                    "source_global_indices": source_global_indices,
                    "source_gripper_rgb_sha256": source_gripper_rgb_sha256,
                    "source_static_rgb_sha256": source_static_rgb_sha256,
                }
            )
        )
        token_valid = request.valid.expand(-1, self.contract.patch_tokens)
        return make_object_summary_target(
            modality=LINGBOT_PREDICTIVE_MODALITY,
            token_features=token_features,
            track_support=track_support,
            token_valid=token_valid,
            token_footprint=torch.full_like(
                token_valid,
                1.0 / self.contract.patch_tokens,
                dtype=torch.float32,
            ),
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=source_batch_digest,
            target_data_digest=target_digest.hexdigest(),
            encoder_digest=self.contract.encoder_digest,
            query_schema_digest=omitted_static_summary_query_schema_digest(
                route_id=self.contract.route_id
            ),
            validity_semantics=LINGBOT_OMITTED_STATIC_SUMMARY_VALIDITY,
            track_identity_keys=track_identity_keys,
        )
