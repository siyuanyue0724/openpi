"""Immutable object-level DINO-video targets for the LingBot-native objective.

The cache contains loss-side summaries only.  It never enters model inputs or
posterior state.  Every record is bound to a source frame, a future frame, the
task-independent physical sidecar, and the exact frozen teacher contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveTarget,
    TargetEncoderMode,
    make_native_predictive_target,
)

LINGBOT_PREDICTIVE_CACHE_SCHEMA = "picf-next.lingbot-predictive-object-cache/v3"
LINGBOT_PREDICTIVE_TARGET_SPACE = "dino_video"
LINGBOT_PREDICTIVE_MODALITY = "vision"
LINGBOT_PREDICTIVE_EFFECTIVE_FPS = "source-fps-divided-by-frame-horizon/v1"
LINGBOT_PREDICTIVE_POOLING = (
    "layernorm-patch/features-weighted-by-task-independent-visible-owner-area/v1"
)
LINGBOT_PREDICTIVE_COVERAGE = "frozen-stream-plan-exact-query-set/v2"
_PAIR_KEYS_SCHEMA = b"picf-next.lingbot-predictive-pair-keys/v1\0"
_MANIFEST_MAXIMUM_BYTES = 32 * 1024 * 1024
_SHARD_OVERHEAD_BYTES = 16 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite real number")
    measured = float(value)
    if not np.isfinite(measured):
        raise ContractError(f"{name} must be a finite real number")
    return measured


def predictive_effective_fps(
    horizons: torch.Tensor,
    *,
    source_fps: float,
) -> torch.Tensor:
    """Match LingBot's released ``dataset_fps / frame_gap`` teacher contract."""

    measured_fps = _finite_float(source_fps, "predictive source FPS")
    if measured_fps <= 0:
        raise ContractError("predictive source FPS must be positive")
    if horizons.ndim != 1 or horizons.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("predictive horizons must be one integer vector")
    if (horizons <= 0).any():
        raise ValueError("predictive horizons must be positive")
    return measured_fps / horizons.to(dtype=torch.float32)


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


def native_predictive_query_schema_digest(
    *,
    target_space: str,
    route_id: int,
    horizons: Sequence[int],
) -> str:
    """Hash source-known future-target semantics independently of labels.

    One immutable future-evidence cache serves both the full-graph posterior
    predictor and the recursively rolled prior predictor. The learned source
    is therefore a request property, not part of target-data identity.
    """

    if not isinstance(target_space, str) or not target_space:
        raise ValueError("predictive target space must be non-empty")
    if isinstance(route_id, bool) or not isinstance(route_id, int) or route_id < 0:
        raise ValueError("predictive route ID must be a non-negative integer")
    frozen_horizons = tuple(horizons)
    if (
        not frozen_horizons
        or tuple(sorted(set(frozen_horizons))) != frozen_horizons
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in frozen_horizons
        )
    ):
        raise ValueError("predictive horizons must be sorted unique positive integers")
    return _sha256_bytes(
        _canonical_json(
            {
                "evidence": "future",
                "horizons": frozen_horizons,
                "route_id": route_id,
                "supported_sources": ("posterior", "prior"),
                "target_space": target_space,
            }
        )
    )


class _PredictivePairDigest:
    """Streaming canonical key digest with strict source-major ordering."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(_PAIR_KEYS_SCHEMA)
        self._previous: tuple[int, int] | None = None
        self.count = 0

    def add(self, pair: tuple[int, int]) -> None:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or isinstance(pair[0], bool)
            or not isinstance(pair[0], int)
            or pair[0] < 0
            or isinstance(pair[1], bool)
            or not isinstance(pair[1], int)
            or pair[1] <= 0
        ):
            raise ContractError(
                "predictive cache pair must be a non-negative source and positive horizon"
            )
        if self._previous is not None and pair <= self._previous:
            raise ContractError("predictive cache pairs must be source-major sorted and unique")
        self._digest.update(pair[0].to_bytes(8, byteorder="big", signed=False))
        self._digest.update(pair[1].to_bytes(8, byteorder="big", signed=False))
        self._previous = pair
        self.count += 1

    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(self.count.to_bytes(8, byteorder="big", signed=False))
        return digest.hexdigest()


def native_predictive_pair_keys_digest(pairs: Iterable[tuple[int, int]]) -> str:
    """Hash one exact ordered cache-key set without retaining target content."""

    accumulator = _PredictivePairDigest()
    for pair in pairs:
        accumulator.add(pair)
    if accumulator.count == 0:
        raise ContractError("predictive pair coverage cannot be empty")
    return accumulator.hexdigest()


def native_predictive_coverage_digest(
    *,
    dataset_tree_sha256: str,
    stream_plan_sha256: str,
    temporal_estimator_sha256: str,
    pair_keys_sha256: str,
    expected_record_count: int,
    horizons: Sequence[int],
) -> str:
    """Bind exact sparse coverage to data, stream plan and temporal estimator."""

    dataset_digest = _sha256(dataset_tree_sha256, "dataset tree sha256")
    plan_digest = _sha256(stream_plan_sha256, "stream plan sha256")
    temporal_digest = _sha256(temporal_estimator_sha256, "temporal estimator sha256")
    pair_digest = _sha256(pair_keys_sha256, "predictive pair keys sha256")
    record_count = _positive_int(expected_record_count, "expected predictive record count")
    frozen_horizons = tuple(horizons)
    native_predictive_query_schema_digest(
        target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
        route_id=0,
        horizons=frozen_horizons,
    )
    return _sha256_bytes(
        _canonical_json(
            {
                "coverage": LINGBOT_PREDICTIVE_COVERAGE,
                "dataset_tree_sha256": dataset_digest,
                "expected_record_count": record_count,
                "horizons": frozen_horizons,
                "pair_keys_sha256": pair_digest,
                "stream_plan_sha256": plan_digest,
                "target_space": LINGBOT_PREDICTIVE_TARGET_SPACE,
                "temporal_estimator_sha256": temporal_digest,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class PredictiveCacheContract:
    dataset_id: str
    dataset_revision: str
    split_name: str
    dataset_tree_sha256: str
    physical_sidecar_manifest_sha256: str
    lingbot_source_commit: str
    lingbot_checkpoint_revision: str
    teacher_config_sha256: str
    teacher_checkpoint_sha256: str
    query_schema_sha256: str
    horizons: tuple[int, ...]
    stream_plan_sha256: str
    temporal_estimator_sha256: str
    pair_keys_sha256: str
    coverage_sha256: str
    expected_record_count: int
    hidden_size: int = 1024
    input_size: int = 256
    patch_tokens: int = 256
    route_id: int = 0
    camera_name: str = "static"
    attention_mode: str = "flex_block_causal"
    use_warmup_frame: bool = True
    source_fps: float = 30.0
    effective_fps_semantics: str = LINGBOT_PREDICTIVE_EFFECTIVE_FPS
    minimum_visible_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "dataset_revision",
            "split_name",
            "lingbot_source_commit",
            "lingbot_checkpoint_revision",
            "camera_name",
            "attention_mode",
            "effective_fps_semantics",
        ):
            _text(getattr(self, name), name)
        for name in (
            "dataset_tree_sha256",
            "physical_sidecar_manifest_sha256",
            "teacher_config_sha256",
            "teacher_checkpoint_sha256",
            "query_schema_sha256",
            "stream_plan_sha256",
            "temporal_estimator_sha256",
            "pair_keys_sha256",
            "coverage_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("hidden_size", "input_size", "patch_tokens"):
            _positive_int(getattr(self, name), name)
        _positive_int(self.expected_record_count, "expected_record_count")
        _nonnegative_int(self.route_id, "route_id")
        if self.camera_name != "static":
            raise ContractError("the frozen LingBot DINO-video target uses the static camera")
        if self.attention_mode != "flex_block_causal" or self.use_warmup_frame is not True:
            raise ContractError("the DINO-video causal target contract changed")
        if self.input_size != 256 or self.patch_tokens != 256 or self.hidden_size != 1024:
            raise ContractError("the released LingBot DINO-video target geometry changed")
        if not np.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ContractError("predictive target source FPS must be finite and positive")
        if self.effective_fps_semantics != LINGBOT_PREDICTIVE_EFFECTIVE_FPS:
            raise ContractError("predictive target effective-FPS semantics changed")
        if (
            not np.isfinite(self.minimum_visible_fraction)
            or not 0 <= self.minimum_visible_fraction < 1
        ):
            raise ContractError("minimum visible fraction must lie in [0,1)")
        expected_query = native_predictive_query_schema_digest(
            target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
            route_id=self.route_id,
            horizons=self.horizons,
        )
        if self.query_schema_sha256 != expected_query:
            raise ContractError("predictive query schema digest differs from its semantics")
        expected_coverage = native_predictive_coverage_digest(
            dataset_tree_sha256=self.dataset_tree_sha256,
            stream_plan_sha256=self.stream_plan_sha256,
            temporal_estimator_sha256=self.temporal_estimator_sha256,
            pair_keys_sha256=self.pair_keys_sha256,
            expected_record_count=self.expected_record_count,
            horizons=self.horizons,
        )
        if self.coverage_sha256 != expected_coverage:
            raise ContractError("predictive coverage digest differs from its semantics")

    @property
    def encoder_payload(self) -> dict[str, object]:
        return {
            "attention_mode": self.attention_mode,
            "checkpoint_revision": self.lingbot_checkpoint_revision,
            "checkpoint_sha256": self.teacher_checkpoint_sha256,
            "config_sha256": self.teacher_config_sha256,
            "effective_fps_semantics": self.effective_fps_semantics,
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "lingbot_source_commit": self.lingbot_source_commit,
            "patch_tokens": self.patch_tokens,
            "source_fps": self.source_fps,
            "use_warmup_frame": self.use_warmup_frame,
        }

    @property
    def encoder_digest(self) -> str:
        return _sha256_bytes(_canonical_json(self.encoder_payload))


@dataclass(frozen=True, slots=True)
class PredictiveObjectCacheRecord:
    source_global_index: int
    target_global_index: int
    horizon: int
    source_rgb_sha256: str
    target_rgb_sha256: str
    identity_keys: tuple[str, ...]
    features: NDArray[np.float16]
    importance: NDArray[np.float32]

    def __post_init__(self) -> None:
        for name in ("source_global_index", "target_global_index", "horizon"):
            _nonnegative_int(getattr(self, name), name)
        if self.horizon <= 0 or self.target_global_index != self.source_global_index + self.horizon:
            raise ContractError("predictive cache target index must equal source plus horizon")
        _sha256(self.source_rgb_sha256, "source RGB sha256")
        _sha256(self.target_rgb_sha256, "target RGB sha256")
        if (
            not isinstance(self.identity_keys, tuple)
            or not self.identity_keys
            or len(set(self.identity_keys)) != len(self.identity_keys)
            or any(not isinstance(key, str) or not key for key in self.identity_keys)
        ):
            raise ContractError("predictive cache identities must be non-empty and unique")
        if (
            not isinstance(self.features, np.ndarray)
            or self.features.dtype != np.float16
            or self.features.ndim != 2
            or self.features.shape[0] != len(self.identity_keys)
            or not np.isfinite(self.features).all()
        ):
            raise ContractError("predictive cache features must be finite float16 object rows")
        if (
            not isinstance(self.importance, np.ndarray)
            or self.importance.dtype != np.float32
            or self.importance.shape != (len(self.identity_keys),)
            or not np.isfinite(self.importance).all()
            or ((self.importance < 0) | (self.importance > 1)).any()
        ):
            raise ContractError("predictive cache importance must be float32 in [0,1]")
        if (self.features[self.importance == 0] != 0).any():
            raise ContractError("unsupported predictive objects must have zero features")


@dataclass(frozen=True, slots=True)
class _PredictiveShardMetadata:
    path: str
    sha256: str
    row_count: int
    object_count: int
    first_source_global_index: int
    first_horizon: int
    last_source_global_index: int
    last_horizon: int


@dataclass(frozen=True, slots=True)
class _LoadedPredictiveShard:
    source_global_indices: NDArray[np.int64]
    target_global_indices: NDArray[np.int64]
    horizons: NDArray[np.int64]
    source_rgb_sha256: NDArray[np.str_]
    target_rgb_sha256: NDArray[np.str_]
    frame_offsets: NDArray[np.int64]
    identity_keys: NDArray[np.str_]
    features: NDArray[np.float16]
    importance: NDArray[np.float32]


def pool_dino_object_summaries(
    patch_features: torch.Tensor,
    *,
    owner_index: NDArray[np.uint8],
    owner_supervised: NDArray[np.bool_],
    identity_keys: tuple[str, ...],
    minimum_visible_fraction: float,
    input_size: int = 256,
) -> tuple[NDArray[np.float16], NDArray[np.float32]]:
    """Pool detached DINO patches using task-independent visible-owner area."""

    if patch_features.ndim != 2 or not patch_features.is_floating_point():
        raise ValueError("DINO patch features must be [tokens,width] floating point")
    if patch_features.requires_grad or not torch.isfinite(patch_features).all():
        raise ValueError("DINO cache features must be finite and detached")
    token_count, width = patch_features.shape
    grid = int(round(token_count**0.5))
    if grid * grid != token_count:
        raise ValueError("DINO patch tokens must form one square spatial grid")
    if (
        owner_index.dtype != np.uint8
        or owner_supervised.dtype != np.bool_
        or owner_index.shape != owner_supervised.shape
        or owner_index.ndim != 2
    ):
        raise ValueError("visible-owner raster and supervision must share one image plane")
    if not identity_keys or int(owner_index.max(initial=0)) > len(identity_keys):
        raise ValueError("visible-owner raster references an absent identity")
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
        raise ValueError("DINO input size must be a positive integer")
    if not np.isfinite(minimum_visible_fraction) or not 0 <= minimum_visible_fraction < 1:
        raise ValueError("minimum visible fraction must lie in [0,1)")

    device = patch_features.device
    supervised = torch.from_numpy(owner_supervised.copy()).to(device=device)
    owners = torch.from_numpy(owner_index.copy()).to(device=device)
    normalized = F.layer_norm(patch_features.detach().float(), (width,))
    summaries = torch.zeros(len(identity_keys), width, device=device, dtype=torch.float32)
    importance = torch.zeros(len(identity_keys), device=device, dtype=torch.float32)
    for object_index in range(len(identity_keys)):
        visible = ((owners == object_index + 1) & supervised).float()[None, None]
        resized_support = F.interpolate(
            visible,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        token_support = F.adaptive_avg_pool2d(resized_support, (grid, grid)).reshape(-1)
        weights = token_support / token_count
        mass = weights.sum()
        if float(mass) > minimum_visible_fraction:
            summaries[object_index] = (weights[:, None] * normalized).sum(dim=0) / mass
            importance[object_index] = mass.clamp_max(1)
    return (
        summaries.cpu().numpy().astype(np.float16, copy=False),
        importance.cpu().numpy().astype(np.float32, copy=False),
    )


class LingBotPredictiveTargetCache:
    """Bounded, content-addressed reader for frozen object-level targets."""

    def __init__(
        self,
        *,
        root: Path,
        manifest_sha256: str,
        contract: PredictiveCacheContract,
        shards: tuple[_PredictiveShardMetadata, ...],
        locator: dict[tuple[int, int], tuple[int, int]],
        memory_capacity: int,
    ) -> None:
        self.root = root
        self.manifest_sha256 = manifest_sha256
        self.contract = contract
        self.shards = shards
        self.locator = locator
        self.memory_capacity = memory_capacity
        self._loaded: OrderedDict[int, _LoadedPredictiveShard] = OrderedDict()

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        manifest_sha256: str,
        dataset_tree_sha256: str,
        physical_sidecar_manifest_sha256: str,
        encoder_digest: str,
        query_schema_sha256: str,
        coverage_sha256: str,
        memory_capacity: int = 2,
    ) -> LingBotPredictiveTargetCache:
        expected_manifest = _sha256(manifest_sha256, "cache manifest sha256")
        expected_dataset = _sha256(dataset_tree_sha256, "dataset tree sha256")
        expected_sidecar = _sha256(
            physical_sidecar_manifest_sha256,
            "physical sidecar manifest sha256",
        )
        expected_encoder = _sha256(encoder_digest, "predictive encoder digest")
        expected_query = _sha256(query_schema_sha256, "predictive query schema sha256")
        expected_coverage = _sha256(coverage_sha256, "predictive coverage sha256")
        memory_capacity = _positive_int(memory_capacity, "predictive cache memory capacity")
        resolved_root = Path(root).resolve()
        manifest_bytes = read_sha256_verified_file_beneath(
            resolved_root,
            "manifest.json",
            expected_sha256=expected_manifest,
            maximum_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
        try:
            raw = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("predictive cache manifest is not valid JSON") from error
        payload = _exact(
            raw,
            "predictive cache manifest",
            {
                "complete",
                "contract",
                "expected_records",
                "record_count",
                "schema",
                "shards",
                "target_space",
            },
        )
        if payload["schema"] != LINGBOT_PREDICTIVE_CACHE_SCHEMA:
            raise ContractError("predictive cache schema changed")
        if payload["complete"] is not True:
            raise ContractError("predictive cache is incomplete")
        if payload["target_space"] != LINGBOT_PREDICTIVE_TARGET_SPACE:
            raise ContractError("predictive cache target space changed")
        contract = cls._parse_contract(payload["contract"])
        if contract.dataset_tree_sha256 != expected_dataset:
            raise ContractError("predictive cache belongs to another dataset tree")
        if contract.physical_sidecar_manifest_sha256 != expected_sidecar:
            raise ContractError("predictive cache belongs to another physical sidecar")
        if contract.encoder_digest != expected_encoder:
            raise ContractError("predictive cache belongs to another teacher encoder")
        if contract.query_schema_sha256 != expected_query:
            raise ContractError("predictive cache query schema differs")
        if contract.coverage_sha256 != expected_coverage:
            raise ContractError("predictive cache coverage differs")

        raw_shards = payload["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ContractError("predictive cache requires at least one shard")
        shards = tuple(
            cls._parse_shard(value, index=index) for index, value in enumerate(raw_shards)
        )
        row_count = _positive_int(payload["record_count"], "predictive record count")
        expected_records = _positive_int(payload["expected_records"], "expected predictive records")
        if (
            row_count != expected_records
            or expected_records != contract.expected_record_count
            or sum(shard.row_count for shard in shards) != row_count
        ):
            raise ContractError("predictive cache did not publish every expected record")

        locator: dict[tuple[int, int], tuple[int, int]] = {}
        pair_digest = _PredictivePairDigest()
        for shard_index, metadata in enumerate(shards):
            loaded = cls._read_and_validate_shard(
                resolved_root,
                metadata,
                hidden_size=contract.hidden_size,
                allowed_horizons=contract.horizons,
            )
            keys = tuple(
                zip(
                    loaded.source_global_indices.tolist(),
                    loaded.horizons.tolist(),
                    strict=True,
                )
            )
            if keys[0] != (metadata.first_source_global_index, metadata.first_horizon) or keys[
                -1
            ] != (metadata.last_source_global_index, metadata.last_horizon):
                raise ContractError("predictive shard bounds differ from its rows")
            for row, key in enumerate(keys):
                pair_digest.add(key)
                if key in locator:
                    raise ContractError("predictive cache contains duplicate source/horizon keys")
                locator[key] = (shard_index, row)
        if (
            pair_digest.count != contract.expected_record_count
            or pair_digest.hexdigest() != contract.pair_keys_sha256
        ):
            raise ContractError("predictive cache keys differ from the frozen training plan")
        return cls(
            root=resolved_root,
            manifest_sha256=expected_manifest,
            contract=contract,
            shards=shards,
            locator=locator,
            memory_capacity=memory_capacity,
        )

    @staticmethod
    def _parse_contract(raw: object) -> PredictiveCacheContract:
        fields = {field.name for field in PredictiveCacheContract.__dataclass_fields__.values()}
        payload = _exact(raw, "predictive cache contract", fields)
        raw_horizons = payload["horizons"]
        if not isinstance(raw_horizons, list):
            raise ContractError("predictive cache horizons must be a list")
        horizons = tuple(
            _positive_int(value, f"predictive horizon[{index}]")
            for index, value in enumerate(raw_horizons)
        )
        source_fps = _finite_float(payload["source_fps"], "source_fps")
        minimum_visible_fraction = _finite_float(
            payload["minimum_visible_fraction"], "minimum_visible_fraction"
        )
        use_warmup_frame = payload["use_warmup_frame"]
        if not isinstance(use_warmup_frame, bool):
            raise ContractError("use_warmup_frame must be boolean")
        return PredictiveCacheContract(
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
            query_schema_sha256=_sha256(payload["query_schema_sha256"], "query_schema_sha256"),
            horizons=horizons,
            stream_plan_sha256=_sha256(payload["stream_plan_sha256"], "stream_plan_sha256"),
            temporal_estimator_sha256=_sha256(
                payload["temporal_estimator_sha256"], "temporal_estimator_sha256"
            ),
            pair_keys_sha256=_sha256(payload["pair_keys_sha256"], "pair_keys_sha256"),
            coverage_sha256=_sha256(payload["coverage_sha256"], "coverage_sha256"),
            expected_record_count=_positive_int(
                payload["expected_record_count"], "expected_record_count"
            ),
            hidden_size=_positive_int(payload["hidden_size"], "hidden_size"),
            input_size=_positive_int(payload["input_size"], "input_size"),
            patch_tokens=_positive_int(payload["patch_tokens"], "patch_tokens"),
            route_id=_nonnegative_int(payload["route_id"], "route_id"),
            camera_name=_text(payload["camera_name"], "camera_name"),
            attention_mode=_text(payload["attention_mode"], "attention_mode"),
            use_warmup_frame=use_warmup_frame,
            source_fps=source_fps,
            effective_fps_semantics=_text(
                payload["effective_fps_semantics"], "effective_fps_semantics"
            ),
            minimum_visible_fraction=minimum_visible_fraction,
        )

    @staticmethod
    def _parse_shard(raw: object, *, index: int) -> _PredictiveShardMetadata:
        fields = {field.name for field in _PredictiveShardMetadata.__dataclass_fields__.values()}
        payload = _exact(raw, f"predictive shard[{index}]", fields)
        return _PredictiveShardMetadata(
            path=_relative_path(payload["path"], "predictive shard path"),
            sha256=_sha256(payload["sha256"], "predictive shard sha256"),
            row_count=_positive_int(payload["row_count"], "predictive shard row count"),
            object_count=_positive_int(payload["object_count"], "predictive shard object count"),
            first_source_global_index=_nonnegative_int(
                payload["first_source_global_index"], "predictive shard first source"
            ),
            first_horizon=_positive_int(payload["first_horizon"], "predictive shard first horizon"),
            last_source_global_index=_nonnegative_int(
                payload["last_source_global_index"], "predictive shard last source"
            ),
            last_horizon=_positive_int(payload["last_horizon"], "predictive shard last horizon"),
        )

    @staticmethod
    def _read_and_validate_shard(
        root: Path,
        metadata: _PredictiveShardMetadata,
        *,
        hidden_size: int,
        allowed_horizons: tuple[int, ...],
    ) -> _LoadedPredictiveShard:
        maximum_bytes = metadata.object_count * hidden_size * 2 + _SHARD_OVERHEAD_BYTES
        payload = read_sha256_verified_file_beneath(
            root,
            metadata.path,
            expected_sha256=metadata.sha256,
            maximum_bytes=maximum_bytes,
        )
        expected_arrays = {
            "features",
            "frame_offsets",
            "horizons",
            "identity_keys",
            "importance",
            "source_global_indices",
            "source_rgb_sha256",
            "target_global_indices",
            "target_rgb_sha256",
        }
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                if set(archive.files) != expected_arrays:
                    raise ContractError("predictive shard arrays differ from schema")
                arrays = {name: archive[name].copy() for name in archive.files}
        except (OSError, ValueError) as error:
            raise ContractError("predictive cache shard is not a safe NPZ archive") from error
        rows = metadata.row_count
        objects = metadata.object_count
        source = arrays["source_global_indices"]
        target = arrays["target_global_indices"]
        horizons = arrays["horizons"]
        offsets = arrays["frame_offsets"]
        identities = arrays["identity_keys"]
        features = arrays["features"]
        importance = arrays["importance"]
        source_hashes = arrays["source_rgb_sha256"]
        target_hashes = arrays["target_rgb_sha256"]
        if (
            source.dtype != np.int64
            or source.shape != (rows,)
            or target.dtype != np.int64
            or target.shape != (rows,)
            or horizons.dtype != np.int64
            or horizons.shape != (rows,)
            or offsets.dtype != np.int64
            or offsets.shape != (rows + 1,)
            or not np.issubdtype(identities.dtype, np.str_)
            or identities.shape != (objects,)
            or features.dtype != np.float16
            or features.shape != (objects, hidden_size)
            or importance.dtype != np.float32
            or importance.shape != (objects,)
            or not np.issubdtype(source_hashes.dtype, np.str_)
            or source_hashes.shape != (rows,)
            or not np.issubdtype(target_hashes.dtype, np.str_)
            or target_hashes.shape != (rows,)
        ):
            raise ContractError("predictive shard shapes or dtypes changed")
        keys = tuple(zip(source.tolist(), horizons.tolist(), strict=True))
        if (
            keys != tuple(sorted(keys))
            or len(set(keys)) != rows
            or not np.array_equal(target, source + horizons)
            or any(int(value) not in allowed_horizons for value in horizons)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != objects
            or (np.diff(offsets) <= 0).any()
            or not np.isfinite(features).all()
            or not np.isfinite(importance).all()
            or ((importance < 0) | (importance > 1)).any()
            or (features[importance == 0] != 0).any()
        ):
            raise ContractError("predictive shard content violates its contract")
        for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            keys_for_row = identities[int(start) : int(stop)].tolist()
            if len(set(keys_for_row)) != len(keys_for_row):
                raise ContractError("predictive shard repeats an identity within one record")
            for digest in (str(source_hashes[row]), str(target_hashes[row])):
                _sha256(digest, "predictive shard RGB sha256")
        return _LoadedPredictiveShard(
            source_global_indices=source,
            target_global_indices=target,
            horizons=horizons,
            source_rgb_sha256=source_hashes,
            target_rgb_sha256=target_hashes,
            frame_offsets=offsets,
            identity_keys=identities,
            features=features,
            importance=importance,
        )

    def _load_shard(self, shard_index: int) -> _LoadedPredictiveShard:
        cached = self._loaded.get(shard_index)
        if cached is not None:
            self._loaded.move_to_end(shard_index)
            return cached
        loaded = self._read_and_validate_shard(
            self.root,
            self.shards[shard_index],
            hidden_size=self.contract.hidden_size,
            allowed_horizons=self.contract.horizons,
        )
        self._loaded[shard_index] = loaded
        while len(self._loaded) > self.memory_capacity:
            self._loaded.popitem(last=False)
        return loaded

    def iter_records(self) -> Iterable[PredictiveObjectCacheRecord]:
        """Yield validated defensive copies in canonical source-major order."""

        for shard_index, metadata in enumerate(self.shards):
            loaded = self._load_shard(shard_index)
            for row in range(metadata.row_count):
                start = int(loaded.frame_offsets[row])
                stop = int(loaded.frame_offsets[row + 1])
                yield PredictiveObjectCacheRecord(
                    source_global_index=int(loaded.source_global_indices[row]),
                    target_global_index=int(loaded.target_global_indices[row]),
                    horizon=int(loaded.horizons[row]),
                    source_rgb_sha256=str(loaded.source_rgb_sha256[row]),
                    target_rgb_sha256=str(loaded.target_rgb_sha256[row]),
                    identity_keys=tuple(str(value) for value in loaded.identity_keys[start:stop]),
                    features=loaded.features[start:stop].copy(),
                    importance=loaded.importance[start:stop].copy(),
                )

    def record_for(
        self,
        *,
        source_global_index: int,
        horizon: int,
    ) -> PredictiveObjectCacheRecord | None:
        """Return one validated defensive record, or ``None`` outside coverage."""

        source = _nonnegative_int(source_global_index, "predictive source global index")
        requested_horizon = _positive_int(horizon, "predictive horizon")
        location = self.locator.get((source, requested_horizon))
        if location is None:
            return None
        shard_index, row = location
        loaded = self._load_shard(shard_index)
        start = int(loaded.frame_offsets[row])
        stop = int(loaded.frame_offsets[row + 1])
        return PredictiveObjectCacheRecord(
            source_global_index=int(loaded.source_global_indices[row]),
            target_global_index=int(loaded.target_global_indices[row]),
            horizon=int(loaded.horizons[row]),
            source_rgb_sha256=str(loaded.source_rgb_sha256[row]),
            target_rgb_sha256=str(loaded.target_rgb_sha256[row]),
            identity_keys=tuple(str(value) for value in loaded.identity_keys[start:stop]),
            features=loaded.features[start:stop].copy(),
            importance=loaded.importance[start:stop].copy(),
        )

    def has_supported_target(self, *, source_global_index: int, horizon: int) -> bool:
        """Return whether one cached future row carries positive visible-object mass."""

        source = _nonnegative_int(source_global_index, "predictive source global index")
        requested_horizon = _positive_int(horizon, "predictive horizon")
        location = self.locator.get((source, requested_horizon))
        if location is None:
            return False
        shard_index, row = location
        loaded = self._load_shard(shard_index)
        start = int(loaded.frame_offsets[row])
        stop = int(loaded.frame_offsets[row + 1])
        return bool((loaded.importance[start:stop] > 0).any())

    def target_for(
        self,
        *,
        source_global_indices: Sequence[int],
        source_rgb_sha256: Sequence[str],
        track_identity_keys: tuple[tuple[str, ...], ...],
        request: NativePredictionRequest,
        device: torch.device | str,
    ) -> NativePredictiveTarget:
        """Align immutable cache rows to structural tracks after host forward."""

        sources = tuple(source_global_indices)
        source_hashes = tuple(source_rgb_sha256)
        batch = request.batch_size
        if request.evidence != PredictionEvidence.FUTURE or request.source not in (
            PredictionSource.POSTERIOR,
            PredictionSource.PRIOR,
        ):
            raise ValueError(
                "the frozen DINO-video cache supervises posterior/prior future queries only"
            )
        if request.query_count <= 0 or batch <= 0:
            raise ValueError("predictive cache lookup requires a non-empty request")
        if (request.route_ids != self.contract.route_id).any():
            raise ValueError("predictive request route differs from the cache contract")
        if any(
            int(value) not in self.contract.horizons
            for value in request.horizons.detach().cpu().reshape(-1).tolist()
        ):
            raise ValueError("predictive request horizon differs from the cache contract")
        target_device = torch.device(device)
        if request.route_ids.device != target_device:
            raise ValueError("predictive request and cache target device must match")
        if (
            len(sources) != batch
            or len(source_hashes) != batch
            or len(track_identity_keys) != batch
        ):
            raise ValueError("predictive cache lookup fields must match the request batch")
        tracks = max(len(keys) for keys in track_identity_keys)
        features = torch.zeros(
            batch,
            tracks,
            request.query_count,
            self.contract.hidden_size,
            dtype=torch.float32,
            device=target_device,
        )
        importance = torch.zeros(
            batch,
            tracks,
            request.query_count,
            dtype=torch.float32,
            device=target_device,
        )
        lookup_audit: list[dict[str, object]] = []
        for batch_index, (source_index, source_hash, expected_identities) in enumerate(
            zip(sources, source_hashes, track_identity_keys, strict=True)
        ):
            _nonnegative_int(source_index, "predictive source global index")
            _sha256(source_hash, "predictive source RGB sha256")
            if not expected_identities:
                raise ValueError("predictive cache lookup requires at least one structural track")
            expected_position = {key: index for index, key in enumerate(expected_identities)}
            if len(expected_position) != len(expected_identities):
                raise ValueError("predictive structural track identities must be unique")
            for query_index in range(request.query_count):
                if not bool(request.valid[batch_index, query_index]):
                    continue
                horizon = int(request.horizons[batch_index, query_index])
                location = self.locator.get((source_index, horizon))
                if location is None:
                    raise KeyError(
                        f"predictive cache omits source={source_index}, horizon={horizon}"
                    )
                shard_index, row = location
                loaded = self._load_shard(shard_index)
                if str(loaded.source_rgb_sha256[row]) != source_hash:
                    raise ContractError("predictive cache source RGB differs from current batch")
                start = int(loaded.frame_offsets[row])
                stop = int(loaded.frame_offsets[row + 1])
                for local_index, identity_key in enumerate(
                    loaded.identity_keys[start:stop].tolist()
                ):
                    target_position = expected_position.get(str(identity_key))
                    if target_position is None:
                        continue
                    mass = float(loaded.importance[start + local_index])
                    if mass <= 0:
                        continue
                    features[batch_index, target_position, query_index] = torch.from_numpy(
                        loaded.features[start + local_index].astype(np.float32)
                    ).to(features)
                    importance[batch_index, target_position, query_index] = mass
                lookup_audit.append(
                    {
                        "horizon": horizon,
                        "source_global_index": source_index,
                        "source_rgb_sha256": source_hash,
                        "target_global_index": int(loaded.target_global_indices[row]),
                        "target_rgb_sha256": str(loaded.target_rgb_sha256[row]),
                    }
                )
        source_batch_digest = _sha256_bytes(_canonical_json(lookup_audit))
        valid = importance > 0
        return make_native_predictive_target(
            modality=LINGBOT_PREDICTIVE_MODALITY,
            features=features,
            valid=valid,
            importance=importance,
            route_ids=request.route_ids,
            horizons=request.horizons,
            source=request.source,
            evidence=request.evidence,
            encoder_mode=TargetEncoderMode.FROZEN,
            source_batch_digest=source_batch_digest,
            target_data_digest=self.manifest_sha256,
            encoder_digest=self.contract.encoder_digest,
            query_schema_digest=self.contract.query_schema_sha256,
            validity_semantics=(
                "positive task-independent visible-owner support; occluded and absent objects "
                "have no denominator"
            ),
            track_identity_keys=track_identity_keys,
        )


def _write_predictive_shard(
    staging: Path,
    *,
    shard_index: int,
    records: Sequence[PredictiveObjectCacheRecord],
) -> dict[str, object]:
    if not records:
        raise ContractError("cannot publish an empty predictive shard")
    offsets = [0]
    identities: list[str] = []
    feature_rows: list[NDArray[np.float16]] = []
    importance_rows: list[NDArray[np.float32]] = []
    for record in records:
        identities.extend(record.identity_keys)
        feature_rows.append(record.features)
        importance_rows.append(record.importance)
        offsets.append(len(identities))
    string_width = max(1, *(len(value) for value in identities))
    path = staging / f"shard-{shard_index:06d}.npz"
    np.savez(
        path,
        source_global_indices=np.asarray(
            [value.source_global_index for value in records], dtype=np.int64
        ),
        target_global_indices=np.asarray(
            [value.target_global_index for value in records], dtype=np.int64
        ),
        horizons=np.asarray([value.horizon for value in records], dtype=np.int64),
        source_rgb_sha256=np.asarray([value.source_rgb_sha256 for value in records], dtype="<U64"),
        target_rgb_sha256=np.asarray([value.target_rgb_sha256 for value in records], dtype="<U64"),
        frame_offsets=np.asarray(offsets, dtype=np.int64),
        identity_keys=np.asarray(identities, dtype=f"<U{string_width}"),
        features=np.concatenate(feature_rows, axis=0),
        importance=np.concatenate(importance_rows, axis=0),
    )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    first = records[0]
    last = records[-1]
    return {
        "first_horizon": first.horizon,
        "first_source_global_index": first.source_global_index,
        "last_horizon": last.horizon,
        "last_source_global_index": last.source_global_index,
        "object_count": len(identities),
        "path": path.name,
        "row_count": len(records),
        "sha256": _sha256_file(path),
    }


def write_predictive_target_cache(
    output_root: str | Path,
    *,
    contract: PredictiveCacheContract,
    records: Iterable[PredictiveObjectCacheRecord],
    shard_rows: int = 2048,
) -> str:
    """Stream and atomically publish exactly the contract's complete coverage."""

    if not isinstance(contract, PredictiveCacheContract):
        raise TypeError("predictive cache writer requires a typed contract")
    if isinstance(shard_rows, bool) or not isinstance(shard_rows, int) or shard_rows <= 0:
        raise ValueError("predictive cache shard_rows must be positive")

    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir()
    published = False
    try:
        shard_metadata: list[dict[str, object]] = []
        shard_records: list[PredictiveObjectCacheRecord] = []
        pair_digest = _PredictivePairDigest()
        for record in records:
            if not isinstance(record, PredictiveObjectCacheRecord):
                raise TypeError("predictive cache records must use the typed record contract")
            actual_pair = record.source_global_index, record.horizon
            pair_digest.add(actual_pair)
            if record.horizon not in contract.horizons:
                raise ContractError("predictive cache record horizon is outside the contract")
            if record.features.shape[1] != contract.hidden_size:
                raise ContractError("predictive cache record width differs from teacher contract")
            shard_records.append(record)
            if pair_digest.count > contract.expected_record_count:
                raise ContractError("predictive cache contains records beyond frozen coverage")
            if len(shard_records) == shard_rows:
                shard_metadata.append(
                    _write_predictive_shard(
                        staging,
                        shard_index=len(shard_metadata),
                        records=shard_records,
                    )
                )
                shard_records = []
        if shard_records:
            shard_metadata.append(
                _write_predictive_shard(
                    staging,
                    shard_index=len(shard_metadata),
                    records=shard_records,
                )
            )
        if (
            pair_digest.count != contract.expected_record_count
            or pair_digest.hexdigest() != contract.pair_keys_sha256
            or not shard_metadata
        ):
            raise ContractError("predictive cache keys differ from frozen coverage")

        contract_payload = {
            field_name: (list(value) if field_name == "horizons" else value)
            for field_name, value in (
                (field.name, getattr(contract, field.name))
                for field in PredictiveCacheContract.__dataclass_fields__.values()
            )
        }
        manifest = {
            "complete": True,
            "contract": contract_payload,
            "expected_records": contract.expected_record_count,
            "record_count": pair_digest.count,
            "schema": LINGBOT_PREDICTIVE_CACHE_SCHEMA,
            "shards": shard_metadata,
            "target_space": LINGBOT_PREDICTIVE_TARGET_SPACE,
        }
        manifest_path = staging / "manifest.json"
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("ascii") + b"\n"
        with manifest_path.open("wb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, output)
        published = True
        parent_descriptor = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return _sha256_bytes(manifest_bytes)
    except BaseException:
        rename_completed = not staging.exists() and output.is_dir()
        if published or rename_completed:
            shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise
