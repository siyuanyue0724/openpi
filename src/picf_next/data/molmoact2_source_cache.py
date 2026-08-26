"""Content-addressed MolmoAct2 source-frame features for stationary PICF training.

The cache is deliberately observation-only.  It exposes the frozen dense
vision bank and prediction-free processor layout, while physical identities,
masks, task text, actions and future state remain in separate adapters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import stat
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin_loss_targets import CalvinSourceFrameLossTargetRequest
from picf_next.data.calvin_physical_supervision_schema import CALVIN_CAMERA_SPECS
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath
from picf_next.hosts.molmoact2_layout import (
    MolmoAct2ImagePatchSpan,
    MolmoAct2VisionPatchLayout,
)
from picf_next.models.evidence import NativeTokenBank

MOLMOACT2_SOURCE_CACHE_SCHEMA = "picf-next.molmoact2-m2-feature-cache.v1"
MOLMOACT2_SOURCE_CACHE_TARGET_CONTRACT = "source_frame"

_MANIFEST_FIELDS = {
    "checkpoint_id",
    "checkpoint_revision",
    "cuda_peak_allocated_bytes",
    "dtype",
    "elapsed_s",
    "foundation_recipe_sha256",
    "gate",
    "loss_target_fields_in_feature_shards",
    "model_input_fields",
    "modality",
    "processor_layout",
    "processor_layout_sha256",
    "records",
    "records_sha256",
    "sample_count",
    "schema",
    "shards",
    "source_coverage_recipe_sha256",
    "task_field_supplied",
    "token_shape",
}
_RECORD_FIELDS = {
    "global_index",
    "instruction",
    "row",
    "sample_key",
    "shard",
    "source_block_index",
    "source_sensor_sha256",
    "split",
    "target_request_contract",
    "task_key",
}
_SHARD_FIELDS = {"bytes", "path", "rows", "sha256"}
_LAYOUT_FIELDS = {
    "image_grid",
    "image_key",
    "image_num_crops",
    "image_token_pooling",
    "patches_per_crop",
    "start",
    "stop",
}
_SOURCE_SENSOR_FIELDS = frozenset(
    str(spec[field])
    for spec in CALVIN_CAMERA_SPECS
    for field in ("source_rgb_field", "source_depth_field")
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != fields:
        raise ContractError(f"{name} fields differ from the frozen cache schema")
    return payload


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return text


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise ContractError(f"{name} must be a {qualifier} integer")
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _open_regular_file_beneath(root: Path, relative: str) -> BinaryIO:
    """Open one immutable cache file without following path-component symlinks."""

    parts = PurePosixPath(relative).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise ContractError(f"Molmo cache root cannot be opened safely: {root}") from exc
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=directory_descriptor)
            except OSError as exc:
                raise ContractError(f"Molmo cache path is unsafe: {relative}") from exc
            os.close(directory_descriptor)
            directory_descriptor = child
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise ContractError(f"Molmo cache file cannot be opened safely: {relative}") from exc
    finally:
        os.close(directory_descriptor)
    return os.fdopen(descriptor, "rb")


@dataclass(frozen=True, slots=True)
class MolmoAct2SourceCacheRecord:
    sample_key: str
    split: str
    source_block_index: int
    global_index: int
    source_sensor_sha256: tuple[tuple[str, str], ...]
    shard: str
    row: int

    def target_request(self) -> CalvinSourceFrameLossTargetRequest:
        return CalvinSourceFrameLossTargetRequest(
            sample_key=self.sample_key,
            source_global_index=self.global_index,
            augmentation_seed=0,
            source_sensor_sha256=self.source_sensor_sha256,
        )


@dataclass(frozen=True, slots=True)
class _MolmoAct2SourceCacheShard:
    path: str
    sha256: str
    rows: int
    size_bytes: int


class MolmoAct2SourceFeatureCache:
    """Lazy bounded reader for task-free, frozen MolmoAct2 visual features."""

    def __init__(
        self,
        *,
        root: Path,
        modality: str,
        token_count: int,
        token_dim: int,
        encoder_contract: str,
        manifest_sha256: str,
        foundation_recipe_sha256: str,
        source_coverage_recipe_sha256: str,
        layout_row: tuple[MolmoAct2ImagePatchSpan, ...],
        records: tuple[MolmoAct2SourceCacheRecord, ...],
        shards: tuple[_MolmoAct2SourceCacheShard, ...],
        memory_capacity: int,
    ) -> None:
        self.root = root
        self.modality = modality
        self.token_count = token_count
        self.token_dim = token_dim
        self.encoder_contract = encoder_contract
        self.manifest_sha256 = manifest_sha256
        self.foundation_recipe_sha256 = foundation_recipe_sha256
        self.source_coverage_recipe_sha256 = source_coverage_recipe_sha256
        self.layout_row = layout_row
        self.records = {record.global_index: record for record in records}
        self.records_by_key = {record.sample_key: record for record in records}
        self.shards = {shard.path: shard for shard in shards}
        self.memory_capacity = memory_capacity
        self._loaded: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        manifest_sha256: str,
        expected_modality: str,
        expected_token_count: int,
        expected_token_dim: int,
        expected_checkpoint_id: str,
        expected_checkpoint_revision: str,
        memory_capacity: int = 2,
    ) -> MolmoAct2SourceFeatureCache:
        """Validate one all-source cache without loading its feature shards."""

        expected_manifest_sha = _sha256(manifest_sha256, "Molmo cache manifest sha256")
        expected_modality = _text(expected_modality, "expected Molmo modality")
        expected_token_count = _integer(
            expected_token_count,
            "expected Molmo token count",
            positive=True,
        )
        expected_token_dim = _integer(
            expected_token_dim,
            "expected Molmo token dimension",
            positive=True,
        )
        expected_checkpoint_id = _text(expected_checkpoint_id, "expected Molmo checkpoint id")
        expected_checkpoint_revision = _text(
            expected_checkpoint_revision,
            "expected Molmo checkpoint revision",
        )
        memory_capacity = _integer(memory_capacity, "Molmo cache capacity", positive=True)
        resolved_root = Path(root).resolve()
        manifest_bytes = read_sha256_verified_file_beneath(
            resolved_root,
            "manifest.json",
            expected_sha256=expected_manifest_sha,
            maximum_bytes=64 * 1024 * 1024,
        )
        try:
            raw = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("Molmo cache manifest is not valid JSON") from exc
        payload = _exact(raw, "Molmo source cache manifest", _MANIFEST_FIELDS)
        if payload["schema"] != MOLMOACT2_SOURCE_CACHE_SCHEMA:
            raise ContractError("Molmo source cache schema changed")
        if payload["modality"] != expected_modality:
            raise ContractError("Molmo source cache modality changed")
        if payload["dtype"] != "bfloat16":
            raise ContractError("Molmo source cache must contain bfloat16 features")
        if payload["token_shape"] != [expected_token_count, expected_token_dim]:
            raise ContractError("Molmo source cache token shape changed")
        if payload["model_input_fields"] != ["tokens", "valid"]:
            raise ContractError("Molmo source cache model-input fields changed")
        if payload["loss_target_fields_in_feature_shards"] != []:
            raise ContractError("Molmo source cache contains loss-target fields")
        if payload["task_field_supplied"] is not False:
            raise ContractError("Molmo source cache supplied a task field")

        _text(payload["gate"], "Molmo source cache gate")
        elapsed = payload["elapsed_s"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or elapsed < 0.0
        ):
            raise ContractError("Molmo source cache elapsed time must be finite and nonnegative")
        _integer(payload["cuda_peak_allocated_bytes"], "Molmo source cache peak CUDA bytes")
        checkpoint_id = _text(payload["checkpoint_id"], "Molmo checkpoint id")
        checkpoint_revision = _text(payload["checkpoint_revision"], "Molmo checkpoint revision")
        foundation_sha = _sha256(payload["foundation_recipe_sha256"], "foundation recipe sha256")
        source_coverage_sha = _sha256(
            payload["source_coverage_recipe_sha256"],
            "source coverage recipe sha256",
        )
        if (checkpoint_id, checkpoint_revision) != (
            expected_checkpoint_id,
            expected_checkpoint_revision,
        ):
            raise ContractError("Molmo source cache checkpoint identity changed")
        raw_layout = payload["processor_layout"]
        if not isinstance(raw_layout, list) or not raw_layout:
            raise ContractError("Molmo source cache processor layout must be a nonempty list")
        layout_sha = _sha256(payload["processor_layout_sha256"], "processor layout sha256")
        if layout_sha != _canonical_sha256(raw_layout):
            raise ContractError("Molmo source cache processor-layout hash changed")
        layout_row = tuple(
            cls._parse_layout_span(value, index=index) for index, value in enumerate(raw_layout)
        )
        MolmoAct2VisionPatchLayout(
            rows=(layout_row,),
            tokens_per_row=expected_token_count,
            semantic_image_keys=True,
        )

        raw_shards = payload["shards"]
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ContractError("Molmo source cache requires nonempty shards")
        shards = tuple(
            cls._parse_shard(value, index=index) for index, value in enumerate(raw_shards)
        )
        shard_paths = tuple(shard.path for shard in shards)
        if len(set(shard_paths)) != len(shard_paths):
            raise ContractError("Molmo source cache shard paths must be unique")
        shard_by_path = {shard.path: shard for shard in shards}

        raw_records = payload["records"]
        if not isinstance(raw_records, list) or not raw_records:
            raise ContractError("Molmo source cache requires nonempty records")
        records_sha = _sha256(payload["records_sha256"], "Molmo cache records sha256")
        if records_sha != _canonical_sha256(raw_records):
            raise ContractError("Molmo source cache records hash changed")
        records = tuple(
            cls._parse_record(value, index=index) for index, value in enumerate(raw_records)
        )
        if _integer(payload["sample_count"], "Molmo cache sample count", positive=True) != len(
            records
        ):
            raise ContractError("Molmo source cache sample count differs from its records")
        global_indices = tuple(record.global_index for record in records)
        sample_keys = tuple(record.sample_key for record in records)
        if global_indices != tuple(sorted(global_indices)) or len(set(global_indices)) != len(
            records
        ):
            raise ContractError("Molmo source cache global indices must be sorted and unique")
        if len(set(sample_keys)) != len(records):
            raise ContractError("Molmo source cache sample keys must be unique")
        locations = tuple((record.shard, record.row) for record in records)
        if len(set(locations)) != len(records):
            raise ContractError("Molmo source cache record locations must be unique")
        expected_locations = {(shard.path, row) for shard in shards for row in range(shard.rows)}
        if set(locations) != expected_locations or any(
            record.shard not in shard_by_path or record.row >= shard_by_path[record.shard].rows
            for record in records
        ):
            raise ContractError("Molmo source cache records and shard rows are not one-to-one")

        return cls(
            root=resolved_root,
            modality=expected_modality,
            token_count=expected_token_count,
            token_dim=expected_token_dim,
            encoder_contract=(
                f"{checkpoint_id}@{checkpoint_revision}/task-free-dense-vision-cache.v1"
            ),
            manifest_sha256=expected_manifest_sha,
            foundation_recipe_sha256=foundation_sha,
            source_coverage_recipe_sha256=source_coverage_sha,
            layout_row=layout_row,
            records=records,
            shards=shards,
            memory_capacity=memory_capacity,
        )

    @staticmethod
    def _parse_layout_span(raw: object, *, index: int) -> MolmoAct2ImagePatchSpan:
        payload = _exact(raw, f"Molmo processor span[{index}]", _LAYOUT_FIELDS)
        raw_grid = payload["image_grid"]
        raw_pooling = payload["image_token_pooling"]
        if (
            not isinstance(raw_grid, list)
            or not isinstance(raw_pooling, list)
            or not raw_pooling
            or any(not isinstance(row, list) or not row for row in raw_pooling)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < -1
                for row in raw_pooling
                for value in row
            )
        ):
            raise ContractError("Molmo processor span arrays must use JSON lists")
        try:
            image_grid_values = tuple(
                _integer(value, "Molmo image-grid value") for value in raw_grid
            )
            if len(image_grid_values) != 4:
                raise ContractError("Molmo image grid must contain four values")
            return MolmoAct2ImagePatchSpan(
                image_key=_text(payload["image_key"], "Molmo image key"),
                start=_integer(payload["start"], "Molmo span start"),
                stop=_integer(payload["stop"], "Molmo span stop", positive=True),
                image_num_crops=_integer(
                    payload["image_num_crops"],
                    "Molmo image crop count",
                    positive=True,
                ),
                patches_per_crop=_integer(
                    payload["patches_per_crop"],
                    "Molmo patches per crop",
                    positive=True,
                ),
                image_grid=(
                    image_grid_values[0],
                    image_grid_values[1],
                    image_grid_values[2],
                    image_grid_values[3],
                ),
                image_token_pooling=tuple(tuple(value for value in row) for row in raw_pooling),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("Molmo processor span is invalid") from exc

    @staticmethod
    def _parse_shard(raw: object, *, index: int) -> _MolmoAct2SourceCacheShard:
        payload = _exact(raw, f"Molmo cache shard[{index}]", _SHARD_FIELDS)
        return _MolmoAct2SourceCacheShard(
            path=_relative_path(payload["path"], "Molmo shard path"),
            sha256=_sha256(payload["sha256"], "Molmo shard sha256"),
            rows=_integer(payload["rows"], "Molmo shard rows", positive=True),
            size_bytes=_integer(payload["bytes"], "Molmo shard bytes", positive=True),
        )

    @staticmethod
    def _parse_record(raw: object, *, index: int) -> MolmoAct2SourceCacheRecord:
        payload = _exact(raw, f"Molmo cache record[{index}]", _RECORD_FIELDS)
        global_index = _integer(payload["global_index"], "Molmo source global index")
        sample_key = _text(payload["sample_key"], "Molmo source sample key")
        if sample_key != f"source-frame-{global_index:07d}":
            raise ContractError("Molmo source sample key differs from its global index")
        if payload["task_key"] != "task-independent-source-frame":
            raise ContractError("Molmo source cache task marker changed")
        if payload["instruction"] != "task field absent":
            raise ContractError("Molmo source cache instruction marker changed")
        if payload["target_request_contract"] != MOLMOACT2_SOURCE_CACHE_TARGET_CONTRACT:
            raise ContractError("Molmo source cache target-request contract changed")
        raw_hashes = payload["source_sensor_sha256"]
        if not isinstance(raw_hashes, list):
            raise ContractError("Molmo source sensor hashes must be a list")
        hashes = []
        for item in raw_hashes:
            if not isinstance(item, list) or len(item) != 2:
                raise ContractError("Molmo source sensor hash rows must be pairs")
            hashes.append(
                (
                    _text(item[0], "Molmo source sensor field"),
                    _sha256(item[1], "Molmo source sensor sha256"),
                )
            )
        sensor_hashes = tuple(hashes)
        names = tuple(name for name, _digest in sensor_hashes)
        if names != tuple(sorted(names)) or set(names) != _SOURCE_SENSOR_FIELDS:
            raise ContractError("Molmo source sensor hashes are incomplete or unsorted")
        return MolmoAct2SourceCacheRecord(
            sample_key=sample_key,
            split=_text(payload["split"], "Molmo source split"),
            source_block_index=_integer(
                payload["source_block_index"],
                "Molmo source block index",
            ),
            global_index=global_index,
            source_sensor_sha256=sensor_hashes,
            shard=_relative_path(payload["shard"], "Molmo record shard path"),
            row=_integer(payload["row"], "Molmo record shard row"),
        )

    def _load_shard(self, relative: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._loaded.get(relative)
        if cached is not None:
            self._loaded.move_to_end(relative)
            return cached
        try:
            shard = self.shards[relative]
        except KeyError as exc:
            raise KeyError(f"unknown Molmo source-cache shard {relative!r}") from exc
        with _open_regular_file_beneath(self.root, relative) as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != shard.size_bytes:
                raise ContractError("Molmo source-cache shard size or file type changed")
            digest = hashlib.sha256()
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
            after_hash = os.fstat(handle.fileno())

            def fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_mtime_ns,
                )

            if fingerprint(before) != fingerprint(after_hash) or digest.hexdigest() != shard.sha256:
                raise ContractError("Molmo source-cache shard content hash changed")
            handle.seek(0)
            try:
                payload = torch.load(handle, map_location="cpu", weights_only=True)
            except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
                raise ContractError("Molmo source-cache shard is not a safe torch payload") from exc
            if fingerprint(before) != fingerprint(os.fstat(handle.fileno())):
                raise ContractError("Molmo source-cache shard changed while loading")
        if not isinstance(payload, Mapping) or set(payload) != {"tokens", "valid"}:
            raise ContractError("Molmo source-cache shard contains non-observation fields")
        tokens = payload["tokens"]
        valid = payload["valid"]
        if (
            not isinstance(tokens, torch.Tensor)
            or not isinstance(valid, torch.Tensor)
            or tokens.dtype != torch.bfloat16
            or tokens.shape != (shard.rows, self.token_count, self.token_dim)
            or valid.dtype != torch.bool
            or valid.shape != tokens.shape[:2]
            or tokens.is_inference()
            or tokens.requires_grad
            or valid.requires_grad
            or not torch.isfinite(tokens).all()
            or (tokens[~valid] != 0.0).any()
        ):
            raise ContractError("Molmo source-cache shard tensor contract changed")
        cached = (tokens.contiguous(), valid.contiguous())
        self._loaded[relative] = cached
        while len(self._loaded) > self.memory_capacity:
            self._loaded.popitem(last=False)
        return cached

    def record(self, global_index: int) -> MolmoAct2SourceCacheRecord:
        try:
            return self.records[global_index]
        except KeyError as exc:
            raise KeyError(f"unknown Molmo source frame {global_index}") from exc

    def target_request(self, global_index: int) -> CalvinSourceFrameLossTargetRequest:
        return self.record(global_index).target_request()

    def vision_layout(self, batch_size: int) -> MolmoAct2VisionPatchLayout:
        batch_size = _integer(batch_size, "Molmo layout batch size", positive=True)
        return MolmoAct2VisionPatchLayout(
            rows=tuple(self.layout_row for _ in range(batch_size)),
            tokens_per_row=self.token_count,
            semantic_image_keys=True,
        )

    def native_bank(
        self,
        global_indices: Sequence[int],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> NativeTokenBank:
        indices = tuple(global_indices)
        if not indices:
            raise ValueError("Molmo source-cache bank requires at least one frame")
        if not dtype.is_floating_point:
            raise ValueError("Molmo source-cache destination dtype must be floating point")
        token_rows = []
        valid_rows = []
        for global_index in indices:
            if not isinstance(global_index, int) or isinstance(global_index, bool):
                raise TypeError("Molmo source-cache indices must be integers")
            record = self.record(global_index)
            tokens, valid = self._load_shard(record.shard)
            token_rows.append(tokens[record.row])
            valid_rows.append(valid[record.row])
        return NativeTokenBank(
            modality=self.modality,
            tokens=torch.stack(token_rows).to(device=device, dtype=dtype),
            valid=torch.stack(valid_rows).to(device=device),
            encoder_contract=self.encoder_contract,
        )
