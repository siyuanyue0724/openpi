"""Immutable frozen-SigLIP2 future targets for paper-faithful FLARE training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CALVIN_HOST_IMAGE_KEYS,
    CalvinMolmoAct2SourceObservation,
    CalvinPhysicalTransitionDataset,
)
from picf_next.lingbot_native.future_latent_alignment import (
    FLARE_GENERIC_TARGET_SCHEMA,
    FLARE_SIGLIP2_MODEL_ID,
    FLARE_SIGLIP2_REVISION,
    FLARE_TARGET_VIEW_ORDER,
    FutureLatentAlignmentConfig,
    FutureLatentTargetBatch,
)

FLARE_TARGET_CACHE_SCHEMA = "picf-next.flare-future-target-cache.v1"
_MANIFEST_NAME = "manifest.json"
_MANIFEST_MAXIMUM_BYTES = 128 * 1024 * 1024
_CALVIN_VIEW_KEY = dict(zip(FLARE_TARGET_VIEW_ORDER, CALVIN_HOST_IMAGE_KEYS, strict=True))


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


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be non-empty text")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _text(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return digest


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact_mapping(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != fields:
        raise ContractError(f"{name} fields differ from the frozen schema")
    return payload


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


def _image_sha256(image: np.ndarray) -> str:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ContractError("FLARE source image must be uint8 HWC RGB")
    digest = hashlib.sha256()
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(image)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FutureLatentCacheContract:
    """Dataset, teacher, schedule, and tensor identity for one immutable cache."""

    dataset_id: str
    dataset_revision: str
    split_name: str
    dataset_tree_sha256: str
    stream_plan_sha256: str
    stream_plan_file_sha256: str
    representation_split_sha256: str
    source_keys_sha256: str
    encoder_config_sha256: str
    encoder_checkpoint_sha256: str
    encoder_processor_sha256: str
    expected_record_count: int
    training_prefix_steps: int
    alignment_config_digest: str
    encoder_model_id: str = FLARE_SIGLIP2_MODEL_ID
    encoder_revision: str = FLARE_SIGLIP2_REVISION
    target_schema: str = FLARE_GENERIC_TARGET_SCHEMA
    tensor_dtype: str = "float32"
    patch_grid: int = 16
    pooling: str = "nonoverlapping-2x2-arithmetic-mean"

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "dataset_revision",
            "split_name",
            "encoder_model_id",
            "encoder_revision",
            "target_schema",
            "tensor_dtype",
            "pooling",
        ):
            _text(getattr(self, name), name)
        for name in (
            "dataset_tree_sha256",
            "stream_plan_sha256",
            "stream_plan_file_sha256",
            "representation_split_sha256",
            "source_keys_sha256",
            "encoder_config_sha256",
            "encoder_checkpoint_sha256",
            "encoder_processor_sha256",
            "alignment_config_digest",
        ):
            _sha256(getattr(self, name), name)
        _positive_int(self.expected_record_count, "FLARE expected record count")
        _positive_int(self.training_prefix_steps, "FLARE training prefix steps")
        _positive_int(self.patch_grid, "FLARE patch grid")
        expected = FutureLatentAlignmentConfig()
        expected.assert_adr209_complete()
        if self.alignment_config_digest != expected.digest:
            raise ContractError("FLARE cache uses a reduced or unknown alignment configuration")
        if (
            self.encoder_model_id != FLARE_SIGLIP2_MODEL_ID
            or self.encoder_revision != FLARE_SIGLIP2_REVISION
            or self.target_schema != FLARE_GENERIC_TARGET_SCHEMA
            or self.tensor_dtype != "float32"
            or self.patch_grid != 16
            or self.pooling != "nonoverlapping-2x2-arithmetic-mean"
        ):
            raise ContractError("FLARE cache differs from the complete frozen generic-target arm")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> FutureLatentCacheContract:
        payload = _exact_mapping(value, "FLARE cache contract", set(cls.__dataclass_fields__))
        return cls(
            dataset_id=_text(payload["dataset_id"], "dataset_id"),
            dataset_revision=_text(payload["dataset_revision"], "dataset_revision"),
            split_name=_text(payload["split_name"], "split_name"),
            dataset_tree_sha256=_sha256(payload["dataset_tree_sha256"], "dataset tree"),
            stream_plan_sha256=_sha256(payload["stream_plan_sha256"], "stream plan"),
            stream_plan_file_sha256=_sha256(
                payload["stream_plan_file_sha256"], "stream plan file"
            ),
            representation_split_sha256=_sha256(
                payload["representation_split_sha256"], "representation split"
            ),
            source_keys_sha256=_sha256(payload["source_keys_sha256"], "source keys"),
            encoder_config_sha256=_sha256(payload["encoder_config_sha256"], "encoder config"),
            encoder_checkpoint_sha256=_sha256(
                payload["encoder_checkpoint_sha256"], "encoder checkpoint"
            ),
            encoder_processor_sha256=_sha256(
                payload["encoder_processor_sha256"], "encoder processor"
            ),
            expected_record_count=_positive_int(
                payload["expected_record_count"], "expected record count"
            ),
            training_prefix_steps=_positive_int(
                payload["training_prefix_steps"], "training prefix steps"
            ),
            alignment_config_digest=_sha256(
                payload["alignment_config_digest"], "alignment config"
            ),
            encoder_model_id=_text(payload["encoder_model_id"], "encoder model ID"),
            encoder_revision=_text(payload["encoder_revision"], "encoder revision"),
            target_schema=_text(payload["target_schema"], "target schema"),
            tensor_dtype=_text(payload["tensor_dtype"], "tensor dtype"),
            patch_grid=_positive_int(payload["patch_grid"], "patch grid"),
            pooling=_text(payload["pooling"], "pooling"),
        )


@dataclass(frozen=True, slots=True)
class FutureLatentCacheRecord:
    sample_key: str
    source_global_index: int
    future_global_index: int
    future_view_sha256: tuple[str, ...]
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        _text(self.sample_key, "FLARE sample key")
        _nonnegative_int(self.source_global_index, "FLARE source index")
        _nonnegative_int(self.future_global_index, "FLARE future index")
        if (
            self.future_global_index - self.source_global_index
            != FutureLatentAlignmentConfig().target_offset_source_frames
        ):
            raise ContractError("FLARE future index must be exactly 16 raw frames after source")
        if len(self.future_view_sha256) != len(FLARE_TARGET_VIEW_ORDER):
            raise ContractError("FLARE record must bind every target view")
        for digest in self.future_view_sha256:
            _sha256(digest, "FLARE future-view digest")
        config = FutureLatentAlignmentConfig()
        if (
            self.tokens.shape != (config.future_token_count, config.target_width)
            or self.tokens.dtype != torch.float32
            or self.tokens.requires_grad
            or not torch.isfinite(self.tokens).all()
        ):
            raise ContractError("FLARE record tokens must be detached finite FP32 [128,1024]")


class FrozenSiglip2FutureEncoder:
    """Frozen official SigLIP2 vision teacher used only while building cache data."""

    def __init__(
        self,
        *,
        model: nn.Module,
        image_processor: Any,
        device: torch.device,
        compute_dtype: torch.dtype,
        config: FutureLatentAlignmentConfig | None = None,
    ) -> None:
        self.config = FutureLatentAlignmentConfig() if config is None else config
        self.config.assert_adr209_complete()
        if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("FLARE teacher compute dtype is unsupported")
        vision_model = getattr(model, "vision_model", model)
        if not isinstance(vision_model, nn.Module):
            raise TypeError("SigLIP2 checkpoint does not expose a vision model")
        self.model = vision_model.eval().to(device=device, dtype=compute_dtype)
        self.model.requires_grad_(False)
        self.image_processor = image_processor
        self.device = device
        self.compute_dtype = compute_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_root: str | os.PathLike[str],
        *,
        device: torch.device,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> FrozenSiglip2FutureEncoder:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError("FLARE cache generation requires transformers") from error
        root = Path(model_root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        processor = AutoImageProcessor.from_pretrained(
            root,
            local_files_only=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            root,
            local_files_only=True,
            dtype=compute_dtype,
        )
        return cls(
            model=model,
            image_processor=processor,
            device=device,
            compute_dtype=compute_dtype,
        )

    @torch.inference_mode()
    def encode(self, observations: Sequence[CalvinMolmoAct2SourceObservation]) -> torch.Tensor:
        if not observations:
            raise ValueError("FLARE encoder batch cannot be empty")
        pooled_views: list[torch.Tensor] = []
        for view_name in self.config.view_order:
            images = [
                observation.images[_CALVIN_VIEW_KEY[view_name]] for observation in observations
            ]
            processed = self.image_processor(images=images, return_tensors="pt")
            pixel_values = processed.get("pixel_values")
            if not isinstance(pixel_values, torch.Tensor):
                raise RuntimeError("SigLIP2 processor omitted pixel_values")
            outputs = self.model(
                pixel_values=pixel_values.to(
                    device=self.device,
                    dtype=self.compute_dtype,
                    non_blocking=True,
                ),
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            expected = (
                len(observations),
                self.config.tokens_per_view * 4,
                self.config.target_width,
            )
            if not isinstance(hidden, torch.Tensor) or hidden.shape != expected:
                raise RuntimeError("SigLIP2 vision output differs from 256 native patch tokens")
            pooled_grid_side = math.isqrt(self.config.tokens_per_view)
            if pooled_grid_side * pooled_grid_side != self.config.tokens_per_view:
                raise RuntimeError("FLARE pooled patch tokens do not form a square grid")
            grid = hidden.float().reshape(
                len(observations),
                pooled_grid_side,
                2,
                pooled_grid_side,
                2,
                self.config.target_width,
            )
            pooled = grid.mean(dim=(2, 4)).reshape(
                len(observations),
                self.config.tokens_per_view,
                self.config.target_width,
            )
            pooled_views.append(pooled)
        targets = torch.cat(pooled_views, dim=1).cpu().contiguous()
        if targets.shape != (
            len(observations),
            self.config.future_token_count,
            self.config.target_width,
        ):
            raise RuntimeError("FLARE encoder dropped a target view or patch token")
        return targets


def build_calvin_future_latent_records(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    sample_keys: Sequence[str],
    encoder: FrozenSiglip2FutureEncoder,
    batch_size: int,
) -> tuple[FutureLatentCacheRecord, ...]:
    """Encode exact raw-frame `t+16` targets without crossing episode resets."""

    _positive_int(batch_size, "FLARE encoder batch size")
    keys = tuple(sample_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ContractError("FLARE source keys must be non-empty and unique")
    records: list[FutureLatentCacheRecord] = []
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start : start + batch_size]
        source_indices: list[int] = []
        future_indices: list[int] = []
        observations: list[CalvinMolmoAct2SourceObservation] = []
        hashes: list[tuple[str, ...]] = []
        for sample_key in batch_keys:
            source_index = dataset.source_global_index_by_key(sample_key)
            future_index = dataset.future_source_global_indices_by_key(
                sample_key,
                count=encoder.config.target_offset_source_frames,
            )[-1]
            observation = dataset.index.molmoact2_source_observation(future_index)
            source_indices.append(source_index)
            future_indices.append(future_index)
            observations.append(observation)
            hashes.append(
                tuple(
                    _image_sha256(observation.images[_CALVIN_VIEW_KEY[name]])
                    for name in FLARE_TARGET_VIEW_ORDER
                )
            )
        targets = encoder.encode(observations)
        for index, sample_key in enumerate(batch_keys):
            records.append(
                FutureLatentCacheRecord(
                    sample_key=sample_key,
                    source_global_index=source_indices[index],
                    future_global_index=future_indices[index],
                    future_view_sha256=hashes[index],
                    tokens=targets[index],
                )
            )
    return tuple(records)


def future_latent_source_keys_digest(
    identities: Sequence[tuple[str, int, int]],
) -> str:
    if not identities:
        raise ContractError("FLARE source coverage cannot be empty")
    digest = hashlib.sha256(b"picf-next.flare-source-keys.v1\0")
    previous: tuple[str, int, int] | None = None
    for sample_key, source_index, future_index in identities:
        identity = (
            _text(sample_key, "FLARE source key"),
            _nonnegative_int(source_index, "FLARE source index"),
            _nonnegative_int(future_index, "FLARE future index"),
        )
        if previous is not None and identity <= previous:
            raise ContractError("FLARE source identities must be sorted and unique")
        digest.update(_canonical_json(identity))
        previous = identity
    digest.update(len(identities).to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _record_tensor_sha256(tokens: torch.Tensor) -> str:
    return _sha256_bytes(tokens.detach().cpu().contiguous().numpy().tobytes(order="C"))


def write_future_latent_target_cache(
    root: str | os.PathLike[str],
    *,
    contract: FutureLatentCacheContract,
    records: Sequence[FutureLatentCacheRecord],
    records_per_shard: int = 256,
) -> str:
    """Write content-addressed safetensor shards and publish the manifest last."""

    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("FLARE cache writing requires safetensors") from error
    _positive_int(records_per_shard, "FLARE records per shard")
    values = tuple(records)
    if len(values) != contract.expected_record_count:
        raise ContractError("FLARE cache record count differs from its frozen contract")
    if len({record.sample_key for record in values}) != len(values):
        raise ContractError("FLARE cache contains duplicate sample keys")
    identities = tuple(
        (record.sample_key, record.source_global_index, record.future_global_index)
        for record in values
    )
    if future_latent_source_keys_digest(identities) != contract.source_keys_sha256:
        raise ContractError("FLARE cache source identities differ from the contract")

    destination = Path(root)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir()
    record_payloads: list[dict[str, object]] = []
    shard_payloads: list[dict[str, object]] = []
    try:
        for shard_index, start in enumerate(range(0, len(values), records_per_shard)):
            shard_records = values[start : start + records_per_shard]
            tensor = torch.stack(tuple(record.tokens for record in shard_records)).contiguous()
            relative = f"shards/targets-{shard_index:05d}.safetensors"
            shard_path = staging / relative
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"targets": tensor}, shard_path)
            shard_payloads.append(
                {
                    "path": relative,
                    "record_count": len(shard_records),
                    "sha256": _sha256_file(shard_path),
                    "shape": list(tensor.shape),
                }
            )
            for row, record in enumerate(shard_records):
                record_payloads.append(
                    {
                        "future_global_index": record.future_global_index,
                        "future_view_sha256": list(record.future_view_sha256),
                        "row": row,
                        "sample_key": record.sample_key,
                        "shard": shard_index,
                        "source_global_index": record.source_global_index,
                        "tensor_sha256": _record_tensor_sha256(record.tokens),
                    }
                )
        unsealed = {
            "contract": contract.to_dict(),
            "records": record_payloads,
            "schema": FLARE_TARGET_CACHE_SCHEMA,
            "shards": shard_payloads,
        }
        manifest_sha256 = _sha256_bytes(_canonical_json(unsealed))
        manifest = {**unsealed, "manifest_sha256": manifest_sha256}
        (staging / _MANIFEST_NAME).write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest_sha256


@dataclass(frozen=True, slots=True)
class _StoredFutureRecord:
    sample_key: str
    source_global_index: int
    future_global_index: int
    future_view_sha256: tuple[str, ...]
    tensor_sha256: str
    shard: int
    row: int


class FutureLatentTargetCache:
    """Hash-verified, lazily sharded FLARE target cache."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        maximum_open_shards: int = 2,
        verify_shards: bool = True,
    ) -> None:
        _positive_int(maximum_open_shards, "FLARE maximum open shards")
        if not isinstance(verify_shards, bool):
            raise TypeError("FLARE shard verification mode must be boolean")
        self.root = Path(root)
        manifest_path = self.root / _MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.stat().st_size > _MANIFEST_MAXIMUM_BYTES:
            raise ContractError("FLARE cache manifest is absent or too large")
        payload = _mapping(json.loads(manifest_path.read_text("ascii")), "FLARE manifest")
        expected_fields = {"contract", "manifest_sha256", "records", "schema", "shards"}
        if set(payload) != expected_fields or payload.get("schema") != FLARE_TARGET_CACHE_SCHEMA:
            raise ContractError("FLARE cache manifest schema differs")
        manifest_sha256 = _sha256(payload["manifest_sha256"], "FLARE manifest digest")
        unsealed = {name: payload[name] for name in ("contract", "records", "schema", "shards")}
        if _sha256_bytes(_canonical_json(unsealed)) != manifest_sha256:
            raise ContractError("FLARE cache manifest digest differs")
        self.contract = FutureLatentCacheContract.from_mapping(payload["contract"])
        self.manifest_sha256 = manifest_sha256
        shard_values = payload["shards"]
        record_values = payload["records"]
        if not isinstance(shard_values, list) or not isinstance(record_values, list):
            raise ContractError("FLARE cache shard and record tables must be lists")
        self._shards: list[tuple[Path, str, tuple[int, ...]]] = []
        config = FutureLatentAlignmentConfig()
        for index, value in enumerate(shard_values):
            shard = _exact_mapping(
                value,
                f"FLARE shard {index}",
                {"path", "record_count", "sha256", "shape"},
            )
            relative = _relative_path(shard["path"], "FLARE shard path")
            shape_value = shard["shape"]
            if not isinstance(shape_value, list) or len(shape_value) != 3:
                raise ContractError("FLARE shard shape must have three dimensions")
            shape = tuple(_positive_int(item, "FLARE shard dimension") for item in shape_value)
            if shape[1:] != (config.future_token_count, config.target_width):
                raise ContractError("FLARE shard shape differs from the complete target")
            record_count = _positive_int(shard["record_count"], "FLARE shard record count")
            if record_count != shape[0]:
                raise ContractError("FLARE shard count and tensor shape differ")
            path = self.root / relative
            expected_digest = _sha256(shard["sha256"], "FLARE shard digest")
            if not path.is_file() or (verify_shards and _sha256_file(path) != expected_digest):
                raise ContractError("FLARE shard is absent or its digest differs")
            self._shards.append((path, expected_digest, shape))
        records: list[_StoredFutureRecord] = []
        for value in record_values:
            record = _exact_mapping(
                value,
                "FLARE cache record",
                {
                    "future_global_index",
                    "future_view_sha256",
                    "row",
                    "sample_key",
                    "shard",
                    "source_global_index",
                    "tensor_sha256",
                },
            )
            views = record["future_view_sha256"]
            if not isinstance(views, list) or len(views) != len(FLARE_TARGET_VIEW_ORDER):
                raise ContractError("FLARE record view digest table differs")
            stored = _StoredFutureRecord(
                sample_key=_text(record["sample_key"], "FLARE sample key"),
                source_global_index=_nonnegative_int(
                    record["source_global_index"], "FLARE source index"
                ),
                future_global_index=_nonnegative_int(
                    record["future_global_index"], "FLARE future index"
                ),
                future_view_sha256=tuple(
                    _sha256(item, "FLARE view digest") for item in views
                ),
                tensor_sha256=_sha256(record["tensor_sha256"], "FLARE tensor digest"),
                shard=_nonnegative_int(record["shard"], "FLARE shard index"),
                row=_nonnegative_int(record["row"], "FLARE shard row"),
            )
            if (
                stored.future_global_index - stored.source_global_index
                != config.target_offset_source_frames
            ):
                raise ContractError("FLARE cache record changed the exact future horizon")
            if stored.shard >= len(self._shards) or stored.row >= self._shards[stored.shard][2][0]:
                raise ContractError("FLARE record points outside its shard")
            records.append(stored)
        if len(records) != self.contract.expected_record_count:
            raise ContractError("FLARE manifest record count differs from its contract")
        if len({record.sample_key for record in records}) != len(records):
            raise ContractError("FLARE manifest contains duplicate sample keys")
        identities = tuple(
            (record.sample_key, record.source_global_index, record.future_global_index)
            for record in records
        )
        if future_latent_source_keys_digest(identities) != self.contract.source_keys_sha256:
            raise ContractError("FLARE manifest source coverage differs from its contract")
        self._record_by_key = {record.sample_key: record for record in records}
        self._maximum_open_shards = maximum_open_shards
        self._open_shards: OrderedDict[int, torch.Tensor] = OrderedDict()

    def _load_shard(self, shard_index: int) -> torch.Tensor:
        cached = self._open_shards.pop(shard_index, None)
        if cached is not None:
            self._open_shards[shard_index] = cached
            return cached
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("FLARE cache loading requires safetensors") from error
        path, _digest, expected_shape = self._shards[shard_index]
        payload = load_file(path, device="cpu")
        if set(payload) != {"targets"}:
            raise ContractError("FLARE shard tensor names differ")
        tensor = payload["targets"]
        if tensor.shape != expected_shape or tensor.dtype != torch.float32:
            raise ContractError("FLARE shard tensor shape or dtype differs")
        if not torch.isfinite(tensor).all():
            raise ContractError("FLARE shard contains NaN or infinity")
        self._open_shards[shard_index] = tensor
        while len(self._open_shards) > self._maximum_open_shards:
            self._open_shards.popitem(last=False)
        return tensor

    def target_for(
        self,
        *,
        sample_keys: Sequence[str],
        source_global_indices: Sequence[int],
        device: torch.device,
    ) -> FutureLatentTargetBatch:
        keys = tuple(sample_keys)
        indices = tuple(source_global_indices)
        if not keys or len(keys) != len(indices):
            raise ContractError("FLARE runtime identities must share one non-empty batch axis")
        rows: list[torch.Tensor] = []
        future_indices: list[int] = []
        for sample_key, source_index in zip(keys, indices, strict=True):
            try:
                record = self._record_by_key[sample_key]
            except KeyError as error:
                raise KeyError(f"FLARE cache omits sample {sample_key!r}") from error
            if record.source_global_index != source_index:
                raise ContractError("FLARE runtime source identity differs from its cache record")
            row = self._load_shard(record.shard)[record.row]
            if _record_tensor_sha256(row) != record.tensor_sha256:
                raise ContractError("FLARE target row digest differs")
            rows.append(row)
            future_indices.append(record.future_global_index)
        return FutureLatentTargetBatch(
            tokens=torch.stack(rows).to(device=device, dtype=torch.float32, non_blocking=True),
            sample_keys=keys,
            source_global_indices=indices,
            future_global_indices=tuple(future_indices),
            manifest_sha256=self.manifest_sha256,
            config_digest=self.contract.alignment_config_digest,
        )


def teacher_asset_digests(model_root: str | os.PathLike[str]) -> dict[str, str]:
    """Hash the exact three teacher assets bound by the cache contract."""

    root = Path(model_root)
    paths = {
        "encoder_checkpoint_sha256": root / "model.safetensors",
        "encoder_config_sha256": root / "config.json",
        "encoder_processor_sha256": root / "preprocessor_config.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("FLARE teacher root omits a required asset")
    return {name: _sha256_file(path) for name, path in paths.items()}


def eligible_calvin_future_keys(
    dataset: CalvinPhysicalTransitionDataset,
    sample_keys: Sequence[str],
    *,
    offset: int = 16,
) -> tuple[str, ...]:
    """Return only keys with an exact same-episode raw-frame future target."""

    _positive_int(offset, "FLARE future offset")
    eligible: list[str] = []
    for sample_key in sample_keys:
        try:
            dataset.future_source_global_indices_by_key(sample_key, count=offset)
        except ContractError:
            continue
        eligible.append(sample_key)
    if not eligible:
        raise ContractError("no CALVIN samples have a complete FLARE future horizon")
    return tuple(eligible)
