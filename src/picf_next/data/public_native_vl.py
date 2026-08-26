"""Content-addressed public vision-language retention records for native Qwen SFT."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath

PUBLIC_NATIVE_VL_MANIFEST_SCHEMA = "picf-next.public-native-vl-retention-manifest.v2"
PUBLIC_NATIVE_VL_FAMILIES = ("referring", "vqa")
PUBLIC_NATIVE_VL_PARTITIONS = ("train", "heldout")
PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_BYTES = 32 * 1024 * 1024
PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_PIXELS = 40_000_000
PUBLIC_NATIVE_VL_MAXIMUM_MANIFEST_BYTES = 8 * 1024 * 1024
PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY = 64
PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY = 32
PUBLIC_NATIVE_VL_RETENTION_WEIGHT = 0.1

PUBLIC_NATIVE_VL_REFERRING_DATASET_ID = "sionic-ai/refcoco_object_detection"
PUBLIC_NATIVE_VL_REFERRING_REVISION = "a913181a10bf239cceb2576ba94e92c031a310a8"
PUBLIC_NATIVE_VL_REFERRING_SOURCE_FILE = "data/train-00000-of-00001.parquet"
PUBLIC_NATIVE_VL_REFERRING_SOURCE_SHA256 = (
    "1723dcd04bf51f8de1226972fb3d8d85e56460dd7ba4e55915f6ebe3c81a1ed5"
)
PUBLIC_NATIVE_VL_VQA_DATASET_ID = "HuggingFaceM4/the_cauldron"
PUBLIC_NATIVE_VL_VQA_REVISION = "847a98a779b1652d65111daf20c972dfcd333605"
PUBLIC_NATIVE_VL_VQA_SOURCE_FILE = "vqav2/train-00000-of-00022-1f40dc68b5c44ca4.parquet"
PUBLIC_NATIVE_VL_VQA_SOURCE_SHA256 = (
    "0e703056fa296fcaed0eb6ae26300cacc7ff699437e0e917ad4915cdc8715034"
)
PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION = (
    321,
    0,
    "visible audit: 'white donut on right' box covers the full multi-donut carton",
)

PublicNativeVLFamily = Literal["referring", "vqa"]
PublicNativeVLPartition = Literal["train", "heldout"]

_SOURCE_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_revision",
        "source_file",
        "source_file_sha256",
        "split",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "assistant_text",
        "family",
        "height",
        "image_file",
        "image_file_sha256",
        "image_rgb_sha256",
        "image_size_bytes",
        "partition",
        "priority_sha256",
        "record_id",
        "record_sha256",
        "source_key",
        "source_row_index",
        "source_subindex",
        "user_text",
        "width",
    }
)
_QUALITY_EXCLUSION_FIELDS = frozenset(
    {
        "family",
        "reason",
        "source_row_index",
        "source_subindex",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "artifact_sha256",
        "family_partition_counts",
        "quality_exclusions",
        "records",
        "schema",
        "sources",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ContractError(f"public native VL {name} must be nonempty text")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ContractError(f"public native VL {name} must be one lowercase SHA-256")
    return text


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"public native VL {name} must be a nonnegative integer")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    result = _require_nonnegative_integer(value, name=name)
    if result <= 0:
        raise ContractError(f"public native VL {name} must be positive")
    return result


def _relative_path(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if "\\" in text or "\0" in text:
        raise ContractError(f"public native VL {name} must use canonical POSIX syntax")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ContractError(f"public native VL {name} must be normalized and relative")
    return text


def native_vl_rgb_sha256(image: NDArray[np.uint8]) -> str:
    """Hash decoded pixels with shape/dtype binding, independent of file encoding."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ContractError("public native VL RGB hash requires HWC uint8 RGB")
    header = _canonical_bytes(
        {"dtype": "uint8", "height": int(array.shape[0]), "width": int(array.shape[1])}
    )
    digest = hashlib.sha256(b"picf-next.public-native-vl-rgb.v1\0")
    digest.update(header)
    digest.update(memoryview(np.ascontiguousarray(array)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NativeVLInstructionRecord:
    """One generic image/instruction/assistant record with no privileged model input."""

    record_id: str
    family: PublicNativeVLFamily
    user_text: str
    assistant_text: str
    image: NDArray[np.uint8]

    def __post_init__(self) -> None:
        _require_text(self.record_id, name="runtime record ID")
        if self.family not in PUBLIC_NATIVE_VL_FAMILIES:
            raise ContractError("public native VL runtime family is unsupported")
        _require_text(self.user_text, name="user text")
        _require_text(self.assistant_text, name="assistant text")
        if (
            not isinstance(self.image, np.ndarray)
            or self.image.dtype != np.uint8
            or self.image.ndim != 3
            or self.image.shape[2] != 3
            or self.image.shape[0] <= 0
            or self.image.shape[1] <= 0
            or self.image.flags.writeable
        ):
            raise ContractError("public native VL runtime image must be immutable HWC uint8 RGB")

    def qwen_user_messages(self, image_value: object | None = None) -> list[dict[str, Any]]:
        visible_image = self.image if image_value is None else image_value
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": visible_image},
                    {"type": "text", "text": self.user_text},
                ],
            }
        ]

    def qwen_messages(self, image_value: object | None = None) -> list[dict[str, Any]]:
        return [
            *self.qwen_user_messages(image_value=image_value),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.assistant_text}],
            },
        ]


@dataclass(frozen=True, slots=True)
class PublicNativeVLSource:
    dataset_id: str
    dataset_revision: str
    split: str
    source_file: str
    source_file_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.dataset_id, name="source dataset ID")
        _require_text(self.dataset_revision, name="source dataset revision")
        if self.split != "train":
            raise ContractError("public native VL retention may use only a train split")
        _relative_path(self.source_file, name="source file")
        _require_sha256(self.source_file_sha256, name="source file SHA-256")

    @classmethod
    def from_dict(cls, value: object) -> PublicNativeVLSource:
        if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
            raise ContractError("public native VL source fields differ from schema")
        return cls(
            dataset_id=_require_text(value["dataset_id"], name="source dataset ID"),
            dataset_revision=_require_text(
                value["dataset_revision"], name="source dataset revision"
            ),
            split=_require_text(value["split"], name="source split"),
            source_file=_relative_path(value["source_file"], name="source file"),
            source_file_sha256=_require_sha256(
                value["source_file_sha256"], name="source file SHA-256"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "source_file": self.source_file,
            "source_file_sha256": self.source_file_sha256,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class PublicNativeVLQualityExclusion:
    family: PublicNativeVLFamily
    source_row_index: int
    source_subindex: int
    reason: str

    def __post_init__(self) -> None:
        if self.family not in PUBLIC_NATIVE_VL_FAMILIES:
            raise ContractError("public native VL quality exclusion family is unsupported")
        _require_nonnegative_integer(
            self.source_row_index,
            name="quality exclusion source row index",
        )
        _require_nonnegative_integer(
            self.source_subindex,
            name="quality exclusion source subindex",
        )
        _require_text(self.reason, name="quality exclusion reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "reason": self.reason,
            "source_row_index": self.source_row_index,
            "source_subindex": self.source_subindex,
        }

    @classmethod
    def from_dict(cls, value: object) -> PublicNativeVLQualityExclusion:
        if not isinstance(value, Mapping) or set(value) != _QUALITY_EXCLUSION_FIELDS:
            raise ContractError("public native VL quality exclusion fields differ from schema")
        return cls(
            family=cast(PublicNativeVLFamily, value["family"]),
            source_row_index=_require_nonnegative_integer(
                value["source_row_index"],
                name="quality exclusion source row index",
            ),
            source_subindex=_require_nonnegative_integer(
                value["source_subindex"],
                name="quality exclusion source subindex",
            ),
            reason=_require_text(value["reason"], name="quality exclusion reason"),
        )


@dataclass(frozen=True, slots=True)
class PublicNativeVLManifestRecord:
    record_id: str
    family: PublicNativeVLFamily
    partition: PublicNativeVLPartition
    source_key: str
    source_row_index: int
    source_subindex: int
    priority_sha256: str
    user_text: str
    assistant_text: str
    image_file: str
    image_file_sha256: str
    image_rgb_sha256: str
    image_size_bytes: int
    width: int
    height: int
    record_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.record_id, name="record ID")
        if self.family not in PUBLIC_NATIVE_VL_FAMILIES:
            raise ContractError("public native VL manifest family is unsupported")
        if self.partition not in PUBLIC_NATIVE_VL_PARTITIONS:
            raise ContractError("public native VL partition is unsupported")
        _require_text(self.source_key, name="source key")
        _require_nonnegative_integer(self.source_row_index, name="source row index")
        _require_nonnegative_integer(self.source_subindex, name="source subindex")
        _require_sha256(self.priority_sha256, name="priority SHA-256")
        _require_text(self.user_text, name="user text")
        _require_text(self.assistant_text, name="assistant text")
        _relative_path(self.image_file, name="image path")
        _require_sha256(self.image_file_sha256, name="image file SHA-256")
        _require_sha256(self.image_rgb_sha256, name="image RGB SHA-256")
        _require_positive_integer(self.image_size_bytes, name="image size")
        _require_positive_integer(self.width, name="image width")
        _require_positive_integer(self.height, name="image height")
        _require_sha256(self.record_sha256, name="record SHA-256")
        if self.record_sha256 != self.computed_sha256:
            raise ContractError("public native VL record SHA-256 changed")

    @classmethod
    def create(
        cls,
        *,
        record_id: str,
        family: PublicNativeVLFamily,
        partition: PublicNativeVLPartition,
        source_key: str,
        source_row_index: int,
        source_subindex: int,
        priority_sha256: str,
        user_text: str,
        assistant_text: str,
        image_file: str,
        image_file_sha256: str,
        image_rgb_sha256: str,
        image_size_bytes: int,
        width: int,
        height: int,
    ) -> PublicNativeVLManifestRecord:
        payload: dict[str, object] = {
            "assistant_text": assistant_text,
            "family": family,
            "height": height,
            "image_file": image_file,
            "image_file_sha256": image_file_sha256,
            "image_rgb_sha256": image_rgb_sha256,
            "image_size_bytes": image_size_bytes,
            "partition": partition,
            "priority_sha256": priority_sha256,
            "record_id": record_id,
            "source_key": source_key,
            "source_row_index": source_row_index,
            "source_subindex": source_subindex,
            "user_text": user_text,
            "width": width,
        }
        return cls(
            record_id=record_id,
            family=family,
            partition=partition,
            source_key=source_key,
            source_row_index=source_row_index,
            source_subindex=source_subindex,
            priority_sha256=priority_sha256,
            user_text=user_text,
            assistant_text=assistant_text,
            image_file=image_file,
            image_file_sha256=image_file_sha256,
            image_rgb_sha256=image_rgb_sha256,
            image_size_bytes=image_size_bytes,
            width=width,
            height=height,
            record_sha256=hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        )

    @property
    def computed_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._payload())).hexdigest()

    def _payload(self) -> dict[str, object]:
        return {
            "assistant_text": self.assistant_text,
            "family": self.family,
            "height": self.height,
            "image_file": self.image_file,
            "image_file_sha256": self.image_file_sha256,
            "image_rgb_sha256": self.image_rgb_sha256,
            "image_size_bytes": self.image_size_bytes,
            "partition": self.partition,
            "priority_sha256": self.priority_sha256,
            "record_id": self.record_id,
            "source_key": self.source_key,
            "source_row_index": self.source_row_index,
            "source_subindex": self.source_subindex,
            "user_text": self.user_text,
            "width": self.width,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "record_sha256": self.record_sha256}

    @classmethod
    def from_dict(cls, value: object) -> PublicNativeVLManifestRecord:
        if not isinstance(value, Mapping) or set(value) != _RECORD_FIELDS:
            raise ContractError("public native VL record fields differ from schema")
        return cls(
            record_id=_require_text(value["record_id"], name="record ID"),
            family=cast(PublicNativeVLFamily, value["family"]),
            partition=cast(PublicNativeVLPartition, value["partition"]),
            source_key=_require_text(value["source_key"], name="source key"),
            source_row_index=_require_nonnegative_integer(
                value["source_row_index"], name="source row index"
            ),
            source_subindex=_require_nonnegative_integer(
                value["source_subindex"], name="source subindex"
            ),
            priority_sha256=_require_sha256(value["priority_sha256"], name="priority SHA-256"),
            user_text=_require_text(value["user_text"], name="user text"),
            assistant_text=_require_text(value["assistant_text"], name="assistant text"),
            image_file=_relative_path(value["image_file"], name="image path"),
            image_file_sha256=_require_sha256(
                value["image_file_sha256"], name="image file SHA-256"
            ),
            image_rgb_sha256=_require_sha256(value["image_rgb_sha256"], name="image RGB SHA-256"),
            image_size_bytes=_require_positive_integer(
                value["image_size_bytes"], name="image size"
            ),
            width=_require_positive_integer(value["width"], name="image width"),
            height=_require_positive_integer(value["height"], name="image height"),
            record_sha256=_require_sha256(value["record_sha256"], name="record SHA-256"),
        )


@dataclass(frozen=True, slots=True)
class PublicNativeVLRetentionManifest:
    sources: Mapping[str, PublicNativeVLSource]
    records: tuple[PublicNativeVLManifestRecord, ...]
    artifact_sha256: str
    quality_exclusions: tuple[PublicNativeVLQualityExclusion, ...] = ()

    def __post_init__(self) -> None:
        sources = dict(self.sources)
        object.__setattr__(self, "sources", MappingProxyType(sources))
        if set(self.sources) != set(PUBLIC_NATIVE_VL_FAMILIES):
            raise ContractError("public native VL manifest requires referring and VQA sources")
        if not self.records:
            raise ContractError("public native VL manifest has no records")
        ids = tuple(record.record_id for record in self.records)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ContractError("public native VL record IDs must be unique and sorted")
        if any(record.source_key != record.family for record in self.records):
            raise ContractError("public native VL record source and family differ")
        source_locations = tuple(
            (record.source_key, record.source_row_index, record.source_subindex)
            for record in self.records
        )
        if len(source_locations) != len(set(source_locations)):
            raise ContractError("public native VL source records must be unique")
        exclusion_locations = tuple(
            (item.family, item.source_row_index, item.source_subindex)
            for item in self.quality_exclusions
        )
        if exclusion_locations != tuple(sorted(exclusion_locations)) or len(
            exclusion_locations
        ) != len(set(exclusion_locations)):
            raise ContractError("public native VL quality exclusions must be unique and sorted")
        if set(source_locations) & set(exclusion_locations):
            raise ContractError("public native VL selected and excluded source records overlap")
        image_files = tuple(record.image_file for record in self.records)
        if len(image_files) != len(set(image_files)):
            raise ContractError("public native VL image artifact paths must be unique")
        global_train_images = {
            record.image_rgb_sha256 for record in self.records if record.partition == "train"
        }
        global_heldout_images = {
            record.image_rgb_sha256 for record in self.records if record.partition == "heldout"
        }
        if global_train_images & global_heldout_images:
            raise ContractError(
                "public native VL train and heldout images overlap across task families"
            )
        for family in PUBLIC_NATIVE_VL_FAMILIES:
            train = self.records_for(cast(PublicNativeVLFamily, family), "train")
            heldout = self.records_for(cast(PublicNativeVLFamily, family), "heldout")
            if not train or not heldout:
                raise ContractError("public native VL each family needs train and heldout records")
            train_images = {record.image_rgb_sha256 for record in train}
            heldout_images = {record.image_rgb_sha256 for record in heldout}
            if len(train_images) != len(train) or len(heldout_images) != len(heldout):
                raise ContractError("public native VL images must be unique within partitions")
            if train_images & heldout_images:
                raise ContractError("public native VL train and heldout images overlap")
        _require_sha256(self.artifact_sha256, name="manifest artifact SHA-256")
        if self.artifact_sha256 != self.computed_sha256:
            raise ContractError("public native VL manifest artifact SHA-256 changed")

    @property
    def family_partition_counts(self) -> dict[str, int]:
        return {
            f"{family}/{partition}": len(
                self.records_for(
                    cast(PublicNativeVLFamily, family),
                    cast(PublicNativeVLPartition, partition),
                )
            )
            for family in PUBLIC_NATIVE_VL_FAMILIES
            for partition in PUBLIC_NATIVE_VL_PARTITIONS
        }

    def records_for(
        self,
        family: PublicNativeVLFamily,
        partition: PublicNativeVLPartition,
    ) -> tuple[PublicNativeVLManifestRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.family == family and record.partition == partition
        )

    def training_record_for_rank(
        self,
        *,
        optimizer_step: int,
        rank: int,
    ) -> PublicNativeVLManifestRecord:
        _require_nonnegative_integer(optimizer_step, name="optimizer step")
        if rank not in (0, 1):
            raise ContractError("public native VL mixed gate requires rank zero or one")
        family: PublicNativeVLFamily = "referring" if rank == 0 else "vqa"
        records = self.records_for(family, "train")
        if optimizer_step >= len(records):
            raise ContractError("public native VL training step exceeds its frozen records")
        return records[optimizer_step]

    def _payload(self) -> dict[str, object]:
        return {
            "family_partition_counts": self.family_partition_counts,
            "quality_exclusions": [item.to_dict() for item in self.quality_exclusions],
            "records": [record.to_dict() for record in self.records],
            "schema": PUBLIC_NATIVE_VL_MANIFEST_SCHEMA,
            "sources": {key: self.sources[key].to_dict() for key in sorted(self.sources)},
        }

    @property
    def computed_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self._payload())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(cls, value: object) -> PublicNativeVLRetentionManifest:
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
            raise ContractError("public native VL manifest fields differ from schema")
        if value["schema"] != PUBLIC_NATIVE_VL_MANIFEST_SCHEMA:
            raise ContractError("public native VL manifest schema changed")
        raw_sources = value["sources"]
        raw_records = value["records"]
        raw_exclusions = value["quality_exclusions"]
        if (
            not isinstance(raw_sources, Mapping)
            or not isinstance(raw_records, list)
            or not isinstance(raw_exclusions, list)
        ):
            raise ContractError("public native VL manifest sources or records are malformed")
        manifest = cls(
            sources={
                _require_text(key, name="source key"): PublicNativeVLSource.from_dict(source)
                for key, source in raw_sources.items()
            },
            records=tuple(PublicNativeVLManifestRecord.from_dict(item) for item in raw_records),
            artifact_sha256=_require_sha256(
                value["artifact_sha256"], name="manifest artifact SHA-256"
            ),
            quality_exclusions=tuple(
                PublicNativeVLQualityExclusion.from_dict(item) for item in raw_exclusions
            ),
        )
        if value["family_partition_counts"] != manifest.family_partition_counts:
            raise ContractError("public native VL manifest counts changed")
        return manifest

    @classmethod
    def create(
        cls,
        *,
        sources: Mapping[str, PublicNativeVLSource],
        records: tuple[PublicNativeVLManifestRecord, ...],
        quality_exclusions: tuple[PublicNativeVLQualityExclusion, ...] = (),
    ) -> PublicNativeVLRetentionManifest:
        source_copy = dict(sources)
        counts = {
            f"{family}/{partition}": sum(
                record.family == family and record.partition == partition for record in records
            )
            for family in PUBLIC_NATIVE_VL_FAMILIES
            for partition in PUBLIC_NATIVE_VL_PARTITIONS
        }
        payload = {
            "family_partition_counts": counts,
            "quality_exclusions": [item.to_dict() for item in quality_exclusions],
            "records": [record.to_dict() for record in records],
            "schema": PUBLIC_NATIVE_VL_MANIFEST_SCHEMA,
            "sources": {key: source_copy[key].to_dict() for key in sorted(source_copy)},
        }
        return cls(
            sources=source_copy,
            records=records,
            artifact_sha256=hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
            quality_exclusions=quality_exclusions,
        )

    @classmethod
    def load(cls, path: str | Path) -> PublicNativeVLRetentionManifest:
        manifest_path = Path(path)
        try:
            value = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractError("public native VL manifest is not valid ASCII JSON") from error
        return cls.from_dict(value)

    def materialize_record(
        self,
        record: PublicNativeVLManifestRecord,
        *,
        artifact_root: str | Path,
    ) -> NativeVLInstructionRecord:
        if record not in self.records:
            raise ContractError("public native VL record is absent from its manifest")
        payload = read_sha256_verified_file_beneath(
            artifact_root,
            record.image_file,
            expected_sha256=record.image_file_sha256,
            maximum_bytes=PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_BYTES,
        )
        if len(payload) != record.image_size_bytes:
            raise ContractError("public native VL image size differs from manifest")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                source_width, source_height = source.size
                if (
                    source_width <= 0
                    or source_height <= 0
                    or source_width * source_height > PUBLIC_NATIVE_VL_MAXIMUM_IMAGE_PIXELS
                ):
                    raise ContractError("public native VL decoded image exceeds the pixel limit")
                source.load()
                image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        except ContractError:
            raise
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ContractError("public native VL image cannot be decoded") from error
        if image.shape != (record.height, record.width, 3):
            raise ContractError("public native VL decoded image shape differs from manifest")
        immutable = np.ascontiguousarray(image).copy()
        immutable.setflags(write=False)
        if native_vl_rgb_sha256(immutable) != record.image_rgb_sha256:
            raise ContractError("public native VL decoded image differs from manifest")
        return NativeVLInstructionRecord(
            record_id=record.record_id,
            family=record.family,
            user_text=record.user_text,
            assistant_text=record.assistant_text,
            image=immutable,
        )


def validate_frozen_public_native_vl_retention_gate(
    manifest: PublicNativeVLRetentionManifest,
    *,
    max_steps: int,
) -> PublicNativeVLRetentionManifest:
    """Bind training to ADR-125's exact sources, subset cardinality and ordering."""

    if not isinstance(manifest, PublicNativeVLRetentionManifest):
        raise TypeError("public native VL frozen gate requires its typed manifest")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not (1 <= max_steps <= PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY)
    ):
        raise ContractError("public native VL frozen gate step count is invalid")
    expected_sources = {
        "referring": PublicNativeVLSource(
            dataset_id=PUBLIC_NATIVE_VL_REFERRING_DATASET_ID,
            dataset_revision=PUBLIC_NATIVE_VL_REFERRING_REVISION,
            split="train",
            source_file=PUBLIC_NATIVE_VL_REFERRING_SOURCE_FILE,
            source_file_sha256=PUBLIC_NATIVE_VL_REFERRING_SOURCE_SHA256,
        ),
        "vqa": PublicNativeVLSource(
            dataset_id=PUBLIC_NATIVE_VL_VQA_DATASET_ID,
            dataset_revision=PUBLIC_NATIVE_VL_VQA_REVISION,
            split="train",
            source_file=PUBLIC_NATIVE_VL_VQA_SOURCE_FILE,
            source_file_sha256=PUBLIC_NATIVE_VL_VQA_SOURCE_SHA256,
        ),
    }
    if dict(manifest.sources) != expected_sources:
        raise ContractError("public native VL frozen gate sources changed")
    expected_counts = {
        "referring/heldout": PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY,
        "referring/train": PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY,
        "vqa/heldout": PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY,
        "vqa/train": PUBLIC_NATIVE_VL_TRAIN_RECORDS_PER_FAMILY,
    }
    if manifest.family_partition_counts != expected_counts:
        raise ContractError("public native VL frozen gate record counts changed")
    expected_exclusions = (
        PublicNativeVLQualityExclusion(
            family="referring",
            source_row_index=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[0],
            source_subindex=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[1],
            reason=PUBLIC_NATIVE_VL_REFERRING_QUALITY_EXCLUSION[2],
        ),
    )
    if manifest.quality_exclusions != expected_exclusions:
        raise ContractError("public native VL frozen gate quality exclusions changed")
    for family in PUBLIC_NATIVE_VL_FAMILIES:
        for partition in PUBLIC_NATIVE_VL_PARTITIONS:
            records = manifest.records_for(
                cast(PublicNativeVLFamily, family),
                cast(PublicNativeVLPartition, partition),
            )
            priorities = tuple(record.priority_sha256 for record in records)
            if priorities != tuple(sorted(priorities)) or len(priorities) != len(set(priorities)):
                raise ContractError("public native VL frozen gate priority order changed")
    return manifest


def load_frozen_public_native_vl_retention_gate(
    *,
    manifest_path: str | Path,
    manifest_file_sha256: str,
    artifact_root: str | Path,
    max_steps: int,
) -> PublicNativeVLRetentionManifest:
    """Load and fully materialize the one preregistered retention artifact."""

    expected_sha256 = _require_sha256(
        manifest_file_sha256,
        name="manifest file SHA-256",
    )
    path = Path(manifest_path)
    root = Path(artifact_root)
    if path.is_symlink() or not path.is_file():
        raise ContractError("public native VL manifest must be a regular file")
    if root.is_symlink() or not root.is_dir():
        raise ContractError("public native VL artifact root must be a directory")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ContractError("public native VL manifest cannot be read") from error
    if not payload or len(payload) > PUBLIC_NATIVE_VL_MAXIMUM_MANIFEST_BYTES:
        raise ContractError("public native VL manifest file size is invalid")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContractError("public native VL manifest file changed")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("public native VL manifest is not valid ASCII JSON") from error
    manifest = validate_frozen_public_native_vl_retention_gate(
        PublicNativeVLRetentionManifest.from_dict(value),
        max_steps=max_steps,
    )
    for record in manifest.records:
        manifest.materialize_record(record, artifact_root=root)
    return manifest
