"""Content-addressed frozen V-JEPA2 token cache for causal action context."""

from __future__ import annotations

import io
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data.causal_video import CausalVideoClip
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath
from picf_next.encoders.vjepa2 import (
    VJEPA2_MODEL_ID,
    VJEPA2_MODEL_REVISION,
    vjepa2_context_only_role,
    vjepa2_dense_geometry,
    vjepa2_dense_timestamps,
)

VJEPA2_CACHE_SCHEMA = "picf-next.vjepa2-causal-token-cache/v1"
VJEPA2_CACHE_AUGMENTATION = "identity-source-rgb/v1"
VJEPA2_CONTEXT_SENSORS = (
    ("observation.images.rgb_static", "vjepa_static"),
    ("observation.images.rgb_gripper", "vjepa_gripper"),
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


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
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
        raise ContractError(f"{name} must be one normalized relative POSIX path")
    return text


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class Vjepa2CachedSensorEntry:
    sensor_key: str
    modality: str
    source_frame_sha256: tuple[str, ...]
    token_count: int
    artifact_path: str | None
    artifact_sha256: str | None


@dataclass(frozen=True, slots=True)
class Vjepa2CachedSampleEntry:
    sample_key: str
    sensors: tuple[Vjepa2CachedSensorEntry, ...]


class Vjepa2FeatureCache:
    """Lazy, bounded reader that revalidates source clips before token use."""

    def __init__(
        self,
        *,
        root: Path,
        dataset_tree_sha256: str,
        encoder_contract: str,
        hidden_size: int,
        image_size: int,
        tubelet_size: int,
        patch_size: int,
        maximum_frames: int,
        entries: tuple[Vjepa2CachedSampleEntry, ...],
        memory_capacity: int,
    ) -> None:
        self.root = root
        self.dataset_tree_sha256 = dataset_tree_sha256
        self.encoder_contract = encoder_contract
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.maximum_frames = maximum_frames
        self.entries = {entry.sample_key: entry for entry in entries}
        self.memory_capacity = memory_capacity
        self._tokens: OrderedDict[str, NDArray[np.float32]] = OrderedDict()

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        manifest_sha256: str,
        dataset_tree_sha256: str,
        memory_capacity: int = 64,
    ) -> Vjepa2FeatureCache:
        """Load one complete immutable manifest and reject stale cache roots."""

        expected_manifest_sha = _sha256(manifest_sha256, "cache manifest sha256")
        expected_dataset_sha = _sha256(dataset_tree_sha256, "dataset tree sha256")
        memory_capacity = _positive_int(memory_capacity, "cache memory capacity")
        resolved_root = Path(root).resolve()
        manifest_bytes = read_sha256_verified_file_beneath(
            resolved_root,
            "manifest.json",
            expected_sha256=expected_manifest_sha,
            maximum_bytes=32 * 1024 * 1024,
        )
        try:
            raw = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("V-JEPA2 cache manifest is not valid JSON") from exc
        payload = _exact(
            raw,
            "V-JEPA2 cache manifest",
            {
                "augmentation_contract",
                "complete",
                "dataset_tree_sha256",
                "encoder",
                "entries",
                "expected_entries",
                "schema",
                "sensors",
            },
        )
        if payload["schema"] != VJEPA2_CACHE_SCHEMA:
            raise ContractError("V-JEPA2 cache schema changed")
        if payload["complete"] is not True:
            raise ContractError("V-JEPA2 cache manifest is incomplete")
        if payload["augmentation_contract"] != VJEPA2_CACHE_AUGMENTATION:
            raise ContractError("V-JEPA2 cache augmentation contract changed")
        observed_dataset_sha = _sha256(
            payload["dataset_tree_sha256"], "manifest dataset tree sha256"
        )
        if observed_dataset_sha != expected_dataset_sha:
            raise ContractError("V-JEPA2 cache belongs to a different dataset tree")

        encoder = _exact(
            payload["encoder"],
            "V-JEPA2 cache encoder",
            {
                "checkpoint_revision",
                "encoder_contract",
                "hidden_size",
                "image_size",
                "maximum_frames",
                "model_id",
                "patch_size",
                "tubelet_size",
            },
        )
        if encoder["model_id"] != VJEPA2_MODEL_ID:
            raise ContractError("V-JEPA2 cache uses an unapproved model id")
        if encoder["checkpoint_revision"] != VJEPA2_MODEL_REVISION:
            raise ContractError("V-JEPA2 cache uses an unapproved checkpoint revision")
        encoder_contract = _text(encoder["encoder_contract"], "encoder contract")
        if not encoder_contract.startswith(f"{VJEPA2_MODEL_ID}@{VJEPA2_MODEL_REVISION}/"):
            raise ContractError("V-JEPA2 encoder contract does not bind its checkpoint")
        hidden_size = _positive_int(encoder["hidden_size"], "V-JEPA2 hidden size")
        image_size = _positive_int(encoder["image_size"], "V-JEPA2 image size")
        tubelet_size = _positive_int(encoder["tubelet_size"], "V-JEPA2 tubelet size")
        patch_size = _positive_int(encoder["patch_size"], "V-JEPA2 patch size")
        maximum_frames = _positive_int(encoder["maximum_frames"], "maximum frames")
        if maximum_frames % tubelet_size or image_size % patch_size:
            raise ContractError("V-JEPA2 cache dimensions do not form complete patches")

        raw_sensors = payload["sensors"]
        if not isinstance(raw_sensors, list):
            raise ContractError("V-JEPA2 cache sensors must be a list")
        sensors = tuple(
            (
                _text(
                    _exact(item, f"sensor[{index}]", {"modality", "sensor_key"})["sensor_key"],
                    "sensor key",
                ),
                _text(_mapping(item, "sensor")["modality"], "sensor modality"),
            )
            for index, item in enumerate(raw_sensors)
        )
        if sensors != VJEPA2_CONTEXT_SENSORS:
            raise ContractError("V-JEPA2 cache sensor/modality order changed")

        raw_entries = payload["entries"]
        if not isinstance(raw_entries, list):
            raise ContractError("V-JEPA2 cache entries must be a list")
        expected_entries = _positive_int(payload["expected_entries"], "expected entries")
        if len(raw_entries) != expected_entries:
            raise ContractError("V-JEPA2 cache did not finish every expected sample")
        entries = tuple(
            cls._parse_entry(
                item,
                index=index,
                image_size=image_size,
                tubelet_size=tubelet_size,
                patch_size=patch_size,
                maximum_frames=maximum_frames,
            )
            for index, item in enumerate(raw_entries)
        )
        keys = tuple(entry.sample_key for entry in entries)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ContractError("V-JEPA2 cache sample keys must be sorted and unique")
        return cls(
            root=resolved_root,
            dataset_tree_sha256=observed_dataset_sha,
            encoder_contract=encoder_contract,
            hidden_size=hidden_size,
            image_size=image_size,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            maximum_frames=maximum_frames,
            entries=entries,
            memory_capacity=memory_capacity,
        )

    @staticmethod
    def _parse_entry(
        raw: object,
        *,
        index: int,
        image_size: int,
        tubelet_size: int,
        patch_size: int,
        maximum_frames: int,
    ) -> Vjepa2CachedSampleEntry:
        payload = _exact(raw, f"cache entry[{index}]", {"sample_key", "sensors"})
        sample_key = _text(payload["sample_key"], "cache sample key")
        raw_sensors = payload["sensors"]
        if not isinstance(raw_sensors, list) or len(raw_sensors) != len(VJEPA2_CONTEXT_SENSORS):
            raise ContractError("each cache entry must contain both ordered camera streams")
        sensors = []
        patches_per_frame = (image_size // patch_size) ** 2
        for sensor_index, (raw_sensor, expected) in enumerate(
            zip(raw_sensors, VJEPA2_CONTEXT_SENSORS, strict=True)
        ):
            sensor = _exact(
                raw_sensor,
                f"cache entry[{index}].sensor[{sensor_index}]",
                {
                    "artifact_path",
                    "artifact_sha256",
                    "modality",
                    "sensor_key",
                    "source_frame_sha256",
                    "token_count",
                },
            )
            sensor_key = _text(sensor["sensor_key"], "cached sensor key")
            modality = _text(sensor["modality"], "cached sensor modality")
            if (sensor_key, modality) != expected:
                raise ContractError("cached sensor identity or order changed")
            raw_hashes = sensor["source_frame_sha256"]
            if not isinstance(raw_hashes, list):
                raise ContractError("cached source hashes must be a list")
            source_hashes = tuple(_sha256(value, "source frame sha256") for value in raw_hashes)
            if len(set(source_hashes)) != len(source_hashes):
                raise ContractError("cached causal clip repeats a source frame")
            if len(source_hashes) > maximum_frames or (
                source_hashes and len(source_hashes) % tubelet_size
            ):
                raise ContractError("cached clip length violates the tubelet contract")
            token_count = _nonnegative_int(sensor["token_count"], "cached token count")
            expected_tokens = len(source_hashes) // tubelet_size * patches_per_frame
            if token_count != expected_tokens:
                raise ContractError("cached token count differs from its source clip")
            artifact_path = sensor["artifact_path"]
            artifact_sha = sensor["artifact_sha256"]
            if token_count == 0:
                if artifact_path is not None or artifact_sha is not None:
                    raise ContractError("an unavailable clip cannot carry a token artifact")
                normalized_path = None
                normalized_sha = None
            else:
                normalized_path = _relative_path(artifact_path, "cached token artifact path")
                normalized_sha = _sha256(artifact_sha, "cached token artifact sha256")
            sensors.append(
                Vjepa2CachedSensorEntry(
                    sensor_key=sensor_key,
                    modality=modality,
                    source_frame_sha256=source_hashes,
                    token_count=token_count,
                    artifact_path=normalized_path,
                    artifact_sha256=normalized_sha,
                )
            )
        return Vjepa2CachedSampleEntry(sample_key=sample_key, sensors=tuple(sensors))

    def _load_tokens(self, entry: Vjepa2CachedSensorEntry) -> NDArray[np.float32]:
        if entry.artifact_path is None or entry.artifact_sha256 is None:
            raise RuntimeError("unavailable V-JEPA2 cache entry has no token artifact")
        cached = self._tokens.get(entry.artifact_sha256)
        if cached is not None:
            self._tokens.move_to_end(entry.artifact_sha256)
            return cached
        maximum_bytes = entry.token_count * self.hidden_size * 4 + 4096
        payload = read_sha256_verified_file_beneath(
            self.root,
            entry.artifact_path,
            expected_sha256=entry.artifact_sha256,
            maximum_bytes=maximum_bytes,
        )
        try:
            array = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ContractError("cached V-JEPA2 token artifact is not a safe NPY array") from exc
        if (
            not isinstance(array, np.ndarray)
            or array.shape != (entry.token_count, self.hidden_size)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ContractError("cached V-JEPA2 token tensor changed shape, dtype or finiteness")
        array.setflags(write=False)
        self._tokens[entry.artifact_sha256] = array
        while len(self._tokens) > self.memory_capacity:
            self._tokens.popitem(last=False)
        return array

    def evidence_for(
        self,
        sample_key: str,
        clips_by_sensor: Mapping[str, CausalVideoClip | None],
    ) -> tuple[DenseEvidence, ...]:
        """Return cached tokens only after exact causal-source revalidation."""

        try:
            entry = self.entries[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown V-JEPA2 cache sample key {sample_key!r}") from exc
        if set(clips_by_sensor) != {sensor_key for sensor_key, _ in VJEPA2_CONTEXT_SENSORS}:
            raise ContractError("runtime V-JEPA2 clips must contain both configured cameras")
        evidence = []
        for sensor in entry.sensors:
            clip = clips_by_sensor[sensor.sensor_key]
            observed_hashes = () if clip is None else clip.source_frame_sha256
            if observed_hashes != sensor.source_frame_sha256:
                raise ContractError("runtime causal clip differs from the frozen V-JEPA2 cache")
            if clip is None:
                tokens = np.empty((0, self.hidden_size), dtype=np.float32)
                geometry = np.empty((0, 3), dtype=np.float32)
                timestamps = np.empty(0, dtype=np.float32)
                confidence = np.empty(0, dtype=np.float32)
                current = np.empty(0, dtype=np.bool_)
                for array in (tokens, geometry, timestamps, confidence, current):
                    array.setflags(write=False)
            else:
                tokens = self._load_tokens(sensor)
                geometry = vjepa2_dense_geometry(
                    frame_count=len(clip.images),
                    image_height=self.image_size,
                    image_width=self.image_size,
                    tubelet_size=self.tubelet_size,
                    patch_size=self.patch_size,
                )
                timestamps = vjepa2_dense_timestamps(
                    clip.frame_timestamps_s,
                    tubelet_size=self.tubelet_size,
                    patches_per_frame=(self.image_size // self.patch_size) ** 2,
                )
                confidence = np.ones(sensor.token_count, dtype=np.float32)
                confidence.setflags(write=False)
                current = vjepa2_context_only_role(sensor.token_count)
            evidence.append(
                DenseEvidence(
                    modality=sensor.modality,
                    encoder_contract=self.encoder_contract,
                    tokens=tokens,
                    available=clip is not None,
                    timestamps=timestamps,
                    confidence=confidence,
                    geometry=geometry,
                    current_measurement_valid=current,
                )
            )
        return tuple(evidence)
