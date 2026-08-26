"""Reproducible CALVIN DIGIT background calibration.

The calibration boundary estimates only the optical no-deformation reference
used by the released AnyTouch2 preprocessor.  It does not infer contact,
object identity, ownership, task relevance, or posterior lifecycle.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.calvin_tactile import (
    CALVIN_TACTILE_SOURCE_COMMIT,
    CALVIN_TACTILE_SOURCE_FILES_SHA256,
    CALVIN_TACTILE_STREAM_NAMES,
)
from picf_next.data.dataset_manifest import read_sha256_verified_file_beneath

CALVIN_TACTILE_CALIBRATION_SCHEMA = "picf-next.calvin-digit-background-calibration/v1"
CALVIN_TACTILE_BACKGROUND_ALGORITHM = "per-pixel-channel-median/v1"
CALVIN_TACTILE_BACKGROUND_ARCHIVE_SCHEMA = "picf-next.calvin-digit-background-archive/v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _readonly(value: NDArray[np.generic]) -> NDArray[np.generic]:
    contiguous = np.ascontiguousarray(value)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    return output


def tactile_background_sha256(value: NDArray[np.generic]) -> str:
    background = np.asarray(value)
    if background.shape != (160, 120, 3) or background.dtype != np.float32:
        raise ContractError("calibrated tactile backgrounds must be 160-by-120-by-3 float32")
    if not np.isfinite(background).all() or (background < 0.0).any() or (background > 255.0).any():
        raise ContractError("calibrated tactile backgrounds must contain finite RGB values")
    digest = hashlib.sha256(b"picf-next.calvin-tactile-background-array/v1\0")
    digest.update(b"<f4\0")
    digest.update(np.asarray(background.shape, dtype="<i8").tobytes(order="C"))
    digest.update(np.ascontiguousarray(background, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CalvinTactileCalibrationSample:
    source_global_index: int
    source_file_sha256: str
    rgb: NDArray[np.uint8]
    deformation_m: NDArray[np.float32]

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_global_index, bool)
            or not isinstance(self.source_global_index, int)
            or self.source_global_index < 0
        ):
            raise ContractError("tactile calibration source index must be nonnegative")
        _require_sha256(self.source_file_sha256, "tactile calibration source file")
        if self.rgb.shape != (160, 120, 6) or self.rgb.dtype != np.uint8:
            raise ContractError("tactile calibration RGB must be 160-by-120-by-6 uint8")
        if self.deformation_m.shape != (160, 120, 2) or self.deformation_m.dtype != np.float32:
            raise ContractError("tactile calibration deformation must be 160-by-120-by-2 float32")
        if not np.isfinite(self.deformation_m).all():
            raise ContractError("tactile calibration deformation must be finite")


@dataclass(frozen=True, slots=True)
class CalvinTactileBackgroundCalibration:
    backgrounds_by_stream: Mapping[str, NDArray[np.float32]]
    candidate_steps_by_stream: Mapping[str, tuple[int, ...]]
    selected_steps_by_stream: Mapping[str, tuple[int, ...]]
    selected_source_sha256_by_stream: Mapping[str, str]
    background_sha256_by_stream: Mapping[str, str]
    background_noise_ceiling_m: float
    validity_thresholds_m: Mapping[str, float]

    def __post_init__(self) -> None:
        expected = set(CALVIN_TACTILE_STREAM_NAMES)
        mappings = (
            self.backgrounds_by_stream,
            self.candidate_steps_by_stream,
            self.selected_steps_by_stream,
            self.selected_source_sha256_by_stream,
            self.background_sha256_by_stream,
            self.validity_thresholds_m,
        )
        if any(set(mapping) != expected for mapping in mappings):
            raise ContractError("tactile calibration mappings must cover both DIGIT streams")
        if (
            not math.isfinite(self.background_noise_ceiling_m)
            or self.background_noise_ceiling_m <= 0.0
        ):
            raise ContractError("tactile background noise ceiling must be finite and positive")
        for name in CALVIN_TACTILE_STREAM_NAMES:
            background = self.backgrounds_by_stream[name]
            if background.flags.writeable:
                raise ContractError("calibrated tactile backgrounds must be immutable")
            if tactile_background_sha256(background) != self.background_sha256_by_stream[name]:
                raise ContractError("calibrated tactile background hash changed")
            candidates = self.candidate_steps_by_stream[name]
            selected = self.selected_steps_by_stream[name]
            if (
                not candidates
                or not selected
                or tuple(sorted(set(candidates))) != candidates
                or tuple(sorted(set(selected))) != selected
                or not set(selected).issubset(candidates)
            ):
                raise ContractError("tactile calibration steps must be sorted nonempty subsets")
            _require_sha256(
                self.selected_source_sha256_by_stream[name],
                "selected tactile calibration sources",
            )
            threshold = float(self.validity_thresholds_m[name])
            if not math.isfinite(threshold) or threshold <= self.background_noise_ceiling_m:
                raise ContractError(
                    "tactile validity thresholds must exceed the background noise ceiling"
                )

    def receipt_payload(self) -> dict[str, object]:
        return {
            "algorithm": CALVIN_TACTILE_BACKGROUND_ALGORITHM,
            "background_noise_ceiling_m": self.background_noise_ceiling_m,
            "streams": {
                name: {
                    "background_sha256": self.background_sha256_by_stream[name],
                    "candidate_count": len(self.candidate_steps_by_stream[name]),
                    "candidate_steps_sha256": _sha256(
                        np.asarray(self.candidate_steps_by_stream[name], dtype="<i8").tobytes()
                    ),
                    "selected_count": len(self.selected_steps_by_stream[name]),
                    "selected_source_sha256": self.selected_source_sha256_by_stream[name],
                    "selected_steps": list(self.selected_steps_by_stream[name]),
                    "validity_threshold_m": float(self.validity_thresholds_m[name]),
                }
                for name in CALVIN_TACTILE_STREAM_NAMES
            },
        }


def build_calvin_tactile_background_calibration(
    samples: Iterable[CalvinTactileCalibrationSample],
    *,
    background_noise_ceiling_m: float,
    validity_thresholds_m: Mapping[str, float],
    minimum_candidates_per_stream: int = 16,
    maximum_selected_per_stream: int = 256,
) -> CalvinTactileBackgroundCalibration:
    """Estimate robust optical references from independently quiet sensors."""

    if (
        not isinstance(minimum_candidates_per_stream, int)
        or isinstance(minimum_candidates_per_stream, bool)
        or minimum_candidates_per_stream <= 0
        or not isinstance(maximum_selected_per_stream, int)
        or isinstance(maximum_selected_per_stream, bool)
        or maximum_selected_per_stream < minimum_candidates_per_stream
    ):
        raise ValueError("tactile calibration sample limits are inconsistent")
    if not math.isfinite(background_noise_ceiling_m) or background_noise_ceiling_m <= 0.0:
        raise ValueError("background_noise_ceiling_m must be finite and positive")
    if set(validity_thresholds_m) != set(CALVIN_TACTILE_STREAM_NAMES):
        raise ContractError("validity thresholds must cover both DIGIT streams")

    candidates: dict[str, list[tuple[int, str, NDArray[np.uint8]]]] = {
        name: [] for name in CALVIN_TACTILE_STREAM_NAMES
    }
    previous_index = -1
    for sample in samples:
        if not isinstance(sample, CalvinTactileCalibrationSample):
            raise TypeError("tactile calibration requires typed samples")
        if sample.source_global_index <= previous_index:
            raise ContractError("tactile calibration samples must be source-unique and sorted")
        previous_index = sample.source_global_index
        for sensor_index, name in enumerate(CALVIN_TACTILE_STREAM_NAMES):
            maximum = float(np.abs(sample.deformation_m[..., sensor_index]).max())
            if maximum <= background_noise_ceiling_m:
                candidates[name].append(
                    (
                        sample.source_global_index,
                        sample.source_file_sha256,
                        sample.rgb[..., 3 * sensor_index : 3 * (sensor_index + 1)],
                    )
                )

    backgrounds: dict[str, NDArray[np.float32]] = {}
    candidate_steps: dict[str, tuple[int, ...]] = {}
    selected_steps: dict[str, tuple[int, ...]] = {}
    selected_sources: dict[str, str] = {}
    background_hashes: dict[str, str] = {}
    for name in CALVIN_TACTILE_STREAM_NAMES:
        stream_candidates = candidates[name]
        if len(stream_candidates) < minimum_candidates_per_stream:
            raise ContractError(
                f"{name} has only {len(stream_candidates)} quiet frames; "
                f"requires {minimum_candidates_per_stream}"
            )
        count = min(maximum_selected_per_stream, len(stream_candidates))
        positions = np.linspace(0, len(stream_candidates) - 1, num=count, dtype=np.int64)
        selected = tuple(stream_candidates[int(position)] for position in positions)
        stack = np.stack([item[2] for item in selected], axis=0)
        background = _readonly(np.median(stack, axis=0).astype(np.float32))
        source_digest = hashlib.sha256(b"picf-next.calvin-tactile-background-sources/v1\0")
        for step, file_digest, _rgb in selected:
            source_digest.update(int(step).to_bytes(8, "big"))
            source_digest.update(bytes.fromhex(file_digest))
        backgrounds[name] = background
        candidate_steps[name] = tuple(item[0] for item in stream_candidates)
        selected_steps[name] = tuple(item[0] for item in selected)
        selected_sources[name] = source_digest.hexdigest()
        background_hashes[name] = tactile_background_sha256(background)

    return CalvinTactileBackgroundCalibration(
        backgrounds_by_stream=backgrounds,
        candidate_steps_by_stream=candidate_steps,
        selected_steps_by_stream=selected_steps,
        selected_source_sha256_by_stream=selected_sources,
        background_sha256_by_stream=background_hashes,
        background_noise_ceiling_m=float(background_noise_ceiling_m),
        validity_thresholds_m={
            name: float(validity_thresholds_m[name]) for name in CALVIN_TACTILE_STREAM_NAMES
        },
    )


def canonical_calibration_receipt_sha256(payload: Mapping[str, object]) -> str:
    return _sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )


@dataclass(frozen=True, slots=True)
class LoadedCalvinTactileBackgrounds:
    backgrounds_by_stream: Mapping[str, NDArray[np.float32]]
    validity_thresholds_m: Mapping[str, float]
    archive_sha256: str
    receipt_sha256: str
    receipt_payload_sha256: str
    dataset_tree_sha256: str

    def __post_init__(self) -> None:
        expected = set(CALVIN_TACTILE_STREAM_NAMES)
        if (
            set(self.backgrounds_by_stream) != expected
            or set(self.validity_thresholds_m) != expected
        ):
            raise ContractError("loaded tactile calibration must cover both DIGIT streams")
        for digest, name in (
            (self.archive_sha256, "tactile background archive"),
            (self.receipt_sha256, "tactile background receipt"),
            (self.receipt_payload_sha256, "tactile background receipt payload"),
            (self.dataset_tree_sha256, "tactile background dataset tree"),
        ):
            _require_sha256(digest, name)
        for name in CALVIN_TACTILE_STREAM_NAMES:
            background = self.backgrounds_by_stream[name]
            if background.flags.writeable:
                raise ContractError("loaded tactile backgrounds must be immutable")
            tactile_background_sha256(background)
            threshold = float(self.validity_thresholds_m[name])
            if not math.isfinite(threshold) or threshold <= 0.0:
                raise ContractError("loaded tactile validity threshold must be positive")


def _exact_mapping(value: object, *, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError(f"{name} fields differ from the frozen schema")
    if any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} keys must be text")
    return value


def load_calvin_tactile_backgrounds(
    archive_path: str | Path,
    receipt_path: str | Path,
    *,
    receipt_sha256: str,
    dataset_tree_sha256: str,
) -> LoadedCalvinTactileBackgrounds:
    """Load only a receipt-authenticated calibration for one dataset tree."""

    expected_receipt = _require_sha256(receipt_sha256, "expected calibration receipt")
    expected_tree = _require_sha256(dataset_tree_sha256, "expected calibration dataset tree")
    receipt_file = Path(receipt_path).resolve()
    receipt_bytes = read_sha256_verified_file_beneath(
        receipt_file.parent,
        receipt_file.name,
        expected_sha256=expected_receipt,
        maximum_bytes=4 * 1024 * 1024,
    )
    try:
        decoded = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("tactile calibration receipt is not valid JSON") from error
    receipt = _exact_mapping(
        decoded,
        fields={
            "archive",
            "calibration",
            "dataset",
            "official_calvin_source",
            "receipt_payload_sha256",
            "sampling",
            "schema",
        },
        name="tactile calibration receipt",
    )
    if receipt["schema"] != CALVIN_TACTILE_CALIBRATION_SCHEMA:
        raise ContractError("tactile calibration receipt schema changed")
    payload_digest = _require_sha256(
        receipt["receipt_payload_sha256"], "calibration receipt payload"
    )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    if canonical_calibration_receipt_sha256(unsigned) != payload_digest:
        raise ContractError("tactile calibration receipt payload hash changed")

    dataset = _exact_mapping(
        receipt["dataset"],
        fields={
            "dataset_id",
            "dataset_revision",
            "file_count",
            "manifest_sha256",
            "split_name",
            "tree_sha256",
        },
        name="tactile calibration dataset",
    )
    if dataset["tree_sha256"] != expected_tree:
        raise ContractError("tactile calibration belongs to another dataset tree")
    source = _exact_mapping(
        receipt["official_calvin_source"],
        fields={"commit", "files_sha256"},
        name="official CALVIN tactile source",
    )
    if (
        source["commit"] != CALVIN_TACTILE_SOURCE_COMMIT
        or source["files_sha256"] != CALVIN_TACTILE_SOURCE_FILES_SHA256
    ):
        raise ContractError("official CALVIN tactile source identity changed")
    archive = _exact_mapping(
        receipt["archive"],
        fields={"path", "sha256"},
        name="tactile background archive",
    )
    archive_digest = _require_sha256(archive["sha256"], "tactile background archive")
    archive_file = Path(archive_path).resolve()
    if str(archive_file) != archive["path"]:
        raise ContractError("tactile background archive path differs from its receipt")
    archive_bytes = read_sha256_verified_file_beneath(
        archive_file.parent,
        archive_file.name,
        expected_sha256=archive_digest,
        maximum_bytes=16 * 1024 * 1024,
    )

    calibration = _exact_mapping(
        receipt["calibration"],
        fields={"algorithm", "background_noise_ceiling_m", "streams"},
        name="tactile calibration",
    )
    if calibration["algorithm"] != CALVIN_TACTILE_BACKGROUND_ALGORITHM:
        raise ContractError("tactile background algorithm changed")
    noise_ceiling = calibration["background_noise_ceiling_m"]
    if (
        isinstance(noise_ceiling, bool)
        or not isinstance(noise_ceiling, (int, float))
        or not math.isfinite(noise_ceiling)
        or noise_ceiling <= 0.0
    ):
        raise ContractError("tactile background noise ceiling is invalid")
    raw_streams = _exact_mapping(
        calibration["streams"],
        fields=set(CALVIN_TACTILE_STREAM_NAMES),
        name="tactile calibration streams",
    )
    stream_receipts = {
        name: _exact_mapping(
            raw_streams[name],
            fields={
                "background_sha256",
                "candidate_count",
                "candidate_steps_sha256",
                "selected_count",
                "selected_source_sha256",
                "selected_steps",
                "validity_threshold_m",
            },
            name=f"{name} tactile calibration",
        )
        for name in CALVIN_TACTILE_STREAM_NAMES
    }
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as arrays:
            expected_arrays = {
                "schema",
                "left_digit",
                "right_digit",
                "left_digit_selected_steps",
                "right_digit_selected_steps",
            }
            if set(arrays.files) != expected_arrays:
                raise ContractError("tactile background archive arrays changed")
            if (
                np.asarray(arrays["schema"]).shape != ()
                or str(np.asarray(arrays["schema"]).item())
                != CALVIN_TACTILE_BACKGROUND_ARCHIVE_SCHEMA
            ):
                raise ContractError("tactile background archive schema changed")
            backgrounds = {
                name: _readonly(np.asarray(arrays[name], dtype=np.float32))
                for name in CALVIN_TACTILE_STREAM_NAMES
            }
            selected_steps = {
                name: tuple(
                    int(value)
                    for value in np.asarray(
                        arrays[f"{name}_selected_steps"], dtype=np.int64
                    ).tolist()
                )
                for name in CALVIN_TACTILE_STREAM_NAMES
            }
    except (OSError, ValueError) as error:
        raise ContractError("tactile background archive is not a safe NPZ") from error

    thresholds: dict[str, float] = {}
    for name in CALVIN_TACTILE_STREAM_NAMES:
        stream = stream_receipts[name]
        raw_selected_steps = stream["selected_steps"]
        if (
            not isinstance(raw_selected_steps, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in raw_selected_steps
            )
            or tuple(sorted(set(raw_selected_steps))) != tuple(raw_selected_steps)
        ):
            raise ContractError("tactile background selected steps are invalid")
        if selected_steps[name] != tuple(raw_selected_steps):
            raise ContractError("tactile background selected steps differ from receipt")
        if len(selected_steps[name]) != stream["selected_count"]:
            raise ContractError("tactile background selected count differs from receipt")
        if tactile_background_sha256(backgrounds[name]) != stream["background_sha256"]:
            raise ContractError("tactile background array differs from receipt")
        threshold = stream["validity_threshold_m"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold <= noise_ceiling
        ):
            raise ContractError("tactile validity threshold does not exceed calibration noise")
        thresholds[name] = float(threshold)

    return LoadedCalvinTactileBackgrounds(
        backgrounds_by_stream=MappingProxyType(backgrounds),
        validity_thresholds_m=MappingProxyType(thresholds),
        archive_sha256=archive_digest,
        receipt_sha256=expected_receipt,
        receipt_payload_sha256=payload_digest,
        dataset_tree_sha256=expected_tree,
    )
