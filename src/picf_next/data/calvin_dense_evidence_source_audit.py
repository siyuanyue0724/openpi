"""Independent source-input audit for frozen CALVIN dense evidence.

The frozen encoders are deliberately absent from this module.  It reconstructs
only the content-addressed inputs from authenticated CALVIN observations and
compares them with the hashes recorded by each cache.  This keeps cache
provenance verification independent of expensive model inference and of the
cache producer's control flow.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from picf_next.content_addressing import canonical_payload_sha256, ndarray_sha256
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinPhysicalTransitionDataset
from picf_next.data.calvin_tactile import (
    CALVIN_TACTILE_FRAME_COUNT,
    calvin_tactile_source_frames,
)
from picf_next.data.calvin_tactile_calibration import LoadedCalvinTactileBackgrounds
from picf_next.data.causal_video import build_calvin_causal_video_clip
from picf_next.data.dense_evidence_cache import FrozenDenseEvidenceCacheBank
from picf_next.encoders.vjepa21 import VJEPA21_CALVIN_VIEW_NAMES, Vjepa21DenseConfig

CALVIN_DENSE_SOURCE_AUDIT_SCHEMA = "picf-next.calvin-dense-source-input-audit/v1"

_AUDIT_PAYLOAD_FIELDS = {
    "cache_manifest_sha256",
    "coverage_plan_sha256",
    "dataset_id",
    "dataset_revision",
    "dataset_tree_sha256",
    "record_count",
    "record_start",
    "record_stop",
    "records_sha256",
    "schema",
    "status",
}

_VJEPA_SENSOR_BY_VIEW = {
    "static": "observation.images.rgb_static",
    "gripper": "observation.images.rgb_gripper",
}
_SOURCE_NAME_BY_SENSOR = {
    "observation.images.rgb_static": "rgb_static",
    "observation.images.rgb_gripper": "rgb_gripper",
    "observation.depth.static": "depth_static",
    "observation.depth.gripper": "depth_gripper",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _frame_values(frame: Any) -> dict[str, np.ndarray]:
    observations = getattr(frame, "sensor_observations", None)
    if not isinstance(observations, tuple) or not observations:
        raise ContractError("CALVIN source audit requires one typed sensor frame")
    values = {item.key: item.value for item in observations}
    if len(values) != len(observations):
        raise ContractError("CALVIN source audit frame repeats a sensor key")
    return values


def calvin_vjepa21_source_input_sha256(
    dataset: CalvinPhysicalTransitionDataset,
    sample_key: str,
    *,
    encoder_contract: str,
    frame_count: int | None = None,
) -> str:
    """Recompute the ordered dual-view causal-video input identity."""

    count = Vjepa21DenseConfig().frame_count if frame_count is None else frame_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ContractError("V-JEPA source-audit frame count must be positive")
    prefix = dataset.evidence_prefix_by_key(sample_key, maximum_source_frames=count)
    ordered_views = []
    for view_name in VJEPA21_CALVIN_VIEW_NAMES:
        clip = build_calvin_causal_video_clip(
            prefix,
            sensor_key=_VJEPA_SENSOR_BY_VIEW[view_name],
            maximum_frames=count,
            tubelet_size=1,
        )
        if clip is None:
            raise ContractError("CALVIN source audit found no causal V-JEPA source clip")
        ordered_views.append(
            {"name": view_name, "source_frame_sha256": list(clip.source_frame_sha256)}
        )
    return canonical_payload_sha256(
        "picf-next.calvin-vjepa21-input/v1",
        {"encoder_contract": encoder_contract, "ordered_views": ordered_views},
    )


def calvin_sonata_source_input_sha256(
    dataset: CalvinPhysicalTransitionDataset,
    sample_key: str,
    *,
    encoder_contract: str,
) -> str:
    """Recompute the raw dual-RGBD, robot-state and timestamp identity."""

    source_global_index = dataset.source_global_index_by_key(sample_key)
    frame = dataset.index.source_picf_evidence_frame(source_global_index)
    values = _frame_values(frame)
    source_frame = {
        source_name: values[sensor_name]
        for sensor_name, source_name in _SOURCE_NAME_BY_SENSOR.items()
    }
    source_frame["robot_obs"] = dataset.index.source_robot_state(source_global_index)
    return canonical_payload_sha256(
        "picf-next.calvin-sonata-input/v1",
        {
            "arrays": {
                name: ndarray_sha256(name, value) for name, value in sorted(source_frame.items())
            },
            "encoder_contract": encoder_contract,
            "timestamp_s": frame.timestamp_s.hex(),
        },
    )


def calvin_anytouch2_source_input_sha256(
    dataset: CalvinPhysicalTransitionDataset,
    sample_key: str,
    *,
    encoder_contract: str,
    calibration: LoadedCalvinTactileBackgrounds,
) -> str:
    """Recompute the causal DIGIT, robot-state and calibration identity."""

    source_global_index = dataset.source_global_index_by_key(sample_key)
    prefix = dataset.evidence_prefix_by_key(
        sample_key,
        maximum_source_frames=CALVIN_TACTILE_FRAME_COUNT,
    )
    first_global_index = source_global_index - len(prefix) + 1
    states = tuple(
        dataset.index.source_robot_state(global_index)
        for global_index in range(first_global_index, source_global_index + 1)
    )
    tactile_sources = [
        tactile for frame in prefix for tactile in calvin_tactile_source_frames(frame)
    ]
    return canonical_payload_sha256(
        "picf-next.calvin-anytouch2-input/v1",
        {
            "calibration_archive_sha256": calibration.archive_sha256,
            "calibration_receipt_sha256": calibration.receipt_sha256,
            "encoder_contract": encoder_contract,
            "robot_states": [
                ndarray_sha256(f"robot_obs[{index}]", state) for index, state in enumerate(states)
            ],
            "tactile_sources": [
                {"stream": item.stream_name, "sha256": item.source_sha256}
                for item in tactile_sources
            ],
        },
    )


def audit_calvin_dense_evidence_source_inputs(
    dataset: CalvinPhysicalTransitionDataset,
    bank: FrozenDenseEvidenceCacheBank,
    *,
    cache_manifest_sha256_by_modality: Mapping[str, str],
    calibration: LoadedCalvinTactileBackgrounds,
    record_start: int = 0,
    record_stop: int | None = None,
    vjepa_frame_count: int | None = None,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Verify every selected cache hash against independently decoded inputs."""

    if bank.modalities != ("anytouch", "sonata", "vjepa"):
        raise ContractError("CALVIN source audit requires exact full-modal cache order")
    if set(cache_manifest_sha256_by_modality) != set(bank.modalities):
        raise ContractError("CALVIN source audit cache-manifest modalities differ")
    manifest_sha256s = {
        name: _sha256(cache_manifest_sha256_by_modality[name], f"{name} cache manifest")
        for name in bank.modalities
    }
    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ContractError("CALVIN source audit requires an authenticated dataset manifest")
    if (
        bank.contracts[0].dataset_id,
        bank.contracts[0].dataset_revision,
        bank.contracts[0].dataset_tree_sha256,
    ) != (manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256):
        raise ContractError("CALVIN source audit cache and dataset identities differ")
    if calibration.dataset_tree_sha256 != manifest.tree_sha256:
        raise ContractError("CALVIN source audit tactile calibration belongs elsewhere")

    stop = bank.record_count if record_stop is None else record_stop
    if (
        isinstance(record_start, bool)
        or not isinstance(record_start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or not 0 <= record_start < stop <= bank.record_count
    ):
        raise ContractError("CALVIN source audit range is outside cache coverage")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ContractError("CALVIN source audit worker count must be positive")

    caches = {cache.contract.modality: cache for cache in bank.caches}
    contracts = {name: cache.contract.encoder_contract for name, cache in caches.items()}
    records_digest = hashlib.sha256(b"picf-next.calvin-dense-source-audit-records/v1\0")

    def audit_record(position: int) -> dict[str, object]:
        locations = {name: cache.records[position] for name, cache in caches.items()}
        identities = {
            (location.source_global_index, location.sample_key) for location in locations.values()
        }
        if len(identities) != 1:
            raise ContractError("CALVIN source audit cache records lost alignment")
        source_global_index, sample_key = next(iter(identities))
        if dataset.source_global_index_by_key(sample_key) != source_global_index:
            raise ContractError("CALVIN source audit sample identity differs from raw data")
        expected = {
            "anytouch": calvin_anytouch2_source_input_sha256(
                dataset,
                sample_key,
                encoder_contract=contracts["anytouch"],
                calibration=calibration,
            ),
            "sonata": calvin_sonata_source_input_sha256(
                dataset,
                sample_key,
                encoder_contract=contracts["sonata"],
            ),
            "vjepa": calvin_vjepa21_source_input_sha256(
                dataset,
                sample_key,
                encoder_contract=contracts["vjepa"],
                frame_count=vjepa_frame_count,
            ),
        }
        observed = {name: location.source_input_sha256 for name, location in locations.items()}
        if observed != expected:
            mismatched = sorted(name for name in expected if observed[name] != expected[name])
            raise ContractError(
                "CALVIN dense cache source identity differs at "
                f"record {position} ({sample_key}): {mismatched}"
            )
        return {
            "position": position,
            "sample_key": sample_key,
            "source_global_index": source_global_index,
            "source_input_sha256": expected,
        }

    def ordered_records() -> Any:
        if workers == 1:
            for position in range(record_start, stop):
                yield audit_record(position)
            return
        batch_size = workers * 4
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch_start in range(record_start, stop, batch_size):
                positions = range(batch_start, min(batch_start + batch_size, stop))
                futures = [executor.submit(audit_record, position) for position in positions]
                for future in futures:
                    yield future.result()

    for completed, record in enumerate(ordered_records(), start=1):
        records_digest.update(
            _canonical_bytes(record)
        )
        records_digest.update(b"\0")
        if progress is not None:
            progress(completed, stop - record_start)

    payload: dict[str, object] = {
        "cache_manifest_sha256": dict(sorted(manifest_sha256s.items())),
        "coverage_plan_sha256": bank.coverage_plan_sha256,
        "dataset_id": manifest.dataset_id,
        "dataset_revision": manifest.dataset_revision,
        "dataset_tree_sha256": manifest.tree_sha256,
        "record_count": stop - record_start,
        "record_start": record_start,
        "record_stop": stop,
        "records_sha256": records_digest.hexdigest(),
        "schema": CALVIN_DENSE_SOURCE_AUDIT_SCHEMA,
        "status": "PASS",
    }
    return {
        **payload,
        "artifact_sha256": canonical_payload_sha256(
            "picf-next.calvin-dense-source-input-audit-artifact/v1",
            payload,
        ),
    }


def validate_calvin_dense_evidence_source_audit(
    value: object,
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_tree_sha256: str,
    coverage_plan_sha256: str,
    cache_manifest_sha256_by_modality: Mapping[str, str],
    record_count: int,
) -> dict[str, object]:
    """Validate one complete content-addressed all-record source audit."""

    if not isinstance(value, Mapping) or set(value) != _AUDIT_PAYLOAD_FIELDS | {"artifact_sha256"}:
        raise ContractError("CALVIN dense source audit fields differ from schema")
    payload = {key: value[key] for key in _AUDIT_PAYLOAD_FIELDS}
    if payload["schema"] != CALVIN_DENSE_SOURCE_AUDIT_SCHEMA or payload["status"] != "PASS":
        raise ContractError("CALVIN dense source audit did not pass the registered schema")
    expected_manifests = {
        name: _sha256(digest, f"{name} cache manifest")
        for name, digest in sorted(cache_manifest_sha256_by_modality.items())
    }
    if set(expected_manifests) != {"anytouch", "sonata", "vjepa"}:
        raise ContractError("CALVIN dense source audit expected cache modalities differ")
    if payload["cache_manifest_sha256"] != expected_manifests:
        raise ContractError("CALVIN dense source audit cache manifests differ")
    if (
        payload["dataset_id"] != dataset_id
        or payload["dataset_revision"] != dataset_revision
        or payload["dataset_tree_sha256"] != _sha256(dataset_tree_sha256, "expected dataset tree")
        or payload["coverage_plan_sha256"]
        != _sha256(coverage_plan_sha256, "expected coverage plan")
    ):
        raise ContractError("CALVIN dense source audit dataset/coverage identity differs")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
        or payload["record_start"] != 0
        or payload["record_stop"] != record_count
        or payload["record_count"] != record_count
    ):
        raise ContractError("CALVIN dense source audit does not cover every record")
    _sha256(payload["records_sha256"], "source-audit records")
    expected_artifact = canonical_payload_sha256(
        "picf-next.calvin-dense-source-input-audit-artifact/v1",
        payload,
    )
    if value["artifact_sha256"] != expected_artifact:
        raise ContractError("CALVIN dense source audit artifact hash changed")
    return dict(value)
