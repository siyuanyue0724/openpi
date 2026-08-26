from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.causal_video import CausalVideoSourceFrame, build_causal_video_clip
from picf_next.data.vjepa2_cache import (
    VJEPA2_CACHE_AUGMENTATION,
    VJEPA2_CACHE_SCHEMA,
    VJEPA2_CONTEXT_SENSORS,
    Vjepa2FeatureCache,
)
from picf_next.encoders.vjepa2 import VJEPA2_MODEL_ID, VJEPA2_MODEL_REVISION

_DATASET_SHA = "d" * 64


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clip(sensor_key: str, *, offset: int = 0):
    frames = []
    for index in range(2):
        image = np.full((8, 8, 3), index + offset, dtype=np.uint8)
        image.setflags(write=False)
        frames.append(
            CausalVideoSourceFrame.from_image(
                image,
                timestamp_s=float(index),
                sensor_key=sensor_key,
            )
        )
    return build_causal_video_clip(
        frames,
        current_timestamp_s=1.0,
        maximum_frames=4,
        tubelet_size=2,
    )


def _cache(tmp_path: Path) -> tuple[Vjepa2FeatureCache, dict[str, object]]:
    entries = []
    sensor_rows = []
    for sensor_index, (sensor_key, modality) in enumerate(VJEPA2_CONTEXT_SENSORS):
        clip = _clip(sensor_key, offset=sensor_index)
        assert clip is not None
        tokens = np.arange(32, dtype=np.float32).reshape(4, 8) + sensor_index
        buffer = io.BytesIO()
        np.save(buffer, tokens, allow_pickle=False)
        artifact = buffer.getvalue()
        relative = f"entries/sample/{modality}.tokens.npy"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact)
        sensor_rows.append(
            {
                "artifact_path": relative,
                "artifact_sha256": _sha(artifact),
                "modality": modality,
                "sensor_key": sensor_key,
                "source_frame_sha256": list(clip.source_frame_sha256),
                "token_count": 4,
            }
        )
    entries.append({"sample_key": "sample-0", "sensors": sensor_rows})
    manifest = {
        "augmentation_contract": VJEPA2_CACHE_AUGMENTATION,
        "complete": True,
        "dataset_tree_sha256": _DATASET_SHA,
        "encoder": {
            "checkpoint_revision": VJEPA2_MODEL_REVISION,
            "encoder_contract": f"{VJEPA2_MODEL_ID}@{VJEPA2_MODEL_REVISION}/fixture/v2",
            "hidden_size": 8,
            "image_size": 32,
            "maximum_frames": 4,
            "model_id": VJEPA2_MODEL_ID,
            "patch_size": 16,
            "tubelet_size": 2,
        },
        "entries": entries,
        "expected_entries": 1,
        "schema": VJEPA2_CACHE_SCHEMA,
        "sensors": [
            {"sensor_key": sensor_key, "modality": modality}
            for sensor_key, modality in VJEPA2_CONTEXT_SENSORS
        ],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    return (
        Vjepa2FeatureCache.load(
            tmp_path,
            manifest_sha256=_sha(manifest_bytes),
            dataset_tree_sha256=_DATASET_SHA,
            memory_capacity=1,
        ),
        manifest,
    )


def test_vjepa2_cache_revalidates_sources_and_loads_immutable_native_tokens(
    tmp_path: Path,
) -> None:
    cache, _manifest = _cache(tmp_path)
    clips = {
        sensor_key: _clip(sensor_key, offset=index)
        for index, (sensor_key, _modality) in enumerate(VJEPA2_CONTEXT_SENSORS)
    }
    evidence = cache.evidence_for("sample-0", clips)

    assert tuple(item.modality for item in evidence) == ("vjepa_static", "vjepa_gripper")
    assert all(item.available and item.tokens.shape == (4, 8) for item in evidence)
    assert all(item.geometry is not None and item.geometry.shape == (4, 3) for item in evidence)
    assert all(not item.effective_current_measurement_valid.any() for item in evidence)
    assert all(not item.tokens.flags.writeable for item in evidence)

    changed = dict(clips)
    changed[VJEPA2_CONTEXT_SENSORS[0][0]] = _clip(VJEPA2_CONTEXT_SENSORS[0][0], offset=9)
    with pytest.raises(ContractError, match="runtime causal clip differs"):
        cache.evidence_for("sample-0", changed)


def test_vjepa2_cache_rejects_incomplete_manifest_and_artifact_tampering(tmp_path: Path) -> None:
    cache, manifest = _cache(tmp_path)
    first = cache.entries["sample-0"].sensors[0]
    assert first.artifact_path is not None
    (tmp_path / first.artifact_path).write_bytes(b"tampered")
    clips = {
        sensor_key: _clip(sensor_key, offset=index)
        for index, (sensor_key, _modality) in enumerate(VJEPA2_CONTEXT_SENSORS)
    }
    with pytest.raises(ContractError, match="content hash mismatch"):
        cache.evidence_for("sample-0", clips)

    manifest["complete"] = False
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    with pytest.raises(ContractError, match="manifest is incomplete"):
        Vjepa2FeatureCache.load(
            tmp_path,
            manifest_sha256=_sha(manifest_bytes),
            dataset_tree_sha256=_DATASET_SHA,
        )
