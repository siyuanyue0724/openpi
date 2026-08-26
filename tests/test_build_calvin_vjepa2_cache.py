from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.vjepa2_cache import VJEPA2_CONTEXT_SENSORS
from tools.build_calvin_vjepa2_cache import (
    _canonical_json,
    _manifest,
    _token_artifact,
    _trusted_partial_artifacts,
)


def _encoder() -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_revision="revision",
        encoder_contract="fixture@revision/dense/v1",
        hidden_size=8,
        image_size=32,
        model_id="fixture",
        patch_size=16,
        tubelet_size=2,
    )


def _partial_manifest(
    root: Path,
    *,
    sample_key: str,
    artifact_path: str,
    artifact_sha256: str,
) -> dict[str, object]:
    source_hashes = ("a" * 64, "b" * 64)
    sensors = []
    for sensor_index, (sensor_key, modality) in enumerate(VJEPA2_CONTEXT_SENSORS):
        if sensor_index == 0:
            sensors.append(
                {
                    "artifact_path": artifact_path,
                    "artifact_sha256": artifact_sha256,
                    "modality": modality,
                    "sensor_key": sensor_key,
                    "source_frame_sha256": list(source_hashes),
                    "token_count": 4,
                }
            )
        else:
            sensors.append(
                {
                    "artifact_path": None,
                    "artifact_sha256": None,
                    "modality": modality,
                    "sensor_key": sensor_key,
                    "source_frame_sha256": [],
                    "token_count": 0,
                }
            )
    expected = _manifest(
        dataset_tree_sha256="d" * 64,
        encoder=_encoder(),
        maximum_frames=4,
        expected_entries=1,
        entries=[],
        complete=False,
    )
    partial = dict(expected)
    partial["entries"] = [{"sample_key": sample_key, "sensors": sensors}]
    (root / "manifest.partial.json").write_bytes(_canonical_json(partial))
    return expected


def test_partial_manifest_is_the_only_resume_authority(tmp_path: Path) -> None:
    sample_key = "sample-0"
    source_hashes = ("a" * 64, "b" * 64)
    tokens = np.arange(32, dtype=np.float32).reshape(4, 8)
    relative, artifact_sha, _size = _token_artifact(
        tmp_path,
        sample_key=sample_key,
        modality="vjepa_static",
        clip_digest="c" * 64,
        tokens=tokens,
        expected_shape=(4, 8),
        trusted_sha256=None,
    )
    expected = _partial_manifest(
        tmp_path,
        sample_key=sample_key,
        artifact_path=relative,
        artifact_sha256=artifact_sha,
    )

    trusted = _trusted_partial_artifacts(
        tmp_path,
        expected_manifest=expected,
        sample_keys=(sample_key,),
    )
    key = (
        sample_key,
        VJEPA2_CONTEXT_SENSORS[0][0],
        "vjepa_static",
        source_hashes,
        4,
        relative,
    )
    assert trusted == {key: artifact_sha}

    (tmp_path / relative).write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="content hash mismatch"):
        _token_artifact(
            tmp_path,
            sample_key=sample_key,
            modality="vjepa_static",
            clip_digest="c" * 64,
            tokens=None,
            expected_shape=(4, 8),
            trusted_sha256=trusted[key],
        )


def test_uncheckpointed_artifact_is_reencoded_instead_of_shape_trusted(tmp_path: Path) -> None:
    original = np.zeros((4, 8), dtype=np.float32)
    relative, original_sha, _size = _token_artifact(
        tmp_path,
        sample_key="sample-0",
        modality="vjepa_static",
        clip_digest="c" * 64,
        tokens=original,
        expected_shape=(4, 8),
        trusted_sha256=None,
    )
    replacement = np.ones((4, 8), dtype=np.float32)
    observed_relative, replacement_sha, _size = _token_artifact(
        tmp_path,
        sample_key="sample-0",
        modality="vjepa_static",
        clip_digest="c" * 64,
        tokens=replacement,
        expected_shape=(4, 8),
        trusted_sha256=None,
    )

    assert observed_relative == relative
    assert replacement_sha != original_sha
    array = np.load(io.BytesIO((tmp_path / relative).read_bytes()), allow_pickle=False)
    np.testing.assert_array_equal(array, replacement)
