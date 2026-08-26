from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from picf_next.contracts import DenseEvidence
from picf_next.data.calvin import CalvinPICFEvidenceFrame, CalvinPICFSensorObservation
from picf_next.data.vjepa2_cache import VJEPA2_CONTEXT_SENSORS
from tools import audit_calvin_vjepa2_cache as audit_tool
from tools.audit_calvin_vjepa2_cache import AUDIT_SCHEMA, audit_cache, load_audit_dataset


def _frame(timestamp_s: float) -> CalvinPICFEvidenceFrame:
    observations = []
    for sensor_index, (sensor_key, _modality) in enumerate(VJEPA2_CONTEXT_SENSORS):
        image = np.full((8, 8, 3), int(timestamp_s) + sensor_index, dtype=np.uint8)
        image.setflags(write=False)
        observations.append(
            CalvinPICFSensorObservation(
                key=sensor_key,
                value=image,
                timestamp_s=timestamp_s,
                units="uint8_rgb",
            )
        )
    return CalvinPICFEvidenceFrame(
        sensor_observations=tuple(observations),
        timestamp_s=timestamp_s,
        delta_t_s=1.0,
    )


class _Dataset:
    sample_keys = ("sample-0", "sample-1")

    def __init__(self) -> None:
        first = _frame(0.0)
        second = _frame(1.0)
        self.prefixes = {"sample-0": (first,), "sample-1": (first, second)}

    def evidence_prefix_by_key(self, sample_key: str, *, maximum_source_frames: int):
        return self.prefixes[sample_key][-maximum_source_frames:]


class _Cache:
    entries = {"sample-0": object(), "sample-1": object()}
    maximum_frames = 4
    tubelet_size = 2
    image_size = 32
    patch_size = 16
    hidden_size = 8
    encoder_contract = "fixture@revision/dense/v1"
    dataset_tree_sha256 = "d" * 64

    def evidence_for(self, sample_key: str, clips_by_sensor):
        del sample_key
        evidence = []
        for sensor_key, modality in VJEPA2_CONTEXT_SENSORS:
            clip = clips_by_sensor[sensor_key]
            token_count = 0 if clip is None else 4
            timestamp = 0.0 if clip is None else clip.current_timestamp_s
            evidence.append(
                DenseEvidence(
                    modality=modality,
                    encoder_contract=self.encoder_contract,
                    tokens=np.zeros((token_count, self.hidden_size), dtype=np.float32),
                    available=clip is not None,
                    timestamps=np.full(token_count, timestamp, dtype=np.float32),
                    confidence=np.ones(token_count, dtype=np.float32),
                    geometry=np.zeros((token_count, 3), dtype=np.float32),
                    current_measurement_valid=np.zeros(token_count, dtype=np.bool_),
                )
            )
        return tuple(evidence)


def test_cache_audit_rebuilds_every_clip_and_reports_context_only_tokens() -> None:
    report = audit_cache(
        dataset=_Dataset(),  # type: ignore[arg-type]
        cache=_Cache(),  # type: ignore[arg-type]
        cache_manifest_sha256="c" * 64,
    )

    assert report["schema"] == AUDIT_SCHEMA
    assert report["complete"] is True
    assert report["samples"] == 2
    assert report["artifact_reads"] == 2
    assert report["current_measurement_rows"] == 0
    assert report["total_token_rows"] == 8
    assert report["frame_count_histogram_across_sensors"] == {"0": 2, "2": 2}
    assert report["token_count_histogram_per_sample"] == {"0": 1, "8": 1}
    assert report["modality_token_rows"] == {"vjepa_gripper": 4, "vjepa_static": 4}


def test_audit_dataset_skips_redundant_full_tree_path_scan(monkeypatch, tmp_path) -> None:
    captured = {}
    index = object()
    dataset = object()

    def fake_load(split_root, **kwargs):
        captured["split_root"] = split_root
        captured.update(kwargs)
        return index

    def fake_dataset(value, *, action_horizon):
        assert value is index
        assert action_horizon == 1
        return dataset

    monkeypatch.setattr(audit_tool.CalvinDatasetIndex, "load", staticmethod(fake_load))
    monkeypatch.setattr(audit_tool, "CalvinStatefulTransitionDataset", fake_dataset)
    manifest = SimpleNamespace(dataset_id="dataset", dataset_revision="revision")

    result = load_audit_dataset(
        dataset_root=tmp_path,
        split="training",
        manifest=manifest,  # type: ignore[arg-type]
    )

    assert result is dataset
    assert captured == {
        "split_root": (tmp_path / "training").resolve(),
        "dataset_id": "dataset",
        "dataset_revision": "revision",
        "verify_files": False,
        "dataset_manifest": manifest,
    }
