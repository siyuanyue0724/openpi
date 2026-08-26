from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from picf_next.contracts import DenseEvidence  # noqa: E402
from picf_next.data.calvin import (  # noqa: E402
    CalvinPICFEvidenceFrame,
    CalvinPICFSensorObservation,
)
from picf_next.data.vjepa2_cache import (  # noqa: E402
    VJEPA2_CONTEXT_SENSORS,
    Vjepa2FeatureCache,
)
from picf_next.hosts.vjepa2_context import CalvinVjepa2CachedContextBuilder  # noqa: E402


class _FakeCache(Vjepa2FeatureCache):
    def __init__(self) -> None:
        self.encoder_contract = "fixture/v1"
        self.hidden_size = 8
        self.image_size = 32
        self.tubelet_size = 2
        self.patch_size = 16
        self.maximum_frames = 4

    def evidence_for(self, sample_key, clips_by_sensor):
        assert sample_key in {"first", "second"}
        output = []
        for sensor_key, modality in VJEPA2_CONTEXT_SENSORS:
            clip = clips_by_sensor[sensor_key]
            count = 0 if clip is None else 4
            tokens = np.full((count, 8), 1.0, dtype=np.float32)
            timestamps = np.zeros(count, dtype=np.float32)
            geometry = np.zeros((count, 3), dtype=np.float32)
            confidence = np.ones(count, dtype=np.float32)
            current = np.zeros(count, dtype=np.bool_)
            for array in (tokens, timestamps, geometry, confidence, current):
                array.setflags(write=False)
            output.append(
                DenseEvidence(
                    modality=modality,
                    encoder_contract=self.encoder_contract,
                    tokens=tokens,
                    available=clip is not None,
                    timestamps=timestamps,
                    confidence=confidence,
                    geometry=geometry,
                    current_measurement_valid=current,
                )
            )
        return tuple(output)


def _frame(timestamp_s: float, value: int) -> CalvinPICFEvidenceFrame:
    observations = []
    for sensor_key, _modality in VJEPA2_CONTEXT_SENSORS:
        image = np.full((8, 8, 3), value, dtype=np.uint8)
        image.setflags(write=False)
        observations.append(
            CalvinPICFSensorObservation(
                key=sensor_key,
                value=image,
                timestamp_s=timestamp_s,
                units="sRGB uint8",
            )
        )
    return CalvinPICFEvidenceFrame(
        sensor_observations=tuple(observations),
        timestamp_s=timestamp_s,
        delta_t_s=1.0,
    )


def test_cached_vjepa2_builder_retains_variable_complete_tubelets_as_context() -> None:
    builder = CalvinVjepa2CachedContextBuilder(
        _FakeCache(),
        device="cpu",
        dtype=torch.float32,
    )
    requests = (
        SimpleNamespace(
            sample_key="first",
            augmentation_seed=1,
            evidence_prefix=(_frame(0.0, 0),),
        ),
        SimpleNamespace(
            sample_key="second",
            augmentation_seed=2,
            evidence_prefix=(_frame(0.0, 0), _frame(1.0, 1)),
        ),
    )
    banks = builder(requests)

    assert builder.token_dims == {"vjepa_static": 8, "vjepa_gripper": 8}
    assert tuple(bank.modality for bank in banks) == ("vjepa_static", "vjepa_gripper")
    assert all(bank.tokens.shape == (2, 4, 8) for bank in banks)
    assert all(bank.valid[0].sum() == 0 and bank.valid[1].sum() == 4 for bank in banks)
    assert all(not bank.current_measurement_valid.any() for bank in banks)
