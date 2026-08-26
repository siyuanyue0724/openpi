from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _load_audit_tool():
    path = Path(__file__).parents[1] / "tools" / "audit_molmoact2_libero.py"
    spec = importlib.util.spec_from_file_location("picf_libero_audit_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load_audit_tool()
DATASET_REVISION = _audit.DATASET_REVISION
_decode_image = _audit._decode_image
_pearson_xyz_action_state_delta = _audit._pearson_xyz_action_state_delta
_phase_indices = _audit._phase_indices


def test_phase_indices_cover_first_middle_and_last_row() -> None:
    assert _phase_indices(10, 15) == (
        ("start", 10),
        ("middle", 12),
        ("end", 14),
    )
    with pytest.raises(ValueError, match="at least two rows"):
        _phase_indices(10, 10)


def test_embedded_image_decoder_is_shape_strict() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color=(1, 2, 3)).save(buffer, format="PNG")
    decoded = _decode_image({"bytes": buffer.getvalue(), "path": "unused.png"})
    assert decoded.size == (256, 256)
    assert decoded.getpixel((0, 0)) == (1, 2, 3)

    bad_buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(bad_buffer, format="PNG")
    with pytest.raises(ValueError, match="unexpected image size"):
        _decode_image({"bytes": bad_buffer.getvalue(), "path": "unused.png"})


def test_xyz_alignment_probe_detects_next_state_delta() -> None:
    rows = []
    state = np.zeros(8, dtype=np.float64)
    for step in range(12):
        action = np.asarray([(-1.0) ** step * (step + 1), step - 5.0, (step % 4) - 1.5, 0, 0, 0, 0])
        rows.append({"action": action.tolist(), "observation.state": state.tolist()})
        state = state.copy()
        state[:3] += 0.01 * action[:3]
    rows.append({"action": [0.0] * 7, "observation.state": state.tolist()})

    correlations = _pearson_xyz_action_state_delta(rows)
    assert correlations == pytest.approx([1.0, 1.0, 1.0])


def test_dataset_revision_is_immutable_commit() -> None:
    assert len(DATASET_REVISION) == 40
    int(DATASET_REVISION, 16)
