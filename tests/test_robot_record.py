from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.robot_record import (
    MOLMOACT2_LIBERO_ACTION_AXES,
    MOLMOACT2_LIBERO_CAMERA_KEYS,
    MOLMOACT2_LIBERO_STATE_AXES,
    decode_molmoact2_libero_row,
    validate_molmoact2_libero_metadata,
)


def _row(frame: int = 2) -> dict:
    return {
        "observation.images.image": {"bytes": b"external", "path": f"frame_{frame:06d}.png"},
        "observation.images.wrist_image": {
            "bytes": b"wrist",
            "path": f"frame_{frame:06d}.png",
        },
        "observation.state": np.arange(8, dtype=np.float32),
        "action": np.arange(7, dtype=np.float32),
        "timestamp": frame / 10.0,
        "frame_index": frame,
        "episode_index": 3,
        "index": 100 + frame,
        "task_index": 4,
    }


def test_record_is_lossless_typed_and_has_no_target_channel() -> None:
    record = decode_molmoact2_libero_row(_row(), task="put the bowl on the plate", episode_length=5)

    assert record.state_axes == MOLMOACT2_LIBERO_STATE_AXES
    assert record.action_axes == MOLMOACT2_LIBERO_ACTION_AXES
    assert tuple(camera.key for camera in record.cameras) == MOLMOACT2_LIBERO_CAMERA_KEYS
    assert record.cameras[0].encoded_bytes == b"external"
    assert record.transition_valid
    assert record.delta_t_s == 0.1
    assert not record.state.flags.writeable
    assert not record.action.flags.writeable
    forbidden = {"mask", "object_id", "role", "scorer", "future_observation"}
    assert forbidden.isdisjoint(record.__dataclass_fields__)


def test_final_frame_is_not_a_posterior_transition() -> None:
    record = decode_molmoact2_libero_row(_row(frame=4), task="task", episode_length=5)
    assert not record.transition_valid
    assert record.action_valid.all()


def test_decoder_rejects_time_shape_and_image_failures() -> None:
    row = _row()
    row["timestamp"] = 8.0
    with pytest.raises(ContractError, match="10 Hz"):
        decode_molmoact2_libero_row(row, task="task", episode_length=5)

    row = _row()
    row["action"] = [0.0] * 6
    with pytest.raises(ContractError, match="shape"):
        decode_molmoact2_libero_row(row, task="task", episode_length=5)

    row = _row()
    row["observation.images.image"] = {"bytes": b"", "path": "frame.png"}
    with pytest.raises(ContractError, match="no embedded"):
        decode_molmoact2_libero_row(row, task="task", episode_length=5)

    row = _row()
    row["frame_index"] = 2.5
    with pytest.raises(ContractError, match="frame_index must be an integer"):
        decode_molmoact2_libero_row(row, task="task", episode_length=5)

    row = _row()
    row["index"] = True
    with pytest.raises(ContractError, match="index must be an integer"):
        decode_molmoact2_libero_row(row, task="task", episode_length=5)

    with pytest.raises(ContractError, match="episode_length must be a positive integer"):
        decode_molmoact2_libero_row(_row(), task="task", episode_length=True)


def test_record_contract_rejects_mutable_arrays_and_non_boolean_transition() -> None:
    record = decode_molmoact2_libero_row(_row(), task="task", episode_length=5)
    with pytest.raises(ContractError, match="must be immutable"):
        replace(record, state=record.state.copy())
    with pytest.raises(ContractError, match="transition_valid must be boolean"):
        replace(record, transition_valid=1)  # type: ignore[arg-type]


def test_public_metadata_is_dimensional_not_semantic_authority() -> None:
    info = {
        "fps": 10,
        "total_episodes": 1693,
        "total_frames": 273465,
        "total_tasks": 40,
        "features": {
            "observation.state": {
                "shape": [8],
                "names": [
                    "x",
                    "y",
                    "z",
                    "quaternion.x",
                    "quaternion.y",
                    "quaternion.z",
                    "quaternion.w",
                    "gripper",
                ],
            },
            "action": {"shape": [7]},
        },
    }
    validate_molmoact2_libero_metadata(info)

    state_names = info["features"]["observation.state"]["names"]
    assert state_names != list(MOLMOACT2_LIBERO_STATE_AXES)
    assert "quaternion" in " ".join(state_names).lower()

    broken = {**info, "features": {**info["features"], "observation.state": {"shape": [7]}}}
    broken["features"]["observation.state"]["shape"] = [7]
    with pytest.raises(ContractError, match="state width"):
        validate_molmoact2_libero_metadata(broken)
