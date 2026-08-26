from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinEpisode
from picf_next.data.calvin_vjepa2_ac import (
    VJEPA2_AC_FRAME_COUNT,
    calvin_vjepa2_ac_frame_indices,
    load_calvin_vjepa2_ac_clip,
    vjepa2_ac_calvin_stride,
    vjepa2_ac_realized_pose_differences,
)
from picf_next.encoders.vjepa2_ac import (
    Vjepa2AcDonorConfig,
    _require_complete_state_load,
    vjepa2_ac_control_actions,
)


def test_vjepa2_ac_calvin_sampling_matches_official_ceil_rule() -> None:
    assert vjepa2_ac_calvin_stride(control_hz=30) == 8
    episode = CalvinEpisode(index=0, start=100, end=200)
    assert calvin_vjepa2_ac_frame_indices(
        episode,
        end_global_index=156,
        control_hz=30,
    ) == (100, 108, 116, 124, 132, 140, 148, 156)

    with pytest.raises(ContractError, match="crosses"):
        calvin_vjepa2_ac_frame_indices(
            episode,
            end_global_index=155,
            control_hz=30,
        )


def test_vjepa2_ac_realized_motion_is_the_exact_relative_pose_geometry() -> None:
    states = np.array(
        [
            [0.0, 0.1, 0.2, 0.30, -0.20, 0.10, 0.04],
            [0.2, 0.0, 0.4, -0.10, 0.45, 0.25, 0.03],
            [0.3, -0.2, 0.5, 0.20, 0.15, -0.35, 0.05],
        ],
        dtype=np.float64,
    )
    actual = vjepa2_ac_realized_pose_differences(states)
    matrices = Rotation.from_euler("xyz", states[:, 3:6]).as_matrix()
    expected_angles = Rotation.from_matrix(
        matrices[1:] @ np.swapaxes(matrices[:-1], 1, 2)
    ).as_euler("xyz")
    expected = np.concatenate(
        (states[1:, :3] - states[:-1, :3], expected_angles, states[1:, -1:] - states[:-1, -1:]),
        axis=1,
    ).astype(np.float32)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert not actual.flags.writeable
    assert not np.allclose(actual[:, 3:6], states[1:, 3:6] - states[:-1, 3:6])


def test_calvin_vjepa2_ac_loader_uses_only_images_and_observed_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = CalvinEpisode(index=0, start=100, end=200)
    index = CalvinDatasetIndex(
        split_root=tmp_path,
        dataset_id="calvin-vjepa2-ac-test",
        dataset_revision="sha256:test",
        control_hz=30,
        episodes=(episode,),
        segments=(),
    )
    requested: list[tuple[int, tuple[str, ...] | None]] = []

    def source_frame(
        global_index: int,
        *,
        fields: tuple[str, ...] | None = None,
        verify_relative_action: bool = True,
    ) -> dict[str, np.ndarray]:
        del verify_relative_action
        requested.append((global_index, fields))
        state = np.zeros(15, dtype=np.float32)
        state[:3] = [global_index * 0.001, -0.2, 0.4]
        state[3:6] = [0.01 * (global_index - 100), 0.1, -0.2]
        state[6] = 0.04 - 0.0001 * (global_index - 100)
        image = np.full((4, 5, 3), global_index % 255, dtype=np.uint8)
        return {"rgb_static": image, "robot_obs": state}

    monkeypatch.setattr(index, "validated_source_frame_arrays", source_frame)
    clip = load_calvin_vjepa2_ac_clip(index, end_global_index=156)

    assert len(requested) == VJEPA2_AC_FRAME_COUNT
    assert all(fields == ("rgb_static", "robot_obs") for _index, fields in requested)
    assert clip.frame_indices == tuple(global_index for global_index, _fields in requested)
    assert clip.images.shape == (8, 4, 5, 3)
    assert clip.states.shape == (8, 7)
    assert clip.realized_motion.shape == (7, 7)
    assert not clip.images.flags.writeable
    assert not clip.states.flags.writeable
    assert not clip.realized_motion.flags.writeable
    np.testing.assert_allclose(np.diff(clip.frame_timestamps_s), 8.0 / 30.0, atol=1e-7)


def test_vjepa2_ac_controls_are_deterministic_non_aliasing_interventions() -> None:
    actions = np.arange(49, dtype=np.float32).reshape(7, 7)
    controls = vjepa2_ac_control_actions(actions, seed=239)

    assert tuple(controls) == ("actual", "zero", "reversed", "shuffled")
    np.testing.assert_array_equal(controls["actual"], actions)
    np.testing.assert_array_equal(controls["zero"], 0.0)
    np.testing.assert_array_equal(controls["reversed"], actions[::-1])
    assert not np.array_equal(controls["shuffled"], actions)
    assert all(not value.flags.writeable for value in controls.values())
    assert all(not np.shares_memory(value, actions) for value in controls.values())
    np.testing.assert_array_equal(
        controls["shuffled"],
        vjepa2_ac_control_actions(actions, seed=239)["shuffled"],
    )


def test_vjepa2_ac_hub_gate_rejects_silent_capacity_or_recipe_changes() -> None:
    Vjepa2AcDonorConfig()
    with pytest.raises(ContractError, match="differs"):
        Vjepa2AcDonorConfig(model_capacity_frames=8)
    with pytest.raises(ContractError, match="differs"):
        Vjepa2AcDonorConfig(autoregressive_steps=1)


def test_vjepa2_ac_checkpoint_load_must_cover_every_model_key() -> None:
    _require_complete_state_load(
        SimpleNamespace(missing_keys=(), unexpected_keys=()),
        component="encoder",
    )
    with pytest.raises(ContractError, match="missing=.*patch_embed"):
        _require_complete_state_load(
            SimpleNamespace(missing_keys=("patch_embed.proj.weight",), unexpected_keys=()),
            component="encoder",
        )
    with pytest.raises(ContractError, match="unexpected=.*obsolete"):
        _require_complete_state_load(
            SimpleNamespace(missing_keys=(), unexpected_keys=("obsolete.weight",)),
            component="predictor",
        )
