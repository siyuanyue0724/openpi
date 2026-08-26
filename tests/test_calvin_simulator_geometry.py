from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data import calvin_simulator_geometry as simulator_geometry
from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_simulator_geometry import (
    build_calvin_object_removal_pair,
    calvin_segmentation_identity_map,
    extract_robot_base_aabb_centres,
    load_calvin_scene_ranges,
    render_calvin_camera_ownership,
    restore_calvin_archived_state,
    scene_for_global_index,
    scene_object_links,
    validate_calvin_environment_checkout,
)
from picf_next.data.dataset_manifest import build_dataset_file_manifest


def _scene_manifest(root: Path):
    return build_dataset_file_manifest(
        root,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        split_name=root.name,
        relative_paths=("scene_info.npy",),
    )


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(checkout: Path, message: str) -> str:
    _git(checkout, "add", ".")
    _git(
        checkout,
        "-c",
        "user.name=PICF Test",
        "-c",
        "user.email=picf-test@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(checkout, "rev-parse", "HEAD")


def _calvin_checkouts(tmp_path: Path) -> tuple[Path, Path, str, str]:
    parent = tmp_path / "calvin"
    parent.mkdir()
    subprocess.run(["git", "init", "-q", str(parent)], check=True)
    environment = parent / "calvin_env"
    environment.mkdir()
    subprocess.run(["git", "init", "-q", str(environment)], check=True)
    (environment / "calvin_env").mkdir()
    (environment / "calvin_env/__init__.py").write_text("# pinned environment\n")
    (environment / "conf").mkdir()
    (environment / "conf/config_data_collection.yaml").write_text("env: pinned\n")
    environment_commit = _commit(environment, "environment")
    parent_commit = _commit(parent, "parent")
    return parent, environment, parent_commit, environment_commit


def test_calvin_environment_checkout_requires_exact_clean_parent_and_submodule(
    tmp_path: Path,
) -> None:
    _parent, environment, parent_commit, environment_commit = _calvin_checkouts(tmp_path)

    assert validate_calvin_environment_checkout(
        environment,
        expected_calvin_commit=parent_commit,
        expected_calvin_env_commit=environment_commit,
    ) == (parent_commit, environment_commit)


def test_calvin_environment_checkout_rejects_dirty_environment(tmp_path: Path) -> None:
    _parent, environment, parent_commit, environment_commit = _calvin_checkouts(tmp_path)
    (environment / "calvin_env/__init__.py").write_text("# modified\n")

    with pytest.raises(ContractError, match="CALVIN environment checkout is dirty"):
        validate_calvin_environment_checkout(
            environment,
            expected_calvin_commit=parent_commit,
            expected_calvin_env_commit=environment_commit,
        )


def test_calvin_environment_checkout_rejects_dirty_parent(tmp_path: Path) -> None:
    parent, environment, parent_commit, environment_commit = _calvin_checkouts(tmp_path)
    (parent / "unexpected.txt").write_text("untracked\n")

    with pytest.raises(ContractError, match="CALVIN checkout is dirty"):
        validate_calvin_environment_checkout(
            environment,
            expected_calvin_commit=parent_commit,
            expected_calvin_env_commit=environment_commit,
        )


def test_calvin_environment_checkout_rejects_wrong_commit(tmp_path: Path) -> None:
    _parent, environment, parent_commit, _environment_commit = _calvin_checkouts(tmp_path)

    with pytest.raises(ContractError, match="CALVIN environment checkout commit differs"):
        validate_calvin_environment_checkout(
            environment,
            expected_calvin_commit=parent_commit,
            expected_calvin_env_commit="0" * 40,
        )


def test_calvin_environment_checkout_rejects_wrong_parent_commit(tmp_path: Path) -> None:
    _parent, environment, _parent_commit, environment_commit = _calvin_checkouts(tmp_path)

    with pytest.raises(ContractError, match="CALVIN checkout commit differs"):
        validate_calvin_environment_checkout(
            environment,
            expected_calvin_commit="0" * 40,
            expected_calvin_env_commit=environment_commit,
        )


def test_calvin_environment_checkout_rejects_non_root_path(tmp_path: Path) -> None:
    _parent, environment, parent_commit, environment_commit = _calvin_checkouts(tmp_path)

    with pytest.raises(ContractError, match="checkout root differs from the required path"):
        validate_calvin_environment_checkout(
            environment / "calvin_env",
            expected_calvin_commit=parent_commit,
            expected_calvin_env_commit=environment_commit,
        )


def test_environment_instantiation_is_gated_by_checkout_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "calvin_env"
    root.mkdir()

    def reject(_root: Path) -> tuple[str, str]:
        raise ContractError("source identity rejected")

    monkeypatch.setattr(simulator_geometry, "validate_calvin_environment_checkout", reject)

    with pytest.raises(ContractError, match="source identity rejected"):
        simulator_geometry.build_calvin_geometry_environment(
            root,
            scene="calvin_scene_A",
        )


class _FakeBullet:
    def __init__(self, *, malformed: bool = False) -> None:
        self.malformed = malformed
        self.velocity_resets: list[tuple[int, tuple[float, ...], tuple[float, ...]]] = []
        self.collision_refreshes = 0

    def resetBaseVelocity(
        self,
        uid: int,
        *,
        linearVelocity: tuple[float, ...],
        angularVelocity: tuple[float, ...],
        physicsClientId: int,
    ) -> None:
        assert physicsClientId == 17
        self.velocity_resets.append((uid, linearVelocity, angularVelocity))

    def performCollisionDetection(self, *, physicsClientId: int) -> None:
        assert physicsClientId == 17
        self.collision_refreshes += 1

    def getBasePositionAndOrientation(self, _uid: int, *, physicsClientId: int):
        assert physicsClientId == 17
        return (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)

    def invertTransform(self, position, orientation):
        assert orientation == (0.0, 0.0, 0.0, 1.0)
        return tuple(-value for value in position), orientation

    def multiplyTransforms(self, inverse_position, _inverse_orientation, center, orientation):
        return tuple(a + b for a, b in zip(inverse_position, center, strict=True)), orientation

    def getAABB(self, body_id: int, link_index: int, *, physicsClientId: int):
        assert physicsClientId == 17
        if self.malformed and link_index == 4:
            return (0.0, 0.0, 0.0), (float("nan"), 1.0, 1.0)
        if (body_id, link_index) == (2, -1):
            return (1.0, 2.0, 3.0), (3.0, 4.0, 5.0)
        if (body_id, link_index) == (5, 4):
            return (0.0, 1.0, 2.0), (2.0, 3.0, 4.0)
        raise AssertionError((body_id, link_index))


class _FakeEnvironment:
    def __init__(self, *, malformed: bool = False) -> None:
        self.p = _FakeBullet(malformed=malformed)
        self.cid = 17
        self.scene_reset_calls = 0
        self.robot_reset_calls = 0
        self.scene = SimpleNamespace(
            movable_objects=(SimpleNamespace(uid=2),),
            reset=self._reset_scene,
        )
        self.robot = SimpleNamespace(robot_uid=23, reset=self._reset_robot)

    def _reset_scene(self, scene_obs: np.ndarray) -> None:
        assert scene_obs.shape == (4,)
        self.scene_reset_calls += 1

    def _reset_robot(self, robot_obs: np.ndarray) -> None:
        assert robot_obs.shape == (15,)
        self.robot_reset_calls += 1

    def reset(self, *args, **kwargs) -> None:
        raise AssertionError("geometry extraction must not advance simulator time")

    def get_info(self):
        return {
            "scene_info": {
                "movable_objects": {"block_red": {"uid": 2}},
                "fixed_objects": {"table": {"uid": 5, "links": {"base": -1, "button_link": 4}}},
            }
        }


class _FakeCamera:
    def __init__(self, name: str, height: int, width: int) -> None:
        self.name = name
        self.height = height
        self.width = width
        self.nearval = 0.01
        self.farval = 10.0
        self.fov = 60.0
        self.aspect = width / height
        if name == "static":
            self.viewMatrix = (1.0,) * 16
            self.projectionMatrix = (2.0,) * 16
        else:
            self.robot_uid = 23
            self.gripper_cam_link = 7

    def process_rgbd(self, raw, _near, _far):
        return raw[2], raw[3]


class _FakeCameraBullet(_FakeBullet):
    ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX = 11

    def __init__(self) -> None:
        super().__init__()
        self.hidden_links: set[tuple[int, int]] = set()
        self.visual_rgba = {
            (2, -1): (0.8, 0.0, 0.0, 1.0),
            (5, 4): (0.1, 0.1, 0.1, 1.0),
        }

    def getVisualShapeData(self, body_id: int, *, physicsClientId: int):
        assert physicsClientId == 17
        return tuple(
            (
                candidate_body,
                link_index,
                0,
                (1.0, 1.0, 1.0),
                b"",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                rgba,
                -1,
            )
            for (candidate_body, link_index), rgba in self.visual_rgba.items()
            if candidate_body == body_id
        )

    def changeVisualShape(
        self,
        body_id: int,
        link_index: int,
        *,
        rgbaColor,
        physicsClientId: int,
    ) -> None:
        assert physicsClientId == 17
        key = (body_id, link_index)
        assert key in self.visual_rgba
        rgba = tuple(float(value) for value in rgbaColor)
        self.visual_rgba[key] = rgba
        if rgba[3] == 0.0:
            self.hidden_links.add(key)
        else:
            self.hidden_links.discard(key)

    def getLinkState(self, *, bodyUniqueId: int, linkIndex: int, physicsClientId: int):
        assert (bodyUniqueId, linkIndex, physicsClientId) == (23, 7, 17)
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    def getMatrixFromQuaternion(self, orientation):
        assert orientation == (0.0, 0.0, 0.0, 1.0)
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def computeViewMatrix(self, position, target, up):
        assert len(position) == len(target) == len(up) == 3
        return (3.0,) * 16

    def computeProjectionMatrixFOV(self, **kwargs):
        assert set(kwargs) == {"fov", "aspect", "nearVal", "farVal"}
        return (4.0,) * 16

    def getCameraImage(
        self,
        *,
        width: int,
        height: int,
        viewMatrix,
        projectionMatrix,
        flags: int,
        physicsClientId: int,
    ):
        assert flags == 11 and physicsClientId == 17
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.ones((height, width), dtype=np.float32)
        segmentation = np.full((height, width), -1, dtype=np.int64)
        if (2, -1) not in self.hidden_links:
            rgb[0, 0] = (200, 0, 0)
            depth[0, 0] = 0.5
            segmentation[0, 0] = 2
        else:
            rgb[0, 0] = (10, 20, 30)
            depth[0, 0] = 1.5
        if (5, 4) not in self.hidden_links:
            rgb[0, 1] = (20, 20, 20)
            depth[0, 1] = 0.75
            segmentation[0, 1] = 5 + (5 << 24)
        else:
            rgb[0, 1] = (40, 50, 60)
            depth[0, 1] = 1.75
        return width, height, rgb, depth, segmentation


class _FakeCameraEnvironment(_FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.p = _FakeCameraBullet()
        self.cameras = (
            _FakeCamera("static", 200, 200),
            _FakeCamera("gripper", 84, 84),
        )


def test_scene_object_links_are_task_independent_and_stable() -> None:
    scene_info = _FakeEnvironment().get_info()["scene_info"]

    objects = scene_object_links(scene_info)

    assert tuple(item.identity_key for item in objects) == (
        "movable/block_red",
        "part/table/button_link",
    )
    assert tuple((item.body_id, item.link_index) for item in objects) == ((2, -1), (5, 4))
    assert calvin_segmentation_identity_map(scene_info) == {
        2: "movable/block_red",
        5 + (5 << 24): "part/table/button_link",
    }


def test_simulator_geometry_is_aabb_center_in_robot_base_frame() -> None:
    environment = _FakeEnvironment()

    keys, geometry = extract_robot_base_aabb_centres(
        environment,
        scene_obs=np.zeros(4, dtype=np.float32),
        robot_obs=np.zeros(15, dtype=np.float32),
    )

    assert keys == ("movable/block_red", "part/table/button_link")
    expected = np.asarray(
        [
            CALVIN_OBJECT_GEOMETRY_CONTRACT.normalize_values((1.0, 1.0, 1.0)),
            CALVIN_OBJECT_GEOMETRY_CONTRACT.normalize_values((0.0, 0.0, 0.0)),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(geometry, expected)
    assert environment.scene_reset_calls == 1
    assert environment.robot_reset_calls == 1
    assert environment.p.velocity_resets == [(2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    assert environment.p.collision_refreshes == 1


def test_camera_ownership_uses_one_shared_geometry_inventory() -> None:
    renders = render_calvin_camera_ownership(
        _FakeCameraEnvironment(),
        identity_keys=("movable/block_red", "part/table/button_link"),
    )

    assert tuple(item.camera_name for item in renders) == ("static", "gripper")
    assert tuple(item.owner_index.shape for item in renders) == ((200, 200), (84, 84))
    for item in renders:
        assert item.owner_index[0, :3].tolist() == [1, 2, 0]
        assert item.depth_m.dtype == np.float32


def test_same_renderer_object_removal_is_target_local_and_restores_exactly() -> None:
    environment = _FakeCameraEnvironment()

    pair = build_calvin_object_removal_pair(
        environment,
        scene_obs=np.zeros(4, dtype=np.float32),
        robot_obs=np.zeros(15, dtype=np.float32),
        source_global_index=101,
        target_identity_key="part/table/button_link",
    )

    assert pair.source_global_index == 101
    assert pair.target_owner_index == 2
    assert pair.identity_keys == ("movable/block_red", "part/table/button_link")
    assert tuple(camera.target_pixel_count for camera in pair.cameras) == (1, 1)
    assert tuple(camera.changed_pixel_count for camera in pair.cameras) == (1, 1)
    assert pair.contract_dict()["model_input_contains_identity_or_owner"] is False
    for camera in pair.cameras:
        assert camera.factual.owner_index[0, :3].tolist() == [1, 2, 0]
        assert camera.removed.owner_index[0, :3].tolist() == [1, 0, 0]
        assert camera.factual_rgb_sha256 != camera.removed_rgb_sha256
    assert environment.p.visual_rgba[(5, 4)] == (0.1, 0.1, 0.1, 1.0)
    assert not environment.p.hidden_links


def test_same_renderer_object_removal_fails_closed_on_ambiguous_visual_shape() -> None:
    environment = _FakeCameraEnvironment()
    original = environment.p.getVisualShapeData

    def duplicated(body_id: int, *, physicsClientId: int):
        values = original(body_id, physicsClientId=physicsClientId)
        return values + values

    environment.p.getVisualShapeData = duplicated

    with pytest.raises(ContractError, match="exactly one visual shape"):
        build_calvin_object_removal_pair(
            environment,
            scene_obs=np.zeros(4, dtype=np.float32),
            robot_obs=np.zeros(15, dtype=np.float32),
            source_global_index=101,
            target_identity_key="movable/block_red",
        )


def test_same_renderer_object_removal_restores_visual_state_after_render_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _FakeCameraEnvironment()
    original_render = simulator_geometry.render_calvin_camera_ownership
    calls = 0

    def fail_removed_render(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic render failure")
        return original_render(*args, **kwargs)

    monkeypatch.setattr(
        simulator_geometry,
        "render_calvin_camera_ownership",
        fail_removed_render,
    )

    with pytest.raises(RuntimeError, match="synthetic render failure"):
        build_calvin_object_removal_pair(
            environment,
            scene_obs=np.zeros(4, dtype=np.float32),
            robot_obs=np.zeros(15, dtype=np.float32),
            source_global_index=101,
            target_identity_key="movable/block_red",
        )
    assert environment.p.visual_rgba[(2, -1)] == (0.8, 0.0, 0.0, 1.0)
    assert not environment.p.hidden_links


def test_archived_state_restore_rejects_nonfinite_or_wrong_shape() -> None:
    environment = _FakeEnvironment()
    with pytest.raises(ContractError, match="state arrays"):
        restore_calvin_archived_state(
            environment,
            scene_obs=np.asarray([0.0, float("nan"), 0.0, 0.0], dtype=np.float32),
            robot_obs=np.zeros(15, dtype=np.float32),
        )
    with pytest.raises(ContractError, match="state arrays"):
        restore_calvin_archived_state(
            environment,
            scene_obs=np.zeros(4, dtype=np.float32),
            robot_obs=np.zeros(14, dtype=np.float32),
        )


def test_simulator_geometry_rejects_malformed_physics_output() -> None:
    with pytest.raises(ContractError, match="AABB is malformed"):
        extract_robot_base_aabb_centres(
            _FakeEnvironment(malformed=True),
            scene_obs=np.zeros(4, dtype=np.float32),
            robot_obs=np.zeros(15, dtype=np.float32),
        )


def test_calvin_scene_ranges_select_abcd_without_overlap(tmp_path: Path) -> None:
    np.save(
        tmp_path / "scene_info.npy",
        {
            "calvin_scene_A": np.asarray([0, 9], dtype=np.int64),
            "calvin_scene_D": np.asarray([20, 29], dtype=np.int64),
            "calvin_scene_B": np.asarray([10, 19], dtype=np.int64),
        },
        allow_pickle=True,
    )

    ranges = load_calvin_scene_ranges(
        tmp_path,
        dataset_manifest=_scene_manifest(tmp_path),
    )

    assert tuple(item.scene for item in ranges) == (
        "calvin_scene_A",
        "calvin_scene_B",
        "calvin_scene_D",
    )
    assert scene_for_global_index(ranges, 0) == "calvin_scene_A"
    assert scene_for_global_index(ranges, 19) == "calvin_scene_B"
    assert scene_for_global_index(ranges, 20) == "calvin_scene_D"
    with pytest.raises(ContractError, match="unique scene"):
        scene_for_global_index(ranges, 30)


def test_calvin_scene_ranges_reject_unknown_or_overlapping_scenes(tmp_path: Path) -> None:
    np.save(
        tmp_path / "scene_info.npy",
        {
            "calvin_scene_A": np.asarray([0, 10], dtype=np.int64),
            "calvin_scene_B": np.asarray([10, 20], dtype=np.int64),
        },
        allow_pickle=True,
    )
    with pytest.raises(ContractError, match="overlap"):
        load_calvin_scene_ranges(
            tmp_path,
            dataset_manifest=_scene_manifest(tmp_path),
        )

    np.save(
        tmp_path / "scene_info.npy",
        {"benchmark_specific_scene": np.asarray([0, 10], dtype=np.int64)},
        allow_pickle=True,
    )
    with pytest.raises(ContractError, match="malformed"):
        load_calvin_scene_ranges(
            tmp_path,
            dataset_manifest=_scene_manifest(tmp_path),
        )
