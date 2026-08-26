from __future__ import annotations

import hashlib
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinLanguageSegment
from picf_next.data.calvin_geometry_schema import calvin_source_state_sha256
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from tools import audit_calvin_behavior_grounding_support as audit


class _ArrayTensor:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def detach(self) -> _ArrayTensor:
        return self

    def cpu(self) -> _ArrayTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Index:
    def __init__(self) -> None:
        self.static = np.full((8, 8, 3), 32, dtype=np.uint8)
        self.gripper = np.full((6, 6, 3), 64, dtype=np.uint8)

    def _arrays(self, global_index: int) -> dict[str, np.ndarray]:
        scene = np.asarray([global_index, 1.0], dtype=np.float64)
        robot = np.zeros(15, dtype=np.float64)
        robot[:3] = np.asarray((-0.34 + 0.1, -0.46, 0.24))
        return {
            "rel_actions": np.ones(7, dtype=np.float32),
            "rgb_gripper": self.gripper,
            "rgb_static": self.static,
            "robot_obs": robot,
            "scene_obs": scene,
        }

    def validated_source_frame_arrays(
        self,
        global_index: int,
        *,
        fields: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        arrays = self._arrays(global_index)
        return {name: arrays[name] for name in fields}


class _Sidecar:
    identity_keys = (
        "movable/block_blue",
        "movable/block_red",
        "movable/block_pink",
        "part/table/slide_link",
        "part/table/plank_link",
        "part/table/drawer_link",
        "part/table/button_link",
        "part/table/led_link",
        "part/table/switch_link",
        "part/table/light_link",
    )

    def __init__(self, index: _Index) -> None:
        self.index = index

    def source_frame(self, global_index: int) -> SimpleNamespace:
        geometry = np.zeros((len(self.identity_keys), 3), dtype=np.float32)
        geometry[0, 0] = (global_index - 10) * 0.1
        static_owner = np.zeros((8, 8), dtype=np.uint8)
        static_owner[2:6, 2:6] = 1
        gripper_owner = np.zeros((6, 6), dtype=np.uint8)
        gripper_owner[1:3, 1:3] = 1
        cameras = (
            SimpleNamespace(
                camera_name="static",
                owner_index=static_owner,
                owner_supervised=np.ones_like(static_owner, dtype=np.bool_),
                source_rgb_sha256=source_array_sha256("rgb_static", self.index.static),
            ),
            SimpleNamespace(
                camera_name="gripper",
                owner_index=gripper_owner,
                owner_supervised=np.ones_like(gripper_owner, dtype=np.bool_),
                source_rgb_sha256=source_array_sha256("rgb_gripper", self.index.gripper),
            ),
        )
        contract = SimpleNamespace(
            dimension=3,
            normalization_offset=(0.0, 0.0, 0.0),
            normalization_scale=(1.0, 1.0, 1.0),
        )
        return SimpleNamespace(
            cameras=cameras,
            geometry=_ArrayTensor(geometry),
            geometry_contract=contract,
            geometry_supervised=_ArrayTensor(np.ones_like(geometry, dtype=np.bool_)),
            identity_keys=self.identity_keys,
        )

    def source_state_sha256(self, global_index: int) -> str:
        arrays = self.index._arrays(global_index)
        return calvin_source_state_sha256(arrays["scene_obs"], arrays["robot_obs"])


def test_behavior_audit_uses_hash_pinned_external_provenance_views() -> None:
    main_source = inspect.getsource(audit.main)
    run_source = inspect.getsource(audit._run)  # noqa: SLF001

    assert 'parser.add_argument("--sidecar-manifest", required=True, type=Path)' in main_source
    assert 'parser.add_argument("--sidecar-manifest-sha256", required=True)' in main_source
    assert 'parser.add_argument("--source-dataset-manifest", required=True' in main_source
    assert 'parser.add_argument("--source-receipt", required=True' in main_source
    assert 'parser.add_argument("--source-receipt-sha256", required=True)' in main_source
    assert "manifest_path=sidecar_manifest" in run_source
    assert "expected_manifest_sha256=args.sidecar_manifest_sha256" in run_source
    assert "validate_calvin_content_identity_migration(source_manifest, manifest)" in run_source
    assert "validate_calvin_official_source_receipt(" in run_source
    assert audit.AUDIT_SCHEMA == "picf-next.calvin-behavior-grounding-support-audit.v3"


def test_behavior_audit_source_receipt_is_hash_pinned(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    payload = b'{"schema":"fixture"}\n'
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    receipt, actual = audit._load_source_receipt(path, expected)  # noqa: SLF001

    assert receipt == {"schema": "fixture"}
    assert actual == expected
    path.write_bytes(b'{"schema":"changed"}\n')
    with pytest.raises(ContractError, match="hash mismatch"):
        audit._load_source_receipt(path, expected)  # noqa: SLF001


def test_exact_segment_audit_binds_actions_states_geometry_and_visibility() -> None:
    index = _Index()
    sidecar = _Sidecar(index)
    segment = CalvinLanguageSegment(
        3,
        10,
        12,
        "push_blue_block_right",
        "push the blue block right",
        1,
    )
    record, keyframes = audit._audit_exact_segment(  # noqa: SLF001
        index=index,  # type: ignore[arg-type]
        sidecar=sidecar,  # type: ignore[arg-type]
        segment=segment,
        scene="calvin_scene_A",
    )
    summary = record["summary"]
    assert isinstance(summary, dict)
    assert summary["target_motion_rank"] == 1
    assert summary["target_max_displacement_m"] == pytest.approx(0.2)
    assert summary["training_authorized"] is False
    assert record["scene"] == "calvin_scene_A"
    assert keyframes == (10, 11, 12)

    png = audit._render_review(  # noqa: SLF001
        index=index,  # type: ignore[arg-type]
        sidecar=sidecar,  # type: ignore[arg-type]
        segment=segment,
        keyframes=keyframes,
        target_identity_key="movable/block_blue",
    )
    with Image.open(__import__("io").BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.width == 780
        assert image.height > 520

    two_frame_png = audit._render_review(  # noqa: SLF001
        index=index,  # type: ignore[arg-type]
        sidecar=sidecar,  # type: ignore[arg-type]
        segment=segment,
        keyframes=(10, 12),
        target_identity_key="movable/block_blue",
    )
    with Image.open(__import__("io").BytesIO(two_frame_png)) as image:
        assert image.width == 520
        assert image.height > 630


def test_behavior_audit_numeric_distribution_is_finite_and_exact() -> None:
    result = audit._numeric_distribution([1.0, 2.0, 3.0])  # noqa: SLF001
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["minimum"] == 1.0
    assert result["maximum"] == 3.0


@pytest.mark.parametrize(
    ("dataset_id", "dataset_revision", "message"),
    (
        ("mees/calvin-debug-dataset", "sha256:" + "1" * 64, "identity"),
        ("mees/calvin/task_ABC_D", "sha256:" + "1" * 64, "identity"),
        (audit.CALVIN_OFFICIAL_DATASET_ID, "1" * 64, "revision"),
        (audit.CALVIN_OFFICIAL_DATASET_ID, "sha256:" + "z" * 64, "revision"),
    ),
)
def test_behavior_audit_rejects_nonofficial_or_noncontent_identity(
    dataset_id: str,
    dataset_revision: str,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        audit._validate_expected_full_dataset_identity(  # noqa: SLF001
            dataset_id,
            dataset_revision,
        )


def test_behavior_audit_accepts_official_content_identity() -> None:
    audit._validate_expected_full_dataset_identity(  # noqa: SLF001
        audit.CALVIN_OFFICIAL_DATASET_ID,
        "sha256:" + "1" * 64,
    )


def test_behavior_audit_atomic_publication_cleans_failed_partial(tmp_path) -> None:
    output = tmp_path / "audit"
    with pytest.raises(FileExistsError):
        audit._publish_artifact_directory(  # noqa: SLF001
            output_dir=output,
            report={"training_authorized": False},
            visual_payloads=[("same.png", b"one"), ("same.png", b"two")],
        )
    assert not output.exists()
    assert not (tmp_path / f".{output.name}.partial-{os.getpid()}").exists()


def test_behavior_audit_publication_is_durable_and_exclusive(tmp_path) -> None:
    output = tmp_path / "audit"
    audit._publish_artifact_directory(  # noqa: SLF001
        output_dir=output,
        report={"training_authorized": False},
        visual_payloads=[("panel.png", b"panel")],
    )

    assert (output / "panel.png").read_bytes() == b"panel"
    assert not tuple(tmp_path.glob(".*.publish-lock"))
    assert not tuple(tmp_path.glob(".*.partial-*"))
    with pytest.raises(FileExistsError):
        audit._publish_artifact_directory(  # noqa: SLF001
            output_dir=output,
            report={"training_authorized": False},
            visual_payloads=[],
        )
