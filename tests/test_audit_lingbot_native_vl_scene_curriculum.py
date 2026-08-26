from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenSceneGroundingRecord,
    CalvinQwenSceneObject,
)
from tools.audit_lingbot_native_vl_scene_curriculum import (
    _PANEL_SIZE,
    _artifact_payload,
    _bbox_intersection_area,
    _canonical_bytes,
    _draw_scene_panel,
    _select_source_disjoint_scene_bank,
    _validated_checkout_revision,
    _verified_sha256_file,
    _write_contact_sheet_pages,
)

_TOOL = Path(__file__).resolve().parents[1] / "tools/audit_lingbot_native_vl_scene_curriculum.py"


def _scene_record(
    global_index: int,
    camera_name: str,
    identity_keys: tuple[str, ...],
) -> CalvinQwenSceneGroundingRecord:
    shape = (200, 200, 3) if camera_name == "static" else (84, 84, 3)
    image = np.full(shape, global_index % 251, dtype=np.uint8)
    image.setflags(write=False)
    objects = tuple(
        CalvinQwenSceneObject(
            identity_key=identity_key,
            bbox_xyxy=(index, index, index + 4, index + 4),
            visible_owner_pixels=16,
            projected_target_mass=0.5,
            positive_visual_token_count=1,
        )
        for index, identity_key in enumerate(identity_keys, start=1)
    )
    return CalvinQwenSceneGroundingRecord(
        global_index=global_index,
        camera_name=camera_name,
        host_image_key=(
            "observation.images.image"
            if camera_name == "static"
            else "observation.images.wrist_image"
        ),
        category_identity_order=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
        objects=objects,
        subpatch_objects=(),
        absent_identity_keys=tuple(
            key for key in CALVIN_QWEN_SCENE_IDENTITY_ORDER if key not in identity_keys
        ),
        minimum_projected_target_mass=0.25,
        visual_lattice=8,
        image_grid_thw=(1, 16, 16),
        patch_size=16,
        merge_size=2,
        image=image,
        source_rgb_sha256=source_array_sha256(
            "rgb_static" if camera_name == "static" else "rgb_gripper",
            image,
        ),
    )


def test_scene_curriculum_audit_geometry_and_canonical_json_are_strict() -> None:
    assert _bbox_intersection_area((0, 0, 10, 10), (5, 4, 12, 20)) == 30
    assert _bbox_intersection_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert _canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError, match="canonical JSON"):
        _canonical_bytes({"loss": float("nan")})
    with pytest.raises(ValueError, match="canonical JSON"):
        _canonical_bytes({1: "non-string key"})
    with pytest.raises(ValueError, match="canonical JSON"):
        _canonical_bytes({"payload": b"not JSON"})


def test_scene_curriculum_artifact_hash_binds_exact_unsigned_content() -> None:
    content: dict[str, object] = {"schema": "test.v1", "status": "PASS", "rows": [1, 2]}
    artifact_sha256, payload = _artifact_payload(content)
    decoded = json.loads(payload)

    assert payload.endswith(b"\n")
    assert decoded["artifact_sha256"] == artifact_sha256
    assert artifact_sha256 == hashlib.sha256(_canonical_bytes(content)).hexdigest()
    assert (
        hashlib.sha256(_canonical_bytes({**content, "rows": [2, 1]})).hexdigest() != artifact_sha256
    )
    with pytest.raises(ValueError, match="already contains"):
        _artifact_payload({**content, "artifact_sha256": "0" * 64})


def test_scene_curriculum_contact_sheets_are_paginated_and_content_addressed(
    tmp_path: Path,
) -> None:
    (tmp_path / "contact_sheets").mkdir()
    panels = [
        _draw_scene_panel(
            _scene_record(10, "static", ("movable/block_blue", "part/table/button_link")),
            title_lines=("task=inspect blue block and button", "camera=static"),
        ),
        *[Image.new("RGB", _PANEL_SIZE, (index, 0, 0)) for index in range(16)],
    ]
    outputs = _write_contact_sheet_pages(
        tmp_path,
        prefix="audit",
        panels=panels,
    )

    assert [item["panel_count"] for item in outputs] == [16, 1]
    assert all(len(item["sha256"]) == 64 for item in outputs)
    for item in outputs:
        path = tmp_path / str(item["path"])
        assert path.is_file()
        with Image.open(path) as rendered:
            assert rendered.format == "PNG"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    with pytest.raises(ValueError, match="at least one"):
        _write_contact_sheet_pages(tmp_path, prefix="empty", panels=[])
    with pytest.raises(ValueError, match="prefix"):
        _write_contact_sheet_pages(tmp_path, prefix="../escape", panels=panels[:1])
    with pytest.raises(ValueError, match="columns"):
        _write_contact_sheet_pages(tmp_path, prefix="columns", panels=panels[:1], columns=True)
    with pytest.raises(ValueError, match="canonical RGB"):
        _write_contact_sheet_pages(
            tmp_path,
            prefix="wrong_shape",
            panels=[Image.new("RGB", (1, 1))],
        )


def test_scene_curriculum_panel_renders_canonical_rgb_geometry() -> None:
    panel = _draw_scene_panel(
        _scene_record(10, "static", ("movable/block_blue",)),
        title_lines=("task=move blue block", "camera=static"),
    )

    assert panel.mode == "RGB"
    assert panel.size == _PANEL_SIZE
    assert np.asarray(panel).var() > 0
    assert np.asarray(panel)[:, 340:].var() > 0


def test_scene_curriculum_hash_binding_rejects_aliases_and_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"schema":"test"}\n')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert _verified_sha256_file(artifact, digest, name="artifact") == digest
    with pytest.raises(ValueError, match="SHA-256 is invalid"):
        _verified_sha256_file(artifact, digest.upper(), name="artifact")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        _verified_sha256_file(artifact, "0" * 64, name="artifact")

    alias = tmp_path / "alias.json"
    alias.symlink_to(artifact)
    with pytest.raises(ValueError, match="one real file"):
        _verified_sha256_file(alias, digest, name="artifact")


def test_scene_curriculum_revision_binding_rejects_dirty_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="ascii")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=PICF Test",
            "-c",
            "user.email=picf-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )

    assert len(_validated_checkout_revision(tmp_path)) == 40
    tracked.write_text("dirty\n", encoding="ascii")
    with pytest.raises(ValueError, match="clean revision-bound"):
        _validated_checkout_revision(tmp_path)


def test_scene_curriculum_bank_is_source_disjoint_deterministic_and_covering() -> None:
    blue = "movable/block_blue"
    button = "part/table/button_link"
    records = {
        (0, "static"): _scene_record(10, "static", (blue,)),
        (1, "static"): _scene_record(11, "static", (blue,)),
        (2, "gripper"): _scene_record(12, "gripper", (button,)),
        (3, "static"): _scene_record(13, "static", (blue, button)),
    }
    selected = _select_source_disjoint_scene_bank(
        records,
        excluded_group_indices={0},
        curriculum_artifact_sha256="a" * 64,
        bank_size=2,
    )
    replay = _select_source_disjoint_scene_bank(
        records,
        excluded_group_indices={0},
        curriculum_artifact_sha256="a" * 64,
        bank_size=2,
    )

    assert selected == replay
    assert len({group_index for group_index, _record in selected}) == 2
    assert all(group_index != 0 for group_index, _record in selected)
    covered = {
        (item.identity_key, record.camera_name)
        for _group_index, record in selected
        for item in record.objects
    }
    assert covered == {(blue, "static"), (button, "static"), (button, "gripper")}


def test_scene_curriculum_bank_covers_cameras_even_for_empty_scenes() -> None:
    blue = "movable/block_blue"
    records = {
        (0, "static"): _scene_record(10, "static", (blue,)),
        (1, "static"): _scene_record(11, "static", (blue,)),
        (2, "gripper"): _scene_record(12, "gripper", ()),
    }

    selected = _select_source_disjoint_scene_bank(
        records,
        excluded_group_indices=set(),
        curriculum_artifact_sha256="b" * 64,
        bank_size=2,
    )

    assert {record.camera_name for _group_index, record in selected} == {
        "static",
        "gripper",
    }


def test_scene_curriculum_bank_backtracks_around_same_group_camera_conflict() -> None:
    blue = "movable/block_blue"
    button = "part/table/button_link"
    records = {
        (0, "static"): _scene_record(10, "static", (blue,)),
        (0, "gripper"): _scene_record(10, "gripper", (button,)),
        (1, "static"): _scene_record(11, "static", (blue,)),
    }

    selected = _select_source_disjoint_scene_bank(
        records,
        excluded_group_indices=set(),
        curriculum_artifact_sha256="d" * 64,
        bank_size=2,
    )

    assert {(group_index, record.camera_name) for group_index, record in selected} == {
        (0, "gripper"),
        (1, "static"),
    }


def test_scene_curriculum_bank_selects_exact_32_after_64_step_arm_exclusion() -> None:
    records = {}
    for group_index in range(72):
        identity_key = CALVIN_QWEN_SCENE_IDENTITY_ORDER[
            group_index % len(CALVIN_QWEN_SCENE_IDENTITY_ORDER)
        ]
        for camera_name in ("static", "gripper"):
            records[(group_index, camera_name)] = _scene_record(
                100 + group_index,
                camera_name,
                (identity_key,),
            )
    excluded = set(range(32))

    selected = _select_source_disjoint_scene_bank(
        records,
        excluded_group_indices=excluded,
        curriculum_artifact_sha256="e" * 64,
    )

    assert len(selected) == 32
    assert len({group_index for group_index, _record in selected}) == 32
    assert len({record.global_index for _group_index, record in selected}) == 32
    assert len({record.source_rgb_sha256 for _group_index, record in selected}) == 32
    assert not ({group_index for group_index, _record in selected} & excluded)
    assert {
        (item.identity_key, record.camera_name)
        for _group_index, record in selected
        for item in record.objects
    } == {
        (identity_key, camera_name)
        for identity_key in CALVIN_QWEN_SCENE_IDENTITY_ORDER
        for camera_name in ("static", "gripper")
    }


def test_scene_curriculum_bank_rejects_non_disjoint_or_misbound_sources() -> None:
    blue = "movable/block_blue"
    canonical = _scene_record(10, "static", (blue,))
    with pytest.raises(ValueError, match="curriculum artifact"):
        _select_source_disjoint_scene_bank(
            {(0, "static"): canonical},
            excluded_group_indices=set(),
            curriculum_artifact_sha256="A" * 64,
            bank_size=1,
        )
    with pytest.raises(ContractError, match="key disagrees with record camera"):
        _select_source_disjoint_scene_bank(
            {(0, "gripper"): canonical},
            excluded_group_indices=set(),
            curriculum_artifact_sha256="a" * 64,
            bank_size=1,
        )
    duplicate_source = {
        (0, "static"): canonical,
        (1, "gripper"): _scene_record(10, "gripper", (blue,)),
    }
    with pytest.raises(ContractError, match="not source-frame disjoint"):
        _select_source_disjoint_scene_bank(
            duplicate_source,
            excluded_group_indices={0},
            curriculum_artifact_sha256="a" * 64,
            bank_size=1,
        )


def test_scene_curriculum_bank_excludes_all_64_arm_source_aliases() -> None:
    blue = "movable/block_blue"
    arm_static = _scene_record(10, "static", (blue,))
    arm_gripper = _scene_record(10, "gripper", (blue,))
    records = {
        (0, "static"): arm_static,
        (0, "gripper"): arm_gripper,
        (1, "static"): _scene_record(11, "static", (blue,)),
        (2, "gripper"): _scene_record(12, "gripper", (blue,)),
    }

    selected = _select_source_disjoint_scene_bank(
        records,
        excluded_group_indices={0},
        curriculum_artifact_sha256="c" * 64,
        bank_size=2,
    )

    assert {group_index for group_index, _record in selected} == {1, 2}
    assert all(record.global_index != 10 for _group_index, record in selected)
    assert all(
        record.source_rgb_sha256
        not in {arm_static.source_rgb_sha256, arm_gripper.source_rgb_sha256}
        for _group_index, record in selected
    )


def test_scene_curriculum_audit_help_works_outside_the_repository(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(_TOOL), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--curriculum-plan" in result.stdout
    assert "--physical-sidecar-root" in result.stdout
    assert "--output-dir" in result.stdout
