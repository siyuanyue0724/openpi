from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_task_applicability import (
    CALVIN_OFFICIAL_ANNOTATIONS_SHA256,
    CALVIN_OFFICIAL_TASKS_SHA256,
    CalvinJointState,
    CalvinTaskApplicabilityState,
    build_same_observation_group,
    calvin_state_applicable_tasks,
    extract_calvin_task_applicability_state,
    load_official_calvin_annotations,
    verify_official_calvin_task_config,
)


def _state() -> CalvinTaskApplicabilityState:
    return CalvinTaskApplicabilityState(
        doors=(
            CalvinJointState("base__drawer", 0.0, 0.0, 0.24),
            CalvinJointState("base__slide", 0.28, 0.0, 0.56),
        ),
        light_states=(("led", 0), ("lightbulb", 1)),
        block_support_links=(
            ("block_blue", "base_link"),
            ("block_pink", "drawer_link"),
        ),
    )


class _FakeBullet:
    def getJointInfo(self, uid: int, joint_index: int, *, physicsClientId: int):
        assert uid == 7
        assert physicsClientId == 3
        limits = {
            0: (0.0, 0.24),
            1: (0.0, 0.56),
        }
        lower, upper = limits[joint_index]
        return (None,) * 8 + (lower, upper)


def _environment_info() -> dict[str, object]:
    return {
        "scene_info": {
            "doors": {
                "base__drawer": {"current_state": 0.0},
                "base__slide": {"current_state": 0.28},
            },
            "lights": {
                "led": {"logical_state": 0},
                "lightbulb": {"logical_state": 1},
            },
            "fixed_objects": {
                "table": {
                    "uid": 11,
                    "links": {
                        "base_link": -1,
                        "plank_link": 4,
                        "drawer_link": 5,
                    },
                }
            },
            "movable_objects": {
                "block_blue": {"contacts": ((0, 0, 11, 0, -1),)},
                "block_pink": {"contacts": ((0, 0, 11, 0, 5),)},
                "block_red": {"contacts": ()},
            },
        }
    }


def _environment(info: dict[str, object] | None = None) -> SimpleNamespace:
    payload = _environment_info() if info is None else info
    return SimpleNamespace(
        cid=3,
        get_info=lambda: payload,
        p=_FakeBullet(),
        scene=SimpleNamespace(
            doors=(
                SimpleNamespace(name="base__drawer", uid=7, joint_index=0),
                SimpleNamespace(name="base__slide", uid=7, joint_index=1),
            )
        ),
    )


def test_state_applicability_uses_official_preconditions_and_exact_targets() -> None:
    tasks = calvin_state_applicable_tasks(_state())
    by_key = {item.task_key: item for item in tasks}

    assert set(by_key) == {
        "lift_blue_block_table",
        "lift_pink_block_drawer",
        "move_slider_left",
        "move_slider_right",
        "open_drawer",
        "turn_off_lightbulb",
        "turn_on_led",
    }
    assert by_key["turn_on_led"].target_identity_key == "part/table/button_link"
    assert by_key["turn_off_lightbulb"].target_identity_key == "part/table/switch_link"
    assert by_key["lift_blue_block_table"].target_identity_key == "movable/block_blue"


def test_state_applicability_rejects_motion_beyond_joint_limit() -> None:
    state = CalvinTaskApplicabilityState(
        doors=(
            CalvinJointState("base__drawer", 0.20, 0.0, 0.24),
            CalvinJointState("base__slide", 0.50, 0.0, 0.56),
        ),
        light_states=(("led", 1), ("lightbulb", 0)),
        block_support_links=(),
    )
    keys = {item.task_key for item in calvin_state_applicable_tasks(state)}

    assert "open_drawer" not in keys
    assert "move_slider_left" not in keys
    assert {"close_drawer", "move_slider_right", "turn_off_led", "turn_on_lightbulb"} <= keys


def test_same_observation_group_has_distinct_visible_targets_and_is_deterministic() -> None:
    tasks = calvin_state_applicable_tasks(_state())
    annotations = {
        item.task_key: (
            f"official prompt a for {item.task_key}",
            f"official prompt b for {item.task_key}",
        )
        for item in tasks
    }
    kwargs = {
        "source_global_index": 17,
        "source_state_sha256": "a" * 64,
        "visible_identity_keys": (
            "movable/block_blue",
            "movable/block_pink",
            "part/table/button_link",
            "part/table/switch_link",
        ),
        "applicable_tasks": tasks,
        "annotations": annotations,
        "maximum_variants": 4,
    }

    first = build_same_observation_group(**kwargs)
    second = build_same_observation_group(**kwargs)

    assert first == second
    assert first is not None
    assert len(first.variants) == 4
    assert len({item.target_identity_key for item in first.variants}) == 4
    assert first.as_dict()["model_input_contains_simulator_state_or_identity"] is False


def test_same_observation_group_skips_frames_without_two_visible_targets() -> None:
    tasks = calvin_state_applicable_tasks(_state())
    annotations = {item.task_key: (item.task_key,) for item in tasks}

    assert (
        build_same_observation_group(
            source_global_index=17,
            source_state_sha256="b" * 64,
            visible_identity_keys=("part/table/button_link",),
            applicable_tasks=tasks,
            annotations=annotations,
            maximum_variants=2,
        )
        is None
    )


def test_same_observation_group_validates_source_before_skipping() -> None:
    tasks = calvin_state_applicable_tasks(_state())
    annotations = {item.task_key: (item.task_key,) for item in tasks}

    with pytest.raises(ContractError, match="source index"):
        build_same_observation_group(
            source_global_index=-1,
            source_state_sha256="b" * 64,
            visible_identity_keys=("part/table/button_link",),
            applicable_tasks=tasks,
            annotations=annotations,
            maximum_variants=2,
        )


def test_annotation_loader_is_content_pinned(tmp_path: Path) -> None:
    source = tmp_path / "annotations.yaml"
    source.write_text("task: [prompt]\n", encoding="ascii")
    assert hashlib.sha256(source.read_bytes()).hexdigest() != CALVIN_OFFICIAL_ANNOTATIONS_SHA256

    with pytest.raises(ContractError, match="SHA-256"):
        load_official_calvin_annotations(source)


def test_task_config_is_content_pinned(tmp_path: Path) -> None:
    source = tmp_path / "tasks.yaml"
    source.write_text("task: predicate\n", encoding="ascii")
    assert hashlib.sha256(source.read_bytes()).hexdigest() != CALVIN_OFFICIAL_TASKS_SHA256

    with pytest.raises(ContractError, match="SHA-256"):
        verify_official_calvin_task_config(source)


def test_joint_state_contract_rejects_out_of_range_position() -> None:
    with pytest.raises(ContractError, match="outside"):
        CalvinJointState("base__drawer", 0.25, 0.0, 0.24)


def test_joint_state_projects_bounded_simulator_constraint_residual() -> None:
    joint = CalvinJointState("base__slide", 0.3044610694927849, 0.0, 0.304)

    assert joint.position == 0.3044610694927849
    assert joint.feasible_position == 0.304

    state = CalvinTaskApplicabilityState(
        doors=(
            CalvinJointState("base__drawer", 0.0, 0.0, 0.275),
            joint,
        ),
        light_states=(("led", 0), ("lightbulb", 1)),
        block_support_links=(),
    )
    keys = {item.task_key for item in calvin_state_applicable_tasks(state)}

    assert "move_slider_right" in keys
    assert "move_slider_left" not in keys


def test_joint_state_rejects_large_constraint_residual() -> None:
    with pytest.raises(ContractError, match="outside"):
        CalvinJointState("base__slide", 0.308, 0.0, 0.304)


def test_environment_extractor_uses_exact_joint_light_and_contact_state() -> None:
    extracted = extract_calvin_task_applicability_state(_environment())

    assert extracted == _state()


def test_environment_extractor_rejects_missing_nested_inventory_entry() -> None:
    info = deepcopy(_environment_info())
    del info["scene_info"]["lights"]["led"]  # type: ignore[index]

    with pytest.raises(ContractError, match="led scene info"):
        extract_calvin_task_applicability_state(_environment(info))


def test_environment_extractor_rejects_coerced_table_link_index() -> None:
    info = deepcopy(_environment_info())
    info["scene_info"]["fixed_objects"]["table"]["links"]["base_link"] = "-1"  # type: ignore[index]

    with pytest.raises(ContractError, match="base_link index"):
        extract_calvin_task_applicability_state(_environment(info))


def test_environment_extractor_rejects_malformed_contact_identifier() -> None:
    info = deepcopy(_environment_info())
    info["scene_info"]["movable_objects"]["block_blue"]["contacts"] = (  # type: ignore[index]
        (0, 0, True, 0, -1),
    )

    with pytest.raises(ContractError, match="contact body ID"):
        extract_calvin_task_applicability_state(_environment(info))


def test_applicability_state_rejects_malformed_nested_records() -> None:
    with pytest.raises(ContractError, match="light-state records"):
        CalvinTaskApplicabilityState(
            doors=_state().doors,
            light_states=(("led", True), ("lightbulb", 0)),  # type: ignore[arg-type]
            block_support_links=(),
        )

    with pytest.raises(ContractError, match="support contacts"):
        CalvinTaskApplicabilityState(
            doors=_state().doors,
            light_states=(("led", 0), ("lightbulb", 1)),
            block_support_links=(("block_blue", 7),),  # type: ignore[arg-type]
        )


def test_same_observation_builder_rejects_text_as_visible_inventory() -> None:
    with pytest.raises(ContractError, match="visible identity keys"):
        build_same_observation_group(
            source_global_index=17,
            source_state_sha256="b" * 64,
            visible_identity_keys="part/table/button_link",  # type: ignore[arg-type]
            applicable_tasks=calvin_state_applicable_tasks(_state()),
            annotations={},
            maximum_variants=2,
        )
