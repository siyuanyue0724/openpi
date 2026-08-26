from __future__ import annotations

from pathlib import Path

import pytest

from picf_next.eval.calvin_task_relevance import (
    CALVIN_SCENE_CONFIG_SHA256,
    CALVIN_TABLE_URDF_SHA256,
    CALVIN_TASK_CONFIG_SHA256,
    CALVIN_TASK_PROTOCOL_SOURCE_SHA256,
    calvin_exact_task_loss_identities,
    calvin_task_physical_relevance,
    calvin_task_physical_relevance_inventory,
    select_hidden_task_rows,
    select_witnessed_task_rows,
    validate_calvin_task_protocol_inventory,
    validate_calvin_task_protocol_source,
    validate_calvin_task_protocol_sources,
)


def test_calvin_task_protocol_inventory_is_complete_and_sorted() -> None:
    inventory = calvin_task_physical_relevance_inventory()
    keys = tuple(item.task_key for item in inventory)
    assert len(keys) == 34
    assert keys == tuple(sorted(keys))
    assert len(set(keys)) == len(keys)


def test_exact_calvin_tasks_name_physical_action_targets() -> None:
    red = calvin_task_physical_relevance("lift_red_block_table")
    led = calvin_task_physical_relevance("turn_on_led")
    slider = calvin_task_physical_relevance("move_slider_left")

    assert red.action_target_identity_keys == ("movable/block_red",)
    assert led.action_target_identity_keys == ("part/table/button_link",)
    assert led.outcome_identity_keys == ("part/table/led_link",)
    assert slider.action_target_identity_keys == ("part/table/slide_link",)
    assert red.exact_action_target is True
    assert calvin_exact_task_loss_identities("lift_red_block_table") == ("movable/block_red",)
    assert calvin_exact_task_loss_identities("stack_block") is None


def test_ambiguous_calvin_tasks_fail_closed() -> None:
    relevance = calvin_task_physical_relevance("stack_block")

    assert relevance.exact_action_target is False
    assert relevance.action_target_identity_keys == ()
    assert "state dependent" in str(relevance.exclusion_reason)
    with pytest.raises(KeyError, match="absent from the pinned CALVIN protocol"):
        calvin_task_physical_relevance("invented_task")


def test_hidden_task_row_requires_witness_age_and_current_miss() -> None:
    selected = select_hidden_task_rows(
        task_key="push_red_block_right",
        identity_keys_by_row=(None, "movable/block_red", "part/table/button_link"),
        row_valid=(False, True, True),
        measurement_age_s=(0.0, 2.0 / 30.0, 3.0 / 30.0),
        currently_measurable_identity_keys=("part/table/button_link",),
        reference_delta_t_s=1.0 / 30.0,
    )

    assert selected.eligible is True
    assert selected.row_indices == (1,)
    assert selected.row_identity_keys == ("movable/block_red",)

    visible = select_hidden_task_rows(
        task_key="push_red_block_right",
        identity_keys_by_row=(None, "movable/block_red", "part/table/button_link"),
        row_valid=(False, True, True),
        measurement_age_s=(0.0, 2.0 / 30.0, 3.0 / 30.0),
        currently_measurable_identity_keys=("movable/block_red",),
        reference_delta_t_s=1.0 / 30.0,
    )
    assert visible.eligible is False
    assert visible.reason == "witnessed action-target identity is currently measurable"


def test_witnessed_task_row_is_selected_before_controlled_occlusion() -> None:
    selected = select_witnessed_task_rows(
        task_key="push_red_block_right",
        identity_keys_by_row=(None, "movable/block_red", "part/table/button_link"),
        row_valid=(False, True, True),
    )

    assert selected.eligible is True
    assert selected.row_indices == (1,)
    assert selected.row_identity_keys == ("movable/block_red",)

    absent = select_witnessed_task_rows(
        task_key="push_red_block_right",
        identity_keys_by_row=(None, "movable/block_blue", "part/table/button_link"),
        row_valid=(False, True, True),
    )
    assert absent.eligible is False
    assert absent.exact_action_target is True

    ambiguous = select_witnessed_task_rows(
        task_key="stack_block",
        identity_keys_by_row=(None, "movable/block_red", "movable/block_blue"),
        row_valid=(False, True, True),
    )
    assert ambiguous.eligible is False
    assert ambiguous.exact_action_target is False


def test_hidden_task_row_rejects_ambiguous_and_inconsistent_attribution() -> None:
    ambiguous = select_hidden_task_rows(
        task_key="stack_block",
        identity_keys_by_row=("movable/block_red", None),
        row_valid=(True, False),
        measurement_age_s=(1.0, 0.0),
        currently_measurable_identity_keys=(),
        reference_delta_t_s=1.0 / 30.0,
    )
    assert ambiguous.eligible is False
    assert ambiguous.exact_action_target is False

    with pytest.raises(ValueError, match="cannot occupy two posterior rows"):
        select_hidden_task_rows(
            task_key="push_red_block_right",
            identity_keys_by_row=("movable/block_red", "movable/block_red"),
            row_valid=(True, True),
            measurement_age_s=(1.0, 1.0),
            currently_measurable_identity_keys=(),
            reference_delta_t_s=1.0 / 30.0,
        )


def test_task_protocol_requires_its_complete_physical_ontology() -> None:
    inventory = (
        "movable/block_blue",
        "movable/block_pink",
        "movable/block_red",
        "part/table/button_link",
        "part/table/drawer_link",
        "part/table/led_link",
        "part/table/light_link",
        "part/table/plank_link",
        "part/table/slide_link",
        "part/table/switch_link",
    )

    required = validate_calvin_task_protocol_inventory(inventory)
    assert set(required) == set(inventory)
    with pytest.raises(ValueError, match="absent from the sidecar"):
        validate_calvin_task_protocol_inventory(inventory[:-1])


def test_reviewed_calvin_task_source_is_hash_pinned(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "references/source_checkouts/calvin/calvin_models/conf/callbacks/rollout/tasks"
        / "new_playtable_tasks.yaml"
    )
    if not source.is_file():
        pytest.skip("the optional official CALVIN source checkout is absent")

    assert validate_calvin_task_protocol_source(source) == CALVIN_TASK_CONFIG_SHA256
    sources = validate_calvin_task_protocol_sources(root / "references/source_checkouts/calvin")
    assert len(sources) == 10
    assert sources == CALVIN_TASK_PROTOCOL_SOURCE_SHA256
    assert set(CALVIN_SCENE_CONFIG_SHA256) == {
        "calvin_scene_A",
        "calvin_scene_B",
        "calvin_scene_C",
        "calvin_scene_D",
    }
    assert set(CALVIN_TABLE_URDF_SHA256) == {
        "calvin_table_A",
        "calvin_table_B",
        "calvin_table_C",
        "calvin_table_D",
    }
    changed = tmp_path / "tasks.yaml"
    changed.write_text("tasks: {}\n", encoding="ascii")
    with pytest.raises(ValueError, match="differs from the reviewed"):
        validate_calvin_task_protocol_source(changed)
