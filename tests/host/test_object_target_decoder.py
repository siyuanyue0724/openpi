from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.data.object_targets import (  # noqa: E402
    ModalityObjectMembership,
    ObjectStateTable,
    build_object_set_target,
)
from picf_next.models.evidence import (  # noqa: E402
    ModalityProjectionSpec,
    MultimodalBindingProjector,
    NativeTokenBank,
)
from tests.geometry_contract import synthetic_geometry_contract  # noqa: E402

GEOMETRY = synthetic_geometry_contract(2)


def _projection():
    torch.manual_seed(701)
    projector = MultimodalBindingProjector(
        (
            ModalityProjectionSpec("vision", 4),
            ModalityProjectionSpec("sonata", 3),
            ModalityProjectionSpec("touch", 2, require_single_active_group=True),
        ),
        binding_dim=6,
    )
    return projector(
        (
            NativeTokenBank(
                "vision",
                torch.randn(1, 3, 4),
                torch.tensor([[True, True, True]]),
            ),
            NativeTokenBank(
                "sonata",
                torch.randn(1, 2, 3),
                torch.tensor([[True, True]]),
            ),
            NativeTokenBank(
                "touch",
                torch.randn(1, 2, 2),
                torch.tensor([[True, True]]),
                group_id=torch.tensor([[7, 7]]),
            ),
        )
    )


def _vision() -> ModalityObjectMembership:
    return ModalityObjectMembership(
        modality="vision",
        object_ids=("red_block", "drawer"),
        probability=torch.tensor([[1.0, 0.0], [0.8, 0.8], [0.0, 0.0]]),
        token_valid=torch.tensor([True, True, True]),
        supervised=torch.tensor([True, True, True]),
    )


def _touch() -> ModalityObjectMembership:
    return ModalityObjectMembership(
        modality="touch",
        object_ids=("red_block",),
        probability=torch.ones(2, 1),
        token_valid=torch.tensor([True, True]),
        supervised=torch.tensor([True, True]),
    )


def test_builds_all_object_union_and_retains_every_token_row() -> None:
    target = build_object_set_target(
        _projection(), batch_index=0, memberships=(_vision(), _touch())
    )

    assert target.num_objects == 2
    assert target.ownership.shape == (7, 3)
    assert target.token_valid.tolist() == [True] * 7
    assert target.supervision_valid.tolist() == [True, True, True, False, False, True, True]
    assert (target.ownership[3:5] == 0.0).all()

    # Object IDs are sorted: drawer, red_block, context.
    torch.testing.assert_close(target.ownership[0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(target.ownership[2], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(target.ownership[5], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(target.ownership[6], target.ownership[5])


def test_overlap_is_soft_exclusive_mass_instead_of_duplicate_hard_labels() -> None:
    target = build_object_set_target(
        _projection(), batch_index=0, memberships=(_vision(), _touch())
    )
    expected = torch.tensor([4.0, 4.0, 1.0]) / 9.0

    torch.testing.assert_close(target.ownership[1], expected)
    torch.testing.assert_close(target.ownership[1].sum(), torch.tensor(1.0))


def test_explicit_categorical_context_preserves_raster_fractions() -> None:
    raster = ModalityObjectMembership(
        modality="vision",
        object_ids=("red_block", "drawer"),
        probability=torch.tensor([[0.4, 0.3], [0.0, 0.0], [1.0, 0.0]]),
        context_probability=torch.tensor([0.3, 1.0, 0.0]),
        token_valid=torch.tensor([True, True, True]),
        supervised=torch.tensor([True, True, True]),
    )
    target = build_object_set_target(_projection(), batch_index=0, memberships=(raster,))

    # Canonical object order is drawer, red_block, context.
    torch.testing.assert_close(target.ownership[0], torch.tensor([0.3, 0.4, 0.3]))
    torch.testing.assert_close(target.ownership[1], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(target.ownership[2], torch.tensor([0.0, 1.0, 0.0]))


def test_explicit_categorical_context_must_form_a_simplex() -> None:
    invalid = ModalityObjectMembership(
        modality="vision",
        object_ids=("red_block",),
        probability=torch.tensor([[0.7], [0.0], [0.0]]),
        context_probability=torch.tensor([0.7, 1.0, 1.0]),
        token_valid=torch.tensor([True, True, True]),
        supervised=torch.tensor([True, True, True]),
    )
    with pytest.raises(ValueError, match="sum to one"):
        build_object_set_target(_projection(), batch_index=0, memberships=(invalid,))


def test_unlabelled_modality_is_ignored_instead_of_becoming_context() -> None:
    target = build_object_set_target(_projection(), batch_index=0, memberships=(_vision(),))

    assert not target.supervision_valid[3:7].any()
    assert (target.ownership[3:7] == 0.0).all()


def test_explicit_empty_inventory_supervises_known_context() -> None:
    empty = ModalityObjectMembership(
        modality="sonata",
        object_ids=(),
        probability=torch.empty(2, 0),
        token_valid=torch.tensor([True, True]),
        supervised=torch.tensor([True, True]),
    )
    target = build_object_set_target(_projection(), batch_index=0, memberships=(empty,))

    assert target.num_objects == 0
    torch.testing.assert_close(target.ownership[3:5], torch.ones(2, 1))


def test_inventory_defaults_to_partial_and_contract_is_type_checked() -> None:
    target = build_object_set_target(
        _projection(),
        batch_index=0,
        memberships=(_vision(),),
    )

    assert target.object_inventory_complete is False

    with pytest.raises(ValueError, match="object_inventory_complete"):
        build_object_set_target(
            _projection(),
            batch_index=0,
            memberships=(_vision(),),
            object_inventory_complete=1,  # type: ignore[arg-type]
        )


def test_state_table_is_reordered_to_canonical_object_order() -> None:
    state = ObjectStateTable(
        object_ids=("red_block", "drawer"),
        geometry=torch.tensor([[9.0, 0.0], [0.0, 3.0]]),
        geometry_variance=torch.tensor([[0.9, 0.0], [0.0, 0.3]]),
        geometry_supervised=torch.tensor([[True, False], [False, True]]),
        geometry_contract=GEOMETRY,
    )
    target = build_object_set_target(
        _projection(), batch_index=0, memberships=(_vision(),), state=state
    )

    torch.testing.assert_close(target.geometry, torch.tensor([[0.0, 3.0], [9.0, 0.0]]))
    torch.testing.assert_close(
        target.geometry_variance,
        torch.tensor([[0.0, 0.3], [0.9, 0.0]]),
    )
    assert torch.equal(
        target.geometry_supervised,
        torch.tensor([[False, True], [True, False]]),
    )


def test_geometry_supervision_requires_geometry_values() -> None:
    state = ObjectStateTable(
        object_ids=("red_block", "drawer"),
        geometry_supervised=torch.ones(2, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="requires geometry values"):
        build_object_set_target(
            _projection(),
            batch_index=0,
            memberships=(_vision(),),
            state=state,
        )


def test_geometry_variance_requires_geometry_values() -> None:
    state = ObjectStateTable(
        object_ids=("red_block", "drawer"),
        geometry_variance=torch.ones(2, 2),
    )

    with pytest.raises(ValueError, match="variance requires geometry values"):
        build_object_set_target(
            _projection(),
            batch_index=0,
            memberships=(_vision(),),
            state=state,
        )


def test_target_preserves_only_explicit_loss_side_temporal_identity_keys() -> None:
    target = build_object_set_target(
        _projection(),
        batch_index=0,
        memberships=(_vision(),),
        temporal_identity_by_object={
            "drawer": "episode-7/body-8",
            "red_block": "episode-7/body-3",
        },
    )

    assert target.temporal_identity_keys == (
        "episode-7/body-8",
        "episode-7/body-3",
    )

    with pytest.raises(ValueError, match="exactly cover"):
        build_object_set_target(
            _projection(),
            batch_index=0,
            memberships=(_vision(),),
            temporal_identity_by_object={"wrong": "episode-7/body-3"},
        )


def test_active_touch_group_must_be_wholly_supervised_and_share_one_object() -> None:
    partly_supervised = ModalityObjectMembership(
        modality="touch",
        object_ids=("red_block",),
        probability=torch.tensor([[1.0], [0.0]]),
        token_valid=torch.tensor([True, True]),
        supervised=torch.tensor([True, False]),
    )
    with pytest.raises(ValueError, match="partly supervised"):
        build_object_set_target(_projection(), batch_index=0, memberships=(partly_supervised,))

    split_objects = ModalityObjectMembership(
        modality="touch",
        object_ids=("red_block", "drawer"),
        probability=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        token_valid=torch.tensor([True, True]),
        supervised=torch.tensor([True, True]),
    )
    with pytest.raises(ValueError, match="share one object assignment"):
        build_object_set_target(_projection(), batch_index=0, memberships=(split_objects,))


def test_rejects_unknown_modality_and_unsupported_object() -> None:
    unknown = ModalityObjectMembership(
        modality="task_owner",
        object_ids=("red_block",),
        probability=torch.ones(1, 1),
        token_valid=torch.tensor([True]),
        supervised=torch.tensor([True]),
    )
    with pytest.raises(ValueError, match="unknown modalities"):
        build_object_set_target(_projection(), batch_index=0, memberships=(unknown,))

    unsupported = ModalityObjectMembership(
        modality="vision",
        object_ids=("ghost",),
        probability=torch.zeros(3, 1),
        token_valid=torch.tensor([True, True, True]),
        supervised=torch.tensor([True, True, True]),
    )
    with pytest.raises(ValueError, match="no supervised support"):
        build_object_set_target(_projection(), batch_index=0, memberships=(unsupported,))
