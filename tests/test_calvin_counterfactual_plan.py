from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_counterfactual_plan import (
    CalvinCounterfactualCandidate,
    CalvinCounterfactualPlanConfig,
    build_calvin_counterfactual_pair_plan,
    load_calvin_counterfactual_pair_plan,
)


def _candidates() -> tuple[CalvinCounterfactualCandidate, ...]:
    identities = (
        "movable/blue",
        "movable/pink",
        "movable/red",
        "part/button",
        "part/slider",
        "part/switch",
    )
    partitions = {"train": (0, 1_000), "validation": (1, 2_000), "heldout": (2, 3_000)}
    rows = []
    for identity_index, identity in enumerate(identities):
        for partition, (segment, base) in partitions.items():
            for offset in range(6):
                rows.append(
                    CalvinCounterfactualCandidate(
                        global_index=base + identity_index * 30 + offset * 3,
                        segment_index=segment,
                        source_partition=partition,
                        scene="calvin_scene_D",
                        identity_key=identity,
                        static_visible_pixels=128 + offset,
                        gripper_visible_pixels=32,
                        task_key=f"task_{partition}_{offset}",
                        instruction=f"instruction {partition} {offset}",
                    )
                )
    return tuple(rows)


def _config() -> CalvinCounterfactualPlanConfig:
    return CalvinCounterfactualPlanConfig(
        train_pairs_per_identity=2,
        validation_pairs_per_train_identity=1,
        heldout_pairs_per_identity=2,
        heldout_identities_per_family=1,
        minimum_total_visible_pixels=100,
        minimum_same_identity_frame_gap=2,
        seed=17,
    )


def _plan(candidates: tuple[CalvinCounterfactualCandidate, ...] | None = None) -> dict[str, object]:
    return build_calvin_counterfactual_pair_plan(
        candidates or _candidates(),
        config=_config(),
        dataset_id="dataset",
        dataset_revision="revision",
        split_name="training",
        source_sidecar_manifest_sha256="a" * 64,
        foundation_m2_recipe_sha256="b" * 64,
        source_segments={"train": (0,), "validation": (1,), "heldout": (2,)},
    )


def test_pair_plan_is_task_independent_and_identity_disjoint() -> None:
    candidates = _candidates()
    changed_text = tuple(
        replace(
            candidate,
            task_key=f"changed_{index}",
            instruction=f"unrelated prompt {index}",
        )
        for index, candidate in enumerate(candidates)
    )

    first = _plan(candidates)
    changed = _plan(changed_text)
    first_keys = [
        (row["partition"], row["global_index"], row["target_identity_key"])
        for row in first["requests"]
    ]
    changed_keys = [
        (row["partition"], row["global_index"], row["target_identity_key"])
        for row in changed["requests"]
    ]

    assert first_keys == changed_keys
    identity = first["identity_partition"]
    assert not set(identity["train_and_validation"]) & set(identity["heldout_only"])
    assert len(identity["heldout_only"]) == 2
    assert first["audit"]["request_count_by_partition"] == {
        "train": 8,
        "validation": 4,
        "heldout": 4,
    }
    assert first["selection_contract"]["task_text_used_for_selection"] is False


def test_pair_plan_round_trip_is_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan(), indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_calvin_counterfactual_pair_plan(path, expected_sha256=digest)

    assert len(loaded.requests_for("train")) == 8
    assert len(loaded.requests_for("validation")) == 4
    assert len(loaded.requests_for("heldout")) == 4
    assert loaded.file_sha256 == digest
    assert len(set(request.global_index for request in loaded.requests)) == 16

    with pytest.raises(ContractError, match="hash differs"):
        load_calvin_counterfactual_pair_plan(path, expected_sha256="f" * 64)


def test_pair_plan_rejects_overlapping_source_frame_partitions() -> None:
    candidates = list(_candidates())
    candidates.append(
        replace(
            candidates[0],
            source_partition="validation",
            segment_index=1,
            identity_key="part/new",
        )
    )

    with pytest.raises(ContractError, match="partitions overlap"):
        _plan(tuple(candidates))


def test_pair_plan_rejects_insufficient_visible_support() -> None:
    candidates = tuple(
        replace(candidate, static_visible_pixels=1, gripper_visible_pixels=0)
        for candidate in _candidates()
    )

    with pytest.raises(ContractError, match="visibility support"):
        _plan(candidates)
