from __future__ import annotations

import copy

import pytest

from tools.compare_adr176_matched_action_curves import compare_curves


def _sample(*, ordinal: int, partition: str, action_loss: float) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "partition": partition,
        "rank": ordinal % 2,
        "sample_key": f"sample-{partition}-{ordinal}",
        "segment_index": ordinal,
        "source_digest": f"digest-{partition}-{ordinal}",
        "source_episode_index": ordinal,
        "source_global_index": ordinal,
        "task_key": f"task-{ordinal}",
        "transition_index": ordinal,
        "action_loss": action_loss,
    }


def _snapshot(*, step: int, picf: bool, offset: float) -> dict[str, object]:
    return {
        "checkpoint_global_step": step,
        "picf_graph_installed": picf,
        "stream_plan_sha256": "a" * 64,
        "representation_split_sha256": "b" * 64,
        "evaluation_plan_sha256": "c" * 64,
        "samples": [
            _sample(ordinal=0, partition="validation", action_loss=0.4 + offset),
            _sample(ordinal=1, partition="validation", action_loss=0.5 + offset),
            _sample(ordinal=2, partition="heldout", action_loss=0.6 + offset),
            _sample(ordinal=3, partition="heldout", action_loss=0.7 + offset),
        ],
    }


def test_compare_curves_reports_picf_advantage() -> None:
    report = compare_curves(
        picf_snapshots=[
            _snapshot(step=0, picf=True, offset=0.0),
            _snapshot(step=100, picf=True, offset=-0.2),
        ],
        lbot_snapshots=[
            _snapshot(step=0, picf=False, offset=0.0),
            _snapshot(step=100, picf=False, offset=-0.1),
        ],
        steps=(0, 100),
        bootstrap_replicates=100,
    )

    assert report["decision"] == "PICF_ACTION_ADVANTAGE"
    assert report["partitions"]["heldout"]["endpoint"]["picf_sample_wins"] == 2
    assert report["partitions"]["validation"]["normalized_auc"]["picf_over_lbot"] < 1


def test_compare_curves_rejects_contract_drift() -> None:
    picf = _snapshot(step=0, picf=True, offset=0.0)
    lbot = _snapshot(step=0, picf=False, offset=0.0)
    lbot["evaluation_plan_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="evaluation_plan_sha256"):
        compare_curves(
            picf_snapshots=[picf, _snapshot(step=100, picf=True, offset=-0.1)],
            lbot_snapshots=[lbot, _snapshot(step=100, picf=False, offset=-0.1)],
            steps=(0, 100),
            bootstrap_replicates=10,
        )


def test_compare_curves_rejects_sample_drift() -> None:
    lbot_100 = _snapshot(step=100, picf=False, offset=-0.1)
    lbot_100["samples"][0]["sample_key"] = "changed"

    with pytest.raises(ValueError, match="fixed sample identities"):
        compare_curves(
            picf_snapshots=[
                _snapshot(step=0, picf=True, offset=0.0),
                _snapshot(step=100, picf=True, offset=-0.1),
            ],
            lbot_snapshots=[
                _snapshot(step=0, picf=False, offset=0.0),
                lbot_100,
            ],
            steps=(0, 100),
            bootstrap_replicates=10,
        )


def test_compare_curves_rejects_model_input_drift() -> None:
    picf_100 = _snapshot(step=100, picf=True, offset=-0.1)
    picf_100["samples"][0]["model_inputs_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="fixed sample identities"):
        compare_curves(
            picf_snapshots=[
                _snapshot(step=0, picf=True, offset=0.0),
                picf_100,
            ],
            lbot_snapshots=[
                _snapshot(step=0, picf=False, offset=0.0),
                _snapshot(step=100, picf=False, offset=-0.1),
            ],
            steps=(0, 100),
            bootstrap_replicates=10,
        )


def test_compare_curves_does_not_mutate_snapshots() -> None:
    picf = [_snapshot(step=0, picf=True, offset=0.0), _snapshot(step=100, picf=True, offset=-0.1)]
    lbot = [_snapshot(step=0, picf=False, offset=0.0), _snapshot(step=100, picf=False, offset=-0.1)]
    expected = copy.deepcopy((picf, lbot))

    compare_curves(
        picf_snapshots=picf,
        lbot_snapshots=lbot,
        steps=(0, 100),
        bootstrap_replicates=10,
    )

    assert (picf, lbot) == expected
