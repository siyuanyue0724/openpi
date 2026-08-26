from __future__ import annotations

import copy

from tools.compare_adr210_causal_warm_action_gate import (
    COLD_SCHEMA,
    LBOT_SCHEMA,
    WARM_SCHEMA,
    compare_gate,
)


def _snapshots() -> tuple[dict, dict, dict]:
    contract = {
        "stream_plan_sha256": "1" * 64,
        "representation_split_sha256": "2" * 64,
        "evaluation_plan_sha256": "3" * 64,
        "lingbot_base_family_sha256": "4" * 64,
    }
    cold_samples = []
    lbot_samples = []
    warm_samples = []
    for ordinal in range(102):
        partition = "validation" if ordinal < 51 else "heldout"
        shared = {
            "sample_key": f"sample-{ordinal:03d}",
            "partition": partition,
            "task_key": f"task-{ordinal % 3}",
            "segment_index": ordinal,
            "source_episode_index": ordinal // 17,
            "source_global_index": 1_000 + ordinal,
            "transition_index": ordinal,
            "source_digest": f"source-{ordinal}",
            "model_inputs_sha256": f"inputs-{ordinal}",
        }
        cold_samples.append(
            {**shared, "action_loss": 1.0, "native_source_rgb_sha256": f"rgb-{ordinal}"}
        )
        lbot_samples.append({**shared, "action_loss": 1.1})
        if ordinal >= 8:
            warm_samples.append(
                {
                    **shared,
                    "action_loss": 0.8,
                    "native_source_rgb_sha256": f"rgb-{ordinal}",
                }
            )
    common = {"status": "PASS", "checkpoint_global_step": 100, **contract}
    warm = {
        **common,
        "schema": WARM_SCHEMA,
        "state_mode": "causal_warm_four_past_frames",
        "picf_graph_installed": True,
        "history_transitions": 4,
        "eligible_sample_count": 94,
        "excluded_samples": [{"ordinal": value} for value in range(8)],
        "samples": warm_samples,
    }
    cold = {
        **common,
        "schema": COLD_SCHEMA,
        "state_mode": "cold_reset",
        "picf_graph_installed": True,
        "samples": cold_samples,
    }
    lbot = {
        **common,
        "schema": LBOT_SCHEMA,
        "picf_graph_installed": False,
        "samples": lbot_samples,
    }
    return warm, cold, lbot


def test_causal_warm_gate_authorizes_two_split_paired_advantage() -> None:
    warm, cold, lbot = _snapshots()
    report = compare_gate(
        warm=warm,
        cold=cold,
        lbot=lbot,
        cold_path=None,
        bootstrap_replicates=100,
        minimum_relative_reduction=0.02,
    )
    assert report["decision"] == "AUTHORIZE_30K"
    assert report["partitions"]["validation"]["warm_minus_cold"][
        "relative_loss_reduction"
    ] == 0.19999999999999996
    assert report["partitions"]["heldout"]["sample_count"] == 51


def test_causal_warm_gate_rejects_a_split_regression() -> None:
    warm, cold, lbot = _snapshots()
    regressed = copy.deepcopy(warm)
    for sample in regressed["samples"]:
        if sample["partition"] == "validation":
            sample["action_loss"] = 1.2
    report = compare_gate(
        warm=regressed,
        cold=cold,
        lbot=lbot,
        cold_path=None,
        bootstrap_replicates=100,
        minimum_relative_reduction=0.02,
    )
    assert report["decision"] == "REJECT_30K"
