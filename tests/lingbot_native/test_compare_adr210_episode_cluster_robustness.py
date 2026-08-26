from __future__ import annotations

from tests.lingbot_native.test_compare_adr210_causal_warm_action_gate import _snapshots
from tools.compare_adr210_causal_warm_action_gate import compare_gate
from tools.compare_adr210_episode_cluster_robustness import compare_cluster_robustness


def _formal(warm: dict, cold: dict, lbot: dict) -> dict:
    return compare_gate(
        warm=warm,
        cold=cold,
        lbot=lbot,
        cold_path=None,
        bootstrap_replicates=100,
        minimum_relative_reduction=0.02,
    )


def test_episode_cluster_robustness_authorizes_consistent_gain() -> None:
    warm, cold, lbot = _snapshots()
    report = compare_cluster_robustness(
        warm=warm,
        cold=cold,
        lbot=lbot,
        formal_gate=_formal(warm, cold, lbot),
        bootstrap_replicates=100,
        minimum_relative_reduction=0.02,
    )
    assert report["decision"] == "ROBUST_AUTHORIZE_30K"
    assert report["partitions"]["validation"]["warm_minus_cold"]["cluster_count"] == 3


def test_episode_cluster_robustness_preserves_formal_no_go() -> None:
    warm, cold, lbot = _snapshots()
    for sample in warm["samples"]:
        if sample["partition"] == "validation":
            sample["action_loss"] = 1.2
    report = compare_cluster_robustness(
        warm=warm,
        cold=cold,
        lbot=lbot,
        formal_gate=_formal(warm, cold, lbot),
        bootstrap_replicates=100,
        minimum_relative_reduction=0.02,
    )
    assert report["decision"] == "FORMAL_NO_GO"
