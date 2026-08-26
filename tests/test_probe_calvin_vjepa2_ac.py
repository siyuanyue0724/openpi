from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from picf_next.data.calvin import CalvinDatasetIndex, CalvinEpisode
from picf_next.data.dataset_manifest import DatasetFileManifest
from tools.aggregate_vjepa2_ac_shards import aggregate_shard_reports
from tools.probe_calvin_vjepa2_ac import (
    _load_gate_index,
    _select_clip_ends,
    summarize_control_reports,
)


def _report(actual: float, zero: float, reversed_value: float, shuffled: float):
    def metrics(value: float) -> dict[str, float]:
        return {"teacher_forced_l1": value, "autoregressive_l1": value + 0.01}

    return {
        "actual": metrics(actual),
        "zero": metrics(zero),
        "reversed": metrics(reversed_value),
        "shuffled": metrics(shuffled),
    }


def _shard_report(shard_index: int, *, shard_count: int = 4) -> dict[str, object]:
    global_ends = tuple(1000 + index * 100 for index in range(32))
    shard_positions = tuple(range(shard_index, len(global_ends), shard_count))
    shard_ends = global_ends[shard_index::shard_count]
    return {
        "checkpoint": {"bytes": 11_760_743_310, "path": "/mnt/model.pt", "sha256": "a" * 64},
        "dataset": {"dataset_id": "calvin", "dataset_revision": "sha256:data"},
        "camera": "rgb_static",
        "conditioning_semantics": "realized_pose_difference_not_policy_command",
        "future_targets_are_loss_only": True,
        "model_parameters_frozen": True,
        "elapsed_seconds": 12.0 + shard_index,
        "selection": {
            "control_seed": 239,
            "global_clip_count": len(global_ends),
            "global_clip_ends": list(global_ends),
            "shard_global_positions": list(shard_positions),
            "shard_clip_ends": list(shard_ends),
            "shard_count": shard_count,
            "shard_index": shard_index,
        },
        "clips": [
            {
                "control_seed": 239 + position,
                "episode_index": global_ends.index(end),
                "frame_indices": [end - 56, end],
                "global_clip_position": position,
                "controls": _report(0.10, 0.15, 0.14, 0.13),
            }
            for position, end in zip(shard_positions, shard_ends, strict=True)
        ],
    }


def test_vjepa2_ac_gate_selects_distinct_source_episodes(tmp_path: Path) -> None:
    episodes = tuple(
        CalvinEpisode(index=index, start=index * 100, end=index * 100 + 80)
        for index in range(20)
    )
    dataset = CalvinDatasetIndex(
        split_root=tmp_path,
        dataset_id="calvin-vjepa2-ac-selection",
        dataset_revision="sha256:test",
        control_hz=30,
        episodes=episodes,
        segments=(),
    )
    ends = _select_clip_ends(dataset, clip_count=16, seed=239)

    assert len(ends) == len(set(dataset.source_episode(end).index for end in ends)) == 16
    assert all(dataset.source_episode(end).start + 56 <= end for end in ends)


def test_vjepa2_ac_gate_skips_exhaustive_tree_scan(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    sentinel = cast(CalvinDatasetIndex, object())

    def fake_load(split_root: Path, **kwargs: object) -> CalvinDatasetIndex:
        captured["split_root"] = split_root
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(CalvinDatasetIndex, "load", staticmethod(fake_load))
    manifest = cast(
        DatasetFileManifest,
        SimpleNamespace(dataset_id="calvin", dataset_revision="sha256:test"),
    )

    result = _load_gate_index(tmp_path / "split", manifest)

    assert result is sentinel
    assert captured["split_root"] == (tmp_path / "split").resolve()
    assert captured["dataset_id"] == "calvin"
    assert captured["dataset_revision"] == "sha256:test"
    assert captured["dataset_manifest"] is manifest
    assert captured["verify_files"] is False


def test_vjepa2_ac_four_gpu_aggregation_requires_complete_exact_partition() -> None:
    reports = tuple(
        json.loads(json.dumps(_shard_report(index), sort_keys=True)) for index in range(4)
    )
    aggregate = aggregate_shard_reports(reports)

    assert aggregate["global_clip_count"] == aggregate["global_episode_count"] == 32
    assert aggregate["shard_count"] == 4
    assert aggregate["authorizes_arm_c"] is True
    assert aggregate["authorizes_policy_training"] is False

    malformed = [dict(report) for report in reports]
    malformed[3]["selection"] = dict(malformed[3]["selection"])
    malformed[3]["selection"]["shard_index"] = 2
    with pytest.raises(ValueError, match="duplicated"):
        aggregate_shard_reports(malformed)


def test_vjepa2_ac_control_summary_requires_consistent_paired_advantage() -> None:
    summary = summarize_control_reports(
        tuple(_report(0.10 + index * 0.001, 0.15, 0.14, 0.13) for index in range(16))
    )
    assert summary["causal_signal_pass"] is True
    assert summary["clip_count"] == 16
    assert summary["minimum_clip_count"] == 16
    assert summary["comparisons"]["zero"]["teacher_forced_l1"][
        "one_sided_sign_test_pvalue"
    ] == pytest.approx(1.0 / 65536.0)


def test_vjepa2_ac_control_summary_rejects_one_failed_control_family() -> None:
    summary = summarize_control_reports(
        tuple(_report(0.10, 0.15, 0.14, 0.08) for _index in range(16))
    )
    assert summary["causal_signal_pass"] is False


def test_vjepa2_ac_control_summary_rejects_an_underpowered_gate() -> None:
    summary = summarize_control_reports(
        tuple(_report(0.10, 0.15, 0.14, 0.13) for _index in range(15))
    )
    assert summary["causal_signal_pass"] is False


def test_vjepa2_ac_control_summary_fails_closed_on_schema_or_nan() -> None:
    malformed = _report(0.10, 0.15, 0.14, 0.13)
    malformed["actual"] = {"teacher_forced_l1": 0.1}
    with pytest.raises(ValueError, match="metrics"):
        summarize_control_reports((malformed,))

    nonfinite = _report(float("nan"), 0.15, 0.14, 0.13)
    with pytest.raises(ValueError, match="invalid loss"):
        summarize_control_reports((nonfinite,))
