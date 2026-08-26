from __future__ import annotations

import copy

import pytest

from picf_next.eval.stationary_replay import (
    STATIONARY_FIXED_REPLAY_METRICS,
    aggregate_replay_measurements,
    compare_stationary_replay_summaries,
    validate_stationary_fixed_replay,
)


def _summary(value: float) -> dict[str, float]:
    return {
        name: (0.5 if "iou" in name or "coverage" in name else value)
        for name in STATIONARY_FIXED_REPLAY_METRICS
    }


def _report() -> dict[str, object]:
    fresh = _summary(1.0)
    candidate = dict(fresh)
    candidate.update(
        {
            "loss_total": 0.9,
            "loss_set": 0.9,
            "loss_dynamics": 0.9,
            "loss_dynamics_survival": 0.9,
            "loss_dynamics_visibility": 0.9,
            "loss_binding": 0.9,
            "assignment_conflicts_per_clip": 0.9,
            "discovery_soft_iou": 0.6,
            "posterior_soft_iou": 0.6,
            "posterior_identity_coverage": 0.6,
        }
    )
    comparison = compare_stationary_replay_summaries(
        fresh_m2=fresh,
        candidate=candidate,
        absolute_tolerance=1e-6,
    )
    checks = {
        f"{split}_{name}": passed
        for split in ("validation", "heldout")
        for name, passed in comparison.items()
    }
    bindings = {
        "audit_code_revision": "1" * 40,
        "candidate_code_revision": "2" * 40,
        **{
            name: character * 64
            for name, character in (
                ("candidate_checkpoint_sha256", "a"),
                ("candidate_report_sha256", "b"),
                ("dataset_manifest_sha256", "c"),
                ("feature_cache_manifest_sha256", "d"),
                ("foundation_recipe_sha256", "e"),
                ("m2_checkpoint_sha256", "f"),
                ("m2_report_sha256", "3"),
                ("physical_sidecar_manifest_sha256", "4"),
                ("source_coverage_recipe_sha256", "5"),
                ("stage_recipe_sha256", "6"),
            )
        },
    }
    measurements = []
    prefixes = (0, 8, 32, 128)
    for split_index, split in enumerate(("validation", "heldout")):
        for optimizer_step, prefix in enumerate(prefixes):
            for rank in range(2):
                start = 1000 * split_index + 200 * optimizer_step + rank * 20
                clip = {
                    "optimizer_step": optimizer_step,
                    "source_range_index": 0,
                    "start_global_index": start,
                    "prefix_length": prefix,
                    "train_length": 2,
                    "train_start_global_index": start + prefix,
                    "stop_global_index": start + prefix + 2,
                }
                for model, metrics in (("fresh_m2", fresh), ("candidate", candidate)):
                    measurements.append(
                        {
                            "clip": clip,
                            "metrics": metrics,
                            "model": model,
                            "optimizer_step": optimizer_step,
                            "rank": rank,
                            "split": split,
                        }
                    )
    return {
        "schema": "picf-next.stationary-fixed-checkpoint-replay.v2",
        "status": "PASS",
        "protocol": {
            "comparison": "same-frozen-clips-fresh-m2-vs-stage-b-candidate.v1",
            "observation_inputs": "task-independent-cached-native-token-bank",
            "target_use": "post-forward-loss-and-evaluation-only",
            "split_names": ["validation", "heldout"],
            "prefix_lengths": [0, 8, 32, 128],
            "train_length": 2,
            "world_size": 2,
            "optimizer_steps_per_split": 4,
            "seed": 20260720,
        },
        "bindings": bindings,
        "plans": {
            split: {"plan_sha256": character * 64, "source_ranges": [[0, 200]]}
            for split, character in (("validation", "b"), ("heldout", "c"))
        },
        "thresholds": {
            "absolute_tolerance": 1e-6,
            "lower_is_better": [
                "loss_total",
                "loss_set",
                "loss_dynamics",
                "loss_dynamics_survival",
                "loss_dynamics_visibility",
                "loss_binding",
                "assignment_conflicts_per_clip",
            ],
            "higher_is_better": [
                "discovery_soft_iou",
                "posterior_soft_iou",
                "posterior_identity_coverage",
            ],
        },
        "splits": {
            split: {
                "clip_count": 8,
                "models": {"fresh_m2": fresh, "candidate": candidate},
                "comparisons": comparison,
            }
            for split in ("validation", "heldout")
        },
        "checks": checks,
        "failed_checks": [],
        "measurements": measurements,
        "long_training_authorized": False,
    }


def test_stationary_fixed_replay_recomputes_decision() -> None:
    report = _report()
    assert validate_stationary_fixed_replay(report)["status"] == "PASS"

    changed = copy.deepcopy(report)
    changed["splits"]["heldout"]["models"]["candidate"]["loss_total"] = 2.0
    with pytest.raises(ValueError, match="comparisons were not recomputed"):
        validate_stationary_fixed_replay(changed)


def test_stationary_replay_aggregation_has_no_hidden_weighting() -> None:
    first = _summary(1.0)
    second = _summary(3.0)
    aggregate = aggregate_replay_measurements((first, second))
    assert aggregate["loss_total"] == 2.0
    assert aggregate["discovery_soft_iou"] == 0.5


def test_stationary_replay_rejects_impossible_proper_scores() -> None:
    report = _report()
    report["measurements"][0]["metrics"]["loss_dynamics_survival"] = -1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_stationary_fixed_replay(report)
