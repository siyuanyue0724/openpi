from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.lattice_feasibility import (
    LATTICE_FEASIBILITY_SCHEMA,
    LATTICE_TASK_KEYS,
    LATTICE_VISUAL_GRID_CACHE_FIELDS,
    LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS,
    LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS,
    configure_native_processor_area_budget,
    configure_native_processor_lattice,
    fractional_token_metrics,
    lattice_feasibility_decision,
    native_lattice_shortest_edge,
    native_processor_area_budget_contract,
    native_processor_expected_grid,
    require_native_visual_grid_cache_populated,
    reset_native_visual_grid_cache,
    select_lattice_segment_indices,
    validate_lattice_feasibility_report,
    validate_native_processor_grid_budget,
    validate_native_processor_record_grid,
)
from tools.probe_lingbot_lattice_feasibility import _paired_seed

_TOOL = Path(__file__).resolve().parents[2] / "tools/probe_lingbot_lattice_feasibility.py"


def test_fractional_metrics_penalize_mixed_tokens_and_rank_target_mass() -> None:
    coarse = fractional_token_metrics([2.0, 0.0], [0.5, 0.0])
    fine = fractional_token_metrics([2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    hidden = fractional_token_metrics([1.0, 0.0], [0.0, 0.0])

    assert coarse["eligible"] is True
    assert coarse["purity"] == pytest.approx(0.5)
    assert fine["purity"] == pytest.approx(1.0)
    assert fine["self_soft_iou"] == pytest.approx(1.0)
    assert fine["fractional_weighted_auc"] > coarse["fractional_weighted_auc"]
    assert hidden["eligible"] is False
    assert hidden["fractional_weighted_auc"] is None


def _sample(
    *,
    lattice: int,
    index: int,
    auc_improves: bool,
    action_delta: float,
) -> dict:
    token_count = 2 * lattice * lattice
    target_mass = [0.0] * token_count
    if lattice == 8:
        target_mass[0] = 0.5
        target_mass[1] = 0.25
    else:
        target_mass[0] = 1.0
        target_mass[1] = 0.5
    scores = [0.0] * token_count
    if auc_improves:
        scores[0] = 2.0
        scores[1] = 1.0
    else:
        scores[-1] = 2.0
    metrics = fractional_token_metrics(scores, target_mass)
    baseline_action = 0.4 + index * 1e-4
    return {
        "sample_key": f"sample-{index}",
        "task_key": f"task-{index}",
        "task": f"instruction {index}",
        "source_global_index": index,
        "segment_index": index,
        "transition_index": 0,
        "augmentation_seed": 1000 + index,
        "flow_seed": 2000 + index,
        "target_identity_keys": [f"object-{index}"],
        "image_sha256": {"static": "a" * 64, "gripper": "b" * 64},
        "input_shapes": {
            "actions": [1, 50, 55],
            "images": [1, 3, (lattice * 2) ** 2, 1536],
            "image_grid_thw": [1, 3, 3],
            "lang_tokens": [1, 72],
        },
        "image_grid_thw": [[1, lattice * 2, lattice * 2]] * 3,
        "image_valid": [True, True, False],
        "dense_task_logits": scores,
        "target_mass": target_mass,
        "eligible": metrics["eligible"],
        "metrics": metrics,
        "official_action_loss": baseline_action + action_delta,
        "forward_seconds": 1.0,
        "peak_cuda_allocated_bytes": 12 * 1024**3,
        "peak_cuda_reserved_bytes": 13 * 1024**3,
    }


def _report() -> dict:
    baseline = [
        _sample(lattice=8, index=index, auc_improves=False, action_delta=0.0) for index in range(12)
    ]
    candidate = [
        _sample(lattice=12, index=index, auc_improves=True, action_delta=0.0) for index in range(12)
    ]
    value = {
        "schema": LATTICE_FEASIBILITY_SCHEMA,
        "baseline_lattice": 8,
        "candidate_lattice": 12,
        "loss_only_supervision": True,
        "target_resolution_happened_after_forward": True,
        "optimizer_created": False,
        "checkpoint_mutated": False,
        "same_parameter_objects_across_arms": True,
        "target_or_mask_fields_in_model_inputs": [],
        "arms": {
            "8": {
                "lattice": 8,
                "processor": {
                    "lattice": 8,
                    "patch_size": 16,
                    "merge_size": 2,
                    "pixels_per_edge": 256,
                    "shortest_edge_area": 256**2,
                    "longest_edge_area": 1024**2,
                },
                "visual_grid_cache_invalidation": {
                    "precompute_grid_thw": True,
                    "fields": list(LATTICE_VISUAL_GRID_CACHE_FIELDS),
                    "none_by_design": list(LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS),
                    "nonempty_before": [],
                    "all_none_after": True,
                    "populated_after_arm": list(LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS),
                },
                "samples": baseline,
            },
            "12": {
                "lattice": 12,
                "processor": {
                    "lattice": 12,
                    "patch_size": 16,
                    "merge_size": 2,
                    "pixels_per_edge": 384,
                    "shortest_edge_area": 384**2,
                    "longest_edge_area": 1024**2,
                },
                "visual_grid_cache_invalidation": {
                    "precompute_grid_thw": True,
                    "fields": list(LATTICE_VISUAL_GRID_CACHE_FIELDS),
                    "none_by_design": list(LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS),
                    "nonempty_before": list(LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS),
                    "all_none_after": True,
                    "populated_after_arm": list(LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS),
                },
                "samples": candidate,
            },
        },
    }
    decision = lattice_feasibility_decision(value)
    value.update(decision)
    value["failures"] = sorted(name for name, passed in value["gates"].items() if not passed)
    value["status"] = "PASS" if not value["failures"] else "FAIL"
    return value


def test_lattice_report_recomputes_paired_gates_and_accepts_strong_candidate() -> None:
    report = _report()
    assert validate_lattice_feasibility_report(report)["status"] == "PASS"
    assert report["aggregates"]["eligible_sample_count"] == 12
    assert report["aggregates"]["median_purity_ratio"] >= 1.15
    assert report["aggregates"]["mean_fractional_weighted_auc_delta"] >= 0.02


def test_lattice_report_rejects_action_regression_and_persisted_tampering() -> None:
    report = _report()
    for sample in report["arms"]["12"]["samples"]:
        sample["official_action_loss"] += 0.02
    report.update(lattice_feasibility_decision(report))
    report["failures"] = sorted(name for name, passed in report["gates"].items() if not passed)
    report["status"] = "FAIL"
    validated = validate_lattice_feasibility_report(report)
    assert validated["gates"]["released_action_path_preserved"] is False

    tampered = copy.deepcopy(report)
    tampered["aggregates"]["mean_action_loss_delta"] = -1.0
    with pytest.raises(ContractError, match="persisted aggregates"):
        validate_lattice_feasibility_report(tampered)


def test_lattice_report_rejects_incomplete_official_grid_cache_cycle() -> None:
    report = _report()
    report["arms"]["12"]["visual_grid_cache_invalidation"]["populated_after_arm"].pop()
    report.update(lattice_feasibility_decision(report))
    report["failures"] = sorted(name for name, passed in report["gates"].items() if not passed)
    report["status"] = "FAIL"

    validated = validate_lattice_feasibility_report(report)
    assert validated["gates"]["native_grid_exact"] is False


def test_lattice_report_serializes_no_eligible_targets_without_infinity() -> None:
    report = _report()
    for arm in report["arms"].values():
        for sample in arm["samples"]:
            sample["target_mass"] = [0.0] * len(sample["target_mass"])
            sample["metrics"] = fractional_token_metrics(
                sample["dense_task_logits"],
                sample["target_mass"],
            )
            sample["eligible"] = False
    report.update(lattice_feasibility_decision(report))
    report["failures"] = sorted(name for name, passed in report["gates"].items() if not passed)
    report["status"] = "FAIL"

    assert validate_lattice_feasibility_report(report)["status"] == "FAIL"
    assert report["aggregates"]["mean_fractional_weighted_auc_delta"] is None


def test_fractional_metrics_reject_nonfinite_or_invalid_mass() -> None:
    with pytest.raises(ContractError, match="finite"):
        fractional_token_metrics([math.nan], [0.5])
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        fractional_token_metrics([0.0], [1.1])


def test_native_lattice_edge_uses_official_patch_and_merge_geometry() -> None:
    assert native_lattice_shortest_edge(8) == 256**2
    assert native_lattice_shortest_edge(12) == 384**2
    with pytest.raises(ValueError, match="lattice"):
        native_lattice_shortest_edge(0)


def test_task_bank_selection_is_source_only_deterministic_and_complete() -> None:
    segments = [
        SimpleNamespace(task_key=task, index=100 * task_index + candidate)
        for task_index, task in enumerate(LATTICE_TASK_KEYS)
        for candidate in range(3)
    ]
    selected = select_lattice_segment_indices(segments)
    assert len(selected) == len(LATTICE_TASK_KEYS)
    assert selected == select_lattice_segment_indices(tuple(reversed(segments)))
    for task_index, index in enumerate(selected):
        assert index in range(100 * task_index, 100 * task_index + 3)

    with pytest.raises(ContractError, match="absent"):
        select_lattice_segment_indices(segments[:-3])


def test_probe_processor_contract_and_paired_seed_are_exact() -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            size={"shortest_edge": 256**2, "longest_edge": 4096**2},
            patch_size=16,
            merge_size=2,
        )
    )
    contract = configure_native_processor_lattice(processor, 12)
    assert contract["pixels_per_edge"] == 384
    assert processor.image_processor.size["shortest_edge"] == 384**2
    assert _paired_seed(7, "sample", "flow") == _paired_seed(7, "sample", "flow")
    assert _paired_seed(7, "sample", "flow") != _paired_seed(
        7,
        "sample",
        "augmentation",
    )


def test_public_processor_uses_official_aspect_preserving_area_budget() -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            size={"shortest_edge": 256**2, "longest_edge": 4096**2},
            patch_size=16,
            merge_size=2,
        )
    )

    contract = configure_native_processor_area_budget(processor, 8)

    assert processor.image_processor.size == {
        "shortest_edge": 256**2,
        "longest_edge": 256**2,
    }
    assert contract == {
        "lattice": 8,
        "mode": "official_qwen_aspect_ratio_preserving_area_budget",
        "patch_size": 16,
        "merge_size": 2,
        "target_image_area": 256**2,
        "maximum_raw_patch_tokens": 256,
        "maximum_merged_visual_tokens": 64,
    }


def test_public_processor_area_budget_rejects_invalid_processor_contracts() -> None:
    with pytest.raises(RuntimeError, match="dynamic-resolution mapping"):
        configure_native_processor_area_budget(SimpleNamespace(), 8)

    for image_processor, message in (
        (SimpleNamespace(size=[], patch_size=16, merge_size=2), "mapping"),
        (
            SimpleNamespace(
                size={"shortest_edge": 0, "longest_edge": 256**2},
                patch_size=16,
                merge_size=2,
            ),
            "area is invalid",
        ),
        (
            SimpleNamespace(
                size={"shortest_edge": 256**2, "longest_edge": 4096**2},
                patch_size=14,
                merge_size=2,
            ),
            "geometry",
        ),
    ):
        with pytest.raises(RuntimeError, match=message):
            configure_native_processor_area_budget(
                SimpleNamespace(image_processor=image_processor),
                8,
            )
    with pytest.raises(ValueError, match="lattice"):
        native_processor_area_budget_contract(True)


@pytest.mark.parametrize("grid", ([[1, 16, 16]], [[1, 12, 18]]))
def test_public_processor_accepts_square_and_nonsquare_grids_within_budget(
    grid: list[list[int]],
) -> None:
    report = validate_native_processor_grid_budget(grid, lattice=8)

    assert report["image_grid_thw"] == grid
    assert report["merged_visual_tokens"] <= 64
    assert report["maximum_merged_visual_tokens"] == 64


@pytest.mark.parametrize(
    ("grid", "message"),
    (
        ([[1, 18, 18]], "exceeds"),
        ([[1, 13, 18]], "divisible"),
        ([[2, 8, 8]], "temporal"),
        ([[1, 8, 8], [1, 8, 8]], "exactly one"),
    ),
)
def test_public_processor_grid_budget_fails_closed(
    grid: list[list[int]],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_native_processor_grid_budget(grid, lattice=8)


def test_public_processor_grid_is_bound_to_official_smart_resize_geometry() -> None:
    expected = native_processor_expected_grid(
        image_height=480,
        image_width=640,
        lattice=8,
    )
    assert expected == [[1, 12, 18]]
    report = validate_native_processor_record_grid(
        expected,
        image_height=480,
        image_width=640,
        lattice=8,
    )
    assert report["merged_visual_tokens"] == 54

    with pytest.raises(RuntimeError, match="smart-resize geometry"):
        validate_native_processor_record_grid(
            [[1, 16, 16]],
            image_height=480,
            image_width=640,
            lattice=8,
        )
    with pytest.raises(ValueError, match="aspect ratio"):
        native_processor_expected_grid(
            image_height=1,
            image_width=201,
            lattice=8,
        )


def test_probe_invalidates_only_official_nonparameter_grid_cache() -> None:
    host = SimpleNamespace(
        config=SimpleNamespace(precompute_grid_thw=True),
        **{
            name: (None if name in LATTICE_VISUAL_GRID_NONE_BY_DESIGN_FIELDS else object())
            for name in LATTICE_VISUAL_GRID_CACHE_FIELDS
        },
    )
    parameter = object()
    host.parameter = parameter
    report = reset_native_visual_grid_cache(host)

    assert report["nonempty_before"] == list(LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS)
    assert all(getattr(host, name) is None for name in LATTICE_VISUAL_GRID_CACHE_FIELDS)
    assert host.parameter is parameter

    with pytest.raises(RuntimeError, match="return contract"):
        require_native_visual_grid_cache_populated(host)
    for name in LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS:
        setattr(host, name, object())
    assert require_native_visual_grid_cache_populated(host) == list(
        LATTICE_VISUAL_GRID_POPULATED_CACHE_FIELDS
    )

    host.config.precompute_grid_thw = False
    with pytest.raises(RuntimeError, match="precompute"):
        reset_native_visual_grid_cache(host)


def test_probe_resolves_loss_only_targets_after_forward_without_optimizer() -> None:
    source = _TOOL.read_text()
    forward = source.index("result = run_native_policy_diagnostic_forward(")
    target = source.index("target_bundle = build_native_calvin_sequence_target_bundle(")
    normalization = source.index("validate_lingbot_calvin_norm_stats(norm_stats_payload)")
    model_load = source.index("with init_empty_weights(), no_init_weights():")
    assert normalization < model_load
    assert forward < target
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "parameter._version" in source
    assert "optimizer_created" in source


def test_probe_direct_script_help_does_not_depend_on_working_directory(
    tmp_path: Path,
) -> None:
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
    assert "native 8x8 and 12x12" in result.stdout
