from __future__ import annotations

import ast
import copy
import math
import sys
from pathlib import Path

import pytest

from tools import run_lingbot_vla2_ltop_g1t as g1t
from tools.run_lingbot_vla2_native_g0 import _implementation_paths

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/run_lingbot_vla2_ltop_g1t.py"


def _gradient_stats(square: float, *, graph: bool = False) -> dict[str, object]:
    return {
        "parameter_elements": 64 if graph else 128,
        "gradient_elements": 0 if square == 0 else 128,
        "gradient_tensors": 0 if square == 0 else 2,
        "square": square,
        "finite": True,
    }


def _gradient_comparisons() -> dict[str, dict[str, object]]:
    baseline = {}
    accumulated = {}
    blocked = {}
    for family, _fragments in g1t.G1T_GRADIENT_FAMILIES:
        graph = family == g1t.G1T_GRAPH_FAMILY
        baseline[family] = _gradient_stats(0.0 if graph else 4.0, graph=graph)
        accumulated[family] = _gradient_stats(0.0 if graph else 16.0, graph=graph)
        blocked[family] = _gradient_stats(0.0 if graph else 4.0, graph=graph)
    comparisons = g1t._build_gradient_comparisons(baseline, accumulated, blocked)
    comparisons[g1t.G1T_GRAPH_FAMILY].update(
        {
            "action_only_blocked_square": 0.0,
            "action_only_blocked_norm": 0.0,
            "action_only_gradient_elements": 0,
            "action_only_finite": True,
        }
    )
    return comparisons


def _loss(value: float = 0.5) -> dict[str, float]:
    return {
        "baseline": value,
        "blocked": value,
        "absolute_error": 0.0,
        "relative_error": 0.0,
    }


def _rank_report(rank: int) -> dict[str, object]:
    routes = {"sha256": str(rank + 1) * 64, "calls": 4, "tokens": 32, "layers": 4}
    metrics = {
        "scalars": {"router_z_loss": 0.01},
        "structured_sha256": {},
        "summary_sha256": str(rank + 3) * 64,
    }
    return {
        "rank": rank,
        "device_name": "NVIDIA A100-PCIE-40GB",
        "sample_keys": [f"sample-{rank}"],
        "episode_keys": [f"episode-{rank}"],
        "episode_ids": [rank + 1],
        "frame_indices": [rank],
        "source_digest": f"source-{rank}",
        "model_input_sha256": f"input-{rank}",
        "model_input_tensors": {"noise": f"noise-{rank}", "time": f"time-{rank}"},
        "noise_sha256": f"noise-{rank}",
        "time_sha256": f"time-{rank}",
        "fixed_rng_sha256": f"rng-{rank}",
        "rng_restoration_equal": True,
        "action_loss": _loss(0.5),
        "total_loss": _loss(0.53),
        "moe_auxiliary": {
            "official_moe_regularizer": _loss(0.03),
            "sequence_wise_moe_loss": _loss(0.02),
            "router_z_loss": _loss(0.01),
            "released_metrics": metrics,
            "blocked_metrics": metrics,
            "metrics_equal": True,
        },
        "velocity": {
            "released": {"sha256": "a" * 64},
            "blocked": {"sha256": "a" * 64},
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
        },
        "released_routes": routes,
        "blocked_routes": routes,
        "routes_equal": True,
        "blocked_repeat_equal": True,
        "blocked_repeat_velocity_max_abs_error": 0.0,
        "blocked_cache": {
            "context_finalized": True,
            "native_width": 100,
            "expanded_width": 140,
            "inserted_rows": 40,
            "all_inserted_action_cache_edges_blocked": True,
            "expanded_action_cache_visible_sha256": "b" * 64,
            "native_valid_sha256": "c" * 64,
            "finite": True,
        },
        "timings": {"released_forward_backward_s": 1.0},
        "cuda_memory_bytes": {"peak_reserved": 1},
        "peak_reserved_gib": 20.0,
    }


def _report() -> dict[str, object]:
    comparisons = _gradient_comparisons()
    return {
        "schema": g1t.G1T_SCHEMA,
        "status": "PASS",
        "failures": [],
        "source_commit": "source",
        "patch_sha256": "patch",
        "patched_source_sha256": {"model.py": "hash"},
        "source_diff_sha256": "diff",
        "checkpoint_revision": "checkpoint",
        "processor_revision": "processor",
        "implementation_sha256": "implementation",
        "architecture_identity": g1t.G1T_ARCHITECTURE,
        "world_size": g1t.G1T_WORLD_SIZE,
        "seed": 20260812,
        "capacity": g1t.G1T_PHYSICAL_CAPACITY,
        "task_query_count": g1t.G1T_TASK_QUERY_COUNT,
        "training_contract": g1t.G1T_TRAINING_CONTRACT,
        "parallel_contract": g1t.G1T_PARALLEL_CONTRACT,
        "fsdp2_placement": "selective-embedding-offload",
        "cuda_allocator": "expandable-segments",
        "maximum_peak_reserved_gib": 39.0,
        "thresholds": dict(g1t.G1T_DEFAULT_THRESHOLDS),
        "dataset_contract": {"status": "PASS"},
        "config_sha256": "config",
        "gradient_families": {
            name: list(fragments) for name, fragments in g1t.G1T_GRADIENT_FAMILIES
        },
        "gradient_comparisons": comparisons,
        "parameter_manifest": {
            "families": {},
            "fsdp2_storage": {"placement": "selective-embedding-offload"},
        },
        "alignment_teacher_prune": {"removed": []},
        "moe_backend": {"name": "eager"},
        "rank_reports": [_rank_report(0), _rank_report(1)],
    }


def test_g1t_delays_torch_and_upstream_imports() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_from_imports = {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert {"numpy", "torch", "lingbotvla", "transformers"}.isdisjoint(
        top_imports | top_from_imports
    )


def test_g1t_parser_freezes_two_gpu_ltop_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--output", str(tmp_path / "g1t.json")])
    args = g1t._parse_args()

    assert args.capacity == 16
    assert args.task_query_count == 4
    assert args.fsdp2_placement == "selective-embedding-offload"
    assert args.cuda_allocator == "expandable-segments"
    assert g1t._thresholds_from_args(args) == g1t.G1T_DEFAULT_THRESHOLDS


def test_g1t_gradient_polarization_recovers_exact_cosine_and_residual() -> None:
    same = g1t._gradient_comparison_from_squares(
        baseline_square=4.0,
        accumulated_square=16.0,
        blocked_square=4.0,
    )
    assert same["dot"] == 4.0
    assert same["cosine"] == 1.0
    assert same["norm_relative_error"] == 0.0
    assert same["residual_relative_norm"] == 0.0

    orthogonal = g1t._gradient_comparison_from_squares(
        baseline_square=4.0,
        accumulated_square=8.0,
        blocked_square=4.0,
    )
    assert orthogonal["dot"] == 0.0
    assert orthogonal["cosine"] == 0.0
    assert orthogonal["residual_relative_norm"] == pytest.approx(math.sqrt(2.0))


def test_g1t_parameter_families_are_exclusive_and_cover_action_and_graph() -> None:
    assert g1t._family_for_parameter("model.action_out_proj.weight") == "action_output"
    assert (
        g1t._family_for_parameter("model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.gate.weight")
        == "action_expert"
    )
    assert (
        g1t._family_for_parameter("model.qwenvl_with_expert.qwenvl.model.layers.0.self_attn.q_proj.weight")
        == "vision_language"
    )
    assert (
        g1t._family_for_parameter("model.qwenvl_with_expert.picf_native_graph.task_query_embeddings")
        == "picf_graph"
    )
    assert g1t._family_for_parameter("unrelated.weight") is None


def test_g1t_report_accepts_strict_action_objective_gradient_and_graph_parity() -> None:
    report = _report()
    assert g1t.validate_ltop_g1t_report(report) == report


def test_g1t_report_flags_moe_objective_mismatch_even_when_action_matches() -> None:
    report = _report()
    comparison = report["rank_reports"][0]["moe_auxiliary"][  # type: ignore[index]
        "sequence_wise_moe_loss"
    ]
    comparison["blocked"] = 0.04
    comparison["absolute_error"] = 0.02
    comparison["relative_error"] = 1.0
    report["failures"] = ["rank 0: sequence_wise_moe_loss parity failed"]
    report["status"] = "FAIL"

    assert g1t.validate_ltop_g1t_report(report)["status"] == "FAIL"


def test_g1t_report_rejects_nonzero_picf_graph_action_gradient() -> None:
    report = _report()
    graph = report["gradient_comparisons"][g1t.G1T_GRAPH_FAMILY]  # type: ignore[index]
    graph["action_only_blocked_square"] = 0.25
    graph["action_only_blocked_norm"] = 0.5
    report["failures"] = ["PICF graph BLOCKED action-only gradient is non-zero"]
    report["status"] = "FAIL"

    assert g1t.validate_ltop_g1t_report(report)["status"] == "FAIL"


def test_g1t_report_rejects_forged_gradient_cosine() -> None:
    report = _report()
    tampered = copy.deepcopy(report)
    tampered["gradient_comparisons"]["action_output"]["cosine"] = 0.5  # type: ignore[index]
    with pytest.raises(ValueError, match="forged cosine"):
        g1t.validate_ltop_g1t_report(tampered)


def test_g1t_source_uses_serial_backward_without_retaining_graphs() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert source.count("backward_loss.backward()") == 1
    assert "retain_graph=True" not in source
    assert 'name="released"' in source
    assert 'name="blocked-accumulated"' in source
    assert 'name="blocked-isolated"' in source
    assert 'name="blocked-action-only"' in source
    assert "policy.zero_grad(set_to_none=True)" in source
    assert "_release_graph(torch, device)" in source


def test_g1t_uses_keyword_only_fsdp2_storage_contract() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "expected_placement=args.fsdp2_placement" in source


def test_g1t_uses_production_prior_stepper_with_deterministic_episode_ids() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "LingBotNativePriorStepper" in source
    assert "episode_ids = _episode_ids(" in source
    assert "episode_ids=episode_ids" in source
    assert "run_native_v3_prior_chain" not in source

    torch = pytest.importorskip("torch")
    first = g1t._episode_ids(
        ("episode-a", "episode-b"),
        torch_module=torch,
        device="cpu",
    )
    second = g1t._episode_ids(
        ("episode-a", "episode-b"),
        torch_module=torch,
        device="cpu",
    )
    assert first.tolist() == second.tolist()
    assert first.tolist()[0] != first.tolist()[1]


def test_route_trace_recomputation_is_detached_from_training_autograd() -> None:
    source = (ROOT / "tools/lingbot_vla2_runtime_helpers.py").read_text(encoding="utf-8")
    route_trace = source[source.index("class _RouteTrace") :]
    assert "with torch.no_grad(), torch.amp.autocast" in route_trace


def test_g1t_implementation_digest_closure_includes_entrypoint_and_native_abi() -> None:
    paths = {
        str(path.relative_to(ROOT))
        for path in _implementation_paths(ROOT, entrypoint=TOOL)
    }
    assert "tools/run_lingbot_vla2_ltop_g1t.py" in paths
    assert "tools/run_lingbot_vla2_native_g0.py" in paths
    assert "src/picf_next/lingbot_native/host.py" in paths
    assert "references/patches/lingbot_vla2_picf_native.patch" in paths
