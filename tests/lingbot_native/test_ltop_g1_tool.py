from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_lingbot_vla2_ltop_g1 as g1
from tools.run_lingbot_vla2_native_g0 import _implementation_paths

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/run_lingbot_vla2_ltop_g1.py"


def _rank_report(rank: int) -> dict[str, object]:
    action_hash = str(rank + 1) * 64
    cache_metadata_hash = str(rank + 3) * 64
    routes = [{"layer": 0, "expert": rank}]
    return {
        "rank": rank,
        "device_name": "NVIDIA A100-PCIE-40GB",
        "sample_keys": [f"sample-{rank}"],
        "episode_keys": [f"episode-{rank}"],
        "frame_indices": [rank],
        "source_digest": f"source-{rank}",
        "model_input_sha256": f"input-{rank}",
        "model_input_tensors": {"state": f"state-{rank}"},
        "flow_noise_sha256": f"noise-{rank}",
        "episode_ids": [rank + 1],
        "address_receipt": {"physical_rows": [0, 1, 2, 3]},
        "prior_trace_sha256": f"prior-{rank}",
        "baseline_action_sha256": action_hash,
        "baseline_repeat_action_sha256": action_hash,
        "blocked_action_sha256": action_hash,
        "blocked_repeat_action_sha256": action_hash,
        "neutral_action_sha256": action_hash,
        "neutral_repeat_action_sha256": action_hash,
        "blocked_cache_metadata_sha256": cache_metadata_hash,
        "neutral_cache_metadata_sha256": cache_metadata_hash,
        "baseline_repeat_bitwise_equal": True,
        "blocked_repeat_bitwise_equal": True,
        "blocked_vs_baseline_bitwise_equal": True,
        "neutral_repeat_bitwise_equal": True,
        "blocked_vs_neutral_bitwise_equal": True,
        "baseline_repeat_max_abs_error": 0.0,
        "blocked_repeat_max_abs_error": 0.0,
        "blocked_vs_baseline_max_abs_error": 0.0,
        "blocked_vs_baseline_mean_abs_error": 0.0,
        "neutral_repeat_max_abs_error": 0.0,
        "blocked_vs_neutral_max_abs_error": 0.0,
        "blocked_vs_neutral_mean_abs_error": 0.0,
        "actions_finite": True,
        "baseline_routes": routes,
        "baseline_repeat_routes": routes,
        "blocked_routes": routes,
        "blocked_repeat_routes": routes,
        "neutral_routes": routes,
        "neutral_repeat_routes": routes,
        "baseline_repeat_routes_equal": True,
        "blocked_repeat_routes_equal": True,
        "blocked_vs_baseline_routes_equal": True,
        "neutral_repeat_routes_equal": True,
        "blocked_vs_neutral_routes_equal": True,
        "blocked_neutral_cache_metadata_equal": True,
        "object_read_action_cache_edge_blocked": True,
        "all_inserted_action_cache_edges_blocked": True,
        "context_finalized": True,
        "timings": {"baseline_action_s": 1.0},
        "cuda_memory_bytes": {"peak_reserved": 1},
    }


def _report() -> dict[str, object]:
    return {
        "schema": g1.G1_SCHEMA,
        "status": "PASS",
        "failures": [],
        "source_commit": "source-commit",
        "patch_sha256": "patch",
        "patched_source_sha256": {"model.py": "patched"},
        "source_diff_sha256": "diff",
        "checkpoint_revision": "checkpoint",
        "processor_revision": "processor",
        "implementation_sha256": "implementation",
        "architecture_identity": g1.G1_ARCHITECTURE,
        "world_size": g1.G1_WORLD_SIZE,
        "seed": 20260812,
        "capacity": g1.G1_PHYSICAL_CAPACITY,
        "task_query_count": g1.G1_TASK_QUERY_COUNT,
        "num_steps": g1.G1_DENOISE_STEPS,
        "inference_contract": g1.G1_INFERENCE_CONTRACT,
        "parallel_contract": g1.G1_PARALLEL_CONTRACT,
        "dataset_contract": {"schema": "dataset"},
        "config_sha256": "config",
        "parameter_manifest": {"active_trainable_numel": 0},
        "alignment_teacher_prune": {"removed": []},
        "moe_inference_backend": {"name": "eager"},
        "rank_reports": [_rank_report(0), _rank_report(1)],
    }


def test_ltop_g1_delays_accelerator_and_upstream_imports() -> None:
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


def test_ltop_g1_parser_preserves_g0_physical_and_relation_capacities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--output", str(tmp_path / "g1.json")])
    args = g1._parse_args()

    assert args.capacity == 16
    assert args.task_query_count == 4
    assert args.num_steps == g1.G1_DENOISE_STEPS


def test_ltop_g1_restores_released_cached_eager_inference_after_config_init() -> None:
    config = SimpleNamespace(
        use_cache=False,
        use_compile=True,
        attention_implementation="flex_cached",
        vit_attn_implementation="flash_attention_2",
    )

    g1.apply_ltop_g1_inference_contract(config)

    assert config.use_cache is True
    assert config.use_compile is False
    assert config.attention_implementation == "eager"
    assert config.vit_attn_implementation == "eager"


def test_ltop_g1_report_validator_accepts_only_exact_pass_evidence() -> None:
    report = _report()
    assert g1.validate_ltop_g1_report(report) == report

    missing = copy.deepcopy(report)
    del missing["rank_reports"][0]["context_finalized"]  # type: ignore[index]
    with pytest.raises(ValueError, match="frozen schema"):
        g1.validate_ltop_g1_report(missing)

    reused = copy.deepcopy(report)
    reused["rank_reports"][1]["sample_keys"] = ["sample-0"]  # type: ignore[index]
    with pytest.raises(ValueError, match="reused one CALVIN sample"):
        g1.validate_ltop_g1_report(reused)

    cache_drift = copy.deepcopy(report)
    cache_drift["inference_contract"]["use_cache"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="released eager inference contract"):
        g1.validate_ltop_g1_report(cache_drift)

    topology_drift = copy.deepcopy(report)
    topology_drift["parallel_contract"]["ep_size"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="proven LingBot parallel contract"):
        g1.validate_ltop_g1_report(topology_drift)


def test_ltop_g1_report_validator_recomputes_failures() -> None:
    report = _report()
    report["rank_reports"][1]["blocked_action_sha256"] = "9" * 64  # type: ignore[index]
    report["rank_reports"][1]["blocked_repeat_action_sha256"] = "9" * 64  # type: ignore[index]
    report["rank_reports"][1]["blocked_vs_baseline_bitwise_equal"] = False  # type: ignore[index]
    report["rank_reports"][1]["blocked_vs_neutral_bitwise_equal"] = False  # type: ignore[index]
    report["rank_reports"][1]["blocked_vs_baseline_max_abs_error"] = 1.0  # type: ignore[index]
    report["failures"] = ["rank 1: blocked_vs_baseline_bitwise_equal is false"]
    report["status"] = "FAIL"
    report["failures"] = [
        "rank 1: blocked_vs_baseline_bitwise_equal is false",
        "rank 1: blocked_vs_neutral_bitwise_equal is false",
        "rank 1: blocked_vs_baseline_max_abs_error=1.0",
    ]
    assert g1.validate_ltop_g1_report(report)["status"] == "FAIL"

    tampered = copy.deepcopy(report)
    tampered["status"] = "PASS"
    with pytest.raises(ValueError, match="status differs"):
        g1.validate_ltop_g1_report(tampered)


def test_ltop_g1_report_validator_rejects_self_inconsistent_evidence() -> None:
    report = _report()
    report["rank_reports"][0]["blocked_cache_metadata_sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="blocked_neutral_cache_metadata_equal"):
        g1.validate_ltop_g1_report(report)

    report = _report()
    report["rank_reports"][0]["blocked_routes"] = [{"layer": 1, "expert": 0}]  # type: ignore[index]
    with pytest.raises(ValueError, match="blocked_repeat_routes_equal"):
        g1.validate_ltop_g1_report(report)

    report = _report()
    report["num_steps"] = g1.G1_DENOISE_STEPS - 1
    with pytest.raises(ValueError, match="released denoise schedule"):
        g1.validate_ltop_g1_report(report)


def test_ltop_g1_implementation_digest_closure_includes_its_entrypoint() -> None:
    paths = {
        str(path.relative_to(ROOT))
        for path in _implementation_paths(ROOT, entrypoint=TOOL)
    }
    assert "tools/run_lingbot_vla2_ltop_g1.py" in paths
    assert "tools/run_lingbot_vla2_native_g0.py" in paths
    assert "src/picf_next/lingbot_native/host.py" in paths
    assert "references/patches/lingbot_vla2_picf_native.patch" in paths


def test_ltop_g1_uses_typed_native_role_for_language_span() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "native_roles == int(NativeRole.LANGUAGE)" in source
    assert "native_roles == 1" not in source
