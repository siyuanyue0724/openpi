from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from tools.smoke_lingbot_vla2_unified_full_weight import (
    _parse_args,
    _validate_unified_dimensions,
    _write_text_durable,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/smoke_lingbot_vla2_unified_full_weight.py"


def _source() -> str:
    return TOOL.read_text()


def test_unified_full_weight_smoke_delays_accelerator_and_host_imports() -> None:
    tree = ast.parse(_source())
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_from_imports = {
        (node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    forbidden = {"lingbotvla", "numpy", "torch", "torchvision", "transformers"}
    assert forbidden.isdisjoint(top_imports | top_from_imports)


def test_unified_full_weight_smoke_defaults_follow_persistent_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "lingbot-vla-v2-unified"
    monkeypatch.setenv("PICF_LINGBOT_SOURCE", str(source))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--checkpoint-dir",
            "checkpoint",
            "--processor-dir",
            "processor",
            "--image",
            "image.png",
            "--output",
            "g0.json",
        ],
    )
    args = _parse_args()
    assert args.source_checkout == source
    assert args.config == source / "configs/vla/robotwin/robotwin.yaml"
    assert args.robot_config == source / "configs/robot_configs/robotwin.yaml"


def test_unified_full_weight_smoke_requires_the_complete_source_patch_pair() -> None:
    source = _source()
    assert "verify_unified_patches(root=root, checkout=args.source_checkout)" in source
    assert "for patch in (args.data_patch, args.graph_patch)" in source
    assert 'patch_states != ["applied", "applied"]' in source
    assert "LINGBOT_SOURCE_COMMIT" in source
    assert "validate_checkpoint(args.checkpoint_dir)" in source
    assert "validate_processor(args.processor_dir)" in source
    assert 'patch_report.get("patched_source_sha256")' in source
    assert "actual_source_hashes != expected_source_hashes" in source


def test_unified_full_weight_smoke_has_neutral_and_enabled_host_paths() -> None:
    source = _source()
    assert "official = policy.sample_actions" in source
    assert "install_lingbot_unified_belief_graph(policy, graph)" in source
    assert "neutral = policy.sample_actions" in source
    assert "unified_belief_context=context" in source
    assert 'report["neutral_action_bitwise_equal"]' in source
    assert "torch.equal(neutral, official)" in source


def test_unified_full_weight_smoke_checks_physical_state_and_no_target_leak() -> None:
    source = _source()
    required_fragments = (
        "assert_deploy_payload_is_causal(raw_observation)",
        "posterior.validate()",
        "posterior.serialize()",
        "lifecycle_normalization_max_abs_error",
        "minimum_geometry_information_eigenvalue",
        "native_sensor_tokens",
        "final_pair_shape",
        "target_only_fields_present",
        "cuda_memory_bytes",
    )
    for fragment in required_fragments:
        assert fragment in source


def test_unified_full_weight_smoke_geometry_schema_is_fail_closed() -> None:
    _validate_unified_dimensions(
        capacity=16,
        content_width=256,
        geometry_width=6,
        uncertainty_width=16,
        num_steps=2,
    )
    with pytest.raises(ValueError, match="exactly six"):
        _validate_unified_dimensions(
            capacity=16,
            content_width=256,
            geometry_width=5,
            uncertainty_width=16,
            num_steps=2,
        )


def test_unified_full_weight_smoke_report_publication_is_atomic(tmp_path: Path) -> None:
    report = tmp_path / "nested" / "g0.json"
    _write_text_durable(report, '{"status":"PASS"}\n')
    assert report.read_text() == '{"status":"PASS"}\n'
    assert not tuple(report.parent.glob("*.tmp"))
