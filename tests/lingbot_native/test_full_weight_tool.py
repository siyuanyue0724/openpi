from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.bootstrap_lingbot_vla2_native import CHECKOUT_RELATIVE_PATH, PATCHED_SOURCES
from tools.smoke_lingbot_vla2_native_full_weight import (
    _parse_args,
    _physical_relation_prompt_error,
    _relation_is_finite,
    _validate_dimensions,
    _validated_patched_source_hashes,
    _write_text_durable,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/smoke_lingbot_vla2_native_full_weight.py"


def _source() -> str:
    return TOOL.read_text()


def test_native_full_weight_tool_delays_every_accelerator_import() -> None:
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


def test_native_full_weight_tool_has_no_historical_unified_path() -> None:
    source = _source()
    assert "picf_next.unified" not in source
    assert "unified_belief" not in source
    assert "action_layer_adapter" not in source
    assert "semantic_scorer" not in source
    assert "lifecycle" not in source


def test_native_full_weight_defaults_follow_persistent_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PICF_LINGBOT_NATIVE_SOURCE", raising=False)
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
    expected = ROOT / CHECKOUT_RELATIVE_PATH
    assert args.source_checkout == expected
    assert args.config == expected / "configs/vla/robotwin/robotwin.yaml"
    assert args.robot_config == expected / "configs/robot_configs/robotwin.yaml"
    assert args.architecture_identity == "content_addressed_task_match_v1"


def test_native_full_weight_configs_follow_an_overridden_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--source-checkout",
            str(source),
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


def test_native_full_weight_tool_verifies_patch_weights_and_neutral_parity() -> None:
    source = _source()
    required = (
        "verify_native_patch(",
        "detect_native_patch_state(args.source_checkout, args.patch)",
        "_validated_patched_source_hashes(args.source_checkout, patch_report)",
        "validate_checkpoint(args.checkpoint_dir)",
        "validate_processor(args.processor_dir)",
        "select_lingbot_deterministic_moe_backend(",
        "official = policy.sample_actions",
        "official_repeat = policy.sample_actions",
        "strip_targetless_alignment_teacher_heads(policy)",
        "targetless = policy.sample_actions",
        "install_lingbot_native_graph(policy, graph)",
        "neutral = policy.sample_actions",
        'report["targetless_action_bitwise_equal"]',
        'report["targetless_route_bitwise_equal"]',
        'report["neutral_action_bitwise_equal"]',
        'report["neutral_route_bitwise_equal"]',
        'report["official_repeat_action_bitwise_equal"]',
        'report["official_repeat_route_bitwise_equal"]',
    )
    for fragment in required:
        assert fragment in source


def test_native_full_weight_patch_contract_covers_model_and_parallel_sources(
    tmp_path: Path,
) -> None:
    expected = {}
    for index, relative in enumerate(PATCHED_SOURCES):
        payload = f"source-{index}".encode()
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected[str(relative)] = hashlib.sha256(payload).hexdigest()

    report: dict[str, object] = {"patched_source_sha256": expected}
    assert _validated_patched_source_hashes(tmp_path, report) == expected
    incomplete: dict[str, object] = {
        "patched_source_sha256": {str(PATCHED_SOURCES[0]): expected[str(PATCHED_SOURCES[0])]},
    }
    with pytest.raises(RuntimeError, match="wrong source hash contract"):
        _validated_patched_source_hashes(tmp_path, incomplete)

    (tmp_path / PATCHED_SOURCES[-1]).write_text("tampered")
    with pytest.raises(RuntimeError, match="differs from immutable patch replay"):
        _validated_patched_source_hashes(tmp_path, report)


def test_native_full_weight_tool_runs_atomic_state_and_prompt_gates() -> None:
    source = _source()
    required = (
        "LingBotNativePolicyRuntime",
        "ExecutedControlBatch.reset_only",
        "continuation_observation",
        "sessions.serialize()",
        "NativeSessionManager.deserialize",
        "session_snapshot_roundtrip_exact",
        "prompt_invariant_physical_posterior_bitwise_equal",
        "native_relations_finite",
        "target_only_fields_present",
        "cuda_memory_bytes",
    )
    for fragment in required:
        assert fragment in source


def test_native_full_weight_tool_supports_the_task_independent_relation_abi() -> None:
    first = SimpleNamespace(
        support_logits=torch.tensor([1.0]),
        visible_support=torch.tensor([0.5]),
        ownership=torch.tensor([0.75]),
        existence=torch.tensor([0.8]),
        existence_logits=torch.tensor([1.4]),
    )
    second = SimpleNamespace(
        support_logits=first.support_logits.clone(),
        visible_support=first.visible_support.clone(),
        ownership=first.ownership.clone(),
        existence=first.existence.clone(),
        existence_logits=first.existence_logits.clone(),
    )

    assert _relation_is_finite(first)
    assert _physical_relation_prompt_error(first, second) == 0.0
    second.ownership = torch.tensor([0.5])
    assert _physical_relation_prompt_error(first, second) == pytest.approx(0.25)
    source = _source()
    assert "successor = graph.task_independent" in source
    assert "task_scorer_surface_absent" in source
    assert "prompt_invariant_physical_relation_bitwise_equal" in source


def test_native_full_weight_dimensions_and_report_are_fail_closed(tmp_path: Path) -> None:
    _validate_dimensions(capacity=16, maximum_control_tokens=8, num_steps=2)
    with pytest.raises(ValueError, match="positive integers"):
        _validate_dimensions(capacity=0, maximum_control_tokens=8, num_steps=2)
    output = tmp_path / "nested" / "native-g0.json"
    _write_text_durable(output, '{"status":"PASS"}\n')
    assert output.read_text() == '{"status":"PASS"}\n'
    assert not tuple(output.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        _write_text_durable(output, '{"status":"REPLACED"}\n')
    assert output.read_text() == '{"status":"PASS"}\n'

    external = tmp_path / "external.json"
    external.write_text("original\n")
    link = tmp_path / "smoke-link.json"
    link.symlink_to(external)
    with pytest.raises(FileExistsError):
        _write_text_durable(link, "replacement\n")
    assert external.read_text() == "original\n"
