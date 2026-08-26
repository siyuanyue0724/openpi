from __future__ import annotations

import ast
import json
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from tools.lingbot_vla2_ltop_stage_runtime import (
    INTERNAL_G2_REPORT,
    LingBotVLA2LTOPStageRuntime,
    _assert_no_meta_state,
    _assert_rank_digest_match,
    _strict_model_only_dcp_load,
    _validate_g2_report,
    _validate_model_only_checkpoint_tree,
)
from tools.run_lingbot_vla2_ltop_g2_core import (
    G2_ARCHITECTURE,
    G2_CAPACITY,
    G2_REPRESENTATION_SCHEMA,
    G2_TASK_QUERY_COUNT,
    G2_WORLD_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/verify_lingbot_vla2_ltop_g2_stage_transfer.py"
HELPER = ROOT / "tools/lingbot_vla2_ltop_stage_runtime.py"


def _checkpoint_tree(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    (model / "__0_0.distcp").write_bytes(b"rank-0")
    (model / "__1_0.distcp").write_bytes(b"rank-1")
    external = tmp_path / "report.json"
    return checkpoint, external


def _report(checkpoint: Path, *, model_identity: dict[str, object]) -> dict[str, object]:
    return {
        "schema": G2_REPRESENTATION_SCHEMA,
        "status": "PASS",
        "failures": [],
        "architecture_identity": G2_ARCHITECTURE,
        "model_identity": model_identity,
        "patch_sha256": "4" * 64,
        "training_scope": "representation",
        "world_size": G2_WORLD_SIZE,
        "steps": 128,
        "capacity": G2_CAPACITY,
        "task_query_count": G2_TASK_QUERY_COUNT,
        "trainable_scope": {"representation_scope": {"schema": "scope-v1"}},
        "checkpoint": {
            "requested": True,
            "path": str(checkpoint.absolute()),
            "format": "lingbot-fsdp2-dcp-model-only",
            "optimizer_saved": False,
            "extra_state_saved": False,
            "stage_transfer_not_exact_resume": True,
            "publication_status": "PASS",
        },
        "rank_reports": [
            {"rank": 0, "model_local_state_sha256": "a" * 64},
            {"rank": 1, "model_local_state_sha256": "b" * 64},
        ],
    }


def _publish_report(
    checkpoint: Path,
    external: Path,
    report: dict[str, object],
) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    (checkpoint / INTERNAL_G2_REPORT).write_bytes(payload)
    external.write_bytes(payload)


def test_stage_restore_checkpoint_tree_is_strictly_model_only(tmp_path: Path) -> None:
    checkpoint, _external = _checkpoint_tree(tmp_path)
    (checkpoint / INTERNAL_G2_REPORT).write_text("{}\n")

    inventory = _validate_model_only_checkpoint_tree(checkpoint)

    assert inventory["root_entries"] == [INTERNAL_G2_REPORT, "model"]
    assert inventory["regular_file_count"] == 4

    (checkpoint / "optimizer").mkdir()
    with pytest.raises(ValueError, match="non-model payloads"):
        _validate_model_only_checkpoint_tree(checkpoint)


def test_stage_restore_report_binds_both_rank_local_digests(tmp_path: Path) -> None:
    checkpoint, external = _checkpoint_tree(tmp_path)
    model_identity = {
        "checkpoint_id": "lingbot",
        "checkpoint_revision": "1" * 40,
        "native_source_commit": "2" * 40,
        "patched_source_sha256": {"model.py": "3" * 64},
    }
    report = _report(checkpoint, model_identity=model_identity)
    _publish_report(checkpoint, external, report)

    loaded, digests, report_sha256 = _validate_g2_report(
        stage_checkpoint=checkpoint,
        external_report=external,
        expected_model_identity=model_identity,
        expected_patch_sha256="4" * 64,
    )

    assert loaded == report
    assert digests == {0: "a" * 64, 1: "b" * 64}
    assert len(report_sha256) == 64

    external.write_text("{}\n")
    with pytest.raises(ValueError, match="reports differ"):
        _validate_g2_report(
            stage_checkpoint=checkpoint,
            external_report=external,
            expected_model_identity=model_identity,
            expected_patch_sha256="4" * 64,
        )


def test_stage_restore_report_rejects_missing_rank_digest(tmp_path: Path) -> None:
    checkpoint, external = _checkpoint_tree(tmp_path)
    model_identity = {
        "checkpoint_id": "lingbot",
        "checkpoint_revision": "1" * 40,
        "native_source_commit": "2" * 40,
        "patched_source_sha256": {"model.py": "3" * 64},
    }
    report = _report(checkpoint, model_identity=model_identity)
    rank_reports = report["rank_reports"]
    assert isinstance(rank_reports, list)
    assert isinstance(rank_reports[1], dict)
    rank_reports[1]["model_local_state_sha256"] = None
    _publish_report(checkpoint, external, report)

    with pytest.raises(ValueError, match="rank 1 model digest"):
        _validate_g2_report(
            stage_checkpoint=checkpoint,
            external_report=external,
            expected_model_identity=model_identity,
            expected_patch_sha256="4" * 64,
        )


def test_stage_restore_no_meta_check_covers_parameters_and_buffers() -> None:
    model = torch.nn.Linear(2, 2)
    _assert_no_meta_state(model, phase="test")

    meta_model = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(RuntimeError, match="meta state"):
        _assert_no_meta_state(meta_model, phase="test")


def test_stage_runtime_digest_check_fails_closed_before_handoff() -> None:
    _assert_rank_digest_match(actual="a" * 64, expected="a" * 64, rank=0)

    with pytest.raises(RuntimeError, match="rank 1 restored model digest differs"):
        _assert_rank_digest_match(actual="a" * 64, expected="b" * 64, rank=1)


def test_stage_runtime_source_has_one_strict_model_only_dcp_load() -> None:
    source = HELPER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    load_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "checkpointer"
    ]

    assert len(load_calls) == 1
    load_call = load_calls[0]
    allow_partial = next(
        keyword.value for keyword in load_call.keywords if keyword.arg == "allow_partial_load"
    )
    assert isinstance(allow_partial, ast.Constant)
    assert allow_partial.value is False
    assert 'state = {"model": policy}' in source
    assert 'set(state) != {"model"}' in source
    assert "build_lingbot_official_optimizer" not in source
    assert "build_lingbot_representation_optimizer" not in source
    assert '"optimizer": optimizer' not in source
    assert '"extra_state":' not in source


def test_stage_runtime_executes_strict_model_only_load(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object], bool]] = []

    class Checkpointer:
        def load(
            self,
            path: str,
            state: dict[str, object],
            *,
            allow_partial_load: bool,
        ) -> None:
            calls.append((path, dict(state), allow_partial_load))

    monkeypatch.setattr(
        "tools.lingbot_vla2_ltop_stage_runtime._distributed_rank_local_call",
        lambda *, action, **_kwargs: action(),
    )
    policy = object()

    _strict_model_only_dcp_load(
        checkpointer=Checkpointer(),
        stage_checkpoint=Path("/checkpoint"),
        policy=policy,
        rank=0,
        dist_module=object(),
    )

    assert calls == [("/checkpoint", {"model": policy}, False)]


def test_stage_runtime_exposes_live_g3_handoff_contract() -> None:
    names = {field.name for field in fields(LingBotVLA2LTOPStageRuntime)}

    assert {
        "contract",
        "policy",
        "graph",
        "graph_config",
        "model_config",
        "training_config",
        "resolved_training_config",
        "representation_scope",
        "optimizer_contract",
        "runtime_modules",
        "rank",
        "local_rank",
        "device",
        "fsdp2_storage_before_load",
        "fsdp2_storage_after_load",
        "expected_model_local_state_sha256",
        "actual_model_local_state_sha256",
    } <= names


def test_stage_runtime_reuses_the_exact_g2b_fsdp2_construction_contract() -> None:
    source = HELPER.read_text(encoding="utf-8")
    for fragment in (
        "LingBotNativeGraphConfig.from_policy(",
        "configure_native_representation_parameter_scope(policy)",
        "build_parallelize_model(",
        "enable_full_shard=True",
        "dp_shard_size=G2_WORLD_SIZE",
        'dp_mode="fsdp2"',
        "enable_gradient_checkpointing=True",
        'init_device="cuda"',
        "fsdp_llm_blocks=False",
        "vlm_fsdp=True",
        "register_native_fsdp_forward_methods(policy)",
        "_validate_fsdp2_parameter_storage(",
        "_model_local_state_digest(policy, torch)",
        'phase="ltop-g2-stage-restore-model-digest-match"',
        "yield LingBotVLA2LTOPStageRuntime(",
        "if dist.is_initialized():",
        "dist.destroy_process_group()",
    ):
        assert fragment in source


def test_stage_restore_tool_delegates_without_copying_bootstrap_or_load() -> None:
    source = TOOL.read_text(encoding="utf-8")

    assert "prepare_lingbot_vla2_ltop_stage_transfer(_stage_request(args))" in source
    assert "with open_lingbot_vla2_ltop_stage_runtime(contract) as runtime:" in source
    for forbidden in (
        "build_parallelize_model(",
        "configure_native_representation_parameter_scope(",
        "checkpointer.load(",
        "load_model_weights(",
    ):
        assert forbidden not in source
