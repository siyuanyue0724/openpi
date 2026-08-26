from __future__ import annotations

import ast
import hashlib
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.run_lingbot_vla2_unified_g1 import (
    G1_EXTRA_STATE_SCHEMA,
    G1_REPORT_SCHEMA,
    _capture_rank_rng_state,
    _checkpoint_boundary_digests,
    _g1_execution_contract_digest,
    _model_family_digest,
    _model_local_state_digest,
    _optimizer_local_state_digest,
    _parse_args,
    _rank_rng_state_digest,
    _restore_rank_rng_state,
    _validate_adamw_state,
    _validate_fsdp2_parameter_storage,
    _validate_paths_and_args,
    _validate_rank_rng_state,
    _validate_resume_extra_state,
    _write_text_durable,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_lingbot_vla2_unified_g1.py"


def _source() -> str:
    return TOOL.read_text()


def _resume_payload(*, global_step: int = 1) -> dict[str, object]:
    rng_state = _capture_rank_rng_state(torch, np, device=None)
    rng_state["torch_cuda"] = b"synthetic-cuda-state"
    snapshot = b"snapshot"
    return {
        "global_step": global_step,
        "model_local_state_sha256": "c" * 64,
        "model_family_digest": "family",
        "next_optimizer_step": global_step,
        "optimizer_local_moment_elements": 14,
        "optimizer_local_state_sha256": "d" * 64,
        "optimizer_state_entries": 2,
        "picf_published_optimizer_step": global_step - 1,
        "picf_session_snapshot": snapshot,
        "picf_session_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "plan_sha256": "a" * 64,
        "rank": 0,
        "rank_rng_state": rng_state,
        "rank_rng_state_sha256": _rank_rng_state_digest(rng_state, require_cuda=True),
        "schema": G1_EXTRA_STATE_SCHEMA,
        "source_digest": "b" * 64,
        "world_size": 2,
    }


def test_g1_tool_delays_accelerator_and_host_imports() -> None:
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
    forbidden = {"lingbotvla", "numpy", "torch", "transformers", "yaml"}
    assert forbidden.isdisjoint(top_imports | top_from_imports)
    assert G1_REPORT_SCHEMA.endswith(".v2")


def test_g1_defaults_follow_the_persistent_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "lingbot-vla-v2-unified"
    monkeypatch.setenv("PICF_LINGBOT_SOURCE", str(source))
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--phase", "fresh"])
    args = _parse_args()
    assert args.source_checkout == source
    assert args.training_config == source / "configs/vla/robotwin/robotwin.yaml"


def test_g1_small_report_publication_is_atomic(tmp_path: Path) -> None:
    report = tmp_path / "nested" / "report.json"
    _write_text_durable(report, '{"status":"PASS"}\n')
    assert report.read_text() == '{"status":"PASS"}\n'
    assert not tuple(report.parent.glob("*.tmp"))


def test_g1_tool_uses_one_official_fsdp_checkpoint_transaction() -> None:
    source = _source()
    required = (
        'dist.init_process_group(backend="nccl")',
        'dp_mode="fsdp2"',
        "install_lingbot_unified_belief_graph(policy, graph)",
        "compute_alignment_losses=False",
        "policy = build_parallelize_model(",
        "enable_full_shard=True",
        "enable_fsdp_offload=True",
        ".to(torch.float32)",
        "_validate_fsdp2_parameter_storage(policy, torch)",
        "run_lingbot_unified_optimizer_attempt(",
        'build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")',
        '"picf_session_snapshot": picf_session_snapshot',
        '"rank_rng_state": rank_rng_state',
        "_checkpoint_boundary_digests(",
        '"resume_checkpoint_boundary_verified": loaded_checkpoint_boundary is not None',
        "_restore_rank_rng_state(resume_rng_state, torch, np, device=device)",
        "LingBotUnifiedLaneSession.from_snapshot(",
        "checkpointer.save(",
        "os.replace(staging_checkpoint, output_checkpoint)",
        "actual_source_hashes != expected_source_hashes",
        "validate_dataset_files(",
        "verify_hashes=True",
        'dataset_contract_report.get("status") != "PASS"',
        "execution_contract_sha256=execution_contract_sha256",
    )
    for fragment in required:
        assert fragment in source
    assert "config.align_params = {}" not in source
    assert ".to(torch.bfloat16)" not in source
    assert "foreach=True" not in source
    assert "foreach=False" in source
    assert "error_if_nonfinite=True" in source
    assert source.index("install_lingbot_unified_belief_graph(policy, graph)") < source.index(
        "policy = build_parallelize_model("
    )


def test_g1_model_family_digest_binds_execution_contract_and_plan() -> None:
    first = _model_family_digest(
        graph_contract_digest="graph-a",
        plan_sha256="1" * 64,
        execution_contract_sha256="2" * 64,
    )
    replay = _model_family_digest(
        graph_contract_digest="graph-a",
        plan_sha256="1" * 64,
        execution_contract_sha256="2" * 64,
    )
    changed = _model_family_digest(
        graph_contract_digest="graph-b",
        plan_sha256="1" * 64,
        execution_contract_sha256="2" * 64,
    )
    changed_execution = _model_family_digest(
        graph_contract_digest="graph-a",
        plan_sha256="1" * 64,
        execution_contract_sha256="3" * 64,
    )
    assert first == replay
    assert first != changed
    assert first != changed_execution
    assert len(first) == 64


def test_g1_fsdp2_storage_contract_requires_fp32_cpu_dtensor_masters() -> None:
    class _Local:
        device = type("Device", (), {"type": "cpu"})()

        def numel(self) -> int:
            return 7

    class _Parameter:
        dtype = "float32"

        @staticmethod
        def to_local() -> _Local:
            return _Local()

    class _Model:
        @staticmethod
        def named_parameters():
            return iter((("weight", _Parameter()),))

    torch_stub = type("Torch", (), {"float32": "float32"})()
    assert _validate_fsdp2_parameter_storage(_Model(), torch_stub) == {
        "parameter_tensors": 1,
        "local_elements": 7,
        "master_dtype": "float32",
        "local_device": "cpu",
    }

    _Parameter.dtype = "bfloat16"
    with pytest.raises(RuntimeError, match="master parameter"):
        _validate_fsdp2_parameter_storage(_Model(), torch_stub)


def test_g1_execution_contract_binds_configs_code_and_optimizer(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source_root = ROOT
    for relative in (
        "src/picf_next/data/calvin.py",
        "src/picf_next/data/calvin_normalization.py",
        "src/picf_next/data/dataset_manifest.py",
        "src/picf_next/data/lingbot_calvin.py",
        "src/picf_next/hosts/lingbot_calvin_training.py",
        "src/picf_next/hosts/lingbot_unified.py",
        "src/picf_next/hosts/lingbot_unified_training.py",
        "src/picf_next/training/control.py",
        "tools/bootstrap_lingbot_vla2.py",
        "tools/run_lingbot_vla2_unified_g1.py",
        "tools/smoke_lingbot_vla2_full_weight.py",
        "tools/verify_lingbot_vla2_patch.py",
        "tools/verify_lingbot_vla2_unified_patch.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    unified = root / "src/picf_next/unified"
    unified.mkdir(parents=True, exist_ok=True)
    for source in (source_root / "src/picf_next/unified").glob("*.py"):
        (unified / source.name).write_bytes(source.read_bytes())

    inputs = {}
    for name in (
        "data_patch",
        "graph_patch",
        "training_config",
        "robot_config",
        "data_config",
        "dataset_manifest",
        "norm_stats",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"{name}\n")
        inputs[name] = path
    args = Namespace(
        **inputs,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        seed=7,
    )
    first, implementation = _g1_execution_contract_digest(
        root=root,
        args=args,
        patched_source_sha256={"host.py": "a" * 64},
    )
    replay, replay_implementation = _g1_execution_contract_digest(
        root=root,
        args=args,
        patched_source_sha256={"host.py": "a" * 64},
    )
    assert first == replay
    assert implementation == replay_implementation

    args.learning_rate = 2e-5
    changed_optimizer, _ = _g1_execution_contract_digest(
        root=root,
        args=args,
        patched_source_sha256={"host.py": "a" * 64},
    )
    assert changed_optimizer != first
    args.learning_rate = 1e-5
    inputs["norm_stats"].write_text("changed\n")
    changed_data, _ = _g1_execution_contract_digest(
        root=root,
        args=args,
        patched_source_sha256={"host.py": "a" * 64},
    )
    assert changed_data != first


def test_g1_resume_extra_state_is_complete_and_step_coherent() -> None:
    payload = _resume_payload()
    assert (
        _validate_resume_extra_state(
            payload,
            expected_global_step=1,
            expected_model_family_digest="family",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )
        is payload
    )
    broken = dict(payload)
    broken["picf_published_optimizer_step"] = 1
    with pytest.raises(ValueError, match="publication step"):
        _validate_resume_extra_state(
            broken,
            expected_global_step=1,
            expected_model_family_digest="family",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )


def test_g1_resume_rejects_incomplete_or_cross_contract_state() -> None:
    payload = _resume_payload(global_step=2)
    incomplete = dict(payload)
    incomplete.pop("source_digest")
    with pytest.raises(ValueError, match="incomplete"):
        _validate_resume_extra_state(
            incomplete,
            expected_global_step=2,
            expected_model_family_digest="family",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )
    wrong_schema = dict(payload)
    wrong_schema["schema"] = "obsolete"
    with pytest.raises(ValueError, match="schema"):
        _validate_resume_extra_state(
            wrong_schema,
            expected_global_step=2,
            expected_model_family_digest="family",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )
    with pytest.raises(ValueError, match="model-family"):
        _validate_resume_extra_state(
            payload,
            expected_global_step=2,
            expected_model_family_digest="different",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )

    wrong_source = dict(payload)
    wrong_source["source_digest"] = "c" * 64
    with pytest.raises(ValueError, match="frozen prior batch"):
        _validate_resume_extra_state(
            wrong_source,
            expected_global_step=2,
            expected_model_family_digest="family",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )

    wrong_type = dict(payload)
    wrong_type["global_step"] = True
    with pytest.raises(ValueError, match="not an integer"):
        _validate_resume_extra_state(
            wrong_type,
            expected_global_step=2,
            expected_model_family_digest="family",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )

    wrong_rank = dict(payload)
    wrong_rank["rank"] = 1
    with pytest.raises(ValueError, match="rank topology"):
        _validate_resume_extra_state(
            wrong_rank,
            expected_global_step=2,
            expected_model_family_digest="family",
            expected_plan_sha256="a" * 64,
            expected_rank=0,
            expected_source_digest="b" * 64,
            expected_world_size=2,
        )


def test_g1_adamw_state_requires_exact_fp32_cpu_continuation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW((parameter,), lr=1e-3)
    parameter.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert _validate_adamw_state(optimizer, torch, expected_step=1) == {
        "optimizer_state_entries": 1,
        "optimizer_local_moment_elements": 4,
    }
    optimizer.state[parameter]["step"].fill_(2)
    with pytest.raises(RuntimeError, match="checkpoint boundary"):
        _validate_adamw_state(optimizer, torch, expected_step=1)


def test_g1_checkpoint_boundary_digests_bind_exact_local_bytes() -> None:
    torch.manual_seed(5)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    rng_state = _capture_rank_rng_state(torch, np, device=None)

    model_digest = _model_local_state_digest(model, torch)
    optimizer_digest = _optimizer_local_state_digest(optimizer, model, torch)
    boundary = _checkpoint_boundary_digests(
        model=model,
        optimizer=optimizer,
        picf_session_snapshot=b"picf-snapshot",
        rank_rng_state=rng_state,
        torch_module=torch,
        require_cuda_rng=False,
    )
    assert boundary["model_local_state_sha256"] == model_digest
    assert boundary["optimizer_local_state_sha256"] == optimizer_digest
    assert all(len(value) == 64 for value in boundary.values())

    with torch.no_grad():
        model.weight[0, 0].add_(1)
    assert _model_local_state_digest(model, torch) != model_digest
    assert _optimizer_local_state_digest(optimizer, model, torch) == optimizer_digest

    first_parameter = next(iter(model.parameters()))
    optimizer.state[first_parameter]["exp_avg"].view(-1)[0].add_(1)
    assert _optimizer_local_state_digest(optimizer, model, torch) != optimizer_digest


def test_g1_rank_rng_state_replays_python_numpy_and_torch() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = _capture_rank_rng_state(torch, np, device=None)
    digest = _rank_rng_state_digest(state, require_cuda=False)
    expected = (random.random(), np.random.random(), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    _restore_rank_rng_state(state, torch, np, device=None)
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
    assert len(digest) == 64
    assert _rank_rng_state_digest(state, require_cuda=False) == digest

    broken = dict(state)
    broken["torch_cuda"] = "not-bytes"
    with pytest.raises(ValueError, match="CUDA"):
        _validate_rank_rng_state(broken, require_cuda=False)


def test_g1_resume_accepts_only_the_frozen_second_step(tmp_path: Path) -> None:
    source_checkout = tmp_path / "source"
    checkpoint_dir = tmp_path / "checkpoint"
    processor_dir = tmp_path / "processor"
    dataset_split = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    for directory in (source_checkout, checkpoint_dir, processor_dir, dataset_split):
        directory.mkdir()
    files = [
        tmp_path / name
        for name in (
            "data.patch",
            "graph.patch",
            "train.yaml",
            "robot.yaml",
            "data.json",
            "manifest.json",
            "norm.json",
        )
    ]
    for file in files:
        file.write_text("{}\n")
    args = Namespace(
        phase="resume",
        source_checkout=source_checkout,
        data_patch=files[0],
        graph_patch=files[1],
        training_config=files[2],
        robot_config=files[3],
        data_config=files[4],
        checkpoint_dir=checkpoint_dir,
        processor_dir=processor_dir,
        dataset_split=dataset_split,
        dataset_manifest=files[5],
        norm_stats=files[6],
        run_dir=run_dir,
        load_global_step=2,
        seed=0,
        capacity=16,
        content_width=256,
        geometry_width=6,
        uncertainty_width=16,
        learning_rate=1e-5,
        max_grad_norm=1.0,
    )
    with pytest.raises(ValueError, match="global step one"):
        _validate_paths_and_args(args)
    args.load_global_step = 1
    _validate_paths_and_args(args)
    args.seed = 0xFFFFFFFF
    with pytest.raises(ValueError, match="uint32"):
        _validate_paths_and_args(args)
