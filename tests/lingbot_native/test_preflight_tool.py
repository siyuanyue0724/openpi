from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.preflight_lingbot_native as preflight
from tools.bootstrap_lingbot_vla2_native import (
    CHECKOUT_RELATIVE_PATH,
    LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256,
    LINGBOT_NATIVE_REQUIREMENTS_SHA256,
)
from tools.preflight_lingbot_native import (
    COMPLETE_ADR74_MISSING_CAPABILITIES,
    FULL_OBJECTIVE_REQUIRED_PATHS,
    REQUIRED_PATHS,
    STATIC_CHECK_PATHS,
    _canonical_python_evidence,
    _write_text_durable,
    deployment_python,
    inspect_hardware_capacity,
    merge_exact_requirements,
    parse_exact_requirements,
    parse_nvidia_smi_inventory,
    probe_python_runtime,
    require_preflight_pass,
    validate_g0_data_assets,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/preflight_lingbot_native.py"


def test_native_preflight_delays_torch_and_lingbot_imports() -> None:
    tree = ast.parse(TOOL.read_text())
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_from_imports = {
        (node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    assert {"torch", "lingbotvla", "transformers"}.isdisjoint(top_imports | top_from_imports)


def test_native_preflight_rejects_cloud_gate_without_tests() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--require-cloud-g0", "--skip-tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--require-cloud-g0 cannot be combined with --skip-tests" in result.stderr


def test_native_gpu_inventory_requires_exact_machine_readable_fields() -> None:
    inventory = parse_nvidia_smi_inventory(
        "0, NVIDIA A100-SXM4-40GB, 40960\n1, NVIDIA A100-SXM4-40GB, 40960\n"
    )
    assert len(inventory) == 2
    assert inventory[0] == {
        "index": 0,
        "name": "NVIDIA A100-SXM4-40GB",
        "memory_mib": 40960,
    }
    with pytest.raises(ValueError, match="exactly three"):
        parse_nvidia_smi_inventory("0, malformed")


def test_native_gpu_inventory_reports_absence_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run without nvidia-smi"),
    )

    assert preflight._gpu_inventory() == []


def test_native_hardware_capacity_requires_explicit_storage_and_frozen_reserves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_host_memory_bytes",
        lambda: preflight.MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
    )
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=preflight.MINIMUM_LINGBOT_FREE_STORAGE_BYTES),
    )
    absent = inspect_hardware_capacity(None)
    assert absent["persistent_storage_root"] is None
    assert absent["free_storage_bytes"] is None
    measured = inspect_hardware_capacity(tmp_path)
    assert measured == {
        "host_memory_bytes": preflight.MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
        "minimum_host_memory_bytes": preflight.MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
        "persistent_storage_root": str(tmp_path.resolve()),
        "free_storage_bytes": preflight.MINIMUM_LINGBOT_FREE_STORAGE_BYTES,
        "minimum_free_storage_bytes": preflight.MINIMUM_LINGBOT_FREE_STORAGE_BYTES,
    }


def test_native_host_memory_is_bounded_by_the_container_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "SC_PAGE_SIZE": 4096,
        "SC_PHYS_PAGES": 256 * 2**30 // 4096,
    }
    monkeypatch.setattr(preflight.os, "sysconf", lambda key: values[key])
    real_read_text = Path.read_text

    def fake_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/sys/fs/cgroup/memory.max"):
            return str(128 * 2**30)
        if path == Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            raise FileNotFoundError(path)
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert preflight._host_memory_bytes() == 128 * 2**30


def test_native_preflight_keeps_long_training_unauthorized() -> None:
    source = TOOL.read_text()
    assert '"G0_full_weight_neutral_parity"' in source
    assert '"G0_two_rank_full_update_and_cold_resume"' in source
    assert '"long_training_authorized": False' in source
    assert '"scientific_acceptance": "PENDING_G1_G8"' in source
    assert '"cloud_data_ready": data_ready' in source
    assert "model_assets_ready and data_ready" in source
    assert "resolve_lingbot_optimizer_contract(" in source
    assert '"released_optimizer_contract": released_optimizer.metadata' in source


def test_native_preflight_requires_all_g0_data_assets(tmp_path: Path, monkeypatch) -> None:
    absent = validate_g0_data_assets(
        dataset_split=None,
        dataset_manifest=None,
        norm_stats=None,
    )
    assert absent == {
        "ready": False,
        "status": "ABSENT",
        "missing": ["dataset_manifest", "dataset_split", "norm_stats"],
    }
    partial = validate_g0_data_assets(
        dataset_split=tmp_path,
        dataset_manifest=None,
        norm_stats=None,
    )
    assert partial["status"] == "FAIL"

    split = tmp_path / "training"
    split.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="ascii")
    norm_path = tmp_path / "norm.json"
    norm_path.write_text(
        (
            '{"source":{"dataset_id":"calvin","dataset_revision":"rev",'
            '"dataset_tree_sha256":"' + "d" * 64 + '"}}'
        ),
        encoding="ascii",
    )

    class Manifest:
        dataset_id = "calvin"
        dataset_revision = "rev"
        split_name = "training"
        tree_sha256 = "d" * 64

    monkeypatch.setattr(preflight, "load_dataset_file_manifest", lambda _path: Manifest())
    monkeypatch.setattr(preflight, "validate_lingbot_calvin_norm_stats", lambda _value: None)
    monkeypatch.setattr(
        preflight,
        "validate_dataset_runtime_binding",
        lambda *_args, **_kwargs: {"status": "PASS", "file_count": 1},
    )
    report = validate_g0_data_assets(
        dataset_split=split,
        dataset_manifest=manifest_path,
        norm_stats=norm_path,
    )
    assert report["ready"] is True
    assert report["status"] == "PASS"
    assert report["validation"]["file_count"] == 1


def test_native_preflight_requires_calvin_structural_objective_assets() -> None:
    required = set(REQUIRED_PATHS)
    assert {
        "src/picf_next/lingbot_native/modalities.py",
        "src/picf_next/data/calvin_loss_targets.py",
        "src/picf_next/data/calvin_target_request.py",
        "src/picf_next/data/calvin_physical_supervision_sidecar.py",
        "src/picf_next/data/qwen3vl_raster.py",
        "src/picf_next/eval/calvin_task_relevance.py",
        "src/picf_next/lingbot_native/calvin_objective.py",
        "src/picf_next/lingbot_native/calvin_supervision.py",
        "src/picf_next/lingbot_native/visual_audit.py",
        "tests/test_qwen3vl_raster_targets.py",
        "docs/78_LINGBOT_NATIVE_FULL_DEPLOYMENT_DEEP_AUDIT_20260722.md",
        "docs/79_LINGBOT_NATIVE_EMPIRICAL_EVIDENCE_CONTRACT.md",
        "docs/80_LINGBOT_NATIVE_MODEL_SPECIFIC_EVALUATION_PRODUCERS_ADR.md",
        "docs/81_LINGBOT_NATIVE_JEPA_CAPACITY_AND_TARGET_AUDIT.md",
        "docs/82_PREDICTIVE_CORRECTION_OBJECTIVE_OWNER_DECISION.md",
        "docs/83_PREDICTIVE_CORRECTION_LOCAL_CLOSURE_AND_ARCHITECT_REVIEW.md",
        "docs/84_CONTINUOUS_POSTERIOR_ESTIMATOR_AND_LOCAL_CLOSURE_AUDIT.md",
        "docs/85_LINGBOT_MOE_DETERMINISM_AND_RELEASED_WEIGHT_G0_AUDIT.md",
        "docs/86_CALVIN_CONTENT_MANIFEST_AND_VERIFIED_READ_SCALABILITY_AUDIT.md",
        "docs/87_PREDICTIVE_FIXED_BATCH_PRODUCTION_CLOSURE.md",
        "docs/88_CLOUD_EXECUTION_LEDGER_20260724.md",
        "docs/89_RELATIVE_IMPORT_AND_LEGACY_REACHABILITY_CLOSURE.md",
        "docs/90_CALVIN_OFFLINE_RUNTIME_AND_PREFLIGHT_V3_CLOSURE.md",
        "docs/91_CALVIN_SELECTIVE_PHYSICAL_SUPERVISION_V4_CLOSURE.md",
        "docs/92_FSDP2_MUTABLE_CONTEXT_ABI_AND_G0_CLOSURE.md",
        "docs/93_LINGBOT_TYPED_CONFIG_SEMANTICS_AND_G0_CLOSURE.md",
        "docs/94_G0_ROUTING_PROVENANCE_ABI_AND_EARLY_VALIDATION.md",
        "docs/97_CALVIN_SELECTIVE_CALIBRATION_V5_TAIL_CLOSURE.md",
        "docs/98_CALVIN_PARALLEL_BUILD_ORCHESTRATION.md",
        "tools/run_calvin_physical_supervision_parallel.py",
    } <= required
    assert {
        "src/picf_next/lingbot_native/current_grid_cache.py",
        "src/picf_next/lingbot_native/empirical_producers.py",
        "src/picf_next/lingbot_native/empirical_statistics.py",
        "src/picf_next/lingbot_native/full_training.py",
        "src/picf_next/lingbot_native/gate_evidence.py",
        "src/picf_next/lingbot_native/predictive_cache.py",
        "src/picf_next/lingbot_native/predictive_decision.py",
        "src/picf_next/lingbot_native/predictive_diagnostics.py",
        "src/picf_next/lingbot_native/predictive_plan.py",
        "src/picf_next/lingbot_native/predictive_probes.py",
        "tools/audit_lingbot_attention_backend_parity.py",
        "tools/audit_lingbot_dino_teacher_causality.py",
        "tools/audit_lingbot_predictive_targets.py",
        "tools/audit_lingbot_predictive_temporal_targets.py",
        "tools/build_lingbot_calvin_current_grid_cache.py",
        "tools/build_lingbot_calvin_predictive_cache.py",
        "tools/build_lingbot_native_empirical_report.py",
        "tools/build_lingbot_native_empirical_observations.py",
        "tools/build_lingbot_native_evaluation_plan.py",
        "tools/build_lingbot_native_gate_decision.py",
        "tools/build_lingbot_predictive_objective_decision.py",
        "tools/build_lingbot_native_training_authorization.py",
        "tools/run_lingbot_vla2_native_full.py",
        "tests/lingbot_native/test_attention_backend_parity_tool.py",
        "tests/lingbot_native/test_current_grid_cache.py",
        "tests/lingbot_native/test_current_grid_cache_builder.py",
        "tests/lingbot_native/test_full_training.py",
        "tests/lingbot_native/test_empirical_producers.py",
        "tests/lingbot_native/test_gate_evidence.py",
        "tests/lingbot_native/test_predictive_cache_builder.py",
        "tests/lingbot_native/test_predictive_cache.py",
        "tests/lingbot_native/test_predictive_diagnostics.py",
        "tests/lingbot_native/test_predictive_objective_decision.py",
        "tests/lingbot_native/test_predictive_temporal_audit.py",
        "tests/lingbot_native/test_predictive_plan.py",
        "tests/lingbot_native/test_predictive_probes.py",
        "tests/lingbot_native/test_teacher_causality_audit.py",
        "tests/lingbot_native/test_training_authorization_tool.py",
        "tests/lingbot_native/test_gate_decision_tool.py",
        "tests/lingbot_native/test_full_runner_contract.py",
        "tests/lingbot_native/test_visual_audit.py",
    } == set(FULL_OBJECTIVE_REQUIRED_PATHS)
    assert COMPLETE_ADR74_MISSING_CAPABILITIES == ()
    assert {
        "src/picf_next/lingbot_native/empirical_statistics.py",
        "src/picf_next/lingbot_native/empirical_producers.py",
        "src/picf_next/lingbot_native/gate_evidence.py",
        "tools/build_lingbot_native_empirical_observations.py",
        "tools/build_lingbot_native_empirical_report.py",
        "tools/build_lingbot_native_evaluation_plan.py",
    } <= set(STATIC_CHECK_PATHS)
    source = TOOL.read_text()
    assert '"g0_action_only_static_ready": local_deployment_pass' in source
    assert '"future_structural_runner_static_ready":' in source
    assert '"complete_adr74_static_ready": complete_adr74_static_ready' in source
    assert '"full_objective_static_ready": future_structural_runner_static_ready' in source
    assert '"full_objective_missing_files": full_objective_missing' in source
    assert '"released_weight_omitted_static_binding_validated": False' in source


def test_native_preflight_report_publication_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "preflight" / "report.json"
    _write_text_durable(output, '{"static_contract_pass":true}\n')
    assert output.read_text() == '{"static_contract_pass":true}\n'
    assert not tuple(output.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        _write_text_durable(output, '{"static_contract_pass":false}\n')
    assert output.read_text() == '{"static_contract_pass":true}\n'

    external = tmp_path / "external.json"
    external.write_text("original\n")
    link = tmp_path / "preflight-link.json"
    link.symlink_to(external)
    with pytest.raises(FileExistsError):
        _write_text_durable(link, "replacement\n")
    assert external.read_text() == "original\n"


def test_native_preflight_exit_status_is_fail_closed_for_requested_gate() -> None:
    passing = {
        "static_contract_pass": True,
        "local_deployment_pass": True,
        "cloud_g0_ready": True,
    }
    require_preflight_pass(passing, tests_executed=True, require_cloud_g0=True)

    with pytest.raises(RuntimeError, match="static contract"):
        require_preflight_pass(
            passing | {"static_contract_pass": False},
            tests_executed=False,
            require_cloud_g0=False,
        )
    with pytest.raises(RuntimeError, match="local deployment"):
        require_preflight_pass(
            passing | {"local_deployment_pass": False},
            tests_executed=True,
            require_cloud_g0=False,
        )
    with pytest.raises(RuntimeError, match="2xA100 G0 readiness"):
        require_preflight_pass(
            passing | {"cloud_g0_ready": False},
            tests_executed=True,
            require_cloud_g0=True,
        )


def test_native_preflight_skip_tests_is_only_a_static_audit() -> None:
    require_preflight_pass(
        {
            "static_contract_pass": True,
            "local_deployment_pass": False,
            "cloud_g0_ready": False,
        },
        tests_executed=False,
        require_cloud_g0=False,
    )


def test_native_preflight_preserves_virtual_environment_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_text("")
    environment = tmp_path / "venv"
    environment.mkdir()
    python = environment / "python"
    python.symlink_to(target)
    selected = deployment_python(Path("venv/python"), root=tmp_path)
    assert selected == python
    assert selected != target
    assert _canonical_python_evidence(selected) == target


def test_native_preflight_derives_exact_runtime_from_immutable_requirements() -> None:
    source = ROOT / CHECKOUT_RELATIVE_PATH
    if not source.exists():
        pytest.skip("optional pinned LingBot requirements are absent")
    requirements = source / "requirements.txt"
    depth_requirements = source / "requirements-depth.txt"
    versions = parse_exact_requirements(
        requirements,
        expected_sha256=LINGBOT_NATIVE_REQUIREMENTS_SHA256,
    )
    assert versions["torch"] == "2.8.0"
    assert versions["datasets"] == "3.6.0"
    assert versions["huggingface-hub"] == "0.34.3"
    depth = parse_exact_requirements(
        depth_requirements,
        expected_sha256=LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256,
    )
    assert depth["accelerate"] == "1.7.0"
    assert depth["trimesh"] == "4.5.1"
    assert set(merge_exact_requirements(versions, depth)) == set(versions) | set(depth)
    with pytest.raises(ValueError, match="repeats pinned packages"):
        merge_exact_requirements({"torch": "2.8.0"}, {"torch": "2.8.0"})
    runtime = probe_python_runtime(Path(sys.executable), ("pytest", "definitely-absent-picf"))
    assert runtime["versions"]["pytest"] is not None
    assert runtime["versions"]["definitely-absent-picf"] is None


def test_run_preflight_composes_local_cloud_and_long_training_gates_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ROOT / CHECKOUT_RELATIVE_PATH
    if not source.exists():
        pytest.skip("optional pinned LingBot requirements are absent")
    expected_runtime = merge_exact_requirements(
        parse_exact_requirements(
            source / "requirements.txt",
            expected_sha256=LINGBOT_NATIVE_REQUIREMENTS_SHA256,
        ),
        parse_exact_requirements(
            source / "requirements-depth.txt",
            expected_sha256=LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256,
        ),
        preflight.LINGBOT_NATIVE_RUNTIME_EXTRAS,
        preflight.LINGBOT_NATIVE_AUDIT_TOOLS,
        preflight.CALVIN_OFFLINE_RUNTIME_EXTRAS,
    )
    state = {
        "command_pass": True,
        "data_ready": True,
        "gpus": [
            {"index": 0, "name": "NVIDIA A100-SXM4-40GB", "memory_mib": 40960},
            {"index": 1, "name": "NVIDIA A100-SXM4-40GB", "memory_mib": 40960},
        ],
        "runtime_versions": dict(expected_runtime),
        "free_storage_bytes": preflight.MINIMUM_LINGBOT_FREE_STORAGE_BYTES,
    }

    selected_source_environments: list[dict[str, str] | None] = []
    command_environments: list[dict[str, str] | None] = []

    def fake_command(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> dict[str, object]:
        assert cwd == ROOT
        command_environments.append(env)
        if "-m" in command and "pytest" in command:
            selected_source_environments.append(env)
        program = command[-1] if "-c" in command else ""
        import_origin = ROOT / "src/picf_next/__init__.py"
        stdout = f"{import_origin}\n" if "pathlib,picf_next" in program else "pass\n"
        passed = bool(state["command_pass"])
        return {
            "command": command,
            "returncode": 0 if passed else 1,
            "stdout_tail": stdout if passed else "",
            "stderr_tail": "" if passed else "synthetic failure",
            "passed": passed,
        }

    monkeypatch.setattr(preflight, "_command", fake_command)
    monkeypatch.setattr(preflight, "verify_native_patch", lambda **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(
        preflight,
        "validate_calvin_offline_source",
        lambda _path: {
            "status": "PASS",
            "calvin_env_root": str((tmp_path / "calvin/calvin_env").resolve()),
            "calvin_commit": "f" * 40,
            "calvin_env_commit": "e" * 40,
            "calvin_requirements_sha256": "c" * 64,
            "calvin_setup_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(preflight, "validate_checkpoint", lambda _path: {"status": "PASS"})
    monkeypatch.setattr(preflight, "validate_processor", lambda _path: {"status": "PASS"})
    monkeypatch.setattr(
        preflight,
        "validate_g0_data_assets",
        lambda **_kwargs: {
            "ready": bool(state["data_ready"]),
            "status": "PASS" if state["data_ready"] else "FAIL",
        },
    )
    monkeypatch.setattr(preflight, "_gpu_inventory", lambda: state["gpus"])
    monkeypatch.setattr(
        preflight,
        "inspect_hardware_capacity",
        lambda storage_root: {
            "host_memory_bytes": preflight.MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
            "minimum_host_memory_bytes": preflight.MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
            "persistent_storage_root": str(storage_root.resolve()),
            "free_storage_bytes": state["free_storage_bytes"],
            "minimum_free_storage_bytes": preflight.MINIMUM_LINGBOT_FREE_STORAGE_BYTES,
        },
    )
    monkeypatch.setattr(
        preflight,
        "probe_python_runtime",
        lambda _python, _packages: {
            "python_major_minor": list(preflight.EXPECTED_PYTHON_MAJOR_MINOR),
            "python_version": "synthetic-python",
            "versions": state["runtime_versions"],
        },
    )

    def run(*, run_tests: bool = True) -> dict[str, object]:
        return preflight.run_preflight(
            root=ROOT,
            python=Path(sys.executable),
            source_checkout=source,
            calvin_env_root=tmp_path / "calvin/calvin_env",
            checkpoint_dir=tmp_path / "checkpoint",
            processor_dir=tmp_path / "processor",
            dataset_split=tmp_path / "dataset",
            dataset_manifest=tmp_path / "manifest.json",
            norm_stats=tmp_path / "norm.json",
            persistent_storage_root=tmp_path,
            run_tests=run_tests,
        )

    passing = run()
    assert selected_source_environments
    assert selected_source_environments[0] is not None
    assert selected_source_environments[0]["PICF_LINGBOT_NATIVE_SOURCE"] == str(source.resolve())
    expected_pythonpath = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    assert all(environment is not None for environment in command_environments)
    assert all(
        environment["PYTHONPATH"].startswith(expected_pythonpath)
        for environment in command_environments
        if environment is not None
    )
    assert passing["status"] == "PASS"
    assert passing["python"] == str(Path(sys.executable).resolve())
    assert passing["local_deployment_pass"] is True
    assert passing["cloud_g0_ready"] is True
    assert passing["long_training_authorized"] is False
    assert passing["scientific_acceptance"] == "PENDING_G1_G8"

    state["gpus"] = state["gpus"][:1]
    assert run()["cloud_hardware_ready"] is False
    assert run()["cloud_g0_ready"] is False

    state["gpus"] = [
        {"index": 0, "name": "NVIDIA A100-SXM4-40GB", "memory_mib": 40960},
        {"index": 1, "name": "NVIDIA A100-SXM4-40GB", "memory_mib": 40960},
    ]
    state["runtime_versions"] = dict(expected_runtime) | {"torch": "0.0.0"}
    assert run()["cloud_runtime_ready"] is False

    state["runtime_versions"] = dict(expected_runtime)
    state["data_ready"] = False
    assert run()["cloud_data_ready"] is False
    assert run()["cloud_g0_ready"] is False

    state["data_ready"] = True
    state["free_storage_bytes"] = preflight.MINIMUM_LINGBOT_FREE_STORAGE_BYTES - 1
    assert run()["cloud_hardware_ready"] is False
    assert run()["cloud_g0_ready"] is False

    state["free_storage_bytes"] = preflight.MINIMUM_LINGBOT_FREE_STORAGE_BYTES
    static_only = run(run_tests=False)
    assert static_only["static_contract_pass"] is True
    assert static_only["local_deployment_pass"] is False
    assert static_only["cloud_g0_ready"] is False
