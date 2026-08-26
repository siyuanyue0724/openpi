#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Fail-closed local deployment and 2xA100 readiness audit for ADR-74."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot native preflight",
)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.capacity import (
    MINIMUM_LINGBOT_FREE_STORAGE_BYTES,
    MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
)
from picf_next.lingbot_native.gate_evidence import COMPLETE_ADR74_MISSING_CAPABILITIES

try:
    from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
    from tools.bootstrap_lingbot_vla2_native import (
        CALVIN_OFFLINE_RUNTIME_EXTRAS,
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_AUDIT_TOOLS,
        LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256,
        LINGBOT_NATIVE_REQUIREMENTS_SHA256,
        LINGBOT_NATIVE_RUNTIME_EXTRAS,
        validate_calvin_offline_source,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        load_lingbot_training_config,
        resolve_lingbot_optimizer_contract,
    )
    from tools.run_lingbot_vla2_native_full import (
        _full_implementation_digest,
        _full_implementation_paths,
    )
    from tools.run_lingbot_vla2_native_g0 import _implementation_digest, _implementation_paths
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        validate_checkpoint,
        validate_processor,
    )
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CALVIN_OFFLINE_RUNTIME_EXTRAS,
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_AUDIT_TOOLS,
        LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256,
        LINGBOT_NATIVE_REQUIREMENTS_SHA256,
        LINGBOT_NATIVE_RUNTIME_EXTRAS,
        validate_calvin_offline_source,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        load_lingbot_training_config,
        resolve_lingbot_optimizer_contract,
    )
    from run_lingbot_vla2_native_full import (  # type: ignore[no-redef]
        _full_implementation_digest,
        _full_implementation_paths,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _implementation_digest,
        _implementation_paths,
    )


EXPECTED_PYTHON_MAJOR_MINOR = (3, 12)
PREFLIGHT_REPORT_SCHEMA = "picf-next.lingbot-native-preflight.v4"

REQUIRED_PATHS = (
    "configs/lingbot/calvin_data.json",
    "configs/lingbot/calvin_robot.yaml",
    "docs/74_LINGBOT_NATIVE_PICF_THEORY_FREEZE_OWNER_REVIEW.md",
    "docs/75_ADR74_TEN_PASS_ARCHITECT_AND_REVIEWER_AUDIT.md",
    "docs/76_ADR74_OWNER_APPROVAL_AND_IMPLEMENTATION_LEDGER.md",
    "docs/77_ADR74_LOCAL_DEPLOYMENT_AND_2XA100_G0_RUNBOOK.md",
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
    "references/patches/lingbot_vla2_picf_native.patch",
    "src/picf_next/artifact_io.py",
    "src/picf_next/objective.py",
    "src/picf_next/data/calvin_loss_targets.py",
    "src/picf_next/data/calvin_target_request.py",
    "src/picf_next/data/calvin_physical_supervision_sidecar.py",
    "src/picf_next/data/qwen3vl_raster.py",
    "src/picf_next/eval/calvin_task_relevance.py",
    "src/picf_next/lingbot_native/calvin.py",
    "src/picf_next/lingbot_native/calvin_objective.py",
    "src/picf_next/lingbot_native/calvin_supervision.py",
    "src/picf_next/lingbot_native/controls.py",
    "src/picf_next/lingbot_native/current_grid_cache.py",
    "src/picf_next/lingbot_native/empirical_producers.py",
    "src/picf_next/lingbot_native/graph.py",
    "src/picf_next/lingbot_native/host.py",
    "src/picf_next/lingbot_native/modalities.py",
    "src/picf_next/lingbot_native/objective.py",
    "src/picf_next/lingbot_native/official_config.py",
    "src/picf_next/lingbot_native/prediction.py",
    "src/picf_next/lingbot_native/predictive_decision.py",
    "src/picf_next/lingbot_native/predictive_diagnostics.py",
    "src/picf_next/lingbot_native/predictive_objective.py",
    "src/picf_next/lingbot_native/predictive_probes.py",
    "src/picf_next/lingbot_native/relations.py",
    "src/picf_next/lingbot_native/runtime.py",
    "src/picf_next/lingbot_native/session.py",
    "src/picf_next/lingbot_native/source_mask.py",
    "src/picf_next/lingbot_native/state.py",
    "src/picf_next/lingbot_native/supervision.py",
    "src/picf_next/lingbot_native/temporal.py",
    "src/picf_next/lingbot_native/training.py",
    "src/picf_next/lingbot_native/visual_audit.py",
    "tools/bootstrap_lingbot_vla2_native.py",
    "tools/build_calvin_dataset_manifest.py",
    "tools/build_calvin_normalization.py",
    "tools/build_lingbot_calvin_norm_stats.py",
    "tools/lingbot_vla2_runtime_helpers.py",
    "tools/run_calvin_physical_supervision_parallel.py",
    "tools/run_lingbot_vla2_native_g0.py",
    "tools/smoke_lingbot_vla2_native_full_weight.py",
    "tests/lingbot_native/test_official_config.py",
    "tests/test_artifact_io.py",
    "tests/test_qwen3vl_raster_targets.py",
)

FULL_OBJECTIVE_REQUIRED_PATHS = (
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
)

STATIC_CHECK_PATHS = tuple(
    sorted(
        {
            "src/picf_next/lingbot_native",
            "src/picf_next/data/calvin_loss_targets.py",
            "src/picf_next/data/calvin_target_request.py",
            "src/picf_next/data/qwen3vl_raster.py",
            "tests/lingbot_native",
            "tests/test_qwen3vl_raster_targets.py",
            "tools/bootstrap_lingbot_vla2_native.py",
            "tools/lingbot_vla2_runtime_helpers.py",
            "tools/run_lingbot_vla2_native_g0.py",
            "tools/smoke_lingbot_vla2_native_full_weight.py",
            "tools/preflight_lingbot_native.py",
            *(path for path in FULL_OBJECTIVE_REQUIRED_PATHS if path.endswith(".py")),
        }
    )
)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source-checkout", type=Path)
    parser.add_argument(
        "--calvin-env-root",
        type=Path,
        default=_environment_path("PICF_CALVIN_ENV_ROOT"),
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument(
        "--dataset-split",
        type=Path,
        default=_environment_path("PICF_DATASET_DIR"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=_environment_path("PICF_DATASET_MANIFEST"),
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=_environment_path("PICF_LINGBOT_NORM_STATS"),
    )
    parser.add_argument(
        "--persistent-storage-root",
        type=Path,
        default=_environment_path("PICF_PERSISTENT_STORAGE_ROOT"),
        help="filesystem that will receive run checkpoints and reports",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--require-cloud-g0",
        action="store_true",
        help="exit nonzero unless the complete 2xA100 G0 readiness gate passes",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value is None or not value.strip() else Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_durable(path: Path, payload: str) -> None:
    write_text_durable_exclusive(path, payload)


def _command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "passed": result.returncode == 0,
    }


def _repository_bound_environment(*, root: Path, source: Path) -> dict[str, str]:
    """Bind every preflight child process to the selected immutable sources."""

    repository_paths = (str(root / "src"), str(root))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_entries = [
        *repository_paths,
        *([] if not existing_pythonpath else [existing_pythonpath]),
    ]
    return {
        **os.environ,
        "PICF_LINGBOT_NATIVE_SOURCE": str(source),
        "PYTHONPATH": os.pathsep.join(pythonpath_entries),
    }


def parse_nvidia_smi_inventory(output: str) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError("nvidia-smi inventory line must have exactly three fields")
        index, name, memory_mib = fields
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
            }
        )
    return inventory


def _gpu_inventory() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return []
    return parse_nvidia_smi_inventory(result.stdout)


def _host_memory_bytes() -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or page_size <= 0
        or isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count <= 0
    ):
        raise RuntimeError("host memory inventory is unavailable")
    physical_bytes = page_size * page_count
    cgroup_limits: list[int] = []
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw_limit = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw_limit == "max":
            continue
        try:
            limit = int(raw_limit)
        except ValueError:
            continue
        if 0 < limit < 2**60:
            cgroup_limits.append(limit)
    return min([physical_bytes, *cgroup_limits])


def inspect_hardware_capacity(storage_root: Path | None) -> dict[str, Any]:
    """Measure the host resources that the documented two-A100 recipe requires."""

    resolved_storage: Path | None = None
    free_storage_bytes: int | None = None
    if storage_root is not None:
        resolved_storage = storage_root.expanduser().resolve(strict=True)
        if not resolved_storage.is_dir():
            raise ValueError("persistent storage root is not a directory")
        free_storage_bytes = shutil.disk_usage(resolved_storage).free
    return {
        "host_memory_bytes": _host_memory_bytes(),
        "minimum_host_memory_bytes": MINIMUM_LINGBOT_HOST_MEMORY_BYTES,
        "persistent_storage_root": (None if resolved_storage is None else str(resolved_storage)),
        "free_storage_bytes": free_storage_bytes,
        "minimum_free_storage_bytes": MINIMUM_LINGBOT_FREE_STORAGE_BYTES,
    }


def parse_exact_requirements(path: Path, *, expected_sha256: str) -> dict[str, str]:
    """Parse an immutable all-exact requirements file into metadata names."""

    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("LingBot-native requirements differ from the pinned contract")
    versions: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise ValueError(f"LingBot requirement line {line_number} is not one exact pin")
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in versions:
            raise ValueError(f"LingBot requirements contain duplicate package {name}")
        versions[name] = match.group(2)
    if not versions:
        raise ValueError("LingBot-native requirements are empty")
    return versions


def merge_exact_requirements(*groups: dict[str, str]) -> dict[str, str]:
    """Merge pinned package groups while rejecting every duplicate declaration."""

    merged: dict[str, str] = {}
    for group in groups:
        overlap = set(merged).intersection(group)
        if overlap:
            raise ValueError(f"LingBot runtime repeats pinned packages: {sorted(overlap)}")
        merged.update(group)
    return merged


def probe_python_runtime(python: Path, packages: tuple[str, ...]) -> dict[str, Any]:
    """Read versions from the selected deployment interpreter, never this process."""

    program = """
import importlib.metadata as metadata
import json
import sys
names = json.loads(sys.argv[1])
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    'python_major_minor': list(sys.version_info[:2]),
    'python_version': sys.version,
    'versions': versions,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python), "-c", program, json.dumps(packages)],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "deployment Python metadata probe failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def deployment_python(path: Path, *, root: Path) -> Path:
    """Keep the venv entry path; resolving its symlink discards venv semantics."""

    expanded = path.expanduser()
    return expanded.absolute() if expanded.is_absolute() else (root / expanded).absolute()


def _canonical_python_evidence(path: Path) -> Path:
    """Return the durable executable referent while execution keeps the venv launcher."""

    if not path.is_file():
        raise ValueError(f"preflight Python executable is absent: {path}")
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"preflight Python executable has no real file referent: {path}")
    return resolved


def validate_g0_data_assets(
    *,
    dataset_split: Path | None,
    dataset_manifest: Path | None,
    norm_stats: Path | None,
) -> dict[str, Any]:
    """Validate the exact dataset inputs required by the two-rank G0 runner."""

    paths = {
        "dataset_split": dataset_split,
        "dataset_manifest": dataset_manifest,
        "norm_stats": norm_stats,
    }
    missing = sorted(name for name, path in paths.items() if path is None)
    if len(missing) == len(paths):
        return {"ready": False, "status": "ABSENT", "missing": missing}
    if missing:
        return {"ready": False, "status": "FAIL", "missing": missing}
    if dataset_split is None or dataset_manifest is None or norm_stats is None:
        raise RuntimeError("validated G0 data paths unexpectedly became unavailable")
    try:
        split = dataset_split.resolve()
        manifest_path = dataset_manifest.resolve()
        norm_path = norm_stats.resolve()
        if not split.is_dir() or not manifest_path.is_file() or not norm_path.is_file():
            raise FileNotFoundError("one or more G0 dataset assets are absent")
        manifest = load_dataset_file_manifest(manifest_path)
        normalization = json.loads(norm_path.read_text(encoding="ascii"))
        validate_lingbot_calvin_norm_stats(normalization)
        source = normalization["source"]
        if (
            source["dataset_id"] != manifest.dataset_id
            or source["dataset_revision"] != manifest.dataset_revision
            or source["dataset_tree_sha256"] != manifest.tree_sha256
            or manifest.split_name != split.name
        ):
            raise ValueError("G0 dataset manifest and LingBot normalization differ")
        validation = validate_dataset_runtime_binding(
            manifest,
            split,
            dataset_id=source["dataset_id"],
            dataset_revision=source["dataset_revision"],
            split_name=split.name,
        )
    except BaseException as error:
        return {
            "ready": False,
            "status": "FAIL",
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "ready": True,
        "status": "PASS",
        "dataset_split": str(split),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "norm_stats_sha256": _sha256(norm_path),
        "validation": validation,
    }


def run_preflight(
    *,
    root: Path,
    python: Path,
    source_checkout: Path | None,
    calvin_env_root: Path | None,
    checkpoint_dir: Path | None,
    processor_dir: Path | None,
    dataset_split: Path | None,
    dataset_manifest: Path | None,
    norm_stats: Path | None,
    persistent_storage_root: Path | None,
    run_tests: bool,
) -> dict[str, Any]:
    root = root.resolve()
    python = deployment_python(python, root=root)
    source = root / CHECKOUT_RELATIVE_PATH if source_checkout is None else source_checkout.resolve()
    implementation_paths = _implementation_paths(root)
    full_implementation_paths = _full_implementation_paths(root)
    required_paths = {root / relative for relative in REQUIRED_PATHS}
    required_paths.update(implementation_paths)
    required_paths.update(full_implementation_paths)
    missing = [str(path.relative_to(root)) for path in sorted(required_paths) if not path.is_file()]
    if missing:
        raise ValueError(f"native deployment package is incomplete: {missing}")
    full_objective_missing = [
        relative for relative in FULL_OBJECTIVE_REQUIRED_PATHS if not (root / relative).is_file()
    ]
    python_evidence = _canonical_python_evidence(python)
    patch = verify_native_patch(root=root, checkout=source, check_apply=True)
    released_training_config = source / "configs/vla/robotwin/robotwin.yaml"
    released_training = load_lingbot_training_config(released_training_config)
    released_optimizer = resolve_lingbot_optimizer_contract(
        released_training,
        requested_learning_rate=1e-4,
    )
    requirements = source / "requirements.txt"
    depth_requirements = source / "requirements-depth.txt"
    expected_runtime = merge_exact_requirements(
        parse_exact_requirements(
            requirements,
            expected_sha256=LINGBOT_NATIVE_REQUIREMENTS_SHA256,
        ),
        parse_exact_requirements(
            depth_requirements,
            expected_sha256=LINGBOT_NATIVE_DEPTH_REQUIREMENTS_SHA256,
        ),
        LINGBOT_NATIVE_RUNTIME_EXTRAS,
        LINGBOT_NATIVE_AUDIT_TOOLS,
        CALVIN_OFFLINE_RUNTIME_EXTRAS,
    )
    if calvin_env_root is None:
        calvin_source_report: dict[str, Any] = {
            "status": "ABSENT",
            "error": "CALVIN environment root was not provided",
        }
    else:
        try:
            calvin_source_report = validate_calvin_offline_source(calvin_env_root)
        except BaseException as error:
            calvin_source_report = {
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
            }
    calvin_source_ready = calvin_source_report.get("status") == "PASS"
    static_checks = {
        "required_files": len(required_paths),
        "implementation_files": [str(path.relative_to(root)) for path in implementation_paths],
        "implementation_sha256": _implementation_digest(root),
        "full_implementation_files": [
            str(path.relative_to(root)) for path in full_implementation_paths
        ],
        "full_implementation_sha256": _full_implementation_digest(root),
        "patch_replay": patch,
        "released_training_config_sha256": _sha256(released_training_config),
        "released_optimizer_contract": released_optimizer.metadata,
        "lingbot_requirements_sha256": _sha256(requirements),
        "lingbot_depth_requirements_sha256": _sha256(depth_requirements),
        "calvin_environment": calvin_source_report,
    }
    commands: list[dict[str, Any]] = []
    selected_source_environment = _repository_bound_environment(root=root, source=source)
    if run_tests:
        checked_paths = list(STATIC_CHECK_PATHS)
        static_commands = (
            [str(python), "-m", "pytest", "-q"],
            [str(python), "-m", "ruff", "check", *checked_paths],
            [
                str(python),
                "-m",
                "ruff",
                "format",
                "--check",
                *checked_paths,
            ],
            [str(python), "-m", "compileall", "-q", *checked_paths],
            ["git", "diff", "--check"],
        )
        for command in static_commands:
            commands.append(
                _command(
                    command,
                    cwd=root,
                    env=selected_source_environment,
                )
            )
    import_probe = _command(
        [
            str(python),
            "-c",
            "import pathlib,picf_next; print(pathlib.Path(picf_next.__file__).resolve())",
        ],
        cwd=root,
        env=selected_source_environment,
    )
    commands.append(import_probe)
    import_origin = import_probe["stdout_tail"].strip()
    import_origin_valid = import_probe["passed"] and Path(import_origin).is_relative_to(
        root / "src"
    )
    command_gate = all(item["passed"] for item in commands)
    static_pass = command_gate and import_origin_valid
    local_deployment_pass = static_pass and run_tests
    future_structural_runner_static_ready = local_deployment_pass and not full_objective_missing
    complete_adr74_static_ready = (
        future_structural_runner_static_ready and not COMPLETE_ADR74_MISSING_CAPABILITIES
    )

    checkpoint_report = None
    processor_report = None
    if checkpoint_dir is not None:
        checkpoint_report = validate_checkpoint(checkpoint_dir)
    if processor_dir is not None:
        processor_report = validate_processor(processor_dir)
    model_assets_ready = checkpoint_report is not None and processor_report is not None
    data_report = validate_g0_data_assets(
        dataset_split=dataset_split,
        dataset_manifest=dataset_manifest,
        norm_stats=norm_stats,
    )
    data_ready = data_report["ready"] is True
    assets_ready = model_assets_ready and data_ready
    runtime_probe = probe_python_runtime(python, tuple(sorted(expected_runtime)))
    packages = runtime_probe["versions"]
    calvin_import_prefix = (
        f"import sys;sys.path.insert(0,{json.dumps(str(calvin_env_root.resolve()))});"
        if calvin_source_ready and calvin_env_root is not None
        else "raise RuntimeError('validated CALVIN environment is absent');"
    )
    host_import_probe = _command(
        [
            str(python),
            "-c",
            (
                calvin_import_prefix
                + "import accelerate,click,cv2,flash_attn,gym,hydra,lerobot,mdm,mlflow,"
                "moderngl,moge,"
                "oss2,plyfile,pythonjsonlogger,torch,torchdata,transformers,trimesh,utils3d;"
                "import pybullet,quaternion;"
                "from calvin_env.envs.play_table_env import PlayTableSimEnv;"
                "from lingbotvla.checkpoint import build_checkpointer;"
                "from lingbotvla.data import VLADataCollatorWithPacking;"
                "from lingbotvla.data.vla_data.utils import FeatureTransform;"
                "from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 "
                "import LingbotVlaV2Policy;"
                "print('lingbot-native-runtime-imports-pass')"
            ),
        ],
        cwd=root,
        env=selected_source_environment,
    )
    cloud_runtime_ready = (
        tuple(runtime_probe["python_major_minor"]) == EXPECTED_PYTHON_MAJOR_MINOR
        and packages == expected_runtime
        and calvin_source_ready
        and host_import_probe["passed"]
    )
    gpus = _gpu_inventory()
    hardware_capacity = inspect_hardware_capacity(persistent_storage_root)
    free_storage_bytes = hardware_capacity["free_storage_bytes"]
    cloud_hardware_ready = (
        len(gpus) == 2
        and all("A100" in gpu["name"] for gpu in gpus)
        and all(gpu["memory_mib"] >= 40000 for gpu in gpus)
        and hardware_capacity["host_memory_bytes"] >= hardware_capacity["minimum_host_memory_bytes"]
        and isinstance(free_storage_bytes, int)
        and free_storage_bytes >= hardware_capacity["minimum_free_storage_bytes"]
    )
    cloud_g0_ready = (
        local_deployment_pass and cloud_runtime_ready and cloud_hardware_ready and assets_ready
    )
    return {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "status": "PASS" if cloud_g0_ready else "FAIL",
        "static_contract_pass": static_pass,
        "local_tests_executed": run_tests,
        "local_deployment_pass": local_deployment_pass,
        "g0_action_only_static_ready": local_deployment_pass,
        "future_structural_runner_static_ready": future_structural_runner_static_ready,
        "complete_adr74_static_ready": complete_adr74_static_ready,
        "complete_adr74_missing_capabilities": list(COMPLETE_ADR74_MISSING_CAPABILITIES),
        "released_weight_omitted_static_binding_validated": False,
        "full_objective_static_ready": future_structural_runner_static_ready,
        "full_objective_missing_files": full_objective_missing,
        "cloud_runtime_ready": cloud_runtime_ready,
        "cloud_hardware_ready": cloud_hardware_ready,
        "cloud_model_assets_ready": model_assets_ready,
        "cloud_data_ready": data_ready,
        "cloud_assets_ready": assets_ready,
        "cloud_g0_ready": cloud_g0_ready,
        "authorized_gates": [
            "G0_full_weight_neutral_parity",
            "G0_two_rank_full_update_and_cold_resume",
        ],
        "long_training_authorized": False,
        "scientific_acceptance": "PENDING_G1_G8",
        "root": str(root),
        "python": str(python_evidence),
        "source_checkout": str(source),
        "static_checks": static_checks,
        "commands": commands,
        "import_origin": import_origin,
        "import_origin_valid": import_origin_valid,
        "python_version": runtime_probe["python_version"],
        "python_major_minor": runtime_probe["python_major_minor"],
        "package_versions": packages,
        "expected_cloud_runtime": expected_runtime,
        "host_import_probe": host_import_probe,
        "gpu_inventory": gpus,
        "hardware_capacity": hardware_capacity,
        "checkpoint": checkpoint_report,
        "processor": processor_report,
        "g0_data": data_report,
    }


def require_preflight_pass(
    report: dict[str, Any],
    *,
    tests_executed: bool,
    require_cloud_g0: bool,
) -> None:
    """Raise unless the gate requested by the CLI completed successfully."""

    if report.get("static_contract_pass") is not True:
        raise RuntimeError("LingBot native static contract preflight failed")
    if tests_executed and report.get("local_deployment_pass") is not True:
        raise RuntimeError("LingBot native local deployment preflight failed")
    if require_cloud_g0 and report.get("cloud_g0_ready") is not True:
        raise RuntimeError("LingBot native 2xA100 G0 readiness preflight failed")


def main() -> None:
    args = _parse_args()
    if args.require_cloud_g0 and args.skip_tests:
        raise ValueError("--require-cloud-g0 cannot be combined with --skip-tests")
    report = run_preflight(
        root=args.root,
        python=args.python,
        source_checkout=args.source_checkout,
        calvin_env_root=args.calvin_env_root,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        dataset_split=args.dataset_split,
        dataset_manifest=args.dataset_manifest,
        norm_stats=args.norm_stats,
        persistent_storage_root=args.persistent_storage_root,
        run_tests=not args.skip_tests,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _write_text_durable(args.output, payload)
    print(payload, end="")
    require_preflight_pass(
        report,
        tests_executed=not args.skip_tests,
        require_cloud_g0=args.require_cloud_g0,
    )


if __name__ == "__main__":
    main()
