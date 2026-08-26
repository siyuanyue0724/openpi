#!/usr/bin/env python3
"""Fail-closed local and cloud preflight for the unified LingBot candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_ID,
        LINGBOT_CHECKPOINT_REVISION,
        LINGBOT_SOURCE_COMMIT,
        QWEN_PROCESSOR_ID,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from tools.verify_lingbot_vla2_patch import detect_patch_state
    from tools.verify_lingbot_vla2_unified_patch import verify_unified_patches
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_CHECKPOINT_ID,
        LINGBOT_CHECKPOINT_REVISION,
        LINGBOT_SOURCE_COMMIT,
        QWEN_PROCESSOR_ID,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from verify_lingbot_vla2_patch import detect_patch_state  # type: ignore[no-redef]
    from verify_lingbot_vla2_unified_patch import (  # type: ignore[no-redef]
        verify_unified_patches,
    )


CONFIG_RELATIVE_PATH = Path("configs/cloud/2xa100_40g_lingbot_unified.json")
PATCH_HASHES = {
    "references/patches/lingbot_vla2_lerobot_data_compat.patch": (
        "d3aed997a51e87048751c893b3bf5d61dca747148255718c680a35f6bdcc0ed7"
    ),
    "references/patches/lingbot_vla2_unified_belief_graph.patch": (
        "d6bea58c84b0dc871d48624945180ad32afdfc92e1655ef85bbe725b0bcf223f"
    ),
}
RUNTIME_VERSIONS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "4.57.3",
    "datasets": "4.1.1",
    "huggingface-hub": "0.34.3",
    "lerobot": "0.4.3",
}
REQUIREMENTS_SHA256 = "4bea8eca2e5e81107332947fe38d9a2787bc6a8fe4d3f875fa7e3d028f48993d"
ORDERED_GATES = [
    "G0_full_weight_neutral_parity",
    "G1_single_batch_gradient_and_restart",
    "G2_representation_and_anchor_visual_audit",
    "G3_long_age_occlusion_and_reappearance",
    "G4_cross_modal_missing_and_corruption",
    "G5_matched_action_learning_curves",
    "G6_calvin_libero_closed_loop",
    "G7_scale_throughput_and_restart",
    "G8_second_host_transfer",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_durable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_output(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


_CGROUP_MEMORY_LIMIT_PATHS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)


def _system_memory_gib(
    *,
    cgroup_limit_paths: tuple[Path, ...] = _CGROUP_MEMORY_LIMIT_PATHS,
) -> float:
    """Return the effective host/container memory ceiling."""

    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    for path in cgroup_limit_paths:
        try:
            raw_limit = path.read_text().strip()
        except OSError:
            continue
        if not raw_limit or raw_limit == "max":
            continue
        try:
            limit_bytes = int(raw_limit)
        except ValueError:
            continue
        if limit_bytes > 0:
            memory_bytes = min(memory_bytes, limit_bytes)
    return memory_bytes / 2**30


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "picf-next.lingbot-unified-cloud.v2":
        raise ValueError("unsupported unified LingBot cloud schema")
    host = config.get("host", {})
    expected_host = {
        "name": "LingBot-VLA2-6B",
        "source_commit": LINGBOT_SOURCE_COMMIT,
        "checkpoint_id": LINGBOT_CHECKPOINT_ID,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_revision": QWEN_PROCESSOR_REVISION,
    }
    for key, expected in expected_host.items():
        if host.get(key) != expected:
            raise ValueError(f"unified host contract changed {key}")
    patches = host.get("patches")
    if (
        not isinstance(patches, list)
        or {item.get("path"): item.get("sha256") for item in patches} != PATCH_HASHES
    ):
        raise ValueError("unified patch paths or hashes changed")

    runtime = config.get("runtime", {})
    if runtime.get("python_major_minor") != [3, 12]:
        raise ValueError("cloud Python contract must remain 3.12")
    for package, expected in RUNTIME_VERSIONS.items():
        if runtime.get(package) != expected:
            raise ValueError(f"cloud runtime version changed {package}")
    if runtime.get("lerobot_install_mode") != "no-deps":
        raise ValueError("LeRobot compatibility install mode must remain no-deps")
    if runtime.get("dataset_manifest_validation") != "full_sha256_before_accelerator":
        raise ValueError("dataset validation must hash the full manifest before accelerator use")
    if runtime.get("lingbot_requirements_sha256") != REQUIREMENTS_SHA256:
        raise ValueError("LingBot requirements digest changed")

    hardware = config.get("hardware", {})
    if hardware.get("gpu_count") != 2:
        raise ValueError("unified profile requires exactly two GPUs")
    if hardware.get("gpu_name_contains") != "A100":
        raise ValueError("unified profile requires A100 GPUs")
    if hardware.get("minimum_gpu_memory_gib", 0) < 39:
        raise ValueError("unified profile requires 40 GB-class GPUs")
    if hardware.get("minimum_system_memory_gib", 0) < 128:
        raise ValueError("unified profile requires at least 128 GiB host memory")
    if hardware.get("minimum_free_storage_gib", 0) < 250:
        raise ValueError("unified profile requires at least 250 GiB free storage")

    paths = config.get("paths", {})
    required_envs = {
        "source_checkout_env": "PICF_LINGBOT_SOURCE",
        "checkpoint_root_env": "PICF_CHECKPOINT_DIR",
        "processor_root_env": "PICF_PROCESSOR_DIR",
        "dataset_root_env": "PICF_DATASET_DIR",
        "dataset_manifest_env": "PICF_DATASET_MANIFEST",
        "lingbot_norm_stats_env": "PICF_LINGBOT_NORM_STATS",
        "run_root_env": "PICF_RUN_DIR",
        "g0_image_env": "PICF_G0_IMAGE",
        "persistent_prefix": "/mnt",
    }
    if paths != required_envs:
        raise ValueError("persistent path contract changed")

    graph = config.get("unified_graph", {})
    expected_graph = {
        "capacity": 16,
        "content_width": 256,
        "geometry_width": 6,
        "geometry_schema": {
            "names": [
                "center.x",
                "center.y",
                "center.z",
                "extent.x",
                "extent.y",
                "extent.z",
            ],
            "units": ["metre"] * 6,
            "frame": "camera",
        },
        "uncertainty_width": 16,
        "modalities": ["vision"],
        "dense_native_tokens_retained": True,
        "categorical_lifecycle": ["continue", "birth", "empty"],
        "hard_identity_threshold": False,
        "mask_or_box_runtime_input": False,
        "missing_modality_is_absent_factor": True,
        "persistent_state_dtype": "float32",
    }
    if graph != expected_graph:
        raise ValueError("unified graph contract changed")

    temporal = config.get("temporal_contract", {})
    expected_temporal = {
        "runtime_memories": 1,
        "ordered_lane_state": "one_detached_posterior",
        "sparse_bptt_steps": [2, 4],
        "maximum_burn_in_steps": 2,
        "packed_horizon_schedule": "powers_of_two_plus_endpoint",
        "future_targets_deploy_visible": False,
    }
    if temporal != expected_temporal:
        raise ValueError("temporal state contract changed")

    objective = config.get("objective_contract", {})
    expected_objective = {
        "primary_term": "action",
        "action_weight": 1.0,
        "valid_count_normalization": True,
        "host_native_terms": "preserve_official_differentiable_total",
        "set_supervision": "optional_loss_side_only",
        "cross_modal_prediction": "disabled_until_second_physical_modality",
        "host_future_queries": "preserve_when_enabled_by_host",
        "belief_overshooting": "disabled_until_G2_current_set_passes",
        "target_tensors_deploy_visible": False,
    }
    if objective != expected_objective:
        raise ValueError("unified objective contract changed")

    g1_profile = config.get("g1_profile", {})
    expected_g1_profile = {
        "launcher": "tools/run_lingbot_vla2_unified_g1.py",
        "phases": ["fresh", "resume"],
        "world_size": 2,
        "data_parallel_mode": "fsdp2",
        "full_shard": True,
        "cpu_offload": True,
        "gradient_checkpointing": True,
        "global_batch_size": 2,
        "micro_batch_size_per_rank": 1,
        "optimizer": "adamw_full_parameter_g1",
        "master_parameter_dtype": "float32",
        "compute_dtype": "bfloat16",
        "attention_implementation": "eager_gate_only",
        "image_augmentation": False,
        "auxiliary_target_losses": False,
        "optimizer_updates_per_phase": 1,
        "temporal_scope": "one_step_fresh_then_one_step_cold_resume",
        "sparse_bptt_enabled": False,
        "rank_local_picf_state_in_official_dcp": True,
        "checkpoint_boundary_verification": ("exact_rank_local_model_optimizer_picf_rng_sha256"),
    }
    if g1_profile != expected_g1_profile:
        raise ValueError("unified G1 execution contract changed")

    experiment = config.get("experiment_contract", {})
    if experiment.get("arms") != [
        "A_native_host",
        "B_dense_evidence_no_persistence",
        "C_unified_picf",
    ]:
        raise ValueError("matched A/B/C arms changed")
    if experiment.get("initial_authorization") != [ORDERED_GATES[0]]:
        raise ValueError("only G0 may be initially authorized")
    if experiment.get("formal_30000_step_authorized") is not False:
        raise ValueError("30k training must remain unauthorized before G0-G5")
    if experiment.get("ordered_gates") != ORDERED_GATES:
        raise ValueError("ordered unified acceptance gates changed")


def static_preflight(
    config: dict[str, Any],
    *,
    root: Path,
    source_checkout: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    validate_config(config)
    patch_hashes = {}
    for relative, expected in PATCH_HASHES.items():
        path = root / relative
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"unified patch digest differs: {relative}")
        patch_hashes[relative] = actual
    source = (
        root / "references/source_checkouts/lingbot-vla-v2-unified"
        if source_checkout is None
        else source_checkout.resolve()
    )
    requirements = source / "requirements.txt"
    if _sha256(requirements) != REQUIREMENTS_SHA256:
        raise ValueError("pinned LingBot requirements file changed")
    if _git_output(source, "rev-parse", "HEAD") != LINGBOT_SOURCE_COMMIT:
        raise ValueError("local LingBot source is not at the pinned commit")
    patch_report = verify_unified_patches(root=root, checkout=source)
    required_files = (
        root / "tools/bootstrap_lingbot_vla2_unified.py",
        root / "tools/bootstrap_lingbot_runtime.py",
        root / "tools/build_lingbot_calvin_norm_stats.py",
        root / "tools/run_lingbot_vla2_unified_g1.py",
        root / "tools/smoke_lingbot_vla2_unified_full_weight.py",
        root / "configs/lingbot/calvin_data.json",
        root / "configs/lingbot/calvin_robot.yaml",
        root / "src/picf_next/data/lingbot_calvin.py",
        root / "src/picf_next/hosts/lingbot_calvin_training.py",
        root / "src/picf_next/hosts/lingbot_unified.py",
        root / "src/picf_next/hosts/lingbot_unified_training.py",
        root / "src/picf_next/unified/codec.py",
        root / "src/picf_next/unified/coreference.py",
        root / "src/picf_next/unified/graph.py",
        root / "src/picf_next/unified/lifecycle.py",
        root / "src/picf_next/unified/objective.py",
        root / "src/picf_next/unified/predictive.py",
        root / "src/picf_next/unified/retrieval.py",
        root / "src/picf_next/unified/state.py",
        root / "src/picf_next/unified/supervision.py",
        root / "src/picf_next/unified/temporal.py",
        root / "docs/66_UNIFIED_PICF_LOCAL_IMPLEMENTATION_CHECKLIST.md",
        root / "docs/67_UNIFIED_PICF_IMPLEMENTATION_AND_DEPLOYMENT_AUDIT.md",
    )
    if any(not path.is_file() for path in required_files):
        raise ValueError("unified cloud execution package is incomplete")
    return {
        "schema": "picf-next.lingbot-unified-preflight-report.v2",
        "mode": "static",
        "source_commit": LINGBOT_SOURCE_COMMIT,
        "source_checkout": str(source),
        "patch_hashes": patch_hashes,
        "patch_replay": patch_report,
        "formal_30000_step_authorized": False,
        "initial_authorization": [ORDERED_GATES[0]],
    }


def runtime_preflight(config: dict[str, Any], *, root: Path) -> dict[str, Any]:
    validate_config(config)
    paths = config["paths"]
    resolved = {}
    persistent_prefix = Path(paths["persistent_prefix"]).resolve()
    for field in (
        "source_checkout_env",
        "checkpoint_root_env",
        "processor_root_env",
        "dataset_root_env",
        "dataset_manifest_env",
        "lingbot_norm_stats_env",
        "run_root_env",
        "g0_image_env",
    ):
        environment_name = paths[field]
        raw = os.environ.get(environment_name)
        if not raw:
            raise ValueError(f"required environment variable is absent: {environment_name}")
        path = Path(raw).resolve()
        if not path.is_relative_to(persistent_prefix):
            raise ValueError(f"{environment_name} must resolve under {persistent_prefix}")
        resolved[environment_name] = path
    checkout = resolved[paths["source_checkout_env"]]
    report = static_preflight(config, root=root, source_checkout=checkout)
    checkpoint = resolved[paths["checkpoint_root_env"]]
    processor = resolved[paths["processor_root_env"]]
    dataset = resolved[paths["dataset_root_env"]]
    dataset_manifest_path = resolved[paths["dataset_manifest_env"]]
    norm_stats_path = resolved[paths["lingbot_norm_stats_env"]]
    run_root = resolved[paths["run_root_env"]]
    image = resolved[paths["g0_image_env"]]
    if (
        not dataset.is_dir()
        or not dataset_manifest_path.is_file()
        or not norm_stats_path.is_file()
        or not image.is_file()
    ):
        raise ValueError("dataset, manifest, normalization and G0 image must exist")
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_report = validate_checkpoint(checkpoint)
    processor_report = validate_processor(processor)

    sys.path.insert(0, str(root / "src"))
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_files,
    )

    dataset_manifest = load_dataset_file_manifest(dataset_manifest_path)
    norm_stats = json.loads(norm_stats_path.read_text())
    validate_lingbot_calvin_norm_stats(norm_stats)
    norm_source = norm_stats["source"]
    if (
        norm_source["dataset_id"] != dataset_manifest.dataset_id
        or norm_source["dataset_revision"] != dataset_manifest.dataset_revision
        or dataset_manifest.split_name != dataset.name
    ):
        raise ValueError("CALVIN manifest, split and LingBot normalization identities differ")
    dataset_validation = validate_dataset_files(
        dataset_manifest,
        dataset,
        dataset_id=norm_source["dataset_id"],
        dataset_revision=norm_source["dataset_revision"],
        split_name=dataset.name,
        verify_hashes=True,
    )

    if _git_output(checkout, "rev-parse", "HEAD") != LINGBOT_SOURCE_COMMIT:
        raise ValueError("prepared unified checkout is not at the pinned commit")
    patch_paths = [root / item["path"] for item in config["host"]["patches"]]
    patch_states = [detect_patch_state(checkout, path) for path in patch_paths]
    if patch_states != ["applied", "applied"]:
        raise ValueError("prepared unified checkout does not contain both patches")
    expected_source_hashes = report["patch_replay"]["patched_source_sha256"]
    actual_source_hashes = {
        relative: _sha256(checkout / relative) for relative in expected_source_hashes
    }
    if actual_source_hashes != expected_source_hashes:
        raise ValueError("prepared unified source differs from the replayed patch bytes")

    actual_python = [sys.version_info.major, sys.version_info.minor]
    if actual_python != config["runtime"]["python_major_minor"]:
        raise ValueError(f"Python runtime differs: {actual_python}")
    versions = {package: importlib.metadata.version(package) for package in RUNTIME_VERSIONS}
    if versions != RUNTIME_VERSIONS:
        raise ValueError(f"installed runtime versions differ: {versions}")

    import_environment = os.environ.copy()
    previous_pythonpath = import_environment.get("PYTHONPATH")
    import_environment["PYTHONPATH"] = (
        str(checkout)
        if not previous_pythonpath
        else os.pathsep.join((str(checkout), previous_pythonpath))
    )
    import_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from lingbotvla.data.vla_data.utils import FeatureTransform; "
                "from lerobot.datasets.utils import load_nested_dataset; "
                "print(FeatureTransform.__name__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=import_environment,
    ).stdout.strip()
    if import_probe != "FeatureTransform":
        raise ValueError("patched LingBot/LeRobot runtime import probe returned unexpected output")

    import torch

    hardware = config["hardware"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != hardware["gpu_count"]:
        raise ValueError("CUDA device count differs from the two-A100 contract")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 2**30
        if hardware["gpu_name_contains"] not in properties.name:
            raise ValueError(f"GPU {index} is not an A100: {properties.name}")
        if memory_gib < hardware["minimum_gpu_memory_gib"]:
            raise ValueError(f"GPU {index} has insufficient memory: {memory_gib:.2f} GiB")
        devices.append({"index": index, "name": properties.name, "memory_gib": memory_gib})
    memory_gib = _system_memory_gib()
    if memory_gib < hardware["minimum_system_memory_gib"]:
        raise ValueError(f"system memory is insufficient: {memory_gib:.2f} GiB")
    free_storage_gib = shutil.disk_usage(run_root).free / 2**30
    if free_storage_gib < hardware["minimum_free_storage_gib"]:
        raise ValueError(f"persistent storage is insufficient: {free_storage_gib:.2f} GiB")
    report.update(
        {
            "mode": "runtime",
            "paths": {key: str(value) for key, value in resolved.items()},
            "checkpoint": checkpoint_report,
            "processor": processor_report,
            "patch_states": patch_states,
            "patched_source_sha256": actual_source_hashes,
            "runtime_versions": versions,
            "lingbot_data_import_probe": import_probe,
            "gpus": devices,
            "system_memory_gib": memory_gib,
            "free_storage_gib": free_storage_gib,
            "g0_image_sha256": _sha256(image),
            "dataset_manifest_sha256": _sha256(dataset_manifest_path),
            "dataset_tree_sha256": dataset_manifest.tree_sha256,
            "dataset_validation": dataset_validation,
            "lingbot_norm_stats_sha256": _sha256(norm_stats_path),
            "lingbot_norm_stats_artifact_sha256": norm_stats["artifact_sha256"],
        }
    )
    return report


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=root / CONFIG_RELATIVE_PATH)
    parser.add_argument("--source-checkout", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = json.loads(args.config.read_text())
    source_checkout = args.source_checkout
    if source_checkout is None:
        raw_source = os.environ.get(config["paths"]["source_checkout_env"])
        source_checkout = None if not raw_source else Path(raw_source)
    report = (
        static_preflight(config, root=args.root, source_checkout=source_checkout)
        if args.dry_run
        else runtime_preflight(config, root=args.root)
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _write_text_durable(args.output, payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
