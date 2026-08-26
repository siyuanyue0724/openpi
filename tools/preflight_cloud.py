#!/usr/bin/env python3
"""Fail-closed static and runtime preflight for frozen 2xA100 profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
for _path in (_ROOT, _SOURCE_ROOT):
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

MOLMO_HOST = {
    "name": "MolmoAct2",
    "source_commit": "c2282820f9b188b60e66ea1636b3efd81c45cbb4",
    "source_checkout": "references/source_checkouts/molmoact2-cloud",
    "training_source_commit": "80633827176a0203064cb141383664fba024e050",
    "training_source_checkout": "references/source_checkouts/molmoact2-lerobot-cloud",
    "training_adapter_patch": ("references/patches/molmoact2_lerobot_action_layer_adapter.patch"),
    "checkpoint_id": "allenai/MolmoAct2",
    "checkpoint_revision": "e432d85f6e039edca44afb93c262f3084ab72a9c",
    "checkpoint_subdir": "molmoact2",
}
MOLMO_RUNTIME = {
    "python_major_minor": [3, 12],
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "5.5.4",
    "datasets": "4.8.5",
    "huggingface-hub": "1.13.0",
    "peft": "0.19.1",
    "accelerate": "1.13.0",
    "lerobot": "0.5.2",
    "scipy": "1.17.1",
    "picf-next": "0.1.0",
    "uv_lock_sha256": "f79437aeed6ac8f6fd83ff1a250136df040ef5e10657df7e280e0f409c21d8a6",
}
LINGBOT_HOST = {
    "name": "LingBot-VLA2",
    "source_commit": "69729b4ef24c63ec25e750915491635f4753be1d",
    "source_checkout": "references/source_checkouts/lingbot-vla-v2",
    "adapter_patch": "references/patches/lingbot_vla2_action_layer_adapter.patch",
    "checkpoint_id": "robbyant/lingbot-vla-v2-6b",
    "checkpoint_revision": "11c703bf6a5c1f45b3b69168482da11fdbba53d7",
    "checkpoint_subdir": "lingbot-vla-v2-6b",
    "processor_id": "Qwen/Qwen3-VL-4B-Instruct",
    "processor_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
    "processor_subdir": "qwen3-vl-4b-processor-config",
}
LINGBOT_RUNTIME = {
    "python_major_minor": [3, 12],
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "4.57.3",
    "datasets": "4.1.1",
    "huggingface-hub": "0.34.3",
    "lerobot": "0.4.3",
    "lerobot_install_mode": "no-deps",
    "lingbot_requirements_sha256": (
        "4bea8eca2e5e81107332947fe38d9a2787bc6a8fe4d3f875fa7e3d028f48993d"
    ),
}
PROFILE_CONTRACTS = {
    "molmoact2-causal-2xa100-40g": {
        "host": MOLMO_HOST,
        "runtime": MOLMO_RUNTIME,
        "distributed": "accelerate_ddp",
        "initial_trainable_scope": "action_expert_picf_and_zero_initialized_adapters",
        "gate_prefix": "M",
        "recipe_stage": "M3_structural_probe",
        "training_contract_kind": "structural",
        "initial_gate_index": 0,
    },
    "molmoact2-m4-action-2xa100-40g": {
        "host": MOLMO_HOST,
        "runtime": MOLMO_RUNTIME,
        "distributed": "accelerate_ddp",
        "initial_trainable_scope": "action_expert_and_zero_initialized_adapters",
        "gate_prefix": "M",
        "recipe_stage": "M4_action_adoption",
        "training_contract_kind": "frozen_stationary_action",
        "initial_gate_index": 4,
    },
    "lingbot-vla2-2xa100-40g": {
        "host": LINGBOT_HOST,
        "runtime": LINGBOT_RUNTIME,
        "distributed": "fsdp2",
        "initial_trainable_scope": "picf_and_zero_initialized_adapters",
        "gate_prefix": "L",
        "recipe_stage": None,
        "training_contract_kind": "structural",
        "initial_gate_index": 0,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cloud/2xa100_40g_gates.json"),
    )
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _validate_training_recipe_contract(
    config: dict[str, Any],
    *,
    root: Path,
) -> dict[str, object]:
    from picf_next.training.recipe import load_training_recipe

    contract = config.get("training_recipe")
    if not isinstance(contract, dict) or set(contract) != {"path", "sha256"}:
        raise ValueError("Molmo cloud profile requires one exact training_recipe contract")
    relative = contract["path"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("training_recipe path must be one nonempty repository-relative path")
    recipe_path = (root / relative).resolve()
    try:
        recipe_path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("training_recipe path escapes the repository") from error
    if not recipe_path.is_file():
        raise ValueError("training_recipe file is absent")
    recipe = load_training_recipe(recipe_path)
    if contract["sha256"] != recipe.recipe_sha256:
        raise ValueError("training_recipe canonical SHA-256 changed")
    host = config["host"]
    if (
        recipe.host.name != host["name"]
        or recipe.host.checkpoint_id != host["checkpoint_id"]
        or recipe.host.checkpoint_revision != host["checkpoint_revision"]
        or recipe.host.source_commit != host["source_commit"]
        or recipe.host.trainer_commit != host["training_source_commit"]
    ):
        raise ValueError("training_recipe host identity differs from the cloud profile")
    profile_contract = PROFILE_CONTRACTS[config["profile"]]
    if recipe.authorization.stage != profile_contract["recipe_stage"]:
        raise ValueError("training_recipe stage differs from the cloud profile")
    contract = config["training_contract"]
    contract_kind = profile_contract["training_contract_kind"]
    if contract_kind == "structural":
        overshooting = contract["picf_objective_contract"]["geometry_overshooting"]
        if list(recipe.geometry_overshooting.horizons) != overshooting["initial_horizons"]:
            raise ValueError("training_recipe overshooting horizons differ from the cloud profile")
    elif contract_kind == "frozen_stationary_action":
        objective = recipe.objective_config
        if (
            objective.action_weight <= 0.0
            or objective.set_weight != 0.0
            or objective.dynamics_weight != 0.0
            or objective.binding_weight != 0.0
            or objective.require_temporal_positive_pairs
            or recipe.geometry_overshooting.config.weight != 0.0
        ):
            raise ValueError("M4 training_recipe must be action-only with frozen PICF losses")
    else:  # pragma: no cover - PROFILE_CONTRACTS is module-owned
        raise RuntimeError(f"unsupported training contract kind: {contract_kind}")
    mirrored = {
        "action_horizon": recipe.dataset.action_horizon,
        "activation_checkpointing": recipe.policy.gradient_checkpointing,
        "detached_context_frames": recipe.detached_context_frames,
        "explicit_flow_randomness_required": (recipe.optimizer.require_explicit_flow_randomness),
        "flow_timesteps": recipe.policy.num_flow_timesteps,
        "gradient_transitions": recipe.gradient_transitions,
        "precision": recipe.policy.model_dtype,
        "sample_plan_algorithm": recipe.dataset.sample_plan_algorithm,
    }
    for name, expected in mirrored.items():
        if contract.get(name) != expected:
            raise ValueError(f"cloud mirror {name} differs from the canonical training recipe")
    expected_scope = profile_contract["initial_trainable_scope"]
    if (
        not recipe.policy.train_action_expert_only
        or contract.get("initial_trainable_scope") != expected_scope
    ):
        raise ValueError("cloud trainable scope differs from the canonical training recipe")
    if recipe.authorization.long_training_authorized:
        raise ValueError("initial cloud profile cannot embed a long-training authorization")
    return recipe.local_preflight_report(root)


def validate_config(config: dict[str, Any], *, root: Path | None = None) -> None:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    if config.get("schema") != "picf-next.cloud-gates.v2":
        raise ValueError("unsupported cloud gate schema")
    profile = config.get("profile")
    if profile not in PROFILE_CONTRACTS:
        raise ValueError(f"unsupported frozen cloud profile: {profile}")
    profile_contract = PROFILE_CONTRACTS[profile]

    hardware = config.get("hardware", {})
    if hardware.get("gpu_count") != 2:
        raise ValueError("the frozen cloud profile requires exactly two GPUs")
    if hardware.get("gpu_name_contains") != "A100":
        raise ValueError("the frozen cloud profile requires A100 GPUs")
    if hardware.get("minimum_gpu_memory_gib", 0) < 39.0:
        raise ValueError("the cloud profile must enforce 40 GB-class GPUs")
    if hardware.get("minimum_system_memory_gib", 0) < 64.0:
        raise ValueError("the cloud profile must reserve enough host memory")
    if hardware.get("minimum_free_storage_gib", 0) < 200.0:
        raise ValueError("the cloud profile must reserve enough checkpoint/data storage")

    if config.get("host") != profile_contract["host"]:
        raise ValueError("the pinned host source/checkpoint contract changed")
    if config.get("runtime") != profile_contract["runtime"]:
        raise ValueError("the pinned host runtime contract changed")

    contract = config.get("training_contract", {})
    prefix = profile_contract["gate_prefix"]
    contract_kind = profile_contract["training_contract_kind"]
    state_factorization = (
        "unit_address_deterministic_content_diagonal_gaussian_geometry_bernoulli_lifecycle"
    )
    if contract_kind == "frozen_stationary_action":
        temporal_training_mode = "frozen_accepted_stationary_core_single_transition"
        objective_contract = {
            "families": ["action"],
            "state_factorization": state_factorization,
            "loss_target_resolution": "host_action_targets_only_no_structural_targets",
            "identity_credit": "accepted_stationary_posterior_frozen_by_hash",
            "geometry_overshooting": {"active": False, "weight": 0.0},
        }
    else:
        temporal_training_mode = "stateful_single_transition"
        objective_contract = {
            "families": ["action", "set", "dynamics", "binding"],
            "state_factorization": state_factorization,
            "loss_target_resolution": "post_forward_by_immutable_sample_key",
            "identity_credit": "checkpointed_loss_track_to_current_address",
            "geometry_overshooting": {
                "authorized_gate": f"{prefix}3_temporal_posterior",
                "family": "dynamics",
                "initial_horizons": [1, 2],
                "start_posterior_detached": True,
                "production_transition_shared": True,
                "future_image_encoding": False,
                "future_content_target": False,
                "lifecycle_overshooting": False,
                "per_axis_missingness": True,
                "episode_boundary_padding": "prefix_zero",
                "update_schedule": {
                    "algorithm": "every_optimizer_step.v1",
                    "fraction_numerator": 1,
                    "fraction_denominator": 1,
                    "importance_scale": 1.0,
                },
            },
        }
    expected = {
        "precision": "bfloat16",
        "distributed": profile_contract["distributed"],
        "activation_checkpointing": True,
        "initial_trainable_scope": profile_contract["initial_trainable_scope"],
        "full_host_adamw_on_2x40g_authorized": False,
        "temporal_training_mode": temporal_training_mode,
        "gradient_transitions": 1,
        "detached_context_frames": 0,
        "action_horizon": 10,
        "object_runtime_targets_forbidden": True,
        "native_dense_tokens_pruned": False,
        "picf_runtime_validation": "full_preflight_metadata_hotpath_full_checkpoint",
        "picf_objective_contract": objective_contract,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"training contract changed required field {key}")
    molmo_profile = profile in {
        "molmoact2-causal-2xa100-40g",
        "molmoact2-m4-action-2xa100-40g",
    }
    if molmo_profile and contract.get("flow_timesteps") != 8:
        raise ValueError("MolmoAct2 flow_timesteps must match the official LIBERO recipe")
    training_control: dict[str, object] = {}
    if molmo_profile:
        if contract.get("molmo_native_prepool_same_forward_local_contract_ready") is not True:
            raise ValueError("Molmo native pre-pool same-forward local contract is not ready")
        if contract.get("picf_dense_bank_contract") != "native_same_forward_visual_bank":
            raise ValueError("Molmo PICF dense-bank contract changed")
        training_control = {
            "sample_plan_algorithm": "sha256-epoch-sort.v1",
            "checkpoint_backend": "accelerate-1.13-save-state",
            "atomic_checkpoint_publication": True,
            "matched_abc_common_plan_required": True,
            "explicit_flow_randomness_required": True,
        }
        if contract_kind == "frozen_stationary_action":
            training_control.update(
                {
                    "stationary_temporal_initialization": (
                        "accepted_full_stage_b_checkpoint_and_evidence_package_hash_bound"
                    ),
                    "action_adoption_arms": [
                        "current_no_posterior",
                        "causal_video_no_posterior",
                        "current_posterior",
                        "causal_video_posterior",
                    ],
                }
            )
        else:
            training_control.update(
                {
                    "picf_current_frame_initialization": (
                        "accepted_axis_constant_m2_checkpoint_hash_bound"
                    ),
                    "m3_occlusion_arms": [
                        "current_no_posterior",
                        "causal_video_no_posterior",
                        "current_posterior",
                        "causal_video_posterior",
                    ],
                }
            )
        for key, value in training_control.items():
            if contract.get(key) != value:
                raise ValueError(f"Molmo training control changed required field {key}")
    if contract.get("experiment_arms") != ["vanilla", "full_evidence", "picf"]:
        raise ValueError("the A/B/C causal arms are not frozen")
    expected_contract_keys = set(expected) | {"experiment_arms"}
    if molmo_profile:
        expected_contract_keys |= {
            "flow_timesteps",
            "molmo_native_prepool_same_forward_local_contract_ready",
            "picf_dense_bank_contract",
            *training_control,
        }
    if set(contract) != expected_contract_keys:
        raise ValueError("training contract contains missing or unrecognized fields")

    gates = config.get("ordered_gates")
    if not isinstance(gates, list) or len(gates) != 7:
        raise ValueError("cloud gates must contain seven ordered stages")
    if [item[:2] for item in gates] != [f"{prefix}{index}" for index in range(7)]:
        raise ValueError(f"cloud gates must be present in {prefix}0-to-{prefix}6 order")
    initial_gate_index = profile_contract["initial_gate_index"]
    expected_authorization = [gates[initial_gate_index]]
    if config.get("initial_authorization") != expected_authorization:
        raise ValueError(f"profile entry authorization must be exactly {expected_authorization[0]}")

    if molmo_profile:
        _validate_training_recipe_contract(config, root=root)
    elif "training_recipe" in config:
        raise ValueError("LingBot ceiling profile cannot consume the Molmo training recipe")

    paths = config.get("paths", {})
    expected_path_keys = {"checkpoint_root_env", "dataset_root_env", "run_root_env"}
    if not isinstance(paths, dict) or set(paths) != expected_path_keys:
        raise ValueError("cloud path environment keys changed")
    optional_secrets = config.get("optional_secret_env", [])
    if not isinstance(optional_secrets, list):
        raise ValueError("optional_secret_env must be a list of names")
    env_names = list(paths.values()) + optional_secrets
    if not all(isinstance(name, str) and name and "=" not in name for name in env_names):
        raise ValueError("configuration may contain environment variable names, not secret values")
    expected_reports = [
        "environment.json",
        "artifact_hashes.json",
        "full_weight_parity.json",
        "cuda_memory_latency.json",
    ]
    if contract_kind == "frozen_stationary_action":
        expected_reports.append("stationary_temporal_acceptance.json")
    if config.get("required_reports") != expected_reports:
        raise ValueError("required cloud reports changed")
    expected_top_level = {
        "hardware",
        "host",
        "initial_authorization",
        "optional_secret_env",
        "ordered_gates",
        "paths",
        "profile",
        "required_reports",
        "runtime",
        "schema",
        "training_contract",
    }
    if molmo_profile:
        expected_top_level.add("training_recipe")
    if set(config) != expected_top_level:
        raise ValueError("cloud profile contains missing or unrecognized top-level fields")


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_molmo_import_origins(
    source_checkout: Path,
    trainer_checkout: Path,
) -> dict[str, str]:
    """Prove runtime imports resolve to the two frozen Molmo source trees."""

    expected = {
        "olmo.hf_model.modeling_molmoact2": (
            source_checkout / "experiments/olmo/hf_model/modeling_molmoact2.py"
        ).resolve(),
        "lerobot.policies.molmoact2.modeling_molmoact2": (
            trainer_checkout / "src/lerobot/policies/molmoact2/modeling_molmoact2.py"
        ).resolve(),
    }
    origins: dict[str, str] = {}
    for module_name, expected_path in expected.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ValueError) as error:
            raise RuntimeError(f"cannot resolve frozen runtime module {module_name}") from error
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve frozen runtime module {module_name}")
        actual_path = Path(spec.origin).resolve()
        if actual_path != expected_path:
            raise RuntimeError(
                f"runtime module {module_name} resolves to {actual_path}, expected {expected_path}"
            )
        origins[module_name] = str(actual_path)
    return origins


def _runtime_versions(config: dict[str, Any]) -> dict[str, object]:
    immutable_keys = {key for key in config["runtime"] if key.endswith("_sha256")}
    non_package_keys = immutable_keys | {"lerobot_install_mode"}
    versions: dict[str, object] = {
        "python_major_minor": [sys.version_info.major, sys.version_info.minor]
    }
    for package in config["runtime"]:
        if package == "python_major_minor" or package in non_package_keys:
            continue
        versions[package] = importlib.metadata.version(package)
    if "lerobot_install_mode" in config["runtime"]:
        versions["lerobot_install_mode"] = config["runtime"]["lerobot_install_mode"]
    return versions


def _validate_host_runtime(
    config: dict[str, Any], *, root: Path, resolved_paths: dict[str, str]
) -> dict[str, Any]:
    host = config["host"]
    checkpoint_root = Path(resolved_paths["checkpoint_root_env"])
    source_checkout = root / host["source_checkout"]
    if _git_head(source_checkout) != host["source_commit"]:
        raise RuntimeError(f"{host['name']} checkout does not match the frozen source commit")

    if host["name"] == "MolmoAct2":
        try:
            from tools.bootstrap_molmoact2 import validate_checkpoint
            from tools.verify_molmoact2_lerobot_patch import detect_patch_state
        except ModuleNotFoundError:
            from bootstrap_molmoact2 import validate_checkpoint
            from verify_molmoact2_lerobot_patch import detect_patch_state

        trainer_checkout = root / host["training_source_checkout"]
        trainer_head = _git_head(trainer_checkout)
        if trainer_head != host["training_source_commit"]:
            raise RuntimeError("MolmoAct2 LeRobot checkout differs from its parent gitlink")
        patch_state = detect_patch_state(
            trainer_checkout,
            root / host["training_adapter_patch"],
        )
        if patch_state != "applied":
            raise RuntimeError(
                "MolmoAct2 LeRobot checkout does not contain the exact PICF adapter patch"
            )
        lock_hash = _sha256(trainer_checkout / "uv.lock")
        if lock_hash != config["runtime"]["uv_lock_sha256"]:
            raise RuntimeError("MolmoAct2 LeRobot uv.lock differs from the frozen contract")
        import_origins = _validate_molmo_import_origins(source_checkout, trainer_checkout)
        return {
            "host_source_head": host["source_commit"],
            "training_source_head": trainer_head,
            "training_patch_state": patch_state,
            "runtime_lock_sha256": lock_hash,
            "runtime_import_origins": import_origins,
            "checkpoint": validate_checkpoint(
                checkpoint_root / host["checkpoint_subdir"],
                validate_weight_shards=False,
            ),
        }

    try:
        from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
        from tools.verify_lingbot_vla2_patch import detect_patch_state
    except ModuleNotFoundError:
        from bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
        from verify_lingbot_vla2_patch import detect_patch_state

    requirements_hash = _sha256(source_checkout / "requirements.txt")
    if requirements_hash != config["runtime"]["lingbot_requirements_sha256"]:
        raise RuntimeError("LingBot requirements.txt differs from the frozen source contract")
    patch_state = detect_patch_state(source_checkout, root / host["adapter_patch"])
    if patch_state != "applied":
        raise RuntimeError("LingBot checkout does not contain the exact PICF adapter patch")
    return {
        "host_source_head": host["source_commit"],
        "host_patch_state": patch_state,
        "runtime_lock_sha256": requirements_hash,
        "checkpoint": validate_checkpoint(checkpoint_root / host["checkpoint_subdir"]),
        "processor": validate_processor(checkpoint_root / host["processor_subdir"]),
    }


def runtime_preflight(config: dict[str, Any], *, root: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("runtime preflight requires the pinned Torch environment") from error

    runtime_versions = _runtime_versions(config)
    expected_runtime = {
        key: value for key, value in config["runtime"].items() if not key.endswith("_sha256")
    }
    if runtime_versions != expected_runtime:
        raise RuntimeError(
            f"runtime package versions differ: {runtime_versions} != {expected_runtime}"
        )

    hardware = config["hardware"]
    system_memory_gib = _system_memory_gib()
    if system_memory_gib < hardware["minimum_system_memory_gib"]:
        raise RuntimeError(f"host exposes only {system_memory_gib:.2f} GiB system memory")
    count = torch.cuda.device_count()
    if count != hardware["gpu_count"]:
        raise RuntimeError(f"expected {hardware['gpu_count']} CUDA devices, found {count}")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 2**30
        if hardware["gpu_name_contains"] not in properties.name:
            raise RuntimeError(f"CUDA device {index} is {properties.name}, not an A100")
        if memory_gib < hardware["minimum_gpu_memory_gib"]:
            raise RuntimeError(f"CUDA device {index} exposes only {memory_gib:.2f} GiB")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"CUDA device {index} does not report bfloat16 support")
        devices.append({"index": index, "name": properties.name, "memory_gib": memory_gib})

    resolved_paths: dict[str, str] = {}
    free_storage = None
    for key, env_name in config["paths"].items():
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"required environment variable {env_name} is unset")
        path = Path(value).expanduser().resolve()
        if key == "run_root_env":
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise RuntimeError(f"required directory does not exist: {path}")
        resolved_paths[key] = str(path)
        if key == "run_root_env":
            free_storage = shutil.disk_usage(path).free / 2**30
    if free_storage is None or free_storage < hardware["minimum_free_storage_gib"]:
        available = "unknown" if free_storage is None else f"{free_storage:.2f}"
        raise RuntimeError(f"run filesystem has only {available} GiB free")

    host_report = _validate_host_runtime(config, root=root, resolved_paths=resolved_paths)
    report = {
        "status": "PASS",
        "profile": config["profile"],
        "source_head": _git_head(root),
        **host_report,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "runtime_versions": runtime_versions,
        "cuda_runtime": torch.version.cuda,
        "system_memory_gib": system_memory_gib,
        "devices": devices,
        "paths": resolved_paths,
        "free_storage_gib": free_storage,
        "authorized_gates": config["initial_authorization"],
    }
    if config["host"]["name"] == "MolmoAct2":
        report["training_recipe"] = _validate_training_recipe_contract(config, root=root)
    return report


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text())
    validate_config(config, root=root)
    report: dict[str, Any] = {
        "status": "STATIC_PASS",
        "profile": config["profile"],
        "authorized_gates": config["initial_authorization"],
    }
    if config["host"]["name"] == "MolmoAct2":
        report["training_recipe"] = _validate_training_recipe_contract(config, root=root)
    if args.check_runtime:
        report = runtime_preflight(config, root=root)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
