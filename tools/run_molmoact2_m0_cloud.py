#!/usr/bin/env python3
"""Fail-closed 2xA100 launcher for the MolmoAct2 M0 acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
for _path in (_ROOT, _SOURCE_ROOT):
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_REPORTS = (
    "environment.json",
    "artifact_hashes.json",
    "full_weight_parity.json",
    "cuda_memory_latency.json",
)
_PROFILE_FOLLOWUP_GATES = {
    "molmoact2-causal-2xa100-40g": ("M1_typed_full_manifest",),
    "molmoact2-m4-action-2xa100-40g": ("M4_action_adoption",),
}
_M0_PROBE_DIMENSIONS = {
    "dense_token_count": 729,
}


def _followup_gates(config: dict[str, Any]) -> tuple[str, ...]:
    profile = config.get("profile")
    try:
        return _PROFILE_FOLLOWUP_GATES[profile]
    except (KeyError, TypeError) as error:
        raise ValueError("this launcher accepts only frozen MolmoAct2 cloud profiles") from error


def _validate_required_reports(config: dict[str, Any]) -> None:
    reports = config.get("required_reports")
    if (
        not isinstance(reports, list)
        or tuple(reports[: len(_REQUIRED_REPORTS)]) != _REQUIRED_REPORTS
    ):
        raise ValueError("the frozen M0 report contract changed")


def _recipe_probe_contract(
    config: dict[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, int], dict[str, object], Path, str]:
    """Resolve every PICF adapter dimension from the exact training recipe."""

    from picf_next.training.recipe import load_training_recipe

    contract = config.get("training_recipe")
    if not isinstance(contract, dict) or set(contract) != {"path", "sha256"}:
        raise ValueError("M0 requires one exact training_recipe contract")
    recipe_path = (root / str(contract["path"])).resolve()
    recipe = load_training_recipe(recipe_path)
    if recipe.recipe_sha256 != contract["sha256"]:
        raise ValueError("M0 training recipe SHA-256 differs from the cloud contract")
    dimensions = {
        **_M0_PROBE_DIMENSIONS,
        "state_dim": recipe.dataset.state_dim,
        "action_dim": recipe.host.action_dim,
        "dense_token_width": recipe.core_config.dense_token_dims[recipe.host.dense_modality],
        "object_count": recipe.core_config.posterior_capacity,
        "object_address_width": recipe.core_config.object_address_dim,
        "object_value_width": recipe.core_config.object_value_dim,
    }
    runtime = {
        "action_horizon": recipe.dataset.action_horizon,
        "control_mode": recipe.policy.control_mode,
        "dtype": recipe.policy.model_dtype,
        "num_steps": recipe.policy.num_inference_steps,
        "setup_type": recipe.policy.setup_type,
    }
    return dimensions, runtime, recipe_path, recipe.recipe_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/cloud/2xa100_40g_gates.json",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--task", default="move the red block to the left")
    parser.add_argument("--setup-type", default="single-arm tabletop manipulation")
    parser.add_argument("--control-mode", default="normalized relative end-effector pose")
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_root(argument: Path | None, environment_name: str) -> Path:
    value = argument if argument is not None else os.environ.get(environment_name)
    if value is None:
        raise RuntimeError(f"{environment_name} is unset and no command-line override was supplied")
    return Path(value).expanduser().resolve()


def _absolute_executable(path: Path) -> Path:
    """Keep a virtualenv entry point instead of resolving its interpreter symlink."""

    return Path(os.path.abspath(path.expanduser()))


def _is_under_mnt(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == Path("/mnt") or Path("/mnt") in resolved.parents


def _run_id(value: str | None) -> str:
    resolved = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not _RUN_ID.fullmatch(resolved):
        raise ValueError(f"invalid M0 run id: {resolved!r}")
    return resolved


def _runner_command(
    *,
    python: Path,
    root: Path,
    config: dict[str, Any],
    checkpoint_root: Path,
    image: Path,
    output: Path,
    task: str,
    setup_type: str,
    control_mode: str,
) -> list[str]:
    host = config["host"]
    dimensions, runtime, recipe_path, recipe_sha256 = _recipe_probe_contract(config, root=root)
    if setup_type != runtime["setup_type"] or control_mode != runtime["control_mode"]:
        raise ValueError("M0 prompt semantics must match the canonical training recipe")
    return [
        str(python),
        str(root / "tools/smoke_molmoact2_lerobot_full_weight.py"),
        "--source-checkout",
        str(root / host["source_checkout"]),
        "--lerobot-checkout",
        str(root / host["training_source_checkout"]),
        "--patch",
        str(root / host["training_adapter_patch"]),
        "--checkpoint-dir",
        str(checkpoint_root / host["checkpoint_subdir"]),
        "--image",
        str(image),
        "--output",
        str(output),
        "--task",
        task,
        "--setup-type",
        setup_type,
        "--control-mode",
        control_mode,
        "--training-recipe",
        str(recipe_path),
        "--training-recipe-sha256",
        recipe_sha256,
        "--action-horizon",
        str(runtime["action_horizon"]),
        "--num-steps",
        str(runtime["num_steps"]),
        "--state-dim",
        str(dimensions["state_dim"]),
        "--action-dim",
        str(dimensions["action_dim"]),
        "--dense-token-count",
        str(dimensions["dense_token_count"]),
        "--dense-token-width",
        str(dimensions["dense_token_width"]),
        "--object-count",
        str(dimensions["object_count"]),
        "--object-address-width",
        str(dimensions["object_address_width"]),
        "--object-value-width",
        str(dimensions["object_value_width"]),
        "--device",
        "cuda:0",
        "--dtype",
        str(runtime["dtype"]),
    ]


def validate_m0_report(
    report: dict[str, Any],
    *,
    config: dict[str, Any],
    root: Path = _ROOT,
) -> None:
    dimensions, runtime, _recipe_path, recipe_sha256 = _recipe_probe_contract(config, root=root)
    if report.get("schema") != "picf-next.molmoact2-lerobot-m0.v3":
        raise ValueError("M0 runner emitted an unsupported report schema")
    if report.get("status") != "PASS" or report.get("gate") != "M0_full_weight_parity":
        raise ValueError("M0 runner did not pass the authorized gate")
    semantics = report.get("semantics", {})
    expected_semantics = {
        "observation_path": "official_molmoact2_lerobot_processor_and_policy",
        "evidence_path": "native_molmo_prepool_patches_plus_synthetic_object_pressure",
        "dense_evidence_is_native_prepool_representation": True,
        "object_evidence_is_synthetic": True,
        "targets_or_masks_in_runtime_input": False,
        "native_molmo_729_same_forward_claimed": True,
        "official_baselines_precede_adapter_registration": True,
    }
    if semantics != expected_semantics:
        raise ValueError("M0 evidence/observation semantics changed")
    parity = report.get("zero_gate_contract", {})
    if parity.get("bitwise_equal") is not True or parity.get("max_abs_error") != 0.0:
        raise ValueError("M0 fixed-noise action parity is not exact")
    if (
        parity.get("official_vs_prepared_max_abs_error") != 0.0
        or parity.get("prepared_vs_zero_gate_max_abs_error") != 0.0
    ):
        raise ValueError("M0 individual action parity comparisons are not exact")
    action_shapes = {
        tuple(parity.get(name, ()))
        for name in (
            "official_action_shape",
            "prepared_action_shape",
            "zero_gate_action_shape",
        )
    }
    action_hashes = {
        parity.get(name)
        for name in (
            "official_action_sha256",
            "prepared_action_sha256",
            "zero_gate_action_sha256",
        )
    }
    if len(action_shapes) != 1 or () in action_shapes:
        raise ValueError("M0 action shapes differ across raw/prepared/zero-gate paths")
    if len(action_hashes) != 1 or None in action_hashes:
        raise ValueError("M0 action hashes differ across raw/prepared/zero-gate paths")
    if parity.get("dense_gate_count") != 36 or parity.get("object_gate_count") != 36:
        raise ValueError("M0 typed residual gate count differs from the released host")
    if parity.get("dense_gate_nonzero") != 0 or parity.get("object_gate_nonzero") != 0:
        raise ValueError("M0 residual gates are not exactly zero")
    for name in (
        "dense_gate_sha256",
        "object_gate_sha256",
        "official_action_sha256",
        "prepared_action_sha256",
        "zero_gate_action_sha256",
    ):
        if not _SHA256.fullmatch(str(parity.get(name, ""))):
            raise ValueError(f"M0 {name} is not a SHA-256 digest")

    action_horizon = report.get("action_horizon")
    action_dim = report.get("action_dim")
    if not isinstance(action_horizon, int) or not isinstance(action_dim, int):
        raise ValueError("M0 action dimensions are absent from the report")
    expected_shape = (1, action_horizon, action_dim)
    if expected_shape != (
        1,
        runtime["action_horizon"],
        dimensions["action_dim"],
    ) or action_shapes != {expected_shape}:
        raise ValueError("M0 action shape differs from the frozen probe contract")
    if report.get("state_dim") != dimensions["state_dim"]:
        raise ValueError("M0 state width differs from the frozen probe contract")
    if report.get("num_steps") != runtime["num_steps"]:
        raise ValueError("M0 flow step count differs from the frozen probe contract")
    if report.get("training_recipe_sha256") != recipe_sha256:
        raise ValueError("M0 report is not bound to the exact training recipe")

    prepared_inputs = report.get("prepared_input_tensors", {})
    if not isinstance(prepared_inputs, dict) or "inputs_embeds" not in prepared_inputs:
        raise ValueError("M0 prepared path omitted inputs_embeds")
    forbidden_prepared = {
        "input_ids",
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
        "pixel_values_videos",
        "video_token_pooling",
        "video_grids",
    }
    if forbidden_prepared & set(prepared_inputs):
        raise ValueError("M0 prepared path retained raw visual inputs")
    for name, item in prepared_inputs.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("shape"), list)
            or not item["shape"]
            or any(not isinstance(value, int) or value <= 0 for value in item["shape"])
            or not isinstance(item.get("dtype"), str)
            or not _SHA256.fullmatch(str(item.get("sha256", "")))
        ):
            raise ValueError(f"M0 prepared tensor manifest is invalid for {name}")
    evidence = report.get("evidence_contract", {})
    embed_shape = prepared_inputs["inputs_embeds"]["shape"]
    if len(embed_shape) != 3 or embed_shape[0] != 1 or embed_shape[-1] != 2560:
        raise ValueError("M0 prepared inputs_embeds shape differs from MolmoAct2")
    if evidence.get("native_input_embedding_width") != 2560:
        raise ValueError("M0 native input embedding width differs from the released host")
    action_condition = report.get("prepared_action_condition_input_ids")
    if (
        not isinstance(action_condition, dict)
        or action_condition.get("shape") != embed_shape[:2]
        or action_condition.get("dtype") != "torch.int64"
        or not _SHA256.fullmatch(str(action_condition.get("sha256", "")))
    ):
        raise ValueError("M0 prepared path omitted exact action condition token identities")
    if evidence.get("dense_valid_count") != evidence.get("dense_token_count"):
        raise ValueError("M0 silently pruned dense evidence")
    if evidence.get("dense_token_count") != 729 or evidence.get("dense_token_width") != 2304:
        raise ValueError("M0 full-token pressure contract changed")
    if evidence.get("prepared_visual_vision_encoder_calls") != 1:
        raise ValueError("M0 did not use exactly one native vision-encoder call")
    expected_evidence = {
        "modality": "molmo_vision_patch",
        "object_count": dimensions["object_count"],
        "object_address_width": dimensions["object_address_width"],
        "object_value_width": dimensions["object_value_width"],
        "dense_context_layers": 36,
        "object_context_layers": 36,
    }
    for key, value in expected_evidence.items():
        if evidence.get(key) != value:
            raise ValueError(f"M0 evidence contract changed required field {key}")
    shard_hashes = report.get("checkpoint_weight_shard_sha256")
    expected_shards = {f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)}
    if (
        not isinstance(shard_hashes, dict)
        or set(shard_hashes) != expected_shards
        or any(not _SHA256.fullmatch(str(value)) for value in shard_hashes.values())
    ):
        raise ValueError("M0 omitted full checkpoint weight-shard hashes")


def _split_reports(raw: dict[str, Any], run_dir: Path, root: Path, config_path: Path) -> None:
    artifact_hashes = {
        "schema": "picf-next.m0-artifacts.v1",
        "training_recipe_sha256": raw["training_recipe_sha256"],
        "assets": raw["assets"],
        "checkpoint_weight_shard_sha256": raw["checkpoint_weight_shard_sha256"],
        "input_image": raw["image"],
        "input_tensor_sha256": {key: item["sha256"] for key, item in raw["input_tensors"].items()},
        "prepared_input_tensor_sha256": {
            key: item["sha256"] for key, item in raw["prepared_input_tensors"].items()
        },
        "prepared_action_condition_input_ids": raw["prepared_action_condition_input_ids"],
        "runner_sha256": _sha256(root / "tools/smoke_molmoact2_lerobot_full_weight.py"),
        "launcher_sha256": _sha256(root / "tools/run_molmoact2_m0_cloud.py"),
        "cloud_config_sha256": _sha256(config_path),
        "raw_report_sha256": _sha256(run_dir / "m0_raw_report.json"),
    }
    parity = {
        "schema": "picf-next.m0-parity.v3",
        "training_recipe_sha256": raw["training_recipe_sha256"],
        "status": raw["status"],
        "gate": raw["gate"],
        "semantics": raw["semantics"],
        "task": raw["task"],
        "seed": raw["seed"],
        "num_steps": raw["num_steps"],
        "prepared_input_tensors": raw["prepared_input_tensors"],
        "prepared_action_condition_input_ids": raw["prepared_action_condition_input_ids"],
        "evidence_contract": raw["evidence_contract"],
        "zero_gate_contract": raw["zero_gate_contract"],
    }
    resources = {
        "schema": "picf-next.m0-cuda-memory-latency.v1",
        "device": raw["device"],
        "device_name": raw["device_name"],
        "dtype": raw["dtype"],
        "timings_s": raw["timings_s"],
        "cuda_memory_bytes": raw["cuda_memory_bytes"],
    }
    _write_json_atomic(run_dir / "artifact_hashes.json", artifact_hashes)
    _write_json_atomic(run_dir / "full_weight_parity.json", parity)
    _write_json_atomic(run_dir / "cuda_memory_latency.json", resources)


def main() -> None:
    args = _parse_args()
    root = _ROOT
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    from tools.preflight_cloud import validate_config

    validate_config(config)
    followup_gates = _followup_gates(config)
    _validate_required_reports(config)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    checkpoint_root = _resolve_root(args.checkpoint_root, config["paths"]["checkpoint_root_env"])
    dataset_root = _resolve_root(args.dataset_root, config["paths"]["dataset_root_env"])
    run_root = _resolve_root(args.run_root, config["paths"]["run_root_env"])
    run_id = _run_id(args.run_id)
    run_dir = run_root / "molmoact2" / "M0_full_weight_parity" / run_id
    copied_image = run_dir / "inputs" / args.image.name
    raw_report = run_dir / "m0_raw_report.json"
    command = _runner_command(
        python=_absolute_executable(args.python),
        root=root,
        config=config,
        checkpoint_root=checkpoint_root,
        image=copied_image,
        output=raw_report,
        task=args.task,
        setup_type=args.setup_type,
        control_mode=args.control_mode,
    )
    manifest = {
        "schema": "picf-next.m0-launch.v1",
        "profile": config["profile"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "python": str(_absolute_executable(args.python)),
        "checkpoint_root": str(checkpoint_root),
        "dataset_root": str(dataset_root),
        "run_root": str(run_root),
        "source_image": str(args.image.resolve()),
        "source_image_sha256": _sha256(args.image),
        "command": command,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    persistent_paths = {
        "checkpoint_root": checkpoint_root,
        "dataset_root": dataset_root,
        "run_root": run_root,
    }
    nonpersistent = {
        name: str(path) for name, path in persistent_paths.items() if not _is_under_mnt(path)
    }
    if nonpersistent:
        raise RuntimeError(f"cloud M0 assets must use persistent /mnt storage: {nonpersistent}")
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing M0 run: {run_dir}")
    copied_image.parent.mkdir(parents=True)
    shutil.copy2(args.image, copied_image)
    if _sha256(copied_image) != manifest["source_image_sha256"]:
        raise RuntimeError("copied M0 input image hash differs from its source")
    _write_json_atomic(run_dir / "launch_manifest.json", manifest)

    environment = os.environ.copy()
    environment[config["paths"]["checkpoint_root_env"]] = str(checkpoint_root)
    environment[config["paths"]["dataset_root_env"]] = str(dataset_root)
    environment[config["paths"]["run_root_env"]] = str(run_root)
    python_paths = [
        str(root),
        str(root / "src"),
        str(root / config["host"]["source_checkout"] / "experiments"),
        str(root / config["host"]["training_source_checkout"] / "src"),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    preflight_command = [
        str(_absolute_executable(args.python)),
        str(root / "tools/preflight_cloud.py"),
        "--config",
        str(config_path),
        "--check-runtime",
        "--json-out",
        str(run_dir / "environment.json"),
    ]
    stages = (("preflight", preflight_command), ("m0_runner", command))
    for stage, stage_command in stages:
        with (
            (run_dir / f"{stage}.stdout.log").open("w") as stdout,
            (run_dir / f"{stage}.stderr.log").open("w") as stderr,
        ):
            result = subprocess.run(
                stage_command,
                cwd=root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        if result.returncode:
            decision = {
                "schema": "picf-next.m0-gate-decision.v1",
                "status": "FAIL",
                "gate": "M0_full_weight_parity",
                "failed_stage": stage,
                "returncode": result.returncode,
                "later_gates_authorized": [],
                "profile": config["profile"],
            }
            _write_json_atomic(run_dir / "gate_decision.json", decision)
            raise SystemExit(result.returncode)

    report = json.loads(raw_report.read_text())
    try:
        validate_m0_report(report, config=config, root=root)
        _split_reports(report, run_dir, root, config_path)
        report_hashes = {name: _sha256(run_dir / name) for name in _REQUIRED_REPORTS}
    except Exception as error:
        _write_json_atomic(
            run_dir / "gate_decision.json",
            {
                "schema": "picf-next.m0-gate-decision.v1",
                "status": "FAIL",
                "gate": "M0_full_weight_parity",
                "failed_stage": "report_validation",
                "error": str(error),
                "later_gates_authorized": [],
                "profile": config["profile"],
            },
        )
        raise
    decision = {
        "schema": "picf-next.m0-gate-decision.v1",
        "status": "PASS",
        "gate": "M0_full_weight_parity",
        "profile": config["profile"],
        "required_report_sha256": report_hashes,
        "later_gates_authorized": list(followup_gates),
    }
    _write_json_atomic(run_dir / "gate_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
