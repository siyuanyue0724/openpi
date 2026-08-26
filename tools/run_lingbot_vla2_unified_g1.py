#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the two-process LingBot unified-PICF G1 train/restart gate.

The tool deliberately has two phases.  ``fresh`` performs one real FSDP2
optimizer update and saves the official distributed checkpoint together with a
rank-local PICF session snapshot.  ``resume`` starts in a new ``torchrun``
process, restores both states, advances the next frozen CALVIN transition and
saves step two.  Accelerator and LingBot imports remain inside ``main`` so the
command contract is locally testable without the 6B checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_REVISION,
        LINGBOT_SOURCE_COMMIT,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from tools.smoke_lingbot_vla2_full_weight import (
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
    )
    from tools.verify_lingbot_vla2_patch import detect_patch_state
    from tools.verify_lingbot_vla2_unified_patch import (
        DATA_PATCH_RELATIVE_PATH,
        GRAPH_PATCH_RELATIVE_PATH,
        verify_unified_patches,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_CHECKPOINT_REVISION,
        LINGBOT_SOURCE_COMMIT,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from smoke_lingbot_vla2_full_weight import (  # type: ignore[no-redef]
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
    )
    from verify_lingbot_vla2_patch import detect_patch_state  # type: ignore[no-redef]
    from verify_lingbot_vla2_unified_patch import (  # type: ignore[no-redef]
        DATA_PATCH_RELATIVE_PATH,
        GRAPH_PATCH_RELATIVE_PATH,
        verify_unified_patches,
    )


G1_WORLD_SIZE = 2
G1_COMPARISON_ID = "lingbot-vla2-unified-g1"
G1_EXTRA_STATE_SCHEMA = "picf-next.lingbot-vla2-unified-g1-extra-state.v1"
G1_REPORT_SCHEMA = "picf-next.lingbot-vla2-unified-g1.v2"
_G1_IMPLEMENTATION_FILES = (
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
)
_EXTRA_STATE_KEYS = frozenset(
    {
        "global_step",
        "model_local_state_sha256",
        "model_family_digest",
        "next_optimizer_step",
        "optimizer_local_moment_elements",
        "optimizer_local_state_sha256",
        "optimizer_state_entries",
        "picf_published_optimizer_step",
        "picf_session_snapshot",
        "picf_session_snapshot_sha256",
        "plan_sha256",
        "rank",
        "rank_rng_state",
        "rank_rng_state_sha256",
        "schema",
        "source_digest",
        "world_size",
    }
)
_RANK_RNG_STATE_KEYS = frozenset(
    {
        "numpy_json",
        "python_json",
        "schema",
        "torch_cpu",
        "torch_cuda",
    }
)


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _write_text_durable(path: Path, payload: str) -> None:
    """Atomically publish one small report and durably link its directory entry."""

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


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = _environment_path("PICF_LINGBOT_SOURCE") or (
        root / "references/source_checkouts/lingbot-vla-v2-unified"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fresh", "resume"), required=True)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=source_default,
    )
    parser.add_argument("--data-patch", type=Path, default=root / DATA_PATCH_RELATIVE_PATH)
    parser.add_argument("--graph-patch", type=Path, default=root / GRAPH_PATCH_RELATIVE_PATH)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=source_default / "configs/vla/robotwin/robotwin.yaml",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=root / "configs/lingbot/calvin_robot.yaml",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=root / "configs/lingbot/calvin_data.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_environment_path("PICF_CHECKPOINT_DIR"),
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=_environment_path("PICF_PROCESSOR_DIR"),
    )
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
        "--run-dir",
        type=Path,
        default=_environment_path("PICF_RUN_DIR"),
    )
    parser.add_argument("--load-global-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--content-width", type=int, default=256)
    parser.add_argument("--geometry-width", type=int, default=6)
    parser.add_argument("--uncertainty-width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def _validate_paths_and_args(args: argparse.Namespace) -> None:
    if args.phase not in {"fresh", "resume"}:
        raise ValueError("G1 phase must be fresh or resume")
    required = {
        "checkpoint-dir": args.checkpoint_dir,
        "processor-dir": args.processor_dir,
        "dataset-split": args.dataset_split,
        "dataset-manifest": args.dataset_manifest,
        "norm-stats": args.norm_stats,
        "run-dir": args.run_dir,
    }
    absent = sorted(name for name, value in required.items() if value is None)
    if absent:
        raise ValueError(f"G1 paths are absent: {absent}")
    files = (
        args.data_patch,
        args.graph_patch,
        args.training_config,
        args.robot_config,
        args.data_config,
        args.dataset_manifest,
        args.norm_stats,
    )
    if any(not Path(path).is_file() for path in files):
        raise FileNotFoundError("one or more G1 source/config/data files are absent")
    directories = (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
    )
    if any(not Path(path).is_dir() for path in directories):
        raise FileNotFoundError("one or more G1 source/model/dataset directories are absent")
    dimensions = (
        args.capacity,
        args.content_width,
        args.geometry_width,
        args.uncertainty_width,
    )
    if any(type(value) is not int for value in dimensions):
        raise TypeError("G1 graph dimensions must be Python integers")
    if min(dimensions) <= 0:
        raise ValueError("G1 graph dimensions must be positive")
    if args.geometry_width != 6:
        raise ValueError("the G1 camera-frame geometry schema is exactly six-dimensional")
    if type(args.seed) is not int or type(args.load_global_step) is not int:
        raise TypeError("G1 seed and load-global-step must be Python integers")
    if args.seed < 0 or args.load_global_step <= 0:
        raise ValueError("G1 seed and load-global-step must be positive/non-negative")
    if args.seed > 0xFFFFFFFF - (G1_WORLD_SIZE - 1):
        raise ValueError("G1 seed plus rank must fit NumPy's uint32 seed domain")
    if args.phase == "resume" and args.load_global_step != 1:
        raise ValueError("G1 resume must cold-start from the fresh phase at global step one")
    if (
        isinstance(args.learning_rate, bool)
        or isinstance(args.max_grad_norm, bool)
        or not isinstance(args.learning_rate, (int, float))
        or not isinstance(args.max_grad_norm, (int, float))
        or not math.isfinite(args.learning_rate)
        or not math.isfinite(args.max_grad_norm)
        or args.learning_rate <= 0
        or args.max_grad_norm <= 0
    ):
        raise ValueError("G1 optimizer controls must be finite and positive")


def _validate_fsdp2_parameter_storage(model: Any, torch_module: Any) -> dict[str, object]:
    """Fail closed unless FSDP2 exposes FP32 CPU-offloaded DTensor masters."""

    parameter_tensors = 0
    local_elements = 0
    for name, parameter in model.named_parameters():
        to_local = getattr(parameter, "to_local", None)
        if not callable(to_local):
            raise RuntimeError(f"FSDP2 did not shard parameter as DTensor: {name}")
        if parameter.dtype != torch_module.float32:
            raise RuntimeError(f"FSDP2 master parameter is not FP32: {name}")
        local: Any = to_local()
        if local.device.type != "cpu":
            raise RuntimeError(f"FSDP2 parameter shard is not CPU-offloaded: {name}")
        parameter_tensors += 1
        local_elements += int(local.numel())
    if parameter_tensors == 0 or local_elements == 0:
        raise RuntimeError("FSDP2 parameter storage contract found no local parameters")
    return {
        "parameter_tensors": parameter_tensors,
        "local_elements": local_elements,
        "master_dtype": "float32",
        "local_device": "cpu",
    }


def _model_family_digest(
    *,
    graph_contract_digest: str,
    plan_sha256: str,
    execution_contract_sha256: str,
) -> str:
    payload = {
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "execution_contract_sha256": execution_contract_sha256,
        "graph_contract_digest": graph_contract_digest,
        "plan_sha256": plan_sha256,
        "source_commit": LINGBOT_SOURCE_COMMIT,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _g1_implementation_digest(root: Path) -> str:
    """Hash the local dependency closure executed by the G1 process."""

    root = root.resolve()
    relative_paths: set[str] = set(_G1_IMPLEMENTATION_FILES)
    relative_paths.update(
        str(path.relative_to(root)) for path in (root / "src/picf_next/unified").glob("*.py")
    )
    missing = sorted(relative for relative in relative_paths if not (root / relative).is_file())
    if missing:
        raise FileNotFoundError(f"G1 implementation dependency files are absent: {missing}")
    payload = {relative: _sha256(root / relative) for relative in sorted(relative_paths)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _g1_execution_contract_digest(
    *,
    root: Path,
    args: argparse.Namespace,
    patched_source_sha256: Mapping[str, str],
) -> tuple[str, str]:
    """Bind every deterministic source/config choice shared by fresh and resume."""

    implementation_sha256 = _g1_implementation_digest(root)
    input_files = {
        "data_patch": args.data_patch,
        "graph_patch": args.graph_patch,
        "training_config": args.training_config,
        "robot_config": args.robot_config,
        "data_config": args.data_config,
        "dataset_manifest": args.dataset_manifest,
        "norm_stats": args.norm_stats,
    }
    payload = {
        "schema": "picf-next.lingbot-vla2-unified-g1-execution.v1",
        "implementation_sha256": implementation_sha256,
        "input_file_sha256": {
            name: _sha256(Path(path)) for name, path in sorted(input_files.items())
        },
        "patched_source_sha256": dict(sorted(patched_source_sha256.items())),
        "optimizer": {
            "algorithm": "torch.optim.AdamW",
            "learning_rate": float(args.learning_rate).hex(),
            "master_parameter_dtype": "float32",
            "max_grad_norm": float(args.max_grad_norm).hex(),
        },
        "sampling": {
            "comparison_id": G1_COMPARISON_ID,
            "global_batch_size": G1_WORLD_SIZE,
            "seed": args.seed,
            "total_steps": 2,
        },
        "topology": {
            "cpu_offload": True,
            "data_parallel_mode": "fsdp2",
            "full_shard": True,
            "gradient_accumulation_steps": 1,
            "world_size": G1_WORLD_SIZE,
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, implementation_sha256


def _validate_resume_extra_state(
    extra_state: Any,
    *,
    expected_global_step: int,
    expected_model_family_digest: str,
    expected_plan_sha256: str,
    expected_rank: int,
    expected_source_digest: str,
    expected_world_size: int,
) -> dict[str, Any]:
    if not isinstance(extra_state, dict) or set(extra_state) != _EXTRA_STATE_KEYS:
        raise ValueError("G1 checkpoint extra state is incomplete")
    if extra_state["schema"] != G1_EXTRA_STATE_SCHEMA:
        raise ValueError("G1 checkpoint extra-state schema differs")
    for key in (
        "global_step",
        "next_optimizer_step",
        "optimizer_local_moment_elements",
        "optimizer_state_entries",
        "picf_published_optimizer_step",
        "rank",
        "world_size",
    ):
        value = extra_state[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"G1 checkpoint {key} is not an integer")
    if extra_state["global_step"] != expected_global_step:
        raise ValueError("G1 checkpoint global step differs from the requested restart")
    if extra_state["next_optimizer_step"] != expected_global_step:
        raise ValueError("G1 checkpoint optimizer/source step is inconsistent")
    if extra_state["picf_published_optimizer_step"] != expected_global_step - 1:
        raise ValueError("G1 checkpoint posterior publication step is inconsistent")
    if extra_state["model_family_digest"] != expected_model_family_digest:
        raise ValueError("G1 checkpoint model-family contract differs")
    if extra_state["plan_sha256"] != expected_plan_sha256:
        raise ValueError("G1 checkpoint frozen stream plan differs")
    if extra_state["rank"] != expected_rank or extra_state["world_size"] != expected_world_size:
        raise ValueError("G1 checkpoint rank topology differs")
    if extra_state["optimizer_state_entries"] <= 0:
        raise ValueError("G1 checkpoint has no optimizer state entries")
    if extra_state["optimizer_local_moment_elements"] <= 0:
        raise ValueError("G1 checkpoint has no optimizer moment elements")
    snapshot = extra_state["picf_session_snapshot"]
    if not isinstance(snapshot, bytes) or not snapshot:
        raise ValueError("G1 checkpoint has no rank-local PICF session snapshot")
    for name in (
        "model_local_state_sha256",
        "optimizer_local_state_sha256",
        "picf_session_snapshot_sha256",
        "rank_rng_state_sha256",
    ):
        _validate_sha256(extra_state[name], name=name)
    if hashlib.sha256(snapshot).hexdigest() != extra_state["picf_session_snapshot_sha256"]:
        raise ValueError("G1 checkpoint PICF session snapshot digest differs")
    _validate_rank_rng_state(extra_state["rank_rng_state"], require_cuda=True)
    if (
        _rank_rng_state_digest(extra_state["rank_rng_state"], require_cuda=True)
        != extra_state["rank_rng_state_sha256"]
    ):
        raise ValueError("G1 checkpoint rank RNG state digest differs")
    source_digest = extra_state["source_digest"]
    if source_digest != expected_source_digest:
        raise ValueError("G1 checkpoint source digest differs from the frozen prior batch")
    return extra_state


def _validate_rank_rng_state(state: Any, *, require_cuda: bool) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != _RANK_RNG_STATE_KEYS:
        raise ValueError("G1 rank RNG state is incomplete")
    if state["schema"] != "picf-next.rank-rng-state.v1":
        raise ValueError("G1 rank RNG state schema changed")
    for name in ("python_json", "numpy_json", "torch_cpu"):
        if not isinstance(state[name], bytes) or not state[name]:
            raise ValueError(f"G1 rank RNG state {name} must be nonempty bytes")
    cuda_state = state["torch_cuda"]
    if not isinstance(cuda_state, bytes) or (require_cuda and not cuda_state):
        raise ValueError("G1 rank RNG CUDA state is absent")
    try:
        python_payload = json.loads(state["python_json"].decode("ascii"))
        numpy_payload = json.loads(state["numpy_json"].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("G1 rank RNG JSON is invalid") from error
    if (
        not isinstance(python_payload, list)
        or len(python_payload) != 3
        or not isinstance(python_payload[0], int)
        or not isinstance(python_payload[1], list)
        or not python_payload[1]
    ):
        raise ValueError("G1 Python RNG payload is invalid")
    if (
        not isinstance(numpy_payload, dict)
        or set(numpy_payload) != {"cached_gaussian", "has_gauss", "keys", "name", "position"}
        or not isinstance(numpy_payload["name"], str)
        or not isinstance(numpy_payload["keys"], list)
        or not numpy_payload["keys"]
        or type(numpy_payload["position"]) is not int
        or type(numpy_payload["has_gauss"]) is not int
    ):
        raise ValueError("G1 NumPy RNG payload is invalid")
    return state


def _capture_rank_rng_state(
    torch_module: Any,
    numpy_module: Any,
    *,
    device: Any | None,
) -> dict[str, Any]:
    """Serialize every process-global RNG that may affect a training update."""

    import random

    python_state = random.getstate()
    numpy_state = numpy_module.random.get_state()
    state = {
        "schema": "picf-next.rank-rng-state.v1",
        "python_json": json.dumps(
            [python_state[0], list(python_state[1]), python_state[2]],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii"),
        "numpy_json": json.dumps(
            {
                "cached_gaussian": float(numpy_state[4]),
                "has_gauss": int(numpy_state[3]),
                "keys": numpy_state[1].tolist(),
                "name": str(numpy_state[0]),
                "position": int(numpy_state[2]),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii"),
        "torch_cpu": bytes(torch_module.get_rng_state().tolist()),
        "torch_cuda": (
            b"" if device is None else bytes(torch_module.cuda.get_rng_state(device).cpu().tolist())
        ),
    }
    return _validate_rank_rng_state(state, require_cuda=device is not None)


def _restore_rank_rng_state(
    state: Any,
    torch_module: Any,
    numpy_module: Any,
    *,
    device: Any | None,
) -> None:
    """Restore the exact per-rank stochastic continuation point."""

    import random

    validated = _validate_rank_rng_state(state, require_cuda=device is not None)
    python_payload = json.loads(validated["python_json"].decode("ascii"))
    random.setstate(
        (
            int(python_payload[0]),
            tuple(int(value) for value in python_payload[1]),
            python_payload[2],
        )
    )
    numpy_payload = json.loads(validated["numpy_json"].decode("ascii"))
    numpy_module.random.set_state(
        (
            numpy_payload["name"],
            numpy_module.asarray(numpy_payload["keys"], dtype=numpy_module.uint32),
            int(numpy_payload["position"]),
            int(numpy_payload["has_gauss"]),
            float(numpy_payload["cached_gaussian"]),
        )
    )
    torch_module.set_rng_state(
        torch_module.tensor(list(validated["torch_cpu"]), dtype=torch_module.uint8, device="cpu")
    )
    if device is not None:
        torch_module.cuda.set_rng_state(
            torch_module.tensor(
                list(validated["torch_cuda"]),
                dtype=torch_module.uint8,
                device="cpu",
            ),
            device=device,
        )


def _rank_rng_state_digest(state: Any, *, require_cuda: bool) -> str:
    validated = _validate_rank_rng_state(state, require_cuda=require_cuda)
    payload = {
        name: (value if isinstance(value, str) else hashlib.sha256(value).hexdigest())
        for name, value in sorted(validated.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"G1 checkpoint {name} is not a lowercase SHA-256 digest")
    return value


def _update_tensor_digest(
    digest: Any,
    *,
    name: str,
    tensor: Any,
    torch_module: Any,
) -> None:
    """Hash one local tensor shard without materializing one giant bytes object."""

    if not torch_module.is_tensor(tensor):
        raise TypeError(f"{name} is not a tensor")
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    local = local.detach().contiguous()
    metadata = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "global_shape": [int(size) for size in tensor.shape],
            "local_shape": [int(size) for size in local.shape],
            "name": name,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = local.reshape(-1).view(torch_module.uint8)
    chunk_size = 8 * 1024 * 1024
    for offset in range(0, int(raw.numel()), chunk_size):
        chunk = raw[offset : offset + chunk_size]
        if chunk.device.type != "cpu":
            chunk = chunk.cpu()
        digest.update(memoryview(chunk.contiguous().numpy()))


def _canonical_optimizer_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer metadata contains a non-finite float")
        return {"float_hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_canonical_optimizer_value(item) for item in value]
    raise TypeError(f"unsupported optimizer metadata type: {type(value).__name__}")


def _model_local_state_digest(model: Any, torch_module: Any) -> str:
    """Bind every rank-local model parameter and buffer byte at a checkpoint boundary."""

    entries = [
        *(("parameter", name, value) for name, value in model.named_parameters()),
        *(("buffer", name, value) for name, value in model.named_buffers()),
    ]
    if not entries:
        raise RuntimeError("cannot digest an empty model state")
    digest = hashlib.sha256(b"picf-next.g1-model-local-state.v1\0")
    for kind, name, value in sorted(entries, key=lambda item: (item[0], item[1])):
        _update_tensor_digest(
            digest,
            name=f"{kind}:{name}",
            tensor=value,
            torch_module=torch_module,
        )
    return digest.hexdigest()


def _optimizer_local_state_digest(optimizer: Any, model: Any, torch_module: Any) -> str:
    """Bind AdamW groups and every rank-local continuation tensor by model name."""

    named_parameters = sorted(model.named_parameters(), key=lambda item: item[0])
    parameter_names = {id(parameter): name for name, parameter in named_parameters}
    if not named_parameters or len(parameter_names) != len(named_parameters):
        raise RuntimeError("G1 model parameter names are empty or ambiguous")
    digest = hashlib.sha256(b"picf-next.g1-optimizer-local-state.v1\0")
    for index, group in enumerate(optimizer.param_groups):
        try:
            names = [parameter_names[id(parameter)] for parameter in group["params"]]
        except KeyError as error:
            raise RuntimeError("optimizer contains a parameter outside the model") from error
        metadata = {
            "group": index,
            "parameters": names,
            "options": {
                name: _canonical_optimizer_value(value)
                for name, value in sorted(group.items())
                if name != "params"
            },
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("cannot digest an empty optimizer state")
    for name, parameter in named_parameters:
        entry = state.get(parameter)
        marker = json.dumps(
            {"name": name, "state": entry is not None},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(marker).to_bytes(8, "big"))
        digest.update(marker)
        if entry is None:
            continue
        if not isinstance(entry, Mapping):
            raise RuntimeError("optimizer parameter state is not a mapping")
        for field, value in sorted(entry.items()):
            if torch_module.is_tensor(value):
                _update_tensor_digest(
                    digest,
                    name=f"state:{name}:{field}",
                    tensor=value,
                    torch_module=torch_module,
                )
                continue
            encoded = json.dumps(
                {
                    "name": f"state:{name}:{field}",
                    "value": _canonical_optimizer_value(value),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _checkpoint_boundary_digests(
    *,
    model: Any,
    optimizer: Any,
    picf_session_snapshot: bytes,
    rank_rng_state: Any,
    torch_module: Any,
    require_cuda_rng: bool,
) -> dict[str, str]:
    if not isinstance(picf_session_snapshot, bytes) or not picf_session_snapshot:
        raise ValueError("PICF session snapshot must be nonempty bytes")
    return {
        "model_local_state_sha256": _model_local_state_digest(model, torch_module),
        "optimizer_local_state_sha256": _optimizer_local_state_digest(
            optimizer,
            model,
            torch_module,
        ),
        "picf_session_snapshot_sha256": hashlib.sha256(picf_session_snapshot).hexdigest(),
        "rank_rng_state_sha256": _rank_rng_state_digest(
            rank_rng_state,
            require_cuda=require_cuda_rng,
        ),
    }


def _local_gradient_square(model: Any, name_fragment: str) -> tuple[float, int]:
    square = 0.0
    count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if name_fragment not in name or gradient is None:
            continue
        local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
        square += float(local.detach().float().square().sum().item())
        count += int(local.numel())
    return square, count


def _validate_adamw_state(
    optimizer: Any,
    torch_module: Any,
    *,
    expected_step: int,
) -> dict[str, int]:
    """Validate the exact FP32 CPU AdamW continuation state for one rank."""

    if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step <= 0:
        raise ValueError("expected AdamW step must be a positive integer")
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("official DCP restart did not restore optimizer state")
    local_moment_elements = 0
    for parameter, entry in state.items():
        if not isinstance(entry, Mapping) or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError("AdamW state schema differs from the frozen G1 contract")
        step = entry["step"]
        if not torch_module.is_tensor(step) or step.numel() != 1:
            raise RuntimeError("AdamW step state must be one tensor scalar")
        local_step = step.to_local() if hasattr(step, "to_local") else step
        if local_step.device.type != "cpu" or not torch_module.isfinite(local_step).all():
            raise RuntimeError("AdamW step state must be finite and CPU-resident")
        if float(local_step.item()) != float(expected_step):
            raise RuntimeError("AdamW step state differs from the checkpoint boundary")
        for name in ("exp_avg", "exp_avg_sq"):
            moment = entry[name]
            if not torch_module.is_tensor(moment) or moment.shape != parameter.shape:
                raise RuntimeError(f"AdamW {name} state has an incompatible parameter schema")
            local = moment.to_local() if hasattr(moment, "to_local") else moment
            if local.dtype != torch_module.float32 or local.device.type != "cpu":
                raise RuntimeError(f"AdamW {name} must use FP32 CPU-offloaded storage")
            local_moment_elements += int(local.numel())
    if local_moment_elements <= 0:
        raise RuntimeError("AdamW checkpoint contains no local moment elements")
    return {
        "optimizer_state_entries": len(state),
        "optimizer_local_moment_elements": local_moment_elements,
    }


def _move_model_inputs(model_inputs: dict[str, Any], *, device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for name, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            moved[name] = value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
                non_blocking=False,
            )
        else:
            moved[name] = value
    return moved


def main() -> None:
    args = _parse_args()
    _validate_paths_and_args(args)
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_unified_patches(root=root, checkout=args.source_checkout)
    if os.environ.get("WORLD_SIZE") != str(G1_WORLD_SIZE):
        raise RuntimeError("G1 must run under torchrun with exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(G1_WORLD_SIZE):
        raise RuntimeError("G1 requires both processes on one two-GPU host")
    if _git_head(args.source_checkout) != LINGBOT_SOURCE_COMMIT:
        raise RuntimeError("prepared LingBot source differs from the pinned commit")
    patch_states = [
        detect_patch_state(args.source_checkout, patch)
        for patch in (args.data_patch, args.graph_patch)
    ]
    if patch_states != ["applied", "applied"]:
        raise RuntimeError(f"prepared LingBot patches are not both applied: {patch_states}")
    patched_source_hashes = patch_report.get("patched_source_sha256")
    if not isinstance(patched_source_hashes, dict) or not all(
        isinstance(relative, str) and isinstance(digest, str)
        for relative, digest in patched_source_hashes.items()
    ):
        raise RuntimeError("patch verifier returned an invalid source-hash contract")
    expected_source_hashes = {
        str(relative): str(digest) for relative, digest in patched_source_hashes.items()
    }
    actual_source_hashes = {
        relative: _sha256(args.source_checkout / relative) for relative in expected_source_hashes
    }
    if actual_source_hashes != expected_source_hashes:
        raise RuntimeError("prepared LingBot source differs from replayed patch bytes")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
    import torch
    import torch.distributed as dist
    from lingbotvla.checkpoint import build_checkpointer
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    try:
        from tools.lingbot_vla2_runtime_helpers import load_lingbot_training_config
    except ModuleNotFoundError:
        from lingbot_vla2_runtime_helpers import load_lingbot_training_config

    from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_files,
    )
    from picf_next.hosts.lingbot_calvin_training import (
        build_lingbot_calvin_stream_plan,
        build_planned_lingbot_calvin_batch,
        collate_lingbot_calvin_training_batch,
        materialize_lingbot_flow_randomness,
    )
    from picf_next.hosts.lingbot_unified import (
        LingBotHostContract,
        LingBotUnifiedBeliefGraph,
        LingBotUnifiedGraphConfig,
        install_lingbot_unified_belief_graph,
    )
    from picf_next.hosts.lingbot_unified_training import (
        LingBotUnifiedForwardResult,
        LingBotUnifiedLaneSession,
        LingBotUnifiedSessionConfig,
        lingbot_optimizer_source_digest,
        run_lingbot_unified_optimizer_attempt,
    )
    from picf_next.unified.codec import BeliefCodecConfig
    from picf_next.unified.state import GeometrySchema

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    try:
        if torch.cuda.device_count() != G1_WORLD_SIZE:
            raise RuntimeError("G1 process sees a CUDA topology other than two devices")
        dataset_contract: list[Any] = [None]
        if rank == 0:
            try:
                checked_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
                checked_norm_stats = json.loads(args.norm_stats.read_text())
                validate_lingbot_calvin_norm_stats(checked_norm_stats)
                norm_source = checked_norm_stats["source"]
                if (
                    norm_source["dataset_id"] != checked_manifest.dataset_id
                    or norm_source["dataset_revision"] != checked_manifest.dataset_revision
                    or checked_manifest.split_name != args.dataset_split.name
                ):
                    raise ValueError(
                        "G1 CALVIN manifest, split and normalization identities differ"
                    )
                dataset_contract[0] = {
                    "status": "PASS",
                    "manifest_sha256": _sha256(args.dataset_manifest),
                    "normalization_sha256": _sha256(args.norm_stats),
                    "validation": validate_dataset_files(
                        checked_manifest,
                        args.dataset_split,
                        dataset_id=norm_source["dataset_id"],
                        dataset_revision=norm_source["dataset_revision"],
                        split_name=args.dataset_split.name,
                        verify_hashes=True,
                    ),
                }
            except BaseException as error:
                dataset_contract[0] = {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(dataset_contract, src=0)
        dataset_contract_report = dataset_contract[0]
        if (
            not isinstance(dataset_contract_report, dict)
            or dataset_contract_report.get("status") != "PASS"
        ):
            raise RuntimeError(f"G1 dataset contract failed: {dataset_contract_report}")
        dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        init_parallel_state(
            dp_size=G1_WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=G1_WORLD_SIZE,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="fsdp2",
        )

        training = load_lingbot_training_config(args.training_config)
        merged, _ = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=2,
        )
        merged["use_cache"] = False
        merged["use_compile"] = False
        merged["attention_implementation"] = "eager"
        merged["vit_attn_implementation"] = "eager"
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        # QWEN_PROCESSOR_REVISION is an exact commit and this load is local-only.
        qwen_config = AutoConfig.from_pretrained(  # nosec B615
            args.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(args.processor_dir.resolve())

        processor = build_processor(str(args.processor_dir.resolve()))
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        with init_empty_weights(), no_init_weights():
            # Match the released LingBot mixed-precision training contract:
            # FP32 master parameters and optimizer state, BF16 only for compute.
            policy = LingbotVlaV2Policy(config=config, eval=False).to(torch.float32)
        load_model_weights(
            policy,
            str(args.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        policy.train()

        contract = LingBotHostContract.from_policy(policy)
        graph = LingBotUnifiedBeliefGraph(
            LingBotUnifiedGraphConfig.from_policy(
                policy,
                codec=BeliefCodecConfig(
                    content_dim=args.content_width,
                    geometry_dim=args.geometry_width,
                    uncertainty_dim=args.uncertainty_width,
                    host_width=contract.prefix_width,
                ),
                geometry_schema=GeometrySchema(
                    names=(
                        "center.x",
                        "center.y",
                        "center.z",
                        "extent.x",
                        "extent.y",
                        "extent.z",
                    ),
                    units=("metre",) * 6,
                    frame="camera",
                ),
                modality_names=("vision",),
                modality_reliability=(1.0,),
            )
        )
        install_lingbot_unified_belief_graph(policy, graph)
        # G1 measures action/PICF integration without requiring unreleased
        # teacher targets. The forward flag disables only loss-side alignment;
        # the released query-token layout remains instantiated and typed.

        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=True,
            enable_fp32=False,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=True,
            fsdp_kwargs={},
            basic_modules=policy._no_split_modules,
            enable_reentrant=False,
            enable_forward_prefetch=False,
            fsdp_llm_blocks=False,
            ignore_norm=False,
            use_depth_align=False,
            split_fused_experts_from_decoder_fsdp=False,
            vlm_fsdp=True,
            use_future_image=False,
        )
        parameter_storage = _validate_fsdp2_parameter_storage(policy, torch)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
        checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")

        # Graph construction must be identical on every rank. After parameters
        # are sharded, each data-parallel rank receives an independent stochastic
        # stream whose exact continuation is checkpointed below.
        import random

        rank_seed = args.seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)

        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=dataset_manifest.dataset_id,
            dataset_revision=dataset_manifest.dataset_revision,
            dataset_manifest=dataset_manifest,
        )
        dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
        plan = build_lingbot_calvin_stream_plan(
            dataset,
            comparison_id=G1_COMPARISON_ID,
            seed=args.seed,
            global_batch_size=G1_WORLD_SIZE,
            total_steps=2,
        )
        execution_contract_sha256, implementation_sha256 = _g1_execution_contract_digest(
            root=root,
            args=args,
            patched_source_sha256=actual_source_hashes,
        )
        model_family_digest = _model_family_digest(
            graph_contract_digest=graph.config.contract_digest,
            plan_sha256=plan.plan_sha256,
            execution_contract_sha256=execution_contract_sha256,
        )
        session_config = LingBotUnifiedSessionConfig(
            model_family_digest=model_family_digest,
            capacity=args.capacity,
            birth_noise_seed=args.seed,
        )
        global_step = 0
        resume_rng_state: dict[str, Any] | None = None
        loaded_checkpoint_boundary: dict[str, str] | None = None
        resume_runtime_rng_verified = False
        if args.phase == "fresh":
            session = LingBotUnifiedLaneSession(graph, session_config)
        else:
            checkpoint_dir = args.run_dir / "checkpoints" / f"global_step_{args.load_global_step}"
            if not checkpoint_dir.is_dir():
                raise FileNotFoundError(checkpoint_dir)
            state = {"model": policy, "optimizer": optimizer, "extra_state": {}}
            checkpointer.load(str(checkpoint_dir), state)
            prior_planned = build_planned_lingbot_calvin_batch(
                plan,
                dataset,
                optimizer_step=args.load_global_step - 1,
                rank=rank,
                world_size=G1_WORLD_SIZE,
                gradient_accumulation_steps=1,
                accumulation_index=0,
                graph=graph,
                capacity=args.capacity,
                device=device,
            )
            expected_source_digest = lingbot_optimizer_source_digest(
                (
                    (
                        prior_planned.training.temporal,
                        {},
                        prior_planned.source_digest,
                    ),
                )
            )
            extra = _validate_resume_extra_state(
                state["extra_state"],
                expected_global_step=args.load_global_step,
                expected_model_family_digest=model_family_digest,
                expected_plan_sha256=plan.plan_sha256,
                expected_rank=rank,
                expected_source_digest=expected_source_digest,
                expected_world_size=G1_WORLD_SIZE,
            )
            restored_optimizer = _validate_adamw_state(
                optimizer,
                torch,
                expected_step=args.load_global_step,
            )
            if any(
                restored_optimizer[name] != extra[name]
                for name in (
                    "optimizer_state_entries",
                    "optimizer_local_moment_elements",
                )
            ):
                raise RuntimeError("restored AdamW state differs from rank-local metadata")
            global_step = args.load_global_step
            session = LingBotUnifiedLaneSession.from_snapshot(
                graph,
                session_config,
                extra["picf_session_snapshot"],
                expected_optimizer_step=extra["picf_published_optimizer_step"],
            )
            resume_rng_state = extra["rank_rng_state"]
            loaded_checkpoint_boundary = _checkpoint_boundary_digests(
                model=policy,
                optimizer=optimizer,
                picf_session_snapshot=session.snapshot(),
                rank_rng_state=resume_rng_state,
                torch_module=torch,
                require_cuda_rng=True,
            )
            expected_checkpoint_boundary = {
                name: extra[name]
                for name in (
                    "model_local_state_sha256",
                    "optimizer_local_state_sha256",
                    "picf_session_snapshot_sha256",
                    "rank_rng_state_sha256",
                )
            }
            if loaded_checkpoint_boundary != expected_checkpoint_boundary:
                mismatches = sorted(
                    name
                    for name in expected_checkpoint_boundary
                    if loaded_checkpoint_boundary[name] != expected_checkpoint_boundary[name]
                )
                raise RuntimeError(
                    "restored G1 checkpoint boundary differs for " + ", ".join(mismatches)
                )

        checkpoint_root = args.run_dir / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        output_checkpoint = checkpoint_root / f"global_step_{global_step + 1}"
        staging_checkpoint = checkpoint_root / f".global_step_{global_step + 1}.incomplete"
        local_conflict = torch.tensor(
            int(output_checkpoint.exists() or staging_checkpoint.exists()),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(local_conflict, op=dist.ReduceOp.MAX)
        if bool(local_conflict.item()):
            raise FileExistsError(
                f"G1 output or incomplete checkpoint already exists: {output_checkpoint}"
            )
        data_config = SimpleNamespace(**json.loads(args.data_config.read_text()))
        feature_transform = FeatureTransform(
            str(args.robot_config.resolve()),
            data_config,
            config,
            processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(args.norm_stats.resolve()),
            use_depth_align=False,
            image_augment=False,
            use_future_image=False,
        )
        planned = build_planned_lingbot_calvin_batch(
            plan,
            dataset,
            optimizer_step=global_step,
            rank=rank,
            world_size=G1_WORLD_SIZE,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            graph=graph,
            capacity=args.capacity,
            device=device,
        )
        collated = collate_lingbot_calvin_training_batch(
            planned.training,
            feature_transform=feature_transform,
            collator=VLADataCollatorWithPacking(),
            augmentation_seeds=planned.augmentation_seeds,
            source_digest=planned.source_digest,
        )
        moved_inputs = _move_model_inputs(
            dict(collated.model_inputs),
            device=device,
            dtype=torch.bfloat16,
        )
        collated = type(collated)(
            model_inputs=moved_inputs,
            temporal=collated.temporal,
            sample_keys=collated.sample_keys,
            source_digest=collated.source_digest,
        )
        collated = materialize_lingbot_flow_randomness(collated, planned)
        if resume_rng_state is not None:
            # Restore after deterministic source materialization so the next
            # model forward starts at the uninterrupted stochastic boundary.
            _restore_rank_rng_state(resume_rng_state, torch, np, device=device)
            restored_rng_digest = _rank_rng_state_digest(
                _capture_rank_rng_state(torch, np, device=device),
                require_cuda=True,
            )
            if loaded_checkpoint_boundary is None or (
                restored_rng_digest != loaded_checkpoint_boundary["rank_rng_state_sha256"]
            ):
                raise RuntimeError("restored process RNG differs from the G1 checkpoint boundary")
            resume_runtime_rng_verified = True

        gradient_metrics: dict[str, float | int | bool] = {}

        def forward_step(model_inputs, context):
            outputs = policy(
                **model_inputs,
                unified_belief_context=context,
                compute_alignment_losses=False,
            )
            return LingBotUnifiedForwardResult(model_outputs=tuple(outputs))

        def optimizer_attempt() -> bool:
            local_finite = torch.ones((), dtype=torch.int32, device=device)
            for parameter in policy.parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                if not torch.isfinite(local).all():
                    local_finite.zero_()
                    break
            dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
            gradient_metrics["all_finite"] = bool(local_finite.item())
            if not bool(local_finite.item()):
                return False

            for metric, fragment in (
                ("graph_gradient", "unified_belief_graph"),
                ("action_gradient", "action_out_proj"),
            ):
                local_square, local_count = _local_gradient_square(policy, fragment)
                reduced = torch.tensor(
                    [local_square, float(local_count)],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                gradient_metrics[f"{metric}_norm"] = math.sqrt(float(reduced[0].item()))
                gradient_metrics[f"{metric}_elements"] = int(reduced[1].item())
            clipped = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                args.max_grad_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            clipped_value: Any = clipped
            full_tensor = getattr(clipped_value, "full_tensor", None)
            if callable(full_tensor):
                clipped_value = full_tensor()
            gradient_metrics["preclip_global_norm"] = float(clipped_value.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return True

        source_transaction = ((collated.temporal, collated.model_inputs, collated.source_digest),)
        started = time.perf_counter()
        attempt = run_lingbot_unified_optimizer_attempt(
            session,
            source_transaction,
            forward_step=forward_step,
            backward_step=lambda loss: loss.backward(),
            optimizer_attempt=optimizer_attempt,
            clear_gradients_after_skip=lambda: optimizer.zero_grad(set_to_none=True),
        )
        dist.barrier()
        elapsed = time.perf_counter() - started
        if not attempt.published:
            raise RuntimeError("G1 optimizer update was skipped; posterior was not published")
        if attempt.source_digest != lingbot_optimizer_source_digest(source_transaction):
            raise RuntimeError("G1 optimizer transaction source digest changed")
        graph_gradient = float(gradient_metrics.get("graph_gradient_norm", 0.0))
        action_gradient = float(gradient_metrics.get("action_gradient_norm", 0.0))
        if not math.isfinite(attempt.normalized_loss):
            raise RuntimeError("G1 action objective is non-finite")
        if graph_gradient <= 0 or action_gradient <= 0:
            raise RuntimeError("G1 did not produce nonzero graph and action gradients")

        rank_rng_state = _capture_rank_rng_state(torch, np, device=device)
        optimizer_state = _validate_adamw_state(
            optimizer,
            torch,
            expected_step=global_step + 1,
        )
        picf_session_snapshot = session.snapshot()
        saved_checkpoint_boundary = _checkpoint_boundary_digests(
            model=policy,
            optimizer=optimizer,
            picf_session_snapshot=picf_session_snapshot,
            rank_rng_state=rank_rng_state,
            torch_module=torch,
            require_cuda_rng=True,
        )
        extra_state = {
            "global_step": global_step + 1,
            "model_family_digest": model_family_digest,
            "next_optimizer_step": global_step + 1,
            **optimizer_state,
            "picf_published_optimizer_step": global_step,
            "picf_session_snapshot": picf_session_snapshot,
            "plan_sha256": plan.plan_sha256,
            "rank": rank,
            "rank_rng_state": rank_rng_state,
            "schema": G1_EXTRA_STATE_SCHEMA,
            "source_digest": attempt.source_digest,
            "world_size": G1_WORLD_SIZE,
            **saved_checkpoint_boundary,
        }
        rank_report = {
            "rank": rank,
            "sample_keys": list(collated.sample_keys),
            "lane_ids": list(collated.temporal.lane_ids),
            "episode_keys": list(collated.temporal.episode_keys),
            "frame_indices": list(collated.temporal.frame_indices),
            "normalized_loss": attempt.normalized_loss,
            "transaction_source_digest": attempt.source_digest,
            "gradient_metrics": gradient_metrics,
            "optimizer_state": optimizer_state,
            "step_time_s": elapsed,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "saved_checkpoint_boundary_sha256": saved_checkpoint_boundary,
            "loaded_checkpoint_boundary_sha256": loaded_checkpoint_boundary,
            "resume_checkpoint_boundary_verified": loaded_checkpoint_boundary is not None,
            "resume_runtime_rng_verified": resume_runtime_rng_verified,
        }
        gathered: list[Any] = [None for _ in range(G1_WORLD_SIZE)]
        dist.all_gather_object(gathered, rank_report)
        report = None
        if rank == 0:
            report = {
                "schema": G1_REPORT_SCHEMA,
                "phase": args.phase,
                "source_commit": LINGBOT_SOURCE_COMMIT,
                "patch_states": patch_states,
                "patched_source_sha256": actual_source_hashes,
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "graph_patch_sha256": _sha256(args.graph_patch),
                "graph_contract_digest": graph.config.contract_digest,
                "execution_contract_sha256": execution_contract_sha256,
                "implementation_sha256": implementation_sha256,
                "model_family_digest": model_family_digest,
                "plan_sha256": plan.plan_sha256,
                "dataset_contract": dataset_contract_report,
                "input_global_step": global_step,
                "saved_global_step": global_step + 1,
                "checkpoint_dir": str(output_checkpoint.resolve()),
                "full_shard": True,
                "fsdp2_cpu_offload": True,
                "parameter_storage": parameter_storage,
                "gradient_checkpointing": True,
                "auxiliary_target_losses_enabled": False,
                "rank_reports": gathered,
            }
        checkpointer.save(
            str(staging_checkpoint),
            {"model": policy, "optimizer": optimizer, "extra_state": extra_state},
            global_steps=None,
        )
        dist.barrier()
        publish_status: list[str | None] = [None]
        if rank == 0:
            try:
                assert report is not None
                report_payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
                _write_text_durable(staging_checkpoint / "g1_report.json", report_payload)
                os.replace(staging_checkpoint, output_checkpoint)
                directory_descriptor = os.open(checkpoint_root, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except BaseException as error:
                publish_status[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(publish_status, src=0)
        if publish_status[0] is not None:
            raise RuntimeError(f"G1 checkpoint publication failed: {publish_status[0]}")
        if rank == 0:
            assert report is not None
            report_path = args.run_dir / f"g1_{args.phase}_step_{global_step + 1}.json"
            _write_text_durable(
                report_path,
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _git_head(checkout: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
