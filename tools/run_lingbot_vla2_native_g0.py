#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Run the two-rank ADR-74 native full-update and cold-resume G0 gate.

``fresh`` performs one real FSDP2 optimizer update and durably checkpoints the
official LingBot model, optimizer, process RNG and detached posterior lanes.
``resume`` starts in a new ``torchrun`` process, verifies that exact boundary,
advances the next frozen CALVIN transition and publishes step two.

Accelerator and LingBot imports remain inside :func:`main`, so argument,
source, patch and report contracts stay locally testable without the 6B model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _repository_import_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _repository_import_text = str(_repository_import_path)
    while _repository_import_text in sys.path:
        sys.path.remove(_repository_import_text)
    sys.path.insert(0, _repository_import_text)

from tools.cuda_allocator_bootstrap import (
    CUDA_ALLOCATOR_MODES,
    bootstrap_cuda_allocator,
    configure_cuda_allocator as _configure_cuda_allocator,
)

_BOOTSTRAPPED_CUDA_ALLOCATOR = (
    bootstrap_cuda_allocator(sys.argv[1:]) if __name__ == "__main__" else None
)

import picf_next as _picf_next_package

if (
    _picf_next_package.__file__ is None
    or Path(_picf_next_package.__file__).resolve().parent
    != (_REPOSITORY_ROOT / "src/picf_next").resolve()
):
    raise RuntimeError("native G0 did not import picf_next from its own repository checkout")

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.capacity import (
    require_checkpoint_write_capacity,
    require_persistent_run_root,
)
from picf_next.training.run_lease import acquire_distributed_run_lease
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    SELECTIVE_EMBEDDING_MODULE,
    SELECTIVE_EMBEDDING_PARAMETER,
    SELECTIVE_FROZEN_VISION_MODULE,
    validate_fsdp2_placement,
)
from picf_next.lingbot_native.gate_evidence import validate_g0_report
from picf_next.lingbot_native.official_config import official_lingbot_data_config

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_REVISION,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        build_lingbot_official_optimizer,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_CHECKPOINT_REVISION,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        build_lingbot_official_optimizer,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )


G0_WORLD_SIZE = 2
G0_COMPARISON_ID = "lingbot-vla2-native-g0"
G0_LEGACY_ARCHITECTURE = "legacy_task_match_v1"
G0_LTOP_ARCHITECTURE = "lingbot_task_query_object_value_read_v1"
G0_ARCHITECTURES = (G0_LEGACY_ARCHITECTURE, G0_LTOP_ARCHITECTURE)
G0_EXTRA_STATE_SCHEMA = "picf-next.lingbot-vla2-native-g0-extra-state.v1"
G0_REPORT_SCHEMA = "picf-next.lingbot-vla2-native-g0.v4"
_EXTRA_STATE_KEYS = frozenset(
    {
        "boundary_sha256",
        "global_step",
        "lane_snapshot",
        "model_family_sha256",
        "next_optimizer_step",
        "optimizer_local_moment_elements",
        "optimizer_state_entries",
        "plan_sha256",
        "rank",
        "rank_rng_state",
        "schema",
        "source_digest",
        "world_size",
    }
)


def _g0_gradient_metric_fragments(
    architecture_identity: str,
) -> tuple[tuple[str, str], ...]:
    if architecture_identity not in G0_ARCHITECTURES:
        raise ValueError("native G0 gradient audit received an unknown architecture")
    fragments = [
        ("native_graph", "picf_native_graph"),
        ("action_output", "action_out_proj"),
    ]
    if architecture_identity == G0_LTOP_ARCHITECTURE:
        fragments.append(
            ("task_query", "picf_native_graph.task_query_embeddings")
        )
    return tuple(fragments)
_BOUNDARY_KEYS = frozenset(
    {
        "lane_snapshot_sha256",
        "model_local_state_sha256",
        "optimizer_local_state_sha256",
        "rank_rng_state_sha256",
    }
)
_RNG_KEYS = frozenset({"numpy_json", "python_json", "torch_cpu", "torch_cuda"})


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = _environment_path("PICF_LINGBOT_NATIVE_SOURCE") or (
        root / CHECKOUT_RELATIVE_PATH
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fresh", "resume"), required=True)
    parser.add_argument("--source-checkout", type=Path, default=source_default)
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
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
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--maximum-optimizer-lag", type=int, default=8)
    parser.add_argument(
        "--architecture-identity",
        choices=G0_ARCHITECTURES,
        default=G0_LEGACY_ARCHITECTURE,
    )
    parser.add_argument("--task-query-count", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--maximum-peak-reserved-gib", type=float, default=39.0)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        help="Explicit full, GPU, or shared-embedding-only parameter-shard placement.",
    )
    parser.add_argument(
        "--cuda-allocator",
        choices=CUDA_ALLOCATOR_MODES,
        default="native",
        help="Explicit pinned-runtime CUDA allocator mode; inherited allocator settings fail.",
    )
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_paths_and_args(args: argparse.Namespace) -> None:
    validate_fsdp2_placement(args.fsdp2_placement)
    if args.cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ValueError("native G0 CUDA allocator mode is unsupported")
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
        raise ValueError(f"native G0 paths are absent: {absent}")
    files = (
        args.patch,
        args.training_config,
        args.robot_config,
        args.data_config,
        args.dataset_manifest,
        args.norm_stats,
    )
    if any(not Path(path).is_file() for path in files):
        raise FileNotFoundError("one or more native G0 source/config/data files are absent")
    directories = (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
    )
    if any(not Path(path).is_dir() for path in directories):
        raise FileNotFoundError("one or more native G0 source/model/dataset directories are absent")
    integers = (
        args.seed,
        args.capacity,
        args.maximum_control_tokens,
        args.maximum_optimizer_lag,
        args.load_global_step,
        args.task_query_count,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("native G0 integer controls must be Python integers")
    if args.seed < 0 or min(integers[1:5]) <= 0 or args.task_query_count < 0:
        raise ValueError("native G0 dimensions and load step must be positive")
    if args.architecture_identity == G0_LTOP_ARCHITECTURE:
        if args.task_query_count <= 0:
            raise ValueError("LTOP G0 requires a positive task-query count")
    elif args.task_query_count:
        raise ValueError("historical G0 architecture cannot declare task-query rows")
    if args.seed > 0xFFFFFFFF - (G0_WORLD_SIZE - 1):
        raise ValueError("native G0 rank seeds must fit NumPy's uint32 domain")
    if args.phase == "resume" and args.load_global_step != 1:
        raise ValueError("native G0 resume must cold-start from fresh global step one")
    for name in ("learning_rate", "max_grad_norm", "maximum_peak_reserved_gib"):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"native G0 {name} must be finite and positive")


def _write_text_durable(path: Path, payload: str) -> None:
    write_text_durable_exclusive(path, payload)


def _distributed_phase_error(
    *,
    error: BaseException | None,
    phase: str,
    rank: int,
    dist_module: Any,
) -> None:
    """Exchange rank-local failures before any rank enters the next phase."""

    if not isinstance(phase, str) or not phase:
        raise ValueError("native G0 distributed phase must be nonempty")
    world_size = int(dist_module.get_world_size())
    if world_size <= 0:
        raise RuntimeError("native G0 distributed world size must be positive")
    local = (
        None
        if error is None
        else {
            "rank": rank,
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error)[:4096],
        }
    )
    gathered: list[Any] = [None for _ in range(world_size)]
    dist_module.all_gather_object(gathered, local)
    failures = tuple(item for item in gathered if item is not None)
    if failures:
        rendered = "; ".join(
            f"rank {item['rank']} {item['phase']} {item['type']}: {item['message']}"
            for item in failures
        )
        raise RuntimeError(f"distributed native G0 phase failed: {rendered}")


def _distributed_rank_local_call(
    *,
    action: Callable[[], Any],
    phase: str,
    rank: int,
    dist_module: Any,
) -> Any:
    """Convert one rank-local failure into a uniform distributed phase failure."""

    if not callable(action):
        raise TypeError("native G0 distributed phase action must be callable")
    result: Any = None
    rank_local_error: BaseException | None = None
    try:
        result = action()
    except BaseException as error:
        rank_local_error = error
    _distributed_phase_error(
        error=rank_local_error,
        phase=phase,
        rank=rank,
        dist_module=dist_module,
    )
    return result


def _fsync_tree(root: Path) -> None:
    """Make every regular file and directory durable before atomic publication."""

    if not root.is_dir() or root.is_symlink():
        raise ValueError("checkpoint staging root must be one real directory")
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint staging tree contains a symlink: {path}")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise ValueError(f"checkpoint staging tree contains a non-regular path: {path}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"checkpoint staging path changed type: {path}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _git_output(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint metadata contains a non-finite float")
        return {"float_hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TypeError(f"unsupported checkpoint metadata type: {type(value).__name__}")


def _update_tensor_digest(digest: Any, *, name: str, tensor: Any, torch_module: Any) -> None:
    if not torch_module.is_tensor(tensor):
        raise TypeError(f"checkpoint field {name} is not a tensor")
    local = tensor.to_local() if callable(getattr(tensor, "to_local", None)) else tensor
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


def _persistent_named_buffers(model: Any) -> list[tuple[str, Any]]:
    """Return buffers covered by PyTorch's model ``state_dict`` contract."""

    entries: list[tuple[str, Any]] = []
    for name, value in model.named_buffers():
        module_name, separator, local_name = name.rpartition(".")
        owner = model.get_submodule(module_name) if separator else model
        excluded = getattr(owner, "_non_persistent_buffers_set", None)
        if not isinstance(excluded, set):
            raise TypeError(f"model module {module_name or '<root>'} has invalid buffer metadata")
        if local_name not in excluded:
            entries.append((name, value))
    return entries


def _model_local_state_digest(model: Any, torch_module: Any) -> str:
    entries = [
        *(("parameter", name, value) for name, value in model.named_parameters()),
        *(("buffer", name, value) for name, value in _persistent_named_buffers(model)),
    ]
    if not entries:
        raise RuntimeError("native G0 cannot digest an empty model")
    digest = hashlib.sha256(b"picf-next.native-g0-model-local.v2\0")
    for kind, name, value in sorted(entries, key=lambda item: (item[0], item[1])):
        _update_tensor_digest(
            digest,
            name=f"{kind}:{name}",
            tensor=value,
            torch_module=torch_module,
        )
    return digest.hexdigest()


def _optimizer_local_state_digest(optimizer: Any, model: Any, torch_module: Any) -> str:
    named_parameters = sorted(model.named_parameters(), key=lambda item: item[0])
    parameter_names = {id(parameter): name for name, parameter in named_parameters}
    if not named_parameters or len(parameter_names) != len(named_parameters):
        raise RuntimeError("native G0 model parameter names are empty or ambiguous")
    digest = hashlib.sha256(b"picf-next.native-g0-optimizer-local.v1\0")
    for index, group in enumerate(optimizer.param_groups):
        try:
            names = [parameter_names[id(parameter)] for parameter in group["params"]]
        except KeyError as error:
            raise RuntimeError("native G0 optimizer owns a parameter outside the model") from error
        encoded = json.dumps(
            {
                "group": index,
                "parameters": names,
                "options": {
                    name: _canonical(value)
                    for name, value in sorted(group.items())
                    if name != "params"
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("native G0 optimizer state is empty")
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
            raise RuntimeError("native G0 optimizer parameter state is not a mapping")
        for field, value in sorted(entry.items()):
            if torch_module.is_tensor(value):
                _update_tensor_digest(
                    digest,
                    name=f"state:{name}:{field}",
                    tensor=value,
                    torch_module=torch_module,
                )
            else:
                encoded = json.dumps(
                    {"name": f"state:{name}:{field}", "value": _canonical(value)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
    return digest.hexdigest()


def _validate_optimizer_state(
    optimizer: Any,
    torch_module: Any,
    *,
    expected_step: int,
) -> dict[str, int]:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("native G0 optimizer has no continuation state")
    moment_elements = 0
    for entry in state.values():
        if not isinstance(entry, Mapping):
            raise RuntimeError("native G0 optimizer parameter state is not a mapping")
        if "step" in entry:
            step = entry["step"]
            with torch_module.no_grad():
                local_step = step.to_local() if callable(getattr(step, "to_local", None)) else step
            if (
                not torch_module.is_tensor(local_step)
                or local_step.numel() != 1
                or not torch_module.isfinite(local_step).all()
                or float(local_step.item()) != float(expected_step)
            ):
                raise RuntimeError("native G0 AdamW step differs from the checkpoint boundary")
            fields = ("exp_avg", "exp_avg_sq")
        elif "momentum_buffer" in entry:
            fields = ("momentum_buffer",)
        else:
            raise RuntimeError("native G0 optimizer state is neither AdamW nor Muon")
        for name in fields:
            value = entry.get(name)
            if value is None or not torch_module.is_tensor(value):
                raise RuntimeError(f"native G0 optimizer state omits {name}")
            with torch_module.no_grad():
                local = value.to_local() if callable(getattr(value, "to_local", None)) else value
            if not torch_module.isfinite(local).all():
                raise RuntimeError(f"native G0 optimizer state {name} is non-finite")
            moment_elements += int(local.numel())
    if moment_elements <= 0:
        raise RuntimeError("native G0 optimizer has no local moment elements")
    return {
        "optimizer_state_entries": len(state),
        "optimizer_local_moment_elements": moment_elements,
    }


def _capture_rank_rng(torch_module: Any, numpy_module: Any, *, device: Any) -> dict[str, bytes]:
    python_state = random.getstate()
    numpy_state = numpy_module.random.get_state()
    return {
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
        "torch_cuda": bytes(torch_module.cuda.get_rng_state(device).cpu().tolist()),
    }


def _validate_rank_rng(state: Any) -> dict[str, bytes]:
    if not isinstance(state, dict) or set(state) != _RNG_KEYS:
        raise ValueError("native G0 rank RNG state is incomplete")
    if any(not isinstance(value, bytes) or not value for value in state.values()):
        raise ValueError("native G0 rank RNG state contains an empty field")
    try:
        json.loads(state["python_json"].decode("ascii"))
        json.loads(state["numpy_json"].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("native G0 rank RNG JSON is invalid") from error
    return state


def _rank_rng_digest(state: Any) -> str:
    validated = _validate_rank_rng(state)
    payload = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(validated.items())}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _restore_rank_rng(state: Any, torch_module: Any, numpy_module: Any, *, device: Any) -> None:
    validated = _validate_rank_rng(state)
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
        torch_module.tensor(list(validated["torch_cpu"]), dtype=torch_module.uint8)
    )
    torch_module.cuda.set_rng_state(
        torch_module.tensor(list(validated["torch_cuda"]), dtype=torch_module.uint8),
        device=device,
    )


def _checkpoint_boundary(
    *,
    model: Any,
    optimizer: Any,
    lane_snapshot: bytes,
    rank_rng_state: Any,
    torch_module: Any,
) -> dict[str, str]:
    if not lane_snapshot:
        raise ValueError("native G0 lane snapshot is empty")
    return {
        "model_local_state_sha256": _model_local_state_digest(model, torch_module),
        "optimizer_local_state_sha256": _optimizer_local_state_digest(
            optimizer, model, torch_module
        ),
        "lane_snapshot_sha256": hashlib.sha256(lane_snapshot).hexdigest(),
        "rank_rng_state_sha256": _rank_rng_digest(rank_rng_state),
    }


def _validate_sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"native G0 {name} is not a lowercase SHA-256 digest")
    return value


def _validate_resume_extra(
    extra: Any,
    *,
    expected_global_step: int,
    expected_model_family_sha256: str,
    expected_plan_sha256: str,
    expected_source_digest: str,
    rank: int,
) -> dict[str, Any]:
    if not isinstance(extra, dict) or set(extra) != _EXTRA_STATE_KEYS:
        raise ValueError("native G0 checkpoint extra state is incomplete")
    if extra["schema"] != G0_EXTRA_STATE_SCHEMA:
        raise ValueError("native G0 checkpoint schema differs")
    if extra["global_step"] != expected_global_step:
        raise ValueError("native G0 checkpoint global step differs")
    if extra["next_optimizer_step"] != expected_global_step:
        raise ValueError("native G0 checkpoint optimizer continuation differs")
    if extra["model_family_sha256"] != expected_model_family_sha256:
        raise ValueError("native G0 model-family contract differs")
    if extra["plan_sha256"] != expected_plan_sha256:
        raise ValueError("native G0 frozen CALVIN plan differs")
    if extra["source_digest"] != expected_source_digest:
        raise ValueError("native G0 frozen CALVIN continuation differs")
    if extra["rank"] != rank or extra["world_size"] != G0_WORLD_SIZE:
        raise ValueError("native G0 checkpoint topology differs")
    if not isinstance(extra["lane_snapshot"], bytes) or not extra["lane_snapshot"]:
        raise ValueError("native G0 checkpoint lane snapshot is absent")
    _validate_rank_rng(extra["rank_rng_state"])
    boundary = extra["boundary_sha256"]
    if not isinstance(boundary, dict) or set(boundary) != _BOUNDARY_KEYS:
        raise ValueError("native G0 checkpoint boundary hashes are incomplete")
    for name, value in boundary.items():
        _validate_sha256(name, value)
    return extra


def _move_model_inputs(
    model_inputs: Mapping[str, Any],
    *,
    device: Any,
    dtype: Any,
    torch_module: Any,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for name, value in model_inputs.items():
        if torch_module.is_tensor(value):
            moved[name] = value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
                non_blocking=False,
            )
        else:
            moved[name] = value
    return moved


def _distributed_gradient_metrics(
    model: Any,
    metric_fragments: tuple[tuple[str, str], ...],
    *,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, float | int | bool]:
    """Reduce gradient health without synchronizing once per parameter."""

    finite_by_device: dict[Any, Any] = {}
    squares_by_device: dict[str, dict[Any, Any]] = {
        name: {} for name, _fragment in metric_fragments
    }
    counts = {name: 0 for name, _fragment in metric_fragments}
    local_finite = None
    reduced = None
    gradient_metrics_error: BaseException | None = None
    try:
        if not metric_fragments or len({name for name, _fragment in metric_fragments}) != len(
            metric_fragments
        ):
            raise ValueError("gradient metric names must be non-empty and unique")
        for parameter_name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            local = (
                gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
            )
            finite = torch_module.isfinite(local).all()
            previous_finite = finite_by_device.get(local.device)
            finite_by_device[local.device] = (
                finite if previous_finite is None else previous_finite & finite
            )
            for metric, fragment in metric_fragments:
                if fragment not in parameter_name:
                    continue
                square = local.detach().float().square().sum()
                device_squares = squares_by_device[metric]
                previous_square = device_squares.get(square.device)
                device_squares[square.device] = (
                    square if previous_square is None else previous_square + square
                )
                counts[metric] += int(local.numel())

        local_finite = torch_module.ones((), dtype=torch_module.int32, device=device)
        for finite in finite_by_device.values():
            local_finite.mul_(finite.to(device=device, dtype=torch_module.int32))
        packed: list[Any] = []
        for metric, _fragment in metric_fragments:
            square = torch_module.zeros((), dtype=torch_module.float64, device=device)
            for device_square in squares_by_device[metric].values():
                square.add_(device_square.to(device=device, dtype=torch_module.float64))
            packed.append(square)
            packed.append(
                torch_module.tensor(
                    float(counts[metric]), dtype=torch_module.float64, device=device
                )
            )
        reduced = torch_module.stack(packed)
    except BaseException as error:
        gradient_metrics_error = error

    all_gather_object = getattr(dist, "all_gather_object", None)
    if callable(all_gather_object):
        get_world_size = getattr(dist, "get_world_size", None)
        world_size = int(get_world_size()) if callable(get_world_size) else 1
        gathered_errors: list[Any] = [None for _ in range(world_size)]
        all_gather_object(
            gathered_errors,
            None
            if gradient_metrics_error is None
            else {
                "type": type(gradient_metrics_error).__name__,
                "message": str(gradient_metrics_error)[:4096],
            },
        )
        failures = tuple(value for value in gathered_errors if value is not None)
        if failures:
            rendered = "; ".join(
                f"rank {rank} {value['type']}: {value['message']}"
                for rank, value in enumerate(gathered_errors)
                if value is not None
            )
            raise RuntimeError(f"distributed gradient traversal failed: {rendered}")
    elif gradient_metrics_error is not None:
        raise gradient_metrics_error
    if local_finite is None or reduced is None:
        raise RuntimeError("distributed gradient metric preparation vanished")

    dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
    all_finite = bool(local_finite.item())
    metrics: dict[str, float | int | bool] = {"all_finite": all_finite}
    if not all_finite:
        return metrics

    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    values = reduced.cpu().tolist()
    for index, (metric, _fragment) in enumerate(metric_fragments):
        metrics[f"{metric}_norm"] = math.sqrt(float(values[2 * index]))
        metrics[f"{metric}_elements"] = int(values[2 * index + 1])
    return metrics


def _validate_fsdp2_parameter_storage(
    model: Any,
    torch_module: Any,
    *,
    expected_placement: str,
    expected_selective_cpu_module_classes: tuple[str, ...] = (),
    expected_selective_cpu_parameter_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    placement = validate_fsdp2_placement(expected_placement)
    selective_classes = tuple(expected_selective_cpu_module_classes)
    selective_prefixes = tuple(expected_selective_cpu_parameter_prefixes)
    if bool(selective_classes) != bool(selective_prefixes):
        raise ValueError(
            "selective CPU module classes and parameter prefixes must be declared together"
        )
    for values, label in (
        (selective_classes, "selective CPU module classes"),
        (selective_prefixes, "selective CPU parameter prefixes"),
    ):
        if (
            len(set(values)) != len(values)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"{label} must be unique nonempty strings")
    if selective_classes and placement not in {
        FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    }:
        raise ValueError(
            "selective-class CPU offload requires selective embedding placement, "
            "optionally with trainable-vision offload"
        )
    parameter_tensors = 0
    local_elements = 0
    device_parameter_tensors = {"cpu": 0, "cuda": 0}
    device_local_elements = {"cpu": 0, "cuda": 0}
    cpu_parameter_names: list[str] = []
    named_parameters = tuple(model.named_parameters())
    for name, parameter in named_parameters:
        to_local = getattr(parameter, "to_local", None)
        if not callable(to_local):
            raise RuntimeError(f"native G0 FSDP2 did not shard parameter as DTensor: {name}")
        if parameter.dtype != torch_module.float32:
            raise RuntimeError(f"native G0 FSDP2 master parameter is not FP32: {name}")
        local = to_local()
        local_device = getattr(local, "device", None)
        device_type = getattr(local_device, "type", None)
        if device_type not in device_parameter_tensors:
            raise RuntimeError(
                f"native G0 FSDP2 parameter shard uses an unsupported device: {name}"
            )
        local_numel = getattr(local, "numel", None)
        if not callable(local_numel):
            raise TypeError(f"native G0 FSDP2 shard is not tensor-like: {name}")
        local_numel_value: Any = local_numel()
        if isinstance(local_numel_value, bool) or not isinstance(local_numel_value, int):
            raise TypeError(f"native G0 FSDP2 shard has a non-integer size: {name}")
        parameter_tensors += 1
        local_elements += local_numel_value
        device_parameter_tensors[device_type] += 1
        device_local_elements[device_type] += local_numel_value
        if device_type == "cpu":
            cpu_parameter_names.append(name)
    if parameter_tensors == 0 or local_elements == 0:
        raise RuntimeError("native G0 FSDP2 storage contract found no parameters")

    expected_cpu_names: list[str]
    if placement == FSDP2_CPU_OFFLOAD:
        if device_parameter_tensors["cuda"] != 0:
            raise RuntimeError("full FSDP2 CPU offload retained CUDA parameter shards")
        expected_cpu_names = sorted(name for name, _parameter in named_parameters)
    elif placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD:
        if device_parameter_tensors["cuda"] == 0:
            raise RuntimeError("selective FSDP2 offload moved every parameter shard to CPU")
        topology = getattr(model, "_lingbot_fsdp2_selective_cpu_modules", None)
        if topology != (SELECTIVE_EMBEDDING_MODULE,):
            raise RuntimeError("selective FSDP2 offload topology differs from its contract")
        class_topology = tuple(
            getattr(model, "_lingbot_fsdp2_selective_cpu_module_classes", ())
        )
        if class_topology != selective_classes:
            raise RuntimeError(
                "selective-class FSDP2 topology differs from its placement contract"
            )
        class_parameter_names: set[str] = set()
        for prefix in selective_prefixes:
            prefix_names = {
                name
                for name, _parameter in named_parameters
                if name.startswith(f"{prefix}.")
            }
            if not prefix_names:
                raise RuntimeError(
                    f"selective-class FSDP2 prefix found no parameters: {prefix}"
                )
            class_parameter_names.update(prefix_names)
        expected_cpu_names = sorted(
            {SELECTIVE_EMBEDDING_PARAMETER, *class_parameter_names}
        )
    elif placement in {
        FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
        FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    }:
        trainable_vision = (
            placement == FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD
        )
        vision_contract = "trainable" if trainable_vision else "frozen"
        if device_parameter_tensors["cuda"] == 0:
            raise RuntimeError("selective FSDP2 offload moved every parameter shard to CPU")
        vlm_topology = getattr(model, "_lingbot_vlm_fsdp2_topology", None)
        if not isinstance(vlm_topology, Mapping):
            raise RuntimeError(
                f"selective {vision_contract}-vision offload omitted the VLM topology"
            )
        vision_modules = vlm_topology.get("vision")
        if (
            not isinstance(vision_modules, tuple)
            or not vision_modules
            or any(not isinstance(name, str) or not name for name in vision_modules)
        ):
            raise RuntimeError(
                f"selective {vision_contract}-vision offload has an invalid vision topology"
            )
        expected_modules = (SELECTIVE_FROZEN_VISION_MODULE, SELECTIVE_EMBEDDING_MODULE)
        topology = getattr(model, "_lingbot_fsdp2_selective_cpu_modules", None)
        if topology != expected_modules:
            raise RuntimeError(
                f"selective {vision_contract}-vision FSDP2 topology differs from its contract"
            )
        vision_parameter_names = {
            name
            for name, _parameter in named_parameters
            if name.startswith(f"{SELECTIVE_FROZEN_VISION_MODULE}.")
        }
        if not vision_parameter_names:
            raise RuntimeError(
                f"selective {vision_contract}-vision offload found no vision parameters"
            )
        mismatched_trainability = [
            name
            for name, parameter in named_parameters
            if name.startswith(f"{SELECTIVE_FROZEN_VISION_MODULE}.")
            and parameter.requires_grad != trainable_vision
        ]
        if mismatched_trainability:
            raise RuntimeError(
                f"selective {vision_contract}-vision offload changed vision trainability: "
                f"{mismatched_trainability[:3]}"
            )
        class_parameter_names: set[str] = set()
        if selective_classes:
            class_topology = tuple(
                getattr(model, "_lingbot_fsdp2_selective_cpu_module_classes", ())
            )
            if class_topology != selective_classes:
                raise RuntimeError(
                    "selective-class FSDP2 topology differs from its placement contract"
                )
            for prefix in selective_prefixes:
                prefix_names = {
                    name
                    for name, _parameter in named_parameters
                    if name.startswith(f"{prefix}.")
                }
                if not prefix_names:
                    raise RuntimeError(
                        f"selective-class FSDP2 prefix found no parameters: {prefix}"
                    )
                class_parameter_names.update(prefix_names)
        expected_cpu_names = sorted(
            {
                SELECTIVE_EMBEDDING_PARAMETER,
                *vision_parameter_names,
                *class_parameter_names,
            }
        )
    else:
        expected_cpu_names = []
        if device_parameter_tensors["cuda"] != parameter_tensors:
            raise RuntimeError("GPU-sharded FSDP2 retained CPU parameter shards")
    if sorted(cpu_parameter_names) != expected_cpu_names:
        actual_set = set(cpu_parameter_names)
        expected_set = set(expected_cpu_names)
        missing = sorted(expected_set - actual_set)[:5]
        unexpected = sorted(actual_set - expected_set)[:5]
        raise RuntimeError(
            "native G0 FSDP2 CPU parameter set differs from its placement contract: "
            f"missing={missing}, unexpected={unexpected}"
        )

    report = {
        "parameter_tensors": parameter_tensors,
        "local_elements": local_elements,
        "master_dtype": "float32",
        "placement": placement,
        "cpu_parameter_tensors": device_parameter_tensors["cpu"],
        "cpu_local_elements": device_local_elements["cpu"],
        "cuda_parameter_tensors": device_parameter_tensors["cuda"],
        "cuda_local_elements": device_local_elements["cuda"],
        "selective_cpu_parameter_names": (
            cpu_parameter_names
            if placement
            in {
                FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
                FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
                FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
            }
            else []
        ),
    }
    if selective_classes:
        report.update(
            {
                "selective_cpu_module_classes": list(selective_classes),
                "selective_cpu_parameter_prefixes": list(selective_prefixes),
            }
        )
    return report


def _resolve_local_module(root: Path, module: str) -> tuple[Path, ...]:
    if module == "picf_next" or module.startswith("picf_next."):
        base = root / "src"
    elif module == "tools" or module.startswith("tools."):
        base = root
    else:
        return ()
    parts = module.split(".")
    module_path = base.joinpath(*parts).with_suffix(".py")
    package_path = base.joinpath(*parts, "__init__.py")
    if module_path.is_file():
        resolved = [module_path]
    elif package_path.is_file():
        resolved = [package_path]
    else:
        raise FileNotFoundError(f"local implementation module is absent: {module}")
    for index in range(1, len(parts)):
        initializer = base.joinpath(*parts[:index], "__init__.py")
        if initializer.is_file():
            resolved.append(initializer)
    return tuple(resolved)


def _local_import_modules(root: Path, path: Path) -> tuple[str, ...]:
    root = root.resolve()
    path = path.resolve()
    relative = path.relative_to(root)
    parts = relative.parts
    if parts[0] == "src":
        parts = parts[1:]
    elif parts[0] != "tools":
        raise ValueError(f"local implementation path is outside src/tools: {path}")
    if parts[-1] == "__init__.py":
        current_module = ".".join(parts[:-1])
        package = current_module
    else:
        current_module = ".".join((*parts[:-1], Path(parts[-1]).stem))
        package = current_module.rpartition(".")[0]

    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    modules.add(node.module)
                continue
            if not package:
                raise ValueError(f"relative import has no package context: {path}:{node.lineno}")
            relative_name = "." * node.level + (node.module or "")
            module = importlib.util.resolve_name(relative_name, package)
            modules.add(module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{module}.{alias.name}"
                try:
                    _resolve_local_module(root, candidate)
                except FileNotFoundError:
                    continue
                modules.add(candidate)
    return tuple(sorted(modules))


def _implementation_paths(
    root: Path,
    *,
    entrypoint: Path | None = None,
) -> tuple[Path, ...]:
    """Resolve the transitive local Python closure imported by one runner."""

    root = root.resolve()
    resolved_entrypoint = (
        root / "tools/run_lingbot_vla2_native_g0.py"
        if entrypoint is None
        else entrypoint.resolve()
    )
    try:
        resolved_entrypoint.relative_to(root)
    except ValueError as exc:
        raise ValueError("implementation entrypoint must be inside the repository") from exc
    pending = [resolved_entrypoint]
    resolved: set[Path] = {root / "references/patches/lingbot_vla2_picf_native.patch"}
    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"native G0 implementation file is absent: {path}")
        resolved.add(path)
        for module in _local_import_modules(root, path):
            for imported in _resolve_local_module(root, module):
                if imported not in resolved:
                    pending.append(imported)
    return tuple(sorted(resolved))


def _implementation_digest(root: Path, *, entrypoint: Path | None = None) -> str:
    relative_paths = tuple(
        str(path.relative_to(root))
        for path in _implementation_paths(root, entrypoint=entrypoint)
    )
    payload = {path: _sha256(root / path) for path in relative_paths}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _execution_contract_digest(
    *,
    root: Path,
    args: argparse.Namespace,
    patched_source_sha256: Mapping[str, str],
    optimizer_contract: Mapping[str, Any],
) -> tuple[str, str]:
    implementation_sha256 = _implementation_digest(root)
    payload = {
        "schema": "picf-next.lingbot-vla2-native-g0-execution.v2",
        "implementation_sha256": implementation_sha256,
        "input_file_sha256": {
            "data_config": _sha256(args.data_config),
            "dataset_manifest": _sha256(args.dataset_manifest),
            "norm_stats": _sha256(args.norm_stats),
            "robot_config": _sha256(args.robot_config),
            "training_config": _sha256(args.training_config),
        },
        "optimizer": {
            "lingbot_release_contract": dict(optimizer_contract),
            "max_grad_norm": float(args.max_grad_norm).hex(),
        },
        "native": {
            "architecture_identity": args.architecture_identity,
            "capacity": args.capacity,
            "maximum_control_tokens": args.maximum_control_tokens,
            "maximum_optimizer_lag": args.maximum_optimizer_lag,
            "maximum_peak_reserved_gib": float(args.maximum_peak_reserved_gib).hex(),
            "task_query_count": args.task_query_count,
        },
        "patched_source_sha256": dict(sorted(patched_source_sha256.items())),
        "sampling": {
            "comparison_id": G0_COMPARISON_ID,
            "global_batch_size": G0_WORLD_SIZE,
            "seed": args.seed,
            "total_steps": 2,
        },
        "topology": {
            "cuda_allocator": args.cuda_allocator,
            "data_parallel_mode": "fsdp2",
            "fsdp2_placement": args.fsdp2_placement,
            "full_shard": True,
            "gradient_accumulation_steps": 1,
            "world_size": G0_WORLD_SIZE,
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, implementation_sha256


def main() -> None:
    args = _parse_args()
    _validate_paths_and_args(args)
    require_persistent_run_root(args.run_dir)
    require_checkpoint_write_capacity(args.run_dir)
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("CUDA allocator pre-bootstrap differs from parsed arguments")
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(
        root=root,
        checkout=args.source_checkout,
        check_apply=True,
    )
    prepared_source = validate_prepared_native_source(
        checkout=args.source_checkout,
        patch_path=args.patch,
    )
    expected_hashes = patch_report.get("patched_source_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("native G0 patch verifier returned no source hashes")
    actual_hashes = prepared_source.get("patched_source_sha256")
    if not isinstance(actual_hashes, dict):
        raise RuntimeError("native G0 prepared source returned no source hashes")
    if actual_hashes != expected_hashes:
        raise RuntimeError("native G0 LingBot source differs from immutable patch replay")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)

    if os.environ.get("WORLD_SIZE") != str(G0_WORLD_SIZE):
        raise RuntimeError("native G0 requires torchrun with exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(G0_WORLD_SIZE):
        raise RuntimeError("native G0 requires both processes on one two-GPU host")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
    import torch
    import torch.distributed as dist

    from picf_next.lingbot_native.torch_dcp_compat import (
        install_torch_2_8_sparse_optimizer_state_backport,
    )

    install_torch_2_8_sparse_optimizer_state_backport(torch)

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
    from lingbotvla.models.vla.lingbot_vla.moe_load_balance import (
        build_moe_load_balance_hook,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )
    from lingbotvla.optim import build_muon_optimizer
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.dataset_manifest import (
        DatasetFileManifest,
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.calvin import (
        CollatedNativeCALVINBatch,
        build_native_calvin_context,
        build_native_calvin_stream_plan,
        build_planned_native_calvin_batch,
        collate_native_calvin_training_batch,
        materialize_native_flow_randomness,
    )
    from picf_next.lingbot_native.host import (
        LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        install_lingbot_native_graph,
    )
    from picf_next.lingbot_native.addresses import address_codebook_sha256
    from picf_next.lingbot_native.state import NativeLayerwisePosteriorState
    from picf_next.lingbot_native.temporal import (
        NativeLaneConfig,
        NativeTrainingLaneBank,
    )
    from picf_next.lingbot_native.training import (
        NativeTrainingLaneCoordinator,
        audit_native_optimizer_coverage,
        native_persistent_output,
        run_native_policy_training_forward,
        run_native_v3_two_pass_policy_training_forward,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    run_lease = None
    try:
        run_lease = acquire_distributed_run_lease(
            args.run_dir,
            rank=rank,
            distributed=dist,
        )
        if torch.cuda.device_count() != G0_WORLD_SIZE:
            raise RuntimeError("native G0 process sees a CUDA topology other than two devices")
        device_properties = torch.cuda.get_device_properties(device)
        if "A100" not in device_properties.name or device_properties.total_memory < 39 * 1024**3:
            raise RuntimeError("native G0 requires two A100 devices with at least 39 GiB each")
        dataset_contract: list[Any] = [None]
        rank_zero_manifest: DatasetFileManifest | None = None
        if rank == 0:
            try:
                rank_zero_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
                norm_stats = json.loads(args.norm_stats.read_text())
                validate_lingbot_calvin_norm_stats(norm_stats)
                source = norm_stats["source"]
                if (
                    source["dataset_id"] != rank_zero_manifest.dataset_id
                    or source["dataset_revision"] != rank_zero_manifest.dataset_revision
                    or source["dataset_tree_sha256"] != rank_zero_manifest.tree_sha256
                    or rank_zero_manifest.split_name != args.dataset_split.name
                ):
                    raise ValueError("native G0 CALVIN manifest and normalization differ")
                dataset_contract[0] = {
                    "status": "PASS",
                    "manifest_sha256": _sha256(args.dataset_manifest),
                    "normalization_sha256": _sha256(args.norm_stats),
                    "validation": validate_dataset_runtime_binding(
                        rank_zero_manifest,
                        args.dataset_split,
                        dataset_id=source["dataset_id"],
                        dataset_revision=source["dataset_revision"],
                        split_name=args.dataset_split.name,
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
            raise RuntimeError(f"native G0 dataset contract failed: {dataset_contract_report}")
        dataset_manifest = (
            rank_zero_manifest
            if rank_zero_manifest is not None
            else load_dataset_file_manifest(args.dataset_manifest.resolve())
        )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        init_parallel_state(
            dp_size=G0_WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=G0_WORLD_SIZE,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="fsdp2",
        )

        training = load_lingbot_training_config(args.training_config)
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=args.learning_rate,
        )
        require_lingbot_exact_resume_contract(optimizer_contract)
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
            policy = LingbotVlaV2Policy(config=config, eval=False).to(torch.float32)
        load_model_weights(
            policy,
            str(args.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        strip_targetless_alignment_teacher_heads(policy)
        policy.train()
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=args.capacity,
            maximum_control_tokens=args.maximum_control_tokens,
            task_query_count=args.task_query_count,
            architecture_identity=args.architecture_identity,
        )
        graph = LingBotNativeGraph(graph_config, device=device, dtype=torch.float32).train()
        install_lingbot_native_graph(policy, graph)

        full_cpu_offload = args.fsdp2_placement == FSDP2_CPU_OFFLOAD
        selective_embedding_offload = args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=optimizer_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=full_cpu_offload,
            enable_shared_embedding_offload=selective_embedding_offload,
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
        register_native_fsdp_forward_methods(policy)
        parameter_storage = _validate_fsdp2_parameter_storage(
            policy,
            torch,
            expected_placement=args.fsdp2_placement,
        )
        optimizer = build_lingbot_official_optimizer(
            policy,
            optimizer_contract,
            build_muon_optimizer=build_muon_optimizer,
            build_moe_load_balance_hook=build_moe_load_balance_hook,
        )
        parameter_manifest = audit_native_optimizer_coverage(
            modules={"policy": policy},
            optimizer=optimizer,
        )
        checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")

        rank_seed = args.seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)

        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=dataset_manifest.dataset_id,
            dataset_revision=dataset_manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=dataset_manifest,
        )
        dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
        plan = build_native_calvin_stream_plan(
            dataset,
            comparison_id=G0_COMPARISON_ID,
            seed=args.seed,
            global_batch_size=G0_WORLD_SIZE,
            total_steps=2,
        )
        execution_sha256, implementation_sha256 = _execution_contract_digest(
            root=root,
            args=args,
            patched_source_sha256=actual_hashes,
            optimizer_contract=optimizer_contract.metadata,
        )
        model_family_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "execution_sha256": execution_sha256,
                    "graph": asdict(graph_config),
                    "plan_sha256": plan.plan_sha256,
                    "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        addressed_codebook_sha256 = None
        if args.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ:
            if graph.episode_address_codebook is None:
                raise RuntimeError("LTOP graph omitted its immutable episode address codebook")
            addressed_codebook_sha256 = address_codebook_sha256(
                graph.episode_address_codebook
            )
        lane_config = NativeLaneConfig(
            model_digest=model_family_sha256,
            schema_digest=plan.plan_sha256,
            capacity=args.capacity,
            host_width=graph_config.host_width,
            maximum_optimizer_lag=args.maximum_optimizer_lag,
            num_layers=(
                graph_config.num_layers
                if args.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ
                else None
            ),
            addressed_architecture_identity=(
                args.architecture_identity
                if args.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ
                else None
            ),
            episode_address_codebook_sha256=addressed_codebook_sha256,
            device=str(device),
            dtype=torch.bfloat16,
        )
        global_step = 0
        resume_rng: dict[str, bytes] | None = None
        loaded_boundary: dict[str, str] | None = None
        if args.phase == "fresh":
            bank = NativeTrainingLaneBank(lane_config)
        else:
            checkpoint_dir = args.run_dir / "checkpoints" / f"global_step_{args.load_global_step}"
            if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                raise FileNotFoundError(checkpoint_dir)
            checkpoint_report_path = checkpoint_dir / "native_g0_report.json"
            if checkpoint_report_path.is_symlink() or not checkpoint_report_path.is_file():
                raise ValueError("native G0 checkpoint lacks its immutable report before load")
            try:
                checkpoint_report = json.loads(checkpoint_report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("native G0 checkpoint report is unreadable before load") from error
            validated_checkpoint_report = validate_g0_report(
                checkpoint_report,
                schema=G0_REPORT_SCHEMA,
                phase="fresh",
                source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
                checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
                world_size=G0_WORLD_SIZE,
                expected_fsdp2_placement=args.fsdp2_placement,
                expected_cuda_allocator=args.cuda_allocator,
            )
            expected_resume_provenance = {
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "execution_contract_sha256": execution_sha256,
                "implementation_sha256": implementation_sha256,
                "model_family_sha256": model_family_sha256,
                "plan_sha256": plan.plan_sha256,
            }
            if any(
                validated_checkpoint_report.get(name) != expected
                for name, expected in expected_resume_provenance.items()
            ):
                raise ValueError("native G0 checkpoint report targets another execution")
            state = {"model": policy, "optimizer": optimizer, "extra_state": {}}
            checkpointer.load(str(checkpoint_dir), state)
            prior_planned = build_planned_native_calvin_batch(
                plan,
                dataset,
                optimizer_step=args.load_global_step - 1,
                rank=rank,
                world_size=G0_WORLD_SIZE,
                gradient_accumulation_steps=1,
                accumulation_index=0,
                device=device,
                dtype=torch.bfloat16,
            )
            extra = _validate_resume_extra(
                state["extra_state"],
                expected_global_step=args.load_global_step,
                expected_model_family_sha256=model_family_sha256,
                expected_plan_sha256=plan.plan_sha256,
                expected_source_digest=prior_planned.source_digest,
                rank=rank,
            )
            optimizer_summary = _validate_optimizer_state(
                optimizer,
                torch,
                expected_step=args.load_global_step,
            )
            if any(
                optimizer_summary[name] != extra[name]
                for name in ("optimizer_state_entries", "optimizer_local_moment_elements")
            ):
                raise RuntimeError("native G0 restored optimizer summary differs")
            bank = NativeTrainingLaneBank.deserialize(lane_config, extra["lane_snapshot"])
            resume_rng = extra["rank_rng_state"]
            loaded_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=bank.serialize(),
                rank_rng_state=resume_rng,
                torch_module=torch,
            )
            if loaded_boundary != extra["boundary_sha256"]:
                raise RuntimeError("native G0 restored checkpoint boundary differs")
            global_step = args.load_global_step

        checkpoint_root = args.run_dir / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        output_checkpoint = checkpoint_root / f"global_step_{global_step + 1}"
        staging_checkpoint = checkpoint_root / f".global_step_{global_step + 1}.incomplete"
        report_path = args.run_dir / f"native_g0_{args.phase}_step_{global_step + 1}.json"
        conflict = torch.tensor(
            int(
                any(
                    path.exists() or path.is_symlink()
                    for path in (output_checkpoint, staging_checkpoint, report_path)
                )
            ),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(conflict, op=dist.ReduceOp.MAX)
        if bool(conflict.item()):
            raise FileExistsError(f"native G0 checkpoint path already exists: {output_checkpoint}")

        def prepare_step_inputs() -> tuple[Any, Any]:
            feature_transform = FeatureTransform(
                str(args.robot_config.resolve()),
                official_lingbot_data_config(json.loads(args.data_config.read_text())),
                config,
                processor,
                chunk_size=config.chunk_size,
                norm_stats_path=str(args.norm_stats.resolve()),
                use_depth_align=False,
                image_augment=False,
                use_future_image=False,
            )
            planned_batch = build_planned_native_calvin_batch(
                plan,
                dataset,
                optimizer_step=global_step,
                rank=rank,
                world_size=G0_WORLD_SIZE,
                gradient_accumulation_steps=1,
                accumulation_index=0,
                device=device,
                dtype=torch.bfloat16,
            )
            collated_batch = collate_native_calvin_training_batch(
                planned_batch.training,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=planned_batch.augmentation_seeds,
                source_digest=planned_batch.source_digest,
            )
            collated_batch = CollatedNativeCALVINBatch(
                model_inputs=_move_model_inputs(
                    collated_batch.model_inputs,
                    device=device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                ),
                controls=collated_batch.controls,
                routing=collated_batch.routing,
                source_digest=collated_batch.source_digest,
                structural_target_requests=collated_batch.structural_target_requests,
                modalities=(
                    None
                    if collated_batch.modalities is None
                    else collated_batch.modalities.to(device=device, dtype=torch.bfloat16)
                ),
            )
            materialized = materialize_native_flow_randomness(
                collated_batch,
                planned_batch,
            )
            torch.cuda.synchronize(device)
            return planned_batch, materialized

        planned, collated = _distributed_rank_local_call(
            action=prepare_step_inputs,
            phase=f"native-g0-step-{global_step}-data-prepare",
            rank=rank,
            dist_module=dist,
        )
        resume_runtime_rng_verified = False
        if resume_rng is not None:

            def restore_runtime_rng() -> bool:
                _restore_rank_rng(resume_rng, torch, np, device=device)
                verified = _rank_rng_digest(
                    _capture_rank_rng(torch, np, device=device)
                ) == _rank_rng_digest(resume_rng)
                if not verified:
                    raise RuntimeError("native G0 restored process RNG differs")
                return verified

            resume_runtime_rng_verified = bool(
                _distributed_rank_local_call(
                    action=restore_runtime_rng,
                    phase=f"native-g0-step-{global_step}-rng-restore",
                    rank=rank,
                    dist_module=dist,
                )
            )

        coordinator = NativeTrainingLaneCoordinator(bank)
        attempt = _distributed_rank_local_call(
            action=lambda: coordinator.begin(
                optimizer_step=global_step,
                source_weight_version=0,
            ),
            phase=f"native-g0-step-{global_step}-lane-begin",
            rank=rank,
            dist_module=dist,
        )
        started = time.perf_counter()

        def prepare_lane_context() -> tuple[Any, Any | None]:
            prepared_batch = attempt.prepare(collated.routing)
            if args.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ:
                if not isinstance(
                    prepared_batch.previous_state,
                    NativeLayerwisePosteriorState,
                ):
                    raise RuntimeError("LTOP lane preparation omitted layerwise posterior memory")
                return prepared_batch, None
            native_context = build_native_calvin_context(
                collated,
                previous_state=prepared_batch.previous_state,
                previous_state_valid=prepared_batch.previous_state_valid,
            )
            return prepared_batch, native_context

        try:
            prepared, context = _distributed_rank_local_call(
                action=prepare_lane_context,
                phase=f"native-g0-step-{global_step}-lane-prepare",
                rank=rank,
                dist_module=dist,
            )

            def forward_and_backward() -> Any:
                if args.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ:
                    result = run_native_v3_two_pass_policy_training_forward(
                        policy,
                        model_inputs=collated.model_inputs,
                        controls=collated.controls,
                        previous_memory=prepared.previous_state,
                        previous_memory_valid=prepared.previous_state_valid,
                        modalities=collated.modalities,
                        prior_control_chunks=collated.effective_prior_control_chunks,
                    ).policy_forward
                else:
                    if context is None:
                        raise RuntimeError("historical native G0 omitted its forward context")
                    result = run_native_policy_training_forward(
                        policy,
                        model_inputs=collated.model_inputs,
                        context=context,
                    )
                result.official_total_loss.backward()
                torch.cuda.synchronize(device)
                return result

            forward = _distributed_rank_local_call(
                action=forward_and_backward,
                phase=f"native-g0-step-{global_step}-forward-backward",
                rank=rank,
                dist_module=dist,
            )

            def stage_posterior() -> None:
                posterior = native_persistent_output(forward.context)
                attempt.stage(
                    prepared,
                    posterior,
                    row_bindings_by_batch=prepared.previous_row_bindings,
                )
                torch.cuda.synchronize(device)

            _distributed_rank_local_call(
                action=stage_posterior,
                phase=f"native-g0-step-{global_step}-posterior-stage",
                rank=rank,
                dist_module=dist,
            )
        except BaseException:
            attempt.abort()
            optimizer.zero_grad(set_to_none=True)
            raise

        gradient_metrics: dict[str, float | int | bool] = {}

        def optimizer_attempt() -> int | None:
            def audit_gradients() -> dict[str, float | int | bool]:
                metrics = _distributed_gradient_metrics(
                    policy,
                    _g0_gradient_metric_fragments(args.architecture_identity),
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )
                if not bool(metrics["all_finite"]):
                    raise FloatingPointError("native G0 produced a non-finite gradient")
                if float(metrics.get("native_graph_norm", 0.0)) <= 0:
                    raise RuntimeError("native G0 produced no gradient in the native graph")
                if float(metrics.get("action_output_norm", 0.0)) <= 0:
                    raise RuntimeError("native G0 produced no gradient in the action output")
                if args.architecture_identity == G0_LTOP_ARCHITECTURE and (
                    int(metrics.get("task_query_elements", 0)) <= 0
                    or float(metrics.get("task_query_norm", 0.0)) <= 0
                ):
                    raise RuntimeError(
                        "LTOP G0 produced no gradient in its native task-query rows"
                    )
                return metrics

            gradient_metrics.update(
                _distributed_rank_local_call(
                    action=audit_gradients,
                    phase=f"native-g0-step-{global_step}-gradient-audit",
                    rank=rank,
                    dist_module=dist,
                )
            )

            def clip_gradients() -> float:
                clipped = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                    foreach=False,
                )
                full_tensor = getattr(clipped, "full_tensor", None)
                if callable(full_tensor):
                    clipped = full_tensor()
                clipped_item = getattr(clipped, "item", None)
                if not callable(clipped_item):
                    raise TypeError("native G0 clipped gradient norm is not a scalar tensor")
                clipped_value: Any = clipped_item()
                if isinstance(clipped_value, bool) or not isinstance(clipped_value, (int, float)):
                    raise TypeError(
                        "native G0 clipped gradient norm did not produce a numeric scalar"
                    )
                return float(clipped_value)

            gradient_metrics["preclip_global_norm"] = _distributed_rank_local_call(
                action=clip_gradients,
                phase=f"native-g0-step-{global_step}-gradient-clip",
                rank=rank,
                dist_module=dist,
            )

            def apply_optimizer_step() -> int:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                return global_step + 1

            return _distributed_rank_local_call(
                action=apply_optimizer_step,
                phase=f"native-g0-step-{global_step}-optimizer-step",
                rank=rank,
                dist_module=dist,
            )

        def finish_optimizer_attempt() -> bool:
            finished = attempt.finish(optimizer_attempt)
            if not finished:
                raise RuntimeError("native G0 optimizer update overflowed or was skipped")
            return finished

        try:
            published = _distributed_rank_local_call(
                action=finish_optimizer_attempt,
                phase=f"native-g0-step-{global_step}-optimizer-finish",
                rank=rank,
                dist_module=dist,
            )
        except BaseException:
            optimizer.zero_grad(set_to_none=True)
            raise

        def build_post_update_state() -> dict[str, Any]:
            if not published:
                raise RuntimeError("native G0 optimizer transaction did not publish")
            step_time_s = time.perf_counter() - started
            rank_rng_state = _capture_rank_rng(torch, np, device=device)
            optimizer_summary = _validate_optimizer_state(
                optimizer,
                torch,
                expected_step=global_step + 1,
            )
            lane_snapshot = bank.serialize()
            saved_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=lane_snapshot,
                rank_rng_state=rank_rng_state,
                torch_module=torch,
            )
            extra_state = {
                "boundary_sha256": saved_boundary,
                "global_step": global_step + 1,
                "lane_snapshot": lane_snapshot,
                "model_family_sha256": model_family_sha256,
                "next_optimizer_step": global_step + 1,
                **optimizer_summary,
                "plan_sha256": plan.plan_sha256,
                "rank": rank,
                "rank_rng_state": rank_rng_state,
                "schema": G0_EXTRA_STATE_SCHEMA,
                "source_digest": collated.source_digest,
                "world_size": G0_WORLD_SIZE,
            }
            rank_report = {
                "rank": rank,
                "sample_keys": list(collated.routing.sample_keys),
                "lane_ids": list(collated.routing.lane_ids),
                "episode_keys": list(collated.routing.episode_keys),
                "frame_indices": list(collated.routing.frame_indices),
                "official_action_loss": float(forward.official_action_loss.detach().float().item()),
                "official_moe_regularizer": float(
                    forward.official_moe_regularizer.detach().float().item()
                ),
                "official_policy_loss": float(forward.official_total_loss.detach().float().item()),
                "gradient_metrics": gradient_metrics,
                "optimizer_state": optimizer_summary,
                "step_time_s": step_time_s,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "saved_boundary_sha256": saved_boundary,
                "loaded_boundary_sha256": loaded_boundary,
                "resume_boundary_verified": loaded_boundary is not None,
                "resume_runtime_rng_verified": resume_runtime_rng_verified,
            }
            return {
                "extra_state": extra_state,
                "maximum_peak_reserved_bytes": int(args.maximum_peak_reserved_gib * 1024**3),
                "rank_report": rank_report,
            }

        post_update_state = _distributed_rank_local_call(
            action=build_post_update_state,
            phase=f"native-g0-step-{global_step}-post-update-audit",
            rank=rank,
            dist_module=dist,
        )
        extra_state = post_update_state["extra_state"]
        maximum_peak_reserved_bytes = post_update_state["maximum_peak_reserved_bytes"]
        rank_report = post_update_state["rank_report"]
        local_memory_ok = torch.tensor(
            int(rank_report["peak_cuda_reserved_bytes"] <= maximum_peak_reserved_bytes),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(local_memory_ok, op=dist.ReduceOp.MIN)
        if not bool(local_memory_ok.item()):
            raise RuntimeError("native G0 exceeded the registered peak CUDA reservation budget")
        gathered: list[Any] = [None for _ in range(G0_WORLD_SIZE)]
        dist.all_gather_object(gathered, rank_report)
        report = None
        precheckpoint_error: list[str | None] = [None]
        if rank == 0:
            try:
                report = {
                    "schema": G0_REPORT_SCHEMA,
                    "phase": args.phase,
                    "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                    "patch_sha256": patch_report["patch_sha256"],
                    "patched_source_sha256": actual_hashes,
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "execution_contract_sha256": execution_sha256,
                    "implementation_sha256": implementation_sha256,
                    "model_family_sha256": model_family_sha256,
                    "plan_sha256": plan.plan_sha256,
                    "dataset_contract": dataset_contract_report,
                    "input_global_step": global_step,
                    "saved_global_step": global_step + 1,
                    "checkpoint_dir": str(output_checkpoint.resolve()),
                    "status": "PASS",
                    "full_shard": True,
                    "fsdp2_placement": args.fsdp2_placement,
                    "cuda_allocator": args.cuda_allocator,
                    "gradient_checkpointing": True,
                    "auxiliary_target_losses_enabled": False,
                    "parameter_storage": parameter_storage,
                    "maximum_peak_reserved_bytes": maximum_peak_reserved_bytes,
                    "parameter_manifest": {
                        "parameter_count": parameter_manifest.parameter_count,
                        "trainable_numel": parameter_manifest.trainable_numel,
                        "schema_sha256": parameter_manifest.schema_sha256,
                    },
                    "rank_reports": gathered,
                }
                validate_g0_report(
                    report,
                    schema=G0_REPORT_SCHEMA,
                    phase=args.phase,
                    source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
                    checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
                    world_size=G0_WORLD_SIZE,
                    require_checkpoint_copy=False,
                    expected_fsdp2_placement=args.fsdp2_placement,
                    expected_cuda_allocator=args.cuda_allocator,
                )
                require_checkpoint_write_capacity(checkpoint_root)
            except BaseException as error:
                precheckpoint_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(precheckpoint_error, src=0)
        if precheckpoint_error[0] is not None:
            raise RuntimeError(
                f"native G0 pre-checkpoint report validation failed: {precheckpoint_error[0]}"
            )
        _distributed_rank_local_call(
            action=lambda: checkpointer.save(
                str(staging_checkpoint),
                {"model": policy, "optimizer": optimizer, "extra_state": extra_state},
                global_steps=None,
            ),
            phase=f"native-g0-step-{global_step}-checkpoint-save",
            rank=rank,
            dist_module=dist,
        )
        publish_error: list[str | None] = [None]
        if rank == 0:
            try:
                if report is None:
                    raise RuntimeError("rank zero did not construct the native G0 report")
                validation_kwargs = {
                    "schema": G0_REPORT_SCHEMA,
                    "phase": args.phase,
                    "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "world_size": G0_WORLD_SIZE,
                    "expected_fsdp2_placement": args.fsdp2_placement,
                    "expected_cuda_allocator": args.cuda_allocator,
                }
                _write_text_durable(
                    staging_checkpoint / "native_g0_report.json",
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
                _fsync_tree(staging_checkpoint)
                os.replace(staging_checkpoint, output_checkpoint)
                descriptor = os.open(checkpoint_root, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                try:
                    validate_g0_report(report, **validation_kwargs)
                except BaseException:
                    os.replace(output_checkpoint, staging_checkpoint)
                    rollback_descriptor = os.open(checkpoint_root, os.O_RDONLY)
                    try:
                        os.fsync(rollback_descriptor)
                    finally:
                        os.close(rollback_descriptor)
                    raise
                try:
                    _write_text_durable(
                        report_path,
                        json.dumps(report, indent=2, sort_keys=True) + "\n",
                    )
                except BaseException:
                    os.replace(output_checkpoint, staging_checkpoint)
                    rollback_descriptor = os.open(checkpoint_root, os.O_RDONLY)
                    try:
                        os.fsync(rollback_descriptor)
                    finally:
                        os.close(rollback_descriptor)
                    raise
            except BaseException as error:
                publish_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(publish_error, src=0)
        if publish_error[0] is not None:
            raise RuntimeError(f"native G0 checkpoint publication failed: {publish_error[0]}")
        dist.barrier()
        if rank == 0:
            if report is None:
                raise RuntimeError("rank zero lost the native G0 report before publication")
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            print(payload, end="")
    finally:
        if run_lease is not None:
            run_lease.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
