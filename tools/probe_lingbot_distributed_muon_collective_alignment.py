#!/usr/bin/env python3
"""Bounded two-rank regression probe for LingBot DistributedMuon.

Run this tool with exactly two local CUDA processes, for example::

    torchrun --standalone --nproc-per-node=2 \
      tools/probe_lingbot_distributed_muon_collective_alignment.py \
      --source-checkout /path/to/LingBot-VLA \
      --output-json /path/to/probe-pass.json

Rank 0 receives a finite gradient for a 3D DTensor sharded along tensor
dimension 1. Rank 1 receives ``grad=None`` for the matching parameter. The
parameter must classify as LingBot's ``MOE_GATHER_3D`` path, and both ranks
must complete the optimizer collective, a post-step barrier, and a post-step
all-gather. Only rank 0 publishes a create-only atomic PASS report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import socket
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from picf_next.artifact_io import write_text_durable_exclusive
from tools.lingbot_vla2_runtime_helpers import clip_lingbot_distributed_l2_grad_norm_

SCHEMA = "picf-next.lingbot-distributed-muon-collective-alignment-probe.v1"
EXPECTED_CLASSIFICATION = "moe_gather_3d"
GLOBAL_SHAPE = (2, 4, 4)
SHARD_DIM = 1
LOCAL_SHAPE = (2, 2, 4)
DEFAULT_TIMEOUT_SECONDS = 90
MIN_TIMEOUT_SECONDS = 10
MAX_TIMEOUT_SECONDS = 300
WATCHDOG_EXIT_CODE = 124
_SHA256_LENGTH = 64


class ProbeContractError(RuntimeError):
    """Raised when the probe is not exercising its exact regression contract."""


class _DeadlineWatchdog:
    """Terminate a rank if a collective or teardown exceeds the wall-clock bound."""

    def __init__(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds
        self._cancelled = threading.Event()
        self._thread = threading.Thread(target=self._wait, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancelled.set()
        self._thread.join(timeout=1.0)

    def _wait(self) -> None:
        if self._cancelled.wait(self._timeout_seconds):
            return
        rank = os.environ.get("RANK", "unknown")
        print(
            f"rank {rank}: DistributedMuon collective probe exceeded "
            f"{self._timeout_seconds}s",
            file=sys.stderr,
            flush=True,
        )
        os._exit(WATCHDOG_EXIT_CODE)


def _bounded_timeout(value: str) -> int:
    try:
        timeout = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be an integer") from exc
    if not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"timeout must be in [{MIN_TIMEOUT_SECONDS}, {MAX_TIMEOUT_SECONDS}] seconds"
        )
    return timeout


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        required=True,
        help="LingBot-VLA source checkout containing lingbotvla/optim/muon.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Absent destination for the rank-0 atomic PASS report",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_bounded_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Overall per-rank wall-clock bound, including distributed teardown",
    )
    return parser.parse_args(argv)


def _parse_required_int(environ: Mapping[str, str], name: str) -> int:
    raw = environ.get(name)
    if raw is None:
        raise ProbeContractError(f"torchrun environment variable {name} is absent")
    try:
        return int(raw)
    except ValueError as exc:
        raise ProbeContractError(f"torchrun environment variable {name} is not an integer") from exc


def _require_launch_environment(environ: Mapping[str, str]) -> dict[str, int]:
    topology = {
        "world_size": _parse_required_int(environ, "WORLD_SIZE"),
        "rank": _parse_required_int(environ, "RANK"),
        "local_world_size": _parse_required_int(environ, "LOCAL_WORLD_SIZE"),
        "local_rank": _parse_required_int(environ, "LOCAL_RANK"),
    }
    if topology["world_size"] != 2 or topology["local_world_size"] != 2:
        raise ProbeContractError("probe requires torchrun with exactly two local ranks")
    if topology["rank"] not in (0, 1) or topology["local_rank"] not in (0, 1):
        raise ProbeContractError("probe rank identifiers must be exactly 0 and 1")
    if topology["rank"] != topology["local_rank"]:
        raise ProbeContractError("probe requires one two-GPU host with rank == local rank")
    return topology


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepare_paths(source_checkout: Path, output_json: Path) -> tuple[Path, Path, Path]:
    try:
        source = source_checkout.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ProbeContractError(f"source checkout does not exist: {source_checkout}") from exc
    if not source.is_dir():
        raise ProbeContractError(f"source checkout is not a directory: {source}")

    muon_source = (source / "lingbotvla" / "optim" / "muon.py").resolve(strict=False)
    if not muon_source.is_file():
        raise ProbeContractError(f"LingBot DistributedMuon source is absent: {muon_source}")
    if not _is_relative_to(muon_source, source):
        raise ProbeContractError("LingBot DistributedMuon source escapes the supplied checkout")

    output = output_json.expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise ProbeContractError(f"output path already exists: {output}")
    if _is_relative_to(output, source):
        raise ProbeContractError("output path must not overlap the supplied source checkout")
    if len({source, muon_source, output}) != 3:
        raise ProbeContractError("probe source and output paths must be distinct")
    return source, muon_source, output


def _validate_visible_devices(environ: Mapping[str, str]) -> None:
    raw = environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) < 2:
        raise ProbeContractError("CUDA_VISIBLE_DEVICES exposes fewer than two devices")
    if len(devices) != len(set(devices)):
        raise ProbeContractError("CUDA_VISIBLE_DEVICES contains duplicate device paths")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _muon_module_key(muon_source: Path) -> str:
    path_digest = hashlib.sha256(os.fsencode(muon_source)).hexdigest()[:20]
    return f"_picf_probe_lingbot_muon_{path_digest}"


def _load_muon_module(muon_source: Path) -> tuple[ModuleType, type[Any]]:
    module_key = _muon_module_key(muon_source)
    if module_key in sys.modules:
        raise ProbeContractError(f"duplicate LingBot source import path: {muon_source}")
    spec = importlib.util.spec_from_file_location(module_key, muon_source)
    if spec is None or spec.loader is None:
        raise ProbeContractError(f"cannot construct an import spec for {muon_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_key, None)
        raise
    imported_path = Path(module.__file__ or "").resolve(strict=True)
    if imported_path != muon_source:
        raise ProbeContractError(
            f"DistributedMuon resolved from {imported_path}, expected {muon_source}"
        )
    optimizer_class = getattr(module, "DistributedMuon", None)
    if not isinstance(optimizer_class, type):
        raise ProbeContractError("supplied checkout does not export DistributedMuon")
    if not callable(getattr(module, "_classify_param", None)):
        raise ProbeContractError("supplied checkout does not expose Muon parameter classification")
    if not hasattr(module, "_KIND_MOE_GATHER_3D"):
        raise ProbeContractError("supplied checkout does not expose MOE_GATHER_3D classification")
    return module, optimizer_class


def _require_finite_tree(value: object, *, path: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProbeContractError(f"{path} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProbeContractError(f"{path} contains a non-string mapping key")
            _require_finite_tree(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tree(child, path=f"{path}[{index}]")
        return
    raise ProbeContractError(f"{path} contains unsupported value type {type(value).__name__}")


def _require_sha256(value: object, *, field: str, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ProbeContractError(f"{field} is not a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ProbeContractError(f"{field} is not a lowercase SHA-256 digest")


def _tensor_digest(tensor: Any, torch: Any) -> str:
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    if not torch.is_tensor(local):
        raise ProbeContractError("tensor digest received a non-tensor value")
    if local.is_sparse:
        raise ProbeContractError("tensor digest does not accept sparse values")
    finite = torch.isfinite(local.detach()).all()
    if not bool(finite.item()):
        raise ProbeContractError("probe encountered a non-finite tensor")
    cpu = local.detach().to(device="cpu").contiguous()
    metadata = json.dumps(
        {"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(metadata)
    digest.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _optimizer_state_summary(optimizer: Any, parameter: Any, torch: Any) -> dict[str, Any]:
    present = parameter in optimizer.state
    if not present:
        return {"digest": None, "entry_count": 0, "keys": [], "present": False}
    state = optimizer.state.get(parameter)
    if not isinstance(state, Mapping):
        raise ProbeContractError("optimizer state entry is not a mapping")
    encoded: dict[str, object] = {}
    for key in sorted(state):
        if not isinstance(key, str):
            raise ProbeContractError("optimizer state contains a non-string key")
        value = state[key]
        if torch.is_tensor(value) or hasattr(value, "to_local"):
            encoded[key] = {"tensor_sha256": _tensor_digest(value, torch)}
        elif isinstance(value, bool) or value is None or isinstance(value, (str, int)):
            encoded[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ProbeContractError("optimizer state contains a non-finite scalar")
            encoded[key] = value
        else:
            raise ProbeContractError(
                f"optimizer state contains unsupported value type {type(value).__name__}"
            )
    serialized = json.dumps(
        encoded,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "digest": hashlib.sha256(serialized).hexdigest(),
        "entry_count": len(state),
        "keys": sorted(state),
        "present": True,
    }


def _validate_preflight_reports(reports: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    if len(reports) != 2:
        raise ProbeContractError("preflight all-gather did not return exactly two rank reports")
    normalized = [dict(report) for report in reports]
    _require_finite_tree(normalized, path="preflight_reports")
    ranks = [report.get("rank") for report in normalized]
    local_ranks = [report.get("local_rank") for report in normalized]
    if set(ranks) != {0, 1} or len(set(ranks)) != 2:
        raise ProbeContractError("preflight reports contain duplicate or missing global ranks")
    if set(local_ranks) != {0, 1} or len(set(local_ranks)) != 2:
        raise ProbeContractError("preflight reports contain duplicate or missing local ranks")
    for report in normalized:
        if report.get("world_size") != 2 or report.get("local_world_size") != 2:
            raise ProbeContractError("preflight report has the wrong two-rank topology")
        if report.get("rank") != report.get("local_rank"):
            raise ProbeContractError("preflight report is not a two-rank single-host launch")
        if report.get("backend") != "nccl":
            raise ProbeContractError("preflight report is not using NCCL")
    for field in ("hostname", "source_checkout", "muon_source_sha256", "output_json"):
        if len({report.get(field) for report in normalized}) != 1:
            raise ProbeContractError(f"preflight ranks disagree on {field}")
    if len({report.get("device_index") for report in normalized}) != 2:
        raise ProbeContractError("preflight reports contain duplicate CUDA device indices")
    for field in ("device_name", "device_capability", "device_total_memory", "torch_version"):
        if len({json.dumps(report.get(field), sort_keys=True) for report in normalized}) != 1:
            raise ProbeContractError(
                f"probe requires homogeneous hardware; ranks disagree on {field}"
            )
    return sorted(normalized, key=lambda report: int(report["rank"]))


def _validate_rank_reports(reports: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    if len(reports) != 2:
        raise ProbeContractError("post-step all-gather did not return exactly two rank reports")
    normalized = [dict(report) for report in reports]
    _require_finite_tree(normalized, path="rank_reports")
    ranks = [report.get("rank") for report in normalized]
    if set(ranks) != {0, 1} or len(set(ranks)) != 2:
        raise ProbeContractError("post-step reports contain duplicate or missing ranks")
    by_rank = {int(report["rank"]): report for report in normalized}
    for rank, report in by_rank.items():
        if report.get("local_rank") != rank:
            raise ProbeContractError("post-step rank and local-rank identities differ")
        if report.get("classification") != EXPECTED_CLASSIFICATION:
            raise ProbeContractError("parameter did not use MOE_GATHER_3D")
        if report.get("global_shape") != list(GLOBAL_SHAPE):
            raise ProbeContractError("post-step report has the wrong global tensor shape")
        if report.get("local_shape") != list(LOCAL_SHAPE):
            raise ProbeContractError("post-step report has the wrong local shard shape")
        if report.get("shard_dimensions") != [SHARD_DIM] or SHARD_DIM == 0:
            raise ProbeContractError("post-step report is not sharded on the required nonzero dim")
        _require_sha256(report.get("parameter_before_sha256"), field="parameter_before_sha256")
        _require_sha256(report.get("parameter_after_sha256"), field="parameter_after_sha256")
        _require_sha256(
            report.get("gradient_sha256"),
            field="gradient_sha256",
            allow_none=True,
        )
        _require_sha256(
            report.get("gradient_after_clip_sha256"),
            field="gradient_after_clip_sha256",
            allow_none=True,
        )
        _require_sha256(
            report.get("optimizer_state_after_sha256"),
            field="optimizer_state_after_sha256",
            allow_none=True,
        )
        if report.get("optimizer_state_before_entry_count") != 0:
            raise ProbeContractError("optimizer state was not empty before the step")

    rank_zero = by_rank[0]
    if rank_zero.get("gradient_present") is not True:
        raise ProbeContractError("rank 0 did not receive its finite gradient")
    if rank_zero.get("gradient_sha256") is None:
        raise ProbeContractError("rank 0 gradient digest is absent")
    if rank_zero.get("gradient_after_clip_sha256") is None:
        raise ProbeContractError("rank 0 clipped gradient digest is absent")
    if rank_zero["gradient_sha256"] == rank_zero["gradient_after_clip_sha256"]:
        raise ProbeContractError("rank 0 gradient was not clipped")
    if not 0.0 < float(rank_zero.get("local_gradient_norm_after_clip", 0.0)) <= 1.000001:
        raise ProbeContractError("rank 0 clipped gradient norm is outside the contract")
    if rank_zero.get("parameter_changed") is not True:
        raise ProbeContractError("rank 0 parameter did not change")
    if rank_zero["parameter_before_sha256"] == rank_zero["parameter_after_sha256"]:
        raise ProbeContractError("rank 0 before/after parameter digests are identical")
    if rank_zero.get("optimizer_state_after_present") is not True:
        raise ProbeContractError("rank 0 did not create optimizer state")
    if not isinstance(rank_zero.get("optimizer_state_after_entry_count"), int):
        raise ProbeContractError("rank 0 optimizer state count is invalid")
    if int(rank_zero["optimizer_state_after_entry_count"]) <= 0:
        raise ProbeContractError("rank 0 optimizer state is empty after the step")
    if rank_zero.get("optimizer_state_after_sha256") is None:
        raise ProbeContractError("rank 0 optimizer state digest is absent")

    grad_none = by_rank[1]
    if (
        grad_none.get("gradient_present") is not False
        or grad_none.get("gradient_sha256") is not None
        or grad_none.get("gradient_after_clip_sha256") is not None
        or grad_none.get("local_gradient_norm_after_clip") is not None
    ):
        raise ProbeContractError("rank 1 did not preserve the exact grad=None contract")
    if len({float(report["preclip_global_norm"]) for report in by_rank.values()}) != 1:
        raise ProbeContractError("ranks disagreed on the preclip global norm")
    if float(rank_zero["preclip_global_norm"]) <= 1.0:
        raise ProbeContractError("probe gradient did not exercise clipping")
    if grad_none.get("parameter_changed") is not False:
        raise ProbeContractError("grad-none rank parameter changed")
    if grad_none["parameter_before_sha256"] != grad_none["parameter_after_sha256"]:
        raise ProbeContractError("grad-none rank before/after parameter digests differ")
    if grad_none.get("optimizer_state_after_present") is not False:
        raise ProbeContractError("grad-none rank created optimizer state")
    if grad_none.get("optimizer_state_after_entry_count") != 0:
        raise ProbeContractError("grad-none rank optimizer state is not empty")
    if grad_none.get("optimizer_state_after_sha256") is not None:
        raise ProbeContractError("grad-none rank published an optimizer state digest")
    return [by_rank[0], by_rank[1]]


def _atomic_write_json_create(path: Path, payload: Mapping[str, object]) -> None:
    _require_finite_tree(payload)
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        write_text_durable_exclusive(path, encoded, encoding="ascii")
    except FileExistsError as exc:
        raise ProbeContractError(f"refusing to replace existing output: {path}") from exc


def _require_cuda_hardware(torch: Any, topology: Mapping[str, int]) -> tuple[Any, dict[str, Any]]:
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise ProbeContractError("probe requires a PyTorch build with NCCL support")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise ProbeContractError("probe requires at least two visible CUDA devices")
    local_rank = topology["local_rank"]
    if local_rank >= torch.cuda.device_count():
        raise ProbeContractError("local rank does not map to a visible CUDA device")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_runtime": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "device_index": local_rank,
        "device_name": properties.name,
        "device_total_memory": int(properties.total_memory),
    }


def _build_parameter_and_gradient(
    *,
    local_rank: int,
    module: ModuleType,
    torch: Any,
    mesh: Any,
    dtensor_class: type[Any],
    shard_class: type[Any],
) -> tuple[Any, Any | None, str, list[int]]:
    if SHARD_DIM <= 0:
        raise ProbeContractError("probe shard dimension must be nonzero")
    global_value = torch.linspace(
        -0.75,
        0.75,
        steps=math.prod(GLOBAL_SHAPE),
        dtype=torch.float32,
        device=torch.device("cuda", local_rank),
    ).reshape(GLOBAL_SHAPE)
    local_value = global_value.narrow(
        SHARD_DIM,
        local_rank * LOCAL_SHAPE[SHARD_DIM],
        LOCAL_SHAPE[SHARD_DIM],
    ).clone()
    if tuple(local_value.shape) != LOCAL_SHAPE:
        raise ProbeContractError("constructed local parameter shard has the wrong shape")
    placements = (shard_class(SHARD_DIM),)
    distributed = dtensor_class.from_local(
        local_value,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
        shape=torch.Size(GLOBAL_SHAPE),
        stride=tuple(global_value.stride()),
    )
    parameter = torch.nn.Parameter(distributed)
    if not isinstance(parameter, dtensor_class):
        raise ProbeContractError("parameter construction did not preserve DTensor identity")
    if tuple(parameter.shape) != GLOBAL_SHAPE:
        raise ProbeContractError("constructed DTensor has the wrong global shape")
    if tuple(parameter.to_local().shape) != LOCAL_SHAPE:
        raise ProbeContractError("constructed DTensor has the wrong local shape")
    shard_dimensions = [
        placement.dim
        for placement in parameter.placements
        if isinstance(placement, shard_class)
    ]
    if shard_dimensions != [SHARD_DIM]:
        raise ProbeContractError("constructed DTensor has the wrong shard placement")
    classification = module._classify_param(parameter)
    expected = module._KIND_MOE_GATHER_3D
    if classification != expected or classification != EXPECTED_CLASSIFICATION:
        raise ProbeContractError(
            f"constructed parameter classified as {classification!r}, expected MOE_GATHER_3D"
        )

    gradient = None
    if local_rank == 0:
        local_gradient = torch.linspace(
            0.125,
            1.0,
            steps=math.prod(LOCAL_SHAPE),
            dtype=torch.float32,
            device=local_value.device,
        ).reshape(LOCAL_SHAPE)
        if not bool(torch.isfinite(local_gradient).all().item()):
            raise ProbeContractError("rank 0 gradient is non-finite")
        gradient = dtensor_class.from_local(
            local_gradient,
            device_mesh=mesh,
            placements=placements,
            run_check=False,
            shape=torch.Size(GLOBAL_SHAPE),
            stride=tuple(global_value.stride()),
        )
        parameter.grad = gradient
    else:
        parameter.grad = None
    return parameter, gradient, classification, shard_dimensions


def _run_probe(
    *,
    source_checkout: Path,
    muon_source: Path,
    output_json: Path,
    topology: Mapping[str, int],
    timeout_seconds: int,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    device, hardware = _require_cuda_hardware(torch, topology)
    module, optimizer_class = _load_muon_module(muon_source)
    process_group_timeout = max(MIN_TIMEOUT_SECONDS, timeout_seconds - 5)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=process_group_timeout),
        device_id=device,
    )
    rank = topology["rank"]
    local_rank = topology["local_rank"]
    source_sha256 = _sha256_file(muon_source)
    preflight = {
        "backend": dist.get_backend(),
        "hostname": socket.gethostname(),
        "local_rank": local_rank,
        "local_world_size": topology["local_world_size"],
        "muon_source_sha256": source_sha256,
        "output_json": str(output_json),
        "rank": rank,
        "source_checkout": str(source_checkout),
        "torch_version": torch.__version__,
        "world_size": topology["world_size"],
        **hardware,
    }
    preflight_reports: list[dict[str, Any] | None] = [None, None]
    dist.all_gather_object(preflight_reports, preflight)
    validated_preflight = _validate_preflight_reports(
        [report for report in preflight_reports if report is not None]
    )
    dist.barrier(device_ids=[local_rank])

    mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("moe_shard",))
    parameter, gradient, classification, shard_dimensions = _build_parameter_and_gradient(
        local_rank=local_rank,
        module=module,
        torch=torch,
        mesh=mesh,
        dtensor_class=DTensor,
        shard_class=Shard,
    )
    parameter_before_sha256 = _tensor_digest(parameter, torch)
    gradient_sha256 = _tensor_digest(gradient, torch) if gradient is not None else None

    optimizer = optimizer_class(
        [parameter],
        lr=1e-3,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=1,
        adjust_lr_fn=None,
    )
    state_before = _optimizer_state_summary(optimizer, parameter, torch)
    if state_before["present"] or state_before["entry_count"] != 0:
        raise ProbeContractError("optimizer state is not fresh before the regression step")

    preclip_global_norm = clip_lingbot_distributed_l2_grad_norm_(
        (parameter,),
        1.0,
        device=device,
        dist_module=dist,
        torch_module=torch,
        error_if_nonfinite=True,
    )
    gradient_after_clip_sha256 = _tensor_digest(gradient, torch) if gradient is not None else None
    local_gradient_norm_after_clip = (
        float(gradient.to_local().float().norm().item()) if gradient is not None else None
    )
    started = time.monotonic()
    optimizer.step()
    torch.cuda.synchronize(device)
    step_elapsed_seconds = time.monotonic() - started
    if not math.isfinite(step_elapsed_seconds):
        raise ProbeContractError("optimizer step elapsed time is non-finite")

    parameter_after_sha256 = _tensor_digest(parameter, torch)
    state_after = _optimizer_state_summary(optimizer, parameter, torch)
    local_report = {
        "classification": classification,
        "gradient_present": gradient is not None,
        "gradient_after_clip_sha256": gradient_after_clip_sha256,
        "gradient_sha256": gradient_sha256,
        "global_shape": list(parameter.shape),
        "local_rank": local_rank,
        "local_gradient_norm_after_clip": local_gradient_norm_after_clip,
        "local_shape": list(parameter.to_local().shape),
        "optimizer_state_after_entry_count": state_after["entry_count"],
        "optimizer_state_after_present": state_after["present"],
        "optimizer_state_after_sha256": state_after["digest"],
        "optimizer_state_before_entry_count": state_before["entry_count"],
        "parameter_after_sha256": parameter_after_sha256,
        "parameter_before_sha256": parameter_before_sha256,
        "parameter_changed": parameter_before_sha256 != parameter_after_sha256,
        "preclip_global_norm": preclip_global_norm,
        "rank": rank,
        "shard_dimensions": shard_dimensions,
        "step_elapsed_seconds": step_elapsed_seconds,
    }
    _require_finite_tree(local_report, path=f"rank_{rank}_report")

    dist.barrier(device_ids=[local_rank])
    rank_reports: list[dict[str, Any] | None] = [None, None]
    dist.all_gather_object(rank_reports, local_report)
    validated_ranks = _validate_rank_reports(
        [report for report in rank_reports if report is not None]
    )
    report = {
        "classification": EXPECTED_CLASSIFICATION,
        "global_shape": list(GLOBAL_SHAPE),
        "muon_source": str(muon_source),
        "muon_source_sha256": source_sha256,
        "optimizer": {
            "class": optimizer_class.__name__,
            "learning_rate": 1e-3,
            "momentum": 0.95,
            "nesterov": True,
            "newton_schulz_steps": 1,
            "weight_decay": 0.0,
        },
        "post_step_collectives": {"all_gather_completed": True, "barrier_completed": True},
        "preflight_rank_reports": validated_preflight,
        "rank_reports": validated_ranks,
        "schema": SCHEMA,
        "shard_dimension": SHARD_DIM,
        "source_checkout": str(source_checkout),
        "status": "PASS",
        "world_size": 2,
    }
    _require_finite_tree(report)
    if rank == 0:
        _atomic_write_json_create(output_json, report)
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True), flush=True)
    # The required post-step barrier and report all-gather already completed.
    # Do not add a publication-only collective: rank 0 owns the create-only
    # artifact, and its nonzero exit status is sufficient to fail torchrun if
    # publication does not complete.
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    topology = _require_launch_environment(os.environ)
    _validate_visible_devices(os.environ)
    source_checkout, muon_source, output_json = _prepare_paths(
        args.source_checkout,
        args.output_json,
    )
    watchdog = _DeadlineWatchdog(args.timeout_seconds)
    watchdog.start()
    try:
        _run_probe(
            source_checkout=source_checkout,
            muon_source=muon_source,
            output_json=output_json,
            topology=topology,
            timeout_seconds=args.timeout_seconds,
        )
    finally:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        finally:
            watchdog.cancel()


if __name__ == "__main__":
    main()
