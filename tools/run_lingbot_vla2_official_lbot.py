#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Run a bounded, source-disjoint LingBot CALVIN action baseline.

LBOT executes the released LingBot policy and optimizer without installing or
invoking PICF.  It consumes the same frozen CALVIN stream and evaluation sample
manifest as P1 so later P3 comparisons change the architecture, not the data.
No physical-supervision sidecar, entity target, posterior, or checkpoint enters
this baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from itertools import combinations
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

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.capacity import require_persistent_run_root
from picf_next.lingbot_native.adr150_lbot_validation import (
    validate_full_modal_action_adoption,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config
from picf_next.lingbot_native.trainable_scope import (
    TRAINABLE_SCOPES,
    TRAINABLE_SCOPE_FROZEN_VISION_HOST,
    TRAINABLE_SCOPE_FULL_HOST,
    lingbot_trainable_scope_receipt,
)
from picf_next.training.run_lease import acquire_distributed_run_lease

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
        validate_prepared_native_source_with_muon_collective_hotfix,
        verify_muon_collective_hotfix,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
        build_lingbot_official_optimizer,
        build_lingbot_base_family_identity,
        clip_lingbot_distributed_l2_grad_norm_,
        load_lingbot_training_config,
        require_lingbot_exact_resume_contract,
        require_lingbot_released_action_sampling_steps,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.lingbot_vla2_ltop_stage_runtime import (
        LingBotVLA2LTOPStageRequest,
        prepare_lingbot_vla2_ltop_stage_transfer,
    )
    from tools.run_lingbot_vla2_ltop_g2_core import (
        _load_contracts,
        _local_representation_contract_items,
        _validate_representation_execution_provenance,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _distributed_rank_local_call,
        _distributed_gradient_metrics,
        _model_local_state_digest,
        _move_model_inputs,
        _validate_fsdp2_parameter_storage,
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
        validate_prepared_native_source_with_muon_collective_hotfix,
        verify_muon_collective_hotfix,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
        build_lingbot_official_optimizer,
        build_lingbot_base_family_identity,
        clip_lingbot_distributed_l2_grad_norm_,
        load_lingbot_training_config,
        require_lingbot_exact_resume_contract,
        require_lingbot_released_action_sampling_steps,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from lingbot_vla2_ltop_stage_runtime import (  # type: ignore[no-redef]
        LingBotVLA2LTOPStageRequest,
        prepare_lingbot_vla2_ltop_stage_transfer,
    )
    from run_lingbot_vla2_ltop_g2_core import (  # type: ignore[no-redef]
        _load_contracts,
        _local_representation_contract_items,
        _validate_representation_execution_provenance,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _distributed_rank_local_call,
        _distributed_gradient_metrics,
        _model_local_state_digest,
        _move_model_inputs,
        _validate_fsdp2_parameter_storage,
    )


LBOT_SUPPORTED_WORLD_SIZES = (2, 4)


def _runtime_lbot_world_size(environment: Mapping[str, str] | None = None) -> int:
    """Resolve the registered single-host LBOT topology from torchrun."""

    values = os.environ if environment is None else environment
    raw = values.get("WORLD_SIZE", "2")
    try:
        world_size = int(raw)
    except ValueError as error:
        raise RuntimeError("LBOT WORLD_SIZE must be a canonical integer") from error
    if raw != str(world_size) or world_size not in LBOT_SUPPORTED_WORLD_SIZES:
        raise RuntimeError("LBOT supports exactly 2 or 4 ranks")
    return world_size


LBOT_WORLD_SIZE = _runtime_lbot_world_size()
LBOT_COMPARISON_ID = "lingbot-vla2-official-calvin-lbot"
PHYSICAL_LBOT_COMPARISON_ID = "lingbot-vla2-native-picf-full"
LBOT_REPORT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot.v1"
ADR150_LBOT_REPORT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot.v2"
ADR176_FROZEN_VISION_LBOT_REPORT_SCHEMA = (
    "picf-next.lingbot-vla2-official-calvin-lbot-frozen-vision.v1"
)
ADR172_EXACT_LBOT_REPORT_SCHEMA = "picf-next.adr172-exact-matched-lbot.v1"
LBOT_CURVE_SNAPSHOT_SCHEMA = "picf-next.lingbot-vla2-official-calvin-lbot-snapshot.v1"
_MAXIMUM_LBOT_STEPS = 20
_MAXIMUM_LBOT_CURVE_STEPS = 2_000
_ADR172_EXACT_STEPS = 256
_ADR172_EXACT_SEED = 20260813
_ADR172_EXACT_EVALUATION_STEPS = tuple(range(0, _ADR172_EXACT_STEPS + 1, 32))
_ADR172_DCP_MODEL_PREFIX = "state.model."
_ADR172_ALLOWED_DCP_EXTRA_PREFIXES = ("state.model.model.qwenvl_with_expert.picf_native_graph.",)
LINGBOT_ATTENTION_IMPLEMENTATIONS = ("eager", "flex_cached")
LINGBOT_COMPILE_MODES = ("disabled", "upstream-default")


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = _environment_path("PICF_LINGBOT_NATIVE_SOURCE") or (
        root / CHECKOUT_RELATIVE_PATH
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, default=source_default)
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument("--training-config", type=Path, default=None)
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
        "--checkpoint-dir", type=Path, default=_environment_path("PICF_CHECKPOINT_DIR")
    )
    parser.add_argument(
        "--processor-dir", type=Path, default=_environment_path("PICF_PROCESSOR_DIR")
    )
    parser.add_argument("--dataset-split", type=Path, default=_environment_path("PICF_DATASET_DIR"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=_environment_path("PICF_DATASET_MANIFEST"),
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=_environment_path("PICF_LINGBOT_NORM_STATS")
    )
    parser.add_argument("--run-dir", type=Path, default=_environment_path("PICF_RUN_DIR"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stream-plan", type=Path)
    parser.add_argument("--stream-plan-sha256")
    parser.add_argument("--representation-split", type=Path)
    parser.add_argument("--representation-split-sha256")
    parser.add_argument("--evaluation-plan", type=Path)
    parser.add_argument("--evaluation-plan-sha256")
    parser.add_argument("--full-modal-action-adoption", type=Path)
    parser.add_argument(
        "--adr172-exact-stream",
        action="store_true",
        help="Run the exact two-rank ADR172 fixed-replay matched LBOT baseline.",
    )
    parser.add_argument("--runtime-hotfix", type=Path)
    parser.add_argument("--stage-checkpoint", type=Path)
    parser.add_argument("--g2-report", type=Path)
    parser.add_argument("--execution-contract", type=Path)
    parser.add_argument("--offline-labels", type=Path)
    parser.add_argument(
        "--physical-event-stream",
        action="store_true",
        help="Use ADR149's unique raw-episode events and deterministic prompt overlay.",
    )
    parser.add_argument(
        "--minimum-future-source-frames",
        type=int,
        default=0,
        help="Restrict physical events to sources with this many consecutive future frames.",
    )
    parser.add_argument("--maximum-control-tokens", type=int, default=64)
    parser.add_argument(
        "--evaluation-steps",
        help="Comma-separated post-update steps; curve mode also requires step zero.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--maximum-peak-reserved-gib", type=float, default=39.0)
    parser.add_argument(
        "--attention-implementation",
        choices=LINGBOT_ATTENTION_IMPLEMENTATIONS,
        default="eager",
        help=(
            "Select the released LingBot joint-attention backend. The default keeps "
            "historical controls unchanged; matched ADR-207 runs explicitly select "
            "flex_cached to match the candidate execution contract."
        ),
    )
    parser.add_argument(
        "--lingbot-compile-mode",
        choices=LINGBOT_COMPILE_MODES,
        default="disabled",
        help="Match the released FSDP-then-whole-model torch.compile training path.",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=TRAINABLE_SCOPES,
        default=TRAINABLE_SCOPE_FULL_HOST,
        help="Keep the released full forward while optionally freezing the Qwen vision tower.",
    )
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    )
    parser.add_argument("--cuda-allocator", choices=CUDA_ALLOCATOR_MODES, default="native")
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    if args.output is None and args.run_dir is not None:
        args.output = args.run_dir / f"official_lbot_steps_{args.steps}.json"
    return args


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _evaluation_steps(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("LBOT evaluation steps must be comma-separated integers") from error
    if not result or result != tuple(sorted(set(result))) or any(item < 0 for item in result):
        raise ValueError("LBOT evaluation steps must be unique sorted non-negative integers")
    return result


def _evaluation_replay_seed(plan_sha256: str, sample_key: str) -> int:
    _require_sha256("LBOT evaluation plan SHA-256", plan_sha256)
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("entity evaluation sample key must be a nonempty string")
    return int.from_bytes(
        hashlib.sha256(f"{plan_sha256}\0{sample_key}".encode("ascii")).digest()[:8],
        "big",
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_schema(value: Any, *, name: str) -> dict[str, Any]:
    properties = getattr(value, "properties", None)
    dtype = getattr(properties, "dtype", None)
    if dtype is None:
        dtype = getattr(value, "dtype", None)
    size = getattr(value, "size", None)
    if callable(size):
        size = size()
    if size is None:
        size = getattr(value, "shape", None)
    if dtype is None or size is None:
        raise TypeError(f"ADR172 shared tensor metadata is not tensor-like: {name}")
    try:
        shape = [int(dimension) for dimension in size]
    except (TypeError, ValueError) as error:
        raise TypeError(f"ADR172 shared tensor shape is invalid: {name}") from error
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"ADR172 shared tensor shape is negative: {name}")
    return {"dtype": str(dtype), "shape": shape}


def _audit_adr172_shared_checkpoint_metadata(
    *,
    checkpoint_metadata: Mapping[str, Any],
    shared_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a G2 DCP can populate exactly the graph-free LingBot state."""

    if not checkpoint_metadata or not shared_state:
        raise ValueError("ADR172 shared checkpoint audit requires non-empty tensor mappings")
    if any(not isinstance(name, str) or not name for name in checkpoint_metadata):
        raise TypeError("ADR172 DCP metadata contains an invalid tensor name")
    if any(not isinstance(name, str) or not name for name in shared_state):
        raise TypeError("ADR172 LingBot state contains an invalid tensor name")

    expected = {
        f"{_ADR172_DCP_MODEL_PREFIX}{name}": _tensor_schema(value, name=name)
        for name, value in shared_state.items()
    }
    checkpoint_keys = frozenset(checkpoint_metadata)
    expected_keys = frozenset(expected)
    missing = sorted(expected_keys - checkpoint_keys)
    extra = sorted(checkpoint_keys - expected_keys)
    rejected_extra = sorted(
        name
        for name in extra
        if not any(name.startswith(prefix) for prefix in _ADR172_ALLOWED_DCP_EXTRA_PREFIXES)
    )
    if missing:
        raise RuntimeError(f"ADR172 G2 DCP omits shared LingBot tensors: {missing[:16]}")
    if rejected_extra:
        raise RuntimeError(f"ADR172 G2 DCP contains non-PICF extra tensors: {rejected_extra[:16]}")
    if not extra:
        raise RuntimeError("ADR172 G2 DCP contains no PICF-only tensors to exclude")

    shared_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for checkpoint_name in sorted(expected):
        state_name = checkpoint_name.removeprefix(_ADR172_DCP_MODEL_PREFIX)
        checkpoint_schema = _tensor_schema(
            checkpoint_metadata[checkpoint_name],
            name=checkpoint_name,
        )
        expected_schema = expected[checkpoint_name]
        row = {"name": state_name, **checkpoint_schema}
        shared_rows.append(row)
        if checkpoint_schema != expected_schema:
            mismatches.append(
                {
                    "name": state_name,
                    "checkpoint": checkpoint_schema,
                    "policy": expected_schema,
                }
            )
    if mismatches:
        raise RuntimeError(f"ADR172 G2 shared tensor schemas differ: {mismatches[:8]}")

    extra_rows = [
        {
            "name": name,
            **_tensor_schema(checkpoint_metadata[name], name=name),
        }
        for name in extra
    ]
    return {
        "status": "PASS",
        "checkpoint_tensor_count": len(checkpoint_keys),
        "shared_tensor_count": len(shared_rows),
        "shared_tensor_schema_sha256": _canonical_sha256(shared_rows),
        "missing_shared_tensors": missing,
        "extra_tensor_count": len(extra_rows),
        "extra_tensors": extra_rows,
        "extra_tensor_schema_sha256": _canonical_sha256(extra_rows),
        "rejected_extra_tensors": rejected_extra,
        "allowed_extra_prefixes": list(_ADR172_ALLOWED_DCP_EXTRA_PREFIXES),
    }


def _adr172_exact_input_hashes(model_inputs: Mapping[str, Any]) -> dict[str, str]:
    required = ("actions", "action_is_pad", "noise", "time")
    missing = [name for name in required if name not in model_inputs]
    if missing:
        raise KeyError(f"ADR172 matched LBOT input omits tensors: {missing}")
    tensor_hashes = {name: _tensor_sha256(model_inputs[name]) for name in required}
    action_targets_sha256 = _canonical_sha256(
        {
            "actions": tensor_hashes["actions"],
            "action_is_pad": tensor_hashes["action_is_pad"],
        }
    )
    return {
        "actions_sha256": tensor_hashes["actions"],
        "action_is_pad_sha256": tensor_hashes["action_is_pad"],
        "action_targets_sha256": action_targets_sha256,
        "noise_sha256": tensor_hashes["noise"],
        "timestep_sha256": tensor_hashes["time"],
    }


def _adr172_exact_input_receipt(
    *,
    sample_key: str,
    replay_seed: int,
    source_digest: str,
    model_inputs: Mapping[str, Any],
    model_inputs_sha256: str,
) -> dict[str, Any]:
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("ADR172 matched LBOT sample key must be non-empty")
    if isinstance(replay_seed, bool) or not isinstance(replay_seed, int) or replay_seed < 0:
        raise ValueError("ADR172 matched LBOT replay seed must be non-negative")
    _require_sha256("ADR172 matched LBOT source digest", source_digest)
    _require_sha256("ADR172 matched LBOT model-input digest", model_inputs_sha256)
    hashes = _adr172_exact_input_hashes(model_inputs)
    comparison = {
        "sample_key": sample_key,
        "sample_sha256": hashlib.sha256(sample_key.encode("utf-8")).hexdigest(),
        "replay_seed": replay_seed,
        "source_digest": source_digest,
        "model_inputs_sha256": model_inputs_sha256,
        **hashes,
    }
    return {
        **comparison,
        "sample_action_noise_timestep_sha256": _canonical_sha256(comparison),
    }


def _require_adr172_exact_input_receipt(
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    if not phase:
        raise ValueError("ADR172 matched LBOT input receipt phase must be non-empty")
    if actual != expected:
        raise RuntimeError(f"ADR172 exact LBOT {phase} input differs from its fixed replay")
    return dict(actual)


def _distributed_phase_error(
    *,
    error: BaseException | None,
    phase: str,
    rank: int,
    dist_module: Any,
) -> None:
    """Exchange rank-local evaluation failures before the next FSDP forward."""

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
    gathered: list[Any] = [None for _ in range(LBOT_WORLD_SIZE)]
    dist_module.all_gather_object(gathered, local)
    failures = tuple(item for item in gathered if item is not None)
    if failures:
        rendered = "; ".join(
            f"rank {item['rank']} {item['phase']} {item['type']}: {item['message']}"
            for item in failures
        )
        raise RuntimeError(f"distributed LBOT action evaluation failed: {rendered}")


def _validate_args(args: argparse.Namespace) -> None:
    validate_fsdp2_placement(args.fsdp2_placement)
    required = {
        "checkpoint-dir": args.checkpoint_dir,
        "processor-dir": args.processor_dir,
        "dataset-split": args.dataset_split,
        "dataset-manifest": args.dataset_manifest,
        "norm-stats": args.norm_stats,
        "run-dir": args.run_dir,
        "output": args.output,
    }
    absent = sorted(name for name, value in required.items() if value is None)
    if absent:
        raise ValueError(f"LBOT paths are absent: {absent}")
    exact_stage_values = (
        args.stage_checkpoint,
        args.g2_report,
        args.execution_contract,
        args.offline_labels,
    )
    if args.adr172_exact_stream:
        if args.runtime_hotfix is None or any(
            value is None for value in exact_stage_values
        ):
            raise ValueError("ADR172 exact LBOT requires all stage and execution contracts")
    elif any(value is not None for value in exact_stage_values):
        raise ValueError("ADR172 stage or execution inputs require --adr172-exact-stream")
    curve_values = (
        args.stream_plan,
        args.stream_plan_sha256,
        args.representation_split,
        args.representation_split_sha256,
        args.evaluation_plan,
        args.evaluation_plan_sha256,
    )
    curve_mode = any(value is not None for value in curve_values)
    if curve_mode and any(value is None for value in curve_values):
        raise ValueError("LBOT curve mode requires all plan, split, and digest arguments")
    if args.runtime_hotfix is not None and not (
        args.adr172_exact_stream or curve_mode
    ):
        raise ValueError("LingBot Muon hotfix requires exact or matched-curve mode")
    if args.adr172_exact_stream and curve_mode:
        raise ValueError("ADR172 exact LBOT cannot also consume the legacy curve contracts")
    if args.adr172_exact_stream and (
        args.physical_event_stream or args.full_modal_action_adoption is not None
    ):
        raise ValueError("ADR172 exact LBOT excludes PICF physical and adoption sidecars")
    if args.full_modal_action_adoption is not None and (
        not curve_mode or not args.physical_event_stream
    ):
        raise ValueError(
            "ADR-150 full-modal adoption can bind only the physical matched curve baseline"
        )
    if (
        isinstance(args.minimum_future_source_frames, bool)
        or not isinstance(args.minimum_future_source_frames, int)
        or args.minimum_future_source_frames < 0
    ):
        raise ValueError("LBOT minimum_future_source_frames must be a non-negative integer")
    if args.minimum_future_source_frames and not args.physical_event_stream:
        raise ValueError(
            "LBOT minimum_future_source_frames requires --physical-event-stream"
        )
    evaluation_steps = _evaluation_steps(args.evaluation_steps)
    if args.adr172_exact_stream:
        if LBOT_WORLD_SIZE != 2:
            raise ValueError("ADR172 exact LBOT requires world_size=2")
        if args.steps != _ADR172_EXACT_STEPS:
            raise ValueError(f"ADR172 exact LBOT requires {_ADR172_EXACT_STEPS} steps")
        if args.seed != _ADR172_EXACT_SEED:
            raise ValueError(f"ADR172 exact LBOT requires seed {_ADR172_EXACT_SEED}")
        if evaluation_steps != _ADR172_EXACT_EVALUATION_STEPS:
            raise ValueError(
                "ADR172 exact LBOT evaluation steps must match the candidate 32-step cadence"
            )
        if args.maximum_control_tokens != 8:
            raise ValueError("ADR172 exact LBOT requires maximum_control_tokens=8")
        if args.fsdp2_placement != FSDP2_SELECTIVE_EMBEDDING_OFFLOAD:
            raise ValueError("ADR172 exact LBOT requires selective embedding offload")
        if args.trainable_scope != TRAINABLE_SCOPE_FULL_HOST:
            raise ValueError("ADR172 exact LBOT requires the released full-host trainable scope")
        if not math.isclose(args.learning_rate, 1e-4, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("ADR172 exact LBOT requires learning_rate=1e-4")
        if not math.isclose(args.max_grad_norm, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("ADR172 exact LBOT requires max_grad_norm=1.0")
    elif curve_mode:
        if not 1 <= args.steps <= _MAXIMUM_LBOT_CURVE_STEPS:
            raise ValueError(
                f"LBOT curve mode is bounded to 1..{_MAXIMUM_LBOT_CURVE_STEPS} optimizer steps"
            )
        if not evaluation_steps or evaluation_steps[0] != 0 or evaluation_steps[-1] != args.steps:
            raise ValueError("LBOT curve evaluation must include step zero and the final step")
        if any(step > args.steps for step in evaluation_steps):
            raise ValueError("LBOT curve evaluation exceeds the optimizer-step budget")
    else:
        if args.evaluation_steps is not None:
            raise ValueError("LBOT smoke mode cannot register curve evaluation steps")
        if not 1 <= args.steps <= _MAXIMUM_LBOT_STEPS:
            raise ValueError(f"LBOT is bounded to 1..{_MAXIMUM_LBOT_STEPS} optimizer steps")

    files = (
        args.patch,
        args.training_config,
        args.robot_config,
        args.data_config,
        args.dataset_manifest,
        args.norm_stats,
        *(
            (args.full_modal_action_adoption,)
            if args.full_modal_action_adoption is not None
            else ()
        ),
        *((args.stream_plan,) if args.stream_plan is not None else ()),
        *((args.representation_split,) if args.representation_split is not None else ()),
        *((args.evaluation_plan,) if args.evaluation_plan is not None else ()),
        *((args.runtime_hotfix,) if args.runtime_hotfix is not None else ()),
        *((args.g2_report,) if args.g2_report is not None else ()),
        *((args.execution_contract,) if args.execution_contract is not None else ()),
        *((args.offline_labels,) if args.offline_labels is not None else ()),
    )
    directories = (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
        args.run_dir,
        *((args.stage_checkpoint,) if args.stage_checkpoint is not None else ()),
    )
    if any(Path(path).is_symlink() or not Path(path).is_file() for path in files):
        raise FileNotFoundError("one or more LBOT source/config/data files are absent")
    if any(not Path(path).is_dir() for path in directories):
        raise FileNotFoundError("one or more LBOT source/model/data directories are absent")
    if args.output.parent.resolve() != args.run_dir.resolve():
        raise ValueError("LBOT output must be a direct child of its persistent run directory")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    integers = (
        args.steps,
        args.seed,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("LBOT integer controls must be Python integers")
    if args.seed < 0:
        raise ValueError("LBOT seed must be non-negative")
    if args.seed > 0xFFFFFFFF - (LBOT_WORLD_SIZE - 1):
        raise ValueError("LBOT rank seeds must fit NumPy's uint32 domain")
    if (
        isinstance(args.maximum_control_tokens, bool)
        or not isinstance(args.maximum_control_tokens, int)
        or args.maximum_control_tokens <= 0
    ):
        raise ValueError("LBOT maximum_control_tokens must be a positive integer")
    for name in (
        "learning_rate",
        "max_grad_norm",
        "maximum_peak_reserved_gib",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"LBOT {name} must be finite and positive")
    if curve_mode:
        _require_sha256("stream plan file SHA-256", args.stream_plan_sha256)
        _require_sha256(
            "representation split file SHA-256",
            args.representation_split_sha256,
        )
        _require_sha256(
            "LBOT evaluation plan file SHA-256",
            args.evaluation_plan_sha256,
        )


def _implementation_provenance(root: Path) -> tuple[dict[str, str], str]:
    relative = (
        "references/patches/lingbot_vla2_picf_native.patch",
        "src/picf_next/lingbot_native/calvin.py",
        "src/picf_next/lingbot_native/adr150_lbot_validation.py",
        "src/picf_next/lingbot_native/entity_evaluation_plan.py",
        "src/picf_next/lingbot_native/representation_split.py",
        "src/picf_next/lingbot_native/trainable_scope.py",
        "src/picf_next/lingbot_native/training.py",
        "tools/lingbot_vla2_runtime_helpers.py",
        "tools/run_lingbot_vla2_official_lbot.py",
    )
    hashes = {path: _sha256(root / path) for path in relative}
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return hashes, digest


def _picf_graph_installed(policy: Any) -> bool:
    """Read the actual patched LingBot host mount instead of a root proxy."""

    try:
        host = policy.model.qwenvl_with_expert
        graph = host.picf_native_graph
    except AttributeError as error:
        raise RuntimeError(
            "LBOT policy differs from the patched host graph mount contract"
        ) from error
    return graph is not None


def _float(value: Any) -> float:
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise RuntimeError("LBOT report encountered a non-finite scalar")
    return result


def _summarize_action_partition(
    samples: list[dict[str, Any]],
    *,
    partition: str,
) -> dict[str, Any]:
    selected = [sample for sample in samples if sample.get("partition") == partition]
    if not selected:
        raise ValueError(f"LBOT action partition is empty: {partition}")

    def mean(name: str) -> float:
        values = [float(sample[name]) for sample in selected]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"LBOT action partition contains non-finite {name}")
        return sum(values) / len(values)

    return {
        "sample_count": len(selected),
        "mean_action_loss": mean("action_loss"),
        "mean_total_loss": mean("total_loss"),
        "mean_moe_regularizer": mean("moe_regularizer"),
        "mean_forward_seconds": mean("forward_seconds"),
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    curve_mode = args.stream_plan is not None
    exact_mode = bool(args.adr172_exact_stream)
    evaluation_mode = curve_mode or exact_mode
    registered_evaluation_steps = _evaluation_steps(args.evaluation_steps)
    require_persistent_run_root(args.run_dir)
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("LBOT CUDA allocator pre-bootstrap differs from parsed arguments")

    root = Path(__file__).resolve().parents[1]
    exact_stage_contract = None
    exact_execution: dict[str, Any] | None = None
    exact_labels: dict[str, Any] | None = None
    if exact_mode:
        exact_stage_contract = prepare_lingbot_vla2_ltop_stage_transfer(
            LingBotVLA2LTOPStageRequest(
                source_checkout=args.source_checkout,
                patch=args.patch,
                runtime_hotfix=args.runtime_hotfix,
                training_config=args.training_config,
                checkpoint_dir=args.checkpoint_dir,
                processor_dir=args.processor_dir,
                stage_checkpoint=args.stage_checkpoint,
                g2_report=args.g2_report,
                seed=args.seed,
                maximum_control_tokens=args.maximum_control_tokens,
                fsdp2_placement=args.fsdp2_placement,
            )
        )
        patch_report = exact_stage_contract.patch_report
        prepared_source = exact_stage_contract.prepared_source
        exact_execution, exact_labels = _load_contracts(
            args.execution_contract,
            args.offline_labels,
            expected_item_count=16,
        )
    elif args.runtime_hotfix is not None:
        patch_report = verify_muon_collective_hotfix(
            root=root,
            checkout=args.source_checkout,
            check_apply=True,
        )
        prepared_source = validate_prepared_native_source_with_muon_collective_hotfix(
            checkout=args.source_checkout,
            patch_path=args.patch,
            hotfix_path=args.runtime_hotfix,
        )
        expected_hashes = patch_report.get("patched_source_sha256")
        actual_hashes = prepared_source.get("patched_source_sha256")
        if not isinstance(expected_hashes, dict) or actual_hashes != expected_hashes:
            raise RuntimeError(
                "LBOT LingBot source differs from immutable native-plus-Muon replay"
            )
    else:
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
        actual_hashes = prepared_source.get("patched_source_sha256")
        if not isinstance(expected_hashes, dict) or actual_hashes != expected_hashes:
            raise RuntimeError("LBOT LingBot source differs from immutable patch replay")
    actual_hashes = prepared_source.get("patched_source_sha256")
    if not isinstance(actual_hashes, dict):
        raise RuntimeError("LBOT prepared source omitted its patched-source digest mapping")
    implementation_files, implementation_sha256 = _implementation_provenance(root)
    full_modal_action_adoption = (
        None
        if args.full_modal_action_adoption is None
        else validate_full_modal_action_adoption(
            json.loads(args.full_modal_action_adoption.read_text(encoding="utf-8"))
        )
    )

    if os.environ.get("WORLD_SIZE") != str(LBOT_WORLD_SIZE):
        raise RuntimeError("LBOT torchrun world-size contract differs")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(LBOT_WORLD_SIZE):
        raise RuntimeError("LBOT requires all ranks on one host")

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
    from torch.distributed.checkpoint import FileSystemReader

    from picf_next.data.calvin import (
        CalvinDatasetIndex,
        CalvinPhysicalTransitionDataset,
        CalvinStatefulTransitionDataset,
    )
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.dataset_manifest import (
        DatasetFileManifest,
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.calvin import (
        CollatedNativeCALVINBatch,
        PlannedNativeCALVINReplayBatch,
        build_native_calvin_episode_domain,
        build_native_calvin_physical_episode_domain,
        build_native_calvin_physical_stream_plan,
        build_native_calvin_stream_plan,
        build_native_calvin_replay_batch,
        build_planned_native_calvin_batch,
        collate_native_calvin_training_batch,
        materialize_native_flow_randomness,
    )
    from picf_next.lingbot_native.entity_evaluation_plan import (
        ENTITY_EVALUATION_PARTITIONS,
        EntityEvaluationPlan,
        build_distributed_entity_evaluation_schedule,
        build_entity_evaluation_plan,
    )
    from picf_next.lingbot_native.training import (
        audit_native_optimizer_coverage,
        run_official_policy_diagnostic_forward,
        run_official_policy_training_forward,
    )
    from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
    from picf_next.training.control import load_frozen_episode_stream_plan

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    run_lease = None
    try:
        run_lease = acquire_distributed_run_lease(args.run_dir, rank=rank, distributed=dist)
        if torch.cuda.device_count() != LBOT_WORLD_SIZE:
            raise RuntimeError(f"LBOT process must see exactly {LBOT_WORLD_SIZE} CUDA devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("LBOT requires A100 devices with at least 39 GiB each")

        artifact_contract: list[Any] = [None]
        if rank == 0:
            try:
                artifact_contract[0] = {
                    "status": "PASS",
                    "checkpoint": validate_checkpoint(args.checkpoint_dir),
                    "processor": validate_processor(args.processor_dir),
                }
            except BaseException as error:
                artifact_contract[0] = {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(artifact_contract, src=0)
        artifact_contract_report = artifact_contract[0]
        if (
            not isinstance(artifact_contract_report, dict)
            or artifact_contract_report.get("status") != "PASS"
        ):
            raise RuntimeError(f"LBOT model artifact contract failed: {artifact_contract_report}")
        checkpoint_report = artifact_contract_report["checkpoint"]
        processor_report = artifact_contract_report["processor"]
        if not isinstance(checkpoint_report, dict) or not isinstance(processor_report, dict):
            raise RuntimeError("LBOT artifact validators returned non-mapping reports")

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
                    raise ValueError("LBOT CALVIN manifest and normalization differ")
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
            raise RuntimeError(f"LBOT dataset contract failed: {dataset_contract_report}")
        dataset_manifest = (
            rank_zero_manifest
            if rank_zero_manifest is not None
            else load_dataset_file_manifest(args.dataset_manifest.resolve())
        )
        if exact_mode:
            _validate_representation_execution_provenance(
                exact_execution,
                dataset_manifest_file_sha256=_sha256(args.dataset_manifest),
                dataset_tree_sha256=dataset_manifest.tree_sha256,
            )
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=dataset_manifest.dataset_id,
            dataset_revision=dataset_manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=dataset_manifest,
        )
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        init_parallel_state(
            dp_size=LBOT_WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=LBOT_WORLD_SIZE,
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
        native_model_patch_sha256 = patch_report.get(
            "native_patch_sha256", patch_report.get("patch_sha256")
        )
        if not isinstance(native_model_patch_sha256, str):
            raise RuntimeError("LingBot native model patch identity is absent")
        lingbot_base_family = build_lingbot_base_family_identity(
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            native_patch_sha256=native_model_patch_sha256,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_report=checkpoint_report,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_report=processor_report,
            attention_implementation=args.attention_implementation,
            trainable_scope=args.trainable_scope,
            optimizer_contract=asdict(optimizer_contract),
            maximum_control_tokens=args.maximum_control_tokens,
        )
        lingbot_base_family_sha256 = lingbot_base_family["artifact_sha256"]
        merged, source_data_mapping = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
        )
        merged["use_cache"] = False
        merged["use_compile"] = args.lingbot_compile_mode == "upstream-default"
        merged["attention_implementation"] = args.attention_implementation
        merged["vit_attn_implementation"] = "eager"
        merged["freeze_vision_encoder"] = args.trainable_scope == TRAINABLE_SCOPE_FROZEN_VISION_HOST
        merged["train_expert_only"] = False
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        require_lingbot_released_action_sampling_steps(config)
        if bool(config.train_expert_only):
            raise RuntimeError("LBOT forbids the expert-only trainable scope")
        expected_frozen_vision = args.trainable_scope == TRAINABLE_SCOPE_FROZEN_VISION_HOST
        if bool(config.freeze_vision_encoder) != expected_frozen_vision:
            raise RuntimeError("LBOT visual trainable scope differs from the CLI contract")
        qwen_config = AutoConfig.from_pretrained(  # nosec B615
            args.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        patch_size = int(qwen_config.vision_config.patch_size)
        merge_size = int(qwen_config.vision_config.spatial_merge_size)
        if patch_size <= 0 or merge_size <= 0:
            raise RuntimeError("LBOT loaded invalid Qwen vision geometry")
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
        alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        if exact_mode and alignment_teacher_prune != exact_stage_contract.g2_report.get(
            "alignment_teacher_prune"
        ):
            raise RuntimeError("ADR172 exact LBOT shared teacher-head topology differs from G2")
        policy.train()
        if _picf_graph_installed(policy):
            raise RuntimeError("LBOT policy unexpectedly contains a PICF graph")
        trainable_scope_receipt = lingbot_trainable_scope_receipt(
            policy,
            scope=args.trainable_scope,
        )
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=optimizer_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=args.fsdp2_placement == FSDP2_CPU_OFFLOAD,
            enable_shared_embedding_offload=(
                args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
            ),
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
        if _picf_graph_installed(policy):
            raise RuntimeError("LBOT FSDP policy unexpectedly contains a PICF graph")
        parameter_storage_before_restore = _validate_fsdp2_parameter_storage(
            policy,
            torch,
            expected_placement=args.fsdp2_placement,
        )
        parameter_storage = parameter_storage_before_restore
        shared_checkpoint_restore: dict[str, Any] | None = None
        if exact_mode:
            checkpoint_metadata = _distributed_rank_local_call(
                action=lambda: (
                    FileSystemReader(args.stage_checkpoint.resolve() / "model")
                    .read_metadata()
                    .state_dict_metadata
                ),
                phase="adr172-matched-lbot-dcp-metadata-read",
                rank=rank,
                dist_module=dist,
            )
            shared_checkpoint_audit = _distributed_rank_local_call(
                action=lambda: _audit_adr172_shared_checkpoint_metadata(
                    checkpoint_metadata=checkpoint_metadata,
                    shared_state=policy.state_dict(),
                ),
                phase="adr172-matched-lbot-shared-key-audit",
                rank=rank,
                dist_module=dist,
            )
            released_local_digest = _model_local_state_digest(policy, torch)
            restored_state = {"model": policy}
            checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
            _distributed_rank_local_call(
                action=lambda: checkpointer.load(
                    str(args.stage_checkpoint.resolve()),
                    restored_state,
                    allow_partial_load=True,
                ),
                phase="adr172-matched-lbot-shared-only-dcp-load",
                rank=rank,
                dist_module=dist,
            )
            if set(restored_state) != {"model"} or restored_state["model"] is not policy:
                raise RuntimeError("ADR172 shared-only DCP load changed the model state boundary")
            if _picf_graph_installed(policy):
                raise RuntimeError("ADR172 shared-only DCP load installed a PICF graph")
            torch.cuda.synchronize(device)
            parameter_storage = _validate_fsdp2_parameter_storage(
                policy,
                torch,
                expected_placement=args.fsdp2_placement,
            )
            loaded_local_digest = _model_local_state_digest(policy, torch)
            gathered_shared_digests: list[Any] = [None for _ in range(LBOT_WORLD_SIZE)]
            dist.all_gather_object(
                gathered_shared_digests,
                {
                    "rank": rank,
                    "released_model_local_state_sha256": released_local_digest,
                    "loaded_shared_model_local_state_sha256": loaded_local_digest,
                },
            )
            if sorted(item["rank"] for item in gathered_shared_digests) != list(
                range(LBOT_WORLD_SIZE)
            ):
                raise RuntimeError("ADR172 shared restore rank digests are incomplete")
            loaded_digest_rows = [
                {
                    "rank": item["rank"],
                    "loaded_shared_model_local_state_sha256": item[
                        "loaded_shared_model_local_state_sha256"
                    ],
                }
                for item in sorted(gathered_shared_digests, key=lambda item: item["rank"])
            ]
            shared_checkpoint_restore = {
                "status": "PASS",
                "load_scope": "shared-lingbot-tensors-only",
                "allow_partial_load": True,
                "stage_checkpoint": str(args.stage_checkpoint.resolve()),
                "stage_checkpoint_inventory": exact_stage_contract.checkpoint_inventory,
                "g2_report_sha256": exact_stage_contract.g2_report_sha256,
                "metadata_audit": shared_checkpoint_audit,
                "parameter_storage_before_restore": parameter_storage_before_restore,
                "parameter_storage_after_restore": parameter_storage,
                "rank_local_state_digests": sorted(
                    gathered_shared_digests,
                    key=lambda item: item["rank"],
                ),
                "loaded_shared_tensor_digest_sha256": _canonical_sha256(loaded_digest_rows),
            }
            policy.requires_grad_(True)
        if args.lingbot_compile_mode == "upstream-default":
            policy = torch.compile(policy)
        lingbot_compile_receipt = {
            "mode": args.lingbot_compile_mode,
            "enabled": args.lingbot_compile_mode == "upstream-default",
            "ordering": "fsdp2_then_whole_model_compile_then_optimizer",
            "backend": "torch_compile_upstream_default",
        }
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

        rank_seed = args.seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
        evaluation_dataset = CalvinStatefulTransitionDataset(
            index,
            action_horizon=config.chunk_size,
        )
        dataset = (
            CalvinPhysicalTransitionDataset(index, action_horizon=config.chunk_size)
            if args.physical_event_stream
            else evaluation_dataset
        )
        representation_split: RepresentationTrialSplit | None = None
        evaluation_plan: EntityEvaluationPlan | None = None
        exact_local_items: tuple[tuple[dict[str, Any], dict[str, Any]], ...] = ()
        exact_runtime_schedule: dict[str, Any] | None = None
        exact_global_item_ids: list[str] | None = None
        plan = None
        if exact_mode:
            if exact_execution is None or exact_labels is None:
                raise AssertionError("validated ADR172 execution contracts disappeared")
            exact_local_items, exact_runtime_schedule = _local_representation_contract_items(
                exact_execution,
                exact_labels,
                rank=rank,
            )
            gathered_item_ids: list[Any] = [None for _ in range(LBOT_WORLD_SIZE)]
            dist.all_gather_object(
                gathered_item_ids,
                [item["item_id"] for item, _label in exact_local_items],
            )
            exact_global_item_ids = sorted(
                item_id for rank_items in gathered_item_ids for item_id in rank_items
            )
            expected_item_ids = sorted(item["item_id"] for item in exact_execution["items"])
            if exact_global_item_ids != expected_item_ids:
                raise RuntimeError("ADR172 exact LBOT rank rebind is not disjoint and exhaustive")
            training_plan_sha256 = exact_runtime_schedule["sha256"]
        elif curve_mode:
            if (
                _sha256(args.stream_plan) != args.stream_plan_sha256
                or _sha256(args.representation_split) != args.representation_split_sha256
                or _sha256(args.evaluation_plan) != args.evaluation_plan_sha256
            ):
                raise ValueError("LBOT curve contract file SHA-256 differs")
            representation_split = RepresentationTrialSplit.load(args.representation_split)
            plan = load_frozen_episode_stream_plan(
                args.stream_plan,
                episodes=build_native_calvin_episode_domain(
                    evaluation_dataset,
                    excluded_source_episode_indices=(
                        representation_split.stream_domain_excluded_source_episode_indices
                    ),
                )
                if not args.physical_event_stream
                else build_native_calvin_physical_episode_domain(
                    dataset,
                    excluded_source_episode_indices=(
                        representation_split.stream_domain_excluded_source_episode_indices
                    ),
                    minimum_future_source_frames=args.minimum_future_source_frames,
                ),
            )
            evaluation_plan = EntityEvaluationPlan.load(args.evaluation_plan)
            if plan.total_steps < args.steps:
                raise ValueError("LBOT curve prefix exceeds the frozen stream plan")
            if plan.global_batch_size != LBOT_WORLD_SIZE:
                raise ValueError("LBOT curve stream plan has the wrong global batch")
            if representation_split.stream_plan_sha256 != plan.plan_sha256:
                raise ValueError("LBOT curve split and stream plan differ")
            if representation_split.training_steps != plan.total_steps:
                raise ValueError("LBOT curve split does not cover the complete stream plan")
            if evaluation_plan.representation_split_sha256 != representation_split.artifact_sha256:
                raise ValueError("LBOT evaluation plan belongs to another split")
            if (
                build_entity_evaluation_plan(
                    representation_split,
                    evaluation_dataset,
                    world_size=LBOT_WORLD_SIZE,
                )
                != evaluation_plan
            ):
                raise ValueError("LBOT evaluation plan is not reproducible from source")
            evaluation_sources = {item.source_episode_index for item in evaluation_plan.items}
            if evaluation_sources.intersection(
                representation_split.training_source_episode_indices
            ):
                raise ValueError("LBOT evaluation overlaps a training source episode")
            rank_forward_counts = [
                len(build_distributed_entity_evaluation_schedule(evaluation_plan, rank=item_rank))
                for item_rank in range(LBOT_WORLD_SIZE)
            ]
            if len(set(rank_forward_counts)) != 1:
                raise RuntimeError("LBOT evaluation padding failed to align collectives")
            training_plan_sha256 = plan.plan_sha256
        else:
            plan = (
                build_native_calvin_physical_stream_plan(
                    dataset,
                    comparison_id=PHYSICAL_LBOT_COMPARISON_ID,
                    seed=args.seed,
                    global_batch_size=LBOT_WORLD_SIZE,
                    total_steps=args.steps,
                    minimum_future_source_frames=args.minimum_future_source_frames,
                )
                if args.physical_event_stream
                else build_native_calvin_stream_plan(
                    dataset,
                    comparison_id=LBOT_COMPARISON_ID,
                    seed=args.seed,
                    global_batch_size=LBOT_WORLD_SIZE,
                    total_steps=args.steps,
                )
            )
            training_plan_sha256 = plan.plan_sha256
        model_family_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "architecture": "released_lingbot_vla2_action_policy",
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "implementation_sha256": implementation_sha256,
                    "plan_sha256": training_plan_sha256,
                    "shared_stage_sha256": (
                        None
                        if shared_checkpoint_restore is None
                        else shared_checkpoint_restore["loaded_shared_tensor_digest_sha256"]
                    ),
                    "trainable_scope_sha256": (
                        trainable_scope_receipt["scope_sha256"]
                        if args.trainable_scope == TRAINABLE_SCOPE_FROZEN_VISION_HOST
                        else None
                    ),
                    "attention_implementation": args.attention_implementation,
                    "lingbot_compile": lingbot_compile_receipt,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        data_mapping = (
            source_data_mapping
            if exact_mode
            else json.loads(args.data_config.read_text(encoding="utf-8"))
        )
        data_mapping_sha256 = _canonical_sha256(data_mapping)
        feature_transform = FeatureTransform(
            str(args.robot_config.resolve()),
            official_lingbot_data_config(data_mapping),
            config,
            processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(args.norm_stats.resolve()),
            use_depth_align=False,
            image_augment=False,
            use_future_image=False,
        )
        evaluation_snapshot_reports: list[dict[str, Any]] = []

        def collate_replay(
            planned: PlannedNativeCALVINReplayBatch,
            *,
            target_device: Any | None = None,
        ) -> CollatedNativeCALVINBatch:
            resolved_device = device if target_device is None else target_device
            collated = collate_native_calvin_training_batch(
                planned.training,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=planned.augmentation_seeds,
                source_digest=planned.source_digest,
            )
            collated = CollatedNativeCALVINBatch(
                model_inputs=_move_model_inputs(
                    collated.model_inputs,
                    device=resolved_device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                ),
                controls=collated.controls,
                routing=collated.routing,
                source_digest=collated.source_digest,
                structural_target_requests=collated.structural_target_requests,
                modalities=(
                    None
                    if collated.modalities is None
                    else collated.modalities.to(
                        device=resolved_device,
                        dtype=torch.bfloat16,
                    )
                ),
                prior_control_chunks=collated.prior_control_chunks,
            )
            return materialize_native_flow_randomness(collated, planned)

        def batch_to_device(batch: CollatedNativeCALVINBatch) -> CollatedNativeCALVINBatch:
            return CollatedNativeCALVINBatch(
                model_inputs=_move_model_inputs(
                    batch.model_inputs,
                    device=device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                ),
                controls=batch.controls,
                routing=batch.routing,
                source_digest=batch.source_digest,
                structural_target_requests=batch.structural_target_requests,
                modalities=(
                    None
                    if batch.modalities is None
                    else batch.modalities.to(device=device, dtype=torch.bfloat16)
                ),
                prior_control_chunks=batch.prior_control_chunks,
            )

        def model_inputs_sha256(model_inputs: Mapping[str, Any]) -> str:
            digest = hashlib.sha256()
            for name in sorted(model_inputs):
                value = model_inputs[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"LBOT model input is not a tensor: {name}")
                local = value.detach().to(device="cpu").contiguous()
                digest.update(name.encode("ascii"))
                digest.update(str(local.dtype).encode("ascii"))
                digest.update(json.dumps(list(local.shape), separators=(",", ":")).encode())
                digest.update(local.view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        exact_scenes: dict[str, list[dict[str, Any]]] = {
            "validation": [],
            "heldout": [],
        }
        if exact_mode:
            for item, _label in exact_local_items:
                planned = build_native_calvin_replay_batch(
                    evaluation_dataset,
                    sample_key=item["sample_key"],
                    lane_id=rank,
                    episode_instance_id=f"ltop-g3/{item['item_id']}",
                    optimizer_step=0,
                    replay_seed=item["replay_seed"],
                    device=device,
                    dtype=torch.bfloat16,
                )
                request = planned.training.structural_target_requests[0]
                if (
                    request.sample_key != item["sample_key"]
                    or request.source_global_index
                    != evaluation_dataset.source_global_index_by_key(item["sample_key"])
                    or request.source_sensor_hash_by_field != item["source_sensor_sha256"]
                ):
                    raise RuntimeError("ADR172 exact LBOT source item differs from execution")
                source_batch = collate_replay(
                    planned,
                    target_device=torch.device("cpu"),
                )
                receipt = _adr172_exact_input_receipt(
                    sample_key=item["sample_key"],
                    replay_seed=item["replay_seed"],
                    source_digest=source_batch.source_digest,
                    model_inputs=source_batch.model_inputs,
                    model_inputs_sha256=model_inputs_sha256(source_batch.model_inputs),
                )
                exact_scenes[item["partition"]].append(
                    {
                        "item": item,
                        "planned": planned,
                        "batch": source_batch,
                        "input_receipt": receipt,
                    }
                )
            for partition in exact_scenes:
                exact_scenes[partition].sort(key=lambda scene: scene["item"]["ordinal"])
                if len(exact_scenes[partition]) != 4:
                    raise RuntimeError(
                        f"ADR172 exact LBOT {partition} scene count differs from four"
                    )

        def snapshot_official_runtime_buffers() -> list[tuple[str, Any, Any]]:
            suffixes = ("avg_topk_sigmoid_score", "tokens_per_expert")
            snapshot = [
                (name, buffer, buffer.detach().clone())
                for name, buffer in policy.named_buffers()
                if name.endswith(suffixes)
            ]
            has_token_counts = any(name.endswith("tokens_per_expert") for name, _, _ in snapshot)
            if not snapshot or not has_token_counts:
                raise RuntimeError("LBOT found no released action-MoE runtime counters")
            return snapshot

        def restore_official_runtime_buffers(snapshot: list[tuple[str, Any, Any]]) -> None:
            with torch.no_grad():
                for name, buffer, saved in snapshot:
                    if buffer.shape != saved.shape or buffer.dtype != saved.dtype:
                        raise RuntimeError(f"LBOT runtime buffer changed contract: {name}")
                    buffer.copy_(saved)

        def run_action_evaluation(checkpoint_global_step: int) -> dict[str, Any]:
            if exact_mode:
                local_entries: tuple[Any, ...] = tuple(
                    [*exact_scenes["heldout"], *exact_scenes["validation"]]
                )
                local_padding = (False,) * len(local_entries)
            else:
                if evaluation_plan is None or representation_split is None:
                    raise RuntimeError("LBOT curve evaluation contract is absent")
                local_schedule = build_distributed_entity_evaluation_schedule(
                    evaluation_plan,
                    rank=rank,
                )
                local_entries = tuple(work.item for work in local_schedule)
                local_padding = tuple(work.is_padding for work in local_schedule)
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "official_lbot_evaluation_start",
                            "checkpoint_global_step": checkpoint_global_step,
                            "samples_per_rank": len(local_entries),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            local_samples: list[dict[str, Any]] = []
            with torch.no_grad():
                for local_index, (entry, is_padding) in enumerate(
                    zip(local_entries, local_padding, strict=True)
                ):
                    replay_seed: int | None = None
                    planned = None
                    collated = None
                    input_receipt: dict[str, Any] | None = None
                    item = entry["item"] if exact_mode else entry
                    prepare_error: BaseException | None = None
                    try:
                        if exact_mode:
                            replay_seed = item["replay_seed"]
                            planned = entry["planned"]
                            collated = batch_to_device(entry["batch"])
                            input_receipt = entry["input_receipt"]
                            input_receipt = _require_adr172_exact_input_receipt(
                                expected=input_receipt,
                                actual=_adr172_exact_input_receipt(
                                    sample_key=collated.routing.sample_keys[0],
                                    replay_seed=replay_seed,
                                    source_digest=collated.source_digest,
                                    model_inputs=collated.model_inputs,
                                    model_inputs_sha256=model_inputs_sha256(collated.model_inputs),
                                ),
                                phase="evaluation",
                            )
                        else:
                            replay_seed = _evaluation_replay_seed(
                                evaluation_plan.artifact_sha256,
                                item.sample_key,
                            )
                            planned = build_native_calvin_replay_batch(
                                evaluation_dataset,
                                sample_key=item.sample_key,
                                lane_id=rank,
                                episode_instance_id=(
                                    f"official-lbot-evaluation/{item.partition}/{item.ordinal}"
                                ),
                                optimizer_step=0,
                                replay_seed=replay_seed,
                                device=device,
                                dtype=torch.bfloat16,
                            )
                            collated = collate_replay(planned)
                    except BaseException as error:
                        prepare_error = error
                    _distributed_phase_error(
                        error=prepare_error,
                        phase=f"checkpoint-{checkpoint_global_step}-sample-{local_index}-prepare",
                        rank=rank,
                        dist_module=dist,
                    )
                    if replay_seed is None or planned is None or collated is None:
                        raise RuntimeError("LBOT action evaluation preparation vanished")

                    result = None
                    forward_seconds = 0.0
                    forward_error: BaseException | None = None
                    runtime_snapshot: list[tuple[str, Any, Any]] | None = None
                    try:
                        runtime_snapshot = snapshot_official_runtime_buffers()
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        with torch.random.fork_rng(devices=[local_rank]):
                            torch.manual_seed(replay_seed)
                            torch.cuda.manual_seed(replay_seed)
                            result = run_official_policy_diagnostic_forward(
                                policy,
                                model_inputs=collated.model_inputs,
                            )
                        torch.cuda.synchronize(device)
                        forward_seconds = time.perf_counter() - started
                    except BaseException as error:
                        forward_error = error
                    finally:
                        if runtime_snapshot is not None:
                            restore_official_runtime_buffers(runtime_snapshot)
                    _distributed_phase_error(
                        error=forward_error,
                        phase=f"checkpoint-{checkpoint_global_step}-sample-{local_index}-forward",
                        rank=rank,
                        dist_module=dist,
                    )
                    if result is None:
                        raise RuntimeError("LBOT action evaluation forward vanished")

                    evidence: dict[str, Any] | None = None
                    evidence_error: BaseException | None = None
                    try:
                        request = collated.structural_target_requests[0]
                        partition = item["partition"] if exact_mode else item.partition
                        ordinal = item["ordinal"] if exact_mode else item.ordinal
                        sample_key = item["sample_key"] if exact_mode else item.sample_key
                        evidence = {
                            "checkpoint_global_step": checkpoint_global_step,
                            "partition": partition,
                            "ordinal": ordinal,
                            "rank": rank,
                            "item_id": item["item_id"] if exact_mode else None,
                            "task_key": request.task_key,
                            "segment_index": request.segment_index,
                            "source_episode_index": (
                                None if exact_mode else item.source_episode_index
                            ),
                            "source_global_index": request.source_global_index,
                            "transition_index": None if exact_mode else item.transition_index,
                            "sample_key": sample_key,
                            "replay_seed": replay_seed,
                            "source_digest": collated.source_digest,
                            "model_inputs_sha256": model_inputs_sha256(collated.model_inputs),
                            "total_loss": _float(result.official_total_loss),
                            "action_loss": _float(result.official_action_loss),
                            "moe_regularizer": _float(result.official_moe_regularizer),
                            "official_output_arity": len(result.official_outputs),
                            "forward_seconds": forward_seconds,
                        }
                        if exact_mode:
                            if input_receipt is None:
                                raise RuntimeError("ADR172 exact evaluation lost its input receipt")
                            evidence["input_receipt"] = input_receipt
                    except BaseException as error:
                        evidence_error = error
                    _distributed_phase_error(
                        error=evidence_error,
                        phase=f"checkpoint-{checkpoint_global_step}-sample-{local_index}-evidence",
                        rank=rank,
                        dist_module=dist,
                    )
                    if evidence is None:
                        raise RuntimeError("LBOT action evaluation evidence vanished")
                    if not is_padding:
                        local_samples.append(evidence)
                    if rank == 0 and (
                        local_index == 0
                        or local_index + 1 == len(local_entries)
                        or (local_index + 1) % 5 == 0
                    ):
                        print(
                            json.dumps(
                                {
                                    "event": "official_lbot_evaluation_progress",
                                    "checkpoint_global_step": checkpoint_global_step,
                                    "completed_per_rank": local_index + 1,
                                    "samples_per_rank": len(local_entries),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

            gathered_samples: list[Any] = [None for _ in range(LBOT_WORLD_SIZE)]
            dist.all_gather_object(gathered_samples, local_samples)
            publication: list[Any] = [None]
            if rank == 0:
                try:
                    samples = sorted(
                        (sample for rank_samples in gathered_samples for sample in rank_samples),
                        key=(
                            (lambda sample: (sample["partition"], int(sample["ordinal"])))
                            if exact_mode
                            else (lambda sample: int(sample["ordinal"]))
                        ),
                    )
                    expected_keys = (
                        [
                            item["sample_key"]
                            for item in sorted(
                                exact_execution["items"],
                                key=lambda value: (value["partition"], value["ordinal"]),
                            )
                        ]
                        if exact_mode
                        else [item.sample_key for item in evaluation_plan.items]
                    )
                    if [sample["sample_key"] for sample in samples] != expected_keys:
                        raise RuntimeError("LBOT curve evaluation sample set changed")
                    evaluation_inputs = [
                        (
                            sample["input_receipt"]
                            if exact_mode
                            else {
                                "sample_key": sample["sample_key"],
                                "source_digest": sample["source_digest"],
                                "model_inputs_sha256": sample["model_inputs_sha256"],
                            }
                        )
                        for sample in samples
                    ]
                    evaluation_input_sha256 = _canonical_sha256(evaluation_inputs)
                    payload = {
                        "schema": LBOT_CURVE_SNAPSHOT_SCHEMA,
                        "status": "PASS",
                        "checkpoint_global_step": checkpoint_global_step,
                        "architecture_identity": "released_lingbot_vla2_action_policy",
                        "picf_graph_installed": False,
                        "physical_sidecar_read": False,
                        "task_scorer_present": False,
                        "action_suffix_executed": True,
                        "posterior_present": False,
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "lingbot_base_family_sha256": lingbot_base_family_sha256,
                        "stream_plan_sha256": training_plan_sha256,
                        "representation_split_sha256": (
                            None
                            if representation_split is None
                            else representation_split.artifact_sha256
                        ),
                        "evaluation_plan_sha256": (
                            None if evaluation_plan is None else evaluation_plan.artifact_sha256
                        ),
                        "adr172_execution_contract_sha256": (
                            _sha256(args.execution_contract) if exact_mode else None
                        ),
                        "adr172_runtime_schedule_sha256": (
                            exact_runtime_schedule["sha256"] if exact_mode else None
                        ),
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "evaluation_inputs": evaluation_inputs if exact_mode else None,
                        "samples": samples,
                        "partition_summaries": {
                            partition: _summarize_action_partition(
                                samples,
                                partition=partition,
                            )
                            for partition in ENTITY_EVALUATION_PARTITIONS
                        },
                    }
                    artifact_sha256 = hashlib.sha256(
                        json.dumps(
                            payload,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    snapshot = {**payload, "artifact_sha256": artifact_sha256}
                    destination = (
                        args.run_dir / f"action_evaluation_step_{checkpoint_global_step:06d}.json"
                    )
                    write_text_durable_exclusive(
                        destination,
                        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                    )
                    publication[0] = {
                        "artifact_sha256": artifact_sha256,
                        "file_sha256": _sha256(destination),
                        "path": str(destination),
                        "checkpoint_global_step": checkpoint_global_step,
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "partition_summaries": snapshot["partition_summaries"],
                    }
                except BaseException as error:
                    publication[0] = {
                        "error": f"{type(error).__name__}: {error}",
                    }
            dist.broadcast_object_list(publication, src=0)
            if not isinstance(publication[0], dict) or "error" in publication[0]:
                raise RuntimeError(f"LBOT curve evaluation publication failed: {publication[0]}")
            dist.barrier()
            return publication[0]

        if 0 in registered_evaluation_steps:
            evaluation_snapshot_reports.append(run_action_evaluation(0))

        rank_steps: list[dict[str, Any]] = []
        maximum_peak_reserved_bytes = int(args.maximum_peak_reserved_gib * 1024**3)
        for optimizer_step in range(args.steps):
            input_receipt: dict[str, Any] | None = None
            if exact_mode:
                scene = exact_scenes["validation"][optimizer_step % len(exact_scenes["validation"])]
                planned = scene["planned"]
                input_receipt = scene["input_receipt"]
            else:
                planned = build_planned_native_calvin_batch(
                    plan,
                    dataset,
                    optimizer_step=optimizer_step,
                    rank=rank,
                    world_size=LBOT_WORLD_SIZE,
                    gradient_accumulation_steps=1,
                    accumulation_index=0,
                    device=device,
                    dtype=torch.bfloat16,
                    maximum_control_tokens=(
                        args.maximum_control_tokens if args.physical_event_stream else None
                    ),
                )
            collated = collate_native_calvin_training_batch(
                planned.training,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=planned.augmentation_seeds,
                source_digest=planned.source_digest,
            )
            collated = CollatedNativeCALVINBatch(
                model_inputs=_move_model_inputs(
                    collated.model_inputs,
                    device=device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                ),
                controls=collated.controls,
                routing=collated.routing,
                source_digest=collated.source_digest,
                structural_target_requests=collated.structural_target_requests,
                modalities=(
                    None
                    if collated.modalities is None
                    else collated.modalities.to(device=device, dtype=torch.bfloat16)
                ),
                prior_control_chunks=collated.prior_control_chunks,
            )
            collated = materialize_native_flow_randomness(collated, planned)
            if exact_mode:
                actual_input_receipt = _adr172_exact_input_receipt(
                    sample_key=collated.routing.sample_keys[0],
                    replay_seed=planned.replay_seed,
                    source_digest=collated.source_digest,
                    model_inputs=collated.model_inputs,
                    model_inputs_sha256=model_inputs_sha256(collated.model_inputs),
                )
                input_receipt = _require_adr172_exact_input_receipt(
                    expected=input_receipt,
                    actual=actual_input_receipt,
                    phase="training",
                )
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            try:
                result = run_official_policy_training_forward(
                    policy,
                    model_inputs=collated.model_inputs,
                )
                result.official_total_loss.backward()
            except BaseException:
                optimizer.zero_grad(set_to_none=True)
                raise

            gradient_metrics: dict[str, float | int | bool] = {}

            gradient_metrics.update(
                _distributed_gradient_metrics(
                    policy,
                    (
                        ("vlm_host", ".qwenvl."),
                        ("action_expert", ".qwen_expert."),
                        ("action_output", "action_out_proj"),
                    ),
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )
            )
            if not bool(gradient_metrics["all_finite"]):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("LBOT produced a non-finite gradient")
            gradient_metrics["preclip_global_norm"] = clip_lingbot_distributed_l2_grad_norm_(
                tuple(policy.parameters()),
                args.max_grad_norm,
                device=device,
                dist_module=dist,
                torch_module=torch,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_time_s = time.perf_counter() - started
            if float(gradient_metrics.get("vlm_host_norm", 0.0)) <= 0:
                raise RuntimeError("LBOT produced no gradient in the shared LingBot VLM")
            if float(gradient_metrics.get("action_expert_norm", 0.0)) <= 0:
                raise RuntimeError("LBOT produced no gradient in the LingBot action expert")
            if float(gradient_metrics.get("action_output_norm", 0.0)) <= 0:
                raise RuntimeError("LBOT produced no gradient in the action output projection")
            rank_steps.append(
                {
                    "global_step": optimizer_step + 1,
                    "sample_keys": list(collated.routing.sample_keys),
                    "lane_ids": list(collated.routing.lane_ids),
                    "frame_indices": list(collated.routing.frame_indices),
                    "reset": list(collated.routing.reset),
                    "source_digest": collated.source_digest,
                    "augmentation_seeds": list(planned.augmentation_seeds),
                    "flow_noise_seeds": list(planned.flow_noise_seeds),
                    "flow_timestep_seeds": list(planned.flow_timestep_seeds),
                    "input_receipt": input_receipt,
                    "total_loss": _float(result.official_total_loss),
                    "action_loss": _float(result.official_action_loss),
                    "moe_regularizer": _float(result.official_moe_regularizer),
                    "official_output_arity": len(result.official_outputs),
                    "picf_graph_installed": _picf_graph_installed(policy),
                    "gradient_metrics": gradient_metrics,
                    "step_time_s": step_time_s,
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
            if rank == 0:
                current = rank_steps[-1]
                print(
                    json.dumps(
                        {
                            "event": "official_lbot_step",
                            "global_step": current["global_step"],
                            "total_loss": current["total_loss"],
                            "action_loss": current["action_loss"],
                            "moe_regularizer": current["moe_regularizer"],
                            "gradient_metrics": current["gradient_metrics"],
                            "step_time_s": current["step_time_s"],
                            "peak_cuda_reserved_bytes": current["peak_cuda_reserved_bytes"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if rank_steps[-1]["peak_cuda_reserved_bytes"] > maximum_peak_reserved_bytes:
                raise RuntimeError("LBOT exceeded the registered CUDA reservation budget")
            completed_step = optimizer_step + 1
            if completed_step in registered_evaluation_steps:
                evaluation_snapshot_reports.append(run_action_evaluation(completed_step))

        gathered: list[Any] = [None for _ in range(LBOT_WORLD_SIZE)]
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "fixed_input_receipts": (
                    [
                        scene["input_receipt"]
                        for partition in ("heldout", "validation")
                        for scene in exact_scenes[partition]
                    ]
                    if exact_mode
                    else None
                ),
                "steps": rank_steps,
            },
        )
        publication_error: list[str | None] = [None]
        report: dict[str, Any] | None = None
        if rank == 0:
            try:
                if [item["checkpoint_global_step"] for item in evaluation_snapshot_reports] != list(
                    registered_evaluation_steps
                ):
                    raise RuntimeError("LBOT curve evaluation checkpoints changed")
                if (
                    evaluation_mode
                    and len(
                        {item["evaluation_input_sha256"] for item in evaluation_snapshot_reports}
                    )
                    != 1
                ):
                    raise RuntimeError("LBOT curve evaluation inputs differ across checkpoints")
                for step_index in range(args.steps):
                    sample_sets = [
                        set(rank_report["steps"][step_index]["sample_keys"])
                        for rank_report in gathered
                    ]
                    if any(
                        left.intersection(right) for left, right in combinations(sample_sets, 2)
                    ):
                        raise RuntimeError("LBOT distributed ranks consumed an overlapping sample")
                    for rank_report in gathered:
                        item = rank_report["steps"][step_index]
                        if item["global_step"] != step_index + 1:
                            raise RuntimeError("LBOT rank report has a non-contiguous step")
                        if item["official_output_arity"] != 11:
                            raise RuntimeError("LBOT rank changed the released output contract")
                        if item["picf_graph_installed"]:
                            raise RuntimeError("LBOT rank installed a PICF graph")
                        if exact_mode and item["input_receipt"] is None:
                            raise RuntimeError("ADR172 exact LBOT omitted a training input receipt")
                report = {
                    "schema": (
                        ADR172_EXACT_LBOT_REPORT_SCHEMA
                        if exact_mode
                        else (
                            ADR176_FROZEN_VISION_LBOT_REPORT_SCHEMA
                            if args.trainable_scope == TRAINABLE_SCOPE_FROZEN_VISION_HOST
                            else (
                                ADR150_LBOT_REPORT_SCHEMA
                                if full_modal_action_adoption is not None
                                else LBOT_REPORT_SCHEMA
                            )
                        )
                    ),
                    "status": "PASS",
                    "architecture_identity": "released_lingbot_vla2_action_policy",
                    "picf_graph_installed": False,
                    "physical_sidecar_read": False,
                    "task_scorer_present": False,
                    "action_suffix_executed": True,
                    "posterior_present": False,
                    "physical_event_stream": args.physical_event_stream,
                    "minimum_future_source_frames": args.minimum_future_source_frames,
                    "maximum_control_tokens": (
                        args.maximum_control_tokens
                        if args.physical_event_stream or exact_mode
                        else None
                    ),
                    "checkpoint_published": False,
                    "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                    "source_patch_sha256": patch_report.get(
                        "native_patch_sha256", patch_report.get("patch_sha256")
                    ),
                    "patched_source_sha256": actual_hashes,
                    "implementation_files": implementation_files,
                    "implementation_sha256": implementation_sha256,
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "checkpoint_assets": checkpoint_report["checkpoint_assets"],
                    "processor_revision": QWEN_PROCESSOR_REVISION,
                    "processor_assets": processor_report["processor_assets"],
                    "dataset_contract": dataset_contract_report,
                    "plan_sha256": training_plan_sha256,
                    "curve_mode": curve_mode,
                    "registered_evaluation_steps": list(registered_evaluation_steps),
                    "representation_split_sha256": (
                        None
                        if representation_split is None
                        else representation_split.artifact_sha256
                    ),
                    "evaluation_plan_sha256": (
                        None if evaluation_plan is None else evaluation_plan.artifact_sha256
                    ),
                    "evaluation_snapshots": evaluation_snapshot_reports,
                    "model_family_sha256": model_family_sha256,
                    "lingbot_base_family": lingbot_base_family,
                    "lingbot_base_family_sha256": lingbot_base_family_sha256,
                    "world_size": LBOT_WORLD_SIZE,
                    "steps": args.steps,
                    "seed": args.seed,
                    "max_grad_norm": args.max_grad_norm,
                    "official_output_arity": 11,
                    "optimizer_contract": asdict(optimizer_contract),
                    "qwen_vision_geometry": {
                        "patch_size": patch_size,
                        "spatial_merge_size": merge_size,
                    },
                    "fsdp2_placement": args.fsdp2_placement,
                    "cuda_allocator": args.cuda_allocator,
                    "attention_implementation": args.attention_implementation,
                    "lingbot_compile": lingbot_compile_receipt,
                    "gradient_checkpointing": True,
                    "parameter_storage": parameter_storage,
                    "parameter_manifest": {
                        "parameter_count": parameter_manifest.parameter_count,
                        "trainable_numel": parameter_manifest.trainable_numel,
                        "schema_sha256": parameter_manifest.schema_sha256,
                    },
                    "alignment_teacher_prune": alignment_teacher_prune,
                    "maximum_peak_reserved_bytes": maximum_peak_reserved_bytes,
                    "rank_reports": gathered,
                }
                if exact_mode:
                    report.update(
                        {
                            "adr172_exact_stream": True,
                            "runtime_hotfix_sha256": exact_stage_contract.runtime_hotfix_report[
                                "runtime_hotfix_sha256"
                            ],
                            "source_data_mapping_sha256": data_mapping_sha256,
                            "source_data_mapping_origin": str(args.training_config.resolve()),
                            "shared_checkpoint_restore": shared_checkpoint_restore,
                            "adr172_execution_contract": {
                                "execution_sha256": _sha256(args.execution_contract),
                                "offline_labels_sha256": _sha256(args.offline_labels),
                                "runtime_schedule": exact_runtime_schedule,
                                "global_item_ids": exact_global_item_ids,
                                "training_partition": "validation",
                                "training_scene_count_per_rank": 4,
                                "rank_assignment": "partition-local-ordinal-modulo-2",
                                "fixed_replay_seed": True,
                                "fixed_optimizer_step": 0,
                            },
                            "gradient_clip": "rank-invariant-distributed-l2",
                        }
                    )
                elif args.runtime_hotfix is not None:
                    report["runtime_hotfix_sha256"] = patch_report[
                        "runtime_hotfix_sha256"
                    ]
                if full_modal_action_adoption is not None:
                    report["full_modal_action_adoption"] = full_modal_action_adoption
                if args.trainable_scope == TRAINABLE_SCOPE_FROZEN_VISION_HOST:
                    report["trainable_scope"] = trainable_scope_receipt
                write_text_durable_exclusive(
                    args.output,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
            except BaseException as error:
                publication_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(publication_error, src=0)
        if publication_error[0] is not None:
            raise RuntimeError(f"LBOT report publication failed: {publication_error[0]}")
        dist.barrier()
        if rank == 0:
            if report is None:
                raise RuntimeError("rank zero lost the LBOT report")
            print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        if run_lease is not None:
            run_lease.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
