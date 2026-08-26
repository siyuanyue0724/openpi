#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Measure released/restored shared-Qwen action compatibility on real CALVIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    _text = str(_path)
    while _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.lingbot_native.lattice_feasibility import (
    LATTICE_BASELINE,
    configure_native_processor_lattice,
    require_native_visual_grid_cache_populated,
    reset_native_visual_grid_cache,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config
from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
from tools.bootstrap_lingbot_vla2_native import (
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MODEL_SOURCE,
    QWEN_PROCESSOR_REVISION,
)
from tools.bootstrap_lingbot_vla2_native_vl import (
    NATIVE_VL_PATCH_RELATIVE_PATH,
    NATIVE_VL_PATCHED_MODEL_SHA256,
    _validate_native_vl_model,
    detect_native_vl_patch_state,
    verify_native_vl_patch,
)
from tools.lingbot_vla2_runtime_helpers import (
    _merge_qwen_config,
    _resolve_training_config,
    load_lingbot_training_config,
    strip_targetless_alignment_teacher_heads,
)
from tools.probe_lingbot_native_vl_grounding import (
    _validate_optional_qwen_restore,
    _validate_qwen_restore_load_result,
)
from tools.probe_qwen3vl_grounding_baseline import _load_probe_report, _model_hashes

OUTPUT_SCHEMA = "picf-next.lingbot-qwen-restore-action-compatibility.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-count", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--visual-lattice", type=int, default=LATTICE_BASELINE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--restore-qwen-dir", type=Path)
    parser.add_argument("--restore-qwen-revision")
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.training_config,
        args.robot_config,
        args.data_config,
        args.dataset_manifest,
        args.norm_stats,
        args.input_report,
        args.source_checkout / MODEL_SOURCE,
        _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    integers = (args.record_count, args.num_steps, args.visual_lattice, args.seed)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise ContractError("action compatibility integer arguments are invalid")
    if args.record_count <= 0 or args.num_steps <= 0 or args.visual_lattice <= 0 or args.seed < 0:
        raise ContractError("action compatibility numeric arguments are invalid")
    if not args.device.startswith("cuda:"):
        raise ContractError("action compatibility probe requires one CUDA device")
    _validate_optional_qwen_restore(args.restore_qwen_dir, args.restore_qwen_revision)


def _select_action_payloads(records: object, count: int) -> tuple[dict[str, object], ...]:
    if not isinstance(records, list):
        raise ContractError("action compatibility input contains no record list")
    unique: list[dict[str, object]] = []
    seen: set[tuple[object, object]] = set()
    for value in records:
        if not isinstance(value, dict):
            raise ContractError("action compatibility record must be a JSON object")
        identity = (value.get("global_index"), value.get("task_key"))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(value)
    if count > len(unique):
        raise ContractError("action compatibility input has too few unique source records")
    indices = tuple((ordinal * len(unique)) // count for ordinal in range(count))
    return tuple(unique[index] for index in indices)


def _record_text(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ContractError(f"action compatibility {name} is invalid")
    return value


def _record_index(payload: dict[str, object]) -> int:
    value = payload.get("global_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("action compatibility global index is invalid")
    return value


def _sample_from_payload(index: Any, payload: dict[str, object], action_horizon: int) -> Any:
    global_index = _record_index(payload)
    task_key = _record_text(payload, "task_key")
    instruction = _record_text(payload, "instruction")
    matches = tuple(
        segment
        for segment in index.segments
        if segment.start <= global_index < segment.end
        and segment.task_key == task_key
        and segment.instruction == instruction
    )
    if len(matches) > 1:
        # The G0 bank is generated by walking each language segment from its
        # start. CALVIN contains duplicated overlapping annotations, so retain
        # the unique originating segment instead of choosing by list order.
        originating = tuple(segment for segment in matches if segment.start == global_index)
        if len(originating) == 1:
            matches = originating
    if len(matches) != 1:
        raise ContractError("action compatibility source record is not uniquely addressable")
    segment = matches[0]
    return index.stateful_transition_sample(
        segment.index,
        global_index,
        action_horizon=action_horizon,
    )


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().float().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _tensor_summary(value: Any) -> dict[str, object]:
    tensor = value.detach().float().cpu()
    return {
        "finite": bool(tensor.isfinite().all().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "shape": [int(item) for item in tensor.shape],
        "sha256": _tensor_sha256(tensor),
        "std": float(tensor.std(unbiased=False).item()),
    }


def _move_model_inputs(
    model_inputs: Mapping[str, Any],
    *,
    device: Any,
    dtype: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Match the production native runner's tensor-device boundary exactly."""

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


def _configure_official_action_sampling(config: Any, *, num_steps: int) -> None:
    """Apply the post-Qwen-merge settings used by LingBot's proven smoke path."""

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ContractError("official LingBot action sampling requires positive num_steps")
    config.use_cache = True
    config.use_compile = False
    config.use_lm_head = True
    config.num_steps = num_steps
    config.attention_implementation = "eager"
    config.vit_attn_implementation = "eager"
    if config.use_cache is not True:
        raise ContractError("official LingBot action sampling requires its KV cache")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    patch_report = verify_native_vl_patch(root=_ROOT, checkout=args.source_checkout)
    overlay = _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH
    if detect_native_vl_patch_state(args.source_checkout, overlay) != "applied":
        raise RuntimeError("action compatibility native-VL source overlay is not applied")
    if _validate_native_vl_model(args.source_checkout / MODEL_SOURCE) != (
        NATIVE_VL_PATCHED_MODEL_SHA256
    ):
        raise RuntimeError("action compatibility source digest differs")
    commit = subprocess.run(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("action compatibility source commit differs")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    input_report = _load_probe_report(args.input_report)
    payloads = _select_action_payloads(input_report.get("records"), args.record_count)

    sys.path.insert(0, str(args.source_checkout.resolve()))
    import torch
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
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
    from transformers.modeling_utils import load_sharded_checkpoint, no_init_weights

    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.calvin import (
        build_native_calvin_training_batch,
        collate_native_calvin_training_batch,
    )
    from picf_next.lingbot_native.vl_cotraining import (
        retie_and_validate_native_qwen_lm_head,
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.name,
    )
    if input_report.get("dataset_manifest_sha256") != manifest.tree_sha256:
        raise ContractError("action compatibility input belongs to another dataset tree")
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )

    training = load_lingbot_training_config(args.training_config)
    merged, _ = _resolve_training_config(
        training,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        num_steps=args.num_steps,
    )
    merged["use_lm_head"] = True
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
    _configure_official_action_sampling(config, num_steps=args.num_steps)

    qwen_restore = None
    if args.restore_qwen_dir is not None:
        qwen_restore = {
            "model_dir": str(args.restore_qwen_dir.resolve()),
            "model_file_sha256": _model_hashes(args.restore_qwen_dir),
            "model_revision": args.restore_qwen_revision,
        }

    processor = build_processor(str(args.processor_dir.resolve()))
    processor_lattice = configure_native_processor_lattice(
        processor,
        args.visual_lattice,
    )
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    load_started = time.perf_counter()
    with init_empty_weights(), no_init_weights():
        policy = LingbotVlaV2Policy(config=config, eval=True).to(dtype)
    preload_tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
    load_model_weights(
        policy,
        str(args.checkpoint_dir.resolve()),
        str(device),
        post_training=True,
        adanorm_time=bool(config.adanorm_time),
    )
    tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
    if tied_parameter_name != preload_tied_parameter_name:
        raise ContractError("Qwen tied parameter name drifted across released-weight loading")
    if args.restore_qwen_dir is not None and qwen_restore is not None:
        qwen_restore["load_result"] = _validate_qwen_restore_load_result(
            load_sharded_checkpoint(
                policy.model.qwenvl_with_expert.qwenvl,
                args.restore_qwen_dir,
                strict=False,
                prefer_safe=True,
            )
        )
        restored_tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
        if restored_tied_parameter_name != tied_parameter_name:
            raise ContractError("Qwen tied parameter name drifted across foundation restoration")
    teacher_prune = strip_targetless_alignment_teacher_heads(policy)
    visual_grid_cache = reset_native_visual_grid_cache(policy.model.qwenvl_with_expert)
    policy.eval()
    load_seconds = time.perf_counter() - load_started

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
    results = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for ordinal, payload in enumerate(payloads):
            sample = _sample_from_payload(index, payload, config.chunk_size)
            raw = build_native_calvin_training_batch(
                (sample,),
                lane_ids=(0,),
                optimizer_step=0,
                device="cpu",
                dtype=torch.float32,
            )
            source_digest = hashlib.sha256(
                f"{sample.sample_key}\0{args.seed + ordinal}".encode()
            ).hexdigest()
            collated = collate_native_calvin_training_batch(
                raw,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=(args.seed + ordinal,),
                source_digest=source_digest,
            )
            model_inputs = _move_model_inputs(
                collated.model_inputs,
                device=device,
                dtype=dtype,
                torch_module=torch,
            )
            actions = model_inputs["actions"]
            generator = torch.Generator(device="cpu").manual_seed(args.seed + ordinal)
            fixed_noise = torch.randn(actions.shape, generator=generator, dtype=torch.float32).to(
                device=device,
                dtype=dtype,
            )
            model_inputs["noise"] = fixed_noise
            model_inputs["time"] = torch.full(
                (1,),
                0.5,
                device=device,
                dtype=dtype,
            )
            inference_inputs = {
                name: value
                for name, value in model_inputs.items()
                if name not in {"actions", "action_is_pad", "joint_mask", "noise", "time"}
            }
            action = policy.sample_actions(**inference_inputs, noise=fixed_noise.clone())
            forward = policy(**model_inputs, compute_alignment_losses=False)
            action_loss = float(forward[1].detach().float().item())
            batch_losses = forward[6].get("batch_mean_losses")
            if not math.isfinite(action_loss) or batch_losses is None:
                raise RuntimeError("action compatibility forward returned invalid loss")
            results.append(
                {
                    "action": _tensor_summary(action),
                    "action_loss": action_loss,
                    "camera_name": payload.get("camera_name"),
                    "global_index": _record_index(payload),
                    "instruction": _record_text(payload, "instruction"),
                    "sample_key": sample.sample_key,
                    "source_digest": source_digest,
                    "task_key": _record_text(payload, "task_key"),
                }
            )
    torch.cuda.synchronize(device)
    visual_grid_cache["populated_after_probe"] = require_native_visual_grid_cache_populated(
        policy.model.qwenvl_with_expert
    )
    elapsed = time.perf_counter() - started
    losses = [float(item["action_loss"]) for item in results]
    report = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "dataset_manifest_sha256": manifest.tree_sha256,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "input_report_sha256": _sha256(args.input_report),
        "load_seconds": load_seconds,
        "mean_action_loss": sum(losses) / len(losses),
        "native_vl_patch_sha256": patch_report["native_vl_patch_sha256"],
        "num_steps": args.num_steps,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "preload_tied_parameter_name": preload_tied_parameter_name,
        "processor_dir": str(args.processor_dir.resolve()),
        "processor_lattice": processor_lattice,
        "qwen_restore": qwen_restore,
        "record_count": len(results),
        "records": results,
        "schema": OUTPUT_SCHEMA,
        "seed": args.seed,
        "source_commit": commit,
        "teacher_prune": teacher_prune,
        "tied_parameter_name": tied_parameter_name,
        "visual_grid_cache": visual_grid_cache,
    }
    write_text_durable_exclusive(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
