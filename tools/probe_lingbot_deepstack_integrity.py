#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Verify released-weight Qwen3-VL DeepStack injection on one real CALVIN sample.

This is a read-only causal diagnostic.  It executes normal, repeated-normal
and DeepStack-zeroed forwards through the same LingBot host and PICF graph.
No target mask, object identity, optimizer or checkpoint mutation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from picf_next.artifact_io import write_text_durable_exclusive
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
        detect_native_patch_state,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        load_lingbot_training_config,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _git_output,
        _move_model_inputs,
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
        detect_native_patch_state,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        load_lingbot_training_config,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _git_output,
        _move_model_inputs,
    )


_DEFAULT_SAMPLE_KEY = "calvin-language-segment-00003181/transition-00000000-frame-01232848"


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = Path(
        os.environ.get(
            "PICF_LINGBOT_NATIVE_SOURCE",
            root / CHECKOUT_RELATIVE_PATH,
        )
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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--sample-key", default=_DEFAULT_SAMPLE_KEY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _source_diff_digest(checkout: Path) -> str:
    return hashlib.sha256(_git_output(checkout, "diff", "--binary").encode()).hexdigest()


def _require_paths(args: argparse.Namespace) -> None:
    for path in (
        args.source_checkout,
        args.patch,
        args.training_config,
        args.robot_config,
        args.data_config,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
        args.dataset_manifest,
        args.norm_stats,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if not args.sample_key:
        raise ValueError("DeepStack probe sample key must be non-empty")
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or args.seed < 0
        or isinstance(args.capacity, bool)
        or not isinstance(args.capacity, int)
        or args.capacity <= 0
        or isinstance(args.maximum_control_tokens, bool)
        or not isinstance(args.maximum_control_tokens, int)
        or args.maximum_control_tokens <= 0
    ):
        raise ValueError("DeepStack seed and graph dimensions are invalid")


def _patched_source_hashes(
    checkout: Path,
    patch_report: Mapping[str, object],
) -> dict[str, str]:
    expected = patch_report.get("patched_source_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeError("native patch verifier returned no source hash contract")
    actual = {str(relative): _sha256(checkout / str(relative)) for relative in sorted(expected)}
    if actual != dict(expected):
        raise RuntimeError("LingBot source differs from immutable patch replay")
    return actual


class _DeepStackTrace:
    """Temporarily observe or zero the official `_apply_deepstack` method."""

    def __init__(self, *, torch_module: Any, module: Any, mode: str) -> None:
        if mode not in {"normal", "zeroed"}:
            raise ValueError("DeepStack trace mode must be normal or zeroed")
        self._torch = torch_module
        self._module = module
        self.mode = mode
        self.injections: list[dict[str, object]] = []
        self._original: Any = None

    def __enter__(self) -> _DeepStackTrace:
        if self._original is not None:
            raise RuntimeError("DeepStack trace cannot be entered twice")
        self._original = self._module._apply_deepstack
        trace = self

        def wrapped(
            module: Any,
            hidden_states: Any,
            layer_idx: int,
            visual_pos_masks: Any,
            deepstack_visual_embeds: Any,
        ) -> Any:
            applicable = (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            )
            if not applicable:
                return trace._original(
                    hidden_states,
                    layer_idx,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
            feature = deepstack_visual_embeds[layer_idx]
            mask = visual_pos_masks.to(device=hidden_states.device, dtype=trace._torch.bool)
            if mask.shape != hidden_states.shape[:2]:
                raise RuntimeError("DeepStack visual mask and prefix hidden shape differ")
            before = hidden_states.detach().clone()
            before_visual = before[mask, :].clone()
            if before_visual.shape != feature.shape:
                raise RuntimeError("DeepStack feature and visual prefix positions differ")
            expected_visual = before_visual + feature.to(
                device=before_visual.device,
                dtype=before_visual.dtype,
            )
            result = (
                hidden_states
                if trace.mode == "zeroed"
                else trace._original(
                    hidden_states,
                    layer_idx,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                )
            )
            after = result.detach()
            visual_delta = after[mask, :].to(trace._torch.float32) - before_visual.to(
                trace._torch.float32
            )
            nonvisual_delta = after[~mask, :].to(trace._torch.float32) - before[~mask, :].to(
                trace._torch.float32
            )
            visual_expected_error = (
                None
                if trace.mode == "zeroed"
                else float(
                    (
                        after[mask, :].to(trace._torch.float32)
                        - expected_visual.to(trace._torch.float32)
                    )
                    .abs()
                    .max()
                    .item()
                )
            )
            from picf_next.lingbot_native.deepstack_integrity import (
                tensor_numeric_summary,
            )

            trace.injections.append(
                {
                    "layer_index": int(layer_idx),
                    "hidden_shape": [int(item) for item in hidden_states.shape],
                    "visual_position_count": int(mask.sum().item()),
                    "feature": tensor_numeric_summary(feature),
                    "visual_delta_rms": float(visual_delta.square().mean().sqrt().item()),
                    "visual_delta_max_abs": float(visual_delta.abs().max().item()),
                    "visual_expected_max_abs_error": visual_expected_error,
                    "nonvisual_max_abs_delta": (
                        0.0
                        if nonvisual_delta.numel() == 0
                        else float(nonvisual_delta.abs().max().item())
                    ),
                }
            )
            return result

        self._module._apply_deepstack = types.MethodType(wrapped, self._module)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._module._apply_deepstack = self._original
        self._original = None


def main() -> None:
    args = _parse_args()
    _require_paths(args)
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(
        root=root,
        checkout=args.source_checkout,
        check_apply=True,
    )
    if _git_output(args.source_checkout, "rev-parse", "HEAD") != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("LingBot native source checkout differs from the pinned commit")
    if detect_native_patch_state(args.source_checkout, args.patch) != "applied":
        raise RuntimeError("LingBot native source patch is not in its exact applied state")
    patched_source_sha256 = _patched_source_hashes(args.source_checkout, patch_report)
    checkpoint_report = validate_checkpoint(args.checkpoint_dir)
    processor_report = validate_processor(args.processor_dir)

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
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
    from transformers import __version__ as transformers_version
    from transformers.modeling_utils import no_init_weights

    from picf_next.data.calvin import (
        CalvinDatasetIndex,
        CalvinStatefulTransitionDataset,
    )
    from picf_next.data.dataset_manifest import load_dataset_file_manifest
    from picf_next.lingbot_native.calvin import (
        audit_native_calvin_model_inputs,
        build_native_calvin_training_batch,
        collate_native_calvin_training_batch,
    )
    from picf_next.lingbot_native.deepstack_integrity import (
        DEEPSTACK_INTEGRITY_SCHEMA,
        deepstack_integrity_gates,
        tensor_difference_summary,
        tensor_numeric_summary,
        validate_deepstack_integrity_report,
    )
    from picf_next.lingbot_native.host import (
        LingBotNativeContext,
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        install_lingbot_native_graph,
    )
    from picf_next.lingbot_native.training import (
        run_native_policy_diagnostic_forward,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("released-weight DeepStack probe requires CUDA")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        args.dataset_split.resolve(),
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
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
    if bool(config.train_expert_only) or bool(config.freeze_vision_encoder):
        raise RuntimeError("DeepStack probe requires the same complete trainable VLM host")
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
    load_started = time.perf_counter()
    with init_empty_weights(), no_init_weights():
        policy = LingbotVlaV2Policy(config=config, eval=False).to(dtype)
    load_model_weights(
        policy,
        str(args.checkpoint_dir.resolve()),
        str(device),
        post_training=True,
        adanorm_time=bool(config.adanorm_time),
    )
    alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
    policy.train()
    load_seconds = time.perf_counter() - load_started

    graph_config = LingBotNativeGraphConfig.from_policy(
        policy,
        capacity=args.capacity,
        maximum_control_tokens=args.maximum_control_tokens,
    )
    graph = LingBotNativeGraph(graph_config, device=device, dtype=dtype).train()
    install_lingbot_native_graph(policy, graph)

    dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
    sample = dataset.by_key(args.sample_key)
    if sample.transition_index != 0:
        raise ValueError("DeepStack integrity probe requires a reset transition")
    raw_batch = build_native_calvin_training_batch(
        (sample,),
        lane_ids=(0,),
        optimizer_step=0,
        device=device,
        dtype=dtype,
    )
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
    collated = collate_native_calvin_training_batch(
        raw_batch,
        feature_transform=feature_transform,
        collator=VLADataCollatorWithPacking(),
        augmentation_seeds=(args.seed,),
        source_digest=hashlib.sha256(f"{args.sample_key}\0{args.seed}".encode()).hexdigest(),
    )
    model_inputs = _move_model_inputs(
        collated.model_inputs,
        device=device,
        dtype=dtype,
        torch_module=torch,
    )
    actions = model_inputs["actions"]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model_inputs["noise"] = torch.randn(
        actions.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    model_inputs["time"] = torch.full(
        (actions.shape[0],),
        0.5,
        device=device,
        dtype=dtype,
    )
    audit_native_calvin_model_inputs(model_inputs, require_randomness=True)

    deepstack_host = policy.model.qwenvl_with_expert
    vision = deepstack_host.qwenvl.visual
    deepstack_visual_indexes = tuple(int(item) for item in vision.deepstack_visual_indexes)
    if not deepstack_visual_indexes:
        raise RuntimeError("released Qwen3-VL host exposes no DeepStack visual levels")

    output_tensors: dict[str, dict[str, torch.Tensor]] = {}
    run_reports: dict[str, dict[str, object]] = {}

    def run_once(run_name: str, mode: str) -> None:
        previous_state_valid = torch.zeros(1, dtype=torch.bool, device=device)
        context = LingBotNativeContext(
            controls=collated.controls,
            previous_state=None,
            previous_state_valid=previous_state_valid,
        )
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with _DeepStackTrace(
            torch_module=torch,
            module=deepstack_host,
            mode=mode,
        ) as trace:
            result = run_native_policy_diagnostic_forward(
                policy,
                model_inputs=model_inputs,
                context=context,
            )
        torch.cuda.synchronize(device)
        posterior = context.posterior_state
        relation = context.relation_output
        if posterior is None or relation is None:
            raise RuntimeError("DeepStack paired forward did not finalize PICF outputs")
        tensors = {
            "posterior_rows": posterior.rows.detach().cpu().clone(),
            "relation_ownership": relation.ownership.detach().cpu().clone(),
            "relation_task_relevance": relation.task_relevance.detach().cpu().clone(),
            "official_action_loss": result.official_action_loss.detach().reshape(1).cpu().clone(),
        }
        output_tensors[run_name] = tensors
        run_reports[run_name] = {
            "mode": mode,
            "injections": trace.injections,
            "outputs": {name: tensor_numeric_summary(value) for name, value in tensors.items()},
            "forward_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }

    run_once("normal", "normal")
    run_once("normal_repeat", "normal")
    run_once("zeroed", "zeroed")

    comparisons = {
        "normal_repeat": {
            name: tensor_difference_summary(
                output_tensors["normal"][name],
                output_tensors["normal_repeat"][name],
            )
            for name in output_tensors["normal"]
        },
        "normal_zeroed": {
            name: tensor_difference_summary(
                output_tensors["normal"][name],
                output_tensors["zeroed"][name],
            )
            for name in output_tensors["normal"]
        },
    }
    sample_images = {
        key: value
        for key, value in sample.host_sample.observation.items()
        if key.startswith("observation.images.") and isinstance(value, np.ndarray)
    }
    report: dict[str, Any] = {
        "schema": DEEPSTACK_INTEGRITY_SCHEMA,
        "repository_commit": _git_output(root, "rev-parse", "HEAD"),
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "source_patch_sha256": patch_report["patch_sha256"],
        "patched_source_sha256": patched_source_sha256,
        "source_diff_sha256": _source_diff_digest(args.source_checkout),
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "checkpoint_assets": checkpoint_report["checkpoint_assets"],
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "processor_assets": processor_report["processor_assets"],
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "dataset_tree_sha256": dataset_manifest.tree_sha256,
        "sample": {
            "sample_key": sample.sample_key,
            "task_key": sample.host_sample.task_key,
            "task": sample.host_sample.observation["task"],
            "source_global_index": sample.host_sample.source_global_index,
            "transition_index": sample.transition_index,
            "image_sha256": {
                name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                for name, value in sorted(sample_images.items())
            },
        },
        "target_or_mask_fields_in_model_inputs": [],
        "input_shapes": {
            name: [int(item) for item in value.shape]
            for name, value in sorted(model_inputs.items())
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "transformers_version": transformers_version,
        "load_model_seconds": load_seconds,
        "alignment_teacher_prune": alignment_teacher_prune,
        "expected_deepstack_count": len(deepstack_visual_indexes),
        "deepstack_visual_indexes": list(deepstack_visual_indexes),
        "language_layer_count": int(deepstack_host.qwenvl.config.text_config.num_hidden_layers),
        "runs": run_reports,
        "comparisons": comparisons,
    }
    report["gates"] = deepstack_integrity_gates(report)
    report["failures"] = sorted(name for name, passed in report["gates"].items() if not passed)
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    report = validate_deepstack_integrity_report(report)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    write_text_durable_exclusive(args.output.resolve(), payload)
    print(payload, end="")
    if report["status"] != "PASS":
        raise RuntimeError(
            "released-weight DeepStack integrity probe failed: " + ", ".join(report["failures"])
        )


if __name__ == "__main__":
    main()
