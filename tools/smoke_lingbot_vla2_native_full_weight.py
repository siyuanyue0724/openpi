#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the released-weight LingBot ADR-74 native G0/G1 integration smoke.

Accelerator and LingBot imports stay inside ``main`` so source verification,
CLI checks and report publication remain testable on a CPU-only workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
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
        PATCHED_SOURCES,
        detect_native_patch_state,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        TARGET_ONLY_FIELDS,
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _RouteTrace,
        _sha256,
        _tensor_sha256,
        load_lingbot_training_config,
        select_lingbot_deterministic_moe_backend,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_g0 import _implementation_digest
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
        PATCHED_SOURCES,
        detect_native_patch_state,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        TARGET_ONLY_FIELDS,
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _RouteTrace,
        _sha256,
        _tensor_sha256,
        load_lingbot_training_config,
        select_lingbot_deterministic_moe_backend,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_g0 import _implementation_digest  # type: ignore[no-redef]


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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="pick up the red block")
    parser.add_argument("--alternate-task", default="move the slider to the left")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument(
        "--architecture-identity",
        default="content_addressed_task_match_v1",
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    if args.robot_config is None:
        args.robot_config = args.source_checkout / "configs/robot_configs/robotwin.yaml"
    return args


def _validate_dimensions(*, capacity: int, maximum_control_tokens: int, num_steps: int) -> None:
    values = (capacity, maximum_control_tokens, num_steps)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("native capacity, control bound and num-steps must be positive integers")


def _source_diff_digest(checkout: Path) -> str:
    return hashlib.sha256(_git_output(checkout, "diff", "--binary").encode()).hexdigest()


def _validated_patched_source_hashes(
    checkout: Path, patch_report: dict[str, object]
) -> dict[str, str]:
    expected = patch_report.get("patched_source_sha256")
    accepted_paths = {str(path) for path in PATCHED_SOURCES}
    if not isinstance(expected, dict) or set(expected) != accepted_paths:
        raise RuntimeError("native patch verifier returned the wrong source hash contract")
    actual = {relative: _sha256(checkout / relative) for relative in sorted(accepted_paths)}
    if actual != expected:
        raise RuntimeError("LingBot native source differs from immutable patch replay")
    return actual


def _write_text_durable(path: Path, payload: str) -> None:
    write_text_durable_exclusive(path, payload)


def _relation_is_finite(relation: Any) -> bool:
    fields = [
        "support_logits",
        "visible_support",
        "ownership",
        "existence",
        "existence_logits",
    ]
    fields.extend(
        name
        for name in (
            "task_relevance",
            "task_relevance_logits",
            "dense_task_grounding",
        )
        if hasattr(relation, name)
    )
    return all(bool(getattr(relation, name).isfinite().all()) for name in fields)


def _physical_relation_prompt_error(first: Any, second: Any) -> float:
    fields = ("support_logits", "ownership", "existence_logits")
    if any(not hasattr(first, name) or not hasattr(second, name) for name in fields):
        raise TypeError("successor smoke requires the physical relation ABI")
    return max(
        float((getattr(first, name).float() - getattr(second, name).float()).abs().max().item())
        for name in fields
    )


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    for path in (
        args.source_checkout,
        args.patch,
        args.config,
        args.robot_config,
        args.image,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    _validate_dimensions(
        capacity=args.capacity,
        maximum_control_tokens=args.maximum_control_tokens,
        num_steps=args.num_steps,
    )
    patch_report = verify_native_patch(
        root=root,
        checkout=args.source_checkout,
        check_apply=True,
    )
    if _git_output(args.source_checkout, "rev-parse", "HEAD") != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("LingBot native source checkout differs from the pinned commit")
    if detect_native_patch_state(args.source_checkout, args.patch) != "applied":
        raise RuntimeError("LingBot native source patch is not in its exact applied state")
    actual_hashes = _validated_patched_source_hashes(args.source_checkout, patch_report)
    checkpoint_report = validate_checkpoint(args.checkpoint_dir)
    processor_report = validate_processor(args.processor_dir)

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
    import torch
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
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
    from lingbotvla.ops import fused_moe
    from PIL import Image
    from torchvision.transforms.v2 import Resize
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    from picf_next.lingbot_native.controls import ExecutedControlBatch
    from picf_next.lingbot_native.host import (
        NATIVE_GRAPH_ARCHITECTURES,
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        install_lingbot_native_graph,
    )
    from picf_next.lingbot_native.runtime import LingBotNativePolicyRuntime
    from picf_next.lingbot_native.session import (
        NativeObservationBatch,
        NativeSessionConfig,
        NativeSessionManager,
    )

    moe_inference_backend = select_lingbot_deterministic_moe_backend(
        action_expert_module=qwen2_action_expert,
        fused_moe_module=fused_moe,
    )
    if args.architecture_identity not in NATIVE_GRAPH_ARCHITECTURES:
        raise ValueError("full-weight smoke architecture identity is unsupported")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("released-weight native G0 requires a CUDA device")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False

    training = load_lingbot_training_config(args.config)
    merged, data_mapping = _resolve_training_config(
        training,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        num_steps=args.num_steps,
    )
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
    config.use_cache = True
    config.use_compile = False
    config.num_steps = args.num_steps
    config.attention_implementation = "eager"
    config.vit_attn_implementation = "eager"

    timings: dict[str, float] = {}
    started = time.perf_counter()
    processor = build_processor(str(args.processor_dir.resolve()))
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    with init_empty_weights(), no_init_weights():
        policy = LingbotVlaV2Policy(config=config, eval=True).to(dtype)
    load_model_weights(
        policy,
        str(args.checkpoint_dir.resolve()),
        str(device),
        post_training=True,
        adanorm_time=bool(config.adanorm_time),
    )
    policy.eval()
    timings["load_model_s"] = time.perf_counter() - started

    feature_transform = FeatureTransform(
        str(args.robot_config.resolve()),
        official_lingbot_data_config(data_mapping),
        config,
        processor,
        chunk_size=config.chunk_size,
        norm_stats_path=str((args.source_checkout / "assets/norm_stats/robotwin.json").resolve()),
    )
    image = Image.open(args.image).convert("RGB")
    chw = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
    chw = Resize((256, 256))(chw)

    def model_inputs_for(task: str) -> tuple[dict[str, torch.Tensor], list[str]]:
        raw = {
            "observation.images.cam_high": chw.clone(),
            "observation.images.cam_left_wrist": chw.clone(),
            "observation.images.cam_right_wrist": chw.clone(),
            "observation.state": torch.zeros(14, dtype=torch.float32),
            "task": task,
        }
        forbidden = sorted(TARGET_ONLY_FIELDS.intersection(raw))
        if forbidden:
            raise RuntimeError(f"target-only fields entered the native observation: {forbidden}")
        observation = feature_transform.apply(raw, policy_eval=True)
        return (
            {
                "images": observation["images"].unsqueeze(0).to(device=device, dtype=dtype),
                "img_masks": observation["img_masks"].unsqueeze(0).to(device=device),
                "lang_tokens": observation["lang_tokens"].unsqueeze(0).to(device=device),
                "lang_masks": observation["lang_masks"].unsqueeze(0).to(device=device),
                "state": observation["state"].unsqueeze(0).to(device=device, dtype=dtype),
                "image_grid_thw": observation["image_grid_thw"].to(device=device),
            },
            forbidden,
        )

    model_inputs, forbidden = model_inputs_for(args.task)
    alternate_inputs, alternate_forbidden = model_inputs_for(args.alternate_task)
    if alternate_forbidden:
        raise RuntimeError("target-only fields entered the alternate prompt observation")
    noise = torch.randn(
        (1, config.n_action_steps, config.max_action_dim),
        generator=torch.Generator(device=device).manual_seed(args.seed),
        device=device,
        dtype=dtype,
    )

    blocks = [layer.mlp for layer in policy.model.qwenvl_with_expert.qwen_expert.model.layers]
    official_trace = _RouteTrace(torch, blocks)
    started = time.perf_counter()
    with torch.inference_mode():
        official = policy.sample_actions(**model_inputs, noise=noise.clone())
    torch.cuda.synchronize(device)
    timings["official_action_s"] = time.perf_counter() - started
    official_routes = official_trace.finish()

    official_repeat_trace = _RouteTrace(torch, blocks)
    started = time.perf_counter()
    with torch.inference_mode():
        official_repeat = policy.sample_actions(**model_inputs, noise=noise.clone())
    torch.cuda.synchronize(device)
    timings["official_repeat_action_s"] = time.perf_counter() - started
    official_repeat_routes = official_repeat_trace.finish()

    alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
    targetless_trace = _RouteTrace(torch, blocks)
    started = time.perf_counter()
    with torch.inference_mode():
        targetless = policy.sample_actions(**model_inputs, noise=noise.clone())
    torch.cuda.synchronize(device)
    timings["targetless_action_s"] = time.perf_counter() - started
    targetless_routes = targetless_trace.finish()

    graph_config = LingBotNativeGraphConfig.from_policy(
        policy,
        capacity=args.capacity,
        maximum_control_tokens=args.maximum_control_tokens,
        architecture_identity=args.architecture_identity,
    )
    graph = LingBotNativeGraph(graph_config, device=device, dtype=dtype).eval()
    install_lingbot_native_graph(policy, graph)

    neutral_trace = _RouteTrace(torch, blocks)
    started = time.perf_counter()
    with torch.inference_mode():
        neutral = policy.sample_actions(**model_inputs, noise=noise.clone())
    torch.cuda.synchronize(device)
    timings["installed_neutral_action_s"] = time.perf_counter() - started
    neutral_routes = neutral_trace.finish()

    session_config = NativeSessionConfig(
        model_digest=f"{LINGBOT_CHECKPOINT_REVISION}:{patch_report['patch_sha256']}",
        capacity=args.capacity,
        host_width=graph_config.host_width,
        num_layers=graph_config.num_layers if graph.layerwise_recurrence else None,
        dtype=dtype,
        device=device,
    )
    sessions = NativeSessionManager(session_config)
    runtime = LingBotNativePolicyRuntime(policy=policy, graph=graph, sessions=sessions)
    reset_controls = ExecutedControlBatch.reset_only(
        batch_size=1,
        action_dim=graph_config.executed_action_dim,
        device=device,
        dtype=dtype,
    )
    reset_observation = NativeObservationBatch(
        environment_keys=("g0-native-session",),
        reset_epochs=(1,),
        observation_sequences=(0,),
        observation_times=torch.tensor([1.0], device=device),
        reset=(True,),
        controls=reset_controls,
    )
    started = time.perf_counter()
    first = runtime.sample_actions(
        reset_observation,
        model_inputs=model_inputs,
        noise=noise.clone(),
    )
    torch.cuda.synchronize(device)
    timings["native_reset_action_s"] = time.perf_counter() - started

    executed = first.actions[:, :1].detach()
    continuation_controls = ExecutedControlBatch(
        values=executed,
        field_valid=torch.ones_like(executed, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool, device=device),
        delta_time=torch.full((1, 1), 0.1, dtype=dtype, device=device),
        reset=torch.zeros(1, 1, dtype=torch.bool, device=device),
        acknowledged=torch.ones(1, 1, dtype=torch.bool, device=device),
    )
    continuation_observation = NativeObservationBatch(
        environment_keys=("g0-native-session",),
        reset_epochs=(1,),
        observation_sequences=(1,),
        observation_times=torch.tensor([2.0], device=device),
        reset=(False,),
        controls=continuation_controls,
    )
    started = time.perf_counter()
    second = runtime.sample_actions(
        continuation_observation,
        model_inputs=model_inputs,
        noise=noise.clone(),
    )
    torch.cuda.synchronize(device)
    timings["native_continuation_action_s"] = time.perf_counter() - started

    snapshot = sessions.serialize()
    restored_sessions = NativeSessionManager.deserialize(session_config, snapshot)
    snapshot_roundtrip_exact = restored_sessions.serialize() == snapshot

    alternate_sessions = NativeSessionManager(session_config)
    alternate_runtime = LingBotNativePolicyRuntime(
        policy=policy,
        graph=graph,
        sessions=alternate_sessions,
    )
    alternate_observation = NativeObservationBatch(
        environment_keys=("g1-prompt-invariance",),
        reset_epochs=(1,),
        observation_sequences=(0,),
        observation_times=torch.tensor([1.0], device=device),
        reset=(True,),
        controls=reset_controls,
    )
    alternate = alternate_runtime.sample_actions(
        alternate_observation,
        model_inputs=alternate_inputs,
        noise=noise.clone(),
    )

    neutral_error = float((neutral.float() - official.float()).abs().max().item())
    targetless_error = float((targetless.float() - official.float()).abs().max().item())
    official_repeat_error = float((official_repeat.float() - official.float()).abs().max().item())
    prompt_state_error = float(
        (alternate.posterior_state.rows.float() - first.posterior_state.rows.float())
        .abs()
        .max()
        .item()
    )
    successor = graph.task_independent
    prompt_relation_error = (
        _physical_relation_prompt_error(first.relations, alternate.relations) if successor else None
    )
    task_scorer_surface_absent = (
        not any(
            hasattr(first.relations, name)
            for name in (
                "task_relevance",
                "task_relevance_logits",
                "dense_task_grounding",
                "match_embeddings",
            )
        )
        if successor
        else None
    )
    report = {
        "schema": (
            "picf-next.lingbot-vla2-task-independent-full-weight-smoke.v1"
            if successor
            else "picf-next.lingbot-vla2-native-full-weight-smoke.v4"
        ),
        "implementation_sha256": _implementation_digest(root),
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "source_patch_sha256": patch_report["patch_sha256"],
        "patched_source_sha256": actual_hashes,
        "source_diff_sha256": _source_diff_digest(args.source_checkout),
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "checkpoint_assets": checkpoint_report["checkpoint_assets"],
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "processor_assets": processor_report["processor_assets"],
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "image": str(args.image.resolve()),
        "image_sha256": _sha256(args.image),
        "task": args.task,
        "alternate_task": args.alternate_task,
        "target_only_fields_present": forbidden,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "num_steps": args.num_steps,
        "native_graph": {
            "capacity": graph_config.capacity,
            "host_width": graph_config.host_width,
            "executed_action_dim": graph_config.executed_action_dim,
            "num_layers": graph_config.num_layers,
            "maximum_control_tokens": graph_config.maximum_control_tokens,
            "object_transition": graph_config.object_transition,
        },
        "input_shapes": {key: list(value.shape) for key, value in model_inputs.items()},
        "moe_inference_backend": moe_inference_backend,
        "official_action_sha256": _tensor_sha256(official),
        "official_repeat_action_sha256": _tensor_sha256(official_repeat),
        "targetless_action_sha256": _tensor_sha256(targetless),
        "installed_neutral_action_sha256": _tensor_sha256(neutral),
        "alignment_teacher_prune": alignment_teacher_prune,
        "official_repeat_action_bitwise_equal": bool(torch.equal(official_repeat, official)),
        "official_repeat_action_max_abs_error": official_repeat_error,
        "targetless_action_bitwise_equal": bool(torch.equal(targetless, official)),
        "targetless_action_max_abs_error": targetless_error,
        "neutral_action_bitwise_equal": bool(torch.equal(neutral, official)),
        "neutral_action_max_abs_error": neutral_error,
        "official_routes": official_routes,
        "official_repeat_routes": official_repeat_routes,
        "targetless_routes": targetless_routes,
        "installed_neutral_routes": neutral_routes,
        "official_repeat_route_bitwise_equal": official_repeat_routes == official_routes,
        "targetless_route_bitwise_equal": official_routes == targetless_routes,
        "neutral_route_bitwise_equal": official_routes == neutral_routes,
        "first_action_sha256": _tensor_sha256(first.actions),
        "second_action_sha256": _tensor_sha256(second.actions),
        "first_prior_sha256": hashlib.sha256(first.prior_state.serialize()).hexdigest(),
        "first_posterior_sha256": hashlib.sha256(first.posterior_state.serialize()).hexdigest(),
        "second_prior_sha256": hashlib.sha256(second.prior_state.serialize()).hexdigest(),
        "second_posterior_sha256": hashlib.sha256(second.posterior_state.serialize()).hexdigest(),
        "native_actions_finite": bool(
            torch.isfinite(first.actions).all() and torch.isfinite(second.actions).all()
        ),
        "native_relations_finite": _relation_is_finite(first.relations)
        and _relation_is_finite(second.relations),
        "session_snapshot_bytes": len(snapshot),
        "session_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "session_snapshot_roundtrip_exact": snapshot_roundtrip_exact,
        "prompt_invariant_physical_posterior_bitwise_equal": bool(
            torch.equal(alternate.posterior_state.rows, first.posterior_state.rows)
        ),
        "prompt_invariant_physical_posterior_max_abs_error": prompt_state_error,
        "alternate_action_sha256": _tensor_sha256(alternate.actions),
        "timings": timings,
        "cuda_memory_bytes": _cuda_memory(torch, device),
        "pid": os.getpid(),
    }
    if successor:
        report["native_graph"]["architecture_identity"] = args.architecture_identity
        report["relation_interface"] = first.relations.interface
        report["task_scorer_surface_absent"] = task_scorer_surface_absent
        report["prompt_invariant_physical_relation_bitwise_equal"] = prompt_relation_error == 0.0
        report["prompt_invariant_physical_relation_max_abs_error"] = prompt_relation_error
    failures: list[str] = []
    if not report["official_repeat_action_bitwise_equal"]:
        failures.append(f"official repeat action max_abs_error={official_repeat_error}")
    if not report["official_repeat_route_bitwise_equal"]:
        failures.append("official repeat action MoE routes changed")
    if not report["targetless_action_bitwise_equal"]:
        failures.append(f"targetless-pruned action max_abs_error={targetless_error}")
    if not report["targetless_route_bitwise_equal"]:
        failures.append("targetless-pruned action MoE routes changed")
    if not report["neutral_action_bitwise_equal"]:
        failures.append(f"installed-neutral action max_abs_error={neutral_error}")
    if not report["neutral_route_bitwise_equal"]:
        failures.append("installed-neutral action MoE routes changed")
    if not report["native_actions_finite"]:
        failures.append("native runtime actions contain NaN or infinity")
    if not report["native_relations_finite"]:
        failures.append("native relation output contains NaN or infinity")
    if not snapshot_roundtrip_exact:
        failures.append("native session snapshot roundtrip differs")
    if not report["prompt_invariant_physical_posterior_bitwise_equal"]:
        failures.append(
            f"task prompt changed physical posterior: max_abs_error={prompt_state_error}"
        )
    if successor and not task_scorer_surface_absent:
        failures.append("task-independent relation exposed a task scorer surface")
    if successor and prompt_relation_error != 0.0:
        failures.append(
            f"task prompt changed physical relation: max_abs_error={prompt_relation_error}"
        )
    report["status"] = "FAIL" if failures else "PASS"
    report["failures"] = failures
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(payload, end="")
    if failures:
        raise RuntimeError("LingBot native released-weight smoke failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
