#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Compare released-weight LingBot features at native 8x8 and 12x12 view lattices.

The paired probe changes only Qwen's official dynamic-resolution setting.  It
uses the same released model, PICF graph, source transition, instruction,
action target, flow noise and timestep in both arms.  Physical object masks are
resolved strictly after each forward and are used only for read-only scoring
and visual evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (
    str(_REPOSITORY_ROOT),
    str(_REPOSITORY_ROOT / "src"),
):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from picf_next.artifact_io import (  # noqa: E402
    write_bytes_durable_exclusive,
    write_text_durable_exclusive,
)
from picf_next.lingbot_native.lattice_feasibility import (  # noqa: E402
    LATTICE_BASELINE,
    LATTICE_CANDIDATE,
    LATTICE_FEASIBILITY_SCHEMA,
    configure_native_processor_lattice,
    fractional_token_metrics,
    lattice_feasibility_decision,
    require_native_visual_grid_cache_populated,
    reset_native_visual_grid_cache,
    select_lattice_segment_indices,
    validate_lattice_feasibility_report,
)
from picf_next.lingbot_native.official_config import (  # noqa: E402
    official_lingbot_data_config,
)

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
    from tools.probe_lingbot_deepstack_integrity import (
        _patched_source_hashes,
        _source_diff_digest,
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
    from probe_lingbot_deepstack_integrity import (  # type: ignore[no-redef]
        _patched_source_hashes,
        _source_diff_digest,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _git_output,
        _move_model_inputs,
    )


def _parse_args() -> argparse.Namespace:
    root = _REPOSITORY_ROOT
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
    parser.add_argument("--physical-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


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
        args.physical_sidecar,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.output, args.visual_dir):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
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
        raise ValueError("lattice seed and graph dimensions are invalid")


def _paired_seed(seed: int, sample_key: str, purpose: str) -> int:
    payload = f"{seed}\0{sample_key}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _slug(value: str, *, maximum: int = 100) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return (normalized or "unnamed")[:maximum]


def _sigmoid_grid(logits: list[float], lattice: int, view_index: int) -> Any:
    import numpy as np

    values = np.asarray(logits, dtype=np.float32)
    start = view_index * lattice * lattice
    local = values[start : start + lattice * lattice].reshape(lattice, lattice)
    return 1.0 / (1.0 + np.exp(-np.clip(local, -30.0, 30.0)))


def _mass_grid(mass: list[float], lattice: int, view_index: int) -> Any:
    import numpy as np

    values = np.asarray(mass, dtype=np.float32)
    start = view_index * lattice * lattice
    return values[start : start + lattice * lattice].reshape(lattice, lattice)


def _overlay(source: Any, heat: Any, *, color: tuple[float, float, float]) -> Any:
    import numpy as np
    from PIL import Image

    rgb = np.asarray(source)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("lattice visual source must be HWC uint8 RGB")
    values = np.asarray(heat, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("lattice visual heat must be one finite grid")
    expanded = np.asarray(
        Image.fromarray(values).resize(
            (rgb.shape[1], rgb.shape[0]),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    expanded = np.clip(expanded, 0.0, 1.0)
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    alpha = (0.75 * expanded)[..., None]
    return np.clip(rgb * (1.0 - alpha) + tint * alpha, 0.0, 255.0).astype(np.uint8)


def _tile(source: Any, title: str, *, size: int = 240) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(source).resize((size, size), Image.Resampling.BILINEAR)
    panel = Image.new("RGB", (size, size + 24), "white")
    panel.paste(image, (0, 24))
    ImageDraw.Draw(panel).text((5, 5), title, fill="black")
    return panel


def _render_sample_visual(
    *,
    source_images: tuple[Any, Any],
    task_key: str,
    instruction: str,
    sample_key: str,
    lattice: int,
    logits: list[float],
    mass: list[float],
    metrics: Mapping[str, object],
) -> bytes:
    from PIL import Image, ImageDraw

    panels = []
    for view_index, (name, source) in enumerate(
        zip(("static", "gripper"), source_images, strict=True)
    ):
        target = _mass_grid(mass, lattice, view_index)
        task = _sigmoid_grid(logits, lattice, view_index)
        panels.extend(
            (
                _tile(source, f"{name}: source"),
                _tile(
                    _overlay(source, target, color=(255.0, 30.0, 30.0)),
                    f"{name}: exact target",
                ),
                _tile(
                    _overlay(source, task, color=(20.0, 190.0, 255.0)),
                    f"{name}: sigmoid(task cosine)",
                ),
            )
        )
    width = sum(panel.width for panel in panels)
    header = 72
    canvas = Image.new("RGB", (width, panels[0].height + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), f"{task_key} | lattice={lattice} | {sample_key}", fill="black")
    draw.text((6, 24), instruction[:180], fill="black")
    draw.text(
        (6, 43),
        (
            f"eligible={metrics['eligible']} purity={metrics['purity']} "
            f"self_iou={metrics['self_soft_iou']} auc={metrics['fractional_weighted_auc']}"
        ),
        fill="black",
    )
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, header))
        offset += panel.width
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _sample_images(sample: object) -> tuple[dict[str, str], tuple[Any, Any]]:
    import numpy as np

    observation = sample.host_sample.observation
    keys = ("observation.images.image", "observation.images.wrist_image")
    values = tuple(np.asarray(observation[key]) for key in keys)
    if any(value.ndim != 3 or value.shape[-1] != 3 for value in values):
        raise RuntimeError("CALVIN source RGB geometry changed")
    digests = {
        name: hashlib.sha256(value.tobytes()).hexdigest()
        for name, value in zip(("static", "gripper"), values, strict=True)
    }
    return digests, values


def main() -> None:
    args = _parse_args()
    _require_paths(args)
    root = _REPOSITORY_ROOT
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

    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.calvin_normalization import (
        validate_lingbot_calvin_norm_stats,
    )
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.eval.calvin_task_relevance import (
        calvin_exact_task_loss_identities,
    )
    from picf_next.lingbot_native.calvin import (
        audit_native_calvin_model_inputs,
        build_native_calvin_training_batch,
        collate_native_calvin_training_batch,
    )
    from picf_next.lingbot_native.calvin_supervision import (
        build_native_calvin_sequence_target_bundle,
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
        raise RuntimeError("released-weight lattice probe requires CUDA")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    norm_stats_payload = json.loads(args.norm_stats.read_text())
    validate_lingbot_calvin_norm_stats(norm_stats_payload)
    normalization_source = norm_stats_payload["source"]
    if (
        normalization_source["dataset_id"] != dataset_manifest.dataset_id
        or normalization_source["dataset_revision"] != dataset_manifest.dataset_revision
        or normalization_source["dataset_tree_sha256"] != dataset_manifest.tree_sha256
        or dataset_manifest.split_name != args.dataset_split.name
    ):
        raise RuntimeError("lattice CALVIN manifest and LingBot normalization differ")
    dataset_runtime_binding = validate_dataset_runtime_binding(
        dataset_manifest,
        args.dataset_split.resolve(),
        dataset_id=normalization_source["dataset_id"],
        dataset_revision=normalization_source["dataset_revision"],
        split_name=args.dataset_split.name,
    )
    index = CalvinDatasetIndex.load(
        args.dataset_split.resolve(),
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
    )
    physical_sidecar = CalvinPhysicalSupervisionSidecar(
        args.physical_sidecar.resolve(),
        index,
    )
    selected_segment_indices = select_lattice_segment_indices(index.segments)
    samples = tuple(
        index.stateful_transition_sample(
            segment_index,
            index.segments[segment_index].start,
            action_horizon=50,
        )
        for segment_index in selected_segment_indices
    )
    if any(sample.transition_index != 0 for sample in samples):
        raise RuntimeError("lattice task bank must contain reset transitions only")

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
    if config.chunk_size != 50:
        raise RuntimeError("released LingBot action horizon differs from 50")
    if bool(config.train_expert_only) or bool(config.freeze_vision_encoder):
        raise RuntimeError("lattice probe requires the same complete trainable VLM host")
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

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    graph_config = LingBotNativeGraphConfig.from_policy(
        policy,
        capacity=args.capacity,
        maximum_control_tokens=args.maximum_control_tokens,
    )
    graph = LingBotNativeGraph(graph_config, device=device, dtype=dtype).train()
    install_lingbot_native_graph(policy, graph)
    parameter_signature = {
        name: (id(parameter), parameter._version) for name, parameter in policy.named_parameters()
    }

    args.visual_dir.mkdir(parents=True)
    arms: dict[str, dict[str, object]] = {}
    processor_contracts: dict[str, dict[str, object]] = {}
    for lattice in (LATTICE_BASELINE, LATTICE_CANDIDATE):
        processor_contract = configure_native_processor_lattice(processor, lattice)
        processor_contracts[str(lattice)] = processor_contract
        cache_invalidation = reset_native_visual_grid_cache(policy.model.qwenvl_with_expert)
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
        arm_samples: list[dict[str, object]] = []
        for sample in samples:
            augmentation_seed = _paired_seed(args.seed, sample.sample_key, "augmentation")
            flow_seed = _paired_seed(args.seed, sample.sample_key, "flow")
            raw_batch = build_native_calvin_training_batch(
                (sample,),
                lane_ids=(0,),
                optimizer_step=0,
                device=device,
                dtype=dtype,
            )
            collated = collate_native_calvin_training_batch(
                raw_batch,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=(augmentation_seed,),
                source_digest=hashlib.sha256(
                    f"{sample.sample_key}\0{augmentation_seed}".encode()
                ).hexdigest(),
            )
            model_inputs = _move_model_inputs(
                collated.model_inputs,
                device=device,
                dtype=dtype,
                torch_module=torch,
            )
            actions = model_inputs["actions"]
            generator = torch.Generator(device=device).manual_seed(flow_seed)
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

            context = LingBotNativeContext(
                controls=collated.controls,
                previous_state=None,
                previous_state_valid=torch.zeros(1, dtype=torch.bool, device=device),
            )
            torch.manual_seed(flow_seed)
            torch.cuda.manual_seed_all(flow_seed)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = run_native_policy_diagnostic_forward(
                policy,
                model_inputs=model_inputs,
                context=context,
            )
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - started
            relation = context.relation_output
            if relation is None or context.posterior_state is None:
                raise RuntimeError("lattice forward did not finalize PICF outputs")
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))

            target_bundle = build_native_calvin_sequence_target_bundle(
                requests_by_time=(collated.structural_target_requests,),
                model_inputs_by_time=(model_inputs,),
                relations=(relation,),
                physical_sidecar=physical_sidecar,
                capacity=args.capacity,
                task_identity_resolver=calvin_exact_task_loss_identities,
                patch_size=int(processor_contract["patch_size"]),
                merge_size=int(processor_contract["merge_size"]),
                minimum_supervised_fraction=0.0,
                capacity_seeds=(flow_seed,),
            )
            task_identities = calvin_exact_task_loss_identities(sample.host_sample.task_key)
            if task_identities is None:
                raise RuntimeError("lattice task bank contains an inexact task")
            identity_to_track = {
                key: index for index, key in enumerate(target_bundle.identity_keys_by_batch[0])
            }
            missing = sorted(set(task_identities) - set(identity_to_track))
            if missing:
                raise RuntimeError(f"lattice exact target identities are absent: {missing}")
            track_indices = [identity_to_track[key] for key in task_identities]
            target_mass_tensor = target_bundle.targets.masks[
                0,
                0,
                track_indices,
            ].amax(dim=0)
            structural_positions = relation.structural_valid[0]
            logits_tensor = relation.dense_task_grounding_logits[0, structural_positions]
            target_mass_tensor = target_mass_tensor[structural_positions]
            expected_tokens = 2 * lattice * lattice
            if logits_tensor.numel() != expected_tokens or target_mass_tensor.numel() != (
                expected_tokens
            ):
                raise RuntimeError("lattice structural token count differs from two valid views")
            logits = logits_tensor.detach().float().cpu().tolist()
            target_mass = target_mass_tensor.detach().float().cpu().tolist()
            metrics = fractional_token_metrics(logits, target_mass)
            image_sha256, source_images = _sample_images(sample)
            visual_name = (
                f"{_slug(sample.host_sample.task_key)}"
                f"__segment-{sample.record.task_index:08d}"
                f"__lattice-{lattice:02d}.png"
            )
            visual_payload = _render_sample_visual(
                source_images=source_images,
                task_key=sample.host_sample.task_key,
                instruction=str(sample.host_sample.observation["task"]),
                sample_key=sample.sample_key,
                lattice=lattice,
                logits=logits,
                mass=target_mass,
                metrics=metrics,
            )
            visual_path = args.visual_dir / visual_name
            write_bytes_durable_exclusive(visual_path, visual_payload)
            arm_samples.append(
                {
                    "sample_key": sample.sample_key,
                    "task_key": sample.host_sample.task_key,
                    "task": sample.host_sample.observation["task"],
                    "source_global_index": sample.host_sample.source_global_index,
                    "segment_index": sample.record.task_index,
                    "transition_index": sample.transition_index,
                    "target_identity_keys": list(task_identities),
                    "image_sha256": image_sha256,
                    "augmentation_seed": augmentation_seed,
                    "flow_seed": flow_seed,
                    "input_shapes": {
                        name: [int(item) for item in value.shape]
                        for name, value in sorted(model_inputs.items())
                    },
                    "image_grid_thw": model_inputs["image_grid_thw"][0].detach().cpu().tolist(),
                    "image_valid": model_inputs["img_masks"][0].detach().cpu().tolist(),
                    "dense_task_logits": logits,
                    "target_mass": target_mass,
                    "eligible": metrics["eligible"],
                    "metrics": metrics,
                    "official_action_loss": float(
                        result.official_action_loss.detach().float().item()
                    ),
                    "forward_seconds": forward_seconds,
                    "peak_cuda_allocated_bytes": peak_allocated,
                    "peak_cuda_reserved_bytes": peak_reserved,
                    "visual_file": visual_name,
                    "visual_sha256": hashlib.sha256(visual_payload).hexdigest(),
                }
            )
            del (
                context,
                logits_tensor,
                model_inputs,
                raw_batch,
                result,
                target_bundle,
                target_mass_tensor,
            )
        cache_invalidation["populated_after_arm"] = require_native_visual_grid_cache_populated(
            policy.model.qwenvl_with_expert
        )
        arms[str(lattice)] = {
            "lattice": lattice,
            "processor": processor_contract,
            "visual_grid_cache_invalidation": cache_invalidation,
            "samples": arm_samples,
        }

    if {
        name: (id(parameter), parameter._version) for name, parameter in policy.named_parameters()
    } != parameter_signature:
        raise RuntimeError("paired lattice arms changed a model parameter object or version")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise RuntimeError("read-only lattice probe unexpectedly populated parameter gradients")

    report: dict[str, Any] = {
        "schema": LATTICE_FEASIBILITY_SCHEMA,
        "repository_commit": _git_output(root, "rev-parse", "HEAD"),
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "source_patch_sha256": patch_report["patch_sha256"],
        "patched_source_sha256": patched_source_sha256,
        "source_diff_sha256": _source_diff_digest(args.source_checkout),
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "checkpoint_assets": checkpoint_report["checkpoint_assets"],
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "processor_assets": processor_report["processor_assets"],
        "processor_contracts": processor_contracts,
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "dataset_tree_sha256": dataset_manifest.tree_sha256,
        "dataset_runtime_binding": dataset_runtime_binding,
        "normalization_sha256": _sha256(args.norm_stats),
        "normalization_artifact_sha256": norm_stats_payload["artifact_sha256"],
        "physical_sidecar_manifest_sha256": _sha256(args.physical_sidecar / "manifest.json"),
        "baseline_lattice": LATTICE_BASELINE,
        "candidate_lattice": LATTICE_CANDIDATE,
        "loss_only_supervision": True,
        "target_resolution_happened_after_forward": True,
        "optimizer_created": False,
        "checkpoint_mutated": False,
        "same_parameter_objects_across_arms": True,
        "target_or_mask_fields_in_model_inputs": [],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "transformers_version": transformers_version,
        "load_model_seconds": load_seconds,
        "alignment_teacher_prune": alignment_teacher_prune,
        "selected_segment_indices": list(selected_segment_indices),
        "arms": arms,
    }
    decision = lattice_feasibility_decision(report)
    report.update(decision)
    report["failures"] = sorted(name for name, passed in report["gates"].items() if not passed)
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    report = validate_lattice_feasibility_report(report)
    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    write_text_durable_exclusive(args.output.resolve(), payload)
    print(payload, end="")
    if report["status"] != "PASS":
        raise RuntimeError(
            "released-weight lattice feasibility probe failed: " + ", ".join(report["failures"])
        )


if __name__ == "__main__":
    main()
