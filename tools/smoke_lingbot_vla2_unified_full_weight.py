#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the released-weight LingBot unified-PICF G0 integration smoke.

Accelerator and LingBot imports stay inside ``main`` so command construction,
source verification and tests remain runnable on a CPU-only workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_REVISION,
        LINGBOT_SOURCE_COMMIT,
        QWEN_PROCESSOR_REVISION,
        REQUIRED_CHECKPOINT_FILES,
        REQUIRED_PROCESSOR_FILES,
        validate_checkpoint,
        validate_processor,
    )
    from tools.smoke_lingbot_vla2_full_weight import (
        TARGET_ONLY_FIELDS,
        _asset_manifest,
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
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
        REQUIRED_CHECKPOINT_FILES,
        REQUIRED_PROCESSOR_FILES,
        validate_checkpoint,
        validate_processor,
    )
    from smoke_lingbot_vla2_full_weight import (  # type: ignore[no-redef]
        TARGET_ONLY_FIELDS,
        _asset_manifest,
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
    )
    from verify_lingbot_vla2_patch import detect_patch_state  # type: ignore[no-redef]
    from verify_lingbot_vla2_unified_patch import (  # type: ignore[no-redef]
        DATA_PATCH_RELATIVE_PATH,
        GRAPH_PATCH_RELATIVE_PATH,
        verify_unified_patches,
    )


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = Path(
        os.environ.get(
            "PICF_LINGBOT_SOURCE",
            root / "references/source_checkouts/lingbot-vla-v2-unified",
        )
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=source_default,
    )
    parser.add_argument("--data-patch", type=Path, default=root / DATA_PATCH_RELATIVE_PATH)
    parser.add_argument("--graph-patch", type=Path, default=root / GRAPH_PATCH_RELATIVE_PATH)
    parser.add_argument(
        "--config",
        type=Path,
        default=source_default / "configs/vla/robotwin/robotwin.yaml",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=source_default / "configs/robot_configs/robotwin.yaml",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="pick up the red block")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--content-width", type=int, default=256)
    parser.add_argument("--geometry-width", type=int, default=6)
    parser.add_argument("--uncertainty-width", type=int, default=16)
    return parser.parse_args()


def _source_diff_digest(checkout: Path) -> str:
    return hashlib.sha256(_git_output(checkout, "diff", "--binary").encode()).hexdigest()


def _write_text_durable(path: Path, payload: str) -> None:
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


def _validate_unified_dimensions(
    *,
    capacity: int,
    content_width: int,
    geometry_width: int,
    uncertainty_width: int,
    num_steps: int,
) -> None:
    if min(capacity, content_width, geometry_width, uncertainty_width, num_steps) <= 0:
        raise ValueError("unified graph dimensions and num-steps must be positive")
    if geometry_width != 6:
        raise ValueError("the G0 camera-frame geometry schema is exactly six-dimensional")


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    required_paths = (
        args.source_checkout,
        args.data_patch,
        args.graph_patch,
        args.config,
        args.robot_config,
        args.image,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    _validate_unified_dimensions(
        capacity=args.capacity,
        content_width=args.content_width,
        geometry_width=args.geometry_width,
        uncertainty_width=args.uncertainty_width,
        num_steps=args.num_steps,
    )
    patch_report = verify_unified_patches(root=root, checkout=args.source_checkout)
    if _git_output(args.source_checkout, "rev-parse", "HEAD") != LINGBOT_SOURCE_COMMIT:
        raise RuntimeError("LingBot unified source checkout differs from the pinned commit")
    patch_states = [
        detect_patch_state(args.source_checkout, patch)
        for patch in (args.data_patch, args.graph_patch)
    ]
    if patch_states != ["applied", "applied"]:
        raise RuntimeError(f"LingBot unified patches are not both applied: {patch_states}")
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
        raise RuntimeError("LingBot unified source differs from replayed patch bytes")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
    import torch
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
    from PIL import Image
    from torchvision.transforms.v2 import Resize
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    try:
        from tools.lingbot_vla2_runtime_helpers import load_lingbot_training_config
    except ModuleNotFoundError:
        from lingbot_vla2_runtime_helpers import load_lingbot_training_config

    from picf_next.hosts.lingbot_unified import (
        LingBotHostContract,
        LingBotUnifiedBeliefGraph,
        LingBotUnifiedContext,
        LingBotUnifiedGraphConfig,
        install_lingbot_unified_belief_graph,
    )
    from picf_next.unified.codec import BeliefCodecConfig
    from picf_next.unified.graph import TokenRole
    from picf_next.unified.state import (
        GeometrySchema,
        deterministic_birth_noise,
        empty_belief_state,
    )
    from picf_next.unified.temporal import assert_deploy_payload_is_causal

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("released-weight unified G0 requires a CUDA device")
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
        SimpleNamespace(**data_mapping),
        config,
        processor,
        chunk_size=config.chunk_size,
        norm_stats_path=str((args.source_checkout / "assets/norm_stats/robotwin.json").resolve()),
    )
    image = Image.open(args.image).convert("RGB")
    chw = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
    chw = Resize((256, 256))(chw)
    raw_observation = {
        "observation.images.cam_high": chw,
        "observation.images.cam_left_wrist": chw,
        "observation.images.cam_right_wrist": chw,
        "observation.state": torch.zeros(14, dtype=torch.float32),
        "task": args.task,
    }
    forbidden = sorted(TARGET_ONLY_FIELDS.intersection(raw_observation))
    if forbidden:
        raise RuntimeError(f"target-only fields entered the G0 observation: {forbidden}")
    assert_deploy_payload_is_causal(raw_observation)
    observation = feature_transform.apply(raw_observation, policy_eval=True)
    model_inputs = {
        "images": observation["images"].unsqueeze(0).to(device=device, dtype=dtype),
        "img_masks": observation["img_masks"].unsqueeze(0).to(device=device),
        "lang_tokens": observation["lang_tokens"].unsqueeze(0).to(device=device),
        "lang_masks": observation["lang_masks"].unsqueeze(0).to(device=device),
        "state": observation["state"].unsqueeze(0).to(device=device, dtype=dtype),
        "image_grid_thw": observation["image_grid_thw"].to(device=device),
    }
    noise = torch.randn(
        (1, config.n_action_steps, config.max_action_dim),
        generator=torch.Generator(device=device).manual_seed(args.seed),
        device=device,
        dtype=dtype,
    )

    started = time.perf_counter()
    with torch.inference_mode():
        official = policy.sample_actions(**model_inputs, noise=noise.clone())
    torch.cuda.synchronize(device)
    timings["official_action_s"] = time.perf_counter() - started

    contract = LingBotHostContract.from_policy(policy)
    codec_config = BeliefCodecConfig(
        content_dim=args.content_width,
        geometry_dim=args.geometry_width,
        uncertainty_dim=args.uncertainty_width,
        host_width=contract.prefix_width,
    )
    graph_config = LingBotUnifiedGraphConfig.from_policy(
        policy,
        codec=codec_config,
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
    graph = LingBotUnifiedBeliefGraph(graph_config)
    install_lingbot_unified_belief_graph(policy, graph)

    started = time.perf_counter()
    with torch.inference_mode():
        neutral = policy.sample_actions(**model_inputs, noise=noise.clone())
    torch.cuda.synchronize(device)
    timings["installed_neutral_action_s"] = time.perf_counter() - started

    initial_state = empty_belief_state(
        batch_size=1,
        capacity=args.capacity,
        content_dim=args.content_width,
        geometry_dim=args.geometry_width,
        uncertainty_dim=args.uncertainty_width,
        device=device,
        dtype=dtype,
    )
    context = LingBotUnifiedContext(
        previous_posterior=initial_state,
        modality_geometry_valid=torch.zeros(
            1,
            1,
            args.capacity,
            args.geometry_width,
            dtype=torch.bool,
            device=device,
        ),
        elapsed_time=torch.zeros(1, device=device),
        previous_executed_action=torch.zeros(
            1,
            contract.executed_action_dim,
            device=device,
        ),
        previous_action_valid=torch.zeros(1, dtype=torch.bool, device=device),
        birth_proposal_noise=deterministic_birth_noise(
            episode_keys=("g0-fixed-episode",),
            frame_indices=(0,),
            capacity=args.capacity,
            content_dim=args.content_width,
            base_seed=args.seed,
            device=device,
        ),
    )
    started = time.perf_counter()
    with torch.inference_mode():
        unified = policy.sample_actions(
            **model_inputs,
            noise=noise.clone(),
            unified_belief_context=context,
        )
    torch.cuda.synchronize(device)
    timings["unified_action_s"] = time.perf_counter() - started

    posterior = context.posterior
    if posterior is None or context.predictive_prior is None or context.final_action_pair is None:
        raise RuntimeError("unified graph did not publish its declared state-write outputs")
    posterior.validate()
    lifecycle_error = float((posterior.lifecycle_probs.sum(dim=-1) - 1).abs().max().item())
    minimum_information_eigenvalue = float(
        torch.linalg.eigvalsh(posterior.geometry_information).min().item()
    )
    serialized = posterior.serialize()
    neutral_error = float((neutral.float() - official.float()).abs().max().item())
    unified_finite = bool(torch.isfinite(unified).all())
    report = {
        "schema": "picf-next.lingbot-vla2-unified-full-weight-smoke.v1",
        "source_commit": LINGBOT_SOURCE_COMMIT,
        "patch_states": patch_states,
        "patched_source_sha256": actual_source_hashes,
        "data_patch_sha256": _sha256(args.data_patch),
        "graph_patch_sha256": _sha256(args.graph_patch),
        "source_diff_sha256": _source_diff_digest(args.source_checkout),
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "checkpoint_assets": _asset_manifest(args.checkpoint_dir, tuple(REQUIRED_CHECKPOINT_FILES)),
        "processor_assets": _asset_manifest(args.processor_dir, tuple(REQUIRED_PROCESSOR_FILES)),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "image": str(args.image.resolve()),
        "image_sha256": _sha256(args.image),
        "task": args.task,
        "target_only_fields_present": forbidden,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "num_steps": args.num_steps,
        "host_contract": {
            "prefix_width": contract.prefix_width,
            "attention_value_width": contract.attention_value_width,
            "num_layers": contract.num_layers,
            "executed_action_dim": contract.executed_action_dim,
            "native_training_query_tokens": contract.native_training_query_tokens,
        },
        "belief_capacity": args.capacity,
        "input_shapes": {key: list(value.shape) for key, value in model_inputs.items()},
        "official_action_sha256": _tensor_sha256(official),
        "installed_neutral_action_sha256": _tensor_sha256(neutral),
        "neutral_action_bitwise_equal": bool(torch.equal(neutral, official)),
        "neutral_action_max_abs_error": neutral_error,
        "unified_action_sha256": _tensor_sha256(unified),
        "unified_action_finite": unified_finite,
        "posterior_serialized_bytes": len(serialized),
        "posterior_sha256": hashlib.sha256(serialized).hexdigest(),
        "lifecycle_normalization_max_abs_error": lifecycle_error,
        "minimum_geometry_information_eigenvalue": minimum_information_eigenvalue,
        "native_sensor_tokens": int(
            (context.native_roles == int(TokenRole.SENSOR)).sum().item()
            if context.native_roles is not None
            else -1
        ),
        "final_pair_shape": list(context.final_action_pair.tokens.shape),
        "timings": timings,
        "cuda_memory_bytes": _cuda_memory(torch, device),
        "pid": os.getpid(),
    }
    failures = []
    if not report["neutral_action_bitwise_equal"]:
        failures.append(f"neutral action max_abs_error={neutral_error}")
    if not unified_finite:
        failures.append("unified action contains NaN or infinity")
    if lifecycle_error > 1e-5:
        failures.append(f"lifecycle normalization error={lifecycle_error}")
    if minimum_information_eigenvalue < -1e-5:
        failures.append(f"geometry information is non-PSD: {minimum_information_eigenvalue}")
    if report["native_sensor_tokens"] <= 0:
        failures.append("no native visual token was typed as a physical sensor")
    report["status"] = "FAIL" if failures else "PASS"
    report["failures"] = failures
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(payload, end="")
    if failures:
        raise RuntimeError("LingBot unified released-weight G0 failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
