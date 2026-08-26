#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the pinned LingBot-VLA 2.0 released-weight PICF parity gate.

The heavy LingBot/Torch imports are intentionally delayed until ``main`` so
the command contract can be tested on a CPU-only development machine. The
runtime path copies the released deployment preprocessing and model
construction, but uses LingBot's streaming weight loader to avoid assembling a
second full state dict in host memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    from tools.verify_lingbot_vla2_patch import detect_patch_state
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_REVISION,
        LINGBOT_SOURCE_COMMIT,
        QWEN_PROCESSOR_REVISION,
        REQUIRED_CHECKPOINT_FILES,
        REQUIRED_PROCESSOR_FILES,
        validate_checkpoint,
        validate_processor,
    )
    from verify_lingbot_vla2_patch import detect_patch_state

TARGET_ONLY_FIELDS = frozenset(
    {
        "action",
        "actions",
        "action_is_pad",
        "joint_mask",
        "mask",
        "masks",
        "object_id",
        "segmentation",
        "teacher",
        "target",
        "targets",
    }
)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=root / "references/source_checkouts/lingbot-vla-v2",
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=root / "references/patches/lingbot_vla2_action_layer_adapter.patch",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root
        / "references/source_checkouts/lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=root
        / "references/source_checkouts/lingbot-vla-v2/configs/robot_configs/robotwin.yaml",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="pick up the red block")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dense-width", type=int, default=32)
    parser.add_argument("--object-address-width", type=int, default=16)
    parser.add_argument("--object-value-width", type=int, default=32)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().to(dtype=tensor.new_zeros(()).float().dtype, device="cpu")
    return hashlib.sha256(array.contiguous().numpy().tobytes()).hexdigest()


def _merge_training_sections(training: dict[str, Any]) -> dict[str, Any]:
    """Match LingBot deploy's model-then-train config merge without mutation."""

    model = training.get("model")
    train = training.get("train")
    if not isinstance(model, dict) or not isinstance(train, dict):
        raise ValueError("LingBot training YAML must contain model and train mappings")
    merged = deepcopy(model)
    merged.update(deepcopy(train))
    return merged


def _resolve_training_config(
    training: dict[str, Any],
    *,
    checkpoint_dir: Path,
    processor_dir: Path,
    num_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve every external release path while preserving official choices."""

    if num_steps <= 0:
        raise ValueError("num-steps must be positive")
    resolved = deepcopy(training)
    merged = _merge_training_sections(resolved)
    merged["tokenizer_path"] = str(processor_dir.resolve())
    merged["model_path"] = str(checkpoint_dir.resolve())
    merged["use_cache"] = True
    merged["use_compile"] = False
    merged["attention_implementation"] = "eager"
    # FlashAttention is not a semantic dependency of this parity gate. Eager
    # vision attention removes a non-pinned extension from the G0 contract.
    merged["vit_attn_implementation"] = "eager"
    merged["num_steps"] = num_steps

    align = deepcopy(merged.get("align_params") or {})
    depth = deepcopy(align.get("depth") or {})
    video = deepcopy(align.get("video") or {})
    depth["morgbd_path"] = str((checkpoint_dir / "depth/model.pt").resolve())
    video["ckpt_path"] = str((checkpoint_dir / "dino_video/teacher_step_10000.pth").resolve())
    video["config_path"] = str((checkpoint_dir / "dino_video/config.yaml").resolve())
    if depth:
        align["depth"] = depth
    if video:
        align["video"] = video
    if align:
        merged["align_params"] = align

    data = resolved.get("data")
    if not isinstance(data, dict):
        raise ValueError("LingBot training YAML must contain a data mapping")
    return merged, deepcopy(data)


def _asset_manifest(root: Path, required: tuple[str, ...]) -> list[dict[str, Any]]:
    manifest = []
    for relative in sorted(required):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return manifest


def _cuda_memory(torch: Any, device: Any) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _merge_qwen_config(config: Any, qwen_config: Any) -> None:
    """Copy the exact Qwen fields used by LingBot's released deploy wrapper."""

    config_dict = qwen_config.to_dict()
    text_keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    }
    text_config = config_dict.get("text_config", {})
    for key in text_keys:
        if key in text_config:
            setattr(config, key, text_config[key])
        elif key in config_dict:
            setattr(config, key, config_dict[key])
    if "vision_config" not in config_dict:
        raise ValueError("pinned Qwen config contains no vision_config")
    config.vision_config = qwen_config.vision_config


class _RouteTrace:
    """Hash LingBot action-MoE top-k choices without changing its forward."""

    def __init__(self, torch: Any, blocks: list[Any]) -> None:
        self._torch = torch
        self._blocks = blocks
        self._digest = hashlib.sha256()
        self.calls = 0
        self.tokens = 0
        self._handles = [
            block.register_forward_pre_hook(self._hook(index, block))
            for index, block in enumerate(blocks)
        ]

    def _hook(self, layer_index: int, block: Any):
        torch = self._torch

        def capture(_module: Any, args: tuple[Any, ...]) -> None:
            hidden = args[0].detach()
            flat = hidden.reshape(-1, hidden.shape[-1])
            with torch.amp.autocast(flat.device.type, enabled=False):
                logits = torch.nn.functional.linear(flat.float(), block.gate.weight.float())
            if block._router_activation == "sigmoid":
                scores = logits.sigmoid()
            else:
                scores = torch.nn.functional.softmax(logits, dim=1, dtype=torch.float32)
            scores = scores + block.e_score_correction_bias.unsqueeze(0)
            selected = torch.topk(scores, block.top_k, dim=-1).indices
            header = f"{layer_index}:{tuple(hidden.shape)}:".encode()
            self._digest.update(header)
            self._digest.update(selected.to(dtype=torch.int64, device="cpu").numpy().tobytes())
            self.calls += 1
            self.tokens += int(selected.shape[0])

        return capture

    def finish(self) -> dict[str, Any]:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return {
            "sha256": self._digest.hexdigest(),
            "calls": self.calls,
            "tokens": self.tokens,
            "layers": len(self._blocks),
        }


def _git_output(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    for path in (args.source_checkout, args.patch, args.config, args.robot_config, args.image):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.dense_width <= 0 or args.object_address_width <= 0 or args.object_value_width <= 0:
        raise ValueError("PICF probe widths must be positive")
    if _git_output(args.source_checkout, "rev-parse", "HEAD") != LINGBOT_SOURCE_COMMIT:
        raise RuntimeError("LingBot source checkout differs from the pinned commit")
    if detect_patch_state(args.source_checkout, args.patch) != "applied":
        raise RuntimeError("LingBot action/data patch must be applied before G0")
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

    from picf_next.hosts.context import PICFActionEvidence
    from picf_next.hosts.lingbot_vla2 import (
        LingBotVLA2PICFAdapter,
        install_lingbot_vla2_picf_adapter,
    )
    from picf_next.models.evidence import NativeTokenBank

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LingBot released-weight G0 requires a CUDA device")
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

    blocks = [layer.mlp for layer in policy.model.qwenvl_with_expert.qwen_expert.model.layers]
    official_trace = _RouteTrace(torch, blocks)
    started = time.perf_counter()
    with torch.inference_mode():
        official = policy.sample_actions(**model_inputs, noise=noise.clone())
    torch.cuda.synchronize(device)
    timings["official_action_s"] = time.perf_counter() - started
    official_routes = official_trace.finish()

    expert = policy.model.qwenvl_with_expert.config.qwen_expert_config
    adapter = LingBotVLA2PICFAdapter(
        hidden_size=expert.hidden_size,
        num_layers=expert.num_hidden_layers,
        num_attention_heads=expert.num_attention_heads,
        num_key_value_heads=expert.num_key_value_heads,
        head_dim=expert.head_dim,
        dense_token_dims={"synthetic_probe": args.dense_width},
        object_address_dim=args.object_address_width,
        object_value_dim=args.object_value_width,
        device=device,
        dtype=dtype,
    )
    install_lingbot_vla2_picf_adapter(policy, adapter)
    dense_tokens = torch.linspace(
        -1,
        1,
        steps=3 * args.dense_width,
        device=device,
        dtype=dtype,
    ).view(1, 3, args.dense_width)
    object_address = torch.linspace(
        -0.5,
        0.5,
        steps=2 * args.object_address_width,
        device=device,
        dtype=dtype,
    ).view(1, 2, args.object_address_width)
    object_value = torch.linspace(
        0.5,
        -0.5,
        steps=2 * args.object_value_width,
        device=device,
        dtype=dtype,
    ).view(1, 2, args.object_value_width)
    evidence = PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                "synthetic_probe",
                dense_tokens,
                torch.ones(1, 3, dtype=torch.bool, device=device),
            ),
        ),
        object_address=object_address,
        object_value=object_value,
        object_valid=torch.ones(1, 2, dtype=torch.bool, device=device),
    )
    context = adapter.prepare_picf_context(evidence)
    picf_trace = _RouteTrace(torch, blocks)
    started = time.perf_counter()
    with torch.inference_mode():
        actual = policy.sample_actions(
            **model_inputs,
            noise=noise.clone(),
            action_layer_context=context,
        )
    torch.cuda.synchronize(device)
    timings["zero_gate_picf_action_s"] = time.perf_counter() - started
    picf_routes = picf_trace.finish()

    max_abs_error = float((actual.float() - official.float()).abs().max().item())
    expected_calls = int(args.num_steps * len(blocks))
    report = {
        "schema": "picf-next.lingbot-vla2-full-weight-smoke.v1",
        "source_commit": LINGBOT_SOURCE_COMMIT,
        "source_patch_sha256": _sha256(args.patch),
        "source_diff_sha256": hashlib.sha256(
            _git_output(args.source_checkout, "diff", "--binary").encode()
        ).hexdigest(),
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
        "input_shapes": {key: list(value.shape) for key, value in model_inputs.items()},
        "official_action_shape": list(official.shape),
        "official_action_sha256": _tensor_sha256(official),
        "zero_gate_picf_action_sha256": _tensor_sha256(actual),
        "zero_gate_action_bitwise_equal": bool(torch.equal(actual, official)),
        "zero_gate_action_max_abs_error": max_abs_error,
        "official_routes": official_routes,
        "zero_gate_picf_routes": picf_routes,
        "expected_action_moe_calls": expected_calls,
        "zero_gate_route_bitwise_equal": official_routes == picf_routes,
        "dense_gates_zero": bool(torch.count_nonzero(adapter.dense_gates).item() == 0),
        "object_gates_zero": bool(torch.count_nonzero(adapter.object_gates).item() == 0),
        "timings": timings,
        "cuda_memory_bytes": _cuda_memory(torch, device),
        "pid": os.getpid(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    failures = []
    if not report["zero_gate_action_bitwise_equal"]:
        failures.append(f"action max_abs_error={max_abs_error}")
    if not report["zero_gate_route_bitwise_equal"]:
        failures.append("MoE route digest/call count differs")
    if official_routes["calls"] != expected_calls:
        failures.append(
            f"official MoE calls {official_routes['calls']} != expected {expected_calls}"
        )
    if failures:
        raise RuntimeError("LingBot released-weight G0 failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
