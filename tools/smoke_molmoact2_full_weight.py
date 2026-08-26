#!/usr/bin/env python3
"""Run the released MolmoAct2 one-pass PICF parity gate on a real image.

This tool requires the pinned MolmoAct2 source checkout on ``PYTHONPATH``. It
uses upstream processor, prompt and model implementations directly; no training
label, mask, sidecar or simulator object ID enters the forward input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from olmo.hf_model.modeling_molmoact2 import (
    MolmoAct2ForConditionalGeneration,
    _build_discrete_state_string,
    _build_robot_text,
)
from PIL import Image
from transformers import AutoProcessor

from picf_next.hosts.context import PICFActionEvidence
from picf_next.hosts.molmoact2 import MolmoAct2PICFForConditionalGeneration

_MODEL_INPUT_KEYS = {
    "input_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
    "attention_mask",
    "token_type_ids",
}


def _strict_revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("revision must be an exact lowercase 40-character commit SHA")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/MolmoAct2")
    parser.add_argument("--revision", default="e432d85f6e039edca44afb93c262f3084ab72a9c")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move the red block to the left")
    parser.add_argument("--setup-type", default="tabletop manipulation")
    parser.add_argument("--control-mode", default="end effector delta pose")
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _resolve_snapshot(args: argparse.Namespace) -> Path:
    revision = _strict_revision(args.revision)
    local = Path(args.model).expanduser()
    if local.exists():
        return local.resolve()
    # revision is strict-validated and the resolved snapshot SHA is checked below.
    snapshot_path = snapshot_download(  # nosec B615
        repo_id=args.model,
        revision=revision,
        local_files_only=args.local_files_only,
    )
    snapshot = Path(snapshot_path).resolve()
    if snapshot.name != revision:
        raise RuntimeError(
            f"resolved snapshot {snapshot.name} differs from pinned revision {revision}"
        )
    return snapshot


def _move_inputs(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in dict(inputs).items():
        if key in _MODEL_INPUT_KEYS and torch.is_tensor(value):
            moved[key] = value.to(device)
    if "input_ids" not in moved:
        raise RuntimeError("MolmoAct2 processor produced no input_ids")
    return moved


def _memory_report(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    args = _parse_args()
    if args.state_dim <= 0 or args.num_steps <= 0:
        raise ValueError("state-dim and num-steps must be positive")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings: dict[str, float] = {}
    started = time.perf_counter()
    snapshot = _resolve_snapshot(args)
    timings["resolve_snapshot_s"] = time.perf_counter() - started

    started = time.perf_counter()
    # snapshot is local and _strict_revision rejects mutable names.
    processor = AutoProcessor.from_pretrained(  # nosec B615
        snapshot,
        revision=_strict_revision(args.revision),
        trust_remote_code=True,
        extra_special_tokens={},
    )
    host = MolmoAct2ForConditionalGeneration.from_pretrained(
        snapshot,
        revision=_strict_revision(args.revision),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    host.eval()
    timings["load_model_s"] = time.perf_counter() - started

    vision_width = host.model.vision_backbone.vit_config.hidden_size * len(
        host.model.vision_backbone.vit_layers
    )
    wrapper = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"molmo_vision_patch": vision_width},
        object_address_dim=1,
        object_value_dim=1,
    ).eval()

    state = np.zeros(args.state_dim, dtype=np.float32)
    text = _build_robot_text(
        task=args.task,
        style="robot_action",
        discrete_state_string=_build_discrete_state_string(
            state,
            int(host.config.num_state_tokens),
        ),
        setup_type=args.setup_type,
        control_mode=args.control_mode,
        add_setup_tokens=bool(host.config.add_setup_tokens),
        add_control_tokens=bool(host.config.add_control_tokens),
        num_images=1,
    )
    image = Image.open(args.image).convert("RGB")
    inputs = _move_inputs(
        processor(text=text, images=[image], return_tensors="pt"),
        device,
    )

    started = time.perf_counter()
    with torch.inference_mode():
        expected = host.model.generate_actions_from_inputs(
            **inputs,
            num_steps=args.num_steps,
            generator=torch.Generator(device=device).manual_seed(20260714),
        )
    timings["official_action_s"] = time.perf_counter() - started

    started = time.perf_counter()
    with torch.inference_mode():
        bundle = wrapper.encode_inputs_for_picf(**inputs)
    timings["one_pass_encoder_s"] = time.perf_counter() - started
    if bundle.vision_patch_bank is None:
        raise RuntimeError("one-pass encoder returned no dense vision bank")

    started = time.perf_counter()
    with torch.inference_mode():
        actual = wrapper.generate_actions_from_inputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            evidence=PICFActionEvidence(
                dense_banks=(bundle.vision_patch_bank,),
                object_address=None,
                object_value=None,
                object_valid=None,
            ),
            encoder_kv_states=bundle.encoder_kv_states,
            encoder_attention_mask=bundle.encoder_attention_mask,
            num_steps=args.num_steps,
            generator=torch.Generator(device=device).manual_seed(20260714),
        )
    timings["bundle_action_s"] = time.perf_counter() - started

    max_abs_error = float((actual.float() - expected.float()).abs().max().item())
    report = {
        "schema": "picf-next.molmoact2-full-weight-smoke.v1",
        "model_argument": args.model,
        "requested_revision": args.revision,
        "snapshot": str(snapshot),
        "snapshot_revision": snapshot.name,
        "image": str(args.image.resolve()),
        "image_sha256": _sha256(args.image),
        "task": args.task,
        "setup_type": args.setup_type,
        "control_mode": args.control_mode,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        "dtype": str(dtype),
        "num_steps": args.num_steps,
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "encoder_kv_layers": len(bundle.encoder_kv_states),
        "vision_patch_shape": list(bundle.vision_patch_bank.tokens.shape),
        "vision_patch_valid": int(bundle.vision_patch_bank.valid.sum().item()),
        "vision_patch_width": vision_width,
        "official_action_shape": list(expected.shape),
        "official_action_sha256": _tensor_sha256(expected),
        "bundle_action_sha256": _tensor_sha256(actual),
        "zero_gate_action_bitwise_equal": bool(torch.equal(actual, expected)),
        "zero_gate_action_max_abs_error": max_abs_error,
        "timings": timings,
        "cuda_memory_bytes": _memory_report(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["zero_gate_action_bitwise_equal"]:
        raise RuntimeError(
            f"released-checkpoint zero-gate action parity failed; max_abs_error={max_abs_error}"
        )


if __name__ == "__main__":
    main()
