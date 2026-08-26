#!/usr/bin/env python3
"""Run strict same-forward M0 on the patched MolmoAct2 LeRobot policy.

The released vision backbone must produce both its unchanged pooled VLM input
and all 729 pre-pooling patches in one differentiable pass. Synthetic object
rows exercise the second typed context domain, but the dense evidence is the
real checkpoint representation. M0 accepts only exact fixed-noise parity among
the raw official path, the prepared native path and zero-gate PICF.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
for _path in (_ROOT, _SOURCE_ROOT):
    while str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

MOLMO_SOURCE_COMMIT = "c2282820f9b188b60e66ea1636b3efd81c45cbb4"
MOLMO_LEROBOT_COMMIT = "80633827176a0203064cb141383664fba024e050"
MOLMO_CHECKPOINT_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"
M0_MODALITY = "molmo_vision_patch"
MOLMO_LEROBOT_MODEL_SOURCE = "src/lerobot/policies/molmoact2/modeling_molmoact2.py"

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


@contextmanager
def _production_inference_context(torch: Any, device: Any, dtype: Any) -> Any:
    """Match the mixed-precision boundary used by Accelerate training."""

    enabled = dtype == torch.bfloat16
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled),
    ):
        yield


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=_ROOT / "references/source_checkouts/molmoact2-cloud",
    )
    parser.add_argument(
        "--lerobot-checkout",
        type=Path,
        default=_ROOT / "references/source_checkouts/molmoact2-lerobot-cloud",
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=_ROOT / "references/patches/molmoact2_lerobot_action_layer_adapter.patch",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move the red block to the left")
    parser.add_argument("--setup-type", default="tabletop manipulation")
    parser.add_argument("--control-mode", default="end effector delta pose")
    parser.add_argument("--training-recipe", type=Path, required=True)
    parser.add_argument("--training-recipe-sha256", required=True)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dense-token-count", type=int, default=729)
    parser.add_argument("--dense-token-width", type=int, default=2304)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--object-address-width", type=int, default=64)
    parser.add_argument("--object-value-width", type=int, default=784)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--skip-weight-shard-hashes",
        action="store_true",
        help="Diagnostic escape hatch only; the cloud M0 launcher never sets this.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty_paths(path: Path) -> set[str]:
    output = subprocess.run(
        ["git", "-C", str(path), "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line[3:] for line in output.splitlines() if line}


def _tensor_sha256(tensor: Any) -> str:
    import torch

    if tensor.dtype == torch.bfloat16:
        data = tensor.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
    else:
        data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _tensor_manifest(tensors: dict[str, Any]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for key, value in sorted(tensors.items())
    }


def _memory_report(torch: Any, device: Any) -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _timed_cuda_call(torch: Any, device: Any, operation: Any) -> tuple[Any, float, dict[str, int]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return result, elapsed, _memory_report(torch, device)


def _validate_positive_args(args: argparse.Namespace) -> None:
    values = {
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
        "num_steps": args.num_steps,
        "dense_token_count": args.dense_token_count,
        "dense_token_width": args.dense_token_width,
        "object_count": args.object_count,
        "object_address_width": args.object_address_width,
        "object_value_width": args.object_value_width,
    }
    bad = {name: value for name, value in values.items() if value <= 0}
    if bad:
        raise ValueError(f"M0 dimensions must be positive: {bad}")
    if args.action_dim > 32:
        raise ValueError("released MolmoAct2 supports at most 32 action dimensions")


def _validate_training_recipe(args: argparse.Namespace) -> str:
    from picf_next.training.recipe import load_training_recipe

    recipe = load_training_recipe(args.training_recipe.expanduser().resolve())
    if recipe.recipe_sha256 != args.training_recipe_sha256:
        raise ValueError("M0 training recipe SHA-256 differs from its launcher contract")
    expected = {
        "action_dim": recipe.host.action_dim,
        "dense_token_width": recipe.core_config.dense_token_dims[recipe.host.dense_modality],
        "object_count": recipe.core_config.posterior_capacity,
        "object_address_width": recipe.core_config.object_address_dim,
        "object_value_width": recipe.core_config.object_value_dim,
    }
    observed = {name: getattr(args, name) for name in expected}
    if observed != expected:
        raise ValueError(
            f"M0 adapter dimensions differ from the training recipe: "
            f"expected={expected}, observed={observed}"
        )
    return recipe.recipe_sha256


def _validate_immutable_assets(args: argparse.Namespace) -> dict[str, object]:
    from tools.bootstrap_molmoact2 import validate_checkpoint
    from tools.verify_molmoact2_lerobot_patch import detect_patch_state

    source = args.source_checkout.expanduser().resolve()
    trainer = args.lerobot_checkout.expanduser().resolve()
    patch = args.patch.expanduser().resolve()
    checkpoint = args.checkpoint_dir.expanduser().resolve()
    if _git_head(source) != MOLMO_SOURCE_COMMIT:
        raise RuntimeError("MolmoAct2 source checkout differs from the M0 pin")
    if _git_head(trainer) != MOLMO_LEROBOT_COMMIT:
        raise RuntimeError("MolmoAct2 LeRobot checkout differs from the M0 pin")
    if detect_patch_state(trainer, patch) != "applied":
        raise RuntimeError("MolmoAct2 LeRobot action-layer patch is not exactly applied")
    source_dirty = _git_dirty_paths(source)
    trainer_dirty = _git_dirty_paths(trainer)
    if source_dirty:
        raise RuntimeError(f"MolmoAct2 source checkout has local modifications: {source_dirty}")
    if trainer_dirty != {MOLMO_LEROBOT_MODEL_SOURCE}:
        raise RuntimeError(
            f"MolmoAct2 trainer checkout differs from the one-file patch contract: {trainer_dirty}"
        )
    return {
        "source_checkout": str(source),
        "source_commit": MOLMO_SOURCE_COMMIT,
        "trainer_checkout": str(trainer),
        "trainer_commit": MOLMO_LEROBOT_COMMIT,
        "patch": str(patch),
        "patch_sha256": _sha256(patch),
        "patch_state": "applied",
        "source_dirty_paths": [],
        "trainer_dirty_paths": sorted(trainer_dirty),
        "checkpoint": validate_checkpoint(
            checkpoint,
            validate_weight_shards=not args.skip_weight_shard_hashes,
        ),
    }


def _move_processor_inputs(inputs: Any, device: Any, torch: Any) -> dict[str, Any]:
    moved = {}
    for key, value in dict(inputs).items():
        if key in _MODEL_INPUT_KEYS and torch.is_tensor(value):
            moved[key] = value.to(device)
    if "input_ids" not in moved:
        raise RuntimeError("MolmoAct2 processor produced no input_ids")
    forbidden = set(moved) - _MODEL_INPUT_KEYS
    if forbidden:
        raise RuntimeError(f"processor input boundary admitted forbidden keys: {forbidden}")
    return moved


def _make_m0_evidence(
    *,
    torch: Any,
    picf_action_evidence: Any,
    dense_bank: Any,
    device: Any,
    dtype: Any,
    seed: int,
    object_count: int,
    address_width: int,
    value_width: int,
) -> tuple[Any, dict[str, Any]]:
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    tokens = dense_bank.tokens
    valid = dense_bank.valid
    if tokens.shape[0] != 1 or tokens.device != device or tokens.dtype != dtype:
        raise ValueError("M0 native dense bank differs from the one-example runtime contract")
    token_count = tokens.shape[1]
    object_address_raw = torch.randn(
        1,
        object_count,
        address_width,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    object_address = torch.nn.functional.normalize(
        object_address_raw.float(),
        dim=-1,
    ).to(dtype)
    object_value = torch.randn(
        1,
        object_count,
        value_width,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    object_valid = torch.ones(1, object_count, dtype=torch.bool, device=device)
    # M0 constructs certain synthetic object hypotheses.  A zero log prior is
    # log(1), so it preserves the pre-prior smoke behavior while exercising the
    # exact typed interface used by the released policy path.
    object_log_prior = torch.zeros(
        1,
        object_count,
        device=device,
        dtype=dtype,
    )
    ownership = torch.zeros(
        1,
        token_count,
        object_count + 1,
        device=device,
        dtype=dtype,
    )
    owner_index = torch.arange(token_count, device=device) % object_count
    ownership[0, torch.arange(token_count, device=device), owner_index] = 1.0
    evidence = picf_action_evidence(
        dense_banks=(dense_bank,),
        object_address=object_address,
        object_value=object_value,
        object_valid=object_valid,
        object_log_prior=object_log_prior,
        dense_ownership=(ownership,),
    )
    tensors = {
        "dense_tokens": tokens,
        "dense_valid": valid,
        "dense_ownership": ownership,
        "object_address": object_address,
        "object_value": object_value,
        "object_valid": object_valid,
        "object_log_prior": object_log_prior,
    }
    return evidence, tensors


def main() -> None:
    args = _parse_args()
    _validate_positive_args(args)
    training_recipe_sha256 = _validate_training_recipe(args)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    asset_validation_started = time.perf_counter()
    assets = _validate_immutable_assets(args)
    asset_validation_s = time.perf_counter() - asset_validation_started

    import numpy as np
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        _build_discrete_state_string,
        _build_robot_text,
    )
    from lerobot.utils.constants import ACTION, OBS_STATE
    from PIL import Image
    from transformers import AutoProcessor

    from picf_next.hosts.context import PICFActionEvidence
    from picf_next.hosts.molmoact2 import (
        MolmoAct2PICFActionExpert,
        install_molmoact2_lerobot_picf_adapter,
        prepare_molmoact2_lerobot_observation,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M0 full-weight acceptance requires an explicit CUDA device")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()

    image = Image.open(args.image)
    if image.mode != "RGB":
        raise ValueError(f"M0 requires an original RGB image, got mode={image.mode!r}")
    image.load()
    state = np.zeros(args.state_dim, dtype=np.float32)
    text = _build_robot_text(
        task=args.task,
        discrete_state_string=_build_discrete_state_string(state, 256),
        setup_type=args.setup_type,
        control_mode=args.control_mode,
        add_setup_tokens=True,
        add_control_tokens=True,
        num_images=1,
    )

    timings: dict[str, float] = {"immutable_asset_validation_s": asset_validation_s}
    started = time.perf_counter()
    # MOLMO_CHECKPOINT_REVISION is an exact commit.
    processor = AutoProcessor.from_pretrained(  # nosec B615
        checkpoint_dir,
        revision=MOLMO_CHECKPOINT_REVISION,
        trust_remote_code=True,
        use_fast=False,
    )
    raw_inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    model_inputs = _move_processor_inputs(raw_inputs, device, torch)
    timings["processor_s"] = time.perf_counter() - started

    config = MolmoAct2Config(
        checkpoint_path=str(checkpoint_dir),
        checkpoint_revision=MOLMO_CHECKPOINT_REVISION,
        action_mode="continuous",
        inference_action_mode="continuous",
        setup_type=args.setup_type,
        control_mode=args.control_mode,
        chunk_size=args.action_horizon,
        n_action_steps=args.action_horizon,
        num_inference_steps=args.num_steps,
        mask_action_dim_padding=True,
        enable_inference_cuda_graph=False,
        model_dtype=args.dtype,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,)),
        },
    )
    started = time.perf_counter()
    policy = MolmoAct2Policy(config).to(device).eval()
    torch.cuda.synchronize(device)
    timings["load_policy_s"] = time.perf_counter() - started

    vision_backbone = policy._backbone().vision_backbone
    original_encode_image = vision_backbone.encode_image
    encode_image_calls = 0

    def counted_encode_image(_self: Any, images: Any) -> Any:
        nonlocal encode_image_calls
        encode_image_calls += 1
        return original_encode_image(images)

    vision_backbone.encode_image = types.MethodType(counted_encode_image, vision_backbone)

    def prepare_native_observation() -> Any:
        with _production_inference_context(torch, device, dtype):
            return prepare_molmoact2_lerobot_observation(policy, model_inputs)

    prepared, timings["same_forward_visual_preparation_s"], preparation_memory = _timed_cuda_call(
        torch, device, prepare_native_observation
    )
    vision_backbone.encode_image = original_encode_image
    if encode_image_calls != 1:
        raise RuntimeError(
            f"M0 same-forward visual preparation called the ViT {encode_image_calls} times"
        )
    dense_bank = prepared.vision_patch_bank
    if dense_bank is None:
        raise RuntimeError("M0 processor image produced no native dense patch bank")
    if dense_bank.modality != M0_MODALITY:
        raise RuntimeError("M0 native dense patch modality changed")
    if dense_bank.tokens.shape[1:] != (args.dense_token_count, args.dense_token_width):
        raise RuntimeError(
            "M0 native dense patch shape changed: "
            f"expected {(args.dense_token_count, args.dense_token_width)}, "
            f"got {tuple(dense_bank.tokens.shape[1:])}"
        )

    action_dim_is_pad = torch.ones(1, 32, dtype=torch.bool, device=device)
    action_dim_is_pad[:, : args.action_dim] = False
    policy_batch = {**model_inputs, "action_dim_is_pad": action_dim_is_pad}
    prepared_policy_batch = {
        **prepared.model_inputs,
        "action_dim_is_pad": action_dim_is_pad,
    }

    def run_baseline() -> Any:
        with _production_inference_context(torch, device, dtype):
            return policy.predict_action_chunk(
                policy_batch,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                num_steps=args.num_steps,
            )

    baseline, timings["official_policy_action_s"], baseline_memory = _timed_cuda_call(
        torch, device, run_baseline
    )

    def run_prepared_baseline() -> Any:
        with _production_inference_context(torch, device, dtype):
            return policy.predict_action_chunk(
                prepared_policy_batch,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                num_steps=args.num_steps,
                action_condition_input_ids=prepared.action_condition_input_ids,
            )

    prepared_baseline, timings["prepared_native_action_s"], prepared_memory = _timed_cuda_call(
        torch, device, run_prepared_baseline
    )

    # Both baselines above run before a PICF module is registered. This makes
    # the first comparison an actual official-policy invariant instead of only
    # checking two execution paths through an already modified module tree.
    adapter = MolmoAct2PICFActionExpert(
        policy._action_expert(),
        dense_token_dims={M0_MODALITY: dense_bank.tokens.shape[-1]},
        object_address_dim=args.object_address_width,
        object_value_dim=args.object_value_width,
    )
    install_molmoact2_lerobot_picf_adapter(policy, adapter)
    policy.eval()
    if not torch.equal(adapter.dense_gates, torch.zeros_like(adapter.dense_gates)):
        raise RuntimeError("M0 dense residual gates are not exactly zero")
    if not torch.equal(adapter.object_gates, torch.zeros_like(adapter.object_gates)):
        raise RuntimeError("M0 object residual gates are not exactly zero")
    loaded_memory = _memory_report(torch, device)
    evidence, evidence_tensors = _make_m0_evidence(
        torch=torch,
        picf_action_evidence=PICFActionEvidence,
        dense_bank=dense_bank,
        device=device,
        dtype=dtype,
        seed=args.seed,
        object_count=args.object_count,
        address_width=args.object_address_width,
        value_width=args.object_value_width,
    )

    def run_zero_gate() -> tuple[Any, Any]:
        with _production_inference_context(torch, device, dtype):
            context = adapter.prepare_picf_context(evidence)
            action = policy.predict_action_chunk(
                prepared_policy_batch,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                num_steps=args.num_steps,
                action_layer_context=context,
                action_condition_input_ids=prepared.action_condition_input_ids,
            )
        return action, context

    (zero_gate, context), timings["zero_gate_full_evidence_action_s"], context_memory = (
        _timed_cuda_call(torch, device, run_zero_gate)
    )
    prepared_max_abs_error = float(
        (prepared_baseline.float() - baseline.float()).abs().max().item()
    )
    zero_gate_max_abs_error = float(
        (zero_gate.float() - prepared_baseline.float()).abs().max().item()
    )
    bitwise_equal = bool(
        torch.equal(prepared_baseline, baseline) and torch.equal(zero_gate, prepared_baseline)
    )
    max_abs_error = max(prepared_max_abs_error, zero_gate_max_abs_error)

    weight_hashes = assets["checkpoint"]["weight_shard_sha256"]
    package_names = (
        "torch",
        "torchvision",
        "transformers",
        "huggingface-hub",
        "lerobot",
        "picf-next",
    )
    report = {
        "schema": "picf-next.molmoact2-lerobot-m0.v3",
        "status": "PASS" if bitwise_equal and max_abs_error == 0.0 else "FAIL",
        "gate": "M0_full_weight_parity",
        "training_recipe_sha256": training_recipe_sha256,
        "semantics": {
            "observation_path": "official_molmoact2_lerobot_processor_and_policy",
            "evidence_path": "native_molmo_prepool_patches_plus_synthetic_object_pressure",
            "dense_evidence_is_native_prepool_representation": True,
            "object_evidence_is_synthetic": True,
            "targets_or_masks_in_runtime_input": False,
            "native_molmo_729_same_forward_claimed": True,
            "official_baselines_precede_adapter_registration": True,
        },
        "assets": assets,
        "image": {
            "path": str(args.image.resolve()),
            "sha256": _sha256(args.image),
            "mode": image.mode,
            "size": list(image.size),
        },
        "task": args.task,
        "setup_type": args.setup_type,
        "control_mode": args.control_mode,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "production_autocast_enabled": dtype == torch.bfloat16,
            "packages": {name: importlib.metadata.version(name) for name in package_names},
        },
        "input_tensors": _tensor_manifest(model_inputs),
        "prepared_input_tensors": _tensor_manifest(dict(prepared.model_inputs)),
        "prepared_action_condition_input_ids": _tensor_manifest(
            {"input_ids": prepared.action_condition_input_ids}
        )["input_ids"],
        "action_dim_is_pad_sha256": _tensor_sha256(action_dim_is_pad),
        "evidence_tensors": _tensor_manifest(evidence_tensors),
        "evidence_contract": {
            "modality": M0_MODALITY,
            "dense_token_count": args.dense_token_count,
            "dense_token_width": args.dense_token_width,
            "native_input_embedding_width": int(prepared.model_inputs["inputs_embeds"].shape[-1]),
            "dense_valid_count": int(evidence_tensors["dense_valid"].sum().item()),
            "object_count": args.object_count,
            "object_address_width": args.object_address_width,
            "object_value_width": args.object_value_width,
            "ownership_context_mass": float(
                evidence_tensors["dense_ownership"][..., -1].float().sum().item()
            ),
            "dense_context_layers": len(context.dense_kv_contexts or ()),
            "object_context_layers": len(context.object_kv_contexts or ()),
            "prepared_visual_vision_encoder_calls": encode_image_calls,
        },
        "zero_gate_contract": {
            "dense_gate_count": int(adapter.dense_gates.numel()),
            "object_gate_count": int(adapter.object_gates.numel()),
            "dense_gate_nonzero": int(torch.count_nonzero(adapter.dense_gates).item()),
            "object_gate_nonzero": int(torch.count_nonzero(adapter.object_gates).item()),
            "dense_gate_sha256": _tensor_sha256(adapter.dense_gates),
            "object_gate_sha256": _tensor_sha256(adapter.object_gates),
            "official_action_shape": list(baseline.shape),
            "prepared_action_shape": list(prepared_baseline.shape),
            "zero_gate_action_shape": list(zero_gate.shape),
            "official_action_sha256": _tensor_sha256(baseline),
            "prepared_action_sha256": _tensor_sha256(prepared_baseline),
            "zero_gate_action_sha256": _tensor_sha256(zero_gate),
            "bitwise_equal": bitwise_equal,
            "max_abs_error": max_abs_error,
            "official_vs_prepared_max_abs_error": prepared_max_abs_error,
            "prepared_vs_zero_gate_max_abs_error": zero_gate_max_abs_error,
        },
        "timings_s": timings,
        "cuda_memory_bytes": {
            "after_policy_and_adapter_load": loaded_memory,
            "same_forward_visual_preparation": preparation_memory,
            "official_action": baseline_memory,
            "prepared_native_action": prepared_memory,
            "zero_gate_full_evidence_action": context_memory,
        },
        "checkpoint_weight_shard_sha256": weight_hashes,
        "environment": {
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError(
            "MolmoAct2 official-policy M0 failed fixed-noise zero-gate parity; "
            f"max_abs_error={max_abs_error}"
        )


if __name__ == "__main__":
    main()
