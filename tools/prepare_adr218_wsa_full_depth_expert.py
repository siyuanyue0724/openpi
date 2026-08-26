#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from picf_next.lingbot_native.wsa_full_depth_adaptation import (
    WSA_COMMIT,
    WSA_PREPROCESS_SHA256,
    adapt_wsa_future_state_dict,
)

CHECKPOINT_PREFIX = "model.mot.mixtures.future_3d."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_wsa_namespace(source_root: Path) -> None:
    src = source_root / "src"
    package_paths = (
        ("lerobot", src / "lerobot"),
        ("lerobot.policies", src / "lerobot/policies"),
        ("lerobot.policies.WSA_Large", src / "lerobot/policies/WSA_Large"),
        ("lerobot.policies.WSA_Large.core", src / "lerobot/policies/WSA_Large/core"),
        ("lerobot.policies.WSA_Large.core.models", src / "lerobot/policies/WSA_Large/core/models"),
        (
            "lerobot.policies.WSA_Large.core.models.wan22",
            src / "lerobot/policies/WSA_Large/core/models/wan22",
        ),
    )
    for name, path in package_paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Pinned WSA package path is missing: {path}")
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _target_shapes(source_root: Path) -> dict[str, tuple[int, ...]]:
    _install_wsa_namespace(source_root)
    from lerobot.policies.WSA_Large.core.models.wan22.future_3d_expert import (  # noqa: PLC0415
        Future3DExpert,
    )

    expert = Future3DExpert(
        hidden_dim=768,
        ffn_dim=3072,
        num_heads=32,
        attn_head_dim=128,
        num_layers=36,
        num_query_tokens=432,
        da3_num_views=2,
        da3_tokens_per_view=1296,
        da3_query_dim=2048,
        query_layer_indices=(17, 23, 29, 35),
        query_mode="slot_noise",
        query_noise_scale=0.5,
        query_noise_min_sigma=0.0,
        query_noise_max_sigma=0.5,
        query_sigma_source="constant",
        slot_pos_scale=0.5,
    )
    shapes = {key: tuple(value.shape) for key, value in expert.state_dict().items()}
    del expert
    return shapes


def _load_future_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        keys = tuple(source.keys())
        future_keys = tuple(key for key in keys if key.startswith(CHECKPOINT_PREFIX))
        if len(future_keys) != 481:
            raise ValueError(f"Expected 481 released Future3D tensors, got {len(future_keys)}")
        for checkpoint_key in future_keys:
            local_key = checkpoint_key.removeprefix(CHECKPOINT_PREFIX)
            if local_key in state:
                raise ValueError(f"Duplicate Future3D key after prefix removal: {local_key}")
            state[local_key] = source.get_tensor(checkpoint_key)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsa-source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source_state = _load_future_state(args.checkpoint)
    target_shapes = _target_shapes(args.wsa_source_root)
    adapted, receipt = adapt_wsa_future_state_dict(source_state, target_shapes)
    if set(adapted) != set(target_shapes):
        raise RuntimeError("Adapted Future3D state does not exactly cover the target constructor")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        adapted,
        str(args.output),
        metadata={
            "schema": "picf-next.adr218-wsa-full-depth-expert.v1",
            "wsa_commit": WSA_COMMIT,
            "wsa_preprocess_sha256": WSA_PREPROCESS_SHA256,
            "depth_adaptation": "nearest_percent_aligned_30_to_36",
            "attention_adaptation": "official_width_resize_24_to_32",
        },
    )
    payload = {
        "schema": "picf-next.adr218-wsa-full-depth-expert-receipt.v1",
        "wsa_commit": WSA_COMMIT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "source_tensor_count": receipt.source_tensor_count,
        "target_tensor_count": receipt.target_tensor_count,
        "copied_tensor_count": receipt.copied_tensor_count,
        "resized_tensor_count": receipt.resized_tensor_count,
        "duplicated_source_tensor_count": receipt.duplicated_source_tensor_count,
        "unused_source_keys": list(receipt.unused_source_keys),
        "target_state_element_count": sum(tensor.numel() for tensor in adapted.values()),
        "target_query_layers": [17, 23, 29, 35],
        "source_depth": 30,
        "target_depth": 36,
        "source_heads": 24,
        "target_heads": 32,
        "future_slots": 432,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
