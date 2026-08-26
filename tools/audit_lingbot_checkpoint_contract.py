#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Compare LingBot's public checkpoint index with a meta-device source model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.bootstrap_lingbot_vla2 import LINGBOT_SOURCE_COMMIT, QWEN_PROCESSOR_REVISION
    from tools.smoke_lingbot_vla2_full_weight import (
        _merge_qwen_config,
        _resolve_training_config,
    )
    from tools.verify_lingbot_vla2_patch import detect_patch_state
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import LINGBOT_SOURCE_COMMIT, QWEN_PROCESSOR_REVISION
    from smoke_lingbot_vla2_full_weight import (
        _merge_qwen_config,
        _resolve_training_config,
    )
    from verify_lingbot_vla2_patch import detect_patch_state


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_state_keys(model_keys: set[str], checkpoint_keys: set[str]) -> dict[str, Any]:
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    return {
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "exact_key_match": not missing and not unexpected,
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
    for path in (
        args.source_checkout,
        args.patch,
        args.config,
        args.checkpoint_dir,
        args.processor_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    commit = _git_output(args.source_checkout, "rev-parse", "HEAD")
    if commit != LINGBOT_SOURCE_COMMIT:
        raise RuntimeError("LingBot checkout differs from the pinned commit")
    if detect_patch_state(args.source_checkout, args.patch) != "applied":
        raise RuntimeError("checkpoint contract requires the exact applied host patch")
    index_path = args.checkpoint_dir / "model.safetensors.index.json"
    checkpoint_config_path = args.checkpoint_dir / "config.json"
    if not index_path.is_file() or not checkpoint_config_path.is_file():
        raise FileNotFoundError("checkpoint index/config is incomplete")

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))
    import torch
    from lingbotvla.models.module_utils import init_empty_weights
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
    from transformers.modeling_utils import no_init_weights

    try:
        from tools.lingbot_vla2_runtime_helpers import load_lingbot_training_config
    except ModuleNotFoundError:
        from lingbot_vla2_runtime_helpers import load_lingbot_training_config

    started = time.perf_counter()
    training = load_lingbot_training_config(args.config)
    merged, _ = _resolve_training_config(
        training,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        num_steps=2,
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
    config.attention_implementation = "eager"
    config.vit_attn_implementation = "eager"
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    with init_empty_weights(), no_init_weights():
        policy = LingbotVlaV2Policy(config=config, eval=True)
    state = policy.state_dict()
    elapsed = time.perf_counter() - started

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index has no weight_map")
    comparison = compare_state_keys(set(state), set(weight_map))
    checkpoint_config = json.loads(checkpoint_config_path.read_text())
    parameters = dict(policy.named_parameters())
    buffers = dict(policy.named_buffers())
    model_numel = sum(parameter.numel() for parameter in parameters.values())
    report = {
        "schema": "picf-next.lingbot-checkpoint-contract.v1",
        "status": "PASS" if comparison["exact_key_match"] else "FAIL",
        "source_commit": commit,
        "source_patch_sha256": _sha256(args.patch),
        "source_diff_sha256": hashlib.sha256(
            _git_output(args.source_checkout, "diff", "--binary").encode()
        ).hexdigest(),
        "training_config": str(args.config.resolve()),
        "training_config_sha256": _sha256(args.config),
        "processor_config_sha256": _sha256(args.processor_dir / "config.json"),
        "checkpoint_index_sha256": _sha256(index_path),
        "checkpoint_config_sha256": _sha256(checkpoint_config_path),
        "checkpoint_config_keys": sorted(checkpoint_config),
        "checkpoint_total_size": int(index.get("metadata", {}).get("total_size", 0)),
        "comparison": comparison,
        "model_parameter_tensors": len(parameters),
        "model_buffer_tensors": len(buffers),
        "model_parameter_count": model_numel,
        "model_layers": int(config.num_hidden_layers),
        "action_expert_layers": len(policy.model.qwenvl_with_expert.qwen_expert.model.layers),
        "action_experts_per_layer": int(config.token_num_experts),
        "action_top_k": int(config.token_top_k),
        "build_elapsed_s": elapsed,
        "device": str(next(iter(state.values())).device),
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not comparison["exact_key_match"]:
        raise RuntimeError("LingBot source/YAML state keys differ from checkpoint index")


if __name__ == "__main__":
    main()
