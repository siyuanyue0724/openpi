#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the exact WLA action source on the released LingBot 36-layer host."
    )
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--wla-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-device", default="cuda:0")
    parser.add_argument("--action-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=224)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gradient_norm(parameters) -> float:
    import torch

    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum().cpu()
    return float(squared.sqrt().item())


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    for path in (
        args.source_checkout,
        args.config,
        args.checkpoint_dir,
        args.processor_dir,
        args.wla_source_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import torch
    import yaml
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
    from transformers import AutoConfig, AutoProcessor
    from transformers.modeling_utils import no_init_weights

    from picf_next.lingbot_wla_shared import LingBotWLASharedInterface
    from picf_next.wla_upstream import load_wla_action_symbols
    from tools.bootstrap_lingbot_vla2 import QWEN_PROCESSOR_REVISION
    from tools.lingbot_vla2_runtime_helpers import (
        _merge_qwen_config,
        _resolve_training_config,
        load_lingbot_training_config,
        select_lingbot_deterministic_moe_backend,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("ADR224 real-host probe requires two visible CUDA devices")
    host_device = torch.device(args.host_device)
    action_device = torch.device(args.action_device)
    if host_device.type != "cuda" or action_device.type != "cuda" or host_device == action_device:
        raise ValueError("host and WLA action expert require distinct CUDA devices")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(host_device)
    torch.cuda.reset_peak_memory_stats(action_device)

    select_lingbot_deterministic_moe_backend(
        action_expert_module=qwen2_action_expert,
        fused_moe_module=fused_moe,
    )
    training = load_lingbot_training_config(args.config)
    merged, _data_mapping = _resolve_training_config(
        training,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        num_steps=2,
    )
    config = LingbotVLAV2Config(**merged)
    for key, value in merged.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    qwen_config = AutoConfig.from_pretrained(
        args.processor_dir,
        revision=QWEN_PROCESSOR_REVISION,
        local_files_only=True,
    )
    _merge_qwen_config(config, qwen_config)
    config.tokenizer_path = str(args.processor_dir.resolve())
    config.use_cache = False
    config.use_compile = False
    config.num_steps = 2
    config.attention_implementation = "eager"
    config.vit_attn_implementation = "eager"

    timings: dict[str, float] = {}
    started = time.perf_counter()
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    with init_empty_weights(), no_init_weights():
        policy = LingbotVlaV2Policy(config=config, eval=False).to(dtype)
    load_model_weights(
        policy,
        str(args.checkpoint_dir.resolve()),
        str(host_device),
        post_training=True,
        adanorm_time=bool(config.adanorm_time),
    )
    policy.eval()
    timings["lingbot_load_s"] = time.perf_counter() - started
    joint = policy.model.qwenvl_with_expert
    released_action_parameter_count = sum(
        parameter.numel() for parameter in joint.qwen_expert.parameters()
    )
    # ADR224 replaces, rather than co-executes, the released LingBot action
    # expert. Keep it intact on CPU for provenance while freeing host GPU space.
    joint.qwen_expert.to("cpu")
    torch.cuda.empty_cache()

    symbols = load_wla_action_symbols(args.wla_source_root)
    source_config = yaml.safe_load(
        (symbols.source.root / "configs/libero_all_image_action.yaml").read_text()
    )
    required = (
        "diffusion_model_cfg",
        "max_action_dim",
        "max_state_dim",
        "chunk_size",
        "num_inference_timesteps",
        "add_pos_embed",
        "max_seq_len",
        "noise_beta_alpha",
        "noise_beta_beta",
        "num_timestep_buckets",
        "noise_s",
        "repeated_diffusion_steps",
    )
    selected = {name: source_config[name] for name in required}
    source_cross_attention_dim = int(selected["diffusion_model_cfg"]["cross_attention_dim"])
    host_width = int(joint.qwenvl.config.text_config.hidden_size)
    selected["diffusion_model_cfg"] = dict(selected["diffusion_model_cfg"])
    selected["diffusion_model_cfg"]["cross_attention_dim"] = host_width
    wla_config = SimpleNamespace(**selected)
    started = time.perf_counter()
    action_head = symbols.action_head(wla_config).to(device=action_device, dtype=dtype)
    interface = LingBotWLASharedInterface(
        action_head=action_head,
        source=symbols.source,
        repeated_diffusion_steps=wla_config.repeated_diffusion_steps,
        host_width=host_width,
        device=host_device,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise RuntimeError(f"WLA source newline is not one LingBot token: {newline_ids}")
    im_end_token_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    interface.initialize_meta_tokens_from_lingbot(
        joint,
        newline_token_id=int(newline_ids[0]),
        im_end_token_id=im_end_token_id,
    )
    interface.eval()
    timings["wla_action_and_query_init_s"] = time.perf_counter() - started

    eos = int(joint.qwenvl.config.text_config.eos_token_id)
    token_ids = torch.full((1, 12), eos, dtype=torch.long, device=host_device)
    alternate_ids = token_ids.clone()
    alternate_ids[:, 3] = max(0, eos - 1)
    valid = torch.ones_like(token_ids, dtype=torch.bool)

    def encode(ids: torch.Tensor):
        prefix = joint.embed_language_tokens(ids).to(dtype=dtype)
        positions = joint.build_prefix_position_ids(
            ids,
            valid.long(),
            image_grid_thw=None,
            video_grid_thw=None,
        )
        attention = valid[:, :, None] & valid[:, None, :]
        return interface.encode_host(
            joint,
            prefix_embeds=prefix,
            attention_mask=attention,
            position_ids=positions,
            visual_pos_masks=torch.zeros_like(valid),
            deepstack_visual_embeds=None,
        )

    started = time.perf_counter()
    with torch.inference_mode():
        host = encode(token_ids)
        alternate_host = encode(alternate_ids)
    torch.cuda.synchronize(host_device)
    timings["two_lingbot_host_forwards_s"] = time.perf_counter() - started
    query_delta = float(
        (
            host.layerwise_query_states[-1].float()
            - alternate_host.layerwise_query_states[-1].float()
        )
        .abs()
        .mean()
        .item()
    )
    if query_delta <= 0.0:
        raise RuntimeError("WLA query states are invariant to a LingBot prompt intervention")

    repeated = wla_config.repeated_diffusion_steps
    action_queries = [
        value.to(action_device).repeat(repeated, 1, 1).requires_grad_(True)
        for value in host.layerwise_query_states
    ]
    actions = torch.randn(
        1,
        wla_config.chunk_size,
        wla_config.max_action_dim,
        device=action_device,
        dtype=dtype,
    ).repeat(repeated, 1, 1)
    action_mask = torch.ones_like(actions)
    state = torch.randn(
        1,
        1,
        wla_config.max_state_dim,
        device=action_device,
        dtype=dtype,
    ).repeat(repeated, 1, 1)
    started = time.perf_counter()
    loss = action_head(action_queries, actions, action_mask, state)
    loss.backward()
    torch.cuda.synchronize(action_device)
    timings["wla_action_forward_backward_s"] = time.perf_counter() - started
    query_gradients = [
        0.0 if value.grad is None else float(value.grad.float().norm().item())
        for value in action_queries
    ]
    used_layers = [index for index, value in enumerate(query_gradients) if value > 0.0]
    if used_layers != list(range(0, 28, 2)):
        raise RuntimeError(f"WLA source changed its alternating conditioning layers: {used_layers}")
    action_gradient_norm = _gradient_norm(action_head.parameters())
    if action_gradient_norm <= 0.0:
        raise RuntimeError("WLA action expert lost its source gradient")

    modeling_source = (
        args.source_checkout
        / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    )
    receipt = {
        "schema": "picf.adr224.lingbot-wla-real-host-probe.v1",
        "passed": True,
        "lingbot_source_head": (
            __import__("subprocess")
            .check_output(
                ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        ),
        "lingbot_modeling_sha256": _sha256(modeling_source),
        "wla_source_commit": symbols.source.commit,
        "wla_critical_files": dict(symbols.source.files),
        "released_action_parameter_count_moved_to_cpu": released_action_parameter_count,
        "wla_action_parameter_count": sum(p.numel() for p in action_head.parameters()),
        "source_cross_attention_dim": source_cross_attention_dim,
        "integrated_cross_attention_dim": host_width,
        "cross_attention_width_adaptation": "source config field only; source code and 28-layer topology unchanged",
        "host_layer_count": len(joint.qwenvl.model.language_model.layers),
        "selected_query_layer_count": len(host.layerwise_query_states),
        "query_shapes": [list(value.shape) for value in host.layerwise_query_states],
        "prompt_intervention_query_mae": query_delta,
        "action_loss": float(loss.detach().float().item()),
        "action_query_gradient_norms": query_gradients,
        "action_used_condition_layers": used_layers,
        "action_parameter_gradient_norm": action_gradient_norm,
        "peak_host_cuda_bytes": int(torch.cuda.max_memory_allocated(host_device)),
        "peak_action_cuda_bytes": int(torch.cuda.max_memory_allocated(action_device)),
        "timings_s": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
