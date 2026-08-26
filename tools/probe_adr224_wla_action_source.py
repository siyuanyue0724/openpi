from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from picf_next.wla_upstream import load_wla_action_symbols


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instantiate and differentiate the exact pinned WLA action expert."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum().cpu()
    return float(squared.sqrt().item())


def main() -> None:
    args = _parse_args()
    symbols = load_wla_action_symbols(args.source_root)
    source_config = yaml.safe_load(
        (args.source_root / "configs/libero_all_image_action.yaml").read_text()
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
    )
    missing = [name for name in required if name not in source_config]
    if missing:
        raise ValueError(f"pinned WLA LIBERO config is missing {missing}")
    config = SimpleNamespace(**{name: source_config[name] for name in required})

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(224)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(224)
        torch.cuda.reset_peak_memory_stats()

    model = symbols.action_head(config).to(device=device, dtype=torch.bfloat16)
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    layer_count = len(model.model.transformer_blocks)
    if layer_count != 28:
        raise RuntimeError(f"WLA action expert lost its 28-layer topology: {layer_count}")

    layerwise = [
        torch.randn(
            1,
            64,
            2048,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(layer_count)
    ]
    actions = torch.randn(
        1,
        config.chunk_size,
        config.max_action_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    action_mask = torch.ones_like(actions)
    state = torch.randn(
        1,
        1,
        config.max_state_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    loss = model(layerwise, actions, action_mask, state)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("exact WLA action expert produced a non-finite scalar loss")
    loss.backward()

    query_gradient_norms = [
        0.0 if value.grad is None else float(value.grad.float().norm().item())
        for value in layerwise
    ]
    used_condition_layers = [
        index for index, norm in enumerate(query_gradient_norms) if norm > 0.0
    ]
    expected_condition_layers = list(range(0, layer_count, 2))
    if used_condition_layers != expected_condition_layers:
        raise RuntimeError(
            "WLA layerwise condition use differs from the published alternating "
            f"cross/self-attention schedule: {used_condition_layers}"
        )
    query_gradient_norm = float(sum(norm * norm for norm in query_gradient_norms) ** 0.5)
    model_gradient_norm = _gradient_norm(parameters)
    if query_gradient_norm <= 0.0 or model_gradient_norm <= 0.0:
        raise RuntimeError("exact WLA action path lost a required gradient")

    receipt = {
        "schema": "picf.adr224.wla-action-source-probe.v1",
        "source_commit": symbols.source.commit,
        "critical_files": dict(symbols.source.files),
        "device": str(device),
        "dtype": "bfloat16",
        "parameter_count": parameter_count,
        "layer_count": layer_count,
        "layerwise_condition_shapes": [list(value.shape) for value in layerwise],
        "action_shape": list(actions.shape),
        "state_shape": list(state.shape),
        "loss": float(loss.detach().float().item()),
        "query_gradient_norm": query_gradient_norm,
        "query_gradient_norms_by_layer": query_gradient_norms,
        "used_condition_layers": used_condition_layers,
        "model_gradient_norm": model_gradient_norm,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
