from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from picf_next.lingbot_wla_world import LingBotWLAWorldExpert


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grad_norm(parameters) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(224)
    world_device = torch.device("cuda:2")
    vae_device = torch.device("cuda:3")
    for device in (world_device, vae_device):
        # PyTorch 2.8 requires the device context to exist before its peak
        # allocator counters can be reset.
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    started = time.monotonic()
    expert = LingBotWLAWorldExpert.from_pinned_source(
        args.source_root,
        args.pretrained_root,
        host_width=2560,
        world_device=world_device,
        vae_device=vae_device,
    )
    load_seconds = time.monotonic() - started
    expert.train()

    visual = torch.randn(
        1,
        256,
        2560,
        device=world_device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    queries = [
        torch.randn(
            1,
            64,
            2560,
            device=world_device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(28)
    ]
    target = torch.rand(1, 3, 512, 512, device=vae_device, dtype=torch.float32) * 2.0 - 1.0

    step_started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = expert(
            target_images=target,
            current_visual_embeddings=visual,
            current_visual_valid=torch.ones(1, 256, dtype=torch.bool, device=world_device),
            layerwise_query_states=queries,
        )
        output.loss.backward()
    step_seconds = time.monotonic() - step_started

    receipt = {
        "schema": "picf-next.adr224-wla-world-source-probe.v1",
        "passed": bool(
            torch.isfinite(output.loss)
            and visual.grad is not None
            and all(query.grad is not None for query in queries)
            and _grad_norm(expert.world_expert.parameters()) > 0.0
            and _grad_norm(expert.connector.parameters()) > 0.0
        ),
        "source_commit": expert.source.commit,
        "source_files": dict(expert.source.files),
        "pretrained_sha256": {
            "transformer": _sha256(
                args.pretrained_root / "transformer/diffusion_pytorch_model.safetensors"
            ),
            "vae": _sha256(args.pretrained_root / "vae/diffusion_pytorch_model.safetensors"),
            "scheduler": _sha256(args.pretrained_root / "scheduler/scheduler_config.json"),
        },
        "host_width_adaptation": 2560,
        "world_layers": len(expert.world_expert.transformer_blocks),
        "condition_shapes": [list(value.shape) for value in output.condition.layerwise_embeddings],
        "condition_mask_shape": list(output.condition.attention_mask.shape),
        "prediction_shape": list(output.prediction.shape),
        "target_velocity_shape": list(output.target_velocity.shape),
        "loss": float(output.loss.detach()),
        "visual_gradient_norm": float(visual.grad.detach().float().norm()),
        "query_gradient_norms": [float(query.grad.detach().float().norm()) for query in queries],
        "connector_gradient_norm": _grad_norm(expert.connector.parameters()),
        "world_gradient_norm": _grad_norm(expert.world_expert.parameters()),
        "vae_trainable_parameters": sum(
            parameter.numel() for parameter in expert.vae.parameters() if parameter.requires_grad
        ),
        "parameter_counts": {
            "world": sum(parameter.numel() for parameter in expert.world_expert.parameters()),
            "connector": sum(parameter.numel() for parameter in expert.connector.parameters()),
            "vae": sum(parameter.numel() for parameter in expert.vae.parameters()),
        },
        "load_seconds": load_seconds,
        "forward_backward_seconds": step_seconds,
        "peak_memory_bytes": {
            "world_device": torch.cuda.max_memory_allocated(world_device),
            "vae_device": torch.cuda.max_memory_allocated(vae_device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
