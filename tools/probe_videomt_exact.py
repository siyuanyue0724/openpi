#!/usr/bin/env python3
"""Run strict-load, output-shape, and temporal continuation probes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from picf_next.videomt_exact.runtime import (
    ExactVidEoMTConfig,
    load_exact_videomt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--seed", type=int, default=198)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("probe dimensions must be divisible by 16")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if args.device == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32

    started = time.perf_counter()
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.checkpoint.resolve(),
            local_dinov3_bundle=args.dinov3_bundle.resolve(),
            num_frames=2,
        ),
        device=args.device,
        dtype=dtype,
    )
    load_seconds = time.perf_counter() - started
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(2, 3, args.height, args.width, generator=generator).to(
        device=args.device,
        dtype=dtype,
    )

    with torch.inference_mode():
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        together = runtime(inputs, resume=False)
        if args.device == "cuda":
            torch.cuda.synchronize()
        together_seconds = time.perf_counter() - started

        first = runtime(inputs[:1], resume=False)
        second = runtime(inputs[1:], resume=True)

    together_class = together.class_logits[:, 1].float()
    resumed_class = second.class_logits[:, 0].float()
    together_mask = together.mask_logits[:, :, 1].float()
    resumed_mask = second.mask_logits[:, :, 0].float()
    together_embedding = together.query_embeddings[:, 1].float()
    resumed_embedding = second.query_embeddings[:, 0].float()
    together_query = together.propagated_queries.float()
    resumed_query = second.propagated_queries.float()

    def difference_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
        difference = (left - right).abs()
        scale = torch.maximum(left.abs().max(), right.abs().max()).clamp_min(1e-12)
        return {
            "max_abs": difference.max().item(),
            "mean_abs": difference.mean().item(),
            "max_abs_over_global_scale": (difference.max() / scale).item(),
        }

    class_stats = difference_stats(together_class, resumed_class)
    mask_stats = difference_stats(together_mask, resumed_mask)
    embedding_stats = difference_stats(together_embedding, resumed_embedding)
    query_stats = difference_stats(together_query, resumed_query)
    class_probability_delta = (
        together_class.softmax(dim=-1) - resumed_class.softmax(dim=-1)
    ).abs().max().item()
    mask_probability_delta = (
        together_mask.sigmoid() - resumed_mask.sigmoid()
    ).abs().max().item()
    query_cosine = torch.nn.functional.cosine_similarity(
        together_query,
        resumed_query,
        dim=-1,
    )
    embedding_cosine = torch.nn.functional.cosine_similarity(
        together_embedding,
        resumed_embedding,
        dim=-1,
    )
    receipt = {
        "load_seconds": load_seconds,
        "two_frame_seconds": together_seconds,
        "device": str(args.device),
        "dtype": str(dtype),
        "parameter_count": sum(parameter.numel() for parameter in runtime.parameters()),
        "class_shape": list(together.class_logits.shape),
        "mask_shape": list(together.mask_logits.shape),
        "query_embedding_shape": list(together.query_embeddings.shape),
        "query_shape": list(together.propagated_queries.shape),
        "auxiliary_outputs": len(together.auxiliary_outputs),
        "continuation": {
            "class_logits": class_stats,
            "mask_logits": mask_stats,
            "query_embedding": embedding_stats,
            "propagated_query": query_stats,
            "class_probability_max_abs": class_probability_delta,
            "mask_probability_max_abs": mask_probability_delta,
            "query_cosine_min": query_cosine.min().item(),
            "query_cosine_mean": query_cosine.mean().item(),
            "query_embedding_cosine_min": embedding_cosine.min().item(),
            "query_embedding_cosine_mean": embedding_cosine.mean().item(),
        },
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
        "finite": bool(
            torch.isfinite(together.class_logits).all()
            and torch.isfinite(together.mask_logits).all()
            and torch.isfinite(together.query_embeddings).all()
            and torch.isfinite(together.propagated_queries).all()
        ),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if dtype == torch.float32:
        accepted = (
            class_stats["max_abs"] <= 2e-5
            and embedding_stats["max_abs"] <= 2e-5
            and query_stats["max_abs"] <= 2e-5
            and mask_probability_delta <= 2e-5
            and query_cosine.min().item() >= 1.0 - 2e-6
            and embedding_cosine.min().item() >= 1.0 - 2e-6
        )
    else:
        accepted = (
            class_probability_delta <= 2e-2
            and mask_probability_delta <= 2e-2
            and query_cosine.min().item() >= 0.999
            and embedding_cosine.min().item() >= 0.999
        )
    if not accepted:
        raise RuntimeError("combined and resumed temporal execution are not functionally equivalent")


if __name__ == "__main__":
    main()
