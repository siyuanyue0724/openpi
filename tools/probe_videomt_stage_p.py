#!/usr/bin/env python3
"""Probe real VidEoMT queries at the value-preserving LingBot Stage-PQ boundary."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
)
from picf_next.videomt_exact.preprocessing import prepare_rgb_frames
from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt
from picf_next.videomt_exact.stage_p import VidEoMTStageP, with_videomt_query_modality_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--golden-panel", action="store_true")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def read_rgb(path: Path, *, golden_panel: bool) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if golden_panel:
            if rgb.width % 4 or rgb.height <= 96:
                raise ValueError("golden panel must contain four equal columns and a 96px header")
            rgb = rgb.crop((0, 96, rgb.width // 4, rgb.height))
        return np.asarray(rgb, dtype=np.uint8).copy()


def controls(device: torch.device, dtype: torch.dtype) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.zeros(1, 1, 7, device=device, dtype=dtype),
        field_valid=torch.ones(1, 1, 7, device=device, dtype=torch.bool),
        token_valid=torch.ones(1, 1, device=device, dtype=torch.bool),
        delta_time=torch.ones(1, 1, device=device, dtype=dtype),
        reset=torch.zeros(1, 1, device=device, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, device=device, dtype=torch.bool),
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if device.type == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32
    prepared = prepare_rgb_frames((read_rgb(args.input, golden_panel=args.golden_panel),))
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.checkpoint.resolve(),
            local_dinov3_bundle=args.dinov3_bundle.resolve(),
            num_frames=1,
        ),
        device=device,
        dtype=dtype,
    )
    stage_p = VidEoMTStageP(runtime)
    model_input = prepared.model_input.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = stage_p(model_input)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started

    query = result.observation.query_tokens.float()
    normalized = torch.nn.functional.normalize(query, dim=-1)
    cosine = normalized @ normalized.transpose(1, 2)
    off_diagonal = cosine[:, ~torch.eye(200, dtype=torch.bool, device=device)]
    singular_values = torch.linalg.svdvals(query[0] - query[0].mean(dim=0, keepdim=True))
    energy = singular_values.square()
    distribution = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).tiny)
    effective_rank = float(torch.exp(-(distribution * distribution.clamp_min(1e-30).log()).sum()))

    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=8,
            host_width=1024,
            executed_action_dim=7,
            num_layers=2,
            modality_specs=with_videomt_query_modality_spec(()),
        ),
        device=device,
        dtype=dtype,
    )
    context = LingBotNativeContext(
        controls=controls(device, dtype),
        modalities=result.modalities,
    )
    with torch.inference_mode():
        projected, valid, _direct_action, relations = graph._project_modalities(
            context,
            prefix=torch.zeros(1, 1, 1024, device=device, dtype=dtype),
        )
    projection = graph.modality_projections["videomt_queries"].weight.detach().float()
    gram_error = float(
        (projection.T @ projection - torch.eye(1024, device=device)).abs().max()
    )
    masks = result.observation.mask_logits.float().sigmoid()
    object_probability = result.observation.object_probability.float()
    report = {
        "schema": "picf-next.videomt-stage-p-probe.v1",
        "interface_identity": result.interface_identity,
        "claim_scope": "real upstream query-to-LingBot boundary; not CALVIN learning quality",
        "input": str(args.input.resolve()),
        "golden_panel_source_crop": args.golden_panel,
        "original_size": list(prepared.original_sizes[0]),
        "model_input_shape": list(model_input.shape),
        "device": str(device),
        "dtype": str(dtype),
        "forward_seconds": forward_seconds,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "query_shape": list(query.shape),
        "mask_shape": list(masks.shape),
        "all_200_query_slots_valid": bool(result.observation.query_valid.all()),
        "native_stream_aliases_query_storage": (
            result.modalities.streams[0].tokens.data_ptr()
            == result.observation.query_tokens.data_ptr()
        ),
        "query_statistics": {
            "centered_matrix_rank": int(torch.linalg.matrix_rank(query[0] - query[0].mean(0))),
            "effective_rank": effective_rank,
            "off_diagonal_cosine_mean": float(off_diagonal.mean()),
            "off_diagonal_cosine_p95": float(torch.quantile(off_diagonal, 0.95)),
        },
        "object_probability": {
            "minimum": float(object_probability.min()),
            "mean": float(object_probability.mean()),
            "maximum": float(object_probability.max()),
        },
        "mask_probability": {
            "minimum": float(masks.min()),
            "mean": float(masks.mean()),
            "maximum": float(masks.max()),
            "mean_query_variance_per_pixel": float(masks.var(dim=1).mean()),
        },
        "lingbot_boundary": {
            "projected_shape": list(projected.shape),
            "all_valid": bool(valid.all()),
            "relation_surface_count": len(relations),
            "projection_gram_max_abs_error": gram_error,
            "finite": bool(torch.isfinite(projected).all()),
        },
    }
    if not (
        report["all_200_query_slots_valid"]
        and report["native_stream_aliases_query_storage"]
        and report["lingbot_boundary"]["all_valid"]
        and report["lingbot_boundary"]["finite"]
        and math.isclose(gram_error, 0.0, abs_tol=0.0)
    ):
        raise RuntimeError("Stage-P exact-token boundary failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
