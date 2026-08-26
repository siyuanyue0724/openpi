#!/usr/bin/env python3
"""Run the complete released VidEoMT graph on one exact causal CALVIN clip.

This is an execution receipt, not a reduced smoke model. It uses the same
content-addressed CALVIN index, five-frame causal adapter, released FP32 donor,
and BF16 LingBot boundary as the Stage-PQ training runner.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.videomt_exact.calvin_stage_p import (
    VIDEOMT_CAUSAL_FRAME_COUNT,
    make_videomt_stage_pq_execution_receipt,
    prepare_calvin_stage_pq_c5,
)
from picf_next.videomt_exact.checkpoint import inspect_published_checkpoint
from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt
from picf_next.videomt_exact.stage_p import VidEoMTStageP

STREAMING_MAX_SCALE_NORMALIZED_ERROR = 1e-4
STREAMING_MAX_CLASS_PROBABILITY_ERROR = 1e-4
STREAMING_MAX_MASK_PROBABILITY_ERROR = 1e-3
STREAMING_MIN_LAST_AXIS_COSINE = 0.99999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-index", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _tensor_receipt(value: torch.Tensor) -> dict[str, object]:
    floating = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()),
        "minimum": float(floating.min()),
        "mean": float(floating.mean()),
        "maximum": float(floating.max()),
    }


def _difference_receipt(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_fp32 = left.detach().float()
    right_fp32 = right.detach().float()
    difference = (left_fp32 - right_fp32).abs()
    scale = torch.maximum(left_fp32.abs().max(), right_fp32.abs().max()).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        left_fp32.reshape(-1, left_fp32.shape[-1]),
        right_fp32.reshape(-1, right_fp32.shape[-1]),
        dim=-1,
        eps=1e-12,
    )
    return {
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "maximum_absolute_error_over_global_scale": float(difference.max() / scale),
        "minimum_last_axis_cosine": float(cosine.min()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the full FP32 Stage-PQ receipt requires a CUDA device")

    split = args.dataset_split.expanduser().resolve()
    manifest_path = args.dataset_manifest.expanduser().resolve()
    manifest = load_dataset_file_manifest(manifest_path)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    prepared = prepare_calvin_stage_pq_c5(index, args.source_index)
    source_files = []
    for source_index in prepared.source_global_indices:
        relative = f"episode_{source_index:07d}.npz"
        source_files.append(manifest.record_for(relative).to_dict())

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_receipt = inspect_published_checkpoint(checkpoint)
    started_load = time.perf_counter()
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=checkpoint,
            local_dinov3_bundle=args.dinov3_bundle.expanduser().resolve(),
            num_frames=VIDEOMT_CAUSAL_FRAME_COUNT,
        ),
        device=device,
        dtype=torch.float32,
    )
    runtime.requires_grad_(False)
    runtime.eval()
    stage_p = VidEoMTStageP(runtime).eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started_load

    parameters = tuple(stage_p.parameters())
    if not parameters or any(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("the complete donor is not frozen")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("the complete donor is not FP32")
    if stage_p.training or runtime.model.training:
        raise RuntimeError("the complete donor is not in released evaluation mode")

    model_input = prepared.frames.model_input.to(device=device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started_forward = time.perf_counter()
    with torch.no_grad():
        result = stage_p(
            model_input,
            host_dtype=torch.bfloat16,
            resume=False,
        )
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started_forward

    started_streaming = time.perf_counter()
    streamed_results = []
    with torch.no_grad():
        for frame_index in range(VIDEOMT_CAUSAL_FRAME_COUNT):
            streamed_results.append(
                stage_p(
                    model_input[frame_index : frame_index + 1],
                    host_dtype=torch.bfloat16,
                    resume=frame_index > 0,
                )
            )
    torch.cuda.synchronize(device)
    streaming_forward_seconds = time.perf_counter() - started_streaming

    streamed_class = torch.cat(
        tuple(item.upstream.class_logits for item in streamed_results),
        dim=1,
    )
    streamed_masks = torch.cat(
        tuple(item.upstream.mask_logits for item in streamed_results),
        dim=2,
    )
    streamed_queries = torch.cat(
        tuple(item.upstream.query_embeddings for item in streamed_results),
        dim=1,
    )
    streamed_state = streamed_results[-1].upstream.propagated_queries
    parity_class = _difference_receipt(result.upstream.class_logits, streamed_class)
    parity_masks = _difference_receipt(result.upstream.mask_logits, streamed_masks)
    parity_queries = _difference_receipt(
        result.upstream.query_embeddings,
        streamed_queries,
    )
    parity_state = _difference_receipt(
        result.upstream.propagated_queries,
        streamed_state,
    )
    class_probability_error = float(
        (result.upstream.class_logits.softmax(dim=-1) - streamed_class.softmax(dim=-1)).abs().max()
    )
    mask_probability_error = float(
        (result.upstream.mask_logits.sigmoid() - streamed_masks.sigmoid()).abs().max()
    )
    parity_passed = bool(
        parity_class["maximum_absolute_error_over_global_scale"]
        <= STREAMING_MAX_SCALE_NORMALIZED_ERROR
        and parity_masks["maximum_absolute_error_over_global_scale"]
        <= STREAMING_MAX_SCALE_NORMALIZED_ERROR
        and parity_queries["maximum_absolute_error_over_global_scale"]
        <= STREAMING_MAX_SCALE_NORMALIZED_ERROR
        and parity_state["maximum_absolute_error_over_global_scale"]
        <= STREAMING_MAX_SCALE_NORMALIZED_ERROR
        and class_probability_error <= STREAMING_MAX_CLASS_PROBABILITY_ERROR
        and mask_probability_error <= STREAMING_MAX_MASK_PROBABILITY_ERROR
        and parity_queries["minimum_last_axis_cosine"] >= STREAMING_MIN_LAST_AXIS_COSINE
        and parity_state["minimum_last_axis_cosine"] >= STREAMING_MIN_LAST_AXIS_COSINE
    )
    execution = make_videomt_stage_pq_execution_receipt(
        prepared,
        result.upstream,
        host_dtype=torch.bfloat16,
    )

    stream = result.modalities.streams[0]
    if stream.tokens.shape != (1, 200, 1024) or not stream.valid.all():
        raise RuntimeError("the Stage-PQ boundary did not retain all 200 queries")
    if stream.canonical_token_ids is None or not torch.equal(
        stream.canonical_token_ids,
        torch.arange(200, device=device).unsqueeze(0),
    ):
        raise RuntimeError("the Stage-PQ query order changed")

    report = {
        "schema": "picf-next.videomt-stage-pq-calvin-probe/v2",
        "claim_scope": (
            "complete released frozen FP32 donor on five real causal CALVIN frames; "
            "not a full LingBot-host or action-quality result"
        ),
        "dataset": {
            "split": str(split),
            "manifest": str(manifest_path),
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "tree_sha256": manifest.tree_sha256,
            "source_files": source_files,
        },
        "checkpoint": asdict(checkpoint_receipt),
        "runtime": {
            "device": str(device),
            "parameter_tensor_count": len(parameters),
            "parameter_numel": sum(parameter.numel() for parameter in parameters),
            "parameter_dtype": "torch.float32",
            "training": False,
            "requires_grad": False,
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "streaming_forward_seconds": streaming_forward_seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "execution": asdict(execution),
        "streaming_parity": {
            "mode": "one_frame_reset_then_four_one_frame_resume_calls",
            "passed": parity_passed,
            "tolerance_contract": {
                "maximum_absolute_error_over_global_scale": (STREAMING_MAX_SCALE_NORMALIZED_ERROR),
                "class_probability_maximum_absolute_error": (STREAMING_MAX_CLASS_PROBABILITY_ERROR),
                "mask_probability_maximum_absolute_error": (STREAMING_MAX_MASK_PROBABILITY_ERROR),
                "minimum_last_axis_cosine": STREAMING_MIN_LAST_AXIS_COSINE,
                "rationale": (
                    "functional FP32 batch-decomposition parity; stricter than the "
                    "measured FP32-to-BF16 host-boundary error"
                ),
            },
            "class_logits": parity_class,
            "mask_logits": parity_masks,
            "query_embeddings": parity_queries,
            "propagated_queries": parity_state,
            "class_probability_maximum_absolute_error": class_probability_error,
            "mask_probability_maximum_absolute_error": mask_probability_error,
        },
        "released_outputs": {
            "class_logits": _tensor_receipt(result.upstream.class_logits),
            "mask_logits": _tensor_receipt(result.upstream.mask_logits),
            "query_embeddings": _tensor_receipt(result.upstream.query_embeddings),
            "propagated_queries": _tensor_receipt(result.upstream.propagated_queries),
            "auxiliary_output_count": len(result.upstream.auxiliary_outputs),
            "auxiliary_note": (
                "the released source emits training-only auxiliary readouts only when "
                "model.training is true; released evaluation mode is preserved"
            ),
        },
        "host_boundary": {
            "name": stream.name,
            "tokens": _tensor_receipt(stream.tokens),
            "valid_shape": list(stream.valid.shape),
            "valid_count": int(stream.valid.sum()),
            "canonical_token_ids": [0, 199],
            "selection_pooling_resampling_or_second_normalization": False,
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output = args.output.expanduser().resolve()
    write_text_durable_exclusive(output, encoded)
    print(encoded, end="")
    if not parity_passed:
        raise RuntimeError("five-frame and one-frame-resume execution differ")


if __name__ == "__main__":
    main()
