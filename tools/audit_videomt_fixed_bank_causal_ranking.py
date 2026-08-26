#!/usr/bin/env python3
"""Evaluate cold versus strictly causal VidEoMT ranking on a fixed action bank."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt
from tools.audit_videomt_temporal_ranking_protocol import (
    _EvaluationStore,
    _atomic_json,
    _causal_outputs,
    _cold_output,
    _prepare_clip,
    _project_one_frame,
    _read_json,
    _render_mode,
)
from picf_next.videomt_exact.checkpoint import sha256_file
from picf_next.videomt_exact.evaluation import evaluate_videomt_anchors


SCHEMA = "picf-next.videomt-fixed-bank-causal-ranking-audit/v1"
MODES = (
    "cold_current",
    "causal_warm_current",
    "causal_history_rank_current",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-checkpoint", required=True, type=Path)
    parser.add_argument("--adapted-checkpoint", required=True, type=Path)
    parser.add_argument("--adapted-checkpoint-sha256", required=True)
    parser.add_argument("--dinov3-bundle", required=True, type=Path)
    parser.add_argument("--source-split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--physical-sidecar-manifest", required=True, type=Path)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--fixed-anchor-report", required=True, type=Path)
    parser.add_argument("--short-edge", type=int, default=480)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-samples", type=int)
    return parser.parse_args()


def _samples(
    report: Mapping[str, object],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if report.get("state_mode") != "cold_reset":
        raise ValueError("fixed anchor report is not the registered cold-reset bank")
    raw = report.get("samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError("fixed anchor report has no samples")
    result: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for value in raw:
        if not isinstance(value, dict):
            raise TypeError("fixed anchor sample is not a mapping")
        partition = value.get("partition")
        source = value.get("source_global_index")
        transition = value.get("transition_index")
        if (
            partition not in {"validation", "heldout"}
            or isinstance(source, bool)
            or not isinstance(source, int)
            or isinstance(transition, bool)
            or not isinstance(transition, int)
        ):
            raise ValueError("fixed anchor sample identity is malformed")
        key = str(partition), source
        if key in seen:
            raise ValueError("fixed anchor report repeats a source sample")
        seen.add(key)
        if source < 4 or transition < 4:
            excluded.append(
                {
                    "sample_key": value.get("sample_key"),
                    "partition": partition,
                    "source_global_index": source,
                    "transition_index": transition,
                    "reason": "fewer_than_four_real_predecessor_frames",
                }
            )
            continue
        result.append(value)
    if not result:
        raise ValueError("fixed anchor report has no causal-prefix-eligible sample")
    return tuple(result), tuple(excluded)


def _top10_record(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    ranked = evaluation.get("ranked_proposals")
    if not isinstance(ranked, list):
        raise TypeError("sample evaluation has no ranked proposals")
    matches = tuple(
        item for item in ranked if isinstance(item, Mapping) and item.get("top_k") == 10
    )
    if len(matches) != 1:
        raise RuntimeError("sample evaluation does not contain one top-10 record")
    return matches[0]


def _aggregate(records: tuple[Mapping[str, object], ...]) -> dict[str, object]:
    if not records:
        raise ValueError("cannot aggregate an empty fixed-bank partition")
    soft = [float(value) for item in records for value in item["soft_ious"]]
    binary = [float(value) for item in records for value in item["binary_ious"]]
    foreground = [
        float(value) for item in records for value in item["foreground_probabilities"]
    ]
    top10 = tuple(_top10_record(item) for item in records)
    top10_soft = [float(value) for item in top10 for value in item["soft_ious"]]
    top10_binary = [float(value) for item in top10 for value in item["binary_ious"]]
    return {
        "sample_count": len(records),
        "object_observation_count": len(soft),
        "mean_soft_iou": float(np.mean(soft)),
        "mean_binary_iou": float(np.mean(binary)),
        "recall_at_50": float(np.mean(np.asarray(binary) >= 0.5)),
        "mean_foreground_probability": float(np.mean(foreground)),
        "top10": {
            "mean_soft_iou": float(np.mean(top10_soft)),
            "mean_binary_iou": float(np.mean(top10_binary)),
            "recall_at_50": float(np.mean(np.asarray(top10_binary) >= 0.5)),
        },
    }


def _init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def main() -> None:
    args = parse_args()
    if args.short_edge <= 0:
        raise ValueError("short edge must be positive")
    if args.maximum_samples is not None and args.maximum_samples <= 0:
        raise ValueError("maximum samples must be positive")
    rank, world_size, local_rank = _init_distributed()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    fixed_report, fixed_report_sha256 = _read_json(args.fixed_anchor_report)
    samples, excluded_samples = _samples(fixed_report)
    if args.maximum_samples is not None:
        samples = samples[: args.maximum_samples]
    local_samples = samples[rank::world_size]
    if not local_samples:
        raise ValueError("distributed rank received no fixed-bank samples")

    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    validate_dataset_runtime_binding(
        dataset_manifest,
        args.source_split_root.resolve(),
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        split_name=args.source_split_root.name,
    )
    index = CalvinDatasetIndex.load(
        args.source_split_root.resolve(),
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(
        args.sidecar_root.resolve(),
        index,
        manifest_path=args.physical_sidecar_manifest.resolve(),
        expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
        eager_coverage_scan=False,
    )
    if sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise RuntimeError("fixed-bank audit requires all-source physical supervision")
    store = _EvaluationStore(index, sidecar)
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.published_checkpoint,
            local_dinov3_bundle=args.dinov3_bundle,
            adapted_checkpoint_path=args.adapted_checkpoint,
            adapted_checkpoint_sha256=args.adapted_checkpoint_sha256,
            num_frames=5,
        ),
        device=device,
        dtype=torch.float32,
    )
    runtime.eval()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_records: list[dict[str, object]] = []
    visual_artifacts: list[dict[str, object]] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for local_ordinal, sample in enumerate(local_samples):
            source = int(sample["source_global_index"])
            prefix = tuple(range(source - 4, source + 1))
            causal_clip = _prepare_clip(store, prefix, short_edge=args.short_edge)
            current_clip = _prepare_clip(store, (source,), short_edge=args.short_edge)
            cold = _cold_output(runtime, current_clip, device=device)
            causal_frames = _causal_outputs(runtime, causal_clip, device=device)
            merged_logits = torch.cat(
                tuple(value.class_logits for value in causal_frames), dim=1
            )
            history_rank = _project_one_frame(
                causal_frames[-1],
                mask_time_index=0,
                rank_logits=merged_logits.mean(dim=1, keepdim=True),
            )
            outputs = {
                "cold_current": cold,
                "causal_warm_current": causal_frames[-1],
                "causal_history_rank_current": history_rank,
            }
            mode_records: dict[str, object] = {}
            for mode, output in outputs.items():
                evaluation = evaluate_videomt_anchors(output, current_clip)
                mode_records[mode] = evaluation.to_dict()
                if rank == 0 and local_ordinal == 0:
                    visual_artifacts.extend(
                        _render_mode(
                            name=mode,
                            output=output,
                            clip=current_clip,
                            evaluation=evaluation,
                            output_dir=output_dir / "visuals",
                        )
                    )
            local_records.append(
                {
                    "sample_key": sample["sample_key"],
                    "partition": sample["partition"],
                    "source_global_index": source,
                    "source_episode_index": sample["source_episode_index"],
                    "transition_index": sample["transition_index"],
                    "causal_prefix_global_indices": list(prefix),
                    "modes": mode_records,
                }
            )

    shard = {
        "rank": rank,
        "world_size": world_size,
        "sample_count": len(local_records),
        "samples": local_records,
        "visual_artifacts": visual_artifacts,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    shard_path = output_dir / f"rank_{rank:02d}.json"
    _atomic_json(shard_path, shard)
    _barrier(world_size)

    if rank == 0:
        shards = []
        for shard_rank in range(world_size):
            value, _sha = _read_json(output_dir / f"rank_{shard_rank:02d}.json")
            shards.append(value)
        all_samples = tuple(
            sample
            for shard in shards
            for sample in shard["samples"]
            if isinstance(sample, Mapping)
        )
        if len(all_samples) != len(samples):
            raise RuntimeError("fixed-bank shards did not cover every selected sample")
        summaries: dict[str, object] = {}
        for mode in MODES:
            partition_values = {}
            for partition in ("validation", "heldout"):
                records = tuple(
                    sample["modes"][mode]
                    for sample in all_samples
                    if sample["partition"] == partition
                )
                partition_values[partition] = _aggregate(records)
            summaries[mode] = partition_values
        cold_reproduction = {}
        for partition in ("validation", "heldout"):
            expected_records = tuple(
                sample for sample in samples if sample["partition"] == partition
            )
            expected = _aggregate(expected_records)
            observed = summaries["cold_current"][partition]
            if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
                raise TypeError("fixed-bank partition summary changed type")
            expected_top10 = expected["top10"]
            observed_top10 = observed["top10"]
            cold_reproduction[partition] = {
                "oracle_mean_binary_iou_delta": float(observed["mean_binary_iou"])
                - float(expected["mean_binary_iou"]),
                "top10_mean_binary_iou_delta": float(observed_top10["mean_binary_iou"])
                - float(expected_top10["mean_binary_iou"]),
            }
        report = {
            "schema": SCHEMA,
            "status": "PASS",
            "claim_scope": (
                "strictly causal source-ranking audit over the registered fixed action bank; "
                "not action evidence and not long-training authorization"
            ),
            "model_changes": [],
            "physical_target_forward_input": False,
            "target_prepared_before_forward_for_evaluation_only": True,
            "assets": {
                "fixed_anchor_report": str(args.fixed_anchor_report.resolve()),
                "fixed_anchor_report_sha256": fixed_report_sha256,
                "published_checkpoint_sha256": sha256_file(args.published_checkpoint),
                "adapted_checkpoint_sha256": sha256_file(args.adapted_checkpoint),
                "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
                "physical_sidecar_manifest_sha256": (
                    args.physical_sidecar_manifest_sha256
                ),
            },
            "protocol": {
                "short_edge": args.short_edge,
                "sample_count": len(samples),
                "source_bank_sample_count": len(samples) + len(excluded_samples),
                "excluded_cold_only_sample_count": len(excluded_samples),
                "excluded_cold_only_samples": list(excluded_samples),
                "world_size": world_size,
                "causal_prefix_frames": 4,
                "precision": "fp32-parameters+released-fp16-autocast",
            },
            "cold_reproduction": cold_reproduction,
            "summaries": summaries,
            "samples": sorted(
                all_samples,
                key=lambda value: (str(value["partition"]), int(value["source_global_index"])),
            ),
            "visual_artifacts": [
                artifact for shard in shards for artifact in shard["visual_artifacts"]
            ],
            "rank_shards": [
                {
                    "rank": shard["rank"],
                    "sample_count": shard["sample_count"],
                    "elapsed_seconds": shard["elapsed_seconds"],
                    "peak_cuda_allocated_bytes": shard["peak_cuda_allocated_bytes"],
                }
                for shard in shards
            ],
        }
        _atomic_json(output_dir / "report.json", report)
        print(json.dumps(report["summaries"], indent=2))
    _barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
