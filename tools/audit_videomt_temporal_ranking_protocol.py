#!/usr/bin/env python3
"""Factor cold state, causal history, and future-window ranking for VidEoMT.

This is a read-only evaluation.  It deliberately reuses the exact adapted
VidEoMT checkpoint, preprocessing, physical targets, and ranking metric.  No
parameter, learned head, label-conditioned input, or training objective is
introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
from picf_next.videomt_exact.calvin_full_dataset import materialize_calvin_videomt_clip
from picf_next.videomt_exact.calvin_targets import (
    PreparedCalvinVidEoMTClip,
    prepare_calvin_videomt_clip,
)
from picf_next.videomt_exact.checkpoint import sha256_file
from picf_next.videomt_exact.evaluation import (
    VidEoMTAnchorEvaluation,
    evaluate_videomt_anchors,
    render_videomt_anchor_panel,
)
from picf_next.videomt_exact.runtime import (
    ExactVidEoMTConfig,
    ExactVidEoMTRuntime,
    load_exact_videomt,
)


SCHEMA = "picf-next.videomt-temporal-ranking-protocol-audit/v1"
MODE_DESCRIPTIONS = {
    "cold_first": "first frame only; learned cold queries; current-frame score and mask",
    "future_rank_first": (
        "first-frame mask ranked by mean class logits from current plus four future frames; "
        "noncausal diagnostic"
    ),
    "historical_future_window": (
        "the historical Stage-D five-frame metric: current plus four future frames, "
        "mean class score, and video-level IoU"
    ),
    "cold_last": "last frame only; learned cold queries; current-frame score and mask",
    "causal_warm_last": (
        "last-frame score and mask after four strictly earlier recurrent frames"
    ),
    "causal_history_rank_last": (
        "last-frame mask ranked by mean class logits over four past frames plus current; "
        "strictly causal diagnostic"
    ),
}


class _EvaluationStore:
    """The exact all-source materialization boundary used by Stage-D."""

    def __init__(
        self,
        index: CalvinDatasetIndex,
        sidecar: CalvinPhysicalSupervisionSidecar,
    ) -> None:
        self.index = index
        self.sidecar = sidecar

    def clip(self, global_indices: tuple[int, ...]) -> Any:
        return materialize_calvin_videomt_clip(self.index, self.sidecar, global_indices)


@dataclass(frozen=True, slots=True)
class _PredictionOutput:
    """Only the two unchanged source tensors consumed by anchor evaluation."""

    class_logits: torch.Tensor
    mask_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.class_logits.ndim != 4 or self.mask_logits.ndim != 5:
            raise ValueError("raw VidEoMT predictions changed rank")
        if (
            self.class_logits.shape[0] != self.mask_logits.shape[0]
            or self.class_logits.shape[1] != self.mask_logits.shape[2]
            or self.class_logits.shape[2] != self.mask_logits.shape[1]
        ):
            raise ValueError("raw VidEoMT prediction axes disagree")
        if not torch.isfinite(self.class_logits).all() or not torch.isfinite(
            self.mask_logits
        ).all():
            raise FloatingPointError("raw VidEoMT predictions are non-finite")


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
    parser.add_argument("--historical-report", required=True, type=Path)
    parser.add_argument("--historical-step", type=int, default=250)
    parser.add_argument("--partition", choices=("train", "heldout"), default="heldout")
    parser.add_argument("--short-edge", type=int, default=480)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-windows", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not a mapping: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _historical_windows(
    report: Mapping[str, object],
    *,
    step: int,
    partition: str,
) -> tuple[tuple[int, ...], ...]:
    evaluations = report.get("evaluations")
    if not isinstance(evaluations, Mapping):
        raise TypeError("historical report has no evaluation mapping")
    step_report = evaluations.get(str(step))
    if not isinstance(step_report, Mapping):
        raise KeyError(f"historical report has no step {step}")
    partition_report = step_report.get(partition)
    if not isinstance(partition_report, Mapping):
        raise KeyError(f"historical report has no {partition!r} evaluation")
    clips = partition_report.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("historical report has no evaluated clips")
    windows: list[tuple[int, ...]] = []
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise TypeError("historical clip record is not a mapping")
        raw = clip.get("global_indices")
        if not isinstance(raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw
        ):
            raise TypeError("historical clip global indices are malformed")
        window = tuple(raw)
        if len(window) != 5 or any(
            right != left + 1 for left, right in zip(window, window[1:], strict=False)
        ):
            raise ValueError("historical evaluation did not use a five-frame source window")
        windows.append(window)
    if len(set(windows)) != len(windows):
        raise ValueError("historical report repeats an evaluation window")
    return tuple(windows)


def _historical_summary(
    report: Mapping[str, object],
    *,
    step: int,
    partition: str,
) -> Mapping[str, object]:
    evaluations = report["evaluations"]
    if not isinstance(evaluations, Mapping):
        raise TypeError("historical evaluations changed type")
    step_report = evaluations[str(step)]
    if not isinstance(step_report, Mapping):
        raise TypeError("historical step report changed type")
    result = step_report[partition]
    if not isinstance(result, Mapping):
        raise TypeError("historical partition report changed type")
    return result


def _project_one_frame(
    output: _PredictionOutput,
    *,
    mask_time_index: int,
    rank_logits: torch.Tensor,
) -> _PredictionOutput:
    """Keep one mask frame while substituting an explicitly declared rank statistic."""

    time = output.class_logits.shape[1]
    if not 0 <= mask_time_index < time:
        raise ValueError("mask time index is outside the VidEoMT output")
    if rank_logits.shape != output.class_logits[:, :1].shape:
        raise ValueError("rank logits must have one VidEoMT time step")
    return _PredictionOutput(
        class_logits=rank_logits,
        mask_logits=output.mask_logits[:, :, mask_time_index : mask_time_index + 1],
    )


def _top10(evaluation: VidEoMTAnchorEvaluation):
    matches = tuple(value for value in evaluation.ranked_proposals if value.top_k == 10)
    if len(matches) != 1:
        raise RuntimeError("VidEoMT evaluation did not expose exactly one top-10 metric")
    return matches[0]


def _aggregate(evaluations: Sequence[VidEoMTAnchorEvaluation]) -> dict[str, object]:
    if not evaluations:
        raise ValueError("cannot aggregate an empty VidEoMT evaluation")
    soft = [value for item in evaluations for value in item.soft_ious]
    binary = [value for item in evaluations for value in item.binary_ious]
    foreground = [value for item in evaluations for value in item.foreground_probabilities]
    top10 = tuple(_top10(item) for item in evaluations)
    top10_soft = [value for item in top10 for value in item.soft_ious]
    top10_binary = [value for item in top10 for value in item.binary_ious]
    return {
        "clip_count": len(evaluations),
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


def _prepare_clip(
    store: _EvaluationStore,
    indices: tuple[int, ...],
    *,
    short_edge: int,
) -> PreparedCalvinVidEoMTClip:
    source = store.clip(indices)
    return prepare_calvin_videomt_clip(
        source.rgb_static,
        source.supervision,
        short_edge=short_edge,
        max_size=short_edge,
    )


def _source_prediction(outputs: Mapping[str, object]) -> _PredictionOutput:
    logits = outputs.get("pred_logits")
    masks = outputs.get("pred_masks")
    if not isinstance(logits, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError("released VidEoMT source did not return prediction tensors")
    return _PredictionOutput(class_logits=logits, mask_logits=masks)


def _cold_output(
    runtime: ExactVidEoMTRuntime,
    clip: PreparedCalvinVidEoMTClip,
    *,
    device: torch.device,
) -> _PredictionOutput:
    original_num_frames = runtime.model.num_frames
    runtime.reset_state()
    try:
        runtime.model.num_frames = 1
        outputs = runtime.model(
            clip.frames.model_input.to(device=device, dtype=torch.float32),
            resume=False,
        )
        return _source_prediction(outputs)
    finally:
        runtime.model.num_frames = original_num_frames
        runtime.reset_state()


def _causal_outputs(
    runtime: ExactVidEoMTRuntime,
    clip: PreparedCalvinVidEoMTClip,
    *,
    device: torch.device,
) -> tuple[_PredictionOutput, ...]:
    frames = clip.frames.model_input.to(device=device, dtype=torch.float32)
    original_num_frames = runtime.model.num_frames
    runtime.reset_state()
    values: list[_PredictionOutput] = []
    try:
        runtime.model.num_frames = 1
        for frame_index in range(frames.shape[0]):
            outputs = runtime.model(frames[frame_index : frame_index + 1], resume=frame_index > 0)
            values.append(_source_prediction(outputs))
    finally:
        runtime.model.num_frames = original_num_frames
        runtime.reset_state()
    return tuple(values)


def _merge_causal_outputs(outputs: Sequence[_PredictionOutput]) -> _PredictionOutput:
    if not outputs:
        raise ValueError("cannot merge an empty causal prediction sequence")
    return _PredictionOutput(
        class_logits=torch.cat(tuple(value.class_logits for value in outputs), dim=1),
        mask_logits=torch.cat(tuple(value.mask_logits for value in outputs), dim=2),
    )


def _render_mode(
    *,
    name: str,
    output: Any,
    clip: PreparedCalvinVidEoMTClip,
    evaluation: VidEoMTAnchorEvaluation,
    output_dir: Path,
    frame_index: int = 0,
) -> list[dict[str, object]]:
    artifacts = []
    for kind, proposal in (("oracle", None), ("top10", _top10(evaluation))):
        path = output_dir / f"{name}_{kind}.png"
        render_videomt_anchor_panel(
            output,
            clip,
            evaluation,
            output_path=path,
            frame_index=frame_index,
            ranked_proposal=proposal,
        )
        artifacts.append(
            {
                "mode": name,
                "kind": kind,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def _metric_delta(
    observed: Mapping[str, object],
    historical: Mapping[str, object],
) -> dict[str, float]:
    historical_ranked = historical.get("ranked_proposals")
    if not isinstance(historical_ranked, Mapping):
        raise TypeError("historical ranked proposals changed type")
    historical_top10 = historical_ranked.get("10")
    if not isinstance(historical_top10, Mapping):
        raise TypeError("historical top-10 report changed type")
    observed_top10 = observed.get("top10")
    if not isinstance(observed_top10, Mapping):
        raise TypeError("observed top-10 report changed type")
    return {
        "oracle_mean_soft_iou": float(observed["mean_soft_iou"])
        - float(historical["mean_soft_iou"]),
        "oracle_mean_binary_iou": float(observed["mean_binary_iou"])
        - float(historical["mean_binary_iou"]),
        "top10_mean_soft_iou": float(observed_top10["mean_soft_iou"])
        - float(historical_top10["mean_soft_iou"]),
        "top10_mean_binary_iou": float(observed_top10["mean_binary_iou"])
        - float(historical_top10["mean_binary_iou"]),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.historical_step < 0 or args.short_edge <= 0:
        raise ValueError("historical step and short edge are outside their valid range")
    if args.maximum_windows is not None and args.maximum_windows <= 0:
        raise ValueError("maximum windows must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the exact temporal ranking audit requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    historical_report, historical_report_sha256 = _read_json(args.historical_report)
    windows = _historical_windows(
        historical_report,
        step=args.historical_step,
        partition=args.partition,
    )
    if args.maximum_windows is not None:
        windows = windows[: args.maximum_windows]
    historical_summary = _historical_summary(
        historical_report,
        step=args.historical_step,
        partition=args.partition,
    )
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
        raise RuntimeError("temporal ranking audit requires all-source physical supervision")
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

    by_mode: dict[str, list[VidEoMTAnchorEvaluation]] = {
        name: [] for name in MODE_DESCRIPTIONS
    }
    clip_records: list[dict[str, object]] = []
    visual_artifacts: list[dict[str, object]] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for ordinal, window in enumerate(windows):
            full_clip = _prepare_clip(store, window, short_edge=args.short_edge)
            first_clip = _prepare_clip(store, (window[0],), short_edge=args.short_edge)
            last_clip = _prepare_clip(store, (window[-1],), short_edge=args.short_edge)

            cold_first = _cold_output(runtime, first_clip, device=device)
            cold_last = _cold_output(runtime, last_clip, device=device)

            runtime.reset_state()
            try:
                source_outputs = runtime.model(
                    full_clip.frames.model_input.to(device=device, dtype=torch.float32),
                    resume=False,
                )
                historical_full = _source_prediction(source_outputs)
            finally:
                runtime.reset_state()

            causal_frames = _causal_outputs(runtime, full_clip, device=device)
            causal_merged = _merge_causal_outputs(causal_frames)

            future_rank_first = _project_one_frame(
                historical_full,
                mask_time_index=0,
                rank_logits=historical_full.class_logits.mean(dim=1, keepdim=True),
            )
            causal_history_rank_last = _project_one_frame(
                causal_merged,
                mask_time_index=4,
                rank_logits=causal_merged.class_logits.mean(dim=1, keepdim=True),
            )
            outputs_and_clips = {
                "cold_first": (cold_first, first_clip),
                "future_rank_first": (future_rank_first, first_clip),
                "historical_future_window": (historical_full, full_clip),
                "cold_last": (cold_last, last_clip),
                "causal_warm_last": (causal_frames[-1], last_clip),
                "causal_history_rank_last": (causal_history_rank_last, last_clip),
            }
            record: dict[str, object] = {
                "ordinal": ordinal,
                "global_indices": list(window),
                "modes": {},
            }
            mode_records = record["modes"]
            if not isinstance(mode_records, dict):
                raise RuntimeError("internal mode record changed type")
            for name, (output, clip) in outputs_and_clips.items():
                evaluation = evaluate_videomt_anchors(output, clip)
                by_mode[name].append(evaluation)
                mode_records[name] = evaluation.to_dict()
                if ordinal == 0:
                    visual_artifacts.extend(
                        _render_mode(
                            name=name,
                            output=output,
                            clip=clip,
                            evaluation=evaluation,
                            output_dir=args.output_dir / "visuals",
                            frame_index=0,
                        )
                    )
            clip_records.append(record)

    summaries = {name: _aggregate(values) for name, values in by_mode.items()}
    historical_delta = _metric_delta(
        summaries["historical_future_window"],
        historical_summary,
    )
    exact_reproduction = all(abs(value) <= 1.0e-5 for value in historical_delta.values())
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PASS" if exact_reproduction else "FAIL",
        "claim_scope": (
            "read-only factorial diagnosis of VidEoMT temporal ranking protocol; "
            "not action evidence and not authorization for long training"
        ),
        "model_changes": [],
        "runtime_inputs": ["rgb_static"],
        "physical_target_forward_input": False,
        "target_prepared_before_forward_for_evaluation_only": True,
        "mode_descriptions": MODE_DESCRIPTIONS,
        "assets": {
            "historical_report": str(args.historical_report.resolve()),
            "historical_report_sha256": historical_report_sha256,
            "published_checkpoint": str(args.published_checkpoint.resolve()),
            "published_checkpoint_sha256": sha256_file(args.published_checkpoint),
            "adapted_checkpoint": str(args.adapted_checkpoint.resolve()),
            "adapted_checkpoint_sha256": sha256_file(args.adapted_checkpoint),
            "dinov3_bundle": str(args.dinov3_bundle.resolve()),
            "dataset_manifest": str(args.dataset_manifest.resolve()),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "physical_sidecar_manifest": str(args.physical_sidecar_manifest.resolve()),
            "physical_sidecar_manifest_sha256": args.physical_sidecar_manifest_sha256,
        },
        "protocol": {
            "historical_step": args.historical_step,
            "partition": args.partition,
            "short_edge": args.short_edge,
            "window_count": len(windows),
            "windows": [list(value) for value in windows],
            "precision": "fp32-parameters+released-fp16-autocast",
        },
        "historical_reproduction": {
            "passed": exact_reproduction,
            "metric_deltas": historical_delta,
        },
        "summaries": summaries,
        "clips": clip_records,
        "visual_artifacts": visual_artifacts,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    if not exact_reproduction:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
