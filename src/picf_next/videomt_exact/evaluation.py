"""Task-blind video-level anchor evaluation for exact VidEoMT outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from picf_next.videomt_exact.calvin_dataset import HashBoundCalvinFrameStore
from picf_next.videomt_exact.calvin_targets import (
    PreparedCalvinVidEoMTClip,
    prepare_calvin_videomt_clip,
)
from picf_next.videomt_exact.class_agnostic import marginalize_videomt_taxonomy
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput, ExactVidEoMTRuntime


@dataclass(frozen=True, slots=True)
class VidEoMTRankedProposalEvaluation:
    top_k: int
    query_indices: tuple[int, ...]
    soft_ious: tuple[float, ...]
    binary_ious: tuple[float, ...]
    foreground_probabilities: tuple[float, ...]
    mean_soft_iou: float
    mean_binary_iou: float
    recall_at_50: float

    def to_dict(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "query_indices": list(self.query_indices),
            "soft_ious": list(self.soft_ious),
            "binary_ious": list(self.binary_ious),
            "foreground_probabilities": list(self.foreground_probabilities),
            "mean_soft_iou": self.mean_soft_iou,
            "mean_binary_iou": self.mean_binary_iou,
            "recall_at_50": self.recall_at_50,
        }


@dataclass(frozen=True, slots=True)
class VidEoMTAnchorEvaluation:
    identity_keys: tuple[str, ...]
    query_indices: tuple[int, ...]
    soft_ious: tuple[float, ...]
    binary_ious: tuple[float, ...]
    foreground_probabilities: tuple[float, ...]
    mean_soft_iou: float
    mean_binary_iou: float
    recall_at_50: float
    mean_foreground_probability: float
    ranked_proposals: tuple[VidEoMTRankedProposalEvaluation, ...]
    foreground_query_counts: tuple[tuple[float, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_keys": list(self.identity_keys),
            "query_indices": list(self.query_indices),
            "soft_ious": list(self.soft_ious),
            "binary_ious": list(self.binary_ious),
            "foreground_probabilities": list(self.foreground_probabilities),
            "mean_soft_iou": self.mean_soft_iou,
            "mean_binary_iou": self.mean_binary_iou,
            "recall_at_50": self.recall_at_50,
            "mean_foreground_probability": self.mean_foreground_probability,
            "ranked_proposals": [value.to_dict() for value in self.ranked_proposals],
            "foreground_query_counts": {
                str(threshold): count for threshold, count in self.foreground_query_counts
            },
        }


def _video_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_pixels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise IoU over all frames and pixels: predictions[Q,T,H,W], targets[N,T,H,W]."""

    if valid_pixels is None:
        valid_pixels = torch.ones(
            targets.shape[1:],
            dtype=torch.bool,
            device=targets.device,
        )
    if valid_pixels.dtype != torch.bool or valid_pixels.shape != targets.shape[1:]:
        raise ValueError("video IoU valid-pixel axes disagree with targets")
    # Evaluation is called beneath the model's fp16 autocast region. Explicit
    # casts alone do not prevent autocast-eligible reductions from returning to
    # fp16, whose finite range is too small for video-sized pixel sums.
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        validity = valid_pixels.flatten().to(dtype=torch.float32)
        if validity.sum() <= 0:
            raise ValueError("video IoU has no measured pixel")
        pred = predictions.flatten(1).to(dtype=torch.float32)
        target = targets.flatten(1).to(dtype=torch.float32)
        pred = pred * validity
        target = target * validity
        intersection = torch.einsum("ql,nl->qn", pred, target)
        union = pred.sum(dim=1, keepdim=True) + target.sum(dim=1).unsqueeze(0) - intersection
        result = intersection / union.clamp_min(1.0e-6)
    if not torch.isfinite(result).all():
        raise FloatingPointError("video IoU produced non-finite FP32 values")
    return result


def _ranked_proposal_evaluations(
    *,
    soft_iou: torch.Tensor,
    binary_iou: torch.Tensor,
    foreground: torch.Tensor,
) -> tuple[VidEoMTRankedProposalEvaluation, ...]:
    query_count, target_count = soft_iou.shape
    if binary_iou.shape != soft_iou.shape or foreground.shape != (query_count,):
        raise ValueError("ranked proposal inputs have incompatible axes")
    ranked_queries = foreground.argsort(descending=True)
    top_ks = tuple(sorted({min(value, query_count) for value in (10, 25, 50, 100, 200)}))
    evaluations = []
    for top_k in top_ks:
        selected = ranked_queries[:top_k]
        selected_soft = soft_iou[selected]
        rows, columns = linear_sum_assignment(-selected_soft.detach().cpu().numpy())
        by_target = {int(target): int(query) for query, target in zip(rows, columns, strict=True)}
        if len(by_target) != min(top_k, target_count):
            raise RuntimeError("ranked proposal matching returned an incomplete assignment")
        query_indices = [-1] * target_count
        soft_values = [0.0] * target_count
        binary_values = [0.0] * target_count
        foreground_values = [0.0] * target_count
        for target, local_query in by_target.items():
            global_query = int(selected[local_query])
            query_indices[target] = global_query
            soft_values[target] = float(soft_iou[global_query, target])
            binary_values[target] = float(binary_iou[global_query, target])
            foreground_values[target] = float(foreground[global_query])
        denominator = target_count
        evaluations.append(
            VidEoMTRankedProposalEvaluation(
                top_k=top_k,
                query_indices=tuple(query_indices),
                soft_ious=tuple(soft_values),
                binary_ious=tuple(binary_values),
                foreground_probabilities=tuple(foreground_values),
                mean_soft_iou=sum(soft_values) / denominator,
                mean_binary_iou=sum(binary_values) / denominator,
                recall_at_50=sum(value >= 0.5 for value in binary_values) / denominator,
            )
        )
    return tuple(evaluations)


def evaluate_videomt_anchors(
    output: ExactVidEoMTOutput,
    clip: PreparedCalvinVidEoMTClip,
) -> VidEoMTAnchorEvaluation:
    """Match one propagated query to each physical identity over the entire clip."""

    if output.class_logits.shape[0] != 1 or output.mask_logits.shape[0] != 1:
        raise ValueError("anchor evaluation currently requires one video")
    target_masks = clip.target["masks"].to(output.mask_logits.device)
    valid_pixels = clip.target.get("valid_pixels")
    if valid_pixels is not None:
        valid_pixels = valid_pixels.to(output.mask_logits.device)
    if target_masks.shape[1] != output.mask_logits.shape[2]:
        raise ValueError("prediction and target clip lengths differ")
    with torch.autocast(device_type=output.mask_logits.device.type, enabled=False):
        mask_logits = F.interpolate(
            output.mask_logits[0].float(),
            size=clip.frames.padded_size,
            mode="bilinear",
            align_corners=False,
        )
        mask_probabilities = mask_logits.sigmoid()
        foreground = marginalize_videomt_taxonomy(output.class_logits[0].float()).softmax(
            dim=-1
        )
        foreground = foreground[..., 0].mean(dim=0)
    if not torch.isfinite(mask_probabilities).all() or not torch.isfinite(foreground).all():
        raise FloatingPointError("anchor evaluation received non-finite FP32 probabilities")

    soft_iou = _video_iou(mask_probabilities, target_masks, valid_pixels)
    binary_iou = _video_iou(
        (mask_probabilities >= 0.5).float(),
        target_masks,
        valid_pixels,
    )
    rows, columns = linear_sum_assignment(-soft_iou.detach().cpu().numpy())
    by_target = {int(target): int(query) for query, target in zip(rows, columns, strict=True)}
    if set(by_target) != set(range(len(clip.identity_keys))):
        raise RuntimeError("video Hungarian matching did not cover every target identity")

    query_indices = tuple(by_target[index] for index in range(len(clip.identity_keys)))
    soft_values = tuple(
        float(soft_iou[query, target]) for target, query in enumerate(query_indices)
    )
    binary_values = tuple(
        float(binary_iou[query, target]) for target, query in enumerate(query_indices)
    )
    foreground_values = tuple(float(foreground[query]) for query in query_indices)
    count = len(query_indices)
    ranked_proposals = _ranked_proposal_evaluations(
        soft_iou=soft_iou,
        binary_iou=binary_iou,
        foreground=foreground,
    )
    return VidEoMTAnchorEvaluation(
        identity_keys=clip.identity_keys,
        query_indices=query_indices,
        soft_ious=soft_values,
        binary_ious=binary_values,
        foreground_probabilities=foreground_values,
        mean_soft_iou=sum(soft_values) / count,
        mean_binary_iou=sum(binary_values) / count,
        recall_at_50=sum(value >= 0.5 for value in binary_values) / count,
        mean_foreground_probability=sum(foreground_values) / count,
        ranked_proposals=ranked_proposals,
        foreground_query_counts=tuple(
            (threshold, int((foreground >= threshold).sum())) for threshold in (0.1, 0.25, 0.5)
        ),
    )


_PALETTE = (
    (230, 57, 70),
    (35, 133, 214),
    (28, 166, 94),
    (246, 166, 35),
    (151, 91, 201),
    (23, 174, 176),
    (222, 91, 160),
    (126, 111, 92),
    (117, 169, 57),
    (76, 86, 166),
    (232, 112, 46),
    (93, 181, 211),
)


def _overlay(base: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    result = base.astype(np.float32).copy()
    for index in range(count):
        mask = labels == index
        if mask.any():
            color = np.asarray(_PALETTE[index % len(_PALETTE)], dtype=np.float32)
            result[mask] = result[mask] * 0.35 + color * 0.65
    return np.clip(result, 0, 255).astype(np.uint8)


def render_videomt_anchor_panel(
    output: ExactVidEoMTOutput,
    clip: PreparedCalvinVidEoMTClip,
    evaluation: VidEoMTAnchorEvaluation,
    *,
    output_path: Path,
    frame_index: int | None = None,
    ranked_proposal: VidEoMTRankedProposalEvaluation | None = None,
) -> None:
    """Render source, physical truth, and oracle or model-ranked query masks."""

    time = output.class_logits.shape[1]
    selected_time = time // 2 if frame_index is None else frame_index
    if not 0 <= selected_time < time:
        raise ValueError("visualization frame index is outside the clip")
    rgb = clip.frames.resized_rgb[selected_time]
    height, width = rgb.shape[:2]
    target = clip.target["masks"][:, selected_time, :height, :width]
    valid_pixels = clip.target.get("valid_pixels")
    selected_validity = (
        torch.ones((height, width), dtype=torch.bool)
        if valid_pixels is None
        else valid_pixels[selected_time, :height, :width]
    )
    truth_labels = torch.full((height, width), -1, dtype=torch.long)
    for identity_index, mask in enumerate(target):
        truth_labels[mask.bool()] = identity_index

    masks = (
        F.interpolate(
            output.mask_logits[0],
            size=clip.frames.padded_size,
            mode="bilinear",
            align_corners=False,
        )[:, selected_time, :height, :width]
        .sigmoid()
        .float()
    )
    foreground = marginalize_videomt_taxonomy(output.class_logits[0]).softmax(dim=-1)
    foreground = foreground[selected_time, :, 0].float()
    if ranked_proposal is None:
        query_indices = evaluation.query_indices
        binary_ious = evaluation.binary_ious
        foreground_probabilities = evaluation.foreground_probabilities
        prediction_title = "Oracle-matched VidEoMT queries"
    else:
        if len(ranked_proposal.query_indices) != len(evaluation.identity_keys) or any(
            query < 0 for query in ranked_proposal.query_indices
        ):
            raise ValueError("ranked panel requires one selected query per physical identity")
        query_indices = ranked_proposal.query_indices
        binary_ious = ranked_proposal.binary_ious
        foreground_probabilities = ranked_proposal.foreground_probabilities
        prediction_title = f"Model-ranked Top-{ranked_proposal.top_k} queries"
    # Query ranking already uses foreground confidence. Keep that confidence in
    # the legend, but render the selected query's mask geometry itself so a
    # low class score cannot make a spatially useful mask invisible to humans.
    selected_masks = torch.stack([masks[query] for query in query_indices], dim=0)
    selected_masks[:, ~selected_validity.to(selected_masks.device)] = 0.0
    best_value, prediction_labels = selected_masks.max(dim=0)
    prediction_labels[best_value < 0.5] = -1

    source = Image.fromarray(rgb)
    truth = Image.fromarray(_overlay(rgb, truth_labels.numpy(), len(evaluation.identity_keys)))
    prediction = Image.fromarray(
        _overlay(rgb, prediction_labels.detach().cpu().numpy(), len(evaluation.identity_keys))
    )
    header = 28
    legend_line = 18
    legend_height = legend_line * len(evaluation.identity_keys) + 12
    canvas = Image.new("RGB", (width * 3, height + header + legend_height), "white")
    canvas.paste(source, (0, header))
    canvas.paste(truth, (width, header))
    canvas.paste(prediction, (width * 2, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "RGB input (task text hidden)", fill="black")
    draw.text((width + 8, 7), "Physical owner target", fill="black")
    draw.text((width * 2 + 8, 7), prediction_title, fill="black")
    y = height + header + 6
    for index, (identity, query, iou, confidence) in enumerate(
        zip(
            evaluation.identity_keys,
            query_indices,
            binary_ious,
            foreground_probabilities,
            strict=True,
        )
    ):
        color = _PALETTE[index % len(_PALETTE)]
        draw.rectangle((8, y + 3, 19, y + 14), fill=color)
        draw.text(
            (26, y),
            f"{identity}  query={query:03d}  video IoU={iou:.3f}  P(object)={confidence:.3f}",
            fill="black",
        )
        y += legend_line
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def evaluate_calvin_anchor_windows(
    *,
    runtime: ExactVidEoMTRuntime,
    store: HashBoundCalvinFrameStore,
    windows: tuple[tuple[int, ...], ...],
    short_edge: int,
    device: torch.device,
    dtype: torch.dtype,
    panel_path: Path,
) -> dict[str, object]:
    """Evaluate a fixed CALVIN window set with oracle and model-ranked metrics."""

    if not windows:
        raise ValueError("CALVIN anchor evaluation requires at least one window")
    runtime.eval()
    clips: list[dict[str, object]] = []
    all_soft: list[float] = []
    all_binary: list[float] = []
    all_foreground: list[float] = []
    all_recall: list[float] = []
    ranked: dict[int, dict[str, list[float]]] = {}
    by_identity: dict[str, dict[str, list[float]]] = {}
    ranked_by_identity: dict[int, dict[str, dict[str, list[float]]]] = {}
    foreground_counts: dict[float, list[int]] = {}
    for index, window in enumerate(windows):
        source = store.clip(window)
        clip = prepare_calvin_videomt_clip(
            source.rgb_static,
            source.supervision,
            short_edge=short_edge,
            max_size=short_edge,
        )
        output = runtime(clip.frames.model_input.to(device=device, dtype=dtype))
        evaluation = evaluate_videomt_anchors(output, clip)
        value = evaluation.to_dict()
        value["global_indices"] = list(window)
        clips.append(value)
        all_soft.extend(evaluation.soft_ious)
        all_binary.extend(evaluation.binary_ious)
        all_foreground.extend(evaluation.foreground_probabilities)
        all_recall.append(evaluation.recall_at_50)
        for identity, soft, binary, confidence in zip(
            evaluation.identity_keys,
            evaluation.soft_ious,
            evaluation.binary_ious,
            evaluation.foreground_probabilities,
            strict=True,
        ):
            values = by_identity.setdefault(
                identity,
                {"soft_iou": [], "binary_iou": [], "foreground_probability": []},
            )
            values["soft_iou"].append(soft)
            values["binary_iou"].append(binary)
            values["foreground_probability"].append(confidence)
        for proposal in evaluation.ranked_proposals:
            values = ranked.setdefault(
                proposal.top_k,
                {"mean_soft_iou": [], "mean_binary_iou": [], "recall_at_50": []},
            )
            values["mean_soft_iou"].append(proposal.mean_soft_iou)
            values["mean_binary_iou"].append(proposal.mean_binary_iou)
            values["recall_at_50"].append(proposal.recall_at_50)
            identity_values = ranked_by_identity.setdefault(proposal.top_k, {})
            for identity, soft, binary, confidence in zip(
                evaluation.identity_keys,
                proposal.soft_ious,
                proposal.binary_ious,
                proposal.foreground_probabilities,
                strict=True,
            ):
                measurements = identity_values.setdefault(
                    identity,
                    {"soft_iou": [], "binary_iou": [], "foreground_probability": []},
                )
                measurements["soft_iou"].append(soft)
                measurements["binary_iou"].append(binary)
                measurements["foreground_probability"].append(confidence)
        for threshold, count in evaluation.foreground_query_counts:
            foreground_counts.setdefault(threshold, []).append(count)
        if index == 0:
            render_videomt_anchor_panel(output, clip, evaluation, output_path=panel_path)
            top_10 = next(value for value in evaluation.ranked_proposals if value.top_k == 10)
            render_videomt_anchor_panel(
                output,
                clip,
                evaluation,
                output_path=panel_path.with_name(f"{panel_path.stem}_top10{panel_path.suffix}"),
                ranked_proposal=top_10,
            )
    return {
        "clip_count": len(clips),
        "object_observation_count": len(all_soft),
        "mean_soft_iou": float(np.mean(all_soft)),
        "mean_binary_iou": float(np.mean(all_binary)),
        "recall_at_50": float(np.mean(all_recall)),
        "mean_foreground_probability": float(np.mean(all_foreground)),
        "ranked_proposals": {
            str(top_k): {
                name: float(np.mean(measurements)) for name, measurements in values.items()
            }
            for top_k, values in sorted(ranked.items())
        },
        "by_identity": {
            identity: {
                "observation_count": len(values["soft_iou"]),
                "mean_soft_iou": float(np.mean(values["soft_iou"])),
                "mean_binary_iou": float(np.mean(values["binary_iou"])),
                "recall_at_50": float(
                    np.mean(np.asarray(values["binary_iou"], dtype=np.float32) >= 0.5)
                ),
                "mean_foreground_probability": float(np.mean(values["foreground_probability"])),
            }
            for identity, values in sorted(by_identity.items())
        },
        "ranked_proposals_by_identity": {
            str(top_k): {
                identity: {
                    "observation_count": len(values["soft_iou"]),
                    "mean_soft_iou": float(np.mean(values["soft_iou"])),
                    "mean_binary_iou": float(np.mean(values["binary_iou"])),
                    "recall_at_50": float(
                        np.mean(np.asarray(values["binary_iou"], dtype=np.float32) >= 0.5)
                    ),
                    "mean_foreground_probability": float(np.mean(values["foreground_probability"])),
                }
                for identity, values in sorted(identity_values.items())
            }
            for top_k, identity_values in sorted(ranked_by_identity.items())
        },
        "mean_foreground_query_counts": {
            str(threshold): float(np.mean(counts))
            for threshold, counts in sorted(foreground_counts.items())
        },
        "clips": clips,
    }
