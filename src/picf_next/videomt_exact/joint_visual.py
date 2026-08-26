"""Read-only human audit panels for the native VidEoMT query bank."""

from __future__ import annotations

import colorsys
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from picf_next.artifact_io import publish_prepared_file_durable_exclusive
from picf_next.videomt_exact.runtime import (
    VIDEOMT_PIXEL_MEAN_255,
    VIDEOMT_PIXEL_STD_255,
    ExactVidEoMTOutput,
)

NATIVE_VIDEOMT_QUERY_VISUAL_SCHEMA = "picf-next.native-videomt-query-visual.v1"
_PRESENTATION_SCORE_THRESHOLD = 0.25
_TOP_QUERY_PANELS = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str, *, maximum: int = 72) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return (normalized or "unnamed")[:maximum]


def _query_color(query_index: int) -> np.ndarray:
    hue = (query_index * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.94)
    return np.asarray((red, green, blue), dtype=np.float32) * 255.0


def _inverse_normalized_rgb(frame: torch.Tensor) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[0] != 3 or not frame.is_floating_point():
        raise ValueError("native VidEoMT visual frame must be floating CHW RGB")
    value = frame.detach().float().cpu()
    mean = value.new_tensor(VIDEOMT_PIXEL_MEAN_255).view(3, 1, 1)
    std = value.new_tensor(VIDEOMT_PIXEL_STD_255).view(3, 1, 1)
    value = value * std + mean
    return (
        value.permute(1, 2, 0)
        .clamp(0, 255)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def _measured_bounds(valid: torch.Tensor) -> tuple[slice, slice]:
    if valid.ndim != 2 or valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError("native VidEoMT visual requires measured current-frame pixels")
    rows = valid.any(dim=1).nonzero().flatten()
    columns = valid.any(dim=0).nonzero().flatten()
    return (
        slice(int(rows[0]), int(rows[-1]) + 1),
        slice(int(columns[0]), int(columns[-1]) + 1),
    )


def _overlay_labels(
    rgb: np.ndarray,
    labels: np.ndarray,
    colors: Mapping[int, np.ndarray],
    *,
    alpha: float = 0.68,
) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    for label, color in colors.items():
        selected = labels == label
        output[selected] = (1.0 - alpha) * output[selected] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)


def _titled_panel(array: np.ndarray, title: str) -> Image.Image:
    header = 25
    panel = Image.new("RGB", (array.shape[1], array.shape[0] + header), "white")
    panel.paste(Image.fromarray(array), (0, header))
    ImageDraw.Draw(panel).text((6, 5), title, fill="black")
    return panel


def _atomic_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(path)
    try:
        image.save(temporary, format="PNG", optimize=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        publish_prepared_file_durable_exclusive(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def render_native_videomt_query_visuals(
    *,
    output_root: Path,
    global_step: int,
    input_weight_global_step: int,
    rank: int,
    normalized_padded_rgb: torch.Tensor,
    clip_targets: Sequence[Mapping[str, torch.Tensor]],
    identity_keys: Sequence[Sequence[str]],
    source_output: ExactVidEoMTOutput,
    sample_keys: Sequence[str],
) -> list[dict[str, object]]:
    """Render all-query geometry without adding a selector to the model graph."""

    if global_step <= 0 or input_weight_global_step != global_step - 1 or rank < 0:
        raise ValueError("native VidEoMT visual has an invalid optimizer boundary")
    if normalized_padded_rgb.ndim != 5 or normalized_padded_rgb.shape[1] != 5:
        raise ValueError("native VidEoMT visual requires one five-frame source batch")
    batch = normalized_padded_rgb.shape[0]
    if not (
        len(clip_targets) == len(identity_keys) == len(sample_keys) == batch
        and source_output.class_logits.shape[:2] == (batch, 1)
        and source_output.mask_logits.shape[:3] == (batch, 200, 1)
    ):
        raise ValueError("native VidEoMT visual batch axes differ")

    object_probability = 1.0 - source_output.class_logits[:, 0].softmax(dim=-1)[..., -1]
    artifacts: list[dict[str, object]] = []
    for batch_index, sample_key in enumerate(sample_keys):
        target = clip_targets[batch_index]
        masks = target["masks"][:, 0].detach().bool().cpu()
        valid = target["valid_pixels"][0].detach().bool().cpu()
        row_slice, column_slice = _measured_bounds(valid)
        rgb = _inverse_normalized_rgb(normalized_padded_rgb[batch_index, 0])[
            row_slice,
            column_slice,
        ]
        height, width = valid.shape
        probabilities = F.interpolate(
            source_output.mask_logits[batch_index, :, 0].detach().float().unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).sigmoid()
        scores = probabilities * object_probability[batch_index].detach().float().unsqueeze(1).unsqueeze(2)
        scores[:, ~valid] = 0.0
        confidence, winner = scores.max(dim=0)
        predicted_labels = winner.cpu().numpy().astype(np.int64)
        predicted_labels[confidence.cpu().numpy() < _PRESENTATION_SCORE_THRESHOLD] = -1
        predicted_labels = predicted_labels[row_slice, column_slice]
        visible_queries = tuple(
            int(value)
            for value in np.unique(predicted_labels)
            if int(value) >= 0
        )
        prediction_colors = {query: _query_color(query) for query in visible_queries}
        prediction_overlay = _overlay_labels(rgb, predicted_labels, prediction_colors)

        truth_labels = np.full((height, width), -1, dtype=np.int64)
        for identity_index, mask in enumerate(masks):
            truth_labels[mask.numpy()] = identity_index
        truth_labels = truth_labels[row_slice, column_slice]
        truth_colors = {
            index: _query_color(index) for index in range(len(identity_keys[batch_index]))
        }
        truth_overlay = _overlay_labels(rgb, truth_labels, truth_colors)

        panels = (
            _titled_panel(rgb, "Augmented source RGB (current only)"),
            _titled_panel(truth_overlay, "Loss-only physical masks"),
            _titled_panel(prediction_overlay, "All-query prediction (query-ID colors)"),
        )
        header_height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), header_height + 92), "white")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        draw = ImageDraw.Draw(canvas)
        top_queries = object_probability[batch_index].detach().float().topk(_TOP_QUERY_PANELS)
        legend_y = header_height + 5
        draw.text(
            (6, legend_y),
            f"sample={sample_key}  display threshold={_PRESENTATION_SCORE_THRESHOLD:.2f} "
            "(visualization only; never used by training)",
            fill="black",
        )
        draw.text(
            (6, legend_y + 20),
            "top P(object): "
            + "  ".join(
                f"q{int(query):03d}={float(probability):.3f}"
                for probability, query in zip(
                    top_queries.values.cpu(),
                    top_queries.indices.cpu(),
                    strict=True,
                )
            ),
            fill="black",
        )
        draw.text(
            (6, legend_y + 40),
            "physical identities: " + ", ".join(identity_keys[batch_index]),
            fill="black",
        )
        draw.text(
            (6, legend_y + 60),
            "visible query IDs: " + ", ".join(f"q{value:03d}" for value in visible_queries),
            fill="black",
        )

        path = (
            output_root
            / "native_videomt_query_visuals"
            / f"step_{global_step:08d}"
            / f"rank_{rank}"
            / f"sample_{batch_index:02d}_{_slug(sample_key)}.png"
        )
        _atomic_png(canvas, path)
        artifacts.append(
            {
                "schema": NATIVE_VIDEOMT_QUERY_VISUAL_SCHEMA,
                "path": str(path),
                "sha256": _sha256(path),
                "sample_key": sample_key,
                "batch_index": batch_index,
                "global_step": global_step,
                "input_weight_global_step": input_weight_global_step,
                "query_count": 200,
                "visible_query_ids": list(visible_queries),
                "physical_identity_keys": list(identity_keys[batch_index]),
                "presentation_score_threshold": _PRESENTATION_SCORE_THRESHOLD,
                "selection_used_by_training": False,
            }
        )
    return artifacts
