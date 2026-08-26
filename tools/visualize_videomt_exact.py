#!/usr/bin/env python3
"""Render released VidEoMT zero-shot outputs without changing its model graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from picf_next.videomt_exact.preprocessing import (
    official_topk_query_classes,
    official_track_scores,
    prepare_rgb_frames,
    resize_query_masks_to_original,
    unique_query_topk,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt

YTVIS_2019_CLASSES = (
    "person", "giant_panda", "lizard", "parrot", "skateboard", "sedan", "ape", "dog",
    "snake", "monkey", "hand", "rabbit", "duck", "cat", "cow", "fish", "train",
    "horse", "turtle", "bear", "motorbike", "giraffe", "leopard", "fox", "deer",
    "owl", "surfboard", "airplane", "truck", "zebra", "tiger", "elephant",
    "snowboard", "boat", "shark", "mouse", "frog", "eagle", "earless_seal",
    "tennis_racket",
)
PALETTE = (
    (230, 57, 70), (38, 166, 154), (244, 162, 47), (69, 123, 157),
    (156, 93, 179), (42, 157, 143), (239, 113, 72), (81, 130, 187),
    (217, 95, 14), (107, 142, 35), (185, 78, 72), (76, 114, 176),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument(
        "--golden-manifest",
        type=Path,
        help="optional CALVIN golden manifest for evaluation-only per-instance oracle coverage",
    )
    parser.add_argument("--title", default="VidEoMT-DINOv3-L zero-shot diagnostic")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--unique-queries", type=int, default=12)
    return parser.parse_args()


def _read_rgb(path: Path, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if crop is not None:
            rgb = rgb.crop(crop)
        return np.asarray(rgb, dtype=np.uint8).copy()


def _stable_calvin_color(key: str) -> np.ndarray:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=3).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.int64)
    return (64 + raw % 176).astype(np.uint8)


def _decode_calvin_golden_panel(
    panel_path: Path,
    manifest_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Recover audit-only instance masks from the committed golden overlay."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        (item for item in manifest["records"] if item["panel"] == panel_path.name),
        None,
    )
    if record is None:
        raise ValueError(f"golden manifest has no record for {panel_path.name}")
    with Image.open(panel_path) as image:
        panel = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if panel.shape[1] % 4 or panel.shape[0] <= 96:
        raise ValueError("CALVIN golden panel geometry is invalid")
    width = panel.shape[1] // 4
    source = panel[96:, :width].copy()
    instance_overlay = panel[96:, 2 * width : 3 * width].copy()
    valid = np.ones(source.shape[:2], dtype=bool)
    valid[:14] = False  # The panel renderer writes a label over each source pane.
    masks: dict[str, np.ndarray] = {}
    for instance in record["visible_instances"]:
        key = str(instance["key"])
        color = _stable_calvin_color(key).astype(np.float32)
        expected = np.clip(0.45 * source.astype(np.float32) + 0.55 * color, 0, 255).astype(
            np.uint8
        )
        exact = np.all(instance_overlay == expected, axis=-1) & valid
        masks[key] = exact
    return source, masks, record


def _overlay_tile(
    frame: np.ndarray,
    mask_logit: torch.Tensor,
    *,
    color: tuple[int, int, int],
    label: str,
    tile_size: int = 288,
) -> Image.Image:
    probability = mask_logit.float().sigmoid().cpu().numpy()
    selected = probability > 0.5
    source = Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    selected_image = Image.fromarray((selected * 255).astype(np.uint8)).resize(
        (tile_size, tile_size), Image.Resampling.NEAREST
    )
    alpha = np.asarray(selected_image, dtype=np.uint8)
    overlay = Image.new("RGBA", source.size, color + (0,))
    overlay.putalpha(Image.fromarray((alpha.astype(np.float32) * 0.48).astype(np.uint8)))
    canvas = Image.new("RGB", (tile_size, tile_size + 54), (20, 20, 20))
    canvas.paste(Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB"), (0, 54))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(245, 245, 245), font=ImageFont.load_default())
    area = float(selected.mean())
    draw.text((8, 29), f"mask>0 area={area:.3f}", fill=(190, 190, 190), font=ImageFont.load_default())
    return canvas


def _source_tile(frame: np.ndarray, *, tile_size: int = 288) -> Image.Image:
    source = Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (tile_size, tile_size + 54), (20, 20, 20))
    canvas.paste(source, (0, 54))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "source RGB", fill=(245, 245, 245), font=ImageFont.load_default())
    draw.text((8, 29), "no ground truth shown", fill=(190, 190, 190), font=ImageFont.load_default())
    return canvas


def _contact_sheet(tiles: list[Image.Image], title: str, destination: Path) -> None:
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    tile_width, tile_height = tiles[0].size
    header = 94
    canvas = Image.new("RGB", (columns * tile_width, header + rows * tile_height), (10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text(
        (12, 35),
        "Official YTVIS-2019 weights; CALVIN is out-of-domain. This is a diagnostic, not a PICF result.",
        fill=(225, 180, 80),
        font=ImageFont.load_default(),
    )
    draw.text(
        (12, 58),
        "Masks are thresholded at the official logit>0 rule; query ranking is stated in each file.",
        fill=(190, 190, 190),
        font=ImageFont.load_default(),
    )
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * tile_width, header + (index // columns) * tile_height))
    canvas.save(destination)


def _partition_image(
    frame: np.ndarray,
    masks: torch.Tensor,
    scores: torch.Tensor,
    query_indices: torch.Tensor,
    destination: Path,
) -> None:
    selected_masks = masks[query_indices, 0].float()
    weighted = selected_masks.sigmoid() * scores[:, None, None].float()
    values, owners = weighted.max(dim=0)
    foreground = selected_masks.max(dim=0).values > 0
    rgb = frame.astype(np.float32)
    result = rgb.copy()
    for index, color in enumerate(PALETTE[: len(query_indices)]):
        region = foreground.cpu().numpy() & (owners.cpu().numpy() == index)
        result[region] = 0.35 * result[region] + 0.65 * np.asarray(color, dtype=np.float32)
    confidence = values.cpu().numpy()
    result[~foreground.cpu().numpy()] *= 0.65
    out = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, min(out.width, 455), 34), fill=(0, 0, 0))
    draw.text((6, 6), f"presentation-only query partition; max confidence={confidence.max():.3f}", fill="white")
    out.save(destination)


def _oracle_tile(
    frame: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    label: str,
    tile_size: int = 288,
) -> Image.Image:
    source = np.asarray(
        Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    pred = np.asarray(
        Image.fromarray((prediction * 255).astype(np.uint8)).resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
    ) > 0
    gt = np.asarray(
        Image.fromarray((target * 255).astype(np.uint8)).resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
    ) > 0
    result = source.copy()
    result[pred & ~gt] = 0.35 * result[pred & ~gt] + 0.65 * np.array((230, 57, 70))
    result[gt & ~pred] = 0.35 * result[gt & ~pred] + 0.65 * np.array((38, 166, 154))
    result[gt & pred] = 0.25 * result[gt & pred] + 0.75 * np.array((244, 200, 55))
    canvas = Image.new("RGB", (tile_size, tile_size + 54), (20, 20, 20))
    canvas.paste(Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)), (0, 54))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(245, 245, 245), font=ImageFont.load_default())
    draw.text(
        (8, 29),
        "yellow=overlap, green=GT-only, red=prediction-only",
        fill=(190, 190, 190),
        font=ImageFont.load_default(),
    )
    return canvas


def _oracle_object_coverage(
    frame: np.ndarray,
    masks: torch.Tensor,
    class_logits: torch.Tensor,
    targets: dict[str, np.ndarray],
    destination: Path,
    title: str,
) -> list[dict[str, object]]:
    probabilities = masks[:, 0].float().sigmoid()
    official_predictions = probabilities > 0.5
    flat_official_predictions = official_predictions.flatten(1)
    probability_thresholds = torch.tensor(
        (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
        dtype=probabilities.dtype,
    )
    threshold_predictions = (
        probabilities[:, None] > probability_thresholds[None, :, None, None]
    ).flatten(2)
    flat_probabilities = probabilities.flatten(1)
    track_scores = official_track_scores(class_logits)
    best_class_scores, best_classes = track_scores.max(dim=-1)
    rows: list[dict[str, object]] = []
    tiles = [_source_tile(frame)]
    for key, target_array in targets.items():
        target = torch.from_numpy(target_array).to(dtype=torch.bool)
        target_flat = target.flatten()
        official_intersection = (
            flat_official_predictions & target_flat[None]
        ).sum(dim=-1).float()
        official_union = (
            flat_official_predictions | target_flat[None]
        ).sum(dim=-1).float().clamp_min(1)
        official_iou = official_intersection / official_union
        best_official_iou, best_official_query = official_iou.max(dim=0)

        threshold_intersection = (
            threshold_predictions & target_flat[None, None]
        ).sum(dim=-1).float()
        threshold_union = (
            threshold_predictions | target_flat[None, None]
        ).sum(dim=-1).float().clamp_min(1)
        threshold_iou = threshold_intersection / threshold_union
        best_threshold_flat = int(threshold_iou.argmax())
        threshold_count = threshold_iou.shape[1]
        query = best_threshold_flat // threshold_count
        threshold_index = best_threshold_flat % threshold_count
        best_threshold_iou = threshold_iou[query, threshold_index]
        best_threshold = probability_thresholds[threshold_index]

        soft_intersection = (flat_probabilities * target_flat[None]).sum(dim=-1)
        soft_union = (
            flat_probabilities.sum(dim=-1) + target_flat.sum() - soft_intersection
        ).clamp_min(1e-8)
        soft_iou = soft_intersection / soft_union
        best_soft_iou, best_soft_query = soft_iou.max(dim=0)

        target_pixels = int(target.sum())
        oracle_prediction = probabilities[query] > best_threshold
        predicted_pixels = int(oracle_prediction.sum())
        true_positive = int(threshold_intersection[query, threshold_index])
        recall = true_positive / max(target_pixels, 1)
        precision = true_positive / max(predicted_pixels, 1)
        category = int(best_classes[query])
        row = {
            "identity_key": key,
            "recovered_target_pixels": target_pixels,
            "best_official_threshold_query": int(best_official_query),
            "best_official_hard_iou_at_logit_gt_zero": float(best_official_iou),
            "best_oracle_threshold_query": query,
            "best_oracle_probability_threshold": float(best_threshold),
            "best_oracle_threshold_iou": float(best_threshold_iou),
            "oracle_threshold_precision": precision,
            "oracle_threshold_recall": recall,
            "best_soft_iou_query": int(best_soft_query),
            "best_soft_iou": float(best_soft_iou),
            "query_best_ytvis_class": YTVIS_2019_CLASSES[category],
            "query_best_ytvis_score": float(best_class_scores[query]),
        }
        rows.append(row)
        tiles.append(
            _oracle_tile(
                frame,
                oracle_prediction.cpu().numpy(),
                target_array,
                label=(
                    f"{key} | q{query} IoU={float(best_threshold_iou):.3f} "
                    f"thr={float(best_threshold):.2f} P={precision:.3f} R={recall:.3f}"
                ),
            )
        )
    _contact_sheet(
        tiles,
        title + " | evaluation-only oracle over all 200 queries",
        destination,
    )
    return rows


def main() -> None:
    args = parse_args()
    if args.unique_queries <= 0 or args.unique_queries > len(PALETTE):
        raise ValueError(f"--unique-queries must be in [1, {len(PALETTE)}]")
    crop = tuple(args.crop) if args.crop is not None else None
    oracle_targets: dict[str, np.ndarray] | None = None
    golden_record: dict[str, object] | None = None
    if args.golden_manifest is not None:
        if len(args.input) != 1 or crop is not None:
            raise ValueError("golden-panel evaluation requires one uncropped --input")
        golden_frame, oracle_targets, golden_record = _decode_calvin_golden_panel(
            args.input[0], args.golden_manifest
        )
        frames = [golden_frame]
    else:
        frames = [_read_rgb(path, crop) for path in args.input]
    prepared = prepare_rgb_frames(frames)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    runtime = load_exact_videomt(
        ExactVidEoMTConfig(
            checkpoint_path=args.checkpoint,
            local_dinov3_bundle=args.dinov3_bundle,
            num_frames=len(frames),
        ),
        device=args.device,
        dtype=dtype,
    )
    with torch.inference_mode():
        output = runtime(prepared.model_input.to(device=args.device, dtype=dtype))
    class_logits = output.class_logits[0].float().cpu()
    masks = resize_query_masks_to_original(
        output.mask_logits[0].float().cpu(),
        padded_size=prepared.padded_size,
        resized_size=prepared.resized_sizes[0],
        original_size=prepared.original_sizes[0],
    )
    official = official_topk_query_classes(class_logits, topk=10)
    unique = unique_query_topk(class_logits, topk=args.unique_queries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    official_tiles = [_source_tile(frames[0])]
    official_rows = []
    for rank, (score, query, category) in enumerate(zip(*official), start=1):
        query_id, category_id = int(query), int(category)
        label = f"official rank {rank}: q{query_id} {YTVIS_2019_CLASSES[category_id]} score={float(score):.3f}"
        official_tiles.append(
            _overlay_tile(frames[0], masks[query_id, 0], color=PALETTE[(rank - 1) % len(PALETTE)], label=label)
        )
        official_rows.append({"rank": rank, "query": query_id, "class": category_id, "class_name": YTVIS_2019_CLASSES[category_id], "score": float(score)})

    unique_tiles = [_source_tile(frames[0])]
    unique_rows = []
    for rank, (score, query, category) in enumerate(zip(*unique), start=1):
        query_id, category_id = int(query), int(category)
        label = f"unique q rank {rank}: q{query_id} {YTVIS_2019_CLASSES[category_id]} score={float(score):.3f}"
        unique_tiles.append(
            _overlay_tile(frames[0], masks[query_id, 0], color=PALETTE[(rank - 1) % len(PALETTE)], label=label)
        )
        unique_rows.append({"rank": rank, "query": query_id, "class": category_id, "class_name": YTVIS_2019_CLASSES[category_id], "score": float(score)})

    _contact_sheet(official_tiles, args.title + " | exact official top-10 query-class pairs", args.output_dir / "official_top10.png")
    _contact_sheet(unique_tiles, args.title + " | presentation-only unique-query ranking", args.output_dir / "unique_queries.png")
    _partition_image(frames[0], masks, unique[0], unique[1], args.output_dir / "query_partition.png")
    oracle_rows = None
    if oracle_targets is not None:
        oracle_rows = _oracle_object_coverage(
            frames[0],
            masks,
            class_logits,
            oracle_targets,
            args.output_dir / "oracle_object_coverage.png",
            args.title,
        )
    receipt = {
        "schema": "picf-next.videomt-exact-zero-shot-visual.v1",
        "claim_scope": "out-of-domain visualization only; not a CALVIN or PICF acceptance result",
        "inputs": [str(path.resolve()) for path in args.input],
        "crop_ltrb": crop,
        "model_input_shape": list(prepared.model_input.shape),
        "original_sizes": prepared.original_sizes,
        "resized_sizes": prepared.resized_sizes,
        "padded_size": prepared.padded_size,
        "dtype": str(dtype),
        "official_top10": official_rows,
        "presentation_unique_queries": unique_rows,
        "golden_record": golden_record,
        "evaluation_only_oracle_object_coverage": oracle_rows,
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
