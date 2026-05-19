#!/usr/bin/env python3
"""Build per-variant and combined GIFs from PICF anchor overlay PNGs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


_STEP_RE = re.compile(r"step_(\d{6}).*__(?P<variant>[^_/]+(?:_[^_/]+)*)\.png$")
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def _parse_variant(path: Path) -> tuple[int, str] | None:
    match = _STEP_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1)), str(match.group("variant"))


def _open_tile(path: Path | None, *, tile_width: int, label: str) -> Image.Image:
    if path is None:
        tile = Image.new("RGB", (tile_width, max(tile_width // 2, 1)), color=(24, 24, 24))
    else:
        img = Image.open(path).convert("RGB")
        if tile_width > 0 and img.width != tile_width:
            tile_height = max(int(round(img.height * (tile_width / max(img.width, 1)))), 1)
            img = img.resize((tile_width, tile_height), resample=_BILINEAR)
        tile = img
    draw = ImageDraw.Draw(tile)
    pad = 5
    text = label.replace("_", " ")
    bbox = draw.textbbox((pad, pad), text)
    bg = (0, 0, 0)
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return tile


def _build_combined_gif(
    grouped: dict[str, list[tuple[int, Path]]],
    *,
    variants: list[str],
    out_path: Path,
    duration_ms: int,
    max_frames: int,
    tile_width: int,
) -> bool:
    by_variant: dict[str, dict[int, Path]] = {
        variant: {step: path for step, path in items}
        for variant, items in grouped.items()
    }
    steps = sorted({step for variant in variants for step in by_variant.get(variant, {})})
    if max_frames > 0:
        steps = steps[-int(max_frames) :]
    if not steps:
        return False
    frames: list[Image.Image] = []
    for step in steps:
        tiles = [
            _open_tile(by_variant.get(variant, {}).get(step), tile_width=tile_width, label=f"{variant} step {step}")
            for variant in variants
        ]
        max_w = max(tile.width for tile in tiles)
        max_h = max(tile.height for tile in tiles)
        padded = []
        for tile in tiles:
            canvas = Image.new("RGB", (max_w, max_h), color=(24, 24, 24))
            canvas.paste(tile, ((max_w - tile.width) // 2, (max_h - tile.height) // 2))
            padded.append(canvas)
        cols = 3
        rows = 2
        frame = Image.new("RGB", (cols * max_w, rows * max_h), color=(24, 24, 24))
        for idx, tile in enumerate(padded[: cols * rows]):
            x = (idx % cols) * max_w
            y = (idx // cols) * max_h
            frame.paste(tile, (x, y))
        frames.append(frame.convert("P", palette=Image.ADAPTIVE))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(duration_ms), 20),
        loop=0,
        optimize=False,
    )
    print(f"combined: {len(frames)} frames -> {out_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--duration-ms", type=int, default=650)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--no-combined", action="store_true", help="Do not write the 2x3 combined overview GIF.")
    parser.add_argument("--combined-name", type=str, default="combined_6view.gif")
    parser.add_argument(
        "--variants",
        type=str,
        default="with_gray,active_only,sidecar_proposals,mask_only,mask_active,mask_with_gray",
        help="Comma-separated overlay variants to export.",
    )
    args = parser.parse_args()

    overlay_dir = args.overlay_dir.expanduser().resolve()
    out_dir = (args.out_dir or (overlay_dir / "gifs")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {part.strip() for part in str(args.variants).split(",") if part.strip()}
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(overlay_dir.glob("*.png")):
        parsed = _parse_variant(path)
        if parsed is None:
            continue
        step, variant = parsed
        if wanted and variant not in wanted:
            continue
        grouped[variant].append((step, path))

    wrote = 0
    for variant, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item[0])
        if args.max_frames > 0:
            items = items[-int(args.max_frames) :]
        if len(items) < 1:
            continue
        frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for _, path in items]
        out_path = out_dir / f"{variant}.gif"
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(int(args.duration_ms), 20),
            loop=0,
            optimize=False,
        )
        print(f"{variant}: {len(frames)} frames -> {out_path}")
        wrote += 1
    if not bool(args.no_combined):
        variants = [part.strip() for part in str(args.variants).split(",") if part.strip()]
        if _build_combined_gif(
            grouped,
            variants=variants,
            out_path=out_dir / str(args.combined_name),
            duration_ms=int(args.duration_ms),
            max_frames=int(args.max_frames),
            tile_width=int(args.tile_width),
        ):
            wrote += 1
    if wrote == 0:
        print(f"No overlay PNGs matched in {overlay_dir}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
