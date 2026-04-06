#!/usr/bin/env python3
"""Download official V-JEPA 2.1 checkpoints into a stable local layout.

Default layout:
  checkpoints/foundation/vjepa2_1/
    manifest.json
    vjepa2_1_vit_base_384/
      vjepa2_1_vitb_dist_vitG_384.pt
    vjepa2_1_vit_large_384/
      vjepa2_1_vitl_dist_vitG_384.pt
    vjepa2_1_vit_giant_384/
      vjepa2_1_vitg_384.pt
    vjepa2_1_vit_gigantic_384/
      vjepa2_1_vitG_384.pt

Examples:
  uv run python scripts/vjepa_ckpt_fetch.py --model vjepa2_1_vit_base_384
  uv run python scripts/vjepa_ckpt_fetch.py --all
  uv run python scripts/vjepa_ckpt_fetch.py --model vjepa2_1_vit_base_384 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request

BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
DEFAULT_OUT_ROOT = Path("checkpoints") / "foundation" / "vjepa2_1"

MODEL_SPECS = {
    "vjepa2_1_vit_base_384": {
        "filename": "vjepa2_1_vitb_dist_vitG_384.pt",
        "description": "V-JEPA 2.1 ViT-B/16 384",
        "checkpoint_key": "ema_encoder",
    },
    "vjepa2_1_vit_large_384": {
        "filename": "vjepa2_1_vitl_dist_vitG_384.pt",
        "description": "V-JEPA 2.1 ViT-L/16 384",
        "checkpoint_key": "ema_encoder",
    },
    "vjepa2_1_vit_giant_384": {
        "filename": "vjepa2_1_vitg_384.pt",
        "description": "V-JEPA 2.1 ViT-g/16 384",
        "checkpoint_key": "target_encoder",
    },
    "vjepa2_1_vit_gigantic_384": {
        "filename": "vjepa2_1_vitG_384.pt",
        "description": "V-JEPA 2.1 ViT-G/16 384",
        "checkpoint_key": "target_encoder",
    },
}


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_manifest(out_root: Path) -> Path:
    manifest = {
        "base_url": BASE_URL,
        "models": {
            model_name: {
                **spec,
                "url": f"{BASE_URL}/{spec['filename']}",
                "relative_path": str(Path(model_name) / spec["filename"]),
            }
            for model_name, spec in MODEL_SPECS.items()
        },
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _resolve_targets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(MODEL_SPECS)
    if args.model is None:
        raise SystemExit("Specify --model <name> or --all.")
    return [args.model]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download V-JEPA 2.1 checkpoints into checkpoints/foundation.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default=None)
    parser.add_argument("--all", action="store_true", help="Download all V-JEPA 2.1 checkpoints.")
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="Root directory for downloaded checkpoints.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoint files.")
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    targets = _resolve_targets(args)
    manifest_path = _write_manifest(out_root)
    print(f"[info] manifest: {manifest_path}")

    for model_name in targets:
        spec = MODEL_SPECS[model_name]
        url = f"{BASE_URL}/{spec['filename']}"
        dst = out_root / model_name / spec["filename"]
        if dst.exists() and not args.force:
            print(f"[skip] {model_name}: exists at {dst}")
            continue
        print(f"[download] {model_name}")
        print(f"  url: {url}")
        print(f"  dst: {dst}")
        _download_file(url, dst)
        print(f"[done] {model_name}: {dst}")

    print()
    print("[next]")
    print("Use the checkpoint path with PICF visual scripts, for example:")
    example_model = targets[0]
    example_file = MODEL_SPECS[example_model]["filename"]
    example_ckpt = out_root / example_model / example_file
    print(
        "UV_CACHE_DIR=/tmp/uvcache uv run --no-sync python "
        "scripts/posterior/posterior_visual_full_check.py "
        f"--repo-root . --calvin-root <CALVIN_ROOT_OR_ZIP> --backend zip --segments 2 --max-points 256 "
        f"--checkpoint-path {example_ckpt}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
