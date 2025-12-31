#!/usr/bin/env python3
"""sonata_weight_gen.py

下载 SpatialLM1.1‑Qwen‑0.5B 的 safetensors，提取 point_backbone(=Sonata) 权重，
保存为一个“纯 state_dict”的 .pth 文件。

默认输出（与 OpenPI cache 结构对齐，便于 pi0_pytorch 自动 resolve）：
  ~/.cache/openpi/openpi-assets/checkpoints/SpatialLM_Sonata_encoder.pth

你可以通过以下方式覆盖输出位置：
  - 命令行：--out /abs/path/to/SpatialLM_Sonata_encoder.pth
  - 或设置 OPENPI_DATA_HOME=/abs/path/to/openpi_cache （输出会落在该目录下）

用法示例：
  cd ~/Documents/openpi
  uv run python scripts/sonata_weight_gen.py

如果你想把 HuggingFace 的 2.3GB safetensors 缓存挪到别处（避免污染默认 ~/.cache/huggingface）：
  HF_HOME=/some/dir uv run python scripts/sonata_weight_gen.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open


DEFAULT_REPO_ID = "manycore-research/SpatialLM1.1-Qwen-0.5B"
DEFAULT_FILE_NAME = "model.safetensors"
DEFAULT_PREFIX = "point_backbone."
DEFAULT_OUT_REL = Path("openpi-assets") / "checkpoints" / "SpatialLM_Sonata_encoder.pth"


def get_openpi_data_home() -> Path:
    env = os.environ.get("OPENPI_DATA_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "openpi"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SpatialLM point_backbone weights into a standalone Sonata state_dict .pth"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace repo id")
    parser.add_argument("--filename", default=DEFAULT_FILE_NAME, help="Filename in the repo (safetensors)")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Key prefix to extract from model.safetensors")
    parser.add_argument(
        "--out",
        default=str(get_openpi_data_home() / DEFAULT_OUT_REL),
        help="Output .pth path (default: under OPENPI_DATA_HOME or ~/.cache/openpi)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not args.force:
        print(f"[skip] exists: {out_path}")
        print("Use --force to overwrite.")
        return

    print(f"Downloading {args.filename!r} from {args.repo_id!r} ...")
    hf_file = hf_hub_download(repo_id=args.repo_id, filename=args.filename)

    prefix = str(args.prefix)
    if prefix and not prefix.endswith("."):
        # Allow passing "point_backbone" -> "point_backbone."
        prefix = prefix + "."

    print(f"Extracting tensors with prefix {prefix!r} ...")
    sonata: dict[str, torch.Tensor] = {}
    with safe_open(hf_file, framework="pt", device="cpu") as sf:
        for key in sf.keys():
            if key.startswith(prefix):
                sonata[key[len(prefix) :]] = sf.get_tensor(key)

    if not sonata:
        raise RuntimeError(
            f"No tensors extracted. Check --prefix. "
            f"Example: --prefix point_backbone. (got {args.prefix!r})"
        )

    print(f"✓ {len(sonata)} tensors extracted.")
    torch.save(sonata, out_path)
    print(f"✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
