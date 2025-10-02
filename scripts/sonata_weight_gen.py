#!/usr/bin/env python3
"""
下载 SpatialLM1.1‑Qwen‑0.5B safetensors，
提取 point_backbone(=Sonata) 权重，
保存到 openpi/src/pretrain/SpatialLM_Sonata_encoder.pth
"""
import pathlib as pl
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open

REPO_ID, FILE_NAME = "manycore-research/SpatialLM1.1-Qwen-0.5B", "model.safetensors"
print(f"Downloading {FILE_NAME} from {REPO_ID} …")
hf_file = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, resume_download=True)

print("Extracting 'point_backbone.' tensors …")
sonata = {}
with safe_open(hf_file, framework="pt", device="cpu") as sf:
    for key in sf.keys():
        if key.startswith("point_backbone."):
            sonata[key[len("point_backbone."):]] = sf.get_tensor(key)
print(f"✓  {len(sonata)} tensors extracted.")

# —— 保存到 <repo>/src/pretrain/SpatialLM_Sonata_encoder.pth ——
repo_root = pl.Path(__file__).resolve().parents[1]       # …/openpi
pretrain_dir = repo_root / "src" / "pretrain"
pretrain_dir.mkdir(parents=True, exist_ok=True)

out_path = pretrain_dir / "SpatialLM_Sonata_encoder.pth"
torch.save(sonata, out_path)
print(f"✓  Saved → {out_path.relative_to(repo_root)}")
