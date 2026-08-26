#!/usr/bin/env bash
set -euo pipefail

output_root="${1:?usage: download_wla_sana_exact.sh OUTPUT_ROOT [BASE_URL]}"
base_url="${2:-https://huggingface.co/SJTU-DENG-Lab/Sana_600M_512px_diffusers_64channels/resolve/main}"

files=(
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors
  vae/config.json
  vae/diffusion_pytorch_model.safetensors
  scheduler/scheduler_config.json
)

mkdir -p "${output_root}"
for relative in "${files[@]}"; do
  destination="${output_root}/${relative}"
  mkdir -p "$(dirname "${destination}")"
  curl \
    --fail \
    --location \
    --retry 8 \
    --retry-all-errors \
    --retry-delay 2 \
    --continue-at - \
    --output "${destination}.partial" \
    "${base_url}/${relative}"
  mv "${destination}.partial" "${destination}"
done

(
  cd "${output_root}"
  sha256sum "${files[@]}" > SHA256SUMS
)
