#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root=$1
repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
source_checkout=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr172-callback-v1}
runtime_hotfix=${PICF_LINGBOT_RUNTIME_HOTFIX:-$repository_root/references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch}
stage_checkpoint=${PICF_ADR172_STAGE_CHECKPOINT:-/mnt/picf-next/checkpoints/adr172-g2b-callback-compatible-2b1b5da-v1}
g2_report=${PICF_ADR172_G2_REPORT:-$stage_checkpoint/ltop_g2_representation_report.json}
execution_contract=${PICF_ADR172_EXECUTION_CONTRACT:-/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json}
offline_labels=${PICF_ADR172_OFFLINE_LABELS:-/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json}
trial_timeout_seconds=${PICF_ADR172_LBOT_TIMEOUT_SECONDS:-10800}

[[ "$repository_root" == /mnt/* && "$source_checkout" == /mnt/* ]] || {
  echo "ADR172 matched LBOT source trees must live under /mnt" >&2
  exit 1
}
[[ "$run_root" == /mnt/* && ! -e "$run_root" && ! -L "$run_root" ]] || {
  echo "ADR172 matched LBOT output must be one absent path under /mnt" >&2
  exit 1
}
[[ -z "$(git -C "$repository_root" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR172 matched LBOT requires an exact clean PICF checkout" >&2
  exit 1
}
[[ "$trial_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo "PICF_ADR172_LBOT_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
}

required=(
  "$python_bin"
  "$repository_root/tools/run_lingbot_vla2_official_lbot.py"
  "$repository_root/references/patches/lingbot_vla2_picf_native.patch"
  "$runtime_hotfix"
  "$source_checkout"
  "$source_checkout/configs/vla/robotwin/robotwin.yaml"
  "/mnt/picf-next/models/lingbot-vla-v2-6b"
  "/mnt/picf-next/models/qwen3-vl-4b-instruct"
  "$stage_checkpoint"
  "$g2_report"
  "/mnt/calvin_data/task_ABC_D/training"
  "/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json"
  "/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json"
  "$execution_contract"
  "$offline_labels"
)
for path in "${required[@]}"; do
  [[ -e "$path" && ! -L "$path" ]] || {
    echo "required ADR172 matched LBOT artifact is absent or indirect: $path" >&2
    exit 1
  }
done

mkdir -p "$run_root"
cd "$repository_root"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec timeout --signal=TERM --kill-after=60s "$trial_timeout_seconds" \
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/run_lingbot_vla2_official_lbot.py \
  --adr172-exact-stream \
  --source-checkout "$source_checkout" \
  --patch "$repository_root/references/patches/lingbot_vla2_picf_native.patch" \
  --runtime-hotfix "$runtime_hotfix" \
  --training-config "$source_checkout/configs/vla/robotwin/robotwin.yaml" \
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b \
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct \
  --stage-checkpoint "$stage_checkpoint" \
  --g2-report "$g2_report" \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json \
  --norm-stats /mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json \
  --execution-contract "$execution_contract" \
  --offline-labels "$offline_labels" \
  --run-dir "$run_root" \
  --output "$run_root/adr172_matched_lbot_2gpu_256.json" \
  --maximum-control-tokens 8 \
  --evaluation-steps 0,32,64,96,128,160,192,224,256 \
  --steps 256 \
  --seed 20260813 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator expandable-segments
