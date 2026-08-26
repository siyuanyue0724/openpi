#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ROOT" >&2
  exit 2
fi

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=$1
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
source_checkout=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr162-muon-align-dbed72e-v1}
runtime_hotfix=${PICF_LINGBOT_RUNTIME_HOTFIX:-$repository_root/references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch}
report=$run_root/ltop_g3_representation_retention_report.json

if [[ "$repository_root" != /mnt/* || "$run_root" != /mnt/* ]]; then
  echo "G3 retention source and output must live under /mnt" >&2
  exit 1
fi
if [[ -e "$run_root" ]]; then
  echo "G3 retention output already exists: $run_root" >&2
  exit 1
fi
for path in \
  "$python_bin" \
  "$source_checkout" \
  "$runtime_hotfix" \
  /mnt/picf-next/checkpoints/adr160-g2b-confirm-49eac80-v3 \
  /mnt/picf-next/checkpoints/adr163-g3-training-540b490-v1 \
  /mnt/picf-next/runs/adr160-g2b-confirm-49eac80-v3/ltop_g2_representation_report.json; do
  if [[ ! -e "$path" ]]; then
    echo "required G3 retention artifact is absent: $path" >&2
    exit 1
  fi
done
if [[ $(git -C "$repository_root" status --porcelain --untracked-files=no) ]]; then
  echo "G3 retention source checkout has tracked modifications" >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
  echo "G3 retention requires exactly two visible GPUs" >&2
  exit 1
fi

mkdir -p "$run_root"
cd "$repository_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/run_lingbot_vla2_ltop_g3_action_mediation.py \
  --source-checkout "$source_checkout" \
  --patch "$repository_root/references/patches/lingbot_vla2_picf_native.patch" \
  --runtime-hotfix "$runtime_hotfix" \
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b \
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct \
  --stage-checkpoint /mnt/picf-next/checkpoints/adr160-g2b-confirm-49eac80-v3 \
  --g2-report /mnt/picf-next/runs/adr160-g2b-confirm-49eac80-v3/ltop_g2_representation_report.json \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json \
  --norm-stats /mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest /mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json \
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d \
  --execution-contract /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json \
  --offline-labels /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json \
  --output "$report" \
  --trained-checkpoint /mnt/picf-next/checkpoints/adr163-g3-training-540b490-v1 \
  --mode gate \
  --phase retention \
  --steps 128 \
  --eval-every 32 \
  --cuda-allocator expandable-segments
