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
trial_timeout_seconds=${PICF_ADR172_TRIAL_TIMEOUT_SECONDS:-10800}
direct_posterior_head_scope=${PICF_ADR172_DIRECT_POSTERIOR_HEAD_SCOPE:-all-action-heads}
case "$direct_posterior_head_scope" in
  all-action-heads) direct_grounding_weight=1.0 ;;
  guidedvla-fixed-object-heads-0-1) direct_grounding_weight=0.001 ;;
  *)
    echo "unknown ADR172 direct-posterior head scope: $direct_posterior_head_scope" >&2
    exit 2
    ;;
esac

[[ "$repository_root" == /mnt/* && "$source_checkout" == /mnt/* ]] || {
  echo "ADR172 trial source trees must live under /mnt" >&2
  exit 1
}
[[ "$run_root" == /mnt/* && ! -e "$run_root" && ! -L "$run_root" ]] || {
  echo "ADR172 trial output must be one absent direct path under /mnt" >&2
  exit 1
}
[[ -z "$(git -C "$repository_root" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR172 trial requires an exact clean PICF checkout" >&2
  exit 1
}
[[ "$trial_timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo "PICF_ADR172_TRIAL_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
}

required=(
  "$repository_root/tools/run_lingbot_vla2_ltop_adr172_direct_posterior.py"
  "$repository_root/src/picf_next/lingbot_native/action_posterior_collector.py"
  "$repository_root/src/picf_next/lingbot_native/action_posterior_learning.py"
  "$repository_root/src/picf_next/lingbot_native/action_posterior_receipt.py"
  "$source_checkout"
  "$repository_root/references/patches/lingbot_vla2_picf_native.patch"
  "$runtime_hotfix"
  "/mnt/picf-next/models/lingbot-vla-v2-6b"
  "/mnt/picf-next/models/qwen3-vl-4b-instruct"
  "$stage_checkpoint"
  "$g2_report"
  "/mnt/calvin_data/task_ABC_D/training"
  "/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json"
  "/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json"
  "/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z"
  "/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json"
  "/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json"
  "/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json"
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || {
    echo "required ADR172 trial artifact is absent: $path" >&2
    exit 1
  }
done

mkdir -p "$run_root"
cd "$repository_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec timeout --signal=TERM --kill-after=60s "$trial_timeout_seconds" \
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/run_lingbot_vla2_ltop_adr172_direct_posterior.py \
  --source-checkout "$source_checkout" \
  --patch "$repository_root/references/patches/lingbot_vla2_picf_native.patch" \
  --runtime-hotfix "$runtime_hotfix" \
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b \
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct \
  --stage-checkpoint "$stage_checkpoint" \
  --g2-report "$g2_report" \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json \
  --norm-stats /mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest /mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json \
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d \
  --execution-contract /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json \
  --offline-labels /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json \
  --output "$run_root/adr172_direct_posterior_trial_training_report.json" \
  --progress-output "$run_root/progress.json" \
  --journal-dir "$run_root/rank_journal" \
  --checkpoint-output "$run_root/checkpoint" \
  --mode direct-trial \
  --phase training \
  --steps 256 \
  --eval-every 32 \
  --progress-every 8 \
  --official-loss-weight 1.0 \
  --physical-set-weight 1.0 \
  --direct-grounding-weight "$direct_grounding_weight" \
  --direct-posterior-head-scope "$direct_posterior_head_scope" \
  --cuda-allocator expandable-segments
