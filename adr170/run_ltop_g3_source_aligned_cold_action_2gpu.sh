#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TRAINING_RUN_ROOT EVALUATION_RUN_ROOT" >&2
  exit 2
fi

training_root=$1
evaluation_root=$2
repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
source_checkout=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr162-muon-align-dbed72e-v1}
runtime_hotfix=${PICF_LINGBOT_RUNTIME_HOTFIX:-$repository_root/references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch}
evaluation_timeout_seconds=${PICF_G3_EVALUATION_TIMEOUT_SECONDS:-}
evaluation_scenes_per_partition=${PICF_G3_EVALUATION_SCENES_PER_PARTITION:-1}
evaluation_action_information_set=${PICF_G3_EVALUATION_ACTION_INFORMATION_SET:-factual}
training_report=$training_root/ltop_g3_source_aligned_trial_training_report.json

case "$evaluation_scenes_per_partition" in
  1)
    evaluation_scope=quick
    default_evaluation_timeout_seconds=3600
    ;;
  4)
    evaluation_scope=full
    default_evaluation_timeout_seconds=14400
    ;;
  *)
    echo "PICF_G3_EVALUATION_SCENES_PER_PARTITION must be 1 or 4" >&2
    exit 1
    ;;
esac
evaluation_timeout_seconds=${evaluation_timeout_seconds:-$default_evaluation_timeout_seconds}
case "$evaluation_action_information_set" in
  factual|mediator-required) ;;
  *)
    echo "PICF_G3_EVALUATION_ACTION_INFORMATION_SET must be factual or mediator-required" >&2
    exit 1
    ;;
esac
evaluation_report=$evaluation_root/ltop_g3_source_aligned_cold_action_${evaluation_scope}_report.json

if [[ "$repository_root" != /mnt/* || "$training_root" != /mnt/* ]]; then
  echo "ADR170 action source and training evidence must live under /mnt" >&2
  exit 1
fi
if [[ "$evaluation_root" != /mnt/* || -e "$evaluation_root" || -L "$evaluation_root" ]]; then
  echo "ADR170 action output must be one absent direct path under /mnt" >&2
  exit 1
fi
if [[ ! "$evaluation_timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "PICF_G3_EVALUATION_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi
for path in \
  "$python_bin" \
  "$training_report" \
  "$source_checkout" \
  "$repository_root/adr170/source_aligned_cold_evaluation_contract.sh" \
  "$repository_root/tools/run_lingbot_vla2_ltop_g3_action_mediation.py" \
  "$repository_root/src/picf_next/lingbot_native/task_action_supervision.py" \
  "$repository_root/references/patches/lingbot_vla2_picf_native.patch" \
  "$runtime_hotfix" \
  /mnt/picf-next/models/lingbot-vla-v2-6b \
  /mnt/picf-next/models/qwen3-vl-4b-instruct \
  /mnt/picf-next/checkpoints/adr160-g2b-confirm-49eac80-v3 \
  /mnt/picf-next/runs/adr160-g2b-confirm-49eac80-v3/ltop_g2_representation_report.json \
  /mnt/calvin_data/task_ABC_D/training \
  /mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json \
  /mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json \
  /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  /mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json \
  /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json \
  /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json; do
  if [[ ! -e "$path" ]]; then
    echo "required ADR170 action artifact is absent: $path" >&2
    exit 1
  fi
done
if [[ $(git -C "$repository_root" status --porcelain=v1 --untracked-files=all) ]]; then
  echo "ADR170 action evaluation requires an exact clean source checkout" >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
  echo "ADR170 action evaluation requires exactly two visible GPUs" >&2
  exit 1
fi

# shellcheck source=adr170/source_aligned_cold_evaluation_contract.sh
source "$repository_root/adr170/source_aligned_cold_evaluation_contract.sh"
trained_checkpoint=$(adr170_resolve_source_aligned_trial_checkpoint "$python_bin" "$training_report")

mkdir -p "$evaluation_root"
cd "$repository_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

set +e
timeout --signal=TERM --kill-after=60s "$evaluation_timeout_seconds" \
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
  --output "$evaluation_report" \
  --trained-checkpoint "$trained_checkpoint" \
  --mode gate \
  --phase evaluation \
  --evaluation-action-information-set "$evaluation_action_information_set" \
  --steps 128 \
  --eval-every 32 \
  --evaluation-scenes-per-partition "$evaluation_scenes_per_partition" \
  --cuda-allocator expandable-segments
evaluation_status=$?
set -e
if [[ "$evaluation_status" -ne 0 ]]; then
  printf '{"schema":"picf-next.adr170-cold-action-runtime-failure.v1","status":"FAIL","exit_code":%d,"timeout_seconds":%d,"evaluation_scenes_per_partition":%d}\n' \
    "$evaluation_status" "$evaluation_timeout_seconds" "$evaluation_scenes_per_partition" \
    >"$evaluation_root/runtime_failure.json"
  exit "$evaluation_status"
fi

adr170_validate_cold_report \
  "$python_bin" "$evaluation_report" "$trained_checkpoint" evaluation \
  "$evaluation_scenes_per_partition" "$evaluation_action_information_set"
