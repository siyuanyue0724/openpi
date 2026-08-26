#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 {ltop-ec-factual|ltop-ec-blocked} RUN_DIR [smoke|pilot]" >&2
  exit 2
fi

arm=$1
run_dir=$2
mode=${3:-pilot}
case "$arm" in
  ltop-ec-factual|ltop-ec-blocked) ;;
  *)
    echo "unsupported LTOP core-pilot arm: $arm" >&2
    exit 2
    ;;
esac
case "$mode" in
  smoke|pilot) ;;
  *)
    echo "unsupported LTOP core-pilot mode: $mode" >&2
    exit 2
    ;;
esac

repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
source_checkout=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr162-muon-align-dbed72e-v1}
runtime_hotfix=${PICF_LINGBOT_RUNTIME_HOTFIX:-$repository_root/references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch}
contract_root=${PICF_CORE_PILOT_CONTRACT_ROOT:-/mnt/picf-next/adr161/contracts/calvin-two-gpu-2k-interleave8-v2}
g3_run_root=${PICF_G3_RUN_ROOT:-/mnt/picf-next/runs/adr160-g3-gate-6d0a411-v1}

required=(
  "$repository_root/tools/run_lingbot_vla2_ltop_core_pilot.py"
  "$source_checkout"
  "$repository_root/references/patches/lingbot_vla2_picf_native.patch"
  "$runtime_hotfix"
  "/mnt/picf-next/models/lingbot-vla-v2-6b"
  "/mnt/picf-next/models/qwen3-vl-4b-instruct"
  "/mnt/picf-next/checkpoints/adr160-g2b-confirm-49eac80-v3"
  "/mnt/picf-next/runs/adr160-g2b-confirm-49eac80-v3/ltop_g2_representation_report.json"
  "$g3_run_root/ltop_g3_action_mediation_report.json"
  "/mnt/calvin_data/task_ABC_D/training"
  "/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json"
  "/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json"
  "/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z"
  "/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json"
  "$contract_root/two-gpu-2k-i8.physical.stream-plan.json"
  "$contract_root/two-gpu-2k-i8.physical.split.json"
  "$contract_root/two-gpu-2k-i8.physical.evaluation-plan.json"
  "/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json"
  "/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "required LTOP core-pilot artifact is absent: $path" >&2
    exit 1
  fi
done
if [[ -e "$run_dir" ]]; then
  echo "LTOP core-pilot run directory already exists: $run_dir" >&2
  exit 1
fi
if [[ "$run_dir" != /mnt/* ]]; then
  echo "LTOP core-pilot outputs must live under /mnt" >&2
  exit 1
fi

mkdir -p "$run_dir"
cd "$repository_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/run_lingbot_vla2_ltop_core_pilot.py \
  --source-checkout "$source_checkout" \
  --patch "$repository_root/references/patches/lingbot_vla2_picf_native.patch" \
  --runtime-hotfix "$runtime_hotfix" \
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b \
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct \
  --stage-checkpoint /mnt/picf-next/checkpoints/adr160-g2b-confirm-49eac80-v3 \
  --g2-report /mnt/picf-next/runs/adr160-g2b-confirm-49eac80-v3/ltop_g2_representation_report.json \
  --g3-report "$g3_run_root/ltop_g3_action_mediation_report.json" \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json \
  --norm-stats /mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest /mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json \
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d \
  --stream-plan "$contract_root/two-gpu-2k-i8.physical.stream-plan.json" \
  --stream-plan-sha256 0481025ca66430ac91562f9356bc60e3fd82bedea00ae34a6a5fa6e8708a74cf \
  --representation-split "$contract_root/two-gpu-2k-i8.physical.split.json" \
  --representation-split-sha256 38a5919be926db83d4cd43be1f6192da92e917a4848790a5bc7d8ea1875b38f0 \
  --evaluation-plan "$contract_root/two-gpu-2k-i8.physical.evaluation-plan.json" \
  --evaluation-plan-sha256 24003a1707f6aff1324bbae5a96e5c88448bc47c0b737388503d358e15001244 \
  --execution-contract /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json \
  --offline-labels /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json \
  --run-dir "$run_dir" \
  --arm "$arm" \
  --action-information-set-policy factual-only \
  --mode "$mode" \
  --cuda-allocator expandable-segments
