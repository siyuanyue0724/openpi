#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_DIR fresh | $0 RUN_DIR resume LOAD_GLOBAL_STEP" >&2
  exit 2
fi

run_dir=$1
phase=$2
load_global_step=${3:-0}
repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
source_checkout=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr172-callback-v1}
runtime_hotfix=${PICF_LINGBOT_RUNTIME_HOTFIX:-$repository_root/references/patches/lingbot_vla2_distributed_muon_collective_alignment.patch}
contract_root=${PICF_LONG_CONTRACT_ROOT:-/mnt/picf-next/adr164/contracts/calvin-two-gpu-30k-interleave8-v2}
g3_acceptance_report=${PICF_G3_ACCEPTANCE_REPORT:?PICF_G3_ACCEPTANCE_REPORT is required}
adr172_cold_report=${PICF_ADR172_COLD_REPORT:?PICF_ADR172_COLD_REPORT is required}
adr172_cold_validation=${PICF_ADR172_COLD_VALIDATION:?PICF_ADR172_COLD_VALIDATION is required}
adr172_retention_report=${PICF_ADR172_RETENTION_REPORT:?PICF_ADR172_RETENTION_REPORT is required}
mode=${PICF_LTOP_MODE:-long}

case "$mode" in
  long)
    total_steps=30000
    segment_steps=2000
    ;;
  restart-smoke)
    total_steps=4
    segment_steps=2
    ;;
  *)
    echo "unsupported ADR174 mode: $mode" >&2
    exit 2
    ;;
esac

required=(
  "$repository_root/tools/run_lingbot_vla2_ltop_core_pilot.py"
  "$source_checkout"
  "$repository_root/references/patches/lingbot_vla2_picf_native.patch"
  "$runtime_hotfix"
  "$g3_acceptance_report"
  "$adr172_cold_report"
  "$adr172_cold_validation"
  "$adr172_retention_report"
  "/mnt/picf-next/models/lingbot-vla-v2-6b"
  "/mnt/picf-next/models/qwen3-vl-4b-instruct"
  "/mnt/picf-next/checkpoints/adr172-g2b-callback-compatible-2b1b5da-v1"
  "/mnt/picf-next/checkpoints/adr172-g2b-callback-compatible-2b1b5da-v1/ltop_g2_representation_report.json"
  "/mnt/calvin_data/task_ABC_D/training"
  "/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json"
  "/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json"
  "/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z"
  "/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json"
  "$contract_root/two-gpu-30k-i8.physical.stream-plan.json"
  "$contract_root/two-gpu-30k-i8.physical.split.json"
  "$contract_root/two-gpu-30k-i8.physical.evaluation-plan.json"
  "/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json"
  "/mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "required ADR174 artifact is absent: $path" >&2
    exit 1
  fi
done
if [[ "$run_dir" != /mnt/* ]]; then
  echo "ADR174 outputs must live under /mnt" >&2
  exit 1
fi

case "$phase" in
  fresh)
    if [[ $# -ne 2 || "$load_global_step" -ne 0 || -e "$run_dir" ]]; then
      echo "fresh ADR174 launch requires one absent run directory" >&2
      exit 2
    fi
    ;;
  resume)
    if [[ $# -ne 3 || ! "$load_global_step" =~ ^[0-9]+$ ]]; then
      echo "resume ADR174 launch requires an integer load step" >&2
      exit 2
    fi
    if (( load_global_step <= 0 || load_global_step > total_steps || load_global_step % segment_steps != 0 )); then
      echo "resume ADR174 step is not a registered boundary: $load_global_step" >&2
      exit 2
    fi
    if [[ ! -f "$run_dir/checkpoints/global_step_$load_global_step/ltop_core_pilot_checkpoint.json" ]]; then
      echo "resume ADR174 checkpoint manifest is absent" >&2
      exit 1
    fi
    ;;
  *)
    echo "ADR174 phase must be fresh or resume" >&2
    exit 2
    ;;
esac

mkdir -p "$run_dir/launcher_logs"
cd "$repository_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

common_args=(
  --source-checkout "$source_checkout"
  --patch "$repository_root/references/patches/lingbot_vla2_picf_native.patch"
  --runtime-hotfix "$runtime_hotfix"
  --checkpoint-dir /mnt/picf-next/models/lingbot-vla-v2-6b
  --processor-dir /mnt/picf-next/models/qwen3-vl-4b-instruct
  --stage-checkpoint /mnt/picf-next/checkpoints/adr172-g2b-callback-compatible-2b1b5da-v1
  --g2-report /mnt/picf-next/checkpoints/adr172-g2b-callback-compatible-2b1b5da-v1/ltop_g2_representation_report.json
  --g3-report "$g3_acceptance_report"
  --adr172-cold-report "$adr172_cold_report"
  --adr172-cold-validation "$adr172_cold_validation"
  --adr172-physical-retention-report "$adr172_retention_report"
  --action-information-set-policy factual-only
  --dataset-split /mnt/calvin_data/task_ABC_D/training
  --dataset-manifest /mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json
  --norm-stats /mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z
  --physical-sidecar-manifest /mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json
  --physical-sidecar-manifest-sha256 ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d
  --stream-plan "$contract_root/two-gpu-30k-i8.physical.stream-plan.json"
  --stream-plan-sha256 d35b4c587fa30e6d23029da4ef2f6cccf08faa83b0bd937ab15379aeb1e69d71
  --representation-split "$contract_root/two-gpu-30k-i8.physical.split.json"
  --representation-split-sha256 0852f5bed788da25b857c0bf3e6e9009ab9887ea44784f366fa9bef0de2904fe
  --evaluation-plan "$contract_root/two-gpu-30k-i8.physical.evaluation-plan.json"
  --evaluation-plan-sha256 e873da94f941bf706629329287d3a9f850041cb6c2dc2fc60a47d85023e473d3
  --execution-contract /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.execution.json
  --offline-labels /mnt/picf-next/adr154/contracts/adr157-g2-smoke4-v1/g2-compact-16.labels.json
  --run-dir "$run_dir"
  --arm ltop-ec-factual
  --mode "$mode"
  --cuda-allocator expandable-segments
)

current_step=$load_global_step
current_phase=$phase
while (( current_step < total_steps )); do
  stop_step=$((current_step + segment_steps))
  if (( stop_step > total_steps )); then
    stop_step=$total_steps
  fi
  log_path="$run_dir/launcher_logs/${current_phase}_${current_step}_to_${stop_step}.log"
  "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/run_lingbot_vla2_ltop_core_pilot.py \
    "${common_args[@]}" \
    --phase "$current_phase" \
    --load-global-step "$current_step" \
    --stop-after-step "$stop_step" \
    2>&1 | tee "$log_path"
  current_step=$stop_step
  current_phase=resume
done

verification_log="$run_dir/launcher_logs/resume_${total_steps}_verify.log"
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/run_lingbot_vla2_ltop_core_pilot.py \
  "${common_args[@]}" \
  --phase resume \
  --load-global-step "$total_steps" \
  --stop-after-step "$total_steps" \
  2>&1 | tee "$verification_log"
