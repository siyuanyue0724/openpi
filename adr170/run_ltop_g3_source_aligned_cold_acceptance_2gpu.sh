#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TRAINING_RUN_ROOT ACCEPTANCE_RUN_ROOT" >&2
  exit 2
fi

training_root=$1
acceptance_root=$2
repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
evaluation_scenes_per_partition=${PICF_G3_EVALUATION_SCENES_PER_PARTITION:-4}

case "$evaluation_scenes_per_partition" in
  1) evaluation_scope=quick ;;
  4) evaluation_scope=full ;;
  *)
    echo "PICF_G3_EVALUATION_SCENES_PER_PARTITION must be 1 or 4" >&2
    exit 1
    ;;
esac

if [[ "$training_root" != /mnt/* || "$acceptance_root" != /mnt/* ]]; then
  echo "ADR170 acceptance inputs and outputs must live under /mnt" >&2
  exit 1
fi
if [[ -e "$acceptance_root" || -L "$acceptance_root" ]]; then
  echo "ADR170 acceptance output must be one absent direct path under /mnt" >&2
  exit 1
fi
for launcher in \
  "$repository_root/adr170/run_ltop_g3_source_aligned_cold_action_2gpu.sh" \
  "$repository_root/adr170/run_ltop_g3_source_aligned_retention_2gpu.sh"; do
  if [[ ! -x "$launcher" ]]; then
    echo "required ADR170 launcher is absent or not executable: $launcher" >&2
    exit 1
  fi
done
if [[ ! -x "$python_bin" ]]; then
  echo "required ADR170 Python runtime is absent: $python_bin" >&2
  exit 1
fi
for path in \
  "$repository_root/tools/validate_ltop_g3_mediator_trial.py" \
  "$repository_root/tools/compose_ltop_g3_source_aligned_acceptance.py"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "required ADR170 acceptance artifact is absent: $path" >&2
    exit 1
  fi
done
if [[ $(git -C "$repository_root" status --porcelain=v1 --untracked-files=all) ]]; then
  echo "ADR170 cold acceptance requires an exact clean source checkout" >&2
  exit 1
fi

mkdir -p "$acceptance_root"
PICF_G3_EVALUATION_SCENES_PER_PARTITION="$evaluation_scenes_per_partition" \
PICF_G3_EVALUATION_ACTION_INFORMATION_SET=factual \
  "$repository_root/adr170/run_ltop_g3_source_aligned_cold_action_2gpu.sh" \
  "$training_root" "$acceptance_root/action-factual-$evaluation_scope"
PICF_G3_EVALUATION_SCENES_PER_PARTITION="$evaluation_scenes_per_partition" \
PICF_G3_EVALUATION_ACTION_INFORMATION_SET=mediator-required \
  "$repository_root/adr170/run_ltop_g3_source_aligned_cold_action_2gpu.sh" \
  "$training_root" "$acceptance_root/action-mediator-required-$evaluation_scope"
"$repository_root/adr170/run_ltop_g3_source_aligned_retention_2gpu.sh" \
  "$training_root" "$acceptance_root/retention"

training_report=$training_root/ltop_g3_source_aligned_trial_training_report.json
arm_validation=$acceptance_root/ltop_g3_source_aligned_arm_validation.json
factual_report=$acceptance_root/action-factual-$evaluation_scope/ltop_g3_source_aligned_cold_action_${evaluation_scope}_report.json
mediator_report=$acceptance_root/action-mediator-required-$evaluation_scope/ltop_g3_source_aligned_cold_action_${evaluation_scope}_report.json
retention_report=$acceptance_root/retention/ltop_g3_source_aligned_representation_retention_report.json
acceptance_report=$acceptance_root/ltop_g3_source_aligned_acceptance.json

cd "$repository_root"
export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

"$python_bin" "$repository_root/tools/validate_ltop_g3_mediator_trial.py" \
  --journal-dir "$training_root/rank_journal" \
  --report "$training_report" \
  --output "$arm_validation"
"$python_bin" "$repository_root/tools/compose_ltop_g3_source_aligned_acceptance.py" \
  --training-report "$training_report" \
  --arm-validation "$arm_validation" \
  --factual-action-report "$factual_report" \
  --mediator-action-report "$mediator_report" \
  --retention-report "$retention_report" \
  --output "$acceptance_report"

printf 'ADR170 source-aligned acceptance passed: %s\n' "$acceptance_report"
