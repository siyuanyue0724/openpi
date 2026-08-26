#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr132-j0-20260804}
OUTPUT=${OUTPUT:?OUTPUT must name one fresh persistent ADR135 bundle}
STEP_ZERO_BASELINE=${STEP_ZERO_BASELINE:-}

ROOT=/mnt/picf-next/adr135
CONTRACTS=$ROOT/contracts
CACHE=$ROOT/cache

K1_SPLIT=$CONTRACTS/k1-1000.split.json
K1_EVALUATION_PLAN=$CONTRACTS/k1-1000.evaluation-plan.json
K1_PREDICTIVE_ROOT=$CACHE/k1-predictive
K1_PREDICTIVE_REPORT=$CACHE/k1-predictive.build_report.json
K1_TARGET_AUDIT=$CONTRACTS/k1-predictive-target-audit.json
K1_TEACHER_AUDIT=$CONTRACTS/k1-teacher-causality-audit.json
K1_TEMPORAL_AUDIT=$CONTRACTS/k1-predictive-temporal-audit.json
K1_CURRENT_GRID_ROOT=$CACHE/k1-current-grid
K1_CURRENT_GRID_REPORT=$CACHE/k1-current-grid.build_report.json

K8_SPLIT=$CONTRACTS/k8-1000-reset-half.split.json
RESET_EVALUATION_PLAN=$CONTRACTS/k8-1000-reset-half.evaluation-plan.json
WARM_EVALUATION_PLAN=$CONTRACTS/k8-1000-reset-half.warm-evaluation-plan.json
FIXED_PAIR_PLAN=$CONTRACTS/fixed-x/training-pair-plan-v3.json
FIXED_TRAINING_AUDIT=$CONTRACTS/fixed-x/training/token-grid-v3.json
FIXED_EVALUATION_PLAN=$CONTRACTS/fixed-x/evaluation-plan-v3.json
FIXED_VALIDATION_AUDIT=$CONTRACTS/fixed-x/validation/token-grid-v3.json
FIXED_HELDOUT_AUDIT=$CONTRACTS/fixed-x/heldout/token-grid-v3.json
K8_PREDICTIVE_ROOT=$CACHE/k8-predictive
K8_PREDICTIVE_REPORT=$CACHE/k8-predictive.build_report.json
K8_TARGET_AUDIT=$CONTRACTS/k8-predictive-target-audit.json
K8_TEACHER_AUDIT=$CONTRACTS/k8-teacher-causality-audit.json
K8_TEMPORAL_AUDIT=$CONTRACTS/k8-predictive-temporal-audit.json
K8_CURRENT_GRID_ROOT=$CACHE/k8-current-grid
K8_CURRENT_GRID_REPORT=$CACHE/k8-current-grid.build_report.json

RUNNER=$WORKTREE/tools/run_lingbot_vla2_native_full.py
ARM_SCRIPT=$WORKTREE/configs/cloud/adr135_matched_medium_horizon_arm.sh
BASELINE_SCRIPT=$WORKTREE/configs/cloud/adr135_step_zero_baseline.sh

if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to replace ADR135 bundle: $OUTPUT" >&2
  exit 1
fi

required_files=(
  "$RUNNER" "$ARM_SCRIPT" "$BASELINE_SCRIPT"
  "$K1_SPLIT" "$K1_EVALUATION_PLAN" "$K1_PREDICTIVE_REPORT"
  "$K1_TARGET_AUDIT" "$K1_TEACHER_AUDIT" "$K1_TEMPORAL_AUDIT"
  "$K1_CURRENT_GRID_REPORT" "$K8_SPLIT" "$RESET_EVALUATION_PLAN"
  "$WARM_EVALUATION_PLAN" "$FIXED_PAIR_PLAN" "$FIXED_TRAINING_AUDIT"
  "$FIXED_EVALUATION_PLAN" "$FIXED_VALIDATION_AUDIT"
  "$FIXED_HELDOUT_AUDIT" "$K8_PREDICTIVE_REPORT" "$K8_TARGET_AUDIT"
  "$K8_TEACHER_AUDIT" "$K8_TEMPORAL_AUDIT" "$K8_CURRENT_GRID_REPORT"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ADR135 bundle input is missing: $path" >&2
    exit 1
  fi
done
for path in "$K1_PREDICTIVE_ROOT" "$K1_CURRENT_GRID_ROOT" \
  "$K8_PREDICTIVE_ROOT" "$K8_CURRENT_GRID_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "ADR135 cache root is missing: $path" >&2
    exit 1
  fi
done
if [[ -n "$STEP_ZERO_BASELINE" && ! -f "$STEP_ZERO_BASELINE" ]]; then
  echo "ADR135 step-zero baseline is missing: $STEP_ZERO_BASELINE" >&2
  exit 1
fi

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

emit() {
  printf '%s=%q\n' "$1" "$2" >>"$TEMP"
}

mkdir -p "$(dirname "$OUTPUT")"
TEMP="$(dirname "$OUTPUT")/.${OUTPUT##*/}.tmp-$$"
if [[ -e "$TEMP" || -L "$TEMP" ]]; then
  echo "ADR135 temporary bundle path exists: $TEMP" >&2
  exit 1
fi
trap 'rm -f "$TEMP"' EXIT

emit BUNDLE_SCHEMA picf-next.adr135-matched-medium-horizon-bundle.v1
emit CODE_WORKTREE "$WORKTREE"
emit RUNNER_SHA256 "$(sha256 "$RUNNER")"
emit ARM_SCRIPT_SHA256 "$(sha256 "$ARM_SCRIPT")"
emit BASELINE_SCRIPT_SHA256 "$(sha256 "$BASELINE_SCRIPT")"

for name in \
  K1_SPLIT K1_EVALUATION_PLAN K1_PREDICTIVE_REPORT K1_TARGET_AUDIT \
  K1_TEACHER_AUDIT K1_TEMPORAL_AUDIT K1_CURRENT_GRID_REPORT K8_SPLIT \
  RESET_EVALUATION_PLAN WARM_EVALUATION_PLAN FIXED_PAIR_PLAN \
  FIXED_TRAINING_AUDIT FIXED_EVALUATION_PLAN FIXED_VALIDATION_AUDIT \
  FIXED_HELDOUT_AUDIT K8_PREDICTIVE_REPORT K8_TARGET_AUDIT \
  K8_TEACHER_AUDIT K8_TEMPORAL_AUDIT K8_CURRENT_GRID_REPORT; do
  path=${!name}
  emit "$name" "$path"
  emit "${name}_SHA256" "$(sha256 "$path")"
done
for name in K1_PREDICTIVE_ROOT K1_CURRENT_GRID_ROOT K8_PREDICTIVE_ROOT K8_CURRENT_GRID_ROOT; do
  emit "$name" "${!name}"
done
if [[ -n "$STEP_ZERO_BASELINE" ]]; then
  emit STEP_ZERO_BASELINE "$STEP_ZERO_BASELINE"
  emit STEP_ZERO_BASELINE_SHA256 "$(sha256 "$STEP_ZERO_BASELINE")"
fi

sync -f "$TEMP"
mv "$TEMP" "$OUTPUT"
trap - EXIT
sync -f "$(dirname "$OUTPUT")"
printf 'bundle=%s\nbundle_sha256=%s\n' "$OUTPUT" "$(sha256 "$OUTPUT")"
