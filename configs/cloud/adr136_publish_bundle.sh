#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr136-content-addressed-set-20260805}
OUTPUT=${OUTPUT:?OUTPUT must name one fresh persistent ADR136 bundle}
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
PUBLISH_SCRIPT=$WORKTREE/configs/cloud/adr136_publish_bundle.sh
ARM_SCRIPT=$WORKTREE/configs/cloud/adr136_content_addressed_set_arm.sh
BASELINE_SCRIPT=$WORKTREE/configs/cloud/adr136_step_zero_baseline.sh
HOST_SOURCE=$WORKTREE/src/picf_next/lingbot_native/host.py
GRAPH_SOURCE=$WORKTREE/src/picf_next/lingbot_native/graph.py
RELATIONS_SOURCE=$WORKTREE/src/picf_next/lingbot_native/relations.py
SUPERVISION_SOURCE=$WORKTREE/src/picf_next/lingbot_native/supervision.py
TASK_RELATION_SOURCE=$WORKTREE/src/picf_next/lingbot_native/task_relation.py
TEMPORAL_SOURCE=$WORKTREE/src/picf_next/lingbot_native/temporal.py
FULL_TRAINING_SOURCE=$WORKTREE/src/picf_next/lingbot_native/full_training.py

if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "refusing to replace ADR136 bundle: $OUTPUT" >&2
  exit 1
fi

required_files=(
  "$RUNNER" "$PUBLISH_SCRIPT" "$ARM_SCRIPT" "$BASELINE_SCRIPT" "$HOST_SOURCE" "$GRAPH_SOURCE"
  "$RELATIONS_SOURCE" "$SUPERVISION_SOURCE" "$TASK_RELATION_SOURCE" "$TEMPORAL_SOURCE"
  "$FULL_TRAINING_SOURCE" "$K1_SPLIT" "$K1_EVALUATION_PLAN" "$K1_PREDICTIVE_REPORT"
  "$K1_TARGET_AUDIT" "$K1_TEACHER_AUDIT" "$K1_TEMPORAL_AUDIT"
  "$K1_CURRENT_GRID_REPORT" "$K8_SPLIT" "$RESET_EVALUATION_PLAN"
  "$WARM_EVALUATION_PLAN" "$FIXED_PAIR_PLAN" "$FIXED_TRAINING_AUDIT"
  "$FIXED_EVALUATION_PLAN" "$FIXED_VALIDATION_AUDIT" "$FIXED_HELDOUT_AUDIT"
  "$K8_PREDICTIVE_REPORT" "$K8_TARGET_AUDIT" "$K8_TEACHER_AUDIT"
  "$K8_TEMPORAL_AUDIT" "$K8_CURRENT_GRID_REPORT"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ADR136 bundle input is missing: $path" >&2
    exit 1
  fi
done
for path in "$K1_PREDICTIVE_ROOT" "$K1_CURRENT_GRID_ROOT" \
  "$K8_PREDICTIVE_ROOT" "$K8_CURRENT_GRID_ROOT"; do
  if [[ ! -d "$path" ]]; then
    echo "ADR136 cache root is missing: $path" >&2
    exit 1
  fi
done
if [[ -n "$STEP_ZERO_BASELINE" && ! -f "$STEP_ZERO_BASELINE" ]]; then
  echo "ADR136 step-zero baseline is missing: $STEP_ZERO_BASELINE" >&2
  exit 1
fi
if [[ -n "$STEP_ZERO_BASELINE" ]]; then
  PYTHON=/opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python
  env PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" - \
    "$STEP_ZERO_BASELINE" "$RESET_EVALUATION_PLAN" <<'PY'
import sys

from picf_next.lingbot_native.representation_baseline import (
    load_representation_evaluation_baseline,
    validate_representation_baseline_plan,
)
from picf_next.lingbot_native.representation_evaluation import RepresentationEvaluationPlan

baseline = load_representation_evaluation_baseline(sys.argv[1])
candidate_plan = RepresentationEvaluationPlan.load(sys.argv[2])
validate_representation_baseline_plan(baseline, candidate_plan=candidate_plan)
PY
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
  echo "ADR136 temporary bundle path exists: $TEMP" >&2
  exit 1
fi
trap 'rm -f "$TEMP"' EXIT

emit BUNDLE_SCHEMA picf-next.adr136-content-addressed-set-bundle.v1
emit OBJECT_TRANSITION content_addressed_set_v1
emit CODE_WORKTREE "$WORKTREE"
emit RUNNER_SHA256 "$(sha256 "$RUNNER")"
emit PUBLISH_SCRIPT_SHA256 "$(sha256 "$PUBLISH_SCRIPT")"
emit ARM_SCRIPT_SHA256 "$(sha256 "$ARM_SCRIPT")"
emit BASELINE_SCRIPT_SHA256 "$(sha256 "$BASELINE_SCRIPT")"
emit HOST_SHA256 "$(sha256 "$HOST_SOURCE")"
emit GRAPH_SHA256 "$(sha256 "$GRAPH_SOURCE")"
emit RELATIONS_SHA256 "$(sha256 "$RELATIONS_SOURCE")"
emit SUPERVISION_SHA256 "$(sha256 "$SUPERVISION_SOURCE")"
emit TASK_RELATION_SHA256 "$(sha256 "$TASK_RELATION_SOURCE")"
emit TEMPORAL_SHA256 "$(sha256 "$TEMPORAL_SOURCE")"
emit FULL_TRAINING_SHA256 "$(sha256 "$FULL_TRAINING_SOURCE")"

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
