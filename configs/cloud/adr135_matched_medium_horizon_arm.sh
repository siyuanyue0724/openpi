#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr132-j0-20260804}
BUNDLE=${ADR135_BUNDLE:?ADR135_BUNDLE must name the immutable contract/cache bundle}
RUN_DIR=${RUN_DIR:?RUN_DIR must name one persistent arm run directory}
LOG=${LOG:?LOG must name one fresh persistent segment log}
ARM=${ARM:?ARM must be M or E}
PHASE=${PHASE:?PHASE must be fresh or resume}
LOAD_GLOBAL_STEP=${LOAD_GLOBAL_STEP:?LOAD_GLOBAL_STEP is required}
INVOCATION_STEPS=${INVOCATION_STEPS:?INVOCATION_STEPS is required}
EVALUATION_STEP=${EVALUATION_STEP:?EVALUATION_STEP is required}

if [[ ! -f "$BUNDLE" ]]; then
  echo "ADR135 bundle is missing: $BUNDLE" >&2
  exit 1
fi
# The bundle is generated locally under /mnt and contains only quoted path/hash assignments.
# shellcheck disable=SC1090
source "$BUNDLE"

required_bundle_names=(
  BUNDLE_SCHEMA CODE_WORKTREE RUNNER_SHA256 ARM_SCRIPT_SHA256
  K8_SPLIT K8_SPLIT_SHA256 RESET_EVALUATION_PLAN RESET_EVALUATION_PLAN_SHA256
  WARM_EVALUATION_PLAN WARM_EVALUATION_PLAN_SHA256 FIXED_PAIR_PLAN
  FIXED_PAIR_PLAN_SHA256 FIXED_TRAINING_AUDIT FIXED_TRAINING_AUDIT_SHA256
  FIXED_EVALUATION_PLAN FIXED_EVALUATION_PLAN_SHA256 FIXED_VALIDATION_AUDIT
  FIXED_VALIDATION_AUDIT_SHA256 FIXED_HELDOUT_AUDIT FIXED_HELDOUT_AUDIT_SHA256
  K8_PREDICTIVE_ROOT K8_PREDICTIVE_REPORT K8_PREDICTIVE_REPORT_SHA256
  K8_TARGET_AUDIT K8_TARGET_AUDIT_SHA256 K8_TEACHER_AUDIT K8_TEACHER_AUDIT_SHA256
  K8_TEMPORAL_AUDIT K8_TEMPORAL_AUDIT_SHA256 K8_CURRENT_GRID_ROOT
  K8_CURRENT_GRID_REPORT K8_CURRENT_GRID_REPORT_SHA256 STEP_ZERO_BASELINE
  STEP_ZERO_BASELINE_SHA256
)
for name in "${required_bundle_names[@]}"; do
  if [[ -z ${!name:-} ]]; then
    echo "ADR135 bundle omitted $name" >&2
    exit 1
  fi
done

if [[ "$BUNDLE_SCHEMA" != picf-next.adr135-matched-medium-horizon-bundle.v1 ]]; then
  echo "ADR135 bundle schema changed: $BUNDLE_SCHEMA" >&2
  exit 1
fi

if [[ "$WORKTREE" != "$CODE_WORKTREE" ]]; then
  echo "ADR135 bundle belongs to another worktree: $CODE_WORKTREE" >&2
  exit 1
fi
if [[ $(sha256sum "$WORKTREE/tools/run_lingbot_vla2_native_full.py" | awk '{print $1}') != "$RUNNER_SHA256" ]]; then
  echo "ADR135 runner changed after bundle publication" >&2
  exit 1
fi
if [[ $(sha256sum "$WORKTREE/configs/cloud/adr135_matched_medium_horizon_arm.sh" | awk '{print $1}') != "$ARM_SCRIPT_SHA256" ]]; then
  echo "ADR135 arm launcher changed after bundle publication" >&2
  exit 1
fi

case "$ARM" in
  M) OWNERSHIP_ESTIMATOR=token_micro_categorical ;;
  E) OWNERSHIP_ESTIMATOR=token_micro_entity_conditional_equal ;;
  *) echo "ARM must be M or E" >&2; exit 1 ;;
esac

case "$PHASE:$LOAD_GLOBAL_STEP:$INVOCATION_STEPS:$EVALUATION_STEP" in
  fresh:0:1:0 | resume:1:199:200 | resume:200:300:500 | resume:500:500:1000) ;;
  *)
    echo "unregistered ADR135 segment" >&2
    exit 1
    ;;
esac

if [[ -e "$LOG" || -L "$LOG" ]]; then
  echo "refusing to reuse ADR135 segment log: $LOG" >&2
  exit 1
fi
if [[ "$PHASE" == fresh ]]; then
  if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
    echo "refusing to reuse ADR135 arm run: $RUN_DIR" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR"
elif [[ ! -d "$RUN_DIR" ]]; then
  echo "ADR135 resume run is missing: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$(dirname "$LOG")"
printf '%s\n' "$$" >"$RUN_DIR/launcher.pid"

SOURCE=/mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2
PYTHON=/opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python

exec env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_ALLOC_CONF \
  -u PICF_LINGBOT_TRAINING_AUTHORIZATION \
  -u PICF_LINGBOT_TRAINING_AUTHORIZATION_SHA256 \
  PYTHONPATH="$WORKTREE/src:$WORKTREE" \
  CUDA_VISIBLE_DEVICES=0,1 \
  "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "$WORKTREE/tools/run_lingbot_vla2_native_full.py" \
  --phase "$PHASE" \
  --training-stage representation \
  --checkpoint-publication always \
  --representation-split "$K8_SPLIT" \
  --representation-split-sha256 "$K8_SPLIT_SHA256" \
  --fixed-observation-pair-plan "$FIXED_PAIR_PLAN" \
  --fixed-observation-pair-plan-sha256 "$FIXED_PAIR_PLAN_SHA256" \
  --fixed-observation-training-audit "$FIXED_TRAINING_AUDIT" \
  --fixed-observation-training-audit-sha256 "$FIXED_TRAINING_AUDIT_SHA256" \
  --fixed-observation-evaluation-plan "$FIXED_EVALUATION_PLAN" \
  --fixed-observation-evaluation-plan-sha256 "$FIXED_EVALUATION_PLAN_SHA256" \
  --fixed-observation-validation-audit "$FIXED_VALIDATION_AUDIT" \
  --fixed-observation-validation-audit-sha256 "$FIXED_VALIDATION_AUDIT_SHA256" \
  --fixed-observation-heldout-audit "$FIXED_HELDOUT_AUDIT" \
  --fixed-observation-heldout-audit-sha256 "$FIXED_HELDOUT_AUDIT_SHA256" \
  --representation-evaluation-plan "$RESET_EVALUATION_PLAN" \
  --representation-evaluation-plan-sha256 "$RESET_EVALUATION_PLAN_SHA256" \
  --representation-warm-evaluation-plan "$WARM_EVALUATION_PLAN" \
  --representation-warm-evaluation-plan-sha256 "$WARM_EVALUATION_PLAN_SHA256" \
  --representation-evaluation-baseline "$STEP_ZERO_BASELINE" \
  --representation-evaluation-baseline-sha256 "$STEP_ZERO_BASELINE_SHA256" \
  --representation-evaluation-steps "$EVALUATION_STEP" \
  --source-checkout "$SOURCE" \
  --patch "$WORKTREE/references/patches/lingbot_vla2_picf_native.patch" \
  --training-config "$SOURCE/configs/vla/robotwin/robotwin.yaml" \
  --robot-config "$WORKTREE/configs/lingbot/calvin_robot.yaml" \
  --data-config "$WORKTREE/configs/lingbot/calvin_data.json" \
  --checkpoint-dir /root/picf-runtime-42a7ad9/models/lingbot-vla-v2-6b \
  --processor-dir /root/picf-runtime-42a7ad9/models/qwen3-vl-4b-instruct \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next/manifests/calvin-training-files.json \
  --norm-stats /mnt/picf-next/manifests/calvin-lingbot-norm-stats.json \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest-sha256 0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4 \
  --physical-visual-acceptance /mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json \
  --physical-visual-acceptance-sha256 4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86 \
  --predictive-cache-root "$K8_PREDICTIVE_ROOT" \
  --predictive-cache-build-report "$K8_PREDICTIVE_REPORT" \
  --predictive-cache-build-report-sha256 "$K8_PREDICTIVE_REPORT_SHA256" \
  --predictive-target-audit "$K8_TARGET_AUDIT" \
  --predictive-target-audit-sha256 "$K8_TARGET_AUDIT_SHA256" \
  --predictive-teacher-causality-audit "$K8_TEACHER_AUDIT" \
  --predictive-teacher-causality-audit-sha256 "$K8_TEACHER_AUDIT_SHA256" \
  --predictive-temporal-audit "$K8_TEMPORAL_AUDIT" \
  --predictive-temporal-audit-sha256 "$K8_TEMPORAL_AUDIT_SHA256" \
  --current-grid-cache-root "$K8_CURRENT_GRID_ROOT" \
  --current-grid-cache-build-report "$K8_CURRENT_GRID_REPORT" \
  --current-grid-cache-build-report-sha256 "$K8_CURRENT_GRID_REPORT_SHA256" \
  --run-dir "$RUN_DIR" \
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --invocation-steps "$INVOCATION_STEPS" \
  --total-planned-steps 1000 \
  --seed 20260721 \
  --capacity 16 \
  --maximum-control-tokens 8 \
  --maximum-optimizer-lag 16 \
  --lane-interleave-factor 8 \
  --local-bptt-probability 0.10 \
  --overshoot-probability 0.05 \
  --source-mask-probability 0.10 \
  --reset-mixture-numerator 1 \
  --reset-mixture-denominator 2 \
  --maximum-peak-reserved-gib 39 \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator expandable-segments \
  --predictive-weight 0.004 \
  --structural-weight 0.004 \
  --support-weight 0 \
  --dense-task-weight 0 \
  --gradient-audit-steps 18,34,50,100,200,500,1000 \
  --source-prediction-mode omitted_static \
  --evidence-profile matched_medium_horizon \
  --visual-audit-every 200 \
  --task-relation-estimator host_native_factorized_task_physical_ownership \
  --ownership-estimator "$OWNERSHIP_ESTIMATOR" \
  >"$LOG" 2>&1
