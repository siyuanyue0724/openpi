#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr136-content-addressed-set-20260805}
BUNDLE=${ADR136_BUNDLE:?ADR136_BUNDLE must name the immutable contract/cache bundle}
RUN_DIR=${RUN_DIR:?RUN_DIR must name one fresh persistent baseline run}
LOG=${LOG:?LOG must name one fresh persistent log}

if [[ ! -f "$BUNDLE" ]]; then
  echo "ADR136 bundle is missing: $BUNDLE" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$BUNDLE"
required_bundle_names=(
  BUNDLE_SCHEMA OBJECT_TRANSITION CODE_WORKTREE RUNNER_SHA256 PUBLISH_SCRIPT_SHA256
  BASELINE_SCRIPT_SHA256
  HOST_SHA256 GRAPH_SHA256 RELATIONS_SHA256 SUPERVISION_SHA256 TASK_RELATION_SHA256
  TEMPORAL_SHA256 FULL_TRAINING_SHA256 K1_SPLIT K1_SPLIT_SHA256
  K1_EVALUATION_PLAN K1_EVALUATION_PLAN_SHA256 K1_PREDICTIVE_ROOT
  K1_PREDICTIVE_REPORT K1_PREDICTIVE_REPORT_SHA256 K1_TARGET_AUDIT
  K1_TARGET_AUDIT_SHA256 K1_TEACHER_AUDIT K1_TEACHER_AUDIT_SHA256
  K1_TEMPORAL_AUDIT K1_TEMPORAL_AUDIT_SHA256 K1_CURRENT_GRID_ROOT
  K1_CURRENT_GRID_REPORT K1_CURRENT_GRID_REPORT_SHA256
)
for name in "${required_bundle_names[@]}"; do
  if [[ -z ${!name:-} ]]; then
    echo "ADR136 bundle omitted $name" >&2
    exit 1
  fi
done
if [[ "$BUNDLE_SCHEMA" != picf-next.adr136-content-addressed-set-bundle.v1 ]]; then
  echo "ADR136 bundle schema changed: $BUNDLE_SCHEMA" >&2
  exit 1
fi
if [[ "$OBJECT_TRANSITION" != content_addressed_set_v1 ]]; then
  echo "ADR136 object transition changed: $OBJECT_TRANSITION" >&2
  exit 1
fi
if [[ "$WORKTREE" != "$CODE_WORKTREE" ]]; then
  echo "ADR136 bundle belongs to another worktree: $CODE_WORKTREE" >&2
  exit 1
fi

assert_sha256() {
  local path=$1
  local expected=$2
  local label=$3
  if [[ $(sha256sum "$path" | awk '{print $1}') != "$expected" ]]; then
    echo "ADR136 $label changed after bundle publication" >&2
    exit 1
  fi
}

assert_sha256 "$WORKTREE/tools/run_lingbot_vla2_native_full.py" "$RUNNER_SHA256" runner
assert_sha256 "$WORKTREE/configs/cloud/adr136_publish_bundle.sh" \
  "$PUBLISH_SCRIPT_SHA256" "bundle publisher"
assert_sha256 "$WORKTREE/configs/cloud/adr136_step_zero_baseline.sh" \
  "$BASELINE_SCRIPT_SHA256" "baseline launcher"
assert_sha256 "$WORKTREE/src/picf_next/lingbot_native/host.py" "$HOST_SHA256" host
assert_sha256 "$WORKTREE/src/picf_next/lingbot_native/graph.py" "$GRAPH_SHA256" graph
assert_sha256 "$WORKTREE/src/picf_next/lingbot_native/relations.py" "$RELATIONS_SHA256" relations
assert_sha256 "$WORKTREE/src/picf_next/lingbot_native/supervision.py" \
  "$SUPERVISION_SHA256" supervision
assert_sha256 "$WORKTREE/src/picf_next/lingbot_native/task_relation.py" \
  "$TASK_RELATION_SHA256" task-relation
assert_sha256 "$WORKTREE/src/picf_next/lingbot_native/temporal.py" "$TEMPORAL_SHA256" temporal
assert_sha256 "$WORKTREE/src/picf_next/lingbot_native/full_training.py" \
  "$FULL_TRAINING_SHA256" full-training

for path in "$RUN_DIR" "$LOG"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing to reuse ADR136 baseline artifact: $path" >&2
    exit 1
  fi
done
mkdir -p "$RUN_DIR" "$(dirname "$LOG")"
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
  --phase fresh \
  --training-stage representation \
  --checkpoint-publication never \
  --representation-split "$K1_SPLIT" \
  --representation-split-sha256 "$K1_SPLIT_SHA256" \
  --representation-evaluation-plan "$K1_EVALUATION_PLAN" \
  --representation-evaluation-plan-sha256 "$K1_EVALUATION_PLAN_SHA256" \
  --representation-evaluation-steps 0,1 \
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
  --predictive-cache-root "$K1_PREDICTIVE_ROOT" \
  --predictive-cache-build-report "$K1_PREDICTIVE_REPORT" \
  --predictive-cache-build-report-sha256 "$K1_PREDICTIVE_REPORT_SHA256" \
  --predictive-target-audit "$K1_TARGET_AUDIT" \
  --predictive-target-audit-sha256 "$K1_TARGET_AUDIT_SHA256" \
  --predictive-teacher-causality-audit "$K1_TEACHER_AUDIT" \
  --predictive-teacher-causality-audit-sha256 "$K1_TEACHER_AUDIT_SHA256" \
  --predictive-temporal-audit "$K1_TEMPORAL_AUDIT" \
  --predictive-temporal-audit-sha256 "$K1_TEMPORAL_AUDIT_SHA256" \
  --current-grid-cache-root "$K1_CURRENT_GRID_ROOT" \
  --current-grid-cache-build-report "$K1_CURRENT_GRID_REPORT" \
  --current-grid-cache-build-report-sha256 "$K1_CURRENT_GRID_REPORT_SHA256" \
  --run-dir "$RUN_DIR" \
  --load-global-step 0 \
  --invocation-steps 1 \
  --total-planned-steps 1000 \
  --seed 20260721 \
  --capacity 16 \
  --maximum-control-tokens 8 \
  --maximum-optimizer-lag 8 \
  --lane-interleave-factor 1 \
  --maximum-peak-reserved-gib 39 \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator expandable-segments \
  --predictive-weight 0.004 \
  --structural-weight 0.004 \
  --support-weight 0 \
  --dense-task-weight 0 \
  --gradient-audit-steps 2,3 \
  --source-prediction-mode omitted_static \
  --evidence-profile loss_visual_trial \
  --visual-audit-every 0 \
  --task-relation-estimator host_native_factorized_task_physical_ownership \
  --ownership-estimator token_micro_categorical \
  >"$LOG" 2>&1
