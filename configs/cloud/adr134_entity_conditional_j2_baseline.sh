#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr132-j0-20260804}
RUN_DIR=${RUN_DIR:?RUN_DIR must name one fresh persistent J2 baseline-source run}
LOG=${LOG:?LOG must name one persistent log file}
PHASE=${PHASE:-fresh}
LOAD_GLOBAL_STEP=${LOAD_GLOBAL_STEP:-0}

case "$PHASE:$LOAD_GLOBAL_STEP" in
  fresh:0)
    EVALUATION_STEPS=0
    ;;
  resume:1)
    EVALUATION_STEPS=2
    ;;
  *)
    echo "ADR-134 baseline transaction permits only fresh:0 or resume:1" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR" "$(dirname "$LOG")"
echo $$ > "$RUN_DIR/launcher-${PHASE}-${LOAD_GLOBAL_STEP}.pid"

# ADR-134 keeps the exact ADR-132 model, data and released weights. This frozen
# K1 invocation binds the new within-family estimator contract and evaluates
# step zero before its single transaction-only update.
exec env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_ALLOC_CONF \
  -u PICF_LINGBOT_TRAINING_AUTHORIZATION \
  -u PICF_LINGBOT_TRAINING_AUTHORIZATION_SHA256 \
  PYTHONPATH="$WORKTREE/src:$WORKTREE" \
  CUDA_VISIBLE_DEVICES=0,1 \
  /opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "$WORKTREE/tools/run_lingbot_vla2_native_full.py" \
  --phase "$PHASE" \
  --training-stage representation \
  --checkpoint-publication always \
  --representation-split "$WORKTREE/references/experiments/lingbot-representation-k1-200-reference-split.json" \
  --representation-split-sha256 392fd6b9ba6b15e015d39a14e5036bbd7eeaad407b44d1a9ab3bfda2835a31b7 \
  --representation-evaluation-plan /mnt/picf-next/probes/representation-evaluation-plan-3b6d367-v3.json \
  --representation-evaluation-plan-sha256 9518c1e646bb3a20fd26ff9b5011e7d2d522f88f367c953d17222c0d8ec960f4 \
  --representation-evaluation-steps "$EVALUATION_STEPS" \
  --source-checkout /mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2 \
  --patch "$WORKTREE/references/patches/lingbot_vla2_picf_native.patch" \
  --training-config /mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2/configs/vla/robotwin/robotwin.yaml \
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
  --predictive-cache-root /mnt/picf-next/cache/adr132-k1-current-estimator-predictive-20260804T090614+0800 \
  --predictive-cache-build-report /mnt/picf-next/cache/adr132-k1-current-estimator-predictive-20260804T090614+0800.build_report.json \
  --predictive-cache-build-report-sha256 5827860fa6e6acf6b54fb770d6982cc1ada015dbcc46b77fde2cb72048064978 \
  --predictive-target-audit /mnt/picf-next/runs/adr132-k1-current-estimator-cache-20260804T090614+0800/predictive-target-audit.json \
  --predictive-target-audit-sha256 d11d7f680ce19c1a4e233c6860a81d74fbba1bcc549a0dee1eed2afa8529f676 \
  --predictive-teacher-causality-audit /mnt/picf-next/runs/adr132-k1-current-estimator-cache-20260804T090614+0800/teacher-causality-audit.json \
  --predictive-teacher-causality-audit-sha256 2fc488bdda93486a4bfadea8135b8c36cb820bdb3e9a4ef37cc07179465c83ee \
  --predictive-temporal-audit /mnt/picf-next/runs/adr132-k1-current-estimator-cache-20260804T090614+0800/predictive-temporal-audit.json \
  --predictive-temporal-audit-sha256 dffbe135da54f242f738e274c2b2cb17323aecd8041b847d0b61f35774f277fe \
  --current-grid-cache-root /mnt/picf-next/cache/adr132-k1-current-estimator-current-grid-20260804T090614+0800 \
  --current-grid-cache-build-report /mnt/picf-next/cache/adr132-k1-current-estimator-current-grid-20260804T090614+0800.build_report.json \
  --current-grid-cache-build-report-sha256 ff92c832e7246fdf896e5016ca187161031b365e18e62909472fe3a939373036 \
  --run-dir "$RUN_DIR" \
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --invocation-steps 1 \
  --total-planned-steps 200 \
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
  --evidence-profile acceptance \
  --visual-audit-every 0 \
  --task-relation-estimator host_native_factorized_task_physical_ownership \
  --ownership-estimator token_micro_entity_conditional_equal \
  >"$LOG" 2>&1
