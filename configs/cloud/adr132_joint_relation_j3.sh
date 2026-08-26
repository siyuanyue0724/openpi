#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr132-j0-20260804}
RUN_DIR=${RUN_DIR:?RUN_DIR must name one fresh persistent J3 run directory}
LOG=${LOG:?LOG must name one persistent log file}
G2_EVIDENCE=${G2_EVIDENCE:?G2_EVIDENCE must name the immutable passed G2 report}
G2_EVIDENCE_SHA256=${G2_EVIDENCE_SHA256:?G2_EVIDENCE_SHA256 must bind the exact G2 report}
G0_EVIDENCE=${G0_EVIDENCE:?G0_EVIDENCE must name the current immutable G0 report}
G0_EVIDENCE_SHA256=${G0_EVIDENCE_SHA256:?G0_EVIDENCE_SHA256 must bind the current G0 report}

if [[ -e "$RUN_DIR" ]]; then
  echo "ADR132 J3 requires a new run directory" >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$(dirname "$LOG")"
echo $$ > "$RUN_DIR/launcher.pid"

exec env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_ALLOC_CONF \
  PYTHONPATH="$WORKTREE/src:$WORKTREE" \
  CUDA_VISIBLE_DEVICES=0,1 \
  /opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "$WORKTREE/tools/run_lingbot_vla2_native_full.py" \
  --phase fresh \
  --training-stage action \
  --checkpoint-publication always \
  --representation-split /mnt/picf-next/probes/representation-k8-reference-bank-gate5470c5a-20260729.split.json \
  --representation-split-sha256 6fba907644ced21bc0b21972e0c385c8beb5c741e1e30ee79f8bed121b0b3c26 \
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
  --predictive-cache-root /mnt/picf-next/cache/adr129-stratified-v3-behavior-h1-k8-20260803T230940+0800 \
  --predictive-cache-build-report /mnt/picf-next/cache/adr129-stratified-v3-behavior-h1-k8-20260803T230940+0800.build_report.json \
  --predictive-cache-build-report-sha256 f2b8e4004211daa456eca812ced03f4e4fc2cd7027969d85a4fc9c8fa23e646a \
  --predictive-target-audit /mnt/picf-next/audits/adr129-stratified-v3-behavior-h1-target-20260803T231930+0800.json \
  --predictive-target-audit-sha256 5cee2c24aadb47443e97c038f9c82de71436f2d250d3da317de0abcd5bb1079b \
  --predictive-teacher-causality-audit /mnt/picf-next/audits/adr129-stratified-v3-behavior-h1-teacher-causality-20260803T231930+0800.json \
  --predictive-teacher-causality-audit-sha256 fd05e002602e6cb98c52bb6c687b987e9b4f68f01f69f0f7a149751306b4cd9a \
  --predictive-temporal-audit /mnt/picf-next/audits/adr129-stratified-v3-behavior-h1-temporal-20260803T231930+0800.json \
  --predictive-temporal-audit-sha256 904b41059e857cf6929acb46c905928a93320773789d76b7d599b4b19a91d4cf \
  --current-grid-cache-root /mnt/picf-next/cache/adr129-stratified-v3-current-grid-k8-20260803T230940+0800 \
  --current-grid-cache-build-report /mnt/picf-next/cache/adr129-stratified-v3-current-grid-k8-20260803T230940+0800.build_report.json \
  --current-grid-cache-build-report-sha256 03eb81ecbc9c45be5cba3190b2f890c87f92aa55eedcf5dcd04ba7d4a567023c \
  --behavior-causal-probe-evidence "$G0_EVIDENCE" \
  --behavior-causal-probe-evidence-sha256 "$G0_EVIDENCE_SHA256" \
  --behavior-posterior-control-probe-evidence "$G2_EVIDENCE" \
  --behavior-posterior-control-probe-evidence-sha256 "$G2_EVIDENCE_SHA256" \
  --run-dir "$RUN_DIR" \
  --load-global-step 0 \
  --invocation-steps 60 \
  --total-planned-steps 200 \
  --seed 20260721 \
  --capacity 16 \
  --maximum-control-tokens 8 \
  --maximum-optimizer-lag 8 \
  --lane-interleave-factor 8 \
  --maximum-peak-reserved-gib 39 \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator expandable-segments \
  --predictive-weight 0.004 \
  --structural-weight 0.004 \
  --support-weight 0 \
  --task-relation-estimator host_native_factorized_task_physical_ownership \
  --dense-task-weight 0 \
  --gradient-audit-steps 2,10,20,40,60 \
  --source-prediction-mode omitted_static \
  --evidence-profile behavior_discrimination_trial \
  --visual-audit-every 1 \
  --behavior-conditioned-prediction \
  >"$LOG" 2>&1
