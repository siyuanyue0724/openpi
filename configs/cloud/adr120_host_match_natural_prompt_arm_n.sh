#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=/mnt/picf-next/runs/representation-host-native-match-natural-prompt-arm-n-20260731
LOG=/mnt/picf-next/logs/representation-host-native-match-natural-prompt-arm-n-20260731.log
echo $$ > "$RUN_DIR/launcher.pid"
exec env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_ALLOC_CONF \
  PYTHONPATH=/mnt/picf-next/audit-checkouts/host-native-match-1d45cb31-20260730/src:/mnt/picf-next/audit-checkouts/host-native-match-1d45cb31-20260730 \
  CUDA_VISIBLE_DEVICES=0,1 \
  /opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  /mnt/picf-next/audit-checkouts/host-native-match-1d45cb31-20260730/tools/run_lingbot_vla2_native_full.py \
  --phase fresh \
  --training-stage representation \
  --checkpoint-publication never \
  --representation-split /mnt/picf-next/probes/representation-k8-reference-bank-gate5470c5a-20260729.split.json \
  --representation-split-sha256 6fba907644ced21bc0b21972e0c385c8beb5c741e1e30ee79f8bed121b0b3c26 \
  --representation-evaluation-plan /mnt/picf-next/probes/representation-k8-reference-bank-gate5470c5a-20260729.evaluation-plan.json \
  --representation-evaluation-plan-sha256 68b6e2c458d4d44077aab33c08954affe51f358dae8bdb3d9ede9e2b0eabdde5 \
  --representation-evaluation-baseline /mnt/picf-next/probes/representation-host-native-match-1d45cb31-step0-baseline-20260730.json \
  --representation-evaluation-baseline-sha256 682290e9694c28641cc55cf6a4ae1c9f938ef60f6c9761937ff03b3abe626d23 \
  --representation-evaluation-steps 0,200 \
  --source-checkout /mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2 \
  --patch /mnt/picf-next/audit-checkouts/host-native-match-1d45cb31-20260730/references/patches/lingbot_vla2_picf_native.patch \
  --training-config /mnt/picf-next/source-checkouts/lingbot-vla-v2-12943f2/configs/vla/robotwin/robotwin.yaml \
  --robot-config /mnt/picf-next/audit-checkouts/host-native-match-1d45cb31-20260730/configs/lingbot/calvin_robot.yaml \
  --data-config /mnt/picf-next/audit-checkouts/host-native-match-1d45cb31-20260730/configs/lingbot/calvin_data.json \
  --checkpoint-dir /root/picf-runtime-42a7ad9/models/lingbot-vla-v2-6b \
  --processor-dir /root/picf-runtime-42a7ad9/models/qwen3-vl-4b-instruct \
  --dataset-split /mnt/calvin_data/task_ABC_D/training \
  --dataset-manifest /mnt/picf-next/manifests/calvin-training-files.json \
  --norm-stats /mnt/picf-next/manifests/calvin-lingbot-norm-stats.json \
  --physical-sidecar-root /mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z \
  --physical-sidecar-manifest-sha256 0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4 \
  --physical-visual-acceptance /mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json \
  --physical-visual-acceptance-sha256 4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86 \
  --predictive-cache-root /mnt/picf-next/cache/representation-k8-gate5470c5a-predictive \
  --predictive-cache-build-report /mnt/picf-next/cache/representation-k8-gate5470c5a-predictive.build_report.json \
  --predictive-cache-build-report-sha256 da8b7f18106fd6ea001f1565202fd2c03b859a3f7c45b2a9114e956176b96e70 \
  --predictive-target-audit /mnt/picf-next/audits/representation-k8-gate5470c5a-predictive-target.json \
  --predictive-target-audit-sha256 46438d19f5c594d05d57c98aac2e4b899e4776edfab2b86f5a1c8a55b4d5520e \
  --predictive-teacher-causality-audit /mnt/picf-next/audits/representation-k8-gate5470c5a-teacher-causality.json \
  --predictive-teacher-causality-audit-sha256 c6ac2f72992bca3ad7b5ded5fce9ac3ecd7bd2b1fdab10777a0097a4abecdcdd \
  --predictive-temporal-audit /mnt/picf-next/audits/representation-k8-gate5470c5a-temporal.json \
  --predictive-temporal-audit-sha256 7706fda2960fbe7774e5fa734961e0fbaea09021aa2c65ee5d1ea49b2eb58b04 \
  --current-grid-cache-root /mnt/picf-next/cache/representation-k8-gate5470c5a-current-grid \
  --current-grid-cache-build-report /mnt/picf-next/cache/representation-k8-gate5470c5a-current-grid.build_report.json \
  --current-grid-cache-build-report-sha256 2616f7a7527831a4f1c0f8364a413b495be17e586c56e78a8f87885583bc4194 \
  --run-dir "$RUN_DIR" \
  --load-global-step 0 \
  --invocation-steps 200 \
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
  --gradient-audit-steps 9,17,20,50,100,200 \
  --source-prediction-mode omitted_static \
  --visual-audit-every 1 \
  --task-relation-estimator local_balanced_sigmoid \
  >"$LOG" 2>&1
