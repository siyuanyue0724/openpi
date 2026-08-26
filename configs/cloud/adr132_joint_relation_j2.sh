#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr132-j0-20260804}
RUN_DIR=${RUN_DIR:?RUN_DIR must name one fresh persistent J2 run directory}
LOG=${LOG:?LOG must name one persistent log file}
REPRESENTATION_BASELINE=${REPRESENTATION_BASELINE:?REPRESENTATION_BASELINE must name the current joint-estimator step-zero baseline}
REPRESENTATION_BASELINE_SHA256=${REPRESENTATION_BASELINE_SHA256:?REPRESENTATION_BASELINE_SHA256 must bind that baseline}

mkdir -p "$RUN_DIR" "$(dirname "$LOG")"
echo $$ > "$RUN_DIR/launcher.pid"

exec env -u PYTORCH_CUDA_ALLOC_CONF -u PYTORCH_ALLOC_CONF \
  -u PICF_LINGBOT_TRAINING_AUTHORIZATION \
  -u PICF_LINGBOT_TRAINING_AUTHORIZATION_SHA256 \
  PYTHONPATH="$WORKTREE/src:$WORKTREE" \
  CUDA_VISIBLE_DEVICES=0,1 \
  /opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "$WORKTREE/tools/run_lingbot_vla2_native_full.py" \
  --phase fresh \
  --training-stage representation \
  --checkpoint-publication never \
  --representation-split /mnt/picf-next/probes/representation-k8-reset-mixture-adr121-e6dbcf6-20260731.split.json \
  --representation-split-sha256 9e35c8aab44d3aa6949e188ba6e0f30308b36d4a52718f0d21a6ca673339ccef \
  --fixed-observation-pair-plan /mnt/picf-next/probes/adr123-d7173ec-fixed-x-pair-plan.json \
  --fixed-observation-pair-plan-sha256 06858f2351afbee99cbc87b240c00aceff1bfdb71b20d75abd6364da34a6570d \
  --fixed-observation-training-audit /mnt/picf-next/probes/adr123-b932f0d-training-resets-full/token_grid.json \
  --fixed-observation-training-audit-sha256 da825e624580e10f1433928235279426c636b226ef4961b9dd7b8d79fb8eb535 \
  --fixed-observation-evaluation-plan /mnt/picf-next/probes/adr123-d7173ec-fixed-x-evaluation-plan.json \
  --fixed-observation-evaluation-plan-sha256 a2ee04954b0b6afc9bf10e956c967dfb9ad57c4a5fb138f9e16302aa22a51ca2 \
  --fixed-observation-validation-audit /mnt/picf-next/probes/adr123-b932f0d-validation-resets-full/token_grid.json \
  --fixed-observation-validation-audit-sha256 28b1cd38d562490cc66cde1c93abf6e8cc4021cbd52aa2511af13e773d4f61f3 \
  --fixed-observation-heldout-audit /mnt/picf-next/probes/adr123-b932f0d-heldout-resets-full/token_grid.json \
  --fixed-observation-heldout-audit-sha256 fda9027fc27e455d99675e1fc7739ed7a06c391a2ec1af95b39cdbe6666c50cf \
  --representation-evaluation-plan /mnt/picf-next/probes/representation-k8-reset-mixture-adr121-e6dbcf6-20260731.reset-evaluation-plan.json \
  --representation-evaluation-plan-sha256 7d2e05143c62acc8dd0f292a40f736e314001883a4ec29a2e44ae8f174c11c5f \
  --representation-warm-evaluation-plan /mnt/picf-next/probes/representation-k8-reset-mixture-adr121-e6dbcf6-20260731.warm-evaluation-plan.json \
  --representation-warm-evaluation-plan-sha256 4991d6c1a27be243b82c5fd5793dc89b6cb3dbcf897204d4cbb8ccaf603240c6 \
  --representation-evaluation-baseline "$REPRESENTATION_BASELINE" \
  --representation-evaluation-baseline-sha256 "$REPRESENTATION_BASELINE_SHA256" \
  --representation-evaluation-steps 0,200 \
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
  --predictive-cache-root /mnt/picf-next/cache/adr132-k8-current-estimator-predictive-20260804T094015+0800 \
  --predictive-cache-build-report /mnt/picf-next/cache/adr132-k8-current-estimator-predictive-20260804T094015+0800.build_report.json \
  --predictive-cache-build-report-sha256 cffffa03453456e9a6df6ce500d131856791c0dd8969b33a40e28e8649c247ba \
  --predictive-target-audit /mnt/picf-next/runs/adr132-k8-current-estimator-cache-20260804T094015+0800/predictive-target-audit.json \
  --predictive-target-audit-sha256 9dbd7b2ffa14cbb8ff19494db94fa35ea562f1c766b861c0d28cd42d737acf10 \
  --predictive-teacher-causality-audit /mnt/picf-next/runs/adr132-k8-current-estimator-cache-20260804T094015+0800/teacher-causality-audit.json \
  --predictive-teacher-causality-audit-sha256 db291d41a15e86f155aa1bcb335b37275f0d26c0c1f51fc5559c4120af1191d7 \
  --predictive-temporal-audit /mnt/picf-next/runs/adr132-k8-current-estimator-cache-20260804T094015+0800/predictive-temporal-audit.json \
  --predictive-temporal-audit-sha256 3b7e9a5aed48d7930983d545211aa8141f534fc1f207d5a28d6670647b8bb987 \
  --current-grid-cache-root /mnt/picf-next/cache/adr132-k8-current-estimator-current-grid-20260804T094015+0800 \
  --current-grid-cache-build-report /mnt/picf-next/cache/adr132-k8-current-estimator-current-grid-20260804T094015+0800.build_report.json \
  --current-grid-cache-build-report-sha256 b635eb35d1c1313bc22b1d40127e7379ed430a03e8e6188ba6bb589b25bae77e \
  --run-dir "$RUN_DIR" \
  --load-global-step 0 \
  --invocation-steps 200 \
  --total-planned-steps 200 \
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
  --gradient-audit-steps 18,34,50,100,200 \
  --source-prediction-mode omitted_static \
  --evidence-profile acceptance \
  --visual-audit-every 1 \
  --task-relation-estimator host_native_factorized_task_physical_ownership \
  >"$LOG" 2>&1
