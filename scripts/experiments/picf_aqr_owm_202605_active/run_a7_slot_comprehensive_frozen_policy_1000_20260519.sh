#!/usr/bin/env bash
set -euo pipefail

# Slot-comprehensive frozen-policy validation.
#
# This is deliberately not an anchor_only probe.  It freezes large pretrained
# perception and PaliGemma/action pressure, but leaves the full PICF
# slot/router/posterior/OEML stack trainable.
#
# Do not use --use-foundation-backbones here: that convenience flag sets
# semantic_trainable=True.  Spell out the real encoders manually.

REPO_ROOT="${REPO_ROOT:-/root/openpi_slot_quality_ea2c5f2}"
cd "${REPO_ROOT}"

EXP="${EXP:-picf_a7_slot_comprehensive_frozen_policy_1000_20260519}"
SIDECAR_ROOT="${SIDECAR_ROOT:-/mnt/picf_sidecars/contact_motion_mask_1000_20260518}"
LOG="/mnt/picf_run_logs/${EXP}.log"
CALVIN_ROOT="${CALVIN_ROOT:-/mnt/calvin_data/task_ABC_D}"
PYTHON_BIN="${PYTHON_BIN:-/root/openpi/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SEGMENTS="${SEGMENTS:-0,1,2,3}"

mkdir -p /mnt/picf_run_logs /mnt/checkpoints/picf_core

export PYTHONUNBUFFERED=1
export PYTHONPATH="src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" scripts/picf_core_train.py \
  --calvin-root "${CALVIN_ROOT}" \
  --backend dir \
  --split training \
  --calvin-segment-indices "${SEGMENTS}" \
  --checkpoint-base-dir /mnt/checkpoints/picf_core \
  --point-backbone sonata \
  --visual-mode encoder \
  --visual-feature-mode hierarchical \
  --tactile-mode encoder \
  --use-tactile \
  --semantic-mode paligemma \
  --semantic-source auto \
  --visual-checkpoint-path /root/openpi/checkpoints/foundation/vjepa2_1/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --tactile-checkpoint-path /root/openpi/checkpoints/foundation/anytouch2/checkpoint-4frames.pth \
  --tactile-backgrounds-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_backgrounds.npz \
  --tactile-calibration-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_fingertip_calibration.json \
  --tactile-contact-stats-path /mnt/checkpoints/picf_core/debug/tactile_calib_task_ABC_D_rgb_latent_full_v8/tactile_contact_stats.json \
  --sonata-checkpoint-path /root/openpi/src/pretrain/SpatialLM_Sonata_encoder.pth \
  --semantic-checkpoint-path /root/openpi/checkpoints/foundation/pi05_base_pytorch \
  --training-strategy fsdp_full_shard \
  --optimizer-sharding none \
  --optimizer-checkpoint-mode model-only \
  --perception-finetune-mode frozen \
  --picf-trainable-scope all \
  --action-horizon 16 \
  --max-points 1024 \
  --accum-steps 1 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-steps 50 \
  --weight-decay 1e-4 \
  --semantic-lr-scale 1e-6 \
  --unroll-steps 2 \
  --burnin-steps 1 \
  --burnin-mode state_only \
  --lambda-action-pos 0.0 \
  --lambda-action-rot 0.0 \
  --lambda-action-gripper 0.0 \
  --picf-action-prefix-stopgrad \
  --lambda-visual-latent 0.0 \
  --lambda-visual-real 0.0 \
  --lambda-tactile-real 0.0 \
  --lambda-point-real 0.0 \
  --lambda-semantic-future-aux 0.0 \
  --disable-aux-budgeting \
  --lambda-anchor-pv 0.0 \
  --lambda-anchor-object-pull 0.35 \
  --anchor-object-pull-sigma-m 0.04 \
  --anchor-object-pull-confirmation-threshold 0.02 \
  --anchor-object-pull-allowed-roles 1 \
  --anchor-object-pull-graph-weight 1.0 \
  --anchor-object-pull-posterior-weight 1.0 \
  --lambda-pv-weak 0.0 \
  --lambda-pt 0.0 \
  --lambda-vl-heatmap-task 0.0 \
  --lambda-vl-heatmap-effector 0.0 \
  --lambda-vl-heatmap-interaction 0.0 \
  --lambda-vl-point-consistency 0.0 \
  --lambda-vl-anchor-diversity 0.0 \
  --lambda-mapg-cycle 0.01 \
  --lambda-mapg-routing 0.0 \
  --lambda-mapg-support-diversity 0.005 \
  --lambda-mapg-geometry-diversity 0.0 \
  --lambda-slot-jepa 0.0 \
  --lambda-support-pred 0.0 \
  --lambda-binding-consistency 0.0 \
  --lambda-aqr-denoising 0.0 \
  --lambda-slot-quality 0.05 \
  --lambda-vcap-unexplained 0.0 \
  --lambda-vcap-duplicate 0.0 \
  --lambda-vcap-count 0.0 \
  --lambda-vcap-continuity 0.0 \
  --lambda-object-explanation-feature 0.0 \
  --lambda-object-explanation-point 0.02 \
  --lambda-object-explanation-contact 0.01 \
  --lambda-object-explanation-duplicate 0.01 \
  --lambda-object-explanation-background 0.005 \
  --tracklet-memory-enabled \
  --tracklet-read-weight 0.0 \
  --proposal-memory-enabled \
  --mvtrack-sidecar-root "${SIDECAR_ROOT}" \
  --mvtrack-sidecar-proposal-nearest-max-gap 8 \
  --proposal-age-decay-steps 8.0 \
  --proposal-shape-quality-enabled \
  --proposal-shape-area-min 0.002 \
  --proposal-shape-area-max 0.35 \
  --proposal-shape-aspect-min 0.20 \
  --proposal-context-quality-power 0.50 \
  --proposal-point-bridge-weight 0.50 \
  --proposal-point-bridge-edge-tau 0.02 \
  --proposal-anchor-seed-enabled \
  --proposal-anchor-seed-pre-reader-enabled \
  --proposal-anchor-seed-rows 2 \
  --proposal-anchor-seed-weight 0.85 \
  --proposal-anchor-seed-token-weight 0.35 \
  --proposal-anchor-seed-score-floor 0.05 \
  --proposal-anchor-seed-point-topk 128 \
  --proposal-anchor-seed-point-power 1.5 \
  --task-owner-proposal-point-bridge-weight 0.50 \
  --task-owner-bias-enabled \
  --task-owner-visual-bias-weight 0.20 \
  --task-owner-proposal-bias-weight 0.50 \
  --task-owner-proposal-point-bias-weight 0.75 \
  --task-owner-proposal-objectness-power 0.50 \
  --task-owner-proposal-static-only \
  --task-owner-proposal-topk 4 \
  --task-owner-proposal-score-floor 0.05 \
  --object-candidate-assignment-enabled \
  --object-candidate-assignment-temperature 0.35 \
  --object-candidate-background-prior 0.25 \
  --object-candidate-background-quality-weight 2.0 \
  --object-candidate-row-support-floor 0.01 \
  --object-candidate-eligible-roles 1 \
  --object-candidate-max-rows-per-candidate 1 \
  --object-candidate-row-capacity 1.25 \
  --object-candidate-row-capacity-iters 10 \
  --object-candidate-point-weight 1.0 \
  --object-candidate-proposal-weight 0.75 \
  --object-candidate-seed-weight 1.25 \
  --object-candidate-task-owner-weight 0.50 \
  --object-candidate-anchor-score-weight 1.0 \
  --object-candidate-point-mix 0.80 \
  --object-candidate-proposal-mix 0.35 \
  --object-candidate-min-shape-quality 0.01 \
  --object-candidate-owner-transport-enabled \
  --object-candidate-owner-roles 1 \
  --object-candidate-owner-min-share 0.75 \
  --object-candidate-owner-point-mix 1.0 \
  --tactile-attach-to-object-owner \
  --tactile-evidence-prob-floor 0.35 \
  --tactile-anchor-prob-on 0.55 \
  --aqr-role-layout object_only \
  --effector-persistent-anchors 0 \
  --effector-observation-anchors 0 \
  --task-effector-queries 0 \
  --aqr-ownership-prior-enabled \
  --aqr-ownership-prior-weight 0.70 \
  --aqr-ownership-point-prior-weight 0.70 \
  --aqr-ownership-point-prior-sigma-m 0.04 \
  --aqr-ownership-temporal-prior-weight 0.35 \
  --aqr-ownership-prior-uniform-mix 0.02 \
  --aqr-same-role-support-competition-enabled \
  --aqr-same-role-support-competition-weight 0.85 \
  --aqr-same-role-support-competition-iters 5 \
  --posterior-file-competition-enabled \
  --posterior-file-competition-min-per-role 1 \
  --posterior-file-competition-max-per-role 1 \
  --posterior-file-competition-min-support 0.02 \
  --posterior-file-competition-relative-score-threshold 0.0 \
  --posterior-file-competition-support-overlap-threshold 0.80 \
  --posterior-file-competition-geometry-duplicate-enabled \
  --posterior-file-competition-geometry-sigma-m 0.04 \
  --posterior-file-competition-geometry-threshold 0.70 \
  --posterior-birth-competition-enabled \
  --posterior-birth-competition-max-per-role 1 \
  --posterior-birth-competition-min-score 0.05 \
  --posterior-birth-competition-inactive-only \
  --posterior-birth-alpha-suppression-power 0.5 \
  --posterior-slot-identity-std 0.02 \
  --task-slot-identity-std 0.02 \
  --posterior-bootstrap-from-observation \
  --posterior-occupancy-prior-enabled \
  --posterior-occupancy-prior-weight 1.0 \
  --posterior-occupancy-prior-sigma-m 0.04 \
  --posterior-occupancy-prior-clip 4.0 \
  --observation-anchor-seed-point-mix 0.35 \
  --recycle-normalize-residual-summary \
  --recycle-residual-norm-mode layernorm \
  --posterior-slotwise-recycle-residual \
  --posterior-binding-signature-memory-enabled \
  --posterior-binding-signature-update-rate 0.20 \
  --posterior-binding-signature-update-max-rate 0.50 \
  --posterior-binding-signature-min-support 0.02 \
  --posterior-binding-signature-owner-weight 0.50 \
  --posterior-binding-signature-dispersion-gate-enabled \
  --posterior-binding-signature-measurement-min-std 0.05 \
  --posterior-binding-signature-measurement-margin-min 0.25 \
  --posterior-binding-signature-measurement-margin-temperature 0.10 \
  --posterior-owner-active-gate-enabled \
  --posterior-owner-active-min 0.30 \
  --posterior-owner-active-bias -10000.0 \
  --posterior-owner-transport-enabled \
  --posterior-owner-transport-roles 1 \
  --posterior-owner-transport-max-per-role 1 \
  --posterior-owner-transport-max-rate 0.85 \
  --posterior-owner-transport-precision-gain 8.0 \
  --posterior-owner-transport-min-mass 0.01 \
  --posterior-owner-transport-assignment-floor 0.50 \
  --posterior-owner-transport-reliability-floor 0.50 \
  --posterior-owner-transport-covariance-scale 0.50 \
  --posterior-owner-transport-inactive-prior 0.35 \
  --posterior-owner-transport-activates-file \
  --posterior-owner-transport-active-threshold 0.05 \
  --no-legacy-local-refinement-opt-in \
  --no-local-refinement-enabled \
  --local-refinement-topk 0 \
  --local-refinement-weight 0.0 \
  --local-refinement-binding-weight 0.0 \
  --aqr-active-slot-filter-enabled \
  --aqr-active-slot-min-per-role 1 \
  --aqr-active-slot-max-per-role 2 \
  --aqr-active-slot-min-confidence 0.05 \
  --aqr-active-slot-overlap-threshold 0.40 \
  --aqr-active-slot-relative-score-threshold 0.80 \
  --aqr-active-slot-geometry-duplicate-enabled \
  --aqr-active-slot-geometry-duplicate-sigma-m 0.05 \
  --aqr-active-slot-geometry-duplicate-threshold 0.45 \
  --no-aqr-context-slot-enabled \
  --grad-clip-mode fixed \
  --grad-clip-norm 5.0 \
  --anchor-overlay-interval 50 \
  --anchor-overlay-max-anchors 64 \
  --anchor-overlay-dump-signatures \
  --log-interval 50 \
  --save-interval 2500 \
  --keep-last-checkpoints 3 \
  --progress \
  --wandb-mode disabled \
  --overwrite \
  --exp-name "${EXP}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-1000}" 2>&1 | tee "${LOG}"
