#!/usr/bin/env bash
set -euo pipefail

EXP="${EXP:-picf_a7_contact_motion_mask12000_20260519}"
REPO="${REPO:-/root/openpi_posterior_vla_clean}"
LOG_DIR="${LOG_DIR:-/mnt/picf_run_logs}"
CALVIN_ROOT="${CALVIN_ROOT:-/mnt/calvin_data/task_ABC_D}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/picf_sidecars/contact_motion_mask_12000_20260519}"
TARGET_FRAMES="${TARGET_FRAMES:-12000}"
PY_BIN="${PY_BIN:-/usr/bin/python}"

mkdir -p "${LOG_DIR}"
cd "${REPO}"
export PYTHONPATH="scripts:src:${PYTHONPATH:-}"

{
  echo "[$(date -Iseconds)] starting ${EXP}"
  echo "repo=${REPO}"
  echo "calvin_root=${CALVIN_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "target_frames=${TARGET_FRAMES}"
  "${PY_BIN}" scripts/picf_contact_motion_sidecar_precompute.py \
    --calvin-root "${CALVIN_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --split training \
    --target-frames "${TARGET_FRAMES}" \
    --max-frames-per-segment 96 \
    --static-stride 4 \
    --gripper-stride 2 \
    --top-fraction 0.020 \
    --min-top-points 24 \
    --min-score 0.015 \
    --box-pad-px 4.0 \
    --max-proposals-per-frame 3 \
    --component-radius-px 10.0 \
    --component-min-points 6 \
    --box-percentile-low 12 \
    --box-percentile-high 88 \
    --mask-samples-per-proposal 96 \
    --preview-count 128 \
    --skip-existing
  echo "[$(date -Iseconds)] finished ${EXP}"
} 2>&1 | tee -a "${LOG_DIR}/${EXP}.log"
