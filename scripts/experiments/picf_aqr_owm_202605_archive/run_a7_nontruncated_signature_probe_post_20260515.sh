#!/usr/bin/env bash
set -euo pipefail
cd /root/openpi_posterior_vla_clean
EXP=${EXP:-picf_a7_diag_nontruncated_signature_u2b1_180_20260515}
RUN_DIR=/mnt/checkpoints/picf_core/picf_core/${EXP}
export PYTHONPATH=src:${PYTHONPATH:-}
PYTHON_BIN=${PYTHON_BIN:-/root/openpi/.venv/bin/python}
"${PYTHON_BIN}" scripts/picf_owm_same_object_probe.py \
  --anchor-overlays "${RUN_DIR}/anchor_overlays" \
  --overlay-source posterior \
  --quadratic-probe all \
  --quadratic-probe-epochs 120 \
  --quadratic-probe-max-pairs 20000 \
  --output "${RUN_DIR}/same_object_probe_overlay.json"
cat "${RUN_DIR}/same_object_probe_overlay.json"
