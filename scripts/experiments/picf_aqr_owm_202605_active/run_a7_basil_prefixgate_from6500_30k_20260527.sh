#!/usr/bin/env bash
set -euo pipefail

# BASL-style structural stabilization for the action rebound failure.
#
# Difference from the 2026-05-26 phase-stabilized control:
#   1. PICF extra-prefix is injected through a fixed output gate.
#   2. PICF core LR uses block-alternating updates: short core update bursts
#      followed by stationary-prefix policy adaptation windows.
#
# This targets the measured root cause directly:
#   action loss rebounds while active/downstream overlap and structural losses
#   remain healthy, meaning the action-visible prefix is moving faster than the
#   PI0.5 action path can track.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PHASE1_EXP="${PHASE1_EXP:-picf_a7_basil_prefixgate_p1_policyonly_from6500_6800_20260527}"
export PHASE2_EXP="${PHASE2_EXP:-picf_a7_basil_prefixgate_p2_blockcore_action2_30k_20260527}"
export PHASE1_END_STEP="${PHASE1_END_STEP:-6800}"
export PHASE2_END_STEP="${PHASE2_END_STEP:-30000}"

# Keep the amplitude high enough for PICF to remain useful, but reduce the
# action-visible perturbation from a moving prefix.  This is an interface
# contract, not a new object supervision signal.
export ACTION_PREFIX_OUTPUT_GATE="${ACTION_PREFIX_OUTPUT_GATE:-0.70}"

# Update PICF core in short bursts, then give the action path a stationary
# prefix recovery interval.  Average core pressure is intentionally lower than
# continuous core0.005 while still preserving cotrain.
export PICF_CORE_LR_SCALE="${PICF_CORE_LR_SCALE:-0.01}"
export PICF_CORE_LR_RUNTIME_MODE="${PICF_CORE_LR_RUNTIME_MODE:-block_alternating}"
export PICF_CORE_LR_BLOCK_START_STEP="${PICF_CORE_LR_BLOCK_START_STEP:-${PHASE1_END_STEP}}"
export PICF_CORE_LR_BLOCK_CYCLE_STEPS="${PICF_CORE_LR_BLOCK_CYCLE_STEPS:-200}"
export PICF_CORE_LR_BLOCK_ACTIVE_STEPS="${PICF_CORE_LR_BLOCK_ACTIVE_STEPS:-40}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
export LOG_INTERVAL="${LOG_INTERVAL:-50}"
export ANCHOR_OVERLAY_INTERVAL="${ANCHOR_OVERLAY_INTERVAL:-100}"

exec "${SCRIPT_DIR}/run_a7_phase_stabilized_from6500_30k_20260526.sh"
