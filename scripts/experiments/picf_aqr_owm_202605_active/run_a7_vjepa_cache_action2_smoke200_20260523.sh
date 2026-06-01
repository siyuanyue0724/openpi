#!/usr/bin/env bash
set -euo pipefail

# 2026-05-23 action-from-start comparison for the V-JEPA cache smoke.
#
# This is intentionally identical to run_a7_vjepa_cache_action05_smoke200_20260523
# except for ACTION_LOSS_WEIGHT=2.0.  It answers whether starting at the legacy
# action scale improves action loss without immediately breaking slot structure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP="${EXP:-picf_a7_vjepa_cache_action2_smoke200_20260523}"
export ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-2.0}"

exec "${SCRIPT_DIR}/run_a7_vjepa_cache_action05_smoke200_20260523.sh"
