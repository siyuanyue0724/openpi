#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ACCEPTANCE_ROOT RESTART_ROOT LONG_ROOT" >&2
  exit 2
fi

acceptance_root=$1
restart_root=$2
long_root=$3
repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${PICF_PYTHON_BIN:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
expected_source_commit=${PICF_ADR170_EXPECTED_SOURCE_COMMIT:?PICF_ADR170_EXPECTED_SOURCE_COMMIT must name the approved exact commit}

for path in "$acceptance_root" "$restart_root" "$long_root"; do
  if [[ "$path" != /mnt/* ]]; then
    echo "ADR170 post-acceptance roots must live under /mnt: $path" >&2
    exit 1
  fi
done

watchdog=$repository_root/tools/watch_ltop_core_long_health.py
semantic_validator=$repository_root/tools/compose_ltop_g3_source_aligned_acceptance.py
action_validator=$repository_root/tools/validate_ltop_g3_cold_action_evidence.py
restart_launcher=$repository_root/adr164/run_ltop_core_restart_smoke_2gpu.sh
long_launcher=$repository_root/adr164/run_ltop_core_long_2gpu.sh

if [[ ! -x "$python_bin" ]]; then
  echo "required ADR170 Python runtime is absent: $python_bin" >&2
  exit 1
fi
if [[ ! "$expected_source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "PICF_ADR170_EXPECTED_SOURCE_COMMIT must be one lowercase 40-hex SHA" >&2
  exit 1
fi
if ! inside_worktree=$(git -C "$repository_root" rev-parse --is-inside-work-tree 2>/dev/null); then
  echo "ADR170 source root is not a readable Git worktree" >&2
  exit 1
fi
if [[ "$inside_worktree" != true ]]; then
  echo "ADR170 source root is not inside a Git worktree" >&2
  exit 1
fi
if ! actual_source_commit=$(git -C "$repository_root" rev-parse --verify 'HEAD^{commit}' 2>/dev/null); then
  echo "ADR170 source commit cannot be resolved" >&2
  exit 1
fi
if [[ "$actual_source_commit" != "$expected_source_commit" ]]; then
  echo "ADR170 source HEAD differs from the operator-approved commit" >&2
  exit 1
fi
if ! source_status=$(git -C "$repository_root" status --porcelain=v1 --untracked-files=all); then
  echo "ADR170 source worktree status cannot be read" >&2
  exit 1
fi
if [[ -n "$source_status" ]]; then
  echo "ADR170 post-acceptance promotion requires an exact clean source checkout" >&2
  exit 1
fi
for path in "$watchdog" "$semantic_validator" "$action_validator"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "required ADR170 post-acceptance tool is absent or indirect: $path" >&2
    exit 1
  fi
done
for path in "$restart_launcher" "$long_launcher"; do
  if [[ ! -x "$path" || -L "$path" ]]; then
    echo "required LTOP launcher is absent, indirect, or not executable: $path" >&2
    exit 1
  fi
done

export PYTHONPATH="$repository_root:$repository_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$python_bin" "$watchdog" chain \
  --acceptance-root "$acceptance_root" \
  --restart-root "$restart_root" \
  --long-root "$long_root" \
  --repository-root "$repository_root" \
  --expected-repository-commit "$expected_source_commit" \
  --semantic-validator "$semantic_validator" \
  --action-validator "$action_validator" \
  --restart-launcher "$restart_launcher" \
  --long-launcher "$long_launcher" \
  --restart-timeout-seconds "${PICF_ADR170_RESTART_TIMEOUT_SECONDS:-14400}" \
  --initial-grace-seconds "${PICF_LTOP_INITIAL_GRACE_SECONDS:-3600}" \
  --checkpoint-boundary-grace-seconds \
    "${PICF_LTOP_CHECKPOINT_BOUNDARY_GRACE_SECONDS:-3600}" \
  --stale-threshold-seconds "${PICF_LTOP_STALE_THRESHOLD_SECONDS:-900}" \
  --poll-interval-seconds "${PICF_LTOP_WATCHDOG_POLL_SECONDS:-30}" \
  --status-heartbeat-seconds "${PICF_LTOP_WATCHDOG_HEARTBEAT_SECONDS:-300}" \
  --termination-grace-seconds "${PICF_LTOP_TERMINATION_GRACE_SECONDS:-120}"
