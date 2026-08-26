#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TRAINING_RUN_ROOT ACCEPTANCE_RUN_ROOT" >&2
  exit 2
fi

training_root=$1
acceptance_root=$2
repository_root=${PICF_REPOSITORY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
evaluation_scenes_per_partition=${PICF_G3_EVALUATION_SCENES_PER_PARTITION:-1}

case "$evaluation_scenes_per_partition" in
  1) evaluation_scope=quick ;;
  4) evaluation_scope=full ;;
  *)
    echo "PICF_G3_EVALUATION_SCENES_PER_PARTITION must be 1 or 4" >&2
    exit 1
    ;;
esac

if [[ "$training_root" != /mnt/* || "$acceptance_root" != /mnt/* ]]; then
  echo "ADR165 acceptance inputs and outputs must live under /mnt" >&2
  exit 1
fi
if [[ -e "$acceptance_root" || -L "$acceptance_root" ]]; then
  echo "ADR165 acceptance output must be one absent direct path under /mnt" >&2
  exit 1
fi
for launcher in \
  "$repository_root/adr165/run_ltop_g3_mediator_cold_action_2gpu.sh" \
  "$repository_root/adr165/run_ltop_g3_mediator_retention_2gpu.sh"; do
  if [[ ! -x "$launcher" ]]; then
    echo "required ADR165 launcher is absent or not executable: $launcher" >&2
    exit 1
  fi
done

mkdir -p "$acceptance_root"
"$repository_root/adr165/run_ltop_g3_mediator_cold_action_2gpu.sh" \
  "$training_root" "$acceptance_root/action-$evaluation_scope"
"$repository_root/adr165/run_ltop_g3_mediator_retention_2gpu.sh" \
  "$training_root" "$acceptance_root/retention"

printf 'ADR165 cold acceptance passed: action=%s retention=%s\n' \
  "$acceptance_root/action-$evaluation_scope" "$acceptance_root/retention"
