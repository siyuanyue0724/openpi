#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTPUT_LOG DEADLINE_EPOCH" >&2
  exit 2
fi

output_log=$1
deadline_epoch=$2

mkdir -p "$(dirname "$output_log")"

while (( $(date +%s) < deadline_epoch )); do
  now=$(date '+%F %T %Z')
  {
    printf '[%s] HOURLY ARCHITECTURE AUDIT\n' "$now"
    printf '%s\n' '1. No simplification: preserve complete successful upstream code, topology, losses, optimizer groups, preprocessing, and interfaces; name every unavoidable adaptation.'
    printf '%s\n' '2. Stop scientific failures immediately: do not wait for round-number steps, tune a scalar, or add a local patch after the registered effect has failed.'
    printf '%s\n' '3. Promotion requires a material whole-curve advantage over the exact LingBot control; about 1% or regression is rejection.'
    printf '%s\n' '4. One large shared model owns semantics, multimodal binding, temporal correction, uncertainty, and action; transport projections are not semantic decision makers.'
    printf '%s\n' '5. Preserve source hashes, negative evidence, checkpoints worth retaining, and recovery commands under /mnt.'
    printf '\n'
  } | tee -a "$output_log"

  now_epoch=$(date +%s)
  remaining=$(( deadline_epoch - now_epoch ))
  (( remaining > 0 )) || break
  (( remaining < 3600 )) && sleep "$remaining" || sleep 3600
done
