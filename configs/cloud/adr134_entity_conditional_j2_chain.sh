#!/usr/bin/env bash
set -euo pipefail

WORKTREE=${PICF_WORKTREE:-/mnt/picf-next/worktrees/adr132-j0-20260804}
STAMP=${STAMP:-$(date +%Y%m%dT%H%M%S%z)}

BASE_RUN=${BASE_RUN:-/mnt/picf-next/runs/adr134-entity-step0-baseline-v1-$STAMP}
BASE_LOG=${BASE_LOG:-/mnt/picf-next/logs/adr134-entity-step0-baseline-v1-$STAMP.log}
BASE_RESUME_LOG=${BASE_RESUME_LOG:-/mnt/picf-next/logs/adr134-entity-step0-baseline-v1-$STAMP.resume.log}
BASELINE=${BASELINE:-/mnt/picf-next/probes/adr134-entity-step0-baseline-v1-$STAMP.json}
BASELINE_REPORT=${BASELINE_REPORT:-$BASELINE.build_report.json}
J2_RUN=${J2_RUN:-/mnt/picf-next/runs/adr134-entity-j2-v1-$STAMP}
J2_LOG=${J2_LOG:-/mnt/picf-next/logs/adr134-entity-j2-v1-$STAMP.log}
STATE=${STATE:-/mnt/picf-next/runs/adr134-entity-j2-chain-v1-$STAMP.state}

SOURCE_PLAN=/mnt/picf-next/probes/representation-evaluation-plan-3b6d367-v3.json
SOURCE_PLAN_SHA256=9518c1e646bb3a20fd26ff9b5011e7d2d522f88f367c953d17222c0d8ec960f4
CANDIDATE_PLAN=/mnt/picf-next/probes/representation-k8-reset-mixture-adr121-e6dbcf6-20260731.reset-evaluation-plan.json
PYTHON=/opt/picf-miniconda3/envs/picf-lingbot-vla2/bin/python

for path in "$BASE_RUN" "$BASE_LOG" "$BASE_RESUME_LOG" "$BASELINE" "$BASELINE_REPORT" "$J2_RUN" "$J2_LOG" "$STATE"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing to reuse J2 chain artifact: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$BASE_LOG")" "$(dirname "$BASELINE")" "$(dirname "$STATE")"
printf 'phase=baseline_running\nstamp=%s\nbase_run=%s\nbase_log=%s\nbase_resume_log=%s\nbaseline=%s\nj2_run=%s\nj2_log=%s\n' \
  "$STAMP" "$BASE_RUN" "$BASE_LOG" "$BASE_RESUME_LOG" "$BASELINE" "$J2_RUN" "$J2_LOG" >"$STATE"

on_error() {
  status=$?
  printf 'phase=failed\nexit_status=%s\n' "$status" >>"$STATE"
  exit "$status"
}
trap on_error ERR

RUN_DIR="$BASE_RUN" LOG="$BASE_LOG" PICF_WORKTREE="$WORKTREE" \
  "$WORKTREE/configs/cloud/adr134_entity_conditional_j2_baseline.sh"

SNAPSHOT="$BASE_RUN/representation_evaluations/global_step_0/representation_evaluation_snapshot.json"
VISUAL_ROOT="$BASE_RUN/representation_evaluations/global_step_0"
SNAPSHOT_SHA256=$(sha256sum "$SNAPSHOT" | awk '{print $1}')

PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" \
  "$WORKTREE/tools/build_lingbot_representation_evaluation_baseline.py" \
  --source-snapshot "$SNAPSHOT" \
  --source-snapshot-sha256 "$SNAPSHOT_SHA256" \
  --source-evaluation-plan "$SOURCE_PLAN" \
  --source-evaluation-plan-sha256 "$SOURCE_PLAN_SHA256" \
  --source-visual-root "$VISUAL_ROOT" \
  --output "$BASELINE" >"$BASELINE_REPORT"

BASELINE_SHA256=$(sha256sum "$BASELINE" | awk '{print $1}')
PYTHONPATH="$WORKTREE/src:$WORKTREE" "$PYTHON" - "$BASELINE" "$CANDIDATE_PLAN" <<'PY'
from pathlib import Path
import sys

from picf_next.lingbot_native.representation_baseline import (
    load_representation_evaluation_baseline,
    validate_representation_baseline_plan,
)
from picf_next.lingbot_native.representation_evaluation import RepresentationEvaluationPlan

baseline = load_representation_evaluation_baseline(Path(sys.argv[1]))
candidate = RepresentationEvaluationPlan.load(Path(sys.argv[2]))
validate_representation_baseline_plan(baseline, candidate_plan=candidate)
if baseline["sample_count"] != 136:
    raise RuntimeError("ADR-134 J2 baseline sample coverage changed")
print(baseline["artifact_sha256"])
PY

printf 'phase=baseline_resume_running\n' >>"$STATE"
PHASE=resume LOAD_GLOBAL_STEP=1 RUN_DIR="$BASE_RUN" LOG="$BASE_RESUME_LOG" \
  PICF_WORKTREE="$WORKTREE" \
  "$WORKTREE/configs/cloud/adr134_entity_conditional_j2_baseline.sh"

printf 'phase=j2_running\nbaseline_file_sha256=%s\n' "$BASELINE_SHA256" >>"$STATE"
RUN_DIR="$J2_RUN" LOG="$J2_LOG" PICF_WORKTREE="$WORKTREE" \
  REPRESENTATION_BASELINE="$BASELINE" \
  REPRESENTATION_BASELINE_SHA256="$BASELINE_SHA256" \
  "$WORKTREE/configs/cloud/adr134_entity_conditional_j2.sh"

printf 'phase=complete\n' >>"$STATE"
