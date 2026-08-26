#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 fresh|resume RUN_DIR STOP_AFTER_STEP LOAD_GLOBAL_STEP" >&2
  exit 2
fi

PHASE=$1
RUN_DIR=$2
STOP_AFTER_STEP=$3
LOAD_GLOBAL_STEP=$4

for value in "$STOP_AFTER_STEP" "$LOAD_GLOBAL_STEP"; do
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "training steps must be canonical non-negative decimal integers" >&2
    exit 2
  }
done
[[ "$STOP_AFTER_STEP" -gt "$LOAD_GLOBAL_STEP" && "$STOP_AFTER_STEP" -le 30000 ]] || {
  echo "stop step must be greater than load step and no greater than 30000" >&2
  exit 2
}
[[ "$RUN_DIR" == /mnt/* ]] || {
  echo "ADR-149 run directory must be persistent under /mnt" >&2
  exit 2
}

case "$PHASE" in
  fresh)
    [[ "$LOAD_GLOBAL_STEP" -eq 0 && "$STOP_AFTER_STEP" -le 2000 && ! -e "$RUN_DIR" ]] || {
      echo "fresh ADR-149 requires step zero, an absent run directory, and a gate at or before 2000" >&2
      exit 2
    }
    ;;
  resume)
    [[ "$LOAD_GLOBAL_STEP" -gt 0 && $((LOAD_GLOBAL_STEP % 2000)) -eq 0 ]] || {
      echo "ADR-149 resume requires a positive 2000-step boundary" >&2
      exit 2
    }
    [[ -d "$RUN_DIR/checkpoints/global_step_$LOAD_GLOBAL_STEP" ]] || {
      echo "ADR-149 resume checkpoint is absent" >&2
      exit 1
    }
    ;;
  *)
    echo "phase must be fresh or resume" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next/manifests/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next/manifests/calvin-lingbot-norm-stats.json}
SIDECAR=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
SIDECAR_SHA=${PICF_CALVIN_PHYSICAL_SIDECAR_SHA256:-0198b9d184069f40f1804de411e25ffb3f3a446fcd61d5dd619e944488244ed4}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json}
VISUAL_ACCEPTANCE_SHA=${PICF_CALVIN_VISUAL_ACCEPTANCE_SHA256:-4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86}
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr149/handoff_20260809}
STREAM_PLAN=$HANDOFF/four-gpu-30k.physical.stream-plan.json
REPRESENTATION_SPLIT=$HANDOFF/four-gpu-30k.physical.split.json
EVALUATION_PLAN=$HANDOFF/four-gpu-30k.physical.evaluation-plan.json
CURRENT_CACHE=${PICF_CURRENT_CACHE_ROOT:-/mnt/picf-next/adr149/full-picf-30k/cache/current-filter-dino-physical-v1}
CURRENT_REPORT=${PICF_CURRENT_CACHE_BUILD_REPORT:-${CURRENT_CACHE}.build_report.json}
FROZEN_MANIFEST=${PICF_ADR149_FROZEN_MANIFEST:-$HANDOFF/frozen_inputs.manifest.json}
FROZEN_INPUTS=${PICF_ADR149_FROZEN_INPUTS:-$HANDOFF/frozen_inputs.sha256}

for path in "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR" "$CURRENT_CACHE"; do
  [[ -d "$path" ]] || {
    echo "required directory is absent: $path" >&2
    exit 1
  }
done
for path in \
  "$PYTHON" \
  "$DATASET_MANIFEST" \
  "$NORM_STATS" \
  "$SIDECAR/manifest.json" \
  "$VISUAL_ACCEPTANCE" \
  "$STREAM_PLAN" \
  "$REPRESENTATION_SPLIT" \
  "$EVALUATION_PLAN" \
  "$CURRENT_CACHE/manifest.json" \
  "$CURRENT_REPORT" \
  "$FROZEN_MANIFEST" \
  "$FROZEN_INPUTS"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required file is absent or indirect: $path" >&2
    exit 1
  }
done

sha256sum --check --strict "$FROZEN_INPUTS"
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-149 execution requires an exact clean checkout" >&2
  exit 1
}

"$PYTHON" - \
  "$FROZEN_MANIFEST" \
  "$(git -C "$REPO" rev-parse HEAD)" \
  "$(git -C "$SOURCE" rev-parse HEAD)" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema") != "picf-next.adr149-frozen-inputs/v1":
    raise SystemExit("ADR-149 frozen input manifest schema differs")
if manifest.get("implementation_commit") != sys.argv[2]:
    raise SystemExit("ADR-149 implementation commit changed after input freezing")
if manifest.get("lingbot_source_commit") != sys.argv[3]:
    raise SystemExit("ADR-149 LingBot source commit changed after input freezing")
print("ADR-149 frozen implementation/source commits=PASS")
PY

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 4 ]] || {
  echo "ADR-149 requires exactly four visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-149 requires four A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

CURRENT_REPORT_SHA=$(sha256sum "$CURRENT_REPORT")
CURRENT_REPORT_SHA=${CURRENT_REPORT_SHA%% *}
STREAM_PLAN_SHA=$(sha256sum "$STREAM_PLAN")
STREAM_PLAN_SHA=${STREAM_PLAN_SHA%% *}
REPRESENTATION_SPLIT_SHA=$(sha256sum "$REPRESENTATION_SPLIT")
REPRESENTATION_SPLIT_SHA=${REPRESENTATION_SPLIT_SHA%% *}
EVALUATION_PLAN_SHA=$(sha256sum "$EVALUATION_PLAN")
EVALUATION_PLAN_SHA=${EVALUATION_PLAN_SHA%% *}

mkdir -p "$RUN_DIR"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO/src:$REPO"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  tools/run_lingbot_vla2_task_independent_full.py \
  --phase "$PHASE" \
  --source-checkout "$SOURCE" \
  --checkpoint-dir "$CHECKPOINT" \
  --processor-dir "$PROCESSOR" \
  --dataset-split "$DATASET" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --norm-stats "$NORM_STATS" \
  --stream-plan "$STREAM_PLAN" \
  --stream-plan-sha256 "$STREAM_PLAN_SHA" \
  --representation-split "$REPRESENTATION_SPLIT" \
  --representation-split-sha256 "$REPRESENTATION_SPLIT_SHA" \
  --evaluation-plan "$EVALUATION_PLAN" \
  --evaluation-plan-sha256 "$EVALUATION_PLAN_SHA" \
  --physical-sidecar-root "$SIDECAR" \
  --physical-sidecar-manifest "$SIDECAR/manifest.json" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA" \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 "$VISUAL_ACCEPTANCE_SHA" \
  --current-grid-cache-root "$CURRENT_CACHE" \
  --current-grid-cache-build-report "$CURRENT_REPORT" \
  --current-grid-cache-build-report-sha256 "$CURRENT_REPORT_SHA" \
  --run-dir "$RUN_DIR" \
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --stop-after-step "$STOP_AFTER_STEP" \
  --posterior-architecture two_pass_v3 \
  --maximum-control-tokens 64 \
  --causal-ablation-mode none \
  --entity-weight 0.08 \
  --predictive-weight 0.004 \
  --local-bptt-probability 0.0 \
  --overshoot-probability 0.0 \
  --source-mask-probability 0.10 \
  --source-mask-token-fraction 0.0625 \
  --source-prediction-mode omitted_static \
  --fsdp2-placement selective-embedding-offload \
  --cuda-allocator native
