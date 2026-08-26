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
  echo "ADR-150 run directory must be persistent under /mnt" >&2
  exit 2
}

case "$PHASE" in
  fresh)
    [[ "$LOAD_GLOBAL_STEP" -eq 0 && "$STOP_AFTER_STEP" -le 2000 && ! -e "$RUN_DIR" ]] || {
      echo "fresh ADR-150 requires step zero, an absent run directory, and a gate at or before 2000" >&2
      exit 2
    }
    ;;
  resume)
    [[ "$LOAD_GLOBAL_STEP" -gt 0 && $((LOAD_GLOBAL_STEP % 2000)) -eq 0 ]] || {
      echo "ADR-150 resume requires a positive 2000-step boundary" >&2
      exit 2
    }
    [[ -d "$RUN_DIR/checkpoints/global_step_$LOAD_GLOBAL_STEP" ]] || {
      echo "ADR-150 resume checkpoint is absent" >&2
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
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
SIDECAR_ROOT=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
SIDECAR_MANIFEST=${PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST:-/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json}
SIDECAR_SHA=${PICF_CALVIN_PHYSICAL_SIDECAR_SHA256:-ee07c57829f895808b4a339ecf35a540c7d794ea0177d17d33ffc1e35ac34a1d}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next-provenance/calvin-physical-visual-review-identity-a60b7934/calvin-physical-visual-acceptance.json}
VISUAL_ACCEPTANCE_SHA=${PICF_CALVIN_VISUAL_ACCEPTANCE_SHA256:-6443c34b6e8180a8ec090d50ee14dbb2e9d0ad6c4a5e2fc0d9f03a1dbd156552}
CONTRACT_ROOT=${PICF_ADR150_CONTRACT_ROOT:-/mnt/picf-next/adr150/contracts/calvin-official-30k-v1}
STREAM_PLAN=$CONTRACT_ROOT/four-gpu-30k.physical.stream-plan.json
REPRESENTATION_SPLIT=$CONTRACT_ROOT/four-gpu-30k.physical.split.json
EVALUATION_PLAN=$CONTRACT_ROOT/four-gpu-30k.physical.evaluation.json
DENSE_COVERAGE=$CONTRACT_ROOT/four-gpu-30k.physical.dense-evidence-coverage.json
CACHE_ROOT=${PICF_ADR150_CACHE_ROOT:-/mnt/picf-next/adr150/caches/calvin-official-30k-v1}
CURRENT_CACHE=${PICF_CURRENT_CACHE_ROOT:-$CACHE_ROOT/current-filter-dino-physical-v1}
CURRENT_REPORT=${PICF_CURRENT_CACHE_BUILD_REPORT:-${CURRENT_CACHE}.build_report.json}
ANYTOUCH_CACHE=${PICF_ANYTOUCH_CACHE_ROOT:-$CACHE_ROOT/anytouch-observed-pose}
SONATA_CACHE=${PICF_SONATA_CACHE_ROOT:-$CACHE_ROOT/sonata-native}
VJEPA_CACHE=${PICF_VJEPA_CACHE_ROOT:-$CACHE_ROOT/vjepa-final}
DENSE_SOURCE_AUDIT=${PICF_DENSE_SOURCE_AUDIT:-$CACHE_ROOT/full-dense-source-input-audit.json}
DENSE_SEMANTIC_AUDIT=${PICF_DENSE_SEMANTIC_AUDIT:-$CACHE_ROOT/full-dense-semantic-audit.json}
RUNTIME_ARCHIVE=${PICF_RUNTIME_ARCHIVE:-/mnt/picf-next/runtime-archives/picf-runtime-restore-probe-94305690cafb-20260808.tar}
RUNTIME_ARCHIVE_RECEIPT=${PICF_RUNTIME_ARCHIVE_RECEIPT:-$RUNTIME_ARCHIVE.sha256}
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr150/handoff_20260810}
FROZEN_MANIFEST=${PICF_ADR150_FROZEN_MANIFEST:-$HANDOFF/frozen_inputs.manifest.json}
FROZEN_INPUTS=${PICF_ADR150_FROZEN_INPUTS:-$HANDOFF/frozen_inputs.sha256}

for path in \
  "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR_ROOT" \
  "$CURRENT_CACHE" "$ANYTOUCH_CACHE" "$SONATA_CACHE" "$VJEPA_CACHE"
do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "required direct directory is absent: $path" >&2
    exit 1
  }
done
for path in \
  "$PYTHON" "$DATASET_MANIFEST" "$NORM_STATS" "$SIDECAR_MANIFEST" \
  "$VISUAL_ACCEPTANCE" "$STREAM_PLAN" "$REPRESENTATION_SPLIT" \
  "$EVALUATION_PLAN" "$DENSE_COVERAGE" "$CURRENT_CACHE/manifest.json" \
  "$CURRENT_REPORT" "$ANYTOUCH_CACHE/manifest.json" \
  "$SONATA_CACHE/manifest.json" "$VJEPA_CACHE/manifest.json" \
  "$DENSE_SOURCE_AUDIT" "$DENSE_SEMANTIC_AUDIT" \
  "$RUNTIME_ARCHIVE" "$RUNTIME_ARCHIVE_RECEIPT" \
  "$FROZEN_MANIFEST" "$FROZEN_INPUTS"
do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "required file is absent or indirect: $path" >&2
    exit 1
  }
done

sha256sum --check --strict "$FROZEN_INPUTS"
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-150 execution requires an exact clean checkout" >&2
  exit 1
}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" - "$FROZEN_MANIFEST" \
  "$(git -C "$REPO" rev-parse HEAD)" \
  "$(git -C "$SOURCE" rev-parse HEAD)" \
  "$REPO" "$PYTHON" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" \
  "$DATASET_MANIFEST" "$NORM_STATS" "$SIDECAR_ROOT" "$SIDECAR_MANIFEST" \
  "$VISUAL_ACCEPTANCE" "$CONTRACT_ROOT" "$CACHE_ROOT" "$CURRENT_CACHE" \
  "$CURRENT_REPORT" "$ANYTOUCH_CACHE" "$SONATA_CACHE" "$VJEPA_CACHE" \
  "$DENSE_SOURCE_AUDIT" "$DENSE_SEMANTIC_AUDIT" \
  "$RUNTIME_ARCHIVE" "$RUNTIME_ARCHIVE_RECEIPT" <<'PY'
import json
from pathlib import Path
import sys

from tools.run_lingbot_vla2_task_independent_full import _implementation_digest

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema") != "picf-next.adr150-frozen-inputs/v2":
    raise SystemExit("ADR-150 frozen input manifest schema differs")
if manifest.get("implementation_commit") != sys.argv[2]:
    raise SystemExit("ADR-150 implementation commit changed after input freezing")
if manifest.get("lingbot_source_commit") != sys.argv[3]:
    raise SystemExit("ADR-150 LingBot source commit changed after input freezing")
if manifest.get("full_modal_cache_verification", {}).get("status") != "PASS":
    raise SystemExit("ADR-150 full-modal cache replay did not pass")
if manifest.get("matched_lbot_validation", {}).get("status") != "PASS":
    raise SystemExit("ADR-150 typed matched-LBOT validation did not pass")
path_names = (
    "repository",
    "python_executable",
    "lingbot_source",
    "lingbot_checkpoint",
    "qwen_processor",
    "dataset_split",
    "dataset_manifest",
    "normalization",
    "physical_sidecar_root",
    "physical_sidecar_manifest",
    "physical_visual_acceptance",
    "contract_root",
    "cache_root",
    "current_grid_cache",
    "current_grid_build_report",
    "anytouch_cache",
    "sonata_cache",
    "vjepa_cache",
    "dense_source_audit",
    "dense_semantic_audit",
    "runtime_archive",
    "runtime_archive_receipt",
)
provided = sys.argv[4:]
if len(provided) != len(path_names):
    raise SystemExit("ADR-150 effective path coverage differs")
expected_paths = manifest.get("canonical_paths")
if not isinstance(expected_paths, dict):
    raise SystemExit("ADR-150 frozen canonical paths are absent")
for name, value in zip(path_names, provided, strict=True):
    if expected_paths.get(name) != str(Path(value).expanduser().resolve()):
        raise SystemExit(f"ADR-150 effective path bypassed frozen input: {name}")
if manifest.get("implementation_sha256") != _implementation_digest(Path(provided[0])):
    raise SystemExit("ADR-150 transitive implementation closure changed after freezing")
print("ADR-150 frozen implementation/source/cache contract=PASS")
PY

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
[[ ${#GPU_ROWS[@]} -eq 4 ]] || {
  echo "ADR-150 requires exactly four visible GPUs" >&2
  exit 1
}
for row in "${GPU_ROWS[@]}"; do
  [[ "$row" == *"A100"* && "$row" == *"40960 MiB"* ]] || {
    echo "ADR-150 requires four A100 40GB GPUs; observed: $row" >&2
    exit 1
  }
done

file_sha256() {
  local value
  value=$(sha256sum "$1")
  printf '%s' "${value%% *}"
}

CURRENT_REPORT_SHA=$(file_sha256 "$CURRENT_REPORT")
STREAM_PLAN_SHA=$(file_sha256 "$STREAM_PLAN")
REPRESENTATION_SPLIT_SHA=$(file_sha256 "$REPRESENTATION_SPLIT")
EVALUATION_PLAN_SHA=$(file_sha256 "$EVALUATION_PLAN")
DENSE_COVERAGE_SHA=$(file_sha256 "$DENSE_COVERAGE")
ANYTOUCH_MANIFEST_SHA=$(file_sha256 "$ANYTOUCH_CACHE/manifest.json")
SONATA_MANIFEST_SHA=$(file_sha256 "$SONATA_CACHE/manifest.json")
VJEPA_MANIFEST_SHA=$(file_sha256 "$VJEPA_CACHE/manifest.json")

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
  --physical-sidecar-root "$SIDECAR_ROOT" \
  --physical-sidecar-manifest "$SIDECAR_MANIFEST" \
  --physical-sidecar-manifest-sha256 "$SIDECAR_SHA" \
  --physical-visual-acceptance "$VISUAL_ACCEPTANCE" \
  --physical-visual-acceptance-sha256 "$VISUAL_ACCEPTANCE_SHA" \
  --current-grid-cache-root "$CURRENT_CACHE" \
  --current-grid-cache-build-report "$CURRENT_REPORT" \
  --current-grid-cache-build-report-sha256 "$CURRENT_REPORT_SHA" \
  --dense-evidence-mode calvin_full_v1 \
  --dense-token-bridge lingbot_task_token_resampler_v1 \
  --dense-evidence-cache-root "$ANYTOUCH_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$ANYTOUCH_MANIFEST_SHA" \
  --dense-evidence-cache-root "$SONATA_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$SONATA_MANIFEST_SHA" \
  --dense-evidence-cache-root "$VJEPA_CACHE" \
  --dense-evidence-cache-manifest-sha256 "$VJEPA_MANIFEST_SHA" \
  --dense-evidence-coverage-plan "$DENSE_COVERAGE" \
  --dense-evidence-coverage-plan-sha256 "$DENSE_COVERAGE_SHA" \
  --run-dir "$RUN_DIR" \
  --load-global-step "$LOAD_GLOBAL_STEP" \
  --stop-after-step "$STOP_AFTER_STEP" \
  --seed 20260721 \
  --capacity 16 \
  --maximum-control-tokens 64 \
  --posterior-architecture two_pass_v3 \
  --learning-rate 1e-4 \
  --max-grad-norm 1.0 \
  --maximum-peak-reserved-gib 39.0 \
  --maximum-optimizer-lag 8 \
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
