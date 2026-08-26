#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MATCHED_LBOT_REPORT" >&2
  exit 2
fi

MATCHED_LBOT_REPORT=$1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${PICF_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${PICF_PYTHON:-/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12}
SOURCE=${PICF_LINGBOT_NATIVE_SOURCE:-/mnt/picf-next/source-checkouts/lingbot-vla-v2-adr147}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next/manifests/calvin-training-files.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next/manifests/calvin-lingbot-norm-stats.json}
SIDECAR=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next/runs/calvin-v5-full-tail-audit-43f5c5a-20260725T074858Z/visual_acceptance.json}
HANDOFF=${PICF_HANDOFF_ROOT:-/mnt/picf-next/adr149/handoff_20260809}
STREAM_PLAN=$HANDOFF/four-gpu-30k.physical.stream-plan.json
REPRESENTATION_SPLIT=$HANDOFF/four-gpu-30k.physical.split.json
EVALUATION_PLAN=$HANDOFF/four-gpu-30k.physical.evaluation-plan.json
CURRENT_CACHE=${PICF_CURRENT_CACHE_ROOT:-/mnt/picf-next/adr149/full-picf-30k/cache/current-filter-dino-physical-v1}
CURRENT_REPORT=${PICF_CURRENT_CACHE_BUILD_REPORT:-${CURRENT_CACHE}.build_report.json}
CURRENT_REPLAY_RECEIPT=${PICF_CURRENT_CACHE_FULL_REPLAY_RECEIPT:-${CURRENT_CACHE}.full_replay_verification.json}
MANIFEST=$HANDOFF/frozen_inputs.manifest.json
RECEIPT=$HANDOFF/frozen_inputs.sha256

[[ "$MATCHED_LBOT_REPORT" == /mnt/* ]] || {
  echo "matched LBOT report must be persistent under /mnt" >&2
  exit 1
}
[[ -d "$REPO" && -d "$SOURCE" && -d "$CURRENT_CACHE" ]] || {
  echo "ADR-149 repository, LingBot source, or exact cache is absent" >&2
  exit 1
}
[[ ! -e "$MANIFEST" && ! -L "$MANIFEST" && ! -e "$RECEIPT" && ! -L "$RECEIPT" ]] || {
  echo "ADR-149 frozen inputs already exist; never overwrite a scientific receipt" >&2
  exit 1
}
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-149 input freezing requires a clean implementation checkout" >&2
  exit 1
}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" - \
  "$REPO" "$SOURCE" <<'PY'
from pathlib import Path
import sys

from tools.bootstrap_lingbot_vla2_native import (
    PATCH_RELATIVE_PATH,
    validate_prepared_native_source,
)

repo = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
validated = validate_prepared_native_source(
    checkout=source,
    patch_path=repo / PATCH_RELATIVE_PATH,
)
if validated.get("patch_state") != "applied":
    raise SystemExit("ADR-149 LingBot source is not in the exact approved patched state")
print("ADR-149 LingBot patched-source receipt=PASS")
PY

INPUTS=(
  "$MATCHED_LBOT_REPORT"
  "$DATASET_MANIFEST"
  "$NORM_STATS"
  "$SIDECAR/manifest.json"
  "$VISUAL_ACCEPTANCE"
  "$STREAM_PLAN"
  "$REPRESENTATION_SPLIT"
  "$EVALUATION_PLAN"
  "$CURRENT_CACHE/manifest.json"
  "$CURRENT_REPORT"
  "$CURRENT_REPLAY_RECEIPT"
)
for path in "$PYTHON" "${INPUTS[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "frozen ADR-149 input is absent or indirect: $path" >&2
    exit 1
  }
done

REPO_COMMIT=$(git -C "$REPO" rev-parse HEAD)
SOURCE_COMMIT=$(git -C "$SOURCE" rev-parse HEAD)
MANIFEST_TMP=$(mktemp "$HANDOFF/.frozen_inputs.manifest.XXXXXX")
RECEIPT_TMP=$(mktemp "$HANDOFF/.frozen_inputs.sha256.XXXXXX")
cleanup() {
  rm -f "$MANIFEST_TMP" "$RECEIPT_TMP"
}
trap cleanup EXIT

"$PYTHON" - \
  "$MATCHED_LBOT_REPORT" \
  "$STREAM_PLAN" \
  "$REPRESENTATION_SPLIT" \
  "$EVALUATION_PLAN" \
  "$CURRENT_CACHE/manifest.json" \
  "$CURRENT_REPORT" \
  "$CURRENT_REPLAY_RECEIPT" \
  "$REPO_COMMIT" \
  "$SOURCE_COMMIT" \
  "$MANIFEST_TMP" \
  "${INPUTS[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


lbot_path, plan_path, split_path, evaluation_path, cache_manifest_path, cache_report_path, cache_replay_path = (
    map(Path, sys.argv[1:8])
)
repo_commit, source_commit = sys.argv[8:10]
output = Path(sys.argv[10])
inputs = tuple(Path(value) for value in sys.argv[11:])

lbot = json.loads(lbot_path.read_text(encoding="utf-8"))
plan = json.loads(plan_path.read_text(encoding="utf-8"))
split = json.loads(split_path.read_text(encoding="utf-8"))
evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
cache_replay = json.loads(cache_replay_path.read_text(encoding="utf-8"))

if lbot.get("schema") not in {
    "picf-next.lingbot-vla2-official-calvin-lbot.v1",
    "picf-next.lingbot-vla2-official-calvin-p0.v1",
}:
    raise SystemExit("matched LBOT schema differs")
if lbot.get("status") != "PASS" or lbot.get("picf_graph_installed") is not False:
    raise SystemExit("matched LBOT is not a passing graph-free control")
if lbot.get("world_size") != 4 or lbot.get("steps") != 200:
    raise SystemExit("matched LBOT must be the four-rank 200-step physical control")
if lbot.get("physical_event_stream") is not True or lbot.get("maximum_control_tokens") != 64:
    raise SystemExit("matched LBOT physical/control contract differs")
if lbot.get("registered_evaluation_steps") != [0, 20, 100, 200]:
    raise SystemExit("matched LBOT registered curve differs")
if lbot.get("checkpoint_published") is not False:
    raise SystemExit("matched LBOT unexpectedly published a checkpoint")
if lbot.get("plan_sha256") != plan.get("plan_sha256"):
    raise SystemExit("matched LBOT and physical stream differ")
if lbot.get("representation_split_sha256") != split.get("artifact_sha256"):
    raise SystemExit("matched LBOT and representation split differ")
if lbot.get("evaluation_plan_sha256") != evaluation.get("artifact_sha256"):
    raise SystemExit("matched LBOT and evaluation plan differ")

cache_manifest_sha256 = digest(cache_manifest_path)
if cache_report.get("cache_manifest_sha256") != cache_manifest_sha256:
    raise SystemExit("current-filter cache manifest and build report differ")
if cache_report.get("expected_record_count") != 120004:
    raise SystemExit("current-filter cache does not cover all 120,004 frozen sources")
if Path(cache_report.get("output_root", "")).resolve() != cache_manifest_path.parent.resolve():
    raise SystemExit("current-filter cache report points at another cache")
if cache_report.get("stream_plan_sha256") != plan.get("plan_sha256"):
    raise SystemExit("current-filter cache belongs to another stream")
if cache_report.get("physical_visual_acceptance_sha256") != (
    "4000dc3394b3027e7cf2a75d54a88b1025314ca503dc6ec2b77f4a63b2163c86"
):
    raise SystemExit("current-filter cache predates the accepted visual audit")
if cache_replay.get("schema") != (
    "picf-next.current-grid-cache-full-replay-verification/v1"
):
    raise SystemExit("current-filter cache full-replay schema differs")
if cache_replay.get("status") != "PASS":
    raise SystemExit("current-filter cache full replay did not pass")
if Path(cache_replay.get("cache_root", "")).resolve() != cache_manifest_path.parent.resolve():
    raise SystemExit("current-filter cache full replay targets another cache")
if cache_replay.get("cache_manifest_sha256") != cache_manifest_sha256:
    raise SystemExit("current-filter cache changed after its full replay")
if cache_replay.get("record_count") != cache_report.get("expected_record_count"):
    raise SystemExit("current-filter cache full replay did not cover every record")
content_stream_sha256 = cache_replay.get("content_stream_sha256")
if (
    not isinstance(content_stream_sha256, str)
    or len(content_stream_sha256) != 64
    or any(character not in "0123456789abcdef" for character in content_stream_sha256)
):
    raise SystemExit("current-filter cache full replay content digest is invalid")

manifest = {
    "schema": "picf-next.adr149-frozen-inputs/v1",
    "implementation_commit": repo_commit,
    "lingbot_source_commit": source_commit,
    "matched_lbot_report": {"path": str(lbot_path), "sha256": digest(lbot_path)},
    "physical_stream_plan_sha256": plan.get("plan_sha256"),
    "representation_split_sha256": split.get("artifact_sha256"),
    "evaluation_plan_sha256": evaluation.get("artifact_sha256"),
    "current_filter_cache_manifest_sha256": cache_manifest_sha256,
    "current_filter_cache_full_replay_sha256": digest(cache_replay_path),
    "current_filter_cache_content_stream_sha256": content_stream_sha256,
    "inputs": [{"path": str(path), "sha256": digest(path)} for path in inputs],
}
encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
with output.open("wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
PY

chmod 0444 "$MANIFEST_TMP"
mv -T "$MANIFEST_TMP" "$MANIFEST"
sha256sum "${INPUTS[@]}" "$MANIFEST" >"$RECEIPT_TMP"
chmod 0444 "$RECEIPT_TMP"
mv -T "$RECEIPT_TMP" "$RECEIPT"
sha256sum --check --strict "$RECEIPT"
echo "ADR-149 frozen input receipt=PASS path=$RECEIPT"
