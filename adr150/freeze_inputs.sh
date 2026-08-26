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
CHECKPOINT=${PICF_LINGBOT_CHECKPOINT:-/mnt/picf-next/models/lingbot-vla-v2-6b}
PROCESSOR=${PICF_QWEN_PROCESSOR:-/mnt/picf-next/models/qwen3-vl-4b-instruct}
DATASET=${PICF_CALVIN_TRAINING_SPLIT:-/mnt/calvin_data/task_ABC_D/training}
DATASET_MANIFEST=${PICF_CALVIN_DATASET_MANIFEST:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/calvin-training-files.json}
SOURCE_DATASET_MANIFEST=${PICF_CALVIN_SOURCE_DATASET_MANIFEST:-/mnt/picf-next/manifests/calvin-training-files.json}
SOURCE_RECEIPT=${PICF_CALVIN_SOURCE_RECEIPT:-/mnt/picf-next-provenance/calvin-task-ABC-D-content-a60b7934/receipt.json}
NORM_STATS=${PICF_CALVIN_NORM_STATS:-/mnt/picf-next-provenance/calvin-normalization-identity-a60b7934/calvin-lingbot-norm-stats.json}
SIDECAR_MANIFEST=${PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST:-/mnt/picf-next-provenance/calvin-sidecar-identity-a60b7934/physical-sidecar-manifest.json}
SIDECAR_ROOT=${PICF_CALVIN_PHYSICAL_SIDECAR:-/mnt/picf-next/targets/calvin-physical-all-source-v5-9fc4ca631026-20260724T184616Z}
VISUAL_ACCEPTANCE=${PICF_CALVIN_VISUAL_ACCEPTANCE:-/mnt/picf-next-provenance/calvin-physical-visual-review-identity-a60b7934/calvin-physical-visual-acceptance.json}
PROJECTION=${PICF_CALVIN_QWEN_PROJECTION:-/mnt/picf-next/manifests/lingbot-calvin-qwen-projection-a60b7934.json}
ASSET_MANIFEST=${PICF_FULL_MODAL_ASSET_MANIFEST:-/mnt/picf-next/manifests/full_modal_assets.json}
TACTILE_CALIBRATION=${PICF_TACTILE_CALIBRATION:-/mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.npz}
TACTILE_CALIBRATION_RECEIPT=${PICF_TACTILE_CALIBRATION_RECEIPT:-/mnt/picf-next-provenance/calvin-tactile-calibration-identity-a60b7934/tactile_backgrounds.receipt.json}
CAMERA_CALIBRATION=${PICF_CAMERA_CALIBRATION:-/mnt/calvin_data/task_ABC_D/calib/cameras.json}
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
MANIFEST=$HANDOFF/frozen_inputs.manifest.json
RECEIPT=$HANDOFF/frozen_inputs.sha256

[[ "$MATCHED_LBOT_REPORT" == /mnt/* ]] || {
  echo "matched LBOT report must be persistent under /mnt" >&2
  exit 1
}
for path in \
  "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" "$SIDECAR_ROOT" \
  "$CURRENT_CACHE" "$ANYTOUCH_CACHE" "$SONATA_CACHE" "$VJEPA_CACHE"
do
  [[ -d "$path" && ! -L "$path" ]] || {
    echo "ADR-150 required direct directory is absent: $path" >&2
    exit 1
  }
done
mkdir -p "$HANDOFF"
RECOVER_ORPHAN_MANIFEST=0
if [[ -f "$MANIFEST" && ! -L "$MANIFEST" && ! -e "$RECEIPT" && ! -L "$RECEIPT" ]]; then
  RECOVER_ORPHAN_MANIFEST=1
elif [[ ! -e "$MANIFEST" && ! -L "$MANIFEST" && ! -e "$RECEIPT" && ! -L "$RECEIPT" ]]; then
  :
else
  echo "ADR-150 frozen inputs already exist; never overwrite a scientific receipt" >&2
  exit 1
fi
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "ADR-150 input freezing requires a clean implementation checkout" >&2
  exit 1
}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" - \
  "$REPO" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" <<'PY'
from pathlib import Path
import sys

from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
from tools.bootstrap_lingbot_vla2_native import PATCH_RELATIVE_PATH, validate_prepared_native_source

repo = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
checkpoint = Path(sys.argv[3]).resolve()
processor = Path(sys.argv[4]).resolve()
validated = validate_prepared_native_source(checkout=source, patch_path=repo / PATCH_RELATIVE_PATH)
if validated.get("patch_state") != "applied":
    raise SystemExit("ADR-150 LingBot source is not in the exact approved patched state")
validate_checkpoint(checkpoint)
validate_processor(processor)
print("ADR-150 LingBot patched-source receipt=PASS")
print("ADR-150 checkpoint/processor asset contract=PASS")
PY

INPUTS=(
  "$PYTHON"
  "$RUNTIME_ARCHIVE"
  "$RUNTIME_ARCHIVE_RECEIPT"
  "$MATCHED_LBOT_REPORT"
  "$DATASET_MANIFEST"
  "$SOURCE_DATASET_MANIFEST"
  "$SOURCE_RECEIPT"
  "$NORM_STATS"
  "$SIDECAR_MANIFEST"
  "$VISUAL_ACCEPTANCE"
  "$PROJECTION"
  "$ASSET_MANIFEST"
  "$TACTILE_CALIBRATION"
  "$TACTILE_CALIBRATION_RECEIPT"
  "$CAMERA_CALIBRATION"
  "$STREAM_PLAN"
  "$REPRESENTATION_SPLIT"
  "$EVALUATION_PLAN"
  "$DENSE_COVERAGE"
  "$CURRENT_CACHE/manifest.json"
  "$CURRENT_REPORT"
  "$ANYTOUCH_CACHE/manifest.json"
  "$ANYTOUCH_CACHE.receipt.json"
  "$SONATA_CACHE/manifest.json"
  "$SONATA_CACHE.receipt.json"
  "$VJEPA_CACHE/manifest.json"
  "$VJEPA_CACHE.receipt.json"
  "$DENSE_SOURCE_AUDIT"
  "$DENSE_SEMANTIC_AUDIT"
)
for path in "${INPUTS[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "frozen ADR-150 input is absent or indirect: $path" >&2
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

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src:$REPO" "$PYTHON" - \
  "$MATCHED_LBOT_REPORT" "$DATASET_MANIFEST" "$SOURCE_DATASET_MANIFEST" \
  "$SOURCE_RECEIPT" "$NORM_STATS" "$SIDECAR_MANIFEST" \
  "$STREAM_PLAN" "$REPRESENTATION_SPLIT" "$EVALUATION_PLAN" \
  "$DENSE_COVERAGE" "$CURRENT_CACHE" "$CURRENT_REPORT" \
  "$ANYTOUCH_CACHE" "$SONATA_CACHE" "$VJEPA_CACHE" \
  "$REPO_COMMIT" "$SOURCE_COMMIT" "$MANIFEST_TMP" "${#INPUTS[@]}" \
  "${INPUTS[@]}" \
  "$REPO" "$PYTHON" "$SOURCE" "$CHECKPOINT" "$PROCESSOR" "$DATASET" \
  "$DATASET_MANIFEST" "$NORM_STATS" "$SIDECAR_ROOT" "$SIDECAR_MANIFEST" \
  "$VISUAL_ACCEPTANCE" "$PROJECTION" "$ASSET_MANIFEST" \
  "$TACTILE_CALIBRATION" "$TACTILE_CALIBRATION_RECEIPT" \
  "$CAMERA_CALIBRATION" "$CONTRACT_ROOT" "$CACHE_ROOT" "$CURRENT_CACHE" \
  "$CURRENT_REPORT" "$ANYTOUCH_CACHE" "$SONATA_CACHE" "$VJEPA_CACHE" \
  "$DENSE_SOURCE_AUDIT" "$DENSE_SEMANTIC_AUDIT" \
  "$RUNTIME_ARCHIVE" "$RUNTIME_ARCHIVE_RECEIPT" <<'PY'
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys

from picf_next.data.calvin_official_source import (
    validate_calvin_content_identity_migration,
    validate_calvin_official_source_receipt,
)
from picf_next.data.calvin_dense_evidence_source_audit import (
    validate_calvin_dense_evidence_source_audit,
)
from picf_next.data.calvin_dense_evidence_audit import (
    validate_calvin_dense_evidence_audit,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.data.dense_evidence_cache import FrozenDenseEvidenceCacheBank
from picf_next.data.dense_evidence_coverage import DenseEvidenceCoveragePlan
from picf_next.full_modal_assets import FullModalAssetManifest
from picf_next.lingbot_native.current_grid_cache import LingBotCurrentGridTargetCache
from picf_next.lingbot_native.adr150_lbot_validation import (
    validate_adr150_matched_lbot_report,
)
from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
from tools.bootstrap_lingbot_vla2_native import (
    LINGBOT_NATIVE_SOURCE_COMMIT,
    PATCH_RELATIVE_PATH,
    validate_prepared_native_source,
    verify_native_patch,
)
from tools.lingbot_vla2_runtime_helpers import (
    load_lingbot_training_config,
    resolve_lingbot_optimizer_contract,
)
from tools.run_lingbot_vla2_official_lbot import _implementation_provenance
from tools.run_lingbot_vla2_task_independent_full import _implementation_digest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    lbot_path,
    target_manifest_path,
    source_manifest_path,
    source_receipt_path,
    norm_path,
    sidecar_manifest_path,
    plan_path,
    split_path,
    evaluation_path,
    coverage_path,
    current_root,
    current_report_path,
    anytouch_root,
    sonata_root,
    vjepa_root,
) = map(Path, sys.argv[1:16])
repo_commit, source_commit = sys.argv[16:18]
output = Path(sys.argv[18])
input_count = int(sys.argv[19])
inputs = tuple(Path(value) for value in sys.argv[20 : 20 + input_count])
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
    "calvin_qwen_projection",
    "full_modal_asset_manifest",
    "tactile_calibration",
    "tactile_calibration_receipt",
    "camera_calibration",
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
path_values = sys.argv[20 + input_count :]
if len(path_values) != len(path_names):
    raise SystemExit("ADR-150 canonical path argument coverage differs")
canonical_paths = {
    name: str(Path(value).expanduser().resolve())
    for name, value in zip(path_names, path_values, strict=True)
}

target_manifest = load_dataset_file_manifest(target_manifest_path)
source_manifest = load_dataset_file_manifest(source_manifest_path)
validate_calvin_content_identity_migration(source_manifest, target_manifest)
validate_calvin_official_source_receipt(
    json.loads(source_receipt_path.read_text(encoding="utf-8")),
    source_manifest=source_manifest,
    source_manifest_sha256=digest(source_manifest_path),
    target_manifest=target_manifest,
    target_manifest_sha256=digest(target_manifest_path),
)

norm = json.loads(norm_path.read_text(encoding="utf-8"))
norm_source = norm.get("source", {})
if (
    norm_source.get("dataset_id") != target_manifest.dataset_id
    or norm_source.get("dataset_revision") != target_manifest.dataset_revision
    or norm_source.get("dataset_tree_sha256") != target_manifest.tree_sha256
):
    raise SystemExit("ADR-150 normalization and official dataset identity differ")

sidecar = json.loads(sidecar_manifest_path.read_text(encoding="utf-8"))
sidecar_sha256 = digest(sidecar_manifest_path)
plan = json.loads(plan_path.read_text(encoding="utf-8"))
split = json.loads(split_path.read_text(encoding="utf-8"))
evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
coverage = DenseEvidenceCoveragePlan.load(coverage_path)
if (
    coverage.dataset_id != target_manifest.dataset_id
    or coverage.dataset_revision != target_manifest.dataset_revision
    or coverage.dataset_tree_sha256 != target_manifest.tree_sha256
    or coverage.stream_plan_sha256 != plan.get("plan_sha256")
    or coverage.representation_split_sha256 != split.get("artifact_sha256")
    or coverage.evaluation_plan_sha256 != evaluation.get("artifact_sha256")
    or coverage.training_visit_count != 120_000
    or len(coverage.records) != 120_068
):
    raise SystemExit("ADR-150 full dense coverage differs from the official 30k contract")

lbot = json.loads(lbot_path.read_text(encoding="utf-8"))
repository = Path(canonical_paths["repository"])
lingbot_source = Path(canonical_paths["lingbot_source"])
checkpoint_validation = validate_checkpoint(Path(canonical_paths["lingbot_checkpoint"]))
processor_validation = validate_processor(Path(canonical_paths["qwen_processor"]))
patch_validation = verify_native_patch(
    root=repository,
    checkout=lingbot_source,
    check_apply=True,
)
prepared_source = validate_prepared_native_source(
    checkout=lingbot_source,
    patch_path=repository / PATCH_RELATIVE_PATH,
)
if source_commit != LINGBOT_NATIVE_SOURCE_COMMIT:
    raise SystemExit("ADR-150 LingBot checkout commit differs from the pinned source revision")
implementation_files, lbot_implementation_sha256 = _implementation_provenance(repository)
processor_config = json.loads(
    (Path(canonical_paths["qwen_processor"]) / "config.json").read_text(encoding="utf-8")
)
vision_config = processor_config.get("vision_config")
if not isinstance(vision_config, dict):
    raise SystemExit("ADR-150 Qwen processor has no vision geometry")
dataset_validation = validate_dataset_runtime_binding(
    target_manifest,
    Path(canonical_paths["dataset_split"]),
    dataset_id=target_manifest.dataset_id,
    dataset_revision=target_manifest.dataset_revision,
    split_name=target_manifest.split_name,
)
optimizer_contract = asdict(
    resolve_lingbot_optimizer_contract(
        load_lingbot_training_config(
            lingbot_source / "configs/vla/robotwin/robotwin.yaml"
        ),
        requested_learning_rate=1e-4,
    )
)
lbot_validation = validate_adr150_matched_lbot_report(
    lbot,
    expected_plan_sha256=plan["plan_sha256"],
    expected_representation_split_sha256=split["artifact_sha256"],
    expected_evaluation_plan_sha256=evaluation["artifact_sha256"],
    expected_seed=20260721,
    expected_implementation_identity={
        "implementation_files": implementation_files,
        "implementation_sha256": lbot_implementation_sha256,
    },
    expected_source_identity={
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "source_patch_sha256": patch_validation["patch_sha256"],
        "patched_source_sha256": prepared_source["patched_source_sha256"],
    },
    expected_model_identity={
        "model_family_sha256": lbot["model_family_sha256"],
        "checkpoint_revision": checkpoint_validation["checkpoint_revision"],
        "checkpoint_assets": checkpoint_validation["checkpoint_assets"],
        "qwen_vision_geometry": {
            "patch_size": vision_config["patch_size"],
            "spatial_merge_size": vision_config["spatial_merge_size"],
        },
        "parameter_storage": lbot["parameter_storage"],
        "parameter_manifest": lbot["parameter_manifest"],
        "alignment_teacher_prune": lbot["alignment_teacher_prune"],
    },
    expected_processor_identity={
        "processor_revision": processor_validation["processor_revision"],
        "processor_assets": processor_validation["processor_assets"],
    },
    expected_dataset_identity={
        "status": "PASS",
        "manifest_sha256": digest(target_manifest_path),
        "normalization_sha256": digest(norm_path),
        "validation": dataset_validation,
    },
    expected_optimizer_contract=optimizer_contract,
)

dense_roots = (anytouch_root, sonata_root, vjepa_root)
dense_manifest_sha256s = tuple(digest(root / "manifest.json") for root in dense_roots)
dense_bank = FrozenDenseEvidenceCacheBank.load(
    dense_roots,
    manifest_sha256s=dense_manifest_sha256s,
    dataset_tree_sha256=target_manifest.tree_sha256,
    memory_capacity=1,
)
if (
    dense_bank.modalities != ("anytouch", "sonata", "vjepa")
    or dense_bank.record_count != len(coverage.records)
    or dense_bank.coverage_plan_sha256 != coverage.artifact_sha256
):
    raise SystemExit("ADR-150 dense cache bank differs from full official coverage")

asset_manifest = FullModalAssetManifest.load(
    canonical_paths["full_modal_asset_manifest"], verify_files=True
)
asset_sha256s = {asset.modality: asset.sha256 for asset in asset_manifest.assets}
dense_cache_by_modality = {cache.contract.modality: cache for cache in dense_bank.caches}
if set(dense_cache_by_modality) != set(asset_sha256s):
    raise SystemExit("ADR-150 dense cache modalities differ from released assets")
for modality, cache in dense_cache_by_modality.items():
    if asset_sha256s[modality] not in cache.contract.encoder_contract:
        raise SystemExit(f"ADR-150 {modality} cache encoder differs from released asset")

dense_source_audit_path = Path(canonical_paths["dense_source_audit"])
dense_source_audit = validate_calvin_dense_evidence_source_audit(
    json.loads(dense_source_audit_path.read_text(encoding="ascii")),
    dataset_id=target_manifest.dataset_id,
    dataset_revision=target_manifest.dataset_revision,
    dataset_tree_sha256=target_manifest.tree_sha256,
    coverage_plan_sha256=coverage.artifact_sha256,
    cache_manifest_sha256_by_modality={
        modality: digest(cache.root / "manifest.json")
        for modality, cache in dense_cache_by_modality.items()
    },
    record_count=len(coverage.records),
)
dense_semantic_audit_path = Path(canonical_paths["dense_semantic_audit"])
dense_semantic_audit = validate_calvin_dense_evidence_audit(
    json.loads(dense_semantic_audit_path.read_text(encoding="ascii")),
    dense_bank,
)

dense_verified_shards = {}
for cache in dense_bank.caches:
    first_by_shard = {}
    for record in cache.records:
        first_by_shard.setdefault(record.shard_index, record)
    if set(first_by_shard) != set(range(len(cache.shards))):
        raise SystemExit("ADR-150 dense cache has an unaddressable shard")
    for record in first_by_shard.values():
        cache.evidence_for(
            source_global_index=record.source_global_index,
            sample_key=record.sample_key,
            source_input_sha256=record.source_input_sha256,
        )
    dense_verified_shards[cache.contract.modality] = len(cache.shards)

current_report = json.loads(current_report_path.read_text(encoding="utf-8"))
current_manifest_sha256 = digest(current_root / "manifest.json")
if (
    current_report.get("cache_manifest_sha256") != current_manifest_sha256
    or current_report.get("expected_record_count") != 120_004
    or Path(current_report.get("output_root", "")).resolve() != current_root.resolve()
    or current_report.get("physical_visual_acceptance_sha256")
    != "6443c34b6e8180a8ec090d50ee14dbb2e9d0ad6c4a5e2fc0d9f03a1dbd156552"
    or current_report.get("stream_plan_sha256") != plan.get("plan_sha256")
):
    raise SystemExit("ADR-150 current-grid build report differs from the official contract")
current_cache = LingBotCurrentGridTargetCache.load(
    current_root,
    manifest_sha256=current_manifest_sha256,
    dataset_tree_sha256=target_manifest.tree_sha256,
    physical_sidecar_manifest_sha256=sidecar_sha256,
    encoder_digest=current_report["teacher_encoder_digest"],
    coverage_sha256=current_report["coverage_sha256"],
    memory_capacity=1,
)
for shard in current_cache.shards:
    if current_cache.record_for(source_global_index=shard.first_source_global_index) is None:
        raise SystemExit("ADR-150 current-grid cache has an unreadable shard")

cache_verification = {
    "status": "PASS",
    "current_grid_manifest_sha256": current_manifest_sha256,
    "current_grid_record_count": len(current_cache.source_global_indices),
    "current_grid_verified_shard_count": len(current_cache.shards),
    "dense_cache_manifest_sha256": {
        modality: digest(cache.root / "manifest.json")
        for modality, cache in sorted(dense_cache_by_modality.items())
    },
    "dense_encoder_asset_sha256": dict(sorted(asset_sha256s.items())),
    "dense_record_count": dense_bank.record_count,
    "dense_verified_shard_count": dense_verified_shards,
    "dense_coverage_artifact_sha256": coverage.artifact_sha256,
    "dense_source_input_audit_artifact_sha256": dense_source_audit["artifact_sha256"],
    "dense_source_input_audit_file_sha256": digest(dense_source_audit_path),
    "dense_semantic_audit_artifact_sha256": dense_semantic_audit["artifact_sha256"],
    "dense_semantic_audit_file_sha256": digest(dense_semantic_audit_path),
}
manifest = {
    "schema": "picf-next.adr150-frozen-inputs/v2",
    "implementation_commit": repo_commit,
    "implementation_sha256": _implementation_digest(Path(canonical_paths["repository"])),
    "lingbot_source_commit": source_commit,
    "canonical_paths": canonical_paths,
    "lingbot_checkpoint_identity": checkpoint_validation,
    "qwen_processor_identity": processor_validation,
    "training_scope": {"first_step": 0, "gate_step": 2000, "declared_final_step": 30000},
    "matched_lbot_report": {
        "path": str(lbot_path),
        "sha256": digest(lbot_path),
        "canonical_sha256": lbot_validation.report_sha256,
    },
    "matched_lbot_validation": lbot_validation.validation_report,
    "physical_stream_plan_sha256": plan.get("plan_sha256"),
    "representation_split_sha256": split.get("artifact_sha256"),
    "evaluation_plan_sha256": evaluation.get("artifact_sha256"),
    "dataset_tree_sha256": target_manifest.tree_sha256,
    "physical_sidecar_manifest_sha256": sidecar_sha256,
    "full_modal_cache_verification": cache_verification,
    "inputs": [{"path": str(path), "sha256": digest(path)} for path in inputs],
}
encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
with output.open("wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
PY

chmod 0444 "$MANIFEST_TMP"
if [[ "$RECOVER_ORPHAN_MANIFEST" -eq 1 ]]; then
  cmp -s "$MANIFEST_TMP" "$MANIFEST" || {
    echo "ADR-150 orphan manifest differs from exact deterministic reconstruction" >&2
    exit 1
  }
  rm -f "$MANIFEST_TMP"
else
  mv -T "$MANIFEST_TMP" "$MANIFEST"
fi
sha256sum "${INPUTS[@]}" "$MANIFEST" >"$RECEIPT_TMP"
chmod 0444 "$RECEIPT_TMP"
mv -T "$RECEIPT_TMP" "$RECEIPT"
sha256sum --check --strict "$RECEIPT"
echo "ADR-150 frozen input receipt=PASS path=$RECEIPT"
