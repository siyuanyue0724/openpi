#!/usr/bin/env bash

adr170_resolve_source_aligned_trial_checkpoint() {
  if [[ $# -ne 2 ]]; then
    echo "usage: adr170_resolve_source_aligned_trial_checkpoint PYTHON_BIN TRAINING_REPORT" >&2
    return 2
  fi

  local python_bin=$1
  local training_report=$2
  "$python_bin" - "$training_report" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
CHECKPOINT_MANIFEST_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v5"
MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
ACTION_SUPERVISION_SCHEMA = "picf-next.task-action-supervision.v1"
TASK_ADDRESS_DEPTH_SCHEMA = "picf-next.action-consumable-task-address-depth.v1"
PICF_SOURCE_CONTRACT_SCHEMA = "picf-next.g3-picf-source-contract.v1"
PICF_CRITICAL_SOURCE_FILES = {
    "tools/run_lingbot_vla2_ltop_g3_action_mediation.py",
    "src/picf_next/lingbot_native/task_address_learning.py",
    "src/picf_next/lingbot_native/task_action_supervision.py",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")


def fail(message):
    raise SystemExit(message)


def require_sha256(value, name):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        fail(f"{name} must be one lowercase SHA-256")
    return value


def require_action_consumable_depth(value, name):
    if not isinstance(value, dict):
        fail(f"{name} must be one JSON object")
    layer_count = value.get("layer_count")
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 2:
        fail(f"{name} layer_count is invalid")
    expected = {
        "schema": TASK_ADDRESS_DEPTH_SCHEMA,
        "producer_layer_index": layer_count - 2,
        "consumer_layer_index": layer_count - 1,
        "layer_count": layer_count,
        "final_layer_excluded": True,
        "reason": "address-output-must-precede-a-later-action-attention-layer",
    }
    if value != expected:
        fail(f"{name} is not action-consumable")
    if layer_count != 36:
        fail(f"{name} differs from the 36-layer LingBot host graph")
    return value


def require_picf_source_contract(value, name):
    expected_fields = {
        "schema", "repository_commit", "repository_tree", "worktree_clean",
        "critical_file_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        fail(f"{name} fields differ")
    if value.get("schema") != PICF_SOURCE_CONTRACT_SCHEMA:
        fail(f"{name} schema differs")
    for field in ("repository_commit", "repository_tree"):
        item = value.get(field)
        if not isinstance(item, str) or GIT_OBJECT.fullmatch(item) is None:
            fail(f"{name}.{field} is malformed")
    if value.get("worktree_clean") is not True:
        fail(f"{name} is not clean")
    files = value.get("critical_file_sha256")
    if not isinstance(files, dict) or set(files) != PICF_CRITICAL_SOURCE_FILES:
        fail(f"{name} critical source set differs")
    for path, digest in files.items():
        require_sha256(digest, f"{name} critical source {path}")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(root, schema):
    if root.is_symlink():
        fail("mediator-trial checkpoint model must be one direct directory")
    directory = root.resolve()
    if not directory.is_dir():
        fail("mediator-trial checkpoint model directory is absent")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            fail("mediator-trial checkpoint model tree contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            fail("mediator-trial checkpoint model tree contains a non-regular entry")
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        fail("mediator-trial checkpoint model tree is empty")
    payload = json.dumps(
        {"schema": schema, "files": files},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def rank_map(payload, name):
    reports = payload.get("rank_reports")
    if not isinstance(reports, list) or len(reports) != 2:
        fail(f"{name} must contain exactly two rank reports")
    result = {}
    for report in reports:
        if not isinstance(report, dict):
            fail(f"{name} contains a non-object rank report")
        rank = report.get("rank")
        if rank not in (0, 1) or rank in result:
            fail(f"{name} rank set is incomplete or duplicated")
        result[rank] = report
    if set(result) != {0, 1}:
        fail(f"{name} rank set is incomplete")
    return result


report_path = Path(sys.argv[1])
if not report_path.is_file() or report_path.is_symlink():
    fail(f"mediator-trial training report is absent: {report_path}")
report = json.loads(report_path.read_text(encoding="ascii"))
if not isinstance(report, dict):
    fail("mediator-trial training report must be a JSON object")
expected = {
    "schema": "picf-next.ltop-g3-training-phase.v1",
    "status": "PASS",
    "failures": [],
    "phase": "training",
    "mode": "mediator-trial",
    "steps": 256,
    "eval_every": 32,
    "world_size": 2,
    "seed": 20260813,
}
for field, value in expected.items():
    if report.get(field) != value:
        fail(f"mediator-trial training report violates {field}: {report.get(field)!r}")
depth_contract = require_action_consumable_depth(
    report.get("training_contract", {}).get("task_address_supervision_depth"),
    "mediator-trial task-address supervision depth",
)
source_contract = require_picf_source_contract(
    report.get("picf_source_contract"),
    "mediator-trial PICF source contract",
)

checkpoint = report.get("checkpoint")
if not isinstance(checkpoint, dict):
    fail("mediator-trial report omits its checkpoint receipt")
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    fail("mediator-trial checkpoint is not the registered model-only format")
if checkpoint.get("optimizer_saved") is not False:
    fail("mediator-trial checkpoint unexpectedly contains optimizer state")
manifest_sha256 = require_sha256(
    checkpoint.get("manifest_sha256"),
    "mediator-trial checkpoint receipt manifest_sha256",
)
if checkpoint.get("model_tree_schema") != MODEL_TREE_SCHEMA:
    fail("mediator-trial checkpoint receipt model_tree_schema differs")
if checkpoint.get("action_supervision_schema") != ACTION_SUPERVISION_SCHEMA:
    fail("source-aligned checkpoint receipt action supervision schema differs")
if checkpoint.get("task_address_supervision_depth") != depth_contract:
    fail("source-aligned checkpoint receipt task-address supervision depth differs")
if checkpoint.get("picf_source_contract") != source_contract:
    fail("source-aligned checkpoint receipt PICF source identity differs")
receipt_model_tree_sha256 = require_sha256(
    checkpoint.get("model_tree_sha256"),
    "mediator-trial checkpoint receipt model_tree_sha256",
)
receipt_training_digests = checkpoint.get("training_final_model_local_state_sha256_by_rank")
if not isinstance(receipt_training_digests, list) or len(receipt_training_digests) != 2:
    fail("mediator-trial checkpoint receipt must contain two training rank digests")
receipt_training_digests = [
    require_sha256(value, f"mediator-trial checkpoint receipt training rank {rank} digest")
    for rank, value in enumerate(receipt_training_digests)
]

checkpoint_value = checkpoint.get("path")
if not isinstance(checkpoint_value, str) or not checkpoint_value:
    fail("mediator-trial checkpoint path is invalid")
checkpoint_path = Path(checkpoint_value)
if not checkpoint_path.is_absolute():
    fail("mediator-trial checkpoint path must be absolute")
if not checkpoint_path.is_dir() or checkpoint_path.is_symlink():
    fail(f"mediator-trial checkpoint is absent: {checkpoint_path}")
checkpoint_path = checkpoint_path.resolve()
if not checkpoint_path.is_relative_to(Path("/mnt")):
    fail("mediator-trial checkpoint must live under /mnt")

root_entries = list(checkpoint_path.iterdir())
if any(path.is_symlink() for path in root_entries):
    fail("mediator-trial checkpoint root contains a symbolic link")
if {path.name for path in root_entries} != {"model", "ltop_g3_training_checkpoint.json"}:
    fail("mediator-trial checkpoint root differs from the model-only ABI")

model_path = checkpoint_path / "model"
if not model_path.is_dir() or model_path.is_symlink():
    fail("mediator-trial checkpoint model directory is absent")
metadata_path = model_path / ".metadata"
if not metadata_path.is_file() or metadata_path.is_symlink():
    fail("mediator-trial checkpoint omits direct DCP model/.metadata")
distcp_paths = list(model_path.glob("*.distcp"))
if not distcp_paths:
    fail("mediator-trial checkpoint omits a direct DCP *.distcp payload")
if any(not path.is_file() or path.is_symlink() for path in distcp_paths):
    fail("mediator-trial checkpoint DCP payload is not one direct regular file")
actual_model_tree_sha256 = directory_tree_sha256(model_path, MODEL_TREE_SCHEMA)
if receipt_model_tree_sha256 != actual_model_tree_sha256:
    fail("mediator-trial checkpoint receipt model-tree SHA-256 differs from disk")

manifest_path = checkpoint_path / "ltop_g3_training_checkpoint.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    fail("mediator-trial checkpoint manifest is absent")
if file_sha256(manifest_path) != manifest_sha256:
    fail("mediator-trial checkpoint manifest SHA-256 differs from its receipt")
manifest = json.loads(manifest_path.read_text(encoding="ascii"))
if not isinstance(manifest, dict):
    fail("mediator-trial checkpoint manifest must be a JSON object")

schedule = (
    report.get("training_contract", {})
    .get("action_information_set_trial", {})
    .get("schedule")
)
schedule_sha256 = schedule.get("sha256") if isinstance(schedule, dict) else None
schedule_sha256 = require_sha256(
    schedule_sha256,
    "mediator-trial report counterbalanced schedule digest",
)
supervision = report.get("training_contract", {}).get("action_supervision")
expected_supervision = {
    "schema": ACTION_SUPERVISION_SCHEMA,
    "official_action_loss": "immutable-source-task-action-pairs-only",
    "crossed_prompt_action_loss": False,
    "crossed_prompts": "representation-and-causal-evaluation-only",
    "ambiguous_source_task_address_loss": False,
    "unobservable_source_target_address_loss": False,
    "unobservable_source_target_policy": (
        "disable-address-only-with-explicit-loss-side-receipt"
    ),
}
if supervision != expected_supervision:
    fail("source-aligned trial report action supervision contract differs")
manifest_expected = {
    "schema": CHECKPOINT_MANIFEST_SCHEMA,
    "status": "PASS",
    "global_step": 256,
    "optimizer_saved": False,
    "format": CHECKPOINT_FORMAT,
    "world_size": 2,
    "model_tree_schema": MODEL_TREE_SCHEMA,
    "model_tree_sha256": actual_model_tree_sha256,
    "action_supervision_schema": ACTION_SUPERVISION_SCHEMA,
    "picf_source_contract": source_contract,
    "task_address_supervision_depth": depth_contract,
    "training_final_model_local_state_sha256_by_rank": receipt_training_digests,
    "action_information_set_schedule_sha256": schedule_sha256,
    "source_stage_checkpoint": report.get("stage_checkpoint"),
    "g2_report_sha256": report.get("g2_report_sha256"),
    "runtime_source_contract": report.get("runtime_source_contract"),
}
for field, value in manifest_expected.items():
    if manifest.get(field) != value:
        fail(f"mediator-trial checkpoint manifest violates {field}")
for field in (
    "format",
    "optimizer_saved",
    "model_tree_schema",
    "model_tree_sha256",
    "action_supervision_schema",
    "picf_source_contract",
    "task_address_supervision_depth",
    "training_final_model_local_state_sha256_by_rank",
):
    if manifest.get(field) != checkpoint.get(field):
        fail(f"mediator-trial checkpoint receipt and manifest differ on {field}")

training_ranks = rank_map(report, "mediator-trial training report")
training_report_digests = [
    require_sha256(
        training_ranks[rank].get("training_final_model_local_state_sha256"),
        f"mediator-trial training rank {rank} terminal model digest",
    )
    for rank in (0, 1)
]
if training_report_digests != receipt_training_digests:
    fail("mediator-trial checkpoint receipt differs from training terminal rank states")
runtime_schedules = [
    require_sha256(
        training_ranks[rank].get("runtime_schedule_sha256"),
        f"mediator-trial training rank {rank} runtime schedule digest",
    )
    for rank in (0, 1)
]
if runtime_schedules[0] != runtime_schedules[1]:
    fail("mediator-trial training ranks used different runtime schedules")
for rank in (0, 1):
    if training_ranks[rank].get("action_information_set_schedule_sha256") != schedule_sha256:
        fail(f"mediator-trial training rank {rank} counterbalanced schedule differs")

print(checkpoint_path)
PY
}

adr170_validate_cold_report() {
  if [[ $# -ne 6 ]]; then
    echo "usage: adr170_validate_cold_report PYTHON_BIN REPORT CHECKPOINT PHASE SCENES ACTION_INFORMATION_SET" >&2
    return 2
  fi

  local python_bin=$1
  local report=$2
  local checkpoint=$3
  local phase=$4
  local scenes=$5
  local action_information_set=$6
  "$python_bin" - "$report" "$checkpoint" "$phase" "$scenes" "$action_information_set" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
CHECKPOINT_MANIFEST_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v5"
MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
ACTION_SUPERVISION_SCHEMA = "picf-next.task-action-supervision.v1"
TASK_ADDRESS_DEPTH_SCHEMA = "picf-next.action-consumable-task-address-depth.v1"
PICF_SOURCE_CONTRACT_SCHEMA = "picf-next.g3-picf-source-contract.v1"
PICF_CRITICAL_SOURCE_FILES = {
    "tools/run_lingbot_vla2_ltop_g3_action_mediation.py",
    "src/picf_next/lingbot_native/task_address_learning.py",
    "src/picf_next/lingbot_native/task_action_supervision.py",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")


def fail(message):
    raise SystemExit(message)


def require_sha256(value, name):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        fail(f"{name} must be one lowercase SHA-256")
    return value


def require_action_consumable_depth(value, name):
    if not isinstance(value, dict):
        fail(f"{name} must be one JSON object")
    layer_count = value.get("layer_count")
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count < 2:
        fail(f"{name} layer_count is invalid")
    expected = {
        "schema": TASK_ADDRESS_DEPTH_SCHEMA,
        "producer_layer_index": layer_count - 2,
        "consumer_layer_index": layer_count - 1,
        "layer_count": layer_count,
        "final_layer_excluded": True,
        "reason": "address-output-must-precede-a-later-action-attention-layer",
    }
    if value != expected:
        fail(f"{name} is not action-consumable")
    if layer_count != 36:
        fail(f"{name} differs from the 36-layer LingBot host graph")
    return value


def require_picf_source_contract(value, name):
    expected_fields = {
        "schema", "repository_commit", "repository_tree", "worktree_clean",
        "critical_file_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        fail(f"{name} fields differ")
    if value.get("schema") != PICF_SOURCE_CONTRACT_SCHEMA:
        fail(f"{name} schema differs")
    for field in ("repository_commit", "repository_tree"):
        item = value.get(field)
        if not isinstance(item, str) or GIT_OBJECT.fullmatch(item) is None:
            fail(f"{name}.{field} is malformed")
    if value.get("worktree_clean") is not True:
        fail(f"{name} is not clean")
    files = value.get("critical_file_sha256")
    if not isinstance(files, dict) or set(files) != PICF_CRITICAL_SOURCE_FILES:
        fail(f"{name} critical source set differs")
    for path, digest in files.items():
        require_sha256(digest, f"{name} critical source {path}")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(root, schema):
    if root.is_symlink():
        fail("ADR170 cold checkpoint model must be one direct directory")
    directory = root.resolve()
    if not directory.is_dir():
        fail("ADR170 cold checkpoint model directory is absent")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            fail("ADR170 cold checkpoint model tree contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            fail("ADR170 cold checkpoint model tree contains a non-regular entry")
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        fail("ADR170 cold checkpoint model tree is empty")
    payload = json.dumps(
        {"schema": schema, "files": files},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def rank_map(payload, name):
    reports = payload.get("rank_reports")
    if not isinstance(reports, list) or len(reports) != 2:
        fail(f"{name} must contain exactly two rank reports")
    result = {}
    for report in reports:
        if not isinstance(report, dict):
            fail(f"{name} contains a non-object rank report")
        rank = report.get("rank")
        if rank not in (0, 1) or rank in result:
            fail(f"{name} rank set is incomplete or duplicated")
        result[rank] = report
    if set(result) != {0, 1}:
        fail(f"{name} rank set is incomplete")
    return result


report_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
phase = sys.argv[3]
scenes = int(sys.argv[4])
expected_action_information_set = sys.argv[5]
schemas = {
    "evaluation": "picf-next.ltop-g3-evaluation-phase.v1",
    "retention": "picf-next.ltop-g3-representation-retention.v1",
}
if phase not in schemas:
    fail(f"unsupported ADR170 cold phase: {phase}")
if phase == "evaluation" and expected_action_information_set not in {
    "factual",
    "mediator-required",
}:
    fail("ADR170 cold action information set is unsupported")
if phase == "retention" and expected_action_information_set != "none":
    fail("ADR170 cold retention must use the none information-set marker")
if not checkpoint_path.is_absolute():
    fail("ADR170 cold checkpoint path must be absolute")
if not checkpoint_path.is_dir() or checkpoint_path.is_symlink():
    fail(f"ADR170 cold checkpoint is absent: {checkpoint_path}")
checkpoint_path = checkpoint_path.resolve()
if not checkpoint_path.is_relative_to(Path("/mnt")):
    fail("ADR170 cold checkpoint must live under /mnt")

root_entries = list(checkpoint_path.iterdir())
if any(path.is_symlink() for path in root_entries):
    fail("ADR170 cold checkpoint root contains a symbolic link")
if {path.name for path in root_entries} != {"model", "ltop_g3_training_checkpoint.json"}:
    fail("ADR170 cold checkpoint root differs from the model-only ABI")
model_path = checkpoint_path / "model"
if not model_path.is_dir() or model_path.is_symlink():
    fail("ADR170 cold checkpoint model directory is absent")
metadata_path = model_path / ".metadata"
if not metadata_path.is_file() or metadata_path.is_symlink():
    fail("ADR170 cold checkpoint omits direct DCP model/.metadata")
distcp_paths = list(model_path.glob("*.distcp"))
if not distcp_paths:
    fail("ADR170 cold checkpoint omits a direct DCP *.distcp payload")
if any(not path.is_file() or path.is_symlink() for path in distcp_paths):
    fail("ADR170 cold checkpoint DCP payload is not one direct regular file")
model_tree_sha256 = directory_tree_sha256(model_path, MODEL_TREE_SCHEMA)

manifest_path = checkpoint_path / "ltop_g3_training_checkpoint.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    fail("ADR170 cold checkpoint manifest is absent")
manifest = json.loads(manifest_path.read_text(encoding="ascii"))
if not isinstance(manifest, dict):
    fail("ADR170 cold checkpoint manifest must be a JSON object")
manifest_expected = {
    "schema": CHECKPOINT_MANIFEST_SCHEMA,
    "status": "PASS",
    "global_step": 256,
    "optimizer_saved": False,
    "format": CHECKPOINT_FORMAT,
    "world_size": 2,
    "model_tree_schema": MODEL_TREE_SCHEMA,
    "model_tree_sha256": model_tree_sha256,
    "action_supervision_schema": ACTION_SUPERVISION_SCHEMA,
}
for field, value in manifest_expected.items():
    if manifest.get(field) != value:
        fail(f"ADR170 cold checkpoint manifest violates {field}")

training_report_path = checkpoint_path.parent / "ltop_g3_source_aligned_trial_training_report.json"
if not training_report_path.is_file() or training_report_path.is_symlink():
    fail("ADR170 cold checkpoint is detached from its mediator-trial training report")
training_report = json.loads(training_report_path.read_text(encoding="ascii"))
if not isinstance(training_report, dict):
    fail("ADR170 mediator-trial training report must be a JSON object")
training_expected = {
    "schema": "picf-next.ltop-g3-training-phase.v1",
    "status": "PASS",
    "failures": [],
    "phase": "training",
    "mode": "mediator-trial",
    "steps": 256,
    "eval_every": 32,
    "world_size": 2,
    "seed": 20260813,
}
for field, value in training_expected.items():
    if training_report.get(field) != value:
        fail(f"ADR170 mediator-trial training report violates {field}")
depth_contract = require_action_consumable_depth(
    training_report.get("training_contract", {}).get("task_address_supervision_depth"),
    "ADR170 mediator-trial task-address supervision depth",
)
source_contract = require_picf_source_contract(
    training_report.get("picf_source_contract"),
    "ADR170 mediator-trial PICF source contract",
)
training_receipt = training_report.get("checkpoint")
if not isinstance(training_receipt, dict):
    fail("ADR170 mediator-trial training report omits its checkpoint receipt")
receipt_path = training_receipt.get("path")
if not isinstance(receipt_path, str) or not Path(receipt_path).is_absolute():
    fail("ADR170 mediator-trial checkpoint receipt path is invalid")
if Path(receipt_path).resolve() != checkpoint_path:
    fail("ADR170 cold checkpoint path differs from the training receipt")
if training_receipt.get("format") != CHECKPOINT_FORMAT:
    fail("ADR170 mediator-trial checkpoint receipt format differs")
if training_receipt.get("optimizer_saved") is not False:
    fail("ADR170 mediator-trial checkpoint receipt contains optimizer state")
receipt_manifest_sha256 = require_sha256(
    training_receipt.get("manifest_sha256"),
    "ADR170 mediator-trial checkpoint receipt manifest digest",
)
if file_sha256(manifest_path) != receipt_manifest_sha256:
    fail("ADR170 cold checkpoint manifest differs from the training receipt")
if training_receipt.get("model_tree_schema") != MODEL_TREE_SCHEMA:
    fail("ADR170 mediator-trial checkpoint receipt model-tree schema differs")
if training_receipt.get("action_supervision_schema") != ACTION_SUPERVISION_SCHEMA:
    fail("ADR170 source-aligned checkpoint receipt supervision schema differs")
if training_receipt.get("task_address_supervision_depth") != depth_contract:
    fail("ADR170 source-aligned checkpoint receipt task-address supervision depth differs")
if training_receipt.get("picf_source_contract") != source_contract:
    fail("ADR170 source-aligned checkpoint receipt PICF source identity differs")
receipt_model_tree_sha256 = require_sha256(
    training_receipt.get("model_tree_sha256"),
    "ADR170 mediator-trial checkpoint receipt model-tree digest",
)
if receipt_model_tree_sha256 != model_tree_sha256:
    fail("ADR170 cold checkpoint model tree differs from the training receipt")
training_digests = training_receipt.get("training_final_model_local_state_sha256_by_rank")
if not isinstance(training_digests, list) or len(training_digests) != 2:
    fail("ADR170 mediator-trial checkpoint receipt must contain two training rank digests")
training_digests = [
    require_sha256(value, f"ADR170 mediator-trial checkpoint training rank {rank} digest")
    for rank, value in enumerate(training_digests)
]
for field in (
    "format",
    "optimizer_saved",
    "model_tree_schema",
    "model_tree_sha256",
    "action_supervision_schema",
    "picf_source_contract",
    "task_address_supervision_depth",
    "training_final_model_local_state_sha256_by_rank",
):
    if manifest.get(field) != training_receipt.get(field):
        fail(f"ADR170 mediator-trial checkpoint receipt and manifest differ on {field}")

schedule = (
    training_report.get("training_contract", {})
    .get("action_information_set_trial", {})
    .get("schedule")
)
schedule_sha256 = schedule.get("sha256") if isinstance(schedule, dict) else None
schedule_sha256 = require_sha256(schedule_sha256, "ADR170 mediator-trial schedule digest")
if manifest.get("action_information_set_schedule_sha256") != schedule_sha256:
    fail("ADR170 cold checkpoint schedule differs from the training report")
supervision = training_report.get("training_contract", {}).get("action_supervision")
expected_supervision = {
    "schema": ACTION_SUPERVISION_SCHEMA,
    "official_action_loss": "immutable-source-task-action-pairs-only",
    "crossed_prompt_action_loss": False,
    "crossed_prompts": "representation-and-causal-evaluation-only",
    "ambiguous_source_task_address_loss": False,
    "unobservable_source_target_address_loss": False,
    "unobservable_source_target_policy": (
        "disable-address-only-with-explicit-loss-side-receipt"
    ),
}
if supervision != expected_supervision:
    fail("ADR170 source-aligned training supervision contract differs")
if manifest.get("task_address_supervision_depth") != depth_contract:
    fail("ADR170 cold checkpoint task-address supervision depth differs")
training_ranks = rank_map(training_report, "ADR170 mediator-trial training report")
training_report_digests = [
    require_sha256(
        training_ranks[rank].get("training_final_model_local_state_sha256"),
        f"ADR170 mediator-trial training rank {rank} terminal model digest",
    )
    for rank in (0, 1)
]
if training_report_digests != training_digests:
    fail("ADR170 mediator-trial checkpoint receipt differs from training terminal states")
training_runtime_schedules = [
    require_sha256(
        training_ranks[rank].get("runtime_schedule_sha256"),
        f"ADR170 mediator-trial training rank {rank} runtime schedule digest",
    )
    for rank in (0, 1)
]
for rank in (0, 1):
    if training_ranks[rank].get("action_information_set_schedule_sha256") != schedule_sha256:
        fail(f"ADR170 mediator-trial training rank {rank} schedule digest differs")

if not report_path.is_file() or report_path.is_symlink():
    fail(f"ADR170 cold report is absent: {report_path}")
payload = json.loads(report_path.read_text(encoding="ascii"))
if not isinstance(payload, dict):
    fail(f"ADR170 cold {phase} report must be a JSON object")
expected = {
    "schema": schemas[phase],
    "status": "PASS",
    "failures": [],
    "phase": phase,
    "mode": "gate",
    "steps": 128,
    "eval_every": 32,
    "world_size": 2,
    "seed": training_report["seed"],
    "trained_checkpoint": str(checkpoint_path),
}
for field, value in expected.items():
    if payload.get(field) != value:
        fail(f"ADR170 cold {phase} report violates {field}")
if require_picf_source_contract(
    payload.get("picf_source_contract"),
    f"ADR170 cold {phase} PICF source contract",
) != source_contract:
    fail(f"ADR170 cold {phase} PICF source identity differs from training")

rank_reports = rank_map(payload, f"ADR170 cold {phase} report")
for rank in (0, 1):
    rank_report = rank_reports[rank]
    cold_loaded = require_sha256(
        rank_report.get("cold_loaded_model_local_state_sha256"),
        f"ADR170 cold {phase} rank {rank} cold-loaded model digest",
    )
    post_evaluation = require_sha256(
        rank_report.get("post_evaluation_model_local_state_sha256"),
        f"ADR170 cold {phase} rank {rank} post-evaluation model digest",
    )
    consumed_tree = require_sha256(
        rank_report.get("trained_checkpoint_model_tree_sha256"),
        f"ADR170 cold {phase} rank {rank} checkpoint model-tree digest",
    )
    if cold_loaded != post_evaluation:
        fail(f"ADR170 cold {phase} rank {rank} mutated persistent model state")
    if cold_loaded != training_digests[rank]:
        fail(f"ADR170 cold {phase} rank {rank} differs from its training terminal state")
    if consumed_tree != model_tree_sha256:
        fail(f"ADR170 cold {phase} rank {rank} consumed another checkpoint tree")
    legacy_alias = rank_report.get("trained_model_local_state_sha256")
    if legacy_alias is not None:
        require_sha256(legacy_alias, f"ADR170 cold {phase} rank {rank} legacy model digest")
        if legacy_alias != cold_loaded:
            fail(f"ADR170 cold {phase} rank {rank} legacy model digest alias differs")
    runtime_schedule = require_sha256(
        rank_report.get("runtime_schedule_sha256"),
        f"ADR170 cold {phase} rank {rank} runtime schedule digest",
    )
    if runtime_schedule != training_runtime_schedules[rank]:
        fail(f"ADR170 cold {phase} rank {rank} runtime schedule differs from training")

if phase == "evaluation":
    if payload.get("evaluation_action_information_set") != expected_action_information_set:
        fail("ADR170 action report used another action information set")
    for rank_report in rank_reports.values():
        history = rank_report.get("history")
        if not isinstance(history, list) or len(history) != 1:
            fail("ADR170 action report must contain one cold evaluation receipt")
        for partition in ("validation", "heldout"):
            partition_report = history[0].get(partition)
            partition_scenes = (
                partition_report.get("scenes") if isinstance(partition_report, dict) else None
            )
            if not isinstance(partition_scenes, list) or len(partition_scenes) != scenes:
                fail(f"ADR170 action report has the wrong {partition} scene count")
else:
    if payload.get("evaluation_action_information_set") is not None:
        fail("ADR170 retention report unexpectedly claims an action information set")
    contract = payload.get("representation_retention_contract")
    if not isinstance(contract, dict) or contract.get("scientific_action_evidence") is not False:
        fail("ADR170 retention report misstates its scientific scope")
    robustness = payload.get("scene_level_robustness")
    if not isinstance(robustness, dict) or set(robustness) != {"validation", "heldout"}:
        fail("ADR170 retention report omits partition robustness")

print(report_path.resolve())
PY
}
