#!/usr/bin/env bash

adr165_resolve_mediator_trial_checkpoint() {
  if [[ $# -ne 2 ]]; then
    echo "usage: adr165_resolve_mediator_trial_checkpoint PYTHON_BIN TRAINING_REPORT" >&2
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
CHECKPOINT_MANIFEST_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v2"
MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def fail(message):
    raise SystemExit(message)


def require_sha256(value, name):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        fail(f"{name} must be one lowercase SHA-256")
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
manifest_expected = {
    "schema": CHECKPOINT_MANIFEST_SCHEMA,
    "status": "PASS",
    "global_step": 256,
    "optimizer_saved": False,
    "format": CHECKPOINT_FORMAT,
    "world_size": 2,
    "model_tree_schema": MODEL_TREE_SCHEMA,
    "model_tree_sha256": actual_model_tree_sha256,
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

adr165_validate_cold_report() {
  if [[ $# -ne 5 ]]; then
    echo "usage: adr165_validate_cold_report PYTHON_BIN REPORT CHECKPOINT PHASE SCENES" >&2
    return 2
  fi

  local python_bin=$1
  local report=$2
  local checkpoint=$3
  local phase=$4
  local scenes=$5
  "$python_bin" - "$report" "$checkpoint" "$phase" "$scenes" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
CHECKPOINT_MANIFEST_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v2"
MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def fail(message):
    raise SystemExit(message)


def require_sha256(value, name):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        fail(f"{name} must be one lowercase SHA-256")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(root, schema):
    if root.is_symlink():
        fail("ADR165 cold checkpoint model must be one direct directory")
    directory = root.resolve()
    if not directory.is_dir():
        fail("ADR165 cold checkpoint model directory is absent")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            fail("ADR165 cold checkpoint model tree contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            fail("ADR165 cold checkpoint model tree contains a non-regular entry")
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        fail("ADR165 cold checkpoint model tree is empty")
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
schemas = {
    "evaluation": "picf-next.ltop-g3-evaluation-phase.v1",
    "retention": "picf-next.ltop-g3-representation-retention.v1",
}
if phase not in schemas:
    fail(f"unsupported ADR165 cold phase: {phase}")
if not checkpoint_path.is_absolute():
    fail("ADR165 cold checkpoint path must be absolute")
if not checkpoint_path.is_dir() or checkpoint_path.is_symlink():
    fail(f"ADR165 cold checkpoint is absent: {checkpoint_path}")
checkpoint_path = checkpoint_path.resolve()
if not checkpoint_path.is_relative_to(Path("/mnt")):
    fail("ADR165 cold checkpoint must live under /mnt")

root_entries = list(checkpoint_path.iterdir())
if any(path.is_symlink() for path in root_entries):
    fail("ADR165 cold checkpoint root contains a symbolic link")
if {path.name for path in root_entries} != {"model", "ltop_g3_training_checkpoint.json"}:
    fail("ADR165 cold checkpoint root differs from the model-only ABI")
model_path = checkpoint_path / "model"
if not model_path.is_dir() or model_path.is_symlink():
    fail("ADR165 cold checkpoint model directory is absent")
metadata_path = model_path / ".metadata"
if not metadata_path.is_file() or metadata_path.is_symlink():
    fail("ADR165 cold checkpoint omits direct DCP model/.metadata")
distcp_paths = list(model_path.glob("*.distcp"))
if not distcp_paths:
    fail("ADR165 cold checkpoint omits a direct DCP *.distcp payload")
if any(not path.is_file() or path.is_symlink() for path in distcp_paths):
    fail("ADR165 cold checkpoint DCP payload is not one direct regular file")
model_tree_sha256 = directory_tree_sha256(model_path, MODEL_TREE_SCHEMA)

manifest_path = checkpoint_path / "ltop_g3_training_checkpoint.json"
if not manifest_path.is_file() or manifest_path.is_symlink():
    fail("ADR165 cold checkpoint manifest is absent")
manifest = json.loads(manifest_path.read_text(encoding="ascii"))
if not isinstance(manifest, dict):
    fail("ADR165 cold checkpoint manifest must be a JSON object")
manifest_expected = {
    "schema": CHECKPOINT_MANIFEST_SCHEMA,
    "status": "PASS",
    "global_step": 256,
    "optimizer_saved": False,
    "format": CHECKPOINT_FORMAT,
    "world_size": 2,
    "model_tree_schema": MODEL_TREE_SCHEMA,
    "model_tree_sha256": model_tree_sha256,
}
for field, value in manifest_expected.items():
    if manifest.get(field) != value:
        fail(f"ADR165 cold checkpoint manifest violates {field}")

training_report_path = checkpoint_path.parent / "ltop_g3_mediator_trial_training_report.json"
if not training_report_path.is_file() or training_report_path.is_symlink():
    fail("ADR165 cold checkpoint is detached from its mediator-trial training report")
training_report = json.loads(training_report_path.read_text(encoding="ascii"))
if not isinstance(training_report, dict):
    fail("ADR165 mediator-trial training report must be a JSON object")
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
        fail(f"ADR165 mediator-trial training report violates {field}")
training_receipt = training_report.get("checkpoint")
if not isinstance(training_receipt, dict):
    fail("ADR165 mediator-trial training report omits its checkpoint receipt")
receipt_path = training_receipt.get("path")
if not isinstance(receipt_path, str) or not Path(receipt_path).is_absolute():
    fail("ADR165 mediator-trial checkpoint receipt path is invalid")
if Path(receipt_path).resolve() != checkpoint_path:
    fail("ADR165 cold checkpoint path differs from the training receipt")
if training_receipt.get("format") != CHECKPOINT_FORMAT:
    fail("ADR165 mediator-trial checkpoint receipt format differs")
if training_receipt.get("optimizer_saved") is not False:
    fail("ADR165 mediator-trial checkpoint receipt contains optimizer state")
receipt_manifest_sha256 = require_sha256(
    training_receipt.get("manifest_sha256"),
    "ADR165 mediator-trial checkpoint receipt manifest digest",
)
if file_sha256(manifest_path) != receipt_manifest_sha256:
    fail("ADR165 cold checkpoint manifest differs from the training receipt")
if training_receipt.get("model_tree_schema") != MODEL_TREE_SCHEMA:
    fail("ADR165 mediator-trial checkpoint receipt model-tree schema differs")
receipt_model_tree_sha256 = require_sha256(
    training_receipt.get("model_tree_sha256"),
    "ADR165 mediator-trial checkpoint receipt model-tree digest",
)
if receipt_model_tree_sha256 != model_tree_sha256:
    fail("ADR165 cold checkpoint model tree differs from the training receipt")
training_digests = training_receipt.get("training_final_model_local_state_sha256_by_rank")
if not isinstance(training_digests, list) or len(training_digests) != 2:
    fail("ADR165 mediator-trial checkpoint receipt must contain two training rank digests")
training_digests = [
    require_sha256(value, f"ADR165 mediator-trial checkpoint training rank {rank} digest")
    for rank, value in enumerate(training_digests)
]
for field in (
    "format",
    "optimizer_saved",
    "model_tree_schema",
    "model_tree_sha256",
    "training_final_model_local_state_sha256_by_rank",
):
    if manifest.get(field) != training_receipt.get(field):
        fail(f"ADR165 mediator-trial checkpoint receipt and manifest differ on {field}")

schedule = (
    training_report.get("training_contract", {})
    .get("action_information_set_trial", {})
    .get("schedule")
)
schedule_sha256 = schedule.get("sha256") if isinstance(schedule, dict) else None
schedule_sha256 = require_sha256(schedule_sha256, "ADR165 mediator-trial schedule digest")
if manifest.get("action_information_set_schedule_sha256") != schedule_sha256:
    fail("ADR165 cold checkpoint schedule differs from the training report")
training_ranks = rank_map(training_report, "ADR165 mediator-trial training report")
training_report_digests = [
    require_sha256(
        training_ranks[rank].get("training_final_model_local_state_sha256"),
        f"ADR165 mediator-trial training rank {rank} terminal model digest",
    )
    for rank in (0, 1)
]
if training_report_digests != training_digests:
    fail("ADR165 mediator-trial checkpoint receipt differs from training terminal states")
training_runtime_schedules = [
    require_sha256(
        training_ranks[rank].get("runtime_schedule_sha256"),
        f"ADR165 mediator-trial training rank {rank} runtime schedule digest",
    )
    for rank in (0, 1)
]
for rank in (0, 1):
    if training_ranks[rank].get("action_information_set_schedule_sha256") != schedule_sha256:
        fail(f"ADR165 mediator-trial training rank {rank} schedule digest differs")

if not report_path.is_file() or report_path.is_symlink():
    fail(f"ADR165 cold report is absent: {report_path}")
payload = json.loads(report_path.read_text(encoding="ascii"))
if not isinstance(payload, dict):
    fail(f"ADR165 cold {phase} report must be a JSON object")
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
        fail(f"ADR165 cold {phase} report violates {field}")

rank_reports = rank_map(payload, f"ADR165 cold {phase} report")
for rank in (0, 1):
    rank_report = rank_reports[rank]
    cold_loaded = require_sha256(
        rank_report.get("cold_loaded_model_local_state_sha256"),
        f"ADR165 cold {phase} rank {rank} cold-loaded model digest",
    )
    post_evaluation = require_sha256(
        rank_report.get("post_evaluation_model_local_state_sha256"),
        f"ADR165 cold {phase} rank {rank} post-evaluation model digest",
    )
    consumed_tree = require_sha256(
        rank_report.get("trained_checkpoint_model_tree_sha256"),
        f"ADR165 cold {phase} rank {rank} checkpoint model-tree digest",
    )
    if cold_loaded != post_evaluation:
        fail(f"ADR165 cold {phase} rank {rank} mutated persistent model state")
    if cold_loaded != training_digests[rank]:
        fail(f"ADR165 cold {phase} rank {rank} differs from its training terminal state")
    if consumed_tree != model_tree_sha256:
        fail(f"ADR165 cold {phase} rank {rank} consumed another checkpoint tree")
    legacy_alias = rank_report.get("trained_model_local_state_sha256")
    if legacy_alias is not None:
        require_sha256(legacy_alias, f"ADR165 cold {phase} rank {rank} legacy model digest")
        if legacy_alias != cold_loaded:
            fail(f"ADR165 cold {phase} rank {rank} legacy model digest alias differs")
    runtime_schedule = require_sha256(
        rank_report.get("runtime_schedule_sha256"),
        f"ADR165 cold {phase} rank {rank} runtime schedule digest",
    )
    if runtime_schedule != training_runtime_schedules[rank]:
        fail(f"ADR165 cold {phase} rank {rank} runtime schedule differs from training")

if phase == "evaluation":
    for rank_report in rank_reports.values():
        history = rank_report.get("history")
        if not isinstance(history, list) or len(history) != 1:
            fail("ADR165 action report must contain one cold evaluation receipt")
        for partition in ("validation", "heldout"):
            partition_report = history[0].get(partition)
            partition_scenes = (
                partition_report.get("scenes") if isinstance(partition_report, dict) else None
            )
            if not isinstance(partition_scenes, list) or len(partition_scenes) != scenes:
                fail(f"ADR165 action report has the wrong {partition} scene count")
else:
    contract = payload.get("representation_retention_contract")
    if not isinstance(contract, dict) or contract.get("scientific_action_evidence") is not False:
        fail("ADR165 retention report misstates its scientific scope")
    robustness = payload.get("scene_level_robustness")
    if not isinstance(robustness, dict) or set(robustness) != {"validation", "heldout"}:
        fail("ADR165 retention report omits partition robustness")

print(report_path.resolve())
PY
}
