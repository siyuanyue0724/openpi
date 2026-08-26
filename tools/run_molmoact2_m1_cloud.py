#!/usr/bin/env python3
"""Fail-closed launcher for the MolmoAct2 M1 typed full-manifest gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from picf_next.data.robot_record import (  # noqa: E402
    MOLMOACT2_LIBERO_DATASET_ID,
    MOLMOACT2_LIBERO_REVISION,
)
from tools.audit_molmoact2_libero_full import (  # noqa: E402
    EXPECTED_DATA_SHARDS,
    EXPECTED_EPISODES,
    EXPECTED_FILE_COUNT,
    EXPECTED_FRAMES,
    EXPECTED_LOCATOR_MISMATCHES,
    EXPECTED_TASKS,
    EXPECTED_TOTAL_BYTES,
)
from tools.run_molmoact2_m0_cloud import validate_m0_report  # noqa: E402

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_FILE_MANIFEST_SHA256 = "f41e1130258e129a105e5ff068ae7cae5d89a0fa4e51a8e0edd0f448baf4af80"
_EXPECTED_TRAIN_MANIFEST_SHA256 = "64537f2a64883d0699141063a679babff35fcac258c7fdca65dcd8b100160f41"
_EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "15231dc417d8b54e574b7e820172ac7b419ef794db79317ece737679b2e83b0e"
)
_EXPECTED_TRAIN_EPISODES = 1518
_EXPECTED_VALIDATION_EPISODES = 175
_EXPECTED_TRAIN_FRAMES = 245510
_EXPECTED_VALIDATION_FRAMES = 27955
_M1_MACHINE_REQUIRED_REPORTS = (
    "launch_manifest.json",
    "environment.json",
    "full_audit/full_audit.json",
    "full_audit/episode_file_locator_overlay.json",
    "manifests/summary.json",
    "manifests/train.jsonl",
    "manifests/validation.jsonl",
    "sample_plan.json",
    "visual_artifacts.json",
    "ddp_launch.json",
    "ddp/report.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs/cloud/2xa100_40g_gates.json",
    )
    parser.add_argument("--m0-run", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _resolve_root(argument: Path | None, environment_name: str) -> Path:
    value = argument if argument is not None else os.environ.get(environment_name)
    if value is None:
        raise RuntimeError(f"{environment_name} is unset and no override was supplied")
    return Path(value).expanduser().resolve()


def _absolute_executable(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _is_under_mnt(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == Path("/mnt") or Path("/mnt") in resolved.parents


def _run_id(value: str | None) -> str:
    resolved = value or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not _RUN_ID.fullmatch(resolved):
        raise ValueError(f"invalid M1 run id: {resolved!r}")
    return resolved


def _clean_git_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        text=True,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("M1 source revision is not a full Git commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=_ROOT,
        text=True,
    )
    if dirty:
        raise RuntimeError("M1 requires a clean worktree bound to one committed source revision")
    return revision


def validate_prior_m0(run_dir: Path, *, config: dict[str, Any]) -> dict[str, Any]:
    decision_path = run_dir / "gate_decision.json"
    raw_report_path = run_dir / "m0_raw_report.json"
    if not decision_path.is_file() or not raw_report_path.is_file():
        raise FileNotFoundError("M1 requires the complete accepted M0 run")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema") != "picf-next.m0-gate-decision.v1"
        or decision.get("status") != "PASS"
        or decision.get("gate") != "M0_full_weight_parity"
        or decision.get("later_gates_authorized") != ["M1_typed_full_manifest"]
    ):
        raise ValueError("prior M0 decision does not authorize M1")
    expected_hashes = decision.get("required_report_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("prior M0 decision omitted required report hashes")
    for relative, expected in expected_hashes.items():
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"prior M0 report hash changed: {relative}")
    raw_report = json.loads(raw_report_path.read_text())
    validate_m0_report(raw_report, config=config, root=_ROOT)
    return {
        "run_dir": str(run_dir),
        "gate_decision_sha256": _sha256(decision_path),
        "raw_report_sha256": _sha256(raw_report_path),
        "checkpoint_manifest_sha256": _canonical_sha256(
            raw_report["checkpoint_weight_shard_sha256"]
        ),
    }


def validate_full_audit(report: dict[str, Any]) -> None:
    if report.get("status") != "PASS_WITH_UPSTREAM_LOCATOR_BUG_MITIGATED":
        raise ValueError("full LIBERO audit did not pass with the accepted loader mitigation")
    if report.get("dataset_id") != MOLMOACT2_LIBERO_DATASET_ID:
        raise ValueError("full LIBERO audit dataset ID changed")
    if report.get("dataset_revision") != MOLMOACT2_LIBERO_REVISION:
        raise ValueError("full LIBERO audit revision changed")
    tree = report.get("tree", {})
    expected_tree = {
        "files": EXPECTED_FILE_COUNT,
        "data_shards": EXPECTED_DATA_SHARDS,
        "bytes": EXPECTED_TOTAL_BYTES,
        "hashes_verified": True,
        "canonical_file_manifest_sha256": _EXPECTED_FILE_MANIFEST_SHA256,
    }
    for key, expected in expected_tree.items():
        if tree.get(key) != expected:
            raise ValueError(f"full LIBERO audit tree field changed: {key}")
    rows = report.get("rows", {})
    if (
        rows.get("episodes") != EXPECTED_EPISODES
        or rows.get("frames") != EXPECTED_FRAMES
        or rows.get("tasks") != EXPECTED_TASKS
        or rows.get("fps") != 10
        or rows.get("state_shape") != [8]
        or rows.get("action_shape") != [7]
    ):
        raise ValueError("full LIBERO row contract changed")
    locator = report.get("upstream_episode_locator", {})
    if (
        locator.get("mismatched_episodes") != EXPECTED_LOCATOR_MISMATCHES
        or locator.get("raw_files_modified") is not False
    ):
        raise ValueError("upstream locator defect/mitigation contract changed")
    visual = report.get("visual_sample", {})
    if visual.get("decoded_images") != 240 or len(visual.get("representatives", ())) != 40:
        raise ValueError("full LIBERO visual representative coverage changed")


def validate_split_summary(summary: dict[str, Any]) -> None:
    expected = {
        "schema": "picf-next.libero-episode-split.v1",
        "dataset_id": MOLMOACT2_LIBERO_DATASET_ID,
        "dataset_revision": MOLMOACT2_LIBERO_REVISION,
        "train_episodes": _EXPECTED_TRAIN_EPISODES,
        "validation_episodes": _EXPECTED_VALIDATION_EPISODES,
        "train_frames": _EXPECTED_TRAIN_FRAMES,
        "validation_frames": _EXPECTED_VALIDATION_FRAMES,
        "tasks_each_arm": EXPECTED_TASKS,
        "train_manifest_sha256": _EXPECTED_TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": _EXPECTED_VALIDATION_MANIFEST_SHA256,
        "locator_fields_used": False,
        "episode_task_stats_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"LIBERO episode split changed required field {key}")


def build_sample_plan(
    *,
    audit_report: dict[str, Any],
    locator_overlay: list[dict[str, Any]],
) -> dict[str, Any]:
    locator_by_episode = {
        int(record["episode_index"]): bool(record["mismatch"]) for record in locator_overlay
    }
    representatives: list[dict[str, Any]] = []
    phase_names = ("start", "middle", "end")
    for record in audit_report["visual_sample"]["representatives"]:
        phase_indices = record.get("phase_global_indices")
        if not isinstance(phase_indices, list) or len(phase_indices) != len(phase_names):
            raise ValueError("full audit representative phase indices are malformed")
        episode_index = int(record["episode_index"])
        for phase, global_index in zip(phase_names, phase_indices, strict=True):
            representatives.append(
                {
                    "sample_key": (
                        f"task-{int(record['task_index']):02d}/"
                        f"episode-{episode_index:04d}/{phase}/global-{int(global_index):06d}"
                    ),
                    "task_index": int(record["task_index"]),
                    "task": str(record["task"]),
                    "episode_index": episode_index,
                    "episode_length": int(record["length"]),
                    "phase": phase,
                    "global_index": int(global_index),
                    "locator_mismatch": locator_by_episode[episode_index],
                }
            )
    representatives.sort(key=lambda row: (row["task_index"], phase_names.index(row["phase"])))
    if len(representatives) != EXPECTED_TASKS * len(phase_names):
        raise ValueError("M1 sample plan must contain 120 deterministic rows")
    if len({row["sample_key"] for row in representatives}) != len(representatives):
        raise ValueError("M1 sample plan contains duplicate sample keys")
    return {
        "schema": "picf-next.molmoact2-m1-sample-plan.v1",
        "dataset_id": MOLMOACT2_LIBERO_DATASET_ID,
        "dataset_revision": MOLMOACT2_LIBERO_REVISION,
        "selection_rule": "one deterministic lower-median episode per task; start/middle/end",
        "tasks": EXPECTED_TASKS,
        "representatives": representatives,
        "representatives_sha256": _canonical_sha256(representatives),
    }


def build_visual_artifact_manifest(
    *,
    audit_dir: Path,
    audit_report: dict[str, Any],
) -> dict[str, Any]:
    visual = audit_report.get("visual_sample", {})
    representatives = visual.get("representatives")
    if not isinstance(representatives, list) or len(representatives) != EXPECTED_TASKS:
        raise ValueError("M1 visual artifact manifest requires all 40 task panels")
    overview_name = visual.get("overview")
    if not isinstance(overview_name, str) or not overview_name:
        raise ValueError("M1 full audit omitted the visual overview")
    overview_path = audit_dir / overview_name
    if not overview_path.is_file():
        raise FileNotFoundError(overview_path)

    panels: list[dict[str, Any]] = []
    for record in sorted(representatives, key=lambda row: int(row["task_index"])):
        task_index = int(record["task_index"])
        panel_name = record.get("panel")
        if not isinstance(panel_name, str) or not panel_name:
            raise ValueError(f"M1 full audit omitted task {task_index} panel identity")
        panel_path = audit_dir / panel_name
        if not panel_path.is_file():
            raise FileNotFoundError(panel_path)
        panels.append(
            {
                "task_index": task_index,
                "task": str(record["task"]),
                "episode_index": int(record["episode_index"]),
                "path": f"full_audit/{panel_name}",
                "sha256": _sha256(panel_path),
            }
        )
    if [item["task_index"] for item in panels] != list(range(EXPECTED_TASKS)):
        raise ValueError("M1 visual panels do not cover task indices 0 through 39 exactly")
    return {
        "schema": "picf-next.molmoact2-m1-visual-artifacts.v1",
        "full_audit_sha256": _sha256(audit_dir / "full_audit.json"),
        "locator_overlay_sha256": _sha256(audit_dir / "episode_file_locator_overlay.json"),
        "overview": {
            "path": f"full_audit/{overview_name}",
            "sha256": _sha256(overview_path),
        },
        "panels": panels,
        "tasks": EXPECTED_TASKS,
        "decoded_images": EXPECTED_TASKS * 3 * 2,
    }


def validate_visual_artifact_manifest(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
) -> None:
    if manifest.get("schema") != "picf-next.molmoact2-m1-visual-artifacts.v1":
        raise ValueError("unsupported M1 visual artifact manifest")
    if manifest.get("tasks") != EXPECTED_TASKS or manifest.get("decoded_images") != 240:
        raise ValueError("M1 visual artifact coverage changed")
    fixed_files = {
        "full_audit/full_audit.json": manifest.get("full_audit_sha256"),
        "full_audit/episode_file_locator_overlay.json": manifest.get("locator_overlay_sha256"),
    }
    overview = manifest.get("overview")
    if not isinstance(overview, dict):
        raise ValueError("M1 visual artifact overview is malformed")
    overview_path = overview.get("path")
    if not isinstance(overview_path, str) or not overview_path.startswith("full_audit/"):
        raise ValueError("M1 visual artifact overview path is malformed")
    fixed_files[overview_path] = overview.get("sha256")
    panels = manifest.get("panels")
    if not isinstance(panels, list) or len(panels) != EXPECTED_TASKS:
        raise ValueError("M1 visual artifact panel inventory is incomplete")
    if [item.get("task_index") for item in panels] != list(range(EXPECTED_TASKS)):
        raise ValueError("M1 visual artifact task ordering changed")
    for item in panels:
        path = item.get("path")
        if not isinstance(path, str) or not path.startswith("full_audit/task_"):
            raise ValueError("M1 visual panel path is malformed")
        fixed_files[path] = item.get("sha256")
    for relative, expected in fixed_files.items():
        if not _SHA256.fullmatch(str(expected or "")):
            raise ValueError(f"M1 visual artifact digest is malformed: {relative}")
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"M1 visual artifact changed after machine audit: {relative}")


def validate_m1_machine_decision(run_dir: Path) -> dict[str, Any]:
    decision_path = run_dir / "machine_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("M1 machine decision is absent")
    decision = json.loads(decision_path.read_text())
    if (
        decision.get("schema") != "picf-next.molmoact2-m1-machine-decision.v1"
        or decision.get("status") != "PASS_PENDING_VISUAL_REVIEW"
        or decision.get("gate") != "M1_typed_full_manifest"
        or decision.get("later_gates_authorized") != []
    ):
        raise ValueError("M1 machine decision is not awaiting visual review")
    hashes = decision.get("required_report_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_M1_MACHINE_REQUIRED_REPORTS):
        raise ValueError("M1 machine decision report inventory changed")
    for relative, expected in hashes.items():
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"M1 machine report changed after acceptance: {relative}")
    visual_manifest = json.loads((run_dir / "visual_artifacts.json").read_text())
    validate_visual_artifact_manifest(run_dir=run_dir, manifest=visual_manifest)
    return decision


def validate_m1_visual_review(
    review: dict[str, Any],
    *,
    run_dir: Path,
) -> None:
    if review.get("schema") != "picf-next.molmoact2-m1-visual-review.v1":
        raise ValueError("unsupported M1 visual review schema")
    if review.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("M1 visual review status must be PASS or FAIL")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("M1 visual review requires a named reviewer")
    if not isinstance(review.get("date"), str) or not review["date"].strip():
        raise ValueError("M1 visual review requires a dated record")
    visual_manifest_path = run_dir / "visual_artifacts.json"
    visual_manifest = json.loads(visual_manifest_path.read_text())
    validate_visual_artifact_manifest(run_dir=run_dir, manifest=visual_manifest)
    expected_bindings = {
        "machine_report_sha256": _sha256(run_dir / "full_audit/full_audit.json"),
        "visual_artifacts_sha256": _sha256(visual_manifest_path),
        "overview_sha256": visual_manifest["overview"]["sha256"],
        "locator_overlay_file_sha256": visual_manifest["locator_overlay_sha256"],
    }
    for key, expected in expected_bindings.items():
        if review.get(key) != expected:
            raise ValueError(f"M1 visual review is not bound to this run: {key}")
    if review.get("overview_tasks_reviewed") != 40 or review.get("decoded_images_reviewed") != 240:
        raise ValueError("M1 visual review did not cover the complete deterministic sample")
    enlarged = review.get("individually_enlarged_task_indices")
    if (
        not isinstance(enlarged, list)
        or len(enlarged) != len(set(enlarged))
        or any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < EXPECTED_TASKS
            for index in enlarged
        )
    ):
        raise ValueError("M1 enlarged visual-review task inventory is malformed")
    if review["status"] == "PASS":
        suite_missing = any(
            not any(start <= index < start + 10 for index in enlarged) for start in range(0, 40, 10)
        )
        if suite_missing:
            raise ValueError("M1 visual review did not enlarge at least one task per LIBERO suite")
        expected_checks = {
            "task_text_matches_external_trajectory": True,
            "start_middle_end_order_is_coherent": True,
            "external_and_wrist_camera_roles_are_consistent": True,
            "button_drawer_container_and_spatial_relation_tasks_are_coherent": True,
            "obvious_cross_episode_or_cross_task_row_mix": False,
        }
        if review.get("checks") != expected_checks:
            raise ValueError("M1 PASS visual review checks are incomplete or changed")
    observations = review.get("observations")
    if (
        not isinstance(observations, list)
        or not observations
        or any(not isinstance(item, str) or not item.strip() for item in observations)
    ):
        raise ValueError("M1 visual review requires concrete observations")


def validate_m1_ddp_report(report: dict[str, Any]) -> None:
    if (
        report.get("schema") != "picf-next.molmoact2-m1-ddp.v1"
        or report.get("status") != "PASS"
        or report.get("gate") != "M1_typed_full_manifest"
        or report.get("world_size") != 2
    ):
        raise ValueError("M1 DDP processor report did not pass")
    dataset = report.get("dataset", {})
    expected_dataset = {
        "id": MOLMOACT2_LIBERO_DATASET_ID,
        "revision": MOLMOACT2_LIBERO_REVISION,
        "selected_episodes": EXPECTED_TASKS,
        "selected_tasks": EXPECTED_TASKS,
        "selected_representative_rows": 120,
        "official_loader": "lerobot.datasets.io_utils.load_nested_dataset",
        "loader_discovers_all_physical_shards_once": True,
        "episode_filter_field": "episode_index",
        "episode_locator_fields_used": False,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise ValueError(f"M1 DDP loader contract changed field {key}")
    typed = report.get("typed_contract", {})
    if (
        typed.get("state_shape") != [8]
        or typed.get("action_shape") != [10, 7]
        or typed.get("delta_t_s") != 0.1
        or typed.get("metadata_state_names_trusted") is not False
    ):
        raise ValueError("M1 typed state/action contract changed")
    processor = report.get("processor", {})
    if (
        processor.get("factory") != "make_molmoact2_pre_post_processors"
        or processor.get("all_representatives_processed") is not True
        or processor.get("action_mode") != "continuous"
        or processor.get("action_horizon") != 10
    ):
        raise ValueError("M1 official processor contract changed")
    no_leak = report.get("no_leak", {})
    required_no_leak = (
        "target_free_action_is_none",
        "target_free_labels_absent",
        "targetful_labels_absent_for_continuous_mode",
        "observation_tensors_exactly_equal_with_and_without_action_target",
    )
    if no_leak.get("representative_rows_checked") != 120 or any(
        no_leak.get(key) is not True for key in required_no_leak
    ):
        raise ValueError("M1 target-free no-leak contract failed")
    continuation = report.get("continuation", {})
    required_continuation = (
        "checkpoint_resume_exact",
        "rank_local_cursor_exact",
        "rng_exact",
        "loader_processor_trace_exact",
        "optimizer_scheduler_model_exact",
        "corrupted_checkpoint_failed_closed_on_all_ranks",
    )
    if any(continuation.get(key) is not True for key in required_continuation):
        raise ValueError("M1 exact continuation/corruption contract failed")
    if not _SHA256.fullmatch(str(continuation.get("model_sha256", ""))):
        raise ValueError("M1 final model digest is malformed")
    resources = report.get("resources")
    if not isinstance(resources, list) or len(resources) != 2:
        raise ValueError("M1 omitted per-rank CUDA resource reports")
    if any("A100" not in str(item.get("device_name", "")) for item in resources):
        raise ValueError("M1 did not run on two A100 devices")


def _run_stage(
    *,
    name: str,
    command: list[str],
    run_dir: Path,
    environment: dict[str, str],
) -> None:
    with (
        (run_dir / f"{name}.stdout.log").open("w") as stdout,
        (run_dir / f"{name}.stderr.log").open("w") as stderr,
    ):
        result = subprocess.run(
            command,
            cwd=_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(f"M1 stage {name} failed with return code {result.returncode}")


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    from tools.preflight_cloud import validate_config

    validate_config(config)
    if config["profile"] != "molmoact2-causal-2xa100-40g":
        raise ValueError("this launcher accepts only the frozen MolmoAct2 cloud profile")
    checkpoint_root = _resolve_root(args.checkpoint_root, config["paths"]["checkpoint_root_env"])
    dataset_root_parent = _resolve_root(args.dataset_root, config["paths"]["dataset_root_env"])
    run_root = _resolve_root(args.run_root, config["paths"]["run_root_env"])
    dataset_root = dataset_root_parent / "molmoact2_libero_full"
    checkpoint_dir = checkpoint_root / config["host"]["checkpoint_subdir"]
    run_id = _run_id(args.run_id)
    run_dir = run_root / "molmoact2" / "M1_typed_full_manifest" / run_id
    python = _absolute_executable(args.python)
    m0_contract = validate_prior_m0(args.m0_run.resolve(), config=config)
    code_revision = _clean_git_revision()
    commands = {
        "preflight": [
            str(python),
            str(_ROOT / "tools/preflight_cloud.py"),
            "--config",
            str(config_path),
            "--check-runtime",
            "--json-out",
            str(run_dir / "environment.json"),
        ],
        "full_audit": [
            str(python),
            str(_ROOT / "tools/audit_molmoact2_libero_full.py"),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(run_dir / "full_audit"),
        ],
        "split_manifest": [
            str(python),
            str(_ROOT / "tools/build_libero_episode_manifests.py"),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(run_dir / "manifests"),
        ],
    }
    manifest = {
        "schema": "picf-next.molmoact2-m1-launch.v1",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "python": str(python),
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_root": str(dataset_root),
        "run_root": str(run_root),
        "prior_m0": m0_contract,
        "code_revision": code_revision,
        "worktree_clean": True,
        "commands": commands,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    for name, path in {
        "checkpoint_root": checkpoint_root,
        "dataset_root": dataset_root,
        "run_root": run_root,
        "m0_run": args.m0_run.resolve(),
    }.items():
        if not _is_under_mnt(path):
            raise RuntimeError(f"M1 {name} must be persisted under /mnt: {path}")
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing M1 run: {run_dir}")
    if not checkpoint_dir.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("M1 checkpoint or complete LIBERO dataset is absent")
    run_dir.mkdir(parents=True)
    _write_json_atomic(run_dir / "launch_manifest.json", manifest)

    environment = os.environ.copy()
    environment[config["paths"]["checkpoint_root_env"]] = str(checkpoint_root)
    environment[config["paths"]["dataset_root_env"]] = str(dataset_root_parent)
    environment[config["paths"]["run_root_env"]] = str(run_root)
    python_paths = [
        str(_ROOT),
        str(_ROOT / "src"),
        str(_ROOT / config["host"]["source_checkout"] / "experiments"),
        str(_ROOT / config["host"]["training_source_checkout"] / "src"),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    current_stage = "initialization"
    try:
        for current_stage in ("preflight", "full_audit", "split_manifest"):
            _run_stage(
                name=current_stage,
                command=commands[current_stage],
                run_dir=run_dir,
                environment=environment,
            )
        audit_report = json.loads((run_dir / "full_audit/full_audit.json").read_text())
        validate_full_audit(audit_report)
        split_summary = json.loads((run_dir / "manifests/summary.json").read_text())
        validate_split_summary(split_summary)
        locator_overlay = json.loads(
            (run_dir / "full_audit/episode_file_locator_overlay.json").read_text()
        )
        sample_plan = build_sample_plan(
            audit_report=audit_report,
            locator_overlay=locator_overlay,
        )
        _write_json_atomic(run_dir / "sample_plan.json", sample_plan)
        visual_artifacts = build_visual_artifact_manifest(
            audit_dir=run_dir / "full_audit",
            audit_report=audit_report,
        )
        _write_json_atomic(run_dir / "visual_artifacts.json", visual_artifacts)

        current_stage = "ddp_loader_processor_resume"
        ddp_command = [
            str(python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(_ROOT / "tools/smoke_molmoact2_m1_ddp.py"),
            "--dataset-root",
            str(dataset_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--sample-plan",
            str(run_dir / "sample_plan.json"),
            "--output-dir",
            str(run_dir / "ddp"),
            "--code-revision",
            code_revision,
            "--host-source-revision",
            config["host"]["source_commit"],
            "--training-source-revision",
            config["host"]["training_source_commit"],
            "--checkpoint-manifest-sha256",
            m0_contract["checkpoint_manifest_sha256"],
            "--dataset-manifest-sha256",
            audit_report["tree"]["canonical_file_manifest_sha256"],
        ]
        commands[current_stage] = ddp_command
        _write_json_atomic(
            run_dir / "ddp_launch.json",
            {
                "schema": "picf-next.molmoact2-m1-ddp-launch.v1",
                "command": ddp_command,
                "cwd": str(_ROOT),
                "code_revision": code_revision,
                "pythonpath": python_paths,
                "sample_plan_sha256": _sha256(run_dir / "sample_plan.json"),
                "world_size": 2,
            },
        )
        _run_stage(
            name=current_stage,
            command=ddp_command,
            run_dir=run_dir,
            environment=environment,
        )
        ddp_report = json.loads((run_dir / "ddp/report.json").read_text())
        validate_m1_ddp_report(ddp_report)
        report_hashes = {name: _sha256(run_dir / name) for name in _M1_MACHINE_REQUIRED_REPORTS}
    except Exception as error:
        _write_json_atomic(
            run_dir / "gate_decision.json",
            {
                "schema": "picf-next.molmoact2-m1-gate-decision.v1",
                "status": "FAIL",
                "gate": "M1_typed_full_manifest",
                "failed_stage": current_stage,
                "error": f"{type(error).__name__}: {error}",
                "later_gates_authorized": [],
            },
        )
        raise

    decision = {
        "schema": "picf-next.molmoact2-m1-machine-decision.v1",
        "status": "PASS_PENDING_VISUAL_REVIEW",
        "gate": "M1_typed_full_manifest",
        "required_report_sha256": report_hashes,
        "later_gates_authorized": [],
        "next_required_action": (
            "inspect the immutable 40-task visual sample and run tools/finalize_molmoact2_m1.py"
        ),
    }
    _write_json_atomic(run_dir / "machine_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
