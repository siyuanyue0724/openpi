from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from stat import S_IXUSR

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "adr165/mediator_trial_cold_evaluation_contract.sh"

CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
CHECKPOINT_MANIFEST_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v2"
MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_tree_sha256(root: Path) -> str:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return hashlib.sha256(
        _canonical_json({"schema": MODEL_TREE_SCHEMA, "files": files}).encode("ascii")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_canonical_json(payload) + "\n", encoding="ascii")


@contextmanager
def _persistent_temp_directory() -> Iterator[Path]:
    candidates = [Path("/mnt")]
    if Path("/mnt").is_dir():
        candidates.extend(path for path in sorted(Path("/mnt").iterdir()) if path.is_dir())
    for candidate in candidates:
        if not os.access(candidate, os.W_OK):
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="adr165-v2-test-", dir=candidate) as value:
                yield Path(value)
                return
        except OSError:
            continue
    pytest.skip("ADR165 functional preflight test requires a writable directory under /mnt")


def _build_v2_artifacts(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    checkpoint = root / "checkpoint-model-only"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"dcp metadata\n")
    (model / "__0_0.distcp").write_bytes(b"model shard\n")

    training_digests = ["1" * 64, "2" * 64]
    schedule_sha256 = "3" * 64
    runtime_schedule_sha256 = "4" * 64
    g2_report_sha256 = "5" * 64
    runtime_source_contract = {"repository_commit": "6" * 40}
    model_tree_sha256 = _directory_tree_sha256(model)
    manifest = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "status": "PASS",
        "global_step": 256,
        "optimizer_saved": False,
        "format": CHECKPOINT_FORMAT,
        "world_size": 2,
        "model_tree_schema": MODEL_TREE_SCHEMA,
        "model_tree_sha256": model_tree_sha256,
        "training_final_model_local_state_sha256_by_rank": training_digests,
        "action_information_set_schedule_sha256": schedule_sha256,
        "action_information_set_counts_by_rank": [
            {"factual": 128, "mediator-required": 128},
            {"factual": 128, "mediator-required": 128},
        ],
        "source_stage_checkpoint": "/mnt/picf-next/checkpoints/g2b",
        "g2_report_sha256": g2_report_sha256,
        "runtime_source_contract": runtime_source_contract,
    }
    manifest_path = checkpoint / "ltop_g3_training_checkpoint.json"
    _write_json(manifest_path, manifest)

    rank_reports = [
        {
            "rank": rank,
            "training_final_model_local_state_sha256": training_digests[rank],
            "runtime_schedule_sha256": runtime_schedule_sha256,
            "action_information_set_schedule_sha256": schedule_sha256,
        }
        for rank in (0, 1)
    ]
    training_report: dict[str, object] = {
        "schema": "picf-next.ltop-g3-training-phase.v1",
        "status": "PASS",
        "failures": [],
        "phase": "training",
        "mode": "mediator-trial",
        "steps": 256,
        "eval_every": 32,
        "world_size": 2,
        "seed": 20260813,
        "stage_checkpoint": "/mnt/picf-next/checkpoints/g2b",
        "g2_report_sha256": g2_report_sha256,
        "runtime_source_contract": runtime_source_contract,
        "training_contract": {
            "action_information_set_trial": {"schedule": {"sha256": schedule_sha256}}
        },
        "rank_reports": rank_reports,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "format": CHECKPOINT_FORMAT,
            "optimizer_saved": False,
            "manifest_sha256": _file_sha256(manifest_path),
            "model_tree_schema": MODEL_TREE_SCHEMA,
            "model_tree_sha256": model_tree_sha256,
            "training_final_model_local_state_sha256_by_rank": training_digests,
        },
    }
    training_path = root / "ltop_g3_mediator_trial_training_report.json"
    _write_json(training_path, training_report)

    cold_report = {
        "schema": "picf-next.ltop-g3-evaluation-phase.v1",
        "status": "PASS",
        "failures": [],
        "phase": "evaluation",
        "mode": "gate",
        "steps": 128,
        "eval_every": 32,
        "world_size": 2,
        "seed": 20260813,
        "trained_checkpoint": str(checkpoint.resolve()),
        "rank_reports": [
            {
                "rank": rank,
                "cold_loaded_model_local_state_sha256": training_digests[rank],
                "post_evaluation_model_local_state_sha256": training_digests[rank],
                "trained_checkpoint_model_tree_sha256": model_tree_sha256,
                "trained_model_local_state_sha256": training_digests[rank],
                "runtime_schedule_sha256": runtime_schedule_sha256,
                "history": [
                    {
                        "validation": {"scenes": [{}]},
                        "heldout": {"scenes": [{}]},
                    }
                ],
            }
            for rank in (0, 1)
        ],
    }
    cold_path = root / "cold.json"
    _write_json(cold_path, cold_report)
    return training_path, cold_path, checkpoint, training_report


def _run_contract(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(CONTRACT))}\n{command}"],
        check=False,
        capture_output=True,
        text=True,
    )


def test_mediator_checkpoint_contract_is_exact_and_model_only() -> None:
    source = _source("adr165/mediator_trial_cold_evaluation_contract.sh")

    assert '"schema": "picf-next.ltop-g3-training-phase.v1"' in source
    assert '"status": "PASS"' in source
    assert '"failures": []' in source
    assert '"phase": "training"' in source
    assert '"mode": "mediator-trial"' in source
    assert '"steps": 256' in source
    assert '"eval_every": 32' in source
    assert '"world_size": 2' in source
    assert 'CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"' in source
    assert 'checkpoint.get("format") != CHECKPOINT_FORMAT' in source
    assert 'checkpoint.get("optimizer_saved") is not False' in source
    assert 'checkpoint_path.is_relative_to(Path("/mnt"))' in source
    assert 'CHECKPOINT_MANIFEST_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v2"' in source
    assert 'MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"' in source
    assert 'checkpoint.get("manifest_sha256")' in source
    assert 'checkpoint.get("model_tree_sha256")' in source
    assert 'checkpoint.get("training_final_model_local_state_sha256_by_rank")' in source
    assert 'model_path / ".metadata"' in source
    assert 'model_path.glob("*.distcp")' in source
    assert "path.is_symlink()" in source
    assert 'manifest.get("action_information_set_schedule_sha256")' in source
    assert '"seed": 20260813' in source
    assert 'training_ranks[rank].get("runtime_schedule_sha256")' in source


def test_v2_preflight_accepts_matching_checkpoint_and_cold_evidence() -> None:
    with _persistent_temp_directory() as root:
        training_path, cold_path, checkpoint, _ = _build_v2_artifacts(root)
        resolved = _run_contract(
            "adr165_resolve_mediator_trial_checkpoint "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(training_path))}"
        )
        assert resolved.returncode == 0, resolved.stderr
        assert resolved.stdout.strip() == str(checkpoint.resolve())

        validated = _run_contract(
            "adr165_validate_cold_report "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(cold_path))} "
            f"{shlex.quote(str(checkpoint))} evaluation 1"
        )
        assert validated.returncode == 0, validated.stderr
        assert validated.stdout.strip() == str(cold_path.resolve())


def test_v2_preflight_rejects_noncanonical_or_changed_checkpoint_digests() -> None:
    with _persistent_temp_directory() as root:
        training_path, _, checkpoint, training_report = _build_v2_artifacts(root)
        receipt = training_report["checkpoint"]
        assert isinstance(receipt, dict)
        receipt["manifest_sha256"] = str(receipt["manifest_sha256"]).upper()
        _write_json(training_path, training_report)
        noncanonical = _run_contract(
            "adr165_resolve_mediator_trial_checkpoint "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(training_path))}"
        )
        assert noncanonical.returncode != 0
        assert "lowercase SHA-256" in noncanonical.stderr

        _, _, _, fresh_training_report = _build_v2_artifacts(root / "fresh")
        fresh_training_path = root / "fresh" / "ltop_g3_mediator_trial_training_report.json"
        fresh_checkpoint = Path(str(fresh_training_report["checkpoint"]["path"]))
        with (fresh_checkpoint / "model" / "__0_0.distcp").open("ab") as stream:
            stream.write(b"tampered")
        changed_tree = _run_contract(
            "adr165_resolve_mediator_trial_checkpoint "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(fresh_training_path))}"
        )
        assert changed_tree.returncode != 0
        assert "model-tree SHA-256 differs from disk" in changed_tree.stderr


def test_v2_preflight_rejects_cold_state_mutation_and_tree_substitution() -> None:
    with _persistent_temp_directory() as root:
        _, cold_path, checkpoint, _ = _build_v2_artifacts(root)
        cold_report = json.loads(cold_path.read_text(encoding="ascii"))
        cold_report["rank_reports"][1]["post_evaluation_model_local_state_sha256"] = "a" * 64
        _write_json(cold_path, cold_report)
        mutated = _run_contract(
            "adr165_validate_cold_report "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(cold_path))} "
            f"{shlex.quote(str(checkpoint))} evaluation 1"
        )
        assert mutated.returncode != 0
        assert "mutated persistent model state" in mutated.stderr

        cold_report["rank_reports"][1]["post_evaluation_model_local_state_sha256"] = cold_report[
            "rank_reports"
        ][1]["cold_loaded_model_local_state_sha256"]
        cold_report["rank_reports"][0]["trained_checkpoint_model_tree_sha256"] = "b" * 64
        _write_json(cold_path, cold_report)
        substituted = _run_contract(
            "adr165_validate_cold_report "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(cold_path))} "
            f"{shlex.quote(str(checkpoint))} evaluation 1"
        )
        assert substituted.returncode != 0
        assert "consumed another checkpoint tree" in substituted.stderr

        manifest = json.loads(
            (checkpoint / "ltop_g3_training_checkpoint.json").read_text(encoding="ascii")
        )
        cold_report["rank_reports"][0]["trained_checkpoint_model_tree_sha256"] = manifest[
            "model_tree_sha256"
        ]
        cold_report["rank_reports"][0]["trained_model_local_state_sha256"] = "c" * 64
        _write_json(cold_path, cold_report)
        legacy_alias = _run_contract(
            "adr165_validate_cold_report "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(cold_path))} "
            f"{shlex.quote(str(checkpoint))} evaluation 1"
        )
        assert legacy_alias.returncode != 0
        assert "legacy model digest alias differs" in legacy_alias.stderr

        cold_report["rank_reports"][0]["trained_model_local_state_sha256"] = cold_report[
            "rank_reports"
        ][0]["cold_loaded_model_local_state_sha256"]
        cold_report["rank_reports"][0]["runtime_schedule_sha256"] = "d" * 64
        _write_json(cold_path, cold_report)
        runtime_schedule = _run_contract(
            "adr165_validate_cold_report "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(cold_path))} "
            f"{shlex.quote(str(checkpoint))} evaluation 1"
        )
        assert runtime_schedule.returncode != 0
        assert "runtime schedule differs from training" in runtime_schedule.stderr


def test_action_launcher_defaults_to_quick_and_allows_only_registered_full_scope() -> None:
    source = _source("adr165/run_ltop_g3_mediator_cold_action_2gpu.sh")

    assert "PICF_G3_EVALUATION_SCENES_PER_PARTITION:-1" in source
    assert "1) evaluation_scope=quick" in source
    assert "4) evaluation_scope=full" in source
    assert "must be 1 or 4" in source
    assert '--evaluation-scenes-per-partition "$evaluation_scenes_per_partition"' in source
    assert "ltop_g3_mediator_cold_action_${evaluation_scope}_report.json" in source


def test_action_launcher_cold_loads_mediator_checkpoint_into_registered_gate() -> None:
    source = _source("adr165/run_ltop_g3_mediator_cold_action_2gpu.sh")

    assert "ltop_g3_mediator_trial_training_report.json" in source
    assert "adr165_resolve_mediator_trial_checkpoint" in source
    assert '--trained-checkpoint "$trained_checkpoint"' in source
    assert "--mode gate" in source
    assert "--phase evaluation" in source
    assert "--steps 128" in source
    assert "--eval-every 32" in source
    assert "timeout --signal=TERM --kill-after=60s" in source
    assert "adr165_validate_cold_report" in source
    assert source.index("adr165_resolve_mediator_trial_checkpoint") < source.index(
        "--phase evaluation"
    )


def test_retention_launcher_uses_the_same_extracted_checkpoint() -> None:
    source = _source("adr165/run_ltop_g3_mediator_retention_2gpu.sh")

    assert "ltop_g3_mediator_trial_training_report.json" in source
    assert "adr165_resolve_mediator_trial_checkpoint" in source
    assert '--trained-checkpoint "$trained_checkpoint"' in source
    assert "--mode gate" in source
    assert "--phase retention" in source
    assert "--steps 128" in source
    assert "--eval-every 32" in source
    assert "ltop_g3_mediator_representation_retention_report.json" in source
    assert "adr165_validate_cold_report" in source


def test_both_cold_launchers_are_fail_closed_under_mnt() -> None:
    for path in (
        "adr165/run_ltop_g3_mediator_cold_action_2gpu.sh",
        "adr165/run_ltop_g3_mediator_retention_2gpu.sh",
    ):
        source = _source(path)
        assert 'repository_root" != /mnt/*' in source
        assert "one absent direct path under /mnt" in source
        assert "status --porcelain=v1 --untracked-files=all" in source
        assert "requires exactly two visible GPUs" in source
        assert "--nproc_per_node=2" in source
        assert "runtime_failure.json" in source


def test_chain_runs_action_before_retention_and_refuses_overwrite() -> None:
    source = _source("adr165/run_ltop_g3_mediator_cold_acceptance_2gpu.sh")

    action = "run_ltop_g3_mediator_cold_action_2gpu.sh"
    retention = "run_ltop_g3_mediator_retention_2gpu.sh"
    assert "ACCEPTANCE_RUN_ROOT" in source
    assert "one absent direct path under /mnt" in source
    assert "action-$evaluation_scope" in source
    assert source.rindex(action) < source.rindex(retention)


def test_adr165_cold_launchers_are_executable() -> None:
    for path in (
        "adr165/run_ltop_g3_mediator_cold_action_2gpu.sh",
        "adr165/run_ltop_g3_mediator_retention_2gpu.sh",
        "adr165/run_ltop_g3_mediator_cold_acceptance_2gpu.sh",
    ):
        assert (ROOT / path).stat().st_mode & S_IXUSR
