import subprocess
from pathlib import Path

from tools.run_lingbot_vla2_task_independent_full import (
    IMPLEMENTATION_FILES,
    _implementation_paths,
)

ROOT = Path(__file__).resolve().parents[2]
ADR150 = ROOT / "adr150"
SCRIPTS = (
    ADR150 / "freeze_inputs.sh",
    ADR150 / "launch_four_gpu_30k.sh",
    ADR150 / "launch_four_gpu_initial_2k.sh",
    ADR150 / "merge_dense_cache_partitions.sh",
    ADR150 / "run_dense_cache_partition.sh",
    ADR150 / "run_full_modal_acceptance_4gpu.sh",
    ADR150 / "run_full_modal_acceptance_suite_4gpu.sh",
    ADR150 / "run_full_picf.sh",
    ADR150 / "run_matched_lbot_4gpu.sh",
)


def _text(name: str) -> str:
    return (ADR150 / name).read_text(encoding="utf-8")


def test_adr150_shell_is_syntactically_valid_and_implementation_bound() -> None:
    for path in SCRIPTS:
        subprocess.run(("bash", "-n", str(path)), check=True)
        assert str(path.relative_to(ROOT)) in IMPLEMENTATION_FILES


def test_adr150_vjepa_partitions_have_one_explicit_bounded_batch_control() -> None:
    script = _text("run_dense_cache_partition.sh")
    assert "PICF_VJEPA_ENCODER_BATCH_SIZE:-8" in script
    assert "V-JEPA encoder batch size must be a canonical positive integer" in script
    assert '--encoder-batch-size "$VJEPA_ENCODER_BATCH_SIZE"' in script


def test_adr150_vjepa_merge_uses_authenticated_partition_index() -> None:
    script = _text("merge_dense_cache_partitions.sh")
    assert "--reference-partitions" in script
    assert "--link-shards" not in script


def test_adr150_pipeline_separates_coverage_identities_and_reuses_valid_index() -> None:
    script = _text("launch_complete_pipeline_when_caches_ready.sh")
    assert "COVERAGE_FILE_SHA=" in script
    assert "COVERAGE_ARTIFACT_SHA=" in script
    assert 'sha256 "$COVERAGE")" == "$COVERAGE_FILE_SHA"' in script
    assert script.count('--coverage-plan-sha256 "$COVERAGE_ARTIFACT_SHA"') == 2
    assert "REUSING_VJEPA_INDEX" in script
    assert "must be jointly absent or jointly direct" in script


def test_adr150_acceptance_suite_is_full_modal_and_cold_restored() -> None:
    runner = _text("run_full_modal_acceptance_4gpu.sh")
    suite = _text("run_full_modal_acceptance_suite_4gpu.sh")
    for mode in (
        "action-adoption-presence",
        "action-adoption-interventions",
        "dcp-uninterrupted",
        "dcp-restored",
    ):
        assert mode in runner
    assert runner.count("--dense-evidence-cache-root") == 3
    assert "--dense-token-bridge lingbot_task_token_resampler_v1" in runner
    assert "--posterior-architecture two_pass_v3" in runner
    assert "--maximum-control-tokens 64" in runner
    assert '--acceptance-mode "$ACCEPTANCE_MODE"' in runner
    assert "dcp_uninterrupted.json" in suite
    assert "dcp_restored.json" in suite
    assert "compose_adr150_action_adoption_core.py" in suite
    assert "compose_adr150_full_modal_action_adoption.py" in suite


def test_adr150_matched_lbot_requires_validated_full_modal_adoption() -> None:
    script = _text("run_matched_lbot_4gpu.sh")
    assert "FULL_MODAL_ACTION_ADOPTION" in script
    assert (
        "full-cache-r18-dcp-boundary-f8b5304/full_modal_action_adoption.json"
        in script
    )
    assert '--full-modal-action-adoption "$FULL_MODAL_ACTION_ADOPTION"' in script


def test_adr150_implementation_identity_covers_recursive_local_imports() -> None:
    relative = {str(path.relative_to(ROOT)) for path in _implementation_paths(ROOT)}
    for required in (
        "src/picf_next/__init__.py",
        "src/picf_next/geometry.py",
        "src/picf_next/objective.py",
        "src/picf_next/lingbot_native/gate_evidence.py",
        "src/picf_next/lingbot_native/temporal.py",
        "tools/verify_lingbot_vla2_patch.py",
    ):
        assert required in relative


def test_adr150_full_run_uses_only_full_official_multimodal_contract() -> None:
    script = _text("run_full_picf.sh")
    assert "calvin-official-30k-v1" in script
    assert "calvin-official-30k-prefix-2k-v1" not in script
    assert "calvin-task-ABC-D-content-a60b7934/calvin-training-files.json" in script
    assert "calvin-normalization-identity-a60b7934" in script
    assert "calvin-sidecar-identity-a60b7934" in script
    assert "calvin-physical-visual-review-identity-a60b7934" in script
    assert "four-gpu-30k.physical.dense-evidence-coverage.json" in script
    assert script.count("--dense-evidence-cache-root") == 3
    assert script.count("--dense-evidence-cache-manifest-sha256") == 3
    assert "--dense-token-bridge lingbot_task_token_resampler_v1" in script
    for modality_root in (
        "anytouch-observed-pose",
        "sonata-native",
        "vjepa-final",
    ):
        assert modality_root in script
    assert "full-dense-source-input-audit.json" in script
    assert "full-dense-semantic-audit.json" in script
    for required in (
        "--posterior-architecture two_pass_v3",
        "--maximum-control-tokens 64",
        "--capacity 16",
        "--maximum-optimizer-lag 8",
        "--entity-weight 0.08",
        "--predictive-weight 0.004",
        "--local-bptt-probability 0.0",
        "--overshoot-probability 0.0",
        "--source-mask-probability 0.10",
        "--source-prediction-mode omitted_static",
        "--fsdp2-placement selective-embedding-offload",
    ):
        assert required in script


def test_adr150_freeze_replays_every_current_and_dense_shard() -> None:
    script = _text("freeze_inputs.sh")
    assert "FrozenDenseEvidenceCacheBank.load" in script
    assert "for record in first_by_shard.values()" in script
    assert "cache.evidence_for(" in script
    assert "for shard in current_cache.shards" in script
    assert "current_cache.record_for(" in script
    assert '"dense_record_count": dense_bank.record_count' in script
    assert '"training_scope": {"first_step": 0, "gate_step": 2000' in script
    assert "never overwrite a scientific receipt" in script
    assert '"schema": "picf-next.adr150-frozen-inputs/v2"' in script
    assert '"canonical_paths": canonical_paths' in script
    assert '"implementation_sha256": _implementation_digest' in script
    assert "FullModalAssetManifest.load" in script
    assert "cache.contract.encoder_contract" in script
    assert '"runtime_archive"' in script
    assert '"dense_source_audit"' in script
    assert '"dense_semantic_audit"' in script
    assert "validate_calvin_dense_evidence_source_audit" in script
    assert "validate_calvin_dense_evidence_audit" in script
    assert '"dense_source_input_audit_artifact_sha256"' in script
    assert "validate_adr150_matched_lbot_report" in script
    assert '"matched_lbot_validation"' in script
    assert "RECOVER_ORPHAN_MANIFEST=1" in script
    assert 'cmp -s "$MANIFEST_TMP" "$MANIFEST"' in script
    assert "orphan manifest differs from exact deterministic reconstruction" in script


def test_adr150_gate_is_one_resumable_30k_identity() -> None:
    initial = _text("launch_four_gpu_initial_2k.sh")
    long = _text("launch_four_gpu_30k.sh")
    run = _text("run_full_picf.sh")
    assert 'exec "$REPO/adr150/run_full_picf.sh" fresh "$RUN_DIR" 2000 0' in initial
    assert (
        'exec "$REPO/adr150/run_full_picf.sh" resume "$RUN_DIR" 30000 "$LOAD_GLOBAL_STEP"' in long
    )
    assert '"evidence_sha256"' in long
    assert "frozen_inputs_sha256" in long
    assert "full_modal_adoption" in long
    assert "evidence_schemas" in long
    assert "observed_paths" in long
    assert 'promotion_root = run_dir / "promotion"' in long
    assert "resume_from_{load_global_step:08d}.json" in long
    assert '--stop-after-step "$STOP_AFTER_STEP"' in run
    assert "fresh ADR-150 requires step zero" in run
    assert "ADR-150 resume requires a positive 2000-step boundary" in run
    assert "effective path bypassed frozen input" in run
    assert "transitive implementation closure changed after freezing" in run
    assert "typed matched-LBOT validation did not pass" in run
    runner = (ROOT / "tools/run_lingbot_vla2_task_independent_full.py").read_text(encoding="utf-8")
    assert "expired_step != CHECKPOINT_EVERY" in runner


def test_adr150_long_promotion_is_hard_blocked_until_typed_validators_exist(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ADR150 / "launch_four_gpu_30k.sh"),
            str(tmp_path / "run"),
            str(tmp_path / "authorization.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "promotion is NO-GO until typed evidence validators" in result.stderr


def test_adr150_lbot_uses_the_same_official_stream_identity() -> None:
    script = _text("run_matched_lbot_4gpu.sh")
    assert "calvin-official-30k-v1" in script
    assert "calvin-task-ABC-D-content-a60b7934/calvin-training-files.json" in script
    assert "calvin-normalization-identity-a60b7934" in script
    assert "--physical-event-stream" in script
    assert "--maximum-control-tokens 64" in script
    assert '--evaluation-steps "$EVALUATION_STEPS"' in script
