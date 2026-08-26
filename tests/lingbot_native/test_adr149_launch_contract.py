from pathlib import Path

from tools.run_lingbot_vla2_task_independent_full import IMPLEMENTATION_FILES

ROOT = Path(__file__).resolve().parents[2]
FREEZER = ROOT / "adr149/freeze_inputs.sh"
LBOT_RUNNER = ROOT / "adr149/run_matched_lbot_4gpu.sh"
FULL_RUNNER = ROOT / "adr149/run_full_picf.sh"
INITIAL_LAUNCHER = ROOT / "adr149/launch_four_gpu_initial_2k.sh"
LONG_LAUNCHER = ROOT / "adr149/launch_four_gpu_30k.sh"
RUNTIME_RESTORE = ROOT / "adr147/restore_four_gpu_runtime.sh"


def test_adr149_launchers_default_to_the_real_non_symlink_runtime() -> None:
    for path in (FREEZER, LBOT_RUNNER, FULL_RUNNER, INITIAL_LAUNCHER, LONG_LAUNCHER):
        source = path.read_text(encoding="utf-8")
        assert "/opt/picf-runtime-restore-probe-94305690cafb/bin/python3.12" in source
        assert "/opt/picf-runtime-restore-probe-94305690cafb/bin/python}" not in source


def test_four_gpu_runtime_preflight_survives_python_optimization() -> None:
    source = RUNTIME_RESTORE.read_text(encoding="utf-8")
    assert 'if torch.__version__ != "2.8.0+cu128"' in source
    assert "if not torch.distributed.is_available()" in source
    assert "if not torch.distributed.is_nccl_available()" in source
    assert "assert torch.__version__" not in source


def test_adr149_launchers_keep_runtime_and_experiment_handoffs_disjoint() -> None:
    for path in (LBOT_RUNNER, INITIAL_LAUNCHER, LONG_LAUNCHER):
        source = path.read_text(encoding="utf-8")
        assert (
            "RUNTIME_HANDOFF=${PICF_RUNTIME_HANDOFF_ROOT:-"
            "/mnt/picf-next/adr147/four_gpu_handoff_20260808}"
        ) in source
        assert (
            'PICF_REPO=$REPO PICF_HANDOFF_ROOT=$RUNTIME_HANDOFF \\\n'
            '  "$REPO/adr147/restore_four_gpu_runtime.sh"'
        ) in source


def test_adr149_frozen_receipt_is_persistent_atomic_and_never_overwritten() -> None:
    source = FREEZER.read_text(encoding="utf-8")
    required = (
        "$HANDOFF/frozen_inputs.manifest.json",
        "$HANDOFF/frozen_inputs.sha256",
        "never overwrite a scientific receipt",
        'git -C "$REPO" status --porcelain=v1 --untracked-files=all',
        "validate_prepared_native_source",
        'validated.get("patch_state") != "applied"',
        'cache_report.get("expected_record_count") != 120004',
        "CURRENT_REPLAY_RECEIPT=${PICF_CURRENT_CACHE_FULL_REPLAY_RECEIPT:-${CURRENT_CACHE}.full_replay_verification.json}",
        'cache_replay.get("status") != "PASS"',
        'cache_replay.get("cache_manifest_sha256") != cache_manifest_sha256',
        'cache_replay.get("record_count") != cache_report.get("expected_record_count")',
        '"current_filter_cache_full_replay_sha256": digest(cache_replay_path)',
        '"current_filter_cache_content_stream_sha256": content_stream_sha256',
        'lbot.get("registered_evaluation_steps") != [0, 20, 100, 200]',
        'mv -T "$MANIFEST_TMP" "$MANIFEST"',
        'mv -T "$RECEIPT_TMP" "$RECEIPT"',
        'sha256sum --check --strict "$RECEIPT"',
    )
    for value in required:
        assert value in source


def test_adr149_matched_lbot_uses_identical_physical_stream_and_control_capacity() -> None:
    source = LBOT_RUNNER.read_text(encoding="utf-8")
    required = (
        "--nproc-per-node=4",
        "four-gpu-30k.physical.stream-plan.json",
        "four-gpu-30k.physical.split.json",
        "four-gpu-30k.physical.evaluation-plan.json",
        "--physical-event-stream",
        "--maximum-control-tokens 64",
        "EVALUATION_STEPS=0,20,100,200",
        "EVALUATION_STEPS=0,20,100,200,500,1000,1500,2000",
        '--evaluation-steps "$EVALUATION_STEPS"',
        '--steps "$STEPS"',
        "supports only the registered 200- or 2000-step curves",
        'git -C "$REPO" status --porcelain=v1 --untracked-files=all',
    )
    for value in required:
        assert value in source


def test_adr149_full_profile_is_two_pass_physical_and_provenance_locked() -> None:
    source = FULL_RUNNER.read_text(encoding="utf-8")
    required = (
        "--nproc-per-node=4",
        "--posterior-architecture two_pass_v3",
        "--maximum-control-tokens 64",
        "--entity-weight 0.08",
        "--predictive-weight 0.004",
        "--local-bptt-probability 0.0",
        "--overshoot-probability 0.0",
        "--source-mask-probability 0.10",
        "--source-prediction-mode omitted_static",
        "four-gpu-30k.physical.stream-plan.json",
        "four-gpu-30k.physical.split.json",
        "four-gpu-30k.physical.evaluation-plan.json",
        '--evaluation-plan "$EVALUATION_PLAN"',
        '--evaluation-plan-sha256 "$EVALUATION_PLAN_SHA"',
        "current-filter-dino-physical-v1",
        'sha256sum --check --strict "$FROZEN_INPUTS"',
        "$HANDOFF/frozen_inputs.manifest.json",
        'manifest.get("implementation_commit") != sys.argv[2]',
        'manifest.get("lingbot_source_commit") != sys.argv[3]',
        '"$STOP_AFTER_STEP" -le 2000',
    )
    for value in required:
        assert value in source
    assert "--predictive-cache-root" not in source
    assert "--posterior-architecture layerwise_v2" not in source
    for path in (
        "adr147/restore_four_gpu_runtime.sh",
        "adr149/freeze_inputs.sh",
        "adr149/launch_four_gpu_initial_2k.sh",
        "adr149/launch_four_gpu_30k.sh",
        "adr149/run_full_picf.sh",
        "adr149/run_matched_lbot_4gpu.sh",
        "src/picf_next/data/calvin_target_request.py",
        "src/picf_next/data/lingbot_calvin.py",
        "tools/build_lingbot_calvin_current_grid_cache.py",
        "tools/build_lingbot_representation_split.py",
        "tools/run_lingbot_vla2_official_lbot.py",
    ):
        assert path in IMPLEMENTATION_FILES


def test_adr149_initial_gate_rejects_nonphysical_or_mismatched_control() -> None:
    source = INITIAL_LAUNCHER.read_text(encoding="utf-8")
    required = (
        'report.get("status") != "PASS"',
        'report.get("picf_graph_installed") is not False',
        'report.get("world_size") != 4',
        'report.get("steps") != 200',
        'report.get("physical_event_stream") is not True',
        'report.get("maximum_control_tokens") != 64',
        'report.get("plan_sha256") != plan.get("plan_sha256")',
        'report.get("representation_split_sha256") != split.get("artifact_sha256")',
        'report.get("evaluation_plan_sha256") != evaluation.get("artifact_sha256")',
        'fresh "$RUN_DIR" 2000 0',
    )
    for value in required:
        assert value in source


def test_adr149_long_launcher_is_evidence_bound_resume_only() -> None:
    source = LONG_LAUNCHER.read_text(encoding="utf-8")
    required = (
        "picf-next.adr149-long-authorization/v1",
        'value["input_global_step"] != 2000',
        'value["maximum_global_step"] != 30000',
        "cold_resume_equivalence",
        "heldout_action",
        "calvin_rollout",
        "full_curve_comparison",
        "visual_review",
        "causal_interventions",
        "gradient_adoption",
        "filter_interaction",
        'resume "$RUN_DIR" 30000 2000',
    )
    for value in required:
        assert value in source
    assert 'fresh "$RUN_DIR" 30000 0' not in source
