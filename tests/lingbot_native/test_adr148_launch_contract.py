from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "adr148/run_full_picf.sh"
INITIAL_LAUNCHER = ROOT / "adr148/launch_four_gpu_initial_2k.sh"
LONG_LAUNCHER = ROOT / "adr148/launch_four_gpu_30k.sh"
FROZEN_INPUTS = ROOT / "adr148/frozen_inputs.sha256"


def test_adr148_full_profile_is_explicit_and_provenance_locked() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        "--nproc-per-node=4",
        "--posterior-architecture layerwise_v2",
        "--entity-weight 0.08",
        "--predictive-weight 0.004",
        "--local-bptt-probability 0.0",
        "--overshoot-probability 0.0",
        "--source-mask-probability 0.10",
        "--source-prediction-mode omitted_static",
        "--current-grid-cache-root",
        "--current-grid-cache-build-report",
        "--current-grid-cache-build-report-sha256",
        "--stream-plan-sha256",
        "--representation-split-sha256",
        'sha256sum --check --strict "$FROZEN_INPUTS"',
        '"$STOP_AFTER_STEP" -le 2000',
        'git -C "$REPO" status --porcelain=v1 --untracked-files=all',
    )
    for value in required:
        assert value in source
    assert "--predictive-cache-root" not in source


def test_adr148_initial_launcher_requires_matched_no_picf_control_and_stops_at_2k() -> None:
    source = INITIAL_LAUNCHER.read_text(encoding="utf-8")
    required = (
        'report.get("status") != "PASS"',
        'report.get("picf_graph_installed") is not False',
        'report.get("world_size") != 4',
        'report.get("steps") != 200',
        'report.get("plan_sha256") != plan.get("plan_sha256")',
        'report.get("seed") != 20260721',
        "float(1e-4).hex()",
        "float(1.0).hex()",
        'grep -Fx "$MATCHED_LBOT_SHA  $MATCHED_LBOT_REPORT" "$FROZEN_INPUTS"',
        'fresh "$RUN_DIR" 2000 0',
    )
    for value in required:
        assert value in source


def test_adr148_long_launcher_is_resume_only_and_evidence_bound() -> None:
    source = LONG_LAUNCHER.read_text(encoding="utf-8")
    required = (
        "picf-next.adr148-long-authorization/v1",
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


def test_adr148_frozen_input_receipt_covers_every_mutable_external_control() -> None:
    records = {}
    for line in FROZEN_INPUTS.read_text(encoding="ascii").splitlines():
        digest, path = line.split("  ", maxsplit=1)
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")
        assert path.startswith("/mnt/")
        assert path not in records
        records[path] = digest
    required_suffixes = (
        "/calvin-training-files.json",
        "/calvin-lingbot-norm-stats.json",
        "/manifest.json",
        "/visual_acceptance.json",
        "/four-gpu-30k.stream-plan.json",
        "/four-gpu-30k.split.json",
        "/current-correction-dino-v1/manifest.json",
        "/current-correction-dino-v1.build_report.json",
        "/robotwin.yaml",
    )
    for suffix in required_suffixes:
        assert any(path.endswith(suffix) for path in records)
    # ADR-148 is an immutable historical receipt produced before the control
    # arm was renamed from P0 to LBOT. New reports use official_lbot_steps_*;
    # the frozen receipt must continue naming the artifact it actually hashed.
    assert any(path.endswith("/official_p0_steps_200.json") for path in records)
