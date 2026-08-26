import hashlib
import subprocess
from pathlib import Path

from tools.bootstrap_lingbot_vla2 import (
    LINGBOT_CHECKPOINT_REVISION,
    QWEN_PROCESSOR_REVISION,
)
from tools.bootstrap_lingbot_vla2_native import (
    LINGBOT_NATIVE_SOURCE_COMMIT,
    PATCH_SHA256,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs/76_ADR74_OWNER_APPROVAL_AND_IMPLEMENTATION_LEDGER.md"
RUNBOOK = ROOT / "docs/77_ADR74_LOCAL_DEPLOYMENT_AND_2XA100_G0_RUNBOOK.md"
EMPIRICAL_CONTRACT = ROOT / "docs/79_LINGBOT_NATIVE_EMPIRICAL_EVIDENCE_CONTRACT.md"
EMPIRICAL_PRODUCERS = ROOT / "docs/80_LINGBOT_NATIVE_MODEL_SPECIFIC_EVALUATION_PRODUCERS_ADR.md"
CONTINUOUS_ESTIMATOR = ROOT / "docs/84_CONTINUOUS_POSTERIOR_ESTIMATOR_AND_LOCAL_CLOSURE_AUDIT.md"
FIXED_BATCH_CLOSURE = ROOT / "docs/87_PREDICTIVE_FIXED_BATCH_PRODUCTION_CLOSURE.md"
RELATIVE_IMPORT_CLOSURE = ROOT / "docs/89_RELATIVE_IMPORT_AND_LEGACY_REACHABILITY_CLOSURE.md"
TYPED_CONFIG_CLOSURE = ROOT / "docs/93_LINGBOT_TYPED_CONFIG_SEMANTICS_AND_G0_CLOSURE.md"
ROUTING_PROVENANCE_CLOSURE = ROOT / "docs/94_G0_ROUTING_PROVENANCE_ABI_AND_EARLY_VALIDATION.md"
K8_REFERENCE_RUNBOOK = ROOT / "docs/110_K8_REFERENCE_BANK_TRIAL_RUNBOOK_20260729.md"
TERMINAL_COMPRESSION_AUDIT = (
    ROOT / "docs/116_TERMINAL_TASK_INTERFACE_AND_INFORMATION_COMPRESSION_AUDIT_20260730.md"
)
README_AUDIT_LEDGER = ROOT / "docs/02_README_AUDIT_LEDGER.md"


def test_readme_audit_ledger_covers_every_tracked_readme_and_digest() -> None:
    result = subprocess.run(
        ("git", "ls-files"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = sorted(
        path for path in result.stdout.splitlines() if Path(path).name.lower().startswith("readme")
    )
    text = README_AUDIT_LEDGER.read_text()
    assert f"All {len(tracked)} tracked matches are accounted for below." in text
    for relative in tracked:
        payload = (ROOT / relative).read_bytes()
        assert f"`{relative}`" in text
        assert hashlib.sha256(payload).hexdigest() in text


def test_native_ledger_and_runbook_bind_every_immutable_revision() -> None:
    text = LEDGER.read_text() + RUNBOOK.read_text()
    for value in (
        LINGBOT_NATIVE_SOURCE_COMMIT,
        LINGBOT_CHECKPOINT_REVISION,
        QWEN_PROCESSOR_REVISION,
        PATCH_SHA256,
    ):
        assert value in text


def test_readme_indexes_the_latest_implementation_closure() -> None:
    text = (ROOT / "README.md").read_text()
    assert RELATIVE_IMPORT_CLOSURE.name in text
    assert TYPED_CONFIG_CLOSURE.name in text
    assert ROUTING_PROVENANCE_CLOSURE.name in text
    assert "latest independent implementation-closure correction" in text
    assert "typed released-config correction" in text
    assert "routing-provenance ABI correction" in text


def test_routing_provenance_closure_records_production_types_and_early_validation() -> None:
    text = " ".join(ROUTING_PROVENANCE_CLOSURE.read_text().split())
    for fragment in (
        "lane_ids",
        "non-negative integers",
        "sample_keys",
        "non-empty strings",
        "before Distributed Checkpoint writes",
        "31fa2dafcbbe0674bcccf6607c2d2539496170e3",
        "ea518b2fdd6a67da5e93b7e7ec8bb965ca5a5d58",
        "G0-B completed finite forward",
        "No `global_step_1` checkpoint",
        "cannot establish useful object rows",
    ):
        assert fragment in text


def test_readme_local_native_verification_pins_the_declared_torch_version() -> None:
    text = (ROOT / "README.md").read_text()
    assert (
        'PICF_LINGBOT_NATIVE_SOURCE="${PICF_LINGBOT_NATIVE_SOURCE:-'
        "$(git rev-parse --show-toplevel)/references/source_checkouts/"
        'lingbot-vla-v2-adr74}"'
    ) in text
    assert "python tools/bootstrap_lingbot_vla2_native.py" in text
    assert '--checkout "$PICF_LINGBOT_NATIVE_SOURCE"' in text
    assert '--source-checkout "$PICF_LINGBOT_NATIVE_SOURCE"' in text
    assert 'PICF_LOCAL_ATTEMPT="${PICF_LOCAL_ATTEMPT:-$(date -u +%Y%m%dT%H%M%SZ)}"' in text
    assert (
        'PICF_LOCAL_PREFLIGHT_OUTPUT="artifacts/adr74-local-preflight-'
        '$(git rev-parse --short=12 HEAD)-$PICF_LOCAL_ATTEMPT.json"'
    ) in text
    assert 'test ! -e "$PICF_LOCAL_PREFLIGHT_OUTPUT"' in text
    assert '--output "$PICF_LOCAL_PREFLIGHT_OUTPUT"' in text
    assert "--output artifacts/adr74_local_deployment_preflight.json" not in text
    assert text.count("--with torch==2.8.0+cpu") == 2
    assert text.count("--with accelerate==1.13.0") == 2
    assert text.count("--find-links https://download.pytorch.org/whl/cpu/torch/") == 2
    assert "python -m pytest -q" in text
    assert ".venv/bin/pytest -q" not in text


def test_native_runbook_freezes_two_stage_g0_and_forbids_long_training() -> None:
    text = RUNBOOK.read_text()
    required = (
        "tools/preflight_lingbot_native.py",
        "tools/smoke_lingbot_vla2_native_full_weight.py",
        "tools/run_lingbot_vla2_native_g0.py",
        "--phase fresh",
        "--phase resume",
        "--nproc_per_node=2",
        "--restore-runtime-pins",
        "--repair-depth-runtime",
        "--install-audit-tools",
        "requirements-depth.txt",
        "3fab839f0be9931dac7c8488eb0e1600c236e183",
        "conda-pack==0.9.2",
        "runtime archive restore probe PASS",
        "picf-lingbot-vla2-1f44c1f.tar.zst",
        "ea5ea4154cf5b40cb3eb9e8025fb2714e714a06c6cff330dc8678133df4d739a",
        "export PICF_ZSTD=/opt/picf-miniconda3/bin/zstd",
        'tar -I "$PICF_ZSTD"',
        "Never replace this archive in place",
        "FUSE",
        "--maximum-peak-reserved-gib 39",
        "--predictive-weight 0.004",
        "--structural-weight 0.004",
        "--gradient-audit-steps 2,3,20,60,120,200,500,1000,2000,5000,10000,20000,30000",
        "--local-bptt-probability 0.10",
        "--overshoot-probability 0.05",
        "--source-mask-probability 0.10",
        "--source-prediction-mode omitted_static",
        "PICF_PHYSICAL_PARTITION_COUNT=8",
        "tools/run_calvin_physical_supervision_parallel.py",
        "start_new_session=True",
        "--resume-completed-partition",
        "--defer-finalize",
        "--finalize-only",
        '--dataset-split "$PICF_DATASET_DIR"',
        '--dataset-manifest "$PICF_DATASET_MANIFEST"',
        '--norm-stats "$PICF_LINGBOT_NORM_STATS"',
        "cloud_model_assets_ready",
        "cloud_data_ready",
        "--visual-audit-every 1",
        "--authorization-manifest",
        "tools/build_lingbot_native_gate_decision.py",
        "tools/build_lingbot_native_evaluation_plan.py",
        "tools/build_lingbot_native_empirical_observations.py",
        "tools/build_lingbot_native_empirical_report.py",
        "tools/build_lingbot_native_training_authorization.py",
        "pilot_step_1_to_120.authorization.json",
        "--invocation-steps 119",
        "G7_PROTOCOL",
        "resume_boundary_verified",
        "resume_runtime_rng_verified",
        "No 200-step selection run, 30k long run",
    )
    for fragment in required:
        assert fragment in text
    assert "sync_files=False" in text


def test_physical_audit_hashes_every_external_contract_before_use() -> None:
    text = RUNBOOK.read_text()
    audit = text.index("tools/audit_calvin_physical_supervision.py")

    assert "tools/probe_lingbot_calvin_projection.py" in text[:audit]
    assert "lingbot-calvin-qwen-projection-${PICF_DATASET_MANIFEST_SHA256:0:12}.json" in text
    assert 'test ! -e "$PICF_CALVIN_QWEN_PROJECTION"' in text[:audit]
    for assignment in (
        "PICF_CALVIN_PHYSICAL_SIDECAR_SHA256=$(sha256sum",
        "PICF_CALVIN_QWEN_PROJECTION_SHA256=$(sha256sum",
    ):
        assert text.index(assignment) < audit
    for argument in (
        '--sidecar-manifest-sha256 "$PICF_CALVIN_PHYSICAL_SIDECAR_SHA256"',
        '--training-projection-contract "$PICF_CALVIN_QWEN_PROJECTION"',
        '--training-projection-contract-sha256 "$PICF_CALVIN_QWEN_PROJECTION_SHA256"',
    ):
        assert argument in text[audit:]


def test_runbook_distinguishes_processor_masks_from_loss_only_owner_masks() -> None:
    text = RUNBOOK.read_text()

    assert "Qwen processor\nimage-validity masks/grid metadata" in text
    assert "These structural masks are\nnot object segmentations" in text
    assert "owner/instance masks and future\ntargets never enter the deploy forward" in text


def test_native_runbook_freezes_fresh_four_arm_capacity_probe() -> None:
    text = RUNBOOK.read_text() + FIXED_BATCH_CLOSURE.read_text()
    for fragment in (
        "PICF_FIXED_BATCH_EXPERIMENT_ID",
        "must be new",
        "run_fixed_batch_arm full_host",
        "run_fixed_batch_arm native_graph_only",
        "run_fixed_batch_arm readout_only",
        "run_fixed_batch_arm shuffled_target",
        "UNDECIDED_REQUIRES_OWNER_REVIEW",
        "optimizer_update_count = curve_point_count - 1",
    ):
        assert fragment in text


def test_native_runbook_uses_fresh_content_identified_full_and_gate_namespaces() -> None:
    text = RUNBOOK.read_text()
    for fragment in (
        'PICF_FULL_RUN_DIR="$PICF_MNT/runs/adr74-full-${PICF_REPO_TREE:0:12}-$PICF_RUN_ATTEMPT"',
        'test ! -e "$PICF_FULL_RUN_DIR"',
        '--run-dir "$PICF_FULL_RUN_DIR"',
        'local runtime_dir="$PICF_FIXED_BATCH_DIR/runtime-${arm}"',
        'test ! -e "$runtime_dir"',
        '--run-dir "$runtime_dir"',
        'PICF_GATE_DIR="$PICF_MNT/runs/adr74-gates-${PICF_REPO_TREE:0:12}-$PICF_RUN_ATTEMPT"',
        'test ! -e "$PICF_GATE_DIR"',
        '--input-full-report "$PICF_FULL_RUN_DIR/native_full_step_1.json"',
    ):
        assert fragment in text
    for stale in (
        '--run-dir "$PICF_MNT/runs/adr74-full"',
        '--run-dir "$PICF_MNT/runs/adr87-fixed-batch"',
        "export PICF_GATE_DIR=$PICF_MNT/runs/adr74-gates",
        '--input-full-report "$PICF_MNT/runs/adr74-full/native_full_step_1.json"',
    ):
        assert stale not in text


def test_empirical_contract_keeps_model_producers_and_statistics_separate() -> None:
    text = EMPIRICAL_CONTRACT.read_text() + EMPIRICAL_PRODUCERS.read_text()
    required = (
        "candidate[j,s,t,e] - reference[j,s,t,e]",
        "seed/task/episode",
        "Frames are aggregated",
        "hash-bound evaluator JSON artifact",
        "model-specific evaluator",
        "G2 current-set/no-object observation producer",
        "HOTA.eval_sequence",
        "successful-prefix",
        "scientific validity | 0/10",
    )
    for fragment in required:
        assert fragment in text


def test_continuous_estimator_forbids_online_prefix_rewrite_and_claim_inflation() -> None:
    text = " ".join(CONTINUOUS_ESTIMATOR.read_text().split())
    for fragment in (
        "The trainable LingBot host is the only semantic authority",
        "remove online full-prefix rewrite",
        "commit or replace a lane",
        "There is no refresh exception",
        "released-weight deployment evidence | 0/10",
        "cannot honestly become 10/10 through more local review",
    ):
        assert fragment in text
    runbook = RUNBOOK.read_text()
    for obsolete in (
        "--refresh-probability",
        "--refresh-after-optimizer-lag",
        "--maximum-recompute-gap",
        'rm -f "$PICF_RUNTIME_ARCHIVE"',
        "tar --zstd",
        'zstd -t "$PICF_RUNTIME_ARCHIVE"',
    ):
        assert obsolete not in runbook


def test_k8_runbook_freezes_reference_bank_and_forbids_claim_inflation() -> None:
    text = K8_REFERENCE_RUNBOOK.read_text()
    for fragment in (
        "not a strict single-variable causal estimate",
        "392fd6b9ba6b15e015d39a14e5036bbd7eeaad407b44d1a9ab3bfda2835a31b7",
        "b325631b03801d1d915edde400602f9d9734884de74f181dc1638fa96b1e8a00",
        "--lane-interleave-factor 8",
        "--evaluation-reference-split",
        "evaluation_reference_preserved=true",
        "sample order and sample identities",
        "source-episode and task exposure",
        "episode replacement and reset count",
        "step 0/1, then fixed 50/100/200",
        "No 30k run follows directly",
        "scientific result: `not run`",
    ):
        assert fragment in text


def test_terminal_compression_audit_rejects_only_the_failed_interface() -> None:
    text = TERMINAL_COMPRESSION_AUDIT.read_text()
    normalized = " ".join(text.split())
    for fragment in (
        "CURRENT TASK-OBJECT INTERFACE REJECTED",
        "single-final-language-state",
        "complete language sequence reduced to the last valid hidden state",
        "V-JEPA, AnyTouch or Sonata",
        "token count unchanged",
        "Preserve every valid short-prompt token",
        "persistent row",
        "remains task agnostic",
        "No replacement is implemented by this ADR",
        "0/29",
        "92bc11f6fd30b784e30e3df10ab89ea25d1ac6cbc919237c83f0cb49e11fbf3c",
        "fa84f04b6084e7af18893d51f3ca87b589022e2521d7e288df2061e2c910377e",
    ):
        assert fragment in text
    assert "NO ACTION ADOPTION OR LONG TRAINING IS AUTHORIZED" in normalized
