from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tools.preflight_lingbot_unified import (
    CONFIG_RELATIVE_PATH,
    ORDERED_GATES,
    _system_memory_gib,
    _write_text_durable,
    static_preflight,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_RELATIVE_PATH
UNIFIED_CHECKOUT = ROOT / "references/source_checkouts/lingbot-vla-v2-unified"
requires_unified_checkout = pytest.mark.skipif(
    not (UNIFIED_CHECKOUT / "requirements.txt").is_file(),
    reason="optional prepared unified LingBot checkout is absent",
)


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def test_unified_cloud_profile_is_static_valid_and_fail_closed() -> None:
    config = _config()
    validate_config(config)
    assert config["schema"] == "picf-next.lingbot-unified-cloud.v2"
    assert config["g1_profile"]["full_shard"] is True
    assert config["g1_profile"]["cpu_offload"] is True
    assert config["g1_profile"]["rank_local_picf_state_in_official_dcp"] is True
    assert config["g1_profile"]["checkpoint_boundary_verification"] == (
        "exact_rank_local_model_optimizer_picf_rng_sha256"
    )
    assert config["g1_profile"]["optimizer_updates_per_phase"] == 1
    assert config["g1_profile"]["sparse_bptt_enabled"] is False
    assert config["runtime"]["lerobot_install_mode"] == "no-deps"
    assert config["runtime"]["dataset_manifest_validation"] == "full_sha256_before_accelerator"
    assert config["paths"]["source_checkout_env"] == "PICF_LINGBOT_SOURCE"
    assert config["experiment_contract"]["ordered_gates"] == ORDERED_GATES
    assert ORDERED_GATES[-2:] == [
        "G7_scale_throughput_and_restart",
        "G8_second_host_transfer",
    ]

    bad = deepcopy(config)
    bad["runtime"]["lerobot_install_mode"] = "resolver-default"
    with pytest.raises(ValueError, match="install mode"):
        validate_config(bad)

    bad = deepcopy(config)
    bad["host"]["patches"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="patch"):
        validate_config(bad)

    bad = deepcopy(config)
    bad["experiment_contract"]["formal_30000_step_authorized"] = True
    with pytest.raises(ValueError, match="30k"):
        validate_config(bad)

    bad = deepcopy(config)
    bad["unified_graph"]["hard_identity_threshold"] = True
    with pytest.raises(ValueError, match="graph"):
        validate_config(bad)

    bad = deepcopy(config)
    bad["unified_graph"]["geometry_schema"]["frame"] = "world"
    with pytest.raises(ValueError, match="graph"):
        validate_config(bad)

    bad = deepcopy(config)
    bad["objective_contract"]["cross_modal_prediction"] = "enabled"
    with pytest.raises(ValueError, match="objective"):
        validate_config(bad)

    bad = deepcopy(config)
    bad["objective_contract"]["target_tensors_deploy_visible"] = True
    with pytest.raises(ValueError, match="objective"):
        validate_config(bad)

    bad = deepcopy(config)
    bad["g1_profile"]["full_shard"] = False
    with pytest.raises(ValueError, match="G1"):
        validate_config(bad)


@requires_unified_checkout
def test_static_preflight_replays_both_patches_and_never_authorizes_training() -> None:
    checkout = UNIFIED_CHECKOUT
    report = static_preflight(_config(), root=ROOT, source_checkout=checkout)
    assert report["patch_replay"]["apply_checked"] is True
    assert report["patch_replay"]["commit"] == _config()["host"]["source_commit"]
    assert len(report["patch_replay"]["patched_source_sha256"]) == 5
    assert set(report["patch_hashes"]) == {item["path"] for item in _config()["host"]["patches"]}
    assert report["formal_30000_step_authorized"] is False
    assert report["initial_authorization"] == [ORDERED_GATES[0]]
    assert report["schema"] == "picf-next.lingbot-unified-preflight-report.v2"
    assert report["source_checkout"] == str(checkout.resolve())


@requires_unified_checkout
def test_preflight_cli_dry_run_is_cpu_only_and_writes_replayable_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/preflight_lingbot_unified.py",
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(result.stdout)
    written = json.loads(output.read_text())
    assert stdout == written
    assert written["mode"] == "static"
    assert written["formal_30000_step_authorized"] is False


def test_runtime_preflight_hashes_every_manifest_file_before_training() -> None:
    source = (ROOT / "tools/preflight_lingbot_unified.py").read_text()
    assert "validate_dataset_files(" in source
    assert "verify_hashes=True" in source


def test_system_memory_uses_the_smallest_finite_cgroup_limit(tmp_path: Path) -> None:
    unlimited = tmp_path / "unlimited"
    finite = tmp_path / "finite"
    malformed = tmp_path / "malformed"
    unlimited.write_text("max\n")
    finite.write_text(str(3 * 2**30))
    malformed.write_text("not-a-number")
    assert _system_memory_gib(
        cgroup_limit_paths=(unlimited, malformed, finite),
    ) == pytest.approx(3.0)


def test_preflight_report_publication_is_atomic(tmp_path: Path) -> None:
    report = tmp_path / "nested" / "preflight.json"
    _write_text_durable(report, '{"status":"PASS"}\n')
    assert report.read_text() == '{"status":"PASS"}\n'
    assert not tuple(report.parent.glob("*.tmp"))
