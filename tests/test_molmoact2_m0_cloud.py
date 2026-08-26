from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tools.run_molmoact2_m0_cloud import (
    _ROOT,
    _followup_gates,
    _is_under_mnt,
    _runner_command,
    _split_reports,
    _validate_required_reports,
    validate_m0_report,
)
from tools.smoke_molmoact2_lerobot_full_weight import _production_inference_context

CONFIG_PATH = _ROOT / "configs/cloud/2xa100_40g_gates.json"
ACTION_CONFIG_PATH = _ROOT / "configs/cloud/2xa100_40g_m4_action_adoption.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _passing_report(config: dict | None = None) -> dict:
    if config is None:
        config = _config()
    return {
        "schema": "picf-next.molmoact2-lerobot-m0.v3",
        "status": "PASS",
        "gate": "M0_full_weight_parity",
        "training_recipe_sha256": config["training_recipe"]["sha256"],
        "semantics": {
            "observation_path": "official_molmoact2_lerobot_processor_and_policy",
            "evidence_path": "native_molmo_prepool_patches_plus_synthetic_object_pressure",
            "dense_evidence_is_native_prepool_representation": True,
            "object_evidence_is_synthetic": True,
            "targets_or_masks_in_runtime_input": False,
            "native_molmo_729_same_forward_claimed": True,
            "official_baselines_precede_adapter_registration": True,
        },
        "prepared_input_tensors": {
            "inputs_embeds": {
                "shape": [1, 17, 2560],
                "dtype": "torch.bfloat16",
                "sha256": "a" * 64,
            },
            "attention_mask": {
                "shape": [1, 17],
                "dtype": "torch.bool",
                "sha256": "b" * 64,
            },
        },
        "prepared_action_condition_input_ids": {
            "shape": [1, 17],
            "dtype": "torch.int64",
            "sha256": "e" * 64,
        },
        "state_dim": 15,
        "action_dim": 7,
        "action_horizon": 10,
        "num_steps": 8,
        "zero_gate_contract": {
            "bitwise_equal": True,
            "max_abs_error": 0.0,
            "official_vs_prepared_max_abs_error": 0.0,
            "prepared_vs_zero_gate_max_abs_error": 0.0,
            "official_action_shape": [1, 10, 7],
            "prepared_action_shape": [1, 10, 7],
            "zero_gate_action_shape": [1, 10, 7],
            "official_action_sha256": "c" * 64,
            "prepared_action_sha256": "c" * 64,
            "zero_gate_action_sha256": "c" * 64,
            "dense_gate_count": 36,
            "object_gate_count": 36,
            "dense_gate_nonzero": 0,
            "object_gate_nonzero": 0,
            "dense_gate_sha256": "d" * 64,
            "object_gate_sha256": "d" * 64,
        },
        "evidence_contract": {
            "modality": "molmo_vision_patch",
            "dense_token_count": 729,
            "dense_token_width": 2304,
            "native_input_embedding_width": 2560,
            "dense_valid_count": 729,
            "object_count": 16,
            "object_address_width": 64,
            "object_value_width": 784,
            "dense_context_layers": 36,
            "object_context_layers": 36,
            "prepared_visual_vision_encoder_calls": 1,
        },
        "checkpoint_weight_shard_sha256": {
            f"model-{index:05d}-of-00005.safetensors": f"{index}" * 64 for index in range(1, 6)
        },
    }


def test_m0_report_acceptance_is_fail_closed() -> None:
    validate_m0_report(_passing_report(), config=_config())
    changed = _passing_report()
    changed["semantics"] = {**changed["semantics"], "targets_or_masks_in_runtime_input": True}
    with pytest.raises(ValueError, match="semantics changed"):
        validate_m0_report(changed, config=_config())


def test_m0_report_rejects_pruning_and_missing_weight_hashes() -> None:
    pruned = _passing_report()
    pruned["evidence_contract"] = {**pruned["evidence_contract"], "dense_valid_count": 728}
    with pytest.raises(ValueError, match="silently pruned"):
        validate_m0_report(pruned, config=_config())

    unhashed = _passing_report()
    unhashed["checkpoint_weight_shard_sha256"] = None
    with pytest.raises(ValueError, match="weight-shard hashes"):
        validate_m0_report(unhashed, config=_config())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["zero_gate_contract"].update(prepared_action_sha256="bad"),
            "hashes differ",
        ),
        (
            lambda report: report["zero_gate_contract"].update(
                official_vs_prepared_max_abs_error=1e-4
            ),
            "individual action parity",
        ),
        (
            lambda report: report["prepared_input_tensors"].update(pixel_values={"sha256": "raw"}),
            "retained raw visual",
        ),
        (
            lambda report: report["prepared_input_tensors"]["inputs_embeds"].update(
                shape=[1, 17, 2304]
            ),
            "inputs_embeds shape",
        ),
        (
            lambda report: report.update(prepared_action_condition_input_ids=None),
            "condition token identities",
        ),
        (
            lambda report: report["evidence_contract"].update(dense_context_layers=35),
            "dense_context_layers",
        ),
        (
            lambda report: report["evidence_contract"].update(
                prepared_visual_vision_encoder_calls=2
            ),
            "exactly one native vision-encoder",
        ),
        (
            lambda report: report["zero_gate_contract"].update(dense_gate_nonzero=1),
            "not exactly zero",
        ),
        (
            lambda report: report["zero_gate_contract"].update(
                official_action_sha256="not-a-digest",
                prepared_action_sha256="not-a-digest",
                zero_gate_action_sha256="not-a-digest",
            ),
            "not a SHA-256 digest",
        ),
        (
            lambda report: report.update(action_horizon=9),
            "action shape differs",
        ),
    ],
)
def test_m0_report_rejects_self_inconsistent_pass_payloads(mutation, message: str) -> None:
    report = _passing_report()
    mutation(report)
    with pytest.raises(ValueError, match=message):
        validate_m0_report(report, config=_config())


def test_m0_command_is_pinned_to_cloud_contract() -> None:
    config = _config()
    command = _runner_command(
        python=Path("/runtime/python"),
        root=_ROOT,
        config=config,
        checkpoint_root=Path("/mnt/checkpoints"),
        image=Path("/mnt/run/raw.png"),
        output=Path("/mnt/run/report.json"),
        task="move the block",
        setup_type="single-arm tabletop manipulation",
        control_mode="normalized relative end-effector pose",
    )

    assert command[0] == "/runtime/python"
    assert command[command.index("--num-steps") + 1] == "8"
    assert command[command.index("--action-horizon") + 1] == "10"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--state-dim") + 1] == "15"
    assert command[command.index("--action-dim") + 1] == "7"
    assert command[command.index("--dense-token-count") + 1] == "729"
    assert command[command.index("--dense-token-width") + 1] == "2304"
    assert command[command.index("--object-count") + 1] == "16"
    assert command[command.index("--object-address-width") + 1] == "64"
    assert command[command.index("--object-value-width") + 1] == "784"
    assert (
        command[command.index("--training-recipe-sha256") + 1]
        == (config["training_recipe"]["sha256"])
    )
    assert "--skip-weight-shard-hashes" not in command


def test_m0_contract_supports_the_frozen_m4_profile_without_rewinding_the_gate() -> None:
    config = json.loads(ACTION_CONFIG_PATH.read_text())
    report = _passing_report(config)

    validate_m0_report(report, config=config)
    _validate_required_reports(config)
    assert _followup_gates(config) == ("M4_action_adoption",)

    command = _runner_command(
        python=Path("/runtime/python"),
        root=_ROOT,
        config=config,
        checkpoint_root=Path("/mnt/checkpoints"),
        image=Path("/mnt/run/raw.png"),
        output=Path("/mnt/run/report.json"),
        task="move the block",
        setup_type="single-arm tabletop manipulation",
        control_mode="normalized relative end-effector pose",
    )
    assert (
        command[command.index("--training-recipe-sha256") + 1]
        == (config["training_recipe"]["sha256"])
    )


def test_m0_contract_rejects_non_molmo_profiles_and_reordered_reports() -> None:
    config = _config()
    config["profile"] = "lingbot-vla2-2xa100-40g"
    with pytest.raises(ValueError, match="only frozen MolmoAct2"):
        _followup_gates(config)

    config = _config()
    config["required_reports"] = list(reversed(config["required_reports"]))
    with pytest.raises(ValueError, match="M0 report contract changed"):
        _validate_required_reports(config)


def test_m0_production_context_supports_float32_master_weights_with_bfloat16_inputs() -> None:
    import torch

    projection = torch.nn.Linear(4, 3, bias=False, dtype=torch.float32)
    inputs = torch.ones(2, 4, dtype=torch.bfloat16)

    with _production_inference_context(torch, torch.device("cpu"), torch.bfloat16):
        assert torch.is_inference_mode_enabled()
        output = projection(inputs)

    assert output.dtype == torch.bfloat16


def test_cloud_output_must_be_persistent() -> None:
    assert _is_under_mnt(Path("/mnt/picf-next/runs"))
    assert not _is_under_mnt(Path("/tmp/picf-next/runs"))


def test_m0_report_split_emits_the_three_derived_report_payloads(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "m0_raw_report.json").write_text("{}\n")
    raw = {
        **_passing_report(),
        "assets": {"source_commit": "source", "trainer_commit": "trainer"},
        "image": {"path": "/mnt/raw.png", "sha256": "image"},
        "input_tensors": {"input_ids": {"sha256": "ids"}},
        "task": "move the block",
        "seed": 17,
        "num_steps": 8,
        "device": "cuda:0",
        "device_name": "A100",
        "dtype": "torch.bfloat16",
        "timings_s": {"official_policy_action_s": 1.0},
        "cuda_memory_bytes": {"official_action": {"peak_allocated": 1}},
    }

    _split_reports(
        raw,
        run_dir,
        root,
        root / "configs/cloud/2xa100_40g_gates.json",
    )

    assert (run_dir / "artifact_hashes.json").is_file()
    assert (run_dir / "full_weight_parity.json").is_file()
    assert (run_dir / "cuda_memory_latency.json").is_file()
    parity = json.loads((run_dir / "full_weight_parity.json").read_text())
    artifacts = json.loads((run_dir / "artifact_hashes.json").read_text())
    assert parity["prepared_input_tensors"] == raw["prepared_input_tensors"]
    assert (
        parity["prepared_action_condition_input_ids"] == raw["prepared_action_condition_input_ids"]
    )
    assert artifacts["prepared_input_tensor_sha256"]["inputs_embeds"] == "a" * 64
    assert (
        artifacts["prepared_action_condition_input_ids"]
        == raw["prepared_action_condition_input_ids"]
    )


def test_tracked_m0_raw_probe_matches_its_reviewed_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "evidence/molmoact2_m0"
    manifest = json.loads((evidence / "manifest.json").read_text())
    image = evidence / manifest["exported_file"]

    assert manifest["status"] == "PASS"
    assert manifest["visual_review"]["raw_camera_frame_only"] is True
    assert manifest["visual_review"]["mask_or_box_overlay"] is False
    assert manifest["visual_review"]["anchor_overlay"] is False
    assert hashlib.sha256(image.read_bytes()).hexdigest() == manifest["exported_sha256"]


def test_m0_runs_both_baselines_before_registering_the_picf_adapter() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/smoke_molmoact2_lerobot_full_weight.py").read_text()

    raw_baseline = source.index('timings["official_policy_action_s"]')
    prepared_baseline = source.index('timings["prepared_native_action_s"]')
    adapter_install = source.index("install_molmoact2_lerobot_picf_adapter(policy, adapter)")
    zero_gate = source.index('timings["zero_gate_full_evidence_action_s"]')

    assert raw_baseline < prepared_baseline < adapter_install < zero_gate
