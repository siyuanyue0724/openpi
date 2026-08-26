from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pytest

from tools.probe_lingbot_distributed_muon_collective_alignment import (
    EXPECTED_CLASSIFICATION,
    GLOBAL_SHAPE,
    LOCAL_SHAPE,
    SCHEMA,
    SHARD_DIM,
    ProbeContractError,
    _atomic_write_json_create,
    _bounded_timeout,
    _load_muon_module,
    _muon_module_key,
    _prepare_paths,
    _require_finite_tree,
    _require_launch_environment,
    _validate_preflight_reports,
    _validate_rank_reports,
    _validate_visible_devices,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/probe_lingbot_distributed_muon_collective_alignment.py"
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64


def _source_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "lingbot-source"
    source = checkout / "lingbotvla" / "optim" / "muon.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            (
                "_KIND_MOE_GATHER_3D = 'moe_gather_3d'",
                "class DistributedMuon:",
                "    pass",
                "def _classify_param(_parameter):",
                "    return _KIND_MOE_GATHER_3D",
                "",
            )
        ),
        encoding="ascii",
    )
    return checkout


def _launch_environment(rank: int = 0) -> dict[str, str]:
    return {
        "LOCAL_RANK": str(rank),
        "LOCAL_WORLD_SIZE": "2",
        "RANK": str(rank),
        "WORLD_SIZE": "2",
    }


def _preflight_report(rank: int) -> dict[str, object]:
    return {
        "backend": "nccl",
        "cuda_device_count": 2,
        "cuda_runtime": "12.8",
        "device_capability": [8, 0],
        "device_index": rank,
        "device_name": "NVIDIA A100-SXM4-40GB",
        "device_total_memory": 40 * 1024**3,
        "hostname": "probe-host",
        "local_rank": rank,
        "local_world_size": 2,
        "muon_source_sha256": _DIGEST_A,
        "output_json": "/mnt/probe/pass.json",
        "rank": rank,
        "source_checkout": "/mnt/lingbot",
        "torch_version": "2.8.0",
        "world_size": 2,
    }


def _rank_report(rank: int) -> dict[str, object]:
    has_gradient = rank == 0
    return {
        "classification": EXPECTED_CLASSIFICATION,
        "gradient_after_clip_sha256": _DIGEST_D if has_gradient else None,
        "gradient_present": has_gradient,
        "gradient_sha256": _DIGEST_C if has_gradient else None,
        "global_shape": list(GLOBAL_SHAPE),
        "local_rank": rank,
        "local_gradient_norm_after_clip": 1.0 if has_gradient else None,
        "local_shape": list(LOCAL_SHAPE),
        "optimizer_state_after_entry_count": 1 if has_gradient else 0,
        "optimizer_state_after_present": has_gradient,
        "optimizer_state_after_sha256": _DIGEST_D if has_gradient else None,
        "optimizer_state_before_entry_count": 0,
        "parameter_after_sha256": _DIGEST_B if has_gradient else _DIGEST_A,
        "parameter_before_sha256": _DIGEST_A,
        "parameter_changed": has_gradient,
        "preclip_global_norm": 2.0,
        "rank": rank,
        "shard_dimensions": [SHARD_DIM],
        "step_elapsed_seconds": 0.25,
    }


def test_probe_source_contains_exact_distributed_regression_contract() -> None:
    source = TOOL.read_text(encoding="utf-8")
    for fragment in (
        'EXPECTED_CLASSIFICATION = "moe_gather_3d"',
        "SHARD_DIM = 1",
        'dist.init_process_group(\n        backend="nccl"',
        "DTensor, Shard",
        "placements = (shard_class(SHARD_DIM),)",
        "parameter.grad = gradient",
        "parameter.grad = None",
        "classification = module._classify_param(parameter)",
        "clip_lingbot_distributed_l2_grad_norm_(",
        "optimizer.step()",
        "dist.barrier(device_ids=[local_rank])",
        "dist.all_gather_object(rank_reports, local_report)",
        '"optimizer_state_after_present": state_after["present"]',
        '"status": "PASS"',
        "write_text_durable_exclusive(path, encoded, encoding=\"ascii\")",
        "os._exit(WATCHDOG_EXIT_CODE)",
    ):
        assert fragment in source


def test_probe_schema_and_tensor_contract_are_fixed() -> None:
    assert SCHEMA.endswith("collective-alignment-probe.v1")
    assert EXPECTED_CLASSIFICATION == "moe_gather_3d"
    assert len(GLOBAL_SHAPE) == len(LOCAL_SHAPE) == 3
    assert SHARD_DIM == 1
    assert SHARD_DIM > 0
    assert GLOBAL_SHAPE[SHARD_DIM] == 2 * LOCAL_SHAPE[SHARD_DIM]


def test_probe_requires_exact_two_rank_single_host_torchrun() -> None:
    assert _require_launch_environment(_launch_environment(1)) == {
        "local_rank": 1,
        "local_world_size": 2,
        "rank": 1,
        "world_size": 2,
    }
    for name, value in (
        ("WORLD_SIZE", "4"),
        ("LOCAL_WORLD_SIZE", "1"),
        ("RANK", "2"),
        ("LOCAL_RANK", "2"),
    ):
        environment = _launch_environment()
        environment[name] = value
        with pytest.raises(ProbeContractError):
            _require_launch_environment(environment)
    crossed = _launch_environment()
    crossed["RANK"] = "1"
    with pytest.raises(ProbeContractError, match="rank == local rank"):
        _require_launch_environment(crossed)


def test_probe_timeout_is_strictly_bounded() -> None:
    assert _bounded_timeout("10") == 10
    assert _bounded_timeout("300") == 300
    for value in ("9", "301", "not-an-integer"):
        with pytest.raises(argparse.ArgumentTypeError):
            _bounded_timeout(value)


def test_probe_rejects_duplicate_or_insufficient_cuda_visibility() -> None:
    _validate_visible_devices({})
    _validate_visible_devices({"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"})
    with pytest.raises(ProbeContractError, match="fewer than two"):
        _validate_visible_devices({"CUDA_VISIBLE_DEVICES": "GPU-a"})
    with pytest.raises(ProbeContractError, match="duplicate"):
        _validate_visible_devices({"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-a"})


def test_probe_paths_are_checkout_agnostic_and_non_overlapping(tmp_path: Path) -> None:
    checkout = _source_checkout(tmp_path)
    output = tmp_path / "reports" / "pass.json"
    source, muon_source, resolved_output = _prepare_paths(checkout, output)
    assert source == checkout.resolve()
    assert muon_source == (checkout / "lingbotvla/optim/muon.py").resolve()
    assert resolved_output == output.resolve()

    with pytest.raises(ProbeContractError, match="overlap"):
        _prepare_paths(checkout, checkout / "probe.json")
    output.parent.mkdir()
    output.write_text("already exists", encoding="ascii")
    with pytest.raises(ProbeContractError, match="already exists"):
        _prepare_paths(checkout, output)


def test_probe_imports_distributed_muon_from_exact_supplied_source(tmp_path: Path) -> None:
    checkout = _source_checkout(tmp_path)
    muon_source = (checkout / "lingbotvla/optim/muon.py").resolve()
    module_key = _muon_module_key(muon_source)
    try:
        module, optimizer_class = _load_muon_module(muon_source)
        assert module.__file__ == str(muon_source)
        assert optimizer_class.__name__ == "DistributedMuon"
        assert module._classify_param(object()) == EXPECTED_CLASSIFICATION
        with pytest.raises(ProbeContractError, match="duplicate"):
            _load_muon_module(muon_source)
    finally:
        sys.modules.pop(module_key, None)


def test_preflight_validation_accepts_only_homogeneous_unique_two_rank_hardware() -> None:
    reports = _validate_preflight_reports([_preflight_report(1), _preflight_report(0)])
    assert [report["rank"] for report in reports] == [0, 1]

    duplicate = [_preflight_report(0), _preflight_report(0)]
    with pytest.raises(ProbeContractError, match="duplicate or missing global ranks"):
        _validate_preflight_reports(duplicate)

    heterogeneous = [_preflight_report(0), _preflight_report(1)]
    heterogeneous[1]["device_name"] = "different GPU"
    with pytest.raises(ProbeContractError, match="homogeneous hardware"):
        _validate_preflight_reports(heterogeneous)

    mismatched_source = [_preflight_report(0), _preflight_report(1)]
    mismatched_source[1]["source_checkout"] = "/mnt/other-lingbot"
    with pytest.raises(ProbeContractError, match="source_checkout"):
        _validate_preflight_reports(mismatched_source)


def test_rank_validation_accepts_grad_none_collective_alignment() -> None:
    reports = _validate_rank_reports([_rank_report(1), _rank_report(0)])
    assert [report["rank"] for report in reports] == [0, 1]
    assert reports[0]["parameter_changed"] is True
    assert reports[1]["parameter_changed"] is False
    assert reports[1]["optimizer_state_after_present"] is False
    assert reports[1]["optimizer_state_after_entry_count"] == 0
    assert reports[1]["optimizer_state_after_sha256"] is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("optimizer_state_after_present", True, "created optimizer state"),
        ("optimizer_state_after_entry_count", 1, "optimizer state is not empty"),
        ("optimizer_state_after_sha256", _DIGEST_D, "optimizer state digest"),
        ("parameter_changed", True, "parameter changed"),
        ("parameter_after_sha256", _DIGEST_B, "before/after parameter digests differ"),
        ("gradient_present", True, "grad=None contract"),
        ("gradient_sha256", _DIGEST_C, "grad=None contract"),
        ("gradient_after_clip_sha256", _DIGEST_D, "grad=None contract"),
        ("local_gradient_norm_after_clip", 1.0, "grad=None contract"),
    ),
)
def test_rank_validation_rejects_grad_none_rank_side_effects(
    field: str,
    value: object,
    message: str,
) -> None:
    reports = [_rank_report(0), _rank_report(1)]
    reports[1][field] = value
    with pytest.raises(ProbeContractError, match=message):
        _validate_rank_reports(reports)


def test_rank_validation_rejects_wrong_classification_and_duplicate_rank() -> None:
    wrong_class = [_rank_report(0), _rank_report(1)]
    wrong_class[0]["classification"] = "moe_local_3d"
    with pytest.raises(ProbeContractError, match="MOE_GATHER_3D"):
        _validate_rank_reports(wrong_class)

    duplicate = [_rank_report(0), _rank_report(0)]
    with pytest.raises(ProbeContractError, match="duplicate or missing ranks"):
        _validate_rank_reports(duplicate)


def test_probe_report_validation_fails_closed_on_non_finite_values() -> None:
    with pytest.raises(ProbeContractError, match="non-finite"):
        _require_finite_tree({"nested": [1.0, math.nan]})
    reports = [_rank_report(0), _rank_report(1)]
    reports[0]["step_elapsed_seconds"] = math.inf
    with pytest.raises(ProbeContractError, match="non-finite"):
        _validate_rank_reports(reports)


def test_atomic_pass_publication_is_create_only_and_leaves_no_partial(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "pass.json"
    payload = {
        "rank_reports": [_rank_report(0), _rank_report(1)],
        "schema": SCHEMA,
        "status": "PASS",
    }
    _atomic_write_json_create(output, payload)
    assert json.loads(output.read_text(encoding="ascii")) == payload
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    with pytest.raises(ProbeContractError, match="refusing to replace"):
        _atomic_write_json_create(output, {"status": "PASS"})
    assert json.loads(output.read_text(encoding="ascii")) == payload
