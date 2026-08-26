from __future__ import annotations

import pytest

from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    SELECTIVE_EMBEDDING_PARAMETER,
    SELECTIVE_FROZEN_VISION_MODULE_PREFIX,
    validate_fsdp2_placement,
    validate_fsdp2_storage_report,
)


def _storage_report(placement: str) -> dict[str, object]:
    if placement == FSDP2_CPU_OFFLOAD:
        cpu_tensors, cpu_elements = 2, 10
        cuda_tensors, cuda_elements = 0, 0
        names: list[str] = []
    elif placement == FSDP2_GPU_SHARDED:
        cpu_tensors, cpu_elements = 0, 0
        cuda_tensors, cuda_elements = 2, 10
        names = []
    elif placement in {
        FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
        FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    }:
        cpu_tensors, cpu_elements = 2, 7
        cuda_tensors, cuda_elements = 1, 3
        names = [
            SELECTIVE_EMBEDDING_PARAMETER,
            f"{SELECTIVE_FROZEN_VISION_MODULE_PREFIX}0.attn.qkv.weight",
        ]
    else:
        cpu_tensors, cpu_elements = 1, 4
        cuda_tensors, cuda_elements = 1, 6
        names = [SELECTIVE_EMBEDDING_PARAMETER]
    return {
        "parameter_tensors": cpu_tensors + cuda_tensors,
        "local_elements": 10,
        "master_dtype": "float32",
        "placement": placement,
        "cpu_parameter_tensors": cpu_tensors,
        "cpu_local_elements": cpu_elements,
        "cuda_parameter_tensors": cuda_tensors,
        "cuda_local_elements": cuda_elements,
        "selective_cpu_parameter_names": names,
    }


@pytest.mark.parametrize(
    "placement",
    [
        FSDP2_CPU_OFFLOAD,
        FSDP2_GPU_SHARDED,
        FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
        FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    ],
)
def test_fsdp2_storage_report_accepts_only_exact_declared_placement(placement: str) -> None:
    report = _storage_report(placement)
    assert validate_fsdp2_placement(placement) == placement
    assert validate_fsdp2_storage_report(report, expected_placement=placement) == report


def test_fsdp2_storage_report_rejects_selective_name_or_accounting_drift() -> None:
    report = _storage_report(FSDP2_SELECTIVE_EMBEDDING_OFFLOAD)
    report["selective_cpu_parameter_names"] = ["model.wrong.weight"]
    with pytest.raises(ValueError, match="declared placement"):
        validate_fsdp2_storage_report(
            report,
            expected_placement=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        )


def test_fsdp2_storage_report_rejects_nonvision_selective_offload() -> None:
    report = _storage_report(FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD)
    report["selective_cpu_parameter_names"] = [
        SELECTIVE_EMBEDDING_PARAMETER,
        "model.action.weight",
    ]
    with pytest.raises(ValueError, match="declared placement"):
        validate_fsdp2_storage_report(
            report,
            expected_placement=FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
        )

    trainable_report = _storage_report(
        FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD
    )
    trainable_report["selective_cpu_parameter_names"] = [
        SELECTIVE_EMBEDDING_PARAMETER,
        "model.action.weight",
    ]
    with pytest.raises(ValueError, match="declared placement"):
        validate_fsdp2_storage_report(
            trainable_report,
            expected_placement=FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
        )

    report = _storage_report(FSDP2_SELECTIVE_EMBEDDING_OFFLOAD)
    report["cuda_local_elements"] = 5
    with pytest.raises(ValueError, match="accounting is inconsistent"):
        validate_fsdp2_storage_report(
            report,
            expected_placement=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        )


@pytest.mark.parametrize("value", [None, True, "", "hybrid", 1])
def test_fsdp2_placement_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        validate_fsdp2_placement(value)
