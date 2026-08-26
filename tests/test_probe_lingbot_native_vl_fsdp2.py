from __future__ import annotations

from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from tools.probe_lingbot_native_vl_fsdp2 import (
    _rank_record_indices,
    _validate_native_vl_fsdp2_placement,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/probe_lingbot_native_vl_fsdp2.py"


def test_rank_record_selection_is_deterministic_and_source_separated() -> None:
    assert _rank_record_indices(16) == (0, 8)
    assert _rank_record_indices(3) == (0, 1)
    with pytest.raises(ContractError, match="too few records"):
        _rank_record_indices(1)


@pytest.mark.parametrize("placement", (FSDP2_GPU_SHARDED, FSDP2_CPU_OFFLOAD))
def test_native_vl_fsdp2_accepts_one_group_tied_weight_placements(
    placement: str,
) -> None:
    assert _validate_native_vl_fsdp2_placement(placement) == placement


def test_native_vl_fsdp2_rejects_split_tied_weight_placement() -> None:
    with pytest.raises(ContractError, match="must belong to one fully_shard group"):
        _validate_native_vl_fsdp2_placement(FSDP2_SELECTIVE_EMBEDDING_OFFLOAD)


def test_g1_source_uses_the_shared_qwen_production_fsdp2_boundary() -> None:
    source = TOOL.read_text()
    required = (
        "retie_and_validate_native_qwen_lm_head(policy)",
        "_validate_native_vl_fsdp2_placement(args.fsdp2_placement)",
        "policy = build_parallelize_model(",
        "enable_fsdp_offload=full_cpu_offload",
        "register_native_vl_fsdp_forward_method(policy)",
        "validate_native_vl_optimizer_membership(policy, optimizer)",
        "loss = run_native_vl_grounding_forward(policy, batch)",
        "loss.backward()",
        '"shared_embedding"',
        '"language_layers"',
        '"vision_layers"',
    )
    assert all(fragment in source for fragment in required)
    assert "optimizer.step(" not in source
    assert "semantic_scorer" not in source
    assert "teacher_model" not in source
