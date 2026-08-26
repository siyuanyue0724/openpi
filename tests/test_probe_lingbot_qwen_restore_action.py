from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from picf_next.contracts import ContractError
from tools.probe_lingbot_qwen_restore_action import (
    _configure_official_action_sampling,
    _move_model_inputs,
    _sample_from_payload,
    _select_action_payloads,
)


def _record(index: int, task: str, camera: str) -> dict[str, object]:
    return {"global_index": index, "task_key": task, "camera_name": camera}


def test_action_payload_selection_deduplicates_cameras_and_spans_bank() -> None:
    records = [
        _record(1, "a", "static"),
        _record(1, "a", "gripper"),
        _record(2, "b", "static"),
        _record(3, "c", "static"),
        _record(4, "d", "static"),
    ]
    selected = _select_action_payloads(records, 2)
    assert [(item["global_index"], item["task_key"]) for item in selected] == [
        (1, "a"),
        (3, "c"),
    ]


def test_action_payload_selection_fails_closed_on_bad_or_small_input() -> None:
    with pytest.raises(ContractError, match="no record list"):
        _select_action_payloads({}, 1)
    with pytest.raises(ContractError, match="too few unique"):
        _select_action_payloads([_record(1, "a", "static")], 2)


class _ActionIndex:
    def __init__(self, segments: tuple[SimpleNamespace, ...]) -> None:
        self.segments = segments

    def stateful_transition_sample(
        self, segment_index: int, global_index: int, *, action_horizon: int
    ) -> tuple[int, int, int]:
        return segment_index, global_index, action_horizon


def _segment(index: int, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        start=start,
        end=end,
        task_key="rotate_pink_block_right",
        instruction="grasp the pink block and turn it right",
    )


def test_action_sample_prefers_unique_originating_segment_for_duplicate_annotations() -> None:
    index = _ActionIndex((_segment(8, 100, 164), _segment(8534, 96, 160)))
    payload = {
        "global_index": 100,
        "task_key": "rotate_pink_block_right",
        "instruction": "grasp the pink block and turn it right",
    }
    assert _sample_from_payload(index, payload, 64) == (8, 100, 64)


def test_action_sample_rejects_unresolved_duplicate_annotations() -> None:
    index = _ActionIndex((_segment(8, 90, 164), _segment(8534, 96, 160)))
    payload = {
        "global_index": 100,
        "task_key": "rotate_pink_block_right",
        "instruction": "grasp the pink block and turn it right",
    }
    with pytest.raises(ContractError, match="not uniquely addressable"):
        _sample_from_payload(index, payload, 64)


def test_move_model_inputs_matches_native_dtype_boundary() -> None:
    metadata = {"out_of_band": True}
    moved = _move_model_inputs(
        {
            "pixels": torch.ones((1, 2), dtype=torch.float32),
            "tokens": torch.ones((1, 2), dtype=torch.int64),
            "metadata": metadata,
        },
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        torch_module=torch,
    )
    assert moved["pixels"].dtype == torch.bfloat16
    assert moved["tokens"].dtype == torch.int64
    assert moved["metadata"] is metadata


def test_official_action_sampling_settings_are_applied_after_config_merge() -> None:
    config = SimpleNamespace(
        attention_implementation="flex_cached",
        use_cache=False,
        use_compile=True,
        use_lm_head=False,
        num_steps=99,
        vit_attn_implementation="flash_attention_2",
    )
    _configure_official_action_sampling(config, num_steps=2)
    assert config.use_cache is True
    assert config.use_compile is False
    assert config.use_lm_head is True
    assert config.num_steps == 2
    assert config.attention_implementation == "eager"
    assert config.vit_attn_implementation == "eager"


def test_official_action_sampling_rejects_invalid_step_count() -> None:
    with pytest.raises(ContractError, match="positive num_steps"):
        _configure_official_action_sampling(SimpleNamespace(), num_steps=0)
