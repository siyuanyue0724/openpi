from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from picf_next.videomt_exact.joint_training import (
    COMPLETE_VIDEOMT_RAW_LOSS_NAMES,
    COMPLETE_VIDEOMT_WEIGHTED_LOSS_NAMES,
    CompleteVidEoMTSourceObjective,
)
from picf_next.videomt_exact.paired_training import (
    HOST_ALIGNED_CURRENT_BOUNDARY,
    run_complete_causal_videomt_training_transaction,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTRuntime
from tests.videomt_exact.test_runtime_contract import _HookCompatibleVidEoMT


class _CompleteObjectiveStub(nn.Module):
    def forward(self, output, targets):
        zero = output.class_logits.sum() * 0 + output.mask_logits.sum() * 0
        raw = {
            name: zero + index for index, name in enumerate(sorted(COMPLETE_VIDEOMT_RAW_LOSS_NAMES))
        }
        weighted = {
            name: zero + index
            for index, name in enumerate(sorted(COMPLETE_VIDEOMT_WEIGHTED_LOSS_NAMES))
        }
        return CompleteVidEoMTSourceObjective(
            total=torch.stack(tuple(weighted.values())).sum(),
            raw_losses=raw,
            weighted_losses=weighted,
            target_count=sum(int(target["labels"].numel()) for target in targets),
        )


def _targets(batch: int) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.zeros(1, dtype=torch.long),
            "ids": torch.zeros(1, 5, dtype=torch.long),
            "masks": torch.ones(1, 5, 2, 2),
            "valid_pixels": torch.ones(5, 2, 2, dtype=torch.bool),
        }
        for _ in range(batch)
    ]


def test_complete_causal_transaction_commits_current_not_future_source_state() -> None:
    model = _HookCompatibleVidEoMT().train()
    runtime = ExactVidEoMTRuntime(object(), model)
    frames = torch.randn(2, 5, 3, 16, 16)
    previous = torch.randn(2, 200, 1024)
    result = run_complete_causal_videomt_training_transaction(
        runtime,
        _CompleteObjectiveStub(),
        normalized_padded_rgb=frames,
        clip_targets=_targets(2),
        previous_queries=previous,
        reset=torch.tensor([True, False]),
    )

    assert result.current_output.class_logits.shape == (2, 1, 200, 41)
    assert result.sequence.merged.class_logits.shape == (2, 5, 200, 41)
    assert len(result.sequence.merged.auxiliary_outputs) == 4
    torch.testing.assert_close(runtime.propagated_queries, result.current_propagated_queries)
    assert not torch.equal(
        result.current_propagated_queries,
        result.sequence.propagated_queries_by_frame[-1],
    )
    result.source_objective.total.backward()
    assert model.class_head.weight.grad is not None
    assert model.q.weight.grad is not None


def test_future_source_supervision_cannot_change_the_host_visible_current_frame() -> None:
    first_model = _HookCompatibleVidEoMT().train()
    second_model = deepcopy(first_model).train()
    first_frames = torch.randn(1, 5, 3, 16, 16)
    second_frames = first_frames.clone()
    second_frames[:, 1:] = torch.randn_like(second_frames[:, 1:]) * 100

    first = run_complete_causal_videomt_training_transaction(
        ExactVidEoMTRuntime(object(), first_model),
        _CompleteObjectiveStub(),
        normalized_padded_rgb=first_frames,
        clip_targets=_targets(1),
        previous_queries=None,
        reset=torch.ones(1, dtype=torch.bool),
    )
    second = run_complete_causal_videomt_training_transaction(
        ExactVidEoMTRuntime(object(), second_model),
        _CompleteObjectiveStub(),
        normalized_padded_rgb=second_frames,
        clip_targets=_targets(1),
        previous_queries=None,
        reset=torch.ones(1, dtype=torch.bool),
    )

    torch.testing.assert_close(
        first.current_output.class_logits, second.current_output.class_logits
    )
    torch.testing.assert_close(first.current_output.mask_logits, second.current_output.mask_logits)
    torch.testing.assert_close(
        first.current_propagated_queries,
        second.current_propagated_queries,
    )
    assert not torch.equal(
        first.sequence.propagated_queries_by_frame[-1],
        second.sequence.propagated_queries_by_frame[-1],
    )


def test_augmented_source_view_cannot_change_host_aligned_online_boundary() -> None:
    first_model = _HookCompatibleVidEoMT().train()
    second_model = deepcopy(first_model).train()
    first_source = torch.randn(1, 5, 3, 16, 16)
    second_source = torch.randn_like(first_source) * 100
    host_current = torch.randn(1, 1, 3, 16, 16)
    previous = torch.randn(1, 200, 1024)

    first = run_complete_causal_videomt_training_transaction(
        ExactVidEoMTRuntime(object(), first_model),
        _CompleteObjectiveStub(),
        normalized_padded_rgb=first_source,
        host_aligned_current_rgb=host_current,
        clip_targets=_targets(1),
        previous_queries=previous,
        reset=torch.zeros(1, dtype=torch.bool),
    )
    second = run_complete_causal_videomt_training_transaction(
        ExactVidEoMTRuntime(object(), second_model),
        _CompleteObjectiveStub(),
        normalized_padded_rgb=second_source,
        host_aligned_current_rgb=host_current,
        clip_targets=_targets(1),
        previous_queries=previous,
        reset=torch.zeros(1, dtype=torch.bool),
    )

    assert first.current_boundary == HOST_ALIGNED_CURRENT_BOUNDARY
    assert second.current_boundary == HOST_ALIGNED_CURRENT_BOUNDARY
    torch.testing.assert_close(first.current_output.mask_logits, second.current_output.mask_logits)
    torch.testing.assert_close(
        first.current_propagated_queries,
        second.current_propagated_queries,
    )
    assert not torch.equal(first.sequence.merged.mask_logits, second.sequence.merged.mask_logits)
    (first.source_objective.total + first.current_output.mask_logits.sum()).backward()
    assert first_model.class_head.weight.grad is not None
    assert first_model.q.weight.grad is not None
