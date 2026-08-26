"""Atomic causal training transaction for paired VidEoMT/LingBot queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from picf_next.videomt_exact.joint_training import (
    CompleteVidEoMTSourceObjective,
)
from picf_next.videomt_exact.runtime import (
    ExactVidEoMTCausalSequenceOutput,
    ExactVidEoMTOutput,
    ExactVidEoMTRuntime,
)


@dataclass(frozen=True, slots=True)
class CompleteCausalVidEoMTTrainingTransaction:
    """Five-frame source supervision with a one-frame host-visible boundary."""

    sequence: ExactVidEoMTCausalSequenceOutput
    source_objective: CompleteVidEoMTSourceObjective
    current_output: ExactVidEoMTOutput
    current_propagated_queries: torch.Tensor

    def __post_init__(self) -> None:
        if self.sequence.merged.class_logits.shape[1] != 5:
            raise ValueError("complete causal source transaction requires five frames")
        if self.current_output is not self.sequence.per_frame[0]:
            raise ValueError("host-visible source output must be the first causal frame")
        expected = self.sequence.propagated_queries_by_frame[0]
        if self.current_propagated_queries is not expected:
            raise ValueError("committed source state must be the current-frame boundary")
        if self.current_output.class_logits.shape[1] != 1:
            raise ValueError("host-visible source output must contain exactly one frame")
        if self.current_propagated_queries.shape != self.sequence.merged.propagated_queries.shape:
            raise ValueError("current source state differs from the complete source ABI")


def run_complete_causal_videomt_training_transaction(
    runtime: ExactVidEoMTRuntime,
    source_objective: nn.Module,
    *,
    normalized_padded_rgb: torch.Tensor,
    clip_targets: Sequence[Mapping[str, torch.Tensor]],
    previous_queries: torch.Tensor | None,
    reset: torch.Tensor,
) -> CompleteCausalVidEoMTTrainingTransaction:
    """Run the complete source objective without exposing future frames to host.

    Frame zero is the current observation at the action decision. Frames one
    through four remain inside the unchanged five-frame source criterion. The
    source runtime is restored to the post-current boundary before returning,
    so future supervision can backpropagate through the current query state but
    cannot be committed as online memory.
    """

    if not isinstance(runtime, ExactVidEoMTRuntime):
        raise TypeError("paired source training requires the exact VidEoMT runtime")
    if not isinstance(source_objective, nn.Module):
        raise TypeError("paired source training requires a module objective")
    if normalized_padded_rgb.ndim == 4:
        batch, time = 1, normalized_padded_rgb.shape[0]
    elif normalized_padded_rgb.ndim == 5:
        batch, time = normalized_padded_rgb.shape[:2]
    else:
        raise ValueError("paired source RGB must be [T,C,H,W] or [B,T,C,H,W]")
    if time != 5:
        raise ValueError("complete paired source training requires exactly five causal frames")
    if reset.shape != (batch,) or reset.dtype != torch.bool:
        raise ValueError("paired source reset mask must be boolean [batch]")
    if len(clip_targets) != batch:
        raise ValueError("paired source targets must contain one clip per batch sample")

    try:
        runtime.bind_mixed_propagated_queries(previous_queries, reset=reset)
        sequence = runtime.forward_causal_sequence(normalized_padded_rgb, resume=True)
        objective = source_objective(sequence.merged, clip_targets)
        if not isinstance(objective, CompleteVidEoMTSourceObjective):
            raise TypeError("complete source objective returned an incompatible result")
        current_output = sequence.per_frame[0]
        current_state = sequence.propagated_queries_by_frame[0]
        runtime.restore_propagated_queries(current_state)
    except BaseException:
        runtime.reset_state()
        raise
    return CompleteCausalVidEoMTTrainingTransaction(
        sequence=sequence,
        source_objective=objective,
        current_output=current_output,
        current_propagated_queries=current_state,
    )
