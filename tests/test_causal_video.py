from __future__ import annotations

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.causal_video import (
    CausalVideoEncoderInput,
    CausalVideoSourceFrame,
    build_causal_video_clip,
    left_pad_causal_video_clip,
)


def _frame(value: int, timestamp_s: float) -> CausalVideoSourceFrame:
    image = np.full((4, 5, 3), value, dtype=np.uint8)
    image.setflags(write=False)
    return CausalVideoSourceFrame.from_image(
        image,
        timestamp_s=timestamp_s,
        sensor_key="external_rgb",
    )


def test_causal_clip_deduplicates_exact_sources_without_padding() -> None:
    first = _frame(1, 0.0)
    second = _frame(2, 0.1)
    third = _frame(3, 0.2)
    clip = build_causal_video_clip(
        (first, second, second, third),
        current_timestamp_s=0.2,
        maximum_frames=64,
        tubelet_size=2,
    )

    assert clip is not None
    assert clip.frame_timestamps_s == (0.1, 0.2)
    assert clip.source_frame_sha256 == (second.source_sha256, third.source_sha256)
    assert len(set(clip.source_frame_sha256)) == 2


def test_causal_clip_requires_unique_content_at_each_timestamp() -> None:
    with pytest.raises(ContractError, match="conflicting source content"):
        build_causal_video_clip(
            (_frame(1, 0.0), _frame(2, 0.0)),
            current_timestamp_s=0.0,
            maximum_frames=4,
            tubelet_size=2,
        )


def test_causal_clip_tail_sampling_keeps_current_and_complete_tubelets() -> None:
    frames = tuple(_frame(index, index / 10.0) for index in range(9))
    clip = build_causal_video_clip(
        frames,
        current_timestamp_s=0.8,
        maximum_frames=4,
        tubelet_size=2,
        frame_step=2,
    )

    assert clip is not None
    assert clip.frame_timestamps_s == (0.2, 0.4, 0.6, 0.8)
    assert clip.images[-1] is frames[-1].image


def test_causal_clip_returns_missing_until_one_complete_tubelet_exists() -> None:
    assert (
        build_causal_video_clip(
            (_frame(1, 0.0),),
            current_timestamp_s=0.0,
            maximum_frames=64,
            tubelet_size=2,
        )
        is None
    )


def test_temporal_interventions_keep_current_observation_fixed() -> None:
    frames = tuple(_frame(index, index / 10.0) for index in range(4))
    clip = build_causal_video_clip(
        frames,
        current_timestamp_s=0.3,
        maximum_frames=4,
        tubelet_size=2,
    )
    assert clip is not None

    permuted = clip.permute_history_content((2, 0, 1))
    assert permuted.images[-1] is clip.images[-1]
    assert permuted.source_frame_sha256[-1] == clip.source_frame_sha256[-1]
    assert permuted.frame_timestamps_s == clip.frame_timestamps_s
    assert permuted.intervention == "history-content-permutation.v1"

    shifted = clip.shift_history_timestamps(0.01)
    assert all(
        actual is expected for actual, expected in zip(shifted.images, clip.images, strict=True)
    )
    assert shifted.frame_timestamps_s[-1] == clip.frame_timestamps_s[-1]
    assert shifted.frame_timestamps_s[:-1] == pytest.approx((0.01, 0.11, 0.21))


def test_invalid_history_timestamp_intervention_fails_closed() -> None:
    clip = build_causal_video_clip(
        (_frame(1, 0.0), _frame(2, 0.1)),
        current_timestamp_s=0.1,
        maximum_frames=2,
        tubelet_size=2,
    )
    assert clip is not None
    with pytest.raises(ContractError, match="strictly increasing"):
        clip.shift_history_timestamps(0.2)


def test_fixed_encoder_padding_is_explicit_causal_and_source_auditable() -> None:
    frames = tuple(_frame(index, index / 10.0) for index in range(3))
    clip = build_causal_video_clip(
        frames,
        current_timestamp_s=0.2,
        maximum_frames=64,
        tubelet_size=1,
    )
    assert clip is not None

    fixed = left_pad_causal_video_clip(clip, frame_count=5)

    assert fixed.padding_count == 2
    assert fixed.source_valid == (False, False, True, True, True)
    assert fixed.images[0] is fixed.images[1] is fixed.images[2]
    assert fixed.frame_timestamps_s == (0.0, 0.0, 0.0, 0.1, 0.2)
    assert fixed.source_frame_sha256[:3] == (frames[0].source_sha256,) * 3


def test_fixed_encoder_input_rejects_hidden_real_frames_in_padding() -> None:
    first = _frame(1, 0.0)
    second = _frame(2, 0.1)
    with pytest.raises(ContractError, match="one left prefix"):
        CausalVideoEncoderInput(
            images=(first.image, second.image),
            frame_timestamps_s=(0.0, 0.1),
            source_frame_sha256=(first.source_sha256, second.source_sha256),
            source_valid=(True, False),
            sensor_key="external_rgb",
            current_timestamp_s=0.1,
        )
