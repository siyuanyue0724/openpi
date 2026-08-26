from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import picf_next.videomt_exact.joint_data as joint_data
from picf_next.contracts import ContractError
from picf_next.videomt_exact.calvin_targets import PreparedCalvinVidEoMTClip
from picf_next.videomt_exact.preprocessing import PreparedVidEoMTFrames


class _Dataset:
    def __init__(self) -> None:
        self.indices = {f"sample-{index}": index for index in range(20)}

    def future_source_global_indices_by_key(
        self,
        sample_key: str,
        *,
        count: int,
    ) -> tuple[int, ...]:
        start = self.indices[sample_key]
        return tuple(range(start + 1, start + count + 1))

    def source_global_index_by_key(self, sample_key: str) -> int:
        return self.indices[sample_key]


def _prepared(height: int, width: int, value: float) -> PreparedCalvinVidEoMTClip:
    frames = PreparedVidEoMTFrames(
        model_input=torch.full((5, 3, height, width), value),
        resized_rgb=tuple(np.zeros((height, width, 3), dtype=np.uint8) for _ in range(5)),
        original_sizes=((height, width),) * 5,
        resized_sizes=((height, width),) * 5,
        padded_size=(height, width),
    )
    return PreparedCalvinVidEoMTClip(
        frames=frames,
        target={
            "labels": torch.zeros(1, dtype=torch.long),
            "ids": torch.zeros((1, 5), dtype=torch.long),
            "masks": torch.ones((1, 5, height, width), dtype=torch.float32),
            "valid_pixels": torch.ones((5, height, width), dtype=torch.bool),
        },
        identity_keys=(f"object-{height}",),
        camera_name="static",
    )


def test_current_future_source_indices_are_current_then_four_future() -> None:
    assert joint_data.current_future_source_indices(_Dataset(), "sample-3") == (3, 4, 5, 6, 7)


def test_current_future_source_indices_reject_nonconsecutive_source() -> None:
    class _NonConsecutive(_Dataset):
        def future_source_global_indices_by_key(
            self,
            sample_key: str,
            *,
            count: int,
        ) -> tuple[int, ...]:
            del sample_key, count
            return (4, 99, 100, 101)

    with pytest.raises(ContractError, match="not five consecutive"):
        joint_data.current_future_source_indices(_NonConsecutive(), "sample-3")


def test_source_eligibility_receipt_covers_every_frozen_domain_key() -> None:
    plan = SimpleNamespace(
        plan_sha256="a" * 64,
        episodes=(
            SimpleNamespace(sample_keys=("sample-1", "sample-5")),
            SimpleNamespace(sample_keys=("sample-9",)),
        ),
    )
    receipt = joint_data.audit_native_videomt_source_eligibility(_Dataset(), plan)

    assert receipt.stream_plan_sha256 == plan.plan_sha256
    assert receipt.required_future_source_frames == 4
    assert receipt.episode_count == 2
    assert receipt.eligible_sample_count == 3
    assert len(receipt.source_windows_sha256) == 64
    assert receipt.to_dict()["artifact_sha256"] == receipt.artifact_sha256


def test_source_eligibility_receipt_rejects_one_incomplete_domain_key() -> None:
    class _TailDataset(_Dataset):
        def future_source_global_indices_by_key(
            self,
            sample_key: str,
            *,
            count: int,
        ) -> tuple[int, ...]:
            if sample_key == "sample-9":
                raise ContractError("crosses a raw episode reset")
            return super().future_source_global_indices_by_key(sample_key, count=count)

    plan = SimpleNamespace(
        plan_sha256="b" * 64,
        episodes=(SimpleNamespace(sample_keys=("sample-1", "sample-9")),),
    )
    with pytest.raises(ContractError, match="crosses a raw episode reset"):
        joint_data.audit_native_videomt_source_eligibility(_TailDataset(), plan)


def test_current_frame_preparation_reads_only_current_rgb_without_targets() -> None:
    class _Index:
        def __init__(self) -> None:
            self.calls: list[tuple[int, tuple[str, ...] | None, bool]] = []

        def validated_source_frame_arrays(
            self,
            global_index: int,
            *,
            fields: tuple[str, ...] | None = None,
            verify_relative_action: bool = True,
        ):
            self.calls.append((global_index, fields, verify_relative_action))
            return {"rgb_static": np.full((200, 200, 3), 17, dtype=np.uint8)}

    index = _Index()
    prepared = joint_data.prepare_native_videomt_current_frame(index, 31)

    assert index.calls == [(31, ("rgb_static",), False)]
    assert prepared.source_global_index == 31
    assert prepared.source_rgb.shape == (200, 200, 3)
    assert not prepared.source_rgb.flags.writeable
    assert len(prepared.source_rgb_sha256) == 64
    assert prepared.frames.model_input.shape[0] == 1


def test_prepare_joint_source_batch_pads_without_leaking_rng(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []
    prepared = iter((_prepared(8, 12, 1.0), _prepared(12, 8, 2.0)))

    def _materialize(_index, _sidecar, indices):
        calls.append(tuple(indices))
        return SimpleNamespace(rgb_static=(object(),) * 5, supervision=(object(),) * 5)

    def _prepare_host(_rgb):
        return PreparedVidEoMTFrames(
            model_input=torch.full((1, 3, 16, 16), 3.0),
            resized_rgb=(np.zeros((16, 16, 3), dtype=np.uint8),),
            original_sizes=((200, 200),),
            resized_sizes=((16, 16),),
            padded_size=(16, 16),
        )

    random.seed(9)
    np.random.seed(10)
    torch.manual_seed(11)
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(()))
    random.seed(9)
    np.random.seed(10)
    torch.manual_seed(11)
    monkeypatch.setattr(joint_data, "materialize_calvin_videomt_clip", _materialize)
    monkeypatch.setattr(
        joint_data,
        "prepare_calvin_videomt_training_clip",
        lambda _rgb, _supervision: next(prepared),
    )
    monkeypatch.setattr(joint_data, "prepare_rgb_frames", _prepare_host)

    batch = joint_data.prepare_native_videomt_source_batch(
        _Dataset(),
        object(),
        object(),
        sample_keys=("sample-1", "sample-8"),
        augmentation_seeds=(101, 202),
        device="cpu",
    )

    assert calls == [(1, 2, 3, 4, 5), (8, 9, 10, 11, 12)]
    assert batch.normalized_padded_rgb.shape == (2, 5, 3, 12, 12)
    assert batch.host_aligned_current_rgb.shape == (2, 1, 3, 16, 16)
    assert torch.equal(
        batch.host_aligned_current_rgb, torch.full_like(batch.host_aligned_current_rgb, 3.0)
    )
    assert batch.clip_targets[0]["masks"].shape == (1, 5, 12, 12)
    assert not batch.clip_targets[0]["valid_pixels"][:, 8:, :].any()
    assert not batch.clip_targets[1]["valid_pixels"][:, :, 8:].any()
    assert batch.target_count == 2
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(())) == expected_torch
