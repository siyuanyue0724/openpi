from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from picf_next.lingbot_native.current_grid_cache import CurrentGridCacheRecord
from picf_next.lingbot_native.predictive_cache import (
    LingBotPredictiveTargetCache,
    PredictiveObjectCacheRecord,
    pool_dino_object_summaries,
)
from tools import audit_lingbot_dino_teacher_causality as teacher_causality
from tools.audit_lingbot_dino_teacher_causality import (
    audit_selected_teacher_pairs,
    select_predictive_records,
)
from tools.build_lingbot_calvin_predictive_cache import (
    OfficialLingBotDinoVideoExtractor,
    _VerifiedStaticFrame,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _SyntheticPredictiveCache(LingBotPredictiveTargetCache):
    def __init__(
        self,
        records: tuple[PredictiveObjectCacheRecord, ...],
        *,
        horizons: tuple[int, ...] | None = None,
    ) -> None:
        self.contract = SimpleNamespace(
            expected_record_count=len(records),
            horizons=(
                tuple(sorted({record.horizon for record in records}))
                if horizons is None
                else horizons
            ),
        )
        self._records = records

    def iter_records(self):
        yield from self._records


def _selection_record(source: int, horizon: int) -> PredictiveObjectCacheRecord:
    return PredictiveObjectCacheRecord(
        source_global_index=source,
        target_global_index=source + horizon,
        horizon=horizon,
        source_rgb_sha256=_sha(f"selection-source-{source}"),
        target_rgb_sha256=_sha(f"selection-target-{source + horizon}"),
        identity_keys=("object/a",),
        features=np.ones((1, 2), dtype=np.float16),
        importance=np.ones((1,), dtype=np.float32),
    )


def test_teacher_causality_selection_preserves_every_horizon(monkeypatch) -> None:
    records = (
        _selection_record(0, 1),
        _selection_record(1, 1),
        _selection_record(100, 2),
    )
    cache = _SyntheticPredictiveCache(records)
    monkeypatch.setattr(
        teacher_causality,
        "_priority",
        lambda record: record.source_global_index,
    )

    selected, scanned = select_predictive_records(cache, maximum_records=2)

    assert scanned == 3
    assert {(record.source_global_index, record.horizon) for record in selected} == {
        (0, 1),
        (100, 2),
    }


def test_teacher_causality_selection_allows_unreachable_declared_horizon(monkeypatch) -> None:
    records = (
        _selection_record(0, 1),
        _selection_record(1, 1),
        _selection_record(100, 2),
    )
    cache = _SyntheticPredictiveCache(records, horizons=(1, 2, 64))
    monkeypatch.setattr(
        teacher_causality,
        "_priority",
        lambda record: record.source_global_index,
    )

    selected, scanned = select_predictive_records(cache, maximum_records=2)

    assert scanned == 3
    assert {record.horizon for record in selected} == {1, 2}


def _frame(index: int) -> _VerifiedStaticFrame:
    owners = np.zeros((16, 16), dtype=np.uint8)
    owners[:, :8] = 1
    owners[:, 8:] = 2
    return _VerifiedStaticFrame(
        global_index=index,
        rgb=np.full((16, 16, 3), index, dtype=np.uint8),
        rgb_sha256=_sha(f"rgb-{index}"),
        physical=SimpleNamespace(identity_keys=("object/a", "object/b")),
        camera=SimpleNamespace(
            owner_index=owners,
            owner_supervised=np.ones((16, 16), dtype=np.bool_),
        ),
    )


def _patch(offset: float) -> torch.Tensor:
    token = torch.arange(256, dtype=torch.float32)[:, None]
    channel = torch.arange(1024, dtype=torch.float32)[None, :] / 1024
    return token + channel + offset * channel.square()


def _record(
    source: _VerifiedStaticFrame,
    target: _VerifiedStaticFrame,
    *,
    patch_index: int | None = None,
) -> PredictiveObjectCacheRecord:
    patch = _patch(float(target.global_index if patch_index is None else patch_index))
    features, importance = pool_dino_object_summaries(
        patch,
        owner_index=target.camera.owner_index,
        owner_supervised=target.camera.owner_supervised,
        identity_keys=target.physical.identity_keys,
        minimum_visible_fraction=0.0,
        input_size=256,
    )
    return PredictiveObjectCacheRecord(
        source_global_index=source.global_index,
        target_global_index=target.global_index,
        horizon=target.global_index - source.global_index,
        source_rgb_sha256=source.rgb_sha256,
        target_rgb_sha256=target.rgb_sha256,
        identity_keys=target.physical.identity_keys,
        features=features,
        importance=importance,
    )


def _current_record(
    source: _VerifiedStaticFrame,
    *,
    patch_index: int | None = None,
) -> CurrentGridCacheRecord:
    return CurrentGridCacheRecord(
        source_global_index=source.global_index,
        source_rgb_sha256=source.rgb_sha256,
        features=_patch(float(source.global_index if patch_index is None else patch_index))
        .half()
        .numpy(),
    )


def _current_record_loader(
    frames: dict[int, _VerifiedStaticFrame],
    *,
    corrupt_source: int | None = None,
):
    records = {
        index: _current_record(
            frame,
            patch_index=index + 1 if index == corrupt_source else None,
        )
        for index, frame in frames.items()
    }
    return records.get


class _ExactExtractor(OfficialLingBotDinoVideoExtractor):
    def __init__(self, *, corrupt_current: bool = False, copy_future: bool = False) -> None:
        self.corrupt_current = corrupt_current
        self.copy_future = copy_future
        self.observed_paired_fps: list[torch.Tensor] = []

    def paired(
        self,
        current_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
        *,
        effective_fps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.observed_paired_fps.append(effective_fps.detach().cpu().clone())
        future = torch.stack(tuple(_patch(float(value)) for value in future_rgb[:, 0, 0, 0, 0]))
        current = torch.stack(tuple(_patch(float(value)) for value in current_rgb[:, 0, 0, 0, 0]))
        if self.copy_future:
            future = current.clone()
        if self.corrupt_current:
            current = current.clone()
            current[0, 0, 0] += 1
        return future, current

    def current(
        self,
        current_rgb: torch.Tensor,
        *,
        effective_fps: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if effective_fps is not None:
            assert isinstance(effective_fps, torch.Tensor)
            torch.testing.assert_close(effective_fps.cpu(), self.observed_paired_fps[-1])
        return torch.stack(tuple(_patch(float(value)) for value in current_rgb[:, 0, 0, 0, 0]))


def test_teacher_causality_audit_replays_current_and_future_exactly() -> None:
    frames = {index: _frame(index) for index in (10, 11, 20, 21)}
    records = (_record(frames[10], frames[11]), _record(frames[20], frames[21]))

    result = audit_selected_teacher_pairs(
        records,
        frame_for=frames.__getitem__,
        current_record_for=_current_record_loader(frames),
        extractor=_ExactExtractor(),
        configured_horizons=(1, 64),
        input_size=256,
        minimum_visible_fraction=0.0,
        source_fps=30.0,
        batch_size=1,
    )

    assert result["status"] == "PASS"
    assert result["current_patch_mismatch_count"] == 0
    assert result["current_cache_patch_mismatch_count"] == 0
    assert result["future_feature_mismatch_count"] == 0
    assert result["future_importance_mismatch_count"] == 0
    assert result["same_call_temporal_pretraining_readiness"] == "PASS"
    assert result["same_call_supported_pair_count"] == 4
    assert result["sampled_horizon_record_counts"] == {"1": 2, "64": 0}


def test_teacher_causality_audit_rejects_future_conditioned_current_patch() -> None:
    frames = {index: _frame(index) for index in (10, 11, 20, 21)}
    records = (_record(frames[10], frames[11]), _record(frames[20], frames[21]))

    result = audit_selected_teacher_pairs(
        records,
        frame_for=frames.__getitem__,
        current_record_for=_current_record_loader(frames),
        extractor=_ExactExtractor(corrupt_current=True),
        configured_horizons=(1,),
        input_size=256,
        minimum_visible_fraction=0.0,
        source_fps=30.0,
        batch_size=2,
    )

    assert result["status"] == "FAIL"
    assert result["current_patch_mismatch_count"] == 1
    assert result["maximum_current_patch_absolute_error"] == 1.0


def test_teacher_causality_audit_uses_per_horizon_effective_fps() -> None:
    frames = {index: _frame(index) for index in (10, 11, 20, 22)}
    records = (_record(frames[10], frames[11]), _record(frames[20], frames[22]))
    extractor = _ExactExtractor()

    result = audit_selected_teacher_pairs(
        records,
        frame_for=frames.__getitem__,
        current_record_for=_current_record_loader(frames),
        extractor=extractor,
        configured_horizons=(1, 2),
        input_size=256,
        minimum_visible_fraction=0.0,
        source_fps=30.0,
        batch_size=2,
    )

    assert result["status"] == "PASS"
    assert len(extractor.observed_paired_fps) == 1
    torch.testing.assert_close(
        extractor.observed_paired_fps[0],
        torch.tensor([30.0, 15.0]),
    )


def test_teacher_causality_audit_rejects_exact_current_copy_future() -> None:
    frames = {index: _frame(index) for index in (10, 11, 20, 21)}
    records = (
        _record(frames[10], frames[11], patch_index=10),
        _record(frames[20], frames[21], patch_index=20),
    )

    result = audit_selected_teacher_pairs(
        records,
        frame_for=frames.__getitem__,
        current_record_for=_current_record_loader(frames),
        extractor=_ExactExtractor(copy_future=True),
        configured_horizons=(1,),
        input_size=256,
        minimum_visible_fraction=0.0,
        source_fps=30.0,
        batch_size=2,
    )

    assert result["current_patch_mismatch_count"] == 0
    assert result["future_feature_mismatch_count"] == 0
    assert result["same_call_temporal_pretraining_readiness"] == "FAIL"
    assert result["same_call_temporal_pretraining_readiness_failures"] == [
        "no_measurable_current_to_future_target_change"
    ]
    assert result["status"] == "FAIL"


def test_teacher_causality_audit_rejects_corrupt_current_cache_patch() -> None:
    frames = {index: _frame(index) for index in (10, 11, 20, 21)}
    records = (_record(frames[10], frames[11]), _record(frames[20], frames[21]))

    result = audit_selected_teacher_pairs(
        records,
        frame_for=frames.__getitem__,
        current_record_for=_current_record_loader(frames, corrupt_source=10),
        extractor=_ExactExtractor(),
        configured_horizons=(1,),
        input_size=256,
        minimum_visible_fraction=0.0,
        source_fps=30.0,
        batch_size=2,
    )

    assert result["current_patch_mismatch_count"] == 0
    assert result["current_cache_patch_mismatch_count"] > 0
    assert result["maximum_current_cache_patch_absolute_error"] > 0
    assert result["status"] == "FAIL"
