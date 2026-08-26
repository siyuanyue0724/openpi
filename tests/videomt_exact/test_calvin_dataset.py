from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.videomt_exact.calvin_dataset import (
    HashBoundCalvinFrameStore,
    build_calvin_videomt_split_plan,
)


class _Store:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.global_indices = values
        self.manifest_sha256 = "a" * 64


def _record(segment: int, step: int, phase: str) -> dict[str, object]:
    return {
        "split": "training",
        "segment_index": segment,
        "step": step,
        "phase": phase,
    }


def test_split_merges_overlaps_before_task_blind_holdout(tmp_path: Path) -> None:
    intervals = ((0, 4), (10, 14), (12, 16), (20, 24), (30, 34))
    records = []
    for segment, (start, end) in enumerate(intervals):
        records.extend(
            (
                _record(segment, start, "start"),
                _record(segment, (start + end) // 2, "mid"),
                _record(segment, end, "end"),
            )
        )
    manifest = {
        "format": "picf-next.calvin-visible-instance-golden.v1",
        "runtime_input": False,
        "task_used_for_instance_selection": False,
        "failures": [],
        "records": records,
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    covered = tuple((*range(0, 5), *range(10, 17), *range(20, 25), *range(30, 35)))

    plan = build_calvin_videomt_split_plan(
        golden_manifest_path=path,
        store=_Store(covered),  # type: ignore[arg-type]
        clip_length=3,
    )

    assert [(value.start, value.end) for value in plan.components] == [
        (0, 4),
        (10, 16),
        (20, 24),
        (30, 34),
    ]
    assert plan.components[2].split == "heldout"
    train_frames = {value for window in plan.train_windows for value in window}
    heldout_frames = {value for window in plan.heldout_windows for value in window}
    assert train_frames.isdisjoint(heldout_frames)
    assert heldout_frames == set(range(20, 25))


def _minimal_store(
    *,
    primary: Path,
    overlay: Path,
    expected_rgb_sha256: str,
) -> HashBoundCalvinFrameStore:
    store = object.__new__(HashBoundCalvinFrameStore)
    store.source_split_root = primary
    store.source_overlay_root = overlay
    store.global_indices = (7,)
    store._records = {  # type: ignore[attr-defined]
        7: SimpleNamespace(camera_values={"static": {"source_rgb_sha256": expected_rgb_sha256}})
    }
    store._rgb_cache_capacity = 2
    store._rgb_cache = OrderedDict()  # type: ignore[attr-defined]
    store._resolved_source_paths = {}  # type: ignore[attr-defined]
    return store


def test_corrupt_primary_uses_only_hash_valid_overlay(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    overlay = tmp_path / "overlay"
    primary.mkdir()
    overlay.mkdir()
    (primary / "episode_0000007.npz").write_bytes(b"\0" * 128)
    rgb = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
    np.savez(overlay / "episode_0000007.npz", rgb_static=rgb)
    store = _minimal_store(
        primary=primary,
        overlay=overlay,
        expected_rgb_sha256=source_array_sha256("rgb_static", rgb),
    )

    loaded = store._source_rgb(7)  # noqa: SLF001 - tests the fail-closed repair boundary.
    receipt = store.audit_source_rgb()

    np.testing.assert_array_equal(loaded, rgb)
    assert receipt["overlay_frame_count"] == 1
    assert receipt["overlay_files"][0]["global_index"] == 7  # type: ignore[index]


def test_overlay_cannot_bypass_rgb_content_hash(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    overlay = tmp_path / "overlay"
    primary.mkdir()
    overlay.mkdir()
    (primary / "episode_0000007.npz").write_bytes(b"\0" * 128)
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    np.savez(overlay / "episode_0000007.npz", rgb_static=rgb)
    store = _minimal_store(
        primary=primary,
        overlay=overlay,
        expected_rgb_sha256="f" * 64,
    )

    with pytest.raises(ContractError, match="no hash-valid CALVIN source RGB"):
        store._source_rgb(7)  # noqa: SLF001 - tests the fail-closed repair boundary.
