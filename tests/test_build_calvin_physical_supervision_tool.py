from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import tools.build_calvin_physical_supervision as physical_tool
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
    CalvinPhysicalSupervisionShard,
)


def _args(
    tmp_path: Path,
    *,
    partition_count: int = 2,
    partition_index: int = 1,
    finalize_only: bool = False,
    defer_finalize: bool = True,
    resume_completed_partition: bool = False,
) -> argparse.Namespace:
    split = tmp_path / "training"
    split.mkdir(exist_ok=True)
    return argparse.Namespace(
        split_root=split,
        calvin_env_root=None,
        output_dir=tmp_path / "sidecar",
        dataset_id="calvin-test",
        dataset_revision="revision-test",
        dataset_manifest=tmp_path / "dataset-manifest.json",
        partition_count=partition_count,
        partition_index=partition_index,
        shard_size=256,
        progress_every=100,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
        finalize_only=finalize_only,
        defer_finalize=defer_finalize,
        resume_completed_partition=resume_completed_partition,
    )


def _install_main_fakes(
    monkeypatch: pytest.MonkeyPatch,
    args: argparse.Namespace,
) -> CalvinDatasetIndex:
    manifest = SimpleNamespace(record_for=lambda relative_path: SimpleNamespace(sha256="a" * 64))
    index = cast(
        CalvinDatasetIndex,
        SimpleNamespace(
            dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision,
            split_root=args.split_root.resolve(),
            episodes=(SimpleNamespace(start=0, end=3),),
            segments=(SimpleNamespace(start=1, end=2),),
        ),
    )
    monkeypatch.setattr(physical_tool, "_parse_args", lambda: args)
    monkeypatch.setattr(physical_tool, "load_dataset_file_manifest", lambda path: manifest)
    monkeypatch.setattr(
        physical_tool,
        "validate_dataset_runtime_binding",
        lambda *positional, **keywords: None,
    )
    monkeypatch.setattr(
        physical_tool.CalvinDatasetIndex,
        "load",
        lambda *positional, **keywords: index,
    )
    monkeypatch.setattr(
        physical_tool,
        "load_calvin_scene_ranges",
        lambda *positional, **keywords: ("scene",),
    )
    monkeypatch.setattr(
        physical_tool,
        "scene_for_global_index",
        lambda scene_ranges, global_index: "scene",
    )
    return index


def test_required_indices_are_sorted_unique_without_python_object_sets() -> None:
    index = cast(
        CalvinDatasetIndex,
        SimpleNamespace(
            segments=(
                SimpleNamespace(start=4, end=6),
                SimpleNamespace(start=2, end=5),
            ),
            episodes=(
                SimpleNamespace(start=8, end=9),
                SimpleNamespace(start=6, end=8),
            ),
        ),
    )

    assert np.array_equal(
        physical_tool._required_indices(
            index,
            CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
        ),
        np.arange(2, 7, dtype=np.int64),
    )
    assert np.array_equal(
        physical_tool._required_indices(
            index,
            CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
        ),
        np.arange(6, 10, dtype=np.int64),
    )


def test_all_source_calibration_retains_low_coverage_as_selective_unknown() -> None:
    record = physical_tool._CameraRecord(
        source_rgb_sha256="a" * 64,
        source_depth_sha256="b" * 64,
        owner_index=np.zeros((2, 2), dtype=np.uint8),
        owner_supervised=np.asarray([[True, False], [False, False]], dtype=np.bool_),
        rgb_mae=20.0,
        depth_mae_m=0.02,
        depth_p95_m=0.2,
        depth_consistent_fraction=0.25,
    )

    physical_tool._validate_calibration(
        "gripper",
        record,
        7,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )


def test_all_source_frame_metrics_are_diagnostic_after_pixel_selectivity() -> None:
    record = physical_tool._CameraRecord(
        source_rgb_sha256="a" * 64,
        source_depth_sha256="b" * 64,
        owner_index=np.zeros((2, 2), dtype=np.uint8),
        owner_supervised=np.zeros((2, 2), dtype=np.bool_),
        rgb_mae=20.0,
        depth_mae_m=0.03,
        depth_p95_m=0.2,
        depth_consistent_fraction=0.0,
    )

    physical_tool._validate_calibration(
        "gripper",
        record,
        7,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )


def test_language_frame_calibration_still_rejects_gross_misalignment() -> None:
    record = physical_tool._CameraRecord(
        source_rgb_sha256="a" * 64,
        source_depth_sha256="b" * 64,
        owner_index=np.zeros((2, 2), dtype=np.uint8),
        owner_supervised=np.ones((2, 2), dtype=np.bool_),
        rgb_mae=20.0,
        depth_mae_m=0.03,
        depth_p95_m=0.2,
        depth_consistent_fraction=1.0,
    )

    with pytest.raises(ContractError, match="depth_mae_m"):
        physical_tool._validate_calibration(
            "gripper",
            record,
            7,
            coverage=CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
        )


def test_atomic_json_uses_the_shared_durable_exclusive_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, bytes]] = []

    def publish(path: Path, payload: bytes) -> Path:
        calls.append((path, payload))
        return path

    monkeypatch.setattr(physical_tool, "write_bytes_durable_exclusive", publish)
    destination = tmp_path / "manifest.json"

    physical_tool._atomic_json(destination, {"value": 3})

    assert calls == [
        (
            destination,
            b'{\n  "value": 3\n}\n',
        )
    ]


def test_resume_completed_partition_verifies_every_shard_before_skipping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _args(tmp_path, resume_completed_partition=True)
    index = _install_main_fakes(monkeypatch, args)
    args.output_dir.mkdir()
    (args.output_dir / "partition_00001.json").write_text("{}")
    calls: list[dict[str, object]] = []

    def load_partition(*positional: object, **keywords: object) -> tuple[()]:
        calls.append(dict(keywords))
        return ()

    monkeypatch.setattr(physical_tool, "_load_partition_shards", load_partition)
    monkeypatch.setattr(
        physical_tool,
        "_extract_partition",
        lambda *positional, **keywords: pytest.fail("completed partition was re-extracted"),
    )
    monkeypatch.setattr(
        physical_tool,
        "_finalize",
        lambda *positional, **keywords: pytest.fail("deferred finalization ran"),
    )

    physical_tool.main()

    assert len(calls) == 1
    assert calls[0]["index"] is index
    expected_indices = calls[0]["expected_indices"]
    assert isinstance(expected_indices, np.ndarray)
    assert np.array_equal(expected_indices, np.arange(4, dtype=np.int64))
    assert '"status": "verified_complete"' in capsys.readouterr().out


def test_existing_partition_is_not_silently_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    _install_main_fakes(monkeypatch, args)
    args.output_dir.mkdir()
    partition = args.output_dir / "partition_00001.json"
    partition.write_text("{}")

    with pytest.raises(FileExistsError, match="resume-completed-partition"):
        physical_tool.main()


def test_completed_aggregate_manifest_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(physical_tool, "_parse_args", lambda: args)
    args.output_dir.mkdir()
    aggregate = args.output_dir / "manifest.json"
    aggregate.write_text("{}")

    with pytest.raises(FileExistsError, match="completed physical artifacts are immutable"):
        physical_tool.main()


def test_dangling_partition_manifest_is_rejected_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, resume_completed_partition=True)
    _install_main_fakes(monkeypatch, args)
    args.output_dir.mkdir()
    partition = args.output_dir / "partition_00001.json"
    partition.symlink_to(args.output_dir / "absent.json")

    with pytest.raises(ContractError, match="not a regular file"):
        physical_tool.main()


def test_finalize_only_and_defer_finalize_are_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, finalize_only=True, defer_finalize=True)
    monkeypatch.setattr(physical_tool, "_parse_args", lambda: args)

    with pytest.raises(ValueError, match="mutually exclusive"):
        physical_tool.main()


def test_finalization_rejects_stale_partition_or_shard_members(
    tmp_path: Path,
) -> None:
    shards = (
        CalvinPhysicalSupervisionShard(
            path="part00000_shard000000.npz",
            sha256="a" * 64,
            first_global_index=0,
            last_global_index=1,
            frame_count=2,
            object_record_count=3,
        ),
    )
    (tmp_path / "partition_00000.json").write_text("{}")
    (tmp_path / shards[0].path).write_bytes(b"expected")

    physical_tool._validate_output_membership(
        tmp_path,
        partition_count=1,
        shards=shards,
    )

    stale = tmp_path / "part00001_shard000000.npz"
    stale.write_bytes(b"stale")
    with pytest.raises(ContractError, match="stale or missing shard"):
        physical_tool._validate_output_membership(
            tmp_path,
            partition_count=1,
            shards=shards,
        )
    stale.unlink()

    (tmp_path / "partition_00001.json").write_text("{}")
    with pytest.raises(ContractError, match="stale or missing partition"):
        physical_tool._validate_output_membership(
            tmp_path,
            partition_count=1,
            shards=shards,
        )


def test_partition_loader_rejects_a_symlink_manifest(
    tmp_path: Path,
) -> None:
    target = tmp_path / "payload.json"
    target.write_text("{}")
    (tmp_path / "partition_00000.json").symlink_to(target)
    index = cast(CalvinDatasetIndex, SimpleNamespace())

    with pytest.raises(ContractError, match="invalid CALVIN physical partition manifest"):
        physical_tool._load_partition_shards(
            tmp_path,
            partition_count=1,
            partition_index=0,
            index=index,
            expected_indices=np.asarray([0], dtype=np.int64),
            scene_info_sha256="a" * 64,
            coverage=CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
        )


def test_calibration_summary_opens_each_shard_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = []
    for shard_index, offset in enumerate((0.0, 0.5)):
        path = tmp_path / f"part00000_shard{shard_index:06d}.npz"
        arrays = {}
        for camera_index, camera in enumerate(("static", "gripper")):
            base = offset + camera_index * 0.1
            arrays.update(
                {
                    f"{camera}_rgb_mae": np.asarray([base + 0.01], dtype=np.float32),
                    f"{camera}_depth_mae_m": np.asarray([base + 0.02], dtype=np.float32),
                    f"{camera}_depth_p95_m": np.asarray([base + 0.03], dtype=np.float32),
                    f"{camera}_depth_consistent_fraction": np.asarray(
                        [1.0 - base - 0.04],
                        dtype=np.float32,
                    ),
                }
            )
        np.savez(path, **arrays)
        shards.append(
            CalvinPhysicalSupervisionShard(
                path=path.name,
                sha256="a" * 64,
                first_global_index=shard_index,
                last_global_index=shard_index,
                frame_count=1,
                object_record_count=1,
            )
        )
    load = np.load
    opened = 0

    def counted_load(*positional: Any, **keywords: Any) -> Any:
        nonlocal opened
        opened += 1
        return load(*positional, **keywords)

    monkeypatch.setattr(physical_tool.np, "load", counted_load)

    summary = physical_tool._calibration_summary(
        tmp_path,
        tuple(shards),
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )

    assert opened == len(shards)
    assert summary["maximum_static_rgb_mae"] == pytest.approx(0.51)
    assert summary["maximum_gripper_depth_p95_m"] == pytest.approx(0.63)
    assert summary["minimum_static_depth_consistent_fraction"] == pytest.approx(0.46)
    assert summary["p01_static_depth_consistent_fraction"] == pytest.approx(0.465)
    assert summary["p05_static_depth_consistent_fraction"] == pytest.approx(0.485)
    assert summary["p50_static_depth_consistent_fraction"] == pytest.approx(0.71)
    assert summary["minimum_gripper_depth_consistent_fraction"] == pytest.approx(0.36)
    assert summary["p01_gripper_depth_consistent_fraction"] == pytest.approx(0.365)
    assert summary["p05_gripper_depth_consistent_fraction"] == pytest.approx(0.385)
    assert summary["p50_gripper_depth_consistent_fraction"] == pytest.approx(0.61)


def test_single_pass_calibration_preserves_committed_sidecar_values() -> None:
    root = Path(__file__).parents[1] / "data/calvin_physical_supervision_v1_p2"
    if not root.exists():
        pytest.skip("optional locally generated CALVIN calibration sidecar is absent")
    manifest = root / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    shards = tuple(CalvinPhysicalSupervisionShard.from_dict(value) for value in payload["shards"])

    measured = physical_tool._calibration_summary(
        root,
        shards,
        coverage=CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
    )

    assert measured == payload["calibration_summary"]


def test_finalize_only_reuses_the_single_materialized_index_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(
        tmp_path,
        finalize_only=True,
        defer_finalize=False,
        partition_index=0,
    )
    index = _install_main_fakes(monkeypatch, args)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def finalize(*positional: object, **keywords: object) -> None:
        calls.append((positional, dict(keywords)))

    monkeypatch.setattr(physical_tool, "_finalize", finalize)

    physical_tool.main()

    assert len(calls) == 1
    assert calls[0][0] == (index,)
    expected_indices = calls[0][1]["expected_indices"]
    assert isinstance(expected_indices, np.ndarray)
    assert np.array_equal(
        expected_indices,
        np.arange(4, dtype=np.int64),
    )
