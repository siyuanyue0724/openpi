from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinEpisode, CalvinLanguageSegment
from picf_next.data.calvin_geometry_schema import sha256_file
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
    CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA,
    CALVIN_PHYSICAL_SUPERVISION_SCHEMA,
    CalvinPhysicalSupervisionShard,
    physical_supervision_manifest_payload,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_task_applicability import (
    calvin_visible_supervised_identity_support,
)
from tools.audit_calvin_physical_supervision import (
    _scan_full_tail,
    _validate_recomputed_manifest_summary,
)


def _index(tmp_path: Path) -> CalvinDatasetIndex:
    root = tmp_path / "training"
    root.mkdir(parents=True)
    np.save(
        root / "scene_info.npy",
        {"calvin_scene_D": np.asarray([10, 14], dtype=np.int64)},
        allow_pickle=True,
    )
    return CalvinDatasetIndex(
        split_root=root,
        dataset_id="fixture/calvin",
        dataset_revision="fixture-v1",
        control_hz=30,
        episodes=(CalvinEpisode(0, 10, 14),),
        segments=(CalvinLanguageSegment(0, 10, 12, "move", "move the block", 0),),
    )


def test_physical_sidecar_import_does_not_load_legacy_semantic_models() -> None:
    script = """
import json
import sys
import picf_next.data.calvin_physical_supervision_sidecar
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith((
        "picf_next.association",
        "picf_next.models",
        "picf_next.posterior",
    ))
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def _write_sidecar(
    root: Path,
    index: CalvinDatasetIndex,
    *,
    invalid_owner: bool = False,
    coverage: str = CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
    all_source_known_fraction: float = 1.0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "part00000_shard000000.npz"
    global_indices = (
        np.asarray([10, 11, 12], dtype=np.int64)
        if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
        else np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    )
    frame_count = len(global_indices)
    keys = np.asarray(
        ["movable/red", "part/table/button"] * frame_count,
        dtype=np.str_,
    )
    geometry = np.arange(frame_count * 6, dtype=np.float32).reshape(frame_count * 2, 3) / 10.0
    static_owner = np.zeros((frame_count, 200, 200), dtype=np.uint8)
    gripper_owner = np.zeros((frame_count, 84, 84), dtype=np.uint8)
    static_owner[:, 0, 0] = 1
    gripper_owner[:, 0, 0] = 2
    if invalid_owner:
        static_owner[-1, 1, 1] = 3
    arrays: dict[str, np.ndarray] = {
        "global_indices": global_indices,
        "source_state_sha256": np.asarray(["a" * 64] * frame_count),
        "frame_offsets": np.arange(0, 2 * frame_count + 1, 2, dtype=np.int64),
        "identity_keys": keys,
        "geometry": geometry,
        "geometry_variance": np.zeros_like(geometry),
        "geometry_supervised": np.ones_like(geometry, dtype=np.bool_),
    }
    for camera, owner in (("static", static_owner), ("gripper", gripper_owner)):
        arrays.update(
            {
                f"{camera}_source_rgb_sha256": np.asarray(["d" * 64] * frame_count),
                f"{camera}_source_depth_sha256": np.asarray(["e" * 64] * frame_count),
                f"{camera}_owner_index": owner,
                f"{camera}_rgb_mae": np.linspace(10.0, 12.0, frame_count, dtype=np.float32),
                f"{camera}_depth_mae_m": np.linspace(0.001, 0.003, frame_count, dtype=np.float32),
                f"{camera}_depth_p95_m": np.linspace(0.004, 0.006, frame_count, dtype=np.float32),
            }
        )
        if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            owner_supervised = np.zeros(owner.shape, dtype=np.bool_)
            known_pixels = int(owner.shape[1] * owner.shape[2] * all_source_known_fraction)
            owner_supervised.reshape(frame_count, -1)[:, :known_pixels] = True
            arrays[f"{camera}_owner_supervised"] = owner_supervised
            arrays[f"{camera}_depth_consistent_fraction"] = owner_supervised.mean(
                axis=(1, 2),
                dtype=np.float64,
            ).astype(np.float32)
    np.savez_compressed(shard_path, **arrays)
    shard = CalvinPhysicalSupervisionShard(
        path=shard_path.name,
        sha256=sha256_file(shard_path),
        first_global_index=10,
        last_global_index=int(global_indices[-1]),
        frame_count=frame_count,
        object_record_count=frame_count * 2,
    )
    summary = {
        f"maximum_{camera}_{metric}": float(arrays[f"{camera}_{metric}"].max())
        for camera in ("static", "gripper")
        for metric in ("rgb_mae", "depth_mae_m", "depth_p95_m")
    }
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        for camera in ("static", "gripper"):
            fractions = arrays[f"{camera}_depth_consistent_fraction"]
            summary[f"minimum_{camera}_depth_consistent_fraction"] = float(fractions.min())
            summary[f"p01_{camera}_depth_consistent_fraction"] = float(
                np.quantile(fractions, 0.01, method="linear")
            )
            summary[f"p05_{camera}_depth_consistent_fraction"] = float(
                np.quantile(fractions, 0.05, method="linear")
            )
            summary[f"p50_{camera}_depth_consistent_fraction"] = float(
                np.quantile(fractions, 0.50, method="linear")
            )
    manifest = physical_supervision_manifest_payload(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        split_name=index.split_root.name,
        scene_info_sha256=sha256_file(index.split_root / "scene_info.npy"),
        global_indices=global_indices,
        shards=(shard,),
        calibration_summary=summary,
        coverage=coverage,
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def test_unified_sidecar_returns_geometry_and_two_camera_owners(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _write_sidecar(root, index)

    provider = CalvinPhysicalSupervisionSidecar(root, index)
    frame = provider(0, 11)

    assert frame.identity_keys == ("movable/red", "part/table/button")
    assert frame.geometry.shape == (2, 3)
    assert tuple(camera.camera_name for camera in frame.cameras) == ("static", "gripper")
    assert frame.cameras[0].owner_index[0, 0] == 1
    assert frame.cameras[1].owner_index[0, 0] == 2
    assert not frame.cameras[0].owner_index.flags.writeable
    assert provider.geometry_frame(0, 11).identity_keys == frame.identity_keys
    assert provider.source_state_sha256(11) == "a" * 64
    support = calvin_visible_supervised_identity_support(frame)
    assert tuple(item.identity_key for item in support) == (
        "movable/red",
        "part/table/button",
    )
    assert tuple(item.total_pixel_count for item in support) == (1, 1)


def test_sidecar_accepts_hash_pinned_external_manifest_view(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    source_manifest = _write_sidecar(root, index)
    external_manifest = tmp_path / "identity-views" / "manifest.json"
    external_manifest.parent.mkdir()
    external_manifest.write_bytes(source_manifest.read_bytes())
    source_manifest.unlink()

    provider = CalvinPhysicalSupervisionSidecar(
        root,
        index,
        manifest_path=external_manifest,
        expected_manifest_sha256=sha256_file(external_manifest),
    )

    assert provider.manifest_sha256 == sha256_file(external_manifest)
    assert provider(0, 10).identity_keys == (
        "movable/red",
        "part/table/button",
    )


def test_sidecar_external_manifest_is_hash_pinned_and_not_followed(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    source_manifest = _write_sidecar(root, index)

    with pytest.raises(ContractError, match="content hash mismatch"):
        CalvinPhysicalSupervisionSidecar(
            root,
            index,
            manifest_path=source_manifest,
            expected_manifest_sha256="0" * 64,
        )

    with pytest.raises(ContractError, match="manifest SHA-256 is invalid"):
        CalvinPhysicalSupervisionSidecar(
            root,
            index,
            manifest_path=source_manifest,
            expected_manifest_sha256="not-a-digest",
        )

    linked_manifest = tmp_path / "linked-manifest.json"
    linked_manifest.symlink_to(source_manifest)
    with pytest.raises(ContractError, match="symlinks or unsafe components"):
        CalvinPhysicalSupervisionSidecar(
            root,
            index,
            manifest_path=linked_manifest,
            expected_manifest_sha256=sha256_file(source_manifest),
        )


def test_sidecar_rejects_two_manifest_sources(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    manifest_path = _write_sidecar(root, index)

    with pytest.raises(TypeError, match="manifest_path and manifest_bytes are exclusive"):
        CalvinPhysicalSupervisionSidecar(
            root,
            index,
            manifest_path=manifest_path,
            manifest_bytes=manifest_path.read_bytes(),
        )


def test_full_tail_audit_scans_real_validated_all_source_sidecar(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "all_source_sidecar"
    manifest_path = _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
        all_source_known_fraction=0.75,
    )
    provider = CalvinPhysicalSupervisionSidecar(root, index)
    manifest = json.loads(manifest_path.read_text())

    scan = _scan_full_tail(provider, manifest)
    errors = _validate_recomputed_manifest_summary(
        manifest["calibration_summary"],
        scan.recomputed_manifest_summary,
    )

    np.testing.assert_array_equal(scan.global_indices, np.arange(10, 15, dtype=np.int64))
    assert all(error == 0.0 for error in errors.values())
    assert scan.distributions["static"]["known_pixel_fraction"]["count"] == 5
    assert scan.distributions["static"]["known_pixel_fraction"]["p50"] == pytest.approx(0.75)


def test_language_sidecar_preserves_v2_manifest_and_rejects_source_access(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    manifest_path = _write_sidecar(root, index)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema"] == CALVIN_PHYSICAL_SUPERVISION_SCHEMA
    assert "coverage" not in manifest
    provider = CalvinPhysicalSupervisionSidecar(root, index)
    assert provider.coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
    with pytest.raises(ContractError, match="not declared for all source frames"):
        provider.source_frame(11)


def test_all_source_sidecar_covers_episode_frames_without_language_identity(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    root = tmp_path / "all_source_sidecar"
    manifest_path = _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema"] == CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_SCHEMA
    assert manifest["coverage"] == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
    assert "calibration_limits" not in manifest
    assert manifest["frame_diagnostics"]["aggregate_frame_metrics"] == "diagnostic-only"
    assert manifest["frame_count"] == 5
    provider = CalvinPhysicalSupervisionSidecar(root, index)
    assert provider.coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
    source_frame = provider.source_frame(14)
    assert source_frame.identity_keys == ("movable/red", "part/table/button")
    assert all(camera.owner_supervised.all() for camera in source_frame.cameras)
    assert all(camera.depth_consistent_fraction == 1.0 for camera in source_frame.cameras)
    assert provider(0, 11).identity_keys == ("movable/red", "part/table/button")
    with pytest.raises(ContractError, match="outside its source episodes"):
        provider.source_frame(15)


def test_all_source_sidecar_can_defer_unconsumed_shard_hashing(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "lazy_all_source_sidecar"
    _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    shard = root / "part00000_shard000000.npz"
    with shard.open("ab") as stream:
        stream.write(b"corrupt-after-manifest")

    provider = CalvinPhysicalSupervisionSidecar(
        root,
        index,
        eager_coverage_scan=False,
    )

    assert provider.coverage_validation == (
        "manifest-bound-lazy-consumed-shard-content-hash/v1"
    )
    with pytest.raises(ContractError, match="hash mismatch"):
        provider.source_frame(10)


def test_all_source_sidecar_keeps_low_coverage_pixels_unknown(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    root = tmp_path / "selective_sidecar"
    manifest_path = _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
        all_source_known_fraction=0.75,
    )
    manifest = json.loads(manifest_path.read_text())

    assert "calibration_limits" not in manifest
    provider = CalvinPhysicalSupervisionSidecar(root, index)
    frame = provider.source_frame(10)
    assert all(camera.owner_supervised.mean() == pytest.approx(0.75) for camera in frame.cameras)
    assert all(camera.depth_consistent_fraction == pytest.approx(0.75) for camera in frame.cameras)


def test_all_source_sidecar_rejects_obsolete_v4_semantics(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "obsolete_sidecar"
    manifest_path = _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "picf-next.calvin-physical-supervision-sidecar.v4"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="unsupported"):
        CalvinPhysicalSupervisionSidecar(root, index)


def test_all_source_frame_aggregate_extrema_remain_diagnostic(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "diagnostic_extrema"
    manifest_path = _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["calibration_summary"]["maximum_gripper_rgb_mae"] = 200.0
    manifest["calibration_summary"]["maximum_gripper_depth_mae_m"] = 1.0
    manifest_path.write_text(json.dumps(manifest))

    provider = CalvinPhysicalSupervisionSidecar(root, index)

    assert provider.source_frame(10).cameras[1].owner_supervised.all()


def test_all_source_sidecar_rejects_frame_diagnostic_contract_drift(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "diagnostic_contract_drift"
    manifest_path = _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["frame_diagnostics"]["aggregate_frame_metrics"] = "acceptance-gate"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="frame-diagnostic"):
        CalvinPhysicalSupervisionSidecar(root, index)


def test_all_source_sidecar_rejects_nonmonotone_coverage_diagnostics(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    root = tmp_path / "nonmonotone_sidecar"
    manifest_path = _write_sidecar(
        root,
        index,
        coverage=CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["calibration_summary"]["p01_static_depth_consistent_fraction"] = 0.5
    manifest["calibration_summary"]["minimum_static_depth_consistent_fraction"] = 0.75
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="monotone"):
        CalvinPhysicalSupervisionSidecar(root, index)


def test_unified_sidecar_rejects_unknown_owner(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _write_sidecar(root, index, invalid_owner=True)
    provider = CalvinPhysicalSupervisionSidecar(root, index)

    with pytest.raises(ContractError, match="unknown physical object"):
        provider(0, 12)


def test_unified_sidecar_rejects_calibration_above_manifest_limit(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    manifest_path = _write_sidecar(root, index)
    manifest = json.loads(manifest_path.read_text())
    manifest["calibration_summary"]["maximum_static_depth_p95_m"] = 0.1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="p95 depth"):
        CalvinPhysicalSupervisionSidecar(root, index)


def test_unified_sidecar_rejects_hash_corruption(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _write_sidecar(root, index)
    with (root / "part00000_shard000000.npz").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ContractError, match="hash mismatch"):
        CalvinPhysicalSupervisionSidecar(root, index)


def test_unified_sidecar_rejects_mutation_after_initial_validation(tmp_path: Path) -> None:
    index = _index(tmp_path)
    root = tmp_path / "sidecar"
    _write_sidecar(root, index)
    provider = CalvinPhysicalSupervisionSidecar(root, index)

    with (root / "part00000_shard000000.npz").open("ab") as handle:
        handle.write(b"post-validation-mutation")

    with pytest.raises(ContractError, match="hash mismatch"):
        provider(0, 11)
