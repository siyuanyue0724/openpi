from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA,
)
from picf_next.data.calvin_simulator_geometry import CalvinSceneRange
from picf_next.lingbot_native.artifact_capacity import (
    LINGBOT_CALVIN_ARTIFACT_CAPACITY_SCHEMA,
    LingBotCalvinArtifactCapacity,
    PhysicalSidecarStorageSample,
)
from tools import plan_lingbot_calvin_artifact_capacity as capacity_tool

_ROOT = Path(__file__).resolve().parents[2]


def _projection(**overrides: object) -> LingBotCalvinArtifactCapacity:
    values: dict[str, object] = {
        "free_bytes": 300 * 1024**3,
        "checkpoint_reserve_bytes": 110 * 1024**3,
        "minimum_headroom_bytes": 20 * 1024**3,
        "physical_total_frames": 1_795_045,
        "required_scenes": ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C"),
        "physical_samples": (
            PhysicalSidecarStorageSample(
                1_000,
                8_000_000,
                9_000.0,
                ("calvin_scene_A",),
                12,
                40,
            ),
            PhysicalSidecarStorageSample(
                1_000,
                10_000_000,
                10_000.0,
                ("calvin_scene_B",),
                10,
                32,
            ),
            PhysicalSidecarStorageSample(
                1_000,
                9_000_000,
                9_500.0,
                ("calvin_scene_C",),
                11,
                36,
            ),
        ),
        "current_grid_record_count": 60_000,
        "predictive_record_count": 3_000,
    }
    values.update(overrides)
    return LingBotCalvinArtifactCapacity(**values)  # type: ignore[arg-type]


def test_capacity_projection_uses_worst_stratum_and_conservative_cache_bounds() -> None:
    projection = _projection()
    report = projection.as_dict()

    assert report["schema"] == LINGBOT_CALVIN_ARTIFACT_CAPACITY_SCHEMA
    assert report["status"] == "PASS"
    assert projection.projected_physical_bytes == 22_438_062_500
    assert projection.projected_current_grid_bytes > 60_000 * 256 * 1024 * 2
    assert projection.projected_predictive_bytes > 3_000 * 12 * 1024 * 2
    assert projection.required_bytes < projection.free_bytes
    assert projection.residual_bytes == projection.free_bytes - projection.required_bytes


def test_capacity_projection_fails_closed_when_required_bytes_exceed_free_space() -> None:
    projection = _projection(free_bytes=120 * 1024**3)

    assert projection.status == "FAIL"
    assert projection.residual_bytes < 0


@pytest.mark.parametrize(
    "overrides,match",
    (
        (
            {
                "physical_samples": (
                    PhysicalSidecarStorageSample(
                        1_500,
                        1_000,
                        1.0,
                        ("calvin_scene_A",),
                        2,
                        10,
                    ),
                    PhysicalSidecarStorageSample(
                        1_500,
                        1_000,
                        1.0,
                        ("calvin_scene_A",),
                        2,
                        10,
                    ),
                )
            },
            "at least three",
        ),
        (
            {
                "physical_samples": (
                    PhysicalSidecarStorageSample(
                        999,
                        1_000,
                        2.0,
                        ("calvin_scene_A",),
                        2,
                        10,
                    ),
                    PhysicalSidecarStorageSample(
                        999,
                        1_000,
                        2.0,
                        ("calvin_scene_B",),
                        2,
                        10,
                    ),
                    PhysicalSidecarStorageSample(
                        999,
                        1_000,
                        2.0,
                        ("calvin_scene_C",),
                        2,
                        10,
                    ),
                )
            },
            "at least 3000",
        ),
        (
            {
                "required_scenes": (
                    "calvin_scene_A",
                    "calvin_scene_B",
                    "calvin_scene_C",
                    "calvin_scene_D",
                )
            },
            "every dataset scene",
        ),
        ({"physical_safety_factor": 0.99}, "at least one"),
    ),
)
def test_capacity_projection_rejects_weak_evidence(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ContractError, match=match):
        _projection(**overrides)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_probe_partition(
    root: Path,
    *,
    partition_index: int,
    global_index: int,
    identity_keys: tuple[str, ...],
) -> None:
    shard = root / f"part{partition_index:05d}_shard000000.npz"
    np.savez_compressed(
        shard,
        global_indices=np.asarray([global_index], dtype=np.int64),
        frame_offsets=np.asarray([0, len(identity_keys)], dtype=np.int64),
        identity_keys=np.asarray(identity_keys, dtype=np.str_),
    )
    indices = np.asarray([global_index], dtype=np.int64)
    manifest = {
        "schema": CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA,
        "coverage": "all_source_frames",
        "dataset_id": "calvin",
        "dataset_revision": "fixture",
        "split_name": "training",
        "partition_count": 3,
        "partition_index": partition_index,
        "frame_count": 1,
        "global_indices_sha256": hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
        "shards": [
            {
                "path": shard.name,
                "sha256": _sha256(shard),
                "frame_count": 1,
            }
        ],
    }
    (root / f"partition_{partition_index:05d}.json").write_text(
        json.dumps(manifest),
        encoding="ascii",
    )


def test_capacity_probe_loader_hashes_strata_and_requires_all_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "probe"
    root.mkdir()
    _write_probe_partition(
        root,
        partition_index=0,
        global_index=10,
        identity_keys=("movable/red_block", "part/table/top"),
    )
    _write_probe_partition(
        root,
        partition_index=1,
        global_index=110,
        identity_keys=("movable/blue_block",),
    )
    _write_probe_partition(
        root,
        partition_index=2,
        global_index=210,
        identity_keys=("movable/pink_block",),
    )
    ranges = (
        CalvinSceneRange("calvin_scene_A", 0, 99),
        CalvinSceneRange("calvin_scene_B", 100, 199),
        CalvinSceneRange("calvin_scene_C", 200, 299),
    )
    monkeypatch.setattr(
        capacity_tool,
        "load_calvin_scene_ranges",
        lambda _root, *, dataset_manifest: ranges,
    )
    index = SimpleNamespace(
        dataset_id="calvin",
        dataset_revision="fixture",
        split_root=tmp_path / "training",
        episodes=(
            SimpleNamespace(start=10, end=10),
            SimpleNamespace(start=110, end=110),
            SimpleNamespace(start=210, end=210),
        ),
    )
    index.split_root.mkdir()

    required, samples, evidence, evidence_sha256 = capacity_tool._load_physical_samples(
        root,
        index=index,  # type: ignore[arg-type]
        dataset_manifest=object(),  # type: ignore[arg-type]
        required_partition_indices=(0, 1, 2),
    )

    assert required == ("calvin_scene_A", "calvin_scene_B", "calvin_scene_C")
    assert tuple(sample.scenes for sample in samples) == (
        ("calvin_scene_A",),
        ("calvin_scene_B",),
        ("calvin_scene_C",),
    )
    assert max(sample.maximum_object_count for sample in samples) == 2
    assert max(sample.maximum_identity_key_characters for sample in samples) == len(
        "movable/blue_block"
    )
    assert len(evidence) == 3
    assert all(value["manifest_sha256"] for value in evidence)
    assert len(evidence_sha256) == 64

    obsolete_manifest_path = root / "partition_00001.json"
    obsolete_manifest = json.loads(obsolete_manifest_path.read_text(encoding="ascii"))
    obsolete_manifest["schema"] = "picf-next.calvin-physical-supervision-partition.v3"
    obsolete_manifest_path.write_text(json.dumps(obsolete_manifest), encoding="ascii")
    with pytest.raises(ContractError, match="provenance differs"):
        capacity_tool._load_physical_samples(
            root,
            index=index,  # type: ignore[arg-type]
            dataset_manifest=object(),  # type: ignore[arg-type]
            required_partition_indices=(0, 1, 2),
        )
    _write_probe_partition(
        root,
        partition_index=1,
        global_index=110,
        identity_keys=("movable/blue_block",),
    )

    _write_probe_partition(
        root,
        partition_index=1,
        global_index=111,
        identity_keys=("movable/blue_block",),
    )
    with pytest.raises(ContractError, match="coverage differs"):
        capacity_tool._load_physical_samples(
            root,
            index=index,  # type: ignore[arg-type]
            dataset_manifest=object(),  # type: ignore[arg-type]
            required_partition_indices=(0, 1, 2),
        )
    _write_probe_partition(
        root,
        partition_index=1,
        global_index=110,
        identity_keys=("movable/blue_block",),
    )
    shard = root / "part00001_shard000000.npz"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ContractError, match="missing or corrupt"):
        capacity_tool._load_physical_samples(
            root,
            index=index,  # type: ignore[arg-type]
            dataset_manifest=object(),  # type: ignore[arg-type]
            required_partition_indices=(0, 1, 2),
        )


def test_capacity_probe_loader_rejects_nonfrozen_strata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "probe"
    root.mkdir()
    for partition_index, global_index in enumerate((10, 110, 210)):
        _write_probe_partition(
            root,
            partition_index=partition_index,
            global_index=global_index,
            identity_keys=(f"movable/block_{partition_index}",),
        )
    monkeypatch.setattr(
        capacity_tool,
        "load_calvin_scene_ranges",
        lambda _root, *, dataset_manifest: (
            CalvinSceneRange("calvin_scene_A", 0, 99),
            CalvinSceneRange("calvin_scene_B", 100, 199),
            CalvinSceneRange("calvin_scene_C", 200, 299),
        ),
    )
    index = SimpleNamespace(
        dataset_id="calvin",
        dataset_revision="fixture",
        split_root=tmp_path / "training",
        episodes=(
            SimpleNamespace(start=10, end=10),
            SimpleNamespace(start=110, end=110),
            SimpleNamespace(start=210, end=210),
        ),
    )
    index.split_root.mkdir()

    with pytest.raises(ContractError, match="frozen strata"):
        capacity_tool._load_physical_samples(
            root,
            index=index,  # type: ignore[arg-type]
            dataset_manifest=object(),  # type: ignore[arg-type]
            required_partition_indices=(0, 1, 3),
        )


def test_capacity_planner_is_in_the_exact_torch_type_contract() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "tools/plan_lingbot_calvin_artifact_capacity.py" in workflow
