from __future__ import annotations

from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
utils = pytest.importorskip("lerobot.datasets.utils")
if not hasattr(utils, "load_nested_dataset"):
    pytest.skip(
        "load_nested_dataset belongs to the isolated LingBot LeRobot 0.4.3 runtime",
        allow_module_level=True,
    )


def _write_shard(path: Path, *, episode_index: int, global_start: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [episode_index, episode_index],
                "index": [global_start, global_start + 1],
                "value": [float(global_start), float(global_start + 1)],
            }
        ),
        path,
    )


def _write_vector_shard(
    path: Path,
    *,
    episode_index: int,
    fixed_size: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_type = pa.list_(pa.float32(), 8) if fixed_size else pa.list_(pa.float32())
    action_type = pa.list_(pa.float32(), 7) if fixed_size else pa.list_(pa.float32())
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([episode_index], type=pa.int64()),
                "index": pa.array([episode_index], type=pa.int64()),
                "observation.state": pa.array([[0.0] * 8], type=state_type),
                "action": pa.array([[0.0] * 7], type=action_type),
            }
        ),
        path,
    )


def test_upstream_nested_loader_reads_each_shard_once_and_filters_by_episode(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_shard(
        data_root / "chunk-000" / "file-000.parquet",
        episode_index=0,
        global_start=0,
    )
    _write_shard(
        data_root / "chunk-000" / "file-001.parquet",
        episode_index=1,
        global_start=2,
    )

    complete = utils.load_nested_dataset(data_root)
    selected = utils.load_nested_dataset(data_root, episodes=[1])

    assert len(complete) == 4
    assert complete["index"] == [0, 1, 2, 3]
    assert len(selected) == 2
    assert selected["episode_index"] == [1, 1]
    assert selected["index"] == [2, 3]


def test_upstream_nested_loader_unifies_known_fixed_and_variable_list_storage(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_vector_shard(
        data_root / "chunk-000" / "file-000.parquet",
        episode_index=0,
        fixed_size=True,
    )
    _write_vector_shard(
        data_root / "chunk-000" / "file-001.parquet",
        episode_index=1,
        fixed_size=False,
    )

    complete = utils.load_nested_dataset(data_root)

    assert len(complete) == 2
    assert complete["episode_index"] == [0, 1]
    assert complete["observation.state"] == [[0.0] * 8, [0.0] * 8]
    assert complete["action"] == [[0.0] * 7, [0.0] * 7]
