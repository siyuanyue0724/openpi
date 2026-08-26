from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "audit_molmoact2_libero_full.py"
    spec = importlib.util.spec_from_file_location("picf_libero_full_audit_tool", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_tool()


def test_git_blob_hash_matches_git_object_contract(tmp_path: Path) -> None:
    path = tmp_path / "readme.txt"
    path.write_bytes(b"hello\n")
    assert audit._git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_revision_tree_loader_requires_exact_revision_filename(tmp_path: Path) -> None:
    tree_dir = tmp_path / ".cache" / "huggingface" / "trees"
    tree_dir.mkdir(parents=True)
    path = tree_dir / f"{audit.MOLMOACT2_LIBERO_REVISION}.json"
    path.write_text(json.dumps({"format_version": 1, "files": {"README.md": {"size": 1}}}))
    assert audit.load_revision_tree(tmp_path) == {"README.md": {"size": 1}}


class _Column:
    def __init__(self, values):
        self.values = values

    def to_pylist(self):
        return self.values

    def to_numpy(self, *, zero_copy_only: bool):
        assert not zero_copy_only
        return np.asarray(self.values)


class _Table:
    def __init__(self, columns):
        self.columns = columns
        self.num_rows = len(columns["index"])

    def __getitem__(self, key):
        return _Column(self.columns[key])


def _episode():
    return {
        "episode_index": 0,
        "task_index": 2,
        "dataset_from_index": 0,
        "dataset_to_index": 3,
        "length": 3,
    }


def _numeric_table():
    return _Table(
        {
            "observation.state": [[float(i)] * 8 for i in range(3)],
            "action": [[float(i)] * 7 for i in range(3)],
            "timestamp": [0.0, 0.1, 0.2],
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
            "index": [0, 1, 2],
            "task_index": [2, 2, 2],
        }
    )


def test_numeric_shard_validator_checks_all_alignment_axes() -> None:
    state, action, task, episodes = audit._validate_numeric_shard(
        _numeric_table(), [_episode()], fps=10
    )
    assert state.shape == (3, 8)
    assert action.shape == (3, 7)
    assert task.tolist() == [2, 2, 2]
    assert episodes == [0]

    broken = _numeric_table()
    broken.columns["task_index"][1] = 3
    with pytest.raises(ValueError, match="task order"):
        audit._validate_numeric_shard(broken, [_episode()], fps=10)


def test_task_representative_is_deterministic_lower_median() -> None:
    episodes = []
    for episode_index in range(4):
        episodes.append(
            {
                "episode_index": episode_index,
                "task_index": 0,
                "data/file_index": 0,
                "dataset_from_index": episode_index * 10,
                "dataset_to_index": episode_index * 10 + 10,
                "length": 10,
            }
        )
    actual = {episode_index: episode_index // 2 for episode_index in range(4)}
    result = audit._task_representatives(episodes, {0: "task"}, actual)
    assert result[0]["episode_index"] == 1
    assert result[0]["file_index"] == 0
    assert result[0]["phase_global_indices"] == [10, 14, 19]


def test_locator_overlay_preserves_claimed_and_actual_file_indices() -> None:
    episodes = [
        {
            "episode_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
        },
        {
            "episode_index": 1,
            "data/chunk_index": 0,
            "data/file_index": 0,
        },
    ]
    overlay = audit.build_episode_locator_overlay(episodes, {0: 0, 1: 1})
    assert overlay[0]["mismatch"] is False
    assert overlay[1] == {
        "episode_index": 1,
        "claimed_chunk_index": 0,
        "claimed_file_index": 0,
        "actual_chunk_index": 0,
        "actual_file_index": 1,
        "mismatch": True,
    }


def test_schema_storage_variant_allows_only_the_two_equivalent_list_encodings() -> None:
    pa = pytest.importorskip("pyarrow")
    image = pa.field("image", pa.struct([pa.field("bytes", pa.binary())]))
    canonical = pa.schema(
        [
            image,
            pa.field("observation.state", pa.list_(pa.float32(), 8)),
            pa.field("action", pa.list_(pa.float32(), 7)),
            pa.field("episode_index", pa.int64()),
        ]
    )
    variable = pa.schema(
        [
            image,
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("action", pa.list_(pa.float32())),
            pa.field("episode_index", pa.int64()),
        ]
    )
    assert audit._schema_storage_variant(canonical, canonical) == "fixed_size_list"
    assert audit._schema_storage_variant(variable, canonical) == "variable_list"

    wrong = variable.set(2, pa.field("action", pa.list_(pa.float64())))
    with pytest.raises(ValueError, match="unsupported"):
        audit._schema_storage_variant(wrong, canonical)
