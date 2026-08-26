from __future__ import annotations

import hashlib
import json

import pytest

from picf_next.contracts import ContractError
from tools.probe_qwen3vl_grounding_baseline import (
    INPUT_SCHEMA,
    _bbox,
    _load_probe_report,
    _model_hashes,
)


def test_qwen_baseline_requires_schema_v3_record_bank(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"schema": INPUT_SCHEMA, "records": [{"global_index": 1}]}),
        encoding="utf-8",
    )
    assert _load_probe_report(report)["schema"] == INPUT_SCHEMA

    report.write_text(
        json.dumps({"schema": "picf-next.lingbot-native-vl-grounding-g0.v1", "records": [{}]}),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="schema-v3"):
        _load_probe_report(report)


def test_qwen_baseline_hashes_every_indexed_weight_shard(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "part-1.safetensors").write_bytes(b"first")
    (tmp_path / "part-2.safetensors").write_bytes(b"second")
    index = {
        "weight_map": {
            "model.a": "part-1.safetensors",
            "model.b": "part-2.safetensors",
            "model.c": "part-2.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    hashes = _model_hashes(tmp_path)
    assert set(hashes) == {
        "config.json",
        "model.safetensors.index.json",
        "part-1.safetensors",
        "part-2.safetensors",
    }
    assert hashes["part-1.safetensors"] == hashlib.sha256(b"first").hexdigest()
    assert hashes["part-2.safetensors"] == hashlib.sha256(b"second").hexdigest()


def test_qwen_baseline_bbox_parser_fails_closed() -> None:
    assert _bbox([1, 2, 3, 4], "box") == (1, 2, 3, 4)
    with pytest.raises(ContractError, match="four-integer"):
        _bbox([1, 2, 3], "box")
    with pytest.raises(ContractError, match="four-integer"):
        _bbox([1, 2, 3, True], "box")
