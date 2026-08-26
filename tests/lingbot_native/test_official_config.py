from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from picf_next.lingbot_native.official_config import official_lingbot_data_config

ROOT = Path(__file__).resolve().parents[2]


def test_official_data_config_serializes_only_the_released_parser_boundary() -> None:
    raw = json.loads((ROOT / "configs/lingbot/calvin_data.json").read_text())
    original = json.loads(json.dumps(raw))

    resolved = official_lingbot_data_config(raw)

    assert raw == original
    assert ast.literal_eval(resolved.joints[0]) == {"arm.position": 14}
    assert ast.literal_eval(resolved.norm_type[0]) == {"arm.position": "bounds_99_woclip"}
    assert resolved.cameras == raw["cameras"]


def test_official_data_config_accepts_already_serialized_declarations() -> None:
    resolved = official_lingbot_data_config(
        {
            "joints": ["{'arm.position': 14}"],
            "norm_type": ["{'arm.position': 'meanstd'}"],
            "cameras": ["camera_top"],
        }
    )
    assert resolved.joints == ["{'arm.position': 14}"]
    assert resolved.norm_type == ["{'arm.position': 'meanstd'}"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("joints", [{"arm.position": True}], "width"),
        ("norm_type", [{"unknown.position": "identity"}], "undeclared"),
        ("cameras", [], "cameras"),
    ),
)
def test_official_data_config_fails_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    raw = {
        "joints": [{"arm.position": 14}],
        "norm_type": [{"arm.position": "identity"}],
        "cameras": ["camera_top"],
    }
    raw[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        official_lingbot_data_config(raw)
