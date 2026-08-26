from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from picf_next.lingbot_native.official_config import official_lingbot_data_config

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
LINGBOT_SOURCE = ROOT / "references/source_checkouts/lingbot-vla-v2-adr74"
if not (LINGBOT_SOURCE / "lingbotvla/data/vla_data/utils.py").is_file():
    pytest.skip("optional pinned LingBot ADR-74 source checkout is absent", allow_module_level=True)
sys.path.insert(0, str(LINGBOT_SOURCE))
try:
    from lingbotvla.data.vla_data.utils import FeatureTransform
except ModuleNotFoundError as error:  # The minimal unit environment omits LingBot-only packages.
    pytest.skip(f"official LingBot data runtime is unavailable: {error}", allow_module_level=True)


def test_official_feature_transform_preserves_calvin_active_dimensions() -> None:
    data_config = official_lingbot_data_config(
        json.loads((ROOT / "configs/lingbot/calvin_data.json").read_text())
    )
    transform = FeatureTransform(
        str(ROOT / "configs/lingbot/calvin_robot.yaml"),
        data_config,
        SimpleNamespace(),
        processor=None,
        do_nomalize=False,
        chunk_size=4,
        return_item_befor_padding=True,
    )
    raw = {
        "observation.state.lingbot": torch.arange(55, dtype=torch.float32),
        "action.lingbot": torch.arange(4 * 55, dtype=torch.float32).reshape(4, 55),
        "action.lingbot_is_pad": torch.tensor([False, False, True, True]),
        "observation.images.camera_top": torch.zeros(3, 200, 200),
        "observation.images.camera_wrist_left": torch.zeros(3, 84, 84),
        "task": "move the blue block",
    }
    mapped = transform.apply(raw)
    assert mapped["observation.state.arm.position"].shape == (7,)
    assert mapped["observation.state.end.position"].shape == (6,)
    assert mapped["observation.state.effector.position"].shape == (1,)
    assert mapped["action.end.position"].shape == (4, 6)
    assert mapped["action.effector.position"].shape == (4, 1)
    assert mapped["action_is_pad"].tolist() == [False, False, True, True]
    assert "action.arm.position" not in mapped
    assert "observation.images.camera_wrist_right" not in mapped


def test_official_feature_transform_consumes_the_generated_norm_stats_schema(
    tmp_path: Path,
) -> None:
    data_config = official_lingbot_data_config(
        json.loads((ROOT / "configs/lingbot/calvin_data.json").read_text())
    )
    norm_stats = {
        "norm_stats": {
            "observation.state.arm.position": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": list(range(7)),
                "q99": [value + 10 for value in range(7)],
            },
            "observation.state.end.position": {
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
                "q01": list(range(14, 20)),
                "q99": [value + 10 for value in range(14, 20)],
            },
            "observation.state.effector.position": {
                "mean": [0.0],
                "std": [1.0],
                "q01": [28.0],
                "q99": [29.0],
            },
            "action.end.position": {
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
                "q01": list(range(14, 20)),
                "q99": [value + 10 for value in range(14, 20)],
            },
            "action.effector.position": {
                "mean": [0.0],
                "std": [1.0],
                "q01": [28.0],
                "q99": [29.0],
            },
        }
    }
    stats_path = tmp_path / "norm.json"
    stats_path.write_text(json.dumps(norm_stats))
    transform = FeatureTransform(
        str(ROOT / "configs/lingbot/calvin_robot.yaml"),
        data_config,
        SimpleNamespace(),
        processor=None,
        do_nomalize=True,
        norm_stats_path=str(stats_path),
        chunk_size=4,
        return_item_befor_padding=True,
    )
    raw = {
        "observation.state.lingbot": torch.arange(55, dtype=torch.float32),
        "action.lingbot": torch.arange(4 * 55, dtype=torch.float32).reshape(4, 55),
        "action.lingbot_is_pad": torch.tensor([False, False, True, True]),
        "observation.images.camera_top": torch.zeros(3, 200, 200),
        "observation.images.camera_wrist_left": torch.zeros(3, 84, 84),
        "task": "move the blue block",
    }
    mapped = transform.apply(raw)
    assert mapped["observation.state.arm.position"].dtype == torch.float64
    torch.testing.assert_close(
        mapped["observation.state.arm.position"],
        torch.full((7,), -1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        mapped["observation.state.end.position"],
        torch.full((6,), -1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        mapped["action.end.position"][0],
        torch.full((6,), -1.0, dtype=torch.float64),
    )
    # The gripper/effector channel is intentionally identity-normalized.
    torch.testing.assert_close(
        mapped["observation.state.effector.position"],
        torch.tensor([28.0]),
    )
