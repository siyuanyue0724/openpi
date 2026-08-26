from __future__ import annotations

import numpy as np
import pytest

from picf_next.contracts import ContractError
from tools.render_calvin_tactile_audit import _task_labels, split_tactile_rgb


def test_split_tactile_rgb_preserves_released_sensor_order() -> None:
    image = np.zeros((2, 3, 6), dtype=np.uint8)
    image[..., :3] = 11
    image[..., 3:] = 22

    left_digit, right_digit = split_tactile_rgb(image)

    assert (left_digit == 11).all()
    assert (right_digit == 22).all()
    with pytest.raises(ContractError, match="H-by-W-by-6"):
        split_tactile_rgb(np.zeros((2, 3, 3), dtype=np.uint8))


def test_task_labels_match_interval_inclusively(tmp_path) -> None:
    path = tmp_path / "auto_lang_ann.npy"
    np.save(
        path,
        {
            "info": {"indx": [(10, 20)]},
            "language": {
                "task": ["turn_on_led"],
                "ann": ["toggle the button to turn on the led"],
            },
        },
        allow_pickle=True,
    )

    labels = _task_labels(path, (9, 10, 20, 21))

    assert labels[9][0] == "unannotated"
    assert labels[10] == ("turn_on_led", "toggle the button to turn on the led")
    assert labels[20] == labels[10]
    assert labels[21][0] == "unannotated"
