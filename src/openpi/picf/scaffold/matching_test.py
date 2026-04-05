import numpy as np

from openpi.picf.scaffold.matching import match_supports


def test_match_supports_is_one_to_one_and_lexicographic() -> None:
    current_x = np.array([[0.0, 0.0, 0.0], [0.003, 0.0, 0.0]], dtype=np.float32)
    current_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1))
    current_active = np.array([True, True])
    current_eid = np.zeros((2, 3), dtype=np.float32)

    prev_x = np.array([[0.0005, 0.0, 0.0], [0.0035, 0.0, 0.0]], dtype=np.float32)
    prev_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1))
    prev_slots = np.array([7, 2], dtype=np.int32)
    prev_eid = np.zeros((2, 3), dtype=np.float32)

    result = match_supports(
        current_x,
        current_n,
        current_active,
        current_eid,
        prev_x,
        prev_n,
        prev_slots,
        prev_eid,
        tau_p=0.01,
        tau_n=0.8,
        rgb_enabled=False,
        lambda_app_match=0.0,
        epsilon_app=1e-6,
    )

    assert result.matched_mask.tolist() == [True, True]
    assert result.pred_idx.tolist() == [7, 2]
    assert result.match_ratio == 1.0
