import numpy as np

from openpi.picf.contracts import RuntimeMeta
from openpi.picf.contracts import ScaffoldDebugMetrics
from openpi.picf.contracts import SupportScaffoldState
from openpi.picf.frame_context import PointFrameContext
from openpi.picf.posterior.config import PosteriorConfig
from openpi.picf.posterior.point_expert import build_point_expert
from openpi.picf.scaffold.pipeline import DeterministicScaffoldConfig


def test_point_expert_gates_low_resolution_supports() -> None:
    posterior_config = PosteriorConfig()
    scaffold_config = DeterministicScaffoldConfig()
    pi = np.zeros((2, 12), dtype=np.float32)
    pi[0, :8] = 1.0 / 8.0
    pi[1, 8:12] = 1.0 / 4.0
    state = SupportScaffoldState(
        pi_geom=pi,
        x=np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]], dtype=np.float32),
        n=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
        r=np.array([0.01, 0.01], dtype=np.float32),
        omega=np.array([1.0, 1.0], dtype=np.float32),
        active_mask=np.array([True, True]),
        pred_idx=np.array([-1, -1], dtype=np.int32),
        matched_mask=np.array([False, False]),
        birth_mask=np.array([True, True]),
        e_id=np.zeros((2, 3), dtype=np.float32),
        s_qry=np.zeros((2, 16), dtype=np.float32),
        G_t=np.eye(4, dtype=np.float32),
        step_id=0,
        segment_id=0,
        runtime_meta=RuntimeMeta(v_pc_scaf=True),
        debug=ScaffoldDebugMetrics(
            num_points_local=10,
            num_active=2,
            num_birth=2,
            match_ratio=0.0,
            mean_radius=0.01,
            normal_fallback_ratio=0.0,
            empty_support_ratio=0.0,
            hold_triggered=False,
            hold_reason=None,
            reindex_failure_rate=0.0,
            normal_flip_ratio=0.0,
            fresh_scaffold=True,
        ),
    )
    points = np.concatenate(
        [
            np.stack([np.array([0.0015 * i, 0.0, 0.0], dtype=np.float32) for i in range(8)], axis=0),
            np.stack([np.array([0.08 + 0.01 * i, 0.0, 0.0], dtype=np.float32) for i in range(4)], axis=0),
        ],
        axis=0,
    )
    frame_context = PointFrameContext(
        points_local=points,
        normals_local=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (12, 1)),
        colors=np.zeros((12, 3), dtype=np.float32),
        local_mask=np.ones((12,), dtype=bool),
        world_to_local=np.eye(4, dtype=np.float32),
        G_t=np.eye(4, dtype=np.float32),
    )

    point = build_point_expert(
        posterior_config=posterior_config,
        scaffold_config=scaffold_config,
        scaffold_state=state,
        frame_context=frame_context,
    )

    assert bool(point.gate[0])
    assert not bool(point.gate[1])
    assert point.anchor_count.tolist() == [8, 3]
