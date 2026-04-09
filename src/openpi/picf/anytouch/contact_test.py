import torch

from openpi.picf.anytouch.contact import contact_normal
from openpi.picf.anytouch.contact import explicit_contact_observation
from openpi.picf.anytouch.contact import pose6d


def test_explicit_contact_observation_drops_missing_terms() -> None:
    assert explicit_contact_observation(
        force_vec=torch.tensor([2.0, 0.0, 0.0]),
        indent_depth_m=None,
        tactile_pressure=None,
        tau_force_n=1.0,
        tau_indent_m=5e-4,
        tau_tactile_pressure=0.1,
    )
    assert not explicit_contact_observation(
        force_vec=torch.tensor([0.1, 0.0, 0.0]),
        indent_depth_m=None,
        tactile_pressure=None,
        tau_force_n=1.0,
        tau_indent_m=5e-4,
        tau_tactile_pressure=0.1,
    )


def test_contact_normal_and_pose6d_shapes() -> None:
    pose = torch.eye(4)
    force = torch.tensor([2.0, 0.0, 0.0])

    normal = contact_normal(pose, force, epsilon_force=1e-6, pose_normal_available=False)
    se3 = pose6d(pose)

    assert normal.shape == (3,)
    assert se3.shape == (6,)
    assert torch.allclose(normal, torch.tensor([1.0, 0.0, 0.0]))
