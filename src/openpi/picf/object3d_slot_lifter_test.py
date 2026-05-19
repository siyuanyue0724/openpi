import torch

from openpi.picf.object3d_slot_lifter import Object3DSlotLifter
from openpi.picf.object3d_slot_lifter import make_object3d_point_features


def test_object3d_slot_lifter_outputs_point_slot_contract() -> None:
    torch.manual_seed(0)
    xyz = torch.randn(2, 32, 3)
    rgb = torch.rand(2, 32, 3)
    view_ids = torch.cat([torch.zeros(2, 16), torch.ones(2, 16)], dim=1).long()
    feats = make_object3d_point_features(xyz, rgb, view_ids, num_views=2)
    model = Object3DSlotLifter(input_dim=feats.shape[-1], slot_dim=32, num_slots=4, num_iterations=2)

    out = model(feats, xyz)

    assert out.slots.shape == (2, 4, 32)
    assert out.point_slot_weights.shape == (2, 32, 5)
    assert out.object_point_priors.shape == (2, 4, 32)
    assert out.centers.shape == (2, 4, 3)
    assert out.covariance_diag.shape == (2, 4, 3)
    assert out.objectness.shape == (2, 4)
    assert out.background_weight.shape == (2, 32)
    torch.testing.assert_close(out.point_slot_weights.sum(dim=-1), torch.ones(2, 32), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(out.centers).all()
    assert torch.isfinite(out.objectness).all()


def test_object3d_point_features_include_view_identity() -> None:
    xyz = torch.zeros(1, 3, 3)
    rgb = torch.zeros(1, 3, 3)
    view_ids = torch.tensor([[0, 1, 1]])
    feats = make_object3d_point_features(xyz, rgb, view_ids, num_views=2)

    assert feats.shape == (1, 3, 9)
    torch.testing.assert_close(feats[0, 0, 6:8], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(feats[0, 1, 6:8], torch.tensor([0.0, 1.0]))

