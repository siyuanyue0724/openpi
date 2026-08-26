from __future__ import annotations

import torch

from picf_next.lingbot_native.physical_relations import (
    PhysicalEntityReadout,
    PhysicalRelationSurfaceInput,
)


def _surface(
    hidden: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> PhysicalRelationSurfaceInput:
    batch, tokens, _width = hidden.shape
    if valid is None:
        valid = torch.ones(batch, tokens, dtype=torch.bool, device=hidden.device)
    canonical = torch.arange(tokens, device=hidden.device).unsqueeze(0).expand(batch, -1)
    canonical = torch.where(valid, canonical, torch.full_like(canonical, -1))
    return PhysicalRelationSurfaceInput(
        name="vjepa",
        geometry_kind="image_grid",
        target_kind="calvin_vjepa21_visible_owner_v1",
        layout="vjepa21.calvin.static-gripper.24x24.v1",
        sensor_hidden=hidden,
        sensor_valid=valid,
        canonical_token_ids=canonical,
    )


def _read(
    readout: PhysicalEntityReadout,
    rows: torch.Tensor,
    surface_hidden: torch.Tensor,
    *,
    surface_valid: torch.Tensor | None = None,
):
    return readout(
        posterior_rows=rows,
        sensor_hidden=torch.tensor(
            [[[0.3, -0.4, 0.7, 0.1], [-0.2, 0.9, 0.5, -0.6]]],
            dtype=rows.dtype,
            device=rows.device,
        ),
        sensor_valid=torch.ones(1, 2, dtype=torch.bool, device=rows.device),
        structural_sensor_valid=torch.ones(
            1,
            2,
            dtype=torch.bool,
            device=rows.device,
        ),
        relation_surfaces=(_surface(surface_hidden, valid=surface_valid),),
    )


def test_native_surface_uses_the_same_exchangeable_row_gauge() -> None:
    torch.manual_seed(401)
    readout = PhysicalEntityReadout(4)
    rows = torch.randn(1, 2, 4)
    surface_hidden = torch.randn(1, 5, 4)

    factual = _read(readout, rows, surface_hidden).surface("vjepa")
    permutation = torch.tensor([1, 0])
    permuted = _read(readout, rows[:, permutation], surface_hidden).surface("vjepa")

    torch.testing.assert_close(
        permuted.support_logits,
        factual.support_logits.index_select(-1, permutation),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        permuted.object_probability,
        factual.object_probability.index_select(-1, permutation),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        permuted.context_probability,
        factual.context_probability,
        rtol=0,
        atol=1e-6,
    )


def test_native_surface_loss_reaches_rows_source_and_one_shared_readout() -> None:
    torch.manual_seed(402)
    readout = PhysicalEntityReadout(4)
    rows = torch.randn(1, 2, 4, requires_grad=True)
    surface_hidden = torch.randn(1, 6, 4, requires_grad=True)
    surface = _read(readout, rows, surface_hidden).surface("vjepa")
    owner = torch.tensor([[0, 0, 1, 1, 2, 2]])

    loss = -surface.ownership_log_probability.gather(-1, owner.unsqueeze(-1)).mean()
    loss.backward()

    for gradient in (
        rows.grad,
        surface_hidden.grad,
        readout.projection.weight.grad,
        readout.no_object.grad,
        readout.temperature_parameter.grad,
    ):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert set(readout.state_dict()) == {
        "existence_projection.bias",
        "existence_projection.weight",
        "no_object",
        "projection.weight",
        "temperature_parameter",
    }


def test_missing_native_surface_is_numerically_and_gradient_absent() -> None:
    torch.manual_seed(403)
    readout = PhysicalEntityReadout(4)
    rows = torch.randn(1, 2, 4, requires_grad=True)
    surface_hidden = torch.randn(1, 5, 4, requires_grad=True)
    valid = torch.zeros(1, 5, dtype=torch.bool)

    surface = _read(
        readout,
        rows,
        surface_hidden,
        surface_valid=valid,
    ).surface("vjepa")

    assert not surface.support_logits.any()
    assert not surface.ownership.any()
    assert torch.isfinite(surface.support_logits).all()
    assert surface.sensor_valid.tolist() == [[False, False, False, False, False]]


def _dense_semantic_fixture(
    *,
    batch: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.full((batch, 2, 24, 24), 2, dtype=torch.long)
    labels[:, :, 2:13, 2:12] = 0
    labels[:, :, 11:22, 13:23] = 1
    prototypes = torch.eye(4)[:3]
    generator = torch.Generator().manual_seed(seed)
    hidden = prototypes[labels] + 0.08 * torch.randn(
        batch,
        2,
        24,
        24,
        4,
        generator=generator,
    )
    return hidden.reshape(batch, 2 * 24 * 24, 4), labels.reshape(batch, -1)


def test_shared_native_affinity_fits_heldout_dense_evidence_without_a_decoder() -> None:
    torch.manual_seed(404)
    readout = PhysicalEntityReadout(4)
    rows = torch.nn.Parameter(torch.randn(1, 2, 4))
    optimizer = torch.optim.Adam((*readout.parameters(), rows), lr=0.05)
    train_hidden, train_labels = _dense_semantic_fixture(batch=4, seed=405)
    test_hidden, test_labels = _dense_semantic_fixture(batch=3, seed=406)

    def predict(hidden: torch.Tensor):
        batch = hidden.shape[0]
        return readout(
            posterior_rows=rows.expand(batch, -1, -1),
            sensor_hidden=hidden[:, :2],
            sensor_valid=torch.ones(batch, 2, dtype=torch.bool),
            structural_sensor_valid=torch.ones(batch, 2, dtype=torch.bool),
            relation_surfaces=(_surface(hidden),),
        ).surface("vjepa")

    with torch.no_grad():
        initial_accuracy = (
            (predict(test_hidden).ownership.argmax(dim=-1) == test_labels).float().mean()
        )
    for _step in range(80):
        optimizer.zero_grad(set_to_none=True)
        surface = predict(train_hidden)
        loss = -surface.ownership_log_probability.gather(
            -1,
            train_labels.unsqueeze(-1),
        ).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        heldout_accuracy = (
            (predict(test_hidden).ownership.argmax(dim=-1) == test_labels).float().mean()
        )

    assert heldout_accuracy >= 0.95
    assert heldout_accuracy - initial_accuracy >= 0.25
