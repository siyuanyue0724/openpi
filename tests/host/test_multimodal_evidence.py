from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
evidence_module = pytest.importorskip("picf_next.models.evidence")

ModalityProjectionSpec = evidence_module.ModalityProjectionSpec
MultimodalBindingProjector = evidence_module.MultimodalBindingProjector
NativeTokenBank = evidence_module.NativeTokenBank


def _projector() -> MultimodalBindingProjector:
    torch.manual_seed(131)
    return MultimodalBindingProjector(
        (
            ModalityProjectionSpec("vjepa", token_dim=7, geometry_dim=3),
            ModalityProjectionSpec(
                "anytouch",
                token_dim=5,
                require_single_active_group=True,
            ),
        ),
        binding_dim=11,
    ).eval()


def _banks():
    torch.manual_seed(137)
    vision_valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    touch_valid = torch.tensor([[True, True, True], [False, False, False]])
    vision_tokens = torch.randn(2, 4, 7) * vision_valid.unsqueeze(-1)
    vision_geometry = torch.randn(2, 4, 3) * vision_valid.unsqueeze(-1)
    touch_tokens = torch.randn(2, 3, 5) * touch_valid.unsqueeze(-1)
    touch_groups = torch.tensor([[12, 12, 12], [-1, -1, -1]], dtype=torch.long)
    return (
        NativeTokenBank("vjepa", vision_tokens, vision_valid, vision_geometry),
        NativeTokenBank(
            "anytouch",
            touch_tokens,
            touch_valid,
            group_id=touch_groups,
        ),
    )


def test_binding_projection_retains_every_native_token_and_bank() -> None:
    projector = _projector()
    banks = _banks()
    output = projector(banks)

    assert output.total_tokens == 7
    assert output.binding_features.shape == (2, 7, 11)
    assert output.token_valid.shape == (2, 7)
    assert torch.equal(output.current_measurement_valid, output.token_valid)
    assert output.token_group_id.shape == (2, 7)
    assert tuple((span.modality, span.start, span.stop) for span in output.spans) == (
        ("vjepa", 0, 4),
        ("anytouch", 4, 7),
    )
    assert all(
        actual is expected for actual, expected in zip(output.native_banks, banks, strict=True)
    )
    assert torch.equal(output.native_banks[0].tokens, banks[0].tokens)
    assert torch.equal(output.native_banks[1].tokens, banks[1].tokens)
    assert output.token_valid.sum() == sum(bank.valid.sum() for bank in banks)
    assert torch.equal(
        output.binding_features[~output.token_valid],
        torch.zeros_like(output.binding_features[~output.token_valid]),
    )


def test_historical_tokens_remain_native_context_but_not_current_measurements() -> None:
    projector = _projector()
    valid = torch.ones(1, 4, dtype=torch.bool)
    current = torch.tensor([[False, False, True, True]])
    timestamps = torch.tensor([[0.0, 0.1, 0.2, 0.2]], dtype=torch.float32)
    bank = NativeTokenBank(
        "vjepa",
        torch.randn(1, 4, 7),
        valid,
        torch.randn(1, 4, 3),
        timestamps=timestamps,
        current_measurement_valid=current,
    )

    output = projector((bank,))
    current_features, current_valid, current_groups = output.current_discovery_inputs()

    assert output.native_banks[0] is bank
    assert output.token_valid.all()
    assert torch.equal(output.current_measurement_valid, current)
    assert torch.equal(current_valid, current)
    assert torch.count_nonzero(current_features[:, :2]) == 0
    assert torch.equal(current_features[:, 2:], output.binding_features[:, 2:])
    assert torch.equal(current_groups, torch.full_like(current_groups, -1))


def test_historical_role_requires_timestamps_and_newest_current_slice() -> None:
    projector = _projector()
    valid = torch.ones(1, 3, dtype=torch.bool)
    current = torch.tensor([[False, False, True]])
    tokens = torch.randn(1, 3, 7)
    geometry = torch.randn(1, 3, 3)

    with pytest.raises(ValueError, match="auditable timestamps"):
        projector(
            (NativeTokenBank("vjepa", tokens, valid, geometry, current_measurement_valid=current),)
        )
    with pytest.raises(ValueError, match="newest evidence timestamp"):
        projector(
            (
                NativeTokenBank(
                    "vjepa",
                    tokens,
                    valid,
                    geometry,
                    timestamps=torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32),
                    current_measurement_valid=torch.tensor([[True, False, False]]),
                ),
            )
        )


def test_permutation_within_modality_preserves_corresponding_projection() -> None:
    projector = _projector()
    banks = _banks()
    expected = projector(banks)
    permutation = torch.tensor([2, 0, 3, 1])
    vision = banks[0]
    permuted_banks = (
        NativeTokenBank(
            vision.modality,
            vision.tokens[:, permutation],
            vision.valid[:, permutation],
            vision.geometry[:, permutation],
        ),
        banks[1],
    )
    actual = projector(permuted_banks)

    torch.testing.assert_close(
        actual.binding_features[:, :4],
        expected.binding_features[:, permutation],
    )
    torch.testing.assert_close(actual.binding_features[:, 4:], expected.binding_features[:, 4:])


def test_active_touch_requires_one_shared_nonnegative_group() -> None:
    projector = _projector()
    banks = _banks()
    touch = banks[1]
    bad_touch = NativeTokenBank(
        touch.modality,
        touch.tokens,
        touch.valid,
        group_id=torch.tensor([[1, 2, 1], [-1, -1, -1]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="share one"):
        projector((banks[0], bad_touch))


def test_visual_object_group_cannot_leak_into_runtime_forward() -> None:
    projector = _projector()
    vision, touch = _banks()
    leaked = NativeTokenBank(
        vision.modality,
        vision.tokens,
        vision.valid,
        vision.geometry,
        group_id=torch.where(
            vision.valid,
            torch.zeros_like(vision.valid, dtype=torch.long),
            torch.full_like(vision.valid, -1, dtype=torch.long),
        ),
    )

    with pytest.raises(ValueError, match="object labels belong only in loss targets"):
        projector((leaked, touch))


def test_inactive_touch_emits_no_valid_token_or_group() -> None:
    projector = _projector()
    vision = _banks()[0]
    inactive_touch = NativeTokenBank(
        "anytouch",
        torch.zeros(2, 3, 5),
        torch.zeros(2, 3, dtype=torch.bool),
        group_id=torch.full((2, 3), -1, dtype=torch.long),
    )
    output = projector((vision, inactive_touch))

    assert not output.token_valid[:, 4:].any()
    assert torch.equal(output.token_group_id[:, 4:], torch.full((2, 3), -1))
    assert torch.equal(output.binding_features[:, 4:], torch.zeros(2, 3, 11))


def test_projection_gradients_do_not_require_mutating_native_banks() -> None:
    projector = _projector().train()
    banks = _banks()
    output = projector(banks)
    output.binding_features.square().mean().backward()

    assert projector.content_projection["vjepa"].weight.grad is not None
    assert projector.content_projection["anytouch"].weight.grad is not None
    assert projector.geometry_projection["vjepa"].weight.grad is not None
    assert projector.modality_embedding.grad is not None


def test_rejects_duplicate_modality_or_nonzero_padding() -> None:
    projector = _projector()
    banks = _banks()
    with pytest.raises(ValueError, match="more than once"):
        projector((banks[0], banks[0]))
    with pytest.raises(ValueError, match="padding"):
        projector(
            (
                NativeTokenBank(
                    "anytouch",
                    torch.ones(1, 2, 5),
                    torch.zeros(1, 2, dtype=torch.bool),
                    group_id=torch.full((1, 2), -1, dtype=torch.long),
                ),
            )
        )
