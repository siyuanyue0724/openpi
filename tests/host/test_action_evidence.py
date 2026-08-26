from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
context_module = pytest.importorskip("picf_next.hosts.context")
interventions_module = pytest.importorskip("picf_next.hosts.interventions")
evidence_module = pytest.importorskip("picf_next.models.evidence")

PICFActionEvidence = context_module.PICFActionEvidence
NativeTokenBank = evidence_module.NativeTokenBank


def _bank() -> NativeTokenBank:
    return NativeTokenBank(
        modality="vision",
        tokens=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]),
        valid=torch.tensor([[True, True, False]]),
    )


def _removal_evidence() -> PICFActionEvidence:
    return PICFActionEvidence(
        dense_banks=(_bank(),),
        object_address=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        object_value=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
        object_valid=torch.ones(1, 2, dtype=torch.bool),
        object_log_prior=torch.tensor([[-0.1, -0.3]]),
        dense_ownership=(torch.tensor([[[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]]),),
    )


def test_row_removal_preserves_dense_tokens_and_returns_ownership_to_context() -> None:
    evidence = _removal_evidence()

    removed = interventions_module.without_object_rows(
        evidence,
        torch.tensor([[True, False]]),
    )

    assert removed.dense_banks is evidence.dense_banks
    assert removed.object_valid is not None
    assert removed.object_valid.tolist() == [[False, True]]
    assert removed.object_address is not None
    assert removed.object_value is not None
    assert removed.object_log_prior is not None
    torch.testing.assert_close(removed.object_address[:, 0], torch.zeros(1, 2))
    torch.testing.assert_close(removed.object_value[:, 0], torch.zeros(1, 3))
    torch.testing.assert_close(removed.object_log_prior[:, 0], torch.zeros(1))
    torch.testing.assert_close(removed.object_address[:, 1], evidence.object_address[:, 1])
    assert removed.dense_ownership is not None
    expected = torch.tensor([[[0.0, 0.1, 0.9], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(removed.dense_ownership[0], expected)
    torch.testing.assert_close(
        evidence.dense_ownership[0],
        torch.tensor([[[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]]),
    )


def test_row_removal_rejects_malformed_selector_or_missing_posterior() -> None:
    evidence = _removal_evidence()
    with pytest.raises(ValueError, match="boolean batch-by-object"):
        interventions_module.without_object_rows(evidence, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="complete object bank"):
        interventions_module.without_object_rows(
            interventions_module.without_posterior(evidence),
            torch.zeros(1, 2, dtype=torch.bool),
        )


def test_row_removal_preserves_bfloat16_under_autocast() -> None:
    source = _removal_evidence()
    bank = source.dense_banks[0]
    evidence = PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                modality=bank.modality,
                tokens=bank.tokens.to(torch.bfloat16),
                valid=bank.valid,
            ),
        ),
        object_address=source.object_address.to(torch.bfloat16),
        object_value=source.object_value.to(torch.bfloat16),
        object_valid=source.object_valid,
        object_log_prior=source.object_log_prior.to(torch.bfloat16),
        dense_ownership=(source.dense_ownership[0].to(torch.bfloat16),),
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        removed = interventions_module.without_object_rows(
            evidence,
            torch.tensor([[True, False]]),
        )

    assert removed.dense_ownership[0].dtype == torch.bfloat16
    assert removed.object_address.dtype == torch.bfloat16
    assert removed.object_value.dtype == torch.bfloat16
    assert removed.object_log_prior.dtype == torch.bfloat16


def test_ownership_weighted_address_is_permutation_invariant_and_context_safe() -> None:
    address = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True, False]])
    ownership = torch.tensor(
        [
            [
                [0.75, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    evidence = PICFActionEvidence(
        dense_banks=(_bank(),),
        object_address=address,
        object_value=torch.zeros(1, 3, 4),
        object_valid=valid,
        object_log_prior=torch.zeros(1, 3),
        dense_ownership=(ownership,),
    )
    actual = evidence.ownership_weighted_addresses()[0]
    torch.testing.assert_close(
        actual,
        torch.tensor([[[0.75, 0.25], [0.0, 0.0], [0.0, 0.0]]]),
    )

    permutation = torch.tensor([1, 0, 2])
    permuted = PICFActionEvidence(
        dense_banks=evidence.dense_banks,
        object_address=address[:, permutation],
        object_value=evidence.object_value[:, permutation],
        object_valid=valid[:, permutation],
        object_log_prior=evidence.object_log_prior[:, permutation],
        dense_ownership=(
            torch.cat((ownership[..., :-1][..., permutation], ownership[..., -1:]), dim=-1),
        ),
    )
    torch.testing.assert_close(permuted.ownership_weighted_addresses()[0], actual)

    metadata_actual = evidence.ownership_weighted_addresses(validate_tensor_values=False)[0]
    torch.testing.assert_close(metadata_actual, actual, atol=0.0, rtol=0.0)


def test_ownership_runtime_validation_flag_must_be_boolean() -> None:
    evidence = PICFActionEvidence(
        dense_banks=(_bank(),),
        object_address=torch.zeros(1, 1, 2),
        object_value=torch.zeros(1, 1, 3),
        object_valid=torch.ones(1, 1, dtype=torch.bool),
        object_log_prior=torch.zeros(1, 1),
    )
    with pytest.raises(ValueError, match="boolean"):
        evidence.ownership_weighted_addresses(validate_tensor_values=1)


def test_ownership_rejects_mass_on_unused_object_or_invalid_token() -> None:
    bank = _bank()
    address = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, False]])
    bad_unused = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    evidence = PICFActionEvidence(
        dense_banks=(bank,),
        object_address=address,
        object_value=torch.zeros(1, 2, 3),
        object_valid=valid,
        object_log_prior=torch.zeros(1, 2),
        dense_ownership=(bad_unused,),
    )
    with pytest.raises(ValueError, match="unused objects"):
        evidence.ownership_weighted_addresses()

    bad_padding = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    evidence = PICFActionEvidence(
        dense_banks=(bank,),
        object_address=address,
        object_value=torch.zeros(1, 2, 3),
        object_valid=valid,
        object_log_prior=torch.zeros(1, 2),
        dense_ownership=(bad_padding,),
    )
    with pytest.raises(ValueError, match="belong exactly to context"):
        evidence.ownership_weighted_addresses()


def test_action_evidence_rejects_nonunit_valid_identity_address() -> None:
    evidence = PICFActionEvidence(
        dense_banks=(),
        object_address=torch.tensor([[[2.0, 0.0]]]),
        object_value=torch.zeros(1, 1, 3),
        object_valid=torch.ones(1, 1, dtype=torch.bool),
        object_log_prior=torch.zeros(1, 1),
    )

    with pytest.raises(ValueError, match="unit norm"):
        evidence.ownership_weighted_addresses()


def test_object_log_prior_is_atomic_finite_nonpositive_and_zero_padded() -> None:
    address = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    value = torch.zeros(1, 2, 3)
    valid = torch.tensor([[True, False]])
    missing = PICFActionEvidence(
        dense_banks=(),
        object_address=address,
        object_value=value,
        object_valid=valid,
    )
    with pytest.raises(ValueError, match="log prior must be all present or absent"):
        missing.batch_size()

    positive = PICFActionEvidence(
        dense_banks=(),
        object_address=address,
        object_value=value,
        object_valid=valid,
        object_log_prior=torch.tensor([[0.1, 0.0]]),
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        positive.ownership_weighted_addresses()

    nonzero_padding = PICFActionEvidence(
        dense_banks=(),
        object_address=address,
        object_value=value,
        object_valid=valid,
        object_log_prior=torch.tensor([[-0.2, -0.3]]),
    )
    with pytest.raises(ValueError, match="unused object log prior"):
        nonzero_padding.ownership_weighted_addresses()


def test_batch_size_rejects_untyped_rank_zero_and_empty_evidence_fields() -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        PICFActionEvidence([], None, None, None).batch_size()

    rank_zero = PICFActionEvidence(
        dense_banks=(),
        object_address=torch.tensor(1.0),
        object_value=torch.tensor(1.0),
        object_valid=torch.tensor(True),
        object_log_prior=torch.tensor(0.0),
    )
    with pytest.raises(ValueError, match="batch dimension"):
        rank_zero.batch_size()

    empty = PICFActionEvidence(
        dense_banks=(),
        object_address=torch.empty(0, 1, 2),
        object_value=torch.empty(0, 1, 3),
        object_valid=torch.empty(0, 1, dtype=torch.bool),
        object_log_prior=torch.empty(0, 1),
    )
    with pytest.raises(ValueError, match="nonempty"):
        empty.batch_size()

    untyped = PICFActionEvidence(
        dense_banks=(),
        object_address=[[1.0]],
        object_value=[[1.0]],
        object_valid=[[True]],
        object_log_prior=[[0.0]],
    )
    with pytest.raises(TypeError, match="must be a tensor"):
        untyped.batch_size()
