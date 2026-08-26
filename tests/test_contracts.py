from __future__ import annotations

import numpy as np
import pytest

from picf_next.contracts import (
    ContractError,
    DenseEvidence,
    ObjectBeliefSet,
    ObjectObservationSet,
    PICFContext,
)


def _belief(capacity: int = 2) -> ObjectBeliefSet:
    address = np.zeros((capacity, 4), dtype=np.float32)
    address[0, 0] = 1.0
    return ObjectBeliefSet(
        address=address,
        content=np.zeros((capacity, 6), dtype=np.float32),
        geometry=np.zeros((capacity, 3), dtype=np.float32),
        geometry_covariance_diag=np.array(
            [[1.0] * 3, [0.0] * 3],
            dtype=np.float32,
        )[:capacity],
        existence=np.array([0.9, 0.0], dtype=np.float32)[:capacity],
        visibility=np.array([0.8, 0.0], dtype=np.float32)[:capacity],
        measurement_age_s=np.array([0.3, 0.0], dtype=np.float32)[:capacity],
        valid=np.array([True, False], dtype=np.bool_)[:capacity],
        age=np.array([3, 0], dtype=np.int64)[:capacity],
    )


def test_missing_modality_cannot_emit_tokens() -> None:
    with pytest.raises(ContractError, match="missing modality"):
        DenseEvidence(
            modality="tactile",
            encoder_contract="anytouch/v1/all-output-tokens",
            tokens=np.zeros((1, 8), dtype=np.float32),
            available=False,
            timestamps=np.zeros(1, dtype=np.float32),
            confidence=np.ones(1, dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modality", 7, "modality"),
        ("encoder_contract", 7, "encoder_contract"),
        ("available", 0, "available"),
        ("tokens", [[0.0, 0.0]], "NumPy array"),
    ],
)
def test_dense_evidence_public_fields_fail_with_contract_errors(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "modality": "vision",
        "encoder_contract": "vision/v1",
        "tokens": np.zeros((1, 2), dtype=np.float32),
        "available": True,
        "timestamps": np.zeros(1, dtype=np.float32),
        "confidence": np.ones(1, dtype=np.float32),
    }
    arguments[field] = value
    with pytest.raises(ContractError, match=message):
        DenseEvidence(**arguments)  # type: ignore[arg-type]


def test_context_requires_immutable_typed_collections() -> None:
    belief = _belief()
    with pytest.raises(ContractError, match="evidence must be a tuple"):
        PICFContext(
            evidence=[],  # type: ignore[arg-type]
            posterior=belief,
            innovation=np.zeros((2, 9), dtype=np.float32),
            ownership=(),
        )
    with pytest.raises(ContractError, match="ownership must be a tuple"):
        PICFContext(
            evidence=(),
            posterior=belief,
            innovation=np.zeros((2, 9), dtype=np.float32),
            ownership=[],  # type: ignore[arg-type]
        )


def test_dense_evidence_rejects_invalid_temporal_and_group_metadata() -> None:
    with pytest.raises(ContractError, match="timestamps"):
        DenseEvidence(
            modality="video",
            encoder_contract="video/v1",
            tokens=np.zeros((1, 2), dtype=np.float32),
            available=True,
            timestamps=np.array([-1.0], dtype=np.float32),
            confidence=np.ones(1, dtype=np.float32),
        )
    with pytest.raises(ContractError, match="group_ids"):
        DenseEvidence(
            modality="touch",
            encoder_contract="touch/v1",
            tokens=np.zeros((1, 2), dtype=np.float32),
            available=True,
            timestamps=np.zeros(1, dtype=np.float32),
            confidence=np.ones(1, dtype=np.float32),
            group_ids=np.array([-2], dtype=np.int64),
        )


def test_multitime_evidence_requires_one_explicit_newest_measurement_role() -> None:
    arguments = {
        "modality": "video",
        "encoder_contract": "video/v1",
        "tokens": np.zeros((3, 2), dtype=np.float32),
        "available": True,
        "timestamps": np.array([0.0, 0.1, 0.2], dtype=np.float32),
        "confidence": np.ones(3, dtype=np.float32),
    }
    with pytest.raises(ContractError, match="explicit current_measurement_valid"):
        DenseEvidence(**arguments)
    with pytest.raises(ContractError, match="newest evidence timestamp"):
        DenseEvidence(
            **arguments,
            current_measurement_valid=np.array([True, False, False], dtype=np.bool_),
        )

    evidence = DenseEvidence(
        **arguments,
        current_measurement_valid=np.array([False, False, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        evidence.effective_current_measurement_valid,
        [False, False, True],
    )


def test_context_requires_ownership_for_every_complete_stream() -> None:
    visual = DenseEvidence(
        modality="video",
        encoder_contract="vjepa2.1/vitg16/final-dense/v1",
        tokens=np.zeros((3, 8), dtype=np.float32),
        available=True,
        timestamps=np.arange(3, dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        current_measurement_valid=np.array([False, False, True], dtype=np.bool_),
    )
    ownership = np.array([[0.7, 0.0, 0.3]] * 3, dtype=np.float32)

    context = PICFContext(
        evidence=(visual,),
        posterior=_belief(),
        innovation=np.zeros((2, 9), dtype=np.float32),
        ownership=(ownership,),
    )

    assert context.evidence_for("video") is visual
    assert context.evidence_for("tactile") is None
    assert visual.token_count == 3


def test_context_rejects_non_normalized_ownership() -> None:
    visual = DenseEvidence(
        modality="video",
        encoder_contract="vjepa2.1/vitg16/final-dense/v1",
        tokens=np.zeros((1, 8), dtype=np.float32),
        available=True,
        timestamps=np.zeros(1, dtype=np.float32),
        confidence=np.ones(1, dtype=np.float32),
    )

    with pytest.raises(ContractError, match="sum to one"):
        PICFContext(
            evidence=(visual,),
            posterior=_belief(),
            innovation=np.zeros((2, 9), dtype=np.float32),
            ownership=(np.array([[0.2, 0.2, 0.2]], dtype=np.float32),),
        )


def test_context_rejects_innovation_with_wrong_dynamic_width() -> None:
    with pytest.raises(ContractError, match="posterior dynamic shape"):
        PICFContext(
            evidence=(),
            posterior=_belief(),
            innovation=np.zeros((2, 12), dtype=np.float32),
            ownership=(),
        )


def test_context_rejects_stale_innovation_on_unused_row() -> None:
    innovation = np.zeros((2, 9), dtype=np.float32)
    innovation[1, 0] = 1.0

    with pytest.raises(ContractError, match="zero innovation"):
        PICFContext(
            evidence=(),
            posterior=_belief(),
            innovation=innovation,
            ownership=(),
        )


def test_context_rejects_ownership_mass_on_unused_object_row() -> None:
    visual = DenseEvidence(
        modality="video",
        encoder_contract="vjepa2.1/vitg16/final-dense/v1",
        tokens=np.zeros((1, 8), dtype=np.float32),
        available=True,
        timestamps=np.zeros(1, dtype=np.float32),
        confidence=np.ones(1, dtype=np.float32),
    )

    with pytest.raises(ContractError, match="unused posterior rows"):
        PICFContext(
            evidence=(visual,),
            posterior=_belief(),
            innovation=np.zeros((2, 9), dtype=np.float32),
            ownership=(np.array([[0.5, 0.25, 0.25]], dtype=np.float32),),
        )


def test_visibility_cannot_exceed_existence() -> None:
    with pytest.raises(ContractError, match="visibility"):
        ObjectBeliefSet(
            address=np.array([[1.0, 0.0]], dtype=np.float32),
            content=np.zeros((1, 2), dtype=np.float32),
            geometry=np.zeros((1, 2), dtype=np.float32),
            geometry_covariance_diag=np.ones((1, 2), dtype=np.float32),
            existence=np.array([0.2], dtype=np.float32),
            visibility=np.array([0.8], dtype=np.float32),
            measurement_age_s=np.zeros(1, dtype=np.float32),
            valid=np.array([True], dtype=np.bool_),
            age=np.array([0], dtype=np.int64),
        )


def test_covariance_covers_only_the_geometry_state() -> None:
    with pytest.raises(ContractError, match="geometry_covariance_diag width"):
        ObjectObservationSet(
            address=np.array([[1.0, 0.0]], dtype=np.float32),
            content=np.zeros((1, 3), dtype=np.float32),
            geometry=np.zeros((1, 4), dtype=np.float32),
            geometry_covariance_diag=np.ones((1, 8), dtype=np.float32),
            existence=np.ones(1, dtype=np.float32),
            valid=np.ones(1, dtype=np.bool_),
        )


def test_grouped_tokens_share_one_object_assignment_distribution() -> None:
    tactile = DenseEvidence(
        modality="tactile",
        encoder_contract="anytouch/v1/all-output-tokens",
        tokens=np.zeros((3, 8), dtype=np.float32),
        available=True,
        timestamps=np.zeros(3, dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        group_ids=np.zeros(3, dtype=np.int64),
    )
    scattered = np.array(
        [[0.8, 0.0, 0.2], [0.2, 0.0, 0.8], [0.8, 0.0, 0.2]],
        dtype=np.float32,
    )

    with pytest.raises(ContractError, match="shared within token group"):
        PICFContext(
            evidence=(tactile,),
            posterior=_belief(),
            innovation=np.zeros((2, 9), dtype=np.float32),
            ownership=(scattered,),
        )


def test_grouped_tokens_may_softly_share_one_object_distribution() -> None:
    tactile = DenseEvidence(
        modality="tactile",
        encoder_contract="anytouch/v1/all-output-tokens",
        tokens=np.zeros((3, 8), dtype=np.float32),
        available=True,
        timestamps=np.zeros(3, dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        group_ids=np.zeros(3, dtype=np.int64),
    )
    shared = np.array([[0.6, 0.0, 0.4]] * 3, dtype=np.float32)

    PICFContext(
        evidence=(tactile,),
        posterior=_belief(),
        innovation=np.zeros((2, 9), dtype=np.float32),
        ownership=(shared,),
    )


def test_unused_posterior_rows_cannot_carry_stale_identity() -> None:
    with pytest.raises(ContractError, match="zero address"):
        ObjectBeliefSet(
            address=np.array([[1.0, 2.0]], dtype=np.float32),
            content=np.zeros((1, 2), dtype=np.float32),
            geometry=np.zeros((1, 2), dtype=np.float32),
            geometry_covariance_diag=np.zeros((1, 2), dtype=np.float32),
            existence=np.zeros(1, dtype=np.float32),
            visibility=np.zeros(1, dtype=np.float32),
            measurement_age_s=np.zeros(1, dtype=np.float32),
            valid=np.zeros(1, dtype=np.bool_),
            age=np.zeros(1, dtype=np.int64),
        )
