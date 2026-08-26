from __future__ import annotations

import numpy as np
import pytest

from picf_next.contracts import ContractError, DenseEvidence

torch = pytest.importorskip("torch")
multimodal_module = pytest.importorskip("picf_next.data.multimodal")
evidence_module = pytest.importorskip("picf_next.models.evidence")

ModalityBatchSpec = multimodal_module.ModalityBatchSpec
collate_dense_evidence = multimodal_module.collate_dense_evidence
ModalityProjectionSpec = evidence_module.ModalityProjectionSpec
MultimodalBindingProjector = evidence_module.MultimodalBindingProjector


def _evidence(
    modality: str,
    count: int,
    width: int,
    *,
    contract: str,
    geometry_dim: int = 0,
    group: int | None = None,
) -> DenseEvidence:
    timestamps = np.linspace(0.0, 0.1, count, dtype=np.float32)
    return DenseEvidence(
        modality=modality,
        encoder_contract=contract,
        tokens=np.arange(count * width, dtype=np.float32).reshape(count, width),
        available=count > 0,
        timestamps=timestamps,
        confidence=np.ones(count, dtype=np.float32),
        geometry=(np.ones((count, geometry_dim), dtype=np.float32) if geometry_dim else None),
        group_ids=(np.full(count, group, dtype=np.int64) if group is not None else None),
        current_measurement_valid=(
            np.isclose(timestamps, timestamps.max(), rtol=0.0, atol=1e-7)
            if count
            else np.zeros(0, dtype=np.bool_)
        ),
    )


def _specs() -> tuple[ModalityBatchSpec, ...]:
    return (
        ModalityBatchSpec("vjepa", "vjepa/test/v1", 4, 2),
        ModalityBatchSpec("sonata", "sonata/test/v1", 3, 3),
        ModalityBatchSpec("anytouch", "anytouch/test/v1", 5, require_single_active_group=True),
    )


def test_multimodal_collator_preserves_every_token_and_represents_missing_as_padding() -> None:
    samples = (
        (
            _evidence("vjepa", 7, 4, contract="vjepa/test/v1", geometry_dim=2),
            _evidence("sonata", 4, 3, contract="sonata/test/v1", geometry_dim=3),
            _evidence("anytouch", 6, 5, contract="anytouch/test/v1", group=9),
        ),
        (
            _evidence("vjepa", 3, 4, contract="vjepa/test/v1", geometry_dim=2),
            _evidence("sonata", 8, 3, contract="sonata/test/v1", geometry_dim=3),
        ),
    )
    banks = collate_dense_evidence(samples, _specs())

    assert [tuple(bank.tokens.shape) for bank in banks] == [(2, 7, 4), (2, 8, 3), (2, 6, 5)]
    assert [bank.valid.sum(dim=1).tolist() for bank in banks] == [[7, 3], [4, 8], [6, 0]]
    assert banks[2].group_id is not None
    assert banks[2].group_id[0].tolist() == [9] * 6
    assert banks[2].group_id[1].tolist() == [-1] * 6
    assert torch.count_nonzero(banks[2].tokens[1]) == 0
    assert banks[0].encoder_contract == "vjepa/test/v1"
    assert [bank.current_measurement_valid.sum(dim=1).tolist() for bank in banks] == [
        [1, 1],
        [1, 1],
        [1, 0],
    ]
    assert all(
        bank.timestamps is not None and bank.timestamps.dtype == torch.float32 for bank in banks
    )

    projector = MultimodalBindingProjector(
        (
            ModalityProjectionSpec("vjepa", 4, 2),
            ModalityProjectionSpec("sonata", 3, 3),
            ModalityProjectionSpec("anytouch", 5, require_single_active_group=True),
        ),
        binding_dim=6,
    )
    output = projector(banks)
    assert output.total_tokens == 21
    assert torch.equal(output.native_banks[0].tokens, banks[0].tokens)
    assert output.current_measurement_valid.sum().item() == 5


def test_multimodal_collator_rejects_contract_drift_and_scattered_touch() -> None:
    wrong_contract = ((_evidence("vjepa", 2, 4, contract="vjepa/wrong", geometry_dim=2),),)
    with pytest.raises(ContractError, match="encoder contract"):
        collate_dense_evidence(wrong_contract, _specs())

    touch = _evidence("anytouch", 3, 5, contract="anytouch/test/v1", group=1)
    scattered = DenseEvidence(
        modality=touch.modality,
        encoder_contract=touch.encoder_contract,
        tokens=touch.tokens,
        available=True,
        timestamps=touch.timestamps,
        confidence=touch.confidence,
        group_ids=np.array([1, 2, 1], dtype=np.int64),
        current_measurement_valid=touch.current_measurement_valid,
    )
    with pytest.raises(ContractError, match="share one group"):
        collate_dense_evidence(((scattered,),), _specs())


def test_group_ids_are_namespaced_across_grouped_modalities() -> None:
    specs = (
        ModalityProjectionSpec("touch", 3, require_single_active_group=True),
        ModalityProjectionSpec("audio_contact", 3, require_single_active_group=True),
    )
    projector = MultimodalBindingProjector(specs, binding_dim=4)
    banks = (
        evidence_module.NativeTokenBank(
            modality="touch",
            tokens=torch.ones((1, 2, 3)),
            valid=torch.ones((1, 2), dtype=torch.bool),
            group_id=torch.zeros((1, 2), dtype=torch.long),
        ),
        evidence_module.NativeTokenBank(
            modality="audio_contact",
            tokens=torch.ones((1, 3, 3)),
            valid=torch.ones((1, 3), dtype=torch.bool),
            group_id=torch.zeros((1, 3), dtype=torch.long),
        ),
    )

    output = projector(banks)

    assert output.token_group_id[0, :2].unique().numel() == 1
    assert output.token_group_id[0, 2:].unique().numel() == 1
    assert output.token_group_id[0, 0] != output.token_group_id[0, 2]
