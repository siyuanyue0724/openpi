from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
configuration = pytest.importorskip("olmo.hf_model.configuration_molmoact2")
modeling = pytest.importorskip("olmo.hf_model.modeling_molmoact2")
context = pytest.importorskip("picf_next.hosts.context")
interventions = pytest.importorskip("picf_next.hosts.interventions")
adapter = pytest.importorskip("picf_next.hosts.molmoact2")
evidence_module = pytest.importorskip("picf_next.models.evidence")

ActionExpert = modeling.ActionExpert
MolmoAct2ActionExpertConfig = configuration.MolmoAct2ActionExpertConfig
PICFActionEvidence = context.PICFActionEvidence
MolmoAct2PICFActionExpert = adapter.MolmoAct2PICFActionExpert
NativeTokenBank = evidence_module.NativeTokenBank


def _expert() -> ActionExpert:
    torch.manual_seed(811)
    expert = ActionExpert(
        MolmoAct2ActionExpertConfig(
            max_action_horizon=3,
            max_action_dim=4,
            hidden_size=16,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            ffn_multiple_of=8,
            timestep_embed_dim=8,
        ),
        llm_dim=16,
        llm_kv_dim=8,
        llm_num_layers=2,
    )
    with torch.no_grad():
        for block in expert.blocks:
            torch.nn.init.xavier_uniform_(block.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.linear.weight)
    return expert


def _evidence(offset: float = 0.0) -> PICFActionEvidence:
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    ownership = torch.tensor([[[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.5, 0.5, 0.0]]])
    cosine = torch.cos(torch.tensor(offset))
    sine = torch.sin(torch.tensor(offset))
    return PICFActionEvidence(
        dense_banks=(NativeTokenBank("vision", tokens, valid),),
        object_address=torch.stack(
            (
                torch.stack((cosine, sine)),
                torch.stack((-sine, cosine)),
            )
        ).unsqueeze(0),
        object_value=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]) + offset,
        object_valid=torch.ones(1, 2, dtype=torch.bool),
        object_log_prior=torch.tensor([[-0.1, -0.3]]),
        dense_ownership=(ownership,),
    )


def _run(adapter: MolmoAct2PICFActionExpert, evidence: PICFActionEvidence) -> torch.Tensor:
    torch.manual_seed(821)
    return adapter(
        torch.randn(1, 3, 4),
        torch.tensor([0.4]),
        encoder_kv_states=[(torch.randn(1, 4, 8), torch.randn(1, 4, 8)) for _ in range(2)],
        encoder_attention_mask=torch.ones(1, 4, dtype=torch.bool),
        evidence=evidence,
    )


def test_joint_row_permutation_is_numerically_action_invariant() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 2},
        object_address_dim=2,
        object_value_dim=3,
    ).eval()
    for branch in (*wrapped.dense_branches, *wrapped.object_branches):
        branch.gate.data.fill_(0.3)
    evidence = _evidence()
    permutation = torch.tensor([1, 0])

    expected = _run(wrapped, evidence)
    actual = _run(
        wrapped,
        interventions.permute_object_rows(
            evidence,
            permutation,
            keep_ownership_fixed=False,
        ),
    )

    # Set attention is permutation invariant over real arithmetic. Reordering
    # rows changes floating-point reduction order, so bitwise equality would
    # require a discontinuous canonical sort in the production model.
    tolerance = 8 * torch.finfo(expected.dtype).eps
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_zero_wrong_address_and_stale_interventions_reach_final_action() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 2},
        object_address_dim=2,
        object_value_dim=3,
    ).eval()
    for branch in (*wrapped.dense_branches, *wrapped.object_branches):
        branch.gate.data.fill_(0.3)
    predicted = _evidence()
    base_action = _run(wrapped, predicted)
    controls = {
        "zero": interventions.without_posterior(predicted),
        "removed_row": interventions.without_object_rows(
            predicted,
            torch.tensor([[True, False]]),
        ),
        "wrong_address": interventions.permute_object_rows(
            predicted,
            torch.tensor([1, 0]),
            keep_ownership_fixed=True,
        ),
        "address_only": interventions.permute_object_addresses(
            predicted,
            torch.tensor([1, 0]),
        ),
        "stale": interventions.stale_posterior(predicted, _evidence(offset=0.7)),
    }

    for name, controlled in controls.items():
        assert not torch.equal(_run(wrapped, controlled), base_action), name


def test_interventions_do_not_mutate_input_and_reject_invalid_controls() -> None:
    evidence = _evidence()
    address = evidence.object_address.clone()
    interventions.permute_object_rows(
        evidence,
        torch.tensor([1, 0]),
        keep_ownership_fixed=True,
    )
    address_only = interventions.permute_object_addresses(
        evidence,
        torch.tensor([1, 0]),
    )
    torch.testing.assert_close(evidence.object_address, address)
    torch.testing.assert_close(address_only.object_address, address[:, [1, 0]])
    assert address_only.object_value is evidence.object_value
    assert address_only.object_valid is evidence.object_valid
    assert address_only.object_log_prior is evidence.object_log_prior
    assert address_only.dense_ownership is evidence.dense_ownership

    with pytest.raises(ValueError, match="bijective"):
        interventions.permute_object_rows(
            evidence,
            torch.tensor([0, 0]),
            keep_ownership_fixed=False,
        )
    with pytest.raises(ValueError, match="complete previous"):
        interventions.stale_posterior(evidence, interventions.without_posterior(evidence))
