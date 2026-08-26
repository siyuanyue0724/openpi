from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.frozen_posterior_diagnostic import (
    DiagnosticInformationNode,
    FrozenPosteriorActionRequest,
    FrozenPosteriorShapeContract,
    FrozenPosteriorVisibility,
    LabelFreePromptVariant,
    LanguagePromptBatch,
    OfflinePromptTargetRows,
    audit_frozen_posterior_visibility,
    capture_factual_posterior_snapshot,
    consistent_row_permutation_arm,
    factual_frozen_posterior_arm,
    frozen_posterior_visibility_edges,
    label_blind_moment_matched_donor_arms,
    label_blind_visibility_removal_arms,
    parse_frozen_posterior_visibility,
    run_frozen_posterior_action_diagnostic,
    score_offline_prompt_switch,
    score_offline_row_selectivity,
)
from picf_next.lingbot_native.state import NativeLayerwisePosteriorState


def _controls(*, batch_size: int = 2, action_dim: int = 3) -> ExecutedControlBatch:
    values = torch.arange(batch_size * action_dim, dtype=torch.float32).reshape(
        batch_size, 1, action_dim
    )
    return ExecutedControlBatch(
        values=values,
        field_valid=torch.ones_like(values, dtype=torch.bool),
        token_valid=torch.ones(batch_size, 1, dtype=torch.bool),
        delta_time=torch.full((batch_size, 1), 0.1),
        reset=torch.zeros(batch_size, 1, dtype=torch.bool),
        acknowledged=torch.ones(batch_size, 1, dtype=torch.bool),
    )


def _snapshot(
    *,
    provenance_id: str,
    offset: float = 0.0,
    batch_size: int = 2,
    layers: int = 3,
    capacity: int = 4,
    width: int = 6,
):
    rows = torch.arange(
        batch_size * layers * capacity * width,
        dtype=torch.float32,
    ).reshape(batch_size, layers, capacity, width)
    rows = rows + offset
    return capture_factual_posterior_snapshot(
        lambda: NativeLayerwisePosteriorState(rows),
        shape_contract=FrozenPosteriorShapeContract(
            num_layers=layers,
            capacity=capacity,
            host_width=width,
        ),
        provenance_id=provenance_id,
    )


def _prompts(batch_size: int = 2) -> tuple[LabelFreePromptVariant, ...]:
    valid = torch.ones(batch_size, 3, dtype=torch.bool)
    return (
        LabelFreePromptVariant(
            name="prompt-blue",
            language=LanguagePromptBatch(
                token_ids=torch.tensor([[11, 12, 13], [11, 12, 13]]),
                token_valid=valid,
            ),
        ),
        LabelFreePromptVariant(
            name="prompt-red",
            language=LanguagePromptBatch(
                token_ids=torch.tensor([[21, 22, 23], [21, 22, 23]]),
                token_valid=valid.clone(),
            ),
        ),
    )


def test_visibility_parser_is_exact_and_fail_closed() -> None:
    assert parse_frozen_posterior_visibility("direct-only") is (
        FrozenPosteriorVisibility.DIRECT_ONLY
    )
    assert parse_frozen_posterior_visibility("language-mediated") is (
        FrozenPosteriorVisibility.LANGUAGE_MEDIATED
    )
    assert parse_frozen_posterior_visibility("both") is FrozenPosteriorVisibility.BOTH
    for invalid in ("DIRECT_ONLY", " direct-only", "direct_only", "posterior-only", ""):
        with pytest.raises(ValueError, match="unknown frozen-posterior visibility"):
            parse_frozen_posterior_visibility(invalid)
    with pytest.raises(TypeError, match="must be a string"):
        parse_frozen_posterior_visibility(FrozenPosteriorVisibility.BOTH)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("visibility", "direct", "mediated"),
    (
        (FrozenPosteriorVisibility.DIRECT_ONLY, True, False),
        (FrozenPosteriorVisibility.LANGUAGE_MEDIATED, False, True),
        (FrozenPosteriorVisibility.BOTH, True, True),
    ),
)
def test_visibility_contracts_are_causally_closed(
    visibility: FrozenPosteriorVisibility,
    direct: bool,
    mediated: bool,
) -> None:
    audit = audit_frozen_posterior_visibility(visibility)
    assert audit.direct_posterior_path is direct
    assert audit.language_mediated_posterior_path is mediated
    assert audit.forbidden_sources_reaching_action == ()
    edges = frozen_posterior_visibility_edges(visibility)
    assert (DiagnosticInformationNode.LANGUAGE, DiagnosticInformationNode.ACTION) in edges
    assert (DiagnosticInformationNode.CONTROL, DiagnosticInformationNode.ACTION) in edges
    assert (DiagnosticInformationNode.PROPRIOCEPTION, DiagnosticInformationNode.ACTION) in edges
    assert all(
        source not in {edge[0] for edge in edges}
        for source in (
            DiagnosticInformationNode.CURRENT_SCENE,
            DiagnosticInformationNode.DENSE_MODALITY,
            DiagnosticInformationNode.EXTERNAL_TRACE,
            DiagnosticInformationNode.PRIOR,
            DiagnosticInformationNode.HOST_AUX,
            DiagnosticInformationNode.MATCH,
        )
    )


def test_factual_correction_snapshot_is_complete_detached_and_copied() -> None:
    source = torch.randn(2, 3, 4, 5, requires_grad=True)
    calls = 0

    def correction() -> NativeLayerwisePosteriorState:
        nonlocal calls
        calls += 1
        return NativeLayerwisePosteriorState(source)

    snapshot = capture_factual_posterior_snapshot(
        correction,
        shape_contract=FrozenPosteriorShapeContract(3, 4, 5),
        provenance_id="episode-1-frame-7",
    )
    assert calls == 1
    assert snapshot.state.layer_rows.shape == (2, 3, 4, 5)
    assert not snapshot.state.layer_rows.requires_grad
    assert snapshot.state.layer_rows.data_ptr() != source.data_ptr()
    torch.testing.assert_close(snapshot.state.layer_rows, source.detach())
    snapshot.assert_intact()

    with pytest.raises(ValueError, match="complete declared layerwise state"):
        capture_factual_posterior_snapshot(
            lambda: NativeLayerwisePosteriorState(torch.zeros(2, 2, 4, 5)),
            shape_contract=FrozenPosteriorShapeContract(3, 4, 5),
            provenance_id="wrong-depth",
        )
    with pytest.raises(TypeError, match="NativeLayerwisePosteriorState"):
        capture_factual_posterior_snapshot(
            lambda: torch.zeros(2, 3, 4, 5),  # type: ignore[arg-type,return-value]
            shape_contract=FrozenPosteriorShapeContract(3, 4, 5),
            provenance_id="wrong-schema",
        )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_snapshot_digest_and_moment_matching_support_host_dtypes(dtype: torch.dtype) -> None:
    factual_rows = torch.arange(2 * 3 * 4 * 8, dtype=torch.float32).reshape(2, 3, 4, 8)
    donor_rows = factual_rows.flip(dims=(-1,)) + 500.0
    contract = FrozenPosteriorShapeContract(3, 4, 8)
    factual = capture_factual_posterior_snapshot(
        lambda: NativeLayerwisePosteriorState(factual_rows.to(dtype)),
        shape_contract=contract,
        provenance_id=f"factual-{dtype}",
    )
    donor = capture_factual_posterior_snapshot(
        lambda: NativeLayerwisePosteriorState(donor_rows.to(dtype)),
        shape_contract=contract,
        provenance_id=f"donor-{dtype}",
    )
    arm = label_blind_moment_matched_donor_arms(factual, donor)[1]
    assert arm.state.layer_rows.dtype == dtype
    factual.assert_intact()
    donor.assert_intact()
    changed = arm.state.layer_rows[:, :, 1].float()
    target = factual.state.layer_rows[:, :, 1].float()
    torch.testing.assert_close(changed.mean(dim=-1), target.mean(dim=-1), atol=0.25, rtol=0)
    torch.testing.assert_close(
        (changed - changed.mean(dim=-1, keepdim=True)).square().mean(dim=-1).sqrt(),
        (target - target.mean(dim=-1, keepdim=True)).square().mean(dim=-1).sqrt(),
        atol=0.25,
        rtol=0,
    )


def test_label_blind_row_arms_cover_every_row_without_target_input() -> None:
    snapshot = _snapshot(provenance_id="factual")
    arms = label_blind_visibility_removal_arms(snapshot)
    assert len(arms) == snapshot.state.capacity
    for row_index, arm in enumerate(arms):
        assert arm.row_index == row_index
        assert not arm.row_visible[:, row_index].any()
        remaining = arm.row_visible.clone()
        remaining[:, row_index] = True
        assert remaining.all()
        torch.testing.assert_close(arm.state.layer_rows, snapshot.state.layer_rows)

    signature = inspect.signature(label_blind_visibility_removal_arms)
    assert "target_row" not in signature.parameters
    assert "labels" not in signature.parameters
    assert "sidecar" not in signature.parameters


def test_moment_matched_donor_replaces_one_row_across_every_layer() -> None:
    factual = _snapshot(provenance_id="factual", offset=5.0)
    donor_rows = factual.state.layer_rows.flip(dims=(-1,)) + 1000.0
    donor = capture_factual_posterior_snapshot(
        lambda: NativeLayerwisePosteriorState(donor_rows),
        shape_contract=factual.shape_contract,
        provenance_id="donor",
    )
    arms = label_blind_moment_matched_donor_arms(factual, donor)
    assert len(arms) == factual.state.capacity
    for row_index, arm in enumerate(arms):
        target = factual.state.layer_rows[:, :, row_index]
        changed = arm.state.layer_rows[:, :, row_index]
        torch.testing.assert_close(changed.mean(dim=-1), target.mean(dim=-1))
        torch.testing.assert_close(
            (changed - changed.mean(dim=-1, keepdim=True)).square().mean(dim=-1).sqrt(),
            (target - target.mean(dim=-1, keepdim=True)).square().mean(dim=-1).sqrt(),
            atol=1e-5,
            rtol=1e-5,
        )
        assert not torch.equal(changed, target)
        for other in range(factual.state.capacity):
            if other != row_index:
                torch.testing.assert_close(
                    arm.state.layer_rows[:, :, other],
                    factual.state.layer_rows[:, :, other],
                )

    with pytest.raises(ValueError, match="different provenance"):
        label_blind_moment_matched_donor_arms(factual, factual)

    degenerate = capture_factual_posterior_snapshot(
        lambda: NativeLayerwisePosteriorState(torch.ones_like(factual.state.layer_rows)),
        shape_contract=factual.shape_contract,
        provenance_id="degenerate-donor",
    )
    with pytest.raises(ValueError, match="degenerate donor row"):
        label_blind_moment_matched_donor_arms(factual, degenerate)


def test_consistent_permutation_moves_the_same_row_at_every_layer() -> None:
    snapshot = _snapshot(provenance_id="factual")
    permutation = (2, 0, 3, 1)
    arm = consistent_row_permutation_arm(snapshot, permutation)
    torch.testing.assert_close(
        arm.state.layer_rows,
        snapshot.state.layer_rows[:, :, torch.tensor(permutation)],
    )
    assert arm.permutation == permutation
    with pytest.raises(ValueError, match="every row exactly once"):
        consistent_row_permutation_arm(snapshot, (0, 0, 1, 2))
    with pytest.raises(TypeError, match="must be integers"):
        consistent_row_permutation_arm(snapshot, (0.0, 1.0, 2.0, 3.0))
    with pytest.raises(TypeError, match="must be integers"):
        consistent_row_permutation_arm(snapshot, (False, 1, 2, 3))


class _RecordingActionReadout:
    def __init__(self) -> None:
        self.requests: list[FrozenPosteriorActionRequest] = []

    def __call__(self, request: FrozenPosteriorActionRequest) -> torch.Tensor:
        self.requests.append(request)
        visible = request.posterior_row_visible[:, None, :, None]
        posterior_mass = (request.posterior.layer_rows * visible).sum(dim=(1, 2, 3))
        language_mass = (
            (request.language.token_ids * request.language.token_valid)
            .sum(dim=1)
            .to(request.inference_noise.dtype)
        )
        scale = (posterior_mass + language_mass)[:, None, None] * 1e-6
        return request.inference_noise + scale


class _PromptSelectiveActionReadout:
    """Infer a row from prompt tokens; no evaluator target metadata is available."""

    def __init__(self) -> None:
        self.requests: list[FrozenPosteriorActionRequest] = []

    def __call__(self, request: FrozenPosteriorActionRequest) -> torch.Tensor:
        self.requests.append(request)
        selected = torch.where(
            request.language.token_ids[:, 0] == 11,
            torch.ones(request.posterior.batch_size, dtype=torch.long),
            torch.full((request.posterior.batch_size,), 2, dtype=torch.long),
        ).to(request.posterior.layer_rows.device)
        batch = torch.arange(request.posterior.batch_size, device=selected.device)
        row_content = request.posterior.layer_rows[batch, -1, selected].mean(dim=-1)
        visible = request.posterior_row_visible[batch, selected].to(row_content.dtype)
        return request.inference_noise + (row_content * visible)[:, None, None] * 1e-3


def test_prompt_switch_uses_identical_noise_and_label_free_forward_requests() -> None:
    snapshot = _snapshot(provenance_id="factual")
    arms = (
        factual_frozen_posterior_arm(snapshot),
        *label_blind_visibility_removal_arms(snapshot),
    )
    noise = torch.randn(2, 5, 3)
    readout = _RecordingActionReadout()
    contracts = tuple(FrozenPosteriorVisibility)
    result = run_frozen_posterior_action_diagnostic(
        readout,
        snapshot=snapshot,
        prompts=_prompts(),
        controls=_controls(),
        proprioception=torch.randn(2, 7),
        inference_noise=noise,
        arms=arms,
        visibility_contracts=contracts,
    )
    expected_count = 2 * len(arms) * len(contracts)
    assert len(readout.requests) == len(result.receipts) == expected_count
    assert {item.inference_noise_sha256 for item in result.receipts} == {
        result.inference_noise_sha256
    }
    for request in readout.requests:
        torch.testing.assert_close(request.inference_noise, noise)

    request_fields = tuple(field.name for field in fields(FrozenPosteriorActionRequest))
    assert request_fields == (
        "language",
        "controls",
        "proprioception",
        "posterior",
        "posterior_row_visible",
        "inference_noise",
        "visibility",
    )
    forbidden_fragments = (
        "rgb",
        "image",
        "dense",
        "modality",
        "history",
        "trace",
        "prior",
        "actions",
        "target",
        "row_index",
        "sidecar",
        "label",
    )
    assert not any(fragment in name for name in request_fields for fragment in forbidden_fragments)
    run_parameters = inspect.signature(run_frozen_posterior_action_diagnostic).parameters
    assert not {"actions", "target_row", "sidecar", "labels"} & set(run_parameters)


def test_action_adapter_mutation_and_shape_mismatch_fail_closed() -> None:
    snapshot = _snapshot(provenance_id="factual")
    common = {
        "snapshot": snapshot,
        "prompts": _prompts()[:1],
        "controls": _controls(),
        "proprioception": torch.randn(2, 7),
        "inference_noise": torch.randn(2, 5, 3),
        "arms": (factual_frozen_posterior_arm(snapshot),),
        "visibility_contracts": (FrozenPosteriorVisibility.DIRECT_ONLY,),
    }

    def mutating(request: FrozenPosteriorActionRequest) -> torch.Tensor:
        request.inference_noise.add_(1)
        return request.inference_noise

    with pytest.raises(RuntimeError, match="mutated its request"):
        run_frozen_posterior_action_diagnostic(mutating, **common)

    def wrong_shape(request: FrozenPosteriorActionRequest) -> torch.Tensor:
        return torch.zeros(request.inference_noise.shape[0], 1)

    with pytest.raises(ValueError, match="matching noise shape"):
        run_frozen_posterior_action_diagnostic(wrong_shape, **common)


def test_offline_target_scoring_occurs_only_after_label_free_forwards() -> None:
    snapshot = _snapshot(provenance_id="factual")
    row_arms = label_blind_visibility_removal_arms(snapshot)
    arms = (factual_frozen_posterior_arm(snapshot), *row_arms)
    readout = _PromptSelectiveActionReadout()
    result = run_frozen_posterior_action_diagnostic(
        readout,
        snapshot=snapshot,
        prompts=_prompts(),
        controls=_controls(),
        proprioception=torch.randn(2, 7),
        inference_noise=torch.randn(2, 5, 3),
        arms=arms,
        visibility_contracts=(FrozenPosteriorVisibility.DIRECT_ONLY,),
    )
    forward_count = len(readout.requests)
    scores = score_offline_row_selectivity(
        result,
        factual_arm_name="factual",
        row_arms=row_arms,
        targets=(
            OfflinePromptTargetRows("prompt-blue", torch.tensor([1, 1])),
            OfflinePromptTargetRows("prompt-red", torch.tensor([2, 2])),
        ),
        visibility=FrozenPosteriorVisibility.DIRECT_ONLY,
    )
    assert len(readout.requests) == forward_count
    assert len(scores) == 2
    for score in scores:
        assert (score.target_effect_rms > score.control_effect_rms).all()
        assert (score.target_to_control_ratio > 2).all()
        torch.testing.assert_close(score.effective_row_count, torch.ones(2))
    switch = score_offline_prompt_switch(
        scores,
        prompt_a="prompt-blue",
        prompt_b="prompt-red",
    )
    assert switch.mean_difference_in_differences > 0
    assert (switch.per_sample_difference_in_differences > 0).all()
