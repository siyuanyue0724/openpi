from __future__ import annotations

import inspect
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.representation_evaluation import RepresentationEvaluationItem
from picf_next.lingbot_native.representation_evaluation_runtime import (
    RepresentationActionDiagnosticGuard,
    _distributed_action_diagnostic_transaction,
    _evaluation_forward_seed,
    _evaluation_history_seed,
    _instruction_for_sample,
    _reconstruct_evaluation_prior,
    _seed_evaluation_forward,
    _target_mass_for_identities,
    _validate_matched_task_control_inputs,
    build_representation_runtime_evidence,
    native_relation_output_sha256,
    native_sequence_targets_sha256,
    run_read_only_representation_action_diagnostic,
    run_representation_checkpoint_evaluation,
)
from picf_next.lingbot_native.representation_stage import (
    configure_native_representation_parameter_scope,
    native_representation_frozen_action_state_sha256,
)
from picf_next.lingbot_native.state import NativePosteriorState
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
)
from picf_next.objective import UnifiedObjective


def _objective() -> NativeCALVINObjectiveResult:
    support_logits = torch.tensor([[[[2.0, -1.0], [-1.0, 2.0], [0.0, 0.0]]]])
    ownership = torch.softmax(
        torch.cat((support_logits, torch.zeros(1, 1, 3, 1)), dim=-1),
        dim=-1,
    )
    predictions = NativeSequencePredictions(
        support_logits=support_logits,
        ownership=ownership,
        existence_logits=torch.zeros(1, 1, 2),
        task_relevance_logits=torch.tensor([[2.0, -1.0]]),
        dense_task_grounding_logits=torch.tensor([[[2.0, -1.0, -0.5]]]),
    )
    targets = NativeSequenceTargets(
        masks=torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]),
        mask_valid=torch.ones(1, 1, 2, 3, dtype=torch.bool),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.tensor([[1.0, 0.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 1, 3),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    return NativeCALVINObjectiveResult(
        objective=UnifiedObjective(
            total=torch.zeros(()),
            normalized_terms={},
            valid_counts={},
        ),
        predictions=predictions,
        targets=targets,
        assignment=SequenceAssignment(torch.tensor([[0, 1]])),
        track_identity_keys_by_batch=(("target", "context"),),
        row_bindings_by_batch=((("target", 0), ("context", 1)),),
        predictive_terms=(),
        structural_terms=(),
    )


def test_warm_prior_replay_is_ordered_detached_and_observation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[object, NativePosteriorState | None, bool]] = []
    observed_bindings: list[tuple[tuple[str, int], ...]] = []

    def fake_context(batch, *, previous_state):
        observed.append((batch, previous_state, torch.is_grad_enabled()))
        return SimpleNamespace(previous_state=previous_state)

    def fake_forward(_policy, *, model_inputs, context):
        del model_inputs
        previous = context.previous_state
        value = 1.0 if previous is None else float(previous.rows.item()) + 1.0
        return SimpleNamespace(
            posterior_state=NativePosteriorState(torch.tensor([[[value]]], requires_grad=True))
        )

    monkeypatch.setattr(
        "picf_next.lingbot_native.representation_evaluation_runtime.build_native_calvin_context",
        fake_context,
    )
    monkeypatch.setattr(
        "picf_next.lingbot_native.representation_evaluation_runtime."
        "run_native_policy_observation_diagnostic_forward",
        fake_forward,
    )
    monkeypatch.setattr(
        "picf_next.lingbot_native.representation_evaluation_runtime._evaluation_objective",
        lambda _context, _batch, *, prior_row_bindings, **_kwargs: (
            observed_bindings.append(prior_row_bindings)
            or SimpleNamespace(row_bindings_by_batch=((("object/a", 0),),))
        ),
    )
    batches = tuple(
        SimpleNamespace(
            model_inputs={"index": index},
            routing=SimpleNamespace(batch_size=1),
            controls=SimpleNamespace(
                reset=torch.tensor([[index in {0, 4}]]),
                token_valid=torch.ones(1, 1, dtype=torch.bool),
            ),
        )
        for index in range(8)
    )

    state, row_bindings = _reconstruct_evaluation_prior(
        nn.Linear(1, 1),
        history_batches=batches,
        physical_sidecar=object(),  # type: ignore[arg-type]
        capacity=2,
        task_identity_resolver=object(),  # type: ignore[arg-type]
        patch_size=14,
        merge_size=2,
        structural_config=object(),  # type: ignore[arg-type]
        minimum_supervised_fraction=0.5,
        capacity_seed=7,
    )

    assert state is not None
    assert state.rows.item() == 8.0
    assert not state.rows.requires_grad
    assert [item[0] for item in observed] == list(batches)
    assert all(not grad_enabled for _, _, grad_enabled in observed)
    assert observed[0][1] is None
    assert [float(previous.rows.item()) for _, previous, _ in observed[1:]] == list(range(1, 8))
    assert row_bindings == (("object/a", 0),)
    assert observed_bindings == [
        (),
        (("object/a", 0),),
        (("object/a", 0),),
        (("object/a", 0),),
        (),
        (("object/a", 0),),
        (("object/a", 0),),
        (("object/a", 0),),
    ]


@pytest.mark.parametrize("expected_transition_index", [0, 8])
def test_instruction_donor_must_match_the_planned_evaluation_age(
    expected_transition_index: int,
) -> None:
    dataset = SimpleNamespace(
        index=SimpleNamespace(
            segments=(SimpleNamespace(start=100, instruction="pick up the blue block"),)
        ),
        locator_by_key=lambda _key: SimpleNamespace(
            segment_index=0,
            global_index=100 + expected_transition_index,
        ),
    )

    instruction = _instruction_for_sample(
        dataset,
        "sample",
        expected_transition_index=expected_transition_index,
    )

    assert instruction == "pick up the blue block"


def test_instruction_donor_rejects_a_transition_from_another_evaluation_age() -> None:
    dataset = SimpleNamespace(
        index=SimpleNamespace(
            segments=(SimpleNamespace(start=100, instruction="pick up the blue block"),)
        ),
        locator_by_key=lambda _key: SimpleNamespace(segment_index=0, global_index=108),
    )

    with pytest.raises(ValueError, match="planned age"):
        _instruction_for_sample(
            dataset,
            "sample",
            expected_transition_index=0,
        )


def test_runtime_evidence_uses_only_observed_tokens_and_matched_rows() -> None:
    objective = _objective()
    evidence = build_representation_runtime_evidence(
        objective,
        structural_sensor_valid=torch.tensor([[True, True, False]]),
        batch_index=0,
    )
    assert evidence.token_evidence["logits"] == [2.0, -1.0]
    assert evidence.token_evidence["target_mass"] == [1.0, 0.0]
    assert evidence.task_row_diagnostic["target_rows"] == [0]
    assert [row["identity_key"] for row in evidence.ownership_rows] == [
        "target",
        "context",
    ]
    assert evidence.ownership_summary["task_target_row_count"] == 1
    assert evidence.target_sha256 == native_sequence_targets_sha256(objective.targets)
    assert evidence.target_mass_by_identity == {
        "target": (1.0, 0.0),
        "context": (0.0, 1.0),
    }
    assert _target_mass_for_identities(evidence, ("context",)) == (0.0, 1.0)
    assert _target_mass_for_identities(evidence, ()) == (0.0, 0.0)
    with pytest.raises(ValueError, match="absent from the scene"):
        _target_mass_for_identities(evidence, ("missing",))


def test_runtime_evidence_retains_sample_without_exact_task_target() -> None:
    objective = _objective()
    targets = replace(
        objective.targets,
        task_relevance=torch.zeros_like(objective.targets.task_relevance),
    )
    objective = replace(objective, targets=targets)

    evidence = build_representation_runtime_evidence(
        objective,
        structural_sensor_valid=torch.tensor([[True, True, False]]),
        batch_index=0,
    )

    assert evidence.token_evidence["target_mass"] == [0.0, 0.0]
    assert evidence.token_evidence["metrics"]["eligible"] is False
    assert evidence.task_row_diagnostic["target_rows"] == []
    assert [row["identity_key"] for row in evidence.ownership_rows] == [
        "target",
        "context",
    ]
    assert all(row["is_task_target"] is False for row in evidence.ownership_rows)
    assert evidence.ownership_summary["task_target_row_count"] == 0
    assert evidence.ownership_summary["target_soft_iou"] is None
    assert evidence.ownership_summary["target_mass_concentration"] is None
    assert evidence.ownership_summary["macro_soft_iou"] is not None
    assert evidence.target_sha256 == native_sequence_targets_sha256(targets)


def test_runtime_tensor_hashes_cover_relation_and_target_values() -> None:
    objective = _objective()
    predictions = objective.predictions
    relation = RelationOutput(
        support_logits=predictions.support_logits[:, 0],
        visible_support=predictions.support_logits[:, 0].sigmoid(),
        ownership=predictions.ownership[:, 0],
        task_relevance=predictions.task_relevance_logits.sigmoid(),
        task_relevance_logits=predictions.task_relevance_logits,
        task_embedding=torch.ones(1, 4),
        row_embeddings=torch.ones(1, 2, 4),
        relation_temperature=torch.ones(1),
        dense_task_grounding=predictions.dense_task_grounding_logits[:, 0].sigmoid(),
        dense_task_grounding_logits=predictions.dense_task_grounding_logits[:, 0],
        existence=predictions.existence_logits[:, 0].sigmoid(),
        existence_logits=predictions.existence_logits[:, 0],
        sensor_valid=torch.ones(1, 3, dtype=torch.bool),
        structural_sensor_valid=torch.tensor([[True, True, False]]),
    )
    baseline = native_relation_output_sha256(relation)
    changed = replace(relation, support_logits=relation.support_logits + 1)
    assert native_relation_output_sha256(changed) != baseline

    target_baseline = native_sequence_targets_sha256(objective.targets)
    changed_targets = replace(
        objective.targets,
        existence=objective.targets.existence * 0,
    )
    assert native_sequence_targets_sha256(changed_targets) != target_baseline


def test_evaluation_forward_seed_is_checkpoint_independent_and_sample_bound() -> None:
    item = RepresentationEvaluationItem(
        partition="heldout",
        ordinal=0,
        rank=0,
        task_key="task-a",
        segment_index=1,
        source_episode_index=2,
        source_global_index=3,
        sample_key="sample-a",
        shuffled_task_sample_key="sample-b",
        shuffled_target_sample_key="sample-c",
        factual_target_identity_keys=("target-a",),
        shuffled_task_target_identity_keys=("target-b",),
        shuffled_target_target_identity_keys=("target-c",),
        factual_task_instruction_sha256="1" * 64,
        shuffled_task_instruction_sha256="2" * 64,
        shuffled_target_instruction_sha256="3" * 64,
    )
    plan = SimpleNamespace(replay_seed_sha256="4" * 64)
    seed = _evaluation_forward_seed(plan, item)
    assert seed == _evaluation_forward_seed(plan, item)
    assert seed != _evaluation_forward_seed(plan, replace(item, ordinal=1, rank=1))


def test_evaluation_history_seed_masks_the_frozen_rank_one_high_bit() -> None:
    forward_seed = 1_317_222_833_225_136_803

    history_seed = _evaluation_history_seed(forward_seed)

    assert history_seed == 6_037_256_957_419_752_467
    assert 0 <= history_seed < 2**63


@pytest.mark.parametrize("seed", [True, -1, 2**63, "7"])
def test_evaluation_history_seed_rejects_invalid_source_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="source forward seed"):
        _evaluation_history_seed(seed)  # type: ignore[arg-type]


def test_evaluation_forward_seed_changes_only_cpu_and_selected_cuda_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_seeds: list[int] = []
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "manual_seed", cuda_seeds.append)
    seed = 741
    expected = torch.Generator(device="cpu").manual_seed(seed).get_state()

    _seed_evaluation_forward(seed, device=torch.device("cuda", 1))

    assert torch.equal(torch.get_rng_state(), expected)
    assert cuda_seeds == [seed]


def test_action_transaction_does_not_enter_a_new_collective_after_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []

    class FakeGuard:
        def __init__(self, _policy, *, expected_scope):
            events.append(("init", expected_scope))

        def __enter__(self):
            events.append(("guard", "enter"))
            return self

        def close(self) -> None:
            events.append(("guard", "close"))

    def fake_phase_error(**kwargs) -> None:
        events.append(("phase", kwargs["phase"]))

    monkeypatch.setattr(
        "picf_next.lingbot_native.representation_evaluation_runtime."
        "RepresentationActionDiagnosticGuard",
        FakeGuard,
    )
    monkeypatch.setattr(
        "picf_next.lingbot_native.representation_evaluation_runtime._distributed_phase_error",
        fake_phase_error,
    )

    with (
        pytest.raises(
            RuntimeError,
            match="body failed",
        ),
        _distributed_action_diagnostic_transaction(
            nn.Linear(1, 1),
            expected_scope=object(),  # type: ignore[arg-type]
            rank=0,
            world_size=2,
            dist_module=object(),
        ),
    ):
        raise RuntimeError("body failed")

    assert ("guard", "close") in events
    assert ("phase", "action-transaction-enter") in events
    assert ("phase", "action-transaction-close") not in events


def test_action_transaction_synchronizes_a_successful_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases: list[str] = []

    class FakeGuard:
        def __init__(self, _policy, *, expected_scope):
            del expected_scope

        def __enter__(self):
            return self

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "picf_next.lingbot_native.representation_evaluation_runtime."
        "RepresentationActionDiagnosticGuard",
        FakeGuard,
    )
    monkeypatch.setattr(
        "picf_next.lingbot_native.representation_evaluation_runtime._distributed_phase_error",
        lambda **kwargs: phases.append(kwargs["phase"]),
    )

    with _distributed_action_diagnostic_transaction(
        nn.Linear(1, 1),
        expected_scope=object(),  # type: ignore[arg-type]
        rank=0,
        world_size=2,
        dist_module=object(),
    ):
        pass

    assert phases == ["action-transaction-enter", "action-transaction-close"]


def test_standard_task_control_uses_the_same_full_action_forward() -> None:
    source = inspect.getsource(run_representation_checkpoint_evaluation)

    assert source.count("_run_action_evaluation_forward(") == 2
    assert "_run_observation_evaluation_forward(" not in source


def _matched_control_fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    factual = SimpleNamespace(
        controls=(torch.tensor([[1.0, 2.0]]),),
        routing=("sample", 0),
        structural_target_requests=("target",),
        source_digest="a" * 64,
        modalities={"touch": torch.tensor([[0.25, 0.75]])},
        model_inputs={
            "images": torch.ones(1, 2),
            "lang_masks": torch.tensor([[True, True]]),
            "lang_tokens": torch.tensor([[1, 2]]),
        },
    )
    shuffled = SimpleNamespace(
        controls=(factual.controls[0].clone(),),
        routing=factual.routing,
        structural_target_requests=factual.structural_target_requests,
        source_digest=factual.source_digest,
        modalities={"touch": factual.modalities["touch"].clone()},
        model_inputs={
            "images": factual.model_inputs["images"].clone(),
            "lang_masks": factual.model_inputs["lang_masks"].clone(),
            "lang_tokens": torch.tensor([[3, 4]]),
        },
    )
    return factual, shuffled


def test_matched_task_control_accepts_only_language_token_changes() -> None:
    factual, shuffled = _matched_control_fixture()
    _validate_matched_task_control_inputs(factual, shuffled)

    shuffled.modalities["touch"][0, 0] = 0.5
    with pytest.raises(ValueError, match="non-language contracts"):
        _validate_matched_task_control_inputs(factual, shuffled)


def test_matched_task_control_requires_an_actual_language_change() -> None:
    factual, shuffled = _matched_control_fixture()
    shuffled.model_inputs["lang_tokens"].copy_(factual.model_inputs["lang_tokens"])
    with pytest.raises(ValueError, match="retained tokenized language"):
        _validate_matched_task_control_inputs(factual, shuffled)


class _ActionExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("e_score_correction_bias", torch.tensor([0.2, -0.1]))
        self.register_buffer("tokens_per_expert", torch.tensor([3.0, 4.0]))
        self.register_buffer("last_tokens_per_expert", torch.tensor([1.0, 2.0]))
        self.register_buffer("avg_topk_sigmoid_score", torch.tensor([0.3]))


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(1, 1)
        self.model = nn.Module()
        self.model.qwenvl_with_expert = nn.Module()
        self.model.qwenvl_with_expert.qwen_expert = _ActionExpert()
        self.model.state_proj = nn.Linear(1, 1)
        self.model.action_in_proj = nn.Linear(1, 1)
        self.model.action_out_proj = nn.Linear(1, 1)
        self.model.action_time_mlp_in = nn.Linear(1, 1)
        self.model.action_time_mlp_out = nn.Linear(1, 1)


def test_read_only_action_diagnostic_restores_official_moe_runtime_buffers() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)
    before = native_representation_frozen_action_state_sha256(policy, expected=scope)

    def forward(
        observed_policy: _Policy,
        *,
        model_inputs: object,
        context: object,
    ) -> str:
        assert model_inputs == {"input": "factual"}
        assert context == "reset"
        expert = observed_policy.model.qwenvl_with_expert.qwen_expert
        expert.tokens_per_expert.add_(5)
        expert.avg_topk_sigmoid_score.fill_(0.8)
        return "official"

    diagnostic = run_read_only_representation_action_diagnostic(
        policy,
        expected_scope=scope,
        model_inputs={"input": "factual"},
        context="reset",
        forward=forward,
    )
    assert diagnostic.result == "official"
    assert diagnostic.changed_buffer_names == (
        "model.qwenvl_with_expert.qwen_expert.avg_topk_sigmoid_score",
        "model.qwenvl_with_expert.qwen_expert.tokens_per_expert",
    )
    assert diagnostic.action_state_sha256 == before
    assert native_representation_frozen_action_state_sha256(policy, expected=scope) == before


def test_read_only_action_diagnostic_accepts_version_only_parameter_change() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)
    before = native_representation_frozen_action_state_sha256(policy, expected=scope)

    def forward(
        observed_policy: _Policy,
        *,
        model_inputs: object,
        context: object,
    ) -> str:
        del model_inputs, context
        parameter = observed_policy.model.action_out_proj.weight
        with torch.no_grad():
            parameter.copy_(parameter)
        return "official"

    diagnostic = run_read_only_representation_action_diagnostic(
        policy,
        expected_scope=scope,
        model_inputs={},
        context=None,
        forward=forward,
    )

    assert diagnostic.result == "official"
    assert diagnostic.content_unchanged_parameter_version_names == ("model.action_out_proj.weight",)
    assert native_representation_frozen_action_state_sha256(policy, expected=scope) == before


def test_read_only_action_diagnostic_rejects_parameter_content_change() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)

    def forward(
        observed_policy: _Policy,
        *,
        model_inputs: object,
        context: object,
    ) -> None:
        del model_inputs, context
        parameter = observed_policy.model.action_out_proj.weight
        with torch.no_grad():
            parameter.add_(1)

    guard = RepresentationActionDiagnosticGuard(
        policy,
        expected_scope=scope,
        forward=forward,
    )
    guard.__enter__()
    try:
        with pytest.raises(RuntimeError, match="non-ephemeral action state"):
            guard.run(model_inputs={}, context=None)
    finally:
        with torch.no_grad():
            policy.model.action_out_proj.weight.sub_(1)
        guard.close()


def test_read_only_action_diagnostic_rejects_and_restores_routing_bias_mutation() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)
    before = native_representation_frozen_action_state_sha256(policy, expected=scope)

    def forward(
        observed_policy: _Policy,
        *,
        model_inputs: object,
        context: object,
    ) -> None:
        del model_inputs, context
        observed_policy.model.qwenvl_with_expert.qwen_expert.e_score_correction_bias.add_(1)

    with pytest.raises(
        RuntimeError,
        match="non-ephemeral action state",
    ):
        run_read_only_representation_action_diagnostic(
            policy,
            expected_scope=scope,
            model_inputs={},
            context=None,
            forward=forward,
        )
    assert native_representation_frozen_action_state_sha256(policy, expected=scope) == before


def test_action_diagnostic_guard_amortizes_multiple_forwards_and_restores_buffers() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)
    before = native_representation_frozen_action_state_sha256(policy, expected=scope)
    calls = 0

    def forward(
        observed_policy: _Policy,
        *,
        model_inputs: object,
        context: object,
    ) -> tuple[object, object]:
        nonlocal calls
        calls += 1
        expert = observed_policy.model.qwenvl_with_expert.qwen_expert
        expert.tokens_per_expert.add_(calls)
        expert.avg_topk_sigmoid_score.fill_(0.5 + calls / 10)
        return model_inputs, context

    with RepresentationActionDiagnosticGuard(
        policy,
        expected_scope=scope,
        forward=forward,
    ) as guard:
        first = guard.run(model_inputs={"sample": 1}, context="a")
        second = guard.run(model_inputs={"sample": 2}, context="b")
        assert first.result == ({"sample": 1}, "a")
        assert second.result == ({"sample": 2}, "b")
        assert first.action_state_sha256 == second.action_state_sha256 == before
    assert calls == 2
    assert native_representation_frozen_action_state_sha256(policy, expected=scope) == before


def test_action_diagnostic_guard_restores_buffers_after_forward_exception() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)
    before = native_representation_frozen_action_state_sha256(policy, expected=scope)

    def forward(
        observed_policy: _Policy,
        *,
        model_inputs: object,
        context: object,
    ) -> None:
        del model_inputs, context
        observed_policy.model.qwenvl_with_expert.qwen_expert.tokens_per_expert.add_(9)
        raise LookupError("diagnostic failed")

    with (
        pytest.raises(LookupError, match="diagnostic failed"),
        RepresentationActionDiagnosticGuard(
            policy,
            expected_scope=scope,
            forward=forward,
        ) as guard,
    ):
        guard.run(model_inputs={}, context=None)
    assert native_representation_frozen_action_state_sha256(policy, expected=scope) == before
