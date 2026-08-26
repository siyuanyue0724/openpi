from __future__ import annotations

import copy

import pytest
import torch

from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.relation_precision_audit import (
    build_relation_score_precision_audit,
    build_relation_score_precision_evidence,
    build_relation_score_precision_sample,
    fp32_task_relation_logits,
    validate_relation_score_precision_audit,
    validate_relation_score_precision_evidence,
)
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
)
from picf_next.objective import UnifiedObjective


def _relation_and_objective() -> tuple[RelationOutput, NativeCALVINObjectiveResult]:
    task = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    rows = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.50390625]]],
        dtype=torch.bfloat16,
    )
    temperature = torch.ones(1, dtype=torch.bfloat16)
    task_logits = torch.einsum("bd,bkd->bk", task, rows) / temperature
    support_logits = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    ownership = torch.softmax(
        torch.cat(
            (support_logits, torch.zeros(1, 2, 1, dtype=torch.bfloat16)),
            dim=-1,
        ),
        dim=-1,
    )
    relation = RelationOutput(
        support_logits=support_logits,
        visible_support=support_logits.sigmoid(),
        ownership=ownership,
        task_relevance=task_logits.sigmoid(),
        task_relevance_logits=task_logits,
        task_embedding=task,
        row_embeddings=rows,
        relation_temperature=temperature,
        dense_task_grounding=torch.full((1, 2), 0.5, dtype=torch.bfloat16),
        dense_task_grounding_logits=torch.zeros(1, 2, dtype=torch.bfloat16),
        existence=torch.full((1, 2), 0.5, dtype=torch.bfloat16),
        existence_logits=torch.zeros(1, 2, dtype=torch.bfloat16),
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    predictions = NativeSequencePredictions(
        support_logits=support_logits[:, None],
        ownership=ownership[:, None],
        existence_logits=relation.existence_logits[:, None],
        task_relevance_logits=task_logits,
        dense_task_grounding_logits=relation.dense_task_grounding_logits[:, None],
    )
    targets = NativeSequenceTargets(
        masks=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        mask_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.tensor([[0.0, 1.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 1, 2),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    objective = NativeCALVINObjectiveResult(
        objective=UnifiedObjective(
            total=torch.zeros(()),
            normalized_terms={},
            valid_counts={},
        ),
        predictions=predictions,
        targets=targets,
        assignment=SequenceAssignment(torch.tensor([[0, 1]])),
        track_identity_keys_by_batch=(("context", "target"),),
        row_bindings_by_batch=((("context", 0), ("target", 1)),),
        predictive_terms=(),
        structural_terms=(),
    )
    return relation, objective


def test_precision_evidence_exposes_bf16_tie_without_changing_embeddings() -> None:
    relation, objective = _relation_and_objective()

    production = relation.task_relevance_logits.detach().float()
    fp32 = fp32_task_relation_logits(relation)
    assert production.tolist() == [[1.0, 1.0]]
    assert fp32.tolist() == [[1.0, 1.00390625]]

    evidence = build_relation_score_precision_evidence(
        relation,
        objective,
        batch_index=0,
    )
    assert evidence["sequence_time_count"] == 1
    assert evidence["source_time"] == 0
    assert evidence["source_side"] == "posterior"
    assert evidence["source_phase"] == 1
    assert evidence["source_binding_valid"] == [True, True]
    assert evidence["matched_pair_collision_count"] == 1
    assert evidence["production_statistics"]["matched_row_unique_count"] == 1
    assert evidence["fp32_statistics"]["matched_row_unique_count"] == 2
    assert evidence["production_statistics"]["worst_target_optimistic_rank"] == 1
    assert evidence["production_statistics"]["worst_target_pessimistic_rank"] == 2
    assert evidence["fp32_statistics"]["worst_target_pessimistic_rank"] == 1
    assert evidence["fp32_statistics"]["target_vs_hardest_negative_logit_margin"] == pytest.approx(
        0.00390625
    )


def test_precision_evidence_and_audit_fail_closed_on_tampering() -> None:
    relation, objective = _relation_and_objective()
    evidence = build_relation_score_precision_evidence(
        relation,
        objective,
        batch_index=0,
    )
    changed = copy.deepcopy(evidence)
    changed["fp32_logits"][1] += 1.0
    with pytest.raises(ValueError, match="FP32 .* was not recomputed"):
        validate_relation_score_precision_evidence(changed)

    changed = copy.deepcopy(evidence)
    changed["source_binding_valid"][1] = False
    with pytest.raises(ValueError, match="source cut"):
        validate_relation_score_precision_evidence(changed)

    samples = [
        build_relation_score_precision_sample(
            sample_key=f"sample-{index}",
            partition="heldout",
            task_key="move_blue_block",
            factual_relation_sha256=f"{index + 1:x}" * 64,
            shuffled_task_relation_sha256=f"{index + 3:x}" * 64,
            factual=evidence,
            shuffled_task=evidence,
        )
        for index in range(2)
    ]
    audit = build_relation_score_precision_audit(
        checkpoint_global_step=20,
        implementation_sha256="a" * 64,
        model_family_sha256="b" * 64,
        representation_split_sha256="c" * 64,
        representation_evaluation_plan_sha256="d" * 64,
        expected_sample_keys=("sample-0", "sample-1"),
        samples=samples,
    )
    assert audit["summary"]["sample_count"] == 2
    assert audit["summary"]["factual_restored_matched_pair_count"] == 2
    assert (
        audit["summary"]["factual_mean_production_matched_unique_fraction"]
        < audit["summary"]["factual_mean_fp32_matched_unique_fraction"]
    )

    changed_audit = copy.deepcopy(audit)
    changed_audit["summary"]["factual_restored_matched_pair_count"] = 0
    with pytest.raises(ValueError, match="summary was not recomputed"):
        validate_relation_score_precision_audit(
            changed_audit,
            expected_sample_keys=("sample-0", "sample-1"),
        )
