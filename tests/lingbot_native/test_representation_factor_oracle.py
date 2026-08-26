from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationItem,
    RepresentationEvaluationPlan,
    build_representation_evaluation_sample,
    build_representation_evaluation_snapshot,
    build_representation_ownership_row,
    build_representation_token_evidence,
    representation_target_mass_sha256,
    summarize_representation_ownership_rows,
)
from picf_next.lingbot_native.representation_factor_oracle import (
    FACTOR_ORACLE_CORNERS,
    LEARNED_S_LEARNED_PI,
    LEARNED_S_ORACLE_PI,
    ORACLE_S_LEARNED_PI,
    ORACLE_S_ORACLE_PI,
    REPRESENTATION_FACTOR_ORACLE_SCOPE,
    build_representation_factor_oracle,
    validate_representation_factor_oracle,
    write_representation_factor_oracle,
)
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
)
from picf_next.lingbot_native.task_diagnostics import build_task_row_diagnostics
from picf_next.objective import UnifiedObjective


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _plan() -> RepresentationEvaluationPlan:
    items = []
    for partition in ("heldout", "validation"):
        samples = (f"{partition}-a", f"{partition}-b", f"{partition}-c")
        tasks = ("task-a", "task-b", "task-c")
        targets = ("object-a", "object-b", "object-c")
        for ordinal, (sample, task, target) in enumerate(zip(samples, tasks, targets, strict=True)):
            task_control = (ordinal + 1) % len(samples)
            target_control = (ordinal + 2) % len(samples)
            items.append(
                RepresentationEvaluationItem(
                    partition=partition,
                    ordinal=ordinal,
                    rank=ordinal % 2,
                    task_key=task,
                    segment_index=ordinal,
                    source_episode_index=ordinal + (0 if partition == "heldout" else 10),
                    source_global_index=ordinal,
                    sample_key=sample,
                    shuffled_task_sample_key=samples[task_control],
                    shuffled_target_sample_key=samples[target_control],
                    factual_target_identity_keys=(target,),
                    shuffled_task_target_identity_keys=(targets[task_control],),
                    shuffled_target_target_identity_keys=(targets[target_control],),
                    factual_task_instruction_sha256=_sha(tasks[ordinal]),
                    shuffled_task_instruction_sha256=_sha(tasks[task_control]),
                    shuffled_target_instruction_sha256=_sha(tasks[target_control]),
                )
            )
    return RepresentationEvaluationPlan(
        representation_split_sha256="1" * 64,
        items=tuple(items),
    )


def _diagnostic(
    target_key: str,
    *,
    task_logits: tuple[float, float],
) -> dict[str, object]:
    support = torch.zeros(1, 1, 2, 2)
    predictions = NativeSequencePredictions(
        support_logits=support,
        ownership=torch.softmax(
            torch.cat((support, torch.zeros(1, 1, 2, 1)), dim=-1),
            dim=-1,
        ),
        existence_logits=torch.zeros(1, 1, 2),
        task_relevance_logits=torch.tensor([task_logits]),
        dense_task_grounding_logits=torch.zeros(1, 1, 2),
    )
    targets = NativeSequenceTargets(
        masks=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        mask_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.tensor([[1.0, 0.0]]),
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
        assignment=SequenceAssignment(torch.tensor([[1, 0]])),
        track_identity_keys_by_batch=((target_key, "context"),),
        row_bindings_by_batch=((("context", 0), (target_key, 1)),),
        predictive_terms=(),
        structural_terms=(),
    )
    return build_task_row_diagnostics(objective)[0]


def _sample(
    item: RepresentationEvaluationItem,
    *,
    checkpoint_global_step: int = 0,
    task_logits: tuple[float, float] = (-8.0, 8.0),
    target_prediction: tuple[float, float] = (1.0, 0.0),
    context_prediction: tuple[float, float] = (0.0, 1.0),
    target_mask: tuple[float, float] = (1.0, 0.0),
    omit_target_row: bool = False,
    mismatch_binding: bool = False,
) -> dict[str, object]:
    target_key = item.factual_target_identity_keys[0]
    diagnostic = _diagnostic(target_key, task_logits=task_logits)
    target_row_index = 0 if mismatch_binding else 1
    target_row = build_representation_ownership_row(
        row_index=target_row_index,
        track_index=0,
        identity_key=target_key,
        is_task_target=True,
        prediction=target_prediction,
        target=target_mask,
        weight=(1.0, 1.0),
    )
    context_row = build_representation_ownership_row(
        row_index=0 if not mismatch_binding else 1,
        track_index=1,
        identity_key="context",
        is_task_target=False,
        prediction=context_prediction,
        target=(0.0, 1.0),
        weight=(1.0, 1.0),
    )
    rows = (context_row,) if omit_target_row else (target_row, context_row)
    token = build_representation_token_evidence((1.0, -1.0), (1.0, 0.0))
    shuffled = build_representation_token_evidence((-1.0, 1.0), (1.0, 0.0))
    shuffled_target_mass = (0.0, 1.0)
    visual = {
        "schema": "picf-next.lingbot-native-relation-visual.v5",
        "path": f"visuals/{item.partition}/{item.ordinal}.png",
        "sha256": "2" * 64,
        "bytes": 100,
        "global_step": checkpoint_global_step,
        "input_weight_global_step": checkpoint_global_step,
        "weight_boundary": "checkpoint_evaluation",
        "rank": item.rank,
        "sample_key": item.sample_key,
        "task": item.task_key,
        "loss_only_labels_visible_to_model": False,
    }
    return build_representation_evaluation_sample(
        checkpoint_global_step=checkpoint_global_step,
        item=item,
        factual_task_instruction_sha256=item.factual_task_instruction_sha256,
        shuffled_task_instruction_sha256=item.shuffled_task_instruction_sha256,
        shuffled_target_instruction_sha256=item.shuffled_target_instruction_sha256,
        factual_token_evidence=token,
        shuffled_task_token_evidence=shuffled,
        shuffled_target_token_evidence=build_representation_token_evidence(
            (1.0, -1.0), shuffled_target_mass
        ),
        factual_task_row_diagnostic=diagnostic,
        shuffled_task_row_diagnostic=diagnostic,
        factual_ownership_rows=rows,
        factual_ownership_summary=summarize_representation_ownership_rows(rows),
        shuffled_task_ownership_rows=rows,
        shuffled_task_ownership_summary=summarize_representation_ownership_rows(rows),
        official_action_loss=0.2,
        factual_forward_seconds=1.0,
        shuffled_task_forward_seconds=1.0,
        peak_cuda_reserved_bytes=1024,
        factual_relation_sha256="3" * 64,
        factual_target_sha256="4" * 64,
        shuffled_task_relation_sha256="5" * 64,
        shuffled_task_target_sha256="4" * 64,
        shuffled_target_target_sha256=representation_target_mass_sha256(
            item.shuffled_target_target_identity_keys,
            shuffled_target_mass,
        ),
        visual_artifact=visual,
    )


def _snapshot(
    *,
    checkpoint_global_step: int = 0,
    task_logits: tuple[float, float] = (-8.0, 8.0),
    target_prediction: tuple[float, float] = (1.0, 0.0),
    context_prediction: tuple[float, float] = (0.0, 1.0),
    stratified_target_area: bool = False,
    ownership_blend: float | None = None,
    omit_target_rows: bool = False,
    mismatch_binding: bool = False,
) -> tuple[RepresentationEvaluationPlan, dict[str, object]]:
    plan = _plan()
    target_masks = {
        "task-a": (0.01, 0.0),
        "task-b": (0.06, 0.0),
        "task-c": (1.0, 0.0),
    }
    samples = []
    for item in plan.items:
        target_mask = target_masks[item.task_key] if stratified_target_area else (1.0, 0.0)
        if ownership_blend is None:
            selected_target_prediction = target_prediction
            selected_context_prediction = context_prediction
        else:
            selected_target_prediction = tuple(
                ownership_blend * expected + (1.0 - ownership_blend) * 0.5
                for expected in target_mask
            )
            selected_context_prediction = (
                (1.0 - ownership_blend) * 0.5,
                ownership_blend + (1.0 - ownership_blend) * 0.5,
            )
        samples.append(
            _sample(
                item,
                checkpoint_global_step=checkpoint_global_step,
                task_logits=task_logits,
                target_prediction=selected_target_prediction,
                context_prediction=selected_context_prediction,
                target_mask=target_mask,
                omit_target_row=omit_target_rows,
                mismatch_binding=mismatch_binding,
            )
        )
    snapshot = build_representation_evaluation_snapshot(
        checkpoint_global_step=checkpoint_global_step,
        implementation_sha256="6" * 64,
        model_family_sha256="7" * 64,
        representation_split_sha256=plan.representation_split_sha256,
        representation_evaluation_plan=plan,
        representation_frozen_action_state_sha256="8" * 64,
        samples=samples,
    )
    return plan, snapshot


def test_factor_oracle_is_conditional_recomputed_and_shapley_closes() -> None:
    plan, snapshot = _snapshot(
        task_logits=(0.0, 0.0),
        target_prediction=(0.7, 0.3),
        context_prediction=(0.3, 0.7),
    )
    artifact = build_representation_factor_oracle(
        snapshot,
        plan=plan,
        partition="heldout",
    )

    assert artifact["scope"] == REPRESENTATION_FACTOR_ORACLE_SCOPE
    assert artifact["supports_unconditional_scene_attribution"] is False
    assert artifact["summary"]["coverage"]["materialized_target_coverage"] == 1.0
    for sample in artifact["samples"]:
        for row in sample["rows"]:
            assert set(row["brier"]) == set(FACTOR_ORACLE_CORNERS)
            assert row["brier"][ORACLE_S_ORACLE_PI] == 0.0
            assert row["semantic_shapley"] + row["ownership_shapley"] == pytest.approx(
                row["total_excess_brier"]
            )
    assert (
        validate_representation_factor_oracle(
            artifact,
            snapshot=snapshot,
            plan=plan,
        )
        == artifact
    )


def test_factor_oracle_separates_semantic_and_ownership_errors() -> None:
    semantic_plan, semantic_snapshot = _snapshot(task_logits=(0.0, 0.0))
    semantic = build_representation_factor_oracle(
        semantic_snapshot,
        plan=semantic_plan,
        partition="heldout",
    )
    semantic_target = semantic["samples"][0]["rows"][0]
    assert semantic_target["brier"][ORACLE_S_LEARNED_PI] == 0.0
    assert semantic_target["brier"][LEARNED_S_ORACLE_PI] > 0.0

    ownership_plan, ownership_snapshot = _snapshot(
        target_prediction=(0.5, 0.5),
        context_prediction=(0.5, 0.5),
    )
    ownership = build_representation_factor_oracle(
        ownership_snapshot,
        plan=ownership_plan,
        partition="heldout",
    )
    ownership_target = ownership["samples"][0]["rows"][0]
    assert ownership_target["brier"][ORACLE_S_LEARNED_PI] > 0.0
    assert (
        ownership_target["brier"][LEARNED_S_ORACLE_PI]
        < ownership_target["brier"][ORACLE_S_LEARNED_PI]
    )
    assert ownership_target["brier"][LEARNED_S_LEARNED_PI] > 0.0


def test_factor_oracle_rejects_row_binding_mismatch() -> None:
    plan, snapshot = _snapshot(mismatch_binding=True)
    with pytest.raises(ValueError, match="row/track binding differs"):
        build_representation_factor_oracle(
            snapshot,
            plan=plan,
            partition="heldout",
        )


def test_factor_oracle_publication_is_exclusive(tmp_path: Path) -> None:
    plan, snapshot = _snapshot()
    artifact = build_representation_factor_oracle(
        snapshot,
        plan=plan,
        partition="heldout",
    )
    output = tmp_path / "factor-oracle.json"
    write_representation_factor_oracle(output, artifact)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        write_representation_factor_oracle(output, artifact)
