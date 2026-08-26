from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

import picf_next.lingbot_native.task_relation as task_relation_module
from picf_next.lingbot_native.relations import RelationOutput, SharedRelationReadout
from picf_next.lingbot_native.supervision import (
    NativeSequenceTargets,
    SequenceAssignment,
)
from picf_next.lingbot_native.task_relation import (
    TaskRelationTargets,
    global_task_relation_term,
    host_native_factorized_task_relation_term,
    host_native_multi_positive_task_relation_term,
    host_native_multi_positive_task_score,
    identity_fingerprint,
    materialize_task_relation_targets,
    multi_positive_task_relation_values,
)


def _targets(
    *,
    task: tuple[float, ...] = (1.0, 0.0),
    censored: tuple[bool, ...] = (False, False),
) -> NativeSequenceTargets:
    tracks = len(task)
    return NativeSequenceTargets(
        masks=torch.zeros(1, 1, tracks, 2),
        mask_valid=torch.ones(1, 1, tracks, 2, dtype=torch.bool),
        existence=torch.ones(1, 1, tracks),
        existence_valid=torch.ones(1, 1, tracks, dtype=torch.bool),
        task_relevance=torch.tensor([task]),
        task_valid=torch.ones(1, tracks, dtype=torch.bool),
        track_valid=torch.ones(1, tracks, dtype=torch.bool),
        capacity_censored=torch.tensor([censored]),
        token_observed_fraction=torch.ones(1, 1, 2),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )


def _visible_targets(
    *,
    occluded: bool = False,
    task: tuple[float, float] = (1.0, 0.0),
) -> NativeSequenceTargets:
    masks = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    if occluded:
        masks.zero_()
    return NativeSequenceTargets(
        masks=masks,
        mask_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.tensor([task]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 1, 2),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )


def _relation(
    task_embedding: torch.Tensor,
    row_embeddings: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> RelationOutput:
    batch, rows, _width = row_embeddings.shape
    tokens = 2
    support = torch.zeros(batch, tokens, rows)
    sensor_valid = torch.ones(batch, tokens, dtype=torch.bool)
    task_logits = torch.einsum("bd,bkd->bk", task_embedding, row_embeddings) / temperature
    existence_logits = torch.zeros(batch, rows)
    return RelationOutput(
        support_logits=support,
        visible_support=support.sigmoid(),
        ownership=torch.softmax(
            torch.cat((support, torch.zeros(batch, tokens, 1)), dim=-1),
            dim=-1,
        ),
        task_relevance=task_logits.sigmoid(),
        task_relevance_logits=task_logits,
        task_embedding=task_embedding,
        row_embeddings=row_embeddings,
        relation_temperature=torch.tensor(
            [temperature],
            dtype=task_embedding.dtype,
            device=task_embedding.device,
        ),
        dense_task_grounding=torch.zeros(batch, tokens),
        dense_task_grounding_logits=torch.zeros(batch, tokens),
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        sensor_valid=sensor_valid,
    )


def _metadata(targets: TaskRelationTargets) -> torch.Tensor:
    batch, rows = targets.row_valid.shape
    return torch.cat(
        (
            targets.row_identity_fingerprints.reshape(batch, rows * 2),
            targets.target_identity_fingerprints.reshape(batch, rows * 2),
            targets.row_valid.to(torch.long),
            targets.target_valid.to(torch.long),
            targets.query_valid.to(torch.long).unsqueeze(1),
        ),
        dim=1,
    )


class _TwoRankGather:
    def __init__(self, *, remote_embeddings: torch.Tensor, remote_metadata: torch.Tensor) -> None:
        self.remote_embeddings = remote_embeddings
        self.remote_metadata = remote_metadata
        self.payload_call = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return True

    @staticmethod
    def get_world_size() -> int:
        return 2

    @staticmethod
    def get_rank() -> int:
        return 0

    def all_gather(self, outputs: list[torch.Tensor], value: torch.Tensor) -> None:
        outputs[0].copy_(value)
        if value.ndim == 1:
            outputs[1].copy_(value)
            return
        remote = self.remote_embeddings if self.payload_call == 0 else self.remote_metadata
        self.payload_call += 1
        outputs[1].copy_(remote)


def _gloo_task_relation_worker(rank: int, init_path: str, output_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        task = torch.tensor(
            [[(0.8, 0.2), (0.2, 0.8)][rank]],
            requires_grad=True,
        )
        rows = torch.tensor(
            [
                [
                    [(0.9, 0.1), (0.1, 0.9)],
                    [(0.8, 0.2), (0.2, 0.8)],
                ][rank]
            ],
            requires_grad=True,
        )
        term = global_task_relation_term(
            _relation(task, rows),
            _targets(task=((1.0, 0.0), (0.0, 1.0))[rank]),
            SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
            identity_keys_by_batch=(("red", "blue"),),
            weight=1.0,
            distributed=dist,
        )
        displayed = term.normalized()
        displayed.backward()

        torch.save(
            {
                "loss": displayed.detach(),
                "task_grad": task.grad,
                "row_grad": rows.grad,
            },
            Path(output_dir) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_identity_fingerprint_is_deterministic_nonzero_and_identity_specific() -> None:
    first = identity_fingerprint("movable/block_red")
    assert first == identity_fingerprint("movable/block_red")
    assert first != (0, 0)
    assert first != identity_fingerprint("movable/block_blue")


def test_materialized_relation_targets_follow_loss_side_assignment() -> None:
    actual = materialize_task_relation_targets(
        _targets(),
        SequenceAssignment(row_to_track=torch.tensor([[1, 0]])),
        identity_keys_by_batch=(("red", "blue"),),
    )
    assert actual.row_valid.tolist() == [[True, True]]
    assert actual.target_valid.tolist() == [[True, False]]
    assert actual.query_valid.tolist() == [True]
    assert tuple(actual.row_identity_fingerprints[0, 0].tolist()) == identity_fingerprint("blue")
    assert tuple(actual.row_identity_fingerprints[0, 1].tolist()) == identity_fingerprint("red")
    assert tuple(actual.target_identity_fingerprints[0, 0].tolist()) == identity_fingerprint("red")


def test_materialized_relation_targets_fail_closed_on_fingerprint_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        task_relation_module,
        "identity_fingerprint",
        lambda _identity: (1, 2),
    )
    with pytest.raises(RuntimeError, match="fingerprint collision"):
        materialize_task_relation_targets(
            _targets(),
            SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
            identity_keys_by_batch=(("red", "blue"),),
        )


def test_multi_positive_relation_requires_every_same_identity_instance() -> None:
    red = identity_fingerprint("red")
    blue = identity_fingerprint("blue")
    task = torch.tensor([[1.0, 0.0]], requires_grad=True)
    rows = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    task_embeddings = torch.cat((task, task.detach()), dim=0)
    relation_targets = TaskRelationTargets(
        row_identity_fingerprints=torch.tensor([[red, blue], [red, blue]]),
        row_valid=torch.ones(2, 2, dtype=torch.bool),
        target_identity_fingerprints=torch.tensor([[red, (0, 0)], [red, (0, 0)]]),
        target_valid=torch.tensor([[True, False], [True, False]]),
        query_valid=torch.ones(2, dtype=torch.bool),
    )
    values, valid = multi_positive_task_relation_values(
        task_embeddings=task_embeddings,
        row_embeddings=rows,
        relation_temperature=torch.tensor([0.1]),
        targets=relation_targets,
    )
    assert valid.tolist() == [True, True]
    torch.testing.assert_close(values[0], values[1])
    values.mean().backward()
    assert task.grad is not None and task.grad.abs().sum() > 0
    assert rows.grad is not None and rows.grad.abs().sum() > 0
    assert rows.grad[0, 0].abs().sum() > 0
    assert rows.grad[1, 0].abs().sum() > 0


def test_relation_loss_is_invariant_to_row_permutation_with_identity_permutation() -> None:
    red = identity_fingerprint("red")
    blue = identity_fingerprint("blue")
    task = torch.tensor([[1.0, 0.0]])
    rows = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]])
    targets = TaskRelationTargets(
        row_identity_fingerprints=torch.tensor([[red, blue]]),
        row_valid=torch.ones(1, 2, dtype=torch.bool),
        target_identity_fingerprints=torch.tensor([[red, (0, 0)]]),
        target_valid=torch.tensor([[True, False]]),
        query_valid=torch.ones(1, dtype=torch.bool),
    )
    factual, _ = multi_positive_task_relation_values(
        task_embeddings=task,
        row_embeddings=rows,
        relation_temperature=torch.tensor([0.1]),
        targets=targets,
    )
    permuted, _ = multi_positive_task_relation_values(
        task_embeddings=task,
        row_embeddings=rows[:, [1, 0]],
        relation_temperature=torch.tensor([0.1]),
        targets=replace(
            targets,
            row_identity_fingerprints=targets.row_identity_fingerprints[:, [1, 0]],
            row_valid=targets.row_valid[:, [1, 0]],
        ),
    )
    torch.testing.assert_close(factual, permuted)


def test_unknown_or_absent_positive_query_is_invalid_not_a_false_negative() -> None:
    red = identity_fingerprint("red")
    blue = identity_fingerprint("blue")
    values, valid = multi_positive_task_relation_values(
        task_embeddings=torch.tensor([[1.0, 0.0]]),
        row_embeddings=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        relation_temperature=torch.tensor([0.1]),
        targets=TaskRelationTargets(
            row_identity_fingerprints=torch.tensor([[blue, (0, 0)]]),
            row_valid=torch.tensor([[True, False]]),
            target_identity_fingerprints=torch.tensor([[red, (0, 0)]]),
            target_valid=torch.tensor([[True, False]]),
            query_valid=torch.tensor([True]),
        ),
    )
    assert valid.tolist() == [False]
    torch.testing.assert_close(values, torch.zeros_like(values))


def test_capacity_censored_target_without_global_positive_is_ignored() -> None:
    relation_targets = materialize_task_relation_targets(
        _targets(censored=(True, False)),
        SequenceAssignment(row_to_track=torch.tensor([[-1, 1]])),
        identity_keys_by_batch=(("red", "blue"),),
    )
    values, valid = multi_positive_task_relation_values(
        task_embeddings=torch.tensor([[1.0, 0.0]]),
        row_embeddings=torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]),
        relation_temperature=torch.tensor([0.1]),
        targets=relation_targets,
    )
    assert valid.tolist() == [False]
    torch.testing.assert_close(values, torch.zeros_like(values))


def test_identity_targets_change_only_loss_side_not_relation_output() -> None:
    task = torch.tensor([[1.0, 0.0]], requires_grad=True)
    rows = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    relation = _relation(task, rows)
    relation_snapshot = tuple(
        value.detach().clone()
        for value in (
            relation.task_embedding,
            relation.row_embeddings,
            relation.task_relevance_logits,
        )
    )
    red = global_task_relation_term(
        relation,
        _targets(task=(1.0, 0.0)),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        identity_keys_by_batch=(("red", "blue"),),
        weight=1.0,
    )
    blue = global_task_relation_term(
        relation,
        _targets(task=(0.0, 1.0)),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        identity_keys_by_batch=(("red", "blue"),),
        weight=1.0,
    )
    assert red.normalized() < blue.normalized()
    for before, after in zip(
        relation_snapshot,
        (
            relation.task_embedding,
            relation.row_embeddings,
            relation.task_relevance_logits,
        ),
        strict=True,
    ):
        torch.testing.assert_close(before, after)


def test_host_match_relation_rejects_global_retrieval_and_trains_scalar_projection() -> None:
    readout = SharedRelationReadout(4, temperature_init=0.1)
    relation = readout(
        posterior_rows=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        sensor_hidden=torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]),
        sensor_valid=torch.ones(1, 1, dtype=torch.bool),
        match_hidden=torch.tensor([[[0.6, 0.4, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0]]]),
    )
    with pytest.raises(ValueError, match="global embedding retrieval is forbidden"):
        global_task_relation_term(
            relation,
            _targets(task=(1.0, 0.0)),
            SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
            identity_keys_by_batch=(("red", "blue"),),
            weight=1.0,
        )
    torch.nn.functional.binary_cross_entropy_with_logits(
        relation.task_relevance_logits,
        torch.tensor([[1.0, 0.0]]),
    ).backward()
    gradient = readout.match_projection.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    assert readout.projection.weight.grad is None
    assert readout.temperature_parameter.grad is None


def test_host_native_task_score_is_competitive_shift_invariant_and_hard_negative_driven() -> None:
    logits = torch.tensor([1.5, 1.4, -2.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 0.0])
    valid = torch.ones(3, dtype=torch.bool)

    actual = host_native_multi_positive_task_score(logits, targets, valid)
    shifted = host_native_multi_positive_task_score(logits + 137.0, targets, valid)

    assert actual is not None and shifted is not None
    torch.testing.assert_close(actual, -torch.log_softmax(logits, dim=0)[0])
    torch.testing.assert_close(actual, shifted)
    actual.backward()
    assert logits.grad is not None
    assert logits.grad[0] < 0
    assert logits.grad[1] > 0
    assert logits.grad[2] > 0
    assert logits.grad[1].abs() > logits.grad[2].abs()


def test_host_native_task_score_supports_normalized_multi_positive_targets() -> None:
    logits = torch.tensor([0.2, 0.7, -0.4], requires_grad=True)
    targets = torch.tensor([0.75, 0.25, 0.0])
    valid = torch.ones(3, dtype=torch.bool)

    actual = host_native_multi_positive_task_score(logits, targets, valid)

    assert actual is not None
    expected_distribution = targets / targets.sum()
    expected = -(expected_distribution * torch.log_softmax(logits, dim=0)).sum()
    torch.testing.assert_close(actual, expected)


def test_host_native_task_score_is_permutation_invariant_and_excludes_unknown_rows() -> None:
    logits = torch.tensor([0.2, 100.0, 0.7, -0.4])
    targets = torch.tensor([0.75, 0.0, 0.25, 0.0])
    valid = torch.tensor([True, False, True, True])
    permutation = torch.tensor([2, 3, 0, 1])

    factual = host_native_multi_positive_task_score(logits, targets, valid)
    permuted = host_native_multi_positive_task_score(
        logits[permutation],
        targets[permutation],
        valid[permutation],
    )
    without_unknown = host_native_multi_positive_task_score(
        logits[[0, 2, 3]],
        targets[[0, 2, 3]],
        torch.ones(3, dtype=torch.bool),
    )

    assert factual is not None and permuted is not None and without_unknown is not None
    torch.testing.assert_close(factual, permuted)
    torch.testing.assert_close(factual, without_unknown)


@pytest.mark.parametrize(
    ("targets", "valid"),
    (
        (torch.tensor([1.0, 0.0]), torch.zeros(2, dtype=torch.bool)),
        (torch.tensor([1.0, 1.0]), torch.ones(2, dtype=torch.bool)),
        (torch.tensor([0.0, 0.0]), torch.ones(2, dtype=torch.bool)),
        (torch.tensor([0.5]), torch.ones(1, dtype=torch.bool)),
    ),
)
def test_host_native_task_score_ignores_unidentifiable_rows(
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    logits = torch.zeros_like(targets, requires_grad=True)
    assert host_native_multi_positive_task_score(logits, targets, valid) is None


def test_host_native_task_term_trains_only_existing_match_path() -> None:
    readout = SharedRelationReadout(4, temperature_init=0.1)
    match_hidden = torch.tensor(
        [[[0.6, 0.4, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0]]],
        requires_grad=True,
    )
    relation = readout(
        posterior_rows=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        sensor_hidden=torch.tensor([[[0.5, 0.5, 0.0, 0.0]]]),
        sensor_valid=torch.ones(1, 1, dtype=torch.bool),
        match_hidden=match_hidden,
    )

    term = host_native_multi_positive_task_relation_term(
        relation,
        _targets(task=(1.0, 0.0)),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        weight=1.0,
    )

    assert term.valid.tolist() == [True]
    term.normalized().backward()
    assert readout.match_projection.weight.grad is not None
    assert readout.match_projection.weight.grad.abs().sum() > 0
    assert match_hidden.grad is not None and match_hidden.grad.abs().sum() > 0
    assert readout.projection.weight.grad is None
    assert readout.temperature_parameter.grad is None
    with pytest.raises(ValueError, match="requires host-native MATCH"):
        host_native_multi_positive_task_relation_term(
            _relation(
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            ),
            _targets(),
            SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
            weight=1.0,
        )


def test_factorized_task_term_trains_match_without_rewriting_physical_ownership() -> None:
    torch.manual_seed(41)
    readout = SharedRelationReadout(4, temperature_init=0.1)
    rows = torch.randn(1, 2, 4, requires_grad=True)
    sensors = torch.randn(1, 2, 4, requires_grad=True)
    match = torch.randn(1, 2, 4, requires_grad=True)
    relation = readout(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
        match_hidden=match,
    )

    term = host_native_factorized_task_relation_term(
        (relation,),
        _visible_targets(),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        weight=1.0,
    )

    assert term.name == "set/task_row"
    assert term.valid.tolist() == [True]
    term.normalized().backward()
    assert rows.grad is None
    assert sensors.grad is None
    assert match.grad is not None and match.grad.abs().sum() > 0
    assert readout.projection.weight.grad is None
    assert readout.match_projection.weight.grad is not None
    assert readout.match_projection.weight.grad.abs().sum() > 0
    assert readout.temperature_parameter.grad is None
    assert readout.existence_projection.weight.grad is None


def test_factorized_task_term_does_not_push_any_physical_row_or_context() -> None:
    torch.manual_seed(42)
    readout = SharedRelationReadout(4, temperature_init=0.1)
    relation = readout(
        posterior_rows=torch.randn(1, 2, 4, requires_grad=True),
        sensor_hidden=torch.randn(1, 2, 4, requires_grad=True),
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
        match_hidden=torch.randn(1, 2, 4, requires_grad=True),
    )
    relation.support_logits.retain_grad()

    term = host_native_factorized_task_relation_term(
        (relation,),
        _visible_targets(),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        weight=1.0,
    )
    term.normalized().backward()

    assert relation.support_logits.grad is None
    assert readout.no_object.grad is None


def test_factorized_task_term_is_invariant_to_spatial_occlusion() -> None:
    torch.manual_seed(43)
    readout = SharedRelationReadout(4, temperature_init=0.1)
    relation = readout(
        posterior_rows=torch.randn(1, 2, 4, requires_grad=True),
        sensor_hidden=torch.randn(1, 2, 4, requires_grad=True),
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
        match_hidden=torch.randn(1, 2, 4, requires_grad=True),
    )
    relation.task_relevance_logits.retain_grad()
    relation.support_logits.retain_grad()
    targets = _visible_targets(task=(1.0, 1.0))
    masks = targets.masks.clone()
    masks[:, :, 1].zero_()
    established = SequenceAssignment(
        row_to_track=torch.tensor([[0, 1]]),
        binding_start_phase=torch.zeros(1, 2, dtype=torch.long),
    )

    visible = host_native_factorized_task_relation_term(
        (relation,),
        targets,
        established,
        weight=1.0,
    )
    occluded = host_native_factorized_task_relation_term(
        (relation,),
        replace(targets, masks=masks),
        established,
        weight=1.0,
    )
    torch.testing.assert_close(visible.normalized(), occluded.normalized())
    occluded.normalized().backward()

    assert relation.task_relevance_logits.grad is not None
    assert relation.task_relevance_logits.grad.abs().sum() > 0
    assert relation.support_logits.grad is None


def test_factorized_task_term_does_not_backpropagate_before_first_identity_evidence() -> None:
    torch.manual_seed(47)
    readout = SharedRelationReadout(4, temperature_init=0.1)
    first_match = torch.randn(1, 2, 4, requires_grad=True)
    second_match = torch.randn(1, 2, 4, requires_grad=True)
    common = {
        "posterior_rows": torch.randn(1, 2, 4),
        "sensor_hidden": torch.randn(1, 2, 4),
        "sensor_valid": torch.ones(1, 2, dtype=torch.bool),
    }
    first = readout(match_hidden=first_match, **common)
    second = readout(match_hidden=second_match, **common)
    first.task_relevance_logits.retain_grad()
    second.task_relevance_logits.retain_grad()
    visible = _visible_targets(task=(1.0, 0.0))
    targets = replace(
        visible,
        masks=torch.cat((torch.zeros_like(visible.masks), visible.masks), dim=1),
        mask_valid=visible.mask_valid.repeat(1, 2, 1, 1),
        existence=visible.existence.repeat(1, 2, 1),
        existence_valid=visible.existence_valid.repeat(1, 2, 1),
        token_observed_fraction=visible.token_observed_fraction.repeat(1, 2, 1),
        inventory_exhaustive=visible.inventory_exhaustive.repeat(1, 2),
    )
    assignment = SequenceAssignment(row_to_track=torch.tensor([[0, 1]]))

    term = host_native_factorized_task_relation_term(
        (first, second),
        targets,
        assignment,
        weight=1.0,
    )
    assert term.valid.tolist() == [True]
    term.normalized().backward()

    assert first.task_relevance_logits.grad is None
    assert first_match.grad is None
    assert second.task_relevance_logits.grad is not None
    assert second_match.grad is not None and second_match.grad.abs().sum() > 0


def test_factorized_task_term_is_row_permutation_invariant_with_assignment() -> None:
    torch.manual_seed(43)
    readout = SharedRelationReadout(4, temperature_init=0.1)
    rows = torch.randn(1, 2, 4)
    sensors = torch.randn(1, 2, 4)
    match = torch.randn(1, 2, 4)
    valid = torch.ones(1, 2, dtype=torch.bool)
    factual = readout(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=match,
    )
    permuted = readout(
        posterior_rows=rows[:, [1, 0]],
        sensor_hidden=sensors,
        sensor_valid=valid,
        match_hidden=match[:, [1, 0]],
    )

    factual_term = host_native_factorized_task_relation_term(
        (factual,),
        _visible_targets(),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        weight=1.0,
    )
    permuted_term = host_native_factorized_task_relation_term(
        (permuted,),
        _visible_targets(),
        SequenceAssignment(row_to_track=torch.tensor([[1, 0]])),
        weight=1.0,
    )
    torch.testing.assert_close(factual_term.normalized(), permuted_term.normalized())


def test_factorized_task_term_treats_multiple_positive_rows_as_independent_marginals() -> None:
    readout = SharedRelationReadout(4, temperature_init=0.1)
    relation = readout(
        posterior_rows=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        sensor_hidden=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
        match_hidden=torch.zeros(1, 2, 4),
    )
    term = host_native_factorized_task_relation_term(
        (relation,),
        _visible_targets(task=(1.0, 1.0)),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        weight=1.0,
    )

    expected = F.binary_cross_entropy_with_logits(
        relation.task_relevance_logits[0].float(),
        torch.ones(2),
        reduction="mean",
    )
    torch.testing.assert_close(term.normalized(), expected)


def test_factorized_task_term_excludes_unknown_rows_without_probability_coupling() -> None:
    readout = SharedRelationReadout(4, temperature_init=0.1)
    with torch.no_grad():
        readout.match_projection.weight.zero_()
        readout.match_projection.weight[0, 0] = 1.0
        readout.match_projection.bias.zero_()
    rows = torch.randn(1, 3, 4)
    sensors = torch.randn(1, 2, 4)
    first_match = torch.tensor(
        [[[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0], [-100.0, 0.0, 0.0, 0.0]]]
    )
    second_match = first_match.clone()
    second_match[0, 2, 0] = 100.0
    targets = replace(
        _visible_targets(),
        inventory_exhaustive=torch.zeros(1, 1, dtype=torch.bool),
    )
    assignment = SequenceAssignment(row_to_track=torch.tensor([[0, 1, -1]]))

    terms = []
    for match in (first_match, second_match):
        relation = readout(
            posterior_rows=rows,
            sensor_hidden=sensors,
            sensor_valid=torch.ones(1, 2, dtype=torch.bool),
            match_hidden=match,
        )
        terms.append(
            host_native_factorized_task_relation_term(
                (relation,),
                targets,
                assignment,
                weight=1.0,
            ).normalized()
        )

    torch.testing.assert_close(terms[0], terms[1])


def test_factorized_task_term_uses_match_when_target_is_fully_occluded() -> None:
    torch.manual_seed(47)
    readout = SharedRelationReadout(4, temperature_init=0.1)
    rows = torch.randn(1, 2, 4, requires_grad=True)
    sensors = torch.randn(1, 2, 4, requires_grad=True)
    match = torch.randn(1, 2, 4, requires_grad=True)
    relation = readout(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
        match_hidden=match,
    )

    term = host_native_factorized_task_relation_term(
        (relation,),
        _visible_targets(occluded=True),
        SequenceAssignment(
            row_to_track=torch.tensor([[0, 1]]),
            binding_start_phase=torch.zeros(1, 2, dtype=torch.long),
        ),
        weight=1.0,
    )
    assert term.valid.tolist() == [True]
    term.normalized().backward()

    assert match.grad is not None and match.grad.abs().sum() > 0
    assert readout.match_projection.weight.grad is not None
    assert readout.match_projection.weight.grad.abs().sum() > 0
    assert rows.grad is None
    assert sensors.grad is None
    assert readout.projection.weight.grad is None
    assert readout.temperature_parameter.grad is None


def test_factorized_task_term_stays_finite_when_exported_probability_underflows() -> None:
    readout = SharedRelationReadout(4, temperature_init=0.1)
    with torch.no_grad():
        readout.match_projection.weight.zero_()
        readout.match_projection.weight[0, 0] = 1.0
        readout.match_projection.bias.zero_()
    match = torch.tensor(
        [[[-1000.0, 0.0, 0.0, 0.0], [1000.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    relation = readout(
        posterior_rows=torch.randn(1, 2, 4),
        sensor_hidden=torch.randn(1, 2, 4),
        sensor_valid=torch.ones(1, 2, dtype=torch.bool),
        match_hidden=match,
    )
    assert relation.task_row_probability is not None
    assert relation.task_row_probability[0, 0] == 0

    term = host_native_factorized_task_relation_term(
        (relation,),
        _visible_targets(),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        weight=1.0,
    )
    value = term.normalized()
    assert torch.isfinite(value)
    torch.testing.assert_close(value, torch.tensor(1000.0))
    value.backward()
    assert match.grad is not None
    assert torch.isfinite(match.grad).all()
    assert match.grad.abs().sum() > 0


def test_two_rank_gather_retains_local_graph_and_applies_gradient_only_scaling() -> None:
    local_task = torch.tensor([[0.8, 0.2]], requires_grad=True)
    local_rows = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]], requires_grad=True)
    remote_task = torch.tensor([[0.2, 0.8]])
    remote_rows = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    local_relation = _relation(local_task, local_rows)
    remote_targets = materialize_task_relation_targets(
        _targets(task=(0.0, 1.0)),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        identity_keys_by_batch=(("red", "blue"),),
    )
    distributed = _TwoRankGather(
        remote_embeddings=torch.cat((remote_task.unsqueeze(1), remote_rows), dim=1),
        remote_metadata=_metadata(remote_targets),
    )
    term = global_task_relation_term(
        local_relation,
        _targets(),
        SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
        identity_keys_by_batch=(("red", "blue"),),
        weight=1.0,
        distributed=distributed,
    )
    displayed = term.normalized()
    displayed.backward()
    assert distributed.payload_call == 2
    assert local_task.grad is not None and local_rows.grad is not None

    reference_task = local_task.detach().clone().requires_grad_()
    reference_rows = local_rows.detach().clone().requires_grad_()
    reference_targets = TaskRelationTargets(
        row_identity_fingerprints=torch.tensor(
            [
                [identity_fingerprint("red"), identity_fingerprint("blue")],
                [identity_fingerprint("red"), identity_fingerprint("blue")],
            ]
        ),
        row_valid=torch.ones(2, 2, dtype=torch.bool),
        target_identity_fingerprints=torch.tensor(
            [
                [identity_fingerprint("red"), (0, 0)],
                [identity_fingerprint("blue"), (0, 0)],
            ]
        ),
        target_valid=torch.tensor([[True, False], [True, False]]),
        query_valid=torch.ones(2, dtype=torch.bool),
    )
    reference_values, reference_valid = multi_positive_task_relation_values(
        task_embeddings=torch.cat((reference_task, remote_task), dim=0),
        row_embeddings=torch.cat((reference_rows, remote_rows), dim=0),
        relation_temperature=torch.tensor([0.1]),
        targets=reference_targets,
    )
    reference_loss = reference_values[reference_valid].mean()
    torch.testing.assert_close(displayed.detach(), reference_loss.detach())
    reference_loss.backward()
    torch.testing.assert_close(local_task.grad, reference_task.grad * 2)
    torch.testing.assert_close(local_rows.grad, reference_rows.grad * 2)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_real_two_rank_gloo_matches_full_matrix_reference(tmp_path: Path) -> None:
    init_path = tmp_path / "gloo-init"
    mp.spawn(
        _gloo_task_relation_worker,
        args=(str(init_path), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    results = [torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True) for rank in range(2)]

    tasks = torch.tensor(
        [[0.8, 0.2], [0.2, 0.8]],
        requires_grad=True,
    )
    rows = torch.tensor(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.8, 0.2], [0.2, 0.8]],
        ],
        requires_grad=True,
    )
    red = identity_fingerprint("red")
    blue = identity_fingerprint("blue")
    values, valid = multi_positive_task_relation_values(
        task_embeddings=tasks,
        row_embeddings=rows,
        relation_temperature=torch.tensor([0.1]),
        targets=TaskRelationTargets(
            row_identity_fingerprints=torch.tensor([[red, blue], [red, blue]]),
            row_valid=torch.ones(2, 2, dtype=torch.bool),
            target_identity_fingerprints=torch.tensor([[red, (0, 0)], [blue, (0, 0)]]),
            target_valid=torch.tensor([[True, False], [True, False]]),
            query_valid=torch.ones(2, dtype=torch.bool),
        ),
    )
    reference = values[valid].mean()
    reference.backward()
    for rank, result in enumerate(results):
        torch.testing.assert_close(result["loss"], reference.detach())
        torch.testing.assert_close(result["task_grad"], tasks.grad[rank : rank + 1] * 2)
        torch.testing.assert_close(result["row_grad"], rows.grad[rank : rank + 1] * 2)


def test_distributed_gather_fails_closed_on_shape_mismatch() -> None:
    relation = _relation(
        torch.tensor([[1.0, 0.0]], requires_grad=True),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True),
    )

    class _BadShape(_TwoRankGather):
        def all_gather(self, outputs: list[torch.Tensor], value: torch.Tensor) -> None:
            outputs[0].copy_(value)
            if value.ndim == 1:
                outputs[1].copy_(value)
                outputs[1][0] += 1
                return
            raise AssertionError("payload gather must not run after shape mismatch")

    with pytest.raises(RuntimeError, match="unequal shapes"):
        global_task_relation_term(
            relation,
            _targets(),
            SequenceAssignment(row_to_track=torch.tensor([[0, 1]])),
            identity_keys_by_batch=(("red", "blue"),),
            weight=1.0,
            distributed=_BadShape(
                remote_embeddings=torch.empty(0),
                remote_metadata=torch.empty(0),
            ),
        )
