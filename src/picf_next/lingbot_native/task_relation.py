"""Loss-only task-to-object estimators over shared-host relation states."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from picf_next.lingbot_native.relations import (
    HOST_NATIVE_MATCH_INTERFACE,
    LEGACY_SHARED_COSINE_INTERFACE,
    RelationOutput,
)
from picf_next.lingbot_native.supervision import (
    NativeSequenceTargets,
    SequenceAssignment,
    assignment_binding_start_phase,
    materialize_row_task_supervision,
)
from picf_next.objective import ObjectiveTerm

LOCAL_BALANCED_TASK_RELATION = "local_balanced_sigmoid"
GLOBAL_MULTIPOSITIVE_TASK_RELATION = "global_multi_positive_softmax"
HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION = "host_native_multi_positive_softmax"
HOST_NATIVE_FACTORIZED_TASK_RELATION = "host_native_factorized_task_physical_ownership"
TASK_RELATION_ESTIMATORS = (
    LOCAL_BALANCED_TASK_RELATION,
    GLOBAL_MULTIPOSITIVE_TASK_RELATION,
    HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION,
    HOST_NATIVE_FACTORIZED_TASK_RELATION,
)

_FINGERPRINT_MASK = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class TaskRelationTargets:
    """Fixed-shape loss metadata for global task-to-row retrieval."""

    row_identity_fingerprints: torch.Tensor
    row_valid: torch.Tensor
    target_identity_fingerprints: torch.Tensor
    target_valid: torch.Tensor
    query_valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.row_identity_fingerprints.ndim != 3:
            raise ValueError("row identity fingerprints must have shape [batch,rows,2]")
        batch, rows, words = self.row_identity_fingerprints.shape
        if words != 2 or self.row_identity_fingerprints.dtype != torch.long:
            raise ValueError("row identity fingerprints must contain two int64 words")
        if self.target_identity_fingerprints.shape != (batch, rows, 2):
            raise ValueError("target identity fingerprints must match row capacity")
        if self.target_identity_fingerprints.dtype != torch.long:
            raise TypeError("target identity fingerprints must be int64")
        for name, value, shape in (
            ("row_valid", self.row_valid, (batch, rows)),
            ("target_valid", self.target_valid, (batch, rows)),
            ("query_valid", self.query_valid, (batch,)),
        ):
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape {shape}")
        values = (
            self.row_valid,
            self.target_identity_fingerprints,
            self.target_valid,
            self.query_valid,
        )
        if any(value.device != self.row_identity_fingerprints.device for value in values):
            raise ValueError("task relation targets must share one device")
        if any(value.requires_grad for value in (self.row_identity_fingerprints, *values)):
            raise ValueError("task relation targets must be detached")


def identity_fingerprint(identity_key: str) -> tuple[int, int]:
    """Return a deterministic nonzero 126-bit loss-side identity fingerprint."""

    if not isinstance(identity_key, str) or not identity_key:
        raise ValueError("physical identity key must be a nonempty string")
    digest = hashlib.sha256(identity_key.encode("utf-8")).digest()
    fingerprint = (
        int.from_bytes(digest[:8], "big") & _FINGERPRINT_MASK,
        int.from_bytes(digest[8:16], "big") & _FINGERPRINT_MASK,
    )
    if fingerprint == (0, 0):
        fingerprint = (
            int.from_bytes(digest[16:24], "big") & _FINGERPRINT_MASK,
            int.from_bytes(digest[24:32], "big") & _FINGERPRINT_MASK,
        )
    if fingerprint == (0, 0):
        raise RuntimeError("physical identity produced the reserved zero fingerprint")
    return fingerprint


def materialize_task_relation_targets(
    targets: NativeSequenceTargets,
    assignment: SequenceAssignment,
    *,
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
) -> TaskRelationTargets:
    """Map exact task identities and assigned rows after the model forward."""

    if not isinstance(targets, NativeSequenceTargets) or not isinstance(
        assignment, SequenceAssignment
    ):
        raise TypeError("global task relation requires typed loss-side targets")
    batch, rows = assignment.row_to_track.shape
    if batch != targets.masks.shape[0] or len(identity_keys_by_batch) != batch:
        raise ValueError("task relation batches differ across predictions and targets")
    device = targets.masks.device
    row_fingerprints = torch.zeros(batch, rows, 2, dtype=torch.long, device=device)
    row_valid = torch.zeros(batch, rows, dtype=torch.bool, device=device)
    target_fingerprints = torch.zeros_like(row_fingerprints)
    target_valid = torch.zeros_like(row_valid)
    query_valid = torch.zeros(batch, dtype=torch.bool, device=device)

    known_fingerprints: dict[tuple[int, int], str] = {}
    for batch_index, identity_keys in enumerate(identity_keys_by_batch):
        if len(identity_keys) > targets.masks.shape[2]:
            raise ValueError("task relation identity inventory exceeds target tracks")
        fingerprints: list[tuple[int, int]] = []
        for identity_key in identity_keys:
            fingerprint = identity_fingerprint(identity_key)
            previous = known_fingerprints.setdefault(fingerprint, identity_key)
            if previous != identity_key:
                raise RuntimeError("physical identity fingerprint collision")
            fingerprints.append(fingerprint)

        for row_index, track_index in enumerate(
            assignment.row_to_track[batch_index].detach().cpu().tolist()
        ):
            if track_index < 0:
                continue
            if track_index >= len(fingerprints):
                raise ValueError("assigned row references an absent physical identity")
            row_fingerprints[batch_index, row_index] = torch.tensor(
                fingerprints[track_index],
                dtype=torch.long,
                device=device,
            )
            row_valid[batch_index, row_index] = True

        valid_tracks = targets.track_valid[batch_index]
        exact_task = bool(
            torch.equal(targets.task_valid[batch_index], valid_tracks)
            and (targets.task_relevance[batch_index] > 0).any()
        )
        if not exact_task:
            continue
        relevant_tracks = (
            ((targets.task_relevance[batch_index] > 0) & valid_tracks).nonzero().flatten()
        )
        if relevant_tracks.numel() > rows:
            raise ValueError("exact task target count exceeds posterior row capacity")
        for target_index, track_index in enumerate(relevant_tracks.detach().cpu().tolist()):
            if track_index >= len(fingerprints):
                raise ValueError("exact task target references an absent identity")
            target_fingerprints[batch_index, target_index] = torch.tensor(
                fingerprints[track_index],
                dtype=torch.long,
                device=device,
            )
            target_valid[batch_index, target_index] = True
        query_valid[batch_index] = bool(relevant_tracks.numel())

    return TaskRelationTargets(
        row_identity_fingerprints=row_fingerprints,
        row_valid=row_valid,
        target_identity_fingerprints=target_fingerprints,
        target_valid=target_valid,
        query_valid=query_valid,
    )


def multi_positive_task_relation_values(
    *,
    task_embeddings: torch.Tensor,
    row_embeddings: torch.Tensor,
    relation_temperature: torch.Tensor,
    targets: TaskRelationTargets,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one multi-positive retrieval loss per exact task query."""

    if task_embeddings.ndim != 2 or row_embeddings.ndim != 3:
        raise ValueError("task and row embeddings must have shapes [batch,d] and [batch,k,d]")
    batch, rows, width = row_embeddings.shape
    if task_embeddings.shape != (batch, width):
        raise ValueError("task and row embedding dimensions differ")
    if targets.row_identity_fingerprints.shape != (batch, rows, 2):
        raise ValueError("task relation labels differ from the embedding batch")
    if relation_temperature.numel() != 1 or not relation_temperature.is_floating_point():
        raise ValueError("relation temperature must be one floating scalar")
    floating = (task_embeddings, row_embeddings, relation_temperature)
    if any(value.device != task_embeddings.device for value in floating):
        raise ValueError("task relation floating tensors must share one device")
    if any(not torch.isfinite(value).all() for value in floating):
        raise ValueError("task relation floating tensors contain NaN or infinity")
    if not bool(relation_temperature.detach() > 0):
        raise ValueError("relation temperature must be positive")
    if targets.row_identity_fingerprints.device != task_embeddings.device:
        raise ValueError("task relation labels and embeddings must share one device")

    flat_rows = row_embeddings.reshape(batch * rows, width)
    scores = torch.matmul(task_embeddings.float(), flat_rows.float().T)
    scores = scores / relation_temperature.detach().float().reshape(())
    flat_row_valid = targets.row_valid.reshape(batch * rows)
    flat_row_fingerprints = targets.row_identity_fingerprints.reshape(batch * rows, 2)
    identity_match = (
        targets.target_identity_fingerprints[:, :, None, :]
        == flat_row_fingerprints[None, None, :, :]
    ).all(dim=-1)
    positive = (identity_match & targets.target_valid[:, :, None]).any(
        dim=1
    ) & flat_row_valid.unsqueeze(0)
    known_row_count = flat_row_valid.sum()
    safe_scores = scores.masked_fill(
        ~flat_row_valid.unsqueeze(0),
        -torch.finfo(scores.dtype).max,
    )
    if not bool(known_row_count):
        safe_scores = torch.zeros_like(safe_scores)
    log_probability = torch.log_softmax(safe_scores, dim=-1)
    positive_count = positive.sum(dim=-1)
    valid = targets.query_valid & (positive_count > 0) & (known_row_count > 0)
    values = -(log_probability * positive.to(log_probability.dtype)).sum(
        dim=-1
    ) / positive_count.clamp_min(1).to(log_probability.dtype)
    values = values.masked_fill(~valid, 0)
    if not torch.isfinite(values).all():
        raise RuntimeError("global task relation loss is non-finite")
    return values, valid


def _distributed_context(distributed: Any) -> tuple[int, int]:
    if distributed is None or not distributed.is_available() or not distributed.is_initialized():
        return 1, 0
    world_size = distributed.get_world_size()
    rank = distributed.get_rank()
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < world_size
    ):
        raise RuntimeError("distributed task relation topology is invalid")
    return world_size, rank


def _all_gather_equal_shape(
    value: torch.Tensor,
    *,
    world_size: int,
    rank: int,
    distributed: Any,
    retain_local_gradient: bool,
) -> torch.Tensor:
    if world_size == 1:
        return value
    shape = torch.tensor(value.shape, dtype=torch.long, device=value.device)
    gathered_shapes = [torch.empty_like(shape) for _ in range(world_size)]
    distributed.all_gather(gathered_shapes, shape)
    if any(not torch.equal(candidate, shape) for candidate in gathered_shapes):
        raise RuntimeError("distributed task relation tensors have unequal shapes")
    contiguous = value.contiguous()
    gathered = [torch.empty_like(contiguous) for _ in range(world_size)]
    distributed.all_gather(gathered, contiguous)
    if retain_local_gradient:
        gathered[rank] = value
    return torch.cat(gathered, dim=0)


def global_task_relation_term(
    relation: RelationOutput,
    targets: NativeSequenceTargets,
    assignment: SequenceAssignment,
    *,
    identity_keys_by_batch: tuple[tuple[str, ...], ...],
    weight: float,
    distributed: Any | None = None,
) -> ObjectiveTerm:
    """Build Qwen-style all-rank retrieval with loss-only object identities."""

    if not isinstance(relation, RelationOutput):
        raise TypeError("global task relation requires a typed relation output")
    if relation.task_interface != LEGACY_SHARED_COSINE_INTERFACE:
        raise ValueError("global embedding retrieval is forbidden for host-native match outputs")
    if distributed is None:
        distributed = torch.distributed
    local_targets = materialize_task_relation_targets(
        targets,
        assignment,
        identity_keys_by_batch=identity_keys_by_batch,
    )
    task_embeddings = relation.task_embedding
    row_embeddings = relation.row_embeddings
    if task_embeddings is None:
        raise ValueError("global task relation requires one explicit task embedding")
    if task_embeddings.shape[:1] != row_embeddings.shape[:1]:
        raise ValueError("task and row relation outputs differ in batch size")
    world_size, rank = _distributed_context(distributed)

    combined_embeddings = torch.cat((task_embeddings.unsqueeze(1), row_embeddings), dim=1)
    global_embeddings = _all_gather_equal_shape(
        combined_embeddings,
        world_size=world_size,
        rank=rank,
        distributed=distributed,
        retain_local_gradient=True,
    )
    global_task_embeddings = global_embeddings[:, 0]
    global_row_embeddings = global_embeddings[:, 1:]

    batch, rows = local_targets.row_valid.shape
    metadata = torch.cat(
        (
            local_targets.row_identity_fingerprints.reshape(batch, rows * 2),
            local_targets.target_identity_fingerprints.reshape(batch, rows * 2),
            local_targets.row_valid.to(torch.long),
            local_targets.target_valid.to(torch.long),
            local_targets.query_valid.to(torch.long).unsqueeze(1),
        ),
        dim=1,
    )
    global_metadata = _all_gather_equal_shape(
        metadata,
        world_size=world_size,
        rank=rank,
        distributed=distributed,
        retain_local_gradient=False,
    )
    offset = 0
    global_row_fingerprints = global_metadata[:, offset : offset + rows * 2].reshape(-1, rows, 2)
    offset += rows * 2
    global_target_fingerprints = global_metadata[:, offset : offset + rows * 2].reshape(-1, rows, 2)
    offset += rows * 2
    global_row_valid = global_metadata[:, offset : offset + rows].to(torch.bool)
    offset += rows
    global_target_valid = global_metadata[:, offset : offset + rows].to(torch.bool)
    offset += rows
    global_query_valid = global_metadata[:, offset].to(torch.bool)
    offset += 1
    if offset != global_metadata.shape[1]:
        raise RuntimeError("distributed task relation metadata parse is incomplete")

    values, valid = multi_positive_task_relation_values(
        task_embeddings=global_task_embeddings,
        row_embeddings=global_row_embeddings,
        relation_temperature=relation.relation_temperature,
        targets=TaskRelationTargets(
            row_identity_fingerprints=global_row_fingerprints,
            row_valid=global_row_valid,
            target_identity_fingerprints=global_target_fingerprints,
            target_valid=global_target_valid,
            query_valid=global_query_valid,
        ),
    )
    if world_size > 1:
        values = values.detach() + world_size * (values - values.detach())
    return ObjectiveTerm(
        name="set/task",
        values=values,
        valid=valid,
        weight=weight,
    )


def host_native_multi_positive_task_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor | None:
    """Return row-competitive soft-target cross entropy for one observation."""

    if logits.ndim != 1 or targets.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError("host-native task inputs must share one row axis")
    if (
        not logits.is_floating_point()
        or not targets.is_floating_point()
        or valid.dtype != torch.bool
    ):
        raise TypeError("host-native task logits/targets must be floating and validity boolean")
    if any(value.device != logits.device for value in (targets, valid)):
        raise ValueError("host-native task inputs must share one device")
    if (
        not torch.isfinite(logits).all()
        or not torch.isfinite(targets).all()
        or ((targets < 0) | (targets > 1)).any()
    ):
        raise ValueError("host-native task inputs must be finite with targets in [0,1]")
    if targets.requires_grad or valid.requires_grad:
        raise ValueError("host-native task supervision must be detached")
    if not valid.any():
        return None

    selected_logits = logits[valid].float()
    selected_targets = targets[valid].to(dtype=selected_logits.dtype)
    if selected_logits.numel() < 2:
        return None
    positive_mass = selected_targets.sum()
    negative_mass = (1 - selected_targets).sum()
    if not bool(positive_mass > 0) or not bool(negative_mass > 0):
        return None
    positive_distribution = selected_targets / positive_mass
    value = -(positive_distribution * F.log_softmax(selected_logits, dim=0)).sum()
    if not torch.isfinite(value):
        raise RuntimeError("host-native task relation loss is non-finite")
    return value


def host_native_multi_positive_task_relation_term(
    relation: RelationOutput,
    targets: NativeSequenceTargets,
    assignment: SequenceAssignment,
    *,
    weight: float,
) -> ObjectiveTerm:
    """Apply row competition directly to the shared-host MATCH logits."""

    if not isinstance(relation, RelationOutput):
        raise TypeError("host-native task relation requires a typed relation output")
    if relation.task_interface != HOST_NATIVE_MATCH_INTERFACE:
        raise ValueError("host-native row competition requires host-native MATCH outputs")
    if relation.task_embedding is not None:
        raise ValueError("host-native row competition forbids a compressed task embedding")
    logits = relation.task_relevance_logits
    if logits.ndim != 2 or assignment.row_to_track.shape != logits.shape:
        raise ValueError("host-native task logits and row assignment must share [batch,rows]")
    if targets.masks.shape[0] != logits.shape[0]:
        raise ValueError("host-native task logits and targets have different batches")

    values: list[torch.Tensor] = []
    valid_samples: list[bool] = []
    for batch_index in range(logits.shape[0]):
        row_task = materialize_row_task_supervision(
            targets,
            assignment,
            batch_index=batch_index,
            dtype=logits.dtype,
        )
        value = host_native_multi_positive_task_score(
            logits[batch_index],
            row_task.target,
            row_task.valid,
        )
        valid_samples.append(value is not None)
        values.append(logits[batch_index].sum() * 0 if value is None else value)
    return ObjectiveTerm(
        name="set/task",
        values=torch.stack(values),
        valid=torch.tensor(valid_samples, dtype=torch.bool, device=logits.device),
        weight=weight,
    )


def _factorized_task_row_score(
    relation: RelationOutput,
    targets: NativeSequenceTargets,
    assignment: SequenceAssignment,
    *,
    batch_index: int,
    binding_valid: torch.Tensor,
) -> torch.Tensor | None:
    """Score proper Bernoulli task-relevance marginals on known rows only."""

    task_row_probability = relation.task_row_probability
    if task_row_probability is None:
        raise ValueError("factorized task supervision requires task-row probabilities")
    if task_row_probability.ndim != 2:
        raise ValueError("task-row probabilities must have shape [batch,rows]")
    batch, rows = task_row_probability.shape
    if rows <= 0 or assignment.row_to_track.shape != (batch, rows):
        raise ValueError("factorized task relation and row assignment axes differ")
    if relation.task_relevance_logits.shape != (batch, rows):
        raise ValueError("factorized relation and task row axes differ")
    if targets.masks.shape[0] != batch:
        raise ValueError("factorized relation and target batches differ")

    row_task = materialize_row_task_supervision(
        targets,
        assignment,
        batch_index=batch_index,
        dtype=relation.task_relevance_logits.dtype,
        binding_valid=binding_valid,
    )
    if not row_task.valid.any():
        return None
    selected_row_target = row_task.target[row_task.valid].float()
    relevant_tracks = (
        targets.track_valid[batch_index]
        & targets.task_valid[batch_index]
        & (targets.task_relevance[batch_index] > 0)
    )
    if (relevant_tracks & targets.capacity_censored[batch_index]).any():
        return None
    known_logits = relation.task_relevance_logits[batch_index, row_task.valid].float()
    value = F.binary_cross_entropy_with_logits(
        known_logits,
        selected_row_target,
        reduction="mean",
    )
    if not torch.isfinite(value):
        raise RuntimeError("factorized task-row loss is non-finite")
    return value


def host_native_factorized_task_relation_term(
    relations: Sequence[RelationOutput],
    targets: NativeSequenceTargets,
    assignment: SequenceAssignment,
    *,
    weight: float,
) -> ObjectiveTerm:
    """Supervise prompt-conditioned rows without rewriting physical ownership.

    The exported anchor event is ``p(relevant(row)|O,Q) p(owner=row|token,O)``.
    This term is the first proper factor and the physical categorical ownership
    NLL is the second. Unknown rows, unlabelled tokens and absent modalities
    create no false negative.
    """

    if not relations or any(not isinstance(value, RelationOutput) for value in relations):
        raise TypeError("factorized task relation requires a non-empty relation sequence")
    if len(relations) != targets.masks.shape[1]:
        raise ValueError("factorized task relation and targets have different time axes")
    final_logits = relations[-1].task_relevance_logits
    batch, rows = final_logits.shape
    if assignment.row_to_track.shape != (batch, rows):
        raise ValueError("factorized task relation and assignment have different row axes")
    for relation in relations:
        if (
            relation.task_interface != HOST_NATIVE_MATCH_INTERFACE
            or relation.task_embedding is not None
            or relation.task_row_probability is None
            or relation.task_row_probability.shape != (batch, rows)
            or relation.task_object_probability is None
            or relation.task_object_probability.shape[:1] != (batch,)
            or relation.task_object_probability.shape[2] != rows
        ):
            raise ValueError("factorized task relation requires host-native task/ownership outputs")

    values: list[torch.Tensor] = []
    valid_samples: list[bool] = []
    binding_start_phase = assignment_binding_start_phase(assignment, targets)
    for batch_index in range(batch):
        time_values: list[torch.Tensor] = []
        for time_index, relation in enumerate(relations):
            value = _factorized_task_row_score(
                relation,
                targets,
                assignment,
                batch_index=batch_index,
                binding_valid=(2 * time_index + 1) >= binding_start_phase[batch_index],
            )
            if value is not None:
                time_values.append(value)
        value = torch.stack(time_values).mean() if time_values else None
        valid_samples.append(value is not None)
        values.append(final_logits[batch_index].sum() * 0 if value is None else value)
    return ObjectiveTerm(
        name="set/task_row",
        values=torch.stack(values),
        valid=torch.tensor(valid_samples, dtype=torch.bool, device=final_logits.device),
        weight=weight,
    )
