"""Fail-closed task/action supervision scopes for native VLA training.

Counterfactual language over a fixed observation is useful representation
evidence, but it does not manufacture a counterfactual robot trajectory.  This
module keeps that distinction explicit without adding a deploy-time model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

TASK_ACTION_SUPERVISION_SCHEMA = "picf-next.task-action-supervision.v1"


class TaskActionSupervisionScope(str, Enum):
    """The losses that one immutable task/action pairing may supervise."""

    FACTUAL_ACTION = "factual-action"
    REPRESENTATION_ONLY = "representation-only"


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    value = _text(value, name=name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class TaskActionSupervisionReceipt:
    """Immutable proof that official action loss is source-task factual."""

    sample_key: str
    source_task_key: str
    source_instruction_sha256: str
    candidate_task_key: str
    candidate_instruction_sha256: str
    source_action_targets_sha256: str
    candidate_action_targets_sha256: str
    scope: TaskActionSupervisionScope

    @property
    def official_action_loss_enabled(self) -> bool:
        return self.scope is TaskActionSupervisionScope.FACTUAL_ACTION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": TASK_ACTION_SUPERVISION_SCHEMA,
            "sample_key": self.sample_key,
            "source_task_key": self.source_task_key,
            "source_instruction_sha256": self.source_instruction_sha256,
            "candidate_task_key": self.candidate_task_key,
            "candidate_instruction_sha256": self.candidate_instruction_sha256,
            "source_action_targets_sha256": self.source_action_targets_sha256,
            "candidate_action_targets_sha256": self.candidate_action_targets_sha256,
            "scope": self.scope.value,
            "official_action_loss_enabled": self.official_action_loss_enabled,
        }


def task_action_supervision_receipt(
    *,
    sample_key: str,
    source_task_key: str,
    source_instruction: str,
    candidate_task_key: str,
    candidate_instruction: str,
    source_action_targets_sha256: str,
    candidate_action_targets_sha256: str,
) -> TaskActionSupervisionReceipt:
    """Classify a same-sample candidate without inventing action supervision.

    The action payload must remain byte-identical.  It is factual action
    supervision only when both the task identity and the complete instruction
    are also unchanged.  A language intervention is representation-only even
    if it describes an object visible in the observation.
    """

    sample_key = _text(sample_key, name="task/action sample key")
    source_task_key = _text(source_task_key, name="source task key")
    source_instruction = _text(source_instruction, name="source instruction")
    candidate_task_key = _text(candidate_task_key, name="candidate task key")
    candidate_instruction = _text(candidate_instruction, name="candidate instruction")
    source_action_targets_sha256 = _sha256(
        source_action_targets_sha256,
        name="source action targets SHA-256",
    )
    candidate_action_targets_sha256 = _sha256(
        candidate_action_targets_sha256,
        name="candidate action targets SHA-256",
    )
    if candidate_action_targets_sha256 != source_action_targets_sha256:
        raise ValueError("same-observation candidate changed immutable action targets")

    source_instruction_sha256 = hashlib.sha256(source_instruction.encode("utf-8")).hexdigest()
    candidate_instruction_sha256 = hashlib.sha256(
        candidate_instruction.encode("utf-8")
    ).hexdigest()
    factual = (
        candidate_task_key == source_task_key
        and candidate_instruction_sha256 == source_instruction_sha256
    )
    return TaskActionSupervisionReceipt(
        sample_key=sample_key,
        source_task_key=source_task_key,
        source_instruction_sha256=source_instruction_sha256,
        candidate_task_key=candidate_task_key,
        candidate_instruction_sha256=candidate_instruction_sha256,
        source_action_targets_sha256=source_action_targets_sha256,
        candidate_action_targets_sha256=candidate_action_targets_sha256,
        scope=(
            TaskActionSupervisionScope.FACTUAL_ACTION
            if factual
            else TaskActionSupervisionScope.REPRESENTATION_ONLY
        ),
    )


def require_factual_action_supervision(receipt: TaskActionSupervisionReceipt) -> None:
    """Reject official action loss on any language-intervened sample."""

    if not isinstance(receipt, TaskActionSupervisionReceipt):
        raise TypeError("official action supervision requires a typed receipt")
    if not receipt.official_action_loss_enabled:
        raise ValueError("official action loss requires the immutable source task and instruction")
