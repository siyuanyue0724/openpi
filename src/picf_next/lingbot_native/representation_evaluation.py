"""Immutable source-disjoint evaluation contracts for representation training.

The evaluation surface is deliberately loss-side.  Evaluation plans are built
from source identities before observations are decoded, while token and
ownership labels are resolved only after the shared LingBot host has run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from picf_next.data.calvin import CalvinStatefulTransitionDataset
from picf_next.lingbot_native.calvin_supervision import TaskIdentityResolver
from picf_next.lingbot_native.lattice_feasibility import fractional_token_metrics
from picf_next.lingbot_native.representation_split import (
    RepresentationEvaluationSegment,
    RepresentationTrialSplit,
)
from picf_next.lingbot_native.task_diagnostics import validate_task_row_diagnostic
from picf_next.lingbot_native.visual_audit import NATIVE_VISUAL_AUDIT_SCHEMA

REPRESENTATION_EVALUATION_PLAN_SCHEMA = "picf-next.lingbot-representation-evaluation-plan.v3"
REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA = (
    "picf-next.lingbot-representation-evaluation-plan.v4"
)
REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA = (
    "picf-next.lingbot-representation-warm-evaluation-plan.v1"
)
REPRESENTATION_TOKEN_EVIDENCE_SCHEMA = "picf-next.lingbot-representation-token-evidence.v1"
REPRESENTATION_OWNERSHIP_ROW_SCHEMA = "picf-next.lingbot-representation-ownership-row.v1"
REPRESENTATION_OWNERSHIP_SUMMARY_SCHEMA = "picf-next.lingbot-representation-ownership-summary.v1"
REPRESENTATION_EVALUATION_SAMPLE_SCHEMA = "picf-next.lingbot-representation-evaluation-sample.v1"
REPRESENTATION_EVALUATION_PARTITION_SCHEMA = (
    "picf-next.lingbot-representation-evaluation-partition.v1"
)
REPRESENTATION_EVALUATION_SNAPSHOT_SCHEMA = (
    "picf-next.lingbot-representation-evaluation-snapshot.v1"
)

REPRESENTATION_EVALUATION_PARTITIONS = ("validation", "heldout")
REPRESENTATION_EVALUATION_WORLD_SIZE = 2

_PLAN_ITEM_FIELDS = frozenset(
    {
        "partition",
        "ordinal",
        "rank",
        "task_key",
        "segment_index",
        "source_episode_index",
        "source_global_index",
        "sample_key",
        "shuffled_task_sample_key",
        "shuffled_target_sample_key",
        "factual_target_identity_keys",
        "shuffled_task_target_identity_keys",
        "shuffled_target_target_identity_keys",
        "factual_task_instruction_sha256",
        "shuffled_task_instruction_sha256",
        "shuffled_target_instruction_sha256",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "representation_split_sha256",
        "world_size",
        "items",
        "artifact_sha256",
    }
)
_REFERENCE_PLAN_FIELDS = _PLAN_FIELDS | {"evaluation_reference_plan_sha256"}
_WARM_PLAN_FIELDS = _PLAN_FIELDS | {"history_transitions"}
_TOKEN_FIELDS = frozenset(
    {
        "schema",
        "logits",
        "target_mass",
        "metrics",
    }
)
_OWNERSHIP_ROW_FIELDS = frozenset(
    {
        "schema",
        "row_index",
        "track_index",
        "identity_key",
        "is_task_target",
        "prediction",
        "target",
        "weight",
        "valid_token_count",
        "intersection",
        "union",
        "prediction_mass",
        "prediction_target_mass",
        "soft_iou",
        "target_mass_concentration",
    }
)
_OWNERSHIP_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "row_count",
        "task_target_row_count",
        "macro_soft_iou",
        "target_soft_iou",
        "target_mass_concentration",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "schema",
        "checkpoint_global_step",
        "partition",
        "ordinal",
        "rank",
        "task_key",
        "segment_index",
        "source_episode_index",
        "source_global_index",
        "sample_key",
        "shuffled_task_sample_key",
        "shuffled_target_sample_key",
        "factual_target_identity_keys",
        "shuffled_task_target_identity_keys",
        "shuffled_target_target_identity_keys",
        "factual_task_instruction_sha256",
        "shuffled_task_instruction_sha256",
        "shuffled_target_instruction_sha256",
        "factual_token_evidence",
        "shuffled_task_token_evidence",
        "shuffled_target_token_evidence",
        "factual_task_row_diagnostic",
        "shuffled_task_row_diagnostic",
        "factual_ownership_rows",
        "factual_ownership_summary",
        "shuffled_task_ownership_rows",
        "shuffled_task_ownership_summary",
        "official_action_loss",
        "forward_seconds",
        "peak_cuda_reserved_bytes",
        "tensor_sha256",
        "visual_artifact",
        "loss_only_labels_visible_to_model",
        "target_resolution_happened_after_forward",
    }
)
_PARTITION_FIELDS = frozenset(
    {
        "schema",
        "partition",
        "sample_count",
        "task_count",
        "token_eligible_sample_count",
        "token_eligible_task_count",
        "control_eligible_sample_count",
        "control_eligible_task_count",
        "mean_task_fractional_weighted_auc",
        "mean_task_target_background_logit_margin",
        "mean_task_shuffled_task_fractional_weighted_auc",
        "mean_task_shuffled_target_fractional_weighted_auc",
        "mean_task_shuffled_task_auc_degradation",
        "mean_task_shuffled_target_auc_degradation",
        "row_eligible_sample_count",
        "row_eligible_task_count",
        "rank_one_task_count",
        "rank_one_task_fraction",
        "mean_task_hardest_negative_logit_margin",
        "ownership_eligible_sample_count",
        "ownership_eligible_task_count",
        "mean_task_target_ownership_soft_iou",
        "mean_task_target_mass_concentration",
        "mean_task_macro_ownership_soft_iou",
        "mean_official_action_loss",
        "maximum_peak_cuda_reserved_bytes",
        "mean_factual_forward_seconds",
        "mean_shuffled_task_forward_seconds",
        "task_metrics",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "checkpoint_global_step",
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "representation_evaluation_plan_sha256",
        "representation_frozen_action_state_sha256",
        "samples",
        "partition_summaries",
        "artifact_sha256",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _identity_keys(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_text(key, name=f"{name} item") for key in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or not value or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be one nonempty sequence")
    return tuple(_finite_float(item, name=f"{name} item") for item in value)


@dataclass(frozen=True, slots=True)
class RepresentationEvaluationItem:
    """One reset-frame evaluation item and its two source-only controls."""

    partition: str
    ordinal: int
    rank: int
    task_key: str
    segment_index: int
    source_episode_index: int
    source_global_index: int
    sample_key: str
    shuffled_task_sample_key: str
    shuffled_target_sample_key: str
    factual_target_identity_keys: tuple[str, ...]
    shuffled_task_target_identity_keys: tuple[str, ...]
    shuffled_target_target_identity_keys: tuple[str, ...]
    factual_task_instruction_sha256: str
    shuffled_task_instruction_sha256: str
    shuffled_target_instruction_sha256: str

    def __post_init__(self) -> None:
        if self.partition not in REPRESENTATION_EVALUATION_PARTITIONS:
            raise ValueError("representation evaluation partition is unsupported")
        for name, value in (
            ("ordinal", self.ordinal),
            ("rank", self.rank),
            ("segment index", self.segment_index),
            ("source episode index", self.source_episode_index),
            ("source global index", self.source_global_index),
        ):
            _nonnegative_int(value, name=f"representation evaluation {name}")
        if self.rank >= REPRESENTATION_EVALUATION_WORLD_SIZE:
            raise ValueError("representation evaluation rank is outside the two-rank topology")
        for name, value in (
            ("task key", self.task_key),
            ("sample key", self.sample_key),
            ("shuffled task sample key", self.shuffled_task_sample_key),
            ("shuffled target sample key", self.shuffled_target_sample_key),
        ):
            _text(value, name=f"representation evaluation {name}")
        if self.sample_key in {
            self.shuffled_task_sample_key,
            self.shuffled_target_sample_key,
        }:
            raise ValueError("representation evaluation control retained its factual sample")
        identity_groups = (
            self.factual_target_identity_keys,
            self.shuffled_task_target_identity_keys,
            self.shuffled_target_target_identity_keys,
        )
        if any(
            not isinstance(group, tuple)
            or any(not isinstance(key, str) or not key for key in group)
            or len(set(group)) != len(group)
            for group in identity_groups
        ):
            raise ValueError("representation evaluation target identities are malformed")
        if self.factual_target_identity_keys:
            if (
                not self.shuffled_task_target_identity_keys
                or not self.shuffled_target_target_identity_keys
                or set(self.factual_target_identity_keys)
                & set(self.shuffled_task_target_identity_keys)
                or set(self.factual_target_identity_keys)
                & set(self.shuffled_target_target_identity_keys)
            ):
                raise ValueError(
                    "representation evaluation control retained a factual target identity"
                )
        elif self.shuffled_task_target_identity_keys or self.shuffled_target_target_identity_keys:
            raise ValueError(
                "representation evaluation inexact task controls must remain target-inexact"
            )
        instruction_sha256 = (
            _sha256(
                self.factual_task_instruction_sha256,
                name="representation factual instruction",
            ),
            _sha256(
                self.shuffled_task_instruction_sha256,
                name="representation shuffled-task instruction",
            ),
            _sha256(
                self.shuffled_target_instruction_sha256,
                name="representation shuffled-target instruction",
            ),
        )
        if len(set(instruction_sha256)) != len(instruction_sha256):
            raise ValueError("representation evaluation control retained an instruction")

    def as_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "ordinal": self.ordinal,
            "rank": self.rank,
            "task_key": self.task_key,
            "segment_index": self.segment_index,
            "source_episode_index": self.source_episode_index,
            "source_global_index": self.source_global_index,
            "sample_key": self.sample_key,
            "shuffled_task_sample_key": self.shuffled_task_sample_key,
            "shuffled_target_sample_key": self.shuffled_target_sample_key,
            "factual_target_identity_keys": list(self.factual_target_identity_keys),
            "shuffled_task_target_identity_keys": list(self.shuffled_task_target_identity_keys),
            "shuffled_target_target_identity_keys": list(self.shuffled_target_target_identity_keys),
            "factual_task_instruction_sha256": self.factual_task_instruction_sha256,
            "shuffled_task_instruction_sha256": self.shuffled_task_instruction_sha256,
            "shuffled_target_instruction_sha256": self.shuffled_target_instruction_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> RepresentationEvaluationItem:
        if not isinstance(value, Mapping) or set(value) != _PLAN_ITEM_FIELDS:
            raise ValueError("representation evaluation item fields differ from schema")
        return cls(
            partition=_text(value["partition"], name="representation evaluation partition"),
            ordinal=_nonnegative_int(
                value["ordinal"],
                name="representation evaluation ordinal",
            ),
            rank=_nonnegative_int(value["rank"], name="representation evaluation rank"),
            task_key=_text(value["task_key"], name="representation evaluation task key"),
            segment_index=_nonnegative_int(
                value["segment_index"],
                name="representation evaluation segment index",
            ),
            source_episode_index=_nonnegative_int(
                value["source_episode_index"],
                name="representation evaluation source episode index",
            ),
            source_global_index=_nonnegative_int(
                value["source_global_index"],
                name="representation evaluation source global index",
            ),
            sample_key=_text(
                value["sample_key"],
                name="representation evaluation sample key",
            ),
            shuffled_task_sample_key=_text(
                value["shuffled_task_sample_key"],
                name="representation shuffled task sample key",
            ),
            shuffled_target_sample_key=_text(
                value["shuffled_target_sample_key"],
                name="representation shuffled target sample key",
            ),
            factual_target_identity_keys=_identity_keys(
                value["factual_target_identity_keys"],
                name="representation factual target identities",
            ),
            shuffled_task_target_identity_keys=_identity_keys(
                value["shuffled_task_target_identity_keys"],
                name="representation shuffled-task target identities",
            ),
            shuffled_target_target_identity_keys=_identity_keys(
                value["shuffled_target_target_identity_keys"],
                name="representation shuffled-target identities",
            ),
            factual_task_instruction_sha256=_sha256(
                value["factual_task_instruction_sha256"],
                name="representation factual instruction",
            ),
            shuffled_task_instruction_sha256=_sha256(
                value["shuffled_task_instruction_sha256"],
                name="representation shuffled-task instruction",
            ),
            shuffled_target_instruction_sha256=_sha256(
                value["shuffled_target_instruction_sha256"],
                name="representation shuffled-target instruction",
            ),
        )


@dataclass(frozen=True, slots=True)
class RepresentationEvaluationPlan:
    """Content-addressed reset or warm bank with bijective negative controls."""

    representation_split_sha256: str
    items: tuple[RepresentationEvaluationItem, ...]
    world_size: int = REPRESENTATION_EVALUATION_WORLD_SIZE
    schema: str = REPRESENTATION_EVALUATION_PLAN_SCHEMA
    evaluation_reference_plan_sha256: str | None = None
    history_transitions: int = 0

    def __post_init__(self) -> None:
        if self.schema not in {
            REPRESENTATION_EVALUATION_PLAN_SCHEMA,
            REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA,
            REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA,
        }:
            raise ValueError("representation evaluation plan schema changed")
        if self.schema == REPRESENTATION_EVALUATION_PLAN_SCHEMA:
            if self.evaluation_reference_plan_sha256 is not None or self.history_transitions != 0:
                raise ValueError("v3 representation evaluation plan cannot name a reference")
        elif self.schema == REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA:
            if self.evaluation_reference_plan_sha256 is None or self.history_transitions != 0:
                raise ValueError("v4 representation evaluation plan requires its v3 reference")
            _sha256(
                self.evaluation_reference_plan_sha256,
                name="representation evaluation reference plan",
            )
        elif self.evaluation_reference_plan_sha256 is not None:
            raise ValueError("warm representation evaluation cannot name a reset reference")
        else:
            if (
                isinstance(self.history_transitions, bool)
                or not isinstance(self.history_transitions, int)
                or self.history_transitions <= 0
            ):
                raise ValueError("warm representation evaluation requires positive history")
        _sha256(
            self.representation_split_sha256,
            name="representation evaluation split",
        )
        if self.world_size != REPRESENTATION_EVALUATION_WORLD_SIZE:
            raise ValueError("representation evaluation world size changed")
        if not self.items:
            raise ValueError("representation evaluation plan is empty")
        expected = tuple(
            sorted(
                self.items,
                key=lambda item: (item.partition, item.task_key, item.segment_index),
            )
        )
        if self.items != expected:
            raise ValueError("representation evaluation items are not canonically ordered")
        for partition in REPRESENTATION_EVALUATION_PARTITIONS:
            values = tuple(item for item in self.items if item.partition == partition)
            self._validate_partition(partition, values)

    @staticmethod
    def _validate_partition(
        partition: str,
        items: tuple[RepresentationEvaluationItem, ...],
    ) -> None:
        if not items:
            raise ValueError(f"representation evaluation {partition} partition is empty")
        if tuple(item.ordinal for item in items) != tuple(range(len(items))):
            raise ValueError(f"representation evaluation {partition} ordinals are not contiguous")
        if any(item.rank != item.ordinal % REPRESENTATION_EVALUATION_WORLD_SIZE for item in items):
            raise ValueError(f"representation evaluation {partition} rank sharding changed")
        factual = tuple(item.sample_key for item in items)
        if len(set(factual)) != len(factual):
            raise ValueError(f"representation evaluation {partition} samples are not unique")
        task_controls = tuple(item.shuffled_task_sample_key for item in items)
        target_controls = tuple(item.shuffled_target_sample_key for item in items)
        if sorted(task_controls) != sorted(factual) or sorted(target_controls) != sorted(factual):
            raise ValueError(f"representation evaluation {partition} controls are not bijections")
        task_by_sample = {item.sample_key: item.task_key for item in items}
        instruction_by_sample = {
            item.sample_key: item.factual_task_instruction_sha256 for item in items
        }
        identities_by_sample = {
            item.sample_key: item.factual_target_identity_keys for item in items
        }
        if any(
            task_by_sample[item.shuffled_task_sample_key] == item.task_key
            or task_by_sample[item.shuffled_target_sample_key] == item.task_key
            for item in items
        ):
            raise ValueError(
                f"representation evaluation {partition} control retained its factual task"
            )
        if any(
            instruction_by_sample[item.shuffled_task_sample_key]
            != item.shuffled_task_instruction_sha256
            or instruction_by_sample[item.shuffled_target_sample_key]
            != item.shuffled_target_instruction_sha256
            for item in items
        ):
            raise ValueError(
                f"representation evaluation {partition} control instruction provenance changed"
            )
        if any(
            identities_by_sample[item.shuffled_task_sample_key]
            != item.shuffled_task_target_identity_keys
            or identities_by_sample[item.shuffled_target_sample_key]
            != item.shuffled_target_target_identity_keys
            for item in items
        ):
            raise ValueError(
                f"representation evaluation {partition} control target provenance changed"
            )
        rank_counts = [
            sum(item.rank == rank for item in items)
            for rank in range(REPRESENTATION_EVALUATION_WORLD_SIZE)
        ]
        if max(rank_counts) - min(rank_counts) > 1:
            raise ValueError(f"representation evaluation {partition} rank load is imbalanced")

    def _payload(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": self.schema,
            "representation_split_sha256": self.representation_split_sha256,
            "world_size": self.world_size,
            "items": [item.as_dict() for item in self.items],
        }
        if self.schema == REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA:
            value["evaluation_reference_plan_sha256"] = self.evaluation_reference_plan_sha256
        elif self.schema == REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA:
            value["history_transitions"] = self.history_transitions
        return value

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    @property
    def replay_seed_sha256(self) -> str:
        """Stable seed identity inherited from the exact historical evaluation bank."""

        if self.evaluation_reference_plan_sha256 is not None:
            return self.evaluation_reference_plan_sha256
        return self.artifact_sha256

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def items_for(self, partition: str, rank: int) -> tuple[RepresentationEvaluationItem, ...]:
        if partition not in REPRESENTATION_EVALUATION_PARTITIONS:
            raise ValueError("representation evaluation partition is unsupported")
        _nonnegative_int(rank, name="representation evaluation rank")
        if rank >= self.world_size:
            raise ValueError("representation evaluation rank is outside the plan")
        return tuple(
            item for item in self.items if item.partition == partition and item.rank == rank
        )

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{self.artifact_sha256[:12]}"
        )
        if (
            destination.exists()
            or destination.is_symlink()
            or temporary.exists()
            or temporary.is_symlink()
        ):
            raise FileExistsError(f"representation evaluation plan path exists: {destination}")
        payload = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        try:
            with temporary.open("x", encoding="ascii") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def from_dict(cls, value: object) -> RepresentationEvaluationPlan:
        if not isinstance(value, Mapping):
            raise ValueError("representation evaluation plan fields differ from schema")
        schema = _text(value.get("schema"), name="representation evaluation plan schema")
        expected_fields = (
            _REFERENCE_PLAN_FIELDS
            if schema == REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA
            else _WARM_PLAN_FIELDS
            if schema == REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA
            else _PLAN_FIELDS
        )
        if set(value) != expected_fields:
            raise ValueError("representation evaluation plan fields differ from schema")
        raw_items = value["items"]
        if not isinstance(raw_items, list):
            raise ValueError("representation evaluation plan items must be a list")
        plan = cls(
            schema=schema,
            representation_split_sha256=_sha256(
                value["representation_split_sha256"],
                name="representation evaluation split",
            ),
            world_size=_nonnegative_int(
                value["world_size"],
                name="representation evaluation world size",
            ),
            items=tuple(RepresentationEvaluationItem.from_dict(item) for item in raw_items),
            evaluation_reference_plan_sha256=(
                _sha256(
                    value["evaluation_reference_plan_sha256"],
                    name="representation evaluation reference plan",
                )
                if schema == REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA
                else None
            ),
            history_transitions=(
                _nonnegative_int(
                    value["history_transitions"],
                    name="representation warm evaluation history",
                )
                if schema == REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA
                else 0
            ),
        )
        expected = _sha256(
            value["artifact_sha256"],
            name="representation evaluation plan artifact",
        )
        if plan.artifact_sha256 != expected:
            raise ValueError("representation evaluation plan artifact SHA-256 changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> RepresentationEvaluationPlan:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid representation evaluation plan: {source}") from error
        return cls.from_dict(value)


def _segment_sample_key(
    dataset: CalvinStatefulTransitionDataset,
    segment: RepresentationEvaluationSegment,
    *,
    transition_index: int,
) -> tuple[str, int]:
    _nonnegative_int(
        transition_index,
        name="representation evaluation transition index",
    )
    try:
        source = dataset.index.segments[segment.segment_index]
        episode = dataset.episode_manifest[segment.segment_index]
    except IndexError as error:
        raise ValueError("representation evaluation segment is absent from the dataset") from error
    if (
        source.index != segment.segment_index
        or source.task_key != segment.task_key
        or int(source.episode_index) != segment.source_episode_index
        or int(source.start) != segment.source_start
        or int(source.end) != segment.source_end
        or episode.segment_index != segment.segment_index
        or transition_index >= len(episode.sample_keys)
    ):
        raise ValueError("representation evaluation segment differs from immutable source")
    sample_key = episode.sample_keys[transition_index]
    locator = dataset.locator_by_key(sample_key)
    if (
        locator.segment_index != segment.segment_index
        or locator.global_index != segment.source_start + transition_index
    ):
        raise ValueError("representation evaluation sample changed source coordinates")
    return sample_key, locator.global_index


def _target_disjoint_donor_rotations(
    group: tuple[str, ...],
    target_identities_by_task: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build two deterministic task bijections without target-identity false negatives."""

    if not group:
        return {}, {}
    if len(group) < 3:
        raise ValueError("representation evaluation target-equivalence stratum needs three tasks")
    if set(group) != set(target_identities_by_task):
        raise ValueError("representation evaluation target-equivalence tasks differ")
    exact = bool(target_identities_by_task[group[0]])
    if any(bool(target_identities_by_task[task]) is not exact for task in group):
        raise ValueError("representation target-equivalence stratum mixed exactness")
    valid_offsets = tuple(
        offset
        for offset in range(1, len(group))
        if not exact
        or all(
            set(target_identities_by_task[task]).isdisjoint(
                target_identities_by_task[group[(index + offset) % len(group)]]
            )
            for index, task in enumerate(group)
        )
    )
    if len(valid_offsets) < 2:
        raise ValueError("representation evaluation cannot form two disjoint target controls")

    def mapping(offset: int) -> dict[str, str]:
        return {task: group[(index + offset) % len(group)] for index, task in enumerate(group)}

    return mapping(valid_offsets[0]), mapping(valid_offsets[1])


def _partition_items(
    dataset: CalvinStatefulTransitionDataset,
    *,
    partition: str,
    segments: tuple[RepresentationEvaluationSegment, ...],
    task_identity_resolver: TaskIdentityResolver,
    history_transitions: int = 0,
    replicates_per_task: int | None = None,
) -> tuple[RepresentationEvaluationItem, ...]:
    if replicates_per_task is not None and (
        isinstance(replicates_per_task, bool)
        or not isinstance(replicates_per_task, int)
        or replicates_per_task <= 0
    ):
        raise ValueError("representation evaluation replicate count must be positive")
    by_task: dict[
        str,
        list[tuple[RepresentationEvaluationSegment, str, str, int]],
    ] = defaultdict(list)
    for segment in segments:
        sample_key, source_global_index = _segment_sample_key(
            dataset,
            segment,
            transition_index=history_transitions,
        )
        source = dataset.index.segments[segment.segment_index]
        if source.task_key != segment.task_key or not source.instruction:
            raise ValueError("representation evaluation instruction differs from immutable source")
        by_task[segment.task_key].append(
            (
                segment,
                sample_key,
                hashlib.sha256(source.instruction.encode("utf-8")).hexdigest(),
                source_global_index,
            )
        )
    tasks = tuple(sorted(by_task))
    if len(tasks) < 3:
        raise ValueError("representation evaluation controls require at least three tasks")
    replicate_counts = {len(by_task[task]) for task in tasks}
    if len(replicate_counts) != 1:
        raise ValueError("representation evaluation task replication differs")
    for task in tasks:
        by_task[task].sort(key=lambda item: item[0].segment_index)
        if replicates_per_task is not None:
            if len(by_task[task]) < replicates_per_task:
                raise ValueError("representation evaluation task lacks requested replicates")
            by_task[task] = by_task[task][:replicates_per_task]

    target_identities_by_task = {
        task: _identity_keys(
            task_identity_resolver(task) or (),
            name=f"representation target identities for {task}",
        )
        for task in tasks
    }

    exact_tasks = tuple(task for task in tasks if target_identities_by_task[task])
    inexact_tasks = tuple(task for task in tasks if not target_identities_by_task[task])
    exact_identities = {task: target_identities_by_task[task] for task in exact_tasks}
    inexact_identities = {task: target_identities_by_task[task] for task in inexact_tasks}
    exact_task_donor, exact_target_donor = _target_disjoint_donor_rotations(
        exact_tasks,
        exact_identities,
    )
    inexact_task_donor, inexact_target_donor = _target_disjoint_donor_rotations(
        inexact_tasks,
        inexact_identities,
    )
    task_donor_by_task = {**exact_task_donor, **inexact_task_donor}
    target_donor_by_task = {**exact_target_donor, **inexact_target_donor}
    if set(task_donor_by_task) != set(tasks) or set(target_donor_by_task) != set(tasks):
        raise RuntimeError("representation evaluation donor construction lost a task")

    factual: list[tuple[RepresentationEvaluationSegment, str, str, int]] = []
    task_donor: dict[str, str] = {}
    target_donor: dict[str, str] = {}
    instruction_by_sample: dict[str, str] = {}
    for task in tasks:
        next_task = task_donor_by_task[task]
        target_task = target_donor_by_task[task]
        for replicate_index, item in enumerate(by_task[task]):
            factual.append(item)
            task_donor[item[1]] = by_task[next_task][replicate_index][1]
            target_donor[item[1]] = by_task[target_task][replicate_index][1]
            instruction_by_sample[item[1]] = item[2]
    factual.sort(key=lambda item: (item[0].task_key, item[0].segment_index))
    return tuple(
        RepresentationEvaluationItem(
            partition=partition,
            ordinal=ordinal,
            rank=ordinal % REPRESENTATION_EVALUATION_WORLD_SIZE,
            task_key=segment.task_key,
            segment_index=segment.segment_index,
            source_episode_index=segment.source_episode_index,
            source_global_index=source_global_index,
            sample_key=sample_key,
            shuffled_task_sample_key=task_donor[sample_key],
            shuffled_target_sample_key=target_donor[sample_key],
            factual_target_identity_keys=target_identities_by_task[segment.task_key],
            shuffled_task_target_identity_keys=target_identities_by_task[
                task_donor_by_task[segment.task_key]
            ],
            shuffled_target_target_identity_keys=target_identities_by_task[
                target_donor_by_task[segment.task_key]
            ],
            factual_task_instruction_sha256=instruction_sha256,
            shuffled_task_instruction_sha256=instruction_by_sample[task_donor[sample_key]],
            shuffled_target_instruction_sha256=instruction_by_sample[target_donor[sample_key]],
        )
        for ordinal, (
            segment,
            sample_key,
            instruction_sha256,
            source_global_index,
        ) in enumerate(factual)
    )


def build_representation_evaluation_plan(
    split: RepresentationTrialSplit,
    dataset: CalvinStatefulTransitionDataset,
    *,
    task_identity_resolver: TaskIdentityResolver,
    evaluation_reference_plan_sha256: str | None = None,
) -> RepresentationEvaluationPlan:
    """Build two deterministic reset-frame banks without decoding observations."""

    if (
        not isinstance(split, RepresentationTrialSplit)
        or not isinstance(dataset, CalvinStatefulTransitionDataset)
        or not callable(task_identity_resolver)
    ):
        raise TypeError("representation evaluation planning requires typed split and dataset")
    manifest = dataset.index.dataset_manifest
    if (
        manifest is None
        or manifest.tree_sha256 != split.dataset_manifest_sha256
        or dataset.index.dataset_id != split.dataset_id
        or dataset.index.dataset_revision != split.dataset_revision
    ):
        raise ValueError("representation evaluation dataset differs from the split")
    items = (
        *_partition_items(
            dataset,
            partition="validation",
            segments=split.validation_segments,
            task_identity_resolver=task_identity_resolver,
        ),
        *_partition_items(
            dataset,
            partition="heldout",
            segments=split.heldout_segments,
            task_identity_resolver=task_identity_resolver,
        ),
    )
    return RepresentationEvaluationPlan(
        representation_split_sha256=split.artifact_sha256,
        items=tuple(
            sorted(
                items,
                key=lambda item: (item.partition, item.task_key, item.segment_index),
            )
        ),
        schema=(
            REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA
            if evaluation_reference_plan_sha256 is not None
            else REPRESENTATION_EVALUATION_PLAN_SCHEMA
        ),
        evaluation_reference_plan_sha256=evaluation_reference_plan_sha256,
    )


def build_representation_warm_evaluation_plan(
    split: RepresentationTrialSplit,
    dataset: CalvinStatefulTransitionDataset,
    *,
    task_identity_resolver: TaskIdentityResolver,
    history_transitions: int = 8,
) -> RepresentationEvaluationPlan:
    """Build one source-only fixed-history sample per task and partition."""

    if (
        not isinstance(split, RepresentationTrialSplit)
        or not isinstance(dataset, CalvinStatefulTransitionDataset)
        or not callable(task_identity_resolver)
    ):
        raise TypeError("warm representation planning requires typed split and dataset")
    if (
        isinstance(history_transitions, bool)
        or not isinstance(history_transitions, int)
        or history_transitions <= 0
    ):
        raise ValueError("warm evaluation requires positive history transitions")
    manifest = dataset.index.dataset_manifest
    if (
        manifest is None
        or manifest.tree_sha256 != split.dataset_manifest_sha256
        or dataset.index.dataset_id != split.dataset_id
        or dataset.index.dataset_revision != split.dataset_revision
    ):
        raise ValueError("warm representation evaluation dataset differs from the split")
    items = (
        *_partition_items(
            dataset,
            partition="validation",
            segments=split.validation_segments,
            task_identity_resolver=task_identity_resolver,
            history_transitions=history_transitions,
            replicates_per_task=1,
        ),
        *_partition_items(
            dataset,
            partition="heldout",
            segments=split.heldout_segments,
            task_identity_resolver=task_identity_resolver,
            history_transitions=history_transitions,
            replicates_per_task=1,
        ),
    )
    return RepresentationEvaluationPlan(
        representation_split_sha256=split.artifact_sha256,
        items=tuple(
            sorted(
                items,
                key=lambda item: (item.partition, item.task_key, item.segment_index),
            )
        ),
        schema=REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA,
        history_transitions=history_transitions,
    )


def build_representation_token_evidence(
    logits: Sequence[float],
    target_mass: Sequence[float],
) -> dict[str, object]:
    """Persist raw task-token evidence and its existing fractional metrics."""

    logit_values = _finite_vector(logits, name="representation token logits")
    mass_values = _finite_vector(target_mass, name="representation token target mass")
    if len(logit_values) != len(mass_values):
        raise ValueError("representation token logits and target mass differ")
    value = {
        "schema": REPRESENTATION_TOKEN_EVIDENCE_SCHEMA,
        "logits": list(logit_values),
        "target_mass": list(mass_values),
        "metrics": fractional_token_metrics(logit_values, mass_values),
    }
    return validate_representation_token_evidence(value)


def representation_target_mass_sha256(
    identity_keys: Sequence[str],
    target_mass: Sequence[float],
) -> str:
    """Bind one post-forward target vector to its preregistered identities."""

    identities = _identity_keys(
        identity_keys,
        name="representation target-mass identities",
    )
    mass = _finite_vector(
        target_mass,
        name="representation target-mass vector",
    )
    if any(value < 0.0 or value > 1.0 for value in mass):
        raise ValueError("representation target mass must lie in [0,1]")
    return _canonical_sha256(
        {
            "schema": "picf-next.lingbot-representation-target-mass.v1",
            "identity_keys": list(identities),
            "target_mass": list(mass),
        }
    )


def validate_representation_token_evidence(value: object) -> dict[str, Any]:
    """Recompute token evidence from persisted logits and fractional masks."""

    if not isinstance(value, dict) or set(value) != _TOKEN_FIELDS:
        raise ValueError("representation token evidence fields differ from schema")
    if value["schema"] != REPRESENTATION_TOKEN_EVIDENCE_SCHEMA:
        raise ValueError("representation token evidence schema changed")
    logits = _finite_vector(value["logits"], name="representation token logits")
    target_mass = _finite_vector(
        value["target_mass"],
        name="representation token target mass",
    )
    if len(logits) != len(target_mass):
        raise ValueError("representation token logits and target mass differ")
    expected = fractional_token_metrics(logits, target_mass)
    if value["metrics"] != expected:
        raise ValueError("representation token metrics were not recomputed")
    return value


def _representation_ownership_row_value(
    *,
    row_index: int,
    track_index: int,
    identity_key: str,
    is_task_target: bool,
    prediction: Sequence[float],
    target: Sequence[float],
    weight: Sequence[float],
) -> dict[str, object]:

    row_index = _nonnegative_int(row_index, name="representation ownership row index")
    track_index = _nonnegative_int(track_index, name="representation ownership track index")
    identity_key = _text(identity_key, name="representation ownership identity")
    if not isinstance(is_task_target, bool):
        raise TypeError("representation ownership task-target flag must be boolean")
    prediction_values = _finite_vector(
        prediction,
        name="representation ownership prediction",
    )
    target_values = _finite_vector(target, name="representation ownership target")
    weight_values = _finite_vector(weight, name="representation ownership weight")
    if len({len(prediction_values), len(target_values), len(weight_values)}) != 1:
        raise ValueError("representation ownership row vectors differ")
    if any(not 0.0 <= item <= 1.0 for item in (*prediction_values, *target_values)):
        raise ValueError("representation ownership probabilities must lie in [0,1]")
    if any(item <= 0.0 for item in weight_values):
        raise ValueError("representation ownership valid-token weights must be positive")

    intersection = math.fsum(
        measured * expected * token_weight
        for measured, expected, token_weight in zip(
            prediction_values,
            target_values,
            weight_values,
            strict=True,
        )
    )
    union = math.fsum(
        token_weight * (measured + expected - measured * expected)
        for measured, expected, token_weight in zip(
            prediction_values,
            target_values,
            weight_values,
            strict=True,
        )
    )
    prediction_mass = math.fsum(
        measured * token_weight
        for measured, token_weight in zip(
            prediction_values,
            weight_values,
            strict=True,
        )
    )
    prediction_target_mass = intersection
    if union <= 0.0 or prediction_mass <= 0.0:
        raise ValueError("representation ownership row has no measurable support")
    value = {
        "schema": REPRESENTATION_OWNERSHIP_ROW_SCHEMA,
        "row_index": row_index,
        "track_index": track_index,
        "identity_key": identity_key,
        "is_task_target": is_task_target,
        "prediction": list(prediction_values),
        "target": list(target_values),
        "weight": list(weight_values),
        "valid_token_count": len(prediction_values),
        "intersection": intersection,
        "union": union,
        "prediction_mass": prediction_mass,
        "prediction_target_mass": prediction_target_mass,
        "soft_iou": intersection / union,
        "target_mass_concentration": prediction_target_mass / prediction_mass,
    }
    return value


def build_representation_ownership_row(
    *,
    row_index: int,
    track_index: int,
    identity_key: str,
    is_task_target: bool,
    prediction: Sequence[float],
    target: Sequence[float],
    weight: Sequence[float],
) -> dict[str, object]:
    """Persist one matched row's weighted soft-IoU and mass concentration."""

    value = _representation_ownership_row_value(
        row_index=row_index,
        track_index=track_index,
        identity_key=identity_key,
        is_task_target=is_task_target,
        prediction=prediction,
        target=target,
        weight=weight,
    )
    return validate_representation_ownership_row(value)


def validate_representation_ownership_row(value: object) -> dict[str, Any]:
    """Recompute one ownership row from persisted valid-token primitives."""

    if not isinstance(value, dict) or set(value) != _OWNERSHIP_ROW_FIELDS:
        raise ValueError("representation ownership row fields differ from schema")
    if value["schema"] != REPRESENTATION_OWNERSHIP_ROW_SCHEMA:
        raise ValueError("representation ownership row schema changed")
    expected = _representation_ownership_row_value(
        row_index=_nonnegative_int(
            value["row_index"],
            name="representation ownership row index",
        ),
        track_index=_nonnegative_int(
            value["track_index"],
            name="representation ownership track index",
        ),
        identity_key=_text(
            value["identity_key"],
            name="representation ownership identity",
        ),
        is_task_target=value["is_task_target"],
        prediction=_finite_vector(
            value["prediction"],
            name="representation ownership prediction",
        ),
        target=_finite_vector(
            value["target"],
            name="representation ownership target",
        ),
        weight=_finite_vector(
            value["weight"],
            name="representation ownership weight",
        ),
    )
    if value != expected:
        raise ValueError("representation ownership row metrics were not recomputed")
    return value


def summarize_representation_ownership_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate all visible matched rows and the exact task-target subset."""

    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes) or not rows:
        raise ValueError("representation ownership summary requires visible rows")
    validated = tuple(validate_representation_ownership_row(dict(row)) for row in rows)
    row_indices = [int(row["row_index"]) for row in validated]
    track_indices = [int(row["track_index"]) for row in validated]
    if len(set(row_indices)) != len(row_indices) or len(set(track_indices)) != len(track_indices):
        raise ValueError("representation ownership summary reused a row or track")
    targets = tuple(row for row in validated if row["is_task_target"])
    target_soft_iou = (
        None if not targets else math.fsum(float(row["soft_iou"]) for row in targets) / len(targets)
    )
    target_prediction_mass = math.fsum(float(row["prediction_mass"]) for row in targets)
    target_prediction_inside = math.fsum(float(row["prediction_target_mass"]) for row in targets)
    concentration = None if not targets else target_prediction_inside / target_prediction_mass
    value = {
        "schema": REPRESENTATION_OWNERSHIP_SUMMARY_SCHEMA,
        "row_count": len(validated),
        "task_target_row_count": len(targets),
        "macro_soft_iou": (math.fsum(float(row["soft_iou"]) for row in validated) / len(validated)),
        "target_soft_iou": target_soft_iou,
        "target_mass_concentration": concentration,
    }
    return validate_representation_ownership_summary(value, rows=validated)


def validate_representation_ownership_summary(
    value: object,
    *,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Recompute one ownership summary from its persisted row evidence."""

    if not isinstance(value, dict) or set(value) != _OWNERSHIP_SUMMARY_FIELDS:
        raise ValueError("representation ownership summary fields differ from schema")
    if value["schema"] != REPRESENTATION_OWNERSHIP_SUMMARY_SCHEMA:
        raise ValueError("representation ownership summary schema changed")
    validated = tuple(validate_representation_ownership_row(dict(row)) for row in rows)
    if not validated:
        raise ValueError("representation ownership summary has no visible rows")
    targets = tuple(row for row in validated if row["is_task_target"])
    expected = {
        "schema": REPRESENTATION_OWNERSHIP_SUMMARY_SCHEMA,
        "row_count": len(validated),
        "task_target_row_count": len(targets),
        "macro_soft_iou": (math.fsum(float(row["soft_iou"]) for row in validated) / len(validated)),
        "target_soft_iou": (
            None
            if not targets
            else math.fsum(float(row["soft_iou"]) for row in targets) / len(targets)
        ),
        "target_mass_concentration": (
            None
            if not targets
            else math.fsum(float(row["prediction_target_mass"]) for row in targets)
            / math.fsum(float(row["prediction_mass"]) for row in targets)
        ),
    }
    if value != expected:
        raise ValueError("representation ownership summary was not recomputed")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    return math.fsum(values) / len(values)


def _optional_mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _validate_evaluation_visual(
    value: object,
    *,
    checkpoint_global_step: int,
    rank: int,
    sample_key: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("representation evaluation visual must be a mapping")
    required = {
        "schema",
        "path",
        "sha256",
        "bytes",
        "global_step",
        "input_weight_global_step",
        "weight_boundary",
        "rank",
        "sample_key",
        "task",
        "loss_only_labels_visible_to_model",
    }
    if not required.issubset(value):
        raise ValueError("representation evaluation visual omits provenance")
    relative_value = value["path"]
    relative = PurePosixPath(relative_value) if isinstance(relative_value, str) else None
    if (
        value["schema"] != NATIVE_VISUAL_AUDIT_SCHEMA
        or value["global_step"] != checkpoint_global_step
        or value["input_weight_global_step"] != checkpoint_global_step
        or value["weight_boundary"] != "checkpoint_evaluation"
        or value["rank"] != rank
        or value["sample_key"] != sample_key
        or value["loss_only_labels_visible_to_model"] is not False
        or relative is None
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("representation evaluation visual provenance changed")
    _sha256(value["sha256"], name="representation evaluation visual")
    _positive_int(value["bytes"], name="representation evaluation visual bytes")
    _text(value["task"], name="representation evaluation visual task")
    return value


def build_representation_evaluation_sample(
    *,
    checkpoint_global_step: int,
    item: RepresentationEvaluationItem,
    factual_task_instruction_sha256: str,
    shuffled_task_instruction_sha256: str,
    shuffled_target_instruction_sha256: str,
    factual_token_evidence: Mapping[str, object],
    shuffled_task_token_evidence: Mapping[str, object],
    shuffled_target_token_evidence: Mapping[str, object],
    factual_task_row_diagnostic: Mapping[str, object],
    shuffled_task_row_diagnostic: Mapping[str, object],
    factual_ownership_rows: Sequence[Mapping[str, object]],
    factual_ownership_summary: Mapping[str, object],
    shuffled_task_ownership_rows: Sequence[Mapping[str, object]],
    shuffled_task_ownership_summary: Mapping[str, object],
    official_action_loss: float,
    factual_forward_seconds: float,
    shuffled_task_forward_seconds: float,
    peak_cuda_reserved_bytes: int,
    factual_relation_sha256: str,
    factual_target_sha256: str,
    shuffled_task_relation_sha256: str,
    shuffled_task_target_sha256: str,
    shuffled_target_target_sha256: str,
    visual_artifact: Mapping[str, object],
) -> dict[str, object]:
    """Build one recomputable factual/control record after both forwards."""

    if not isinstance(item, RepresentationEvaluationItem):
        raise TypeError("representation evaluation sample requires a planned item")
    value = {
        "schema": REPRESENTATION_EVALUATION_SAMPLE_SCHEMA,
        "checkpoint_global_step": checkpoint_global_step,
        **item.as_dict(),
        "factual_task_instruction_sha256": factual_task_instruction_sha256,
        "shuffled_task_instruction_sha256": shuffled_task_instruction_sha256,
        "shuffled_target_instruction_sha256": shuffled_target_instruction_sha256,
        "factual_token_evidence": dict(factual_token_evidence),
        "shuffled_task_token_evidence": dict(shuffled_task_token_evidence),
        "shuffled_target_token_evidence": dict(shuffled_target_token_evidence),
        "factual_task_row_diagnostic": dict(factual_task_row_diagnostic),
        "shuffled_task_row_diagnostic": dict(shuffled_task_row_diagnostic),
        "factual_ownership_rows": [dict(row) for row in factual_ownership_rows],
        "factual_ownership_summary": dict(factual_ownership_summary),
        "shuffled_task_ownership_rows": [dict(row) for row in shuffled_task_ownership_rows],
        "shuffled_task_ownership_summary": dict(shuffled_task_ownership_summary),
        "official_action_loss": official_action_loss,
        "forward_seconds": {
            "factual": factual_forward_seconds,
            "shuffled_task": shuffled_task_forward_seconds,
        },
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "tensor_sha256": {
            "factual_relation": factual_relation_sha256,
            "factual_target": factual_target_sha256,
            "shuffled_task_relation": shuffled_task_relation_sha256,
            "shuffled_task_target": shuffled_task_target_sha256,
            "shuffled_target_target": shuffled_target_target_sha256,
        },
        "visual_artifact": dict(visual_artifact),
        "loss_only_labels_visible_to_model": False,
        "target_resolution_happened_after_forward": True,
    }
    return validate_representation_evaluation_sample(value, expected_item=item)


def validate_representation_evaluation_sample(
    value: object,
    *,
    expected_item: RepresentationEvaluationItem | None = None,
) -> dict[str, Any]:
    """Recompute one evaluation sample from its persisted primitive evidence."""

    if not isinstance(value, dict) or set(value) != _SAMPLE_FIELDS:
        raise ValueError("representation evaluation sample fields differ from schema")
    if value["schema"] != REPRESENTATION_EVALUATION_SAMPLE_SCHEMA:
        raise ValueError("representation evaluation sample schema changed")
    checkpoint_global_step = _nonnegative_int(
        value["checkpoint_global_step"],
        name="representation evaluation checkpoint step",
    )
    item = RepresentationEvaluationItem.from_dict({name: value[name] for name in _PLAN_ITEM_FIELDS})
    if expected_item is not None and item != expected_item:
        raise ValueError("representation evaluation sample differs from its frozen plan")
    factual_instruction = _sha256(
        value["factual_task_instruction_sha256"],
        name="representation factual instruction",
    )
    shuffled_instruction = _sha256(
        value["shuffled_task_instruction_sha256"],
        name="representation shuffled instruction",
    )
    shuffled_target_instruction = _sha256(
        value["shuffled_target_instruction_sha256"],
        name="representation shuffled-target instruction",
    )
    if len({factual_instruction, shuffled_instruction, shuffled_target_instruction}) != 3:
        raise ValueError("representation control retained an instruction")

    factual_token = validate_representation_token_evidence(dict(value["factual_token_evidence"]))
    shuffled_task_token = validate_representation_token_evidence(
        dict(value["shuffled_task_token_evidence"])
    )
    shuffled_target_token = validate_representation_token_evidence(
        dict(value["shuffled_target_token_evidence"])
    )
    if factual_token["target_mass"] != shuffled_task_token["target_mass"]:
        raise ValueError("representation shuffled-task control changed the factual target")
    if factual_token["logits"] != shuffled_target_token["logits"]:
        raise ValueError("representation shuffled-target control changed factual logits")

    validate_task_row_diagnostic(dict(value["factual_task_row_diagnostic"]))
    validate_task_row_diagnostic(dict(value["shuffled_task_row_diagnostic"]))
    factual_rows = tuple(
        validate_representation_ownership_row(dict(row)) for row in value["factual_ownership_rows"]
    )
    shuffled_rows = tuple(
        validate_representation_ownership_row(dict(row))
        for row in value["shuffled_task_ownership_rows"]
    )
    if not factual_rows or not shuffled_rows:
        raise ValueError("representation evaluation sample has no visible ownership rows")
    validate_representation_ownership_summary(
        dict(value["factual_ownership_summary"]),
        rows=factual_rows,
    )
    validate_representation_ownership_summary(
        dict(value["shuffled_task_ownership_summary"]),
        rows=shuffled_rows,
    )

    action_loss = _finite_float(
        value["official_action_loss"],
        name="representation evaluation action loss",
    )
    if action_loss < 0:
        raise ValueError("representation evaluation action loss must be non-negative")
    forward_seconds = value["forward_seconds"]
    if not isinstance(forward_seconds, dict) or set(forward_seconds) != {
        "factual",
        "shuffled_task",
    }:
        raise ValueError("representation evaluation forward timing fields changed")
    for name in ("factual", "shuffled_task"):
        if (
            _finite_float(
                forward_seconds[name],
                name=f"representation evaluation {name} forward seconds",
            )
            <= 0
        ):
            raise ValueError("representation evaluation forward time must be positive")
    _positive_int(
        value["peak_cuda_reserved_bytes"],
        name="representation evaluation peak CUDA reservation",
    )
    tensor_sha256 = value["tensor_sha256"]
    if not isinstance(tensor_sha256, dict) or set(tensor_sha256) != {
        "factual_relation",
        "factual_target",
        "shuffled_task_relation",
        "shuffled_task_target",
        "shuffled_target_target",
    }:
        raise ValueError("representation evaluation tensor hash fields changed")
    for name, digest in tensor_sha256.items():
        _sha256(digest, name=f"representation evaluation {name}")
    if tensor_sha256["factual_target"] != tensor_sha256["shuffled_task_target"]:
        raise ValueError("representation shuffled-task control changed target tensor")
    expected_shuffled_target_sha256 = representation_target_mass_sha256(
        item.shuffled_target_target_identity_keys,
        shuffled_target_token["target_mass"],
    )
    if tensor_sha256["shuffled_target_target"] != expected_shuffled_target_sha256:
        raise ValueError("representation shuffled-target target hash was not recomputed")
    visual = _validate_evaluation_visual(
        value["visual_artifact"],
        checkpoint_global_step=checkpoint_global_step,
        rank=item.rank,
        sample_key=item.sample_key,
    )
    if hashlib.sha256(str(visual["task"]).encode("utf-8")).hexdigest() != (factual_instruction):
        raise ValueError("representation evaluation visual task differs from its instruction")
    if (
        value["loss_only_labels_visible_to_model"] is not False
        or value["target_resolution_happened_after_forward"] is not True
    ):
        raise ValueError("representation evaluation loss-side boundary changed")
    return value


def _metric(
    token_evidence: Mapping[str, object],
    name: str,
) -> float | None:
    metrics = token_evidence["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("representation token metrics are malformed")
    value = metrics[name]
    return None if value is None else _finite_float(value, name=f"representation token {name}")


def _task_level_metrics(
    samples: Sequence[dict[str, Any]],
) -> dict[str, object]:
    factual_auc: list[float] = []
    factual_margin: list[float] = []
    control_factual_auc: list[float] = []
    shuffled_task_auc: list[float] = []
    shuffled_target_auc: list[float] = []
    row_ranks: list[int] = []
    row_margins: list[float] = []
    target_soft_iou: list[float] = []
    target_concentration: list[float] = []
    macro_soft_iou: list[float] = []
    action_losses: list[float] = []
    for sample in samples:
        factual = sample["factual_token_evidence"]
        shuffled_task = sample["shuffled_task_token_evidence"]
        shuffled_target = sample["shuffled_target_token_evidence"]
        auc = _metric(factual, "fractional_weighted_auc")
        margin = _metric(factual, "target_background_logit_margin")
        if auc is not None and margin is not None:
            factual_auc.append(auc)
            factual_margin.append(margin)
        task_control_auc = _metric(shuffled_task, "fractional_weighted_auc")
        target_control_auc = _metric(shuffled_target, "fractional_weighted_auc")
        if auc is not None and task_control_auc is not None and target_control_auc is not None:
            control_factual_auc.append(auc)
            shuffled_task_auc.append(task_control_auc)
            shuffled_target_auc.append(target_control_auc)

        row = sample["factual_task_row_diagnostic"]
        rank = row["worst_target_rank"]
        row_margin = row["target_vs_hardest_negative_logit_margin"]
        if rank is not None and row_margin is not None:
            row_ranks.append(_positive_int(rank, name="representation task-row rank"))
            row_margins.append(_finite_float(row_margin, name="representation task-row margin"))
        ownership = sample["factual_ownership_summary"]
        target_iou = ownership["target_soft_iou"]
        concentration = ownership["target_mass_concentration"]
        if target_iou is not None and concentration is not None:
            target_soft_iou.append(_finite_float(target_iou, name="representation target soft-IoU"))
            target_concentration.append(
                _finite_float(
                    concentration,
                    name="representation target mass concentration",
                )
            )
        macro_soft_iou.append(
            _finite_float(
                ownership["macro_soft_iou"],
                name="representation macro soft-IoU",
            )
        )
        action_losses.append(
            _finite_float(
                sample["official_action_loss"],
                name="representation action loss",
            )
        )

    factual_auc_mean = _optional_mean(factual_auc)
    shuffled_task_auc_mean = _optional_mean(shuffled_task_auc)
    shuffled_target_auc_mean = _optional_mean(shuffled_target_auc)
    control_factual_auc_mean = _optional_mean(control_factual_auc)
    control_eligible = len(control_factual_auc)
    return {
        "sample_count": len(samples),
        "token_eligible_sample_count": len(factual_auc),
        "control_eligible_sample_count": control_eligible,
        "fractional_weighted_auc": factual_auc_mean,
        "target_background_logit_margin": _optional_mean(factual_margin),
        "shuffled_task_fractional_weighted_auc": shuffled_task_auc_mean,
        "shuffled_target_fractional_weighted_auc": shuffled_target_auc_mean,
        "shuffled_task_auc_degradation": (
            None
            if control_factual_auc_mean is None or shuffled_task_auc_mean is None
            else control_factual_auc_mean - shuffled_task_auc_mean
        ),
        "shuffled_target_auc_degradation": (
            None
            if control_factual_auc_mean is None or shuffled_target_auc_mean is None
            else control_factual_auc_mean - shuffled_target_auc_mean
        ),
        "row_eligible_sample_count": len(row_ranks),
        "worst_target_rank": None if not row_ranks else max(row_ranks),
        "hardest_negative_logit_margin": _optional_mean(row_margins),
        "ownership_eligible_sample_count": len(target_soft_iou),
        "target_ownership_soft_iou": _optional_mean(target_soft_iou),
        "target_mass_concentration": _optional_mean(target_concentration),
        "macro_ownership_soft_iou": _mean(
            macro_soft_iou,
            name="representation task macro ownership",
        ),
        "official_action_loss": _mean(
            action_losses,
            name="representation task action loss",
        ),
    }


def summarize_representation_evaluation_partition(
    samples: Sequence[Mapping[str, object]],
    *,
    partition: str,
) -> dict[str, object]:
    """Macro-average sample evidence by task before aggregating a partition."""

    if partition not in REPRESENTATION_EVALUATION_PARTITIONS:
        raise ValueError("representation evaluation partition is unsupported")
    validated = tuple(validate_representation_evaluation_sample(dict(sample)) for sample in samples)
    value = _representation_partition_value(validated, partition=partition)
    return validate_representation_evaluation_partition(
        value,
        samples=validated,
        partition=partition,
    )


def validate_representation_evaluation_partition(
    value: object,
    *,
    samples: Sequence[Mapping[str, object]],
    partition: str,
) -> dict[str, Any]:
    """Recompute one task-macro partition summary from sample evidence."""

    if not isinstance(value, dict) or set(value) != _PARTITION_FIELDS:
        raise ValueError("representation evaluation partition fields differ from schema")
    if value["schema"] != REPRESENTATION_EVALUATION_PARTITION_SCHEMA:
        raise ValueError("representation evaluation partition schema changed")
    validated = tuple(validate_representation_evaluation_sample(dict(sample)) for sample in samples)
    expected = _representation_partition_value(validated, partition=partition)
    if value != expected:
        raise ValueError("representation evaluation partition was not recomputed")
    return value


def _representation_partition_value(
    samples: Sequence[dict[str, Any]],
    *,
    partition: str,
) -> dict[str, object]:
    """Internal non-recursive implementation for partition validation."""

    if not samples or any(sample["partition"] != partition for sample in samples):
        raise ValueError("representation evaluation samples differ from the partition")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_task[str(sample["task_key"])].append(sample)
    task_metrics = {
        task_key: _task_level_metrics(by_task[task_key]) for task_key in sorted(by_task)
    }

    def task_values(name: str) -> list[float]:
        return [
            _finite_float(item[name], name=f"representation task metric {name}")
            for item in task_metrics.values()
            if item[name] is not None
        ]

    token_tasks = task_values("fractional_weighted_auc")
    control_task = task_values("shuffled_task_auc_degradation")
    control_target = task_values("shuffled_target_auc_degradation")
    if len(control_task) != len(control_target):
        raise ValueError("representation control task coverage differs")
    row_tasks = [item for item in task_metrics.values() if item["worst_target_rank"] is not None]
    ownership_tasks = task_values("target_ownership_soft_iou")
    return {
        "schema": REPRESENTATION_EVALUATION_PARTITION_SCHEMA,
        "partition": partition,
        "sample_count": len(samples),
        "task_count": len(task_metrics),
        "token_eligible_sample_count": sum(
            _nonnegative_int(
                item["token_eligible_sample_count"],
                name="representation task token eligible sample count",
            )
            for item in task_metrics.values()
        ),
        "token_eligible_task_count": len(token_tasks),
        "control_eligible_sample_count": sum(
            _nonnegative_int(
                item["control_eligible_sample_count"],
                name="representation task control eligible sample count",
            )
            for item in task_metrics.values()
        ),
        "control_eligible_task_count": len(control_task),
        "mean_task_fractional_weighted_auc": _optional_mean(token_tasks),
        "mean_task_target_background_logit_margin": _optional_mean(
            task_values("target_background_logit_margin")
        ),
        "mean_task_shuffled_task_fractional_weighted_auc": _optional_mean(
            task_values("shuffled_task_fractional_weighted_auc")
        ),
        "mean_task_shuffled_target_fractional_weighted_auc": _optional_mean(
            task_values("shuffled_target_fractional_weighted_auc")
        ),
        "mean_task_shuffled_task_auc_degradation": _optional_mean(control_task),
        "mean_task_shuffled_target_auc_degradation": _optional_mean(control_target),
        "row_eligible_sample_count": sum(
            _nonnegative_int(
                item["row_eligible_sample_count"],
                name="representation task row eligible sample count",
            )
            for item in task_metrics.values()
        ),
        "row_eligible_task_count": len(row_tasks),
        "rank_one_task_count": sum(item["worst_target_rank"] == 1 for item in row_tasks),
        "rank_one_task_fraction": (
            None
            if not row_tasks
            else sum(item["worst_target_rank"] == 1 for item in row_tasks) / len(row_tasks)
        ),
        "mean_task_hardest_negative_logit_margin": _optional_mean(
            task_values("hardest_negative_logit_margin")
        ),
        "ownership_eligible_sample_count": sum(
            _nonnegative_int(
                item["ownership_eligible_sample_count"],
                name="representation task ownership eligible sample count",
            )
            for item in task_metrics.values()
        ),
        "ownership_eligible_task_count": len(ownership_tasks),
        "mean_task_target_ownership_soft_iou": _optional_mean(ownership_tasks),
        "mean_task_target_mass_concentration": _optional_mean(
            task_values("target_mass_concentration")
        ),
        "mean_task_macro_ownership_soft_iou": _mean(
            task_values("macro_ownership_soft_iou"),
            name="representation partition macro ownership",
        ),
        "mean_official_action_loss": _mean(
            task_values("official_action_loss"),
            name="representation partition action loss",
        ),
        "maximum_peak_cuda_reserved_bytes": max(
            _nonnegative_int(
                sample["peak_cuda_reserved_bytes"],
                name="representation sample peak CUDA reserved bytes",
            )
            for sample in samples
        ),
        "mean_factual_forward_seconds": _mean(
            [float(sample["forward_seconds"]["factual"]) for sample in samples],
            name="representation factual forward time",
        ),
        "mean_shuffled_task_forward_seconds": _mean(
            [float(sample["forward_seconds"]["shuffled_task"]) for sample in samples],
            name="representation shuffled-task forward time",
        ),
        "task_metrics": task_metrics,
    }


def build_representation_evaluation_snapshot(
    *,
    checkpoint_global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    representation_split_sha256: str,
    representation_evaluation_plan: RepresentationEvaluationPlan,
    representation_frozen_action_state_sha256: str,
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build one immutable checkpoint snapshot over both evaluation banks."""

    value = {
        "schema": REPRESENTATION_EVALUATION_SNAPSHOT_SCHEMA,
        "status": "PASS",
        "checkpoint_global_step": checkpoint_global_step,
        "implementation_sha256": implementation_sha256,
        "model_family_sha256": model_family_sha256,
        "representation_split_sha256": representation_split_sha256,
        "representation_evaluation_plan_sha256": (representation_evaluation_plan.artifact_sha256),
        "representation_frozen_action_state_sha256": (representation_frozen_action_state_sha256),
        "samples": [dict(sample) for sample in samples],
        "partition_summaries": {},
    }
    validated_samples = tuple(
        validate_representation_evaluation_sample(dict(sample)) for sample in samples
    )
    value["partition_summaries"] = {
        partition: _representation_partition_value(
            tuple(sample for sample in validated_samples if sample["partition"] == partition),
            partition=partition,
        )
        for partition in REPRESENTATION_EVALUATION_PARTITIONS
    }
    payload = dict(value)
    value["artifact_sha256"] = _canonical_sha256(payload)
    return validate_representation_evaluation_snapshot(
        value,
        plan=representation_evaluation_plan,
    )


def validate_representation_evaluation_snapshot(
    value: object,
    *,
    plan: RepresentationEvaluationPlan,
) -> dict[str, Any]:
    """Recompute a complete checkpoint snapshot and reject missing bank rows."""

    if not isinstance(value, dict) or set(value) != _SNAPSHOT_FIELDS:
        raise ValueError("representation evaluation snapshot fields differ from schema")
    if value["schema"] != REPRESENTATION_EVALUATION_SNAPSHOT_SCHEMA or value["status"] != "PASS":
        raise ValueError("representation evaluation snapshot status or schema changed")
    if not isinstance(plan, RepresentationEvaluationPlan):
        raise TypeError("representation evaluation snapshot requires its frozen plan")
    checkpoint = _nonnegative_int(
        value["checkpoint_global_step"],
        name="representation evaluation checkpoint step",
    )
    for name in (
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "representation_frozen_action_state_sha256",
    ):
        _sha256(value[name], name=f"representation evaluation snapshot {name}")
    if (
        value["representation_split_sha256"] != plan.representation_split_sha256
        or value["representation_evaluation_plan_sha256"] != plan.artifact_sha256
    ):
        raise ValueError("representation evaluation snapshot targets another plan")
    raw_samples = value["samples"]
    if not isinstance(raw_samples, list) or len(raw_samples) != len(plan.items):
        raise ValueError("representation evaluation snapshot sample coverage changed")
    samples = tuple(
        validate_representation_evaluation_sample(sample, expected_item=item)
        for sample, item in zip(raw_samples, plan.items, strict=True)
    )
    if any(sample["checkpoint_global_step"] != checkpoint for sample in samples):
        raise ValueError("representation evaluation snapshot mixes checkpoints")
    summaries = value["partition_summaries"]
    if not isinstance(summaries, dict) or set(summaries) != set(
        REPRESENTATION_EVALUATION_PARTITIONS
    ):
        raise ValueError("representation evaluation snapshot partitions changed")
    expected_summaries = {
        partition: _representation_partition_value(
            tuple(sample for sample in samples if sample["partition"] == partition),
            partition=partition,
        )
        for partition in REPRESENTATION_EVALUATION_PARTITIONS
    }
    if summaries != expected_summaries:
        raise ValueError("representation evaluation snapshot summaries were not recomputed")
    payload = {name: value[name] for name in _SNAPSHOT_FIELDS if name != "artifact_sha256"}
    expected_sha256 = _sha256(
        value["artifact_sha256"],
        name="representation evaluation snapshot artifact",
    )
    if _canonical_sha256(payload) != expected_sha256:
        raise ValueError("representation evaluation snapshot artifact SHA-256 changed")
    return value


def validate_representation_evaluation_visual_files(
    value: object,
    *,
    plan: RepresentationEvaluationPlan,
    output_root: str | Path,
) -> tuple[Path, ...]:
    """Reopen every declared visual and verify its bytes below one fixed root."""

    snapshot = validate_representation_evaluation_snapshot(value, plan=plan)
    root = Path(output_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("representation evaluation visual root is not a directory")
    observed: list[Path] = []
    for sample in snapshot["samples"]:
        visual = sample["visual_artifact"]
        relative = PurePosixPath(visual["path"])
        path = root.joinpath(*relative.parts)
        if path.is_symlink():
            raise ValueError("representation evaluation visual must not be a symlink")
        try:
            with path.open("rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("representation evaluation visual is not a regular file")
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except FileNotFoundError as error:
            raise ValueError("representation evaluation visual is absent") from error
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("representation evaluation visual escaped its output root")
        if metadata.st_size != visual["bytes"] or digest.hexdigest() != visual["sha256"]:
            raise ValueError("representation evaluation visual bytes differ from provenance")
        observed.append(resolved)
    if len(set(observed)) != len(observed):
        raise ValueError("representation evaluation visual path was reused")
    return tuple(observed)
