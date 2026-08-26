"""Source-disjoint fixed-X evaluation banks for the shared LingBot host.

The plan chooses two different, physically true instructions per immutable
observation. Selection is offline and model-independent. Runtime semantics,
identity labels, simulator proofs, and audit metadata remain loss-side.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, cast

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.fixed_observation import (
    FIXED_OBSERVATION_MAXIMUM_TORCH_SEED,
    FixedObservationAudit,
    FixedObservationGroup,
    FixedObservationPairPlan,
    FixedObservationVariant,
)
from picf_next.lingbot_native.representation_evaluation import (
    validate_representation_ownership_row,
    validate_representation_ownership_summary,
    validate_representation_token_evidence,
)
from picf_next.lingbot_native.task_diagnostics import validate_task_row_diagnostic
from picf_next.lingbot_native.visual_audit import NATIVE_VISUAL_AUDIT_SCHEMA

FIXED_OBSERVATION_EVALUATION_PLAN_SCHEMA = "picf-next.lingbot-fixed-observation-evaluation-plan.v2"
FIXED_OBSERVATION_EVALUATION_ALGORITHM = "truth-audited-source-disjoint-balanced-pairs.v2"
FIXED_OBSERVATION_EVALUATION_PARTITIONS = ("validation", "heldout")
FIXED_OBSERVATION_EVALUATION_WORLD_SIZE = 2
FIXED_OBSERVATION_EVALUATION_SAMPLE_SCHEMA = (
    "picf-next.lingbot-fixed-observation-evaluation-sample.v1"
)
FIXED_OBSERVATION_EVALUATION_SNAPSHOT_SCHEMA = (
    "picf-next.lingbot-fixed-observation-evaluation-snapshot.v3"
)
FIXED_OBSERVATION_FORWARD_EQUIVALENCE_SCHEMA = (
    "picf-next.lingbot-fixed-observation-forward-equivalence.v2"
)
FIXED_OBSERVATION_MASS_STRATA = ("lower_third", "middle_third", "upper_third")

_VARIANT_RESULT_FIELDS = frozenset(
    {
        "alternate_target_token_evidence",
        "forward_seconds",
        "instruction_sha256",
        "own_target_token_evidence",
        "ownership_rows",
        "ownership_summary",
        "relation_sha256",
        "target_sha256",
        "task_row_diagnostic",
        "variant",
        "visual_artifact",
    }
)
_PAIR_METRIC_FIELDS = frozenset(
    {
        "dense_bidirectional_positive",
        "dense_mean_diagonal_advantage",
        "dense_variant_diagonal_advantages",
        "fractional_auc_bidirectional_positive",
        "fractional_auc_mean_diagonal_advantage",
        "fractional_auc_variant_diagonal_advantages",
        "relation_output_changed",
        "row_bidirectional_positive",
        "row_mean_diagonal_advantage",
        "row_variant_diagonal_advantages",
    }
)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("fixed-X evaluation value is not canonical finite JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _hash_order(*parts: object) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"picf-next.fixed-observation-evaluation-order.v1\0")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _atomic_write(path: Path, value: object) -> None:
    write_bytes_durable_exclusive(path, _canonical_bytes(value) + b"\n")


@dataclass(frozen=True, slots=True)
class FixedObservationEvaluationItem:
    """One source observation and two true prompt-conditioned targets."""

    partition: str
    ordinal: int
    rank: int
    group: FixedObservationGroup
    variants: tuple[FixedObservationVariant, FixedObservationVariant]
    replay_seed: int

    def __post_init__(self) -> None:
        if self.partition not in FIXED_OBSERVATION_EVALUATION_PARTITIONS:
            raise ValueError("fixed-X evaluation item partition is unsupported")
        _nonnegative_int(self.ordinal, name="fixed-X evaluation ordinal")
        _nonnegative_int(self.rank, name="fixed-X evaluation rank")
        if self.rank >= FIXED_OBSERVATION_EVALUATION_WORLD_SIZE:
            raise ValueError("fixed-X evaluation rank is outside its world")
        if not isinstance(self.group, FixedObservationGroup):
            raise TypeError("fixed-X evaluation item requires one audited group")
        if (
            not isinstance(self.variants, tuple)
            or len(self.variants) != 2
            or any(not isinstance(item, FixedObservationVariant) for item in self.variants)
            or any(item not in self.group.variants for item in self.variants)
            or self.variants[0].task_key == self.variants[1].task_key
            or self.variants[0].target_identity_key == self.variants[1].target_identity_key
        ):
            raise ValueError("fixed-X evaluation item requires two distinct audited variants")
        _nonnegative_int(self.replay_seed, name="fixed-X evaluation replay seed")
        if self.replay_seed > FIXED_OBSERVATION_MAXIMUM_TORCH_SEED:
            raise ValueError("fixed-X evaluation replay seed exceeds PyTorch's signed range")

    def as_dict(self) -> dict[str, object]:
        return {
            "group": self.group.as_dict(),
            "ordinal": self.ordinal,
            "partition": self.partition,
            "rank": self.rank,
            "replay_seed": self.replay_seed,
            "variants": [item.as_dict() for item in self.variants],
        }

    @classmethod
    def from_dict(cls, value: object) -> FixedObservationEvaluationItem:
        expected = {
            "group",
            "ordinal",
            "partition",
            "rank",
            "replay_seed",
            "variants",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("fixed-X evaluation item fields differ from schema")
        variants = value["variants"]
        if not isinstance(variants, list) or len(variants) != 2:
            raise ValueError("fixed-X evaluation variants are malformed")
        parsed_variants = tuple(FixedObservationVariant.from_dict(item) for item in variants)
        return cls(
            partition=_text(value["partition"], name="fixed-X evaluation partition"),
            ordinal=_nonnegative_int(
                value["ordinal"],
                name="fixed-X evaluation ordinal",
            ),
            rank=_nonnegative_int(value["rank"], name="fixed-X evaluation rank"),
            group=FixedObservationGroup.from_dict(value["group"]),
            variants=(parsed_variants[0], parsed_variants[1]),
            replay_seed=_nonnegative_int(
                value["replay_seed"],
                name="fixed-X evaluation replay seed",
            ),
        )


@dataclass(frozen=True, slots=True)
class FixedObservationEvaluationPlan:
    """Content-addressed validation and held-out same-observation bank."""

    dataset_tree_sha256: str
    comparison_id: str
    stream_plan_sha256: str
    representation_split_file_sha256: str
    representation_split_artifact_sha256: str
    training_pair_plan_sha256: str
    training_audit_file_sha256: str
    training_audit_artifact_sha256: str
    validation_audit_file_sha256: str
    validation_audit_artifact_sha256: str
    heldout_audit_file_sha256: str
    heldout_audit_artifact_sha256: str
    items: tuple[FixedObservationEvaluationItem, ...]
    world_size: int = FIXED_OBSERVATION_EVALUATION_WORLD_SIZE

    def __post_init__(self) -> None:
        for name in (
            "dataset_tree_sha256",
            "stream_plan_sha256",
            "representation_split_file_sha256",
            "representation_split_artifact_sha256",
            "training_pair_plan_sha256",
            "training_audit_file_sha256",
            "training_audit_artifact_sha256",
            "validation_audit_file_sha256",
            "validation_audit_artifact_sha256",
            "heldout_audit_file_sha256",
            "heldout_audit_artifact_sha256",
        ):
            _sha256(getattr(self, name), name=f"fixed-X evaluation {name}")
        _text(self.comparison_id, name="fixed-X evaluation comparison ID")
        if self.world_size != FIXED_OBSERVATION_EVALUATION_WORLD_SIZE:
            raise ValueError("fixed-X evaluation world size changed")
        if (
            not isinstance(self.items, tuple)
            or not self.items
            or any(not isinstance(item, FixedObservationEvaluationItem) for item in self.items)
        ):
            raise ValueError("fixed-X evaluation plan requires typed items")
        expected = tuple(
            sorted(
                self.items,
                key=lambda item: (
                    FIXED_OBSERVATION_EVALUATION_PARTITIONS.index(item.partition),
                    item.ordinal,
                ),
            )
        )
        if self.items != expected:
            raise ValueError("fixed-X evaluation items are not canonically ordered")
        source_states: set[str] = set()
        source_samples: set[str] = set()
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS:
            values = tuple(item for item in self.items if item.partition == partition)
            if not values or tuple(item.ordinal for item in values) != tuple(range(len(values))):
                raise ValueError(f"fixed-X evaluation {partition} ordinals changed")
            if any(item.rank != item.ordinal % self.world_size for item in values):
                raise ValueError(f"fixed-X evaluation {partition} sharding changed")
            rank_counts = Counter(item.rank for item in values)
            if max(rank_counts.values()) - min(rank_counts.values()) > 1:
                raise ValueError(f"fixed-X evaluation {partition} rank load changed")
            tasks = {variant.task_key for item in values for variant in item.variants}
            targets = {variant.target_identity_key for item in values for variant in item.variants}
            if len(tasks) < 2 or len(targets) < 2:
                raise ValueError(f"fixed-X evaluation {partition} lost semantic breadth")
            for item in values:
                if (
                    item.group.source_state_sha256 in source_states
                    or item.group.stateful_sample_key in source_samples
                ):
                    raise ValueError("fixed-X evaluation repeated a source observation")
                source_states.add(item.group.source_state_sha256)
                source_samples.add(item.group.stateful_sample_key)
        combined_rank_counts = tuple(
            sum(item.rank == rank for item in self.items) for rank in range(self.world_size)
        )
        if any(count == 0 for count in combined_rank_counts) or len(set(combined_rank_counts)) != 1:
            raise ValueError("fixed-X evaluation combined rank load must be equal")

    @property
    def content(self) -> dict[str, object]:
        return {
            "algorithm": FIXED_OBSERVATION_EVALUATION_ALGORITHM,
            "comparison_id": self.comparison_id,
            "dataset_tree_sha256": self.dataset_tree_sha256,
            "heldout_audit_artifact_sha256": self.heldout_audit_artifact_sha256,
            "heldout_audit_file_sha256": self.heldout_audit_file_sha256,
            "items": [item.as_dict() for item in self.items],
            "representation_split_artifact_sha256": (self.representation_split_artifact_sha256),
            "representation_split_file_sha256": (self.representation_split_file_sha256),
            "schema": FIXED_OBSERVATION_EVALUATION_PLAN_SCHEMA,
            "stream_plan_sha256": self.stream_plan_sha256,
            "training_audit_artifact_sha256": (self.training_audit_artifact_sha256),
            "training_audit_file_sha256": self.training_audit_file_sha256,
            "training_pair_plan_sha256": self.training_pair_plan_sha256,
            "validation_audit_artifact_sha256": (self.validation_audit_artifact_sha256),
            "validation_audit_file_sha256": self.validation_audit_file_sha256,
            "world_size": self.world_size,
        }

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self.content)

    def items_for(
        self,
        partition: str,
        rank: int,
    ) -> tuple[FixedObservationEvaluationItem, ...]:
        if partition not in FIXED_OBSERVATION_EVALUATION_PARTITIONS:
            raise ValueError("fixed-X evaluation partition is unsupported")
        _nonnegative_int(rank, name="fixed-X evaluation rank")
        if rank >= self.world_size:
            raise ValueError("fixed-X evaluation rank is outside its world")
        return tuple(
            item for item in self.items if item.partition == partition and item.rank == rank
        )

    @property
    def task_histogram(self) -> dict[str, dict[str, int]]:
        return {
            partition: dict(
                sorted(
                    Counter(
                        variant.task_key
                        for item in self.items
                        if item.partition == partition
                        for variant in item.variants
                    ).items()
                )
            )
            for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
        }

    @property
    def target_histogram(self) -> dict[str, dict[str, int]]:
        return {
            partition: dict(
                sorted(
                    Counter(
                        variant.target_identity_key
                        for item in self.items
                        if item.partition == partition
                        for variant in item.variants
                    ).items()
                )
            )
            for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
        }

    def write(self, path: str | Path) -> None:
        _atomic_write(
            Path(path),
            {**self.content, "artifact_sha256": self.artifact_sha256},
        )

    @classmethod
    def from_dict(cls, value: object) -> FixedObservationEvaluationPlan:
        expected = {
            *FixedObservationEvaluationPlan.__dataclass_fields__.keys(),
            "algorithm",
            "artifact_sha256",
            "schema",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("fixed-X evaluation plan fields differ from schema")
        if (
            value["schema"] != FIXED_OBSERVATION_EVALUATION_PLAN_SCHEMA
            or value["algorithm"] != FIXED_OBSERVATION_EVALUATION_ALGORITHM
        ):
            raise ValueError("fixed-X evaluation plan schema or algorithm changed")
        raw_items = value["items"]
        if not isinstance(raw_items, list):
            raise ValueError("fixed-X evaluation items are malformed")
        plan = cls(
            dataset_tree_sha256=_sha256(
                value["dataset_tree_sha256"],
                name="fixed-X evaluation dataset tree",
            ),
            comparison_id=_text(
                value["comparison_id"],
                name="fixed-X evaluation comparison ID",
            ),
            stream_plan_sha256=_sha256(
                value["stream_plan_sha256"],
                name="fixed-X evaluation stream plan",
            ),
            representation_split_file_sha256=_sha256(
                value["representation_split_file_sha256"],
                name="fixed-X evaluation split file",
            ),
            representation_split_artifact_sha256=_sha256(
                value["representation_split_artifact_sha256"],
                name="fixed-X evaluation split artifact",
            ),
            training_pair_plan_sha256=_sha256(
                value["training_pair_plan_sha256"],
                name="fixed-X evaluation training pair plan",
            ),
            training_audit_file_sha256=_sha256(
                value["training_audit_file_sha256"],
                name="fixed-X evaluation training audit file",
            ),
            training_audit_artifact_sha256=_sha256(
                value["training_audit_artifact_sha256"],
                name="fixed-X evaluation training audit artifact",
            ),
            validation_audit_file_sha256=_sha256(
                value["validation_audit_file_sha256"],
                name="fixed-X evaluation validation audit file",
            ),
            validation_audit_artifact_sha256=_sha256(
                value["validation_audit_artifact_sha256"],
                name="fixed-X evaluation validation audit artifact",
            ),
            heldout_audit_file_sha256=_sha256(
                value["heldout_audit_file_sha256"],
                name="fixed-X evaluation heldout audit file",
            ),
            heldout_audit_artifact_sha256=_sha256(
                value["heldout_audit_artifact_sha256"],
                name="fixed-X evaluation heldout audit artifact",
            ),
            items=tuple(FixedObservationEvaluationItem.from_dict(item) for item in raw_items),
            world_size=_nonnegative_int(
                value["world_size"],
                name="fixed-X evaluation world size",
            ),
        )
        expected_artifact = _sha256(
            value["artifact_sha256"],
            name="fixed-X evaluation plan artifact",
        )
        if plan.artifact_sha256 != expected_artifact:
            raise ValueError("fixed-X evaluation plan artifact SHA-256 changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> FixedObservationEvaluationPlan:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid fixed-X evaluation plan: {source}") from error
        return cls.from_dict(value)


def _pair_score(
    variants: tuple[FixedObservationVariant, FixedObservationVariant],
    *,
    task_counts: Counter[str],
    target_counts: Counter[str],
    selected_tasks: set[str],
    selected_targets: set[str],
    tie_breaker: bytes,
) -> tuple[object, ...]:
    tasks = tuple(item.task_key for item in variants)
    targets = tuple(item.target_identity_key for item in variants)
    projected_tasks = Counter(task_counts)
    projected_targets = Counter(target_counts)
    projected_tasks.update(tasks)
    projected_targets.update(targets)
    new_facets = len(set(tasks) - selected_tasks) + len(set(targets) - selected_targets)
    masses = tuple(math.log(max(item.target_mass, 1e-12)) for item in variants)
    return (
        -new_facets,
        max(projected_targets.values()),
        sum(value * value for value in projected_targets.values()),
        max(projected_tasks.values()),
        sum(value * value for value in projected_tasks.values()),
        abs(masses[0] - masses[1]),
        tie_breaker,
    )


def _partition_items(
    audit: FixedObservationAudit,
) -> tuple[FixedObservationEvaluationItem, ...]:
    if audit.partition not in FIXED_OBSERVATION_EVALUATION_PARTITIONS:
        raise ValueError("fixed-X evaluation requires validation or heldout audit")
    task_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    selected_tasks: set[str] = set()
    selected_targets: set[str] = set()
    items = []
    for ordinal, group in enumerate(audit.groups):
        candidates = []
        for variants in combinations(group.variants, 2):
            tie = _hash_order(
                audit.report_artifact_sha256,
                audit.partition,
                group.source_state_sha256,
                variants[0].task_key,
                variants[1].task_key,
            )
            candidates.append(
                (
                    _pair_score(
                        variants,
                        task_counts=task_counts,
                        target_counts=target_counts,
                        selected_tasks=selected_tasks,
                        selected_targets=selected_targets,
                        tie_breaker=tie,
                    ),
                    variants,
                )
            )
        if not candidates:
            raise RuntimeError("fixed-X evaluation group has no true task pair")
        _score, variants = min(candidates, key=lambda item: item[0])
        replay_seed = (
            int.from_bytes(
                _hash_order(
                    audit.report_artifact_sha256,
                    audit.partition,
                    ordinal,
                    group.source_state_sha256,
                    "replay",
                )[:8],
                "big",
            )
            & FIXED_OBSERVATION_MAXIMUM_TORCH_SEED
        )
        items.append(
            FixedObservationEvaluationItem(
                partition=audit.partition,
                ordinal=ordinal,
                rank=ordinal % FIXED_OBSERVATION_EVALUATION_WORLD_SIZE,
                group=group,
                variants=variants,
                replay_seed=replay_seed,
            )
        )
        task_counts.update(item.task_key for item in variants)
        target_counts.update(item.target_identity_key for item in variants)
        selected_tasks.update(item.task_key for item in variants)
        selected_targets.update(item.target_identity_key for item in variants)
    if selected_tasks != set(audit.task_keys):
        raise ValueError(f"fixed-X {audit.partition} bank failed complete task coverage")
    if selected_targets != set(audit.target_identity_keys):
        raise ValueError(f"fixed-X {audit.partition} bank failed complete target coverage")
    return tuple(items)


def build_fixed_observation_evaluation_plan(
    *,
    training_audit: FixedObservationAudit,
    validation_audit: FixedObservationAudit,
    heldout_audit: FixedObservationAudit,
    training_pair_plan: FixedObservationPairPlan,
) -> FixedObservationEvaluationPlan:
    """Build validation/heldout true-prompt pairs with source isolation."""

    audits = (training_audit, validation_audit, heldout_audit)
    if tuple(item.partition for item in audits) != (
        "training",
        "validation",
        "heldout",
    ):
        raise ValueError("fixed-X evaluation audits are out of partition order")
    if not isinstance(training_pair_plan, FixedObservationPairPlan):
        raise TypeError("fixed-X evaluation requires its frozen training-pair plan")
    identities = {
        (
            item.dataset_manifest_file_sha256,
            item.dataset_tree_sha256,
            item.comparison_id,
            item.stream_plan_sha256,
            item.representation_split_file_sha256,
            item.representation_split_artifact_sha256,
            item.training_projection_contract_sha256,
            item.training_projection_payload_sha256,
        )
        for item in audits
    }
    if len(identities) != 1:
        raise ValueError("fixed-X partition audits belong to different source splits")
    training_identity = (
        training_pair_plan.audit_report_file_sha256,
        training_pair_plan.audit_artifact_sha256,
        training_pair_plan.stream_plan_sha256,
        training_pair_plan.representation_split_file_sha256,
        training_pair_plan.representation_split_artifact_sha256,
    )
    audit_identity = (
        training_audit.report_file_sha256,
        training_audit.report_artifact_sha256,
        training_audit.stream_plan_sha256,
        training_audit.representation_split_file_sha256,
        training_audit.representation_split_artifact_sha256,
    )
    if training_identity != audit_identity:
        raise ValueError("fixed-X evaluation and training-pair audits differ")
    source_episodes = [{group.source_episode_index for group in audit.groups} for audit in audits]
    for left, right in combinations(source_episodes, 2):
        if left.intersection(right):
            raise ValueError("fixed-X partition audits share source episodes")
    source_states = [{group.source_state_sha256 for group in audit.groups} for audit in audits]
    for left, right in combinations(source_states, 2):
        if left.intersection(right):
            raise ValueError("fixed-X partition audits share source observations")

    (
        _dataset_manifest_file_sha256,
        dataset_tree_sha256,
        comparison_id,
        stream_plan_sha256,
        representation_split_file_sha256,
        representation_split_artifact_sha256,
        _training_projection_contract_sha256,
        _training_projection_payload_sha256,
    ) = next(iter(identities))
    return FixedObservationEvaluationPlan(
        dataset_tree_sha256=dataset_tree_sha256,
        comparison_id=comparison_id,
        stream_plan_sha256=stream_plan_sha256,
        representation_split_file_sha256=representation_split_file_sha256,
        representation_split_artifact_sha256=representation_split_artifact_sha256,
        training_pair_plan_sha256=training_pair_plan.artifact_sha256,
        training_audit_file_sha256=training_audit.report_file_sha256,
        training_audit_artifact_sha256=training_audit.report_artifact_sha256,
        validation_audit_file_sha256=validation_audit.report_file_sha256,
        validation_audit_artifact_sha256=validation_audit.report_artifact_sha256,
        heldout_audit_file_sha256=heldout_audit.report_file_sha256,
        heldout_audit_artifact_sha256=heldout_audit.report_artifact_sha256,
        items=(
            *_partition_items(validation_audit),
            *_partition_items(heldout_audit),
        ),
    )


def fixed_observation_evaluation_mass_strata(
    plan: FixedObservationEvaluationPlan,
) -> dict[tuple[str, int], str]:
    """Freeze rank-based mass thirds without introducing an acceptance threshold."""

    if not isinstance(plan, FixedObservationEvaluationPlan):
        raise TypeError("fixed-X mass strata require the immutable evaluation plan")
    result: dict[tuple[str, int], str] = {}
    for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS:
        items = sorted(
            (item for item in plan.items if item.partition == partition),
            key=lambda item: (
                min(variant.target_mass for variant in item.variants),
                item.group.source_state_sha256,
                item.ordinal,
            ),
        )
        for position, item in enumerate(items):
            stratum_index = min(
                len(FIXED_OBSERVATION_MASS_STRATA) - 1,
                position * len(FIXED_OBSERVATION_MASS_STRATA) // len(items),
            )
            result[(partition, item.ordinal)] = FIXED_OBSERVATION_MASS_STRATA[stratum_index]
    if len(result) != len(plan.items):
        raise RuntimeError("fixed-X mass stratification omitted an evaluation item")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_mean(values: Sequence[float]) -> float | None:
    return None if not values else math.fsum(values) / len(values)


def _token_metric(
    evidence: Mapping[str, object],
    name: str,
) -> float | None:
    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("fixed-X token metrics are malformed")
    value = metrics.get(name)
    return None if value is None else _finite_float(value, name=f"fixed-X token {name}")


def _validate_visual_artifact(
    value: object,
    *,
    checkpoint_global_step: int,
    item: FixedObservationEvaluationItem,
    variant: FixedObservationVariant,
) -> dict[str, Any]:
    required = {
        "bytes",
        "global_step",
        "input_weight_global_step",
        "loss_only_labels_visible_to_model",
        "path",
        "rank",
        "sample_key",
        "schema",
        "sha256",
        "task",
        "weight_boundary",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("fixed-X visual artifact omits provenance")
    relative_value = value["path"]
    relative = PurePosixPath(relative_value) if isinstance(relative_value, str) else None
    if (
        value["schema"] != NATIVE_VISUAL_AUDIT_SCHEMA
        or value["global_step"] != checkpoint_global_step
        or value["input_weight_global_step"] != checkpoint_global_step
        or value["weight_boundary"] != "fixed_observation_checkpoint_evaluation"
        or value["rank"] != item.rank
        or value["sample_key"] != item.group.stateful_sample_key
        or value["task"] != variant.instruction
        or value["loss_only_labels_visible_to_model"] is not False
        or relative is None
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("fixed-X visual artifact provenance changed")
    _sha256(value["sha256"], name="fixed-X visual SHA-256")
    _positive_int(value["bytes"], name="fixed-X visual byte count")
    return value


def _validate_variant_result(
    value: object,
    *,
    checkpoint_global_step: int,
    item: FixedObservationEvaluationItem,
    expected_variant: FixedObservationVariant,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _VARIANT_RESULT_FIELDS:
        raise ValueError("fixed-X variant result fields differ from schema")
    variant = FixedObservationVariant.from_dict(value["variant"])
    if variant != expected_variant:
        raise ValueError("fixed-X variant result differs from its frozen plan")
    if (
        _sha256(value["instruction_sha256"], name="fixed-X instruction")
        != variant.instruction_sha256
    ):
        raise ValueError("fixed-X runtime instruction differs from its frozen variant")
    own = validate_representation_token_evidence(dict(value["own_target_token_evidence"]))
    alternate = validate_representation_token_evidence(
        dict(value["alternate_target_token_evidence"])
    )
    if own["logits"] != alternate["logits"]:
        raise ValueError("fixed-X alternate target changed the completed forward logits")
    if (
        math.fsum(float(item) for item in own["target_mass"]) <= 0
        or math.fsum(float(item) for item in alternate["target_mass"]) <= 0
    ):
        raise ValueError("fixed-X runtime target has no measured mass")
    task_row = validate_task_row_diagnostic(dict(value["task_row_diagnostic"]))
    if task_row["target_identity_keys"] != [variant.target_identity_key]:
        raise ValueError("fixed-X task-row target differs from its prompt variant")
    ownership_rows = tuple(
        validate_representation_ownership_row(dict(row)) for row in value["ownership_rows"]
    )
    if not ownership_rows:
        raise ValueError("fixed-X variant has no visible ownership rows")
    validate_representation_ownership_summary(
        dict(value["ownership_summary"]),
        rows=ownership_rows,
    )
    _sha256(value["relation_sha256"], name="fixed-X relation")
    _sha256(value["target_sha256"], name="fixed-X target")
    if _finite_float(value["forward_seconds"], name="fixed-X forward time") <= 0:
        raise ValueError("fixed-X forward time must be positive")
    _validate_visual_artifact(
        value["visual_artifact"],
        checkpoint_global_step=checkpoint_global_step,
        item=item,
        variant=variant,
    )
    return value


def _row_identity_logit(
    diagnostic: Mapping[str, object],
    identity_key: str,
) -> float | None:
    validated = validate_task_row_diagnostic(dict(diagnostic))
    identities = validated["identity_keys"]
    if identity_key not in identities:
        raise ValueError("fixed-X prompt-switch target identity is absent from the scene")
    track_index = identities.index(identity_key)
    rows = [
        row
        for row, assigned_track in enumerate(validated["row_to_track"])
        if assigned_track == track_index and validated["row_task_valid"][row]
    ]
    if not rows:
        return None
    if len(rows) != 1:
        raise RuntimeError("fixed-X assignment materialized one identity more than once")
    return _finite_float(
        validated["task_logits"][rows[0]],
        name="fixed-X task-row identity logit",
    )


def _paired_advantages(
    first_own: float | None,
    first_alternate: float | None,
    second_own: float | None,
    second_alternate: float | None,
) -> tuple[list[float | None], float | None, bool | None]:
    advantages = [
        None if first_own is None or first_alternate is None else first_own - first_alternate,
        None if second_own is None or second_alternate is None else second_own - second_alternate,
    ]
    complete = tuple(item for item in advantages if item is not None)
    return (
        advantages,
        None if len(complete) != 2 else math.fsum(complete) / 2,
        None if len(complete) != 2 else all(item > 0 for item in complete),
    )


def fixed_observation_prompt_switch_metrics(
    variant_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Measure diagonal prompt-target preference with no fitted scorer."""

    if len(variant_results) != 2:
        raise ValueError("fixed-X prompt-switch metrics require two variant results")
    first, second = variant_results
    first_own = first["own_target_token_evidence"]
    first_alternate = first["alternate_target_token_evidence"]
    second_own = second["own_target_token_evidence"]
    second_alternate = second["alternate_target_token_evidence"]
    if not all(
        isinstance(item, Mapping)
        for item in (first_own, first_alternate, second_own, second_alternate)
    ):
        raise ValueError("fixed-X prompt-switch token evidence is malformed")
    first_own = cast(Mapping[str, object], first_own)
    first_alternate = cast(Mapping[str, object], first_alternate)
    second_own = cast(Mapping[str, object], second_own)
    second_alternate = cast(Mapping[str, object], second_alternate)

    dense_values, dense_mean, dense_positive = _paired_advantages(
        _token_metric(first_own, "target_background_logit_margin"),
        _token_metric(first_alternate, "target_background_logit_margin"),
        _token_metric(second_own, "target_background_logit_margin"),
        _token_metric(second_alternate, "target_background_logit_margin"),
    )
    auc_values, auc_mean, auc_positive = _paired_advantages(
        _token_metric(first_own, "fractional_weighted_auc"),
        _token_metric(first_alternate, "fractional_weighted_auc"),
        _token_metric(second_own, "fractional_weighted_auc"),
        _token_metric(second_alternate, "fractional_weighted_auc"),
    )
    first_variant = FixedObservationVariant.from_dict(first["variant"])
    second_variant = FixedObservationVariant.from_dict(second["variant"])
    first_row = first["task_row_diagnostic"]
    second_row = second["task_row_diagnostic"]
    if not isinstance(first_row, Mapping) or not isinstance(second_row, Mapping):
        raise ValueError("fixed-X prompt-switch row evidence is malformed")
    row_values, row_mean, row_positive = _paired_advantages(
        _row_identity_logit(first_row, first_variant.target_identity_key),
        _row_identity_logit(first_row, second_variant.target_identity_key),
        _row_identity_logit(second_row, second_variant.target_identity_key),
        _row_identity_logit(second_row, first_variant.target_identity_key),
    )
    value = {
        "dense_variant_diagonal_advantages": dense_values,
        "dense_mean_diagonal_advantage": dense_mean,
        "dense_bidirectional_positive": dense_positive,
        "fractional_auc_variant_diagonal_advantages": auc_values,
        "fractional_auc_mean_diagonal_advantage": auc_mean,
        "fractional_auc_bidirectional_positive": auc_positive,
        "row_variant_diagonal_advantages": row_values,
        "row_mean_diagonal_advantage": row_mean,
        "row_bidirectional_positive": row_positive,
        "relation_output_changed": first["relation_sha256"] != second["relation_sha256"],
    }
    if set(value) != _PAIR_METRIC_FIELDS:
        raise RuntimeError("fixed-X prompt-switch metric fields changed")
    return value


def build_fixed_observation_evaluation_sample(
    *,
    checkpoint_global_step: int,
    item: FixedObservationEvaluationItem,
    mass_stratum: str,
    variant_results: Sequence[Mapping[str, object]],
    source_digest: str,
    non_language_model_inputs_sha256: str,
    language_model_inputs_sha256: Sequence[str],
    peak_cuda_reserved_bytes: int,
) -> dict[str, object]:
    """Build one recomputable two-true-prompt causal evaluation record."""

    value = {
        "schema": FIXED_OBSERVATION_EVALUATION_SAMPLE_SCHEMA,
        "checkpoint_global_step": checkpoint_global_step,
        "item": item.as_dict(),
        "mass_stratum": mass_stratum,
        "variant_results": [dict(result) for result in variant_results],
        "pair_metrics": fixed_observation_prompt_switch_metrics(variant_results),
        "source_digest": source_digest,
        "model_input_sha256": {
            "language": list(language_model_inputs_sha256),
            "non_language": non_language_model_inputs_sha256,
        },
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "non_language_inputs_equal": True,
        "loss_only_labels_visible_to_model": False,
        "target_resolution_happened_after_forward": True,
    }
    return validate_fixed_observation_evaluation_sample(value, expected_item=item)


def validate_fixed_observation_evaluation_sample(
    value: object,
    *,
    expected_item: FixedObservationEvaluationItem | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "checkpoint_global_step",
        "item",
        "loss_only_labels_visible_to_model",
        "mass_stratum",
        "model_input_sha256",
        "non_language_inputs_equal",
        "pair_metrics",
        "peak_cuda_reserved_bytes",
        "schema",
        "source_digest",
        "target_resolution_happened_after_forward",
        "variant_results",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("fixed-X evaluation sample fields differ from schema")
    if value["schema"] != FIXED_OBSERVATION_EVALUATION_SAMPLE_SCHEMA:
        raise ValueError("fixed-X evaluation sample schema changed")
    checkpoint = _nonnegative_int(
        value["checkpoint_global_step"],
        name="fixed-X checkpoint step",
    )
    item = FixedObservationEvaluationItem.from_dict(value["item"])
    if expected_item is not None and item != expected_item:
        raise ValueError("fixed-X evaluation sample differs from its frozen plan")
    if value["mass_stratum"] not in FIXED_OBSERVATION_MASS_STRATA:
        raise ValueError("fixed-X evaluation mass stratum changed")
    variants = value["variant_results"]
    if not isinstance(variants, list) or len(variants) != 2:
        raise ValueError("fixed-X evaluation sample requires two variant results")
    validated_variants = tuple(
        _validate_variant_result(
            result,
            checkpoint_global_step=checkpoint,
            item=item,
            expected_variant=variant,
        )
        for result, variant in zip(variants, item.variants, strict=True)
    )
    expected_metrics = fixed_observation_prompt_switch_metrics(validated_variants)
    if value["pair_metrics"] != expected_metrics:
        raise ValueError("fixed-X prompt-switch metrics were not recomputed")
    _sha256(value["source_digest"], name="fixed-X evaluation source digest")
    input_hashes = value["model_input_sha256"]
    if not isinstance(input_hashes, dict) or set(input_hashes) != {
        "language",
        "non_language",
    }:
        raise ValueError("fixed-X model-input hash fields changed")
    _sha256(input_hashes["non_language"], name="fixed-X non-language model inputs")
    language_hashes = input_hashes["language"]
    if not isinstance(language_hashes, list) or len(language_hashes) != 2:
        raise ValueError("fixed-X language model-input hashes are malformed")
    validated_language = tuple(
        _sha256(item, name="fixed-X language model inputs") for item in language_hashes
    )
    if len(set(validated_language)) != 2:
        raise ValueError("fixed-X evaluation retained identical tokenized language")
    _positive_int(
        value["peak_cuda_reserved_bytes"],
        name="fixed-X peak CUDA reservation",
    )
    if (
        value["non_language_inputs_equal"] is not True
        or value["loss_only_labels_visible_to_model"] is not False
        or value["target_resolution_happened_after_forward"] is not True
    ):
        raise ValueError("fixed-X causal or leakage boundary changed")
    return value


def _metric_values(
    samples: Sequence[dict[str, Any]],
    name: str,
) -> list[float]:
    values = []
    for sample in samples:
        value = sample["pair_metrics"][name]
        if value is not None:
            values.append(_finite_float(value, name=f"fixed-X aggregate {name}"))
    return values


def _aggregate_fixed_observation_samples(
    samples: Sequence[dict[str, Any]],
) -> dict[str, object]:
    if not samples:
        raise ValueError("fixed-X aggregate requires evaluation samples")
    dense = _metric_values(samples, "dense_mean_diagonal_advantage")
    auc = _metric_values(samples, "fractional_auc_mean_diagonal_advantage")
    row = _metric_values(samples, "row_mean_diagonal_advantage")

    def positive_fraction(name: str) -> float | None:
        eligible = [
            sample["pair_metrics"][name]
            for sample in samples
            if sample["pair_metrics"][name] is not None
        ]
        return None if not eligible else sum(value is True for value in eligible) / len(eligible)

    ownership_values = []
    forward_times = []
    for sample in samples:
        for result in sample["variant_results"]:
            target_iou = result["ownership_summary"]["target_soft_iou"]
            if target_iou is not None:
                ownership_values.append(
                    _finite_float(target_iou, name="fixed-X target ownership soft-IoU")
                )
            forward_times.append(
                _finite_float(result["forward_seconds"], name="fixed-X forward time")
            )
    return {
        "sample_count": len(samples),
        "dense_eligible_sample_count": len(dense),
        "mean_dense_diagonal_advantage": _optional_mean(dense),
        "dense_bidirectional_positive_fraction": positive_fraction("dense_bidirectional_positive"),
        "fractional_auc_eligible_sample_count": len(auc),
        "mean_fractional_auc_diagonal_advantage": _optional_mean(auc),
        "fractional_auc_bidirectional_positive_fraction": positive_fraction(
            "fractional_auc_bidirectional_positive"
        ),
        "row_eligible_sample_count": len(row),
        "mean_row_diagonal_advantage": _optional_mean(row),
        "row_bidirectional_positive_fraction": positive_fraction("row_bidirectional_positive"),
        "relation_output_changed_fraction": (
            sum(sample["pair_metrics"]["relation_output_changed"] is True for sample in samples)
            / len(samples)
        ),
        "mean_target_ownership_soft_iou": _optional_mean(ownership_values),
        "mean_forward_seconds_per_variant": math.fsum(forward_times) / len(forward_times),
        "maximum_peak_cuda_reserved_bytes": max(
            _positive_int(
                sample["peak_cuda_reserved_bytes"],
                name="fixed-X peak CUDA reservation",
            )
            for sample in samples
        ),
    }


def summarize_fixed_observation_evaluation_partition(
    samples: Sequence[Mapping[str, object]],
    *,
    partition: str,
) -> dict[str, object]:
    if partition not in FIXED_OBSERVATION_EVALUATION_PARTITIONS:
        raise ValueError("fixed-X evaluation partition is unsupported")
    validated = tuple(
        validate_fixed_observation_evaluation_sample(dict(sample)) for sample in samples
    )
    if not validated or any(sample["item"]["partition"] != partition for sample in validated):
        raise ValueError("fixed-X samples differ from their evaluation partition")
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tasks: set[str] = set()
    targets: set[str] = set()
    for sample in validated:
        by_stratum[str(sample["mass_stratum"])].append(sample)
        item = FixedObservationEvaluationItem.from_dict(sample["item"])
        tasks.update(variant.task_key for variant in item.variants)
        targets.update(variant.target_identity_key for variant in item.variants)
    if set(by_stratum) != set(FIXED_OBSERVATION_MASS_STRATA):
        raise ValueError("fixed-X partition lost one preregistered mass stratum")
    return {
        "partition": partition,
        "task_count": len(tasks),
        "target_count": len(targets),
        "overall": _aggregate_fixed_observation_samples(validated),
        "mass_strata": {
            name: _aggregate_fixed_observation_samples(by_stratum[name])
            for name in FIXED_OBSERVATION_MASS_STRATA
        },
    }


def build_fixed_observation_forward_equivalence_probe(
    *,
    plan: FixedObservationEvaluationPlan,
    item: FixedObservationEvaluationItem,
    checkpoint_global_step: int,
    model_inputs_sha256: str,
    relation_sha256: str,
    repeated_relation_sha256: str,
    repeat_forward_seconds: float,
) -> dict[str, object]:
    """Record one exact-input, exact-seed repeat without creating model evidence."""

    value = {
        "schema": FIXED_OBSERVATION_FORWARD_EQUIVALENCE_SCHEMA,
        "checkpoint_global_step": checkpoint_global_step,
        "partition": item.partition,
        "ordinal": item.ordinal,
        "rank": item.rank,
        "stateful_sample_key": item.group.stateful_sample_key,
        "variant_index": 0,
        "instruction_sha256": item.variants[0].instruction_sha256,
        "forward_seed": item.replay_seed,
        "model_inputs_sha256": model_inputs_sha256,
        "relation_sha256": relation_sha256,
        "repeated_relation_sha256": repeated_relation_sha256,
        "repeat_forward_seconds": repeat_forward_seconds,
        "same_model_inputs": True,
        "same_forward_seed": True,
        "same_previous_state": True,
        "relation_outputs_equal": relation_sha256 == repeated_relation_sha256,
    }
    return validate_fixed_observation_forward_equivalence_probe(
        value,
        plan=plan,
        expected_item=item,
    )


def validate_fixed_observation_forward_equivalence_probe(
    value: object,
    *,
    plan: FixedObservationEvaluationPlan,
    expected_item: FixedObservationEvaluationItem | None = None,
) -> dict[str, Any]:
    expected = {
        "checkpoint_global_step",
        "forward_seed",
        "instruction_sha256",
        "model_inputs_sha256",
        "ordinal",
        "partition",
        "rank",
        "relation_outputs_equal",
        "relation_sha256",
        "repeat_forward_seconds",
        "repeated_relation_sha256",
        "same_forward_seed",
        "same_model_inputs",
        "same_previous_state",
        "schema",
        "stateful_sample_key",
        "variant_index",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("fixed-X forward-equivalence fields differ from schema")
    if value["schema"] != FIXED_OBSERVATION_FORWARD_EQUIVALENCE_SCHEMA:
        raise ValueError("fixed-X forward-equivalence schema changed")
    _nonnegative_int(
        value["checkpoint_global_step"],
        name="fixed-X repeat checkpoint step",
    )
    partition = _text(value["partition"], name="fixed-X repeat partition")
    ordinal = _nonnegative_int(value["ordinal"], name="fixed-X repeat ordinal")
    rank = _nonnegative_int(value["rank"], name="fixed-X repeat rank")
    matches = tuple(
        item for item in plan.items if item.partition == partition and item.ordinal == ordinal
    )
    if len(matches) != 1:
        raise ValueError("fixed-X repeat item is absent from its plan")
    item = matches[0]
    if expected_item is not None and item != expected_item:
        raise ValueError("fixed-X repeat item differs from its designated control")
    if (
        rank != item.rank
        or value["stateful_sample_key"] != item.group.stateful_sample_key
        or value["variant_index"] != 0
        or value["instruction_sha256"] != item.variants[0].instruction_sha256
        or value["forward_seed"] != item.replay_seed
    ):
        raise ValueError("fixed-X repeat source or prompt identity changed")
    _sha256(value["model_inputs_sha256"], name="fixed-X repeat model inputs")
    first_relation = _sha256(
        value["relation_sha256"],
        name="fixed-X first relation",
    )
    repeated_relation = _sha256(
        value["repeated_relation_sha256"],
        name="fixed-X repeated relation",
    )
    if (
        _finite_float(
            value["repeat_forward_seconds"],
            name="fixed-X repeated forward time",
        )
        <= 0
    ):
        raise ValueError("fixed-X repeated forward time must be positive")
    if (
        value["same_model_inputs"] is not True
        or value["same_forward_seed"] is not True
        or value["same_previous_state"] is not True
        or value["relation_outputs_equal"] is not True
        or first_relation != repeated_relation
    ):
        raise ValueError("fixed-X exact-input forward is not reproducible")
    return value


def build_fixed_observation_evaluation_snapshot(
    *,
    checkpoint_global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    representation_split_sha256: str,
    plan: FixedObservationEvaluationPlan,
    representation_frozen_action_state_sha256: str,
    samples: Sequence[Mapping[str, object]],
    forward_equivalence_probes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    value = {
        "schema": FIXED_OBSERVATION_EVALUATION_SNAPSHOT_SCHEMA,
        "status": "PASS",
        "checkpoint_global_step": checkpoint_global_step,
        "implementation_sha256": implementation_sha256,
        "model_family_sha256": model_family_sha256,
        "representation_split_sha256": representation_split_sha256,
        "fixed_observation_evaluation_plan_sha256": plan.artifact_sha256,
        "representation_frozen_action_state_sha256": (representation_frozen_action_state_sha256),
        "samples": [dict(sample) for sample in samples],
        "forward_equivalence_probes": [dict(probe) for probe in forward_equivalence_probes],
        "partition_summaries": {},
    }
    validated = tuple(
        validate_fixed_observation_evaluation_sample(dict(sample)) for sample in samples
    )
    value["partition_summaries"] = {
        partition: summarize_fixed_observation_evaluation_partition(
            tuple(sample for sample in validated if sample["item"]["partition"] == partition),
            partition=partition,
        )
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
    }
    value["artifact_sha256"] = _canonical_sha256(value)
    return validate_fixed_observation_evaluation_snapshot(value, plan=plan)


def validate_fixed_observation_evaluation_snapshot(
    value: object,
    *,
    plan: FixedObservationEvaluationPlan,
) -> dict[str, Any]:
    expected = {
        "artifact_sha256",
        "checkpoint_global_step",
        "fixed_observation_evaluation_plan_sha256",
        "forward_equivalence_probes",
        "implementation_sha256",
        "model_family_sha256",
        "partition_summaries",
        "representation_frozen_action_state_sha256",
        "representation_split_sha256",
        "samples",
        "schema",
        "status",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("fixed-X evaluation snapshot fields differ from schema")
    if (
        value["schema"] != FIXED_OBSERVATION_EVALUATION_SNAPSHOT_SCHEMA
        or value["status"] != "PASS"
        or value["fixed_observation_evaluation_plan_sha256"] != plan.artifact_sha256
    ):
        raise ValueError("fixed-X evaluation snapshot identity changed")
    checkpoint = _nonnegative_int(
        value["checkpoint_global_step"],
        name="fixed-X checkpoint step",
    )
    for name in (
        "implementation_sha256",
        "model_family_sha256",
        "representation_split_sha256",
        "representation_frozen_action_state_sha256",
    ):
        _sha256(value[name], name=f"fixed-X snapshot {name}")
    raw_probes = value["forward_equivalence_probes"]
    if not isinstance(raw_probes, list) or len(raw_probes) != plan.world_size:
        raise ValueError("fixed-X forward-equivalence probe count changed")
    designated = tuple(
        (plan.items_for("validation", rank) + plan.items_for("heldout", rank))[0]
        for rank in range(plan.world_size)
    )
    if tuple(probe.get("rank") for probe in raw_probes if isinstance(probe, dict)) != tuple(
        range(plan.world_size)
    ):
        raise ValueError("fixed-X forward-equivalence probes are not rank ordered")
    for raw_probe, expected_item in zip(raw_probes, designated, strict=True):
        probe = validate_fixed_observation_forward_equivalence_probe(
            raw_probe,
            plan=plan,
            expected_item=expected_item,
        )
        if probe["checkpoint_global_step"] != checkpoint:
            raise ValueError("fixed-X repeat probe checkpoint differs from its snapshot")
    raw_samples = value["samples"]
    if not isinstance(raw_samples, list) or len(raw_samples) != len(plan.items):
        raise ValueError("fixed-X evaluation snapshot sample count changed")
    strata = fixed_observation_evaluation_mass_strata(plan)
    validated = []
    for raw, item in zip(raw_samples, plan.items, strict=True):
        sample = validate_fixed_observation_evaluation_sample(raw, expected_item=item)
        if sample["checkpoint_global_step"] != checkpoint:
            raise ValueError("fixed-X evaluation sample checkpoint differs from its snapshot")
        if sample["mass_stratum"] != strata[(item.partition, item.ordinal)]:
            raise ValueError("fixed-X evaluation sample mass stratum changed")
        validated.append(sample)
    expected_summaries = {
        partition: summarize_fixed_observation_evaluation_partition(
            tuple(sample for sample in validated if sample["item"]["partition"] == partition),
            partition=partition,
        )
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
    }
    if value["partition_summaries"] != expected_summaries:
        raise ValueError("fixed-X evaluation partition summaries were not recomputed")
    payload = dict(value)
    artifact = _sha256(payload.pop("artifact_sha256"), name="fixed-X snapshot artifact")
    if artifact != _canonical_sha256(payload):
        raise ValueError("fixed-X evaluation snapshot artifact SHA-256 changed")
    return value


def validate_fixed_observation_evaluation_visual_files(
    snapshot: Mapping[str, object],
    *,
    plan: FixedObservationEvaluationPlan,
    output_root: Path,
) -> None:
    validated = validate_fixed_observation_evaluation_snapshot(dict(snapshot), plan=plan)
    for sample in validated["samples"]:
        for result in sample["variant_results"]:
            visual = result["visual_artifact"]
            path = output_root / str(visual["path"])
            if (
                not path.is_file()
                or path.stat().st_size != visual["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != visual["sha256"]
            ):
                raise ValueError("fixed-X evaluation visual file differs from its report")
