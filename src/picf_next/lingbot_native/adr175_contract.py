"""Content-addressed broad-support strata evidence for ADR-175.

The contract is deliberately metadata-only.  It binds one source-disjoint
representation split to one frozen physical-event stream and replays the exact
prompt-selection algorithm used by training.  It never materializes a CALVIN
sample and therefore never reads images, masks, sidecars, or model outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.data.calvin import (
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.eval.calvin_task_relevance import (
    CalvinTaskPhysicalRelevance,
    calvin_task_physical_relevance_inventory,
)
from picf_next.lingbot_native.calvin import (
    build_native_calvin_physical_episode_domain,
    build_native_calvin_physical_sample_domain,
    native_calvin_sample_plan_instance_id,
    select_native_calvin_physical_prompt_segment,
)
from picf_next.lingbot_native.entity_evaluation_plan import EntityEvaluationPlan
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
from picf_next.training.control import (
    FrozenEpisodeStreamPlan,
    FrozenSamplePlan,
    TrainingPlan,
)

ADR175_BROAD_SUPPORT_CONTRACT_SCHEMA = "picf-next.adr175-broad-support-contract.v2"
ADR175_MATCHED_ARM_INPUT_SCHEMA = "picf-next.adr175-matched-arm-input-receipt.v2"
ADR175_EXACT_STRATUM = "exact-task-object"
ADR175_AMBIGUOUS_STRATUM = "ambiguous-set-only"
ADR175_AMBIGUOUS_SET_ONLY = "__ambiguous_set_only__"

_PARTITIONS = ("validation", "heldout")
_PARTITION_ORDER = {partition: index for index, partition in enumerate(_PARTITIONS)}
_EXPECTED_EXACT_TASK_COUNT = 29
_EXPECTED_AMBIGUOUS_TASK_COUNT = 5


def canonical_json_bytes(value: object) -> bytes:
    """Encode one finite JSON value with a stable byte representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("ADR-175 value is not finite canonical JSON") from error


def canonical_sha256(value: object) -> str:
    """Return the canonical JSON SHA-256 of one value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    text = _require_text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return text


def _require_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    result = tuple(_require_text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique strings")
    return result


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    result = tuple(_require_nonnegative_int(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must contain unique sorted integers")
    return result


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"ADR-175 JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _identity_tuple(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique strings")
    return values


@dataclass(frozen=True, slots=True)
class Adr175Stratum:
    """One task-conditioned exact identity tuple or fail-closed set-only row."""

    task_key: str
    kind: str
    identity_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.task_key, "ADR-175 stratum task key")
        _identity_tuple(self.identity_keys, "ADR-175 stratum identities")
        if self.kind == ADR175_EXACT_STRATUM:
            if not self.identity_keys or ADR175_AMBIGUOUS_SET_ONLY in self.identity_keys:
                raise ValueError("exact ADR-175 strata require physical target identities")
        elif self.kind == ADR175_AMBIGUOUS_STRATUM:
            if self.identity_keys:
                raise ValueError("ambiguous ADR-175 strata cannot name a target identity")
        else:
            raise ValueError("ADR-175 stratum kind changed")

    @property
    def key(self) -> tuple[str, tuple[str, ...] | str]:
        identity: tuple[str, ...] | str = (
            self.identity_keys if self.kind == ADR175_EXACT_STRATUM else ADR175_AMBIGUOUS_SET_ONLY
        )
        return self.task_key, identity

    def as_dict(self) -> dict[str, object]:
        return {
            "identity_keys": list(self.identity_keys),
            "kind": self.kind,
            "task_key": self.task_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> Adr175Stratum:
        fields = {"identity_keys", "kind", "task_key"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("ADR-175 stratum fields differ from schema")
        return cls(
            task_key=_require_text(value["task_key"], "ADR-175 stratum task key"),
            kind=_require_text(value["kind"], "ADR-175 stratum kind"),
            identity_keys=_string_tuple(
                value["identity_keys"],
                "ADR-175 stratum identities",
            ),
        )


@dataclass(frozen=True, slots=True)
class Adr175EvaluationItem:
    """One source-only evaluation segment annotated by the frozen task protocol."""

    partition: str
    task_key: str
    segment_index: int
    source_episode_index: int
    exact_action_target: bool
    action_target_identity_keys: tuple[str, ...]
    outcome_identity_keys: tuple[str, ...]
    known_participant_identity_keys: tuple[str, ...]
    ambiguity_reason: str | None
    stratum: Adr175Stratum

    def __post_init__(self) -> None:
        if self.partition not in _PARTITIONS:
            raise ValueError("ADR-175 evaluation partition changed")
        _require_text(self.task_key, "ADR-175 evaluation task key")
        _require_nonnegative_int(self.segment_index, "ADR-175 evaluation segment index")
        _require_nonnegative_int(
            self.source_episode_index,
            "ADR-175 evaluation source episode index",
        )
        for name, values in (
            ("action-target identities", self.action_target_identity_keys),
            ("outcome identities", self.outcome_identity_keys),
            ("known-participant identities", self.known_participant_identity_keys),
        ):
            _identity_tuple(values, f"ADR-175 evaluation {name}")
        if not isinstance(self.exact_action_target, bool):
            raise TypeError("ADR-175 exact-action-target flag must be boolean")
        if not isinstance(self.stratum, Adr175Stratum) or self.stratum.task_key != self.task_key:
            raise ValueError("ADR-175 evaluation stratum task differs")
        if self.exact_action_target:
            if (
                not self.action_target_identity_keys
                or self.ambiguity_reason is not None
                or self.stratum.kind != ADR175_EXACT_STRATUM
                or self.stratum.identity_keys != self.action_target_identity_keys
            ):
                raise ValueError("exact ADR-175 evaluation semantics are inconsistent")
        elif (
            self.action_target_identity_keys
            or not isinstance(self.ambiguity_reason, str)
            or not self.ambiguity_reason
            or self.stratum.kind != ADR175_AMBIGUOUS_STRATUM
            or self.stratum.identity_keys
        ):
            raise ValueError(
                "ambiguous ADR-175 evaluation items must fail closed without a singleton target"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "action_target_identity_keys": list(self.action_target_identity_keys),
            "ambiguity_reason": self.ambiguity_reason,
            "exact_action_target": self.exact_action_target,
            "known_participant_identity_keys": list(self.known_participant_identity_keys),
            "outcome_identity_keys": list(self.outcome_identity_keys),
            "partition": self.partition,
            "segment_index": self.segment_index,
            "source_episode_index": self.source_episode_index,
            "stratum": self.stratum.as_dict(),
            "task_key": self.task_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> Adr175EvaluationItem:
        fields = {
            "action_target_identity_keys",
            "ambiguity_reason",
            "exact_action_target",
            "known_participant_identity_keys",
            "outcome_identity_keys",
            "partition",
            "segment_index",
            "source_episode_index",
            "stratum",
            "task_key",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("ADR-175 evaluation item fields differ from schema")
        exact = value["exact_action_target"]
        if not isinstance(exact, bool):
            raise TypeError("ADR-175 exact-action-target flag must be boolean")
        reason = value["ambiguity_reason"]
        if reason is not None and not isinstance(reason, str):
            raise TypeError("ADR-175 ambiguity reason must be a string or null")
        return cls(
            partition=_require_text(value["partition"], "ADR-175 evaluation partition"),
            task_key=_require_text(value["task_key"], "ADR-175 evaluation task key"),
            segment_index=_require_nonnegative_int(
                value["segment_index"],
                "ADR-175 evaluation segment index",
            ),
            source_episode_index=_require_nonnegative_int(
                value["source_episode_index"],
                "ADR-175 evaluation source episode index",
            ),
            exact_action_target=exact,
            action_target_identity_keys=_string_tuple(
                value["action_target_identity_keys"],
                "ADR-175 action-target identities",
            ),
            outcome_identity_keys=_string_tuple(
                value["outcome_identity_keys"],
                "ADR-175 outcome identities",
            ),
            known_participant_identity_keys=_string_tuple(
                value["known_participant_identity_keys"],
                "ADR-175 known-participant identities",
            ),
            ambiguity_reason=reason,
            stratum=Adr175Stratum.from_dict(value["stratum"]),
        )


@dataclass(frozen=True, slots=True)
class Adr175TrainingPrefixCoverage:
    """Visit count and source-episode support for one frozen task stratum."""

    task_key: str
    exact_action_target: bool
    stratum: Adr175Stratum
    visit_count: int
    unique_source_episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_text(self.task_key, "ADR-175 training coverage task key")
        if not isinstance(self.exact_action_target, bool):
            raise TypeError("ADR-175 coverage exact-action-target flag must be boolean")
        if not isinstance(self.stratum, Adr175Stratum) or self.stratum.task_key != self.task_key:
            raise ValueError("ADR-175 training coverage stratum task differs")
        if self.exact_action_target != (self.stratum.kind == ADR175_EXACT_STRATUM):
            raise ValueError("ADR-175 training coverage exactness differs from its stratum")
        _require_positive_int(self.visit_count, "ADR-175 training stratum visit count")
        if not self.unique_source_episode_indices or self.unique_source_episode_indices != tuple(
            sorted(set(self.unique_source_episode_indices))
        ):
            raise ValueError(
                "ADR-175 training stratum source episodes must be nonempty, unique, and sorted"
            )
        for source_episode_index in self.unique_source_episode_indices:
            _require_nonnegative_int(
                source_episode_index,
                "ADR-175 training stratum source episode index",
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "exact_action_target": self.exact_action_target,
            "stratum": self.stratum.as_dict(),
            "task_key": self.task_key,
            "unique_source_episode_indices": list(self.unique_source_episode_indices),
            "visit_count": self.visit_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> Adr175TrainingPrefixCoverage:
        fields = {
            "exact_action_target",
            "stratum",
            "task_key",
            "unique_source_episode_indices",
            "visit_count",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("ADR-175 training coverage fields differ from schema")
        exact = value["exact_action_target"]
        if not isinstance(exact, bool):
            raise TypeError("ADR-175 coverage exact-action-target flag must be boolean")
        return cls(
            task_key=_require_text(value["task_key"], "ADR-175 coverage task key"),
            exact_action_target=exact,
            stratum=Adr175Stratum.from_dict(value["stratum"]),
            visit_count=_require_positive_int(
                value["visit_count"],
                "ADR-175 training stratum visit count",
            ),
            unique_source_episode_indices=_integer_tuple(
                value["unique_source_episode_indices"],
                "ADR-175 training stratum source episodes",
            ),
        )


_CONTRACT_FIELDS = {
    "ambiguous_task_count",
    "artifact_sha256",
    "comparison_id",
    "dataset_id",
    "dataset_manifest_sha256",
    "dataset_revision",
    "entity_evaluation_plan_artifact_sha256",
    "evaluation_items",
    "exact_task_count",
    "global_batch_size",
    "matched_arm_input_sha256",
    "plan_total_steps",
    "representation_split_artifact_sha256",
    "schema",
    "segments_per_task",
    "stream_plan_sha256",
    "task_relevance_inventory_sha256",
    "training_coverage",
    "training_prefix_prompt_receipts_sha256",
    "training_prefix_receipt_sha256",
    "training_prefix_sample_count",
    "training_prefix_sample_keys_sha256",
    "training_prefix_selected_segment_indices_sha256",
    "training_prefix_steps",
    "training_prefix_unique_source_episode_indices",
}


def _normalized_task_semantics(
    *,
    task_key: str,
    exact_action_target: bool,
    action_target_identity_keys: tuple[str, ...],
    outcome_identity_keys: tuple[str, ...],
    known_participant_identity_keys: tuple[str, ...],
    ambiguity_reason: str | None,
    stratum: Adr175Stratum,
) -> dict[str, object]:
    return {
        "action_target_identity_keys": list(action_target_identity_keys),
        "ambiguity_reason": ambiguity_reason,
        "exact_action_target": exact_action_target,
        "known_participant_identity_keys": list(known_participant_identity_keys),
        "outcome_identity_keys": list(outcome_identity_keys),
        "stratum": stratum.as_dict(),
        "task_key": task_key,
    }


def _inventory_sha256_from_evaluation(
    items: tuple[Adr175EvaluationItem, ...],
) -> str:
    by_task: dict[str, dict[str, object]] = {}
    for item in items:
        semantics = _normalized_task_semantics(
            task_key=item.task_key,
            exact_action_target=item.exact_action_target,
            action_target_identity_keys=item.action_target_identity_keys,
            outcome_identity_keys=item.outcome_identity_keys,
            known_participant_identity_keys=item.known_participant_identity_keys,
            ambiguity_reason=item.ambiguity_reason,
            stratum=item.stratum,
        )
        previous = by_task.setdefault(item.task_key, semantics)
        if previous != semantics:
            raise ValueError("ADR-175 evaluation task semantics differ across segments")
    return canonical_sha256([by_task[key] for key in sorted(by_task)])


def _matched_arm_input_sha256(
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_manifest_sha256: str,
    comparison_id: str,
    stream_plan_sha256: str,
    representation_split_artifact_sha256: str,
    entity_evaluation_plan_artifact_sha256: str,
    task_relevance_inventory_sha256: str,
    training_prefix_steps: int,
    training_prefix_sample_keys_sha256: str,
    training_prefix_prompt_receipts_sha256: str,
    training_prefix_selected_segment_indices_sha256: str,
    training_prefix_receipt_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "comparison_id": comparison_id,
            "dataset_id": dataset_id,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_revision": dataset_revision,
            "entity_evaluation_plan_artifact_sha256": (
                entity_evaluation_plan_artifact_sha256
            ),
            "representation_split_artifact_sha256": (representation_split_artifact_sha256),
            "schema": ADR175_MATCHED_ARM_INPUT_SCHEMA,
            "stream_plan_sha256": stream_plan_sha256,
            "task_relevance_inventory_sha256": task_relevance_inventory_sha256,
            "training_prefix_prompt_receipts_sha256": (training_prefix_prompt_receipts_sha256),
            "training_prefix_receipt_sha256": training_prefix_receipt_sha256,
            "training_prefix_sample_keys_sha256": training_prefix_sample_keys_sha256,
            "training_prefix_selected_segment_indices_sha256": (
                training_prefix_selected_segment_indices_sha256
            ),
            "training_prefix_steps": training_prefix_steps,
        }
    )


@dataclass(frozen=True, slots=True)
class Adr175BroadSupportContract:
    """Immutable task/object strata and training-prefix identity for all arms."""

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    comparison_id: str
    stream_plan_sha256: str
    representation_split_artifact_sha256: str
    entity_evaluation_plan_artifact_sha256: str
    task_relevance_inventory_sha256: str
    plan_total_steps: int
    global_batch_size: int
    training_prefix_steps: int
    training_prefix_sample_count: int
    training_prefix_sample_keys_sha256: str
    training_prefix_prompt_receipts_sha256: str
    training_prefix_selected_segment_indices_sha256: str
    training_prefix_receipt_sha256: str
    matched_arm_input_sha256: str
    training_prefix_unique_source_episode_indices: tuple[int, ...]
    segments_per_task: int
    exact_task_count: int
    ambiguous_task_count: int
    evaluation_items: tuple[Adr175EvaluationItem, ...]
    training_coverage: tuple[Adr175TrainingPrefixCoverage, ...]
    schema: str = ADR175_BROAD_SUPPORT_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ADR175_BROAD_SUPPORT_CONTRACT_SCHEMA:
            raise ValueError("ADR-175 broad-support contract schema changed")
        _require_text(self.dataset_id, "ADR-175 dataset id")
        _require_text(self.dataset_revision, "ADR-175 dataset revision")
        _require_sha256(self.dataset_manifest_sha256, "ADR-175 dataset manifest sha256")
        _require_text(self.comparison_id, "ADR-175 comparison id")
        for name, value in (
            ("stream plan sha256", self.stream_plan_sha256),
            ("representation split artifact sha256", self.representation_split_artifact_sha256),
            (
                "entity evaluation plan artifact sha256",
                self.entity_evaluation_plan_artifact_sha256,
            ),
            ("task relevance inventory sha256", self.task_relevance_inventory_sha256),
            ("training sample-key sha256", self.training_prefix_sample_keys_sha256),
            ("training prompt-receipt sha256", self.training_prefix_prompt_receipts_sha256),
            (
                "training selected-segment sha256",
                self.training_prefix_selected_segment_indices_sha256,
            ),
            ("training prefix receipt sha256", self.training_prefix_receipt_sha256),
            ("matched-arm input sha256", self.matched_arm_input_sha256),
        ):
            _require_sha256(value, f"ADR-175 {name}")
        _require_positive_int(self.plan_total_steps, "ADR-175 plan total steps")
        _require_positive_int(self.global_batch_size, "ADR-175 global batch size")
        _require_positive_int(self.training_prefix_steps, "ADR-175 training prefix steps")
        _require_positive_int(
            self.training_prefix_sample_count,
            "ADR-175 training prefix sample count",
        )
        _require_positive_int(self.segments_per_task, "ADR-175 segments per task")
        if self.training_prefix_steps > self.plan_total_steps:
            raise ValueError("ADR-175 training prefix exceeds the frozen plan")
        if self.training_prefix_sample_count != (
            self.training_prefix_steps * self.global_batch_size
        ):
            raise ValueError("ADR-175 training prefix sample count changed")
        if self.exact_task_count != _EXPECTED_EXACT_TASK_COUNT:
            raise ValueError("ADR-175 exact task inventory must contain exactly 29 tasks")
        if self.ambiguous_task_count != _EXPECTED_AMBIGUOUS_TASK_COUNT:
            raise ValueError("ADR-175 ambiguous task inventory must contain exactly 5 tasks")
        if (
            not self.training_prefix_unique_source_episode_indices
            or self.training_prefix_unique_source_episode_indices
            != tuple(sorted(set(self.training_prefix_unique_source_episode_indices)))
        ):
            raise ValueError(
                "ADR-175 training-prefix source episodes must be nonempty, unique, and sorted"
            )

        expected_items = tuple(
            sorted(
                self.evaluation_items,
                key=lambda item: (
                    _PARTITION_ORDER[item.partition],
                    item.task_key,
                    item.segment_index,
                ),
            )
        )
        if not self.evaluation_items or self.evaluation_items != expected_items:
            raise ValueError("ADR-175 evaluation items must be nonempty and canonically sorted")
        segment_keys = {(item.partition, item.segment_index) for item in self.evaluation_items}
        if len(segment_keys) != len(self.evaluation_items):
            raise ValueError("ADR-175 evaluation items contain duplicate segments")
        partition_tasks = {
            partition: {
                item.task_key for item in self.evaluation_items if item.partition == partition
            }
            for partition in _PARTITIONS
        }
        if partition_tasks["validation"] != partition_tasks["heldout"]:
            raise ValueError("ADR-175 evaluation partition task coverage differs")
        task_keys = partition_tasks["validation"]
        if len(task_keys) != self.exact_task_count + self.ambiguous_task_count:
            raise ValueError("ADR-175 evaluation task inventory is incomplete")
        for partition in _PARTITIONS:
            counts = Counter(
                item.task_key for item in self.evaluation_items if item.partition == partition
            )
            if set(counts) != task_keys or any(
                count != self.segments_per_task for count in counts.values()
            ):
                raise ValueError("ADR-175 evaluation replicate cardinality changed")
        exact_tasks = {item.task_key for item in self.evaluation_items if item.exact_action_target}
        if len(exact_tasks) != self.exact_task_count:
            raise ValueError("ADR-175 evaluation exact-task count changed")
        if len(task_keys - exact_tasks) != self.ambiguous_task_count:
            raise ValueError("ADR-175 evaluation ambiguous-task count changed")
        if _inventory_sha256_from_evaluation(self.evaluation_items) != (
            self.task_relevance_inventory_sha256
        ):
            raise ValueError("ADR-175 task relevance inventory SHA-256 changed")

        expected_coverage = tuple(sorted(self.training_coverage, key=lambda item: item.task_key))
        if not self.training_coverage or self.training_coverage != expected_coverage:
            raise ValueError("ADR-175 training coverage must be nonempty and sorted")
        coverage_tasks = {item.task_key for item in self.training_coverage}
        if len(coverage_tasks) != len(self.training_coverage) or coverage_tasks != task_keys:
            raise ValueError("ADR-175 training prefix does not cover the full task inventory")
        if sum(item.visit_count for item in self.training_coverage) != (
            self.training_prefix_sample_count
        ):
            raise ValueError("ADR-175 training stratum visits do not cover the prefix")
        coverage_sources = tuple(
            sorted(
                {
                    source_episode_index
                    for item in self.training_coverage
                    for source_episode_index in item.unique_source_episode_indices
                }
            )
        )
        if coverage_sources != self.training_prefix_unique_source_episode_indices:
            raise ValueError("ADR-175 training-prefix source coverage changed")
        coverage_exact = {
            item.task_key for item in self.training_coverage if item.exact_action_target
        }
        if coverage_exact != exact_tasks:
            raise ValueError("ADR-175 evaluation and training stratum exactness differ")

        expected_matched_receipt = _matched_arm_input_sha256(
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            comparison_id=self.comparison_id,
            stream_plan_sha256=self.stream_plan_sha256,
            representation_split_artifact_sha256=(self.representation_split_artifact_sha256),
            entity_evaluation_plan_artifact_sha256=(
                self.entity_evaluation_plan_artifact_sha256
            ),
            task_relevance_inventory_sha256=self.task_relevance_inventory_sha256,
            training_prefix_steps=self.training_prefix_steps,
            training_prefix_sample_keys_sha256=self.training_prefix_sample_keys_sha256,
            training_prefix_prompt_receipts_sha256=(self.training_prefix_prompt_receipts_sha256),
            training_prefix_selected_segment_indices_sha256=(
                self.training_prefix_selected_segment_indices_sha256
            ),
            training_prefix_receipt_sha256=self.training_prefix_receipt_sha256,
        )
        if self.matched_arm_input_sha256 != expected_matched_receipt:
            raise ValueError("ADR-175 matched-arm input receipt changed")

    def _payload(self) -> dict[str, object]:
        return {
            "ambiguous_task_count": self.ambiguous_task_count,
            "comparison_id": self.comparison_id,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "entity_evaluation_plan_artifact_sha256": (
                self.entity_evaluation_plan_artifact_sha256
            ),
            "evaluation_items": [item.as_dict() for item in self.evaluation_items],
            "exact_task_count": self.exact_task_count,
            "global_batch_size": self.global_batch_size,
            "matched_arm_input_sha256": self.matched_arm_input_sha256,
            "plan_total_steps": self.plan_total_steps,
            "representation_split_artifact_sha256": (self.representation_split_artifact_sha256),
            "schema": self.schema,
            "segments_per_task": self.segments_per_task,
            "stream_plan_sha256": self.stream_plan_sha256,
            "task_relevance_inventory_sha256": self.task_relevance_inventory_sha256,
            "training_coverage": [item.as_dict() for item in self.training_coverage],
            "training_prefix_prompt_receipts_sha256": (self.training_prefix_prompt_receipts_sha256),
            "training_prefix_receipt_sha256": self.training_prefix_receipt_sha256,
            "training_prefix_sample_count": self.training_prefix_sample_count,
            "training_prefix_sample_keys_sha256": self.training_prefix_sample_keys_sha256,
            "training_prefix_selected_segment_indices_sha256": (
                self.training_prefix_selected_segment_indices_sha256
            ),
            "training_prefix_steps": self.training_prefix_steps,
            "training_prefix_unique_source_episode_indices": list(
                self.training_prefix_unique_source_episode_indices
            ),
        }

    @property
    def artifact_sha256(self) -> str:
        return canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def write(self, path: str | Path) -> Path:
        payload = (
            json.dumps(
                self.as_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
        return write_bytes_durable_exclusive(path, payload)

    @classmethod
    def from_dict(cls, value: object) -> Adr175BroadSupportContract:
        if not isinstance(value, Mapping) or set(value) != _CONTRACT_FIELDS:
            raise ValueError("ADR-175 broad-support contract fields differ from schema")
        raw_evaluation = value["evaluation_items"]
        raw_coverage = value["training_coverage"]
        if not isinstance(raw_evaluation, list) or not isinstance(raw_coverage, list):
            raise ValueError("ADR-175 evaluation and coverage records must be JSON lists")
        contract = cls(
            schema=_require_text(value["schema"], "ADR-175 contract schema"),
            dataset_id=_require_text(value["dataset_id"], "ADR-175 dataset id"),
            dataset_revision=_require_text(
                value["dataset_revision"],
                "ADR-175 dataset revision",
            ),
            dataset_manifest_sha256=_require_sha256(
                value["dataset_manifest_sha256"],
                "ADR-175 dataset manifest sha256",
            ),
            comparison_id=_require_text(
                value["comparison_id"],
                "ADR-175 comparison id",
            ),
            stream_plan_sha256=_require_sha256(
                value["stream_plan_sha256"],
                "ADR-175 stream plan sha256",
            ),
            representation_split_artifact_sha256=_require_sha256(
                value["representation_split_artifact_sha256"],
                "ADR-175 representation split artifact sha256",
            ),
            entity_evaluation_plan_artifact_sha256=_require_sha256(
                value["entity_evaluation_plan_artifact_sha256"],
                "ADR-175 entity evaluation plan artifact sha256",
            ),
            task_relevance_inventory_sha256=_require_sha256(
                value["task_relevance_inventory_sha256"],
                "ADR-175 task relevance inventory sha256",
            ),
            plan_total_steps=_require_positive_int(
                value["plan_total_steps"],
                "ADR-175 plan total steps",
            ),
            global_batch_size=_require_positive_int(
                value["global_batch_size"],
                "ADR-175 global batch size",
            ),
            training_prefix_steps=_require_positive_int(
                value["training_prefix_steps"],
                "ADR-175 training prefix steps",
            ),
            training_prefix_sample_count=_require_positive_int(
                value["training_prefix_sample_count"],
                "ADR-175 training prefix sample count",
            ),
            training_prefix_sample_keys_sha256=_require_sha256(
                value["training_prefix_sample_keys_sha256"],
                "ADR-175 training sample-key sha256",
            ),
            training_prefix_prompt_receipts_sha256=_require_sha256(
                value["training_prefix_prompt_receipts_sha256"],
                "ADR-175 training prompt-receipt sha256",
            ),
            training_prefix_selected_segment_indices_sha256=_require_sha256(
                value["training_prefix_selected_segment_indices_sha256"],
                "ADR-175 training selected-segment sha256",
            ),
            training_prefix_receipt_sha256=_require_sha256(
                value["training_prefix_receipt_sha256"],
                "ADR-175 training prefix receipt sha256",
            ),
            matched_arm_input_sha256=_require_sha256(
                value["matched_arm_input_sha256"],
                "ADR-175 matched-arm input sha256",
            ),
            training_prefix_unique_source_episode_indices=_integer_tuple(
                value["training_prefix_unique_source_episode_indices"],
                "ADR-175 training-prefix source episodes",
            ),
            segments_per_task=_require_positive_int(
                value["segments_per_task"],
                "ADR-175 segments per task",
            ),
            exact_task_count=_require_positive_int(
                value["exact_task_count"],
                "ADR-175 exact task count",
            ),
            ambiguous_task_count=_require_positive_int(
                value["ambiguous_task_count"],
                "ADR-175 ambiguous task count",
            ),
            evaluation_items=tuple(Adr175EvaluationItem.from_dict(item) for item in raw_evaluation),
            training_coverage=tuple(
                Adr175TrainingPrefixCoverage.from_dict(item) for item in raw_coverage
            ),
        )
        expected = _require_sha256(
            value["artifact_sha256"],
            "ADR-175 artifact sha256",
        )
        if contract.artifact_sha256 != expected:
            raise ValueError("ADR-175 broad-support artifact SHA-256 changed")
        return contract

    @classmethod
    def load(cls, path: str | Path) -> Adr175BroadSupportContract:
        source = Path(path)
        if source.is_symlink():
            raise ValueError("ADR-175 contract must be one direct regular file")
        try:
            text = source.read_text(encoding="ascii")
            value = json.loads(text, object_pairs_hook=_duplicate_rejecting_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid ADR-175 broad-support contract: {source}") from error
        return cls.from_dict(value)


def _validated_relevance_inventory() -> tuple[CalvinTaskPhysicalRelevance, ...]:
    inventory = calvin_task_physical_relevance_inventory()
    task_keys = tuple(item.task_key for item in inventory)
    if task_keys != tuple(sorted(set(task_keys))):
        raise ValueError("CALVIN task relevance inventory is not canonical")
    exact = tuple(item for item in inventory if item.exact_action_target)
    ambiguous = tuple(item for item in inventory if not item.exact_action_target)
    if len(exact) != _EXPECTED_EXACT_TASK_COUNT or len(ambiguous) != (
        _EXPECTED_AMBIGUOUS_TASK_COUNT
    ):
        raise ValueError("CALVIN task relevance inventory is not the reviewed 29+5 protocol")
    if any(not item.action_target_identity_keys for item in exact):
        raise ValueError("exact CALVIN task relevance omitted its physical target")
    if any(
        item.action_target_identity_keys
        or not item.exclusion_reason
        or len(item.action_target_identity_keys) == 1
        for item in ambiguous
    ):
        raise ValueError("ambiguous CALVIN task relevance emitted a singleton target")
    return inventory


def _stratum_for_relevance(relevance: CalvinTaskPhysicalRelevance) -> Adr175Stratum:
    return Adr175Stratum(
        task_key=relevance.task_key,
        kind=(ADR175_EXACT_STRATUM if relevance.exact_action_target else ADR175_AMBIGUOUS_STRATUM),
        identity_keys=(
            relevance.action_target_identity_keys if relevance.exact_action_target else ()
        ),
    )


def _task_semantics_payload(
    inventory: tuple[CalvinTaskPhysicalRelevance, ...],
) -> list[dict[str, object]]:
    return [
        _normalized_task_semantics(
            task_key=relevance.task_key,
            exact_action_target=relevance.exact_action_target,
            action_target_identity_keys=relevance.action_target_identity_keys,
            outcome_identity_keys=relevance.outcome_identity_keys,
            known_participant_identity_keys=relevance.known_participant_identity_keys,
            ambiguity_reason=relevance.exclusion_reason,
            stratum=_stratum_for_relevance(relevance),
        )
        for relevance in inventory
    ]


def build_adr175_broad_support_contract(
    *,
    dataset: CalvinPhysicalTransitionDataset,
    stream_plan: TrainingPlan,
    representation_split: RepresentationTrialSplit,
    entity_evaluation_plan: EntityEvaluationPlan,
    training_prefix_steps: int,
) -> Adr175BroadSupportContract:
    """Build one arm-independent ADR-175 companion without sensor decoding."""

    if not isinstance(dataset, CalvinPhysicalTransitionDataset):
        raise TypeError("ADR-175 requires the unique CALVIN physical-event dataset")
    if not isinstance(stream_plan, FrozenSamplePlan | FrozenEpisodeStreamPlan):
        raise TypeError("ADR-175 requires one frozen physical-event training plan")
    if not isinstance(representation_split, RepresentationTrialSplit):
        raise TypeError("ADR-175 requires one typed representation trial split")
    if not isinstance(entity_evaluation_plan, EntityEvaluationPlan):
        raise TypeError("ADR-175 requires one typed entity evaluation plan")
    training_prefix_steps = _require_positive_int(
        training_prefix_steps,
        "ADR-175 training prefix steps",
    )
    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("ADR-175 requires a content-addressed CALVIN dataset")
    dataset_identity = (
        dataset.index.dataset_id,
        dataset.index.dataset_revision,
        manifest.tree_sha256,
    )
    plan_identity = (
        stream_plan.dataset_id,
        stream_plan.dataset_revision,
        stream_plan.dataset_manifest_sha256,
    )
    split_identity = (
        representation_split.dataset_id,
        representation_split.dataset_revision,
        representation_split.dataset_manifest_sha256,
    )
    if dataset_identity != plan_identity or dataset_identity != split_identity:
        raise ValueError("ADR-175 dataset, plan, and split identities differ")
    if representation_split.comparison_id != stream_plan.comparison_id:
        raise ValueError("ADR-175 split and stream comparison identities differ")
    if representation_split.stream_plan_sha256 != stream_plan.plan_sha256:
        raise ValueError("ADR-175 split is not bound to the supplied stream plan")
    if representation_split.training_steps != stream_plan.total_steps:
        raise ValueError("ADR-175 split must bind the complete frozen plan budget")
    if entity_evaluation_plan.representation_split_sha256 != (
        representation_split.artifact_sha256
    ):
        raise ValueError("ADR-175 entity evaluation plan is not bound to its split")
    if entity_evaluation_plan.world_size != stream_plan.global_batch_size:
        raise ValueError("ADR-175 entity evaluation and training world sizes differ")
    if training_prefix_steps > stream_plan.total_steps:
        raise ValueError("ADR-175 training prefix exceeds the frozen stream plan")

    evaluation_segments = {
        (partition, segment.segment_index): segment
        for partition, segments in (
            ("validation", representation_split.validation_segments),
            ("heldout", representation_split.heldout_segments),
        )
        for segment in segments
    }
    stateful_dataset = CalvinStatefulTransitionDataset(dataset.index, action_horizon=1)
    for item in entity_evaluation_plan.items:
        try:
            segment = evaluation_segments[(item.partition, item.segment_index)]
        except KeyError as error:
            raise ValueError(
                "ADR-175 entity evaluation item lies outside the reviewed split"
            ) from error
        if (
            item.task_key != segment.task_key
            or item.source_episode_index != segment.source_episode_index
            or not segment.source_start <= item.source_global_index < segment.source_end
            or item.transition_index != item.source_global_index - segment.source_start
        ):
            raise ValueError("ADR-175 entity evaluation source coordinates changed")
        try:
            locator = stateful_dataset.locator_by_key(item.sample_key)
        except KeyError as error:
            raise ValueError(
                "ADR-175 entity evaluation sample key is absent from the dataset"
            ) from error
        if (
            locator.segment_index != item.segment_index
            or locator.global_index != item.source_global_index
        ):
            raise ValueError("ADR-175 entity evaluation sample key changed source identity")

    evaluation_sources = representation_split.evaluation_source_episode_indices
    if representation_split.stream_domain_excluded_source_episode_indices != evaluation_sources:
        raise ValueError(
            "ADR-175 broad-support stream must exclude the full evaluation source domain"
        )
    if isinstance(stream_plan, FrozenSamplePlan):
        expected_sample_domain = build_native_calvin_physical_sample_domain(
            dataset,
            excluded_source_episode_indices=evaluation_sources,
        )
        if stream_plan.sample_keys != expected_sample_domain:
            raise ValueError(
                "ADR-175 sample plan must contain the exact evaluation-excluded physical domain"
            )
    else:
        expected_episode_domain = build_native_calvin_physical_episode_domain(
            dataset,
            excluded_source_episode_indices=evaluation_sources,
        )
        if stream_plan.episodes != expected_episode_domain:
            raise ValueError(
                "ADR-175 stream plan must contain the exact evaluation-excluded physical domain"
            )

    inventory = _validated_relevance_inventory()
    relevance_by_task = {item.task_key: item for item in inventory}
    inventory_task_keys = set(relevance_by_task)
    task_relevance_inventory_sha256 = canonical_sha256(_task_semantics_payload(inventory))

    evaluation_items: list[Adr175EvaluationItem] = []
    for partition, segments in (
        ("validation", representation_split.validation_segments),
        ("heldout", representation_split.heldout_segments),
    ):
        observed_tasks = {segment.task_key for segment in segments}
        if observed_tasks != inventory_task_keys:
            missing = sorted(inventory_task_keys - observed_tasks)
            extra = sorted(observed_tasks - inventory_task_keys)
            raise ValueError(
                "ADR-175 evaluation task inventory differs from the reviewed protocol: "
                f"missing={missing}, extra={extra}"
            )
        for selected in segments:
            try:
                source = dataset.index.segments[selected.segment_index]
            except IndexError as error:
                raise ValueError("ADR-175 evaluation segment is absent from CALVIN") from error
            observed = (
                source.task_key,
                int(source.episode_index),
                int(source.start),
                int(source.end),
            )
            expected = (
                selected.task_key,
                selected.source_episode_index,
                selected.source_start,
                selected.source_end,
            )
            if observed != expected:
                raise ValueError("ADR-175 evaluation segment metadata changed")
            relevance = relevance_by_task[selected.task_key]
            evaluation_items.append(
                Adr175EvaluationItem(
                    partition=partition,
                    task_key=relevance.task_key,
                    segment_index=selected.segment_index,
                    source_episode_index=selected.source_episode_index,
                    exact_action_target=relevance.exact_action_target,
                    action_target_identity_keys=relevance.action_target_identity_keys,
                    outcome_identity_keys=relevance.outcome_identity_keys,
                    known_participant_identity_keys=(relevance.known_participant_identity_keys),
                    ambiguity_reason=relevance.exclusion_reason,
                    stratum=_stratum_for_relevance(relevance),
                )
            )
    canonical_evaluation = tuple(
        sorted(
            evaluation_items,
            key=lambda item: (
                _PARTITION_ORDER[item.partition],
                item.task_key,
                item.segment_index,
            ),
        )
    )

    task_visits: Counter[str] = Counter()
    task_sources: dict[str, set[int]] = defaultdict(set)
    sample_keys: list[str] = []
    prompt_receipts: list[str] = []
    selected_segment_indices: list[int] = []
    prefix_receipts: list[dict[str, object]] = []
    prefix_sources: set[int] = set()
    evaluation_source_set = set(evaluation_sources)
    split_training_sources = set(representation_split.training_source_episode_indices)
    for optimizer_step in range(training_prefix_steps):
        batch = stream_plan.global_batch(optimizer_step)
        occurrences = (
            tuple(
                {
                    "episode_instance_id": native_calvin_sample_plan_instance_id(
                        optimizer_step=optimizer_step,
                        sample=sample,
                    ),
                    "episode_key": None,
                    "global_slot": global_slot,
                    "lane_id": None,
                    "sample": sample,
                    "transition_index": None,
                }
                for global_slot, sample in enumerate(batch.samples)
            )
            if isinstance(stream_plan, FrozenSamplePlan)
            else tuple(
                {
                    "episode_instance_id": transition.episode_instance_id,
                    "episode_key": transition.episode_key,
                    "global_slot": global_slot,
                    "lane_id": transition.lane_id,
                    "sample": transition.sample,
                    "transition_index": transition.transition_index,
                }
                for global_slot, transition in enumerate(batch.transitions)
            )
        )
        if len(occurrences) != stream_plan.global_batch_size:
            raise RuntimeError("ADR-175 frozen stream changed its global batch cardinality")
        for occurrence in occurrences:
            sample = occurrence["sample"]
            sample_key = sample.sample_key
            selected_segment_index, prompt_receipt = select_native_calvin_physical_prompt_segment(
                dataset,
                sample_key=sample_key,
                plan_sha256=stream_plan.plan_sha256,
                episode_instance_id=occurrence["episode_instance_id"],
            )
            segment = dataset.index.segments[selected_segment_index]
            event = dataset.event_by_key(sample_key)
            source_episode_index = int(event.episode.index)
            if int(segment.episode_index) != source_episode_index:
                raise RuntimeError("ADR-175 prompt selection crossed a physical source episode")
            if source_episode_index in evaluation_source_set:
                raise ValueError("ADR-175 training prefix reused an evaluation source episode")
            if source_episode_index not in split_training_sources:
                raise ValueError("ADR-175 training prefix is absent from split training evidence")
            try:
                relevance = relevance_by_task[segment.task_key]
            except KeyError as error:
                raise ValueError(
                    f"ADR-175 training prompt uses an unreviewed task {segment.task_key!r}"
                ) from error
            stratum = _stratum_for_relevance(relevance)
            task_visits[segment.task_key] += 1
            task_sources[segment.task_key].add(source_episode_index)
            prefix_sources.add(source_episode_index)
            sample_keys.append(sample_key)
            prompt_receipts.append(prompt_receipt)
            selected_segment_indices.append(selected_segment_index)
            prefix_receipts.append(
                {
                    "augmentation_seed": sample.augmentation_seed,
                    "episode_instance_id": occurrence["episode_instance_id"],
                    "episode_key": (
                        occurrence["episode_key"]
                        or f"calvin-source-episode-{source_episode_index:08d}"
                    ),
                    "flow_noise_seed": sample.flow_noise_seed,
                    "flow_timestep_seed": sample.flow_timestep_seed,
                    "global_slot": occurrence["global_slot"],
                    "lane_id": occurrence["lane_id"],
                    "optimizer_step": optimizer_step,
                    "prompt_selection_receipt_sha256": prompt_receipt,
                    "sample_index": sample.sample_index,
                    "sample_key": sample_key,
                    "selected_segment_index": selected_segment_index,
                    "source_episode_index": source_episode_index,
                    "stratum": stratum.as_dict(),
                    "task_key": segment.task_key,
                    "transition_index": (
                        occurrence["transition_index"]
                        if occurrence["transition_index"] is not None
                        else int(event.event_index)
                    ),
                }
            )

    missing_prefix_tasks = sorted(inventory_task_keys - set(task_visits))
    if missing_prefix_tasks:
        raise ValueError(
            f"ADR-175 training prefix lacks broad task coverage: {missing_prefix_tasks}"
        )
    coverage = tuple(
        Adr175TrainingPrefixCoverage(
            task_key=relevance.task_key,
            exact_action_target=relevance.exact_action_target,
            stratum=_stratum_for_relevance(relevance),
            visit_count=task_visits[relevance.task_key],
            unique_source_episode_indices=tuple(sorted(task_sources[relevance.task_key])),
        )
        for relevance in inventory
    )
    sample_keys_sha256 = canonical_sha256(sample_keys)
    prompt_receipts_sha256 = canonical_sha256(prompt_receipts)
    selected_segments_sha256 = canonical_sha256(selected_segment_indices)
    prefix_receipt_sha256 = canonical_sha256(prefix_receipts)
    matched_arm_input_sha256 = _matched_arm_input_sha256(
        dataset_id=dataset.index.dataset_id,
        dataset_revision=dataset.index.dataset_revision,
        dataset_manifest_sha256=manifest.tree_sha256,
        comparison_id=stream_plan.comparison_id,
        stream_plan_sha256=stream_plan.plan_sha256,
        representation_split_artifact_sha256=representation_split.artifact_sha256,
        entity_evaluation_plan_artifact_sha256=(entity_evaluation_plan.artifact_sha256),
        task_relevance_inventory_sha256=task_relevance_inventory_sha256,
        training_prefix_steps=training_prefix_steps,
        training_prefix_sample_keys_sha256=sample_keys_sha256,
        training_prefix_prompt_receipts_sha256=prompt_receipts_sha256,
        training_prefix_selected_segment_indices_sha256=selected_segments_sha256,
        training_prefix_receipt_sha256=prefix_receipt_sha256,
    )
    return Adr175BroadSupportContract(
        dataset_id=dataset.index.dataset_id,
        dataset_revision=dataset.index.dataset_revision,
        dataset_manifest_sha256=manifest.tree_sha256,
        comparison_id=stream_plan.comparison_id,
        stream_plan_sha256=stream_plan.plan_sha256,
        representation_split_artifact_sha256=representation_split.artifact_sha256,
        entity_evaluation_plan_artifact_sha256=(entity_evaluation_plan.artifact_sha256),
        task_relevance_inventory_sha256=task_relevance_inventory_sha256,
        plan_total_steps=stream_plan.total_steps,
        global_batch_size=stream_plan.global_batch_size,
        training_prefix_steps=training_prefix_steps,
        training_prefix_sample_count=len(sample_keys),
        training_prefix_sample_keys_sha256=sample_keys_sha256,
        training_prefix_prompt_receipts_sha256=prompt_receipts_sha256,
        training_prefix_selected_segment_indices_sha256=selected_segments_sha256,
        training_prefix_receipt_sha256=prefix_receipt_sha256,
        matched_arm_input_sha256=matched_arm_input_sha256,
        training_prefix_unique_source_episode_indices=tuple(sorted(prefix_sources)),
        segments_per_task=representation_split.segments_per_task,
        exact_task_count=sum(item.exact_action_target for item in inventory),
        ambiguous_task_count=sum(not item.exact_action_target for item in inventory),
        evaluation_items=canonical_evaluation,
        training_coverage=coverage,
    )
