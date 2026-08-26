"""Frozen source coverage for full-modal evidence caches.

The plan is the exact image of a bounded training stream and its source-disjoint
evaluation plan on CALVIN's canonical physical-event axis. It decides no object
identity, task relevance, anchor ownership, or sensor availability.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from picf_next.content_addressing import canonical_payload_sha256
from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.lingbot_native.entity_evaluation_plan import (
    EntityEvaluationPlan,
    build_entity_evaluation_plan,
)
from picf_next.lingbot_native.representation_split import (
    RepresentationTrialSplit,
    verify_representation_trial_split_training_evidence,
)
from picf_next.training.control import FrozenEpisodeStreamPlan

DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1 = "picf-next.dense-evidence-coverage-plan/v1"
DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA = "picf-next.dense-evidence-coverage-plan/v2"
CALVIN_FULL_DENSE_MODALITIES = ("anytouch", "sonata", "vjepa")

_RECORD_FIELDS = frozenset({"partition", "sample_key", "source_global_index"})
_PLAN_FIELDS_V1 = frozenset(
    {
        "artifact_sha256",
        "dataset_id",
        "dataset_revision",
        "dataset_tree_sha256",
        "evaluation_item_count",
        "evaluation_plan_sha256",
        "modalities",
        "records",
        "records_sha256",
        "representation_split_sha256",
        "schema",
        "stream_plan_sha256",
        "training_visit_count",
        "training_visits_sha256",
    }
)
_PLAN_FIELDS_V2 = _PLAN_FIELDS_V1 | frozenset(
    {
        "evaluation_history_transition_count",
        "evaluation_history_visit_count",
        "evaluation_history_visits_sha256",
        "evaluation_record_count",
    }
)
_PARTITIONS = frozenset({"evaluation", "training"})


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, name: str) -> str:
    result = _text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class DenseEvidenceCoverageRecord:
    source_global_index: int
    sample_key: str
    partition: str

    def __post_init__(self) -> None:
        _nonnegative_integer(self.source_global_index, "coverage source global index")
        _text(self.sample_key, "coverage sample key")
        if self.partition not in _PARTITIONS:
            raise ContractError("coverage partition must be training or evaluation")

    def as_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "sample_key": self.sample_key,
            "source_global_index": self.source_global_index,
        }

    @classmethod
    def from_dict(cls, value: object) -> DenseEvidenceCoverageRecord:
        if not isinstance(value, Mapping) or set(value) != _RECORD_FIELDS:
            raise ContractError("dense evidence coverage record fields differ from schema")
        return cls(
            source_global_index=_nonnegative_integer(
                value["source_global_index"], "coverage source global index"
            ),
            sample_key=_text(value["sample_key"], "coverage sample key"),
            partition=_text(value["partition"], "coverage partition"),
        )


@dataclass(frozen=True, slots=True)
class DenseEvidenceCoveragePlan:
    dataset_id: str
    dataset_revision: str
    dataset_tree_sha256: str
    stream_plan_sha256: str
    representation_split_sha256: str
    evaluation_plan_sha256: str
    training_visit_count: int
    training_visits_sha256: str
    evaluation_item_count: int
    records: tuple[DenseEvidenceCoverageRecord, ...]
    evaluation_record_count: int | None = None
    evaluation_history_transition_count: int = 0
    evaluation_history_visit_count: int = 0
    evaluation_history_visits_sha256: str | None = None
    modalities: tuple[str, ...] = CALVIN_FULL_DENSE_MODALITIES
    schema: str = DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in {
            DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1,
            DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA,
        }:
            raise ContractError("dense evidence coverage plan schema changed")
        _text(self.dataset_id, "coverage dataset id")
        _text(self.dataset_revision, "coverage dataset revision")
        for value, name in (
            (self.dataset_tree_sha256, "coverage dataset tree sha256"),
            (self.stream_plan_sha256, "coverage stream plan sha256"),
            (self.representation_split_sha256, "coverage representation split sha256"),
            (self.evaluation_plan_sha256, "coverage evaluation plan sha256"),
            (self.training_visits_sha256, "coverage training visits sha256"),
        ):
            _sha256(value, name)
        _nonnegative_integer(self.training_visit_count, "coverage training visit count")
        _nonnegative_integer(self.evaluation_item_count, "coverage evaluation item count")
        evaluation_record_count = (
            self.evaluation_item_count
            if self.evaluation_record_count is None
            else _nonnegative_integer(
                self.evaluation_record_count,
                "coverage evaluation record count",
            )
        )
        history_transition_count = _nonnegative_integer(
            self.evaluation_history_transition_count,
            "coverage evaluation history transition count",
        )
        history_visit_count = _nonnegative_integer(
            self.evaluation_history_visit_count,
            "coverage evaluation history visit count",
        )
        history_visits_sha256 = self.evaluation_history_visits_sha256
        if history_visits_sha256 is None:
            history_visits_sha256 = canonical_payload_sha256(
                "picf-next.dense-evidence-evaluation-history-visits/v1",
                [],
            )
        _sha256(
            history_visits_sha256,
            "coverage evaluation history visits sha256",
        )
        object.__setattr__(self, "evaluation_record_count", evaluation_record_count)
        object.__setattr__(
            self,
            "evaluation_history_visits_sha256",
            history_visits_sha256,
        )
        if self.training_visit_count <= 0 or self.evaluation_item_count <= 0:
            raise ContractError("dense evidence coverage requires training and evaluation rows")
        if self.modalities != CALVIN_FULL_DENSE_MODALITIES:
            raise ContractError("CALVIN full-modal coverage modality order changed")
        if not self.records or any(
            not isinstance(record, DenseEvidenceCoverageRecord) for record in self.records
        ):
            raise ContractError("dense evidence coverage requires typed records")
        identities = tuple(
            (record.source_global_index, record.sample_key) for record in self.records
        )
        if identities != tuple(sorted(identities)):
            raise ContractError("dense evidence coverage records must be source sorted")
        if len({index for index, _ in identities}) != len(identities):
            raise ContractError("dense evidence coverage source indices must be unique")
        if len({sample_key for _, sample_key in identities}) != len(identities):
            raise ContractError("dense evidence coverage sample keys must be unique")
        training_count = sum(record.partition == "training" for record in self.records)
        evaluation_count = sum(record.partition == "evaluation" for record in self.records)
        if not 0 < training_count <= self.training_visit_count:
            raise ContractError("coverage training records exceed or omit training visits")
        if self.schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1:
            if (
                evaluation_count != self.evaluation_item_count
                or evaluation_record_count != self.evaluation_item_count
                or history_transition_count != 0
                or history_visit_count != 0
            ):
                raise ContractError("v1 coverage requires one record per evaluation item")
        else:
            if evaluation_count != evaluation_record_count:
                raise ContractError("coverage evaluation record count differs from records")
            if evaluation_record_count < self.evaluation_item_count:
                raise ContractError("coverage evaluation records omit current evaluation items")
            if history_transition_count == 0 and history_visit_count != 0:
                raise ContractError("zero-history coverage cannot contain history visits")
            if history_transition_count > 0 and (
                history_visit_count == 0
                or history_visit_count % history_transition_count != 0
            ):
                raise ContractError("coverage history visits do not form complete prefixes")

    @property
    def records_sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes([record.as_dict() for record in self.records]))

    @property
    def record_identities(self) -> tuple[tuple[int, str], ...]:
        return tuple((record.source_global_index, record.sample_key) for record in self.records)

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "dataset_tree_sha256": self.dataset_tree_sha256,
            "evaluation_item_count": self.evaluation_item_count,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "modalities": list(self.modalities),
            "records": [record.as_dict() for record in self.records],
            "records_sha256": self.records_sha256,
            "representation_split_sha256": self.representation_split_sha256,
            "schema": self.schema,
            "stream_plan_sha256": self.stream_plan_sha256,
            "training_visit_count": self.training_visit_count,
            "training_visits_sha256": self.training_visits_sha256,
        }
        if self.schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA:
            payload.update(
                {
                    "evaluation_history_transition_count": (
                        self.evaluation_history_transition_count
                    ),
                    "evaluation_history_visit_count": self.evaluation_history_visit_count,
                    "evaluation_history_visits_sha256": (
                        self.evaluation_history_visits_sha256
                    ),
                    "evaluation_record_count": self.evaluation_record_count,
                }
            )
        return payload

    @property
    def artifact_sha256(self) -> str:
        namespace = (
            "picf-next.dense-evidence-coverage-plan-artifact/v1"
            if self.schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1
            else "picf-next.dense-evidence-coverage-plan-artifact/v2"
        )
        return canonical_payload_sha256(
            namespace,
            self._payload(),
        )

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{self.artifact_sha256[:12]}"
        )
        payload = json.dumps(self.as_dict(), indent=2, sort_keys=True).encode("ascii") + b"\n"
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def from_dict(cls, value: object) -> DenseEvidenceCoveragePlan:
        if not isinstance(value, Mapping):
            raise ContractError("dense evidence coverage plan fields differ from schema")
        schema = value.get("schema")
        expected_fields = (
            _PLAN_FIELDS_V1
            if schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1
            else _PLAN_FIELDS_V2
        )
        if set(value) != expected_fields:
            raise ContractError("dense evidence coverage plan fields differ from schema")
        raw_records = value["records"]
        raw_modalities = value["modalities"]
        if not isinstance(raw_records, list) or not isinstance(raw_modalities, list):
            raise ContractError("dense evidence coverage records/modalities must be lists")
        plan = cls(
            schema=_text(value["schema"], "coverage schema"),
            dataset_id=_text(value["dataset_id"], "coverage dataset id"),
            dataset_revision=_text(value["dataset_revision"], "coverage dataset revision"),
            dataset_tree_sha256=_sha256(
                value["dataset_tree_sha256"], "coverage dataset tree sha256"
            ),
            stream_plan_sha256=_sha256(value["stream_plan_sha256"], "coverage stream plan sha256"),
            representation_split_sha256=_sha256(
                value["representation_split_sha256"],
                "coverage representation split sha256",
            ),
            evaluation_plan_sha256=_sha256(
                value["evaluation_plan_sha256"], "coverage evaluation plan sha256"
            ),
            training_visit_count=_nonnegative_integer(
                value["training_visit_count"], "coverage training visit count"
            ),
            training_visits_sha256=_sha256(
                value["training_visits_sha256"], "coverage training visits sha256"
            ),
            evaluation_item_count=_nonnegative_integer(
                value["evaluation_item_count"], "coverage evaluation item count"
            ),
            records=tuple(DenseEvidenceCoverageRecord.from_dict(item) for item in raw_records),
            evaluation_record_count=(
                _nonnegative_integer(
                    value["evaluation_record_count"],
                    "coverage evaluation record count",
                )
                if schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA
                else None
            ),
            evaluation_history_transition_count=(
                _nonnegative_integer(
                    value["evaluation_history_transition_count"],
                    "coverage evaluation history transition count",
                )
                if schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA
                else 0
            ),
            evaluation_history_visit_count=(
                _nonnegative_integer(
                    value["evaluation_history_visit_count"],
                    "coverage evaluation history visit count",
                )
                if schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA
                else 0
            ),
            evaluation_history_visits_sha256=(
                _sha256(
                    value["evaluation_history_visits_sha256"],
                    "coverage evaluation history visits sha256",
                )
                if schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA
                else None
            ),
            modalities=tuple(_text(item, "coverage modality") for item in raw_modalities),
        )
        if plan.records_sha256 != _sha256(value["records_sha256"], "coverage records sha256"):
            raise ContractError("dense evidence coverage record digest changed")
        if plan.artifact_sha256 != _sha256(value["artifact_sha256"], "coverage artifact sha256"):
            raise ContractError("dense evidence coverage artifact SHA-256 changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> DenseEvidenceCoveragePlan:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractError(f"invalid dense evidence coverage plan: {source}") from error
        return cls.from_dict(payload)


def build_calvin_dense_evidence_coverage_plan(
    *,
    stream_plan: FrozenEpisodeStreamPlan,
    representation_split: RepresentationTrialSplit,
    evaluation_plan: EntityEvaluationPlan,
    physical_dataset: CalvinPhysicalTransitionDataset,
    evaluation_dataset: CalvinStatefulTransitionDataset,
    training_step_prefix: int | None = None,
    evaluation_history_transitions: int = 0,
    schema: str = DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA,
) -> DenseEvidenceCoveragePlan:
    """Project a frozen protocol prefix and its evaluation onto physical events."""

    if not isinstance(stream_plan, FrozenEpisodeStreamPlan):
        raise TypeError("dense evidence coverage requires a frozen episode stream")
    if not isinstance(representation_split, RepresentationTrialSplit):
        raise TypeError("dense evidence coverage requires a representation split")
    if not isinstance(evaluation_plan, EntityEvaluationPlan):
        raise TypeError("dense evidence coverage requires an entity evaluation plan")
    manifest = physical_dataset.index.dataset_manifest
    evaluation_manifest = evaluation_dataset.index.dataset_manifest
    if manifest is None or evaluation_manifest is None or manifest != evaluation_manifest:
        raise ContractError("dense evidence datasets lack one identical file manifest")
    identity = (manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256)
    if identity != (
        representation_split.dataset_id,
        representation_split.dataset_revision,
        representation_split.dataset_manifest_sha256,
    ):
        raise ContractError("dense evidence split belongs to another dataset")
    if stream_plan.plan_sha256 != representation_split.stream_plan_sha256:
        raise ContractError("dense evidence stream and representation split differ")
    if evaluation_plan.representation_split_sha256 != representation_split.artifact_sha256:
        raise ContractError("dense evidence evaluation and representation split differ")
    if stream_plan.total_steps != representation_split.training_steps:
        raise ContractError("dense evidence stream does not cover the split training budget")
    if training_step_prefix is None:
        training_step_prefix = stream_plan.total_steps
    if (
        isinstance(training_step_prefix, bool)
        or not isinstance(training_step_prefix, int)
        or not 0 < training_step_prefix <= stream_plan.total_steps
    ):
        raise ContractError("dense evidence training-step prefix must lie inside the frozen stream")
    if (
        isinstance(evaluation_history_transitions, bool)
        or not isinstance(evaluation_history_transitions, int)
        or evaluation_history_transitions < 0
    ):
        raise ContractError("dense evidence evaluation history must be non-negative")
    if schema not in {
        DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1,
        DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA,
    }:
        raise ContractError("dense evidence coverage plan schema changed")
    if schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1 and (
        evaluation_history_transitions != 0
    ):
        raise ContractError("v1 coverage cannot encode evaluation history")
    verify_representation_trial_split_training_evidence(
        representation_split,
        stream_plan,
        physical_dataset,
    )
    if (
        build_entity_evaluation_plan(
            representation_split,
            evaluation_dataset,
            world_size=evaluation_plan.world_size,
        )
        != evaluation_plan
    ):
        raise ContractError("dense evidence evaluation plan is not source reproducible")

    training_visits: list[tuple[int, str]] = []
    training_records: dict[int, str] = {}
    for optimizer_step in range(training_step_prefix):
        for transition in stream_plan.global_batch(optimizer_step).transitions:
            sample_key = transition.sample.sample_key
            source_global_index = physical_dataset.source_global_index_by_key(sample_key)
            previous = training_records.setdefault(source_global_index, sample_key)
            if previous != sample_key:
                raise ContractError("one training source index maps to multiple physical keys")
            training_visits.append((source_global_index, sample_key))

    canonical_key_by_source = {
        physical_dataset.source_global_index_by_key(sample_key): sample_key
        for sample_key in physical_dataset.sample_keys
    }
    evaluation_records: dict[int, str] = {}
    evaluation_history_visits: list[tuple[str, int, int, str]] = []
    for item in evaluation_plan.items:
        try:
            sample_key = canonical_key_by_source[item.source_global_index]
        except KeyError as error:
            raise ContractError("evaluation item has no canonical physical event") from error
        previous = evaluation_records.setdefault(item.source_global_index, sample_key)
        if previous != sample_key:
            raise ContractError("one evaluation source index maps to multiple physical keys")
        if evaluation_history_transitions == 0:
            continue
        if item.transition_index < evaluation_history_transitions:
            continue
        history_keys = evaluation_dataset.history_sample_keys(item.sample_key)[
            -evaluation_history_transitions:
        ]
        if len(history_keys) != evaluation_history_transitions:
            raise ContractError("evaluation history crosses a language reset")
        history_source_indices = tuple(
            evaluation_dataset.source_global_index_by_key(history_key)
            for history_key in history_keys
        )
        if history_source_indices != tuple(
            range(
                item.source_global_index - evaluation_history_transitions,
                item.source_global_index,
            )
        ):
            raise ContractError("evaluation history is not source consecutive")
        for history_source_index in history_source_indices:
            try:
                history_sample_key = canonical_key_by_source[history_source_index]
            except KeyError as error:
                raise ContractError("evaluation history has no canonical physical event") from error
            previous = evaluation_records.setdefault(
                history_source_index,
                history_sample_key,
            )
            if previous != history_sample_key:
                raise ContractError("one evaluation history index maps to multiple physical keys")
            evaluation_history_visits.append(
                (
                    item.partition,
                    item.ordinal,
                    history_source_index,
                    history_sample_key,
                )
            )
    if set(training_records) & set(evaluation_records):
        raise ContractError("dense evidence training and evaluation coverage overlap")

    records = tuple(
        sorted(
            (
                *(
                    DenseEvidenceCoverageRecord(index, key, "training")
                    for index, key in training_records.items()
                ),
                *(
                    DenseEvidenceCoverageRecord(index, key, "evaluation")
                    for index, key in evaluation_records.items()
                ),
            ),
            key=lambda record: (record.source_global_index, record.sample_key),
        )
    )
    return DenseEvidenceCoveragePlan(
        schema=schema,
        dataset_id=identity[0],
        dataset_revision=identity[1],
        dataset_tree_sha256=identity[2],
        stream_plan_sha256=stream_plan.plan_sha256,
        representation_split_sha256=representation_split.artifact_sha256,
        evaluation_plan_sha256=evaluation_plan.artifact_sha256,
        training_visit_count=len(training_visits),
        training_visits_sha256=canonical_payload_sha256(
            "picf-next.dense-evidence-training-visits/v1",
            training_visits,
        ),
        evaluation_item_count=len(evaluation_plan.items),
        records=records,
        evaluation_record_count=len(evaluation_records),
        evaluation_history_transition_count=evaluation_history_transitions,
        evaluation_history_visit_count=len(evaluation_history_visits),
        evaluation_history_visits_sha256=canonical_payload_sha256(
            "picf-next.dense-evidence-evaluation-history-visits/v1",
            evaluation_history_visits,
        ),
    )
