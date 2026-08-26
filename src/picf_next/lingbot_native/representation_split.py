"""Source-disjoint data contract for bounded LingBot representation trials."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from picf_next.data.calvin import (
    CalvinLanguageSegment,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.lingbot_native.calvin import (
    build_native_calvin_episode_domain,
    build_native_calvin_physical_episode_domain,
    build_native_calvin_physical_sample_domain,
    native_calvin_sample_plan_instance_id,
)
from picf_next.training.control import (
    FrozenEpisodeStreamPlan,
    FrozenResetMixtureStreamPlan,
    FrozenSamplePlan,
    TrainingPlan,
)

REPRESENTATION_TRIAL_SPLIT_SCHEMA = "picf-next.lingbot-representation-trial-split.v1"
REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA = (
    "picf-next.lingbot-representation-trial-split.reference-evaluation.v2"
)
_EVALUATION_PARTITIONS = ("validation", "heldout")
_SEGMENT_FIELDS = {
    "segment_index",
    "source_end",
    "source_episode_index",
    "source_start",
    "task_key",
}
_SPLIT_FIELDS = {
    "artifact_sha256",
    "comparison_id",
    "dataset_id",
    "dataset_manifest_sha256",
    "dataset_revision",
    "heldout_segments",
    "partition_seed",
    "schema",
    "segments_per_task",
    "stream_plan_sha256",
    "training_sample_count",
    "training_sample_keys_sha256",
    "training_segment_indices",
    "training_source_episode_indices",
    "training_source_global_indices_sha256",
    "training_steps",
    "validation_segments",
}
_REFERENCE_SPLIT_FIELDS = _SPLIT_FIELDS | {
    "evaluation_reference_split_artifact_sha256",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    result = tuple(_require_nonnegative_int(item, name) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must contain unique sorted integers")
    return result


@dataclass(frozen=True, slots=True)
class RepresentationEvaluationSegment:
    """One immutable language segment selected without reading target rasters."""

    task_key: str
    segment_index: int
    source_episode_index: int
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        _require_text(self.task_key, "evaluation task key")
        _require_nonnegative_int(self.segment_index, "evaluation segment index")
        _require_nonnegative_int(
            self.source_episode_index,
            "evaluation source episode index",
        )
        _require_nonnegative_int(self.source_start, "evaluation source start")
        _require_nonnegative_int(self.source_end, "evaluation source end")
        if self.source_end <= self.source_start:
            raise ValueError("evaluation segment must contain at least one transition")

    def as_dict(self) -> dict[str, object]:
        return {
            "task_key": self.task_key,
            "segment_index": self.segment_index,
            "source_episode_index": self.source_episode_index,
            "source_start": self.source_start,
            "source_end": self.source_end,
        }

    @classmethod
    def from_dict(cls, value: object) -> RepresentationEvaluationSegment:
        if not isinstance(value, Mapping) or set(value) != _SEGMENT_FIELDS:
            raise ValueError("representation evaluation segment fields differ from schema")
        return cls(
            task_key=_require_text(value["task_key"], "evaluation task key"),
            segment_index=_require_nonnegative_int(
                value["segment_index"],
                "evaluation segment index",
            ),
            source_episode_index=_require_nonnegative_int(
                value["source_episode_index"],
                "evaluation source episode index",
            ),
            source_start=_require_nonnegative_int(
                value["source_start"],
                "evaluation source start",
            ),
            source_end=_require_nonnegative_int(
                value["source_end"],
                "evaluation source end",
            ),
        )


def _validate_evaluation_partition(
    name: str,
    segments: tuple[RepresentationEvaluationSegment, ...],
    *,
    segments_per_task: int,
) -> frozenset[int]:
    if not segments:
        raise ValueError(f"{name} representation evaluation partition is empty")
    if segments != tuple(sorted(segments, key=lambda item: (item.task_key, item.segment_index))):
        raise ValueError(f"{name} representation evaluation segments must be sorted")
    segment_indices = [item.segment_index for item in segments]
    if len(set(segment_indices)) != len(segment_indices):
        raise ValueError(f"{name} representation evaluation segments are not unique")
    task_counts = Counter(item.task_key for item in segments)
    if any(count != segments_per_task for count in task_counts.values()):
        raise ValueError(f"{name} representation task cardinality changed")
    for task_key in task_counts:
        task_source_episodes = {
            item.source_episode_index for item in segments if item.task_key == task_key
        }
        if len(task_source_episodes) != segments_per_task:
            raise ValueError(f"{name} representation task {task_key!r} reused a source episode")
    return frozenset(item.source_episode_index for item in segments)


@dataclass(frozen=True, slots=True)
class RepresentationTrialSplit:
    """Content-addressed train/validation/held-out source isolation evidence."""

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    comparison_id: str
    stream_plan_sha256: str
    partition_seed: int
    training_steps: int
    training_sample_count: int
    training_sample_keys_sha256: str
    training_source_global_indices_sha256: str
    training_segment_indices: tuple[int, ...]
    training_source_episode_indices: tuple[int, ...]
    segments_per_task: int
    validation_segments: tuple[RepresentationEvaluationSegment, ...]
    heldout_segments: tuple[RepresentationEvaluationSegment, ...]
    schema: str = REPRESENTATION_TRIAL_SPLIT_SCHEMA
    evaluation_reference_split_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema not in {
            REPRESENTATION_TRIAL_SPLIT_SCHEMA,
            REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
        }:
            raise ValueError("representation trial split schema changed")
        if self.schema == REPRESENTATION_TRIAL_SPLIT_SCHEMA:
            if self.evaluation_reference_split_artifact_sha256 is not None:
                raise ValueError("v1 representation split cannot reference another evaluation bank")
        else:
            _require_sha256(
                self.evaluation_reference_split_artifact_sha256,
                "representation evaluation reference split artifact sha256",
            )
        _require_text(self.dataset_id, "representation dataset id")
        _require_text(self.dataset_revision, "representation dataset revision")
        _require_sha256(
            self.dataset_manifest_sha256,
            "representation dataset manifest sha256",
        )
        _require_text(self.comparison_id, "representation comparison id")
        _require_sha256(self.stream_plan_sha256, "representation stream plan sha256")
        _require_nonnegative_int(self.partition_seed, "representation partition seed")
        _require_positive_int(self.training_steps, "representation training steps")
        _require_positive_int(
            self.training_sample_count,
            "representation training sample count",
        )
        _require_sha256(
            self.training_sample_keys_sha256,
            "representation training sample-key sha256",
        )
        _require_sha256(
            self.training_source_global_indices_sha256,
            "representation training source-index sha256",
        )
        _require_positive_int(self.segments_per_task, "representation segments per task")
        for name, values in (
            ("training segment indices", self.training_segment_indices),
            ("training source episode indices", self.training_source_episode_indices),
        ):
            if not values or values != tuple(sorted(set(values))):
                raise ValueError(f"representation {name} must be nonempty, unique, and sorted")
            for value in values:
                _require_nonnegative_int(value, f"representation {name}")
        validation_sources = _validate_evaluation_partition(
            "validation",
            self.validation_segments,
            segments_per_task=self.segments_per_task,
        )
        heldout_sources = _validate_evaluation_partition(
            "heldout",
            self.heldout_segments,
            segments_per_task=self.segments_per_task,
        )
        training_sources = frozenset(self.training_source_episode_indices)
        if training_sources & validation_sources:
            raise ValueError("representation validation reused a training source episode")
        if training_sources & heldout_sources:
            raise ValueError("representation held-out reused a training source episode")
        if validation_sources & heldout_sources:
            raise ValueError("representation validation and held-out source episodes overlap")
        validation_tasks = {item.task_key for item in self.validation_segments}
        heldout_tasks = {item.task_key for item in self.heldout_segments}
        if validation_tasks != heldout_tasks:
            raise ValueError("representation validation and held-out task coverage differs")

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "comparison_id": self.comparison_id,
            "stream_plan_sha256": self.stream_plan_sha256,
            "partition_seed": self.partition_seed,
            "training_steps": self.training_steps,
            "training_sample_count": self.training_sample_count,
            "training_sample_keys_sha256": self.training_sample_keys_sha256,
            "training_source_global_indices_sha256": (self.training_source_global_indices_sha256),
            "training_segment_indices": list(self.training_segment_indices),
            "training_source_episode_indices": list(self.training_source_episode_indices),
            "segments_per_task": self.segments_per_task,
            "validation_segments": [item.as_dict() for item in self.validation_segments],
            "heldout_segments": [item.as_dict() for item in self.heldout_segments],
        }
        if self.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA:
            payload["evaluation_reference_split_artifact_sha256"] = (
                self.evaluation_reference_split_artifact_sha256
            )
        return payload

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self._payload())

    @property
    def evaluation_source_episode_indices(self) -> tuple[int, ...]:
        """Source-level holdout domain shared by planning, caches, and training."""

        return tuple(
            sorted(
                {
                    item.source_episode_index
                    for item in (*self.validation_segments, *self.heldout_segments)
                }
            )
        )

    @property
    def stream_domain_excluded_source_episode_indices(self) -> tuple[int, ...]:
        """Episodes removed before constructing the bound stream domain.

        A v1 split selects evaluation episodes only after freezing its stream, so
        those dormant episodes remain part of the stream's hashed episode domain.
        A reference-evaluation v2 split does the opposite: it preserves an older
        evaluation bank and excludes that bank before constructing the candidate
        stream. Replaying both schemas with one exclusion rule makes v1 plans
        impossible to reconstruct and weakens the scientific identity boundary.
        """

        if self.schema == REPRESENTATION_TRIAL_SPLIT_SCHEMA:
            return ()
        return self.evaluation_source_episode_indices

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(cls, value: object) -> RepresentationTrialSplit:
        if not isinstance(value, Mapping):
            raise ValueError("representation trial split fields differ from schema")
        schema = value.get("schema")
        expected_fields = (
            _SPLIT_FIELDS
            if schema == REPRESENTATION_TRIAL_SPLIT_SCHEMA
            else _REFERENCE_SPLIT_FIELDS
            if schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
            else None
        )
        if expected_fields is None or set(value) != expected_fields:
            raise ValueError("representation trial split fields differ from schema")
        split = cls(
            schema=_require_text(schema, "representation split schema"),
            dataset_id=_require_text(value["dataset_id"], "representation dataset id"),
            dataset_revision=_require_text(
                value["dataset_revision"],
                "representation dataset revision",
            ),
            dataset_manifest_sha256=_require_sha256(
                value["dataset_manifest_sha256"],
                "representation dataset manifest sha256",
            ),
            comparison_id=_require_text(
                value["comparison_id"],
                "representation comparison id",
            ),
            stream_plan_sha256=_require_sha256(
                value["stream_plan_sha256"],
                "representation stream plan sha256",
            ),
            partition_seed=_require_nonnegative_int(
                value["partition_seed"],
                "representation partition seed",
            ),
            training_steps=_require_positive_int(
                value["training_steps"],
                "representation training steps",
            ),
            training_sample_count=_require_positive_int(
                value["training_sample_count"],
                "representation training sample count",
            ),
            training_sample_keys_sha256=_require_sha256(
                value["training_sample_keys_sha256"],
                "representation training sample-key sha256",
            ),
            training_source_global_indices_sha256=_require_sha256(
                value["training_source_global_indices_sha256"],
                "representation training source-index sha256",
            ),
            training_segment_indices=_integer_tuple(
                value["training_segment_indices"],
                "representation training segment indices",
            ),
            training_source_episode_indices=_integer_tuple(
                value["training_source_episode_indices"],
                "representation training source episode indices",
            ),
            segments_per_task=_require_positive_int(
                value["segments_per_task"],
                "representation segments per task",
            ),
            validation_segments=_segment_tuple(
                value["validation_segments"],
                "validation",
            ),
            heldout_segments=_segment_tuple(value["heldout_segments"], "heldout"),
            evaluation_reference_split_artifact_sha256=(
                None
                if schema == REPRESENTATION_TRIAL_SPLIT_SCHEMA
                else _require_sha256(
                    value["evaluation_reference_split_artifact_sha256"],
                    "representation evaluation reference split artifact sha256",
                )
            ),
        )
        expected = _require_sha256(
            value["artifact_sha256"],
            "representation split artifact sha256",
        )
        if split.artifact_sha256 != expected:
            raise ValueError("representation trial split artifact SHA-256 changed")
        return split

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{self.artifact_sha256[:12]}"
        )
        temporary.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> RepresentationTrialSplit:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid representation trial split: {source}") from error
        return cls.from_dict(value)


def _segment_tuple(value: object, name: str) -> tuple[RepresentationEvaluationSegment, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} representation segments must be a JSON list")
    return tuple(RepresentationEvaluationSegment.from_dict(item) for item in value)


def _episode_partition(
    *,
    dataset_manifest_sha256: str,
    stream_plan_sha256: str,
    partition_seed: int,
    source_episode_index: int,
) -> str:
    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "schema": REPRESENTATION_TRIAL_SPLIT_SCHEMA,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "stream_plan_sha256": stream_plan_sha256,
                "partition_seed": partition_seed,
                "source_episode_index": source_episode_index,
            }
        )
    ).digest()
    return _EVALUATION_PARTITIONS[digest[0] & 1]


def _candidate_digest(
    *,
    partition: str,
    partition_seed: int,
    task_key: str,
    source_episode_index: int,
    segment_index: int,
) -> bytes:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "schema": REPRESENTATION_TRIAL_SPLIT_SCHEMA,
                "partition": partition,
                "partition_seed": partition_seed,
                "task_key": task_key,
                "source_episode_index": source_episode_index,
                "segment_index": segment_index,
            }
        )
    ).digest()


def _select_evaluation_segments(
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
    *,
    partition: str,
    partition_seed: int,
    stream_plan_sha256: str,
    training_source_episodes: frozenset[int],
    segments_per_task: int,
) -> tuple[RepresentationEvaluationSegment, ...]:
    index = dataset.index
    manifest = index.dataset_manifest
    if manifest is None:
        raise ValueError("representation evaluation selection requires a dataset manifest")
    candidates_by_task: dict[str, list[CalvinLanguageSegment]] = {}
    for segment in index.segments:
        source_episode_index = int(segment.episode_index)
        if source_episode_index in training_source_episodes:
            continue
        assigned = _episode_partition(
            dataset_manifest_sha256=manifest.tree_sha256,
            stream_plan_sha256=stream_plan_sha256,
            partition_seed=partition_seed,
            source_episode_index=source_episode_index,
        )
        if assigned == partition:
            candidates_by_task.setdefault(segment.task_key, []).append(segment)
    all_tasks = tuple(sorted({segment.task_key for segment in index.segments}))
    selected: list[RepresentationEvaluationSegment] = []
    for task_key in all_tasks:
        candidates = candidates_by_task.get(task_key, [])
        candidates.sort(
            key=lambda segment: (
                _candidate_digest(
                    partition=partition,
                    partition_seed=partition_seed,
                    task_key=task_key,
                    source_episode_index=int(segment.episode_index),
                    segment_index=int(segment.index),
                ),
                int(segment.index),
            )
        )
        selected_source_episodes: set[int] = set()
        for segment in candidates:
            source_episode_index = int(segment.episode_index)
            if source_episode_index in selected_source_episodes:
                continue
            selected.append(
                RepresentationEvaluationSegment(
                    task_key=segment.task_key,
                    segment_index=int(segment.index),
                    source_episode_index=source_episode_index,
                    source_start=int(segment.start),
                    source_end=int(segment.end),
                )
            )
            selected_source_episodes.add(source_episode_index)
            if len(selected_source_episodes) == segments_per_task:
                break
        if len(selected_source_episodes) != segments_per_task:
            raise ValueError(
                f"{partition} source partition has only "
                f"{len(selected_source_episodes)} independent episodes for task "
                f"{task_key!r}; required {segments_per_task}"
            )
    return tuple(sorted(selected, key=lambda item: (item.task_key, item.segment_index)))


@dataclass(frozen=True, slots=True)
class _RepresentationTrainingEvidence:
    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    comparison_id: str
    stream_plan_sha256: str
    training_steps: int
    training_sample_count: int
    training_sample_keys_sha256: str
    training_source_global_indices_sha256: str
    training_segment_indices: tuple[int, ...]
    training_source_episode_indices: tuple[int, ...]


def _build_representation_training_evidence(
    stream_plan: TrainingPlan,
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
    *,
    training_steps: int,
) -> _RepresentationTrainingEvidence:
    if not isinstance(
        stream_plan,
        (FrozenSamplePlan, FrozenEpisodeStreamPlan, FrozenResetMixtureStreamPlan),
    ):
        raise TypeError("representation split requires one supported frozen training plan")
    if not isinstance(dataset, CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset):
        raise TypeError("representation split requires a typed CALVIN dataset")
    training_steps = _require_positive_int(training_steps, "representation training steps")
    if training_steps > stream_plan.total_steps:
        raise ValueError("representation training prefix exceeds the frozen stream plan")
    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("representation split requires a content-addressed dataset")
    plan_identity = (
        stream_plan.dataset_id,
        stream_plan.dataset_revision,
        stream_plan.dataset_manifest_sha256,
    )
    dataset_identity = (
        dataset.index.dataset_id,
        dataset.index.dataset_revision,
        manifest.tree_sha256,
    )
    if plan_identity != dataset_identity:
        raise ValueError("representation stream plan and dataset identities differ")
    if isinstance(stream_plan, FrozenSamplePlan):
        dataset_sample_keys = frozenset(dataset.sample_keys)
        if any(sample_key not in dataset_sample_keys for sample_key in stream_plan.sample_keys):
            raise ValueError("representation sample plan contains a sample absent from the dataset")
    else:
        dataset_episodes = {episode.episode_key: episode for episode in dataset.episode_manifest}
        for episode in stream_plan.episodes:
            observed = dataset_episodes.get(episode.episode_key)
            if observed is None or observed.sample_keys != episode.sample_keys:
                raise ValueError("representation stream plan episode manifest differs from dataset")

    training_sample_keys: list[str] = []
    training_source_global_indices: list[int] = []
    training_segment_indices: set[int] = set()
    training_source_episode_indices: set[int] = set()
    for optimizer_step in range(training_steps):
        global_batch = stream_plan.global_batch(optimizer_step)
        occurrences = (
            tuple(
                (
                    sample,
                    native_calvin_sample_plan_instance_id(
                        optimizer_step=optimizer_step,
                        sample=sample,
                    ),
                )
                for sample in global_batch.samples
            )
            if isinstance(stream_plan, FrozenSamplePlan)
            else tuple(
                (transition.sample, transition.episode_instance_id)
                for transition in global_batch.transitions
            )
        )
        for planned_sample, episode_instance_id in occurrences:
            sample_key = planned_sample.sample_key
            training_sample_keys.append(sample_key)
            if isinstance(dataset, CalvinPhysicalTransitionDataset):
                from picf_next.lingbot_native.calvin import (
                    select_native_calvin_physical_prompt_segment,
                )

                event = dataset.event_by_key(sample_key)
                selected_segment, _receipt = select_native_calvin_physical_prompt_segment(
                    dataset,
                    sample_key=sample_key,
                    plan_sha256=stream_plan.plan_sha256,
                    episode_instance_id=episode_instance_id,
                )
                training_source_global_indices.append(event.global_index)
                training_segment_indices.add(selected_segment)
                training_source_episode_indices.add(event.episode.index)
            else:
                locator = dataset.locator_by_key(sample_key)
                segment = dataset.index.segments[locator.segment_index]
                training_source_global_indices.append(locator.global_index)
                training_segment_indices.add(locator.segment_index)
                training_source_episode_indices.add(int(segment.episode_index))
    return _RepresentationTrainingEvidence(
        dataset_id=stream_plan.dataset_id,
        dataset_revision=stream_plan.dataset_revision,
        dataset_manifest_sha256=stream_plan.dataset_manifest_sha256,
        comparison_id=stream_plan.comparison_id,
        stream_plan_sha256=stream_plan.plan_sha256,
        training_steps=training_steps,
        training_sample_count=len(training_sample_keys),
        training_sample_keys_sha256=_sha256(training_sample_keys),
        training_source_global_indices_sha256=_sha256(training_source_global_indices),
        training_segment_indices=tuple(sorted(training_segment_indices)),
        training_source_episode_indices=tuple(sorted(training_source_episode_indices)),
    )


def verify_representation_trial_split_training_evidence(
    split: RepresentationTrialSplit,
    stream_plan: TrainingPlan,
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
) -> RepresentationTrialSplit:
    """Rebuild the train partition from the bound plan instead of trusting JSON claims."""

    if not isinstance(split, RepresentationTrialSplit):
        raise TypeError("representation split verification requires a typed split")
    evidence = _build_representation_training_evidence(
        stream_plan,
        dataset,
        training_steps=split.training_steps,
    )
    expected = {
        "dataset_id": evidence.dataset_id,
        "dataset_revision": evidence.dataset_revision,
        "dataset_manifest_sha256": evidence.dataset_manifest_sha256,
        "comparison_id": evidence.comparison_id,
        "stream_plan_sha256": evidence.stream_plan_sha256,
        "training_steps": evidence.training_steps,
        "training_sample_count": evidence.training_sample_count,
        "training_sample_keys_sha256": evidence.training_sample_keys_sha256,
        "training_source_global_indices_sha256": (evidence.training_source_global_indices_sha256),
        "training_segment_indices": evidence.training_segment_indices,
        "training_source_episode_indices": evidence.training_source_episode_indices,
    }
    mismatches = tuple(name for name, value in expected.items() if getattr(split, name) != value)
    if mismatches:
        raise ValueError(
            f"representation split training evidence differs from the frozen plan: {mismatches}"
        )
    return split


def build_representation_trial_split(
    stream_plan: TrainingPlan,
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
    *,
    training_steps: int,
    partition_seed: int,
    segments_per_task: int = 2,
) -> RepresentationTrialSplit:
    """Freeze a target-independent split around an exact bounded training prefix."""

    partition_seed = _require_nonnegative_int(
        partition_seed,
        "representation partition seed",
    )
    segments_per_task = _require_positive_int(
        segments_per_task,
        "representation segments per task",
    )
    evidence = _build_representation_training_evidence(
        stream_plan,
        dataset,
        training_steps=training_steps,
    )
    training_sources = frozenset(evidence.training_source_episode_indices)

    validation = _select_evaluation_segments(
        dataset,
        partition="validation",
        partition_seed=partition_seed,
        stream_plan_sha256=stream_plan.plan_sha256,
        training_source_episodes=training_sources,
        segments_per_task=segments_per_task,
    )
    heldout = _select_evaluation_segments(
        dataset,
        partition="heldout",
        partition_seed=partition_seed,
        stream_plan_sha256=stream_plan.plan_sha256,
        training_source_episodes=training_sources,
        segments_per_task=segments_per_task,
    )
    return RepresentationTrialSplit(
        dataset_id=evidence.dataset_id,
        dataset_revision=evidence.dataset_revision,
        dataset_manifest_sha256=evidence.dataset_manifest_sha256,
        comparison_id=evidence.comparison_id,
        stream_plan_sha256=evidence.stream_plan_sha256,
        partition_seed=partition_seed,
        training_steps=evidence.training_steps,
        training_sample_count=evidence.training_sample_count,
        training_sample_keys_sha256=evidence.training_sample_keys_sha256,
        training_source_global_indices_sha256=evidence.training_source_global_indices_sha256,
        training_segment_indices=evidence.training_segment_indices,
        training_source_episode_indices=evidence.training_source_episode_indices,
        segments_per_task=segments_per_task,
        validation_segments=validation,
        heldout_segments=heldout,
    )


def build_representation_trial_split_with_reference_evaluation(
    stream_plan: TrainingPlan,
    dataset: CalvinStatefulTransitionDataset | CalvinPhysicalTransitionDataset,
    *,
    training_steps: int,
    evaluation_reference: RepresentationTrialSplit,
    require_equal_training_budget: bool = True,
) -> RepresentationTrialSplit:
    """Bind a new training stream to one exact, source-disjoint evaluation bank."""

    if not isinstance(evaluation_reference, RepresentationTrialSplit):
        raise TypeError("representation evaluation reference must be a trial split")
    if not isinstance(require_equal_training_budget, bool):
        raise TypeError("reference training-budget equality control must be boolean")
    evidence = _build_representation_training_evidence(
        stream_plan,
        dataset,
        training_steps=training_steps,
    )
    reference_identity = (
        evaluation_reference.dataset_id,
        evaluation_reference.dataset_revision,
        evaluation_reference.dataset_manifest_sha256,
        evaluation_reference.comparison_id,
    )
    candidate_identity = (
        evidence.dataset_id,
        evidence.dataset_revision,
        evidence.dataset_manifest_sha256,
        evidence.comparison_id,
    )
    if reference_identity != candidate_identity:
        raise ValueError("representation evaluation reference belongs to another experiment")
    if (
        require_equal_training_budget
        and evidence.training_steps != evaluation_reference.training_steps
    ):
        raise ValueError("referenced representation trials must use the same training budget")

    excluded_sources = evaluation_reference.evaluation_source_episode_indices
    if isinstance(stream_plan, FrozenSamplePlan):
        if isinstance(dataset, CalvinPhysicalTransitionDataset):
            expected_sample_keys = build_native_calvin_physical_sample_domain(
                dataset,
                excluded_source_episode_indices=excluded_sources,
            )
        else:
            excluded_set = frozenset(excluded_sources)
            expected_sample_keys = tuple(
                sample_key
                for sample_key in dataset.sample_keys
                if int(
                    dataset.index.segments[
                        dataset.locator_by_key(sample_key).segment_index
                    ].episode_index
                )
                not in excluded_set
            )
        if stream_plan.sample_keys != expected_sample_keys:
            raise ValueError(
                "referenced evaluation sample domain must exactly exclude the evaluation episodes"
            )
    else:
        expected_episodes = (
            build_native_calvin_physical_episode_domain(
                dataset,
                excluded_source_episode_indices=excluded_sources,
            )
            if isinstance(dataset, CalvinPhysicalTransitionDataset)
            else build_native_calvin_episode_domain(
                dataset,
                excluded_source_episode_indices=excluded_sources,
            )
        )
        if stream_plan.episodes != expected_episodes:
            raise ValueError(
                "referenced evaluation stream domain must exactly exclude the evaluation episodes"
            )

    segments_by_index = {int(segment.index): segment for segment in dataset.index.segments}
    for item in (
        *evaluation_reference.validation_segments,
        *evaluation_reference.heldout_segments,
    ):
        observed = segments_by_index.get(item.segment_index)
        if observed is None or (
            observed.task_key,
            int(observed.episode_index),
            int(observed.start),
            int(observed.end),
        ) != (
            item.task_key,
            item.source_episode_index,
            item.source_start,
            item.source_end,
        ):
            raise ValueError("representation evaluation reference differs from source metadata")

    evaluation_sources = {
        item.source_episode_index
        for item in (
            *evaluation_reference.validation_segments,
            *evaluation_reference.heldout_segments,
        )
    }
    if set(evidence.training_source_episode_indices) & evaluation_sources:
        raise ValueError("referenced evaluation episodes overlap the candidate training stream")
    if (
        require_equal_training_budget
        and evidence.training_sample_count != evaluation_reference.training_sample_count
    ):
        raise ValueError("referenced representation trials must use the same sample budget")
    return RepresentationTrialSplit(
        dataset_id=evidence.dataset_id,
        dataset_revision=evidence.dataset_revision,
        dataset_manifest_sha256=evidence.dataset_manifest_sha256,
        comparison_id=evidence.comparison_id,
        stream_plan_sha256=evidence.stream_plan_sha256,
        partition_seed=evaluation_reference.partition_seed,
        training_steps=evidence.training_steps,
        training_sample_count=evidence.training_sample_count,
        training_sample_keys_sha256=evidence.training_sample_keys_sha256,
        training_source_global_indices_sha256=(evidence.training_source_global_indices_sha256),
        training_segment_indices=evidence.training_segment_indices,
        training_source_episode_indices=evidence.training_source_episode_indices,
        segments_per_task=evaluation_reference.segments_per_task,
        validation_segments=evaluation_reference.validation_segments,
        heldout_segments=evaluation_reference.heldout_segments,
        schema=REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
        evaluation_reference_split_artifact_sha256=(evaluation_reference.artifact_sha256),
    )
