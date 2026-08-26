"""Content-addressed, task-independent entity evaluation frames."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from picf_next.data.calvin import CalvinStatefulTransitionDataset
from picf_next.lingbot_native.representation_split import (
    RepresentationEvaluationSegment,
    RepresentationTrialSplit,
)

ENTITY_EVALUATION_PLAN_SCHEMA = "picf-next.entity-evaluation-plan.v2"
ENTITY_EVALUATION_PARTITIONS = ("validation", "heldout")
ENTITY_EVALUATION_WORLD_SIZE = 2
ENTITY_EVALUATION_WORLD_SIZES = (2, 4)

_ITEM_FIELDS = frozenset(
    {
        "partition",
        "ordinal",
        "rank",
        "task_key",
        "segment_index",
        "source_episode_index",
        "source_global_index",
        "transition_index",
        "sample_key",
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


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class EntityEvaluationItem:
    """One source-selected frame; ``task_key`` is audit metadata only."""

    partition: str
    ordinal: int
    rank: int
    task_key: str
    segment_index: int
    source_episode_index: int
    source_global_index: int
    transition_index: int
    sample_key: str

    def __post_init__(self) -> None:
        if self.partition not in ENTITY_EVALUATION_PARTITIONS:
            raise ValueError("entity evaluation partition is unsupported")
        for name, value in (
            ("ordinal", self.ordinal),
            ("rank", self.rank),
            ("segment index", self.segment_index),
            ("source episode index", self.source_episode_index),
            ("source global index", self.source_global_index),
            ("transition index", self.transition_index),
        ):
            _nonnegative_int(value, name=f"entity evaluation {name}")
        if self.rank >= max(ENTITY_EVALUATION_WORLD_SIZES):
            raise ValueError("entity evaluation rank is outside a supported topology")
        _text(self.task_key, name="entity evaluation task key")
        _text(self.sample_key, name="entity evaluation sample key")

    def as_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "ordinal": self.ordinal,
            "rank": self.rank,
            "task_key": self.task_key,
            "segment_index": self.segment_index,
            "source_episode_index": self.source_episode_index,
            "source_global_index": self.source_global_index,
            "transition_index": self.transition_index,
            "sample_key": self.sample_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> EntityEvaluationItem:
        if not isinstance(value, Mapping) or set(value) != _ITEM_FIELDS:
            raise ValueError("entity evaluation item fields differ from schema")
        return cls(
            partition=_text(value["partition"], name="entity evaluation partition"),
            ordinal=_nonnegative_int(value["ordinal"], name="entity evaluation ordinal"),
            rank=_nonnegative_int(value["rank"], name="entity evaluation rank"),
            task_key=_text(value["task_key"], name="entity evaluation task key"),
            segment_index=_nonnegative_int(
                value["segment_index"], name="entity evaluation segment index"
            ),
            source_episode_index=_nonnegative_int(
                value["source_episode_index"],
                name="entity evaluation source episode index",
            ),
            source_global_index=_nonnegative_int(
                value["source_global_index"],
                name="entity evaluation source global index",
            ),
            transition_index=_nonnegative_int(
                value["transition_index"],
                name="entity evaluation transition index",
            ),
            sample_key=_text(value["sample_key"], name="entity evaluation sample key"),
        )


@dataclass(frozen=True, slots=True)
class DistributedEntityEvaluationWorkItem:
    """One collective-aligned evaluation forward and its publication status."""

    item: EntityEvaluationItem
    is_padding: bool

    def __post_init__(self) -> None:
        if not isinstance(self.item, EntityEvaluationItem):
            raise TypeError("distributed evaluation work requires an entity item")
        if not isinstance(self.is_padding, bool):
            raise TypeError("distributed evaluation padding flag must be boolean")


@dataclass(frozen=True, slots=True)
class EntityEvaluationPlan:
    """Primary validation frames plus two source-disjoint heldout episodes/task."""

    representation_split_sha256: str
    items: tuple[EntityEvaluationItem, ...]
    world_size: int = ENTITY_EVALUATION_WORLD_SIZE
    schema: str = ENTITY_EVALUATION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENTITY_EVALUATION_PLAN_SCHEMA:
            raise ValueError("entity evaluation plan schema changed")
        _sha256(
            self.representation_split_sha256,
            name="entity evaluation representation split SHA-256",
        )
        if self.world_size not in ENTITY_EVALUATION_WORLD_SIZES:
            raise ValueError("entity evaluation plan requires two or four ranks")
        if not self.items:
            raise ValueError("entity evaluation plan is empty")
        expected_order = tuple(
            sorted(
                self.items,
                key=lambda item: (item.partition, item.task_key, item.segment_index),
            )
        )
        if self.items != expected_order:
            raise ValueError("entity evaluation items must be sorted by partition and task")
        if tuple(item.ordinal for item in self.items) != tuple(range(len(self.items))):
            raise ValueError("entity evaluation ordinals must be contiguous")
        if any(item.rank != item.ordinal % self.world_size for item in self.items):
            raise ValueError("entity evaluation rank assignment changed")
        if len({item.sample_key for item in self.items}) != len(self.items):
            raise ValueError("entity evaluation sample keys must be unique")
        task_counts_by_partition = {
            partition: Counter(
                item.task_key for item in self.items if item.partition == partition
            )
            for partition in ENTITY_EVALUATION_PARTITIONS
        }
        if any(not counts for counts in task_counts_by_partition.values()):
            raise ValueError("entity evaluation partition is empty")
        if set(task_counts_by_partition["validation"]) != set(
            task_counts_by_partition["heldout"]
        ):
            raise ValueError("entity evaluation task coverage differs across partitions")
        if any(count != 1 for count in task_counts_by_partition["validation"].values()):
            raise ValueError("entity evaluation validation must contain one episode per task")
        if any(count != 2 for count in task_counts_by_partition["heldout"].values()):
            raise ValueError(
                "entity evaluation heldout requires two source episodes per task"
            )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "representation_split_sha256": self.representation_split_sha256,
            "world_size": self.world_size,
            "items": [item.as_dict() for item in self.items],
        }

    @property
    def artifact_sha256(self) -> str:
        return _digest(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

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
    def from_dict(cls, value: object) -> EntityEvaluationPlan:
        if not isinstance(value, Mapping) or set(value) != _PLAN_FIELDS:
            raise ValueError("entity evaluation plan fields differ from schema")
        raw_items = value["items"]
        if not isinstance(raw_items, list):
            raise ValueError("entity evaluation plan items must be a list")
        plan = cls(
            schema=_text(value["schema"], name="entity evaluation schema"),
            representation_split_sha256=_sha256(
                value["representation_split_sha256"],
                name="entity evaluation representation split SHA-256",
            ),
            world_size=_nonnegative_int(value["world_size"], name="entity evaluation world size"),
            items=tuple(EntityEvaluationItem.from_dict(item) for item in raw_items),
        )
        expected = _sha256(value["artifact_sha256"], name="entity evaluation artifact SHA-256")
        if plan.artifact_sha256 != expected:
            raise ValueError("entity evaluation plan artifact SHA-256 changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> EntityEvaluationPlan:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid entity evaluation plan: {source}") from error
        return cls.from_dict(value)


def build_distributed_entity_evaluation_schedule(
    plan: EntityEvaluationPlan,
    *,
    rank: int,
) -> tuple[DistributedEntityEvaluationWorkItem, ...]:
    """Match PyTorch's non-dropping distributed-sampler padding semantics.

    FSDP evaluation forwards contain collectives, so every rank must execute
    the same number of forwards. Padding replays the first immutable items but
    callers must never publish those repeated results.
    """

    if not isinstance(plan, EntityEvaluationPlan):
        raise TypeError("distributed evaluation scheduling requires an entity plan")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < plan.world_size:
        raise ValueError("distributed evaluation rank is outside the plan topology")
    samples_per_rank = (len(plan.items) + plan.world_size - 1) // plan.world_size
    total_size = samples_per_rank * plan.world_size
    padding_count = total_size - len(plan.items)
    global_schedule = tuple(
        DistributedEntityEvaluationWorkItem(item=item, is_padding=False)
        for item in plan.items
    ) + tuple(
        DistributedEntityEvaluationWorkItem(
            item=plan.items[index % len(plan.items)],
            is_padding=True,
        )
        for index in range(padding_count)
    )
    rank_schedule = global_schedule[rank:total_size:plan.world_size]
    if len(rank_schedule) != samples_per_rank:
        raise RuntimeError("distributed evaluation schedule is not collective aligned")
    scientific_items = tuple(work.item for work in rank_schedule if not work.is_padding)
    if scientific_items != tuple(item for item in plan.items if item.rank == rank):
        raise RuntimeError("distributed evaluation schedule changed scientific rank assignment")
    return rank_schedule


def build_distributed_causal_warm_evaluation_schedule(
    plan: EntityEvaluationPlan,
    *,
    rank: int,
    history_transitions: int,
) -> tuple[DistributedEntityEvaluationWorkItem, ...]:
    """Collective-align only samples with a real same-segment history prefix.

    Padding repeats eligible work solely to equalize FSDP forward counts.  The
    repeated work is marked and must not enter scientific publication.
    """

    if not isinstance(plan, EntityEvaluationPlan):
        raise TypeError("causal-warm scheduling requires an entity plan")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < plan.world_size:
        raise ValueError("causal-warm evaluation rank is outside the plan topology")
    if (
        isinstance(history_transitions, bool)
        or not isinstance(history_transitions, int)
        or history_transitions <= 0
    ):
        raise ValueError("causal-warm history length must be a positive integer")
    eligible = tuple(
        item for item in plan.items if item.transition_index >= history_transitions
    )
    if not eligible:
        raise ValueError("causal-warm evaluation plan has no eligible samples")
    samples_per_rank = (len(eligible) + plan.world_size - 1) // plan.world_size
    total_size = samples_per_rank * plan.world_size
    padding_count = total_size - len(eligible)
    global_schedule = tuple(
        DistributedEntityEvaluationWorkItem(item=item, is_padding=False)
        for item in eligible
    ) + tuple(
        DistributedEntityEvaluationWorkItem(
            item=eligible[index % len(eligible)],
            is_padding=True,
        )
        for index in range(padding_count)
    )
    rank_schedule = global_schedule[rank:total_size:plan.world_size]
    if len(rank_schedule) != samples_per_rank:
        raise RuntimeError("causal-warm schedule is not collective aligned")
    published = tuple(
        work.item
        for schedule_rank in range(plan.world_size)
        for work in global_schedule[schedule_rank:total_size:plan.world_size]
        if not work.is_padding
    )
    if tuple(sorted(published, key=lambda item: item.ordinal)) != eligible:
        raise RuntimeError("causal-warm schedule changed its scientific sample set")
    return rank_schedule


def _selection_digest(
    *,
    split_sha256: str,
    partition: str,
    task_key: str,
    segment_index: int,
) -> bytes:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "schema": ENTITY_EVALUATION_PLAN_SCHEMA,
                "split_sha256": split_sha256,
                "partition": partition,
                "task_key": task_key,
                "segment_index": segment_index,
            }
        )
    ).digest()


def _select_segments(
    segments: tuple[RepresentationEvaluationSegment, ...],
    *,
    split_sha256: str,
    partition: str,
    task_key: str,
) -> tuple[tuple[RepresentationEvaluationSegment, bytes], ...]:
    candidates = tuple(segment for segment in segments if segment.task_key == task_key)
    if len(candidates) < 2:
        raise ValueError("entity evaluation task lacks two source-disjoint segments")
    ranked = tuple(
        sorted(
            (
                (
                    _selection_digest(
                        split_sha256=split_sha256,
                        partition=partition,
                        task_key=task_key,
                        segment_index=segment.segment_index,
                    ),
                    segment,
                )
                for segment in candidates
            ),
            key=lambda item: (item[0], item[1].segment_index),
        )
    )
    return tuple((segment, digest) for digest, segment in ranked)


def build_entity_evaluation_plan(
    split: RepresentationTrialSplit,
    dataset: CalvinStatefulTransitionDataset,
    *,
    world_size: int = ENTITY_EVALUATION_WORLD_SIZE,
) -> EntityEvaluationPlan:
    """Select frames from source metadata without decoding observations or labels."""

    if not isinstance(split, RepresentationTrialSplit):
        raise TypeError("entity evaluation planning requires a representation split")
    if not isinstance(dataset, CalvinStatefulTransitionDataset):
        raise TypeError("entity evaluation planning requires a CALVIN stateful dataset")
    manifest = dataset.index.dataset_manifest
    if (
        manifest is None
        or manifest.tree_sha256 != split.dataset_manifest_sha256
        or dataset.index.dataset_id != split.dataset_id
        or dataset.index.dataset_revision != split.dataset_revision
    ):
        raise ValueError("entity evaluation dataset differs from its split")

    segments_by_partition = {
        "validation": split.validation_segments,
        "heldout": split.heldout_segments,
    }
    task_sets = {
        partition: tuple(sorted({segment.task_key for segment in segments}))
        for partition, segments in segments_by_partition.items()
    }
    if task_sets["validation"] != task_sets["heldout"]:
        raise ValueError("entity evaluation split task coverage differs")

    selected: list[tuple[str, str, RepresentationEvaluationSegment, bytes]] = []
    for partition in ENTITY_EVALUATION_PARTITIONS:
        for task_key in task_sets[partition]:
            candidates = _select_segments(
                segments_by_partition[partition],
                split_sha256=split.artifact_sha256,
                partition=partition,
                task_key=task_key,
            )
            selected.extend(
                (partition, task_key, segment, digest)
                for segment, digest in (
                    candidates[:1] if partition == "validation" else candidates
                )
            )
    selected.sort(key=lambda item: (item[0], item[1], item[2].segment_index))

    items: list[EntityEvaluationItem] = []
    for ordinal, (partition, task_key, segment, digest) in enumerate(selected):
        try:
            source = dataset.index.segments[segment.segment_index]
            episode = dataset.episode_manifest[segment.segment_index]
        except IndexError as error:
            raise ValueError("entity evaluation segment is absent from the dataset") from error
        if (
            source.index != segment.segment_index
            or source.task_key != task_key
            or int(source.episode_index) != segment.source_episode_index
            or int(source.start) != segment.source_start
            or int(source.end) != segment.source_end
            or episode.segment_index != segment.segment_index
            or not episode.sample_keys
        ):
            raise ValueError("entity evaluation segment differs from immutable source")
        transition_index = int.from_bytes(digest[:8], "big") % len(episode.sample_keys)
        sample_key = episode.sample_keys[transition_index]
        locator = dataset.locator_by_key(sample_key)
        if (
            locator.segment_index != segment.segment_index
            or locator.global_index != segment.source_start + transition_index
        ):
            raise ValueError("entity evaluation sample changed source coordinates")
        items.append(
            EntityEvaluationItem(
                partition=partition,
                ordinal=ordinal,
                rank=ordinal % world_size,
                task_key=task_key,
                segment_index=segment.segment_index,
                source_episode_index=segment.source_episode_index,
                source_global_index=locator.global_index,
                transition_index=transition_index,
                sample_key=sample_key,
            )
        )
    plan = EntityEvaluationPlan(
        representation_split_sha256=split.artifact_sha256,
        items=tuple(items),
        world_size=world_size,
    )
    sources_by_partition: dict[str, set[int]] = defaultdict(set)
    for item in plan.items:
        sources_by_partition[item.partition].add(item.source_episode_index)
    if sources_by_partition["validation"] & sources_by_partition["heldout"]:
        raise RuntimeError("entity evaluation partitions overlap by source episode")
    return plan
