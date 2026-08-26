"""Frozen causal prompt interventions for bounded representation training."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from picf_next.data.calvin import CalvinStatefulTransitionDataset
from picf_next.lingbot_native.calvin import (
    NativeCALVINTrainingBatch,
    PlannedNativeCALVINBatch,
)
from picf_next.training.control import FrozenEpisodeStreamPlan, PlannedStreamTransition

REPRESENTATION_TASK_INTERVENTION_SCHEMA = "picf-next.lingbot-representation-task-intervention.v1"
REPRESENTATION_TASK_INTERVENTION_ALGORITHM = (
    "sha256-target-disjoint-episode-round-bipartite-matching.v3"
)
_MATCHING_ATTEMPTS = 128
MAXIMUM_AVOIDABLE_DONOR_TARGET_RUN = 2

TaskIdentityResolver = Callable[[str], tuple[str, ...] | None]
_SlotKey = tuple[int, str, str, str]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("task intervention is not canonical finite JSON") from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    result = _require_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identity_keys(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_require_text(item, name=name) for item in value)
    if len(set(result)) != len(result) or result != tuple(sorted(result)):
        raise ValueError(f"{name} must contain sorted unique identities")
    return result


def _slot_key(
    optimizer_step: int,
    lane_id: str,
    episode_instance_id: str,
    sample_key: str,
) -> _SlotKey:
    return optimizer_step, lane_id, episode_instance_id, sample_key


def _hash_order(*parts: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"picf-next.representation-task-intervention-order.v1\0")
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


@dataclass(frozen=True, slots=True)
class RepresentationTaskInterventionSlot:
    """One primary stream slot and its optional exact-target donor."""

    optimizer_step: int
    lane_id: str
    episode_instance_id: str
    sample_key: str
    task_key: str
    instruction_sha256: str
    target_identity_keys: tuple[str, ...]
    donor_optimizer_step: int | None = None
    donor_lane_id: str | None = None
    donor_episode_instance_id: str | None = None
    donor_sample_key: str | None = None
    donor_task_key: str | None = None
    donor_instruction_sha256: str | None = None
    donor_target_identity_keys: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.optimizer_step, name="slot optimizer_step")
        for name in ("lane_id", "episode_instance_id", "sample_key", "task_key"):
            _require_text(getattr(self, name), name=f"slot {name}")
        _require_sha256(self.instruction_sha256, name="slot instruction_sha256")
        identities = _identity_keys(
            self.target_identity_keys,
            name="slot target_identity_keys",
        )
        donor_values = (
            self.donor_optimizer_step,
            self.donor_lane_id,
            self.donor_episode_instance_id,
            self.donor_sample_key,
            self.donor_task_key,
            self.donor_instruction_sha256,
            self.donor_target_identity_keys,
        )
        if not identities:
            if any(value is not None for value in donor_values):
                raise ValueError("inexact task intervention slot cannot have a donor")
            return
        if any(value is None for value in donor_values):
            raise ValueError("exact task intervention slot requires a complete donor")
        if self.donor_optimizer_step is None or self.donor_target_identity_keys is None:
            raise RuntimeError("validated exact task donor unexpectedly vanished")
        _require_nonnegative_int(
            self.donor_optimizer_step,
            name="slot donor_optimizer_step",
        )
        for name in (
            "donor_lane_id",
            "donor_episode_instance_id",
            "donor_sample_key",
            "donor_task_key",
        ):
            _require_text(getattr(self, name), name=f"slot {name}")
        _require_sha256(
            self.donor_instruction_sha256,
            name="slot donor_instruction_sha256",
        )
        donor_identities = _identity_keys(
            self.donor_target_identity_keys,
            name="slot donor_target_identity_keys",
        )
        if not donor_identities:
            raise ValueError("exact task intervention donor must have an exact target")
        if set(identities).intersection(donor_identities):
            raise ValueError("task intervention donor target is not disjoint")
        if self.recipient_key == self.donor_key:
            raise ValueError("task intervention cannot donate a slot to itself")

    @property
    def recipient_key(self) -> _SlotKey:
        return _slot_key(
            self.optimizer_step,
            self.lane_id,
            self.episode_instance_id,
            self.sample_key,
        )

    @property
    def donor_key(self) -> _SlotKey | None:
        if self.donor_optimizer_step is None:
            return None
        if (
            self.donor_lane_id is None
            or self.donor_episode_instance_id is None
            or self.donor_sample_key is None
        ):
            raise RuntimeError("exact task intervention donor key is incomplete")
        return _slot_key(
            self.donor_optimizer_step,
            self.donor_lane_id,
            self.donor_episode_instance_id,
            self.donor_sample_key,
        )

    @property
    def intervened(self) -> bool:
        return self.donor_key is not None

    def as_dict(self) -> dict[str, Any]:
        donor = None
        if self.intervened:
            donor = {
                "episode_instance_id": self.donor_episode_instance_id,
                "instruction_sha256": self.donor_instruction_sha256,
                "lane_id": self.donor_lane_id,
                "optimizer_step": self.donor_optimizer_step,
                "sample_key": self.donor_sample_key,
                "target_identity_keys": list(self.donor_target_identity_keys or ()),
                "task_key": self.donor_task_key,
            }
        return {
            "donor": donor,
            "episode_instance_id": self.episode_instance_id,
            "instruction_sha256": self.instruction_sha256,
            "lane_id": self.lane_id,
            "optimizer_step": self.optimizer_step,
            "sample_key": self.sample_key,
            "target_identity_keys": list(self.target_identity_keys),
            "task_key": self.task_key,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RepresentationTaskInterventionSlot:
        expected = {
            "donor",
            "episode_instance_id",
            "instruction_sha256",
            "lane_id",
            "optimizer_step",
            "sample_key",
            "target_identity_keys",
            "task_key",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("task intervention slot fields differ from the schema")
        donor = value["donor"]
        donor_values: dict[str, Any]
        if donor is None:
            donor_values = {
                "donor_optimizer_step": None,
                "donor_lane_id": None,
                "donor_episode_instance_id": None,
                "donor_sample_key": None,
                "donor_task_key": None,
                "donor_instruction_sha256": None,
                "donor_target_identity_keys": None,
            }
        else:
            donor_expected = {
                "episode_instance_id",
                "instruction_sha256",
                "lane_id",
                "optimizer_step",
                "sample_key",
                "target_identity_keys",
                "task_key",
            }
            if not isinstance(donor, dict) or set(donor) != donor_expected:
                raise ValueError("task intervention donor fields differ from the schema")
            donor_values = {
                "donor_optimizer_step": donor["optimizer_step"],
                "donor_lane_id": donor["lane_id"],
                "donor_episode_instance_id": donor["episode_instance_id"],
                "donor_sample_key": donor["sample_key"],
                "donor_task_key": donor["task_key"],
                "donor_instruction_sha256": donor["instruction_sha256"],
                "donor_target_identity_keys": tuple(donor["target_identity_keys"]),
            }
        return cls(
            optimizer_step=value["optimizer_step"],
            lane_id=value["lane_id"],
            episode_instance_id=value["episode_instance_id"],
            sample_key=value["sample_key"],
            task_key=value["task_key"],
            instruction_sha256=value["instruction_sha256"],
            target_identity_keys=tuple(value["target_identity_keys"]),
            **donor_values,
        )


@dataclass(frozen=True, slots=True)
class RepresentationTaskInterventionPlan:
    """Complete content-addressed prompt intervention over one frozen stream."""

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    comparison_id: str
    seed: int
    stream_plan_sha256: str
    total_steps: int
    global_batch_size: int
    matching_attempt: int
    slots: tuple[RepresentationTaskInterventionSlot, ...]

    def __post_init__(self) -> None:
        for name in ("dataset_id", "dataset_revision", "comparison_id"):
            _require_text(getattr(self, name), name=f"plan {name}")
        _require_sha256(
            self.dataset_manifest_sha256,
            name="plan dataset_manifest_sha256",
        )
        _require_sha256(self.stream_plan_sha256, name="plan stream_plan_sha256")
        _require_nonnegative_int(self.seed, name="plan seed")
        _require_positive_int(self.total_steps, name="plan total_steps")
        _require_positive_int(self.global_batch_size, name="plan global_batch_size")
        _require_nonnegative_int(self.matching_attempt, name="plan matching_attempt")
        if self.matching_attempt >= _MATCHING_ATTEMPTS:
            raise ValueError("task intervention matching attempt is outside the algorithm")
        if len(self.slots) != self.total_steps * self.global_batch_size:
            raise ValueError("task intervention plan does not cover the full primary stream")
        expected_order = tuple(
            sorted(self.slots, key=lambda item: (item.optimizer_step, item.lane_id))
        )
        if self.slots != expected_order:
            raise ValueError("task intervention slots are not canonically ordered")
        recipient_keys = tuple(item.recipient_key for item in self.slots)
        if len(set(recipient_keys)) != len(recipient_keys):
            raise ValueError("task intervention recipient slots are not unique")
        by_key = {item.recipient_key: item for item in self.slots}
        exact = tuple(item for item in self.slots if item.intervened)
        donor_keys = tuple(item.donor_key for item in exact)
        exact_keys = {item.recipient_key for item in exact}
        if set(donor_keys) != exact_keys or len(donor_keys) != len(set(donor_keys)):
            raise ValueError("task intervention donor mapping is not an exact-slot bijection")
        for item in exact:
            donor_key = item.donor_key
            if donor_key is None:
                raise RuntimeError("validated exact slot unexpectedly lost its donor")
            donor_source = by_key[donor_key]
            if (
                item.donor_task_key != donor_source.task_key
                or item.donor_instruction_sha256 != donor_source.instruction_sha256
                or item.donor_target_identity_keys != donor_source.target_identity_keys
            ):
                raise ValueError("task intervention donor metadata differs from its source slot")

        natural_tasks = Counter(item.task_key for item in exact)
        donor_tasks = Counter(item.donor_task_key for item in exact)
        natural_targets = Counter(item.target_identity_keys for item in exact)
        donor_targets = Counter(item.donor_target_identity_keys for item in exact)
        if natural_tasks != donor_tasks or natural_targets != donor_targets:
            raise ValueError("task intervention changed exact task or target marginals")

        by_episode: dict[
            str,
            list[RepresentationTaskInterventionSlot],
        ] = defaultdict(list)
        for item in exact:
            by_episode[item.episode_instance_id].append(item)
        target_classes_by_count: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        for episode_items in by_episode.values():
            natural = {item.target_identity_keys for item in episode_items}
            if len(natural) != 1:
                raise ValueError("task intervention episode changed its natural target")
            target_classes_by_count[len(episode_items)].update(natural)
        visit_position_by_key: dict[_SlotKey, tuple[int, int]] = {}
        for episode_items in by_episode.values():
            ordered = tuple(sorted(episode_items, key=lambda item: item.optimizer_step))
            for visit_index, item in enumerate(ordered):
                visit_position_by_key[item.recipient_key] = (len(ordered), visit_index)
        for item in exact:
            donor_key = item.donor_key
            if donor_key is None:
                raise RuntimeError("validated exact slot unexpectedly lost its donor")
            if visit_position_by_key[item.recipient_key] != visit_position_by_key[donor_key]:
                raise ValueError(
                    "task intervention donor differs in episode visit count or ordinal"
                )
        for episode, episode_items in by_episode.items():
            ordered = tuple(sorted(episode_items, key=lambda item: item.optimizer_step))
            natural_target = ordered[0].target_identity_keys
            eligible_targets = {
                target
                for target in target_classes_by_count[len(ordered)]
                if set(natural_target).isdisjoint(target)
            }
            donor_sequence = tuple(item.donor_target_identity_keys for item in ordered)
            if any(target is None for target in donor_sequence):
                raise RuntimeError("validated donor target unexpectedly vanished")
            same_target_run = 0
            previous_target: tuple[str, ...] | None = None
            for donor_target in donor_sequence:
                same_target_run = same_target_run + 1 if donor_target == previous_target else 1
                previous_target = donor_target
                if (
                    len(eligible_targets) > 1
                    and same_target_run > MAXIMUM_AVOIDABLE_DONOR_TARGET_RUN
                ):
                    raise ValueError(
                        f"task intervention exceeded the avoidable donor-target run in {episode}"
                    )
            if (
                len(eligible_targets) > 1
                and len(donor_sequence) > 1
                and len(set(donor_sequence)) < 2
            ):
                raise ValueError(
                    f"task intervention retained one donor target for the full episode {episode}"
                )

    @property
    def content(self) -> dict[str, Any]:
        return {
            "algorithm": REPRESENTATION_TASK_INTERVENTION_ALGORITHM,
            "comparison_id": self.comparison_id,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "global_batch_size": self.global_batch_size,
            "matching_attempt": self.matching_attempt,
            "schema": REPRESENTATION_TASK_INTERVENTION_SCHEMA,
            "seed": self.seed,
            "slots": [item.as_dict() for item in self.slots],
            "stream_plan_sha256": self.stream_plan_sha256,
            "total_steps": self.total_steps,
        }

    @property
    def artifact_sha256(self) -> str:
        return _sha256_json(self.content)

    @property
    def exact_slot_count(self) -> int:
        return sum(item.intervened for item in self.slots)

    @property
    def inexact_slot_count(self) -> int:
        return len(self.slots) - self.exact_slot_count

    def slot_for(
        self, transition: PlannedStreamTransition, *, optimizer_step: int
    ) -> RepresentationTaskInterventionSlot:
        key = _slot_key(
            optimizer_step,
            transition.lane_id,
            transition.episode_instance_id,
            transition.sample.sample_key,
        )
        matches = tuple(item for item in self.slots if item.recipient_key == key)
        if len(matches) != 1:
            raise ValueError("planned transition is absent from the task intervention")
        return matches[0]

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        if target.exists() or target.is_symlink() or temporary.exists():
            raise FileExistsError(target)
        payload = {**self.content, "artifact_sha256": self.artifact_sha256}
        try:
            with temporary.open("xb") as handle:
                handle.write(_canonical_json_bytes(payload))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def from_dict(cls, value: Any) -> RepresentationTaskInterventionPlan:
        expected = {
            "algorithm",
            "artifact_sha256",
            "comparison_id",
            "dataset_id",
            "dataset_manifest_sha256",
            "dataset_revision",
            "global_batch_size",
            "matching_attempt",
            "schema",
            "seed",
            "slots",
            "stream_plan_sha256",
            "total_steps",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("task intervention plan fields differ from the schema")
        if value["schema"] != REPRESENTATION_TASK_INTERVENTION_SCHEMA:
            raise ValueError("task intervention plan schema changed")
        if value["algorithm"] != REPRESENTATION_TASK_INTERVENTION_ALGORITHM:
            raise ValueError("task intervention plan algorithm changed")
        if not isinstance(value["slots"], list):
            raise ValueError("task intervention plan slots must be a list")
        plan = cls(
            dataset_id=value["dataset_id"],
            dataset_revision=value["dataset_revision"],
            dataset_manifest_sha256=value["dataset_manifest_sha256"],
            comparison_id=value["comparison_id"],
            seed=value["seed"],
            stream_plan_sha256=value["stream_plan_sha256"],
            total_steps=value["total_steps"],
            global_batch_size=value["global_batch_size"],
            matching_attempt=value["matching_attempt"],
            slots=tuple(
                RepresentationTaskInterventionSlot.from_dict(item) for item in value["slots"]
            ),
        )
        expected_sha256 = _require_sha256(
            value["artifact_sha256"],
            name="task intervention artifact_sha256",
        )
        if plan.artifact_sha256 != expected_sha256:
            raise ValueError("task intervention artifact SHA-256 changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> RepresentationTaskInterventionPlan:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid task intervention plan: {source}") from error
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class _NaturalSlot:
    optimizer_step: int
    transition: PlannedStreamTransition
    task_key: str
    instruction: str
    instruction_sha256: str
    target_identity_keys: tuple[str, ...]

    @property
    def key(self) -> _SlotKey:
        return _slot_key(
            self.optimizer_step,
            self.transition.lane_id,
            self.transition.episode_instance_id,
            self.transition.sample.sample_key,
        )


def _dataset_identity(
    dataset: CalvinStatefulTransitionDataset,
) -> tuple[str, str, str]:
    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("task intervention requires a content-addressed dataset")
    return dataset.index.dataset_id, dataset.index.dataset_revision, manifest.tree_sha256


def _sample_task_instruction(
    dataset: CalvinStatefulTransitionDataset,
    sample_key: str,
) -> tuple[str, str]:
    locator = dataset.locator_by_key(sample_key)
    try:
        segment = dataset.index.segments[locator.segment_index]
    except IndexError as error:
        raise ValueError("task intervention sample has no immutable language segment") from error
    task_key = dataset.task_key_by_key(sample_key)
    if segment.task_key != task_key or not segment.instruction:
        raise ValueError("task intervention language differs from the immutable dataset")
    return task_key, segment.instruction


def _natural_slots(
    stream_plan: FrozenEpisodeStreamPlan,
    dataset: CalvinStatefulTransitionDataset,
    *,
    task_identity_resolver: TaskIdentityResolver,
) -> tuple[_NaturalSlot, ...]:
    result: list[_NaturalSlot] = []
    for optimizer_step in range(stream_plan.total_steps):
        for transition in stream_plan.global_batch(optimizer_step).transitions:
            task_key, instruction = _sample_task_instruction(
                dataset,
                transition.sample.sample_key,
            )
            resolved = task_identity_resolver(task_key)
            target_identity_keys = (
                ()
                if resolved is None
                else tuple(sorted(set(_identity_keys(resolved, name="resolved target identities"))))
            )
            if resolved is not None and not target_identity_keys:
                raise ValueError("exact task identity resolver returned no target")
            result.append(
                _NaturalSlot(
                    optimizer_step=optimizer_step,
                    transition=transition,
                    task_key=task_key,
                    instruction=instruction,
                    instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                    target_identity_keys=target_identity_keys,
                )
            )
    return tuple(sorted(result, key=lambda item: (item.optimizer_step, item.transition.lane_id)))


def _maximum_episode_matching(
    recipient_order: tuple[str, ...],
    candidates: Mapping[str, tuple[str, ...]],
) -> dict[str, str] | None:
    donor_owner: dict[str, str] = {}

    def augment(recipient_episode: str, seen: set[str]) -> bool:
        for donor_episode in candidates[recipient_episode]:
            if donor_episode in seen:
                continue
            seen.add(donor_episode)
            owner = donor_owner.get(donor_episode)
            if owner is None or augment(owner, seen):
                donor_owner[donor_episode] = recipient_episode
                return True
        return False

    for recipient_episode in recipient_order:
        if not augment(recipient_episode, set()):
            return None
    return {
        recipient_episode: donor_episode for donor_episode, recipient_episode in donor_owner.items()
    }


def _episode_round_matching(
    episodes: tuple[tuple[_NaturalSlot, ...], ...],
    *,
    stream_plan_sha256: str,
    seed: int,
    attempt: int,
    visit_count: int,
) -> dict[_SlotKey, _NaturalSlot] | None:
    sequences = {items[0].transition.episode_instance_id: items for items in episodes}
    representatives = {episode: items[0] for episode, items in sequences.items()}
    previous_target: dict[str, tuple[str, ...]] = {}
    same_target_run: dict[str, int] = {}
    complete: dict[_SlotKey, _NaturalSlot] = {}

    for visit_index in range(visit_count):
        recipient_order = tuple(
            sorted(
                representatives,
                key=lambda episode: (
                    _hash_order(
                        stream_plan_sha256,
                        str(seed),
                        str(attempt),
                        str(visit_count),
                        str(visit_index),
                        "recipient-episode",
                        episode,
                    ),
                    episode,
                ),
            )
        )
        candidates: dict[str, tuple[str, ...]] = {}
        for recipient_episode in recipient_order:
            recipient = representatives[recipient_episode]
            base = tuple(
                donor_episode
                for donor_episode, donor in representatives.items()
                if set(recipient.target_identity_keys).isdisjoint(donor.target_identity_keys)
            )
            eligible_target_classes = {
                representatives[donor_episode].target_identity_keys for donor_episode in base
            }
            previous = previous_target.get(recipient_episode)
            if (
                previous is not None
                and len(eligible_target_classes) > 1
                and same_target_run.get(recipient_episode, 0) >= MAXIMUM_AVOIDABLE_DONOR_TARGET_RUN
            ):
                base = tuple(
                    donor_episode
                    for donor_episode in base
                    if representatives[donor_episode].target_identity_keys != previous
                )
            candidates[recipient_episode] = tuple(
                sorted(
                    base,
                    key=lambda donor_episode: (
                        int(
                            previous is not None
                            and representatives[donor_episode].target_identity_keys == previous
                        ),
                        _hash_order(
                            stream_plan_sha256,
                            str(seed),
                            str(attempt),
                            str(visit_count),
                            str(visit_index),
                            "episode-edge",
                            recipient_episode,
                            donor_episode,
                        ),
                        donor_episode,
                    ),
                )
            )

        recipient_to_donor = _maximum_episode_matching(recipient_order, candidates)
        if recipient_to_donor is None:
            return None
        if set(recipient_to_donor) != set(representatives):
            raise RuntimeError("episode-round matching lost an exact recipient")
        for recipient_episode, donor_episode in recipient_to_donor.items():
            recipient = sequences[recipient_episode][visit_index]
            donor = sequences[donor_episode][visit_index]
            complete[recipient.key] = donor
            same_target_run[recipient_episode] = (
                same_target_run.get(recipient_episode, 0) + 1
                if donor.target_identity_keys == previous_target.get(recipient_episode)
                else 1
            )
            previous_target[recipient_episode] = donor.target_identity_keys
    return complete


def _longitudinal_target_disjoint_matching(
    exact_slots: tuple[_NaturalSlot, ...],
    *,
    stream_plan_sha256: str,
    seed: int,
    attempt: int,
) -> dict[_SlotKey, _NaturalSlot] | None:
    by_episode: dict[str, list[_NaturalSlot]] = defaultdict(list)
    for slot in exact_slots:
        by_episode[slot.transition.episode_instance_id].append(slot)
    by_visit_count: dict[int, list[tuple[_NaturalSlot, ...]]] = defaultdict(list)
    for episode, items in sorted(by_episode.items()):
        ordered = tuple(sorted(items, key=lambda item: item.optimizer_step))
        first = ordered[0]
        if any(
            item.task_key != first.task_key
            or item.instruction_sha256 != first.instruction_sha256
            or item.target_identity_keys != first.target_identity_keys
            or item.transition.episode_instance_id != episode
            for item in ordered
        ):
            raise ValueError("one stream episode changed natural task semantics")
        by_visit_count[len(ordered)].append(ordered)

    complete: dict[_SlotKey, _NaturalSlot] = {}
    for visit_count, episode_group in sorted(by_visit_count.items()):
        episodes = tuple(episode_group)
        if len(episodes) < 2:
            return None
        matched = _episode_round_matching(
            episodes,
            stream_plan_sha256=stream_plan_sha256,
            seed=seed,
            attempt=attempt,
            visit_count=visit_count,
        )
        if matched is None:
            return None
        complete.update(matched)
    if set(complete) != {item.key for item in exact_slots}:
        raise RuntimeError("longitudinal matching lost an exact transition slot")
    return complete


def build_representation_task_intervention_plan(
    stream_plan: FrozenEpisodeStreamPlan,
    dataset: CalvinStatefulTransitionDataset,
    *,
    task_identity_resolver: TaskIdentityResolver,
) -> RepresentationTaskInterventionPlan:
    """Build an exact-marginal, target-disjoint intervention over primary slots."""

    if not isinstance(stream_plan, FrozenEpisodeStreamPlan):
        raise TypeError("task intervention requires a frozen episode stream plan")
    if not isinstance(dataset, CalvinStatefulTransitionDataset):
        raise TypeError("task intervention requires a stateful CALVIN dataset")
    dataset_id, dataset_revision, manifest_sha256 = _dataset_identity(dataset)
    plan_identity = (
        stream_plan.dataset_id,
        stream_plan.dataset_revision,
        stream_plan.dataset_manifest_sha256,
    )
    if plan_identity != (dataset_id, dataset_revision, manifest_sha256):
        raise ValueError("task intervention dataset differs from the frozen stream")

    natural_slots = _natural_slots(
        stream_plan,
        dataset,
        task_identity_resolver=task_identity_resolver,
    )
    exact_slots = tuple(item for item in natural_slots if item.target_identity_keys)
    if len(exact_slots) < 2:
        raise ValueError("task intervention requires at least two exact-target slots")

    matching: dict[_SlotKey, _NaturalSlot] | None = None
    matching_attempt = -1
    for attempt in range(_MATCHING_ATTEMPTS):
        candidate = _longitudinal_target_disjoint_matching(
            exact_slots,
            stream_plan_sha256=stream_plan.plan_sha256,
            seed=stream_plan.seed,
            attempt=attempt,
        )
        if candidate is not None:
            matching = candidate
            matching_attempt = attempt
            break
    if matching is None:
        raise ValueError(
            "frozen stream cannot form a longitudinal target-disjoint task intervention"
        )

    slots: list[RepresentationTaskInterventionSlot] = []
    for recipient in natural_slots:
        donor = matching.get(recipient.key)
        donor_fields: dict[str, Any] = {}
        if donor is not None:
            donor_fields = {
                "donor_optimizer_step": donor.optimizer_step,
                "donor_lane_id": donor.transition.lane_id,
                "donor_episode_instance_id": donor.transition.episode_instance_id,
                "donor_sample_key": donor.transition.sample.sample_key,
                "donor_task_key": donor.task_key,
                "donor_instruction_sha256": donor.instruction_sha256,
                "donor_target_identity_keys": donor.target_identity_keys,
            }
        slots.append(
            RepresentationTaskInterventionSlot(
                optimizer_step=recipient.optimizer_step,
                lane_id=recipient.transition.lane_id,
                episode_instance_id=recipient.transition.episode_instance_id,
                sample_key=recipient.transition.sample.sample_key,
                task_key=recipient.task_key,
                instruction_sha256=recipient.instruction_sha256,
                target_identity_keys=recipient.target_identity_keys,
                **donor_fields,
            )
        )
    return RepresentationTaskInterventionPlan(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        dataset_manifest_sha256=manifest_sha256,
        comparison_id=stream_plan.comparison_id,
        seed=stream_plan.seed,
        stream_plan_sha256=stream_plan.plan_sha256,
        total_steps=stream_plan.total_steps,
        global_batch_size=stream_plan.global_batch_size,
        matching_attempt=matching_attempt,
        slots=tuple(slots),
    )


def apply_representation_task_intervention(
    planned: PlannedNativeCALVINBatch,
    plan: RepresentationTaskInterventionPlan,
    dataset: CalvinStatefulTransitionDataset,
) -> PlannedNativeCALVINBatch:
    """Replace only prompt and loss-side task semantics before host collation."""

    if not isinstance(planned, PlannedNativeCALVINBatch):
        raise TypeError("task intervention requires a planned native CALVIN batch")
    if not isinstance(plan, RepresentationTaskInterventionPlan):
        raise TypeError("task intervention requires its immutable plan")
    if not isinstance(dataset, CalvinStatefulTransitionDataset):
        raise TypeError("task intervention requires a stateful CALVIN dataset")
    if planned.task_intervention_sha256 is not None:
        raise ValueError("task intervention may be applied only once")
    if planned.plan_sha256 != plan.stream_plan_sha256:
        raise ValueError("planned batch and task intervention stream differ")
    if _dataset_identity(dataset) != (
        plan.dataset_id,
        plan.dataset_revision,
        plan.dataset_manifest_sha256,
    ):
        raise ValueError("planned task intervention dataset identity changed")

    host_items: list[dict[str, Any]] = []
    requests = []
    transitions = planned.plan_microbatch.transitions
    training = planned.training
    for transition, host_item, request in zip(
        transitions,
        training.host_items,
        training.structural_target_requests,
        strict=True,
    ):
        slot = plan.slot_for(
            transition,
            optimizer_step=planned.plan_microbatch.optimizer_step,
        )
        natural_instruction = host_item["task"]
        if (
            not isinstance(natural_instruction, str)
            or hashlib.sha256(natural_instruction.encode("utf-8")).hexdigest()
            != slot.instruction_sha256
            or request.task_key != slot.task_key
        ):
            raise ValueError("natural batch language differs from the intervention source")
        replaced_item = dict(host_item)
        replaced_request = request
        if slot.intervened:
            if slot.donor_sample_key is None or slot.donor_task_key is None:
                raise RuntimeError("intervened task slot lost donor metadata")
            donor_task_key, donor_instruction = _sample_task_instruction(
                dataset,
                slot.donor_sample_key,
            )
            if (
                donor_task_key != slot.donor_task_key
                or hashlib.sha256(donor_instruction.encode("utf-8")).hexdigest()
                != slot.donor_instruction_sha256
            ):
                raise ValueError("donor language differs from the immutable dataset")
            replaced_item["task"] = donor_instruction
            replaced_request = replace(request, task_key=donor_task_key)
        host_items.append(replaced_item)
        requests.append(replaced_request)

    intervened_training = replace(
        training,
        host_items=tuple(host_items),
        structural_target_requests=tuple(requests),
    )
    if not isinstance(intervened_training, NativeCALVINTrainingBatch):
        raise RuntimeError("task intervention produced an invalid native batch")
    return replace(
        planned,
        training=intervened_training,
        task_intervention_sha256=plan.artifact_sha256,
    )
