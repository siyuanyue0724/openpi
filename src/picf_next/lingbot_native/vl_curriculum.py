"""Content-addressed exhaustive grounding curriculum for shared Qwen training."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import networkx as nx

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.fixed_observation import (
    FixedObservationAudit,
    FixedObservationGroup,
    FixedObservationPairPlan,
    FixedObservationVariant,
    NativeVLGroundingAudit,
    NativeVLGroundingGroup,
)

NATIVE_VL_CURRICULUM_SCHEMA = "picf-next.native-vl-grounding-curriculum.v3"
NATIVE_VL_CURRICULUM_ALGORITHM = "camera-compatible-blossom-dual-lattice-gradient.v1"
NATIVE_VL_CURRICULUM_VARIANT_POLICY = "all-token-measurable-physical-targets.v1"
NATIVE_VL_CURRICULUM_LATTICES = (8, 14)
NATIVE_VL_CURRICULUM_MAXIMUM_BYTES = 64 * 1024 * 1024


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
        raise ValueError("native VL curriculum is not canonical finite JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _hash_order(*parts: object) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"picf-next.native-vl-curriculum-order.v1\0")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


@dataclass(frozen=True, slots=True)
class NativeVLGroundingCurriculumBatch:
    """One two-rank same-image microbatch at one native Qwen lattice."""

    variant_indices: tuple[int, int]
    visual_lattice: int
    camera_name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.variant_indices, tuple)
            or len(self.variant_indices) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.variant_indices
            )
            or self.variant_indices[0] == self.variant_indices[1]
        ):
            raise ValueError("native VL curriculum requires two distinct variant indices")
        if self.visual_lattice not in NATIVE_VL_CURRICULUM_LATTICES:
            raise ValueError("native VL curriculum lattice differs from the frozen support")
        _text(self.camera_name, name="native VL curriculum camera")

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_name": self.camera_name,
            "variant_indices": list(self.variant_indices),
            "visual_lattice": self.visual_lattice,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeVLGroundingCurriculumBatch:
        expected = {"camera_name", "variant_indices", "visual_lattice"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("native VL curriculum batch fields differ from schema")
        variants = value["variant_indices"]
        if not isinstance(variants, list) or len(variants) != 2:
            raise ValueError("native VL curriculum variant indices are malformed")
        return cls(
            camera_name=_text(value["camera_name"], name="native VL curriculum camera"),
            variant_indices=(
                _nonnegative_int(variants[0], name="native VL curriculum variant index"),
                _nonnegative_int(variants[1], name="native VL curriculum variant index"),
            ),
            visual_lattice=_nonnegative_int(
                value["visual_lattice"],
                name="native VL curriculum lattice",
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeVLGroundingCurriculumStep:
    """One optimizer update that averages both native-resolution microbatches."""

    optimizer_step: int
    group_index: int
    batches: tuple[NativeVLGroundingCurriculumBatch, NativeVLGroundingCurriculumBatch]

    def __post_init__(self) -> None:
        _nonnegative_int(self.optimizer_step, name="native VL curriculum optimizer step")
        _nonnegative_int(self.group_index, name="native VL curriculum group index")
        if (
            not isinstance(self.batches, tuple)
            or len(self.batches) != len(NATIVE_VL_CURRICULUM_LATTICES)
            or any(
                not isinstance(batch, NativeVLGroundingCurriculumBatch) for batch in self.batches
            )
            or tuple(batch.visual_lattice for batch in self.batches)
            != NATIVE_VL_CURRICULUM_LATTICES
            or self.batches[1].variant_indices != tuple(reversed(self.batches[0].variant_indices))
            or self.batches[1].camera_name != self.batches[0].camera_name
        ):
            raise ValueError("native VL curriculum step is not one rank-balanced scale pair")

    def as_dict(self) -> dict[str, object]:
        return {
            "batches": [batch.as_dict() for batch in self.batches],
            "group_index": self.group_index,
            "optimizer_step": self.optimizer_step,
        }

    @classmethod
    def from_dict(cls, value: object) -> NativeVLGroundingCurriculumStep:
        expected = {"batches", "group_index", "optimizer_step"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("native VL curriculum step fields differ from schema")
        batches = value["batches"]
        if not isinstance(batches, list) or len(batches) != 2:
            raise ValueError("native VL curriculum step batches are malformed")
        parsed = tuple(NativeVLGroundingCurriculumBatch.from_dict(batch) for batch in batches)
        return cls(
            optimizer_step=_nonnegative_int(
                value["optimizer_step"],
                name="native VL curriculum optimizer step",
            ),
            group_index=_nonnegative_int(
                value["group_index"],
                name="native VL curriculum group index",
            ),
            batches=(parsed[0], parsed[1]),
        )


@dataclass(frozen=True, slots=True)
class NativeVLGroundingCurriculumPlan:
    """Exhaustive loss-only schedule over audited semantics and two scales."""

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    comparison_id: str
    seed: int
    stream_plan_sha256: str
    component_schedule_sha256: str
    representation_split_file_sha256: str
    representation_split_artifact_sha256: str
    training_projection_contract_sha256: str
    training_projection_payload_sha256: str
    source_pair_plan_file_sha256: str
    source_pair_plan_artifact_sha256: str
    audit_report_file_sha256: str
    audit_artifact_sha256: str
    source_variant_count: int
    object_row_addressable_variant_count: int
    visual_lattices: tuple[int, int]
    groups: tuple[FixedObservationGroup, ...]
    steps: tuple[NativeVLGroundingCurriculumStep, ...]

    def __post_init__(self) -> None:
        for name in ("dataset_id", "dataset_revision", "comparison_id"):
            _text(getattr(self, name), name=f"native VL curriculum {name}")
        for name in (
            "dataset_manifest_sha256",
            "stream_plan_sha256",
            "component_schedule_sha256",
            "representation_split_file_sha256",
            "representation_split_artifact_sha256",
            "training_projection_contract_sha256",
            "training_projection_payload_sha256",
            "source_pair_plan_file_sha256",
            "source_pair_plan_artifact_sha256",
            "audit_report_file_sha256",
            "audit_artifact_sha256",
        ):
            _sha256(getattr(self, name), name=f"native VL curriculum {name}")
        _nonnegative_int(self.seed, name="native VL curriculum seed")
        source_variant_count = _positive_int(
            self.source_variant_count,
            name="native VL curriculum source variant count",
        )
        object_row_count = _positive_int(
            self.object_row_addressable_variant_count,
            name="native VL curriculum object-row variant count",
        )
        if self.visual_lattices != NATIVE_VL_CURRICULUM_LATTICES:
            raise ValueError("native VL curriculum changed the frozen lattice support")
        if (
            not isinstance(self.groups, tuple)
            or not self.groups
            or any(not isinstance(group, FixedObservationGroup) for group in self.groups)
        ):
            raise ValueError("native VL curriculum requires typed source groups")
        source_keys = tuple(group.stateful_sample_key for group in self.groups)
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("native VL curriculum repeats a source group")
        measurable_count = sum(len(group.variants) for group in self.groups)
        if not object_row_count <= measurable_count <= source_variant_count:
            raise ValueError("native VL curriculum variant-policy counts are inconsistent")
        if (
            not isinstance(self.steps, tuple)
            or not self.steps
            or any(not isinstance(step, NativeVLGroundingCurriculumStep) for step in self.steps)
            or tuple(step.optimizer_step for step in self.steps) != tuple(range(len(self.steps)))
        ):
            raise ValueError("native VL curriculum steps must be contiguous typed values")
        self._validate_coverage()

    def _validate_coverage(self) -> None:
        exposure: Counter[tuple[int, int, int]] = Counter()
        rank_tasks = (Counter(), Counter())
        rank_targets = (Counter(), Counter())
        for step in self.steps:
            if step.group_index >= len(self.groups):
                raise ValueError("native VL curriculum step references an absent group")
            group = self.groups[step.group_index]
            for batch in step.batches:
                if any(index >= len(group.variants) for index in batch.variant_indices):
                    raise ValueError("native VL curriculum batch references an absent variant")
                variants = tuple(group.variants[index] for index in batch.variant_indices)
                if (
                    variants[0].task_key == variants[1].task_key
                    or variants[0].target_identity_key == variants[1].target_identity_key
                ):
                    raise ValueError("native VL curriculum rank pair lost target causality")
                for rank, (variant_index, variant) in enumerate(
                    zip(batch.variant_indices, variants, strict=True)
                ):
                    exposure[(step.group_index, variant_index, batch.visual_lattice)] += 1
                    rank_tasks[rank][variant.task_key] += 1
                    rank_targets[rank][variant.target_identity_key] += 1

        expected_steps = 0
        for group_index, group in enumerate(self.groups):
            if len(group.variants) < 2:
                raise ValueError("native VL curriculum source has fewer than two variants")
            targets = tuple(variant.target_identity_key for variant in group.variants)
            if len(set(targets)) != len(targets):
                raise ValueError("native VL curriculum source repeats a target identity")
            expected_steps += math.ceil(len(group.variants) / 2)
            per_lattice_duplicates: dict[int, int] = {}
            for lattice in self.visual_lattices:
                counts = tuple(
                    exposure[(group_index, variant_index, lattice)]
                    for variant_index in range(len(group.variants))
                )
                expected_duplicate_count = len(group.variants) % 2
                if any(count not in (1, 2) for count in counts):
                    raise ValueError("native VL curriculum variant coverage is not exhaustive")
                duplicates = tuple(index for index, count in enumerate(counts) if count == 2)
                if len(duplicates) != expected_duplicate_count:
                    raise ValueError("native VL curriculum odd-group duplicate count changed")
                if duplicates:
                    per_lattice_duplicates[lattice] = duplicates[0]
            if per_lattice_duplicates and len(set(per_lattice_duplicates.values())) != 1:
                raise ValueError("native VL curriculum odd duplicate differs across scales")
        if len(self.steps) != expected_steps:
            raise ValueError("native VL curriculum step count differs from exhaustive coverage")
        if rank_tasks[0] != rank_tasks[1] or rank_targets[0] != rank_targets[1]:
            raise ValueError("native VL curriculum rank assignment is semantically biased")

    @property
    def content(self) -> dict[str, object]:
        return {
            "algorithm": NATIVE_VL_CURRICULUM_ALGORITHM,
            "audit_artifact_sha256": self.audit_artifact_sha256,
            "audit_report_file_sha256": self.audit_report_file_sha256,
            "comparison_id": self.comparison_id,
            "component_schedule_sha256": self.component_schedule_sha256,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "groups": [group.as_dict() for group in self.groups],
            "representation_split_artifact_sha256": (self.representation_split_artifact_sha256),
            "representation_split_file_sha256": self.representation_split_file_sha256,
            "schema": NATIVE_VL_CURRICULUM_SCHEMA,
            "seed": self.seed,
            "source_pair_plan_artifact_sha256": self.source_pair_plan_artifact_sha256,
            "source_pair_plan_file_sha256": self.source_pair_plan_file_sha256,
            "source_variant_count": self.source_variant_count,
            "steps": [step.as_dict() for step in self.steps],
            "stream_plan_sha256": self.stream_plan_sha256,
            "training_projection_contract_sha256": (self.training_projection_contract_sha256),
            "training_projection_payload_sha256": self.training_projection_payload_sha256,
            "variant_policy": NATIVE_VL_CURRICULUM_VARIANT_POLICY,
            "visual_lattices": list(self.visual_lattices),
            "object_row_addressable_variant_count": (self.object_row_addressable_variant_count),
        }

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self.content)

    @property
    def rank_task_histograms(self) -> tuple[dict[str, int], dict[str, int]]:
        values = (Counter(), Counter())
        for step in self.steps:
            group = self.groups[step.group_index]
            for batch in step.batches:
                for rank, index in enumerate(batch.variant_indices):
                    values[rank][group.variants[index].task_key] += 1
        return (dict(sorted(values[0].items())), dict(sorted(values[1].items())))

    @property
    def rank_target_histograms(self) -> tuple[dict[str, int], dict[str, int]]:
        values = (Counter(), Counter())
        for step in self.steps:
            group = self.groups[step.group_index]
            for batch in step.batches:
                for rank, index in enumerate(batch.variant_indices):
                    values[rank][group.variants[index].target_identity_key] += 1
        return (dict(sorted(values[0].items())), dict(sorted(values[1].items())))

    def resolve_step(
        self,
        optimizer_step: int,
    ) -> tuple[
        FixedObservationGroup,
        tuple[
            tuple[int, str, tuple[FixedObservationVariant, FixedObservationVariant]],
            tuple[int, str, tuple[FixedObservationVariant, FixedObservationVariant]],
        ],
    ]:
        index = _nonnegative_int(
            optimizer_step,
            name="native VL curriculum optimizer step",
        )
        if index >= len(self.steps):
            raise IndexError("native VL curriculum optimizer step is outside the plan")
        step = self.steps[index]
        group = self.groups[step.group_index]
        first, second = step.batches
        return (
            group,
            (
                (
                    first.visual_lattice,
                    first.camera_name,
                    (
                        group.variants[first.variant_indices[0]],
                        group.variants[first.variant_indices[1]],
                    ),
                ),
                (
                    second.visual_lattice,
                    second.camera_name,
                    (
                        group.variants[second.variant_indices[0]],
                        group.variants[second.variant_indices[1]],
                    ),
                ),
            ),
        )

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        write_bytes_durable_exclusive(
            destination,
            _canonical_bytes({**self.content, "artifact_sha256": self.artifact_sha256}) + b"\n",
        )

    @classmethod
    def from_dict(cls, value: object) -> NativeVLGroundingCurriculumPlan:
        expected = {
            "algorithm",
            "artifact_sha256",
            "audit_artifact_sha256",
            "audit_report_file_sha256",
            "comparison_id",
            "component_schedule_sha256",
            "dataset_id",
            "dataset_manifest_sha256",
            "dataset_revision",
            "groups",
            "object_row_addressable_variant_count",
            "representation_split_artifact_sha256",
            "representation_split_file_sha256",
            "schema",
            "seed",
            "source_pair_plan_artifact_sha256",
            "source_pair_plan_file_sha256",
            "source_variant_count",
            "steps",
            "stream_plan_sha256",
            "training_projection_contract_sha256",
            "training_projection_payload_sha256",
            "variant_policy",
            "visual_lattices",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("native VL curriculum plan fields differ from schema")
        if (
            value["schema"] != NATIVE_VL_CURRICULUM_SCHEMA
            or value["algorithm"] != NATIVE_VL_CURRICULUM_ALGORITHM
            or value["variant_policy"] != NATIVE_VL_CURRICULUM_VARIANT_POLICY
        ):
            raise ValueError("native VL curriculum schema or algorithm changed")
        artifact = _sha256(
            value["artifact_sha256"],
            name="native VL curriculum artifact SHA-256",
        )
        content = {name: child for name, child in value.items() if name != "artifact_sha256"}
        if _canonical_sha256(content) != artifact:
            raise ValueError("native VL curriculum artifact SHA-256 changed")
        groups = value["groups"]
        steps = value["steps"]
        lattices = value["visual_lattices"]
        if (
            not isinstance(groups, list)
            or not isinstance(steps, list)
            or not isinstance(lattices, list)
            or len(lattices) != 2
        ):
            raise ValueError("native VL curriculum collections are malformed")
        return cls(
            dataset_id=_text(value["dataset_id"], name="native VL curriculum dataset ID"),
            dataset_revision=_text(
                value["dataset_revision"],
                name="native VL curriculum dataset revision",
            ),
            dataset_manifest_sha256=_sha256(
                value["dataset_manifest_sha256"],
                name="native VL curriculum dataset manifest SHA-256",
            ),
            comparison_id=_text(
                value["comparison_id"],
                name="native VL curriculum comparison ID",
            ),
            seed=_nonnegative_int(value["seed"], name="native VL curriculum seed"),
            stream_plan_sha256=_sha256(
                value["stream_plan_sha256"],
                name="native VL curriculum stream-plan SHA-256",
            ),
            component_schedule_sha256=_sha256(
                value["component_schedule_sha256"],
                name="native VL curriculum component-schedule SHA-256",
            ),
            representation_split_file_sha256=_sha256(
                value["representation_split_file_sha256"],
                name="native VL curriculum split file SHA-256",
            ),
            representation_split_artifact_sha256=_sha256(
                value["representation_split_artifact_sha256"],
                name="native VL curriculum split artifact SHA-256",
            ),
            training_projection_contract_sha256=_sha256(
                value["training_projection_contract_sha256"],
                name="native VL curriculum projection contract SHA-256",
            ),
            training_projection_payload_sha256=_sha256(
                value["training_projection_payload_sha256"],
                name="native VL curriculum projection payload SHA-256",
            ),
            source_pair_plan_file_sha256=_sha256(
                value["source_pair_plan_file_sha256"],
                name="native VL curriculum pair-plan file SHA-256",
            ),
            source_pair_plan_artifact_sha256=_sha256(
                value["source_pair_plan_artifact_sha256"],
                name="native VL curriculum pair-plan artifact SHA-256",
            ),
            audit_report_file_sha256=_sha256(
                value["audit_report_file_sha256"],
                name="native VL curriculum audit file SHA-256",
            ),
            audit_artifact_sha256=_sha256(
                value["audit_artifact_sha256"],
                name="native VL curriculum audit artifact SHA-256",
            ),
            source_variant_count=_positive_int(
                value["source_variant_count"],
                name="native VL curriculum source variant count",
            ),
            object_row_addressable_variant_count=_positive_int(
                value["object_row_addressable_variant_count"],
                name="native VL curriculum object-row variant count",
            ),
            visual_lattices=(
                _nonnegative_int(lattices[0], name="native VL curriculum lattice"),
                _nonnegative_int(lattices[1], name="native VL curriculum lattice"),
            ),
            groups=tuple(FixedObservationGroup.from_dict(group) for group in groups),
            steps=tuple(NativeVLGroundingCurriculumStep.from_dict(step) for step in steps),
        )

    @classmethod
    def load(cls, path: str | Path) -> NativeVLGroundingCurriculumPlan:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("native VL curriculum plan must be one real file")
        try:
            if source.stat().st_size > NATIVE_VL_CURRICULUM_MAXIMUM_BYTES:
                raise ValueError("native VL curriculum plan exceeds the maximum size")
            value = json.loads(source.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native VL curriculum plan is not valid JSON") from error
        return cls.from_dict(value)


def _ordered_groups(
    groups: tuple[FixedObservationGroup, ...],
    *,
    seed: int,
    audit_artifact_sha256: str,
) -> tuple[int, ...]:
    remaining = list(range(len(groups)))
    task_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    episode_counts: Counter[int] = Counter()
    ordered = []
    while remaining:
        scored = []
        for group_index in remaining:
            group = groups[group_index]
            task_values = tuple(task_counts[item.task_key] + 1 for item in group.variants)
            target_values = tuple(
                target_counts[item.target_identity_key] + 1 for item in group.variants
            )
            score = (
                max(task_values),
                sum(task_values),
                max(target_values),
                sum(target_values),
                scene_counts[group.scene] + 1,
                episode_counts[group.source_episode_index] + 1,
                _hash_order(
                    audit_artifact_sha256,
                    seed,
                    group.source_state_sha256,
                    "group",
                ),
            )
            scored.append((score, group_index))
        _, selected = min(scored, key=lambda item: item[0])
        remaining.remove(selected)
        ordered.append(selected)
        group = groups[selected]
        task_counts.update(item.task_key for item in group.variants)
        target_counts.update(item.target_identity_key for item in group.variants)
        scene_counts[group.scene] += 1
        episode_counts[group.source_episode_index] += 1
    return tuple(ordered)


def _compatible_camera(
    group: NativeVLGroundingGroup,
    left_index: int,
    right_index: int,
) -> str | None:
    if left_index == right_index:
        return None
    left = group.group.variants[left_index]
    right = group.group.variants[right_index]
    if left.task_key == right.task_key or left.target_identity_key == right.target_identity_key:
        return None
    common = set(group.visible_camera_names[left_index]).intersection(
        group.visible_camera_names[right_index]
    )
    if not common:
        return None
    return "static" if "static" in common else sorted(common)[0]


def _has_perfect_camera_matching(
    group: NativeVLGroundingGroup,
    slot_variants: tuple[int, ...],
    remaining_slots: tuple[int, ...],
) -> bool:
    graph = nx.Graph()
    graph.add_nodes_from(remaining_slots)
    for position, left_slot in enumerate(remaining_slots):
        for right_slot in remaining_slots[position + 1 :]:
            if (
                _compatible_camera(
                    group,
                    slot_variants[left_slot],
                    slot_variants[right_slot],
                )
                is not None
            ):
                graph.add_edge(left_slot, right_slot)
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    return len(matching) * 2 == len(remaining_slots)


def _canonical_camera_matching(
    group: NativeVLGroundingGroup,
    slot_variants: tuple[int, ...],
) -> tuple[tuple[int, int, str], ...]:
    remaining = list(range(len(slot_variants)))
    selected: list[tuple[int, int, str]] = []
    while remaining:
        left_slot = remaining[0]
        for right_slot in remaining[1:]:
            camera_name = _compatible_camera(
                group,
                slot_variants[left_slot],
                slot_variants[right_slot],
            )
            if camera_name is None:
                continue
            remainder = tuple(slot for slot in remaining if slot not in (left_slot, right_slot))
            if _has_perfect_camera_matching(group, slot_variants, remainder):
                selected.append(
                    (
                        slot_variants[left_slot],
                        slot_variants[right_slot],
                        camera_name,
                    )
                )
                remaining = list(remainder)
                break
        else:
            raise ValueError("native VL curriculum has no camera-compatible perfect matching")
    return tuple(selected)


def _variant_pairs(
    native_group: NativeVLGroundingGroup,
    *,
    seed: int,
    audit_artifact_sha256: str,
) -> tuple[tuple[int, int, str], ...]:
    group = native_group.group
    ordered = sorted(
        range(len(group.variants)),
        key=lambda index: _hash_order(
            audit_artifact_sha256,
            seed,
            group.source_state_sha256,
            group.variants[index].task_key,
            "variant",
        ),
    )
    duplicate_candidates: tuple[int | None, ...] = tuple(ordered) if len(ordered) % 2 else (None,)
    for duplicate in duplicate_candidates:
        slot_variants = tuple((*ordered, duplicate) if duplicate is not None else ordered)
        remaining_slots = tuple(range(len(slot_variants)))
        if _has_perfect_camera_matching(native_group, slot_variants, remaining_slots):
            return _canonical_camera_matching(native_group, slot_variants)
    raise ValueError("native VL curriculum source has no camera-compatible perfect matching")


def build_native_vl_grounding_curriculum(
    pair_plan: FixedObservationPairPlan,
    source_audit: NativeVLGroundingAudit,
    *,
    pair_plan_file_sha256: str,
) -> NativeVLGroundingCurriculumPlan:
    """Cover every audited variant at both scales without model-dependent selection."""

    if not isinstance(pair_plan, FixedObservationPairPlan):
        raise TypeError("native VL curriculum requires one fixed-X pair plan")
    if not isinstance(source_audit, NativeVLGroundingAudit):
        raise TypeError("native VL curriculum requires one measurable-target audit")
    audit = source_audit.fixed_x_audit
    if not isinstance(audit, FixedObservationAudit) or audit.partition != "training":
        raise ValueError("native VL curriculum requires the training audit")
    file_sha256 = _sha256(
        pair_plan_file_sha256,
        name="native VL curriculum source pair-plan file SHA-256",
    )
    if (
        pair_plan.audit_report_file_sha256 != audit.report_file_sha256
        or pair_plan.audit_artifact_sha256 != audit.report_artifact_sha256
        or pair_plan.comparison_id != audit.comparison_id
        or pair_plan.stream_plan_sha256 != audit.stream_plan_sha256
        or pair_plan.representation_split_file_sha256 != audit.representation_split_file_sha256
        or pair_plan.representation_split_artifact_sha256
        != audit.representation_split_artifact_sha256
        or pair_plan.training_projection_contract_sha256
        != audit.training_projection_contract_sha256
        or pair_plan.training_projection_payload_sha256 != audit.training_projection_payload_sha256
        or pair_plan.dataset_manifest_sha256 != audit.dataset_tree_sha256
    ):
        raise ValueError("native VL curriculum inputs belong to different evidence")
    source_groups = tuple(native_group.group for native_group in source_audit.groups)
    audited_by_key = {group.stateful_sample_key: group for group in source_groups}
    if any(
        (expanded := audited_by_key.get(pair.group.stateful_sample_key)) is None
        or replace(expanded, variants=pair.group.variants) != pair.group
        or tuple(variant for variant in expanded.variants if variant in set(pair.group.variants))
        != pair.group.variants
        or any(variant not in pair.group.variants for variant in pair.variants)
        for pair in pair_plan.pairs
    ):
        raise ValueError("native VL curriculum source pair plan differs from its audit")

    group_order = _ordered_groups(
        source_groups,
        seed=pair_plan.seed,
        audit_artifact_sha256=audit.report_artifact_sha256,
    )
    steps = []
    for group_index in group_order:
        native_group = source_audit.groups[group_index]
        pairs = _variant_pairs(
            native_group,
            seed=pair_plan.seed,
            audit_artifact_sha256=audit.report_artifact_sha256,
        )
        for left_index, right_index, camera_name in pairs:
            steps.append(
                NativeVLGroundingCurriculumStep(
                    optimizer_step=len(steps),
                    group_index=group_index,
                    batches=(
                        NativeVLGroundingCurriculumBatch(
                            variant_indices=(left_index, right_index),
                            visual_lattice=NATIVE_VL_CURRICULUM_LATTICES[0],
                            camera_name=camera_name,
                        ),
                        NativeVLGroundingCurriculumBatch(
                            variant_indices=(right_index, left_index),
                            visual_lattice=NATIVE_VL_CURRICULUM_LATTICES[1],
                            camera_name=camera_name,
                        ),
                    ),
                )
            )

    return NativeVLGroundingCurriculumPlan(
        dataset_id=pair_plan.dataset_id,
        dataset_revision=pair_plan.dataset_revision,
        dataset_manifest_sha256=pair_plan.dataset_manifest_sha256,
        comparison_id=pair_plan.comparison_id,
        seed=pair_plan.seed,
        stream_plan_sha256=pair_plan.stream_plan_sha256,
        component_schedule_sha256=pair_plan.component_schedule_sha256,
        representation_split_file_sha256=pair_plan.representation_split_file_sha256,
        representation_split_artifact_sha256=(pair_plan.representation_split_artifact_sha256),
        training_projection_contract_sha256=pair_plan.training_projection_contract_sha256,
        training_projection_payload_sha256=pair_plan.training_projection_payload_sha256,
        source_pair_plan_file_sha256=file_sha256,
        source_pair_plan_artifact_sha256=pair_plan.artifact_sha256,
        audit_report_file_sha256=audit.report_file_sha256,
        audit_artifact_sha256=audit.report_artifact_sha256,
        source_variant_count=source_audit.source_variant_count,
        object_row_addressable_variant_count=(source_audit.object_row_addressable_variant_count),
        visual_lattices=NATIVE_VL_CURRICULUM_LATTICES,
        groups=source_groups,
        steps=tuple(steps),
    )
