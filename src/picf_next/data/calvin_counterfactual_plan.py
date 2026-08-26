"""Task-independent planning contract for CALVIN object-removal pairs.

The planner reads loss-only physical visibility metadata, never task semantics,
and partitions exact-removal interventions before any optimization is run.  It
keeps source-domain validation and unseen-target-identity evaluation separate
so a mechanism smoke cannot be mistaken for generalization evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from picf_next.contracts import ContractError

CALVIN_COUNTERFACTUAL_PAIR_PLAN_SCHEMA: Final = "picf-next.calvin-counterfactual-pair-plan.v1"
CALVIN_COUNTERFACTUAL_PARTITIONS: Final = ("train", "validation", "heldout")


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be nonempty text")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ContractError(f"{name} must be positive")
    return result


def _sha256_text(value: object, name: str) -> str:
    result = _nonempty_text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one lowercase SHA-256")
    return result


def _exact_mapping(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a string-keyed mapping")
    result = cast(Mapping[str, object], value)
    if set(result) != fields:
        raise ContractError(f"{name} fields differ from the pinned contract")
    return result


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{name} must be a nonempty integer list")
    result = tuple(_nonnegative_integer(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{name} must not contain duplicates")
    return result


def _text_tuple(value: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ContractError(f"{name} must be a {'possibly empty ' if allow_empty else ''}text list")
    result = tuple(_nonempty_text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{name} must not contain duplicates")
    return result


def _identity_family(identity_key: str) -> str:
    family, separator, member = identity_key.partition("/")
    if not separator or not family or not member:
        raise ContractError("CALVIN physical identity must be FAMILY/MEMBER")
    return family


@dataclass(frozen=True, slots=True)
class CalvinCounterfactualCandidate:
    """One target visible in one deployable, language-conditioned source frame."""

    global_index: int
    segment_index: int
    source_partition: str
    scene: str
    identity_key: str
    static_visible_pixels: int
    gripper_visible_pixels: int
    task_key: str
    instruction: str

    def __post_init__(self) -> None:
        _nonnegative_integer(self.global_index, "candidate.global_index")
        _nonnegative_integer(self.segment_index, "candidate.segment_index")
        if self.source_partition not in CALVIN_COUNTERFACTUAL_PARTITIONS:
            raise ContractError("candidate source partition is invalid")
        _nonempty_text(self.scene, "candidate.scene")
        _identity_family(_nonempty_text(self.identity_key, "candidate.identity_key"))
        _nonnegative_integer(self.static_visible_pixels, "candidate.static_visible_pixels")
        _nonnegative_integer(self.gripper_visible_pixels, "candidate.gripper_visible_pixels")
        if self.static_visible_pixels + self.gripper_visible_pixels <= 0:
            raise ContractError("counterfactual candidate must be visible")
        _nonempty_text(self.task_key, "candidate.task_key")
        _nonempty_text(self.instruction, "candidate.instruction")

    @property
    def total_visible_pixels(self) -> int:
        return self.static_visible_pixels + self.gripper_visible_pixels


@dataclass(frozen=True, slots=True)
class CalvinCounterfactualPlanConfig:
    """Dataset-agnostic sampling controls for a finite intervention bank."""

    train_pairs_per_identity: int
    validation_pairs_per_train_identity: int
    heldout_pairs_per_identity: int
    heldout_identities_per_family: int
    minimum_total_visible_pixels: int
    minimum_same_identity_frame_gap: int
    seed: int

    def __post_init__(self) -> None:
        _positive_integer(self.train_pairs_per_identity, "train_pairs_per_identity")
        _positive_integer(
            self.validation_pairs_per_train_identity,
            "validation_pairs_per_train_identity",
        )
        _positive_integer(self.heldout_pairs_per_identity, "heldout_pairs_per_identity")
        _positive_integer(
            self.heldout_identities_per_family,
            "heldout_identities_per_family",
        )
        _positive_integer(self.minimum_total_visible_pixels, "minimum_total_visible_pixels")
        _nonnegative_integer(
            self.minimum_same_identity_frame_gap,
            "minimum_same_identity_frame_gap",
        )
        _nonnegative_integer(self.seed, "seed")

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CalvinCounterfactualPairRequest:
    partition: str
    global_index: int
    source_segment_index: int
    scene: str
    target_identity_key: str
    static_visible_pixels: int
    gripper_visible_pixels: int
    task_key: str
    instruction: str

    def __post_init__(self) -> None:
        if self.partition not in CALVIN_COUNTERFACTUAL_PARTITIONS:
            raise ContractError("counterfactual request partition is invalid")
        _nonnegative_integer(self.global_index, "request.global_index")
        _nonnegative_integer(self.source_segment_index, "request.source_segment_index")
        _nonempty_text(self.scene, "request.scene")
        _identity_family(_nonempty_text(self.target_identity_key, "request.target_identity_key"))
        _nonnegative_integer(self.static_visible_pixels, "request.static_visible_pixels")
        _nonnegative_integer(self.gripper_visible_pixels, "request.gripper_visible_pixels")
        if self.static_visible_pixels + self.gripper_visible_pixels <= 0:
            raise ContractError("counterfactual request target must be visible")
        _nonempty_text(self.task_key, "request.task_key")
        _nonempty_text(self.instruction, "request.instruction")

    @property
    def key(self) -> tuple[int, str]:
        return self.global_index, self.target_identity_key

    def to_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "global_index": self.global_index,
            "source_segment_index": self.source_segment_index,
            "scene": self.scene,
            "target_identity_key": self.target_identity_key,
            "static_visible_pixels": self.static_visible_pixels,
            "gripper_visible_pixels": self.gripper_visible_pixels,
            "task_key": self.task_key,
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, value: object) -> CalvinCounterfactualPairRequest:
        payload = _exact_mapping(
            value,
            "counterfactual request",
            {
                "partition",
                "global_index",
                "source_segment_index",
                "scene",
                "target_identity_key",
                "static_visible_pixels",
                "gripper_visible_pixels",
                "task_key",
                "instruction",
            },
        )
        return cls(
            partition=_nonempty_text(payload["partition"], "request.partition"),
            global_index=_nonnegative_integer(payload["global_index"], "request.global_index"),
            source_segment_index=_nonnegative_integer(
                payload["source_segment_index"], "request.source_segment_index"
            ),
            scene=_nonempty_text(payload["scene"], "request.scene"),
            target_identity_key=_nonempty_text(
                payload["target_identity_key"], "request.target_identity_key"
            ),
            static_visible_pixels=_nonnegative_integer(
                payload["static_visible_pixels"], "request.static_visible_pixels"
            ),
            gripper_visible_pixels=_nonnegative_integer(
                payload["gripper_visible_pixels"], "request.gripper_visible_pixels"
            ),
            task_key=_nonempty_text(payload["task_key"], "request.task_key"),
            instruction=_nonempty_text(payload["instruction"], "request.instruction"),
        )


def _stable_digest(*values: object) -> bytes:
    return hashlib.sha256(":".join(str(value) for value in values).encode("utf-8")).digest()


def _eligible_candidates(
    candidates: Iterable[CalvinCounterfactualCandidate],
    *,
    minimum_total_visible_pixels: int,
) -> tuple[CalvinCounterfactualCandidate, ...]:
    result = tuple(
        candidate
        for candidate in candidates
        if candidate.total_visible_pixels >= minimum_total_visible_pixels
    )
    if not result:
        raise ContractError("no CALVIN counterfactual candidates pass visibility support")
    keys = [
        (candidate.source_partition, candidate.global_index, candidate.identity_key)
        for candidate in result
    ]
    if len(set(keys)) != len(keys):
        raise ContractError("CALVIN counterfactual candidates contain duplicates")
    by_frame: dict[int, set[str]] = defaultdict(set)
    for candidate in result:
        by_frame[candidate.global_index].add(candidate.source_partition)
    if any(len(partitions) != 1 for partitions in by_frame.values()):
        raise ContractError("counterfactual source partitions overlap in frame coordinates")
    return result


def _identity_partition(
    candidates: Sequence[CalvinCounterfactualCandidate],
    config: CalvinCounterfactualPlanConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_partition_identity: dict[tuple[str, str], int] = defaultdict(int)
    for candidate in candidates:
        by_partition_identity[candidate.source_partition, candidate.identity_key] += 1
    identities = sorted({candidate.identity_key for candidate in candidates})
    capable = [
        identity
        for identity in identities
        if by_partition_identity["train", identity] >= config.train_pairs_per_identity
        and by_partition_identity["validation", identity]
        >= config.validation_pairs_per_train_identity
        and by_partition_identity["heldout", identity] >= config.heldout_pairs_per_identity
    ]
    families: dict[str, list[str]] = defaultdict(list)
    for identity in capable:
        families[_identity_family(identity)].append(identity)
    if not families:
        raise ContractError("no identity family has complete train/validation/heldout support")
    heldout: list[str] = []
    for family, members in sorted(families.items()):
        required = config.heldout_identities_per_family + 1
        if len(members) < required:
            raise ContractError(
                f"identity family {family!r} has fewer than {required} supported identities"
            )
        ordered = sorted(
            members,
            key=lambda identity: _stable_digest(config.seed, "heldout-identity", identity),
        )
        heldout.extend(ordered[: config.heldout_identities_per_family])
    heldout_set = set(heldout)
    train = tuple(sorted(identity for identity in capable if identity not in heldout_set))
    heldout_tuple = tuple(sorted(heldout_set))
    if not train or not heldout_tuple or set(train) & set(heldout_tuple):
        raise ContractError("counterfactual identity partition is degenerate")
    return train, heldout_tuple


def _select_partition_requests(
    candidates: Sequence[CalvinCounterfactualCandidate],
    *,
    partition: str,
    identities: Sequence[str],
    count_per_identity: int,
    minimum_frame_gap: int,
    seed: int,
) -> tuple[CalvinCounterfactualPairRequest, ...]:
    by_identity: dict[str, list[CalvinCounterfactualCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.source_partition == partition and candidate.identity_key in identities:
            by_identity[candidate.identity_key].append(candidate)
    for identity in identities:
        by_identity[identity].sort(
            key=lambda candidate: _stable_digest(
                seed,
                partition,
                identity,
                candidate.global_index,
            )
        )
    selected: dict[str, list[CalvinCounterfactualCandidate]] = {
        identity: [] for identity in identities
    }
    used_frames: set[int] = set()
    identity_order = sorted(
        identities,
        key=lambda identity: _stable_digest(seed, partition, "identity-order", identity),
    )
    for _round in range(count_per_identity):
        for identity in identity_order:
            chosen = next(
                (
                    candidate
                    for candidate in by_identity[identity]
                    if candidate.global_index not in used_frames
                    and all(
                        abs(candidate.global_index - prior.global_index) >= minimum_frame_gap
                        for prior in selected[identity]
                    )
                ),
                None,
            )
            if chosen is None:
                raise ContractError(
                    f"cannot select {count_per_identity} separated {partition} pairs "
                    f"for identity {identity!r}"
                )
            selected[identity].append(chosen)
            used_frames.add(chosen.global_index)
    requests = [
        CalvinCounterfactualPairRequest(
            partition=partition,
            global_index=candidate.global_index,
            source_segment_index=candidate.segment_index,
            scene=candidate.scene,
            target_identity_key=candidate.identity_key,
            static_visible_pixels=candidate.static_visible_pixels,
            gripper_visible_pixels=candidate.gripper_visible_pixels,
            task_key=candidate.task_key,
            instruction=candidate.instruction,
        )
        for identity in sorted(selected)
        for candidate in selected[identity]
    ]
    return tuple(
        sorted(
            requests,
            key=lambda request: (request.global_index, request.target_identity_key),
        )
    )


def build_calvin_counterfactual_pair_plan(
    candidates: Iterable[CalvinCounterfactualCandidate],
    *,
    config: CalvinCounterfactualPlanConfig,
    dataset_id: str,
    dataset_revision: str,
    split_name: str,
    source_sidecar_manifest_sha256: str,
    foundation_m2_recipe_sha256: str,
    source_segments: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    """Build a deterministic plan without consulting task text or target names."""

    if not isinstance(config, CalvinCounterfactualPlanConfig):
        raise TypeError("config must be CalvinCounterfactualPlanConfig")
    dataset_id = _nonempty_text(dataset_id, "dataset_id")
    dataset_revision = _nonempty_text(dataset_revision, "dataset_revision")
    split_name = _nonempty_text(split_name, "split_name")
    source_sidecar_manifest_sha256 = _sha256_text(
        source_sidecar_manifest_sha256,
        "source_sidecar_manifest_sha256",
    )
    foundation_m2_recipe_sha256 = _sha256_text(
        foundation_m2_recipe_sha256,
        "foundation_m2_recipe_sha256",
    )
    if set(source_segments) != set(CALVIN_COUNTERFACTUAL_PARTITIONS):
        raise ContractError("counterfactual source segment partitions are incomplete")
    normalized_segments = {
        partition: tuple(int(value) for value in source_segments[partition])
        for partition in CALVIN_COUNTERFACTUAL_PARTITIONS
    }
    if any(
        not values or len(set(values)) != len(values) for values in normalized_segments.values()
    ):
        raise ContractError("counterfactual source segment lists must be nonempty and unique")
    flattened = [value for values in normalized_segments.values() for value in values]
    if len(set(flattened)) != len(flattened):
        raise ContractError("counterfactual source segment partitions overlap")

    eligible = _eligible_candidates(
        candidates,
        minimum_total_visible_pixels=config.minimum_total_visible_pixels,
    )
    for candidate in eligible:
        if candidate.segment_index not in normalized_segments[candidate.source_partition]:
            raise ContractError("candidate segment disagrees with its source partition")
    train_identities, heldout_identities = _identity_partition(eligible, config)
    requests = (
        *_select_partition_requests(
            eligible,
            partition="train",
            identities=train_identities,
            count_per_identity=config.train_pairs_per_identity,
            minimum_frame_gap=config.minimum_same_identity_frame_gap,
            seed=config.seed,
        ),
        *_select_partition_requests(
            eligible,
            partition="validation",
            identities=train_identities,
            count_per_identity=config.validation_pairs_per_train_identity,
            minimum_frame_gap=config.minimum_same_identity_frame_gap,
            seed=config.seed,
        ),
        *_select_partition_requests(
            eligible,
            partition="heldout",
            identities=heldout_identities,
            count_per_identity=config.heldout_pairs_per_identity,
            minimum_frame_gap=config.minimum_same_identity_frame_gap,
            seed=config.seed,
        ),
    )
    if len({request.global_index for request in requests}) != len(requests):
        raise ContractError("counterfactual plan reuses one source frame")
    scenes = tuple(sorted({request.scene for request in requests}))
    counts = {
        partition: sum(request.partition == partition for request in requests)
        for partition in CALVIN_COUNTERFACTUAL_PARTITIONS
    }
    return {
        "schema": CALVIN_COUNTERFACTUAL_PAIR_PLAN_SCHEMA,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split_name": split_name,
        "source_sidecar_manifest_sha256": source_sidecar_manifest_sha256,
        "foundation_m2_recipe_sha256": foundation_m2_recipe_sha256,
        "selection_contract": {
            **config.to_dict(),
            "candidate_source": "loss-only depth-consistent physical owner rasters",
            "task_text_used_for_selection": False,
            "target_identity_exposed_to_model_input": False,
            "synthetic_removal_supervises_lifecycle": False,
            "source_frame_reuse": False,
        },
        "source_segments": {
            partition: list(normalized_segments[partition])
            for partition in CALVIN_COUNTERFACTUAL_PARTITIONS
        },
        "identity_partition": {
            "train_and_validation": list(train_identities),
            "heldout_only": list(heldout_identities),
        },
        "audit": {
            "request_count_by_partition": counts,
            "scene_count": len(scenes),
            "scenes": list(scenes),
            "cross_scene_generalization_tested": len(scenes) > 1,
            "identity_partitions_disjoint": True,
            "source_partitions_disjoint": True,
        },
        "requests": [request.to_dict() for request in requests],
    }


@dataclass(frozen=True, slots=True)
class CalvinCounterfactualPairPlan:
    path: Path
    file_sha256: str
    dataset_id: str
    dataset_revision: str
    split_name: str
    source_sidecar_manifest_sha256: str
    foundation_m2_recipe_sha256: str
    source_segments: Mapping[str, tuple[int, ...]]
    train_identities: tuple[str, ...]
    heldout_identities: tuple[str, ...]
    requests: tuple[CalvinCounterfactualPairRequest, ...]
    payload: Mapping[str, object]

    @property
    def keys(self) -> tuple[tuple[int, str], ...]:
        return tuple(sorted(request.key for request in self.requests))

    def requests_for(self, partition: str) -> tuple[CalvinCounterfactualPairRequest, ...]:
        if partition not in CALVIN_COUNTERFACTUAL_PARTITIONS:
            raise ValueError(f"unknown counterfactual partition: {partition}")
        return tuple(request for request in self.requests if request.partition == partition)


def load_calvin_counterfactual_pair_plan(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> CalvinCounterfactualPairPlan:
    source = Path(path).resolve()
    try:
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("CALVIN counterfactual plan is not valid JSON") from error
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 is not None and file_sha256 != _sha256_text(
        expected_sha256, "expected plan SHA-256"
    ):
        raise ContractError("CALVIN counterfactual plan hash differs")
    payload = _exact_mapping(
        raw,
        "CALVIN counterfactual plan",
        {
            "schema",
            "dataset_id",
            "dataset_revision",
            "split_name",
            "source_sidecar_manifest_sha256",
            "foundation_m2_recipe_sha256",
            "selection_contract",
            "source_segments",
            "identity_partition",
            "audit",
            "requests",
        },
    )
    if payload["schema"] != CALVIN_COUNTERFACTUAL_PAIR_PLAN_SCHEMA:
        raise ContractError("CALVIN counterfactual plan schema changed")
    source_segment_payload = _exact_mapping(
        payload["source_segments"],
        "source_segments",
        set(CALVIN_COUNTERFACTUAL_PARTITIONS),
    )
    source_segments = {
        partition: _integer_tuple(source_segment_payload[partition], f"source_segments.{partition}")
        for partition in CALVIN_COUNTERFACTUAL_PARTITIONS
    }
    flattened_segments = [value for values in source_segments.values() for value in values]
    if len(set(flattened_segments)) != len(flattened_segments):
        raise ContractError("CALVIN counterfactual source segment partitions overlap")
    identity_payload = _exact_mapping(
        payload["identity_partition"],
        "identity_partition",
        {"train_and_validation", "heldout_only"},
    )
    train_identities = _text_tuple(
        identity_payload["train_and_validation"],
        "identity_partition.train_and_validation",
    )
    heldout_identities = _text_tuple(
        identity_payload["heldout_only"],
        "identity_partition.heldout_only",
    )
    if set(train_identities) & set(heldout_identities):
        raise ContractError("CALVIN counterfactual identity partitions overlap")
    raw_requests = payload["requests"]
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ContractError("CALVIN counterfactual plan has no requests")
    requests = tuple(CalvinCounterfactualPairRequest.from_dict(value) for value in raw_requests)
    if len({request.key for request in requests}) != len(requests):
        raise ContractError("CALVIN counterfactual requests contain duplicate keys")
    if len({request.global_index for request in requests}) != len(requests):
        raise ContractError("CALVIN counterfactual requests reuse source frames")
    for request in requests:
        if request.source_segment_index not in source_segments[request.partition]:
            raise ContractError("counterfactual request source segment is in the wrong partition")
        expected_identities = (
            heldout_identities if request.partition == "heldout" else train_identities
        )
        if request.target_identity_key not in expected_identities:
            raise ContractError("counterfactual request identity is in the wrong partition")
    audit = _exact_mapping(
        payload["audit"],
        "audit",
        {
            "request_count_by_partition",
            "scene_count",
            "scenes",
            "cross_scene_generalization_tested",
            "identity_partitions_disjoint",
            "source_partitions_disjoint",
        },
    )
    counts = _exact_mapping(
        audit["request_count_by_partition"],
        "audit.request_count_by_partition",
        set(CALVIN_COUNTERFACTUAL_PARTITIONS),
    )
    for partition in CALVIN_COUNTERFACTUAL_PARTITIONS:
        actual = sum(request.partition == partition for request in requests)
        if counts[partition] != actual or actual <= 0:
            raise ContractError("counterfactual plan partition count differs from requests")
    if (
        audit["identity_partitions_disjoint"] is not True
        or audit["source_partitions_disjoint"] is not True
    ):
        raise ContractError("counterfactual plan does not certify disjoint partitions")
    return CalvinCounterfactualPairPlan(
        path=source,
        file_sha256=file_sha256,
        dataset_id=_nonempty_text(payload["dataset_id"], "dataset_id"),
        dataset_revision=_nonempty_text(payload["dataset_revision"], "dataset_revision"),
        split_name=_nonempty_text(payload["split_name"], "split_name"),
        source_sidecar_manifest_sha256=_sha256_text(
            payload["source_sidecar_manifest_sha256"],
            "source_sidecar_manifest_sha256",
        ),
        foundation_m2_recipe_sha256=_sha256_text(
            payload["foundation_m2_recipe_sha256"],
            "foundation_m2_recipe_sha256",
        ),
        source_segments=source_segments,
        train_identities=train_identities,
        heldout_identities=heldout_identities,
        requests=requests,
        payload=payload,
    )
