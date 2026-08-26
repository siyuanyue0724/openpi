"""Truth-audited same-observation prompt pairs for representation training.

This module is deliberately outside the learned graph. It binds immutable
CALVIN source observations to complete official instructions and loss-only
physical identities, then schedules two different true tasks over identical
sensor/state bytes. It contains no learned selector, score, lifecycle rule, or
model input other than the selected natural-language instruction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinStatefulTransitionDataset,
    CalvinStatefulTransitionSample,
)
from picf_next.data.calvin_geometry_schema import calvin_source_state_sha256
from picf_next.data.calvin_target_request import (
    native_calvin_structural_target_request,
)
from picf_next.data.calvin_task_applicability import CalvinSameObservationVariant
from picf_next.data.calvin_token_grid_support import (
    CalvinTokenGridIdentitySupport,
    CalvinTokenGridViewSupport,
)
from picf_next.data.dataset_manifest import DATASET_RUNTIME_VERIFICATION_MODE
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
    token_supervision_policy_sha256,
)
from picf_next.lingbot_native.calvin import (
    PlannedNativeCALVINBatch,
    build_native_calvin_training_batch,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
)
from picf_next.training.control import (
    FrozenResetMixtureStreamPlan,
    PlannedStreamTransition,
    derive_subseed,
)

CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA_V2 = "picf-next.calvin-same-observation-token-grid-audit.v2"
CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA = "picf-next.calvin-same-observation-token-grid-audit.v3"
FIXED_OBSERVATION_PAIR_PLAN_SCHEMA = "picf-next.lingbot-fixed-observation-pair-plan.v1"
FIXED_OBSERVATION_PAIR_PLAN_ALGORITHM = "truth-audited-reset-pair-balanced-sha256.v1"
FIXED_OBSERVATION_MAXIMUM_TORCH_SEED = (1 << 63) - 1

_MAXIMUM_REPORT_BYTES = 32 * 1024 * 1024
_PARTITIONS = ("training", "validation", "heldout")
_SENSOR_FIELDS = frozenset({"depth_gripper", "depth_static", "rgb_gripper", "rgb_static"})
_REPORT_FIELDS_V2 = frozenset(
    {
        "acceptance_scope",
        "applicability_artifact_sha256",
        "applicability_report_sha256",
        "artifact_sha256",
        "dataset_manifest_sha256",
        "dataset_runtime_binding",
        "group_count",
        "groups",
        "leakage_contract",
        "measurement_contract",
        "physical_sidecar_manifest_sha256",
        "representation_split",
        "schema",
        "status",
        "summary",
        "training_projection_contract_sha256",
        "training_projection_payload_sha256",
        "training_supervision_policy",
        "training_supervision_policy_sha256",
        "visual_artifacts",
    }
)
_REPORT_FIELDS = frozenset(
    {
        *_REPORT_FIELDS_V2,
        "rejected_groups",
        "rejected_visual_artifacts",
        "source_group_count",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "fixed_x_group_eligible",
        "retained_target_identity_keys",
        "retained_task_keys",
        "scene",
        "source_global_index",
        "source_sensor_sha256",
        "source_state_sha256",
        "stateful_reset_binding",
        "variants",
    }
)
_RESET_FIELDS = frozenset(
    {
        "language_segment_index",
        "source_episode_index",
        "source_instruction_sha256",
        "source_task_key",
        "stateful_episode_key",
        "stateful_sample_key",
        "transition_index",
    }
)
_VARIANT_FIELDS = frozenset(
    {
        "fixed_x_diagnostic_eligible",
        "instruction",
        "instruction_sha256",
        "proof",
        "support",
        "target_identity_key",
        "task_key",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "identity_key",
        "maximum_target_probability",
        "measurable",
        "object_row_addressable",
        "positive_token_count",
        "strict_categorical_winner_token_count",
        "strict_object_winner_token_count",
        "target_mass",
        "views",
    }
)
_VIEW_SUPPORT_FIELDS = frozenset(
    {
        "camera_name",
        "maximum_target_probability",
        "measurable",
        "merged_grid_hw",
        "object_row_addressable",
        "positive_token_count",
        "strict_categorical_winner_token_count",
        "strict_object_winner_token_count",
        "target_mass",
    }
)
_REPRESENTATION_SPLIT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "comparison_id",
        "file_sha256",
        "partition",
        "partition_segment_count",
        "partition_source_episode_count",
        "schema",
        "stream_plan_sha256",
    }
)
_VISUAL_FIELDS = frozenset({"file", "png_sha256", "source_global_index"})
_SUMMARY_FIELDS_V2 = frozenset(
    {
        "dropped_variant_count",
        "eligible_group_count",
        "ineligible_group_count",
        "retained_target_histogram",
        "retained_task_histogram",
        "retained_variant_count",
        "source_variant_count",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        *_SUMMARY_FIELDS_V2,
        "addressable_target_histogram",
        "addressable_task_histogram",
        "addressable_variant_count",
        "stranded_addressable_variant_count",
    }
)
_EXPECTED_LEAKAGE_CONTRACT = {
    "model_input_contains_applicability_proof": False,
    "model_input_contains_complete_natural_instruction": True,
    "model_input_contains_identity_or_owner": False,
    "model_input_contains_representation_split_metadata": False,
    "model_input_contains_simulator_state": False,
    "model_input_contains_stateful_binding": False,
    "model_input_contains_target": False,
    "model_input_contains_task_key": False,
}
_EXPECTED_MEASUREMENT_CONTRACT = {
    "absolute_pixel_or_probability_threshold": None,
    "context_is_not_an_object_row": True,
    "fixed_x_retention_rule": (
        "target-owner-mass-strictly-exceeds-every-other-physical-object-in-"
        "at-least-one-supervised-merged-token"
    ),
    "model_input": False,
    "projection": "exact-pinned-qwen3vl-patch-and-spatial-merger-addresses",
    "target_measure": "known-owner-mass-conditioned-within-token",
}


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
        raise ValueError("fixed-observation value is not canonical finite JSON") from error


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


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _hash_order(*parts: object) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"picf-next.fixed-observation-pair-order.v1\0")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _atomic_write(path: Path, value: object) -> None:
    write_bytes_durable_exclusive(path, _canonical_bytes(value) + b"\n")


@dataclass(frozen=True, slots=True)
class FixedObservationVariant:
    """One complete true prompt and its loss-only physical target."""

    task_key: str
    instruction: str
    instruction_sha256: str
    target_identity_key: str
    target_mass: float

    def __post_init__(self) -> None:
        task_key = _text(self.task_key, name="fixed-observation task key")
        instruction = _text(
            self.instruction,
            name="fixed-observation instruction",
        )
        _text(self.target_identity_key, name="fixed-observation target identity")
        _sha256(
            self.instruction_sha256,
            name="fixed-observation instruction SHA-256",
        )
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != (self.instruction_sha256):
            raise ValueError("fixed-observation instruction digest changed")
        if not task_key.strip():
            raise ValueError("fixed-observation task key cannot be whitespace")
        _positive_float(self.target_mass, name="fixed-observation target mass")

    def as_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "instruction_sha256": self.instruction_sha256,
            "target_identity_key": self.target_identity_key,
            "target_mass": self.target_mass,
            "task_key": self.task_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> FixedObservationVariant:
        expected = {
            "instruction",
            "instruction_sha256",
            "target_identity_key",
            "target_mass",
            "task_key",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("fixed-observation variant fields differ from schema")
        return cls(
            task_key=_text(value["task_key"], name="fixed-observation task key"),
            instruction=_text(
                value["instruction"],
                name="fixed-observation instruction",
            ),
            instruction_sha256=_sha256(
                value["instruction_sha256"],
                name="fixed-observation instruction SHA-256",
            ),
            target_identity_key=_text(
                value["target_identity_key"],
                name="fixed-observation target identity",
            ),
            target_mass=_positive_float(
                value["target_mass"],
                name="fixed-observation target mass",
            ),
        )


@dataclass(frozen=True, slots=True)
class FixedObservationGroup:
    """One exact source observation with at least two audited true tasks."""

    scene: str
    source_global_index: int
    source_state_sha256: str
    source_sensor_sha256: tuple[tuple[str, str], ...]
    source_episode_index: int
    source_task_key: str
    source_instruction_sha256: str
    stateful_episode_key: str
    stateful_sample_key: str
    variants: tuple[FixedObservationVariant, ...]

    def __post_init__(self) -> None:
        _text(self.scene, name="fixed-observation scene")
        _nonnegative_int(
            self.source_global_index,
            name="fixed-observation source index",
        )
        _sha256(
            self.source_state_sha256,
            name="fixed-observation source-state SHA-256",
        )
        _nonnegative_int(
            self.source_episode_index,
            name="fixed-observation source episode",
        )
        _text(self.source_task_key, name="fixed-observation source task key")
        _sha256(
            self.source_instruction_sha256,
            name="fixed-observation source instruction SHA-256",
        )
        _text(self.stateful_episode_key, name="fixed-observation episode key")
        _text(self.stateful_sample_key, name="fixed-observation sample key")
        if (
            not isinstance(self.source_sensor_sha256, tuple)
            or {name for name, _digest in self.source_sensor_sha256} != _SENSOR_FIELDS
            or tuple(name for name, _digest in self.source_sensor_sha256)
            != tuple(sorted(_SENSOR_FIELDS))
        ):
            raise ValueError("fixed-observation sensor hashes are incomplete or unsorted")
        for _name, digest in self.source_sensor_sha256:
            _sha256(digest, name="fixed-observation sensor SHA-256")
        if (
            not isinstance(self.variants, tuple)
            or len(self.variants) < 2
            or any(not isinstance(item, FixedObservationVariant) for item in self.variants)
        ):
            raise ValueError("fixed-observation group requires two typed variants")
        tasks = tuple(item.task_key for item in self.variants)
        targets = tuple(item.target_identity_key for item in self.variants)
        if len(set(tasks)) != len(tasks) or len(set(targets)) != len(targets):
            raise ValueError("fixed-observation variants require distinct tasks and targets")

    @property
    def source_sensor_hash_by_field(self) -> dict[str, str]:
        return dict(self.source_sensor_sha256)

    def as_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "source_episode_index": self.source_episode_index,
            "source_global_index": self.source_global_index,
            "source_instruction_sha256": self.source_instruction_sha256,
            "source_sensor_sha256": dict(self.source_sensor_sha256),
            "source_state_sha256": self.source_state_sha256,
            "source_task_key": self.source_task_key,
            "stateful_episode_key": self.stateful_episode_key,
            "stateful_sample_key": self.stateful_sample_key,
            "variants": [item.as_dict() for item in self.variants],
        }

    @classmethod
    def from_dict(cls, value: object) -> FixedObservationGroup:
        expected = {
            "scene",
            "source_episode_index",
            "source_global_index",
            "source_instruction_sha256",
            "source_sensor_sha256",
            "source_state_sha256",
            "source_task_key",
            "stateful_episode_key",
            "stateful_sample_key",
            "variants",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("fixed-observation group fields differ from schema")
        sensors = value["source_sensor_sha256"]
        variants = value["variants"]
        if not isinstance(sensors, Mapping) or not isinstance(variants, list):
            raise ValueError("fixed-observation group payload is malformed")
        return cls(
            scene=_text(value["scene"], name="fixed-observation scene"),
            source_global_index=_nonnegative_int(
                value["source_global_index"],
                name="fixed-observation source index",
            ),
            source_state_sha256=_sha256(
                value["source_state_sha256"],
                name="fixed-observation source-state SHA-256",
            ),
            source_sensor_sha256=tuple(
                (
                    _text(name, name="fixed-observation sensor field"),
                    _sha256(digest, name="fixed-observation sensor SHA-256"),
                )
                for name, digest in sorted(sensors.items())
            ),
            source_episode_index=_nonnegative_int(
                value["source_episode_index"],
                name="fixed-observation source episode",
            ),
            source_task_key=_text(
                value["source_task_key"],
                name="fixed-observation source task key",
            ),
            source_instruction_sha256=_sha256(
                value["source_instruction_sha256"],
                name="fixed-observation source instruction SHA-256",
            ),
            stateful_episode_key=_text(
                value["stateful_episode_key"],
                name="fixed-observation episode key",
            ),
            stateful_sample_key=_text(
                value["stateful_sample_key"],
                name="fixed-observation sample key",
            ),
            variants=tuple(FixedObservationVariant.from_dict(item) for item in variants),
        )


@dataclass(frozen=True, slots=True)
class FixedObservationAudit:
    """Validated loss-side token audit for one frozen split partition."""

    partition: str
    report_file_sha256: str
    report_artifact_sha256: str
    dataset_manifest_file_sha256: str
    dataset_tree_sha256: str
    representation_split_file_sha256: str
    representation_split_artifact_sha256: str
    comparison_id: str
    stream_plan_sha256: str
    training_projection_contract_sha256: str
    training_projection_payload_sha256: str
    groups: tuple[FixedObservationGroup, ...]

    def __post_init__(self) -> None:
        if self.partition not in _PARTITIONS:
            raise ValueError("fixed-observation audit partition is unsupported")
        for name in (
            "report_file_sha256",
            "report_artifact_sha256",
            "dataset_manifest_file_sha256",
            "dataset_tree_sha256",
            "representation_split_file_sha256",
            "representation_split_artifact_sha256",
            "stream_plan_sha256",
            "training_projection_contract_sha256",
            "training_projection_payload_sha256",
        ):
            _sha256(getattr(self, name), name=f"fixed-observation audit {name}")
        _text(self.comparison_id, name="fixed-observation comparison ID")
        if not self.groups or any(
            not isinstance(item, FixedObservationGroup) for item in self.groups
        ):
            raise ValueError("fixed-observation audit requires typed groups")
        source_indices = tuple(item.source_global_index for item in self.groups)
        states = tuple(item.source_state_sha256 for item in self.groups)
        samples = tuple(item.stateful_sample_key for item in self.groups)
        if source_indices != tuple(sorted(source_indices)):
            raise ValueError("fixed-observation groups must be source-index sorted")
        if any(len(set(values)) != len(values) for values in (source_indices, states, samples)):
            raise ValueError("fixed-observation groups repeat a source identity")

    @property
    def task_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted({variant.task_key for group in self.groups for variant in group.variants})
        )

    @property
    def target_identity_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {variant.target_identity_key for group in self.groups for variant in group.variants}
            )
        )


def _parse_token_grid_support(value: object) -> CalvinTokenGridIdentitySupport:
    if not isinstance(value, Mapping) or set(value) != _SUPPORT_FIELDS:
        raise ValueError("fixed-observation token support fields changed")
    raw_views = value["views"]
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("fixed-observation token support views are malformed")
    views = []
    for raw_view in raw_views:
        if not isinstance(raw_view, Mapping) or set(raw_view) != _VIEW_SUPPORT_FIELDS:
            raise ValueError("fixed-observation view support fields changed")
        merged_grid_hw = raw_view["merged_grid_hw"]
        if (
            not isinstance(merged_grid_hw, list)
            or len(merged_grid_hw) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in merged_grid_hw
            )
        ):
            raise ValueError("fixed-observation merged token grid is malformed")
        view = CalvinTokenGridViewSupport(
            camera_name=_text(
                raw_view["camera_name"],
                name="fixed-observation support camera",
            ),
            merged_grid_hw=(merged_grid_hw[0], merged_grid_hw[1]),
            target_mass=_nonnegative_float(
                raw_view["target_mass"],
                name="fixed-observation view target mass",
            ),
            maximum_target_probability=_nonnegative_float(
                raw_view["maximum_target_probability"],
                name="fixed-observation view maximum target probability",
            ),
            positive_token_count=_nonnegative_int(
                raw_view["positive_token_count"],
                name="fixed-observation view positive token count",
            ),
            strict_object_winner_token_count=_nonnegative_int(
                raw_view["strict_object_winner_token_count"],
                name="fixed-observation view strict object-winner count",
            ),
            strict_categorical_winner_token_count=_nonnegative_int(
                raw_view["strict_categorical_winner_token_count"],
                name="fixed-observation view strict categorical-winner count",
            ),
        )
        if dict(raw_view) != view.as_dict():
            raise ValueError("fixed-observation view support derivations changed")
        views.append(view)
    support = CalvinTokenGridIdentitySupport(
        identity_key=_text(
            value["identity_key"],
            name="fixed-observation support identity",
        ),
        views=tuple(views),
    )
    if dict(value) != support.as_dict():
        raise ValueError("fixed-observation token support derivations changed")
    return support


def _parse_variant(
    value: object,
    *,
    index: int,
    require_object_row_addressable: bool = True,
) -> FixedObservationVariant | None:
    if not isinstance(value, Mapping) or set(value) != _VARIANT_FIELDS:
        raise ValueError(f"fixed-observation variant {index} fields changed")
    applicable = CalvinSameObservationVariant(
        task_key=_text(value["task_key"], name="fixed-observation task key"),
        instruction=_text(
            value["instruction"],
            name="fixed-observation instruction",
        ),
        instruction_sha256=_sha256(
            value["instruction_sha256"],
            name="fixed-observation instruction SHA-256",
        ),
        target_identity_key=_text(
            value["target_identity_key"],
            name="fixed-observation target identity",
        ),
        proof=_text(
            value["proof"],
            name="fixed-observation applicability proof",
        ),
    )
    support = _parse_token_grid_support(value["support"])
    if support.identity_key != applicable.target_identity_key:
        raise ValueError("fixed-observation support belongs to another identity")
    eligible = value["fixed_x_diagnostic_eligible"]
    if not isinstance(eligible, bool):
        raise TypeError("fixed-observation eligibility must be boolean")
    if eligible != support.object_row_addressable:
        raise ValueError("fixed-observation eligibility and token support disagree")
    if require_object_row_addressable:
        if not eligible:
            return None
        if not support.measurable:
            raise ValueError("eligible fixed-observation target is not measurable")
    elif not support.measurable:
        return None
    return FixedObservationVariant(
        task_key=applicable.task_key,
        instruction=applicable.instruction,
        instruction_sha256=applicable.instruction_sha256,
        target_identity_key=applicable.target_identity_key,
        target_mass=_positive_float(
            support.target_mass,
            name="fixed-observation target mass",
        ),
    )


@dataclass(frozen=True, slots=True)
class _ParsedAuditGroupRecord:
    scene: str
    source_global_index: int
    source_state_sha256: str
    source_sensor_sha256: tuple[tuple[str, str], ...]
    source_episode_index: int
    source_task_key: str
    source_instruction_sha256: str
    stateful_episode_key: str
    stateful_sample_key: str
    retained_variants: tuple[FixedObservationVariant, ...]
    source_variant_count: int
    eligible: bool

    def as_group(self) -> FixedObservationGroup:
        if not self.eligible:
            raise ValueError("ineligible fixed-observation record cannot become a group")
        return FixedObservationGroup(
            scene=self.scene,
            source_global_index=self.source_global_index,
            source_state_sha256=self.source_state_sha256,
            source_sensor_sha256=self.source_sensor_sha256,
            source_episode_index=self.source_episode_index,
            source_task_key=self.source_task_key,
            source_instruction_sha256=self.source_instruction_sha256,
            stateful_episode_key=self.stateful_episode_key,
            stateful_sample_key=self.stateful_sample_key,
            variants=self.retained_variants,
        )


def _parse_group_record(value: object, *, index: int) -> _ParsedAuditGroupRecord:
    if not isinstance(value, Mapping) or set(value) != _GROUP_FIELDS:
        raise ValueError(f"fixed-observation group {index} fields changed")
    eligible = value["fixed_x_group_eligible"]
    if not isinstance(eligible, bool):
        raise TypeError("fixed-observation group eligibility must be boolean")
    sensors = value["source_sensor_sha256"]
    reset = value["stateful_reset_binding"]
    variants = value["variants"]
    if (
        not isinstance(sensors, Mapping)
        or set(sensors) != _SENSOR_FIELDS
        or not isinstance(reset, Mapping)
        or set(reset) != _RESET_FIELDS
        or not isinstance(variants, list)
    ):
        raise ValueError("fixed-observation source binding is malformed")
    source_index = _nonnegative_int(
        value["source_global_index"],
        name="fixed-observation source index",
    )
    segment_index = _nonnegative_int(
        reset["language_segment_index"],
        name="fixed-observation segment index",
    )
    transition_index = _nonnegative_int(
        reset["transition_index"],
        name="fixed-observation transition index",
    )
    expected_episode = f"calvin-language-segment-{segment_index:08d}"
    expected_sample = f"{expected_episode}/transition-00000000-frame-{source_index:08d}"
    if (
        transition_index != 0
        or reset["stateful_episode_key"] != expected_episode
        or reset["stateful_sample_key"] != expected_sample
    ):
        raise ValueError("fixed-observation group is not an exact stateful reset")
    parsed = tuple(
        item
        for variant_index, raw_variant in enumerate(variants)
        if (item := _parse_variant(raw_variant, index=variant_index)) is not None
    )
    retained_tasks = value["retained_task_keys"]
    retained_targets = value["retained_target_identity_keys"]
    if (
        not isinstance(retained_tasks, list)
        or not isinstance(retained_targets, list)
        or tuple(retained_tasks) != tuple(item.task_key for item in parsed)
        or tuple(retained_targets) != tuple(item.target_identity_key for item in parsed)
    ):
        raise ValueError("fixed-observation retained variants changed")
    if eligible != (len(parsed) >= 2):
        raise ValueError("fixed-observation group eligibility was not recomputed")
    return _ParsedAuditGroupRecord(
        scene=_text(value["scene"], name="fixed-observation scene"),
        source_global_index=source_index,
        source_state_sha256=_sha256(
            value["source_state_sha256"],
            name="fixed-observation source-state SHA-256",
        ),
        source_sensor_sha256=tuple(
            (
                name,
                _sha256(
                    sensors[name],
                    name=f"fixed-observation {name} SHA-256",
                ),
            )
            for name in sorted(_SENSOR_FIELDS)
        ),
        source_episode_index=_nonnegative_int(
            reset["source_episode_index"],
            name="fixed-observation source episode",
        ),
        source_task_key=_text(
            reset["source_task_key"],
            name="fixed-observation source task key",
        ),
        source_instruction_sha256=_sha256(
            reset["source_instruction_sha256"],
            name="fixed-observation source instruction SHA-256",
        ),
        stateful_episode_key=expected_episode,
        stateful_sample_key=expected_sample,
        retained_variants=parsed,
        source_variant_count=len(variants),
        eligible=eligible,
    )


def _parse_group(value: object, *, index: int) -> FixedObservationGroup:
    record = _parse_group_record(value, index=index)
    if not record.eligible:
        raise ValueError("fixed-observation eligible inventory contains an ineligible group")
    return record.as_group()


def _validate_visual_inventory(
    value: object,
    *,
    records: tuple[_ParsedAuditGroupRecord, ...],
    name: str,
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != len(records)
        or any(
            not isinstance(item, Mapping)
            or set(item) != _VISUAL_FIELDS
            or item["source_global_index"] != record.source_global_index
            or not isinstance(item["file"], str)
            or not item["file"].endswith(".png")
            for item, record in zip(value, records, strict=True)
        )
    ):
        raise ValueError(f"fixed-observation {name} visual inventory changed")
    for item in value:
        _sha256(
            item["png_sha256"],
            name=f"fixed-observation {name} visual PNG SHA-256",
        )


def load_fixed_observation_audit(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_partition: str,
) -> FixedObservationAudit:
    """Load one exact token-grid report and recompute its semantic contract."""

    expected_file = _sha256(
        expected_file_sha256,
        name="fixed-observation report expected SHA-256",
    )
    if expected_partition not in _PARTITIONS:
        raise ValueError("fixed-observation expected partition is unsupported")
    source = Path(path).expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError("fixed-observation report must be one real file")
    payload = source.read_bytes()
    if len(payload) > _MAXIMUM_REPORT_BYTES:
        raise ValueError("fixed-observation report exceeds the maximum size")
    observed_file = hashlib.sha256(payload).hexdigest()
    if observed_file != expected_file:
        raise ValueError("fixed-observation report file SHA-256 changed")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fixed-observation report is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("fixed-observation report must be a mapping")
    schema = value.get("schema")
    if schema == CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA_V2:
        report_fields = _REPORT_FIELDS_V2
    elif schema == CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA:
        report_fields = _REPORT_FIELDS
    else:
        raise ValueError("fixed-observation report schema changed")
    if set(value) != report_fields:
        raise ValueError("fixed-observation report fields differ from schema")
    artifact = _sha256(
        value["artifact_sha256"],
        name="fixed-observation report artifact SHA-256",
    )
    content = {key: child for key, child in value.items() if key != "artifact_sha256"}
    if _canonical_sha256(content) != artifact:
        raise ValueError("fixed-observation report artifact SHA-256 changed")
    if value["status"] != "PASS":
        raise ValueError("fixed-observation report did not pass")
    if value["leakage_contract"] != _EXPECTED_LEAKAGE_CONTRACT:
        raise ValueError("fixed-observation leakage contract changed")

    scope = value["acceptance_scope"]
    expected_scope = {
        "fixed_x_evaluation_bank_authorized": expected_partition != "training",
        "fixed_x_partition_artifact_authorized": True,
        "fixed_x_training_stream_plan_authorized": expected_partition == "training",
        "raw_owner_visibility_proven": True,
        "representation_partition_isolation_proven": True,
        "source_state_and_sensor_hash_binding_proven": True,
        "stateful_reset_addressability_proven": True,
        "token_grid_measurability_proven_for_retained_variants": True,
        "training_authorized": False,
    }
    if scope != expected_scope:
        raise ValueError("fixed-observation acceptance scope changed")
    if value["measurement_contract"] != _EXPECTED_MEASUREMENT_CONTRACT:
        raise ValueError("fixed-observation measurement contract changed")
    for digest_field in (
        "applicability_artifact_sha256",
        "applicability_report_sha256",
        "physical_sidecar_manifest_sha256",
    ):
        _sha256(value[digest_field], name=f"fixed-observation {digest_field}")
    supervision_policy = build_known_pixel_token_supervision_policy()
    if value["training_supervision_policy"] != supervision_policy or value[
        "training_supervision_policy_sha256"
    ] != token_supervision_policy_sha256(supervision_policy):
        raise ValueError("fixed-observation token supervision policy changed")

    runtime = value["dataset_runtime_binding"]
    split = value["representation_split"]
    groups = value["groups"]
    group_count = value["group_count"]
    is_v3 = schema == CALVIN_FIXED_OBSERVATION_AUDIT_SCHEMA
    rejected_groups = value["rejected_groups"] if is_v3 else []
    rejected_visuals = value["rejected_visual_artifacts"] if is_v3 else []
    source_group_count = value["source_group_count"] if is_v3 else group_count
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("dataset_manifest_self_consistent") is not True
        or runtime.get("dataset_runtime_verified_read_required") is not True
        or runtime.get("dataset_verification_mode") != DATASET_RUNTIME_VERIFICATION_MODE
        or not isinstance(split, Mapping)
        or set(split) != _REPRESENTATION_SPLIT_FIELDS
        or split.get("schema") != REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
        or split.get("partition") != expected_partition
        or not isinstance(groups, list)
        or isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count <= 0
        or len(groups) != group_count
        or not isinstance(rejected_groups, list)
        or isinstance(source_group_count, bool)
        or not isinstance(source_group_count, int)
        or source_group_count != group_count + len(rejected_groups)
    ):
        raise ValueError("fixed-observation report partition or group count changed")
    eligible_records = tuple(
        _parse_group_record(item, index=index) for index, item in enumerate(groups)
    )
    rejected_records = tuple(
        _parse_group_record(item, index=index) for index, item in enumerate(rejected_groups)
    )
    if any(not record.eligible for record in eligible_records) or any(
        record.eligible for record in rejected_records
    ):
        raise ValueError("fixed-observation eligible and rejected inventories disagree")
    all_records = (*eligible_records, *rejected_records)
    for name, identities in (
        ("source index", tuple(record.source_global_index for record in all_records)),
        ("source state", tuple(record.source_state_sha256 for record in all_records)),
        ("stateful sample", tuple(record.stateful_sample_key for record in all_records)),
    ):
        if len(set(identities)) != len(identities):
            raise ValueError(f"fixed-observation reports repeat a {name}")
    for name, records in (
        ("eligible", eligible_records),
        ("rejected", rejected_records),
    ):
        indices = tuple(record.source_global_index for record in records)
        if indices != tuple(sorted(indices)):
            raise ValueError(f"fixed-observation {name} inventory is not source sorted")
    parsed_groups = tuple(record.as_group() for record in eligible_records)
    _validate_visual_inventory(
        value["visual_artifacts"],
        records=eligible_records,
        name="eligible",
    )
    _validate_visual_inventory(
        rejected_visuals,
        records=rejected_records,
        name="rejected",
    )

    summary = value["summary"]
    expected_summary_fields = _SUMMARY_FIELDS if is_v3 else _SUMMARY_FIELDS_V2
    if not isinstance(summary, Mapping) or set(summary) != expected_summary_fields:
        raise ValueError("fixed-observation summary is malformed")
    retained = tuple(variant for group in parsed_groups for variant in group.variants)
    retained_task_histogram = dict(sorted(Counter(item.task_key for item in retained).items()))
    retained_target_histogram = dict(
        sorted(Counter(item.target_identity_key for item in retained).items())
    )
    addressable = (
        *retained,
        *(variant for record in rejected_records for variant in record.retained_variants),
    )
    source_variant_count = sum(record.source_variant_count for record in all_records)
    common_summary = (
        summary.get("eligible_group_count") != len(parsed_groups)
        or summary.get("ineligible_group_count") != len(rejected_records)
        or summary.get("retained_variant_count") != len(retained)
        or summary.get("source_variant_count") != source_variant_count
        or summary.get("dropped_variant_count") != source_variant_count - len(retained)
        or summary.get("retained_task_histogram") != retained_task_histogram
        or summary.get("retained_target_histogram") != retained_target_histogram
    )
    if common_summary:
        raise ValueError("fixed-observation summary differs from parsed groups")
    if is_v3:
        addressable_task_histogram = dict(
            sorted(Counter(item.task_key for item in addressable).items())
        )
        addressable_target_histogram = dict(
            sorted(Counter(item.target_identity_key for item in addressable).items())
        )
        if (
            summary.get("addressable_variant_count") != len(addressable)
            or summary.get("stranded_addressable_variant_count") != len(addressable) - len(retained)
            or summary.get("addressable_task_histogram") != addressable_task_histogram
            or summary.get("addressable_target_histogram") != addressable_target_histogram
            or set(retained_task_histogram) != set(addressable_task_histogram)
            or set(retained_target_histogram) != set(addressable_target_histogram)
        ):
            raise ValueError(
                "fixed-observation rejected groups lose addressable task or target coverage"
            )

    return FixedObservationAudit(
        partition=expected_partition,
        report_file_sha256=observed_file,
        report_artifact_sha256=artifact,
        dataset_manifest_file_sha256=_sha256(
            value["dataset_manifest_sha256"],
            name="fixed-observation dataset manifest file SHA-256",
        ),
        dataset_tree_sha256=_sha256(
            runtime.get("dataset_tree_sha256"),
            name="fixed-observation dataset tree SHA-256",
        ),
        representation_split_file_sha256=_sha256(
            split["file_sha256"],
            name="fixed-observation split file SHA-256",
        ),
        representation_split_artifact_sha256=_sha256(
            split["artifact_sha256"],
            name="fixed-observation split artifact SHA-256",
        ),
        comparison_id=_text(
            split["comparison_id"],
            name="fixed-observation comparison ID",
        ),
        stream_plan_sha256=_sha256(
            split["stream_plan_sha256"],
            name="fixed-observation stream-plan SHA-256",
        ),
        training_projection_contract_sha256=_sha256(
            value["training_projection_contract_sha256"],
            name="fixed-observation projection contract SHA-256",
        ),
        training_projection_payload_sha256=_sha256(
            value["training_projection_payload_sha256"],
            name="fixed-observation projection payload SHA-256",
        ),
        groups=parsed_groups,
    )


@dataclass(frozen=True, slots=True)
class NativeVLGroundingGroup:
    """One measurable target group with audited per-variant camera support."""

    group: FixedObservationGroup
    visible_camera_names: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.group, FixedObservationGroup):
            raise TypeError("native VL grounding group requires one audited source group")
        if not isinstance(self.visible_camera_names, tuple) or len(
            self.visible_camera_names
        ) != len(self.group.variants):
            raise ValueError("native VL grounding camera support differs from its variants")
        for cameras in self.visible_camera_names:
            if (
                not isinstance(cameras, tuple)
                or not cameras
                or any(not isinstance(camera, str) or not camera for camera in cameras)
                or cameras != tuple(sorted(set(cameras)))
            ):
                raise ValueError("native VL grounding camera support must be nonempty and sorted")


@dataclass(frozen=True, slots=True)
class NativeVLGroundingAudit:
    """All measurable physical targets from one validated fixed-X source audit."""

    fixed_x_audit: FixedObservationAudit
    groups: tuple[NativeVLGroundingGroup, ...]
    source_variant_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_x_audit, FixedObservationAudit):
            raise TypeError("native VL grounding audit requires one fixed-observation audit")
        source_count = _positive_int(
            self.source_variant_count,
            name="native VL grounding source variant count",
        )
        if (
            not isinstance(self.groups, tuple)
            or len(self.groups) != len(self.fixed_x_audit.groups)
            or any(not isinstance(group, NativeVLGroundingGroup) for group in self.groups)
        ):
            raise ValueError("native VL grounding groups differ from the fixed-X source")
        for fixed_group, native_group in zip(
            self.fixed_x_audit.groups,
            self.groups,
            strict=True,
        ):
            expanded_group = native_group.group
            fixed_variants = set(fixed_group.variants)
            retained_in_expanded_order = tuple(
                variant for variant in expanded_group.variants if variant in fixed_variants
            )
            if (
                replace(expanded_group, variants=fixed_group.variants) != fixed_group
                or retained_in_expanded_order != fixed_group.variants
            ):
                raise ValueError("native VL grounding expansion changed fixed-X evidence")
        object_row_count = self.object_row_addressable_variant_count
        measurable_count = self.measurable_variant_count
        if not object_row_count <= measurable_count <= source_count:
            raise ValueError("native VL grounding variant counts are inconsistent")

    @property
    def object_row_addressable_variant_count(self) -> int:
        return sum(len(group.variants) for group in self.fixed_x_audit.groups)

    @property
    def measurable_variant_count(self) -> int:
        return sum(len(group.group.variants) for group in self.groups)

    @property
    def task_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    variant.task_key
                    for native_group in self.groups
                    for variant in native_group.group.variants
                }
            )
        )

    @property
    def target_identity_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    variant.target_identity_key
                    for native_group in self.groups
                    for variant in native_group.group.variants
                }
            )
        )


def load_native_vl_grounding_audit(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_partition: str,
) -> NativeVLGroundingAudit:
    """Retain every measurable loss-side target, including non-winning token support."""

    fixed = load_fixed_observation_audit(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_partition=expected_partition,
    )
    source = Path(path).expanduser().absolute()
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != fixed.report_file_sha256:
        raise ValueError("native VL grounding source changed after fixed-X validation")
    value = json.loads(payload)
    raw_groups = value["groups"]
    summary = value["summary"]
    if not isinstance(raw_groups, list) or len(raw_groups) != len(fixed.groups):
        raise ValueError("native VL grounding source groups changed after fixed-X validation")
    expanded_groups = []
    for fixed_group, raw_group in zip(fixed.groups, raw_groups, strict=True):
        if not isinstance(raw_group, Mapping) or raw_group.get("source_global_index") != (
            fixed_group.source_global_index
        ):
            raise ValueError("native VL grounding source order changed after validation")
        raw_variants = raw_group.get("variants")
        if not isinstance(raw_variants, list):
            raise ValueError("native VL grounding source variants are malformed")
        measurable = []
        visible_camera_names = []
        for variant_index, raw_variant in enumerate(raw_variants):
            item = _parse_variant(
                raw_variant,
                index=variant_index,
                require_object_row_addressable=False,
            )
            if item is None:
                continue
            if not isinstance(raw_variant, Mapping):
                raise ValueError("native VL grounding source variant is malformed")
            support = _parse_token_grid_support(raw_variant["support"])
            cameras = tuple(sorted(view.camera_name for view in support.views if view.measurable))
            if not cameras:
                raise ValueError("native VL grounding measurable target has no visible camera")
            measurable.append(item)
            visible_camera_names.append(cameras)
        measurable_tuple = tuple(measurable)
        fixed_variants = set(fixed_group.variants)
        if (
            tuple(item for item in measurable_tuple if item in fixed_variants)
            != fixed_group.variants
        ):
            raise ValueError("native VL grounding expansion lost an object-row target")
        expanded_groups.append(
            NativeVLGroundingGroup(
                group=replace(fixed_group, variants=measurable_tuple),
                visible_camera_names=tuple(visible_camera_names),
            )
        )
    if not isinstance(summary, Mapping):
        raise ValueError("native VL grounding source summary is malformed")
    result = NativeVLGroundingAudit(
        fixed_x_audit=fixed,
        groups=tuple(expanded_groups),
        source_variant_count=_positive_int(
            summary.get("source_variant_count"),
            name="native VL grounding source variant count",
        ),
    )
    if result.object_row_addressable_variant_count != _positive_int(
        summary.get("retained_variant_count"),
        name="native VL grounding object-row variant count",
    ):
        raise ValueError("native VL grounding object-row count changed after validation")
    return result


@dataclass(frozen=True, slots=True)
class FixedObservationPair:
    """One global reset update with two prompt variants over one source."""

    optimizer_step: int
    lane_ids: tuple[str, str]
    group: FixedObservationGroup
    variants: tuple[FixedObservationVariant, FixedObservationVariant]
    augmentation_seed: int
    flow_noise_seed: int
    flow_timestep_seed: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.optimizer_step, name="fixed-X optimizer step")
        if (
            not isinstance(self.lane_ids, tuple)
            or len(self.lane_ids) != 2
            or any(not isinstance(value, str) or not value for value in self.lane_ids)
            or len(set(self.lane_ids)) != 2
        ):
            raise ValueError("fixed-X pair requires two unique lane IDs")
        if not isinstance(self.group, FixedObservationGroup):
            raise TypeError("fixed-X pair requires one typed source group")
        if (
            not isinstance(self.variants, tuple)
            or len(self.variants) != 2
            or any(not isinstance(item, FixedObservationVariant) for item in self.variants)
            or self.variants[0] == self.variants[1]
            or self.variants[0].task_key == self.variants[1].task_key
            or self.variants[0].target_identity_key == self.variants[1].target_identity_key
        ):
            raise ValueError("fixed-X pair requires two distinct task/target variants")
        if any(item not in self.group.variants for item in self.variants):
            raise ValueError("fixed-X pair variant is absent from its audited group")
        for name, value in (
            ("augmentation seed", self.augmentation_seed),
            ("flow-noise seed", self.flow_noise_seed),
            ("flow-timestep seed", self.flow_timestep_seed),
        ):
            _nonnegative_int(value, name=f"fixed-X {name}")
            if value > FIXED_OBSERVATION_MAXIMUM_TORCH_SEED:
                raise ValueError(f"fixed-X {name} exceeds PyTorch's signed range")

    def variant_for_lane(self, lane_id: str) -> FixedObservationVariant:
        try:
            index = self.lane_ids.index(lane_id)
        except ValueError as error:
            raise ValueError("planned lane is absent from its fixed-X pair") from error
        return self.variants[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "augmentation_seed": self.augmentation_seed,
            "flow_noise_seed": self.flow_noise_seed,
            "flow_timestep_seed": self.flow_timestep_seed,
            "group": self.group.as_dict(),
            "lane_ids": list(self.lane_ids),
            "optimizer_step": self.optimizer_step,
            "variants": [item.as_dict() for item in self.variants],
        }

    @classmethod
    def from_dict(cls, value: object) -> FixedObservationPair:
        expected = {
            "augmentation_seed",
            "flow_noise_seed",
            "flow_timestep_seed",
            "group",
            "lane_ids",
            "optimizer_step",
            "variants",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("fixed-X pair fields differ from schema")
        lane_ids = value["lane_ids"]
        variants = value["variants"]
        if (
            not isinstance(lane_ids, list)
            or len(lane_ids) != 2
            or any(not isinstance(item, str) for item in lane_ids)
            or not isinstance(variants, list)
            or len(variants) != 2
        ):
            raise ValueError("fixed-X pair lanes or variants are malformed")
        parsed_variants = tuple(FixedObservationVariant.from_dict(item) for item in variants)
        return cls(
            optimizer_step=_nonnegative_int(
                value["optimizer_step"],
                name="fixed-X optimizer step",
            ),
            lane_ids=(lane_ids[0], lane_ids[1]),
            group=FixedObservationGroup.from_dict(value["group"]),
            variants=(parsed_variants[0], parsed_variants[1]),
            augmentation_seed=_nonnegative_int(
                value["augmentation_seed"],
                name="fixed-X augmentation seed",
            ),
            flow_noise_seed=_nonnegative_int(
                value["flow_noise_seed"],
                name="fixed-X flow-noise seed",
            ),
            flow_timestep_seed=_nonnegative_int(
                value["flow_timestep_seed"],
                name="fixed-X flow-timestep seed",
            ),
        )


@dataclass(frozen=True, slots=True)
class FixedObservationPairPlan:
    """Content-addressed reset overlay; causal stream batches remain untouched."""

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    comparison_id: str
    seed: int
    stream_plan_sha256: str
    component_schedule_sha256: str
    audit_report_file_sha256: str
    audit_artifact_sha256: str
    representation_split_file_sha256: str
    representation_split_artifact_sha256: str
    training_projection_contract_sha256: str
    training_projection_payload_sha256: str
    candidate_group_count: int
    available_task_keys: tuple[str, ...]
    available_target_identity_keys: tuple[str, ...]
    pairs: tuple[FixedObservationPair, ...]
    _pair_by_step: dict[int, FixedObservationPair] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in ("dataset_id", "dataset_revision", "comparison_id"):
            _text(getattr(self, name), name=f"fixed-X plan {name}")
        for name in (
            "dataset_manifest_sha256",
            "stream_plan_sha256",
            "component_schedule_sha256",
            "audit_report_file_sha256",
            "audit_artifact_sha256",
            "representation_split_file_sha256",
            "representation_split_artifact_sha256",
            "training_projection_contract_sha256",
            "training_projection_payload_sha256",
        ):
            _sha256(getattr(self, name), name=f"fixed-X plan {name}")
        _nonnegative_int(self.seed, name="fixed-X plan seed")
        _positive_int(
            self.candidate_group_count,
            name="fixed-X candidate group count",
        )
        for name, values in (
            ("available tasks", self.available_task_keys),
            ("available targets", self.available_target_identity_keys),
        ):
            if (
                not isinstance(values, tuple)
                or not values
                or values != tuple(sorted(set(values)))
                or any(not isinstance(item, str) or not item for item in values)
            ):
                raise ValueError(f"fixed-X {name} must be sorted unique text")
        if (
            not isinstance(self.pairs, tuple)
            or not self.pairs
            or any(not isinstance(item, FixedObservationPair) for item in self.pairs)
        ):
            raise ValueError("fixed-X plan requires typed pairs")
        if self.pairs != tuple(sorted(self.pairs, key=lambda item: item.optimizer_step)):
            raise ValueError("fixed-X pairs must be optimizer-step sorted")
        steps = tuple(item.optimizer_step for item in self.pairs)
        sources = tuple(item.group.stateful_sample_key for item in self.pairs)
        if len(set(steps)) != len(steps) or len(set(sources)) != len(sources):
            raise ValueError("fixed-X plan repeats a step or source observation")
        if len(self.pairs) > self.candidate_group_count:
            raise ValueError("fixed-X plan selected more groups than were available")
        selected_tasks = {variant.task_key for pair in self.pairs for variant in pair.variants}
        selected_targets = {
            variant.target_identity_key for pair in self.pairs for variant in pair.variants
        }
        if selected_tasks != set(self.available_task_keys):
            raise ValueError("fixed-X plan failed complete available-task coverage")
        if selected_targets != set(self.available_target_identity_keys):
            raise ValueError("fixed-X plan failed complete available-target coverage")
        object.__setattr__(
            self,
            "_pair_by_step",
            {item.optimizer_step: item for item in self.pairs},
        )

    @property
    def content(self) -> dict[str, object]:
        return {
            "algorithm": FIXED_OBSERVATION_PAIR_PLAN_ALGORITHM,
            "audit_artifact_sha256": self.audit_artifact_sha256,
            "audit_report_file_sha256": self.audit_report_file_sha256,
            "available_target_identity_keys": list(self.available_target_identity_keys),
            "available_task_keys": list(self.available_task_keys),
            "candidate_group_count": self.candidate_group_count,
            "comparison_id": self.comparison_id,
            "component_schedule_sha256": self.component_schedule_sha256,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "pairs": [item.as_dict() for item in self.pairs],
            "representation_split_artifact_sha256": (self.representation_split_artifact_sha256),
            "representation_split_file_sha256": (self.representation_split_file_sha256),
            "schema": FIXED_OBSERVATION_PAIR_PLAN_SCHEMA,
            "seed": self.seed,
            "stream_plan_sha256": self.stream_plan_sha256,
            "training_projection_contract_sha256": (self.training_projection_contract_sha256),
            "training_projection_payload_sha256": (self.training_projection_payload_sha256),
        }

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self.content)

    @property
    def task_histogram(self) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    variant.task_key for pair in self.pairs for variant in pair.variants
                ).items()
            )
        )

    @property
    def target_histogram(self) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    variant.target_identity_key for pair in self.pairs for variant in pair.variants
                ).items()
            )
        )

    def pair_for_step(self, optimizer_step: int) -> FixedObservationPair | None:
        _nonnegative_int(optimizer_step, name="fixed-X optimizer step")
        return self._pair_by_step.get(optimizer_step)

    def slot_for(
        self,
        transition: PlannedStreamTransition,
        *,
        optimizer_step: int,
    ) -> tuple[FixedObservationPair, FixedObservationVariant]:
        pair = self.pair_for_step(optimizer_step)
        if pair is None:
            raise ValueError("optimizer step is not a fixed-X reset")
        return pair, pair.variant_for_lane(transition.lane_id)

    def write(self, path: str | Path) -> None:
        _atomic_write(
            Path(path),
            {**self.content, "artifact_sha256": self.artifact_sha256},
        )

    @classmethod
    def from_dict(cls, value: object) -> FixedObservationPairPlan:
        expected = {
            *FixedObservationPairPlan.__dataclass_fields__.keys(),
        }
        expected.discard("_pair_by_step")
        expected.update({"algorithm", "artifact_sha256", "schema"})
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("fixed-X plan fields differ from schema")
        if (
            value["schema"] != FIXED_OBSERVATION_PAIR_PLAN_SCHEMA
            or value["algorithm"] != FIXED_OBSERVATION_PAIR_PLAN_ALGORITHM
        ):
            raise ValueError("fixed-X plan schema or algorithm changed")
        pairs = value["pairs"]
        tasks = value["available_task_keys"]
        targets = value["available_target_identity_keys"]
        if (
            not isinstance(pairs, list)
            or not isinstance(tasks, list)
            or not isinstance(
                targets,
                list,
            )
        ):
            raise ValueError("fixed-X plan sequences are malformed")
        plan = cls(
            dataset_id=_text(value["dataset_id"], name="fixed-X dataset ID"),
            dataset_revision=_text(
                value["dataset_revision"],
                name="fixed-X dataset revision",
            ),
            dataset_manifest_sha256=_sha256(
                value["dataset_manifest_sha256"],
                name="fixed-X dataset manifest SHA-256",
            ),
            comparison_id=_text(
                value["comparison_id"],
                name="fixed-X comparison ID",
            ),
            seed=_nonnegative_int(value["seed"], name="fixed-X seed"),
            stream_plan_sha256=_sha256(
                value["stream_plan_sha256"],
                name="fixed-X stream-plan SHA-256",
            ),
            component_schedule_sha256=_sha256(
                value["component_schedule_sha256"],
                name="fixed-X component-schedule SHA-256",
            ),
            audit_report_file_sha256=_sha256(
                value["audit_report_file_sha256"],
                name="fixed-X audit report SHA-256",
            ),
            audit_artifact_sha256=_sha256(
                value["audit_artifact_sha256"],
                name="fixed-X audit artifact SHA-256",
            ),
            representation_split_file_sha256=_sha256(
                value["representation_split_file_sha256"],
                name="fixed-X split file SHA-256",
            ),
            representation_split_artifact_sha256=_sha256(
                value["representation_split_artifact_sha256"],
                name="fixed-X split artifact SHA-256",
            ),
            training_projection_contract_sha256=_sha256(
                value["training_projection_contract_sha256"],
                name="fixed-X projection contract SHA-256",
            ),
            training_projection_payload_sha256=_sha256(
                value["training_projection_payload_sha256"],
                name="fixed-X projection payload SHA-256",
            ),
            candidate_group_count=_positive_int(
                value["candidate_group_count"],
                name="fixed-X candidate count",
            ),
            available_task_keys=tuple(tasks),
            available_target_identity_keys=tuple(targets),
            pairs=tuple(FixedObservationPair.from_dict(item) for item in pairs),
        )
        expected_artifact = _sha256(
            value["artifact_sha256"],
            name="fixed-X plan artifact SHA-256",
        )
        if plan.artifact_sha256 != expected_artifact:
            raise ValueError("fixed-X plan artifact SHA-256 changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> FixedObservationPairPlan:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid fixed-X pair plan: {source}") from error
        return cls.from_dict(value)


def _pair_score(
    variants: tuple[FixedObservationVariant, FixedObservationVariant],
    *,
    task_counts: Counter[str],
    target_counts: Counter[str],
    available_tasks: frozenset[str],
    available_targets: frozenset[str],
    selected_tasks: set[str],
    selected_targets: set[str],
    tie_breaker: bytes,
) -> tuple[object, ...]:
    return (
        *_pair_balance_score(
            variants,
            task_counts=task_counts,
            target_counts=target_counts,
            available_tasks=available_tasks,
            available_targets=available_targets,
            selected_tasks=selected_tasks,
            selected_targets=selected_targets,
        ),
        tie_breaker,
    )


def _pair_balance_score(
    variants: tuple[FixedObservationVariant, FixedObservationVariant],
    *,
    task_counts: Counter[str],
    target_counts: Counter[str],
    available_tasks: frozenset[str],
    available_targets: frozenset[str],
    selected_tasks: set[str],
    selected_targets: set[str],
) -> tuple[int, int, int, int, int]:
    tasks = tuple(item.task_key for item in variants)
    targets = tuple(item.target_identity_key for item in variants)
    new_facets = len(set(tasks) - selected_tasks) + len(set(targets) - selected_targets)
    task_increments = Counter(tasks)
    target_increments = Counter(targets)
    projected_tasks = tuple(
        task_counts.get(key, 0) + task_increments.get(key, 0) for key in available_tasks
    )
    projected_targets = tuple(
        target_counts.get(key, 0) + target_increments.get(key, 0) for key in available_targets
    )
    return (
        -new_facets,
        max(projected_targets),
        sum(value**2 for value in projected_targets),
        max(projected_tasks),
        sum(value**2 for value in projected_tasks),
    )


def validate_fixed_observation_group_source_index(
    index: CalvinDatasetIndex,
    group: FixedObservationGroup,
    *,
    action_horizon: int = 1,
) -> CalvinStatefulTransitionSample:
    """Rebind one audited group without materializing the full transition map."""

    if not isinstance(index, CalvinDatasetIndex):
        raise TypeError("fixed-X source validation requires a CALVIN dataset index")
    if not isinstance(group, FixedObservationGroup):
        raise TypeError("fixed-X source validation requires one audited group")
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or action_horizon <= 0
    ):
        raise ValueError("fixed-X source action horizon must be positive")

    prefix = "calvin-language-segment-"
    suffix = group.stateful_episode_key.removeprefix(prefix)
    if (
        not group.stateful_episode_key.startswith(prefix)
        or len(suffix) != 8
        or any(character < "0" or character > "9" for character in suffix)
    ):
        raise ValueError("fixed-X group has a malformed stateful episode key")
    segment_index = int(suffix)
    if segment_index >= len(index.segments):
        raise ValueError("fixed-X group references an absent language segment")
    segment = index.segments[segment_index]
    if segment.index != segment_index:
        raise ValueError("fixed-X language segment order differs from its identity")

    sample = index.stateful_transition_sample(
        segment_index,
        group.source_global_index,
        action_horizon=action_horizon,
    )
    source_episode = index.source_episode(group.source_global_index)
    if (
        sample.episode_key != group.stateful_episode_key
        or sample.sample_key != group.stateful_sample_key
        or sample.transition_index != 0
        or sample.record.global_index != group.source_global_index
        or sample.host_sample.source_global_index != group.source_global_index
        or int(source_episode.index) != group.source_episode_index
        or segment.task_key != group.source_task_key
        or hashlib.sha256(segment.instruction.encode("utf-8")).hexdigest()
        != group.source_instruction_sha256
    ):
        raise ValueError("fixed-X group differs from the immutable stateful sample")
    source_arrays = index.validated_source_frame_arrays(
        group.source_global_index,
        fields=("robot_obs", "scene_obs"),
    )
    if (
        calvin_source_state_sha256(
            source_arrays["scene_obs"],
            source_arrays["robot_obs"],
        )
        != group.source_state_sha256
    ):
        raise ValueError("fixed-X group source-state hash differs from the dataset")
    target_request = native_calvin_structural_target_request(sample)
    if target_request.source_sensor_sha256 != group.source_sensor_sha256:
        raise ValueError("fixed-X group sensor hashes differ from the dataset")
    return sample


def validate_fixed_observation_group_source(
    dataset: CalvinStatefulTransitionDataset,
    group: FixedObservationGroup,
    *,
    action_horizon: int = 1,
) -> CalvinStatefulTransitionSample:
    """Rebind one audited group through a materialized stateful key map."""

    if not isinstance(dataset, CalvinStatefulTransitionDataset):
        raise TypeError("fixed-X source validation requires the stateful CALVIN dataset")
    if not isinstance(group, FixedObservationGroup):
        raise TypeError("fixed-X source validation requires one audited group")
    locator = dataset.locator_by_key(group.stateful_sample_key)
    if (
        locator.global_index != group.source_global_index
        or dataset.index.segments[locator.segment_index].index != locator.segment_index
    ):
        raise ValueError("fixed-X group differs from its materialized stateful key map")
    sample = validate_fixed_observation_group_source_index(
        dataset.index,
        group,
        action_horizon=action_horizon,
    )
    if locator.segment_index != sample.record.task_index:
        raise ValueError("fixed-X key-map segment differs from the immutable sample")
    return sample


def build_fixed_observation_pair_plan(
    stream_plan: FrozenResetMixtureStreamPlan,
    dataset: CalvinStatefulTransitionDataset,
    audit: FixedObservationAudit,
) -> FixedObservationPairPlan:
    """Build balanced fixed-X reset pairs from the original reset source pool."""

    if not isinstance(stream_plan, FrozenResetMixtureStreamPlan):
        raise TypeError("fixed-X pairing requires a frozen reset-mixture stream")
    if stream_plan.global_batch_size != 2:
        raise ValueError("the first fixed-X trial requires global batch size two")
    if not isinstance(dataset, CalvinStatefulTransitionDataset):
        raise TypeError("fixed-X pairing requires the stateful CALVIN dataset")
    if not isinstance(audit, FixedObservationAudit) or audit.partition != "training":
        raise ValueError("fixed-X training requires the training audit partition")
    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ValueError("fixed-X pairing requires a content-addressed dataset")
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
    if dataset_identity != plan_identity:
        raise ValueError("fixed-X stream and dataset identities differ")
    if (
        audit.dataset_tree_sha256 != manifest.tree_sha256
        or audit.comparison_id != stream_plan.comparison_id
        or audit.stream_plan_sha256 != stream_plan.plan_sha256
    ):
        raise ValueError("fixed-X audit belongs to another dataset or stream")

    reset_sources = dict(
        zip(
            stream_plan.reset_sample_keys,
            stream_plan.reset_source_global_indices,
            strict=True,
        )
    )
    candidates = tuple(
        group for group in audit.groups if group.stateful_sample_key in reset_sources
    )
    if len(candidates) < stream_plan.reset_step_count:
        raise ValueError("fixed-X audit has insufficient original reset-pool groups")
    for group in candidates:
        if reset_sources[group.stateful_sample_key] != group.source_global_index:
            raise ValueError("fixed-X group source differs from the reset stream")
        validate_fixed_observation_group_source(dataset, group)

    available_tasks = frozenset(
        variant.task_key for group in candidates for variant in group.variants
    )
    available_targets = frozenset(
        variant.target_identity_key for group in candidates for variant in group.variants
    )
    remaining = list(candidates)
    task_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    selected_tasks: set[str] = set()
    selected_targets: set[str] = set()
    pairs: list[FixedObservationPair] = []
    reset_steps = tuple(
        step
        for step in range(stream_plan.total_steps)
        if stream_plan.component_for_step(step) == "reset"
    )

    for optimizer_step in reset_steps:
        scored: list[
            tuple[
                tuple[int, int, int, int, int],
                int,
                tuple[FixedObservationVariant, FixedObservationVariant],
            ]
        ] = []
        for group_index, group in enumerate(remaining):
            for pair_variants in combinations(group.variants, 2):
                scored.append(
                    (
                        _pair_balance_score(
                            pair_variants,
                            task_counts=task_counts,
                            target_counts=target_counts,
                            available_tasks=available_tasks,
                            available_targets=available_targets,
                            selected_tasks=selected_tasks,
                            selected_targets=selected_targets,
                        ),
                        group_index,
                        pair_variants,
                    )
                )
        if not scored:
            raise RuntimeError("fixed-X pairing exhausted audited candidates")
        best_balance = min(item[0] for item in scored)
        finalists = [item for item in scored if item[0] == best_balance]
        _tie, group_index, pair_variants = min(
            [
                (
                    _hash_order(
                        audit.report_artifact_sha256,
                        stream_plan.plan_sha256,
                        stream_plan.seed,
                        optimizer_step,
                        remaining[group_index].source_state_sha256,
                        pair_variants[0].task_key,
                        pair_variants[1].task_key,
                    ),
                    group_index,
                    pair_variants,
                )
                for _balance, group_index, pair_variants in finalists
            ],
            key=lambda item: item[0],
        )
        group = remaining.pop(group_index)
        global_batch = stream_plan.global_batch(optimizer_step)
        lane_ids = tuple(item.lane_id for item in global_batch.transitions)
        if len(lane_ids) != 2:
            raise RuntimeError("fixed-X global batch lost its two lane assignments")
        if (
            _hash_order(
                audit.report_artifact_sha256,
                optimizer_step,
                group.source_state_sha256,
                "rank-order",
            )[0]
            & 1
        ):
            pair_variants = (pair_variants[1], pair_variants[0])
        seed_coordinates = (
            FIXED_OBSERVATION_PAIR_PLAN_ALGORITHM,
            audit.report_artifact_sha256,
            str(optimizer_step),
            group.source_state_sha256,
        )
        pair = FixedObservationPair(
            optimizer_step=optimizer_step,
            lane_ids=(lane_ids[0], lane_ids[1]),
            group=group,
            variants=(pair_variants[0], pair_variants[1]),
            augmentation_seed=derive_subseed(
                stream_plan.seed,
                *seed_coordinates,
                "augmentation",
            ),
            flow_noise_seed=derive_subseed(
                stream_plan.seed,
                *seed_coordinates,
                "flow-noise",
            ),
            flow_timestep_seed=derive_subseed(
                stream_plan.seed,
                *seed_coordinates,
                "flow-timestep",
            ),
        )
        pairs.append(pair)
        task_counts.update(item.task_key for item in pair.variants)
        target_counts.update(item.target_identity_key for item in pair.variants)
        selected_tasks.update(item.task_key for item in pair.variants)
        selected_targets.update(item.target_identity_key for item in pair.variants)

    return FixedObservationPairPlan(
        dataset_id=stream_plan.dataset_id,
        dataset_revision=stream_plan.dataset_revision,
        dataset_manifest_sha256=stream_plan.dataset_manifest_sha256,
        comparison_id=stream_plan.comparison_id,
        seed=stream_plan.seed,
        stream_plan_sha256=stream_plan.plan_sha256,
        component_schedule_sha256=stream_plan.component_schedule_sha256,
        audit_report_file_sha256=audit.report_file_sha256,
        audit_artifact_sha256=audit.report_artifact_sha256,
        representation_split_file_sha256=(audit.representation_split_file_sha256),
        representation_split_artifact_sha256=(audit.representation_split_artifact_sha256),
        training_projection_contract_sha256=(audit.training_projection_contract_sha256),
        training_projection_payload_sha256=(audit.training_projection_payload_sha256),
        candidate_group_count=len(candidates),
        available_task_keys=tuple(sorted(available_tasks)),
        available_target_identity_keys=tuple(sorted(available_targets)),
        pairs=tuple(pairs),
    )


def apply_fixed_observation_pair(
    planned: PlannedNativeCALVINBatch,
    plan: FixedObservationPairPlan,
    dataset: CalvinStatefulTransitionDataset,
) -> PlannedNativeCALVINBatch:
    """Replace one reset shard with its audited source/prompt pair member."""

    if not isinstance(planned, PlannedNativeCALVINBatch):
        raise TypeError("fixed-X application requires a planned CALVIN batch")
    if not isinstance(plan, FixedObservationPairPlan):
        raise TypeError("fixed-X application requires its immutable plan")
    if not isinstance(dataset, CalvinStatefulTransitionDataset):
        raise TypeError("fixed-X application requires the stateful CALVIN dataset")
    if planned.task_intervention_sha256 is not None:
        raise ValueError("fixed-X pairing cannot follow another prompt intervention")
    if planned.fixed_observation_pair_sha256 is not None:
        raise ValueError("fixed-X pairing may be applied only once")
    if planned.plan_sha256 != plan.stream_plan_sha256:
        raise ValueError("planned batch and fixed-X stream differ")
    optimizer_step = planned.plan_microbatch.optimizer_step
    pair = plan.pair_for_step(optimizer_step)
    if pair is None:
        return planned
    transitions = planned.plan_microbatch.transitions
    if len(transitions) != 1 or planned.training.routing.batch_size != 1:
        raise ValueError("fixed-X two-rank trial requires one local sample per rank")
    transition = transitions[0]
    pair, variant = plan.slot_for(
        transition,
        optimizer_step=optimizer_step,
    )
    if transition.transition_index != 0:
        raise ValueError("fixed-X overlay may replace only a reset transition")
    sample = validate_fixed_observation_group_source(
        dataset,
        pair.group,
        action_horizon=dataset.action_horizon,
    )
    training = build_native_calvin_training_batch(
        (sample,),
        lane_ids=planned.training.routing.lane_ids,
        optimizer_step=optimizer_step,
        device=planned.training.controls.values.device,
        dtype=planned.training.controls.values.dtype,
        episode_keys=(f"fixed-x-reset-{optimizer_step:08d}/{pair.group.stateful_episode_key}",),
        frame_indices=(0,),
        reset=(True,),
    )
    request = training.structural_target_requests[0]
    if (
        request.source_global_index != pair.group.source_global_index
        or request.source_sensor_sha256 != pair.group.source_sensor_sha256
    ):
        raise ValueError("materialized fixed-X source differs from its audit")
    host_item = dict(training.host_items[0])
    host_item["task"] = variant.instruction
    original_host_item = planned.training.host_items[0]
    for name in ("action.lingbot", "action.lingbot_is_pad"):
        if host_item[name].shape != original_host_item[name].shape:
            raise ValueError(
                f"fixed-X replacement {name} shape differs from the native training horizon"
            )
    training = replace(
        training,
        host_items=(host_item,),
        structural_target_requests=(replace(request, task_key=variant.task_key),),
    )
    return replace(
        planned,
        training=training,
        augmentation_seeds=(pair.augmentation_seed,),
        flow_noise_seeds=(pair.flow_noise_seed,),
        flow_timestep_seeds=(pair.flow_timestep_seed,),
        fixed_observation_pair_sha256=plan.artifact_sha256,
    )
