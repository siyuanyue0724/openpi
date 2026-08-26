"""Strict offline validation for the ADR-175 matched three-arm experiment.

The validator is intentionally independent of every training executable.  It
accepts content-addressed, typed arm reports and recomputes the preregistered
acceptance gates without initializing Torch, CUDA, CALVIN, or mutable runner
state.  Every schema is exact and every comparison fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from picf_next.contracts import ContractError
from picf_next.eval.calvin_task_relevance import calvin_task_physical_relevance_inventory

ADR175_ARM_REPORT_SCHEMA = "picf-next.adr175-matched-arm-report.v1"
ADR175_VALIDATION_RESULT_SCHEMA = "picf-next.adr175-matched-three-arm-validation.v1"

ADR175_ARMS = ("lbot", "physical-set", "native-attention")
ADR175_TREATMENT_ARMS = ("physical-set", "native-attention")
ADR175_MILESTONES = (0, 250, 500, 1000, 2000)
ADR175_TOTAL_STEPS = 2000
ADR175_EXACT_STRATA_COUNT = 29
ADR175_REQUIRED_JOINTLY_POSITIVE_STRATA = 22
ADR175_MAXIMUM_ACTION_AUC_RATIO = 1.02
ADR175_NATIVE_ENTITY_SET_NONINFERIORITY_MARGIN = 0.0
_FLOAT_GATE_TOLERANCE = 1.0e-12
_CALVIN_RELEVANCE = calvin_task_physical_relevance_inventory()
ADR175_EXACT_TASK_TARGETS = tuple(
    sorted(
        (
            item.task_key,
            tuple(sorted(item.action_target_identity_keys)),
        )
        for item in _CALVIN_RELEVANCE
        if item.exact_action_target
    )
)
ADR175_AMBIGUOUS_TASKS = tuple(
    sorted(item.task_key for item in _CALVIN_RELEVANCE if not item.exact_action_target)
)
if len(ADR175_EXACT_TASK_TARGETS) != ADR175_EXACT_STRATA_COUNT or len(ADR175_AMBIGUOUS_TASKS) != 5:
    raise RuntimeError("frozen CALVIN task relevance inventory changed under ADR-175")

_SHARED_CONTRACT_FIELDS = frozenset(
    {
        "broad_support_contract_sha256",
        "broad_support_contract_file_sha256",
        "matched_arm_input_sha256",
        "dataset_contract_sha256",
        "physical_sidecar_manifest_sha256",
        "stream_plan_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
        "shared_initialization_sha256",
        "shared_optimizer_contract_sha256",
        "source_commit",
        "source_patch_sha256",
        "patched_source_sha256",
        "implementation_sha256",
        "checkpoint_contract_sha256",
        "processor_contract_sha256",
        "objective_sha256",
        "vision_geometry_sha256",
        "runtime_contract_sha256",
        "total_steps",
    }
)
_STEP_RECEIPT_FIELDS = frozenset(
    {
        "global_step",
        "execution_input_sha256",
        "sample_sha256",
        "action_target_sha256",
        "noise_sha256",
        "time_sha256",
        "prompt_sha256",
    }
)
_PARTITION_VALUES_FIELDS = frozenset({"validation", "heldout"})
_MILESTONE_FIELDS = frozenset(
    {
        "global_step",
        "posterior_adoption",
        "conditional_selectivity",
        "action_loss",
        "entity_set_score",
    }
)
_TARGET_VALIDITY_FIELDS = frozenset({"task_key", "target_valid"})
_EXACT_STRATUM_FIELDS = frozenset(
    {
        "stratum_id",
        "task_key",
        "target_identity_keys",
        "validation_score",
        "heldout_score",
        "validation_censored",
        "heldout_censored",
        "validation_sample_count",
        "heldout_sample_count",
        "validation_observable_sample_count",
        "heldout_observable_sample_count",
        "observability_receipt_sha256",
    }
)
_BOOTSTRAP_FIELDS = frozenset(
    {
        "cluster_unit",
        "cluster_count",
        "confidence_level",
        "resampling_scheme",
        "replicates",
        "seed",
        "reference_arm",
        "candidate_arm",
        "raw_estimate",
        "raw_lower_bound",
        "normalized_estimate",
        "normalized_lower_bound",
    }
)
_ARM_UNSIGNED_FIELDS = frozenset(
    {
        "schema",
        "status",
        "arm",
        "raw_report_file_sha256",
        "evaluation_evidence_sha256",
        "picf_treatment_contract_sha256",
        "shared_contract",
        "picf_graph_sha256",
        "picf_initialization_sha256",
        "exact_observability_sha256",
        "ambiguous_target_validity",
        "step_receipts",
        "milestones",
        "exact_strata",
        "heldout_selectivity_bootstrap",
    }
)
_ARM_FIELDS = frozenset((*_ARM_UNSIGNED_FIELDS, "artifact_sha256"))
_ARM_DIGEST_FIELDS = frozenset({"arm", "artifact_sha256"})
_GATE_FIELDS = frozenset({"name", "passed", "evidence"})
_VALIDATION_UNSIGNED_FIELDS = frozenset({"schema", "status", "arm_report_sha256", "gates"})
_VALIDATION_FIELDS = frozenset((*_VALIDATION_UNSIGNED_FIELDS, "artifact_sha256"))
_GATE_NAMES = (
    "exact_arm_set",
    "shared_contract_identity",
    "step_receipts_identical",
    "picf_graph_initialization_identity",
    "ambiguous_target_validity",
    "milestone_coverage",
    "separated_adoption_selectivity",
    "exact_strata_joint_support",
    "heldout_selectivity_bootstrap",
    "action_auc",
    "entity_set_improvement",
)


def canonical_json_bytes(value: object) -> bytes:
    """Return the one canonical JSON encoding used by all ADR-175 artifacts."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ContractError("ADR-175 artifact is not finite canonical JSON") from error


def canonical_sha256(value: object) -> str:
    """Hash one value after canonical JSON normalization."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(
    value: object,
    *,
    name: str,
    fields: frozenset[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be one string-keyed JSON object")
    if fields is not None and set(value) != fields:
        missing = sorted(fields.difference(value))
        extra = sorted(set(value).difference(fields))
        raise ContractError(f"{name} fields differ from schema; missing={missing}, extra={extra}")
    return value


def _list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be one JSON array")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be one nonempty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one lowercase SHA-256 digest")
    return result


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name=name)


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be one integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be at least {minimum}")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be one finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ContractError(f"{name} must be at most {maximum}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be one boolean")
    return value


def _canonical_object(value: object, *, name: str) -> dict[str, Any]:
    payload = _mapping(value, name=name)
    return json.loads(canonical_json_bytes(payload).decode("ascii"))


@dataclass(frozen=True, slots=True)
class ADR175SharedContract:
    """The identities that must be byte-equivalent across all three arms."""

    broad_support_contract_sha256: str
    broad_support_contract_file_sha256: str
    matched_arm_input_sha256: str
    dataset_contract_sha256: str
    physical_sidecar_manifest_sha256: str
    stream_plan_sha256: str
    representation_split_sha256: str
    evaluation_plan_sha256: str
    shared_initialization_sha256: str
    shared_optimizer_contract_sha256: str
    source_commit: str
    source_patch_sha256: str
    patched_source_sha256: str
    implementation_sha256: str
    checkpoint_contract_sha256: str
    processor_contract_sha256: str
    objective_sha256: str
    vision_geometry_sha256: str
    runtime_contract_sha256: str
    total_steps: int

    @classmethod
    def from_dict(cls, value: object) -> ADR175SharedContract:
        payload = _mapping(value, name="ADR-175 shared contract", fields=_SHARED_CONTRACT_FIELDS)
        contract = cls(
            broad_support_contract_sha256=_sha256(
                payload["broad_support_contract_sha256"],
                name="broad-support contract",
            ),
            broad_support_contract_file_sha256=_sha256(
                payload["broad_support_contract_file_sha256"],
                name="broad-support contract file",
            ),
            matched_arm_input_sha256=_sha256(
                payload["matched_arm_input_sha256"],
                name="matched-arm input",
            ),
            dataset_contract_sha256=_sha256(
                payload["dataset_contract_sha256"], name="dataset contract"
            ),
            physical_sidecar_manifest_sha256=_sha256(
                payload["physical_sidecar_manifest_sha256"],
                name="physical sidecar manifest",
            ),
            stream_plan_sha256=_sha256(payload["stream_plan_sha256"], name="stream plan"),
            representation_split_sha256=_sha256(
                payload["representation_split_sha256"], name="representation split"
            ),
            evaluation_plan_sha256=_sha256(
                payload["evaluation_plan_sha256"], name="evaluation plan"
            ),
            shared_initialization_sha256=_sha256(
                payload["shared_initialization_sha256"], name="shared initialization"
            ),
            shared_optimizer_contract_sha256=_sha256(
                payload["shared_optimizer_contract_sha256"],
                name="shared optimizer contract",
            ),
            source_commit=_text(payload["source_commit"], name="source commit"),
            source_patch_sha256=_sha256(payload["source_patch_sha256"], name="source patch"),
            patched_source_sha256=_sha256(payload["patched_source_sha256"], name="patched source"),
            implementation_sha256=_sha256(payload["implementation_sha256"], name="implementation"),
            checkpoint_contract_sha256=_sha256(
                payload["checkpoint_contract_sha256"], name="checkpoint contract"
            ),
            processor_contract_sha256=_sha256(
                payload["processor_contract_sha256"], name="processor contract"
            ),
            objective_sha256=_sha256(payload["objective_sha256"], name="objective"),
            vision_geometry_sha256=_sha256(
                payload["vision_geometry_sha256"], name="vision geometry"
            ),
            runtime_contract_sha256=_sha256(
                payload["runtime_contract_sha256"], name="runtime contract"
            ),
            total_steps=_integer(payload["total_steps"], name="total steps", minimum=1),
        )
        if contract.total_steps != ADR175_TOTAL_STEPS:
            raise ContractError(
                f"ADR-175 total_steps must be exactly {ADR175_TOTAL_STEPS}, "
                f"got {contract.total_steps}"
            )
        return contract

    def to_dict(self) -> dict[str, object]:
        return {
            "broad_support_contract_sha256": self.broad_support_contract_sha256,
            "broad_support_contract_file_sha256": (self.broad_support_contract_file_sha256),
            "matched_arm_input_sha256": self.matched_arm_input_sha256,
            "dataset_contract_sha256": self.dataset_contract_sha256,
            "physical_sidecar_manifest_sha256": self.physical_sidecar_manifest_sha256,
            "stream_plan_sha256": self.stream_plan_sha256,
            "representation_split_sha256": self.representation_split_sha256,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "shared_initialization_sha256": self.shared_initialization_sha256,
            "shared_optimizer_contract_sha256": self.shared_optimizer_contract_sha256,
            "source_commit": self.source_commit,
            "source_patch_sha256": self.source_patch_sha256,
            "patched_source_sha256": self.patched_source_sha256,
            "implementation_sha256": self.implementation_sha256,
            "checkpoint_contract_sha256": self.checkpoint_contract_sha256,
            "processor_contract_sha256": self.processor_contract_sha256,
            "objective_sha256": self.objective_sha256,
            "vision_geometry_sha256": self.vision_geometry_sha256,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "total_steps": self.total_steps,
        }


@dataclass(frozen=True, slots=True)
class ADR175StepReceipt:
    """One update's complete matched-randomness and matched-input receipt."""

    global_step: int
    execution_input_sha256: str
    sample_sha256: str
    action_target_sha256: str
    noise_sha256: str
    time_sha256: str
    prompt_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> ADR175StepReceipt:
        payload = _mapping(value, name="ADR-175 step receipt", fields=_STEP_RECEIPT_FIELDS)
        return cls(
            global_step=_integer(payload["global_step"], name="receipt global_step", minimum=1),
            execution_input_sha256=_sha256(
                payload["execution_input_sha256"],
                name="execution input receipt",
            ),
            sample_sha256=_sha256(payload["sample_sha256"], name="sample receipt"),
            action_target_sha256=_sha256(
                payload["action_target_sha256"], name="action target receipt"
            ),
            noise_sha256=_sha256(payload["noise_sha256"], name="noise receipt"),
            time_sha256=_sha256(payload["time_sha256"], name="time receipt"),
            prompt_sha256=_sha256(payload["prompt_sha256"], name="prompt receipt"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "execution_input_sha256": self.execution_input_sha256,
            "sample_sha256": self.sample_sha256,
            "action_target_sha256": self.action_target_sha256,
            "noise_sha256": self.noise_sha256,
            "time_sha256": self.time_sha256,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass(frozen=True, slots=True)
class ADR175PartitionValues:
    """Validation and heldout values for one explicitly named metric channel."""

    validation: float
    heldout: float

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        name: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> ADR175PartitionValues:
        payload = _mapping(value, name=name, fields=_PARTITION_VALUES_FIELDS)
        return cls(
            validation=_number(
                payload["validation"],
                name=f"{name}.validation",
                minimum=minimum,
                maximum=maximum,
            ),
            heldout=_number(
                payload["heldout"],
                name=f"{name}.heldout",
                minimum=minimum,
                maximum=maximum,
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {"validation": self.validation, "heldout": self.heldout}


@dataclass(frozen=True, slots=True)
class ADR175Milestone:
    """One frozen evaluation milestone with non-conflated representation metrics."""

    global_step: int
    posterior_adoption: ADR175PartitionValues | None
    conditional_selectivity: ADR175PartitionValues | None
    action_loss: ADR175PartitionValues
    entity_set_score: ADR175PartitionValues | None

    @classmethod
    def from_dict(cls, value: object, *, arm: str) -> ADR175Milestone:
        payload = _mapping(value, name="ADR-175 milestone", fields=_MILESTONE_FIELDS)
        if arm == "lbot":
            picf_metrics = (
                payload["posterior_adoption"],
                payload["conditional_selectivity"],
                payload["entity_set_score"],
            )
            if any(metric is not None for metric in picf_metrics):
                raise ContractError(
                    "LBOT must publish null posterior adoption, conditional selectivity, "
                    "and entity-set score"
                )
            posterior_adoption = None
            conditional_selectivity = None
            entity_set_score = None
        else:
            if any(
                payload[name] is None
                for name in (
                    "posterior_adoption",
                    "conditional_selectivity",
                    "entity_set_score",
                )
            ):
                raise ContractError("PICF treatment milestones require all PICF-only metrics")
            posterior_adoption = ADR175PartitionValues.from_dict(
                payload["posterior_adoption"],
                name="posterior adoption",
                minimum=0.0,
                maximum=1.0,
            )
            conditional_selectivity = ADR175PartitionValues.from_dict(
                payload["conditional_selectivity"],
                name="conditional selectivity",
            )
            entity_set_score = ADR175PartitionValues.from_dict(
                payload["entity_set_score"],
                name="entity set score",
                minimum=0.0,
                maximum=1.0,
            )
        return cls(
            global_step=_integer(payload["global_step"], name="milestone global_step", minimum=0),
            posterior_adoption=posterior_adoption,
            conditional_selectivity=conditional_selectivity,
            action_loss=ADR175PartitionValues.from_dict(
                payload["action_loss"],
                name="action loss",
                minimum=0.0,
            ),
            entity_set_score=entity_set_score,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "posterior_adoption": (
                None if self.posterior_adoption is None else self.posterior_adoption.to_dict()
            ),
            "conditional_selectivity": (
                None
                if self.conditional_selectivity is None
                else self.conditional_selectivity.to_dict()
            ),
            "action_loss": self.action_loss.to_dict(),
            "entity_set_score": (
                None if self.entity_set_score is None else self.entity_set_score.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class ADR175AmbiguousTargetValidity:
    """The fail-closed target-row eligibility for one ambiguous CALVIN task."""

    task_key: str
    target_valid: bool

    @classmethod
    def from_dict(cls, value: object) -> ADR175AmbiguousTargetValidity:
        payload = _mapping(
            value,
            name="ADR-175 ambiguous target validity",
            fields=_TARGET_VALIDITY_FIELDS,
        )
        return cls(
            task_key=_text(payload["task_key"], name="ambiguous task key"),
            target_valid=_boolean(payload["target_valid"], name="ambiguous target_valid"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"task_key": self.task_key, "target_valid": self.target_valid}


@dataclass(frozen=True, slots=True)
class ADR175ExactStratumOutcome:
    """One exact task/object stratum's raw validation and heldout score."""

    stratum_id: str
    task_key: str
    target_identity_keys: tuple[str, ...]
    validation_score: float
    heldout_score: float
    validation_censored: bool
    heldout_censored: bool
    validation_sample_count: int
    heldout_sample_count: int
    validation_observable_sample_count: int
    heldout_observable_sample_count: int
    observability_receipt_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> ADR175ExactStratumOutcome:
        payload = _mapping(value, name="ADR-175 exact stratum", fields=_EXACT_STRATUM_FIELDS)
        task_key = _text(payload["task_key"], name="exact stratum task_key")
        raw_keys = _list(payload["target_identity_keys"], name="target identity keys")
        identity_keys = tuple(
            _text(item, name=f"target identity key {index}") for index, item in enumerate(raw_keys)
        )
        if not identity_keys or identity_keys != tuple(sorted(set(identity_keys))):
            raise ContractError("exact stratum target identity keys must be sorted and unique")
        stratum_id = _sha256(payload["stratum_id"], name="exact stratum id")
        expected_id = canonical_sha256(
            {"task_key": task_key, "target_identity_keys": list(identity_keys)}
        )
        if stratum_id != expected_id:
            raise ContractError("exact stratum id is not derived from task and target identities")
        if task_key in ADR175_AMBIGUOUS_TASKS:
            raise ContractError("ambiguous task appeared in the exact task/object strata")
        if (task_key, identity_keys) not in ADR175_EXACT_TASK_TARGETS:
            raise ContractError("exact stratum is absent from the frozen CALVIN task protocol")
        validation_score = _number(
            payload["validation_score"],
            name="exact stratum validation score",
            minimum=0.0,
            maximum=1.0,
        )
        heldout_score = _number(
            payload["heldout_score"],
            name="exact stratum heldout score",
            minimum=0.0,
            maximum=1.0,
        )
        validation_censored = _boolean(payload["validation_censored"], name="validation censored")
        heldout_censored = _boolean(payload["heldout_censored"], name="heldout censored")
        validation_sample_count = _integer(
            payload["validation_sample_count"],
            name="validation sample count",
            minimum=1,
        )
        heldout_sample_count = _integer(
            payload["heldout_sample_count"],
            name="heldout sample count",
            minimum=1,
        )
        if validation_sample_count != 1 or heldout_sample_count != 2:
            raise ContractError("exact strata require validation=1 and heldout=2 samples")
        validation_observable = _integer(
            payload["validation_observable_sample_count"],
            name="validation observable sample count",
            minimum=0,
        )
        heldout_observable = _integer(
            payload["heldout_observable_sample_count"],
            name="heldout observable sample count",
            minimum=0,
        )
        if (
            validation_observable > validation_sample_count
            or heldout_observable > heldout_sample_count
        ):
            raise ContractError("observable sample count exceeds exact stratum sample count")
        if validation_censored is not (validation_observable < validation_sample_count):
            raise ContractError("validation censor flag differs from target observability")
        if heldout_censored is not (heldout_observable < heldout_sample_count):
            raise ContractError("heldout censor flag differs from target observability")
        if (validation_censored and validation_score != 0.0) or (
            heldout_censored and heldout_score != 0.0
        ):
            raise ContractError("censored exact stratum scores must be zero")
        return cls(
            stratum_id=stratum_id,
            task_key=task_key,
            target_identity_keys=identity_keys,
            validation_score=validation_score,
            heldout_score=heldout_score,
            validation_censored=validation_censored,
            heldout_censored=heldout_censored,
            validation_sample_count=validation_sample_count,
            heldout_sample_count=heldout_sample_count,
            validation_observable_sample_count=validation_observable,
            heldout_observable_sample_count=heldout_observable,
            observability_receipt_sha256=_sha256(
                payload["observability_receipt_sha256"],
                name="exact stratum observability receipt",
            ),
        )

    def inventory_dict(self) -> dict[str, object]:
        return {
            "stratum_id": self.stratum_id,
            "task_key": self.task_key,
            "target_identity_keys": list(self.target_identity_keys),
            "validation_censored": self.validation_censored,
            "heldout_censored": self.heldout_censored,
            "validation_sample_count": self.validation_sample_count,
            "heldout_sample_count": self.heldout_sample_count,
            "validation_observable_sample_count": (self.validation_observable_sample_count),
            "heldout_observable_sample_count": self.heldout_observable_sample_count,
            "observability_receipt_sha256": self.observability_receipt_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.inventory_dict(),
            "validation_score": self.validation_score,
            "heldout_score": self.heldout_score,
        }


@dataclass(frozen=True, slots=True)
class ADR175ClusteredBootstrap:
    """Heldout selectivity uncertainty clustered by source episode."""

    cluster_unit: str
    cluster_count: int
    confidence_level: float
    resampling_scheme: str
    replicates: int
    seed: int
    reference_arm: str
    candidate_arm: str
    raw_estimate: float
    raw_lower_bound: float
    normalized_estimate: float
    normalized_lower_bound: float

    @classmethod
    def from_dict(cls, value: object) -> ADR175ClusteredBootstrap:
        payload = _mapping(value, name="ADR-175 clustered bootstrap", fields=_BOOTSTRAP_FIELDS)
        result = cls(
            cluster_unit=_text(payload["cluster_unit"], name="bootstrap cluster_unit"),
            cluster_count=_integer(
                payload["cluster_count"], name="bootstrap cluster_count", minimum=2
            ),
            confidence_level=_number(
                payload["confidence_level"],
                name="bootstrap confidence level",
                minimum=0.0,
                maximum=1.0,
            ),
            resampling_scheme=_text(
                payload["resampling_scheme"],
                name="bootstrap resampling scheme",
            ),
            replicates=_integer(
                payload["replicates"],
                name="bootstrap replicates",
                minimum=1,
            ),
            seed=_integer(payload["seed"], name="bootstrap seed", minimum=0),
            reference_arm=_text(payload["reference_arm"], name="bootstrap reference arm"),
            candidate_arm=_text(payload["candidate_arm"], name="bootstrap candidate arm"),
            raw_estimate=_number(payload["raw_estimate"], name="raw bootstrap estimate"),
            raw_lower_bound=_number(payload["raw_lower_bound"], name="raw bootstrap lower bound"),
            normalized_estimate=_number(
                payload["normalized_estimate"], name="normalized bootstrap estimate"
            ),
            normalized_lower_bound=_number(
                payload["normalized_lower_bound"],
                name="normalized bootstrap lower bound",
            ),
        )
        if result.cluster_unit != "source_episode":
            raise ContractError("ADR-175 bootstrap must cluster by source_episode")
        if result.confidence_level.hex() != (0.95).hex():
            raise ContractError("ADR-175 bootstrap confidence level must be exactly 0.95")
        if result.resampling_scheme != "paired_global_source_episode_bayesian":
            raise ContractError("ADR-175 bootstrap resampling scheme changed")
        if result.replicates != 10_000 or result.seed != 20260816:
            raise ContractError("ADR-175 bootstrap replicate count or seed changed")
        if result.reference_arm != "physical-set" or result.candidate_arm != "native-attention":
            raise ContractError(
                "ADR-175 bootstrap must compare native-attention against physical-set"
            )
        if result.raw_lower_bound > result.raw_estimate:
            raise ContractError("raw bootstrap lower bound exceeds its estimate")
        if result.normalized_lower_bound > result.normalized_estimate:
            raise ContractError("normalized bootstrap lower bound exceeds its estimate")
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_unit": self.cluster_unit,
            "cluster_count": self.cluster_count,
            "confidence_level": self.confidence_level,
            "resampling_scheme": self.resampling_scheme,
            "replicates": self.replicates,
            "seed": self.seed,
            "reference_arm": self.reference_arm,
            "candidate_arm": self.candidate_arm,
            "raw_estimate": self.raw_estimate,
            "raw_lower_bound": self.raw_lower_bound,
            "normalized_estimate": self.normalized_estimate,
            "normalized_lower_bound": self.normalized_lower_bound,
        }


@dataclass(frozen=True, slots=True)
class ADR175ArmReport:
    """One complete, content-addressed ADR-175 arm report."""

    schema: str
    status: str
    arm: str
    raw_report_file_sha256: str
    evaluation_evidence_sha256: str
    picf_treatment_contract_sha256: str | None
    shared_contract: ADR175SharedContract
    picf_graph_sha256: str | None
    picf_initialization_sha256: str | None
    exact_observability_sha256: str | None
    ambiguous_target_validity: tuple[ADR175AmbiguousTargetValidity, ...]
    step_receipts: tuple[ADR175StepReceipt, ...]
    milestones: tuple[ADR175Milestone, ...]
    exact_strata: tuple[ADR175ExactStratumOutcome, ...] | None
    heldout_selectivity_bootstrap: ADR175ClusteredBootstrap | None
    artifact_sha256: str

    @classmethod
    def _from_unsigned_mapping(cls, value: object) -> ADR175ArmReport:
        payload = _mapping(value, name="ADR-175 arm report", fields=_ARM_UNSIGNED_FIELDS)
        schema = _text(payload["schema"], name="ADR-175 arm report schema")
        status = _text(payload["status"], name="ADR-175 arm report status")
        arm = _text(payload["arm"], name="ADR-175 arm")
        if schema != ADR175_ARM_REPORT_SCHEMA:
            raise ContractError("ADR-175 arm report schema changed")
        if status != "COMPLETE":
            raise ContractError("ADR-175 arm report status must be COMPLETE")
        if arm not in ADR175_ARMS:
            raise ContractError(f"unknown ADR-175 arm: {arm!r}")
        raw_report_file_sha256 = _sha256(
            payload["raw_report_file_sha256"],
            name="raw-report file",
        )
        evaluation_evidence_sha256 = _sha256(
            payload["evaluation_evidence_sha256"],
            name="evaluation evidence",
        )
        treatment_contract_sha256 = _optional_sha256(
            payload["picf_treatment_contract_sha256"],
            name="PICF treatment contract",
        )

        graph_sha256 = _optional_sha256(payload["picf_graph_sha256"], name="PICF graph")
        initialization_sha256 = _optional_sha256(
            payload["picf_initialization_sha256"], name="PICF initialization"
        )
        exact_observability_sha256 = _optional_sha256(
            payload["exact_observability_sha256"], name="exact observability"
        )
        if arm == "lbot":
            if (
                graph_sha256 is not None
                or initialization_sha256 is not None
                or treatment_contract_sha256 is not None
                or exact_observability_sha256 is not None
            ):
                raise ContractError(
                    "LBOT arm must not publish PICF graph, initialization, treatment, or "
                    "observability artifacts"
                )
        elif (
            graph_sha256 is None
            or initialization_sha256 is None
            or treatment_contract_sha256 is None
            or exact_observability_sha256 is None
        ):
            raise ContractError(
                "PICF treatment arms require graph, initialization, and observability digests"
            )

        target_validity = tuple(
            ADR175AmbiguousTargetValidity.from_dict(item)
            for item in _list(
                payload["ambiguous_target_validity"],
                name="ambiguous target validity rows",
            )
        )
        if tuple(item.task_key for item in target_validity) != ADR175_AMBIGUOUS_TASKS:
            raise ContractError("ambiguous target validity inventory or order changed")
        if any(item.target_valid for item in target_validity):
            raise ContractError("all five ambiguous tasks must publish target_valid=false")

        receipts = tuple(
            ADR175StepReceipt.from_dict(item)
            for item in _list(payload["step_receipts"], name="step receipts")
        )
        expected_receipt_steps = tuple(range(1, ADR175_TOTAL_STEPS + 1))
        if tuple(item.global_step for item in receipts) != expected_receipt_steps:
            raise ContractError("step receipts must cover every update 1..2000 exactly once")

        milestones = tuple(
            ADR175Milestone.from_dict(item, arm=arm)
            for item in _list(payload["milestones"], name="evaluation milestones")
        )
        if tuple(item.global_step for item in milestones) != ADR175_MILESTONES:
            raise ContractError("ADR-175 milestones must be exactly 0/250/500/1000/2000")

        if arm == "lbot":
            if payload["exact_strata"] is not None:
                raise ContractError("LBOT exact_strata must be explicitly null")
            if payload["heldout_selectivity_bootstrap"] is not None:
                raise ContractError("LBOT heldout selectivity bootstrap must be explicitly null")
            exact_strata = None
            bootstrap = None
        else:
            exact_strata = tuple(
                ADR175ExactStratumOutcome.from_dict(item)
                for item in _list(payload["exact_strata"], name="exact task/object strata")
            )
            if len(exact_strata) != ADR175_EXACT_STRATA_COUNT:
                raise ContractError(
                    f"ADR-175 requires exactly {ADR175_EXACT_STRATA_COUNT} exact strata"
                )
            stratum_ids = tuple(item.stratum_id for item in exact_strata)
            task_keys = tuple(item.task_key for item in exact_strata)
            if stratum_ids != tuple(sorted(set(stratum_ids))):
                raise ContractError("exact strata must be sorted by unique stratum_id")
            if len(set(task_keys)) != ADR175_EXACT_STRATA_COUNT:
                raise ContractError("ADR-175 exact strata must contain 29 unique task keys")
            observed_inventory = tuple(
                sorted((item.task_key, item.target_identity_keys) for item in exact_strata)
            )
            if observed_inventory != ADR175_EXACT_TASK_TARGETS:
                raise ContractError("exact strata do not equal the frozen 29-task CALVIN inventory")

            if arm == "physical-set":
                if payload["heldout_selectivity_bootstrap"] is not None:
                    raise ContractError(
                        "physical-set bootstrap must be null; it is the reference arm"
                    )
                bootstrap = None
            else:
                if payload["heldout_selectivity_bootstrap"] is None:
                    raise ContractError("native-attention requires a heldout selectivity bootstrap")
                bootstrap = ADR175ClusteredBootstrap.from_dict(
                    payload["heldout_selectivity_bootstrap"]
                )
                if (
                    bootstrap.reference_arm != "physical-set"
                    or bootstrap.candidate_arm != "native-attention"
                ):
                    raise ContractError(
                        "heldout bootstrap must compare native-attention against physical-set"
                    )

        return cls(
            schema=schema,
            status=status,
            arm=arm,
            raw_report_file_sha256=raw_report_file_sha256,
            evaluation_evidence_sha256=evaluation_evidence_sha256,
            picf_treatment_contract_sha256=treatment_contract_sha256,
            shared_contract=ADR175SharedContract.from_dict(payload["shared_contract"]),
            picf_graph_sha256=graph_sha256,
            picf_initialization_sha256=initialization_sha256,
            exact_observability_sha256=exact_observability_sha256,
            ambiguous_target_validity=target_validity,
            step_receipts=receipts,
            milestones=milestones,
            exact_strata=exact_strata,
            heldout_selectivity_bootstrap=bootstrap,
            artifact_sha256="",
        )

    @classmethod
    def from_unsigned_dict(cls, value: object) -> ADR175ArmReport:
        """Parse, normalize, and content-address one unsigned arm report."""

        unsigned = cls._from_unsigned_mapping(value)
        return replace(unsigned, artifact_sha256=canonical_sha256(unsigned.to_unsigned_dict()))

    @classmethod
    def from_dict(cls, value: object) -> ADR175ArmReport:
        payload = _mapping(value, name="ADR-175 signed arm report", fields=_ARM_FIELDS)
        artifact_sha256 = _sha256(payload["artifact_sha256"], name="ADR-175 arm report artifact")
        unsigned = {name: payload[name] for name in _ARM_UNSIGNED_FIELDS}
        report = cls._from_unsigned_mapping(unsigned)
        expected_sha256 = canonical_sha256(report.to_unsigned_dict())
        if artifact_sha256 != expected_sha256:
            raise ContractError("ADR-175 arm report artifact SHA-256 changed")
        return replace(report, artifact_sha256=artifact_sha256)

    def to_unsigned_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "arm": self.arm,
            "raw_report_file_sha256": self.raw_report_file_sha256,
            "evaluation_evidence_sha256": self.evaluation_evidence_sha256,
            "picf_treatment_contract_sha256": self.picf_treatment_contract_sha256,
            "shared_contract": self.shared_contract.to_dict(),
            "picf_graph_sha256": self.picf_graph_sha256,
            "picf_initialization_sha256": self.picf_initialization_sha256,
            "exact_observability_sha256": self.exact_observability_sha256,
            "ambiguous_target_validity": [
                item.to_dict() for item in self.ambiguous_target_validity
            ],
            "step_receipts": [item.to_dict() for item in self.step_receipts],
            "milestones": [item.to_dict() for item in self.milestones],
            "exact_strata": (
                None
                if self.exact_strata is None
                else [item.to_dict() for item in self.exact_strata]
            ),
            "heldout_selectivity_bootstrap": (
                None
                if self.heldout_selectivity_bootstrap is None
                else self.heldout_selectivity_bootstrap.to_dict()
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.to_unsigned_dict(), "artifact_sha256": self.artifact_sha256}


def seal_adr175_arm_report(value: object) -> dict[str, object]:
    """Return the normalized, content-addressed JSON form of one arm report."""

    return ADR175ArmReport.from_unsigned_dict(value).to_dict()


@dataclass(frozen=True, slots=True)
class ADR175ArmDigest:
    arm: str
    artifact_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> ADR175ArmDigest:
        payload = _mapping(value, name="ADR-175 arm digest", fields=_ARM_DIGEST_FIELDS)
        arm = _text(payload["arm"], name="validation arm digest arm")
        if arm not in ADR175_ARMS:
            raise ContractError("validation result contains an unknown arm digest")
        return cls(
            arm=arm,
            artifact_sha256=_sha256(payload["artifact_sha256"], name="validation arm artifact"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"arm": self.arm, "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class ADR175GateResult:
    """One independently recomputed acceptance gate."""

    name: str
    passed: bool
    evidence_json: str

    @classmethod
    def create(cls, name: str, evidence: Mapping[str, object]) -> ADR175GateResult:
        if name not in _GATE_NAMES:
            raise ContractError(f"unknown ADR-175 gate: {name!r}")
        normalized = _canonical_object(evidence, name=f"{name} gate evidence")
        return cls(
            name=name,
            passed=True,
            evidence_json=canonical_json_bytes(normalized).decode("ascii"),
        )

    @classmethod
    def from_dict(cls, value: object) -> ADR175GateResult:
        payload = _mapping(value, name="ADR-175 gate result", fields=_GATE_FIELDS)
        name = _text(payload["name"], name="ADR-175 gate name")
        if name not in _GATE_NAMES:
            raise ContractError(f"unknown ADR-175 gate: {name!r}")
        if _boolean(payload["passed"], name=f"{name} gate passed") is not True:
            raise ContractError("a PASS validation artifact cannot contain a failed gate")
        evidence = _canonical_object(payload["evidence"], name=f"{name} gate evidence")
        return cls(
            name=name,
            passed=True,
            evidence_json=canonical_json_bytes(evidence).decode("ascii"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence": json.loads(self.evidence_json),
        }


@dataclass(frozen=True, slots=True)
class ADR175ValidationResult:
    """Content-addressed PASS evidence for the complete three-arm gate."""

    schema: str
    status: str
    arm_report_sha256: tuple[ADR175ArmDigest, ...]
    gates: tuple[ADR175GateResult, ...]
    artifact_sha256: str

    @classmethod
    def from_unsigned_dict(cls, value: object) -> ADR175ValidationResult:
        payload = _mapping(
            value,
            name="ADR-175 validation result",
            fields=_VALIDATION_UNSIGNED_FIELDS,
        )
        schema = _text(payload["schema"], name="ADR-175 validation result schema")
        status = _text(payload["status"], name="ADR-175 validation result status")
        if schema != ADR175_VALIDATION_RESULT_SCHEMA or status != "PASS":
            raise ContractError("ADR-175 validation result schema or status changed")
        digests = tuple(
            ADR175ArmDigest.from_dict(item)
            for item in _list(payload["arm_report_sha256"], name="validation arm digests")
        )
        if tuple(item.arm for item in digests) != ADR175_ARMS:
            raise ContractError("validation arm digest inventory or order changed")
        gates = tuple(
            ADR175GateResult.from_dict(item)
            for item in _list(payload["gates"], name="ADR-175 gates")
        )
        if tuple(item.name for item in gates) != _GATE_NAMES:
            raise ContractError("ADR-175 validation gate inventory or order changed")
        result = cls(
            schema=schema,
            status=status,
            arm_report_sha256=digests,
            gates=gates,
            artifact_sha256="",
        )
        return replace(result, artifact_sha256=canonical_sha256(result.to_unsigned_dict()))

    @classmethod
    def from_dict(cls, value: object) -> ADR175ValidationResult:
        payload = _mapping(
            value,
            name="signed ADR-175 validation result",
            fields=_VALIDATION_FIELDS,
        )
        artifact_sha256 = _sha256(
            payload["artifact_sha256"], name="ADR-175 validation result artifact"
        )
        unsigned = {name: payload[name] for name in _VALIDATION_UNSIGNED_FIELDS}
        result = cls.from_unsigned_dict(unsigned)
        if result.artifact_sha256 != artifact_sha256:
            raise ContractError("ADR-175 validation result artifact SHA-256 changed")
        return replace(result, artifact_sha256=artifact_sha256)

    def to_unsigned_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "arm_report_sha256": [item.to_dict() for item in self.arm_report_sha256],
            "gates": [item.to_dict() for item in self.gates],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.to_unsigned_dict(), "artifact_sha256": self.artifact_sha256}


def _normalized_action_auc(report: ADR175ArmReport, partition: str) -> float:
    values = [getattr(milestone.action_loss, partition) for milestone in report.milestones]
    weighted = math.fsum(
        (right.global_step - left.global_step) * (left_value + right_value) / 2.0
        for left, right, left_value, right_value in zip(
            report.milestones[:-1],
            report.milestones[1:],
            values[:-1],
            values[1:],
            strict=True,
        )
    )
    return weighted / ADR175_TOTAL_STEPS


def _parse_reports(
    reports: Sequence[ADR175ArmReport | Mapping[str, object]],
) -> dict[str, ADR175ArmReport]:
    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise ContractError("ADR-175 validation requires one sequence of three arm reports")
    if len(reports) != len(ADR175_ARMS):
        raise ContractError("ADR-175 validation requires exactly three arm reports")
    parsed = tuple(
        report if isinstance(report, ADR175ArmReport) else ADR175ArmReport.from_dict(report)
        for report in reports
    )
    by_arm = {report.arm: report for report in parsed}
    if len(by_arm) != len(parsed) or set(by_arm) != set(ADR175_ARMS):
        raise ContractError("ADR-175 arm set must be exactly lbot/physical-set/native-attention")
    return by_arm


def validate_adr175_matched_three_arm(
    reports: Sequence[ADR175ArmReport | Mapping[str, object]],
) -> ADR175ValidationResult:
    """Recompute every preregistered ADR-175 acceptance gate.

    Physical-set is the representation reference and native-attention is the
    candidate for the 22/29 exact-stratum and heldout clustered-selectivity
    gates.  LBOT is used only for action AUC.  Both PICF arms must improve their
    own entity-set score from step 0, and native-attention must be non-inferior
    to physical-set under the explicitly frozen margin.
    """

    by_arm = _parse_reports(reports)
    ordered = tuple(by_arm[arm] for arm in ADR175_ARMS)
    lbot = by_arm["lbot"]
    physical = by_arm["physical-set"]
    native = by_arm["native-attention"]

    gates: list[ADR175GateResult] = [
        ADR175GateResult.create("exact_arm_set", {"arms": list(ADR175_ARMS)})
    ]

    if any(report.shared_contract != lbot.shared_contract for report in ordered[1:]):
        raise ContractError("dataset/stream/split/eval/init/shared optimizer contract differs")
    gates.append(
        ADR175GateResult.create(
            "shared_contract_identity",
            {
                "shared_contract_sha256": canonical_sha256(lbot.shared_contract.to_dict()),
                "total_steps": lbot.shared_contract.total_steps,
            },
        )
    )

    if any(report.step_receipts != lbot.step_receipts for report in ordered[1:]):
        raise ContractError("per-step sample/action/noise/time/prompt receipts differ across arms")
    gates.append(
        ADR175GateResult.create(
            "step_receipts_identical",
            {
                "receipt_count": len(lbot.step_receipts),
                "step_receipts_sha256": canonical_sha256(
                    [item.to_dict() for item in lbot.step_receipts]
                ),
            },
        )
    )

    if physical.picf_graph_sha256 != native.picf_graph_sha256:
        raise ContractError("physical-set and native-attention PICF graph digests differ")
    if physical.picf_initialization_sha256 != native.picf_initialization_sha256:
        raise ContractError("physical-set and native-attention PICF initialization digests differ")
    if physical.picf_treatment_contract_sha256 != native.picf_treatment_contract_sha256:
        raise ContractError("physical-set and native-attention treatment contracts differ")
    if physical.exact_observability_sha256 != native.exact_observability_sha256:
        raise ContractError("physical-set and native-attention observability receipts differ")
    gates.append(
        ADR175GateResult.create(
            "picf_graph_initialization_identity",
            {
                "picf_graph_sha256": physical.picf_graph_sha256,
                "picf_initialization_sha256": physical.picf_initialization_sha256,
                "picf_treatment_contract_sha256": (physical.picf_treatment_contract_sha256),
                "exact_observability_sha256": physical.exact_observability_sha256,
            },
        )
    )

    gates.append(
        ADR175GateResult.create(
            "ambiguous_target_validity",
            {
                "tasks": list(ADR175_AMBIGUOUS_TASKS),
                "target_valid": False,
                "verified_arm_count": len(ordered),
            },
        )
    )
    gates.append(
        ADR175GateResult.create(
            "milestone_coverage",
            {"global_steps": list(ADR175_MILESTONES)},
        )
    )
    gates.append(
        ADR175GateResult.create(
            "separated_adoption_selectivity",
            {
                "channels": ["posterior_adoption", "conditional_selectivity"],
                "available_arms": list(ADR175_TREATMENT_ARMS),
                "lbot_values": None,
                "milestone_count_per_arm": len(ADR175_MILESTONES),
            },
        )
    )

    if physical.exact_strata is None or native.exact_strata is None:
        raise ContractError("PICF treatment arms did not publish exact task/object strata")
    reference_inventory = tuple(item.inventory_dict() for item in physical.exact_strata)
    if tuple(item.inventory_dict() for item in native.exact_strata) != reference_inventory:
        raise ContractError("exact task/object stratum inventory differs across treatment arms")
    physical_strata = {item.stratum_id: item for item in physical.exact_strata}
    jointly_positive = tuple(
        item.stratum_id
        for item in native.exact_strata
        if not item.validation_censored
        and not item.heldout_censored
        and item.validation_score > physical_strata[item.stratum_id].validation_score
        and item.heldout_score > physical_strata[item.stratum_id].heldout_score
    )
    censored = tuple(
        item.stratum_id
        for item in native.exact_strata
        if item.validation_censored or item.heldout_censored
    )
    if len(jointly_positive) < ADR175_REQUIRED_JOINTLY_POSITIVE_STRATA:
        raise ContractError("native-attention exact support is below 22/29 jointly positive strata")
    gates.append(
        ADR175GateResult.create(
            "exact_strata_joint_support",
            {
                "jointly_positive_count": len(jointly_positive),
                "required_count": ADR175_REQUIRED_JOINTLY_POSITIVE_STRATA,
                "total_count": ADR175_EXACT_STRATA_COUNT,
                "reference_arm": "physical-set",
                "candidate_arm": "native-attention",
                "jointly_positive_stratum_ids": list(jointly_positive),
                "censored_stratum_ids": list(censored),
            },
        )
    )

    bootstrap = native.heldout_selectivity_bootstrap
    if bootstrap is None:
        raise ContractError("native-attention heldout selectivity bootstrap is missing")
    if bootstrap.raw_lower_bound <= 0.0 or bootstrap.normalized_lower_bound <= 0.0:
        raise ContractError(
            "native-attention heldout raw and normalized clustered bootstrap lower bounds "
            "must both be positive"
        )
    gates.append(
        ADR175GateResult.create(
            "heldout_selectivity_bootstrap",
            bootstrap.to_dict(),
        )
    )

    action_aucs = {
        arm: {
            partition: _normalized_action_auc(by_arm[arm], partition)
            for partition in ("validation", "heldout")
        }
        for arm in ADR175_ARMS
    }
    if any(action_aucs["lbot"][partition] <= 0.0 for partition in ("validation", "heldout")):
        raise ContractError("LBOT action AUC must be positive in both partitions")
    action_ratios: dict[str, dict[str, float]] = {}
    for arm in ADR175_TREATMENT_ARMS:
        action_ratios[arm] = {}
        for partition in ("validation", "heldout"):
            ratio = action_aucs[arm][partition] / action_aucs["lbot"][partition]
            action_ratios[arm][partition] = ratio
            if ratio > ADR175_MAXIMUM_ACTION_AUC_RATIO and not math.isclose(
                ratio,
                ADR175_MAXIMUM_ACTION_AUC_RATIO,
                rel_tol=_FLOAT_GATE_TOLERANCE,
                abs_tol=_FLOAT_GATE_TOLERANCE,
            ):
                raise ContractError(
                    f"{arm} {partition} action AUC is more than 2% worse than matched LBOT"
                )
    gates.append(
        ADR175GateResult.create(
            "action_auc",
            {
                "normalized_auc": action_aucs,
                "ratio_to_lbot": action_ratios,
                "maximum_ratio": ADR175_MAXIMUM_ACTION_AUC_RATIO,
            },
        )
    )

    entity_self_improvement: dict[str, dict[str, float]] = {}
    for arm in ADR175_TREATMENT_ARMS:
        initial = by_arm[arm].milestones[0].entity_set_score
        final = by_arm[arm].milestones[-1].entity_set_score
        if initial is None or final is None:
            raise ContractError(f"{arm} is missing treatment-only entity-set metrics")
        entity_self_improvement[arm] = {}
        for partition in ("validation", "heldout"):
            delta = getattr(final, partition) - getattr(initial, partition)
            entity_self_improvement[arm][partition] = delta
            if delta <= 0.0:
                raise ContractError(
                    f"{arm} final {partition} entity-set score did not improve from step 0"
                )

    physical_final = physical.milestones[-1].entity_set_score
    native_final = native.milestones[-1].entity_set_score
    if physical_final is None or native_final is None:
        raise ContractError("treatment final entity-set metrics are missing")
    native_minus_physical: dict[str, float] = {}
    for partition in ("validation", "heldout"):
        difference = getattr(native_final, partition) - getattr(physical_final, partition)
        native_minus_physical[partition] = difference
        if difference < -ADR175_NATIVE_ENTITY_SET_NONINFERIORITY_MARGIN and not math.isclose(
            difference,
            -ADR175_NATIVE_ENTITY_SET_NONINFERIORITY_MARGIN,
            rel_tol=_FLOAT_GATE_TOLERANCE,
            abs_tol=_FLOAT_GATE_TOLERANCE,
        ):
            raise ContractError(
                f"native-attention final {partition} entity-set score is inferior to physical-set"
            )
    gates.append(
        ADR175GateResult.create(
            "entity_set_improvement",
            {
                "global_step": ADR175_TOTAL_STEPS,
                "self_improvement_from_step0": entity_self_improvement,
                "native_minus_physical_set": native_minus_physical,
                "native_noninferiority_margin": (ADR175_NATIVE_ENTITY_SET_NONINFERIORITY_MARGIN),
            },
        )
    )

    unsigned_result = {
        "schema": ADR175_VALIDATION_RESULT_SCHEMA,
        "status": "PASS",
        "arm_report_sha256": [
            {"arm": report.arm, "artifact_sha256": report.artifact_sha256} for report in ordered
        ],
        "gates": [gate.to_dict() for gate in gates],
    }
    return ADR175ValidationResult.from_unsigned_dict(unsigned_result)


__all__ = [
    "ADR175_AMBIGUOUS_TASKS",
    "ADR175_ARMS",
    "ADR175_ARM_REPORT_SCHEMA",
    "ADR175_EXACT_STRATA_COUNT",
    "ADR175_EXACT_TASK_TARGETS",
    "ADR175_MILESTONES",
    "ADR175_NATIVE_ENTITY_SET_NONINFERIORITY_MARGIN",
    "ADR175_REQUIRED_JOINTLY_POSITIVE_STRATA",
    "ADR175_TOTAL_STEPS",
    "ADR175_VALIDATION_RESULT_SCHEMA",
    "ADR175ArmReport",
    "ADR175ValidationResult",
    "canonical_json_bytes",
    "canonical_sha256",
    "seal_adr175_arm_report",
    "validate_adr175_matched_three_arm",
]
