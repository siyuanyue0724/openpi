"""Strict, hash-addressed assembly for one PICF training hypothesis.

The recipe is intentionally narrower than a generic configuration framework.
Every accepted field changes model mathematics, training credit, optimizer
semantics or experiment authorization. Unknown fields fail closed so a typo
cannot silently create a different experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CALVIN_ACTION_AXES,
    CALVIN_CONTROL_HZ,
    CALVIN_DEBUG_DATASET_ID,
    CALVIN_DEBUG_REVISION,
    CALVIN_HOST_IMAGE_KEYS,
    CALVIN_STATE_AXES,
)
from picf_next.data.calvin_geometry_schema import (
    CALVIN_GEOMETRY_SIDECAR_SCHEMA,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
)
from picf_next.data.calvin_normalization import (
    CALVIN_NORMALIZATION_SCHEMA,
    CALVIN_NORMALIZATION_SCHEMAS,
    validate_calvin_normalization_artifact,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_SUPERVISION_SCHEMA,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    read_sha256_verified_file_beneath,
)
from picf_next.geometry import PhysicalGeometryContract
from picf_next.models.binding_loss import (
    BindingLossConfig,
    MultimodalBindingCriterion,
    TemporalAddressBindingCriterion,
)
from picf_next.models.core import PICFCore, PICFCoreConfig
from picf_next.models.discovery import (
    ObjectDiscoveryConfig,
    ObjectExistenceCalibration,
)
from picf_next.models.dynamics_loss import (
    ObjectDynamicsCriterion,
    ObjectDynamicsLossConfig,
    ObjectGeometryOvershootingConfig,
    ObjectGeometryOvershootingCriterion,
)
from picf_next.models.evidence import ModalityProjectionSpec
from picf_next.models.objective import PICFObjective, PICFObjectiveConfig
from picf_next.models.set_loss import ObjectSetCriterion, ObjectSetLossConfig
from picf_next.models.temporal import TemporalFilterConfig

if TYPE_CHECKING:
    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.calvin_rollout_targets import CalvinPhysicalGeometryProvider
    from picf_next.hosts.molmoact2_training import (
        CalvinStatefulLossTargetBuilder,
        MolmoAct2PICFTrainingConfig,
    )


RECIPE_SCHEMA = "picf-next.training-recipe.v2"
_MOLMO_VISION_PATCH_MODALITY = "molmo_vision_patch"
_MAXIMUM_RECIPE_ARTIFACT_BYTES = 64 * 1024 * 1024
_AUTHORIZATION_STAGES = {
    "M3_structural_probe",
    "M4_action_adoption",
    "M5_three_seed_abc",
    "M6_long_train",
}


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return cast(Mapping[str, object], value)


def _exact(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != fields:
        missing = sorted(fields - set(payload))
        unknown = sorted(set(payload) - fields)
        raise ValueError(f"{name} fields differ from schema; missing={missing}, unknown={unknown}")
    return payload


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be one finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be one finite number")
    return converted


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return converted


def _positive_float(value: object, name: str) -> float:
    converted = _nonnegative_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of nonempty strings")
    return tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))


def _finite_float_pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two finite numbers")
    return (
        _finite_float(value[0], f"{name}[0]"),
        _finite_float(value[1], f"{name}[1]"),
    )


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"{name} must be a repository-relative path")
    return text


def _verified_artifact_bytes(
    root: Path,
    relative: str,
    expected_sha256: str,
    name: str,
) -> bytes:
    try:
        return read_sha256_verified_file_beneath(
            root,
            relative,
            expected_sha256=expected_sha256,
            maximum_bytes=_MAXIMUM_RECIPE_ARTIFACT_BYTES,
        )
    except FileNotFoundError as error:
        raise ContractError(f"{name} is absent: {relative}") from error
    except ContractError as error:
        raise ContractError(
            f"training artifact SHA-256 changed or its path is unsafe: {relative}"
        ) from error


def _artifact_json(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ContractError(f"{name} must contain one JSON object")
    return value


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class TrainingAuthorization:
    stage: str
    max_optimizer_steps: int
    long_training_authorized: bool
    hypothesis_status: str

    def __post_init__(self) -> None:
        if self.stage not in _AUTHORIZATION_STAGES:
            raise ValueError(f"unsupported training authorization stage {self.stage!r}")
        _positive_int(self.max_optimizer_steps, "max_optimizer_steps")
        if not isinstance(self.long_training_authorized, bool):
            raise ValueError("long_training_authorized must be boolean")
        _text(self.hypothesis_status, "hypothesis_status")
        if self.long_training_authorized != (self.stage == "M6_long_train"):
            raise ValueError("only an M6_long_train recipe may authorize long training")


@dataclass(frozen=True, slots=True)
class CalvinDatasetRecipe:
    dataset_id: str
    dataset_revision: str
    split_name: str
    control_hz: int
    state_axes: tuple[str, ...]
    action_axes: tuple[str, ...]
    image_keys: tuple[str, ...]
    action_horizon: int
    sample_plan_algorithm: str

    def __post_init__(self) -> None:
        if self.dataset_id != CALVIN_DEBUG_DATASET_ID:
            raise ValueError("the M3 recipe must use the audited CALVIN debug dataset")
        if self.dataset_revision != CALVIN_DEBUG_REVISION:
            raise ValueError("the CALVIN dataset revision differs from the audited archive")
        if self.split_name != "training":
            raise ValueError("the M3 recipe must use the CALVIN training split")
        if self.control_hz != CALVIN_CONTROL_HZ:
            raise ValueError("the CALVIN control rate differs from the data adapter")
        if self.state_axes != CALVIN_STATE_AXES:
            raise ValueError("the CALVIN state axes differ from the data adapter")
        if self.action_axes != CALVIN_ACTION_AXES:
            raise ValueError("the CALVIN action axes differ from the data adapter")
        if self.image_keys != CALVIN_HOST_IMAGE_KEYS:
            raise ValueError("the CALVIN image keys differ from the MolmoAct2 bridge")
        if self.action_horizon != 10:
            raise ValueError("the M3 action horizon is preregistered as exactly 10")
        if self.sample_plan_algorithm != "sha256-epoch-sort.v1":
            raise ValueError("the stateful sample-plan algorithm changed")

    @property
    def state_dim(self) -> int:
        return len(self.state_axes)

    @property
    def action_dim(self) -> int:
        return len(self.action_axes)


@dataclass(frozen=True, slots=True)
class MolmoAct2PolicyRecipe:
    n_obs_steps: int
    action_mode: str
    inference_action_mode: str
    setup_type: str
    control_mode: str
    normalize_language: bool
    add_setup_tokens: bool
    add_control_tokens: bool
    normalize_gripper: bool
    num_state_tokens: int
    expected_max_action_dim: int
    num_flow_timesteps: int
    flow_matching_cutoff: float
    flow_matching_time_offset: float
    flow_matching_time_scale: float
    flow_matching_beta_alpha: float
    flow_matching_beta_beta: float
    num_inference_steps: int
    mask_action_dim_padding: bool
    enable_inference_cuda_graph: bool
    enable_lora_vlm: bool
    enable_lora_action_expert: bool
    enable_knowledge_insulation: bool
    freeze_embedding: bool
    train_action_expert_only: bool
    gradient_checkpointing: bool
    model_dtype: str
    optimizer_lr: float
    optimizer_action_expert_lr: float
    optimizer_betas: tuple[float, float]
    optimizer_eps: float
    optimizer_weight_decay: float
    scheduler_warmup_steps: int
    scheduler_decay_steps: int
    scheduler_decay_lr: float

    def __post_init__(self) -> None:
        if self.n_obs_steps != 1:
            raise ValueError("PICF streaming consumes exactly one current host observation")
        if self.action_mode != "continuous" or self.inference_action_mode != "continuous":
            raise ValueError("the causal CALVIN path requires continuous MolmoAct2 actions")
        for name in ("setup_type", "control_mode"):
            _text(getattr(self, name), name)
        for name in ("normalize_language", "add_setup_tokens", "add_control_tokens"):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must remain enabled for the frozen prompt contract")
        if self.normalize_gripper is not False:
            raise ValueError("CALVIN gripper axes are excluded by the exact normalization mask")
        if self.num_state_tokens != 256 or self.expected_max_action_dim != 32:
            raise ValueError("released MolmoAct2 state/action token dimensions changed")
        if self.num_flow_timesteps != 8 or self.num_inference_steps != 8:
            raise ValueError(
                "the M3 MolmoAct2 flow schedule is preregistered as exactly eight steps"
            )
        expected_flow = {
            "flow_matching_cutoff": 1.0,
            "flow_matching_time_offset": 0.001,
            "flow_matching_time_scale": 0.999,
            "flow_matching_beta_alpha": 1.0,
            "flow_matching_beta_beta": 1.5,
        }
        for name, expected in expected_flow.items():
            if float(getattr(self, name)) != expected:
                raise ValueError(f"{name} differs from the pinned official continuous objective")
        if self.mask_action_dim_padding is not True:
            raise ValueError("padded MolmoAct2 action dimensions must be masked")
        if self.enable_inference_cuda_graph is not False:
            raise ValueError(
                "M3 keeps CUDA graphs disabled until parity and deployment probes pass"
            )
        if any(
            getattr(self, name)
            for name in (
                "enable_lora_vlm",
                "enable_lora_action_expert",
                "enable_knowledge_insulation",
            )
        ):
            raise ValueError(
                "M3 does not mix LoRA or knowledge insulation into the PICF hypothesis"
            )
        if not self.freeze_embedding or not self.train_action_expert_only:
            raise ValueError(
                "M3 trains only the action expert, PICF core and typed residual adapters"
            )
        if not self.gradient_checkpointing:
            raise ValueError("2xA100-40G production requires activation checkpointing")
        if self.model_dtype != "bfloat16":
            raise ValueError("the 2xA100 training precision is fixed to bfloat16")
        _positive_float(self.optimizer_lr, "optimizer_lr")
        _positive_float(self.optimizer_action_expert_lr, "optimizer_action_expert_lr")
        if len(self.optimizer_betas) != 2 or any(
            not 0.0 <= float(value) < 1.0 for value in self.optimizer_betas
        ):
            raise ValueError("optimizer_betas must contain two values in [0, 1)")
        _positive_float(self.optimizer_eps, "optimizer_eps")
        _nonnegative_float(self.optimizer_weight_decay, "optimizer_weight_decay")
        if (
            not isinstance(self.scheduler_warmup_steps, int)
            or isinstance(self.scheduler_warmup_steps, bool)
            or self.scheduler_warmup_steps < 0
        ):
            raise ValueError("scheduler_warmup_steps must be non-negative")
        _positive_int(self.scheduler_decay_steps, "scheduler_decay_steps")
        if self.scheduler_warmup_steps >= self.scheduler_decay_steps:
            raise ValueError("scheduler warmup must end before scheduler decay")
        _positive_float(self.scheduler_decay_lr, "scheduler_decay_lr")


@dataclass(frozen=True, slots=True)
class TrainingArtifactsRecipe:
    dataset_file_manifest_path: str
    dataset_file_manifest_sha256: str
    dataset_tree_sha256: str
    normalization_path: str
    normalization_file_sha256: str
    normalization_payload_sha256: str
    physical_sidecar_manifest_path: str
    physical_sidecar_manifest_sha256: str
    geometry_sidecar_manifest_path: str
    geometry_sidecar_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_file_manifest_path",
            "normalization_path",
            "physical_sidecar_manifest_path",
            "geometry_sidecar_manifest_path",
        ):
            _relative_path(getattr(self, name), name)
        for name in (
            "dataset_file_manifest_sha256",
            "dataset_tree_sha256",
            "normalization_file_sha256",
            "normalization_payload_sha256",
            "physical_sidecar_manifest_sha256",
            "geometry_sidecar_manifest_sha256",
        ):
            _sha256(getattr(self, name), name)

    def validate_repository(
        self,
        root: str | Path,
        *,
        dataset: CalvinDatasetRecipe,
        geometry_contract: PhysicalGeometryContract,
    ) -> dict[str, object]:
        repository = Path(root).resolve()
        dataset_manifest_payload = _verified_artifact_bytes(
            repository,
            self.dataset_file_manifest_path,
            self.dataset_file_manifest_sha256,
            "dataset file manifest",
        )
        normalization_payload = _verified_artifact_bytes(
            repository,
            self.normalization_path,
            self.normalization_file_sha256,
            "normalization artifact",
        )
        physical_payload = _verified_artifact_bytes(
            repository,
            self.physical_sidecar_manifest_path,
            self.physical_sidecar_manifest_sha256,
            "physical-supervision manifest",
        )
        geometry_payload = _verified_artifact_bytes(
            repository,
            self.geometry_sidecar_manifest_path,
            self.geometry_sidecar_manifest_sha256,
            "geometry-sidecar manifest",
        )

        dataset_manifest = DatasetFileManifest.from_dict(
            _artifact_json(dataset_manifest_payload, "dataset file manifest")
        )
        if dataset_manifest.tree_sha256 != self.dataset_tree_sha256:
            raise ContractError("dataset tree SHA-256 differs from the training recipe")
        if (
            dataset_manifest.dataset_id != dataset.dataset_id
            or dataset_manifest.dataset_revision != dataset.dataset_revision
            or dataset_manifest.split_name != dataset.split_name
        ):
            raise ContractError("dataset file manifest identity differs from the training recipe")
        normalization = _artifact_json(normalization_payload, "normalization artifact")
        validate_calvin_normalization_artifact(normalization)
        if normalization.get("schema") not in CALVIN_NORMALIZATION_SCHEMAS:
            raise ContractError("normalization schema changed")
        if (
            normalization.get("schema") == CALVIN_NORMALIZATION_SCHEMA
            and normalization.get("dataset_tree_sha256") != dataset_manifest.tree_sha256
        ):
            raise ContractError("normalization dataset tree differs from the training recipe")
        if normalization["artifact_sha256"] != self.normalization_payload_sha256:
            raise ContractError("normalization payload SHA-256 differs from the recipe")
        manifests: dict[str, Mapping[str, object]] = {
            "physical": _artifact_json(physical_payload, "physical manifest"),
            "geometry": _artifact_json(geometry_payload, "geometry manifest"),
        }
        physical = manifests["physical"]
        geometry = manifests["geometry"]
        if physical.get("schema") != CALVIN_PHYSICAL_SUPERVISION_SCHEMA:
            raise ContractError("physical-supervision manifest schema changed")
        if geometry.get("schema") != CALVIN_GEOMETRY_SIDECAR_SCHEMA:
            raise ContractError("geometry-sidecar manifest schema changed")
        for payload in (normalization, physical, geometry):
            if (
                payload.get("dataset_id") != dataset.dataset_id
                or payload.get("dataset_revision") != dataset.dataset_revision
            ):
                raise ContractError("training artifact dataset identity differs from the recipe")
        for manifest in (physical, geometry):
            if manifest.get("split_name") != dataset.split_name:
                raise ContractError("training sidecar split differs from the recipe")
            if manifest.get("runtime_input") is not False:
                raise ContractError("loss-only sidecars must never be runtime model input")
            if manifest.get("geometry_contract_sha256") != geometry_contract.fingerprint:
                raise ContractError("sidecar geometry contract differs from the recipe")
        if physical.get("task_conditioned") is not False:
            raise ContractError("physical supervision must remain task-independent")
        if physical.get("global_indices_sha256") != geometry.get("global_indices_sha256"):
            raise ContractError("physical and geometry sidecars cover different frames")
        if physical.get("frame_count") != geometry.get("frame_count"):
            raise ContractError("physical and geometry sidecar frame counts differ")
        if physical.get("object_record_count") != geometry.get("object_record_count"):
            raise ContractError("physical and geometry sidecar object counts differ")
        sample_count = normalization.get("sample_count")
        frame_count = physical.get("frame_count")
        if (
            not isinstance(sample_count, int)
            or not isinstance(frame_count, int)
            or sample_count < frame_count
        ):
            raise ContractError("normalization and sidecar coverage counts are inconsistent")
        return {
            "dataset_file_count": len(dataset_manifest.files),
            "dataset_total_size_bytes": dataset_manifest.total_size_bytes,
            "dataset_tree_sha256": dataset_manifest.tree_sha256,
            "geometry_object_records": geometry["object_record_count"],
            "normalization_samples": sample_count,
            "physical_frames": frame_count,
            "normalization_payload_sha256": normalization["artifact_sha256"],
        }


@dataclass(frozen=True, slots=True)
class HostRecipe:
    name: str
    checkpoint_id: str
    checkpoint_revision: str
    source_commit: str
    trainer_commit: str
    dense_modality: str
    action_dim: int

    def __post_init__(self) -> None:
        for name in ("name", "checkpoint_id", "dense_modality"):
            _text(getattr(self, name), name)
        for name in ("checkpoint_revision", "source_commit", "trainer_commit"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be one full lowercase git commit")
        _positive_int(self.action_dim, "host action_dim")
        if self.name != "MolmoAct2" or self.dense_modality != _MOLMO_VISION_PATCH_MODALITY:
            raise ValueError("recipe v2 currently assembles only the audited MolmoAct2 host")


@dataclass(frozen=True, slots=True)
class GeometryOvershootingRecipe:
    config: ObjectGeometryOvershootingConfig
    horizons: tuple[int, ...]
    schedule_algorithm: str
    fraction_numerator: int
    fraction_denominator: int

    def __post_init__(self) -> None:
        if self.horizons != tuple(sorted(set(self.horizons))) or any(
            not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
            for horizon in self.horizons
        ):
            raise ValueError("overshooting horizons must be unique increasing positive integers")
        if self.config.weight > 0.0 and not self.horizons:
            raise ValueError("active geometry overshooting requires explicit horizons")
        if self.config.weight == 0.0 and self.horizons:
            raise ValueError("inactive geometry overshooting cannot declare horizons")
        if self.schedule_algorithm != "every_optimizer_step.v1":
            raise ValueError("recipe v2 supports only deterministic every-step overshooting")
        _positive_int(self.fraction_denominator, "overshooting fraction denominator")
        if (
            not isinstance(self.fraction_numerator, int)
            or isinstance(self.fraction_numerator, bool)
            or not 0 < self.fraction_numerator <= self.fraction_denominator
        ):
            raise ValueError("overshooting fraction must lie in (0, 1]")
        if self.fraction_numerator != self.fraction_denominator:
            raise ValueError("every_optimizer_step schedule requires an exact unit fraction")


@dataclass(frozen=True, slots=True)
class OptimizerRecipe:
    host_parameter_groups: str
    picf_core_lr: float
    gradient_clip_norm: float
    require_explicit_flow_randomness: bool

    def __post_init__(self) -> None:
        if self.host_parameter_groups != "MolmoAct2Policy.get_optim_params":
            raise ValueError("host optimizer groups must come from the official policy")
        for name in ("picf_core_lr", "gradient_clip_norm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.require_explicit_flow_randomness is not True:
            raise ValueError("matched training requires explicit checkpointable flow randomness")


@dataclass(frozen=True, slots=True)
class PICFTrainingRecipe:
    authorization: TrainingAuthorization
    dataset: CalvinDatasetRecipe
    host: HostRecipe
    policy: MolmoAct2PolicyRecipe
    artifacts: TrainingArtifactsRecipe
    geometry_contract: PhysicalGeometryContract
    core_config: PICFCoreConfig
    set_loss_config: ObjectSetLossConfig
    dynamics_loss_config: ObjectDynamicsLossConfig
    binding_loss_config: BindingLossConfig
    objective_config: PICFObjectiveConfig
    geometry_overshooting: GeometryOvershootingRecipe
    optimizer: OptimizerRecipe
    detached_context_frames: int
    gradient_transitions: int

    def __post_init__(self) -> None:
        if self.dataset.action_dim != self.host.action_dim:
            raise ValueError("CALVIN and host action dimensions differ")
        if self.core_config.temporal.action_dim != self.host.action_dim:
            raise ValueError("host and temporal action dimensions differ")
        if self.core_config.temporal.geometry_contract != self.geometry_contract:
            raise ValueError("recipe and core geometry contracts differ")
        if self.geometry_contract != CALVIN_OBJECT_GEOMETRY_CONTRACT:
            raise ValueError("CALVIN recipe geometry differs from the v3 physical sidecar")
        if self.host.dense_modality not in self.core_config.dense_token_dims:
            raise ValueError("host dense modality is absent from the PICF projection specs")
        if self.geometry_overshooting.config.weight > 0.0:
            if self.objective_config.dynamics_weight <= 0.0:
                raise ValueError("active overshooting requires an active dynamics family")
            if self.dynamics_loss_config.geometry_nll_weight <= 0.0:
                raise ValueError("active overshooting requires calibrated geometry NLL")
        if (
            self.objective_config.require_temporal_positive_pairs
            and self.objective_config.binding_weight <= 0.0
        ):
            raise ValueError("required temporal identity credit needs binding loss")
        if self.binding_loss_config.objective != "sigmoid":
            raise ValueError(
                "runtime identity association requires calibrated sigmoid address binding"
            )
        temporal = self.core_config.temporal
        if not math.isclose(
            temporal.association_address_temperature,
            self.binding_loss_config.temperature,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("runtime address temperature must equal temporal binding temperature")
        if not math.isclose(
            temporal.association_address_logit_bias,
            self.binding_loss_config.effective_logit_bias,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("runtime address bias must equal temporal binding logit bias")
        if self.set_loss_config.geometry_weight <= 0.0:
            raise ValueError(
                "physical geometry discovery must train its robust calibrated objective"
            )
        if self.detached_context_frames != 0 or self.gradient_transitions != 1:
            raise ValueError("stateful production training requires the audited 0+1 exposure path")
        if self.optimizer.gradient_clip_norm != 1.0:
            raise ValueError("PICF and official host gradient clipping must share norm 1.0")
        if (
            self.core_config.temporal.empty_bank_birth_to_clutter_prior_odds
            < self.core_config.temporal.recurrent_birth_to_clutter_prior_odds
        ):
            raise ValueError("empty-bank birth odds cannot be below recurrent birth odds")
        expected_validation = "metadata" if self.authorization.stage == "M6_long_train" else "full"
        if self.core_config.runtime_validation != expected_validation:
            raise ValueError(
                f"{self.authorization.stage} requires {expected_validation} runtime validation"
            )
        if (
            self.authorization.stage == "M3_structural_probe"
            and self.geometry_overshooting.horizons != (1, 2)
        ):
            raise ValueError("M3 overshooting horizons are preregistered as exactly (1, 2)")
        if self.authorization.stage == "M3_structural_probe" and (
            self.policy.scheduler_warmup_steps != 10
            or self.policy.scheduler_decay_steps != self.authorization.max_optimizer_steps
        ):
            raise ValueError("M3 scheduler is preregistered as 10 warmup steps over 200 steps")

    @property
    def recipe_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    def assert_optimizer_steps_authorized(self, optimizer_steps: int) -> None:
        requested = _positive_int(optimizer_steps, "requested optimizer_steps")
        if requested > self.authorization.max_optimizer_steps:
            raise PermissionError(
                f"recipe authorizes at most {self.authorization.max_optimizer_steps} "
                f"optimizer steps, requested {requested}"
            )

    def build_core(self) -> PICFCore:
        return self.core_config.build()

    def build_objective(self) -> PICFObjective:
        binding = MultimodalBindingCriterion(self.binding_loss_config)
        return PICFObjective(
            self.objective_config,
            set_criterion=ObjectSetCriterion(config=self.set_loss_config),
            dynamics_criterion=ObjectDynamicsCriterion(self.dynamics_loss_config),
            geometry_overshooting_criterion=ObjectGeometryOvershootingCriterion(
                self.geometry_overshooting.config
            ),
            binding_criterion=binding,
            temporal_binding_criterion=TemporalAddressBindingCriterion(self.binding_loss_config),
        )

    def build_host_training_config(self) -> MolmoAct2PICFTrainingConfig:
        from picf_next.hosts.molmoact2_training import MolmoAct2PICFTrainingConfig

        return MolmoAct2PICFTrainingConfig(
            detached_context_frames=self.detached_context_frames,
            gradient_transitions=self.gradient_transitions,
            picf_core_lr=self.optimizer.picf_core_lr,
            require_explicit_flow_randomness=(self.optimizer.require_explicit_flow_randomness),
        )

    def build_calvin_loss_target_builder(
        self,
        index: CalvinDatasetIndex,
        physical_sidecar: CalvinPhysicalSupervisionSidecar,
        geometry_provider: CalvinPhysicalGeometryProvider,
    ) -> CalvinStatefulLossTargetBuilder:
        """Bind visible/lifecycle and future-geometry targets to this recipe.

        The concrete builders validate their own CALVIN index, sidecar and
        provider types. Keeping this import lazy allows strict local recipe
        inspection without installing the full MolmoAct2 runtime.
        """

        from picf_next.hosts.molmoact2_training import (
            CalvinGeometryOvershootingTargetBuilder,
            CalvinVisibleObjectTargetBuilder,
            compose_calvin_loss_target_builders,
        )

        builders: list[CalvinStatefulLossTargetBuilder] = [
            CalvinVisibleObjectTargetBuilder(physical_sidecar)
        ]
        if self.geometry_overshooting.config.weight > 0.0:
            builders.append(
                CalvinGeometryOvershootingTargetBuilder(
                    index,
                    geometry_contract=self.geometry_contract,
                    geometry_provider=geometry_provider,
                    maximum_horizon=max(self.geometry_overshooting.horizons),
                    supervised_horizons=self.geometry_overshooting.horizons,
                )
            )
        return compose_calvin_loss_target_builders(*builders)

    def validate_repository_artifacts(self, root: str | Path) -> dict[str, object]:
        return self.artifacts.validate_repository(
            root,
            dataset=self.dataset,
            geometry_contract=self.geometry_contract,
        )

    def local_preflight_report(self, root: str | Path | None = None) -> dict[str, object]:
        core = self.build_core()
        objective = self.build_objective()
        core_parameters = sum(parameter.numel() for parameter in core.parameters())
        objective_parameters = sum(parameter.numel() for parameter in objective.parameters())
        report: dict[str, object] = {
            "action_dim": self.host.action_dim,
            "action_horizon": self.dataset.action_horizon,
            "authorization_stage": self.authorization.stage,
            "core_parameters": core_parameters,
            "dataset_id": self.dataset.dataset_id,
            "dataset_revision": self.dataset.dataset_revision,
            "flow_timesteps": self.policy.num_flow_timesteps,
            "geometry_contract_sha256": self.geometry_contract.fingerprint,
            "long_training_authorized": self.authorization.long_training_authorized,
            "maximum_optimizer_steps": self.authorization.max_optimizer_steps,
            "object_address_dim": self.core_config.object_address_dim,
            "object_value_dim": self.core_config.object_value_dim,
            "objective_parameters": objective_parameters,
            "overshooting_horizons": self.geometry_overshooting.horizons,
            "posterior_capacity": self.core_config.posterior_capacity,
            "recipe_sha256": self.recipe_sha256,
            "schema": RECIPE_SCHEMA,
            "state_dim": self.dataset.state_dim,
        }
        if root is not None:
            report["artifacts"] = self.validate_repository_artifacts(root)
        return report

    def to_dict(self) -> dict[str, object]:
        discovery = self.core_config.discovery
        temporal = self.core_config.temporal
        return {
            "artifacts": {
                "dataset_file_manifest_path": self.artifacts.dataset_file_manifest_path,
                "dataset_file_manifest_sha256": (self.artifacts.dataset_file_manifest_sha256),
                "dataset_tree_sha256": self.artifacts.dataset_tree_sha256,
                "geometry_sidecar_manifest_path": (self.artifacts.geometry_sidecar_manifest_path),
                "geometry_sidecar_manifest_sha256": (
                    self.artifacts.geometry_sidecar_manifest_sha256
                ),
                "normalization_file_sha256": self.artifacts.normalization_file_sha256,
                "normalization_path": self.artifacts.normalization_path,
                "normalization_payload_sha256": (self.artifacts.normalization_payload_sha256),
                "physical_sidecar_manifest_path": (self.artifacts.physical_sidecar_manifest_path),
                "physical_sidecar_manifest_sha256": (
                    self.artifacts.physical_sidecar_manifest_sha256
                ),
            },
            "authorization": {
                "hypothesis_status": self.authorization.hypothesis_status,
                "long_training_authorized": self.authorization.long_training_authorized,
                "max_optimizer_steps": self.authorization.max_optimizer_steps,
                "stage": self.authorization.stage,
            },
            "dataset": {
                "action_axes": self.dataset.action_axes,
                "action_horizon": self.dataset.action_horizon,
                "control_hz": self.dataset.control_hz,
                "dataset_id": self.dataset.dataset_id,
                "dataset_revision": self.dataset.dataset_revision,
                "image_keys": self.dataset.image_keys,
                "sample_plan_algorithm": self.dataset.sample_plan_algorithm,
                "split_name": self.dataset.split_name,
                "state_axes": self.dataset.state_axes,
            },
            "core": {
                "binding_dim": self.core_config.binding_dim,
                "discovery": {
                    "dropout": discovery.dropout,
                    "ffn_multiplier": discovery.ffn_multiplier,
                    "hidden_dim": discovery.hidden_dim,
                    "initial_variance": discovery.initial_variance,
                    "minimum_variance": discovery.minimum_variance,
                    "num_heads": discovery.num_heads,
                    "num_layers": discovery.num_layers,
                    "num_queries": discovery.num_queries,
                    "unmatched_query_weight": (
                        discovery.existence_calibration.unmatched_query_weight
                    ),
                },
                "modalities": [
                    {
                        "geometry_dim": spec.geometry_dim,
                        "name": spec.name,
                        "require_single_active_group": spec.require_single_active_group,
                        "token_dim": spec.token_dim,
                    }
                    for spec in self.core_config.modality_specs
                ],
                "posterior_capacity": self.core_config.posterior_capacity,
                "runtime_validation": self.core_config.runtime_validation,
                "state": {
                    "address_dim": temporal.address_dim,
                    "content_dim": temporal.content_dim,
                },
                "temporal": {
                    "association_address_logit_bias": (temporal.association_address_logit_bias),
                    "association_address_temperature": (temporal.association_address_temperature),
                    "dropout": temporal.dropout,
                    "empty_bank_birth_to_clutter_prior_odds": (
                        temporal.empty_bank_birth_to_clutter_prior_odds
                    ),
                    "ffn_multiplier": temporal.ffn_multiplier,
                    "hidden_dim": temporal.hidden_dim,
                    "initial_detection_probability": temporal.initial_detection_probability,
                    "initial_process_variance": temporal.initial_process_variance,
                    "initial_survival_probability": temporal.initial_survival_probability,
                    "minimum_variance": temporal.minimum_variance,
                    "num_heads": temporal.num_heads,
                    "num_layers": temporal.num_layers,
                    "recurrent_birth_to_clutter_prior_odds": (
                        temporal.recurrent_birth_to_clutter_prior_odds
                    ),
                    "reference_delta_t_s": temporal.reference_delta_t_s,
                },
            },
            "geometry_contract": self.geometry_contract.to_dict(),
            "host": {
                "action_dim": self.host.action_dim,
                "checkpoint_id": self.host.checkpoint_id,
                "checkpoint_revision": self.host.checkpoint_revision,
                "dense_modality": self.host.dense_modality,
                "name": self.host.name,
                "source_commit": self.host.source_commit,
                "trainer_commit": self.host.trainer_commit,
            },
            "policy": {
                "action_mode": self.policy.action_mode,
                "add_control_tokens": self.policy.add_control_tokens,
                "add_setup_tokens": self.policy.add_setup_tokens,
                "control_mode": self.policy.control_mode,
                "enable_inference_cuda_graph": self.policy.enable_inference_cuda_graph,
                "enable_knowledge_insulation": self.policy.enable_knowledge_insulation,
                "enable_lora_action_expert": self.policy.enable_lora_action_expert,
                "enable_lora_vlm": self.policy.enable_lora_vlm,
                "expected_max_action_dim": self.policy.expected_max_action_dim,
                "flow_matching_beta_alpha": self.policy.flow_matching_beta_alpha,
                "flow_matching_beta_beta": self.policy.flow_matching_beta_beta,
                "flow_matching_cutoff": self.policy.flow_matching_cutoff,
                "flow_matching_time_offset": self.policy.flow_matching_time_offset,
                "flow_matching_time_scale": self.policy.flow_matching_time_scale,
                "freeze_embedding": self.policy.freeze_embedding,
                "gradient_checkpointing": self.policy.gradient_checkpointing,
                "inference_action_mode": self.policy.inference_action_mode,
                "mask_action_dim_padding": self.policy.mask_action_dim_padding,
                "model_dtype": self.policy.model_dtype,
                "n_obs_steps": self.policy.n_obs_steps,
                "normalize_gripper": self.policy.normalize_gripper,
                "normalize_language": self.policy.normalize_language,
                "num_flow_timesteps": self.policy.num_flow_timesteps,
                "num_inference_steps": self.policy.num_inference_steps,
                "num_state_tokens": self.policy.num_state_tokens,
                "optimizer_lr": self.policy.optimizer_lr,
                "optimizer_action_expert_lr": self.policy.optimizer_action_expert_lr,
                "optimizer_betas": self.policy.optimizer_betas,
                "optimizer_eps": self.policy.optimizer_eps,
                "optimizer_weight_decay": self.policy.optimizer_weight_decay,
                "scheduler_decay_lr": self.policy.scheduler_decay_lr,
                "scheduler_decay_steps": self.policy.scheduler_decay_steps,
                "scheduler_warmup_steps": self.policy.scheduler_warmup_steps,
                "setup_type": self.policy.setup_type,
                "train_action_expert_only": self.policy.train_action_expert_only,
            },
            "objective": {
                "binding": {
                    "logit_bias": self.binding_loss_config.logit_bias,
                    "minimum_object_mass": self.binding_loss_config.minimum_object_mass,
                    "objective": self.binding_loss_config.objective,
                    "temperature": self.binding_loss_config.temperature,
                },
                "dynamics": {
                    "content_cosine_weight": self.dynamics_loss_config.content_cosine_weight,
                    "geometry_nll_weight": self.dynamics_loss_config.geometry_nll_weight,
                    "probability_epsilon": self.dynamics_loss_config.probability_epsilon,
                    "survival_weight": self.dynamics_loss_config.survival_weight,
                    "visibility_weight": self.dynamics_loss_config.visibility_weight,
                },
                "geometry_overshooting": {
                    "fraction_denominator": self.geometry_overshooting.fraction_denominator,
                    "fraction_numerator": self.geometry_overshooting.fraction_numerator,
                    "horizons": self.geometry_overshooting.horizons,
                    "schedule_algorithm": self.geometry_overshooting.schedule_algorithm,
                    "weight": self.geometry_overshooting.config.weight,
                },
                "set": {
                    "address_cosine_weight": self.set_loss_config.address_cosine_weight,
                    "content_cosine_weight": self.set_loss_config.content_cosine_weight,
                    "existence_weight": self.set_loss_config.existence_weight,
                    "geometry_weight": self.set_loss_config.geometry_weight,
                    "localization_confidence_weight": (
                        self.set_loss_config.localization_confidence_weight
                    ),
                    "ownership_ce_weight": self.set_loss_config.ownership_ce_weight,
                    "ownership_dice_weight": self.set_loss_config.ownership_dice_weight,
                },
                "weights": {
                    "action": self.objective_config.action_weight,
                    "binding": self.objective_config.binding_weight,
                    "dynamics": self.objective_config.dynamics_weight,
                    "require_temporal_positive_pairs": (
                        self.objective_config.require_temporal_positive_pairs
                    ),
                    "set": self.objective_config.set_weight,
                },
            },
            "optimizer": {
                "gradient_clip_norm": self.optimizer.gradient_clip_norm,
                "host_parameter_groups": self.optimizer.host_parameter_groups,
                "picf_core_lr": self.optimizer.picf_core_lr,
                "require_explicit_flow_randomness": (
                    self.optimizer.require_explicit_flow_randomness
                ),
            },
            "schema": RECIPE_SCHEMA,
            "streaming": {
                "detached_context_frames": self.detached_context_frames,
                "gradient_transitions": self.gradient_transitions,
            },
        }


def load_training_recipe(path: str | Path) -> PICFTrainingRecipe:
    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"training recipe is not valid JSON: {source}") from error
    return training_recipe_from_dict(payload)


def training_recipe_from_dict(value: object) -> PICFTrainingRecipe:
    payload = _exact(
        value,
        "training recipe",
        {
            "artifacts",
            "authorization",
            "core",
            "dataset",
            "geometry_contract",
            "host",
            "objective",
            "optimizer",
            "policy",
            "schema",
            "streaming",
        },
    )
    if payload["schema"] != RECIPE_SCHEMA:
        raise ValueError("unsupported training recipe schema")

    authorization_payload = _exact(
        payload["authorization"],
        "authorization",
        {"hypothesis_status", "long_training_authorized", "max_optimizer_steps", "stage"},
    )
    authorization = TrainingAuthorization(
        stage=_text(authorization_payload["stage"], "authorization.stage"),
        max_optimizer_steps=_positive_int(
            authorization_payload["max_optimizer_steps"],
            "authorization.max_optimizer_steps",
        ),
        long_training_authorized=_boolean(
            authorization_payload["long_training_authorized"],
            "authorization.long_training_authorized",
        ),
        hypothesis_status=_text(
            authorization_payload["hypothesis_status"],
            "authorization.hypothesis_status",
        ),
    )
    dataset_payload = _exact(
        payload["dataset"],
        "dataset",
        {
            "action_axes",
            "action_horizon",
            "control_hz",
            "dataset_id",
            "dataset_revision",
            "image_keys",
            "sample_plan_algorithm",
            "split_name",
            "state_axes",
        },
    )
    dataset = CalvinDatasetRecipe(
        dataset_id=_text(dataset_payload["dataset_id"], "dataset.dataset_id"),
        dataset_revision=_text(dataset_payload["dataset_revision"], "dataset.dataset_revision"),
        split_name=_text(dataset_payload["split_name"], "dataset.split_name"),
        control_hz=_positive_int(dataset_payload["control_hz"], "dataset.control_hz"),
        state_axes=_text_tuple(dataset_payload["state_axes"], "dataset.state_axes"),
        action_axes=_text_tuple(dataset_payload["action_axes"], "dataset.action_axes"),
        image_keys=_text_tuple(dataset_payload["image_keys"], "dataset.image_keys"),
        action_horizon=_positive_int(dataset_payload["action_horizon"], "dataset.action_horizon"),
        sample_plan_algorithm=_text(
            dataset_payload["sample_plan_algorithm"], "dataset.sample_plan_algorithm"
        ),
    )
    host_payload = _exact(
        payload["host"],
        "host",
        {
            "action_dim",
            "checkpoint_id",
            "checkpoint_revision",
            "dense_modality",
            "name",
            "source_commit",
            "trainer_commit",
        },
    )
    host = HostRecipe(
        name=_text(host_payload["name"], "host.name"),
        checkpoint_id=_text(host_payload["checkpoint_id"], "host.checkpoint_id"),
        checkpoint_revision=_text(host_payload["checkpoint_revision"], "host.checkpoint_revision"),
        source_commit=_text(host_payload["source_commit"], "host.source_commit"),
        trainer_commit=_text(host_payload["trainer_commit"], "host.trainer_commit"),
        dense_modality=_text(host_payload["dense_modality"], "host.dense_modality"),
        action_dim=_positive_int(host_payload["action_dim"], "host.action_dim"),
    )
    policy_payload = _exact(
        payload["policy"],
        "policy",
        {
            "action_mode",
            "add_control_tokens",
            "add_setup_tokens",
            "control_mode",
            "enable_inference_cuda_graph",
            "enable_knowledge_insulation",
            "enable_lora_action_expert",
            "enable_lora_vlm",
            "expected_max_action_dim",
            "flow_matching_beta_alpha",
            "flow_matching_beta_beta",
            "flow_matching_cutoff",
            "flow_matching_time_offset",
            "flow_matching_time_scale",
            "freeze_embedding",
            "gradient_checkpointing",
            "inference_action_mode",
            "mask_action_dim_padding",
            "model_dtype",
            "n_obs_steps",
            "normalize_gripper",
            "normalize_language",
            "num_flow_timesteps",
            "num_inference_steps",
            "num_state_tokens",
            "optimizer_lr",
            "optimizer_action_expert_lr",
            "optimizer_betas",
            "optimizer_eps",
            "optimizer_weight_decay",
            "scheduler_decay_lr",
            "scheduler_decay_steps",
            "scheduler_warmup_steps",
            "setup_type",
            "train_action_expert_only",
        },
    )
    policy = MolmoAct2PolicyRecipe(
        n_obs_steps=_positive_int(policy_payload["n_obs_steps"], "policy.n_obs_steps"),
        action_mode=_text(policy_payload["action_mode"], "policy.action_mode"),
        inference_action_mode=_text(
            policy_payload["inference_action_mode"], "policy.inference_action_mode"
        ),
        setup_type=_text(policy_payload["setup_type"], "policy.setup_type"),
        control_mode=_text(policy_payload["control_mode"], "policy.control_mode"),
        normalize_language=_boolean(
            policy_payload["normalize_language"], "policy.normalize_language"
        ),
        add_setup_tokens=_boolean(policy_payload["add_setup_tokens"], "policy.add_setup_tokens"),
        add_control_tokens=_boolean(
            policy_payload["add_control_tokens"], "policy.add_control_tokens"
        ),
        normalize_gripper=_boolean(policy_payload["normalize_gripper"], "policy.normalize_gripper"),
        num_state_tokens=_positive_int(
            policy_payload["num_state_tokens"], "policy.num_state_tokens"
        ),
        expected_max_action_dim=_positive_int(
            policy_payload["expected_max_action_dim"], "policy.expected_max_action_dim"
        ),
        num_flow_timesteps=_positive_int(
            policy_payload["num_flow_timesteps"], "policy.num_flow_timesteps"
        ),
        flow_matching_cutoff=_positive_float(
            policy_payload["flow_matching_cutoff"], "policy.flow_matching_cutoff"
        ),
        flow_matching_time_offset=_positive_float(
            policy_payload["flow_matching_time_offset"], "policy.flow_matching_time_offset"
        ),
        flow_matching_time_scale=_positive_float(
            policy_payload["flow_matching_time_scale"], "policy.flow_matching_time_scale"
        ),
        flow_matching_beta_alpha=_positive_float(
            policy_payload["flow_matching_beta_alpha"], "policy.flow_matching_beta_alpha"
        ),
        flow_matching_beta_beta=_positive_float(
            policy_payload["flow_matching_beta_beta"], "policy.flow_matching_beta_beta"
        ),
        num_inference_steps=_positive_int(
            policy_payload["num_inference_steps"], "policy.num_inference_steps"
        ),
        mask_action_dim_padding=_boolean(
            policy_payload["mask_action_dim_padding"], "policy.mask_action_dim_padding"
        ),
        enable_inference_cuda_graph=_boolean(
            policy_payload["enable_inference_cuda_graph"],
            "policy.enable_inference_cuda_graph",
        ),
        enable_lora_vlm=_boolean(policy_payload["enable_lora_vlm"], "policy.enable_lora_vlm"),
        enable_lora_action_expert=_boolean(
            policy_payload["enable_lora_action_expert"], "policy.enable_lora_action_expert"
        ),
        enable_knowledge_insulation=_boolean(
            policy_payload["enable_knowledge_insulation"],
            "policy.enable_knowledge_insulation",
        ),
        freeze_embedding=_boolean(policy_payload["freeze_embedding"], "policy.freeze_embedding"),
        train_action_expert_only=_boolean(
            policy_payload["train_action_expert_only"], "policy.train_action_expert_only"
        ),
        gradient_checkpointing=_boolean(
            policy_payload["gradient_checkpointing"], "policy.gradient_checkpointing"
        ),
        model_dtype=_text(policy_payload["model_dtype"], "policy.model_dtype"),
        optimizer_lr=_positive_float(policy_payload["optimizer_lr"], "policy.optimizer_lr"),
        optimizer_action_expert_lr=_positive_float(
            policy_payload["optimizer_action_expert_lr"],
            "policy.optimizer_action_expert_lr",
        ),
        optimizer_betas=_finite_float_pair(
            policy_payload["optimizer_betas"], "policy.optimizer_betas"
        ),
        optimizer_eps=_positive_float(policy_payload["optimizer_eps"], "policy.optimizer_eps"),
        optimizer_weight_decay=_nonnegative_float(
            policy_payload["optimizer_weight_decay"], "policy.optimizer_weight_decay"
        ),
        scheduler_warmup_steps=_nonnegative_int(
            policy_payload["scheduler_warmup_steps"], "policy.scheduler_warmup_steps"
        ),
        scheduler_decay_steps=_positive_int(
            policy_payload["scheduler_decay_steps"], "policy.scheduler_decay_steps"
        ),
        scheduler_decay_lr=_positive_float(
            policy_payload["scheduler_decay_lr"], "policy.scheduler_decay_lr"
        ),
    )
    artifacts_payload = _exact(
        payload["artifacts"],
        "artifacts",
        {
            "dataset_file_manifest_path",
            "dataset_file_manifest_sha256",
            "dataset_tree_sha256",
            "geometry_sidecar_manifest_path",
            "geometry_sidecar_manifest_sha256",
            "normalization_file_sha256",
            "normalization_path",
            "normalization_payload_sha256",
            "physical_sidecar_manifest_path",
            "physical_sidecar_manifest_sha256",
        },
    )
    artifacts = TrainingArtifactsRecipe(
        dataset_file_manifest_path=_relative_path(
            artifacts_payload["dataset_file_manifest_path"],
            "artifacts.dataset_file_manifest_path",
        ),
        dataset_file_manifest_sha256=_sha256(
            artifacts_payload["dataset_file_manifest_sha256"],
            "artifacts.dataset_file_manifest_sha256",
        ),
        dataset_tree_sha256=_sha256(
            artifacts_payload["dataset_tree_sha256"], "artifacts.dataset_tree_sha256"
        ),
        normalization_path=_relative_path(
            artifacts_payload["normalization_path"], "artifacts.normalization_path"
        ),
        normalization_file_sha256=_sha256(
            artifacts_payload["normalization_file_sha256"],
            "artifacts.normalization_file_sha256",
        ),
        normalization_payload_sha256=_sha256(
            artifacts_payload["normalization_payload_sha256"],
            "artifacts.normalization_payload_sha256",
        ),
        physical_sidecar_manifest_path=_relative_path(
            artifacts_payload["physical_sidecar_manifest_path"],
            "artifacts.physical_sidecar_manifest_path",
        ),
        physical_sidecar_manifest_sha256=_sha256(
            artifacts_payload["physical_sidecar_manifest_sha256"],
            "artifacts.physical_sidecar_manifest_sha256",
        ),
        geometry_sidecar_manifest_path=_relative_path(
            artifacts_payload["geometry_sidecar_manifest_path"],
            "artifacts.geometry_sidecar_manifest_path",
        ),
        geometry_sidecar_manifest_sha256=_sha256(
            artifacts_payload["geometry_sidecar_manifest_sha256"],
            "artifacts.geometry_sidecar_manifest_sha256",
        ),
    )
    geometry = PhysicalGeometryContract.from_dict(
        _mapping(payload["geometry_contract"], "geometry_contract")
    )

    core_payload = _exact(
        payload["core"],
        "core",
        {
            "binding_dim",
            "discovery",
            "modalities",
            "posterior_capacity",
            "runtime_validation",
            "state",
            "temporal",
        },
    )
    state = _exact(core_payload["state"], "core.state", {"address_dim", "content_dim"})
    address_dim = _positive_int(state["address_dim"], "address_dim")
    content_dim = _positive_int(state["content_dim"], "content_dim")
    binding_dim = _positive_int(core_payload["binding_dim"], "binding_dim")
    modalities_payload = core_payload["modalities"]
    if not isinstance(modalities_payload, list) or not modalities_payload:
        raise ValueError("core.modalities must be a nonempty list")
    modalities_list = []
    for index, item in enumerate(modalities_payload):
        modality = _exact(
            item,
            f"core.modalities[{index}]",
            {"geometry_dim", "name", "require_single_active_group", "token_dim"},
        )
        modalities_list.append(
            ModalityProjectionSpec(
                name=_text(modality["name"], f"core.modalities[{index}].name"),
                token_dim=_positive_int(
                    modality["token_dim"], f"core.modalities[{index}].token_dim"
                ),
                geometry_dim=_nonnegative_int(
                    modality["geometry_dim"], f"core.modalities[{index}].geometry_dim"
                ),
                require_single_active_group=_boolean(
                    modality["require_single_active_group"],
                    f"core.modalities[{index}].require_single_active_group",
                ),
            )
        )
    modalities = tuple(modalities_list)
    discovery_payload = _exact(
        core_payload["discovery"],
        "core.discovery",
        {
            "dropout",
            "ffn_multiplier",
            "hidden_dim",
            "initial_variance",
            "minimum_variance",
            "num_heads",
            "num_layers",
            "num_queries",
            "unmatched_query_weight",
        },
    )
    discovery = ObjectDiscoveryConfig(
        input_dim=binding_dim,
        hidden_dim=_positive_int(discovery_payload["hidden_dim"], "core.discovery.hidden_dim"),
        num_queries=_positive_int(discovery_payload["num_queries"], "core.discovery.num_queries"),
        num_layers=_positive_int(discovery_payload["num_layers"], "core.discovery.num_layers"),
        num_heads=_positive_int(discovery_payload["num_heads"], "core.discovery.num_heads"),
        address_dim=address_dim,
        content_dim=content_dim,
        geometry_dim=geometry.dimension,
        geometry_contract=geometry,
        ffn_multiplier=_positive_int(
            discovery_payload["ffn_multiplier"], "core.discovery.ffn_multiplier"
        ),
        dropout=_nonnegative_float(discovery_payload["dropout"], "core.discovery.dropout"),
        initial_variance=_positive_float(
            discovery_payload["initial_variance"], "core.discovery.initial_variance"
        ),
        minimum_variance=_positive_float(
            discovery_payload["minimum_variance"], "core.discovery.minimum_variance"
        ),
        existence_calibration=ObjectExistenceCalibration(
            unmatched_query_weight=_positive_float(
                discovery_payload["unmatched_query_weight"],
                "core.discovery.unmatched_query_weight",
            )
        ),
    )

    temporal_payload = _exact(
        core_payload["temporal"],
        "core.temporal",
        {
            "association_address_logit_bias",
            "association_address_temperature",
            "dropout",
            "empty_bank_birth_to_clutter_prior_odds",
            "ffn_multiplier",
            "hidden_dim",
            "initial_detection_probability",
            "initial_process_variance",
            "initial_survival_probability",
            "minimum_variance",
            "num_heads",
            "num_layers",
            "recurrent_birth_to_clutter_prior_odds",
            "reference_delta_t_s",
        },
    )
    temporal = TemporalFilterConfig(
        address_dim=address_dim,
        content_dim=content_dim,
        geometry_dim=geometry.dimension,
        geometry_contract=geometry,
        action_dim=host.action_dim,
        reference_delta_t_s=_positive_float(
            temporal_payload["reference_delta_t_s"], "core.temporal.reference_delta_t_s"
        ),
        hidden_dim=_positive_int(temporal_payload["hidden_dim"], "core.temporal.hidden_dim"),
        num_layers=_positive_int(temporal_payload["num_layers"], "core.temporal.num_layers"),
        num_heads=_positive_int(temporal_payload["num_heads"], "core.temporal.num_heads"),
        ffn_multiplier=_positive_int(
            temporal_payload["ffn_multiplier"], "core.temporal.ffn_multiplier"
        ),
        dropout=_nonnegative_float(temporal_payload["dropout"], "core.temporal.dropout"),
        minimum_variance=_positive_float(
            temporal_payload["minimum_variance"], "core.temporal.minimum_variance"
        ),
        initial_process_variance=_positive_float(
            temporal_payload["initial_process_variance"],
            "core.temporal.initial_process_variance",
        ),
        initial_survival_probability=_positive_float(
            temporal_payload["initial_survival_probability"],
            "core.temporal.initial_survival_probability",
        ),
        initial_detection_probability=_positive_float(
            temporal_payload["initial_detection_probability"],
            "core.temporal.initial_detection_probability",
        ),
        empty_bank_birth_to_clutter_prior_odds=_positive_float(
            temporal_payload["empty_bank_birth_to_clutter_prior_odds"],
            "core.temporal.empty_bank_birth_to_clutter_prior_odds",
        ),
        recurrent_birth_to_clutter_prior_odds=_positive_float(
            temporal_payload["recurrent_birth_to_clutter_prior_odds"],
            "core.temporal.recurrent_birth_to_clutter_prior_odds",
        ),
        association_address_temperature=_positive_float(
            temporal_payload["association_address_temperature"],
            "core.temporal.association_address_temperature",
        ),
        association_address_logit_bias=_finite_float(
            temporal_payload["association_address_logit_bias"],
            "core.temporal.association_address_logit_bias",
        ),
    )
    core = PICFCoreConfig(
        modality_specs=modalities,
        binding_dim=binding_dim,
        discovery=discovery,
        temporal=temporal,
        posterior_capacity=_positive_int(
            core_payload["posterior_capacity"], "core.posterior_capacity"
        ),
        runtime_validation=_text(core_payload["runtime_validation"], "core.runtime_validation"),
    )

    objective_payload = _exact(
        payload["objective"],
        "objective",
        {"binding", "dynamics", "geometry_overshooting", "set", "weights"},
    )
    set_payload = _exact(
        objective_payload["set"],
        "objective.set",
        {
            "address_cosine_weight",
            "content_cosine_weight",
            "existence_weight",
            "geometry_weight",
            "localization_confidence_weight",
            "ownership_ce_weight",
            "ownership_dice_weight",
        },
    )
    set_config = ObjectSetLossConfig(
        existence_weight=_nonnegative_float(
            set_payload["existence_weight"], "objective.set.existence_weight"
        ),
        localization_confidence_weight=_nonnegative_float(
            set_payload["localization_confidence_weight"],
            "objective.set.localization_confidence_weight",
        ),
        ownership_ce_weight=_nonnegative_float(
            set_payload["ownership_ce_weight"], "objective.set.ownership_ce_weight"
        ),
        ownership_dice_weight=_nonnegative_float(
            set_payload["ownership_dice_weight"], "objective.set.ownership_dice_weight"
        ),
        address_cosine_weight=_nonnegative_float(
            set_payload["address_cosine_weight"], "objective.set.address_cosine_weight"
        ),
        content_cosine_weight=_nonnegative_float(
            set_payload["content_cosine_weight"], "objective.set.content_cosine_weight"
        ),
        geometry_weight=_nonnegative_float(
            set_payload["geometry_weight"], "objective.set.geometry_weight"
        ),
    )
    dynamics_payload = _exact(
        objective_payload["dynamics"],
        "objective.dynamics",
        {
            "content_cosine_weight",
            "geometry_nll_weight",
            "probability_epsilon",
            "survival_weight",
            "visibility_weight",
        },
    )
    dynamics_config = ObjectDynamicsLossConfig(
        content_cosine_weight=_nonnegative_float(
            dynamics_payload["content_cosine_weight"],
            "objective.dynamics.content_cosine_weight",
        ),
        geometry_nll_weight=_nonnegative_float(
            dynamics_payload["geometry_nll_weight"], "objective.dynamics.geometry_nll_weight"
        ),
        survival_weight=_nonnegative_float(
            dynamics_payload["survival_weight"], "objective.dynamics.survival_weight"
        ),
        visibility_weight=_nonnegative_float(
            dynamics_payload["visibility_weight"], "objective.dynamics.visibility_weight"
        ),
        probability_epsilon=_positive_float(
            dynamics_payload["probability_epsilon"], "objective.dynamics.probability_epsilon"
        ),
    )
    binding_payload = _exact(
        objective_payload["binding"],
        "objective.binding",
        {"logit_bias", "minimum_object_mass", "objective", "temperature"},
    )
    binding_objective = _text(binding_payload["objective"], "objective.binding.objective")
    if binding_objective not in {"sigmoid", "multi_positive_infonce"}:
        raise ValueError("objective.binding.objective is unsupported")
    raw_logit_bias = binding_payload["logit_bias"]
    binding_config = BindingLossConfig(
        objective=cast(Literal["sigmoid", "multi_positive_infonce"], binding_objective),
        temperature=_positive_float(
            binding_payload["temperature"], "objective.binding.temperature"
        ),
        logit_bias=(
            None
            if raw_logit_bias is None
            else _finite_float(raw_logit_bias, "objective.binding.logit_bias")
        ),
        minimum_object_mass=_positive_float(
            binding_payload["minimum_object_mass"], "objective.binding.minimum_object_mass"
        ),
    )
    weights = _exact(
        objective_payload["weights"],
        "objective.weights",
        {"action", "binding", "dynamics", "require_temporal_positive_pairs", "set"},
    )
    objective = PICFObjectiveConfig(
        action_weight=_nonnegative_float(weights["action"], "objective.weights.action"),
        set_weight=_nonnegative_float(weights["set"], "objective.weights.set"),
        dynamics_weight=_nonnegative_float(weights["dynamics"], "objective.weights.dynamics"),
        binding_weight=_nonnegative_float(weights["binding"], "objective.weights.binding"),
        require_temporal_positive_pairs=_boolean(
            weights["require_temporal_positive_pairs"],
            "objective.weights.require_temporal_positive_pairs",
        ),
    )
    overshooting_payload = _exact(
        objective_payload["geometry_overshooting"],
        "objective.geometry_overshooting",
        {
            "fraction_denominator",
            "fraction_numerator",
            "horizons",
            "schedule_algorithm",
            "weight",
        },
    )
    raw_horizons = overshooting_payload["horizons"]
    if not isinstance(raw_horizons, (list, tuple)):
        raise ValueError("geometry overshooting horizons must be a sequence")
    overshooting = GeometryOvershootingRecipe(
        config=ObjectGeometryOvershootingConfig(
            weight=_nonnegative_float(
                overshooting_payload["weight"], "objective.geometry_overshooting.weight"
            )
        ),
        horizons=tuple(
            _positive_int(item, f"objective.geometry_overshooting.horizons[{index}]")
            for index, item in enumerate(raw_horizons)
        ),
        schedule_algorithm=_text(
            overshooting_payload["schedule_algorithm"],
            "objective.geometry_overshooting.schedule_algorithm",
        ),
        fraction_numerator=_positive_int(
            overshooting_payload["fraction_numerator"],
            "objective.geometry_overshooting.fraction_numerator",
        ),
        fraction_denominator=_positive_int(
            overshooting_payload["fraction_denominator"],
            "objective.geometry_overshooting.fraction_denominator",
        ),
    )
    optimizer_payload = _exact(
        payload["optimizer"],
        "optimizer",
        {
            "gradient_clip_norm",
            "host_parameter_groups",
            "picf_core_lr",
            "require_explicit_flow_randomness",
        },
    )
    optimizer = OptimizerRecipe(
        host_parameter_groups=_text(
            optimizer_payload["host_parameter_groups"], "optimizer.host_parameter_groups"
        ),
        picf_core_lr=_positive_float(optimizer_payload["picf_core_lr"], "optimizer.picf_core_lr"),
        gradient_clip_norm=_positive_float(
            optimizer_payload["gradient_clip_norm"], "optimizer.gradient_clip_norm"
        ),
        require_explicit_flow_randomness=_boolean(
            optimizer_payload["require_explicit_flow_randomness"],
            "optimizer.require_explicit_flow_randomness",
        ),
    )
    streaming = _exact(
        payload["streaming"],
        "streaming",
        {"detached_context_frames", "gradient_transitions"},
    )
    return PICFTrainingRecipe(
        authorization=authorization,
        dataset=dataset,
        host=host,
        policy=policy,
        artifacts=artifacts,
        geometry_contract=geometry,
        core_config=core,
        set_loss_config=set_config,
        dynamics_loss_config=dynamics_config,
        binding_loss_config=binding_config,
        objective_config=objective,
        geometry_overshooting=overshooting,
        optimizer=optimizer,
        detached_context_frames=_nonnegative_int(
            streaming["detached_context_frames"], "streaming.detached_context_frames"
        ),
        gradient_transitions=_positive_int(
            streaming["gradient_transitions"], "streaming.gradient_transitions"
        ),
    )


def write_preflight_report(
    recipe: PICFTrainingRecipe,
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_json(recipe.local_preflight_report(root)) + b"\n")
