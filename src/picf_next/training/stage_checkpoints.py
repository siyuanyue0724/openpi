"""Strict checkpoint boundaries between PICF representation, temporal and action stages."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from picf_next.models.objective import PICFObjective
from picf_next.training.stationary_temporal import STATIONARY_TEMPORAL_EXECUTION_CONTRACT

STATIONARY_TEMPORAL_CHECKPOINT_SCHEMA = "picf-next.stationary-temporal-core.v1"
STATIONARY_TEMPORAL_PROVENANCE_SCHEMA = "picf-next.stationary-temporal-provenance.v1"
STATIONARY_TEMPORAL_STAGE = "M3_stationary_temporal_calibration"

_LEGACY_SENSOR_PARAMETER_FRAGMENTS = (
    "visibility_persistence_head",
    "visibility_reappearance_head",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _git_sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("code revision must be one lowercase 40-character Git SHA")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _exact_mapping(value: object, name: str, fields: set[str]) -> Mapping[str, object]:
    payload = _mapping(value, name)
    if set(payload) != fields:
        raise ValueError(f"{name} fields differ from its frozen schema")
    return payload


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parameter_scope_sha256(
    core: nn.Module,
    objective: PICFObjective,
) -> tuple[str, str]:
    """Hash exact Stage-B trainable/frozen parameter identities and tensor schemas."""

    records: dict[bool, list[dict[str, object]]] = {True: [], False: []}
    for prefix, module in (("core", core), ("objective", objective)):
        for name, parameter in module.named_parameters():
            records[parameter.requires_grad].append(
                {
                    "name": f"{prefix}.{name}",
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
            )
    if not records[True]:
        raise ValueError("stationary temporal stage has no trainable parameters")
    return _canonical_sha256(records[True]), _canonical_sha256(records[False])


@dataclass(frozen=True, slots=True)
class StationaryTemporalCheckpointProvenance:
    stage_recipe_sha256: str
    source_coverage_recipe_sha256: str
    foundation_recipe_sha256: str
    m2_checkpoint_sha256: str
    feature_cache_manifest_sha256: str
    dataset_manifest_sha256: str
    physical_sidecar_manifest_sha256: str
    clip_plan_sha256: str
    trainable_parameter_scope_sha256: str
    frozen_parameter_scope_sha256: str
    code_revision: str
    optimizer_steps: int
    state_parameter_version: int
    recurrent_state_serialized: bool = False
    execution_contract: str = STATIONARY_TEMPORAL_EXECUTION_CONTRACT
    stage: str = STATIONARY_TEMPORAL_STAGE
    schema: str = STATIONARY_TEMPORAL_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATIONARY_TEMPORAL_PROVENANCE_SCHEMA:
            raise ValueError("stationary temporal provenance schema changed")
        if self.stage != STATIONARY_TEMPORAL_STAGE:
            raise ValueError("stationary temporal checkpoint stage changed")
        if self.execution_contract != STATIONARY_TEMPORAL_EXECUTION_CONTRACT:
            raise ValueError("stationary temporal execution contract changed")
        for name in (
            "stage_recipe_sha256",
            "source_coverage_recipe_sha256",
            "foundation_recipe_sha256",
            "m2_checkpoint_sha256",
            "feature_cache_manifest_sha256",
            "dataset_manifest_sha256",
            "physical_sidecar_manifest_sha256",
            "clip_plan_sha256",
            "trainable_parameter_scope_sha256",
            "frozen_parameter_scope_sha256",
        ):
            _sha256(getattr(self, name), name)
        _git_sha(self.code_revision)
        _positive_integer(self.optimizer_steps, "stationary temporal optimizer steps")
        _positive_integer(self.state_parameter_version, "stationary state parameter version")
        if self.state_parameter_version != self.optimizer_steps:
            raise ValueError("stationary state parameter version must equal completed updates")
        if self.recurrent_state_serialized is not False:
            raise ValueError("stationary temporal checkpoints cannot serialize recurrent state")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "execution_contract": self.execution_contract,
            "stage_recipe_sha256": self.stage_recipe_sha256,
            "source_coverage_recipe_sha256": self.source_coverage_recipe_sha256,
            "foundation_recipe_sha256": self.foundation_recipe_sha256,
            "m2_checkpoint_sha256": self.m2_checkpoint_sha256,
            "feature_cache_manifest_sha256": self.feature_cache_manifest_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "physical_sidecar_manifest_sha256": self.physical_sidecar_manifest_sha256,
            "clip_plan_sha256": self.clip_plan_sha256,
            "trainable_parameter_scope_sha256": self.trainable_parameter_scope_sha256,
            "frozen_parameter_scope_sha256": self.frozen_parameter_scope_sha256,
            "code_revision": self.code_revision,
            "optimizer_steps": self.optimizer_steps,
            "state_parameter_version": self.state_parameter_version,
            "recurrent_state_serialized": self.recurrent_state_serialized,
        }

    @classmethod
    def from_dict(cls, value: object) -> StationaryTemporalCheckpointProvenance:
        fields = {
            "schema",
            "stage",
            "execution_contract",
            "stage_recipe_sha256",
            "source_coverage_recipe_sha256",
            "foundation_recipe_sha256",
            "m2_checkpoint_sha256",
            "feature_cache_manifest_sha256",
            "dataset_manifest_sha256",
            "physical_sidecar_manifest_sha256",
            "clip_plan_sha256",
            "trainable_parameter_scope_sha256",
            "frozen_parameter_scope_sha256",
            "code_revision",
            "optimizer_steps",
            "state_parameter_version",
            "recurrent_state_serialized",
        }
        payload = _exact_mapping(value, "stationary temporal provenance", fields)
        if not isinstance(payload["recurrent_state_serialized"], bool):
            raise ValueError("recurrent_state_serialized must be boolean")
        return cls(
            schema=cast(str, payload["schema"]),
            stage=cast(str, payload["stage"]),
            execution_contract=cast(str, payload["execution_contract"]),
            stage_recipe_sha256=_sha256(payload["stage_recipe_sha256"], "stage_recipe_sha256"),
            source_coverage_recipe_sha256=_sha256(
                payload["source_coverage_recipe_sha256"],
                "source_coverage_recipe_sha256",
            ),
            foundation_recipe_sha256=_sha256(
                payload["foundation_recipe_sha256"], "foundation_recipe_sha256"
            ),
            m2_checkpoint_sha256=_sha256(payload["m2_checkpoint_sha256"], "m2_checkpoint_sha256"),
            feature_cache_manifest_sha256=_sha256(
                payload["feature_cache_manifest_sha256"],
                "feature_cache_manifest_sha256",
            ),
            dataset_manifest_sha256=_sha256(
                payload["dataset_manifest_sha256"], "dataset_manifest_sha256"
            ),
            physical_sidecar_manifest_sha256=_sha256(
                payload["physical_sidecar_manifest_sha256"],
                "physical_sidecar_manifest_sha256",
            ),
            clip_plan_sha256=_sha256(payload["clip_plan_sha256"], "clip_plan_sha256"),
            trainable_parameter_scope_sha256=_sha256(
                payload["trainable_parameter_scope_sha256"],
                "trainable_parameter_scope_sha256",
            ),
            frozen_parameter_scope_sha256=_sha256(
                payload["frozen_parameter_scope_sha256"],
                "frozen_parameter_scope_sha256",
            ),
            code_revision=_git_sha(payload["code_revision"]),
            optimizer_steps=_positive_integer(payload["optimizer_steps"], "optimizer_steps"),
            state_parameter_version=_positive_integer(
                payload["state_parameter_version"], "state_parameter_version"
            ),
            recurrent_state_serialized=payload["recurrent_state_serialized"],
        )


def load_picf_current_frame_checkpoint(
    core: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, tuple[str, ...]]:
    """Load accepted M2 observation weights while leaving posterior state fresh."""

    path = Path(checkpoint_path)
    if sha256_file(path) != _sha256(expected_sha256, "M2 checkpoint SHA-256"):
        raise ValueError("PICF M2 checkpoint changed between validation and loading")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"model"}:
        raise ValueError("PICF M2 checkpoint payload changed")
    raw_state = payload["model"]
    if not isinstance(raw_state, Mapping):
        raise ValueError("PICF M2 checkpoint payload changed")
    state = dict(raw_state)
    if not state or any(
        not isinstance(name, str)
        or not isinstance(value, torch.Tensor)
        or not name.startswith(("projector.", "discovery."))
        for name, value in state.items()
    ):
        raise ValueError("PICF M2 checkpoint escaped current-frame module prefixes")
    variance_weight = state.get("discovery.variance_head.weight")
    if not isinstance(variance_weight, torch.Tensor) or bool(torch.count_nonzero(variance_weight)):
        raise ValueError("PICF M2 checkpoint retained conditional variance weights")
    incompatible = core.load_state_dict(state, strict=False)
    fresh_localization_confidence = {
        "discovery.localization_confidence_head.weight",
        "discovery.localization_confidence_head.bias",
    }
    missing = set(incompatible.missing_keys)
    if incompatible.unexpected_keys or any(
        not name.startswith("posterior_filter.") and name not in fresh_localization_confidence
        for name in incompatible.missing_keys
    ):
        raise ValueError("PICF M2 checkpoint does not map exactly onto current-frame core modules")
    if missing & fresh_localization_confidence:
        if missing & fresh_localization_confidence != fresh_localization_confidence:
            raise ValueError("PICF M2 checkpoint contains a partial localization-confidence head")
        confidence_head = getattr(
            getattr(core, "discovery", None),
            "localization_confidence_head",
            None,
        )
        if (
            not isinstance(confidence_head, nn.Linear)
            or bool(torch.count_nonzero(confidence_head.weight))
            or bool(torch.count_nonzero(confidence_head.bias))
        ):
            raise ValueError(
                "fresh localization-confidence head lost its audited zero initialization"
            )
    variance_head = getattr(getattr(core, "discovery", None), "variance_head", None)
    if variance_head is None or variance_head.weight.requires_grad:
        raise ValueError("loaded PICF core lost the frozen axis-constant variance contract")
    return {
        "loaded_keys": tuple(sorted(state)),
        "fresh_keys": tuple(sorted(incompatible.missing_keys)),
    }


def _validated_tensor_state(value: object, name: str) -> dict[str, torch.Tensor]:
    payload = _mapping(value, name)
    if not payload:
        raise ValueError(f"{name} cannot be empty")
    state: dict[str, torch.Tensor] = {}
    for key, raw_tensor in payload.items():
        if not isinstance(raw_tensor, torch.Tensor):
            raise ValueError(f"{name} must contain tensors only")
        if raw_tensor.requires_grad or (
            raw_tensor.is_floating_point() and not bool(torch.isfinite(raw_tensor).all())
        ):
            raise ValueError(f"{name} contains an invalid tensor")
        tensor = raw_tensor.detach()
        state[key] = tensor
    if any(fragment in key for key in state for fragment in _LEGACY_SENSOR_PARAMETER_FRAGMENTS):
        raise ValueError(f"{name} contains a legacy non-identifiable sensor head")
    return state


def _cpu_state(module: nn.Module, name: str) -> dict[str, torch.Tensor]:
    state = {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in module.state_dict().items()
    }
    return _validated_tensor_state(state, name)


def save_stationary_temporal_checkpoint(
    path: str | Path,
    *,
    core: nn.Module,
    objective: PICFObjective,
    provenance: StationaryTemporalCheckpointProvenance,
) -> str:
    """Atomically publish a full temporal core without optimizer or recurrent state."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("stationary temporal checkpoint destination already exists")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("stationary temporal checkpoint parent must be an existing real directory")
    trainable_sha, frozen_sha = parameter_scope_sha256(core, objective)
    if (
        provenance.trainable_parameter_scope_sha256 != trainable_sha
        or provenance.frozen_parameter_scope_sha256 != frozen_sha
    ):
        raise ValueError("stationary temporal parameter scope differs from checkpoint provenance")
    payload = {
        "schema": STATIONARY_TEMPORAL_CHECKPOINT_SCHEMA,
        "provenance": provenance.to_dict(),
        "core": _cpu_state(core, "stationary temporal core state"),
        "objective": _cpu_state(objective, "stationary temporal objective state"),
    }
    temporary = destination.with_name(f".{destination.name}.incomplete-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("stationary temporal checkpoint staging path already exists")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(destination)


def _read_stationary_temporal_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[
    StationaryTemporalCheckpointProvenance,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("stationary temporal checkpoint must be one regular file")
    if sha256_file(path) != _sha256(expected_sha256, "temporal checkpoint SHA-256"):
        raise ValueError("stationary temporal checkpoint content hash changed")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    payload = _exact_mapping(
        raw,
        "stationary temporal checkpoint",
        {"schema", "provenance", "core", "objective"},
    )
    if payload["schema"] != STATIONARY_TEMPORAL_CHECKPOINT_SCHEMA:
        raise ValueError("stationary temporal checkpoint schema changed")
    provenance = StationaryTemporalCheckpointProvenance.from_dict(payload["provenance"])
    core_state = _validated_tensor_state(payload["core"], "stationary temporal core state")
    objective_state = _validated_tensor_state(
        payload["objective"], "stationary temporal objective state"
    )
    if not any(name.startswith("posterior_filter.") for name in core_state):
        raise ValueError("stationary temporal checkpoint omitted the posterior filter")
    return provenance, core_state, objective_state


def inspect_stationary_temporal_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
) -> StationaryTemporalCheckpointProvenance:
    """Validate complete checkpoint structure without constructing a host model."""

    provenance, _core_state, _objective_state = _read_stationary_temporal_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
    )
    return provenance


def load_stationary_temporal_checkpoint(
    core: nn.Module,
    objective: PICFObjective,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    expected_provenance: StationaryTemporalCheckpointProvenance | None = None,
) -> StationaryTemporalCheckpointProvenance:
    """Load an exact full Stage-B checkpoint; partial M2 state cannot pass this boundary."""

    provenance, core_state, objective_state = _read_stationary_temporal_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
    )
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("stationary temporal checkpoint provenance differs from acceptance")
    trainable_sha, frozen_sha = parameter_scope_sha256(core, objective)
    if (
        provenance.trainable_parameter_scope_sha256 != trainable_sha
        or provenance.frozen_parameter_scope_sha256 != frozen_sha
    ):
        raise ValueError("stationary temporal parameter scope differs from the target modules")
    if set(core_state) != set(core.state_dict()):
        raise ValueError("stationary temporal checkpoint core schema differs from the model")
    if set(objective_state) != set(objective.state_dict()):
        raise ValueError("stationary temporal checkpoint objective schema differs from the model")
    core.load_state_dict(core_state, strict=True)
    objective.load_state_dict(objective_state, strict=True)
    return provenance
