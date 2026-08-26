"""Exact parameter ownership for shared-host representation and action adoption."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn

REPRESENTATION_SCOPE_SCHEMA = "picf-next.lingbot-representation-scope.v1"
_ACTION_STATE_HASH_CHUNK_BYTES = 64 * 1024 * 1024

NATIVE_ACTION_ONLY_PARAMETER_PREFIXES = (
    "model.qwenvl_with_expert.qwen_expert.",
    "model.state_proj.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeParameterDescriptor:
    """One optimizer-visible parameter identity without tensor values."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": self.numel,
        }


@dataclass(frozen=True, slots=True)
class NativeActionStateTensorDigest:
    """One local frozen-action tensor value at an audited boundary."""

    name: str
    kind: str
    dtype: str
    local_shape: tuple[int, ...]
    numel: int
    value_sha256: str

    def __post_init__(self) -> None:
        if not self.name or self.kind not in {"parameter", "buffer"}:
            raise ValueError("native action-state tensor identity is invalid")
        if not self.dtype or any(value < 0 for value in self.local_shape) or self.numel < 0:
            raise ValueError("native action-state tensor metadata is invalid")
        if len(self.value_sha256) != 64:
            raise ValueError("native action-state tensor digest is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
            "local_shape": list(self.local_shape),
            "numel": self.numel,
            "value_sha256": self.value_sha256,
        }


@dataclass(frozen=True, slots=True)
class NativeRepresentationParameterScope:
    """Frozen production and representation parameter partitions."""

    production_trainable: tuple[NativeParameterDescriptor, ...]
    production_frozen: tuple[NativeParameterDescriptor, ...]
    representation_trainable: tuple[NativeParameterDescriptor, ...]
    action_frozen: tuple[NativeParameterDescriptor, ...]
    schema: str = REPRESENTATION_SCOPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPRESENTATION_SCOPE_SCHEMA:
            raise ValueError("native representation parameter-scope schema changed")
        groups = (
            self.production_trainable,
            self.production_frozen,
            self.representation_trainable,
            self.action_frozen,
        )
        if any(tuple(sorted(group, key=lambda value: value.name)) != group for group in groups):
            raise ValueError("native representation parameter descriptors must be sorted")
        if not self.production_trainable or not self.representation_trainable:
            raise ValueError("native representation scope requires trainable parameters")
        if not self.action_frozen:
            raise ValueError("native representation scope found no action-only parameters")
        production_names = {value.name for value in self.production_trainable}
        representation_names = {value.name for value in self.representation_trainable}
        action_names = {value.name for value in self.action_frozen}
        if representation_names | action_names != production_names:
            raise ValueError("representation and action partitions do not cover production")
        if representation_names & action_names:
            raise ValueError("representation and action parameter partitions overlap")
        all_names = production_names | {value.name for value in self.production_frozen}
        if len(all_names) != len(self.production_trainable) + len(self.production_frozen):
            raise ValueError("native production parameter descriptors contain duplicate names")

    @property
    def production_trainable_sha256(self) -> str:
        return _canonical_digest([value.as_dict() for value in self.production_trainable])

    @property
    def production_frozen_sha256(self) -> str:
        return _canonical_digest([value.as_dict() for value in self.production_frozen])

    @property
    def representation_trainable_sha256(self) -> str:
        return _canonical_digest([value.as_dict() for value in self.representation_trainable])

    @property
    def action_frozen_sha256(self) -> str:
        return _canonical_digest([value.as_dict() for value in self.action_frozen])

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "production_trainable_sha256": self.production_trainable_sha256,
            "production_frozen_sha256": self.production_frozen_sha256,
            "representation_trainable_sha256": self.representation_trainable_sha256,
            "action_frozen_sha256": self.action_frozen_sha256,
            "production_trainable_numel": sum(value.numel for value in self.production_trainable),
            "production_frozen_numel": sum(value.numel for value in self.production_frozen),
            "representation_trainable_numel": sum(
                value.numel for value in self.representation_trainable
            ),
            "action_frozen_numel": sum(value.numel for value in self.action_frozen),
        }


def _descriptor(name: str, parameter: nn.Parameter) -> NativeParameterDescriptor:
    return NativeParameterDescriptor(
        name=name,
        shape=tuple(parameter.shape),
        dtype=str(parameter.dtype),
        numel=parameter.numel(),
    )


def _descriptors(
    policy: nn.Module,
    *,
    trainable: bool,
) -> tuple[NativeParameterDescriptor, ...]:
    return tuple(
        sorted(
            (
                _descriptor(name, parameter)
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad is trainable
            ),
            key=lambda value: value.name,
        )
    )


def _is_action_only(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in NATIVE_ACTION_ONLY_PARAMETER_PREFIXES)


def is_native_action_only_parameter(name: str) -> bool:
    """Return whether one canonical policy parameter belongs only to action generation."""

    if not isinstance(name, str) or not name:
        raise ValueError("native parameter name must be nonempty text")
    return _is_action_only(name)


def native_representation_action_state_tensor_digest(
    name: str,
    kind: str,
    tensor: torch.Tensor,
) -> NativeActionStateTensorDigest:
    """Hash one local action tensor with the boundary-manifest contract."""

    if not isinstance(name, str) or not name or kind not in {"parameter", "buffer"}:
        raise ValueError("native action-state tensor identity is invalid")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("native action-state hashing found a non-tensor")
    to_local = getattr(tensor, "to_local", None)
    local = to_local() if callable(to_local) else tensor
    if not isinstance(local, torch.Tensor):
        raise TypeError("native action-state hashing found a non-tensor local shard")
    byte_view = local.detach().contiguous().view(torch.uint8).reshape(-1)
    value_digest = hashlib.sha256()
    for offset in range(0, byte_view.numel(), _ACTION_STATE_HASH_CHUNK_BYTES):
        cpu_chunk = byte_view[offset : offset + _ACTION_STATE_HASH_CHUNK_BYTES].cpu()
        value_digest.update(memoryview(cpu_chunk.numpy()))
    return NativeActionStateTensorDigest(
        name=name,
        kind=kind,
        dtype=str(local.dtype),
        local_shape=tuple(local.shape),
        numel=local.numel(),
        value_sha256=value_digest.hexdigest(),
    )


def native_representation_frozen_action_state_sha256(
    policy: nn.Module,
    *,
    expected: NativeRepresentationParameterScope,
) -> str:
    """Hash every local action-only parameter shard and buffer without gathering it."""

    return native_representation_action_state_manifest_sha256(
        native_representation_frozen_action_state_manifest(
            policy,
            expected=expected,
        )
    )


def native_representation_frozen_action_state_manifest(
    policy: nn.Module,
    *,
    expected: NativeRepresentationParameterScope,
) -> tuple[NativeActionStateTensorDigest, ...]:
    """Describe every local frozen-action tensor so a drift names its source."""

    if not isinstance(policy, nn.Module) or not isinstance(
        expected, NativeRepresentationParameterScope
    ):
        raise TypeError("native action-state hashing requires typed scope inputs")
    named_parameters = dict(policy.named_parameters())
    expected_parameter_names = {value.name for value in expected.action_frozen}
    action_parameter_names = {
        name for name in named_parameters if _is_action_only(name)
    }
    if not expected_parameter_names or not expected_parameter_names <= action_parameter_names:
        raise RuntimeError("native action-state hashing lost a frozen action parameter")
    if not action_parameter_names:
        raise RuntimeError("native action-state hashing found no action parameter")
    if any(named_parameters[name].requires_grad for name in action_parameter_names):
        raise RuntimeError("native action-state hashing found a trainable action parameter")

    selected: list[tuple[str, str, torch.Tensor]] = [
        (name, "parameter", named_parameters[name]) for name in sorted(action_parameter_names)
    ]
    selected.extend(
        (name, "buffer", buffer)
        for name, buffer in sorted(policy.named_buffers())
        if _is_action_only(name)
    )
    if not selected:
        raise RuntimeError("native action-state hashing found no action tensors")

    records: list[NativeActionStateTensorDigest] = []
    observed_names: set[str] = set()
    for name, kind, tensor in selected:
        if name in observed_names:
            raise RuntimeError("native action-state hashing found a duplicate tensor name")
        observed_names.add(name)
        records.append(native_representation_action_state_tensor_digest(name, kind, tensor))
    return tuple(sorted(records, key=lambda value: value.name))


def native_representation_action_state_manifest_sha256(
    manifest: tuple[NativeActionStateTensorDigest, ...],
) -> str:
    """Hash a previously captured local action-state manifest."""

    if not manifest or tuple(sorted(manifest, key=lambda value: value.name)) != manifest:
        raise ValueError("native action-state manifest must be nonempty and sorted")
    if len({value.name for value in manifest}) != len(manifest):
        raise ValueError("native action-state manifest contains duplicate names")
    return _canonical_digest([value.as_dict() for value in manifest])


def native_representation_action_state_changes(
    before: tuple[NativeActionStateTensorDigest, ...],
    after: tuple[NativeActionStateTensorDigest, ...],
) -> tuple[str, ...]:
    """Return every added, removed or changed action-state tensor name."""

    for manifest in (before, after):
        if not manifest or tuple(sorted(manifest, key=lambda value: value.name)) != manifest:
            raise ValueError("native action-state manifests must be nonempty and sorted")
        if len({value.name for value in manifest}) != len(manifest):
            raise ValueError("native action-state manifest contains duplicate names")
    before_by_name = {value.name: value for value in before}
    after_by_name = {value.name: value for value in after}
    return tuple(
        name
        for name in sorted(before_by_name.keys() | after_by_name.keys())
        if before_by_name.get(name) != after_by_name.get(name)
    )


def configure_native_representation_parameter_scope(
    policy: nn.Module,
) -> NativeRepresentationParameterScope:
    """Freeze only released action-only parameters and preserve all other policy choices."""

    if not isinstance(policy, nn.Module):
        raise TypeError("native representation scope requires a torch module")
    named = tuple(policy.named_parameters())
    if not named:
        raise ValueError("native representation policy has no parameters")
    production_trainable = _descriptors(policy, trainable=True)
    production_frozen = _descriptors(policy, trainable=False)
    initial_trainable_names = {value.name for value in production_trainable}
    matched_prefixes: set[str] = set()
    action_frozen: list[NativeParameterDescriptor] = []
    for name, parameter in named:
        if not _is_action_only(name):
            continue
        matched_prefixes.update(
            prefix for prefix in NATIVE_ACTION_ONLY_PARAMETER_PREFIXES if name.startswith(prefix)
        )
        if name in initial_trainable_names:
            parameter.requires_grad_(False)
            action_frozen.append(_descriptor(name, parameter))
    missing_prefixes = tuple(
        prefix for prefix in NATIVE_ACTION_ONLY_PARAMETER_PREFIXES if prefix not in matched_prefixes
    )
    if missing_prefixes:
        raise ValueError(f"native representation action topology is incomplete: {missing_prefixes}")
    scope = NativeRepresentationParameterScope(
        production_trainable=production_trainable,
        production_frozen=production_frozen,
        representation_trainable=_descriptors(policy, trainable=True),
        action_frozen=tuple(sorted(action_frozen, key=lambda value: value.name)),
    )
    return verify_native_representation_parameter_scope(policy, expected=scope)


def verify_native_representation_parameter_scope(
    policy: nn.Module,
    *,
    expected: NativeRepresentationParameterScope,
) -> NativeRepresentationParameterScope:
    """Require exact FSDP-stable representation and frozen-action partitions."""

    if not isinstance(policy, nn.Module) or not isinstance(
        expected, NativeRepresentationParameterScope
    ):
        raise TypeError("native representation scope verification requires typed inputs")
    observed_trainable = _descriptors(policy, trainable=True)
    observed_frozen = _descriptors(policy, trainable=False)
    expected_frozen = tuple(
        sorted(
            (*expected.production_frozen, *expected.action_frozen),
            key=lambda value: value.name,
        )
    )
    if observed_trainable != expected.representation_trainable:
        raise RuntimeError("native representation trainable parameter scope changed")
    if observed_frozen != expected_frozen:
        raise RuntimeError("native representation frozen parameter scope changed")
    return expected


def restore_native_joint_adoption_parameter_scope(
    policy: nn.Module,
    *,
    expected: NativeRepresentationParameterScope,
) -> NativeRepresentationParameterScope:
    """Restore the exact production trainability captured before representation."""

    if not isinstance(policy, nn.Module) or not isinstance(
        expected, NativeRepresentationParameterScope
    ):
        raise TypeError("native joint-adoption scope restoration requires typed inputs")
    production_trainable_names = {value.name for value in expected.production_trainable}
    observed_names: set[str] = set()
    for name, parameter in policy.named_parameters():
        observed_names.add(name)
        parameter.requires_grad_(name in production_trainable_names)
    expected_names = production_trainable_names | {
        value.name for value in expected.production_frozen
    }
    if observed_names != expected_names:
        raise RuntimeError("native joint-adoption policy parameter names changed")
    if _descriptors(policy, trainable=True) != expected.production_trainable:
        raise RuntimeError("native joint-adoption trainable scope was not restored")
    if _descriptors(policy, trainable=False) != expected.production_frozen:
        raise RuntimeError("native joint-adoption frozen scope was not restored")
    return expected
