"""Pinned PyTorch DCP compatibility for sparse optimizer state.

PyTorch 2.8 initializes optimizer state before an in-place DCP load, then
assumes every trainable parameter had state in the checkpoint. Parameters that
were unused by the saved loss violate that assumption. PyTorch fixed the issue
upstream in commit ``edd8d356b6d9a00cfa34fa323578e5cf1c7e0463``.

LingBot is pinned to PyTorch 2.8, so native training installs that exact
behavioral backport in-process. The model, optimizer update and checkpoint
format are unchanged.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Collection, Mapping, MutableMapping
from typing import Any

import torch.distributed.checkpoint.state_dict as _torch_dcp_state_dict

UPSTREAM_COMMIT = "edd8d356b6d9a00cfa34fa323578e5cf1c7e0463"
_PYTORCH_2_8_SPLIT_SHA256 = "643c75e7e87679393d22e2a9f908ab3f965e1798881f99656a93fec618f64a1a"
_INSTALL_MARKER = "_picf_next_sparse_optimizer_state_backport"
LINGBOT_DCP_OPTIMIZER_STATE_PREFIX = "state.optim.state."


def prune_synthetic_optimizer_state_from_dcp_template(
    optim_state_dict: MutableMapping[str, Any],
    *,
    checkpoint_metadata_keys: Collection[str],
    metadata_prefix: str = LINGBOT_DCP_OPTIMIZER_STATE_PREFIX,
) -> dict[str, int]:
    """Remove cold-load template state that was absent from the checkpoint.

    DCP loads Stateful objects in place. Its optimizer template initialization
    performs a zero-gradient optimizer step, creating state for every trainable
    parameter even when the saved optimizer had no state for an unused
    parameter. The checkpoint metadata is the authoritative record of which
    parameter states were actually saved.
    """

    state = optim_state_dict.get("state")
    if not isinstance(state, MutableMapping):
        raise TypeError("optimizer state dictionary has no mutable 'state' mapping")
    metadata = frozenset(checkpoint_metadata_keys)
    if not all(isinstance(key, str) for key in metadata):
        raise TypeError("DCP checkpoint metadata keys must be strings")

    removed_parameters = 0
    preserved_parameters = 0
    for parameter_fqn, parameter_state in tuple(state.items()):
        if not isinstance(parameter_fqn, str):
            raise TypeError("DCP optimizer state must use parameter FQN keys")
        if not isinstance(parameter_state, Mapping):
            raise TypeError(f"optimizer state for {parameter_fqn!r} is not a mapping")
        field_names = set(parameter_state)
        if not all(isinstance(field, str) for field in field_names):
            raise TypeError(f"optimizer state for {parameter_fqn!r} has non-string fields")

        saved_fields = {
            field
            for field in field_names
            if (
                (base := f"{metadata_prefix}{parameter_fqn}.{field}") in metadata
                or any(key.startswith(f"{base}.") for key in metadata)
            )
        }
        if not saved_fields:
            del state[parameter_fqn]
            removed_parameters += 1
            continue
        if saved_fields != field_names:
            missing = sorted(field_names - saved_fields)
            raise RuntimeError(
                f"checkpoint optimizer state for {parameter_fqn!r} is incomplete: {missing}"
            )
        preserved_parameters += 1

    return {
        "preserved_parameters": preserved_parameters,
        "removed_synthetic_parameters": removed_parameters,
    }


def _split_optim_state_dict_backport(
    model: Any,
    optim: Any,
    optim_state_dict: Any,
    info: Any,
) -> Any:
    """Backport PyTorch PR #165228 to the pinned 2.8 implementation."""

    state_key = _torch_dcp_state_dict._STATE
    param_groups_key = _torch_dcp_state_dict._PG
    params_key = _torch_dcp_state_dict._PARAMS

    loaded_state = optim_state_dict[state_key]
    loaded_param_groups = optim_state_dict[param_groups_key]
    if not isinstance(loaded_state, Mapping) or not isinstance(loaded_param_groups, list):
        raise TypeError("DCP optimizer state dictionary is malformed")

    state: dict[Any, Any] = {}
    param_groups: list[dict[str, Any]] = []
    result = {state_key: state, param_groups_key: param_groups}
    param_group_mapping: dict[int, int] = {}

    if all(isinstance(key, int) for key in loaded_state):
        return optim_state_dict

    for optimizer_group in optim.param_groups:
        param_groups.append({params_key: []})
        for parameter in optimizer_group[params_key]:
            for fqn in info.fqn_param_mapping[parameter]:
                if fqn in info.shared_params_mapping:
                    in_params = any(
                        fqn in loaded_group[params_key] for loaded_group in loaded_param_groups
                    )
                else:
                    in_params = True
                if not in_params:
                    continue

                result_params = param_groups[-1][params_key]
                if not isinstance(result_params, list):
                    raise AssertionError(f"Expected list, got {type(result_params)}")
                result_params.append(fqn)
                if parameter.requires_grad:
                    if fqn in loaded_state:
                        state[fqn] = loaded_state[fqn]
                    elif info.strict:
                        raise RuntimeError(
                            f"Missing optimizer state for parameter '{fqn}' in checkpoint. "
                            "The parameter requires gradients but has no saved optimizer state. "
                            "To load anyway, use StateDictOptions(strict=False)."
                        )
                for loaded_group in loaded_param_groups:
                    if fqn in loaded_group[params_key]:
                        param_group_mapping[id(loaded_group)] = len(param_groups) - 1

        if len(optimizer_group[params_key]) == 0:
            empty_groups = [
                loaded_group
                for loaded_group in loaded_param_groups
                if len(loaded_group[params_key]) == 0
            ]
            if len(empty_groups) != 1:
                raise ValueError(
                    "There are param groups that have zero parameters. "
                    "In such a case, DSD only support exactly one param group "
                    "with zero parameters. But the loaded state_dict has zero "
                    "or more than one param groups with zero parameters."
                )
            if len(loaded_param_groups) != len(optim.param_groups):
                raise ValueError(
                    "When there is a parameter group that has zero parameters, "
                    "multiple optimizers are not supported."
                )
            param_group_mapping[id(empty_groups[0])] = len(param_groups) - 1

    for loaded_group in loaded_param_groups:
        group_index = param_group_mapping.get(id(loaded_group), -1)
        if group_index == -1:
            continue
        for key, value in loaded_group.items():
            if key != params_key:
                param_groups[group_index][key] = value

    return result


def install_torch_2_8_sparse_optimizer_state_backport(torch_module: Any) -> dict[str, Any]:
    """Install the exact upstream unused-parameter DCP behavior fail-closed."""

    version = str(getattr(torch_module, "__version__", ""))
    if version.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(f"PICF DCP backport requires PyTorch 2.8.0, found {version!r}")

    state_dict_module = _torch_dcp_state_dict
    installed = getattr(state_dict_module, _INSTALL_MARKER, None)
    if installed is not None:
        if (
            installed != UPSTREAM_COMMIT
            or state_dict_module._split_optim_state_dict is not _split_optim_state_dict_backport
        ):
            raise RuntimeError("PyTorch DCP backport marker is inconsistent")
        return {
            "schema": "picf-next.torch-dcp-sparse-optimizer-state.v1",
            "status": "already-installed",
            "torch_version": version,
            "upstream_commit": UPSTREAM_COMMIT,
        }

    original = state_dict_module._split_optim_state_dict
    original_sha256 = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if original_sha256 != _PYTORCH_2_8_SPLIT_SHA256:
        raise RuntimeError(
            "PyTorch 2.8 DCP optimizer-state implementation differs from the audited source"
        )

    state_dict_module._split_optim_state_dict = _split_optim_state_dict_backport
    setattr(state_dict_module, _INSTALL_MARKER, UPSTREAM_COMMIT)
    return {
        "schema": "picf-next.torch-dcp-sparse-optimizer-state.v1",
        "status": "installed",
        "torch_version": version,
        "upstream_commit": UPSTREAM_COMMIT,
        "replaced_source_sha256": original_sha256,
    }
