from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

import picf_next.lingbot_native.representation_stage as representation_stage
from picf_next.lingbot_native.representation_stage import (
    configure_native_representation_parameter_scope,
    native_representation_action_state_changes,
    native_representation_action_state_manifest_sha256,
    native_representation_action_state_tensor_digest,
    native_representation_frozen_action_state_manifest,
    native_representation_frozen_action_state_sha256,
    restore_native_joint_adoption_parameter_scope,
    verify_native_representation_parameter_scope,
)
from picf_next.lingbot_native.training import audit_native_optimizer_coverage


class _JointHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwenvl = nn.Linear(3, 3)
        self.qwen_expert = nn.Linear(3, 3)
        self.qwen_expert.register_buffer("routing_bias", torch.zeros(3))
        self.picf_native_graph = nn.Linear(3, 3)


class _Flow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwenvl_with_expert = _JointHost()
        self.state_proj = nn.Linear(3, 3)
        self.action_in_proj = nn.Linear(3, 3)
        self.action_out_proj = nn.Linear(3, 3)
        self.action_time_mlp_in = nn.Linear(3, 3)
        self.action_time_mlp_out = nn.Linear(3, 3)
        self.observation_query = nn.Parameter(torch.ones(3))
        self.frozen_aux = nn.Parameter(torch.zeros(3), requires_grad=False)


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Flow()


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(60, dtype=torch.float32).reshape(5, 12)[:, ::2],
        torch.arange(63, dtype=torch.bfloat16).reshape(7, 9).transpose(0, 1),
        torch.empty(0, dtype=torch.float16),
    ],
)
def test_action_state_tensor_digest_matches_monolithic_byte_sha256(
    tensor: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(representation_stage, "_ACTION_STATE_HASH_CHUNK_BYTES", 7)
    contiguous = tensor.detach().contiguous()
    expected = hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()

    observed = native_representation_action_state_tensor_digest(
        "model.action_out_proj.weight",
        "parameter",
        tensor,
    )

    assert observed.dtype == str(tensor.dtype)
    assert observed.local_shape == tuple(tensor.shape)
    assert observed.numel == tensor.numel()
    assert observed.value_sha256 == expected


def test_representation_scope_freezes_only_action_and_restores_joint_policy() -> None:
    policy = _Policy()
    production_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }

    scope = configure_native_representation_parameter_scope(policy)

    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    action_names = {value.name for value in scope.action_frozen}
    assert trainable_names | action_names == production_names
    assert not trainable_names & action_names
    assert {
        "model.qwenvl_with_expert.qwenvl.weight",
        "model.qwenvl_with_expert.picf_native_graph.weight",
        "model.observation_query",
    } <= trainable_names
    assert {
        "model.qwenvl_with_expert.qwen_expert.weight",
        "model.state_proj.weight",
        "model.action_in_proj.weight",
        "model.action_out_proj.weight",
        "model.action_time_mlp_in.weight",
        "model.action_time_mlp_out.weight",
    } <= action_names
    assert verify_native_representation_parameter_scope(policy, expected=scope) == scope
    metadata = scope.as_dict()
    assert metadata["representation_trainable_numel"] > 0
    assert metadata["action_frozen_numel"] > 0
    assert all(
        isinstance(metadata[name], str) and len(metadata[name]) == 64
        for name in (
            "production_trainable_sha256",
            "production_frozen_sha256",
            "representation_trainable_sha256",
            "action_frozen_sha256",
        )
    )

    optimizer = torch.optim.AdamW(
        tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    )
    audit_native_optimizer_coverage(modules={"policy": policy}, optimizer=optimizer)

    assert restore_native_joint_adoption_parameter_scope(policy, expected=scope) == scope
    assert {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    } == production_names


def test_representation_scope_verification_rejects_trainability_drift() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)
    policy.model.observation_query.requires_grad_(False)

    with pytest.raises(RuntimeError, match="trainable parameter scope changed"):
        verify_native_representation_parameter_scope(policy, expected=scope)


def test_representation_action_state_digest_tracks_parameters_and_buffers_only() -> None:
    policy = _Policy()
    scope = configure_native_representation_parameter_scope(policy)
    baseline_manifest = native_representation_frozen_action_state_manifest(
        policy,
        expected=scope,
    )
    baseline = native_representation_frozen_action_state_sha256(
        policy,
        expected=scope,
    )
    assert native_representation_action_state_manifest_sha256(baseline_manifest) == baseline
    by_name = {value.name: value for value in baseline_manifest}
    assert by_name["model.qwenvl_with_expert.qwen_expert.weight"].kind == "parameter"
    assert by_name["model.qwenvl_with_expert.qwen_expert.routing_bias"].kind == "buffer"

    with torch.no_grad():
        policy.model.observation_query.add_(1)
    assert (
        native_representation_frozen_action_state_sha256(
            policy,
            expected=scope,
        )
        == baseline
    )

    with torch.no_grad():
        policy.model.qwenvl_with_expert.qwen_expert.weight.add_(1)
    parameter_changed = native_representation_frozen_action_state_sha256(
        policy,
        expected=scope,
    )
    assert parameter_changed != baseline
    parameter_manifest = native_representation_frozen_action_state_manifest(
        policy,
        expected=scope,
    )
    assert native_representation_action_state_changes(
        baseline_manifest,
        parameter_manifest,
    ) == ("model.qwenvl_with_expert.qwen_expert.weight",)

    with torch.no_grad():
        policy.model.qwenvl_with_expert.qwen_expert.routing_bias.add_(1)
    assert (
        native_representation_frozen_action_state_sha256(
            policy,
            expected=scope,
        )
        != parameter_changed
    )
    buffer_manifest = native_representation_frozen_action_state_manifest(
        policy,
        expected=scope,
    )
    assert native_representation_action_state_changes(
        parameter_manifest,
        buffer_manifest,
    ) == ("model.qwenvl_with_expert.qwen_expert.routing_bias",)


def test_representation_action_state_digest_includes_pre_frozen_action_parameters() -> None:
    policy = _Policy()
    pre_frozen_name = "model.qwenvl_with_expert.qwen_expert.weight"
    dict(policy.named_parameters())[pre_frozen_name].requires_grad_(False)

    scope = configure_native_representation_parameter_scope(policy)
    manifest = native_representation_frozen_action_state_manifest(
        policy,
        expected=scope,
    )

    assert pre_frozen_name in {value.name for value in manifest}
    assert pre_frozen_name not in {value.name for value in scope.action_frozen}
    assert pre_frozen_name in {value.name for value in scope.production_frozen}


def test_representation_scope_rejects_incomplete_action_topology() -> None:
    policy = _Policy()
    del policy.model.action_time_mlp_out

    with pytest.raises(ValueError, match="action topology is incomplete"):
        configure_native_representation_parameter_scope(policy)
