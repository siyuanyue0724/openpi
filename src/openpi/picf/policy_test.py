from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
import torch

from openpi.picf.contracts import PicfObservation
from openpi.picf.policy import PicfPi05Policy


def _dummy_observation() -> PicfObservation:
    return PicfObservation(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_static=np.zeros((8, 8), dtype=np.float32),
        robot_obs=np.zeros((15,), dtype=np.float32),
        prompt="test",
        step_id=1,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
        action=np.zeros((7,), dtype=np.float32),
    )


def test_policy_act_requires_action_generator_when_v22_core_is_active() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=torch.ones((4, 8), dtype=torch.float32),
                )
            )

    policy = PicfPi05Policy(core=_DummyCore(), semantic_encoder=None)
    with pytest.raises(RuntimeError, match="requires PI0.5 action generation"):
        policy.act(_dummy_observation())


def test_policy_act_requires_action_generator_when_legacy_core_path_is_active() -> None:
    class _LegacyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

        def step(self, *_args, **_kwargs):
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.zeros((7,), dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    policy = PicfPi05Policy(core=_LegacyCore(), semantic_encoder=None)
    with pytest.raises(RuntimeError, match="requires PI0.5 action generation"):
        policy.act(_dummy_observation())


def test_policy_forward_train_transition_uses_conditioned_control_pi_prefix() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)
            self.pi_prefix = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(pi_prefix_tokens=self.pi_prefix),
            )

        def finalize_with_action(self, _observation, observed, *, action_future):
            assert observed.conditioned_control.pi_prefix_tokens is self.pi_prefix
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.as_tensor(action_future, dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = None

        def encode_observation(self, _observation):
            return torch.zeros((2, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            assert semantic_override.shape == (2, 4)
            self.seen_prefix = extra_prefix_tokens.clone()
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
            }

    core = _DummyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic)
    target = torch.ones((2, 7), dtype=torch.float32)
    result = policy.forward_train_transition(
        _dummy_observation(),
        action_chunk_target=target,
    )

    torch.testing.assert_close(semantic.seen_prefix, core.pi_prefix)
    torch.testing.assert_close(result.flow_override["predicted_chunk"], target)


def test_policy_forward_train_transition_can_disable_picf_action_condition() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                require_pi0_action_generator=True,
                picf_action_condition_enabled=False,
            )
            self.pi_prefix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            self.control_tokens = torch.ones((5, 4), dtype=torch.float32)

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=self.pi_prefix,
                    tokens=self.control_tokens,
                ),
            )

        def finalize_with_action(self, _observation, observed, *, action_future):
            assert observed.conditioned_control.pi_prefix_tokens is self.pi_prefix
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.as_tensor(action_future, dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = "unset"
            self.seen_context = "unset"

        def encode_observation(self, _observation):
            return torch.zeros((2, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(
            self,
            semantic_override,
            *,
            extra_prefix_tokens,
            action_chunk_target,
        ):
            del semantic_override
            self.seen_prefix = extra_prefix_tokens
            zero = torch.tensor(0.0)
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
                "picf_action_condition_enabled": zero,
            }

    core = _DummyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic)
    target = torch.ones((2, 7), dtype=torch.float32)
    result = policy.forward_train_transition(_dummy_observation(), action_chunk_target=target)

    assert semantic.seen_prefix is None
    assert semantic.seen_context == "unset"
    assert result.output.debug["pi_action_condition_enabled"] == pytest.approx(0.0)
    torch.testing.assert_close(result.flow_override["predicted_chunk"], target)


def test_policy_forward_train_transition_can_append_gated_action_context() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                require_pi0_action_generator=True,
                action_prefix_stopgrad=False,
                action_prefix_teacher_mode="off",
                action_context_tokens=2,
                action_context_stopgrad=True,
                action_context_norm_mode="rmsnorm",
                action_context_rms_target=2.0,
                action_context_output_gate=0.5,
                action_context_include_query_tokens=False,
                conditioned_control_queries=1,
            )
            self.pi_prefix = torch.ones((1, 4), dtype=torch.float32)
            self.control_tokens = torch.tensor(
                [
                    [3.0, 4.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0, 9.0],  # query token, excluded by default
                ],
                dtype=torch.float32,
                requires_grad=True,
            )

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=self.pi_prefix,
                    tokens=self.control_tokens,
                ),
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.as_tensor(action_future, dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = None

        def encode_observation(self, _observation):
            return torch.zeros((2, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            del semantic_override
            self.seen_prefix = extra_prefix_tokens
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
            }

    core = _DummyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic)
    target = torch.ones((2, 7), dtype=torch.float32)
    result = policy.forward_train_transition(_dummy_observation(), action_chunk_target=target)

    assert semantic.seen_prefix is not None
    assert semantic.seen_prefix.shape == (3, 4)
    torch.testing.assert_close(semantic.seen_prefix[0], core.pi_prefix[0])
    context_rms = torch.sqrt(torch.mean(semantic.seen_prefix[1:].detach().square(), dim=-1))
    torch.testing.assert_close(context_rms, torch.ones_like(context_rms), atol=1.0e-4, rtol=1.0e-4)
    assert semantic.seen_prefix[1:].requires_grad is False
    assert result.output.debug["pi_context_token_count"] == pytest.approx(2.0)
    assert result.output.debug["pi_context_gate"] == pytest.approx(0.5)
    assert result.output.debug["pi_action_condition_token_count"] == pytest.approx(3.0)


def test_policy_forward_train_transition_can_fuse_context_without_growing_prefix() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                require_pi0_action_generator=True,
                action_prefix_stopgrad=False,
                action_prefix_teacher_mode="off",
                action_context_tokens=2,
                action_context_integration="prefix_fusion",
                action_context_stopgrad=True,
                action_context_norm_mode="rmsnorm",
                action_context_rms_target=1.0,
                action_context_output_gate=0.25,
                action_context_include_query_tokens=False,
                conditioned_control_queries=1,
            )
            self.pi_prefix = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            self.control_tokens = torch.tensor(
                [
                    [0.0, 0.0, 3.0, 4.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0, 9.0],  # query token, excluded by default
                ],
                dtype=torch.float32,
                requires_grad=True,
            )

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=self.pi_prefix,
                    tokens=self.control_tokens,
                ),
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.as_tensor(action_future, dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = None

        def encode_observation(self, _observation):
            return torch.zeros((2, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            del semantic_override
            self.seen_prefix = extra_prefix_tokens
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
            }

    core = _DummyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic)
    target = torch.ones((2, 7), dtype=torch.float32)
    result = policy.forward_train_transition(_dummy_observation(), action_chunk_target=target)

    assert semantic.seen_prefix is not None
    assert semantic.seen_prefix.shape == core.pi_prefix.shape
    assert not torch.allclose(semantic.seen_prefix, core.pi_prefix)
    fused_rms = torch.sqrt(torch.mean(semantic.seen_prefix.detach().square(), dim=-1))
    original_rms = torch.sqrt(torch.mean(core.pi_prefix.detach().square(), dim=-1))
    assert torch.all(fused_rms <= original_rms + 1.0e-5)
    assert result.output.debug["pi_context_token_count"] == pytest.approx(2.0)
    assert result.output.debug["pi_context_gate"] == pytest.approx(0.25)
    assert result.output.debug["pi_context_fused_prefix_token_count"] == pytest.approx(2.0)
    assert result.output.debug["pi_action_condition_token_count"] == pytest.approx(2.0)


def test_policy_forward_train_transition_can_route_context_to_action_side_adapter() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                require_pi0_action_generator=True,
                action_prefix_stopgrad=False,
                action_prefix_teacher_mode="off",
                action_context_tokens=2,
                action_context_integration="suffix_cross_attention",
                action_context_stopgrad=True,
                action_context_norm_mode="rmsnorm",
                action_context_rms_target=1.0,
                action_context_output_gate=0.25,
                action_context_include_query_tokens=False,
                conditioned_control_queries=1,
            )
            self.pi_prefix = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            self.control_tokens = torch.tensor(
                [
                    [0.0, 0.0, 3.0, 4.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0, 9.0],
                ],
                dtype=torch.float32,
                requires_grad=True,
            )

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=self.pi_prefix,
                    tokens=self.control_tokens,
                ),
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.as_tensor(action_future, dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = "unset"
            self.seen_context = None

        def encode_observation(self, _observation):
            return torch.zeros((2, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(
            self,
            semantic_override,
            *,
            extra_prefix_tokens,
            extra_action_context_tokens,
            action_chunk_target,
        ):
            del semantic_override
            self.seen_prefix = extra_prefix_tokens
            self.seen_context = extra_action_context_tokens
            zero = torch.tensor(0.0)
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
                "picf_action_context_adapter_token_count": zero + float(extra_action_context_tokens.shape[0]),
                "picf_action_context_adapter_gate": zero + 0.125,
                "picf_action_context_adapter_attention_entropy_mean": zero + 0.5,
                "picf_action_context_adapter_residual_rms_mean": zero + 0.25,
            }

    core = _DummyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic)
    target = torch.ones((2, 7), dtype=torch.float32)
    result = policy.forward_train_transition(_dummy_observation(), action_chunk_target=target)

    assert semantic.seen_prefix is None
    assert semantic.seen_context is not None
    assert semantic.seen_context.shape == (4, 4)
    torch.testing.assert_close(semantic.seen_context[:2], core.pi_prefix)
    assert semantic.seen_context[2:].requires_grad is False
    assert result.output.debug["pi_context_token_count"] == pytest.approx(2.0)
    assert result.output.debug["pi_action_condition_token_count"] == pytest.approx(4.0)
    assert result.output.debug["pi_context_adapter_token_count"] == pytest.approx(4.0)
    assert result.output.debug["pi_context_adapter_gate"] == pytest.approx(0.125)


def test_policy_action_prefix_ema_teacher_stabilizes_train_prefix() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(
                require_pi0_action_generator=True,
                action_prefix_stopgrad=False,
                action_prefix_teacher_mode="ema",
                action_prefix_teacher_ema_decay=0.5,
                action_prefix_teacher_blend=1.0,
                lambda_action_prefix_trust=0.25,
            )
            self.pi_prefix = torch.zeros((2, 4), dtype=torch.float32)
            self.action_prefix_teacher_tokens = torch.zeros((2, 4), dtype=torch.float32)
            self.action_prefix_teacher_initialized = torch.zeros((), dtype=torch.float32)

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(pi_prefix_tokens=self.pi_prefix),
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.as_tensor(action_future, dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefixes: list[torch.Tensor] = []

        def encode_observation(self, _observation):
            return torch.zeros((2, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            del semantic_override
            self.seen_prefixes.append(extra_prefix_tokens.detach().clone())
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
            }

    core = _DummyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic)
    target = torch.ones((2, 7), dtype=torch.float32)

    first_prefix = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    second_prefix = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    core.pi_prefix = first_prefix
    first = policy.forward_train_transition(_dummy_observation(), action_chunk_target=target)
    core.pi_prefix = second_prefix
    second = policy.forward_train_transition(_dummy_observation(), action_chunk_target=target)

    torch.testing.assert_close(semantic.seen_prefixes[0], first_prefix)
    torch.testing.assert_close(semantic.seen_prefixes[1], first_prefix)
    assert first.flow_override["picf_action_prefix_trust_loss"].item() == pytest.approx(0.0)
    assert second.flow_override["picf_action_prefix_trust_loss"].item() > 0.0
    assert second.output.debug["pi_prefix_teacher_mode_enabled"] == pytest.approx(1.0)


def test_policy_forward_train_transition_uses_action_chunk_when_single_action_missing() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)
            self.seen_action_future = None

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=torch.ones((4, 8), dtype=torch.float32),
                )
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            self.seen_action_future = torch.as_tensor(action_future, dtype=torch.float32)
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=self.seen_action_future[0],
                    action_chunk=self.seen_action_future,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def encode_observation(self, _observation):
            return torch.zeros((2, 8), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            del semantic_override, extra_prefix_tokens
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
            }

    observation = _dummy_observation()
    observation.action = None
    target = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    core = _DummyCore()
    policy = PicfPi05Policy(core=core, semantic_encoder=_SemanticEncoder())

    result = policy.forward_train_transition(
        observation,
        action_chunk_target=target,
    )

    torch.testing.assert_close(core.seen_action_future, target)
    torch.testing.assert_close(result.output.state.predictive.action_chunk, target)


def test_policy_forward_train_transition_returns_recurrent_carry_when_core_supports_it() -> None:
    carry = object()

    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=torch.ones((4, 8), dtype=torch.float32),
                )
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.as_tensor(action_future, dtype=torch.float32),
                    action_chunk=None,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

        def make_recurrent_carry(self, state):
            assert state is not None
            return carry

    class _SemanticEncoder:
        def encode_observation(self, _observation):
            return torch.zeros((2, 8), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            del semantic_override, extra_prefix_tokens
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.zeros((7,), dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32),
            }

    policy = PicfPi05Policy(core=_DummyCore(), semantic_encoder=_SemanticEncoder())
    target = torch.ones((2, 7), dtype=torch.float32)
    result = policy.forward_train_transition(
        _dummy_observation(),
        action_chunk_target=target,
    )

    assert result.next_state is carry


def test_policy_forward_train_transition_legacy_core_uses_flow_override_when_available() -> None:
    class _LegacyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)
            self.prefix = torch.ones((2, 8), dtype=torch.float32)

        def step(self, *_args, **_kwargs):
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=torch.zeros((7,), dtype=torch.float32),
                    action_chunk=None,
                    action_condition_tokens=self.prefix,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = None

        def encode_observation(self, _observation):
            return torch.zeros((2, 8), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            del semantic_override
            self.seen_prefix = extra_prefix_tokens.clone()
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": torch.full((7,), 2.0, dtype=torch.float32),
                "predicted_chunk": torch.as_tensor(action_chunk_target, dtype=torch.float32) + 1.0,
            }

    target = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    core = _LegacyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic)

    result = policy.forward_train_transition(
        _dummy_observation(),
        action_chunk_target=target,
    )

    torch.testing.assert_close(semantic.seen_prefix, core.prefix)
    torch.testing.assert_close(result.output.state.predictive.action, torch.full((7,), 2.0, dtype=torch.float32))
    torch.testing.assert_close(result.output.state.predictive.action_chunk, target + 1.0)


def test_policy_forward_train_transition_ablated_bypasses_picf_prefix_and_state() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = "unset"

        def encode_observation(self, _observation):
            return torch.zeros((2, 8), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def compute_action_flow_loss(self, semantic_override, *, extra_prefix_tokens, action_chunk_target):
            assert semantic_override.shape == (2, 8)
            self.seen_prefix = extra_prefix_tokens
            target = torch.as_tensor(action_chunk_target, dtype=torch.float32)
            return {
                "total": torch.tensor(0.25),
                "action_pos": torch.tensor(0.1),
                "action_rot": torch.tensor(0.1),
                "action_gripper": torch.tensor(0.05),
                "predicted_action": target[0],
                "predicted_chunk": target,
            }

    target = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=_DummyCore(), semantic_encoder=semantic, picf_enabled=False)

    result = policy.forward_train_transition(
        _dummy_observation(),
        action_chunk_target=target,
    )

    assert semantic.seen_prefix is None
    assert result.output is None
    assert result.next_state is None
    torch.testing.assert_close(result.flow_override["predicted_chunk"], target)


def test_policy_act_ablated_samples_without_picf_state() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = "unset"

        def encode_observation(self, _observation):
            return torch.ones((2, 8), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def sample_action_chunk(self, semantic_override, *, extra_prefix_tokens):
            assert semantic_override.shape == (2, 8)
            self.seen_prefix = extra_prefix_tokens
            return torch.arange(14, dtype=torch.float32).reshape(2, 7)

    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=_DummyCore(), semantic_encoder=semantic, picf_enabled=False)

    result = policy.act(_dummy_observation())

    assert semantic.seen_prefix is None
    assert result.state is None
    assert result.output is None
    torch.testing.assert_close(result.action, torch.arange(7, dtype=torch.float32))
    torch.testing.assert_close(result.action_chunk, torch.arange(14, dtype=torch.float32).reshape(2, 7))


def test_policy_act_can_reuse_picf_observe_prefix_for_inference_latency(monkeypatch) -> None:
    monkeypatch.setenv("OPENPI_PICF_TIMING_BREAKDOWN", "1")
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)
            self.observe_calls = 0

        def observe_step(self, *_args, **_kwargs):
            self.observe_calls += 1
            prefix = torch.full((2, 4), float(self.observe_calls), dtype=torch.float32)
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(pi_prefix_tokens=prefix),
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            chunk = torch.as_tensor(action_future, dtype=torch.float32)
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=chunk[0],
                    action_chunk=chunk,
                )
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix_values: list[float] = []

        def encode_observation(self, _observation):
            return torch.ones((2, 8), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def sample_action_chunk(self, _semantic_override, *, extra_prefix_tokens):
            self.seen_prefix_values.append(float(extra_prefix_tokens[0, 0]))
            return torch.zeros((2, 7), dtype=torch.float32)

    core = _DummyCore()
    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=core, semantic_encoder=semantic, inference_observe_interval=3)

    first = _dummy_observation()
    second = dataclasses.replace(_dummy_observation(), reset_scaffold=False, step_id=2)
    third = dataclasses.replace(_dummy_observation(), reset_scaffold=False, step_id=3)
    fourth = dataclasses.replace(_dummy_observation(), reset_scaffold=False, step_id=4)

    result1 = policy.act(first)
    result2 = policy.act(second, previous=result1.state)
    result3 = policy.act(third, previous=result2.state)
    result4 = policy.act(fourth, previous=result3.state)

    assert core.observe_calls == 2
    assert semantic.seen_prefix_values == [1.0, 1.0, 1.0, 2.0]
    assert result2.debug["timing"]["picf_observe_reused"] == 1.0
    assert result3.debug["timing"]["picf_observe_reused"] == 1.0
    assert result4.debug["timing"]["picf_observe_reused"] == 0.0


def test_policy_act_sanitizes_picf_inference_prefix_before_action_sampler(monkeypatch) -> None:
    monkeypatch.setenv("OPENPI_PICF_INFERENCE_PREFIX_VALUE_CLIP", "10")
    monkeypatch.setenv("OPENPI_PICF_INFERENCE_PREFIX_MAX_RMS", "4")

    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

        def observe_step(self, *_args, **_kwargs):
            prefix = torch.tensor([[float("nan"), float("inf"), 1000.0, -1000.0]], dtype=torch.float32)
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(pi_prefix_tokens=prefix),
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            chunk = torch.as_tensor(action_future, dtype=torch.float32)
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=chunk[0],
                    action_chunk=chunk,
                ),
                posterior=types.SimpleNamespace(tokens=torch.zeros((1, 4), dtype=torch.float32)),
                conditioned_control=types.SimpleNamespace(pi_prefix_tokens=torch.zeros((1, 4), dtype=torch.float32)),
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.seen_prefix = None

        def encode_observation(self, _observation):
            return torch.ones((1, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def sample_action_chunk(self, _semantic_override, *, extra_prefix_tokens):
            self.seen_prefix = extra_prefix_tokens.detach().clone()
            assert torch.isfinite(extra_prefix_tokens).all()
            assert float(extra_prefix_tokens.abs().max()) <= 10.0
            rms = torch.sqrt(torch.mean(extra_prefix_tokens.square(), dim=-1))
            assert float(rms.max()) <= 4.0001
            return torch.zeros((1, 7), dtype=torch.float32)

    semantic = _SemanticEncoder()
    policy = PicfPi05Policy(core=_DummyCore(), semantic_encoder=semantic)
    result = policy.act(_dummy_observation())

    assert semantic.seen_prefix is not None
    assert result.debug["inference_prefix_nonfinite"] == 1.0
    assert result.debug["inference_prefix_value_clipped"] == 1.0
    assert result.debug["inference_prefix_rms_clipped"] == 1.0


def test_policy_act_reuses_last_finite_chunk_when_sampler_returns_nonfinite() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=torch.ones((1, 4), dtype=torch.float32),
                )
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            chunk = torch.as_tensor(action_future, dtype=torch.float32)
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=chunk[0],
                    action_chunk=chunk,
                ),
                posterior=types.SimpleNamespace(tokens=torch.zeros((1, 4), dtype=torch.float32)),
                conditioned_control=types.SimpleNamespace(pi_prefix_tokens=torch.zeros((1, 4), dtype=torch.float32)),
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode_observation(self, _observation):
            return torch.ones((1, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def sample_action_chunk(self, _semantic_override, *, extra_prefix_tokens):
            del extra_prefix_tokens
            self.calls += 1
            if self.calls == 1:
                return torch.arange(7, dtype=torch.float32).reshape(1, 7)
            return torch.full((1, 7), float("nan"), dtype=torch.float32)

    policy = PicfPi05Policy(core=_DummyCore(), semantic_encoder=_SemanticEncoder())
    first = policy.act(_dummy_observation())
    second = policy.act(dataclasses.replace(_dummy_observation(), reset_scaffold=False, step_id=2), previous=first.state)

    torch.testing.assert_close(second.action_chunk, first.action_chunk)
    torch.testing.assert_close(second.action, first.action)
    assert second.debug["inference_action_chunk_nonfinite"] == 1.0
    assert second.debug["inference_action_chunk_fallback_last"] == 1.0


def test_policy_act_sanitizes_nonfinite_recurrent_state_before_return() -> None:
    class _DummyCore:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(require_pi0_action_generator=True)

        def observe_step(self, *_args, **_kwargs):
            return types.SimpleNamespace(
                conditioned_control=types.SimpleNamespace(
                    pi_prefix_tokens=torch.ones((1, 4), dtype=torch.float32),
                )
            )

        def finalize_with_action(self, _observation, _observed, *, action_future):
            chunk = torch.as_tensor(action_future, dtype=torch.float32)
            state = types.SimpleNamespace(
                predictive=types.SimpleNamespace(
                    action=chunk[0],
                    action_chunk=chunk,
                ),
                posterior=types.SimpleNamespace(
                    tokens=torch.tensor([[1.0, float("nan"), float("inf"), -float("inf")]], dtype=torch.float32),
                    slot_address=torch.zeros((1, 4), dtype=torch.float32),
                    slot_content=torch.zeros((1, 4), dtype=torch.float32),
                ),
                conditioned_control=types.SimpleNamespace(pi_prefix_tokens=torch.zeros((1, 4), dtype=torch.float32)),
            )
            return types.SimpleNamespace(state=state, debug={})

    class _SemanticEncoder:
        def encode_observation(self, _observation):
            return torch.ones((1, 4), dtype=torch.float32)

        def supports_pi0_action_generation(self):
            return True

        def sample_action_chunk(self, _semantic_override, *, extra_prefix_tokens):
            del extra_prefix_tokens
            return torch.zeros((1, 7), dtype=torch.float32)

    policy = PicfPi05Policy(core=_DummyCore(), semantic_encoder=_SemanticEncoder())
    result = policy.act(_dummy_observation())

    assert result.debug["inference_state_sanitized"] == 1.0
    assert torch.isfinite(result.state.posterior.tokens).all()
