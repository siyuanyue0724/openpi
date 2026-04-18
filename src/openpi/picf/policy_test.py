from __future__ import annotations

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
