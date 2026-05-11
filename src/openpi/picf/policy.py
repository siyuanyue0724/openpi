from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import numpy as np
import torch

from openpi.picf.contracts import PicfObservation
from openpi.picf.core import PicfCoreOutput
from openpi.picf.core import PicfCoreState
from openpi.picf.core import PicfFullCore
from openpi.picf.core import PicfPreviousState
from openpi.picf.fsdp_utils import call_module_forward_or_method
@dataclasses.dataclass(frozen=True)
class PicfPolicyTrainResult:
    output: PicfCoreOutput | None
    observed: Any | None
    semantic_override: Any | None
    flow_override: dict[str, torch.Tensor] | None
    next_state: PicfPreviousState | None


@dataclasses.dataclass(frozen=True)
class PicfPolicyActResult:
    action: torch.Tensor
    action_chunk: torch.Tensor | None
    state: PicfCoreState | None
    debug: dict[str, float]
    output: PicfCoreOutput | None


class PicfPi05Policy:
    def __init__(
        self,
        *,
        core: PicfFullCore,
        semantic_encoder: torch.nn.Module | None,
        picf_enabled: bool = True,
    ) -> None:
        self.core = core
        self.semantic_encoder = semantic_encoder
        self.picf_enabled = bool(picf_enabled)

    def _supports_action_generation(self) -> bool:
        return bool(
            self.semantic_encoder is not None
            and bool(getattr(self.semantic_encoder, "supports_pi0_action_generation", lambda: False)())
        )

    def _requires_action_generation(self) -> bool:
        return bool(getattr(getattr(self.core, "config", None), "require_pi0_action_generator", False))

    def _require_action_generation(self) -> None:
        if not self._requires_action_generation():
            return
        if not self._supports_action_generation():
            raise RuntimeError(
                "PICF v2.2 requires PI0.5 action generation. "
                "No semantic action generator is available for this policy."
            )

    def _legacy_action_condition_tokens(self, output: PicfCoreOutput) -> Any:
        predictive = getattr(output.state, "predictive", None)
        prefix = None if predictive is None else getattr(predictive, "action_condition_tokens", None)
        if prefix is None:
            raise RuntimeError(
                "Legacy PICF core fallback requires predictive.action_condition_tokens "
                "to drive PI0.5 action generation."
            )
        return prefix

    def _action_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if bool(getattr(getattr(self.core, "config", None), "action_prefix_stopgrad", False)):
            return tokens.detach()
        return tokens

    def encode_semantic(self, observation: PicfObservation) -> Any | None:
        if self.semantic_encoder is None:
            return None
        return call_module_forward_or_method(self.semantic_encoder, "encode_observation", "encode_observation", observation)

    def recurrent_state(self, state: PicfCoreState | None) -> PicfPreviousState | None:
        if state is None:
            return None
        if hasattr(self.core, "make_recurrent_carry"):
            return self.core.make_recurrent_carry(state)
        return state

    def burnin_recurrent_transition(
        self,
        current: PicfObservation,
        *,
        previous: PicfPreviousState | None = None,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
    ) -> PicfPreviousState:
        if not self.picf_enabled:
            raise RuntimeError("PICF recurrent burn-in requires picf_enabled=True.")
        if not hasattr(self.core, "recurrent_burnin_step"):
            fallback = self.forward_train_transition(
                current,
                previous=previous,
                point_features_override=point_features_override,
                visual_map_override=visual_map_override,
                action_chunk_target=None,
            )
            if fallback.next_state is None:
                raise RuntimeError("PICF burn-in fallback did not produce a recurrent state.")
            return fallback.next_state
        return self.core.recurrent_burnin_step(
            current,
            previous=previous,
            point_features_override=point_features_override,
            visual_map_override=visual_map_override,
            action_future=current.action_chunk if current.action_chunk is not None else current.action,
        )

    def _pi05_only_train_transition(
        self,
        current: PicfObservation,
        *,
        semantic_override: Any | None,
        action_chunk_target: torch.Tensor | np.ndarray | None,
    ) -> PicfPolicyTrainResult:
        self._require_action_generation()
        teacher_action = self._teacher_forced_action_future(
            current,
            action_chunk_target=action_chunk_target,
        )
        if teacher_action is None:
            raise RuntimeError(
                "PI0.5-only ablation training requires a teacher-forced action or action chunk target."
            )
        flow_override = call_module_forward_or_method(
            self.semantic_encoder,
            "compute_action_flow_loss",
            "compute_action_flow_loss",
            semantic_override,
            extra_prefix_tokens=None,
            action_chunk_target=teacher_action,
        )
        return PicfPolicyTrainResult(
            output=None,
            observed=None,
            semantic_override=semantic_override,
            flow_override=flow_override,
            next_state=None,
        )

    @staticmethod
    def _action_from_sampled_chunk(action_chunk: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.as_tensor(action_chunk)
        if chunk.ndim == 1:
            return chunk[:7], chunk
        return chunk[0, :7], chunk

    def _legacy_core_step(
        self,
        observation: PicfObservation,
        *,
        previous: PicfCoreState | None,
        point_features_override: torch.Tensor | np.ndarray | None,
        visual_map_override: torch.Tensor | np.ndarray | None,
        semantic_override: Any | None,
        action_future: torch.Tensor | np.ndarray | None,
    ) -> PicfCoreOutput:
        step = self.core.step
        signature = inspect.signature(step)
        kwargs: dict[str, Any] = {}
        for name, value in (
            ("previous", previous),
            ("point_features_override", point_features_override),
            ("visual_map_override", visual_map_override),
            ("semantic_override", semantic_override),
            ("action_future", action_future),
        ):
            if name in signature.parameters:
                kwargs[name] = value
        return step(observation, **kwargs)

    def _teacher_forced_action_future(
        self,
        observation: PicfObservation,
        *,
        action_chunk_target: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor | np.ndarray | None:
        if action_chunk_target is not None:
            return action_chunk_target
        if observation.action is not None:
            return observation.action
        return None

    def forward_train_transition(
        self,
        current: PicfObservation,
        *,
        previous: PicfCoreState | None = None,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
        semantic_override: Any | None = None,
        action_chunk_target: torch.Tensor | np.ndarray | None = None,
    ) -> PicfPolicyTrainResult:
        if semantic_override is None and self.semantic_encoder is not None:
            semantic_override = self.encode_semantic(current)
        if not self.picf_enabled:
            return self._pi05_only_train_transition(
                current,
                semantic_override=semantic_override,
                action_chunk_target=action_chunk_target,
            )
        if not hasattr(self.core, "observe_step"):
            output = self._legacy_core_step(
                current,
                previous=previous,
                point_features_override=point_features_override,
                visual_map_override=visual_map_override,
                semantic_override=semantic_override,
                action_future=current.action,
            )
            flow_override: dict[str, torch.Tensor] | None = None
            if action_chunk_target is not None:
                self._require_action_generation()
                if self._supports_action_generation():
                    flow_override = call_module_forward_or_method(
                        self.semantic_encoder,
                        "compute_action_flow_loss",
                        "compute_action_flow_loss",
                        semantic_override,
                        extra_prefix_tokens=self._action_prefix_tokens(self._legacy_action_condition_tokens(output)),
                        action_chunk_target=action_chunk_target,
                    )
                    output.state.predictive.action = flow_override["predicted_action"]
                    output.state.predictive.action_chunk = flow_override["predicted_chunk"]
            return PicfPolicyTrainResult(
                output=output,
                observed=None,
                semantic_override=semantic_override,
                flow_override=flow_override,
                next_state=self.recurrent_state(output.state),
            )
        observed = self.core.observe_step(
            current,
            previous=previous,
            point_features_override=point_features_override,
            visual_map_override=visual_map_override,
            semantic_override=semantic_override,
        )
        flow_override: dict[str, torch.Tensor] | None = None
        if action_chunk_target is not None:
            self._require_action_generation()
            flow_override = call_module_forward_or_method(
                self.semantic_encoder,
                "compute_action_flow_loss",
                "compute_action_flow_loss",
                semantic_override,
                extra_prefix_tokens=self._action_prefix_tokens(observed.conditioned_control.pi_prefix_tokens),
                action_chunk_target=action_chunk_target,
            )
        output = self.core.finalize_with_action(
            current,
            observed,
            action_future=self._teacher_forced_action_future(
                current,
                action_chunk_target=action_chunk_target,
            ),
        )
        if flow_override is not None:
            output.state.predictive.action = flow_override["predicted_action"]
            output.state.predictive.action_chunk = flow_override["predicted_chunk"]
        return PicfPolicyTrainResult(
            output=output,
            observed=observed,
            semantic_override=semantic_override,
            flow_override=flow_override,
            next_state=self.recurrent_state(output.state),
        )

    @torch.no_grad()
    def act(
        self,
        observation: PicfObservation,
        *,
        previous: PicfCoreState | None = None,
        point_features_override: torch.Tensor | np.ndarray | None = None,
        visual_map_override: torch.Tensor | np.ndarray | None = None,
        semantic_override: Any | None = None,
    ) -> PicfPolicyActResult:
        if semantic_override is None:
            semantic_override = self.encode_semantic(observation)
        if not self.picf_enabled:
            self._require_action_generation()
            action_chunk = call_module_forward_or_method(
                self.semantic_encoder,
                "sample_action_chunk",
                "sample_action_chunk",
                semantic_override,
                extra_prefix_tokens=None,
            )
            action, normalized_chunk = self._action_from_sampled_chunk(action_chunk)
            return PicfPolicyActResult(
                action=action,
                action_chunk=normalized_chunk,
                state=None,
                debug={"picf_enabled": 0.0},
                output=None,
            )
        if not hasattr(self.core, "observe_step"):
            self._require_action_generation()
            output = self._legacy_core_step(
                observation,
                previous=previous,
                point_features_override=point_features_override,
                visual_map_override=visual_map_override,
                semantic_override=semantic_override,
                action_future=None,
            )
            if self._supports_action_generation():
                action_chunk = call_module_forward_or_method(
                    self.semantic_encoder,
                    "sample_action_chunk",
                    "sample_action_chunk",
                    semantic_override,
                    extra_prefix_tokens=self._legacy_action_condition_tokens(output),
                )
                if not hasattr(self.core, "refresh_predictive_state_for_action"):
                    raise RuntimeError(
                        "Legacy PICF core fallback requires refresh_predictive_state_for_action "
                        "to finalize PI0.5 sampled actions."
                    )
                output.state.predictive = self.core.refresh_predictive_state_for_action(
                    observation,
                    output.state,
                    action_future=action_chunk,
                )
            return PicfPolicyActResult(
                action=output.state.predictive.action,
                action_chunk=getattr(output.state.predictive, "action_chunk", None),
                state=output.state,
                debug=output.debug,
                output=output,
            )
        self._require_action_generation()
        observed = self.core.observe_step(
            observation,
            previous=previous,
            point_features_override=point_features_override,
            visual_map_override=visual_map_override,
            semantic_override=semantic_override,
        )
        action_chunk = call_module_forward_or_method(
            self.semantic_encoder,
            "sample_action_chunk",
            "sample_action_chunk",
            semantic_override,
            extra_prefix_tokens=observed.conditioned_control.pi_prefix_tokens,
        )
        output = self.core.finalize_with_action(observation, observed, action_future=action_chunk)
        return PicfPolicyActResult(
            action=output.state.predictive.action,
            action_chunk=output.state.predictive.action_chunk,
            state=output.state,
            debug=output.debug,
            output=output,
        )
