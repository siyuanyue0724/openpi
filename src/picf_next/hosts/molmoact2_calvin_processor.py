"""Official MolmoAct2 preprocessing split for causal CALVIN training.

The observation path never receives demonstrator actions. The target path
reuses the pinned official action normalization and padding steps without
tokenizing images a second time. Both paths fail closed on upstream pipeline
drift so a LeRobot update cannot silently change the training boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from picf_next.data.calvin import (
    CALVIN_HOST_IMAGE_KEYS,
    CalvinMolmoAct2SourceObservation,
    CalvinPICFEvidenceFrame,
    CalvinStatefulTransitionSample,
)
from picf_next.hosts.molmoact2_training import MolmoAct2HostObservationView

_OFFICIAL_STEP_TYPES = (
    "RenameObservationsProcessorStep",
    "AddBatchDimensionProcessorStep",
    "MolmoAct2MaskedNormalizerProcessorStep",
    "MolmoAct2ClampNormalizedProcessorStep",
    "MolmoAct2PackInputsProcessorStep",
    "DeviceProcessorStep",
)
_MODEL_OBSERVATION_KEYS = frozenset(
    {
        "input_ids",
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
        "pixel_values_videos",
        "video_token_pooling",
        "video_grids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
    }
)
_ACTION_TARGET_KEYS = frozenset({"action", "action_dim_is_pad", "action_horizon_is_pad"})
_CALVIN_RGB_TO_HOST = {
    "observation.images.rgb_static": CALVIN_HOST_IMAGE_KEYS[0],
    "observation.images.rgb_gripper": CALVIN_HOST_IMAGE_KEYS[1],
}


def _frozen_steps(preprocessor: object) -> tuple[object, ...]:
    steps = tuple(getattr(preprocessor, "steps", ()))
    observed = tuple(type(step).__name__ for step in steps)
    if observed != _OFFICIAL_STEP_TYPES:
        raise ValueError(
            "MolmoAct2 preprocessing steps differ from the pinned official pipeline: "
            f"expected={_OFFICIAL_STEP_TYPES}, observed={observed}"
        )
    return steps


class CalvinMolmoAct2ProcessorBridge:
    """Split one official processor into target-free and target-only callables."""

    def __init__(self, preprocessor: object, *, action_horizon: int) -> None:
        if not callable(preprocessor):
            raise TypeError("MolmoAct2 preprocessor must be callable")
        if (
            not isinstance(action_horizon, int)
            or isinstance(action_horizon, bool)
            or action_horizon <= 0
        ):
            raise ValueError("MolmoAct2 action horizon must be positive")
        steps = _frozen_steps(preprocessor)
        pack = steps[4]
        if getattr(pack, "action_mode", None) != "continuous":
            raise ValueError("causal CALVIN training requires continuous MolmoAct2 actions")
        if getattr(pack, "chunk_size", None) != action_horizon:
            raise ValueError("MolmoAct2 processor chunk size differs from the CALVIN horizon")
        if getattr(pack, "env_action_dim", None) != 7:
            raise ValueError("MolmoAct2 processor must declare the seven CALVIN action axes")
        if getattr(pack, "max_action_dim", None) != 32:
            raise ValueError("released MolmoAct2 action padding width changed")
        if not callable(getattr(pack, "_pad_action", None)):
            raise TypeError("pinned MolmoAct2 packer no longer exposes its action padding")

        self.preprocessor = preprocessor
        self.action_horizon = action_horizon
        self._normalizer = steps[2]
        self._clamp = steps[3]
        self._packer = pack
        self._device = steps[5]

    @classmethod
    def from_official_config(
        cls,
        config: object,
        *,
        dataset_stats: dict[str, dict[str, Any]],
        dataset_meta: object | None = None,
    ) -> CalvinMolmoAct2ProcessorBridge:
        """Construct through the pinned public LeRobot processor factory."""

        from lerobot.policies.molmoact2.processor_molmoact2 import (
            make_molmoact2_pre_post_processors,
        )

        if getattr(config, "action_mode", None) != "continuous":
            raise ValueError("CALVIN processor construction requires action_mode='continuous'")
        preprocessor, _postprocessor = make_molmoact2_pre_post_processors(
            config,
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )
        return cls(preprocessor, action_horizon=int(config.chunk_size))

    def build_observation_inputs(
        self,
        evidence_prefixes: tuple[tuple[CalvinPICFEvidenceFrame, ...], ...],
        host_views: tuple[MolmoAct2HostObservationView, ...],
    ) -> Mapping[str, torch.Tensor]:
        """Run official image/text/state processing with no action argument path."""

        if not evidence_prefixes or len(evidence_prefixes) != len(host_views):
            raise ValueError("CALVIN evidence prefixes and host views must form one nonempty batch")

        images: dict[str, list[np.ndarray]] = {key: [] for key in CALVIN_HOST_IMAGE_KEYS}
        states: list[tuple[float, ...]] = []
        tasks: list[str] = []
        for prefix, view in zip(evidence_prefixes, host_views, strict=True):
            if len(prefix) != 1:
                raise ValueError("stateful 0+1 preprocessing requires exactly one current frame")
            frame = prefix[0]
            if abs(frame.timestamp_s - view.timestamp_s) > 1e-7:
                raise ValueError("CALVIN host view and evidence frame timestamps differ")
            if not all(view.state_valid):
                raise ValueError(
                    "MolmoAct2 has no missing-state mask; CALVIN state must be complete"
                )
            sensor_by_key = {sensor.key: sensor.value for sensor in frame.sensor_observations}
            if len(sensor_by_key) != len(frame.sensor_observations):
                raise ValueError("CALVIN evidence frame contains duplicate sensor keys")
            for source_key, host_key in _CALVIN_RGB_TO_HOST.items():
                value = sensor_by_key.get(source_key)
                if value is None:
                    raise ValueError(f"CALVIN evidence frame is missing {source_key}")
                images[host_key].append(np.asarray(value))
            states.append(view.state)
            tasks.append(view.task)

        return self._process_observation_batch(
            images=images,
            states=states,
            tasks=tasks,
        )

    def build_source_observation_inputs(
        self,
        observations: tuple[CalvinMolmoAct2SourceObservation, ...],
    ) -> Mapping[str, torch.Tensor]:
        """Run the official processor without supplying any language field."""

        if not observations or any(
            not isinstance(observation, CalvinMolmoAct2SourceObservation)
            for observation in observations
        ):
            raise ValueError("MolmoAct2 source observations must form one nonempty typed batch")
        if any(not observation.state_valid.all() for observation in observations):
            raise ValueError("MolmoAct2 has no missing-state mask; source state must be complete")
        images = {
            key: [np.asarray(observation.images[key]) for observation in observations]
            for key in CALVIN_HOST_IMAGE_KEYS
        }
        states = [
            tuple(float(value) for value in observation.state) for observation in observations
        ]
        return self._process_observation_batch(
            images=images,
            states=states,
            tasks=None,
        )

    def _process_observation_batch(
        self,
        *,
        images: Mapping[str, list[np.ndarray]],
        states: list[tuple[float, ...]],
        tasks: list[str] | None,
    ) -> Mapping[str, torch.Tensor]:
        """Materialize one target-free official observation batch."""

        if not states or set(images) != set(CALVIN_HOST_IMAGE_KEYS):
            raise ValueError("MolmoAct2 observation batch fields are incomplete")
        if any(len(values) != len(states) for values in images.values()):
            raise ValueError("MolmoAct2 image batch lengths differ from state batch")
        if tasks is not None and len(tasks) != len(states):
            raise ValueError("MolmoAct2 task batch length differs from observations")
        batch: dict[str, Any] = {
            key: torch.from_numpy(np.stack(values)) for key, values in images.items()
        }
        batch["observation.state"] = torch.as_tensor(states, dtype=torch.float32)
        if tasks is not None:
            batch["task"] = tasks
        processed = self.preprocessor(batch)
        if not isinstance(processed, Mapping):
            raise TypeError("official MolmoAct2 preprocessor must return one mapping")
        if processed.get("action") is not None or "labels" in processed:
            raise ValueError("target-free MolmoAct2 processing emitted an action or labels")
        inputs = {key: value for key, value in processed.items() if key in _MODEL_OBSERVATION_KEYS}
        if "input_ids" not in inputs or "pixel_values" not in inputs:
            raise ValueError("MolmoAct2 CALVIN processing omitted required visual model inputs")
        if any(not isinstance(value, torch.Tensor) for value in inputs.values()):
            raise TypeError("MolmoAct2 model inputs must all be tensors")
        if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != len(states):
            raise ValueError("MolmoAct2 input_ids differ from the CALVIN batch size")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and attention_mask.shape != inputs["input_ids"].shape:
            raise ValueError("MolmoAct2 attention mask does not align with input_ids")
        return inputs

    def build_action_targets(
        self,
        samples: tuple[CalvinStatefulTransitionSample, ...],
    ) -> Mapping[str, torch.Tensor]:
        """Apply official action normalization/clamp/padding without image work."""

        if not samples:
            raise ValueError("CALVIN action target batch cannot be empty")
        if any(sample.host_sample.action.shape != (self.action_horizon, 7) for sample in samples):
            raise ValueError("CALVIN action chunks differ from the processor contract")

        from lerobot.types import TransitionKey

        action = torch.from_numpy(np.stack([sample.host_sample.action for sample in samples]))
        action_is_pad = torch.from_numpy(
            np.stack([sample.host_sample.action_is_pad for sample in samples])
        )
        transition: dict[Any, Any] = {
            TransitionKey.ACTION: action,
            TransitionKey.COMPLEMENTARY_DATA: {},
        }
        transition = self._normalizer(transition)
        transition = self._clamp(transition)
        normalized = transition.get(TransitionKey.ACTION)
        if not isinstance(normalized, torch.Tensor):
            raise TypeError("official MolmoAct2 normalizer did not return tensor actions")
        padded, horizon_is_pad, dim_is_pad = self._packer._pad_action(
            normalized,
            action_is_pad,
        )
        transition[TransitionKey.ACTION] = padded
        transition[TransitionKey.COMPLEMENTARY_DATA] = {
            "action_dim_is_pad": dim_is_pad,
            "action_horizon_is_pad": horizon_is_pad,
        }
        transition = self._device(transition)
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if not isinstance(complementary, Mapping):
            raise TypeError("MolmoAct2 target padding omitted complementary masks")
        targets = {
            "action": transition[TransitionKey.ACTION],
            "action_dim_is_pad": complementary["action_dim_is_pad"],
            "action_horizon_is_pad": complementary["action_horizon_is_pad"],
        }
        if set(targets) != _ACTION_TARGET_KEYS or any(
            not isinstance(value, torch.Tensor) for value in targets.values()
        ):
            raise TypeError("MolmoAct2 action target contract changed")
        if targets["action"].shape != (len(samples), self.action_horizon, 32):
            raise ValueError("MolmoAct2 padded action target shape changed")
        if targets["action_dim_is_pad"].shape != (len(samples), 32):
            raise ValueError("MolmoAct2 action-dimension padding shape changed")
        if targets["action_horizon_is_pad"].shape != (
            len(samples),
            self.action_horizon,
        ):
            raise ValueError("MolmoAct2 action-horizon padding shape changed")
        if not torch.isfinite(targets["action"]).all():
            raise ValueError("MolmoAct2 normalized action targets are non-finite")
        return targets
