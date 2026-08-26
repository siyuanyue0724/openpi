"""CALVIN raw-input and encoded-evidence boundaries for PICF modalities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data.calvin import CALVIN_CONTRACT
from picf_next.data.calvin_tactile import CALVIN_TACTILE_STREAM_NAMES
from picf_next.data.robot_record import RobotTransitionRecord


@dataclass(frozen=True, slots=True)
class CalvinEncoderInputs:
    static_rgb: NDArray[np.uint8]
    wrist_rgb: NDArray[np.uint8]
    static_depth: NDArray[np.float32]
    wrist_depth: NDArray[np.float32]
    tactile_rgb: tuple[NDArray[np.uint8], NDArray[np.uint8]]
    tactile_depth: tuple[NDArray[np.float32], NDArray[np.float32]]
    timestamp_s: float


def validate_calvin_evidence_timestamps(
    evidence: tuple[DenseEvidence, ...],
    *,
    observation_timestamp_s: float,
) -> None:
    """Reject cached evidence that lies after the observation it conditions."""

    if not isinstance(evidence, tuple) or any(
        not isinstance(item, DenseEvidence) for item in evidence
    ):
        raise ContractError("CALVIN evidence timestamps require a typed evidence tuple")
    if not np.isfinite(observation_timestamp_s) or observation_timestamp_s < 0.0:
        raise ContractError("CALVIN observation timestamp is invalid")
    # Frozen encoder contracts store timestamp metadata as float32. Accept the
    # exact upward-rounded representation of this observation, but no later
    # float32 value; one CALVIN frame is orders of magnitude farther away.
    maximum_causal_timestamp = max(
        observation_timestamp_s + 1e-7,
        float(np.float32(observation_timestamp_s)),
    )
    for item in evidence:
        if item.token_count and (item.timestamps > maximum_causal_timestamp).any():
            raise ContractError(f"{item.modality} evidence contains a future timestamp")


def calvin_encoder_inputs(record: RobotTransitionRecord) -> CalvinEncoderInputs:
    """Expose every CALVIN sensor array without preprocessing or target data."""

    if record.contract != CALVIN_CONTRACT:
        raise ContractError("CALVIN encoder input builder received another dataset contract")
    values = {observation.key: observation.value for observation in record.array_observations}
    required = {
        "observation.images.rgb_static",
        "observation.images.rgb_gripper",
        "observation.depth.static",
        "observation.depth.gripper",
        "observation.tactile.rgb",
        "observation.tactile.depth",
    }
    if set(values) != required:
        raise ContractError("CALVIN encoder inputs differ from the complete sensor contract")
    tactile_rgb = values["observation.tactile.rgb"]
    tactile_depth = values["observation.tactile.depth"]
    return CalvinEncoderInputs(
        static_rgb=values["observation.images.rgb_static"],
        wrist_rgb=values["observation.images.rgb_gripper"],
        static_depth=values["observation.depth.static"],
        wrist_depth=values["observation.depth.gripper"],
        tactile_rgb=(tactile_rgb[..., :3], tactile_rgb[..., 3:]),
        tactile_depth=(tactile_depth[..., 0], tactile_depth[..., 1]),
        timestamp_s=record.timestamp_s,
    )


def validate_calvin_encoded_evidence(
    record: RobotTransitionRecord,
    evidence: tuple[DenseEvidence, ...],
    *,
    active_touch_streams: tuple[str, ...],
) -> None:
    """Validate real encoder products before they enter a PICF batch.

    Measurement validity is an explicit upstream signal. It is intentionally not
    guessed from tactile-image intensity inside this function.
    """

    if (
        not isinstance(active_touch_streams, tuple)
        or any(name not in CALVIN_TACTILE_STREAM_NAMES for name in active_touch_streams)
        or len(set(active_touch_streams)) != len(active_touch_streams)
        or active_touch_streams
        != tuple(name for name in CALVIN_TACTILE_STREAM_NAMES if name in active_touch_streams)
    ):
        raise ContractError("active_touch_streams must be an ordered unique CALVIN stream tuple")

    if record.contract != CALVIN_CONTRACT:
        raise ContractError("CALVIN evidence validator received another dataset contract")
    mapping = {item.modality: item for item in evidence}
    if len(mapping) != len(evidence):
        raise ContractError("CALVIN evidence contains duplicate modalities")
    required = {"vjepa", "sonata", "anytouch"}
    if set(mapping) != required:
        raise ContractError("CALVIN evidence must explicitly represent all three modalities")
    if not mapping["vjepa"].available or not mapping["vjepa"].token_count:
        raise ContractError("CALVIN V-JEPA evidence must retain its complete valid token set")
    if not mapping["sonata"].available or not mapping["sonata"].token_count:
        raise ContractError("CALVIN Sonata evidence must retain its complete valid token set")

    touch = mapping["anytouch"]
    if active_touch_streams:
        if not touch.available or not touch.token_count or touch.group_ids is None:
            raise ContractError("active CALVIN touch requires complete AnyTouch sensor groups")
        groups = np.unique(touch.group_ids)
        if not np.array_equal(groups, np.arange(len(active_touch_streams), dtype=groups.dtype)):
            raise ContractError("AnyTouch groups must map one-to-one to active sensor streams")
    elif touch.available or touch.token_count:
        raise ContractError("inactive CALVIN touch must inject no AnyTouch tokens")

    validate_calvin_evidence_timestamps(
        evidence,
        observation_timestamp_s=record.timestamp_s,
    )
