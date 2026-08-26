#!/usr/bin/env python3
"""Measure fixed-noise action dependence on a trained M4 PICF posterior.

This is a read-only checkpoint audit. It does not add a training branch or an
acceptance threshold. For each requested checkpoint rank it restores the exact
stream posterior, advances through a deterministic post-checkpoint observation
prefix, and evaluates consecutive action targets under causal evidence
interventions. Measurement-age strata separate rows whose expected measurement
age spans at least one nominal observation interval from rows with only
sub-frame residual age.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src"
_MOLMO_EXPERIMENTS = _ROOT / "references/source_checkouts/molmoact2-cloud/experiments"
for _path in (_ROOT, _SOURCE_ROOT, _MOLMO_EXPERIMENTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_AGE_SELECTIONS = (
    "any",
    "sub-reference-step",
    "at-least-one-reference-step",
)
_RELEVANCE_SELECTIONS = (
    "any",
    "task-relevant-hidden",
    "controlled-task-occlusion",
)
_FLOW_SEED_MODES = ("training-plan", "source-frame-matched")


@dataclass(frozen=True, slots=True)
class _PreparedActionStep:
    optimizer_plan_step: int
    planned_transition: Any
    sample: Any
    evidence: Any
    final_belief: Any
    policy_batch: Mapping[str, Any]
    flow_timesteps: Any
    flow_noise: Any
    action_condition_input_ids: Any
    vision_patch_layout: Any
    core_output: Any = None
    temporal_config: Any = None


@dataclass(frozen=True, slots=True)
class _AuditIdentityAttribution:
    next_keys_by_row: tuple[tuple[str | None, ...], ...]
    currently_measurable_identity_keys: tuple[str, ...]
    current_set_match_by_identity: Mapping[str, Mapping[str, Any]]
    track_conflicts: int
    task_selection: Any


@dataclass(frozen=True, slots=True)
class _VisualCameraSnapshot:
    image_key: str
    source: np.ndarray
    ownership: np.ndarray


@dataclass(frozen=True, slots=True)
class _ActionVisualSnapshot:
    optimizer_plan_step: int
    global_source_step: int
    episode_instance_id: str | None
    transition_index: int | None
    task: str
    task_key: str
    cameras: tuple[_VisualCameraSnapshot, ...]
    valid: np.ndarray
    address_mean: np.ndarray
    content_mean: np.ndarray
    geometry_mean: np.ndarray
    existence: np.ndarray
    visibility: np.ndarray
    measurement_age_s: np.ndarray
    log_prior: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=_ROOT / "configs/training/molmoact2_calvin_m4_action_adoption.json",
    )
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--foundation-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--stationary-acceptance-report", type=Path, required=True)
    parser.add_argument("--stationary-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--calvin-protocol-source-root",
        type=Path,
        help="optional CALVIN checkout whose reviewed evaluator sources must match exactly",
    )
    parser.add_argument(
        "--same-renderer-removal-dir",
        type=Path,
        help=(
            "optional immutable same-renderer factual/removed probe store; unmatched "
            "frames are skipped instead of falling back to synthetic RGB fill"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--extended-plan-steps", type=int, default=200)
    parser.add_argument("--maximum-pair-search-steps", type=int, default=16)
    parser.add_argument("--samples-per-rank", type=int, default=1)
    parser.add_argument(
        "--selection-age-stratum",
        choices=_AGE_SELECTIONS,
        default="any",
    )
    parser.add_argument(
        "--selection-relevance",
        choices=_RELEVANCE_SELECTIONS,
        default="any",
        help=(
            "optionally require either a naturally hidden task-target row or an "
            "evaluator-only controlled occlusion of a previously witnessed task target; "
            "neither mode changes training"
        ),
    )
    parser.add_argument("--maximum-samples-per-source-episode", type=int)
    parser.add_argument("--flow-randomness-repeats", type=int, default=1)
    parser.add_argument(
        "--flow-randomness-seed-mode",
        choices=_FLOW_SEED_MODES,
        default="training-plan",
    )
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument(
        "--temporal-visual-history-steps",
        type=int,
        default=0,
        help=(
            "retain this many contiguous CPU-side runtime snapshots and render each "
            "selected old row through time; zero disables temporal rendering"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise RuntimeError("audit source revision is not one full git commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("full-weight action intervention audit requires a clean worktree")
    return revision


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="ascii") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validated_checkpoint_file(
    checkpoint: Path,
    control: Mapping[str, Any],
    name: str,
) -> tuple[Path, str]:
    state_files = control.get("state_files")
    record = state_files.get(name) if isinstance(state_files, dict) else None
    path = checkpoint / name
    if (
        not isinstance(record, dict)
        or not path.is_file()
        or path.is_symlink()
        or record.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError(f"checkpoint file {name} is absent or differs in size")
    actual = _sha256(path)
    if record.get("sha256") != actual:
        raise ValueError(f"checkpoint file {name} differs from its control hash")
    return path, actual


def _checkpoint_contract(checkpoint: Path) -> tuple[dict[str, Any], Path, str]:
    control = _read_json(checkpoint / "picf_control.json", "checkpoint control")
    if control.get("schema") != "picf-next.checkpoint-control-manifest.v2":
        raise ValueError("unsupported training checkpoint control schema")
    model_path, model_sha256 = _validated_checkpoint_file(
        checkpoint,
        control,
        "model.safetensors",
    )
    return control, model_path, model_sha256


def _validate_extended_plan_prefix(reference: Any, extended: Any) -> int:
    if extended.total_steps <= reference.total_steps:
        raise ValueError("extended action-audit plan must be longer than the checkpoint plan")
    for step in range(reference.total_steps):
        if extended.global_batch(step) != reference.global_batch(step):
            raise ValueError(f"extended action-audit plan changed checkpoint step {step + 1}")
    return reference.total_steps


def _extend_plan_for_read_only_audit(reference: Any, *, total_steps: int) -> Any:
    """Extend deterministic samples without granting another optimizer step."""

    from picf_next.training.control import FrozenEpisodeStreamPlan

    if total_steps <= reference.total_steps:
        raise ValueError("read-only action-audit plan must exceed checkpoint plan length")
    return FrozenEpisodeStreamPlan(
        dataset_id=reference.dataset_id,
        dataset_revision=reference.dataset_revision,
        dataset_manifest_sha256=reference.dataset_manifest_sha256,
        episodes=reference.episodes,
        comparison_id=reference.comparison_id,
        seed=reference.seed,
        global_batch_size=reference.global_batch_size,
        total_steps=total_steps,
    )


def _intervention_indices(valid: Any, log_prior: Any) -> tuple[Any, Any, tuple[int, int]]:
    """Choose the two strongest valid rows without task- or dataset-specific rules."""

    import torch

    if valid.dtype != torch.bool or valid.ndim != 2 or valid.shape[0] != 1:
        raise ValueError("action intervention requires one boolean object-valid row")
    if log_prior.shape != valid.shape or log_prior.device != valid.device:
        raise ValueError("object log prior must align with object validity")
    valid_indices = torch.nonzero(valid[0], as_tuple=False).flatten()
    if valid_indices.numel() < 2:
        raise ValueError("wrong-address intervention requires at least two valid objects")
    ordered = valid_indices[
        torch.argsort(log_prior[0, valid_indices].float(), descending=True, stable=True)
    ]
    first, second = int(ordered[0].item()), int(ordered[1].item())
    permutation = torch.arange(valid.shape[1], device=valid.device, dtype=torch.long)
    permutation[first], permutation[second] = (
        permutation[second].clone(),
        permutation[first].clone(),
    )
    removed = torch.zeros_like(valid)
    removed[0, first] = True
    return permutation, removed, (first, second)


def _target_address_permutation(
    valid: Any,
    log_prior: Any,
    target_rows: Any,
) -> tuple[Any, tuple[int, ...]]:
    """Swap every selected target address with a strong valid control row."""

    import torch

    if valid.dtype != torch.bool or valid.ndim != 2 or valid.shape[0] != 1:
        raise ValueError("target-address intervention requires one boolean validity row")
    if log_prior.shape != valid.shape or log_prior.device != valid.device:
        raise ValueError("target-address intervention requires aligned object log priors")
    if (
        not isinstance(target_rows, torch.Tensor)
        or target_rows.dtype != torch.bool
        or target_rows.shape != valid.shape
        or target_rows.device != valid.device
    ):
        raise ValueError("target-address rows must align with posterior validity")
    if bool((target_rows & ~valid).any()):
        raise ValueError("every controlled target row must remain valid after occlusion")
    selected = torch.nonzero(target_rows[0], as_tuple=False).flatten()
    if selected.numel() == 0:
        raise ValueError("target-address intervention requires at least one target row")
    controls = torch.nonzero(valid[0] & ~target_rows[0], as_tuple=False).flatten()
    if controls.numel() < selected.numel():
        raise ValueError("target-address intervention lacks distinct valid control rows")
    ordered_controls = controls[
        torch.argsort(log_prior[0, controls].float(), descending=True, stable=True)
    ][: selected.numel()]
    permutation = torch.arange(valid.shape[1], device=valid.device, dtype=torch.long)
    for target, control in zip(selected.tolist(), ordered_controls.tolist(), strict=True):
        permutation[target], permutation[control] = (
            permutation[control].clone(),
            permutation[target].clone(),
        )
    return permutation, tuple(int(row) for row in ordered_controls.tolist())


def _fixed_flow_probe(
    *,
    policy: Any,
    action_adapter: Any,
    policy_batch: Mapping[str, Any],
    action_condition_input_ids: Any,
    evidence: Any,
    flow_timesteps: Any,
    flow_noise: Any,
) -> tuple[float, Any]:
    """Return official mean flow loss and its final velocity field."""

    captured: list[Any] = []

    def capture_velocity(_module: Any, _inputs: object, output: Any) -> None:
        captured.append(output.detach().float())

    final_layer = policy._action_expert().final_layer
    handle = final_layer.register_forward_hook(capture_velocity)
    try:
        context = action_adapter.prepare_picf_context(evidence)
        loss, _metrics = policy(
            dict(policy_batch),
            reduction="mean",
            action_layer_context=context,
            flow_timesteps=flow_timesteps,
            flow_noise=flow_noise,
            action_condition_input_ids=action_condition_input_ids,
        )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("fixed-flow intervention did not capture one velocity field")
    if loss.ndim != 0 or not bool(loss.isfinite()):
        raise ValueError("fixed-flow intervention produced a non-finite action loss")
    return float(loss.detach().float().cpu().item()), captured[0]


def _valid_velocity_rms(reference: Any, changed: Any, policy_batch: Mapping[str, Any]) -> float:
    import torch

    if reference.shape != changed.shape or reference.ndim != 3:
        raise ValueError("velocity fields must share [flow-batch, horizon, action] shape")
    action = policy_batch.get("action")
    if not isinstance(action, torch.Tensor) or action.ndim != 3:
        raise ValueError("velocity comparison requires the official action target")
    batch_size, horizon, action_dim = action.shape
    if reference.shape[0] % batch_size or reference.shape[1:] != (horizon, action_dim):
        raise ValueError("velocity field differs from official action geometry")
    flow_count = reference.shape[0] // batch_size
    valid = torch.ones(
        (batch_size, flow_count, horizon, action_dim),
        device=reference.device,
        dtype=torch.bool,
    )
    action_dim_is_pad = policy_batch.get("action_dim_is_pad")
    if action_dim_is_pad is not None:
        if action_dim_is_pad.shape != (batch_size, action_dim):
            raise ValueError("action-dimension padding mask is malformed")
        valid &= ~action_dim_is_pad.to(reference.device, dtype=torch.bool)[:, None, None, :]
    action_horizon_is_pad = policy_batch.get("action_horizon_is_pad")
    if action_horizon_is_pad is not None:
        if action_horizon_is_pad.shape != (batch_size, horizon):
            raise ValueError("action-horizon padding mask is malformed")
        valid &= ~action_horizon_is_pad.to(reference.device, dtype=torch.bool)[:, None, :, None]
    difference = (reference - changed).reshape(batch_size, flow_count, horizon, action_dim)
    selected = difference[valid]
    if selected.numel() == 0:
        raise ValueError("velocity comparison has no valid action coordinates")
    return float(selected.square().mean().sqrt().cpu().item())


def _prepare_action_step(
    stack: Any,
    planned_transition: Any,
    initial_belief: Any,
    *,
    optimizer_plan_step: int,
    evidence_frame_override: Any | None = None,
) -> _PreparedActionStep:
    from picf_next.data.calvin import CalvinPICFEvidenceFrame
    from picf_next.hosts.molmoact2_training import (
        action_evidence_from_core,
        assemble_calvin_stateful_molmoact2_transition,
        materialize_molmoact2_flow_randomness,
    )

    module = stack.module
    bridge = module.joint_bridge.sequence_bridge
    sample = module.dataset.by_key(planned_transition.sample.sample_key)
    if sample.episode_key != planned_transition.episode_key:
        raise ValueError("planned episode differs from the CALVIN sample")
    if sample.transition_index != planned_transition.transition_index:
        raise ValueError("planned transition index differs from the CALVIN sample")
    observation_builder = module.build_host_observation_inputs
    if evidence_frame_override is not None:
        if not isinstance(evidence_frame_override, CalvinPICFEvidenceFrame):
            raise TypeError("action observation override must be a CALVIN evidence frame")
        source_frame = sample.picf_evidence_frame
        source_sensors = source_frame.sensor_observations
        changed_sensors = evidence_frame_override.sensor_observations
        if (
            evidence_frame_override.timestamp_s != source_frame.timestamp_s
            or evidence_frame_override.delta_t_s != source_frame.delta_t_s
            or len(changed_sensors) != len(source_sensors)
        ):
            raise ValueError("action observation override changed frame geometry or time")
        for source, changed in zip(source_sensors, changed_sensors, strict=True):
            if (
                source.key != changed.key
                or source.timestamp_s != changed.timestamp_s
                or source.units != changed.units
                or source.value.shape != changed.value.shape
                or source.value.dtype != changed.value.dtype
            ):
                raise ValueError("action observation override changed a sensor contract")

        def observation_builder(_prefixes: Any, host_views: Any) -> Mapping[str, Any]:
            return module.build_host_observation_inputs(
                ((evidence_frame_override,),),
                host_views,
            )

    transition = assemble_calvin_stateful_molmoact2_transition(
        (sample,),
        native_banks=(),
        build_host_batch=module.build_host_batch,
        build_host_observation_inputs=observation_builder,
        tensor_device=initial_belief.address_mean.device,
        tensor_dtype=initial_belief.address_mean.dtype,
    )
    if transition.host_batch is None:
        raise RuntimeError("action intervention lost the official target batch")
    actions = transition.host_batch.get("action")
    if actions is None:
        raise ValueError("action intervention requires the official action target")
    flow_timesteps, flow_noise = materialize_molmoact2_flow_randomness(
        bridge.policy,
        (planned_transition.sample,),
        actions,
        transition_index=planned_transition.transition_index,
    )
    transition = replace(
        transition,
        flow_timesteps=flow_timesteps,
        flow_noise=flow_noise,
    )
    (
        native_banks,
        direct_context_banks,
        policy_batch,
        prepared_timesteps,
        prepared_noise,
        vision_patch_layout,
        action_condition_input_ids,
    ) = bridge._prepare_transition(transition, require_host_batch=True)
    if policy_batch is None or prepared_timesteps is None or prepared_noise is None:
        raise RuntimeError("prepared action intervention lost policy inputs or flow randomness")
    output = bridge.core(
        native_banks,
        initial_belief,
        transition.previous_executed_action,
        transition.delta_t_s,
    )
    evidence = action_evidence_from_core(
        output,
        direct_context_banks=direct_context_banks,
        include_posterior=True,
    )
    return _PreparedActionStep(
        optimizer_plan_step=optimizer_plan_step,
        planned_transition=planned_transition,
        sample=sample,
        evidence=evidence,
        final_belief=output.posterior.belief,
        policy_batch=policy_batch,
        flow_timesteps=prepared_timesteps,
        flow_noise=prepared_noise,
        action_condition_input_ids=action_condition_input_ids,
        vision_patch_layout=vision_patch_layout,
        core_output=output,
        temporal_config=bridge.core.posterior_filter.config,
    )


def _clone_object_belief(belief: Any) -> Any:
    """Clone a posterior so counterfactual forwards share exactly one prior."""

    return replace(
        belief,
        address_mean=belief.address_mean.detach().clone(),
        content_mean=belief.content_mean.detach().clone(),
        geometry_mean=belief.geometry_mean.detach().clone(),
        geometry_covariance_diag=belief.geometry_covariance_diag.detach().clone(),
        existence_logits=belief.existence_logits.detach().clone(),
        visibility_given_existence_logits=(
            belief.visibility_given_existence_logits.detach().clone()
        ),
        measurement_age_s=belief.measurement_age_s.detach().clone(),
        valid=belief.valid.detach().clone(),
        age=belief.age.detach().clone(),
    )


def _validate_controlled_observation_pair(
    clean: _PreparedActionStep,
    occluded: _PreparedActionStep,
    *,
    image_patch_token_id: int,
) -> dict[str, Any]:
    """Prove that the paired branches differ only at visual embedding tokens."""

    import torch

    if clean.sample.sample_key != occluded.sample.sample_key:
        raise ValueError("controlled observation pair must use one source sample")
    if clean.planned_transition is not occluded.planned_transition:
        raise ValueError("controlled observation pair must use one planned transition")
    if clean.policy_batch.keys() != occluded.policy_batch.keys():
        raise ValueError("controlled observation pair changed official policy fields")
    changed_fields = []
    for key in clean.policy_batch:
        first = clean.policy_batch[key]
        second = occluded.policy_batch[key]
        if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
            raise TypeError("controlled policy fields must remain tensors")
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError("controlled observation pair changed policy tensor geometry")
        if not torch.equal(first, second):
            changed_fields.append(key)
    if tuple(changed_fields) != ("inputs_embeds",):
        raise ValueError(
            "controlled observation pair must change exactly the inputs_embeds field; "
            f"changed={changed_fields}"
        )
    for name in ("flow_timesteps", "flow_noise", "action_condition_input_ids"):
        first = getattr(clean, name)
        second = getattr(occluded, name)
        if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
            raise TypeError(f"controlled observation pair lost tensor field {name}")
        if not torch.equal(first, second):
            raise ValueError(f"controlled observation pair changed {name}")
    if clean.vision_patch_layout != occluded.vision_patch_layout:
        raise ValueError("controlled observation pair changed the vision patch layout")
    if not isinstance(image_patch_token_id, int) or isinstance(image_patch_token_id, bool):
        raise TypeError("image patch token id must be an integer")
    clean_embeddings = clean.policy_batch["inputs_embeds"]
    occluded_embeddings = occluded.policy_batch["inputs_embeds"]
    input_ids = clean.action_condition_input_ids
    if clean_embeddings.ndim != 3 or input_ids.shape != clean_embeddings.shape[:2]:
        raise ValueError("controlled observation embeddings do not align with token identities")
    changed_tokens = torch.ne(clean_embeddings, occluded_embeddings).any(dim=-1)
    visual_tokens = input_ids == image_patch_token_id
    changed_nonvisual_tokens = changed_tokens & ~visual_tokens
    if changed_nonvisual_tokens.any():
        raise ValueError("controlled observation changed inputs_embeds outside image-patch tokens")
    changed_visual_token_count = int((changed_tokens & visual_tokens).sum().item())
    if changed_visual_token_count <= 0:
        raise ValueError("controlled observation did not change an image-patch embedding")
    return {
        "changed_policy_fields": changed_fields,
        "changed_nonvisual_token_count": int(changed_nonvisual_tokens.sum().item()),
        "changed_visual_token_count": changed_visual_token_count,
        "clean_visual_carrier_sha256": _tensor_sha256(clean_embeddings),
        "flow_noise_sha256": _tensor_sha256(clean.flow_noise),
        "flow_timesteps_sha256": _tensor_sha256(clean.flow_timesteps),
        "image_patch_token_id": image_patch_token_id,
        "nonvisual_policy_fields_exact": True,
        "occluded_visual_carrier_sha256": _tensor_sha256(occluded_embeddings),
        "same_action_target": True,
        "same_initial_posterior": True,
        "visual_carrier_field": "inputs_embeds",
        "visual_token_count": int(visual_tokens.sum().item()),
    }


def _advance_audit_identity_track(
    current: _PreparedActionStep,
    *,
    previous_keys_by_row: tuple[tuple[str | None, ...], ...],
    target_builder: Any,
    set_criterion: Any,
    reference_delta_t_s: float,
) -> _AuditIdentityAttribution:
    """Attribute rows after forward using the canonical loss-only target path."""

    import torch

    from picf_next.eval.calvin_task_relevance import select_hidden_task_rows
    from picf_next.hosts.molmoact2_training import (
        CalvinStatefulLossTargetLayout,
        calvin_visible_object_target_request,
    )
    from picf_next.models.objective import _advance_one_loss_track_frame

    output = current.core_output
    if output is None:
        raise ValueError("audit identity attribution requires the deploy-visible core output")
    projection = output.projection
    layout = CalvinStatefulLossTargetLayout(
        token_valid=projection.current_measurement_valid.detach().clone(),
        spans=projection.spans,
        target_dtype=torch.float32,
        rollout_input_dtype=output.posterior.belief.address_mean.dtype,
        vision_patch_layout=current.vision_patch_layout,
    )
    request = calvin_visible_object_target_request(
        current.sample,
        augmentation_seed=current.planned_transition.sample.augmentation_seed,
    )
    targets = target_builder((request,), layout)
    if targets.set_targets is None or len(targets.set_targets) != 1:
        raise RuntimeError("audit identity attribution lost its visible-object target")
    target = targets.set_targets[0]
    match = set_criterion(output.discovery, (target,)).matches[0]
    identity_keys = tuple(target.temporal_identity_keys or ())
    if len(identity_keys) != target.num_objects:
        raise ValueError("audit identity attribution requires complete temporal identity keys")
    valid = target.supervision_valid
    current_set_match_by_identity: dict[str, dict[str, Any]] = {}
    for prediction_index, target_index in zip(
        match.prediction_indices.tolist(),
        match.target_indices.tolist(),
        strict=True,
    ):
        predicted = output.discovery.ownership[0, valid, prediction_index].float()
        expected = target.ownership[valid, target_index].float()
        intersection = (predicted * expected).sum()
        union = predicted.sum() + expected.sum() - intersection
        current_set_match_by_identity[identity_keys[target_index]] = {
            "discovery_existence_probability": float(
                output.discovery.existence[0, prediction_index].detach().float().cpu()
            ),
            "discovery_localization_confidence": float(
                output.discovery.localization_confidence[0, prediction_index].detach().float().cpu()
            ),
            "discovery_query": int(prediction_index),
            "matched_soft_iou": float(
                (intersection / union.clamp_min(1e-6)).detach().float().cpu()
            ),
            "predicted_ownership_mass_on_supervised_tokens": float(predicted.sum().detach().cpu()),
            "target_ownership_mass": float(expected.sum().detach().cpu()),
            "target_positive_support_token_count": int((expected > 0.0).sum().cpu()),
        }
    next_keys, conflicts = _advance_one_loss_track_frame(
        output,
        (target,),
        (match,),
        previous_keys_by_row,
    )
    valid = output.posterior.belief.valid[0].detach().cpu().tolist()
    ages = output.posterior.belief.measurement_age_s[0].detach().float().cpu().tolist()
    selection = select_hidden_task_rows(
        task_key=current.sample.host_sample.task_key,
        identity_keys_by_row=next_keys[0],
        row_valid=tuple(bool(value) for value in valid),
        measurement_age_s=tuple(float(value) for value in ages),
        currently_measurable_identity_keys=tuple(target.temporal_identity_keys or ()),
        reference_delta_t_s=reference_delta_t_s,
    )
    return _AuditIdentityAttribution(
        next_keys_by_row=next_keys,
        currently_measurable_identity_keys=tuple(target.temporal_identity_keys or ()),
        current_set_match_by_identity=current_set_match_by_identity,
        track_conflicts=conflicts,
        task_selection=selection,
    )


def _source_frame_flow_seed_contract(
    *,
    global_source_step: int,
    repeat_index: int,
) -> tuple[int, int, int]:
    """Return flow seeds shared by every view of one physical source frame."""

    from picf_next.training.control import derive_subseed

    if (
        not isinstance(global_source_step, int)
        or isinstance(global_source_step, bool)
        or global_source_step < 0
    ):
        raise ValueError("global source step must be a non-negative integer")
    if not isinstance(repeat_index, int) or isinstance(repeat_index, bool) or repeat_index < 0:
        raise ValueError("flow repeat index must be a non-negative integer")
    root = derive_subseed(
        0,
        "m4-action-causal-audit-source-frame.v1",
        str(global_source_step),
        str(repeat_index),
    )
    return (
        derive_subseed(root, "flow-noise"),
        derive_subseed(root, "flow-timestep"),
        global_source_step,
    )


def _task_protocol_inventory_for_sample(
    sidecar: Any,
    sample: Any,
) -> tuple[str, ...]:
    """Read the protocol inventory through the sample's declared coverage."""

    from picf_next.eval.calvin_task_relevance import (
        validate_calvin_task_protocol_inventory,
    )

    frame = sidecar(sample.record.task_index, sample.record.global_index)
    return validate_calvin_task_protocol_inventory(frame.identity_keys)


def _tensor_sha256(tensor: Any) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("flow randomness hash requires a tensor")
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _materialize_flow_repeats(
    current: _PreparedActionStep,
    stack: Any,
    *,
    repeat_count: int,
    seed_mode: str,
) -> tuple[tuple[tuple[Any, Any], ...], tuple[dict[str, Any], ...]]:
    from picf_next.hosts.molmoact2_training import materialize_molmoact2_flow_randomness
    from picf_next.training.control import derive_subseed

    if not isinstance(repeat_count, int) or isinstance(repeat_count, bool) or repeat_count <= 0:
        raise ValueError("flow randomness repeat count must be a positive integer")
    if seed_mode not in _FLOW_SEED_MODES:
        raise ValueError(f"unsupported flow randomness seed mode: {seed_mode!r}")
    planned_sample = current.planned_transition.sample
    transition_index = current.planned_transition.transition_index
    actions = current.policy_batch.get("action")
    if actions is None:
        raise ValueError("flow randomness repeats require the official action target")
    policy = stack.module.joint_bridge.sequence_bridge.policy
    values = []
    contracts = []
    for repeat_index in range(repeat_count):
        if seed_mode == "training-plan" and repeat_index == 0:
            timesteps = current.flow_timesteps
            noise = current.flow_noise
            coordinate = transition_index
        else:
            if seed_mode == "source-frame-matched":
                noise_seed, timestep_seed, coordinate = _source_frame_flow_seed_contract(
                    global_source_step=current.sample.record.global_index,
                    repeat_index=repeat_index,
                )
            else:
                noise_seed = derive_subseed(
                    planned_sample.flow_noise_seed,
                    "m4-action-causal-audit-repeat.v1",
                    str(repeat_index),
                )
                timestep_seed = derive_subseed(
                    planned_sample.flow_timestep_seed,
                    "m4-action-causal-audit-repeat.v1",
                    str(repeat_index),
                )
                coordinate = transition_index
            audit_sample = replace(
                planned_sample,
                flow_noise_seed=noise_seed,
                flow_timestep_seed=timestep_seed,
            )
            timesteps, noise = materialize_molmoact2_flow_randomness(
                policy,
                (audit_sample,),
                actions,
                transition_index=coordinate,
            )
        if timesteps is None or noise is None:
            raise RuntimeError("flow randomness repeat lost timestep or noise tensors")
        values.append((timesteps, noise))
        contracts.append(
            {
                "noise_sha256": _tensor_sha256(noise),
                "repeat_index_zero_based": repeat_index,
                "timestep_sha256": _tensor_sha256(timesteps),
                "transition_coordinate": coordinate,
            }
        )
    return tuple(values), tuple(contracts)


def _rows_at_least_one_reference_step_old(
    current: _PreparedActionStep,
    *,
    reference_delta_t_s: float,
) -> Any:
    """Select deploy-visible posterior rows whose expected age spans one frame."""

    import torch

    valid = current.evidence.object_valid
    age = current.final_belief.measurement_age_s
    if valid is None or valid.dtype != torch.bool or valid.shape != age.shape:
        raise ValueError("age-row intervention requires aligned posterior validity")
    if (
        isinstance(reference_delta_t_s, bool)
        or not math.isfinite(reference_delta_t_s)
        or reference_delta_t_s <= 0.0
    ):
        raise ValueError("age-row intervention requires finite positive reference delta time")
    return valid & (age >= reference_delta_t_s)


def _evaluate_interventions(
    current: _PreparedActionStep,
    stale: Any,
    stack: Any,
    *,
    flow_randomness: Sequence[tuple[Any, Any]],
    reference_delta_t_s: float,
    task_relevant_hidden_rows: Any | None = None,
    controlled_task_target_rows: Any | None = None,
    clean_observation_reference: _PreparedActionStep | None = None,
) -> dict[str, Any]:
    import torch

    from picf_next.hosts.interventions import (
        permute_object_addresses,
        permute_object_rows,
        stale_posterior,
        without_object_rows,
        without_posterior,
    )

    evidence = current.evidence
    if evidence.object_valid is None or evidence.object_log_prior is None:
        raise ValueError("Arm C intervention requires a complete posterior action bank")
    permutation, removed, swapped_rows = _intervention_indices(
        evidence.object_valid,
        evidence.object_log_prior,
    )
    old_rows = _rows_at_least_one_reference_step_old(
        current,
        reference_delta_t_s=reference_delta_t_s,
    )
    if task_relevant_hidden_rows is not None:
        if (
            not isinstance(task_relevant_hidden_rows, torch.Tensor)
            or task_relevant_hidden_rows.dtype != torch.bool
            or task_relevant_hidden_rows.shape != old_rows.shape
            or task_relevant_hidden_rows.device != old_rows.device
        ):
            raise ValueError("task-relevant hidden-row mask must align with posterior rows")
        if bool((task_relevant_hidden_rows & ~old_rows).any()):
            raise ValueError("task-relevant hidden rows must also satisfy the old-row contract")
    target_address_permutation = None
    target_address_control_rows: tuple[int, ...] = ()
    if controlled_task_target_rows is not None:
        target_address_permutation, target_address_control_rows = _target_address_permutation(
            evidence.object_valid,
            evidence.object_log_prior,
            controlled_task_target_rows,
        )
        # The caller already requires a MAP MISS from the marginal lifecycle.
        # Measurement age is the null-probability-weighted predicted age, so a
        # legitimate soft MISS is generally below one full reference interval.
        # Requiring a full interval here would silently demand zero non-MAP
        # association mass and contradict the probabilistic gate.
        if clean_observation_reference is None:
            raise ValueError("controlled target intervention requires its clean observation pair")
    elif clean_observation_reference is not None:
        raise ValueError("clean observation reference is only valid for controlled target rows")
    conditions = {
        "baseline": evidence,
        "without_posterior": without_posterior(evidence),
        "joint_row_permutation": permute_object_rows(
            evidence,
            permutation,
            keep_ownership_fixed=False,
        ),
        "wrong_address": permute_object_rows(
            evidence,
            permutation,
            keep_ownership_fixed=True,
        ),
        "remove_max_prior_row": without_object_rows(evidence, removed),
        "remove_rows_at_least_one_reference_step": without_object_rows(evidence, old_rows),
        "stale_previous_frame": stale_posterior(evidence, stale),
    }
    if task_relevant_hidden_rows is not None:
        conditions["remove_task_relevant_hidden_rows"] = without_object_rows(
            evidence,
            task_relevant_hidden_rows,
        )
    if controlled_task_target_rows is not None:
        if target_address_permutation is None:  # pragma: no cover - guarded above
            raise RuntimeError("controlled target address permutation was not materialized")
        conditions["remove_controlled_task_target_rows"] = without_object_rows(
            evidence,
            controlled_task_target_rows,
        )
        conditions["wrong_controlled_task_target_address"] = permute_object_addresses(
            evidence,
            target_address_permutation,
        )
    policy = stack.module.joint_bridge.sequence_bridge.policy
    adapter = stack.module.joint_bridge.sequence_bridge.action_adapter
    if not flow_randomness:
        raise ValueError("action intervention requires at least one flow-randomness repeat")
    measurements: dict[str, list[dict[str, float]]] = {name: [] for name in conditions}
    if clean_observation_reference is not None:
        measurements["clean_current_observation"] = []
    replay_loss_deltas = []
    replay_velocity_rms_values = []
    joint_loss_deltas = []
    joint_velocity_rms_values = []
    for flow_timesteps, flow_noise in flow_randomness:
        baseline_loss, baseline_velocity = _fixed_flow_probe(
            policy=policy,
            action_adapter=adapter,
            policy_batch=current.policy_batch,
            action_condition_input_ids=current.action_condition_input_ids,
            evidence=conditions["baseline"],
            flow_timesteps=flow_timesteps,
            flow_noise=flow_noise,
        )
        measurements["baseline"].append(
            {
                "action_loss": baseline_loss,
                "loss_delta_from_baseline": 0.0,
                "velocity_rms_from_baseline": 0.0,
            }
        )
        if clean_observation_reference is not None:
            clean_loss, clean_velocity = _fixed_flow_probe(
                policy=policy,
                action_adapter=adapter,
                policy_batch=clean_observation_reference.policy_batch,
                action_condition_input_ids=(clean_observation_reference.action_condition_input_ids),
                evidence=clean_observation_reference.evidence,
                flow_timesteps=flow_timesteps,
                flow_noise=flow_noise,
            )
            measurements["clean_current_observation"].append(
                {
                    "action_loss": clean_loss,
                    "loss_delta_from_baseline": clean_loss - baseline_loss,
                    "velocity_rms_from_baseline": _valid_velocity_rms(
                        baseline_velocity,
                        clean_velocity,
                        current.policy_batch,
                    ),
                }
            )
            del clean_velocity
        for name, changed in conditions.items():
            if name == "baseline":
                continue
            loss, velocity = _fixed_flow_probe(
                policy=policy,
                action_adapter=adapter,
                policy_batch=current.policy_batch,
                action_condition_input_ids=current.action_condition_input_ids,
                evidence=changed,
                flow_timesteps=flow_timesteps,
                flow_noise=flow_noise,
            )
            measurements[name].append(
                {
                    "action_loss": loss,
                    "loss_delta_from_baseline": loss - baseline_loss,
                    "velocity_rms_from_baseline": _valid_velocity_rms(
                        baseline_velocity,
                        velocity,
                        current.policy_batch,
                    ),
                }
            )
            del velocity
        replay_loss, replay_velocity = _fixed_flow_probe(
            policy=policy,
            action_adapter=adapter,
            policy_batch=current.policy_batch,
            action_condition_input_ids=current.action_condition_input_ids,
            evidence=evidence,
            flow_timesteps=flow_timesteps,
            flow_noise=flow_noise,
        )
        replay_loss_deltas.append(abs(replay_loss - baseline_loss))
        replay_velocity_rms_values.append(
            _valid_velocity_rms(baseline_velocity, replay_velocity, current.policy_batch)
        )
        joint = measurements["joint_row_permutation"][-1]
        joint_loss_deltas.append(abs(joint["loss_delta_from_baseline"]))
        joint_velocity_rms_values.append(joint["velocity_rms_from_baseline"])
        del baseline_velocity, replay_velocity

    results = {}
    for name, records in measurements.items():
        deltas = [record["loss_delta_from_baseline"] for record in records]
        results[name] = {
            field: sum(record[field] for record in records) / len(records)
            for field in (
                "action_loss",
                "loss_delta_from_baseline",
                "velocity_rms_from_baseline",
            )
        }
        results[name].update(
            {
                "flow_repeat_count": len(records),
                "loss_delta_by_flow_repeat": deltas,
                "loss_delta_standard_error_over_flow_repeats": (
                    statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else None
                ),
                "positive_loss_delta_repeats": sum(delta > 0.0 for delta in deltas),
            }
        )
    torch.cuda.empty_cache()
    return {
        "conditions": results,
        "integrity": {
            "baseline_replay_action_loss_abs_delta": max(replay_loss_deltas),
            "baseline_replay_velocity_rms": max(replay_velocity_rms_values),
            "baseline_replay_exact": all(value == 0.0 for value in replay_loss_deltas)
            and all(value == 0.0 for value in replay_velocity_rms_values),
            "flow_repeat_count": len(flow_randomness),
            "joint_permutation_action_loss_abs_delta": max(joint_loss_deltas),
            "joint_permutation_velocity_rms": max(joint_velocity_rms_values),
        },
        "removed_rows_at_least_one_reference_step": [
            int(row)
            for row in torch.nonzero(old_rows[0], as_tuple=False).flatten().detach().cpu().tolist()
        ],
        "removed_task_relevant_hidden_rows": (
            None
            if task_relevant_hidden_rows is None
            else [
                int(row)
                for row in torch.nonzero(task_relevant_hidden_rows[0], as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            ]
        ),
        "controlled_task_target_address_control_rows": list(target_address_control_rows),
        "controlled_task_target_rows": (
            None
            if controlled_task_target_rows is None
            else [
                int(row)
                for row in torch.nonzero(controlled_task_target_rows[0], as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            ]
        ),
        "removed_row": swapped_rows[0],
        "swapped_rows": list(swapped_rows),
    }


def _measurement_summary(
    step: _PreparedActionStep,
    *,
    reference_delta_t_s: float,
) -> dict[str, Any]:
    """Describe deploy-visible posterior freshness without reading loss targets."""

    import torch

    belief = step.final_belief
    valid = step.evidence.object_valid
    if valid is None or valid.dtype != torch.bool or valid.shape != belief.valid.shape:
        raise ValueError("measurement summary requires aligned posterior validity")
    if bool((valid & ~belief.valid).any()):
        raise ValueError("action evidence selected a row absent from the posterior belief")
    selected_age = belief.measurement_age_s[valid].detach().float()
    selected_visibility = belief.visibility[valid].detach().float()
    selected_existence = belief.existence[valid].detach().float()
    if selected_age.numel() == 0:
        raise ValueError("measurement summary requires at least one valid posterior row")
    if (
        isinstance(reference_delta_t_s, bool)
        or not math.isfinite(reference_delta_t_s)
        or reference_delta_t_s <= 0.0
    ):
        raise ValueError("reference delta time must be finite and positive")
    normalized_age = selected_age / reference_delta_t_s
    positive_expected_age = selected_age > 0.0
    return {
        "has_row_with_expected_age_at_least_one_reference_step": bool(
            (normalized_age >= 1.0).any().item()
        ),
        "maximum_measurement_age_s": float(selected_age.max().cpu().item()),
        "maximum_measurement_age_reference_steps": float(normalized_age.max().cpu().item()),
        "mean_existence_probability": float(selected_existence.mean().cpu().item()),
        "mean_measurement_age_s": float(selected_age.mean().cpu().item()),
        "mean_measurement_age_reference_steps": float(normalized_age.mean().cpu().item()),
        "mean_visibility_probability": float(selected_visibility.mean().cpu().item()),
        "minimum_visibility_probability": float(selected_visibility.min().cpu().item()),
        "positive_expected_age_rows": int(positive_expected_age.sum().cpu().item()),
        "reference_delta_t_s": reference_delta_t_s,
        "valid_rows": int(selected_age.numel()),
    }


def _association_query_ownership_summary(
    step: _PreparedActionStep,
    *,
    query: int,
) -> dict[str, Any]:
    """Summarize one discovery query without consulting supervision targets."""

    core = step.core_output
    discovery = core.discovery
    ownership = discovery.ownership
    ownership_logits = discovery.ownership_logits
    if ownership.ndim != 3 or ownership.shape[0] != 1:
        raise ValueError("association diagnostics require one discovery batch row")
    if not 0 <= query < ownership.shape[-1] - 1:
        raise ValueError("association diagnostic query lies outside discovery capacity")
    if ownership_logits.shape != ownership.shape:
        raise ValueError("association diagnostic ownership logits are misaligned")

    relative_logits = ownership_logits[0, :, query].float() - ownership_logits[0, :, -1].float()
    valid = discovery.token_valid[0]
    modality_mass: dict[str, float] = {}
    for span in core.projection.spans:
        modality_mass[span.modality] = float(
            ownership[0, span.start : span.stop, query].detach().float().sum().cpu()
        )

    camera_mass: dict[str, dict[str, float | int]] = {}
    layout = step.vision_patch_layout
    vision_spans = [span for span in core.projection.spans if span.modality == "molmo_vision_patch"]
    if layout is not None and len(layout.rows) == 1 and len(vision_spans) == 1:
        vision_start = vision_spans[0].start
        for image_span in layout.rows[0]:
            start = vision_start + image_span.start
            stop = start + image_span.patches_per_crop
            if stop > vision_spans[0].stop:
                raise ValueError("association diagnostic camera crop exceeds vision span")
            camera_mass[str(image_span.image_key)] = {
                "ownership_mass": float(
                    ownership[0, start:stop, query].detach().float().sum().cpu()
                ),
                "positive_odds_patch_count": int(
                    ((relative_logits[start:stop] > 0.0) & valid[start:stop]).sum().cpu()
                ),
            }
    return {
        "camera_ownership": camera_mass,
        "context_competition_positive_odds_token_count": int(
            ((relative_logits > 0.0) & valid).sum().cpu()
        ),
        "modality_ownership_mass": modality_mass,
        "total_ownership_mass": float(ownership[0, :, query].detach().float().sum().cpu()),
    }


def _association_target_row_diagnostics(
    step: _PreparedActionStep,
    *,
    posterior_row: int,
) -> dict[str, Any] | None:
    """Decompose the exact runtime edge odds behind one posterior-row decision.

    This is evaluator-only telemetry over quantities already used by the
    probabilistic filter. It neither changes association nor introduces a new
    score or threshold.
    """

    import torch
    from torch.nn import functional as F

    from picf_next.models.binding_loss import spherical_relation_logits
    from picf_next.models.filter import BERNOULLI_PRUNING_PROBABILITY

    core = step.core_output
    config = step.temporal_config
    if core is None or config is None:
        return None
    posterior = core.posterior
    predicted = posterior.prior_prediction.belief
    discovery = core.discovery
    if predicted.valid.ndim != 2 or predicted.valid.shape[0] != 1:
        raise ValueError("association diagnostics require one predicted belief row")
    if not 0 <= posterior_row < predicted.valid.shape[1]:
        raise ValueError("association diagnostic posterior row lies outside capacity")
    if not bool(predicted.valid[0, posterior_row].item()):
        raise ValueError("association diagnostic posterior row was absent from the prior")

    prior_address = F.normalize(predicted.address_mean.float(), dim=-1)
    observation_address = F.normalize(discovery.address_mean.float(), dim=-1)
    address_cosine = torch.einsum(
        "bkd,bqd->bkq",
        prior_address,
        observation_address,
    ).clamp(min=-1.0, max=1.0)
    address_llr = spherical_relation_logits(
        address_cosine,
        logit_scale=posterior.address_relation_logit_scale,
        logit_bias=posterior.address_relation_logit_bias,
    )
    geometry_residual = discovery.geometry_mean.float().unsqueeze(
        1
    ) - predicted.geometry_mean.float().unsqueeze(2)
    innovation_variance = (
        2.0
        * (
            predicted.geometry_covariance_diag.float().unsqueeze(2)
            + discovery.geometry_variance.float().unsqueeze(1)
        )
    ).clamp_min(config.minimum_variance)
    geometry_nll = 0.5 * (
        geometry_residual.square() / innovation_variance
        + innovation_variance.log()
        + math.log(2.0 * math.pi)
    ).sum(dim=-1)

    probability_floor = torch.finfo(torch.float32).eps
    prior_existence = predicted.existence.float()
    prior_detection = torch.sigmoid(predicted.visibility_given_existence_logits.float())
    observation_probability = discovery.measurement_probability.float()
    bank_is_empty = ~predicted.valid.any(dim=1, keepdim=True)
    empty_birth_odds = observation_probability.new_full(
        observation_probability.shape,
        config.empty_bank_birth_to_clutter_prior_odds,
    )
    recurrent_birth_odds = observation_probability.new_full(
        observation_probability.shape,
        config.recurrent_birth_to_clutter_prior_odds,
    )
    birth_odds = torch.where(bank_is_empty, empty_birth_odds, recurrent_birth_odds)
    row_null_mass = (1.0 - prior_existence * prior_detection).clamp_min(probability_floor)
    observation_null_mass = (
        1.0 - observation_probability + birth_odds * observation_probability
    ).clamp_min(probability_floor)
    log_edge_odds = (
        prior_existence.clamp_min(probability_floor).log().unsqueeze(2)
        + prior_detection.clamp_min(probability_floor).log().unsqueeze(2)
        + observation_probability.clamp_min(probability_floor).log().unsqueeze(1)
        + address_llr
        - geometry_nll
        - row_null_mass.log().unsqueeze(2)
        - observation_null_mass.log().unsqueeze(1)
    )

    match_probability = posterior.match_probability.detach().float()
    null_probability = posterior.null_probability.detach().float()
    existence_given_no_detection = prior_existence * (1.0 - prior_detection) / row_null_mass
    detected_mass = match_probability.sum(dim=-1)
    missed_alive_mass = null_probability * existence_given_no_detection
    posterior_existence_before_capacity = (detected_mass + missed_alive_mass).clamp(
        min=0.0, max=1.0
    )
    retained_existing = predicted.valid & (
        posterior_existence_before_capacity > BERNOULLI_PRUNING_PROBABILITY
    )
    birth_probability = posterior.birth_probability.detach().float()
    existing_origin_probability = match_probability.sum(dim=1)
    retained_birth = (birth_probability > BERNOULLI_PRUNING_PROBABILITY) & (
        birth_probability > existing_origin_probability
    )
    candidate_score = torch.cat(
        (
            posterior_existence_before_capacity.masked_fill(~retained_existing, -torch.inf),
            birth_probability.masked_fill(~retained_birth, -torch.inf),
        ),
        dim=1,
    )
    capacity = predicted.valid.shape[1]
    selected_candidate_indices = (
        candidate_score[0]
        .topk(
            capacity,
            sorted=True,
        )
        .indices
    )
    target_candidate_score = candidate_score[0, posterior_row]
    target_selected = bool(
        retained_existing[0, posterior_row].item()
        and (selected_candidate_indices == posterior_row).any().item()
    )
    higher_score_candidates = int((candidate_score[0] > target_candidate_score).sum().cpu())
    finite_candidate_count = int(torch.isfinite(candidate_score[0]).sum().cpu())
    selected_candidates = []
    for index in selected_candidate_indices.cpu().tolist():
        score = float(candidate_score[0, index].cpu())
        if not math.isfinite(score):
            continue
        selected_candidates.append(
            {
                "candidate_index": int(index),
                "kind": "existing" if index < capacity else "birth",
                "local_index": int(index if index < capacity else index - capacity),
                "score": score,
            }
        )
    topk_boundary_score = float(candidate_score[0, selected_candidate_indices[-1]].cpu())

    row_match = posterior.match_probability[0, posterior_row].detach().float()
    query_count = row_match.numel()
    top_queries = torch.argsort(row_match, descending=True)[: min(3, query_count)].tolist()
    query_records = []
    for query in top_queries:
        residual = geometry_residual[0, posterior_row, query].detach().float().cpu()
        variance = innovation_variance[0, posterior_row, query].detach().float().cpu()
        mapped_row = int(posterior.observation_to_posterior[0, query].item())
        query_records.append(
            {
                "address_cosine": float(address_cosine[0, posterior_row, query].cpu()),
                "address_log_likelihood_ratio": float(
                    address_llr[0, posterior_row, query].detach().float().cpu()
                ),
                "discovery_existence_probability": float(
                    discovery.existence[0, query].detach().float().cpu()
                ),
                "discovery_localization_confidence": float(
                    discovery.localization_confidence[0, query].detach().float().cpu()
                ),
                "discovery_mask_quality": float(
                    discovery.mask_quality[0, query].detach().float().cpu()
                ),
                "discovery_measurement_probability": float(
                    observation_probability[0, query].detach().float().cpu()
                ),
                "geometry_innovation_variance": variance.tolist(),
                "geometry_negative_log_likelihood": float(
                    geometry_nll[0, posterior_row, query].detach().float().cpu()
                ),
                "geometry_residual": residual.tolist(),
                "log_edge_odds": float(
                    log_edge_odds[0, posterior_row, query].detach().float().cpu()
                ),
                "map_assigned_posterior_row": mapped_row,
                "match_probability": float(row_match[query].cpu()),
                "ownership": _association_query_ownership_summary(step, query=query),
                "query": int(query),
                "query_is_mutual_map_for_target_row": mapped_row == posterior_row,
            }
        )
    return {
        "address_relation_logit_bias": float(
            posterior.address_relation_logit_bias.detach().float().cpu()
        ),
        "address_relation_logit_scale": float(
            posterior.address_relation_logit_scale.detach().float().cpu()
        ),
        "map_query": int(top_queries[0]),
        "null_probability": float(
            posterior.null_probability[0, posterior_row].detach().float().cpu()
        ),
        "prior_detection_probability": float(
            prior_detection[0, posterior_row].detach().float().cpu()
        ),
        "prior_existence_probability": float(
            prior_existence[0, posterior_row].detach().float().cpu()
        ),
        "prior_row_null_mass": float(row_null_mass[0, posterior_row].detach().float().cpu()),
        "finite_capacity_projection": {
            "birth_candidate_count": int(retained_birth[0].sum().cpu()),
            "candidate_score": float(target_candidate_score.cpu()),
            "detected_mass": float(detected_mass[0, posterior_row].cpu()),
            "existence_given_no_detection": float(
                existence_given_no_detection[0, posterior_row].cpu()
            ),
            "existing_candidate_count": int(retained_existing[0].sum().cpu()),
            "finite_candidate_count": finite_candidate_count,
            "higher_score_candidate_count": higher_score_candidates,
            "missed_alive_mass": float(missed_alive_mass[0, posterior_row].cpu()),
            "posterior_existence_before_capacity": float(
                posterior_existence_before_capacity[0, posterior_row].cpu()
            ),
            "selected_as_existing": target_selected,
            "selected_candidates": selected_candidates,
            "selected_candidate_indices": [
                int(value) for value in selected_candidate_indices.cpu().tolist()
            ],
            "target_candidate_rank": (
                higher_score_candidates + 1
                if math.isfinite(float(target_candidate_score.cpu()))
                else None
            ),
            "topk_boundary_score": (
                topk_boundary_score if math.isfinite(topk_boundary_score) else None
            ),
        },
        "top_queries": query_records,
    }


def _controlled_target_row_summary(
    clean: _PreparedActionStep,
    occluded: _PreparedActionStep,
    target_rows: Sequence[int],
    *,
    reference_delta_t_s: float,
    clean_identity_attribution: _AuditIdentityAttribution | None = None,
    target_identity_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Separate retained Bernoulli support from MAP action extraction.

    A counterfactual image edit can correctly drive a witnessed row below the
    equal-cost MAP extraction boundary without deleting that row from the
    finite posterior.  Treating the action-bank mask as posterior validity
    would conflate those two decisions and report a false lifecycle death.
    """

    import torch

    from picf_next.posterior import (
        BIRTH_EVENT,
        DEATH_EVENT,
        MATCH_EVENT,
        MISS_EVENT,
        UNUSED_EVENT,
    )

    if not target_rows:
        raise ValueError("controlled target summary requires at least one row")
    if (clean_identity_attribution is None) != (target_identity_keys is None):
        raise ValueError("controlled target loss-side attribution is atomic")
    if target_identity_keys is not None and len(target_identity_keys) != len(target_rows):
        raise ValueError("controlled target identities must align with posterior rows")
    clean_action_valid = clean.evidence.object_valid
    occluded_action_valid = occluded.evidence.object_valid
    clean_belief_valid = clean.final_belief.valid
    occluded_belief_valid = occluded.final_belief.valid
    if (
        clean_action_valid is None
        or occluded_action_valid is None
        or clean_action_valid.dtype != torch.bool
        or occluded_action_valid.dtype != torch.bool
        or clean_belief_valid.dtype != torch.bool
        or occluded_belief_valid.dtype != torch.bool
        or clean_action_valid.shape != clean_belief_valid.shape
        or occluded_action_valid.shape != occluded_belief_valid.shape
        or clean_belief_valid.shape != occluded_belief_valid.shape
    ):
        raise ValueError("controlled target summary requires aligned posterior rows")
    if bool((clean_action_valid & ~clean_belief_valid).any()) or bool(
        (occluded_action_valid & ~occluded_belief_valid).any()
    ):
        raise ValueError("controlled target action extraction exceeds posterior support")
    posterior = occluded.core_output.posterior
    clean_posterior = (
        None if clean.core_output is None else getattr(clean.core_output, "posterior", None)
    )
    clean_event_type = (
        None if clean_posterior is None else getattr(clean_posterior, "event_type", None)
    )
    if clean_event_type is not None and clean_event_type.shape != clean_belief_valid.shape:
        raise ValueError("controlled factual event telemetry differs from posterior rows")
    event_type = posterior.event_type
    null_probability = posterior.null_probability
    match_probability = posterior.match_probability
    if (
        event_type.shape != clean_belief_valid.shape
        or null_probability.shape != clean_belief_valid.shape
        or match_probability.ndim != 3
        or match_probability.shape[:2] != clean_belief_valid.shape
    ):
        raise ValueError("controlled target summary requires aligned association telemetry")
    age_threshold_s = float(
        torch.as_tensor(
            reference_delta_t_s,
            dtype=occluded.final_belief.measurement_age_s.dtype,
        )
        .float()
        .item()
    )
    ownership_by_modality: dict[str, list[float]] = {}
    if occluded.evidence.dense_ownership is not None:
        for bank, ownership in zip(
            occluded.evidence.dense_banks,
            occluded.evidence.dense_ownership,
            strict=True,
        ):
            ownership_by_modality[bank.modality] = [
                float(ownership[0, :, row].detach().float().sum().cpu()) for row in target_rows
            ]
    records = []
    event_names = {
        UNUSED_EVENT: "unused",
        MATCH_EVENT: "match",
        MISS_EVENT: "miss",
        BIRTH_EVENT: "birth",
        DEATH_EVENT: "death",
    }
    for index, row in enumerate(target_rows):
        if not 0 <= row < clean_belief_valid.shape[1]:
            raise ValueError("controlled target row lies outside posterior capacity")
        clean_is_retained = bool(clean_belief_valid[0, row].item())
        occluded_is_retained = bool(occluded_belief_valid[0, row].item())
        clean_is_action_exposed = bool(clean_action_valid[0, row].item())
        occluded_is_action_exposed = bool(occluded_action_valid[0, row].item())
        row_event_type = int(event_type[0, row].item())
        if row_event_type not in event_names:
            raise ValueError("controlled target row has an unknown lifecycle event")
        row_match_probability = match_probability[0, row].detach().float()
        clean_association = _association_target_row_diagnostics(
            clean,
            posterior_row=row,
        )
        occluded_association = _association_target_row_diagnostics(
            occluded,
            posterior_row=row,
        )
        matched_query_comparison = None
        if clean_association is not None and occluded_association is not None:
            clean_query = clean_association["map_query"]
            occluded_query = occluded_association["map_query"]
            clean_discovery = clean.core_output.discovery
            occluded_discovery = occluded.core_output.discovery
            matched_query_comparison = {
                "address_cosine_clean_to_occluded": float(
                    torch.dot(
                        clean_discovery.address_mean[0, clean_query].detach().float(),
                        occluded_discovery.address_mean[0, occluded_query].detach().float(),
                    ).cpu()
                ),
                "clean_query": clean_query,
                "occluded_query": occluded_query,
                "query_feature_cosine_clean_to_occluded": float(
                    torch.nn.functional.cosine_similarity(
                        clean_discovery.query_features[0, clean_query].detach().float(),
                        occluded_discovery.query_features[0, occluded_query].detach().float(),
                        dim=0,
                    ).cpu()
                ),
            }
        target_identity_key = (
            None if target_identity_keys is None else str(target_identity_keys[index])
        )
        clean_loss_side_set_match = (
            None
            if clean_identity_attribution is None or target_identity_key is None
            else clean_identity_attribution.current_set_match_by_identity.get(target_identity_key)
        )
        if clean_identity_attribution is not None and clean_loss_side_set_match is None:
            raise ValueError("controlled target lacks a clean loss-side set match")
        clean_runtime_query_equals_loss_side_query = (
            None
            if clean_association is None or clean_loss_side_set_match is None
            else clean_association["map_query"] == clean_loss_side_set_match["discovery_query"]
        )
        if matched_query_comparison is not None:
            matched_query_comparison["clean_runtime_query_equals_loss_side_query"] = (
                clean_runtime_query_equals_loss_side_query
            )
        clean_row_event_type = (
            None if clean_event_type is None else int(clean_event_type[0, row].item())
        )
        if clean_row_event_type is not None and clean_row_event_type not in event_names:
            raise ValueError("controlled factual target has an unknown lifecycle event")
        clean_target_recognized = (
            clean_row_event_type == MATCH_EVENT
            and clean_runtime_query_equals_loss_side_query is True
        )
        records.append(
            {
                "address_cosine_clean_to_occluded": (
                    float(
                        np.dot(
                            clean.final_belief.address_mean[0, row].detach().float().cpu().numpy(),
                            occluded.final_belief.address_mean[0, row]
                            .detach()
                            .float()
                            .cpu()
                            .numpy(),
                        )
                    )
                    if clean_is_retained and occluded_is_retained
                    else None
                ),
                "clean_action_exposed": clean_is_action_exposed,
                "clean_association": clean_association,
                "clean_posterior_retained": clean_is_retained,
                "clean_loss_side_set_match": clean_loss_side_set_match,
                "clean_runtime_event": (
                    None if clean_row_event_type is None else event_names[clean_row_event_type]
                ),
                "clean_runtime_event_code": clean_row_event_type,
                "clean_runtime_query_equals_loss_side_query": (
                    clean_runtime_query_equals_loss_side_query
                ),
                "clean_target_recognized": clean_target_recognized,
                "existence_probability": (
                    float(occluded.final_belief.existence[0, row].detach().float().cpu())
                    if occluded_is_retained
                    else 0.0
                ),
                "measurement_age_reference_steps": (
                    float(
                        occluded.final_belief.measurement_age_s[0, row].detach().float().cpu()
                        / reference_delta_t_s
                    )
                    if occluded_is_retained
                    else 0.0
                ),
                "measurement_age_s": (
                    float(occluded.final_belief.measurement_age_s[0, row].detach().float().cpu())
                    if occluded_is_retained
                    else 0.0
                ),
                "maximum_match_probability": float(row_match_probability.max().cpu()),
                "null_probability": float(null_probability[0, row].detach().float().cpu()),
                "occluded_action_exposed": occluded_is_action_exposed,
                "occluded_association": occluded_association,
                "occluded_posterior_retained": occluded_is_retained,
                "ownership_mass_by_modality": {
                    modality: values[index] for modality, values in ownership_by_modality.items()
                },
                "posterior_row": row,
                "matched_query_comparison": matched_query_comparison,
                "runtime_event": event_names[row_event_type],
                "runtime_event_code": row_event_type,
                "total_match_probability": float(row_match_probability.sum().cpu()),
                "target_identity_key": target_identity_key,
                "visibility_probability": (
                    float(occluded.final_belief.visibility[0, row].detach().float().cpu())
                    if occluded_is_retained
                    else 0.0
                ),
            }
        )
    return {
        "age_threshold_s_in_posterior_dtype": age_threshold_s,
        "all_target_rows_map_missed": all(
            record["runtime_event_code"] == MISS_EVENT for record in records
        ),
        "all_target_rows_action_exposed": all(
            record["occluded_action_exposed"] for record in records
        ),
        "all_target_rows_clean_recognized": all(
            record["clean_target_recognized"] for record in records
        ),
        "all_target_rows_remain_in_posterior": all(
            record["occluded_posterior_retained"] for record in records
        ),
        "all_target_rows_expected_age_at_least_one_reference_step": all(
            record["measurement_age_s"] >= age_threshold_s for record in records
        ),
        "rows": records,
    }


def _visual_color(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 212],
            [0, 128, 128],
            [220, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
        ],
        dtype=np.float32,
    )
    return palette[index % len(palette)]


def _posterior_overlay(
    source: np.ndarray,
    probability: np.ndarray,
    row_indices: Sequence[int],
) -> np.ndarray:
    """Tint one global processor crop by runtime posterior ownership only."""

    from PIL import Image

    patch_count = probability.shape[0]
    side = math.isqrt(patch_count)
    if side * side != patch_count or probability.shape[1] != len(row_indices) + 1:
        raise ValueError("action visual ownership is not one square patch grid plus context")
    if source.ndim != 3 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("action visual source must be uint8 RGB")
    object_probability = np.clip(probability[:, :-1].astype(np.float32), 0.0, 1.0)
    object_mass = np.clip(object_probability.sum(axis=-1), 0.0, 1.0)
    colors = np.stack([_visual_color(row) for row in row_indices])
    weighted_color = object_probability @ colors
    weighted_color = np.divide(
        weighted_color,
        object_mass[:, None],
        out=np.zeros_like(weighted_color),
        where=object_mass[:, None] > 0.0,
    )
    height, width = source.shape[:2]
    tint = np.asarray(
        Image.fromarray(
            weighted_color.reshape(side, side, 3).astype(np.uint8),
        ).resize((width, height), Image.Resampling.NEAREST),
        dtype=np.float32,
    )
    alpha = np.asarray(
        Image.fromarray(object_mass.reshape(side, side)).resize(
            (width, height), Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    )[..., None]
    result = (1.0 - 0.65 * alpha) * source.astype(np.float32) + 0.65 * alpha * tint
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _draw_posterior_centroids(
    image: Any,
    probability: np.ndarray,
    row_indices: Sequence[int],
) -> None:
    from PIL import ImageDraw

    side = math.isqrt(probability.shape[0])
    if side * side != probability.shape[0]:
        raise ValueError("action visual centroid ownership is not square")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    yy, xx = np.mgrid[0:side, 0:side]
    for column, row in enumerate(row_indices):
        mass = probability[:, column].reshape(side, side)
        total = float(mass.sum())
        if total <= 1e-5:
            continue
        x = float((mass * (xx + 0.5)).sum() / total) / side * width
        y = float((mass * (yy + 0.5)).sum() / total) / side * height
        color = tuple(int(value) for value in _visual_color(row))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white", width=2)


def _visual_panel(
    title: str,
    source: np.ndarray,
    probability: np.ndarray | None,
    row_indices: Sequence[int],
) -> Any:
    from PIL import Image, ImageDraw

    image = Image.fromarray(source).resize((320, 320), Image.Resampling.NEAREST)
    if probability is not None:
        _draw_posterior_centroids(image, probability, row_indices)
    canvas = Image.new("RGB", (320, 350), "white")
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((7, 8), title, fill="black")
    return canvas


def _safe_visual_component(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part)[:80] or "unknown-task"


def _runtime_visual_snapshot(
    step: _PreparedActionStep,
    *,
    source_images: Mapping[str, np.ndarray] | None = None,
) -> _ActionVisualSnapshot:
    """Detach the deploy-visible ownership and posterior needed for temporal review."""

    import torch

    from picf_next.hosts.molmoact2_layout import MOLMO_VISION_PATCH_MODALITY

    layout = step.vision_patch_layout
    evidence = step.evidence
    ownership = evidence.dense_ownership
    if layout is None or len(layout.rows) != 1 or ownership is None:
        raise ValueError("action visual requires one explicit vision layout and dense ownership")
    vision_indices = [
        index
        for index, bank in enumerate(evidence.dense_banks)
        if bank.modality == MOLMO_VISION_PATCH_MODALITY
    ]
    if len(vision_indices) != 1:
        raise ValueError("action visual requires exactly one Molmo vision bank")
    raw_ownership = ownership[vision_indices[0]]
    if raw_ownership.ndim != 3 or raw_ownership.shape[0] != 1:
        raise ValueError("action visual requires one ownership batch row")
    valid = evidence.object_valid
    log_prior = evidence.object_log_prior
    if (
        valid is None
        or valid.dtype != torch.bool
        or valid.ndim != 2
        or valid.shape[0] != 1
        or log_prior is None
        or log_prior.shape != valid.shape
    ):
        raise ValueError("action visual requires aligned posterior validity and log prior")
    capacity = valid.shape[1]
    if raw_ownership.shape[-1] != capacity + 1:
        raise ValueError("action visual ownership capacity differs from posterior capacity")

    raw = raw_ownership[0].detach().float().cpu().numpy()
    cameras = []
    for image_span in layout.rows[0]:
        source = np.asarray(
            step.sample.host_sample.observation.get(image_span.image_key)
            if source_images is None
            else source_images.get(image_span.image_key)
        )
        if source.ndim != 3 or source.shape[-1] != 3 or source.dtype != np.uint8:
            raise ValueError("action visual source must be uint8 RGB")
        stop = image_span.start + image_span.patches_per_crop
        if stop > image_span.stop or stop > raw.shape[0]:
            raise ValueError("action visual global crop exceeds its image span")
        cameras.append(
            _VisualCameraSnapshot(
                image_key=str(image_span.image_key),
                source=source.copy(),
                ownership=raw[image_span.start : stop].copy(),
            )
        )
    if not cameras:
        raise ValueError("action visual requires at least one camera crop")

    belief = step.final_belief
    planned = step.planned_transition
    episode_instance_id = getattr(planned, "episode_instance_id", None)
    transition_index = getattr(planned, "transition_index", None)
    if episode_instance_id is not None and not isinstance(episode_instance_id, str):
        raise ValueError("visual episode instance ID must be a string")
    if transition_index is not None and (
        not isinstance(transition_index, int) or isinstance(transition_index, bool)
    ):
        raise ValueError("visual transition index must be an integer")
    return _ActionVisualSnapshot(
        optimizer_plan_step=step.optimizer_plan_step,
        global_source_step=int(step.sample.record.global_index),
        episode_instance_id=episode_instance_id,
        transition_index=transition_index,
        task=str(step.sample.record.task),
        task_key=str(step.sample.host_sample.task_key),
        cameras=tuple(cameras),
        valid=valid[0].detach().cpu().numpy().copy(),
        address_mean=belief.address_mean[0].detach().float().cpu().numpy().copy(),
        content_mean=belief.content_mean[0].detach().float().cpu().numpy().copy(),
        geometry_mean=belief.geometry_mean[0].detach().float().cpu().numpy().copy(),
        existence=belief.existence[0].detach().float().cpu().numpy().copy(),
        visibility=belief.visibility[0].detach().float().cpu().numpy().copy(),
        measurement_age_s=belief.measurement_age_s[0].detach().float().cpu().numpy().copy(),
        log_prior=log_prior[0].detach().float().cpu().numpy().copy(),
    )


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return None
    return float(np.dot(first, second) / (first_norm * second_norm))


def _ownership_summary(probability: np.ndarray) -> dict[str, float | None]:
    if probability.ndim != 1:
        raise ValueError("row ownership summary requires one flat patch vector")
    side = math.isqrt(probability.size)
    if side * side != probability.size:
        raise ValueError("row ownership summary requires one square patch grid")
    mass = float(probability.sum())
    if mass <= 1e-8:
        return {"mass": mass, "centroid_x": None, "centroid_y": None}
    yy, xx = np.mgrid[0:side, 0:side]
    grid = probability.reshape(side, side)
    return {
        "mass": mass,
        "centroid_x": float((grid * (xx + 0.5)).sum() / mass / side),
        "centroid_y": float((grid * (yy + 0.5)).sum() / mass / side),
    }


def _old_row_trajectory_records(
    history: Sequence[_ActionVisualSnapshot],
    row: int,
    *,
    reference_delta_t_s: float,
) -> list[dict[str, Any]]:
    """Describe one persistent row without consulting labels or action targets."""

    if not history:
        raise ValueError("old-row trajectory requires non-empty history")
    if not math.isfinite(reference_delta_t_s) or reference_delta_t_s <= 0.0:
        raise ValueError("old-row trajectory reference delta time must be finite and positive")
    selected = history[-1]
    capacity = selected.valid.size
    if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < capacity:
        raise ValueError("old-row trajectory index lies outside posterior capacity")
    if not bool(selected.valid[row]):
        raise ValueError("selected old row is not valid in the current posterior")
    records = []
    previous_transition = None
    for snapshot in history:
        if snapshot.valid.size != capacity:
            raise ValueError("old-row trajectory posterior capacity changed")
        if (
            snapshot.episode_instance_id != selected.episode_instance_id
            or snapshot.task != selected.task
        ):
            raise ValueError("old-row trajectory crossed an episode or task boundary")
        if previous_transition is not None and snapshot.transition_index != previous_transition + 1:
            raise ValueError("old-row trajectory is not transition-contiguous")
        previous_transition = snapshot.transition_index
        valid = bool(snapshot.valid[row])
        camera_ownership = {
            camera.image_key: _ownership_summary(camera.ownership[:, row])
            for camera in snapshot.cameras
        }
        records.append(
            {
                "address_cosine_to_current": (
                    _cosine_similarity(snapshot.address_mean[row], selected.address_mean[row])
                    if valid
                    else None
                ),
                "camera_ownership": camera_ownership,
                "content_cosine_to_current": (
                    _cosine_similarity(snapshot.content_mean[row], selected.content_mean[row])
                    if valid
                    else None
                ),
                "existence_probability": float(snapshot.existence[row]) if valid else 0.0,
                "geometry_l2_to_current": (
                    float(np.linalg.norm(snapshot.geometry_mean[row] - selected.geometry_mean[row]))
                    if valid
                    else None
                ),
                "global_source_step": snapshot.global_source_step,
                "measurement_age_reference_steps": (
                    float(snapshot.measurement_age_s[row] / reference_delta_t_s) if valid else 0.0
                ),
                "optimizer_plan_step_zero_based": snapshot.optimizer_plan_step,
                "posterior_valid": valid,
                "transition_index": snapshot.transition_index,
                "visibility_probability": float(snapshot.visibility[row]) if valid else 0.0,
            }
        )
    return records


def _render_old_row_trajectories(
    *,
    path_stem: Path,
    history: Sequence[_ActionVisualSnapshot],
    rows: Sequence[int],
    rank: int,
    rank_sample_index: int,
    reference_delta_t_s: float,
    loss_delta_from_baseline: float,
) -> tuple[dict[str, Any], ...]:
    """Render runtime-only temporal evidence for each causally removed old row."""

    from PIL import Image, ImageDraw

    if not rows:
        raise ValueError("old-row temporal visual requires at least one row")
    artifacts = []
    for row in rows:
        records = _old_row_trajectory_records(
            history,
            row,
            reference_delta_t_s=reference_delta_t_s,
        )
        frame_height = 382
        header_height = 104
        canvas = Image.new("RGB", (640, header_height + frame_height * len(history)), "white")
        draw = ImageDraw.Draw(canvas)
        selected = history[-1]
        draw.text(
            (8, 8),
            (
                f"rank={rank} sample={rank_sample_index} old posterior row={row:02d} "
                f"selected frame={selected.global_source_step}"
            ),
            fill="black",
        )
        draw.text((8, 31), f"instruction={selected.task}"[:105], fill="black")
        draw.text(
            (8, 54),
            f"old-row removal action-loss delta={loss_delta_from_baseline:+.8f}",
            fill="black",
        )
        draw.text(
            (8, 77),
            "Runtime RGB/ownership only; no mask, bbox, simulator ID, or action target.",
            fill="black",
        )
        for frame_index, (snapshot, record) in enumerate(zip(history, records, strict=True)):
            y = header_height + frame_index * frame_height
            ownership_text = ", ".join(
                f"{key.split('.')[-1]} mass={value['mass']:.3f}"
                for key, value in record["camera_ownership"].items()
            )
            draw.text(
                (8, y + 4),
                (
                    f"frame={snapshot.global_source_step} transition={snapshot.transition_index} "
                    f"valid={record['posterior_valid']} "
                    f"exist={record['existence_probability']:.4f} "
                    f"visible={record['visibility_probability']:.4f} "
                    f"age={record['measurement_age_reference_steps']:.3f} | {ownership_text}"
                )[:112],
                fill="black",
            )
            for camera_index, camera in enumerate(snapshot.cameras[:2]):
                probability = np.concatenate(
                    (camera.ownership[:, row : row + 1], camera.ownership[:, -1:]),
                    axis=-1,
                )
                overlay = _posterior_overlay(camera.source, probability, (row,))
                panel = _visual_panel(camera.image_key, overlay, probability, (row,))
                canvas.paste(panel, (320 * camera_index, y + 32))
        path = path_stem.with_name(f"{path_stem.name}_oldrow{row:02d}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        artifacts.append(
            {
                "bytes": path.stat().st_size,
                "frames": len(history),
                "global_source_steps": [item.global_source_step for item in history],
                "kind": "runtime_old_posterior_row_trajectory",
                "path": path.name,
                "posterior_row": row,
                "rank": rank,
                "rank_sample_index_zero_based": rank_sample_index,
                "sha256": _sha256(path),
                "task_key": selected.task_key,
            }
        )
    return tuple(artifacts)


def _render_controlled_occlusion_visual(
    *,
    path: Path,
    clean: _PreparedActionStep,
    occluded: _PreparedActionStep,
    occlusion: Any,
    target_rows: Sequence[int],
    rank: int,
    rank_sample_index: int,
) -> dict[str, Any]:
    """Render the exact evaluator perturbation and both runtime anchor fields."""

    from PIL import Image, ImageDraw

    if not target_rows:
        raise ValueError("controlled visual requires at least one target posterior row")
    removed_sensor_by_key = {
        sensor.key: sensor.value for sensor in occlusion.evidence_frame.sensor_observations
    }
    occluded_sources = {
        camera.host_image_key: np.asarray(removed_sensor_by_key[camera.source_observation_key])
        for camera in occlusion.cameras
    }
    factual_evidence_frame = getattr(occlusion, "factual_evidence_frame", None)
    if factual_evidence_frame is None:
        factual_sources = None
    else:
        factual_sensor_by_key = {
            sensor.key: sensor.value for sensor in factual_evidence_frame.sensor_observations
        }
        factual_sources = {
            camera.host_image_key: np.asarray(factual_sensor_by_key[camera.source_observation_key])
            for camera in occlusion.cameras
        }
    clean_snapshot = _runtime_visual_snapshot(clean, source_images=factual_sources)
    occluded_snapshot = _runtime_visual_snapshot(
        occluded,
        source_images=occluded_sources,
    )
    if tuple(camera.image_key for camera in clean_snapshot.cameras) != tuple(
        camera.image_key for camera in occluded_snapshot.cameras
    ):
        raise ValueError("controlled visual camera order changed between branches")
    report_by_host = {camera.host_image_key: camera for camera in occlusion.cameras}
    header_height = 132
    canvas = Image.new(
        "RGB",
        (1280, header_height + 350 * len(clean_snapshot.cameras)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    method = str(occlusion.contract_dict().get("method", "unknown"))
    draw.text(
        (8, 8),
        (
            f"CONTROLLED ABSENCE rank={rank} sample={rank_sample_index} "
            f"frame={clean_snapshot.global_source_step} rows={list(target_rows)}"
        ),
        fill="black",
    )
    draw.text((8, 31), f"task={clean_snapshot.task}"[:170], fill="black")
    draw.text(
        (8, 54),
        f"target identities={list(occlusion.target_identity_keys)}"[:170],
        fill="black",
    )
    draw.text(
        (8, 77),
        f"Evaluator annotation stays outside the model; paired visual method={method}"[:170],
        fill="black",
    )
    draw.text(
        (8, 100),
        "Columns: factual+target bbox | factual ownership | removed | removed ownership",
        fill="black",
    )
    for camera_index, (clean_camera, occluded_camera) in enumerate(
        zip(clean_snapshot.cameras, occluded_snapshot.cameras, strict=True)
    ):
        report = report_by_host[clean_camera.image_key]
        clean_marked = Image.fromarray(clean_camera.source.copy())
        clean_draw = ImageDraw.Draw(clean_marked)
        if report.target_bbox_xyxy is not None:
            clean_draw.rectangle(report.target_bbox_xyxy, outline=(0, 255, 0), width=2)
        occluded_marked = Image.fromarray(occluded_camera.source.copy())
        occluded_draw = ImageDraw.Draw(occluded_marked)
        if report.occluder_bbox_xyxy is not None:
            occluded_draw.rectangle(report.occluder_bbox_xyxy, outline=(255, 0, 0), width=2)
        clean_probability = np.concatenate(
            (
                clean_camera.ownership[:, list(target_rows)],
                clean_camera.ownership[:, -1:],
            ),
            axis=-1,
        )
        occluded_probability = np.concatenate(
            (
                occluded_camera.ownership[:, list(target_rows)],
                occluded_camera.ownership[:, -1:],
            ),
            axis=-1,
        )
        panels = (
            _visual_panel(
                f"{clean_camera.image_key} factual / green target",
                np.asarray(clean_marked),
                None,
                (),
            ),
            _visual_panel(
                "factual target-row ownership",
                _posterior_overlay(
                    clean_camera.source,
                    clean_probability,
                    target_rows,
                ),
                clean_probability,
                target_rows,
            ),
            _visual_panel(
                f"removed / red bbox fraction={report.occluded_fraction:.4f}",
                np.asarray(occluded_marked),
                None,
                (),
            ),
            _visual_panel(
                "removed target-row ownership",
                _posterior_overlay(
                    occluded_camera.source,
                    occluded_probability,
                    target_rows,
                ),
                occluded_probability,
                target_rows,
            ),
        )
        y = header_height + camera_index * 350
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (320 * panel_index, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {
        "bytes": path.stat().st_size,
        "global_source_step": clean.sample.record.global_index,
        "kind": "controlled_task_target_occlusion_pair",
        "path": path.name,
        "rank": rank,
        "rank_sample_index_zero_based": rank_sample_index,
        "sha256": _sha256(path),
        "target_identity_keys": list(occlusion.target_identity_keys),
        "target_rows": list(target_rows),
        "task_key": clean.sample.host_sample.task_key,
    }


def _record_controlled_occlusion_rejection(
    *,
    output_dir: Path,
    clean: _PreparedActionStep,
    occluded: _PreparedActionStep,
    occlusion: Any,
    target_rows: Sequence[int],
    target_state: Mapping[str, Any],
    input_pair_integrity: Mapping[str, Any],
    reason: str,
    rank: int,
    rank_sample_index: int,
    audit_source_revision: str,
) -> dict[str, Any]:
    """Persist the first fail-closed controlled candidate instead of losing evidence."""

    stem = (
        f"rank{rank:02d}_rejected_frame{clean.sample.record.global_index:07d}_"
        f"{_safe_visual_component(clean.sample.host_sample.task_key)}_"
        f"{_safe_visual_component(reason)}"
    )
    visual = _render_controlled_occlusion_visual(
        path=output_dir / f"{stem}.png",
        clean=clean,
        occluded=occluded,
        occlusion=occlusion,
        target_rows=target_rows,
        rank=rank,
        rank_sample_index=rank_sample_index,
    )
    record = {
        "audit_source_revision": audit_source_revision,
        "evaluator_perturbation": occlusion.contract_dict(),
        "input_pair_integrity": dict(input_pair_integrity),
        "model_input_contains_loss_targets": False,
        "reason": reason,
        "schema": "picf-next.m4-controlled-occlusion-rejection.v1",
        "target_row_state": dict(target_state),
        "visual_artifact": visual,
    }
    _atomic_json(output_dir / f"{stem}.json", record)
    return record


def _record_controlled_pair_evidence(
    *,
    output_dir: Path,
    clean: _PreparedActionStep,
    occluded: _PreparedActionStep,
    occlusion: Any,
    target_rows: Sequence[int],
    target_state: Mapping[str, Any],
    input_pair_integrity: Mapping[str, Any],
    outcome: str,
    rank: int,
    rank_sample_index: int,
    audit_source_revision: str,
) -> dict[str, Any]:
    """Persist every immutable same-renderer pair, including passing cases."""

    stem = (
        f"rank{rank:02d}_pair_plan{clean.optimizer_plan_step:06d}_"
        f"frame{clean.sample.record.global_index:07d}_"
        f"{_safe_visual_component(clean.sample.host_sample.task_key)}_"
        f"{_safe_visual_component(outcome)}"
    )
    visual = _render_controlled_occlusion_visual(
        path=output_dir / f"{stem}.png",
        clean=clean,
        occluded=occluded,
        occlusion=occlusion,
        target_rows=target_rows,
        rank=rank,
        rank_sample_index=rank_sample_index,
    )
    record = {
        "audit_source_revision": audit_source_revision,
        "evaluator_perturbation": occlusion.contract_dict(),
        "input_pair_integrity": dict(input_pair_integrity),
        "model_input_contains_loss_targets": False,
        "outcome": outcome,
        "schema": "picf-next.m4-controlled-observation-pair.v1",
        "target_row_state": dict(target_state),
        "visual_artifact": visual,
    }
    _atomic_json(output_dir / f"{stem}.json", record)
    return record


def _render_action_visual(
    *,
    path: Path,
    step: _PreparedActionStep,
    rank: int,
    rank_sample_index: int,
    measurement: Mapping[str, Any],
    interventions: Mapping[str, Any],
) -> dict[str, Any]:
    """Render model-input RGB and runtime ownership without loss-only targets."""

    from PIL import Image, ImageDraw

    snapshot = _runtime_visual_snapshot(step)
    row_indices = tuple(int(row) for row in np.flatnonzero(snapshot.valid))
    if not row_indices:
        raise ValueError("action visual requires at least one valid posterior row")

    panels = []
    for camera in snapshot.cameras:
        local = np.concatenate(
            (camera.ownership[:, row_indices], camera.ownership[:, -1:]),
            axis=-1,
        )
        overlay = _posterior_overlay(camera.source, local, row_indices)
        camera_panels = (
            _visual_panel(f"{camera.image_key}: source RGB", camera.source, None, ()),
            _visual_panel(
                f"{camera.image_key}: runtime posterior ownership",
                overlay,
                local,
                row_indices,
            ),
        )
        row = Image.new("RGB", (640, 350), "white")
        for panel_index, panel in enumerate(camera_panels):
            row.paste(panel, (320 * panel_index, 0))
        panels.append(row)

    reference_delta_t_s = float(measurement["reference_delta_t_s"])
    header_height = 128
    legend_height = 25 * len(row_indices) + 48
    canvas = Image.new(
        "RGB",
        (640, header_height + 350 * len(panels) + legend_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        (
            f"rank={rank} sample={rank_sample_index} plan_step={step.optimizer_plan_step} "
            f"frame={step.sample.record.global_index} task={step.sample.host_sample.task_key}"
        ),
        fill="black",
    )
    draw.text((8, 31), f"instruction={step.sample.record.task}"[:105], fill="black")
    no_posterior_delta = interventions["conditions"]["without_posterior"][
        "loss_delta_from_baseline"
    ]
    wrong_address_delta = interventions["conditions"]["wrong_address"]["loss_delta_from_baseline"]
    selected_row_condition = (
        "remove_task_relevant_hidden_rows"
        if "remove_task_relevant_hidden_rows" in interventions["conditions"]
        else "remove_rows_at_least_one_reference_step"
    )
    old_row_delta = interventions["conditions"][selected_row_condition]["loss_delta_from_baseline"]
    draw.text(
        (8, 54),
        (
            f"max expected age={measurement['maximum_measurement_age_reference_steps']:.3f} "
            f"frames | no-posterior delta={no_posterior_delta:+.8f} | "
            f"wrong-address delta={wrong_address_delta:+.8f}"
        ),
        fill="black",
    )
    draw.text(
        (8, 78),
        f"selected-row removal delta={old_row_delta:+.8f} | "
        "no mask, bbox, simulator ID, or action target is drawn.",
        fill="black",
    )
    draw.text(
        (8, 102),
        (
            "Overlay uses each camera's first/global processor crop; "
            "every dense token remains in action."
        ),
        fill="black",
    )
    for panel_index, panel in enumerate(panels):
        canvas.paste(panel, (0, header_height + 350 * panel_index))

    legend_y = header_height + 350 * len(panels) + 8
    for row in row_indices:
        color = tuple(int(value) for value in _visual_color(row))
        existence = float(snapshot.existence[row])
        visibility = float(snapshot.visibility[row])
        age_steps = float(snapshot.measurement_age_s[row] / reference_delta_t_s)
        log_prior = float(snapshot.log_prior[row])
        draw.rectangle((8, legend_y, 24, legend_y + 12), fill=color)
        draw.text(
            (32, legend_y - 2),
            (
                f"posterior row={row:02d} existence={existence:.5f} "
                f"visibility={visibility:.5f} expected_age={age_steps:.3f} "
                f"log_prior={log_prior:.5f}"
            ),
            fill="black",
        )
        legend_y += 25
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {
        "bytes": path.stat().st_size,
        "global_source_step": step.sample.record.global_index,
        "kind": "current_runtime_posterior_ownership",
        "path": path.name,
        "rank": rank,
        "rank_sample_index_zero_based": rank_sample_index,
        "sha256": _sha256(path),
        "task_key": step.sample.host_sample.task_key,
    }


def _condition_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("condition means require at least one intervention sample")
    names = tuple(rows[0]["interventions"]["conditions"])
    if any(tuple(row["interventions"]["conditions"]) != names for row in rows):
        raise ValueError("action intervention conditions differ across samples")
    conditions = {}
    for name in names:
        records = [row["interventions"]["conditions"][name] for row in rows]
        conditions[name] = {
            field: sum(float(record[field]) for record in records) / len(records)
            for field in (
                "action_loss",
                "loss_delta_from_baseline",
                "velocity_rms_from_baseline",
            )
        }
        conditions[name]["positive_loss_delta_samples"] = sum(
            float(record["loss_delta_from_baseline"]) > 0.0 for record in records
        )
    return conditions


def _measurement_matches_age_selection(
    measurement: Mapping[str, Any],
    selection: str,
) -> bool:
    if selection not in _AGE_SELECTIONS:
        raise ValueError(f"unsupported measurement-age selection: {selection!r}")
    threshold = measurement.get("has_row_with_expected_age_at_least_one_reference_step")
    if not isinstance(threshold, bool):
        raise ValueError("measurement-age selection requires one boolean threshold result")
    if selection == "any":
        return True
    return threshold is (selection == "at-least-one-reference-step")


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("action intervention aggregate requires at least one sample")
    conditions = _condition_means(rows)
    ranks = sorted({int(row.get("rank", index)) for index, row in enumerate(rows)})
    for name, summary in conditions.items():
        positive_ranks = 0
        for rank in ranks:
            selected = [
                row for index, row in enumerate(rows) if int(row.get("rank", index)) == rank
            ]
            mean_delta = sum(
                float(row["interventions"]["conditions"][name]["loss_delta_from_baseline"])
                for row in selected
            ) / len(selected)
            positive_ranks += mean_delta > 0.0
        summary["positive_loss_delta_ranks"] = positive_ranks
    causal_names = tuple(
        name
        for name in (
            "without_posterior",
            "wrong_address",
            "remove_max_prior_row",
            "remove_rows_at_least_one_reference_step",
            "remove_task_relevant_hidden_rows",
            "remove_controlled_task_target_rows",
            "wrong_controlled_task_target_address",
            "stale_previous_frame",
        )
        if name in conditions
    )
    maximum_velocity_effect = max(
        conditions[name]["velocity_rms_from_baseline"] for name in causal_names
    )
    all_integrity_exact = all(
        bool(row["interventions"]["integrity"]["baseline_replay_exact"]) for row in rows
    )
    all_joint_permutations_exact = all(
        float(row["interventions"]["integrity"]["joint_permutation_action_loss_abs_delta"]) == 0.0
        and float(row["interventions"]["integrity"]["joint_permutation_velocity_rms"]) == 0.0
        for row in rows
    )
    strata = {}
    if all(isinstance(row.get("measurement"), Mapping) for row in rows):
        for name, expected in (
            ("maximum_expected_age_below_one_reference_step", False),
            ("maximum_expected_age_at_least_one_reference_step", True),
        ):
            selected = [
                row
                for row in rows
                if bool(row["measurement"]["has_row_with_expected_age_at_least_one_reference_step"])
                is expected
            ]
            strata[name] = {
                "conditions": _condition_means(selected) if selected else None,
                "sample_count": len(selected),
            }
    return {
        "all_baseline_replays_exact": all_integrity_exact,
        "all_joint_permutations_exact": all_joint_permutations_exact,
        "conditions": conditions,
        "maximum_causal_velocity_rms": maximum_velocity_effect,
        "measurement_interpretation": (
            "NO_MEASURABLE_POSTERIOR_EFFECT"
            if maximum_velocity_effect == 0.0
            else "MEASURABLE_POSTERIOR_EFFECT_DIRECTION_REQUIRES_BOUNDED_GATE"
        ),
        "m4_acceptance": "NOT_DECIDED_BY_READ_ONLY_DIAGNOSTIC",
        "rank_count": len(ranks),
        "sample_count": len(rows),
        "strata": strata,
    }


def _selection_outcome(
    rank_search_results: Sequence[Mapping[str, Any]],
    *,
    samples_per_rank: int,
) -> str:
    if not rank_search_results:
        raise ValueError("selection outcome requires rank search results")
    if (
        not isinstance(samples_per_rank, int)
        or isinstance(samples_per_rank, bool)
        or samples_per_rank <= 0
    ):
        raise ValueError("samples per rank must be positive")
    selected = []
    for result in rank_search_results:
        count = result.get("selected_samples")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("rank search selected-sample counts must be nonnegative integers")
        selected.append(count)
    requested = len(rank_search_results) * samples_per_rank
    found = sum(selected)
    if found == requested:
        return "REQUESTED_SAMPLE_COUNT_SATISFIED"
    if found == 0:
        return "NO_ELIGIBLE_SAMPLE_IN_SEARCH_WINDOW"
    return "PARTIAL_ELIGIBLE_SAMPLE_COVERAGE"


def main() -> None:
    args = _parse_args()
    import torch
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from safetensors.torch import load_model

    from picf_next.eval.calvin_controlled_occlusion import (
        build_calvin_controlled_rgb_occlusion,
    )
    from picf_next.eval.calvin_same_renderer_removal import (
        CalvinSameRendererRemovalStore,
    )
    from picf_next.eval.calvin_task_relevance import (
        CALVIN_TASK_PROTOCOL_SOURCE_SHA256,
        select_witnessed_task_rows,
        validate_calvin_task_protocol_sources,
    )
    from picf_next.hosts.molmoact2_training import CalvinVisibleObjectTargetBuilder
    from picf_next.training.molmoact2_calvin import (
        build_calvin_episode_stream_plan,
        build_molmoact2_calvin_training_stack,
        build_molmoact2_policy_config,
        load_calvin_training_assets,
    )
    from picf_next.training.recipe import load_training_recipe
    from picf_next.training.stationary_acceptance import (
        validate_stationary_temporal_acceptance,
    )
    from picf_next.training.stream_state import PosteriorStreamStateGroup

    started = time.perf_counter()
    audit_source_revision = _git_revision(_ROOT)
    output = args.output.expanduser().resolve()
    if Path("/mnt") not in output.parents:
        raise ValueError("cloud full-weight audit output must be a strict descendant of /mnt")
    if output.exists() or output.with_name(f".{output.name}.incomplete").exists():
        raise FileExistsError(output)
    visual_output_dir = None
    if args.visual_output_dir is not None:
        visual_output_dir = args.visual_output_dir.expanduser().resolve()
        if Path("/mnt") not in visual_output_dir.parents:
            raise ValueError("cloud action visuals must be a strict descendant of /mnt")
        if visual_output_dir.exists() or visual_output_dir.is_symlink():
            raise FileExistsError(visual_output_dir)
        visual_output_dir.mkdir(parents=True)
    checkpoint = args.training_checkpoint.expanduser().resolve()
    control, model_path, model_sha256 = _checkpoint_contract(checkpoint)
    contract = control.get("contract")
    progress = control.get("progress")
    plan_control = control.get("plan")
    if not isinstance(contract, dict) or not isinstance(progress, dict):
        raise ValueError("training checkpoint contract/progress is malformed")
    if not isinstance(plan_control, dict):
        raise ValueError("training checkpoint plan is malformed")
    arm = contract.get("arm_config", {}).get("causal_factorization")
    if not isinstance(arm, dict) or arm.get("id") != "C":
        raise ValueError("M4 action intervention audit requires the trained Arm C checkpoint")
    if (
        arm.get("include_causal_video") is not False
        or arm.get("include_posterior_action_context") is not True
    ):
        raise ValueError("Arm C causal factorization changed")
    checkpoint_steps = progress.get("successful_optimizer_steps")
    if (
        not isinstance(checkpoint_steps, int)
        or isinstance(checkpoint_steps, bool)
        or checkpoint_steps <= 0
        or progress.get("attempted_optimizer_steps") != checkpoint_steps
    ):
        raise ValueError("action checkpoint is not one completed successful prefix")
    checkpoint_plan_steps = plan_control.get("total_steps")
    if not isinstance(checkpoint_plan_steps, int) or checkpoint_plan_steps < checkpoint_steps:
        raise ValueError("action checkpoint plan length is malformed")
    if args.extended_plan_steps <= checkpoint_plan_steps:
        raise ValueError("extended action-audit plan must exceed checkpoint plan length")
    if args.maximum_pair_search_steps < 2:
        raise ValueError("action intervention requires at least two pair-search steps")
    if (
        not isinstance(args.samples_per_rank, int)
        or isinstance(args.samples_per_rank, bool)
        or args.samples_per_rank <= 0
    ):
        raise ValueError("samples per rank must be a positive integer")
    if args.maximum_samples_per_source_episode is not None and (
        not isinstance(args.maximum_samples_per_source_episode, int)
        or isinstance(args.maximum_samples_per_source_episode, bool)
        or args.maximum_samples_per_source_episode <= 0
    ):
        raise ValueError("maximum samples per source episode must be a positive integer")
    if (
        not isinstance(args.flow_randomness_repeats, int)
        or isinstance(args.flow_randomness_repeats, bool)
        or args.flow_randomness_repeats <= 0
    ):
        raise ValueError("flow randomness repeats must be a positive integer")
    if (
        not isinstance(args.temporal_visual_history_steps, int)
        or isinstance(args.temporal_visual_history_steps, bool)
        or args.temporal_visual_history_steps < 0
        or args.temporal_visual_history_steps == 1
    ):
        raise ValueError("temporal visual history steps must be zero or at least two")
    if args.temporal_visual_history_steps > 0 and visual_output_dir is None:
        raise ValueError("temporal visual history requires --visual-output-dir")
    if (
        args.selection_relevance == "controlled-task-occlusion"
        and args.selection_age_stratum != "any"
    ):
        raise ValueError(
            "controlled task occlusion audits every first eligible pair and requires "
            "--selection-age-stratum any"
        )
    if (
        args.same_renderer_removal_dir is not None
        and args.selection_relevance != "controlled-task-occlusion"
    ):
        raise ValueError("same-renderer removal probes require controlled-task-occlusion selection")

    ranks = tuple(args.ranks)
    world_size = contract.get("world_size")
    if (
        not isinstance(world_size, int)
        or not ranks
        or len(set(ranks)) != len(ranks)
        or any(not 0 <= rank < world_size for rank in ranks)
    ):
        raise ValueError("requested audit ranks differ from checkpoint world topology")
    if contract.get("gradient_accumulation_steps") != 1:
        raise ValueError("M4 action intervention audit requires one accumulation stream")

    recipe = load_training_recipe(args.recipe.resolve())
    if recipe.recipe_sha256 != contract.get("common_config", {}).get("recipe_sha256"):
        raise ValueError("action intervention recipe differs from the checkpoint")
    accepted = validate_stationary_temporal_acceptance(
        report_path=args.stationary_acceptance_report.expanduser().resolve(),
        checkpoint_path=args.stationary_checkpoint.expanduser().resolve(),
    )
    if accepted.contract_dict() != contract.get("arm_config", {}).get(
        "stationary_temporal_initialization"
    ):
        raise ValueError("accepted stationary core differs from checkpoint initialization")
    assets = load_calvin_training_assets(
        recipe,
        repository_root=_ROOT,
        split_root=args.dataset_split_root.expanduser().resolve(),
    )
    same_renderer_removal_store = (
        None
        if args.same_renderer_removal_dir is None
        else CalvinSameRendererRemovalStore(
            args.same_renderer_removal_dir.expanduser().resolve(),
            dataset_id=recipe.dataset.dataset_id,
            dataset_revision=recipe.dataset.dataset_revision,
        )
    )
    task_protocol_sources = None
    task_protocol_source_root = None
    task_protocol_sources_verified = False
    task_protocol_inventory = None
    target_builder = None
    if args.selection_relevance != "any":
        task_protocol_source_root = (
            args.calvin_protocol_source_root.expanduser().resolve()
            if args.calvin_protocol_source_root is not None
            else _ROOT / "references/source_checkouts/calvin"
        )
        expected_source_paths = tuple(
            task_protocol_source_root / relative_path
            for relative_path in CALVIN_TASK_PROTOCOL_SOURCE_SHA256
        )
        if args.calvin_protocol_source_root is not None or all(
            path.is_file() for path in expected_source_paths
        ):
            task_protocol_sources = validate_calvin_task_protocol_sources(task_protocol_source_root)
            task_protocol_sources_verified = True
        else:
            task_protocol_sources = dict(CALVIN_TASK_PROTOCOL_SOURCE_SHA256)
        first_sample = assets.dataset.by_key(assets.dataset.sample_keys[0])
        task_protocol_inventory = _task_protocol_inventory_for_sample(
            assets.physical_sidecar,
            first_sample,
        )
        target_builder = CalvinVisibleObjectTargetBuilder(assets.physical_sidecar)
    checkpoint_plan = build_calvin_episode_stream_plan(
        recipe,
        assets.dataset,
        comparison_id=str(contract["comparison_id"]),
        seed=int(plan_control["seed"]),
        global_batch_size=int(contract["optimizer_global_batch_size"]),
        total_steps=checkpoint_plan_steps,
    )
    if checkpoint_plan.plan_sha256 != control.get("plan_sha256"):
        raise ValueError("reconstructed checkpoint plan differs from checkpoint control")
    extended_plan = _extend_plan_for_read_only_audit(
        checkpoint_plan,
        total_steps=args.extended_plan_steps,
    )
    validated_prefix_steps = _validate_extended_plan_prefix(checkpoint_plan, extended_plan)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-weight M4 action intervention audit requires CUDA")
    policy_config = build_molmoact2_policy_config(
        recipe,
        checkpoint_path=args.foundation_checkpoint_dir.expanduser().resolve(),
    )
    policy = MolmoAct2Policy(policy_config).to(device).eval()
    stack = build_molmoact2_calvin_training_stack(
        recipe,
        policy=policy,
        assets=assets,
        accepted_temporal_core=accepted,
        include_posterior_action_context=True,
    )
    module = stack.module.to(device).eval()
    missing, unexpected = load_model(module, model_path, strict=True, device=str(device))
    if missing or unexpected:
        raise RuntimeError("strict action-checkpoint load reported key drift")
    image_patch_token_id = getattr(
        policy._backbone().config,
        "image_patch_id",
        None,
    )
    if not isinstance(image_patch_token_id, int) or isinstance(image_patch_token_id, bool):
        raise RuntimeError("MolmoAct2 backbone lacks one integer image_patch_id")
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    rank_rows: list[dict[str, Any]] = []
    rank_search_results: list[dict[str, Any]] = []
    visual_artifacts: list[dict[str, Any]] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for rank in ranks:
            state_name = f"picf_rank_state_{rank:05d}.pt"
            state_path, state_sha256 = _validated_checkpoint_file(
                checkpoint,
                control,
                state_name,
            )
            state_payload = torch.load(state_path, map_location="cpu", weights_only=True)
            raw_streams = state_payload.get("streams") if isinstance(state_payload, dict) else None
            raw_stream = (
                raw_streams.get("accumulation-00000") if isinstance(raw_streams, dict) else None
            )
            raw_belief = raw_stream.get("belief") if isinstance(raw_stream, dict) else None
            reference = raw_belief.get("address_mean") if isinstance(raw_belief, dict) else None
            if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
                raise ValueError("rank-local posterior state has no floating address belief")
            streams = PosteriorStreamStateGroup.for_rank_partition(
                recipe.core_config.temporal,
                extended_plan,
                rank=rank,
                world_size=world_size,
                gradient_accumulation_steps=1,
                capacity=recipe.core_config.posterior_capacity,
                device=device,
                dtype=reference.dtype,
                max_parameter_lag=0,
            )
            streams.load_state_dict(state_payload)
            stream = streams["accumulation-00000"]
            versions = set(stream.state_parameter_versions)
            if len(versions) != 1:
                raise ValueError("rank-local posterior lanes have different parameter versions")
            state_parameter_version = next(iter(versions))
            previous_evidence = None
            previous_episode = None
            previous_task = None
            previous_step = None
            selected_count = 0
            selected_source_episodes: dict[str, int] = {}
            searched_steps = 0
            consecutive_pair_frames = 0
            age_eligible_pair_frames = 0
            relevance_eligible_pair_frames = 0
            jointly_eligible_pair_frames = 0
            task_relevance_reasons: Counter[str] = Counter()
            task_relevance_exact_frames = 0
            task_relevance_eligible_frames = 0
            task_relevance_track_conflicts = 0
            controlled_occlusion_reasons: Counter[str] = Counter()
            controlled_occlusion_pairs_prepared = 0
            previous_identity_attribution = None
            visual_history: deque[_ActionVisualSnapshot] | None = (
                deque(maxlen=args.temporal_visual_history_steps)
                if args.temporal_visual_history_steps > 0
                else None
            )
            search_stop = min(
                args.extended_plan_steps,
                checkpoint_steps + args.maximum_pair_search_steps,
            )
            for plan_step in range(checkpoint_steps, search_stop):
                microbatch = extended_plan.microbatch_for_rank(
                    plan_step,
                    rank=rank,
                    world_size=world_size,
                    gradient_accumulation_steps=1,
                    accumulation_index=0,
                )
                if len(microbatch.transitions) != 1:
                    raise ValueError("action intervention requires one sample per rank")
                planned = microbatch.transitions[0]
                initial_belief = stream.prepare_planned_transitions(
                    microbatch.transitions,
                    current_parameter_version=state_parameter_version,
                )
                prior_identity_keys_by_row = stream.pending_loss_track_keys_by_row
                prepared = _prepare_action_step(
                    stack,
                    planned,
                    (
                        _clone_object_belief(initial_belief)
                        if args.selection_relevance == "controlled-task-occlusion"
                        else initial_belief
                    ),
                    optimizer_plan_step=plan_step,
                )
                searched_steps += 1
                same_stream = (
                    previous_evidence is not None
                    and previous_episode == planned.episode_instance_id
                    and previous_task == prepared.sample.record.task
                    and planned.transition_index == previous_step + 1
                )
                controlled_selection = None
                controlled_occlusion = None
                controlled_clean_prepared = prepared
                controlled_pair_integrity = None
                controlled_pair_is_same_renderer = False
                occluded_prepared = None
                if args.selection_relevance == "controlled-task-occlusion":
                    controlled_selection = select_witnessed_task_rows(
                        task_key=prepared.sample.host_sample.task_key,
                        identity_keys_by_row=prior_identity_keys_by_row[0],
                        row_valid=tuple(
                            bool(value) for value in initial_belief.valid[0].detach().cpu().tolist()
                        ),
                    )
                    if not same_stream:
                        controlled_reason = "source stream is not contiguous"
                    elif previous_identity_attribution is None:
                        controlled_reason = "previous clean frame has no loss-side attribution"
                    elif previous_identity_attribution.track_conflicts:
                        controlled_reason = "previous clean frame has a row-attribution conflict"
                    elif not controlled_selection.eligible:
                        controlled_reason = controlled_selection.reason
                    elif same_renderer_removal_store is not None:
                        controlled_occlusion = same_renderer_removal_store(
                            prepared.sample.picf_evidence_frame,
                            global_index=prepared.sample.record.global_index,
                            target_identity_keys=(controlled_selection.action_target_identity_keys),
                        )
                        if controlled_occlusion is None:
                            controlled_reason = (
                                "no immutable same-renderer pair for this frame and target"
                            )
                        else:
                            controlled_clean_prepared = _prepare_action_step(
                                stack,
                                planned,
                                _clone_object_belief(initial_belief),
                                optimizer_plan_step=plan_step,
                                evidence_frame_override=(
                                    controlled_occlusion.factual_evidence_frame
                                ),
                            )
                            occluded_prepared = _prepare_action_step(
                                stack,
                                planned,
                                _clone_object_belief(initial_belief),
                                optimizer_plan_step=plan_step,
                                evidence_frame_override=controlled_occlusion.evidence_frame,
                            )
                            controlled_pair_integrity = _validate_controlled_observation_pair(
                                controlled_clean_prepared,
                                occluded_prepared,
                                image_patch_token_id=image_patch_token_id,
                            )
                            controlled_pair_is_same_renderer = True
                            controlled_reason = "same-renderer task-target pair prepared"
                            controlled_occlusion_pairs_prepared += 1
                    else:
                        physical_frame = assets.physical_sidecar(
                            prepared.sample.record.task_index,
                            prepared.sample.record.global_index,
                        )
                        try:
                            controlled_occlusion = build_calvin_controlled_rgb_occlusion(
                                prepared.sample.picf_evidence_frame,
                                physical_frame,
                                target_identity_keys=(
                                    controlled_selection.action_target_identity_keys
                                ),
                            )
                        except ValueError as error:
                            if str(error) != (
                                "controlled occlusion target has no visible pixel in either RGB "
                                "camera"
                            ):
                                raise
                            controlled_reason = "task target has no visible RGB pixel to occlude"
                        else:
                            occluded_prepared = _prepare_action_step(
                                stack,
                                planned,
                                _clone_object_belief(initial_belief),
                                optimizer_plan_step=plan_step,
                                evidence_frame_override=controlled_occlusion.evidence_frame,
                            )
                            controlled_pair_integrity = _validate_controlled_observation_pair(
                                prepared,
                                occluded_prepared,
                                image_patch_token_id=image_patch_token_id,
                            )
                            controlled_reason = "controlled task-target pair prepared"
                            controlled_occlusion_pairs_prepared += 1
                    controlled_occlusion_reasons[controlled_reason] += 1
                identity_attribution = None
                controlled_clean_identity_attribution = None
                if target_builder is not None:
                    identity_attribution = _advance_audit_identity_track(
                        prepared,
                        previous_keys_by_row=prior_identity_keys_by_row,
                        target_builder=target_builder,
                        set_criterion=module.joint_bridge.objective.set_criterion,
                        reference_delta_t_s=(recipe.core_config.temporal.reference_delta_t_s),
                    )
                    selection = identity_attribution.task_selection
                    task_relevance_reasons[selection.reason] += 1
                    task_relevance_exact_frames += int(selection.exact_action_target)
                    task_relevance_eligible_frames += int(selection.eligible)
                    task_relevance_track_conflicts += identity_attribution.track_conflicts
                    controlled_clean_identity_attribution = (
                        identity_attribution
                        if controlled_clean_prepared is prepared
                        else _advance_audit_identity_track(
                            controlled_clean_prepared,
                            previous_keys_by_row=prior_identity_keys_by_row,
                            target_builder=target_builder,
                            set_criterion=module.joint_bridge.objective.set_criterion,
                            reference_delta_t_s=(recipe.core_config.temporal.reference_delta_t_s),
                        )
                    )
                if visual_history is not None:
                    if not same_stream:
                        visual_history.clear()
                    visual_history.append(_runtime_visual_snapshot(prepared))
                audit_prepared = (
                    occluded_prepared
                    if args.selection_relevance == "controlled-task-occlusion"
                    and occluded_prepared is not None
                    else prepared
                )
                controlled_target_state = None
                if (
                    args.selection_relevance == "controlled-task-occlusion"
                    and occluded_prepared is not None
                ):
                    if controlled_selection is None:
                        raise RuntimeError("controlled pair lost its witnessed target rows")
                    controlled_target_state = _controlled_target_row_summary(
                        controlled_clean_prepared,
                        occluded_prepared,
                        controlled_selection.row_indices,
                        reference_delta_t_s=recipe.core_config.temporal.reference_delta_t_s,
                        clean_identity_attribution=controlled_clean_identity_attribution,
                        target_identity_keys=controlled_selection.row_identity_keys,
                    )
                    if not controlled_target_state["all_target_rows_clean_recognized"]:
                        reason = "factual branch did not MAP-associate the supervised target"
                        if visual_output_dir is not None:
                            _record_controlled_occlusion_rejection(
                                output_dir=visual_output_dir,
                                clean=controlled_clean_prepared,
                                occluded=occluded_prepared,
                                occlusion=controlled_occlusion,
                                target_rows=controlled_selection.row_indices,
                                target_state=controlled_target_state,
                                input_pair_integrity=controlled_pair_integrity,
                                reason=reason,
                                rank=rank,
                                rank_sample_index=selected_count,
                                audit_source_revision=audit_source_revision,
                            )
                        raise RuntimeError(
                            f"{reason}; target_state="
                            f"{json.dumps(controlled_target_state, sort_keys=True)}"
                        )
                    if not controlled_target_state["all_target_rows_remain_in_posterior"]:
                        reason = "controlled absence deleted a witnessed Bernoulli component"
                        if visual_output_dir is not None:
                            _record_controlled_occlusion_rejection(
                                output_dir=visual_output_dir,
                                clean=controlled_clean_prepared,
                                occluded=occluded_prepared,
                                occlusion=controlled_occlusion,
                                target_rows=controlled_selection.row_indices,
                                target_state=controlled_target_state,
                                input_pair_integrity=controlled_pair_integrity,
                                reason=reason,
                                rank=rank,
                                rank_sample_index=selected_count,
                                audit_source_revision=audit_source_revision,
                            )
                        raise RuntimeError(
                            f"{reason}; target_state="
                            f"{json.dumps(controlled_target_state, sort_keys=True)}"
                        )
                    if not controlled_target_state["all_target_rows_map_missed"]:
                        reason = "controlled absence MAP-associated a witnessed target row"
                        if visual_output_dir is not None:
                            _record_controlled_occlusion_rejection(
                                output_dir=visual_output_dir,
                                clean=controlled_clean_prepared,
                                occluded=occluded_prepared,
                                occlusion=controlled_occlusion,
                                target_rows=controlled_selection.row_indices,
                                target_state=controlled_target_state,
                                input_pair_integrity=controlled_pair_integrity,
                                reason=reason,
                                rank=rank,
                                rank_sample_index=selected_count,
                                audit_source_revision=audit_source_revision,
                            )
                        raise RuntimeError(
                            f"{reason}; target_state="
                            f"{json.dumps(controlled_target_state, sort_keys=True)}"
                        )
                    if not controlled_target_state["all_target_rows_action_exposed"]:
                        controlled_occlusion_reasons[
                            "target retained below MAP action extraction"
                        ] += 1
                    if controlled_pair_is_same_renderer and visual_output_dir is not None:
                        controlled_evidence_record = _record_controlled_pair_evidence(
                            output_dir=visual_output_dir,
                            clean=controlled_clean_prepared,
                            occluded=occluded_prepared,
                            occlusion=controlled_occlusion,
                            target_rows=controlled_selection.row_indices,
                            target_state=controlled_target_state,
                            input_pair_integrity=controlled_pair_integrity,
                            outcome="measurement-gate-passed",
                            rank=rank,
                            rank_sample_index=selected_count,
                            audit_source_revision=audit_source_revision,
                        )
                        visual_artifacts.append(controlled_evidence_record["visual_artifact"])
                valid_count = (
                    0
                    if audit_prepared.evidence.object_valid is None
                    else int(audit_prepared.evidence.object_valid.sum().item())
                )
                stream.commit_chunk(
                    prepared.final_belief,
                    transition_count=1,
                    state_parameter_version=state_parameter_version,
                    final_loss_track_keys_by_row=(
                        None
                        if identity_attribution is None
                        else identity_attribution.next_keys_by_row
                    ),
                )
                if same_stream and valid_count >= 2:
                    consecutive_pair_frames += 1
                    if previous_evidence is None:
                        raise RuntimeError("consecutive intervention pair lost stale evidence")
                    measurement = _measurement_summary(
                        audit_prepared,
                        reference_delta_t_s=recipe.core_config.temporal.reference_delta_t_s,
                    )
                    source_episode = prepared.sample.episode_key
                    episode_count = selected_source_episodes.get(source_episode, 0)
                    under_episode_cap = (
                        args.maximum_samples_per_source_episode is None
                        or episode_count < args.maximum_samples_per_source_episode
                    )
                    if args.selection_relevance == "any":
                        relevance_matches = True
                    elif args.selection_relevance == "task-relevant-hidden":
                        relevance_matches = (
                            identity_attribution is not None
                            and identity_attribution.task_selection.eligible
                        )
                    else:
                        relevance_matches = (
                            controlled_selection is not None
                            and controlled_selection.eligible
                            and controlled_occlusion is not None
                            and occluded_prepared is not None
                            and controlled_pair_integrity is not None
                            and controlled_target_state is not None
                            and controlled_target_state["all_target_rows_action_exposed"]
                        )
                    age_matches = _measurement_matches_age_selection(
                        measurement,
                        args.selection_age_stratum,
                    )
                    age_eligible_pair_frames += int(age_matches)
                    relevance_eligible_pair_frames += int(relevance_matches)
                    jointly_eligible_pair_frames += int(age_matches and relevance_matches)
                    if age_matches and relevance_matches and under_episode_cap:
                        flow_randomness, flow_contract = _materialize_flow_repeats(
                            audit_prepared,
                            stack,
                            repeat_count=args.flow_randomness_repeats,
                            seed_mode=args.flow_randomness_seed_mode,
                        )
                        task_relevant_hidden_mask = None
                        controlled_task_target_mask = None
                        if args.selection_relevance == "task-relevant-hidden":
                            if identity_attribution is None:
                                raise RuntimeError(
                                    "strict task relevance lost post-forward attribution"
                                )
                            task_relevant_hidden_mask = torch.zeros_like(
                                prepared.evidence.object_valid
                            )
                            task_relevant_hidden_mask[
                                0,
                                list(identity_attribution.task_selection.row_indices),
                            ] = True
                        if args.selection_relevance == "controlled-task-occlusion":
                            if (
                                controlled_selection is None
                                or controlled_occlusion is None
                                or occluded_prepared is None
                                or controlled_pair_integrity is None
                                or controlled_target_state is None
                            ):
                                raise RuntimeError(
                                    "controlled task selection lost its paired observation"
                                )
                            controlled_task_target_mask = torch.zeros_like(
                                audit_prepared.evidence.object_valid
                            )
                            controlled_task_target_mask[
                                0,
                                list(controlled_selection.row_indices),
                            ] = True
                        interventions = _evaluate_interventions(
                            audit_prepared,
                            previous_evidence,
                            stack,
                            flow_randomness=flow_randomness,
                            reference_delta_t_s=(recipe.core_config.temporal.reference_delta_t_s),
                            task_relevant_hidden_rows=task_relevant_hidden_mask,
                            controlled_task_target_rows=controlled_task_target_mask,
                            clean_observation_reference=(
                                controlled_clean_prepared
                                if args.selection_relevance == "controlled-task-occlusion"
                                else None
                            ),
                        )
                        valid = audit_prepared.evidence.object_valid
                        log_prior = audit_prepared.evidence.object_log_prior
                        if valid is None or log_prior is None:
                            raise RuntimeError("selected Arm C evidence lost its posterior bank")
                        row = {
                            "episode_instance_id": (
                                prepared.planned_transition.episode_instance_id
                            ),
                            "flow_randomness": {
                                "official_flow_timesteps_per_repeat": (
                                    recipe.policy.num_flow_timesteps
                                ),
                                "repeats": flow_contract,
                                "seed_mode": args.flow_randomness_seed_mode,
                            },
                            "global_source_step": prepared.sample.record.global_index,
                            "interventions": interventions,
                            "measurement": measurement,
                            "optimizer_plan_step_zero_based": prepared.optimizer_plan_step,
                            "posterior_log_priors": [
                                float(value)
                                for value in (
                                    log_prior[0, valid[0]].detach().float().cpu().tolist()
                                )
                            ],
                            "posterior_valid_rows": int(valid.sum().item()),
                            "rank": rank,
                            "rank_sample_index_zero_based": selected_count,
                            "rank_state_sha256": state_sha256,
                            "sample_key": prepared.sample.sample_key,
                            "source_episode_key": source_episode,
                            "task": prepared.sample.record.task,
                            "task_key": prepared.sample.host_sample.task_key,
                            "transition_index": prepared.planned_transition.transition_index,
                        }
                        if identity_attribution is not None:
                            selection = identity_attribution.task_selection
                            row["loss_side_task_relevance"] = {
                                "action_target_identity_keys": list(
                                    selection.action_target_identity_keys
                                ),
                                "currently_measurable_identity_keys": list(
                                    identity_attribution.currently_measurable_identity_keys
                                ),
                                "eligible": selection.eligible,
                                "exact_action_target": selection.exact_action_target,
                                "identity_keys_by_posterior_row": list(
                                    identity_attribution.next_keys_by_row[0]
                                ),
                                "reason": selection.reason,
                                "row_identity_keys": list(selection.row_identity_keys),
                                "row_indices": list(selection.row_indices),
                                "track_conflicts_this_frame": (
                                    identity_attribution.track_conflicts
                                ),
                            }
                        if args.selection_relevance == "controlled-task-occlusion":
                            if (
                                controlled_selection is None
                                or controlled_occlusion is None
                                or occluded_prepared is None
                                or controlled_pair_integrity is None
                            ):
                                raise RuntimeError(
                                    "controlled task report lost its paired observation"
                                )
                            row["controlled_task_occlusion"] = {
                                "evaluator_perturbation": controlled_occlusion.contract_dict(),
                                "input_pair_integrity": controlled_pair_integrity,
                                "prior_identity_keys_by_posterior_row": list(
                                    prior_identity_keys_by_row[0]
                                ),
                                "previous_clean_track_conflicts": (
                                    previous_identity_attribution.track_conflicts
                                    if previous_identity_attribution is not None
                                    else None
                                ),
                                "target_row_state": controlled_target_state,
                                "witness_selection": {
                                    "action_target_identity_keys": list(
                                        controlled_selection.action_target_identity_keys
                                    ),
                                    "exact_action_target": (
                                        controlled_selection.exact_action_target
                                    ),
                                    "reason": controlled_selection.reason,
                                    "row_identity_keys": list(
                                        controlled_selection.row_identity_keys
                                    ),
                                    "row_indices": list(controlled_selection.row_indices),
                                },
                            }
                        if visual_history is not None:
                            if args.selection_relevance == "task-relevant-hidden":
                                intervention_rows = interventions[
                                    "removed_task_relevant_hidden_rows"
                                ]
                            elif args.selection_relevance == "controlled-task-occlusion":
                                intervention_rows = interventions["controlled_task_target_rows"]
                            else:
                                intervention_rows = interventions[
                                    "removed_rows_at_least_one_reference_step"
                                ]
                            if intervention_rows is None:
                                raise RuntimeError("selected intervention rows are absent")
                            old_rows = tuple(int(old_row) for old_row in intervention_rows)
                            row["old_row_trajectories"] = {
                                str(old_row): _old_row_trajectory_records(
                                    tuple(visual_history),
                                    old_row,
                                    reference_delta_t_s=(
                                        recipe.core_config.temporal.reference_delta_t_s
                                    ),
                                )
                                for old_row in old_rows
                            }
                        if visual_output_dir is not None:
                            visual_name = (
                                f"rank{rank:02d}_sample{selected_count:02d}_"
                                f"frame{prepared.sample.record.global_index:07d}_"
                                f"{_safe_visual_component(prepared.sample.host_sample.task_key)}.png"
                            )
                            if args.selection_relevance == "controlled-task-occlusion":
                                if (
                                    controlled_selection is None
                                    or controlled_occlusion is None
                                    or occluded_prepared is None
                                ):
                                    raise RuntimeError(
                                        "controlled task visual lost its paired observation"
                                    )
                                visual = _render_controlled_occlusion_visual(
                                    path=visual_output_dir / visual_name,
                                    clean=controlled_clean_prepared,
                                    occluded=occluded_prepared,
                                    occlusion=controlled_occlusion,
                                    target_rows=controlled_selection.row_indices,
                                    rank=rank,
                                    rank_sample_index=selected_count,
                                )
                            else:
                                visual = _render_action_visual(
                                    path=visual_output_dir / visual_name,
                                    step=prepared,
                                    rank=rank,
                                    rank_sample_index=selected_count,
                                    measurement=measurement,
                                    interventions=interventions,
                                )
                            row["visual_artifact"] = visual
                            visual_artifacts.append(visual)
                            if (
                                visual_history is not None
                                and args.selection_relevance != "controlled-task-occlusion"
                            ):
                                temporal_visuals = _render_old_row_trajectories(
                                    path_stem=(
                                        visual_output_dir
                                        / f"rank{rank:02d}_sample{selected_count:02d}_"
                                        f"frame{prepared.sample.record.global_index:07d}_"
                                        f"{_safe_visual_component(prepared.sample.host_sample.task_key)}_history"
                                    ),
                                    history=tuple(visual_history),
                                    rows=old_rows,
                                    rank=rank,
                                    rank_sample_index=selected_count,
                                    reference_delta_t_s=(
                                        recipe.core_config.temporal.reference_delta_t_s
                                    ),
                                    loss_delta_from_baseline=interventions["conditions"][
                                        (
                                            "remove_task_relevant_hidden_rows"
                                            if args.selection_relevance == "task-relevant-hidden"
                                            else "remove_rows_at_least_one_reference_step"
                                        )
                                    ]["loss_delta_from_baseline"],
                                )
                                row["temporal_visual_artifacts"] = temporal_visuals
                                visual_artifacts.extend(temporal_visuals)
                        rank_rows.append(row)
                        selected_source_episodes[source_episode] = episode_count + 1
                        selected_count += 1
                previous_evidence = prepared.evidence
                previous_episode = planned.episode_instance_id
                previous_task = prepared.sample.record.task
                previous_step = planned.transition_index
                previous_identity_attribution = identity_attribution
                del prepared, controlled_clean_prepared, audit_prepared, occluded_prepared
                torch.cuda.empty_cache()
                if selected_count == args.samples_per_rank:
                    break
            rank_search_results.append(
                {
                    "age_eligible_consecutive_pair_frames": age_eligible_pair_frames,
                    "consecutive_two_object_pair_frames": consecutive_pair_frames,
                    "jointly_eligible_consecutive_pair_frames": jointly_eligible_pair_frames,
                    "plan_step_start_inclusive": checkpoint_steps,
                    "plan_step_stop_exclusive": checkpoint_steps + searched_steps,
                    "rank": rank,
                    "relevance_eligible_consecutive_pair_frames": (relevance_eligible_pair_frames),
                    "requested_samples": args.samples_per_rank,
                    "searched_steps": searched_steps,
                    "selected_samples": selected_count,
                    "controlled_occlusion": {
                        "pairs_prepared": controlled_occlusion_pairs_prepared,
                        "reason_counts": dict(sorted(controlled_occlusion_reasons.items())),
                    },
                    "task_relevance": {
                        "eligible_frames": task_relevance_eligible_frames,
                        "exact_task_frames": task_relevance_exact_frames,
                        "reason_counts": dict(sorted(task_relevance_reasons.items())),
                        "track_conflicts": task_relevance_track_conflicts,
                    },
                }
            )
            del streams, previous_evidence
            gc.collect()
            torch.cuda.empty_cache()

    adapter = module.joint_bridge.sequence_bridge.action_adapter
    report = {
        "aggregate": _aggregate(rank_rows) if rank_rows else None,
        "audit_source_revision": audit_source_revision,
        "checkpoint": {
            "completed_optimizer_steps": checkpoint_steps,
            "control_manifest_sha256": _sha256(checkpoint / "picf_control.json"),
            "model_sha256": model_sha256,
            "path": str(checkpoint),
            "training_source_revision": contract.get("code_revision"),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "gates": {
            "dense_abs_mean": float(adapter.dense_gates.detach().float().abs().mean().cpu()),
            "object_abs_mean": float(adapter.object_gates.detach().float().abs().mean().cpu()),
        },
        "plan": {
            "checkpoint_plan_sha256": checkpoint_plan.plan_sha256,
            "extension_execution": "read_only_no_optimizer.v1",
            "extended_plan_sha256": extended_plan.plan_sha256,
            "extended_plan_steps": extended_plan.total_steps,
            "flow_randomness_repeats": args.flow_randomness_repeats,
            "flow_randomness_seed_mode": args.flow_randomness_seed_mode,
            "maximum_pair_search_steps": args.maximum_pair_search_steps,
            "maximum_samples_per_source_episode": (args.maximum_samples_per_source_episode),
            "reference_delta_t_s": recipe.core_config.temporal.reference_delta_t_s,
            "samples_per_rank": args.samples_per_rank,
            "selection_age_stratum": args.selection_age_stratum,
            "selection_relevance": args.selection_relevance,
            "temporal_visual_history_steps": args.temporal_visual_history_steps,
            "validated_exact_prefix_steps": validated_prefix_steps,
        },
        "rank_results": rank_rows,
        "recipe_sha256": recipe.recipe_sha256,
        "same_renderer_removal_store": (
            None
            if same_renderer_removal_store is None
            else {
                "keys": [list(key) for key in same_renderer_removal_store.keys],
                "root": str(same_renderer_removal_store.root),
                "summary_sha256": same_renderer_removal_store.summary_sha256,
            }
        ),
        "schema": "picf-next.m4-action-intervention-audit.v12",
        "search_results": rank_search_results,
        "selection_outcome": _selection_outcome(
            rank_search_results,
            samples_per_rank=args.samples_per_rank,
        ),
        "stationary_temporal_initialization": accepted.contract_dict(),
        "task_relevance_protocol": {
            "controlled_occlusion_evaluator_only": (
                args.selection_relevance == "controlled-task-occlusion"
            ),
            "enabled": args.selection_relevance != "any",
            "post_forward_loss_side_only": True,
            "runtime_or_training_input": False,
            "sidecar_identity_keys": task_protocol_inventory,
            "source_files_root": str(task_protocol_source_root),
            "source_files_verified_in_this_run": task_protocol_sources_verified,
            "source_sha256": task_protocol_sources,
        },
        "visual_artifacts": {
            "directory": None if visual_output_dir is None else str(visual_output_dir),
            "files": visual_artifacts,
            "model_input_contains_loss_targets": False,
            "runtime_only_no_loss_targets": (
                args.selection_relevance != "controlled-task-occlusion"
            ),
            "visualization_contains_evaluator_only_target_annotation": (
                args.selection_relevance == "controlled-task-occlusion"
            ),
        },
    }
    _atomic_json(output, report)
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
