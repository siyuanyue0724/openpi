"""Canonical MolmoAct2/CALVIN assembly for the strict PICF recipe.

This module owns no learned component. It closes the deployment graph from one
hash-addressed recipe and rejects policy, dataset, artifact or parameter-scope
drift before Accelerate wraps the model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch import nn

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.calvin_geometry_sidecar import CalvinPhysicalGeometrySidecar
from picf_next.data.calvin_normalization import (
    CALVIN_NORMALIZATION_SCHEMA,
    official_molmoact2_dataset_stats,
    validate_calvin_normalization_artifact,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    read_sha256_verified_file_beneath,
    validate_dataset_files,
)
from picf_next.hosts.molmoact2 import (
    MolmoAct2PICFActionExpert,
    install_molmoact2_lerobot_picf_adapter,
)
from picf_next.hosts.molmoact2_calvin_processor import CalvinMolmoAct2ProcessorBridge
from picf_next.hosts.molmoact2_training import (
    CalvinStatefulMolmoAct2TrainingModule,
    CalvinStatefulNativeBankBuilder,
    MolmoAct2PICFJointTrainingBridge,
    MolmoAct2PICFTrainingBridge,
)
from picf_next.models.core import PICFCore
from picf_next.training.control import (
    EpisodeSampleSequence,
    FrozenEpisodeStreamPlan,
)
from picf_next.training.recipe import PICFTrainingRecipe
from picf_next.training.stage_checkpoints import load_stationary_temporal_checkpoint
from picf_next.training.stationary_acceptance import AcceptedStationaryTemporalCore

_MAXIMUM_RECIPE_ARTIFACT_BYTES = 64 * 1024 * 1024
_ACTION_TRAINING_STAGES = {"M4_action_adoption", "M5_three_seed_abc", "M6_long_train"}


@dataclass(frozen=True, slots=True)
class CalvinTrainingAssets:
    index: CalvinDatasetIndex
    dataset: CalvinStatefulTransitionDataset
    dataset_manifest: DatasetFileManifest
    normalization_payload: dict[str, object]
    physical_sidecar: CalvinPhysicalSupervisionSidecar
    geometry_sidecar: CalvinPhysicalGeometrySidecar


@dataclass(frozen=True, slots=True)
class MolmoAct2CalvinTrainingStack:
    policy_config: object
    processor: CalvinMolmoAct2ProcessorBridge
    assets: CalvinTrainingAssets
    module: CalvinStatefulMolmoAct2TrainingModule
    accepted_temporal_core: AcceptedStationaryTemporalCore


def _artifact_root(repository_root: Path, manifest_path: str) -> Path:
    return (repository_root.resolve() / manifest_path).resolve().parent


def _verified_artifact(
    repository: Path,
    relative_path: str,
    expected_sha256: str,
) -> bytes:
    return read_sha256_verified_file_beneath(
        repository,
        relative_path,
        expected_sha256=expected_sha256,
        maximum_bytes=_MAXIMUM_RECIPE_ARTIFACT_BYTES,
    )


def _json_object(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ContractError(f"{name} must contain one JSON object")
    return value


def load_calvin_training_assets(
    recipe: PICFTrainingRecipe,
    *,
    repository_root: str | Path,
    split_root: str | Path,
) -> CalvinTrainingAssets:
    """Load every data-side dependency after content and identity validation."""

    if not isinstance(recipe, PICFTrainingRecipe):
        raise TypeError("CALVIN assembly requires a PICFTrainingRecipe")
    repository = Path(repository_root).resolve()
    recipe.validate_repository_artifacts(repository)
    split = Path(split_root).resolve()
    if split.name != recipe.dataset.split_name:
        raise ContractError("CALVIN split path name differs from the training recipe")
    dataset_manifest = DatasetFileManifest.from_dict(
        _json_object(
            _verified_artifact(
                repository,
                recipe.artifacts.dataset_file_manifest_path,
                recipe.artifacts.dataset_file_manifest_sha256,
            ),
            "dataset file manifest",
        )
    )
    validate_dataset_files(
        dataset_manifest,
        split,
        dataset_id=recipe.dataset.dataset_id,
        dataset_revision=recipe.dataset.dataset_revision,
        split_name=recipe.dataset.split_name,
        verify_hashes=True,
    )
    index = CalvinDatasetIndex.load(
        split,
        dataset_id=recipe.dataset.dataset_id,
        dataset_revision=recipe.dataset.dataset_revision,
        verify_files=True,
        dataset_manifest=dataset_manifest,
    )
    dataset = CalvinStatefulTransitionDataset(
        index,
        action_horizon=recipe.dataset.action_horizon,
    )
    normalization = _json_object(
        _verified_artifact(
            repository,
            recipe.artifacts.normalization_path,
            recipe.artifacts.normalization_file_sha256,
        ),
        "normalization artifact",
    )
    validate_calvin_normalization_artifact(normalization)
    if (
        normalization.get("schema") == CALVIN_NORMALIZATION_SCHEMA
        and normalization.get("dataset_tree_sha256") != dataset_manifest.tree_sha256
    ):
        raise ContractError("normalization dataset tree differs from the loaded manifest")
    if normalization["ordered_sample_keys_sha256"] != _ordered_sample_keys_sha256(
        dataset.sample_keys
    ):
        raise ContractError("normalization sample order differs from the stateful dataset")
    physical_manifest = _verified_artifact(
        repository,
        recipe.artifacts.physical_sidecar_manifest_path,
        recipe.artifacts.physical_sidecar_manifest_sha256,
    )
    physical = CalvinPhysicalSupervisionSidecar(
        _artifact_root(repository, recipe.artifacts.physical_sidecar_manifest_path),
        index,
        manifest_bytes=physical_manifest,
        verify_hashes=True,
    )
    geometry_manifest = _verified_artifact(
        repository,
        recipe.artifacts.geometry_sidecar_manifest_path,
        recipe.artifacts.geometry_sidecar_manifest_sha256,
    )
    geometry = CalvinPhysicalGeometrySidecar(
        _artifact_root(repository, recipe.artifacts.geometry_sidecar_manifest_path),
        index,
        manifest_bytes=geometry_manifest,
        verify_hashes=True,
    )
    return CalvinTrainingAssets(
        index=index,
        dataset=dataset,
        dataset_manifest=dataset_manifest,
        normalization_payload=normalization,
        physical_sidecar=physical,
        geometry_sidecar=geometry,
    )


def _ordered_sample_keys_sha256(sample_keys: tuple[str, ...]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample_key in sample_keys:
        encoded = sample_key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_molmoact2_policy_config(
    recipe: PICFTrainingRecipe,
    *,
    checkpoint_path: str | Path,
) -> object:
    """Materialize the exact pinned LeRobot config without loading weights."""

    if not isinstance(recipe, PICFTrainingRecipe):
        raise TypeError("MolmoAct2 config construction requires a PICFTrainingRecipe")
    lerobot_configs = import_module("lerobot.configs")
    molmoact2_config = import_module("lerobot.policies.molmoact2.configuration_molmoact2")
    lerobot_constants = import_module("lerobot.utils.constants")
    FeatureType = lerobot_configs.FeatureType
    PolicyFeature = lerobot_configs.PolicyFeature
    MolmoAct2Config = molmoact2_config.MolmoAct2Config
    ACTION = lerobot_constants.ACTION
    OBS_STATE = lerobot_constants.OBS_STATE

    policy = recipe.policy
    config = MolmoAct2Config(
        checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
        checkpoint_revision=recipe.host.checkpoint_revision,
        checkpoint_force_download=False,
        trust_remote_code=True,
        device="cuda",
        use_amp=False,
        use_peft=False,
        push_to_hub=False,
        n_obs_steps=policy.n_obs_steps,
        chunk_size=recipe.dataset.action_horizon,
        n_action_steps=recipe.dataset.action_horizon,
        action_mode=policy.action_mode,
        inference_action_mode=policy.inference_action_mode,
        setup_type=policy.setup_type,
        control_mode=policy.control_mode,
        image_keys=list(recipe.dataset.image_keys),
        normalize_language=policy.normalize_language,
        add_setup_tokens=policy.add_setup_tokens,
        add_control_tokens=policy.add_control_tokens,
        normalize_gripper=policy.normalize_gripper,
        num_state_tokens=policy.num_state_tokens,
        expected_max_action_dim=policy.expected_max_action_dim,
        num_flow_timesteps=policy.num_flow_timesteps,
        flow_matching_cutoff=policy.flow_matching_cutoff,
        flow_matching_time_offset=policy.flow_matching_time_offset,
        flow_matching_time_scale=policy.flow_matching_time_scale,
        flow_matching_beta_alpha=policy.flow_matching_beta_alpha,
        flow_matching_beta_beta=policy.flow_matching_beta_beta,
        num_inference_steps=policy.num_inference_steps,
        mask_action_dim_padding=policy.mask_action_dim_padding,
        enable_inference_cuda_graph=policy.enable_inference_cuda_graph,
        enable_lora_vlm=policy.enable_lora_vlm,
        enable_lora_action_expert=policy.enable_lora_action_expert,
        enable_knowledge_insulation=policy.enable_knowledge_insulation,
        freeze_embedding=policy.freeze_embedding,
        train_action_expert_only=policy.train_action_expert_only,
        gradient_checkpointing=policy.gradient_checkpointing,
        model_dtype=policy.model_dtype,
        optimizer_lr=policy.optimizer_lr,
        optimizer_action_expert_lr=policy.optimizer_action_expert_lr,
        optimizer_betas=policy.optimizer_betas,
        optimizer_eps=policy.optimizer_eps,
        optimizer_weight_decay=policy.optimizer_weight_decay,
        optimizer_grad_clip_norm=recipe.optimizer.gradient_clip_norm,
        scheduler_warmup_steps=policy.scheduler_warmup_steps,
        scheduler_decay_steps=policy.scheduler_decay_steps,
        scheduler_decay_lr=policy.scheduler_decay_lr,
        input_features={
            OBS_STATE: PolicyFeature(
                type=FeatureType.STATE,
                shape=(recipe.dataset.state_dim,),
            )
        },
        output_features={
            ACTION: PolicyFeature(
                type=FeatureType.ACTION,
                shape=(recipe.dataset.action_dim,),
            )
        },
    )
    assert_molmoact2_policy_config(recipe, config)
    return config


def assert_molmoact2_policy_config(recipe: PICFTrainingRecipe, config: object) -> None:
    """Reject runtime mutation of every policy field that changes this experiment."""

    policy = recipe.policy
    expected = {
        "action_mode": policy.action_mode,
        "add_control_tokens": policy.add_control_tokens,
        "add_setup_tokens": policy.add_setup_tokens,
        "chunk_size": recipe.dataset.action_horizon,
        "checkpoint_force_download": False,
        "checkpoint_revision": recipe.host.checkpoint_revision,
        "control_mode": policy.control_mode,
        "enable_inference_cuda_graph": policy.enable_inference_cuda_graph,
        "enable_knowledge_insulation": policy.enable_knowledge_insulation,
        "enable_lora_action_expert": policy.enable_lora_action_expert,
        "enable_lora_vlm": policy.enable_lora_vlm,
        "expected_max_action_dim": policy.expected_max_action_dim,
        "flow_matching_beta_alpha": policy.flow_matching_beta_alpha,
        "flow_matching_beta_beta": policy.flow_matching_beta_beta,
        "flow_matching_cutoff": policy.flow_matching_cutoff,
        "flow_matching_time_offset": policy.flow_matching_time_offset,
        "flow_matching_time_scale": policy.flow_matching_time_scale,
        "freeze_embedding": policy.freeze_embedding,
        "gradient_checkpointing": policy.gradient_checkpointing,
        "image_keys": list(recipe.dataset.image_keys),
        "inference_action_mode": policy.inference_action_mode,
        "mask_action_dim_padding": policy.mask_action_dim_padding,
        "model_dtype": policy.model_dtype,
        "n_action_steps": recipe.dataset.action_horizon,
        "n_obs_steps": policy.n_obs_steps,
        "normalize_gripper": policy.normalize_gripper,
        "normalize_language": policy.normalize_language,
        "num_flow_timesteps": policy.num_flow_timesteps,
        "num_inference_steps": policy.num_inference_steps,
        "num_state_tokens": policy.num_state_tokens,
        "optimizer_action_expert_lr": policy.optimizer_action_expert_lr,
        "optimizer_betas": policy.optimizer_betas,
        "optimizer_eps": policy.optimizer_eps,
        "optimizer_grad_clip_norm": recipe.optimizer.gradient_clip_norm,
        "optimizer_lr": policy.optimizer_lr,
        "optimizer_weight_decay": policy.optimizer_weight_decay,
        "push_to_hub": False,
        "scheduler_decay_lr": policy.scheduler_decay_lr,
        "scheduler_decay_steps": policy.scheduler_decay_steps,
        "scheduler_warmup_steps": policy.scheduler_warmup_steps,
        "setup_type": policy.setup_type,
        "train_action_expert_only": policy.train_action_expert_only,
        "trust_remote_code": True,
        "use_amp": False,
        "use_peft": False,
    }
    changed = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if changed:
        raise ContractError(f"MolmoAct2 policy config differs from recipe: {changed}")
    input_features = getattr(config, "input_features", {})
    output_features = getattr(config, "output_features", {})
    state = input_features.get("observation.state")
    action = output_features.get("action")
    if getattr(state, "shape", None) != (recipe.dataset.state_dim,):
        raise ContractError("MolmoAct2 state feature differs from the CALVIN recipe")
    if getattr(action, "shape", None) != (recipe.dataset.action_dim,):
        raise ContractError("MolmoAct2 action feature differs from the CALVIN recipe")


def _validate_trainable_scope(policy: nn.Module, core: nn.Module) -> None:
    policy_trainable = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    unexpected = {
        name
        for name in policy_trainable
        if "action_expert" not in name and not name.startswith("action_layer_adapter.")
    }
    if unexpected:
        raise ContractError(
            "MolmoAct2 trainable scope escaped action expert/PICF adapter: "
            f"{sorted(unexpected)[:8]}"
        )
    if not any("action_expert" in name for name in policy_trainable):
        raise ContractError("MolmoAct2 action expert is unexpectedly frozen")
    if not any(name.startswith("action_layer_adapter.") for name in policy_trainable):
        raise ContractError("PICF action-layer adapter is unexpectedly frozen")
    trainable_core = {
        name for name, parameter in core.named_parameters() if parameter.requires_grad
    }
    if trainable_core:
        raise ContractError(
            f"action-stage PICF core must be stationary; trainable={sorted(trainable_core)[:8]}"
        )


def _move_core_with_fp32_parameter_storage(
    core: PICFCore,
    host_parameter: torch.Tensor,
) -> PICFCore:
    """Place PICF on the host device without quantizing trainable parameters.

    MolmoAct2 stores its released weights and activations in bfloat16, but PICF
    is trained from scratch at a 1e-4 learning rate.  Storing those parameters
    in bfloat16 can make an AdamW update smaller than one representable unit and
    permanently freeze initialized lifecycle biases.  Accelerator autocast
    still evaluates the forward in bfloat16 and the typed posterior/action
    boundaries restore the activation dtype; only parameter and optimizer-state
    storage remain float32.
    """

    core.to(device=host_parameter.device, dtype=torch.float32)
    unexpected = {
        (name, parameter.dtype)
        for name, parameter in core.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    }
    if unexpected:
        raise ContractError(f"PICF parameter storage escaped float32: {sorted(unexpected)}")
    return core


def _validate_action_only_recipe(recipe: PICFTrainingRecipe) -> None:
    objective = recipe.objective_config
    if (
        objective.action_weight <= 0.0
        or objective.set_weight != 0.0
        or objective.dynamics_weight != 0.0
        or objective.binding_weight != 0.0
        or objective.require_temporal_positive_pairs
        or recipe.geometry_overshooting.config.weight != 0.0
    ):
        raise ContractError(
            "stationary-core action learning requires an explicit action-only objective"
        )


def build_molmoact2_calvin_training_stack(
    recipe: PICFTrainingRecipe,
    *,
    policy: nn.Module,
    assets: CalvinTrainingAssets,
    accepted_temporal_core: AcceptedStationaryTemporalCore,
    build_native_banks: CalvinStatefulNativeBankBuilder | None = None,
    native_evidence_history_frames: int = 1,
    action_context_token_dims: Mapping[str, int] | None = None,
    include_posterior_action_context: bool = True,
) -> MolmoAct2CalvinTrainingStack:
    """Install PICF and close the one-transition target-separated training graph."""

    if not isinstance(policy, nn.Module):
        raise TypeError("MolmoAct2 training stack requires a loaded policy module")
    if recipe.authorization.stage not in _ACTION_TRAINING_STAGES:
        raise ContractError(
            "stateful action assembly requires an M4+ recipe and an accepted temporal core"
        )
    if not isinstance(accepted_temporal_core, AcceptedStationaryTemporalCore):
        raise TypeError("action assembly requires one validated stationary temporal core")
    _validate_action_only_recipe(recipe)
    assert_molmoact2_policy_config(recipe, getattr(policy, "config", None))
    get_expert = getattr(policy, "_action_expert", None)
    if not callable(get_expert):
        raise TypeError("loaded policy is not the pinned MolmoAct2 LeRobot host")
    host_parameter = next(policy.parameters())
    # Keep every scratch-trained PICF parameter in float32. The surrounding
    # Accelerate autocast controls activation compute independently.
    core = _move_core_with_fp32_parameter_storage(recipe.build_core(), host_parameter)
    objective = recipe.build_objective()
    loaded_provenance = load_stationary_temporal_checkpoint(
        core,
        objective,
        accepted_temporal_core.checkpoint_path,
        expected_sha256=accepted_temporal_core.checkpoint_sha256,
        expected_provenance=accepted_temporal_core.provenance,
    )
    if loaded_provenance != accepted_temporal_core.provenance:  # pragma: no cover
        raise RuntimeError("loaded stationary provenance changed after strict validation")
    core.requires_grad_(False)
    objective.requires_grad_(False)
    context_dims = dict(action_context_token_dims or {})
    duplicated_modalities = sorted(set(context_dims) & set(recipe.core_config.dense_token_dims))
    if duplicated_modalities:
        raise ContractError(
            f"action-only context duplicates posterior modalities: {duplicated_modalities}"
        )
    dense_token_dims = {**recipe.core_config.dense_token_dims, **context_dims}
    adapter = MolmoAct2PICFActionExpert(
        get_expert(),
        dense_token_dims=dense_token_dims,
        object_address_dim=recipe.core_config.object_address_dim,
        object_value_dim=recipe.core_config.object_value_dim,
        validate_tensor_values=recipe.core_config.runtime_validation == "full",
    )
    if not include_posterior_action_context:
        adapter.set_posterior_action_context_trainable(False)
    install_molmoact2_lerobot_picf_adapter(policy, adapter)
    sequence = MolmoAct2PICFTrainingBridge(
        policy,
        core,
        replace(
            recipe.build_host_training_config(),
            include_posterior_action_context=include_posterior_action_context,
        ),
    )
    joint = MolmoAct2PICFJointTrainingBridge(sequence, objective)
    stats = official_molmoact2_dataset_stats(assets.normalization_payload)
    processor = CalvinMolmoAct2ProcessorBridge.from_official_config(
        policy.config,
        dataset_stats=stats,
    )
    module = CalvinStatefulMolmoAct2TrainingModule(
        assets.dataset,
        joint,
        build_host_batch=processor.build_action_targets,
        build_host_observation_inputs=processor.build_observation_inputs,
        build_native_banks=build_native_banks,
        build_loss_targets=None,
        native_evidence_history_frames=native_evidence_history_frames,
    )
    _validate_trainable_scope(policy, core)
    module.get_optim_params()
    return MolmoAct2CalvinTrainingStack(
        policy_config=policy.config,
        processor=processor,
        assets=assets,
        module=module,
        accepted_temporal_core=accepted_temporal_core,
    )


def build_molmoact2_optimizer_and_scheduler(
    recipe: PICFTrainingRecipe,
    stack: MolmoAct2CalvinTrainingStack,
) -> tuple[torch.optim.Optimizer, Any]:
    """Use the pinned official optimizer/scheduler builders on complete groups."""

    config = stack.policy_config
    assert_molmoact2_policy_config(recipe, config)
    get_optimizer_preset = getattr(config, "get_optimizer_preset", None)
    if not callable(get_optimizer_preset):
        raise TypeError("MolmoAct2 policy config has no optimizer preset builder")
    optimizer_config: Any = get_optimizer_preset()
    if float(optimizer_config.grad_clip_norm) != recipe.optimizer.gradient_clip_norm:
        raise ContractError("official optimizer clipping differs from the recipe")
    optimizer = optimizer_config.build(stack.module.get_optim_params())
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("MolmoAct2 optimizer preset must build one optimizer")
    get_scheduler_preset = getattr(config, "get_scheduler_preset", None)
    if not callable(get_scheduler_preset):
        raise TypeError("MolmoAct2 policy config has no scheduler preset builder")
    scheduler_config: Any = get_scheduler_preset()
    if scheduler_config is None:
        raise ContractError("MolmoAct2 scheduler preset is absent")
    scheduler = scheduler_config.build(
        optimizer,
        num_training_steps=recipe.authorization.max_optimizer_steps,
    )
    if scheduler is None:
        raise ContractError("MolmoAct2 scheduler preset did not build a scheduler")
    return optimizer, scheduler


def build_calvin_episode_stream_plan(
    recipe: PICFTrainingRecipe,
    dataset: CalvinStatefulTransitionDataset,
    *,
    comparison_id: str,
    seed: int,
    global_batch_size: int,
    total_steps: int,
) -> FrozenEpisodeStreamPlan:
    recipe.assert_optimizer_steps_authorized(total_steps)
    if dataset.action_horizon != recipe.dataset.action_horizon:
        raise ContractError("CALVIN stateful dataset horizon differs from the recipe")
    episodes = tuple(
        EpisodeSampleSequence(
            episode_key=episode.episode_key,
            sample_keys=episode.sample_keys,
        )
        for episode in dataset.episode_manifest
    )
    return FrozenEpisodeStreamPlan(
        dataset_id=recipe.dataset.dataset_id,
        dataset_revision=recipe.dataset.dataset_revision,
        dataset_manifest_sha256=recipe.artifacts.dataset_tree_sha256,
        episodes=episodes,
        comparison_id=comparison_id,
        seed=seed,
        global_batch_size=global_batch_size,
        total_steps=total_steps,
    )
