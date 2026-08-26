from __future__ import annotations

import math
from pathlib import Path
from types import MethodType
from typing import Any, Final, Mapping, NamedTuple

import torch
from torch import nn

from picf_next.lingbot_wla_calvin import run_lingbot_wla_full_calvin_objective
from picf_next.lingbot_wla_shared import (
    LingBotWLASharedInterface,
    predict_lingbot_wla_calvin_actions,
    run_lingbot_wla_calvin_forward,
)
from picf_next.lingbot_wla_randomness import paired_wla_inference_rng, paired_wla_rng
from picf_next.lingbot_wla_world import LingBotWLAWorldExpert


WLA_BACKEND_IDENTITY: Final = "wla-155ac94e-complete-action-world-on-lingbot"
WLA_ACTION_BLOCK_CLASS: Final = "BasicTransformerBlock"
WLA_WORLD_BLOCK_CLASS: Final = "SanaTransformerBlock"
WLA_CONNECTOR_BLOCK_CLASS: Final = "Qwen2EncoderLayer"
WLA_HOST_TEXT_BLOCK_CLASS: Final = "Qwen3VLTextDecoderLayer"
WLA_HOST_TEXT_FSDP_PARAMETER_PREFIX: Final = (
    "model.qwenvl_with_expert.qwenvl.model.language_model.layers"
)
WLA_ACTION_FSDP_PARAMETER_PREFIX: Final = (
    "picf_wla_action_interface.action_head.model.transformer_blocks"
)
WLA_WORLD_FSDP_PARAMETER_PREFIX: Final = (
    "picf_wla_world_expert.world_expert.transformer_blocks"
)
WLA_OFFICIAL_LEARNING_RATE: Final = 5.0e-5
WLA_OFFICIAL_MINIMUM_LEARNING_RATE: Final = 5.0e-6
WLA_OFFICIAL_WARMUP_STEPS: Final = 1000
WLA_OFFICIAL_SCHEDULE_STEPS: Final = 100000
WLA_OFFICIAL_BETAS: Final = (0.9, 0.95)
WLA_OFFICIAL_EPS: Final = 1.0e-8
WLA_OFFICIAL_WEIGHT_DECAY: Final = 1.0e-8
WLA_OFFICIAL_GRADIENT_CLIP_NORM: Final = 1.0

_RELEASED_ACTION_FLOW_COMPONENTS: Final = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


class LingBotWLARootOutput(NamedTuple):
    """FSDP-visible root result for the registered WLA forward method.

    Composable FSDP discovers tensors with ``torch.utils._pytree.tree_flatten``
    before it reshards root-owned parameters.  A plain dataclass is an opaque
    pytree leaf, while a named tuple preserves the typed attribute ABI and
    exposes every differentiable result to FSDP's pre-backward hooks.
    """

    total_loss: torch.Tensor
    action_loss: torch.Tensor
    world_loss: torch.Tensor | None
    native_root_outputs: tuple[torch.Tensor, ...]


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _wla_modules(policy: nn.Module) -> tuple[LingBotWLASharedInterface, LingBotWLAWorldExpert]:
    action = getattr(policy, "picf_wla_action_interface", None)
    world = getattr(policy, "picf_wla_world_expert", None)
    if not isinstance(action, LingBotWLASharedInterface) or not isinstance(
        world,
        LingBotWLAWorldExpert,
    ):
        raise RuntimeError("complete LingBot WLA backend is not installed")
    return action, world


def lingbot_wla_is_installed(policy: nn.Module) -> bool:
    try:
        _wla_modules(policy)
    except RuntimeError:
        return False
    return callable(getattr(policy, "picf_wla_calvin_forward", None))


def _picf_wla_calvin_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    picf_native_context: Any,
    target_images: torch.Tensor | None,
    require_world: bool,
) -> LingBotWLARootOutput:
    if not isinstance(require_world, bool):
        raise TypeError("WLA world-objective requirement must be boolean")
    if require_world != (target_images is not None):
        raise ValueError("WLA factual training requires exactly one target-image batch")
    action_interface, world_expert = _wla_modules(policy)
    inputs = dict(model_inputs)
    action_device = next(action_interface.action_head.parameters()).device
    with paired_wla_rng(inputs, device=action_device):
        if require_world:
            if target_images is None:
                raise AssertionError("validated WLA world objective lost its targets")
            result = run_lingbot_wla_full_calvin_objective(
                policy,
                action_interface,
                world_expert,
                model_inputs=inputs,
                picf_native_context=picf_native_context,
                target_images=target_images,
            )
            return LingBotWLARootOutput(
                total_loss=result.loss,
                action_loss=result.action.loss,
                world_loss=result.world.loss,
                native_root_outputs=result.native_root_outputs,
            )
        result = run_lingbot_wla_calvin_forward(
            policy,
            action_interface,
            model_inputs=inputs,
            picf_native_context=picf_native_context,
        )
        return LingBotWLARootOutput(
            total_loss=result.action.loss,
            action_loss=result.action.loss,
            world_loss=None,
            native_root_outputs=result.native_root_outputs,
        )


def _picf_wla_prior_forward(
    policy: nn.Module,
    *,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    inputs_embeds: list[torch.Tensor | None],
    visual_pos_masks: torch.Tensor | None,
    picf_native_context: Any,
) -> tuple[list[torch.Tensor | None], None, list[torch.Tensor]]:
    """Own recurrent PICF propagation with WLA's shared LingBot host."""

    action_interface, _ = _wla_modules(policy)
    flow = getattr(policy, "model", None)
    joint = getattr(flow, "qwenvl_with_expert", None)
    if not isinstance(joint, nn.Module):
        raise TypeError("WLA prior rollout cannot find LingBot's shared host")
    if hasattr(joint, "qwen_expert"):
        raise RuntimeError("WLA prior rollout must not revive LingBot's replaced action expert")
    return action_interface.run_prior_rollout(
        joint,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        visual_pos_masks=visual_pos_masks,
        picf_native_context=picf_native_context,
    )


def _picf_wla_sample_actions(
    policy: nn.Module,
    images: torch.Tensor,
    img_masks: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    picf_native_context: Any = None,
) -> torch.Tensor:
    """Expose WLA's exact ``predict_action`` through LingBot's deployment ABI."""

    if image_grid_thw is None:
        raise ValueError("complete WLA inference requires LingBot image_grid_thw")
    action_interface, _ = _wla_modules(policy)
    model_inputs = {
        "images": images,
        "img_masks": img_masks,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "state": state,
        "image_grid_thw": image_grid_thw,
    }
    action_device = next(action_interface.action_head.parameters()).device
    if noise is None:
        return predict_lingbot_wla_calvin_actions(
            policy,
            action_interface,
            model_inputs=model_inputs,
            picf_native_context=picf_native_context,
        )
    with paired_wla_inference_rng(noise, device=action_device):
        return predict_lingbot_wla_calvin_actions(
            policy,
            action_interface,
            model_inputs=model_inputs,
            picf_native_context=picf_native_context,
        )


def _released_action_receipt(policy: nn.Module) -> dict[str, Any]:
    flow = getattr(policy, "model", None)
    joint = getattr(flow, "qwenvl_with_expert", None)
    if not isinstance(flow, nn.Module) or not isinstance(joint, nn.Module):
        raise TypeError("WLA installation requires the released LingBot policy topology")
    expert = getattr(joint, "qwen_expert", None)
    if not isinstance(expert, nn.Module):
        raise RuntimeError("released LingBot action expert is absent before WLA replacement")
    components: dict[str, int] = {"qwen_expert": _module_parameter_count(expert)}
    for name in _RELEASED_ACTION_FLOW_COMPONENTS:
        module = getattr(flow, name, None)
        if not isinstance(module, nn.Module):
            raise RuntimeError(f"released LingBot action component is absent: {name}")
        components[name] = _module_parameter_count(module)
    return {
        "components": components,
        "parameter_count": sum(components.values()),
    }


def _remove_released_action(policy: nn.Module) -> dict[str, Any]:
    receipt = _released_action_receipt(policy)
    flow = policy.model
    joint = flow.qwenvl_with_expert
    delattr(joint, "qwen_expert")
    for name in _RELEASED_ACTION_FLOW_COMPONENTS:
        delattr(flow, name)
    if hasattr(joint, "qwen_expert") or any(
        hasattr(flow, name) for name in _RELEASED_ACTION_FLOW_COMPONENTS
    ):
        raise RuntimeError("released LingBot action replacement was incomplete")
    return receipt


def register_lingbot_wla_fsdp_units(policy: nn.Module) -> tuple[str, ...]:
    action, world = _wla_modules(policy)
    action_blocks = tuple(action.action_head.model.transformer_blocks)
    world_blocks = tuple(world.world_expert.transformer_blocks)
    connector_layers = tuple(world.connector[0].layers)
    if len(action_blocks) != 28 or {type(value).__name__ for value in action_blocks} != {
        WLA_ACTION_BLOCK_CLASS
    }:
        raise RuntimeError("complete WLA action-block topology changed")
    if len(world_blocks) != 28 or {type(value).__name__ for value in world_blocks} != {
        WLA_WORLD_BLOCK_CLASS
    }:
        raise RuntimeError("complete WLA SANA-block topology changed")
    if len(connector_layers) != 1 or type(connector_layers[0]).__name__ != (
        WLA_CONNECTOR_BLOCK_CLASS
    ):
        raise RuntimeError("complete WLA connector topology changed")
    existing = tuple(getattr(policy, "_no_split_modules", ()))
    if not existing:
        raise RuntimeError("LingBot policy exposes no native FSDP unit classes")
    units = tuple(
        dict.fromkeys(
            (
                *existing,
                WLA_ACTION_BLOCK_CLASS,
                WLA_WORLD_BLOCK_CLASS,
                WLA_CONNECTOR_BLOCK_CLASS,
            )
        )
    )
    policy._no_split_modules = list(units)
    return units


def install_lingbot_wla_backend(
    policy: nn.Module,
    *,
    source_root: Path | str,
    pretrained_root: Path | str,
    tokenizer: Any,
    chunk_size: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    if lingbot_wla_is_installed(policy) or hasattr(policy, "picf_wla_action_interface"):
        raise RuntimeError("complete LingBot WLA backend may be installed only once")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("LingBot WLA chunk size must be positive")
    flow = getattr(policy, "model", None)
    joint = getattr(flow, "qwenvl_with_expert", None)
    if not isinstance(flow, nn.Module) or not isinstance(joint, nn.Module):
        raise TypeError("complete WLA backend requires the released LingBot policy")
    host_width = int(getattr(joint.qwenvl.config.text_config, "hidden_size", -1))
    if host_width != 2560:
        raise ValueError("complete WLA integration is pinned to LingBot's 2560-wide host")

    action = LingBotWLASharedInterface.from_pinned_source(
        source_root,
        host_width=host_width,
        max_action_dim=55,
        max_state_dim=55,
        chunk_size=chunk_size,
        device=device,
        dtype=dtype,
    )
    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    newline_ids = base_tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise RuntimeError(f"WLA newline is not one LingBot token: {newline_ids}")
    im_end_token_id = int(base_tokenizer.convert_tokens_to_ids("<|im_end|>"))
    action.initialize_meta_tokens_from_lingbot(
        joint,
        newline_token_id=int(newline_ids[0]),
        im_end_token_id=im_end_token_id,
    )
    world = LingBotWLAWorldExpert.from_pinned_source(
        source_root,
        pretrained_root,
        host_width=host_width,
        world_device=device,
        vae_device=device,
    )
    action.train()
    world.train()

    released_action = _remove_released_action(policy)
    policy.add_module("picf_wla_action_interface", action)
    policy.add_module("picf_wla_world_expert", world)
    policy.picf_native_prior_forward = MethodType(_picf_wla_prior_forward, policy)
    policy.picf_wla_calvin_forward = MethodType(_picf_wla_calvin_forward, policy)
    policy.sample_actions = MethodType(_picf_wla_sample_actions, policy)
    fsdp_units = register_lingbot_wla_fsdp_units(policy)
    receipt = {
        "schema": "picf-next.adr224-lingbot-wla-installation.v1",
        "backend_identity": WLA_BACKEND_IDENTITY,
        "source_commit": action.source.commit,
        "source_files": dict(action.source.files),
        "source_tree_file_count": action.source.tree_file_count,
        "source_tree_sha256": action.source.tree_sha256,
        "source_tree_receipt_sha256": action.source.tree_receipt_sha256,
        "released_action_removed": released_action,
        "host_width": host_width,
        "chunk_size": chunk_size,
        "max_action_dim": 55,
        "max_state_dim": 55,
        "action_layers": len(action.action_head.model.transformer_blocks),
        "world_layers": len(world.world_expert.transformer_blocks),
        "connector_layers": len(world.connector[0].layers),
        "parameter_counts": {
            "action_interface": _module_parameter_count(action),
            "world_expert_with_frozen_vae": _module_parameter_count(world),
            "frozen_vae": _module_parameter_count(world.vae),
        },
        "frozen_vae_trainable_parameters": sum(
            value.numel() for value in world.vae.parameters() if value.requires_grad
        ),
        "fsdp_unit_classes": list(fsdp_units),
        "adaptations": [
            "RynnBrain-2048 host replaced by complete LingBot/PICF-2560 host",
            "WLA action/state dimensions expanded to LingBot CALVIN's masked 55D ABI",
            "explicit WLA single t-8 history observation replaced by PICF's causal posterior memory",
            "PICF prior propagation moved from the replaced LingBot action expert to the shared LingBot/WLA host",
        ],
        "action_inference": "upstream ActionHead.predict_action, exact 32-step Euler sampler",
    }
    if receipt["frozen_vae_trainable_parameters"] != 0:
        raise RuntimeError("complete WLA installation unfroze its source VAE")
    policy._picf_wla_installation_receipt = receipt
    return receipt


def register_lingbot_wla_fsdp_forward(policy: nn.Module) -> dict[str, Any]:
    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    if not isinstance(policy, FSDPModule) or not lingbot_wla_is_installed(policy):
        raise RuntimeError("WLA root method registration requires the installed FSDP2 policy")
    register_fsdp_forward_method(policy, "picf_wla_calvin_forward")
    register_fsdp_forward_method(policy, "sample_actions")
    receipt = {
        "schema": "picf-next.adr224-wla-fsdp-forward.v1",
        "registered_root_methods": ["picf_wla_calvin_forward", "sample_actions"],
    }
    policy._picf_wla_fsdp_forward_receipt = receipt
    return receipt


def audit_lingbot_wla_fsdp_topology(policy: nn.Module) -> dict[str, Any]:
    """Verify every module whose internal block is called directly is sharded."""

    from torch.distributed.fsdp import FSDPModule

    if not isinstance(policy, FSDPModule):
        raise RuntimeError("complete WLA policy root is not FSDP2")
    action, world = _wla_modules(policy)
    groups = {
        "action": tuple(action.action_head.model.transformer_blocks),
        "world": tuple(world.world_expert.transformer_blocks),
        "connector": tuple(world.connector[0].layers),
    }
    expected = {"action": 28, "world": 28, "connector": 1}
    for name, modules in groups.items():
        if len(modules) != expected[name] or any(
            not isinstance(module, FSDPModule) for module in modules
        ):
            raise RuntimeError(f"complete WLA {name} FSDP2 topology differs")
    unit_classes = tuple(getattr(policy, "_no_split_modules", ()))
    required_unit_classes = {
        WLA_ACTION_BLOCK_CLASS,
        WLA_WORLD_BLOCK_CLASS,
        WLA_CONNECTOR_BLOCK_CLASS,
    }
    if not required_unit_classes <= set(unit_classes):
        raise RuntimeError("complete WLA pre-FSDP unit registration was not preserved")
    return {
        "schema": "picf-next.adr224-wla-fsdp2-topology.v1",
        "official_distributed_backend": "DeepSpeed-ZeRO-1",
        "integration_distributed_backend": "LingBot-FSDP2",
        "adaptation": True,
        "root_sharded": True,
        "action_block_count": len(groups["action"]),
        "world_block_count": len(groups["world"]),
        "connector_block_count": len(groups["connector"]),
        # Composable FSDP mutates each registered module into a dynamic FSDP
        # class.  Re-running the pre-wrap concrete-class registrar here would
        # reject the very wrappers audited above.  Report the frozen pre-wrap
        # registration instead of mutating topology during post-wrap audit.
        "unit_classes": list(unit_classes),
    }


def _wla_trainable_named_parameters(
    policy: nn.Module,
) -> tuple[tuple[str, nn.Parameter], ...]:
    _, world = _wla_modules(policy)
    values = tuple(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    identities = tuple(id(parameter) for _, parameter in values)
    if not values or len(identities) != len(set(identities)):
        raise RuntimeError("WLA policy has no trainable parameters or shares them unexpectedly")
    vae_ids = {id(parameter) for parameter in world.vae.parameters()}
    if any(id(parameter) in vae_ids for _, parameter in values):
        raise RuntimeError("frozen WLA VAE reached the trainable parameter set")
    return values


def build_lingbot_wla_optimizer(policy: nn.Module) -> torch.optim.AdamW:
    """Reproduce Transformers 4.57.1 ``Trainer.create_optimizer`` for WLA.

    Upstream WLA trains its language/vision host, connector, world expert, and
    action expert with one AdamW.  Splitting the donor from the LingBot/PICF
    host would change that optimization system and is therefore forbidden.
    """

    from transformers.trainer_pt_utils import get_parameter_names

    named = _wla_trainable_named_parameters(policy)
    forbidden_name_patterns = [
        r"bias",
        r"layernorm",
        r"rmsnorm",
        r"(?:^|\.)norm(?:$|\.)",
        r"_norm(?:$|\.)",
    ]
    decay_names = set(
        get_parameter_names(policy, [nn.LayerNorm], forbidden_name_patterns)
    )
    decay = [parameter for name, parameter in named if name in decay_names]
    no_decay = [parameter for name, parameter in named if name not in decay_names]
    if not decay or not no_decay:
        raise RuntimeError("complete WLA optimizer requires nonempty decay and no-decay groups")
    return torch.optim.AdamW(
        (
            {"params": decay, "weight_decay": WLA_OFFICIAL_WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ),
        lr=WLA_OFFICIAL_LEARNING_RATE,
        betas=WLA_OFFICIAL_BETAS,
        eps=WLA_OFFICIAL_EPS,
    )


def build_lingbot_wla_scheduler(
    optimizer: torch.optim.AdamW,
) -> torch.optim.lr_scheduler.LRScheduler:
    if type(optimizer).__name__ != "AdamW":
        raise TypeError("complete WLA schedule requires its single AdamW")
    from transformers import get_scheduler

    return get_scheduler(
        "cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=WLA_OFFICIAL_WARMUP_STEPS,
        num_training_steps=WLA_OFFICIAL_SCHEDULE_STEPS,
        scheduler_specific_kwargs={"min_lr": WLA_OFFICIAL_MINIMUM_LEARNING_RATE},
    )


def audit_lingbot_wla_optimizer(
    policy: nn.Module,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    parameters = _wla_trainable_named_parameters(policy)
    expected_ids = {id(parameter) for _, parameter in parameters}
    owners = {key: 0 for key in expected_ids}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in owners:
                raise RuntimeError("WLA AdamW owns a frozen or foreign parameter")
            owners[id(parameter)] += 1
    invalid = {key: value for key, value in owners.items() if value != 1}
    if invalid or scheduler.optimizer is not optimizer:
        raise RuntimeError("complete WLA optimizer or scheduler ownership is invalid")
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("complete WLA AdamW must preserve two decay groups")
    group = optimizer.param_groups[0]
    observed = {
        # ``get_scheduler`` applies the warmup lambda at construction, so the
        # live step-0 LR is correctly zero.  The immutable optimizer contract
        # lives in ``defaults``/``initial_lr`` rather than the scheduled value.
        "learning_rate": float(optimizer.defaults["lr"]),
        "betas": tuple(float(value) for value in group["betas"]),
        "eps": float(group["eps"]),
        "weight_decay": tuple(
            float(value["weight_decay"]) for value in optimizer.param_groups
        ),
    }
    expected = {
        "learning_rate": WLA_OFFICIAL_LEARNING_RATE,
        "betas": WLA_OFFICIAL_BETAS,
        "eps": WLA_OFFICIAL_EPS,
        "weight_decay": (WLA_OFFICIAL_WEIGHT_DECAY, 0.0),
    }
    initial_learning_rates = tuple(
        float(value.get("initial_lr", float("nan"))) for value in optimizer.param_groups
    )
    current_learning_rate = float(scheduler.get_last_lr()[0])
    if (
        observed != expected
        or initial_learning_rates != (WLA_OFFICIAL_LEARNING_RATE,) * 2
        or not math.isfinite(current_learning_rate)
        or not 0.0 <= current_learning_rate <= WLA_OFFICIAL_LEARNING_RATE
    ):
        raise RuntimeError("complete WLA optimizer hyperparameters differ from upstream")
    return {
        "schema": "picf-next.adr224-wla-optimizer-scheduler.v1",
        "trainable_parameter_tensors": len(parameters),
        "trainable_parameter_elements": sum(
            parameter.numel() for _, parameter in parameters
        ),
        "optimizer": "single torch.optim.AdamW via Transformers-4.57.1 Trainer contract",
        "optimizer_hyperparameters": expected,
        "scheduler": "transformers.get_scheduler(cosine_with_min_lr)",
        "warmup_steps": WLA_OFFICIAL_WARMUP_STEPS,
        "schedule_steps": WLA_OFFICIAL_SCHEDULE_STEPS,
        "minimum_learning_rate": WLA_OFFICIAL_MINIMUM_LEARNING_RATE,
        "gradient_clip_norm": WLA_OFFICIAL_GRADIENT_CLIP_NORM,
        "current_learning_rate": current_learning_rate,
    }
