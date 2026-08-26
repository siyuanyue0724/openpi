from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from picf_next.lingbot_wla_install import (
    WLA_ACTION_BLOCK_CLASS,
    WLA_ACTION_FSDP_PARAMETER_PREFIX,
    WLA_HOST_TEXT_BLOCK_CLASS,
    WLA_HOST_TEXT_FSDP_PARAMETER_PREFIX,
    WLA_WORLD_BLOCK_CLASS,
    WLA_WORLD_FSDP_PARAMETER_PREFIX,
    LingBotWLARootOutput,
)


def test_wla_root_output_exposes_every_differentiable_tensor_to_fsdp() -> None:
    total = torch.tensor(1.0, requires_grad=True)
    action = torch.tensor(2.0, requires_grad=True)
    world = torch.tensor(3.0, requires_grad=True)
    native = (
        torch.ones(2, requires_grad=True),
        torch.ones(3, requires_grad=True),
    )

    output = LingBotWLARootOutput(
        total_loss=total,
        action_loss=action,
        world_loss=world,
        native_root_outputs=native,
    )
    flat, _ = tree_flatten(output)

    assert output.total_loss is total
    assert tuple(value for value in flat if isinstance(value, torch.Tensor)) == (
        total,
        action,
        world,
        *native,
    )


def test_wla_selective_offload_contract_targets_only_exact_transformer_blocks() -> None:
    assert WLA_HOST_TEXT_BLOCK_CLASS == "Qwen3VLTextDecoderLayer"
    assert WLA_ACTION_BLOCK_CLASS == "BasicTransformerBlock"
    assert WLA_WORLD_BLOCK_CLASS == "SanaTransformerBlock"
    assert WLA_HOST_TEXT_FSDP_PARAMETER_PREFIX == (
        "model.qwenvl_with_expert.qwenvl.model.language_model.layers"
    )
    assert WLA_ACTION_FSDP_PARAMETER_PREFIX == (
        "picf_wla_action_interface.action_head.model.transformer_blocks"
    )
    assert WLA_WORLD_FSDP_PARAMETER_PREFIX == (
        "picf_wla_world_expert.world_expert.transformer_blocks"
    )
    assert "connector" not in WLA_ACTION_FSDP_PARAMETER_PREFIX
    assert "connector" not in WLA_WORLD_FSDP_PARAMETER_PREFIX
    assert "connector" not in WLA_HOST_TEXT_FSDP_PARAMETER_PREFIX
