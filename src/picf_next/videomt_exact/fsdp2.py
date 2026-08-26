"""Function-preserving FSDP2 and rematerialization for complete VidEoMT."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

VIDEOMT_DINOV3_L_BLOCKS = 24
VIDEOMT_COMPLETE_TRAINABLE_PARAMETERS = 315_986_985
VIDEOMT_FSDP2_EXECUTION_UNITS_PER_BLOCK = 2


def _source_blocks(model: nn.Module) -> tuple[nn.Module, ...]:
    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("complete VidEoMT source lacks its DINO block ModuleList")
    result = tuple(blocks)
    if len(result) != VIDEOMT_DINOV3_L_BLOCKS or len({id(block) for block in result}) != len(
        result
    ):
        raise ValueError("complete VidEoMT DINO block inventory drifted")
    return result


def _source_execution_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    """Return modules that the released manual block path actually calls.

    VidEoMT executes each DINO block through ``model._block_forward`` and calls
    its attention and MLP children directly.  Wrapping the parent block would
    therefore bypass both checkpoint and FSDP hooks.
    """

    attention_name = "attention" if bool(getattr(model, "is_v3", False)) else "attn"
    result: list[nn.Module] = []
    for index, block in enumerate(_source_blocks(model)):
        attention = getattr(block, attention_name, None)
        mlp = getattr(block, "mlp", None)
        if not isinstance(attention, nn.Module) or not isinstance(mlp, nn.Module):
            raise TypeError(
                f"complete VidEoMT block {index} lacks its called attention/MLP modules"
            )
        result.extend((attention, mlp))
    expected = VIDEOMT_DINOV3_L_BLOCKS * VIDEOMT_FSDP2_EXECUTION_UNITS_PER_BLOCK
    if len(result) != expected or len({id(module) for module in result}) != expected:
        raise ValueError("complete VidEoMT execution-module inventory drifted")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VidEoMTFSDP2Receipt:
    block_count: int
    checkpointed_block_count: int
    sharded_module_count: int
    parameter_tensor_count: int
    parameter_numel: int
    trainable_parameter_numel: int
    parameter_dtype: str
    reduction_dtype: str
    output_dtype: str
    cpu_offload: bool

    def __post_init__(self) -> None:
        if (
            self.block_count != VIDEOMT_DINOV3_L_BLOCKS
            or self.checkpointed_block_count != self.block_count
            or self.sharded_module_count
            != self.block_count * VIDEOMT_FSDP2_EXECUTION_UNITS_PER_BLOCK + 1
        ):
            raise ValueError("VidEoMT FSDP2 block coverage is incomplete")
        if self.parameter_numel != VIDEOMT_COMPLETE_TRAINABLE_PARAMETERS:
            raise ValueError("VidEoMT FSDP2 parameter inventory drifted")
        if self.trainable_parameter_numel != self.parameter_numel:
            raise ValueError("VidEoMT FSDP2 unexpectedly froze source parameters")


def apply_exact_videomt_activation_checkpointing(model: nn.Module) -> tuple[nn.Module, ...]:
    """Checkpoint both execution modules reached by every released DINO block."""

    blocks = _source_blocks(model)
    identities = {id(module) for module in _source_execution_modules(model)}
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=lambda module: id(module) in identities,
    )
    wrapped = _source_blocks(model)
    wrapped_execution_modules = _source_execution_modules(model)
    if any(
        not hasattr(module, "_checkpoint_wrapped_module")
        for module in wrapped_execution_modules
    ):
        raise RuntimeError("VidEoMT activation checkpointing omitted a called block child")
    for block in wrapped:
        attention_name = "attention" if bool(getattr(model, "is_v3", False)) else "attn"
        if not hasattr(getattr(block, attention_name), "_checkpoint_wrapped_module") or not hasattr(
            block.mlp, "_checkpoint_wrapped_module"
        ):
            raise RuntimeError("VidEoMT activation checkpointing left one block incomplete")
    return wrapped


def parallelize_exact_videomt_fsdp2(
    model: nn.Module,
    *,
    parameter_dtype: torch.dtype = torch.bfloat16,
    reduction_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
) -> tuple[nn.Module, VidEoMTFSDP2Receipt]:
    """Shard the complete source after wrapping all expensive transformer blocks."""

    if not isinstance(model, nn.Module):
        raise TypeError("VidEoMT FSDP2 requires one source module")
    if parameter_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("VidEoMT FSDP2 parameter dtype is unsupported")
    if reduction_dtype not in (torch.float32, torch.float64):
        raise ValueError("VidEoMT FSDP2 reductions require wide floating precision")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("VidEoMT FSDP2 output dtype is unsupported")
    parameters = tuple(model.parameters())
    parameter_numel = sum(parameter.numel() for parameter in parameters)
    trainable_numel = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    if parameter_numel != VIDEOMT_COMPLETE_TRAINABLE_PARAMETERS:
        raise ValueError("complete VidEoMT source parameter inventory drifted before FSDP2")
    if trainable_numel != parameter_numel:
        raise ValueError("complete VidEoMT source must remain fully trainable")

    blocks = apply_exact_videomt_activation_checkpointing(model)
    execution_modules = _source_execution_modules(model)
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=parameter_dtype,
        reduce_dtype=reduction_dtype,
        output_dtype=output_dtype,
    )
    offload_policy = CPUOffloadPolicy() if cpu_offload else None
    shard_kwargs = {"mp_policy": mixed_precision}
    if offload_policy is not None:
        shard_kwargs["offload_policy"] = offload_policy
    for module in execution_modules:
        fully_shard(module, **shard_kwargs)
    fully_shard(model, **shard_kwargs)
    receipt = VidEoMTFSDP2Receipt(
        block_count=len(blocks),
        checkpointed_block_count=sum(
            all(
                hasattr(module, "_checkpoint_wrapped_module")
                for module in (
                    getattr(
                        block,
                        "attention" if bool(getattr(model, "is_v3", False)) else "attn",
                    ),
                    block.mlp,
                )
            )
            for block in blocks
        ),
        sharded_module_count=len(execution_modules) + 1,
        parameter_tensor_count=len(parameters),
        parameter_numel=parameter_numel,
        trainable_parameter_numel=trainable_numel,
        parameter_dtype=str(parameter_dtype),
        reduction_dtype=str(reduction_dtype),
        output_dtype=str(output_dtype),
        cpu_offload=cpu_offload,
    )
    return model, receipt
