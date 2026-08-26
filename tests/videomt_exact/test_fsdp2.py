from __future__ import annotations

import pytest
import torch
from torch import nn

import picf_next.videomt_exact.fsdp2 as source_fsdp2


class _Backbone(nn.Module):
    def __init__(self, *, blocks: int = 24) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(blocks))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = value + block.attention(block.norm1(value))
            value = value + block.mlp(block.norm2(value))
        return value


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(4)
        self.attention = nn.Linear(4, 4)
        self.norm2 = nn.LayerNorm(4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, _value: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the released VidEoMT path bypasses block.forward")


class _Encoder(nn.Module):
    def __init__(self, *, blocks: int = 24) -> None:
        super().__init__()
        self.backbone = _Backbone(blocks=blocks)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.backbone(value)


class _Source(nn.Module):
    def __init__(self, *, blocks: int = 24) -> None:
        super().__init__()
        self.is_v3 = True
        self.encoder = _Encoder(blocks=blocks)
        self.head = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value))


def test_exact_activation_checkpointing_covers_every_source_block() -> None:
    source = _Source()
    value = torch.randn(3, 4, requires_grad=True)
    expected = source(value)

    blocks = source_fsdp2.apply_exact_videomt_activation_checkpointing(source)

    assert len(blocks) == source_fsdp2.VIDEOMT_DINOV3_L_BLOCKS
    assert all(hasattr(block.attention, "_checkpoint_wrapped_module") for block in blocks)
    assert all(hasattr(block.mlp, "_checkpoint_wrapped_module") for block in blocks)
    output = source(value)
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert all(block.attention._checkpoint_wrapped_module.weight.grad is not None for block in blocks)
    assert all(block.mlp._checkpoint_wrapped_module.weight.grad is not None for block in blocks)


def test_exact_activation_checkpointing_rejects_incomplete_backbone() -> None:
    with pytest.raises(ValueError, match="block inventory drifted"):
        source_fsdp2.apply_exact_videomt_activation_checkpointing(_Source(blocks=23))


def test_exact_fsdp2_shards_every_block_and_root(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _Source()
    parameter_numel = sum(parameter.numel() for parameter in source.parameters())
    sharded: list[nn.Module] = []
    shard_kwargs: list[dict[str, object]] = []

    def _record(module: nn.Module, **kwargs: object) -> None:
        sharded.append(module)
        shard_kwargs.append(kwargs)

    monkeypatch.setattr(
        source_fsdp2,
        "VIDEOMT_COMPLETE_TRAINABLE_PARAMETERS",
        parameter_numel,
    )
    monkeypatch.setattr(source_fsdp2, "fully_shard", _record)

    parallelized, receipt = source_fsdp2.parallelize_exact_videomt_fsdp2(source)

    assert parallelized is source
    assert len(sharded) == (
        source_fsdp2.VIDEOMT_DINOV3_L_BLOCKS
        * source_fsdp2.VIDEOMT_FSDP2_EXECUTION_UNITS_PER_BLOCK
        + 1
    )
    assert sharded[-1] is source
    expected_execution_modules = tuple(
        module
        for block in source.encoder.backbone.blocks
        for module in (block.attention, block.mlp)
    )
    assert tuple(sharded[:-1]) == expected_execution_modules
    assert all("mp_policy" in kwargs for kwargs in shard_kwargs)
    assert all("offload_policy" not in kwargs for kwargs in shard_kwargs)
    assert receipt.checkpointed_block_count == source_fsdp2.VIDEOMT_DINOV3_L_BLOCKS
    assert receipt.sharded_module_count == len(sharded)
    assert receipt.parameter_numel == parameter_numel
    assert receipt.trainable_parameter_numel == parameter_numel


def test_exact_fsdp2_cpu_offload_applies_public_policy_to_every_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _Source()
    parameter_numel = sum(parameter.numel() for parameter in source.parameters())
    shard_kwargs: list[dict[str, object]] = []
    policies: list[object] = []

    class _Policy:
        pass

    def _policy() -> _Policy:
        result = _Policy()
        policies.append(result)
        return result

    def _record(_module: nn.Module, **kwargs: object) -> None:
        shard_kwargs.append(kwargs)

    monkeypatch.setattr(
        source_fsdp2,
        "VIDEOMT_COMPLETE_TRAINABLE_PARAMETERS",
        parameter_numel,
    )
    monkeypatch.setattr(source_fsdp2, "CPUOffloadPolicy", _policy)
    monkeypatch.setattr(source_fsdp2, "fully_shard", _record)

    _parallelized, receipt = source_fsdp2.parallelize_exact_videomt_fsdp2(
        source,
        cpu_offload=True,
    )

    assert len(policies) == 1
    assert shard_kwargs
    assert all(kwargs["offload_policy"] is policies[0] for kwargs in shard_kwargs)
    assert receipt.cpu_offload is True


def test_exact_fsdp2_rejects_frozen_source(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _Source()
    parameter_numel = sum(parameter.numel() for parameter in source.parameters())
    next(source.parameters()).requires_grad_(False)
    monkeypatch.setattr(
        source_fsdp2,
        "VIDEOMT_COMPLETE_TRAINABLE_PARAMETERS",
        parameter_numel,
    )

    with pytest.raises(ValueError, match="fully trainable"):
        source_fsdp2.parallelize_exact_videomt_fsdp2(source)
