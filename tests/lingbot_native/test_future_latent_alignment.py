from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from picf_next.contracts import ContractError
from picf_next.data.calvin import CALVIN_HOST_IMAGE_KEYS, CalvinMolmoAct2SourceObservation
from picf_next.lingbot_native.future_latent_alignment import (
    FutureLatentAlignmentConfig,
    FutureLatentTargetBatch,
    LingBotFutureLatentAlignment,
    future_latent_objective_contribution,
)
from picf_next.lingbot_native.future_latent_cache import (
    FrozenSiglip2FutureEncoder,
    FutureLatentCacheContract,
    FutureLatentCacheRecord,
    FutureLatentTargetCache,
    future_latent_source_keys_digest,
    write_future_latent_target_cache,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _target_batch(batch: int = 2) -> FutureLatentTargetBatch:
    config = FutureLatentAlignmentConfig()
    return FutureLatentTargetBatch(
        tokens=torch.randn(batch, config.future_token_count, config.target_width),
        sample_keys=tuple(f"sample-{index}" for index in range(batch)),
        source_global_indices=tuple(100 + index for index in range(batch)),
        future_global_indices=tuple(116 + index for index in range(batch)),
        manifest_sha256=_digest("manifest"),
        config_digest=config.digest,
    )


def test_complete_alignment_appends_all_tokens_and_captures_the_frozen_layer() -> None:
    torch.manual_seed(4)
    config = FutureLatentAlignmentConfig()
    alignment = LingBotFutureLatentAlignment(config)
    target = _target_batch()
    context = alignment.new_forward_context(target)
    suffix = torch.randn(2, 33, config.action_hidden_width, requires_grad=True)
    valid = torch.ones(2, 33, dtype=torch.bool)
    blocks = torch.zeros(2, 33, dtype=torch.bool)

    embeddings, extended_valid, extended_blocks = alignment.append_future_tokens(
        suffix_embeddings=suffix,
        suffix_valid=valid,
        suffix_blocks=blocks,
        context=context,
    )
    assert embeddings.shape == (2, 161, 768)
    assert extended_valid.all()
    assert not extended_blocks[:, 33:].any()
    for layer in range(config.action_layer_count):
        context.record_action_hidden(
            action_hidden=embeddings,
            layer_index=layer,
            layer_count=config.action_layer_count,
        )
    with pytest.raises(RuntimeError, match="omitted in-forward FLARE finalization"):
        context.finalized_result(require_grad=True)
    result = context.finalize(alignment, require_grad=True)
    assert context.finalized_result(require_grad=True) is result
    assert result.prediction.shape == (2, 128, 1024)
    assert torch.allclose(result.weighted_loss, result.raw_loss * 0.2)
    (suffix.square().mean() + result.weighted_loss).backward()
    assert alignment.future_tokens.weight.grad is not None
    assert alignment.embedding_decoder[0].weight.grad is not None


def test_alignment_respects_the_fsdp_mixed_precision_decoder_boundary() -> None:
    config = FutureLatentAlignmentConfig()
    alignment = LingBotFutureLatentAlignment(config).to(dtype=torch.bfloat16)
    context = alignment.new_forward_context(_target_batch(1))
    suffix = torch.randn(1, 33, config.action_hidden_width, requires_grad=True)
    hidden, _valid, _blocks = alignment.append_future_tokens(
        suffix_embeddings=suffix,
        suffix_valid=torch.ones(1, 33, dtype=torch.bool),
        suffix_blocks=torch.zeros(1, 33, dtype=torch.bool),
        context=context,
    )
    context.record_action_hidden(
        action_hidden=hidden,
        layer_index=config.capture_layer_index,
        layer_count=config.action_layer_count,
    )

    result = context.finalize(alignment, require_grad=True)

    assert result.prediction.dtype == torch.bfloat16
    assert result.raw_loss.dtype == torch.float32
    result.weighted_loss.backward()
    assert alignment.embedding_decoder[0].weight.grad is not None


def test_alignment_typed_capture_survives_whole_module_compile() -> None:
    """The Python context may graph-break, but its captured tensor must stay attached."""

    class CompiledCapture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.alignment = LingBotFutureLatentAlignment()

        def forward(
            self,
            suffix: torch.Tensor,
            valid: torch.Tensor,
            blocks: torch.Tensor,
            context: object,
        ) -> torch.Tensor:
            hidden, _valid, _blocks = self.alignment.append_future_tokens(
                suffix_embeddings=suffix,
                suffix_valid=valid,
                suffix_blocks=blocks,
                context=context,
            )
            context.record_action_hidden(
                action_hidden=hidden,
                layer_index=self.alignment.config.capture_layer_index,
                layer_count=self.alignment.config.action_layer_count,
            )
            return context.finalize(self.alignment, require_grad=True).weighted_loss

    torch.manual_seed(11)
    eager = CompiledCapture()
    compiled_source = CompiledCapture()
    compiled_source.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_source, backend="eager")
    target = _target_batch(1)
    suffix_eager = torch.randn(1, 33, 768, requires_grad=True)
    suffix_compiled = suffix_eager.detach().clone().requires_grad_(True)
    valid = torch.ones(1, 33, dtype=torch.bool)
    blocks = torch.zeros(1, 33, dtype=torch.bool)

    eager_loss = eager(
        suffix_eager,
        valid,
        blocks,
        eager.alignment.new_forward_context(target),
    )
    compiled_loss = compiled(
        suffix_compiled,
        valid,
        blocks,
        compiled.alignment.new_forward_context(target),
    )
    assert torch.equal(compiled_loss, eager_loss)
    eager_loss.backward()
    compiled_loss.backward()
    assert torch.equal(suffix_compiled.grad, suffix_eager.grad)
    assert torch.equal(
        compiled.alignment.embedding_decoder[0].weight.grad,
        eager.alignment.embedding_decoder[0].weight.grad,
    )


def test_alignment_contract_rejects_resource_motivated_reductions() -> None:
    with pytest.raises(ValueError, match="complete frozen FLARE"):
        reduced = FutureLatentAlignmentConfig(future_token_count=64, tokens_per_view=32)
        reduced.assert_adr209_complete()
    with pytest.raises(ValueError, match="exactly 16"):
        target = _target_batch(1)
        FutureLatentTargetBatch(
            tokens=target.tokens,
            sample_keys=target.sample_keys,
            source_global_indices=(100,),
            future_global_indices=(115,),
            manifest_sha256=target.manifest_sha256,
            config_digest=target.config_digest,
        )


def test_lambda_zero_control_retains_the_complete_arm_and_only_zeros_its_objective() -> None:
    config = FutureLatentAlignmentConfig()
    alignment = LingBotFutureLatentAlignment(config)
    context = alignment.new_forward_context(_target_batch(1))
    suffix = torch.randn(1, 33, config.action_hidden_width, requires_grad=True)
    hidden, _valid, _blocks = alignment.append_future_tokens(
        suffix_embeddings=suffix,
        suffix_valid=torch.ones(1, 33, dtype=torch.bool),
        suffix_blocks=torch.zeros(1, 33, dtype=torch.bool),
        context=context,
    )
    context.record_action_hidden(
        action_hidden=hidden,
        layer_index=config.capture_layer_index,
        layer_count=config.action_layer_count,
    )
    result = context.finalize(alignment, require_grad=True)

    candidate = future_latent_objective_contribution(result, scale=1.0)
    control = future_latent_objective_contribution(result, scale=0.0)
    assert candidate is not result.weighted_loss
    assert torch.equal(candidate, result.weighted_loss)
    assert control.requires_grad
    assert control.item() == 0.0
    with pytest.raises(ValueError, match="exactly zero or one"):
        future_latent_objective_contribution(result, scale=0.5)


class _FakeProcessor:
    def __call__(self, *, images: list[np.ndarray], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        markers = torch.tensor([float(image[0, 0, 0]) for image in images])
        return {"pixel_values": markers[:, None, None, None].expand(-1, 3, 256, 256)}


class _FakeVisionModel(nn.Module):
    def forward(self, *, pixel_values: torch.Tensor, return_dict: bool) -> SimpleNamespace:
        assert return_dict
        batch = pixel_values.shape[0]
        marker = pixel_values[:, 0, 0, 0].float().reshape(batch, 1, 1) * 1000
        patch = torch.arange(256, device=pixel_values.device).reshape(1, 256, 1)
        hidden = (marker + patch).expand(batch, 256, 1024)
        return SimpleNamespace(last_hidden_state=hidden.to(dtype=pixel_values.dtype))


def _observation(static: int, gripper: int) -> CalvinMolmoAct2SourceObservation:
    static_image = np.full((200, 200, 3), static, dtype=np.uint8)
    gripper_image = np.full((84, 84, 3), gripper, dtype=np.uint8)
    static_image.setflags(write=False)
    gripper_image.setflags(write=False)
    state = np.zeros(15, dtype=np.float32)
    valid = np.ones(15, dtype=np.bool_)
    state.setflags(write=False)
    valid.setflags(write=False)
    return CalvinMolmoAct2SourceObservation(
        images={
            CALVIN_HOST_IMAGE_KEYS[0]: static_image,
            CALVIN_HOST_IMAGE_KEYS[1]: gripper_image,
        },
        state=state,
        state_valid=valid,
        timestamp_s=0.0,
        delta_t_s=0.05,
    )


def test_frozen_teacher_retains_both_views_and_exact_2x2_pooling() -> None:
    encoder = FrozenSiglip2FutureEncoder(
        model=_FakeVisionModel(),
        image_processor=_FakeProcessor(),
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
    )
    targets = encoder.encode((_observation(1, 2), _observation(3, 4)))
    assert targets.shape == (2, 128, 1024)
    assert targets.dtype == torch.float32
    assert targets[0, 0, 0].item() == pytest.approx(1008.5)
    assert targets[0, 63, 0].item() == pytest.approx(1246.5)
    assert targets[0, 64, 0].item() == pytest.approx(2008.5)
    assert targets[1, 64, 0].item() == pytest.approx(4008.5)


def test_future_cache_round_trip_binds_every_identity_and_tensor(tmp_path) -> None:
    config = FutureLatentAlignmentConfig()
    records = tuple(
        FutureLatentCacheRecord(
            sample_key=f"sample-{index}",
            source_global_index=100 + index,
            future_global_index=116 + index,
            future_view_sha256=(_digest(f"static-{index}"), _digest(f"gripper-{index}")),
            tokens=torch.full((128, 1024), float(index), dtype=torch.float32),
        )
        for index in range(3)
    )
    identities = tuple(
        (record.sample_key, record.source_global_index, record.future_global_index)
        for record in records
    )
    contract = FutureLatentCacheContract(
        dataset_id="calvin",
        dataset_revision="revision",
        split_name="training",
        dataset_tree_sha256=_digest("tree"),
        stream_plan_sha256=_digest("plan"),
        stream_plan_file_sha256=_digest("plan-file"),
        representation_split_sha256=_digest("representation-split"),
        source_keys_sha256=future_latent_source_keys_digest(identities),
        encoder_config_sha256=_digest("config"),
        encoder_checkpoint_sha256=_digest("checkpoint"),
        encoder_processor_sha256=_digest("processor"),
        expected_record_count=len(records),
        training_prefix_steps=250,
        alignment_config_digest=config.digest,
    )
    root = tmp_path / "cache"
    manifest_sha256 = write_future_latent_target_cache(
        root,
        contract=contract,
        records=records,
        records_per_shard=2,
    )
    cache = FutureLatentTargetCache(root, maximum_open_shards=1)
    target = cache.target_for(
        sample_keys=("sample-2", "sample-0"),
        source_global_indices=(102, 100),
        device=torch.device("cpu"),
    )
    assert target.manifest_sha256 == manifest_sha256
    assert target.future_global_indices == (118, 116)
    assert target.tokens.shape == (2, 128, 1024)
    assert target.tokens[0].eq(2).all()
    assert target.tokens[1].eq(0).all()
    repeated = cache.target_for(
        sample_keys=("sample-1", "sample-1"),
        source_global_indices=(101, 101),
        device=torch.device("cpu"),
    )
    assert repeated.tokens.shape == (2, 128, 1024)
    assert repeated.tokens.eq(1).all()
    with pytest.raises(ContractError, match="source identity"):
        cache.target_for(
            sample_keys=("sample-0",),
            source_global_indices=(999,),
            device=torch.device("cpu"),
        )
