from __future__ import annotations

import pytest
import torch
from torch import nn

from picf_next.videomt_exact.runtime import (
    ExactVidEoMTOutput,
    ExactVidEoMTRuntime,
    normalize_rgb_255,
)


class _HookCompatibleVidEoMT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Embedding(200, 1024)
        self.class_head = nn.Linear(1024, 41)
        self.mask_head = nn.Identity()
        self.upscale = nn.Identity()
        self.num_frames = 5
        self.last_query_embed: torch.Tensor | None = None
        self.resume_values: list[bool] = []

    def _clear_memory(self) -> None:
        self.last_query_embed = None

    def forward(self, value: torch.Tensor, *, resume: bool = False):
        assert value.ndim == 4 and value.shape[1:] == (3, 16, 16)
        self.resume_values.append(resume)
        time = self.num_frames if value.shape[0] % self.num_frames == 0 else value.shape[0]
        batch = value.shape[0] // time
        if not resume:
            self._clear_memory()
        base = value.mean(dim=(1, 2, 3)).reshape(batch, time, 1, 1)
        query_ids = torch.arange(200, dtype=value.dtype).reshape(1, 200, 1)
        queries = (base + query_ids.reshape(1, 1, 200, 1) / 200).expand(
            batch, time, 200, 1024
        ).clone()
        if self.last_query_embed is not None:
            queries = queries + self.last_query_embed.unsqueeze(1)
        flat_queries = queries.flatten(0, 1)
        features = value.mean(dim=1, keepdim=True)[:, :, :2, :2].expand(-1, 1024, -1, -1)
        class_logits = self.class_head(flat_queries).reshape(batch, time, 200, 41)
        mask_embeddings = self.mask_head(flat_queries)
        mask_features = self.upscale(features)
        masks = torch.einsum("bqc,bchw->bqhw", mask_embeddings, mask_features)
        self.last_query_embed = queries[:, -1].clone()
        auxiliary = tuple(
            {
                "pred_logits": class_logits + layer,
                "pred_masks": masks.reshape(batch, time, 200, 2, 2).permute(0, 2, 1, 3, 4)
                + layer,
            }
            for layer in range(4)
        )
        return {
            "pred_logits": class_logits,
            "pred_masks": masks.reshape(batch, time, 200, 2, 2).permute(0, 2, 1, 3, 4),
            "aux_outputs": auxiliary,
        }


def test_rgb_normalization_requires_explicit_255_scale() -> None:
    rgb = torch.tensor([[[[0]], [[127]], [[255]]]], dtype=torch.uint8)
    normalized = normalize_rgb_255(rgb)
    expected = (
        rgb.float() - torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1)
    ) / torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1)
    torch.testing.assert_close(normalized, expected)

    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        normalize_rgb_255(torch.full((1, 3, 2, 2), 256.0))


def test_output_contract_rejects_reduced_query_bank() -> None:
    with pytest.raises(ValueError, match="released architecture"):
        ExactVidEoMTOutput(
            class_logits=torch.zeros(1, 1, 20, 41),
            mask_logits=torch.zeros(1, 20, 1, 16, 16),
            query_embeddings=torch.zeros(1, 1, 20, 1024),
            propagated_queries=torch.zeros(1, 20, 1024),
            auxiliary_outputs=(),
        )


def test_runtime_captures_the_exact_released_mask_decoder_factorization() -> None:
    runtime = ExactVidEoMTRuntime(object(), _HookCompatibleVidEoMT())
    output = runtime(torch.randn(2, 3, 16, 16), resume=False)

    assert output.latest_mask_embeddings.shape == (1, 200, 1024)
    assert output.latest_mask_features.shape == (1, 1024, 2, 2)
    reconstructed = torch.einsum(
        "bqc,bchw->bqhw",
        output.latest_mask_embeddings,
        output.latest_mask_features,
    )
    torch.testing.assert_close(reconstructed, output.mask_logits[:, :, -1])


def test_runtime_streams_native_resume_and_reassembles_every_auxiliary_read() -> None:
    model = _HookCompatibleVidEoMT().train()
    runtime = ExactVidEoMTRuntime(object(), model)
    frames = torch.arange(3 * 3 * 16 * 16, dtype=torch.float32).reshape(3, 3, 16, 16)

    sequence = runtime.forward_causal_sequence(frames, resume=False)

    assert model.num_frames == 5
    assert model.resume_values == [False, True, True]
    assert sequence.merged.class_logits.shape == (1, 3, 200, 41)
    assert sequence.merged.mask_logits.shape == (1, 200, 3, 2, 2)
    assert sequence.merged.query_embeddings.shape == (1, 3, 200, 1024)
    assert len(sequence.merged.auxiliary_outputs) == 4
    for layer, auxiliary in enumerate(sequence.merged.auxiliary_outputs):
        assert auxiliary["pred_logits"].shape == (1, 3, 200, 41)
        assert auxiliary["pred_masks"].shape == (1, 200, 3, 2, 2)
        torch.testing.assert_close(
            auxiliary["pred_logits"],
            sequence.merged.class_logits + layer,
        )
    torch.testing.assert_close(
        sequence.propagated_queries_by_frame[-1],
        sequence.merged.propagated_queries,
    )


def test_runtime_batches_exact_mixed_reset_and_resume_query_seeds() -> None:
    model = _HookCompatibleVidEoMT().train()
    runtime = ExactVidEoMTRuntime(object(), model)
    previous = torch.randn(2, 200, 1024)
    reset = torch.tensor([True, False])

    initial = runtime.bind_mixed_propagated_queries(previous, reset=reset)

    torch.testing.assert_close(initial[0], model.q.weight)
    torch.testing.assert_close(initial[1], previous[1])
    frames = torch.randn(2, 3, 3, 16, 16)
    sequence = runtime.forward_causal_sequence(frames, resume=True)
    assert model.resume_values == [True, True, True]
    assert sequence.merged.class_logits.shape == (2, 3, 200, 41)
    assert sequence.merged.mask_logits.shape == (2, 200, 3, 2, 2)
    assert all(state.shape == (2, 200, 1024) for state in sequence.propagated_queries_by_frame)
    runtime.restore_propagated_queries(sequence.propagated_queries_by_frame[0])
    torch.testing.assert_close(model.last_query_embed, sequence.propagated_queries_by_frame[0])


def test_runtime_rejects_resume_without_a_cached_query_state() -> None:
    runtime = ExactVidEoMTRuntime(object(), _HookCompatibleVidEoMT())
    with pytest.raises(ValueError, match="require propagated"):
        runtime.bind_mixed_propagated_queries(
            None,
            reset=torch.tensor([True, False]),
        )


def test_runtime_uses_fsdp2_root_lifecycle_for_query_state_binding() -> None:
    model = _HookCompatibleVidEoMT().train()
    events: list[str] = []

    def unshard(*, async_op: bool) -> None:
        assert async_op is False
        events.append("unshard")

    def reshard() -> None:
        events.append("reshard")

    model.unshard = unshard
    model.reshard = reshard
    runtime = ExactVidEoMTRuntime(object(), model)

    runtime.bind_mixed_propagated_queries(
        None,
        reset=torch.ones(1, dtype=torch.bool),
    )
    assert events == ["unshard"]
    runtime.reset_state()
    assert events == ["unshard", "reshard"]


def test_runtime_reshards_fsdp2_root_when_query_binding_fails() -> None:
    model = _HookCompatibleVidEoMT().train()
    events: list[str] = []
    model.unshard = lambda *, async_op: events.append(f"unshard:{async_op}")
    model.reshard = lambda: events.append("reshard")
    runtime = ExactVidEoMTRuntime(object(), model)

    with pytest.raises(ValueError, match="cached VidEoMT queries"):
        runtime.bind_mixed_propagated_queries(
            torch.randn(1, 200, 1024, dtype=torch.float64),
            reset=torch.zeros(1, dtype=torch.bool),
        )
    assert events == ["unshard:False", "reshard"]


def test_runtime_validates_restored_state_against_fsdp2_compute_dtype() -> None:
    model = _HookCompatibleVidEoMT().train()
    events: list[str] = []

    def unshard(*, async_op: bool) -> None:
        assert async_op is False
        events.append("unshard")
        model.q.weight.data = model.q.weight.data.to(torch.bfloat16)

    def reshard() -> None:
        events.append("reshard")
        model.q.weight.data = model.q.weight.data.to(torch.float32)

    model.unshard = unshard
    model.reshard = reshard
    runtime = ExactVidEoMTRuntime(object(), model)
    restored = torch.randn(1, 200, 1024, dtype=torch.bfloat16)

    runtime.restore_propagated_queries(restored)

    assert events == ["unshard", "reshard"]
    assert model.q.weight.dtype == torch.float32
    assert model.last_query_embed is restored
    assert model.last_query_embed.dtype == torch.bfloat16
