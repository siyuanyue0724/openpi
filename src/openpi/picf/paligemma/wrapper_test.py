from __future__ import annotations

import pytest
import torch

from openpi.models_pytorch.transformers_replace.models.paligemma.safe_ops import merge_image_features_dense
from openpi.models_pytorch.transformers_replace.models.paligemma.safe_ops import replace_oov_image_tokens
from openpi.models_pytorch.pi0_pytorch import _ensure_transformers_replace_is_ready
from openpi.picf.paligemma.wrapper import _checkpoint_inputs_require_grad
from openpi.picf.paligemma.wrapper import _enable_gradient_checkpointing_non_reentrant
from openpi.picf.paligemma.wrapper import _masked_position_ids
from openpi.picf.paligemma.wrapper import _recover_flow_target
from openpi.picf.paligemma.wrapper import _repair_missing_tied_embeddings
from openpi.picf.paligemma.wrapper import _take_valid_prefix_tokens
from openpi.picf.paligemma.wrapper import _Pi0PaliGemmaSemanticEncoder


class _TinyLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)


class _TinyPaliGemma(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = _TinyLanguageModel()
        self.lm_head = torch.nn.Linear(3, 4, bias=False)


def test_repair_missing_tied_embeddings_copies_lm_head_weights() -> None:
    model = _TinyPaliGemma()
    with torch.no_grad():
        model.model.language_model.embed_tokens.weight.zero_()
        model.lm_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            )
        )

    remaining = _repair_missing_tied_embeddings(
        model,
        missing_keys=["model.language_model.embed_tokens.weight"],
    )

    assert remaining == []
    torch.testing.assert_close(
        model.model.language_model.embed_tokens.weight,
        model.lm_head.weight,
    )


def test_repair_missing_tied_embeddings_leaves_unrelated_missing_keys() -> None:
    model = _TinyPaliGemma()
    remaining = _repair_missing_tied_embeddings(
        model,
        missing_keys=["some.other.weight"],
    )
    assert remaining == ["some.other.weight"]


def test_prepare_image_accepts_resize_with_pad_batch1_squeeze(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32

    def _fake_resize(_images, _h, _w):
        return torch.zeros((224, 224, 3), dtype=torch.float32)

    import openpi.picf.paligemma.wrapper as wrapper_mod

    monkeypatch.setattr(wrapper_mod.image_tools, "resize_with_pad_torch", _fake_resize)
    image = torch.zeros((32, 48, 3), dtype=torch.float32).numpy()
    out = _Pi0PaliGemmaSemanticEncoder._prepare_image(encoder, image)
    assert tuple(out.shape) == (1, 3, 224, 224)


def test_checkpoint_inputs_require_grad_detects_tensor_inputs() -> None:
    no_grad = torch.zeros(2)
    requires_grad = torch.zeros(2, requires_grad=True)

    assert _checkpoint_inputs_require_grad(no_grad) is False
    assert _checkpoint_inputs_require_grad(no_grad, requires_grad) is True


def test_enable_gradient_checkpointing_prefers_non_reentrant() -> None:
    class _DummyModule:
        def __init__(self) -> None:
            self.kwargs = None

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.kwargs = gradient_checkpointing_kwargs

    module = _DummyModule()
    enabled, non_reentrant = _enable_gradient_checkpointing_non_reentrant(module)

    assert enabled is True
    assert non_reentrant is True
    assert module.kwargs == {"use_reentrant": False}


def test_enable_gradient_checkpointing_falls_back_when_kwargs_unsupported() -> None:
    class _LegacyModule:
        def __init__(self) -> None:
            self.calls = 0

        def gradient_checkpointing_enable(self):
            self.calls += 1

    module = _LegacyModule()
    enabled, non_reentrant = _enable_gradient_checkpointing_non_reentrant(module)

    assert enabled is True
    assert non_reentrant is False
    assert module.calls == 1


def test_apply_checkpoint_skips_checkpoint_when_inputs_have_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.trainable = True
    encoder.training = True
    encoder.gradient_checkpointing_enabled = True

    def _boom(*args, **kwargs):
        raise AssertionError("checkpoint should not be invoked for non-grad inputs")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _boom)
    output = _Pi0PaliGemmaSemanticEncoder._apply_checkpoint(encoder, lambda x: x + 1, torch.zeros(2))
    torch.testing.assert_close(output, torch.ones(2))


def test_apply_checkpoint_uses_checkpoint_when_inputs_require_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.trainable = True
    encoder.training = True
    encoder.gradient_checkpointing_enabled = True
    called = {"value": False}

    def _checkpoint(func, *args, **kwargs):
        called["value"] = True
        return func(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _checkpoint)
    x = torch.zeros(2, requires_grad=True)
    output = _Pi0PaliGemmaSemanticEncoder._apply_checkpoint(encoder, lambda y: y + 2, x)
    torch.testing.assert_close(output, torch.full((2,), 2.0))
    assert called["value"] is True


def test_masked_position_ids_keep_valid_positions_and_zero_pad() -> None:
    pad_mask = torch.tensor([[True, True, True, False, False], [True, False, True, False, False]])

    position_ids = _masked_position_ids(pad_mask)

    torch.testing.assert_close(
        position_ids,
        torch.tensor([[0, 1, 2, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.int64),
    )


def test_recover_flow_target_inverts_pi05_training_parameterization() -> None:
    target = torch.tensor([[[0.2, -0.1], [0.4, 0.3]]], dtype=torch.float32)
    noise = torch.tensor([[[1.0, -0.5], [0.7, 0.8]]], dtype=torch.float32)
    time = torch.tensor([0.25], dtype=torch.float32)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * target
    u_t = noise - target

    recovered = _recover_flow_target(x_t, u_t, time_expanded)

    torch.testing.assert_close(recovered, target)


def test_take_valid_prefix_tokens_uses_right_padded_prefix_slice() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    pad_mask = torch.tensor([True, True, True, True, False, False])
    got = _take_valid_prefix_tokens(hidden, pad_mask)
    torch.testing.assert_close(got, hidden[:4])


def test_take_valid_prefix_tokens_rejects_non_right_padded_mask() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    pad_mask = torch.tensor([True, False, True, False, False, False])
    with pytest.raises(RuntimeError, match="right-padded"):
        _take_valid_prefix_tokens(hidden, pad_mask)


def test_replace_oov_image_tokens_uses_where_without_boolean_setitem() -> None:
    input_ids = torch.tensor([[5, 9, 1, 9], [9, 2, 3, 9]], dtype=torch.int64)
    replaced, mask = replace_oov_image_tokens(input_ids, image_token_id=9, vocab_size=8)
    torch.testing.assert_close(mask, input_ids == 9)
    torch.testing.assert_close(
        replaced,
        torch.tensor([[5, 0, 1, 0], [0, 2, 3, 0]], dtype=torch.int64),
    )


def test_merge_image_features_dense_matches_special_image_slots() -> None:
    inputs_embeds = torch.zeros((1, 6, 2), dtype=torch.float32)
    input_ids = torch.tensor([[9, 9, 4, 5, 9, 9]], dtype=torch.int64)
    image_features = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]], dtype=torch.float32)

    merged = merge_image_features_dense(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        image_features=image_features,
        image_token_id=9,
    )

    torch.testing.assert_close(
        merged,
        torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [0.0, 0.0], [0.0, 0.0], [3.0, 30.0], [4.0, 40.0]]],
            dtype=torch.float32,
        ),
    )


def test_build_paligemma_with_expert_uses_runtime_patched_gemma_classes() -> None:
    _ensure_transformers_replace_is_ready()
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    model = _Pi0PaliGemmaSemanticEncoder._build_paligemma_with_expert(
        encoder,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        precision="float32",
        pi05=True,
    )

    layer = model.gemma_expert.model.layers[0]
    assert hasattr(layer.input_layernorm, "dense")
    assert hasattr(layer.post_attention_layernorm, "dense")
