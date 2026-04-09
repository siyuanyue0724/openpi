from __future__ import annotations

import pytest
import torch

from openpi.picf.paligemma.wrapper import _checkpoint_inputs_require_grad
from openpi.picf.paligemma.wrapper import _repair_missing_tied_embeddings
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
