from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.action_posterior_collector import (
    RegisteredActionPosteriorReceiptCollector,
)
from picf_next.lingbot_native.action_posterior_receipt import (
    ActionPosteriorAttentionReceipt,
    LingBotActionAttentionLayout,
)


def _surface() -> dict[str, torch.Tensor | LingBotActionAttentionLayout]:
    layout = LingBotActionAttentionLayout(
        batch_size=1,
        query_count=3,
        key_count=5,
        native_prefix_count=2,
        compact_prefix_count=4,
        state_query_slice=slice(0, 1),
        action_query_slice=slice(1, 3),
        posterior_key_indices=torch.tensor([2, 3]),
        posterior_key_valid=torch.tensor([[True, True]]),
        expanded_posterior_indices=torch.tensor([7, 8]),
        selected_inserted_indices=torch.tensor([7, 8]),
    )
    return {
        "query_states": torch.randn(1, 3, 2, 4),
        "key_states": torch.randn(1, 5, 1, 4),
        "attention_mask": torch.ones(1, 3, 5, dtype=torch.bool),
        "layout": layout,
    }


def _collect(
    collector: RegisteredActionPosteriorReceiptCollector,
    *,
    layer_index: int,
    layer_count: int = 6,
) -> ActionPosteriorAttentionReceipt | None:
    return collector(**_surface(), layer_index=layer_index, layer_count=layer_count)


def test_collector_replays_only_registered_layers_and_finalizes_in_sorted_order() -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(4, 2))

    assert _collect(collector, layer_index=1) is None
    layer_four = _collect(collector, layer_index=4)
    layer_two = _collect(collector, layer_index=2)
    receipts = collector.finalize()

    assert collector.registered_layer_indices == (2, 4)
    assert collector.collected_layer_indices == (2, 4)
    assert collector.layer_count == 6
    assert receipts == (layer_two, layer_four)
    assert all(isinstance(receipt, ActionPosteriorAttentionReceipt) for receipt in receipts)
    assert all(receipt.posterior_attention.shape == (1, 2, 2, 2) for receipt in receipts)


def test_unregistered_layer_does_not_replay_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(2,))

    def forbidden_replay(**_kwargs: object) -> ActionPosteriorAttentionReceipt:
        raise AssertionError("unregistered layers must not replay action attention")

    monkeypatch.setattr(
        "picf_next.lingbot_native.action_posterior_collector.action_posterior_attention_receipt",
        forbidden_replay,
    )
    assert _collect(collector, layer_index=1) is None
    assert collector.collected_layer_indices == ()


@pytest.mark.parametrize(
    ("indices", "exception", "message"),
    [
        ((), ValueError, "at least one"),
        ((1, 1), ValueError, "unique"),
        ((-1,), ValueError, "non-negative"),
        ((True,), TypeError, "integers"),
        ((1.0,), TypeError, "integers"),
    ],
)
def test_collector_rejects_invalid_layer_registration(
    indices: tuple[object, ...],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        RegisteredActionPosteriorReceiptCollector(registered_layer_indices=indices)


def test_collector_rejects_duplicate_registered_callback() -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(2,))
    _collect(collector, layer_index=2)
    with pytest.raises(RuntimeError, match="duplicate.*layer 2"):
        _collect(collector, layer_index=2)


def test_collector_rejects_missing_registered_layer_at_finalize() -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(2, 4))
    _collect(collector, layer_index=2)
    with pytest.raises(RuntimeError, match=r"missing.*\(4,\)"):
        collector.finalize()


def test_collector_rejects_inconsistent_layer_count_even_on_unregistered_layer() -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(2,))
    _collect(collector, layer_index=0, layer_count=6)
    with pytest.raises(ValueError, match="disagree on layer_count"):
        _collect(collector, layer_index=1, layer_count=7)


@pytest.mark.parametrize(
    ("layer_index", "layer_count", "exception"),
    [
        (6, 6, ValueError),
        (-1, 6, ValueError),
        (0, 0, ValueError),
        (True, 6, TypeError),
        (0, False, TypeError),
    ],
)
def test_collector_rejects_invalid_explicit_callback_identity(
    layer_index: object,
    layer_count: object,
    exception: type[Exception],
) -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(2,))
    with pytest.raises(exception, match="explicit"):
        collector(**_surface(), layer_index=layer_index, layer_count=layer_count)


def test_collector_rejects_registered_layer_outside_model_layer_count() -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(2, 6))
    with pytest.raises(ValueError, match="outside layer_count"):
        _collect(collector, layer_index=0, layer_count=6)


def test_reset_starts_a_new_single_forward_and_releases_old_receipts() -> None:
    collector = RegisteredActionPosteriorReceiptCollector(registered_layer_indices=(2, 4))
    _collect(collector, layer_index=2)
    _collect(collector, layer_index=4)
    first = collector.finalize()

    with pytest.raises(RuntimeError, match="reset.*another forward"):
        _collect(collector, layer_index=2)

    collector.reset()
    assert collector.layer_count is None
    assert collector.collected_layer_indices == ()
    _collect(collector, layer_index=4)
    _collect(collector, layer_index=2)
    second = collector.finalize()

    assert tuple(receipt.layer_index for receipt in first) == (2, 4)
    assert tuple(receipt.layer_index for receipt in second) == (2, 4)
    assert all(new is not old for new, old in zip(second, first, strict=True))
