"""Strict single-forward collection of registered action-posterior receipts."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from picf_next.lingbot_native.action_posterior_receipt import (
    ActionPosteriorAttentionLayout,
    ActionPosteriorAttentionReceipt,
    action_posterior_attention_receipt,
)


class RegisteredActionPosteriorReceiptCollector:
    """Collect exact receipts for one preregistered set of transformer layers.

    The collector is a callback for the released LingBot attention surface. It
    validates explicit layer identity on every callback, but replays attention
    only for preregistered layers. Between :meth:`reset` calls it represents one
    policy forward and retains only the resulting compact receipts.

    Target validity and loss semantics deliberately live outside this class.
    """

    __slots__ = (
        "_finalized",
        "_layer_count",
        "_receipts",
        "_registered_layer_indices",
    )

    def __init__(self, *, registered_layer_indices: Iterable[int]) -> None:
        indices = tuple(registered_layer_indices)
        if not indices:
            raise ValueError("at least one action-posterior layer must be registered")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("registered action-posterior layer indices must be integers")
        if any(index < 0 for index in indices):
            raise ValueError("registered action-posterior layer indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("registered action-posterior layer indices must be unique")
        self._registered_layer_indices = tuple(sorted(indices))
        self._receipts: dict[int, ActionPosteriorAttentionReceipt] = {}
        self._layer_count: int | None = None
        self._finalized = False

    @property
    def registered_layer_indices(self) -> tuple[int, ...]:
        """The immutable, sorted layer registration for every forward."""

        return self._registered_layer_indices

    @property
    def collected_layer_indices(self) -> tuple[int, ...]:
        """Layers collected so far, sorted independently of callback order."""

        return tuple(sorted(self._receipts))

    @property
    def layer_count(self) -> int | None:
        """The explicit model layer count established by the current forward."""

        return self._layer_count

    def reset(self) -> None:
        """Release receipts and begin a new single-forward collection."""

        self._receipts.clear()
        self._layer_count = None
        self._finalized = False

    def __call__(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layout: ActionPosteriorAttentionLayout,
        layer_index: int,
        layer_count: int,
    ) -> ActionPosteriorAttentionReceipt | None:
        """Validate one layer callback and collect it only when registered."""

        if self._finalized:
            raise RuntimeError("reset the action-posterior collector before another forward")
        self._validate_layer_identity(layer_index=layer_index, layer_count=layer_count)
        if self._layer_count is None:
            if self._registered_layer_indices[-1] >= layer_count:
                raise ValueError("a registered action-posterior layer is outside layer_count")
            self._layer_count = layer_count
        elif layer_count != self._layer_count:
            raise ValueError("action-posterior callbacks disagree on layer_count")

        if layer_index not in self._registered_layer_indices:
            return None
        if layer_index in self._receipts:
            raise RuntimeError(f"duplicate action-posterior callback for layer {layer_index}")

        receipt = action_posterior_attention_receipt(
            query_states=query_states,
            key_states=key_states,
            attention_mask=attention_mask,
            layout=layout,
            layer_index=layer_index,
            layer_count=layer_count,
        )
        self._receipts[layer_index] = receipt
        return receipt

    def finalize(self) -> tuple[ActionPosteriorAttentionReceipt, ...]:
        """Return receipts in registered layer order, rejecting omissions."""

        if self._finalized:
            return tuple(self._receipts[index] for index in self._registered_layer_indices)
        missing = tuple(
            index for index in self._registered_layer_indices if index not in self._receipts
        )
        if missing:
            raise RuntimeError(f"missing registered action-posterior layers: {missing}")
        self._finalized = True
        return tuple(self._receipts[index] for index in self._registered_layer_indices)

    @staticmethod
    def _validate_layer_identity(*, layer_index: int, layer_count: int) -> None:
        if (
            isinstance(layer_index, bool)
            or isinstance(layer_count, bool)
            or not isinstance(layer_index, int)
            or not isinstance(layer_count, int)
        ):
            raise TypeError("action-posterior callback layer identity must be explicit integers")
        if layer_count <= 0 or not 0 <= layer_index < layer_count:
            raise ValueError("action-posterior callback has an invalid explicit layer identity")
