"""Typed, causal controls executed between two observation events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ExecutedControlBatch:
    """A bounded sequence of acknowledged controls, never requested actions.

    Oversized asynchronous intervals fail closed. A source may instead raise
    its frozen ``U_max`` or provide a separately audited ontology-aware
    compressor; this generic contract never averages incompatible fields.
    """

    values: torch.Tensor
    field_valid: torch.Tensor
    token_valid: torch.Tensor
    delta_time: torch.Tensor
    reset: torch.Tensor
    acknowledged: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or min(self.values.shape) <= 0:
            raise ValueError("control values must have shape [batch, controls, action_dim]")
        if not self.values.is_floating_point() or not torch.isfinite(self.values).all():
            raise ValueError("control values must be finite floating point")
        if self.field_valid.shape != self.values.shape or self.field_valid.dtype != torch.bool:
            raise ValueError("field_valid must be boolean and match control values")
        expected = self.values.shape[:2]
        for name, value in (
            ("token_valid", self.token_valid),
            ("reset", self.reset),
            ("acknowledged", self.acknowledged),
        ):
            if value.shape != expected or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape [batch, controls]")
        if self.delta_time.shape != expected or not self.delta_time.is_floating_point():
            raise ValueError("delta_time must be floating point with shape [batch, controls]")
        tensors = (
            self.field_valid,
            self.token_valid,
            self.delta_time,
            self.reset,
            self.acknowledged,
        )
        if any(value.device != self.values.device for value in tensors):
            raise ValueError("all executed-control tensors must share one device")
        if not torch.isfinite(self.delta_time).all() or (self.delta_time < 0).any():
            raise ValueError("control delta_time must be finite and non-negative")
        if (self.field_valid & ~self.token_valid.unsqueeze(-1)).any():
            raise ValueError("invalid control tokens cannot contain valid fields")
        if (self.reset & ~self.token_valid).any():
            raise ValueError("reset markers must be valid control tokens")
        if (self.token_valid & ~self.acknowledged).any():
            raise ValueError("every valid control/reset event must be execution-acknowledged")
        if (self.acknowledged & ~self.token_valid).any():
            raise ValueError("invalid control padding cannot be acknowledged")
        if ((~self.field_valid) & (self.values != 0)).any():
            raise ValueError("invalid control fields must be exactly zero")
        if ((~self.token_valid) & ((self.delta_time != 0) | self.reset)).any():
            raise ValueError("invalid control tokens must have zero metadata")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def token_count(self) -> int:
        return self.values.shape[1]

    @property
    def action_dim(self) -> int:
        return self.values.shape[2]

    def validate_bound(self, maximum_tokens: int) -> None:
        if isinstance(maximum_tokens, bool) or not isinstance(maximum_tokens, int):
            raise TypeError("maximum control count must be an integer")
        if maximum_tokens <= 0:
            raise ValueError("maximum control count must be positive")
        if self.token_count > maximum_tokens:
            raise ValueError(
                f"executed-control interval has {self.token_count} tokens, exceeding frozen "
                f"U_max={maximum_tokens}; increase U_max or use an audited typed adapter"
            )

    def canonical_features(self) -> torch.Tensor:
        """Linear-interface features; nonlinear dynamics remain in LingBot."""

        dtype = self.values.dtype
        dt = self.delta_time.to(dtype)
        return torch.cat(
            (
                self.values,
                self.field_valid.to(dtype),
                dt.unsqueeze(-1),
                torch.log1p(dt).unsqueeze(-1),
                self.reset.to(dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

    @classmethod
    def reset_only(
        cls,
        *,
        batch_size: int,
        action_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ExecutedControlBatch:
        if min(batch_size, action_dim) <= 0:
            raise ValueError("reset batch and action dimensions must be positive")
        shape = (batch_size, 1, action_dim)
        return cls(
            values=torch.zeros(shape, device=device, dtype=dtype),
            field_valid=torch.zeros(shape, device=device, dtype=torch.bool),
            token_valid=torch.ones((batch_size, 1), device=device, dtype=torch.bool),
            delta_time=torch.zeros((batch_size, 1), device=device, dtype=dtype),
            reset=torch.ones((batch_size, 1), device=device, dtype=torch.bool),
            acknowledged=torch.ones((batch_size, 1), device=device, dtype=torch.bool),
        )


def executed_control_chain_reset(
    controls: Sequence[ExecutedControlBatch],
) -> torch.Tensor:
    """Return per-sample reset truth over one exact, ordered control chain."""

    if not controls or any(not isinstance(value, ExecutedControlBatch) for value in controls):
        raise ValueError("executed-control reset reduction requires a non-empty typed chain")
    reference = controls[0]
    if any(
        value.batch_size != reference.batch_size
        or value.action_dim != reference.action_dim
        or value.values.device != reference.values.device
        or value.values.dtype != reference.values.dtype
        or value.delta_time.dtype != reference.delta_time.dtype
        for value in controls
    ):
        raise ValueError(
            "executed-control reset chain must share batch, action width, device, and dtype"
        )
    return torch.stack(
        tuple((value.reset & value.token_valid).any(dim=1) for value in controls),
        dim=0,
    ).any(dim=0)


def concatenate_executed_controls(
    controls: tuple[ExecutedControlBatch, ...],
) -> ExecutedControlBatch:
    """Concatenate one contiguous control interval without semantic compression."""

    if not controls or any(not isinstance(value, ExecutedControlBatch) for value in controls):
        raise ValueError("executed-control concatenation requires one or more typed batches")
    reference = controls[0]
    identity = (reference.batch_size, reference.action_dim, reference.values.device)
    if any(
        (value.batch_size, value.action_dim, value.values.device) != identity for value in controls
    ):
        raise ValueError(
            "concatenated executed controls must share batch, action width, and device"
        )
    if any(
        value.values.dtype != reference.values.dtype
        or value.delta_time.dtype != reference.delta_time.dtype
        for value in controls
    ):
        raise TypeError("concatenated executed controls must share floating dtypes")
    if any(value.reset.any() for value in controls):
        raise ValueError("a future executed-control interval cannot cross an episode reset")
    return ExecutedControlBatch(
        values=torch.cat(tuple(value.values for value in controls), dim=1),
        field_valid=torch.cat(tuple(value.field_valid for value in controls), dim=1),
        token_valid=torch.cat(tuple(value.token_valid for value in controls), dim=1),
        delta_time=torch.cat(tuple(value.delta_time for value in controls), dim=1),
        reset=torch.cat(tuple(value.reset for value in controls), dim=1),
        acknowledged=torch.cat(tuple(value.acknowledged for value in controls), dim=1),
    )
