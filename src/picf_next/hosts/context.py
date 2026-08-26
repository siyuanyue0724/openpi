"""Host-neutral action-side evidence contract shared by VLA adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from picf_next.models.evidence import NativeTokenBank


@dataclass(frozen=True, slots=True)
class PICFActionEvidence:
    """Complete dense evidence plus persistent object address and state.

    Dense banks retain native token widths until each host adapter projects
    them. Optional ownership adds a persistent-address component without
    replacing native content. Object keys accept address only; dynamic state,
    uncertainty and innovation belong in values. The host adapter validates
    shape, dtype, device and padding before use.
    """

    dense_banks: tuple[NativeTokenBank, ...]
    object_address: torch.Tensor | None
    object_value: torch.Tensor | None
    object_valid: torch.Tensor | None
    object_log_prior: torch.Tensor | None = None
    dense_ownership: tuple[torch.Tensor, ...] | None = None

    def batch_size(self) -> int | None:
        """Return one unambiguous evidence batch size, or ``None`` when empty."""

        if not isinstance(self.dense_banks, tuple):
            raise TypeError("dense banks must be an immutable tuple")
        if self.dense_ownership is not None and not isinstance(self.dense_ownership, tuple):
            raise TypeError("dense ownership must be an immutable tuple when supplied")
        object_fields = (
            self.object_address,
            self.object_value,
            self.object_valid,
            self.object_log_prior,
        )
        if any(value is not None for value in object_fields) and not all(
            value is not None for value in object_fields
        ):
            raise ValueError(
                "object address, value, validity and log prior must be all present or absent"
            )
        tensors: list[tuple[str, torch.Tensor]] = []
        for index, bank in enumerate(self.dense_banks):
            if not isinstance(bank, NativeTokenBank):
                raise TypeError(f"dense bank {index} must be a NativeTokenBank")
            tensors.append((f"dense bank {index} tokens", bank.tokens))
        for name, value in zip(
            ("object address", "object value", "object validity", "object log prior"),
            object_fields,
            strict=True,
        ):
            if value is not None:
                tensors.append((name, value))
        if self.dense_ownership is not None:
            tensors.extend(
                (f"dense ownership {index}", value)
                for index, value in enumerate(self.dense_ownership)
            )
        sizes: set[int] = set()
        for name, value in tensors:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.ndim == 0:
                raise ValueError(f"{name} must expose a batch dimension")
            if value.shape[0] <= 0:
                raise ValueError(f"{name} batch dimension must be nonempty")
            sizes.add(value.shape[0])
        if len(sizes) > 1:
            raise ValueError("dense, ownership and object evidence must share one batch size")
        return next(iter(sizes), None)

    def ownership_weighted_addresses(
        self,
        *,
        validate_tensor_values: bool = True,
    ) -> tuple[torch.Tensor | None, ...]:
        """Return one persistent-address mixture per native dense token.

        Ownership is optional so the full-evidence/no-posterior control can use
        exactly the same dense path. When present, its last category is context
        and contributes no fabricated object address. Jointly permuting object
        rows and ownership columns leaves the result unchanged.
        """

        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")
        self.batch_size()
        self.validate_object_identity(validate_tensor_values=validate_tensor_values)
        if self.dense_ownership is None:
            return tuple(None for _ in self.dense_banks)
        if self.object_address is None or self.object_valid is None:
            raise ValueError("dense ownership requires an object address bank")
        if len(self.dense_ownership) != len(self.dense_banks):
            raise ValueError("dense ownership must align one-to-one with dense banks")
        if (
            self.object_address.ndim != 3
            or self.object_valid.shape != self.object_address.shape[:2]
        ):
            raise ValueError("object address and validity must align batch-by-object")
        if self.object_valid.dtype != torch.bool:
            raise ValueError("object validity must be boolean")

        object_count = self.object_address.shape[1]
        output = []
        for bank, ownership in zip(self.dense_banks, self.dense_ownership, strict=True):
            expected = (*bank.tokens.shape[:2], object_count + 1)
            if ownership.shape != expected:
                raise ValueError(f"dense ownership for {bank.modality} must have shape {expected}")
            if not torch.is_floating_point(ownership):
                raise ValueError(f"dense ownership for {bank.modality} must be floating")
            if ownership.device != bank.tokens.device or ownership.dtype != bank.tokens.dtype:
                raise ValueError(
                    f"dense ownership for {bank.modality} must match its token dtype/device"
                )
            if (
                self.object_address.device != ownership.device
                or self.object_address.dtype != ownership.dtype
                or self.object_valid.device != ownership.device
            ):
                raise ValueError("object bank and dense ownership must share dtype/device")
            if validate_tensor_values:
                if not torch.isfinite(ownership).all():
                    raise ValueError(f"dense ownership for {bank.modality} must be finite floating")
                if (ownership < 0.0).any():
                    raise ValueError(f"dense ownership for {bank.modality} cannot be negative")
                tolerance = max(1e-5, torch.finfo(ownership.dtype).eps)
                if not torch.allclose(
                    ownership.float().sum(dim=-1),
                    torch.ones_like(ownership[..., 0], dtype=torch.float32),
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    raise ValueError(
                        f"dense ownership for {bank.modality} must sum to one with context"
                    )
                invalid_object_mass = ownership[..., :-1].masked_select(
                    ~self.object_valid.unsqueeze(1)
                )
                if invalid_object_mass.numel() and (invalid_object_mass != 0.0).any():
                    raise ValueError(
                        f"dense ownership for {bank.modality} assigns mass to unused objects"
                    )
                if (~bank.valid).any():
                    invalid_rows = ownership[~bank.valid]
                    if (invalid_rows[..., :-1] != 0.0).any() or (
                        invalid_rows[..., -1] != 1.0
                    ).any():
                        raise ValueError(
                            f"invalid {bank.modality} tokens must belong exactly to context"
                        )
            output.append(torch.einsum("bnk,bkd->bnd", ownership[..., :-1], self.object_address))
        return tuple(output)

    def validate_object_identity(self, *, validate_tensor_values: bool = True) -> None:
        """Validate the shared spherical identity-key contract once per host call."""

        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")
        self.batch_size()
        if self.object_address is None:
            return
        if self.object_valid is None or self.object_log_prior is None:
            raise RuntimeError("object evidence violated its validated atomic-field invariant")
        if self.object_address.ndim != 3:
            raise ValueError("object address must be rank three")
        if self.object_address.shape[-1] <= 0:
            raise ValueError("object address width must be positive")
        if (
            self.object_valid.dtype != torch.bool
            or self.object_valid.shape != self.object_address.shape[:2]
        ):
            raise ValueError("object validity must be bool batch-by-object")
        if not torch.is_floating_point(self.object_address):
            raise ValueError("object address must be floating point")
        if (
            self.object_log_prior.shape != self.object_valid.shape
            or not torch.is_floating_point(self.object_log_prior)
            or self.object_log_prior.device != self.object_address.device
            or self.object_log_prior.dtype != self.object_address.dtype
        ):
            raise ValueError(
                "object log prior must be floating batch-by-object with the address dtype/device"
            )
        if not validate_tensor_values:
            return
        if not torch.isfinite(self.object_address).all():
            raise ValueError("object address contains NaN or infinity")
        if (self.object_address[~self.object_valid] != 0.0).any():
            raise ValueError("unused object addresses must be exactly zero")
        if not torch.isfinite(self.object_log_prior).all():
            raise ValueError("object log prior contains NaN or infinity")
        if (self.object_log_prior[self.object_valid] > 0.0).any():
            raise ValueError("valid object log prior cannot exceed log probability zero")
        if (self.object_log_prior[~self.object_valid] != 0.0).any():
            raise ValueError("unused object log prior must be exactly zero")
        valid_addresses = self.object_address[self.object_valid]
        tolerance = max(1e-5, torch.finfo(self.object_address.dtype).eps)
        if valid_addresses.numel() and not torch.allclose(
            torch.linalg.vector_norm(valid_addresses.float(), dim=-1),
            torch.ones(
                valid_addresses.shape[0],
                dtype=torch.float32,
                device=valid_addresses.device,
            ),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("valid object addresses must have unit norm")
