"""Pure diagnostic interventions over host-neutral PICF action evidence.

These transformations are evaluation controls, not trainable model branches.
They make action adoption falsifiable without changing dense observations or
the host action convention.
"""

from __future__ import annotations

import torch

from .context import PICFActionEvidence


def without_posterior(evidence: PICFActionEvidence) -> PICFActionEvidence:
    """Keep complete dense evidence while removing all posterior structure."""

    return PICFActionEvidence(
        dense_banks=evidence.dense_banks,
        object_address=None,
        object_value=None,
        object_valid=None,
        object_log_prior=None,
        dense_ownership=None,
    )


def without_object_rows(
    evidence: PICFActionEvidence,
    rows: torch.Tensor,
) -> PICFActionEvidence:
    """Remove selected object rows while preserving every native dense token.

    Ownership previously assigned to a removed row is transferred exactly to
    context. This is the row-level analogue of ``without_posterior`` and is only
    an evaluation intervention; it does not mask or alter sensor content.
    """

    if (
        evidence.object_address is None
        or evidence.object_value is None
        or evidence.object_valid is None
        or evidence.object_log_prior is None
    ):
        raise ValueError("object-row removal requires a complete object bank")
    if rows.dtype != torch.bool or rows.shape != evidence.object_valid.shape:
        raise ValueError("rows must be a boolean batch-by-object tensor")
    if rows.device != evidence.object_valid.device:
        raise ValueError("rows and object bank must share a device")

    removed = rows & evidence.object_valid
    retained = ~removed
    ownership = evidence.dense_ownership
    if ownership is not None:
        updated = []
        for item in ownership:
            retained_columns = retained.unsqueeze(1).to(dtype=item.dtype)
            removed_columns = removed.unsqueeze(1).to(dtype=item.dtype)
            retained_mass = item[..., :-1] * retained_columns
            removed_mass = (item[..., :-1] * removed_columns).sum(
                dim=-1,
                keepdim=True,
                dtype=item.dtype,
            )
            updated.append(
                torch.cat((retained_mass, item[..., -1:] + removed_mass), dim=-1).to(
                    dtype=item.dtype
                )
            )
        ownership = tuple(updated)

    result = PICFActionEvidence(
        dense_banks=evidence.dense_banks,
        object_address=evidence.object_address * retained.unsqueeze(-1),
        object_value=evidence.object_value * retained.unsqueeze(-1),
        object_valid=evidence.object_valid & retained,
        object_log_prior=evidence.object_log_prior * retained,
        dense_ownership=ownership,
    )
    result.ownership_weighted_addresses()
    return result


def _validate_object_permutation(
    evidence: PICFActionEvidence,
    permutation: torch.Tensor,
) -> None:
    if (
        evidence.object_address is None
        or evidence.object_value is None
        or evidence.object_log_prior is None
        or evidence.object_valid is None
    ):
        raise ValueError("object-row intervention requires a complete object bank")
    object_count = evidence.object_address.shape[1]
    if permutation.dtype != torch.long or permutation.shape != (object_count,):
        raise ValueError("permutation must be a long vector covering object capacity")
    if permutation.device != evidence.object_address.device:
        raise ValueError("permutation and object bank must share a device")
    if not torch.equal(
        torch.sort(permutation).values,
        torch.arange(object_count, device=permutation.device),
    ):
        raise ValueError("object-row intervention requires a bijective permutation")


def permute_object_rows(
    evidence: PICFActionEvidence,
    permutation: torch.Tensor,
    *,
    keep_ownership_fixed: bool,
) -> PICFActionEvidence:
    """Permute object rows jointly or break address/ownership correspondence.

    A joint permutation is a mathematical negative control and must not alter
    action.  Keeping ownership fixed is the wrong-address intervention: native
    token contents stay unchanged while their persistent identity coordinate is
    deliberately rebound to another object state.
    """

    _validate_object_permutation(evidence, permutation)
    if evidence.object_address is None:
        raise RuntimeError("validated object evidence lost its address bank")

    ownership = evidence.dense_ownership
    if ownership is not None and not keep_ownership_fixed:
        ownership = tuple(
            torch.cat((item[..., :-1][..., permutation], item[..., -1:]), dim=-1)
            for item in ownership
        )
    return PICFActionEvidence(
        dense_banks=evidence.dense_banks,
        object_address=evidence.object_address[:, permutation],
        object_value=evidence.object_value[:, permutation],
        object_valid=evidence.object_valid[:, permutation],
        object_log_prior=evidence.object_log_prior[:, permutation],
        dense_ownership=ownership,
    )


def permute_object_addresses(
    evidence: PICFActionEvidence,
    permutation: torch.Tensor,
) -> PICFActionEvidence:
    """Break persistent identity-to-state correspondence without changing rows.

    Unlike :func:`permute_object_rows`, this intervention permutes only the
    spherical identity key. Dynamic value, validity, prior and dense ownership
    stay on their original rows. It is therefore identifiable even for a hidden
    row with no current dense ownership. This is an evaluation control only.
    """

    _validate_object_permutation(evidence, permutation)
    if (
        evidence.object_address is None
        or evidence.object_value is None
        or evidence.object_valid is None
        or evidence.object_log_prior is None
    ):
        raise RuntimeError("validated object evidence lost a required field")
    return PICFActionEvidence(
        dense_banks=evidence.dense_banks,
        object_address=evidence.object_address[:, permutation],
        object_value=evidence.object_value,
        object_valid=evidence.object_valid,
        object_log_prior=evidence.object_log_prior,
        dense_ownership=evidence.dense_ownership,
    )


def stale_posterior(
    current: PICFActionEvidence,
    stale: PICFActionEvidence,
) -> PICFActionEvidence:
    """Pair current dense observations with a previous posterior snapshot."""

    if (
        stale.object_address is None
        or stale.object_value is None
        or stale.object_valid is None
        or stale.object_log_prior is None
    ):
        raise ValueError("stale intervention requires a complete previous object bank")
    if current.dense_ownership is not None and stale.dense_ownership is None:
        raise ValueError("structured current evidence requires stale ownership metadata")
    if len(current.dense_banks) != len(stale.dense_banks):
        raise ValueError("current and stale evidence must expose the same modality banks")
    for current_bank, stale_bank in zip(current.dense_banks, stale.dense_banks, strict=True):
        if current_bank.modality != stale_bank.modality:
            raise ValueError("current and stale modality order must match")
        if current_bank.tokens.shape[:2] != stale_bank.tokens.shape[:2]:
            raise ValueError("current and stale token geometry must match")
    return PICFActionEvidence(
        dense_banks=current.dense_banks,
        object_address=stale.object_address,
        object_value=stale.object_value,
        object_valid=stale.object_valid,
        object_log_prior=stale.object_log_prior,
        dense_ownership=stale.dense_ownership,
    )
