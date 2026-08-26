"""Label-closed causal contracts for LTOP action mediation.

This module contains no model and no trainable parameter.  It defines the
evaluation arms that perturb the production ``OBJECT_READ`` path, seals
deploy-input-only action receipts, and performs target-aware scoring only in a
separate offline function.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch

from picf_next.lingbot_native.host import ObjectReadActionIntervention


class LTOPActionArmKind(str, Enum):
    FACTUAL = "factual"
    ROW_REMOVAL = "row-removal"
    BLOCKED = "blocked"
    BLOCKED_ROW_REMOVAL = "blocked-row-removal"


@dataclass(frozen=True, slots=True)
class LTOPActionMediationArm:
    """One label-blind intervention on the production LTOP causal graph."""

    name: str
    kind: LTOPActionArmKind
    object_read_action_intervention: ObjectReadActionIntervention
    object_read_source_row_visible: torch.Tensor
    row_index: int | None = None

    def __post_init__(self) -> None:
        visible = self.object_read_source_row_visible
        if not self.name:
            raise ValueError("LTOP action arm requires a non-empty name")
        if not isinstance(self.kind, LTOPActionArmKind):
            raise TypeError("LTOP action arm kind must be parsed")
        if not isinstance(
            self.object_read_action_intervention,
            ObjectReadActionIntervention,
        ):
            raise TypeError("LTOP action arm intervention must be parsed")
        if (
            not isinstance(visible, torch.Tensor)
            or visible.ndim != 2
            or visible.dtype != torch.bool
            or visible.requires_grad
            or visible.shape[0] <= 0
            or visible.shape[1] <= 0
        ):
            raise ValueError("LTOP action arm visibility must be bool [batch,capacity]")
        if self.row_index is not None and not 0 <= self.row_index < visible.shape[1]:
            raise ValueError("LTOP action arm row index lies outside capacity")
        removed = (~visible).sum(dim=1)
        if self.kind in {LTOPActionArmKind.FACTUAL, LTOPActionArmKind.BLOCKED}:
            if self.row_index is not None or bool(removed.any().item()):
                raise ValueError("factual/blocked LTOP action arm must expose every source row")
        else:
            if self.row_index is None or not bool((removed == 1).all().item()):
                raise ValueError("row-removal LTOP action arm must remove exactly one row")
            if bool(visible[:, self.row_index].any().item()):
                raise ValueError("LTOP action arm did not remove its declared row")
        expected_intervention = (
            ObjectReadActionIntervention.BLOCKED
            if self.kind in {LTOPActionArmKind.BLOCKED, LTOPActionArmKind.BLOCKED_ROW_REMOVAL}
            else ObjectReadActionIntervention.FACTUAL
        )
        if self.object_read_action_intervention is not expected_intervention:
            raise ValueError("LTOP action arm kind and edge intervention disagree")


def build_label_blind_ltop_action_arms(
    *,
    batch_size: int,
    capacity: int,
    device: torch.device | str,
) -> tuple[LTOPActionMediationArm, ...]:
    """Enumerate factual, every-row removal, and blocked-path controls."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("LTOP action arms require a positive batch size")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 1:
        raise ValueError("LTOP action arms require capacity greater than one")
    factual_visible = torch.ones((batch_size, capacity), dtype=torch.bool, device=device)
    arms = [
        LTOPActionMediationArm(
            name="factual",
            kind=LTOPActionArmKind.FACTUAL,
            object_read_action_intervention=ObjectReadActionIntervention.FACTUAL,
            object_read_source_row_visible=factual_visible.clone(),
        ),
        LTOPActionMediationArm(
            name="factual-repeat",
            kind=LTOPActionArmKind.FACTUAL,
            object_read_action_intervention=ObjectReadActionIntervention.FACTUAL,
            object_read_source_row_visible=factual_visible.clone(),
        ),
    ]
    for row_index in range(capacity):
        visible = factual_visible.clone()
        visible[:, row_index] = False
        arms.append(
            LTOPActionMediationArm(
                name=f"remove-row-{row_index}",
                kind=LTOPActionArmKind.ROW_REMOVAL,
                object_read_action_intervention=ObjectReadActionIntervention.FACTUAL,
                object_read_source_row_visible=visible,
                row_index=row_index,
            )
        )
    arms.append(
        LTOPActionMediationArm(
            name="blocked",
            kind=LTOPActionArmKind.BLOCKED,
            object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
            object_read_source_row_visible=factual_visible.clone(),
        )
    )
    for row_index in range(capacity):
        visible = factual_visible.clone()
        visible[:, row_index] = False
        arms.append(
            LTOPActionMediationArm(
                name=f"blocked-remove-row-{row_index}",
                kind=LTOPActionArmKind.BLOCKED_ROW_REMOVAL,
                object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
                object_read_source_row_visible=visible,
                row_index=row_index,
            )
        )
    return tuple(arms)


def direct_posterior_action_row_visibility(
    arm: LTOPActionMediationArm,
) -> torch.Tensor:
    """Map the registered causal arms onto the direct posterior-row surface.

    Factual and row-removal arms expose the declared posterior rows. Blocked
    controls remove every posterior row while retaining the row-removal label
    only as an execution-integrity placebo.
    """

    if not isinstance(arm, LTOPActionMediationArm):
        raise TypeError("direct posterior visibility requires a typed LTOP action arm")
    visible = arm.object_read_source_row_visible
    if arm.kind in {LTOPActionArmKind.BLOCKED, LTOPActionArmKind.BLOCKED_ROW_REMOVAL}:
        return torch.zeros_like(visible)
    return visible.detach().clone()


def _tensor_sha256(value: torch.Tensor) -> str:
    local = value.detach().contiguous().cpu()
    header = f"{local.dtype}|{tuple(local.shape)}|".encode()
    return hashlib.sha256(header + local.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class LTOPActionReceipt:
    """One sealed action result produced without target metadata."""

    prompt_name: str
    sample_keys: tuple[str, ...]
    arm_name: str
    arm_kind: LTOPActionArmKind
    row_index: int | None
    deploy_inputs_sha256: str
    inference_randomness_sha256: str
    source_visibility_sha256: str
    active_action_mask: torch.Tensor
    active_action_mask_sha256: str
    action_output: torch.Tensor
    action_output_sha256: str

    def __post_init__(self) -> None:
        if not self.prompt_name or not self.arm_name:
            raise ValueError("LTOP action receipt requires prompt and arm names")
        if not self.sample_keys or any(not value for value in self.sample_keys):
            raise ValueError("LTOP action receipt requires non-empty sample keys")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ValueError("LTOP action receipt sample keys must be unique")
        for name, value in (
            ("deploy inputs", self.deploy_inputs_sha256),
            ("inference randomness", self.inference_randomness_sha256),
            ("source visibility", self.source_visibility_sha256),
            ("active action mask", self.active_action_mask_sha256),
            ("action output", self.action_output_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"LTOP action receipt {name} digest is not SHA-256")
        output = self.action_output
        if (
            not isinstance(output, torch.Tensor)
            or output.shape[0] != len(self.sample_keys)
            or output.ndim < 2
            or not output.is_floating_point()
            or output.requires_grad
            or not bool(torch.isfinite(output).all().item())
        ):
            raise ValueError("LTOP action receipt output is invalid")
        if self.action_output_sha256 != _tensor_sha256(output):
            raise ValueError("LTOP action receipt output digest differs")
        active = self.active_action_mask
        if (
            not isinstance(active, torch.Tensor)
            or active.shape != output.shape
            or active.dtype != torch.bool
            or active.device != output.device
            or active.requires_grad
        ):
            raise ValueError("LTOP active action mask must be bool and match the action output")
        if bool((active.flatten(1).sum(dim=1) == 0).any().item()):
            raise ValueError("every LTOP action receipt sample requires an active action element")
        if self.active_action_mask_sha256 != _tensor_sha256(active):
            raise ValueError("LTOP action receipt active action mask digest differs")
        if self.arm_kind in {LTOPActionArmKind.FACTUAL, LTOPActionArmKind.BLOCKED}:
            if self.row_index is not None:
                raise ValueError("factual/blocked LTOP action receipt cannot name one row")
        elif self.row_index is None:
            raise ValueError("row-removal LTOP action receipt must name one row")


def seal_ltop_action_receipt(
    *,
    prompt_name: str,
    sample_keys: Sequence[str],
    arm: LTOPActionMediationArm,
    deploy_inputs_sha256: str,
    inference_randomness_sha256: str,
    action_output: torch.Tensor,
    joint_mask: torch.Tensor,
    action_is_pad: torch.Tensor,
    executed_source_row_visible: torch.Tensor | None = None,
) -> LTOPActionReceipt:
    """Detach and seal one label-free production action observation."""

    if not isinstance(arm, LTOPActionMediationArm):
        raise TypeError("LTOP action receipt requires a typed arm")
    if not isinstance(action_output, torch.Tensor):
        raise TypeError("LTOP action receipt output must be a tensor")
    output = action_output.detach().clone()
    if (
        not isinstance(joint_mask, torch.Tensor)
        or joint_mask.shape != output.shape
        or joint_mask.dtype != torch.bool
        or joint_mask.device != output.device
        or joint_mask.requires_grad
    ):
        raise ValueError("LTOP joint mask must be bool and match the action output")
    if (
        not isinstance(action_is_pad, torch.Tensor)
        or action_is_pad.shape != output.shape[:-1]
        or action_is_pad.dtype != torch.bool
        or action_is_pad.device != output.device
        or action_is_pad.requires_grad
    ):
        raise ValueError("LTOP action padding mask must be bool and match action leading axes")
    active_action_mask = (joint_mask & ~action_is_pad[..., None]).detach().clone()
    if bool((active_action_mask.flatten(1).sum(dim=1) == 0).any().item()):
        raise ValueError("every LTOP action receipt sample requires an active action element")
    source_row_visible = (
        direct_posterior_action_row_visibility(arm)
        if executed_source_row_visible is None
        else executed_source_row_visible
    )
    if (
        not isinstance(source_row_visible, torch.Tensor)
        or source_row_visible.shape != arm.object_read_source_row_visible.shape
        or source_row_visible.dtype != torch.bool
        or source_row_visible.device != arm.object_read_source_row_visible.device
        or source_row_visible.requires_grad
    ):
        raise ValueError("executed LTOP source visibility must be bool [batch,capacity]")
    return LTOPActionReceipt(
        prompt_name=prompt_name,
        sample_keys=tuple(sample_keys),
        arm_name=arm.name,
        arm_kind=arm.kind,
        row_index=arm.row_index,
        deploy_inputs_sha256=deploy_inputs_sha256,
        inference_randomness_sha256=inference_randomness_sha256,
        source_visibility_sha256=_tensor_sha256(source_row_visible),
        active_action_mask=active_action_mask,
        active_action_mask_sha256=_tensor_sha256(active_action_mask),
        action_output=output,
        action_output_sha256=_tensor_sha256(output),
    )


@dataclass(frozen=True, slots=True)
class OfflineLTOPActionTargets:
    """Target metadata loaded only after all forward receipts are sealed."""

    prompt_name: str
    sample_keys: tuple[str, ...]
    target_rows: torch.Tensor
    matched_distractor_rows: torch.Tensor

    def __post_init__(self) -> None:
        batch = len(self.sample_keys)
        if not self.prompt_name or batch <= 0 or len(set(self.sample_keys)) != batch:
            raise ValueError("offline LTOP targets require unique sample keys and a prompt")
        for name, value in (
            ("target", self.target_rows),
            ("matched distractor", self.matched_distractor_rows),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (batch,)
                or value.dtype != torch.long
                or value.requires_grad
            ):
                raise ValueError(f"offline LTOP {name} rows must be int64 [batch]")
        if self.target_rows.device != self.matched_distractor_rows.device:
            raise ValueError("offline LTOP row labels must share a device")
        if bool((self.target_rows == self.matched_distractor_rows).any().item()):
            raise ValueError("offline LTOP target and distractor rows must differ")


@dataclass(frozen=True, slots=True)
class LTOPActionMediationScore:
    prompt_name: str
    sample_keys: tuple[str, ...]
    active_action_counts: torch.Tensor
    blocked_placebo_integrity_verified: bool
    replay_floor_rms: torch.Tensor
    factual_all_posterior_block_effect_rms: torch.Tensor
    factual_target_effect_rms: torch.Tensor
    factual_distractor_effect_rms: torch.Tensor
    factual_target_minus_distractor: torch.Tensor
    factual_target_effect_over_all_posterior_block: torch.Tensor
    factual_distractor_effect_over_all_posterior_block: torch.Tensor
    factual_selectivity_over_all_posterior_block: torch.Tensor
    mean_factual_all_posterior_block_effect_rms: float
    mean_factual_target_minus_distractor: float
    mean_factual_selectivity_over_all_posterior_block: float


def _active_rms_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    active_action_mask: torch.Tensor,
) -> torch.Tensor:
    if left.shape != right.shape or left.device != right.device:
        raise ValueError("LTOP action receipt velocities have incompatible shapes/devices")
    if (
        active_action_mask.shape != left.shape
        or active_action_mask.dtype != torch.bool
        or active_action_mask.device != left.device
    ):
        raise ValueError("LTOP active action mask differs from the scored action surface")
    active_counts = active_action_mask.flatten(1).sum(dim=1)
    if bool((active_counts == 0).any().item()):
        raise ValueError("every scored LTOP sample requires an active action element")
    squared_difference = (left.float() - right.float()).square()
    squared_difference = squared_difference.masked_fill(~active_action_mask, 0.0)
    return (squared_difference.flatten(1).sum(dim=1) / active_counts.float()).sqrt()


def _zero_safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return a finite diagnostic ratio, with zero reserved for an exact zero denominator."""

    nonzero = denominator != 0
    safe_denominator = torch.where(nonzero, denominator, torch.ones_like(denominator))
    return torch.where(nonzero, numerator / safe_denominator, torch.zeros_like(numerator))


def score_offline_ltop_action_mediation(
    receipts: Sequence[LTOPActionReceipt],
    *,
    targets: OfflineLTOPActionTargets,
    capacity: int,
) -> LTOPActionMediationScore:
    """Score active-action posterior effects after the forward boundary closes.

    The blocked row-labelled arms are integrity placebos under the direct
    all-posterior-hidden semantics. This function reports raw effects and
    dimensionless diagnostics; acceptance thresholds belong to the caller.
    """

    receipt_items = tuple(receipts)
    if not receipt_items:
        raise ValueError("offline LTOP scoring requires sealed receipts")
    if any(not isinstance(item, LTOPActionReceipt) for item in receipt_items):
        raise TypeError("offline LTOP scoring requires typed action receipts")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 1:
        raise ValueError("offline LTOP scoring requires capacity greater than one")
    selected = tuple(item for item in receipt_items if item.prompt_name == targets.prompt_name)
    if not selected or any(item.sample_keys != targets.sample_keys for item in selected):
        raise ValueError("offline LTOP targets do not match receipt sample provenance")
    deploy_digests = {item.deploy_inputs_sha256 for item in selected}
    randomness_digests = {item.inference_randomness_sha256 for item in selected}
    if len(deploy_digests) != 1 or len(randomness_digests) != 1:
        raise ValueError("LTOP action arms did not share deploy inputs and fixed randomness")
    table = {item.arm_name: item for item in selected}
    if len(table) != len(selected):
        raise ValueError("offline LTOP action receipts contain duplicate arm names")
    required = {"factual", "factual-repeat", "blocked"}
    required.update(f"remove-row-{row}" for row in range(capacity))
    required.update(f"blocked-remove-row-{row}" for row in range(capacity))
    missing = sorted(required - set(table))
    extra = sorted(set(table) - required)
    if missing or extra:
        raise ValueError(
            f"offline LTOP action receipt arms differ: missing={missing}, extra={extra}"
        )

    expected_metadata = {
        "factual": (LTOPActionArmKind.FACTUAL, None),
        "factual-repeat": (LTOPActionArmKind.FACTUAL, None),
        "blocked": (LTOPActionArmKind.BLOCKED, None),
    }
    expected_metadata.update(
        {f"remove-row-{row}": (LTOPActionArmKind.ROW_REMOVAL, row) for row in range(capacity)}
    )
    expected_metadata.update(
        {
            f"blocked-remove-row-{row}": (LTOPActionArmKind.BLOCKED_ROW_REMOVAL, row)
            for row in range(capacity)
        }
    )
    batch_size = len(targets.sample_keys)
    factual_visibility = torch.ones((batch_size, capacity), dtype=torch.bool)
    blocked_visibility_sha256 = _tensor_sha256(torch.zeros_like(factual_visibility))
    for name, item in table.items():
        expected_kind, expected_row = expected_metadata[name]
        if item.arm_kind is not expected_kind or item.row_index != expected_row:
            raise ValueError(f"offline LTOP action receipt metadata differs for {name}")
        if item.action_output_sha256 != _tensor_sha256(item.action_output):
            raise ValueError(f"offline LTOP action receipt output digest differs for {name}")
        if item.active_action_mask_sha256 != _tensor_sha256(item.active_action_mask):
            raise ValueError(f"offline LTOP active action mask digest differs for {name}")
        expected_visibility = factual_visibility.clone()
        if expected_kind in {
            LTOPActionArmKind.BLOCKED,
            LTOPActionArmKind.BLOCKED_ROW_REMOVAL,
        }:
            expected_visibility.zero_()
        elif expected_row is not None:
            expected_visibility[:, expected_row] = False
        if item.source_visibility_sha256 != _tensor_sha256(expected_visibility):
            if expected_kind in {
                LTOPActionArmKind.BLOCKED,
                LTOPActionArmKind.BLOCKED_ROW_REMOVAL,
            }:
                raise ValueError(f"offline LTOP blocked placebo visibility differs for {name}")
            raise ValueError(f"offline LTOP executed source visibility differs for {name}")

    factual_receipt = table["factual"]
    active_action_mask = factual_receipt.active_action_mask
    for name, item in table.items():
        if (
            item.active_action_mask_sha256 != factual_receipt.active_action_mask_sha256
            or item.active_action_mask.shape != active_action_mask.shape
            or item.active_action_mask.device != active_action_mask.device
            or not torch.equal(item.active_action_mask, active_action_mask)
        ):
            raise ValueError(f"offline LTOP active action masks differ for {name}")

    blocked_receipt = table["blocked"]
    if blocked_receipt.source_visibility_sha256 != blocked_visibility_sha256:
        raise ValueError("offline LTOP blocked placebo visibility is not all-posterior-hidden")
    for row in range(capacity):
        placebo = table[f"blocked-remove-row-{row}"]
        if placebo.source_visibility_sha256 != blocked_receipt.source_visibility_sha256:
            raise ValueError("offline LTOP blocked placebo visibility differs across row labels")
        if placebo.action_output_sha256 != blocked_receipt.action_output_sha256:
            raise ValueError("offline LTOP blocked placebo output differs across row labels")

    factual = factual_receipt.action_output
    repeat = table["factual-repeat"].action_output
    blocked = blocked_receipt.action_output
    active_action_counts = active_action_mask.flatten(1).sum(dim=1)
    replay_floor = _active_rms_difference(factual, repeat, active_action_mask)
    all_posterior_block = _active_rms_difference(factual, blocked, active_action_mask)
    factual_effects = torch.stack(
        [
            _active_rms_difference(
                factual,
                table[f"remove-row-{row}"].action_output,
                active_action_mask,
            )
            for row in range(capacity)
        ],
        dim=1,
    )
    target_rows = targets.target_rows.to(device=factual_effects.device)
    distractor_rows = targets.matched_distractor_rows.to(device=factual_effects.device)
    target_invalid = ((target_rows < 0) | (target_rows >= capacity)).any()
    distractor_invalid = ((distractor_rows < 0) | (distractor_rows >= capacity)).any()
    if bool((target_invalid | distractor_invalid).item()):
        raise ValueError("offline LTOP rows lie outside capacity")
    factual_target = factual_effects.gather(1, target_rows[:, None]).squeeze(1)
    factual_distractor = factual_effects.gather(1, distractor_rows[:, None]).squeeze(1)
    factual_delta = factual_target - factual_distractor
    target_over_all_block = _zero_safe_ratio(factual_target, all_posterior_block)
    distractor_over_all_block = _zero_safe_ratio(factual_distractor, all_posterior_block)
    selectivity_over_all_block = _zero_safe_ratio(factual_delta, all_posterior_block)
    return LTOPActionMediationScore(
        prompt_name=targets.prompt_name,
        sample_keys=targets.sample_keys,
        active_action_counts=active_action_counts.detach().clone(),
        blocked_placebo_integrity_verified=True,
        replay_floor_rms=replay_floor.detach().clone(),
        factual_all_posterior_block_effect_rms=all_posterior_block.detach().clone(),
        factual_target_effect_rms=factual_target.detach().clone(),
        factual_distractor_effect_rms=factual_distractor.detach().clone(),
        factual_target_minus_distractor=factual_delta.detach().clone(),
        factual_target_effect_over_all_posterior_block=target_over_all_block.detach().clone(),
        factual_distractor_effect_over_all_posterior_block=(
            distractor_over_all_block.detach().clone()
        ),
        factual_selectivity_over_all_posterior_block=(selectivity_over_all_block.detach().clone()),
        mean_factual_all_posterior_block_effect_rms=float(all_posterior_block.mean().item()),
        mean_factual_target_minus_distractor=float(factual_delta.mean().item()),
        mean_factual_selectivity_over_all_posterior_block=float(
            selectivity_over_all_block.mean().item()
        ),
    )
