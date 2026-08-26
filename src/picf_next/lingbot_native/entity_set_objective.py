"""Task-independent current-frame entity matching and proper set losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

SAM31_SOURCE_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"


def sam31_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Unreduced SAM 3.1/RetinaNet focal loss, without its Triton wrapper."""

    if inputs.shape != targets.shape:
        raise ValueError("focal inputs and targets must have identical shapes")
    if not inputs.is_floating_point() or not targets.is_floating_point():
        raise TypeError("focal inputs and targets must be floating point")
    if not 0 <= alpha <= 1 or gamma < 0:
        raise ValueError("focal alpha and gamma are outside their valid range")
    probability = inputs.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    target_probability = probability * targets + (1 - probability) * (1 - targets)
    loss = cross_entropy * ((1 - target_probability) ** gamma)
    alpha_weight = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_weight * loss


def sam31_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """SAM 3.1 Dice loss over the final axis, retaining leading axes."""

    if inputs.shape != targets.shape or inputs.ndim < 1:
        raise ValueError("dice inputs and targets must have one identical non-scalar shape")
    if not inputs.is_floating_point() or not targets.is_floating_point():
        raise TypeError("dice inputs and targets must be floating point")
    if weight is None:
        weight = torch.ones_like(targets)
    if (
        weight.shape != targets.shape
        or not weight.is_floating_point()
        or not torch.isfinite(weight).all()
        or (weight < 0).any()
    ):
        raise ValueError("dice weight must be finite, non-negative and match targets")
    probability = inputs.sigmoid()
    numerator = 2 * (probability * targets * weight).sum(dim=-1)
    denominator = (probability * weight).sum(dim=-1) + (targets * weight).sum(dim=-1)
    return 1 - (numerator + 1) / (denominator + 1)


@dataclass(frozen=True, slots=True)
class PhysicalFramePredictions:
    """One frame of prompt-free physical entity predictions."""

    support_logits: torch.Tensor
    ownership_log_probability: torch.Tensor
    existence_logits: torch.Tensor
    sensor_valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.support_logits.ndim != 3:
            raise ValueError("support logits must have shape [batch,tokens,rows]")
        batch, tokens, rows = self.support_logits.shape
        if self.ownership_log_probability.shape != (batch, tokens, rows + 1):
            raise ValueError("ownership log probabilities must append one context class")
        if self.existence_logits.shape != (batch, rows):
            raise ValueError("existence logits must have shape [batch,rows]")
        if self.sensor_valid.shape != (batch, tokens) or self.sensor_valid.dtype != torch.bool:
            raise ValueError("sensor validity must be boolean [batch,tokens]")
        tensors = (
            self.support_logits,
            self.ownership_log_probability,
            self.existence_logits,
        )
        if any(
            not value.is_floating_point()
            or not torch.isfinite(value).all()
            or value.device != self.support_logits.device
            for value in tensors
        ):
            raise ValueError("physical predictions must be finite floating tensors on one device")
        if self.sensor_valid.device != self.support_logits.device:
            raise ValueError("sensor validity and predictions must share one device")
        active = self.sensor_valid
        tolerance = max(1e-5, 2 * torch.finfo(self.ownership_log_probability.dtype).eps)
        if not torch.allclose(
            torch.logsumexp(self.ownership_log_probability.float(), dim=-1)[active],
            torch.zeros_like(self.ownership_log_probability[..., 0].float()[active]),
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError("active ownership log probabilities must form a simplex")


@dataclass(frozen=True, slots=True)
class PhysicalFrameTargets:
    """Loss-only visible-instance targets with explicit unknown evidence."""

    masks: torch.Tensor
    mask_valid: torch.Tensor
    existence: torch.Tensor
    existence_valid: torch.Tensor
    track_valid: torch.Tensor
    capacity_censored: torch.Tensor
    token_observed_fraction: torch.Tensor
    inventory_exhaustive: torch.Tensor
    token_measure_weight: torch.Tensor | None = None
    exclusive_ownership: bool = False

    def __post_init__(self) -> None:
        if self.masks.ndim != 3:
            raise ValueError("physical masks must have shape [batch,tracks,tokens]")
        batch, tracks, tokens = self.masks.shape
        if (
            not self.masks.is_floating_point()
            or not torch.isfinite(self.masks).all()
            or ((self.masks < 0) | (self.masks > 1)).any()
            or self.masks.requires_grad
        ):
            raise ValueError("physical masks must be detached finite probabilities")
        expected_boolean = {
            "mask_valid": (self.mask_valid, (batch, tracks, tokens)),
            "existence_valid": (self.existence_valid, (batch, tracks)),
            "track_valid": (self.track_valid, (batch, tracks)),
            "capacity_censored": (self.capacity_censored, (batch, tracks)),
            "inventory_exhaustive": (self.inventory_exhaustive, (batch,)),
        }
        for name, (value, shape) in expected_boolean.items():
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape {shape}")
        if (
            self.existence.shape != (batch, tracks)
            or not self.existence.is_floating_point()
            or not torch.isfinite(self.existence).all()
            or ((self.existence < 0) | (self.existence > 1)).any()
            or self.existence.requires_grad
        ):
            raise ValueError("existence targets must be detached finite probabilities")
        if (
            self.token_observed_fraction.shape != (batch, tokens)
            or not self.token_observed_fraction.is_floating_point()
            or not torch.isfinite(self.token_observed_fraction).all()
            or ((self.token_observed_fraction < 0) | (self.token_observed_fraction > 1)).any()
            or self.token_observed_fraction.requires_grad
        ):
            raise ValueError("observed token fractions must be detached values in [0,1]")
        if self.token_measure_weight is not None and (
            self.token_measure_weight.shape != (batch, tokens)
            or not self.token_measure_weight.is_floating_point()
            or not torch.isfinite(self.token_measure_weight).all()
            or (self.token_measure_weight < 0).any()
            or self.token_measure_weight.requires_grad
        ):
            raise ValueError("physical token measure must be detached finite and non-negative")
        tensors = (
            self.mask_valid,
            self.existence,
            self.existence_valid,
            self.track_valid,
            self.capacity_censored,
            self.token_observed_fraction,
            self.inventory_exhaustive,
            *((self.token_measure_weight,) if self.token_measure_weight is not None else ()),
        )
        if any(value.device != self.masks.device for value in tensors):
            raise ValueError("physical targets must share one device")
        if (self.capacity_censored & ~self.track_valid).any():
            raise ValueError("only valid tracks may be capacity-censored")
        invalid = ~self.track_valid
        if (self.mask_valid & invalid.unsqueeze(-1)).any():
            raise ValueError("invalid tracks cannot carry mask supervision")
        if (self.existence_valid & invalid).any():
            raise ValueError("invalid tracks cannot carry existence supervision")
        if self.exclusive_ownership:
            visible_mass = (self.masks * self.mask_valid).sum(dim=1)
            if (visible_mass > 1 + 1e-5).any():
                raise ValueError("exclusive ownership targets cannot overlap")

    @property
    def token_measure(self) -> torch.Tensor:
        """Resolution-invariant integration measure over each token surface."""

        if self.token_measure_weight is None:
            return torch.ones_like(self.token_observed_fraction)
        return self.token_measure_weight


@dataclass(frozen=True, slots=True)
class PhysicalFrameAssignment:
    """One loss-side row gauge; -1 denotes an unmatched no-object row.

    ``carried_rows`` identifies matched identities established before the
    current observation. Such a row may legitimately have no current-frame
    target evidence under occlusion. ``reserved_rows`` instead identifies a
    prior identity absent from the local target axis and therefore remains
    unmatched.
    """

    row_to_track: torch.Tensor
    reserved_rows: torch.Tensor | None = None
    carried_rows: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.row_to_track.ndim != 2 or self.row_to_track.dtype != torch.long:
            raise ValueError("row assignment must be long [batch,rows]")
        if (self.row_to_track < -1).any():
            raise ValueError("unmatched rows must use -1")
        if self.reserved_rows is not None:
            if (
                self.reserved_rows.shape != self.row_to_track.shape
                or self.reserved_rows.dtype != torch.bool
                or self.reserved_rows.device != self.row_to_track.device
            ):
                raise ValueError("reserved rows must be boolean and match the row assignment")
            if (self.reserved_rows & (self.row_to_track >= 0)).any():
                raise ValueError("a row cannot be both matched and anonymously reserved")
        if self.carried_rows is not None:
            if (
                self.carried_rows.shape != self.row_to_track.shape
                or self.carried_rows.dtype != torch.bool
                or self.carried_rows.device != self.row_to_track.device
            ):
                raise ValueError("carried rows must be boolean and match the row assignment")
            if (self.carried_rows & (self.row_to_track < 0)).any():
                raise ValueError("only a matched row can carry a prior physical identity")

    @property
    def reserved(self) -> torch.Tensor:
        """Rows occupied by a prior identity absent from this target frame."""

        if self.reserved_rows is None:
            return torch.zeros_like(self.row_to_track, dtype=torch.bool)
        return self.reserved_rows

    @property
    def carried(self) -> torch.Tensor:
        """Matched rows whose identity was bound before this observation."""

        if self.carried_rows is None:
            return torch.zeros_like(self.row_to_track, dtype=torch.bool)
        return self.carried_rows


@dataclass(frozen=True, slots=True)
class PhysicalSetLoss:
    total: torch.Tensor
    mask_focal: torch.Tensor
    mask_dice: torch.Tensor
    existence_focal: torch.Tensor
    ownership_nll: torch.Tensor
    assignment: PhysicalFrameAssignment


def eligible_physical_tracks(
    targets: PhysicalFrameTargets,
    batch_index: int,
) -> torch.Tensor:
    """Return the exact track domain consumed by loss-side assignment.

    A track is assignable when the current frame supplies either spatial mask
    evidence or a known positive existence target. Keeping this predicate
    public prevents evaluation and training from silently defining different
    physical entity sets under occlusion or partial annotation.
    """

    if not 0 <= batch_index < targets.masks.shape[0]:
        raise IndexError("physical track batch index is out of range")
    visible_positive = (
        targets.masks[batch_index] * targets.mask_valid[batch_index].to(targets.masks.dtype)
    ).sum(dim=-1) > 0
    existence_positive = targets.existence_valid[batch_index] & (targets.existence[batch_index] > 0)
    evidence = visible_positive | existence_positive
    return (
        (targets.track_valid[batch_index] & ~targets.capacity_censored[batch_index] & evidence)
        .nonzero()
        .flatten()
    )


def physical_pairwise_assignment_cost(
    predictions: PhysicalFramePredictions,
    targets: PhysicalFrameTargets,
    *,
    batch_index: int,
    track_indices: torch.Tensor,
    focal_alpha: float,
    focal_gamma: float,
) -> torch.Tensor:
    logits = predictions.support_logits[batch_index].float().transpose(0, 1)
    rows, tokens = logits.shape
    target = targets.masks[batch_index, track_indices].float()
    valid = targets.mask_valid[batch_index, track_indices]
    observed = (
        targets.token_observed_fraction[batch_index].float()
        * targets.token_measure[batch_index].float()
    )
    weight = (
        valid.float()
        * observed.unsqueeze(0)
        * predictions.sensor_valid[batch_index].float().unsqueeze(0)
    )
    expanded_logits = logits[:, None, :].expand(rows, target.shape[0], tokens)
    expanded_target = target.unsqueeze(0).expand_as(expanded_logits)
    expanded_weight = weight.unsqueeze(0).expand_as(expanded_logits)
    denominator = expanded_weight.sum(dim=-1).clamp_min(1)
    focal = (
        sam31_sigmoid_focal_loss(
            expanded_logits,
            expanded_target,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
        * expanded_weight
    ).sum(dim=-1) / denominator
    dice = sam31_dice_loss(
        expanded_logits,
        expanded_target,
        weight=expanded_weight,
    )
    target_existence = (
        targets.existence[batch_index, track_indices].float().unsqueeze(0).expand(rows, -1)
    )
    existence = sam31_sigmoid_focal_loss(
        predictions.existence_logits[batch_index].float().unsqueeze(1).expand_as(target_existence),
        target_existence,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )
    existence = existence * targets.existence_valid[
        batch_index,
        track_indices,
    ].float().unsqueeze(0)
    return focal + dice + existence


@torch.no_grad()
def match_physical_frame_entities(
    predictions: PhysicalFramePredictions,
    targets: PhysicalFrameTargets,
    *,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> PhysicalFrameAssignment:
    """Match class-free visible entities with one-to-one Hungarian assignment."""

    batch, tokens, rows = predictions.support_logits.shape
    if targets.masks.shape[0] != batch or targets.masks.shape[2] != tokens:
        raise ValueError("physical prediction and target axes differ")
    assignment = torch.full((batch, rows), -1, dtype=torch.long, device=targets.masks.device)
    for batch_index in range(batch):
        eligible = eligible_physical_tracks(targets, batch_index)
        if eligible.numel() > rows:
            raise ValueError("uncensored physical targets exceed row capacity")
        if eligible.numel() == 0:
            continue
        cost = physical_pairwise_assignment_cost(
            predictions,
            targets,
            batch_index=batch_index,
            track_indices=eligible,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        row_indices, relative_tracks = linear_sum_assignment(cost.cpu().numpy())
        rows_tensor = torch.as_tensor(row_indices, dtype=torch.long, device=assignment.device)
        tracks_tensor = eligible.index_select(
            0,
            torch.as_tensor(relative_tracks, dtype=torch.long, device=eligible.device),
        )
        assignment[batch_index, rows_tensor] = tracks_tensor
    return PhysicalFrameAssignment(assignment)


def physical_frame_set_loss(
    predictions: PhysicalFramePredictions,
    targets: PhysicalFrameTargets,
    *,
    assignment: PhysicalFrameAssignment | None = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    mask_focal_weight: float = 1.0,
    mask_dice_weight: float = 1.0,
    existence_weight: float = 1.0,
    ownership_weight: float = 1.0,
) -> PhysicalSetLoss:
    """Evaluate matched masks, no-object rows and optional context ownership."""

    weights = (mask_focal_weight, mask_dice_weight, existence_weight, ownership_weight)
    if any(not isinstance(value, (int, float)) or value < 0 for value in weights):
        raise ValueError("physical set weights must be non-negative real values")
    if assignment is None:
        assignment = match_physical_frame_entities(
            predictions,
            targets,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
    batch, tokens, rows = predictions.support_logits.shape
    if assignment.row_to_track.shape != (batch, rows):
        raise ValueError("physical assignment differs from prediction rows")

    focal_values: list[torch.Tensor] = []
    dice_values: list[torch.Tensor] = []
    existence_values: list[torch.Tensor] = []
    ownership_numerators: list[torch.Tensor] = []
    ownership_denominators: list[torch.Tensor] = []
    for batch_index in range(batch):
        row_to_track = assignment.row_to_track[batch_index]
        matched = row_to_track >= 0
        if matched.any():
            row_indices = matched.nonzero().flatten()
            track_indices = row_to_track[row_indices]
            logits = predictions.support_logits[batch_index, :, row_indices].float().transpose(0, 1)
            target = targets.masks[batch_index, track_indices].float()
            valid = targets.mask_valid[batch_index, track_indices]
            weight = (
                valid.float()
                * targets.token_observed_fraction[batch_index].float().unsqueeze(0)
                * targets.token_measure[batch_index].float().unsqueeze(0)
                * predictions.sensor_valid[batch_index].float().unsqueeze(0)
            )
            focal_values.append(
                (
                    sam31_sigmoid_focal_loss(
                        logits,
                        target,
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )
                    * weight
                ).sum(dim=-1)
                / weight.sum(dim=-1).clamp_min(1)
            )
            dice_values.append(sam31_dice_loss(logits, target, weight=weight))

        existence_target = torch.zeros(
            rows,
            dtype=predictions.existence_logits.dtype,
            device=predictions.existence_logits.device,
        )
        existence_valid = torch.zeros(rows, dtype=torch.bool, device=matched.device)
        if matched.any():
            row_indices = matched.nonzero().flatten()
            track_indices = row_to_track[row_indices]
            existence_target[row_indices] = targets.existence[
                batch_index,
                track_indices,
            ].to(existence_target.dtype)
            existence_valid[row_indices] = targets.existence_valid[
                batch_index,
                track_indices,
            ]
        if bool(targets.inventory_exhaustive[batch_index]):
            existence_valid[~matched & ~assignment.reserved[batch_index]] = True
        existence_values.append(
            sam31_sigmoid_focal_loss(
                predictions.existence_logits[batch_index],
                existence_target,
                alpha=focal_alpha,
                gamma=focal_gamma,
            )[existence_valid]
        )

        if targets.exclusive_ownership:
            expected_rows = torch.zeros(
                tokens,
                rows,
                dtype=torch.float32,
                device=targets.masks.device,
            )
            supervised = targets.token_observed_fraction[batch_index] > 0
            for row_index, track_index in enumerate(row_to_track.tolist()):
                if track_index < 0:
                    continue
                expected_rows[:, row_index] = (
                    targets.masks[batch_index, track_index]
                    * targets.mask_valid[batch_index, track_index]
                ).float()
            censored = targets.capacity_censored[batch_index]
            if censored.any():
                censored_visible = (
                    targets.mask_valid[batch_index, censored]
                    & (targets.masks[batch_index, censored] > 0)
                ).any(dim=0)
                supervised &= ~censored_visible
            expected_context = (1 - expected_rows.sum(dim=-1, keepdim=True)).clamp_min(0)
            expected = torch.cat((expected_rows, expected_context), dim=-1)
            expected = expected / expected.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(expected.dtype).tiny
            )
            if not bool(targets.inventory_exhaustive[batch_index]):
                supervised &= expected_rows.sum(dim=-1) > 0
            valid = supervised & predictions.sensor_valid[batch_index]
            if valid.any():
                token_weight = (
                    targets.token_observed_fraction[batch_index, valid].float()
                    * targets.token_measure[batch_index, valid].float()
                )
                token_nll = -(
                    expected[valid]
                    * predictions.ownership_log_probability[batch_index, valid].float()
                ).sum(dim=-1)
                ownership_numerators.append((token_nll * token_weight).sum())
                ownership_denominators.append(token_weight.sum())

    reference = predictions.support_logits.sum() * 0

    def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        nonempty = [value.reshape(-1) for value in values if value.numel()]
        return torch.cat(nonempty).mean() if nonempty else reference

    mask_focal = mean_or_zero(focal_values)
    mask_dice = mean_or_zero(dice_values)
    existence_focal = mean_or_zero(existence_values)
    ownership_nll = (
        torch.stack(ownership_numerators).sum()
        / torch.stack(ownership_denominators)
        .sum()
        .clamp_min(torch.finfo(predictions.support_logits.dtype).tiny)
        if ownership_numerators
        else reference
    )
    total = (
        mask_focal_weight * mask_focal
        + mask_dice_weight * mask_dice
        + existence_weight * existence_focal
        + ownership_weight * ownership_nll
    )
    return PhysicalSetLoss(
        total=total,
        mask_focal=mask_focal,
        mask_dice=mask_dice,
        existence_focal=existence_focal,
        ownership_nll=ownership_nll,
        assignment=assignment,
    )
