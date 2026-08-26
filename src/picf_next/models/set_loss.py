"""Permutation-invariant training targets for task-independent object discovery.

The no-gradient rectangular matching and matched-set supervision follow the
semantics of Mask2Former at commit
``9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`` (MIT), specifically
``HungarianMatcher`` and ``SetCriterion``.  This is a clean-room adaptation for
open-world token ownership. The final ownership term is a category-balanced
one-vs-rest composite score evaluated on one query-plus-context softmax simplex,
so a large context region cannot erase small physical objects. It is a
discriminative segmentation surrogate, not a calibrated categorical
likelihood. Targets never enter the discovery forward pass, there is no closed
class vocabulary, and unmatched queries are trained as non-objects only when
the target explicitly declares a complete inventory.

The geometry head uses a robust mean objective plus a detached Gaussian scale
calibration term. A softplus-positive diagonal variance avoids hard-clamp
gradient dead zones, while PICF deliberately keeps predicted uncertainty from
rescaling the shared geometry-mean gradient. PICF also adds declared target
measurement variance and per-coordinate supervision masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from picf_next.geometry import PhysicalGeometryContract

if TYPE_CHECKING:
    from .discovery import ObjectDiscoveryOutput


@dataclass(frozen=True, slots=True)
class ObjectSetTarget:
    """One sample's unordered physical-object targets.

    ``ownership`` has shape ``[tokens, objects + context]``.  Supervised token
    rows are probability distributions; therefore overlap, uncertain
    boundaries and context are represented without assigning one token to two
    hard labels. Invalid padding and unsupervised rows are exactly zero and
    never contribute to a loss. ``token_valid`` follows the model evidence,
    while ``token_supervised`` is a selective likelihood mask. This distinction
    prevents a modality without labels from being falsely trained as context.
    ``object_inventory_complete`` describes only the current observable target
    set under the producer's annotation ontology. When it is false, named
    objects are positive examples but unmatched queries are unknown rather than
    non-objects. It is not a cross-occlusion survival inventory; that is a
    separate loss-side contract. This prevents partial inventories in a mixed
    dataset from suppressing real, unlabelled objects.

    Optional state targets are supervision-only observations.  They are never
    model inputs and may be omitted independently when a dataset does not
    provide a trustworthy field. ``geometry_supervised`` may additionally omit
    unknown coordinates per object; omitted values must be exactly zero.
    ``temporal_identity_keys`` are likewise
    loss-only physical track keys ordered with the target objects; they never
    name a runtime posterior row or enter discovery.
    """

    ownership: torch.Tensor
    token_valid: torch.Tensor
    token_supervised: torch.Tensor | None = None
    object_inventory_complete: bool = False
    address: torch.Tensor | None = None
    content: torch.Tensor | None = None
    geometry: torch.Tensor | None = None
    geometry_variance: torch.Tensor | None = None
    geometry_supervised: torch.Tensor | None = None
    geometry_contract: PhysicalGeometryContract | None = None
    temporal_identity_keys: tuple[str, ...] | None = None

    @property
    def num_objects(self) -> int:
        return self.ownership.shape[1] - 1

    @property
    def supervision_valid(self) -> torch.Tensor:
        return self.token_valid if self.token_supervised is None else self.token_supervised


@dataclass(frozen=True, slots=True)
class ObjectSetMatcherConfig:
    existence_cost: float = 1.0
    ownership_ce_cost: float = 1.0
    ownership_dice_cost: float = 1.0
    address_cost: float = 0.0
    content_cost: float = 0.0
    geometry_cost: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.existence_cost,
            self.ownership_ce_cost,
            self.ownership_dice_cost,
            self.address_cost,
            self.content_cost,
            self.geometry_cost,
        )
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0.0 for value in values
        ):
            raise ValueError("matching costs must be nonnegative")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one matching cost must be positive")


@dataclass(frozen=True, slots=True)
class ObjectSetLossConfig:
    existence_weight: float = 1.0
    localization_confidence_weight: float = 1.0
    ownership_ce_weight: float = 1.0
    ownership_dice_weight: float = 1.0
    address_cosine_weight: float = 0.0
    content_cosine_weight: float = 0.0
    geometry_weight: float = 1.0

    def __post_init__(self) -> None:
        weighted = {
            "existence_weight": self.existence_weight,
            "localization_confidence_weight": self.localization_confidence_weight,
            "ownership_ce_weight": self.ownership_ce_weight,
            "ownership_dice_weight": self.ownership_dice_weight,
            "address_cosine_weight": self.address_cosine_weight,
            "content_cosine_weight": self.content_cosine_weight,
            "geometry_weight": self.geometry_weight,
        }
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0.0
            for value in weighted.values()
        ):
            raise ValueError("loss weights must be nonnegative")
        if not any(value > 0.0 for value in weighted.values()):
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True, slots=True)
class SetMatch:
    prediction_indices: torch.Tensor
    target_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObjectSetLossOutput:
    losses: dict[str, torch.Tensor]
    matches: tuple[SetMatch, ...]

    @property
    def total(self) -> torch.Tensor:
        return self.losses["loss_total"]


def _pairwise_binary_ce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(min=1e-6, max=1.0 - 1e-6)
    log_probability = probability.log().transpose(0, 1).unsqueeze(2)
    log_complement = torch.log1p(-probability).transpose(0, 1).unsqueeze(2)
    target = target.unsqueeze(0)
    positive = -(target * log_probability).sum(dim=1) / target.sum(dim=1).clamp_min(1e-6)
    negative_target = 1.0 - target
    negative = -(negative_target * log_complement).sum(dim=1) / negative_target.sum(
        dim=1
    ).clamp_min(1e-6)
    return 0.5 * (positive + negative)


def _balanced_binary_ce_by_category(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return macro-balanced binary log scores and active target categories.

    ``probability`` is still one categorical simplex per token. Treating each
    simplex marginal as one-vs-rest only defines a class-balanced segmentation
    surrogate; it does not create independent ownership heads.
    """

    probability = probability.clamp(min=1e-6, max=1.0 - 1e-6)
    positive_mass = target.sum(dim=0)
    negative_target = 1.0 - target
    negative_mass = negative_target.sum(dim=0)
    positive = -(target * probability.log()).sum(dim=0) / positive_mass.clamp_min(1e-6)
    negative = -(negative_target * torch.log1p(-probability)).sum(dim=0) / negative_mass.clamp_min(
        1e-6
    )
    return 0.5 * (positive + negative), positive_mass > 0.0


def _pairwise_dice(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = probability.transpose(0, 1)
    target = target.transpose(0, 1)
    numerator = 2.0 * torch.einsum("qn,mn->qm", probability, target)
    denominator = probability.sum(dim=1, keepdim=True) + target.sum(dim=1).unsqueeze(0)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def _pairwise_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction[:, None, :] - target[None, :, :]).abs().mean(dim=-1)


def _pairwise_masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervised: torch.Tensor,
) -> torch.Tensor:
    residual = (prediction[:, None, :] - target[None, :, :]).abs()
    weight = supervised.to(dtype=residual.dtype).unsqueeze(0)
    return (residual * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def _pairwise_cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - prediction @ target.transpose(0, 1)


class ObjectSetHungarianMatcher(nn.Module):
    """No-gradient one-to-one matching of transient queries to target objects."""

    def __init__(self, config: ObjectSetMatcherConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectSetMatcherConfig()

    @torch.no_grad()
    def forward(
        self,
        output: ObjectDiscoveryOutput,
        targets: tuple[ObjectSetTarget, ...] | list[ObjectSetTarget],
    ) -> tuple[SetMatch, ...]:
        _validate_batch(output, targets)
        matches = []
        for batch_index, target in enumerate(targets):
            num_objects = target.num_objects
            if num_objects == 0:
                empty = torch.empty(0, dtype=torch.long, device=output.existence_logits.device)
                matches.append(SetMatch(empty, empty))
                continue

            valid = target.supervision_valid
            # Matching is a discrete control decision.  Compute every
            # probability, logarithm and distance in float32 even when the
            # trainable representation uses bf16.  In bf16, ``1 - 1e-6``
            # rounds to one and the binary-CE clamp can otherwise produce
            # ``log(0)`` before the final CPU cast.
            probability = output.ownership[batch_index, valid, :-1].float()
            target_objects = target.ownership[valid, :-1].float()
            existence_logits = output.existence_logits[batch_index].float().unsqueeze(1)
            cost = self.config.existence_cost * F.softplus(-existence_logits)
            cost = cost.expand(-1, num_objects).clone()
            if valid.any():
                cost += self.config.ownership_ce_cost * _pairwise_binary_ce(
                    probability, target_objects
                )
                cost += self.config.ownership_dice_cost * _pairwise_dice(
                    probability, target_objects
                )
            if self.config.address_cost > 0.0 and target.address is not None:
                cost += self.config.address_cost * _pairwise_cosine_distance(
                    output.address_mean[batch_index].float(),
                    target.address.float(),
                )
            if self.config.content_cost > 0.0 and target.content is not None:
                cost += self.config.content_cost * _pairwise_cosine_distance(
                    output.content_mean[batch_index].float(),
                    target.content.float(),
                )
            if self.config.geometry_cost > 0.0 and target.geometry is not None:
                geometry_supervised = target.geometry_supervised
                if geometry_supervised is None:
                    cost += self.config.geometry_cost * _pairwise_l1(
                        output.geometry_mean[batch_index].float(),
                        target.geometry.float(),
                    )
                else:
                    cost += self.config.geometry_cost * _pairwise_masked_l1(
                        output.geometry_mean[batch_index].float(),
                        target.geometry.float(),
                        geometry_supervised,
                    )

            if not torch.isfinite(cost).all():
                raise ValueError("matching cost contains NaN or infinity")
            prediction_indices, target_indices = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )
            device = output.existence_logits.device
            matches.append(
                SetMatch(
                    torch.as_tensor(prediction_indices, dtype=torch.long, device=device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=device),
                )
            )
        return tuple(matches)


class ObjectSetCriterion(nn.Module):
    """Matched set loss with an explicit non-object and context category."""

    def __init__(
        self,
        matcher: ObjectSetHungarianMatcher | None = None,
        config: ObjectSetLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.matcher = matcher or ObjectSetHungarianMatcher()
        self.config = config or ObjectSetLossConfig()

    def forward(
        self,
        output: ObjectDiscoveryOutput,
        targets: tuple[ObjectSetTarget, ...] | list[ObjectSetTarget],
    ) -> ObjectSetLossOutput:
        _validate_batch(output, targets)
        # Auxiliary probabilistic losses form the calibrated control plane and
        # stay in float32 under AMP.  Casts retain gradients to bf16 model
        # activations while avoiding coarse clamps/logarithms and covariance
        # arithmetic.
        zero = output.existence_logits.float().sum() * 0.0
        stage_outputs = (*output.auxiliary_outputs, output)
        stage_losses = []
        stage_matches = []
        for stage_output in stage_outputs:
            matches = self.matcher(stage_output, targets)
            stage_matches.append(matches)
            stage_losses.append(
                self._segmentation_losses(stage_output, targets, matches, zero=zero)
            )
        matches = stage_matches[-1]

        losses = {
            name: torch.stack([stage[name] for stage in stage_losses]).mean()
            for name in (
                "loss_existence",
                "loss_localization_confidence",
                "loss_ownership_ce",
                "loss_ownership_dice",
            )
        }
        address_terms = []
        content_terms = []
        geometry_mean_terms = []
        geometry_calibration_terms = []
        geometry_terms = []

        for batch_index, (target, match) in enumerate(zip(targets, matches, strict=True)):
            if target.address is not None and match.prediction_indices.numel():
                predicted_address = output.address_mean[
                    batch_index, match.prediction_indices
                ].float()
                target_address = target.address[match.target_indices].float()
                address_terms.append(
                    1.0
                    - F.cosine_similarity(
                        predicted_address,
                        target_address,
                        dim=-1,
                    )
                )

            if target.content is not None and match.prediction_indices.numel():
                content_terms.append(
                    1.0
                    - F.cosine_similarity(
                        output.content_mean[batch_index, match.prediction_indices].float(),
                        target.content[match.target_indices].float(),
                        dim=-1,
                    )
                )
            if target.geometry is not None and match.prediction_indices.numel():
                target_variance = target.geometry_variance
                if target_variance is None:
                    target_variance = torch.zeros_like(target.geometry)
                predicted_geometry = output.geometry_mean[
                    batch_index, match.prediction_indices
                ].float()
                expected_geometry = target.geometry[match.target_indices].float()
                robust_error = F.smooth_l1_loss(
                    predicted_geometry,
                    expected_geometry,
                    beta=1.0,
                    reduction="none",
                )
                combined_variance = (
                    output.geometry_variance[batch_index, match.prediction_indices].float()
                    + target_variance[match.target_indices].float()
                ).clamp_min(1e-8)
                squared_residual = (predicted_geometry - expected_geometry).square()
                calibration = 0.5 * (
                    squared_residual.detach() / combined_variance + combined_variance.log()
                )
                geometry_objective = robust_error + calibration
                geometry_supervised = target.geometry_supervised
                if geometry_supervised is None:
                    geometry_mean_terms.append(robust_error.mean(dim=-1))
                    geometry_calibration_terms.append(calibration.mean(dim=-1))
                    geometry_terms.append(geometry_objective.mean(dim=-1))
                else:
                    selected = geometry_supervised[match.target_indices]
                    supervised_count = selected.sum(dim=-1)
                    active = supervised_count > 0
                    if active.any():
                        geometry_mean_terms.append(
                            (robust_error * selected).sum(dim=-1)[active] / supervised_count[active]
                        )
                        geometry_calibration_terms.append(
                            (calibration * selected).sum(dim=-1)[active] / supervised_count[active]
                        )
                        geometry_terms.append(
                            (geometry_objective * selected).sum(dim=-1)[active]
                            / supervised_count[active]
                        )

        losses["loss_address_cosine"] = _mean_terms(address_terms, zero)
        losses["loss_content_cosine"] = _mean_terms(content_terms, zero)
        losses["loss_geometry_mean"] = _mean_terms(geometry_mean_terms, zero)
        losses["loss_geometry_calibration"] = _mean_terms(geometry_calibration_terms, zero)
        losses["loss_geometry"] = _mean_terms(geometry_terms, zero)
        losses["loss_total"] = (
            self.config.existence_weight * losses["loss_existence"]
            + self.config.localization_confidence_weight * losses["loss_localization_confidence"]
            + self.config.ownership_ce_weight * losses["loss_ownership_ce"]
            + self.config.ownership_dice_weight * losses["loss_ownership_dice"]
            + self.config.address_cosine_weight * losses["loss_address_cosine"]
            + self.config.content_cosine_weight * losses["loss_content_cosine"]
            + self.config.geometry_weight * losses["loss_geometry"]
        )
        return ObjectSetLossOutput(losses=losses, matches=matches)

    @staticmethod
    def _segmentation_losses(
        output: ObjectDiscoveryOutput,
        targets: tuple[ObjectSetTarget, ...] | list[ObjectSetTarget],
        matches: tuple[SetMatch, ...],
        *,
        zero: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        existence_numerator = zero
        existence_denominator = zero
        localization_confidence_terms = []
        ownership_numerator = zero
        ownership_denominator = zero
        dice_terms = []
        query_count = output.existence_logits.shape[1]

        for batch_index, (target, match) in enumerate(zip(targets, matches, strict=True)):
            existence_logits = output.existence_logits[batch_index].float()
            if target.object_inventory_complete:
                negative_weight = output.existence_calibration.unmatched_query_weight
                per_query_loss = negative_weight * F.softplus(existence_logits)
                per_query_weight = torch.full_like(existence_logits, negative_weight)
            else:
                negative_weight = 0.0
                per_query_loss = torch.zeros_like(existence_logits)
                per_query_weight = torch.zeros_like(existence_logits)
            if match.prediction_indices.numel():
                matched_logits = existence_logits[match.prediction_indices]
                per_query_loss[match.prediction_indices] = F.softplus(-matched_logits)
                per_query_weight[match.prediction_indices] = 1.0
            existence_numerator = existence_numerator + per_query_loss.sum()
            existence_denominator = existence_denominator + per_query_weight.sum()

            valid = target.supervision_valid
            if not valid.any():
                continue
            remapped_target = torch.zeros(
                (int(valid.sum()), query_count + 1),
                device=output.ownership.device,
                dtype=torch.float32,
            )
            remapped_target[:, -1] = target.ownership[valid, -1].float()
            if match.prediction_indices.numel():
                remapped_target[:, match.prediction_indices] = target.ownership[valid][
                    :, match.target_indices
                ].float()
            category_loss, active_category = _balanced_binary_ce_by_category(
                output.ownership[batch_index, valid].float(),
                remapped_target,
            )
            ownership_numerator = ownership_numerator + category_loss[active_category].sum()
            ownership_denominator = ownership_denominator + active_category.sum()

            if match.prediction_indices.numel():
                predicted = output.ownership[batch_index, valid][
                    :, match.prediction_indices
                ].float()
                expected = target.ownership[valid][:, match.target_indices].float()
                # A separate quality head predicts conditional expected support
                # fidelity. As in Mask Scoring R-CNN and CoTracker confidence
                # supervision, correctness is computed from the detached
                # prediction so this auxiliary score cannot improve its own
                # label by moving the ownership mask. Matching also remains
                # independent of confidence.
                target_mass = expected.sum(dim=0)
                quality_supervised = target_mass > 0.0
                if quality_supervised.any():
                    intersection = (predicted * expected).sum(dim=0)
                    union = predicted.sum(dim=0) + target_mass - intersection
                    soft_iou = (
                        (intersection / union.clamp_min(1e-6)).detach().clamp(min=0.0, max=1.0)
                    )
                    confidence_logits = output.localization_confidence_logits[
                        batch_index, match.prediction_indices
                    ].float()
                    localization_confidence_terms.append(
                        F.binary_cross_entropy_with_logits(
                            confidence_logits[quality_supervised],
                            soft_iou[quality_supervised],
                            reduction="none",
                        )
                    )
                numerator = 2.0 * (predicted * expected).sum(dim=0)
                denominator = predicted.sum(dim=0) + expected.sum(dim=0)
                dice_terms.append(1.0 - (numerator + 1.0) / (denominator + 1.0))

        return {
            "loss_existence": existence_numerator / existence_denominator.clamp_min(1.0),
            "loss_localization_confidence": _mean_terms(
                localization_confidence_terms,
                zero,
            ),
            "loss_ownership_ce": ownership_numerator / ownership_denominator.clamp_min(1.0),
            "loss_ownership_dice": _mean_terms(dice_terms, zero),
        }


def _mean_terms(terms: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    if not terms:
        return zero
    return torch.cat([term.reshape(-1) for term in terms]).mean()


def _validate_batch(
    output: ObjectDiscoveryOutput,
    targets: tuple[ObjectSetTarget, ...] | list[ObjectSetTarget],
) -> None:
    batch_size, token_count, category_count = output.ownership.shape
    query_count = output.existence_logits.shape[1]
    if len(targets) != batch_size:
        raise ValueError("target count must equal discovery batch size")
    if category_count != query_count + 1:
        raise ValueError("ownership categories must equal queries plus context")
    for batch_index, target in enumerate(targets):
        if not isinstance(target.object_inventory_complete, bool):
            raise ValueError("target object_inventory_complete must be a bool")
        if target.ownership.ndim != 2 or target.ownership.shape[0] != token_count:
            raise ValueError("target ownership must be token-by-object-plus-context")
        if target.num_objects < 0 or target.num_objects > query_count:
            raise ValueError("target object count exceeds discovery query capacity")
        if target.token_valid.dtype != torch.bool or target.token_valid.shape != (token_count,):
            raise ValueError("target token_valid must be a bool token vector")
        if target.ownership.device != output.ownership.device:
            raise ValueError("targets and discovery output must share a device")
        if target.token_valid.device != output.ownership.device:
            raise ValueError("target validity and discovery output must share a device")
        if not torch.equal(target.token_valid, output.token_valid[batch_index]):
            raise ValueError("target and discovery token validity must match exactly")
        supervised = target.supervision_valid
        if supervised.dtype != torch.bool or supervised.shape != (token_count,):
            raise ValueError("target token_supervised must be a bool token vector")
        if supervised.device != output.ownership.device:
            raise ValueError("target supervision validity and output must share a device")
        if (supervised & ~target.token_valid).any():
            raise ValueError("target supervision cannot include invalid evidence tokens")
        if not torch.is_floating_point(target.ownership):
            raise ValueError("target ownership must use a floating dtype")
        if target.ownership.requires_grad:
            raise ValueError("loss-only target ownership must not require gradients")
        if not torch.isfinite(target.ownership).all() or (target.ownership < 0.0).any():
            raise ValueError("target ownership must be finite and nonnegative")
        if (target.ownership[~supervised] != 0.0).any():
            raise ValueError("invalid or unsupervised target ownership must be exactly zero")
        if supervised.any():
            row_sum = target.ownership[supervised].float().sum(dim=-1)
            tolerance = max(1e-5, torch.finfo(target.ownership.dtype).eps)
            if not torch.allclose(
                row_sum,
                torch.ones_like(row_sum),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError("supervised target ownership rows must sum to one")
        elif target.num_objects:
            state_available = any(
                value is not None for value in (target.address, target.content, target.geometry)
            )
            if not state_available:
                raise ValueError("target objects require ownership or state supervision")
        for name, value, prediction in (
            ("address", target.address, output.address_mean),
            ("content", target.content, output.content_mean),
            ("geometry", target.geometry, output.geometry_mean),
        ):
            if value is None:
                continue
            expected = (target.num_objects, prediction.shape[-1])
            if value.shape != expected:
                raise ValueError(f"target {name} must have shape {expected}")
            if value.device != output.ownership.device or not torch.is_floating_point(value):
                raise ValueError(f"target {name} must be floating and colocated with output")
            if value.requires_grad:
                raise ValueError(f"loss-only target {name} must not require gradients")
            if not torch.isfinite(value).all():
                raise ValueError(f"target {name} contains NaN or infinity")
            if name == "address" and value.numel():
                norm = torch.linalg.vector_norm(value.float(), dim=-1)
                tolerance = max(1e-5, torch.finfo(value.dtype).eps)
                if not torch.allclose(
                    norm,
                    torch.ones_like(norm),
                    atol=tolerance,
                    rtol=tolerance,
                ):
                    raise ValueError("target address rows must have unit norm")
        geometry_supervised = target.geometry_supervised
        geometry_variance = target.geometry_variance
        geometry_contract = target.geometry_contract
        if target.geometry is None:
            if geometry_contract is not None or geometry_variance is not None:
                raise ValueError("geometry metadata cannot be supplied without geometry")
        elif not isinstance(geometry_contract, PhysicalGeometryContract):
            raise ValueError("target geometry requires a physical geometry contract")
        elif geometry_contract != output.geometry_contract:
            raise ValueError("target and discovery geometry contracts differ")
        if geometry_variance is not None:
            if target.geometry is None or geometry_variance.shape != target.geometry.shape:
                raise ValueError("target geometry_variance must align with target geometry")
            if geometry_variance.device != output.ownership.device or not torch.is_floating_point(
                geometry_variance
            ):
                raise ValueError("target geometry variance must be floating and colocated")
            if geometry_variance.requires_grad:
                raise ValueError("loss-only target geometry variance must not require gradients")
            if not torch.isfinite(geometry_variance).all() or (geometry_variance < 0.0).any():
                raise ValueError("target geometry variance must be finite and nonnegative")
        if geometry_supervised is not None:
            if target.geometry is None:
                raise ValueError("geometry supervision requires target geometry")
            if (
                geometry_supervised.shape != target.geometry.shape
                or geometry_supervised.dtype != torch.bool
            ):
                raise ValueError(
                    "target geometry_supervised must be a bool object-by-geometry tensor"
                )
            if geometry_supervised.device != output.ownership.device:
                raise ValueError("target geometry supervision must be colocated with output")
            if geometry_supervised.requires_grad:
                raise ValueError("loss-only geometry supervision must not require gradients")
            if (target.geometry[~geometry_supervised] != 0.0).any():
                raise ValueError("unsupervised target geometry coordinates must be exactly zero")
            if (
                geometry_variance is not None
                and (geometry_variance[~geometry_supervised] != 0.0).any()
            ):
                raise ValueError("unsupervised target geometry variance must be exactly zero")
        identity_keys = target.temporal_identity_keys
        if identity_keys is not None:
            if len(identity_keys) != target.num_objects:
                raise ValueError("temporal identity keys must align with target objects")
            if any(not isinstance(key, str) or not key for key in identity_keys) or len(
                set(identity_keys)
            ) != len(identity_keys):
                raise ValueError(
                    "temporal identity keys must be nonempty strings and unique within one frame"
                )
