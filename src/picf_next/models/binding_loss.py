"""Selective cross-modal and cross-time binding over physical objects.

The sigmoid objective follows the pair-label semantics of Google Big Vision's
SigLIP trainer at commit ``0127fb6b337ee2a27bf4e54dea79cff176527356``
(``big_vision/trainers/proj/image_text/siglip.py``), under Apache-2.0.  This is a
clean-room PyTorch adaptation for multiple relation views of unordered objects.
Its scale follows the pinned official
``configs/proj/image_text/siglip_lit_coco.py`` initialization. Unlike the
released image-text classifier, the shared relation exposed to the posterior is
an actual log density ratio. The sigmoid training classifier subtracts the
explicit per-graph log noise ratio, so changing object count or available
modalities cannot silently change the runtime meaning of the relation score.
Training targets never enter discovery or action forward paths. Temporal
address binding adapts normalized cross-time relation semantics from
SlotContrast commit ``55ec66dc02eeade630805789ef4a6c5df06f21ff`` (MIT), but
uses explicit loss-only physical keys instead of assuming fixed slot-row
correspondence. It follows STAITUS (arXiv:2606.23436) in aligning appearance
identity without forcing geometry or dynamic content to stay constant; its
public code was unavailable at the audit date.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .evidence import BindingProjectionOutput
from .set_loss import ObjectSetTarget

if TYPE_CHECKING:
    from .discovery import ObjectDiscoveryOutput
    from .set_loss import SetMatch


@dataclass(frozen=True, slots=True)
class BindingLossConfig:
    """Initialization for one mutually exclusive density-ratio objective.

    ``logit_bias`` initializes the log density ratio, not the binary
    classifier's class-prior intercept. For sigmoid training, the latter is
    applied separately by subtracting the observed
    ``log(N_negative / N_positive)`` inside each relation graph. The default
    sigmoid affine initialization retains the released SigLIP values and the
    filter's conservative empty-observation behavior; it is an initial
    condition, not a claim that the untrained score is calibrated. InfoNCE is
    shift invariant and starts at zero.
    """

    objective: Literal["sigmoid", "multi_positive_infonce"] = "sigmoid"
    temperature: float = 0.1
    logit_bias: float | None = None
    minimum_object_mass: float = 1e-4

    def __post_init__(self) -> None:
        if self.objective not in {"sigmoid", "multi_positive_infonce"}:
            raise ValueError(f"unsupported binding objective {self.objective}")
        if (
            isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0.0
        ):
            raise ValueError("binding temperature must be positive")
        if self.logit_bias is not None and (
            isinstance(self.logit_bias, bool) or not math.isfinite(self.logit_bias)
        ):
            raise ValueError("binding logit_bias must be finite")
        if (
            isinstance(self.minimum_object_mass, bool)
            or not math.isfinite(self.minimum_object_mass)
            or self.minimum_object_mass <= 0.0
        ):
            raise ValueError("minimum_object_mass must be positive")
        if self.objective == "multi_positive_infonce" and self.logit_bias not in {None, 0.0}:
            raise ValueError("logit_bias is identifiable only for the sigmoid objective")

    @property
    def effective_logit_bias(self) -> float:
        if self.logit_bias is not None:
            return self.logit_bias
        return -2.71 if self.objective == "sigmoid" else 0.0


@dataclass(frozen=True, slots=True)
class BindingLossOutput:
    loss: torch.Tensor
    object_modality_views: int
    positive_pairs: int
    negative_pairs: int


@dataclass(frozen=True, slots=True)
class TemporalAddressBindingOutput:
    loss: torch.Tensor
    address_views: int
    null_address_views: int
    positive_pairs: int
    negative_pairs: int
    null_negative_pairs: int
    eligible_samples: int
    covered_eligible_samples: int


@dataclass(frozen=True, slots=True)
class _ObjectModalityViews:
    embeddings: torch.Tensor
    object_index: torch.Tensor
    modality_index: torch.Tensor
    count: torch.Tensor


class SphericalRelationCalibration(nn.Module):
    """Learn one normalized-pair log density-ratio scale and bias.

    This is the PyTorch scalar parameterization used by the pinned Big Vision
    SigLIP source: the positive scale is stored in log space and exponentiated,
    while the affine bias is unconstrained. Here the bias belongs to the
    physical relation distribution; the training class prior is applied only
    inside :func:`sigmoid_density_ratio_loss`. Keeping these two scalars in one
    relation distribution avoids an external reliability scorer.
    """

    def __init__(self, config: BindingLossConfig) -> None:
        super().__init__()
        if config.objective != "sigmoid":
            raise ValueError("learned spherical calibration requires sigmoid binding")
        self.logit_scale_parameter = nn.Parameter(
            torch.tensor(math.log(1.0 / config.temperature), dtype=torch.float32)
        )
        self.logit_bias = nn.Parameter(
            torch.tensor(config.effective_logit_bias, dtype=torch.float32)
        )

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.logit_scale_parameter.float().exp()

    def forward(self, cosine: torch.Tensor) -> torch.Tensor:
        return cosine.float() * self.logit_scale + self.logit_bias.float()


def spherical_relation_logits(
    cosine: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """Return a spherical relation log density ratio with strict typing."""

    for name, value in (("logit_scale", logit_scale), ("logit_bias", logit_bias)):
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"relation {name} must be one scalar tensor")
        if value.device != cosine.device or not torch.is_floating_point(value):
            raise ValueError(f"relation {name} must be floating and colocated")
        if not torch.isfinite(value.float()).all():
            raise ValueError(f"relation {name} must be finite")
    if (logit_scale.detach().float() <= 0.0).any():
        raise ValueError("relation logit_scale must be positive")
    return cosine.float() * logit_scale.float() + logit_bias.float()


def sigmoid_density_ratio_loss(
    log_likelihood_ratio: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    """Estimate one relation LLR with explicit noise-prior correction.

    With one positive draw and ``nu`` negative draws, the optimal uncorrected
    sigmoid classifier is ``log p_positive/p_negative - log(nu)``. Runtime
    association needs the first term, not a score whose intercept changes with
    relation-graph cardinality. We therefore train the classifier with

    ``classifier_logit = relation_llr - log(N_negative / N_positive)``.

    This preserves the released SigLIP pair-label estimator while making the
    learned relation a calibrated density ratio. The bounded per-graph outer
    normalization changes gradient scale, not the population optimum. A graph
    lacking either class cannot identify a density ratio and contributes exact
    zero.
    """

    if (
        log_likelihood_ratio.shape != positive.shape
        or positive.shape != negative.shape
        or positive.dtype != torch.bool
        or negative.dtype != torch.bool
    ):
        raise ValueError("density-ratio logits and boolean pair masks must align")
    if (positive & negative).any():
        raise ValueError("positive and negative density-ratio pairs must be disjoint")

    positive_count = positive.sum()
    negative_count = negative.sum()
    estimable = (positive_count > 0) & (negative_count > 0)
    # Clamp only protects the inactive branch from log(0). Counts are discrete,
    # loss-side graph statistics and never receive gradients.
    log_noise_ratio = (
        negative_count.clamp_min(1).float().log() - positive_count.clamp_min(1).float().log()
    ).detach()
    classifier_logits = log_likelihood_ratio - log_noise_ratio
    selected = positive | negative
    labels = torch.where(
        positive, torch.ones_like(classifier_logits), -torch.ones_like(classifier_logits)
    )
    pair_loss = -F.logsigmoid(labels * classifier_logits)
    return (pair_loss * selected).sum() / selected.sum().clamp_min(1) * estimable


class MultimodalBindingCriterion(nn.Module):
    """Align the same physical object across available modalities.

    Native token count cannot change this objective directly.  Tokens are first
    averaged inside each supervised physical-object/modality cell, then only
    cross-modality cells are compared.  Same-object cells are positives;
    different-object cells are negatives.  Context, unsupervised tokens and
    absent modality cells produce no pair.
    """

    def __init__(self, config: BindingLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or BindingLossConfig()
        self.relation = (
            SphericalRelationCalibration(self.config)
            if self.config.objective == "sigmoid"
            else None
        )

    def forward(
        self,
        projection: BindingProjectionOutput,
        targets: tuple[ObjectSetTarget, ...] | list[ObjectSetTarget],
    ) -> BindingLossOutput:
        _validate_inputs(projection, targets)
        zero = projection.binding_features.float().sum() * 0.0
        losses = []
        active_samples = []
        counts = torch.zeros(3, dtype=torch.long, device=projection.binding_features.device)

        for batch_index, target in enumerate(targets):
            views = _pool_object_modality_views(
                projection,
                target,
                batch_index,
                minimum_mass=self.config.minimum_object_mass,
            )
            counts[0] = counts[0] + views.count
            embeddings = F.normalize(views.embeddings.float(), dim=-1)
            cosine = embeddings @ embeddings.transpose(0, 1)
            if self.relation is not None:
                logits = self.relation(cosine)
            else:
                logits = cosine / self.config.temperature

            different_modality = views.modality_index[:, None] != views.modality_index[None, :]
            same_object = views.object_index[:, None] == views.object_index[None, :]
            positive = different_modality & same_object
            negative = different_modality & ~same_object
            positive_count = positive.sum()
            negative_count = negative.sum()
            counts[1] = counts[1] + positive_count
            counts[2] = counts[2] + negative_count
            # A relation graph needs both distributions. Positive-only
            # attraction cannot identify a density ratio, while negative-only
            # repulsion would let missing-modality patterns change the task.
            active = (positive_count > 0) & (negative_count > 0)
            active_samples.append(active)

            if self.config.objective == "sigmoid":
                losses.append(sigmoid_density_ratio_loss(logits, positive, negative))
            else:
                anchor_active = positive.any(dim=1)
                fallback = (
                    torch.eye(
                        logits.shape[0],
                        dtype=torch.bool,
                        device=logits.device,
                    )
                    & ~anchor_active[:, None]
                )
                candidate_logits = logits.masked_fill(~(different_modality | fallback), -torch.inf)
                positive_logits = logits.masked_fill(~(positive | fallback), -torch.inf)
                candidate_partition = torch.logsumexp(candidate_logits, dim=1)
                positive_partition = torch.logsumexp(positive_logits, dim=1)
                anchor_loss = torch.where(
                    anchor_active,
                    candidate_partition - positive_partition,
                    torch.zeros_like(candidate_partition),
                )
                losses.append(anchor_loss.sum() / anchor_active.sum().clamp_min(1) * active)

        if losses:
            active_count = torch.stack(active_samples).sum().clamp_min(1)
            loss = torch.stack(losses).sum() / active_count
        else:  # A zero-sized batch is rejected by input validation; retained for typing.
            loss = zero
        view_count, positive_count, negative_count = counts.detach().cpu().tolist()
        return BindingLossOutput(
            loss=loss,
            object_modality_views=view_count,
            positive_pairs=positive_count,
            negative_pairs=negative_count,
        )


class TemporalAddressBindingCriterion(nn.Module):
    """Align supervised same-object address views across time.

    The relation uses the same density-ratio family as multimodal binding, not
    a separate representation objective. Only loss-side physical identity keys
    define pairs. Geometry and dynamic content are deliberately excluded so
    motion cannot be trained away in the name of temporal consistency.

    A complete current-frame inventory additionally makes every unmatched
    discovery query a known null identity view. Such a view is a negative for
    every physical identity from another time, but null-null pairs are omitted:
    there is no physical identity relation to learn between two non-objects.
    Partial inventories retain unmatched queries as unknown and never turn
    missing annotation into negative supervision.
    """

    def __init__(self, config: BindingLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or BindingLossConfig()

    def forward(
        self,
        discoveries: Sequence[ObjectDiscoveryOutput],
        targets: Sequence[Sequence[ObjectSetTarget]],
        matches: Sequence[Sequence[SetMatch]],
        *,
        initial_address: torch.Tensor | None = None,
        initial_valid: torch.Tensor | None = None,
        initial_identity_keys_by_row: Sequence[Sequence[str | None]] | None = None,
        relation_logit_scale: torch.Tensor | None = None,
        relation_logit_bias: torch.Tensor | None = None,
    ) -> TemporalAddressBindingOutput:
        discoveries = tuple(discoveries)
        targets = tuple(tuple(frame) for frame in targets)
        matches = tuple(tuple(frame) for frame in matches)
        if not discoveries or len(targets) != len(discoveries) or len(matches) != len(discoveries):
            raise ValueError("temporal binding requires aligned nonempty frame sequences")
        batch_size = discoveries[0].address_mean.shape[0]
        if any(
            discovery.address_mean.shape[0] != batch_size
            or len(frame_targets) != batch_size
            or len(frame_matches) != batch_size
            for discovery, frame_targets, frame_matches in zip(
                discoveries, targets, matches, strict=True
            )
        ):
            raise ValueError("temporal binding frame batches must agree")
        initial_supplied = (
            initial_address is not None,
            initial_valid is not None,
            initial_identity_keys_by_row is not None,
        )
        if any(initial_supplied) and not all(initial_supplied):
            raise ValueError("initial temporal address, validity and identity keys are atomic")
        relation_supplied = relation_logit_scale is not None, relation_logit_bias is not None
        if any(relation_supplied) and not all(relation_supplied):
            raise ValueError("temporal relation scale and bias are atomic")
        frozen_initial_keys: tuple[tuple[str | None, ...], ...] | None = None
        frozen_initial_address: torch.Tensor | None = None
        if (
            initial_address is not None
            and initial_valid is not None
            and initial_identity_keys_by_row is not None
        ):
            if initial_address.ndim != 3 or initial_address.shape[0] != batch_size:
                raise ValueError("initial temporal address must be batch-by-row-by-address")
            if (
                initial_valid.dtype != torch.bool
                or initial_valid.shape != initial_address.shape[:2]
            ):
                raise ValueError("initial temporal validity must be bool batch-by-row")
            if (
                initial_address.device != discoveries[0].address_mean.device
                or initial_valid.device != initial_address.device
                or initial_address.dtype != discoveries[0].address_mean.dtype
            ):
                raise ValueError("initial temporal state must share discovery dtype/device")
            frozen_initial_keys = tuple(tuple(keys) for keys in initial_identity_keys_by_row)
            frozen_initial_address = initial_address
            if len(frozen_initial_keys) != batch_size or any(
                len(keys) != initial_address.shape[1] for keys in frozen_initial_keys
            ):
                raise ValueError("initial temporal identity keys must be batch-by-row")
            for batch_index, keys in enumerate(frozen_initial_keys):
                present = [key for key in keys if key is not None]
                if any(not isinstance(key, str) or not key for key in present):
                    raise ValueError("initial temporal keys must be nonempty strings or None")
                if len(set(present)) != len(present):
                    raise ValueError("initial temporal keys must be unique within each sample")
                if any(
                    key is not None and not bool(initial_valid[batch_index, row])
                    for row, key in enumerate(keys)
                ):
                    raise ValueError("initial temporal keys cannot name unused posterior rows")

        zero = discoveries[0].address_mean.float().sum() * 0.0
        losses = []
        view_count = 0
        null_view_count = 0
        positive_count = 0
        negative_count = 0
        null_negative_count = 0
        eligible_samples = 0
        covered_eligible_samples = 0
        for batch_index in range(batch_size):
            embeddings = []
            identity_keys: list[str | None] = []
            time_indices = []
            physical_identity_view = []
            identity_times: dict[str, set[int]] = {}
            if frozen_initial_keys is not None:
                if frozen_initial_address is None:
                    raise RuntimeError("validated temporal keys lost their initial address tensor")
                for row, identity_key in enumerate(frozen_initial_keys[batch_index]):
                    if identity_key is not None:
                        embeddings.append(frozen_initial_address[batch_index, row].detach())
                        identity_keys.append(identity_key)
                        time_indices.append(-1)
                        physical_identity_view.append(True)
                        identity_times.setdefault(identity_key, set()).add(-1)
            for time_index, (discovery, frame_targets, frame_matches) in enumerate(
                zip(discoveries, targets, matches, strict=True)
            ):
                target = frame_targets[batch_index]
                identity = target.temporal_identity_keys
                if identity is None:
                    continue
                if len(identity) != target.num_objects or any(
                    not isinstance(key, str) or not key for key in identity
                ):
                    raise ValueError("temporal identity keys must align with target objects")
                if len(set(identity)) != len(identity):
                    raise ValueError("temporal identity keys must be unique within one frame")
                for identity_key in identity:
                    identity_times.setdefault(identity_key, set()).add(time_index)
                match = frame_matches[batch_index]
                if match.prediction_indices.numel() != match.target_indices.numel():
                    raise ValueError("temporal binding received a malformed set match")
                for prediction_index, target_index in zip(
                    match.prediction_indices.tolist(),
                    match.target_indices.tolist(),
                    strict=True,
                ):
                    embeddings.append(discovery.address_mean[batch_index, prediction_index])
                    identity_keys.append(identity[target_index])
                    time_indices.append(time_index)
                    physical_identity_view.append(True)
                if target.object_inventory_complete:
                    query_count = discovery.address_mean.shape[1]
                    matched_query = torch.zeros(
                        query_count,
                        dtype=torch.bool,
                        device=match.prediction_indices.device,
                    )
                    if match.prediction_indices.numel():
                        if (
                            match.prediction_indices.unique().numel()
                            != match.prediction_indices.numel()
                        ):
                            raise ValueError("temporal binding matches cannot repeat a query")
                        matched_query[match.prediction_indices] = True
                    unmatched_query = (~matched_query).nonzero(as_tuple=False).flatten()
                    embeddings.extend(
                        discovery.address_mean[batch_index, unmatched_query].unbind(dim=0)
                    )
                    unmatched_count = unmatched_query.numel()
                    identity_keys.extend([None] * unmatched_count)
                    time_indices.extend([time_index] * unmatched_count)
                    physical_identity_view.extend([False] * unmatched_count)
                    null_view_count += unmatched_count
            view_count += len(embeddings)
            sample_is_eligible = any(len(times) >= 2 for times in identity_times.values())
            eligible_samples += int(sample_is_eligible)
            if len(embeddings) < 2:
                continue

            normalized = F.normalize(torch.stack(embeddings).float(), dim=-1)
            cosine = normalized @ normalized.transpose(0, 1)
            if relation_logit_scale is not None and relation_logit_bias is not None:
                logits = spherical_relation_logits(
                    cosine,
                    logit_scale=relation_logit_scale,
                    logit_bias=relation_logit_bias,
                )
            else:
                logits = cosine / self.config.temperature + self.config.effective_logit_bias
            time = torch.tensor(time_indices, dtype=torch.long, device=logits.device)
            different_time = time[:, None] != time[None, :]
            is_physical = torch.tensor(
                physical_identity_view,
                dtype=torch.bool,
                device=logits.device,
            )
            same_identity = torch.tensor(
                [[left == right for right in identity_keys] for left in identity_keys],
                dtype=torch.bool,
                device=logits.device,
            ) & (is_physical[:, None] & is_physical[None, :])
            physical_pair = is_physical[:, None] & is_physical[None, :]
            null_relation = is_physical[:, None] ^ is_physical[None, :]
            positive = different_time & same_identity
            null_negative = different_time & null_relation
            negative = different_time & ((physical_pair & ~same_identity) | null_relation)
            if not positive.any():
                continue
            positive_count += int(positive.sum().item())
            negative_count += int(negative.sum().item())
            null_negative_count += int(null_negative.sum().item())
            # A positive-only graph can state invariance but cannot identify a
            # match/noise density ratio. Keep its pair diagnostics, but do not
            # claim temporal-credit coverage or optimize an arbitrary offset.
            if not negative.any():
                continue
            covered_eligible_samples += int(sample_is_eligible)

            if self.config.objective == "sigmoid":
                losses.append(sigmoid_density_ratio_loss(logits, positive, negative))
            else:
                anchor_losses = []
                for anchor in range(logits.shape[0]):
                    candidates = different_time[anchor]
                    positives = positive[anchor]
                    if positives.any():
                        anchor_losses.append(
                            torch.logsumexp(logits[anchor, candidates], dim=0)
                            - torch.logsumexp(logits[anchor, positives], dim=0)
                        )
                if anchor_losses:
                    losses.append(torch.stack(anchor_losses).mean())

        loss = torch.stack(losses).mean() if losses else zero
        return TemporalAddressBindingOutput(
            loss=loss,
            address_views=view_count,
            null_address_views=null_view_count,
            positive_pairs=positive_count,
            negative_pairs=negative_count,
            null_negative_pairs=null_negative_count,
            eligible_samples=eligible_samples,
            covered_eligible_samples=covered_eligible_samples,
        )


def _pool_object_modality_views(
    projection: BindingProjectionOutput,
    target: ObjectSetTarget,
    batch_index: int,
    *,
    minimum_mass: float,
) -> _ObjectModalityViews:
    features = projection.binding_features[batch_index]
    modality = projection.modality_index[batch_index]
    supervised = target.supervision_valid
    object_membership = target.ownership[:, :-1]
    modality_ids = torch.unique(modality[supervised])
    # [tokens, objects, modalities]. This is the same weighted mean as the
    # former nested loop, but the number of objects no longer controls host
    # synchronization or Python launch count.
    selected = supervised[:, None] & (modality[:, None] == modality_ids[None, :])
    weight = object_membership[:, :, None].float() * selected[:, None, :]
    mass = weight.sum(dim=0)
    pooled = torch.einsum("tom,td->omd", weight, features.float())
    pooled = pooled / mass.clamp_min(minimum_mass).unsqueeze(-1)
    keep = mass >= minimum_mass
    object_index = torch.arange(target.num_objects, device=features.device)[:, None].expand_as(mass)
    modality_index = modality_ids[None, :].expand_as(mass)
    return _ObjectModalityViews(
        embeddings=pooled[keep],
        object_index=object_index[keep],
        modality_index=modality_index[keep],
        count=keep.sum(),
    )


def _validate_inputs(
    projection: BindingProjectionOutput,
    targets: tuple[ObjectSetTarget, ...] | list[ObjectSetTarget],
) -> None:
    batch_size, token_count, _ = projection.binding_features.shape
    if len(targets) != batch_size:
        raise ValueError("target count must equal binding batch size")
    for batch_index, target in enumerate(targets):
        if target.ownership.shape[0] != token_count:
            raise ValueError("binding targets must align with projected tokens")
        if not torch.is_floating_point(target.ownership):
            raise ValueError("binding ownership must use a floating dtype")
        if target.ownership.requires_grad:
            raise ValueError("loss-only binding ownership must not require gradients")
        if not torch.isfinite(target.ownership).all() or (target.ownership < 0.0).any():
            raise ValueError("binding ownership must be finite and nonnegative")
        if target.token_valid.dtype != torch.bool:
            raise ValueError("binding target validity must be boolean")
        if target.supervision_valid.dtype != torch.bool:
            raise ValueError("binding target supervision must be boolean")
        if not torch.equal(
            target.token_valid,
            projection.current_measurement_valid[batch_index],
        ):
            raise ValueError("binding target and current-measurement validity must match exactly")
        if target.ownership.device != projection.binding_features.device:
            raise ValueError("binding targets and projected features must share a device")
        if target.supervision_valid.device != projection.binding_features.device:
            raise ValueError("binding supervision and projected features must share a device")
        if (target.supervision_valid & ~target.token_valid).any():
            raise ValueError("binding supervision cannot include invalid tokens")
        if (target.ownership[~target.supervision_valid] != 0.0).any():
            raise ValueError("unsupervised binding ownership must be exactly zero")
        if target.supervision_valid.any():
            row_sum = target.ownership[target.supervision_valid].float().sum(dim=-1)
            tolerance = max(1e-5, torch.finfo(target.ownership.dtype).eps)
            if not torch.allclose(
                row_sum,
                torch.ones_like(row_sum),
                atol=tolerance,
                rtol=tolerance,
            ):
                raise ValueError("supervised binding ownership rows must sum to one")
