from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as fn

from openpi.picf.contracts import PicfObservation
from openpi.picf.core.contracts import PicfCoreOutput
from openpi.picf.core.contracts import PicfCoreState
from openpi.picf.core.pipeline import PicfFullCore


@dataclasses.dataclass(frozen=True)
class PicfFutureTargets:
    visual_latent: torch.Tensor | None
    visual_real: torch.Tensor | None
    tactile_real: torch.Tensor | None
    point_real: torch.Tensor | None
    availability: torch.Tensor
    posterior_tokens: torch.Tensor | None = None
    posterior_support_summary: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class PicfTransitionLossConfig:
    lambda_action_pos: float = 2.0
    lambda_action_rot: float = 2.0
    lambda_action_gripper: float = 2.0
    lambda_visual_latent: float = 0.2
    lambda_visual_real: float = 0.1
    lambda_tactile_real: float = 0.3
    lambda_point_real: float = 0.3
    lambda_semantic_future_aux: float = 0.25
    lambda_anchor_pv: float = 0.1
    lambda_pv_weak: float = 0.02
    lambda_pt: float = 1.0
    tau_pv: float = 0.07
    tau_pt: float = 0.07
    tau_route_p: float = 0.1
    tau_route_v: float = 0.1
    pt_bag_radius_m: float = 0.045
    pt_bag_sigma_m: float = 0.015
    pt_bag_kmin: int = 32
    pt_back_slack_m: float = 0.008
    p_align_on: float = 0.55
    p_align_off: float = 0.35
    tactile_aux_force_scale: float = 1.0
    tactile_aux_indent_scale: float = 5e-4
    tactile_aux_pressure_scale: float = 0.1
    tactile_aux_pose_scale: float = 0.10
    tactile_aux_huber_delta: float = 1.0
    enable_aux_budgeting: bool = True
    aux_budget_physical_ratio: float = 0.20
    aux_budget_semantic_ratio: float = 0.10
    aux_budget_alignment_ratio: float = 0.05
    aux_budget_floor: float = 0.25
    lambda_vl_heatmap_task: float = 0.0
    lambda_vl_heatmap_effector: float = 0.0
    lambda_vl_heatmap_interaction: float = 0.0
    lambda_vl_point_consistency: float = 0.0
    lambda_vl_anchor_diversity: float = 0.0
    vl_heatmap_sigma_patches: float = 1.5
    vl_point_consistency_eps: float = 1e-6
    vl_anchor_diversity_radius_m: float = 0.04
    lambda_mapg_siglip: float = 0.0
    lambda_mapg_vicreg: float = 0.0
    lambda_mapg_cycle: float = 0.02
    lambda_mapg_masked_modality: float = 0.0
    lambda_mapg_routing: float = 0.0
    lambda_mapg_support_diversity: float = 0.01
    lambda_mapg_geometry_diversity: float = 0.0
    lambda_slot_jepa: float = 0.0
    lambda_support_pred: float = 0.0
    lambda_binding_consistency: float = 0.0
    lambda_aqr_denoising: float = 0.0
    mapg_siglip_tau: float = 0.07
    mapg_vicreg_var_target: float = 1.0
    mapg_vicreg_cov_weight: float = 0.04
    mapg_support_div_margin_visual: float = 0.15
    mapg_support_div_margin_point: float = 0.15
    mapg_support_div_margin_tactile: float = 0.25
    mapg_support_div_margin_posterior: float = 0.10
    mapg_support_div_sigma_visual_patches: float = 1.0
    mapg_support_div_sigma_point_m: float = 0.04
    mapg_geometry_diversity_margin: float = 1.0
    mapg_geometry_diversity_jitter_m: float = 0.005


@dataclasses.dataclass(frozen=True)
class PicfAlignmentLossConfig:
    lambda_anchor_pv: float = 1.0
    lambda_pv_weak: float = 0.2
    lambda_pt: float = 1.0
    tau_pv: float = 0.07
    tau_pt: float = 0.07
    tau_route_p: float = 0.1
    tau_route_v: float = 0.1
    pt_bag_radius_m: float = 0.045
    pt_bag_sigma_m: float = 0.015
    pt_bag_kmin: int = 32
    pt_back_slack_m: float = 0.008
    p_align_on: float = 0.55
    p_align_off: float = 0.35


@dataclasses.dataclass(frozen=True)
class PicfTransitionLossBreakdown:
    total: torch.Tensor
    action: torch.Tensor
    action_active7: torch.Tensor
    action_pos: torch.Tensor
    action_rot: torch.Tensor
    action_gripper: torch.Tensor
    visual_latent: torch.Tensor
    visual_real: torch.Tensor
    tactile_real: torch.Tensor
    tactile_map: torch.Tensor
    tactile_aux: torch.Tensor
    point_real: torch.Tensor
    semantic_future_aux: torch.Tensor
    semantic_group_raw: torch.Tensor
    semantic_group_capped: torch.Tensor
    physical_aux: torch.Tensor
    physical_aux_capped: torch.Tensor
    alignment: torch.Tensor
    alignment_raw: torch.Tensor
    total_minus_action: torch.Tensor
    anchor_pv: torch.Tensor
    pv_weak: torch.Tensor
    pt: torch.Tensor
    availability: torch.Tensor
    physical_aux_budget_scale: torch.Tensor
    semantic_aux_budget_scale: torch.Tensor
    alignment_budget_scale: torch.Tensor
    vl_router: torch.Tensor
    vl_heatmap_task: torch.Tensor
    vl_heatmap_effector: torch.Tensor
    vl_heatmap_interaction: torch.Tensor
    vl_point_consistency: torch.Tensor
    vl_anchor_diversity: torch.Tensor
    mapg_graph: torch.Tensor
    mapg_siglip: torch.Tensor
    mapg_vicreg: torch.Tensor
    mapg_cycle: torch.Tensor
    mapg_masked_modality: torch.Tensor
    mapg_routing: torch.Tensor
    mapg_support_diversity: torch.Tensor
    mapg_geometry_diversity: torch.Tensor
    slot_jepa: torch.Tensor
    support_pred: torch.Tensor
    binding_consistency: torch.Tensor
    aqr_denoising: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.item()),
            "action": float(self.action.item()),
            "action_active7": float(self.action_active7.item()),
            "action_pos": float(self.action_pos.item()),
            "action_rot": float(self.action_rot.item()),
            "action_gripper": float(self.action_gripper.item()),
            "visual_latent": float(self.visual_latent.item()),
            "visual_real": float(self.visual_real.item()),
            "tactile_real": float(self.tactile_real.item()),
            "tactile_map": float(self.tactile_map.item()),
            "tactile_aux": float(self.tactile_aux.item()),
            "point_real": float(self.point_real.item()),
            "semantic_future_aux": float(self.semantic_future_aux.item()),
            "semantic_group_raw": float(self.semantic_group_raw.item()),
            "semantic_group_capped": float(self.semantic_group_capped.item()),
            "physical_aux": float(self.physical_aux.item()),
            "physical_aux_capped": float(self.physical_aux_capped.item()),
            "alignment": float(self.alignment.item()),
            "alignment_raw": float(self.alignment_raw.item()),
            "total_minus_action": float(self.total_minus_action.item()),
            "anchor_pv": float(self.anchor_pv.item()),
            "pv_weak": float(self.pv_weak.item()),
            "pt": float(self.pt.item()),
            "physical_aux_budget_scale": float(self.physical_aux_budget_scale.item()),
            "semantic_aux_budget_scale": float(self.semantic_aux_budget_scale.item()),
            "alignment_budget_scale": float(self.alignment_budget_scale.item()),
            "vl_router": float(self.vl_router.item()),
            "vl_heatmap_task": float(self.vl_heatmap_task.item()),
            "vl_heatmap_effector": float(self.vl_heatmap_effector.item()),
            "vl_heatmap_interaction": float(self.vl_heatmap_interaction.item()),
            "vl_point_consistency": float(self.vl_point_consistency.item()),
            "vl_anchor_diversity": float(self.vl_anchor_diversity.item()),
            "mapg_graph": float(self.mapg_graph.item()),
            "mapg_siglip": float(self.mapg_siglip.item()),
            "mapg_vicreg": float(self.mapg_vicreg.item()),
            "mapg_cycle": float(self.mapg_cycle.item()),
            "mapg_masked_modality": float(self.mapg_masked_modality.item()),
            "mapg_routing": float(self.mapg_routing.item()),
            "mapg_support_diversity": float(self.mapg_support_diversity.item()),
            "mapg_geometry_diversity": float(self.mapg_geometry_diversity.item()),
            "slot_jepa": float(self.slot_jepa.item()),
            "support_pred": float(self.support_pred.item()),
            "binding_consistency": float(self.binding_consistency.item()),
            "aqr_denoising": float(self.aqr_denoising.item()),
        }


@dataclasses.dataclass(frozen=True)
class PicfAlignmentLossBreakdown:
    total: torch.Tensor
    anchor_pv: torch.Tensor
    pv_weak: torch.Tensor
    pt: torch.Tensor
    candidate_edges: int
    candidate_density: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.item()),
            "anchor_pv": float(self.anchor_pv.item()),
            "pv_weak": float(self.pv_weak.item()),
            "pt": float(self.pt.item()),
            "candidate_edges": float(self.candidate_edges),
            "candidate_density": float(self.candidate_density),
        }


def extract_future_targets(
    core: PicfFullCore,
    observation: PicfObservation,
    *,
    visual_map_override: torch.Tensor | np.ndarray | None = None,
) -> PicfFutureTargets:
    # Future targets are supervision targets derived from the next observation.
    # They must be treated as stop-gradient teacher signals rather than a second
    # trainable branch of the same loss graph.
    with torch.no_grad():
        targets, availability = core.extract_targets(observation, visual_map_override=visual_map_override)
    return PicfFutureTargets(
        visual_latent=targets["visual_latent"],
        visual_real=targets["visual_real"],
        tactile_real=targets["tactile_real"],
        point_real=targets["point_real"],
        availability=availability,
    )


def _posterior_support_summary(posterior: Any | None) -> torch.Tensor | None:
    if posterior is None:
        return None
    tokens = getattr(posterior, "tokens", None)
    if tokens is None or tokens.numel() == 0:
        return None
    device = tokens.device
    dtype = tokens.dtype
    slot_count = int(tokens.shape[0])

    def _slot_scalar(name: str) -> torch.Tensor:
        value = getattr(posterior, name, None)
        if value is None:
            return torch.zeros((slot_count,), device=device, dtype=dtype)
        scalar = value.to(device=device, dtype=dtype).reshape(-1)
        if scalar.numel() < slot_count:
            scalar = fn.pad(scalar, (0, slot_count - scalar.numel()))
        return scalar[:slot_count]

    alpha = _slot_scalar("alpha").clamp(0.0, 1.0)
    support_mass = _slot_scalar("support_mass").clamp(0.0, 1.0)
    contact_prob = _slot_scalar("contact_prob").clamp(0.0, 1.0)
    binding = getattr(posterior, "binding", None)
    if binding is not None and binding.numel() > 0:
        bind_conf = binding.to(device=device, dtype=dtype).reshape(slot_count, -1).amax(dim=-1).clamp(0.0, 1.0)
    else:
        bind_conf = torch.zeros((slot_count,), device=device, dtype=dtype)
    return torch.stack([alpha, support_mass, contact_prob, bind_conf], dim=-1).detach()


def _binding_consistency_loss(
    posterior: Any | None,
    future: PicfFutureTargets,
    *,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Bind sharply now and keep persistent slots aligned to the next posterior.

    The first term discourages diffuse current bindings. The second term is a
    leakage-free temporal identity term: current posterior slots must identify
    their detached next-posterior counterpart better than other slots. It uses
    posterior targets only as stop-gradient teachers and therefore does not feed
    future observations into the action path.
    """

    if posterior is None:
        return _zero_weight_loss(None, reference)

    terms: list[torch.Tensor] = []
    binding = getattr(posterior, "binding", None)
    if binding is not None and binding.numel() > 0 and binding.shape[-1] > 1:
        prob = binding.to(device=reference.device, dtype=reference.dtype).clamp_min(eps)
        prob = prob / torch.clamp(prob.sum(dim=-1, keepdim=True), min=eps)
        entropy = -(prob * torch.log(prob)).sum(dim=-1) / math.log(float(prob.shape[-1]))
        terms.append(entropy.mean())

    current_tokens = getattr(posterior, "tokens", None)
    future_tokens = future.posterior_tokens
    if current_tokens is not None and future_tokens is not None and current_tokens.numel() > 0 and future_tokens.numel() > 0:
        current = current_tokens.to(device=reference.device, dtype=reference.dtype)
        target = future_tokens.detach().to(device=reference.device, dtype=reference.dtype)
        slot_count = min(int(current.shape[0]), int(target.shape[0]))
        width = min(int(current.shape[-1]), int(target.shape[-1]))
        if slot_count > 1 and width > 0:
            current = fn.normalize(current[:slot_count, :width], dim=-1)
            target = fn.normalize(target[:slot_count, :width], dim=-1)
            logits = (current @ target.t()) / 0.1
            assign_row = torch.softmax(logits, dim=-1)
            assign_col = torch.softmax(logits.t(), dim=-1).t()
            assign = 0.5 * (assign_row + assign_col)
            matched_target = fn.normalize(assign @ target, dim=-1)
            matched_current = fn.normalize(assign.t() @ current, dim=-1)
            forward = 1.0 - torch.sum(current * matched_target, dim=-1)
            backward = 1.0 - torch.sum(target * matched_current, dim=-1)
            weight = torch.ones((slot_count,), device=reference.device, dtype=reference.dtype)
            alpha = getattr(posterior, "alpha", None)
            if alpha is not None:
                current_alpha = alpha.to(device=reference.device, dtype=reference.dtype).reshape(-1)[:slot_count].clamp(0.0, 1.0)
                weight = weight * current_alpha
            if future.posterior_support_summary is not None and future.posterior_support_summary.numel() > 0:
                future_alpha = future.posterior_support_summary.detach().to(device=reference.device, dtype=reference.dtype)
                if future_alpha.ndim >= 2 and future_alpha.shape[1] > 0:
                    weight = weight * future_alpha.reshape(future_alpha.shape[0], -1)[:slot_count, 0].clamp(0.0, 1.0)
            if bool((weight.sum() > eps).item()):
                temporal = 0.5 * (
                    (forward * weight).sum() / torch.clamp(weight.sum(), min=eps)
                    + (backward * weight).sum() / torch.clamp(weight.sum(), min=eps)
                )
            else:
                temporal = 0.5 * (forward.mean() + backward.mean())
            terms.append(temporal)

    if not terms:
        return _zero_weight_loss(binding, reference)
    return sum(terms) / float(len(terms))


def _matched_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reference: torch.Tensor,
    eps: float,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation-tolerant detached target matching for slot-level teachers.

    The loss keeps future observations on the target side only, but avoids the
    old index-aligned assumption by softly matching predicted slots to detached
    future slots using cosine cost. This is a differentiable Sinkhorn-lite
    assignment suitable for guarded low-weight OWM losses.
    """

    if pred.numel() == 0 or target.numel() == 0:
        zero = _zero_weight_loss(pred, reference)
        return zero, torch.zeros((0, 0), device=reference.device, dtype=reference.dtype)
    slot_count = min(int(pred.shape[0]), int(target.shape[0]))
    width = min(int(pred.shape[-1]), int(target.shape[-1]))
    if slot_count == 0 or width == 0:
        zero = _zero_weight_loss(pred, reference)
        return zero, torch.zeros((0, 0), device=reference.device, dtype=reference.dtype)
    pred_w = pred[:slot_count, :width]
    target_w = target.detach().to(device=pred.device, dtype=pred.dtype)[:slot_count, :width]
    pred_n = fn.normalize(pred_w, dim=-1)
    target_n = fn.normalize(target_w, dim=-1)
    cost = 1.0 - (pred_n @ target_n.T)
    logits = -cost / max(float(temperature), eps)
    assign_row = torch.softmax(logits, dim=-1)
    assign_col = torch.softmax(logits.T, dim=-1).T
    assign = 0.5 * (assign_row + assign_col)
    matched_target = assign @ target_w
    forward = fn.mse_loss(pred_w, matched_target)
    matched_pred = assign.T @ pred_w
    backward = fn.mse_loss(matched_pred, target_w)
    return 0.5 * (forward + backward), assign.detach()


def _aqr_support_denoising_loss(
    state: PicfCoreState,
    *,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Training-only pseudo-target denoising over confident typed supports.

    This auxiliary never creates inference-time queries and never writes to the
    posterior/action path. It only asks already-produced AQR support distributions
    to be self-consistent around their detached high-confidence peaks. That keeps
    it mathematically a guarded teacher signal, not a new source of truth.
    """

    graph = getattr(state, "anchor_prior_graph", None)
    if graph is None:
        return _zero_like(reference)

    terms: list[torch.Tensor] = []
    priors: tuple[torch.Tensor | None, ...] = (
        getattr(graph, "visual_priors", None),
        getattr(graph, "point_priors", None),
        getattr(graph, "vjepa_temporal_priors", None),
        getattr(graph, "pg_priors", None),
        getattr(graph, "tracklet_priors", None),
        getattr(graph, "proposal_priors", None),
        getattr(graph, "local_priors", None),
    )
    for prior in priors:
        if prior is None or prior.numel() == 0 or prior.shape[-1] <= 1:
            continue
        prob = prior.to(device=reference.device, dtype=reference.dtype).clamp_min(0.0)
        prob = prob / torch.clamp(prob.sum(dim=-1, keepdim=True), min=eps)
        peak = prob.max(dim=-1).values.detach()
        uniform = torch.full_like(peak, 1.0 / float(prob.shape[-1]))
        active = (peak > torch.clamp(uniform + 0.05, max=0.95)).to(dtype=prob.dtype)
        if not bool((active.sum() > 0).item()):
            terms.append(_zero_weight_loss(prior, reference))
            continue
        target = prob.detach().argmax(dim=-1)
        ce = fn.nll_loss(torch.log(torch.clamp(prob, min=eps)), target, reduction="none")
        terms.append((ce * active).sum() / torch.clamp(active.sum(), min=eps))

    if not terms:
        return _zero_like(reference)
    return sum(terms) / float(len(terms))


def future_targets_from_current_targets(
    targets: dict[str, torch.Tensor | None],
    availability: torch.Tensor,
    *,
    posterior: Any | None = None,
) -> PicfFutureTargets:
    """Convert current-step targets into detached teacher targets.

    This is used by the window trainer to reuse the shared middle frame in an
    unrolled window: transition t+1 already computes the current-step targets
    for frame_{t+1}, so transition t can consume those same values as detached
    future supervision instead of rebuilding them a second time.
    """

    def _maybe_detach(x: torch.Tensor | None) -> torch.Tensor | None:
        return None if x is None else x.detach()

    return PicfFutureTargets(
        visual_latent=_maybe_detach(targets.get("visual_latent")),
        visual_real=_maybe_detach(targets.get("visual_real")),
        tactile_real=_maybe_detach(targets.get("tactile_real")),
        point_real=_maybe_detach(targets.get("point_real")),
        availability=availability.detach(),
        posterior_tokens=_maybe_detach(getattr(posterior, "tokens", None)),
        posterior_support_summary=_posterior_support_summary(posterior),
    )


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _zero_weight_loss(pred: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    """Return an exact-zero scalar that still keeps `pred` in the autograd graph.

    DDP requires parameters to participate in the graph on every rank. Some PICF
    supervision branches are data-dependent (for example, tactile/point targets can
    be temporarily unavailable). In those cases the mathematically correct behavior
    is "zero contribution", but engineering-wise we still want the head to appear in
    the graph with zero gradient instead of becoming entirely unused on one rank.
    """
    if pred is None:
        return _zero_like(reference)
    return pred.reshape(-1).sum() * 0.0


def _zero_weight_sum(reference: torch.Tensor, *preds: torch.Tensor | None) -> torch.Tensor:
    loss = _zero_like(reference)
    used = False
    for pred in preds:
        if pred is None:
            continue
        loss = loss + (pred.reshape(-1).sum() * 0.0)
        used = True
    return loss if used else _zero_like(reference)


def make_action_only_transition_loss(
    *,
    reference: torch.Tensor,
    action_loss_override: torch.Tensor,
    action_pos_override: torch.Tensor | None = None,
    action_rot_override: torch.Tensor | None = None,
    action_gripper_override: torch.Tensor | None = None,
    availability_dim: int = 4,
) -> PicfTransitionLossBreakdown:
    """Build a transition-loss record for PI0.5-only ablation runs.

    The ablated mode removes PICF future/alignment supervision entirely but the
    trainer and metric logger still expect the canonical loss dictionary shape.
    This helper keeps that contract stable while making every non-action branch
    an exact zero tensor on the correct device and dtype.
    """

    zero = _zero_like(reference)
    action_pos = action_pos_override if action_pos_override is not None else action_loss_override
    action_rot = action_rot_override if action_rot_override is not None else action_loss_override
    action_gripper = action_gripper_override if action_gripper_override is not None else action_loss_override
    action_active7 = _action_active7_loss(action_pos, action_rot, action_gripper)
    availability = torch.zeros((int(availability_dim),), device=reference.device, dtype=reference.dtype)
    return PicfTransitionLossBreakdown(
        total=action_loss_override,
        action=action_loss_override,
        action_active7=action_active7,
        action_pos=action_pos,
        action_rot=action_rot,
        action_gripper=action_gripper,
        visual_latent=zero,
        visual_real=zero,
        tactile_real=zero,
        tactile_map=zero,
        tactile_aux=zero,
        point_real=zero,
        semantic_future_aux=zero,
        semantic_group_raw=zero,
        semantic_group_capped=zero,
        physical_aux=zero,
        physical_aux_capped=zero,
        alignment=zero,
        alignment_raw=zero,
        total_minus_action=zero,
        anchor_pv=zero,
        pv_weak=zero,
        pt=zero,
        availability=availability,
        physical_aux_budget_scale=zero,
        semantic_aux_budget_scale=zero,
        alignment_budget_scale=zero,
        vl_router=zero,
        vl_heatmap_task=zero,
        vl_heatmap_effector=zero,
        vl_heatmap_interaction=zero,
        vl_point_consistency=zero,
        vl_anchor_diversity=zero,
        mapg_graph=zero,
        mapg_siglip=zero,
        mapg_vicreg=zero,
        mapg_cycle=zero,
        mapg_masked_modality=zero,
        mapg_routing=zero,
        mapg_support_diversity=zero,
        mapg_geometry_diversity=zero,
        slot_jepa=zero,
        support_pred=zero,
        binding_consistency=zero,
        aqr_denoising=zero,
    )


def _sanitize_probability_tensor(x: torch.Tensor, *, eps: float, interior: bool) -> torch.Tensor:
    if x.numel() == 0:
        return x
    finite = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    lo = eps if interior else 0.0
    hi = (1.0 - eps) if interior else 1.0
    return torch.clamp(finite, min=lo, max=hi)


def _branch_is_usable(
    *,
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    pred_available: torch.Tensor,
    target_available: torch.Tensor,
) -> bool:
    return pred is not None and target is not None and bool(pred_available.item()) and bool(target_available.item())


def _action_target_tensor(action_target: torch.Tensor | np.ndarray | None, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if action_target is None:
        return None
    target = torch.as_tensor(action_target, device=device, dtype=dtype).reshape(-1)
    if target.numel() < 7:
        target = fn.pad(target, (0, 7 - target.numel()))
    elif target.numel() > 7:
        target = target[:7]
    return target


def _tactile_split_loss(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    *,
    latent_dim: int,
    grid_dim: int,
    config: PicfTransitionLossConfig,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred is None or target is None:
        zero = _zero_weight_loss(pred, reference)
        return zero, zero, zero
    latent_dim = max(int(latent_dim), 0)
    pred_latent = pred[:latent_dim]
    target_latent = target[:latent_dim]
    latent_loss = fn.mse_loss(pred_latent, target_latent) if latent_dim > 0 else _zero_like(reference)
    pred_map = pred[latent_dim : latent_dim + grid_dim]
    target_map = target[latent_dim : latent_dim + grid_dim]
    map_loss = fn.l1_loss(pred_map, target_map)
    pred_aux = pred[latent_dim + grid_dim :]
    target_aux = target[latent_dim + grid_dim :]
    if pred_aux.numel() == 0 or target_aux.numel() == 0:
        aux_loss = _zero_like(reference)
        nonzero_terms = [latent_loss, map_loss]
        total = torch.stack(nonzero_terms).mean()
        return map_loss, aux_loss, total
    contact_loss = fn.binary_cross_entropy_with_logits(pred_aux[:1], target_aux[:1])
    force_scale = max(float(config.tactile_aux_force_scale), 1e-6)
    indent_scale = max(float(config.tactile_aux_indent_scale), 1e-6)
    pressure_scale = max(float(config.tactile_aux_pressure_scale), 1e-6)
    pose_scale = max(float(config.tactile_aux_pose_scale), 1e-6)
    delta = max(float(config.tactile_aux_huber_delta), 1e-6)
    scaled_pred = torch.stack(
        [
            pred_aux[1] / force_scale,
            pred_aux[2] / indent_scale,
            pred_aux[3] / pressure_scale,
            pred_aux[4],
            pred_aux[5] / pose_scale,
            pred_aux[6] / pose_scale,
            pred_aux[7] / pose_scale,
        ]
    )
    scaled_target = torch.stack(
        [
            target_aux[1] / force_scale,
            target_aux[2] / indent_scale,
            target_aux[3] / pressure_scale,
            target_aux[4],
            target_aux[5] / pose_scale,
            target_aux[6] / pose_scale,
            target_aux[7] / pose_scale,
        ]
    )
    aux_reg = fn.huber_loss(scaled_pred, scaled_target, delta=delta, reduction="mean")
    aux_loss = 0.5 * (contact_loss + aux_reg)
    total = torch.stack([latent_loss, map_loss, aux_loss]).mean()
    return map_loss, aux_loss, total


def _point_split_loss(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    *,
    latent_dim: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if pred is None or target is None:
        return _zero_weight_loss(pred, reference)
    latent_dim = max(int(latent_dim), 0)
    pred_latent = pred[:latent_dim]
    target_latent = target[:latent_dim]
    latent_loss = fn.mse_loss(pred_latent, target_latent) if latent_dim > 0 else _zero_like(reference)
    pred_occ = pred[latent_dim:]
    target_occ = target[latent_dim:]
    occ_loss = fn.binary_cross_entropy_with_logits(pred_occ, target_occ)
    return torch.stack([latent_loss, occ_loss]).mean()


def _budgeted_group(
    loss: torch.Tensor,
    *,
    action_loss: torch.Tensor,
    enabled: bool,
    ratio: float,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        return loss, torch.ones((), device=loss.device, dtype=loss.dtype)
    budget_base = torch.clamp(action_loss.detach(), min=float(floor))
    budget = budget_base * float(ratio)
    scale = torch.clamp(budget / torch.clamp(loss.detach(), min=1e-6), max=1.0)
    return loss * scale, scale


def _action_active7_loss(
    action_pos: torch.Tensor,
    action_rot: torch.Tensor,
    action_gripper: torch.Tensor,
) -> torch.Tensor:
    return ((3.0 * action_pos) + (3.0 * action_rot) + action_gripper) / 7.0


def _weighted_action_override_loss(
    action_loss_override: torch.Tensor,
    action_pos: torch.Tensor,
    action_rot: torch.Tensor,
    action_gripper: torch.Tensor,
    *,
    config: PicfTransitionLossConfig,
) -> torch.Tensor:
    """Scale PI0.5 flow loss when action lambda weights are overridden.

    The PI0.5 action expert returns a canonical full-chunk flow loss plus
    first-action component diagnostics. The canonical loss is the parity path
    for default training, but anchor-only probes need the action lambdas to
    remain authoritative. We therefore preserve exact default behavior and use
    the component diagnostics only to compute a detached scale factor for
    non-default lambda settings.
    """

    default = PicfTransitionLossConfig()
    default_weighted = (
        (float(default.lambda_action_pos) * action_pos)
        + (float(default.lambda_action_rot) * action_rot)
        + (float(default.lambda_action_gripper) * action_gripper)
    )
    requested_weighted = (
        (float(config.lambda_action_pos) * action_pos)
        + (float(config.lambda_action_rot) * action_rot)
        + (float(config.lambda_action_gripper) * action_gripper)
    )
    scale = requested_weighted.detach() / torch.clamp(default_weighted.detach(), min=1e-6)
    return action_loss_override * torch.clamp(scale, min=0.0)


def _world_translation_from_transform(
    transform: torch.Tensor | np.ndarray | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if transform is None:
        return None
    value = torch.as_tensor(transform, device=device, dtype=dtype)
    if value.shape[-2:] == (4, 4):
        return value[..., :3, 3].reshape(-1, 3)[0]
    flat = value.reshape(-1)
    if flat.numel() < 3:
        return None
    return flat[:3]


def _visual_gaussian_target_from_world_xyz(
    core: PicfFullCore,
    state: PicfCoreState,
    xyz_world: torch.Tensor | None,
    *,
    source_hw: tuple[int, int],
    sigma_patches: float,
    reference: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = state.token_field.projective_geometry
    if (
        xyz_world is None
        or core.camera_model is None
        or geometry is None
        or geometry.visual_grid_index.numel() == 0
    ):
        return reference.new_zeros((0,)), torch.zeros((), device=reference.device, dtype=torch.bool)

    visual_grid = geometry.visual_grid_index.to(device=reference.device, dtype=reference.dtype)
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    if source_h <= 1 or source_w <= 1:
        return reference.new_zeros((visual_grid.shape[0],)), torch.zeros((), device=reference.device, dtype=torch.bool)

    C_T_W = torch.as_tensor(core.camera_model.C_T_W, device=reference.device, dtype=reference.dtype)
    homo = torch.cat([xyz_world.to(device=reference.device, dtype=reference.dtype), reference.new_ones((1,))], dim=0)
    xyz_cam = (C_T_W @ homo)[:3]
    z = xyz_cam[2]
    valid = bool(torch.isfinite(z).item()) and float(z.item()) > float(core.config.z_min_m)
    if not valid:
        return reference.new_zeros((visual_grid.shape[0],)), torch.zeros((), device=reference.device, dtype=torch.bool)

    uv_x = (float(core.camera_model.fx) * xyz_cam[0] / torch.clamp(z, min=float(core.config.z_min_m))) + float(core.camera_model.cx)
    uv_y = (float(core.camera_model.fy) * xyz_cam[1] / torch.clamp(z, min=float(core.config.z_min_m))) + float(core.camera_model.cy)
    valid = (
        bool(torch.isfinite(uv_x).item())
        and bool(torch.isfinite(uv_y).item())
        and 0.0 <= float(uv_x.item()) <= float(source_w - 1)
        and 0.0 <= float(uv_y.item()) <= float(source_h - 1)
    )
    if not valid:
        return reference.new_zeros((visual_grid.shape[0],)), torch.zeros((), device=reference.device, dtype=torch.bool)

    grid_w = int(torch.max(visual_grid[:, 0]).item()) + 1
    grid_h = int(torch.max(visual_grid[:, 1]).item()) + 1
    center = torch.stack(
        [
            uv_x * (float(grid_w - 1) / max(source_w - 1, 1)),
            uv_y * (float(grid_h - 1) / max(source_h - 1, 1)),
        ],
        dim=0,
    )
    sigma = max(float(sigma_patches), eps)
    dist2 = torch.sum((visual_grid - center[None, :]) ** 2, dim=-1)
    target = torch.exp(-dist2 / (2.0 * sigma * sigma))
    total = target.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= eps:
        return reference.new_zeros((visual_grid.shape[0],)), torch.zeros((), device=reference.device, dtype=torch.bool)
    return target / torch.clamp(total, min=eps), torch.ones((), device=reference.device, dtype=torch.bool)


def _heatmap_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    valid: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0 or target.numel() == 0 or logits.shape[0] != target.shape[0] or not bool(valid.item()):
        return _zero_weight_loss(logits, reference)
    log_probs = torch.log_softmax(logits.reshape(-1).float(), dim=0)
    target_prob = target.reshape(-1).float().detach()
    target_prob = target_prob / torch.clamp(target_prob.sum(), min=1e-6)
    return (-(target_prob * log_probs).sum()).to(device=reference.device, dtype=reference.dtype)


def _js_distribution_loss(pred: torch.Tensor, target: torch.Tensor, *, eps: float) -> torch.Tensor:
    pred = torch.clamp(torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    target = torch.clamp(torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    pred = pred / torch.clamp(pred.sum(dim=-1, keepdim=True), min=eps)
    target = target / torch.clamp(target.sum(dim=-1, keepdim=True), min=eps)
    midpoint = 0.5 * (pred + target)
    kl_pred = torch.sum(pred * (torch.log(torch.clamp(pred, min=eps)) - torch.log(torch.clamp(midpoint, min=eps))), dim=-1)
    kl_target = torch.sum(target * (torch.log(torch.clamp(target, min=eps)) - torch.log(torch.clamp(midpoint, min=eps))), dim=-1)
    return 0.5 * (kl_pred + kl_target)


def _vl_point_consistency_loss(
    state: PicfCoreState,
    *,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    grounding = state.vl_grounding
    readout = state.task_readout
    if (
        grounding is None
        or not bool(grounding.valid.item())
        or readout.point_public_attention is None
        or readout.point_public_attention.numel() == 0
        or grounding.task_point_prior.numel() == 0
    ):
        preds = () if grounding is None else (grounding.task_point_prior, grounding.interaction_point_prior, grounding.effector_point_prior)
        return _zero_weight_sum(reference, *preds)
    local_count = int(readout.point_weights.shape[0])
    point_count = int(grounding.task_point_prior.shape[0])
    if local_count == 0 or point_count == 0:
        return _zero_weight_sum(reference, grounding.task_point_prior, grounding.interaction_point_prior, grounding.effector_point_prior)
    direct = readout.point_public_attention[:local_count, :point_count]
    if direct.numel() == 0:
        return _zero_weight_sum(reference, direct, grounding.task_point_prior)
    role_ids = (
        readout.local_role_ids.to(device=direct.device, dtype=torch.long)
        if readout.local_role_ids is not None and readout.local_role_ids.numel() == local_count
        else torch.ones((local_count,), device=direct.device, dtype=torch.long)
    )
    scene_prior = grounding.task_point_prior.to(device=direct.device, dtype=direct.dtype) + grounding.interaction_point_prior.to(device=direct.device, dtype=direct.dtype)
    eff_prior = grounding.effector_point_prior.to(device=direct.device, dtype=direct.dtype)
    targets = torch.where((role_ids == 0)[:, None], eff_prior[None, :], scene_prior[None, :])
    target_mass = torch.clamp(targets.sum(dim=-1), min=0.0)
    valid_rows = target_mass > eps
    if not bool(valid_rows.any().item()):
        return _zero_weight_sum(reference, direct, targets)
    losses = _js_distribution_loss(direct[valid_rows].detach(), targets[valid_rows], eps=eps)
    return losses.mean().to(device=reference.device, dtype=reference.dtype)


def _vl_anchor_diversity_loss(
    state: PicfCoreState,
    *,
    reference: torch.Tensor,
    radius_m: float,
) -> torch.Tensor:
    grounding = state.vl_grounding
    if grounding is None or grounding.anchor_x.numel() == 0 or grounding.anchor_roles.numel() == 0:
        preds = () if grounding is None else (grounding.anchor_x,)
        return _zero_weight_sum(reference, *preds)
    roles = grounding.anchor_roles.to(device=grounding.anchor_x.device, dtype=torch.long)
    scene_x = grounding.anchor_x[(roles == 1) | (roles == 2)]
    if scene_x.shape[0] < 2:
        return _zero_weight_sum(reference, grounding.anchor_x)
    dists = torch.cdist(scene_x, scene_x)
    offdiag = ~torch.eye(scene_x.shape[0], device=scene_x.device, dtype=torch.bool)
    radius = max(float(radius_m), 1e-6)
    return torch.exp(-dists[offdiag] / radius).mean().to(device=reference.device, dtype=reference.dtype)


def _vl_router_loss(
    core: PicfFullCore,
    state: PicfCoreState,
    next_observation: PicfObservation,
    *,
    config: PicfTransitionLossConfig,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grounding = state.vl_grounding
    zero = _zero_like(reference)
    if grounding is None:
        return zero, zero, zero, zero, zero, zero

    source_hw = tuple(int(dim) for dim in np.asarray(next_observation.rgb_static).shape[:2])
    current_xyz = _world_translation_from_transform(state.G_t, device=reference.device, dtype=reference.dtype)
    next_transform = next_observation.G_t
    if next_transform is None and getattr(core, "local_frame", None) is not None:
        next_transform = core.local_frame.make_transform(next_observation.robot_obs)
    next_xyz = _world_translation_from_transform(next_transform, device=reference.device, dtype=reference.dtype)
    eff_target, eff_valid = _visual_gaussian_target_from_world_xyz(
        core,
        state,
        current_xyz,
        source_hw=source_hw,
        sigma_patches=float(config.vl_heatmap_sigma_patches),
        reference=reference,
        eps=float(config.vl_point_consistency_eps),
    )
    int_target, int_valid = _visual_gaussian_target_from_world_xyz(
        core,
        state,
        next_xyz,
        source_hw=source_hw,
        sigma_patches=float(config.vl_heatmap_sigma_patches),
        reference=reference,
        eps=float(config.vl_point_consistency_eps),
    )
    task_loss = _heatmap_ce_loss(
        grounding.task_heatmap_logits,
        int_target,
        valid=int_valid,
        reference=reference,
    )
    eff_loss = _heatmap_ce_loss(
        grounding.effector_heatmap_logits,
        eff_target,
        valid=eff_valid,
        reference=reference,
    )
    int_loss = _heatmap_ce_loss(
        grounding.interaction_heatmap_logits,
        int_target,
        valid=int_valid,
        reference=reference,
    )
    point_loss = _vl_point_consistency_loss(
        state,
        reference=reference,
        eps=max(float(config.vl_point_consistency_eps), 1e-8),
    )
    div_loss = _vl_anchor_diversity_loss(
        state,
        reference=reference,
        radius_m=float(config.vl_anchor_diversity_radius_m),
    )
    total = (
        (float(config.lambda_vl_heatmap_task) * task_loss)
        + (float(config.lambda_vl_heatmap_effector) * eff_loss)
        + (float(config.lambda_vl_heatmap_interaction) * int_loss)
        + (float(config.lambda_vl_point_consistency) * point_loss)
        + (float(config.lambda_vl_anchor_diversity) * div_loss)
    )
    return total, task_loss, eff_loss, int_loss, point_loss, div_loss


def _mapg_pair_embeddings(
    state: PicfCoreState,
    *,
    eps: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    graph = state.anchor_prior_graph
    if graph is None:
        return {}, {}
    result: dict[str, torch.Tensor] = {"graph": fn.normalize(graph.anchor_tokens.float(), dim=-1).to(dtype=graph.anchor_tokens.dtype)}
    masks: dict[str, torch.Tensor] = {
        "graph": torch.ones((graph.anchor_tokens.shape[0],), device=graph.anchor_tokens.device, dtype=torch.bool)
    }
    if graph.visual_priors.numel() > 0 and state.token_field.visual_align_embeddings.numel() > 0:
        result["visual"] = fn.normalize(graph.visual_priors @ state.token_field.visual_align_embeddings, dim=-1)
        masks["visual"] = graph.visual_priors.sum(dim=-1) > eps
    if graph.point_priors is not None and graph.point_priors.numel() > 0 and state.token_field.point_align_embeddings.numel() > 0:
        result["point"] = fn.normalize(graph.point_priors @ state.token_field.point_align_embeddings, dim=-1)
        masks["point"] = graph.point_priors.sum(dim=-1) > eps
    if graph.tactile_priors is not None and graph.tactile_priors.numel() > 0 and state.token_field.tactile_align_embeddings.numel() > 0:
        result["tactile"] = fn.normalize(graph.tactile_priors @ state.token_field.tactile_align_embeddings, dim=-1)
        masks["tactile"] = graph.tactile_priors.sum(dim=-1) > eps
    if graph.posterior_priors is not None and graph.posterior_priors.numel() > 0 and state.posterior.tokens.numel() > 0:
        result["posterior"] = fn.normalize(graph.posterior_priors @ state.posterior.tokens, dim=-1)
        masks["posterior"] = graph.posterior_priors.sum(dim=-1) > eps
    result = {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in result.items()}
    return result, masks


def _mapg_siglip_loss(state: PicfCoreState, *, reference: torch.Tensor, tau: float, eps: float) -> torch.Tensor:
    embeddings, masks = _mapg_pair_embeddings(state, eps=eps)
    names = [name for name in ("visual", "point", "tactile", "posterior", "graph") if name in embeddings]
    if len(names) < 2:
        return _zero_weight_sum(reference, *embeddings.values())
    losses = []
    temperature = max(float(tau), eps)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left = embeddings[left_name]
            right = embeddings[right_name]
            if left.shape != right.shape or left.shape[0] == 0:
                continue
            valid = masks.get(left_name, torch.ones((left.shape[0],), device=left.device, dtype=torch.bool))
            valid = valid & masks.get(right_name, torch.ones((right.shape[0],), device=right.device, dtype=torch.bool)).to(device=valid.device)
            if int(valid.sum().item()) < 2:
                continue
            left = left[valid]
            right = right[valid]
            logits = (left @ right.T) / temperature
            labels = torch.eye(logits.shape[0], device=logits.device, dtype=logits.dtype)
            targets = (2.0 * labels) - 1.0
            losses.append(fn.softplus(-targets * logits).mean())
    if not losses:
        return _zero_weight_sum(reference, *embeddings.values())
    return torch.stack(losses).mean().to(device=reference.device, dtype=reference.dtype)


def _mapg_vicreg_loss(
    state: PicfCoreState,
    *,
    reference: torch.Tensor,
    var_target: float,
    cov_weight: float,
    eps: float,
) -> torch.Tensor:
    graph = state.anchor_prior_graph
    if graph is None or graph.anchor_tokens.shape[0] < 2:
        return _zero_weight_sum(reference, None if graph is None else graph.anchor_tokens)
    embeddings, masks = _mapg_pair_embeddings(state, eps=eps)
    losses = []
    for name, values in embeddings.items():
        mask = masks.get(name, torch.ones((values.shape[0],), device=values.device, dtype=torch.bool))
        if int(mask.sum().item()) < 2:
            continue
        x = values[mask].float()
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
        var_loss = torch.mean(fn.relu(float(var_target) - std))
        cov = (x.T @ x) / max(x.shape[0] - 1, 1)
        offdiag = cov - torch.diag(torch.diag(cov))
        cov_loss = (offdiag.pow(2).sum() / max(x.shape[1], 1)) * float(cov_weight)
        losses.append(var_loss + cov_loss)
    if not losses:
        return _zero_weight_sum(reference, graph.anchor_tokens)
    return torch.stack(losses).mean().to(device=reference.device, dtype=reference.dtype)


def _mapg_cycle_loss(state: PicfCoreState, *, reference: torch.Tensor, eps: float) -> torch.Tensor:
    graph = state.anchor_prior_graph
    geometry = state.token_field.projective_geometry
    if (
        graph is None
        or graph.point_priors is None
        or graph.visual_priors.numel() == 0
        or graph.point_priors.numel() == 0
        or geometry is None
        or geometry.projective_compatibility.numel() == 0
    ):
        preds = () if graph is None else (graph.visual_priors, graph.point_priors if graph.point_priors is not None else graph.visual_priors)
        return _zero_weight_sum(reference, *preds)
    compat = torch.clamp(torch.nan_to_num(geometry.projective_compatibility.to(device=reference.device, dtype=reference.dtype), nan=0.0), min=0.0)
    projectable = state.token_field.point_projectable_mask
    if projectable is not None and projectable.shape == (compat.shape[0],):
        compat = compat * projectable.to(device=reference.device, dtype=reference.dtype)[:, None]
    if not bool((compat.sum() > eps).item()):
        return _zero_weight_sum(reference, graph.visual_priors, graph.point_priors)
    visual = torch.clamp(
        torch.nan_to_num(graph.visual_priors.to(device=reference.device, dtype=reference.dtype), nan=0.0, posinf=0.0, neginf=0.0),
        min=0.0,
    )
    point = torch.clamp(
        torch.nan_to_num(graph.point_priors.to(device=reference.device, dtype=reference.dtype), nan=0.0, posinf=0.0, neginf=0.0),
        min=0.0,
    )
    if visual.shape[0] != point.shape[0] or visual.shape[-1] != compat.shape[1] or point.shape[-1] != compat.shape[0]:
        return _zero_weight_sum(reference, graph.visual_priors, graph.point_priors)

    point_given_visual = compat / torch.clamp(compat.sum(dim=0, keepdim=True), min=eps)
    visual_given_point = compat / torch.clamp(compat.sum(dim=1, keepdim=True), min=eps)
    point_from_visual = visual @ point_given_visual.T
    visual_from_point = point @ visual_given_point

    visual = visual / torch.clamp(visual.sum(dim=-1, keepdim=True), min=eps)
    point = point / torch.clamp(point.sum(dim=-1, keepdim=True), min=eps)
    point_from_visual = point_from_visual / torch.clamp(point_from_visual.sum(dim=-1, keepdim=True), min=eps)
    visual_from_point = visual_from_point / torch.clamp(visual_from_point.sum(dim=-1, keepdim=True), min=eps)

    losses = []
    point_valid = (point.sum(dim=-1) > eps) & (point_from_visual.sum(dim=-1) > eps)
    if bool(point_valid.any().item()):
        losses.append(_js_distribution_loss(point[point_valid], point_from_visual[point_valid], eps=eps).mean())
    visual_valid = (visual.sum(dim=-1) > eps) & (visual_from_point.sum(dim=-1) > eps)
    if bool(visual_valid.any().item()):
        losses.append(_js_distribution_loss(visual[visual_valid], visual_from_point[visual_valid], eps=eps).mean())

    visual_cycle = point_from_visual @ visual_given_point
    point_cycle = visual_from_point @ point_given_visual.T
    visual_cycle = visual_cycle / torch.clamp(visual_cycle.sum(dim=-1, keepdim=True), min=eps)
    point_cycle = point_cycle / torch.clamp(point_cycle.sum(dim=-1, keepdim=True), min=eps)
    cycle_visual_valid = (visual.sum(dim=-1) > eps) & (visual_cycle.sum(dim=-1) > eps)
    if bool(cycle_visual_valid.any().item()):
        losses.append(0.5 * _js_distribution_loss(visual[cycle_visual_valid], visual_cycle[cycle_visual_valid], eps=eps).mean())
    cycle_point_valid = (point.sum(dim=-1) > eps) & (point_cycle.sum(dim=-1) > eps)
    if bool(cycle_point_valid.any().item()):
        losses.append(0.5 * _js_distribution_loss(point[cycle_point_valid], point_cycle[cycle_point_valid], eps=eps).mean())
    if not losses:
        return _zero_weight_sum(reference, graph.visual_priors, graph.point_priors)
    return torch.stack(losses).mean().to(device=reference.device, dtype=reference.dtype)


def _mapg_masked_modality_loss(state: PicfCoreState, *, reference: torch.Tensor, eps: float) -> torch.Tensor:
    embeddings, masks = _mapg_pair_embeddings(state, eps=eps)
    graph_embed = embeddings.get("graph")
    if graph_embed is None:
        return _zero_weight_sum(reference, *embeddings.values())
    losses = []
    for name in ("visual", "point", "tactile", "posterior"):
        target = embeddings.get(name)
        if target is None or target.shape != graph_embed.shape:
            continue
        target_mask = masks.get(name, torch.ones((target.shape[0],), device=target.device, dtype=torch.bool))
        others = []
        other_masks = []
        for other_name, other in embeddings.items():
            if other_name in (name, "graph") or other.shape != target.shape:
                continue
            others.append(other)
            other_masks.append(masks.get(other_name, torch.ones((other.shape[0],), device=other.device, dtype=torch.bool)).to(device=target.device))
        if not others:
            continue
        stacked = torch.stack(others, dim=0)
        stacked_masks = torch.stack(other_masks, dim=0).to(dtype=target.dtype)
        denom = torch.clamp(stacked_masks.sum(dim=0, keepdim=False), min=1.0)
        pred = torch.sum(stacked * stacked_masks[:, :, None], dim=0) / denom[:, None]
        pred = fn.normalize(torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        valid = target_mask.to(device=target.device) & (stacked_masks.sum(dim=0) > 0)
        if not bool(valid.any().item()):
            continue
        losses.append(1.0 - torch.sum(pred[valid] * target[valid].detach(), dim=-1).mean())
    if not losses:
        return _zero_weight_sum(reference, *embeddings.values())
    return torch.stack(losses).mean().to(device=reference.device, dtype=reference.dtype)


def _mapg_routing_loss(state: PicfCoreState, *, reference: torch.Tensor, eps: float) -> torch.Tensor:
    graph = state.anchor_prior_graph
    if graph is None:
        return _zero_weight_sum(reference)
    losses = []
    for assignment in (graph.obs_slot_assignment, graph.task_assignment):
        if assignment is None or assignment.numel() == 0:
            continue
        probs = torch.clamp(assignment.to(device=reference.device, dtype=reference.dtype), min=eps)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
        coverage = assignment.sum(dim=0)
        if coverage.numel() > 0:
            coverage = coverage / torch.clamp(coverage.sum(), min=eps)
            target = torch.full_like(coverage, 1.0 / max(int(coverage.numel()), 1))
            balance = _js_distribution_loss(coverage[None, :], target[None, :], eps=eps).mean()
        else:
            balance = entropy * 0.0
        losses.append(entropy + balance)
    if graph.obs_slot_assignment is not None and graph.visual_priors.numel() > 0:
        obs_visual = getattr(state.observation_anchors, "routing_mass_visual", None)
        if obs_visual is not None and obs_visual.numel() > 0 and obs_visual.shape[-1] == graph.visual_priors.shape[-1]:
            pred = graph.obs_slot_assignment.to(device=reference.device, dtype=reference.dtype) @ graph.visual_priors.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            target = obs_visual.to(device=reference.device, dtype=reference.dtype)
            valid = (pred.sum(dim=-1) > eps) & (target.sum(dim=-1) > eps)
            if bool(valid.any().item()):
                losses.append(_js_distribution_loss(pred[valid], target[valid], eps=eps).mean())
        obs_point = getattr(state.observation_anchors, "point_weights", None)
        if (
            graph.point_priors is not None
            and graph.point_priors.numel() > 0
            and obs_point is not None
            and obs_point.numel() > 0
            and obs_point.shape[-1] == graph.point_priors.shape[-1]
        ):
            pred = graph.obs_slot_assignment.to(device=reference.device, dtype=reference.dtype) @ graph.point_priors.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            target = obs_point.to(device=reference.device, dtype=reference.dtype)
            valid = (pred.sum(dim=-1) > eps) & (target.sum(dim=-1) > eps)
            if bool(valid.any().item()):
                losses.append(_js_distribution_loss(pred[valid], target[valid], eps=eps).mean())
    if graph.task_assignment is not None:
        task_visual = getattr(state.task_readout, "visual_weights", None)
        if task_visual is not None and task_visual.numel() > 0 and graph.visual_priors.numel() > 0 and task_visual.shape[-1] == graph.visual_priors.shape[-1]:
            pred = graph.task_assignment.to(device=reference.device, dtype=reference.dtype) @ graph.visual_priors.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            target = task_visual.to(device=reference.device, dtype=reference.dtype)
            valid = (pred.sum(dim=-1) > eps) & (target.sum(dim=-1) > eps)
            if bool(valid.any().item()):
                losses.append(_js_distribution_loss(pred[valid], target[valid], eps=eps).mean())
        if graph.point_priors is not None and graph.point_priors.numel() > 0 and state.task_readout.point_weights.numel() > 0 and state.task_readout.point_weights.shape[-1] == graph.point_priors.shape[-1]:
            pred = graph.task_assignment.to(device=reference.device, dtype=reference.dtype) @ graph.point_priors.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            target = state.task_readout.point_weights.to(device=reference.device, dtype=reference.dtype)
            valid = (pred.sum(dim=-1) > eps) & (target.sum(dim=-1) > eps)
            if bool(valid.any().item()):
                losses.append(_js_distribution_loss(pred[valid], target[valid], eps=eps).mean())
    if not losses:
        return _zero_weight_sum(reference, graph.anchor_tokens)
    return torch.stack(losses).mean().to(device=reference.device, dtype=reference.dtype)


def _mapg_anchor_usage(
    graph,
    *,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    usage = torch.zeros((graph.anchor_tokens.shape[0],), device=reference.device, dtype=reference.dtype)
    used = False
    for assignment in (graph.obs_slot_assignment, graph.task_assignment):
        if assignment is None or assignment.numel() == 0 or assignment.shape[-1] != usage.shape[0]:
            continue
        usage = usage + torch.clamp(assignment.to(device=reference.device, dtype=reference.dtype), min=0.0).sum(dim=0)
        used = True
    if not used:
        return usage
    return usage / torch.clamp(usage.max(), min=eps)


def _mapg_identity_overlap(priors: torch.Tensor, *, eps: float) -> torch.Tensor:
    priors = torch.clamp(torch.nan_to_num(priors.float(), nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    priors = priors / torch.clamp(priors.sum(dim=-1, keepdim=True), min=eps)
    cross = priors @ priors.T
    self_mass = torch.clamp(torch.diag(cross), min=eps)
    return cross / torch.sqrt(torch.clamp(self_mass[:, None] * self_mass[None, :], min=eps))


def _mapg_visual_kernel_overlap(state: PicfCoreState, priors: torch.Tensor, *, sigma_patches: float, eps: float) -> torch.Tensor:
    geometry = state.token_field.projective_geometry
    if geometry is None or geometry.visual_grid_index.shape[0] != priors.shape[-1]:
        return _mapg_identity_overlap(priors, eps=eps)
    coords = geometry.visual_grid_index.to(device=priors.device, dtype=torch.float32)
    if coords.numel() == 0:
        return _mapg_identity_overlap(priors, eps=eps)
    coords = torch.round(coords).to(dtype=torch.long)
    coords = coords - coords.min(dim=0, keepdim=True).values
    width = int(coords[:, 0].max().item()) + 1
    height = int(coords[:, 1].max().item()) + 1
    support_count = int(priors.shape[-1])
    if width <= 0 or height <= 0 or (width * height) > max(4 * support_count, 32768):
        return _mapg_identity_overlap(priors, eps=eps)
    flat_index = (coords[:, 1] * width + coords[:, 0]).to(device=priors.device, dtype=torch.long)
    grid_flat = torch.zeros((priors.shape[0], height * width), device=priors.device, dtype=torch.float32)
    grid_flat.scatter_add_(1, flat_index[None, :].expand(priors.shape[0], -1), priors.to(dtype=torch.float32))
    grid = grid_flat.reshape(priors.shape[0], 1, height, width)
    sigma = max(float(sigma_patches), eps)
    radius = max(int(torch.ceil(torch.tensor(3.0 * sigma)).item()), 1)
    offsets = torch.arange(-radius, radius + 1, device=priors.device, dtype=torch.float32)
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel = kernel / torch.clamp(kernel.sum(), min=eps)
    blur = fn.conv2d(grid, kernel.reshape(1, 1, -1, 1), padding=(radius, 0))
    blur = fn.conv2d(blur, kernel.reshape(1, 1, 1, -1), padding=(0, radius))
    blur_flat = blur.reshape(priors.shape[0], height * width)
    cross = grid_flat @ blur_flat.T
    self_mass = torch.clamp(torch.diag(cross), min=eps)
    return cross / torch.sqrt(torch.clamp(self_mass[:, None] * self_mass[None, :], min=eps))


def _mapg_point_kernel_overlap(state: PicfCoreState, priors: torch.Tensor, *, sigma_m: float, eps: float) -> torch.Tensor:
    positions = state.token_field.point_positions_world
    if positions is None or positions.shape[0] != priors.shape[-1] or priors.shape[-1] > 4096:
        return _mapg_identity_overlap(priors, eps=eps)
    positions = positions.to(device=priors.device, dtype=torch.float32)
    dist2 = torch.cdist(positions, positions).pow(2)
    sigma = max(float(sigma_m), eps)
    kernel = torch.exp(-dist2 / (2.0 * sigma * sigma))
    priors_f = torch.clamp(torch.nan_to_num(priors.float(), nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
    priors_f = priors_f / torch.clamp(priors_f.sum(dim=-1, keepdim=True), min=eps)
    cross = (priors_f @ kernel) @ priors_f.T
    self_mass = torch.clamp(torch.diag(cross), min=eps)
    return cross / torch.sqrt(torch.clamp(self_mass[:, None] * self_mass[None, :], min=eps))


def _mapg_support_overlap_loss(
    state: PicfCoreState,
    *,
    config: PicfTransitionLossConfig,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    graph = state.anchor_prior_graph
    if graph is None or graph.anchor_tokens.shape[0] < 2:
        return _zero_weight_sum(reference, None if graph is None else graph.anchor_tokens)
    usage = _mapg_anchor_usage(graph, reference=reference, eps=eps)
    if not bool((usage.sum() > eps).item()):
        return _zero_weight_sum(reference, graph.anchor_tokens)
    roles = graph.anchor_roles.to(device=reference.device, dtype=torch.long)
    same_role = roles[:, None] == roles[None, :]
    pair_mask = torch.triu(same_role, diagonal=1)
    confidence = torch.clamp(graph.anchor_confidence.to(device=reference.device, dtype=reference.dtype), min=0.0, max=1.0)
    base_weight = usage[:, None] * usage[None, :] * confidence[:, None] * confidence[None, :]
    base_weight = torch.where(pair_mask, base_weight, torch.zeros_like(base_weight))
    if not bool((base_weight.sum() > eps).item()):
        return _zero_weight_sum(reference, graph.anchor_tokens)

    def _one_modality(priors: torch.Tensor | None, overlap: torch.Tensor | None, margin: float) -> torch.Tensor | None:
        if priors is None or priors.numel() == 0 or overlap is None:
            return None
        priors_ref = priors.to(device=reference.device, dtype=reference.dtype)
        valid = priors_ref.sum(dim=-1) > eps
        weights = torch.where(valid[:, None] & valid[None, :], base_weight, torch.zeros_like(base_weight))
        if not bool((weights.sum() > eps).item()):
            return None
        penalty = fn.relu(overlap.to(device=reference.device, dtype=reference.dtype) - float(margin)).pow(2)
        return (weights * penalty).sum() / torch.clamp(weights.sum(), min=eps)

    losses = []
    losses.append(
        _one_modality(
            graph.visual_priors,
            _mapg_visual_kernel_overlap(
                state,
                graph.visual_priors.to(device=reference.device, dtype=reference.dtype),
                sigma_patches=float(config.mapg_support_div_sigma_visual_patches),
                eps=eps,
            ),
            float(config.mapg_support_div_margin_visual),
        )
    )
    if graph.point_priors is not None:
        losses.append(
            _one_modality(
                graph.point_priors,
                _mapg_point_kernel_overlap(
                    state,
                    graph.point_priors.to(device=reference.device, dtype=reference.dtype),
                    sigma_m=float(config.mapg_support_div_sigma_point_m),
                    eps=eps,
                ),
                float(config.mapg_support_div_margin_point),
            )
        )
    if graph.tactile_priors is not None:
        losses.append(
            _one_modality(
                graph.tactile_priors,
                _mapg_identity_overlap(graph.tactile_priors.to(device=reference.device, dtype=reference.dtype), eps=eps),
                float(config.mapg_support_div_margin_tactile),
            )
        )
    if graph.posterior_priors is not None:
        losses.append(
            _one_modality(
                graph.posterior_priors,
                _mapg_identity_overlap(graph.posterior_priors.to(device=reference.device, dtype=reference.dtype), eps=eps),
                float(config.mapg_support_div_margin_posterior),
            )
        )
    losses = [loss for loss in losses if loss is not None]
    if not losses:
        return _zero_weight_sum(reference, graph.anchor_tokens, graph.visual_priors)
    return torch.stack(losses).mean().to(device=reference.device, dtype=reference.dtype)


def _mapg_geometry_diversity_loss(
    state: PicfCoreState,
    *,
    config: PicfTransitionLossConfig,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    graph = state.anchor_prior_graph
    if graph is None or graph.anchor_x is None or graph.anchor_S is None or graph.anchor_x.shape[0] < 2:
        return _zero_weight_sum(reference, None if graph is None else graph.anchor_tokens)
    usage = _mapg_anchor_usage(graph, reference=reference, eps=eps)
    if not bool((usage.sum() > eps).item()):
        return _zero_weight_sum(reference, graph.anchor_tokens)
    x = graph.anchor_x.to(device=reference.device, dtype=torch.float32)
    S = graph.anchor_S.to(device=reference.device, dtype=torch.float32)
    valid = graph.geometry_valid.to(device=reference.device)
    roles = graph.anchor_roles.to(device=reference.device, dtype=torch.long)
    confidence = torch.clamp(graph.anchor_confidence.to(device=reference.device, dtype=reference.dtype), min=0.0, max=1.0)
    same_role = roles[:, None] == roles[None, :]
    pair_mask = torch.triu(same_role & valid[:, None] & valid[None, :], diagonal=1)
    if not bool(pair_mask.any().item()):
        return _zero_weight_sum(reference, graph.anchor_tokens, graph.anchor_x, graph.anchor_S)
    diff = x[:, None, :] - x[None, :, :]
    jitter = max(float(config.mapg_geometry_diversity_jitter_m), eps)
    eye = torch.eye(3, device=reference.device, dtype=torch.float32)
    cov = S[:, None, :, :] + S[None, :, :, :] + ((jitter * jitter) * eye[None, None, :, :])
    flat_cov = cov[pair_mask]
    flat_diff = diff[pair_mask]
    solved = torch.linalg.solve(flat_cov, flat_diff[..., None]).squeeze(-1)
    d2 = torch.clamp(torch.sum(flat_diff * solved, dim=-1), min=0.0)
    distance = torch.sqrt(d2 + eps)
    weights = usage[:, None] * usage[None, :] * confidence[:, None] * confidence[None, :]
    flat_weights = torch.clamp(weights[pair_mask].to(device=reference.device, dtype=reference.dtype), min=0.0)
    if not bool((flat_weights.sum() > eps).item()):
        return _zero_weight_sum(reference, graph.anchor_tokens, graph.anchor_x, graph.anchor_S)
    penalty = fn.relu(float(config.mapg_geometry_diversity_margin) - distance.to(dtype=reference.dtype)).pow(2)
    return (flat_weights * penalty).sum() / torch.clamp(flat_weights.sum(), min=eps)


def _mapg_graph_loss(
    state: PicfCoreState,
    *,
    config: PicfTransitionLossConfig,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    graph = state.anchor_prior_graph
    zero = _zero_like(reference)
    if graph is None:
        return zero, zero, zero, zero, zero, zero, zero, zero
    eps = max(float(config.vl_point_consistency_eps), 1e-8)
    siglip = _mapg_siglip_loss(state, reference=reference, tau=float(config.mapg_siglip_tau), eps=eps)
    vicreg = _mapg_vicreg_loss(
        state,
        reference=reference,
        var_target=float(config.mapg_vicreg_var_target),
        cov_weight=float(config.mapg_vicreg_cov_weight),
        eps=eps,
    )
    cycle = _mapg_cycle_loss(state, reference=reference, eps=eps)
    masked = _mapg_masked_modality_loss(state, reference=reference, eps=eps)
    routing = _mapg_routing_loss(state, reference=reference, eps=eps)
    support_diversity = _mapg_support_overlap_loss(state, config=config, reference=reference, eps=eps)
    geometry_diversity = _mapg_geometry_diversity_loss(state, config=config, reference=reference, eps=eps)
    total = (
        (float(config.lambda_mapg_siglip) * siglip)
        + (float(config.lambda_mapg_vicreg) * vicreg)
        + (float(config.lambda_mapg_cycle) * cycle)
        + (float(config.lambda_mapg_masked_modality) * masked)
        + (float(config.lambda_mapg_routing) * routing)
        + (float(config.lambda_mapg_support_diversity) * support_diversity)
        + (float(config.lambda_mapg_geometry_diversity) * geometry_diversity)
    )
    return total, siglip, vicreg, cycle, masked, routing, support_diversity, geometry_diversity


def _routing_responsibilities(mass: torch.Tensor, *, eps: float) -> torch.Tensor:
    if mass.numel() == 0:
        return mass
    mass = torch.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0)
    mass = torch.clamp(mass, min=0.0)
    denom = torch.clamp(mass.sum(dim=0, keepdim=True), min=eps)
    return mass / denom


def _routing_support_gate(
    support_mass: torch.Tensor,
    *,
    tau: float,
    eps: float,
) -> torch.Tensor:
    if support_mass.numel() == 0:
        return support_mass
    support_mass = torch.nan_to_num(support_mass, nan=0.0, posinf=0.0, neginf=0.0)
    support_mass = torch.clamp(support_mass, min=0.0)
    return support_mass / torch.clamp(support_mass + tau, min=eps)


def _routing_consistency(
    state: PicfCoreState,
    *,
    config: PicfAlignmentLossConfig,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = state.observation_anchors
    point_resp = _routing_responsibilities(obs.routing_mass_point, eps=eps)
    visual_resp = _routing_responsibilities(obs.routing_mass_visual, eps=eps)
    point_support = obs.routing_support_point
    visual_support = obs.routing_support_visual
    point_gate = obs.routing_gate_point
    visual_gate = obs.routing_gate_visual
    if point_support.numel() != obs.routing_mass_point.shape[1]:
        point_support = obs.routing_mass_point.sum(dim=0)
        point_gate = _routing_support_gate(point_support, tau=max(float(config.tau_route_p), eps), eps=eps)
    if visual_support.numel() != obs.routing_mass_visual.shape[1]:
        visual_support = obs.routing_mass_visual.sum(dim=0)
        visual_gate = _routing_support_gate(visual_support, tau=max(float(config.tau_route_v), eps), eps=eps)
    routing = (point_resp.T @ visual_resp) * point_gate[:, None] * visual_gate[None, :]
    return routing, point_support, visual_support, point_gate, visual_gate


def _point_tactile_alignment(
    state: PicfCoreState,
    *,
    config: PicfAlignmentLossConfig,
    eps: float,
) -> torch.Tensor:
    token_field = state.token_field
    point_embed = token_field.point_align_embeddings
    tactile_embed = token_field.tactile_align_embeddings
    point_positions = token_field.point_positions_world
    if point_positions is None or point_positions.shape != token_field.point_positions.shape:
        point_positions = token_field.point_positions
    tactile_positions = token_field.tactile_positions_world
    tactile_prob = token_field.tactile_contact_prob
    tactile_gate = token_field.tactile_contact_gate
    tactile_normals = token_field.tactile_normals_world
    tactile_group_ids = token_field.tactile_group_ids
    zero = token_field.fused_tokens.new_zeros(())
    if point_embed.shape[0] == 0 or tactile_embed.shape[0] == 0 or point_positions.shape[0] == 0 or tactile_positions.shape[0] == 0:
        return _zero_weight_sum(zero, point_embed, tactile_embed)
    if point_embed.shape[0] != point_positions.shape[0]:
        raise RuntimeError(
            "PICF point-tactile alignment contract violated: "
            f"point_embed.shape[0]={int(point_embed.shape[0])} "
            f"!= point_positions.shape[0]={int(point_positions.shape[0])}"
        )
    if tactile_embed.shape[0] != tactile_positions.shape[0]:
        raise RuntimeError(
            "PICF point-tactile alignment contract violated: "
            f"tactile_embed.shape[0]={int(tactile_embed.shape[0])} "
            f"!= tactile_positions.shape[0]={int(tactile_positions.shape[0])}"
        )
    if tactile_prob is not None and tactile_prob.numel() != tactile_embed.shape[0]:
        if (
            tactile_group_ids is not None
            and tactile_group_ids.numel() == tactile_embed.shape[0]
            and tactile_prob.numel() > 0
            and int(tactile_group_ids.max().item()) < int(tactile_prob.numel())
        ):
            tactile_prob = tactile_prob.index_select(0, tactile_group_ids)
        else:
            tactile_prob = None
    if tactile_prob is None or tactile_prob.numel() != tactile_embed.shape[0]:
        if (
            tactile_gate is not None
            and tactile_group_ids is not None
            and tactile_group_ids.numel() == tactile_embed.shape[0]
            and tactile_gate.numel() > 0
            and int(tactile_group_ids.max().item()) < int(tactile_gate.numel())
        ):
            tactile_prob = tactile_gate.index_select(0, tactile_group_ids)
        else:
            tactile_prob = tactile_gate
    if tactile_prob is None or tactile_prob.numel() != tactile_embed.shape[0]:
        tactile_prob = torch.ones((tactile_embed.shape[0],), device=tactile_embed.device, dtype=tactile_embed.dtype)
    tau_pt = max(float(config.tau_pt), eps)
    radius = max(float(config.pt_bag_radius_m), eps)
    sigma = max(float(config.pt_bag_sigma_m), eps)
    k_min = max(int(config.pt_bag_kmin), 1)
    p_align_on = max(float(config.p_align_on), eps)
    p_align_off = max(min(float(config.p_align_off), p_align_on - eps), 0.0)
    losses = []
    weights = []
    for tactile_index in range(tactile_embed.shape[0]):
        weight = torch.clamp((tactile_prob[tactile_index] - p_align_off) / max(p_align_on - p_align_off, eps), min=0.0, max=1.0)
        if float(weight.item()) <= 0.0:
            continue
        diffs = point_positions - tactile_positions[tactile_index][None, :]
        dists = torch.linalg.norm(diffs, dim=-1)
        candidate_mask = dists <= radius
        if tactile_normals is not None and tactile_normals.shape == tactile_positions.shape:
            normal = tactile_normals[tactile_index]
            candidate_mask = candidate_mask & ((diffs @ normal) >= -float(config.pt_back_slack_m))
        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1)
        if candidate_idx.numel() < k_min:
            candidate_idx = torch.topk(dists, k=min(k_min, dists.shape[0]), largest=False).indices
        if candidate_idx.numel() == 0:
            continue
        selected_dists = dists[candidate_idx]
        alpha = torch.exp(-(selected_dists**2) / (2.0 * (sigma**2)))
        pooled = torch.sum(alpha[:, None] * point_embed[candidate_idx], dim=0) / torch.clamp(alpha.sum(), min=eps)
        logits = (pooled @ tactile_embed.T) / tau_pt
        target = torch.tensor([tactile_index], device=logits.device, dtype=torch.long)
        losses.append(weight * fn.cross_entropy(logits[None, :], target))
        weights.append(weight)
    if not losses:
        return _zero_weight_sum(zero, point_embed, tactile_embed)
    return torch.stack(losses).sum() / torch.clamp(torch.stack(weights).sum(), min=eps)


def _validate_alignment_contract(
    state: PicfCoreState,
    *,
    candidate_mask: torch.Tensor,
    projective: torch.Tensor,
    routing: torch.Tensor,
) -> None:
    token_field = state.token_field
    point_count = int(token_field.point_tokens.shape[0])
    visual_count = int(token_field.visual_tokens.shape[0])
    expected = (point_count, visual_count)
    actual_mask = tuple(int(dim) for dim in candidate_mask.shape)
    actual_projective = tuple(int(dim) for dim in projective.shape)
    actual_routing = tuple(int(dim) for dim in routing.shape)
    if actual_mask != expected:
        raise RuntimeError(
            "PICF alignment contract violated: candidate mask shape mismatch. "
            f"expected={expected} got={actual_mask}"
        )
    if actual_projective != expected:
        raise RuntimeError(
            "PICF alignment contract violated: projective compatibility shape mismatch. "
            f"expected={expected} got={actual_projective}"
        )
    if actual_routing != expected:
        raise RuntimeError(
            "PICF alignment contract violated: routing shape mismatch. "
            f"expected={expected} got={actual_routing}"
        )
    if int(token_field.point_align_embeddings.shape[0]) != point_count:
        raise RuntimeError(
            "PICF alignment contract violated: point alignment embedding count mismatch. "
            f"point_tokens={point_count} point_align_embeddings={int(token_field.point_align_embeddings.shape[0])}"
        )
    if int(token_field.visual_align_embeddings.shape[0]) != visual_count:
        raise RuntimeError(
            "PICF alignment contract violated: visual alignment embedding count mismatch. "
            f"visual_tokens={visual_count} visual_align_embeddings={int(token_field.visual_align_embeddings.shape[0])}"
        )
    fusion_attention = token_field.fusion_attention_mean
    if fusion_attention is not None:
        total_tokens = int(token_field.fused_tokens.shape[0])
        if tuple(int(dim) for dim in fusion_attention.shape) != (total_tokens, total_tokens):
            raise RuntimeError(
                "PICF alignment contract violated: fusion attention shape mismatch. "
                f"expected={(total_tokens, total_tokens)} got={tuple(int(dim) for dim in fusion_attention.shape)}"
            )


def compute_alignment_loss(
    state: PicfCoreState,
    *,
    config: PicfAlignmentLossConfig | None = None,
) -> PicfAlignmentLossBreakdown:
    cfg = config or PicfAlignmentLossConfig()
    token_field = state.token_field
    zero = token_field.fused_tokens.new_zeros(())
    pt = _point_tactile_alignment(state, config=cfg, eps=1e-6)
    geometry = token_field.projective_geometry
    if geometry is None or geometry.projective_candidate_mask.numel() == 0:
        zero_align = _zero_weight_sum(
            zero,
            token_field.point_align_embeddings,
            token_field.visual_align_embeddings,
        )
        total = zero_align + (cfg.lambda_pt * pt)
        return PicfAlignmentLossBreakdown(total=total, anchor_pv=zero_align, pv_weak=zero_align, pt=pt, candidate_edges=0, candidate_density=0.0)

    candidate_mask = geometry.projective_candidate_mask
    projective = _sanitize_probability_tensor(geometry.projective_compatibility, eps=1e-6, interior=False)
    candidate_edges = int(candidate_mask.sum().item())
    candidate_density = float(candidate_edges / max(candidate_mask.numel(), 1))
    if candidate_edges == 0:
        zero_align = _zero_weight_sum(
            zero,
            token_field.point_align_embeddings,
            token_field.visual_align_embeddings,
        )
        total = zero_align + (cfg.lambda_pt * pt)
        return PicfAlignmentLossBreakdown(total=total, anchor_pv=zero_align, pv_weak=zero_align, pt=pt, candidate_edges=0, candidate_density=candidate_density)

    routing, _, _, _, _ = _routing_consistency(state, config=cfg, eps=1e-6)
    routing = _sanitize_probability_tensor(routing, eps=1e-6, interior=True)
    _validate_alignment_contract(
        state,
        candidate_mask=candidate_mask,
        projective=projective,
        routing=routing,
    )
    candidate_weight = candidate_mask.to(dtype=projective.dtype)
    candidate_count = torch.clamp(candidate_weight.sum(), min=1.0)
    anchor_pv = (
        fn.binary_cross_entropy(
            routing,
            projective,
            weight=projective,
            reduction="none",
        )
        * candidate_weight
    ).sum() / candidate_count

    pv_weak = zero
    tau_pv = max(float(cfg.tau_pv), 1e-6)
    point_embed = token_field.point_align_embeddings
    visual_embed = token_field.visual_align_embeddings
    if point_embed.shape[0] > 0 and visual_embed.shape[0] > 1:
        losses = []
        for u in range(visual_embed.shape[0]):
            weights = projective[:, u] * candidate_weight[:, u]
            if float(weights.sum().item()) <= 0.0:
                continue
            bag = torch.sum(weights[:, None] * point_embed, dim=0) / torch.clamp(weights.sum(), min=1e-6)
            bag = fn.normalize(bag[None, :], dim=-1)[0]
            logits = (bag @ visual_embed.T) / tau_pv
            target = torch.tensor([u], device=logits.device, dtype=torch.long)
            losses.append(fn.cross_entropy(logits[None, :], target))
        if losses:
            pv_weak = torch.stack(losses).mean()
        else:
            pv_weak = _zero_weight_sum(zero, point_embed, visual_embed)
    else:
        pv_weak = _zero_weight_sum(zero, point_embed, visual_embed)

    total = (
        (cfg.lambda_anchor_pv * anchor_pv)
        + (cfg.lambda_pv_weak * pv_weak)
        + (cfg.lambda_pt * pt)
    )
    return PicfAlignmentLossBreakdown(
        total=total,
        anchor_pv=anchor_pv,
        pv_weak=pv_weak,
        pt=pt,
        candidate_edges=candidate_edges,
        candidate_density=candidate_density,
    )


def compute_transition_loss(
    core: PicfFullCore,
    output_t: PicfCoreOutput,
    next_observation: PicfObservation,
    *,
    action_target: torch.Tensor | np.ndarray | None,
    next_visual_map_override: torch.Tensor | np.ndarray | None = None,
    config: PicfTransitionLossConfig | None = None,
    action_loss_override: torch.Tensor | None = None,
    action_pos_override: torch.Tensor | None = None,
    action_rot_override: torch.Tensor | None = None,
    action_gripper_override: torch.Tensor | None = None,
    future_targets_override: PicfFutureTargets | None = None,
) -> PicfTransitionLossBreakdown:
    cfg = config or PicfTransitionLossConfig()
    predictive = output_t.state.predictive
    pred_cache = predictive.physical_prediction_cache
    semantic_future_cache = predictive.prediction_cache
    future = (
        future_targets_override
        if future_targets_override is not None
        else extract_future_targets(core, next_observation, visual_map_override=next_visual_map_override)
    )
    tactile_grid_dim = int(core.config.tactile_real_grid**2)
    alignment = compute_alignment_loss(
        output_t.state,
        config=PicfAlignmentLossConfig(
            lambda_anchor_pv=cfg.lambda_anchor_pv,
            lambda_pv_weak=cfg.lambda_pv_weak,
            lambda_pt=cfg.lambda_pt,
            tau_pv=cfg.tau_pv,
            tau_pt=cfg.tau_pt,
            tau_route_p=cfg.tau_route_p,
            tau_route_v=cfg.tau_route_v,
        ),
    )

    if action_loss_override is not None:
        action_pos = action_pos_override if action_pos_override is not None else action_loss_override
        action_rot = action_rot_override if action_rot_override is not None else action_loss_override
        action_gripper = action_gripper_override if action_gripper_override is not None else action_loss_override
        action_loss = _weighted_action_override_loss(
            action_loss_override,
            action_pos,
            action_rot,
            action_gripper,
            config=cfg,
        )
    else:
        action_target_t = _action_target_tensor(
            action_target,
            device=predictive.action.device,
            dtype=predictive.action.dtype,
        )
        if action_target_t is None:
            action_pos = _zero_weight_loss(predictive.action[:3], predictive.action)
            action_rot = _zero_weight_loss(predictive.action[3:6], predictive.action)
            action_gripper = _zero_weight_loss(predictive.action[6:], predictive.action)
        else:
            action_pos = fn.l1_loss(predictive.action[:3], action_target_t[:3])
            action_rot = fn.l1_loss(predictive.action[3:6], action_target_t[3:6])
            action_gripper = fn.l1_loss(predictive.action[6:], action_target_t[6:])
        action_loss = (
            (cfg.lambda_action_pos * action_pos)
            + (cfg.lambda_action_rot * action_rot)
            + (cfg.lambda_action_gripper * action_gripper)
        )
    action_active7 = _action_active7_loss(action_pos, action_rot, action_gripper)

    if _branch_is_usable(
        pred=pred_cache.visual_latent,
        target=future.visual_latent,
        pred_available=pred_cache.availability[0],
        target_available=future.availability[0],
    ):
        visual_latent = fn.mse_loss(pred_cache.visual_latent, future.visual_latent)
    else:
        visual_latent = _zero_weight_loss(pred_cache.visual_latent, predictive.action)

    if _branch_is_usable(
        pred=pred_cache.visual_real,
        target=future.visual_real,
        pred_available=pred_cache.availability[1],
        target_available=future.availability[1],
    ):
        visual_real = fn.l1_loss(pred_cache.visual_real, future.visual_real)
    else:
        visual_real = _zero_weight_loss(pred_cache.visual_real, predictive.action)

    if _branch_is_usable(
        pred=pred_cache.tactile_real,
        target=future.tactile_real,
        pred_available=pred_cache.availability[2],
        target_available=future.availability[2],
    ):
        tactile_map, tactile_aux, tactile_real = _tactile_split_loss(
            pred_cache.tactile_real,
            future.tactile_real,
            latent_dim=int(core.config.tactile_latent_dim),
            grid_dim=tactile_grid_dim,
            config=cfg,
            reference=predictive.action,
        )
    else:
        tactile_map = _zero_weight_loss(pred_cache.tactile_real, predictive.action)
        tactile_aux = _zero_weight_loss(pred_cache.tactile_real, predictive.action)
        tactile_real = _zero_weight_loss(pred_cache.tactile_real, predictive.action)

    if _branch_is_usable(
        pred=pred_cache.point_real,
        target=future.point_real,
        pred_available=pred_cache.availability[3],
        target_available=future.availability[3],
    ):
        point_real = _point_split_loss(
            pred_cache.point_real,
            future.point_real,
            latent_dim=int(core.config.point_latent_dim),
            reference=predictive.action,
        )
    else:
        point_real = _zero_weight_loss(pred_cache.point_real, predictive.action)

    if _branch_is_usable(
        pred=semantic_future_cache.visual_latent,
        target=future.visual_latent,
        pred_available=semantic_future_cache.availability[0],
        target_available=future.availability[0],
    ):
        semantic_visual_latent = fn.mse_loss(semantic_future_cache.visual_latent, future.visual_latent)
    else:
        semantic_visual_latent = _zero_weight_loss(semantic_future_cache.visual_latent, predictive.action)

    if _branch_is_usable(
        pred=semantic_future_cache.visual_real,
        target=future.visual_real,
        pred_available=semantic_future_cache.availability[1],
        target_available=future.availability[1],
    ):
        semantic_visual_real = fn.l1_loss(semantic_future_cache.visual_real, future.visual_real)
    else:
        semantic_visual_real = _zero_weight_loss(semantic_future_cache.visual_real, predictive.action)

    if _branch_is_usable(
        pred=semantic_future_cache.tactile_real,
        target=future.tactile_real,
        pred_available=semantic_future_cache.availability[2],
        target_available=future.availability[2],
    ):
        _, _, semantic_tactile_real = _tactile_split_loss(
            semantic_future_cache.tactile_real,
            future.tactile_real,
            latent_dim=int(core.config.tactile_latent_dim),
            grid_dim=tactile_grid_dim,
            config=cfg,
            reference=predictive.action,
        )
    else:
        semantic_tactile_real = _zero_weight_loss(semantic_future_cache.tactile_real, predictive.action)

    if _branch_is_usable(
        pred=semantic_future_cache.point_real,
        target=future.point_real,
        pred_available=semantic_future_cache.availability[3],
        target_available=future.availability[3],
    ):
        semantic_point_real = _point_split_loss(
            semantic_future_cache.point_real,
            future.point_real,
            latent_dim=int(core.config.point_latent_dim),
            reference=predictive.action,
        )
    else:
        semantic_point_real = _zero_weight_loss(semantic_future_cache.point_real, predictive.action)

    semantic_future_aux = (
        (cfg.lambda_visual_latent * semantic_visual_latent)
        + (cfg.lambda_visual_real * semantic_visual_real)
        + (cfg.lambda_tactile_real * semantic_tactile_real)
        + (cfg.lambda_point_real * semantic_point_real)
    )
    (
        vl_router_raw,
        vl_heatmap_task,
        vl_heatmap_effector,
        vl_heatmap_interaction,
        vl_point_consistency,
        vl_anchor_diversity,
    ) = _vl_router_loss(
        core,
        output_t.state,
        next_observation,
        config=cfg,
        reference=predictive.action,
    )
    (
        mapg_graph_raw,
        mapg_siglip,
        mapg_vicreg,
        mapg_cycle,
        mapg_masked_modality,
        mapg_routing,
        mapg_support_diversity,
        mapg_geometry_diversity,
    ) = _mapg_graph_loss(
        output_t.state,
        config=cfg,
        reference=predictive.action,
    )
    physical_aux = (
        (cfg.lambda_visual_latent * visual_latent)
        + (cfg.lambda_visual_real * visual_real)
        + (cfg.lambda_tactile_real * tactile_real)
        + (cfg.lambda_point_real * point_real)
    )
    slot_assignment = None
    if predictive.slot_prediction_tokens is not None:
        slot_tokens = predictive.slot_prediction_tokens
        if future.posterior_tokens is not None and future.posterior_tokens.numel() > 0:
            target_slots = future.posterior_tokens.detach().to(device=slot_tokens.device, dtype=slot_tokens.dtype)
            slot_jepa, slot_assignment = _matched_prediction_loss(
                slot_tokens,
                target_slots,
                reference=predictive.action,
                eps=float(core.config.epsilon_a),
            )
        elif future.visual_latent is not None and bool(future.availability[0].item()):
            target_visual = future.visual_latent.detach().reshape(-1)
            hidden_dim = int(slot_tokens.shape[-1])
            if hidden_dim > 0 and target_visual.numel() % hidden_dim == 0:
                target_summary = target_visual.reshape(-1, hidden_dim).mean(dim=0)
                slot_jepa = fn.mse_loss(slot_tokens.mean(dim=0), target_summary)
            else:
                slot_jepa = _zero_weight_loss(slot_tokens, predictive.action)
        else:
            slot_jepa = _zero_weight_loss(slot_tokens, predictive.action)
    else:
        slot_jepa = _zero_weight_loss(predictive.slot_prediction_tokens, predictive.action)

    if predictive.slot_prediction_supports is not None:
        if future.posterior_support_summary is not None and future.posterior_support_summary.numel() > 0:
            support_target = future.posterior_support_summary.detach().to(
                device=predictive.slot_prediction_supports.device,
                dtype=predictive.slot_prediction_supports.dtype,
            )
            slot_count = min(int(support_target.shape[0]), int(predictive.slot_prediction_supports.shape[0]))
            width = min(int(support_target.shape[-1]), int(predictive.slot_prediction_supports.shape[-1]))
            support_target = support_target[:slot_count, :width]
            pred_support = predictive.slot_prediction_supports[:slot_count, :width]
            if (
                slot_assignment is not None
                and slot_assignment.numel() > 0
                and slot_assignment.shape[0] >= slot_count
                and slot_assignment.shape[1] >= slot_count
            ):
                matched_support = slot_assignment[:slot_count, :slot_count].to(device=pred_support.device, dtype=pred_support.dtype) @ support_target.clamp(0.0, 1.0)
                support_pred = fn.mse_loss(pred_support, matched_support)
            else:
                support_pred, _ = _matched_prediction_loss(
                    pred_support,
                    support_target.clamp(0.0, 1.0),
                    reference=predictive.action,
                    eps=float(core.config.epsilon_a),
                )
        else:
            support_target = future.availability.detach().to(
                device=predictive.slot_prediction_supports.device,
                dtype=predictive.slot_prediction_supports.dtype,
            )
            support_target = support_target.clamp(0.0, 1.0)[None, :].expand_as(predictive.slot_prediction_supports)
            support_pred = fn.mse_loss(predictive.slot_prediction_supports, support_target)
    else:
        support_pred = _zero_weight_loss(None, predictive.action)

    binding_consistency = _binding_consistency_loss(
        output_t.state.posterior,
        future,
        reference=predictive.action,
        eps=float(core.config.epsilon_a),
    )
    aqr_denoising = _aqr_support_denoising_loss(
        output_t.state,
        reference=predictive.action,
        eps=float(core.config.epsilon_a),
    )

    guarded_owm_aux = (
        (cfg.lambda_slot_jepa * slot_jepa)
        + (cfg.lambda_support_pred * support_pred)
        + (cfg.lambda_binding_consistency * binding_consistency)
        + (cfg.lambda_aqr_denoising * aqr_denoising)
    )
    semantic_group = cfg.lambda_semantic_future_aux * semantic_future_aux
    alignment_group = alignment.total + vl_router_raw + mapg_graph_raw + guarded_owm_aux
    physical_aux_capped, physical_scale = _budgeted_group(
        physical_aux,
        action_loss=action_loss,
        enabled=bool(cfg.enable_aux_budgeting),
        ratio=float(cfg.aux_budget_physical_ratio),
        floor=float(cfg.aux_budget_floor),
    )
    semantic_group_capped, semantic_scale = _budgeted_group(
        semantic_group,
        action_loss=action_loss,
        enabled=bool(cfg.enable_aux_budgeting),
        ratio=float(cfg.aux_budget_semantic_ratio),
        floor=float(cfg.aux_budget_floor),
    )
    alignment_group_capped, alignment_scale = _budgeted_group(
        alignment_group,
        action_loss=action_loss,
        enabled=bool(cfg.enable_aux_budgeting),
        ratio=float(cfg.aux_budget_alignment_ratio),
        floor=float(cfg.aux_budget_floor),
    )

    total = (
        action_loss
        + physical_aux_capped
        + semantic_group_capped
        + alignment_group_capped
    )
    return PicfTransitionLossBreakdown(
        total=total,
        action=action_loss,
        action_active7=action_active7,
        action_pos=action_pos,
        action_rot=action_rot,
        action_gripper=action_gripper,
        visual_latent=visual_latent,
        visual_real=visual_real,
        tactile_real=tactile_real,
        tactile_map=tactile_map,
        tactile_aux=tactile_aux,
        point_real=point_real,
        semantic_future_aux=semantic_future_aux,
        semantic_group_raw=semantic_group,
        semantic_group_capped=semantic_group_capped,
        physical_aux=physical_aux,
        physical_aux_capped=physical_aux_capped,
        alignment=alignment_group_capped,
        alignment_raw=alignment.total,
        total_minus_action=total - action_loss,
        anchor_pv=alignment.anchor_pv,
        pv_weak=alignment.pv_weak,
        pt=alignment.pt,
        availability=future.availability,
        physical_aux_budget_scale=physical_scale,
        semantic_aux_budget_scale=semantic_scale,
        alignment_budget_scale=alignment_scale,
        vl_router=vl_router_raw,
        vl_heatmap_task=vl_heatmap_task,
        vl_heatmap_effector=vl_heatmap_effector,
        vl_heatmap_interaction=vl_heatmap_interaction,
        vl_point_consistency=vl_point_consistency,
        vl_anchor_diversity=vl_anchor_diversity,
        mapg_graph=mapg_graph_raw,
        mapg_siglip=mapg_siglip,
        mapg_vicreg=mapg_vicreg,
        mapg_cycle=mapg_cycle,
        mapg_masked_modality=mapg_masked_modality,
        mapg_routing=mapg_routing,
        mapg_support_diversity=mapg_support_diversity,
        mapg_geometry_diversity=mapg_geometry_diversity,
        slot_jepa=slot_jepa,
        support_pred=support_pred,
        binding_consistency=binding_consistency,
        aqr_denoising=aqr_denoising,
    )


def detach_core_state(state: Any) -> Any:
    if isinstance(state, torch.Tensor):
        return state.detach()
    if dataclasses.is_dataclass(state):
        values = {field.name: detach_core_state(getattr(state, field.name)) for field in dataclasses.fields(state)}
        return type(state)(**values)
    if isinstance(state, dict):
        return {key: detach_core_state(value) for key, value in state.items()}
    if isinstance(state, list):
        return [detach_core_state(value) for value in state]
    if isinstance(state, tuple):
        return tuple(detach_core_state(value) for value in state)
    return state
