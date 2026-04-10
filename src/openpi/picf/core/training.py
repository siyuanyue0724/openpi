from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class PicfTransitionLossConfig:
    lambda_action_pos: float = 1.0
    lambda_action_rot: float = 1.0
    lambda_action_gripper: float = 1.0
    lambda_visual_latent: float = 0.2
    lambda_visual_real: float = 0.1
    lambda_tactile_real: float = 0.3
    lambda_point_real: float = 0.3
    lambda_semantic_future_aux: float = 0.25
    lambda_anchor_pv: float = 0.1
    lambda_pv_weak: float = 0.02
    lambda_focus_pv: float = 0.0
    lambda_pt: float = 1.0
    tau_pv: float = 0.07
    tau_pt: float = 0.07
    tau_route_p: float = 0.1
    tau_route_v: float = 0.1


@dataclasses.dataclass(frozen=True)
class PicfAlignmentLossConfig:
    lambda_anchor_pv: float = 1.0
    lambda_pv_weak: float = 0.2
    lambda_focus_pv: float = 0.0
    lambda_pt: float = 1.0
    tau_pv: float = 0.07
    tau_pt: float = 0.07
    tau_route_p: float = 0.1
    tau_route_v: float = 0.1


@dataclasses.dataclass(frozen=True)
class PicfTransitionLossBreakdown:
    total: torch.Tensor
    action: torch.Tensor
    action_pos: torch.Tensor
    action_rot: torch.Tensor
    action_gripper: torch.Tensor
    visual_latent: torch.Tensor
    visual_real: torch.Tensor
    tactile_real: torch.Tensor
    point_real: torch.Tensor
    semantic_future_aux: torch.Tensor
    alignment: torch.Tensor
    anchor_pv: torch.Tensor
    pv_weak: torch.Tensor
    focus_pv: torch.Tensor
    pt: torch.Tensor
    availability: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.item()),
            "action": float(self.action.item()),
            "action_pos": float(self.action_pos.item()),
            "action_rot": float(self.action_rot.item()),
            "action_gripper": float(self.action_gripper.item()),
            "visual_latent": float(self.visual_latent.item()),
            "visual_real": float(self.visual_real.item()),
            "tactile_real": float(self.tactile_real.item()),
            "point_real": float(self.point_real.item()),
            "semantic_future_aux": float(self.semantic_future_aux.item()),
            "alignment": float(self.alignment.item()),
            "anchor_pv": float(self.anchor_pv.item()),
            "pv_weak": float(self.pv_weak.item()),
            "focus_pv": float(self.focus_pv.item()),
            "pt": float(self.pt.item()),
        }


@dataclasses.dataclass(frozen=True)
class PicfAlignmentLossBreakdown:
    total: torch.Tensor
    anchor_pv: torch.Tensor
    pv_weak: torch.Tensor
    focus_pv: torch.Tensor
    pt: torch.Tensor
    candidate_edges: int
    candidate_density: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.item()),
            "anchor_pv": float(self.anchor_pv.item()),
            "pv_weak": float(self.pv_weak.item()),
            "focus_pv": float(self.focus_pv.item()),
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
    targets, availability = core.extract_targets(observation, visual_map_override=visual_map_override)
    return PicfFutureTargets(
        visual_latent=targets["visual_latent"],
        visual_real=targets["visual_real"],
        tactile_real=targets["tactile_real"],
        point_real=targets["point_real"],
        availability=availability,
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
    point_positions = token_field.point_positions
    tactile_positions = token_field.tactile_positions_world
    tactile_gate = token_field.tactile_contact_gate
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
    tau_pt = max(float(config.tau_pt), eps)
    distances = torch.cdist(tactile_positions, point_positions)
    nearest_point = torch.argmin(distances, dim=1)
    if nearest_point.numel() > 0:
        min_index = int(nearest_point.min().item())
        max_index = int(nearest_point.max().item())
        if min_index < 0 or max_index >= int(point_embed.shape[0]):
            raise RuntimeError(
                "PICF point-tactile alignment nearest-point index out of bounds: "
                f"valid=[0,{int(point_embed.shape[0]) - 1}] got min={min_index} max={max_index}"
            )
    if tactile_gate.numel() != tactile_embed.shape[0]:
        tactile_gate = torch.ones((tactile_embed.shape[0],), device=tactile_embed.device, dtype=tactile_embed.dtype)
    losses = []
    weights = []
    for tactile_index in range(tactile_embed.shape[0]):
        weight = tactile_gate[tactile_index]
        if float(weight.item()) <= 0.0:
            continue
        point_index = int(nearest_point[tactile_index].item())
        logits = (point_embed[point_index] @ tactile_embed.T) / tau_pt
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
            token_field.fusion_attention_mean,
        )
        total = zero_align + (cfg.lambda_pt * pt)
        return PicfAlignmentLossBreakdown(total=total, anchor_pv=zero_align, pv_weak=zero_align, focus_pv=zero_align, pt=pt, candidate_edges=0, candidate_density=0.0)

    candidate_mask = geometry.projective_candidate_mask
    projective = _sanitize_probability_tensor(geometry.projective_compatibility, eps=1e-6, interior=False)
    candidate_edges = int(candidate_mask.sum().item())
    candidate_density = float(candidate_edges / max(candidate_mask.numel(), 1))
    if candidate_edges == 0:
        zero_align = _zero_weight_sum(
            zero,
            token_field.point_align_embeddings,
            token_field.visual_align_embeddings,
            token_field.fusion_attention_mean,
        )
        total = zero_align + (cfg.lambda_pt * pt)
        return PicfAlignmentLossBreakdown(total=total, anchor_pv=zero_align, pv_weak=zero_align, focus_pv=zero_align, pt=pt, candidate_edges=0, candidate_density=candidate_density)

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

    focus_pv = zero
    fusion_attention = token_field.fusion_attention_mean
    point_count = token_field.point_tokens.shape[0]
    visual_count = token_field.visual_tokens.shape[0]
    if fusion_attention is not None and point_count > 0 and visual_count > 0:
        pv_attention = fusion_attention[point_count : point_count + visual_count, :point_count]
        focus_losses = []
        for u in range(visual_count):
            focus_weight = candidate_weight[:, u]
            if float(focus_weight.sum().item()) <= 0.0:
                continue
            numerator = torch.sum(pv_attention[u] * projective[:, u] * focus_weight) + 1e-6
            denominator = torch.sum(pv_attention[u]) + 1e-6
            focus_losses.append(-torch.log(torch.clamp(numerator / denominator, min=1e-6)))
        if focus_losses:
            focus_pv = torch.stack(focus_losses).mean()
        else:
            focus_pv = _zero_weight_sum(zero, fusion_attention)
    else:
        focus_pv = _zero_weight_sum(zero, fusion_attention)

    total = (
        (cfg.lambda_anchor_pv * anchor_pv)
        + (cfg.lambda_pv_weak * pv_weak)
        + (cfg.lambda_focus_pv * focus_pv)
        + (cfg.lambda_pt * pt)
    )
    return PicfAlignmentLossBreakdown(
        total=total,
        anchor_pv=anchor_pv,
        pv_weak=pv_weak,
        focus_pv=focus_pv,
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
) -> PicfTransitionLossBreakdown:
    cfg = config or PicfTransitionLossConfig()
    predictive = output_t.state.predictive
    pred_cache = predictive.physical_prediction_cache
    semantic_future_cache = predictive.prediction_cache
    future = extract_future_targets(core, next_observation, visual_map_override=next_visual_map_override)
    alignment = compute_alignment_loss(
        output_t.state,
        config=PicfAlignmentLossConfig(
            lambda_anchor_pv=cfg.lambda_anchor_pv,
            lambda_pv_weak=cfg.lambda_pv_weak,
            lambda_focus_pv=cfg.lambda_focus_pv,
            lambda_pt=cfg.lambda_pt,
            tau_pv=cfg.tau_pv,
            tau_pt=cfg.tau_pt,
            tau_route_p=cfg.tau_route_p,
            tau_route_v=cfg.tau_route_v,
        ),
    )

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
        tactile_real = fn.l1_loss(pred_cache.tactile_real, future.tactile_real)
    else:
        tactile_real = _zero_weight_loss(pred_cache.tactile_real, predictive.action)

    if _branch_is_usable(
        pred=pred_cache.point_real,
        target=future.point_real,
        pred_available=pred_cache.availability[3],
        target_available=future.availability[3],
    ):
        point_real = fn.binary_cross_entropy_with_logits(pred_cache.point_real, future.point_real)
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
        semantic_tactile_real = fn.l1_loss(semantic_future_cache.tactile_real, future.tactile_real)
    else:
        semantic_tactile_real = _zero_weight_loss(semantic_future_cache.tactile_real, predictive.action)

    if _branch_is_usable(
        pred=semantic_future_cache.point_real,
        target=future.point_real,
        pred_available=semantic_future_cache.availability[3],
        target_available=future.availability[3],
    ):
        semantic_point_real = fn.binary_cross_entropy_with_logits(semantic_future_cache.point_real, future.point_real)
    else:
        semantic_point_real = _zero_weight_loss(semantic_future_cache.point_real, predictive.action)

    semantic_future_aux = (
        (cfg.lambda_visual_latent * semantic_visual_latent)
        + (cfg.lambda_visual_real * semantic_visual_real)
        + (cfg.lambda_tactile_real * semantic_tactile_real)
        + (cfg.lambda_point_real * semantic_point_real)
    )

    total = (
        action_loss
        + (cfg.lambda_visual_latent * visual_latent)
        + (cfg.lambda_visual_real * visual_real)
        + (cfg.lambda_tactile_real * tactile_real)
        + (cfg.lambda_point_real * point_real)
        + (cfg.lambda_semantic_future_aux * semantic_future_aux)
        + alignment.total
    )
    return PicfTransitionLossBreakdown(
        total=total,
        action=action_loss,
        action_pos=action_pos,
        action_rot=action_rot,
        action_gripper=action_gripper,
        visual_latent=visual_latent,
        visual_real=visual_real,
        tactile_real=tactile_real,
        point_real=point_real,
        semantic_future_aux=semantic_future_aux,
        alignment=alignment.total,
        anchor_pv=alignment.anchor_pv,
        pv_weak=alignment.pv_weak,
        focus_pv=alignment.focus_pv,
        pt=alignment.pt,
        availability=future.availability,
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
