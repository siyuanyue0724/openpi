#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(x.sum(dim=-1, keepdim=True), min=eps)


def _file_competition_reference(
    support: torch.Tensor,
    dustbin: torch.Tensor,
    *,
    x: torch.Tensor,
    roles: torch.Tensor,
    alpha: torch.Tensor,
    support_overlap_threshold: float = 0.80,
    geometry_sigma_m: float = 0.04,
    geometry_threshold: float = 0.70,
    min_per_role: int = 1,
) -> dict[str, torch.Tensor]:
    """Pure reference math for posterior file/no-object competition.

    This mirrors the intended assignment model, not the module implementation:
    same-role persistent files are capacity. Only distinct support/geometry
    owners should update from current observation evidence; duplicate rows are
    moved into the observation dustbin and total mass is conserved.
    """

    eps = 1e-8
    support = torch.clamp(torch.nan_to_num(support.float()), min=0.0)
    dustbin = torch.clamp(torch.nan_to_num(dustbin.float()), min=0.0)
    k = support.shape[0]
    mass = support.sum(dim=1)
    cond = _normalize_rows(support, eps=eps)
    norm = torch.sqrt(torch.clamp((cond * cond).sum(dim=1), min=eps))
    overlap = (cond @ cond.T) / torch.clamp(norm[:, None] * norm[None, :], min=eps)
    overlap = torch.clamp(torch.nan_to_num(overlap), 0.0, 1.0)
    duplicate = torch.where(overlap >= support_overlap_threshold, overlap, torch.zeros_like(overlap))
    dist2 = torch.cdist(x.float()[:, :3], x.float()[:, :3]).pow(2)
    geom = torch.exp(-dist2 / (2.0 * geometry_sigma_m * geometry_sigma_m))
    duplicate = torch.maximum(duplicate, torch.where(geom >= geometry_threshold, geom, torch.zeros_like(geom)))
    duplicate = duplicate - torch.diag(torch.diag(duplicate))
    alpha = torch.clamp(alpha.float(), 0.0, 1.0)
    if cond.shape[1] > 1:
        top2 = torch.topk(cond, k=2, dim=1).values
        margin = torch.clamp(top2[:, 0] - top2[:, 1], 0.0, 1.0)
    else:
        margin = torch.ones_like(mass)
    score = mass * (0.25 + 0.75 * alpha) * (0.25 + 0.75 * margin)
    active = torch.zeros((k,), dtype=torch.float32)
    for role in torch.unique(roles).tolist():
        idxs = torch.nonzero(roles == int(role), as_tuple=False).squeeze(-1)
        kept: list[int] = []
        order = idxs[torch.argsort(score[idxs], descending=True)]
        for idx_t in order.tolist():
            idx = int(idx_t)
            if kept:
                kept_t = torch.as_tensor(kept, dtype=torch.long)
                if float(duplicate[idx, kept_t].max().item()) > 0.0 and len(kept) >= min_per_role:
                    continue
            kept.append(idx)
        if not kept and idxs.numel() > 0 and min_per_role > 0:
            kept.append(int(order[0].item()))
        active[torch.as_tensor(kept, dtype=torch.long)] = 1.0
    demoted = support * (1.0 - active[:, None])
    return {
        "support": support * active[:, None],
        "dustbin": dustbin + demoted.sum(dim=0),
        "active": active,
        "demoted_mass": demoted.sum(dim=1),
        "duplicate": duplicate,
    }


def run_checks() -> list[Check]:
    checks: list[Check] = []
    dustbin = torch.tensor([0.2, 0.2, 0.2])
    roles = torch.tensor([1, 1, 1])
    alpha = torch.ones(3)

    duplicate_support = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    x_far = torch.tensor([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.40, 0.0, 0.0]])
    out = _file_competition_reference(duplicate_support, dustbin, x=x_far, roles=roles, alpha=alpha)
    checks.append(
        Check(
            "same_support_duplicate_demotes_one_file",
            bool(out["active"].tolist() == [1.0, 0.0, 1.0] and out["demoted_mass"][1] > 0.99),
            "Two same-role files with identical support should not both update; the duplicate mass moves to dustbin.",
        )
    )
    before = duplicate_support.sum() + dustbin.sum()
    after = out["support"].sum() + out["dustbin"].sum()
    checks.append(
        Check(
            "measurement_mass_is_conserved",
            bool(torch.allclose(before, after, atol=1e-6)),
            "Demotion must be a no-object reassignment, not mass deletion.",
        )
    )

    distinct_support = torch.eye(3)
    out = _file_competition_reference(distinct_support, dustbin, x=x_far, roles=roles, alpha=alpha)
    checks.append(
        Check(
            "distinct_support_keeps_capacity",
            bool(out["active"].tolist() == [1.0, 1.0, 1.0]),
            "Same-role files with distinct support owners should remain available.",
        )
    )

    x_close = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.30, 0.0, 0.0]])
    out = _file_competition_reference(distinct_support, dustbin, x=x_close, roles=roles, alpha=alpha)
    checks.append(
        Check(
            "geometry_duplicate_demotes_even_with_distinct_support",
            bool(out["active"].tolist() == [1.0, 0.0, 1.0]),
            "Two files with nearly identical physical centers should not both claim object identity.",
        )
    )
    return checks


def main() -> None:
    checks = run_checks()
    payload = {
        "ok": all(check.ok for check in checks),
        "checks": [check.__dict__ for check in checks],
    }
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
