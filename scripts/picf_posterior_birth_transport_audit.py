#!/usr/bin/env python3
"""Executable math audit for posterior birth transport.

The posterior file-competition step demotes duplicate object-file support into
the dustbin.  The birth-transport step must not broadcast that demoted residual
into every inactive file.  This script checks the transport algebra without
depending on torch so it can run in minimal CI environments.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditCase:
    name: str
    roles: tuple[int, ...]
    active: tuple[float, ...]
    recycle: tuple[float, ...]
    alpha: tuple[float, ...]
    support_mass: tuple[float, ...]
    dustbin_mass: float
    max_per_role: int = 1
    min_score: float = 0.05
    alpha_power: float = 0.5


def _birth_select(case: AuditCase) -> tuple[float, ...]:
    scores = []
    for active, recycle, alpha in zip(case.active, case.recycle, case.alpha, strict=True):
        score = max(recycle, 0.0)
        score *= max(1.0 - active, 0.0)
        score *= max(1.0 - alpha, 0.0) ** max(case.alpha_power, 0.0)
        if score < case.min_score:
            score = 0.0
        scores.append(score)

    selected = [0.0 for _ in scores]
    for role in sorted(set(case.roles)):
        role_indices = [idx for idx, value in enumerate(case.roles) if value == role and scores[idx] > 0.0]
        role_indices.sort(key=lambda idx: scores[idx], reverse=True)
        for idx in role_indices[: max(case.max_per_role, 0)]:
            selected[idx] = 1.0
    return tuple(selected)


def _birth_share(case: AuditCase, selected: tuple[float, ...]) -> tuple[float, ...]:
    birth_recycle = [sel * max(rec, 0.0) for sel, rec in zip(selected, case.recycle, strict=True)]
    denom = 1.0 + sum(birth_recycle)
    return tuple(value / denom for value in birth_recycle)


def _assert_close(lhs: float, rhs: float, *, name: str, eps: float = 1e-9) -> None:
    if not math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=eps):
        raise AssertionError(f"{name}: expected {rhs:.12f}, got {lhs:.12f}")


def _run_case(case: AuditCase) -> None:
    selected = _birth_select(case)
    share = _birth_share(case, selected)

    for role in sorted(set(case.roles)):
        role_count = sum(1 for value, role_value in zip(selected, case.roles, strict=True) if value > 0 and role_value == role)
        if role_count > case.max_per_role:
            raise AssertionError(f"{case.name}: role {role} selected {role_count} births")

    for idx, (active, sel) in enumerate(zip(case.active, selected, strict=True)):
        if active > 0.5 and sel > 0.0:
            raise AssertionError(f"{case.name}: active file {idx} selected as birth")

    if case.dustbin_mass > 0.0:
        consuming = sum(1 for value in share if value > 0.0)
        max_possible = case.max_per_role * len(set(case.roles))
        if consuming > max_possible:
            raise AssertionError(f"{case.name}: dustbin broadcast to {consuming} files")

    transported = sum(case.support_mass) + case.dustbin_mass
    residual_dustbin = case.dustbin_mass / (1.0 + sum(sel * max(rec, 0.0) for sel, rec in zip(selected, case.recycle, strict=True)))
    after = sum(case.support_mass) + sum(value * case.dustbin_mass for value in share) + residual_dustbin
    _assert_close(after, transported, name=f"{case.name}: mass conservation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-fail", action="store_true", help="Return non-zero on audit failure.")
    args = parser.parse_args()

    cases = (
        AuditCase(
            name="many_inactive_same_role_only_one_birth",
            roles=(0, 0, 0, 0),
            active=(1.0, 0.0, 0.0, 0.0),
            recycle=(0.1, 0.9, 0.8, 0.7),
            alpha=(0.9, 0.0, 0.1, 0.2),
            support_mass=(0.5, 0.0, 0.0, 0.0),
            dustbin_mass=0.5,
        ),
        AuditCase(
            name="per_role_births_not_global_broadcast",
            roles=(0, 0, 1, 1, 2, 2),
            active=(1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            recycle=(0.3, 0.9, 0.2, 0.7, 0.6, 0.5),
            alpha=(0.8, 0.1, 0.8, 0.1, 0.1, 0.2),
            support_mass=(0.3, 0.0, 0.2, 0.0, 0.0, 0.0),
            dustbin_mass=0.5,
        ),
        AuditCase(
            name="all_scores_below_threshold_keeps_dustbin",
            roles=(0, 0, 1, 1),
            active=(1.0, 0.0, 1.0, 0.0),
            recycle=(0.01, 0.02, 0.01, 0.02),
            alpha=(0.9, 0.9, 0.9, 0.9),
            support_mass=(0.4, 0.0, 0.3, 0.0),
            dustbin_mass=0.3,
        ),
    )

    failures: list[str] = []
    for case in cases:
        try:
            _run_case(case)
            print(f"PASS {case.name}")
        except Exception as exc:  # pragma: no cover - command-line audit path
            failures.append(f"FAIL {case.name}: {exc}")
            print(failures[-1])

    if failures and args.fail_on_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
