#!/usr/bin/env python3
"""Audit the action-visible semantics of active/context/reserve AQR anchors.

The posterior file competition path creates inactive/no-object reserve files.
Some inactive files are duplicate/no-object reserve capacity, while other
inactive files can still be real scene context. The graph prefix therefore uses
a tri-state downstream weight: active object = 1.0, context object = small
weight, reserve/dustbin = 0.0. This audit checks that dataflow and a small
mathematical example.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def _contains(source: str, *needles: str) -> bool:
    return all(needle in source for needle in needles)


def _close_matrix(left: list[list[float]], right: list[list[float]], *, tol: float = 1.0e-9) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right, strict=True):
        if len(left_row) != len(right_row):
            return False
        for left_value, right_value in zip(left_row, right_row, strict=True):
            if abs(float(left_value) - float(right_value)) > tol:
                return False
    return True


def _math_tristate_gate_routes_active_context_reserve() -> Check:
    graph_tokens = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    downstream_weight = [1.0, 0.15, 0.0]
    gated = [[value * downstream_weight[row] for value in token] for row, token in enumerate(graph_tokens)]
    expected = [[1.0, 2.0, 3.0], [0.6, 0.75, 0.9], [0.0, 0.0, 0.0]]
    return Check(
        "math_tristate_gate_routes_active_context_reserve",
        _close_matrix(gated, expected),
        "Graph anchors must route as active=full, context=low-weight, reserve=zero action-prefix evidence.",
    )


def _math_background_context_survives_reserve_gate() -> Check:
    posterior = [[1.0, 0.0], [9.0, 9.0]]
    posterior_active = [1.0, 0.0]
    graph = [[2.0, 0.0], [0.5, 0.5], [8.0, 8.0]]
    graph_downstream_weight = [1.0, 0.15, 0.0]
    global_context = [0.25, 0.75]
    task_context = [2.0, 3.0]
    gated_posterior = [
        [value * posterior_active[row] for value in token] for row, token in enumerate(posterior)
    ]
    weighted_graph = [[value * graph_downstream_weight[row] for value in token] for row, token in enumerate(graph)]
    prefix = gated_posterior + weighted_graph + [global_context, task_context]
    ok = (
        prefix[1] == [0.0, 0.0]
        and prefix[2] == [2.0, 0.0]
        and prefix[3] == [0.075, 0.075]
        and prefix[4] == [0.0, 0.0]
        and prefix[5] == global_context
        and prefix[6] == task_context
    )
    return Check(
        "math_background_context_survives_reserve_gate",
        bool(ok),
        "Reserve rows are zeroed, context rows survive with low weight, and global/task context remains.",
    )


def run_checks() -> list[Check]:
    pipeline = _read("src/openpi/picf/core/pipeline.py")
    trainer = _read("scripts/picf_core_train.py")
    docs = _read("docs/PICF_AQR_OWM_PROFESSOR_GRADE_BINDING_FOLLOWTHROUGH_20260516_TEMP.md")
    return [
        Check(
            "pipeline_gates_inactive_posterior_tokens",
            _contains(
                pipeline,
                "posterior_gate = self._posterior_file_active_gate",
                "posterior_gate * _add_role_embedding(control_posterior_tokens",
                "posterior_self_bias[:, ~active_key] = -1.0e4",
            ),
            "Inactive posterior object files must not become control prefix or posterior self-attention keys.",
        ),
        Check(
            "pipeline_routes_graph_downstream_weight",
            _contains(
                pipeline,
                "def _aqr_downstream_slot_weights",
                "anchor_downstream_weight",
                "graph_weight = torch.clamp",
                "graph_tokens = graph_tokens * graph_weight",
                "Active object anchors are full action evidence, context",
            ),
            "AQR graph anchors must use active/context/reserve downstream weights in the action prefix.",
        ),
        Check(
            "overlay_exports_dual_views",
            _contains(
                trainer,
                "variant_name=\"with_gray\"",
                "variant_name=\"active_only\"",
                "include_inactive=True",
                "include_inactive=False",
                "with_gray includes context/reserve graph or posterior",
                "active_only hides those non-full-weight files",
            ),
            "Overlay diagnostics must separate reserve-capacity audit from active object-binding audit.",
        ),
        Check(
            "docs_record_background_semantics",
            _contains(
                docs,
                "Background Tokens And Gray Reserve Files",
                "active orange/green/etc. files",
                "context files",
                "gray reserve files",
                "background tokens",
            ),
            "Docs must state that background/context is retained without forcing it into active object files.",
        ),
        _math_tristate_gate_routes_active_context_reserve(),
        _math_background_context_survives_reserve_gate(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    checks = run_checks()
    result = {"ok": all(check.ok for check in checks), "checks": [check.__dict__ for check in checks]}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_fail and not result["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
