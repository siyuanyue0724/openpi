#!/usr/bin/env python3
"""Validate the frozen backbone decision ledger without network access."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    from tools.audit_upstream_sources import validate_evidence_symbols
except ModuleNotFoundError:  # Direct `python tools/...` execution.
    from audit_upstream_sources import validate_evidence_symbols

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CODE_SCOPES = {
    "announcement_only",
    "inference",
    "training_scaffold",
    "post_training",
    "full_training",
}
_WEIGHT_STATES = {"absent", "gated", "public"}
_DATA_STATES = {
    "private_summary",
    "gated_partial",
    "public_manifest",
    "open_source_derived_description",
    "user_supplied_manifests",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("references/backbone_candidates.json"),
    )
    parser.add_argument("--check-checkouts", action="store_true")
    return parser.parse_args()


def audit_candidates(path: Path, *, root: Path, check_checkouts: bool) -> dict[str, int | str]:
    data = json.loads(path.read_text())
    if data.get("schema") != "picf-next.backbone-candidates.v1":
        raise ValueError("unsupported backbone candidate schema")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 5:
        raise ValueError("at least five backbone candidates are required")

    by_name: dict[str, dict] = {}
    for candidate in candidates:
        name = candidate.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError(f"invalid or duplicate candidate name {name!r}")
        if candidate.get("code_scope") not in _CODE_SCOPES:
            raise ValueError(f"unsupported code scope for {name}")
        if candidate.get("weights") not in _WEIGHT_STATES:
            raise ValueError(f"unsupported weight state for {name}")
        if candidate.get("foundation_data") not in _DATA_STATES:
            raise ValueError(f"unsupported foundation data state for {name}")
        if not _COMMIT.fullmatch(candidate.get("commit", "")):
            raise ValueError(f"candidate {name} must pin a full commit")
        if not candidate.get("official_repo", "").startswith("https://github.com/"):
            raise ValueError(f"candidate {name} must use an official GitHub source")
        symbols = candidate.get("evidence_symbols")
        if (
            not isinstance(symbols, list)
            or not symbols
            or not all("::" in item for item in symbols)
        ):
            raise ValueError(f"candidate {name} requires exact evidence symbols")
        roles = candidate.get("roles")
        limitations = candidate.get("limitations")
        if not isinstance(roles, list) or not roles:
            raise ValueError(f"candidate {name} requires at least one role")
        if not isinstance(limitations, list) or len(limitations) < 2:
            raise ValueError(f"candidate {name} requires explicit limitations")
        for key in (
            "public_control_dataset",
            "continuous_action_path",
            "deep_action_context",
            "cross_embodiment_action_schema",
            "runtime_deployment",
        ):
            if not isinstance(candidate.get(key), bool):
                raise ValueError(f"candidate {name} requires boolean {key}")
        if check_checkouts:
            checkout = root / candidate["checkout"]
            if not (checkout / ".git").exists():
                raise ValueError(f"candidate checkout is absent: {checkout}")
            actual = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if actual != candidate["commit"]:
                raise ValueError(f"candidate {name} checkout {actual} differs from pinned commit")
            validate_evidence_symbols(checkout, symbols, source_name=name)
            submodule = candidate.get("source_submodule")
            if submodule is not None:
                _validate_source_submodule(
                    parent=checkout,
                    specification=submodule,
                    root=root,
                    source_name=name,
                )
        by_name[name] = candidate

    decision = data.get("decision", {})
    for role in (
        "causal_host",
        "deployment_host",
        "primary_unified_host",
        "architecture_reference",
        "training_infrastructure_reference",
        "capability_watch",
    ):
        if decision.get(role) not in by_name:
            raise ValueError(f"decision role {role} must name a candidate")

    causal = by_name[decision["causal_host"]]
    if causal["weights"] != "public" or causal["code_scope"] not in {
        "post_training",
        "full_training",
    }:
        raise ValueError("causal host must have public weights and complete post-training code")
    if not causal["public_control_dataset"]:
        raise ValueError("causal host must have a reproducible controlled-data route")
    causal_requirements = (
        "public_control_dataset",
        "continuous_action_path",
        "deep_action_context",
    )
    if not all(causal[key] for key in causal_requirements):
        raise ValueError("causal host does not satisfy the controlled experiment boundary")

    deployment = by_name[decision["deployment_host"]]
    if deployment["weights"] != "public" or deployment["code_scope"] not in {
        "post_training",
        "full_training",
    }:
        raise ValueError("deployment host must release weights and trainable code")
    if not all(
        deployment[key]
        for key in ("cross_embodiment_action_schema", "runtime_deployment", "deep_action_context")
    ):
        raise ValueError("deployment host lacks a required deployment property")

    primary = by_name[decision["primary_unified_host"]]
    if primary["weights"] != "public" or primary["code_scope"] not in {
        "post_training",
        "full_training",
    }:
        raise ValueError("primary unified host must release weights and trainable code")
    if not all(
        primary[key]
        for key in (
            "public_control_dataset",
            "continuous_action_path",
            "deep_action_context",
            "cross_embodiment_action_schema",
            "runtime_deployment",
        )
    ):
        raise ValueError("primary unified host lacks a required production property")
    if primary["name"] != deployment["name"]:
        raise ValueError("primary and deployment host must be identical for matched PICF arms")
    if primary["name"] != causal["name"]:
        raise ValueError("primary and causal host must be identical for matched PICF arms")

    architecture_reference = by_name[decision["architecture_reference"]]
    if architecture_reference["weights"] == "public":
        raise ValueError("a no-weight architecture reference cannot claim public weights")

    watch = by_name[decision["capability_watch"]]
    if watch["code_scope"] == "announcement_only" and watch["weights"] == "public":
        raise ValueError("an announcement-only candidate cannot claim public usable weights")
    return {
        "candidates": len(candidates),
        "causal_host": causal["name"],
        "deployment_host": deployment["name"],
        "primary_unified_host": primary["name"],
        "architecture_reference": architecture_reference["name"],
    }


def _validate_source_submodule(
    *,
    parent: Path,
    specification: dict,
    root: Path,
    source_name: str,
) -> None:
    required = {"path", "url", "commit", "checkout", "evidence_symbols"}
    if not isinstance(specification, dict) or set(specification) != required:
        raise ValueError(f"candidate {source_name} has an invalid source_submodule")
    commit = specification["commit"]
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError(f"candidate {source_name} submodule must pin a full commit")
    if not specification["url"].startswith("https://github.com/"):
        raise ValueError(f"candidate {source_name} submodule must use an official GitHub URL")
    gitlink = (
        subprocess.run(
            ["git", "-C", str(parent), "ls-tree", "HEAD", specification["path"]],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split()
    )
    if len(gitlink) != 4 or gitlink[:2] != ["160000", "commit"]:
        raise ValueError(f"candidate {source_name} source_submodule is not a gitlink")
    if gitlink[2] != commit:
        raise ValueError(
            f"candidate {source_name} parent pins submodule {gitlink[2]} instead of {commit}"
        )
    checkout = root / specification["checkout"]
    if not (checkout / ".git").exists():
        raise ValueError(f"candidate {source_name} submodule checkout is absent: {checkout}")
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != commit:
        raise ValueError(
            f"candidate {source_name} submodule checkout {actual} differs from {commit}"
        )
    validate_evidence_symbols(
        checkout,
        specification["evidence_symbols"],
        source_name=f"{source_name} source_submodule",
    )


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    result = audit_candidates(path, root=root, check_checkouts=args.check_checkouts)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
