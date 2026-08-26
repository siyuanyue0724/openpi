#!/usr/bin/env python3
"""Tensor-compare PICF entity-set primitives with one pinned SAM 3.1 checkout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from picf_next.artifact_io import write_text_durable_exclusive  # noqa: E402
from picf_next.lingbot_native.entity_set_objective import (  # noqa: E402
    SAM31_SOURCE_COMMIT,
    sam31_dice_loss,
    sam31_sigmoid_focal_loss,
)

OUTPUT_SCHEMA = "picf-next.sam31-entity-set-semantics-audit.v1"
LOSS_SOURCE_SHA256 = "72cfbc8750be577d3bcb11279e6c9d62f83db89dd446c08e1339e8e4a5944ee0"
MATCHER_SOURCE_SHA256 = "31d8c49b773a229b558a229204111c7dab09a8a9ec09a070e74e92008c2c4a83"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_function(path: Path, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(selected) != 1:
        raise RuntimeError(f"SAM source must contain exactly one {name} function")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


def _maximum_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.double() - right.double()).abs().max().item())


def audit_sam31_entity_set_semantics(source_checkout: Path) -> dict[str, object]:
    source_checkout = source_checkout.resolve()
    loss_source = source_checkout / "sam3/train/loss/loss_fns.py"
    matcher_source = source_checkout / "sam3/train/matcher.py"
    if not loss_source.is_file() or not matcher_source.is_file():
        raise FileNotFoundError("SAM 3.1 checkout lacks matcher/loss sources")
    commit = subprocess.run(
        ["git", "-C", str(source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SAM31_SOURCE_COMMIT:
        raise RuntimeError("SAM 3.1 checkout commit differs from the architecture contract")
    source_hashes = {
        "loss_fns.py": _sha256(loss_source),
        "matcher.py": _sha256(matcher_source),
    }
    if source_hashes != {
        "loss_fns.py": LOSS_SOURCE_SHA256,
        "matcher.py": MATCHER_SOURCE_SHA256,
    }:
        raise RuntimeError("SAM 3.1 source bytes differ from the audited contract")

    reference_focal = _source_function(
        loss_source,
        "sigmoid_focal_loss",
        {"torch": torch, "F": F},
    )
    reference_dice = _source_function(
        loss_source,
        "_dice_loss",
        {"torch": torch},
    )
    reference_matching = _source_function(
        matcher_source,
        "_do_matching",
        {"np": np, "linear_sum_assignment": linear_sum_assignment},
    )

    generator = torch.Generator().manual_seed(20260805)
    inputs = torch.randn(7, 19, generator=generator, dtype=torch.float64)
    targets = (torch.rand(7, 19, generator=generator, dtype=torch.float64) > 0.63).double()
    focal_reference = reference_focal(
        inputs,
        targets,
        1,
        alpha=0.25,
        gamma=2,
        reduce=False,
        triton=False,
    )
    focal_candidate = sam31_sigmoid_focal_loss(inputs, targets)
    dice_reference = reference_dice(
        inputs,
        targets,
        1,
        loss_on_multimask=False,
        reduce=False,
    )
    dice_candidate = sam31_dice_loss(inputs, targets)

    cost = np.array(
        [[0.4, 1.2, 0.1], [0.2, 0.3, 1.5], [1.1, 0.05, 0.7], [0.9, 0.8, 0.6]],
        dtype=np.float64,
    )
    reference_rows = np.asarray(reference_matching(cost), dtype=np.int64)
    row_indices, target_indices = linear_sum_assignment(cost)
    candidate_rows = row_indices[np.argsort(target_indices)]
    focal_delta = _maximum_delta(focal_candidate, focal_reference)
    dice_delta = _maximum_delta(dice_candidate, dice_reference)
    matching_equal = bool(np.array_equal(reference_rows, candidate_rows))
    passed = focal_delta == 0.0 and dice_delta == 0.0 and matching_equal
    return {
        "checks": {
            "dice_max_abs_delta": dice_delta,
            "focal_max_abs_delta": focal_delta,
            "hungarian_rows_equal": matching_equal,
            "reference_hungarian_rows": reference_rows.tolist(),
        },
        "schema": OUTPUT_SCHEMA,
        "source_commit": commit,
        "source_hashes": source_hashes,
        "status": "PASS" if passed else "FAIL",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = audit_sam31_entity_set_semantics(args.source_checkout)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        write_text_durable_exclusive(args.output, text)
    print(text, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
