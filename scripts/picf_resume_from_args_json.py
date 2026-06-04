"""Resume a PICF training run from a saved args.json with explicit overrides.

This runner is intentionally small: it preserves the exact historical training
contract recorded by ``scripts/picf_core_train.py`` and only applies overrides
listed on the command line.  It avoids hand-copying hundreds of flags when a
diagnostic must differ by one or two controlled variables.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


def _parse_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _apply_override(payload: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"Override must be key=value, got {assignment!r}.")
    key, value = assignment.split("=", 1)
    key = key.strip().replace("-", "_")
    if not key:
        raise ValueError(f"Override key is empty in {assignment!r}.")
    payload[key] = _parse_value(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PICF training from a saved args.json plus exact overrides.")
    parser.add_argument("--args-json", required=True, type=Path)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one saved training argument. Repeatable. Hyphens in KEY are normalized to underscores.",
    )
    parser.add_argument(
        "--print-effective-args",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the patched argument dictionary before training starts.",
    )
    cli_args = parser.parse_args()

    # Import lazily so --help for this runner does not import optional training dependencies.
    from scripts.picf_core_train import (
        _init_logging,
        _normalize_train_args,
        _validate_backbone_args,
        _validate_train_args,
        train,
    )

    payload = json.loads(cli_args.args_json.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected {cli_args.args_json} to contain a JSON object.")

    for assignment in cli_args.set:
        _apply_override(payload, assignment)

    # Defaults for fields added after older checkpoint args were written.
    payload.setdefault("semantic_action_context_readout_aux_weight", 0.0)
    payload.setdefault("semantic_action_context_readout_aux_loss", "smooth_l1")
    payload.setdefault("semantic_action_context_readout_aux_huber_delta", 1.0)

    args = argparse.Namespace(**payload)
    _normalize_train_args(args)
    _validate_train_args(args)
    _validate_backbone_args(args)

    if cli_args.print_effective_args:
        print(json.dumps(vars(args), ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        sys.stderr.flush()

    _init_logging()
    train(args)


if __name__ == "__main__":
    main()
