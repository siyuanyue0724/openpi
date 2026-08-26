#!/usr/bin/env python3
"""Compose independently produced action-adoption and DCP acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.adr150_lbot_validation import (
    validate_full_modal_action_adoption,
)
from picf_next.lingbot_native.full_modal_adoption import (
    compose_full_modal_action_adoption,
)

_MAXIMUM_INPUT_BYTES = 64 * 1024 * 1024


def _direct_json(path: Path, name: str) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be one direct regular file")
    if resolved.stat().st_size > _MAXIMUM_INPUT_BYTES:
        raise ValueError(f"{name} exceeds the acceptance evidence size limit")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must contain one string-keyed JSON object")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--dcp-uninterrupted", type=Path, required=True)
    parser.add_argument("--dcp-restored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be one absent direct path")
    return args


def main() -> None:
    args = _parse_args()
    composed = compose_full_modal_action_adoption(
        core=_direct_json(args.core, "action-adoption core"),
        uninterrupted=_direct_json(args.dcp_uninterrupted, "uninterrupted DCP evidence"),
        restored=_direct_json(args.dcp_restored, "restored DCP evidence"),
    )
    canonical = validate_full_modal_action_adoption(composed)
    encoded = (json.dumps(canonical, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_text_durable_exclusive(args.output, encoded.decode())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
