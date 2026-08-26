#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Assemble raw ADR-175 runs, seal arm reports, and run strict validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="ADR-175 matched three-arm assembler",
)

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.data.dataset_manifest import file_sha256
from picf_next.lingbot_native.adr175_assembly import (
    Adr175AssemblyContractIdentity,
    assemble_adr175_arm_reports,
)
from picf_next.lingbot_native.adr175_contract import Adr175BroadSupportContract
from picf_next.lingbot_native.adr175_validation import validate_adr175_matched_three_arm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lbot-raw-report", type=Path, required=True)
    parser.add_argument("--physical-set-raw-report", type=Path, required=True)
    parser.add_argument("--native-attention-raw-report", type=Path, required=True)
    parser.add_argument("--broad-support-contract", type=Path, required=True)
    parser.add_argument("--broad-support-contract-file-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> tuple[dict[str, object], str]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes.decode("ascii"))
    if not isinstance(payload, dict):
        raise TypeError(f"ADR-175 raw report must be an object: {path}")
    return payload, hashlib.sha256(raw_bytes).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def main() -> None:
    args = _parse_args()
    if args.output_directory.exists() or args.output_directory.is_symlink():
        raise FileExistsError(args.output_directory)
    if file_sha256(args.broad_support_contract) != (
        args.broad_support_contract_file_sha256
    ):
        raise ValueError("ADR-175 broad-support contract file SHA-256 differs")
    contract = Adr175BroadSupportContract.load(args.broad_support_contract)
    loaded = {
        "lbot": _load(args.lbot_raw_report),
        "physical-set": _load(args.physical_set_raw_report),
        "native-attention": _load(args.native_attention_raw_report),
    }
    raw_reports = {arm: value[0] for arm, value in loaded.items()}
    raw_report_file_sha256_by_arm = {arm: value[1] for arm, value in loaded.items()}
    sealed = assemble_adr175_arm_reports(
        raw_reports,
        broad_support_contract=Adr175AssemblyContractIdentity.from_contract(contract),
        raw_report_file_sha256_by_arm=raw_report_file_sha256_by_arm,
    )
    validation = validate_adr175_matched_three_arm(tuple(sealed.values())).to_dict()

    args.output_directory.mkdir(parents=True)
    for arm, report in sealed.items():
        write_bytes_durable_exclusive(
            args.output_directory / f"{arm}.arm-report.json",
            _canonical_bytes(report),
        )
    write_bytes_durable_exclusive(
        args.output_directory / "validation.json",
        _canonical_bytes(validation),
    )
    index = {
        "arm_report_artifact_sha256": {
            arm: report["artifact_sha256"] for arm, report in sealed.items()
        },
        "broad_support_contract_artifact_sha256": contract.artifact_sha256,
        "raw_report_file_sha256": raw_report_file_sha256_by_arm,
        "schema": "picf-next.adr175-arm-assembly-index.v1",
        "validation_artifact_sha256": validation["artifact_sha256"],
        "validation_status": validation["status"],
    }
    write_bytes_durable_exclusive(
        args.output_directory / "index.json",
        _canonical_bytes(index),
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
