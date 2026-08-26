#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Cold-restore the model-only LTOP G2b stage-transfer checkpoint.

This tool starts in a fresh two-rank process set and delegates the exact G2b
bootstrap and strict model-only restore to the reusable LTOP stage runtime. It
then publishes the rank-local topology, no-meta, and digest acceptance report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _repository_import_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _repository_import_text = str(_repository_import_path)
    while _repository_import_text in sys.path:
        sys.path.remove(_repository_import_text)
    sys.path.insert(0, _repository_import_text)

import picf_next as _picf_next_package

if (
    _picf_next_package.__file__ is None
    or Path(_picf_next_package.__file__).resolve().parent
    != (_REPOSITORY_ROOT / "src/picf_next").resolve()
):
    raise RuntimeError("G2b stage restore did not import picf_next from its own checkout")

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from tools.bootstrap_lingbot_vla2_native import (
    CHECKOUT_RELATIVE_PATH,
    PATCH_RELATIVE_PATH,
)
from tools.lingbot_vla2_ltop_stage_runtime import (
    LingBotVLA2LTOPStageRequest,
    open_lingbot_vla2_ltop_stage_runtime,
    prepare_lingbot_vla2_ltop_stage_transfer,
)
from tools.run_lingbot_vla2_ltop_g2_core import G2_WORLD_SIZE


RESTORE_SCHEMA = "picf-next.ltop-g2-stage-transfer-cold-restore.v1"


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value is None else Path(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=_environment_path("PICF_LINGBOT_NATIVE_SOURCE")
        or _REPOSITORY_ROOT / CHECKOUT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=_REPOSITORY_ROOT / PATCH_RELATIVE_PATH,
    )
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_environment_path("PICF_CHECKPOINT_DIR"),
        help="Pinned released LingBot checkpoint used to materialize the exact model.",
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=_environment_path("PICF_PROCESSOR_DIR"),
    )
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--g2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    )
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _stage_request(args: argparse.Namespace) -> LingBotVLA2LTOPStageRequest:
    if args.checkpoint_dir is None or args.processor_dir is None:
        raise FileNotFoundError(
            "G2b stage restore requires both released checkpoint and processor paths"
        )
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    return LingBotVLA2LTOPStageRequest(
        source_checkout=args.source_checkout,
        patch=args.patch,
        training_config=args.training_config,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        stage_checkpoint=args.stage_checkpoint,
        g2_report=args.g2_report,
        seed=args.seed,
        maximum_control_tokens=args.maximum_control_tokens,
        fsdp2_placement=args.fsdp2_placement,
    )


def main() -> None:
    args = _parse_args()
    contract = prepare_lingbot_vla2_ltop_stage_transfer(_stage_request(args))

    with open_lingbot_vla2_ltop_stage_runtime(contract) as runtime:
        dist = runtime.runtime_modules.dist
        rank_report = runtime.rank_report()
        gathered: list[dict[str, Any] | None] = [None] * G2_WORLD_SIZE
        dist.all_gather_object(gathered, rank_report)
        outcome: list[Any] = [None, None]
        if runtime.rank == 0:
            rank_reports = sorted(
                (value for value in gathered if value is not None),
                key=lambda value: value["rank"],
            )
            failures = [
                f"rank {value['rank']}: restored model digest differs from G2b save"
                for value in rank_reports
                if not value["digest_match"]
            ]
            report = {
                "schema": RESTORE_SCHEMA,
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
                "world_size": G2_WORLD_SIZE,
                "model_identity": contract.model_identity,
                "source_g2_report": str(contract.request.g2_report.absolute()),
                "source_g2_report_sha256": contract.g2_report_sha256,
                "stage_checkpoint": str(contract.request.stage_checkpoint.absolute()),
                "checkpoint_inventory": contract.checkpoint_inventory,
                "load_contract": {
                    "state_keys": ["model"],
                    "allow_partial_load": False,
                    "optimizer_loaded": False,
                    "extra_state_loaded": False,
                    "fresh_process_set": True,
                },
                "fsdp2_topology": {
                    "dp_size": G2_WORLD_SIZE,
                    "dp_replicate_size": 1,
                    "dp_shard_size": G2_WORLD_SIZE,
                    "tp_size": 1,
                    "ep_size": 1,
                    "pp_size": 1,
                    "cp_size": 1,
                    "ulysses_size": 1,
                    "dp_mode": "fsdp2",
                    "placement": contract.request.fsdp2_placement,
                },
                "rank_reports": rank_reports,
            }
            try:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_text_durable_exclusive(
                    args.output,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
                outcome = [report["status"], failures]
            except BaseException as error:
                outcome = ["WRITE_FAILED", [f"{type(error).__name__}: {error}"]]
        dist.broadcast_object_list(outcome, src=0)
        if outcome[0] != "PASS":
            raise RuntimeError(f"G2b stage-transfer cold restore rejected: {outcome[1]}")
        dist.barrier()


if __name__ == "__main__":
    main()
