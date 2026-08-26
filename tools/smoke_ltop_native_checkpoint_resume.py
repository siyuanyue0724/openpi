#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Exercise LingBot's native DCP wrapper across one real cold process restart."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _repository_import_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _repository_import_text = str(_repository_import_path)
    while _repository_import_text in sys.path:
        sys.path.remove(_repository_import_text)
    sys.path.insert(0, _repository_import_text)

from picf_next.artifact_io import write_text_durable_exclusive
from tools.run_lingbot_vla2_ltop_core_pilot import (
    CORE_PILOT_CHECKPOINT_EXTRA_SCHEMA,
    CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA,
    CORE_PILOT_CHECKPOINT_SCHEMA,
    CORE_PILOT_COLD_RESUME_SCHEMA,
    _all_gather_checkpoint_provenance_rank_receipts,
    _checkpoint_provenance_sha256,
    _detached_prior_boundary,
    _file_sha256,
    _validate_checkpoint_manifest,
    _validate_resume_extra,
)
from tools.run_lingbot_vla2_native_g0 import (
    _checkpoint_boundary,
    _fsync_tree,
    _rank_rng_digest,
    _validate_optimizer_state,
    _write_text_durable,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fresh", "resume"), required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _capture_cpu_rng(torch_module: Any, numpy_module: Any) -> dict[str, bytes]:
    python_state = random.getstate()
    numpy_state = numpy_module.random.get_state()
    return {
        "python_json": json.dumps(
            [python_state[0], list(python_state[1]), python_state[2]],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii"),
        "numpy_json": json.dumps(
            {
                "cached_gaussian": float(numpy_state[4]),
                "has_gauss": int(numpy_state[3]),
                "keys": numpy_state[1].tolist(),
                "name": str(numpy_state[0]),
                "position": int(numpy_state[2]),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii"),
        "torch_cpu": bytes(torch_module.get_rng_state().tolist()),
        "torch_cuda": b"cpu-only-ltop-native-checkpoint-smoke",
    }


def _restore_cpu_rng(
    state: dict[str, bytes],
    torch_module: Any,
    numpy_module: Any,
) -> None:
    python_payload = json.loads(state["python_json"].decode("ascii"))
    random.setstate(
        (
            int(python_payload[0]),
            tuple(int(value) for value in python_payload[1]),
            python_payload[2],
        )
    )
    numpy_payload = json.loads(state["numpy_json"].decode("ascii"))
    numpy_module.random.set_state(
        (
            numpy_payload["name"],
            numpy_module.asarray(numpy_payload["keys"], dtype=numpy_module.uint32),
            int(numpy_payload["position"]),
            int(numpy_payload["has_gauss"]),
            float(numpy_payload["cached_gaussian"]),
        )
    )
    torch_module.set_rng_state(
        torch_module.tensor(list(state["torch_cpu"]), dtype=torch_module.uint8)
    )


def _source_provenance(source_checkout: Path) -> dict[str, Any]:
    checkpointer = source_checkout / "lingbotvla/checkpoint/checkpointer.py"
    if checkpointer.is_symlink() or not checkpointer.is_file():
        raise FileNotFoundError("LingBot native checkpoint implementation is absent")
    return {
        "schema": CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA,
        "mode": "native-dcp-cold-process-smoke",
        "source_checkout": str(source_checkout.resolve()),
        "source_checkpointer_sha256": _file_sha256(checkpointer),
        "smoke_runner_sha256": _file_sha256(Path(__file__).resolve()),
    }


def _tensor_sha256(value: Any) -> str:
    payload = json.dumps(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _stream_source_digest(step: int) -> str:
    return hashlib.sha256(f"frozen-stream-step-{step}".encode("ascii")).hexdigest()


def _run_step(
    *,
    step: int,
    model: Any,
    optimizer: Any,
    torch_module: Any,
    numpy_module: Any,
) -> dict[str, str]:
    python_draw = random.random()
    numpy_draw = float(numpy_module.random.random())
    planned = torch_module.tensor(
        [[float(step), -0.5 * float(step), 0.25 * float(step)]],
        dtype=torch_module.float32,
    )
    inputs = planned + 0.01 * torch_module.rand_like(planned)
    loss = model(inputs).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "input_sha256": _tensor_sha256(inputs),
        "loss_hex": float(loss.detach().item()).hex(),
        "numpy_draw_hex": numpy_draw.hex(),
        "python_draw_hex": python_draw.hex(),
    }


def _distributed_preflight(
    *,
    rank: int,
    dist: Any,
    action: Any,
    name: str,
) -> None:
    result: list[str | None] = [None]
    if rank == 0:
        try:
            action()
        except BaseException as error:
            result[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise RuntimeError(f"LTOP native checkpoint smoke {name} failed: {result[0]}")


def _distributed_rank_action(
    *,
    rank: int,
    dist: Any,
    action: Any,
    name: str,
) -> None:
    result = {"rank": rank, "error": None}
    try:
        action()
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
    gathered: list[Any] = [None, None]
    dist.all_gather_object(gathered, result)
    failures = [value for value in gathered if value["error"] is not None]
    if failures:
        raise RuntimeError(f"LTOP native checkpoint smoke {name} failed: {failures}")


def main() -> None:
    args = _parse_args()
    source_checkout = args.source_checkout.resolve()
    sys.path.insert(0, str(source_checkout))

    import numpy as np
    import torch
    import torch.distributed as dist
    from lingbotvla.checkpoint import build_checkpointer

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("LTOP native checkpoint smoke requires exactly two ranks")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.run_dir / "checkpoints"
    output = checkpoint_root / "global_step_2"
    staging = checkpoint_root / ".global_step_2.incomplete"
    manifest_path = output / "ltop_core_pilot_checkpoint.json"
    receipt_path = args.run_dir / "cold_resume_receipt.json"
    reference_path = args.run_dir / "step3_reference" / f"rank_{rank}.json"
    provenance = _source_provenance(source_checkout)
    provenance_sha256 = _checkpoint_provenance_sha256(provenance)
    provenance_rank_receipts = _all_gather_checkpoint_provenance_rank_receipts(
        distributed=dist,
        rank=rank,
        checkpoint_provenance_sha256=provenance_sha256,
    )
    source_digest = _stream_source_digest(2)

    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.GELU(),
        torch.nn.Linear(4, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    random.seed(100 + rank)
    np.random.seed(100 + rank)
    torch.manual_seed(100)
    checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")

    if args.phase == "fresh":
        _distributed_preflight(
            rank=rank,
            dist=dist,
            name="fresh preflight",
            action=lambda: (
                (_ for _ in ()).throw(FileExistsError(output))
                if output.exists() or output.is_symlink()
                else None
            ),
        )
        step_reports = [
            _run_step(
                step=step,
                model=model,
                optimizer=optimizer,
                torch_module=torch,
                numpy_module=np,
            )
            for step in (1, 2)
        ]
        rng = _capture_cpu_rng(torch, np)
        lane_snapshot = _detached_prior_boundary(2)
        optimizer_summary = _validate_optimizer_state(
            optimizer,
            torch,
            expected_step=2,
        )
        boundary = _checkpoint_boundary(
            model=model,
            optimizer=optimizer,
            lane_snapshot=lane_snapshot,
            rank_rng_state=rng,
            torch_module=torch,
        )
        extra = {
            "schema": CORE_PILOT_CHECKPOINT_EXTRA_SCHEMA,
            "rank": rank,
            "world_size": 2,
            "global_step": 2,
            "next_optimizer_step": 2,
            "source_digest": source_digest,
            "provenance": provenance,
            "provenance_sha256": provenance_sha256,
            "rank_rng_state": rng,
            "lane_snapshot": lane_snapshot,
            "boundary_sha256": boundary,
            **optimizer_summary,
        }
        gathered: list[Any] = [None, None]
        dist.all_gather_object(gathered, {"rank": rank, "boundary": boundary})

        def prepare_staging() -> None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            if staging.is_symlink():
                raise ValueError("checkpoint smoke staging cannot be a symbolic link")
            if staging.exists():
                if not staging.is_dir():
                    raise ValueError("checkpoint smoke staging is not a directory")
                shutil.rmtree(staging)

        _distributed_preflight(
            rank=rank,
            dist=dist,
            name="staging preflight",
            action=prepare_staging,
        )
        checkpointer.save(
            str(staging),
            {"model": model, "optimizer": optimizer, "extra_state": extra},
            global_steps=None,
        )
        dist.barrier()

        def publish() -> None:
            payload = {
                "schema": CORE_PILOT_CHECKPOINT_SCHEMA,
                "status": "PASS",
                "global_step": 2,
                "next_optimizer_step": 2,
                "world_size": 2,
                "arm": "ltop-ec-factual",
                "provenance": provenance,
                "provenance_rank_receipts": provenance_rank_receipts,
                "provenance_sha256": provenance_sha256,
                "rank_boundaries": sorted(gathered, key=lambda value: value["rank"]),
            }
            _validate_checkpoint_manifest(
                payload,
                expected_global_step=2,
                expected_arm="ltop-ec-factual",
                expected_provenance=provenance,
            )
            _write_text_durable(
                staging / "ltop_core_pilot_checkpoint.json",
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )
            _fsync_tree(staging)
            os.replace(staging, output)
            descriptor = os.open(checkpoint_root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        _distributed_preflight(
            rank=rank,
            dist=dist,
            name="checkpoint publication",
            action=publish,
        )
        if _rank_rng_digest(_capture_cpu_rng(torch, np)) != boundary[
            "rank_rng_state_sha256"
        ]:
            raise RuntimeError("LTOP native checkpoint smoke save consumed RNG state")
        step3_report = _run_step(
            step=3,
            model=model,
            optimizer=optimizer,
            torch_module=torch,
            numpy_module=np,
        )
        step3_rng = _capture_cpu_rng(torch, np)
        step3_optimizer = _validate_optimizer_state(
            optimizer,
            torch,
            expected_step=3,
        )
        step3_boundary = _checkpoint_boundary(
            model=model,
            optimizer=optimizer,
            lane_snapshot=_detached_prior_boundary(3),
            rank_rng_state=step3_rng,
            torch_module=torch,
        )

        def publish_reference() -> None:
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            write_text_durable_exclusive(
                reference_path,
                json.dumps(
                    {
                        "checkpoint_global_step": 2,
                        "continued_global_step": 3,
                        "rank": rank,
                        "step_reports": step_reports,
                        "step3": step3_report,
                        "step3_boundary": step3_boundary,
                        "step3_optimizer_state": step3_optimizer,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )

        _distributed_rank_action(
            rank=rank,
            dist=dist,
            name="step-three reference publication",
            action=publish_reference,
        )
    else:
        manifest = _validate_checkpoint_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            expected_global_step=2,
            expected_arm="ltop-ec-factual",
            expected_provenance=provenance,
        )
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        checkpointer.load(str(output), state)
        extra = _validate_resume_extra(
            state["extra_state"],
            expected_global_step=2,
            expected_source_digest=source_digest,
            expected_provenance=provenance,
            rank=rank,
        )
        optimizer_summary = _validate_optimizer_state(
            optimizer,
            torch,
            expected_step=2,
        )
        if any(
            optimizer_summary[name] != extra[name]
            for name in ("optimizer_state_entries", "optimizer_local_moment_elements")
        ):
            raise RuntimeError("LTOP native checkpoint smoke optimizer summary differs")
        boundary = _checkpoint_boundary(
            model=model,
            optimizer=optimizer,
            lane_snapshot=extra["lane_snapshot"],
            rank_rng_state=extra["rank_rng_state"],
            torch_module=torch,
        )
        expected_boundary = next(
            item["boundary"] for item in manifest["rank_boundaries"] if item["rank"] == rank
        )
        if boundary != extra["boundary_sha256"] or boundary != expected_boundary:
            raise RuntimeError("LTOP native checkpoint smoke restored boundary differs")
        _restore_cpu_rng(extra["rank_rng_state"], torch, np)
        rng_verified = (
            _rank_rng_digest(_capture_cpu_rng(torch, np))
            == boundary["rank_rng_state_sha256"]
        )
        if not rng_verified:
            raise RuntimeError("LTOP native checkpoint smoke RNG restore differs")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        step3_report = _run_step(
            step=3,
            model=model,
            optimizer=optimizer,
            torch_module=torch,
            numpy_module=np,
        )
        if step3_report != reference["step3"]:
            raise RuntimeError("LTOP native checkpoint smoke step-three input differs")
        step3_rng = _capture_cpu_rng(torch, np)
        step3_optimizer = _validate_optimizer_state(
            optimizer,
            torch,
            expected_step=3,
        )
        step3_boundary = _checkpoint_boundary(
            model=model,
            optimizer=optimizer,
            lane_snapshot=_detached_prior_boundary(3),
            rank_rng_state=step3_rng,
            torch_module=torch,
        )
        if (
            step3_optimizer != reference["step3_optimizer_state"]
            or step3_boundary != reference["step3_boundary"]
        ):
            raise RuntimeError("LTOP native checkpoint smoke step-three continuation differs")
        gathered: list[Any] = [None, None]
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "boundary": boundary,
                "optimizer_state": optimizer_summary,
                "runtime_rng_verified": True,
                "continued_global_step": 3,
                "step3_input_sha256": step3_report["input_sha256"],
                "step3_optimizer_state": step3_optimizer,
                "step3_boundary": step3_boundary,
            },
        )

        def publish_receipt() -> None:
            payload = {
                "schema": CORE_PILOT_COLD_RESUME_SCHEMA,
                "status": "PASS",
                "global_step": 2,
                "continued_global_step": 3,
                "checkpoint_manifest_sha256": _file_sha256(manifest_path),
                "provenance_sha256": provenance_sha256,
                "rank_loads": sorted(gathered, key=lambda value: value["rank"]),
            }
            write_text_durable_exclusive(
                receipt_path,
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )

        _distributed_preflight(
            rank=rank,
            dist=dist,
            name="cold-resume receipt",
            action=publish_receipt,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
