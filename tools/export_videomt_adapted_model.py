#!/usr/bin/env python3
"""Export one authenticated tensor-only full VidEoMT adaptation artifact."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from picf_next.videomt_exact.checkpoint import (
    ADAPTED_MODEL_CHECKPOINT_SCHEMA,
    ADAPTED_MODEL_NUMEL,
    ADAPTED_MODEL_TENSORS,
    COMPLETE_DISTRIBUTED_CHECKPOINT_SCHEMA,
    COMPLETE_TRAINING_REPORT_SCHEMA,
    PUBLISHED_CHECKPOINT_SHA256,
    adapted_videomt_model_state,
    inspect_published_checkpoint,
    sha256_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--training-checkpoint-sha256", required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--training-report-sha256", required=True)
    parser.add_argument("--released-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    checkpoint = args.training_checkpoint.expanduser().resolve()
    report_path = args.training_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    receipt_output = args.receipt_output.expanduser().resolve()
    if output.exists() or receipt_output.exists():
        raise FileExistsError("adapted output and receipt must both be absent")
    if sha256_file(checkpoint) != args.training_checkpoint_sha256:
        raise ValueError("training checkpoint SHA-256 differs")
    if sha256_file(report_path) != args.training_report_sha256:
        raise ValueError("training report SHA-256 differs")
    released = inspect_published_checkpoint(args.released_checkpoint)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != COMPLETE_TRAINING_REPORT_SCHEMA:
        raise ValueError("complete donor training report schema differs")
    if report.get("status") != "COMPLETE":
        raise ValueError("complete donor training report did not finish")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        COMPLETE_DISTRIBUTED_CHECKPOINT_SCHEMA
    ):
        raise ValueError("complete donor training checkpoint schema differs")
    model = payload.get("model")
    if not isinstance(model, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in model.items()
    ):
        raise TypeError("complete donor model state must contain only named tensors")
    if len(model) != ADAPTED_MODEL_TENSORS or sum(
        value.numel() for value in model.values()
    ) != ADAPTED_MODEL_NUMEL:
        raise ValueError("training checkpoint does not contain the complete donor state")

    global_step = payload.get("global_step")
    split_plan_sha256 = payload.get("split_plan_sha256")
    implementation_sha256 = payload.get("implementation_sha256")
    if report.get("implementation_sha256") != implementation_sha256:
        raise ValueError("training checkpoint and report implementation differ")
    if report.get("dataset", {}).get("split_plan_sha256") != split_plan_sha256:
        raise ValueError("training checkpoint and report split differ")
    matching_checkpoint = [
        item
        for item in report.get("checkpoints", ())
        if isinstance(item, dict)
        and item.get("global_step") == global_step
        and item.get("checkpoint_sha256") == args.training_checkpoint_sha256
        and item.get("checkpoint_bytes") == checkpoint.stat().st_size
    ]
    if len(matching_checkpoint) != 1:
        raise ValueError("training report does not authenticate this checkpoint")
    assets = report.get("assets")
    if not isinstance(assets, dict) or assets.get("released_checkpoint_sha256") != (
        released.sha256
    ):
        raise ValueError("training report names another released checkpoint")
    if released.sha256 != PUBLISHED_CHECKPOINT_SHA256:
        raise ValueError("adaptation does not descend from the published VidEoMT release")

    source = {
        "checkpoint_schema": COMPLETE_DISTRIBUTED_CHECKPOINT_SCHEMA,
        "checkpoint_sha256": args.training_checkpoint_sha256,
        "report_schema": COMPLETE_TRAINING_REPORT_SCHEMA,
        "report_sha256": args.training_report_sha256,
        "global_step": global_step,
        "split_plan_sha256": split_plan_sha256,
        "implementation_sha256": implementation_sha256,
        "dataset_manifest_sha256": assets.get("dataset_manifest_sha256"),
        "physical_sidecar_manifest_sha256": assets.get(
            "physical_sidecar_manifest_sha256"
        ),
        "released_checkpoint_sha256": released.sha256,
    }
    export_payload = {
        "schema": ADAPTED_MODEL_CHECKPOINT_SCHEMA,
        "source": source,
        "model": model,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(export_payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)

    output_sha256 = sha256_file(output)
    adapted_receipt, _state = adapted_videomt_model_state(
        output,
        expected_sha256=output_sha256,
    )
    receipt = {
        "schema": "picf-next.videomt-adapted-model-export/v1",
        "artifact": asdict(adapted_receipt),
        "source_checkpoint_bytes": checkpoint.stat().st_size,
        "source_report_bytes": report_path.stat().st_size,
    }
    _atomic_json(receipt_output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
