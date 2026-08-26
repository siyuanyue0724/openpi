#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Replay a LingBot G0 record bank through pinned original Qwen3-VL weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    _root_text = str(_path)
    while _root_text in sys.path:
        sys.path.remove(_root_text)
    sys.path.insert(0, _root_text)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import CALVIN_CAMERA_SPECS
from picf_next.data.calvin_qwen_grounding import CalvinQwenGroundingRecord
from picf_next.lingbot_native.vl_cotraining import (
    build_native_vl_generation_batch,
    build_native_vl_grounding_batch,
    generate_native_vl_answer,
    parse_native_vl_grounding_answer,
    qwen_grounding_bbox_iou,
    qwen_target_center_in_bbox,
)

INPUT_SCHEMA = "picf-next.lingbot-native-vl-grounding-g0.v3"
OUTPUT_SCHEMA = "picf-next.qwen3vl-grounding-baseline.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for path in (args.dataset_manifest, args.input_report, args.model_dir / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.dataset_split, args.model_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    if len(args.model_revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.model_revision
    ):
        raise ContractError("Qwen baseline revision must be one lowercase Git commit")
    if not args.device.startswith("cuda:") or not 1 <= args.max_new_tokens <= 256:
        raise ContractError("Qwen baseline runtime arguments are invalid")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be nonempty text")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


def _bbox(value: object, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ContractError(f"{name} must be a four-integer JSON list")
    return value[0], value[1], value[2], value[3]


def _camera_spec(camera_name: str) -> dict[str, Any]:
    matches = tuple(item for item in CALVIN_CAMERA_SPECS if item["camera_name"] == camera_name)
    if len(matches) != 1:
        raise ContractError("Qwen baseline record has an unknown camera")
    return dict(matches[0])


def _load_probe_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid native VL G0 report: {path}") from error
    report = _mapping(value, "native VL G0 report")
    if report.get("schema") != INPUT_SCHEMA:
        raise ContractError("Qwen baseline requires a schema-v3 native VL report")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("native VL G0 report contains no records")
    return report


def _model_hashes(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("Qwen model index is invalid") from error
    index_mapping = _mapping(index, "Qwen model index")
    weight_map = index_mapping.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ContractError("Qwen model index has no weight map")
    shard_names = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or not name.endswith(".safetensors") for name in shard_names):
        raise ContractError("Qwen model index names malformed weight shards")
    required_paths = (
        model_dir / "config.json",
        index_path,
        *(model_dir / name for name in shard_names),
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    paths = tuple(sorted(path for path in model_dir.iterdir() if path.is_file()))
    return {path.name: _sha256(path) for path in paths}


def _record_from_payload(index: Any, value: object) -> CalvinQwenGroundingRecord:
    payload = _mapping(value, "native VL G0 record")
    camera_name = _text(payload.get("camera_name"), "native VL camera")
    spec = _camera_spec(camera_name)
    global_index = _integer(payload.get("global_index"), "native VL source index")
    source_field = _text(spec.get("source_rgb_field"), "native VL source field")
    arrays = index.validated_source_frame_arrays(global_index, fields=(source_field,))
    image = np.ascontiguousarray(arrays[source_field]).copy()
    image.setflags(write=False)
    record = CalvinQwenGroundingRecord(
        global_index=global_index,
        task_key=_text(payload.get("task_key"), "native VL task key"),
        instruction=_text(payload.get("instruction"), "native VL instruction"),
        target_identity_key=_text(payload.get("target_identity_key"), "native VL target identity"),
        camera_name=camera_name,
        host_image_key=_text(payload.get("host_image_key"), "native VL host image key"),
        bbox_xyxy=_bbox(payload.get("bbox_xyxy"), "native VL target bbox"),
        image=image,
        source_rgb_sha256=_text(payload.get("source_rgb_sha256"), "native VL source RGB hash"),
    )
    correct = _mapping(payload.get("correct_answer"), "native VL correct answer")
    if (
        correct.get("assistant_text") != record.assistant_text
        or _bbox(correct.get("qwen_bbox_xyxy"), "native VL normalized target bbox")
        != record.qwen_bbox_xyxy
    ):
        raise ContractError("native VL target serialization differs from its record")
    return record


def _candidate_records(
    record: CalvinQwenGroundingRecord,
    value: object,
) -> tuple[tuple[str, CalvinQwenGroundingRecord], ...]:
    payload = _mapping(value, "native VL G0 record")
    raw = payload.get("distractors")
    if not isinstance(raw, list) or not raw:
        raise ContractError("native VL G0 record has no distractors")
    output = []
    for item in raw:
        candidate = _mapping(item, "native VL distractor")
        identity = _text(candidate.get("distractor_identity_key"), "native VL distractor identity")
        replaced = replace(
            record,
            bbox_xyxy=_bbox(candidate.get("bbox_xyxy"), "native VL distractor bbox"),
        )
        if (
            candidate.get("assistant_text") != replaced.assistant_text
            or _bbox(candidate.get("qwen_bbox_xyxy"), "native VL normalized distractor bbox")
            != replaced.qwen_bbox_xyxy
        ):
            raise ContractError("native VL distractor serialization differs from its box")
        output.append((identity, replaced))
    return tuple(output)


def _score(model: Any, batch: Any) -> float:
    output = model(**batch.model_kwargs(), use_cache=False, return_dict=True)
    loss = getattr(output, "loss", None)
    if loss is None or not hasattr(loss, "item"):
        raise ContractError("original Qwen grounding forward returned no scalar loss")
    value = float(loss.item())
    if not math.isfinite(value) or value < 0.0:
        raise ContractError("original Qwen grounding loss is invalid")
    return value


def _score_payload(record: CalvinQwenGroundingRecord, loss: float, tokens: int) -> dict[str, Any]:
    return {
        "assistant_text": record.assistant_text,
        "bbox_xyxy": list(record.bbox_xyxy),
        "mean_token_nll": loss,
        "qwen_bbox_xyxy": list(record.qwen_bbox_xyxy),
        "sequence_nll": loss * tokens,
        "supervised_token_count": tokens,
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    report = _load_probe_report(args.input_report)
    model_hashes = _model_hashes(args.model_dir)

    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.name,
    )
    if report.get("dataset_manifest_sha256") != manifest.tree_sha256:
        raise ContractError("native VL G0 report belongs to another dataset tree")
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    raw_records = report["records"]
    records = tuple(_record_from_payload(index, item) for item in raw_records)
    distractors = tuple(
        _candidate_records(record, item) for record, item in zip(records, raw_records, strict=True)
    )

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    # _validate_args requires an exact commit and both loads are local-only.
    processor = AutoProcessor.from_pretrained(  # nosec B615
        args.model_dir,
        revision=args.model_revision,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(  # nosec
        args.model_dir,
        revision=args.model_revision,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    ).to(device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    results = []
    started = time.perf_counter()
    with torch.inference_mode():
        for record, candidate_records in zip(records, distractors, strict=True):
            correct_batch = build_native_vl_grounding_batch(record, processor).to(
                device, pixel_dtype=dtype
            )
            correct = _score_payload(
                record,
                _score(model, correct_batch),
                correct_batch.supervised_token_count,
            )
            candidates = []
            for identity, candidate_record in candidate_records:
                batch = build_native_vl_grounding_batch(candidate_record, processor).to(
                    device, pixel_dtype=dtype
                )
                candidates.append(
                    {
                        "distractor_identity_key": identity,
                        **_score_payload(
                            candidate_record,
                            _score(model, batch),
                            batch.supervised_token_count,
                        ),
                    }
                )
            hardest_mean = min(candidates, key=lambda item: float(item["mean_token_nll"]))
            hardest_sequence = min(candidates, key=lambda item: float(item["sequence_nll"]))
            generation_batch = build_native_vl_generation_batch(record, processor).to(
                device, pixel_dtype=dtype
            )
            generated = generate_native_vl_answer(
                model,
                generation_batch,
                processor.tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            parsed_generation = parse_native_vl_grounding_answer(generated)
            generated_iou = (
                qwen_grounding_bbox_iou(
                    parsed_generation.bbox_qwen_xyxy,
                    record.qwen_bbox_xyxy,
                )
                if parsed_generation.bbox_qwen_xyxy is not None
                else 0.0
            )
            generated_center_hit = (
                qwen_target_center_in_bbox(
                    parsed_generation.bbox_qwen_xyxy,
                    record.qwen_bbox_xyxy,
                )
                if parsed_generation.bbox_qwen_xyxy is not None
                else False
            )
            results.append(
                {
                    "camera_name": record.camera_name,
                    "correct_answer": correct,
                    "distractors": candidates,
                    "generated_bbox_qwen_xyxy": (
                        list(parsed_generation.bbox_qwen_xyxy)
                        if parsed_generation.bbox_qwen_xyxy is not None
                        else None
                    ),
                    "generated_bbox_schema_valid": parsed_generation.schema_valid,
                    "generated_target_center_hit": generated_center_hit,
                    "generated_target_iou": generated_iou,
                    "generated_text": generated,
                    "global_index": record.global_index,
                    "instruction": record.instruction,
                    "mean_token_nll_margin": float(hardest_mean["mean_token_nll"])
                    - float(correct["mean_token_nll"]),
                    "sequence_nll_margin": float(hardest_sequence["sequence_nll"])
                    - float(correct["sequence_nll"]),
                    "target_identity_key": record.target_identity_key,
                    "task_key": record.task_key,
                }
            )
    elapsed = time.perf_counter() - started
    margins = [float(item["mean_token_nll_margin"]) for item in results]
    sequence_margins = [float(item["sequence_nll_margin"]) for item in results]
    generation_ious = [float(item["generated_target_iou"]) for item in results]
    output = {
        "dataset_manifest_sha256": manifest.tree_sha256,
        "elapsed_seconds": elapsed,
        "mean_sequence_nll_margin": sum(sequence_margins) / len(sequence_margins),
        "mean_token_nll_margin": sum(margins) / len(margins),
        "mean_generated_target_iou": sum(generation_ious) / len(generation_ious),
        "model_dir": str(args.model_dir.resolve()),
        "model_file_sha256": model_hashes,
        "model_revision": args.model_revision,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "positive_sequence_margin_count": sum(value > 0.0 for value in sequence_margins),
        "positive_token_margin_count": sum(value > 0.0 for value in margins),
        "generated_bbox_count": sum(
            item["generated_bbox_qwen_xyxy"] is not None for item in results
        ),
        "generated_bbox_schema_valid_count": sum(
            bool(item["generated_bbox_schema_valid"]) for item in results
        ),
        "generated_target_center_hit_count": sum(
            bool(item["generated_target_center_hit"]) for item in results
        ),
        "record_count": len(results),
        "records": results,
        "schema": OUTPUT_SCHEMA,
        "source_probe_report_sha256": _sha256(args.input_report),
    }
    args.output_dir.mkdir(parents=True)
    write_text_durable_exclusive(
        args.output_dir / "report.json",
        json.dumps(output, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
