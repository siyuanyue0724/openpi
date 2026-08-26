#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Probe released LingBot/Qwen native grounding on real CALVIN records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    _text = str(_path)
    while _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin_qwen_grounding import (
    CalvinQwenGroundingDistractor,
    CalvinQwenGroundingRecord,
    build_calvin_qwen_grounding_distractors,
    build_calvin_qwen_grounding_records,
)
from picf_next.lingbot_native.runtime_provenance import (
    adr127_runtime_python_trees_contract,
)
from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
from tools.bootstrap_lingbot_vla2_native import (
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MODEL_SOURCE,
    QWEN_PROCESSOR_REVISION,
)
from tools.bootstrap_lingbot_vla2_native_vl import (
    NATIVE_VL_PATCH_RELATIVE_PATH,
    NATIVE_VL_PATCHED_MODEL_SHA256,
    _validate_native_vl_model,
    detect_native_vl_patch_state,
    verify_native_vl_patch,
)
from tools.lingbot_vla2_runtime_helpers import (
    _merge_qwen_config,
    _resolve_training_config,
    load_lingbot_training_config,
    strip_targetless_alignment_teacher_heads,
)
from tools.probe_qwen3vl_grounding_baseline import _model_hashes

NATIVE_VL_G0_SCHEMA = "picf-next.lingbot-native-vl-grounding-g0.v4"
RESTORED_QWEN_G0_SCHEMA = "picf-next.lingbot-restored-qwen-grounding-g0.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ProbeRecord:
    record: CalvinQwenGroundingRecord
    distractors: tuple[CalvinQwenGroundingDistractor, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-count", type=int, default=16)
    parser.add_argument("--frames-per-segment", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument("--restore-qwen-dir", type=Path)
    parser.add_argument("--restore-qwen-revision")
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_args(args: argparse.Namespace) -> None:
    files = (
        args.training_config,
        args.dataset_manifest,
        args.source_checkout / MODEL_SOURCE,
        _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH,
    )
    directories = (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
        args.physical_sidecar_root,
    )
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in directories:
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    if (
        args.record_count <= 0
        or args.frames_per_segment <= 0
        or not 1 <= args.max_new_tokens <= 256
        or args.seed < 0
    ):
        raise ContractError("native VL probe numeric arguments are invalid")
    if not args.device.startswith("cuda:"):
        raise ContractError("released-weight native VL probe requires one CUDA device")
    if len(args.picf_code_revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.picf_code_revision
    ):
        raise ContractError("PICF code revision must be one lowercase Git commit")
    _validate_optional_qwen_restore(args.restore_qwen_dir, args.restore_qwen_revision)


def _validate_optional_qwen_restore(model_dir: object, revision: object) -> None:
    if model_dir is None and revision is None:
        return
    if not isinstance(model_dir, Path) or not isinstance(revision, str):
        raise ContractError("Qwen restore directory and revision must be provided together")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ContractError("Qwen restore revision must be one lowercase Git commit")


def _validate_qwen_restore_load_result(result: object) -> dict[str, list[str]]:
    raw_missing = getattr(result, "missing_keys", None)
    raw_unexpected = getattr(result, "unexpected_keys", None)
    if (
        not isinstance(raw_missing, list | tuple)
        or not isinstance(raw_unexpected, list | tuple)
        or any(not isinstance(name, str) for name in (*raw_missing, *raw_unexpected))
    ):
        raise ContractError("original Qwen restore result keys are malformed")
    missing: list[str] = sorted(raw_missing)
    unexpected: list[str] = sorted(raw_unexpected)
    if missing not in ([], ["lm_head.weight"]):
        raise ContractError(f"original Qwen restore has unexpected missing tensors: {missing}")
    if unexpected:
        raise ContractError(f"original Qwen restore has unexpected tensors: {unexpected}")
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _source_images(arrays: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.image": arrays["rgb_static"],
        "observation.images.wrist_image": arrays["rgb_gripper"],
    }


def _select_probe_records(
    index: Any, sidecar: Any, count: int, frames: int
) -> tuple[_ProbeRecord, ...]:
    selected: list[_ProbeRecord] = []
    selected_keys: set[tuple[str, str]] = set()
    for segment in index.segments:
        stop = min(int(segment.end), int(segment.start) + frames - 1)
        for global_index in range(int(segment.start), stop + 1):
            arrays = dict(
                index.validated_source_frame_arrays(
                    global_index,
                    fields=("rgb_gripper", "rgb_static"),
                )
            )
            physical = sidecar.source_frame(global_index)
            records = build_calvin_qwen_grounding_records(
                global_index=global_index,
                task_key=segment.task_key,
                instruction=segment.instruction,
                observation_images=_source_images(arrays),
                physical_frame=physical,
            )
            for record in records:
                key = (record.task_key, record.camera_name)
                if key in selected_keys:
                    continue
                distractors = build_calvin_qwen_grounding_distractors(record, physical)
                if not distractors:
                    continue
                selected.append(_ProbeRecord(record=record, distractors=distractors))
                selected_keys.add(key)
                if len(selected) == count:
                    return tuple(selected)
            if any(item.record.task_key == segment.task_key for item in selected):
                break
    if len(selected) < count:
        raise ContractError(
            f"only {len(selected)} diverse visible grounding records have distractor boxes"
        )
    return tuple(selected)


def _render_probe_record(item: _ProbeRecord, output: Path) -> str:
    from PIL import Image, ImageDraw, ImageFont

    record = item.record
    source = Image.fromarray(record.image).convert("RGB")
    header_height = 72
    panel = Image.new("RGB", (source.width, source.height + header_height), "white")
    panel.paste(source, (0, header_height))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    title = f"step={record.global_index} task={record.task_key} camera={record.camera_name}"
    detail = f"target={record.target_identity_key} bbox={record.bbox_xyxy}"
    instruction = record.instruction[:110]
    draw.text((4, 4), title, fill="black", font=font)
    draw.text((4, 24), detail, fill="black", font=font)
    draw.text((4, 44), instruction, fill="black", font=font)
    x0, y0, x1, y1 = record.bbox_xyxy
    draw.rectangle(
        (x0, y0 + header_height, x1 - 1, y1 - 1 + header_height), outline="lime", width=2
    )
    for distractor in item.distractors:
        dx0, dy0, dx1, dy1 = distractor.candidate_record.bbox_xyxy
        draw.rectangle(
            (dx0, dy0 + header_height, dx1 - 1, dy1 - 1 + header_height),
            outline="red",
            width=1,
        )
    panel.save(output, format="PNG")
    return _sha256(output)


def _record_payload(item: _ProbeRecord) -> dict[str, object]:
    record = item.record
    return {
        "bbox_xyxy": list(record.bbox_xyxy),
        "camera_name": record.camera_name,
        "global_index": record.global_index,
        "host_image_key": record.host_image_key,
        "instruction": record.instruction,
        "source_rgb_sha256": record.source_rgb_sha256,
        "target_identity_key": record.target_identity_key,
        "task_key": record.task_key,
    }


def _scored_answer_payload(
    *,
    record: CalvinQwenGroundingRecord,
    loss: float,
    supervised_token_count: int,
) -> dict[str, object]:
    if not math.isfinite(loss) or loss < 0.0 or supervised_token_count <= 0:
        raise ContractError("native VL scored answer is invalid")
    return {
        "assistant_text": record.assistant_text,
        "bbox_xyxy": list(record.bbox_xyxy),
        "mean_token_nll": loss,
        "qwen_bbox_xyxy": list(record.qwen_bbox_xyxy),
        "sequence_nll": loss * supervised_token_count,
        "supervised_token_count": supervised_token_count,
    }


def _payload_float(payload: dict[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"native VL scored answer {name!r} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"native VL scored answer {name!r} is non-finite")
    return result


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    root = _ROOT.resolve()
    overlay = root / NATIVE_VL_PATCH_RELATIVE_PATH
    patch_report = verify_native_vl_patch(root=root, checkout=args.source_checkout)
    if detect_native_vl_patch_state(args.source_checkout, overlay) != "applied":
        raise RuntimeError("native VL probe source overlay is not applied")
    if _validate_native_vl_model(args.source_checkout / MODEL_SOURCE) != (
        NATIVE_VL_PATCHED_MODEL_SHA256
    ):
        raise RuntimeError("native VL probe source digest differs")
    commit = subprocess.run(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("native VL probe source commit differs")
    runtime_python_trees = adr127_runtime_python_trees_contract(
        repo_root=root,
        revision=args.picf_code_revision,
        source_checkout=args.source_checkout,
    )
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)

    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
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
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(args.physical_sidecar_root, index)
    selected = _select_probe_records(
        index,
        sidecar,
        args.record_count,
        args.frames_per_segment,
    )

    args.output_dir.mkdir(parents=True)
    visual_dir = args.output_dir / "visuals"
    visual_dir.mkdir()
    visual_hashes = {}
    for ordinal, item in enumerate(selected):
        name = (
            f"{ordinal:03d}_step-{item.record.global_index}_"
            f"task-{item.record.task_key}_camera-{item.record.camera_name}.png"
        )
        visual_hashes[name] = _render_probe_record(item, visual_dir / name)

    sys.path.insert(0, str(args.source_checkout.resolve()))
    import torch
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )
    from transformers import AutoConfig
    from transformers.modeling_utils import load_sharded_checkpoint, no_init_weights

    from picf_next.lingbot_native.vl_cotraining import (
        build_native_vl_generation_batch,
        build_native_vl_grounding_batch,
        generate_native_vl_answer,
        parse_native_vl_grounding_answer,
        qwen_grounding_bbox_iou,
        qwen_target_center_in_bbox,
        retie_and_validate_native_qwen_lm_head,
        run_native_vl_grounding_forward,
    )

    device = torch.device(args.device)
    model_dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    training = load_lingbot_training_config(args.training_config)
    merged, _ = _resolve_training_config(
        training,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        num_steps=1,
    )
    merged["use_cache"] = False
    merged["use_compile"] = False
    merged["use_lm_head"] = True
    merged["attention_implementation"] = "eager"
    merged["vit_attn_implementation"] = "eager"
    config = LingbotVLAV2Config(**merged)
    for key, value in merged.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    # QWEN_PROCESSOR_REVISION is an exact commit and this load is local-only.
    qwen_config = AutoConfig.from_pretrained(  # nosec B615
        args.processor_dir,
        revision=QWEN_PROCESSOR_REVISION,
        local_files_only=True,
    )
    _merge_qwen_config(config, qwen_config)
    config.tokenizer_path = str(args.processor_dir.resolve())
    config.use_lm_head = True

    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    with init_empty_weights(), no_init_weights():
        policy = LingbotVlaV2Policy(config=config, eval=True).to(model_dtype)
    preload_tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
    load_model_weights(
        policy,
        str(args.checkpoint_dir.resolve()),
        str(device),
        post_training=True,
        adanorm_time=bool(config.adanorm_time),
    )
    tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
    if tied_parameter_name != preload_tied_parameter_name:
        raise ContractError("Qwen tied parameter name drifted across released-weight loading")
    qwen_restore = None
    if args.restore_qwen_dir is not None:
        restore_hashes = _model_hashes(args.restore_qwen_dir)
        qwen = policy.model.qwenvl_with_expert.qwenvl
        restore_result = load_sharded_checkpoint(
            qwen,
            args.restore_qwen_dir,
            strict=False,
            prefer_safe=True,
        )
        qwen_restore = {
            "load_result": _validate_qwen_restore_load_result(restore_result),
            "model_dir": str(args.restore_qwen_dir.resolve()),
            "model_file_sha256": restore_hashes,
            "model_revision": args.restore_qwen_revision,
        }
        restored_tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
        if restored_tied_parameter_name != tied_parameter_name:
            raise ContractError("Qwen tied parameter name drifted across foundation restoration")
    teacher_prune = strip_targetless_alignment_teacher_heads(policy)
    policy.eval()
    processor = build_processor(str(args.processor_dir.resolve()))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    results = []
    started = time.perf_counter()
    with torch.inference_mode():
        for item in selected:
            correct_batch = build_native_vl_grounding_batch(item.record, processor).to(
                device,
                pixel_dtype=model_dtype,
            )
            correct_loss = float(run_native_vl_grounding_forward(policy, correct_batch).item())
            correct_answer = _scored_answer_payload(
                record=item.record,
                loss=correct_loss,
                supervised_token_count=correct_batch.supervised_token_count,
            )
            generation_batch = build_native_vl_generation_batch(item.record, processor).to(
                device,
                pixel_dtype=model_dtype,
            )
            generated_text = generate_native_vl_answer(
                policy.model.qwenvl_with_expert.qwenvl,
                generation_batch,
                processor.tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            generated = parse_native_vl_grounding_answer(generated_text)
            generated_iou = (
                qwen_grounding_bbox_iou(
                    generated.bbox_qwen_xyxy,
                    item.record.qwen_bbox_xyxy,
                )
                if generated.bbox_qwen_xyxy is not None
                else 0.0
            )
            generated_center_hit = (
                qwen_target_center_in_bbox(
                    generated.bbox_qwen_xyxy,
                    item.record.qwen_bbox_xyxy,
                )
                if generated.bbox_qwen_xyxy is not None
                else False
            )
            distractor_values = []
            for distractor in item.distractors:
                distractor_batch = build_native_vl_grounding_batch(
                    distractor.candidate_record,
                    processor,
                ).to(device, pixel_dtype=model_dtype)
                distractor_loss = float(
                    run_native_vl_grounding_forward(policy, distractor_batch).item()
                )
                distractor_values.append(
                    {
                        "distractor_identity_key": distractor.distractor_identity_key,
                        **_scored_answer_payload(
                            record=distractor.candidate_record,
                            loss=distractor_loss,
                            supervised_token_count=distractor_batch.supervised_token_count,
                        ),
                    }
                )
            hardest_mean = min(
                distractor_values,
                key=lambda value: _payload_float(value, "mean_token_nll"),
            )
            hardest_sequence = min(
                distractor_values,
                key=lambda value: _payload_float(value, "sequence_nll"),
            )
            results.append(
                {
                    **_record_payload(item),
                    "correct_answer": correct_answer,
                    "correct_nll": correct_loss,
                    "distractors": distractor_values,
                    "generated_bbox_qwen_xyxy": (
                        list(generated.bbox_qwen_xyxy)
                        if generated.bbox_qwen_xyxy is not None
                        else None
                    ),
                    "generated_bbox_schema_valid": generated.schema_valid,
                    "generated_target_center_hit": generated_center_hit,
                    "generated_target_iou": generated_iou,
                    "generated_text": generated_text,
                    "generation_prompt_token_count": generation_batch.prompt_token_count,
                    "hardest_distractor_identity_key": hardest_mean["distractor_identity_key"],
                    "hardest_distractor_nll": _payload_float(hardest_mean, "mean_token_nll"),
                    "hardest_sequence_distractor_identity_key": hardest_sequence[
                        "distractor_identity_key"
                    ],
                    "nll_margin": _payload_float(hardest_mean, "mean_token_nll") - correct_loss,
                    "sequence_nll_margin": (
                        _payload_float(hardest_sequence, "sequence_nll")
                        - _payload_float(correct_answer, "sequence_nll")
                    ),
                }
            )
    elapsed = time.perf_counter() - started
    margins = [float(value["nll_margin"]) for value in results]
    sequence_margins = [float(value["sequence_nll_margin"]) for value in results]
    generation_ious = [float(value["generated_target_iou"]) for value in results]
    vocabulary_size = int(policy.model.qwenvl_with_expert.qwenvl.config.text_config.vocab_size)
    if (
        adr127_runtime_python_trees_contract(
            repo_root=root,
            revision=args.picf_code_revision,
            source_checkout=args.source_checkout,
        )
        != runtime_python_trees
    ):
        raise ContractError("native VL grounding probe runtime source changed during execution")
    report = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "dataset_manifest_sha256": manifest.tree_sha256,
        "elapsed_seconds": elapsed,
        "mean_nll_margin": sum(margins) / len(margins),
        "mean_sequence_nll_margin": sum(sequence_margins) / len(sequence_margins),
        "mean_generated_target_iou": sum(generation_ious) / len(generation_ious),
        "model_dtype": str(model_dtype),
        "native_vl_patch_sha256": patch_report["native_vl_patch_sha256"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "picf_code_revision": args.picf_code_revision,
        "positive_margin_count": sum(value > 0.0 for value in margins),
        "positive_sequence_margin_count": sum(value > 0.0 for value in sequence_margins),
        "generated_bbox_count": sum(
            value["generated_bbox_qwen_xyxy"] is not None for value in results
        ),
        "generated_bbox_schema_valid_count": sum(
            bool(value["generated_bbox_schema_valid"]) for value in results
        ),
        "generated_target_center_hit_count": sum(
            bool(value["generated_target_center_hit"]) for value in results
        ),
        "processor_dir": str(args.processor_dir.resolve()),
        "preload_tied_parameter_name": preload_tied_parameter_name,
        "record_count": len(results),
        "records": results,
        "qwen_restore": qwen_restore,
        "runtime_python_trees": runtime_python_trees,
        "schema": RESTORED_QWEN_G0_SCHEMA if qwen_restore is not None else NATIVE_VL_G0_SCHEMA,
        "source_commit": commit,
        "teacher_prune": teacher_prune,
        "tied_parameter_name": tied_parameter_name,
        "uniform_vocabulary_nll": math.log(vocabulary_size),
        "vocabulary_size": vocabulary_size,
        "visual_sha256": visual_hashes,
    }
    write_text_durable_exclusive(
        args.output_dir / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
