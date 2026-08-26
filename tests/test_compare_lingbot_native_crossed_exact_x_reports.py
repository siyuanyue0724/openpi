from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    native_vl_fixed_x_pair_geometry_metrics,
    native_vl_fixed_x_partition_summary,
)
from tools import compare_lingbot_native_crossed_exact_x_reports as comparator


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _variant(
    *,
    source: int,
    target_bbox: tuple[int, int, int, int],
    prediction_bbox: tuple[int, int, int, int],
) -> dict[str, object]:
    instruction = "turn on the light bulb"
    grounding_request = f"Task: {instruction}\nLocate the physical object."
    target_answer = f"[{ {'label': 'light switch', 'bbox_2d': list(target_bbox)} }]".replace(
        "'", '"'
    )
    generated_text = (
        f"[{ {'label': 'light switch', 'bbox_2d': list(prediction_bbox)} }]".replace("'", '"')
        + "<|im_end|>"
    )
    return {
        "camera_name": "static",
        "generated_bbox_qwen_xyxy": list(prediction_bbox),
        "generated_bbox_schema_valid": True,
        "generated_label": "light switch",
        "generated_label_present": True,
        "generated_label_schema_valid": True,
        "generated_text": generated_text,
        "grounding_request": grounding_request,
        "grounding_request_sha256": hashlib.sha256(grounding_request.encode("utf-8")).hexdigest(),
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "normalized_label_exact_match": True,
        "source_episode_index": source,
        "source_global_index": source * 10,
        "source_rgb_sha256": _sha(f"rgb-{source}"),
        "source_state_sha256": _sha(f"state-{source}"),
        "target_answer": target_answer,
        "target_answer_sha256": hashlib.sha256(target_answer.encode("utf-8")).hexdigest(),
        "target_bbox_qwen_xyxy": list(target_bbox),
        "target_identity_key": "part/table/switch_link",
        "target_label": "light switch",
        "task_key": "turn_on_lightbulb",
    }


def _pair(
    first: dict[str, object],
    second: dict[str, object],
    *,
    index: int,
) -> dict[str, object]:
    predictions = tuple(tuple(row["generated_bbox_qwen_xyxy"]) for row in (first, second))
    targets = tuple(tuple(row["target_bbox_qwen_xyxy"]) for row in (first, second))
    metrics = native_vl_fixed_x_pair_geometry_metrics(predictions, targets)  # type: ignore[arg-type]
    variants = []
    for row, geometry in zip((first, second), metrics["variants"], strict=True):
        variants.append({**row, **geometry})
    return {
        "pair_key": _sha(f"pair-{index}"),
        "pair_metrics": {key: value for key, value in metrics.items() if key != "variants"},
        "variants": variants,
        "visual": {"file": f"pair-{index}.png", "sha256": _sha(f"visual-{index}")},
    }


def _report(*, arm: str, miss_last: bool = False, last_target_shift: int = 0) -> dict[str, object]:
    boxes = (
        (10, 10, 110, 110),
        (800, 800, 900, 900),
        (400 + last_target_shift, 400, 500 + last_target_shift, 500),
    )
    predictions = list(boxes)
    if miss_last:
        predictions[-1] = boxes[0]
    records = [
        _variant(source=index + 1, target_bbox=box, prediction_bbox=predictions[index])
        for index, box in enumerate(boxes)
    ]
    pairs = [
        _pair(copy.deepcopy(records[0]), copy.deepcopy(records[1]), index=0),
        _pair(copy.deepcopy(records[1]), copy.deepcopy(records[2]), index=1),
    ]
    summary = native_vl_fixed_x_partition_summary(pairs)
    return {
        "checkpoint_model_file_sha256": {"base.safetensors": _sha("base")},
        "crossed_exact_x_evaluation": {
            "elapsed_seconds": 2.0,
            "enabled": True,
            "evaluation_plan_artifact_sha256": _sha("plan-artifact"),
            "evaluation_plan_file_sha256": _sha("plan-file"),
            "results": pairs,
            "summary": summary,
            "unique_record_count": 3,
        },
        "dataset_manifest_sha256": _sha("dataset"),
        "eligible_item_count": 4,
        "evaluation_plan_artifact_sha256": _sha("fixed-plan-artifact"),
        "evaluation_plan_file_sha256": _sha("fixed-plan-file"),
        "excluded_items": [],
        "item_limit_per_partition": 1,
        "max_new_tokens": 64,
        "native_vl_patch_sha256": _sha("patch"),
        "partition": "heldout",
        "physical_sidecar_manifest_sha256": _sha("sidecar"),
        "picf_code_revision": "1" * 40,
        "preload_tied_parameter_name": "embed_tokens.weight",
        "processor_lattice": {"visual_lattice": 8},
        "public_vl_retention": {"enabled": False},
        "qwen_restore": {
            "load_result": {"missing_keys": [], "unexpected_keys": []},
            "model_dir": f"/{arm}",
            "model_file_sha256": {
                "config.json": _sha("config"),
                "model.safetensors": _sha(arm),
            },
            "model_revision": "2" * 40,
        },
        "runtime_python_trees": {"picf": _sha("tree")},
        "scene_evaluation": {"enabled": False},
        "schema": comparator.INPUT_SCHEMA,
        "seed": 20260802,
        "selected_item_count": 1,
        "source_commit": "3" * 40,
        "teacher_prune": {"removed": True},
        "tied_parameter_name": "embed_tokens.weight",
    }


@pytest.fixture(autouse=True)
def _small_cardinality(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comparator, "EXPECTED_PAIR_COUNT", 2)
    monkeypatch.setattr(comparator, "EXPECTED_UNIQUE_RECORD_COUNT", 3)


def test_exact_x_comparison_recomputes_and_deduplicates() -> None:
    result = comparator.compare_lingbot_native_crossed_exact_x_reports(
        _report(arm="candidate"),
        _report(arm="control"),
    )

    assert result["evidence_integrity_status"] == "PASS"
    assert result["visual_file_integrity_status"] == "NOT_REHASHED"
    assert result["scientific_gate"]["status"] == "PASS"
    assert result["unique_record"]["candidate"] == {
        "center_hit_count": 3,
        "label_exact_count": 3,
        "mean_own_target_iou": 1.0,
        "record_count": 3,
    }


def test_exact_x_comparison_reports_strict_stop_without_authorizing_long_run() -> None:
    result = comparator.compare_lingbot_native_crossed_exact_x_reports(
        _report(arm="candidate", miss_last=True),
        _report(arm="control"),
    )

    assert result["evidence_integrity_status"] == "PASS"
    assert result["scientific_gate"]["status"] == "FAIL"
    assert result["scientific_gate"]["switch_static_miss_count"] == 1
    assert result["long_run_authorized"] is False


def test_exact_x_comparison_rejects_summary_and_repeated_generation_tampering() -> None:
    candidate = _report(arm="candidate")
    control = _report(arm="control")
    candidate["crossed_exact_x_evaluation"]["summary"]["own_target_center_hit_count"] -= 1
    with pytest.raises(ContractError, match="summary was not recomputed"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(candidate, control)

    candidate = _report(arm="candidate")
    duplicate = candidate["crossed_exact_x_evaluation"]["results"][1]["variants"][0]
    duplicate["generated_text"] = '[{"label":"light switch","bbox_2d":[20,20,120,120]}]<|im_end|>'
    duplicate["generated_bbox_qwen_xyxy"] = [20, 20, 120, 120]
    with pytest.raises(ContractError, match="repeated record generation changed"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(candidate, control)


def test_exact_x_comparison_rejects_candidate_control_binding_drift() -> None:
    candidate = _report(arm="candidate")
    control = _report(arm="control", last_target_shift=10)

    with pytest.raises(ContractError, match="pair bindings differ"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(candidate, control)

    control = _report(arm="control")
    control["crossed_exact_x_evaluation"]["evaluation_plan_file_sha256"] = _sha(
        "different-exact-plan"
    )
    with pytest.raises(ContractError, match="candidate/control plans differ"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(candidate, control)


def _write_visuals(root: Path) -> None:
    for index in range(2):
        (root / f"pair-{index}.png").write_bytes(f"visual-{index}".encode("ascii"))


def test_exact_x_comparison_rehashes_visual_files(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    control_root = tmp_path / "control"
    candidate_root.mkdir()
    control_root.mkdir()
    _write_visuals(candidate_root)
    _write_visuals(control_root)

    result = comparator.compare_lingbot_native_crossed_exact_x_reports(
        _report(arm="candidate"),
        _report(arm="control"),
        candidate_visual_root=candidate_root,
        control_visual_root=control_root,
    )
    assert result["visual_file_integrity_status"] == "PASS"

    (candidate_root / "pair-1.png").write_bytes(b"tampered")
    with pytest.raises(ContractError, match="visual digest changed"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(
            _report(arm="candidate"),
            _report(arm="control"),
            candidate_visual_root=candidate_root,
            control_visual_root=control_root,
        )


def test_exact_x_comparison_rejects_partial_or_escaping_visual_roots(tmp_path: Path) -> None:
    candidate = _report(arm="candidate")
    control = _report(arm="control")
    with pytest.raises(ContractError, match="supplied for both arms"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(
            candidate,
            control,
            candidate_visual_root=tmp_path,
        )

    candidate["crossed_exact_x_evaluation"]["results"][0]["visual"]["file"] = "../pair-0.png"
    with pytest.raises(ContractError, match="escapes its report root"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(
            candidate,
            control,
            candidate_visual_root=tmp_path,
            control_visual_root=tmp_path,
        )

    with pytest.raises(ContractError, match="visual root must be an existing directory"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(
            _report(arm="candidate"),
            _report(arm="control"),
            candidate_visual_root=tmp_path / "missing",
            control_visual_root=tmp_path,
        )

    candidate_root = tmp_path / "candidate"
    control_root = tmp_path / "control"
    outside_root = tmp_path / "outside"
    candidate_root.mkdir()
    control_root.mkdir()
    outside_root.mkdir()
    _write_visuals(candidate_root)
    _write_visuals(control_root)
    (outside_root / "pair-0.png").write_bytes(b"visual-0")
    (candidate_root / "redirect").symlink_to(outside_root, target_is_directory=True)
    candidate = _report(arm="candidate")
    candidate["crossed_exact_x_evaluation"]["results"][0]["visual"]["file"] = "redirect/pair-0.png"
    with pytest.raises(ContractError, match="regular file"):
        comparator.compare_lingbot_native_crossed_exact_x_reports(
            candidate,
            _report(arm="control"),
            candidate_visual_root=candidate_root,
            control_visual_root=control_root,
        )
