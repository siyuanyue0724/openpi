from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.probe_lingbot_native_vl_fixed_x as fixed_x_module
from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenGroundingRecord,
)
from picf_next.lingbot_native.crossed_bounded_plan import CrossedBoundedRecord
from picf_next.lingbot_native.crossed_evaluation import CrossedEvaluationPair
from picf_next.lingbot_native.vl_cotraining import (
    parse_native_vl_grounding_answer,
    parse_native_vl_scene_grounding_answer,
)
from tools.probe_lingbot_native_vl_fixed_x import (
    OUTPUT_SCHEMA,
    SCENE_AUDIT_SCHEMA,
    SCENE_MAX_NEW_TOKENS_LIMIT,
    _calvin_semantic_evidence,
    _load_scene_audit_report,
    _materialize_crossed_x_records,
    _materialize_scene_bank,
    _normalize_generated_answer,
    _pair_geometry_metrics,
    _public_retention_summary,
    _render_crossed_x_pair,
    _render_pair,
    _render_public_referring_prediction,
    _render_scene_pair,
    _scene_audit_canonical_bytes,
    _scene_generation_budget_contract,
    _select_pairs,
    _semantic_partition_summary,
    _validate_scene_bank_task_keys,
)

_TOOL = Path(__file__).resolve().parents[1] / "tools/probe_lingbot_native_vl_fixed_x.py"


def test_fixed_x_semantic_evidence_uses_a_new_schema() -> None:
    assert OUTPUT_SCHEMA == "picf-next.lingbot-native-vl-fixed-x-g0.v8"


def test_scene_generation_budget_covers_longest_legal_answer() -> None:
    assert _scene_generation_budget_contract((79, 271), max_new_tokens=512) == {
        "configured_max_new_tokens": 512,
        "headroom_tokens": 241,
        "maximum_target_supervised_tokens": 271,
        "minimum_target_supervised_tokens": 79,
        "target_record_count": 2,
    }


def test_scene_generation_budget_rejects_truncation_and_invalid_values() -> None:
    with pytest.raises(ContractError, match="cannot emit the longest legal target"):
        _scene_generation_budget_contract((79, 271), max_new_tokens=256)
    with pytest.raises(ContractError, match="token counts are invalid"):
        _scene_generation_budget_contract((), max_new_tokens=512)
    with pytest.raises(ContractError, match="generation budget is invalid"):
        _scene_generation_budget_contract((79, 271), max_new_tokens=SCENE_MAX_NEW_TOKENS_LIMIT + 1)


def _write_scene_audit(path: Path) -> str:
    content = {
        "arm_steps": [
            {
                "global_index": index,
                "group_index": index,
                "source_rgb_sha256": f"{index + 1:064x}",
            }
            for index in range(64)
        ],
        "dataset_tree_sha256": "d" * 64,
        "physical_sidecar_manifest_sha256": "s" * 64,
        "picf_code_revision": "c" * 40,
        "schema": SCENE_AUDIT_SCHEMA,
        "source_disjoint_scene_bank": [
            {"bank_index": index, "group_index": index + 64} for index in range(32)
        ],
        "status": "PASS",
        "visual_lattice": 8,
    }
    artifact_sha256 = hashlib.sha256(_scene_audit_canonical_bytes(content)).hexdigest()
    payload = _scene_audit_canonical_bytes({**content, "artifact_sha256": artifact_sha256}) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_scene_audit_loader_is_file_and_content_fail_closed(tmp_path) -> None:
    report = tmp_path / "report.json"
    file_sha256 = _write_scene_audit(report)
    loaded = _load_scene_audit_report(report, expected_file_sha256=file_sha256)
    assert loaded["schema"] == SCENE_AUDIT_SCHEMA
    assert len(loaded["source_disjoint_scene_bank"]) == 32
    link = tmp_path / "report-link.json"
    link.symlink_to(report)
    with pytest.raises(ContractError, match="file binding changed"):
        _load_scene_audit_report(link, expected_file_sha256=file_sha256)

    tampered = json.loads(report.read_text())
    tampered["visual_lattice"] = 7
    report.write_text(json.dumps(tampered))
    tampered_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
    with pytest.raises(ContractError, match="artifact digest changed"):
        _load_scene_audit_report(report, expected_file_sha256=tampered_sha256)


def test_scene_bank_materialization_rejects_arm_source_reuse() -> None:
    audit = {
        "arm_steps": [
            {
                "global_index": index,
                "group_index": index,
                "source_rgb_sha256": f"{index + 1:064x}",
            }
            for index in range(64)
        ],
        "dataset_tree_sha256": "d" * 64,
        "physical_sidecar_manifest_sha256": "s" * 64,
        "picf_code_revision": "c" * 40,
        "source_disjoint_scene_bank": [
            {
                "bank_index": index,
                "camera_name": "static",
                "global_index": index,
                "group_index": index if index == 0 else index + 64,
                "object_identity_keys": [],
                "source_rgb_sha256": f"{index + 100:064x}",
                "task_keys": ["task-a", "task-b"],
            }
            for index in range(32)
        ],
        "visual_lattice": 8,
    }
    sidecar = SimpleNamespace(manifest_sha256="s" * 64)
    with pytest.raises(ContractError, match="not source-disjoint"):
        _materialize_scene_bank(
            audit=audit,
            index=SimpleNamespace(),
            sidecar=sidecar,
            dataset_tree_sha256="d" * 64,
            picf_code_revision="c" * 40,
        )


def test_scene_bank_task_keys_preserve_validated_variable_width() -> None:
    assert _validate_scene_bank_task_keys(["task-a", "task-b", "task-c", "task-d"]) == (
        "task-a",
        "task-b",
        "task-c",
        "task-d",
    )


def test_scene_bank_materialization_accepts_real_four_task_provenance(monkeypatch) -> None:
    identity_key = CALVIN_QWEN_SCENE_IDENTITY_ORDER[0]
    canonical_text = "canonical"
    reverse_text = "reverse"

    def build_record(*, global_index, image, category_identity_order, **_kwargs):
        assistant_text = (
            canonical_text
            if tuple(category_identity_order) == CALVIN_QWEN_SCENE_IDENTITY_ORDER
            else reverse_text
        )
        item = SimpleNamespace(
            identity_key=identity_key,
            bbox_xyxy=(1, 2, 3, 4),
            visible_owner_pixels=4,
            projected_target_mass=1.0,
            positive_visual_token_count=1,
        )
        return SimpleNamespace(
            assistant_text=assistant_text,
            image=image,
            objects=(item,),
            source_rgb_sha256=f"{global_index + 1000:064x}",
            subpatch_objects=(),
        )

    monkeypatch.setattr(
        fixed_x_module,
        "build_calvin_qwen_scene_grounding_record",
        build_record,
    )
    audit = {
        "arm_steps": [
            {
                "global_index": index,
                "group_index": index,
                "source_rgb_sha256": f"{index + 1:064x}",
            }
            for index in range(64)
        ],
        "dataset_tree_sha256": "d" * 64,
        "physical_sidecar_manifest_sha256": "s" * 64,
        "picf_code_revision": "c" * 40,
        "source_disjoint_scene_bank": [
            {
                "bank_index": index,
                "camera_name": "static",
                "canonical_answer_sha256": hashlib.sha256(
                    canonical_text.encode("utf-8")
                ).hexdigest(),
                "global_index": index + 100,
                "group_index": index + 100,
                "object_identity_keys": [identity_key],
                "reverse_answer_sha256": hashlib.sha256(reverse_text.encode("utf-8")).hexdigest(),
                "source_rgb_sha256": f"{index + 1100:064x}",
                "task_keys": ["task-a", "task-b", "task-c", "task-d"],
            }
            for index in range(32)
        ],
        "visual_lattice": 8,
    }
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    index = SimpleNamespace(
        validated_source_frame_arrays=lambda *_args, **_kwargs: (
            ("rgb_gripper", image),
            ("rgb_static", image),
        )
    )
    sidecar = SimpleNamespace(
        manifest_sha256="s" * 64,
        source_frame=lambda _global_index: SimpleNamespace(),
    )

    pairs = _materialize_scene_bank(
        audit=audit,
        index=index,
        sidecar=sidecar,
        dataset_tree_sha256="d" * 64,
        picf_code_revision="c" * 40,
    )

    assert len(pairs) == 32
    assert all(pair.task_keys == ("task-a", "task-b", "task-c", "task-d") for pair in pairs)


@pytest.mark.parametrize(
    "task_keys",
    ([], [""], ["task-a", "task-a"], ["task-a", 1]),
)
def test_scene_bank_task_keys_reject_malformed_provenance(task_keys: list[object]) -> None:
    with pytest.raises(ContractError, match="task keys are invalid"):
        _validate_scene_bank_task_keys(task_keys)


def _pair(partition: str, ordinal: int) -> SimpleNamespace:
    return SimpleNamespace(item=SimpleNamespace(partition=partition, ordinal=ordinal))


def _calvin_record() -> CalvinQwenGroundingRecord:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image.setflags(write=False)
    return CalvinQwenGroundingRecord(
        global_index=3,
        task_key="turn_on_led",
        instruction="press the black button",
        target_identity_key="part/table/button_link",
        camera_name="static",
        host_image_key="observation.images.image",
        bbox_xyxy=(10, 20, 30, 40),
        image=image,
        source_rgb_sha256=source_array_sha256("rgb_static", image),
    )


def test_select_pairs_stratifies_limit_per_partition() -> None:
    pairs = (
        _pair("validation", 0),
        _pair("validation", 1),
        _pair("heldout", 0),
        _pair("heldout", 1),
    )
    selected = _select_pairs(pairs, partition="all", limit_per_partition=1)
    assert [(pair.item.partition, pair.item.ordinal) for pair in selected] == [
        ("validation", 0),
        ("heldout", 0),
    ]


def test_public_fixed_x_uses_training_consistent_bounded_processor() -> None:
    source = _TOOL.read_text()
    processor = source.index("public_processor = build_processor")
    configure = source.index("configure_native_processor_area_budget(", processor)
    supervised_preprocess = source.index(
        "supervised_batch = build_native_vl_grounding_batch(", configure
    )
    generation_preprocess = source.index(
        "generation_batch = build_native_vl_generation_batch(", supervised_preprocess
    )
    validate = source.index("grid_budget = validate_native_processor_record_grid(", configure)
    supervised_transfer = source.index("supervised_batch = supervised_batch.to(", validate)
    supervised_forward = source.index("run_native_vl_grounding_forward(", supervised_transfer)
    generation_transfer = source.index("generation_batch = generation_batch.to(", validate)
    generation_forward = source.index(
        "generated_text = generate_native_vl_answer(", generation_transfer
    )

    assert (
        processor
        < configure
        < supervised_preprocess
        < generation_preprocess
        < validate
        < supervised_transfer
        < supervised_forward
        < generation_transfer
        < generation_forward
    )
    assert "supervised_grid_thw != generation_grid_thw" in source
    assert "LATTICE_BASELINE" in source[configure:validate]
    assert '"processor": public_processor_contract' in source


def test_select_pairs_rejects_requested_coverage_shortfall() -> None:
    with pytest.raises(ContractError, match="too few eligible validation"):
        _select_pairs(
            (_pair("validation", 0),),
            partition="validation",
            limit_per_partition=2,
        )


def test_pair_geometry_requires_prompt_conditioned_diagonal_switch() -> None:
    targets = ((100, 100, 200, 200), (700, 700, 800, 800))
    passing = _pair_geometry_metrics(
        ((90, 90, 210, 210), (690, 690, 810, 810)),
        targets,
    )
    assert passing["prediction_bbox_changed"] is True
    assert passing["bidirectional_own_only_center_hit"] is True
    assert passing["mean_diagonal_iou_advantage"] > 0.0

    collapsed = _pair_geometry_metrics(
        ((90, 90, 210, 210), (90, 90, 210, 210)),
        targets,
    )
    assert collapsed["prediction_bbox_changed"] is False
    assert collapsed["bidirectional_own_only_center_hit"] is False
    assert collapsed["mean_diagonal_iou_advantage"] == 0.0


def test_pair_geometry_counts_missing_generation_as_failure() -> None:
    metrics = _pair_geometry_metrics(
        (None, (690, 690, 810, 810)),
        ((100, 100, 200, 200), (700, 700, 800, 800)),
    )
    assert metrics["prediction_bbox_changed"] is False
    assert metrics["bidirectional_own_only_center_hit"] is False
    assert metrics["variants"][0]["own_target_iou"] == 0.0


def test_calvin_semantic_evidence_and_partition_summary_are_recomputable() -> None:
    record = _calvin_record()
    exact_text = f'[{{"label":" Push   Button ","bbox_2d":{list(record.qwen_bbox_xyxy)}}}]'
    wrong_text = f'[{{"label":"drawer","bbox_2d":{list(record.qwen_bbox_xyxy)}}}]'
    exact = {
        "generated_text": exact_text,
        **_calvin_semantic_evidence(
            record,
            parse_native_vl_grounding_answer(exact_text),
        ),
        "target_bbox_qwen_xyxy": list(record.qwen_bbox_xyxy),
    }
    wrong = {
        "generated_text": wrong_text,
        **_calvin_semantic_evidence(
            record,
            parse_native_vl_grounding_answer(wrong_text),
        ),
        "target_bbox_qwen_xyxy": list(record.qwen_bbox_xyxy),
    }

    assert exact["generated_label"] == "Push   Button"
    assert exact["normalized_label_exact_match"] is True
    assert wrong["normalized_label_exact_match"] is False
    assert exact["grounding_request"] == record.grounding_request
    assert (
        exact["grounding_request_sha256"]
        == hashlib.sha256(record.grounding_request.encode("utf-8")).hexdigest()
    )
    assert exact["target_answer"] == record.assistant_text
    assert (
        exact["target_answer_sha256"]
        == hashlib.sha256(record.assistant_text.encode("utf-8")).hexdigest()
    )

    summary = _semantic_partition_summary([{"variants": [exact, wrong]}])
    assert summary == {
        "generated_label_present_count": 2,
        "generated_label_schema_valid_count": 2,
        "item_count": 1,
        "normalized_label_exact_match_count": 1,
        "variant_count": 2,
    }

    malformed = {**wrong, "generated_label_schema_valid": 1}
    with pytest.raises(ContractError, match="semantic flags"):
        _semantic_partition_summary([{"variants": [exact, malformed]}])


def test_fixed_x_visual_renders_both_tasks_and_geometry(tmp_path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    records = (
        SimpleNamespace(
            bbox_xyxy=(2, 3, 8, 10),
            global_index=42,
            image=image,
            target_identity_key="movable/block_blue",
            task_key="lift_blue_block_table",
        ),
        SimpleNamespace(
            bbox_xyxy=(20, 11, 28, 19),
            global_index=42,
            image=image,
            target_identity_key="part/table/slide_link",
            task_key="move_slider_left",
        ),
    )
    variants = (
        SimpleNamespace(
            instruction="pick up the blue block",
            target_identity_key="movable/block_blue",
            task_key="lift_blue_block_table",
        ),
        SimpleNamespace(
            instruction="push the sliding door left",
            target_identity_key="part/table/slide_link",
            task_key="move_slider_left",
        ),
    )
    pair = SimpleNamespace(
        item=SimpleNamespace(
            group=SimpleNamespace(source_global_index=42),
            ordinal=0,
            partition="validation",
            variants=variants,
        ),
        records=records,
    )
    output = tmp_path / "fixed-x.png"
    digest = _render_pair(pair, ((70, 150, 270, 500), (650, 500, 950, 950)), output)
    assert output.is_file()
    assert len(digest) == 64


def test_crossed_x_materialization_rebinds_all_model_visible_fields() -> None:
    record = _calvin_record()
    evidence = CrossedBoundedRecord(
        group_index=0,
        variant_index=0,
        global_index=record.global_index,
        source_episode_index=7,
        source_state_sha256="a" * 64,
        camera_name=record.camera_name,
        source_rgb_sha256=record.source_rgb_sha256,
        task_key=record.task_key,
        instruction_sha256=hashlib.sha256(record.instruction.encode("utf-8")).hexdigest(),
        target_identity_key=record.target_identity_key,
        bbox_qwen_xyxy=record.qwen_bbox_xyxy,
    )
    plan = SimpleNamespace(
        unique_records=(evidence,),
        resolve_record=lambda groups, frozen: (groups[frozen.group_index], "variant"),
    )
    curriculum = SimpleNamespace(groups=("group",))

    materialized = _materialize_crossed_x_records(
        index=SimpleNamespace(),
        sidecar=SimpleNamespace(),
        plan=plan,
        curriculum=curriculum,
        materialize_record=lambda **_kwargs: record,
    )

    assert materialized[0].evidence == evidence
    assert materialized[0].record == record


def test_crossed_x_visual_renders_two_physical_sources(tmp_path) -> None:
    first_record = _calvin_record()
    second_image = np.ones((200, 200, 3), dtype=np.uint8) * 32
    second_image.setflags(write=False)
    second_record = CalvinQwenGroundingRecord(
        global_index=9,
        task_key=first_record.task_key,
        instruction=first_record.instruction,
        target_identity_key=first_record.target_identity_key,
        camera_name=first_record.camera_name,
        host_image_key=first_record.host_image_key,
        bbox_xyxy=(150, 150, 185, 185),
        image=second_image,
        source_rgb_sha256=source_array_sha256("rgb_static", second_image),
    )

    def evidence(record: CalvinQwenGroundingRecord, group_index: int) -> CrossedBoundedRecord:
        return CrossedBoundedRecord(
            group_index=group_index,
            variant_index=0,
            global_index=record.global_index,
            source_episode_index=20 + group_index,
            source_state_sha256=f"{group_index + 1:x}" * 64,
            camera_name=record.camera_name,
            source_rgb_sha256=record.source_rgb_sha256,
            task_key=record.task_key,
            instruction_sha256=hashlib.sha256(record.instruction.encode("utf-8")).hexdigest(),
            target_identity_key=record.target_identity_key,
            bbox_qwen_xyxy=record.qwen_bbox_xyxy,
        )

    pair = CrossedEvaluationPair(
        first=evidence(first_record, 0),
        second=evidence(second_record, 1),
    )
    output = tmp_path / "crossed-x.png"

    digest = _render_crossed_x_pair(
        pair_index=0,
        pair=pair,
        records=(first_record, second_record),
        predictions=(first_record.qwen_bbox_xyxy, second_record.qwen_bbox_xyxy),
        output=output,
    )

    assert output.is_file()
    assert len(digest) == 64


def test_public_retention_summary_keeps_nll_and_family_specific_metrics() -> None:
    rows = [
        {
            "family": "referring",
            "generated_bbox_qwen_xyxy": [100, 100, 200, 200],
            "generated_bbox_schema_valid": True,
            "mean_token_nll": 0.5,
            "supervised_token_count": 4,
            "target_center_hit": True,
            "target_iou": 0.75,
        },
        {
            "family": "vqa",
            "mean_token_nll": 1.0,
            "normalized_exact_match": True,
            "supervised_token_count": 2,
        },
    ]
    summary = _public_retention_summary(rows)
    assert summary["referring"]["mean_record_nll"] == 0.5
    assert summary["referring"]["target_center_hit_count"] == 1
    assert summary["vqa"]["normalized_exact_match_count"] == 1
    assert summary["vqa"]["token_weighted_mean_nll"] == 1.0
    assert _normalize_generated_answer(" Blue  <|im_end|>\n") == "blue"


def test_public_referring_visual_renders_target_and_prediction(tmp_path) -> None:
    output = tmp_path / "public-referring.png"
    digest = _render_public_referring_prediction(
        image=np.zeros((20, 30, 3), dtype=np.uint8),
        record_id="referring-heldout-0000",
        user_text="Locate the blue object.",
        target=(100, 100, 300, 400),
        prediction=(120, 110, 290, 390),
        output=output,
    )
    assert output.is_file()
    assert len(digest) == 64


def test_scene_pair_visual_renders_expected_and_generated_objects(tmp_path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    scene_object = SimpleNamespace(
        bbox_xyxy=(2, 3, 12, 15),
        identity_key="movable/block_blue",
    )
    records = tuple(
        SimpleNamespace(
            camera_name="static",
            category_identity_order=order,
            global_index=42,
            image=image,
            objects=(scene_object,),
        )
        for order in (
            CALVIN_QWEN_SCENE_IDENTITY_ORDER,
            tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER)),
        )
    )
    pair = SimpleNamespace(
        bank_index=0,
        group_index=77,
        records=records,
        task_keys=("lift_blue_block_table", "move_slider_left"),
    )
    generated = parse_native_vl_scene_grounding_answer(
        '[{"label":"blue block","bbox_2d":[70,150,400,750]}]'
    )
    output = tmp_path / "scene-pair.png"
    digest = _render_scene_pair(pair, (generated, generated), output)
    assert output.is_file()
    assert len(digest) == 64
