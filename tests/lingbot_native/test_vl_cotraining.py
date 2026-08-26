from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenGroundingRecord,
    CalvinQwenSceneGroundingRecord,
    build_calvin_qwen_grounding_records,
)
from picf_next.data.public_native_vl import NativeVLInstructionRecord
from picf_next.lingbot_native.vl_cotraining import (
    NATIVE_VL_GENERATION_MAX_NEW_TOKENS,
    NATIVE_VL_IGNORE_INDEX,
    NativeVLGeneratedGrounding,
    NativeVLGroundingBatch,
    build_counterfactual_scene_grounding_records,
    build_native_vl_generation_batch,
    build_native_vl_grounding_batch,
    compose_native_vl_vla_loss,
    configure_native_vl_grounding_trainable_scope,
    generate_native_vl_answer,
    native_vl_stream_factor,
    parse_native_vl_grounding_answer,
    parse_native_vl_scene_grounding_answer,
    qwen_grounding_bbox_iou,
    qwen_target_center_in_bbox,
    retie_and_validate_native_qwen_lm_head,
    run_native_vl_grounding_forward,
    validate_native_vl_generation_model_kwargs,
    validate_native_vl_model_kwargs,
    validate_native_vl_optimizer_membership,
    validate_tied_qwen_lm_head,
    verify_native_vl_grounding_trainable_scope,
)


def _record() -> CalvinQwenGroundingRecord:
    image = torch.zeros(200, 200, 3, dtype=torch.uint8).numpy()
    image.setflags(write=False)
    return CalvinQwenGroundingRecord(
        global_index=3,
        task_key="turn_on_led",
        instruction="press the black button",
        target_identity_key="part/table/button_link",
        camera_name="static",
        host_image_key="observation.images.image",
        bbox_xyxy=(1, 2, 8, 9),
        image=image,
        source_rgb_sha256=source_array_sha256("rgb_static", image),
    )


def _public_record() -> NativeVLInstructionRecord:
    image = torch.zeros(24, 32, 3, dtype=torch.uint8).numpy()
    image.setflags(write=False)
    return NativeVLInstructionRecord(
        record_id="vqa-train-0000",
        family="vqa",
        user_text="Is the blue object visible?",
        assistant_text="blue",
        image=image,
    )


def _scene_frame_and_targets() -> tuple[
    CalvinPhysicalSupervisionFrame,
    tuple[CalvinQwenGroundingRecord, CalvinQwenGroundingRecord],
]:
    static_image = np.zeros((200, 200, 3), dtype=np.uint8)
    gripper_image = np.zeros((84, 84, 3), dtype=np.uint8)
    static_owner = np.zeros((200, 200), dtype=np.uint8)
    for slot, owner_id in enumerate(range(1, 11)):
        row = (slot // 5) * 24
        column = (slot % 5) * 24
        static_owner[row : row + 13, column : column + 13] = owner_id
    gripper_owner = np.zeros((84, 84), dtype=np.uint8)

    def camera(camera_name: str, owner_index: np.ndarray, image: np.ndarray):
        is_static = camera_name == "static"
        return CalvinVisibleOwnerRaster(
            camera_name=camera_name,
            host_image_key=(
                "observation.images.image" if is_static else "observation.images.wrist_image"
            ),
            owner_index=owner_index,
            owner_supervised=np.ones_like(owner_index, dtype=np.bool_),
            source_rgb_sha256=source_array_sha256(
                "rgb_static" if is_static else "rgb_gripper",
                image,
            ),
            source_depth_sha256=("1" if is_static else "2") * 64,
            rgb_mae=0.0,
            depth_mae_m=0.0,
            depth_p95_m=0.0,
            depth_consistent_fraction=1.0,
        )

    dimension = CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension
    frame = CalvinPhysicalSupervisionFrame(
        identity_keys=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
        geometry=torch.zeros(10, dimension),
        geometry_variance=torch.zeros(10, dimension),
        geometry_supervised=torch.ones(10, dimension, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(
            camera("static", static_owner, static_image),
            camera("gripper", gripper_owner, gripper_image),
        ),
    )
    observation_images = {
        "observation.images.image": static_image,
        "observation.images.wrist_image": gripper_image,
    }
    button = build_calvin_qwen_grounding_records(
        global_index=44,
        task_key="turn_on_led",
        instruction="press the button",
        observation_images=observation_images,
        physical_frame=frame,
    )[0]
    switch = build_calvin_qwen_grounding_records(
        global_index=44,
        task_key="turn_on_lightbulb",
        instruction="flip the switch",
        observation_images=observation_images,
        physical_frame=frame,
    )[0]
    return frame, (button, switch)


class _Processor:
    def __init__(self, tokens: torch.Tensor | None = None) -> None:
        self.tokens = (
            torch.tensor([[1, 77091, 2, 10, 11, 151645, 12]], dtype=torch.long)
            if tokens is None
            else tokens
        )
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        return {
            "input_ids": self.tokens,
            "attention_mask": torch.ones_like(self.tokens),
            "pixel_values": torch.zeros(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


class _Tokenizer:
    @staticmethod
    def decode(_tokens, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "<|im_start|>assistant\n"


class _GenerationProcessor:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.zeros(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }


def _batch() -> NativeVLGroundingBatch:
    return build_native_vl_grounding_batch(_record(), _Processor())


def test_official_qwen_preprocessing_supervises_only_assistant_answer() -> None:
    processor = _Processor()
    batch = build_native_vl_grounding_batch(_record(), processor)
    assert batch.batch_size == 1
    assert batch.supervised_token_count == 4
    assert batch.labels.tolist() == [
        [NATIVE_VL_IGNORE_INDEX, NATIVE_VL_IGNORE_INDEX, NATIVE_VL_IGNORE_INDEX, 10, 11, 151645, 12]
    ]
    assert set(batch.model_kwargs()) == {
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "labels",
        "pixel_values",
    }
    processor_image = processor.messages[0]["content"][0]["image"]
    assert processor_image.flags.writeable


def test_official_qwen_preprocessing_accepts_generic_public_instruction() -> None:
    processor = _Processor()
    batch = build_native_vl_grounding_batch(_public_record(), processor)
    assert batch.supervised_token_count == 4
    assert processor.messages[0]["role"] == "user"
    assert processor.messages[1]["role"] == "assistant"
    assert processor.messages[1]["content"][0]["text"] == "blue"


def test_official_qwen_preprocessing_accepts_scene_grounding_without_side_channels() -> None:
    frame, targets = _scene_frame_and_targets()
    scene = build_counterfactual_scene_grounding_records(
        targets,
        frame,
        visual_lattice=8,
    )[0]
    processor = _Processor()
    batch = build_native_vl_grounding_batch(scene, processor)

    assert batch.supervised_token_count == 4
    assert len(processor.messages) == 2
    assert processor.messages[0]["role"] == "user"
    request = processor.messages[0]["content"][1]["text"]
    assert "press the button" not in request
    assert "flip the switch" not in request
    assert all(identity_key not in request for identity_key in CALVIN_QWEN_SCENE_IDENTITY_ORDER)
    assert processor.messages[1]["content"][0]["text"] == scene.assistant_text


def test_counterfactual_scene_records_change_only_category_and_answer_order() -> None:
    frame, targets = _scene_frame_and_targets()
    canonical, reverse = build_counterfactual_scene_grounding_records(
        targets,
        frame,
        visual_lattice=8,
    )

    assert isinstance(canonical, CalvinQwenSceneGroundingRecord)
    assert np.array_equal(canonical.image, targets[0].image)
    assert np.array_equal(reverse.image, targets[1].image)
    assert not canonical.image.flags.writeable
    assert not reverse.image.flags.writeable
    assert np.array_equal(canonical.image, reverse.image)
    assert canonical.source_rgb_sha256 == reverse.source_rgb_sha256
    assert canonical.category_identity_order == CALVIN_QWEN_SCENE_IDENTITY_ORDER
    assert reverse.category_identity_order == tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER))
    assert tuple(item.identity_key for item in canonical.objects) == tuple(
        reversed(tuple(item.identity_key for item in reverse.objects))
    )
    assert {
        item.identity_key: (item.bbox_xyxy, item.visible_owner_pixels) for item in canonical.objects
    } == {
        item.identity_key: (item.bbox_xyxy, item.visible_owner_pixels) for item in reverse.objects
    }


def test_native_qwen_generation_allows_natural_answer_word_in_user_prompt() -> None:
    processor = _GenerationProcessor()
    build_native_vl_generation_batch(_public_record(), processor)
    assert len(processor.messages) == 1
    assert processor.messages[0]["role"] == "user"


def test_qwen_preprocessing_rejects_missing_or_duplicate_assistant_markers() -> None:
    missing = _Processor(torch.tensor([[1, 2, 151645]], dtype=torch.long))
    with pytest.raises(ContractError, match="one assistant header"):
        build_native_vl_grounding_batch(_record(), missing)
    duplicate = _Processor(
        torch.tensor([[77091, 2, 4, 151645, 77091, 2, 5, 151645]], dtype=torch.long)
    )
    with pytest.raises(ContractError, match="one assistant header"):
        build_native_vl_grounding_batch(_record(), duplicate)


def test_qwen_generation_preprocessing_has_no_teacher_answer_or_labels() -> None:
    processor = _GenerationProcessor()
    batch = build_native_vl_generation_batch(_record(), processor)
    assert batch.prompt_token_count == 3
    assert set(batch.model_kwargs()) == {
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
    }
    assert len(processor.messages) == 1
    assert processor.messages[0]["role"] == "user"
    assert processor.messages[0]["content"][0]["image"].flags.writeable
    invalid = batch.model_kwargs()
    invalid["labels"] = torch.zeros_like(batch.input_ids)
    with pytest.raises(ContractError, match="approved Qwen surface"):
        validate_native_vl_generation_model_kwargs(invalid)


class _GenerateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        prompt = kwargs["input_ids"]
        continuation = torch.tensor([[90, 91]], dtype=torch.long)
        return torch.cat((prompt, continuation), dim=1)


class _GeneratedTokenizer:
    @staticmethod
    def decode(tokens, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        assert tokens.tolist() == [90, 91]
        return '[{"bbox_2d":[1,2,3,4]}]'


def test_native_qwen_generation_uses_prompt_only_and_decodes_only_new_tokens() -> None:
    batch = build_native_vl_generation_batch(_record(), _GenerationProcessor())
    model = _GenerateModel()
    answer = generate_native_vl_answer(
        model,
        batch,
        _GeneratedTokenizer(),
        max_new_tokens=32,
    )
    assert answer == '[{"bbox_2d":[1,2,3,4]}]'
    assert set(model.kwargs) == {
        "attention_mask",
        "do_sample",
        "image_grid_thw",
        "input_ids",
        "max_new_tokens",
        "pixel_values",
        "use_cache",
    }
    answer = generate_native_vl_answer(
        model,
        batch,
        _GeneratedTokenizer(),
        max_new_tokens=NATIVE_VL_GENERATION_MAX_NEW_TOKENS,
    )
    assert answer == '[{"bbox_2d":[1,2,3,4]}]'
    assert model.kwargs["max_new_tokens"] == NATIVE_VL_GENERATION_MAX_NEW_TOKENS
    with pytest.raises(ContractError, match=r"\[1,1024\]"):
        generate_native_vl_answer(model, batch, _GeneratedTokenizer(), max_new_tokens=0)
    with pytest.raises(ContractError, match=r"\[1,1024\]"):
        generate_native_vl_answer(
            model,
            batch,
            _GeneratedTokenizer(),
            max_new_tokens=NATIVE_VL_GENERATION_MAX_NEW_TOKENS + 1,
        )


def test_native_qwen_generation_parser_reports_schema_and_recoverable_geometry() -> None:
    strict = parse_native_vl_grounding_answer(
        '```json\n[{"bbox_2d":[100,200,300,400]}]\n```<|im_end|>'
    )
    assert strict.bbox_qwen_xyxy == (100, 200, 300, 400)
    assert strict.schema_valid
    assert strict.generated_label is None
    assert not strict.label_present
    assert not strict.label_schema_valid

    official_labeled = parse_native_vl_grounding_answer(
        '[{"bbox_2d":[100,200,300,400],"label":"slider"}]'
    )
    assert official_labeled.bbox_qwen_xyxy == (100, 200, 300, 400)
    assert official_labeled.schema_valid
    assert official_labeled.generated_label == "slider"
    assert official_labeled.label_present
    assert official_labeled.label_schema_valid

    invalid_label = parse_native_vl_grounding_answer('[{"bbox_2d":[100,200,300,400],"label":"  "}]')
    assert invalid_label.schema_valid
    assert invalid_label.generated_label is None
    assert invalid_label.label_present
    assert not invalid_label.label_schema_valid

    malformed_outer_list = parse_native_vl_grounding_answer(
        '```json\n[\n{"bbox_2d":[100,200,300,400],"label":"slider"}\n```<|im_end|>'
    )
    assert malformed_outer_list.bbox_qwen_xyxy == (100, 200, 300, 400)
    assert not malformed_outer_list.schema_valid
    assert malformed_outer_list.generated_label == "slider"
    assert malformed_outer_list.label_present
    assert not malformed_outer_list.label_schema_valid

    invalid = parse_native_vl_grounding_answer('[{"bbox_2d":[300,200,100,400]}]')
    assert invalid.bbox_qwen_xyxy is None
    assert not invalid.schema_valid

    with pytest.raises(ContractError, match="strict nonempty label"):
        NativeVLGeneratedGrounding(
            bbox_qwen_xyxy=(100, 200, 300, 400),
            schema_valid=True,
            label_present=True,
            label_schema_valid=True,
        )


def test_native_qwen_scene_parser_requires_complete_unique_json_objects() -> None:
    parsed = parse_native_vl_scene_grounding_answer(
        "```json\n"
        '[{"label":"blue block","bbox_2d":[1,2,30,40]},'
        '{"label":"push button","bbox_2d":[100,200,300,400]}]'
        "\n```<|im_end|>"
    )
    assert parsed.schema_valid
    assert tuple(item.label for item in parsed.objects) == ("blue block", "push button")
    assert tuple(item.bbox_qwen_xyxy for item in parsed.objects) == (
        (1, 2, 30, 40),
        (100, 200, 300, 400),
    )

    invalid_answers = (
        "{}",
        '[{"label":"blue block","bbox_2d":[1,2,30,40]},'
        '{"label":" BLUE   BLOCK ","bbox_2d":[2,3,31,41]}]',
        '[{"label":"blue block","bbox_2d":[30,2,1,40]}]',
        '[{"bbox_2d":[1,2,30,40]}]',
        '[[{"label":"blue block","bbox_2d":[1,2,30,40]}]]',
        '[{"label":"blue block","bbox_2d":[1,2,30,40],"score":1}]',
        '[{"label":"blue block","bbox_2d":[1,2,30,40]}',
    )
    for answer in invalid_answers:
        invalid = parse_native_vl_scene_grounding_answer(answer)
        assert not invalid.schema_valid
        assert invalid.objects == ()

    empty = parse_native_vl_scene_grounding_answer("[]")
    assert empty.schema_valid
    assert empty.objects == ()


def test_native_qwen_generation_geometry_metrics_are_continuous_and_threshold_free() -> None:
    predicted = (100, 100, 300, 300)
    target = (200, 200, 400, 400)
    assert qwen_grounding_bbox_iou(predicted, target) == pytest.approx(1.0 / 7.0)
    assert qwen_target_center_in_bbox(predicted, target)
    assert not qwen_target_center_in_bbox((0, 0, 100, 100), target)


def test_native_vl_batch_rejects_prompt_or_mismatched_supervision() -> None:
    batch = _batch()
    labels = batch.labels.clone()
    labels[0, 0] = batch.input_ids[0, 0]
    with pytest.raises(ContractError, match="exactly assistant"):
        NativeVLGroundingBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=labels,
            assistant_token_mask=batch.assistant_token_mask,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
        )
    labels = batch.labels.clone()
    labels[batch.assistant_token_mask] += 1
    with pytest.raises(ContractError, match="equal their input"):
        NativeVLGroundingBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=labels,
            assistant_token_mask=batch.assistant_token_mask,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
        )


def test_forbidden_privileged_model_kwarg_fails_closed() -> None:
    kwargs = _batch().model_kwargs()
    kwargs["owner_index"] = torch.zeros(1)
    with pytest.raises(ContractError, match="approved Qwen surface"):
        validate_native_vl_model_kwargs(kwargs)


class _Policy(nn.Module):
    def __init__(self, detached: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.detached = detached

    def picf_native_vl_forward(self, **kwargs):
        assert set(kwargs) == {
            "attention_mask",
            "image_grid_thw",
            "input_ids",
            "labels",
            "pixel_values",
        }
        loss = self.scale.square()
        return loss.detach() if self.detached else loss


def test_native_vl_forward_requires_finite_attached_scalar() -> None:
    loss = run_native_vl_grounding_forward(_Policy(), _batch())
    assert loss.item() == pytest.approx(4.0)
    with pytest.raises(ContractError, match="detached"):
        run_native_vl_grounding_forward(_Policy(detached=True), _batch())


def test_optional_grounding_factor_requires_explicit_weight_and_schedule() -> None:
    robot = torch.tensor(2.0, requires_grad=True)
    grounding = torch.tensor(3.0, requires_grad=True)
    assert compose_native_vl_vla_loss(robot, None, grounding_weight=0.1) is robot
    assert compose_native_vl_vla_loss(
        robot,
        grounding,
        grounding_weight=0.1,
    ).item() == pytest.approx(2.3)
    assert tuple(
        native_vl_stream_factor(index, robot_steps=9, grounding_steps=1) for index in range(20)
    ) == (
        *(("robot",) * 9),
        "grounding",
        *(("robot",) * 9),
        "grounding",
    )
    with pytest.raises(ContractError, match="robot steps"):
        native_vl_stream_factor(0, robot_steps=0, grounding_steps=1)
    with pytest.raises(ContractError, match="grounding steps"):
        native_vl_stream_factor(0, robot_steps=1, grounding_steps=True)


class _TiedPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embedding = nn.Embedding(8, 4)
        lm_head = nn.Linear(4, 8, bias=False)
        lm_head.weight = embedding.weight
        qwen = nn.Module()
        qwen.model = nn.Module()
        qwen.model.language_model = nn.Module()
        qwen.model.language_model.embed_tokens = embedding
        qwen.lm_head = lm_head
        self.model = nn.Module()
        self.model.qwenvl_with_expert = nn.Module()
        self.model.qwenvl_with_expert.qwenvl = qwen


class _AdaptationPolicy(_TiedPolicy):
    def __init__(self) -> None:
        super().__init__()
        qwen = self.model.qwenvl_with_expert.qwenvl
        qwen.model.language_model.layers = nn.ModuleList([nn.Linear(4, 4)])
        qwen.model.visual = nn.Module()
        qwen.model.visual.patch_embed = nn.Linear(4, 4)
        qwen.model.visual.merger = nn.Linear(4, 4)
        self.model.qwenvl_with_expert.qwen_expert = nn.Linear(4, 4)
        self.picf_native_graph = nn.Linear(4, 4)


class _UntiedPolicy(_TiedPolicy):
    def __init__(self) -> None:
        super().__init__()
        qwen = self.model.qwenvl_with_expert.qwenvl
        qwen.lm_head.weight = nn.Parameter(torch.zeros_like(qwen.lm_head.weight))

        def tie_weights() -> None:
            qwen.lm_head.weight = qwen.model.language_model.embed_tokens.weight

        qwen.tie_weights = tie_weights


def test_tied_qwen_head_and_optimizer_membership_are_explicit() -> None:
    policy = _TiedPolicy()
    name = validate_tied_qwen_lm_head(policy)
    assert name.endswith("embed_tokens.weight")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    assert validate_native_vl_optimizer_membership(policy, optimizer) == name
    empty = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=1e-4)
    with pytest.raises(ContractError, match="absent from the optimizer"):
        validate_native_vl_optimizer_membership(policy, empty)


def test_qwen_native_tie_hook_repairs_and_proves_alias() -> None:
    policy = _UntiedPolicy()
    with pytest.raises(ContractError, match="not tied"):
        validate_tied_qwen_lm_head(policy)
    assert retie_and_validate_native_qwen_lm_head(policy).endswith("embed_tokens.weight")


def test_grounding_adaptation_scope_is_shared_qwen_language_and_merger_only() -> None:
    policy = _AdaptationPolicy()
    scope = configure_native_vl_grounding_trainable_scope(policy)
    assert scope.trainable_numel > 0
    assert all(
        ".qwenvl.model.language_model." in name or ".qwenvl.model.visual.merger." in name
        for name in scope.parameter_names
    )
    assert not any(
        "qwen_expert" in name or "picf_native_graph" in name for name in scope.parameter_names
    )
    assert not policy.model.qwenvl_with_expert.qwenvl.model.visual.patch_embed.weight.requires_grad
    assert verify_native_vl_grounding_trainable_scope(policy, expected=scope) == scope

    policy.picf_native_graph.weight.requires_grad_(True)
    with pytest.raises(ContractError, match="scope"):
        verify_native_vl_grounding_trainable_scope(policy, expected=scope)
