from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import MappingProxyType

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import NativeRole
from picf_next.lingbot_native.host import LingBotNativeContext
from picf_next.lingbot_native.modalities import NativeModalityBatch, NativeModalityStream
from picf_next.lingbot_native.relation_depth_probe import (
    RELATION_DEPTH_PROBE_ARM,
    RELATION_DEPTH_PROBE_CURVE_NAMES,
    RELATION_DEPTH_PROBE_CURVE_POINT_COUNT,
    RELATION_DEPTH_PROBE_GLOBAL_REFERENCES,
    RELATION_DEPTH_PROBE_LEARNING_RATES,
    RELATION_DEPTH_PROBE_SCHEMA,
    RELATION_DEPTH_PROBE_UPDATE_COUNT,
    RELATION_DEPTH_PROBE_VISUAL_POINTS,
    RELATION_DEPTH_PROBE_WEIGHT_DECAY,
    LingBotRelationDepthCapture,
    build_relation_depth_probe_bank,
    relation_depth_candidates,
    relation_depth_decisions,
    relation_depth_inputs,
    relation_depth_probe_subject,
    relation_depth_recovery_summary,
    relation_depth_surfaces,
    validate_relation_depth_probe_report,
)
from picf_next.lingbot_native.relation_geometry_probe import (
    RelationProbeSampleMetadata,
    RelationProbeSampleSelection,
)
from picf_next.lingbot_native.visual_audit import NATIVE_VISUAL_AUDIT_SCHEMA


class _FakeLingBotLayer(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        *_args: object,
        compute_kqv: bool = False,
        output_atten: bool = False,
        **_kwargs: object,
    ) -> object:
        if compute_kqv:
            return hidden, hidden, hidden
        if output_atten:
            return hidden
        return hidden


class _FakeLingBotPolicy(nn.Module):
    def __init__(self, *, num_layers: int = 36) -> None:
        super().__init__()
        language_model = nn.Module()
        language_model.layers = nn.ModuleList([_FakeLingBotLayer() for _ in range(num_layers)])
        language_model.norm = nn.Identity()
        qwen_model = nn.Module()
        qwen_model.language_model = language_model
        qwenvl = nn.Module()
        qwenvl.model = qwen_model
        qwenvl_with_expert = nn.Module()
        qwenvl_with_expert.qwenvl = qwenvl
        self.model = nn.Module()
        self.model.qwenvl_with_expert = qwenvl_with_expert


def _native_context() -> LingBotNativeContext:
    controls = ExecutedControlBatch.reset_only(
        batch_size=2,
        action_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    modalities = NativeModalityBatch(
        streams=(
            NativeModalityStream(
                name="touch",
                tokens=torch.zeros(2, 2, 4),
                valid=torch.tensor([[True, False], [True, True]]),
            ),
        )
    )
    roles = torch.tensor(
        [
            [
                int(NativeRole.SENSOR),
                int(NativeRole.SENSOR),
                int(NativeRole.LANGUAGE),
                int(NativeRole.LANGUAGE),
                int(NativeRole.HOST_AUX),
            ],
            [
                int(NativeRole.SENSOR),
                int(NativeRole.SENSOR),
                int(NativeRole.LANGUAGE),
                int(NativeRole.LANGUAGE),
                int(NativeRole.HOST_AUX),
            ],
        ],
        dtype=torch.long,
    )
    valid = torch.tensor(
        [
            [True, True, True, True, False],
            [True, False, True, False, False],
        ]
    )
    return LingBotNativeContext(
        controls=controls,
        modalities=modalities,
        native_roles=roles,
        native_valid=valid,
        instruction_last_index=torch.tensor([3, 2], dtype=torch.long),
    )


def test_relation_depth_surfaces_are_equal_quartile_ends() -> None:
    assert tuple(
        (surface.name, surface.layer_index, surface.post_final_norm)
        for surface in relation_depth_surfaces(36)
    ) == (
        ("q1", 8, False),
        ("q2", 17, False),
        ("q3", 26, False),
        ("final", 35, True),
    )
    with pytest.raises(ValueError, match="at least four"):
        relation_depth_surfaces(3)


def test_relation_depth_capture_reads_post_deepstack_next_layer_inputs() -> None:
    policy = _FakeLingBotPolicy()
    language_model = policy.model.qwenvl_with_expert.qwenvl.model.language_model
    capture = LingBotRelationDepthCapture(policy)
    expected: dict[str, list[torch.Tensor]] = {name: [] for name in ("q1", "q2", "q3", "final")}

    with capture:
        for replay in range(2):
            for name, next_layer_index, offset in (
                ("q1", 9, 100),
                ("q2", 18, 200),
                ("q3", 27, 300),
            ):
                hidden = torch.full((2, 7, 8), float(offset + replay))
                expected[name].append(hidden)
                language_model.layers[next_layer_index](hidden, compute_kqv=True)
                language_model.layers[next_layer_index](
                    torch.full_like(hidden, -1),
                    output_atten=True,
                )
            final = torch.full((2, 7, 8), float(400 + replay))
            expected["final"].append(final)
            language_model.norm(final)

    snapshot = capture.snapshot(expected_forward_count=2)
    assert isinstance(snapshot, MappingProxyType)
    for name, expected_values in expected.items():
        assert len(snapshot[name]) == 2
        for observed, expected_value in zip(snapshot[name], expected_values, strict=True):
            torch.testing.assert_close(observed, expected_value)
            assert observed.data_ptr() != expected_value.data_ptr()
            assert not observed.requires_grad


def test_relation_depth_capture_rejects_incomplete_replay() -> None:
    policy = _FakeLingBotPolicy()
    language_model = policy.model.qwenvl_with_expert.qwenvl.model.language_model
    capture = LingBotRelationDepthCapture(policy)
    with capture:
        hidden = torch.zeros(2, 7, 8)
        language_model.layers[9](hidden, compute_kqv=True)
        language_model.layers[18](hidden, compute_kqv=True)
        language_model.layers[27](hidden, compute_kqv=True)
        language_model.norm(hidden)
    with pytest.raises(RuntimeError, match="instead of 2"):
        capture.snapshot(expected_forward_count=2)


def test_relation_depth_inputs_reproduce_production_role_slices() -> None:
    context = _native_context()
    capacity = 3
    hidden = torch.arange(2 * 17 * 8, dtype=torch.float32).reshape(2, 17, 8)

    inputs = relation_depth_inputs(hidden, context=context, capacity=capacity)

    torch.testing.assert_close(inputs.posterior_rows, hidden[:, 11:14])
    torch.testing.assert_close(
        inputs.sensor_hidden,
        torch.cat((hidden[:, :5], hidden[:, 5:7]), dim=1),
    )
    assert torch.equal(
        inputs.sensor_valid,
        torch.tensor(
            [
                [True, True, False, False, False, True, False],
                [True, False, False, False, False, True, True],
            ]
        ),
    )
    assert torch.equal(
        inputs.structural_sensor_valid,
        torch.tensor(
            [
                [True, True, False, False, False, False, False],
                [True, False, False, False, False, False, False],
            ]
        ),
    )
    torch.testing.assert_close(inputs.match_hidden, hidden[:, 14:17])
    torch.testing.assert_close(
        inputs.legacy_instruction_hidden,
        torch.stack((hidden[0, 3], hidden[1, 2])),
    )

    with pytest.raises(ValueError, match="layout"):
        relation_depth_inputs(hidden[:, :-1], context=context, capacity=capacity)


def test_relation_depth_probe_bank_is_identical_disjoint_and_rng_neutral() -> None:
    torch.manual_seed(91)
    rng_before = torch.random.get_rng_state().clone()
    bank, candidates, initialization_sha256 = build_relation_depth_probe_bank(
        host_width=8,
        num_layers=36,
        seed=17,
        device="cpu",
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert len(bank) == len(candidates) == 20
    assert len(initialization_sha256) == 64
    assert tuple(candidate.candidate_id for candidate in candidates) == tuple(bank)
    assert tuple(candidate.learning_rate for candidate in candidates[:5]) == (
        RELATION_DEPTH_PROBE_LEARNING_RATES
    )

    reference = bank[candidates[0].candidate_id]
    reference_state = reference.state_dict()
    trainable_ids: set[int] = set()
    for candidate in candidates:
        readout = bank[candidate.candidate_id]
        for name, value in readout.state_dict().items():
            torch.testing.assert_close(value, reference_state[name])
        assert {
            name for name, parameter in readout.named_parameters() if parameter.requires_grad
        } == {
            "no_object",
            "temperature_parameter",
            "projection.weight",
        }
        for parameter in readout.parameters():
            if parameter.requires_grad:
                assert id(parameter) not in trainable_ids
                trainable_ids.add(id(parameter))
    assert sum(parameter.numel() for parameter in bank.parameters() if parameter.requires_grad) == (
        20 * (8 * 8 + 8 + 1)
    )


def _linear_curve(start: float, stop: float) -> list[float]:
    updates = RELATION_DEPTH_PROBE_UPDATE_COUNT
    return [start + (stop - start) * point / updates for point in range(updates + 1)]


def _sample_selection_dict() -> dict[str, object]:
    return RelationProbeSampleSelection(
        selection_start_global_step=0,
        selected_global_step=37,
        inspected_step_count=38,
        capacity=3,
        samples_by_rank=tuple(
            RelationProbeSampleMetadata(
                sample_key=f"rank{rank}/current",
                task_key="push_blue_block_left",
                available_future_transitions=1,
                target_identity_keys=("object/a",),
                inventory_identity_keys=("object/a", "object/b"),
                target_supervised_pixel_counts=(10,),
            )
            for rank in range(2)
        ),
    ).as_dict()


def _frozen_readout_scope_dict(width: int) -> dict[str, object]:
    parameters = [
        {
            "name": "picf_native_graph.relation_readout.no_object",
            "shape": [width],
            "dtype": "torch.float32",
            "numel": width,
        },
        {
            "name": "picf_native_graph.relation_readout.projection.weight",
            "shape": [width, width],
            "dtype": "torch.float32",
            "numel": width * width,
        },
        {
            "name": "picf_native_graph.relation_readout.temperature_parameter",
            "shape": [1],
            "dtype": "torch.float32",
            "numel": 1,
        },
    ]
    schema = hashlib.sha256(
        json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {
        "arm": "existing_readout_frozen_host",
        "parameter_count": len(parameters),
        "trainable_numel": width * width + width + 1,
        "schema_sha256": schema,
        "parameters": parameters,
    }


def _depth_report() -> dict[str, object]:
    point_zero = RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["point_zero"]
    full_host = RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["structural_full_host_point_40"]
    rank_references = RELATION_DEPTH_PROBE_GLOBAL_REFERENCES["rank_task_soft_iou"]
    provenance = {
        "source_commit": "source",
        "checkpoint_revision": "checkpoint",
        "patch_sha256": "a" * 64,
        "execution_contract_sha256": "b" * 64,
        "implementation_sha256": "c" * 64,
        "model_family_sha256": "d" * 64,
        "plan_sha256": "e" * 64,
        "dataset_manifest_sha256": "f" * 64,
        "physical_sidecar_manifest_sha256": "0" * 64,
        "seed": 7,
        "fixed_sample_global_step": 37,
        "sample_selection": _sample_selection_dict(),
        "forward_seed_by_rank": [1, 2],
        "probe_seed_by_rank": [3, 3],
        "frame_sample_keys_by_rank": [
            ["rank0/current", "rank0/next"],
            ["rank1/current", "rank1/next"],
        ],
        "frame_source_digests_by_rank": [
            ["1" * 64, "2" * 64],
            ["3" * 64, "4" * 64],
        ],
        "row_bindings_by_rank": [
            [["object/a", 0], ["object/b", 1]],
            [["object/a", 0], ["object/b", 1]],
        ],
        "official_action_by_rank": [0.1, 0.2],
        "candidate_initialization_sha256": "1" * 64,
        "host_width": 8,
        "host_layer_count": 36,
        "surfaces": [surface.as_dict() for surface in relation_depth_surfaces(36)],
        "capture": {
            "intermediate_hook": "next_layer_compute_kqv_input_after_block_and_deepstack",
            "final_hook": "post_final_norm",
            "forward_count": 2,
            "feature_dtype": "float32",
            "policy_grad_enabled": False,
            "prediction_queries": "absent",
        },
        "objective": {
            "optimized_term": "set/ownership",
            "observed_terms": list(RELATION_DEPTH_PROBE_CURVE_NAMES),
            "window": "fixed_two_frame_detached_host",
            "labels_are_loss_side_only": True,
            "row_gauge": "production_point_zero_then_frozen",
            "official_policy_loss": "observed_not_optimized",
        },
        "optimizer": {
            "algorithm": "torch.optim.AdamW",
            "learning_rate_hex_grid": [
                value.hex() for value in RELATION_DEPTH_PROBE_LEARNING_RATES
            ],
            "weight_decay_hex": RELATION_DEPTH_PROBE_WEIGHT_DECAY.hex(),
            "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "warmup_updates": 0,
            "update_count": RELATION_DEPTH_PROBE_UPDATE_COUNT,
            "distributed_gradient": "rank_sum_div_world_size",
        },
        "global_references": {
            "point_zero": dict(point_zero),
            "structural_full_host_point_40": dict(full_host),
            "rank_task_soft_iou": [dict(value) for value in rank_references],
        },
    }
    candidates = []
    for candidate in relation_depth_candidates(36):
        global_curves = {
            "ownership": _linear_curve(
                point_zero["ownership"],
                point_zero["ownership"] - 0.6 * (point_zero["ownership"] - full_host["ownership"]),
            ),
            "ownership_nll": _linear_curve(10.0, 5.0),
            "macro_soft_iou": _linear_curve(
                point_zero["macro_soft_iou"],
                point_zero["macro_soft_iou"]
                + 0.6 * (full_host["macro_soft_iou"] - point_zero["macro_soft_iou"]),
            ),
            "task_soft_iou": [],
        }
        rank_reports = []
        for rank, reference in enumerate(rank_references):
            task_curve = _linear_curve(
                reference["point_zero"],
                reference["point_zero"]
                + 0.6 * (reference["structural_full_host_point_40"] - reference["point_zero"]),
            )
            rank_curves = {
                name: list(global_curves[name])
                for name in RELATION_DEPTH_PROBE_CURVE_NAMES
                if name != "task_soft_iou"
            }
            rank_curves["task_soft_iou"] = task_curve
            rank_reports.append(
                {
                    "rank": rank,
                    "curves": rank_curves,
                    "gradient_norm_at_first_update": 0.1,
                    "visual_artifacts_by_point": [
                        {
                            "curve_point": point,
                            "artifacts": [
                                {
                                    "schema": NATIVE_VISUAL_AUDIT_SCHEMA,
                                    "rank": rank,
                                    "global_step": point + 1,
                                    "input_weight_global_step": point,
                                    "weight_boundary": "pre_update_forward",
                                    "path": (
                                        f"{candidate.candidate_id}/visuals/"
                                        f"step_{point + 1:08d}/rank_{rank}/example.png"
                                    ),
                                    "sha256": f"{rank + point + 1:064x}",
                                    "bytes": 1,
                                    "batch_index": 0,
                                    "sample_key": f"rank{rank}/current",
                                    "task": "push blue block left",
                                    "identity_keys": ["object/a", "object/b"],
                                    "source_time": 0,
                                    "source_side": "posterior",
                                    "source_phase": 1,
                                    "binding_start_phase": [1, 1, 2],
                                    "source_binding_valid": [True, True, False],
                                    "row_to_track": [0, 1, -1],
                                    "sequence_row_to_track": [0, 1, -1],
                                    "row_existence": [0.7, 0.2, 0.1],
                                    "row_task_relevance": [0.8, 0.1, 0.1],
                                    "row_matched_soft_iou": [0.2, None, None],
                                    "anchor_surface": "task_object_probability.max(row)",
                                    "views": [{"name": "primary"}],
                                    "loss_only_labels_visible_to_model": False,
                                }
                            ],
                        }
                        for point in RELATION_DEPTH_PROBE_VISUAL_POINTS
                    ],
                    "evaluation_times_s": [0.1] * RELATION_DEPTH_PROBE_CURVE_POINT_COUNT,
                }
            )
        global_curves["task_soft_iou"] = [
            (
                rank_reports[0]["curves"]["task_soft_iou"][point]
                + rank_reports[1]["curves"]["task_soft_iou"][point]
            )
            / 2
            for point in range(RELATION_DEPTH_PROBE_CURVE_POINT_COUNT)
        ]
        recovery = relation_depth_recovery_summary(
            global_curves=global_curves,
            rank_task_curves=[report["curves"]["task_soft_iou"] for report in rank_reports],
        )
        candidates.append(
            {
                "candidate": candidate.as_dict(),
                "trainable_numel": 8 * 8 + 8 + 1,
                "global_curves": global_curves,
                "rank_reports": rank_reports,
                "recovery": recovery,
            }
        )
    return {
        "schema": RELATION_DEPTH_PROBE_SCHEMA,
        "status": "PASS",
        "arm": RELATION_DEPTH_PROBE_ARM,
        "subject_sha256": relation_depth_probe_subject(
            provenance,
            curve_point_count=RELATION_DEPTH_PROBE_CURVE_POINT_COUNT,
        ),
        "provenance": provenance,
        "policy_parameter_boundary": _frozen_readout_scope_dict(8),
        "candidate_initialization_sha256": "1" * 64,
        "curve_point_count": RELATION_DEPTH_PROBE_CURVE_POINT_COUNT,
        "optimizer_update_count": RELATION_DEPTH_PROBE_UPDATE_COUNT,
        "candidates": candidates,
        "depth_decisions": relation_depth_decisions(candidates, num_layers=36),
        "maximum_peak_reserved_bytes": 1,
        "total_time_s": 1.0,
    }


def test_relation_depth_report_recomputes_rank_means_and_recovery() -> None:
    report = _depth_report()
    assert validate_relation_depth_probe_report(report)["status"] == "PASS"
    assert all(item["recoverable"] for item in report["depth_decisions"])

    tampered = deepcopy(report)
    tampered["candidates"][0]["global_curves"]["ownership"][-1] += 0.1
    with pytest.raises(ValueError, match="rank mean"):
        validate_relation_depth_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["candidates"].pop()
    with pytest.raises(ValueError, match="candidate grid"):
        validate_relation_depth_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["provenance"]["probe_seed_by_rank"][1] += 1
    with pytest.raises(ValueError, match="rank invariant"):
        validate_relation_depth_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["provenance"]["frame_sample_keys_by_rank"][0][0] = "wrong/current"
    with pytest.raises(ValueError, match="selected samples"):
        validate_relation_depth_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["candidates"][0]["rank_reports"][0]["visual_artifacts_by_point"][0]["artifacts"][0][
        "path"
    ] = "../escaped.png"
    with pytest.raises(ValueError, match="visual artifact provenance"):
        validate_relation_depth_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["candidates"][0]["rank_reports"][0]["visual_artifacts_by_point"][0]["artifacts"][0][
        "row_to_track"
    ][-1] = None
    with pytest.raises(ValueError, match="visual row metadata"):
        validate_relation_depth_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["candidates"][0]["rank_reports"][0]["visual_artifacts_by_point"][0]["artifacts"][0][
        "row_to_track"
    ][-1] = 2
    with pytest.raises(ValueError, match="visual row metadata"):
        validate_relation_depth_probe_report(tampered)
