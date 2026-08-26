from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import torch

import picf_next.lingbot_native.relation_bilinear_probe as bilinear_probe
from picf_next.lingbot_native.relation_bilinear_probe import (
    RELATION_BILINEAR_PROBE_ARM,
    RELATION_BILINEAR_PROBE_CURVE_NAMES,
    RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT,
    RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES,
    RELATION_BILINEAR_PROBE_LEARNING_RATES,
    RELATION_BILINEAR_PROBE_MODES,
    RELATION_BILINEAR_PROBE_SCHEMA,
    RELATION_BILINEAR_PROBE_UPDATE_COUNT,
    RELATION_BILINEAR_PROBE_VISUAL_POINTS,
    RELATION_BILINEAR_PROBE_WEIGHT_DECAY,
    FullRankBilinearRelationReadout,
    build_relation_bilinear_probe_bank,
    relation_bilinear_candidates,
    relation_bilinear_control_identity_sha256,
    relation_bilinear_decisions,
    relation_bilinear_probe_subject,
    validate_relation_bilinear_probe_report,
)
from picf_next.lingbot_native.relation_depth_probe import (
    build_relation_depth_probe_bank,
    relation_depth_recovery_summary,
    relation_depth_surfaces,
)
from picf_next.lingbot_native.relation_geometry_probe import (
    RelationProbeSampleMetadata,
    RelationProbeSampleSelection,
)
from picf_next.lingbot_native.visual_audit import NATIVE_VISUAL_AUDIT_SCHEMA


def _inputs(width: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(31)
    return {
        "posterior_rows": torch.randn(2, 3, width),
        "sensor_hidden": torch.randn(2, 5, width),
        "sensor_valid": torch.tensor(
            [
                [True, True, False, True, False],
                [True, False, True, True, True],
            ]
        ),
        "instruction_hidden": torch.randn(2, width),
        "structural_sensor_valid": torch.tensor(
            [
                [True, False, False, True, False],
                [True, False, False, True, False],
            ]
        ),
    }


def test_bilinear_identity_point_zero_matches_legacy_identity_control() -> None:
    width = 8
    control = FullRankBilinearRelationReadout(width, mode="unconstrained")
    inputs = _inputs(width)
    expected = control(**inputs)

    for mode in RELATION_BILINEAR_PROBE_MODES:
        readout = FullRankBilinearRelationReadout(width, mode=mode)
        readout.load_state_dict(control.state_dict(), strict=True)
        observed = readout(**inputs)
        for name in (
            "support_logits",
            "visible_support",
            "ownership",
            "task_relevance",
            "task_relevance_logits",
            "dense_task_grounding",
            "dense_task_grounding_logits",
            "existence",
            "existence_logits",
        ):
            torch.testing.assert_close(getattr(observed, name), getattr(expected, name))
        assert torch.equal(observed.sensor_valid, expected.sensor_valid)
        assert torch.equal(observed.structural_valid, expected.structural_valid)


def test_bilinear_modes_encode_the_preregistered_symmetry_split() -> None:
    symmetric = FullRankBilinearRelationReadout(2, mode="symmetric_indefinite")
    unconstrained = FullRankBilinearRelationReadout(2, mode="unconstrained")
    weight = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
    with torch.no_grad():
        symmetric.projection.weight.copy_(weight)
        unconstrained.projection.weight.copy_(weight)

    torch.testing.assert_close(
        symmetric.effective_projection(),
        torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
    )
    torch.testing.assert_close(unconstrained.effective_projection(), weight)

    common = {
        "sensor_valid": torch.ones(1, 2, dtype=torch.bool),
        "instruction_hidden": torch.tensor([[1.0, 0.0]]),
    }
    forward = {
        **common,
        "sensor_hidden": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "posterior_rows": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
    }
    symmetric_logits = symmetric(**forward).support_logits[0]
    unconstrained_logits = unconstrained(**forward).support_logits[0]
    torch.testing.assert_close(symmetric_logits[0, 0], symmetric_logits[1, 1])
    assert unconstrained_logits[0, 0] > 0
    assert unconstrained_logits[1, 1] < 0


def test_bilinear_forward_matches_the_direct_matrix_equation() -> None:
    readout = FullRankBilinearRelationReadout(4, mode="unconstrained")
    inputs = _inputs(4)
    with torch.no_grad():
        readout.projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.2, -0.1, 0.3],
                    [-0.4, 0.8, 0.5, 0.1],
                    [0.6, -0.2, 1.1, -0.3],
                    [0.2, 0.7, -0.5, 0.9],
                ]
            )
        )
    output = readout(**inputs)

    sensors = torch.nn.functional.normalize(
        inputs["sensor_hidden"],
        dim=-1,
        eps=readout.norm_epsilon,
    )
    rows = torch.nn.functional.normalize(
        inputs["posterior_rows"],
        dim=-1,
        eps=readout.norm_epsilon,
    )
    no_object = torch.nn.functional.normalize(
        readout.no_object,
        dim=0,
        eps=readout.norm_epsilon,
    )
    projection = readout.projection.weight
    expected_support = (
        torch.einsum(
            "bnd,de,bke->bnk",
            sensors,
            projection,
            rows,
        )
        / readout.temperature
    )
    expected_no_object = (
        torch.einsum(
            "bnd,de,e->bn",
            sensors,
            projection,
            no_object,
        )
        / readout.temperature
    )
    expected_ownership = torch.softmax(
        torch.cat((expected_support, expected_no_object.unsqueeze(-1)), dim=-1),
        dim=-1,
    )
    expected_ownership = expected_ownership * inputs["sensor_valid"].unsqueeze(-1)

    torch.testing.assert_close(output.support_logits, expected_support)
    torch.testing.assert_close(output.ownership, expected_ownership)


def test_bilinear_ownership_update_does_not_rewrite_task_geometry() -> None:
    readout = FullRankBilinearRelationReadout(4, mode="unconstrained")
    inputs = _inputs(4)
    before = readout(**inputs)
    with torch.no_grad():
        readout.projection.weight.add_(torch.randn_like(readout.projection.weight))
        readout.temperature_parameter.add_(1.0)
    after = readout(**inputs)

    assert not torch.allclose(before.support_logits, after.support_logits)
    torch.testing.assert_close(before.task_relevance_logits, after.task_relevance_logits)
    torch.testing.assert_close(
        before.dense_task_grounding_logits,
        after.dense_task_grounding_logits,
    )


def test_bilinear_no_object_uses_the_same_cross_role_geometry() -> None:
    readout = FullRankBilinearRelationReadout(2, mode="unconstrained")
    with torch.no_grad():
        readout.no_object.copy_(torch.tensor([0.0, 1.0]))
    inputs = {
        "posterior_rows": torch.tensor([[[1.0, 0.0]]]),
        "sensor_hidden": torch.tensor([[[1.0, 0.0]]]),
        "sensor_valid": torch.ones(1, 1, dtype=torch.bool),
        "instruction_hidden": torch.tensor([[1.0, 0.0]]),
    }
    before = readout(**inputs)
    with torch.no_grad():
        readout.projection.weight[0, 1] = 2.0
    after = readout(**inputs)

    torch.testing.assert_close(before.support_logits, after.support_logits)
    assert after.ownership[0, 0, -1] > before.ownership[0, 0, -1]


def test_bilinear_probe_bank_is_equal_capacity_disjoint_and_rng_neutral() -> None:
    torch.manual_seed(73)
    rng_before = torch.random.get_rng_state().clone()
    bank, candidates, initialization_sha256 = build_relation_bilinear_probe_bank(
        host_width=8,
        seed=19,
        device="cpu",
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert len(bank) == len(candidates) == 10
    assert len(initialization_sha256) == 64
    assert tuple(candidate.candidate_id for candidate in candidates) == tuple(bank)
    assert tuple(candidate.learning_rate for candidate in candidates[:5]) == (
        RELATION_BILINEAR_PROBE_LEARNING_RATES
    )
    reference_state = bank[candidates[0].candidate_id].state_dict()
    parameter_ids: set[int] = set()
    for candidate in candidates:
        readout = bank[candidate.candidate_id]
        assert readout.mode == candidate.mode
        for name, value in readout.state_dict().items():
            torch.testing.assert_close(value, reference_state[name])
        assert {
            name for name, parameter in readout.named_parameters() if parameter.requires_grad
        } == {
            "projection.weight",
            "no_object",
            "temperature_parameter",
        }
        for parameter in readout.parameters():
            if parameter.requires_grad:
                assert id(parameter) not in parameter_ids
                parameter_ids.add(id(parameter))
    assert sum(parameter.numel() for parameter in bank.parameters() if parameter.requires_grad) == (
        len(candidates) * (8 * 8 + 8 + 1)
    )


def test_bilinear_disjoint_candidate_backward_matches_summed_loss() -> None:
    summed_bank, candidates, _ = build_relation_bilinear_probe_bank(
        host_width=8,
        seed=19,
        device="cpu",
    )
    sequential_bank, _, _ = build_relation_bilinear_probe_bank(
        host_width=8,
        seed=19,
        device="cpu",
    )
    inputs = _inputs(8)

    def losses(bank: torch.nn.ModuleDict) -> tuple[torch.Tensor, ...]:
        return tuple(
            -bank[candidate.candidate_id](**inputs)
            .ownership[..., 0][inputs["sensor_valid"]]
            .clamp_min(1e-8)
            .log()
            .mean()
            for candidate in candidates
        )

    torch.stack(losses(summed_bank)).sum().backward()
    for loss in losses(sequential_bank):
        loss.backward()

    for candidate in candidates:
        summed_parameters = dict(summed_bank[candidate.candidate_id].named_parameters())
        sequential_parameters = dict(sequential_bank[candidate.candidate_id].named_parameters())
        for name in (
            "projection.weight",
            "no_object",
            "temperature_parameter",
        ):
            torch.testing.assert_close(
                sequential_parameters[name].grad,
                summed_parameters[name].grad,
            )


def test_bilinear_probe_reuses_the_exact_depth_control_initialization() -> None:
    bilinear_bank, bilinear_candidates, bilinear_sha256 = build_relation_bilinear_probe_bank(
        host_width=8,
        seed=19,
        device="cpu",
    )
    depth_bank, depth_candidates, depth_sha256 = build_relation_depth_probe_bank(
        host_width=8,
        num_layers=4,
        seed=19,
        device="cpu",
    )

    assert bilinear_sha256 == depth_sha256
    bilinear_state = bilinear_bank[bilinear_candidates[0].candidate_id].state_dict()
    depth_state = depth_bank[depth_candidates[0].candidate_id].state_dict()
    assert tuple(bilinear_state) == tuple(depth_state)
    for name in bilinear_state:
        torch.testing.assert_close(bilinear_state[name], depth_state[name])


def test_bilinear_decisions_require_adjacent_learning_rates_per_mode() -> None:
    reports = []
    for candidate in relation_bilinear_candidates():
        passes = (
            candidate.mode == "unconstrained" and candidate.learning_rate_index in (1, 2)
        ) or (candidate.mode == "symmetric_indefinite" and candidate.learning_rate_index == 4)
        reports.append(
            {
                "candidate": candidate.as_dict(),
                "recovery": {"passes_half_recovery": passes},
            }
        )

    decisions = relation_bilinear_decisions(reports)
    assert decisions == [
        {
            "mode": "symmetric_indefinite",
            "passing_learning_rate_indices": [4],
            "adjacent_passing_pairs": [],
            "recoverable": False,
        },
        {
            "mode": "unconstrained",
            "passing_learning_rate_indices": [1, 2],
            "adjacent_passing_pairs": [[1, 2]],
            "recoverable": True,
        },
    ]


def _linear_curve(start: float, stop: float) -> list[float]:
    updates = RELATION_BILINEAR_PROBE_UPDATE_COUNT
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


def _bilinear_report() -> dict[str, object]:
    point_zero = RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES["point_zero"]
    full_host = RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES["structural_full_host_point_40"]
    rank_references = RELATION_BILINEAR_PROBE_GLOBAL_REFERENCES["rank_task_soft_iou"]
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
            "intermediate_hook": ("next_layer_compute_kqv_input_after_block_and_deepstack"),
            "final_hook": "post_final_norm",
            "forward_count": 2,
            "feature_dtype": "float32",
            "policy_grad_enabled": False,
            "prediction_queries": "absent",
        },
        "objective": {
            "optimized_term": "set/ownership",
            "observed_terms": list(RELATION_BILINEAR_PROBE_CURVE_NAMES),
            "window": "fixed_two_frame_detached_host",
            "labels_are_loss_side_only": True,
            "row_gauge": "production_point_zero_then_frozen",
            "official_policy_loss": "observed_not_optimized",
        },
        "optimizer": {
            "algorithm": "torch.optim.AdamW",
            "learning_rate_hex_grid": [
                value.hex() for value in RELATION_BILINEAR_PROBE_LEARNING_RATES
            ],
            "weight_decay_hex": RELATION_BILINEAR_PROBE_WEIGHT_DECAY.hex(),
            "scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "warmup_updates": 0,
            "update_count": RELATION_BILINEAR_PROBE_UPDATE_COUNT,
            "distributed_gradient": "rank_sum_div_world_size",
        },
        "global_references": {
            "point_zero": dict(point_zero),
            "structural_full_host_point_40": dict(full_host),
            "rank_task_soft_iou": [dict(value) for value in rank_references],
        },
    }
    candidate_reports = []
    for candidate in relation_bilinear_candidates():
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
            rank_curves = {
                name: list(global_curves[name])
                for name in RELATION_BILINEAR_PROBE_CURVE_NAMES
                if name != "task_soft_iou"
            }
            rank_curves["task_soft_iou"] = _linear_curve(
                reference["point_zero"],
                reference["point_zero"]
                + 0.6 * (reference["structural_full_host_point_40"] - reference["point_zero"]),
            )
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
                        for point in RELATION_BILINEAR_PROBE_VISUAL_POINTS
                    ],
                    "evaluation_times_s": [0.1] * RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT,
                }
            )
        global_curves["task_soft_iou"] = [
            sum(report["curves"]["task_soft_iou"][point] for report in rank_reports) / 2
            for point in range(RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT)
        ]
        recovery = relation_depth_recovery_summary(
            global_curves=global_curves,
            rank_task_curves=[report["curves"]["task_soft_iou"] for report in rank_reports],
        )
        candidate_reports.append(
            {
                "candidate": candidate.as_dict(),
                "trainable_numel": 8 * 8 + 8 + 1,
                "global_curves": global_curves,
                "rank_reports": rank_reports,
                "recovery": recovery,
            }
        )
    return {
        "schema": RELATION_BILINEAR_PROBE_SCHEMA,
        "status": "PASS",
        "arm": RELATION_BILINEAR_PROBE_ARM,
        "subject_sha256": relation_bilinear_probe_subject(
            provenance,
            curve_point_count=RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT,
        ),
        "provenance": provenance,
        "policy_parameter_boundary": _frozen_readout_scope_dict(8),
        "candidate_initialization_sha256": "1" * 64,
        "curve_point_count": RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT,
        "optimizer_update_count": RELATION_BILINEAR_PROBE_UPDATE_COUNT,
        "candidates": candidate_reports,
        "mode_decisions": relation_bilinear_decisions(candidate_reports),
        "maximum_peak_reserved_bytes": 1,
        "total_time_s": 1.0,
    }


def test_bilinear_report_recomputes_aggregates_and_rejects_tampering(
    monkeypatch,
) -> None:
    report = _bilinear_report()
    monkeypatch.setattr(
        bilinear_probe,
        "RELATION_BILINEAR_C_CONTROL_IDENTITY_SHA256",
        relation_bilinear_control_identity_sha256(report["provenance"]),
    )
    assert validate_relation_bilinear_probe_report(report)["status"] == "PASS"
    assert all(item["recoverable"] for item in report["mode_decisions"])

    tampered = deepcopy(report)
    tampered["candidates"][0]["global_curves"]["ownership"][-1] += 0.1
    with pytest.raises(ValueError, match="rank mean"):
        validate_relation_bilinear_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["candidates"].pop()
    with pytest.raises(ValueError, match="candidate grid"):
        validate_relation_bilinear_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["provenance"]["probe_seed_by_rank"][1] += 1
    with pytest.raises(ValueError, match="rank invariant"):
        validate_relation_bilinear_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["provenance"]["dataset_manifest_sha256"] = "9" * 64
    tampered["subject_sha256"] = relation_bilinear_probe_subject(
        tampered["provenance"],
        curve_point_count=RELATION_BILINEAR_PROBE_CURVE_POINT_COUNT,
    )
    with pytest.raises(ValueError, match="frozen C control identity"):
        validate_relation_bilinear_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["candidates"][0]["candidate"]["mode"] = "unconstrained"
    with pytest.raises(ValueError, match="descriptor"):
        validate_relation_bilinear_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["mode_decisions"][0]["recoverable"] = False
    with pytest.raises(ValueError, match="mode decision"):
        validate_relation_bilinear_probe_report(tampered)

    tampered = deepcopy(report)
    tampered["candidates"][0]["rank_reports"][0]["visual_artifacts_by_point"][0]["artifacts"][0][
        "path"
    ] = "../escaped.png"
    with pytest.raises(ValueError, match="visual artifact provenance"):
        validate_relation_bilinear_probe_report(tampered)
