from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.fixed_batch_probe import (
    PREDICTIVE_FIXED_BATCH_ARM_REPORT_SCHEMA,
    FixedBatchTrainableScope,
    ShuffledCurrentGridTargetCache,
    assemble_predictive_fixed_batch_experiment,
    configure_fixed_batch_trainable_scope,
    fixed_batch_probe_subject,
    shuffled_predictive_target,
    validate_predictive_fixed_batch_arm_report,
    validate_predictive_fixed_batch_experiment_report,
    verify_fixed_batch_trainable_scope,
)
from picf_next.lingbot_native.host import LingBotNativeGraph, LingBotNativeGraphConfig
from picf_next.lingbot_native.prediction import PredictionEvidence, PredictionSource
from picf_next.lingbot_native.predictive_objective import (
    TargetEncoderMode,
    make_native_predictive_target,
)
from picf_next.lingbot_native.predictive_probes import PREDICTIVE_FIXED_BATCH_ARMS
from tools.assemble_lingbot_predictive_fixed_batch_fit import (
    _supported_regular_link_count,
    build_experiment,
)
from tools.bootstrap_lingbot_vla2 import LINGBOT_CHECKPOINT_REVISION
from tools.bootstrap_lingbot_vla2_native import LINGBOT_NATIVE_SOURCE_COMMIT


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _graph() -> LingBotNativeGraph:
    return LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=3,
            host_width=8,
            executed_action_dim=2,
            num_layers=3,
            prediction_address_width=2,
            predictive_target_widths=(("dino_video", 4),),
        )
    )


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.host = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.picf_native_graph = _graph()


def _target() -> object:
    return make_native_predictive_target(
        modality="dino_video",
        features=torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 2.0]], [[3.0, 4.0]]]],
        ),
        valid=torch.ones(1, 3, 1, dtype=torch.bool),
        importance=torch.tensor([[[1.0], [0.5], [0.25]]]),
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.zeros(1, 1, dtype=torch.long),
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
        encoder_mode=TargetEncoderMode.FROZEN,
        source_batch_digest=_sha("source"),
        target_data_digest=_sha("target"),
        encoder_digest=_sha("encoder"),
        query_schema_digest=_sha("query"),
        validity_semantics="visible object summaries",
        track_identity_keys=(("red_block", "blue_block", "button"),),
    )


def test_readout_only_scope_freezes_every_non_interface_parameter() -> None:
    full_policy = _Policy()
    full = configure_fixed_batch_trainable_scope(
        full_policy,
        full_policy.picf_native_graph,
        arm="full_host",
    )
    readout_policy = _Policy()
    readout = configure_fixed_batch_trainable_scope(
        readout_policy,
        readout_policy.picf_native_graph,
        arm="readout_only",
    )
    graph_policy = _Policy()
    native_graph = configure_fixed_batch_trainable_scope(
        graph_policy,
        graph_policy.picf_native_graph,
        arm="native_graph_only",
    )

    assert full.trainable_numel > native_graph.trainable_numel > readout.trainable_numel
    assert all(parameter.requires_grad for parameter in full_policy.parameters())
    assert not any(parameter.requires_grad for parameter in readout_policy.host.parameters())
    assert not any(parameter.requires_grad for parameter in graph_policy.host.parameters())
    assert all(parameter.requires_grad for parameter in graph_policy.picf_native_graph.parameters())
    assert native_graph.component_names == ("picf_native_graph",)
    assert readout.component_names == (
        "prediction_role",
        "prediction_route_embeddings",
        "prediction_horizon_projection",
        "prediction_address_projection",
        "predictive_readouts",
    )
    assert all(
        any(component in name for component in readout.component_names)
        for name in readout.parameter_names
    )
    assert (
        verify_fixed_batch_trainable_scope(
            readout_policy,
            readout_policy.picf_native_graph,
            expected=readout,
        )
        == readout
    )
    assert (
        verify_fixed_batch_trainable_scope(
            graph_policy,
            graph_policy.picf_native_graph,
            expected=native_graph,
        )
        == native_graph
    )
    readout_policy.host[0].weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="distributed wrapping changed"):
        verify_fixed_batch_trainable_scope(
            readout_policy,
            readout_policy.picf_native_graph,
            expected=readout,
        )


def test_shuffled_target_changes_only_supervised_feature_association() -> None:
    target = _target()
    factual, factual_distance = shuffled_predictive_target(target, curve_index=0)
    shuffled, distance = shuffled_predictive_target(target, curve_index=1)
    reshuffled, reshuffled_distance = shuffled_predictive_target(target, curve_index=2)

    torch.testing.assert_close(factual.features, target.features)
    assert factual_distance == 0
    assert distance > 0
    assert reshuffled_distance > 0
    assert not torch.equal(shuffled.features, target.features)
    assert not torch.equal(reshuffled.features, target.features)
    assert not torch.equal(reshuffled.features, shuffled.features)
    assert torch.equal(shuffled.valid, target.valid)
    assert torch.equal(shuffled.importance, target.importance)
    assert torch.equal(shuffled.route_ids, target.route_ids)
    assert torch.equal(shuffled.horizons, target.horizons)
    assert shuffled.track_identity_keys == target.track_identity_keys
    assert shuffled.target_data_digest != target.target_data_digest


def test_shuffled_cache_records_each_step_without_exposing_other_cache_methods() -> None:
    target = _target()

    class Cache:
        contract = object()

        def current_correction_summary_target_for(self, **_kwargs: object) -> object:
            return target

    cache = ShuffledCurrentGridTargetCache(Cache())
    cache.begin_curve_point(1)
    returned = cache.current_correction_summary_target_for()

    assert returned.track_identity_keys == target.track_identity_keys
    assert cache.maximum_distance > 0
    assert not hasattr(cache, "target_for")


def _scope(arm: str) -> FixedBatchTrainableScope:
    readout = arm == "readout_only"
    readout_descriptors = (
        (
            "picf_native_graph.prediction_address_projection.weight",
            (2, 8),
            "torch.float32",
            16,
        ),
        (
            "picf_native_graph.prediction_horizon_projection.weight",
            (8, 8),
            "torch.float32",
            64,
        ),
        ("picf_native_graph.prediction_role", (8,), "torch.float32", 8),
        ("picf_native_graph.prediction_route_embeddings", (1, 8), "torch.float32", 8),
        (
            "picf_native_graph.predictive_readouts.dino_video.weight",
            (4, 8),
            "torch.float32",
            32,
        ),
    )
    native_graph_descriptors = (
        ("picf_native_graph.object_queries", (3, 8), "torch.float32", 24),
        *readout_descriptors,
    )
    descriptors = (
        readout_descriptors
        if readout
        else native_graph_descriptors
        if arm == "native_graph_only"
        else (("host.weight", (3, 8), "torch.float32", 24), *native_graph_descriptors)
    )
    serialized = [
        {
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
            "numel": numel,
        }
        for name, shape, dtype, numel in descriptors
    ]
    return FixedBatchTrainableScope(
        arm=arm,
        parameter_count=len(descriptors),
        trainable_numel=sum(descriptor[3] for descriptor in descriptors),
        schema_sha256=hashlib.sha256(
            json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        component_names=(
            (
                "prediction_role",
                "prediction_route_embeddings",
                "prediction_horizon_projection",
                "prediction_address_projection",
                "predictive_readouts",
            )
            if readout
            else ("picf_native_graph",)
            if arm == "native_graph_only"
            else ()
        ),
        parameter_descriptors=descriptors,
    )


def _shared_host_gradient_probe(arm: str) -> dict[str, object] | None:
    if arm not in {"full_host", "shuffled_target"}:
        return None
    return {
        "all_finite": True,
        "gradient_elements": {"early": 2560, "middle": 2560, "late": 2560},
        "gradient_norms": {"early": 0.3, "middle": 0.2, "late": 0.1},
        "parameter_paths": {
            "early": "model.host.language_model.layers.0.input_layernorm.weight",
            "middle": "model.host.language_model.layers.18.input_layernorm.weight",
            "late": "model.host.language_model.layers.35.input_layernorm.weight",
        },
        "probe": "lingbot.language_model.input_layernorm",
        "world_size": 2,
    }


def _provenance() -> dict[str, object]:
    return {
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "patch_sha256": _sha("patch"),
        "execution_contract_sha256": _sha("execution"),
        "implementation_sha256": _sha("implementation"),
        "model_family_sha256": _sha("model"),
        "plan_sha256": _sha("plan"),
        "dataset_manifest_sha256": _sha("dataset"),
        "physical_sidecar_manifest_sha256": _sha("sidecar"),
        "predictive_cache_manifest_sha256": _sha("predictive"),
        "current_grid_cache_manifest_sha256": _sha("current"),
        "seed": 7,
        "fixed_sample_global_step": 0,
        "frame_sample_keys_by_rank": [
            ["rank0-frame0", "rank0-frame1"],
            ["rank1-frame0", "rank1-frame1"],
        ],
        "frame_source_digests_by_rank": [
            [_sha("rank0-frame0-source"), _sha("rank0-frame1-source")],
            [_sha("rank1-frame0-source"), _sha("rank1-frame1-source")],
        ],
        "objective": {
            "optimized_family": "predictive",
            "target": "prior_to_current_object_summary",
            "window": "fixed_two_frame_local_bptt",
            "labels_are_loss_side_only": True,
        },
        "optimizer": {
            "algorithm": "lingbot_distributed_muon_with_adamw_fallback",
            "learning_rate_hex": (1e-4).hex(),
            "weight_decay_hex": (0.0).hex(),
            "scheduler": "constant",
            "moe_load_balance_hook_enabled": False,
            "update_count": 2,
        },
    }


def _arm_report(arm: str) -> dict[str, object]:
    provenance = _provenance()
    curves = {
        "full_host": [1.0, 0.6, 0.3],
        "native_graph_only": [1.0, 0.7, 0.5],
        "readout_only": [1.0, 0.8, 0.7],
        "shuffled_target": [1.0, 0.95, 0.9],
    }
    shuffle = [0.0, 0.2, 0.1] if arm == "shuffled_target" else [0.0, 0.0, 0.0]
    rank_reports = [
        {
            "rank": rank,
            "frame_sample_keys": list(provenance["frame_sample_keys_by_rank"][rank]),
            "frame_source_digests": list(provenance["frame_source_digests_by_rank"][rank]),
            "loss_curve": list(curves[arm]),
            "shuffle_distance_curve": list(shuffle),
            "step_times_s": [1.0, 1.1, 1.2],
            "peak_reserved_bytes": 100 + rank,
        }
        for rank in range(2)
    ]
    return {
        "schema": PREDICTIVE_FIXED_BATCH_ARM_REPORT_SCHEMA,
        "status": "PASS",
        "arm": arm,
        "subject_sha256": fixed_batch_probe_subject(
            provenance,
            curve_point_count=3,
        ),
        "provenance": provenance,
        "trainable_scope": _scope(arm).as_dict(),
        "curve_point_count": 3,
        "optimizer_update_count": 2,
        "global_loss_curve": list(curves[arm]),
        "global_shuffle_distance_curve": list(shuffle),
        "rank_reports": rank_reports,
        "shared_host_gradient_probe": _shared_host_gradient_probe(arm),
        "moe_routing_bias_unchanged": True,
        "maximum_peak_reserved_bytes": 101,
        "total_time_s": 3.3,
    }


def test_arm_reports_and_four_arm_experiment_are_recomputed() -> None:
    reports = OrderedDict((arm, _arm_report(arm)) for arm in PREDICTIVE_FIXED_BATCH_ARMS)
    for report in reports.values():
        validate_predictive_fixed_batch_arm_report(report)
    result = assemble_predictive_fixed_batch_experiment(
        reports,
        report_sha256=OrderedDict((arm, _sha(f"{arm}-report")) for arm in reports),
    )

    parsed = validate_predictive_fixed_batch_experiment_report(result)
    assert parsed["scientific_acceptance"] == "UNDECIDED_REQUIRES_OWNER_REVIEW"
    assert parsed["diagnostics"]["full_host_final_advantage_over_native_graph"] == pytest.approx(
        0.2
    )
    assert parsed["diagnostics"]["full_host_final_advantage_over_readout"] == pytest.approx(0.4)


def test_fixed_batch_reports_fail_closed_on_tampering() -> None:
    report = _arm_report("shuffled_target")
    edited = copy.deepcopy(report)
    edited["global_shuffle_distance_curve"][2] = 0.0
    edited["rank_reports"][0]["shuffle_distance_curve"][2] = 0.0
    edited["rank_reports"][1]["shuffle_distance_curve"][2] = 0.0
    with pytest.raises(ValueError, match="every post-initial"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(report)
    edited["rank_reports"][0]["shuffle_distance_curve"][2] = 0.0
    edited["rank_reports"][1]["shuffle_distance_curve"][2] = 0.2
    with pytest.raises(ValueError, match="both ranks"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(report)
    edited["global_shuffle_distance_curve"][1] = 0.0
    with pytest.raises(ValueError, match="rank mean"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(report)
    edited["optimizer_update_count"] = 3
    with pytest.raises(ValueError, match="curve points minus one"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(report)
    edited["provenance"]["optimizer"]["update_count"] = 1
    with pytest.raises(ValueError, match="optimizer contract"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(_arm_report("full_host"))
    edited["shared_host_gradient_probe"]["gradient_norms"]["middle"] = 0.0
    with pytest.raises(ValueError, match="every shared-host depth"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(_arm_report("native_graph_only"))
    edited["shared_host_gradient_probe"] = _shared_host_gradient_probe("full_host")
    with pytest.raises(ValueError, match="frozen-host"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(_arm_report("full_host"))
    edited["trainable_scope"]["parameters"][0]["dtype"] = "torch.float16"
    with pytest.raises(ValueError, match="schema digest"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(_arm_report("full_host"))
    edited["rank_reports"][1]["frame_source_digests"][1] = _sha("wrong-continuation")
    with pytest.raises(ValueError, match="rank data"):
        validate_predictive_fixed_batch_arm_report(edited)

    edited = copy.deepcopy(_arm_report("readout_only"))
    edited["trainable_scope"]["parameters"][0]["name"] = "host.unrelated.weight"
    edited["trainable_scope"]["schema_sha256"] = hashlib.sha256(
        json.dumps(
            edited["trainable_scope"]["parameters"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="prediction-interface"):
        validate_predictive_fixed_batch_arm_report(edited)

    reports = OrderedDict((arm, _arm_report(arm)) for arm in PREDICTIVE_FIXED_BATCH_ARMS)
    reports["readout_only"]["provenance"]["seed"] = 8
    reports["readout_only"]["subject_sha256"] = fixed_batch_probe_subject(
        reports["readout_only"]["provenance"],
        curve_point_count=3,
    )
    with pytest.raises(ValueError, match="one experiment subject"):
        assemble_predictive_fixed_batch_experiment(
            reports,
            report_sha256=OrderedDict((arm, _sha(f"{arm}-report")) for arm in reports),
        )

    reports = OrderedDict((arm, _arm_report(arm)) for arm in PREDICTIVE_FIXED_BATCH_ARMS)
    shuffled_parameters = reports["shuffled_target"]["trainable_scope"]["parameters"]
    shuffled_parameters[0]["dtype"] = "torch.bfloat16"
    reports["shuffled_target"]["trainable_scope"]["schema_sha256"] = hashlib.sha256(
        json.dumps(
            shuffled_parameters,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="trainability differ"):
        assemble_predictive_fixed_batch_experiment(
            reports,
            report_sha256=OrderedDict((arm, _sha(f"{arm}-report")) for arm in reports),
        )

    reports = OrderedDict((arm, _arm_report(arm)) for arm in PREDICTIVE_FIXED_BATCH_ARMS)
    reports["readout_only"]["global_loss_curve"][0] = 1.1
    for rank_report in reports["readout_only"]["rank_reports"]:
        rank_report["loss_curve"][0] = 1.1
    with pytest.raises(ValueError, match="initial model/loss"):
        assemble_predictive_fixed_batch_experiment(
            reports,
            report_sha256=OrderedDict((arm, _sha(f"{arm}-report")) for arm in reports),
        )

    reports = OrderedDict((arm, _arm_report(arm)) for arm in PREDICTIVE_FIXED_BATCH_ARMS)
    for arm in ("full_host", "shuffled_target"):
        parameters = reports[arm]["trainable_scope"]["parameters"]
        parameters[1]["name"] = "picf_native_graph.object_queries_decoy"
        reports[arm]["trainable_scope"]["schema_sha256"] = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    with pytest.raises(ValueError, match="exact strict subset"):
        assemble_predictive_fixed_batch_experiment(
            reports,
            report_sha256=OrderedDict((arm, _sha(f"{arm}-report")) for arm in reports),
        )


def test_fixed_batch_assembler_binds_exact_arm_files(tmp_path: Path) -> None:
    paths: OrderedDict[str, Path] = OrderedDict()
    digests: OrderedDict[str, str] = OrderedDict()
    for arm in PREDICTIVE_FIXED_BATCH_ARMS:
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(_arm_report(arm), sort_keys=True), encoding="ascii")
        paths[arm] = path
        digests[arm] = hashlib.sha256(path.read_bytes()).hexdigest()

    result = build_experiment(paths, report_sha256=digests)
    assert result["status"] == "PASS"
    digests["readout_only"] = _sha("tampered")
    with pytest.raises(ValueError, match="expected digest"):
        build_experiment(paths, report_sha256=digests)

    digests["readout_only"] = hashlib.sha256(paths["readout_only"].read_bytes()).hexdigest()
    external = tmp_path / "external-readout.json"
    paths["readout_only"].replace(external)
    paths["readout_only"].symlink_to(external)
    with pytest.raises(ValueError, match="readable regular file"):
        build_experiment(paths, report_sha256=digests)


def test_fixed_batch_assembler_accepts_persistent_fuse_zero_link_semantics() -> None:
    regular_mode = 0o100444
    directory_mode = 0o040755

    assert _supported_regular_link_count(mode=regular_mode, link_count=0)
    assert _supported_regular_link_count(mode=regular_mode, link_count=1)
    assert not _supported_regular_link_count(mode=regular_mode, link_count=2)
    assert not _supported_regular_link_count(mode=directory_mode, link_count=0)
