from __future__ import annotations

import json
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

import picf_next.lingbot_native.relation_bilinear_probe as bilinear_probe
import tools.run_lingbot_vla2_native_full as full_runner
from picf_next.lingbot_native.host import LingBotNativeGraph, LingBotNativeGraphConfig
from picf_next.lingbot_native.relation_bilinear_probe import (
    RELATION_BILINEAR_PROBE_ARM,
    FullRankBilinearRelationReadout,
    relation_bilinear_control_identity_sha256,
    validate_relation_bilinear_probe_report,
)
from picf_next.lingbot_native.relation_depth_probe import (
    RELATION_DEPTH_PROBE_ARM,
    relation_depth_surfaces,
    validate_relation_depth_probe_report,
)
from picf_next.lingbot_native.relation_geometry_probe import (
    RelationProbeSampleMetadata,
    RelationProbeSampleSelection,
    configure_relation_geometry_trainable_scope,
)


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.picf_native_graph = LingBotNativeGraph(
            LingBotNativeGraphConfig(
                capacity=3,
                host_width=4,
                executed_action_dim=2,
                num_layers=4,
                prediction_address_width=2,
                predictive_target_widths=(("dino_video", 4),),
            )
        )
        self.register_buffer("e_score_correction_bias", torch.tensor([0.2, -0.1]))
        self.register_buffer("tokens_per_expert", torch.tensor([3.0, 4.0]))


class _FakeCuda:
    @staticmethod
    def manual_seed_all(_seed: int) -> None:
        return None

    @staticmethod
    def reset_peak_memory_stats(_device: torch.device) -> None:
        return None

    @staticmethod
    def max_memory_reserved(_device: torch.device) -> int:
        return 1


class _TorchProxy:
    Tensor = torch.Tensor
    bfloat16 = torch.bfloat16
    bool = torch.bool
    cuda = _FakeCuda()
    float32 = torch.float32
    float64 = torch.float64
    nn = torch.nn

    def __getattr__(self, name: str) -> object:
        return getattr(torch, name)


class _Dist:
    class ReduceOp:
        SUM = "sum"

    @staticmethod
    def all_reduce(value: torch.Tensor, *, op: str) -> None:
        assert op == _Dist.ReduceOp.SUM
        value.mul_(2)

    @staticmethod
    def all_gather_object(outputs: list[object], value: object) -> None:
        outputs[0] = value
        if isinstance(value, dict) and "candidate_reports" in value:
            peer = deepcopy(value)
            peer["rank"] = 1
            peer["forward_seed"] = int(peer["forward_seed"]) + 1
            peer["frame_sample_keys"] = ["rank1/current", "rank1/next"]
            peer["frame_source_digests"] = ["3" * 64, "4" * 64]
            for report in peer["candidate_reports"]:
                report["rank"] = 1
                for visual in report["visual_artifacts_by_point"]:
                    visual["artifacts"][0]["rank"] = 1
                    visual["artifacts"][0]["sample_key"] = "rank1/current"
            outputs[1] = peer
        else:
            outputs[1] = value

    @staticmethod
    def broadcast_object_list(_values: list[object], *, src: int) -> None:
        assert src == 0

    @staticmethod
    def barrier() -> None:
        return None


class _Capture:
    def __init__(self, _policy: nn.Module) -> None:
        self.surfaces = relation_depth_surfaces(4)

    def __enter__(self) -> _Capture:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None

    def snapshot(
        self,
        *,
        expected_forward_count: int,
    ) -> MappingProxyType[str, tuple[torch.Tensor, ...]]:
        assert expected_forward_count == 2
        return MappingProxyType(
            {
                surface.name: (
                    torch.zeros(1, 12, 4),
                    torch.ones(1, 12, 4),
                )
                for surface in self.surfaces
            }
        )


class _DepthInputs:
    def __init__(self, offset: float) -> None:
        generator = torch.Generator().manual_seed(int(offset) + 31)
        self.rows = torch.randn(1, 3, 4, generator=generator)
        self.sensors = torch.randn(1, 5, 4, generator=generator)
        self.valid = torch.ones(1, 5, dtype=torch.bool)
        self.match = torch.randn(1, 3, 4, generator=generator)
        self.instruction = torch.randn(1, 4, generator=generator)

    def read(self, readout: nn.Module) -> object:
        common = dict(
            posterior_rows=self.rows,
            sensor_hidden=self.sensors,
            sensor_valid=self.valid,
            structural_sensor_valid=self.valid,
        )
        if isinstance(readout, FullRankBilinearRelationReadout):
            return readout(**common, instruction_hidden=self.instruction)
        return readout(**common, match_hidden=self.match)


def _sample_selection() -> RelationProbeSampleSelection:
    samples = tuple(
        RelationProbeSampleMetadata(
            sample_key=f"rank{rank}/current",
            task_key="push_blue_block_left",
            available_future_transitions=1,
            target_identity_keys=("object/a",),
            inventory_identity_keys=("object/a",),
            target_supervised_pixel_counts=(10,),
        )
        for rank in range(2)
    )
    return RelationProbeSampleSelection(
        selection_start_global_step=0,
        selected_global_step=0,
        inspected_step_count=1,
        capacity=3,
        samples_by_rank=samples,
    )


@pytest.mark.parametrize(
    ("arm", "expected_candidate_count", "report_validator"),
    (
        (
            RELATION_DEPTH_PROBE_ARM,
            20,
            validate_relation_depth_probe_report,
        ),
        (
            RELATION_BILINEAR_PROBE_ARM,
            10,
            validate_relation_bilinear_probe_report,
        ),
    ),
)
def test_external_relation_runner_executes_full_grid_and_validates_report(
    tmp_path,
    monkeypatch,
    arm,
    expected_candidate_count,
    report_validator,
) -> None:
    policy = _Policy()
    graph = policy.picf_native_graph
    scope = configure_relation_geometry_trainable_scope(
        policy,
        graph,
        arm="existing_readout_frozen_host",
    )
    output = tmp_path / f"{arm}.json"
    visual_root = tmp_path / f"{arm}-visuals"
    args = SimpleNamespace(
        capacity=3,
        minimum_supervised_fraction=1.0,
        relation_geometry_fixed_batch_arm=arm,
        relation_geometry_fixed_batch_curve_points=41,
        relation_geometry_fixed_batch_output=output,
        relation_geometry_fixed_batch_sample_step=0,
        relation_geometry_fixed_batch_visual_root=visual_root,
        seed=7,
    )
    primary_plan = SimpleNamespace(
        continuation=False,
        training=SimpleNamespace(host_items=({"task": "push blue block"},)),
    )
    continuation_plan = SimpleNamespace(continuation=True)

    def batch(*, continuation: bool) -> SimpleNamespace:
        return SimpleNamespace(
            controls=SimpleNamespace(
                reset=torch.tensor([False]),
                token_valid=torch.tensor([True]),
            ),
            modalities=None,
            routing=SimpleNamespace(
                batch_size=1,
                sample_keys=("rank0/next" if continuation else "rank0/current",),
            ),
            source_digest=("2" if continuation else "1") * 64,
            model_inputs={},
            structural_target_requests=(),
        )

    def context_type(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            **kwargs,
            posterior_state=None,
            relation_output=None,
        )

    def run_policy(
        _policy: nn.Module,
        *,
        model_inputs: object,
        context: SimpleNamespace,
    ) -> SimpleNamespace:
        assert model_inputs == {}
        context.posterior_state = SimpleNamespace(rows=torch.zeros(1, 3, 4))
        context.relation_output = SimpleNamespace()
        return SimpleNamespace(
            context=context,
            official_total_loss=torch.tensor(0.2),
            official_action_loss=torch.tensor(0.1),
        )

    def run_observation(
        _policy: nn.Module,
        *,
        model_inputs: object,
        context: SimpleNamespace,
    ) -> SimpleNamespace:
        assert model_inputs == {}
        context.posterior_state = SimpleNamespace(rows=torch.zeros(1, 3, 4))
        context.relation_output = SimpleNamespace()
        return context

    def compose_objective(*, relations: tuple[object, ...], **_kwargs: object) -> object:
        first = relations[0]
        if hasattr(first, "ownership"):
            ownership = -first.ownership[..., 0].clamp_min(1e-6).log().mean()
            nll = ownership + first.support_logits.square().mean() * 0
        else:
            ownership = torch.tensor(1.0)
            nll = torch.tensor(1.0)
        return SimpleNamespace(
            objective=SimpleNamespace(
                normalized_terms={
                    "set/ownership": ownership,
                    "set/ownership_nll": nll,
                }
            ),
            row_bindings_by_batch=((("object/a", 0),),),
        )

    monkeypatch.setattr(full_runner, "LingBotRelationDepthCapture", _Capture)
    monkeypatch.setattr(
        full_runner,
        "relation_depth_inputs",
        lambda hidden, **_kwargs: _DepthInputs(float(hidden.sum().item())),
    )
    monkeypatch.setattr(
        full_runner,
        "build_task_row_diagnostics",
        lambda _objective: ({"target_rows": [0]},),
    )
    monkeypatch.setattr(
        full_runner,
        "validate_task_row_diagnostics",
        lambda value, *, expected_batch_size: value,
    )
    if arm == RELATION_BILINEAR_PROBE_ARM:
        original_validator = full_runner.validate_relation_bilinear_probe_report

        def validate_synthetic_bilinear_report(
            value: object,
        ) -> dict[str, object]:
            if not isinstance(value, dict) or not isinstance(
                value.get("provenance"),
                dict,
            ):
                raise TypeError("synthetic relation-bilinear report is malformed")
            monkeypatch.setattr(
                bilinear_probe,
                "RELATION_BILINEAR_C_CONTROL_IDENTITY_SHA256",
                relation_bilinear_control_identity_sha256(value["provenance"]),
            )
            return original_validator(value)

        monkeypatch.setattr(
            full_runner,
            "validate_relation_bilinear_probe_report",
            validate_synthetic_bilinear_report,
        )

    full_runner._run_external_relation_fixed_batch_arm(
        args=args,
        rank=0,
        device=torch.device("cpu"),
        dist=_Dist(),
        torch_module=_TorchProxy(),
        policy=policy,
        graph=graph,
        trainable_scope=scope,
        sample_selection=_sample_selection(),
        stream_plan=SimpleNamespace(plan_sha256="5" * 64),
        dataset=object(),
        collate_planned=lambda planned: batch(continuation=planned.continuation),
        build_planned_batch=lambda *_args, **_kwargs: primary_plan,
        build_continuation_batch=lambda *_args, **_kwargs: continuation_plan,
        context_type=context_type,
        run_policy_diagnostic=run_policy,
        run_observation_diagnostic=run_observation,
        compose_objective=compose_objective,
        physical_sidecar=SimpleNamespace(manifest_sha256="6" * 64),
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=14,
        merge_size=2,
        objective_config=object(),
        structural_config=object(),
        derive_subseed_fn=lambda *_args: 11,
        temporal_batch_seed_fn=lambda **_kwargs: 13,
        matched_row_soft_iou_fn=lambda **_kwargs: [0.1, None, None],
        render_relation_visuals=lambda **kwargs: [
            {
                "schema": "picf-next.lingbot-native-relation-visual.v5",
                "rank": kwargs["rank"],
                "global_step": kwargs["global_step"],
                "input_weight_global_step": kwargs["global_step"] - 1,
                "weight_boundary": "pre_update_forward",
                "path": "visuals/example.png",
                "sha256": "a" * 64,
                "bytes": 1,
                "batch_index": 0,
                "sample_key": "rank0/current",
                "task": "push blue block",
                "identity_keys": ["object/a"],
                "source_time": 0,
                "source_side": "posterior",
                "source_phase": 1,
                "binding_start_phase": [1, 2, 2],
                "source_binding_valid": [True, False, False],
                "row_to_track": [0, -1, -1],
                "sequence_row_to_track": [0, -1, -1],
                "row_existence": [0.8, 0.1, 0.1],
                "row_task_relevance": [0.8, 0.1, 0.1],
                "row_matched_soft_iou": [0.1, None, None],
                "anchor_surface": "task_object_probability.max(row)",
                "views": [{"name": "primary"}],
                "loss_only_labels_visible_to_model": False,
            }
        ],
        patch_sha256="7" * 64,
        execution_sha256="8" * 64,
        implementation_sha256="9" * 64,
        model_family_sha256="b" * 64,
        dataset_contract_report={"manifest_sha256": "c" * 64},
    )

    report = report_validator(json.loads(output.read_text()))
    assert report["curve_point_count"] == 41
    assert report["optimizer_update_count"] == 40
    assert len(report["candidates"]) == expected_candidate_count
    assert all(len(value["global_curves"]["ownership"]) == 41 for value in report["candidates"])
    assert torch.count_nonzero(policy.tokens_per_expert) == 0
