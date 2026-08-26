from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from picf_next.lingbot_native.full_modal_adoption import (
    ACTION_ADOPTION_CORE_SCHEMA,
    ACTION_ADOPTION_INTERVENTIONS_SCHEMA,
    ACTION_ADOPTION_PRESENCE_SCHEMA,
    ACTION_DCP_PHASE_SCHEMA,
    DENSE_MODALITIES,
    DENSE_PRESENCE_SUBSETS,
    action_adoption_metric_fragments,
    action_projection_drift_report,
    aggregate_rank_action_outputs,
    aggregate_rank_state_digests,
    capture_action_projection_output,
    captured_action_outputs_sha256,
    compose_action_adoption_core,
    compose_full_modal_action_adoption,
    dense_presence_code,
    dense_presence_name,
    directory_tree_sha256,
    distributed_action_adoption_gradients,
    distributed_maximum_action_drift,
    intervene_modality,
    make_action_adoption_interventions_report,
    make_action_adoption_presence_report,
    make_action_dcp_phase_report,
    parameter_group_names,
    resolve_action_adoption_parameter_groups,
    single_captured_action_output,
    with_dense_presence,
)
from picf_next.lingbot_native.modalities import NativeModalityBatch, NativeModalityStream


def _batch() -> NativeModalityBatch:
    streams = []
    for index, name in enumerate(("anytouch", "proprioception", "sonata", "vjepa")):
        token_count = 1 if name == "proprioception" else 3
        tokens = torch.arange(
            1 + index * 20,
            1 + index * 20 + 2 * token_count * 2,
            dtype=torch.float32,
        ).reshape(2, token_count, 2)
        valid = torch.ones(2, token_count, dtype=torch.bool)
        metadata = None
        if name != "proprioception":
            metadata = torch.arange(
                1 + index * 40,
                1 + index * 40 + 2 * token_count * 3,
                dtype=torch.float32,
            ).reshape(2, token_count, 3)
        canonical_token_ids = (
            None if name == "proprioception" else torch.arange(token_count).expand(2, -1).clone()
        )
        streams.append(
            NativeModalityStream(
                name,
                tokens,
                valid,
                metadata,
                canonical_token_ids,
            )
        )
    return NativeModalityBatch(tuple(streams))


def test_all_dense_presence_subsets_are_exact_and_keep_proprioception() -> None:
    source = _batch()
    assert tuple(dense_presence_name(value) for value in DENSE_PRESENCE_SUBSETS) == (
        "none",
        "anytouch",
        "sonata",
        "vjepa",
        "anytouch+sonata",
        "anytouch+vjepa",
        "sonata+vjepa",
        "anytouch+sonata+vjepa",
    )
    assert tuple(dense_presence_code(value) for value in DENSE_PRESENCE_SUBSETS) == (
        "none",
        "A",
        "S",
        "V",
        "AS",
        "AV",
        "SV",
        "ASV",
    )
    for present in DENSE_PRESENCE_SUBSETS:
        observed = with_dense_presence(source, present)
        counts = {stream.name: stream.token_count for stream in observed.streams}
        assert counts["proprioception"] == 1
        for modality in ("anytouch", "sonata", "vjepa"):
            assert counts[modality] == (3 if modality in present else 0)


@pytest.mark.parametrize(
    "intervention",
    ("value_zero", "metadata_zero", "value_permutation", "joint_permutation"),
)
def test_modality_interventions_change_only_the_requested_content(intervention: str) -> None:
    source = _batch()
    result = intervene_modality(
        source,
        modality="sonata",
        intervention=intervention,
    )
    observed = result.batch
    assert result.changed_elements > 0
    assert result.token_permutations == ((2, 0, 1), (2, 0, 1))
    assert result.valid_before == result.valid_after
    before = next(stream for stream in source.streams if stream.name == "sonata")
    after = next(stream for stream in observed.streams if stream.name == "sonata")
    assert torch.equal(before.valid, after.valid)
    assert before.canonical_token_ids is not None
    assert after.canonical_token_ids is not None
    if intervention == "value_zero":
        assert not after.tokens.any()
        assert torch.equal(before.metadata, after.metadata)
        assert torch.equal(before.canonical_token_ids, after.canonical_token_ids)
    elif intervention == "metadata_zero":
        assert torch.equal(before.tokens, after.tokens)
        assert not after.metadata.any()
        assert torch.equal(before.canonical_token_ids, after.canonical_token_ids)
    elif intervention == "value_permutation":
        assert not torch.equal(before.tokens, after.tokens)
        assert torch.equal(before.metadata, after.metadata)
        assert torch.equal(before.canonical_token_ids, after.canonical_token_ids)
    else:
        assert not torch.equal(before.tokens, after.tokens)
        assert not torch.equal(before.metadata, after.metadata)
        assert not torch.equal(before.canonical_token_ids, after.canonical_token_ids)
        assert torch.equal(
            before.tokens.sort(dim=1).values,
            after.tokens.sort(dim=1).values,
        )
    for before_other, after_other in zip(source.streams, observed.streams, strict=True):
        if before_other.name != "sonata":
            assert before_other is after_other


@pytest.mark.parametrize(
    "intervention",
    ("value_zero", "metadata_zero", "value_permutation", "joint_permutation"),
)
def test_modality_intervention_rejects_an_unchanged_or_unavailable_arm(
    intervention: str,
) -> None:
    source = _batch()
    streams = tuple(
        NativeModalityStream(
            stream.name,
            stream.tokens[:, :0] if stream.name == "anytouch" else stream.tokens,
            stream.valid[:, :0] if stream.name == "anytouch" else stream.valid,
            stream.metadata[:, :0]
            if stream.name == "anytouch" and stream.metadata is not None
            else stream.metadata,
            stream.canonical_token_ids[:, :0]
            if stream.name == "anytouch" and stream.canonical_token_ids is not None
            else stream.canonical_token_ids,
        )
        for stream in source.streams
    )
    with pytest.raises(ValueError, match="valid source value"):
        intervene_modality(
            NativeModalityBatch(streams),
            modality="anytouch",
            intervention=intervention,
        )
    empty_batch = NativeModalityBatch(streams)
    observed = intervene_modality(
        empty_batch,
        modality="anytouch",
        intervention=intervention,
        require_change=False,
    )
    assert observed.batch is empty_batch
    assert observed.changed_elements == 0
    assert observed.token_permutations == ((), ())


def _named_parameters() -> tuple[tuple[str, torch.nn.Parameter], ...]:
    values: list[tuple[str, torch.nn.Parameter]] = []
    prefix = "model.qwenvl_with_expert."
    for modality in ("anytouch", "sonata", "vjepa"):
        values.append(
            (
                prefix + f"picf_native_graph.modality_projections.{modality}.weight",
                torch.nn.Parameter(torch.ones(2, 2)),
            )
        )
        values.append(
            (
                prefix + f"picf_native_graph.modality_metadata_projections.{modality}.weight",
                torch.nn.Parameter(torch.ones(2, 2)),
            )
        )
    for layer in (0, 18, 35):
        values.append(
            (
                prefix + f"qwenvl.model.language_model.layers.{layer}.input_layernorm.weight",
                torch.nn.Parameter(torch.ones(2)),
            )
        )
    values.extend(
        (
            (
                prefix + "qwen_expert.layers.0.self_attn.q_proj.weight",
                torch.nn.Parameter(torch.ones(2, 2)),
            ),
            (
                prefix + "qwen_expert.layers.1.mlp.gate_proj.weight",
                torch.nn.Parameter(torch.ones(2, 2)),
            ),
            (prefix + "action_out_proj.weight", torch.nn.Parameter(torch.ones(2, 2))),
        )
    )
    return tuple(values)


def test_parameter_groups_resolve_exact_disjoint_production_paths() -> None:
    groups = resolve_action_adoption_parameter_groups(_named_parameters())
    serialized = parameter_group_names(groups)
    assert tuple(serialized) == (
        "anytouch_value_adapter",
        "anytouch_metadata_adapter",
        "sonata_value_adapter",
        "sonata_metadata_adapter",
        "vjepa_value_adapter",
        "vjepa_metadata_adapter",
        "host_layer_0",
        "host_layer_18",
        "host_layer_35",
        "action_expert",
        "action_output",
    )
    assert len(serialized["action_expert"]) == 2
    fragments = dict(action_adoption_metric_fragments(groups))
    assert fragments["action_expert"] == ".qwen_expert."
    assert fragments["anytouch_value_adapter"] == serialized["anytouch_value_adapter"][0]


def test_parameter_groups_fail_closed_on_missing_or_ambiguous_paths() -> None:
    values = _named_parameters()
    with pytest.raises(ValueError, match="resolved 0 parameters"):
        resolve_action_adoption_parameter_groups(values[:-1])
    duplicate = (*values, ("other." + values[0][0], torch.nn.Parameter(torch.ones(2, 2))))
    with pytest.raises(ValueError, match="resolved 2 parameters"):
        resolve_action_adoption_parameter_groups(duplicate)


class _SingleRankDist:
    class ReduceOp:
        SUM = "sum"
        MIN = "min"
        MAX = "max"

    @staticmethod
    def all_reduce(_value: torch.Tensor, *, op: str) -> None:
        assert op in {"sum", "min", "max"}

    @staticmethod
    def get_world_size() -> int:
        return 1

    @staticmethod
    def all_gather_object(output: list[object], value: object) -> None:
        output[0] = value


def test_distributed_action_gradients_distinguish_absent_and_zero_paths() -> None:
    present = torch.nn.Parameter(torch.ones(2, 2))
    present.grad = torch.full_like(present, 2.0)
    zero = torch.nn.Parameter(torch.ones(3))
    zero.grad = torch.zeros_like(zero)
    absent = torch.nn.Parameter(torch.ones(1))
    from picf_next.lingbot_native.full_modal_adoption import ActionAdoptionParameterGroup

    groups = (
        ActionAdoptionParameterGroup("present", ("present",), (present,)),
        ActionAdoptionParameterGroup("zero", ("zero",), (zero,)),
        ActionAdoptionParameterGroup("absent", ("absent",), (absent,)),
    )
    observed = distributed_action_adoption_gradients(
        groups,
        device=torch.device("cpu"),
        dist=_SingleRankDist(),
    )
    assert observed["present"] == {"norm": 4.0, "elements": 4}
    assert observed["zero"] == {"norm": 0.0, "elements": 3}
    assert observed["absent"] == {"norm": None, "elements": 0}


def test_action_projection_capture_observes_exactly_one_suffix_call() -> None:
    model = torch.nn.Module()
    model.action_out_proj = torch.nn.Linear(2, 3, bias=False)
    with capture_action_projection_output(model) as captured:
        expected = model.action_out_proj(torch.ones(1, 2))
    assert torch.equal(single_captured_action_output(captured), expected)
    assert captured[0].device.type == "cpu"
    assert captured[0].dtype == torch.float32
    with pytest.raises(RuntimeError, match="captured 0"):
        single_captured_action_output([])


def test_action_projection_capture_sees_a_hook_added_after_torch_compile() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_out_proj = torch.nn.Linear(2, 3, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.action_out_proj(value)

    compiled = torch.compile(Model(), backend="eager")
    value = torch.ones(1, 2)
    compiled(value)

    with capture_action_projection_output(compiled) as captured:
        expected = compiled(value)

    assert torch.equal(single_captured_action_output(captured), expected)


def test_action_projection_capture_localizes_distributed_output_in_the_hook() -> None:
    class LocalizableTensor(torch.Tensor):
        localization_calls = 0

        @staticmethod
        def __new__(cls, value: torch.Tensor) -> LocalizableTensor:
            return torch.Tensor._make_subclass(cls, value, require_grad=value.requires_grad)

        def to_local(self) -> torch.Tensor:
            type(self).localization_calls += 1
            return self.as_subclass(torch.Tensor)

    class Projection(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return LocalizableTensor(value)

    model = torch.nn.Module()
    model.action_out_proj = Projection()
    with capture_action_projection_output(model) as captured:
        model.action_out_proj(torch.ones(1, 2))
    observed = single_captured_action_output(captured)
    assert type(observed) is torch.Tensor
    assert LocalizableTensor.localization_calls == 1


def test_action_projection_drift_report_preserves_sub_loss_tensor_evidence() -> None:
    reference = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    candidate = torch.tensor([[1.0, -1.999, 3.002]], dtype=torch.float32)
    report = action_projection_drift_report(reference, candidate)
    assert report["shape"] == [1, 3]
    assert report["element_count"] == 3
    assert report["nonzero_count"] == 2
    assert report["nonzero_fraction"] == pytest.approx(2 / 3)
    assert report["max_abs"] == pytest.approx(0.002, abs=1e-6)
    assert report["rms"] == pytest.approx(
        ((0.001**2 + 0.002**2) / 3) ** 0.5,
        abs=1e-6,
    )
    assert report["reference_sha256"] != report["candidate_sha256"]


def test_action_projection_drift_report_rejects_invalid_comparisons() -> None:
    with pytest.raises(ValueError, match="different shapes"):
        action_projection_drift_report(torch.ones(1), torch.ones(2))
    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        action_projection_drift_report(torch.ones(1), torch.tensor([float("nan")]))


def test_action_drift_localizes_shards_before_the_single_explicit_reduction() -> None:
    class LocalShard:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value
            self.calls = 0

        def to_local(self) -> torch.Tensor:
            self.calls += 1
            return self.value

    left = LocalShard(torch.tensor([[1.0, 2.0]]))
    right = LocalShard(torch.tensor([[1.0, 2.25]]))
    measured = distributed_maximum_action_drift(
        left,
        right,
        dist=_SingleRankDist(),
    )
    assert measured == 0.25
    assert left.calls == right.calls == 1


def test_action_drift_uses_the_maximum_gathered_rank_value() -> None:
    class GatherDist:
        @staticmethod
        def get_world_size() -> int:
            return 3

        @staticmethod
        def all_gather_object(output: list[object], value: object) -> None:
            output[:] = [value, 0.75, 0.1]

    measured = distributed_maximum_action_drift(
        torch.tensor([1.0]),
        torch.tensor([1.25]),
        dist=GatherDist(),
    )
    assert measured == 0.75


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _rank_boundaries(prefix: str) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "boundary": {
                field: _digest(f"{prefix}-{rank}-{field}")
                for field in (
                    "model_local_state_sha256",
                    "optimizer_local_state_sha256",
                    "lane_snapshot_sha256",
                    "rank_rng_state_sha256",
                )
            },
        }
        for rank in range(2)
    ]


def test_distributed_state_and_action_digests_bind_rank_order() -> None:
    boundaries = _rank_boundaries("saved")
    observed = aggregate_rank_state_digests(boundaries)
    assert set(observed) == {
        "model_sha256",
        "optimizer_sha256",
        "lane_sha256",
        "rng_sha256",
    }
    assert len(set(observed.values())) == 4
    action = aggregate_rank_action_outputs(
        [
            {"rank": 0, "action_output_sha256": _digest("action-0")},
            {"rank": 1, "action_output_sha256": _digest("action-1")},
        ]
    )
    assert len(action) == 64
    reversed_boundaries = list(reversed(boundaries))
    with pytest.raises(ValueError, match="ordered and contiguous"):
        aggregate_rank_state_digests(reversed_boundaries)


def test_distributed_state_digest_rejects_report_aliases_at_checkpoint_boundary() -> None:
    boundaries = [
        {
            "rank": 0,
            "boundary": {
                field: _digest(field)
                for field in ("model_sha256", "optimizer_sha256", "lane_sha256", "rng_sha256")
            },
        }
    ]
    with pytest.raises(ValueError, match="boundary fields differ"):
        aggregate_rank_state_digests(boundaries)


def test_action_projection_sequence_digest_is_order_and_value_sensitive() -> None:
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[3.0, 4.0]])
    factual = captured_action_outputs_sha256([first, second])
    assert factual == captured_action_outputs_sha256([first.clone(), second.clone()])
    assert factual != captured_action_outputs_sha256([second, first])
    assert factual != captured_action_outputs_sha256([first, second + 1])
    with pytest.raises(ValueError, match="at least one"):
        captured_action_outputs_sha256([])


def test_checkpoint_tree_digest_rejects_symlinks_and_changes_with_bytes(tmp_path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "metadata.json").write_text("one", encoding="utf-8")
    first = directory_tree_sha256(root)
    assert first == directory_tree_sha256(root)
    (root / "metadata.json").write_text("two", encoding="utf-8")
    assert first != directory_tree_sha256(root)
    (root / "indirect").symlink_to(root / "metadata.json")
    with pytest.raises(ValueError, match="symbolic link"):
        directory_tree_sha256(root)


def _core_report() -> dict[str, object]:
    value: dict[str, object] = {
        "schema": ACTION_ADOPTION_CORE_SCHEMA,
        "status": "PASS",
        "action_loss_only": True,
        "probe_optimizer_step": 17,
        "nonzero_gradient_min_norm": 1e-12,
        "presence_subsets": [{"name": "placeholder"}],
        "modality_interventions": [{"modality": "placeholder"}],
        "active_anytouch_sample_keys": ["touch"],
        "parameter_groups": {"action_output": ["action_out_proj.weight"]},
    }
    from picf_next.lingbot_native import full_modal_adoption as module

    value["artifact_sha256"] = module._canonical_sha256(value)
    return value


def test_fresh_process_action_adoption_phases_compose_only_matching_samples() -> None:
    sample_keys = ["sample-a", "sample-b"]
    presence = make_action_adoption_presence_report(
        probe_optimizer_step=17,
        nonzero_gradient_min_norm=1e-12,
        presence_subsets=[
            {"name": dense_presence_code(subset), "sample_keys": sample_keys}
            for subset in DENSE_PRESENCE_SUBSETS
        ],
        active_anytouch_sample_keys=["sample-a"],
        parameter_groups={"action_output": ["action_out_proj.weight"]},
    )
    interventions = make_action_adoption_interventions_report(
        probe_optimizer_step=17,
        modality_interventions=[
            {"modality": modality, "sample_keys": sample_keys} for modality in DENSE_MODALITIES
        ],
        active_anytouch_sample_keys=["sample-a"],
    )
    assert presence["schema"] == ACTION_ADOPTION_PRESENCE_SCHEMA
    assert interventions["schema"] == ACTION_ADOPTION_INTERVENTIONS_SCHEMA
    core = compose_action_adoption_core(
        presence=presence,
        interventions=interventions,
    )
    assert core["schema"] == ACTION_ADOPTION_CORE_SCHEMA
    assert core["probe_optimizer_step"] == 17

    mismatched = copy.deepcopy(interventions)
    mismatched["modality_interventions"][0]["sample_keys"] = ["sample-a", "sample-c"]
    from picf_next.lingbot_native import full_modal_adoption as module

    unsigned = {key: value for key, value in mismatched.items() if key != "artifact_sha256"}
    mismatched["artifact_sha256"] = module._canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="inconsistent global samples"):
        compose_action_adoption_core(
            presence=presence,
            interventions=mismatched,
        )


def _continuation(boundary: dict[str, str]) -> dict[str, object]:
    return {
        "global_step": 2,
        "model_sha256": _digest("next-model"),
        "optimizer_sha256": _digest("next-optimizer"),
        "lane_sha256": _digest("next-lane"),
        "rng_sha256": _digest("next-rng"),
        "action_output_sha256": _digest("next-action"),
        "action_loss": 0.25,
    }


def test_dcp_phase_reports_compose_only_exact_cold_continuations() -> None:
    boundary = aggregate_rank_state_digests(_rank_boundaries("saved"))
    continuation = _continuation(boundary)
    uninterrupted = make_action_dcp_phase_report(
        phase="uninterrupted",
        process_sha256=_digest("process-a"),
        checkpoint_artifact_sha256=_digest("checkpoint"),
        boundary=boundary,
        next_step=continuation,
    )
    restored = make_action_dcp_phase_report(
        phase="restored",
        process_sha256=_digest("process-b"),
        checkpoint_artifact_sha256=_digest("checkpoint"),
        boundary=boundary,
        next_step=continuation,
    )
    assert uninterrupted["schema"] == ACTION_DCP_PHASE_SCHEMA
    combined = compose_full_modal_action_adoption(
        core=_core_report(),
        uninterrupted=uninterrupted,
        restored=restored,
    )
    assert combined["status"] == "PASS"
    assert combined["dcp_cold_restore"]["saved_boundary"] == boundary

    changed = copy.deepcopy(restored)
    changed["next_step"]["action_loss"] = 0.3
    from picf_next.lingbot_native import full_modal_adoption as module

    unsigned = {key: value for key, value in changed.items() if key != "artifact_sha256"}
    changed["artifact_sha256"] = module._canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="next-step continuation"):
        compose_full_modal_action_adoption(
            core=_core_report(),
            uninterrupted=uninterrupted,
            restored=changed,
        )
