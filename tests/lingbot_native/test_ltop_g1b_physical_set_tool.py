from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import run_lingbot_vla2_ltop_g1b_physical_set as g1b

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/run_lingbot_vla2_ltop_g1b_physical_set.py"


def _distinct_rows() -> torch.Tensor:
    rows = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    rows[0, 0, 1, 0] = 1.0
    rows[0, 0, 2, 1] = 2.0
    rows[0, 1, 1, 2] = 3.0
    rows[0, 1, 2, 3] = 4.0
    return rows


def _rank_report(rank: int) -> dict[str, object]:
    surface = {
        "noncollapse": {
            "derivation": "pair_l2 > 2 * exact_replay_max_row_l2",
            "per_layer": [
                {
                    "replay_max_row_l2_floor": 0.0,
                    "identifiability_threshold_l2": 0.0,
                    "maximum_pair_l2": 1.0,
                    "stable_distinct_pair_count": 1,
                    "pair_count": 120,
                    "noncollapsed": True,
                }
            ],
            "all_layers_noncollapsed": True,
        },
        "permutation": {
            "derivation": "permuted-arm error <= exact-replay error in the same metric",
            "row_permutation": list(range(g1b.G1B_PHYSICAL_CAPACITY)),
            "per_layer": [
                {
                    "replay_max_abs_floor": 0.0,
                    "equivariance_max_abs_error": 0.0,
                    "replay_max_row_l2_floor": 0.0,
                    "equivariance_max_row_l2_error": 0.0,
                    "replay_pairwise_set_floor": 0.0,
                    "pairwise_set_error": 0.0,
                    "row_identity_permuted_only": True,
                }
            ],
            "all_layers_equivariant": True,
        },
        "prompt_invariance": {
            "derivation": "prompt-swap error <= exact-replay error in the same metric",
            "per_layer": [
                {
                    "replay_max_abs_floor": 0.0,
                    "prompt_max_abs_error": 0.0,
                    "replay_max_row_l2_floor": 0.0,
                    "prompt_max_row_l2_error": 0.0,
                    "prompt_invariant": True,
                }
            ],
            "all_layers_prompt_invariant": True,
        },
    }
    return {
        "rank": rank,
        "same_scene_non_language_exact": True,
        "prompt_changed": True,
        "address_permutation_consistent": True,
        "contexts_finalized": True,
        "rows_finite": True,
        "prior": copy.deepcopy(surface),
        "posterior": copy.deepcopy(surface),
    }


def _report() -> dict[str, object]:
    return {
        "schema": g1b.G1B_SCHEMA,
        "status": "PASS",
        "failures": [],
        "architecture_identity": g1b.G1B_ARCHITECTURE,
        "world_size": g1b.G1B_WORLD_SIZE,
        "parameter_manifest": {"active_trainable_numel": 0},
        "rank_reports": [_rank_report(0), _rank_report(1)],
    }


def test_g1b_delays_accelerator_and_upstream_imports() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_imports.update(
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "torch" not in top_imports
    assert "transformers" not in top_imports
    assert "lingbotvla" not in top_imports


def test_g1b_is_independent_observation_only_evaluation() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "run_lingbot_vla2_ltop_g1" not in source
    assert "run_native_policy_observation_diagnostic_forward" in source
    assert "_build_prompt_swap_plan" in source
    assert "_apply_prompt_swap" in source
    assert "permuted_prior = natural_prior.permute_rows(row_permutation)" in source
    assert "sample_actions(" not in source
    assert ".backward(" not in source
    assert "run_native_policy_training_forward" not in source
    assert "torch.optim" not in source
    assert "if bootstrap_rank == 0:" in source
    assert "dist.broadcast_object_list(asset_validation_box, src=0)" in source


def test_noncollapse_uses_two_endpoint_replay_bound() -> None:
    rows = _distinct_rows()
    metrics = g1b.replay_noncollapse_metrics(rows, rows.clone(), torch_module=torch)
    assert metrics["derivation"] == "pair_l2 > 2 * exact_replay_max_row_l2"
    assert metrics["replay_max_row_l2_floor"] == 0.0
    assert metrics["all_layers_noncollapsed"] is True
    assert all(item["identifiability_threshold_l2"] == 0.0 for item in metrics["per_layer"])
    assert all(item["stable_distinct_pair_count"] > 0 for item in metrics["per_layer"])

    collapsed = torch.zeros_like(rows)
    collapsed_metrics = g1b.replay_noncollapse_metrics(
        collapsed,
        collapsed.clone(),
        torch_module=torch,
    )
    assert collapsed_metrics["all_layers_noncollapsed"] is False
    assert all(item["stable_distinct_pair_count"] == 0 for item in collapsed_metrics["per_layer"])


def test_noncollapse_rejects_separation_inside_replay_uncertainty() -> None:
    rows = torch.zeros(1, 1, 2, 1)
    rows[0, 0, 1, 0] = 1.0
    repeat = rows.clone()
    repeat[0, 0, 0, 0] = 0.6
    metrics = g1b.replay_noncollapse_metrics(rows, repeat, torch_module=torch)
    layer = metrics["per_layer"][0]
    assert layer["replay_max_row_l2_floor"] == pytest.approx(0.6)
    assert layer["identifiability_threshold_l2"] == pytest.approx(1.2)
    assert layer["maximum_pair_l2"] == pytest.approx(1.0)
    assert layer["noncollapsed"] is False


def test_permutation_gate_accepts_only_known_equivariance_within_replay_floor() -> None:
    rows = _distinct_rows()
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted = rows.index_select(2, permutation)
    metrics = g1b.permutation_equivariance_metrics(
        rows,
        rows.clone(),
        permuted,
        permutation,
        torch_module=torch,
    )
    assert metrics["all_layers_equivariant"] is True
    assert all(item["equivariance_max_abs_error"] == 0.0 for item in metrics["per_layer"])
    assert all(item["pairwise_set_error"] == 0.0 for item in metrics["per_layer"])

    corrupted = permuted.clone()
    corrupted[0, 1, 0, 0] += 0.25
    rejected = g1b.permutation_equivariance_metrics(
        rows,
        rows.clone(),
        corrupted,
        permutation,
        torch_module=torch,
    )
    assert rejected["all_layers_equivariant"] is False


def test_permutation_gate_inherits_nonzero_replay_floor_without_tuned_tolerance() -> None:
    rows = _distinct_rows()
    repeat = rows.clone()
    repeat[0, 0, 0, 0] += 0.125
    permutation = torch.tensor([1, 2, 0], dtype=torch.long)
    observed = rows.index_select(2, permutation)
    observed[0, 0, 2, 0] += 0.125
    metrics = g1b.permutation_equivariance_metrics(
        rows,
        repeat,
        observed,
        permutation,
        torch_module=torch,
    )
    first = metrics["per_layer"][0]
    assert first["replay_max_abs_floor"] == pytest.approx(0.125)
    assert first["equivariance_max_abs_error"] == pytest.approx(0.125)
    assert first["row_identity_permuted_only"] is True


def test_normalized_permutation_recovery_tracks_behavioral_row_identity() -> None:
    rows = _distinct_rows()
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted = rows.index_select(2, permutation) * 7.0
    metrics = g1b.normalized_permutation_recovery_metrics(
        rows,
        permuted,
        permutation,
        torch_module=torch,
    )
    assert metrics["all_layers_recovered"] is True
    assert all(item["correct_rows"] == 3 for item in metrics["per_layer"])
    assert all(item["minimum_competitor_margin_l2"] > 0 for item in metrics["per_layer"])

    corrupted = permuted.clone()
    corrupted[:, :, 0] = permuted[:, :, 1]
    rejected = g1b.normalized_permutation_recovery_metrics(
        rows,
        corrupted,
        permutation,
        torch_module=torch,
    )
    assert rejected["all_layers_recovered"] is False


def test_task_address_attention_metrics_separate_prompt_shift_from_replay() -> None:
    natural = torch.tensor(
        [[[[0.8, 0.2], [0.1, 0.9]], [[0.7, 0.3], [0.2, 0.8]]]],
        dtype=torch.float32,
    )
    repeat = natural.clone()
    permutation = torch.tensor([1, 0], dtype=torch.long)
    permuted = natural.index_select(-1, permutation)
    prompt = natural.flip(-1)
    metrics = g1b.task_address_attention_metrics(
        natural,
        repeat,
        permuted,
        prompt,
        permutation,
        torch_module=torch,
    )
    assert metrics["all_layers_permutation_within_replay"] is True
    assert metrics["prompt_responsive_layer_count"] == 2
    assert metrics["per_layer"][-1]["natural_top_rows"] == [[0, 1]]
    assert metrics["per_layer"][-1]["prompt_top_rows"] == [[1, 0]]


def test_task_address_attention_metrics_reject_wrong_permuted_distribution() -> None:
    natural = torch.tensor([[[[0.8, 0.2], [0.1, 0.9]]]], dtype=torch.float32)
    permutation = torch.tensor([1, 0], dtype=torch.long)
    metrics = g1b.task_address_attention_metrics(
        natural,
        natural.clone(),
        natural.clone(),
        natural.clone(),
        permutation,
        torch_module=torch,
    )
    assert metrics["all_layers_permutation_within_replay"] is False


def test_prompt_gate_uses_exact_replay_as_its_only_threshold() -> None:
    rows = _distinct_rows()
    repeat = rows.clone()
    repeat[0, 0, 0, 0] += 0.125
    prompt = rows.clone()
    prompt[0, 0, 0, 0] += 0.125
    metrics = g1b.prompt_invariance_metrics(
        rows,
        repeat,
        prompt,
        torch_module=torch,
    )
    assert metrics["all_layers_prompt_invariant"] is True

    prompt[0, 0, 0, 0] += 0.001
    rejected = g1b.prompt_invariance_metrics(
        rows,
        repeat,
        prompt,
        torch_module=torch,
    )
    assert rejected["all_layers_prompt_invariant"] is False


def test_metric_functions_reject_malformed_rows_and_permutations() -> None:
    rows = _distinct_rows()
    with pytest.raises(ValueError, match="differ in shape"):
        g1b.prompt_invariance_metrics(
            rows,
            rows[:, :, :2],
            rows,
            torch_module=torch,
        )
    with pytest.raises(ValueError, match="every row"):
        g1b.permutation_equivariance_metrics(
            rows,
            rows,
            rows,
            torch.tensor([0, 0, 2]),
            torch_module=torch,
        )


def test_report_validator_recomputes_strict_verdict() -> None:
    report = _report()
    assert g1b.validate_ltop_g1b_report(report) is report

    failed = copy.deepcopy(report)
    failed_prompt = failed["rank_reports"][1]["posterior"]["prompt_invariance"]
    failed_prompt["per_layer"][0]["prompt_max_abs_error"] = 0.25
    failed_prompt["per_layer"][0]["prompt_max_row_l2_error"] = 0.25
    failed_prompt["per_layer"][0]["prompt_invariant"] = False
    failed_prompt["all_layers_prompt_invariant"] = False
    failed["failures"] = [
        "rank 1: posterior.prompt_invariance.all_layers_prompt_invariant is false"
    ]
    failed["status"] = "FAIL"
    assert g1b.validate_ltop_g1b_report(failed) is failed

    tampered = copy.deepcopy(failed)
    tampered["status"] = "PASS"
    with pytest.raises(ValueError, match="status differs"):
        g1b.validate_ltop_g1b_report(tampered)

    omitted = copy.deepcopy(report)
    omitted["rank_reports"] = omitted["rank_reports"][:1]
    with pytest.raises(ValueError, match="exactly two"):
        g1b.validate_ltop_g1b_report(omitted)

    metric_tamper = copy.deepcopy(report)
    metric_tamper["rank_reports"][0]["prior"]["noncollapse"]["per_layer"][0]["maximum_pair_l2"] = (
        0.0
    )
    with pytest.raises(ValueError, match="contradicts noncollapse metrics"):
        g1b.validate_ltop_g1b_report(metric_tamper)


def test_real_prompt_swap_plan_is_deterministic_and_changes_only_prompt_semantics() -> None:
    instructions = {
        "sample-a": ("task-a", "move the red block"),
        "sample-b": ("task-b", "open the drawer"),
        "sample-c": ("task-c", "turn on the light"),
        "sample-d": ("task-d", "push the blue block"),
    }

    class Dataset:
        def __init__(self) -> None:
            ordered = tuple(instructions)
            self._indices = {sample_key: index for index, sample_key in enumerate(ordered)}
            self.index = SimpleNamespace(
                segments=tuple(
                    SimpleNamespace(task_key=instructions[key][0], instruction=instructions[key][1])
                    for key in ordered
                )
            )

        def locator_by_key(self, sample_key: str) -> SimpleNamespace:
            return SimpleNamespace(segment_index=self._indices[sample_key])

        def task_key_by_key(self, sample_key: str) -> str:
            return instructions[sample_key][0]

    def transition(lane: str, episode: str, sample_key: str) -> SimpleNamespace:
        return SimpleNamespace(
            lane_id=lane,
            episode_instance_id=episode,
            sample=SimpleNamespace(sample_key=sample_key),
        )

    transitions = {
        0: (
            transition("lane-0", "episode-a", "sample-a"),
            transition("lane-1", "episode-b", "sample-b"),
        ),
        1: (
            transition("lane-0", "episode-c", "sample-c"),
            transition("lane-1", "episode-d", "sample-d"),
        ),
    }
    stream = SimpleNamespace(
        total_steps=2,
        plan_sha256="a" * 64,
        global_batch=lambda step: SimpleNamespace(transitions=transitions[step]),
    )

    dataset = Dataset()
    prompt_plan = g1b._build_prompt_swap_plan(stream, dataset)
    assert prompt_plan == g1b._build_prompt_swap_plan(stream, dataset)
    assert prompt_plan["evaluation_step"] == 0
    assert len(prompt_plan["slots"]) == g1b.G1B_WORLD_SIZE
    assert all(
        slot["recipient_instruction_sha256"] != slot["donor_instruction_sha256"]
        for slot in prompt_plan["slots"]
    )

    @dataclass(frozen=True)
    class Request:
        sample_key: str
        task_key: str

    @dataclass(frozen=True)
    class Training:
        host_items: tuple[dict[str, str], ...]
        structural_target_requests: tuple[Request, ...]

    @dataclass(frozen=True)
    class Microbatch:
        optimizer_step: int
        transitions: tuple[SimpleNamespace, ...]

    @dataclass(frozen=True)
    class Planned:
        training: Training
        plan_microbatch: Microbatch
        plan_sha256: str
        task_intervention_sha256: str | None = None

    recipient = transitions[0][0]
    natural_task_key, natural_instruction = instructions[recipient.sample.sample_key]
    planned = Planned(
        training=Training(
            host_items=({"task": natural_instruction},),
            structural_target_requests=(
                Request(sample_key=recipient.sample.sample_key, task_key=natural_task_key),
            ),
        ),
        plan_microbatch=Microbatch(optimizer_step=0, transitions=(recipient,)),
        plan_sha256=stream.plan_sha256,
    )
    swapped = g1b._apply_prompt_swap(planned, prompt_plan, dataset)
    assert swapped.training.host_items[0]["task"] != natural_instruction
    assert swapped.training.structural_target_requests[0].task_key != natural_task_key
    assert swapped.task_intervention_sha256 == prompt_plan["artifact_sha256"]


def test_row_permutation_is_nonidentity_and_complete() -> None:
    permutation = g1b._nonidentity_row_permutation(
        16,
        0,
        torch_module=torch,
        device=torch.device("cpu"),
    )
    assert not torch.equal(permutation, torch.arange(16))
    assert torch.equal(permutation.sort().values, torch.arange(16))
