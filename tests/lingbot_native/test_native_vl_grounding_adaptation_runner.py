from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.runtime_provenance import python_source_tree_contract
from tools.train_lingbot_native_vl_grounding import (
    ADR127_GRADIENT_AUDIT_STEPS,
    ADR127_INITIAL_QWEN_REVISION,
    ADR127_MAX_STEPS,
    ADR127_SCHEDULE_TOTAL_STEPS,
    ADR128_ADAM_BETA1,
    ADR128_ADAM_BETA2,
    ADR128_ADAM_EPS,
    ADR128_INITIAL_QWEN_REVISION,
    ADR128_LEARNING_RATE,
    ADR128_MAX_GRAD_NORM,
    ADR128_MAX_STEPS,
    ADR128_SCHEDULE_TOTAL_STEPS,
    ADR128_SEED,
    ADR128_SMOKE_STEPS,
    ADR128_WEIGHT_DECAY,
    CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE,
    CALVIN_FACTOR_TARGET_ONLY,
    CALVIN_FACTOR_TARGET_REPEAT_CONTROL,
    CURRICULUM_OBSERVATION_DUAL_LATTICE,
    CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE,
    _calvin_factor_weights,
    _counterfactual_gradient_audit_status,
    _learning_rate_for_step,
    _require_variant_pair,
    _select_curriculum_microbatches,
    _summarize_gradient_triple_rows,
    _validate_calvin_factor_mode,
    _validate_crossed_bounded_mode,
    _validate_git_revision,
    _validate_public_vl_retention_args,
    _validate_sha256,
    _validate_training_horizons,
)

_TOOL = Path(__file__).resolve().parents[2] / "tools/train_lingbot_native_vl_grounding.py"


def test_grounding_adaptation_requires_exact_lowercase_git_revision() -> None:
    revision = "a" * 40
    assert _validate_git_revision(revision) == revision
    for invalid in ("a" * 39, "a" * 41, "A" * 40, "z" * 40):
        with pytest.raises(ContractError, match="Git commit"):
            _validate_git_revision(invalid)


def test_grounding_adaptation_requires_exact_curriculum_file_digest() -> None:
    digest = "a" * 64
    assert _validate_sha256(digest, name="curriculum") == digest
    for invalid in ("a" * 63, "a" * 65, "A" * 64, "z" * 64):
        with pytest.raises(ContractError, match="SHA-256"):
            _validate_sha256(invalid, name="curriculum")


def test_grounding_adaptation_binds_complete_python_source_trees(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "a.py").write_text("VALUE = 1\n")
    (nested / "b.py").write_text("VALUE = 2\n")
    (nested / "ignored.txt").write_text("not runtime Python\n")

    first = python_source_tree_contract({"runtime": source})
    assert first["file_count"] == 2
    assert isinstance(first["tree_sha256"], str)
    assert len(first["tree_sha256"]) == 64

    (nested / "b.py").write_text("VALUE = 3\n")
    second = python_source_tree_contract({"runtime": source})
    assert second["file_count"] == 2
    assert second["tree_sha256"] != first["tree_sha256"]


def test_grounding_adaptation_schedule_is_explicit_warmup_then_cosine() -> None:
    values = tuple(
        _learning_rate_for_step(
            step,
            schedule_total_steps=5,
            warmup_steps=2,
            base_learning_rate=1e-6,
        )
        for step in range(5)
    )
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(5e-7)
    assert values[2] == pytest.approx(1e-6)
    assert values[3] == pytest.approx(7.5e-7)
    assert values[4] == pytest.approx(2.5e-7)


def test_grounding_adaptation_schedule_rejects_implicit_or_invalid_values() -> None:
    with pytest.raises(ContractError, match="indices"):
        _learning_rate_for_step(
            1,
            schedule_total_steps=1,
            warmup_steps=0,
            base_learning_rate=1e-6,
        )
    with pytest.raises(ContractError, match="learning rate"):
        _learning_rate_for_step(
            0,
            schedule_total_steps=1,
            warmup_steps=0,
            base_learning_rate=0.0,
        )


def test_grounding_adaptation_bounded_run_keeps_full_schedule_horizon() -> None:
    bounded_last = _learning_rate_for_step(
        31,
        schedule_total_steps=432,
        warmup_steps=0,
        base_learning_rate=1e-6,
    )
    compressed_last = _learning_rate_for_step(
        31,
        schedule_total_steps=32,
        warmup_steps=0,
        base_learning_rate=1e-6,
    )

    assert bounded_last == pytest.approx(9.873481060625123e-7)
    assert compressed_last == pytest.approx(2.407636663901557e-9)
    assert bounded_last > 400 * compressed_last


def test_grounding_adaptation_horizons_are_explicit_and_fail_closed() -> None:
    assert _validate_training_horizons(
        max_steps=32,
        schedule_total_steps=432,
        warmup_steps=0,
    ) == (32, 432, 0)
    for values in (
        {"max_steps": 32, "schedule_total_steps": 31, "warmup_steps": 0},
        {"max_steps": 32, "schedule_total_steps": 432, "warmup_steps": 32},
        {"max_steps": True, "schedule_total_steps": 432, "warmup_steps": 0},
    ):
        with pytest.raises(ContractError, match="training horizons"):
            _validate_training_horizons(**values)


def test_grounding_adaptation_materialization_requires_exact_variant_pair() -> None:
    first = object()
    second = object()
    assert _require_variant_pair((first, second)) == (first, second)
    for invalid in (None, (), (first,), (first, second, object()), [first, second]):
        with pytest.raises(ContractError, match="two-variant tuple"):
            _require_variant_pair(invalid)


def test_grounding_adaptation_binds_and_verifies_the_declared_visual_lattice() -> None:
    source = _TOOL.read_text()
    configure = source.index("processor_lattice = configure_native_processor_lattice(")
    preprocess = source.index("batch = build_native_vl_grounding_batch(factor_record, processor)")
    forward = source.index("loss = run_native_vl_grounding_forward(policy, batch)", preprocess)

    assert configure < preprocess < forward
    assert 'parser.add_argument("--visual-lattice"' in source
    assert "image_grid_thw != expected_grid_thw" in source
    assert '"processor_lattices": processor_lattices' in source


def test_grounding_curriculum_averages_both_scales_before_one_update() -> None:
    source = _TOOL.read_text()
    backward = source.index("(effective_weight * loss).backward()")
    gradient_check = source.index("gradient_metrics = _distributed_gradient_metrics(")
    optimizer_step = source.index("optimizer.step()")

    assert backward < gradient_check < optimizer_step
    assert '"exhaustive_dual_lattice_curriculum"' in source
    assert "curriculum_plan.resolve_step(step)" in source
    assert "!= sharded_scope.trainable_numel" in source
    assert "did not cover its complete trainable scope" in source


def _retention_args(**overrides) -> Namespace:
    values = {
        "curriculum_observation_mode": CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE,
        "max_steps": 1,
        "public_vl_retention_manifest": None,
        "public_vl_retention_manifest_sha256": None,
        "public_vl_retention_root": None,
        "public_vl_retention_weight": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_grounding_public_retention_is_optional_but_atomic_and_frozen() -> None:
    assert _validate_public_vl_retention_args(_retention_args()) is None
    with pytest.raises(ContractError, match="all present"):
        _validate_public_vl_retention_args(
            _retention_args(public_vl_retention_manifest=Path("manifest.json"))
        )
    complete = {
        "public_vl_retention_manifest": Path("manifest.json"),
        "public_vl_retention_manifest_sha256": "a" * 64,
        "public_vl_retention_root": Path("retention"),
        "public_vl_retention_weight": 0.1,
    }
    with pytest.raises(ContractError, match="official-native-once"):
        _validate_public_vl_retention_args(
            _retention_args(
                **complete,
                curriculum_observation_mode=CURRICULUM_OBSERVATION_DUAL_LATTICE,
            )
        )
    with pytest.raises(ContractError, match="differs from ADR-125"):
        _validate_public_vl_retention_args(
            _retention_args(**{**complete, "public_vl_retention_weight": 0.2})
        )


def test_grounding_public_retention_updates_same_qwen_before_one_optimizer_step() -> None:
    source = _TOOL.read_text()
    calvin_backward = source.index("(effective_weight * loss).backward()")
    retention_forward = source.index(
        "retention_loss = run_native_vl_grounding_forward(policy, retention_batch)"
    )
    retention_backward = source.index(
        "(PUBLIC_NATIVE_VL_RETENTION_WEIGHT * retention_loss).backward()"
    )
    gradient_check = source.index("gradient_metrics = _distributed_gradient_metrics(")
    optimizer_step = source.index("optimizer.step()")

    assert (
        calvin_backward < retention_forward < retention_backward < gradient_check < optimizer_step
    )
    assert '"global_loss_factors"' in source
    assert '"rank_streams": {"0": "referring", "1": "vqa"}' in source


def test_grounding_public_retention_has_a_distinct_bounded_official_processor() -> None:
    source = _TOOL.read_text()
    configure = source.index(
        "retention_processor_contract = configure_native_processor_area_budget("
    )
    preprocess = source.index(
        "retention_batch = build_native_vl_grounding_batch(\n"
        "                    retention_record,\n"
        "                    retention_processor,"
    )
    budget = source.index("retention_grid_budget = validate_native_processor_record_grid(")
    transfer = source.index("retention_batch = retention_batch.to(", budget)
    forward = source.index(
        "retention_loss = run_native_vl_grounding_forward(policy, retention_batch)"
    )

    assert configure < preprocess < budget < transfer < forward
    assert '"processor": retention_processor_contract' in source


def test_grounding_curriculum_can_select_official_native_observation_once() -> None:
    batches = (
        (8, "static", ("left", "right")),
        (14, "static", ("right", "left")),
    )

    assert (
        _select_curriculum_microbatches(
            batches,
            observation_mode=CURRICULUM_OBSERVATION_DUAL_LATTICE,
        )
        == batches
    )
    assert (
        _select_curriculum_microbatches(
            batches,
            observation_mode=CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE,
        )
        == batches[:1]
    )
    with pytest.raises(ContractError, match="signed dual-lattice"):
        _select_curriculum_microbatches(
            batches[:1],
            observation_mode=CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE,
        )
    with pytest.raises(ContractError, match="unsupported"):
        _select_curriculum_microbatches(batches, observation_mode="custom")


def _adr127_args(**overrides) -> Namespace:
    values = {
        "calvin_factor_mode": CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE,
        "adr127_smoke": False,
        "counterfactual_gradient_audit": True,
        "curriculum_observation_mode": CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE,
        "curriculum_plan": Path("curriculum.json"),
        "initial_qwen_revision": ADR127_INITIAL_QWEN_REVISION,
        "max_steps": ADR127_MAX_STEPS,
        "schedule_total_steps": ADR127_SCHEDULE_TOTAL_STEPS,
        "warmup_steps": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_grounding_calvin_factor_mixtures_are_explicit_and_matched() -> None:
    assert _calvin_factor_weights(CALVIN_FACTOR_TARGET_ONLY) == (("target", 1.0),)
    assert _calvin_factor_weights(CALVIN_FACTOR_TARGET_REPEAT_CONTROL) == (
        ("target", 0.5),
        ("target_repeat", 0.5),
    )
    assert _calvin_factor_weights(CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE) == (
        ("target", 0.5),
        ("scene", 0.5),
    )
    with pytest.raises(ContractError, match="unsupported"):
        _calvin_factor_weights("custom")


def test_grounding_adr127_factor_modes_are_frozen_and_fail_closed() -> None:
    for mode in (
        CALVIN_FACTOR_TARGET_REPEAT_CONTROL,
        CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE,
    ):
        args = _adr127_args(
            calvin_factor_mode=mode,
            counterfactual_gradient_audit=(mode == CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE),
        )
        assert _validate_calvin_factor_mode(args, public_retention_enabled=True) == mode

    assert (
        _validate_calvin_factor_mode(
            _adr127_args(
                calvin_factor_mode=CALVIN_FACTOR_TARGET_ONLY,
                counterfactual_gradient_audit=False,
                curriculum_plan=None,
                max_steps=3,
            ),
            public_retention_enabled=False,
        )
        == CALVIN_FACTOR_TARGET_ONLY
    )
    invalid_overrides = (
        {"curriculum_plan": None},
        {"curriculum_observation_mode": CURRICULUM_OBSERVATION_DUAL_LATTICE},
        {"max_steps": ADR127_MAX_STEPS - 1},
        {"schedule_total_steps": ADR127_SCHEDULE_TOTAL_STEPS - 1},
        {"warmup_steps": 1},
        {"initial_qwen_revision": "a" * 40},
    )
    for overrides in invalid_overrides:
        with pytest.raises(ContractError, match="frozen experiment"):
            _validate_calvin_factor_mode(
                _adr127_args(**overrides),
                public_retention_enabled=True,
            )
    with pytest.raises(ContractError, match="frozen experiment"):
        _validate_calvin_factor_mode(
            _adr127_args(),
            public_retention_enabled=False,
        )

    for mode in (
        CALVIN_FACTOR_TARGET_REPEAT_CONTROL,
        CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE,
    ):
        smoke = _adr127_args(
            adr127_smoke=True,
            calvin_factor_mode=mode,
            counterfactual_gradient_audit=False,
            max_steps=1,
        )
        assert _validate_calvin_factor_mode(smoke, public_retention_enabled=True) == mode
    with pytest.raises(ContractError, match="gradient audit mode"):
        _validate_calvin_factor_mode(
            _adr127_args(counterfactual_gradient_audit=False),
            public_retention_enabled=True,
        )


def _adr128_args(**overrides) -> Namespace:
    values = {
        "adam_beta1": ADR128_ADAM_BETA1,
        "adam_beta2": ADR128_ADAM_BETA2,
        "adam_eps": ADR128_ADAM_EPS,
        "adr127_smoke": False,
        "adr128_smoke": False,
        "calvin_factor_mode": CALVIN_FACTOR_TARGET_ONLY,
        "counterfactual_gradient_audit": False,
        "crossed_arm": "candidate",
        "crossed_bounded_plan": Path("crossed.json"),
        "crossed_bounded_plan_sha256": "a" * 64,
        "curriculum_observation_mode": CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE,
        "curriculum_plan": Path("curriculum.json"),
        "initial_qwen_revision": ADR128_INITIAL_QWEN_REVISION,
        "learning_rate": ADR128_LEARNING_RATE,
        "max_grad_norm": ADR128_MAX_GRAD_NORM,
        "max_steps": ADR128_MAX_STEPS,
        "schedule_total_steps": ADR128_SCHEDULE_TOTAL_STEPS,
        "seed": ADR128_SEED,
        "warmup_steps": 0,
        "weight_decay": ADR128_WEIGHT_DECAY,
    }
    values.update(overrides)
    return Namespace(**values)


def test_grounding_adr128_contract_is_matched_bounded_and_fail_closed() -> None:
    assert _validate_crossed_bounded_mode(
        _adr128_args(),
        public_retention_enabled=True,
    )
    assert (
        _validate_crossed_bounded_mode(
            _adr128_args(
                crossed_bounded_plan=None, crossed_bounded_plan_sha256=None, crossed_arm=None
            ),
            public_retention_enabled=False,
        )
        is False
    )
    assert _validate_crossed_bounded_mode(
        _adr128_args(adr128_smoke=True, max_steps=ADR128_SMOKE_STEPS),
        public_retention_enabled=True,
    )
    for overrides, message in (
        ({"curriculum_plan": None}, "frozen experiment"),
        ({"crossed_arm": None}, "all present"),
        ({"max_steps": ADR128_MAX_STEPS - 1}, "horizon"),
        ({"learning_rate": ADR128_LEARNING_RATE * 2}, "optimizer"),
        ({"seed": ADR128_SEED + 1}, "frozen experiment"),
        ({"initial_qwen_revision": "b" * 40}, "frozen experiment"),
    ):
        with pytest.raises(ContractError, match=message):
            _validate_crossed_bounded_mode(
                _adr128_args(**overrides),
                public_retention_enabled=True,
            )
    with pytest.raises(ContractError, match="frozen experiment"):
        _validate_crossed_bounded_mode(
            _adr128_args(),
            public_retention_enabled=False,
        )


def test_grounding_adr128_materializes_every_record_before_model_allocation() -> None:
    source = _TOOL.read_text()
    materialization = source.index("crossed_materialization_report = None")
    record_validation = source.index("_validate_crossed_materialized_record(", materialization)
    model_allocation = source.index("with init_empty_weights(), no_init_weights():")
    training_step = source.index("for step in range(args.max_steps):")

    assert materialization < record_validation < model_allocation < training_step
    assert '"unique_record_count": len(planned_records)' in source
    assert '"crossed_cpu_materialization": crossed_materialization_report' in source
    assert "crossed_plan.resolve_record(" in source
    assert "materialize_fixed_observation_native_vl_record(" in source


def test_grounding_adr127_triple_gradient_summary_is_exact_and_fail_closed() -> None:
    prefix = "model.qwenvl_with_expert.qwenvl.model."
    names = (
        f"{prefix}language_model.embed_tokens.weight",
        f"{prefix}visual.merger.linear_fc1.weight",
    )
    summary = _summarize_gradient_triple_rows(
        target_scene_names=names,
        target_scene_rows=[[1.0, 4.0, 9.0, 2.0], [-1.0, 1.0, 1.0, 3.0]],
        target_public_names=names,
        target_public_rows=[[-1.0, 4.0, 1.0, 2.0], [0.0, 1.0, 4.0, 3.0]],
        scene_public_names=names,
        scene_public_rows=[[2.0, 9.0, 1.0, 2.0], [0.0, 1.0, 4.0, 3.0]],
    )
    assert summary["global"]["mixed_gradient_descends"] == {
        "target": True,
        "scene": True,
        "public": True,
    }
    reports = [
        {
            "completed_updates_before_audit": step,
            "summary": summary,
        }
        for step in ADR127_GRADIENT_AUDIT_STEPS
    ]
    assert _counterfactual_gradient_audit_status(reports) == "PASS"
    reports[2]["summary"] = {
        **summary,
        "global": {
            **summary["global"],
            "mixed_gradient_descends": {
                "target": False,
                "scene": True,
                "public": True,
            },
        },
    }
    assert _counterfactual_gradient_audit_status(reports) == "FAIL"

    with pytest.raises(ContractError, match="repeated gradient norms changed"):
        _summarize_gradient_triple_rows(
            target_scene_names=names,
            target_scene_rows=[[1.0, 4.0, 9.0, 2.0], [-1.0, 1.0, 1.0, 3.0]],
            target_public_names=names,
            target_public_rows=[[-1.0, 5.0, 1.0, 2.0], [0.0, 1.0, 4.0, 3.0]],
            scene_public_names=names,
            scene_public_rows=[[2.0, 9.0, 1.0, 2.0], [0.0, 1.0, 4.0, 3.0]],
        )


def test_grounding_adr127_scene_and_control_share_one_optimizer_boundary() -> None:
    source = _TOOL.read_text()
    materialize = source.index("records = materialize_fixed_observation_native_vl_records(")
    scene = source.index("scene_records = build_counterfactual_scene_grounding_records(")
    factor_loop = source.index("for factor_name, factor_weight in factor_weights:")
    backward = source.index("(effective_weight * loss).backward()")
    gradient_check = source.index("gradient_metrics = _distributed_gradient_metrics(")
    optimizer_step = source.index("optimizer.step()")

    assert materialize < scene < factor_loop < backward < gradient_check < optimizer_step
    assert '"target_repeat": target_record' in source
    assert '"scene": scene_record' in source
    assert '"calvin_factor_contract"' in source


def test_grounding_adaptation_bootstraps_and_reports_allocator_before_torch() -> None:
    source = _TOOL.read_text()
    bootstrap = source.index("bootstrap_cuda_allocator(sys.argv[1:])")
    torch_import = source.index("    import torch\n")

    assert bootstrap < torch_import
    assert '"cuda_allocator": args.cuda_allocator' in source
    assert "CUDA allocator pre-bootstrap differs from parsed arguments" in source


def test_grounding_adaptation_help_exposes_visual_lattice_from_any_directory(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    result = subprocess.run(
        [sys.executable, str(_TOOL), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--visual-lattice" in result.stdout
    assert "--curriculum-observation-mode" in result.stdout
    assert "--calvin-factor-mode" in result.stdout
    assert "--adr127-smoke" in result.stdout
    assert "--adr128-smoke" in result.stdout
    assert "--counterfactual-gradient-audit" in result.stdout
    assert "--curriculum-plan" in result.stdout
    assert "--curriculum-plan-sha256" in result.stdout
    assert "--crossed-bounded-plan" in result.stdout
    assert "--crossed-bounded-plan-sha256" in result.stdout
    assert "--crossed-arm" in result.stdout
    assert "--cuda-allocator" in result.stdout
    assert "--schedule-total-steps" in result.stdout
    assert "--public-vl-retention-manifest" in result.stdout
    assert "--public-vl-retention-manifest-sha256" in result.stdout
    assert "--public-vl-retention-root" in result.stdout
    assert "--public-vl-retention-weight" in result.stdout
