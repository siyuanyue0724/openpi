from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
load_training_recipe = pytest.importorskip("picf_next.training.recipe").load_training_recipe
validate_axis_calibrated_m2 = pytest.importorskip(
    "picf_next.training.m2_acceptance"
).validate_axis_calibrated_m2
load_picf_current_frame_checkpoint = pytest.importorskip(
    "picf_next.training.stage_checkpoints"
).load_picf_current_frame_checkpoint
training_cli = pytest.importorskip("tools.train_molmoact2_calvin_picf")
_atomic_json = training_cli._atomic_json
_build_picf_run_contract = training_cli._build_picf_run_contract
_distributed_system_telemetry = training_cli._distributed_system_telemetry
_mean_metrics = training_cli._mean_metrics
_optimizer_step_observability = training_cli._optimizer_step_observability
_publish_fresh_run_metadata = training_cli._publish_fresh_run_metadata
_reconcile_metrics_for_resume = training_cli._reconcile_metrics_for_resume
_load_vjepa2_cache_for_arm = training_cli._load_vjepa2_cache_for_arm
_action_arm_spec = training_cli._action_arm_spec
_sha256_file = training_cli._sha256_file
_synchronize_step_timing = training_cli._synchronize_step_timing
_validate_scheduler_epoch = training_cli._validate_scheduler_epoch
_validate_persistent_run_root = training_cli._validate_persistent_run_root
_validate_m0_for_mode = training_cli._validate_m0_for_mode
_validate_stationary_core_for_mode = training_cli._validate_stationary_core_for_mode
_validate_training_checkpoint = training_cli._validate_training_checkpoint

_ROOT = Path(__file__).resolve().parents[1]


class _FakeAccelerator:
    device = torch.device("cpu")

    def __init__(self, *, mismatched_schema: bool = False) -> None:
        self.mismatched_schema = mismatched_schema
        self.reduce_calls = 0

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        remote = tensor.clone()
        if self.mismatched_schema:
            remote[0] += 1
        return torch.cat((tensor, remote))

    def reduce(self, tensor: torch.Tensor, *, reduction: str) -> torch.Tensor:
        assert reduction == "mean"
        self.reduce_calls += 1
        return tensor


def test_action_scheduler_must_follow_successful_global_updates() -> None:
    scheduler = SimpleNamespace(state_dict=lambda: {"last_epoch": 7})
    assert _validate_scheduler_epoch(scheduler, successful_optimizer_steps=7) == 7
    with pytest.raises(RuntimeError, match="global optimizer progress"):
        _validate_scheduler_epoch(scheduler, successful_optimizer_steps=8)


def test_m0_report_is_optional_only_for_static_verify() -> None:
    assert _validate_m0_for_mode(
        report_path=None,
        cloud_config={},
        verify_only=True,
    ) == (None, None)
    with pytest.raises(ValueError, match="training requires --m0-report"):
        _validate_m0_for_mode(
            report_path=None,
            cloud_config={},
            verify_only=False,
        )


def _accepted_m2_report(
    checkpoint_sha256: str,
    metrics_sha256: str,
) -> dict[str, object]:
    protocol = {
        "checkpoint_selection": "fixed-current-frame-best-no-reselection",
        "fit_data": "train-fixed-match-residuals-only",
        "fit_objective": "axiswise-diagonal-gaussian-nll-with-declared-target-variance",
        "fitted_parameter_names": ["discovery.variance_head.bias"],
        "frozen_parameter_names": ["discovery.variance_head.weight"],
        "matching_mean_and_representation": "frozen-exactly",
        "validation_and_heldout": "evaluation-only-never-fit",
        "variance_dependency": "axis-only-task-identity-and-query-independent",
    }
    checks = {
        "heldout_error_to_variance_ratio_in_bounds": True,
        "heldout_nll_below_reset": True,
        "legacy_variance_weight_zero": True,
        "nonvariance_state_exact": True,
        "softplus_roundtrip_within_tolerance": True,
        "train_nll_not_above_reset": True,
        "validation_nll_below_reset": True,
    }
    return {
        "schema": "picf-next.molmoact2-m2-axis-variance-calibration.v1",
        "status": "CALIBRATED_CANDIDATE",
        "protocol": protocol,
        "decision": {
            "status": "PASS",
            "checks": checks,
            "failed_checks": [],
            "later_gates_authorized": ["M3_bounded_mechanism_smoke"],
            "long_training_authorized": False,
        },
        "data_isolation": {
            "fit_split": "train",
            "fit_rows": 10,
            "evaluation_only_rows": {"validation": 5, "heldout": 5},
        },
        "state_isolation": {
            "initial_nonvariance_state_sha256": "1" * 64,
            "post_extraction_nonvariance_state_sha256": "1" * 64,
            "final_nonvariance_state_sha256": "1" * 64,
            "nonvariance_state_exact": True,
            "legacy_variance_weight_zero": True,
        },
        "input_sha256": {
            "checkpoints/current_frame_best.pt": "2" * 64,
            "config": "3" * 64,
            "evaluation_report.json": "4" * 64,
            "feature_cache/manifest.json": "5" * 64,
            "launch_manifest.json": "6" * 64,
            "residual_permutation_probe/report.json": "7" * 64,
            "training_report.json": "8" * 64,
        },
        "output_sha256": {
            "current_frame_axis_calibrated.pt": checkpoint_sha256,
            "metrics.json": metrics_sha256,
        },
    }


def test_stationary_core_is_optional_only_for_static_verify(tmp_path: Path) -> None:
    assert (
        _validate_stationary_core_for_mode(
            report_path=None,
            checkpoint_path=None,
            verify_only=True,
        )
        is None
    )
    with pytest.raises(ValueError, match="requires both"):
        _validate_stationary_core_for_mode(
            report_path=None,
            checkpoint_path=tmp_path / "checkpoint.pt",
            verify_only=False,
        )


def test_m2_calibration_report_and_checkpoint_are_hash_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "current_frame_axis_calibrated.pt"
    torch.save({"model": {}}, checkpoint)
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(_accepted_m2_report(_sha256_file(checkpoint), _sha256_file(metrics)))
    )

    binding = validate_axis_calibrated_m2(
        report_path=report,
        checkpoint_path=checkpoint,
    )

    assert binding is not None
    assert binding["checkpoint_sha256"] == _sha256_file(checkpoint)
    assert binding["metrics_sha256"] == _sha256_file(metrics)
    assert binding["feature_cache_manifest_sha256"] == "5" * 64
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="absent or changed"):
        validate_axis_calibrated_m2(
            report_path=report,
            checkpoint_path=checkpoint,
        )


def test_m2_calibration_rejects_incomplete_acceptance_evidence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "current_frame_axis_calibrated.pt"
    torch.save({"model": {}}, checkpoint)
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    payload = _accepted_m2_report(_sha256_file(checkpoint), _sha256_file(metrics))
    payload["decision"]["checks"].pop("heldout_nll_below_reset")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="failed acceptance check"):
        validate_axis_calibrated_m2(
            report_path=report,
            checkpoint_path=checkpoint,
        )


class _CheckpointCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(2, 2)
        self.discovery = torch.nn.Module()
        self.discovery.mean_head = torch.nn.Linear(2, 2)
        self.discovery.variance_head = torch.nn.Linear(2, 2)
        self.discovery.variance_head.weight.requires_grad_(False)
        self.discovery.localization_confidence_head = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.discovery.localization_confidence_head.weight.zero_()
            self.discovery.localization_confidence_head.bias.zero_()
        self.posterior_filter = torch.nn.Linear(2, 2)


def test_m2_loader_initializes_only_current_frame_and_keeps_posterior_fresh(
    tmp_path: Path,
) -> None:
    source = _CheckpointCore()
    target = _CheckpointCore()
    with torch.no_grad():
        source.projector.weight.fill_(3.0)
        source.discovery.mean_head.weight.fill_(4.0)
        source.discovery.variance_head.weight.zero_()
        source.discovery.variance_head.bias.fill_(0.5)
    current_state = {
        name: value
        for name, value in source.state_dict().items()
        if name.startswith(("projector.", "discovery."))
        and not name.startswith("discovery.localization_confidence_head.")
    }
    posterior_before = {
        name: value.clone() for name, value in target.posterior_filter.state_dict().items()
    }
    checkpoint = tmp_path / "m2.pt"
    torch.save({"model": current_state}, checkpoint)

    result = load_picf_current_frame_checkpoint(
        target,
        checkpoint,
        expected_sha256=_sha256_file(checkpoint),
    )

    assert result["loaded_keys"] == tuple(sorted(current_state))
    assert {name for name in result["fresh_keys"] if not name.startswith("posterior_filter.")} == {
        "discovery.localization_confidence_head.bias",
        "discovery.localization_confidence_head.weight",
    }
    torch.testing.assert_close(target.projector.weight, source.projector.weight)
    torch.testing.assert_close(
        target.discovery.mean_head.weight,
        source.discovery.mean_head.weight,
    )
    assert not target.discovery.variance_head.weight.requires_grad
    for name, value in target.posterior_filter.state_dict().items():
        torch.testing.assert_close(value, posterior_before[name])


def test_m2_loader_rejects_nonneutral_fresh_localization_confidence(
    tmp_path: Path,
) -> None:
    source = _CheckpointCore()
    target = _CheckpointCore()
    with torch.no_grad():
        source.discovery.variance_head.weight.zero_()
        target.discovery.localization_confidence_head.bias.fill_(1.0)
    current_state = {
        name: value
        for name, value in source.state_dict().items()
        if name.startswith(("projector.", "discovery."))
        and not name.startswith("discovery.localization_confidence_head.")
    }
    checkpoint = tmp_path / "m2.pt"
    torch.save({"model": current_state}, checkpoint)

    with pytest.raises(ValueError, match="zero initialization"):
        load_picf_current_frame_checkpoint(
            target,
            checkpoint,
            expected_sha256=_sha256_file(checkpoint),
        )


def test_m2_loader_accepts_exact_modern_current_frame_checkpoint(tmp_path: Path) -> None:
    source = _CheckpointCore()
    target = _CheckpointCore()
    del source.posterior_filter
    del target.posterior_filter
    with torch.no_grad():
        source.projector.weight.fill_(3.0)
        source.discovery.mean_head.weight.fill_(4.0)
        source.discovery.variance_head.weight.zero_()
        source.discovery.variance_head.bias.fill_(0.5)
    checkpoint = tmp_path / "modern-m2.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    result = load_picf_current_frame_checkpoint(
        target,
        checkpoint,
        expected_sha256=_sha256_file(checkpoint),
    )

    assert result["fresh_keys"] == ()
    assert result["loaded_keys"] == tuple(sorted(source.state_dict()))
    torch.testing.assert_close(target.projector.weight, source.projector.weight)
    torch.testing.assert_close(
        target.discovery.localization_confidence_head.weight,
        source.discovery.localization_confidence_head.weight,
    )


def test_m2_loader_rejects_partial_or_unrelated_current_frame_state(
    tmp_path: Path,
) -> None:
    source = _CheckpointCore()
    target = _CheckpointCore()
    with torch.no_grad():
        source.discovery.variance_head.weight.zero_()
    base_state = {
        name: value
        for name, value in source.state_dict().items()
        if name.startswith(("projector.", "discovery."))
        and not name.startswith("discovery.localization_confidence_head.")
    }
    cases = {
        "partial-confidence": {
            **base_state,
            "discovery.localization_confidence_head.weight": (
                source.discovery.localization_confidence_head.weight
            ),
        },
        "missing-current-frame": {
            name: value for name, value in base_state.items() if name != "discovery.mean_head.bias"
        },
    }

    for name, state in cases.items():
        checkpoint = tmp_path / f"{name}.pt"
        torch.save({"model": state}, checkpoint)
        with pytest.raises(ValueError, match="checkpoint"):
            load_picf_current_frame_checkpoint(
                target,
                checkpoint,
                expected_sha256=_sha256_file(checkpoint),
            )


def test_run_contract_uses_verified_dataset_tree_and_validates_plan() -> None:
    recipe = load_training_recipe(_ROOT / "configs/training/molmoact2_calvin_m3_probe.json")
    plan_sha256 = "b" * 64
    plan = SimpleNamespace(
        plan_sha256=plan_sha256,
        comparison_id="m3-contract-fixture",
        dataset_id=recipe.dataset.dataset_id,
        dataset_revision=recipe.dataset.dataset_revision,
        dataset_manifest_sha256=recipe.artifacts.dataset_tree_sha256,
        global_batch_size=2,
    )
    assets = SimpleNamespace(
        dataset_manifest=SimpleNamespace(tree_sha256=recipe.artifacts.dataset_tree_sha256)
    )
    args = SimpleNamespace(
        comparison_id=plan.comparison_id,
        global_batch_size=2,
        gradient_accumulation_steps=1,
    )
    stationary_core_binding = {
        "acceptance_report_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
    }

    contract = _build_picf_run_contract(
        recipe=recipe,
        plan=plan,
        assets=assets,
        args=args,
        code_revision="1" * 40,
        checkpoint_manifest_sha256="a" * 64,
        stationary_core_binding=stationary_core_binding,
        action_arm=_action_arm_spec("D"),
        vjepa2_binding={"manifest_sha256": "e" * 64},
        world_size=2,
    )

    assert contract.dataset_manifest_sha256 == recipe.artifacts.dataset_tree_sha256
    assert contract.sample_plan_sha256 == plan_sha256
    assert contract.arm_config["stationary_temporal_initialization"] == stationary_core_binding
    assert contract.arm == "picf"
    assert contract.arm_config["causal_factorization"] == _action_arm_spec("D")
    assert contract.arm_config["vjepa2_cache"] == {"manifest_sha256": "e" * 64}


def test_run_contract_rejects_verified_tree_that_differs_from_frozen_plan() -> None:
    recipe = load_training_recipe(_ROOT / "configs/training/molmoact2_calvin_m3_probe.json")
    plan = SimpleNamespace(
        plan_sha256="b" * 64,
        comparison_id="m3-contract-negative-fixture",
        dataset_id=recipe.dataset.dataset_id,
        dataset_revision=recipe.dataset.dataset_revision,
        dataset_manifest_sha256=recipe.artifacts.dataset_tree_sha256,
        global_batch_size=2,
    )
    assets = SimpleNamespace(dataset_manifest=SimpleNamespace(tree_sha256="c" * 64))
    args = SimpleNamespace(
        comparison_id=plan.comparison_id,
        global_batch_size=2,
        gradient_accumulation_steps=1,
    )

    with pytest.raises(ValueError, match="manifest hashes differ"):
        _build_picf_run_contract(
            recipe=recipe,
            plan=plan,
            assets=assets,
            args=args,
            code_revision="1" * 40,
            checkpoint_manifest_sha256="a" * 64,
            stationary_core_binding={"checkpoint_sha256": "d" * 64},
            action_arm=_action_arm_spec("C"),
            vjepa2_binding=None,
            world_size=2,
        )


def test_action_arm_cache_contract_is_atomic_and_role_specific(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_sha = "d" * 64
    for arm in ("A", "C"):
        assert _load_vjepa2_cache_for_arm(
            arm_spec=_action_arm_spec(arm),
            cache_root=None,
            cache_manifest_sha256=None,
            cache_memory_capacity=64,
            dataset_tree_sha256=dataset_sha,
            require_persistent_root=False,
        ) == (None, None)
        with pytest.raises(ValueError, match="forbid"):
            _load_vjepa2_cache_for_arm(
                arm_spec=_action_arm_spec(arm),
                cache_root=tmp_path,
                cache_manifest_sha256="a" * 64,
                cache_memory_capacity=64,
                dataset_tree_sha256=dataset_sha,
                require_persistent_root=False,
            )

    with pytest.raises(ValueError, match="require"):
        _load_vjepa2_cache_for_arm(
            arm_spec=_action_arm_spec("B"),
            cache_root=None,
            cache_manifest_sha256=None,
            cache_memory_capacity=64,
            dataset_tree_sha256=dataset_sha,
            require_persistent_root=False,
        )

    fake = SimpleNamespace(
        dataset_tree_sha256=dataset_sha,
        encoder_contract="fixture/v1",
        entries={"sample": object()},
        hidden_size=8,
        maximum_frames=4,
    )
    monkeypatch.setattr(
        training_cli.Vjepa2FeatureCache,
        "load",
        lambda root, **kwargs: fake,
    )
    cache, binding = _load_vjepa2_cache_for_arm(
        arm_spec=_action_arm_spec("D"),
        cache_root=tmp_path,
        cache_manifest_sha256="a" * 64,
        cache_memory_capacity=16,
        dataset_tree_sha256=dataset_sha,
        require_persistent_root=False,
    )
    assert cache is fake
    assert binding == {
        "dataset_tree_sha256": dataset_sha,
        "encoder_contract": "fixture/v1",
        "entries": 1,
        "hidden_size": 8,
        "manifest_sha256": "a" * 64,
        "maximum_frames": 4,
        "root": str(tmp_path.resolve()),
    }


def test_mean_metrics_averages_microbatches_after_schema_handshake() -> None:
    accelerator = _FakeAccelerator()

    result = _mean_metrics(
        accelerator,
        (
            {"action": 1.0, "binding": 4.0},
            {"action": 3.0, "binding": 8.0},
        ),
    )

    assert result == {"action": 2.0, "binding": 6.0}
    assert accelerator.reduce_calls == 2


def test_mean_metrics_rejects_cross_rank_key_drift_before_reduction() -> None:
    accelerator = _FakeAccelerator(mismatched_schema=True)

    with pytest.raises(RuntimeError, match="metric key schema differs across ranks"):
        _mean_metrics(accelerator, ({"action": 1.0},))

    assert accelerator.reduce_calls == 0


def test_mean_metrics_rejects_local_schema_drift_and_nonfinite_values() -> None:
    accelerator = _FakeAccelerator()

    with pytest.raises(RuntimeError, match="schema differs across local microbatches"):
        _mean_metrics(accelerator, ({"action": 1.0}, {"binding": 2.0}))
    with pytest.raises(FloatingPointError, match="non-finite"):
        _mean_metrics(accelerator, ({"action": float("nan")},))
    with pytest.raises(ValueError, match="at least one"):
        _mean_metrics(accelerator, ())

    assert accelerator.reduce_calls == 0


def test_optimizer_and_distributed_system_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        microsteps=(
            SimpleNamespace(loss=3.0, grad_norm=None),
            SimpleNamespace(loss=1.0, grad_norm=4.0),
        )
    )
    optimizer = SimpleNamespace(param_groups=({"lr": 1e-5}, {"lr": 1e-4}))
    assert _optimizer_step_observability(result, optimizer) == {
        "system_optimizer_loss": 2.0,
        "system_synchronized_grad_norm": 4.0,
        "system_learning_rate_min": 1e-5,
        "system_learning_rate_max": 1e-4,
    }

    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 10)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 20)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 40)
    telemetry = _distributed_system_telemetry(
        _FakeAccelerator(),
        elapsed_seconds=2.5,
    )
    assert telemetry["system_train_step_wall_seconds_rank_mean"] == 2.5
    assert telemetry["system_train_step_wall_seconds_rank_max"] == 2.5
    assert telemetry["system_cuda_peak_allocated_bytes_rank_max"] == 30.0


def test_step_timing_synchronizes_only_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "synchronize", observed.append)

    _synchronize_step_timing(SimpleNamespace(device=torch.device("cpu")))
    _synchronize_step_timing(SimpleNamespace(device=torch.device("cuda", 1)))

    assert observed == [torch.device("cuda", 1)]


def test_atomic_json_syncs_parent_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(training_cli, "_fsync_directory", observed.append)
    destination = tmp_path / "run" / "metadata.json"

    _atomic_json(destination, {"status": "PASS"})

    assert destination.read_text() == '{\n  "status": "PASS"\n}\n'
    assert observed == [destination.parent]
    with pytest.raises(FileExistsError):
        _atomic_json(destination, {"status": "PASS"})


def test_fresh_run_metadata_is_published_as_one_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "trial"

    _publish_fresh_run_metadata(
        run_root,
        static_report={"status": "PASS"},
        sample_plan={"plan_sha256": "a" * 64},
    )

    assert json.loads((run_root / "static_preflight.json").read_text()) == {"status": "PASS"}
    assert json.loads((run_root / "sample_plan.json").read_text()) == {"plan_sha256": "a" * 64}
    assert not tuple(run_root.parent.glob(".trial.incomplete-*"))
    with pytest.raises(FileExistsError):
        _publish_fresh_run_metadata(
            run_root,
            static_report={"status": "PASS"},
            sample_plan={"plan_sha256": "a" * 64},
        )


def test_fresh_run_metadata_failure_never_publishes_partial_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs" / "trial"
    real_atomic_json = training_cli._atomic_json
    calls = 0

    def fail_second_write(path: Path, payload: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected sample-plan failure")
        real_atomic_json(path, payload)

    monkeypatch.setattr(training_cli, "_atomic_json", fail_second_write)
    with pytest.raises(OSError, match="injected"):
        _publish_fresh_run_metadata(
            run_root,
            static_report={"status": "PASS"},
            sample_plan={"plan_sha256": "a" * 64},
        )

    assert not run_root.exists()
    incomplete = tuple(run_root.parent.glob(".trial.incomplete-*"))
    assert len(incomplete) == 1
    assert (incomplete[0] / "static_preflight.json").is_file()


def test_cloud_run_root_must_be_below_mnt() -> None:
    _validate_persistent_run_root(Path("/mnt/picf-next/runs/trial"))
    for invalid in (Path("/mnt"), Path("/tmp/trial"), Path("/")):
        with pytest.raises(RuntimeError, match="strict descendant"):
            _validate_persistent_run_root(invalid)


def test_metrics_resume_reconciliation_truncates_uncommitted_tail(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    for step in range(1, 4):
        training_cli._append_metrics(
            path,
            {
                "attempted_optimizer_steps": step,
                "metrics": {"loss": 1.0 / step},
                "optimizer_step_skipped": False,
                "successful_optimizer_steps": step,
            },
        )

    removed = _reconcile_metrics_for_resume(
        path,
        attempted_optimizer_steps=2,
        successful_optimizer_steps=2,
    )

    assert removed == 1
    assert [
        json.loads(line)["attempted_optimizer_steps"] for line in path.read_text().splitlines()
    ] == [1, 2]


def test_metrics_resume_reconciliation_rejects_corrupt_progress(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"attempted_optimizer_steps":1,"metrics":{"loss":1.0},'
        '"optimizer_step_skipped":false,"successful_optimizer_steps":0}\n'
    )

    with pytest.raises(ValueError, match="progression"):
        _reconcile_metrics_for_resume(
            path,
            attempted_optimizer_steps=1,
            successful_optimizer_steps=1,
        )


def test_training_checkpoint_must_equal_recipe_and_accepted_m0(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def validate(checkpoint_dir: object, *, validate_weight_shards: bool) -> dict[str, object]:
        observed.update(
            checkpoint_dir=checkpoint_dir,
            validate_weight_shards=validate_weight_shards,
        )
        return {
            "checkpoint_id": "model/id",
            "checkpoint_revision": "1" * 40,
            "weight_shard_sha256": {"model-00001.safetensors": "a" * 64},
        }

    monkeypatch.setattr(training_cli, "validate_checkpoint", validate)
    _validate_training_checkpoint(
        checkpoint_dir=tmp_path,
        m0_report={"checkpoint_weight_shard_sha256": {"model-00001.safetensors": "a" * 64}},
        checkpoint_id="model/id",
        checkpoint_revision="1" * 40,
    )

    assert observed == {"checkpoint_dir": tmp_path, "validate_weight_shards": True}


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("checkpoint_id", "other/id", "checkpoint id"),
        ("checkpoint_revision", "2" * 40, "checkpoint revision"),
        (
            "weight_shard_sha256",
            {"model-00001.safetensors": "b" * 64},
            "weight shards",
        ),
    ],
)
def test_training_checkpoint_identity_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    assets: dict[str, object] = {
        "checkpoint_id": "model/id",
        "checkpoint_revision": "1" * 40,
        "weight_shard_sha256": {"model-00001.safetensors": "a" * 64},
    }
    assets[field] = bad_value
    monkeypatch.setattr(
        training_cli,
        "validate_checkpoint",
        lambda _path, *, validate_weight_shards: assets,
    )

    with pytest.raises(ValueError, match=message):
        _validate_training_checkpoint(
            checkpoint_dir=tmp_path,
            m0_report={"checkpoint_weight_shard_sha256": {"model-00001.safetensors": "a" * 64}},
            checkpoint_id="model/id",
            checkpoint_revision="1" * 40,
        )
