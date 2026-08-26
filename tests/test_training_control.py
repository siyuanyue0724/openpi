from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import picf_next.training.control as control_module
from picf_next.training.control import (
    EXPERIMENT_ARMS,
    EpisodeSampleSequence,
    ExperimentRunContract,
    FrozenEpisodeStreamPlan,
    FrozenSamplePlan,
    RunProgress,
    TrainingPlan,
    derive_subseed,
    validate_control_manifest,
    validate_matched_abc,
    write_control_manifest,
)

_DATA_SHA = "a" * 64
_CHECKPOINT_SHA = "b" * 64
_CODE_REVISION = "1" * 40
_HOST_REVISION = "2" * 40
_TRAINING_REVISION = "3" * 40


def _plan(*, comparison_id: str = "matched-seed-7") -> FrozenSamplePlan:
    return FrozenSamplePlan(
        dataset_id="fixture",
        dataset_revision="fixture-v1",
        dataset_manifest_sha256=_DATA_SHA,
        sample_keys=tuple(f"episode-00/frame-{index:04d}" for index in range(19)),
        comparison_id=comparison_id,
        seed=7,
        global_batch_size=8,
        total_steps=9,
    )


def _contract(
    plan: TrainingPlan,
    arm: str,
    *,
    common_config: dict | None = None,
    arm_config: dict | None = None,
) -> ExperimentRunContract:
    return ExperimentRunContract.build(
        arm=arm,
        comparison_id=plan.comparison_id,
        code_revision=_CODE_REVISION,
        host_name="MolmoAct2",
        host_source_revision=_HOST_REVISION,
        training_source_revision=_TRAINING_REVISION,
        foundation_checkpoint_id="allenai/MolmoAct2",
        foundation_checkpoint_revision="e432d85f6e039edca44afb93c262f3084ab72a9c",
        checkpoint_manifest_sha256=_CHECKPOINT_SHA,
        dataset_id=plan.dataset_id,
        dataset_revision=plan.dataset_revision,
        dataset_manifest_sha256=plan.dataset_manifest_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
        world_size=2,
        gradient_accumulation_steps=2,
        precision="bfloat16",
        action_convention="molmoact2-continuous-v1",
        detached_context_frames=2,
        gradient_transitions=2,
        trainable_scope="action-expert-picf-adapters",
        common_config=common_config or {"optimizer": {"name": "adamw", "lr": 1e-4}},
        arm_config=arm_config or {"arm": arm},
    )


def test_frozen_plan_is_random_access_deterministic_and_epoch_varying() -> None:
    first = _plan()
    second = _plan()
    assert first.plan_sha256 == second.plan_sha256
    assert first.global_batch(0) == first.global_batch(0) == second.global_batch(0)
    assert len({sample.sample_key for sample in first.global_batch(0).samples}) == 8
    assert first.global_batch(0).samples != first.global_batch(first.batches_per_epoch).samples
    assert all(0 <= sample.flow_noise_seed < 2**63 for sample in first.global_batch(0).samples)
    sample = first.global_batch(0).samples[0]
    assert sample.flow_noise_seed != sample.flow_timestep_seed
    assert derive_subseed(sample.flow_noise_seed, "transition", "0") == derive_subseed(
        sample.flow_noise_seed, "transition", "0"
    )
    assert derive_subseed(sample.flow_noise_seed, "transition", "0") != derive_subseed(
        sample.flow_noise_seed, "transition", "1"
    )


def test_global_plan_is_invariant_to_rank_and_accumulation_partition() -> None:
    plan = _plan()
    expected = plan.global_batch(3).samples
    two_by_two = tuple(
        sample
        for accumulation_index in range(2)
        for rank in range(2)
        for sample in plan.microbatch_for_rank(
            3,
            rank=rank,
            world_size=2,
            gradient_accumulation_steps=2,
            accumulation_index=accumulation_index,
        ).samples
    )
    four_by_one = tuple(
        sample
        for rank in range(4)
        for sample in plan.microbatch_for_rank(
            3,
            rank=rank,
            world_size=4,
            gradient_accumulation_steps=1,
            accumulation_index=0,
        ).samples
    )
    assert two_by_two == expected
    assert four_by_one == expected


def test_plan_rejects_invalid_partition_and_duplicate_sample_keys() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="divisible"):
        plan.microbatch_for_rank(
            0,
            rank=0,
            world_size=3,
            gradient_accumulation_steps=1,
            accumulation_index=0,
        )
    with pytest.raises(ValueError, match="unique"):
        replace(plan, sample_keys=("same", "same", *(f"key-{index}" for index in range(17))))
    with pytest.raises(ValueError, match="global_batch_size"):
        replace(plan, global_batch_size=True)
    with pytest.raises(ValueError, match="total_steps"):
        replace(plan, total_steps=True)
    with pytest.raises(IndexError, match="optimizer_step"):
        plan.global_batch(True)
    with pytest.raises(ValueError, match="world_size"):
        plan.microbatch_for_rank(
            0,
            rank=0,
            world_size=True,
            gradient_accumulation_steps=1,
            accumulation_index=0,
        )


def test_plan_metadata_round_trip_fails_on_reordered_manifest(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "sample-plan.json"
    plan.write_metadata(path)
    loaded = FrozenSamplePlan.from_metadata(path, sample_keys=plan.sample_keys)
    assert loaded.plan_sha256 == plan.plan_sha256
    with pytest.raises(ValueError, match="ordered sample manifest"):
        FrozenSamplePlan.from_metadata(path, sample_keys=tuple(reversed(plan.sample_keys)))

    payload = json.loads(path.read_text())
    payload["metadata"]["total_steps"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        FrozenSamplePlan.from_metadata(path, sample_keys=plan.sample_keys)


def test_plan_metadata_publication_rejects_existing_or_stale_state(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "sample-plan.json"
    plan.write_metadata(path)

    with pytest.raises(FileExistsError):
        plan.write_metadata(path)

    path.unlink()
    stale = path.with_name(f".{path.name}.tmp-stale-process")
    stale.write_text("partial")
    with pytest.raises(FileExistsError):
        plan.write_metadata(path)
    assert stale.read_text() == "partial"


def test_matched_abc_requires_identical_common_contract() -> None:
    plan = _plan()
    contracts = tuple(_contract(plan, arm) for arm in sorted(EXPERIMENT_ARMS))
    assert validate_matched_abc(contracts) == contracts[0].fairness_sha256
    mismatched = (*contracts[:2], replace(contracts[2], precision="float32"))
    with pytest.raises(ValueError, match="not identical"):
        validate_matched_abc(mismatched)
    with pytest.raises(ValueError, match="vanilla"):
        validate_matched_abc((*contracts[:2], contracts[1]))


def test_run_contract_validates_plan_and_finite_json() -> None:
    plan = _plan()
    contract = _contract(plan, "picf")
    contract.validate_plan(plan)
    with pytest.raises(ValueError, match="hashes differ"):
        replace(contract, sample_plan_sha256="c" * 64).validate_plan(plan)
    with pytest.raises(ValueError, match="finite canonical JSON"):
        _contract(plan, "picf", common_config={"bad": float("nan")})
    with pytest.raises(ValueError, match="world_size"):
        replace(contract, world_size=True)


def test_progress_counts_attempts_successes_and_consumed_samples() -> None:
    plan = _plan()
    contract = _contract(plan, "picf")
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    progress.advance_optimizer_step(optimizer_step_was_skipped=False)
    progress.advance_optimizer_step(optimizer_step_was_skipped=True)
    assert progress.next_plan_step == 2
    assert progress.successful_optimizer_steps == 1
    assert progress.consumed_global_samples == 16

    restored = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    restored.load_state_dict(progress.state_dict())
    assert restored.state_dict() == progress.state_dict()

    corrupt = progress.state_dict()
    corrupt["consumed_global_samples"] += 1
    with pytest.raises(ValueError, match="consumed-sample"):
        restored.load_state_dict(corrupt)


def test_control_manifest_detects_contract_plan_and_payload_corruption(tmp_path: Path) -> None:
    plan = _plan()
    contract = _contract(plan, "picf")
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    progress.advance_optimizer_step(optimizer_step_was_skipped=False)
    state_path = tmp_path / "model.safetensors"
    state_path.write_bytes(b"serialized model state")
    path = tmp_path / "picf_control.json"
    write_control_manifest(path, contract=contract, plan=plan, progress=progress)
    assert validate_control_manifest(path, contract=contract, plan=plan) == progress.state_dict()

    payload = json.loads(path.read_text())
    payload["contract"]["precision"] = "float32"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="payload is corrupt"):
        validate_control_manifest(path, contract=contract, plan=plan)


def test_control_json_unpublishes_after_post_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "picf_control.json"
    replaced = False
    original_replace = control_module.os.replace
    original_fsync = control_module.os.fsync

    def track_replace(source: Path, destination: Path) -> None:
        nonlocal replaced
        original_replace(source, destination)
        replaced = True

    def fail_parent_fsync(descriptor: int) -> None:
        if replaced:
            raise OSError("injected post-rename fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(control_module.os, "replace", track_replace)
    monkeypatch.setattr(control_module.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="post-rename"):
        control_module._atomic_write_json(output, {"schema": 1})

    assert not output.exists()
    assert not tuple(tmp_path.glob(".picf_control.json.tmp-*"))


def test_control_json_detects_completed_rename_and_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "picf_control.json"
    original_replace = control_module.os.replace

    def replace_then_raise(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        raise OSError("injected exception after rename syscall")

    monkeypatch.setattr(control_module.os, "replace", replace_then_raise)
    with pytest.raises(OSError, match="after rename syscall"):
        control_module._atomic_write_json(output, {"schema": 1})
    assert not output.exists()

    target = tmp_path / "target.json"
    target.write_text("do not replace\n", encoding="ascii")
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        control_module._atomic_write_json(output, {"schema": 1})
    assert target.read_text(encoding="ascii") == "do not replace\n"


def test_control_manifest_detects_serialized_state_corruption(tmp_path: Path) -> None:
    plan = _plan()
    contract = _contract(plan, "picf")
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    state_path = tmp_path / "optimizer.bin"
    state_path.write_bytes(b"optimizer-state-v1")
    path = tmp_path / "picf_control.json"
    write_control_manifest(path, contract=contract, plan=plan, progress=progress)

    state_path.write_bytes(b"optimizer-state-v2")
    with pytest.raises(ValueError, match="state files are missing, added, or corrupt"):
        validate_control_manifest(path, contract=contract, plan=plan)


def test_control_manifest_detects_added_or_removed_state_files(tmp_path: Path) -> None:
    plan = _plan()
    contract = _contract(plan, "picf")
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    state_path = tmp_path / "random_states_0.pkl"
    state_path.write_bytes(b"rng")
    path = tmp_path / "picf_control.json"
    write_control_manifest(path, contract=contract, plan=plan, progress=progress)

    extra = tmp_path / "unexpected.bin"
    extra.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="state files are missing, added, or corrupt"):
        validate_control_manifest(path, contract=contract, plan=plan)
    extra.unlink()
    state_path.unlink()
    with pytest.raises(ValueError, match="no serialized state files"):
        validate_control_manifest(path, contract=contract, plan=plan)


def test_control_manifest_accepts_the_frozen_episode_stream_plan(tmp_path: Path) -> None:
    episodes = tuple(
        EpisodeSampleSequence(
            f"episode-{episode}",
            tuple(f"episode-{episode}/frame-{frame}" for frame in range(3 + episode)),
        )
        for episode in range(4)
    )
    plan = FrozenEpisodeStreamPlan(
        dataset_id="fixture",
        dataset_revision="fixture-v1",
        dataset_manifest_sha256=_DATA_SHA,
        episodes=episodes,
        comparison_id="matched-stream-seed-7",
        seed=7,
        global_batch_size=4,
        total_steps=9,
    )
    contract = replace(
        _contract(plan, "picf"),
        detached_context_frames=0,
        gradient_transitions=1,
    )
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    (tmp_path / "model.safetensors").write_bytes(b"state")
    control = tmp_path / "picf_control.json"
    write_control_manifest(control, contract=contract, plan=plan, progress=progress)
    assert (
        validate_control_manifest(
            control,
            contract=contract,
            plan=plan,
        )
        == progress.state_dict()
    )
