from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
accelerate = pytest.importorskip("accelerate")

from picf_next.models.temporal import ObjectBeliefBatch, TemporalFilterConfig  # noqa: E402
from picf_next.training import stateful_runner as stateful_runner_module  # noqa: E402
from picf_next.training.accelerate_runner import (  # noqa: E402
    load_accelerate_checkpoint,
    register_progress_for_checkpointing,
    save_accelerate_checkpoint,
)
from picf_next.training.control import (  # noqa: E402
    EpisodeSampleSequence,
    ExperimentRunContract,
    FrozenEpisodeStreamPlan,
    RunProgress,
)
from picf_next.training.stateful_runner import (  # noqa: E402
    StatefulEpisodeTrainingRunner,
    StatefulForwardOutput,
)
from picf_next.training.stream_state import PosteriorStreamStateGroup  # noqa: E402
from tests.geometry_contract import synthetic_geometry_contract  # noqa: E402

GEOMETRY = synthetic_geometry_contract(1)


def _temporal_config() -> TemporalFilterConfig:
    return TemporalFilterConfig(
        address_dim=2,
        content_dim=2,
        geometry_dim=1,
        geometry_contract=GEOMETRY,
        action_dim=1,
        reference_delta_t_s=0.1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )


def _plan(*, total_steps: int = 6) -> FrozenEpisodeStreamPlan:
    episodes = tuple(
        EpisodeSampleSequence(
            episode_key=f"episode-{episode}",
            sample_keys=tuple(
                f"episode-{episode}/transition-{transition}" for transition in range(4)
            ),
        )
        for episode in range(4)
    )
    return FrozenEpisodeStreamPlan(
        dataset_id="stateful-runner-fixture",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        episodes=episodes,
        comparison_id="stateful-runner-seed-43",
        seed=43,
        global_batch_size=4,
        total_steps=total_steps,
    )


def _contract(plan: FrozenEpisodeStreamPlan) -> ExperimentRunContract:
    return ExperimentRunContract.build(
        arm="picf",
        comparison_id=plan.comparison_id,
        code_revision="1" * 40,
        host_name="tiny-stateful-linear",
        host_source_revision="2" * 40,
        training_source_revision="3" * 40,
        foundation_checkpoint_id="fixture/tiny-stateful-linear",
        foundation_checkpoint_revision="fixture-v1",
        checkpoint_manifest_sha256="b" * 64,
        dataset_id=plan.dataset_id,
        dataset_revision=plan.dataset_revision,
        dataset_manifest_sha256=plan.dataset_manifest_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
        world_size=1,
        gradient_accumulation_steps=2,
        precision="float32",
        action_convention="scalar-regression-v1",
        detached_context_frames=0,
        gradient_transitions=1,
        trainable_scope="all",
        common_config={"optimizer": "adamw", "scheduler": "linear"},
        arm_config={"picf": True},
    )


def _progress(plan: FrozenEpisodeStreamPlan, contract_hash: str = "c" * 64) -> RunProgress:
    return RunProgress(
        contract_sha256=contract_hash,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )


def _stream_group(
    plan: FrozenEpisodeStreamPlan,
    *,
    gradient_accumulation_steps: int = 2,
) -> PosteriorStreamStateGroup:
    return PosteriorStreamStateGroup.for_rank_partition(
        _temporal_config(),
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        capacity=2,
        dtype=torch.float32,
        max_parameter_lag=0,
    )


def _next_belief(
    initial: ObjectBeliefBatch,
    prediction: torch.Tensor,
) -> ObjectBeliefBatch:
    delta = prediction.reshape(-1, 1, 1)
    valid = torch.ones_like(initial.valid)
    address = torch.stack((prediction.cos(), prediction.sin()), dim=-1)
    address = address.expand(-1, initial.address_mean.shape[1], -1)
    return ObjectBeliefBatch(
        address_mean=address,
        content_mean=initial.content_mean + 0.5 * delta.expand_as(initial.content_mean),
        geometry_mean=initial.geometry_mean + delta.expand_as(initial.geometry_mean),
        geometry_covariance_diag=torch.full_like(initial.geometry_covariance_diag, 0.2),
        existence_logits=initial.existence_logits + prediction.reshape(-1, 1),
        visibility_given_existence_logits=(
            initial.visibility_given_existence_logits + prediction.reshape(-1, 1)
        ),
        measurement_age_s=initial.measurement_age_s + 0.1,
        valid=valid,
        age=initial.age + 1,
    )


def _forward_for(model: Any, seen_initials: list[ObjectBeliefBatch] | None = None):
    def forward(
        microbatch,
        initial: ObjectBeliefBatch,
        _loss_track_keys_by_row,
    ) -> StatefulForwardOutput:
        if seen_initials is not None:
            seen_initials.append(
                ObjectBeliefBatch(
                    **{
                        field: getattr(initial, field).detach().clone()
                        for field in (
                            "address_mean",
                            "content_mean",
                            "geometry_mean",
                            "geometry_covariance_diag",
                            "existence_logits",
                            "visibility_given_existence_logits",
                            "measurement_age_s",
                            "valid",
                            "age",
                        )
                    }
                )
            )
        x = torch.tensor(
            [transition.sample.sample_index / 10.0 for transition in microbatch.transitions],
            dtype=torch.float32,
        ).reshape(-1, 1)
        prediction = model(x)
        target = 0.25 * x - 0.1
        loss = torch.nn.functional.mse_loss(prediction, target)
        return StatefulForwardOutput(
            loss=loss,
            final_belief=_next_belief(initial, prediction),
            metrics={"mean_sample_index": float(x.mean())},
        )

    return forward


def _tracked_forward_for(
    model: Any,
    seen_tracks: list[tuple[tuple[str | None, ...], ...]] | None = None,
):
    base_forward = _forward_for(model)

    def forward(microbatch, initial, loss_tracks) -> StatefulForwardOutput:
        if seen_tracks is not None:
            seen_tracks.append(loss_tracks)
        output = base_forward(microbatch, initial, loss_tracks)
        final_tracks = tuple(
            (
                f"{transition.episode_instance_id}:object-a",
                f"{transition.episode_instance_id}:object-b",
            )
            for transition in microbatch.transitions
        )
        return StatefulForwardOutput(
            loss=output.loss,
            final_belief=output.final_belief,
            metrics=output.metrics,
            final_loss_track_keys_by_row=final_tracks,
        )

    return forward


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, tuple | list):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_stateful_runner_carries_one_transition_and_tracks_parameter_version() -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    progress = _progress(plan)
    streams = _stream_group(plan)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        plan=plan,
        progress=progress,
        stream_state=streams,
        max_grad_norm=1.0,
    )

    seen_initials: list[ObjectBeliefBatch] = []
    first = runner.run_optimizer_step(_forward_for(model, seen_initials))
    assert first.plan_step == 0
    assert first.parameter_version_before == 0
    assert first.parameter_version_after == 1
    assert not first.optimizer_step_was_skipped
    assert tuple(step.synchronization_boundary for step in first.microsteps) == (False, True)
    assert all(not initial.valid.any() for initial in seen_initials)
    assert progress.attempted_optimizer_steps == 1
    assert not streams.has_pending_chunks

    seen_initials.clear()
    second = runner.run_optimizer_step(_forward_for(model, seen_initials))
    assert second.parameter_version_before == 1
    assert second.parameter_version_after == 2
    assert all(initial.valid.all() for initial in seen_initials)
    assert all(
        stream.next_transition_indices == (2, 2)
        for name in streams.stream_names
        for stream in (streams[name],)
    )
    assert all(
        stream.state_parameter_versions == (0, 0)
        for name in streams.stream_names
        for stream in (streams[name],)
    )
    accelerator.end_training()


def test_stateful_runner_carries_loss_tracks_with_the_same_stream_transaction() -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    streams = _stream_group(plan)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        plan=plan,
        progress=_progress(plan),
        stream_state=streams,
    )
    seen_tracks: list[tuple[tuple[str | None, ...], ...]] = []
    forward = _tracked_forward_for(model, seen_tracks)

    runner.run_optimizer_step(forward)
    committed_after_first = tuple(
        streams[name].loss_track_keys_by_row for name in streams.stream_names
    )
    assert seen_tracks == [((None, None), (None, None))] * 2
    assert all(
        all(key is not None for row in shard for key in row) for shard in committed_after_first
    )

    seen_tracks.clear()
    runner.run_optimizer_step(forward)
    assert tuple(seen_tracks) == committed_after_first
    accelerator.end_training()


def test_stateful_runner_batches_posterior_preparation_agreement(monkeypatch) -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        plan=plan,
        progress=_progress(plan),
        stream_state=_stream_group(plan),
    )
    original = stateful_runner_module.distributed_rank_local_call
    labels = []

    def recording_call(accelerator, *, label, action):
        labels.append(label)
        return original(accelerator, label=label, action=action)

    monkeypatch.setattr(stateful_runner_module, "distributed_rank_local_call", recording_call)
    runner.run_optimizer_step(_forward_for(model))

    assert labels == [
        "posterior preparation for optimizer step",
        "posterior and progress optimizer-step commit",
    ]
    accelerator.end_training()


def test_stateful_runner_poison_on_progress_commit_failure(monkeypatch) -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    progress = _progress(plan)
    streams = _stream_group(plan, gradient_accumulation_steps=1)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        plan=plan,
        progress=progress,
        stream_state=streams,
    )

    def fail_progress(*, optimizer_step_was_skipped: bool) -> None:
        del optimizer_step_was_skipped
        raise RuntimeError("injected progress commit failure")

    monkeypatch.setattr(progress, "advance_optimizer_step", fail_progress)
    with pytest.raises(RuntimeError, match="injected progress commit failure"):
        runner.run_optimizer_step(_forward_for(model))

    assert runner.failed
    assert progress.attempted_optimizer_steps == 0
    assert not streams.has_pending_chunks
    with pytest.raises(RuntimeError, match="restore a completed checkpoint"):
        runner.run_optimizer_step(_forward_for(model))
    accelerator.end_training()


def test_stateful_runner_aborts_all_shards_and_poison_on_late_forward_failure() -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    initial_parameters = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    progress = _progress(plan)
    streams = _stream_group(plan)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        plan=plan,
        progress=progress,
        stream_state=streams,
    )
    valid_forward = _forward_for(model)

    def fail_second(microbatch, initial, loss_tracks):
        if microbatch.accumulation_index == 1:
            raise RuntimeError("injected late shard failure")
        return valid_forward(microbatch, initial, loss_tracks)

    with pytest.raises(RuntimeError, match="injected late shard failure"):
        runner.run_optimizer_step(fail_second)
    assert runner.failed
    assert not streams.has_pending_chunks
    assert progress.attempted_optimizer_steps == 0
    assert all(streams[name].next_transition_indices == (0, 0) for name in streams.stream_names)
    assert all(not streams[name].belief.valid.any() for name in streams.stream_names)
    _assert_nested_equal(accelerator.unwrap_model(model).state_dict(), initial_parameters)
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(RuntimeError, match="restore a completed checkpoint"):
        runner.run_optimizer_step(valid_forward)
    accelerator.end_training()


def test_late_forward_failure_does_not_commit_an_earlier_shards_loss_tracks() -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    streams = _stream_group(plan)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        plan=plan,
        progress=_progress(plan),
        stream_state=streams,
    )
    valid_forward = _tracked_forward_for(model)

    def fail_second(microbatch, initial, loss_tracks):
        if microbatch.accumulation_index == 1:
            raise RuntimeError("injected late loss-track failure")
        return valid_forward(microbatch, initial, loss_tracks)

    with pytest.raises(RuntimeError, match="injected late loss-track failure"):
        runner.run_optimizer_step(fail_second)

    assert all(
        streams[name].loss_track_keys_by_row == ((None, None), (None, None))
        for name in streams.stream_names
    )
    assert all(streams[name].next_transition_indices == (0, 0) for name in streams.stream_names)
    accelerator.end_training()


def test_stateful_runner_rejects_nonfinite_posterior_before_backward() -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=1)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    progress = _progress(plan)
    streams = _stream_group(plan, gradient_accumulation_steps=1)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        plan=plan,
        progress=progress,
        stream_state=streams,
    )
    valid_forward = _forward_for(model)

    def nonfinite_state(microbatch, initial, loss_tracks):
        output = valid_forward(microbatch, initial, loss_tracks)
        output.final_belief.geometry_covariance_diag[0, 0, 0] = float("nan")
        return output

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        runner.run_optimizer_step(nonfinite_state)
    assert runner.failed
    assert progress.attempted_optimizer_steps == 0
    assert not streams.has_pending_chunks
    accelerator.end_training()


def test_stateful_runner_checkpoint_reproduces_next_posterior_loss_and_update(
    tmp_path: Path,
) -> None:
    plan = _plan()
    contract = _contract(plan)
    accelerator = accelerate.Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )
    torch.manual_seed(71)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.05 * step,
    )
    progress = _progress(plan, contract.contract_sha256)
    streams = _stream_group(plan)
    register_progress_for_checkpointing(accelerator, progress)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    runner = StatefulEpisodeTrainingRunner(
        accelerator=accelerator,
        model=model,
        state_producer=torch.nn.Identity(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        plan=plan,
        progress=progress,
        stream_state=streams,
        max_grad_norm=1.0,
    )

    for _ in range(2):
        runner.run_optimizer_step(_forward_for(model))
    checkpoint = save_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=tmp_path / "checkpoint-000002",
        contract=contract,
        plan=plan,
        progress=progress,
        rank_state=streams,
    )
    uninterrupted_outputs = tuple(runner.run_optimizer_step(_forward_for(model)) for _ in range(2))
    uninterrupted_losses = tuple(
        tuple(microstep.loss for microstep in output.microsteps) for output in uninterrupted_outputs
    )
    uninterrupted_model = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
    uninterrupted_scheduler = copy.deepcopy(scheduler.state_dict())
    uninterrupted_progress = copy.deepcopy(progress.state_dict())
    uninterrupted_streams = copy.deepcopy(streams.state_dict())

    load_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=checkpoint,
        contract=contract,
        plan=plan,
        progress=progress,
        rank_state=streams,
    )
    resumed_outputs = tuple(runner.run_optimizer_step(_forward_for(model)) for _ in range(2))
    resumed_losses = tuple(
        tuple(microstep.loss for microstep in output.microsteps) for output in resumed_outputs
    )

    assert resumed_losses == uninterrupted_losses
    _assert_nested_equal(accelerator.unwrap_model(model).state_dict(), uninterrupted_model)
    _assert_nested_equal(optimizer.state_dict(), uninterrupted_optimizer)
    _assert_nested_equal(scheduler.state_dict(), uninterrupted_scheduler)
    assert progress.state_dict() == uninterrupted_progress
    _assert_nested_equal(streams.state_dict(), uninterrupted_streams)
    accelerator.end_training()


def test_stateful_runner_rejects_a_trainable_state_producer() -> None:
    plan = _plan()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)

    with pytest.raises(ValueError, match="requires a frozen state producer"):
        StatefulEpisodeTrainingRunner(
            accelerator=accelerator,
            model=model,
            state_producer=torch.nn.Linear(1, 1),
            optimizer=optimizer,
            plan=plan,
            progress=_progress(plan),
            stream_state=_stream_group(plan),
        )
    accelerator.end_training()
