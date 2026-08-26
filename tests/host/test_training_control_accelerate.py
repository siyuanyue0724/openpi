from __future__ import annotations

import copy
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
accelerate = pytest.importorskip("accelerate")

from picf_next.training.accelerate_runner import (  # noqa: E402
    accelerated_microstep,
    load_accelerate_checkpoint,
    register_progress_for_checkpointing,
    save_accelerate_checkpoint,
)
from picf_next.training.control import (  # noqa: E402
    ExperimentRunContract,
    FrozenSamplePlan,
    RunProgress,
    write_control_manifest,
)


class _RankState:
    def __init__(self, value: int = 0) -> None:
        self.value = torch.tensor(value, dtype=torch.int64)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    @staticmethod
    def validate_state_dict(state: object) -> None:
        if not isinstance(state, dict) or set(state) != {"value"}:
            raise ValueError("rank state is malformed")
        value = state["value"]
        if not isinstance(value, torch.Tensor) or value.shape or value.dtype != torch.int64:
            raise ValueError("rank state value is malformed")

    def load_state_dict(self, state: object) -> None:
        self.validate_state_dict(state)
        assert isinstance(state, dict)
        value = state["value"]
        assert isinstance(value, torch.Tensor)
        self.value = value.clone()


def _plan_and_contract() -> tuple[FrozenSamplePlan, ExperimentRunContract]:
    plan = FrozenSamplePlan(
        dataset_id="accelerate-fixture",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        sample_keys=tuple(f"sample-{index:03d}" for index in range(13)),
        comparison_id="accelerate-resume-seed-19",
        seed=19,
        global_batch_size=4,
        total_steps=6,
    )
    contract = ExperimentRunContract.build(
        arm="picf",
        comparison_id=plan.comparison_id,
        code_revision="1" * 40,
        host_name="tiny-linear",
        host_source_revision="2" * 40,
        training_source_revision="3" * 40,
        foundation_checkpoint_id="fixture/tiny-linear",
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
    return plan, contract


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


def test_accelerate_checkpoint_reproduces_next_updates_and_rng(tmp_path: Path) -> None:
    plan, contract = _plan_and_contract()
    accelerator = accelerate.Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )
    torch.manual_seed(29)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.05 * step,
    )
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    rank_state = _RankState()
    register_progress_for_checkpointing(accelerator, progress)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    def run_optimizer_step(step: int) -> tuple[str, ...]:
        observed = []
        for accumulation_index in range(2):
            microbatch = plan.microbatch_for_rank(
                step,
                rank=0,
                world_size=1,
                gradient_accumulation_steps=2,
                accumulation_index=accumulation_index,
            )
            observed.extend(sample.sample_key for sample in microbatch.samples)

            def forward_loss(planned=microbatch):
                values = []
                for sample in planned.samples:
                    generator = torch.Generator(device="cpu").manual_seed(sample.flow_noise_seed)
                    noise = torch.rand((), generator=generator) * 0.01
                    values.append(float(sample.sample_index) / 10.0 + noise)
                x = torch.stack([torch.as_tensor(value) for value in values]).reshape(-1, 1)
                prediction = model(x)
                target = 0.3 * x + 0.1
                return torch.nn.functional.mse_loss(prediction, target)

            result = accelerated_microstep(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                forward_loss=forward_loss,
                lr_scheduler=scheduler,
                max_grad_norm=1.0,
            )
        assert result.synchronization_boundary
        progress.advance_optimizer_step(
            optimizer_step_was_skipped=result.optimizer_step_was_skipped
        )
        rank_state.value.add_(1)
        return tuple(observed)

    for step in range(2):
        run_optimizer_step(step)
    checkpoint = save_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=tmp_path / "checkpoint-000002",
        contract=contract,
        plan=plan,
        progress=progress,
        rank_state=rank_state,
    )
    expected_rng_after_load = torch.rand(5)
    uninterrupted_trace = tuple(run_optimizer_step(step) for step in range(2, 4))
    uninterrupted_model = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
    uninterrupted_scheduler = copy.deepcopy(scheduler.state_dict())
    uninterrupted_progress = copy.deepcopy(progress.state_dict())
    uninterrupted_rank_state = rank_state.value.clone()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(17.0)
    progress.advance_optimizer_step(optimizer_step_was_skipped=True)
    rank_state.value.fill_(999)
    torch.manual_seed(999)

    with pytest.raises(ValueError, match="another run contract"):
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=checkpoint,
            contract=replace(contract, precision="bfloat16"),
            plan=plan,
            progress=progress,
            rank_state=rank_state,
        )

    with pytest.raises(ValueError, match="rank-local state files"):
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=checkpoint,
            contract=contract,
            plan=plan,
            progress=progress,
        )
    assert rank_state.value.item() == 999

    corrupt_checkpoint = tmp_path / "checkpoint-corrupt-rank-state"
    shutil.copytree(checkpoint, corrupt_checkpoint)
    torch.save(
        {"value": torch.tensor(1.0)},
        corrupt_checkpoint / "picf_rank_state_00000.pt",
    )
    (corrupt_checkpoint / "picf_control.json").unlink()
    write_control_manifest(
        corrupt_checkpoint / "picf_control.json",
        contract=contract,
        plan=plan,
        progress=progress,
    )
    model_before_rejected_load = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    with pytest.raises(RuntimeError, match="checkpoint state prevalidation"):
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=corrupt_checkpoint,
            contract=contract,
            plan=plan,
            progress=progress,
            rank_state=rank_state,
        )
    _assert_nested_equal(
        accelerator.unwrap_model(model).state_dict(),
        model_before_rejected_load,
    )
    assert rank_state.value.item() == 999

    load_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=checkpoint,
        contract=contract,
        plan=plan,
        progress=progress,
        rank_state=rank_state,
    )
    assert rank_state.value.item() == 2
    torch.testing.assert_close(torch.rand(5), expected_rng_after_load, rtol=0.0, atol=0.0)
    resumed_trace = tuple(run_optimizer_step(step) for step in range(2, 4))

    assert resumed_trace == uninterrupted_trace
    _assert_nested_equal(accelerator.unwrap_model(model).state_dict(), uninterrupted_model)
    _assert_nested_equal(optimizer.state_dict(), uninterrupted_optimizer)
    _assert_nested_equal(scheduler.state_dict(), uninterrupted_scheduler)
    assert progress.state_dict() == uninterrupted_progress
    torch.testing.assert_close(rank_state.value, uninterrupted_rank_state, rtol=0.0, atol=0.0)

    invalid_destination = tmp_path / "invalid-rank-state"
    with pytest.raises(TypeError, match="state_dict"):
        save_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=invalid_destination,
            contract=contract,
            plan=plan,
            progress=progress,
            rank_state=object(),
        )
    assert not invalid_destination.exists()
    assert not invalid_destination.with_name(f".{invalid_destination.name}.incomplete").exists()
    with pytest.raises(TypeError, match="load_state_dict"):
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=checkpoint,
            contract=contract,
            plan=plan,
            progress=progress,
            rank_state=object(),
        )
    invalid_loader = _RankState()
    invalid_loader.validate_state_dict = None  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="validate_state_dict"):
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=checkpoint,
            contract=contract,
            plan=plan,
            progress=progress,
            rank_state=invalid_loader,
        )
    accelerator.end_training()


def test_accelerated_microstep_preserves_local_forward_exception_type() -> None:
    accelerator = accelerate.Accelerator(cpu=True)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)

    def fail_forward() -> torch.Tensor:
        raise LookupError("injected local data failure")

    with pytest.raises(LookupError, match="injected local data failure"):
        accelerated_microstep(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            forward_loss=fail_forward,
        )
    accelerator.end_training()


def test_accelerated_microstep_preserves_local_scheduler_failure() -> None:
    accelerator = accelerate.Accelerator(cpu=True)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)

    class FailingScheduler:
        @staticmethod
        def step() -> None:
            raise LookupError("injected local scheduler failure")

    with pytest.raises(LookupError, match="injected local scheduler failure"):
        accelerated_microstep(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            forward_loss=lambda: model(torch.ones(1, 1)).square().mean(),
            lr_scheduler=FailingScheduler(),
        )
    accelerator.end_training()
