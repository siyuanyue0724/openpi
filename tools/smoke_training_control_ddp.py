#!/usr/bin/env python3
"""Two-process CPU proof for topology-invariant plans and exact resume."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# This executable is an intentionally CPU-only distributed proof.  Hide any
# workstation GPU before importing torch so a one-GPU host cannot make rank 1
# pass an invalid CUDA device ordinal to Accelerate's process barrier.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.distributed as dist
from accelerate import Accelerator

from picf_next.training.accelerate_runner import (
    accelerated_microstep,
    distributed_rank_local_call,
    load_accelerate_checkpoint,
    register_progress_for_checkpointing,
    save_accelerate_checkpoint,
)
from picf_next.training.control import ExperimentRunContract, FrozenSamplePlan, RunProgress


class _RankCursorState:
    def __init__(self, rank: int) -> None:
        self.value = torch.tensor(rank * 1000, dtype=torch.int64)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    @staticmethod
    def _validated_value(state: object) -> torch.Tensor:
        if not isinstance(state, dict) or set(state) != {"value"}:
            raise ValueError("rank cursor state is malformed")
        value = state["value"]
        if not isinstance(value, torch.Tensor) or value.shape or value.dtype != torch.int64:
            raise ValueError("rank cursor value is malformed")
        return value

    def validate_state_dict(self, state: object) -> None:
        self._validated_value(state)

    def load_state_dict(self, state: object) -> None:
        self.value = self._validated_value(state).clone()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _plan_and_contract(world_size: int) -> tuple[FrozenSamplePlan, ExperimentRunContract]:
    plan = FrozenSamplePlan(
        dataset_id="ddp-fixture",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        sample_keys=tuple(f"episode-{index // 8:02d}/frame-{index:04d}" for index in range(29)),
        comparison_id="ddp-resume-seed-31",
        seed=31,
        global_batch_size=8,
        total_steps=6,
    )
    contract = ExperimentRunContract.build(
        arm="picf",
        comparison_id=plan.comparison_id,
        code_revision="1" * 40,
        host_name="tiny-ddp-linear",
        host_source_revision="2" * 40,
        training_source_revision="3" * 40,
        foundation_checkpoint_id="fixture/tiny-ddp-linear",
        foundation_checkpoint_revision="fixture-v1",
        checkpoint_manifest_sha256="b" * 64,
        dataset_id=plan.dataset_id,
        dataset_revision=plan.dataset_revision,
        dataset_manifest_sha256=plan.dataset_manifest_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
        world_size=world_size,
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
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(
                left.detach().cpu().contiguous(),
                right.detach().cpu().contiguous(),
            )
        ):
            raise AssertionError("tensor state differs after checkpoint resume")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError("mapping state differs after checkpoint resume")
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, tuple | list):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError("sequence state differs after checkpoint resume")
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    elif left != right:
        raise AssertionError("scalar state differs after checkpoint resume")


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _sample_tensor(samples: tuple[Any, ...]) -> torch.Tensor:
    values = []
    for sample in samples:
        generator = torch.Generator(device="cpu").manual_seed(sample.flow_noise_seed)
        noise = torch.rand((), generator=generator) * 0.01
        values.append(float(sample.sample_index) / 10.0 + noise)
    return torch.stack([torch.as_tensor(value) for value in values]).reshape(-1, 1)


def _single_process_reference(plan: FrozenSamplePlan) -> dict[str, torch.Tensor]:
    torch.manual_seed(37)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.04 * step,
    )
    for step in range(4):
        x = _sample_tensor(plan.global_batch(step).samples)
        prediction = model(x)
        target = -0.2 * x + 0.4
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return copy.deepcopy(model.state_dict())


def main() -> None:
    args = _parse_args()
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )
    if accelerator.num_processes != 2:
        raise RuntimeError("this smoke must run under exactly two distributed processes")
    plan, contract = _plan_and_contract(accelerator.num_processes)
    torch.manual_seed(37)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.04 * step,
    )
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=plan.plan_sha256,
        optimizer_global_batch_size=plan.global_batch_size,
    )
    rank_state = _RankCursorState(accelerator.process_index)
    register_progress_for_checkpointing(accelerator, progress)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    def run_optimizer_step(step: int) -> tuple[str, ...]:
        observed_global: list[str] = []
        for accumulation_index in range(2):
            microbatch = plan.microbatch_for_rank(
                step,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
                gradient_accumulation_steps=2,
                accumulation_index=accumulation_index,
            )
            local_keys = [sample.sample_key for sample in microbatch.samples]
            gathered: list[list[str] | None] = [None] * accelerator.num_processes
            dist.all_gather_object(gathered, local_keys)
            for rank_keys in gathered:
                if rank_keys is None:
                    raise AssertionError("distributed sample-key gather is incomplete")
                observed_global.extend(rank_keys)

            def forward_loss(planned=microbatch):
                x = _sample_tensor(planned.samples)
                prediction = model(x)
                target = -0.2 * x + 0.4
                return torch.nn.functional.mse_loss(prediction, target)

            result = accelerated_microstep(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                forward_loss=forward_loss,
                lr_scheduler=scheduler,
                max_grad_norm=1.0,
            )
        expected = tuple(sample.sample_key for sample in plan.global_batch(step).samples)
        if tuple(observed_global) != expected:
            raise AssertionError("rank/accumulation shards do not reconstruct the global plan")
        if not result.synchronization_boundary:
            raise AssertionError("optimizer step did not end at a synchronization boundary")
        progress.advance_optimizer_step(
            optimizer_step_was_skipped=result.optimizer_step_was_skipped
        )
        rank_state.value.add_(1)
        return expected

    for step in range(2):
        run_optimizer_step(step)
    checkpoint = save_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=args.output_dir / "checkpoint-000002",
        contract=contract,
        plan=plan,
        progress=progress,
        rank_state=rank_state,
    )
    if accelerator.is_main_process:
        for rank in range(accelerator.num_processes):
            payload = torch.load(
                checkpoint / f"picf_rank_state_{rank:05d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if int(payload["value"].item()) != rank * 1000 + 2:
                raise AssertionError("checkpoint did not preserve distinct rank-local cursors")
    accelerator.wait_for_everyone()
    expected_rng_after_load = torch.rand(5)
    uninterrupted_trace = tuple(run_optimizer_step(step) for step in range(2, 4))
    uninterrupted_model = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
    uninterrupted_scheduler = copy.deepcopy(scheduler.state_dict())
    uninterrupted_progress = copy.deepcopy(progress.state_dict())
    uninterrupted_rank_state = rank_state.value.clone()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(11.0)
    progress.advance_optimizer_step(optimizer_step_was_skipped=True)
    rank_state.value.fill_(-1)
    torch.manual_seed(12345 + accelerator.process_index)

    load_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=checkpoint,
        contract=contract,
        plan=plan,
        progress=progress,
        rank_state=rank_state,
    )
    if int(rank_state.value.item()) != accelerator.process_index * 1000 + 2:
        raise AssertionError("rank-local cursor did not resume exactly")
    if not torch.equal(torch.rand(5), expected_rng_after_load):
        raise AssertionError("per-rank torch RNG did not resume exactly")
    resumed_trace = tuple(run_optimizer_step(step) for step in range(2, 4))
    if resumed_trace != uninterrupted_trace:
        raise AssertionError("resumed sample/noise trace differs from uninterrupted trace")
    _assert_nested_equal(accelerator.unwrap_model(model).state_dict(), uninterrupted_model)
    _assert_nested_equal(optimizer.state_dict(), uninterrupted_optimizer)
    _assert_nested_equal(scheduler.state_dict(), uninterrupted_scheduler)
    if progress.state_dict() != uninterrupted_progress:
        raise AssertionError("resumed progress differs from uninterrupted progress")
    if not torch.equal(rank_state.value, uninterrupted_rank_state):
        raise AssertionError("resumed rank-local cursor differs from uninterrupted cursor")

    reference_state = _single_process_reference(plan)
    for name, parameter in accelerator.unwrap_model(model).state_dict().items():
        torch.testing.assert_close(parameter, reference_state[name], rtol=1e-6, atol=1e-7)

    checkpoint_collision_failed_closed = False
    try:
        save_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=checkpoint,
            contract=contract,
            plan=plan,
            progress=progress,
            rank_state=rank_state,
        )
    except FileExistsError:
        checkpoint_collision_failed_closed = True
    collision_count = accelerator.reduce(
        torch.tensor(int(checkpoint_collision_failed_closed), dtype=torch.int64),
        reduction="sum",
    )
    if int(collision_count.item()) != accelerator.num_processes:
        raise AssertionError("checkpoint collision did not fail on every rank")

    invalid_manifest_failed_closed = False
    try:
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=args.output_dir / "missing-checkpoint",
            contract=contract,
            plan=plan,
            progress=progress,
            rank_state=rank_state,
        )
    except RuntimeError as exc:
        invalid_manifest_failed_closed = "rank-zero checkpoint control validation failed" in str(
            exc
        )
    invalid_manifest_count = accelerator.reduce(
        torch.tensor(int(invalid_manifest_failed_closed), dtype=torch.int64),
        reduction="sum",
    )
    if int(invalid_manifest_count.item()) != accelerator.num_processes:
        raise AssertionError("invalid checkpoint manifest did not fail on every rank")

    nonfinite_failed_closed = False

    def rank_local_nonfinite_loss() -> torch.Tensor:
        finite = accelerator.unwrap_model(model).weight.sum() * 0.0
        if accelerator.process_index == 1:
            return finite + torch.tensor(float("nan"))
        return finite

    try:
        accelerated_microstep(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            forward_loss=rank_local_nonfinite_loss,
            lr_scheduler=scheduler,
            max_grad_norm=1.0,
        )
    except FloatingPointError:
        nonfinite_failed_closed = True
    nonfinite_count = accelerator.reduce(
        torch.tensor(int(nonfinite_failed_closed), dtype=torch.int64),
        reduction="sum",
    )
    if int(nonfinite_count.item()) != accelerator.num_processes:
        raise AssertionError("rank-local non-finite loss did not fail on every rank")

    rank_local_forward_error_failed_closed = False

    def rank_local_forward_error() -> torch.Tensor:
        if accelerator.process_index == 1:
            raise RuntimeError("injected rank-local forward failure")
        return accelerator.unwrap_model(model).weight.sum() * 0.0

    try:
        accelerated_microstep(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            forward_loss=rank_local_forward_error,
            lr_scheduler=scheduler,
            max_grad_norm=1.0,
        )
    except FloatingPointError as exc:
        rank_local_forward_error_failed_closed = "injected rank-local forward failure" in str(exc)
    forward_error_count = accelerator.reduce(
        torch.tensor(int(rank_local_forward_error_failed_closed), dtype=torch.int64),
        reduction="sum",
    )
    if int(forward_error_count.item()) != accelerator.num_processes:
        raise AssertionError("rank-local forward exception did not fail on every rank")

    rank_local_prepare_error_failed_closed = False

    def rank_local_prepare_error() -> int:
        if accelerator.process_index == 1:
            raise ValueError("injected rank-local posterior preparation failure")
        return 1

    try:
        distributed_rank_local_call(
            accelerator,
            label="injected posterior preparation",
            action=rank_local_prepare_error,
        )
    except RuntimeError as exc:
        rank_local_prepare_error_failed_closed = (
            "injected rank-local posterior preparation failure" in str(exc)
        )
    prepare_error_count = accelerator.reduce(
        torch.tensor(int(rank_local_prepare_error_failed_closed), dtype=torch.int64),
        reduction="sum",
    )
    if int(prepare_error_count.item()) != accelerator.num_processes:
        raise AssertionError("rank-local preparation exception did not fail on every rank")

    rank_local_update_error_failed_closed = False

    class RankLocalFailingScheduler:
        def step(self) -> None:
            if accelerator.process_index == 1:
                raise LookupError("injected rank-local scheduler failure")

    for _ in range(2):
        try:
            accelerated_microstep(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                forward_loss=lambda: model(torch.ones(1, 1)).square().mean(),
                lr_scheduler=RankLocalFailingScheduler(),
                max_grad_norm=1.0,
            )
        except RuntimeError as exc:
            rank_local_update_error_failed_closed = "injected rank-local scheduler failure" in str(
                exc
            )
            break
    update_error_count = accelerator.reduce(
        torch.tensor(int(rank_local_update_error_failed_closed), dtype=torch.int64),
        reduction="sum",
    )
    if int(update_error_count.item()) != accelerator.num_processes:
        raise AssertionError("rank-local optimizer/scheduler exception did not fail on every rank")

    model_hash = _model_sha256(accelerator.unwrap_model(model))
    gathered_hashes: list[str | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered_hashes, model_hash)
    if len(set(gathered_hashes)) != 1:
        raise AssertionError("DDP ranks ended with different model parameters")
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "attempted_optimizer_steps": progress.attempted_optimizer_steps,
            "checkpoint_resume_exact": True,
            "checkpoint_collision_failed_closed": True,
            "global_batch_size": plan.global_batch_size,
            "invalid_manifest_failed_closed": True,
            "model_sha256": model_hash,
            "nonfinite_loss_failed_closed": True,
            "rank_local_forward_error_failed_closed": True,
            "rank_local_prepare_error_failed_closed": True,
            "rank_local_update_error_failed_closed": True,
            "plan_sha256": plan.plan_sha256,
            "rank_local_checkpoint_state_exact": True,
            "rank_partition_exact": True,
            "schema": "picf-next.training-control-ddp-smoke.v1",
            "single_process_gradient_equivalent": True,
            "world_size": accelerator.num_processes,
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
