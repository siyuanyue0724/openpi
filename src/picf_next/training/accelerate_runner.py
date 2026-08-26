"""Small Accelerate adapter for exact PICF training-state continuation.

The update order follows the pinned LeRobot trainer.  Distributed wrapping,
mixed precision, optimizer/scaler state and RNG serialization remain delegated
to Hugging Face Accelerate; PICF adds only strict experiment identity and an
atomic completion marker around those official state files.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from picf_next.training.control import (
    ExperimentRunContract,
    RunProgress,
    TrainingPlan,
    validate_control_manifest,
    write_control_manifest,
)


@dataclass(frozen=True, slots=True)
class AcceleratedMicrostepOutput:
    loss: float
    grad_norm: float | None
    synchronization_boundary: bool
    optimizer_step_was_skipped: bool


def _run_main_process_action(
    accelerator: Any,
    *,
    label: str,
    action: Callable[[], None],
) -> None:
    """Broadcast rank-zero filesystem failures instead of stranding peers."""

    _run_main_process_value(accelerator, label=label, action=action)


def _run_main_process_value(
    accelerator: Any,
    *,
    label: str,
    action: Callable[[], Any],
) -> Any:
    """Execute one filesystem read/write on rank zero and broadcast its result."""

    caught: Exception | None = None
    error_message: str | None = None
    value: Any = None
    if accelerator.is_main_process:
        try:
            value = action()
        except Exception as exc:  # pragma: no cover - exercised through failure injection
            caught = exc
            error_message = f"{type(exc).__name__}: {exc}"
    if int(accelerator.num_processes) > 1:
        from accelerate.utils import broadcast_object_list

        payload = [error_message, value]
        broadcast_object_list(payload, from_process=0)
        error_message, value = payload
    if error_message is not None:
        if caught is not None and int(accelerator.num_processes) == 1:
            raise caught
        raise RuntimeError(f"rank-zero {label} failed: {error_message}")
    return value


def distributed_main_process_call(
    accelerator: Any,
    *,
    label: str,
    action: Callable[[], Any],
) -> Any:
    """Run one rank-zero action and broadcast either its value or failure."""

    return _run_main_process_value(accelerator, label=label, action=action)


def _run_each_process_action(
    accelerator: Any,
    *,
    label: str,
    action: Callable[[], None],
) -> None:
    """Run one rank-local filesystem action and make every rank see failures."""

    error_message: str | None = None
    try:
        action()
    except Exception as exc:  # pragma: no cover - multi-rank failure injection only
        error_message = f"rank {accelerator.process_index}: {type(exc).__name__}: {exc}"
    if int(accelerator.num_processes) > 1:
        from accelerate.utils import gather_object

        errors = gather_object([error_message])
    else:
        errors = [error_message]
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError(f"rank-local {label} failed: {'; '.join(failures)}")


def distributed_rank_local_call(
    accelerator: Any,
    *,
    label: str,
    action: Callable[[], Any],
) -> Any:
    """Run a rank-local action and agree on success before later collectives.

    The success path uses one scalar reduction. Python error payloads are
    gathered only after a failure, so hot-path state preparation does not incur
    an object collective.
    """

    if int(accelerator.num_processes) == 1:
        return action()

    import torch
    from accelerate.utils import gather_object

    value: Any = None
    error_message: str | None = None
    try:
        value = action()
    except Exception as exc:  # pragma: no cover - exercised by distributed probe
        error_message = f"rank {accelerator.process_index}: {type(exc).__name__}: {exc}"
    success_count = accelerator.reduce(
        torch.tensor(
            int(error_message is None),
            device=accelerator.device,
            dtype=torch.int64,
        ),
        reduction="sum",
    )
    if int(success_count.item()) != int(accelerator.num_processes):
        errors = gather_object([error_message])
        failures = [error for error in errors if error is not None]
        raise RuntimeError(f"rank-local {label} failed: {'; '.join(failures)}")
    return value


def _fsync_checkpoint_tree(checkpoint_dir: Path) -> None:
    """Make completed state files and directory entries durable before publication."""

    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint state cannot contain symlinks: {path}")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in checkpoint_dir.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rank_state_path(checkpoint_dir: Path, rank: int) -> Path:
    return checkpoint_dir / f"picf_rank_state_{rank:05d}.pt"


def _validate_rank_state_files(
    checkpoint_dir: Path,
    *,
    world_size: int,
    rank_state_expected: bool,
) -> None:
    observed = sorted(checkpoint_dir.glob("picf_rank_state_*.pt"))
    expected = (
        [_rank_state_path(checkpoint_dir, rank) for rank in range(world_size)]
        if rank_state_expected
        else []
    )
    if observed != expected:
        raise ValueError("checkpoint rank-local state files differ from the active run contract")


def accelerated_microstep(
    *,
    accelerator: Any,
    model: Any,
    optimizer: Any,
    forward_loss: Callable[[], Any],
    lr_scheduler: Any | None = None,
    max_grad_norm: float = 0.0,
) -> AcceleratedMicrostepOutput:
    """Run one microstep using the official Accelerate accumulation contract."""

    import torch

    if isinstance(max_grad_norm, bool) or not math.isfinite(max_grad_norm) or max_grad_norm < 0.0:
        raise ValueError("max_grad_norm must be finite and non-negative")
    model.train()
    grad_norm_value: float | None = None
    with accelerator.accumulate(model):
        forward_error: Exception | None = None
        error_message: str | None = None
        loss: Any = None
        try:
            with accelerator.autocast():
                loss = forward_loss()
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("forward_loss must return one scalar torch.Tensor")
            local_finite = torch.isfinite(loss.detach()).to(
                device=accelerator.device,
                dtype=torch.int64,
            )
        except Exception as exc:
            forward_error = exc
            error_message = f"rank {accelerator.process_index}: {type(exc).__name__}: {exc}"
            local_finite = torch.zeros((), device=accelerator.device, dtype=torch.int64)
        finite_count = accelerator.reduce(local_finite, reduction="sum")
        if int(finite_count.item()) != int(accelerator.num_processes):
            if error_message is None and int(local_finite.item()) == 0:
                error_message = f"rank {accelerator.process_index}: non-finite loss"
            if int(accelerator.num_processes) > 1:
                from accelerate.utils import gather_object

                errors = gather_object([error_message])
                failures = [error for error in errors if error is not None]
                raise FloatingPointError(
                    "forward failed on at least one distributed rank: " + "; ".join(failures)
                )
            if forward_error is not None:
                raise forward_error
            raise FloatingPointError("non-finite loss")
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("distributed forward agreement lost its scalar loss")
        accelerator.backward(loss)
        synchronization_boundary = bool(accelerator.sync_gradients)

        def apply_local_update() -> tuple[float | None, bool]:
            local_grad_norm: float | None = None
            if synchronization_boundary:
                clip_limit = max_grad_norm if max_grad_norm > 0.0 else float("inf")
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_limit)
                local_grad_norm = float(grad_norm.detach().float().item())
                if not math.isfinite(local_grad_norm):
                    raise FloatingPointError("non-finite synchronized gradient norm")
            optimizer.step()
            step_was_skipped = bool(accelerator.optimizer_step_was_skipped)
            if lr_scheduler is not None and synchronization_boundary and not step_was_skipped:
                lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return local_grad_norm, step_was_skipped

        if synchronization_boundary:
            # Backward/gradient collectives are already complete.  Agree on every
            # rank-local optimizer/scheduler result before a caller can enter a
            # posterior commit or checkpoint collective.  A partial local update
            # still requires checkpoint restore, but it can no longer strand peers.
            grad_norm_value, optimizer_step_was_skipped = distributed_rank_local_call(
                accelerator,
                label="optimizer synchronization update",
                action=apply_local_update,
            )
            if int(accelerator.num_processes) > 1:
                skipped_count = accelerator.reduce(
                    torch.tensor(
                        int(optimizer_step_was_skipped),
                        device=accelerator.device,
                        dtype=torch.int64,
                    ),
                    reduction="sum",
                )
                if int(skipped_count.item()) not in {0, int(accelerator.num_processes)}:
                    raise RuntimeError("optimizer skip state differs across distributed ranks")
        else:
            # Accelerate's wrapped optimizer and zero_grad are no-ops while
            # gradients are accumulating, so an extra collective is unnecessary.
            grad_norm_value, optimizer_step_was_skipped = apply_local_update()
    return AcceleratedMicrostepOutput(
        loss=float(loss.detach().float().item()),
        grad_norm=grad_norm_value,
        synchronization_boundary=synchronization_boundary,
        optimizer_step_was_skipped=optimizer_step_was_skipped,
    )


def register_progress_for_checkpointing(accelerator: Any, progress: RunProgress) -> None:
    accelerator.register_for_checkpointing(progress)


def save_accelerate_checkpoint(
    *,
    accelerator: Any,
    checkpoint_dir: str | Path,
    contract: ExperimentRunContract,
    plan: TrainingPlan,
    progress: RunProgress,
    rank_state: Any | None = None,
) -> Path:
    """Save official distributed state and atomically publish one checkpoint."""

    import torch

    final = Path(checkpoint_dir)
    staging = final.with_name(f".{final.name}.incomplete")
    if rank_state is not None and not callable(getattr(rank_state, "state_dict", None)):
        raise TypeError("rank_state must expose state_dict")
    if not bool(accelerator.sync_gradients):
        raise RuntimeError("checkpointing is allowed only at an optimizer synchronization boundary")
    contract.validate_plan(plan)
    progress.validate_capacity(plan)
    if progress.contract_sha256 != contract.contract_sha256:
        raise ValueError("progress and run contract hashes differ")
    if progress.sample_plan_sha256 != plan.plan_sha256:
        raise ValueError("progress and frozen sample plan hashes differ")
    accelerator.wait_for_everyone()
    collision = accelerator.reduce(
        torch.tensor(
            int(final.exists() or staging.exists()),
            device=accelerator.device,
            dtype=torch.int64,
        ),
        reduction="sum",
    )
    if int(collision.item()) > 0:
        raise FileExistsError(
            "checkpoint destination or incomplete staging directory already exists and "
            f"requires explicit audit: final={final}, staging={staging}"
        )
    accelerator.wait_for_everyone()

    def create_staging() -> None:
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.mkdir()

    _run_main_process_action(
        accelerator,
        label="checkpoint staging creation",
        action=create_staging,
    )
    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=str(staging), safe_serialization=True)
    accelerator.wait_for_everyone()

    if rank_state is not None:

        def save_rank_state() -> None:
            import torch

            destination = _rank_state_path(staging, int(accelerator.process_index))
            temporary = destination.with_suffix(".pt.incomplete")
            if destination.exists() or temporary.exists():
                raise FileExistsError(destination)
            torch.save(rank_state.state_dict(), temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)

        _run_each_process_action(
            accelerator,
            label="checkpoint state serialization",
            action=save_rank_state,
        )
        accelerator.wait_for_everyone()

    def publish_checkpoint() -> None:
        _validate_rank_state_files(
            staging,
            world_size=int(accelerator.num_processes),
            rank_state_expected=rank_state is not None,
        )
        _fsync_checkpoint_tree(staging)
        write_control_manifest(
            staging / "picf_control.json",
            contract=contract,
            plan=plan,
            progress=progress,
        )
        _fsync_checkpoint_tree(staging)
        _fsync_directory(staging)
        os.replace(staging, final)
        _fsync_directory(final.parent)

    _run_main_process_action(
        accelerator,
        label="checkpoint publication",
        action=publish_checkpoint,
    )
    accelerator.wait_for_everyone()
    return final


def load_accelerate_checkpoint(
    *,
    accelerator: Any,
    checkpoint_dir: str | Path,
    contract: ExperimentRunContract,
    plan: TrainingPlan,
    progress: RunProgress,
    rank_state: Any | None = None,
) -> None:
    """Validate experiment identity before loading official distributed state."""

    checkpoint = Path(checkpoint_dir)
    control_path = checkpoint / "picf_control.json"
    if rank_state is not None:
        if not callable(getattr(rank_state, "load_state_dict", None)):
            raise TypeError("rank_state must expose load_state_dict")
        if not callable(getattr(rank_state, "validate_state_dict", None)):
            raise TypeError("rank_state must expose validate_state_dict")

    def validate_checkpoint() -> dict[str, Any]:
        expected = validate_control_manifest(
            control_path,
            contract=contract,
            plan=plan,
        )
        _validate_rank_state_files(
            checkpoint,
            world_size=int(accelerator.num_processes),
            rank_state_expected=rank_state is not None,
        )
        return expected

    expected_progress = _run_main_process_value(
        accelerator,
        label="checkpoint control validation",
        action=validate_checkpoint,
    )
    accelerator.wait_for_everyone()
    if rank_state is not None:

        def validate_rank_state() -> None:
            import torch

            state = torch.load(
                _rank_state_path(checkpoint, int(accelerator.process_index)),
                map_location="cpu",
                weights_only=True,
            )
            rank_state.validate_state_dict(state)

        _run_each_process_action(
            accelerator,
            label="checkpoint state prevalidation",
            action=validate_rank_state,
        )
        accelerator.wait_for_everyone()
    accelerator.load_state(str(checkpoint))
    accelerator.wait_for_everyone()
    if rank_state is not None:

        def load_rank_state() -> None:
            import torch

            state = torch.load(
                _rank_state_path(checkpoint, int(accelerator.process_index)),
                map_location="cpu",
                weights_only=True,
            )
            rank_state.load_state_dict(state)

        _run_each_process_action(
            accelerator,
            label="checkpoint state restoration",
            action=load_rank_state,
        )
        accelerator.wait_for_everyone()

    def validate_loaded_progress() -> None:
        if progress.state_dict() != expected_progress:
            raise ValueError("Accelerate custom progress state differs from the control manifest")
        progress.validate_capacity(plan)

    distributed_rank_local_call(
        accelerator,
        label="checkpoint progress postvalidation",
        action=validate_loaded_progress,
    )
