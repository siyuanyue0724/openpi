"""Dependency-light implementation of the official CALVIN five-task protocol.

The control flow follows ``mees/calvin`` evaluation: reset the environment once
per five-task sequence, reset the policy at every subtask, allow at most 360
environment steps, and stop the complete sequence at the first failed subtask.
Environment construction and the official task oracle remain injected so this
module can be tested without PyBullet or Hydra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

CALVIN_SEQUENCE_LENGTH = 5
CALVIN_MAX_SUBTASK_STEPS = 360


class CalvinPolicy(Protocol):
    def reset(self) -> None: ...

    def step(self, observation: Mapping[str, Any], instruction: str) -> NDArray: ...


class CalvinEnvironment(Protocol):
    def reset(self, *, robot_obs: NDArray, scene_obs: NDArray) -> Any: ...

    def get_obs(self) -> Mapping[str, Any]: ...

    def get_info(self) -> Mapping[str, Any]: ...

    def step(self, action: NDArray) -> tuple[Mapping[str, Any], Any, Any, Mapping[str, Any]]: ...


class CalvinTaskOracle(Protocol):
    def get_task_info_for_set(
        self,
        start_info: Mapping[str, Any],
        current_info: Mapping[str, Any],
        tasks: set[str],
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class CalvinEvaluationSequence:
    initial_state: Any
    subtasks: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.subtasks) != CALVIN_SEQUENCE_LENGTH or any(not task for task in self.subtasks):
            raise ValueError("official CALVIN evaluation requires exactly five named subtasks")


@dataclass(frozen=True, slots=True)
class CalvinSubtaskResult:
    task_key: str
    instruction: str
    success: bool
    steps: int

    def __post_init__(self) -> None:
        if not self.task_key or not self.instruction or self.steps <= 0:
            raise ValueError("CALVIN subtask result is incomplete")


@dataclass(frozen=True, slots=True)
class CalvinSequenceResult:
    subtasks: tuple[str, ...]
    attempted: tuple[CalvinSubtaskResult, ...]

    def __post_init__(self) -> None:
        if len(self.subtasks) != CALVIN_SEQUENCE_LENGTH:
            raise ValueError("CALVIN result must retain the complete five-task sequence")
        if not self.attempted or len(self.attempted) > CALVIN_SEQUENCE_LENGTH:
            raise ValueError("CALVIN result must contain one to five attempted subtasks")
        if tuple(item.task_key for item in self.attempted) != self.subtasks[: len(self.attempted)]:
            raise ValueError("CALVIN attempted tasks must be a sequence prefix")
        failures = [index for index, item in enumerate(self.attempted) if not item.success]
        if failures and failures != [len(self.attempted) - 1]:
            raise ValueError("CALVIN evaluation must stop immediately after the first failure")
        if not failures and len(self.attempted) != CALVIN_SEQUENCE_LENGTH:
            raise ValueError("a successful CALVIN prefix must continue to the next subtask")

    @property
    def successful_subtasks(self) -> int:
        return sum(item.success for item in self.attempted)

    @property
    def total_steps(self) -> int:
        return sum(item.steps for item in self.attempted)


@dataclass(frozen=True, slots=True)
class CalvinEvaluationSummary:
    sequence_count: int
    success_rates: tuple[float, ...]
    average_successful_length: float
    total_environment_steps: int

    def __post_init__(self) -> None:
        if self.sequence_count <= 0 or len(self.success_rates) != CALVIN_SEQUENCE_LENGTH:
            raise ValueError("CALVIN summary shape is invalid")
        if any(rate < 0.0 or rate > 1.0 for rate in self.success_rates):
            raise ValueError("CALVIN success rates must lie in [0, 1]")


def _instruction(
    annotations: Mapping[str, Sequence[str] | str],
    task_key: str,
) -> str:
    if task_key not in annotations:
        raise KeyError(f"CALVIN validation annotations do not contain {task_key!r}")
    value = annotations[task_key]
    if isinstance(value, str):
        instruction = value
    else:
        if not value:
            raise ValueError(f"CALVIN task {task_key!r} has no language annotation")
        instruction = str(value[0])
    if not instruction:
        raise ValueError(f"CALVIN task {task_key!r} has an empty language annotation")
    return instruction


def _policy_action(value: Any) -> NDArray[np.float32]:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("CALVIN policy action must be one finite float32 vector of shape (7,)")
    return action


def evaluate_calvin_sequence(
    sequence: CalvinEvaluationSequence,
    *,
    environment: CalvinEnvironment,
    policy: CalvinPolicy,
    task_oracle: CalvinTaskOracle,
    annotations: Mapping[str, Sequence[str] | str],
    decode_initial_state: Callable[[Any], tuple[NDArray, NDArray]],
    max_subtask_steps: int = CALVIN_MAX_SUBTASK_STEPS,
    on_step: Callable[[str, str, int, Mapping[str, Any], NDArray], None] | None = None,
) -> CalvinSequenceResult:
    """Evaluate one sequence with exact fail-fast CALVIN control flow."""

    if (
        not isinstance(max_subtask_steps, int)
        or isinstance(max_subtask_steps, bool)
        or max_subtask_steps <= 0
    ):
        raise ValueError("CALVIN maximum subtask steps must be positive")
    robot_obs, scene_obs = decode_initial_state(sequence.initial_state)
    environment.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    results: list[CalvinSubtaskResult] = []
    for task_key in sequence.subtasks:
        instruction = _instruction(annotations, task_key)
        observation = environment.get_obs()
        policy.reset()
        start_info = environment.get_info()
        success = False
        steps = 0
        for step_index in range(max_subtask_steps):
            action = _policy_action(policy.step(observation, instruction))
            observation, _, _, current_info = environment.step(action)
            steps = step_index + 1
            if on_step is not None:
                on_step(task_key, instruction, steps, observation, action)
            completed = task_oracle.get_task_info_for_set(start_info, current_info, {task_key})
            if len(completed) > 0:
                success = True
                break
        results.append(
            CalvinSubtaskResult(
                task_key=task_key,
                instruction=instruction,
                success=success,
                steps=steps,
            )
        )
        if not success:
            break
    return CalvinSequenceResult(subtasks=sequence.subtasks, attempted=tuple(results))


def summarize_calvin_results(
    results: Sequence[CalvinSequenceResult],
) -> CalvinEvaluationSummary:
    if not results:
        raise ValueError("cannot summarize an empty CALVIN evaluation")
    lengths = np.asarray([result.successful_subtasks for result in results], dtype=np.float64)
    success_rates = tuple(float(np.mean(lengths >= threshold)) for threshold in range(1, 6))
    return CalvinEvaluationSummary(
        sequence_count=len(results),
        success_rates=success_rates,
        average_successful_length=float(lengths.mean()),
        total_environment_steps=sum(result.total_steps for result in results),
    )


def evaluate_calvin_sequences(
    sequences: Sequence[CalvinEvaluationSequence],
    **kwargs: Any,
) -> tuple[tuple[CalvinSequenceResult, ...], CalvinEvaluationSummary]:
    if not sequences:
        raise ValueError("CALVIN evaluation requires at least one sequence")
    results = tuple(evaluate_calvin_sequence(sequence, **kwargs) for sequence in sequences)
    return results, summarize_calvin_results(results)


def calvin_sequences_from_official(
    values: Sequence[tuple[Any, Sequence[str]]],
) -> tuple[CalvinEvaluationSequence, ...]:
    """Validate the output of official ``get_sequences`` at the local boundary."""

    return tuple(
        CalvinEvaluationSequence(initial_state=initial_state, subtasks=tuple(subtasks))
        for initial_state, subtasks in values
    )
