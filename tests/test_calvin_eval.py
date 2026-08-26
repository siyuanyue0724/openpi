from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from picf_next.eval.calvin import (
    CalvinEvaluationSequence,
    evaluate_calvin_sequence,
    summarize_calvin_results,
)


class _Policy:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, observation: Mapping[str, Any], instruction: str) -> np.ndarray:
        del observation, instruction
        return np.zeros(7, dtype=np.float32)


class _Environment:
    def __init__(self) -> None:
        self.steps = 0
        self.reset_calls = 0

    def reset(self, *, robot_obs: np.ndarray, scene_obs: np.ndarray) -> None:
        assert robot_obs.shape == (1,) and scene_obs.shape == (1,)
        self.steps = 0
        self.reset_calls += 1

    def get_obs(self) -> dict[str, int]:
        return {"step": self.steps}

    def get_info(self) -> dict[str, int]:
        return {"step": self.steps}

    def step(self, action: np.ndarray):
        assert action.shape == (7,)
        self.steps += 1
        return self.get_obs(), 0.0, False, self.get_info()


class _Oracle:
    def __init__(self, required_steps: dict[str, int]) -> None:
        self.required_steps = required_steps

    def get_task_info_for_set(self, start_info, current_info, tasks):
        task = next(iter(tasks))
        elapsed = current_info["step"] - start_info["step"]
        return {task} if elapsed >= self.required_steps[task] else set()


def _sequence() -> CalvinEvaluationSequence:
    return CalvinEvaluationSequence(initial_state="state", subtasks=("a", "b", "c", "d", "e"))


def _decode(_: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(1), np.zeros(1)


def test_calvin_eval_stops_entire_five_task_sequence_on_first_failure() -> None:
    environment = _Environment()
    policy = _Policy()
    result = evaluate_calvin_sequence(
        _sequence(),
        environment=environment,
        policy=policy,
        task_oracle=_Oracle({"a": 1, "b": 2, "c": 10, "d": 1, "e": 1}),
        annotations={key: [f"instruction {key}"] for key in "abcde"},
        decode_initial_state=_decode,
        max_subtask_steps=3,
    )

    assert result.successful_subtasks == 2
    assert [item.task_key for item in result.attempted] == ["a", "b", "c"]
    assert [item.steps for item in result.attempted] == [1, 2, 3]
    assert policy.reset_calls == 3
    assert environment.reset_calls == 1
    assert environment.steps == 6


def test_calvin_eval_summary_matches_official_one_to_five_success_rates() -> None:
    first = evaluate_calvin_sequence(
        _sequence(),
        environment=_Environment(),
        policy=_Policy(),
        task_oracle=_Oracle({key: 1 for key in "abcde"}),
        annotations={key: f"instruction {key}" for key in "abcde"},
        decode_initial_state=_decode,
    )
    second = evaluate_calvin_sequence(
        _sequence(),
        environment=_Environment(),
        policy=_Policy(),
        task_oracle=_Oracle({"a": 1, "b": 10, "c": 1, "d": 1, "e": 1}),
        annotations={key: f"instruction {key}" for key in "abcde"},
        decode_initial_state=_decode,
        max_subtask_steps=2,
    )
    summary = summarize_calvin_results((first, second))

    assert summary.success_rates == (1.0, 0.5, 0.5, 0.5, 0.5)
    assert summary.average_successful_length == 3.0
    assert summary.total_environment_steps == 8


def test_calvin_eval_rejects_malformed_policy_action() -> None:
    policy = _Policy()
    policy.step = lambda observation, instruction: np.zeros(6, dtype=np.float32)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        evaluate_calvin_sequence(
            _sequence(),
            environment=_Environment(),
            policy=policy,
            task_oracle=_Oracle({key: 1 for key in "abcde"}),
            annotations={key: key for key in "abcde"},
            decode_initial_state=_decode,
        )
