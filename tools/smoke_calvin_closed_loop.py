#!/usr/bin/env python3
"""Run the PICF evaluator against a real official CALVIN PyBullet environment.

This is an environment/wiring smoke with a deterministic no-op action. It is
not a policy score. The local smoke intentionally instantiates the official
static+gripper camera configuration because the separate tactile simulator
dependency must pass its own gate before online AnyTouch evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from picf_next.data.calvin import (
    CALVIN_DEBUG_DATASET_ID,
    CALVIN_DEBUG_REVISION,
    CalvinDatasetIndex,
)
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.eval.calvin import CalvinEvaluationSequence, evaluate_calvin_sequence
from tools.build_calvin_golden import _build_environment, _close_environment

DEFAULT_TASKS = (
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "open_drawer",
)


class NoOpCalvinPolicy:
    def reset(self) -> None:
        pass

    def step(self, observation, instruction) -> np.ndarray:
        del observation, instruction
        return np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-id", default=CALVIN_DEBUG_DATASET_ID)
    parser.add_argument("--dataset-revision", default=CALVIN_DEBUG_REVISION)
    parser.add_argument("--calvin-env-root", required=True, type=Path)
    parser.add_argument("--calvin-models-root", required=True, type=Path)
    parser.add_argument("--max-subtask-steps", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import hydra

    validation = args.dataset_root.resolve() / "validation"
    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        validation,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        dataset_manifest=manifest,
    )
    global_index = index.episodes[0].start
    source = index.frame_path(global_index)
    frame = index.validated_source_frame_arrays(
        global_index,
        fields=("robot_obs", "scene_obs"),
    )
    initial_state = (frame["robot_obs"], frame["scene_obs"])

    config_root = args.calvin_models_root.resolve() / "conf"
    task_config = OmegaConf.load(
        config_root / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_config)
    annotations = OmegaConf.to_container(
        OmegaConf.load(config_root / "annotations" / "new_playtable_validation.yaml"),
        resolve=True,
    )
    if not isinstance(annotations, dict):
        raise TypeError("official CALVIN validation annotations must be a mapping")

    environment, _ = _build_environment(args.calvin_env_root.resolve())
    try:
        result = evaluate_calvin_sequence(
            CalvinEvaluationSequence(initial_state=initial_state, subtasks=DEFAULT_TASKS),
            environment=environment,
            policy=NoOpCalvinPolicy(),
            task_oracle=task_oracle,
            annotations=annotations,
            decode_initial_state=lambda value: value,
            max_subtask_steps=args.max_subtask_steps,
        )
    finally:
        _close_environment(environment)

    report = {
        "format": "picf-next.calvin-closed-loop-smoke/v1",
        "policy": "deterministic-no-op; not a score",
        "camera_configuration": "official-static-and-gripper",
        "source_frame": str(source.relative_to(args.dataset_root.resolve())),
        "max_subtask_steps": args.max_subtask_steps,
        "successful_subtasks": result.successful_subtasks,
        "attempted": [
            {
                "task_key": item.task_key,
                "instruction": item.instruction,
                "success": item.success,
                "steps": item.steps,
            }
            for item in result.attempted
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
