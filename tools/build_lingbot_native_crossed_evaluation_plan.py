#!/usr/bin/env python3
"""Build the frozen held-out exact-X evaluation plan for ADR-128."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot crossed evaluation-plan builder",
)

from picf_next.lingbot_native.crossed_evaluation import (  # noqa: E402
    CROSSED_EVALUATION_MAXIMUM_BYTES,
    CrossedEvaluationPlan,
    build_crossed_evaluation_plan,
)
from picf_next.lingbot_native.vl_curriculum import (  # noqa: E402
    NATIVE_VL_CURRICULUM_MAXIMUM_BYTES,
    NativeVLGroundingCurriculumPlan,
)
from tools.build_lingbot_native_crossed_bounded_plan import (  # noqa: E402
    _load_verified_json,
    _validated_checkout_revision,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", required=True, type=Path)
    parser.add_argument("--curriculum-sha256", required=True)
    parser.add_argument("--scene-audit", required=True, type=Path)
    parser.add_argument("--scene-audit-sha256", required=True)
    parser.add_argument("--episode-split", required=True, type=Path)
    parser.add_argument("--episode-split-sha256", required=True)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkout = Path(__file__).resolve().parents[1]
    revision = _validated_checkout_revision(checkout)
    if revision != args.picf_code_revision:
        raise ValueError("crossed evaluation PICF revision differs from its checkout")
    curriculum_json, curriculum_file_sha256 = _load_verified_json(
        args.curriculum,
        expected_sha256=args.curriculum_sha256,
        maximum_bytes=NATIVE_VL_CURRICULUM_MAXIMUM_BYTES,
        name="native VL curriculum",
    )
    curriculum = NativeVLGroundingCurriculumPlan.from_dict(curriculum_json)
    scene_audit, scene_file_sha256 = _load_verified_json(
        args.scene_audit,
        expected_sha256=args.scene_audit_sha256,
        maximum_bytes=64 * 1024 * 1024,
        name="scene audit",
    )
    episode_split, split_file_sha256 = _load_verified_json(
        args.episode_split,
        expected_sha256=args.episode_split_sha256,
        maximum_bytes=CROSSED_EVALUATION_MAXIMUM_BYTES,
        name="episode split",
    )
    plan = build_crossed_evaluation_plan(
        curriculum,
        scene_audit,
        episode_split,
        curriculum_file_sha256=curriculum_file_sha256,
        scene_audit_file_sha256=scene_file_sha256,
        episode_split_file_sha256=split_file_sha256,
        picf_code_revision=revision,
    )
    if _validated_checkout_revision(checkout) != revision:
        raise ValueError("crossed evaluation checkout changed before plan publication")
    output = args.output.expanduser().absolute()
    plan.write(output)
    if CrossedEvaluationPlan.load(output) != plan:
        raise RuntimeError("crossed evaluation plan changed during durable round trip")
    payload = output.read_bytes()
    print(
        json.dumps(
            {
                "artifact_sha256": plan.artifact_sha256,
                "checkpoint_selection_authorized": False,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "output": str(output),
                "summary": plan.summary,
                "training_authorized": False,
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
