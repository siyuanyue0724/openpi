from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .posterior_full_check import run_full_check as run_point_only_full_check
    from .posterior_visual_acceptance_check import run_visual_acceptance_check
    from .posterior_visual_invariant_audit import run_visual_invariant_audit
    from .posterior_visual_replay_smoke import run_visual_smoke
    from .posterior_visual_stage1_spec_audit import run_visual_stage1_spec_audit
except ImportError:
    from posterior_full_check import run_full_check as run_point_only_full_check
    from posterior_visual_acceptance_check import run_visual_acceptance_check
    from posterior_visual_invariant_audit import run_visual_invariant_audit
    from posterior_visual_replay_smoke import run_visual_smoke
    from posterior_visual_stage1_spec_audit import run_visual_stage1_spec_audit


def run_visual_full_check(
    *,
    repo_root: str,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 256,
    checkpoint_path: str | None = None,
    model_name: str = "vjepa2_1_vit_base_384",
    arch_name_override: str | None = None,
    img_size: int = 384,
    num_frames: int = 64,
    patch_size: int = 16,
    tubelet_size: int = 2,
    device: str | None = None,
    dtype: str = "bfloat16",
    use_last_two_mean: bool = False,
) -> dict:
    point_only = run_point_only_full_check(
        repo_root=repo_root,
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
    )
    spec = run_visual_stage1_spec_audit(repo_root)
    visual_only_smoke = run_visual_smoke(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        mode="visual_only",
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        arch_name_override=arch_name_override,
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        device=device,
        dtype=dtype,
        use_last_two_mean=use_last_two_mean,
    )
    point_visual_smoke = run_visual_smoke(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        mode="point_visual",
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        arch_name_override=arch_name_override,
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        device=device,
        dtype=dtype,
        use_last_two_mean=use_last_two_mean,
    )
    visual_only_invariants = run_visual_invariant_audit(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        mode="visual_only",
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        arch_name_override=arch_name_override,
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        device=device,
        dtype=dtype,
        use_last_two_mean=use_last_two_mean,
    )
    point_visual_acceptance = run_visual_acceptance_check(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        mode="point_visual",
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        arch_name_override=arch_name_override,
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        device=device,
        dtype=dtype,
        use_last_two_mean=use_last_two_mean,
    )
    checks = {
        "point_only_full_pass": bool(point_only["all_pass"]),
        "visual_stage1_spec_pass": bool(spec["all_pass"]),
        "visual_only_smoke_has_gate": bool(visual_only_smoke["mean_visual_gate_ratio"] > 0.0),
        "visual_only_invariants_pass": bool(visual_only_invariants["all_pass"]),
        "point_visual_smoke_has_both_gains": bool(
            point_visual_smoke["mean_point_precision_gain_count"] > 0.0
            and point_visual_smoke["mean_visual_precision_gain_count"] > 0.0
        ),
        "point_visual_acceptance_pass": bool(point_visual_acceptance["all_pass"]),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "point_only": point_only,
        "visual_stage1_spec": spec,
        "visual_only_smoke": visual_only_smoke,
        "visual_only_invariants": visual_only_invariants,
        "point_visual_smoke": point_visual_smoke,
        "point_visual_acceptance": point_visual_acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full PICF visual posterior stage-1 check.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--model-name", default="vjepa2_1_vit_base_384")
    parser.add_argument("--arch-name-override", default=None)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--use-last-two-mean", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_visual_full_check(
        repo_root=args.repo_root,
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        num_segments=args.segments,
        stride=args.stride,
        max_points=args.max_points,
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        arch_name_override=args.arch_name_override,
        img_size=args.img_size,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        device=args.device,
        dtype=args.dtype,
        use_last_two_mean=args.use_last_two_mean,
    )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
