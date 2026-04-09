from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .posterior_visual_invariant_audit import run_visual_invariant_audit
    from .posterior_visual_replay_smoke import run_visual_smoke
except ImportError:
    from posterior_visual_invariant_audit import run_visual_invariant_audit
    from posterior_visual_replay_smoke import run_visual_smoke


def run_visual_acceptance_check(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    mode: str = "point_visual",
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 1024,
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
    sonata_ckpt_path: str | None = None,
    point_device: str | None = None,
) -> dict:
    smoke = run_visual_smoke(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        mode=mode,
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
        sonata_ckpt_path=sonata_ckpt_path,
        point_device=point_device,
    )
    invariants = run_visual_invariant_audit(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        mode=mode,
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
        sonata_ckpt_path=sonata_ckpt_path,
        point_device=point_device,
    )
    checks = {
        "invariants_pass": bool(invariants["all_pass"]),
        "no_nans": bool(smoke["nan_count_total"] == 0),
        "var_bounds_valid": bool(smoke["min_var_block"] >= 1e-4 and smoke["max_var_block"] <= 10.0),
    }
    if mode in {"point_only", "point_visual"}:
        checks["point_gate_nonzero"] = bool(smoke["mean_point_gate_ratio"] > 0.0)
        checks["point_precision_gain_nonzero"] = bool(smoke["mean_point_precision_gain_count"] > 0.0)
    if mode in {"visual_only", "point_visual"}:
        checks["visual_gate_nonzero"] = bool(smoke["mean_visual_gate_ratio"] > 0.0)
        checks["visual_precision_gain_nonzero"] = bool(smoke["mean_visual_precision_gain_count"] > 0.0)
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "smoke": smoke,
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acceptance gate for PICF visual posterior stage-1.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--mode", default="point_visual", choices=["point_only", "visual_only", "point_visual"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=1024)
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
    parser.add_argument("--sonata-ckpt-path", default=None)
    parser.add_argument("--point-device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_visual_acceptance_check(
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        mode=args.mode,
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
        sonata_ckpt_path=args.sonata_ckpt_path,
        point_device=args.point_device,
    )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
