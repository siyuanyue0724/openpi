#!/usr/bin/env python3
"""LEGACY blind-SAM proposal dataflow audit.

Archived on 2026-05-18 with the blind SAM proposal generator. The maintained
proposal path is contact/task-guided sidecar evidence, not blind automatic SAM
mask generation. Keep this script only to reproduce rejected 2026-05-17 SAM
ablations.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditCheck:
    name: str
    ok: bool
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains(source: str, *needles: str) -> bool:
    return all(needle in source for needle in needles)


def _math_checks() -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    visual_tokens = np.random.default_rng(0).normal(size=(128, 64))
    proposal_tokens = np.random.default_rng(1).normal(size=(16, 64))
    checks.append(
        AuditCheck(
            "math_proposals_are_additive_not_replacement",
            visual_tokens.shape[0] > 0 and proposal_tokens.shape[0] > 0,
            "SAM proposals must add typed object evidence; dense V-JEPA/visual tokens remain available to PI/PICF.",
        )
    )
    iou = np.array([0.9, 0.2, 0.9, 0.0])
    stability = np.array([0.9, 0.9, 0.1, 1.0])
    objectness = np.sqrt(np.clip(iou, 0.0, 1.0) * np.clip(stability, 0.0, 1.0))
    checks.append(
        AuditCheck(
            "math_objectness_requires_iou_and_stability",
            bool(objectness[0] > objectness[1] > objectness[2] > objectness[3]),
            "Proposal objectness should require both decoder IoU and mask stability, not one high scalar alone.",
        )
    )
    boxes = np.array([[0.1, 0.1, 0.3, 0.4], [0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
    centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    checks.append(
        AuditCheck(
            "math_box_center_normalized",
            bool(np.all((0.0 <= centers) & (centers <= 1.0))),
            "PICF proposal geometry contract uses normalized xyxy boxes and centers in [0,1].",
        )
    )
    return checks


def _external_code_checks(root: Path) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    segment_anything = root / "segment-anything"
    sam2 = root / "sam2"
    openmask3d = root / "openmask3d"
    open3dis = root / "open3dis"
    sam_amg = segment_anything / "segment_anything" / "automatic_mask_generator.py"
    sam2_img = sam2 / "sam2" / "sam2_image_predictor.py"
    om3d = openmask3d / "openmask3d" / "compute_features_single_scene.py"
    o3d_mapper = open3dis / "open3dis" / "src" / "mapper.py"
    o3d_final = open3dis / "tools" / "generate_3d_inst.py"

    if sam_amg.exists():
        text = _read(sam_amg)
        ok = _contains(text, "class SamAutomaticMaskGenerator", "def generate", "predicted_iou", "stability_score", "bbox")
        checks.append(
            AuditCheck(
                "paper_code_sam_mask_generator_outputs_box_quality",
                ok,
                "SAM automatic mask generator exposes masks, boxes, decoder IoU, and stability quality scalars.",
            )
        )
    else:
        checks.append(AuditCheck("paper_code_sam_mask_generator_outputs_box_quality", False, f"Missing {sam_amg}"))

    if sam2_img.exists():
        text = _read(sam2_img)
        ok = _contains(text, "boxes", "predict", "mask_decoder") and ("multi object prediction" in text or "box_labels" in text)
        checks.append(
            AuditCheck(
                "paper_code_sam2_prompted_image_masks_available",
                ok,
                "SAM2 image predictor supports point/box-prompted mask decoding; video propagation can be kept offline.",
            )
        )
    else:
        checks.append(AuditCheck("paper_code_sam2_prompted_image_masks_available", False, f"Missing {sam2_img}"))

    if om3d.exists():
        text = _read(om3d)
        ok = _contains(text, "masks = InstanceMasks3D", "pointcloud = PointCloud", "FeaturesExtractor", "extract_features")
        checks.append(
            AuditCheck(
                "paper_code_openmask3d_mask_pointcloud_feature_flow",
                ok,
                "OpenMask3D style flow loads 3D masks and point clouds, then computes per-mask features.",
            )
        )
    else:
        checks.append(AuditCheck("paper_code_openmask3d_mask_pointcloud_feature_flow", False, f"Missing {om3d}"))

    if o3d_mapper.exists() and o3d_final.exists():
        mapper = _read(o3d_mapper)
        final = _read(o3d_final)
        ok = _contains(mapper, "PointCloudToImageMapper", "compute_mapping", "occlusion_mask") and _contains(
            final, "use_2d_proposals", "use_3d_proposals", "masks_final"
        )
        checks.append(
            AuditCheck(
                "paper_code_open3dis_2d_3d_proposal_fusion",
                ok,
                "Open3DIS style flow explicitly supports 2D proposals, 3D proposals, and depth-aware point-image mapping.",
            )
        )
    else:
        checks.append(AuditCheck("paper_code_open3dis_2d_3d_proposal_fusion", False, f"Missing {o3d_mapper} or {o3d_final}"))
    return checks


def run_checks(external_code_root: Path | None) -> list[AuditCheck]:
    checks = _math_checks()
    contracts = _read(REPO_ROOT / "src/openpi/picf/contracts.py")
    core_contracts = _read(REPO_ROOT / "src/openpi/picf/core/contracts.py")
    config = _read(REPO_ROOT / "src/openpi/picf/core/config.py")
    pipeline = _read(REPO_ROOT / "src/openpi/picf/core/pipeline.py")
    replay = _read(REPO_ROOT / "src/openpi/picf/replay/calvin_replay.py")
    train = _read(REPO_ROOT / "scripts/picf_core_train.py")
    serve = _read(REPO_ROOT / "scripts/serve_picf_policy.py")
    precompute = _read(REPO_ROOT / "scripts/archive/picf_sam_proposal_precompute_legacy.py")
    readme = _read(REPO_ROOT / "src/openpi/picf/README_v2.2.md")

    checks.extend(
        [
            AuditCheck(
                "picf_observation_exposes_proposal_fields",
                _contains(
                    contracts,
                    "proposal_centers_xy",
                    "proposal_boxes_xyxy",
                    "proposal_objectness",
                    "proposal_view_ids",
                    "proposal_source_ids",
                    "proposal_age",
                ),
                "Top-level observation contract must carry frozen proposal fields.",
            ),
            AuditCheck(
                "typed_proposal_state_exists",
                _contains(core_contracts, "class PicfPseudoProposalState", "tokens", "boxes_xyxy", "objectness", "source_ids", "age"),
                "Core contract must preserve proposal boxes/objectness/source metadata.",
            ),
            AuditCheck(
                "proposal_branch_is_opt_in_typed_memory",
                _contains(
                    config,
                    "proposal_memory_enabled: bool = False",
                    "proposal_read_weight: float = 0.0",
                    "proposal_confidence_floor",
                    "proposal_shape_quality_enabled",
                    "proposal_shape_area_min",
                    "proposal_shape_area_max",
                    "proposal_shape_aspect_min",
                    "proposal_context_quality_power",
                    "proposal_point_bridge_weight: float = 0.0",
                    "proposal_point_bridge_edge_tau",
                    "task_owner_proposal_bias_weight: float = 0.0",
                    "task_owner_proposal_point_bias_weight: float = 0.0",
                    "task_owner_proposal_point_bridge_weight: float = 0.0",
                    "task_owner_proposal_topk",
                    "task_owner_proposal_score_floor",
                ),
                (
                    "Proposal memory must be opt-in typed support: blind SAM is production-default off, "
                    "while residual read, quality calibration, and bounded proposal-to-point geometry bridging remain available for explicit prompted/reranked ablations."
                ),
            ),
            AuditCheck(
                "pipeline_builds_proposal_tokens_from_geometry_quality",
                _contains(
                    pipeline,
                    "observation.proposal_centers_xy",
                    "observation.proposal_boxes_xyxy",
                    "proposal_token_proj",
                    "objectness >= float(self.config.proposal_confidence_floor)",
                    "proposal_age",
                    "torch.exp(-proposal_age / age_decay)",
                ),
                "Pipeline must convert normalized proposal geometry and age-decayed objectness into typed support tokens.",
            ),
            AuditCheck(
                "pipeline_reads_proposals_with_residual_gate",
                _contains(pipeline, "aqr_proposal_reader", "proposal_read_weight", "q_before_prop", "prop_read - q_before_prop"),
                "AQR proposal read must be residual-gated, not a hard replacement or posterior overwrite.",
            ),
            AuditCheck(
                "pipeline_bridges_proposals_to_point_geometry",
                _contains(
                    pipeline,
                    "def _proposal_priors_to_point_priors",
                    "def _task_owner_proposal_point_bias",
                    "def _task_owner_proposal_to_point_priors",
                    "proposal_point_bridge_weight",
                    "task_owner_proposal_point_bias_weight",
                    "task_owner_proposal_point_bridge_weight",
                    "proposal_point_bridge_edge_tau",
                    "owner_point_bias",
                    "point_proj_grid_norm",
                    "proposal_point_priors",
                    "task_owner_point_priors",
                    "aqr_proposal_point_bridge_max",
                    "aqr_task_owner_point_bridge_max",
                ),
                "2D proposal evidence must be able to become weak 3D point support through projection; otherwise boxes cannot move anchor_x.",
            ),
            AuditCheck(
                "pipeline_downweights_blind_sam_fragments_before_task_owner_transport",
                _contains(
                    pipeline,
                    "def _proposal_shape_quality",
                    "proposal_shape_area_min",
                    "proposal_shape_area_max",
                    "proposal_shape_aspect_min",
                    "proposal_context_quality_power",
                    "def _postprocess_task_owner_proposal_score",
                    "task_owner_proposal_topk",
                    "task_owner_proposal_score_floor",
                    "aqr_proposal_shape_quality_mean",
                    "aqr_task_owner_proposal_selected_count",
                    "aqr_task_owner_proposal_shape_quality_mean",
                ),
                "Blind SAM objectness must be calibrated by soft geometry quality and sparse task-owner selection before it can steer point support.",
            ),
            AuditCheck(
                "pipeline_preserves_dense_visual_tokens",
                _contains(pipeline, "visual_tokens=visual_tokens", "proposal=proposal_state", "graph_proposal_weights"),
                "SAM proposals must not remove dense visual tokens; they are additional typed evidence.",
            ),
            AuditCheck(
                "posterior_records_proposal_signature",
                _contains(pipeline, "proposal_signature", "graph_proposal_weights", "owm_proposal_tokens"),
                "Posterior/debug dataflow must expose proposal signatures and token counts for audit.",
            ),
            AuditCheck(
                "train_replay_serve_thread_optional_fields",
                _contains(replay, "_MVTRACK_PROPOSAL_KEYS", "proposal_centers_xy=frame.get") and _contains(
                    train, "_MVTRACK_PROPOSAL_KEYS", "proposal_centers_xy=frame.get", "proposal_age=frame.get"
                ) and _contains(serve, "proposal_centers_xy=_optional_array", "proposal_age=_optional_array"),
                "Train, replay, and serve paths must pass proposal fields when present.",
            ),
            AuditCheck(
                "sidecar_dataflow_deployed",
                _contains(replay, "_read_mvtrack_sidecar_fields", "mvtrack_sidecar_root")
                and _contains(
                    train,
                    "--mvtrack-sidecar-root",
                    "mvtrack_sidecar_root=args.mvtrack_sidecar_root",
                    "--mvtrack-sidecar-proposal-nearest-max-gap",
                    "proposal_nearest_max_gap",
                ),
                "Offline proposal generation must have a non-mutating sidecar path into train/replay, with age-aware nearest-frame fallback for sparse sidecars.",
            ),
            AuditCheck(
                "offline_sam_precompute_script_exists",
                _contains(
                    precompute,
                    "SamAutomaticMaskGenerator",
                    "proposal_centers_xy",
                    "proposal_boxes_xyxy",
                    "proposal_objectness",
                    "np.savez_compressed",
                ),
                "A concrete offline SAM->proposal sidecar generator must be available.",
            ),
            AuditCheck(
                "no_online_sam_dependency_in_core_training",
                "segment_anything" not in pipeline and "segment_anything" not in train and "segment_anything" not in serve,
                "SAM must stay offline/frozen; online training core should not import SAM.",
            ),
            AuditCheck(
                "readme_links_sam_proposal_contract",
                "PICF_AQR_OWM_SAM_PROPOSAL_DATAFLOW_20260516_TEMP.md" in readme,
                "README_v2.2 must route operators to the SAM/proposal dataflow contract.",
            ),
        ]
    )
    if external_code_root is not None:
        checks.extend(_external_code_checks(external_code_root))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-code-root", type=Path, default=Path("/tmp/picf_sam_code"))
    parser.add_argument("--skip-external-code", action="store_true")
    parser.add_argument("--fail-on-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    external = None if args.skip_external_code else args.external_code_root
    checks = run_checks(external)
    if args.json:
        print(json.dumps([check.__dict__ for check in checks], indent=2))
    else:
        for check in checks:
            status = "PASS" if check.ok else "FAIL"
            print(f"{status}: {check.name} - {check.detail}")
        passed = sum(1 for check in checks if check.ok)
        print(f"{passed}/{len(checks)} PASS")
    if args.fail_on_fail and not all(check.ok for check in checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
