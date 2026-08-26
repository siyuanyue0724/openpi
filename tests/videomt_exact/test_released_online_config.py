from __future__ import annotations

from pathlib import Path

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from picf_next._vendor.videomt.config import add_videomt_config
from picf_next.videomt_exact.calvin_targets import (
    VIDEOMT_YTVIS19_CLIP_LENGTH,
    VIDEOMT_YTVIS19_CROP_SIZE,
    VIDEOMT_YTVIS19_CROP_TYPE,
    VIDEOMT_YTVIS19_TRAIN_MAX_SIZE,
    VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES,
)
from picf_next.videomt_exact.optimizer import (
    VIDEOMT_BASE_LR,
    VIDEOMT_LAYERWISE_LR_DECAY,
    VIDEOMT_NON_VIT_WARMUP_STEPS,
    VIDEOMT_POLY_POWER,
    VIDEOMT_RELEASED_TOTAL_STEPS,
    VIDEOMT_VIT_WARMUP_STEPS,
    VIDEOMT_WEIGHT_DECAY,
)
from picf_next.videomt_exact.runtime import (
    VIDEOMT_DINOV3_L_QUERIES,
    VIDEOMT_DINOV3_L_SEGMENTER_BLOCKS,
)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "references/source_snapshots" / "videomt-025b9384214bf28cd90d51846464615dd4f443ac"
SELECTED_CONFIG = SNAPSHOT / "configs/ytvis19/videomt/dinov3/vit-large/videomt_online_ViTL.yaml"


def _resolved_config():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(str(SELECTED_CONFIG))
    cfg.freeze()
    return cfg


def test_local_contract_matches_resolved_selected_online_config() -> None:
    cfg = _resolved_config()
    assert cfg.MODEL.META_ARCHITECTURE == "videomt_online"
    assert cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES == VIDEOMT_DINOV3_L_QUERIES
    assert tuple(cfg.MODEL.BACKBONE.SEGMENTER_BLOCKS) == VIDEOMT_DINOV3_L_SEGMENTER_BLOCKS
    assert cfg.INPUT.SAMPLING_FRAME_NUM == VIDEOMT_YTVIS19_CLIP_LENGTH
    assert cfg.INPUT.SAMPLING_FRAME_RANGE == 2
    assert cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING == "choice_by_clip"
    assert tuple(cfg.INPUT.MIN_SIZE_TRAIN) == VIDEOMT_YTVIS19_TRAIN_SHORT_EDGES
    assert cfg.INPUT.MAX_SIZE_TRAIN == VIDEOMT_YTVIS19_TRAIN_MAX_SIZE
    assert cfg.INPUT.CROP.ENABLED is True
    assert cfg.INPUT.CROP.TYPE == VIDEOMT_YTVIS19_CROP_TYPE
    assert tuple(cfg.INPUT.CROP.SIZE) == VIDEOMT_YTVIS19_CROP_SIZE
    assert cfg.INPUT.RANDOM_FLIP == "flip_by_clip"

    assert cfg.SOLVER.IMS_PER_BATCH == 8
    assert cfg.SOLVER.AMP.ENABLED is True
    assert cfg.SOLVER.BASE_LR == VIDEOMT_BASE_LR
    assert cfg.SOLVER.LLRD == VIDEOMT_LAYERWISE_LR_DECAY
    assert cfg.SOLVER.WEIGHT_DECAY == VIDEOMT_WEIGHT_DECAY
    assert tuple(cfg.SOLVER.WARMUP_STEPS) == (
        VIDEOMT_NON_VIT_WARMUP_STEPS,
        VIDEOMT_VIT_WARMUP_STEPS,
    )
    assert cfg.SOLVER.POLY_POWER == VIDEOMT_POLY_POWER
    assert cfg.SOLVER.MAX_ITER == VIDEOMT_RELEASED_TOTAL_STEPS


def test_frozen_upstream_trainer_does_not_activate_declared_gradient_clipping() -> None:
    source = (SNAPSHOT / "train_net_video.py").read_text(encoding="utf-8")
    build_optimizer = source.split("def build_optimizer(cls, cfg, model):", maxsplit=1)[1]
    build_optimizer = build_optimizer.split("def build_lr_scheduler", maxsplit=1)[0]
    assert "optimizer = AdamW(param_groups" in build_optimizer
    assert "maybe_add_gradient_clipping" not in build_optimizer
    assert "return optimizer" in build_optimizer
