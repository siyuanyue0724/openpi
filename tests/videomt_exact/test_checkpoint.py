from __future__ import annotations

from pathlib import Path

import pytest
import torch

from picf_next.videomt_exact import checkpoint as checkpoint_module
from picf_next.videomt_exact.checkpoint import hf_dinov3_state_from_published


def test_hf_conversion_only_reverses_upstream_aliases() -> None:
    state = {
        "backbone.encoder.backbone.patch_embed.cls_token": torch.randn(1, 1, 1024),
        "backbone.encoder.backbone.patch_embed.patch_embeddings.weight": torch.randn(
            1024, 3, 16, 16
        ),
        "backbone.encoder.backbone.blocks.0.norm1.weight": torch.randn(1024),
        "backbone.encoder.backbone.norm.weight": torch.randn(1024),
        "backbone.q.weight": torch.randn(200, 1024),
    }

    converted, sources = hf_dinov3_state_from_published(state)

    assert set(converted) == {
        "embeddings.cls_token",
        "embeddings.mask_token",
        "embeddings.patch_embeddings.weight",
        "layer.0.norm1.weight",
        "norm.weight",
    }
    torch.testing.assert_close(
        converted["embeddings.cls_token"],
        state["backbone.encoder.backbone.patch_embed.cls_token"],
    )
    assert not converted["embeddings.mask_token"].any()
    assert "constructor_zero_only" in sources["embeddings.mask_token"]


def test_hf_conversion_rejects_alias_collisions() -> None:
    state = {
        "backbone.encoder.backbone.patch_embed.cls_token": torch.randn(1, 1, 1024),
        "backbone.encoder.backbone.embeddings.cls_token": torch.randn(1, 1, 1024),
    }
    try:
        hf_dinov3_state_from_published(state)
    except ValueError as error:
        assert "collision" in str(error)
    else:
        raise AssertionError("conversion must reject two source tensors mapped to one target")


def test_adapted_checkpoint_requires_full_authenticated_tensor_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released_sha256 = "b" * 64
    state = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "buffer": torch.ones(2),
    }
    artifact = tmp_path / "adapted.pt"
    torch.save(
        {
            "schema": checkpoint_module.ADAPTED_MODEL_CHECKPOINT_SCHEMA,
            "source": {
                "checkpoint_schema": (
                    checkpoint_module.COMPLETE_DISTRIBUTED_CHECKPOINT_SCHEMA
                ),
                "checkpoint_sha256": "1" * 64,
                "report_schema": checkpoint_module.COMPLETE_TRAINING_REPORT_SCHEMA,
                "report_sha256": "2" * 64,
                "global_step": 250,
                "split_plan_sha256": "3" * 64,
                "implementation_sha256": "4" * 64,
                "dataset_manifest_sha256": "5" * 64,
                "physical_sidecar_manifest_sha256": "6" * 64,
                "released_checkpoint_sha256": released_sha256,
            },
            "model": state,
        },
        artifact,
    )
    monkeypatch.setattr(checkpoint_module, "ADAPTED_MODEL_TENSORS", len(state))
    monkeypatch.setattr(
        checkpoint_module,
        "ADAPTED_MODEL_NUMEL",
        sum(value.numel() for value in state.values()),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "PUBLISHED_CHECKPOINT_SHA256",
        released_sha256,
    )
    digest = checkpoint_module.sha256_file(artifact)

    receipt, loaded = checkpoint_module.adapted_videomt_model_state(
        artifact,
        expected_sha256=digest,
    )

    assert receipt.is_complete
    assert receipt.global_step == 250
    assert set(loaded) == set(state)
    torch.testing.assert_close(loaded["weight"], state["weight"])
    with pytest.raises(ValueError, match="SHA-256 differs"):
        checkpoint_module.adapted_videomt_model_state(
            artifact,
            expected_sha256="0" * 64,
        )
