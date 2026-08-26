from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from picf_next.lingbot_native.predictive_cache import (
    LINGBOT_PREDICTIVE_TARGET_SPACE,
    PredictiveCacheContract,
    native_predictive_coverage_digest,
    native_predictive_pair_keys_digest,
    native_predictive_query_schema_digest,
)
from tools.build_lingbot_calvin_predictive_cache import (
    OfficialLingBotDinoVideoExtractor,
    _extract_predictive_batch,
    _resolve_training_config,
    _VerifiedStaticFrame,
    _video_config,
    _write_build_report,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _contract() -> PredictiveCacheContract:
    pairs = ((10, 1),)
    horizons = (1,)
    dataset_digest = _sha("dataset")
    stream_plan_digest = _sha("stream-plan")
    temporal_digest = _sha("temporal")
    pair_digest = native_predictive_pair_keys_digest(pairs)
    return PredictiveCacheContract(
        dataset_id="calvin",
        dataset_revision="fixture",
        split_name="training",
        dataset_tree_sha256=dataset_digest,
        physical_sidecar_manifest_sha256=_sha("sidecar"),
        lingbot_source_commit="2838c",
        lingbot_checkpoint_revision="released",
        teacher_config_sha256=_sha("teacher-config"),
        teacher_checkpoint_sha256=_sha("teacher-checkpoint"),
        query_schema_sha256=native_predictive_query_schema_digest(
            target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
            route_id=0,
            horizons=horizons,
        ),
        horizons=horizons,
        stream_plan_sha256=stream_plan_digest,
        temporal_estimator_sha256=temporal_digest,
        pair_keys_sha256=pair_digest,
        coverage_sha256=native_predictive_coverage_digest(
            dataset_tree_sha256=dataset_digest,
            stream_plan_sha256=stream_plan_digest,
            temporal_estimator_sha256=temporal_digest,
            pair_keys_sha256=pair_digest,
            expected_record_count=len(pairs),
            horizons=horizons,
        ),
        expected_record_count=1,
    )


def test_training_config_defaults_to_selected_source_checkout(tmp_path: Path) -> None:
    source_checkout = tmp_path / "external-lingbot"
    expected = source_checkout / "configs/vla/robotwin/robotwin.yaml"
    explicit = tmp_path / "frozen-training.yaml"

    assert _resolve_training_config(source_checkout, None) == expected
    assert _resolve_training_config(source_checkout, explicit) == explicit


def _frame(
    global_index: int,
    *,
    rgb_value: int,
    owner_index: np.ndarray,
) -> _VerifiedStaticFrame:
    return _VerifiedStaticFrame(
        global_index=global_index,
        rgb=np.full((200, 200, 3), rgb_value, dtype=np.uint8),
        rgb_sha256=_sha(f"rgb-{global_index}"),
        physical=SimpleNamespace(identity_keys=("object/a", "object/b")),
        camera=SimpleNamespace(
            owner_index=owner_index,
            owner_supervised=np.ones((200, 200), dtype=np.bool_),
        ),
    )


def test_released_lingbot_video_config_matches_frozen_teacher_contract() -> None:
    checkout = ROOT / "references/source_checkouts/lingbot-vla-v2-adr74"
    if not checkout.exists():
        pytest.skip("optional pinned LingBot training config is absent")
    path = checkout / "configs/vla/robotwin/robotwin.yaml"
    config = _video_config(path)
    assert config["attention_mode"] == "flex_block_causal"
    assert config["input_size"] == 256
    assert config["num_backbone_tokens"] == 256
    assert config["dim_out"] == 1024


def test_predictive_build_report_is_atomic_and_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.build_report.json"
    report = {"cache": "verified", "records": 7}

    _write_build_report(path, report)

    assert path.read_text(encoding="ascii") == ('{\n  "cache": "verified",\n  "records": 7\n}\n')
    assert not tuple(tmp_path.glob(".*.tmp"))

    with pytest.raises(FileExistsError):
        _write_build_report(path, {"cache": "must-not-overwrite"})
    assert json.loads(path.read_text(encoding="ascii")) == report


def test_predictive_build_report_refuses_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("original\n", encoding="ascii")
    path = tmp_path / "cache.build_report.json"
    path.symlink_to(target)

    with pytest.raises(FileExistsError):
        _write_build_report(path, {"cache": "must-not-follow"})
    assert target.read_text(encoding="ascii") == "original\n"


def test_official_teacher_adapter_honors_requested_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
train:
  align_params:
    video:
      attention_mode: flex_block_causal
      input_size: 256
      num_future_frames: 1
      use_warmup_frame: true
      effective_fps: 1.0
      n_blocks: 1
      cls_pool: last
      num_backbone_tokens: 256
      dim_out: 1024
      use_patch_loss: true
      use_current_patch_loss: true
      use_cls_loss: false
""".lstrip(),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    module = types.ModuleType("lingbotvla.models.vla.vision_models.module_utils")

    def build_video_model(config: dict[str, object]) -> object:
        captured.update(config)
        return object()

    module.build_video_model = build_video_model  # type: ignore[attr-defined]
    module.get_video_target = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "lingbotvla.models.vla.vision_models.module_utils",
        module,
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())

    OfficialLingBotDinoVideoExtractor(
        source_checkout=tmp_path,
        checkpoint_dir=tmp_path,
        training_config=config_path,
        device=torch.device("cuda:1"),
    )

    assert captured["video"]["device"] == "cuda:1"  # type: ignore[index]


def test_official_teacher_adapter_exposes_same_call_current_and_future(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        """
train:
  align_params:
    video:
      attention_mode: flex_block_causal
      input_size: 256
      num_future_frames: 1
      use_warmup_frame: true
      effective_fps: 1.0
      n_blocks: 1
      cls_pool: last
      num_backbone_tokens: 256
      dim_out: 1024
      use_patch_loss: true
      use_current_patch_loss: true
      use_cls_loss: false
""".lstrip(),
        encoding="utf-8",
    )
    observed: dict[str, object] = {}
    module = types.ModuleType("lingbotvla.models.vla.vision_models.module_utils")
    module.build_video_model = lambda _config: object()  # type: ignore[attr-defined]

    def get_video_target(
        _teacher: object,
        current: torch.Tensor,
        future: torch.Tensor,
        _config: dict[str, object],
        *,
        effective_fps: torch.Tensor | float,
    ) -> dict[str, torch.Tensor | None]:
        observed["current"] = current.clone()
        observed["future"] = future.clone()
        observed["effective_fps"] = effective_fps
        shape = (current.shape[0], 256, 1024)
        return {
            "patch": torch.full(shape, 2.0, device=current.device),
            "current_patch": torch.full(shape, 1.0, device=current.device),
            "cls": None,
        }

    module.get_video_target = get_video_target  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "lingbotvla.models.vla.vision_models.module_utils",
        module,
    )
    monkeypatch.setattr(sys, "path", sys.path.copy())
    extractor = OfficialLingBotDinoVideoExtractor(
        source_checkout=tmp_path,
        checkpoint_dir=tmp_path,
        training_config=config_path,
        device=torch.device("cpu"),
    )
    current = torch.zeros(2, 1, 3, 16, 16)
    future = torch.ones(2, 1, 3, 16, 16)

    requested_fps = torch.tensor([30.0, 15.0])
    future_patch, current_patch = extractor.paired(
        current,
        future,
        effective_fps=requested_fps,
    )

    torch.testing.assert_close(observed["effective_fps"], requested_fps)
    assert torch.equal(observed["current"], current)
    assert torch.equal(observed["future"], future)
    assert future_patch.shape == current_patch.shape == (2, 256, 1024)
    assert future_patch.unique().item() == 2.0
    assert current_patch.unique().item() == 1.0
    assert torch.equal(
        extractor(current, future, effective_fps=requested_fps),
        future_patch,
    )


def test_batch_extraction_uses_future_physical_owner_without_task_conditioning() -> None:
    source_owner = np.zeros((200, 200), dtype=np.uint8)
    source_owner[:, :100] = 2
    target_owner = np.zeros((200, 200), dtype=np.uint8)
    target_owner[:, :100] = 1
    source = _frame(10, rgb_value=10, owner_index=source_owner)
    target = _frame(11, rgb_value=11, owner_index=target_owner)
    observed: dict[str, torch.Tensor] = {}

    def extractor(
        current: torch.Tensor,
        future: torch.Tensor,
        *,
        effective_fps: torch.Tensor,
    ) -> torch.Tensor:
        observed["current"] = current.clone()
        observed["future"] = future.clone()
        observed["effective_fps"] = effective_fps.clone()
        token_offsets = torch.arange(256, dtype=torch.float32)[:, None]
        channels = torch.arange(1024, dtype=torch.float32)[None, :]
        return (token_offsets + channels).unsqueeze(0)

    (record,) = _extract_predictive_batch(
        ((source, target, 1),),
        extractor=extractor,
        contract=_contract(),
    )

    assert observed["current"].shape == (1, 1, 3, 200, 200)
    assert observed["future"].shape == (1, 1, 3, 200, 200)
    assert observed["current"].unique().item() == 10
    assert observed["future"].unique().item() == 11
    torch.testing.assert_close(observed["effective_fps"], torch.tensor([30.0]))
    assert record.source_global_index == 10
    assert record.target_global_index == 11
    assert record.identity_keys == ("object/a", "object/b")
    assert record.importance[0] == pytest.approx(0.5)
    assert record.importance[1] == pytest.approx(0.0)
    assert np.any(record.features[0] != 0)
    assert not np.any(record.features[1])
