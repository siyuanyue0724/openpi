from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from picf_next.contracts import ContractError
from picf_next.encoders.vjepa2 import (
    VJEPA2_MODEL_ID,
    VJEPA2_MODEL_REVISION,
    Vjepa2DenseEncoder,
    _strict_checkpoint_revision,
    _strict_positive_config_int,
    vjepa2_context_only_role,
    vjepa2_dense_geometry,
    vjepa2_dense_timestamps,
)


def test_vjepa2_geometry_matches_conv3d_flatten_order() -> None:
    geometry = vjepa2_dense_geometry(
        frame_count=4,
        image_height=4,
        image_width=4,
        tubelet_size=2,
        patch_size=2,
    )
    assert geometry.shape == (8, 3)
    np.testing.assert_allclose(geometry[0], [0.25, 0.25, 0.25])
    np.testing.assert_allclose(geometry[3], [0.25, 0.75, 0.75])
    np.testing.assert_allclose(geometry[4], [0.75, 0.25, 0.25])
    assert not geometry.flags.writeable


def test_vjepa2_timestamps_repeat_each_tubelet_center_over_spatial_grid() -> None:
    timestamps = vjepa2_dense_timestamps(
        [0.0, 0.1, 0.2, 0.3],
        tubelet_size=2,
        patches_per_frame=4,
    )
    np.testing.assert_allclose(timestamps, [0.05] * 4 + [0.25] * 4)
    assert timestamps.dtype == np.float32
    assert not timestamps.flags.writeable


def test_vjepa2_clip_features_are_read_only_action_context() -> None:
    role = vjepa2_context_only_role(8)
    assert role.dtype == np.bool_
    assert not role.any()
    assert not role.flags.writeable


def test_vjepa2_boundary_rejects_partial_tubelets_and_non_monotonic_time() -> None:
    with pytest.raises(ContractError, match="must divide"):
        vjepa2_dense_geometry(
            frame_count=3,
            image_height=4,
            image_width=4,
            tubelet_size=2,
            patch_size=2,
        )
    with pytest.raises(ContractError, match="strictly increasing"):
        vjepa2_dense_timestamps([0.0, 0.1, 0.1, 0.2], tubelet_size=2, patches_per_frame=4)


@pytest.mark.parametrize("value", [True, 64.0, "64", 0, -1])
def test_vjepa2_checkpoint_dimensions_require_exact_positive_integers(value: object) -> None:
    with pytest.raises(RuntimeError, match="must be a positive integer"):
        _strict_positive_config_int("frames_per_clip", value)


@pytest.mark.parametrize("revision", [None, "main", "A" * 40, "g" * 40, "a" * 39])
def test_vjepa2_checkpoint_revision_requires_an_exact_commit(revision: object) -> None:
    with pytest.raises(ValueError, match="exact lowercase commit SHA"):
        _strict_checkpoint_revision(revision)


def _install_fake_pretrained_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_revision: str,
) -> tuple[list[tuple[str, str, dict[str, object]]], object]:
    calls: list[tuple[str, str, dict[str, object]]] = []
    float32 = object()
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.float16 = object()
    fake_torch.float32 = float32

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                _commit_hash=resolved_revision,
                frames_per_clip=64,
                image_size=256,
                tubelet_size=2,
                patch_size=16,
                hidden_size=1024,
            )
            self.frozen = False
            self.training = True
            self.device = None

        def requires_grad_(self, enabled: bool) -> FakeModel:
            self.frozen = not enabled
            return self

        def eval(self) -> FakeModel:
            self.training = False
            return self

        def to(self, device: str) -> FakeModel:
            self.device = device
            return self

    model = FakeModel()

    class FakeProcessorLoader:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            calls.append(("processor", model_id, kwargs))
            return object()

    class FakeModelLoader:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            calls.append(("model", model_id, kwargs))
            return model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoVideoProcessor = FakeProcessorLoader
    fake_transformers.AutoModel = FakeModelLoader
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return calls, float32


def test_vjepa2_loader_requests_and_verifies_one_immutable_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, float32 = _install_fake_pretrained_modules(
        monkeypatch,
        resolved_revision=VJEPA2_MODEL_REVISION,
    )

    encoder = Vjepa2DenseEncoder.from_pretrained(local_files_only=True)

    assert encoder.model_id == VJEPA2_MODEL_ID
    assert encoder.checkpoint_revision == VJEPA2_MODEL_REVISION
    assert encoder.encoder_contract.startswith(f"{VJEPA2_MODEL_ID}@{VJEPA2_MODEL_REVISION}/")
    assert [kind for kind, _model_id, _kwargs in calls] == ["processor", "model"]
    for _kind, model_id, kwargs in calls:
        assert model_id == VJEPA2_MODEL_ID
        assert kwargs["revision"] == VJEPA2_MODEL_REVISION
        assert kwargs["local_files_only"] is True
    assert calls[1][2]["dtype"] is float32


def test_vjepa2_loader_rejects_revision_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pretrained_modules(monkeypatch, resolved_revision="f" * 40)

    with pytest.raises(RuntimeError, match="differs from the requested immutable revision"):
        Vjepa2DenseEncoder.from_pretrained()
