from __future__ import annotations

import pytest
import torch

from openpi.models_pytorch.transformers_replace.models.paligemma.safe_ops import merge_image_features_dense
from openpi.models_pytorch.transformers_replace.models.paligemma.safe_ops import replace_oov_image_tokens
from openpi.models_pytorch.gemma_pytorch import _apply_tokenwise_in_chunks as _gemma_apply_tokenwise_in_chunks
from openpi.models_pytorch.gemma_pytorch import _gated_residual
from openpi.models_pytorch.pi0_pytorch import _ensure_transformers_replace_is_ready
from openpi.picf.paligemma.wrapper import _checkpoint_inputs_require_grad
from openpi.picf.paligemma.wrapper import _enable_gradient_checkpointing_non_reentrant
from openpi.picf.paligemma.wrapper import _masked_position_ids
from openpi.picf.paligemma.wrapper import _recover_flow_target
from openpi.picf.paligemma.wrapper import _repair_missing_tied_embeddings
from openpi.picf.paligemma.wrapper import _stage_local_pi0_config
from openpi.picf.paligemma.wrapper import _stage_pi0_checkpoint_if_needed
from openpi.picf.paligemma.wrapper import _take_valid_prefix_tokens
from openpi.picf.paligemma.wrapper import _Pi0PaliGemmaSemanticEncoder
from openpi.picf.paligemma.wrapper import PaliGemmaSemanticEncoder
from openpi.picf.paligemma.config import PaliGemmaSemanticConfig


class _TinyLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)


class _TinyPaliGemma(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = _TinyLanguageModel()
        self.lm_head = torch.nn.Linear(3, 4, bias=False)


class _TinyGemmaExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(3, 4, bias=False)


class _TinyPaliGemmaWithExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = _TinyPaliGemma()
        self.gemma_expert = _TinyGemmaExpert()


class _TinyRuntimeSelfAttn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 3)
        self.k_proj = torch.nn.Linear(3, 3)
        self.v_proj = torch.nn.Linear(3, 3)
        self.o_proj = torch.nn.Linear(3, 3)


class _TinyRuntimeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyRuntimeSelfAttn()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.GELU(), torch.nn.Linear(6, 3))


class _TinyRuntimeLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 3)
        self.layers = torch.nn.ModuleList([_TinyRuntimeLayer(), _TinyRuntimeLayer()])


class _TinyRuntimeGemmaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyRuntimeLayer(), _TinyRuntimeLayer()])


class _TinyRuntimePaligemma(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyRuntimeLanguageModel()
        self.vision_tower = torch.nn.Sequential(torch.nn.Linear(3, 3))
        self.multi_modal_projector = torch.nn.Linear(3, 3)


class _TinyRuntimeGemmaExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyRuntimeGemmaModel()


class _TinyRuntimePaliGemmaWithExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma = _TinyRuntimePaligemma()
        self.gemma_expert = _TinyRuntimeGemmaExpert()


def test_repair_missing_tied_embeddings_copies_lm_head_weights() -> None:
    model = _TinyPaliGemma()
    with torch.no_grad():
        model.model.language_model.embed_tokens.weight.zero_()
        model.lm_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            )
        )

    remaining = _repair_missing_tied_embeddings(
        model,
        missing_keys=["model.language_model.embed_tokens.weight"],
    )

    assert remaining == []
    torch.testing.assert_close(
        model.model.language_model.embed_tokens.weight,
        model.lm_head.weight,
    )


def test_repair_missing_tied_embeddings_leaves_unrelated_missing_keys() -> None:
    model = _TinyPaliGemma()
    remaining = _repair_missing_tied_embeddings(
        model,
        missing_keys=["some.other.weight"],
    )
    assert remaining == ["some.other.weight"]


def test_drop_unused_generation_heads_removes_runtime_dead_lm_heads() -> None:
    class _DummyEncoder:
        pass

    encoder = _DummyEncoder()
    encoder.paligemma_with_expert = _TinyPaliGemmaWithExpert()

    _Pi0PaliGemmaSemanticEncoder._drop_unused_generation_heads(encoder)

    assert encoder.paligemma_with_expert.paligemma.lm_head is None
    assert encoder.paligemma_with_expert.gemma_expert.lm_head is None


def test_fsdp_runtime_leaf_module_specs_cover_direct_call_hot_path_modules() -> None:
    class _DummyEncoder:
        pass

    encoder = _DummyEncoder()
    encoder.paligemma_with_expert = _TinyRuntimePaliGemmaWithExpert()
    encoder.action_in_proj = torch.nn.Linear(7, 3)
    encoder.action_out_proj = torch.nn.Linear(3, 7)
    encoder.time_mlp_in = torch.nn.Linear(3, 3)
    encoder.time_mlp_out = torch.nn.Linear(3, 3)

    specs = _Pi0PaliGemmaSemanticEncoder.fsdp_runtime_leaf_module_specs(encoder)

    assert ("embed_tokens", "uniform_recursive") in [(name, mode) for _, name, mode in specs]
    assert ("vision_tower", "mixed_root") not in [(name, mode) for _, name, mode in specs]
    assert ("multi_modal_projector", "uniform_recursive") not in [(name, mode) for _, name, mode in specs]
    assert sum(1 for _, name, _ in specs if name == "q_proj") == 4
    assert sum(1 for _, name, _ in specs if name == "mlp") == 4
    assert ("action_out_proj", "uniform_recursive") in [(name, mode) for _, name, mode in specs]


def test_outer_semantic_encoder_proxies_runtime_leaf_specs() -> None:
    class _DummyInner:
        def fsdp_runtime_leaf_module_specs(self):
            return [("parent", "child", "uniform_recursive")]

    outer = object.__new__(PaliGemmaSemanticEncoder)
    outer.encoder = _DummyInner()

    assert PaliGemmaSemanticEncoder.fsdp_runtime_leaf_module_specs(outer) == [
        ("parent", "child", "uniform_recursive")
    ]


def test_build_paligemma_with_expert_passes_split_chunk_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.config = PaliGemmaSemanticConfig(
        projection_chunk_size=96,
        mlp_chunk_size=48,
        tokenwise_chunk_size=64,
    )

    captured: dict[str, int] = {}

    import openpi.picf.paligemma.wrapper as wrapper_mod

    monkeypatch.setattr(wrapper_mod._gemma, "get_config", lambda name: name)

    class _DummyModel:
        def __init__(
            self,
            paligemma_config,
            action_expert_config,
            *,
            use_adarms,
            precision,
            tokenwise_chunk_size,
            projection_chunk_size,
            mlp_chunk_size,
        ) -> None:
            captured["tokenwise_chunk_size"] = tokenwise_chunk_size
            captured["projection_chunk_size"] = projection_chunk_size
            captured["mlp_chunk_size"] = mlp_chunk_size
            captured["use_adarms"] = int(bool(use_adarms[1]))
            captured["precision_is_bf16"] = int(precision == "bfloat16")

    monkeypatch.setattr(wrapper_mod, "PaliGemmaWithExpertModel", _DummyModel)

    _Pi0PaliGemmaSemanticEncoder._build_paligemma_with_expert(
        encoder,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        precision="bfloat16",
        pi05=True,
    )

    assert captured["tokenwise_chunk_size"] == 64
    assert captured["projection_chunk_size"] == 96
    assert captured["mlp_chunk_size"] == 48
    assert captured["use_adarms"] == 1
    assert captured["precision_is_bf16"] == 1


def test_build_paligemma_with_expert_falls_back_to_legacy_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.config = PaliGemmaSemanticConfig(
        projection_chunk_size=None,
        mlp_chunk_size=None,
        tokenwise_chunk_size=72,
    )

    captured: dict[str, int] = {}

    import openpi.picf.paligemma.wrapper as wrapper_mod

    monkeypatch.setattr(wrapper_mod._gemma, "get_config", lambda name: name)

    class _DummyModel:
        def __init__(
            self,
            paligemma_config,
            action_expert_config,
            *,
            use_adarms,
            precision,
            tokenwise_chunk_size,
            projection_chunk_size,
            mlp_chunk_size,
        ) -> None:
            captured["tokenwise_chunk_size"] = tokenwise_chunk_size
            captured["projection_chunk_size"] = projection_chunk_size
            captured["mlp_chunk_size"] = mlp_chunk_size

    monkeypatch.setattr(wrapper_mod, "PaliGemmaWithExpertModel", _DummyModel)

    _Pi0PaliGemmaSemanticEncoder._build_paligemma_with_expert(
        encoder,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        precision="bfloat16",
        pi05=False,
    )

    assert captured["tokenwise_chunk_size"] == 72
    assert captured["projection_chunk_size"] is None
    assert captured["mlp_chunk_size"] is None


def test_stage_pi0_checkpoint_if_needed_copies_to_local_cache_when_forced(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "pi0_src"
    cache_dir = tmp_path / "pi0_cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    checkpoint = source_dir / "model.safetensors"
    config_json = source_dir / "config.json"
    checkpoint.write_bytes(b"checkpoint-bytes")
    config_json.write_text('{"paligemma_variant":"gemma_2b"}', encoding="utf-8")

    import openpi.picf.paligemma.wrapper as wrapper_mod

    monkeypatch.setenv("OPENPI_STAGE_PI0_CHECKPOINT", "1")
    monkeypatch.setenv("OPENPI_LOCAL_CHECKPOINT_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(wrapper_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(wrapper_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(wrapper_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(wrapper_mod.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(wrapper_mod.dist, "barrier", lambda: None)

    staged_checkpoint, staged_config = _stage_pi0_checkpoint_if_needed(checkpoint, config_json)

    assert staged_checkpoint != checkpoint
    assert staged_checkpoint.read_bytes() == checkpoint.read_bytes()
    assert staged_config is not None
    assert staged_config.read_text(encoding="utf-8") == config_json.read_text(encoding="utf-8")
    assert str(staged_checkpoint).startswith(str(cache_dir))


def test_stage_local_pi0_config_rewrites_checkpoint_and_config_paths(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "pi0_cfg"
    cache_dir = tmp_path / "pi0_cfg_cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    checkpoint = source_dir / "model.safetensors"
    config_json = source_dir / "config.json"
    checkpoint.write_bytes(b"checkpoint")
    config_json.write_text("{}", encoding="utf-8")

    import openpi.picf.paligemma.wrapper as wrapper_mod

    monkeypatch.setenv("OPENPI_STAGE_PI0_CHECKPOINT", "1")
    monkeypatch.setenv("OPENPI_LOCAL_CHECKPOINT_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(wrapper_mod.dist, "is_available", lambda: False)
    monkeypatch.setattr(wrapper_mod.dist, "is_initialized", lambda: False)

    config = PaliGemmaSemanticConfig(
        source="pi0_pytorch",
        checkpoint_path=str(source_dir),
        checkpoint_config_path=str(config_json),
    )

    staged = _stage_local_pi0_config(config)

    assert staged != config
    assert staged.checkpoint_path is not None
    assert staged.checkpoint_config_path is not None
    assert staged.checkpoint_path.startswith(str(cache_dir))
    assert staged.checkpoint_config_path.startswith(str(cache_dir))


def test_prepare_image_accepts_resize_with_pad_batch1_squeeze(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32

    def _fake_resize(_images, _h, _w):
        return torch.zeros((224, 224, 3), dtype=torch.float32)

    import openpi.picf.paligemma.wrapper as wrapper_mod

    monkeypatch.setattr(wrapper_mod.image_tools, "resize_with_pad_torch", _fake_resize)
    image = torch.zeros((32, 48, 3), dtype=torch.float32).numpy()
    out = _Pi0PaliGemmaSemanticEncoder._prepare_image(encoder, image)
    assert tuple(out.shape) == (1, 3, 224, 224)


def test_checkpoint_inputs_require_grad_detects_tensor_inputs() -> None:
    no_grad = torch.zeros(2)
    requires_grad = torch.zeros(2, requires_grad=True)

    assert _checkpoint_inputs_require_grad(no_grad) is False
    assert _checkpoint_inputs_require_grad(no_grad, requires_grad) is True


def test_enable_gradient_checkpointing_prefers_non_reentrant() -> None:
    class _DummyModule:
        def __init__(self) -> None:
            self.kwargs = None

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.kwargs = gradient_checkpointing_kwargs

    module = _DummyModule()
    enabled, non_reentrant = _enable_gradient_checkpointing_non_reentrant(module)

    assert enabled is True
    assert non_reentrant is True
    assert module.kwargs == {"use_reentrant": False}


def test_enable_gradient_checkpointing_falls_back_when_kwargs_unsupported() -> None:
    class _LegacyModule:
        def __init__(self) -> None:
            self.calls = 0

        def gradient_checkpointing_enable(self):
            self.calls += 1

    module = _LegacyModule()
    enabled, non_reentrant = _enable_gradient_checkpointing_non_reentrant(module)

    assert enabled is True
    assert non_reentrant is False
    assert module.calls == 1


def test_gemma_tokenwise_chunk_helper_matches_direct_apply() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(3, 5)
    x = torch.randn(2, 7, 3)
    expected = layer(x)
    actual = _gemma_apply_tokenwise_in_chunks(x, layer, chunk_size=3)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_apply_checkpoint_skips_checkpoint_when_inputs_have_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.trainable = True
    encoder.training = True
    encoder.gradient_checkpointing_enabled = True

    def _boom(*args, **kwargs):
        raise AssertionError("checkpoint should not be invoked for non-grad inputs")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _boom)
    output = _Pi0PaliGemmaSemanticEncoder._apply_checkpoint(encoder, lambda x: x + 1, torch.zeros(2))
    torch.testing.assert_close(output, torch.ones(2))


def test_apply_checkpoint_uses_checkpoint_when_inputs_require_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.trainable = True
    encoder.training = True
    encoder.gradient_checkpointing_enabled = True
    called = {"value": False}

    def _checkpoint(func, *args, **kwargs):
        called["value"] = True
        return func(*args)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _checkpoint)
    x = torch.zeros(2, requires_grad=True)
    output = _Pi0PaliGemmaSemanticEncoder._apply_checkpoint(encoder, lambda y: y + 2, x)
    torch.testing.assert_close(output, torch.full((2,), 2.0))
    assert called["value"] is True


def test_apply_checkpoint_skips_outer_checkpoint_when_native_gradient_checkpointing_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NativeGC(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gradient_checkpointing = True

    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.trainable = True
    encoder.training = True
    encoder.gradient_checkpointing_enabled = True
    encoder.paligemma_with_expert = type(
        "_Runtime",
        (),
        {
            "paligemma": type(
                "_Paligemma",
                (),
                {
                    "language_model": _NativeGC(),
                    "vision_tower": _NativeGC(),
                },
            )(),
            "gemma_expert": type("_Expert", (), {"model": _NativeGC()})(),
        },
    )()

    def _boom(*args, **kwargs):
        raise AssertionError("outer checkpoint should be skipped when native checkpointing is already active")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _boom)
    x = torch.zeros(2, requires_grad=True)
    output = _Pi0PaliGemmaSemanticEncoder._apply_checkpoint(encoder, lambda y: y + 3, x)
    torch.testing.assert_close(output, torch.full((2,), 3.0))


def test_masked_position_ids_keep_valid_positions_and_zero_pad() -> None:
    pad_mask = torch.tensor([[True, True, True, False, False], [True, False, True, False, False]])

    position_ids = _masked_position_ids(pad_mask)

    torch.testing.assert_close(
        position_ids,
        torch.tensor([[0, 1, 2, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.int64),
    )


def test_recover_flow_target_inverts_pi05_training_parameterization() -> None:
    target = torch.tensor([[[0.2, -0.1], [0.4, 0.3]]], dtype=torch.float32)
    noise = torch.tensor([[[1.0, -0.5], [0.7, 0.8]]], dtype=torch.float32)
    time = torch.tensor([0.25], dtype=torch.float32)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * target
    u_t = noise - target

    recovered = _recover_flow_target(x_t, u_t, time_expanded)

    torch.testing.assert_close(recovered, target)


def test_take_valid_prefix_tokens_uses_right_padded_prefix_slice() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    pad_mask = torch.tensor([True, True, True, True, False, False])
    got = _take_valid_prefix_tokens(hidden, pad_mask)
    torch.testing.assert_close(got, hidden[:4])


def test_take_valid_prefix_tokens_rejects_non_right_padded_mask() -> None:
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    pad_mask = torch.tensor([True, False, True, False, False, False])
    with pytest.raises(RuntimeError, match="right-padded"):
        _take_valid_prefix_tokens(hidden, pad_mask)


def test_replace_oov_image_tokens_uses_where_without_boolean_setitem() -> None:
    input_ids = torch.tensor([[5, 9, 1, 9], [9, 2, 3, 9]], dtype=torch.int64)
    replaced, mask = replace_oov_image_tokens(input_ids, image_token_id=9, vocab_size=8)
    torch.testing.assert_close(mask, input_ids == 9)
    torch.testing.assert_close(
        replaced,
        torch.tensor([[5, 0, 1, 0], [0, 2, 3, 0]], dtype=torch.int64),
    )


def test_merge_image_features_dense_matches_special_image_slots() -> None:
    inputs_embeds = torch.zeros((1, 6, 2), dtype=torch.float32)
    input_ids = torch.tensor([[9, 9, 4, 5, 9, 9]], dtype=torch.int64)
    image_features = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]], dtype=torch.float32)

    merged = merge_image_features_dense(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        image_features=image_features,
        image_token_id=9,
    )

    torch.testing.assert_close(
        merged,
        torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [0.0, 0.0], [0.0, 0.0], [3.0, 30.0], [4.0, 40.0]]],
            dtype=torch.float32,
        ),
    )


def test_build_paligemma_with_expert_uses_runtime_patched_gemma_classes() -> None:
    _ensure_transformers_replace_is_ready()
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    model = _Pi0PaliGemmaSemanticEncoder._build_paligemma_with_expert(
        encoder,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        precision="float32",
        pi05=True,
    )

    layer = model.gemma_expert.model.layers[0]
    assert hasattr(layer.input_layernorm, "dense")
    assert hasattr(layer.post_attention_layernorm, "dense")


def test_gated_residual_compat_fallback_matches_pi05_semantics() -> None:
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    y = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    gate = torch.tensor([[0.5, 2.0]], dtype=torch.float32)

    got = _gated_residual(x, y, gate)

    torch.testing.assert_close(got, x + y * gate)
