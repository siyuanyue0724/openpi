from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.models_pytorch.transformers_replace.models.paligemma.safe_ops import merge_image_features_dense
from openpi.models_pytorch.transformers_replace.models.paligemma.safe_ops import replace_oov_image_tokens
from openpi.models_pytorch.gemma_pytorch import _apply_tokenwise_in_chunks as _gemma_apply_tokenwise_in_chunks
from openpi.models_pytorch.gemma_pytorch import _gated_residual
from openpi.models_pytorch.pi0_pytorch import _ensure_transformers_replace_is_ready
from openpi.picf.action_normalization import PicfStateNormalizer
from openpi.picf.contracts import PicfObservation
from openpi.picf.paligemma.wrapper import _checkpoint_inputs_require_grad
from openpi.picf.paligemma.wrapper import _action_flow_objective_loss
from openpi.picf.paligemma.wrapper import _enable_gradient_checkpointing_non_reentrant
from openpi.picf.paligemma.wrapper import _masked_position_ids
from openpi.picf.paligemma.wrapper import _recover_flow_target
from openpi.picf.paligemma.wrapper import _repair_missing_tied_embeddings
from openpi.picf.paligemma.wrapper import _stage_local_pi0_config
from openpi.picf.paligemma.wrapper import _stage_pi0_checkpoint_if_needed
from openpi.picf.paligemma.wrapper import _take_valid_prefix_tokens
from openpi.picf.paligemma.wrapper import _Pi0PaliGemmaSemanticEncoder
from openpi.picf.paligemma.wrapper import PaliGemmaSemanticFeatures
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


class _TinyWrappedLinear(torch.nn.Module):
    """FSDP-like wrapper that forwards to a Linear but is not an nn.Linear."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.module = linear

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.module(inputs)


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
    encoder.action_context_in_proj = torch.nn.Linear(6, 3, bias=False)
    encoder.action_context_q_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_k_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_v_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_out_proj = torch.nn.Linear(3, 3, bias=False)

    specs = _Pi0PaliGemmaSemanticEncoder.fsdp_runtime_leaf_module_specs(encoder)

    assert ("embed_tokens", "uniform_recursive") in [(name, mode) for _, name, mode in specs]
    assert ("vision_tower", "mixed_root") not in [(name, mode) for _, name, mode in specs]
    assert ("multi_modal_projector", "uniform_recursive") not in [(name, mode) for _, name, mode in specs]
    assert sum(1 for _, name, _ in specs if name == "q_proj") == 4
    assert sum(1 for _, name, _ in specs if name == "mlp") == 4
    assert ("action_out_proj", "uniform_recursive") in [(name, mode) for _, name, mode in specs]
    assert "action_context_in_proj" not in [name for _, name, _ in specs]
    assert "action_context_q_proj" not in [name for _, name, _ in specs]
    assert "action_context_out_proj" not in [name for _, name, _ in specs]


def test_pi0_trainable_scope_action_head_only_freezes_semantic_stack() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.trainable = True
    encoder.trainable_scope = "action_head_only"
    encoder.paligemma_with_expert = _TinyRuntimePaliGemmaWithExpert()
    encoder.action_in_proj = torch.nn.Linear(7, 3)
    encoder.action_out_proj = torch.nn.Linear(3, 7)
    encoder.time_mlp_in = torch.nn.Linear(3, 3)
    encoder.time_mlp_out = torch.nn.Linear(3, 3)

    _Pi0PaliGemmaSemanticEncoder._apply_trainable_scope(encoder)

    trainable = {name for name, param in encoder.named_parameters() if param.requires_grad}
    assert "paligemma_with_expert.paligemma.language_model.embed_tokens.weight" not in trainable
    assert "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight" not in trainable
    assert "action_in_proj.weight" in trainable
    assert "action_out_proj.weight" in trainable
    assert "time_mlp_in.weight" in trainable
    assert "time_mlp_out.weight" in trainable


def test_pi0_trainable_scope_all_unfreezes_semantic_stack() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.trainable = True
    encoder.trainable_scope = "all"
    encoder.paligemma_with_expert = _TinyRuntimePaliGemmaWithExpert()
    encoder.action_in_proj = torch.nn.Linear(7, 3)
    encoder.action_out_proj = torch.nn.Linear(3, 7)
    encoder.time_mlp_in = torch.nn.Linear(3, 3)
    encoder.time_mlp_out = torch.nn.Linear(3, 3)

    _Pi0PaliGemmaSemanticEncoder._apply_trainable_scope(encoder)

    assert all(param.requires_grad for param in encoder.parameters())


def test_pi0_trainable_scope_backbone_only_matches_historical_full_cotrain_boundary() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.trainable = True
    encoder.trainable_scope = "backbone_only"
    encoder.paligemma_with_expert = _TinyRuntimePaliGemmaWithExpert()
    encoder.action_in_proj = torch.nn.Linear(7, 3)
    encoder.action_out_proj = torch.nn.Linear(3, 7)
    encoder.time_mlp_in = torch.nn.Linear(3, 3)
    encoder.time_mlp_out = torch.nn.Linear(3, 3)

    _Pi0PaliGemmaSemanticEncoder._apply_trainable_scope(encoder)

    trainable = {name for name, param in encoder.named_parameters() if param.requires_grad}
    assert "paligemma_with_expert.paligemma.language_model.embed_tokens.weight" in trainable
    assert "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight" in trainable
    assert "action_in_proj.weight" not in trainable
    assert "action_out_proj.weight" not in trainable
    assert "time_mlp_in.weight" not in trainable
    assert "time_mlp_out.weight" not in trainable


def test_pi0_trainable_scope_action_adapter_only_trains_only_context_adapter() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.trainable = True
    encoder.trainable_scope = "action_adapter_only"
    encoder.action_expert_router_enabled = False
    encoder.paligemma_with_expert = _TinyRuntimePaliGemmaWithExpert()
    encoder.action_in_proj = torch.nn.Linear(7, 3)
    encoder.action_out_proj = torch.nn.Linear(3, 7)
    encoder.time_mlp_in = torch.nn.Linear(3, 3)
    encoder.time_mlp_out = torch.nn.Linear(3, 3)
    encoder.action_context_in_proj = torch.nn.Linear(6, 3, bias=False)
    encoder.action_context_q_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_k_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_v_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_out_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_gate_logit = torch.nn.Parameter(torch.tensor([-2.0]))
    encoder.action_context_readout_query = torch.nn.Parameter(torch.randn(2, 3))
    encoder.action_context_readout_q_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_readout_k_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_readout_v_proj = torch.nn.Linear(3, 3, bias=False)
    encoder.action_context_readout_out_proj = torch.nn.Linear(3, 7)
    encoder.action_context_flow_residual_gate_logit = torch.nn.Parameter(torch.tensor([-2.0]))

    _Pi0PaliGemmaSemanticEncoder._apply_trainable_scope(encoder)

    trainable = {name for name, param in encoder.named_parameters() if param.requires_grad}
    assert "paligemma_with_expert.paligemma.language_model.embed_tokens.weight" not in trainable
    assert "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight" not in trainable
    assert "action_in_proj.weight" not in trainable
    assert "action_out_proj.weight" not in trainable
    assert "time_mlp_in.weight" not in trainable
    assert "time_mlp_out.weight" not in trainable
    assert "action_context_in_proj.weight" in trainable
    assert "action_context_q_proj.weight" in trainable
    assert "action_context_k_proj.weight" in trainable
    assert "action_context_v_proj.weight" in trainable
    assert "action_context_out_proj.weight" in trainable
    assert "action_context_gate_logit" in trainable
    assert "action_context_readout_query" in trainable
    assert "action_context_readout_q_proj.weight" in trainable
    assert "action_context_readout_k_proj.weight" in trainable
    assert "action_context_readout_v_proj.weight" in trainable
    assert "action_context_readout_out_proj.weight" in trainable
    assert "action_context_flow_residual_gate_logit" in trainable


def test_pi0_trainable_scope_action_adapter_only_trains_router_when_enabled() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.trainable = True
    encoder.trainable_scope = "action_adapter_only"
    encoder.action_expert_router_enabled = True
    encoder.paligemma_with_expert = _TinyRuntimePaliGemmaWithExpert()
    encoder.action_in_proj = torch.nn.Linear(7, 4)
    encoder.action_out_proj = torch.nn.Linear(4, 7)
    encoder.time_mlp_in = torch.nn.Linear(4, 4)
    encoder.time_mlp_out = torch.nn.Linear(4, 4)
    encoder.action_context_in_proj = torch.nn.Linear(6, 4, bias=False)
    encoder.action_context_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_out_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_gate_logit = torch.nn.Parameter(torch.tensor([-2.0]))
    encoder.action_context_readout_query = torch.nn.Parameter(torch.randn(2, 4))
    encoder.action_context_readout_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_out_proj = torch.nn.Linear(4, 7)
    encoder.action_context_flow_residual_gate_logit = torch.nn.Parameter(torch.tensor([-2.0]))
    encoder.action_expert_router_summary_proj = torch.nn.Linear(6, 4, bias=False)
    encoder.action_expert_router_summary_pair_proj = torch.nn.Linear(12, 4, bias=False)
    encoder.action_expert_router_norm = torch.nn.LayerNorm(4)
    encoder.action_expert_router_logits = torch.nn.Linear(4, 3)
    encoder.action_expert_router_down = torch.nn.ModuleList(
        [torch.nn.Linear(4, 2, bias=False) for _ in range(3)]
    )
    encoder.action_expert_router_up = torch.nn.ModuleList(
        [torch.nn.Linear(2, 4, bias=False) for _ in range(3)]
    )
    encoder.action_expert_router_gate_logit = torch.nn.Parameter(torch.tensor([-2.5]))

    _Pi0PaliGemmaSemanticEncoder._apply_trainable_scope(encoder)

    trainable = {name for name, param in encoder.named_parameters() if param.requires_grad}
    assert "action_in_proj.weight" not in trainable
    assert "action_context_q_proj.weight" in trainable
    assert "action_context_readout_query" in trainable
    assert "action_context_readout_out_proj.weight" in trainable
    assert "action_context_flow_residual_gate_logit" in trainable
    assert "action_expert_router_summary_proj.weight" in trainable
    assert "action_expert_router_summary_pair_proj.weight" in trainable
    assert "action_expert_router_logits.weight" in trainable
    assert "action_expert_router_down.0.weight" in trainable
    assert "action_expert_router_up.0.weight" in trainable
    assert "action_expert_router_gate_logit" in trainable


def test_pi0_action_context_adapter_keeps_suffix_shape_and_reports_metrics() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.action_context_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_out_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_gate_logit = torch.nn.Parameter(torch.tensor([0.0]))
    encoder.action_context_adapter_rms_cap = True
    _Pi0PaliGemmaSemanticEncoder._reset_action_context_adapter_parameters(encoder)

    suffix = torch.randn((2, 3, 4), dtype=torch.float32)
    context = torch.randn((2, 5, 4), dtype=torch.float32)
    adapted, metrics = _Pi0PaliGemmaSemanticEncoder._apply_action_context_adapter(encoder, suffix, context)

    assert adapted.shape == suffix.shape
    assert not torch.allclose(adapted, suffix)
    assert metrics["picf_action_context_adapter_token_count"].item() == pytest.approx(5.0)
    assert metrics["picf_action_context_adapter_gate"].item() == pytest.approx(0.5)
    assert torch.isfinite(metrics["picf_action_context_adapter_attention_entropy_mean"])
    assert torch.isfinite(metrics["picf_action_context_adapter_residual_rms_mean"])


def test_pi0_action_context_adapter_projects_prefix_width_context_to_action_width() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.action_context_in_proj = torch.nn.Linear(6, 4, bias=False)
    encoder.action_context_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_out_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_gate_logit = torch.nn.Parameter(torch.tensor([0.0]))
    encoder.action_context_adapter_rms_cap = True
    _Pi0PaliGemmaSemanticEncoder._reset_action_context_adapter_parameters(encoder)

    suffix = torch.randn((1, 3, 4), dtype=torch.float32)
    prefix_width_context = torch.randn((1, 5, 6), dtype=torch.float32)
    adapted, metrics = _Pi0PaliGemmaSemanticEncoder._apply_action_context_adapter(
        encoder,
        suffix,
        prefix_width_context,
    )

    assert adapted.shape == suffix.shape
    assert metrics["picf_action_context_adapter_token_count"].item() == pytest.approx(5.0)


def test_pi0_action_context_adapter_accepts_wrapped_input_projection() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.action_context_in_proj = _TinyWrappedLinear(torch.nn.Linear(6, 4, bias=False))
    encoder.action_context_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_out_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_gate_logit = torch.nn.Parameter(torch.tensor([0.0]))
    encoder.action_context_adapter_rms_cap = True

    suffix = torch.randn((1, 3, 4), dtype=torch.float32)
    prefix_width_context = torch.randn((1, 5, 6), dtype=torch.float32)
    adapted, metrics = _Pi0PaliGemmaSemanticEncoder._apply_action_context_adapter(
        encoder,
        suffix,
        prefix_width_context,
    )

    assert adapted.shape == suffix.shape
    assert metrics["picf_action_context_adapter_token_count"].item() == pytest.approx(5.0)


def test_pi0_action_context_readout_aux_reports_loss_and_metrics() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.action_context_readout_aux_weight = 0.25
    encoder.action_context_readout_aux_loss = "smooth_l1"
    encoder.action_context_readout_aux_huber_delta = 1.0
    encoder.action_context_readout_query = torch.nn.Parameter(torch.empty(3, 4))
    encoder.action_context_readout_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_out_proj = torch.nn.Linear(4, 7)
    _Pi0PaliGemmaSemanticEncoder._reset_action_context_readout_parameters(encoder)

    context = torch.randn((2, 5, 4), dtype=torch.float32)
    target = torch.randn((2, 3, 7), dtype=torch.float32)
    weighted, metrics = _Pi0PaliGemmaSemanticEncoder._compute_action_context_readout_aux(
        encoder,
        context,
        target,
    )

    assert weighted.requires_grad
    assert torch.isfinite(weighted)
    assert metrics["picf_action_context_readout_enabled"].item() == pytest.approx(1.0)
    assert metrics["picf_action_context_readout_weight"].item() == pytest.approx(0.25)
    assert metrics["picf_action_context_readout_token_count"].item() == pytest.approx(5.0)
    assert torch.isfinite(metrics["picf_action_context_readout_loss"])
    assert torch.isfinite(metrics["picf_action_context_readout_mse"])
    assert torch.isfinite(metrics["picf_action_context_readout_attention_entropy_mean"])


def test_pi0_action_context_token_aux_reports_ce_and_accuracy() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.model_action_dim = 7
    encoder.action_context_token_aux_weight = 0.125
    encoder.action_context_token_aux_bins = 16
    encoder.action_context_token_aux_clip = 1.0
    encoder.action_context_readout_query = torch.nn.Parameter(torch.empty(3, 4))
    encoder.action_context_readout_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_out_proj = torch.nn.Linear(4, 7)
    encoder.action_context_token_readout_out_proj = torch.nn.Linear(4, 7 * 16)
    _Pi0PaliGemmaSemanticEncoder._reset_action_context_readout_parameters(encoder)

    context = torch.randn((2, 5, 4), dtype=torch.float32)
    target = torch.randn((2, 3, 7), dtype=torch.float32).clamp(min=-1.0, max=1.0)
    weighted, metrics = _Pi0PaliGemmaSemanticEncoder._compute_action_context_token_aux(
        encoder,
        context,
        target,
    )

    assert weighted.requires_grad
    assert torch.isfinite(weighted)
    assert metrics["picf_action_context_token_aux_enabled"].item() == pytest.approx(1.0)
    assert metrics["picf_action_context_token_aux_weight"].item() == pytest.approx(0.125)
    assert metrics["picf_action_context_token_aux_bins"].item() == pytest.approx(16.0)
    assert metrics["picf_action_context_token_aux_clip"].item() == pytest.approx(1.0)
    assert metrics["picf_action_context_token_aux_token_count"].item() == pytest.approx(5.0)
    assert torch.isfinite(metrics["picf_action_context_token_aux_loss"])
    assert torch.isfinite(metrics["picf_action_context_token_aux_accuracy"])
    assert 0.0 <= metrics["picf_action_context_token_aux_accuracy"].item() <= 1.0
    assert torch.isfinite(metrics["picf_action_context_token_aux_attention_entropy_mean"])


def test_pi0_action_context_flow_residual_changes_deployed_velocity_and_reports_metrics() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.action_context_flow_residual_enabled = False
    encoder.action_context_flow_residual_time_floor = 0.05
    encoder.action_context_flow_residual_rms_cap = True
    encoder.action_context_flow_residual_gate_logit = torch.nn.Parameter(torch.tensor([0.0]))
    encoder.action_context_readout_query = torch.nn.Parameter(torch.empty(3, 4))
    encoder.action_context_readout_q_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_k_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_v_proj = torch.nn.Linear(4, 4, bias=False)
    encoder.action_context_readout_out_proj = torch.nn.Linear(4, 7)
    _Pi0PaliGemmaSemanticEncoder._reset_action_context_readout_parameters(encoder)

    v_t = torch.randn((2, 3, 7), dtype=torch.float32)
    x_t = torch.randn((2, 3, 7), dtype=torch.float32)
    time = torch.full((2, 1, 1), 0.5, dtype=torch.float32)
    context = torch.randn((2, 5, 4), dtype=torch.float32)
    target = torch.randn((2, 3, 7), dtype=torch.float32)

    disabled, disabled_metrics = _Pi0PaliGemmaSemanticEncoder._apply_action_context_flow_residual(
        encoder,
        v_t,
        x_t,
        time,
        context,
        target=target,
    )
    torch.testing.assert_close(disabled, v_t)
    assert disabled_metrics == {}

    encoder.action_context_flow_residual_enabled = True
    adapted, metrics = _Pi0PaliGemmaSemanticEncoder._apply_action_context_flow_residual(
        encoder,
        v_t,
        x_t,
        time,
        context,
        target=target,
    )

    assert adapted.shape == v_t.shape
    assert not torch.allclose(adapted, v_t)
    assert metrics["picf_action_context_flow_residual_enabled"].item() == pytest.approx(1.0)
    assert metrics["picf_action_context_flow_residual_gate"].item() == pytest.approx(0.5)
    assert metrics["picf_action_context_flow_residual_token_count"].item() == pytest.approx(5.0)
    assert torch.isfinite(metrics["picf_action_context_flow_residual_rms_mean"])
    assert torch.isfinite(metrics["picf_action_context_flow_context_velocity_rms_mean"])
    assert torch.isfinite(metrics["picf_action_context_flow_context_target_mse"])


def test_pi0_action_expert_router_starts_as_noop_and_reports_metrics() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.action_expert_router_enabled = True
    encoder.action_expert_router_temperature = 1.0
    encoder.action_expert_router_rms_cap = True
    encoder.action_context_in_proj = torch.nn.Linear(6, 4, bias=False)
    encoder.action_expert_router_summary_proj = torch.nn.Linear(6, 4, bias=False)
    encoder.action_expert_router_summary_pair_proj = torch.nn.Linear(12, 4, bias=False)
    encoder.action_expert_router_norm = torch.nn.LayerNorm(4)
    encoder.action_expert_router_logits = torch.nn.Linear(4, 3)
    encoder.action_expert_router_down = torch.nn.ModuleList(
        [torch.nn.Linear(4, 2, bias=False) for _ in range(3)]
    )
    encoder.action_expert_router_up = torch.nn.ModuleList(
        [torch.nn.Linear(2, 4, bias=False) for _ in range(3)]
    )
    encoder.action_expert_router_gate_logit = torch.nn.Parameter(torch.tensor([-2.5]))
    _Pi0PaliGemmaSemanticEncoder._reset_action_expert_router_parameters(encoder)

    suffix = torch.randn((2, 3, 4), dtype=torch.float32)
    context = torch.randn((2, 5, 6), dtype=torch.float32)
    features = PaliGemmaSemanticFeatures(
        tokens=torch.zeros((2, 1, 6), dtype=torch.float32),
        summary=torch.randn((2, 12), dtype=torch.float32),
    )
    adapted, metrics = _Pi0PaliGemmaSemanticEncoder._apply_action_expert_router(
        encoder,
        suffix,
        features,
        context,
    )

    torch.testing.assert_close(adapted, suffix)
    assert metrics["picf_action_expert_router_enabled"].item() == pytest.approx(1.0)
    assert metrics["picf_action_expert_router_gate"].item() == pytest.approx(float(torch.sigmoid(torch.tensor(-2.5))))
    assert metrics["picf_action_expert_router_entropy_mean"].item() == pytest.approx(np.log(3.0), rel=1e-5)
    assert metrics["picf_action_expert_router_top_weight_mean"].item() == pytest.approx(1.0 / 3.0, rel=1e-5)
    assert metrics["picf_action_expert_router_residual_rms_mean"].item() == pytest.approx(0.0)


def test_pi0_action_expert_router_disabled_is_exact_noop() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.action_expert_router_enabled = False

    suffix = torch.randn((1, 3, 4), dtype=torch.float32)
    features = PaliGemmaSemanticFeatures(
        tokens=torch.zeros((1, 1, 6), dtype=torch.float32),
        summary=torch.randn((1, 6), dtype=torch.float32),
    )
    adapted, metrics = _Pi0PaliGemmaSemanticEncoder._apply_action_expert_router(
        encoder,
        suffix,
        features,
        None,
    )

    assert adapted is suffix
    assert metrics == {}


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("all", True),
        ("backbone_only", True),
        ("model_only", True),
        ("action_head_only", False),
    ],
)
def test_pi0_trainable_scope_backbone_only_keeps_native_checkpointing_boundary(
    scope: str,
    expected: bool,
) -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.trainable = True
    encoder.trainable_scope = scope

    assert _Pi0PaliGemmaSemanticEncoder._trains_semantic_backbone(encoder) is expected


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


def test_build_paligemma_with_expert_omits_chunk_kwargs_for_legacy_constructor(
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

    class _LegacyDummyModel:
        def __init__(
            self,
            paligemma_config,
            action_expert_config,
            *,
            use_adarms,
            precision,
        ) -> None:
            captured["use_adarms"] = int(bool(use_adarms[1]))
            captured["precision_is_bf16"] = int(precision == "bfloat16")

    monkeypatch.setattr(wrapper_mod, "PaliGemmaWithExpertModel", _LegacyDummyModel)

    _Pi0PaliGemmaSemanticEncoder._build_paligemma_with_expert(
        encoder,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        precision="bfloat16",
        pi05=True,
    )

    assert captured["use_adarms"] == 1
    assert captured["precision_is_bf16"] == 1


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


def test_paligemma_view_transform_records_resize_with_pad_metadata() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    transform = _Pi0PaliGemmaSemanticEncoder._view_transform(encoder, image, target_h=224, target_w=224)

    assert transform.original_hw == (100, 200)
    assert transform.target_hw == (224, 224)
    assert transform.resized_hw == (112, 224)
    assert transform.pad_top == 56
    assert transform.pad_bottom == 56
    assert transform.pad_left == 0
    assert transform.pad_right == 0
    assert transform.scale_y == pytest.approx(1.12)
    assert transform.scale_x == pytest.approx(1.12)


def test_state_for_prompt_uses_state_normalizer_when_configured() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.model_action_dim = 4
    encoder.prompt_state_normalizer = PicfStateNormalizer(
        mean=np.zeros((2,), dtype=np.float32),
        std=np.ones((2,), dtype=np.float32),
        q01=np.asarray([-1.0, 0.0], dtype=np.float32),
        q99=np.asarray([1.0, 2.0], dtype=np.float32),
        mode="quantile",
    )
    observation = PicfObservation(
        rgb_static=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_static=np.zeros((2, 2), dtype=np.float32),
        robot_obs=np.asarray([0.0, 2.0], dtype=np.float32),
        prompt="task",
        step_id=0,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
    )

    state = _Pi0PaliGemmaSemanticEncoder._state_for_prompt(encoder, observation)

    np.testing.assert_allclose(state, np.asarray([0.0, 1.0], dtype=np.float32), atol=1e-6)


def test_state_for_prompt_falls_back_to_legacy_clip_without_state_normalizer() -> None:
    encoder = object.__new__(_Pi0PaliGemmaSemanticEncoder)
    encoder.model_action_dim = 4
    encoder.prompt_state_normalizer = None
    observation = PicfObservation(
        rgb_static=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_static=np.zeros((2, 2), dtype=np.float32),
        robot_obs=np.asarray([2.5, -3.0], dtype=np.float32),
        prompt="task",
        step_id=0,
        segment_id=0,
        timestamp_s=0.0,
        reset_scaffold=True,
    )

    state = _Pi0PaliGemmaSemanticEncoder._state_for_prompt(encoder, observation)

    np.testing.assert_allclose(state, np.asarray([1.0, -1.0], dtype=np.float32), atol=1e-6)


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


def test_action_flow_objective_loss_keeps_modes_distinct() -> None:
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    pred = torch.tensor([0.0, 2.0, -4.0], dtype=torch.float32)

    mse = _action_flow_objective_loss(target, pred, mode="mse")
    l1 = _action_flow_objective_loss(target, pred, mode="l1")
    huber = _action_flow_objective_loss(target, pred, mode="huber", huber_delta=1.0)

    assert mse.item() == pytest.approx((0.0 + 4.0 + 16.0) / 3.0)
    assert l1.item() == pytest.approx(2.0)
    assert 0.0 < huber.item() < mse.item()


def test_action_flow_objective_loss_rejects_unknown_mode() -> None:
    target = torch.zeros(2)
    pred = torch.ones(2)

    with pytest.raises(ValueError, match="Unsupported action_flow_loss"):
        _action_flow_objective_loss(target, pred, mode="banana")


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
