from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.lingbot_vla2_runtime_helpers import (
    LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
    build_lingbot_base_family_identity,
    build_lingbot_fixed_batch_probe_optimizer,
    build_lingbot_official_optimizer,
    build_lingbot_query_only_optimizer,
    build_lingbot_representation_optimizer,
    clip_lingbot_distributed_l2_grad_norm_,
    configure_picf_optimizer_learning_rates,
    load_lingbot_training_config,
    require_lingbot_exact_resume_contract,
    require_lingbot_released_action_sampling_steps,
    resolve_lingbot_optimizer_contract,
    select_lingbot_deterministic_moe_backend,
    strip_targetless_alignment_teacher_heads,
)


class _LocalDist:
    class ReduceOp:
        SUM = "sum"

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return False


def test_lingbot_base_family_identity_binds_shared_initialization() -> None:
    keyword_arguments = {
        "source_commit": "1" * 40,
        "native_patch_sha256": "2" * 64,
        "checkpoint_revision": "3" * 40,
        "checkpoint_report": {
            "checkpoint_assets": [{"path": "model.safetensors", "sha256": "4" * 64}]
        },
        "processor_revision": "5" * 40,
        "processor_report": {
            "processor_assets": [{"path": "tokenizer.json", "sha256": "6" * 64}]
        },
        "attention_implementation": "flex_cached",
        "trainable_scope": "full-host",
        "optimizer_contract": {"learning_rate": 1e-4},
        "maximum_control_tokens": 64,
    }

    first = build_lingbot_base_family_identity(**keyword_arguments)
    second = build_lingbot_base_family_identity(**keyword_arguments)

    assert first == second
    assert first["schema"] == "picf-next.lingbot-base-family.v1"
    assert first["architecture"] == "released_lingbot_vla2_action_policy"
    assert len(first["artifact_sha256"]) == 64


def test_lingbot_base_family_identity_rejects_absent_assets() -> None:
    with pytest.raises(ValueError, match="checkpoint assets are absent"):
        build_lingbot_base_family_identity(
            source_commit="1" * 40,
            native_patch_sha256="2" * 64,
            checkpoint_revision="3" * 40,
            checkpoint_report={"checkpoint_assets": []},
            processor_revision="5" * 40,
            processor_report={
                "processor_assets": [{"path": "tokenizer.json", "sha256": "6" * 64}]
            },
            attention_implementation="flex_cached",
            trainable_scope="full-host",
            optimizer_contract={"learning_rate": 1e-4},
            maximum_control_tokens=64,
        )


def test_picf_learning_rate_stratification_preserves_parameter_coverage() -> None:
    host = nn.Linear(3, 4)
    graph = nn.Module()
    graph.picf_projection = nn.Linear(4, 5)
    graph.modality_bridge = nn.Linear(5, 6)
    optimizer = torch.optim.AdamW(
        (*host.parameters(), *graph.parameters()),
        lr=1e-4,
    )

    receipt = configure_picf_optimizer_learning_rates(
        optimizer,
        graph,
        picf_multiplier=2.0,
        modality_bridge_multiplier=0.5,
    )

    groups = {group["picf_learning_rate_role"]: group for group in optimizer.param_groups}
    assert set(groups) == {
        "lingbot_host",
        "picf_graph",
        "pretrained_modality_bridge",
    }
    assert groups["lingbot_host"]["lr"] == pytest.approx(1e-4)
    assert groups["picf_graph"]["lr"] == pytest.approx(2e-4)
    assert groups["pretrained_modality_bridge"]["lr"] == pytest.approx(5e-5)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == 6
    assert receipt["parameter_count"] == {
        "lingbot_host": 2,
        "picf_graph": 2,
        "pretrained_modality_bridge": 2,
    }


def test_picf_learning_rate_stratification_excludes_frozen_source_modules() -> None:
    host = nn.Linear(3, 4)
    graph = nn.Module()
    graph.picf_projection = nn.Linear(4, 5)
    graph.source_mask_head = nn.Linear(5, 6)
    graph.source_mask_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        tuple(
            parameter
            for parameter in (*host.parameters(), *graph.parameters())
            if parameter.requires_grad
        ),
        lr=1e-4,
    )

    receipt = configure_picf_optimizer_learning_rates(
        optimizer,
        graph,
        picf_multiplier=2.0,
        modality_bridge_multiplier=0.5,
    )

    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert not optimizer_ids.intersection(
        id(parameter) for parameter in graph.source_mask_head.parameters()
    )
    assert receipt["parameter_count"]["picf_graph"] == 2


def test_picf_learning_rate_stratification_rejects_frozen_optimizer_membership() -> None:
    graph = nn.Module()
    graph.picf_projection = nn.Linear(4, 5)
    graph.source_mask_head = nn.Linear(5, 6)
    graph.source_mask_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(tuple(graph.parameters()), lr=1e-4)

    with pytest.raises(RuntimeError, match="contains frozen PICF parameter"):
        configure_picf_optimizer_learning_rates(
            optimizer,
            graph,
            picf_multiplier=2.0,
            modality_bridge_multiplier=0.5,
        )


def test_distributed_l2_clip_matches_single_rank_global_norm() -> None:
    first = nn.Parameter(torch.tensor([3.0, 4.0]))
    second = nn.Parameter(torch.tensor([0.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = None

    norm = clip_lingbot_distributed_l2_grad_norm_(
        (first, second),
        2.5,
        device=torch.device("cpu"),
        dist_module=_LocalDist,
        torch_module=torch,
    )

    assert norm == pytest.approx(5.0)
    assert first.grad.tolist() == pytest.approx([1.5, 2.0])
    assert second.grad is None


def test_distributed_l2_clip_rejects_nonfinite_gradients() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])

    with pytest.raises(FloatingPointError, match="non-finite"):
        clip_lingbot_distributed_l2_grad_norm_(
            (parameter,),
            1.0,
            device=torch.device("cpu"),
            dist_module=_LocalDist,
            torch_module=torch,
        )


def test_released_action_sampling_contract_is_explicit_and_fail_closed() -> None:
    assert LINGBOT_RELEASED_ACTION_SAMPLING_STEPS == 10
    require_lingbot_released_action_sampling_steps(SimpleNamespace(num_steps=10))
    with pytest.raises(ValueError, match="released 10-step"):
        require_lingbot_released_action_sampling_steps(SimpleNamespace(num_steps=30_000))
    with pytest.raises(ValueError, match="released 10-step"):
        require_lingbot_released_action_sampling_steps(SimpleNamespace(num_steps=True))


def test_native_fsdp_registration_includes_action_and_native_roots() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "tools/lingbot_vla2_runtime_helpers.py").read_text(encoding="utf-8")
    start = source.index("def register_native_fsdp_forward_methods(")
    stop = source.index("\n\ndef _sha256", start)
    body = source[start:stop]

    assert '"sample_actions"' in body
    assert '"picf_native_prior_forward"' in body
    assert '"picf_native_observation_forward"' in body
    assert '"picf_native_frozen_posterior_action_forward"' in body


class _TargetlessFlow(nn.Module):
    use_depth_align = True
    use_future_depth = True
    use_current_video_patch = True
    use_future_video_patch = True
    use_future_video_cls = False
    use_current_shared_task_proj = True
    use_shared_future_task_proj = True

    def __init__(self) -> None:
        super().__init__()
        self.depth_align_embs = nn.Parameter(torch.randn(2, 4))
        self.future_depth_align_embs = nn.Parameter(torch.randn(2, 4))
        self.current_video_align_embs = nn.Parameter(torch.randn(2, 4))
        self.future_video_align_embs = nn.Parameter(torch.randn(2, 4))
        self.current_shared_task_proj = nn.Linear(8, 4)
        self.future_shared_task_proj = nn.Linear(8, 4)
        self.depth_align_head = nn.Linear(4, 3)
        self.future_depth_align_head = nn.Linear(4, 3)
        self.current_video_align_head = nn.Linear(4, 3)
        self.future_video_align_head = nn.Linear(4, 3)

    def forward(self, value: torch.Tensor, *, compute_alignment_losses: bool = True):
        if compute_alignment_losses:
            value = self.depth_align_head(value)
        return value


class _Policy(nn.Module):
    def __init__(self, flow: _TargetlessFlow | None = None) -> None:
        super().__init__()
        self.model = _TargetlessFlow() if flow is None else flow


def _optimizer_training() -> dict[str, object]:
    return {
        "train": {
            "optimizer": "muon",
            "lr": 1e-4,
            "weight_decay": 0.0,
            "lr_decay_style": "constant",
            "lr_warmup_ratio": 0.0,
            "lr_start": 0.0,
            "use_moe": True,
            "use_moe_expert_lr": True,
            "token_moe_layers": list(range(36)),
            "token_num_experts": 32,
            "token_top_k": 4,
            "bias_update_speed": 0.0,
            "sequence_wise_loss_coeff": 1e-3,
            "sequence_wise_mode": "per_sequence",
            "router_z_loss_coeff": 1e-4,
            "router_activation": "sigmoid",
            "routed_scaling_factor": 4.0,
            "use_shared_expert_gate": False,
            "enable_fp32": True,
        }
    }


def test_released_yaml_normalizes_official_typed_numeric_scalars(tmp_path) -> None:
    config = tmp_path / "training.yaml"
    config.write_text(
        """
model: {}
train:
  sequence_wise_loss_coeff: 1e-3
  router_z_loss_coeff: 1e-4
data: {}
""".lstrip()
    )

    loaded = load_lingbot_training_config(config)

    assert loaded["train"]["sequence_wise_loss_coeff"] == 0.001
    assert isinstance(loaded["train"]["sequence_wise_loss_coeff"], float)
    assert loaded["train"]["router_z_loss_coeff"] == 0.0001
    assert isinstance(loaded["train"]["router_z_loss_coeff"], float)


@pytest.mark.parametrize(
    "relative",
    (
        "tools/audit_lingbot_checkpoint_contract.py",
        "tools/build_lingbot_calvin_predictive_cache.py",
        "tools/preflight_lingbot_native.py",
        "tools/run_lingbot_vla2_native_full.py",
        "tools/run_lingbot_vla2_native_g0.py",
        "tools/run_lingbot_vla2_unified_g1.py",
        "tools/smoke_lingbot_vla2_full_weight.py",
        "tools/smoke_lingbot_vla2_native_full_weight.py",
        "tools/smoke_lingbot_vla2_unified_full_weight.py",
    ),
)
def test_every_direct_released_config_consumer_uses_official_scalar_semantics(
    relative: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / relative).read_text(encoding="utf-8")

    assert "load_lingbot_training_config(" in source
    assert "yaml.safe_load" not in source


def test_deterministic_moe_selector_uses_released_fallback() -> None:
    def robby() -> None:
        return None

    action_expert = SimpleNamespace(robby_moe_forward=robby)
    fused_moe = SimpleNamespace(fused_moe_forward=lambda: None)
    report = select_lingbot_deterministic_moe_backend(
        action_expert_module=action_expert,
        fused_moe_module=fused_moe,
    )

    assert report == {
        "schema": "picf-next.lingbot-moe-inference-backend.v1",
        "selected": "fused_moe_forward",
        "fused_fallback_available": True,
        "robby_available_before_selection": True,
        "robby_disabled": True,
    }
    assert action_expert.robby_moe_forward is None


def test_deterministic_moe_selector_rejects_missing_fallback() -> None:
    action_expert = SimpleNamespace(robby_moe_forward=lambda: None)
    fused_moe = SimpleNamespace(fused_moe_forward=None)
    with pytest.raises(RuntimeError, match="fallback is unavailable"):
        select_lingbot_deterministic_moe_backend(
            action_expert_module=action_expert,
            fused_moe_module=fused_moe,
        )


def test_released_optimizer_contract_is_explicit_and_invokes_official_builders() -> None:
    contract = resolve_lingbot_optimizer_contract(
        _optimizer_training(),
        requested_learning_rate=1e-4,
    )
    assert contract.adamw_betas == (0.9, 0.95)
    assert contract.weight_decay == 0.0
    assert contract.scheduler == "constant"
    assert contract.sequence_wise_loss_coeff == 0.001
    assert contract.router_z_loss_coeff == 0.0001
    assert contract.router_activation == "sigmoid"
    assert contract.routed_scaling_factor == 4.0
    assert contract.use_shared_expert_gate is False
    assert contract.enable_fp32 is True
    assert contract.enable_mixed_precision is True
    assert contract.metadata["builder"] == "lingbotvla.optim.build_muon_optimizer"

    calls: dict[str, object] = {}

    class Optimizer:
        def register_step_pre_hook(self, hook: object) -> None:
            calls["registered_hook"] = hook

    def build_muon(model: object, arguments: object, **kwargs: object) -> Optimizer:
        calls["model"] = model
        calls["arguments"] = arguments
        calls["optimizer_kwargs"] = kwargs
        return Optimizer()

    def build_hook(model: object, **kwargs: object) -> object:
        calls["hook_model"] = model
        calls["hook_kwargs"] = kwargs
        return "hook"

    model = nn.Linear(2, 2)
    optimizer = build_lingbot_official_optimizer(
        model,
        contract,
        build_muon_optimizer=build_muon,
        build_moe_load_balance_hook=build_hook,
    )
    assert isinstance(optimizer, Optimizer)
    assert calls["model"] is model and calls["hook_model"] is model
    assert calls["optimizer_kwargs"] == {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "adamw_betas": (0.9, 0.95),
        "adamw_eps": 1e-8,
    }
    assert calls["hook_kwargs"] == {
        "coeff": 0.0,
        "bias_centering": False,
        "update_interval": 1,
    }
    assert calls["registered_hook"] == "hook"


def test_fixed_batch_probe_optimizer_omits_only_the_stateful_moe_hook() -> None:
    contract = resolve_lingbot_optimizer_contract(
        _optimizer_training(),
        requested_learning_rate=1e-4,
    )
    calls: dict[str, object] = {}

    def build_muon(model: object, arguments: object, **kwargs: object) -> object:
        calls["model"] = model
        calls["arguments"] = arguments
        calls["optimizer_kwargs"] = kwargs
        return "optimizer"

    model = nn.Linear(2, 2)
    optimizer = build_lingbot_fixed_batch_probe_optimizer(
        model,
        contract,
        build_muon_optimizer=build_muon,
    )

    assert optimizer == "optimizer"
    assert calls["model"] is model
    assert calls["optimizer_kwargs"] == {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "adamw_betas": (0.9, 0.95),
        "adamw_eps": 1e-8,
    }


def test_representation_optimizer_preserves_update_rule_without_action_hook() -> None:
    contract = resolve_lingbot_optimizer_contract(
        _optimizer_training(),
        requested_learning_rate=1e-4,
    )
    calls: dict[str, object] = {}

    def build_muon(model: object, arguments: object, **kwargs: object) -> object:
        calls["model"] = model
        calls["arguments"] = arguments
        calls["optimizer_kwargs"] = kwargs
        return "representation-optimizer"

    model = nn.Linear(2, 2)
    optimizer = build_lingbot_representation_optimizer(
        model,
        contract,
        build_muon_optimizer=build_muon,
    )

    assert optimizer == "representation-optimizer"
    assert calls["model"] is model
    assert calls["optimizer_kwargs"] == {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "adamw_betas": (0.9, 0.95),
        "adamw_eps": 1e-8,
    }


def test_query_only_optimizer_uses_released_adamw_fallback() -> None:
    contract = resolve_lingbot_optimizer_contract(
        _optimizer_training(),
        requested_learning_rate=1e-4,
    )
    calls: dict[str, object] = {}

    def build_optimizer(model: object, **kwargs: object) -> object:
        calls["model"] = model
        calls["optimizer_kwargs"] = kwargs
        return "query-only-optimizer"

    model = nn.Embedding(4, 8)
    optimizer = build_lingbot_query_only_optimizer(
        model,
        contract,
        build_optimizer=build_optimizer,
    )

    assert optimizer == "query-only-optimizer"
    assert calls["model"] is model
    assert calls["optimizer_kwargs"] == {
        "lr": 1e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "fused": False,
        "optimizer_type": "adamw",
        "post_training": True,
    }


def test_released_optimizer_contract_rejects_silent_recipe_drift() -> None:
    with pytest.raises(ValueError, match="learning rate differs"):
        resolve_lingbot_optimizer_contract(
            _optimizer_training(),
            requested_learning_rate=1e-5,
        )
    changed = _optimizer_training()
    changed["train"]["lr_decay_style"] = "cosine"  # type: ignore[index]
    with pytest.raises(ValueError, match="identity constant schedule"):
        resolve_lingbot_optimizer_contract(changed, requested_learning_rate=1e-4)

    malformed = _optimizer_training()
    malformed["train"]["router_z_loss_coeff"] = "1e-4"  # type: ignore[index]
    with pytest.raises(TypeError, match="router_z_loss_coeff"):
        resolve_lingbot_optimizer_contract(malformed, requested_learning_rate=1e-4)

    invalid_router = _optimizer_training()
    invalid_router["train"]["router_activation"] = "unknown"  # type: ignore[index]
    with pytest.raises(ValueError, match="router_activation"):
        resolve_lingbot_optimizer_contract(invalid_router, requested_learning_rate=1e-4)


def test_exact_resume_contract_rejects_unserialized_load_balance_windows() -> None:
    contract = resolve_lingbot_optimizer_contract(
        _optimizer_training(),
        requested_learning_rate=1e-4,
    )
    require_lingbot_exact_resume_contract(contract)

    with pytest.raises(ValueError, match="does not serialize its phase"):
        require_lingbot_exact_resume_contract(
            replace(contract, bias_update_interval=2),
        )
    with pytest.raises(TypeError, match="frozen optimizer contract"):
        require_lingbot_exact_resume_contract(SimpleNamespace())  # type: ignore[arg-type]


def test_targetless_alignment_prune_removes_only_terminal_teacher_heads() -> None:
    policy = _Policy()
    retained = {
        name: getattr(policy.model, name)
        for name in (
            "current_shared_task_proj",
            "current_video_align_embs",
            "depth_align_embs",
            "future_depth_align_embs",
            "future_shared_task_proj",
            "future_video_align_embs",
        )
    }
    expected_removed_numel = sum(
        parameter.numel()
        for name, parameter in policy.model.named_parameters()
        if "align_head" in name
    )

    report = strip_targetless_alignment_teacher_heads(policy)

    assert report["removed_numel"] == expected_removed_numel
    assert report["removed_storage_bytes"] == expected_removed_numel * 4
    assert [value["name"] for value in report["removed"]] == [
        "current_video_align_head",
        "depth_align_head",
        "future_depth_align_head",
        "future_video_align_head",
    ]
    for name, value in retained.items():
        assert getattr(policy.model, name) is value
    assert all(not hasattr(policy.model, value["name"]) for value in report["removed"])
    assert policy.model._picf_targetless_alignment_teacher_prune is report


def test_targetless_alignment_prune_rejects_missing_query_producer() -> None:
    policy = _Policy()
    del policy.model.future_video_align_embs
    with pytest.raises(RuntimeError, match="lose query producers"):
        strip_targetless_alignment_teacher_heads(policy)


def test_targetless_alignment_prune_rejects_shared_query_parameters() -> None:
    flow = _TargetlessFlow()
    flow.depth_align_embs = flow.depth_align_head.weight
    with pytest.raises(RuntimeError, match="shares parameters with retained queries"):
        strip_targetless_alignment_teacher_heads(_Policy(flow))


def test_targetless_alignment_prune_rejects_a_second_application() -> None:
    policy = _Policy()
    strip_targetless_alignment_teacher_heads(policy)
    with pytest.raises(RuntimeError, match="topology differs"):
        strip_targetless_alignment_teacher_heads(policy)
