from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from picf_next.lingbot_native.wsa_da3_loss import WSADA3TeacherTargets  # noqa: E402
from picf_next.lingbot_native.wsa_future_expert_runtime import (  # noqa: E402
    WSA_FSDP_POST_METHOD,
    WSA_FSDP_QKV_METHOD,
    WSAFutureExpertRuntime,
)
from picf_next.lingbot_native.wsa_lingbot_install import (  # noqa: E402
    WSA_ADAMW_NAME_PATTERN,
    WSALingBotAttentionIntervention,
    WSALingBotForwardContract,
    WSALingBotForwardRole,
    WSALingBotStepLedger,
    _wsa_synchronous_picf_training_forward,
    audit_wsa_lingbot_scheduler,
    build_wsa_lingbot_scheduler,
    configure_wsa_lingbot_optimizer_contract,
    register_wsa_lingbot_fsdp_units,
)


class Future3DBlock(torch.nn.Module):
    pass


class ChangedBlock(torch.nn.Module):
    pass


class Future3DExpert(torch.nn.Module):
    def __init__(self, block_type: type[torch.nn.Module]) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([block_type(), block_type()])


class Policy(torch.nn.Module):
    _no_split_modules = ["Qwen2DecoderLayer"]


def _runtime(block_type: type[torch.nn.Module]) -> SimpleNamespace:
    expert = Future3DExpert(block_type)
    return SimpleNamespace(future=SimpleNamespace(expert=expert))


def test_complete_future_blocks_join_native_fsdp_units_without_replacement() -> None:
    policy = Policy()
    units = register_wsa_lingbot_fsdp_units(policy, _runtime(Future3DBlock))
    assert units == ("Qwen2DecoderLayer", "Future3DBlock", "Future3DExpert")
    assert policy._no_split_modules == list(units)


def test_changed_future_block_class_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="topology changed"):
        register_wsa_lingbot_fsdp_units(Policy(), _runtime(ChangedBlock))


@dataclass(frozen=True)
class OptimizerContract:
    learning_rate: float = 1.0e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1.0e-8
    weight_decay: float = 1.0e-2
    muon_exclude_name_patterns: tuple[str, ...] = ("embed_tokens",)


def test_wsa_uses_the_released_lingbot_adamw_fallback_without_duplicates() -> None:
    configured = configure_wsa_lingbot_optimizer_contract(OptimizerContract())
    assert configured.muon_exclude_name_patterns == (
        "embed_tokens",
        WSA_ADAMW_NAME_PATTERN,
    )
    assert configure_wsa_lingbot_optimizer_contract(configured) == configured


def test_wsa_optimizer_routing_does_not_assume_host_weight_decay_matches() -> None:
    configured = configure_wsa_lingbot_optimizer_contract(
        OptimizerContract(learning_rate=5e-5, weight_decay=0.0)
    )
    assert configured.muon_exclude_name_patterns[-1] == WSA_ADAMW_NAME_PATTERN


def test_wsa_optimizer_backend_is_single_tensor_without_changing_donor_math() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src/picf_next/lingbot_native/wsa_lingbot_install.py"
    ).read_text(encoding="utf-8")
    assert "foreach=False" in source
    assert "fused=False" in source


def test_adr221_launcher_binds_verified_da3_runtime_overlay() -> None:
    launcher = (
        Path(__file__).resolve().parents[2] / "adr221/run_full_source_wsa_2gpu.sh"
    ).read_text(encoding="utf-8")
    assert "DA3_RUNTIME_OVERLAY=/mnt/picf-next/adr221/python-overlays/" in launcher
    assert "DA3_RUNTIME_MANIFEST_SHA256=" in launcher
    assert 'sha256sum -c "$DA3_RUNTIME_MANIFEST"' in launcher
    assert "$DA3_RUNTIME_OVERLAY:${PICF_RUNTIME_PYTHON_OVERLAY:-" in launcher


def test_adr221_dual_gpu_preserves_compile_and_offloads_trainable_vision() -> None:
    launcher = (
        Path(__file__).resolve().parents[2] / "adr221/run_full_source_wsa_2gpu.sh"
    ).read_text()

    assert "export PICF_LINGBOT_COMPILE_MODE=upstream-default" in launcher
    assert "PICF_ARCHITECTURE_PROFILE=adr221_native_videomt_wsa_full_modal_v1" in launcher
    assert "PICF_FSDP2_PLACEMENT:-selective-embedding-trainable-vision-offload" in launcher
    assert "lingbot-vla-v2-adr221-combined-offload-v1" in launcher
    assert "Vision remains" in launcher

    runner = (
        Path(__file__).resolve().parents[2]
        / "tools/run_lingbot_vla2_task_independent_full.py"
    ).read_text()
    assert (
        "verify_selective_trainable_vision_with_selective_class_cpu_offload("
        in runner
    )
    assert (
        "validate_prepared_native_source_with_trainable_vision_and_selective_class_offload("
        in runner
    )


def test_adr221_four_gpu_reuses_exact_model_with_four_gpu_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    base = (root / "adr221/run_full_source_wsa_2gpu.sh").read_text()
    launcher = (root / "adr221/run_full_source_wsa_4gpu.sh").read_text()

    assert "PICF_WORLD_SIZE=${PICF_WORLD_SIZE:-2}" in base
    assert "export PICF_WORLD_SIZE=4" in launcher
    assert "native-query-posterior-4gpu-30k-v1" in launcher
    assert launcher.count("native-query-posterior-4gpu-30k-v1") == 2
    assert 'exec "$SCRIPT_DIR/run_full_source_wsa_2gpu.sh" "$@"' in launcher


def test_adr222_launcher_changes_only_the_registered_information_profile() -> None:
    root = Path(__file__).resolve().parents[2]
    base = (root / "adr221/run_full_source_wsa_2gpu.sh").read_text()
    two_gpu = (root / "adr222/run_world_token_wsa_2gpu.sh").read_text()
    four_gpu = (root / "adr222/run_world_token_wsa_4gpu.sh").read_text()

    assert "export PICF_ARCHITECTURE_PROFILE=adr221_native_videomt_wsa_full_modal_v1" in base
    assert "PICF_WSA_ARCHITECTURE_PROFILE" in base
    assert (
        "PICF_WSA_ARCHITECTURE_PROFILE=adr222_native_videomt_world_token_wsa_v1"
        in two_gpu
    )
    shared_launcher = (root / "adr178/run_direct_action_posterior_full_modal.sh").read_text()
    assert "adr222_native_videomt_world_token_wsa_v1" in shared_launcher
    assert "PICF_WORLD_SIZE=4" in four_gpu
    assert "run_world_token_wsa_2gpu.sh" in four_gpu


def test_wsa_optimizer_is_audited_before_scheduler_mutates_live_lr() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "tools/run_lingbot_vla2_task_independent_full.py"
    ).read_text(encoding="utf-8")
    install = source.index("wsa_donor_optimizer = install_wsa_lingbot_optimizer")
    optimizer_audit = source.index(
        "wsa_optimizer_receipt = audit_wsa_lingbot_optimizer", install
    )
    scheduler = source.index("wsa_scheduler = build_wsa_lingbot_scheduler", install)
    scheduler_audit = source.index(
        "wsa_scheduler_receipt = audit_wsa_lingbot_scheduler", scheduler
    )
    assert install < optimizer_audit < scheduler < scheduler_audit


def _upstream_wsa_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_train_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_steps = int(total_train_steps * 0.05)
    warmup_steps = min(max(warmup_steps, 0), total_train_steps - 1)
    remaining_steps = max(total_train_steps - warmup_steps, 1)
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=remaining_steps,
        eta_min=1.0e-4 * 0.01,
    )
    if warmup_steps <= 0:
        return main_scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )


def test_wsa_scheduler_matches_every_upstream_step_and_cold_resume() -> None:
    total_train_steps = 200
    candidate_parameter = torch.nn.Parameter(torch.ones(()))
    reference_parameter = torch.nn.Parameter(torch.ones(()))
    candidate_optimizer = torch.optim.AdamW([candidate_parameter], lr=1.0e-4)
    reference_optimizer = torch.optim.AdamW([reference_parameter], lr=1.0e-4)
    candidate = build_wsa_lingbot_scheduler(
        candidate_optimizer,
        total_train_steps=total_train_steps,
    )
    reference = _upstream_wsa_scheduler(
        reference_optimizer,
        total_train_steps=total_train_steps,
    )
    candidate_trace = [candidate.get_last_lr()]
    reference_trace = [reference.get_last_lr()]
    saved_scheduler_state = None
    saved_optimizer_state = None
    for step in range(total_train_steps):
        candidate_optimizer.step()
        reference_optimizer.step()
        candidate.step()
        reference.step()
        candidate_trace.append(candidate.get_last_lr())
        reference_trace.append(reference.get_last_lr())
        if step == 72:
            saved_scheduler_state = candidate.state_dict()
            saved_optimizer_state = candidate_optimizer.state_dict()
    assert candidate_trace == reference_trace
    assert candidate_trace[0] == [pytest.approx(1.0e-5)]
    assert candidate_trace[-1] == [pytest.approx(1.0e-6)]
    assert saved_scheduler_state is not None and saved_optimizer_state is not None

    resumed_parameter = torch.nn.Parameter(torch.ones(()))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1.0e-4)
    resumed = build_wsa_lingbot_scheduler(
        resumed_optimizer,
        total_train_steps=total_train_steps,
    )
    resumed_optimizer.load_state_dict(saved_optimizer_state)
    resumed.load_state_dict(saved_scheduler_state)
    assert resumed.state_dict() == saved_scheduler_state
    resumed_trace = [resumed.get_last_lr()]
    for _ in range(73, total_train_steps):
        resumed_optimizer.step()
        resumed.step()
        resumed_trace.append(resumed.get_last_lr())
    assert resumed_trace == candidate_trace[73:]
    receipt = audit_wsa_lingbot_scheduler(
        resumed,
        resumed_optimizer,
        total_train_steps=total_train_steps,
    )
    assert receipt["warmup_steps"] == 10
    assert receipt["minimum_learning_rate"] == pytest.approx(1.0e-6)


def test_wsa_staged_fsdp_method_names_are_namespaced_and_distinct() -> None:
    assert WSA_FSDP_QKV_METHOD == "adr218_wsa_build_attention_io"
    assert WSA_FSDP_POST_METHOD == "adr218_wsa_apply_post"
    assert WSA_FSDP_QKV_METHOD != WSA_FSDP_POST_METHOD


def _teacher_targets() -> WSADA3TeacherTargets:
    return WSADA3TeacherTargets(
        layers=tuple(
            torch.empty(1, 2592, 2048, device="meta") for _ in range(4)
        ),
        view_valid=torch.ones(1, 2, dtype=torch.bool),
    )


def test_wsa_forward_roles_separate_factual_supervision_from_measurement() -> None:
    factual = WSALingBotForwardContract(
        role=WSALingBotForwardRole.PRIMARY_FACTUAL,
        teacher_targets=_teacher_targets(),
    )
    assert factual.teacher_targets is not None
    measurement = WSALingBotForwardContract(
        role=WSALingBotForwardRole.MEASUREMENT_ONLY,
    )
    assert measurement.teacher_targets is None
    with pytest.raises(TypeError, match="requires typed DA3 targets"):
        WSALingBotForwardContract(role=WSALingBotForwardRole.PRIMARY_FACTUAL)
    with pytest.raises(ValueError, match="cannot carry teacher targets"):
        WSALingBotForwardContract(
            role=WSALingBotForwardRole.MEASUREMENT_ONLY,
            teacher_targets=_teacher_targets(),
        )


def test_wsa_measurement_callback_is_typed_and_excluded_from_factual_loss() -> None:
    def callback(_output: object) -> None:
        return None

    measurement = WSALingBotForwardContract(
        role=WSALingBotForwardRole.MEASUREMENT_ONLY,
        measurement_callback=callback,
    )
    assert measurement.measurement_callback is callback
    with pytest.raises(ValueError, match="cannot retain measurement"):
        WSALingBotForwardContract(
            role=WSALingBotForwardRole.PRIMARY_FACTUAL,
            teacher_targets=_teacher_targets(),
            measurement_callback=callback,
        )
    with pytest.raises(TypeError, match="must be callable"):
        WSALingBotForwardContract(
            role=WSALingBotForwardRole.MEASUREMENT_ONLY,
            measurement_callback=object(),  # type: ignore[arg-type]
        )


def test_wsa_attention_intervention_is_typed_and_measurement_only() -> None:
    intervention = WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION
    measurement = WSALingBotForwardContract(
        role=WSALingBotForwardRole.MEASUREMENT_ONLY,
        attention_intervention=intervention,
    )
    assert measurement.attention_intervention is intervention
    with pytest.raises(ValueError, match="cannot carry an intervention"):
        WSALingBotForwardContract(
            role=WSALingBotForwardRole.PRIMARY_FACTUAL,
            teacher_targets=_teacher_targets(),
            attention_intervention=intervention,
        )
    with pytest.raises(TypeError, match="must be typed"):
        WSALingBotForwardContract(
            role=WSALingBotForwardRole.MEASUREMENT_ONLY,
            attention_intervention="block_future_to_action",  # type: ignore[arg-type]
        )


def test_synchronous_wsa_forward_preserves_action_attention_callback() -> None:
    source = inspect.getsource(_wsa_synchronous_picf_training_forward)
    assert "picf_action_attention_callback=picf_action_attention_callback" in source


def test_wsa_step_ledger_requires_exactly_one_factual_supervision() -> None:
    ledger = WSALingBotStepLedger()
    ledger.record(WSALingBotForwardRole.MEASUREMENT_ONLY)
    ledger.record(WSALingBotForwardRole.PRIMARY_FACTUAL)
    ledger.close()
    assert ledger.receipt() == {
        "schema": "picf-next.adr218-wsa-step-ledger.v1",
        "primary_factual_calls": 1,
        "measurement_only_calls": 1,
        "closed": True,
    }
    missing = WSALingBotStepLedger()
    with pytest.raises(RuntimeError, match="exactly one"):
        missing.close()
    duplicate = WSALingBotStepLedger()
    duplicate.record(WSALingBotForwardRole.PRIMARY_FACTUAL)
    duplicate.record(WSALingBotForwardRole.PRIMARY_FACTUAL)
    with pytest.raises(RuntimeError, match="got 2"):
        duplicate.close()


class _PrecisionBoundaryExpert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedding = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        self.query_layer_indices = (0,)
        self.received: dict[str, object] | None = None

    def pre_dit(self, **kwargs: object) -> dict[str, object]:
        self.received = dict(kwargs)
        return {
            "tokens": torch.zeros(1, 1, 4, dtype=kwargs["dtype"]),
            "freqs": torch.zeros(1, 1, 2, dtype=kwargs["dtype"]),
            "t_mod": torch.zeros(1, 6, 4, dtype=kwargs["dtype"]),
        }


def test_wsa_prepare_uses_expert_compute_dtype_across_mixed_precision_boundary() -> None:
    runtime = WSAFutureExpertRuntime.__new__(WSAFutureExpertRuntime)
    torch.nn.Module.__init__(runtime)
    runtime.expert = _PrecisionBoundaryExpert().float()
    prepared = runtime.prepare(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        timestep=torch.zeros(1, dtype=torch.bfloat16),
        query_noise_sigma=torch.ones(1, dtype=torch.bfloat16),
    )
    assert runtime.expert.received is not None
    assert runtime.expert.received["dtype"] == torch.float32
    assert runtime.expert.received["timestep"].dtype == torch.float32
    assert runtime.expert.received["query_noise_sigma"].dtype == torch.float32
    assert prepared.tokens.dtype == torch.float32
