# ruff: noqa: E402, I001
"""Reusable exact G2b model-only stage-transfer runtime for later LTOP gates."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _repository_import_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _repository_import_text = str(_repository_import_path)
    while _repository_import_text in sys.path:
        sys.path.remove(_repository_import_text)
    sys.path.insert(0, _repository_import_text)

import picf_next as _picf_next_package

if (
    _picf_next_package.__file__ is None
    or Path(_picf_next_package.__file__).resolve().parent
    != (_REPOSITORY_ROOT / "src/picf_next").resolve()
):
    raise RuntimeError("LTOP stage runtime did not import picf_next from its own checkout")

from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from tools.bootstrap_lingbot_vla2 import (
    LINGBOT_CHECKPOINT_ID,
    LINGBOT_CHECKPOINT_REVISION,
    QWEN_PROCESSOR_REVISION,
    validate_checkpoint,
    validate_processor,
)
from tools.bootstrap_lingbot_vla2_native import (
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MUON_SOURCE,
    validate_prepared_native_source,
    validate_prepared_native_source_with_muon_collective_hotfix,
    verify_native_patch,
    verify_muon_collective_hotfix,
)
from tools.lingbot_vla2_runtime_helpers import (
    LingBotOptimizerContract,
    _merge_qwen_config,
    _resolve_training_config,
    load_lingbot_training_config,
    register_native_fsdp_forward_methods,
    resolve_lingbot_optimizer_contract,
    strip_targetless_alignment_teacher_heads,
)
from tools.run_lingbot_vla2_ltop_g2_core import (
    G2_ARCHITECTURE,
    G2_CAPACITY,
    G2_REPRESENTATION_SCHEMA,
    G2_TASK_QUERY_COUNT,
    G2_WORLD_SIZE,
)
from tools.run_lingbot_vla2_native_g0 import (
    _distributed_rank_local_call,
    _model_local_state_digest,
    _validate_fsdp2_parameter_storage,
)


INTERNAL_G2_REPORT = "ltop_g2_representation_report.json"
MODEL_ONLY_CHECKPOINT_ENTRIES = frozenset({"model", INTERNAL_G2_REPORT})
FORBIDDEN_CHECKPOINT_PAYLOADS = frozenset({"ema", "optimizer", "extra_state"})


@dataclass(frozen=True, slots=True)
class LingBotVLA2LTOPStageRequest:
    """Immutable inputs needed to reconstruct the accepted G2b stage exactly."""

    source_checkout: Path
    patch: Path
    training_config: Path
    checkpoint_dir: Path
    processor_dir: Path
    stage_checkpoint: Path
    g2_report: Path
    runtime_hotfix: Path | None = None
    seed: int = 20260812
    maximum_control_tokens: int = 8
    fsdp2_placement: str = FSDP2_SELECTIVE_EMBEDDING_OFFLOAD


@dataclass(frozen=True, slots=True)
class LingBotVLA2LTOPStageContract:
    """Validated immutable evidence binding one exact G2b stage checkpoint."""

    request: LingBotVLA2LTOPStageRequest
    prepared_source: dict[str, Any]
    patch_report: dict[str, Any]
    runtime_hotfix_report: dict[str, Any] | None
    checkpoint_inventory: dict[str, Any]
    model_identity: dict[str, Any]
    g2_report: dict[str, Any]
    expected_rank_digests: dict[int, str]
    g2_report_sha256: str


@dataclass(frozen=True, slots=True)
class LingBotVLA2LTOPRuntimeModules:
    """Dynamically imported runtime modules retained for the caller's live process."""

    torch: Any
    dist: Any


@dataclass(slots=True)
class LingBotVLA2LTOPStageRuntime:
    """Live restored runtime yielded before the distributed process group is destroyed."""

    contract: LingBotVLA2LTOPStageContract
    policy: Any
    graph: Any
    graph_config: Any
    model_config: Any
    training_config: dict[str, Any]
    resolved_training_config: dict[str, Any]
    representation_scope: Any
    optimizer_contract: LingBotOptimizerContract
    runtime_modules: LingBotVLA2LTOPRuntimeModules
    rank: int
    local_rank: int
    device: Any
    fsdp2_storage_before_load: dict[str, Any]
    fsdp2_storage_after_load: dict[str, Any]
    expected_model_local_state_sha256: str
    actual_model_local_state_sha256: str
    model_build_s: float
    dcp_load_s: float

    def rank_report(self) -> dict[str, Any]:
        torch = self.runtime_modules.torch
        return {
            "rank": self.rank,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "expected_model_local_state_sha256": self.expected_model_local_state_sha256,
            "actual_model_local_state_sha256": self.actual_model_local_state_sha256,
            "digest_match": (
                self.actual_model_local_state_sha256 == self.expected_model_local_state_sha256
            ),
            "meta_state_names_before_load": [],
            "meta_state_names_after_load": [],
            "fsdp2_storage_before_load": self.fsdp2_storage_before_load,
            "fsdp2_storage_after_load": self.fsdp2_storage_after_load,
            "timings": {
                "model_build_s": self.model_build_s,
                "dcp_load_s": self.dcp_load_s,
            },
            "cuda_memory_bytes": {
                "allocated": int(torch.cuda.memory_allocated(self.device)),
                "reserved": int(torch.cuda.memory_reserved(self.device)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(self.device)),
            },
        }


def ltop_stage_runtime_source_contract(
    contract: LingBotVLA2LTOPStageContract,
) -> dict[str, Any]:
    """Return the immutable model-overlay and optimizer-runtime source identity."""

    return {
        "native_patch_sha256": contract.patch_report["patch_sha256"],
        "runtime_hotfix_sha256": (
            None
            if contract.runtime_hotfix_report is None
            else contract.runtime_hotfix_report["runtime_hotfix_sha256"]
        ),
        "runtime_patched_source_sha256": contract.prepared_source["patched_source_sha256"],
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            value[key] = child
        return value

    try:
        decoded = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], decoded)


def _lower_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _require_real_directory(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be one real directory")


def _require_real_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be one real file")


def _validate_model_only_checkpoint_tree(checkpoint: Path) -> dict[str, Any]:
    _require_real_directory(checkpoint, name="G2b stage checkpoint")
    entries = {path.name for path in checkpoint.iterdir()}
    forbidden = sorted(entries & FORBIDDEN_CHECKPOINT_PAYLOADS)
    if forbidden:
        raise ValueError(f"G2b stage checkpoint contains non-model payloads: {forbidden}")
    if entries != MODEL_ONLY_CHECKPOINT_ENTRIES:
        raise ValueError("G2b stage checkpoint root differs from the model-only contract")

    model_dir = checkpoint / "model"
    _require_real_directory(model_dir, name="G2b DCP model directory")
    regular_files: list[dict[str, Any]] = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"G2b stage checkpoint contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"G2b stage checkpoint contains a non-regular path: {path}")
        regular_files.append(
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "bytes": path.stat().st_size,
            }
        )
    model_file_names = {path.name for path in model_dir.iterdir() if path.is_file()}
    if ".metadata" not in model_file_names:
        raise ValueError("G2b DCP model directory omits .metadata")
    if not any(name.endswith(".distcp") for name in model_file_names):
        raise ValueError("G2b DCP model directory contains no rank shard")
    return {
        "root_entries": sorted(entries),
        "regular_file_count": len(regular_files),
        "regular_files": regular_files,
    }


def _validate_g2_report(
    *,
    stage_checkpoint: Path,
    external_report: Path,
    expected_model_identity: dict[str, Any],
    expected_patch_sha256: str,
) -> tuple[dict[str, Any], dict[int, str], str]:
    internal_report = stage_checkpoint / INTERNAL_G2_REPORT
    _require_real_file(external_report, name="external G2b report")
    _require_real_file(internal_report, name="checkpoint-internal G2b report")
    external_bytes = external_report.read_bytes()
    internal_bytes = internal_report.read_bytes()
    if external_bytes != internal_bytes:
        raise ValueError("external and checkpoint-internal G2b reports differ")
    report = _load_json_object(internal_bytes, name="G2b report")

    expected_scalars = {
        "schema": G2_REPRESENTATION_SCHEMA,
        "status": "PASS",
        "architecture_identity": G2_ARCHITECTURE,
        "training_scope": "representation",
        "world_size": G2_WORLD_SIZE,
        "capacity": G2_CAPACITY,
        "task_query_count": G2_TASK_QUERY_COUNT,
    }
    for field, expected in expected_scalars.items():
        if report.get(field) != expected:
            raise ValueError(f"G2b report {field} differs from the stage-transfer contract")
    if report.get("failures") != []:
        raise ValueError("G2b report contains scientific failures")
    if report.get("model_identity") != expected_model_identity:
        raise ValueError("G2b report model identity differs from the reconstructed model")
    if report.get("patch_sha256") != expected_patch_sha256:
        raise ValueError("G2b report patch digest differs from the reconstructed model")

    checkpoint = report.get("checkpoint")
    expected_checkpoint = {
        "requested": True,
        "path": str(stage_checkpoint.absolute()),
        "format": "lingbot-fsdp2-dcp-model-only",
        "optimizer_saved": False,
        "extra_state_saved": False,
        "stage_transfer_not_exact_resume": True,
        "publication_status": "PASS",
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(field) != expected for field, expected in expected_checkpoint.items()
    ):
        raise ValueError("G2b report checkpoint contract is incomplete or targets another path")

    trainable_scope = report.get("trainable_scope")
    if not isinstance(trainable_scope, dict) or not isinstance(
        trainable_scope.get("representation_scope"), dict
    ):
        raise ValueError("G2b report omits the representation parameter scope")
    rank_reports = report.get("rank_reports")
    if not isinstance(rank_reports, list) or len(rank_reports) != G2_WORLD_SIZE:
        raise ValueError("G2b report must contain exactly two rank reports")
    expected_digests: dict[int, str] = {}
    for rank_report in rank_reports:
        if not isinstance(rank_report, dict):
            raise ValueError("G2b rank report is not an object")
        rank = rank_report.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < G2_WORLD_SIZE:
            raise ValueError("G2b rank report contains an invalid rank")
        if rank in expected_digests:
            raise ValueError("G2b report contains duplicate ranks")
        expected_digests[rank] = _lower_sha256(
            rank_report.get("model_local_state_sha256"),
            name=f"G2b rank {rank} model digest",
        )
    if set(expected_digests) != set(range(G2_WORLD_SIZE)):
        raise ValueError("G2b report rank set differs from the two-rank contract")
    return report, expected_digests, _sha256_bytes(internal_bytes)


def _meta_state_names(model: Any) -> tuple[str, ...]:
    names: list[str] = []
    entries = [
        *(("parameter", name, value) for name, value in model.named_parameters()),
        *(("buffer", name, value) for name, value in model.named_buffers()),
    ]
    for kind, name, value in entries:
        local = value.to_local() if callable(getattr(value, "to_local", None)) else value
        if bool(getattr(value, "is_meta", False)) or bool(getattr(local, "is_meta", False)):
            names.append(f"{kind}:{name}")
            continue
        device = getattr(local, "device", None)
        if getattr(device, "type", None) == "meta":
            names.append(f"{kind}:{name}")
    return tuple(sorted(names))


def _assert_no_meta_state(model: Any, *, phase: str) -> None:
    names = _meta_state_names(model)
    if names:
        preview = ", ".join(names[:8])
        raise RuntimeError(f"G2b stage restore found meta state {phase}: {preview}")


def _assert_rank_digest_match(*, actual: str, expected: str, rank: int) -> None:
    if actual != expected:
        raise RuntimeError(f"rank {rank} restored model digest differs from G2b save")


def _validate_request(request: LingBotVLA2LTOPStageRequest) -> None:
    required = {
        "source checkout": request.source_checkout,
        "patch": request.patch,
        "training config": request.training_config,
        "released checkpoint": request.checkpoint_dir,
        "processor": request.processor_dir,
        "stage checkpoint": request.stage_checkpoint,
        "external G2 report": request.g2_report,
    }
    if request.runtime_hotfix is not None:
        required["runtime optimizer hotfix"] = request.runtime_hotfix
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"G2b stage restore required paths are absent: {missing}")
    if isinstance(request.seed, bool) or not isinstance(request.seed, int) or request.seed < 0:
        raise ValueError("G2b stage restore seed must be a non-negative integer")
    if (
        isinstance(request.maximum_control_tokens, bool)
        or not isinstance(request.maximum_control_tokens, int)
        or request.maximum_control_tokens <= 0
    ):
        raise ValueError("G2b stage restore maximum control tokens must be positive")
    if request.fsdp2_placement not in FSDP2_PLACEMENTS:
        raise ValueError("G2b stage restore FSDP2 placement is unsupported")


def prepare_lingbot_vla2_ltop_stage_transfer(
    request: LingBotVLA2LTOPStageRequest,
) -> LingBotVLA2LTOPStageContract:
    """Validate all immutable source, checkpoint, and report evidence before CUDA work."""

    _validate_request(request)
    patch_report = verify_native_patch(
        root=_REPOSITORY_ROOT,
        checkout=request.source_checkout,
        check_apply=True,
    )
    runtime_hotfix_report: dict[str, Any] | None = None
    if request.runtime_hotfix is None:
        prepared_source = validate_prepared_native_source(
            checkout=request.source_checkout,
            patch_path=request.patch,
        )
        if prepared_source.get("patched_source_sha256") != patch_report.get(
            "patched_source_sha256"
        ):
            raise RuntimeError(
                "G2b stage restore LingBot source differs from immutable patch replay"
            )
    else:
        runtime_hotfix_report = verify_muon_collective_hotfix(
            root=_REPOSITORY_ROOT,
            checkout=request.source_checkout,
            check_apply=True,
        )
        prepared_source = validate_prepared_native_source_with_muon_collective_hotfix(
            checkout=request.source_checkout,
            patch_path=request.patch,
            hotfix_path=request.runtime_hotfix,
        )
        base_hashes = patch_report.get("patched_source_sha256")
        runtime_hashes = prepared_source.get("patched_source_sha256")
        if not isinstance(base_hashes, dict) or not isinstance(runtime_hashes, dict):
            raise RuntimeError("G2b stage restore source replay omits patched-source digests")
        unchanged = {key: value for key, value in base_hashes.items() if key != str(MUON_SOURCE)}
        actual_unchanged = {
            key: value for key, value in runtime_hashes.items() if key != str(MUON_SOURCE)
        }
        if actual_unchanged != unchanged:
            raise RuntimeError("runtime optimizer hotfix changed model-bearing LingBot sources")
        if runtime_hashes != runtime_hotfix_report.get("patched_source_sha256"):
            raise RuntimeError("runtime optimizer hotfix source differs from immutable replay")
    validate_checkpoint(request.checkpoint_dir)
    validate_processor(request.processor_dir)
    checkpoint_inventory = _validate_model_only_checkpoint_tree(request.stage_checkpoint)
    model_identity = {
        "checkpoint_id": LINGBOT_CHECKPOINT_ID,
        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
        "native_source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        # G2 binds the model-bearing native overlay. The optimizer-only runtime
        # hotfix is separately attested and cannot alter this checkpoint identity.
        "patched_source_sha256": cast(dict[str, str], patch_report["patched_source_sha256"]),
    }
    g2_report, expected_digests, g2_report_sha256 = _validate_g2_report(
        stage_checkpoint=request.stage_checkpoint,
        external_report=request.g2_report,
        expected_model_identity=model_identity,
        expected_patch_sha256=cast(str, patch_report["patch_sha256"]),
    )
    return LingBotVLA2LTOPStageContract(
        request=request,
        prepared_source=prepared_source,
        patch_report=patch_report,
        runtime_hotfix_report=runtime_hotfix_report,
        checkpoint_inventory=checkpoint_inventory,
        model_identity=model_identity,
        g2_report=g2_report,
        expected_rank_digests=expected_digests,
        g2_report_sha256=g2_report_sha256,
    )


def _strict_model_only_dcp_load(
    *,
    checkpointer: Any,
    stage_checkpoint: Path,
    policy: Any,
    rank: int,
    dist_module: Any,
) -> None:
    state = {"model": policy}
    _distributed_rank_local_call(
        action=lambda: checkpointer.load(
            str(stage_checkpoint),
            state,
            allow_partial_load=False,
        ),
        phase="ltop-g2-stage-transfer-model-only-dcp-load",
        rank=rank,
        dist_module=dist_module,
    )
    if set(state) != {"model"} or state["model"] is not policy:
        raise RuntimeError("LingBot DCP load changed the model-only state boundary")


@contextmanager
def open_lingbot_vla2_ltop_stage_runtime(
    contract: LingBotVLA2LTOPStageContract,
) -> Iterator[LingBotVLA2LTOPStageRuntime]:
    """Build and restore the exact G2b runtime, keeping distributed state alive in scope."""

    request = contract.request
    sys.dont_write_bytecode = True
    source_path = str(request.source_checkout.resolve())
    while source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)

    import torch
    import torch.distributed as dist
    from lingbotvla.checkpoint import build_checkpointer
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    from picf_next.lingbot_native.host import (
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        install_lingbot_native_graph,
    )
    from picf_next.lingbot_native.representation_stage import (
        configure_native_representation_parameter_scope,
        verify_native_representation_parameter_scope,
    )

    if os.environ.get("WORLD_SIZE") != str(G2_WORLD_SIZE):
        raise RuntimeError("G2b stage restore requires torchrun with exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(G2_WORLD_SIZE):
        raise RuntimeError("G2b stage restore requires both ranks on one two-GPU host")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if torch.cuda.device_count() != G2_WORLD_SIZE:
            raise RuntimeError("G2b stage restore sees a CUDA topology other than two devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("G2b stage restore requires two A100 devices with at least 39 GiB")
        torch.manual_seed(request.seed)
        torch.cuda.manual_seed_all(request.seed)
        torch.cuda.reset_peak_memory_stats(device)
        init_parallel_state(
            dp_size=G2_WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=G2_WORLD_SIZE,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="fsdp2",
        )

        training = load_lingbot_training_config(request.training_config)
        train_section = training.get("train")
        if not isinstance(train_section, dict):
            raise ValueError("G2b stage restore LingBot training config omits train")
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=float(train_section.get("lr", 5.0e-5)),
        )
        steps = contract.g2_report.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("G2b report contains an invalid training step count")
        merged, _data_mapping = _resolve_training_config(
            training,
            checkpoint_dir=request.checkpoint_dir,
            processor_dir=request.processor_dir,
            num_steps=steps,
        )
        merged.update(
            {
                "use_cache": False,
                "use_compile": False,
                "attention_implementation": "eager",
                "vit_attn_implementation": "eager",
            }
        )
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        qwen_config = AutoConfig.from_pretrained(  # nosec B615
            request.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(request.processor_dir.resolve())

        build_started = time.perf_counter()
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        with init_empty_weights(), no_init_weights():
            policy = LingbotVlaV2Policy(config=config, eval=False).to(torch.float32)
        load_model_weights(
            policy,
            str(request.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        if alignment_teacher_prune != contract.g2_report.get("alignment_teacher_prune"):
            raise RuntimeError("G2b stage restore alignment-teacher topology differs")
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=G2_CAPACITY,
            maximum_control_tokens=request.maximum_control_tokens,
            task_query_count=G2_TASK_QUERY_COUNT,
            architecture_identity=G2_ARCHITECTURE,
        )
        graph = LingBotNativeGraph(graph_config, device=device, dtype=torch.float32)
        install_lingbot_native_graph(policy, graph)
        if graph.task_query_embeddings is None:
            raise RuntimeError("G2b stage restore graph omitted TASK_QUERY embeddings")
        policy.train()
        representation_scope = configure_native_representation_parameter_scope(policy)
        expected_scope = contract.g2_report["trainable_scope"]["representation_scope"]
        if representation_scope.as_dict() != expected_scope:
            raise RuntimeError("G2b stage restore representation scope differs from save time")
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=optimizer_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=request.fsdp2_placement == FSDP2_CPU_OFFLOAD,
            enable_shared_embedding_offload=(
                request.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
            ),
            fsdp_kwargs={},
            basic_modules=policy._no_split_modules,
            enable_reentrant=False,
            enable_forward_prefetch=False,
            fsdp_llm_blocks=False,
            ignore_norm=False,
            use_depth_align=False,
            split_fused_experts_from_decoder_fsdp=False,
            vlm_fsdp=True,
            use_future_image=False,
        )
        register_native_fsdp_forward_methods(policy)
        verify_native_representation_parameter_scope(
            policy,
            expected=representation_scope,
        )
        pre_restore_storage = _distributed_rank_local_call(
            action=lambda: _validate_fsdp2_parameter_storage(
                policy,
                torch,
                expected_placement=request.fsdp2_placement,
            ),
            phase="ltop-g2-stage-restore-preload-topology",
            rank=rank,
            dist_module=dist,
        )
        _distributed_rank_local_call(
            action=lambda: _assert_no_meta_state(policy, phase="before DCP load"),
            phase="ltop-g2-stage-restore-preload-no-meta",
            rank=rank,
            dist_module=dist,
        )
        build_duration_s = time.perf_counter() - build_started

        checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
        load_started = time.perf_counter()
        _strict_model_only_dcp_load(
            checkpointer=checkpointer,
            stage_checkpoint=request.stage_checkpoint,
            policy=policy,
            rank=rank,
            dist_module=dist,
        )
        torch.cuda.synchronize(device)
        load_duration_s = time.perf_counter() - load_started
        _distributed_rank_local_call(
            action=lambda: _assert_no_meta_state(policy, phase="after DCP load"),
            phase="ltop-g2-stage-restore-postload-no-meta",
            rank=rank,
            dist_module=dist,
        )
        post_restore_storage = _distributed_rank_local_call(
            action=lambda: _validate_fsdp2_parameter_storage(
                policy,
                torch,
                expected_placement=request.fsdp2_placement,
            ),
            phase="ltop-g2-stage-restore-postload-topology",
            rank=rank,
            dist_module=dist,
        )
        if post_restore_storage != pre_restore_storage:
            raise RuntimeError("G2b DCP load changed the rank-local FSDP2 topology")
        actual_digest = _distributed_rank_local_call(
            action=lambda: _model_local_state_digest(policy, torch),
            phase="ltop-g2-stage-restore-model-digest",
            rank=rank,
            dist_module=dist,
        )
        expected_digest = contract.expected_rank_digests[rank]
        _distributed_rank_local_call(
            action=lambda: _assert_rank_digest_match(
                actual=actual_digest,
                expected=expected_digest,
                rank=rank,
            ),
            phase="ltop-g2-stage-restore-model-digest-match",
            rank=rank,
            dist_module=dist,
        )
        yield LingBotVLA2LTOPStageRuntime(
            contract=contract,
            policy=policy,
            graph=graph,
            graph_config=graph_config,
            model_config=config,
            training_config=training,
            resolved_training_config=merged,
            representation_scope=representation_scope,
            optimizer_contract=optimizer_contract,
            runtime_modules=LingBotVLA2LTOPRuntimeModules(torch=torch, dist=dist),
            rank=rank,
            local_rank=local_rank,
            device=device,
            fsdp2_storage_before_load=pre_restore_storage,
            fsdp2_storage_after_load=post_restore_storage,
            expected_model_local_state_sha256=expected_digest,
            actual_model_local_state_sha256=actual_digest,
            model_build_s=build_duration_s,
            dcp_load_s=load_duration_s,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
