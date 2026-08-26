from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    SELECTIVE_EMBEDDING_MODULE,
    SELECTIVE_EMBEDDING_PARAMETER,
    SELECTIVE_FROZEN_VISION_MODULE,
)
from tools.bootstrap_lingbot_vla2_native import CHECKOUT_RELATIVE_PATH
from tools.cuda_allocator_bootstrap import (
    CUDA_ALLOCATOR_ALTERNATE_ENVIRONMENT_VARIABLE,
    CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
    CUDA_EXPANDABLE_SEGMENTS_CONFIG,
    CUDA_MALLOC_ASYNC_CONFIG,
)
from tools.cuda_allocator_bootstrap import (
    configure_cuda_allocator as _configure_cuda_allocator,
)
from tools.run_lingbot_vla2_native_g0 import (
    G0_LEGACY_ARCHITECTURE,
    G0_LTOP_ARCHITECTURE,
    G0_WORLD_SIZE,
    _checkpoint_boundary,
    _distributed_gradient_metrics,
    _distributed_rank_local_call,
    _execution_contract_digest,
    _fsync_tree,
    _g0_gradient_metric_fragments,
    _implementation_digest,
    _implementation_paths,
    _local_import_modules,
    _model_local_state_digest,
    _move_model_inputs,
    _parse_args,
    _rank_rng_digest,
    _validate_fsdp2_parameter_storage,
    _validate_optimizer_state,
    _validate_paths_and_args,
    _write_text_durable,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/run_lingbot_vla2_native_g0.py"
ROBOT_CONFIG = ROOT / "configs/lingbot/calvin_robot.yaml"


def _source() -> str:
    return TOOL.read_text()


_RANK_FAILURE_PHASES = (
    "forward-backward",
    "posterior-stage",
    "gradient-audit",
    "gradient-clip",
    "optimizer-step",
    "optimizer-finish",
    "checkpoint-save",
)


def _run_rank_failure_probe(
    rank: int,
    init_method: str,
    publish_root: str,
    result_queue: Any,
) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=G0_WORLD_SIZE,
        timeout=timedelta(seconds=10),
    )
    try:
        observed: list[str] = []
        for phase in _RANK_FAILURE_PHASES:

            def injected_action(current_phase: str = phase) -> str:
                if rank == 1:
                    raise RuntimeError(f"injected {current_phase} failure")
                return current_phase

            try:
                _distributed_rank_local_call(
                    action=injected_action,
                    phase=phase,
                    rank=rank,
                    dist_module=dist,
                )
            except RuntimeError as error:
                message = str(error)
                if f"injected {phase} failure" not in message or "rank 1" not in message:
                    raise AssertionError(
                        f"phase {phase} propagated the wrong failure: {message}"
                    ) from error
                observed.append(phase)
            else:
                (Path(publish_root) / f"{phase}-rank-{rank}.published").write_text("unsafe\n")
                raise AssertionError(f"phase {phase} returned after a remote-rank failure")
            dist.barrier()
        result_queue.put((rank, tuple(observed)))
    finally:
        dist.destroy_process_group()


def test_native_g0_delays_every_accelerator_and_host_import() -> None:
    tree = ast.parse(_source())
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    top_from_imports = {
        (node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)
    }
    forbidden = {"lingbotvla", "numpy", "torch", "transformers"}
    assert forbidden.isdisjoint(top_imports | top_from_imports)


@pytest.mark.parametrize(
    "tool",
    [
        TOOL,
        ROOT / "tools/run_lingbot_vla2_native_full.py",
    ],
)
def test_native_runner_imports_its_own_checkout_over_a_stale_editable(
    tmp_path: Path,
    tool: Path,
) -> None:
    stale = tmp_path / "stale"
    package = stale / "picf_next"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('stale editable imported')\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(stale)

    result = subprocess.run(
        [sys.executable, str(tool), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "stale editable imported" not in result.stderr


@pytest.mark.parametrize(
    "tool",
    [
        TOOL,
        ROOT / "tools/run_lingbot_vla2_native_full.py",
    ],
)
def test_native_runner_configures_allocator_before_torch_import(
    tmp_path: Path,
    tool: Path,
) -> None:
    wrapper = tmp_path / "probe_allocator_import_order.py"
    wrapper.write_text(
        "\n".join(
            (
                "import builtins",
                "import json",
                "import os",
                "import runpy",
                "import sys",
                f"tool = {str(tool)!r}",
                f"environment_name = {CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE!r}",
                f"expected = {CUDA_EXPANDABLE_SEGMENTS_CONFIG!r}",
                "original_import = builtins.__import__",
                "observed = []",
                "def guarded_import(name, *args, **kwargs):",
                "    if name == 'torch' or name.startswith('torch.'):",
                "        observed.append(os.environ.get(environment_name))",
                "        if observed[-1] != expected:",
                "            raise RuntimeError('torch imported before allocator bootstrap')",
                "    return original_import(name, *args, **kwargs)",
                "builtins.__import__ = guarded_import",
                "sys.argv = [tool, '--cuda-allocator', 'expandable-segments', '--help']",
                "try:",
                "    runpy.run_path(tool, run_name='__main__')",
                "except SystemExit as error:",
                "    exit_code = error.code",
                "else:",
                "    exit_code = 0",
                "print(json.dumps({'exit_code': exit_code, 'observed': observed}))",
            )
        )
        + "\n"
    )
    environment = dict(os.environ)
    environment.pop(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, None)
    environment.pop(CUDA_ALLOCATOR_ALTERNATE_ENVIRONMENT_VARIABLE, None)

    result = subprocess.run(
        [sys.executable, str(wrapper)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    probe = result.stdout.strip().splitlines()[-1]
    value = json.loads(probe)
    assert value["exit_code"] == 0
    assert value["observed"]
    assert set(value["observed"]) == {CUDA_EXPANDABLE_SEGMENTS_CONFIG}


def test_native_g0_has_no_historical_semantic_runtime() -> None:
    source = _source()
    for forbidden in (
        "picf_next.unified",
        "unified_belief",
        "action_layer_adapter",
        "semantic_scorer",
        "lifecycle",
        "confidence_controller",
    ):
        assert forbidden not in source


def test_native_g0_defaults_to_exact_native_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PICF_LINGBOT_NATIVE_SOURCE", raising=False)
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--phase", "fresh"])
    args = _parse_args()
    expected = ROOT / CHECKOUT_RELATIVE_PATH
    assert args.source_checkout == expected
    assert args.training_config == expected / "configs/vla/robotwin/robotwin.yaml"
    assert args.learning_rate == 1e-4
    assert args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
    assert args.cuda_allocator == "native"
    assert args.architecture_identity == G0_LEGACY_ARCHITECTURE
    assert args.task_query_count == 0
    assert G0_WORLD_SIZE == 2


def test_native_g0_ltop_audits_its_native_task_query_gradient_separately() -> None:
    assert _g0_gradient_metric_fragments(G0_LEGACY_ARCHITECTURE) == (
        ("native_graph", "picf_native_graph"),
        ("action_output", "action_out_proj"),
    )
    assert _g0_gradient_metric_fragments(G0_LTOP_ARCHITECTURE) == (
        ("native_graph", "picf_native_graph"),
        ("action_output", "action_out_proj"),
        ("task_query", "picf_native_graph.task_query_embeddings"),
    )
    with pytest.raises(ValueError, match="unknown architecture"):
        _g0_gradient_metric_fragments("unknown")


def test_native_g0_training_config_follows_an_overridden_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--phase",
            "fresh",
            "--source-checkout",
            str(source),
        ],
    )

    args = _parse_args()

    assert args.source_checkout == source
    assert args.training_config == source / "configs/vla/robotwin/robotwin.yaml"


def test_native_g0_allocator_is_explicit_and_rejects_ambient_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.delenv(CUDA_ALLOCATOR_ALTERNATE_ENVIRONMENT_VARIABLE, raising=False)
    _configure_cuda_allocator("native")
    assert CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE not in os.environ

    _configure_cuda_allocator("expandable-segments")
    assert os.environ[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] == CUDA_EXPANDABLE_SEGMENTS_CONFIG
    with pytest.raises(RuntimeError, match="refuses inherited"):
        _configure_cuda_allocator("expandable-segments")

    monkeypatch.delenv(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE)
    _configure_cuda_allocator("cuda-malloc-async")
    assert os.environ[CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE] == CUDA_MALLOC_ASYNC_CONFIG

    monkeypatch.delenv(CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE)
    monkeypatch.setenv(
        CUDA_ALLOCATOR_ALTERNATE_ENVIRONMENT_VARIABLE,
        "backend:cudaMallocAsync",
    )
    with pytest.raises(RuntimeError, match="refuses inherited"):
        _configure_cuda_allocator("native")


def test_native_g0_requires_the_audited_normalization_argument() -> None:
    config = ROBOT_CONFIG.read_text()
    assert "norm_stats: __REQUIRES_EXPLICIT_AUDITED_NORM_STATS_PATH__" in config
    assert "/mnt/" not in config
    assert "norm_stats_path=str(args.norm_stats.resolve())" in _source()


def test_native_g0_explicitly_types_every_root_model_input() -> None:
    source = {
        "images": torch.ones(1, 2, dtype=torch.float32),
        "tokens": torch.ones(1, 2, dtype=torch.long),
        "valid": torch.ones(1, 2, dtype=torch.bool),
        "metadata": "preserved",
    }
    moved = _move_model_inputs(
        source,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        torch_module=torch,
    )

    assert moved["images"].dtype == torch.bfloat16
    assert moved["tokens"].dtype == torch.long
    assert moved["valid"].dtype == torch.bool
    assert moved["metadata"] == "preserved"


def test_native_g0_source_contains_full_update_and_cold_resume_contract() -> None:
    source = _source()
    required = (
        'dist.init_process_group(backend="cpu:gloo,cuda:nccl")',
        "verify_native_patch(",
        "validate_prepared_native_source(",
        "sys.dont_write_bytecode = True",
        "require_persistent_run_root(args.run_dir)",
        "acquire_distributed_run_lease(",
        "validate_checkpoint(args.checkpoint_dir)",
        "validate_processor(args.processor_dir)",
        "build_parallelize_model(",
        "install_torch_2_8_sparse_optimizer_state_backport(torch)",
        "register_native_fsdp_forward_methods(policy)",
        "strip_targetless_alignment_teacher_heads(policy)",
        "require_lingbot_exact_resume_contract(optimizer_contract)",
        "build_lingbot_official_optimizer(",
        "enable_fp32=optimizer_contract.enable_fp32",
        'dp_mode="fsdp2"',
        "enable_fsdp_offload=full_cpu_offload",
        "enable_shared_embedding_offload=selective_embedding_offload",
        '"fsdp2_placement": args.fsdp2_placement',
        '"expected_fsdp2_placement": args.fsdp2_placement',
        "run_native_policy_training_forward(",
        "run_native_v3_two_pass_policy_training_forward(",
        "NativeTrainingLaneCoordinator(bank)",
        "attempt.finish(optimizer_attempt)",
        "checkpointer.save(",
        "checkpointer.load(",
        "NativeTrainingLaneBank.deserialize",
        "_restore_rank_rng(",
        "loaded_boundary != extra",
        "resume_runtime_rng_verified",
        "require_checkpoint_write_capacity(args.run_dir)",
        "require_checkpoint_write_capacity(checkpoint_root)",
        "require_checkpoint_copy=False",
        "precheckpoint_error",
        "dist.broadcast_object_list(precheckpoint_error, src=0)",
        "native G0 pre-checkpoint report validation failed",
        "validate_g0_report(report, **validation_kwargs)",
        "os.replace(output_checkpoint, staging_checkpoint)",
    )
    for fragment in required:
        assert fragment in source
    assert 'dist.init_process_group(backend="nccl")' not in source
    prepublication = source.index("require_checkpoint_copy=False")
    precheckpoint_capacity = source.index("require_checkpoint_write_capacity(checkpoint_root)")
    precheckpoint_broadcast = source.index("dist.broadcast_object_list(precheckpoint_error, src=0)")
    precheckpoint_failure = source.index("native G0 pre-checkpoint report validation failed")
    checkpoint_save = source.index("checkpointer.save(")
    dcp_backport = source.index("install_torch_2_8_sparse_optimizer_state_backport(torch)")
    checkpointer_import = source.index("from lingbotvla.checkpoint import build_checkpointer")
    checkpoint_load = source.index("checkpointer.load(")
    resume_report_validation = source.index(
        "native G0 checkpoint lacks its immutable report before load"
    )
    publication = source.index("os.replace(staging_checkpoint, output_checkpoint)")
    strict_validation = source.index("validate_g0_report(report, **validation_kwargs)")
    rollback = source.index("os.replace(output_checkpoint, staging_checkpoint)")
    assert (
        prepublication
        < precheckpoint_capacity
        < precheckpoint_broadcast
        < precheckpoint_failure
        < checkpoint_save
        < publication
        < strict_validation
        < rollback
    )
    assert dcp_backport < checkpointer_import < resume_report_validation < checkpoint_load
    assert source.rindex("_write_text_durable(", 0, source.index('print(payload, end="")')) < (
        source.index("dist.broadcast_object_list(publish_error, src=0)")
    )


def test_native_g0_orders_rank_error_exchange_before_commit_and_publication() -> None:
    source = _source()
    phases = (
        "data-prepare",
        "lane-prepare",
        "forward-backward",
        "posterior-stage",
        "gradient-audit",
        "gradient-clip",
        "optimizer-step",
        "optimizer-finish",
        "post-update-audit",
        "checkpoint-save",
    )
    phase_positions = [
        source.index(f'phase=f"native-g0-step-{{global_step}}-{phase}"') for phase in phases
    ]
    assert phase_positions == sorted(phase_positions)
    assert source.index("attempt.stage(") < phase_positions[3] < source.index("attempt.finish(")
    assert source.index("optimizer.step()") < phase_positions[6] < phase_positions[7]
    checkpoint_save = source.index("checkpointer.save(")
    checkpoint_publish = source.index("os.replace(staging_checkpoint, output_checkpoint)")
    assert checkpoint_save < phase_positions[-1] < checkpoint_publish


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
def test_native_g0_rank_local_failures_exit_bounded_without_publication(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_method = f"file://{tmp_path / 'gloo-init'}"
    processes = [
        context.Process(
            target=_run_rank_failure_probe,
            args=(rank, init_method, str(tmp_path), result_queue),
        )
        for rank in range(G0_WORLD_SIZE)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        hung = [process for process in processes if process.is_alive()]
        for process in hung:
            process.terminate()
            process.join(timeout=5)
        assert not hung, "rank-local failure left a distributed worker blocked"
        assert [process.exitcode for process in processes] == [0, 0]
        results = sorted(result_queue.get(timeout=2) for _ in processes)
        assert results == [(rank, _RANK_FAILURE_PHASES) for rank in range(G0_WORLD_SIZE)]
        assert not tuple(tmp_path.glob("*.published"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()


def test_native_g0_paths_and_report_are_fail_closed(tmp_path: Path) -> None:
    directory_names = ("source", "checkpoint", "processor", "split", "run")
    directories = {name: tmp_path / name for name in directory_names}
    for path in directories.values():
        path.mkdir()
    file_names = ("patch", "training", "robot", "data", "manifest", "norm")
    files = {name: tmp_path / f"{name}.json" for name in file_names}
    for path in files.values():
        path.write_text("{}")
    args = argparse.Namespace(
        phase="fresh",
        source_checkout=directories["source"],
        patch=files["patch"],
        training_config=files["training"],
        robot_config=files["robot"],
        data_config=files["data"],
        checkpoint_dir=directories["checkpoint"],
        processor_dir=directories["processor"],
        dataset_split=directories["split"],
        dataset_manifest=files["manifest"],
        norm_stats=files["norm"],
        run_dir=directories["run"],
        load_global_step=1,
        seed=7,
        capacity=16,
        maximum_control_tokens=8,
        maximum_optimizer_lag=8,
        architecture_identity=G0_LEGACY_ARCHITECTURE,
        task_query_count=0,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        maximum_peak_reserved_gib=39.0,
        fsdp2_placement="cpu-offload",
        cuda_allocator="native",
    )
    _validate_paths_and_args(args)
    args.capacity = 0
    with pytest.raises(ValueError, match="positive"):
        _validate_paths_and_args(args)
    args.capacity = 16
    args.architecture_identity = G0_LTOP_ARCHITECTURE
    with pytest.raises(ValueError, match="positive task-query"):
        _validate_paths_and_args(args)
    args.task_query_count = 4
    _validate_paths_and_args(args)
    args.architecture_identity = G0_LEGACY_ARCHITECTURE
    with pytest.raises(ValueError, match="cannot declare task-query"):
        _validate_paths_and_args(args)
    args.task_query_count = 0
    args.cuda_allocator = "unsupported"
    with pytest.raises(ValueError, match="CUDA allocator"):
        _validate_paths_and_args(args)

    output = tmp_path / "nested" / "native-g0.json"
    _write_text_durable(output, '{"status":"PASS"}\n')
    assert output.read_text() == '{"status":"PASS"}\n'
    assert not tuple(output.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        _write_text_durable(output, '{"status":"REPLACED"}\n')
    assert output.read_text() == '{"status":"PASS"}\n'

    target = tmp_path / "target.json"
    target.write_text("original\n")
    link = tmp_path / "report-link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        _write_text_durable(link, "replacement\n")
    assert target.read_text() == "original\n"


@pytest.mark.parametrize(
    ("device_type", "placement"),
    [("cpu", FSDP2_CPU_OFFLOAD), ("cuda", FSDP2_GPU_SHARDED)],
)
def test_native_g0_validates_explicit_fsdp2_parameter_placement(
    device_type: str,
    placement: str,
) -> None:
    class _Local:
        device = type("Device", (), {"type": device_type})()

        def numel(self) -> int:
            return 7

    class _Parameter:
        dtype = "float32"

        @staticmethod
        def to_local() -> _Local:
            return _Local()

    class _Model:
        @staticmethod
        def named_parameters():
            return iter((("weight", _Parameter()),))

    torch_stub = type("Torch", (), {"float32": "float32"})()
    assert _validate_fsdp2_parameter_storage(
        _Model(),
        torch_stub,
        expected_placement=placement,
    ) == {
        "parameter_tensors": 1,
        "local_elements": 7,
        "master_dtype": "float32",
        "placement": placement,
        "cpu_parameter_tensors": int(device_type == "cpu"),
        "cpu_local_elements": 7 if device_type == "cpu" else 0,
        "cuda_parameter_tensors": int(device_type == "cuda"),
        "cuda_local_elements": 7 if device_type == "cuda" else 0,
        "selective_cpu_parameter_names": [],
    }
    wrong_placement = FSDP2_GPU_SHARDED if device_type == "cpu" else FSDP2_CPU_OFFLOAD
    with pytest.raises(RuntimeError, match="FSDP2"):
        _validate_fsdp2_parameter_storage(
            _Model(),
            torch_stub,
            expected_placement=wrong_placement,
        )


def test_native_g0_validates_only_the_shared_embedding_on_cpu() -> None:
    class _Local:
        def __init__(self, device_type: str, elements: int) -> None:
            self.device = type("Device", (), {"type": device_type})()
            self._elements = elements

        def numel(self) -> int:
            return self._elements

    class _Parameter:
        dtype = "float32"

        def __init__(self, device_type: str, elements: int) -> None:
            self._local = _Local(device_type, elements)

        def to_local(self) -> _Local:
            return self._local

    class _Model:
        _lingbot_fsdp2_selective_cpu_modules = (SELECTIVE_EMBEDDING_MODULE,)

        @staticmethod
        def named_parameters():
            return iter(
                (
                    (SELECTIVE_EMBEDDING_PARAMETER, _Parameter("cpu", 7)),
                    ("model.action.weight", _Parameter("cuda", 3)),
                )
            )

    torch_stub = type("Torch", (), {"float32": "float32"})()
    assert _validate_fsdp2_parameter_storage(
        _Model(),
        torch_stub,
        expected_placement=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    ) == {
        "parameter_tensors": 2,
        "local_elements": 10,
        "master_dtype": "float32",
        "placement": FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        "cpu_parameter_tensors": 1,
        "cpu_local_elements": 7,
        "cuda_parameter_tensors": 1,
        "cuda_local_elements": 3,
        "selective_cpu_parameter_names": [SELECTIVE_EMBEDDING_PARAMETER],
    }


def test_native_g0_validates_declared_selective_class_parameters_on_cpu() -> None:
    wsa_prefix = "model.qwenvl_with_expert.adr218_wsa_training_runtime.future.expert"

    class _Local:
        def __init__(self, device_type: str, elements: int) -> None:
            self.device = type("Device", (), {"type": device_type})()
            self._elements = elements

        def numel(self) -> int:
            return self._elements

    class _Parameter:
        dtype = "float32"

        def __init__(self, device_type: str, elements: int) -> None:
            self._local = _Local(device_type, elements)

        def to_local(self) -> _Local:
            return self._local

    class _Model:
        _lingbot_fsdp2_selective_cpu_modules = (SELECTIVE_EMBEDDING_MODULE,)
        _lingbot_fsdp2_selective_cpu_module_classes = (
            "Future3DBlock",
            "Future3DExpert",
        )

        @staticmethod
        def named_parameters():
            return iter(
                (
                    (SELECTIVE_EMBEDDING_PARAMETER, _Parameter("cpu", 7)),
                    (f"{wsa_prefix}.blocks.0.attn.weight", _Parameter("cpu", 11)),
                    (f"{wsa_prefix}.head.weight", _Parameter("cpu", 5)),
                    ("model.action.weight", _Parameter("cuda", 3)),
                )
            )

    torch_stub = type("Torch", (), {"float32": "float32"})()
    report = _validate_fsdp2_parameter_storage(
        _Model(),
        torch_stub,
        expected_placement=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
        expected_selective_cpu_module_classes=("Future3DBlock", "Future3DExpert"),
        expected_selective_cpu_parameter_prefixes=(wsa_prefix,),
    )
    assert report["cpu_parameter_tensors"] == 3
    assert report["cuda_parameter_tensors"] == 1
    assert report["selective_cpu_module_classes"] == [
        "Future3DBlock",
        "Future3DExpert",
    ]
    assert report["selective_cpu_parameter_prefixes"] == [wsa_prefix]
    assert set(report["selective_cpu_parameter_names"]) == {
        SELECTIVE_EMBEDDING_PARAMETER,
        f"{wsa_prefix}.blocks.0.attn.weight",
        f"{wsa_prefix}.head.weight",
    }


def test_native_g0_validates_only_frozen_vision_blocks_and_embedding_on_cpu() -> None:
    vision_module = "model.qwenvl_with_expert.qwenvl.model.visual.blocks.0"
    vision_parameter = f"{vision_module}.attn.qkv.weight"

    class _Local:
        def __init__(self, device_type: str, elements: int) -> None:
            self.device = type("Device", (), {"type": device_type})()
            self._elements = elements

        def numel(self) -> int:
            return self._elements

    class _Parameter:
        dtype = "float32"

        def __init__(self, device_type: str, elements: int, *, requires_grad: bool) -> None:
            self._local = _Local(device_type, elements)
            self.requires_grad = requires_grad

        def to_local(self) -> _Local:
            return self._local

    class _Model:
        _lingbot_vlm_fsdp2_topology = {
            "text": ("model.qwenvl_with_expert.qwenvl.model.language_model.layers.0",),
            "vision": (vision_module,),
        }
        _lingbot_fsdp2_selective_cpu_modules = (
            SELECTIVE_FROZEN_VISION_MODULE,
            SELECTIVE_EMBEDDING_MODULE,
        )

        @staticmethod
        def named_parameters():
            return iter(
                (
                    (vision_parameter, _Parameter("cpu", 4, requires_grad=False)),
                    (
                        SELECTIVE_EMBEDDING_PARAMETER,
                        _Parameter("cpu", 7, requires_grad=False),
                    ),
                    ("model.action.weight", _Parameter("cuda", 3, requires_grad=True)),
                )
            )

    torch_stub = type("Torch", (), {"float32": "float32"})()
    report = _validate_fsdp2_parameter_storage(
        _Model(),
        torch_stub,
        expected_placement=FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
    )
    assert report["cpu_parameter_tensors"] == 2
    assert report["cuda_parameter_tensors"] == 1
    assert set(report["selective_cpu_parameter_names"]) == {
        SELECTIVE_EMBEDDING_PARAMETER,
        vision_parameter,
    }


def test_native_g0_validates_trainable_vision_and_selective_class_cpu_storage() -> None:
    vision_module = "model.qwenvl_with_expert.qwenvl.model.visual.blocks.0"
    vision_parameter = f"{vision_module}.attn.qkv.weight"
    wsa_prefix = "model.qwenvl_with_expert.adr218_wsa_training_runtime.future.expert"
    wsa_parameter = f"{wsa_prefix}.blocks.0.attn.weight"

    class _Local:
        def __init__(self, device_type: str, elements: int) -> None:
            self.device = type("Device", (), {"type": device_type})()
            self._elements = elements

        def numel(self) -> int:
            return self._elements

    class _Parameter:
        dtype = "float32"

        def __init__(self, device_type: str, elements: int, *, requires_grad: bool) -> None:
            self._local = _Local(device_type, elements)
            self.requires_grad = requires_grad

        def to_local(self) -> _Local:
            return self._local

    class _Model:
        _lingbot_vlm_fsdp2_topology = {
            "text": ("model.qwenvl_with_expert.qwenvl.model.language_model.layers.0",),
            "vision": (vision_module,),
        }
        _lingbot_fsdp2_selective_cpu_modules = (
            SELECTIVE_FROZEN_VISION_MODULE,
            SELECTIVE_EMBEDDING_MODULE,
        )
        _lingbot_fsdp2_selective_cpu_module_classes = (
            "Future3DBlock",
            "Future3DExpert",
        )

        @staticmethod
        def named_parameters():
            return iter(
                (
                    (vision_parameter, _Parameter("cpu", 4, requires_grad=True)),
                    (
                        SELECTIVE_EMBEDDING_PARAMETER,
                        _Parameter("cpu", 7, requires_grad=True),
                    ),
                    (wsa_parameter, _Parameter("cpu", 11, requires_grad=True)),
                    ("model.action.weight", _Parameter("cuda", 3, requires_grad=True)),
                )
            )

    torch_stub = type("Torch", (), {"float32": "float32"})()
    report = _validate_fsdp2_parameter_storage(
        _Model(),
        torch_stub,
        expected_placement=FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
        expected_selective_cpu_module_classes=("Future3DBlock", "Future3DExpert"),
        expected_selective_cpu_parameter_prefixes=(wsa_prefix,),
    )
    assert report["cpu_parameter_tensors"] == 3
    assert report["cuda_parameter_tensors"] == 1
    assert set(report["selective_cpu_parameter_names"]) == {
        SELECTIVE_EMBEDDING_PARAMETER,
        vision_parameter,
        wsa_parameter,
    }


def test_native_g0_implementation_digest_covers_transitive_local_imports() -> None:
    paths = {str(path.relative_to(ROOT)) for path in _implementation_paths(ROOT)}
    required = {
        "src/picf_next/__init__.py",
        "src/picf_next/contracts.py",
        "src/picf_next/data/calvin.py",
        "src/picf_next/data/lingbot_calvin.py",
        "src/picf_next/data/lingbot_libero.py",
        "src/picf_next/data/robot_record.py",
        "src/picf_next/lingbot_native/capacity.py",
        "src/picf_next/lingbot_native/training.py",
        "tools/bootstrap_lingbot_vla2.py",
        "tools/bootstrap_lingbot_vla2_native.py",
        "tools/lingbot_vla2_runtime_helpers.py",
        "tools/run_lingbot_vla2_native_g0.py",
    }
    assert required <= paths
    assert {
        "src/picf_next/association.py",
        "src/picf_next/hosts/context.py",
        "src/picf_next/hosts/lingbot_vla2.py",
        "src/picf_next/posterior.py",
    }.isdisjoint(paths)
    digest = _implementation_digest(ROOT)
    assert len(digest) == 64
    assert digest == _implementation_digest(ROOT)


def test_native_g0_execution_contract_binds_allocator_and_placement() -> None:
    args = SimpleNamespace(
        data_config=TOOL,
        dataset_manifest=TOOL,
        norm_stats=TOOL,
        robot_config=TOOL,
        training_config=TOOL,
        max_grad_norm=1.0,
        capacity=16,
        maximum_control_tokens=8,
        maximum_optimizer_lag=8,
        architecture_identity=G0_LEGACY_ARCHITECTURE,
        task_query_count=0,
        maximum_peak_reserved_gib=39.0,
        fsdp2_placement="cpu-offload",
        cuda_allocator="native",
        seed=20260721,
    )
    kwargs = {
        "root": ROOT,
        "args": args,
        "patched_source_sha256": {"lingbotvla/native.py": "a" * 64},
        "optimizer_contract": {"algorithm": "official-muon"},
    }
    initial = _execution_contract_digest(**kwargs)

    args.fsdp2_placement = "gpu-sharded"
    assert _execution_contract_digest(**kwargs) != initial
    args.fsdp2_placement = "cpu-offload"

    args.cuda_allocator = "expandable-segments"
    assert _execution_contract_digest(**kwargs) != initial

    args.cuda_allocator = "native"
    args.architecture_identity = G0_LTOP_ARCHITECTURE
    args.task_query_count = 4
    assert _execution_contract_digest(**kwargs) != initial


def test_local_import_closure_resolves_relative_package_submodules(tmp_path: Path) -> None:
    package = tmp_path / "src/picf_next/fixture"
    package.mkdir(parents=True)
    initializer = package / "__init__.py"
    initializer.write_text("from . import leaf\nfrom .nested import VALUE\n")
    (package / "leaf.py").write_text("VALUE = 1\n")
    (package / "nested.py").write_text("VALUE = 2\n")

    assert set(_local_import_modules(tmp_path, initializer)) == {
        "picf_next.fixture",
        "picf_next.fixture.leaf",
        "picf_next.fixture.nested",
    }


def test_native_g0_fsyncs_complete_staging_tree_and_rejects_symlinks(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    nested = staging / "model"
    nested.mkdir(parents=True)
    (nested / "rank0.distcp").write_bytes(b"checkpoint")
    (staging / "metadata.json").write_text("{}")
    _fsync_tree(staging)
    assert (nested / "rank0.distcp").read_bytes() == b"checkpoint"
    (staging / "unsafe").symlink_to(nested / "rank0.distcp")
    with pytest.raises(ValueError, match="symlink"):
        _fsync_tree(staging)


def test_native_g0_rng_digest_is_order_stable_and_content_sensitive() -> None:
    state = {
        "python_json": b"[3,[1,2],null]",
        "numpy_json": b'{"keys":[1]}',
        "torch_cpu": b"cpu",
        "torch_cuda": b"cuda",
    }
    digest = _rank_rng_digest(state)
    assert digest == _rank_rng_digest(dict(reversed(tuple(state.items()))))
    changed = dict(state)
    changed["torch_cuda"] = b"cuda-changed"
    assert _rank_rng_digest(changed) != digest


def test_distributed_gradient_metrics_aggregate_without_per_parameter_reads() -> None:
    class LocalDist:
        class ReduceOp:
            MIN = "min"
            SUM = "sum"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op in {LocalDist.ReduceOp.MIN, LocalDist.ReduceOp.SUM}

        @staticmethod
        def get_world_size() -> int:
            return 1

        @staticmethod
        def all_gather_object(values: list[object], value: object) -> None:
            values[0] = value

    model = torch.nn.Module()
    model.native_graph = torch.nn.Linear(2, 2, bias=False)
    model.action_out_proj = torch.nn.Linear(2, 1, bias=False)
    model.native_graph.weight.grad = torch.full_like(model.native_graph.weight, 2.0)
    model.action_out_proj.weight.grad = torch.full_like(model.action_out_proj.weight, 3.0)

    metrics = _distributed_gradient_metrics(
        model,
        (("native", "native_graph"), ("action", "action_out_proj")),
        device=torch.device("cpu"),
        dist=LocalDist,
        torch_module=torch,
    )
    assert metrics == {
        "all_finite": True,
        "native_norm": 4.0,
        "native_elements": 4,
        "action_norm": pytest.approx(3.0 * 2**0.5),
        "action_elements": 2,
    }

    model.native_graph.weight.grad[0, 0] = float("nan")
    failed = _distributed_gradient_metrics(
        model,
        (("native", "native_graph"),),
        device=torch.device("cpu"),
        dist=LocalDist,
        torch_module=torch,
    )
    assert failed == {"all_finite": False}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA device")
def test_distributed_gradient_metrics_support_mixed_cpu_cuda_placement() -> None:
    class LocalDist:
        class ReduceOp:
            MIN = "min"
            SUM = "sum"

        @staticmethod
        def all_reduce(_value: torch.Tensor, *, op: str) -> None:
            assert op in {LocalDist.ReduceOp.MIN, LocalDist.ReduceOp.SUM}

        @staticmethod
        def get_world_size() -> int:
            return 1

        @staticmethod
        def all_gather_object(values: list[object], value: object) -> None:
            values[0] = value

    model = torch.nn.Module()
    model.mixed_cpu = torch.nn.Linear(2, 2, bias=False, device="cpu")
    model.mixed_cuda = torch.nn.Linear(2, 1, bias=False, device="cuda")
    model.mixed_cpu.weight.grad = torch.full_like(model.mixed_cpu.weight, 2.0)
    model.mixed_cuda.weight.grad = torch.full_like(model.mixed_cuda.weight, 3.0)

    metrics = _distributed_gradient_metrics(
        model,
        (("mixed", "mixed_"),),
        device=torch.device("cuda"),
        dist=LocalDist,
        torch_module=torch,
    )
    assert metrics == {
        "all_finite": True,
        "mixed_norm": pytest.approx((4 * 2.0**2 + 2 * 3.0**2) ** 0.5),
        "mixed_elements": 6,
    }


def test_native_g0_checkpoint_boundary_binds_model_optimizer_lane_and_rng() -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(4, 3)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    summary = _validate_optimizer_state(optimizer, torch, expected_step=1)
    assert summary["optimizer_state_entries"] == 2
    assert summary["optimizer_local_moment_elements"] > 0

    rng = {
        "python_json": b"[3,[1,2],null]",
        "numpy_json": b'{"keys":[1]}',
        "torch_cpu": b"cpu",
        "torch_cuda": b"cuda",
    }
    first = _checkpoint_boundary(
        model=model,
        optimizer=optimizer,
        lane_snapshot=b"lane-state",
        rank_rng_state=rng,
        torch_module=torch,
    )
    repeated = _checkpoint_boundary(
        model=model,
        optimizer=optimizer,
        lane_snapshot=b"lane-state",
        rank_rng_state=rng,
        torch_module=torch,
    )
    assert repeated == first

    with torch.no_grad():
        model.weight.add_(1)
    changed_model = _checkpoint_boundary(
        model=model,
        optimizer=optimizer,
        lane_snapshot=b"lane-state",
        rank_rng_state=rng,
        torch_module=torch,
    )
    assert changed_model["model_local_state_sha256"] != first["model_local_state_sha256"]
    assert changed_model["optimizer_local_state_sha256"] == first["optimizer_local_state_sha256"]
    changed_lane = _checkpoint_boundary(
        model=model,
        optimizer=optimizer,
        lane_snapshot=b"different-lane-state",
        rank_rng_state=rng,
        torch_module=torch,
    )
    assert changed_lane["lane_snapshot_sha256"] != first["lane_snapshot_sha256"]


def test_model_boundary_matches_persistent_state_dict_buffer_semantics() -> None:
    class Nested(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("persistent_stat", torch.ones(2), persistent=True)
            self.register_buffer("runtime_counter", torch.ones(2), persistent=False)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.nested = Nested()

    model = Model()
    first = _model_local_state_digest(model, torch)
    assert "nested.runtime_counter" not in model.state_dict()

    model.nested.runtime_counter.add_(7)
    assert _model_local_state_digest(model, torch) == first

    model.nested.persistent_stat.add_(1)
    assert _model_local_state_digest(model, torch) != first


def test_native_g0_optimizer_validation_accepts_official_muon_momentum() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = SimpleNamespace(
        state={parameter: {"momentum_buffer": torch.full_like(parameter, 0.5)}}
    )
    summary = _validate_optimizer_state(optimizer, torch, expected_step=7)
    assert summary == {
        "optimizer_state_entries": 1,
        "optimizer_local_moment_elements": 6,
    }
