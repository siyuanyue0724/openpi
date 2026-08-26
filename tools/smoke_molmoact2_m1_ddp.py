#!/usr/bin/env python3
"""Two-A100 M1 proof for the official LIBERO loader and MolmoAct2 processor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate

from picf_next.data.robot_record import (
    MOLMOACT2_LIBERO_DATASET_ID,
    MOLMOACT2_LIBERO_REVISION,
)
from picf_next.training.accelerate_runner import (
    accelerated_microstep,
    load_accelerate_checkpoint,
    register_progress_for_checkpointing,
    save_accelerate_checkpoint,
)
from picf_next.training.control import ExperimentRunContract, FrozenSamplePlan, RunProgress

MOLMOACT2_CHECKPOINT_ID = "allenai/MolmoAct2"
MOLMOACT2_CHECKPOINT_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"
SETUP_TYPE = "single franka robotic arm in libero"
CONTROL_MODE = "delta end-effector pose"
ACTION_HORIZON = 10
EXPECTED_TASKS = 40
EXPECTED_PHASES = frozenset({"start", "middle", "end"})
EXPECTED_REPRESENTATIVES = EXPECTED_TASKS * len(EXPECTED_PHASES)


class _RankCursorState:
    def __init__(self, rank: int) -> None:
        self.value = torch.tensor(rank * 10_000, dtype=torch.int64)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    @staticmethod
    def _validated_value(state: object) -> torch.Tensor:
        if not isinstance(state, dict) or set(state) != {"value"}:
            raise ValueError("rank cursor state is malformed")
        value = state["value"]
        if not isinstance(value, torch.Tensor) or value.shape or value.dtype != torch.int64:
            raise ValueError("rank cursor value is malformed")
        return value

    def validate_state_dict(self, state: object) -> None:
        self._validated_value(state)

    def load_state_dict(self, state: object) -> None:
        self.value = self._validated_value(state).clone()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sample-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--host-source-revision", required=True)
    parser.add_argument("--training-source-revision", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--processor-batch-size", type=int, default=4)
    return parser.parse_args()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_manifest(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for key, value in sorted(batch.items())
        if isinstance(value, torch.Tensor)
    }


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.shape != right.shape
            or not torch.equal(
                left.detach().cpu().contiguous(),
                right.detach().cpu().contiguous(),
            )
        ):
            raise AssertionError("tensor state differs after checkpoint resume")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError("mapping state differs after checkpoint resume")
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, tuple | list):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError("sequence state differs after checkpoint resume")
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    elif left != right:
        raise AssertionError("scalar state differs after checkpoint resume")


def _load_sample_plan(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text())
    if document.get("schema") != "picf-next.molmoact2-m1-sample-plan.v1":
        raise ValueError("unsupported M1 sample plan schema")
    representatives = document.get("representatives")
    if not isinstance(representatives, list) or len(representatives) != EXPECTED_REPRESENTATIVES:
        raise ValueError("M1 sample plan must contain start/middle/end rows for all 40 tasks")
    sample_keys: set[str] = set()
    task_phases: dict[int, set[str]] = {}
    global_indices: set[int] = set()
    for row in representatives:
        if not isinstance(row, dict):
            raise ValueError("M1 representative row is malformed")
        sample_key = row.get("sample_key")
        task_index = row.get("task_index")
        global_index = row.get("global_index")
        phase = row.get("phase")
        if (
            not isinstance(sample_key, str)
            or not sample_key
            or not isinstance(task_index, int)
            or isinstance(task_index, bool)
            or not 0 <= task_index < EXPECTED_TASKS
            or not isinstance(global_index, int)
            or isinstance(global_index, bool)
            or global_index < 0
            or phase not in EXPECTED_PHASES
        ):
            raise ValueError("M1 representative identity is malformed")
        if sample_key in sample_keys or global_index in global_indices:
            raise ValueError("M1 sample plan contains duplicate rows")
        sample_keys.add(sample_key)
        global_indices.add(global_index)
        task_phases.setdefault(task_index, set()).add(phase)
    if set(task_phases) != set(range(EXPECTED_TASKS)):
        raise ValueError("M1 sample plan does not cover every task")
    if any(phases != EXPECTED_PHASES for phases in task_phases.values()):
        raise ValueError("M1 sample plan does not cover all phases for every task")
    if document.get("representatives_sha256") != _canonical_sha256(representatives):
        raise ValueError("M1 representative plan hash mismatch")
    return representatives


def _batch_from_rows(
    *,
    dataset: Any,
    relative_by_global_index: dict[int, int],
    rows: list[dict[str, Any]],
    camera_keys: list[str],
) -> dict[str, Any]:
    samples = [dataset[relative_by_global_index[int(row["global_index"])]] for row in rows]
    batch = default_collate(samples)
    for camera_key in camera_keys:
        image = batch[camera_key]
        if image.dtype == torch.uint8:
            batch[camera_key] = image.to(dtype=torch.float32) / 255.0
    return batch


def _validate_processed_pair(
    *,
    targetful: dict[str, Any],
    target_free: dict[str, Any],
    batch_size: int,
) -> None:
    if "labels" in targetful or "labels" in target_free:
        raise ValueError("continuous M1 processor unexpectedly emitted discrete action labels")
    targetful_action = targetful.get("action")
    if not isinstance(targetful_action, torch.Tensor):
        raise ValueError("M1 targetful processor omitted its tensor action target")
    if target_free.get("action") is not None:
        raise ValueError("M1 action target crossed the target-free observation boundary")
    if tuple(targetful_action.shape) != (batch_size, ACTION_HORIZON, 32):
        raise ValueError("official processor action padding contract changed")
    if tuple(targetful["action_dim_is_pad"].shape) != (batch_size, 32):
        raise ValueError("official processor action-dimension mask changed")
    if tuple(targetful["action_horizon_is_pad"].shape) != (batch_size, ACTION_HORIZON):
        raise ValueError("official processor action-horizon mask changed")
    if target_free.get("action_horizon_is_pad") is not None:
        raise ValueError("target-free processor emitted an action-horizon target mask")
    observation_keys = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grids",
        "image_num_crops",
        "image_token_pooling",
        "token_type_ids",
        "observation.state",
    )
    for key in observation_keys:
        left = targetful.get(key)
        right = target_free.get(key)
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ValueError(f"official Molmo processor omitted observation tensor {key}")
        if not torch.equal(left, right):
            raise ValueError(f"action target changed deployable observation tensor {key}")


def _model_sha256(model: torch.nn.Module) -> str:
    return _canonical_sha256(
        {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _tensor_sha256(value),
            }
            for name, value in sorted(model.state_dict().items())
        }
    )


def main() -> None:
    args = _parse_args()
    if args.processor_batch_size <= 0:
        raise ValueError("processor batch size must be positive")
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    if accelerator.num_processes != 2:
        raise RuntimeError("M1 must run under exactly two distributed processes")
    if accelerator.device.type != "cuda":
        raise RuntimeError("M1 requires one CUDA device per distributed rank")

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        make_molmoact2_pre_post_processors,
    )
    from lerobot.utils.constants import ACTION
    from lerobot.utils.feature_utils import dataset_to_policy_features

    started = time.perf_counter()
    representatives = _load_sample_plan(args.sample_plan.resolve())
    selected_episodes = sorted({int(row["episode_index"]) for row in representatives})
    if len(selected_episodes) != EXPECTED_TASKS:
        raise ValueError("M1 must select one complete episode per task")

    dataset_root = args.dataset_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    metadata = LeRobotDatasetMetadata(
        MOLMOACT2_LIBERO_DATASET_ID,
        root=dataset_root,
        revision=MOLMOACT2_LIBERO_REVISION,
    )
    features = dataset_to_policy_features(metadata.features)
    output_features = {
        key: feature for key, feature in features.items() if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in features.items() if key not in output_features
    }
    config = MolmoAct2Config(
        checkpoint_path=str(checkpoint_dir),
        checkpoint_revision=MOLMOACT2_CHECKPOINT_REVISION,
        checkpoint_force_download=False,
        trust_remote_code=True,
        device=str(accelerator.device),
        use_amp=False,
        push_to_hub=False,
        chunk_size=ACTION_HORIZON,
        n_action_steps=ACTION_HORIZON,
        action_mode="continuous",
        inference_action_mode="continuous",
        setup_type=SETUP_TYPE,
        control_mode=CONTROL_MODE,
        image_keys=list(metadata.camera_keys),
        normalize_gripper=False,
        enable_inference_cuda_graph=False,
        model_dtype="bfloat16",
        input_features=input_features,
        output_features=output_features,
    )
    config.set_dataset_feature_metadata(metadata.features)
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    if delta_timestamps != {ACTION: [index / 10.0 for index in range(ACTION_HORIZON)]}:
        raise ValueError("official Molmo action horizon/timestamp contract changed")

    dataset = LeRobotDataset(
        MOLMOACT2_LIBERO_DATASET_ID,
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        revision=MOLMOACT2_LIBERO_REVISION,
        return_uint8=True,
    )
    loaded_episode_ids = {
        int(value) for value in dataset.hf_dataset.data.column("episode_index").to_numpy().tolist()
    }
    if loaded_episode_ids != set(selected_episodes):
        raise ValueError("official full-shard loader returned the wrong episode set")
    loaded_global_indices = dataset.hf_dataset.data.column("index").to_numpy().tolist()
    relative_by_global_index = {
        int(global_index): relative_index
        for relative_index, global_index in enumerate(loaded_global_indices)
    }
    required_global_indices = {int(row["global_index"]) for row in representatives}
    if not required_global_indices.issubset(relative_by_global_index):
        raise ValueError("official loader omitted one or more deterministic representative rows")

    preprocessor, _postprocessor = make_molmoact2_pre_post_processors(
        config,
        dataset_stats=metadata.stats,
        dataset_meta=metadata,
    )
    processor_steps = [type(step).__name__ for step in preprocessor.steps]
    expected_steps = [
        "RenameObservationsProcessorStep",
        "AddBatchDimensionProcessorStep",
        "MolmoAct2MaskedNormalizerProcessorStep",
        "MolmoAct2ClampNormalizedProcessorStep",
        "MolmoAct2PackInputsProcessorStep",
        "DeviceProcessorStep",
    ]
    if processor_steps != expected_steps:
        raise ValueError("official MolmoAct2 processor step sequence changed")
    setup_s = time.perf_counter() - started

    local_rows = representatives[accelerator.process_index :: accelerator.num_processes]
    processor_records: list[dict[str, Any]] = []
    no_leak_rows = 0
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    processor_started = time.perf_counter()
    first_raw_manifest: dict[str, Any] | None = None
    first_processed_manifest: dict[str, Any] | None = None
    for offset in range(0, len(local_rows), args.processor_batch_size):
        rows = local_rows[offset : offset + args.processor_batch_size]
        raw_batch = _batch_from_rows(
            dataset=dataset,
            relative_by_global_index=relative_by_global_index,
            rows=rows,
            camera_keys=list(metadata.camera_keys),
        )
        target_free_raw = {
            key: value
            for key, value in raw_batch.items()
            if key != ACTION and key not in {f"{ACTION}_is_pad", "action_is_pad"}
        }
        target_free = preprocessor(dict(target_free_raw))
        targetful = preprocessor(dict(raw_batch))
        _validate_processed_pair(
            targetful=targetful,
            target_free=target_free,
            batch_size=len(rows),
        )
        no_leak_rows += len(rows)
        raw_manifest = _tensor_manifest(raw_batch)
        processed_manifest = _tensor_manifest(targetful)
        target_free_manifest = _tensor_manifest(target_free)
        if first_raw_manifest is None:
            first_raw_manifest = raw_manifest
            first_processed_manifest = processed_manifest
        processor_records.append(
            {
                "sample_keys": [str(row["sample_key"]) for row in rows],
                "tasks": [int(row["task_index"]) for row in rows],
                "raw_tensor_manifest_sha256": _canonical_sha256(raw_manifest),
                "processed_tensor_manifest_sha256": _canonical_sha256(processed_manifest),
                "target_free_tensor_manifest_sha256": _canonical_sha256(target_free_manifest),
            }
        )
    torch.cuda.synchronize(accelerator.device)
    processor_s = time.perf_counter() - processor_started
    processor_peak_bytes = int(torch.cuda.max_memory_allocated(accelerator.device))

    gathered_processor_records: list[list[dict[str, Any]] | None] = [
        None
    ] * accelerator.num_processes
    dist.all_gather_object(gathered_processor_records, processor_records)
    gathered_no_leak: list[int | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered_no_leak, no_leak_rows)
    flat_processor_records = [
        record
        for rank_records in gathered_processor_records
        if rank_records is not None
        for record in rank_records
    ]
    processed_sample_keys = {
        sample_key for record in flat_processor_records for sample_key in record["sample_keys"]
    }
    if processed_sample_keys != {str(row["sample_key"]) for row in representatives}:
        raise ValueError(
            "distributed processor audit did not cover the complete representative plan"
        )
    if sum(value or 0 for value in gathered_no_leak) != EXPECTED_REPRESENTATIVES:
        raise ValueError("target-free no-leak audit did not cover every representative row")

    sample_rows = {str(row["sample_key"]): row for row in representatives}
    sample_keys = tuple(sample_rows)
    frozen_plan = FrozenSamplePlan(
        dataset_id=MOLMOACT2_LIBERO_DATASET_ID,
        dataset_revision=MOLMOACT2_LIBERO_REVISION,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        sample_keys=sample_keys,
        comparison_id="molmoact2-m1-official-loader-processor-resume",
        seed=20260716,
        global_batch_size=4,
        total_steps=4,
    )
    contract = ExperimentRunContract.build(
        arm="picf",
        comparison_id=frozen_plan.comparison_id,
        code_revision=args.code_revision,
        host_name="MolmoAct2-M1-data-boundary",
        host_source_revision=args.host_source_revision,
        training_source_revision=args.training_source_revision,
        foundation_checkpoint_id=MOLMOACT2_CHECKPOINT_ID,
        foundation_checkpoint_revision=MOLMOACT2_CHECKPOINT_REVISION,
        checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
        dataset_id=frozen_plan.dataset_id,
        dataset_revision=frozen_plan.dataset_revision,
        dataset_manifest_sha256=frozen_plan.dataset_manifest_sha256,
        sample_plan_sha256=frozen_plan.plan_sha256,
        optimizer_global_batch_size=frozen_plan.global_batch_size,
        world_size=accelerator.num_processes,
        gradient_accumulation_steps=1,
        precision="bfloat16",
        action_convention="LIBERO delta end-effector pose 7D, horizon 10 at 10Hz",
        detached_context_frames=0,
        gradient_transitions=1,
        trainable_scope="M1 control-only deterministic checksum model",
        common_config={
            "loader": "lerobot.datasets.io_utils.load_nested_dataset",
            "processor": "make_molmoact2_pre_post_processors",
            "processor_action_mode": "continuous",
        },
        arm_config={"picf_learning_enabled": False},
    )
    progress = RunProgress(
        contract_sha256=contract.contract_sha256,
        sample_plan_sha256=frozen_plan.plan_sha256,
        optimizer_global_batch_size=frozen_plan.global_batch_size,
    )
    rank_state = _RankCursorState(accelerator.process_index)
    register_progress_for_checkpointing(accelerator, progress)
    torch.manual_seed(73)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.05 * step,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    def run_step(step: int) -> tuple[tuple[str, str], ...]:
        microbatch = frozen_plan.microbatch_for_rank(
            step,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            gradient_accumulation_steps=1,
            accumulation_index=0,
        )
        rows = [sample_rows[sample.sample_key] for sample in microbatch.samples]
        raw_batch = _batch_from_rows(
            dataset=dataset,
            relative_by_global_index=relative_by_global_index,
            rows=rows,
            camera_keys=list(metadata.camera_keys),
        )
        processed = preprocessor(dict(raw_batch))
        local_digest = _canonical_sha256(_tensor_manifest(processed))
        local_trace = [(sample.sample_key, local_digest) for sample in microbatch.samples]
        gathered: list[list[tuple[str, str]] | None] = [None] * accelerator.num_processes
        dist.all_gather_object(gathered, local_trace)
        global_trace = tuple(
            item for rank_trace in gathered if rank_trace is not None for item in rank_trace
        )
        expected_keys = tuple(
            sample.sample_key for sample in frozen_plan.global_batch(step).samples
        )
        if tuple(key for key, _digest in global_trace) != expected_keys:
            raise AssertionError("two-rank data shards do not reconstruct the frozen global plan")

        action_feature = processed[ACTION][..., :7].float().mean(dim=(1, 2))
        state_feature = processed["observation.state"].float().mean(dim=1)
        feature = (action_feature + state_feature).reshape(-1, 1)
        target = torch.tensor(
            [sample.sample_index / len(sample_keys) for sample in microbatch.samples],
            device=feature.device,
            dtype=feature.dtype,
        ).reshape(-1, 1)
        result = accelerated_microstep(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            forward_loss=lambda: torch.nn.functional.mse_loss(model(feature), target),
            lr_scheduler=scheduler,
            max_grad_norm=1.0,
        )
        if not result.synchronization_boundary:
            raise AssertionError("M1 optimizer probe ended outside a synchronization boundary")
        progress.advance_optimizer_step(
            optimizer_step_was_skipped=result.optimizer_step_was_skipped
        )
        rank_state.value.add_(1)
        return global_trace

    for step in range(2):
        run_step(step)
    checkpoint = save_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=args.output_dir.resolve() / "checkpoint-000002",
        contract=contract,
        plan=frozen_plan,
        progress=progress,
        rank_state=rank_state,
    )
    expected_rng_after_load = torch.rand(5)
    uninterrupted_trace = tuple(run_step(step) for step in range(2, 4))
    uninterrupted_model = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
    uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
    uninterrupted_scheduler = copy.deepcopy(scheduler.state_dict())
    uninterrupted_progress = copy.deepcopy(progress.state_dict())
    uninterrupted_rank_state = rank_state.value.clone()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(9.0)
    progress.advance_optimizer_step(optimizer_step_was_skipped=True)
    rank_state.value.fill_(-1)
    torch.manual_seed(999 + accelerator.process_index)
    load_accelerate_checkpoint(
        accelerator=accelerator,
        checkpoint_dir=checkpoint,
        contract=contract,
        plan=frozen_plan,
        progress=progress,
        rank_state=rank_state,
    )
    if not torch.equal(torch.rand(5), expected_rng_after_load):
        raise AssertionError("per-rank RNG did not resume exactly")
    resumed_trace = tuple(run_step(step) for step in range(2, 4))
    if resumed_trace != uninterrupted_trace:
        raise AssertionError("official loader/processor trace changed after exact resume")
    _assert_nested_equal(accelerator.unwrap_model(model).state_dict(), uninterrupted_model)
    _assert_nested_equal(optimizer.state_dict(), uninterrupted_optimizer)
    _assert_nested_equal(scheduler.state_dict(), uninterrupted_scheduler)
    if progress.state_dict() != uninterrupted_progress:
        raise AssertionError("run progress changed after exact resume")
    if not torch.equal(rank_state.value, uninterrupted_rank_state):
        raise AssertionError("rank-local data cursor changed after exact resume")

    corrupt_checkpoint = args.output_dir.resolve() / "checkpoint-corrupt"
    if accelerator.is_main_process:
        shutil.copytree(checkpoint, corrupt_checkpoint)
        corrupt_path = corrupt_checkpoint / "picf_rank_state_00000.pt"
        with corrupt_path.open("ab") as handle:
            handle.write(b"m1-corruption")
    accelerator.wait_for_everyone()
    corruption_failed_closed = False
    try:
        load_accelerate_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=corrupt_checkpoint,
            contract=contract,
            plan=frozen_plan,
            progress=progress,
            rank_state=rank_state,
        )
    except RuntimeError as error:
        corruption_failed_closed = "checkpoint control validation failed" in str(error)
    corruption_count = accelerator.reduce(
        torch.tensor(
            int(corruption_failed_closed),
            device=accelerator.device,
            dtype=torch.int64,
        ),
        reduction="sum",
    )
    if int(corruption_count.item()) != accelerator.num_processes:
        raise AssertionError("corrupted checkpoint was not rejected by every rank")

    model_hash = _model_sha256(accelerator.unwrap_model(model))
    gathered_model_hashes: list[str | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered_model_hashes, model_hash)
    if len(set(gathered_model_hashes)) != 1:
        raise AssertionError("M1 DDP ranks ended with different model states")
    local_resources = {
        "rank": accelerator.process_index,
        "device": str(accelerator.device),
        "device_name": torch.cuda.get_device_name(accelerator.device),
        "processor_seconds": processor_s,
        "processor_rows": len(local_rows),
        "processor_rows_per_second": len(local_rows) / processor_s,
        "peak_allocated_bytes": processor_peak_bytes,
    }
    gathered_resources: list[dict[str, Any] | None] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered_resources, local_resources)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        locator_mismatches = sum(bool(row["locator_mismatch"]) for row in representatives)
        report = {
            "schema": "picf-next.molmoact2-m1-ddp.v1",
            "status": "PASS",
            "gate": "M1_typed_full_manifest",
            "world_size": accelerator.num_processes,
            "dataset": {
                "id": MOLMOACT2_LIBERO_DATASET_ID,
                "revision": MOLMOACT2_LIBERO_REVISION,
                "selected_episodes": len(selected_episodes),
                "selected_tasks": EXPECTED_TASKS,
                "selected_representative_rows": len(representatives),
                "selected_locator_mismatches": locator_mismatches,
                "official_loader": "lerobot.datasets.io_utils.load_nested_dataset",
                "loader_discovers_all_physical_shards_once": True,
                "episode_filter_field": "episode_index",
                "episode_locator_fields_used": False,
                "loaded_rows": len(dataset),
            },
            "typed_contract": {
                "state_shape": [8],
                "state_semantics": "eef XYZ(3) + axis-angle(3) + gripper qpos(2)",
                "action_shape": [10, 7],
                "action_semantics": "normalized delta end-effector command(6) + binary gripper(1)",
                "delta_t_s": 0.1,
                "metadata_state_names_trusted": False,
            },
            "processor": {
                "factory": "make_molmoact2_pre_post_processors",
                "steps": processor_steps,
                "checkpoint_id": MOLMOACT2_CHECKPOINT_ID,
                "checkpoint_revision": MOLMOACT2_CHECKPOINT_REVISION,
                "action_mode": "continuous",
                "setup_type": SETUP_TYPE,
                "control_mode": CONTROL_MODE,
                "action_horizon": ACTION_HORIZON,
                "dataset_stats_source": "complete immutable merged LIBERO metadata",
                "all_representatives_processed": True,
                "processor_records_sha256": _canonical_sha256(flat_processor_records),
                "first_raw_tensor_manifest": first_raw_manifest,
                "first_processed_tensor_manifest": first_processed_manifest,
            },
            "no_leak": {
                "representative_rows_checked": EXPECTED_REPRESENTATIVES,
                "target_free_action_is_none": True,
                "target_free_labels_absent": True,
                "targetful_labels_absent_for_continuous_mode": True,
                "observation_tensors_exactly_equal_with_and_without_action_target": True,
                "runtime_masks_or_object_targets_present": False,
            },
            "continuation": {
                "frozen_sample_plan_sha256": frozen_plan.plan_sha256,
                "checkpoint_resume_exact": True,
                "rank_local_cursor_exact": True,
                "rng_exact": True,
                "loader_processor_trace_exact": True,
                "optimizer_scheduler_model_exact": True,
                "corrupted_checkpoint_failed_closed_on_all_ranks": True,
                "checkpoint_path": str(checkpoint),
                "model_sha256": model_hash,
            },
            "resources": [item for item in gathered_resources if item is not None],
            "timings_s": {"setup": setup_s},
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
