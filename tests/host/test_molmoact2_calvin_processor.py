from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo.hf_model.modeling_molmoact2")
configs = pytest.importorskip("lerobot.configs")
processor_module = pytest.importorskip("lerobot.policies.molmoact2.processor_molmoact2")
lerobot_processor = pytest.importorskip("lerobot.processor")
constants = pytest.importorskip("lerobot.utils.constants")
calvin_module = pytest.importorskip("picf_next.data.calvin")
bridge_module = pytest.importorskip("picf_next.hosts.molmoact2_calvin_processor")
training_module = pytest.importorskip("picf_next.hosts.molmoact2_training")
calvin_test_module = pytest.importorskip("tests.test_calvin_data")

FeatureType = configs.FeatureType
NormalizationMode = configs.NormalizationMode
PolicyFeature = configs.PolicyFeature
MolmoAct2ClampNormalizedProcessorStep = processor_module.MolmoAct2ClampNormalizedProcessorStep
MolmoAct2MaskedNormalizerProcessorStep = processor_module.MolmoAct2MaskedNormalizerProcessorStep
MolmoAct2PackInputsProcessorStep = processor_module.MolmoAct2PackInputsProcessorStep
AddBatchDimensionProcessorStep = lerobot_processor.AddBatchDimensionProcessorStep
DeviceProcessorStep = lerobot_processor.DeviceProcessorStep
RenameObservationsProcessorStep = lerobot_processor.RenameObservationsProcessorStep
ACTION = constants.ACTION
OBS_STATE = constants.OBS_STATE
CalvinDatasetIndex = calvin_module.CalvinDatasetIndex
CalvinStatefulTransitionDataset = calvin_module.CalvinStatefulTransitionDataset
CalvinMolmoAct2ProcessorBridge = bridge_module.CalvinMolmoAct2ProcessorBridge
MolmoAct2HostObservationView = training_module.MolmoAct2HostObservationView
_write_split = calvin_test_module._write_split
_split_manifest = calvin_test_module._split_manifest


class _TargetFreeFakeOfficialPipeline:
    def __init__(self, steps: tuple[object, ...]) -> None:
        self.steps = steps
        self.calls = 0
        self.last_batch_keys: set[str] = set()

    def __call__(self, batch):
        self.calls += 1
        self.last_batch_keys = set(batch)
        assert "action" not in batch
        batch_size = int(batch[OBS_STATE].shape[0])
        return {
            "input_ids": torch.ones(batch_size, 5, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.bool),
            "pixel_values": batch["observation.images.image"].float(),
            "image_token_pooling": torch.zeros(batch_size, 1, 1, dtype=torch.long),
            "image_grids": torch.ones(batch_size, 2, dtype=torch.long),
            "image_num_crops": torch.ones(batch_size, dtype=torch.long),
            "action": None,
            OBS_STATE: batch[OBS_STATE],
            "task": batch.get("task"),
        }


def _processor(*, chunk_size: int = 4) -> _TargetFreeFakeOfficialPipeline:
    normalizer = MolmoAct2MaskedNormalizerProcessorStep(
        features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        norm_map={
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        },
        stats={},
    )
    packer = object.__new__(MolmoAct2PackInputsProcessorStep)
    packer.action_mode = "continuous"
    packer.chunk_size = chunk_size
    packer.env_action_dim = 7
    packer.max_action_dim = 32
    steps = (
        RenameObservationsProcessorStep(),
        AddBatchDimensionProcessorStep(),
        normalizer,
        MolmoAct2ClampNormalizedProcessorStep(),
        packer,
        DeviceProcessorStep(device="cpu"),
    )
    return _TargetFreeFakeOfficialPipeline(steps)


def _dataset(tmp_path: Path) -> CalvinStatefulTransitionDataset:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    return CalvinStatefulTransitionDataset(index, action_horizon=4)


def _view(sample) -> MolmoAct2HostObservationView:
    record = sample.record
    return MolmoAct2HostObservationView(
        task=record.task,
        embodiment=record.embodiment,
        control_mode=record.control_mode,
        control_frame=record.control_frame,
        state_axes=record.state_axes,
        state_units=record.state_units,
        state=tuple(float(value) for value in record.state),
        state_valid=tuple(bool(value) for value in record.state_valid),
        timestamp_s=record.timestamp_s,
        delta_t_s=record.delta_t_s,
    )


def test_processor_bridge_separates_target_free_observation_and_action_only_paths(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    samples = (dataset[0], dataset[1])
    pipeline = _processor()
    bridge = CalvinMolmoAct2ProcessorBridge(pipeline, action_horizon=4)

    observation = bridge.build_observation_inputs(
        tuple((sample.picf_evidence_frame,) for sample in samples),
        tuple(_view(sample) for sample in samples),
    )
    assert pipeline.calls == 1
    assert set(observation) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
    }
    assert "action" not in observation
    assert OBS_STATE not in observation

    targets = bridge.build_action_targets(samples)
    assert pipeline.calls == 1
    assert set(targets) == {"action", "action_dim_is_pad", "action_horizon_is_pad"}
    assert targets["action"].shape == (2, 4, 32)
    torch.testing.assert_close(
        targets["action"][..., :7],
        torch.from_numpy(np.stack([sample.host_sample.action for sample in samples])),
    )
    assert targets["action_dim_is_pad"][:, :7].sum() == 0
    assert targets["action_dim_is_pad"][:, 7:].all()
    assert torch.equal(
        targets["action_horizon_is_pad"],
        torch.from_numpy(np.stack([sample.host_sample.action_is_pad for sample in samples])),
    )


def test_processor_bridge_accepts_task_independent_source_frames(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    observations = tuple(
        dataset.index.molmoact2_source_observation(global_index) for global_index in (10, 17)
    )
    pipeline = _processor()
    bridge = CalvinMolmoAct2ProcessorBridge(pipeline, action_horizon=4)

    inputs = bridge.build_source_observation_inputs(observations)

    assert pipeline.calls == 1
    assert "task" not in pipeline.last_batch_keys
    assert set(inputs) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_token_pooling",
        "image_grids",
        "image_num_crops",
    }
    assert inputs["input_ids"].shape[0] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda pipeline: setattr(pipeline.steps[4], "action_mode", "both"), "continuous"),
        (lambda pipeline: setattr(pipeline.steps[4], "env_action_dim", 8), "seven CALVIN"),
        (lambda pipeline: setattr(pipeline.steps[4], "chunk_size", 3), "chunk size"),
        (
            lambda pipeline: setattr(
                pipeline,
                "steps",
                (*pipeline.steps[:-1], SimpleNamespace()),
            ),
            "steps differ",
        ),
    ],
)
def test_processor_bridge_fails_closed_on_official_pipeline_drift(mutation, message: str) -> None:
    pipeline = _processor()
    mutation(pipeline)
    with pytest.raises(ValueError, match=message):
        CalvinMolmoAct2ProcessorBridge(pipeline, action_horizon=4)
