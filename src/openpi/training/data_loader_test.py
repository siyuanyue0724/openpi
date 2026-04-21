import os

os.environ["JAX_PLATFORMS"] = "cpu"

import dataclasses

import jax
import pytest

from openpi.models import pi0_config
from openpi.picf.test_utils import build_mini_calvin_dataset
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)
    try:
        loader = _data_loader.create_data_loader(
            config,
            # Skip since we may not have the data available.
            skip_norm_stats=True,
            num_batches=2,
            shuffle=True,
        )
    except Exception as exc:
        pytest.skip(f"real dataset unavailable in test environment: {exc}")
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_calvin_loader_dir_backend_with_sonata_config(tmp_path):
    calvin_root = build_mini_calvin_dataset(tmp_path / "dir_case", make_zip=False)
    config = _config.get_config("pi05_calvin_sonata")
    config = dataclasses.replace(
        config,
        batch_size=1,
        num_workers=0,
        model=dataclasses.replace(config.model, action_horizon=4),
        data=dataclasses.replace(
            config.data,
            calvin_root=calvin_root,
            backend="dir",
            split="training",
            cameras_json_path=f"{calvin_root}/calib/cameras.json",
            max_points=128,
            stride=2,
        ),
    )

    loader = _data_loader.create_data_loader(
        config,
        framework="pytorch",
        skip_norm_stats=True,
        num_batches=1,
    )
    observation, actions = next(iter(loader))

    assert observation.state.shape[0] == 1
    assert "pointcloud" in observation.point_clouds
    assert observation.point_clouds["pointcloud"].shape == (1, 128, 9)
    assert actions.shape == (1, config.model.action_horizon, config.model.action_dim)


def test_create_calvin_loader_zip_backend_with_sonata_config(tmp_path):
    calvin_root = build_mini_calvin_dataset(tmp_path / "zip_case", make_zip=True)
    config = _config.get_config("pi05_calvin_sonata")
    config = dataclasses.replace(
        config,
        batch_size=1,
        num_workers=0,
        model=dataclasses.replace(config.model, action_horizon=4),
        data=dataclasses.replace(
            config.data,
            calvin_root=calvin_root,
            backend="zip",
            split="training",
            cameras_json_path=None,
            max_points=128,
            stride=2,
        ),
    )

    loader = _data_loader.create_data_loader(
        config,
        framework="pytorch",
        skip_norm_stats=True,
        num_batches=1,
    )
    observation, actions = next(iter(loader))

    assert observation.state.shape[0] == 1
    assert "pointcloud" in observation.point_clouds
    assert observation.point_clouds["pointcloud"].shape == (1, 128, 9)
    assert actions.shape == (1, config.model.action_horizon, config.model.action_dim)
