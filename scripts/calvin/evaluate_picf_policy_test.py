from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_picf_policy as sut


def test_build_policy_example_includes_depth_gripper_and_reset() -> None:
    obs = {
        "rgb_obs": {
            "rgb_static": np.zeros((2, 2, 3), dtype=np.uint8),
            "rgb_gripper": np.ones((2, 2, 3), dtype=np.uint8),
        },
        "depth_obs": {
            "depth_static": np.zeros((2, 2), dtype=np.float32),
            "depth_gripper": np.ones((2, 2), dtype=np.float32),
        },
        "robot_obs": np.arange(7, dtype=np.float32),
    }
    payload = sut._build_policy_example(obs, "open drawer", needs_reset=True)
    assert payload["openpi/reset"] is True
    assert payload["prompt"] == "open drawer"
    assert "observation/depth_gripper" in payload


def test_discretize_calvin_gripper_maps_last_dimension_to_binary() -> None:
    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.01], dtype=np.float32)
    discrete = sut._discretize_calvin_gripper(action)
    assert discrete.shape == (7,)
    assert discrete[-1] == -1.0


def test_main_runs_without_pytorch_lightning_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    fake_eval = types.ModuleType("evaluation.evaluate_policy")

    def make_env(dataset_path: str):
        calls["dataset_path"] = dataset_path
        return "ENV"

    def evaluate_policy(model, env, *, epoch, eval_log_dir, debug, create_plan_tsne):
        calls["model"] = model
        calls["env"] = env
        calls["epoch"] = epoch
        calls["eval_log_dir"] = eval_log_dir
        calls["debug"] = debug
        calls["create_plan_tsne"] = create_plan_tsne

    fake_eval.NUM_SEQUENCES = None
    fake_eval.make_env = make_env
    fake_eval.evaluate_policy = evaluate_policy
    fake_pkg = types.ModuleType("evaluation")
    fake_pkg.evaluate_policy = fake_eval

    monkeypatch.setitem(sys.modules, "evaluation", fake_pkg)
    monkeypatch.setitem(sys.modules, "evaluation.evaluate_policy", fake_eval)

    class _DummyModel:
        def __init__(self, *, host: str, port: int, **kwargs):
            self.host = host
            self.port = port
            self.kwargs = kwargs
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(sut, "_PicfCalvinModel", _DummyModel)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_picf_policy.py",
            "--dataset_path",
            str(tmp_path / "dataset"),
            "--eval_log_dir",
            str(tmp_path / "eval"),
            "--num_sequences",
            "20",
            "--epoch_tag",
            "step5000",
            "--save_video",
            "--video_dir",
            str(tmp_path / "videos"),
            "--server_host",
            "127.0.0.1",
            "--server_port",
            "8000",
            "--calvin_agent_root",
            str(tmp_path / "calvin_agent"),
        ],
    )

    sut.main()

    assert fake_eval.NUM_SEQUENCES == 20
    assert calls["dataset_path"] == str(tmp_path / "dataset")
    assert calls["env"] == "ENV"
    assert calls["epoch"] == "step5000"
    assert calls["eval_log_dir"] == str(tmp_path / "eval")
    assert calls["debug"] is False
    assert calls["create_plan_tsne"] is False
    assert calls["model"].kwargs["save_anchor_debug"] is False
    assert calls["model"].kwargs["save_prediction_debug"] is False
    assert calls["model"].closed is True
