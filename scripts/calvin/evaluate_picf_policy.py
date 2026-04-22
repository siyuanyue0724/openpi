from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "packages" / "openpi-client" / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import numpy as np
import torch
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def _repo_root() -> Path:
    return _REPO_ROOT


def _build_policy_example(obs: dict[str, Any], goal: str, *, needs_reset: bool) -> dict[str, Any]:
    depth_obs = obs.get("depth_obs", {})
    rgb_obs = obs.get("rgb_obs", {})
    example = {
        "observation/image": np.asarray(rgb_obs["rgb_static"], dtype=np.uint8),
        "observation/wrist_image": np.asarray(rgb_obs["rgb_gripper"], dtype=np.uint8),
        "observation/depth": np.asarray(depth_obs["depth_static"], dtype=np.float32),
        "observation/state": np.asarray(obs["robot_obs"], dtype=np.float32),
        "prompt": goal,
        "openpi/reset": bool(needs_reset),
    }
    depth_gripper = depth_obs.get("depth_gripper")
    if depth_gripper is not None:
        example["observation/depth_gripper"] = np.asarray(depth_gripper, dtype=np.float32)
    return example


def _discretize_calvin_gripper(action: np.ndarray) -> np.ndarray:
    out = np.array(action, dtype=np.float32, copy=True).reshape(-1)
    out[-1] = 1.0 if out[-1] >= 0.0 else -1.0
    return out


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _patch_calvin_git_hash_lookup() -> None:
    try:
        import calvin_env.utils.utils as calvin_utils
    except Exception:
        return

    original = getattr(calvin_utils, "get_git_commit_hash", None)
    if not callable(original):
        return

    def _safe_get_git_commit_hash(repo_path):
        try:
            return original(repo_path)
        except Exception as exc:
            print(f"WARNING unable to resolve calvin_env git hash: {exc}")
            return "unknown"

    calvin_utils.get_git_commit_hash = _safe_get_git_commit_hash
    try:
        import calvin_env.envs.play_table_env as play_table_env

        play_table_env.get_git_commit_hash = _safe_get_git_commit_hash
    except Exception:
        pass


class _PicfCalvinModel:
    def __init__(self, *, host: str, port: int):
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self._needs_reset = True

    def reset(self):
        self._needs_reset = True

    def step(self, obs, goal):
        if isinstance(goal, bytes):
            goal = goal.decode("utf-8", "ignore")
        elif not isinstance(goal, str):
            goal = str(goal)
        example = _build_policy_example(obs, goal, needs_reset=self._needs_reset)
        self._needs_reset = False
        out = self.policy.infer(example)
        actions = np.asarray(out["actions"], dtype=np.float32)
        action = actions[0] if actions.ndim == 2 else actions.reshape(-1)
        return _discretize_calvin_gripper(action)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PICF websocket policy on CALVIN without editing upstream evaluator.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--eval_log_dir", required=True)
    parser.add_argument("--num_sequences", type=int, default=10)
    parser.add_argument("--server_host", default=os.environ.get("OPENPI_SERVER_HOST", "127.0.0.1"))
    parser.add_argument("--server_port", type=int, default=int(os.environ.get("OPENPI_SERVER_PORT", "8000")))
    parser.add_argument("--epoch_tag", default=os.environ.get("OPENPI_EVAL_TAG", "picf"))
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", default=os.environ.get("CALVIN_VIDEO_DIR", "/mnt/calvin_eval_logs/videos"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--calvin_agent_root", default="/mnt/calvin/calvin_models/calvin_agent")
    args = parser.parse_args()

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    calvin_root = Path(args.calvin_agent_root).expanduser()
    if str(calvin_root) not in sys.path:
        sys.path.insert(0, str(calvin_root))

    import evaluation.evaluate_policy as upstream_eval

    _seed_everything(0)
    _patch_calvin_git_hash_lookup()
    upstream_eval.NUM_SEQUENCES = int(args.num_sequences)
    if args.save_video:
        os.environ["CALVIN_SAVE_VIDEO"] = "1"
        os.environ["CALVIN_VIDEO_DIR"] = str(args.video_dir)
    else:
        os.environ.pop("CALVIN_SAVE_VIDEO", None)
        os.environ.pop("CALVIN_VIDEO_DIR", None)

    env = upstream_eval.make_env(args.dataset_path)
    model = _PicfCalvinModel(host=args.server_host, port=args.server_port)
    upstream_eval.evaluate_policy(
        model,
        env,
        epoch=str(args.epoch_tag),
        eval_log_dir=args.eval_log_dir,
        debug=bool(args.debug),
        create_plan_tsne=False,
    )


if __name__ == "__main__":
    main()
