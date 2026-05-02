from __future__ import annotations

import argparse
import json
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
    def __init__(
        self,
        *,
        host: str,
        port: int,
        save_anchor_debug: bool = False,
        anchor_debug_dir: str | Path | None = None,
        save_prediction_debug: bool = False,
        prediction_debug_dir: str | Path | None = None,
    ):
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self._needs_reset = True
        self._save_anchor_debug = bool(save_anchor_debug)
        self._anchor_debug_dir = Path(anchor_debug_dir).expanduser() if anchor_debug_dir is not None else None
        self._save_prediction_debug = bool(save_prediction_debug)
        self._prediction_debug_dir = Path(prediction_debug_dir).expanduser() if prediction_debug_dir is not None else None
        self._episode_index = -1
        self._step_index = 0
        self._anchor_jsonl = None
        self._anchor_writer = None
        self._prediction_jsonl = None
        self._prediction_writer = None
        self._pending_prediction_debug = None
        self._pending_prediction_goal = ""
        self._pending_prediction_step = -1
        if self._save_anchor_debug:
            if self._anchor_debug_dir is None:
                self._anchor_debug_dir = Path("/mnt/calvin_eval_logs/anchor_debug")
            self._anchor_debug_dir.mkdir(parents=True, exist_ok=True)
        if self._save_prediction_debug:
            if self._prediction_debug_dir is None:
                self._prediction_debug_dir = Path("/mnt/calvin_eval_logs/prediction_debug")
            self._prediction_debug_dir.mkdir(parents=True, exist_ok=True)

    def reset(self):
        self._needs_reset = True

    def close(self):
        if self._anchor_writer is not None:
            self._anchor_writer.release()
            self._anchor_writer = None
        if self._anchor_jsonl is not None:
            self._anchor_jsonl.close()
            self._anchor_jsonl = None
        if self._prediction_writer is not None:
            self._prediction_writer.release()
            self._prediction_writer = None
        if self._prediction_jsonl is not None:
            self._prediction_jsonl.close()
            self._prediction_jsonl = None

    def _start_anchor_episode(self, frame: np.ndarray) -> None:
        self.close()
        if not self._save_anchor_debug or self._anchor_debug_dir is None:
            return
        self._episode_index += 1
        self._step_index = 0
        self._anchor_jsonl = open(self._anchor_debug_dir / "anchor_debug.jsonl", "a", encoding="utf-8")
        try:
            import cv2

            h, w = int(frame.shape[0]), int(frame.shape[1])
            video_path = self._anchor_debug_dir / f"anchor_overlay_ep{self._episode_index:04d}.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
            self._anchor_writer = writer if writer.isOpened() else None
        except Exception as exc:
            print(f"WARNING unable to create anchor overlay video writer: {exc}")
            self._anchor_writer = None

    def _start_prediction_episode(self) -> None:
        if not self._save_prediction_debug or self._prediction_debug_dir is None:
            return
        self._prediction_jsonl = open(self._prediction_debug_dir / "prediction_compare.jsonl", "a", encoding="utf-8")
        try:
            import cv2

            video_path = self._prediction_debug_dir / f"prediction_compare_ep{self._episode_index:04d}.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (384, 128))
            self._prediction_writer = writer if writer.isOpened() else None
        except Exception as exc:
            print(f"WARNING unable to create prediction comparison video writer: {exc}")
            self._prediction_writer = None

    @staticmethod
    def _draw_anchor_points(frame: np.ndarray, anchor_debug: dict[str, Any]) -> np.ndarray:
        try:
            import cv2
        except Exception:
            return frame

        canvas = np.array(frame, copy=True)
        if canvas.ndim != 3 or canvas.shape[-1] != 3:
            return canvas
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        def _section(section: str):
            payload = anchor_debug.get(section, {}) if isinstance(anchor_debug, dict) else {}
            return payload if isinstance(payload, dict) else {}

        def _points(section: str):
            payload = _section(section)
            pixels = payload.get("pixel")
            return [] if pixels is None else pixels

        def _roles(section: str):
            payload = _section(section)
            roles = payload.get("role_ids") or payload.get("local_role_ids")
            return [] if roles is None else roles

        def _draw(points, color, radius=4, marker="circle", roles=None, role_colors=None):
            for idx, item in enumerate(points):
                if item is None or len(item) < 2:
                    continue
                x, y = int(round(float(item[0]))), int(round(float(item[1])))
                if x < 0 or y < 0 or x >= canvas.shape[1] or y >= canvas.shape[0]:
                    continue
                item_color = color
                if roles is not None and idx < len(roles) and role_colors is not None:
                    try:
                        item_color = role_colors.get(int(roles[idx]), color)
                    except Exception:
                        item_color = color
                if marker == "cross":
                    cv2.drawMarker(canvas, (x, y), item_color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                else:
                    cv2.circle(canvas, (x, y), radius, item_color, thickness=2)
                if idx < 24:
                    cv2.putText(canvas, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, item_color, 1, cv2.LINE_AA)

        _draw(
            _points("observation"),
            (0, 220, 255),
            radius=3,
            marker="circle",
            roles=_roles("observation"),
            role_colors={0: (0, 140, 255), 1: (0, 255, 255)},
        )
        _draw(
            _points("posterior"),
            (0, 0, 255),
            radius=5,
            marker="circle",
            roles=_roles("posterior"),
            role_colors={0: (180, 0, 255), 1: (0, 0, 255)},
        )
        _draw(
            _points("task"),
            (255, 255, 0),
            radius=4,
            marker="cross",
            roles=_roles("task"),
            role_colors={0: (255, 0, 255), 1: (255, 255, 0)},
        )
        cv2.putText(
            canvas,
            "obs local=orange obs object=yellow posterior local=purple posterior object=red task local=magenta task object=cyan",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _record_anchor_debug(self, frame: np.ndarray, goal: str, out: dict[str, Any]) -> None:
        if not self._save_anchor_debug or self._anchor_debug_dir is None:
            return
        anchor_debug = out.get("anchor_debug")
        if not isinstance(anchor_debug, dict):
            return
        if self._anchor_jsonl is not None:
            record = {
                "episode": int(self._episode_index),
                "step": int(self._step_index),
                "goal": goal,
                "anchor_debug": anchor_debug,
            }
            self._anchor_jsonl.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._anchor_jsonl.flush()
        if self._anchor_writer is not None:
            self._anchor_writer.write(self._draw_anchor_points(frame, anchor_debug))

    @staticmethod
    def _real_visual_grid(frame: np.ndarray, grid: int) -> np.ndarray:
        grid = max(int(grid), 1)
        try:
            import cv2

            return cv2.resize(np.asarray(frame, dtype=np.float32) / 255.0, (grid, grid), interpolation=cv2.INTER_AREA)
        except Exception:
            image = np.asarray(frame, dtype=np.float32) / 255.0
            ys = np.linspace(0, max(image.shape[0] - 1, 0), grid).astype(np.int64)
            xs = np.linspace(0, max(image.shape[1] - 1, 0), grid).astype(np.int64)
            return image[ys][:, xs]

    @staticmethod
    def _prediction_panel(real_grid: np.ndarray, prediction_debug: dict[str, Any]) -> tuple[np.ndarray, dict[str, float | None]]:
        import cv2

        def _grid(name: str) -> np.ndarray | None:
            value = prediction_debug.get(name)
            if value is None:
                return None
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 3 or array.shape[-1] != 3:
                return None
            return array

        physical = _grid("physical_visual_real")
        conditioned = _grid("conditioned_visual_real")
        metrics: dict[str, float | None] = {
            "l1_physical": None if physical is None else float(np.mean(np.abs(np.clip(physical, 0.0, 1.0) - real_grid))),
            "l1_conditioned": None
            if conditioned is None
            else float(np.mean(np.abs(np.clip(conditioned, 0.0, 1.0) - real_grid))),
        }

        def _tile(title: str, image: np.ndarray | None) -> np.ndarray:
            small = np.zeros_like(real_grid) if image is None else np.clip(image, 0.0, 1.0)
            tile = (small * 255.0).astype(np.uint8)
            tile = cv2.resize(tile, (128, 128), interpolation=cv2.INTER_NEAREST)
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            cv2.putText(tile, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            return tile

        return np.concatenate(
            [_tile("real_next", real_grid), _tile("phys_pred", physical), _tile("cond_pred", conditioned)],
            axis=1,
        ), metrics

    def _record_prediction_compare(self, frame: np.ndarray) -> None:
        if not self._save_prediction_debug or self._prediction_debug_dir is None:
            return
        if not isinstance(self._pending_prediction_debug, dict):
            return
        grid = int(self._pending_prediction_debug.get("visual_real_grid") or 4)
        real_grid = self._real_visual_grid(frame, grid)
        metrics: dict[str, float | None] = {"l1_physical": None, "l1_conditioned": None}
        if self._prediction_writer is not None:
            try:
                panel, metrics = self._prediction_panel(real_grid, self._pending_prediction_debug)
                self._prediction_writer.write(panel)
            except Exception as exc:
                print(f"WARNING unable to render prediction comparison panel: {exc}")
        if self._prediction_jsonl is not None:
            record = {
                "episode": int(self._episode_index),
                "step": int(self._step_index),
                "goal": self._pending_prediction_goal,
                "predicted_from_step": int(self._pending_prediction_step),
                "real_visual_grid": real_grid.tolist(),
                "prediction_debug": self._pending_prediction_debug,
                **metrics,
            }
            self._prediction_jsonl.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._prediction_jsonl.flush()

    def step(self, obs, goal):
        if isinstance(goal, bytes):
            goal = goal.decode("utf-8", "ignore")
        elif not isinstance(goal, str):
            goal = str(goal)
        if self._needs_reset and (self._save_anchor_debug or self._save_prediction_debug):
            rgb_obs = obs.get("rgb_obs", {})
            frame = np.asarray(rgb_obs.get("rgb_static"), dtype=np.uint8)
            if frame.ndim == 3:
                if self._save_anchor_debug:
                    self._start_anchor_episode(frame)
                else:
                    self.close()
                    self._episode_index += 1
                    self._step_index = 0
                self._pending_prediction_debug = None
                self._pending_prediction_goal = ""
                self._pending_prediction_step = -1
                self._start_prediction_episode()
        example = _build_policy_example(obs, goal, needs_reset=self._needs_reset)
        self._needs_reset = False
        current_step_index = int(self._step_index)
        if self._save_prediction_debug:
            frame = np.asarray(example["observation/image"], dtype=np.uint8)
            self._record_prediction_compare(frame)
        out = self.policy.infer(example)
        if self._save_anchor_debug:
            frame = np.asarray(example["observation/image"], dtype=np.uint8)
            self._record_anchor_debug(frame, goal, out)
        if self._save_prediction_debug:
            self._pending_prediction_debug = out.get("prediction_debug")
            self._pending_prediction_goal = goal
            self._pending_prediction_step = current_step_index
        if self._save_anchor_debug or self._save_prediction_debug:
            self._step_index += 1
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
    parser.add_argument("--save_anchor_debug", action="store_true")
    parser.add_argument("--anchor_debug_dir", default=os.environ.get("PICF_ANCHOR_DEBUG_DIR", "/mnt/calvin_eval_logs/anchor_debug"))
    parser.add_argument("--save_prediction_debug", action="store_true")
    parser.add_argument(
        "--prediction_debug_dir",
        default=os.environ.get("PICF_PREDICTION_DEBUG_DIR", "/mnt/calvin_eval_logs/prediction_debug"),
    )
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
    model = _PicfCalvinModel(
        host=args.server_host,
        port=args.server_port,
        save_anchor_debug=bool(args.save_anchor_debug),
        anchor_debug_dir=args.anchor_debug_dir,
        save_prediction_debug=bool(args.save_prediction_debug),
        prediction_debug_dir=args.prediction_debug_dir,
    )
    try:
        upstream_eval.evaluate_policy(
            model,
            env,
            epoch=str(args.epoch_tag),
            eval_log_dir=args.eval_log_dir,
            debug=bool(args.debug),
            create_plan_tsne=False,
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
