from __future__ import annotations

import argparse
import json
import logging
import socket
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "packages" / "openpi-client" / "src"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import numpy as np
import torch
from openpi_client import base_policy as _base_policy

import picf_core_train as _trainer
from openpi.picf.contracts import PicfObservation
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def _as_sensor_names_arg(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    raise TypeError(f"Unsupported tactile_sensor_names payload: {type(value).__name__}")


def _as_sensor_offsets_arg(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        blocks: list[str] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise TypeError(f"Unsupported tactile_sensor_offsets_m item: {item!r}")
            blocks.append(",".join(str(float(component)) for component in item))
        return ";".join(blocks)
    raise TypeError(f"Unsupported tactile_sensor_offsets_m payload: {type(value).__name__}")


def _resolve_checkpoint_dir(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.name == "latest.pt":
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        checkpoint_dir = Path(payload["checkpoint_dir"]).expanduser()
        return checkpoint_dir.parent, checkpoint_dir
    if candidate.is_dir() and (candidate / "model.pt").is_file() and (candidate / "metadata.pt").is_file():
        return candidate.parent, candidate
    if candidate.is_dir() and (candidate / "latest.pt").is_file():
        payload = torch.load(candidate / "latest.pt", map_location="cpu", weights_only=False)
        checkpoint_dir = Path(payload["checkpoint_dir"]).expanduser()
        return candidate, checkpoint_dir
    raise FileNotFoundError(
        f"Could not resolve PICF checkpoint from {candidate}. Expected a step dir with model.pt/metadata.pt "
        "or an output dir containing latest.pt."
    )


def _load_runtime_args(checkpoint_dir: Path) -> argparse.Namespace:
    metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
    args_dict = dict(metadata["args"])
    if "tactile_sensor_names" in args_dict:
        args_dict["tactile_sensor_names"] = _as_sensor_names_arg(args_dict["tactile_sensor_names"])
    if "tactile_sensor_offsets_m" in args_dict:
        args_dict["tactile_sensor_offsets_m"] = _as_sensor_offsets_arg(args_dict["tactile_sensor_offsets_m"])
    args = argparse.Namespace(**args_dict)
    _trainer._normalize_train_args(args)
    _trainer._validate_train_args(args)
    _trainer._validate_backbone_args(args)
    return args


def _load_model_state_only(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> int:
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    model_state = torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=False)
    metadata = torch.load(checkpoint_dir / "metadata.pt", map_location="cpu", weights_only=False)
    try:
        module.load_state_dict(model_state, strict=True)
    except RuntimeError:
        try:
            _trainer._load_state_dict_picf_compat(module, model_state)
        except RuntimeError:
            try:
                module.core.load_state_dict(model_state, strict=True)
            except RuntimeError:
                _trainer._load_state_dict_picf_compat(module.core, model_state)
    return int(metadata.get("step", 0))


def _visual_override_if_needed(
    trainer: _trainer._PicfWindowTrainer,
    observation: PicfObservation,
) -> torch.Tensor | np.ndarray | None:
    if not trainer.use_visual_override:
        return None
    return _trainer._rgb_visual_override(observation.rgb_static, grid=trainer.visual_grid)


class _PicfCheckpointPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        trainer: _trainer._PicfWindowTrainer,
        *,
        checkpoint_dir: Path,
        checkpoint_step: int,
        frame_dt_s: float = 1.0 / 30.0,
    ) -> None:
        self._trainer = trainer.eval()
        self._core = trainer.core
        self._semantic_encoder = trainer.semantic_encoder
        self._frame_dt_s = float(frame_dt_s)
        self._segment_id = 0
        self._step_id = 0
        self._previous: Any | None = None
        self._last_prompt = ""
        self._metadata = {
            "checkpoint_format": "picf_trainer_v2",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": int(checkpoint_step),
            "action_dim": 7,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _reset_episode(self) -> None:
        self._segment_id += 1
        self._step_id = 0
        self._previous = None

    def _build_observation(self, obs: dict[str, Any], *, reset: bool) -> PicfObservation:
        prompt = str(obs.get("prompt", self._last_prompt))
        self._last_prompt = prompt
        return PicfObservation(
            rgb_static=np.asarray(obs["observation/image"], dtype=np.uint8),
            depth_static=np.asarray(obs["observation/depth"], dtype=np.float32),
            rgb_gripper=None
            if "observation/wrist_image" not in obs or obs["observation/wrist_image"] is None
            else np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
            depth_gripper=None
            if "observation/depth_gripper" not in obs or obs["observation/depth_gripper"] is None
            else np.asarray(obs["observation/depth_gripper"], dtype=np.float32),
            robot_obs=np.asarray(obs["observation/state"], dtype=np.float32),
            prompt=prompt,
            step_id=int(self._step_id),
            segment_id=int(self._segment_id),
            timestamp_s=float(self._step_id) * self._frame_dt_s,
            reset_scaffold=bool(reset),
            proprio=np.asarray(obs["observation/state"], dtype=np.float32),
        )

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        reset = bool(obs.get("openpi/reset", False))
        if reset:
            self._reset_episode()
        observation = self._build_observation(obs, reset=reset)
        semantic_override = None
        if self._semantic_encoder is not None:
            with torch.inference_mode():
                semantic_override = self._semantic_encoder.encode_observation(observation)
        visual_override = _visual_override_if_needed(self._trainer, observation)
        with torch.inference_mode():
            output = self._core.step(
                observation,
                previous=self._previous,
                visual_map_override=visual_override,
                semantic_override=semantic_override,
                action_future=None,
            )
        self._previous = output.state
        action = output.state.predictive.action.detach().to(device="cpu", dtype=torch.float32).numpy()
        self._step_id += 1
        return {
            "actions": action[None, :],
            "debug": output.debug,
        }


def _build_policy(*, checkpoint_path: Path, device: torch.device) -> _PicfCheckpointPolicy:
    output_dir, checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)
    args = _load_runtime_args(checkpoint_dir)
    args.device = str(device)
    core, semantic_encoder, use_visual_override = _trainer._build_model(args, device=device)
    trainer = _trainer._PicfWindowTrainer(
        core,
        semantic_encoder=semantic_encoder,
        visual_grid=int(args.visual_grid),
        use_visual_override=use_visual_override,
        loss_config=_trainer._build_loss_config(args),
    ).to(device)
    backgrounds = _trainer._load_tactile_backgrounds_npz(getattr(args, "tactile_backgrounds_path", None))
    source = _trainer._CalvinTransitionSource(
        args.calvin_root,
        split=args.split,
        backend=args.backend,
        unroll_steps=args.unroll_steps,
        use_wrist_rgb=True,
        use_tactile=bool(args.tactile_mode == "encoder"),
        tactile_sensor_names=args.tactile_sensor_names,
        tactile_sensor_offsets_m=args.tactile_sensor_offsets_m,
        tactile_calibration=getattr(args, "tactile_calibration_path", None),
        tactile_backgrounds_by_sensor=backgrounds,
        use_scene_obs=True,
    )
    try:
        _trainer._materialize_model_parameters(trainer, source=source, rank=0)
    finally:
        source.close()
    checkpoint_step = _load_model_state_only(checkpoint_dir=checkpoint_dir, model=trainer, device=device)
    return _PicfCheckpointPolicy(
        trainer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=checkpoint_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a PICF trainer checkpoint over websocket.")
    parser.add_argument("--checkpoint", required=True, help="PICF checkpoint dir or output dir containing latest.pt.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    policy = _build_policy(checkpoint_path=Path(args.checkpoint), device=device)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating PICF server (host: %s, ip: %s)", hostname, local_ip)
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
