#!/usr/bin/env python3
"""Install or validate the audited LingBot/LeRobot compatibility runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

LINGBOT_REQUIREMENTS_SHA256 = "4bea8eca2e5e81107332947fe38d9a2787bc6a8fe4d3f875fa7e3d028f48993d"
RUNTIME_VERSIONS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "transformers": "4.57.3",
    "datasets": "4.1.1",
    "huggingface-hub": "0.34.3",
    "lerobot": "0.4.3",
}


def runtime_install_commands(
    *, python: Path, source_checkout: Path, uv_command: str
) -> list[list[str]]:
    """Return the only accepted install order for the compatibility overlay."""

    return [
        [
            uv_command,
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(source_checkout / "requirements.txt"),
        ],
        [
            uv_command,
            "pip",
            "install",
            "--python",
            str(python),
            f"datasets=={RUNTIME_VERSIONS['datasets']}",
        ],
        [
            uv_command,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            f"lerobot=={RUNTIME_VERSIONS['lerobot']}",
        ],
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text_durable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _run_json(python: Path, source_checkout: Path, program: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_checkout.resolve())
    completed = subprocess.run(
        [str(python), "-c", program],
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"runtime probe failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"runtime probe produced no JSON: {completed.stdout}")
    return json.loads(lines[-1])


def validate_runtime(python: Path, source_checkout: Path) -> dict[str, Any]:
    requirements = source_checkout / "requirements.txt"
    if not requirements.is_file() or _sha256(requirements) != LINGBOT_REQUIREMENTS_SHA256:
        raise ValueError("LingBot requirements.txt differs from the audited source")
    report = _run_json(
        python,
        source_checkout,
        """
import importlib.metadata as metadata
import json
import sys
versions = {name: metadata.version(name) for name in (
    'torch', 'torchvision', 'transformers', 'datasets', 'huggingface-hub', 'lerobot'
)}
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
from lingbotvla.data.vla_data.utils import _as_feature_mapping
from lerobot.datasets.utils import load_nested_dataset
if _as_feature_mapping({'x': 1}) != {'x': 1}:
    raise RuntimeError('LingBot mapping parser rejected a native mapping')
if _as_feature_mapping("{'y': 2}") != {'y': 2}:
    raise RuntimeError('LingBot mapping parser rejected a serialized mapping')
print(json.dumps({
    'python_major_minor': list(sys.version_info[:2]),
    'versions': versions,
    'policy': LingbotVlaV2Policy.__name__,
    'lerobot_nested_loader': load_nested_dataset.__name__,
    'yaml_mapping_parser': 'PASS',
}))
""",
    )
    if report.get("python_major_minor") != [3, 12]:
        raise RuntimeError("LingBot compatibility runtime requires Python 3.12")
    if report.get("versions") != RUNTIME_VERSIONS:
        raise RuntimeError(f"runtime versions differ: {report.get('versions')}")
    return report


def validate_real_loader(python: Path, source_checkout: Path, dataset_root: Path) -> dict[str, Any]:
    if not (dataset_root / "data/chunk-000/file-146.parquet").is_file():
        raise ValueError("full pinned LIBERO dataset is absent")
    escaped = json.dumps(str((dataset_root / "data").resolve()))
    return _run_json(
        python,
        source_checkout,
        f"""
import json
from pathlib import Path
from lerobot.datasets.utils import load_nested_dataset
dataset = load_nested_dataset(Path({escaped}), episodes=[378, 379])
episodes = sorted(int(value) for value in dataset.unique('episode_index'))
state_widths = sorted({{len(row) for row in dataset['observation.state']}})
action_widths = sorted({{len(row) for row in dataset['action']}})
report = {{
    'rows': len(dataset),
    'episodes': episodes,
    'state_widths': state_widths,
    'action_widths': action_widths,
    'crossed_storage_variants': ['file-146.parquet', 'file-147.parquet'],
}}
expected = {{
    'rows': 364,
    'episodes': [378, 379],
    'state_widths': [8],
    'action_widths': [7],
}}
changed = {{key: (report[key], value) for key, value in expected.items() if report[key] != value}}
if changed:
    raise RuntimeError(f'LingBot real-loader contract differs: {{changed}}')
print(json.dumps(report))
""",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--uv-command", default="uv")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Resolving a venv launcher follows its symlink to the base interpreter and
    # silently drops the selected environment's site-packages.
    python = args.python.absolute()
    source = args.source_checkout.resolve()
    if not python.is_file() or not (source / ".git").exists():
        raise FileNotFoundError("runtime Python or LingBot source checkout is absent")
    if args.install:
        for command in runtime_install_commands(
            python=python,
            source_checkout=source,
            uv_command=args.uv_command,
        ):
            subprocess.run(command, check=True)
    report: dict[str, Any] = {
        "schema": "picf-next.lingbot-runtime.v1",
        "status": "PASS",
        "runtime": validate_runtime(python, source),
        "requirements_sha256": LINGBOT_REQUIREMENTS_SHA256,
        "lerobot_install_mode": "no-deps",
        "compatibility_exception": (
            "LeRobot 0.4.3 package metadata excludes Torch 2.8; only its audited "
            "dataset API is used with LingBot's pinned compute stack"
        ),
    }
    if args.dataset_root is not None:
        report["real_loader"] = validate_real_loader(python, source, args.dataset_root.resolve())
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        _write_text_durable(args.json_out, encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
