from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_native_wheel_extra_pins_the_official_lingbot_torch_runtime() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert project["project"]["optional-dependencies"]["lingbot-native"] == [
        "omegaconf==2.3.0",
        "torch==2.8.0",
    ]


def test_native_import_does_not_eagerly_load_historical_posterior_runtime() -> None:
    program = """
import json
import sys
import picf_next.lingbot_native.calvin
forbidden = [
    'picf_next.association',
    'picf_next.geometry',
    'picf_next.posterior',
]
print(json.dumps([name for name in forbidden if name in sys.modules]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_legacy_root_exports_remain_lazy_and_compatible() -> None:
    program = """
import json
import sys
import picf_next
before = 'picf_next.association' in sys.modules
value = picf_next.AssociationResult
after = 'picf_next.association' in sys.modules
print(json.dumps({'before': before, 'after': after, 'name': value.__name__}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {"before": False, "after": True, "name": "AssociationResult"}
