from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_builder():
    path = Path(__file__).parents[1] / "tools" / "build_calvin_golden.py"
    spec = importlib.util.spec_from_file_location("picf_calvin_golden_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_builder = _load_builder()
Segment = _builder.Segment
_phase_steps = _builder._phase_steps
_scene_instance_map = _builder._scene_instance_map


def test_scene_inventory_uses_physical_scene_only_and_excludes_base_link() -> None:
    scene_info = {
        "movable_objects": {"block_red": {"uid": 2}},
        "fixed_objects": {
            "table": {
                "uid": 5,
                "links": {"base": -1, "button_link": 0, "switch_link": 1},
            }
        },
    }

    assert tuple(inspect.signature(_scene_instance_map).parameters) == ("scene_info",)
    assert _scene_instance_map(scene_info) == {
        2: "movable/block_red",
        16777221: "part/table/button_link",
        33554437: "part/table/switch_link",
    }


def test_phase_sampling_ignores_task_and_prompt_content() -> None:
    first = Segment(index=0, start=10, end=20, task="task_a", prompt="prompt_a")
    second = Segment(index=0, start=10, end=20, task="task_b", prompt="prompt_b")

    expected = [("start", 10), ("mid", 15), ("end", 20)]
    assert _phase_steps(first, ("start", "mid", "end")) == expected
    assert _phase_steps(second, ("start", "mid", "end")) == expected


def test_environment_builder_uses_shared_pinned_factory(monkeypatch) -> None:
    environment = object()
    pybullet = SimpleNamespace(ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX=1)
    calls: list[tuple[Path, str, bool]] = []

    def build(root: Path, *, scene: str, include_cameras: bool):
        calls.append((root, scene, include_cameras))
        return environment

    monkeypatch.setattr(_builder, "build_calvin_geometry_environment", build)
    monkeypatch.setitem(sys.modules, "pybullet", pybullet)
    root = Path("/tmp/pinned-calvin-env")

    assert _builder._build_environment(root) == (environment, pybullet)
    assert calls == [(root, "calvin_scene_D", True)]
