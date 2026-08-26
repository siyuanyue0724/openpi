# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from picf_next.training.stationary_stage import load_stationary_temporal_stage_recipe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/molmoact2_calvin_m3_stationary_temporal.json"


def test_stationary_stage_derives_only_action_free_structural_objective() -> None:
    stage = load_stationary_temporal_stage_recipe(CONFIG)
    historical = stage.load_foundation(ROOT)
    structural = stage.structural_foundation(ROOT)

    assert structural.objective_config.action_weight == 0.0
    assert historical.objective_config.action_weight == 1.0
    assert structural.core_config == historical.core_config
    assert structural.set_loss_config == historical.set_loss_config
    assert structural.dynamics_loss_config == historical.dynamics_loss_config
    assert structural.binding_loss_config == historical.binding_loss_config
    assert structural.geometry_overshooting == historical.geometry_overshooting
    assert structural.objective_config.set_weight == historical.objective_config.set_weight
    assert (
        structural.objective_config.dynamics_weight == historical.objective_config.dynamics_weight
    )
    assert structural.objective_config.binding_weight == historical.objective_config.binding_weight
    assert stage.load_source_coverage(ROOT).split.train_ranges == ((358482, 360282),)


def test_stationary_stage_optimizer_is_bounded_and_decays_to_declared_floor() -> None:
    stage = load_stationary_temporal_stage_recipe(CONFIG)
    module = torch.nn.Linear(2, 2)
    optimizer, scheduler = stage.build_optimizer_and_scheduler(module)
    observed = []
    for _step in range(stage.optimizer.optimizer_steps):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert observed[0] == pytest.approx(stage.optimizer.learning_rate / 10.0)
    assert max(observed) == pytest.approx(stage.optimizer.learning_rate)
    assert observed[-1] > stage.optimizer.minimum_learning_rate
    assert scheduler.get_last_lr()[0] == pytest.approx(
        stage.optimizer.minimum_learning_rate,
        rel=5e-4,
    )


def test_stationary_stage_rejects_unreviewed_memory_schedule(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["clip"]["prefix_lengths"] = [0, 8, 64]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="preregistered"):
        load_stationary_temporal_stage_recipe(path)
