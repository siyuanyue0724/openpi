from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo.hf_model.modeling_molmoact2")
contracts = pytest.importorskip("picf_next.contracts")
calvin = pytest.importorskip("picf_next.data.calvin")
assembly = pytest.importorskip("picf_next.training.molmoact2_calvin")
recipe_module = pytest.importorskip("picf_next.training.recipe")
ContractError = contracts.ContractError
CalvinStatefulEpisodeManifest = calvin.CalvinStatefulEpisodeManifest
MolmoAct2CalvinTrainingStack = assembly.MolmoAct2CalvinTrainingStack
assert_molmoact2_policy_config = assembly.assert_molmoact2_policy_config
build_calvin_episode_stream_plan = assembly.build_calvin_episode_stream_plan
build_molmoact2_optimizer_and_scheduler = assembly.build_molmoact2_optimizer_and_scheduler
build_molmoact2_policy_config = assembly.build_molmoact2_policy_config
move_core_with_fp32_parameter_storage = assembly._move_core_with_fp32_parameter_storage
validate_action_only_recipe = assembly._validate_action_only_recipe
validate_trainable_scope = assembly._validate_trainable_scope
load_training_recipe = recipe_module.load_training_recipe

_ROOT = Path(__file__).resolve().parents[1]
_RECIPE = _ROOT / "configs/training/molmoact2_calvin_m3_probe.json"
_ACTION_RECIPE = _ROOT / "configs/training/molmoact2_calvin_m4_action_adoption.json"


def test_policy_config_is_fully_derived_from_the_strict_recipe(tmp_path: Path) -> None:
    recipe = load_training_recipe(_RECIPE)
    config = build_molmoact2_policy_config(recipe, checkpoint_path=tmp_path / "checkpoint")

    assert config.chunk_size == 10
    assert config.n_action_steps == 10
    assert config.num_flow_timesteps == 8
    assert config.scheduler_warmup_steps == 10
    assert config.scheduler_decay_steps == 200
    assert config.train_action_expert_only is True
    assert config.gradient_checkpointing is True
    assert config.input_features["observation.state"].shape == (15,)
    assert config.output_features["action"].shape == (7,)

    config.num_flow_timesteps = 7
    with pytest.raises(ContractError, match="policy config differs"):
        assert_molmoact2_policy_config(recipe, config)


def test_episode_stream_plan_is_recipe_bounded_and_deterministic() -> None:
    recipe = load_training_recipe(_RECIPE)
    dataset = SimpleNamespace(
        action_horizon=10,
        episode_manifest=(
            CalvinStatefulEpisodeManifest(0, "episode-a", ("a0", "a1")),
            CalvinStatefulEpisodeManifest(1, "episode-b", ("b0", "b1", "b2")),
        ),
    )
    first = build_calvin_episode_stream_plan(
        recipe,
        dataset,
        comparison_id="assembly-test",
        seed=29,
        global_batch_size=2,
        total_steps=200,
    )
    second = build_calvin_episode_stream_plan(
        recipe,
        dataset,
        comparison_id="assembly-test",
        seed=29,
        global_batch_size=2,
        total_steps=200,
    )

    assert first.plan_sha256 == second.plan_sha256
    assert first.global_batch(0) == second.global_batch(0)
    with pytest.raises(PermissionError, match="at most 200"):
        build_calvin_episode_stream_plan(
            recipe,
            dataset,
            comparison_id="assembly-test",
            seed=29,
            global_batch_size=2,
            total_steps=201,
        )


class _OptimizerSurface(torch.nn.Module):
    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.ones(()))
        self.learning_rate = learning_rate

    def get_optim_params(self):
        return [{"params": [self.parameter], "lr": self.learning_rate}]


class _TrainableScopePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_expert = torch.nn.Linear(2, 2)
        self.action_layer_adapter = torch.nn.Linear(2, 2)


class _TrainableScopeCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discovery = torch.nn.Module()
        self.discovery.variance_head = torch.nn.Linear(2, 2)
        self.other = torch.nn.Linear(2, 2)
        self.discovery.variance_head.weight.requires_grad_(False)


def test_action_trainable_scope_requires_the_complete_picf_core_to_be_frozen() -> None:
    policy = _TrainableScopePolicy()
    core = _TrainableScopeCore()
    core.requires_grad_(False)

    validate_trainable_scope(policy, core)

    core.other.weight.requires_grad_(True)
    with pytest.raises(ContractError, match="core must be stationary"):
        validate_trainable_scope(policy, core)

    core.other.weight.requires_grad_(False)
    policy.other = torch.nn.Linear(2, 2)
    with pytest.raises(ContractError, match="escaped action expert"):
        validate_trainable_scope(policy, core)


def test_m4_recipe_is_explicitly_action_only() -> None:
    action = load_training_recipe(_ACTION_RECIPE)
    validate_action_only_recipe(action)

    structural = load_training_recipe(_RECIPE)
    with pytest.raises(ContractError, match="action-only objective"):
        validate_action_only_recipe(structural)


def test_core_move_keeps_all_picf_parameters_in_float32() -> None:
    recipe = load_training_recipe(_RECIPE)
    core = recipe.build_core()
    expected = {name: parameter.detach().clone() for name, parameter in core.named_parameters()}
    host_parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))

    moved = move_core_with_fp32_parameter_storage(core, host_parameter)

    for name, parameter in moved.named_parameters():
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter, expected[name])


def test_float32_picf_parameter_storage_resolves_sub_bfloat16_updates() -> None:
    recipe = load_training_recipe(_RECIPE)
    host_parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    core = move_core_with_fp32_parameter_storage(recipe.build_core(), host_parameter)
    bias = core.posterior_filter.transition.detectability_if_detected_head.bias
    before = bias.detach().clone()
    bias.grad = torch.ones_like(bias)
    optimizer = torch.optim.AdamW(
        [bias],
        lr=recipe.optimizer.picf_core_lr,
        weight_decay=0.0,
    )

    optimizer.step()

    assert bias.dtype == torch.float32
    assert not torch.equal(bias, before)
    torch.testing.assert_close(
        before - bias,
        torch.full_like(bias, recipe.optimizer.picf_core_lr),
        atol=1e-7,
        rtol=0.0,
    )


def test_official_optimizer_and_short_probe_scheduler_are_used(tmp_path: Path) -> None:
    recipe = load_training_recipe(_RECIPE)
    config = build_molmoact2_policy_config(recipe, checkpoint_path=tmp_path / "checkpoint")
    module = _OptimizerSurface(recipe.policy.optimizer_action_expert_lr)
    stack = MolmoAct2CalvinTrainingStack(
        policy_config=config,
        processor=None,  # type: ignore[arg-type]
        assets=None,  # type: ignore[arg-type]
        module=module,  # type: ignore[arg-type]
        accepted_temporal_core=None,  # type: ignore[arg-type]
    )

    optimizer, scheduler = build_molmoact2_optimizer_and_scheduler(recipe, stack)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert scheduler.optimizer is optimizer
    assert optimizer.param_groups[0]["initial_lr"] == recipe.policy.optimizer_action_expert_lr
    assert optimizer.param_groups[0]["lr"] < optimizer.param_groups[0]["initial_lr"]
